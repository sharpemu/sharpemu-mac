// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Diagnostics;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.ShaderCompiler.Metal;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Gpu.Metal.V2;

// A V#-described image table the guest indexes per lane (GTA V's ray-traced passes hold tens
// of thousands of entries, thousands of them distinct) is read through a device table of
// texture handles: one 16-byte entry per guest entry, a gpuResourceID and the class variant
// that reads it, and a null entry after the last. The shader indexes it directly, so the
// table's size never reaches the pipeline.
//
// Why handles in a device buffer and not an argument-buffer texture array: on Apple GPUs both
// hold the same 64-bit resource IDs, but an array's length is part of the shader, so every
// table size would be a new pipeline, and the shader could not read one entry as another
// texture type. Here the table is plain memory the shader reinterprets per variant.
//
// The textures some pass writes (render targets, storage images, anything the GPU modified)
// are declared to each encoder that reads them with one useResources:count: message per
// table: a residency set makes memory resident but takes no part in hazard tracking, and the
// reading pass has to be ordered after their writers (and any later writer after it). The
// rest only have to be resident, and a residency set on the queue does that once: declaring
// every texture of a 27,000-entry table to each encoder held each command buffer about 105 ms
// between its commit and the kernel taking it, with the GPU idle (Ghost of Tsushima's menu,
// 3 fps; 5.5 with the set). Without residency sets (before macOS 15) every texture is declared.
internal sealed partial class MetalRenderHost
{
    // SHARPEMU_TRACE_BINDLESS=<program hash>: each bindless table entry that program looks up - its variant and the
    // guest image its descriptor resolves to - so a table-selected read that comes back zero can be told apart.
    private static readonly ulong TraceBindlessHash =
        ulong.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_TRACE_BINDLESS")?.Replace("0x", string.Empty, StringComparison.OrdinalIgnoreCase),
            System.Globalization.NumberStyles.HexNumber, System.Globalization.CultureInfo.InvariantCulture, out var traced) ? traced : 0;

    private static readonly nint SelUseResourcesUsage = MetalNative.Selector("useResources:count:usage:");
    private static readonly nint SelUseResourcesStages = MetalNative.Selector("useResources:count:usage:stages:");
    private static readonly nint SelBindlessContents = MetalNative.Selector("contents");
    private static readonly nint SelBindlessNewBuffer = MetalNative.Selector("newBufferWithLength:options:");

    // Device tables kept for tables seen recently; the oldest idle ones go first. Ghost of Tsushima's material
    // programs read each texture of a material through a table of its own (19 a program), so a frame keeps hundreds.
    private const int MaxBindlessDeviceTables = 1024;

    private sealed class BindlessDeviceTable
    {
        public BindlessImageTable? Source;
        public nint Buffer;
        public nint Contents;
        public ulong GpuAddress;
        public ulong Capacity;
        public ulong LastUseSerial;
        public ulong[] WrittenIds = [];
        public ulong WrittenNullId;
        public bool Written;
        public nint[] Textures = [];
        public ulong[] ResourceIds = [];
        public long DeclaredEncoder = -1;
        // The textures declared to each encoder: those some pass writes, or all without a residency set.
        public readonly List<nint> Declared = [];
        public ulong DeclaredSignature;

        // Each descriptor's image and request as the last full resolve found them, reused
        // while the image cache has neither added nor dropped an image since (Invalid for
        // one a lookup must repeat on every bind).
        public ResourceSlotIdentifier[] Images = [];
        public ImageRequest[] Requests = [];
        public long ResolvedImageSet = -1;

        // Each descriptor's view as its last acquire stamped it: while the stamp holds, a bind takes the view
        // again without acquiring it (a material's textures, bound by every draw of its programs).
        public MetalImageCache.TextureMemoStamp?[] Stamps = [];

        // The image cache's binding epoch and image set the last full pass over the entries resolved at, -1 when
        // that pass left something a later bind must look at again; and the submission whose use it stamped.
        public long MemoBinding = -1;
        public long MemoImageSet = -1;
        public ulong MemoSerial;
    }

    // Each descriptor's request as its words and the program's image shape build it, which
    // neither the image set nor a bind changes, shared by every table: an entry whose lookup
    // repeats on every bind (a DCC surface) repeats only the lookup, and a table the title
    // rewrites (a new table object, with nothing resolved) finds its unchanged descriptors
    // here. Building them again was a third of the GPU worker in Ghost of Tsushima's world.
    private const int BindlessResolutionLimit = 1 << 18;

    private readonly Dictionary<TextureMemoKey, TextureRequestResolution> _bindlessResolutions = new();

    private TextureRequestResolution ResolveBindlessTexture(uint[] words, in ShaderImageShape shape)
    {
        if (words.Length < 8)
        {
            return ImageRequestBuilders.Texture(words, shape);
        }

        var key = new TextureMemoKey(words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7], shape);
        if (!_bindlessResolutions.TryGetValue(key, out var resolution))
        {
            resolution = ImageRequestBuilders.Texture(words, shape);
            if (_bindlessResolutions.Count >= BindlessResolutionLimit)
            {
                _bindlessResolutions.Clear();
            }

            _bindlessResolutions[key] = resolution;
        }

        return resolution;
    }

    private readonly Dictionary<BindlessImageTable, BindlessDeviceTable> _bindlessTables =
        new(ReferenceEqualityComparer.Instance);
    private readonly List<(nint Buffer, ulong Capacity, ulong RetireSerial)> _retiredBindlessBuffers = [];
    private readonly List<BindlessDeviceTable> _bindlessDeclarations = [];
    private long _encoderEpoch;

    private long _bindlessResolveTicks;
    private long _bindlessResolves;
    private long _bindlessDescriptors;
    private long _bindlessLookups;
    internal long BindlessDeclaredTextures;
    internal int BindlessResidentTextures => _bindlessResidency?.Count ?? -1;
    internal long BindlessNewTables, BindlessFullResolves, BindlessPartialResolves, BindlessInvalidEntries, BindlessWideChanges;

    /// <summary>
    /// The host time spent turning bindless tables into device tables, how many tables and
    /// descriptors that covered, and how many of those descriptors were looked up in the image
    /// cache rather than reused, since the last call.
    /// </summary>
    internal (TimeSpan Elapsed, long Tables, long Descriptors, long Lookups) TakeBindlessCost() =>
        (Stopwatch.GetElapsedTime(0, Interlocked.Exchange(ref _bindlessResolveTicks, 0)),
         Interlocked.Exchange(ref _bindlessResolves, 0),
         Interlocked.Exchange(ref _bindlessDescriptors, 0),
         Interlocked.Exchange(ref _bindlessLookups, 0));

    // Every table of the stage: each distinct descriptor's view, then the device table that
    // names them, rewritten only when the handles it holds change.
    private void BindBindlessImages(MetalStageBindings stage, ShaderResourceInfo info, ResourceSnapshot snapshot)
    {
        var tables = snapshot.BindlessImages;
        var layout = stage.Program.Bindings!;
        if (tables.Length != layout.BindlessImageTableCount)
        {
            throw Fatal(
                $"The bindless image tables do not match the program: tables={tables.Length} " +
                $"layout={layout.BindlessImageTableCount} hash=0x{stage.Program.Hash:X16}.");
        }

        if (stage.BindlessTableAddresses.Length < tables.Length)
        {
            stage.BindlessTableAddresses = new ulong[tables.Length];
        }

        var started = Stopwatch.GetTimestamp();
        var serial = SubmissionSerial;
        foreach (var table in tables)
        {
            if (table.Slot >= tables.Length || table.VariantImages.Length == 0 ||
                table.Descriptors.Length != table.DescriptorVariants.Length)
            {
                throw Fatal($"A bindless image table is malformed: slot={table.Slot} hash=0x{stage.Program.Hash:X16}.");
            }

            var device = DeviceTableOf(table);
            var count = table.Descriptors.Length;
            if (TraceBindlessHash == stage.Program.Hash)
            {
                Console.Error.WriteLine(
                    $"[METAL-V2][BINDLESS] hash=0x{stage.Program.Hash:X16} slot={table.Slot} entries={count} variants={table.VariantImages.Length}");
            }
            // Nothing a binding of a cached image resolves to has changed since the last full pass (the epoch
            // counts every such change, MetalImageEpochs) and the image set is the one it resolved against: every
            // entry's view, hazard and residency are as that pass left them. What a reacquire would still do is
            // stamp each image as used by this submission. Ghost of Tsushima binds tables of ~500 entries some
            // 1500 times a second, and the pass over them was a fifth of the GPU worker.
            var bindingEpoch = _imageCache.Epochs.Binding;
            var memoImageSet = _imageCache.ImageSetVersion;
            if (BindlessTableMemo && device.MemoBinding == bindingEpoch && device.MemoImageSet == memoImageSet &&
                !_imageCache.HasDeferredRefreshes && TraceBindlessHash != stage.Program.Hash)
            {
                BindlessTableMemoHits++;
                if (device.MemoSerial != serial)
                {
                    _imageCache.NoteMemoisedTextureUses(device.Stamps);
                    device.MemoSerial = serial;
                }

                BindlessDeclaredTextures += device.Declared.Count;
            }
            else
            {
                if (device.MemoBinding < 0) BindlessMemoMissUnset++;
                else if (_imageCache.HasDeferredRefreshes) BindlessMemoMissDeferred++;
                else if (device.MemoBinding != bindingEpoch) BindlessMemoMissBinding++;
                else BindlessMemoMissImageSet++;
                ResolveBindlessEntries(stage, info, table, device, count);
                var memoised = !_imageCache.HasDeferredRefreshes && _imageCache.Epochs.Binding == bindingEpoch &&
                    _imageCache.ImageSetVersion == memoImageSet;
                for (var index = 0; memoised && index < count; index++)
                {
                    memoised = device.Images[index].IsValid && device.Stamps[index] is not null && device.Textures[index] != 0;
                }

                device.MemoBinding = memoised ? bindingEpoch : -1;
                device.MemoImageSet = memoImageSet;
                device.MemoSerial = serial;
            }

            // A null or past-the-end entry reads the root's null texture, of variant 0's class.
            var root = (int)table.VariantImages[0];
            if (root >= stage.ImageCount || stage.TextureIds[root] == 0)
            {
                throw Fatal($"A bindless image table has no null texture: slot={table.Slot} hash=0x{stage.Program.Hash:X16}.");
            }

            TraceBindlessTable(stage, table, device);
            WriteDeviceTable(device, table, stage.TextureIds[root], serial);
            device.LastUseSerial = serial;
            stage.BindlessTableAddresses[table.Slot] = device.GpuAddress;
            NoteUsedResource(device.Buffer, Mtl.ResourceUsageRead);
            _bindlessDeclarations.Add(device);
            _bindlessResolves++;
            _bindlessDescriptors += count;
        }

        BindlessResidency?.PruneIfDue(_bindlessTables.Values, SubmissionSerial, CompletedSerial);
        _bindlessResolveTicks += Stopwatch.GetTimestamp() - started;
    }

    // One full pass over a table's entries: each descriptor's image (looked up again only when the image set
    // changed over it), its view, and whether a pass may write it (declared) or it only needs residency.
    private void ResolveBindlessEntries(
        MetalStageBindings stage, ShaderResourceInfo info, BindlessImageTable table, BindlessDeviceTable device, int count)
    {
        var residency = BindlessResidency;
        device.Declared.Clear();
        var declaredSignature = 0UL;
        // A lookup finds what it found last time unless an image came or went since: the
        // descriptors are fixed per table, and only the cache's image set can change. An
        // image added or dropped changes only the lookups over its guest bytes.
        var imageSet = _imageCache.ImageSetVersion;
        var reuse = device.ResolvedImageSet == imageSet;
        var changes = !reuse && _imageCache.CollectImageSetChanges(device.ResolvedImageSet, imageSet, _bindlessImageSetChanges)
            ? _bindlessImageSetChanges
            : null;
        if (!reuse)
        {
            if (changes is null) BindlessFullResolves++;
            else
            {
                BindlessPartialResolves++;
                if (changes.Count != 0 && changes[^1].End == ulong.MaxValue) BindlessWideChanges++;
            }
        }
        for (var index = 0; index < count; index++)
        {
            ResourceSlotIdentifier identifier;
            ImageRequest request;
            long pagesBefore;
            var lookedUp = false;
            if (device.Images[index].IsValid &&
                (reuse || (changes is not null && !OverlapsAny(changes, device.Requests[index].Description.Data))))
            {
                identifier = device.Images[index];
                request = device.Requests[index];
                pagesBefore = 0;
            }
            else
            {
                lookedUp = true;
                if (!device.Images[index].IsValid) BindlessInvalidEntries++;
                var variant = table.DescriptorVariants[index];
                if (variant >= table.VariantImages.Length || table.VariantImages[variant] >= (uint)info.Images.Count)
                {
                    throw Fatal(
                        $"A bindless descriptor names an unknown variant: slot={table.Slot} descriptor={index} " +
                        $"variant={variant} hash=0x{stage.Program.Hash:X16}.");
                }

                var image = info.Images[(int)table.VariantImages[variant]];
                var resolution = ResolveBindlessTexture(table.Descriptors[index], ShapeOf(image));
                request = resolution.Request;
                pagesBefore = _imageCache.RequestPageVersions(in request);
                identifier = _imageCache.FindImage(ref request, resolution.ExactFormat);
                _bindlessLookups++;
                resolution = resolution with { Request = request };
                identifier = ImageRequestBuilders.ValidateTextureOwner(
                    _imageCache.GetImage(identifier).Description, ResourceSlotIdentifier.Invalid, identifier, resolution);
                // A DCC surface's lookup materializes a clear the GPU may have left since.
                device.Images[index] = request.Description.Metadata.Kind == MetadataKind.Dcc
                    ? ResourceSlotIdentifier.Invalid
                    : identifier;
                device.Requests[index] = request;
                if (TraceBindlessHash == stage.Program.Hash)
                {
                    ref readonly var described = ref request.Description;
                    Console.Error.WriteLine(
                        $"[METAL-V2][BINDLESS] hash=0x{stage.Program.Hash:X16} slot={table.Slot} entry={index} variant={variant} " +
                        $"format={described.PixelFormat} guest={described.GuestFormat} tile={described.TileMode} " +
                        $"extent={described.Extent.Width}x{described.Extent.Height}x{described.Extent.Depth} " +
                        $"at=0x{described.Data.Address:X}+0x{described.Data.Size:X} T#=[{string.Join(",", table.Descriptors[index].Select(word => word.ToString("X8")))}]");
                }
            }

            var textureRequest = request with { Role = ImageRole.Texture };
            nint texture;
            if (!lookedUp && device.Textures[index] != 0 && device.Stamps[index] is { } stamp &&
                _imageCache.TryReacquireTextureView(identifier, in textureRequest, in stamp))
            {
                texture = device.Textures[index];
            }
            else
            {
                if (!lookedUp)
                    pagesBefore = _imageCache.RequestPageVersions(in request);
                texture = _imageCache.AcquireTextureViewForMemo(
                    identifier, textureRequest, pagesBefore, out device.ResourceIds[index], out device.Stamps[index]);
            }

            device.Textures[index] = texture;
            // A texture a pass can write (an attachment or a storage image, or one the GPU has written) is a hazard.
            var cached = residency is null ? null : _imageCache.GetImage(identifier);
            if (cached is null || cached.IsGpuModified ||
                (cached.Shape.Usage & (Mtl.TextureUsageRenderTarget | Mtl.TextureUsageShaderWrite)) != 0)
            {
                device.Declared.Add(texture);
                declaredSignature = (declaredSignature ^ (ulong)texture) * 0x100000001B3UL;
            }
            else
            {
                residency!.Add(texture);
            }
        }

        residency?.Commit();
        BindlessDeclaredTextures += device.Declared.Count;
        if (device.DeclaredSignature != declaredSignature)
        {
            // The encoder has not been told about these yet.
            device.DeclaredSignature = declaredSignature;
            device.DeclaredEncoder = -1;
        }

        // Images this pass added or dropped are changes after the version it started from.
        device.ResolvedImageSet = imageSet;
    }

    // SHARPEMU_METAL_V2_BINDLESS_TABLE_MEMO=0 walks every entry of a bindless table on every bind again.
    private static readonly bool BindlessTableMemo =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_BINDLESS_TABLE_MEMO") != "0";

    internal long BindlessTableMemoHits;

    // Why a bind missed the table memo: no memo (a new table, or the last pass left an entry to look at again), a
    // deferred refresh waiting, the binding epoch moved, or only the image set did.
    internal long BindlessMemoMissUnset, BindlessMemoMissDeferred, BindlessMemoMissBinding, BindlessMemoMissImageSet;

    // The residency set of bindless textures no pass writes, on the render queue; null when residency sets are unavailable.
    private BindlessTextureResidency? _bindlessResidency;
    private bool _bindlessResidencyTried;

    private BindlessTextureResidency? BindlessResidency
    {
        get
        {
            if (!_bindlessResidencyTried)
            {
                _bindlessResidencyTried = true;
                _bindlessResidency = BindlessDeclareAll ? null : BindlessTextureResidency.TryCreate(_device, _queue);
            }

            return _bindlessResidency;
        }
    }

    // SHARPEMU_METAL_V2_BINDLESS_DECLARE_ALL=1 declares every bindless texture to each encoder, as before residency sets.
    private static readonly bool BindlessDeclareAll =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_BINDLESS_DECLARE_ALL") == "1";

    private sealed class BindlessTextureResidency
    {
        private static readonly nint SelAddAllocation = MetalNative.Selector("addAllocation:");
        private static readonly nint SelRemoveAllocation = MetalNative.Selector("removeAllocation:");
        private static readonly nint SelCommit = MetalNative.Selector("commit");
        private static readonly nint SelRetainObject = MetalNative.Selector("retain");
        private static readonly nint SelReleaseObject = MetalNative.Selector("release");
        private static readonly long PruneIntervalTicks = Stopwatch.Frequency * 2;

        private readonly nint _set;
        // Each member is retained here until it leaves the set, whatever the set itself keeps.
        private readonly HashSet<nint> _members = [];
        // Members no table names any more, with the serial they were last possibly read by; they leave
        // the set once that submission has completed.
        private readonly List<(nint Texture, ulong Serial)> _leaving = [];
        private bool _dirty;
        private long _lastPrune = Stopwatch.GetTimestamp();

        private BindlessTextureResidency(nint set) => _set = set;

        public int Count => _members.Count;

        public static BindlessTextureResidency? TryCreate(nint device, nint queue)
        {
            var create = MetalNative.Selector("newResidencySetWithDescriptor:error:");
            var attach = MetalNative.Selector("addResidencySet:");
            var respondsTo = MetalNative.Selector("respondsToSelector:");
            if (device == 0 || queue == 0 || !MetalNative.SendBool(device, respondsTo, create) ||
                !MetalNative.SendBool(queue, respondsTo, attach))
            {
                return null;
            }

            var pool = MetalNative.objc_autoreleasePoolPush();
            try
            {
                var descriptor = MetalNative.Send(MetalNative.Class("MTLResidencySetDescriptor"), MetalNative.Selector("new"));
                MetalNative.SendVoid(descriptor, MetalNative.Selector("setLabel:"), MetalNative.NsString("SharpEmu bindless textures"));
                nint error = 0;
                var set = MetalNative.Send(device, create, descriptor, ref error);
                MetalNative.SendVoid(descriptor, SelReleaseObject);
                if (set == 0)
                {
                    Console.Error.WriteLine(
                        $"[METAL-V2][WARN] The bindless texture residency set could not be made ({MetalNative.DescribeError(error)}); every bindless texture is declared to its encoders.");
                    return null;
                }

                MetalNative.SendVoid(queue, attach, set);
                return new BindlessTextureResidency(set);
            }
            finally
            {
                MetalNative.objc_autoreleasePoolPop(pool);
            }
        }

        public void Add(nint texture)
        {
            if (texture == 0 || !_members.Add(texture))
            {
                return;
            }

            MetalNative.Send(texture, SelRetainObject);
            MetalNative.SendVoid(_set, SelAddAllocation, texture);
            _dirty = true;
        }

        public void Commit()
        {
            if (!_dirty)
            {
                return;
            }

            MetalNative.SendVoid(_set, SelCommit);
            _dirty = false;
        }

        // Every couple of seconds: members no device table names start leaving, and those whose last
        // possible reader has completed leave.
        public void PruneIfDue(IEnumerable<BindlessDeviceTable> tables, ulong serial, ulong completed)
        {
            var now = Stopwatch.GetTimestamp();
            if (now - _lastPrune < PruneIntervalTicks)
            {
                return;
            }

            _lastPrune = now;
            for (var index = _leaving.Count - 1; index >= 0; index--)
            {
                var (texture, lastSerial) = _leaving[index];
                if (lastSerial > completed)
                {
                    continue;
                }

                _leaving.RemoveAt(index);
                if (_members.Remove(texture))
                {
                    MetalNative.SendVoid(_set, SelRemoveAllocation, texture);
                    MetalNative.SendVoid(texture, SelReleaseObject);
                    _dirty = true;
                }
            }

            var named = new HashSet<nint>();
            foreach (var table in tables)
            {
                foreach (var texture in table.Textures)
                {
                    named.Add(texture);
                }
            }

            var alreadyLeaving = _leaving.Select(static leaving => leaving.Texture).ToHashSet();
            foreach (var member in _members)
            {
                if (!named.Contains(member) && !alreadyLeaving.Contains(member))
                {
                    _leaving.Add((member, serial));
                }
            }

            Commit();
        }

        // Shutdown only, once every submission has drained.
        public void Dispose()
        {
            foreach (var member in _members)
            {
                MetalNative.SendVoid(_set, SelRemoveAllocation, member);
                MetalNative.SendVoid(member, SelReleaseObject);
            }

            _members.Clear();
            _leaving.Clear();
            MetalNative.SendVoid(_set, SelCommit);
            MetalNative.SendVoid(_set, SelReleaseObject);
        }
    }

    private readonly List<(ulong Start, ulong End)> _bindlessImageSetChanges = [];

    // Whether the guest range meets one of the ranges, which are sorted by start and do not overlap.
    private static bool OverlapsAny(List<(ulong Start, ulong End)> ranges, GuestSpan range)
    {
        if (ranges.Count == 0)
        {
            return false;
        }

        if (range.Size == 0)
        {
            return true;
        }

        var end = range.Address + range.Size;
        int low = 0, high = ranges.Count;
        while (low < high)
        {
            var middle = (low + high) >>> 1;
            if (ranges[middle].Start < end) low = middle + 1;
            else high = middle;
        }

        return low != 0 && ranges[low - 1].End > range.Address;
    }

    private BindlessDeviceTable DeviceTableOf(BindlessImageTable table)
    {
        if (!_bindlessTables.TryGetValue(table, out var device))
        {
            ReleaseIdleBindlessTables();
            device = new BindlessDeviceTable { Source = table };
            BindlessNewTables++;
            _bindlessTables.Add(table, device);
        }

        var count = table.Descriptors.Length;
        if (device.Textures.Length != count)
        {
            device.Textures = new nint[count];
            device.ResourceIds = new ulong[count];
            device.Images = new ResourceSlotIdentifier[count];
            device.Requests = new ImageRequest[count];
            device.Stamps = new MetalImageCache.TextureMemoStamp?[count];
            device.ResolvedImageSet = -1;
        }

        return device;
    }

    // Rewrites the entries when a handle changed. A buffer the GPU may still read - by a
    // submitted command buffer or by work recorded into the open one - is retired and a
    // fresh one written, so no recorded dispatch sees another's entries.
    private void WriteDeviceTable(BindlessDeviceTable device, BindlessImageTable table, ulong nullId, ulong serial)
    {
        var unchanged = device.Written && device.WrittenNullId == nullId &&
            device.WrittenIds.AsSpan().SequenceEqual(device.ResourceIds);
        var bytes = ((ulong)table.Entries.Length + 1) * Gen5MslArgumentLayout.BindlessEntryBytes;
        if (unchanged && device.Capacity >= bytes)
        {
            return;
        }

        var completed = CompletedSerial;
        var busy = device.Buffer != 0 && (device.LastUseSerial == serial || device.LastUseSerial > completed);
        if (device.Buffer == 0 || busy || device.Capacity < bytes)
        {
            if (device.Buffer != 0)
            {
                _retiredBindlessBuffers.Add((device.Buffer, device.Capacity, device.LastUseSerial));
            }

            device.Buffer = TakeBindlessBuffer(bytes, completed, out device.Capacity);
            device.Contents = MetalNative.Send(device.Buffer, SelBindlessContents);
            device.GpuAddress = MetalNative.SendGpuAddress(device.Buffer, SelGpuAddress);
        }

        unsafe
        {
            var span = new Span<byte>((void*)device.Contents, checked((int)bytes));
            var entries = table.Entries;
            for (var entry = 0; entry < entries.Length; entry++)
            {
                var descriptor = entries[entry];
                var slot = span.Slice(entry * Gen5MslArgumentLayout.BindlessEntryBytes, Gen5MslArgumentLayout.BindlessEntryBytes);
                if (descriptor < 0)
                {
                    BinaryPrimitives.WriteUInt64LittleEndian(slot, nullId);
                    BinaryPrimitives.WriteUInt64LittleEndian(slot[8..], 0);
                    continue;
                }

                BinaryPrimitives.WriteUInt64LittleEndian(slot, device.ResourceIds[descriptor]);
                BinaryPrimitives.WriteUInt64LittleEndian(slot[8..], table.DescriptorVariants[descriptor]);
            }

            var sentinel = span.Slice(entries.Length * Gen5MslArgumentLayout.BindlessEntryBytes, Gen5MslArgumentLayout.BindlessEntryBytes);
            BinaryPrimitives.WriteUInt64LittleEndian(sentinel, nullId);
            BinaryPrimitives.WriteUInt64LittleEndian(sentinel[8..], 0);
        }

        if (device.WrittenIds.Length != device.ResourceIds.Length)
        {
            device.WrittenIds = new ulong[device.ResourceIds.Length];
        }

        device.ResourceIds.AsSpan().CopyTo(device.WrittenIds);
        device.WrittenNullId = nullId;
        device.Written = true;
    }

    private nint TakeBindlessBuffer(ulong bytes, ulong completed, out ulong capacity)
    {
        for (var index = 0; index < _retiredBindlessBuffers.Count; index++)
        {
            var retired = _retiredBindlessBuffers[index];
            if (retired.RetireSerial > completed || retired.Capacity < bytes)
            {
                continue;
            }

            _retiredBindlessBuffers.RemoveAt(index);
            capacity = retired.Capacity;
            return retired.Buffer;
        }

        // Rounded up so a table that grows a little reuses its buffer.
        capacity = Math.Max(4096UL, System.Numerics.BitOperations.RoundUpToPowerOf2(bytes));
        var buffer = MetalNative.SendNewBuffer(_device, SelBindlessNewBuffer, (nuint)capacity, Mtl.ResourceStorageModeShared);
        if (buffer == 0)
        {
            throw Fatal($"A bindless image table buffer could not be created: bytes=0x{capacity:X}.");
        }

        SetLabel(buffer, "SharpEmu bindless image table");
        return buffer;
    }

    // Drops device tables of snapshots no longer drawn once the GPU is done with them.
    private void ReleaseIdleBindlessTables()
    {
        var completed = CompletedSerial;
        if (_bindlessTables.Count >= MaxBindlessDeviceTables)
        {
            var oldest = _bindlessTables
                .Where(pair => pair.Value.LastUseSerial <= completed)
                .OrderBy(pair => pair.Value.LastUseSerial)
                .Take(_bindlessTables.Count - MaxBindlessDeviceTables + 1)
                .ToArray();
            foreach (var (table, device) in oldest)
            {
                _bindlessTables.Remove(table);
                if (device.Buffer != 0)
                {
                    _retiredBindlessBuffers.Add((device.Buffer, device.Capacity, device.LastUseSerial));
                }
            }
        }

        // Keep a few completed buffers for reuse and release the rest.
        var kept = 0;
        for (var index = _retiredBindlessBuffers.Count - 1; index >= 0; index--)
        {
            var retired = _retiredBindlessBuffers[index];
            if (retired.RetireSerial > completed || ++kept <= 4)
            {
                continue;
            }

            MetalNative.SendVoid(retired.Buffer, SelRelease);
            _retiredBindlessBuffers.RemoveAt(index);
        }
    }

    // Each table's textures, declared once per encoder in one message.
    private unsafe void DeclareBindlessImages(nint encoder, bool graphics, nuint stages)
    {
        foreach (var device in _bindlessDeclarations)
        {
            if (device.DeclaredEncoder == _encoderEpoch || device.Declared.Count == 0)
            {
                continue;
            }

            device.DeclaredEncoder = _encoderEpoch;
            var declared = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(device.Declared);
            if (declared.Length == 0)
            {
                continue;
            }

            fixed (nint* textures = declared)
            {
                if (graphics)
                {
                    MetalNative.SendUseResourcesStages(
                        encoder, SelUseResourcesStages, textures, (nuint)declared.Length, Mtl.ResourceUsageRead, stages);
                }
                else
                {
                    MetalNative.SendUseResources(
                        encoder, SelUseResourcesUsage, textures, (nuint)declared.Length, Mtl.ResourceUsageRead);
                }
            }
        }
    }

    // Shutdown only, once every submission has drained.
    private void DisposeBindlessImages()
    {
        foreach (var device in _bindlessTables.Values)
        {
            if (device.Buffer != 0)
            {
                MetalNative.SendVoid(device.Buffer, SelRelease);
            }
        }

        _bindlessTables.Clear();
        foreach (var retired in _retiredBindlessBuffers)
        {
            MetalNative.SendVoid(retired.Buffer, SelRelease);
        }

        _retiredBindlessBuffers.Clear();
        _bindlessResidency?.Dispose();
        _bindlessResidency = null;
    }

    // SHARPEMU_METAL_V2_TRACE_BINDLESS=<program hash>[,...] (diagnostic): each bindless table of those programs once, with
    // the descriptor and the resolved image of its first entries, to tell a wrong texture from a wrong lookup.
    private static readonly HashSet<ulong> TracedBindlessPrograms = new(
        (Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_TRACE_BINDLESS") ?? string.Empty)
            .Split(',', StringSplitOptions.RemoveEmptyEntries)
            .Select(text => ulong.TryParse(text.Replace("0x", string.Empty), System.Globalization.NumberStyles.HexNumber, null, out var hash) ? hash : 0));
    private readonly HashSet<(ulong, uint)> _tracedBindlessTables = new();

    private void TraceBindlessTable(MetalStageBindings stage, BindlessImageTable table, BindlessDeviceTable device)
    {
        if (TracedBindlessPrograms.Count == 0 || !TracedBindlessPrograms.Contains(stage.Program.Hash) ||
            !_tracedBindlessTables.Add((stage.Program.Hash, table.Slot)))
            return;
        var text = new System.Text.StringBuilder(
            $"[METAL-V2][BINDLESS-TRACE] hash=0x{stage.Program.Hash:X16} slot={table.Slot} address=0x{table.Address:X} " +
            $"entries={table.Entries.Length} descriptors={table.Descriptors.Length} variants={table.VariantImages.Length}");
        for (var entry = 0; entry < Math.Min(table.Entries.Length, 6); entry++)
        {
            var descriptor = table.Entries[entry];
            text.Append($"\n    entry {entry} -> descriptor {descriptor}");
            if (descriptor < 0)
                continue;
            var words = table.Descriptors[descriptor];
            text.Append($" words={string.Join(" ", words.Select(word => word.ToString("X8")))}");
            ref readonly var resolved = ref device.Requests[descriptor].Description;
            text.Append($" -> {resolved.PixelFormat}/{resolved.TileMode} {resolved.Extent.Width}x{resolved.Extent.Height} " +
                $"levels={resolved.Resources.Levels} @0x{resolved.Data.Address:X}+0x{resolved.Data.Size:X}");
        }

        // Every entry in short: the T# type, extent and format, or null.
        var all = new System.Text.StringBuilder();
        for (var entry = 0; entry < table.Entries.Length; entry++)
        {
            var descriptor = table.Entries[entry];
            if (descriptor < 0)
            {
                all.Append($" {entry}:null");
                continue;
            }

            var words = table.Descriptors[descriptor];
            all.Append($" {entry}:t{(words[3] >> 28) & 0xF}/f{(words[1] >> 20) & 0x1FF}");
        }

        text.Append($"\n    all{all}");
        Console.Error.WriteLine(text.ToString());
    }
}
