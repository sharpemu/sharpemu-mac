// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Numerics;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.VideoOut;
using SharpEmu.ShaderCompiler.Metal;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Gpu.Metal.V2;

// The bindless heap (see BindlessHeap): a program that reads an image or sampler from a heap V# at a record only its
// shader knows reads it through an element table, one 8-byte handle (a gpuResourceID) per element and heap array,
// slot-major (slot * Capacity + element), the arrays Vulkan keeps in descriptor set 1. Every element starts null: a zero
// image handle, which the shader reads as zeros without touching a texture, or the null sampler. Each heap view of a draw
// (a heap V# and the offset of a descriptor in its records) holds a run of elements (BindlessHeapViewTable), which the
// draw's shader data names. The shader marks each element it reads in a feedback bitset; once a submission completes,
// the host reads the record's T# or S# from guest memory and writes its texture or sampler. The records of a view are read
// again periodically while draws read the view, and the textures of a view are made current, and noted as used by the
// submission, before the first draw of each submission that reads it (and again when an image changed since). A record
// that holds an integer texture gets an integer view, and the feedback buffer tells the shader the element's numeric
// class. The port of VulkanVideoPresenter.BindlessHeap.cs.
//
// Both buffers are shared storage written by the CPU while command buffers may read them, as Vulkan's update-after-bind
// set is: a command in flight that reads an element being resolved reads the old (null) or the new handle. The textures
// no pass writes go into the bindless residency set; those a pass may write are declared to every encoder of a draw that
// reads the heap, as the bindless image tables' are (MetalRenderHost.BindlessImages.cs).
internal sealed partial class MetalRenderHost
{
    // SHARPEMU_METAL_BINDLESS_HEAP=0 leaves the heap off: programs are planned as before (bindless image tables, the
    // 256-candidate indirect limit, off-class candidates read null).
    internal static readonly bool BindlessHeapEnabled =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_BINDLESS_HEAP") != "0";

    public bool SupportsBindlessHeap => BindlessHeapEnabled;

    // Records are read again after this many submissions, so a rewritten record gets its new texture.
    private const ulong HeapRevalidationSerials = 16;

    // A heavy frame can take hundreds of submissions. Views stay resident across frames so a once-per-frame pass can read
    // them before the next completed submission resolves any missing entries.
    private const ulong HeapIdleSerials = 256;
    private const long HeapIdleFrames = 2;

    private static ulong HeapElementTableBytes =>
        (ulong)BindlessHeap.ArrayCount * BindlessHeap.Capacity * Gen5MslArgumentLayout.HeapElementBytes;

    private static ulong HeapFeedbackBytes => (ulong)BindlessHeap.FeedbackWords * sizeof(uint);

    // One record the shaders marked: the words it was resolved from and what the element table holds for it.
    private sealed class HeapEntry
    {
        public int Slot;
        public uint Index;
        public BindlessHeapViewTable.Region? Region;
        public uint[] Words = [];
        public bool Resolved;
        public ImageNumericClass NumericClass;
        public ResourceSlotIdentifier Image;
        public ImageRequest Request;
        public MetalImageCache.TextureMemoStamp? Stamp;
        public nint Texture;
        public ulong ResourceId;
        // A texture some pass may write: declared to the encoders instead of made resident.
        public bool Declared;
        // A DCC surface's lookup materializes a clear the GPU may have left since: looked up on every pass.
        public bool LookupEachPass;
    }

    // The entries of each view in record order, and when they were last compared with the guest's records and made
    // current.
    private sealed class HeapRegionEntries
    {
        public readonly List<HeapEntry> Order = [];
        public ulong ValidatedSerial;
        public ulong CurrentSerial = ulong.MaxValue;
        public long CurrentBinding = -1;
        public long CurrentImageSet = -1;
    }

    private nint _heapElements;
    private nint _heapElementsContents;
    private ulong _heapElementsAddress;
    private nint _heapFeedback;
    private nint _heapFeedbackContents;
    private ulong _heapFeedbackAddress;
    private ulong _heapNullSamplerId;
    private uint[] _heapMarked = [];
    private uint[] _heapClasses = [];
    private readonly Dictionary<(int Slot, uint Index), HeapEntry> _heapEntries = new();
    // The image arrays share their elements (a record has one element in each), and so do the two sampler arrays.
    private readonly BindlessHeapViewTable _heapImageViews = new(BindlessHeap.CallRecordFirstIndex);
    private readonly BindlessHeapViewTable _heapSamplerViews = new(BindlessHeap.CallRecordFirstIndex);
    private readonly Dictionary<BindlessHeapViewTable.Region, HeapRegionEntries> _heapRegionEntries =
        new(ReferenceEqualityComparer.Instance);
    private Action<BindlessHeapViewTable.Region>? _heapRegionReleased;
    private ulong _heapPolledSerial = ulong.MaxValue;
    private ulong _heapRevalidatedSerial;
    private bool _heapPrepared;
    private readonly List<HeapEntry> _heapStaleEntries = [];
    // The heap's textures as one bindless device table: Textures names every one for the residency prune, Declared
    // those declared to the encoders.
    private readonly BindlessDeviceTable _heapTextures = new();
    private bool _heapTexturesChanged;
    private int _heapReports;

    internal long HeapResolvedEntries, HeapReleasedRegions;

    private bool HeapCreated => _heapElements != 0;

    // The element table, its null elements and the feedback bitsets, made at the first draw that reads the heap.
    private unsafe void EnsureBindlessHeap()
    {
        if (HeapCreated)
        {
            return;
        }

        const nuint options = Mtl.ResourceStorageModeShared | Mtl.ResourceHazardTrackingModeUntracked;
        var elements = MetalNative.SendNewBuffer(_device, SelBindlessNewBuffer, (nuint)HeapElementTableBytes, options);
        var feedback = MetalNative.SendNewBuffer(_device, SelBindlessNewBuffer, (nuint)HeapFeedbackBytes, options);
        if (elements == 0 || feedback == 0)
        {
            throw Fatal($"The bindless heap buffers could not be created: elements=0x{HeapElementTableBytes:X} feedback=0x{HeapFeedbackBytes:X}.");
        }

        SetLabel(elements, "SharpEmu bindless heap elements");
        SetLabel(feedback, "SharpEmu bindless heap feedback");
        _heapElements = elements;
        _heapElementsContents = MetalNative.Send(elements, SelBindlessContents);
        _heapElementsAddress = MetalNative.SendGpuAddress(elements, SelGpuAddress);
        _heapFeedback = feedback;
        _heapFeedbackContents = MetalNative.Send(feedback, SelBindlessContents);
        _heapFeedbackAddress = MetalNative.SendGpuAddress(feedback, SelGpuAddress);
        new Span<byte>((void*)_heapElementsContents, checked((int)HeapElementTableBytes)).Clear();
        new Span<byte>((void*)_heapFeedbackContents, checked((int)HeapFeedbackBytes)).Clear();
        _heapMarked = new uint[BindlessHeap.MarkWords];
        _heapClasses = new uint[BindlessHeap.ClassWords];
        _heapRegionReleased = ReleaseHeapRegion;

        // Every sampler element is the null sampler (an all-zero S#) until its record is read.
        Span<uint> nullSampler = stackalloc uint[4];
        _ = GetSampler(new SamplerDescriptorWords(nullSampler), integerView: false, out _heapNullSamplerId);
        for (var slot = BindlessHeap.SamplerSlot; slot < BindlessHeap.ArrayCount; slot++)
        {
            ElementSpan(slot).Fill(_heapNullSamplerId);
        }

        Console.Error.WriteLine(
            $"[METAL-V2][INFO] Bindless heap: image_arrays={BindlessHeap.ImageKinds.Length} sampler_arrays=2 " +
            $"capacity={BindlessHeap.Capacity} elements=0x{HeapElementTableBytes:X}");
    }

    private unsafe Span<ulong> ElementSpan(int slot) =>
        new((void*)(_heapElementsContents + (nint)((ulong)slot * BindlessHeap.Capacity * sizeof(ulong))), (int)BindlessHeap.Capacity);

    private unsafe Span<uint> FeedbackWords =>
        new((void*)_heapFeedbackContents, (int)BindlessHeap.FeedbackWords);

    // Shutdown only, once every submission has drained; the image cache owns the textures.
    private void DisposeBindlessHeap()
    {
        _heapEntries.Clear();
        _callRecordEntries.Clear();
        _callRecordTextures.Order.Clear();
        _heapRegionEntries.Clear();
        _heapImageViews.ReleaseIdle(ulong.MaxValue, ulong.MaxValue, static _ => { });
        _heapSamplerViews.ReleaseIdle(ulong.MaxValue, ulong.MaxValue, static _ => { });
        if (_heapElements != 0)
        {
            MetalNative.SendVoid(_heapElements, SelRelease);
            MetalNative.SendVoid(_heapFeedback, SelRelease);
        }

        _heapElements = _heapFeedback = 0;
        _heapElementsContents = _heapFeedbackContents = 0;
    }

    // Once per preparation, before the images of the first stage that reads the heap are found: the records marked by
    // submissions completed since the last look, and the views idle long enough to give up.
    private void PrepareBindlessHeap()
    {
        if (_heapPrepared)
        {
            return;
        }

        _heapPrepared = true;
        EnsureBindlessHeap();
        var completed = CompletedSerial;
        if (completed != _heapPolledSerial)
        {
            _heapPolledSerial = completed;
            ResolveMarkedRecords();
        }

        var serial = SubmissionSerial;
        if (serial - _heapRevalidatedSerial >= HeapRevalidationSerials)
        {
            ReleaseIdleHeapViews(serial);
        }
    }

    // The runs of elements of a stage's heap views, as the stage's shader data (HeapView, two dwords each), each view's
    // records compared with the guest's when due, and its textures current for this submission.
    private void WriteHeapViews(ReadOnlySpan<BindlessHeapDescriptor> heaps, Span<uint> words)
    {
        var serial = SubmissionSerial;
        var completed = CompletedSerial;
        var frame = Interlocked.Read(ref _flips);
        for (var index = 0; index < heaps.Length; index++)
        {
            ref readonly var heap = ref heaps[index];
            var region = (heap.Sampler ? _heapSamplerViews : _heapImageViews).Acquire(
                heap, serial, completed, _heapRegionReleased!, out var moved, out var grownFrom, frame);
            if (region is not null && grownFrom is not null)
            {
                CopyHeapEntries(grownFrom, region);
            }

            var view = region?.View ?? HeapView.None;
            words[index * (int)HeapView.DwordCount] = view.Word0;
            words[index * (int)HeapView.DwordCount + 1] = view.Word1;
            if (region is null)
            {
                if (heap.Base != 0 && heap.Records != 0 && _heapReports++ < 16)
                {
                    Console.Error.WriteLine(
                        $"[METAL-V2][WARN] Bindless heap view has no elements left: heap=0x{heap.Base:X16} records={heap.Records} " +
                        $"stride={heap.Stride}; its records read as null.");
                }

                continue;
            }

            if (!_heapRegionEntries.TryGetValue(region, out var entries))
            {
                continue;
            }

            // Only a view some draw reads is compared with the guest's records: an idle view's memory may hold anything,
            // and is read again when a draw comes back to it. A run that moved to this heap is compared now: the records
            // that are the same textures at the new address keep their elements.
            if (moved || serial - entries.ValidatedSerial >= HeapRevalidationSerials)
            {
                RevalidateHeapRegion(entries, serial);
            }

            if (!heap.Sampler)
            {
                MakeHeapRegionCurrent(entries, serial);
            }
        }
    }

    // A heap that grew past its run takes a larger one: the records its shaders read through the old run get their
    // elements in the new run now, so a draw reads them at once instead of null until the next marks come back.
    private void CopyHeapEntries(BindlessHeapViewTable.Region from, BindlessHeapViewTable.Region to)
    {
        if (!_heapRegionEntries.TryGetValue(from, out var entries))
        {
            return;
        }

        foreach (var entry in entries.Order.ToArray())
        {
            var index = to.View.First + (entry.Index - from.View.First);
            if (!to.Holds(index) || _heapEntries.ContainsKey((entry.Slot, index)))
            {
                continue;
            }

            _heapMarked[entry.Slot * (int)BindlessHeap.FeedbackWordsPerArray + (int)(index >> 5)] |= 1u << (int)(index & 31);
            AddHeapEntry(entry.Slot, index, to);
        }
    }

    private void AddHeapEntry(int slot, uint index, BindlessHeapViewTable.Region region)
    {
        var entry = new HeapEntry { Slot = slot, Index = index, Region = region };
        _heapEntries[(slot, index)] = entry;
        if (!_heapRegionEntries.TryGetValue(region, out var entries))
        {
            entries = new HeapRegionEntries { ValidatedSerial = SubmissionSerial };
            _heapRegionEntries.Add(region, entries);
        }

        var position = entries.Order.Count;
        while (position > 0 && (entries.Order[position - 1].Index > index ||
                   (entries.Order[position - 1].Index == index && entries.Order[position - 1].Slot > slot)))
        {
            position--;
        }

        entries.Order.Insert(position, entry);
        Span<uint> words = stackalloc uint[8];
        entry.Words = words[..ReadHeapRecord(entry, words)].ToArray();
        ResolveHeapEntry(entry);
    }

    // A view gives its elements up: they read null again, and a later view that takes them starts unmarked.
    private void ReleaseHeapRegion(BindlessHeapViewTable.Region region)
    {
        HeapReleasedRegions++;
        var sampler = region.Heap.Sampler;
        if (_heapRegionEntries.Remove(region, out var entries))
        {
            foreach (var entry in entries.Order)
            {
                _heapEntries.Remove((entry.Slot, entry.Index));
                _heapMarked[entry.Slot * (int)BindlessHeap.FeedbackWordsPerArray + (int)(entry.Index >> 5)] &= ~(1u << (int)(entry.Index & 31));
                DetachHeapEntry(entry);
                if (sampler)
                {
                    ElementSpan(entry.Slot)[(int)entry.Index] = _heapNullSamplerId;
                }
                else
                {
                    WriteHeapImage(entry.Slot, entry.Index, 0);
                }
            }
        }

        // The marks of the words the run touches: a neighbour's shader marks its element again when it reads it.
        var firstSlot = sampler ? BindlessHeap.SamplerSlot : 0;
        var endSlot = sampler ? BindlessHeap.ArrayCount : BindlessHeap.ImageKinds.Length;
        var firstWord = (int)(region.View.First >> 5);
        var endWord = (int)((region.View.First + region.Extent + 31) >> 5);
        var feedback = FeedbackWords;
        for (var slot = firstSlot; slot < endSlot; slot++)
        {
            var start = slot * (int)BindlessHeap.FeedbackWordsPerArray;
            for (var word = start + firstWord; word < start + endWord; word++)
            {
                Volatile.Write(ref feedback[word], 0);
            }
        }
    }

    // The marks of the submissions completed since the last look: each fresh mark of an element a view holds becomes an
    // entry, its record read and resolved.
    private void ResolveMarkedRecords()
    {
        var marked = FeedbackWords[..(int)BindlessHeap.MarkWords];
        for (var word = 0; word < marked.Length; word++)
        {
            var fresh = Volatile.Read(ref marked[word]) & ~_heapMarked[word];
            if (fresh == 0)
            {
                continue;
            }

            _heapMarked[word] |= fresh;
            var slot = word / (int)BindlessHeap.FeedbackWordsPerArray;
            var first = (uint)(word % (int)BindlessHeap.FeedbackWordsPerArray) * 32;
            while (fresh != 0)
            {
                var index = first + (uint)BitOperations.TrailingZeroCount(fresh);
                fresh &= fresh - 1;
                if (index >= BindlessHeap.CallRecordFirstIndex)
                {
                    // Call records own these elements; the shader reads a heap record past the heap's range as null.
                    continue;
                }

                // An element no view holds was marked before its view gave it up.
                var sampler = slot >= BindlessHeap.ImageKinds.Length;
                if ((sampler ? _heapSamplerViews : _heapImageViews).Find(index) is not { } region)
                {
                    _heapMarked[word] &= ~(1u << (int)(index & 31));
                    continue;
                }

                AddHeapEntry(slot, index, region);
            }
        }
    }

    private void ReleaseIdleHeapViews(ulong serial)
    {
        _heapRevalidatedSerial = serial;
        var frame = Interlocked.Read(ref _flips);
        if (serial > HeapIdleSerials && frame > HeapIdleFrames)
        {
            var completed = CompletedSerial;
            _heapImageViews.ReleaseIdle(serial - HeapIdleSerials, completed, _heapRegionReleased!, frame - HeapIdleFrames);
            _heapSamplerViews.ReleaseIdle(serial - HeapIdleSerials, completed, _heapRegionReleased!, frame - HeapIdleFrames);
        }
    }

    // The entries of one view, in record order, compared with their records now: a changed record is resolved again.
    private void RevalidateHeapRegion(HeapRegionEntries entries, ulong serial)
    {
        entries.ValidatedSerial = serial;
        Span<uint> words = stackalloc uint[8];
        foreach (var entry in entries.Order.ToArray())
        {
            if (entry.Region is null || !_heapEntries.TryGetValue((entry.Slot, entry.Index), out var current) ||
                !ReferenceEquals(current, entry))
            {
                // Released by a resolve of an earlier entry (the view lost its run).
                continue;
            }

            var read = words[..ReadHeapRecord(entry, words)];
            // A T# that had no texture when it was last looked at (its memory was not mapped) is looked at again.
            if (read.SequenceEqual(entry.Words) &&
                (entry.Resolved || entry.Slot >= BindlessHeap.ImageKinds.Length ||
                 HeapRecordClass(entry, read) == ImageNumericClass.Unsupported))
            {
                continue;
            }

            entry.Words = read.ToArray();
            ResolveHeapEntry(entry);
        }
    }

    // Each texture of a view as an acquire now would give it, and noted as used by this submission (an image the cache
    // drops is retired after the last submission that noted it): once per submission, and again when an image changed
    // since the last pass.
    private void MakeHeapRegionCurrent(HeapRegionEntries entries, ulong serial)
    {
        var binding = _imageCache.Epochs.Binding;
        var imageSet = _imageCache.ImageSetVersion;
        if (entries.CurrentSerial == serial && entries.CurrentBinding == binding && entries.CurrentImageSet == imageSet &&
            !_imageCache.HasDeferredRefreshes)
        {
            return;
        }

        _heapStaleEntries.Clear();
        using (_imageCache.HoldForReacquire())
        {
            foreach (var entry in entries.Order)
            {
                if (!entry.Resolved)
                {
                    continue;
                }

                if (entry.LookupEachPass || entry.Stamp is not { } stamp ||
                    !_imageCache.TryReacquireTextureViewHeld(entry.Image, entry.Request, in stamp))
                {
                    _heapStaleEntries.Add(entry);
                }
            }
        }

        foreach (var entry in _heapStaleEntries)
        {
            ResolveHeapEntry(entry);
        }

        _heapStaleEntries.Clear();
        // What a pass may write moves the binding epoch: the entries are classified again.
        if (entries.CurrentBinding != binding)
        {
            foreach (var entry in entries.Order)
            {
                if (entry.Resolved && entry.Declared != HeapTextureDeclared(entry.Image))
                {
                    entry.Declared = !entry.Declared;
                    _heapTexturesChanged = true;
                }
            }
        }

        entries.CurrentSerial = serial;
        entries.CurrentBinding = binding;
        entries.CurrentImageSet = imageSet;
    }

    private bool HeapTextureDeclared(ResourceSlotIdentifier identifier)
    {
        if (BindlessResidency is null)
        {
            return true;
        }

        var cached = _imageCache.GetImage(identifier);
        return cached.IsGpuModified || (cached.Shape.Usage & (Mtl.TextureUsageRenderTarget | Mtl.TextureUsageShaderWrite)) != 0;
    }

    // The words of the entry's record (8 for an image, an R128 T# padded with zeros; 4 for a sampler), or none when its
    // view has no such record or its memory cannot be read.
    private int ReadHeapRecord(HeapEntry entry, Span<uint> words)
    {
        if (entry.Region is not { } region || !region.Holds(entry.Index) || entry.Index - region.View.First >= region.View.Records)
        {
            return 0;
        }

        var count = checked((int)(region.Heap.DescriptorBytes / sizeof(uint)));
        var address = region.View.DescriptorAddress(region.Heap, entry.Index - region.View.First);
        var record = words[..count];
        if (!TryReadResidentGuestBytes(address, System.Runtime.InteropServices.MemoryMarshal.AsBytes(record), clean: true))
        {
            // Bytes a recorded GPU write may still own: read word by word, which waits for the write.
            for (var word = 0; word < count; word++)
            {
                if (!TryReadGuestWord(address + (ulong)word * sizeof(uint), out record[word]))
                {
                    return 0;
                }
            }
        }

        if (region.Heap is { Sampler: false, R128: true })
        {
            // An R128 T# is its four words; the rest of the T# is zero.
            words[count..8].Clear();
            count = 8;
        }

        return count;
    }

    private void ResolveHeapEntry(HeapEntry entry)
    {
        HeapResolvedEntries++;
        DetachHeapEntry(entry);
        if (entry.Slot >= BindlessHeap.ImageKinds.Length)
        {
            WriteHeapSampler(entry);
            return;
        }

        var numericClass = HeapRecordClass(entry, entry.Words);
        var resourceId = numericClass != ImageNumericClass.Unsupported ? AcquireHeapImage(entry, numericClass) : 0;
        WriteHeapImage(entry.Slot, entry.Index, resourceId);
    }

    // The numeric class of the texture an array of this slot holds for the record, or Unsupported when it holds none:
    // every class holds float textures, and the ones with integer forms hold integer textures too.
    private static ImageNumericClass HeapRecordClass(HeapEntry entry, ReadOnlySpan<uint> words)
    {
        var numericClass = words.Length == 8
            ? ResourceMaterializer.HeapImageNumericClass(words, entry.Region is { Heap.R128: true })
            : ImageNumericClass.Unsupported;
        return numericClass == ImageNumericClass.Float || BindlessHeap.HoldsIntegers(entry.Slot) ? numericClass : ImageNumericClass.Unsupported;
    }

    private static ShaderImageShape HeapImageShape(int slot, ImageNumericClass numericClass, bool r128)
    {
        var heapClass = BindlessHeap.ImageClasses[slot];
        var dimension = heapClass.Dimension;
        return new ShaderImageShape(
            Volume: dimension == ImageDimension.Dim3D,
            Arrayed: heapClass.Cube || dimension is ImageDimension.Dim1DArray or ImageDimension.Dim2DArray,
            Cube: heapClass.Cube,
            Storage: false,
            DynamicMip: false,
            NumericClass: numericClass switch
            {
                ImageNumericClass.Uint => TextureNumericClass.Uint,
                ImageNumericClass.Sint => TextureNumericClass.Sint,
                _ => TextureNumericClass.Float,
            },
            OneDimensional: dimension is ImageDimension.Dim1D or ImageDimension.Dim1DArray,
            R128: r128,
            DepthCompare: heapClass.Compare);
    }

    private static bool IsDepthViewFormat(GpuFormat format) =>
        format is GpuFormat.D16Unorm or GpuFormat.D32Sfloat or GpuFormat.D24UnormS8Uint or GpuFormat.D32SfloatS8Uint or GpuFormat.D16UnormS8Uint;

    // Finds the record's image the way a bindless table entry finds its texture (ResolveBindlessEntry); the texture's
    // gpuResourceID, or 0 when the record names no texture this array can hold.
    private ulong AcquireHeapImage(HeapEntry entry, ImageNumericClass numericClass)
    {
        TextureRequestResolution resolution;
        try
        {
            using (ImageRequestBuilders.RejectInvalidDescriptors())
            {
                resolution = ImageRequestBuilders.Texture(entry.Words,
                    HeapImageShape(entry.Slot, numericClass, entry.Region is { Heap.R128: true }));
            }
        }
        catch (TextureDescriptorRejectedException)
        {
            return 0;
        }

        var request = resolution.Request;
        // The record of a table may hold no T# the guest uses (a record it has not filled, one whose memory it uses for
        // other data by now, or the T# of a texture it has unloaded): words that name no mapped guest memory are no image
        // for now.
        var data = request.Description.Data;
        if (ImageDescription.IsEmptyRange(data) || MappedDeviceAddressBytes(data.Address, data.Size) < data.Size)
        {
            return 0;
        }

        var pagesBefore = _imageCache.RequestPageVersions(in request);
        var identifier = _imageCache.FindImage(ref request, resolution.ExactFormat);
        resolution = resolution with { Request = request };
        identifier = ImageRequestBuilders.ValidateTextureOwner(
            _imageCache.GetImage(identifier).Description, ResourceSlotIdentifier.Invalid, identifier, resolution);
        var image = _imageCache.GetImage(identifier);
        var view = request.View;
        // Depth-reference sampling needs a depth view; the view has the numeric class the shader reads the element by.
        if (!image.SupportsViewType(in view) ||
            ViewFormatRules.IsIntegerFormat(view.Format) != (numericClass != ImageNumericClass.Float) ||
            (BindlessHeap.ImageClasses[entry.Slot].Compare && !IsDepthViewFormat(view.Format)))
        {
            return 0;
        }

        var textureRequest = request with { Role = ImageRole.Texture };
        entry.Texture = _imageCache.AcquireTextureViewForMemo(
            identifier, textureRequest, pagesBefore, out entry.ResourceId, out entry.Stamp);
        if (entry.Texture == 0 || entry.ResourceId == 0)
        {
            entry.Texture = 0;
            entry.ResourceId = 0;
            entry.Stamp = null;
            return 0;
        }

        entry.Image = identifier;
        entry.Request = textureRequest;
        entry.Resolved = true;
        entry.NumericClass = numericClass;
        entry.Declared = HeapTextureDeclared(identifier);
        entry.LookupEachPass = request.Description.Metadata.Kind == MetadataKind.Dcc;
        _heapTexturesChanged = true;
        return entry.ResourceId;
    }

    private void DetachHeapEntry(HeapEntry entry)
    {
        if (entry.Texture != 0)
        {
            _heapTexturesChanged = true;
        }

        entry.Image = ResourceSlotIdentifier.Invalid;
        entry.Stamp = null;
        entry.Texture = 0;
        entry.ResourceId = 0;
        entry.Resolved = false;
    }

    // The element's handle, with its numeric class for the shaders: a handle is cleared before its class changes and
    // written after, so a shader never reads a texture through the declaration of another class.
    private void WriteHeapImage(int slot, uint index, ulong resourceId)
    {
        var elements = ElementSpan(slot);
        Volatile.Write(ref elements[(int)index], 0UL);
        WriteHeapElementClass(index);
        if (resourceId != 0)
        {
            Interlocked.MemoryBarrier();
            Volatile.Write(ref elements[(int)index], resourceId);
        }
    }

    // The numeric class of the element at index: that of the integer view an array holds there (the arrays hold views of
    // one record at an index), or float.
    private void WriteHeapElementClass(uint index)
    {
        var code = 0u;
        for (var slot = 0; slot < BindlessHeap.ImageKinds.Length && code == 0; slot++)
        {
            if (_heapEntries.TryGetValue((slot, index), out var entry) && entry.Resolved)
            {
                code = BindlessHeap.ClassCode(entry.NumericClass);
            }
        }

        var word = (int)(index >> (int)BindlessHeap.ClassWordShift);
        var shift = (int)((index & (BindlessHeap.ClassesPerWord - 1)) * BindlessHeap.ClassBits);
        var classes = (_heapClasses[word] & ~(BindlessHeap.ClassMask << shift)) | (code << shift);
        if (classes == _heapClasses[word])
        {
            return;
        }

        _heapClasses[word] = classes;
        Volatile.Write(ref FeedbackWords[(int)BindlessHeap.MarkWords + word], classes);
    }

    private void WriteHeapSampler(HeapEntry entry)
    {
        var resourceId = _heapNullSamplerId;
        if (entry.Words.Length == 4 && !(entry.Words[0] == 0xFEFEFEFE && entry.Words[1] == 0xFEFEFEFE))
        {
            Span<uint> native = stackalloc uint[4] { entry.Words[0], entry.Words[1], entry.Words[2], entry.Words[3] };
            if (entry.Slot != BindlessHeap.CompareSamplerSlot)
            {
                native[0] &= ~(0x7u << 12);
            }

            if (GetSampler(new SamplerDescriptorWords(native), integerView: false, out var found) != 0 && found != 0)
            {
                resourceId = found;
                entry.Resolved = true;
            }
        }

        Volatile.Write(ref ElementSpan(entry.Slot)[(int)entry.Index], resourceId);
    }

    // The stage's heap views, once its program's layout reads the heap: prepared before its images (PrepareBindings).
    private void PrepareHeapViews(MetalStageBindings stage, BindingLayout layout, ResourceSnapshot snapshot)
    {
        if (!BindlessHeapEnabled)
        {
            throw Fatal($"A program reads bindless heap records with the heap off (SHARPEMU_METAL_BINDLESS_HEAP=0): hash=0x{stage.Program.Hash:X16}.");
        }

        if (snapshot.Heaps.Length != layout.HeapViewCount)
        {
            throw Fatal(
                $"The heap views do not match the layout: views={snapshot.Heaps.Length} layout={layout.HeapViewCount} " +
                $"hash=0x{stage.Program.Hash:X16}.");
        }

        PrepareBindlessHeap();
        var count = checked((int)(layout.HeapViewCount * HeapView.DwordCount));
        if (stage.HeapViews.Length < count)
        {
            stage.HeapViews = new uint[count];
        }

        WriteHeapViews(snapshot.Heaps, stage.HeapViews.AsSpan(0, count));
    }

    // The heap buffers named to the encoders, and its textures resident or declared, for a stage that reads the heap.
    private void BindBindlessHeap()
    {
        NoteUsedResource(_heapElements, Mtl.ResourceUsageRead);
        NoteUsedResource(_heapFeedback, Mtl.ResourceUsageRead | Mtl.ResourceUsageWrite);
        if (_heapTexturesChanged)
        {
            _heapTexturesChanged = false;
            var residency = BindlessResidency;
            var textures = new List<nint>();
            _heapTextures.Declared.Clear();
            var signature = 0UL;
            // The heap's records, and the call records' (MetalRenderHost.ShaderCalls.cs).
            foreach (var entry in _heapEntries.Values.Concat(_callRecordEntries.Values))
            {
                if (entry.Texture == 0)
                {
                    continue;
                }

                textures.Add(entry.Texture);
                if (entry.Declared || residency is null)
                {
                    _heapTextures.Declared.Add(entry.Texture);
                    signature = (signature ^ (ulong)entry.Texture) * 0x100000001B3UL;
                }
                else
                {
                    residency.Add(entry.Texture);
                }
            }

            _heapTextures.Textures = [.. textures];
            residency?.Commit();
            if (_heapTextures.DeclaredSignature != signature)
            {
                // The encoder has not been told about these yet.
                _heapTextures.DeclaredSignature = signature;
                _heapTextures.DeclaredEncoder = -1;
            }
        }

        if (!_bindlessDeclarations.Contains(_heapTextures))
        {
            _bindlessDeclarations.Add(_heapTextures);
            BindlessDeclaredTextures += _heapTextures.Declared.Count;
        }

        BindlessResidency?.PruneIfDue(BindlessResidencyTables, SubmissionSerial, CompletedSerial);
    }

    private ulong RequireHeapAddress(ulong address, MetalStageBindings stage) =>
        address != 0
            ? address
            : throw Fatal($"A program reads bindless heap records before the heap exists: hash=0x{stage.Program.Hash:X16}.");

    // Every device table whose textures the residency set keeps: the bindless image tables', and the heap's.
    private IEnumerable<BindlessDeviceTable> BindlessResidencyTables =>
        HeapCreated ? _bindlessTables.Values.Append(_heapTextures) : _bindlessTables.Values;
}
