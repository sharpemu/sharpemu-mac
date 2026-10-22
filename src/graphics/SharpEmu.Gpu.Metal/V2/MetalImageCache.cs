// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text;
using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Scheduling;

namespace SharpEmu.Gpu.Metal.V2;

// What the cache needs from the render host to move image contents: a command
// buffer with no open encoder, GPU fills of guest memory, texture clears, and a
// way to finish recorded work before the CPU reads guest bytes.
internal interface IMetalImageCacheOwner
{
    nint Device { get; }

    // The serial of the command buffer being recorded, without touching its
    // encoders: noting that an image is GPU-written must not end the render pass.
    ulong CurrentSerial { get; }

    // The newest submission the GPU has finished.
    ulong CompletedSerial { get; }

    // Ends any open encoder and hands over the command buffer being recorded.
    IMetalSubmissionContext BeginTransfer(MetalTransferCause cause);

    // Lands the command-processor copies held for the commit that write these guest bytes, so
    // a transfer reading them next sees them (MetalRenderHost.LabelsAtCommit).
    void LandHeldCopies(ulong address, ulong size)
    {
    }

    // The same, for a transfer that reads or writes only these textures (zero for an
    // unused one). A clear still waiting for its pass has to land before the transfer
    // only when the transfer touches the cleared texture; every other one keeps waiting.
    IMetalSubmissionContext BeginTransfer(MetalTransferCause cause, nint texture, nint otherTexture = 0) =>
        BeginTransfer(cause);

    // Recorded or in-flight GPU work still writes into this guest range, so its bytes
    // are not settled yet.
    bool HasPendingGpuWrite(ulong address, ulong size);

    // Fills a guest byte range with a repeating dword on the GPU.
    void FillGuestRange(ulong address, ulong size, uint value);

    // Recorded work stores into this guest range, so a later read of these bytes
    // waits for the submission being recorded and a read of anything else does not.
    void NoteGuestWrite(ulong address, ulong size);

    // Finishes every recorded command buffer so the CPU sees the GPU's writes.
    void WaitForGpu();

    // Blocks a thread other than the GPU worker until the submission with this serial has
    // completed, asking the worker to commit it if it is still being recorded. False when it
    // did not complete in time, or when called on the worker, which would wait for itself.
    bool WaitForSubmissionOffWorker(ulong serial, int timeoutMilliseconds) => CompletedSerial >= serial;

    // Hands work to the GPU worker from another thread without waiting for it. False when it
    // was not handed over: on the worker itself, or with the worker gone.
    bool TryRunOnWorker(Action work) => false;

    void ClearTextureColor(nint texture, ulong pixelFormat, in SubresourceRange range, uint width, uint height, in GpuClearColor color);

    void ClearTextureDepthStencil(
        nint texture,
        ulong pixelFormat,
        in SubresourceRange range,
        uint width,
        uint height,
        float depth,
        byte stencil,
        bool clearDepth,
        bool clearStencil);
}

/// <summary>
/// The address-keyed image store, with the semantics of the Vulkan
/// <c>GuestImageCache</c>: images are found by guest range through a page-owner
/// table, CPU writes are caught by the shared HLE page guard, and GPU writes are
/// written back to guest memory lazily.
///
/// Unified memory removes most of the Vulkan machinery: guest memory is the single
/// source of truth for every image, so a replacement, a growth or an eviction is
/// "write the texture back, then drop it" rather than an image-to-image copy. There
/// are no size caps, no CPU detiling and no qualified-descriptor list: an unknown
/// format maps to a default and logs once.
/// </summary>
internal sealed partial class MetalImageCache : IGuestImageStore, IDisposable
{
    private const ulong TicksBeforeRemoval = 32;
    private const ulong MiB = 1024 * 1024;

    private static readonly nint SelAlloc = MetalNative.Selector("alloc");
    private static readonly nint SelInit = MetalNative.Selector("init");
    private static readonly nint SelRelease = MetalNative.Selector("release");
    private static readonly nint SelNewTexture = MetalNative.Selector("newTextureWithDescriptor:");
    private static readonly nint SelSetTextureType = MetalNative.Selector("setTextureType:");
    private static readonly nint SelSetPixelFormat = MetalNative.Selector("setPixelFormat:");
    private static readonly nint SelSetWidth = MetalNative.Selector("setWidth:");
    private static readonly nint SelSetHeight = MetalNative.Selector("setHeight:");
    private static readonly nint SelSetDepth = MetalNative.Selector("setDepth:");
    private static readonly nint SelSetMipmapLevelCount = MetalNative.Selector("setMipmapLevelCount:");
    private static readonly nint SelSetArrayLength = MetalNative.Selector("setArrayLength:");
    private static readonly nint SelSetSampleCount = MetalNative.Selector("setSampleCount:");
    private static readonly nint SelSetUsage = MetalNative.Selector("setUsage:");
    private static readonly nint SelSetStorageMode = MetalNative.Selector("setStorageMode:");
    private static readonly nint SelSetCompressionType = MetalNative.Selector("setCompressionType:");
    private static readonly nint SelNewBufferTexture = MetalNative.Selector("newTextureWithDescriptor:offset:bytesPerRow:");
    private static readonly nint SelMinimumLinearAlignment =
        MetalNative.Selector("minimumLinearTextureAlignmentForPixelFormat:");

    private readonly IMetalImageCacheOwner _owner;
    private readonly IMetalImageTransfer _transfer;
    private readonly MetalGuestMemory _guestMemory;
    private readonly PageGuard _pages;
    private readonly IGuestBackedSpace _backing;
    private readonly RegionLock _lock = new(RegionLock.Category.ImageCache);
    private readonly MetalImageSlots _slots = new();
    private readonly MetalImagePageOwners _pageOwners = new();

    /// <summary>The staged guest ranges whose page watches and invalidations share this cache's hooks.</summary>
    public MetalStagedRangeCache? StagedRanges { get; set; }
    private readonly SortedDictionary<ulong, SurfaceMetadata> _surfaceMetadata = new();
    private readonly Dictionary<GpuFormat, ResourceSlotIdentifier> _nullImages = new();

    // Evicted images whose Metal objects a recorded submission still names.
    private readonly List<(ulong Serial, MetalCachedImage Image)> _retiring = new();
    private ulong _totalUsedMemory;
    private ulong _collectionTick;
    // Collection ticks at which the current and the previous presented frame began:
    // an image used in either is part of the working set and is never evicted for
    // budget, whatever its tick age. Ticks advance per completed submission, and a
    // 4K frame spans hundreds of them, so a tick-age rule alone evicts images the same
    // frame is still about to use and the cache thrashes (Demon's Souls at 3 GB:
    // 64 MB targets written back and re-uploaded every frame, 0.5 fps).
    private ulong _frameStartTick;
    private ulong _previousFrameStartTick;
    private uint _queryEpoch;
    private long _imageSetVersion;
    private long _createdImages;
    private long _uploads;
    private long _writebacks;
    private long _evictions;
    private bool _watchingStopped;
    private bool _disposed;
    private readonly Dictionary<ulong, ulong> _linearAlignments = new();
    private long _linearViews;
    private long _linearViewPromotions;

    public MetalImageCache(
        IMetalImageCacheOwner owner,
        IMetalImageTransfer transfer,
        MetalGuestMemory guestMemory,
        PageGuard pages,
        IGuestBackedSpace backing)
    {
        _owner = owner;
        _transfer = transfer;
        _guestMemory = guestMemory;
        MetalBlockCubeDump.Peek ??= guestMemory.TryPeek;
        _pages = pages;
        _backing = backing;
        InitializeLossy();
        ConfigureMemoryPolicy(
            MetalImageBudgetSettings.FromEnvironment(MetalHostMemory.Physical),
            MetalImageRetirementSettings.FromEnvironment(),
            MetalHostMemory.Sample,
            System.Diagnostics.Stopwatch.GetTimestamp,
            static () => MetalHostMemory.PressureEvents);
    }

    public ulong TotalUsedMemory => _totalUsedMemory;

    public int ImageCount => _slots.Count;

    /// <summary>Images that are live textures over the guest window.</summary>
    public long LinearViewCount => Volatile.Read(ref _linearViews);

    /// <summary>Window views that became owned textures since the last read.</summary>
    public long TakeLinearViewPromotions() => Interlocked.Exchange(ref _linearViewPromotions, 0);

    public (long Images, long Uploads, long Writebacks, long Evictions) Counters =>
        (Volatile.Read(ref _createdImages), Volatile.Read(ref _uploads),
            Volatile.Read(ref _writebacks), Volatile.Read(ref _evictions));

    private static bool IsValidRange(ulong address, ulong size) =>
        ImageDescription.IsValidRange(new GuestSpan(address, size));

    /// <summary>
    /// The image for a request: an existing one when the backing matches or the
    /// overlap rules say so, otherwise a new one. The request's view may come back
    /// retargeted at a mip or a layer of a larger cached image.
    /// </summary>
    public ResourceSlotIdentifier FindImage(ref ImageRequest request, bool exactFormat = false)
    {
        ValidateRequest(in request);
        if (ImageDescription.IsEmptyRange(request.Description.Data))
        {
            using var nullHeld = _lock.Hold();
            return GetNullImage(in request);
        }

        var metadataBaseLayer = request.View.BaseLayer;
        ResourceSlotIdentifier result;
        bool scanDcc;
        long imageWriteSequence;
        // One hold for the lookup and the DCC metadata the request brings: a texture
        // binding takes the lock once, and only a DCC slice scan takes it again.
        using (var held = _lock.Hold())
        {
            result = LookUpImageLocked(ref request, exactFormat);
            scanDcc = AdoptDccMetadataLocked(result, in request, out imageWriteSequence);
        }

        if (scanDcc)
        {
            MaterializeDccClear(result, in request, metadataBaseLayer, imageWriteSequence);
        }

        return result;
    }

    // Caller holds the lock.
    private ResourceSlotIdentifier LookUpImageLocked(ref ImageRequest request, bool exactFormat)
    {
        var candidates = FindImagesInRange(request.Description.Data.Address, request.Description.Data.Size, pageOverlap: false);
        var result = ResourceSlotIdentifier.Invalid;
        foreach (var candidate in candidates)
        {
            if (HasSameBacking(_slots[candidate].Description, request.Description, exactFormat))
            {
                result = candidate;
            }
        }

        var viewMip = -1;
        var viewLayer = -1;
        if (!result.IsValid)
        {
            foreach (var candidate in candidates)
            {
                // Resolving one overlap may release another candidate, or the image an
                // earlier one chose, so each step starts from what is still live.
                if (_slots.TryGet(candidate) is null)
                {
                    continue;
                }

                if (result.IsValid && _slots.TryGet(result) is null)
                {
                    result = ResourceSlotIdentifier.Invalid;
                }

                viewMip = -1;
                viewLayer = -1;
                var merged = result.IsValid ? _slots[result].Description : request.Description;
                var overlap = ResolveOverlap(in merged, in request, candidate, result, exactFormat);
                if (overlap.Image.IsValid)
                {
                    result = overlap.Image;
                    viewMip = overlap.Mip;
                    viewLayer = overlap.Layer;
                }
            }
        }

        if (result.IsValid && _slots.TryGet(result) is null)
        {
            result = ResourceSlotIdentifier.Invalid;
            viewMip = -1;
            viewLayer = -1;
        }

        if (result.IsValid)
        {
            var resolved = _slots[result];
            var formatMismatch = exactFormat && resolved.Description.PixelFormat != request.Description.PixelFormat;
            var fewerResources = resolved.Description.Resources < request.Description.Resources;
            if (formatMismatch || fewerResources || !resolved.SupportsViewType(request.View))
            {
                var cause = formatMismatch
                    ? MetalImageReleaseCause.FormatMismatch
                    : fewerResources ? MetalImageReleaseCause.FewerResources : MetalImageReleaseCause.ViewType;
                ReportRelease(cause, resolved.Description, request.Description, request.View);
                ReleaseImage(result, cause);
                result = ResourceSlotIdentifier.Invalid;
                viewMip = -1;
                viewLayer = -1;
            }
        }

        if (!result.IsValid)
        {
            var sampledOnly = request.Role == ImageRole.Texture;
            result = InsertImage(
                request.Description, sampledOnly, request.Role,
                sampledOnly ? DesiredFirstLevel(in request.Description, request.View.BaseLevel) : 0,
                lossyAllowed: sampledOnly && ViewKeepsLossy(in request));
        }

        var image = _slots[result];
        if (viewMip >= 0)
        {
            request.View = request.View with { BaseLevel = (uint)viewMip };
        }

        if (viewLayer >= 0)
        {
            request.View = request.View with { BaseLayer = (uint)viewLayer };
        }

        image.LastAccessTick = _collectionTick;
        return result;
    }

    public MetalCachedImage GetImage(ResourceSlotIdentifier imageIdentifier) => _slots[imageIdentifier];

    /// <summary>Changes whenever an image is added to or dropped from the cache.</summary>
    public long ImageSetVersion => Volatile.Read(ref _imageSetVersion);

    /// <summary>The counts of the mutations a memo of this cache's state has to see (MetalImageEpochs).</summary>
    public MetalImageEpochs Epochs { get; } = new();

    public bool TryGetImage(ResourceSlotIdentifier imageIdentifier, out MetalCachedImage image)
    {
        using var held = _lock.Hold();
        var found = _slots.TryGet(imageIdentifier);
        image = found!;
        return found is not null;
    }

    private long _aliasesKept;
    private long _aliasPublishes;
    private long _aliasNarrowWrites;
    private long _aliasReadsSkipped;

    /// <summary>
    /// How often an alias was kept instead of dropped, resynchronized, and handed over to a
    /// write over just the bytes the two share.
    /// </summary>
    public (long Kept, long Published, long Narrow, long ReadsSkipped) AliasCounters =>
        (Interlocked.Exchange(ref _aliasesKept, 0), Interlocked.Exchange(ref _aliasPublishes, 0),
         Interlocked.Exchange(ref _aliasNarrowWrites, 0), Interlocked.Exchange(ref _aliasReadsSkipped, 0));

    // SHARPEMU_METAL_V2_ALIAS_SKIP_MISMATCH=0 publishes every GPU-written alias a read
    // overlaps, whatever its layout.
    private static readonly bool AliasReadSkipMismatchEnabled =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_ALIAS_SKIP_MISMATCH") != "0";

    // Whether bytes laid out for one image read back as anything but noise through the
    // other: the same tile mode, element size and row stride, so a texel of one is a
    // texel of the other. A depth-tiled shadow array over a render-target-tiled 720p
    // transient, or a 4 byte surface over an 8 byte one, shares addresses and nothing
    // else; on the console such a read returns scrambled bytes too.
    // The row stride is in elements: a block-compressed pitch counts texels, four to a
    // block, so a BC3 surface and the RGBA32Uint view a compute shader writes its blocks
    // through (UE5's virtual texture pool) have the same rows at a quarter of the pitch.
    // Comparing texel pitches skipped every such read, and the pool kept stale tiles.
    internal static bool SameTexelLayout(in ImageDescription a, in ImageDescription b) =>
        a.TileMode == b.TileMode && a.BytesPerBlock == b.BytesPerBlock &&
        (ulong)a.BlockExtent.Width * a.BytesPerBlock == (ulong)b.BlockExtent.Width * b.BytesPerBlock;

    /// <summary>
    /// Makes this image the one that agrees with guest memory, against every other
    /// cached image over the same bytes. Two images can describe one surface — a depth
    /// buffer that is also sampled as a colour format needs a texture of each — and
    /// then the one the GPU last wrote owns the contents. Publishing it and marking
    /// this one stale is what the old drop-and-recreate did implicitly.
    ///
    /// <paramref name="keepsDepthPlane"/> says the write is to the stencil plane alone.
    ///
    /// Caller holds the lock.
    /// </summary>
    // SHARPEMU_METAL_V2_NEWER_WRITER_WINS=0 restores publishing every GPU-modified alias into the
    // image being acquired, whichever of the two wrote the shared bytes last.
    private static readonly bool NewerWriterWins =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_NEWER_WRITER_WINS") != "0";

    private long _aliasNewerKept;

    /// <summary>Alias syncs that kept this image's newer contents instead of publishing an older alias.</summary>
    public long TakeAliasNewerKept() => Interlocked.Exchange(ref _aliasNewerKept, 0);

    private void SynchronizeAliases(
        ResourceSlotIdentifier imageIdentifier, bool takesGpuOwnership, bool keepsDepthPlane = false)
    {
        var image = _slots[imageIdentifier];
        var address = image.Description.Data.Address;
        var size = image.Description.Data.Size;
        if (takesGpuOwnership)
        {
            // No lossy image may hold, or be made over, what this image is about to write.
            NoteGpuWrittenImageLocked(image);
        }

        SynchronizeStencilPlanes(imageIdentifier, image, takesGpuOwnership);
        if (ImageDescription.IsEmptyRange(image.Description.Data) || !_pageOwners.MayHaveOwners(address, size))
        {
            return;
        }

        foreach (var otherIdentifier in FindImagesInRange(address, size, pageOverlap: false))
        {
            if (otherIdentifier == imageIdentifier ||
                _slots.TryGet(otherIdentifier) is not { } other || !other.Overlaps(address, size))
            {
                continue;
            }

            if (takesGpuOwnership && other.IsWindowLinearView)
            {
                // This image is about to write bytes the window view reads in place, and
                // the write lands in a texture, not in guest memory.
                PromoteLinearView(other);
            }

            var otherOwedGuestMemory = other.IsGpuModified;
            if (other.IsGpuModified)
            {
                // Only the bytes the two actually share change hands. The overlap may be
                // a fraction of either surface, and publishing or reloading the rest of
                // them moves bytes neither image is asking about: a 64 KiB overlap on a
                // 19 MiB target used to cost a full retile of one and a full detile of
                // the other, every time the pair was used.
                ReportAliasPair(image, other);
                var sharedStart = Math.Max(image.Description.Data.Address, other.Description.Data.Address);
                var sharedEnd = Math.Min(
                    image.Description.Data.Address + image.Description.Data.Size,
                    other.Description.Data.Address + other.Description.Data.Size);
                var sharedSize = sharedEnd > sharedStart ? sharedEnd - sharedStart : 0;
                var (otherStart, otherEnd) = other.OverlapWithin(sharedStart, sharedSize);
                if (other.IsCpuDirty && otherEnd > otherStart &&
                    otherStart >= other.DirtyStart && otherEnd <= other.DirtyEnd)
                {
                    // The other image already gave these bytes up to a write through an
                    // alias and owns only the rest: its texture is stale here, so it has
                    // nothing to publish, and the image that wrote them since - this one,
                    // or one this loop publishes on its own turn - is what holds them.
                    continue;
                }

                if (NewerWriterWins && image.IsGpuModified && image.GpuWriteSequence > other.GpuWriteSequence)
                {
                    // This image wrote the shared bytes after the other did - typically a DCC
                    // fast clear this lookup just materialized - so its texture holds what the
                    // guest last put there. Publishing the other into it would put the older
                    // contents back over the newer: UE5's shadow mask came out of its clear
                    // white, then had the lower half of an overlapping transient copied over
                    // it, and its per-object shadow channel left every character unlit; the
                    // refraction buffer kept stale offsets the same way (the screen blocks).
                    // The other gives the run up instead, and takes it back from this image
                    // through the ordinary path when it is used again.
                    if (!TryHandOverAliasRun(other, otherStart, otherEnd, out _))
                    {
                        WriteBackToGuest(other, MetalWritebackCause.ImageOverlap);
                        other.ClearGpuModified();
                        other.MarkCpuDirty(MetalImageDirtyCause.AliasWrite);
                    }

                    _aliasNewerKept++;
                    NoteAliasPublication("newer-kept", image, other, otherOwedGuestMemory, otherStart, otherEnd);
                    continue;
                }

                if (TryPublishAliasOnDevice(image, other))
                {
                    _aliasPublishes++;
                    NoteAliasPublication("on-device", image, other, otherOwedGuestMemory, otherStart, otherEnd);
                    // A device copy leaves the source GPU-modified, so a write through this
                    // image still has to take ownership from it, or both would publish.
                    if (takesGpuOwnership)
                    {
                        other.ClearGpuModified();
                        other.MarkCpuDirty(MetalImageDirtyCause.AliasWrite);
                    }

                    continue;
                }

                if (!takesGpuOwnership && AliasReadSkipMismatchEnabled &&
                    !SameTexelLayout(in image.Description, in other.Description))
                {
                    // A read through a layout the written bytes do not fit gets noise on the
                    // console and would get noise here, after a retile of the other image
                    // out and this one back in. Demon's Souls binds a shadow array over the
                    // memory its 720p post-process transients reuse and never samples it in
                    // scenes without shadow casters: three such round trips a frame at the
                    // title. The texture keeps what it last held; the other stays
                    // GPU-modified, so a reader whose layout does fit is still served.
                    _aliasReadsSkipped++;
                    NoteAliasPublication("read-skip-layout", image, other, otherOwedGuestMemory, otherStart, otherEnd);
                    continue;
                }

                var (dirtyStart, dirtyEnd) = image.OverlapWithin(sharedStart, sharedSize);
                var declined = MetalAliasFallbackReason.None;
                if (takesGpuOwnership && TryHandOverAliasRun(other, otherStart, otherEnd, out declined))
                {
                    // The other image keeps GPU ownership of everything but the shared
                    // run, so it is neither written back whole nor reloaded whole: a small
                    // transient placed inside a 4K target costs its own bytes, not the
                    // target's 64 MiB, on every use of the pair.
                    MarkAliasRunDirty(image, dirtyStart, dirtyEnd);
                    _aliasPublishes++;
                    _aliasNarrowWrites++;
                    NoteAliasPublication("narrow-write", image, other, otherOwedGuestMemory, otherStart, otherEnd);
                    continue;
                }

                if (takesGpuOwnership)
                {
                    // The other image gives up GPU ownership below, and nothing writes back
                    // an image that does not own it: a run published on its own would leave
                    // the rest of the other image in its texture only, for the refresh that
                    // follows to replace with the stale guest bytes.
                    NoteAliasFallback(write: true, declined, image, other, otherOwedGuestMemory, otherStart, otherEnd);
                    WriteBackToGuest(other, MetalWritebackCause.ImageOverlap);
                    NoteAliasPublication("whole-write", image, other, otherOwedGuestMemory, otherStart, otherEnd);
                }
                else
                {
                    var whole = PublishAliasRange(other, sharedStart, sharedSize);
                    if (whole != MetalAliasFallbackReason.None)
                    {
                        NoteAliasFallback(write: false, whole, image, other, otherOwedGuestMemory, otherStart, otherEnd);
                    }

                    NoteAliasPublication(
                        whole == MetalAliasFallbackReason.None ? "read-run" : "whole-read",
                        image, other, otherOwedGuestMemory, otherStart, otherEnd);
                }

                MarkAliasRunDirty(image, dirtyStart, dirtyEnd);
                _aliasPublishes++;
            }

            if (takesGpuOwnership && keepsDepthPlane && !otherOwedGuestMemory &&
                !(image.Description.HasStencil &&
                  other.Overlaps(image.Description.Stencil.Address, image.Description.Stencil.Size)))
            {
                // A pass that writes only the stencil plane leaves the bytes this image
                // shares with the other one - its depth plane - as they were, so an alias
                // that owed guest memory nothing still agrees with both.
                continue;
            }

            if (takesGpuOwnership)
            {
                // Whatever this image is about to write leaves the others stale - but only
                // over the bytes it shares with them. The rest of the other image already
                // agrees with guest memory when it owes guest memory nothing and its page
                // watch still reports every CPU store, so its next refresh needs only the
                // shared run. Otherwise the whole image is marked as before: an image
                // still GPU-modified here has bytes only its texture holds, and one with
                // no watch may have missed CPU stores a whole refresh would pick up.
                var narrow = !other.IsGpuModified && other.IsWatched;
                other.ClearGpuModified();
                var (staleStart, staleEnd) = other.OverlapWithin(address, size);
                if (narrow && staleEnd > staleStart)
                {
                    other.MarkCpuDirty(MetalImageDirtyCause.AliasWrite, staleStart, staleEnd);
                }
                else
                {
                    other.MarkCpuDirty(MetalImageDirtyCause.AliasWrite);
                }
            }
        }
    }

    // Depth images with a stencil plane. The page owner index holds an image's depth
    // bytes only, and the stencil plane lives elsewhere (DB_STENCIL_*_BASE), so an image
    // over those bytes - the R8 view a deferred lighting pass samples the stencil through
    // - would never find the depth image that wrote them.
    private readonly HashSet<ResourceSlotIdentifier> _stencilOwners = new();

    /// <summary>
    /// The stencil plane counterpart of the alias rules: the stencil a depth target writes
    /// lands in its texture, and an image that reads those bytes must see it, as one that
    /// writes them must leave the depth image reloading its stencil. A shader that samples
    /// the plane as R8 is how a title tells lit geometry from sky, so a stale plane
    /// unlights the whole scene.
    ///
    /// Caller holds the lock.
    /// </summary>
    private void SynchronizeStencilPlanes(
        ResourceSlotIdentifier imageIdentifier, MetalCachedImage image, bool takesGpuOwnership)
    {
        if (!StencilPlaneSyncEnabled || _stencilOwners.Count == 0 ||
            ImageDescription.IsEmptyRange(image.Description.Data))
        {
            return;
        }

        var address = image.Description.Data.Address;
        var size = image.Description.Data.Size;
        foreach (var ownerIdentifier in _stencilOwners)
        {
            if (ownerIdentifier == imageIdentifier || _slots.TryGet(ownerIdentifier) is not { } depth ||
                !depth.Description.HasStencil || depth.Overlaps(address, size) ||
                !GuestRangeOverlap.Bytes(depth.Description.Stencil.Address, depth.Description.Stencil.Size, address, size))
            {
                continue;
            }

            NoteForeignStencilShare(depth, image);
            var (start, end) = image.OverlapWithin(depth.Description.Stencil.Address, depth.Description.Stencil.Size);
            if (depth.IsGpuModified)
            {
                // Both planes go out together: only the depth image's texture holds them.
                WriteBackToGuest(depth, MetalWritebackCause.ImageOverlap);
                MarkAliasRunDirty(image, start, end);
                _stencilPlanePublishes++;
            }

            if (takesGpuOwnership)
            {
                // Written back above if it owed anything. The depth plane its texture holds is
                // still the newest; only the stencil is this image's from now on, and it is
                // copied over on the device when the depth image binds next. Reloading both
                // planes from guest memory instead round-trips the depth plane, and a reload
                // that does not give back exactly what was written left Black Myth: Wukong's
                // scene depth as garbage or zeros, every pass that tests or samples it with it.
                depth.ClearGpuModified();
                if (StencilDeviceHandOver)
                {
                    depth.StencilPlaneStale = true;
                }
                else
                {
                    depth.MarkCpuDirty(MetalImageDirtyCause.AliasWrite);
                }
            }
        }

        if (!image.Description.HasStencil)
        {
            return;
        }

        // A depth image binds: an image that wrote its stencil bytes since hands them over,
        // on the device when it can (the depth plane stays as the texture holds it), and
        // otherwise by publishing them while the depth image reloads both planes from guest
        // memory.
        var stencil = image.Description.Stencil;
        foreach (var otherIdentifier in FindImagesInRange(stencil.Address, stencil.Size, pageOverlap: false))
        {
            if (otherIdentifier == imageIdentifier || _slots.TryGet(otherIdentifier) is not { } other ||
                !other.Overlaps(stencil.Address, stencil.Size) || image.Overlaps(other.Description.Data.Address, other.Description.Data.Size))
            {
                continue;
            }

            NoteForeignStencilShare(image, other);
            var handedOver = (other.IsGpuModified || image.StencilPlaneStale) && TryCopyStencilOnDevice(image, other);
            if (!handedOver && other.IsGpuModified)
            {
                // The depth plane is written out first so the reload keeps it, and the
                // other image second so its stencil bytes are the ones left in guest memory.
                WriteBackToGuest(image, MetalWritebackCause.ImageOverlap);
                WriteBackToGuest(other, MetalWritebackCause.ImageOverlap);
                image.MarkCpuDirty(MetalImageDirtyCause.AliasWrite);
                _stencilPlanePublishes++;
            }

            if (takesGpuOwnership)
            {
                // Published above if it owed anything, or now held by the depth image's
                // stencil, so a whole reload loses nothing.
                other.ClearGpuModified();
                other.MarkCpuDirty(MetalImageDirtyCause.AliasWrite);
            }
        }

        if (image.StencilPlaneStale)
        {
            // The image that took the stencil over is gone (its bytes were published when it
            // was released): only guest memory has them now.
            image.StencilPlaneStale = false;
            image.MarkCpuDirty(MetalImageDirtyCause.AliasWrite);
        }
    }

    // SHARPEMU_METAL_V2_STENCIL_DEVICE=0 hands a stencil plane written through another image
    // back to its depth image through guest memory again (both planes reloaded).
    private static readonly bool StencilDeviceHandOver =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_STENCIL_DEVICE") != "0";

    private long _stencilDeviceHandOvers;

    // The stencil an 8-bit image over a depth image's stencil bytes holds, copied into the
    // depth texture's stencil plane. Caller holds the lock.
    private bool TryCopyStencilOnDevice(MetalCachedImage depth, MetalCachedImage other)
    {
        if (!StencilDeviceHandOver || depth.Texture == 0 || other.Texture == 0 ||
            depth.IsCpuDirty || other.IsCpuDirty ||
            depth.IsWindowLinearView || other.IsWindowLinearView ||
            depth.FirstLevel != 0 || other.FirstLevel != 0 ||
            other.Description.Data.Address != depth.Description.Stencil.Address)
        {
            return false;
        }

        if (depth.HoldsStencilFrom(other))
        {
            depth.StencilPlaneStale = false;
            return true;
        }

        EnsureSparseBacked(depth);
        var context = _owner.BeginTransfer(MetalTransferCause.Writeback, other.Texture, depth.Texture);
        if (!_transfer.TryEncodeStencilFromColor(
                context, in other.Description, other.Texture, in depth.Description, depth.Texture))
        {
            return false;
        }

        // The stencil plane changed on the GPU and guest memory does not have it yet.
        depth.MarkGpuModified(depthPlaneChanged: false);
        depth.NoteStencilFrom(other);
        _stencilDeviceHandOvers++;
        return true;
    }

    private long _stencilPlanePublishes;

    // The plane read on purpose is an 8-bit depth-tiled image at DB_STENCIL_*_BASE. Any
    // other surface over it is memory the title reuses, which the hand-over above treats
    // like any alias; said once per pair, since a whole-plane publish into such a surface
    // is the first thing to rule out when bands of garbage cross a frame.
    private static void NoteForeignStencilShare(MetalCachedImage depth, MetalCachedImage other)
    {
        ref readonly var plane = ref depth.Description.Stencil;
        ref readonly var data = ref other.Description.Data;
        if (data.Address == plane.Address && other.Description.BytesPerBlock == 1 &&
            other.Description.TileMode == GuestTileMode.Depth)
        {
            return;
        }

        MetalV2Diagnostics.WarnOnce(
            $"stencil-plane-foreign:{plane.Address:X}:{data.Address:X}:{other.Description.PixelFormat}",
            $"A surface that is not a view of a stencil plane shares its bytes: plane=0x{plane.Address:X16}+0x{plane.Size:X} " +
            $"surface=0x{data.Address:X16}+0x{data.Size:X} {other.Description.Extent.Width}x{other.Description.Extent.Height} " +
            $"format={other.Description.PixelFormat} tile={other.Description.TileMode}.");
    }

    // SHARPEMU_METAL_V2_STENCIL_PLANES=0 leaves stencil planes unsynchronised again, for a
    // comparison.
    private static readonly bool StencilPlaneSyncEnabled =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_STENCIL_PLANES") != "0";

    /// <summary>Stencil planes handed between a depth image and an image over them.</summary>
    public long TakeStencilPlanePublishes() => Interlocked.Exchange(ref _stencilPlanePublishes, 0);

    public long TakeStencilDeviceHandOvers() => Interlocked.Exchange(ref _stencilDeviceHandOvers, 0);

    // A write through an alias takes over the run a GPU-written image shares with it and
    // leaves that image GPU-modified for the rest: the run is published, so guest memory
    // holds the image's bytes there until the write through the alias lands, and the
    // image is marked stale over the run alone, so its next use reloads just that run.
    // False - and the caller writes the image back whole, as before - unless that reload
    // is certain to be a narrow one: an image dirty over the run while it stays
    // GPU-modified must never be refreshed whole, since that would replace bytes only its
    // texture holds. A run already stale on the image is refused for the same reason
    // when it lies outside the stale interval, which one interval cannot hold apart.
    private bool TryHandOverAliasRun(
        MetalCachedImage other, ulong start, ulong end, out MetalAliasFallbackReason declined)
    {
        declined = !SubRangeTransfersEnabled ? MetalAliasFallbackReason.SubRangeDisabled
            : end <= start ? MetalAliasFallbackReason.EmptyRun
            : other.Description.HasStencil ? MetalAliasFallbackReason.Stencil
            : start == 0 && end >= other.Description.Data.Size ? MetalAliasFallbackReason.RunIsWholeOther
            : end - start > other.Description.Data.Size / 2 ? MetalAliasFallbackReason.RunPastHalf
            : other.IsCpuDirty && (start < other.DirtyStart || end > other.DirtyEnd)
                ? MetalAliasFallbackReason.OtherDirtyOutsideRun
            : MetalAliasFallbackReason.None;
        if (declined != MetalAliasFallbackReason.None)
        {
            return false;
        }

        if (!other.IsPublished(start, end) &&
            !TryWriteBackRun(other, start, end, MetalWritebackCause.ImageOverlap, out declined))
        {
            return false;
        }

        other.MarkCpuDirty(MetalImageDirtyCause.AliasWrite, start, end);
        return true;
    }

    // Marks the run an image shares with a GPU-written alias stale. An image that is
    // itself GPU-modified may already be stale over a run another alias took from it,
    // and one interval cannot hold two runs with bytes only its texture holds between
    // them - a refresh of the merged interval would replace those bytes with the guest's.
    // The run it holds is refreshed first instead. That is safe here: an alias still
    // owing those bytes overlaps this image too, so this same loop publishes it and marks
    // the run stale again.
    private void MarkAliasRunDirty(MetalCachedImage image, ulong start, ulong end)
    {
        if (image.IsGpuModified && image.IsCpuDirty && end > start &&
            (start > image.DirtyEnd || end < image.DirtyStart))
        {
            RefreshFromGuest(image);
        }

        image.MarkCpuDirty(MetalImageDirtyCause.AliasWrite, start, end);
    }

    /// <summary>The texture a shader binds for one image request.</summary>
    public nint AcquireTextureView(ResourceSlotIdentifier imageIdentifier, in ImageRequest request) =>
        AcquireTextureView(imageIdentifier, in request, out _);

    /// <summary>The texture a shader binds for one image request, with its cached gpuResourceID.</summary>
    public nint AcquireTextureView(ResourceSlotIdentifier imageIdentifier, in ImageRequest request, out ulong resourceId)
    {
        using var held = _lock.Hold();
        // An attachment waiting for its pass is not marked written yet; anything bound
        // now must see it the way the acquire left it before the wait existed.
        DrainDeferredRefreshesLocked();
        var image = _slots[imageIdentifier];
        image.LastAccessTick = _collectionTick;
        if (request.Role == ImageRole.StorageImage)
        {
            PromoteLinearView(image);
            EnsureStorageUsageLocked(image);
            EnsureSparseBacked(image);
            WatchImage(imageIdentifier);
            SynchronizeAliases(imageIdentifier, takesGpuOwnership: true);
            RefreshFromGuest(image);
            image.Uses.Storage = true;
            image.MarkGpuModified();
            NoteLinearGpuWrite(image);
            image.GpuWriteSerial = _owner.CurrentSerial;
            image.NoteUse(image.GpuWriteSerial);
        }
        else
        {
            if (image.IsWindowLinearView &&
                (request.Role != ImageRole.Texture || !image.CanWindowViewServe(request.View)))
            {
                // A render target needs an owned texture, and so does a view whose format
                // or component swizzle the window texture cannot give.
                PromoteLinearView(image);
            }

            // Every sampled binding comes through here, the ones that reuse an earlier
            // lookup too, so this is where the texture follows the binding's BASE_LEVEL.
            if (request.Role == ImageRole.Texture)
            {
                FitResidentLevelsLocked(image, in request.View);
            }
            else
            {
                PromoteLinearView(image);
            }

            WatchImage(imageIdentifier);
            SynchronizeAliases(imageIdentifier, takesGpuOwnership: false);
            RefreshFromGuest(image);
            image.Uses.Texture = true;
            image.NoteUse(_owner.CurrentSerial);
        }

        MaterializeSampledDccClear(imageIdentifier, image, in request);
        if (request.Role == ImageRole.Texture && image.Description.IsDepth &&
            TryDepthTwinView(image, in request.View, out var depthView))
        {
            // A colour twin sampled through the depth texture: a same-format view, whose
            // swizzle needs no usage the depth texture lacks.
            return image.GetOrCreateView(depthView, out resourceId, sameFormatSwizzle: true);
        }

        EnsureViewUsageLocked(image, in request.View);
        return image.GetOrCreateView(request.View, out resourceId);
    }

    /// <summary>
    /// A DCC fast clear stays pending until the surface binds as a colour target again.
    /// A shader that samples the surface before that must still see the cleared contents,
    /// and MaterializeDccClear cannot do it: a texture request carries no metadata range
    /// of its own, and the register clear colour it would need is only known while the
    /// surface is a target. What the image remembers from when it *was* a target is
    /// enough, so the clear is materialized here from the tracked state, or from the
    /// guest's own DCC bytes when the fill was never seen.
    ///
    /// Without this a sampled surface reads as its raw DCC-coded bytes: a flat field of
    /// the repeated key with the odd block of real payload, which is what Demon's Souls
    /// showed instead of Boletarian Palace.
    ///
    /// Caller holds the lock.
    /// </summary>
    private void MaterializeSampledDccClear(
        ResourceSlotIdentifier imageIdentifier, MetalCachedImage image, in ImageRequest request)
    {
        const byte DccClearToZero = 0x00;
        ref readonly var description = ref image.Description;
        if (description.Metadata.Kind != MetadataKind.Dcc || description.Metadata.Range.Size != 0 ||
            description.Resources.Levels != 1)
        {
            return;
        }

        var metadataAddress = description.Metadata.Range.Address;
        var sliceSize = description.DccSliceSize;
        var view = request.View;
        var layers = description.TransferLayers;
        for (var layer = view.BaseLayer; layer < view.BaseLayer + view.LayerCount && layer < layers; layer++)
        {
            // A tracked clear only counts when it was a clear to zero: every other code
            // needs the register colour, which a sampled binding does not have.
            var tracked = IsMetadataClearedLocked(metadataAddress, layer, out var trackedValue) &&
                (byte)trackedValue == DccClearToZero;
            GpuClearColor clear = default;
            var guestSlice = 0UL;
            if (!tracked)
            {
                // Metadata that recorded work is still writing is not settled enough to read.
                if (sliceSize == 0 ||
                    _owner.HasPendingGpuWrite(metadataAddress + sliceSize * layer, sliceSize) ||
                    !TryReadGuestDccClear(metadataAddress, sliceSize, layer, out guestSlice, out var code) ||
                    !PackedClearValue.TryDecodeDccColor(
                        view.Format, code * 0x01010101u, description.Metadata,
                        allowClearToRegister: false, out clear))
                {
                    continue;
                }

                // Nothing marks the guest bytes, so the claim is what keeps the next
                // sample from clearing over whatever is drawn into the surface in between.
                if (!TryClaimMaterializedDccLocked(imageIdentifier, guestSlice, sliceSize, code))
                {
                    Interlocked.Increment(ref _dccRepeatsSkipped);
                    continue;
                }
            }

            ClearImageRange(imageIdentifier, view.Format, new SubresourceRange(view.BaseLevel, 1, layer, 1), in clear);
            Interlocked.Increment(ref _sampledDccClears);
            if (tracked)
            {
                SetMetadataSliceLocked(metadataAddress, layer, false);
            }
        }
    }

    /// <summary>
    /// The DCC key of one layer, when every byte of that layer's metadata slice carries
    /// it. A slice that is not uniform is not a fast clear and is left alone.
    ///
    /// Caller holds the lock.
    /// </summary>
    private bool TryReadGuestDccClear(ulong address, ulong sliceSize, uint layer, out ulong guestSlice, out byte code)
    {
        guestSlice = 0;
        code = 0;
        const ulong MetadataBlockSize = 0x1000;
        if (sliceSize == 0 || sliceSize % MetadataBlockSize != 0 || address % MetadataBlockSize != 0 ||
            sliceSize > ulong.MaxValue / (layer + 1UL))
        {
            return false;
        }

        guestSlice = address + (sliceSize * layer);
        Span<byte> firstByte = stackalloc byte[1];
        if (!_backing.TryReadBacking(guestSlice, firstByte))
        {
            return false;
        }

        code = firstByte[0];
        var buffer = System.Buffers.ArrayPool<byte>.Shared.Rent(
            checked((int)Math.Min(sliceSize, 64 * 1024)));
        try
        {
            return IsUniformSlice(guestSlice, sliceSize, code, buffer);
        }
        finally
        {
            System.Buffers.ArrayPool<byte>.Shared.Return(buffer);
        }
    }

    private long _sampledDccClears;

    /// <summary>How many sampled surfaces had a pending DCC clear materialized.</summary>
    public long TakeSampledDccClears() => Interlocked.Exchange(ref _sampledDccClears, 0);

    /// <param name="deferRefresh">
    /// The caller begins a pass on the view before anything else uses the image, or drains
    /// the wait (MetalImageCache.DeferredRefresh): the upload of a CPU-dirty image is left
    /// to that pass.
    /// </param>
    public nint AcquireColorTargetView(
        ResourceSlotIdentifier imageIdentifier, in ImageRequest request, bool deferRefresh = false)
    {
        using var held = _lock.Hold();
        var image = _slots[imageIdentifier];
        DrainOverlappingDeferredRefreshesLocked(image);
        image.LastAccessTick = _collectionTick;
        PromoteLinearView(image);
        EnsureViewUsageLocked(image, in request.View);
        WatchImage(imageIdentifier);
        SynchronizeAliases(imageIdentifier, takesGpuOwnership: true);
        // The pass about to begin may clear or overwrite all of it; BeginRendering decides
        // whether the upload is worth making (MetalImageCache.DeferredRefresh).
        var defer = deferRefresh && CanDeferRefresh(image);
        if (!defer)
        {
            RefreshFromGuest(image);
        }

        // After the upload, which gives back the tiles of layers whose guest bytes are zero:
        // the pass writes these layers (a sparse image never waits for its pass to upload).
        MapSparseLayers(image, request.View.BaseLayer, request.View.LayerCount);
        image.Uses.RenderTarget = true;
        if (request.Description.Metadata.Kind == MetadataKind.Dcc && request.Description.Metadata.Range.Size == 0)
        {
            var changed = !MetalImageEpochs.SameMetadata(in image.Description.Metadata, in request.Description.Metadata);
            image.Description.Metadata = request.Description.Metadata;
            RegisterDccMetadata(request.Description);
            if (changed)
            {
                Epochs.NoteShapeChanged();
            }
        }

        if (defer)
        {
            image.NoteUse(_owner.CurrentSerial);
            var deferredView = image.GetOrCreateView(request.View);
            DeferRefreshLocked(deferredView, image, depth: false, stencilOnly: false);
            return deferredView;
        }

        image.MarkGpuModified();
        NoteLinearGpuWrite(image);
        image.GpuWriteSerial = _owner.CurrentSerial;
        image.NoteUse(image.GpuWriteSerial);
        return image.GetOrCreateView(request.View);
    }

    /// <param name="writes">
    /// False when the pass only tests against the target: no depth or stencil write and
    /// no load clear. Such a pass reads the image like a sampled texture, so it neither
    /// takes the bytes from the image's aliases nor leaves the image owing guest memory a
    /// writeback - which is what made SILENT HILL retile its 4K depth plane out and back
    /// through the R32 alias it samples for every soft-particle draw.
    /// </param>
    /// <param name="writesDepth">
    /// False when what the pass writes is the stencil plane alone. The depth plane then
    /// stays what it was, and so does every alias that holds a copy of it.
    /// </param>
    /// <param name="deferRefresh">As for <see cref="AcquireColorTargetView"/>.</param>
    public nint AcquireDepthTargetView(
        ResourceSlotIdentifier imageIdentifier,
        in ImageRequest request,
        bool writes = true,
        bool writesDepth = true,
        bool deferRefresh = false)
    {
        using var held = _lock.Hold();
        var image = _slots[imageIdentifier];
        DrainOverlappingDeferredRefreshesLocked(image);
        image.LastAccessTick = _collectionTick;
        PromoteLinearView(image);
        writes |= !ReadOnlyDepthPassesEnabled;

        writesDepth |= !ReadOnlyDepthPassesEnabled;
        var stencilOnly = writes && !writesDepth && image.Description.HasStencil;
        if (writes)
        {
            WatchImage(imageIdentifier);
        }

        var stencil = request.Description.HasStencil
            ? ImageStencilPlane(in image.Description, in request.Description)
            : default;
        SynchronizeAliases(imageIdentifier, takesGpuOwnership: writes, keepsDepthPlane: stencilOnly);
        // Only a target the pass writes can be one it clears. The refresh reads the
        // stencil plane from where the image has it, so a request that moves the plane
        // refreshes now, before the description changes.
        var defer = deferRefresh && writes && CanDeferRefresh(image) &&
            (!request.Description.HasStencil || stencil == image.Description.Stencil);
        if (!defer)
        {
            RefreshFromGuest(image);
        }

        // After the upload, as for a colour target.
        if (writes)
        {
            MapSparseLayers(image, request.View.BaseLayer, request.View.LayerCount);
        }

        image.Uses.DepthTarget = true;
        if (request.Description.HasMetadata)
        {
            var before = image.Description.Metadata;
            image.Description.Metadata = request.Description.Metadata;
            image.Description.Metadata.Range = ImageHtileRange(in image.Description, in request.Description);
            RegisterHtileMetadata(request.Description);
            if (!MetalImageEpochs.SameMetadata(in before, in image.Description.Metadata))
            {
                Epochs.NoteShapeChanged();
            }
        }

        if (request.Description.HasStencil)
        {
            image.Description.Stencil = stencil;
        }

        if (defer)
        {
            image.NoteUse(_owner.CurrentSerial);
            var deferredView = image.GetOrCreateView(request.View);
            DeferRefreshLocked(deferredView, image, depth: true, stencilOnly);
            return deferredView;
        }

        if (writes)
        {
            image.MarkGpuModified(depthPlaneChanged: !stencilOnly);
            NoteLinearGpuWrite(image);
            image.GpuWriteSerial = _owner.CurrentSerial;
        }

        image.NoteUse(writes ? image.GpuWriteSerial : _owner.CurrentSerial);
        return image.GetOrCreateView(request.View);
    }

    // The stencil plane of a depth image, from a depth target request that binds some of
    // its layers. DB_STENCIL_*_BASE always names slice 0 and DB_DEPTH_VIEW only picks the
    // slices a pass writes, so a request for slices 0..SLICE_MAX carries SLICE_MAX + 1
    // stencil slices: one padded 64 KiB-block S8 surface each, laid end to end from the
    // base. The image keeps the plane for every one of its layers - a cube bound one face
    // at a time still owns six stencil slices.
    internal static GuestSpan ImageStencilPlane(in ImageDescription image, in ImageDescription request) =>
        ImageLayerPlane(in image, in request, request.Stencil, "stencil");

    // The HTILE range of a depth image, the same way: DB_HTILE_DATA_BASE also names slice 0,
    // so a request for slices 0..SLICE_MAX carries SLICE_MAX + 1 HTILE slices, and the
    // image keeps the HTILE of every one of its layers.
    internal static GuestSpan ImageHtileRange(in ImageDescription image, in ImageDescription request) =>
        ImageLayerPlane(in image, in request, request.Metadata.Range, "HTILE");

    // A plane the request carries one slice per bound layer of, over every image layer:
    // the request's per-slice stride, rebased to the image's slice 0.
    private static GuestSpan ImageLayerPlane(
        in ImageDescription image, in ImageDescription request, GuestSpan plane, string name)
    {
        var imageLayers = image.Resources.Layers;
        var requestLayers = request.Resources.Layers;
        var imageSlice = imageLayers == 0 ? 0 : image.Data.Size / imageLayers;
        if (imageLayers == 0 || requestLayers == 0 || requestLayers > imageLayers ||
            image.Data.Size % imageLayers != 0 || plane.Size % requestLayers != 0 ||
            request.Data.Address < image.Data.Address || imageSlice == 0 ||
            (request.Data.Address - image.Data.Address) % imageSlice != 0)
        {
            throw GpuFatal.Fatal(
                $"A {name} request does not bind layers of its depth image: " +
                $"image=0x{image.Data.Address:X16}+0x{image.Data.Size:X} layers={imageLayers} " +
                $"request=0x{request.Data.Address:X16}+0x{request.Data.Size:X} layers={requestLayers} " +
                $"{name}=0x{plane.Address:X16}+0x{plane.Size:X}.");
        }

        var firstLayer = (request.Data.Address - image.Data.Address) / imageSlice;
        var planeSlice = plane.Size / requestLayers;
        if (firstLayer + requestLayers > imageLayers || plane.Address < planeSlice * firstLayer)
        {
            throw GpuFatal.Fatal(
                $"A {name} request reaches past its depth image: firstLayer={firstLayer} layers={requestLayers} " +
                $"imageLayers={imageLayers} {name}=0x{plane.Address:X16}+0x{plane.Size:X}.");
        }

        return new GuestSpan(plane.Address - (planeSlice * firstLayer), planeSlice * imageLayers);
    }

    // SHARPEMU_METAL_V2_READONLY_DEPTH=0 treats every depth attachment as written again.
    private static readonly bool ReadOnlyDepthPassesEnabled =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_READONLY_DEPTH") != "0";

    public void MarkGpuWritten(ResourceSlotIdentifier imageIdentifier)
    {
        using var held = _lock.Hold();
        if (_slots.TryGet(imageIdentifier) is not { } image)
        {
            return;
        }

        PromoteLinearView(image);
        EnsureSparseBacked(image);
        WatchImage(imageIdentifier);
        // A resolve or a depth copy writes one subresource, and taking GPU ownership
        // drops whatever the image still owes from guest memory and its aliases. Take
        // both first, as a colour target does, or every subresource the write leaves
        // alone keeps whatever the texture held before.
        SynchronizeAliases(imageIdentifier, takesGpuOwnership: true);
        RefreshFromGuest(image);
        image.MarkGpuModified();
        NoteLinearGpuWrite(image);
        image.GpuWriteSerial = _owner.CurrentSerial;
    }

    /// <summary>
    /// A blit or resolve is about to write into this image's texture: a window view would
    /// write the guest bytes with no record of it, so it gets an owned texture first, and a
    /// texture held from its BASE_LEVEL gets the whole chain the write names levels of.
    /// </summary>
    public void PrepareImageWrite(MetalCachedImage image)
    {
        if (!image.IsWindowLinearView && image.FirstLevel == 0 && !image.Shape.Lossy && image.Sparse is null)
        {
            return;
        }

        using var held = _lock.Hold();
        PromoteLinearView(image);
        EnsureSparseBacked(image);
    }

    public void RefreshImage(ResourceSlotIdentifier imageIdentifier)
    {
        using var held = _lock.Hold();
        DrainDeferredRefreshesLocked();
        if (_slots.TryGet(imageIdentifier) is { } image)
        {
            WatchImage(imageIdentifier);
            SynchronizeAliases(imageIdentifier, takesGpuOwnership: false);
            RefreshFromGuest(image);
        }
    }

    /// <summary>
    /// Writes every GPU-modified image that overlaps the range back into guest
    /// memory. Called before a buffer binding, an incompatible image request or a
    /// host read can observe the bytes.
    /// </summary>
    public void FlushGpuWrites(ulong address, ulong size, MetalWritebackCause cause)
    {
        using var phaseScope = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.GpuWriteFlush);
        if (!IsValidRange(address, size) || !_pageOwners.MayHaveOwners(address, size))
        {
            return;
        }

        using var held = _lock.Hold();
        foreach (var imageIdentifier in FindImagesInRange(address, size, pageOverlap: false))
        {
            if (_slots.TryGet(imageIdentifier) is { } image && image.GpuOverlaps(address, size))
            {
                WriteBackToGuest(image, cause);
            }
        }
    }

    /// <summary>A GPU shader or transfer wrote these guest bytes: the images reload.</summary>
    public void InvalidateMemoryFromGpu(ulong address, ulong size)
    {
        if (!IsValidRange(address, size))
        {
            return;
        }

        if (DccTraceActive && DccTraceAddress != 0 && DccTraceAddress >= address && DccTraceAddress - address < size &&
            Interlocked.Increment(ref _dccTraceLines) <= DccTraceMax)
        {
            Console.Error.WriteLine(
                $"[METAL-V2][DCC] gpu-write range=0x{address:X}+0x{size:X} claims={_materializedDccSlices.Count} " +
                $"deferred={_deferredDccSlices.Count} fills={_recordedDccFills.Count}");
        }

        // A GPU write into DCC metadata is the guest asking for, or replacing, a clear.
        if (_materializedDccSlices.Count != 0 || _deferredDccSlices.Count != 0 || _recordedDccFills.Count != 0)
        {
            using var rearmPhase = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.DccRearm);
            using var rearmHeld = _lock.Hold();
            ReArmMaterializedDccLocked(address, size);
            NoteRecordedDccStoreLocked(address, size);
        }

        if (_lossyTextures)
        {
            // Recorded with no image over the bytes too: a sampled image made over them
            // later holds what the GPU wrote, and is not made lossy (MetalImageCache.Lossy.cs).
            _gpuWrittenBytes.Mark(address, size);
        }

        if (!_pageOwners.MayHaveOwners(address, size))
        {
            return;
        }

        using var held = _lock.Hold();
        foreach (var imageIdentifier in FindImagesInRange(address, size, pageOverlap: true))
        {
            if (_slots.TryGet(imageIdentifier) is not { } image || !image.Overlaps(address, size))
            {
                continue;
            }

            if (image.Shape.Lossy && image.Texture != 0)
            {
                // It reloads what the GPU wrote, which it may not approximate. The new
                // texture is dirty whole, as the reload below has to be.
                ReshapeLevelsLocked(image, image.FirstLevel, dropLossy: true);
                _lossyGpuBytePromotions++;
            }

            MetalV2Diagnostics.WarnOnce(
                $"gpu-invalidate:{size:X}:{image.Description.Data.Size:X}",
                $"A GPU write to guest memory invalidated a cached image, so the image uploads again: " +
                $"write=0x{address:X16}+0x{size:X} image=0x{image.Description.Data.Address:X16}+0x{image.Description.Data.Size:X} " +
                $"extent={image.Description.Extent.Width}x{image.Description.Extent.Height} " +
                $"format={image.Description.PixelFormat} gpu_modified={image.IsGpuModified}.");
            image.ClearGpuModified();
            // Only the bytes the write names: a shader store into one corner of a surface
            // leaves the rest of it as good as it was, and a page-granular overlap is the
            // one case that cannot be narrowed, which FindImagesInRange already separates.
            var (start, end) = image.OverlapWithin(address, size);
            if (end > start)
            {
                image.MarkCpuDirty(MetalImageDirtyCause.GpuWrite, start, end);
            }
            else
            {
                image.MarkCpuDirty(MetalImageDirtyCause.GpuWrite);
            }

            UnwatchImage(imageIdentifier);
        }
    }

    /// <summary>
    /// Whether a live image covers these pages. Lock-free and never wrong in the
    /// direction that matters: a false answer means no image can own the bytes,
    /// which is what lets the command stream read a packet without the full test.
    /// </summary>
    public bool MayOwn(ulong address, ulong size) =>
        IsValidRange(address, size) && _pageOwners.MayHaveOwners(address, size);

    /// <summary>True while a GPU-modified image owns bytes in the range.</summary>
    public bool HasGpuModifiedImageBytes(ulong address, ulong size)
    {
        if (!IsValidRange(address, size) || !_pageOwners.MayHaveOwners(address, size))
        {
            return false;
        }

        using var held = _lock.Hold();
        foreach (var imageIdentifier in FindImagesInRange(address, size, pageOverlap: false))
        {
            if (_slots.TryGet(imageIdentifier) is { } image && image.GpuOverlaps(address, size))
            {
                return true;
            }
        }

        return false;
    }

    // Two surfaces over exactly the same guest bytes hand their contents over on the
    // device, with guest memory left out of it entirely. The pair that costs the most is
    // a depth-stencil target and the single-channel float alias something samples it
    // through: the round trip retiles the depth plane out to guest memory and detiles it
    // straight back, and the device can copy it instead. A 16-bit depth plane and its R16
    // alias go both ways, so the colour side can be the source and the depth the target.
    //
    // Guest memory is not updated, and does not need to be: the written image stays
    // GPU-modified, so the lazy writeback rules still publish it the moment the CPU, a
    // buffer binding or a differently shaped request wants those bytes.
    private bool TryPublishAliasOnDevice(MetalCachedImage image, MetalCachedImage other)
    {
        if (!DeviceAliasPublishEnabled || image.Texture == 0 || other.Texture == 0 ||
            image.IsWindowLinearView || other.IsWindowLinearView ||
            image.FirstLevel != 0 || other.FirstLevel != 0 ||
            image.Texture == other.Texture || other.IsCpuDirty ||
            image.Description.Data.Address != other.Description.Data.Address ||
            image.Description.Data.Size != other.Description.Data.Size)
        {
            return false;
        }

        // Nothing has touched either texture since the last copy between them: the alias
        // already holds what the copy would put there. A depth target that only tests
        // depth (soft particles) leaves its version alone, so every draw after the first
        // samples the alias without a copy - and without the copy ending the open pass.
        if (AliasReuseEnabled && image.StillHoldsPublicationFrom(other))
        {
            _deviceAliasReuses++;
            return true;
        }

        EnsureSparseBacked(image);
        var context = _owner.BeginTransfer(MetalTransferCause.Writeback, other.Texture, image.Texture);
        if (!_transfer.TryEncodeTextureToTexture(
                context, in other.Description, other.Texture, in image.Description, image.Texture))
        {
            return false;
        }

        if (context.WantsEncodedLabels)
        {
            context.ProfileEncoded(ProfileLabel("alias-on-device", in image.Description));
        }
        // The target now holds what the source holds, so it owes guest memory no upload.
        image.RefreshComplete();
        image.NotePublishedFrom(other);
        _deviceAliasPublishes++;
        return true;
    }

    private long _deviceAliasPublishes;
    private long _deviceAliasReuses;

    // SHARPEMU_METAL_V2_ALIAS_REUSE=0 copies the alias on every use again, for a comparison.
    private static readonly bool AliasReuseEnabled =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_ALIAS_REUSE") != "0";

    /// <summary>Alias publications whose copy the alias already held.</summary>
    public long TakeDeviceAliasReuses() => Interlocked.Exchange(ref _deviceAliasReuses, 0);

    /// <summary>Alias publications the device served without touching guest memory.</summary>
    public long TakeDeviceAliasPublishes() => Interlocked.Exchange(ref _deviceAliasPublishes, 0);

    // SHARPEMU_METAL_V2_ALIAS_ON_DEVICE=0 sends every alias through guest memory again.
    // The device copy only claims a 32-bit depth plane and an R32 colour alias, and a
    // 16-bit depth plane and an R16 colour alias either way round (switched off alone by
    // SHARPEMU_METAL_V2_D16_ALIAS_DEVICE=0), and a block-compressed surface and the integer
    // view of its blocks either way round (SHARPEMU_METAL_V2_BLOCK_ALIAS_DEVICE=0), with the
    // same tile mode, pitch and footprint
    // (MetalImageTransfer.TryEncodeTextureToTexture): neither transfer converts anything
    // for those pairs, so texel for texel the target gets exactly the bits the round trip
    // would have detiled into it.
    private static readonly bool DeviceAliasPublishEnabled =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_ALIAS_ON_DEVICE") != "0";

    // Publishes just the shared bytes of a GPU-written image, leaving the texture
    // authoritative for the rest of it. The image stays GPU-modified, so what is left is
    // still published when something asks for it; the published run is what stops the
    // same bytes being written again on every use of the pair.
    //
    // Returns why the image was written back whole instead, or None.
    private MetalAliasFallbackReason PublishAliasRange(MetalCachedImage other, ulong address, ulong size)
    {
        var (start, end) = other.OverlapWithin(address, size);
        if (!SubRangeTransfersEnabled || end <= start || other.Description.HasStencil)
        {
            WriteBackToGuest(other, MetalWritebackCause.ImageOverlap);
            return !SubRangeTransfersEnabled ? MetalAliasFallbackReason.SubRangeDisabled
                : end <= start ? MetalAliasFallbackReason.EmptyRun
                : MetalAliasFallbackReason.Stencil;
        }

        if (other.IsPublished(start, end))
        {
            return MetalAliasFallbackReason.None;
        }

        if (!TryWriteBackRun(other, start, end, MetalWritebackCause.ImageOverlap, out var failure))
        {
            WriteBackToGuest(other, MetalWritebackCause.ImageOverlap);
            return start == 0 && end >= other.Description.Data.Size
                ? MetalAliasFallbackReason.RunIsWholeOther
                : failure;
        }

        return MetalAliasFallbackReason.None;
    }

    // Retiles one run of a GPU-written image into guest memory. Unlike a whole writeback
    // this leaves the image GPU-modified: the texture still owns every byte outside the
    // run, and saying otherwise would lose them the next time anything read the image.
    private bool TryWriteBackRun(
        MetalCachedImage image, ulong start, ulong end, MetalWritebackCause cause,
        out MetalAliasFallbackReason failure)
    {
        using var censusScope = MetalGpuReferenceCensus.Scope(MetalRefKind.ImageWriteback);
        ref readonly var description = ref image.Description;
        if (image.Texture == 0 || ImageDescription.IsEmptyRange(description.Data))
        {
            failure = MetalAliasFallbackReason.NoTexture;
            return false;
        }

        if (!TryCoveringRun(in description, start, end, out var aligned, out var alignedEnd))
        {
            failure = MetalAliasFallbackReason.CoveringRunRefused;
            return false;
        }

        if (image.IsCpuDirty && aligned < image.DirtyEnd && alignedEnd > image.DirtyStart)
        {
            RefreshBeforeWriteback(image);
        }

        var destination = _guestMemory.ResolveWritable(
            description.Data.Address + aligned, alignedEnd - aligned);
        if (!destination.IsValid || destination.Size < alignedEnd - aligned)
        {
            failure = MetalAliasFallbackReason.RunUnmapped;
            return false;
        }

        _owner.LandHeldCopies(description.Data.Address + aligned, alignedEnd - aligned);
        var context = _owner.BeginTransfer(MetalTransferCause.Writeback, image.Texture);
        var range = new SubresourceRange(0, description.Resources.Levels, 0, description.TransferLayers);
        _transfer.EncodeTextureToGuest(
            context, in description, range, image.Texture, destination, default, aligned, alignedEnd - aligned);
        if (context.WantsEncodedLabels)
        {
            context.ProfileEncoded(ProfileLabel("writeback-run", in description));
        }
        image.MarkPublished(aligned, alignedEnd);
        NoteOwnGuestWrite(description.Data.Address + aligned, alignedEnd - aligned);
        _writebackCauses[(int)cause]++;
        _partialWritebacks++;
        _partialWritebackBytesSaved += (long)(description.Data.Size - (alignedEnd - aligned));
        Interlocked.Increment(ref _writebacks);
        failure = MetalAliasFallbackReason.None;
        return true;
    }

    // One line per distinct pair of aliasing surfaces, with the bytes each round trip
    // moves, so the cost can be attributed to the surfaces rather than to a counter.
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<
        (GpuFormat, uint, uint, GpuFormat, uint, uint), byte> _reportedAliasPairs = new();

    private static void ReportAliasPair(MetalCachedImage requested, MetalCachedImage overlapped)
    {
        ref readonly var wanted = ref requested.Description;
        ref readonly var held = ref overlapped.Description;
        // The report is once per pair of surface shapes, but it is asked for on every bind
        // that aliases; answer the repeats from a struct key so they build no strings.
        var pairKey = (wanted.PixelFormat, wanted.Extent.Width, wanted.Extent.Height,
            held.PixelFormat, held.Extent.Width, held.Extent.Height);
        if (_reportedAliasPairs.ContainsKey(pairKey) || !_reportedAliasPairs.TryAdd(pairKey, 0))
        {
            return;
        }

        var overlapStart = Math.Max(wanted.Data.Address, held.Data.Address);
        var overlapEnd = Math.Min(
            wanted.Data.Address + wanted.Data.Size, held.Data.Address + held.Data.Size);
        var shared = overlapEnd > overlapStart ? overlapEnd - overlapStart : 0;
        MetalV2Diagnostics.WarnOnce(
            $"alias-pair:{wanted.PixelFormat}:{wanted.Extent.Width}x{wanted.Extent.Height}:" +
            $"{held.PixelFormat}:{held.Extent.Width}x{held.Extent.Height}",
            "Two cached images share guest bytes, so every use of one publishes the other and " +
            "reloads itself whole: " +
            $"requested=<{AliasImageText(in wanted)}>, overlapped=<{AliasImageText(in held)}>, " +
            $"shared=0x{shared:X} of 0x{wanted.Data.Size:X}/0x{held.Data.Size:X}.");
    }

    private static bool SameExtent(in GpuExtent3D left, in GpuExtent3D right) =>
        left.Width == right.Width && left.Height == right.Height && left.Depth == right.Depth;

    // The cached image holds exactly the requested surface; only the view may differ.
    private static bool HasSameBacking(in ImageDescription cached, in ImageDescription requested, bool exactFormat)
    {
        if (cached.Data.Address != requested.Data.Address || cached.Data.Size != requested.Data.Size ||
            !SameExtent(cached.Extent, requested.Extent) || cached.Samples != requested.Samples ||
            cached.BytesPerBlock != requested.BytesPerBlock || cached.TileMode != requested.TileMode ||
            cached.Resources != requested.Resources ||
            !ViewFormatRules.AreCompatible(cached.PixelFormat, requested.PixelFormat))
        {
            return false;
        }

        if (cached.Type != requested.Type && !SameExtent(requested.Extent, new GpuExtent3D(1, 1, 1)))
        {
            return false;
        }

        return !exactFormat || cached.PixelFormat == requested.PixelFormat;
    }

    private readonly record struct OverlapResolution(ResourceSlotIdentifier Image, int Mip = -1, int Layer = -1);

    // Resolves a cached image that shares guest bytes with the request. Guest memory
    // is authoritative, so anything the cached image cannot serve is written back and
    // dropped instead of being copied image-to-image.
    private OverlapResolution ResolveOverlap(
        in ImageDescription requested,
        in ImageRequest request,
        ResourceSlotIdentifier cachedImageIdentifier,
        ResourceSlotIdentifier mergedImageIdentifier,
        bool exactFormat)
    {
        if (_slots.TryGet(cachedImageIdentifier) is not { } cached)
        {
            return new OverlapResolution(mergedImageIdentifier);
        }

        ref var cachedInfo = ref cached.Description;
        var role = request.Role;
        var safeToDelete = _collectionTick - Math.Min(_collectionTick, cached.LastAccessTick) > TicksBeforeRemoval;
        var requestedBlock = requested.BytesPerBlock * requested.Samples;
        var cachedBlock = cachedInfo.BytesPerBlock * cachedInfo.Samples;
        var requestedExtent = requested.BlockExtent;
        var cachedExtent = cachedInfo.BlockExtent;
        var sameSurface = requested.Data.Address == cachedInfo.Data.Address &&
            requestedExtent.Width == cachedExtent.Width && requestedExtent.Height == cachedExtent.Height &&
            requestedBlock == cachedBlock;

        // Same surface, reinterpreted: a compatible format binds through a view. The block
        // extent compares pitches, and a padded pitch hides the width: Black Myth: Wukong's
        // transient heap puts a 30x34 and a 60x34 RGBA16F target (both pitch 128) at one
        // address, and the 60x34 motion-blur tile target bound the 30x34 texture, so only
        // the left half of its tiles was ever written.
        if (sameSurface && !exactFormat && requested.Data.Size == cachedInfo.Data.Size &&
            SameExtent(requested.Extent, cachedInfo.Extent) &&
            requested.Resources == cachedInfo.Resources && requested.TileMode == cachedInfo.TileMode &&
            requested.Type == cachedInfo.Type && requested.IsBlock == cachedInfo.IsBlock &&
            ViewFormatRules.AreCompatible(cachedInfo.PixelFormat, requested.PixelFormat))
        {
            return new OverlapResolution(mergedImageIdentifier.IsValid ? mergedImageIdentifier : cachedImageIdentifier);
        }

        // One layer or mip of the cached image binds as a view of it, at its base address
        // too: layer 0 of an array is the address the array starts at. Kept as an alias
        // instead, Demon's Souls' 3840x2160 four-layer R32Uint G-buffer and the 2D target
        // of its first layer took ownership from each other on every draw - a whole
        // writeback and upload of both and a broken render pass, 19 times a frame.
        var requestedMip = requested.FindMatchingMipLevel(cachedInfo);
        if (requestedMip >= 0)
        {
            var layer = requested.FindMatchingArraySlice(cachedInfo, requestedMip);
            if (layer >= 0 && (!exactFormat || cachedInfo.PixelFormat == requested.PixelFormat))
            {
                return new OverlapResolution(cachedImageIdentifier, requestedMip, layer);
            }
        }

        // The cached image is a piece of the requested one: its GPU contents go back to
        // guest memory, from where the new, larger image uploads them.
        if (cachedInfo.FindMatchingMipLevel(requested) >= 0)
        {
            ReportRelease(MetalImageReleaseCause.SubsetOfLarger, cachedInfo, requested, default);
            ReleaseImage(cachedImageIdentifier, MetalImageReleaseCause.SubsetOfLarger);
            return new OverlapResolution(mergedImageIdentifier);
        }

        if (sameSurface && !exactFormat)
        {
            // A depth surface sampled as its colour twin binds the depth texture itself
            // (MetalImageCache.DepthTwin.cs), and a colour twin that only served samples
            // gives way to the depth target that now owns the bytes.
            if (ServesAsDepthTwin(cached, in requested, in request))
            {
                _depthTwinBinds++;
                return new OverlapResolution(mergedImageIdentifier.IsValid ? mergedImageIdentifier : cachedImageIdentifier);
            }

            if (IsReleasableColorTwin(cached, in requested, role))
            {
                _depthTwinReleases++;
                ReportRelease(MetalImageReleaseCause.SameAddressReplaced, cachedInfo, requested, default);
                ReleaseImage(cachedImageIdentifier, MetalImageReleaseCause.SameAddressReplaced);
                return new OverlapResolution(mergedImageIdentifier);
            }
        }

        if (sameSurface)
        {
            // Anything else at the same address needs its own texture. A surface the
            // title is still using both ways keeps both, because dropping one of a
            // pair it alternates between every frame recreates a texture and uploads
            // it again each time: Silent Hill's 3840x2160 depth buffer, written as
            // D32S8 and then sampled as R32F, paid three transfers of it per frame.
            // Aliases stay coherent through SynchronizeAliases on acquisition.
            if (!safeToDelete)
            {
                _aliasesKept++;
                return new OverlapResolution(mergedImageIdentifier);
            }

            ReportRelease(MetalImageReleaseCause.SameAddressReplaced, cachedInfo, requested, default);
            ReleaseImage(cachedImageIdentifier, MetalImageReleaseCause.SameAddressReplaced);
            return new OverlapResolution(mergedImageIdentifier);
        }

        _ = role;
        if (requested.Data.Address >= cachedInfo.Data.Address && safeToDelete)
        {
            ReportRelease(MetalImageReleaseCause.OverlapEvicted, cachedInfo, requested, default);
            ReleaseImage(cachedImageIdentifier, MetalImageReleaseCause.OverlapEvicted);
        }

        return new OverlapResolution(mergedImageIdentifier);
    }

    private void ValidateRequest(in ImageRequest request)
    {
        request.Description.Validate();
        ref readonly var description = ref request.Description;
        var view = request.View;
        if (view.Format == GpuFormat.Undefined || view.LevelCount == 0 || view.LayerCount == 0 ||
            view.BaseLevel >= description.Resources.Levels ||
            view.LevelCount > description.Resources.Levels - view.BaseLevel ||
            (!description.IsVolume && (view.BaseLayer >= description.Resources.Layers ||
                view.LayerCount > description.Resources.Layers - view.BaseLayer)))
        {
            throw GpuFatal.Fatal(
                $"The image view description is invalid: format={(int)view.Format} mip={view.BaseLevel}+{view.LevelCount} " +
                $"layer={view.BaseLayer}+{view.LayerCount} levels={description.Resources.Levels} layers={description.Resources.Layers}.");
        }
    }

    // A descriptor with no memory still needs something to bind; one 1x1 image per format serves.
    private ResourceSlotIdentifier GetNullImage(in ImageRequest request)
    {
        var format = request.Description.PixelFormat;
        if (_nullImages.TryGetValue(format, out var found) && _slots.TryGet(found) is not null)
        {
            return found;
        }

        var description = ImageDescription.Create();
        description.PixelFormat = format;
        description.GuestFormat = request.Description.GuestFormat;
        description.Type = GuestImageType.Color2D;
        description.Extent = new GpuExtent3D(1, 1, 1);
        description.Resources = SubresourceCount.Single;
        description.Pitch = 1;
        description.BytesPerBlock = Math.Max(request.Description.BytesPerBlock, 1);
        description.Samples = 1;
        description.TileMode = GuestTileMode.Linear;
        description.MipLayout[0] = new MipLevelLayout { Offset = 0, Size = description.BytesPerBlock, Pitch = 1, Height = 1 };
        var imageIdentifier = InsertImage(description);
        _slots[imageIdentifier].RefreshComplete();
        _nullImages[format] = imageIdentifier;
        return imageIdentifier;
    }

    // firstLevel above zero makes a sampled texture of the guest levels from there down
    // (MetalImageCache.BaseLevel.cs).
    private ResourceSlotIdentifier InsertImage(
        in ImageDescription description, bool sampledOnly = false, ImageRole role = ImageRole.Texture,
        uint firstLevel = 0, bool lossyAllowed = false)
    {
        var admitted = Admit(in description);
        var shape = firstLevel != 0 ? _transfer.ShapeFor(in admitted, firstLevel) : CreationShape(in admitted, role);
        nint viewTexture = 0;
        var windowView = sampledOnly && firstLevel == 0 && TryCreateLinearView(in admitted, ref shape, out viewTexture);
        if (!windowView && sampledOnly && lossyAllowed)
        {
            shape = LossyShapeFor(in admitted, shape);
        }

        MetalSparseResidency? sparse = null;
        var texture = windowView ? viewTexture : CreateImageTexture(in shape, in admitted, out sparse);
        // Stamped before the collection below runs: an image that has never been used
        // reads as the oldest one in the cache, so an over-budget collection evicted the
        // image it was making room for and the caller read a freed slot.
        var image = new MetalCachedImage(in admitted, in shape, texture)
        {
            LastAccessTick = _collectionTick,
            NeededWindowFrame = _idleClock.Frames,
            LevelsChangedFrame = _idleClock.Frames,
            Sparse = sparse,
        };
        if (windowView)
        {
            image.MarkWindowLinearView();
            Interlocked.Increment(ref _linearViews);
        }

        if (firstLevel != 0)
        {
            var (start, end) = _transfer.LevelBytes(in admitted, firstLevel);
            image.SetResidentLevels(firstLevel, start, end);
            _baseLevelCreated++;
        }

        image.AllocatedBytes = TextureAllocatedBytes(image);
        NoteRecreatedLocked(in admitted);

        image.Epochs = Epochs;
        var imageIdentifier = _slots.Insert(image);
        Interlocked.Increment(ref _imageSetVersion);
        Epochs.NoteBindingChanged();
        if (!ImageDescription.IsEmptyRange(description.Data))
        {
            AddToIndex(imageIdentifier);
        }

        Interlocked.Increment(ref _createdImages);
        CollectIfOverBudget();
        return imageIdentifier;
    }

    // Minimum linear texture alignment of a pixel format, asked of the device once.
    private ulong LinearAlignment(ulong pixelFormat)
    {
        if (!_linearAlignments.TryGetValue(pixelFormat, out var alignment))
        {
            alignment = (ulong)MetalV2Native.SendWithValue(_owner.Device, SelMinimumLinearAlignment, (nuint)pixelFormat);
            _linearAlignments[pixelFormat] = alignment;
        }

        return alignment;
    }

    // A sampled-only linear image whose bytes sit in one guest window becomes a texture over
    // those bytes. Caller holds the lock. False leaves the image to the owned-texture path.
    private bool TryCreateLinearView(in ImageDescription description, ref MetalTextureShape shape, out nint texture)
    {
        texture = 0;
        if (!MetalLinearWindowView.Enabled || _watchingStopped || description.Data.Size == 0 ||
            description.TileMode != GuestTileMode.Linear || description.Type != GuestImageType.Color2D)
        {
            return false;
        }

        // An image over bytes another image also covers would need the alias bookkeeping
        // that owned textures carry; only an image alone on its pages is worth the view.
        if (_pageOwners.MayHaveOwners(description.Data.Address, description.Data.Size) &&
            FindImagesInRange(description.Data.Address, description.Data.Size, pageOverlap: true).Count != 0)
        {
            return false;
        }

        if (description.IsBlock || !MetalLinearWindowView.SupportsLinearFormat(shape.PixelFormat) ||
            !_guestMemory.TryResolveInPlace(description.Data.Address, description.Data.Size, out var range) ||
            !MetalLinearWindowView.Qualifies(
                in description, in shape, range.Offset, LinearAlignment(shape.PixelFormat), out var bytesPerRow))
        {
            return false;
        }

        // ShaderRead alone, plus the pixel-format view usage a view of another format needs.
        var viewShape = shape with { Usage = (ulong)(MtlTextureUsage.ShaderRead | MtlTextureUsage.PixelFormatView) };
        var descriptor = MetalNative.Send(
            MetalNative.Send(MetalNative.Class("MTLTextureDescriptor"), SelAlloc), SelInit);
        if (descriptor == 0)
        {
            return false;
        }

        try
        {
            MetalV2Native.SendVoidValue(descriptor, SelSetTextureType, (nuint)viewShape.TextureType);
            MetalV2Native.SendVoidValue(descriptor, SelSetPixelFormat, (nuint)viewShape.PixelFormat);
            MetalV2Native.SendVoidValue(descriptor, SelSetWidth, viewShape.Width);
            MetalV2Native.SendVoidValue(descriptor, SelSetHeight, viewShape.Height);
            MetalV2Native.SendVoidValue(descriptor, SelSetUsage, (nuint)viewShape.Usage);
            // The window buffer is shared; a texture over it has to be too.
            MetalV2Native.SendVoidValue(descriptor, SelSetStorageMode, Mtl.StorageModeShared);
            texture = MetalV2Native.SendNewBufferTexture(
                range.Buffer, SelNewBufferTexture, descriptor, (nuint)range.Offset, (nuint)bytesPerRow);
        }
        finally
        {
            MetalNative.SendVoid(descriptor, SelRelease);
        }

        if (texture == 0)
        {
            return false;
        }

        shape = viewShape;
        return true;
    }

    /// <summary>
    /// Gives a window view an owned texture, to be filled from guest memory like any other
    /// image. The old texture waits for the submissions that bound it. Caller holds the lock.
    ///
    /// Every path that writes an image or binds it as anything but a sampled texture comes
    /// here first, so a texture held from its BASE_LEVEL gets the whole chain here too: a
    /// write may name any level, and a GPU-written image is retiled back whole.
    /// </summary>
    private void PromoteLinearView(MetalCachedImage image)
    {
        if (image.FirstLevel != 0)
        {
            ReshapeLevelsLocked(image, 0, dropLossy: true);
            _baseLevelPromotions++;
            return;
        }

        // A lossy texture is sampled-only; whatever else is about to use the image needs a
        // texture that can be written and kept exactly.
        if (image.Shape.Lossy)
        {
            ReshapeLevelsLocked(image, 0, dropLossy: true);
            return;
        }

        if (!image.IsWindowLinearView)
        {
            return;
        }

        var shape = _transfer.ShapeFor(image.Description);
        var texture = CreateTexture(in shape, image.Description);
        RetireImage(image.PromoteToOwned(texture, in shape));
        RecountAllocatedLocked(image);
        _releaseCauses[(int)MetalImageReleaseCause.LinearViewPromoted]++;
        Interlocked.Decrement(ref _linearViews);
        Interlocked.Increment(ref _linearViewPromotions);
    }

    // An image format the transfer cannot carry would be a fatal error there, and a
    // texture is never worth stopping the title for: the description takes the
    // size-equivalent format instead, and the substitution is logged once.
    private static ImageDescription Admit(in ImageDescription description)
    {
        if (MetalV2Formats.IsSupported(description.PixelFormat))
        {
            return description;
        }

        var substitute = MetalV2Formats.SubstituteFor(description.BytesPerBlock);
        MetalV2Diagnostics.WarnOnce(
            $"admit:{(int)description.PixelFormat}",
            $"No Metal format carries this image; using the size-equivalent one: " +
            $"format={(int)description.PixelFormat} guestFormat={(uint)description.GuestFormat} " +
            $"bytesPerBlock={description.BytesPerBlock} substitute={(int)substitute}.");
        var admitted = description;
        admitted.PixelFormat = substitute;
        return admitted;
    }

    private nint CreateTexture(in MetalTextureShape shape, in ImageDescription description)
    {
        var descriptor = NewTextureDescriptor(in shape);
        try
        {
            var texture = MetalNative.Send(_owner.Device, SelNewTexture, descriptor);
            if (texture == 0)
            {
                throw GpuFatal.Fatal(
                    $"A Metal texture could not be created: format={shape.PixelFormat} type={shape.TextureType} " +
                    $"extent={shape.Width}x{shape.Height}x{shape.Depth} mips={shape.MipLevels} layers={shape.ArrayLength} " +
                    $"samples={shape.SampleCount} usage=0x{shape.Usage:X} address=0x{description.Data.Address:X16}.");
            }

            return texture;
        }
        finally
        {
            MetalNative.SendVoid(descriptor, SelRelease);
        }
    }

    // A private-storage texture descriptor for the shape; the caller releases it.
    private static nint NewTextureDescriptor(in MetalTextureShape shape)
    {
        var descriptor = MetalNative.Send(
            MetalNative.Send(MetalNative.Class("MTLTextureDescriptor"), SelAlloc), SelInit);
        if (descriptor == 0)
        {
            throw GpuFatal.Fatal("A Metal texture descriptor could not be created.");
        }

        MetalV2Native.SendVoidValue(descriptor, SelSetTextureType, (nuint)shape.TextureType);
        MetalV2Native.SendVoidValue(descriptor, SelSetPixelFormat, (nuint)shape.PixelFormat);
        MetalV2Native.SendVoidValue(descriptor, SelSetWidth, Math.Max(shape.Width, 1));
        MetalV2Native.SendVoidValue(descriptor, SelSetHeight, Math.Max(shape.Height, 1));
        MetalV2Native.SendVoidValue(descriptor, SelSetDepth, Math.Max(shape.Depth, 1));
        MetalV2Native.SendVoidValue(descriptor, SelSetMipmapLevelCount, Math.Max(shape.MipLevels, 1));
        MetalV2Native.SendVoidValue(descriptor, SelSetArrayLength, Math.Max(shape.ArrayLength, 1));
        MetalV2Native.SendVoidValue(descriptor, SelSetSampleCount, Math.Max(shape.SampleCount, 1));
        MetalV2Native.SendVoidValue(descriptor, SelSetUsage, (nuint)shape.Usage);
        MetalV2Native.SendVoidValue(descriptor, SelSetStorageMode, Mtl.StorageModePrivate);
        // Metal asserts on a lossy descriptor with a usage that cannot be compressed, so
        // the usage decides here whatever the shape asks for.
        if (shape.Lossy && (shape.Usage & CompressionBreakingUsage) == 0)
        {
            MetalV2Native.SendVoidValue(descriptor, SelSetCompressionType, CompressionTypeLossy);
        }

        return descriptor;
    }

    // How much of an image one transfer may map and stage at a time. A guest-memory
    // window is 4 GiB, but the staging a tiled transfer asks for is as large as the piece
    // it is given, so a very large image is moved a few layers at a time rather than
    // asking the ring for hundreds of megabytes in one submission.
    private const ulong MaxTransferPieceBytes = 32UL * 1024 * 1024;

    // The pieces an image is transferred in: one piece for anything that maps whole, and
    // otherwise runs of whole layers, which is the only split whose guest bytes are
    // contiguous and whose subresources are expressible as a SubresourceRange. An image
    // that cannot be split that way stays a single piece and reports itself if it fails.
    private static bool TrySplitLayers(in ImageDescription description, out uint layers, out ulong layerBytes)
    {
        layers = description.TransferLayers;
        layerBytes = 0;
        if (layers <= 1 || description.Resources.Levels != 1 || description.IsVolume ||
            description.Data.Size % layers != 0)
        {
            return false;
        }

        layerBytes = description.Data.Size / layers;
        return layerBytes != 0;
    }

    // Uploads the guest bytes of a CPU-dirty image through the tiler.
    private void RefreshFromGuest(MetalCachedImage image)
    {
        using var censusScope = MetalGpuReferenceCensus.Scope(MetalRefKind.ImageUpload);
        if (!image.IsCpuDirty || image.Texture == 0 || ImageDescription.IsEmptyRange(image.Description.Data))
        {
            return;
        }

        ref readonly var description = ref image.Description;
        // A texture held from its BASE_LEVEL reads only the coarser levels' bytes; guest
        // memory changing anywhere else changes nothing it holds. (An empty interval is a
        // new image, dirty over all of it.)
        var residentStart = image.ResidentStart;
        var residentEnd = image.ResidentEnd;
        if (image.FirstLevel != 0 && image.DirtyEnd > image.DirtyStart &&
            (image.DirtyEnd <= residentStart || image.DirtyStart >= residentEnd))
        {
            image.RefreshComplete();
            return;
        }

        if (TryRefreshDirtyRun(image))
        {
            return;
        }

        if (image.IsGpuModified)
        {
            // Stale over a run an alias took while the texture still owns the rest: a
            // whole upload would replace bytes guest memory was never given. The run
            // stays stale rather than costing them.
            MetalV2Diagnostics.InfoOnce(
                $"alias-run-refresh:{description.PixelFormat}:{description.Extent.Width}x{description.Extent.Height}",
                "A GPU-written image stale over an alias run could not reload just that run, so it keeps " +
                $"its texture there: {description.PixelFormat} {description.Extent.Width}x{description.Extent.Height} " +
                $"0x{description.Data.Address:X16}+0x{description.Data.Size:X} " +
                $"dirty=0x{image.DirtyStart:X}..0x{image.DirtyEnd:X}.");
            return;
        }

        PrepareSparseUpload(image, residentStart, residentEnd);
        _owner.LandHeldCopies(description.Data.Address + residentStart, residentEnd - residentStart);
        var context = _owner.BeginTransfer(MetalTransferCause.Upload, image.Texture);
        var residentSize = residentEnd - residentStart;
        var whole = residentSize <= MaxTransferPieceBytes
            ? _guestMemory.Resolve(description.Data.Address + residentStart, residentSize)
            : default;
        if (whole.IsValid && whole.Size >= residentSize)
        {
            var stencil = description.HasStencil
                ? _guestMemory.Resolve(description.Stencil.Address, description.Stencil.Size)
                : default;
            _transfer.EncodeGuestToTexture(
                context, in description, ResidentRange(image), whole, stencil, image.Texture,
                residentStart, residentSize, image.FirstLevel);
        }
        else if (!UploadInPieces(context, image))
        {
            // Nothing reached the texture, so it still disagrees with guest memory: the
            // image stays CPU-dirty and the next use tries again, rather than sampling
            // whatever the texture held as though it were the guest's bytes.
            return;
        }

        _uploadCauses[(int)image.DirtyCause]++;
        if (context.WantsEncodedLabels)
        {
            context.ProfileEncoded(ProfileLabel("upload", in description));
        }
        image.RefreshComplete();
        Interlocked.Increment(ref _uploads);
    }

    // A surface whose dirty bytes are a small part of it is refreshed over just those
    // bytes: the run is widened to where the transfer plan says both its ends fall on
    // block-row boundaries, so every subresource it carries is carried whole. False when
    // the run is most of the image, cannot be aligned or cannot be mapped - the whole
    // image is then refreshed as before, because a partial refresh that loses bytes would
    // leave the texture disagreeing with guest memory for good.
    private bool TryRefreshDirtyRun(MetalCachedImage image)
    {
        ref readonly var description = ref image.Description;
        // Only what the texture's levels read counts: the whole image, or for a texture
        // held from its BASE_LEVEL the coarser levels' run.
        var residentStart = image.ResidentStart;
        var residentEnd = image.ResidentEnd;
        var dirtyStart = Math.Max(image.DirtyStart, residentStart);
        var dirtyEnd = Math.Min(image.DirtyEnd, residentEnd);
        if (!SubRangeTransfersEnabled || description.HasStencil ||
            dirtyEnd <= dirtyStart ||
            dirtyEnd - dirtyStart > (residentEnd - residentStart) / 2)
        {
            return false;
        }

        if (!TryCoveringRun(in description, dirtyStart, dirtyEnd, out var start, out var end))
        {
            return false;
        }

        // The run's ends widen outward to cuts of the whole chain; the resident run's own
        // ends are where its levels' transfers start and stop, so cutting there loses none.
        start = Math.Max(start, residentStart);
        end = Math.Min(end, residentEnd);
        var source = _guestMemory.Resolve(description.Data.Address + start, end - start);
        if (!source.IsValid || source.Size < end - start)
        {
            return false;
        }

        PrepareSparseUpload(image, start, end);
        _owner.LandHeldCopies(description.Data.Address + start, end - start);
        var context = _owner.BeginTransfer(MetalTransferCause.Upload, image.Texture);
        _transfer.EncodeGuestToTexture(
            context, in description, ResidentRange(image), source, default, image.Texture, start, end - start,
            image.FirstLevel);
        _uploadCauses[(int)image.DirtyCause]++;
        _partialUploads++;
        _partialUploadBytesSaved += (long)(residentEnd - residentStart - (end - start));
        if (context.WantsEncodedLabels)
        {
            context.ProfileEncoded(ProfileLabel("upload-run", in description));
        }

        if (image.FirstLevel != 0)
        {
            // What is still dirty lies outside the levels the texture holds.
            image.RefreshComplete();
        }
        else
        {
            image.RefreshComplete(start, end);
        }

        Interlocked.Increment(ref _uploads);
        return true;
    }

    // The run a restricted transfer may be given for a dirty range. The plan's own
    // alignments move *into* a range - a start forward, an end back - because they exist
    // to cut an image into runs that together cover it, where a byte one run drops the
    // next one picks up. A single run has no next one, so here each end is moved *out*
    // instead: the start back and the end forward, to the nearest cuts no part of the
    // transfer plan straddles - a block row of a tiled surface, a texel row of a linear
    // one, a layer of a depth plane. Each cut is confirmed by the opposite alignment
    // standing still on it, and an end that cannot be confirmed reaches the image's end
    // or its first byte instead. A run that widens to the whole image is refused, and
    // the caller transfers the whole image as before: bytes a run dropped would stay
    // stale, and a texture that disagrees with guest memory never catches up.
    private bool TryCoveringRun(
        in ImageDescription description, ulong dirtyStart, ulong dirtyEnd, out ulong start, out ulong end) =>
        TryCoveringRun(_transfer, in description, dirtyStart, dirtyEnd, out start, out end);

    internal static bool TryCoveringRun(
        IMetalImageTransfer transfer, in ImageDescription description,
        ulong dirtyStart, ulong dirtyEnd, out ulong start, out ulong end)
    {
        // A colour surface begins a run at the cut it is dirty from: the texel row of a
        // linear one, the block row of a tiled one - a tile row of a 4 KiB mode, a macro
        // tile row of a 64 KiB one. The depth kernels have no device test for a run past
        // the first byte, so a depth run still starts at the image's first byte.
        start = description.IsDepth
            ? 0
            : transfer.AlignRunEnd(in description, 0, Math.Min(dirtyStart, dirtyEnd));
        if (transfer.AlignRunStart(in description, start) != start)
        {
            // Not a boundary a run may begin at after all; from the first byte, which is.
            start = 0;
        }

        // The first cut at or after the dirty end, confirmed as one a run may end at.
        end = Math.Min(transfer.AlignRunStart(in description, dirtyEnd), description.Data.Size);
        if (end < dirtyEnd || transfer.AlignRunEnd(in description, start, end) != end)
        {
            // Nothing between the dirty end and the image's end can be dropped, so a run
            // that reaches the end covers whatever the alignment would have cut.
            end = description.Data.Size;
        }

        return end > start && end - start < description.Data.Size;
    }

    private long _partialUploads;
    private long _partialUploadBytesSaved;
    private long _partialWritebacks;
    private long _partialWritebackBytesSaved;

    /// <summary>What the sub-range transfers have saved, for the periodic report.</summary>
    public (long Uploads, long UploadBytes, long Writebacks, long WritebackBytes) TakePartialTransfers() =>
        (Interlocked.Exchange(ref _partialUploads, 0),
         Interlocked.Exchange(ref _partialUploadBytesSaved, 0),
         Interlocked.Exchange(ref _partialWritebacks, 0),
         Interlocked.Exchange(ref _partialWritebackBytesSaved, 0));

    // SHARPEMU_METAL_V2_SUBRANGE=0 refreshes and publishes whole images again, for a
    // comparison or a bisect.
    private static readonly bool SubRangeTransfersEnabled =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_SUBRANGE") != "0";

    // Moves an image that is too large to map or stage whole, a run of layers at a time.
    // Layers that are not one window-resident span are cut into the runs that are, so an
    // image whose guest range crosses a window edge still lands whole.
    private bool UploadInPieces(IMetalSubmissionContext context, MetalCachedImage image)
    {
        ref readonly var description = ref image.Description;
        if (!TrySplitLayers(in description, out var layers, out var layerBytes))
        {
            // One window, never the ring: a range this large is past what the ring
            // stages, and staging it would carry only its prefix. A texture held from its
            // BASE_LEVEL maps the coarser levels' run alone (a split image has one level,
            // so the layer pieces below never have a first level to keep).
            var residentStart = image.ResidentStart;
            var residentSize = image.ResidentEnd - residentStart;
            var source = _guestMemory.ResolveWritable(description.Data.Address + residentStart, residentSize);
            if (!source.IsValid)
            {
                return UploadInRuns(context, image, residentStart, image.ResidentEnd);
            }

            var stencil = description.HasStencil
                ? _guestMemory.Resolve(description.Stencil.Address, description.Stencil.Size)
                : default;
            _transfer.EncodeGuestToTexture(
                context, in description, ResidentRange(image), source, stencil, image.Texture,
                residentStart, residentSize, image.FirstLevel);
            return true;
        }

        var batch = (uint)Math.Clamp(MaxTransferPieceBytes / layerBytes, 1, layers);
        var stencilLayerBytes = description.HasStencil && layers != 0 ? description.Stencil.Size / layers : 0;
        MetalV2Diagnostics.InfoOnce(
            "image-pieces",
            $"An image is transferred in pieces because it does not map or stage whole: " +
            $"address=0x{description.Data.Address:X16} size=0x{description.Data.Size:X} layers={layers} " +
            $"batch={batch} ({_guestMemory.DescribeResolve(description.Data.Address, description.Data.Size)}).");
        var moved = false;
        for (uint first = 0; first < layers; first += batch)
        {
            var count = Math.Min(batch, layers - first);
            var pieceBase = layerBytes * first;
            var pieceSize = layerBytes * count;
            // A piece past the run size - a single layer that large - is past what the
            // ring stages whole, so only a window may carry it in one transfer.
            var piece = pieceSize <= MaxTransferPieceBytes
                ? _guestMemory.Resolve(description.Data.Address + pieceBase, pieceSize)
                : _guestMemory.ResolveWritable(description.Data.Address + pieceBase, pieceSize);
            if (!piece.IsValid || piece.Size < pieceSize)
            {
                // These layers are not one span a transfer can read; cut them into the
                // runs that are rather than leaving the whole piece stale.
                moved |= UploadInRuns(context, image, pieceBase, pieceBase + pieceSize);
                continue;
            }

            var stencilPiece = description.HasStencil && stencilLayerBytes != 0
                ? _guestMemory.Resolve(
                    description.Stencil.Address + (stencilLayerBytes * first), stencilLayerBytes * count)
                : default;
            var range = new SubresourceRange(0, description.Resources.Levels, first, count);
            _transfer.EncodeGuestToTexture(
                context, in description, range, piece, stencilPiece, image.Texture, pieceBase, pieceSize);
            moved = true;
        }

        return moved;
    }

    // A GPU-written image stale over a run an alias took from it holds old bytes there,
    // and a writeback that carries them would overwrite whatever the alias has published
    // since. The run is reloaded from guest memory first, so the writeback puts back
    // what guest memory already holds. It stays stale afterwards: the alias may not have
    // published yet, and when it does this image has to reload the run again.
    private void RefreshBeforeWriteback(MetalCachedImage image)
    {
        if (!image.IsCpuDirty)
        {
            return;
        }

        var (cause, start, end) = (image.DirtyCause, image.DirtyStart, image.DirtyEnd);
        RefreshFromGuest(image);
        image.MarkCpuDirty(cause, start, end);
    }

    // Retiles a GPU-written image back into guest memory; afterwards guest memory and
    // the texture agree, so the image stays cached and clean.
    private void WriteBackToGuest(MetalCachedImage image, MetalWritebackCause cause)
    {
        using var censusScope = MetalGpuReferenceCensus.Scope(MetalRefKind.ImageWriteback);
        if (!image.IsGpuModified || image.Texture == 0 || ImageDescription.IsEmptyRange(image.Description.Data))
        {
            image.ClearGpuModified();
            return;
        }

        RefreshBeforeWriteback(image);
        ref readonly var description = ref image.Description;
        _owner.LandHeldCopies(description.Data.Address, description.Data.Size);
        var destination = description.Data.Size <= MaxTransferPieceBytes
            ? _guestMemory.ResolveWritable(description.Data.Address, description.Data.Size)
            : default;
        var context = _owner.BeginTransfer(MetalTransferCause.Writeback, image.Texture);
        if (destination.IsValid && destination.Size >= description.Data.Size)
        {
            var stencil = description.HasStencil
                ? _guestMemory.ResolveWritable(description.Stencil.Address, description.Stencil.Size)
                : default;
            var range = new SubresourceRange(0, description.Resources.Levels, 0, description.TransferLayers);
            _transfer.EncodeTextureToGuest(
                context, in description, range, image.Texture, destination, stencil, 0, description.Data.Size);
        }
        else if (!WriteBackInPieces(context, image))
        {
            return;
        }

        if (context.WantsEncodedLabels)
        {
            context.ProfileEncoded(ProfileLabel("writeback", in description));
        }
        image.ClearGpuModified();
        // The retile is a GPU store into guest memory: until it completes the bytes
        // in guest memory are the old ones, so a read of them has to wait for it.
        NoteOwnGuestWrite(description.Data.Address, description.Data.Size);
        // And a guest store into them waits for it too (MetalImageCache.WritebackFence.cs).
        FenceWriteback(description.Data.Address, description.Data.Size, context.SubmissionSerial, cause);
        if (description.HasStencil)
        {
            // Noted whether or not every piece of the plane found a window: the hazard is
            // that a read sees stale bytes, and over-noting only costs a wait.
            NoteOwnGuestWrite(description.Stencil.Address, description.Stencil.Size);
            FenceWriteback(description.Stencil.Address, description.Stencil.Size, context.SubmissionSerial, cause);
        }

        _writebackCauses[(int)cause]++;
        Interlocked.Increment(ref _writebacks);
    }

    // The writeback counterpart of UploadInPieces: a run of layers at a time, writing
    // only through windows, because a retile stores into guest memory and the upload ring
    // is nobody's guest memory. A run with no writable window keeps the texture
    // authoritative for those layers and says so.
    private bool WriteBackInPieces(IMetalSubmissionContext context, MetalCachedImage image)
    {
        ref readonly var description = ref image.Description;
        if (!TrySplitLayers(in description, out var layers, out var layerBytes))
        {
            return WriteBackInRuns(context, image, 0, description.Data.Size);
        }

        var batch = (uint)Math.Clamp(MaxTransferPieceBytes / layerBytes, 1, layers);
        var stencilLayerBytes = description.HasStencil && layers != 0 ? description.Stencil.Size / layers : 0;
        var moved = false;
        for (uint first = 0; first < layers; first += batch)
        {
            var count = Math.Min(batch, layers - first);
            var pieceBase = layerBytes * first;
            var pieceSize = layerBytes * count;
            var piece = _guestMemory.ResolveWritable(description.Data.Address + pieceBase, pieceSize);
            if (!piece.IsValid || piece.Size < pieceSize)
            {
                // These layers are not one backed span; cut them into the runs that are
                // rather than leaving the whole piece in the texture.
                moved |= WriteBackInRuns(context, image, pieceBase, pieceBase + pieceSize);
                continue;
            }

            var stencilPiece = description.HasStencil && stencilLayerBytes != 0
                ? _guestMemory.ResolveWritable(
                    description.Stencil.Address + (stencilLayerBytes * first), stencilLayerBytes * count)
                : default;
            var range = new SubresourceRange(0, description.Resources.Levels, first, count);
            _transfer.EncodeTextureToGuest(
                context, in description, range, image.Texture, piece, stencilPiece, pieceBase, pieceSize);
            moved = true;
        }

        return moved;
    }

    // The last resort of a writeback: a range of an image whose guest bytes are not one
    // backed span at all. Each run is the longest prefix one window covers, pulled back
    // to a boundary no subresource straddles, and the transfer is restricted to it. A run
    // that resolves to nothing is a hole in the guest mapping: those bytes stay in the
    // texture, and the next run starts after them.
    private bool WriteBackInRuns(IMetalSubmissionContext context, MetalCachedImage image, ulong from, ulong to)
    {
        ref readonly var description = ref image.Description;
        var range = new SubresourceRange(0, description.Resources.Levels, 0, description.TransferLayers);
        var size = Math.Min(to, description.Data.Size);
        // The stencil plane is named by layers, not by the depth plane's bytes: one
        // layer of stencil is a byte per texel where one layer of depth is four, so a
        // run's stencil bytes are its layers' stencil bytes.
        var layers = description.TransferLayers;
        var layerBytes = layers != 0 ? description.Data.Size / layers : 0;
        var stencilLayerBytes = description.HasStencil && layers != 0 ? description.Stencil.Size / layers : 0;
        var written = 0UL;
        // What the walk did, kept only once something is left behind: a walk that covers
        // the image says nothing and allocates nothing.
        StringBuilder? trace = null;
        for (var offset = from; offset < size;)
        {
            var start = _transfer.AlignRunStart(in description, offset);
            if (start > offset)
            {
                // The step over a hole landed inside a transfer whose other half it
                // stepped past; that half carries nothing either.
                trace ??= Opened(from, written);
                Trace(trace, "skip", offset, Math.Min(size, start) - offset);
                offset = start;
                continue;
            }

            var remaining = Math.Min(size - offset, MaxTransferPieceBytes);
            var run = _guestMemory.ResolveWritablePrefix(description.Data.Address + offset, remaining);
            var end = run.IsValid
                ? _transfer.AlignRunEnd(in description, offset, offset + run.Size)
                : offset;
            if (end <= offset)
            {
                // Nothing of the image both starts here and fits in what one window
                // covers. Step over everything that window reached - no subresource can
                // start inside it either - and at least one page, so a range no window
                // covers at all still advances.
                var step = Math.Min(size - offset, Math.Max(run.Size, HoleStepBytes));
                trace ??= Opened(from, written);
                Trace(trace, "skip", offset, step);
                offset += step;
                continue;
            }

            var pieceSize = end - offset;
            var stencil = stencilLayerBytes != 0 && layerBytes != 0
                ? _guestMemory.ResolveWritable(
                    description.Stencil.Address + (stencilLayerBytes * (offset / layerBytes)),
                    stencilLayerBytes * Math.Max(1UL, pieceSize / layerBytes))
                : default;
            _transfer.EncodeTextureToGuest(
                context, in description, range, image.Texture,
                run with { Size = pieceSize }, stencil, offset, pieceSize);
            written += pieceSize;
            if (trace != null)
            {
                Trace(trace, "run", offset, pieceSize);
            }

            offset = end;
        }

        if (written < size - from)
        {
            // Counted from what the runs carried rather than from the steps over holes:
            // a block row the boundary falls inside is not carried either, and saying so
            // is how a surface that comes out part stale is recognised.
            MetalV2Diagnostics.WarnOnce(
                "writeback-unresolved",
                $"A GPU-written image has guest bytes no run covers, so those stay in the texture: " +
                $"address=0x{description.Data.Address + from:X16} size=0x{size - from:X} " +
                $"unwritten=0x{size - from - written:X} " +
                $"{ProfileLabel("writeback", in description)} runs=[{trace}] " +
                $"({_guestMemory.DescribeResolve(description.Data.Address, description.Data.Size)}).");
        }

        return written != 0;
    }

    /// <summary>One run of an upload walk: image bytes, and whether the ring stages them.</summary>
    internal readonly record struct UploadRun(ulong Offset, ulong Size, bool Staged);

    // The upload counterpart of WriteBackInRuns: a range of an image whose guest bytes
    // are not one window-resident span - most often a large image that crosses the edge
    // between two windows. Each run is the longest prefix one window covers, pulled back
    // to a boundary no subresource straddles, and the transfer is restricted to it. A run
    // is at most MaxTransferPieceBytes; asking for it moves the window edge inside it out of
    // the way when the command buffer has not used the windows beside that edge yet, so a
    // backed byte reaches the texture through a window, and the ring stages only what no
    // window holds now.
    private bool UploadInRuns(IMetalSubmissionContext context, MetalCachedImage image, ulong from, ulong to)
    {
        ref readonly var description = ref image.Description;
        var address = description.Data.Address;
        var runs = new List<UploadRun>();
        var uncovered = PlanUploadRuns(
            _transfer, in description, from, to, MaxTransferPieceBytes,
            (offset, size) => _guestMemory.ResolveWritablePrefix(address + offset, size).Size, runs);
        var range = ResidentRange(image);
        // The stencil plane is named by layers, as in WriteBackInRuns.
        var layers = description.TransferLayers;
        var layerBytes = layers != 0 ? description.Data.Size / layers : 0;
        var stencilLayerBytes = description.HasStencil && layers != 0 ? description.Stencil.Size / layers : 0;
        var moved = 0UL;
        var staged = 0UL;
        foreach (var run in runs)
        {
            var source = run.Staged
                ? _guestMemory.Resolve(address + run.Offset, run.Size)
                : _guestMemory.ResolveWritable(address + run.Offset, run.Size);
            if (!source.IsValid || source.Size < run.Size)
            {
                uncovered += run.Size;
                continue;
            }

            var stencil = stencilLayerBytes != 0 && layerBytes != 0
                ? _guestMemory.Resolve(
                    description.Stencil.Address + (stencilLayerBytes * (run.Offset / layerBytes)),
                    stencilLayerBytes * Math.Max(1UL, run.Size / layerBytes))
                : default;
            _transfer.EncodeGuestToTexture(
                context, in description, range, source with { Size = run.Size }, stencil, image.Texture,
                run.Offset, run.Size, image.FirstLevel);
            moved += run.Size;
            staged += run.Staged ? run.Size : 0;
        }

        MetalV2Diagnostics.InfoOnce(
            "image-upload-runs",
            $"An image that is not one window-resident span is uploaded in runs: " +
            $"address=0x{address + from:X16} size=0x{Math.Min(to, description.Data.Size) - from:X} " +
            $"runs={runs.Count} staged=0x{staged:X} " +
            $"({_guestMemory.DescribeResolve(address, description.Data.Size)}).");
        if (uncovered != 0)
        {
            MetalV2Diagnostics.WarnOnce(
                "upload-unresolved",
                $"An image has guest bytes no upload run covers, so the texture keeps what it holds there: " +
                $"address=0x{address + from:X16} size=0x{Math.Min(to, description.Data.Size) - from:X} " +
                $"uncovered=0x{uncovered:X} {ProfileLabel("upload", in description)} " +
                $"({_guestMemory.DescribeResolve(address, description.Data.Size)}).");
        }

        return moved != 0;
    }

    // The runs an upload of [from, to) is cut into, given how many bytes from an image
    // offset one window holds (zero when none holds the first byte). The walk is the one
    // WriteBackInRuns makes: start where a run may begin, take the window's prefix up to
    // maxRunBytes, pull its end back to a cut no subresource straddles. Where that leaves
    // nothing - no window holds a whole block row from here - the bytes up to the first
    // cut past what the window holds are staged instead, never more than maxRunBytes, so
    // the ring stays inside its cap. Returns the bytes no run carries.
    internal static ulong PlanUploadRuns(
        IMetalImageTransfer transfer, in ImageDescription description, ulong from, ulong to, ulong maxRunBytes,
        Func<ulong, ulong, ulong> windowPrefix, List<UploadRun> runs)
    {
        var size = Math.Min(to, description.Data.Size);
        var uncovered = 0UL;
        for (var offset = from; offset < size;)
        {
            var start = transfer.AlignRunStart(in description, offset);
            if (start > offset)
            {
                // Inside a transfer whose head a step stepped past; nothing carries it.
                uncovered += Math.Min(size, start) - offset;
                offset = start;
                continue;
            }

            var remaining = Math.Min(size - offset, maxRunBytes);
            var resident = Math.Min(windowPrefix(offset, remaining), remaining);
            var end = resident != 0
                ? transfer.AlignRunEnd(in description, offset, offset + resident)
                : offset;
            if (end > offset)
            {
                runs.Add(new UploadRun(offset, end - offset, Staged: false));
                offset = end;
                continue;
            }

            // A hole in the guest mapping, or a window edge that cannot move in this
            // command buffer: stage up to the first cut past what the window holds, and at
            // least past one page.
            var cut = transfer.AlignRunStart(in description, offset + Math.Max(resident, HoleStepBytes));
            end = transfer.AlignRunEnd(in description, offset, Math.Min(Math.Min(cut, size), offset + remaining));
            if (end <= offset)
            {
                // One block row larger than a run: nothing may carry it.
                var step = Math.Min(size - offset, Math.Max(resident, HoleStepBytes));
                uncovered += step;
                offset += step;
                continue;
            }

            runs.Add(new UploadRun(offset, end - offset, Staged: true));
            offset = end;
        }

        return uncovered;
    }

    // The trace is opened at the first hole, so the runs before it are not in it; what
    // they carried goes in as one entry, and the entries then account for every byte.
    private static StringBuilder Opened(ulong from, ulong carried)
    {
        var trace = new StringBuilder();
        if (carried != 0)
        {
            Trace(trace, "run", from, carried);
        }

        return trace;
    }

    // What a run walk did, for the warning that follows a walk which left bytes behind:
    // which offsets carried and which were stepped over says whether the loss is a gap
    // in the guest mapping or a cut the transfer cannot make.
    private static void Trace(StringBuilder trace, string kind, ulong offset, ulong bytes)
    {
        if (trace.Length > 512)
        {
            return;
        }

        if (trace.Length != 0)
        {
            trace.Append(' ');
        }

        trace.Append($"{kind}@0x{offset:X}+0x{bytes:X}");
    }

    // How far a run walk steps past guest bytes no window covers. One guest page: the
    // smallest unit a mapping can begin on, so no backed run is skipped.
    private const ulong HoleStepBytes = 4096;

    // What a transfer is, for the GPU profiler: the shape the tiler actually walks.
    private static string ProfileLabel(string direction, in ImageDescription description) =>
        $"{direction} format={description.PixelFormat} type={description.Type} " +
        $"extent={description.Extent.Width}x{description.Extent.Height}x{description.Extent.Depth} " +
        $"levels={description.Resources.Levels} layers={description.TransferLayers} " +
        $"tile={description.TileMode} samples={description.Samples} bytes=0x{description.Data.Size:X}";

    // Linear images the GPU wrote since the last end-of-pipe write. Linear is the layout a
    // guest CPU reads in place - a readback staging surface - so these are the GPU writes
    // the guest can read without asking the GPU for them.
    private readonly HashSet<MetalCachedImage> _unpublishedLinearWrites = new();

    private void NoteLinearGpuWrite(MetalCachedImage image)
    {
        if (image.Description.TileMode == GuestTileMode.Linear)
        {
            _unpublishedLinearWrites.Add(image);
        }
    }

    /// <summary>
    /// Writes back every linear image the GPU wrote since the last call. An end-of-pipe
    /// write tells the guest that the work before it is done, and a CPU that then reads a
    /// linear surface that work wrote reads guest memory, which a cached texture reaches
    /// only through a writeback. SILENT HILL: The Short Message reads its exposure this
    /// way; without it the pre-exposure stays 1 and the title backdrop renders black.
    /// </summary>
    public void PublishLinearGpuWrites()
    {
        using var held = _lock.Hold();
        if (_unpublishedLinearWrites.Count == 0)
        {
            return;
        }

        foreach (var image in _unpublishedLinearWrites)
        {
            WriteBackToGuest(image, MetalWritebackCause.HostRead);
        }

        _unpublishedLinearWrites.Clear();
    }

    /// <summary>Writes every GPU-modified image back, then lets the CPU read guest memory.</summary>
    public void FlushAllGpuWrites()
    {
        using var held = _lock.Hold();
        var pending = new List<MetalCachedImage>();
        _slots.ForEach((_, image) =>
        {
            if (image.IsGpuModified)
            {
                pending.Add(image);
            }
        });
        foreach (var image in pending)
        {
            WriteBackToGuest(image, MetalWritebackCause.DrainAll);
        }
    }

    private readonly long[] _uploadCauses = new long[5];
    private readonly long[] _writebackCauses = new long[7];
    private readonly long[] _releaseCauses = new long[11];

    private static readonly string[] _releaseCauseNames =
        ["format", "resources", "viewtype", "same_address", "subset", "overlap", "budget", "unmapped", "linear_promoted", "retired", "stale"];

    /// <summary>The release reasons since the last read, as "name=count" pairs.</summary>
    public void AppendReleaseCauses(System.Text.StringBuilder target)
    {
        for (var index = 0; index < _releaseCauses.Length; index++)
        {
            var count = Interlocked.Exchange(ref _releaseCauses[index], 0);
            if (count != 0)
            {
                target.Append(" drop_").Append(_releaseCauseNames[index]).Append('=').Append(count);
            }
        }
    }

    // Names the two descriptions behind a release, once per shape pair. A cache that
    // recreates a surface every frame does it for one reason with one pair of shapes.
    private void ReportRelease(
        MetalImageReleaseCause cause,
        in ImageDescription cached,
        in ImageDescription requested,
        in ImageViewDescription view)
    {
        MetalV2Diagnostics.WarnOnce(
            $"image-drop:{cause}:{cached.PixelFormat}:{requested.PixelFormat}:{cached.Extent.Width}x{cached.Extent.Height}" +
            $":{requested.Extent.Width}x{requested.Extent.Height}:{cached.Resources.Levels}:{requested.Resources.Levels}",
            $"A cached image is dropped and recreated: cause={cause} " +
            $"cached=0x{cached.Data.Address:X16}+0x{cached.Data.Size:X} {cached.Extent.Width}x{cached.Extent.Height}x{cached.Extent.Depth} " +
            $"{cached.PixelFormat} type={cached.Type} tile={cached.TileMode} " +
            $"levels={cached.Resources.Levels} layers={cached.Resources.Layers} samples={cached.Samples} " +
            $"requested=0x{requested.Data.Address:X16}+0x{requested.Data.Size:X} " +
            $"{requested.Extent.Width}x{requested.Extent.Height}x{requested.Extent.Depth} " +
            $"{requested.PixelFormat} type={requested.Type} tile={requested.TileMode} " +
            $"levels={requested.Resources.Levels} layers={requested.Resources.Layers} samples={requested.Samples} " +
            $"view_type={view.Type} view_format={view.Format} view_levels={view.BaseLevel}+{view.LevelCount} " +
            $"view_layers={view.BaseLayer}+{view.LayerCount}.");
    }

    /// <summary>
    /// Where the uploads and the writebacks since the last read came from. A static
    /// menu that uploads and writes back every frame is churning, and only the cause
    /// says which side of the host is asking for it.
    /// </summary>
    public (long Created, long CpuWrite, long GpuWrite, long Invalidated, long Alias,
        long Binding, long Storage, long HostRead, long GuestCopy, long Overlap, long Release, long Drain)
        TakeTransferCauses()
    {
        var uploads = (
            Interlocked.Exchange(ref _uploadCauses[0], 0),
            Interlocked.Exchange(ref _uploadCauses[1], 0),
            Interlocked.Exchange(ref _uploadCauses[2], 0),
            Interlocked.Exchange(ref _uploadCauses[3], 0),
            Interlocked.Exchange(ref _uploadCauses[4], 0));
        return (uploads.Item1, uploads.Item2, uploads.Item3, uploads.Item4, uploads.Item5,
            Interlocked.Exchange(ref _writebackCauses[0], 0),
            Interlocked.Exchange(ref _writebackCauses[1], 0),
            Interlocked.Exchange(ref _writebackCauses[2], 0),
            Interlocked.Exchange(ref _writebackCauses[3], 0),
            Interlocked.Exchange(ref _writebackCauses[4], 0),
            Interlocked.Exchange(ref _writebackCauses[5], 0),
            Interlocked.Exchange(ref _writebackCauses[6], 0));
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        ReleaseAllFencedWritebacks();
        // Teardown follows a full GPU wait, so nothing is still naming these.
        foreach (var (_, retiring) in _retiring)
        {
            retiring.Dispose();
        }

        _retiring.Clear();
        _slots.ForEach(static (_, image) => image.Dispose());
    }

    /// <summary>Flushes GPU-owned contents and drops every image.</summary>
    public void Shutdown()
    {
        try
        {
            FlushAllGpuWrites();
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"[METAL-V2][WARN] The image cache could not drain on shutdown: {exception.Message}");
        }

        using (var held = _lock.Hold())
        {
            var registered = new List<ResourceSlotIdentifier>();
            _slots.ForEach((imageIdentifier, image) =>
            {
                if (image.Registered)
                {
                    registered.Add(imageIdentifier);
                }
            });
            foreach (var imageIdentifier in registered)
            {
                RemoveFromIndex(imageIdentifier);
            }
        }

        Dispose();
    }
}
