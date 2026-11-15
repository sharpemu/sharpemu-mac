// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Scheduling;

namespace SharpEmu.Gpu.Metal.V2;

// Why a cached image has to be uploaded from guest memory again. Counted per cause
// so a run can say whether the churn comes from the guest writing the bytes or from
// the host invalidating them itself.
internal enum MetalImageDirtyCause : byte
{
    Created,
    CpuWrite,
    GpuWrite,
    Invalidated,
    AliasWrite,
}

// Why a cached image was dropped and a new one created in its place. A title that
// recreates the same surface every frame pays an upload and a writeback for each
// cycle, and the reason is the only thing that says which rule to relax.
internal enum MetalImageReleaseCause : byte
{
    FormatMismatch,
    FewerResources,
    ViewType,
    SameAddressReplaced,
    SubsetOfLarger,
    OverlapEvicted,
    Budget,
    Unmapped,
    // A texture that read the guest window in place became an owned texture because the
    // image was written, stored to, copied into, or needed a view the window texture cannot give.
    LinearViewPromoted,
    // Not used for the retirement age (frames and seconds), whatever the budget.
    Retired,
    // Never written by the GPU, behind guest memory already and unwatched: nothing a read could
    // use without uploading it again, only the cache lock on every CPU store into its pages.
    Stale,
}

/// <summary>What a writeback carried to guest memory.</summary>
internal enum MetalWritebackOutcome
{
    // Every byte of the image, or there was nothing to carry.
    Complete,
    // Some bytes no run could carry; guest memory holds the old ones there.
    Partial,
    // Nothing: the texture alone still holds the image, which stays GPU-written.
    Failed,
}

// Why a GPU-written image had to be published into guest memory.
internal enum MetalWritebackCause : byte
{
    BufferBinding,
    StorageDescriptor,
    HostRead,
    GuestCopy,
    ImageOverlap,
    Release,
    DrainAll,
}

// What a cached image has been bound as; the upload role follows from it.
internal struct MetalImageUses
{
    public bool Texture;
    public bool Storage;
    public bool RenderTarget;
    public bool DepthTarget;
    public bool VideoOut;
}

// One guest image resident on the device: its description, the MTLTexture behind it,
// the views the shaders bind, and who owns its current contents.
internal sealed class MetalCachedImage : IDisposable
{
    private static readonly nint SelRelease = MetalNative.Selector("release");
    private static readonly nint SelGpuResourceId = MetalNative.Selector("gpuResourceID");
    private static readonly nint SelNewTextureView =
        MetalNative.Selector("newTextureViewWithPixelFormat:textureType:levels:slices:");
    private static readonly nint SelNewSwizzledTextureView =
        MetalNative.Selector("newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:");
    private static long _gpuWriteCounter;

    // The flip path reaches GetOrCreateView from the GPU worker while the cache
    // reaches it under its own lock, so the view list needs one of its own. It
    // used to be reached rarely enough for the race to stay latent: before the
    // guest swizzle was honoured, the common whole-image view returned the
    // texture without ever touching the list.
    private readonly object _viewGate = new();
    private readonly List<(ImageViewDescription Description, nint Texture, ulong ResourceId)> _views = new();
    private nint _textureIdHandle;
    private ulong _textureId;
    private bool _cpuDirty;
    private bool _gpuModified;
    private bool _disposed;
    private bool _windowLinearView;

    /// <summary>
    /// The cache's mutation counts, which this image bumps when its texture, views or
    /// ownership state change (MetalImageEpochs). Set while the image is in the cache; an
    /// image retired from it, or never in it, counts nothing.
    /// </summary>
    public MetalImageEpochs? Epochs { get; set; }

    /// <summary>
    /// The split GPU process's pending-writeback summary (MetalImageCache.PendingWritebacks), which
    /// holds this image's pages while it is GPU-modified. Set while the image is in the cache;
    /// setting it moves the claim, and null drops it.
    /// </summary>
    public IPendingWritebackPages? PendingWritebacks
    {
        get => _pendingWritebacks;
        set
        {
            ReleasePendingWriteback();
            _pendingWritebacks = value;
            if (_gpuModified)
            {
                ClaimPendingWriteback();
            }
        }
    }

    private IPendingWritebackPages? _pendingWritebacks;
    private ulong _pendingAddress;
    private ulong _pendingSize;

    /// <summary>
    /// Moves wherever this image bumps a count of <see cref="Epochs"/> (and where the cache changes
    /// its metadata): a memo of a binding of this image alone is stale once it moved.
    /// </summary>
    public long BindingVersion;

    /// <summary>
    /// The owner-list counts over this image's bytes when an alias sync last found no other image
    /// there, or -1 (MetalImageCache.SynchronizeAliases).
    /// </summary>
    public long AlonePageVersions = -1;

    /// <summary>
    /// What the last alias sync of this image left beside it (MetalImageCache.SynchronizeAliases):
    /// the other images over its bytes, the versions of all of them, this image's contents, the
    /// owner-list counts there and the kind of sync, so a repeat that would change nothing is
    /// skipped. Null when it was not recorded.
    /// </summary>
    public MetalCachedImage[]? SyncedOthers;
    public long SyncedPageVersions = -1;
    public long SyncedVersion;
    public long SyncedContent;
    public byte SyncedKind;

    public MetalCachedImage(in ImageDescription description, in MetalTextureShape shape, nint texture)
    {
        Description = description;
        Shape = shape;
        Texture = texture;
        ResidentEnd = description.Data.Size;
        // A newly created image starts with the guest's bytes as its contents, so the
        // first use uploads unless the GPU takes ownership first. Stale over all of it, said
        // as such: an empty interval would let the next run widen it to that run alone.
        _cpuDirty = !ImageDescription.IsEmptyRange(description.Data);
        if (_cpuDirty)
        {
            _stale[0] = (0, description.Data.Size);
            _staleCount = 1;
            StaleCauses = 1 << (int)MetalImageDirtyCause.Created;
        }

        NoteContentChanged();
    }

    private static long _contentCounter;

    /// <summary>
    /// Changes whenever what the texture holds may have changed: a GPU write is about to
    /// happen, guest memory got newer bytes for it, an upload or a copy landed, or the
    /// texture itself was replaced. One counter across every image, so a version names
    /// one state of one texture and can never match another image's.
    /// </summary>
    public long ContentVersion { get; private set; }

    public void NoteContentChanged()
    {
        // The texture is about to hold something else: chunks whose guest copy it holds get
        // their bytes back from it first (single-copy textures).
        SingleCopyRecord?.Owner.OnContentChanging(this);
        BumpContentVersion();
    }

    private void BumpContentVersion()
    {
        ContentVersion = Interlocked.Increment(ref _contentCounter);
        DepthPlaneVersion = ContentVersion;
    }

    /// <summary>The single-copy record of the chunks this texture holds the guest copy of, or null.</summary>
    internal MetalSingleCopy.Record? SingleCopyRecord;

    /// <summary>The content version the single-copy scan last saw, and the timestamp it first saw it at.</summary>
    internal long SingleCopySeenVersion;

    internal long SingleCopySeenAt;

    /// <summary>
    /// The version of the depth plane alone: every change moves it with
    /// <see cref="ContentVersion"/> except a GPU write that can only reach the stencil plane.
    /// </summary>
    public long DepthPlaneVersion { get; private set; }

    // Only the stencil plane is about to change; what the depth plane holds stays as it is.
    private void NoteStencilChanged() => ContentVersion = Interlocked.Increment(ref _contentCounter);

    // What the last device publication into this image copied: the source texture and
    // the version of its depth plane, and this image's own version once the copy landed.
    // While all three still hold, this texture already holds what a new copy would put
    // there. The device publication copies a depth plane into a colour alias, or a 16-bit
    // colour alias into a depth plane with no stencil, and nothing else
    // (MetalImageTransfer.TryEncodeTextureToTexture), so a stencil plane is never part of
    // what either side holds from the other; a colour source's DepthPlaneVersion moves
    // with every change of its texture.
    private nint _publishedSourceTexture;
    private long _publishedSourceVersion;
    private long _publishedOwnVersion;

    /// <summary>Records that this image now holds a device copy of the source's depth plane as it is.</summary>
    public void NotePublishedFrom(MetalCachedImage source)
    {
        _publishedSourceTexture = source.Texture;
        _publishedSourceVersion = source.DepthPlaneVersion;
        _publishedOwnVersion = ContentVersion;
    }

    /// <summary>
    /// True when the last device publication into this image was from this source, and
    /// neither this texture nor the source's depth plane has changed since: a copy now
    /// would rewrite the same texels.
    /// </summary>
    public bool StillHoldsPublicationFrom(MetalCachedImage source) =>
        !_cpuDirty && Texture != 0 && source.Texture != 0 &&
        _publishedSourceTexture == source.Texture &&
        _publishedSourceVersion == source.DepthPlaneVersion &&
        _publishedOwnVersion == ContentVersion;

    // The stencil plane a device copy last brought into this depth texture from an image
    // over the plane's bytes (MetalImageCache.SynchronizeStencilPlanes): that image's
    // texture and content version.
    private nint _stencilSourceTexture;
    private long _stencilSourceVersion;

    /// <summary>
    /// Set when an image that writes this depth image's stencil bytes takes them over: the
    /// next time this image binds, its stencil plane is copied from that image on the
    /// device, and the depth plane its texture holds stays as it is. The image may still be
    /// GPU-modified for its depth plane; its writeback then leaves the stencil bytes alone.
    /// </summary>
    public bool StencilPlaneStale { get; set; }

    /// <summary>
    /// The stencil-owner set version (MetalImageCache._stencilOwnersVersion) at which no other image's
    /// stencil plane lay over this image's bytes; -1 when not known. Which planes overlap follows from
    /// the images' ranges alone, so while the set holds the walk over it finds nothing again.
    /// </summary>
    public long NoForeignStencilVersion { get; set; } = -1;

    /// <summary>
    /// An image over this depth image's stencil bytes whose texture holds a newer stencil, to copy in on the
    /// device once this image has reloaded its depth plane from guest memory (MetalImageCache.SynchronizeStencilPlanes).
    /// </summary>
    public MetalCachedImage? PendingStencilSource { get; set; }

    /// <summary>The cache slot of <see cref="PendingStencilSource"/>, to tell it is still the live image there.</summary>
    public SharpEmu.Libs.Gpu.Buffers.ResourceSlotIdentifier PendingStencilSourceSlot { get; set; }

    /// <summary>Records that this texture's stencil plane is a device copy of the source as it is.</summary>
    public void NoteStencilFrom(MetalCachedImage source)
    {
        _stencilSourceTexture = source.Texture;
        _stencilSourceVersion = source.ContentVersion;
        StencilPlaneStale = false;
    }

    /// <summary>
    /// True while the last stencil copy into this texture came from the source and the
    /// source has not changed since: a copy now would write the same stencil.
    /// </summary>
    public bool HoldsStencilFrom(MetalCachedImage source) =>
        Texture != 0 && source.Texture != 0 &&
        _stencilSourceTexture == source.Texture && _stencilSourceVersion == source.ContentVersion;

    // The other direction, on the 8-bit image over a depth image's stencil bytes: the depth
    // texture its last device copy came from, that image's version then, and this image's
    // own version once the copy landed.
    private nint _stencilCopySourceTexture;
    private long _stencilCopySourceVersion;
    private long _stencilCopyOwnVersion;

    /// <summary>Records that this texture now holds a device copy of the depth image's stencil plane as it is.</summary>
    public void NoteStencilCopyOf(MetalCachedImage depth)
    {
        _stencilCopySourceTexture = depth.Texture;
        _stencilCopySourceVersion = depth.ContentVersion;
        _stencilCopyOwnVersion = ContentVersion;
    }

    /// <summary>
    /// True while neither this texture nor the depth image has changed since the last
    /// stencil copy out of it: a copy now would write the same texels.
    /// </summary>
    public bool StillHoldsStencilCopyOf(MetalCachedImage depth) =>
        !_cpuDirty && Texture != 0 && depth.Texture != 0 &&
        _stencilCopySourceTexture == depth.Texture &&
        _stencilCopySourceVersion == depth.ContentVersion &&
        _stencilCopyOwnVersion == ContentVersion;

    public ImageDescription Description;

    public MetalTextureShape Shape { get; private set; }

    // Every texture this image refers to is counted in MetalTileViews, which keeps the
    // tiler's integer view of it until the last image lets go of the texture.
    public nint Texture
    {
        get => _texture;
        private set
        {
            if (_texture == value)
            {
                return;
            }

            MetalTileViews.Reference(value);
            MetalTileViews.Unreference(_texture);
            _texture = value;
        }
    }

    private nint _texture;

    /// <summary>
    /// The guest mip level the texture's level 0 holds. Zero for the whole chain; above zero
    /// for a sampled texture held from its T# BASE_LEVEL (MetalImageCache.BaseLevel.cs), whose
    /// finer levels the GPU cannot address and the host does not allocate. The description
    /// stays the guest's full chain either way: identity, guest offsets and the page index
    /// are the guest's, only the texture and what is uploaded into it are cut.
    /// </summary>
    public uint FirstLevel { get; private set; }

    /// <summary>
    /// The guest bytes, as offsets into the image, that the levels the texture holds read:
    /// the whole image for the whole chain, the coarser levels' run for a texture from a
    /// later level. Uploads and the page watch cover this run and nothing else.
    /// </summary>
    public ulong ResidentStart { get; private set; }

    public ulong ResidentEnd { get; private set; }

    /// <summary>No sampled binding needed any level in the window.</summary>
    public const uint NoNeededLevel = uint.MaxValue;

    /// <summary>
    /// The finest guest level a sampled binding needed (its BASE_LEVEL as the texture would
    /// be cut for it) in the current window of presented frames, begun at
    /// <see cref="NeededWindowFrame"/>, and in the window before it. Together they hold every
    /// binding of at least the last window, so a texture is cut down only to a level nothing
    /// bound in that time needed finer than (MetalImageCache.BaseLevel.cs).
    /// </summary>
    public uint NeededLevel = NoNeededLevel;

    public uint NeededLevelBefore = NoNeededLevel;

    public long NeededWindowFrame;

    /// <summary>The presented frame the texture's levels were last set at: made, grown or shrunk.</summary>
    public long LevelsChangedFrame;

    /// <summary>Whether that change was a shrink, so a grow soon after it is a reversal.</summary>
    public bool LevelsLastShrunk;

    /// <summary>Doublings of the shrink window, one per shrink a binding soon reversed.</summary>
    public byte ShrinkBackoff;

    /// <summary>Records which guest levels a newly made texture holds and the bytes they read.</summary>
    public void SetResidentLevels(uint firstLevel, ulong start, ulong end)
    {
        FirstLevel = firstLevel;
        ResidentStart = start;
        ResidentEnd = end;
    }

    public MetalImageUses Uses;

    /// <summary>
    /// The image's bytes are recorded as GPU-written (MetalImageCache.Lossy.cs); set the
    /// first time it takes GPU ownership, so later writes skip the record.
    /// </summary>
    public bool GpuBytesNoted;

    public bool Registered;

    public uint QueryEpoch;

    public ulong WatchBegin;

    public ulong WatchEnd;

    private ulong _lastAccessTick;

    public ulong LastAccessTick
    {
        get => _lastAccessTick;
        set
        {
            _lastAccessTick = value;
            if (SharpEmu.Logging.MemoryTelemetry.Enabled)
            {
                LastAccessTimestamp = Volatile.Read(ref AccessClock);
            }
        }
    }

    /// <summary>Stopwatch ticks of the last use; kept only with SHARPEMU_LOG_MEMORY=1 (idle-age census).</summary>
    public long LastAccessTimestamp;

    // The time a use is stamped with: read once per collection (MetalImageCache.Collect, at the end of every guest
    // submission), not per use. Every texture binding and attachment sets LastAccessTick, and a clock read there
    // was ~1.7% of Demon's Souls' GPU worker in every SHARPEMU_LOG_MEMORY=1 run; the census buckets idle images by
    // a second, ten and sixty, which one submission's lag does not move.
    internal static long AccessClock = System.Diagnostics.Stopwatch.GetTimestamp();

    // The submission serial of the command buffer that last wrote the image, so a
    // writeback knows whether the GPU work it depends on has been recorded.
    public ulong GpuWriteSerial;

    // The newest submission that bound this image's texture or any of its views.
    // An argument buffer holds a gpuResourceID, which Metal does not retain, so
    // the Metal objects have to outlive every command buffer that named them.
    public ulong LastUseSerial;

    public void NoteUse(ulong serial)
    {
        if (serial > LastUseSerial)
        {
            LastUseSerial = serial;
        }
    }

    public long GpuWriteSequence { get; private set; }

    public bool IsCpuDirty => _cpuDirty;

    /// <summary>
    /// True while the texture is a linear texture over the guest window buffer: it reads
    /// the guest bytes in place, so it has no contents of its own to upload, watch or write
    /// back. The image still owns and releases the texture object; the texture retains the
    /// window buffer, which is never freed under it.
    /// </summary>
    public bool IsWindowLinearView => _windowLinearView;

    /// <summary>Marks the texture the image was made with as a window view: never dirty, never uploaded.</summary>
    public void MarkWindowLinearView()
    {
        _windowLinearView = true;
        _cpuDirty = false;
        _staleCount = 0;
        NoteContentChanged();
    }

    /// <summary>
    /// Whether the window texture can serve this view: the whole image in its own format
    /// and type. A component swizzle is fine; a buffer-backed texture makes those views.
    /// </summary>
    public bool CanWindowViewServe(in ImageViewDescription requested)
    {
        return MetalV2Formats.PixelFormat(requested.Format) == Shape.PixelFormat &&
            MetalV2Formats.ViewTextureType(requested.Type, Shape.SampleCount) == Shape.TextureType &&
            requested.BaseLevel == 0 && requested.LevelCount >= Shape.MipLevels &&
            requested.BaseLayer == 0 && requested.LayerCount >= Shape.ArrayLength;
    }

    /// <summary>
    /// Gives the image an owned texture in place of its window view. The old texture and its
    /// views move to the returned image, which the cache retires once the GPU is done with it.
    /// The new texture holds nothing yet, so the image is CPU-dirty over its whole range.
    /// </summary>
    public MetalCachedImage PromoteToOwned(nint texture, in MetalTextureShape shape)
    {
        var retired = new MetalCachedImage(in Description, Shape, Texture) { LastUseSerial = LastUseSerial };
        lock (_viewGate)
        {
            retired._views.AddRange(_views);
            _views.Clear();
        }

        Texture = texture;
        Shape = shape;
        _windowLinearView = false;
        MarkCpuDirty(MetalImageDirtyCause.Invalidated);
        NoteContentChanged();
        BindingVersion++;
        Epochs?.NoteShapeChanged();
        return retired;
    }

    /// <summary>
    /// Gives the image a texture holding the guest levels from <paramref name="firstLevel"/>,
    /// in place of one holding a different set. The old texture and its views move to the
    /// returned image, which the cache retires once the GPU is done with it. The new texture
    /// holds nothing yet, so the image is CPU-dirty over its whole range; only a sampled-only
    /// image is ever cut, so guest memory holds everything the old texture did.
    /// </summary>
    public MetalCachedImage ReplaceLevels(nint texture, in MetalTextureShape shape, uint firstLevel, ulong start, ulong end)
    {
        if (_gpuModified || _windowLinearView)
        {
            throw GpuFatal.Fatal(
                $"Only a sampled-only owned texture changes its levels: gpuModified={_gpuModified} " +
                $"windowView={_windowLinearView} address=0x{Description.Data.Address:X16}.");
        }

        var retired = new MetalCachedImage(in Description, Shape, Texture) { LastUseSerial = LastUseSerial };
        lock (_viewGate)
        {
            retired._views.AddRange(_views);
            _views.Clear();
        }

        Texture = texture;
        Shape = shape;
        // A replacement texture is an ordinary one; the sparse grow path sets its record after.
        Sparse = null;
        SetResidentLevels(firstLevel, start, end);
        MarkCpuDirty(MetalImageDirtyCause.Invalidated);
        NoteContentChanged();
        BindingVersion++;
        Epochs?.NoteShapeChanged();
        return retired;
    }

    /// <summary>
    /// Gives the image a texture of the same shape with more usage, which the caller has
    /// already filled with a device copy of the current one. The old texture and its views
    /// move to the returned image, which the cache retires once the copy and every
    /// submission that bound them are done. Nothing the image owes changes: the new texture
    /// holds what the old one held.
    /// </summary>
    public MetalCachedImage ReplaceWithCopy(nint texture, in MetalTextureShape shape, ulong copySerial)
    {
        var retired = new MetalCachedImage(in Description, Shape, Texture)
        {
            LastUseSerial = Math.Max(LastUseSerial, copySerial),
        };
        lock (_viewGate)
        {
            retired._views.AddRange(_views);
            _views.Clear();
        }

        Texture = texture;
        Shape = shape;
        Sparse = null;
        NoteContentChanged();
        BindingVersion++;
        Epochs?.NoteShapeChanged();
        return retired;
    }

    /// <summary>
    /// The texture usage a view needs that the texture may lack: pixel-format-view for a
    /// view of another format, unless the two formats are only the sRGB and linear
    /// variants of one format, which Metal views without it; and for a component swizzle,
    /// which this cache makes on a texture that has it (a texture it cannot promote, such as
    /// a compressed one, still gets a same-format swizzled view: GetOrCreateView).
    /// </summary>
    public ulong UsageNeededFor(in ImageViewDescription requested)
    {
        var viewFormat = MetalV2Formats.PixelFormat(requested.Format);
        SwizzleOf(requested.Mapping, out var identitySwizzle);
        return (viewFormat == Shape.PixelFormat || IsSrgbTwin(viewFormat)) && identitySwizzle
            ? 0
            : Mtl.TextureUsagePixelFormatView;
    }

    // The sRGB or linear variant of the texture's own format: Metal makes that view of a
    // texture without the pixel-format-view usage.
    private bool IsSrgbTwin(ulong viewFormat) =>
        viewFormat != Shape.PixelFormat &&
        MetalGuestFormats.LinearOf((MtlPixelFormat)(uint)viewFormat) ==
            MetalGuestFormats.LinearOf((MtlPixelFormat)(uint)Shape.PixelFormat);

    public bool IsGpuModified => _gpuModified;

    public bool IsWatched => WatchBegin != 0 && WatchEnd != 0;

    // The guest bytes the image covers; a view-only image has none. Telemetry only: the
    // budget counts AllocatedBytes, which is what the device really holds.
    public ulong AccountedSize => Texture == 0 ? 0 : (Description.Data.Size + 1023) & ~1023UL;

    /// <summary>
    /// MTLTexture.allocatedSize of the texture the image owns, asked once when the texture is
    /// made (padding, mip tails, D32S8 and substituted formats included); zero for a window
    /// view, which holds no memory of its own.
    /// </summary>
    public ulong AllocatedBytes;

    /// <summary>The tile backing of a sparse texture (MetalImageCache.Sparse.cs); null for an ordinary one.</summary>
    public MetalSparseResidency? Sparse;

    /// <summary>What the cache added to its total for this image when it registered it, taken back on removal.</summary>
    public ulong CountedBytes;

    // Why the next upload will happen; purely diagnostic.
    public MetalImageDirtyCause DirtyCause { get; private set; } = MetalImageDirtyCause.Created;

    /// <summary>
    /// Every cause that marked the image stale since it was last clean, as bits of 1 &lt;&lt; cause.
    /// <see cref="DirtyCause"/> keeps only the first, so a CPU-written image an alias hand-over
    /// later marks over another run still says CpuWrite there; the upload prelude reads this.
    /// </summary>
    public byte StaleCauses { get; private set; }

    /// <summary>
    /// A device copy (an alias or stencil copy from another texture) wrote this texture since its
    /// last whole refresh: the texture holds bytes guest memory may not, so the upload prelude never
    /// moves a whole upload of it.
    /// </summary>
    public bool DeviceCopiedSinceRefresh;

    /// <summary>The most stale runs, and the most published runs, an image holds apart with run sets on.</summary>
    public const int MaxRuns = 4;

    /// <summary>
    /// How many stale runs and how many published runs this image holds apart. One - a single
    /// merged interval each, as before - unless the cache admitted it with run sets on
    /// (SHARPEMU_METAL_V2_ALIAS_RUN_SETS), when it is <see cref="MaxRuns"/>.
    /// </summary>
    public int RunCapacity { get; set; } = 1;

    /// <summary>Where the cache counts what run sets do; null outside a cache.</summary>
    public MetalRunSetStats? RunStats;

    // Sorted, and none overlaps or touches the next.
    private readonly (ulong Start, ulong End)[] _stale = new (ulong, ulong)[MaxRuns];
    private int _staleCount;
    private readonly (ulong Start, ulong End)[] _published = new (ulong, ulong)[MaxRuns];
    private int _publishedCount;

    /// <summary>
    /// The guest bytes of this image, as offsets into it, that guest memory holds a newer
    /// version of than the texture does. With one run (run sets off) a second range that
    /// does not touch the first widens it to cover both, which costs an upload of the bytes
    /// between them and never an upload too few - for an image the texture owns no byte of.
    /// A GPU-written image holds bytes only its texture has between two runs, so its runs are
    /// kept apart, up to <see cref="RunCapacity"/>.
    /// </summary>
    public ReadOnlySpan<(ulong Start, ulong End)> StaleRuns => _stale.AsSpan(0, _staleCount);

    public int StaleRunCount => _staleCount;

    /// <summary>Where the first stale run starts: with one run, the stale interval.</summary>
    public ulong DirtyStart => _staleCount == 0 ? 0 : _stale[0].Start;

    /// <summary>Where the last stale run ends.</summary>
    public ulong DirtyEnd => _staleCount == 0 ? 0 : _stale[_staleCount - 1].End;

    /// <summary>
    /// The bytes already published to guest memory since this image was last written by
    /// the GPU. A partial publication leaves the texture authoritative for the rest, so
    /// the image stays GPU-modified and this is what keeps the next alias from publishing
    /// the same bytes again.
    /// </summary>
    public ReadOnlySpan<(ulong Start, ulong End)> PublishedRuns => _published.AsSpan(0, _publishedCount);

    // Adds [start, end) to sorted runs that neither overlap nor touch, absorbing every run it overlaps or touches,
    // into `into` (room for one more than the runs), and returns how many runs that makes.
    internal static int UnionRun(
        ReadOnlySpan<(ulong Start, ulong End)> runs, ulong start, ulong end, Span<(ulong Start, ulong End)> into)
    {
        var count = 0;
        var placed = false;
        foreach (var run in runs)
        {
            if (run.End < start)
            {
                into[count++] = run;
            }
            else if (run.Start > end)
            {
                if (!placed)
                {
                    into[count++] = (start, end);
                    placed = true;
                }

                into[count++] = run;
            }
            else
            {
                start = Math.Min(start, run.Start);
                end = Math.Max(end, run.End);
            }
        }

        if (!placed)
        {
            into[count++] = (start, end);
        }

        return count;
    }

    // Removes [start, end) from sorted runs into `into` (room for one more than the runs: a cut from the middle of
    // one makes two), and returns how many runs are left.
    internal static int SubtractRun(
        ReadOnlySpan<(ulong Start, ulong End)> runs, ulong start, ulong end, Span<(ulong Start, ulong End)> into)
    {
        var count = 0;
        foreach (var run in runs)
        {
            if (run.End <= start || run.Start >= end)
            {
                into[count++] = run;
                continue;
            }

            if (run.Start < start)
            {
                into[count++] = (run.Start, start);
            }

            if (run.End > end)
            {
                into[count++] = (end, run.End);
            }
        }

        return count;
    }

    private static bool Contains(ReadOnlySpan<(ulong Start, ulong End)> runs, ulong start, ulong end)
    {
        foreach (var run in runs)
        {
            if (run.Start <= start && end <= run.End)
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>True when the texture is stale over all of <c>[start, end)</c>: one run holds it.</summary>
    public bool IsStaleOver(ulong start, ulong end) => _cpuDirty && Contains(StaleRuns, start, end);

    /// <summary>True when the texture is stale over any byte of <c>[start, end)</c>.</summary>
    public bool StaleOverlaps(ulong start, ulong end)
    {
        foreach (var run in StaleRuns)
        {
            if (run.Start < end && run.End > start)
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Whether <c>[start, end)</c> can be marked stale without widening a run over bytes between
    /// two runs: the runs it does not overlap or touch, and it, fit in <see cref="RunCapacity"/>.
    /// </summary>
    public bool CanAddStaleRun(ulong start, ulong end)
    {
        if (!_cpuDirty)
        {
            return true;
        }

        var apart = 0;
        foreach (var run in StaleRuns)
        {
            if (run.End < start || run.Start > end)
            {
                apart++;
            }
        }

        return apart + 1 <= RunCapacity;
    }

    /// <summary>The run <c>[start, end)</c> becomes once marked stale: it and every run it overlaps or touches.</summary>
    public (ulong Start, ulong End) MergedStaleRun(ulong start, ulong end)
    {
        foreach (var run in StaleRuns)
        {
            if (run.End >= start && run.Start <= end)
            {
                start = Math.Min(start, run.Start);
                end = Math.Max(end, run.End);
            }
        }

        return (start, end);
    }

    /// <summary>
    /// Whether marking <c>[start, end)</c> current leaves none of it stale. A cut from the middle
    /// of a run makes two; with one run (or a full set) the run is kept whole instead
    /// (<see cref="RefreshComplete(ulong, ulong)"/>), so bytes that are not guest memory's must not
    /// be put there.
    /// </summary>
    public bool CanCutStale(ulong start, ulong end)
    {
        foreach (var run in StaleRuns)
        {
            if (run.Start < start && run.End > end)
            {
                return _staleCount < RunCapacity;
            }
        }

        return true;
    }

    /// <summary>Where the first published run starts.</summary>
    public ulong PublishedStart => _publishedCount == 0 ? 0 : _published[0].Start;

    /// <summary>Where the last published run ends.</summary>
    public ulong PublishedEnd => _publishedCount == 0 ? 0 : _published[_publishedCount - 1].End;

    public void MarkCpuDirty(MetalImageDirtyCause cause = MetalImageDirtyCause.Invalidated) =>
        MarkCpuDirty(cause, 0, Description.Data.Size);

    /// <summary>Guest memory is newer than the texture over <c>[start, end)</c> of the image.</summary>
    public void MarkCpuDirty(MetalImageDirtyCause cause, ulong start, ulong end)
    {
        // A window view reads guest memory in place, so guest memory is never newer.
        if (_windowLinearView)
        {
            return;
        }

        end = Math.Min(end, Description.Data.Size);
        if (end <= start)
        {
            return;
        }

        // Guest memory got newer bytes; the texture itself is unchanged until the upload, so
        // released chunks stay restorable from it (the upload's read restores what it covers).
        BumpContentVersion();
        StaleCauses = (byte)((_cpuDirty ? StaleCauses : 0) | (1 << (int)cause));
        if (!_cpuDirty)
        {
            DirtyCause = cause;
            _stale[0] = (start, end);
            _staleCount = 1;
            _cpuDirty = true;
            BindingVersion++;
            Epochs?.NoteBindingChanged();
            return;
        }

        if (Contains(StaleRuns, start, end))
        {
            return;
        }

        if (RunCapacity <= 1)
        {
            _stale[0] = (Math.Min(_stale[0].Start, start), Math.Max(_stale[0].End, end));
        }
        else
        {
            Span<(ulong Start, ulong End)> runs = stackalloc (ulong, ulong)[MaxRuns + 1];
            var count = UnionRun(StaleRuns, start, end, runs);
            if (count > RunCapacity)
            {
                count = MergeClosest(runs[..count]);
            }

            runs[..count].CopyTo(_stale);
            _staleCount = count;
            RunStats?.NoteSets(count);
        }

        BindingVersion++;
        Epochs?.NoteBindingChanged();
    }

    // One run too many: the two runs with the least between them become one. The bytes between them are marked
    // stale too, which is exact only while guest memory holds them for this image - always, when the texture owns
    // none of its bytes; for a GPU-written image the cache never lets it come to this (CanAddStaleRun), and if it
    // does anyway it is counted and warned about, since a refresh of the merged run replaces bytes only the
    // texture holds.
    private int MergeClosest(Span<(ulong Start, ulong End)> runs)
    {
        var closest = 0;
        for (var index = 1; index + 1 < runs.Length; index++)
        {
            if (runs[index + 1].Start - runs[index].End < runs[closest + 1].Start - runs[closest].End)
            {
                closest = index;
            }
        }

        runs[closest] = (runs[closest].Start, runs[closest + 1].End);
        runs[(closest + 2)..].CopyTo(runs[(closest + 1)..]);
        RunStats?.NoteMerged(unpublished: _gpuModified);
        if (_gpuModified)
        {
            MetalV2Diagnostics.WarnOnce(
                $"run-set-unpublished-merge:{Description.PixelFormat}",
                "A GPU-written image's stale runs were merged over bytes only its texture holds; its next refresh " +
                $"replaces them with older guest bytes: {Description.PixelFormat} {Description.Extent.Width}x" +
                $"{Description.Extent.Height} 0x{Description.Data.Address:X16}+0x{Description.Data.Size:X}.");
        }

        return runs.Length - 1;
    }

    /// <summary>The whole image is in the texture again.</summary>
    public void RefreshComplete()
    {
        NoteContentChanged();
        var wasDirty = _cpuDirty;
        _cpuDirty = false;
        _staleCount = 0;
        DeviceCopiedSinceRefresh = false;
        if (wasDirty)
        {
            BindingVersion++;
            Epochs?.NoteBindingChanged();
        }
    }

    /// <summary>Only <c>[start, end)</c> was uploaded; whatever is left stays dirty.</summary>
    public void RefreshComplete(ulong start, ulong end)
    {
        if (!_cpuDirty)
        {
            return;
        }

        NoteContentChanged();
        if (start <= DirtyStart && end >= DirtyEnd)
        {
            RefreshComplete();
            return;
        }

        if (RunCapacity > 1)
        {
            // A cut from the middle of a run makes two of it while the set has room; a full
            // set keeps that run whole, as one interval does (CanCutStale).
            Span<(ulong Start, ulong End)> runs = stackalloc (ulong, ulong)[MaxRuns + 1];
            var count = SubtractRun(StaleRuns, start, end, runs);
            if (count > RunCapacity ||
                (count == _staleCount && runs[..count].SequenceEqual(StaleRuns)))
            {
                return;
            }

            runs[..count].CopyTo(_stale);
            _staleCount = count;
            RunStats?.NoteSets(count);
            BindingVersion++;
            Epochs?.NoteBindingChanged();
            return;
        }

        // A cut from the front or the back narrows the interval; a cut from the middle
        // would split it, and one interval cannot hold two, so it keeps the wider half.
        var (dirtyStart, dirtyEnd) = _stale[0];
        if (start <= dirtyStart && end > dirtyStart)
        {
            dirtyStart = end;
        }
        else if (end >= dirtyEnd && start < dirtyEnd)
        {
            dirtyEnd = start;
        }
        else
        {
            return;
        }

        if (dirtyEnd <= dirtyStart)
        {
            RefreshComplete();
            return;
        }

        _stale[0] = (dirtyStart, dirtyEnd);
        BindingVersion++;
        Epochs?.NoteBindingChanged();
    }

    /// <param name="depthPlaneChanged">
    /// False for a depth-stencil target the GPU writes only the stencil of.
    /// </param>
    public void MarkGpuModified(bool depthPlaneChanged = true) => MarkGpuModified(depthPlaneChanged, keepStale: false);

    /// <summary>
    /// As <see cref="MarkGpuModified(bool)"/> for a write that touches no stale byte: every stale run stays as it
    /// is (MetalImageCache.TryKeepOtherLayersStaleLocked), so nothing about how the image binds moves with it.
    /// </summary>
    public void MarkGpuModifiedKeepingStale(bool depthPlaneChanged = true) =>
        MarkGpuModified(depthPlaneChanged, keepStale: true);

    private void MarkGpuModified(bool depthPlaneChanged, bool keepStale)
    {
        // Every write path gives a texture held from a later level the whole chain first
        // (MetalImageCache.PromoteLinearView): a writeback of a cut texture would retile its
        // levels into the wrong guest offsets.
        if (FirstLevel != 0)
        {
            throw GpuFatal.Fatal(
                $"A texture held from guest level {FirstLevel} was written without the whole chain: " +
                $"address=0x{Description.Data.Address:X16}.");
        }

        if (depthPlaneChanged)
        {
            NoteContentChanged();
        }
        else
        {
            NoteStencilChanged();
        }

        var wasModified = _gpuModified;
        var wasDirty = _cpuDirty;
        var wasPublished = _publishedCount != 0;
        // Claimed before the image is GPU-modified: the guest process never sees it so unclaimed.
        // Every stale run goes: a caller whose write leaves some of them stale marks those again
        // after this (MetalImageCache.CompleteDeferredLocked).
        ClaimPendingWriteback();
        _gpuModified = true;
        if (keepStale)
        {
            wasDirty = false;
        }
        else
        {
            _cpuDirty = false;
            _staleCount = 0;
        }

        _publishedCount = 0;
        GpuWriteSequence = Interlocked.Increment(ref _gpuWriteCounter);
        if (!wasModified)
        {
            BindingVersion++;
            Epochs?.NoteGpuModified();
        }
        else if (wasDirty || wasPublished)
        {
            BindingVersion++;
            Epochs?.NoteBindingChanged();
        }
    }

    /// <summary>Records that <c>[start, end)</c> of this image now also sits in guest memory.</summary>
    public void MarkPublished(ulong start, ulong end)
    {
        end = Math.Min(end, Description.Data.Size);
        if (end <= start || Contains(PublishedRuns, start, end))
        {
            return;
        }

        if (RunCapacity > 1)
        {
            // Runs with unpublished bytes between them stay apart: widening one over the gap
            // would claim bytes guest memory never got. A set one too full forgets its shortest
            // run, which only costs publishing it again.
            Span<(ulong Start, ulong End)> runs = stackalloc (ulong, ulong)[MaxRuns + 1];
            var count = UnionRun(PublishedRuns, start, end, runs);
            if (count > RunCapacity)
            {
                count = DropShortest(runs[..count]);
            }

            runs[..count].CopyTo(_published);
            _publishedCount = count;
        }
        else if (_publishedCount == 0)
        {
            _published[0] = (start, end);
            _publishedCount = 1;
        }
        else if (start > _published[0].End || end < _published[0].Start)
        {
            // Two runs with unpublished bytes between them: one interval cannot hold both,
            // and widening it over the gap would claim bytes guest memory never got. The
            // longer run is kept; forgetting the other only costs publishing it again.
            if (end - start <= _published[0].End - _published[0].Start)
            {
                return;
            }

            _published[0] = (start, end);
        }
        else
        {
            _published[0] = (Math.Min(_published[0].Start, start), Math.Max(_published[0].End, end));
        }

        BindingVersion++;
        Epochs?.NoteBindingChanged();
    }

    private static int DropShortest(Span<(ulong Start, ulong End)> runs)
    {
        var shortest = 0;
        for (var index = 1; index < runs.Length; index++)
        {
            // On a tie the later run goes, as one interval keeps the earlier half.
            if (runs[index].End - runs[index].Start <= runs[shortest].End - runs[shortest].Start)
            {
                shortest = index;
            }
        }

        runs[(shortest + 1)..].CopyTo(runs[shortest..]);
        return runs.Length - 1;
    }

    /// <summary>True when guest memory already holds this run of the image.</summary>
    public bool IsPublished(ulong start, ulong end) => Contains(PublishedRuns, start, end);

    /// <summary>
    /// The texture got bytes over <c>[start, end)</c> guest memory does not hold (a copy from an alias that still
    /// owes them): what was published there is not this image's any more. A cut from the middle of the one run
    /// (or of a full set) keeps the longer side; forgetting a run only costs publishing it again.
    /// </summary>
    public void ForgetPublished(ulong start, ulong end)
    {
        Span<(ulong Start, ulong End)> runs = stackalloc (ulong, ulong)[MaxRuns + 1];
        var count = SubtractRun(PublishedRuns, start, end, runs);
        if (count == _publishedCount && runs[..count].SequenceEqual(PublishedRuns))
        {
            return;
        }

        if (count > RunCapacity)
        {
            count = DropShortest(runs[..count]);
        }

        runs[..count].CopyTo(_published);
        _publishedCount = count;
        BindingVersion++;
        Epochs?.NoteBindingChanged();
    }

    public void ClearGpuModified()
    {
        var wasModified = _gpuModified;
        _gpuModified = false;
        ReleasePendingWriteback();
        _publishedCount = 0;
        if (wasModified)
        {
            BindingVersion++;
            Epochs?.NoteBindingChanged();
        }
    }

    // A CPU write inside the image's bytes makes it dirty; a write that only shares a
    // tracker page with it does too, since the page watch cannot tell them apart.
    public void InvalidateCpuWrite(ulong address, ulong size)
    {
        // The whole image is dirtied either way: the caller drops the page watch for the
        // image and a page fault reports a fixed small size, so later stores go unseen
        // until the next whole-image refresh.
        if (GuestRangeOverlap.Bytes(Description.Data.Address, Description.Data.Size, address, size)
            || GuestRangeOverlap.Pages(Description.Data.Address, Description.Data.Size, address, size))
        {
            MarkCpuDirty(MetalImageDirtyCause.CpuWrite);
            ClearGpuModified();
        }
    }

    private void ClaimPendingWriteback()
    {
        if (_pendingSize != 0 || _pendingWritebacks is not { } pending || ImageDescription.IsEmptyRange(Description.Data))
        {
            return;
        }

        _pendingAddress = Description.Data.Address;
        _pendingSize = Description.Data.Size;
        pending.Add(_pendingAddress, _pendingSize);
    }

    private void ReleasePendingWriteback()
    {
        if (_pendingSize == 0)
        {
            return;
        }

        _pendingWritebacks!.Remove(_pendingAddress, _pendingSize);
        _pendingSize = 0;
    }

    /// <summary>
    /// A guest byte range intersected with this image, as offsets into the image. Empty
    /// when the range does not reach it.
    /// </summary>
    public (ulong Start, ulong End) OverlapWithin(ulong address, ulong size)
    {
        var imageStart = Description.Data.Address;
        var imageEnd = imageStart + Description.Data.Size;
        var rangeEnd = size > ulong.MaxValue - address ? ulong.MaxValue : address + size;
        var start = Math.Max(imageStart, address);
        var end = Math.Min(imageEnd, rangeEnd);
        return end > start ? (start - imageStart, end - imageStart) : (0, 0);
    }

    public bool Overlaps(ulong address, ulong size, bool pages = false) => pages
        ? GuestRangeOverlap.Pages(Description.Data.Address, Description.Data.Size, address, size)
        : GuestRangeOverlap.Bytes(Description.Data.Address, Description.Data.Size, address, size);

    public bool GpuOverlaps(ulong address, ulong size) => _gpuModified && Overlaps(address, size);

    public bool SupportsViewType(in ImageViewDescription view)
    {
        var type = Shape.TextureType;
        return view.Type switch
        {
            GpuImageViewType.Type3D => type == Mtl.TextureType3D,
            GpuImageViewType.Type1D or GpuImageViewType.Type1DArray =>
                type is Mtl.TextureType1D or Mtl.TextureType1DArray,
            // A volume serves a flat view too: Vulkan makes a 2D-array view of a 3D
            // image to render it slice by slice, and Metal names the slice with the
            // attachment's depth plane. Refusing it would drop and recreate the volume
            // on every acquisition - the description makes a 3D texture again - which
            // is a 24 MB texture released out from under the submissions still reading
            // it, every pass of a title's volumetric fog.
            _ => type is Mtl.TextureType2D or Mtl.TextureType2DArray or Mtl.TextureTypeCube or
                Mtl.TextureType2DMultisample or Mtl.TextureType2DMultisampleArray or Mtl.TextureType3D,
        };
    }

    /// <summary>
    /// The texture a shader or attachment binds for one view. A view over the whole
    /// image with its own format reuses the image texture; anything narrower (a mip,
    /// a slice range, a reinterpreted format) gets an MTLTexture view, which is a
    /// handle onto the same allocation and never copies.
    /// </summary>
    // The guest T#'s DST_SEL, which the Vulkan backend puts on its image view's
    // component mapping. Without it a single-channel font atlas samples as
    // (r, 0, 0, 1), so a shader that reads the glyph coverage out of .a gets 1
    // everywhere and every glyph quad fills in solid.
    internal static MtlTextureSwizzleChannels SwizzleOf(in GpuComponentMapping mapping, out bool identity)
    {
        var red = Channel(mapping.R, MtlTextureSwizzle.Red);
        var green = Channel(mapping.G, MtlTextureSwizzle.Green);
        var blue = Channel(mapping.B, MtlTextureSwizzle.Blue);
        var alpha = Channel(mapping.A, MtlTextureSwizzle.Alpha);
        identity = red == (byte)MtlTextureSwizzle.Red && green == (byte)MtlTextureSwizzle.Green &&
            blue == (byte)MtlTextureSwizzle.Blue && alpha == (byte)MtlTextureSwizzle.Alpha;
        return new MtlTextureSwizzleChannels { Red = red, Green = green, Blue = blue, Alpha = alpha };
    }

    private static byte Channel(GpuComponentSwizzle component, MtlTextureSwizzle identity) => (byte)(component switch
    {
        GpuComponentSwizzle.Zero => MtlTextureSwizzle.Zero,
        GpuComponentSwizzle.One => MtlTextureSwizzle.One,
        GpuComponentSwizzle.R => MtlTextureSwizzle.Red,
        GpuComponentSwizzle.G => MtlTextureSwizzle.Green,
        GpuComponentSwizzle.B => MtlTextureSwizzle.Blue,
        GpuComponentSwizzle.A => MtlTextureSwizzle.Alpha,
        _ => identity,
    });

    public nint GetOrCreateView(in ImageViewDescription requested) => GetOrCreateView(requested, out _);

    // The once-only warnings of GetOrCreateView are keyed by strings formatted from the request, and the
    // keys were formatted on every bind though only the first of each could report. An image remembers
    // the request of the last warning it asked about: asked again with the same one, the answer is
    // known. A benign race with the flip path costs a repeated ask, which the global set still answers.
    private (ulong Format, GpuComponentSwizzle R, GpuComponentSwizzle G, GpuComponentSwizzle B, GpuComponentSwizzle A, bool Set) _swizzleWarning;
    private (ulong Format, ulong ViewFormat, ulong Type, ulong ViewType, bool Set) _reinterpretWarning;

    private bool SwizzleWarningAsked(in GpuComponentMapping mapping)
    {
        if (!SharpEmu.ShaderCompiler.WorkerScratch.Enabled)
        {
            return false;
        }

        var asked = ((ulong)Shape.PixelFormat, mapping.R, mapping.G, mapping.B, mapping.A, true);
        if (_swizzleWarning.Equals(asked))
        {
            return true;
        }

        _swizzleWarning = asked;
        return false;
    }

    private bool ReinterpretWarningAsked(ulong viewFormat, ulong viewType)
    {
        if (!SharpEmu.ShaderCompiler.WorkerScratch.Enabled)
        {
            return false;
        }

        var asked = ((ulong)Shape.PixelFormat, viewFormat, (ulong)Shape.TextureType, viewType, true);
        if (_reinterpretWarning.Equals(asked))
        {
            return true;
        }

        _reinterpretWarning = asked;
        return false;
    }

    // The gpuResourceID of the backing texture; it is fixed for a given MTLTexture, so it is
    // asked once per handle rather than once per bind.
    private ulong TextureResourceId()
    {
        if (_textureIdHandle != Texture)
        {
            _textureId = MetalNative.SendGpuResourceId(Texture, SelGpuResourceId);
            _textureIdHandle = Texture;
        }

        return _textureId;
    }

    /// <summary>The view for a request, with its cached gpuResourceID.</summary>
    public nint GetOrCreateView(in ImageViewDescription requested, out ulong resourceId)
    {
        using var phaseScope = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.ImageViewLookup);
        resourceId = 0;
        if (Texture == 0)
        {
            throw GpuFatal.Fatal(
                $"An image without a texture cannot make a view: address=0x{Description.Data.Address:X16}.");
        }

        var viewFormat = MetalV2Formats.PixelFormat(requested.Format);
        if (Shape.PixelFormat == MtlFormat.Depth32FloatStencil8 && viewFormat == MtlFormat.R8Uint)
        {
            // The stencil plane read as 8-bit integers (MetalImageCache.StencilTwin.cs).
            viewFormat = MtlFormat.X32Stencil8;
        }

        var viewType = MetalV2Formats.ViewTextureType(requested.Type, Shape.SampleCount);
        var swizzle = SwizzleOf(requested.Mapping, out var identitySwizzle);
        // MTLTextureUsagePixelFormatView is only needed to reinterpret the pixel
        // format or the swizzle. Narrowing a view to a level or a slice of the
        // same format needs nothing, so a depth or compressed image -- which the
        // cache creates without that usage -- can still have the subresource view
        // a mipped or layered attachment binds.
        var viewable = (Shape.Usage & Mtl.TextureUsagePixelFormatView) != 0;
        if (viewable && MetalGuestFormats.HasSrgbTwin((MtlPixelFormat)Shape.PixelFormat))
        {
            // A compressed texture has the view usage only for its sRGB/linear twin (see
            // MetalImageTransfer.UsageOf); any other format stays unviewed, as before, since
            // Metal raises on a reinterpretation it does not allow. The twin keeps its swizzle.
            viewable = viewFormat == Shape.PixelFormat ||
                MetalGuestFormats.AreSrgbTwins((MtlPixelFormat)Shape.PixelFormat, (MtlPixelFormat)viewFormat);
        }

        // A view that keeps the texture's format may swizzle without the usage (checked under
        // API validation for depth, BC1/BC4/BC5 and R8): a BC4 mask whose T# replicates R reads
        // zero in G and B otherwise, which blanks every material layer a shader masks with .g.
        if (!viewable && !identitySwizzle && viewFormat != Shape.PixelFormat)
        {
            // Metal raises on a swizzled reinterpretation of a texture without the usage, so the view
            // reads the components in the texture's own order, and that is reported. The report's
            // key is formatted on every bind, so this image remembers the one it last asked about.
            if (!SwizzleWarningAsked(requested.Mapping))
            {
                MetalV2Diagnostics.WarnOnce(
                    $"view-swizzle:{Shape.PixelFormat}:{requested.Mapping}",
                    $"A texture without the pixel-format-view usage cannot swizzle; the view keeps the " +
                    $"texture's component order: format={Shape.PixelFormat} mapping={requested.Mapping} " +
                    $"usage=0x{Shape.Usage:X} address=0x{Description.Data.Address:X16}.");
            }

            identitySwizzle = true;
        }

        // A volume rendered slice by slice asks for the 2D-array view Vulkan makes of a
        // 3D image. Metal has no such view and raises rather than returning nil, but it
        // does not need one: a render pass names the slice of a 3D attachment with its
        // depth plane, which the attachment already carries, so the volume binds whole.
        if (Shape.TextureType == Mtl.TextureType3D && viewType != Mtl.TextureType3D)
        {
            if ((requested.Usage & (GpuImageUsage.ColorAttachmentBit | GpuImageUsage.DepthStencilAttachmentBit)) == 0)
            {
                // Read as something other than a volume, the binding is wrong either
                // way; being wrong is better than raising, which ends the process.
                MetalV2Diagnostics.WarnOnce(
                    $"volume-view:{viewType}",
                    $"A volume cannot be viewed as another texture type; binding the volume instead: " +
                    $"type=3D->{viewType} format={Shape.PixelFormat}->{viewFormat} " +
                    $"layers={requested.BaseLayer}+{requested.LayerCount} usage={requested.Usage}.");
            }

            resourceId = TextureResourceId();
            return Texture;
        }

        // Nor can a flat texture be viewed as a volume: a binding whose descriptor is a volume over an image the
        // cache holds flat is wrong either way, and Metal raises on the view.
        if (Shape.TextureType != Mtl.TextureType3D && viewType == Mtl.TextureType3D)
        {
            MetalV2Diagnostics.WarnOnce(
                $"flat-volume-view:{Shape.TextureType}",
                $"A flat texture cannot be viewed as a volume; binding the texture instead: type={Shape.TextureType}->3D " +
                $"format={Shape.PixelFormat}->{viewFormat} extent={Description.Extent.Width}x{Description.Extent.Height}x" +
                $"{Description.Extent.Depth} address=0x{Description.Data.Address:X16}.{Environment.NewLine}{Environment.StackTrace}");
            resourceId = TextureResourceId();
            return Texture;
        }

        // LOD remapping for a texture held from a later level. A request names guest levels:
        // its BaseLevel is the T# BASE_LEVEL, which on the PS5 is a view base, not a clamp -
        // the texture unit treats level BASE_LEVEL as the base map, so implicit LOD, an
        // explicit LOD (sample_l, image_load_mip), resinfo's width/height for a LOD and its
        // level count (LAST_LEVEL - BASE_LEVEL + 1) are all relative to it, and so is the
        // sampler's MIN_LOD. The translated shader therefore never sees an absolute level:
        // it samples a view whose level 0 is BASE_LEVEL, and Metal's own level(), bias(),
        // min_lod_clamp, get_width(lod) and get_num_mip_levels() are relative to the view
        // exactly the same way. A texture whose level 0 is guest level FirstLevel only moves
        // where that view starts, to BASE_LEVEL - FirstLevel; the shader, the sampler and
        // every explicit-LOD fetch stay as they are. The cache never hands out a view below
        // FirstLevel (FitResidentLevelsLocked grows the texture first), so the clamp below
        // only keeps a broken caller from making Metal raise.
        var textureLevel = requested.BaseLevel >= FirstLevel ? requested.BaseLevel - FirstLevel : 0;
        if (requested.BaseLevel < FirstLevel)
        {
            MetalV2Diagnostics.WarnOnce(
                $"view-below-first-level:{Description.Data.Address:X}",
                $"A view names a guest level the texture does not hold; it starts at the texture's first level " +
                $"instead: level={requested.BaseLevel} firstLevel={FirstLevel} address=0x{Description.Data.Address:X16}.");
        }

        var wholeImage = textureLevel == 0 && requested.LevelCount >= Shape.MipLevels &&
            requested.BaseLayer == 0 && requested.LayerCount >= Shape.ArrayLength;
        if (wholeImage && identitySwizzle && viewFormat == Shape.PixelFormat && viewType == Shape.TextureType)
        {
            resourceId = TextureResourceId();
            return Texture;
        }

        lock (_viewGate)
        {
            for (var index = 0; index < _views.Count; index++)
            {
                if (_views[index].Description.Equals(requested))
                {
                    resourceId = _views[index].ResourceId;
                    return _views[index].Texture;
                }
            }
        }

        if (!viewable && viewFormat != Shape.PixelFormat && !IsSrgbTwin(viewFormat))
        {
            // Metal raises rather than returning nil for a reinterpreting view of
            // a texture that was not created for one, and a raised ObjC exception
            // on the GPU worker takes the process down. A compressed image is the
            // usual case: it is sampled whole, in its own format, so binding it
            // unviewed is what the draw wanted anyway.
            if (!ReinterpretWarningAsked(viewFormat, viewType))
            {
                MetalV2Diagnostics.WarnOnce(
                    $"view-usage:{Shape.PixelFormat}->{viewFormat}:{Shape.TextureType}->{viewType}",
                    $"A texture without the pixel-format-view usage cannot be reinterpreted; binding the image instead: " +
                    $"format={Shape.PixelFormat}->{viewFormat} type={Shape.TextureType}->{viewType} " +
                    $"usage=0x{Shape.Usage:X} levels={requested.BaseLevel}+{requested.LevelCount} " +
                    $"layers={requested.BaseLayer}+{requested.LayerCount}.");
            }

            resourceId = TextureResourceId();
            return Texture;
        }

        textureLevel = Math.Min(textureLevel, Shape.MipLevels - 1);
        var levelCount = Math.Min(requested.LevelCount, Shape.MipLevels - textureLevel);
        var baseLayer = Math.Min(requested.BaseLayer, Shape.ArrayLength - 1);
        var layerCount = Math.Min(requested.LayerCount, Shape.ArrayLength - baseLayer);
        if (viewType == Mtl.TextureType3D)
        {
            // A volume has one slice; its depth is not an array dimension.
            baseLayer = 0;
            layerCount = 1;
        }

        var levelRange = new NsRange { Location = textureLevel, Length = Math.Max(levelCount, 1) };
        var sliceRange = new NsRange { Location = baseLayer, Length = Math.Max(layerCount, 1) };
        var view = identitySwizzle
            ? MetalV2Native.SendNewTextureView(
                Texture, SelNewTextureView, (nuint)viewFormat, (nuint)viewType, levelRange, sliceRange)
            : MetalV2Native.SendNewSwizzledTextureView(
                Texture, SelNewSwizzledTextureView, (nuint)viewFormat, (nuint)viewType, levelRange, sliceRange, swizzle);
        if (view == 0)
        {
            // A refused reinterpretation is not fatal for a texture: the image itself
            // still carries the bytes, so the draw samples the unreinterpreted texture.
            MetalV2Diagnostics.WarnOnce(
                $"view:{Shape.PixelFormat}->{viewFormat}:{Shape.TextureType}->{viewType}",
                $"A texture view was refused; binding the image instead: format={Shape.PixelFormat}->{viewFormat} " +
                $"type={Shape.TextureType}->{viewType} levels={requested.BaseLevel}+{levelCount} layers={baseLayer}+{layerCount}.");
            resourceId = TextureResourceId();
            return Texture;
        }

        lock (_viewGate)
        {
            // Another thread may have made the same view while this one was
            // asking Metal for it; keep theirs so the list holds one of each.
            for (var index = 0; index < _views.Count; index++)
            {
                if (_views[index].Description.Equals(requested))
                {
                    MetalNative.SendVoid(view, SelRelease);
                    resourceId = _views[index].ResourceId;
                    return _views[index].Texture;
                }
            }

            resourceId = MetalNative.SendGpuResourceId(view, SelGpuResourceId);
            _views.Add((requested, view, resourceId));
        }

        return view;
    }

    /// <summary>Replaces the backing texture, dropping every view of the old one.</summary>
    public void ReplaceTexture(nint texture, in MetalTextureShape shape)
    {
        ReleaseViews();
        // The old texture's tiler view goes before the texture does, so a texture made
        // later at the same handle never finds it.
        var old = Texture;
        Texture = texture;
        if (old != 0)
        {
            MetalNative.SendVoid(old, SelRelease);
        }

        Shape = shape;
        Sparse = null;
        NoteContentChanged();
        BindingVersion++;
        Epochs?.NoteShapeChanged();
    }

    private void ReleaseViews()
    {
        lock (_viewGate)
        {
            foreach (var (_, view, _) in _views)
            {
                MetalNative.SendVoid(view, SelRelease);
            }

            _views.Clear();
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        ReleaseViews();
        var texture = Texture;
        Texture = 0;
        if (texture != 0)
        {
            MetalNative.SendVoid(texture, SelRelease);
        }
    }
}
