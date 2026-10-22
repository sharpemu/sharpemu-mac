// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Scheduling;

namespace SharpEmu.Gpu.Metal.V2;

// Registration in the page owner index, page watches, CPU-write invalidation,
// eviction under memory pressure, and the surface metadata (DCC and HTile) state.
internal sealed partial class MetalImageCache
{
    private void AddToIndex(ResourceSlotIdentifier imageIdentifier)
    {
        var image = _slots[imageIdentifier];
        if (image.Registered)
        {
            return;
        }

        if (!MetalImagePageOwners.TryGetPageRange(
                image.Description.Data.Address, image.Description.Data.Size, out _, out _))
        {
            throw GpuFatal.Fatal(
                $"The image registration is outside the guest address space: address=0x{image.Description.Data.Address:X16} " +
                $"size=0x{image.Description.Data.Size:X}.");
        }

        // Counted before anything can find the image or arm its watch: a CPU write that
        // reads no owner without the lock must never have an image over its bytes.
        _pageOwners.Add(image.Description.Data.Address, image.Description.Data.Size, imageIdentifier);
        // Chunks another texture holds the guest copy of are no longer that texture's alone.
        SingleCopy?.OnImageRegistered(image.Description.Data.Address, image.Description.Data.Size);
        if (image.Description.HasStencil)
        {
            _stencilOwners.Add(imageIdentifier);
        }

        image.Registered = true;
        image.CountedBytes = image.AllocatedBytes;
        _totalUsedMemory += image.CountedBytes;
    }

    // The image's texture was replaced (window view promoted, usage widened): the total follows
    // the new texture's real size. Caller holds the lock.
    private void RecountAllocatedLocked(MetalCachedImage image)
    {
        image.AllocatedBytes = TextureAllocatedBytes(image);
        if (!image.Registered)
        {
            return;
        }

        _totalUsedMemory -= Math.Min(image.CountedBytes, _totalUsedMemory);
        image.CountedBytes = image.AllocatedBytes;
        _totalUsedMemory += image.CountedBytes;
    }

    private void RemoveFromIndex(ResourceSlotIdentifier imageIdentifier)
    {
        var image = _slots.TryGet(imageIdentifier);
        if (image is null || !image.Registered)
        {
            return;
        }

        // The watch goes first and the count last, once no lookup can find the image.
        UnwatchImage(imageIdentifier);
        _pageOwners.Remove(image.Description.Data.Address, image.Description.Data.Size, imageIdentifier);
        _stencilOwners.Remove(imageIdentifier);

        _totalUsedMemory -= Math.Min(image.CountedBytes, _totalUsedMemory);
        image.CountedBytes = 0;
        image.Registered = false;
    }

    // Publishes a GPU-written image back into guest memory, then drops it. Guest
    // memory is the single source of truth, so nothing is lost by dropping a texture.
    private void ReleaseImage(ResourceSlotIdentifier imageIdentifier, MetalImageReleaseCause cause)
    {
        _releaseCauses[(int)cause]++;
        if (_slots.TryGet(imageIdentifier) is not { } image)
        {
            return;
        }

        Interlocked.Increment(ref _imageSetVersion);
        NoteReleasedForTelemetry(image, cause);
        // Chunks whose guest copy the texture holds get their bytes back before it goes.
        SingleCopy?.OnImageReleased(image, cause);

        ForgetDeferredRefreshesLocked(image);

        if (image.IsWindowLinearView)
        {
            Interlocked.Decrement(ref _linearViews);
        }

        if (image.IsGpuModified)
        {
            WriteBackToGuest(image, MetalWritebackCause.Release);
            // The retile reads the texture in the command buffer being recorded: the texture
            // outlives that submission whatever bound it last.
            image.NoteUse(_owner.CurrentSerial);
        }

        if (image.Description.HasMetadata)
        {
            _surfaceMetadata.Remove(image.Description.Metadata.Range.Address);
        }

        ForgetImageBookkeepingLocked(imageIdentifier, image);
        NoteDroppedLocked(image, cause);
        RemoveFromIndex(imageIdentifier);
        _slots.Erase(imageIdentifier);
        RetireImage(image);
        image.Epochs = null;
        Epochs.NoteBindingChanged();
    }

    // What the cache keeps about an image outside its slot: claims on its DCC slices, clears
    // deferred for it, and a pending linear publication. Keyed by the slot identifier, they
    // would otherwise outlive it until the metadata happened to be rewritten.
    private void ForgetImageBookkeepingLocked(ResourceSlotIdentifier imageIdentifier, MetalCachedImage image)
    {
        _unpublishedLinearWrites.Remove(image);
        if (_materializedDccSlices.Count != 0)
        {
            var removed = _materializedDccSlices.RemoveWhere(entry =>
            {
                if (entry.Image != imageIdentifier)
                {
                    return false;
                }

                UnindexDccClaim(entry.Address);
                UnstandDccClaimLocked(entry.Image, entry.Address);
                return true;
            });
            if (removed != 0)
            {
                Epochs.NoteDccChanged();
            }
        }

        if (_deferredDccSlices.Count != 0)
        {
            List<(ResourceSlotIdentifier, ulong)>? stale = null;
            foreach (var key in _deferredDccSlices.Keys)
            {
                if (key.Image == imageIdentifier)
                {
                    (stale ??= []).Add(key);
                }
            }

            if (stale is not null)
            {
                foreach (var key in stale)
                {
                    _deferredDccSlices.Remove(key);
                }

                Epochs.NoteDccChanged();
            }
        }
    }

    // A command buffer names a texture through an argument buffer, which holds a
    // gpuResourceID that Metal does not retain: releasing the texture while a
    // recorded-but-unfinished submission still names it leaves the GPU driver
    // messaging freed memory. So an evicted image's Metal objects wait for the
    // submission that last bound them, and the slot is free either way.
    private void RetireImage(MetalCachedImage image)
    {
        if (image.LastUseSerial <= _owner.CompletedSerial)
        {
            image.Dispose();
            return;
        }

        _retiring.Add((image.LastUseSerial, image));
    }

    // Drops the Metal objects of every evicted image the GPU has finished with.
    private void DrainRetiredImages()
    {
        if (_retiring.Count == 0)
        {
            return;
        }

        var completed = _owner.CompletedSerial;
        var kept = 0;
        for (var index = 0; index < _retiring.Count; index++)
        {
            var entry = _retiring[index];
            if (entry.Serial <= completed)
            {
                entry.Image.Dispose();
                continue;
            }

            _retiring[kept++] = entry;
        }

        _retiring.RemoveRange(kept, _retiring.Count - kept);
    }

    private void WatchImage(ResourceSlotIdentifier imageIdentifier) => WatchImageLocked(_slots[imageIdentifier]);

    private void WatchImageLocked(MetalCachedImage image)
    {
        // A window view reads guest memory in place: a store to it needs no notice.
        if (_watchingStopped || !image.Registered || image.IsWindowLinearView ||
            ImageDescription.IsEmptyRange(image.Description.Data))
        {
            return;
        }

        // The bytes the texture's levels read: the whole image, or for a texture held from
        // its BASE_LEVEL only the coarser levels', so the guest reusing the finer levels'
        // pages costs this image nothing.
        var begin = image.Description.Data.Address + image.ResidentStart;
        var end = image.Description.Data.Address + image.ResidentEnd;
        if (begin == image.WatchBegin && end == image.WatchEnd)
        {
            return;
        }

        if (image.IsWatched)
        {
            UnwatchImageLocked(image);
        }

        image.WatchBegin = begin;
        image.WatchEnd = end;
        _pages.AddWatch(begin, end - begin, blockReads: false);
    }

    private void UnwatchImage(ResourceSlotIdentifier imageIdentifier)
    {
        if (_slots.TryGet(imageIdentifier) is { } image)
        {
            UnwatchImageLocked(image);
        }
    }

    private void UnwatchImageLocked(MetalCachedImage image)
    {
        if (!image.IsWatched)
        {
            return;
        }

        var address = image.WatchBegin;
        var size = image.WatchEnd - image.WatchBegin;
        image.WatchBegin = 0;
        image.WatchEnd = 0;
        if (size != 0)
        {
            _pages.RemoveWatch(address, size, blockReads: false);
        }
    }

    // Caller holds the lock; the query epoch marks each image at most once.
    private ImageQueryResults FindImagesInRange(ulong address, ulong size, bool pageOverlap)
    {
        var result = new ImageQueryResults();
        if (!MetalImagePageOwners.TryGetPageRange(address, size, out var first, out var lastExclusive))
        {
            return result;
        }

        var queryEpoch = ++_queryEpoch;
        if (queryEpoch == 0)
        {
            _slots.ForEach(static (_, image) => image.QueryEpoch = 0);
            queryEpoch = ++_queryEpoch;
        }

        for (var page = first; page < lastExclusive; page++)
        {
            var owners = _pageOwners.Find(page);
            if (owners is null)
            {
                continue;
            }

            for (var index = 0; index < owners.Count; index++)
            {
                var imageIdentifier = owners[index];
                var image = _slots.TryGet(imageIdentifier);
                if (image is null || image.QueryEpoch == queryEpoch)
                {
                    continue;
                }

                image.QueryEpoch = queryEpoch;
                if (image.Overlaps(address, size, pageOverlap))
                {
                    result.Add(imageIdentifier);
                }
            }
        }

        return result;
    }

    // A CPU write on the faulting thread. Most writes (command building, labels)
    // touch no image page, and the live-owner counts answer those without the lock.
    bool IGuestImageStore.MarkCpuWrite(ulong address, ulong size)
    {
        if (!IsValidRange(address, size))
        {
            return false;
        }

        // A writeback still in flight over the page lands first; the store waits for it.
        var fenced = Volatile.Read(ref _fencedWritebackCount) != 0 && WaitForFencedWritebacks(address, size);

        // A cached staged copy over the page is stale from here; the store may land once its watch is gone.
        var stagedHit = StagedRanges?.OnCpuWrite(address, size) ?? false;
        var mayOwn = _pageOwners.MayHaveOwners(address, size);
        if (!mayOwn && !MetadataWatchOverlaps(address, size))
        {
            return stagedHit | fenced;
        }

        CountLockedCpuWrite(mayOwn);
        if (mayOwn)
        {
            // A GPU-written image's bytes reach guest memory before the store lands over them
            // (MetalImageCache.CpuStores.cs).
            PublishBeforeCpuStore(address, size);
        }

        using var held = _lock.Hold();
        var metadataHit = OnMetadataCpuWriteLocked(address, size);
        var imageHit = mayOwn && InvalidateAliases(address, size);
        CountOwnerCpuWriteOutcome(mayOwn, imageHit);
        return imageHit | metadataHit | stagedHit | fenced;
    }

    // A managed write of exactly these bytes. Unlike a fault it names the bytes it
    // changes, so an image is marked stale over those bytes alone.
    bool IGuestImageStore.MarkPreciseCpuWrite(ulong address, ulong size)
    {
        if (!IsValidRange(address, size))
        {
            return false;
        }

        var stagedHit = false;
        if (StagedRanges is { } staged && staged.MayCover(address, size))
        {
            stagedHit = staged.OnPreciseCpuWrite(address, size, _guestMemory.IsBacked(address, size));
        }

        var mayOwn = _pageOwners.MayHaveOwners(address, size);
        if (!mayOwn && !MetadataWatchOverlaps(address, size))
        {
            return stagedHit;
        }

        CountLockedCpuWrite(mayOwn);
        var backed = mayOwn && _guestMemory.IsBacked(address, size);
        using var held = _lock.Hold();
        var metadataHit = OnMetadataCpuWriteLocked(address, size);
        var imageHit = mayOwn && InvalidatePreciseWrite(address, size, backed);
        CountOwnerCpuWriteOutcome(mayOwn, imageHit);
        return imageHit | metadataHit | stagedHit;
    }

    /// <summary>The host wrote exactly <c>[address, address + size)</c> of guest memory.</summary>
    public void InvalidateMemory(ulong address, ulong size)
    {
        if (!IsValidRange(address, size))
        {
            return;
        }

        StagedRanges?.OnHostWrite(address, size);
        var backed = _guestMemory.IsBacked(address, size);
        using var held = _lock.Hold();
        InvalidatePreciseWrite(address, size, backed);
    }

    // A write of known bytes through the backing never meets the page watch, so the
    // watch can stay and keeps catching the guest's own stores to the rest of the image:
    // a watched image that holds nothing the GPU wrote is stale over the written bytes
    // and nowhere else, and an image that only shares a page with them is not stale at
    // all. Re-uploading the whole image instead - and dropping its watch, to take it
    // again on the next use - is what a packet-sized host fill next to a texture cost,
    // hundreds of times a frame, each upload ending the render pass it fell in.
    // A write that goes through the guest mapping needs the watch gone first, and an
    // image already without one, or with bytes only its texture holds, is handled as a
    // fault is: the whole image is refreshed.
    private bool InvalidatePreciseWrite(ulong address, ulong size, bool backed)
    {
        if (!backed)
        {
            return InvalidateAliases(address, size);
        }

        var covered = false;
        foreach (var imageIdentifier in FindImagesInRange(address, size, pageOverlap: false))
        {
            if (_slots.TryGet(imageIdentifier) is not { } image)
            {
                continue;
            }

            covered = true;
            var (start, end) = image.OverlapWithin(address, size);
            if (image.IsWatched && !image.IsGpuModified)
            {
                image.MarkCpuDirty(MetalImageDirtyCause.CpuWrite, start, end);
                continue;
            }

            image.InvalidateCpuWrite(address, size);
            UnwatchImage(imageIdentifier);
        }

        return covered;
    }

    private bool InvalidateAliases(ulong address, ulong size)
    {
        var covered = false;
        foreach (var imageIdentifier in FindImagesInRange(address, size, pageOverlap: true))
        {
            if (_slots.TryGet(imageIdentifier) is not { } image)
            {
                continue;
            }

            covered = true;
            image.InvalidateCpuWrite(address, size);
            UnwatchImage(imageIdentifier);
        }

        return covered;
    }

    // The guest unmapped the range: drop its metadata and its images without a writeback.
    void IGuestImageStore.Unregister(ulong address, ulong size)
    {
        if (!IsValidRange(address, size))
        {
            return;
        }

        StagedRanges?.OnUnmap(address, size);
        _guestMemory.OnGuestUnmap(address, size);
        SingleCopy?.OnGuestUnmap(address, size);
        using var held = _lock.Hold();
        var end = address + size;
        var stale = new List<ulong>();
        foreach (var metadataAddress in _surfaceMetadata.Keys)
        {
            if (metadataAddress >= address && metadataAddress < end)
            {
                stale.Add(metadataAddress);
            }
        }

        foreach (var metadataAddress in stale)
        {
            _surfaceMetadata.Remove(metadataAddress);
        }

        foreach (var imageIdentifier in FindImagesInRange(address, size, pageOverlap: false))
        {
            if (_slots.TryGet(imageIdentifier) is not { } image)
            {
                continue;
            }

            // The pages are gone; there is nowhere to write the contents back to.
            image.ClearGpuModified();
            ReleaseImage(imageIdentifier, MetalImageReleaseCause.Unmapped);
        }
    }

    /// <summary>
    /// Releases every page watch and takes no new ones. The runtime frees the page
    /// guard as soon as the guest has left, and it refuses to be freed while a watch
    /// is outstanding, so this runs from the shutting-down thread rather than waiting
    /// for the GPU worker to notice.
    /// </summary>
    public void StopWatching()
    {
        StagedRanges?.StopWatching();
        SingleCopy?.StopWatching();
        using var held = _lock.Hold();
        _watchingStopped = true;
        ReleaseAllFencedWritebacks();
        foreach (var (watchBegin, watchSize) in _metadataWatches)
        {
            _pages.RemoveWatch(watchBegin, watchSize, blockReads: false);
        }

        _metadataWatches.Clear();
        PublishMetadataWatchesLocked();
        var watched = new List<ResourceSlotIdentifier>();
        _slots.ForEach((imageIdentifier, image) =>
        {
            if (image.IsWatched)
            {
                watched.Add(imageIdentifier);
            }
        });
        foreach (var imageIdentifier in watched)
        {
            UnwatchImage(imageIdentifier);
        }
    }

    /// <summary>Marks a presented frame: images used since the previous mark stay resident.</summary>
    public void MarkFrame()
    {
        using var held = _lock.Hold();
        _previousFrameStartTick = _frameStartTick;
        _frameStartTick = _collectionTick;
        _idleClock.MarkFrame(_collectionTick);
    }

    /// <summary>
    /// Advances the collection tick, re-reads the budget when it is due, retires images past
    /// the retirement age and evicts stale images over the budget.
    /// </summary>
    public void Collect()
    {
        using var held = _lock.Hold();
        _collectionTick++;
        DrainRetiredImages();
        PruneFencedWritebacks();
        var now = _clock();
        _idleClock.Sample(now, _collectionTick);
        MaybeEvaluateBudgetLocked(now);
        HousekeepLocked(now);
        CollectIfOverBudget(now);
        SingleCopy?.Tick(this, now);
    }

    private void CollectIfOverBudget() => CollectIfOverBudget(_clock());

    // Eviction under memory pressure: the least recently used images outside the working set
    // go first, and a GPU-modified image is written back before its texture is dropped. When
    // the working set alone is over the budget the cache runs over rather than thrashing.
    private void CollectIfOverBudget(long now)
    {
        if (_totalUsedMemory <= _memoryBudgetBytes)
        {
            return;
        }

        var candidates = new List<(ulong Tick, ResourceSlotIdentifier Image)>();
        _slots.ForEach((imageIdentifier, image) =>
        {
            if (IsBudgetEvictableLocked(image, now))
            {
                candidates.Add((image.LastAccessTick, imageIdentifier));
            }
        });
        candidates.Sort(static (left, right) => left.Tick.CompareTo(right.Tick));
        var target = _memoryBudgetBytes - _memoryBudgetBytes / 8;
        foreach (var (_, imageIdentifier) in candidates)
        {
            if (_totalUsedMemory <= target)
            {
                break;
            }

            ReleaseImage(imageIdentifier, MetalImageReleaseCause.Budget);
            Interlocked.Increment(ref _evictions);
        }

        if (_totalUsedMemory > _memoryBudgetBytes)
        {
            MetalV2Diagnostics.InfoOnce(
                "image-budget",
                $"The image cache is over its budget with no evictable image: used={_totalUsedMemory / MiB}MB " +
                $"budget={_memoryBudgetBytes / MiB}MB in_use={_inUseBytes / MiB}MB images={_slots.Count}.");
        }
    }

    // Metadata slices whose clear has already been applied, by image, slice address and
    // clear code. This is the only record of it: guest memory is never marked, because
    // transient aliasing reuses a metadata range as live data (Poppy Playtime keeps
    // indirect-dispatch counters there) and a sentinel written into it is read back as
    // that data. Keyed on the code so a different clear colour at the same metadata is a
    // new clear, and on the image so a second image over the same bytes is its own.
    // A rewrite of the metadata by the guest, by the GPU or by a CPU store removes the
    // claims over it, so a guest that clears every frame still gets one clear per clear
    // while the same surface looked up 24 times between clears (The Oregon Trail) is
    // cleared once.
    private readonly HashSet<(ResourceSlotIdentifier Image, ulong Address, byte Code)> _materializedDccSlices = new();

    // The claims above, counted by the 4 KiB page their slice starts on. Every GPU store the
    // stream records (each command-processor write, each writable buffer binding) re-arms
    // the claims over its bytes, and walking every claim for each one cost the worker a
    // scan per store; a store whose pages hold no claim start skips the walk. Exact: a
    // claim is removed only when its slice start lies in the stored range, and that start
    // is then on one of the range's pages. SHARPEMU_METAL_V2_DCC_CLAIM_INDEX=0 walks every
    // claim on every store, as before.
    private readonly Dictionary<ulong, int> _dccClaimPages = new();
    private static readonly bool DccClaimIndexEnabled = MetalV2Diagnostics.DccClaimPageIndex;
    private const int DccClaimPageShift = 12;
    private const ulong DccClaimIndexMaxPages = 64;

    // Caller holds the lock.
    private void IndexDccClaim(ulong address)
    {
        System.Runtime.InteropServices.CollectionsMarshal.GetValueRefOrAddDefault(
            _dccClaimPages, address >> DccClaimPageShift, out _)++;
    }

    // Caller holds the lock.
    private void UnindexDccClaim(ulong address)
    {
        var page = address >> DccClaimPageShift;
        ref var count = ref System.Runtime.InteropServices.CollectionsMarshal.GetValueRefOrNullRef(_dccClaimPages, page);
        if (System.Runtime.CompilerServices.Unsafe.IsNullRef(ref count))
        {
            return;
        }

        if (--count <= 0)
        {
            _dccClaimPages.Remove(page);
        }
    }

    // Whether a claim may start in the range; false only when none does. Caller holds the lock.
    private bool MayHoldDccClaimStart(ulong address, ulong size)
    {
        if (!DccClaimIndexEnabled)
        {
            return true;
        }

        if (size == 0)
        {
            return false;
        }

        var first = address >> DccClaimPageShift;
        var last = (address + (size - 1)) >> DccClaimPageShift;
        if (address + (size - 1) < address || last - first >= DccClaimIndexMaxPages)
        {
            return true;
        }

        for (var page = first; page <= last; page++)
        {
            if (_dccClaimPages.ContainsKey(page))
            {
                return true;
            }
        }

        return false;
    }

    // The claims above that still stand without a look at their bytes, by image and slice
    // address, with the code each was claimed for. A slice found here is a repeat with no
    // pending-write query and no read of the metadata: the full scan would read the code
    // the claim was taken for and reach the same repeat.
    //
    // An entry is added when a claim is taken (after a settled scan, or from a recorded fill
    // whose code is what lands) and again when a settled scan finds its claim standing. It
    // goes with the claim, at every re-arm of the metadata (a CPU store into the watched
    // range, a recorded GPU store, a fill, ClearMetadata) and when the image is released.
    // It also goes, keeping the claim, wherever recorded work stores into guest memory
    // without re-arming claims: image writebacks and alias publications, staged-storage
    // writebacks, kernel fills and copies, end-of-pipe labels and unbounded stores. Those
    // slices take the full path again, which is what notices whether their bytes changed.
    // Writes through the raw backing (TryWriteBacking) reach no hook; the full path sees
    // them only by re-reading, which SHARPEMU_METAL_V2_DCC_CLAIM_FIRST_VERIFY reports.
    private readonly Dictionary<(ResourceSlotIdentifier Image, ulong Address), byte> _standingDccClaims = new();

    /// <summary>SHARPEMU_METAL_V2_DCC_CLAIM_FIRST: a standing claim settles its slice unread.</summary>
    internal bool DccClaimFirst { get; set; } = MetalV2Diagnostics.DccClaimFirst;

    /// <summary>The full path runs for every slice, and a standing claim it disagrees with is reported.</summary>
    internal bool DccClaimFirstVerify { get; set; } = MetalV2Diagnostics.DccClaimFirstVerify;

    private long _dccClaimFirstMismatches;
    private const int DccClaimFirstMismatchLines = 64;

    /// <summary>Standing claims the full path disagreed with, under the verify mode.</summary>
    internal long TakeDccClaimFirstMismatches() => Interlocked.Exchange(ref _dccClaimFirstMismatches, 0);

    /// <summary>
    /// Recorded work stores into guest memory without re-arming the claims there: the
    /// slices starting in the range take the full path until a settled scan finds them
    /// unchanged. Callable with or without the cache lock held: a writeback encoded under it
    /// can end the encoders and record the staged stores, which land here.
    /// </summary>
    public void ForgetStandingDccClaims(ulong address, ulong size)
    {
        if (Volatile.Read(ref _standingDccClaimCount) == 0)
        {
            return;
        }

        if (_lock.IsHeldByCurrentThread)
        {
            ForgetStandingDccClaimsLocked(address, size);
            return;
        }

        using var held = _lock.Hold();
        ForgetStandingDccClaimsLocked(address, size);
    }

    // A writeback or alias publication this cache records into guest memory: the store
    // does not re-arm claims, so the slices it reaches stop standing.
    private void NoteOwnGuestWrite(ulong address, ulong size)
    {
        ForgetStandingDccClaims(address, size);
        _owner.NoteGuestWrite(address, size);
    }

    /// <summary>As above, for a store that names no range: every slice takes the full path.</summary>
    public void ForgetAllStandingDccClaims()
    {
        if (Volatile.Read(ref _standingDccClaimCount) == 0)
        {
            return;
        }

        if (_lock.IsHeldByCurrentThread)
        {
            _standingDccClaims.Clear();
            Volatile.Write(ref _standingDccClaimCount, 0);
            return;
        }

        using var held = _lock.Hold();
        _standingDccClaims.Clear();
        Volatile.Write(ref _standingDccClaimCount, 0);
    }

    /// <summary>Runs <paramref name="body"/> under the cache lock, as a writeback encoded inside a lookup does.</summary>
    internal void RunLockedForTest(Action body)
    {
        using var held = _lock.Hold();
        body();
    }

    // _standingDccClaims.Count, for the callers above to skip the lock when there is none.
    // Entries are added by lookups on the GPU worker, which is also what records stores.
    private int _standingDccClaimCount;

    /// <summary>Caller holds the lock.</summary>
    private void StandDccClaimLocked(ResourceSlotIdentifier image, ulong address, byte code)
    {
        _standingDccClaims[(image, address)] = code;
        Volatile.Write(ref _standingDccClaimCount, _standingDccClaims.Count);
    }

    /// <summary>Caller holds the lock.</summary>
    private void UnstandDccClaimLocked(ResourceSlotIdentifier image, ulong address)
    {
        if (_standingDccClaims.Remove((image, address)))
        {
            Volatile.Write(ref _standingDccClaimCount, _standingDccClaims.Count);
        }
    }

    /// <summary>Caller holds the lock.</summary>
    private void ForgetStandingDccClaimsLocked(ulong address, ulong size)
    {
        if (_standingDccClaims.Count == 0 || !MayHoldDccClaimStart(address, size))
        {
            return;
        }

        // Removing the current entry leaves the enumeration valid.
        foreach (var key in _standingDccClaims.Keys)
        {
            if (key.Address >= address && key.Address - address < size)
            {
                _standingDccClaims.Remove(key);
            }
        }

        Volatile.Write(ref _standingDccClaimCount, _standingDccClaims.Count);
    }

    // Whether an image the GPU has modified holds bytes of the slice. Caller holds the lock.
    private bool GpuModifiedImageOverlapsLocked(ulong address, ulong size)
    {
        if (!_pageOwners.MayHaveOwners(address, size))
        {
            return false;
        }

        foreach (var imageIdentifier in FindImagesInRange(address, size, pageOverlap: false))
        {
            if (_slots.TryGet(imageIdentifier) is { } image && image.GpuOverlaps(address, size))
            {
                return true;
            }
        }

        return false;
    }

    // Whether any recorded fill reaches the slice. Caller holds the lock.
    private bool RecordedDccFillOverlapsLocked(ulong address, ulong size)
    {
        foreach (var (start, fill) in _recordedDccFills)
        {
            if (address < start + fill.Size && start < address + size)
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Marks in <paramref name="standing"/> the slices of one lookup whose claim stands, with
    /// the code it stands for in <paramref name="codes"/>, all under one hold of the lock. A
    /// slice a recorded fill still reaches is left to the full path, which settles the fill,
    /// and so is one under a GPU-modified image: rendering into it records no guest write, so
    /// the claim outlives it, and the full path's pending query defers the slice until the
    /// image's bytes are written back. Returns how many were marked.
    /// </summary>
    private int ProbeStandingDccClaims(
        ResourceSlotIdentifier imageIdentifier, ulong firstAddress, ulong sliceSize, uint count,
        Span<bool> standing, Span<byte> codes)
    {
        if (Volatile.Read(ref _standingDccClaimCount) == 0)
        {
            return 0;
        }

        var found = 0;
        using var held = _lock.Hold();
        for (var slice = 0u; slice < count; slice++)
        {
            var address = firstAddress + sliceSize * slice;
            if (!_standingDccClaims.TryGetValue((imageIdentifier, address), out var code) ||
                (_recordedDccFills.Count != 0 && RecordedDccFillOverlapsLocked(address, sliceSize)) ||
                GpuModifiedImageOverlapsLocked(address, sliceSize))
            {
                continue;
            }

            standing[(int)slice] = true;
            codes[(int)slice] = code;
            found++;
            if (!DccClaimFirstVerify && _deferredDccSlices.Count != 0)
            {
                // What the claimed repeat of the full path does with a deferral.
                RemoveDccDeferralLocked(imageIdentifier, address);
            }
        }

        return found;
    }

    private void ReportDccClaimFirstMismatch(
        ResourceSlotIdentifier imageIdentifier, ulong address, byte claimed, string full)
    {
        if (Interlocked.Increment(ref _dccClaimFirstMismatches) <= DccClaimFirstMismatchLines)
        {
            Console.Error.WriteLine(
                $"[METAL-V2][DCC][VERIFY] a standing claim disagrees with the full path: image={imageIdentifier} " +
                $"meta=0x{address:X} claimed=0x{claimed:X2} full={full}");
        }
    }

    private long _dccRepeatsSkipped;
    private long _dccSlicesCoalesced;

    // Page-watched metadata ranges, 4 KiB aligned, address to size. A CPU store into
    // one re-arms the claims over it and releases the watch; the next claim watches it
    // again.
    private readonly Dictionary<ulong, ulong> _metadataWatches = new();

    // The watched ranges as the CPU-write fast path reads them, without the lock: sorted
    // by Begin and disjoint, because a watch that would overlap another is merged into
    // it. Replaced whole, never edited, so a reader sees one consistent set. Republished
    // under the lock whenever _metadataWatches changes, before a new watch reaches the
    // page guard, so a store that faults on a metadata page always finds it here.
    private MetadataWatchSpan[] _metadataWatchSnapshot = [];

    // Reused under the lock by the watch scans and the snapshot build.
    private readonly List<(ulong Begin, ulong Size)> _metadataWatchScratch = [];
    private readonly List<MetadataWatchSpan> _metadataWatchSorted = [];

    internal readonly record struct MetadataWatchSpan(ulong Begin, ulong End);

    // CPU writes that took the cache lock in this window: because an image may own the
    // page, or because the write reached a watched metadata range.
    private long _cpuWritesLockedForOwner;
    private long _cpuWritesLockedForMetadata;

    // Of the writes locked because an image may own the page, those that found an image
    // over their bytes and those that found none: the second is lock traffic the owner
    // index could have answered without the lock.
    private long _cpuWritesOwnerFound;
    private long _cpuWritesOwnerNone;

    public (long Owner, long Metadata, long OwnerFound, long OwnerNone) TakeLockedCpuWrites() =>
        (Interlocked.Exchange(ref _cpuWritesLockedForOwner, 0),
         Interlocked.Exchange(ref _cpuWritesLockedForMetadata, 0),
         Interlocked.Exchange(ref _cpuWritesOwnerFound, 0),
         Interlocked.Exchange(ref _cpuWritesOwnerNone, 0));

    private void CountOwnerCpuWriteOutcome(bool mayOwn, bool imageHit)
    {
        if (!mayOwn)
        {
            return;
        }

        if (imageHit)
        {
            Interlocked.Increment(ref _cpuWritesOwnerFound);
        }
        else
        {
            Interlocked.Increment(ref _cpuWritesOwnerNone);
        }
    }

    private void CountLockedCpuWrite(bool mayOwn)
    {
        if (mayOwn)
        {
            Interlocked.Increment(ref _cpuWritesLockedForOwner);
        }
        else
        {
            Interlocked.Increment(ref _cpuWritesLockedForMetadata);
        }
    }

    /// <summary>
    /// Whether <c>[address, address + size)</c> meets a watched metadata range. Lock free:
    /// it reads the published snapshot, so a CPU write that reaches no metadata need not
    /// wait for the cache lock however many other ranges are watched.
    /// </summary>
    internal bool MetadataWatchOverlaps(ulong address, ulong size) =>
        AnySpanOverlaps(Volatile.Read(ref _metadataWatchSnapshot), address, size);

    /// <summary>Whether a range meets any of <paramref name="spans"/>, sorted by Begin and disjoint.</summary>
    internal static bool AnySpanOverlaps(ReadOnlySpan<MetadataWatchSpan> spans, ulong address, ulong size)
    {
        if (spans.Length == 0 || size == 0)
        {
            return false;
        }

        // The last span that begins before the range ends. Spans are disjoint and sorted,
        // so their ends ascend too: if that one ends at or before the range, all do.
        var end = address + size;
        int low = 0, high = spans.Length - 1, found = -1;
        while (low <= high)
        {
            var middle = low + ((high - low) >> 1);
            if (spans[middle].Begin < end)
            {
                found = middle;
                low = middle + 1;
            }
            else
            {
                high = middle - 1;
            }
        }

        return found >= 0 && spans[found].End > address;
    }

    /// <summary>Caller holds the lock.</summary>
    private void PublishMetadataWatchesLocked()
    {
        if (_metadataWatches.Count == 0)
        {
            Volatile.Write(ref _metadataWatchSnapshot, []);
            return;
        }

        // Adjacent watches are published as one span: the snapshot answers whether a range meets
        // any watch, which their union answers alike, and an image's consecutive DCC slices are
        // separate watches. Published whole per change, Black Myth: Wukong's thousands of
        // watches made a large object each time, and those were its full collections.
        var sorted = _metadataWatchSorted;
        sorted.Clear();
        foreach (var (watchBegin, watchSize) in _metadataWatches)
        {
            sorted.Add(new MetadataWatchSpan(watchBegin, watchBegin + watchSize));
        }

        sorted.Sort(static (left, right) => left.Begin.CompareTo(right.Begin));
        var merged = 0;
        for (var index = 0; index < sorted.Count; index++)
        {
            if (merged != 0 && sorted[merged - 1].End >= sorted[index].Begin)
            {
                sorted[merged - 1] = sorted[merged - 1] with { End = Math.Max(sorted[merged - 1].End, sorted[index].End) };
            }
            else
            {
                sorted[merged++] = sorted[index];
            }
        }

        var spans = new MetadataWatchSpan[merged];
        sorted.CopyTo(0, spans, 0, merged);
        Volatile.Write(ref _metadataWatchSnapshot, spans);
    }

    /// <summary>Materializations skipped because the clear was already applied.</summary>
    public long TakeDccRepeatsSkipped() => Interlocked.Exchange(ref _dccRepeatsSkipped, 0);

    /// <summary>DCC slice clears that rode along in another slice's pass since the last call.</summary>
    public long TakeDccSlicesCoalesced() => Interlocked.Exchange(ref _dccSlicesCoalesced, 0);

    private bool TryClaimMaterializedDccLocked(ResourceSlotIdentifier image, ulong address, ulong size, byte code)
    {
        if (!_materializedDccSlices.Add((image, address, code)))
        {
            return false;
        }

        StandDccClaimLocked(image, address, code);
        IndexDccClaim(address);
        WatchMetadataLocked(address, size);
        Epochs.NoteDccChanged();
        return true;
    }

    private void ReArmMaterializedDccLocked(ulong address, ulong size)
    {
        if (_materializedDccSlices.Count != 0 && MayHoldDccClaimStart(address, size))
        {
            var removed = _materializedDccSlices.RemoveWhere(entry =>
            {
                if (entry.Address < address || entry.Address - address >= size)
                {
                    return false;
                }

                UnindexDccClaim(entry.Address);
                UnstandDccClaimLocked(entry.Image, entry.Address);
                return true;
            });
            if (removed != 0)
            {
                Epochs.NoteDccChanged();
            }

            if (DccTraceActive && removed != 0 && DccTraceCovers(address, size) && Interlocked.Increment(ref _dccTraceLines) <= DccTraceMax)
            {
                Console.Error.WriteLine($"[METAL-V2][DCC] re-arm range=0x{address:X}+0x{size:X} claims={removed}");
            }
        }

        // A new clear of the metadata starts a new deferral: what was drawn before it is
        // what the new clear is meant to wipe.
        if (_deferredDccSlices.Count != 0)
        {
            foreach (var key in _deferredDccSlices.Keys)
            {
                if (key.Address >= address && key.Address - address < size)
                {
                    RemoveDccDeferralLocked(key.Image, key.Address);
                }
            }
        }
    }

    // Slices whose clear a lookup met while recorded work was still writing the metadata,
    // by image and slice address, with the image's GPU-write sequence at that lookup. The
    // clear could not be applied then, and the target was then drawn: per draw the colour
    // target lookup skips the slice and the draw lands on the uncleared image. Applying the
    // clear once the metadata settles — at the next draw's lookup or at the texture lookup
    // that samples the result — would wipe those draws, which is what made The Oregon
    // Trail's menu art flicker black. A clear the image has been written past since it was
    // deferred is dropped instead. Keyed per slice, not per image, so drawing one layer of
    // an array does not cost the next layer its clear.
    private readonly Dictionary<(ResourceSlotIdentifier Image, ulong Address), long> _deferredDccSlices = new();
    private long _dccClearsDropped;

    /// <summary>Deferred DCC clears dropped because the image was drawn to since.</summary>
    public long TakeDccClearsDropped() => Interlocked.Exchange(ref _dccClearsDropped, 0);

    // The metadata settled to something that is not a clear: there is no deferred clear left.
    private void ForgetDeferredDccSlice(ResourceSlotIdentifier image, ulong address)
    {
        if (_deferredDccSlices.Count == 0)
        {
            return;
        }

        using var held = _lock.Hold();
        RemoveDccDeferralLocked(image, address);
    }

    /// <summary>Caller holds the lock.</summary>
    private void RemoveDccDeferralLocked(ResourceSlotIdentifier image, ulong address)
    {
        if (_deferredDccSlices.Remove((image, address)))
        {
            Epochs.NoteDccChanged();
        }
    }

    // Clear-code fills of DCC metadata that run as recorded GPU work, by metadata address.
    // While such a fill is still in flight its bytes cannot be read back, so a lookup
    // would defer the clear and the draws that follow land on the uncleared surface —
    // which is how OCTOPATH TRAVELER II's separate translucency kept every frame's
    // particles and drew them as trails across the traveler map. The fill is recorded
    // in submission order before the lookup, so the code it stores is the metadata's
    // content by the time the clear executes. A record lasts until any other store
    // reaches its range; AwaitingOwnWrite lets the fill's own dispatch, bound right
    // after it was absorbed, through.
    private readonly Dictionary<ulong, RecordedDccFill> _recordedDccFills = new();

    private struct RecordedDccFill
    {
        public ulong Size;
        public byte Code;
        public bool AwaitingOwnWrite;
    }

    /// <summary>Caller holds the lock.</summary>
    private void RecordDccFillLocked(ulong address, ulong size, byte code, bool record)
    {
        ForgetRecordedDccFillsLocked(address, size);
        if (record)
        {
            _recordedDccFills[address] = new RecordedDccFill { Size = size, Code = code, AwaitingOwnWrite = true };
            Epochs.NoteDccChanged();
        }
    }

    /// <summary>
    /// A recorded GPU store reached this range. The store a fill's own dispatch makes
    /// keeps that fill; every other one replaces whatever the fill stored.
    ///
    /// Caller holds the lock.
    /// </summary>
    private void NoteRecordedDccStoreLocked(ulong address, ulong size)
    {
        if (_recordedDccFills.Count == 0)
        {
            return;
        }

        // Enumerated in place, not over a copy: this runs for every GPU store. Removing the
        // current entry and writing a value through its reference leave the enumeration valid.
        foreach (var (start, fill) in _recordedDccFills)
        {
            if (address >= start + fill.Size || start >= address + size)
            {
                continue;
            }

            if (fill.AwaitingOwnWrite && address >= start && address + size <= start + fill.Size)
            {
                System.Runtime.InteropServices.CollectionsMarshal.GetValueRefOrNullRef(_recordedDccFills, start).AwaitingOwnWrite = false;
                Epochs.NoteDccChanged();
                continue;
            }

            _recordedDccFills.Remove(start);
            Epochs.NoteDccChanged();
        }
    }

    /// <summary>Caller holds the lock.</summary>
    private void ForgetRecordedDccFillsLocked(ulong address, ulong size)
    {
        if (_recordedDccFills.Count == 0)
        {
            return;
        }

        // In place, not over a copy: removing the current entry leaves the enumeration valid.
        foreach (var (start, fill) in _recordedDccFills)
        {
            if (address < start + fill.Size && start < address + size)
            {
                _recordedDccFills.Remove(start);
                Epochs.NoteDccChanged();
            }
        }
    }

    /// <summary>
    /// The clear code a recorded fill leaves in one metadata slice whose bytes are still
    /// being written.
    ///
    /// Caller holds the lock.
    /// </summary>
    private bool TryFindRecordedDccFillLocked(ulong address, ulong size, out byte code)
    {
        code = 0;
        foreach (var (start, fill) in _recordedDccFills)
        {
            if (address >= start && address + size <= start + fill.Size)
            {
                code = fill.Code;
                return true;
            }
        }

        return false;
    }

    private enum DccClearDecision
    {
        Apply,
        Repeat,
        Drop,
    }

    /// <summary>
    /// Claims a uniform clear for this slice and decides what becomes of it: applied,
    /// already applied, or dropped because the image was written after the clear was
    /// deferred. A dropped clear stays claimed, so later lookups do not apply it either.
    /// The scan around it runs outside the cache lock — it reads guest backing — so the
    /// claim takes the lock itself.
    /// </summary>
    private DccClearDecision DecideDccClear(
        ResourceSlotIdentifier image, ulong address, ulong size, byte code, long imageWriteSequence)
    {
        using var held = _lock.Hold();
        var deferred = _deferredDccSlices.Remove((image, address), out var deferredAt);
        if (deferred)
        {
            Epochs.NoteDccChanged();
        }

        if (!TryClaimMaterializedDccLocked(image, address, size, code))
        {
            return DccClearDecision.Repeat;
        }

        return deferred && imageWriteSequence > deferredAt ? DccClearDecision.Drop : DccClearDecision.Apply;
    }

    private void WatchMetadataLocked(ulong address, ulong size)
    {
        const ulong Page = TrackerLayout.PageBytes;
        if (_watchingStopped || size == 0 || address > ulong.MaxValue - size - Page)
        {
            return;
        }

        var begin = address & ~(Page - 1);
        var end = (address + size + Page - 1) & ~(Page - 1);
        // The snapshot is the union of the watches, republished under the lock at every change:
        // a range it misses meets no watch, and the dictionary is scanned only when it does.
        // The scans collect into a reused list instead of copying the dictionary: a copy per
        // call was a large object, and with OnMetadataCpuWriteLocked's Black Myth: Wukong
        // collected garbage twice as often.
        if (AnySpanOverlaps(_metadataWatchSnapshot, begin, end - begin))
        {
            var scratch = _metadataWatchScratch;
            scratch.Clear();
            foreach (var (watchBegin, watchSize) in _metadataWatches)
            {
                if (watchBegin <= begin && end <= watchBegin + watchSize)
                {
                    return;
                }

                if (watchBegin < end && begin < watchBegin + watchSize)
                {
                    scratch.Add((watchBegin, watchSize));
                }
            }

            // A watch the new range partly overlaps is replaced by the union, so each page
            // carries at most one count from here. Watches are disjoint, so only those that
            // meet the new range can meet the union.
            foreach (var (watchBegin, watchSize) in scratch)
            {
                _pages.RemoveWatch(watchBegin, watchSize, blockReads: false);
                _metadataWatches.Remove(watchBegin);
                begin = Math.Min(begin, watchBegin);
                end = Math.Max(end, watchBegin + watchSize);
            }
        }

        _metadataWatches[begin] = end - begin;
        // Published before the page guard takes the watch: a store that faults on it
        // must see the range, or the fast path would turn the fault away.
        PublishMetadataWatchesLocked();
        _pages.AddWatch(begin, end - begin, blockReads: false);
    }

    // A CPU store reached a watched metadata range: the claims over it are forgotten
    // and the watch is released so the store can land.
    private bool OnMetadataCpuWriteLocked(ulong address, ulong size)
    {
        const ulong Page = TrackerLayout.PageBytes;
        if (_metadataWatches.Count == 0 || size == 0)
        {
            return false;
        }

        var begin = address & ~(Page - 1);
        var end = (address + size + Page - 1) & ~(Page - 1);
        var hit = false;
        // The union snapshot answers a store that meets no watch; the rest scan the dictionary
        // into the reused list (see WatchMetadataLocked).
        if (!AnySpanOverlaps(_metadataWatchSnapshot, begin, end - begin))
        {
            return false;
        }

        var scratch = _metadataWatchScratch;
        scratch.Clear();
        foreach (var (watchBegin, watchSize) in _metadataWatches)
        {
            if (watchBegin < end && begin < watchBegin + watchSize)
            {
                scratch.Add((watchBegin, watchSize));
            }
        }

        foreach (var (watchBegin, watchSize) in scratch)
        {
            ReArmMaterializedDccLocked(watchBegin, watchSize);
            ForgetRecordedDccFillsLocked(watchBegin, watchSize);
            _pages.RemoveWatch(watchBegin, watchSize, blockReads: false);
            _metadataWatches.Remove(watchBegin);
            hit = true;
        }

        if (hit)
        {
            PublishMetadataWatchesLocked();
        }

        return hit;
    }

    private void RegisterDccMetadata(in ImageDescription description)
    {
        var address = description.Metadata.Range.Address;
        if (!_surfaceMetadata.TryGetValue(address, out var metadata))
        {
            metadata = new SurfaceMetadata { Kind = SurfaceMetadataKind.Dcc };
            _surfaceMetadata.Add(address, metadata);
        }
        else if (metadata.Kind == SurfaceMetadataKind.PendingDcc)
        {
            metadata.Kind = SurfaceMetadataKind.Dcc;
        }
        else if (metadata.Kind != SurfaceMetadataKind.Dcc)
        {
            return;
        }

        metadata.Size = Math.Max(metadata.Size, description.DccSliceSize * description.TransferLayers);
        WatchMetadataLocked(address, metadata.Size);
    }

    private void RegisterHtileMetadata(in ImageDescription description)
    {
        var address = description.Metadata.Range.Address;
        if (!_surfaceMetadata.TryGetValue(address, out var metadata))
        {
            _surfaceMetadata.Add(address, new SurfaceMetadata
            {
                Kind = SurfaceMetadataKind.HTile,
                ClearMask = description.HtileClearMask,
                Size = description.Metadata.Range.Size,
            });
            return;
        }

        if (metadata.Kind == SurfaceMetadataKind.PendingDcc)
        {
            metadata.Kind = SurfaceMetadataKind.HTile;
            metadata.ClearMask = description.HtileClearMask;
            metadata.FillValue = 0xffffffff;
            metadata.FillSize = 0;
        }

        if (metadata.Kind == SurfaceMetadataKind.HTile)
        {
            metadata.Size = description.Metadata.Range.Size;
        }
    }

    public bool IsMetadata(ulong address)
    {
        using var held = _lock.Hold();
        return _surfaceMetadata.TryGetValue(address, out var found) && found.Kind != SurfaceMetadataKind.PendingDcc;
    }

    /// <summary>
    /// Whether a binding of <paramref name="size"/> bytes at the address spans the metadata
    /// surface registered there. A surface whose size is not known is answered by its
    /// address alone, as <see cref="IsMetadata"/> does.
    /// </summary>
    public bool CoversMetadata(ulong address, ulong size)
    {
        using var held = _lock.Hold();
        return _surfaceMetadata.TryGetValue(address, out var found) && found.Kind != SurfaceMetadataKind.PendingDcc &&
            (found.Size == 0 || size >= found.Size);
    }

    public bool IsMetadataCleared(ulong address, uint slice, out uint fillValue)
    {
        using var held = _lock.Hold();
        return IsMetadataClearedLocked(address, slice, out fillValue);
    }

    /// <summary>As above, for a caller that already holds the lock.</summary>
    private bool IsMetadataClearedLocked(ulong address, uint slice, out uint fillValue)
    {
        fillValue = 0;
        if (!_surfaceMetadata.TryGetValue(address, out var found) ||
            found.Kind == SurfaceMetadataKind.PendingDcc || slice >= 32)
        {
            return false;
        }

        fillValue = found.FillValue;
        return (found.ClearMask & (1u << (int)slice)) != 0;
    }

    public bool IsMetadataCleared(ulong address, uint slice) => IsMetadataCleared(address, slice, out _);

    public bool SetMetadataSlice(ulong address, uint slice, bool isClear)
    {
        using var held = _lock.Hold();
        return SetMetadataSliceLocked(address, slice, isClear);
    }

    /// <summary>As above, for a caller that already holds the lock.</summary>
    private bool SetMetadataSliceLocked(ulong address, uint slice, bool isClear)
    {
        if (!_surfaceMetadata.TryGetValue(address, out var found) ||
            found.Kind == SurfaceMetadataKind.PendingDcc || slice >= 32)
        {
            return false;
        }

        if (isClear)
        {
            found.ClearMask |= 1u << (int)slice;
        }
        else
        {
            found.ClearMask &= ~(1u << (int)slice);
        }

        return true;
    }

    // A broad clear applies to CMask, FMask and HTile; DCC needs a validated fill value.
    public bool ClearMetadata(ulong address)
    {
        using var held = _lock.Hold();
        // A broad clear of DCC metadata is the guest clearing it again, whether or not
        // the clear can be applied here.
        ReArmMaterializedDccLocked(
            address, _surfaceMetadata.TryGetValue(address, out var existing) ? Math.Max(existing.Size, 1) : 1);
        if (!_surfaceMetadata.TryGetValue(address, out var found) ||
            found.Kind is SurfaceMetadataKind.PendingDcc or SurfaceMetadataKind.Dcc)
        {
            return false;
        }

        found.ClearMask = uint.MaxValue;
        return true;
    }

    public bool OverlapsDccMetadata(ulong address, ulong size)
    {
        using var held = _lock.Hold();
        foreach (var (start, metadata) in _surfaceMetadata)
        {
            if (metadata.Kind == SurfaceMetadataKind.Dcc && metadata.Size != 0 &&
                address < start + metadata.Size && start < address + size)
            {
                return true;
            }
        }

        foreach (var (start, metadataSize) in _adoptedDccMetadata)
        {
            if (address < start + metadataSize && start < address + size)
            {
                return true;
            }
        }

        return false;
    }

    // The DCC metadata ranges images took with their size, by address. Adoption takes such
    // metadata out of _surfaceMetadata — its slices are read for a clear instead — so this
    // is what still knows the range is metadata when the guest fills it again. Entries
    // outlive their images: a stale one only lets a later fill of those bytes run on the
    // host with its code recorded, which writes the same bytes a dispatch would.
    private readonly Dictionary<ulong, ulong> _adoptedDccMetadata = new();

    /// <summary>
    /// A fill of known DCC metadata that runs on the host, by the CPU or a blit. Its code
    /// is what the metadata holds once the fill lands, so a lookup that meets the bytes
    /// still in flight takes the clear from here instead of deferring it — a deferred
    /// clear is dropped once the target is drawn, which left Demon's Souls' UI layer
    /// holding whatever the overlapping HDR target had put in its memory.
    /// </summary>
    public void RecordDccFill(ulong address, ulong size, uint fillValue)
    {
        if (!IsValidRange(address, size))
        {
            return;
        }

        var code = (byte)fillValue;
        var clearCode = fillValue == code * 0x01010101u && code is 0x00 or 0x20 or 0x40 or 0x80 or 0xc0;
        using var held = _lock.Hold();
        ReArmMaterializedDccLocked(address, size);
        RecordDccFillLocked(address, size, code, clearCode);
    }

    public bool TryAbsorbDccFill(ulong address, ulong size, uint fillValue)
    {
        if (!IsValidRange(address, size))
        {
            return false;
        }

        // A DCC fill repeats one byte code; only the known deferred-clear codes count as clear.
        var code = (byte)fillValue;
        var dccClearMask = fillValue != code * 0x01010101u ? 0u : code switch
        {
            0x00 or 0x20 or 0x40 or 0x80 or 0xc0 => uint.MaxValue,
            _ => 0u,
        };
        using var held = _lock.Hold();
        // A guest fill of this metadata is the guest asking for the clear again.
        ReArmMaterializedDccLocked(address, size);
        _surfaceMetadata.TryGetValue(address, out var found);
        // Only a fill this returns false for runs as written; one tracked here never does.
        RecordDccFillLocked(
            address, size, code, dccClearMask != 0 && (found is null || found.Kind == SurfaceMetadataKind.PendingDcc));
        if (found is null)
        {
            // The fill may precede color-target discovery; a pending entry stays
            // invisible to the metadata queries until the target classifies it.
            _surfaceMetadata.Add(address, new SurfaceMetadata
            {
                Kind = SurfaceMetadataKind.PendingDcc,
                ClearMask = dccClearMask,
                FillValue = fillValue,
                FillSize = size,
            });
            return false;
        }

        if (found.Kind == SurfaceMetadataKind.PendingDcc)
        {
            found.ClearMask = dccClearMask;
            found.FillValue = fillValue;
            found.FillSize = size;
            return false;
        }

        if (found.Kind == SurfaceMetadataKind.Dcc)
        {
            found.ClearMask = dccClearMask;
            found.FillValue = fillValue;
            found.FillSize = size;
            return true;
        }

        return false;
    }

    /// <summary>
    /// The image takes the DCC metadata a request names, in the lookup's own hold. True
    /// when its slices are then to be scanned for a clear by MaterializeDccClear, with
    /// the image's GPU write sequence as it stood before that scan. Caller holds the lock.
    /// </summary>
    private bool AdoptDccMetadataLocked(
        ResourceSlotIdentifier imageIdentifier, in ImageRequest request, out long imageWriteSequence)
    {
        imageWriteSequence = 0;
        if (request.Description.Metadata.Kind != MetadataKind.Dcc ||
            _slots.TryGet(imageIdentifier) is not { } found)
        {
            return false;
        }

        ref readonly var description = ref request.Description;
        var range = description.Metadata.Range;
        // Taken before this lookup clears anything: its own clears are not draws.
        imageWriteSequence = found.GpuWriteSequence;
        if (!MetalImageEpochs.SameMetadata(in found.Description.Metadata, in description.Metadata))
        {
            found.Description.Metadata = description.Metadata;
            Epochs.NoteShapeChanged();
        }
        if (range.Size == 0 || description.Resources.Levels != 1 || found.Description.Resources.Levels != 1)
        {
            return false;
        }

        _surfaceMetadata.Remove(range.Address);
        _adoptedDccMetadata[range.Address] = Math.Max(
            range.Size, _adoptedDccMetadata.TryGetValue(range.Address, out var adopted) ? adopted : 0);
        return true;
    }

    /// <summary>
    /// A uniform DCC key over a whole slice means the surface is logically cleared to
    /// one color. The texture takes that color and the metadata becomes uncompressed,
    /// which is what makes the following draws and texture reads see the clear.
    /// </summary>
    private void MaterializeDccClear(
        ResourceSlotIdentifier imageIdentifier, in ImageRequest request, uint metadataBaseLayer, long imageWriteSequence)
    {
        using var phaseScope = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.DccClearScan);
        ref readonly var description = ref request.Description;
        var range = description.Metadata.Range;
        var layers = description.TransferLayers;
        const ulong MetadataBlockSize = 0x1000;
        if (!ImageDescription.IsValidRange(range) || range.Address % MetadataBlockSize != 0 || layers == 0 ||
            range.Size % layers != 0 || (range.Size / layers) % MetadataBlockSize != 0)
        {
            MetalV2Diagnostics.WarnOnce(
                "dcc-slices",
                $"DCC metadata is not aligned 4 KiB slices; the deferred clear is ignored: address=0x{range.Address:X16} " +
                $"size=0x{range.Size:X} layers={layers}.");
            return;
        }

        var view = request.View;
        var volume = description.IsVolume && view.Type == GpuImageViewType.Type3D;
        var first = volume ? 0u : metadataBaseLayer;
        var imageFirst = volume ? 0u : view.BaseLayer;
        var count = volume ? description.Extent.Depth : view.LayerCount;
        if (first >= layers || count == 0 || count > layers - first)
        {
            return;
        }

        var sliceSize = range.Size / layers;
        // Slices whose claim stands are repeats before anything is read: no pending-write
        // query and no read of their bytes (see _standingDccClaims). Under the verify mode
        // they take the full path too, and its answer is compared with the claim's.
        const int StackSlices = 256;
        Span<bool> standing = count <= StackSlices ? stackalloc bool[StackSlices] : new bool[count];
        Span<byte> standingCodes = count <= StackSlices ? stackalloc byte[StackSlices] : new byte[count];
        standing = standing[..(int)count];
        standing.Clear();
        var standingCount = DccClaimFirst
            ? ProbeStandingDccClaims(imageIdentifier, range.Address + sliceSize * first, sliceSize, count, standing, standingCodes)
            : 0;
        var verify = standingCount != 0 && DccClaimFirstVerify;
        if (standingCount != 0 && !verify)
        {
            Interlocked.Add(ref _dccRepeatsSkipped, standingCount);
            if (DccTraceActive)
            {
                for (var slice = 0u; slice < count; slice++)
                {
                    if (standing[(int)slice])
                    {
                        TraceDcc(imageIdentifier, in request, range.Address + sliceSize * (first + slice), sliceSize,
                            $"claim-standing code=0x{standingCodes[(int)slice]:X2}");
                    }
                }
            }

            if (standingCount == count)
            {
                return;
            }
        }

        // Rented on the first slice that is scanned whole.
        byte[]? buffer = null;
        // Adjacent slices that take the same clear are cleared together, as one pass over
        // all of their layers or depth planes. A volume the game clears whole through its
        // metadata is otherwise one pass per plane: UE4's 64-plane translucency volumes are
        // hundreds of passes a frame, each costing far more than its pixels.
        var run = default(DccClearRun);
        try
        {
            Span<byte> firstByte = stackalloc byte[1];
            for (var slice = 0u; slice < count; slice++)
            {
                var claimStands = standing[(int)slice];
                if (claimStands && !verify)
                {
                    continue;
                }

                var address = range.Address + sliceSize * (first + slice);
                // The metadata may still be being written by recorded work; scanning it
                // now reads half-written data, and claiming it would hide the real clear.
                // The deferral is remembered, so a draw that lands before it settles is not
                // wiped by applying it later.
                var pending = _owner.HasPendingGpuWrite(address, sliceSize);
                // Settled metadata says what it holds through its guest bytes; the first one
                // is read before the lock, which the backing does not need.
                var readable = !pending && _backing.TryReadBacking(address, firstByte);
                byte code;
                var recorded = false;
                GpuClearColor clear;
                using (var held = _lock.Hold())
                {
                    if (!pending)
                    {
                        // A slice whose bytes have settled has no further use for a recorded
                        // fill: from then on the guest bytes are what count.
                        ForgetRecordedDccFillsLocked(address, sliceSize);
                        if (!readable)
                        {
                            TraceDcc(imageIdentifier, in request, address, sliceSize, "unreadable");
                            continue;
                        }

                        code = firstByte[0];
                    }
                    else if (TryFindRecordedDccFillLocked(address, sliceSize, out code))
                    {
                        recorded = true;
                    }
                    else
                    {
                        if (_deferredDccSlices.TryAdd((imageIdentifier, address), imageWriteSequence))
                        {
                            Epochs.NoteDccChanged();
                        }

                        TraceDcc(imageIdentifier, in request, address, sliceSize, "pending-deferred");
                        continue;
                    }

                    if (!PackedClearValue.TryDecodeDccColor(
                            view.Format,
                            code * 0x01010101u,
                            description.Metadata,
                            allowClearToRegister: request.Role == ImageRole.ColorTarget,
                            out clear))
                    {
                        RemoveDccDeferralLocked(imageIdentifier, address);
                        if (claimStands)
                        {
                            ReportDccClaimFirstMismatch(
                                imageIdentifier, address, standingCodes[(int)slice], $"not-a-clear code=0x{code:X2}");
                        }

                        if (DccTraceActive)
                        {
                            TraceDcc(imageIdentifier, in request, address, sliceSize, $"not-a-clear code=0x{code:X2} recorded={recorded}");
                        }

                        continue;
                    }

                    // A slice already cleared with this code is settled without scanning it
                    // again: every rewrite of the metadata removes the claim, so a claim that
                    // stands means the bytes still hold the clear it was taken for. This is
                    // the repeat DecideDccClear would reach after the scan, deferral and all.
                    if (_materializedDccSlices.Contains((imageIdentifier, address, code)))
                    {
                        RemoveDccDeferralLocked(imageIdentifier, address);
                        Interlocked.Increment(ref _dccRepeatsSkipped);
                        if (claimStands && code != standingCodes[(int)slice])
                        {
                            ReportDccClaimFirstMismatch(
                                imageIdentifier, address, standingCodes[(int)slice], $"claimed-repeat code=0x{code:X2}");
                        }

                        if (!recorded)
                        {
                            // Read settled and found claimed: from here the claim settles
                            // the slice alone again.
                            StandDccClaimLocked(imageIdentifier, address, code);
                        }

                        if (DccTraceActive)
                        {
                            TraceDcc(imageIdentifier, in request, address, sliceSize, $"claimed-repeat code=0x{code:X2} recorded={recorded}");
                        }

                        continue;
                    }
                }

                if (claimStands)
                {
                    // The full path is about to scan a slice the claim would have settled.
                    ReportDccClaimFirstMismatch(
                        imageIdentifier, address, standingCodes[(int)slice], $"unclaimed code=0x{code:X2} recorded={recorded}");
                }

                buffer ??= System.Buffers.ArrayPool<byte>.Shared.Rent(checked((int)Math.Min(sliceSize, 64 * 1024)));
                if (!recorded && !IsUniformSlice(address, sliceSize, code, buffer))
                {
                    if (DccTraceEnabled)
                    {
                        TraceDcc(imageIdentifier, in request, address, sliceSize,
                            $"not-uniform code=0x{code:X2} {DescribeDccSlice(address, sliceSize, code)}");
                    }

                    ForgetDeferredDccSlice(imageIdentifier, address);
                    continue;
                }

                // Nothing marks the metadata once it is materialized, so the lookups that
                // follow see the same clear code again — in The Oregon Trail twenty-four
                // times a frame, two render-pass teardowns apiece. The claim remembers
                // what has been done; a rewrite of the metadata re-arms it, so a surface
                // the game clears every frame keeps working.
                var decision = DecideDccClear(imageIdentifier, address, sliceSize, code, imageWriteSequence);
                if (DccTraceActive)
                {
                    TraceDcc(imageIdentifier, in request, address, sliceSize, $"decision={decision} code=0x{code:X2} recorded={recorded}");
                }

                if (decision == DccClearDecision.Repeat)
                {
                    Interlocked.Increment(ref _dccRepeatsSkipped);
                    continue;
                }

                if (decision == DccClearDecision.Drop)
                {
                    Interlocked.Increment(ref _dccClearsDropped);
                    continue;
                }

                if (run.Count != 0 && !run.Extends(imageFirst + slice, in clear))
                {
                    ClearDccRun(imageIdentifier, in view, in run);
                    run = default;
                }

                if (run.Count == 0)
                {
                    run = new DccClearRun(imageFirst + slice, 0, clear);
                }

                run.Count++;
            }

            if (run.Count != 0)
            {
                ClearDccRun(imageIdentifier, in view, in run);
            }
        }
        finally
        {
            if (buffer is not null)
            {
                System.Buffers.ArrayPool<byte>.Shared.Return(buffer);
            }
        }
    }

    // Consecutive slices of one image that a DCC scan decided to clear to one colour.
    private struct DccClearRun(uint first, uint count, GpuClearColor clear)
    {
        public readonly uint First = first;
        public uint Count = count;
        public readonly GpuClearColor Clear = clear;

        // The colour is compared by its bits: the decoded value is what the pass loads.
        public readonly bool Extends(uint slice, in GpuClearColor clear) =>
            slice == First + Count &&
            clear.Uint32_0 == Clear.Uint32_0 && clear.Uint32_1 == Clear.Uint32_1 &&
            clear.Uint32_2 == Clear.Uint32_2 && clear.Uint32_3 == Clear.Uint32_3;
    }

    private void ClearDccRun(ResourceSlotIdentifier imageIdentifier, in ImageViewDescription view, in DccClearRun run)
    {
        using var held = _lock.Hold();
        if (_slots.TryGet(imageIdentifier) is null)
        {
            return;
        }

        Interlocked.Add(ref _dccSlicesCoalesced, run.Count - 1);
        ClearImageRange(
            imageIdentifier,
            view.Format,
            new SubresourceRange(view.BaseLevel, view.LevelCount, run.First, run.Count),
            in run.Clear);
    }

    private bool IsUniformSlice(ulong address, ulong size, byte code, byte[] buffer)
    {
        for (var position = 0UL; position < size;)
        {
            var span = buffer.AsSpan(0, (int)Math.Min(size - position, (ulong)buffer.Length));
            if (!_backing.TryReadBacking(address + position, span) || span.IndexOfAnyExcept(code) >= 0)
            {
                return false;
            }

            position += (ulong)span.Length;
        }

        return true;
    }

    /// <summary>
    /// Clears one subresource range of an image to a color, in the format of the view
    /// the clear was decoded for, and takes GPU ownership.
    ///
    /// Caller holds the lock.
    /// </summary>
    private void ClearImageRange(
        ResourceSlotIdentifier imageIdentifier,
        GpuFormat viewFormat,
        in SubresourceRange range,
        in GpuClearColor clear)
    {
        var image = _slots[imageIdentifier];
        if (image.Texture == 0)
        {
            return;
        }

        PromoteLinearView(image);
        if (image.Shape.TextureType != Mtl.TextureType3D)
        {
            MapSparseLayers(image, range.BaseLayer, range.LayerCount);
        }
        var shape = image.Shape;
        var volume = shape.TextureType == Mtl.TextureType3D;
        var layers = volume ? Math.Max(shape.Depth >> (int)range.BaseLevel, 1) : shape.ArrayLength;
        var fullImage = range.BaseLevel == 0 && range.LevelCount >= shape.MipLevels &&
            range.BaseLayer == 0 && range.LayerCount >= layers;
        WatchImage(imageIdentifier);
        // The clear is a write like any pass's, and the newest one to these bytes: an alias
        // the GPU wrote earlier hands them over now, as it would to a pass binding the image.
        // Left GPU-modified, it still looked like the owner when the cleared image bound as
        // a target next, and the bytes it published were reloaded over the clear. That is
        // how the scene's 1440p transients drew bands across Demon's Souls' 4K HUD layer,
        // which a DCC fast clear empties every frame over memory they reuse.
        DrainOverlappingDeferredRefreshesLocked(image);
        SynchronizeAliases(imageIdentifier, takesGpuOwnership: true);
        if (!fullImage)
        {
            // Taking GPU ownership drops the image's dirty state, so guest bytes not yet
            // in the texture are uploaded first; otherwise the slices this clear leaves
            // alone would keep stale contents and later be written back over the guest's.
            RefreshFromGuest(image);
        }

        // The clear colour was decoded for the view's format. A reinterpreting view of
        // the one subresource carries it there when the texture's own format differs —
        // an sRGB target over a UNORM image would otherwise store the unencoded value.
        // A volume has no 2D view of one plane, so it is cleared in its own format, and
        // only a colour-renderable view format is worth a view a render pass can clear.
        var reinterprets = !volume && PackedClearValue.SupportsDccFixedColor(viewFormat) &&
            MetalV2Formats.PixelFormat(viewFormat) != shape.PixelFormat;
        var layerCount = Math.Max(range.LayerCount, 1);
        var clearView = new ImageViewDescription(
            viewFormat,
            layerCount == 1 ? GpuImageViewType.Type2D : GpuImageViewType.Type2DArray,
            GpuImageAspect.ColorBit,
            range.BaseLevel,
            1,
            range.BaseLayer,
            layerCount,
            default,
            GpuImageUsage.ColorAttachmentBit);
        if (reinterprets)
        {
            EnsureViewUsageLocked(image, in clearView);
        }

        var texture = image.Texture;
        var clearRange = range;
        if (reinterprets)
        {
            var view = image.GetOrCreateView(clearView);
            if (view != image.Texture)
            {
                texture = view;
                clearRange = new SubresourceRange(0, 1, 0, layerCount);
            }
        }

        var width = Math.Max(image.Description.Extent.Width >> (int)range.BaseLevel, 1);
        var height = Math.Max(image.Description.Extent.Height >> (int)range.BaseLevel, 1);
        _owner.ClearTextureColor(texture, image.Shape.PixelFormat, in clearRange, width, height, in clear);
        image.RefreshComplete();
        image.MarkGpuModified();
        NoteLinearGpuWrite(image);
        image.GpuWriteSerial = _owner.CurrentSerial;
    }

    /// <summary>
    /// A guest fill whose range is exactly one cached image clears that image on the
    /// GPU instead of writing its bytes: the clear is what the title meant, and the
    /// image stays resident.
    /// </summary>
    public bool TryClearImageFromBuffer(ulong address, ulong size, uint packedClear)
    {
        if (!IsValidRange(address, size))
        {
            return false;
        }

        using var held = _lock.Hold();
        var selected = ResourceSlotIdentifier.Invalid;
        foreach (var imageIdentifier in FindImagesInRange(address, size, pageOverlap: false))
        {
            if (_slots.TryGet(imageIdentifier) is not { } candidate ||
                candidate.Description.Data.Address != address || candidate.Description.Data.Size != size ||
                candidate.Texture == 0)
            {
                continue;
            }

            if (selected.IsValid && selected != imageIdentifier)
            {
                return false;
            }

            selected = imageIdentifier;
        }

        if (!selected.IsValid)
        {
            return false;
        }

        var image = _slots[selected];
        PromoteLinearView(image);
        EnsureSparseBacked(image);
        ref readonly var description = ref image.Description;
        var range = new SubresourceRange(0, image.Shape.MipLevels, 0, image.Shape.ArrayLength);
        if (description.IsDepth)
        {
            if (!PackedClearValue.TryDecodeDepth(description.PixelFormat, packedClear, out var depth))
            {
                return false;
            }

            WatchImage(selected);
            _owner.ClearTextureDepthStencil(
                image.Texture,
                image.Shape.PixelFormat,
                in range,
                description.Extent.Width,
                description.Extent.Height,
                depth,
                0,
                clearDepth: true,
                clearStencil: false);
        }
        else
        {
            if (!PackedClearValue.TryDecodeBufferColor(description.PixelFormat, packedClear, out var color))
            {
                return false;
            }

            WatchImage(selected);
            _owner.ClearTextureColor(
                image.Texture,
                image.Shape.PixelFormat,
                in range,
                description.Extent.Width,
                description.Extent.Height,
                in color);
        }

        image.RefreshComplete();
        image.MarkGpuModified();
        NoteLinearGpuWrite(image);
        image.GpuWriteSerial = _owner.CurrentSerial;
        return true;
    }

    /// <summary>
    /// The deferred clear a DCC attachment should load, decoded from the metadata
    /// state; consuming it hands the clear to the render pass.
    /// </summary>
    public bool TryConsumeDccAttachmentClear(
        in ImageRequest request,
        in ColorTargetResolution resolution,
        out GpuClearColor clearValue)
    {
        clearValue = default;
        if (request.Description.Metadata.Kind != MetadataKind.Dcc || request.Description.Metadata.Range.Size != 0)
        {
            return false;
        }

        var view = request.View;
        var metadataAddress = request.Description.Metadata.Range.Address;
        if (!IsMetadataCleared(metadataAddress, view.BaseLayer, out var metadataValue) ||
            !TryDecodeDccClear(
                (byte)metadataValue,
                resolution.MetadataClearSupported,
                resolution.MetadataFixedClearSupported,
                resolution.ColorClearValue,
                out clearValue))
        {
            return false;
        }

        for (var layer = 1u; layer < view.LayerCount; layer++)
        {
            if (!IsMetadataCleared(metadataAddress, view.BaseLayer + layer))
            {
                return false;
            }
        }

        // Consume only after every layer of the view can be materialized together.
        for (var layer = 0u; layer < view.LayerCount; layer++)
        {
            _ = SetMetadataSlice(metadataAddress, view.BaseLayer + layer, false);
        }

        return true;
    }

    private static bool TryDecodeDccClear(
        byte code,
        bool registerClearSupported,
        bool fixedClearSupported,
        in GpuClearColor registerClear,
        out GpuClearColor clearValue)
    {
        clearValue = default;
        switch (code)
        {
            case 0x00:
                return true;
            case 0x20:
                clearValue = registerClear;
                return registerClearSupported;
            case 0x40:
                clearValue.Float32_3 = 1f;
                return fixedClearSupported;
            case 0x80:
                clearValue.Float32_0 = 1f;
                clearValue.Float32_1 = 1f;
                clearValue.Float32_2 = 1f;
                return fixedClearSupported;
            case 0xc0:
                clearValue.Float32_0 = 1f;
                clearValue.Float32_1 = 1f;
                clearValue.Float32_2 = 1f;
                clearValue.Float32_3 = 1f;
                return fixedClearSupported;
            default:
                return false;
        }
    }
}
