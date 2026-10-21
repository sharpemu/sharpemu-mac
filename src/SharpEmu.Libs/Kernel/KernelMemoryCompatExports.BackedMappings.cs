// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.HLE.GuestMemory;
using SharpEmu.Libs.Agc;

namespace SharpEmu.Libs.Kernel;

public static partial class KernelMemoryCompatExports
{
    private const int MemoryNoSpace = unchecked((int)0x8002000C);
    private const int MemoryAccessDenied = unchecked((int)0x8002000D);
    private const int MemoryFault = unchecked((int)0x8002000E);
    private const int MemoryInvalidArgument = unchecked((int)0x80020016);
    // Titles use sceKernelReserveVirtualRange as an address-space reservation
    // primitive. Some of them ask for hundreds of GiB even though only small
    // portions are ever mapped. Windows cannot always create one contiguous
    // placeholder that large after the executable and runtime are loaded.
    private const ulong SparseReservationThreshold = 0x10_0000_0000UL;
    // macOS reserves 0x10_0000_0000..0x70_0000_0000 for graphics memory (see
    // DefaultMapSearchBase), so sparse reservations start above that window there.
    private static ulong SparseReservationBase =>
        OperatingSystem.IsMacOS() ? 0x70_0000_0000UL : 0x40_0000_0000UL;
    private const ulong UserAddressLimit = 0xFC_0000_0000UL;
    private static FlexibleBackingPool _flexibleBacking =
        new(GuestMemoryLayout.FlexibleOffset, GuestMemoryLayout.FlexibleBytes);
    private static IGuestBackedSpace? _backingOwner;
    private static int _prtMapWarning;
    private static int _staleOwnerWarning;
    private static int _registeredFreeRangeWarning;

    internal static bool TryReadPrtBacking(IGuestBackedSpace backing, ulong address, Span<byte> destination)
    {
        var size = (ulong)destination.Length;
        lock (_memoryGate)
        {
            if (!ReferenceEquals(backing, _backingOwner) ||
                !KernelRuntimeCompatExports.ContainsPrtRange(address, size))
                return false;

            // Keep mappings stable while copying resident bytes and clearing reserved gaps.
            destination.Clear();
            foreach (var region in GetMappingSlices(address, size))
            {
                if (region.IsReserved)
                    continue;
                if (backing.IsBackedRange(region.Address, region.Length))
                {
                    if (!backing.TryReadBacking(region.Address, destination.Slice((int)(region.Address - address), (int)region.Length)))
                        return false;
                    continue;
                }

                var regionEnd = region.Address + region.Length;
                for (var chunk = region.Address; chunk < regionEnd;)
                {
                    var next = Math.Min(regionEnd, (chunk & ~(PrtResidencyProbeBytes - 1)) + PrtResidencyProbeBytes);
                    if (backing.IsBackedRange(chunk, next - chunk) &&
                        !backing.TryReadBacking(chunk, destination.Slice((int)(chunk - address), (int)(next - chunk))))
                        return false;
                    chunk = next;
                }
            }
            return true;
        }
    }

    private const ulong PrtResidencyProbeBytes = 0x4000;

    internal static bool TryWritePrtBacking(IGuestBackedSpace backing, ulong address, ReadOnlySpan<byte> source)
    {
        var size = (ulong)source.Length;
        lock (_memoryGate)
        {
            if (!ReferenceEquals(backing, _backingOwner) ||
                !KernelRuntimeCompatExports.ContainsPrtRange(address, size))
                return false;

            var regions = GetMappingSlices(address, size);
            if (regions.Any(region => !region.IsReserved && !backing.IsBackedRange(region.Address, region.Length)))
                return false;

            foreach (var region in regions)
            {
                if (region.IsReserved)
                    continue;
                if (!backing.TryWriteBacking(region.Address, source.Slice((int)(region.Address - address), (int)region.Length)))
                    return false;
            }
            return true;
        }
    }

    internal static FlexibleBackingPool SetFlexibleBackingForTests(FlexibleBackingPool pool)
    {
        lock (_memoryGate)
        {
            if (_flexibleBacking.Used != 0)
                throw new InvalidOperationException("Flexible mappings are still active.");
            var previous = _flexibleBacking;
            _flexibleBacking = pool;
            return previous;
        }
    }

    public static void ResetBackingMappings(IGuestBackedSpace? owner)
    {
        RunMappingTransaction(() =>
        {
            ResetBackingMappingsCore(owner);
            return 0;
        });
    }

    private static void ResetBackingMappingsCore(IGuestBackedSpace? owner)
    {
        lock (_memoryGate)
        {
            if (owner is null || !ReferenceEquals(owner, _backingOwner))
                return;
            foreach (var region in _mappedRegions.Values)
                GuestGpuMemoryHook.NoteUnmapped(region.Address, region.Length);
            _mappedRegions.Clear();
            _directAllocations.Reset();
            _flexibleBacking.Reset();
            ResetMemoryPoolLocked();
            ResetJitSharedMemoryLocked();
            _nextVirtualAddress = 0;
            _backingOwner = null;
        }
    }

    private static IGuestBackedSpace? ResolveBackingSpace(CpuContext ctx)
    {
        if (!KernelVirtualRangeAllocator.TryResolveAddressSpace(ctx.Memory, out var addressSpace) ||
            addressSpace is not IGuestBackedSpace space)
            return null;
        if (_backingOwner is not null && !ReferenceEquals(_backingOwner, space))
        {
            // A guest call must not take the process down: the caller reports the failure.
            if (Interlocked.Exchange(ref _staleOwnerWarning, 1) == 0)
                Console.Error.WriteLine("[LOADER][WARN] A memory call used a new guest address space while the previous one is still active.");
            return null;
        }
        _backingOwner = space;
        return space;
    }

    private static bool TryDecodeMappedProtection(int value, out GuestPageProtection protection)
    {
        protection = GuestPageProtection.None;
        // CPU, GPU, AMPR and ACP bits are the whole sys/dmem.h set; AMPR/ACP-only access has
        // no CPU or GPU view to protect. GPU-only access stays host-readable: the GPU layers
        // read and write those pages through the host view.
        if ((value & ~0x3FF) != 0 || (value != 0 && (value & 0x37) == 0))
            return false;
        if ((value & (OrbisProtCpuRead | OrbisProtGpuRead)) != 0)
            protection |= GuestPageProtection.Read;
        if ((value & (OrbisProtCpuWrite | OrbisProtGpuWrite)) != 0)
            protection |= GuestPageProtection.Read | GuestPageProtection.Write;
        if ((value & OrbisProtCpuExec) != 0)
            protection |= GuestPageProtection.Execute;
        return true;
    }

    // Alignment is zero or a power of two; callers raise it to at least one 16 KiB page.
    private static bool IsValidMapRange(ulong length, ulong alignment) =>
        length != 0 && IsAligned(length, OrbisPageSize) && (alignment & (alignment - 1)) == 0;

    private static MappedRegion SliceMapping(MappedRegion region, ulong start, ulong end) => region with
    {
        Address = start,
        Length = end - start,
        DirectStart = region.IsDirect ? region.DirectStart + start - region.Address : 0,
        BackingOffset = region.IsBacked ? region.BackingOffset + start - region.Address : 0,
    };

    // Empty or reserved-only address space that no GPU-registered span reaches.
    private static bool IsUntouchedByGpu(ulong address, ulong size)
    {
        if (size == 0 || ulong.MaxValue - address < size)
            return false;
        lock (_memoryGate)
        {
            foreach (var region in GetMappingSlices(address, size))
            {
                if (!region.IsReserved)
                    return false;
            }
        }

        return !GuestGpuMemoryHook.OverlapsRegistered(address, size);
    }

    private static MappedRegion[] GetMappingSlices(ulong address, ulong size, bool clip = true)
    {
        var end = address + size;
        List<MappedRegion>? slices = null;
        foreach (var region in _mappedRegions.FromAddress(address))
        {
            if (region.Address >= end)
                break;
            if (address >= region.Address + region.Length)
                continue;
            (slices ??= []).Add(clip ? SliceMapping(region, Math.Max(address, region.Address),
                Math.Min(end, region.Address + region.Length)) : region);
        }
        return slices?.ToArray() ?? [];
    }

    private static bool MappingsCoverRange(MappedRegion[] regions, ulong address, ulong size)
    {
        var current = address;
        foreach (var region in regions)
        {
            if (region.Address != current)
                return false;
            current += region.Length;
        }
        return current == address + size;
    }

    private static void RemoveMappingLocked(ulong address, ulong size)
    {
        using var removalProfile = GuestMemoryProfile.Measure(GuestMemoryProfile.Operation.KernelMappingRemoval);
        ReplaceMappedRegionRangeLocked(new MappedRegion(address, size, 0, false, false, 0, IsReserved: true));
        using (GuestMemoryProfile.Measure(GuestMemoryProfile.Operation.KernelMappingTreeRemoval))
            _mappedRegions.Remove(address);
    }

    private static void RestoreViews(IGuestBackedSpace space, IReadOnlyList<MappedRegion> regions)
    {
        for (var index = regions.Count - 1; index >= 0; index--)
        {
            var region = regions[index];
            if (!TryDecodeMappedProtection(region.Protection, out var protection) ||
                !space.TryMapBacked(region.Address, region.Length, region.BackingOffset, protection, out _))
                Environment.FailFast("Cannot restore a guest backing view.");
        }
    }

    private static void RestoreGpuMappings(IEnumerable<MappedRegion> regions)
    {
        foreach (var region in regions)
            if (!region.IsReserved && TryDecodeMappedProtection(region.Protection, out var mode))
                GuestGpuMemoryHook.NoteMapped(region.Address, region.Length, mode);
    }

    private static bool TryUnmapViews(IGuestBackedSpace space, MappedRegion[] regions)
    {
        using var viewsProfile = GuestMemoryProfile.Measure(GuestMemoryProfile.Operation.KernelUnmapViews);
        var removed = new List<MappedRegion>();
        using (GuestMemoryProfile.Measure(GuestMemoryProfile.Operation.KernelUnmapValidation))
        {
            foreach (var region in regions)
            {
                if (region.IsReserved)
                    continue;
                if (!region.IsBacked || !space.IsBackedView(region.Address))
                    return false;
            }
        }
        using (GuestMemoryProfile.Measure(GuestMemoryProfile.Operation.KernelUnmapGpuNotification))
        {
            foreach (var region in regions)
                GuestGpuMemoryHook.NoteUnmapped(region.Address, region.Length);
        }
        foreach (var region in regions)
        {
            if (region.IsReserved)
                continue;
            bool unmapped;
            using (GuestMemoryProfile.Measure(GuestMemoryProfile.Operation.KernelUnmapBackingCall))
                unmapped = space.TryUnmapBacked(region.Address, region.Length);
            if (!unmapped)
            {
                using var rollbackProfile = GuestMemoryProfile.Measure(GuestMemoryProfile.Operation.KernelUnmapRollback);
                RestoreViews(space, removed);
                RestoreGpuMappings(regions);
                return false;
            }
            removed.Add(region);
        }
        return true;
    }

    // Clears [address, address + size) for a fixed mapping. NO_OVERWRITE refuses only live
    // mappings: a reserved range is free to map into. The replaced slices come back, still
    // owning their backing: the caller hands it back with CommitReplacedLocked once the new
    // mapping stands, or puts the slices back with RestoreReplacedLocked when it fails.
    private static bool TryReplaceWithHole(IGuestBackedSpace space, ulong address, ulong size, bool noOverwrite,
        out MappedRegion[] replaced)
    {
        replaced = [];
        var regions = GetMappingSlices(address, size);
        if (noOverwrite && regions.Any(region => !region.IsReserved))
            return false;
        if (MappingsCoverRange(regions, address, size) && regions.All(region => region.IsReserved))
        {
            if (!space.TryHoldRange(address, size))
                return false;
            UnregisterFreeRange(address, size);
            replaced = regions;
            return true;
        }
        if (!TryUnmapViews(space, regions))
            return false;
        if (!space.TryHoldRange(address, size))
        {
            RestoreViews(space, regions.Where(region => !region.IsReserved).ToArray());
            RestoreGpuMappings(regions);
            return false;
        }
        ReplaceMappedRegionRangeLocked(new MappedRegion(address, size, 0, false, false, 0, IsReserved: true));
        replaced = regions;
        return true;
    }

    private static void CommitReplacedLocked(MappedRegion[] replaced, bool releaseToHost = true)
    {
        foreach (var region in replaced)
            ReleaseBackingOwnershipLocked(region, releaseToHost);
    }

    // After a replacement that gave up its old backing before the new mapping stood
    // (releaseToHost: false, so a failure could still restore the old bytes): the old flexible
    // bytes the new mapping did not take go to the host now.
    private static void ReleaseReplacedFlexibleLocked(MappedRegion[] replaced)
    {
        foreach (var region in replaced)
            if (region.IsFlexible && !region.IsReserved)
                ReleaseFreedBackingLocked(region.BackingOffset, region.Length,
                    _flexibleBacking.FreeRangeContaining(region.BackingOffset));
    }

    // Unmapped pages hand their backing back to whoever lent it, and the host gets the whole
    // chunks of it that are now free (a chunked guest backing; nothing otherwise).
    private static void ReleaseBackingOwnershipLocked(MappedRegion region, bool releaseToHost = true)
    {
        if (region.IsReserved)
            return;
        if (region.IsFlexible)
        {
            foreach (var (offset, size) in _flexibleBacking.Release(region.Address, region.Length))
                if (releaseToHost)
                    ReleaseFreedBackingLocked(offset, size, _flexibleBacking.FreeRangeContaining(offset));
        }
        else if (region.IsPooled)
        {
            ReturnPoolPagesLocked(region.BackingOffset, region.Length);
            if (releaseToHost)
                ReleaseFreedBackingLocked(region.BackingOffset, region.Length,
                    PoolFreeRangeContainingLocked(region.BackingOffset));
        }
    }

    // [start, start + length) just became free inside freeRun, the whole free range holding it.
    // Every chunk of the backing that the freed bytes touch and that lies wholly inside the free
    // range goes back to the host, so a chunk only partly freed now is released by whichever
    // later free completes it. The PS5 leaves released memory's contents undefined (a fresh
    // allocation is not promised zero: docs/macos/ps5-internals.md §6 points 3-4 and the open
    // question on dmem zeroing), and a released chunk reads zero, which is one such value.
    private static void ReleaseFreedBackingLocked(ulong start, ulong length, (ulong Start, ulong Length) freeRun)
    {
        if (_backingOwner is not { } space || length == 0 || freeRun.Length == 0)
            return;
        var chunk = space.BackingChunkBytes;
        if (chunk == 0)
            return;
        var first = Math.Max(freeRun.Start, start - start % chunk);
        var last = Math.Min(freeRun.Start + freeRun.Length, AlignUp(start + length, chunk));
        first = AlignUp(first, chunk);
        last -= last % chunk;
        if (first < last)
            _ = space.TryReleaseBacking(first, last - first);
    }

    // Undoes TryReplaceWithHole after the new mapping failed: the views, GPU registration,
    // backing ownership and table records come back as they were.
    private static void RestoreReplacedLocked(IGuestBackedSpace space, ulong address, ulong size, MappedRegion[] replaced,
        bool ownershipReleased = false)
    {
        if (replaced.Length == 0)
        {
            RemoveMappingLocked(address, size);
            return;
        }
        if (MappingsCoverRange(replaced, address, size) && replaced.All(region => region.IsReserved))
            return;
        var backed = replaced.Where(region => region.IsBacked).ToArray();
        if (ownershipReleased)
            foreach (var region in backed)
                ReclaimBackingOwnershipLocked(region);
        RestoreViews(space, backed);
        RestoreGpuMappings(replaced);
        RemoveMappingLocked(address, size);
        foreach (var region in replaced)
            ReplaceMappedRegionRangeLocked(region);
    }

    private static void ReclaimBackingOwnershipLocked(MappedRegion region)
    {
        if (region.IsFlexible && !_flexibleBacking.TryReclaim(region.Address, region.Length, region.BackingOffset))
            Environment.FailFast("Cannot restore flexible backing ownership.");
        if (region.IsPooled && !region.IsReserved && !TryTakePoolPagesLocked(region.BackingOffset, region.Length))
            Environment.FailFast("Cannot restore memory pool ownership.");
    }

    // Free and reserved ranges are never registered with the GPU, and unregistering hands the GPU
    // worker a request to wait for. A direct transaction holds the mapping lock that GPU reads take,
    // so waiting there would deadlock; it relies on this range having nothing to unregister.
    private static void UnregisterFreeRange(ulong address, ulong size)
    {
        if (!GuestGpuMemoryHook.OverlapsRegistered(address, size))
            return;
        if (_directMappingTransaction)
        {
            // RunMappingTransaction rechecks the range under the mapping lock before it runs a
            // transaction off the GPU worker, so this is a stale registration; waiting here
            // would deadlock, and a guest call must not throw.
            if (Interlocked.Exchange(ref _registeredFreeRangeWarning, 1) == 0)
                Console.Error.WriteLine(
                    $"[LOADER][WARN] A free guest range is still registered with the GPU: address=0x{address:X16} size=0x{size:X}.");
            return;
        }
        GuestGpuMemoryHook.NoteUnmapped(address, size);
    }

    private static bool TrySelectBackingAddress(IGuestBackedSpace space, ulong requested, ulong length,
        ulong alignment, ulong flags, out ulong address, bool reuseReservation = true)
    {
        if (!TrySelectReplacingBackingAddress(space, requested, length, alignment, flags, out address, out var replaced,
                reuseReservation))
            return false;
        CommitReplacedLocked(replaced);
        return true;
    }

    // The fixed path also reports the slices it replaced, for RestoreReplacedLocked.
    private static bool TrySelectReplacingBackingAddress(IGuestBackedSpace space, ulong requested, ulong length,
        ulong alignment, ulong flags, out ulong address, out MappedRegion[] replaced, bool reuseReservation = true)
    {
        using var selectionProfile = GuestMemoryProfile.Measure(GuestMemoryProfile.Operation.MappingAddressSelection);
        address = 0;
        replaced = [];
        if ((flags & OrbisKernelMapFixed) != 0)
        {
            // The host cannot place anything in its blocked window, and nothing here may
            // translate a guest address; fail before the request touches any state.
            if (GuestVaHole.IsFixedRequestBlocked(requested, length))
                return false;
            if (!TryReplaceWithHole(space, requested, length, (flags & OrbisKernelMapNoOverwrite) != 0, out replaced))
            {
                if (requested >= 0x10_0000_0000UL && requested < 0xFC_0000_0000UL &&
                    Interlocked.Exchange(ref _prtMapWarning, 1) == 0)
                    Console.Error.WriteLine(
                        $"[LOADER][WARN] Shared mapping cannot replace the reserved aperture: address=0x{requested:X16} size=0x{length:X} flags=0x{flags:X} " +
                        $"regions=[{string.Join(",", GetMappingSlices(requested, length).Select(region => $"0x{region.Address:X}+0x{region.Length:X}:{(region.IsReserved ? "reserved" : region.IsDirect ? "direct" : region.IsFlexible ? "flexible" : "other")}:backed={space.IsBackedView(region.Address)}"))}].");
                return false;
            }
            address = requested;
            return true;
        }
        var desired = requested != 0 ? requested : DefaultMapSearchBase;
        while (true)
        {
            desired = GuestVaHole.SkipForSearch(desired, length);
            var hintedRegions = reuseReservation && desired == requested && requested != 0
                ? GetMappingSlices(requested, length) : [];
            var reusableHint = hintedRegions.Length != 0 && requested % alignment == 0 &&
                MappingsCoverRange(hintedRegions, requested, length) && hintedRegions.All(region => region.IsReserved);
            if (!reusableHint)
                desired = FindAvailableMappingAddress(desired, length, alignment);
            if (desired == 0)
                return false;
            if (GuestVaHole.SkipForSearch(desired, length) != desired)
            {
                desired = GuestVaHole.SkipForSearch(desired, length);
                continue;
            }
            // On Windows, search lower addresses before using the startup data reservation.
            var held = requested == 0 && !OperatingSystem.IsWindows()
                ? space.TryHoldAvailableRange(desired, length, alignment, out address)
                : space.TryHoldRangeAtOrAbove(desired, length, alignment, out address);
            if (!held)
                return false;
            var overlap = GetMappingSlices(address, length, clip: false);
            if (overlap.Length == 0 || (reuseReservation && address == requested &&
                MappingsCoverRange(GetMappingSlices(address, length), address, length) && overlap.All(region => region.IsReserved)))
                break;
            // Skip the complete reservation, not only the requested slice.
            desired = overlap[^1].Address + overlap[^1].Length;
        }
        if (address == 0)
            return false;
        UnregisterFreeRange(address, length);
        return true;
    }

    // Skip kernel reservations before asking the host to reserve a candidate.
    private static ulong FindAvailableMappingAddress(ulong desired, ulong length, ulong alignment)
    {
        if (_mappedRegions.TryFindAvailableAddress(desired, length, alignment, out var available))
            return available;
        var padding = (alignment - desired % alignment) % alignment;
        if (padding > ulong.MaxValue - desired)
            return 0;
        var candidate = desired + padding;
        foreach (var region in _mappedRegions.FromAddress(candidate))
        {
            if (length > ulong.MaxValue - candidate)
                return 0;
            if (region.Address >= candidate + length)
                break;
            var regionEnd = region.Address + region.Length;
            if (regionEnd <= candidate)
                continue;
            padding = (alignment - regionEnd % alignment) % alignment;
            if (padding > ulong.MaxValue - regionEnd)
                return 0;
            candidate = regionEnd + padding;
        }
        return length <= ulong.MaxValue - candidate ? candidate : 0;
    }

    internal static int ReserveBackingRange(CpuContext ctx, ulong pointer, ulong length, ulong flags, ulong alignment)
        => RunMappingTransaction(
            () => ReserveBackingRangeCore(ctx, pointer, length, flags, alignment),
            () => (flags & OrbisKernelMapFixed) == 0 ||
                (ctx.TryReadUInt64(pointer, out var requested) && IsUntouchedByGpu(requested, length)));

    // A batch wraps its entries in one transaction, so the entries must not hand off again.
    [ThreadStatic]
    private static bool _insideMappingTransaction;

    // Set while a transaction runs on the guest thread instead of the GPU worker.
    [ThreadStatic]
    private static bool _directMappingTransaction;

    private static int RunMappingTransaction(Func<int> transaction, Func<bool>? onlyAddsOutsideGpuMemory = null)
    {
        // A nested transaction (a batch entry) runs inside its batch's: the batch already holds
        // the mapping lock when it runs on the guest thread, so this test comes first.
        if (_insideMappingTransaction)
            return transaction();
        // GPU handoff must precede locks needed by image and buffer reads.
        if (Monitor.IsEntered(_memoryGate))
            throw new InvalidOperationException("Cannot start a mapping transaction while holding the mapping lock.");
        if (GuestGpuMemoryHook.Current is not { } memory)
            return transaction();

        // A transaction that only places new mappings in address space the GPU has never been given
        // (kernel-chosen placements, or empty/reserved ranges outside every registered span) cannot
        // race a GPU read and unregisters nothing. It runs here instead of queueing behind the GPU
        // worker, which during loading is busy compiling pipelines: Demon's Souls spent seconds per
        // reservation waiting for a queue slot while the change itself took a millisecond.
        if (onlyAddsOutsideGpuMemory?.Invoke() == true)
        {
            _insideMappingTransaction = _directMappingTransaction = true;
            try
            {
                // Another guest thread can map (and register with the GPU) the range between the
                // unlocked check and the transaction. Every kernel mapping registers under this
                // lock, so the recheck here holds until the transaction is done; when it fails
                // the change goes through the GPU worker like any other.
                lock (_memoryGate)
                {
                    if (onlyAddsOutsideGpuMemory())
                        return transaction();
                }
            }
            finally
            {
                _insideMappingTransaction = _directMappingTransaction = false;
            }
        }

        var result = MemoryFault;
        memory.RunMappingChange(() =>
        {
            _insideMappingTransaction = true;
            try
            {
                result = transaction();
            }
            finally
            {
                _insideMappingTransaction = false;
            }
        });
        return result;
    }

    // A memory pool passes the requested address by value and tags the reservation as pooled.
    private static int ReserveBackingRangeCore(CpuContext ctx, ulong pointer, ulong length, ulong flags, ulong alignment,
        ulong? requestedAddress = null, MappingTraits traits = MappingTraits.None)
    {
        if (pointer == 0 || !IsValidMapRange(length, alignment))
            return MemoryInvalidArgument;
        var requested = requestedAddress ?? 0;
        if (requestedAddress is null && !ctx.TryReadUInt64(pointer, out requested))
        {
            Console.Error.WriteLine($"[LOADER][WARN] reserve_virtual pointer read failed pointer=0x{pointer:X16} size=0x{length:X}");
            return MemoryFault;
        }
        alignment = Math.Max(alignment, OrbisPageSize);
        if ((flags & OrbisKernelMapFixed) != 0 &&
            (requested == 0 || !IsAligned(requested, OrbisPageSize) || requested > ulong.MaxValue - length))
            return MemoryInvalidArgument;
        // Probe the result slot first, so a fault leaves no reservation behind.
        if (!ctx.TryWriteUInt64(pointer, requested))
            return MemoryFault;
        lock (_memoryGate)
        {
            var space = ResolveBackingSpace(ctx);
            var address = 0UL;
            var sparseSelected = (flags & OrbisKernelMapFixed) == 0 &&
                TrySelectSparseReservationAddress(requested, length, alignment, flags, out address);
            var selected = sparseSelected;
            var replaced = Array.Empty<MappedRegion>();
            if (!selected && space is not null)
            {
                selected = TrySelectReplacingBackingAddress(space, requested, length, alignment, flags,
                    out address, out replaced, reuseReservation: false);
            }

            // A sparse reservation is only a guest mapping-table entry and
            // intentionally does not need a host backing space yet. The host
            // range is acquired later when a subrange is committed or mapped.
            if (!selected || (!sparseSelected && space is null))
            {
                Console.Error.WriteLine($"[LOADER][WARN] reserve_virtual address selection failed requested=0x{requested:X16} size=0x{length:X} sparse={sparseSelected} backing={space is not null}");
                return MemoryNoSpace;
            }
            if (ShouldTraceDirectMemory())
                Console.Error.WriteLine($"[LOADER][TRACE] reserve_virtual selected address=0x{address:X16} requested=0x{requested:X16} size=0x{length:X} sparse={sparseSelected}");
            if (!ctx.TryWriteUInt64(pointer, address))
            {
                Console.Error.WriteLine($"[LOADER][WARN] reserve_virtual pointer write failed pointer=0x{pointer:X16} address=0x{address:X16}");
                if (!sparseSelected)
                    RestoreReplacedLocked(space!, address, length, replaced);
                return MemoryFault;
            }
            CommitReplacedLocked(replaced);
            ReplaceMappedRegionRangeLocked(new MappedRegion(address, length, 0, false, false, 0, IsReserved: true,
                Traits: traits));
            if (ShouldTraceDirectMemory())
                Console.Error.WriteLine($"[LOADER][TRACE] reserve_virtual address=0x{address:X} requested=0x{requested:X} size=0x{length:X} flags=0x{flags:X}");
            return 0;
        }
    }

    // Keep very large virtual reservations in the guest mapping table without
    // requiring one equally large host placeholder. Actual backed mappings still
    // reserve their individual host ranges when they are committed below this
    // reservation. Fixed reservations never take this path: they must preserve
    // the exact host address semantics of the original request.
    private static bool TrySelectSparseReservationAddress(
        ulong requested,
        ulong length,
        ulong alignment,
        ulong flags,
        out ulong address)
    {
        address = 0;
        if ((flags & OrbisKernelMapFixed) != 0 || length < SparseReservationThreshold ||
            length > UserAddressLimit || alignment == 0)
        {
            return false;
        }

        var candidate = requested == 0
            ? SparseReservationBase
            : Math.Max(requested, SparseReservationBase);
        candidate = AlignUp(candidate, alignment);
        if (candidate == 0)
        {
            return false;
        }

        // The requested address is a hint for a non-fixed reservation. If the
        // hint is too high to contain the complete range, restart from the
        // sparse aperture instead of treating the hint as a lower bound.
        if (candidate > UserAddressLimit - length)
        {
            candidate = AlignUp(SparseReservationBase, alignment);
            // A range too large for the aperture can only live at the guest's own hint.
            if (candidate > UserAddressLimit - length)
                candidate = AlignUp(requested, alignment);
            if (candidate == 0 || candidate > UserAddressLimit - length)
                return false;
        }

        foreach (var region in _mappedRegions.Values)
        {
            if (region.Address >= candidate + length)
            {
                break;
            }

            var regionEnd = region.Address + region.Length;
            if (regionEnd <= candidate)
            {
                continue;
            }

            candidate = AlignUp(regionEnd, alignment);
            if (candidate == 0 || candidate > UserAddressLimit - length)
            {
                return false;
            }
        }

        if (candidate == 0 || candidate > UserAddressLimit - length ||
            _mappedRegions.Values.Any(region =>
                region.Address < candidate + length && candidate < region.Address + region.Length))
        {
            return false;
        }

        address = candidate;
        return true;
    }

    private static bool HasPhysicalSpan(ulong start, ulong length)
        => _directAllocations.ContainsAllocatedRange(start, length);

    private static bool TryReleaseDirectMemoryRangeLocked(CpuContext ctx, ulong start, ulong length)
    {
        using var releaseProfile = GuestMemoryProfile.Measure(GuestMemoryProfile.Operation.KernelDirectRelease);
        if (!HasPhysicalSpan(start, length))
            return false;
        var end = start + length;
        MappedRegion[] aliases;
        using (GuestMemoryProfile.Measure(GuestMemoryProfile.Operation.DirectReleaseAliasSearch))
        {
            aliases = _mappedRegions.FindDirectOverlaps(start, length)
                .Select(region => SliceMapping(region,
                    region.Address + Math.Max(start, region.DirectStart) - region.DirectStart,
                    region.Address + Math.Min(end, region.DirectStart + region.Length) - region.DirectStart)).ToArray();
        }
        if (aliases.Length != 0)
        {
            var space = ResolveBackingSpace(ctx);
            if (space is null || !TryUnmapViews(space, aliases))
                return false;
        }
        foreach (var alias in aliases)
        {
            RemoveMappingLocked(alias.Address, alias.Length);
        }
        using (GuestMemoryProfile.Measure(GuestMemoryProfile.Operation.DirectAllocationRelease))
            _directAllocations.ReleaseRange(start, length);
        ReleaseFreedBackingLocked(start, length, _directAllocations.FreeRangeContaining(start));
        return true;
    }

    private static int ProtectMappedRange(CpuContext ctx, ulong address, ulong length, int protection, int? memoryType = null)
    {
        if (GuestGpuMemoryHook.Traces(address, length))
            GuestGpuMemoryHook.Trace(address, length, $"kernel-protect raw=0x{protection:X}");
        if (address == 0 || length == 0 || !TryDecodeMappedProtection(protection, out var mode) ||
            !TryNormalizeProtectRange(address, length, out var start, out var size))
            return MemoryInvalidArgument;
        lock (_memoryGate)
        {
            var regions = GetMappingSlices(start, size);
            var backed = regions.Any(region => region.IsBacked);
            if (backed)
            {
                start = address & ~(OrbisPageSize - 1);
                if (length > ulong.MaxValue - (address - start) - (OrbisPageSize - 1))
                    return MemoryInvalidArgument;
                size = AlignUp(length + address - start, OrbisPageSize);
                regions = GetMappingSlices(start, size);
                if (!MappingsCoverRange(regions, start, size) || regions.Any(region => region.IsReserved) ||
                    !KernelVirtualRangeAllocator.TryResolveAddressSpace(ctx.Memory, out var space) ||
                    (!GuestGpuMemoryHook.NoteProtected(start, size, mode) && !space.TryProtect(start, size, mode)))
                    return MemoryAccessDenied;
            }
            else if (!GuestGpuMemoryHook.NoteProtected(start, size, mode) && !TryProtectHostRange(start, size, protection))
                return unchecked((int)0x80020002);
            _ = TryApplyMappedRegionProtectionLocked(start, size, protection, memoryType);
            return 0;
        }
    }
}
