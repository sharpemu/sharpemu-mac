// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;

namespace SharpEmu.Libs.Kernel;

// Memory pools: a pool reservation is a virtual range tagged Pooled; Expand moves direct
// memory into the pool's supply, Commit backs 64 KiB pages of a reservation from that supply
// and Decommit hands them back, leaving the reservation in place. Committed pages are backed
// views like direct ones, but their physical pages belong to the pool, not to a direct
// mapping, so releasing them returns them to the supply.
public static partial class KernelMemoryCompatExports
{
    private const ulong PoolReserveGranularity = 2UL * 1024 * 1024;
    private const ulong PoolPageSize = 64UL * 1024;
    private const int PoolOpCommit = 1;
    private const int PoolOpDecommit = 2;
    private const int PoolOpProtect = 3;
    private const int PoolOpTypeProtect = 4;
    private const int PoolOpMove = 5;
    private const int PoolBatchEntrySize = 32;

    // Physical ranges the pool owns that no commit holds, keyed by start.
    private static readonly SortedList<ulong, ulong> _poolFree = new();
    private static ulong _poolFreeBytes;
    private static ulong _poolCommittedBytes;

    private static void ResetMemoryPoolLocked()
    {
        _poolFree.Clear();
        _poolFreeBytes = 0;
        _poolCommittedBytes = 0;
    }

    private static void ReturnPoolPagesLocked(ulong start, ulong length)
    {
        AddPoolSupplyLocked(start, length);
        _poolCommittedBytes -= Math.Min(_poolCommittedBytes, length);
    }

    private static void AddPoolSupplyLocked(ulong start, ulong length)
    {
        var end = start + length;
        for (var index = _poolFree.Count - 1; index >= 0; index--)
        {
            var rangeStart = _poolFree.Keys[index];
            var rangeEnd = rangeStart + _poolFree.Values[index];
            if (rangeStart > end || rangeEnd < start)
                continue;
            start = Math.Min(start, rangeStart);
            end = Math.Max(end, rangeEnd);
            _poolFree.RemoveAt(index);
        }
        _poolFree.Add(start, end - start);
        _poolFreeBytes += length;
    }

    // The uncommitted pool range that holds start, merged with its neighbours; empty when none does.
    private static (ulong Start, ulong Length) PoolFreeRangeContainingLocked(ulong start)
    {
        for (var index = _poolFree.Count - 1; index >= 0; index--)
        {
            var rangeStart = _poolFree.Keys[index];
            if (rangeStart > start)
                continue;
            return start - rangeStart < _poolFree.Values[index] ? (rangeStart, _poolFree.Values[index]) : default;
        }
        return default;
    }

    // Takes exactly [start, start + length) back out of the supply.
    private static bool TryTakePoolPagesLocked(ulong start, ulong length)
    {
        for (var index = _poolFree.Count - 1; index >= 0; index--)
        {
            var rangeStart = _poolFree.Keys[index];
            if (rangeStart > start)
                continue;
            var rangeEnd = rangeStart + _poolFree.Values[index];
            if (rangeEnd < start + length)
                return false;
            _poolFree.RemoveAt(index);
            if (rangeStart < start)
                _poolFree.Add(rangeStart, start - rangeStart);
            if (start + length < rangeEnd)
                _poolFree.Add(start + length, rangeEnd - start - length);
            _poolFreeBytes -= length;
            _poolCommittedBytes += length;
            return true;
        }
        return false;
    }

    // Plans physical runs for a commit, lowest supply first; nothing is taken yet.
    private static List<(ulong Physical, ulong Length)> PlanPoolPagesLocked(ulong length)
    {
        var runs = new List<(ulong, ulong)>();
        var remaining = length;
        foreach (var range in _poolFree)
        {
            var count = Math.Min(remaining, range.Value);
            runs.Add((range.Key, count));
            remaining -= count;
            if (remaining == 0)
                break;
        }
        return runs;
    }

    private static bool IsPoolRange(ulong address, ulong length) =>
        address != 0 && length != 0 && IsAligned(address, PoolPageSize) && IsAligned(length, PoolPageSize) &&
        address <= ulong.MaxValue - length;

    [SysAbiExport(
        Nid = "pU-QydtGcGY",
        ExportName = "sceKernelMemoryPoolReserve",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelMemoryPoolReserve(CpuContext ctx)
    {
        var requested = ctx[CpuRegister.Rdi];
        var length = ctx[CpuRegister.Rsi];
        var alignment = ctx[CpuRegister.Rdx];
        var flags = ctx[CpuRegister.Rcx] & (OrbisKernelMapFixed | OrbisKernelMapNoOverwrite);
        var output = ctx[CpuRegister.R8];
        if ((flags & OrbisKernelMapFixed) != 0)
            requested = GuestVaHole.RelocateFixedRequest(requested, length);
        if (output == 0 || length == 0 || !IsAligned(length, PoolReserveGranularity) ||
            (alignment != 0 && ((alignment & (alignment - 1)) != 0 || !IsAligned(alignment, PoolReserveGranularity))))
            return MemoryInvalidArgument;
        alignment = Math.Max(alignment, PoolReserveGranularity);
        if ((flags & OrbisKernelMapFixed) != 0 && !IsAligned(requested, alignment))
            return MemoryInvalidArgument;
        return RunMappingTransaction(
            () => ReserveBackingRangeCore(ctx, output, length, flags, alignment, requested, MappingTraits.Pooled),
            () => (flags & OrbisKernelMapFixed) == 0 || IsUntouchedByGpu(requested, length));
    }

    [SysAbiExport(
        Nid = "qCSfqDILlns",
        ExportName = "sceKernelMemoryPoolExpand",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelMemoryPoolExpand(CpuContext ctx)
    {
        var searchStart = ctx[CpuRegister.Rdi];
        var searchEnd = ctx[CpuRegister.Rsi];
        var length = ctx[CpuRegister.Rdx];
        var alignment = ctx[CpuRegister.Rcx];
        var output = ctx[CpuRegister.R8];
        if ((long)searchStart < 0 || searchEnd <= searchStart || length == 0 || !IsAligned(length, PoolPageSize) ||
            output == 0 ||
            (alignment != 0 && ((alignment & (alignment - 1)) != 0 || !IsAligned(alignment, PoolPageSize))))
            return MemoryInvalidArgument;
        if (!ctx.TryWriteUInt64(output, 0))
            return MemoryFault;
        lock (_memoryGate)
        {
            if (!_directAllocations.TryAllocate(searchStart, Math.Min(searchEnd, _directMemoryBudget), length,
                    Math.Max(alignment, PoolPageSize), 0, out var physical))
                return MemoryNoSpace;
            if (!ctx.TryWriteUInt64(output, physical))
            {
                _directAllocations.ReleaseRange(physical, length);
                return MemoryFault;
            }
            AddPoolSupplyLocked(physical, length);
            return 0;
        }
    }

    [SysAbiExport(
        Nid = "Vzl66WmfLvk",
        ExportName = "sceKernelMemoryPoolCommit",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelMemoryPoolCommit(CpuContext ctx)
    {
        var address = ctx[CpuRegister.Rdi];
        var length = ctx[CpuRegister.Rsi];
        var type = unchecked((int)ctx[CpuRegister.Rdx]);
        var protection = unchecked((int)ctx[CpuRegister.Rcx]);
        return RunMappingTransaction(() => PoolCommitCore(ctx, address, length, type, protection));
    }

    private static int PoolCommitCore(CpuContext ctx, ulong address, ulong length, int type, int protection)
    {
        address = GuestVaHole.TranslateRange(address, length);
        if (!IsPoolRange(address, length) || !IsValidMemoryType(type) || (protection & OrbisProtCpuExec) != 0 ||
            !TryDecodeMappedProtection(protection, out var mode))
            return MemoryInvalidArgument;
        lock (_memoryGate)
        {
            var slices = GetMappingSlices(address, length);
            if (!MappingsCoverRange(slices, address, length) ||
                slices.Any(region => !region.IsPooled || !region.IsReserved))
                return MemoryAccessDenied;
            if (length > _poolFreeBytes)
                return MemoryNoSpace;
            var space = ResolveBackingSpace(ctx);
            if (space is null ||
                !TrySelectReplacingBackingAddress(space, address, length, PoolPageSize, OrbisKernelMapFixed,
                    out _, out var replaced))
                return MemoryNoSpace;

            var runs = PlanPoolPagesLocked(length);
            var mapped = new List<MappedRegion>();
            var cursor = address;
            foreach (var (physical, runLength) in runs)
            {
                var region = new MappedRegion(cursor, runLength, protection, false, false, 0, physical,
                    MemoryType: type, Traits: MappingTraits.Pooled);
                if (!space.TryClearBacking(physical, runLength) ||
                    !space.TryMapBacked(cursor, runLength, physical, mode, out _))
                {
                    foreach (var done in mapped)
                        _ = space.TryUnmapBacked(done.Address, done.Length);
                    RestoreReplacedLocked(space, address, length, replaced);
                    return MemoryNoSpace;
                }
                mapped.Add(region);
                cursor += runLength;
            }

            CommitReplacedLocked(replaced);
            foreach (var region in mapped)
            {
                _ = TryTakePoolPagesLocked(region.BackingOffset, region.Length);
                ReplaceMappedRegionRangeLocked(region);
            }
            GuestGpuMemoryHook.NoteMapped(address, length, mode);
            return 0;
        }
    }

    [SysAbiExport(
        Nid = "LXo1tpFqJGs",
        ExportName = "sceKernelMemoryPoolDecommit",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelMemoryPoolDecommit(CpuContext ctx)
    {
        var address = ctx[CpuRegister.Rdi];
        var length = ctx[CpuRegister.Rsi];
        return RunMappingTransaction(() => PoolDecommitCore(ctx, address, length));
    }

    private static int PoolDecommitCore(CpuContext ctx, ulong address, ulong length)
    {
        address = GuestVaHole.TranslateRange(address, length);
        if (!IsPoolRange(address, length))
            return MemoryInvalidArgument;
        lock (_memoryGate)
        {
            var slices = GetMappingSlices(address, length);
            if (!MappingsCoverRange(slices, address, length) || slices.Any(region => !region.IsPooled))
                return MemoryAccessDenied;
            var committed = slices.Where(region => !region.IsReserved).ToArray();
            if (committed.Length != 0)
            {
                var space = ResolveBackingSpace(ctx);
                if (space is null || !TryUnmapViews(space, committed))
                    return MemoryAccessDenied;
                foreach (var region in committed)
                    ReleaseBackingOwnershipLocked(region);
            }
            ReplaceMappedRegionRangeLocked(new MappedRegion(address, length, 0, false, false, 0, IsReserved: true,
                Traits: MappingTraits.Pooled));
            return 0;
        }
    }

    [SysAbiExport(
        Nid = "e1GIcX4AlY4",
        ExportName = "sceKernelMemoryPoolMove",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelMemoryPoolMove(CpuContext ctx)
    {
        var destination = ctx[CpuRegister.Rdi];
        var source = ctx[CpuRegister.Rsi];
        var length = ctx[CpuRegister.Rdx];
        return RunMappingTransaction(() => PoolMoveCore(ctx, destination, source, length));
    }

    // Moves committed pages to a reserved part of a pool: the physical pages (and so the
    // contents) follow, the source becomes reserved again.
    private static int PoolMoveCore(CpuContext ctx, ulong destination, ulong source, ulong length)
    {
        destination = GuestVaHole.TranslateRange(destination, length);
        source = GuestVaHole.TranslateRange(source, length);
        if (!IsPoolRange(destination, length) || !IsPoolRange(source, length) ||
            (destination < source + length && source < destination + length))
            return MemoryInvalidArgument;
        lock (_memoryGate)
        {
            var sourceSlices = GetMappingSlices(source, length);
            var destinationSlices = GetMappingSlices(destination, length);
            if (!MappingsCoverRange(sourceSlices, source, length) ||
                sourceSlices.Any(region => !region.IsPooled || region.IsReserved) ||
                !MappingsCoverRange(destinationSlices, destination, length) ||
                destinationSlices.Any(region => !region.IsPooled || !region.IsReserved))
                return MemoryAccessDenied;
            var space = ResolveBackingSpace(ctx);
            if (space is null ||
                !TrySelectReplacingBackingAddress(space, destination, length, PoolPageSize, OrbisKernelMapFixed,
                    out _, out var replaced))
                return MemoryNoSpace;
            if (!TryUnmapViews(space, sourceSlices))
            {
                RestoreReplacedLocked(space, destination, length, replaced);
                return MemoryAccessDenied;
            }

            var moved = sourceSlices.Select(region => region with { Address = destination + region.Address - source })
                .ToArray();
            for (var index = 0; index < moved.Length; index++)
            {
                var region = moved[index];
                if (TryDecodeMappedProtection(region.Protection, out var mode) &&
                    space.TryMapBacked(region.Address, region.Length, region.BackingOffset, mode, out _))
                    continue;
                for (var undo = index - 1; undo >= 0; undo--)
                    _ = space.TryUnmapBacked(moved[undo].Address, moved[undo].Length);
                RestoreReplacedLocked(space, destination, length, replaced);
                RestoreViews(space, sourceSlices);
                RestoreGpuMappings(sourceSlices);
                return MemoryNoSpace;
            }

            CommitReplacedLocked(replaced);
            ReplaceMappedRegionRangeLocked(new MappedRegion(source, length, 0, false, false, 0, IsReserved: true,
                Traits: MappingTraits.Pooled));
            foreach (var region in moved)
                ReplaceMappedRegionRangeLocked(region);
            RestoreGpuMappings(moved);
            return 0;
        }
    }

    [SysAbiExport(
        Nid = "YN878uKRBbE",
        ExportName = "sceKernelMemoryPoolBatch",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelMemoryPoolBatch(CpuContext ctx)
    {
        var entries = ctx[CpuRegister.Rdi];
        var count = unchecked((int)ctx[CpuRegister.Rsi]);
        var processedAddress = ctx[CpuRegister.Rdx];
        if (count < 0 || (entries == 0 && count != 0))
            return MemoryInvalidArgument;

        // Entries run in order and the batch stops at the first one that fails.
        var processed = 0;
        var result = 0;
        var raw = new byte[PoolBatchEntrySize];
        for (; processed < count; processed++)
        {
            var entry = entries + (ulong)processed * PoolBatchEntrySize;
            if (!TryReadCompat(ctx, entry, raw))
            {
                result = MemoryFault;
                break;
            }
            var op = BitConverter.ToInt32(raw, 0);
            var first = BitConverter.ToUInt64(raw, 8);
            var second = BitConverter.ToUInt64(raw, 16);
            var third = BitConverter.ToUInt64(raw, 24);
            var protection = raw[24];
            var type = raw[25];
            result = op switch
            {
                PoolOpCommit => RunMappingTransaction(() => PoolCommitCore(ctx, first, second, type, protection)),
                PoolOpDecommit => RunMappingTransaction(() => PoolDecommitCore(ctx, first, second)),
                PoolOpProtect => ProtectPoolRange(ctx, first, second, protection, null),
                PoolOpTypeProtect => ProtectPoolRange(ctx, first, second, protection, type),
                PoolOpMove => RunMappingTransaction(() => PoolMoveCore(ctx, first, second, third)),
                _ => MemoryInvalidArgument,
            };
            if (result != 0)
                break;
        }

        if (processedAddress != 0 && !TryWriteUInt32Compat(ctx, processedAddress, (uint)processed))
            return MemoryFault;
        return result;
    }

    private static int ProtectPoolRange(CpuContext ctx, ulong address, ulong length, int protection, int? type)
    {
        address = GuestVaHole.TranslateRange(address, length);
        if (!IsPoolRange(address, length) || (type is { } value && !IsValidMemoryType(value)))
            return MemoryInvalidArgument;
        lock (_memoryGate)
        {
            var slices = GetMappingSlices(address, length);
            if (!MappingsCoverRange(slices, address, length) ||
                slices.Any(region => !region.IsPooled || region.IsReserved))
                return MemoryAccessDenied;
        }
        return ProtectMappedRange(ctx, address, length, protection, type);
    }

    [SysAbiExport(
        Nid = "bvD+95Q6asU",
        ExportName = "sceKernelMemoryPoolGetBlockStats",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelMemoryPoolGetBlockStats(CpuContext ctx)
    {
        var output = ctx[CpuRegister.Rdi];
        var size = unchecked((int)ctx[CpuRegister.Rsi]);
        if (size < 0)
            return MemoryInvalidArgument;
        if (size == 0)
            return 0;
        if (output == 0)
            return MemoryFault;
        // Four int32 counts of 64 KiB blocks: available and allocated, each flushed and cached.
        // Nothing here tracks cache state, so every block counts as flushed.
        Span<byte> stats = stackalloc byte[16];
        lock (_memoryGate)
        {
            BitConverter.TryWriteBytes(stats[0..4], (int)(_poolFreeBytes / PoolPageSize));
            BitConverter.TryWriteBytes(stats[8..12], (int)(_poolCommittedBytes / PoolPageSize));
        }
        return TryWriteCompat(ctx, output, stats[..Math.Min(size, 16)]) ? 0 : MemoryFault;
    }
}
