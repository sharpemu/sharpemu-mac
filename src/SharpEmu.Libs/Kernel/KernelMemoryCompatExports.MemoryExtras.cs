// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;

namespace SharpEmu.Libs.Kernel;

// mmap and the smaller memory queries: direct memory type, flexible release, page table
// statistics.
public static partial class KernelMemoryCompatExports
{
    private const ulong PosixMapShared = 0x1;
    private const ulong PosixMapPrivate = 0x2;
    private const ulong PosixMapAnonymous = 0x1000;
    private const ulong PosixMapExclusive = 0x4000;
    private const ulong PageTableGranule = 2UL * 1024 * 1024;
    private const int MmapCopyChunk = 1024 * 1024;

    [SysAbiExport(
        Nid = "PGhQHd-dzv8",
        ExportName = "sceKernelMmap",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelMmap(CpuContext ctx)
    {
        // The result pointer is the 7th argument, the first stack slot above the return address.
        if (!ctx.TryGetImportStackArgument(0, out var resultAddress) &&
            !ctx.TryReadUInt64(ctx[CpuRegister.Rsp] + sizeof(ulong), out resultAddress))
            return MemoryFault;
        if (resultAddress == 0)
            return MemoryInvalidArgument;
        // Probe the result slot first, so a fault leaves no mapping behind.
        if (!ctx.TryWriteUInt64(resultAddress, 0))
            return MemoryFault;
        var result = MmapCore(ctx, out var address);
        if (result != 0)
            return result;
        return ctx.TryWriteUInt64(resultAddress, address) ? 0 : MemoryFault;
    }

    [SysAbiExport(
        Nid = "BPE9s9vQQXo",
        ExportName = "mmap",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixMmap(CpuContext ctx)
    {
        var result = MmapCore(ctx, out var address);
        if (result == 0)
        {
            ctx[CpuRegister.Rax] = address;
            return 0;
        }
        KernelRuntimeCompatExports.TrySetErrno(ctx, result & 0xFFFF);
        ctx[CpuRegister.Rax] = ulong.MaxValue;
        return -1;
    }

    // mmap(addr, len, prot, flags, fd, offset). Anonymous memory and file contents both land
    // in flexible memory; a file mapping is a copy, so MAP_SHARED stores never reach the file.
    private static int MmapCore(CpuContext ctx, out ulong address)
    {
        address = 0;
        var requested = ctx[CpuRegister.Rdi];
        var length = ctx[CpuRegister.Rsi];
        var protection = unchecked((int)ctx[CpuRegister.Rdx]);
        var flags = ctx[CpuRegister.Rcx] & 0xFFFF_FFFFUL;
        // Only a fixed request is held to its address; a hint inside the window is searched around.
        if ((flags & OrbisKernelMapFixed) != 0 && length != 0 && length <= ulong.MaxValue - OrbisPageSize)
            requested = GuestVaHole.RelocateFixedRequest(requested, AlignUp(length, OrbisPageSize));
        var fd = unchecked((int)ctx[CpuRegister.R8]);
        var offset = ctx[CpuRegister.R9];
        var anonymous = (flags & PosixMapAnonymous) != 0;
        var fixedAddress = (flags & OrbisKernelMapFixed) != 0;
        if (length == 0 || length > ulong.MaxValue - OrbisPageSize ||
            (flags & (PosixMapShared | PosixMapPrivate)) == (PosixMapShared | PosixMapPrivate) ||
            ((flags & PosixMapExclusive) != 0 && !fixedAddress) ||
            !TryDecodeMappedProtection(protection, out var mode))
            return MemoryInvalidArgument;
        if (anonymous ? fd != -1 || offset != 0
                : (flags & (PosixMapShared | PosixMapPrivate)) == 0 || !IsAligned(offset, OrbisPageSize))
            return MemoryInvalidArgument;
        length = AlignUp(length, OrbisPageSize);
        if (fixedAddress && (!IsAligned(requested, OrbisPageSize) || requested == 0 ||
            requested > ulong.MaxValue - length))
            return MemoryInvalidArgument;
        if (!fixedAddress)
            requested &= ~(OrbisPageSize - 1);
        // MAP_EXCL refuses to replace a live mapping, which is NO_OVERWRITE here.
        var mapFlags = (flags & (OrbisKernelMapFixed | OrbisKernelMapNoOverwrite | OrbisKernelMapNoCoalesce | 0xFF000000UL)) |
            ((flags & PosixMapExclusive) != 0 ? OrbisKernelMapNoOverwrite : 0);
        if (!TryResolveMapAlignment(mapFlags, MapFlexibleAllowedFlags, 0, out var alignment))
            return MemoryInvalidArgument;

        Stream? stream = null;
        if (!anonymous)
        {
            lock (_fdGate)
                _openFiles.TryGetValue(fd, out stream);
            if (stream is null)
                return MemoryBadDescriptor;
            if (!stream.CanRead || !stream.CanSeek ||
                ((flags & PosixMapShared) != 0 && (protection & OrbisProtCpuWrite) != 0 && !stream.CanWrite))
                return MemoryAccessDenied;
        }

        var mappedOut = 0UL;
        var outcome = RunMappingTransaction(() =>
        {
            lock (_memoryGate)
            {
                var mapped = MapFlexibleLocked(ctx, 0, requested, length, protection, mode, mapFlags, alignment, null,
                    out var mappedAddress);
                if (mapped != 0)
                    return mapped;
                if (stream is not null && !TryCopyFileIntoMappingLocked(ctx, stream, offset, mappedAddress, length))
                {
                    _ = UnmapLocked(ctx, mappedAddress, length);
                    return unchecked((int)0x80020005);
                }
                mappedOut = mappedAddress;
                return 0;
            }
        }, () => !fixedAddress || IsUntouchedByGpu(requested, length));
        address = outcome == 0 ? mappedOut : 0;
        return outcome;
    }

    private static bool TryCopyFileIntoMappingLocked(CpuContext ctx, Stream stream, ulong offset, ulong address,
        ulong length)
    {
        var space = ResolveBackingSpace(ctx);
        if (space is null)
            return false;
        var buffer = new byte[MmapCopyChunk];
        lock (stream)
        {
            var position = stream.Position;
            try
            {
                if (offset >= (ulong)stream.Length)
                    return true;
                stream.Position = (long)offset;
                var copied = 0UL;
                while (copied < length)
                {
                    var read = stream.Read(buffer, 0, (int)Math.Min((ulong)buffer.Length, length - copied));
                    if (read <= 0)
                        break;
                    if (!space.TryWriteBacking(address + copied, buffer.AsSpan(0, read)))
                        return false;
                    copied += (ulong)read;
                }
                return true;
            }
            catch (IOException)
            {
                return false;
            }
            finally
            {
                stream.Position = position;
            }
        }
    }

    // Undoes a mapping made under the same lock.
    private static int UnmapLocked(CpuContext ctx, ulong address, ulong length)
    {
        var regions = GetMappingSlices(address, length);
        var space = ResolveBackingSpace(ctx);
        if (space is null || !TryUnmapViews(space, regions.Where(region => !region.IsReserved).ToArray()))
            return MemoryAccessDenied;
        foreach (var region in regions)
            ReleaseBackingOwnershipLocked(region);
        RemoveMappingLocked(address, length);
        return 0;
    }

    [SysAbiExport(
        Nid = "teiItL2boFw",
        ExportName = "sceKernelReleaseFlexibleMemory",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelReleaseFlexibleMemory(CpuContext ctx) => KernelMunmap(ctx);

    // The v2 call takes the v1 arguments (search start/end, length, alignment, type, *out);
    // no header in reach shows it differs.
    [SysAbiExport(
        Nid = "YikqzdTOmxY",
        ExportName = "sceKernelAllocateDirectMemory2",
        Target = Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelAllocateDirectMemory2(CpuContext ctx) => KernelAllocateDirectMemory(ctx);

    [SysAbiExport(
        Nid = "BC+OG5m9+bw",
        ExportName = "sceKernelGetDirectMemoryType",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelGetDirectMemoryType(CpuContext ctx)
    {
        var offset = ctx[CpuRegister.Rdi];
        var typeAddress = ctx[CpuRegister.Rsi];
        var startAddress = ctx[CpuRegister.Rdx];
        var endAddress = ctx[CpuRegister.Rcx];
        if (typeAddress == 0 || startAddress == 0 || endAddress == 0)
            return MemoryInvalidArgument;
        DirectMemoryAllocationMap.Allocation allocation;
        lock (_memoryGate)
        {
            if (offset >= _directMemoryBudget || !_directAllocations.TryFindAllocation(offset, false, out allocation))
                return MemoryNotFound;
        }
        return TryWriteInt32(ctx, typeAddress, allocation.MemoryType) &&
            ctx.TryWriteUInt64(startAddress, allocation.Start) &&
            ctx.TryWriteUInt64(endAddress, allocation.Start + allocation.Length)
                ? 0
                : MemoryFault;
    }

    // Counts in 2 MiB page-table granules: the totals cover the title's whole budget, so a
    // title never sees the tables run out before its memory does.
    [SysAbiExport(
        Nid = "tZ2yplY8MBY",
        ExportName = "sceKernelGetPageTableStats",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelGetPageTableStats(CpuContext ctx)
    {
        var outputs = new[] { ctx[CpuRegister.Rdi], ctx[CpuRegister.Rsi], ctx[CpuRegister.Rdx], ctx[CpuRegister.Rcx] };
        if (outputs.Any(address => address == 0))
            return MemoryFault;
        ulong total, cpuUsed = 0, gpuUsed = 0;
        lock (_memoryGate)
        {
            total = (_directMemoryBudget + _flexibleMemoryBudget + PageTableGranule - 1) / PageTableGranule;
            foreach (var region in _mappedRegions.Values)
            {
                if (region.IsReserved)
                    continue;
                var granules = (region.Length + PageTableGranule - 1) / PageTableGranule;
                if ((region.Protection & (OrbisProtCpuRead | OrbisProtCpuWrite | OrbisProtCpuExec)) != 0)
                    cpuUsed += granules;
                if ((region.Protection & (OrbisProtGpuRead | OrbisProtGpuWrite)) != 0)
                    gpuUsed += granules;
            }
        }
        var cpuTotal = (int)Math.Min(total, int.MaxValue);
        var values = new[]
        {
            cpuTotal,
            (int)Math.Max(0, (long)cpuTotal - (long)Math.Min(cpuUsed, int.MaxValue)),
            cpuTotal,
            (int)Math.Max(0, (long)cpuTotal - (long)Math.Min(gpuUsed, int.MaxValue)),
        };
        for (var index = 0; index < outputs.Length; index++)
            if (!TryWriteInt32(ctx, outputs[index], values[index]))
                return MemoryFault;
        return 0;
    }
}
