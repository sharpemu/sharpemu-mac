// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Text;
using SharpEmu.HLE;

namespace SharpEmu.Libs.Kernel;

// The guest VA map: every range the kernel hands out lives in _mappedRegions, one record per
// run of identical state. A record carries its kind (reserved, direct, flexible, pooled, JIT),
// protection, memory type and name, so splitting a record (protect, rename, partial unmap)
// carries all of them along and removing it drops them. Queries coalesce neighbouring records
// the way the kernel merges map entries. Memory the loader and the runtime map on the guest's
// behalf (module segments, thread stacks, TLS, runtime stubs) is registered here too, so the
// guest can query it like the kernel's own mappings.
public static partial class KernelMemoryCompatExports
{
    private const int MemoryNotFound = unchecked((int)0x80020002);
    private const int MemoryBadDescriptor = unchecked((int)0x80020009);
    private const int MemoryBusy = unchecked((int)0x80020010);
    private const int MemoryTryAgain = unchecked((int)0x80020023);
    private const int MemoryNotSupported = unchecked((int)0x8002002D);
    private const int MemoryNameTooLong = unchecked((int)0x8002003F);
    private const ulong OrbisKernelMapNoOverwrite = 0x80;
    private const ulong OrbisKernelMapNoCoalesce = 0x400000;
    // sys/dmem.h: memory types run up to SCE_KERNEL_MTYPE_END (exclusive).
    private const int OrbisMemoryTypeEnd = 21;
    private const byte VirtualQueryFlexible = 0x01;
    private const byte VirtualQueryDirect = 0x02;
    private const byte VirtualQueryStack = 0x04;
    private const byte VirtualQueryPooled = 0x08;
    private const byte VirtualQueryCommitted = 0x10;
    private const byte VirtualQueryGpuPrt = 0x20;
    private const int VirtualQueryNameOffset = 33;

    [Flags]
    private enum MappingTraits : byte
    {
        None = 0,
        // A memory-pool reservation (with IsReserved) or a committed pool page.
        Pooled = 0x01,
        // SCE_KERNEL_MAP_NO_COALESCE: the range never merges with its neighbours.
        NoCoalesce = 0x02,
        // A view of a JIT shared-memory object; the object owns the backing.
        Jit = 0x04,
        // Mapped by the loader or the runtime, not through a kernel memory call.
        Host = 0x08,
        // A thread stack (with Host).
        Stack = 0x10,
    }

    private readonly record struct MappedRegion(ulong Address, ulong Length, int Protection, bool IsFlexible,
        bool IsDirect, ulong DirectStart, ulong BackingOffset = 0, bool IsReserved = false,
        int MemoryType = 0, string? Name = null, MappingTraits Traits = MappingTraits.None)
    {
        public bool IsPooled => (Traits & MappingTraits.Pooled) != 0;
        public bool IsJit => (Traits & MappingTraits.Jit) != 0;
        public bool IsNoCoalesce => (Traits & MappingTraits.NoCoalesce) != 0;
        public bool IsHost => (Traits & MappingTraits.Host) != 0;
        public bool IsStack => (Traits & MappingTraits.Stack) != 0;

        // Committed to a view of the shared backing object.
        public bool IsBacked => !IsReserved && (IsDirect || IsFlexible || IsPooled || IsJit);
    }

    private readonly record struct VirtualRange(ulong Start, ulong End, MappedRegion Region, bool IsStack);

    // FreeBSD merges neighbouring map entries with one state; direct, JIT and loader mappings
    // keep the entry of their own map call, and NO_COALESCE opts a range out.
    private static bool CanCoalesce(MappedRegion left, MappedRegion right) =>
        TryAddU64(left.Address, left.Length, out var leftEnd) && leftEnd == right.Address &&
        !left.IsDirect && !right.IsDirect && !left.IsJit && !right.IsJit && !left.IsHost && !right.IsHost &&
        !left.IsNoCoalesce && !right.IsNoCoalesce &&
        left.IsReserved == right.IsReserved && left.IsFlexible == right.IsFlexible &&
        left.Traits == right.Traits && left.Protection == right.Protection &&
        left.MemoryType == right.MemoryType && string.Equals(left.Name, right.Name, StringComparison.Ordinal);

    private static bool TryResolveVirtualRangeLocked(ulong address, bool findNext, out VirtualRange range)
    {
        range = default;
        if (!TryFindVirtualQueryRegionLocked(address, findNext, out var region) ||
            !TryAddU64(region.Address, region.Length, out var end))
            return false;

        var start = region.Address;
        var current = region;
        while (current.Address != 0 && _mappedRegions.TryFindAtOrBelow(current.Address - 1, out var previous) &&
               CanCoalesce(previous, current))
        {
            start = previous.Address;
            current = previous;
        }

        current = region;
        while (_mappedRegions.TryGetValue(end, out var next) && CanCoalesce(current, next) &&
               TryAddU64(next.Address, next.Length, out var nextEnd))
        {
            end = nextEnd;
            current = next;
        }

        range = new VirtualRange(start, end, region with { Address = start, Length = end - start }, region.IsStack);
        return true;
    }

    private static bool TryResolveVirtualRange(ulong address, bool findNext, out VirtualRange range)
    {
        lock (_memoryGate)
            return TryResolveVirtualRangeLocked(address, findNext, out range);
    }

    // What the loader or the runtime maps for the guest: the role decides how a query reports it.
    public enum HostMappingKind
    {
        // A module segment, named after its module.
        Module,
        // A guest thread stack: VirtualQuery and IsStack report it as a stack.
        Stack,
        // A thread's TLS block.
        ThreadLocalStorage,
        // Runtime-owned guest memory: stubs, callback and exception stacks, HLE data.
        Runtime,
    }

    // Records a range the host mapped itself. The range is widened to whole 16 KiB pages and
    // replaces whatever the table held there; the host memory itself is not touched.
    public static void RegisterHostMapping(ulong address, ulong length, int protection, HostMappingKind kind,
        string? name = null)
    {
        if (length == 0 || address > ulong.MaxValue - length)
            return;
        var start = address & ~(OrbisPageSize - 1);
        var end = AlignUp(address + length, OrbisPageSize);
        if (end <= start)
            return;
        if (name is { Length: >= OrbisKernelMaximumNameLength })
            name = name[..(OrbisKernelMaximumNameLength - 1)];
        var traits = MappingTraits.Host | (kind == HostMappingKind.Stack ? MappingTraits.Stack : MappingTraits.None);
        lock (_memoryGate)
            ReplaceMappedRegionRangeLocked(new MappedRegion(start, end - start, protection, false, false, 0,
                Name: name, Traits: traits));
    }

    // Drops the host records over a range the host released; kernel mappings there stay.
    public static void UnregisterHostMapping(ulong address, ulong length)
    {
        if (length == 0 || address > ulong.MaxValue - length)
            return;
        var start = address & ~(OrbisPageSize - 1);
        var end = AlignUp(address + length, OrbisPageSize);
        lock (_memoryGate)
        {
            foreach (var region in GetMappingSlices(start, end - start))
                if (region.IsHost)
                    RemoveMappingLocked(region.Address, region.Length);
        }
    }

    private static int WriteVirtualQueryInfo(CpuContext ctx, ulong infoAddress, VirtualRange range)
    {
        var region = range.Region;
        Span<byte> payload = stackalloc byte[OrbisVirtualQueryInfoSize];
        payload.Clear();
        byte state = 0;
        if (region.IsFlexible)
            state |= VirtualQueryFlexible;
        if (region.IsDirect)
            state |= VirtualQueryDirect;
        if (range.IsStack)
            state |= VirtualQueryStack;
        if (region.IsPooled)
            state |= VirtualQueryPooled;
        if (!region.IsReserved)
            state |= VirtualQueryCommitted;
        if (!range.IsStack && KernelRuntimeCompatExports.OverlapsPrtAperture(range.Start, range.End - range.Start))
            state |= VirtualQueryGpuPrt;

        var offset = region.IsDirect ? region.DirectStart
            : region.IsPooled && !region.IsReserved ? region.BackingOffset : 0;
        BinaryPrimitives.WriteUInt64LittleEndian(payload[0..8], range.Start);
        BinaryPrimitives.WriteUInt64LittleEndian(payload[8..16], range.End);
        BinaryPrimitives.WriteUInt64LittleEndian(payload[16..24], offset);
        BinaryPrimitives.WriteInt32LittleEndian(payload[24..28], region.Protection);
        BinaryPrimitives.WriteInt32LittleEndian(payload[28..32], region.MemoryType);
        payload[32] = state;
        var name = region.Name ?? (range.IsStack ? "stack" : region.IsDirect ? "direct" : region.IsFlexible ? "flexible" : null);
        if (!string.IsNullOrEmpty(name))
        {
            var nameBytes = Encoding.UTF8.GetBytes(name);
            nameBytes.AsSpan(0, Math.Min(nameBytes.Length, OrbisKernelMaximumNameLength - 1))
                .CopyTo(payload.Slice(VirtualQueryNameOffset, OrbisKernelMaximumNameLength));
        }
        // ammUsage and gpuMaskId stay zero: no AMM or GPU-mask state is modelled.
        return TryWriteCompat(ctx, infoAddress, payload) ? 0 : MemoryFault;
    }

    [SysAbiExport(
        Nid = "rVjRvHJ0X6c",
        ExportName = "sceKernelVirtualQuery",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelVirtualQuery(CpuContext ctx)
    {
        var queryAddress = GuestVaHole.Translate(ctx[CpuRegister.Rdi]);
        var flags = ctx[CpuRegister.Rsi];
        var infoAddress = ctx[CpuRegister.Rdx];
        var infoSize = ctx[CpuRegister.Rcx];
        if (infoAddress == 0 || infoSize < OrbisVirtualQueryInfoSize || (flags & ~1UL) != 0)
            return MemoryInvalidArgument;
        if (!TryResolveVirtualRange(queryAddress, findNext: (flags & 1) != 0, out var range))
            return MemoryAccessDenied;
        var result = WriteVirtualQueryInfo(ctx, infoAddress, range);
        if (result == 0)
            ctx[CpuRegister.Rax] = 0;
        return result;
    }

    [SysAbiExport(
        Nid = "WFcfL2lzido",
        ExportName = "sceKernelQueryMemoryProtection",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelQueryMemoryProtection(CpuContext ctx)
    {
        var queryAddress = GuestVaHole.Translate(ctx[CpuRegister.Rdi]);
        var startOut = ctx[CpuRegister.Rsi];
        var endOut = ctx[CpuRegister.Rdx];
        var protectionOut = ctx[CpuRegister.Rcx];
        if (!TryResolveVirtualRange(queryAddress, findNext: false, out var range))
            return MemoryAccessDenied;

        // The end is exclusive, as the map entry reports it.
        if ((startOut != 0 && !ctx.TryWriteUInt64(startOut, range.Start)) ||
            (endOut != 0 && !ctx.TryWriteUInt64(endOut, range.End)) ||
            (protectionOut != 0 && !TryWriteInt32(ctx, protectionOut, range.Region.Protection)))
            return MemoryFault;
        return 0;
    }

    [SysAbiExport(
        Nid = "yDBwVAolDgg",
        ExportName = "sceKernelIsStack",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelIsStack(CpuContext ctx)
    {
        var address = GuestVaHole.Translate(ctx[CpuRegister.Rdi]);
        var startOut = ctx[CpuRegister.Rsi];
        var endOut = ctx[CpuRegister.Rdx];

        // A mapped non-stack address still succeeds with both bounds cleared: libc's VM allocator
        // reads them back, and stale values become invalid fixed-range reservations.
        ulong start = 0, end = 0;
        if (TryResolveVirtualRange(address, findNext: false, out var range))
        {
            if (range.IsStack)
                (start, end) = (range.Start, range.End);
        }
        // The host heap behind libc's malloc is mapped without a table record.
        else if (!IsHostRangeAccessible(address, 1, writeAccess: false))
            return MemoryAccessDenied;

        if ((startOut != 0 && !ctx.TryWriteUInt64(startOut, start)) ||
            (endOut != 0 && !ctx.TryWriteUInt64(endOut, end)))
            return MemoryFault;
        ctx[CpuRegister.Rax] = 0;
        return 0;
    }

    [SysAbiExport(
        Nid = "DGMG3JshrZU",
        ExportName = "sceKernelSetVirtualRangeName",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelSetVirtualRangeName(CpuContext ctx)
    {
        var length = ctx[CpuRegister.Rsi];
        var address = GuestVaHole.TranslateRange(ctx[CpuRegister.Rdi], length);
        var nameAddress = ctx[CpuRegister.Rdx];
        if (address == 0 || length == 0 || address > ulong.MaxValue - length)
            return MemoryInvalidArgument;
        var result = TryReadRangeName(ctx, nameAddress, out var name);
        if (result != 0)
            return result;
        RenameMappedRange(address, length, name);
        ctx[CpuRegister.Rax] = 0;
        return 0;
    }

    [SysAbiExport(
        Nid = "mkgXxsoxWHg",
        ExportName = "sceKernelClearVirtualRangeName",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelClearVirtualRangeName(CpuContext ctx)
    {
        var length = ctx[CpuRegister.Rsi];
        var address = GuestVaHole.TranslateRange(ctx[CpuRegister.Rdi], length);
        if (address == 0 || length == 0 || address > ulong.MaxValue - length)
            return MemoryInvalidArgument;
        RenameMappedRange(address, length, null);
        ctx[CpuRegister.Rax] = 0;
        return 0;
    }

    // A name is at most 31 bytes plus its terminator.
    private static int TryReadRangeName(CpuContext ctx, ulong nameAddress, out string? name)
    {
        name = null;
        if (nameAddress == 0)
            return MemoryFault;
        Span<byte> bytes = stackalloc byte[OrbisKernelMaximumNameLength];
        for (var index = 0; index < bytes.Length; index++)
        {
            if (!TryReadCompat(ctx, nameAddress + (ulong)index, bytes.Slice(index, 1)))
                return MemoryFault;
            if (bytes[index] == 0)
            {
                name = Encoding.UTF8.GetString(bytes[..index]);
                return 0;
            }
        }
        return MemoryNameTooLong;
    }

    // Names follow their pages: a sub-range rename splits the record, unmapped pages are skipped.
    private static void RenameMappedRange(ulong address, ulong length, string? name)
    {
        var start = AlignDown(address, OrbisPageSize);
        var end = address + length;
        end = end > ulong.MaxValue - (OrbisPageSize - 1) ? ulong.MaxValue & ~(OrbisPageSize - 1) : AlignUp(end, OrbisPageSize);
        if (end <= start)
            return;
        lock (_memoryGate)
            UpdateMappedRangeLocked(start, end - start, region => region with { Name = string.IsNullOrEmpty(name) ? null : name });
    }

    // Rewrites every record slice inside [address, address + size), splitting at the edges.
    private static void UpdateMappedRangeLocked(ulong address, ulong size, Func<MappedRegion, MappedRegion> update)
    {
        foreach (var slice in GetMappingSlices(address, size))
            ReplaceMappedRegionRangeLocked(update(slice));
    }

    private static bool IsValidMemoryType(int memoryType) => memoryType is >= 0 and < OrbisMemoryTypeEnd;
}
