// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text;
using SharpEmu.HLE;
using SharpEmu.Libs.Kernel;
using Xunit;

namespace SharpEmu.Libs.Tests.Kernel;

// The memory exports added for W3: mmap, memory pools, JIT shared memory and the smaller
// queries. The test backing is 128 MiB: direct memory sits below 64 MiB and the flexible
// pool above it.
[Collection(KernelMemoryCompatStateCollection.Name)]
public sealed class KernelMemoryExportTests
{
    private const int Einval = unchecked((int)0x80020016);
    private const int Eacces = unchecked((int)0x8002000D);
    private const int Enomem = unchecked((int)0x8002000C);
    private const int Efault = unchecked((int)0x8002000E);
    private const int Ebadf = unchecked((int)0x80020009);
    private const int Enoent = unchecked((int)0x80020002);
    private const ulong PoolChunk = 0x10000;

    private sealed class Fixture : IDisposable
    {
        private readonly FlexibleBackingPool _previous;

        public Fixture()
        {
            Test = new BackedKernelMemory();
            Pool = new FlexibleBackingPool(0x4000000, 0x4000000);
            _previous = KernelMemoryCompatExports.SetFlexibleBackingForTests(Pool);
        }

        public BackedKernelMemory Test { get; }
        public FlexibleBackingPool Pool { get; }
        public CpuContext Context => Test.Context;
        public ulong Output => Test.Output;

        public void Dispose()
        {
            KernelMemoryCompatExports.ResetBackingMappings(Test.Memory);
            KernelMemoryCompatExports.SetFlexibleBackingForTests(_previous);
            Test.Dispose();
        }

        public int Call(Func<CpuContext, int> export, params ulong[] arguments)
        {
            CpuRegister[] registers = [CpuRegister.Rdi, CpuRegister.Rsi, CpuRegister.Rdx, CpuRegister.Rcx, CpuRegister.R8, CpuRegister.R9];
            for (var index = 0; index < arguments.Length; index++)
                Context[registers[index]] = arguments[index];
            return export(Context);
        }

        public ulong Read(ulong address)
        {
            Assert.True(Context.TryReadUInt64(address, out var value));
            return value;
        }

        public uint Read32(ulong address)
        {
            Assert.True(Context.TryReadUInt32(address, out var value));
            return value;
        }
    }

    [Fact]
    public void DirectMemoryTypeReportsTheAllocationHoldingAnOffset()
    {
        using var f = new Fixture();
        Assert.Equal(0, f.Call(KernelMemoryCompatExports.KernelAllocateDirectMemory2, 0, 0x1000000, 0x20000, 0x10000, 3, f.Output));
        var start = f.Read(f.Output);

        Assert.Equal(0, f.Call(KernelMemoryCompatExports.KernelGetDirectMemoryType, start + 0x14000, f.Output + 0x10, f.Output + 0x18, f.Output + 0x20));
        Assert.Equal(3u, f.Read32(f.Output + 0x10));
        Assert.Equal(start, f.Read(f.Output + 0x18));
        Assert.Equal(start + 0x20000, f.Read(f.Output + 0x20));

        Assert.Equal(Enoent, f.Call(KernelMemoryCompatExports.KernelGetDirectMemoryType, start + 0x20000, f.Output + 0x10, f.Output + 0x18, f.Output + 0x20));
        Assert.Equal(Einval, f.Call(KernelMemoryCompatExports.KernelGetDirectMemoryType, start, 0, f.Output + 0x18, f.Output + 0x20));
    }

    [Fact]
    public void PrtApertureReadsBackWhatWasSet()
    {
        using var f = new Fixture();
        try
        {
            Assert.Equal(0, f.Call(KernelRuntimeCompatExports.KernelSetPrtAperture, 1, 0x20_0000_0000, 0x40000));
            Assert.Equal(0, f.Call(KernelRuntimeCompatExports.KernelGetPrtAperture, 1, f.Output + 0x10, f.Output + 0x18));
            Assert.Equal(0x20_0000_0000UL, f.Read(f.Output + 0x10));
            Assert.Equal(0x40000UL, f.Read(f.Output + 0x18));
            Assert.Equal(Einval, f.Call(KernelRuntimeCompatExports.KernelGetPrtAperture, 3, f.Output + 0x10, f.Output + 0x18));
            Assert.Equal(Efault, f.Call(KernelRuntimeCompatExports.KernelGetPrtAperture, 0, 0, f.Output + 0x18));
        }
        finally
        {
            Assert.Equal(0, f.Call(KernelRuntimeCompatExports.KernelSetPrtAperture, 1, 0x20_0000_0000, 0));
        }
    }

    [Fact]
    public void APrtApertureInTheBlockedWindowMovesWithItsTilesButReadsBackAsSet()
    {
        using var f = new Fixture();
        GuestVaHole.RelocatesOverride = true;
        try
        {
            Assert.Equal(0, f.Call(KernelRuntimeCompatExports.KernelSetPrtAperture, 1, 0x20_0000_0000, 0x40000));
            Assert.Equal(0, f.Call(KernelRuntimeCompatExports.KernelGetPrtAperture, 1, f.Output + 0x10, f.Output + 0x18));
            Assert.Equal(0x20_0000_0000UL, f.Read(f.Output + 0x10));
            var relocated = 0x20_0000_0000UL + GuestVaHole.Shift;
            Assert.True(KernelRuntimeCompatExports.ContainsPrtRange(relocated, 0x1000));
            Assert.True(KernelRuntimeCompatExports.ContainsPrtRange(0x20_0000_0000, 0x1000));
            Assert.True(KernelRuntimeCompatExports.OverlapsPrtAperture(relocated + 0x3F000, 0x2000));
            Assert.False(KernelRuntimeCompatExports.ContainsPrtRange(relocated + 0x40000, 0x1000));
        }
        finally
        {
            Assert.Equal(0, f.Call(KernelRuntimeCompatExports.KernelSetPrtAperture, 1, 0x20_0000_0000, 0));
            GuestVaHole.RelocatesOverride = null;
        }
    }

    [Fact]
    public void PageTableStatsCountMappedGranules()
    {
        using var f = new Fixture();
        Assert.Equal(Efault, f.Call(KernelMemoryCompatExports.KernelGetPageTableStats, f.Output, 0, f.Output + 8, f.Output + 12));
        Assert.Equal(0, f.Call(KernelMemoryCompatExports.KernelGetPageTableStats, f.Output, f.Output + 4, f.Output + 8, f.Output + 12));
        var total = f.Read32(f.Output);
        var available = f.Read32(f.Output + 4);
        Assert.True(total > 0 && available <= total);

        f.Test.Flexible(0x200000);
        Assert.Equal(0, f.Call(KernelMemoryCompatExports.KernelGetPageTableStats, f.Output, f.Output + 4, f.Output + 8, f.Output + 12));
        Assert.Equal(available - 1, f.Read32(f.Output + 4));
    }

    [Fact]
    public void ReleaseFlexibleMemoryUnmapsAndReturnsTheBacking()
    {
        using var f = new Fixture();
        var address = f.Test.Flexible(0x8000);
        Assert.Equal(0x8000UL, f.Pool.Used);
        Assert.Equal(0, f.Call(KernelMemoryCompatExports.KernelReleaseFlexibleMemory, address, 0x8000));
        Assert.Equal(0UL, f.Pool.Used);
    }

    [Fact]
    public void AnonymousMmapMapsFlexibleMemoryAndReportsPosixErrors()
    {
        using var f = new Fixture();
        // A partial page rounds up to a whole one.
        Assert.Equal(0, f.Call(KernelMemoryCompatExports.PosixMmap, 0, 0x6000, 3, 0x1002, ulong.MaxValue, 0));
        var address = f.Context[CpuRegister.Rax];
        Assert.NotEqual(0UL, address);
        Assert.Equal(0x8000UL, f.Pool.Used);
        Assert.True(f.Context.TryWriteUInt64(address + 0x7FF8, 0x1122334455667788));
        Assert.Equal(0x1122334455667788UL, f.Read(address + 0x7FF8));
        Assert.Equal(0x01u, f.Test.QueryStateFlags(address) & 0x01);
        Assert.Equal(0, f.Test.Unmap(address, 0x8000));

        // MAP_EXCL without MAP_FIXED, and an anonymous mapping with a descriptor.
        Assert.Equal(-1, f.Call(KernelMemoryCompatExports.PosixMmap, 0, 0x4000, 3, 0x5002, ulong.MaxValue, 0));
        Assert.Equal(ulong.MaxValue, f.Context[CpuRegister.Rax]);
        Assert.Equal(-1, f.Call(KernelMemoryCompatExports.PosixMmap, 0, 0x4000, 3, 0x1002, 3, 0));
        Assert.Equal(-1, f.Call(KernelMemoryCompatExports.PosixMmap, 0, 0, 3, 0x1002, ulong.MaxValue, 0));
        Assert.Equal(0UL, f.Pool.Used);
    }

    [Fact]
    public void SceKernelMmapWritesTheResultThroughTheStackArgument()
    {
        using var f = new Fixture();
        f.Context[CpuRegister.Rsp] = f.Output + 0x800;
        Assert.True(f.Context.TryWriteUInt64(f.Output + 0x808, f.Output + 0x40));
        Assert.Equal(0, f.Call(KernelMemoryCompatExports.KernelMmap, 0, 0x4000, 3, 0x1002, ulong.MaxValue, 0));
        var address = f.Read(f.Output + 0x40);
        Assert.NotEqual(0UL, address);
        Assert.Equal(0, f.Test.Unmap(address, 0x4000));

        Assert.Equal(Ebadf, f.Call(KernelMemoryCompatExports.KernelMmap, 0, 0x4000, 1, 0x2, 0x7FFF_0000, 0));
        Assert.Equal(Einval, f.Call(KernelMemoryCompatExports.KernelMmap, 0, 0x4000, 1, 0x3, 0x7FFF_0000, 0));
        Assert.Equal(Einval, f.Call(KernelMemoryCompatExports.KernelMmap, 0, 0x4000, 1, 0x0, 0x7FFF_0000, 0));
        Assert.Equal(Einval, f.Call(KernelMemoryCompatExports.KernelMmap, 0, 0x4000, 1, 0x2, 0x7FFF_0000, 0x1000));
    }

    [Fact]
    public void FileMmapCopiesTheFileFromTheOffset()
    {
        using var f = new Fixture();
        var identity = Guid.NewGuid().ToString("N");
        var mountPoint = $"/mmap-{identity}";
        var hostRoot = Path.Combine(Path.GetTempPath(), $"sharpemu-mmap-{identity}");
        Directory.CreateDirectory(hostRoot);
        var payload = new byte[0x6000];
        for (var index = 0; index < payload.Length; index++)
            payload[index] = (byte)(index * 7);
        File.WriteAllBytes(Path.Combine(hostRoot, "DATA.BIN"), payload);
        KernelMemoryCompatExports.RegisterGuestPathMount(mountPoint, hostRoot);
        var fd = -1;
        try
        {
            Assert.True(f.Context.Memory.TryWrite(f.Output + 0x200, Encoding.ASCII.GetBytes($"{mountPoint}/DATA.BIN\0")));
            Assert.Equal(0, f.Call(KernelExports.KernelOpen, f.Output + 0x200, 0));
            fd = unchecked((int)f.Context[CpuRegister.Rax]);

            Assert.Equal(0, f.Call(KernelMemoryCompatExports.PosixMmap, 0, 0x4000, 1, 0x2, (ulong)fd, 0x4000));
            var address = f.Context[CpuRegister.Rax];
            Span<byte> mapped = stackalloc byte[0x4000];
            Assert.True(f.Context.Memory.TryRead(address, mapped));
            Assert.True(mapped[..0x2000].SequenceEqual(payload.AsSpan(0x4000)));
            Assert.True(mapped[0x2000..].IndexOfAnyExcept((byte)0) < 0);
            Assert.Equal(0, f.Test.Unmap(address, 0x4000));

            // A read-only descriptor cannot back a writable shared mapping.
            f.Context[CpuRegister.Rsp] = f.Output + 0x800;
            Assert.True(f.Context.TryWriteUInt64(f.Output + 0x808, f.Output + 0x40));
            Assert.Equal(Eacces, f.Call(KernelMemoryCompatExports.KernelMmap, 0, 0x4000, 3, 0x1, (ulong)fd, 0));
        }
        finally
        {
            if (fd >= 0)
                _ = f.Call(KernelMemoryCompatExports.KernelClose, (ulong)fd);
            _ = KernelMemoryCompatExports.UnregisterGuestPathMount(mountPoint);
            Directory.Delete(hostRoot, recursive: true);
        }
    }

    private static ulong PoolReserve(Fixture f, ulong length)
    {
        Assert.Equal(0, f.Call(KernelMemoryCompatExports.KernelMemoryPoolReserve, 0, length, 0, 0, f.Output + 0x30));
        return f.Read(f.Output + 0x30);
    }

    private static (uint Available, uint Allocated) BlockStats(Fixture f)
    {
        Assert.Equal(0, f.Call(KernelMemoryCompatExports.KernelMemoryPoolGetBlockStats, f.Output + 0x60, 16));
        return (f.Read32(f.Output + 0x60), f.Read32(f.Output + 0x68));
    }

    [Fact]
    public void MemoryPoolValidatesItsArguments()
    {
        using var f = new Fixture();
        Assert.Equal(Einval, f.Call(KernelMemoryCompatExports.KernelMemoryPoolReserve, 0, 0x100000, 0, 0, f.Output + 0x30));
        Assert.Equal(Einval, f.Call(KernelMemoryCompatExports.KernelMemoryPoolReserve, 0, 0x200000, 0, 0, 0));
        Assert.Equal(Einval, f.Call(KernelMemoryCompatExports.KernelMemoryPoolReserve, 0, 0x200000, 0x300000, 0, f.Output + 0x30));
        Assert.Equal(Einval, f.Call(KernelMemoryCompatExports.KernelMemoryPoolExpand, 0x100000, 0x100000, PoolChunk, 0, f.Output + 0x38));
        Assert.Equal(Einval, f.Call(KernelMemoryCompatExports.KernelMemoryPoolExpand, 0, 0x1000000, 0x8000, 0, f.Output + 0x38));
        Assert.Equal(Einval, f.Call(KernelMemoryCompatExports.KernelMemoryPoolExpand, 0, 0x1000000, PoolChunk, 0, 0));
        Assert.Equal(Efault, f.Call(KernelMemoryCompatExports.KernelMemoryPoolGetBlockStats, 0, 16));

        var address = PoolReserve(f, 0x200000);
        // Nothing in the pool yet.
        Assert.Equal(Enomem, f.Call(KernelMemoryCompatExports.KernelMemoryPoolCommit, address, PoolChunk, 0, 3, 0));
        Assert.Equal(Einval, f.Call(KernelMemoryCompatExports.KernelMemoryPoolCommit, address, 0x4000, 0, 3, 0));
        Assert.Equal(Einval, f.Call(KernelMemoryCompatExports.KernelMemoryPoolCommit, address, PoolChunk, 0, 7, 0));

        // A range outside any pool reservation.
        var flexible = f.Test.Flexible(PoolChunk);
        Assert.Equal(Eacces, f.Call(KernelMemoryCompatExports.KernelMemoryPoolCommit, flexible, PoolChunk, 0, 3, 0));
        Assert.Equal(Eacces, f.Call(KernelMemoryCompatExports.KernelMemoryPoolDecommit, flexible, PoolChunk, 0));
    }

    [Fact]
    public void MemoryPoolCommitsAndDecommitsPagesFromItsSupply()
    {
        using var f = new Fixture();
        var address = PoolReserve(f, 0x400000);
        Assert.Equal(0x08u, f.Test.QueryStateFlags(address) & 0x18);

        Assert.Equal(0, f.Call(KernelMemoryCompatExports.KernelMemoryPoolExpand, 0, 0x2000000, 0x200000, 0, f.Output + 0x38));
        var physical = f.Read(f.Output + 0x38);
        Assert.Equal((32u, 0u), BlockStats(f));

        Assert.Equal(0, f.Call(KernelMemoryCompatExports.KernelMemoryPoolCommit, address + 0x10000, 0x20000, 3, 3, 0));
        Assert.Equal((30u, 2u), BlockStats(f));
        Assert.True(f.Context.TryWriteUInt64(address + 0x2FFF8, 0xCAFEF00D));
        Assert.Equal(0xCAFEF00DUL, f.Read(address + 0x2FFF8));
        Assert.Equal(0x18u, f.Test.QueryStateFlags(address + 0x10000) & 0x18);
        Assert.Equal(physical, f.Read(f.Output + 0x110));
        // Committing over committed pages is refused.
        Assert.Equal(Eacces, f.Call(KernelMemoryCompatExports.KernelMemoryPoolCommit, address, 0x20000, 0, 3, 0));

        Assert.Equal(0, f.Call(KernelMemoryCompatExports.KernelMemoryPoolDecommit, address, 0x40000, 0));
        Assert.Equal((32u, 0u), BlockStats(f));
        Assert.Equal(0x08u, f.Test.QueryStateFlags(address + 0x10000) & 0x18);
        Assert.Equal(f.Test.Query(address), (address, address + 0x400000));

        // munmap of committed pool pages hands them back as well.
        Assert.Equal(0, f.Call(KernelMemoryCompatExports.KernelMemoryPoolCommit, address, PoolChunk, 0, 3, 0));
        Assert.Equal(0, f.Test.Unmap(address, 0x400000));
        Assert.Equal((32u, 0u), BlockStats(f));
    }

    [Fact]
    public void MemoryPoolMoveCarriesThePagesAndTheirContents()
    {
        using var f = new Fixture();
        var address = PoolReserve(f, 0x200000);
        Assert.Equal(0, f.Call(KernelMemoryCompatExports.KernelMemoryPoolExpand, 0, 0x2000000, 0x100000, 0, f.Output + 0x38));
        Assert.Equal(0, f.Call(KernelMemoryCompatExports.KernelMemoryPoolCommit, address, PoolChunk, 0, 3, 0));
        Assert.True(f.Context.TryWriteUInt64(address + 8, 0x5A5A5A5A));

        var destination = address + 0x100000;
        Assert.Equal(Einval, f.Call(KernelMemoryCompatExports.KernelMemoryPoolMove, address + 0x8000 * 0, address, PoolChunk, 0));
        Assert.Equal(Eacces, f.Call(KernelMemoryCompatExports.KernelMemoryPoolMove, destination, address + PoolChunk, PoolChunk, 0));
        Assert.Equal(0, f.Call(KernelMemoryCompatExports.KernelMemoryPoolMove, destination, address, PoolChunk, 0));
        Assert.Equal(0x5A5A5A5AUL, f.Read(destination + 8));
        Assert.Equal(0x18u, f.Test.QueryStateFlags(destination) & 0x18);
        Assert.Equal(0x08u, f.Test.QueryStateFlags(address) & 0x18);
        Assert.Equal((15u, 1u), BlockStats(f));
    }

    [Fact]
    public void MemoryPoolBatchRunsEntriesInOrderAndStopsAtTheFirstFailure()
    {
        using var f = new Fixture();
        var address = PoolReserve(f, 0x200000);
        Assert.Equal(0, f.Call(KernelMemoryCompatExports.KernelMemoryPoolExpand, 0, 0x2000000, 0x100000, 0, f.Output + 0x38));
        var entries = f.Output + 0x400;
        void Entry(int index, int op, ulong first, ulong second, ulong third)
        {
            var entry = entries + (ulong)index * 32;
            Assert.True(f.Context.TryWriteUInt32(entry, (uint)op));
            Assert.True(f.Context.TryWriteUInt32(entry + 4, 0));
            Assert.True(f.Context.TryWriteUInt64(entry + 8, first));
            Assert.True(f.Context.TryWriteUInt64(entry + 16, second));
            Assert.True(f.Context.TryWriteUInt64(entry + 24, third));
        }
        Entry(0, 1, address, 0x20000, 0x0203);   // commit, prot 3, type 2
        Entry(1, 3, address, PoolChunk, 0x01);    // protect read-only
        Entry(2, 2, address + PoolChunk, PoolChunk, 0); // decommit
        Entry(3, 9, 0, 0, 0);                     // unknown op
        Entry(4, 2, address, PoolChunk, 0);

        Assert.Equal(Einval, f.Call(KernelMemoryCompatExports.KernelMemoryPoolBatch, entries, 5, f.Output + 0x50, 0));
        Assert.Equal(3u, f.Read32(f.Output + 0x50));
        Assert.Equal((15u, 1u), BlockStats(f));
        Assert.Equal(0x18u, f.Test.QueryStateFlags(address) & 0x18);
        Assert.Equal(1u, f.Read32(f.Output + 0x118) & 0xFF);
        Assert.Equal(2u, f.Read32(f.Output + 0x11C));

        Assert.Equal(Einval, f.Call(KernelMemoryCompatExports.KernelMemoryPoolBatch, entries, unchecked((ulong)-1), f.Output + 0x50, 0));
    }

    [Fact]
    public void ConcurrentPoolCommitsKeepTheSupplyConsistent()
    {
        using var f = new Fixture();
        var address = PoolReserve(f, 0x200000);
        Assert.Equal(0, f.Call(KernelMemoryCompatExports.KernelMemoryPoolExpand, 0, 0x2000000, 0x100000, 0, f.Output + 0x38));
        var failures = 0;
        var threads = Enumerable.Range(0, 4).Select(index => new Thread(() =>
        {
            var context = new CpuContext(f.Test.Memory, Generation.Gen5);
            var chunk = address + (ulong)index * 0x40000;
            for (var round = 0; round < 100; round++)
            {
                context[CpuRegister.Rdi] = chunk;
                context[CpuRegister.Rsi] = 0x20000;
                context[CpuRegister.Rdx] = 0;
                context[CpuRegister.Rcx] = 3;
                if (KernelMemoryCompatExports.KernelMemoryPoolCommit(context) != 0)
                    Interlocked.Increment(ref failures);
                context[CpuRegister.Rdi] = chunk;
                context[CpuRegister.Rsi] = 0x20000;
                if (KernelMemoryCompatExports.KernelMemoryPoolDecommit(context) != 0)
                    Interlocked.Increment(ref failures);
            }
        })).ToArray();
        foreach (var thread in threads)
            thread.Start();
        foreach (var thread in threads)
            thread.Join();
        Assert.Equal(0, failures);
        Assert.Equal((16u, 0u), BlockStats(f));
    }

    [Fact]
    public void JitSharedMemoryViewsShareBytesWithinTheDescriptorProtection()
    {
        using var f = new Fixture();
        Assert.Equal(Einval, f.Call(KernelMemoryCompatExports.KernelJitCreateSharedMemory, 0, 0x2000, 7, f.Output + 0x10));
        Assert.Equal(0, f.Call(KernelMemoryCompatExports.KernelJitCreateSharedMemory, 0, 0x8000, 7, f.Output + 0x10));
        var fd = f.Read32(f.Output + 0x10);
        Assert.Equal(0x8000UL, f.Pool.Used);

        Assert.Equal(0, f.Call(KernelMemoryCompatExports.KernelJitCreateAliasOfSharedMemory, fd, 3, f.Output + 0x14));
        var writable = f.Read32(f.Output + 0x14);
        Assert.Equal(Eacces, f.Call(KernelMemoryCompatExports.KernelJitCreateAliasOfSharedMemory, writable, 7, f.Output + 0x18));
        Assert.Equal(Ebadf, f.Call(KernelMemoryCompatExports.KernelJitCreateAliasOfSharedMemory, 0x7FFF_0000, 1, f.Output + 0x18));

        Assert.True(f.Context.TryWriteUInt64(f.Output + 0x20, 0));
        Assert.Equal(0, f.Call(KernelMemoryCompatExports.KernelJitMapSharedMemory, writable, 3, f.Output + 0x20));
        var writeView = f.Read(f.Output + 0x20);
        Assert.True(f.Context.TryWriteUInt64(f.Output + 0x28, 0));
        Assert.Equal(0, f.Call(KernelMemoryCompatExports.KernelJitMapSharedMemory, fd, 5, f.Output + 0x28));
        var executeView = f.Read(f.Output + 0x28);
        Assert.NotEqual(writeView, executeView);
        Assert.Equal(Eacces, f.Call(KernelMemoryCompatExports.KernelJitMapSharedMemory, writable, 4, f.Output + 0x28));

        Assert.True(f.Context.TryWriteUInt64(writeView + 0x7000, 0xC3C3C3C3));
        Assert.Equal(0xC3C3C3C3UL, f.Read(executeView + 0x7000));

        // The backing stays while a view or a descriptor is left.
        Assert.Equal(0, f.Call(KernelMemoryCompatExports.KernelClose, fd));
        Assert.Equal(0, f.Call(KernelMemoryCompatExports.KernelClose, writable));
        Assert.Equal(0x8000UL, f.Pool.Used);
        Assert.Equal(0, f.Test.Unmap(writeView, 0x8000));
        Assert.Equal(0x8000UL, f.Pool.Used);
        Assert.Equal(0, f.Test.Unmap(executeView, 0x8000));
        Assert.Equal(0UL, f.Pool.Used);
    }
}
