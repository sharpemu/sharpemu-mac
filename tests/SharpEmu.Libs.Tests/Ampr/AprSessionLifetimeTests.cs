// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.HLE.FileSystem;
using SharpEmu.Libs.Ampr;
using SharpEmu.Libs.Kernel;
using SharpEmu.Libs.Tests.FileSystem;
using Xunit;

namespace SharpEmu.Libs.Tests.Ampr;

[CollectionDefinition("APR session lifetime", DisableParallelization = true)]
public sealed class AprSessionLifetimeCollection
{
    public const string Name = "APR session lifetime";
}

[Collection(AprSessionLifetimeCollection.Name)]
public sealed class AprSessionLifetimeTests
{
    private const ulong Base = 0x3_0000_0000;
    private const ulong Gate = Base + 0x800;
    private const ulong Done = Base + 0x810;
    private const ulong Destination = Base + 0x1000;

    [Fact]
    public void StopJoinsAnActiveReadBeforeArchiveAndMemoryCanBeReleased()
    {
        var directory = Directory.CreateTempSubdirectory("sharpemu-apr-lifetime-").FullName;
        var archivePath = Path.Combine(directory, "game.zar");
        var root = Path.Combine(directory, "cache");
        var bytes = Enumerable.Range(0, 4096).Select(i => (byte)i).ToArray();
        File.WriteAllBytes(archivePath, ZArchiveTestWriter.Build([("asset.bin", bytes)]));
        App0Archive.Mount(ZArchive.Open(archivePath)!, root);
        var memory = new ObservedMemory { BlockFill = true };
        var session = AmprExports.BeginRuntimeSession(memory);
        Task? stopping = null;
        try
        {
            var ctx = new CpuContext(memory, Generation.Gen5);
            var first = BuildBuffer(ctx, 0x100);
            var fileId = AmprFileRegistry.RegisterAprResolvedPath("/app0/asset.bin", Path.Combine(root, "asset.bin"));
            ctx[CpuRegister.Rsp] = Base + 0x900;
            Assert.True(ctx.TryWriteUInt64(Base + 0x908, 0));
            ctx[CpuRegister.Rdi] = first;
            ctx[CpuRegister.Rcx] = fileId;
            ctx[CpuRegister.R8] = Destination;
            ctx[CpuRegister.R9] = (ulong)bytes.Length;
            Assert.Equal(0, AmprExports.AprCommandBufferReadFile(ctx));
            Assert.Equal(0, Submit(ctx, first));
            Assert.True(memory.Entered.Wait(TimeSpan.FromSeconds(5)));

            var queued = BuildBuffer(ctx, 0x400);
            AppendWrite(ctx, queued, Done, 1);
            Assert.Equal(0, Submit(ctx, queued));
            stopping = Task.Run(session.Dispose);
            // Rejection proves that stop has acquired the admission gate. The blocked
            // callback still owns the archive and destination until shutdown joins it.
            Assert.True(SpinWait.SpinUntil(() => Submit(ctx, queued) != 0, TimeSpan.FromSeconds(5)));
            Assert.False(stopping.IsCompleted);
            Assert.True(App0Archive.IsMounted);
            memory.Release.Set();
            Assert.True(stopping.Wait(TimeSpan.FromSeconds(5)));
            Assert.Equal(0UL, Read(ctx, Done));
            var actual = new byte[bytes.Length];
            Assert.True(memory.TryRead(Destination, actual));
            Assert.Equal(bytes, actual);
            App0Archive.Unmount();
            memory.Released = true;
            // A late submission is rejected without capturing or writing guest memory.
            Assert.NotEqual(0, Submit(ctx, queued));
            Assert.Equal(0, memory.AccessAfterRelease);
            session.Dispose();
        }
        finally
        {
            memory.Release.Set();
            session.Dispose();
            stopping?.Wait(TimeSpan.FromSeconds(5));
            App0Archive.Unmount();
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void StopCancelsAnUnsatisfiedAddressWaitAndANewSessionCanReuseTheMemory()
    {
        var memory = new ObservedMemory();
        var ctx = new CpuContext(memory, Generation.Gen5);
        using (var session = AmprExports.BeginRuntimeSession(memory))
        {
            var buffer = BuildBuffer(ctx, 0x100);
            var manager = new ModuleManager();
            manager.RegisterExports(SharpEmu.Generated.SysAbiExportRegistry.CreateExports(Generation.Gen5));
            ctx[CpuRegister.Rdi] = buffer;
            ctx[CpuRegister.Rsi] = Gate;
            ctx[CpuRegister.Rdx] = 7;
            ctx[CpuRegister.Rcx] = 0;
            ctx[CpuRegister.R8] = 0;
            Assert.Equal(OrbisGen2Result.ORBIS_GEN2_OK, manager.Dispatch("V7GQTEeUfhw", ctx));
            AppendWrite(ctx, buffer, Done, 1);
            Assert.Equal(0, Submit(ctx, buffer));
            Assert.True(memory.Entered.Wait(TimeSpan.FromSeconds(5)));
            var stop = Task.Run(session.Dispose);
            Assert.True(stop.Wait(TimeSpan.FromSeconds(5)));
            Assert.Equal(0UL, Read(ctx, Done));
            Assert.NotEqual(0, Submit(ctx, buffer));
        }
        using (AmprExports.BeginRuntimeSession(memory))
        {
            var buffer = BuildBuffer(ctx, 0x400);
            AppendWrite(ctx, buffer, Done, 2);
            Assert.Equal(0, Submit(ctx, buffer, returnId: true));
            Assert.True(ctx.TryReadUInt32(Base + 0x40, out var id));
            ctx[CpuRegister.Rdi] = id;
            Assert.Equal(0, KernelAprCompatExports.KernelAprWaitCommandBuffer(ctx));
            Assert.Equal(2UL, Read(ctx, Done));
        }
    }

    [Fact]
    public void CacheEvictionDoesNotCloseAFileWithAnActiveReadLease()
    {
        var directory = Directory.CreateTempSubdirectory("sharpemu-apr-lease-").FullName;
        var path = Path.Combine(directory, "asset.bin");
        var bytes = new byte[] { 4, 5, 6, 7 };
        File.WriteAllBytes(path, bytes);
        var memory = new ObservedMemory { BlockFill = true };
        var ctx = new CpuContext(memory, Generation.Gen5);
        using var session = AmprExports.BeginRuntimeSession(memory);
        try
        {
            var buffer = BuildReadBuffer(ctx, path, bytes.Length);
            Assert.Equal(0, Submit(ctx, buffer, returnId: true));
            Assert.True(ctx.TryReadUInt32(Base + 0x40, out var id));
            Assert.True(memory.Entered.Wait(TimeSpan.FromSeconds(5)));
            // Exercise the same eviction used for EMFILE recovery and session teardown.
            typeof(AmprExports).GetMethod("EvictAllCachedHostFiles",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)!.Invoke(null, null);
            memory.Release.Set();
            ctx[CpuRegister.Rdi] = id;
            Assert.Equal(0, KernelAprCompatExports.KernelAprWaitCommandBuffer(ctx));
            Assert.True(ctx.TryReadUInt64(Base + 0x60, out var result));
            Assert.Equal(0UL, result);
            var actual = new byte[bytes.Length];
            Assert.True(memory.TryRead(Destination, actual));
            Assert.Equal(bytes, actual);
        }
        finally
        {
            memory.Release.Set();
            session.Dispose();
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void RestartWithTheSameArchivePathUsesTheNewMount()
    {
        var directory = Directory.CreateTempSubdirectory("sharpemu-apr-remount-").FullName;
        var path = Path.Combine(directory, "game.zar");
        var root = Path.Combine(directory, "cache");
        var memory = new ObservedMemory();
        var ctx = new CpuContext(memory, Generation.Gen5);
        try
        {
            foreach (var value in new byte[] { 1, 2 })
            {
                File.WriteAllBytes(path, ZArchiveTestWriter.Build([("asset.bin", new byte[] { value })]));
                App0Archive.Mount(ZArchive.Open(path)!, root);
                using (AmprExports.BeginRuntimeSession(memory))
                {
                    var buffer = BuildReadBuffer(ctx, Path.Combine(root, "asset.bin"), 1);
                    Assert.Equal(0, Submit(ctx, buffer, returnId: true));
                    Assert.True(ctx.TryReadUInt32(Base + 0x40, out var id));
                    ctx[CpuRegister.Rdi] = id;
                    Assert.Equal(0, KernelAprCompatExports.KernelAprWaitCommandBuffer(ctx));
                    var actual = new byte[1];
                    Assert.True(memory.TryRead(Destination, actual));
                    Assert.Equal(value, actual[0]);
                }
                App0Archive.Unmount();
            }
        }
        finally
        {
            App0Archive.Unmount();
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void StopWaitsForAnAdmittedIdWriteWithoutHoldingTheGlobalGate()
    {
        var memory = new ObservedMemory();
        var ctx = new CpuContext(memory, Generation.Gen5);
        var session = AmprExports.BeginRuntimeSession(memory);
        var buffer = BuildBuffer(ctx, 0x100);
        AppendWrite(ctx, buffer, Done, 1);
        memory.BlockIdWrite = true;
        Task<int>? producer = null;
        Task? stop = null;
        try
        {
            producer = Task.Run(() => Submit(ctx, buffer, returnId: true));
            Assert.True(memory.Entered.Wait(TimeSpan.FromSeconds(5)));
            stop = Task.Run(session.Dispose);
            var stopping = session.GetType().GetField("Stopping",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
            // Captured command state can be cached. Park the actual id write so
            // admission is deterministically inside guest memory, outside the gate.
            Assert.True(SpinWait.SpinUntil(() => (bool)stopping.GetValue(session)!, TimeSpan.FromSeconds(5)));
            Assert.False(stop.IsCompleted);

            // A newly allocated tracker must resolve the stopped root owner too.
            var lateContext = new CpuContext(new SharpEmu.Core.Cpu.TrackedCpuMemory(memory), Generation.Gen5);
            var late = Task.Run(() => Submit(lateContext, buffer));
            Assert.True(late.Wait(TimeSpan.FromSeconds(5)));
            Assert.NotEqual(0, late.Result);
            memory.Release.Set();
            Assert.True(producer.Wait(TimeSpan.FromSeconds(5)));
            Assert.NotEqual(0, producer.Result);
            Assert.True(stop.Wait(TimeSpan.FromSeconds(5)));
            memory.BlockIdWrite = false;
            Assert.Equal(0UL, Read(ctx, Done));
            memory.Released = true;
            var afterStop = new CpuContext(new SharpEmu.Core.Cpu.TrackedCpuMemory(memory), Generation.Gen5);
            Assert.NotEqual(0, Submit(afterStop, buffer));
            Assert.Equal(0, memory.AccessAfterRelease);
        }
        finally
        {
            memory.Release.Set();
            session.Dispose();
            producer?.Wait(TimeSpan.FromSeconds(5));
            stop?.Wait(TimeSpan.FromSeconds(5));
        }
    }

    private static ulong BuildReadBuffer(CpuContext ctx, string path, int length)
    {
        var buffer = BuildBuffer(ctx, 0x100);
        var id = AmprFileRegistry.RegisterAprResolvedPath("/app0/" + Guid.NewGuid() + "/asset.bin", path);
        ctx[CpuRegister.Rsp] = Base + 0x900;
        Assert.True(ctx.TryWriteUInt64(Base + 0x908, 0));
        ctx[CpuRegister.Rdi] = buffer;
        ctx[CpuRegister.Rcx] = id;
        ctx[CpuRegister.R8] = Destination;
        ctx[CpuRegister.R9] = (ulong)length;
        Assert.Equal(0, AmprExports.AprCommandBufferReadFile(ctx));
        return buffer;
    }

    private static ulong BuildBuffer(CpuContext ctx, ulong offset)
    {
        var buffer = Base + offset;
        ctx[CpuRegister.Rdi] = buffer;
        ctx[CpuRegister.Rsi] = buffer + 0x40;
        ctx[CpuRegister.Rdx] = 0x100;
        Assert.Equal(0, AmprExports.CommandBufferConstructor(ctx));
        Assert.Equal(0, AmprExports.CommandBufferSetBuffer(ctx));
        return buffer;
    }

    private static void AppendWrite(CpuContext ctx, ulong buffer, ulong address, ulong value)
    {
        var manager = new ModuleManager();
        manager.RegisterExports(SharpEmu.Generated.SysAbiExportRegistry.CreateExports(Generation.Gen5));
        ctx[CpuRegister.Rdi] = buffer;
        ctx[CpuRegister.Rsi] = address;
        ctx[CpuRegister.Rdx] = value;
        Assert.Equal(OrbisGen2Result.ORBIS_GEN2_OK, manager.Dispatch("j0+3uJMxYJY", ctx));
    }

    private static int Submit(CpuContext ctx, ulong buffer, bool returnId = false)
    {
        return AmprExports.SubmitCommandBuffer(ctx, buffer, priority: 0,
            resultAddress: Base + 0x60, outIdAddress: returnId ? Base + 0x40 : 0,
            dsReturnAddress: 0, out _);
    }

    private static ulong Read(CpuContext ctx, ulong address)
    {
        Assert.True(ctx.TryReadUInt64(address, out var value));
        return value;
    }

    private sealed class ObservedMemory : ICpuMemory
    {
        private readonly FakeCpuMemory _inner = new(Base, 0x4000);
        public readonly ManualResetEventSlim Entered = new();
        public readonly ManualResetEventSlim Release = new();
        public bool BlockFill;
        public volatile bool BlockIdWrite;
        public volatile bool Released;
        public int AccessAfterRelease;

        private void CheckAlive()
        {
            if (Released) Interlocked.Increment(ref AccessAfterRelease);
        }

        public bool TryRead(ulong address, Span<byte> destination)
        {
            CheckAlive();
            if (address == Gate) Entered.Set();
            return _inner.TryRead(address, destination);
        }

        public bool TryWrite(ulong address, ReadOnlySpan<byte> source)
        {
            CheckAlive();
            if (BlockIdWrite && address == Base + 0x40)
            {
                Entered.Set();
                Release.Wait();
            }
            return _inner.TryWrite(address, source);
        }

        public bool TryFillInPlace<TState>(ulong address, int length, TState state,
            GuestSpanFill<TState> fill, out int filled)
        {
            CheckAlive();
            if (BlockFill)
            {
                Entered.Set();
                Release.Wait();
            }
            var bytes = new byte[length];
            filled = fill(state, bytes);
            return _inner.TryWrite(address, bytes.AsSpan(0, filled));
        }
    }
}
