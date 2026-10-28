// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE;
using SharpEmu.HLE.Host;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// A storage binding that crosses from one guest mapping into another whose backing pages
/// are elsewhere has no window and is bound as a ring copy. Silent Hill's lightmap table
/// was such a binding: the compute scatter that fills it stored into the copy, nothing
/// carried the stores back, and the base pass read zeros. The guest range here is two
/// mappings, the first megabyte of guest memory at the start of the backing and the second
/// at its end, with the binding across the edge between them.
/// </summary>
public sealed unsafe class MetalStagedStorageTests(ITestOutputHelper output)
{
    private const ulong GuestBase = 0x2_0000_0000;
    private const ulong BackingSize = 4 * 1024 * 1024;
    private const ulong MappingBytes = 1024 * 1024;
    private const ulong SecondMappingBacking = 3 * MappingBytes;
    private const ulong WindowBytes = 1024 * 1024;
    private const ulong MaxWindowBytes = 4 * WindowBytes;

    private const ulong Address = GuestBase + MappingBytes - (32 * 1024);
    private const ulong Size = 64 * 1024;

    private static readonly nint SelBlit = MetalNative.Selector("blitCommandEncoder");
    private static readonly nint SelEnd = MetalNative.Selector("endEncoding");
    private static readonly nint SelNewFence = MetalNative.Selector("newFence");
    private static readonly nint SelUpdateFence = MetalNative.Selector("updateFence:");
    private static readonly nint SelWaitForFence = MetalNative.Selector("waitForFence:");
    private static readonly nint SelRelease = MetalNative.Selector("release");
    private static readonly nint SelFill = MetalNative.Selector("fillBuffer:range:value:");

    private sealed class ScatteredBacking : IGuestBackedSpace, IDisposable
    {
        private readonly void* _block = NativeMemory.AlignedAlloc((nuint)BackingSize, 16384);

        public ScatteredBacking() => NativeMemory.Clear(_block, (nuint)BackingSize);

        public void Dispose() => NativeMemory.AlignedFree(_block);

        // The backing offset of a guest address and how many bytes its mapping has left.
        private static bool Map(ulong address, out ulong offset, out ulong left)
        {
            offset = left = 0;
            if (address < GuestBase || address - GuestBase >= 2 * MappingBytes)
            {
                return false;
            }

            var guest = address - GuestBase;
            var first = guest < MappingBytes;
            offset = first ? guest : SecondMappingBacking + (guest - MappingBytes);
            left = (first ? MappingBytes : 2 * MappingBytes) - guest;
            return true;
        }

        public bool TryHoldRange(ulong address, ulong size) => false;
        public bool TryHoldRangeAtOrAbove(ulong searchStart, ulong size, ulong alignment, out ulong address) { address = 0; return false; }
        public bool TryMapBacked(ulong address, ulong size, ulong backingOffset, GuestPageProtection protection, out HostViewFailure failure) { failure = HostViewFailure.None; return false; }
        public bool TryUnmapBacked(ulong address, ulong size) => false;
        public bool TryClearBacking(ulong offset, ulong size) => false;
        public bool IsBackedView(ulong address) => Map(address, out _, out _);
        public bool IsBackedRange(ulong address, ulong size) => Map(address, out _, out var left) && size <= left;
        public (ulong Base, ulong Size) BackingAliasRange => ((ulong)_block, BackingSize);

        // One mapping at a time: a range across the edge has no single alias.
        public bool TryResolveBackingAlias(ulong address, ulong size, out ulong alias)
        {
            alias = Map(address, out var offset, out var left) && size <= left ? (ulong)_block + offset : 0;
            return alias != 0;
        }

        public bool TryWriteBacking(ulong address, ReadOnlySpan<byte> data)
        {
            for (var done = 0; done < data.Length;)
            {
                if (!Map(address + (ulong)done, out var offset, out var left))
                {
                    return false;
                }

                var count = (int)Math.Min(left, (ulong)(data.Length - done));
                data.Slice(done, count).CopyTo(new Span<byte>((byte*)_block + offset, count));
                done += count;
            }

            return true;
        }

        public bool TryReadBacking(ulong address, Span<byte> data)
        {
            for (var done = 0; done < data.Length;)
            {
                if (!Map(address + (ulong)done, out var offset, out var left))
                {
                    return false;
                }

                var count = (int)Math.Min(left, (ulong)(data.Length - done));
                new ReadOnlySpan<byte>((byte*)_block + offset, count).CopyTo(data.Slice(done, count));
                done += count;
            }

            return true;
        }
    }

    private sealed class ScatteredMemory(ScatteredBacking backing) : ICpuMemory
    {
        public bool TryRead(ulong virtualAddress, Span<byte> destination) => backing.TryReadBacking(virtualAddress, destination);
        public bool TryWrite(ulong virtualAddress, ReadOnlySpan<byte> source) => backing.TryWriteBacking(virtualAddress, source);
        public bool CanRead(ulong address, ulong size) => backing.TryReadBacking(address, new byte[size]);
    }

    // The binding has no window, so it is staged; each mapping on its own has one.
    [Fact]
    public void TheBindingAcrossTheMappingEdgeHasNoWindow()
    {
        if (!RequireDevice())
        {
            return;
        }

        using var backing = new ScatteredBacking();
        using var memory = new MetalGuestMemory(
            Device.Value, backing, new ScatteredMemory(backing), WindowBytes, MaxWindowBytes);

        Assert.False(memory.TryResolveInPlace(Address, Size, out _));
        Assert.True(memory.TryResolveInPlace(Address, Size / 2, out _));
        Assert.True(memory.TryResolveInPlace(Address + (Size / 2), Size / 2, out _));
    }

    // What the shader stores into the copy lands in guest memory on both sides of the edge.
    [Fact]
    public void StoresIntoAStagedCopyReachGuestMemoryOnceTheirWorkIsRecorded()
    {
        if (!RequireDevice())
        {
            return;
        }

        using var backing = new ScatteredBacking();
        using var harness = new MetalTransferHarness(Device.Value);
        using var memory = new MetalGuestMemory(
            Device.Value, backing, new ScatteredMemory(backing), WindowBytes, MaxWindowBytes);
        var staged = new MetalStagedStorage(memory);

        harness.Begin();
        var copy = memory.Stage(Address, Size);
        staged.Add(Address, Size, copy, memory.Duplicate(copy), written: true);

        // Before its draw is recorded the copy has nothing to write back.
        Assert.Empty(staged.EncodeWritebacks(harness.CommandBuffer));

        // The shader's stores, then the writeback behind them.
        var fence = MetalNative.Send(Device.Value, SelNewFence);
        var store = MetalNative.Send(harness.CommandBuffer, SelBlit);
        Fill(store, copy, 0xCD);
        MetalNative.SendVoid(store, SelUpdateFence, fence);
        MetalNative.SendVoid(store, SelEnd);
        staged.Arm();
        var wait = MetalNative.Send(harness.CommandBuffer, SelBlit);
        MetalNative.SendVoid(wait, SelWaitForFence, fence);
        MetalNative.SendVoid(wait, SelEnd);
        Assert.Equal([(Address, Size)], staged.EncodeWritebacks(harness.CommandBuffer));
        staged.MarkWrittenBack();
        harness.Submit();
        MetalNative.SendVoid(fence, SelRelease);

        var bytes = new byte[Size];
        Assert.True(backing.TryReadBacking(Address, bytes));
        Assert.All(bytes, value => Assert.Equal((byte)0xCD, value));
        Assert.False(staged.HasRecordedStores);
    }

    // Bytes the shader left alone are not put back: something else wrote them after the
    // copy was staged, and the writeback keeps that.
    [Fact]
    public void AWritebackLeavesTheBytesTheShaderDidNotStore()
    {
        if (!RequireDevice())
        {
            return;
        }

        using var backing = new ScatteredBacking();
        using var harness = new MetalTransferHarness(Device.Value);
        using var memory = new MetalGuestMemory(
            Device.Value, backing, new ScatteredMemory(backing), WindowBytes, MaxWindowBytes);
        var staged = new MetalStagedStorage(memory);
        backing.TryWriteBacking(Address, Enumerable.Repeat((byte)0x11, (int)Size).ToArray());

        harness.Begin();
        var copy = memory.Stage(Address, Size);
        staged.Add(Address, Size, copy, memory.Duplicate(copy), written: true);
        // Another writer changes the whole range after it was staged.
        backing.TryWriteBacking(Address, Enumerable.Repeat((byte)0x22, (int)Size).ToArray());

        // The shader stores into the 4 KiB on either side of the edge only.
        var stored = new MetalGuestRange(copy.Buffer, copy.Offset + (Size / 2) - 4096, 8192, copy.GpuAddress);
        var store = MetalNative.Send(harness.CommandBuffer, SelBlit);
        Fill(store, stored, 0xCD);
        MetalNative.SendVoid(store, SelEnd);
        harness.Submit();

        harness.Begin();
        staged.Arm();
        staged.EncodeWritebacks(harness.CommandBuffer);
        harness.Submit();

        var bytes = new byte[Size];
        Assert.True(backing.TryReadBacking(Address, bytes));
        for (var index = 0; index < bytes.Length; index++)
        {
            var inside = index >= (int)(Size / 2) - 4096 && index < (int)(Size / 2) + 4096;
            Assert.Equal(inside ? (byte)0xCD : (byte)0x22, bytes[index]);
        }
    }

    // A pristine twin the GPU copies from the staged copy, ahead of the shader's stores, is
    // what the writeback compares against: the shader's bytes land, and a write from
    // elsewhere after the copy was staged survives everywhere else.
    [Fact]
    public void AGpuSnapshotIsThePristineTheWritebackComparesAgainst()
    {
        if (!RequireDevice())
        {
            return;
        }

        using var backing = new ScatteredBacking();
        using var harness = new MetalTransferHarness(Device.Value);
        using var memory = new MetalGuestMemory(
            Device.Value, backing, new ScatteredMemory(backing), WindowBytes, MaxWindowBytes);
        var staged = new MetalStagedStorage(memory);
        backing.TryWriteBacking(Address, Enumerable.Repeat((byte)0x11, (int)Size).ToArray());

        harness.Begin();
        var copy = memory.Stage(Address, Size);
        var pristine = memory.Allocate(Size, 256, out var pristineBytes);
        pristineBytes.Fill(0x77);
        staged.EncodeSnapshot(harness.CommandBuffer, copy, pristine);
        staged.Add(Address, Size, copy, pristine, written: true);
        harness.Submit();
        Assert.All(harness.Read(pristine.Buffer, pristine.Offset, Size), value => Assert.Equal((byte)0x11, value));

        backing.TryWriteBacking(Address, Enumerable.Repeat((byte)0x22, (int)Size).ToArray());
        harness.Begin();
        var stored = new MetalGuestRange(copy.Buffer, copy.Offset + (Size / 2) - 4096, 8192, copy.GpuAddress);
        var store = MetalNative.Send(harness.CommandBuffer, SelBlit);
        Fill(store, stored, 0xCD);
        MetalNative.SendVoid(store, SelEnd);
        harness.Submit();

        harness.Begin();
        staged.Arm();
        staged.EncodeWritebacks(harness.CommandBuffer);
        harness.Submit();

        var bytes = new byte[Size];
        Assert.True(backing.TryReadBacking(Address, bytes));
        for (var index = 0; index < bytes.Length; index++)
        {
            var inside = index >= (int)(Size / 2) - 4096 && index < (int)(Size / 2) + 4096;
            Assert.Equal(inside ? (byte)0xCD : (byte)0x22, bytes[index]);
        }
    }

    // The GPU copies the pristine twin only where the blit ends no pass, and ends a compute
    // encoder only for a range it would stage on the GPU whole.
    [Theory]
    [InlineData(false, false, 64 * 1024, true)]
    [InlineData(false, false, 64 * 1024 - 1, false)]
    [InlineData(true, false, 64 * 1024 * 1024, false)]
    [InlineData(false, true, 1024 * 1024, false)]
    [InlineData(false, true, 4 * 1024 * 1024, true)]
    public void TheGpuSnapshotsThePristineCopyOnlyWhereItEndsNoPass(bool renderOpen, bool computeOpen, ulong size, bool gpu)
    {
        Assert.Equal(gpu && MetalRenderHost.GpuPristine, MetalRenderHost.SnapshotsOnGpu(renderOpen, computeOpen, size));
    }

    // A copy recorded after a GPU write to its bytes that has not run yet holds that write,
    // not what the CPU saw when it was recorded, and so does its pristine copy.
    [Fact]
    public void AGatheredCopySeesTheGpuWriteRecordedBeforeIt()
    {
        if (!RequireDevice())
        {
            return;
        }

        using var backing = new ScatteredBacking();
        using var harness = new MetalTransferHarness(Device.Value);
        using var memory = new MetalGuestMemory(
            Device.Value, backing, new ScatteredMemory(backing), WindowBytes, MaxWindowBytes);
        var staged = new MetalStagedStorage(memory);
        backing.TryWriteBacking(Address, Enumerable.Repeat((byte)0x11, (int)Size).ToArray());

        harness.Begin();
        var fence = MetalNative.Send(Device.Value, SelNewFence);
        var write = MetalNative.Send(harness.CommandBuffer, SelBlit);
        Fill(write, memory.ResolveWritable(Address, Size / 2), 0xAB);
        Fill(write, memory.ResolveWritable(Address + (Size / 2), Size / 2), 0xAB);
        MetalNative.SendVoid(write, SelUpdateFence, fence);
        MetalNative.SendVoid(write, SelEnd);
        var wait = MetalNative.Send(harness.CommandBuffer, SelBlit);
        MetalNative.SendVoid(wait, SelWaitForFence, fence);
        MetalNative.SendVoid(wait, SelEnd);

        var copy = memory.Stage(Address, Size);
        var pristine = memory.Allocate(Size, 256, out _);
        staged.EncodeGather(harness.CommandBuffer, Address, copy, pristine);
        harness.Submit();
        MetalNative.SendVoid(fence, SelRelease);

        Assert.All(harness.Read(copy.Buffer, copy.Offset, Size), value => Assert.Equal((byte)0xAB, value));
        Assert.All(harness.Read(pristine.Buffer, pristine.Offset, Size), value => Assert.Equal((byte)0xAB, value));
    }

    // Every binding of the range shares one copy for the submission; a write from
    // anywhere else drops a copy with no stores of its own.
    [Fact]
    public void BindingsOfOneRangeShareACopyUntilSomethingElseWritesIt()
    {
        var staged = new MetalStagedStorage(null!);
        var copy = new MetalGuestRange(1, 0x100, Size, 0x1000);
        staged.Add(Address, Size, copy, copy, written: false);

        Assert.True(staged.TryShare(Address, Size, written: false, out var shared));
        Assert.Equal(copy, shared);
        Assert.False(staged.TryShare(Address, Size / 2, written: false, out _));

        staged.Forget(Address + Size - 1, 1);
        Assert.False(staged.TryShare(Address, Size, written: false, out _));
    }

    // Stores into a copy keep it through writes from elsewhere and through the end of the
    // submission until the work that makes them is recorded; a binding of other bytes
    // over them waits for them only once they are.
    [Fact]
    public void ACopyWithStoresIsKeptUntilItsWorkIsRecorded()
    {
        var staged = new MetalStagedStorage(null!);
        var copy = new MetalGuestRange(1, 0x100, Size, 0x1000);
        staged.Add(Address, Size, copy, copy, written: true);

        staged.Forget(Address, Size);
        staged.EndSubmission();
        Assert.True(staged.TryShare(Address, Size, written: false, out _));
        Assert.False(staged.HasRecordedStoresOver(Address + 16, 16));

        staged.Arm();
        Assert.True(staged.HasRecordedStoresOver(Address + 16, 16));
        Assert.False(staged.HasRecordedStoresOver(Address + Size, 16));

        // Bound written again for the next draw: its stores are not recorded yet.
        Assert.True(staged.TryShare(Address, Size, written: true, out _));
        Assert.False(staged.HasRecordedStores);
    }

    private static void Fill(nint encoder, in MetalGuestRange range, byte value)
    {
        Assert.True(range.IsValid);
        MetalNative.SendFillBuffer(encoder, SelFill, range.Buffer,
            new NsRange { Location = (nuint)range.Offset, Length = (nuint)range.Size }, value);
    }

    private bool RequireDevice()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Staged storage copies run on the device only behind the exclusive device gate.");
            return false;
        }

        Assert.True(OperatingSystem.IsMacOS());
        Assert.NotEqual(0, Device.Value);
        return true;
    }

    private static readonly Lazy<nint> Device = new(MetalNative.MTLCreateSystemDefaultDevice);
}
