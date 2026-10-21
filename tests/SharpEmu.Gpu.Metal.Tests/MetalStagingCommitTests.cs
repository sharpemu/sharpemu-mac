// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.HLE.Host;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// The upload rings and the transfer staging a submission borrows return to their pools
/// only when its command buffer has completed, so a buffer that keeps recording holds
/// every ring and slab it touched and each pool grows by one more. Once polls stopped
/// committing, one of Silent Hill's loading buffers took the ring pool to 9.9 GB. The
/// host commits the buffer when what it borrowed passes a budget: between packets at
/// the next transfer, inside a draw's preparation at the next preparation, since the
/// images that preparation already acquired carry its serial.
/// </summary>
public sealed unsafe class MetalStagingCommitTests(ITestOutputHelper output)
{
    private const ulong GuestBase = 0x2_0000_0000;
    private const ulong Budget = 1024 * 1024;

    [Fact]
    public void ATransferBetweenPacketsCommitsTheBufferOnceItsStagingPassesTheBudget()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.TransientCommitBytes = Budget;
        var owner = (IMetalImageCacheOwner)host;

        var context = owner.BeginTransfer(MetalTransferCause.Upload);
        var serial = host.SubmissionSerial;
        _ = host.StagingTiler!.Acquire(context, Budget / 4);
        Assert.Equal(Budget / 4, host.RecordingTransientBytes);

        // Under the budget the next transfer records into the same buffer.
        _ = owner.BeginTransfer(MetalTransferCause.Upload);
        Assert.Equal(serial, host.SubmissionSerial);
        Assert.Equal(0L, host.TakeTransientCommits());

        _ = host.StagingTiler.Acquire(context, Budget);
        Assert.True(host.RecordingTransientBytes >= Budget);

        // Over it, the transfer that follows opens a new one and the slabs are no longer
        // the recording buffer's.
        _ = owner.BeginTransfer(MetalTransferCause.Upload);
        Assert.Equal(serial + 1, host.SubmissionSerial);
        Assert.Equal(1L, host.TakeTransientCommits());
        Assert.Equal(0UL, host.RecordingTransientBytes);
    }

    [Fact]
    public void ARingAllocationCountsTheWholeRingAgainstTheBuffer()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.TransientCommitBytes = Budget;
        var owner = (IMetalImageCacheOwner)host;

        _ = owner.BeginTransfer(MetalTransferCause.GuestCopy);
        var serial = host.SubmissionSerial;
        // A reservation far smaller than the budget still lends the buffer a whole
        // ring, which is far larger than it.
        _ = host.GuestMemory.Allocate(4096, 16, out _);
        Assert.Equal(host.GuestMemory.RingBytes, host.RecordingTransientBytes);
        Assert.True(host.RecordingTransientBytes >= Budget);

        _ = owner.BeginTransfer(MetalTransferCause.GuestCopy);
        Assert.Equal(serial + 1, host.SubmissionSerial);
        Assert.Equal(1L, host.TakeTransientCommits());
        Assert.Equal(0UL, host.RecordingTransientBytes);
    }

    // A ring an earlier, unfinished buffer still holds stays in the pool whatever the next
    // buffer does, so that buffer counts only what it reserves there; once the earlier one
    // has completed the ring is free again and the buffer that takes it counts it whole.
    // Counting a held ring whole again made every buffer pass the budget after a few small
    // uploads once the pool had a couple of large rings.
    [Fact]
    public void ARingAnUnfinishedBufferStillHoldsCountsOnlyWhatTheNextBufferReserves()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var memory = rig.Host.GuestMemory;

        memory.BeginSubmission(10, 0);
        _ = memory.Allocate(4096, 16, out _);
        var ring = memory.RingBytes;
        Assert.Equal(ring, memory.RecordingBytes);

        memory.BeginSubmission(11, 0);
        _ = memory.Allocate(4096, 16, out _);
        _ = memory.Allocate(4096, 16, out _);
        Assert.Equal(ring, memory.RingBytes);
        Assert.Equal(2 * 4096UL, memory.RecordingBytes);

        memory.BeginSubmission(12, 11);
        _ = memory.Allocate(4096, 16, out _);
        Assert.Equal(ring, memory.RingBytes);
        Assert.Equal(ring, memory.RecordingBytes);
    }

    [Fact]
    public void ATransferInsideAPreparationLeavesTheCommitToTheNextPreparation()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.TransientCommitBytes = Budget;
        var owner = (IMetalImageCacheOwner)host;

        ulong serial;
        using (host.BeginPreparation())
        {
            var context = owner.BeginTransfer(MetalTransferCause.Upload);
            serial = host.SubmissionSerial;
            _ = host.StagingTiler!.Acquire(context, Budget);
            _ = owner.BeginTransfer(MetalTransferCause.Upload);
            Assert.Equal(serial, host.SubmissionSerial);
            Assert.Equal(0L, host.TakeTransientCommits());
        }

        using (host.BeginPreparation())
        {
            Assert.Equal(serial + 1, host.SubmissionSerial);
            Assert.Equal(1L, host.TakeTransientCommits());
        }
    }

    [Fact]
    public void ASlabLentAgainCountsTowardTheBufferThatTookIt()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        var owner = (IMetalImageCacheOwner)host;
        var tiler = host.StagingTiler!;

        var first = owner.BeginTransfer(MetalTransferCause.Upload);
        _ = tiler.Acquire(first, Budget);
        var lent = tiler.RecordingBytes;
        Assert.True(lent >= Budget);
        Assert.Equal(host.SubmissionSerial, tiler.RecordingSerial);

        // The slab returns once its buffer has completed, and the buffer that takes it
        // next starts its own count from it.
        host.FlushAndWaitForGpu();
        var second = owner.BeginTransfer(MetalTransferCause.Upload);
        _ = tiler.Acquire(second, Budget);
        Assert.Equal(host.SubmissionSerial, tiler.RecordingSerial);
        Assert.Equal(lent, tiler.RecordingBytes);
        Assert.Equal(lent, tiler.StagingBytes);
    }

    private sealed class AddressSpace : IGuestAddressSpace
    {
        public ulong ProtectionPageSize => 4096;
        public bool TryProtect(ulong address, ulong size, GuestPageProtection protection) => true;
        public ulong AllocateAt(ulong desiredAddress, ulong size, bool executable = true, bool allowAlternative = true) => throw new NotSupportedException();
        public bool TryBackFixedRange(ulong address, ulong size, bool executable) => throw new NotSupportedException();
        public bool TryAllocateAtOrAbove(ulong desiredAddress, ulong size, bool executable, ulong alignment, out ulong actualAddress) => throw new NotSupportedException();
        public bool TryEnsureRangeCommitted(ulong address, ulong size) => throw new NotSupportedException();
        public bool TryAllocateGuestMemory(ulong size, ulong alignment, out ulong address) => throw new NotSupportedException();
        public bool TryFreeGuestMemory(ulong address) => throw new NotSupportedException();
    }

    // One page-aligned native block seen as guest memory at GuestBase.
    private sealed class Backing : IGuestBackedSpace, ICpuMemory, IDisposable
    {
        public const ulong Size = 1024 * 1024;
        private readonly void* _block = NativeMemory.AlignedAlloc((nuint)Size, 16384);

        public Backing() => NativeMemory.Clear(_block, (nuint)Size);

        public void Dispose() => NativeMemory.AlignedFree(_block);

        private bool Contains(ulong address, ulong length) => address >= GuestBase && address - GuestBase + length <= Size;

        public bool TryHoldRange(ulong address, ulong size) => false;
        public bool TryHoldRangeAtOrAbove(ulong searchStart, ulong size, ulong alignment, out ulong address) { address = 0; return false; }
        public bool TryMapBacked(ulong address, ulong size, ulong backingOffset, GuestPageProtection protection, out HostViewFailure failure) { failure = HostViewFailure.None; return false; }
        public bool TryUnmapBacked(ulong address, ulong size) => false;
        public bool TryClearBacking(ulong offset, ulong size) => false;
        public bool IsBackedView(ulong address) => Contains(address, 1);
        public bool IsBackedRange(ulong address, ulong size) => Contains(address, size);
        public (ulong Base, ulong Size) BackingAliasRange => ((ulong)_block, Size);

        public bool TryResolveBackingAlias(ulong address, ulong size, out ulong alias)
        {
            alias = Contains(address, size) ? (ulong)_block + (address - GuestBase) : 0;
            return alias != 0;
        }

        public bool TryWriteBacking(ulong address, ReadOnlySpan<byte> data) => TryWrite(address, data);
        public bool TryReadBacking(ulong address, Span<byte> data) => TryRead(address, data);

        public bool TryRead(ulong virtualAddress, Span<byte> destination)
        {
            if (!Contains(virtualAddress, (ulong)destination.Length)) return false;
            new ReadOnlySpan<byte>((byte*)_block + (virtualAddress - GuestBase), destination.Length).CopyTo(destination);
            return true;
        }

        public bool TryWrite(ulong virtualAddress, ReadOnlySpan<byte> source)
        {
            if (!Contains(virtualAddress, (ulong)source.Length)) return false;
            source.CopyTo(new Span<byte>((byte*)_block + (virtualAddress - GuestBase), source.Length));
            return true;
        }

        public bool CanRead(ulong address, ulong size) => Contains(address, size);
    }

    private sealed class Rig : IDisposable
    {
        private readonly nint _queue;
        private readonly AddressSpace _space = new();
        private readonly PageGuard _pages;
        private readonly Backing _backing = new();

        public Rig(nint device)
        {
            _queue = MetalNative.Send(device, MetalNative.Selector("newCommandQueue"));
            _pages = new PageGuard(_space);
            Host = new MetalRenderHost(device, _queue, _backing, _backing, _pages, new MetalGpuWorkerRelay(static () => { }));
        }

        public MetalRenderHost Host { get; }

        public void Dispose()
        {
            Host.ImageCache.StopWatching();
            Host.Dispose();
            _pages.Dispose();
            _backing.Dispose();
            MetalNative.SendVoid(_queue, MetalNative.Selector("release"));
        }
    }

    private bool RequireDevice()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("The transient budget commits on the device only behind the exclusive device gate.");
            return false;
        }

        Assert.True(OperatingSystem.IsMacOS());
        Assert.NotEqual(0, Device.Value);
        return true;
    }

    private static readonly Lazy<nint> Device = new(MetalNative.MTLCreateSystemDefaultDevice);
}
