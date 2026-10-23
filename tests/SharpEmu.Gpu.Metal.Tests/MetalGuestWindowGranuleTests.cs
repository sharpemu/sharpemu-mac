// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE;
using SharpEmu.HLE.Host;
using SharpEmu.Libs.Gpu.Abstractions;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// Guest memory windows over whole granules of the backing: a range inside one granule binds
/// that granule's window, a range across granule edges a span window over exactly the
/// granules it covers, in place, so it sees the GPU writes before it in stream order and its
/// own stores reach guest memory. A small backing with 1 MiB granules stands in for the
/// 32 MiB ones; device work is behind the exclusive device gate.
/// </summary>
public sealed unsafe class MetalGuestWindowGranuleTests(ITestOutputHelper output)
{
    internal const ulong GuestBase = 0x2_0000_0000;
    internal const ulong BackingSize = 4 * 1024 * 1024;
    internal const ulong GranuleBytes = 1024 * 1024;
    internal const ulong MaxWindowBytes = 2 * GranuleBytes;

    // 32 KiB on either side of the first granule edge.
    private const ulong StraddleAddress = GuestBase + GranuleBytes - (32 * 1024);
    private const ulong StraddleSize = 64 * 1024;

    private static readonly nint SelBlit = MetalNative.Selector("blitCommandEncoder");
    private static readonly nint SelEnd = MetalNative.Selector("endEncoding");
    private static readonly nint SelNewFence = MetalNative.Selector("newFence");
    private static readonly nint SelUpdateFence = MetalNative.Selector("updateFence:");
    private static readonly nint SelWaitForFence = MetalNative.Selector("waitForFence:");
    private static readonly nint SelRelease = MetalNative.Selector("release");
    private static readonly nint SelFill = MetalNative.Selector("fillBuffer:range:value:");
    private static readonly nint SelCopy =
        MetalNative.Selector("copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:");

    internal sealed class Backing : IGuestBackedSpace, IDisposable
    {
        private readonly void* _block = NativeMemory.AlignedAlloc((nuint)BackingSize, 16384);

        public Backing() => NativeMemory.Clear(_block, (nuint)BackingSize);

        public void Dispose() => NativeMemory.AlignedFree(_block);

        private static bool Contains(ulong address, ulong length) =>
            address >= GuestBase && address - GuestBase + length <= BackingSize;

        public bool TryHoldRange(ulong address, ulong size) => false;
        public bool TryHoldRangeAtOrAbove(ulong searchStart, ulong size, ulong alignment, out ulong address) { address = 0; return false; }
        public bool TryMapBacked(ulong address, ulong size, ulong backingOffset, GuestPageProtection protection, out HostViewFailure failure) { failure = HostViewFailure.None; return false; }
        public bool TryUnmapBacked(ulong address, ulong size) => false;
        public bool TryClearBacking(ulong offset, ulong size) => false;
        public bool IsBackedView(ulong address) => Contains(address, 1);
        public bool IsBackedRange(ulong address, ulong size) => Contains(address, size);
        public (ulong Base, ulong Size) BackingAliasRange => ((ulong)_block, BackingSize);
        public object? BackingAliasSnapshot => Snapshot;
        public bool IsBackingAliasSnapshotCurrent(object snapshot) => WindowsAvailable && ReferenceEquals(snapshot, Snapshot);
        internal bool WindowsAvailable = true;
        internal object Snapshot = new();
        internal ulong MappingOffset;
        internal int Resolves;

        public bool TryResolveBackingAlias(ulong address, ulong size, out ulong alias)
        {
            Resolves++;
            alias = WindowsAvailable && Contains(address, size) && MappingOffset + address - GuestBase + size <= BackingSize
                ? (ulong)_block + MappingOffset + (address - GuestBase) : 0;
            return alias != 0;
        }

        public bool TryWriteBacking(ulong address, ReadOnlySpan<byte> data)
        {
            if (!Contains(address, (ulong)data.Length)) return false;
            data.CopyTo(new Span<byte>((byte*)_block + (address - GuestBase), data.Length));
            return true;
        }

        public bool TryReadBacking(ulong address, Span<byte> data)
        {
            if (!Contains(address, (ulong)data.Length)) return false;
            new ReadOnlySpan<byte>((byte*)_block + (address - GuestBase), data.Length).CopyTo(data);
            return true;
        }
    }

    // The CPU view of the same bytes, which is what the ring staged from.
    internal sealed class BackingMemory(Backing backing) : ICpuMemory
    {
        public bool TryRead(ulong virtualAddress, Span<byte> destination) => backing.TryReadBacking(virtualAddress, destination);
        public bool TryWrite(ulong virtualAddress, ReadOnlySpan<byte> source) => backing.TryWriteBacking(virtualAddress, source);
        public bool CanRead(ulong address, ulong size) => backing.IsBackedRange(address, size);
    }

    private static MetalGuestMemory Memory(Backing backing, ulong maxUploadBytes = 64UL * 1024 * 1024) =>
        new(Device.Value, backing, new BackingMemory(backing), GranuleBytes, MaxWindowBytes, maxUploadBytes);

    [Fact]
    public void WindowLookupReusesOnlyTheSameMappingPartitionAndSubmission()
    {
        if (!RequireDevice()) return;
        using var backing = new Backing();
        using var memory = Memory(backing);
        memory.BeginSubmission(1, 0);
        var address = GuestBase + 256;
        var first = memory.ResolveWritable(address, 16);
        var resolves = backing.Resolves;
        Assert.Equal(first, memory.ResolveWritable(address, 16));
        Assert.Equal(resolves, backing.Resolves);
        Assert.Equal(1, memory.WindowLookupHits);
        // Admission can be revoked without replacing the mapping snapshot.
        backing.WindowsAvailable = false;
        Assert.False(memory.ResolveWritable(address, 16).IsValid);
        Assert.Equal(1, memory.WindowLookupHits);
        backing.WindowsAvailable = true;
        Assert.True(memory.ResolveWritable(address, 16).IsValid);
        Assert.Equal(1, memory.WindowLookupHits);
        backing.MappingOffset = GranuleBytes;
        backing.Snapshot = new();
        var remapped = memory.ResolveWritable(address, 16);
        Assert.NotEqual(first.GpuAddress, remapped.GpuAddress);
        Assert.Equal(1, memory.WindowLookupHits);
        backing.MappingOffset = 0; backing.Snapshot = new();
        memory.BeginSubmission(2, 0);
        var straddle = memory.ResolveWritable(StraddleAddress, StraddleSize);
        Assert.True(straddle.IsValid);
        var changed = memory.ResolveWritable(address, 16);
        Assert.Equal(straddle.Buffer, changed.Buffer);
        Assert.Equal(changed, memory.ResolveWritable(address, 16));
        Assert.Equal(2, memory.WindowLookupHits);
        memory.BeginSubmission(3, 0);
        Assert.True(memory.ResolveWritable(address, 16).IsValid);
        Assert.Equal(2, memory.WindowLookupHits);
    }

    // Two buffers over the same pages are not coherent inside a command buffer on Apple GPUs:
    // a GPU write through one and a later read through the other in the same command buffer
    // read the old bytes, fence or not. So every resolve of a byte in one command buffer has
    // to hand out the same buffer. A range across the edge resolved first moves the edge, and
    // its two halves resolved after it lie in the same window: a GPU write through the halves
    // is what a read through the whole range sees, in the same command buffer.
    [Fact]
    public void WritesAndReadsThroughDifferentResolvesInOneCommandBufferAgree()
    {
        if (!RequireDevice())
        {
            return;
        }

        using var backing = new Backing();
        using var harness = new MetalTransferHarness(Device.Value);
        using var memory = Memory(backing);
        backing.TryWriteBacking(StraddleAddress, Enumerable.Repeat((byte)0x11, (int)StraddleSize).ToArray());

        harness.Begin();
        memory.BeginSubmission(1, 0);
        var straddle = memory.Resolve(StraddleAddress, StraddleSize);
        Assert.True(memory.IsInPlace(straddle));
        var half = StraddleSize / 2;
        var low = memory.Resolve(StraddleAddress, half);
        var high = memory.Resolve(StraddleAddress + half, half);
        Assert.Equal(straddle.Buffer, low.Buffer);
        Assert.Equal(straddle.Buffer, high.Buffer);

        var write = MetalNative.Send(harness.CommandBuffer, SelBlit);
        Fill(write, low, 0xAB);
        Fill(write, high, 0xAB);
        // The windows are untracked, so the read is ordered after the write the way the
        // backend orders guest work: a fence between the two encoders.
        var fence = MetalNative.Send(Device.Value, SelNewFence);
        MetalNative.SendVoid(write, SelUpdateFence, fence);
        MetalNative.SendVoid(write, SelEnd);

        var readback = harness.Buffer(StraddleSize);
        var read = MetalNative.Send(harness.CommandBuffer, SelBlit);
        MetalNative.SendVoid(read, SelWaitForFence, fence);
        MetalNative.SendCopyBuffer(read, SelCopy, straddle.Buffer, (nuint)straddle.Offset, readback, 0, (nuint)StraddleSize);
        MetalNative.SendVoid(read, SelEnd);
        harness.Submit();
        MetalNative.SendVoid(fence, SelRelease);

        Assert.All(harness.Read(readback, 0, StraddleSize), value => Assert.Equal((byte)0xAB, value));
        Assert.Equal(0UL, memory.RingFallbacks);
    }

    // The other order: the halves were resolved first, each in its own window, and this
    // command buffer may already have written through them. The edge cannot move now without
    // giving those bytes a second GPU address, so the whole range gets no window in this
    // command buffer - never one that aliases the halves' - and the edge moves as the next
    // one begins, after which the range and its halves share one window again.
    [Fact]
    public void ARangeAcrossAnEdgeThisCommandBufferUsedWaitsForTheNextOne()
    {
        if (!RequireDevice())
        {
            return;
        }

        using var backing = new Backing();
        using var memory = Memory(backing);
        memory.BeginSubmission(1, 0);
        var half = StraddleSize / 2;
        var low = memory.Resolve(StraddleAddress, half);
        var high = memory.Resolve(StraddleAddress + half, half);
        Assert.NotEqual(low.Buffer, high.Buffer);

        Assert.False(memory.TryResolveInPlace(StraddleAddress, StraddleSize, out _));
        Assert.False(memory.ResolveWritable(StraddleAddress, StraddleSize).IsValid);
        var staged = memory.Resolve(StraddleAddress, StraddleSize);
        Assert.False(memory.IsInPlace(staged));
        // A prefix is what the window holding its first byte holds as the backing is cut now.
        var prefix = memory.ResolveWritablePrefix(StraddleAddress, StraddleSize);
        Assert.Equal(low.Buffer, prefix.Buffer);
        Assert.Equal(half, prefix.Size);
        var (moves, conflicts, _) = memory.WindowLayoutCounts;
        Assert.Equal(0L, moves);
        Assert.True(conflicts >= 1);

        memory.BeginSubmission(2, 1);
        Assert.Equal(1L, memory.WindowLayoutCounts.Moves);
        var whole = memory.Resolve(StraddleAddress, StraddleSize);
        Assert.True(memory.IsInPlace(whole));
        Assert.Equal(whole.Buffer, memory.Resolve(StraddleAddress, half).Buffer);
        Assert.Equal(whole.Buffer, memory.Resolve(StraddleAddress + half, half).Buffer);

        // The windows the move replaced are released once the submission that replaced them completes.
        var released = memory.WindowCounts.Released;
        memory.NoteCompleted(2);
        Assert.Equal(released + 2, memory.WindowCounts.Released);
    }

    // A written binding across the edge stores into guest memory on both sides of it.
    [Fact]
    public void AStraddlingWrittenBindingStoresIntoGuestMemory()
    {
        if (!RequireDevice())
        {
            return;
        }

        using var backing = new Backing();
        using var harness = new MetalTransferHarness(Device.Value);
        using var memory = Memory(backing);

        harness.Begin();
        var straddle = memory.Resolve(StraddleAddress, StraddleSize);
        var write = MetalNative.Send(harness.CommandBuffer, SelBlit);
        Fill(write, straddle, 0xCD);
        MetalNative.SendVoid(write, SelEnd);
        harness.Submit();

        var bytes = new byte[StraddleSize];
        Assert.True(backing.TryReadBacking(StraddleAddress, bytes));
        Assert.All(bytes, value => Assert.Equal((byte)0xCD, value));
        Assert.Equal(0UL, memory.RingFallbacks);
    }

    // What a binding names is the window it lies in and nothing else: a range inside one
    // granule names that granule's window, one across an edge the window the edge moved to
    // make (the granule before it, grown by what the range takes of the next), and the
    // submission's named bytes are those windows, not the backing.
    [Fact]
    public void ABindingNamesOnlyTheWindowItLiesIn()
    {
        if (!RequireDevice())
        {
            return;
        }

        using var backing = new Backing();
        using var memory = Memory(backing);
        memory.BeginSubmission(1, 0);

        var inside = memory.Resolve(GuestBase + (3 * GranuleBytes) + 0x1000, 0x2000);
        Assert.True(memory.IsInPlace(inside));
        Assert.Equal(0x1000UL, inside.Offset);
        AssertNamed(memory, [3]);

        var straddle = memory.Resolve(StraddleAddress, StraddleSize);
        Assert.Equal(GranuleBytes - (32 * 1024), straddle.Offset);
        AssertNamed(memory, [0, 1, 3]);
        Assert.Equal(2, memory.TouchedWindows.Count);
        Assert.Equal(straddle.Buffer, memory.Resolve(StraddleAddress + 0x100, 0x100 + (32 * 1024)).Buffer);
        Assert.Equal(GranuleBytes + (32 * 1024) + GranuleBytes, memory.WindowSetBytes(memory.TouchedWindows));

        // A fresh submission names nothing until it resolves something.
        memory.BeginSubmission(2, 1);
        Assert.True(memory.TouchedWindows.IsEmpty);
        Assert.Equal(0UL, memory.RingFallbacks);
        output.WriteLine($"windows alive: {memory.WindowCounts.Alive}");
    }

    // Windows nothing used for the idle count are cut back to one per granule once the GPU
    // has completed past them, and the buffer of the window the edge had moved to make is
    // released; a later range across the same edge moves it again.
    [Fact]
    public void IdleWindowsAreCutBackAndTheEdgeMovesAgainOnDemand()
    {
        if (!RequireDevice())
        {
            return;
        }

        using var backing = new Backing();
        using var memory = Memory(backing);
        memory.BeginSubmission(1, 0);
        memory.Resolve(StraddleAddress, StraddleSize);
        memory.Resolve(GuestBase + 0x100, 0x100);
        // One window holds both: granule 0 grown by the 32 KiB the range takes of granule 1.
        Assert.Equal(1, memory.WindowCounts.Alive);
        var moves = memory.WindowLayoutCounts.Moves;

        var later = MetalGuestMemory.IdleWindowSerials * 2;
        memory.BeginSubmission(later, later - 1);
        memory.NoteCompleted(later);
        var counts = memory.WindowCounts;
        Assert.Equal(1L, counts.Released);
        Assert.Equal(0, counts.Alive);

        var again = memory.Resolve(StraddleAddress, StraddleSize);
        Assert.True(memory.IsInPlace(again));
        Assert.Equal(1, memory.WindowCounts.Alive);
        Assert.Equal(moves + 1, memory.WindowLayoutCounts.Moves);
    }

    // A range larger than the largest window binds the largest window's worth of its prefix
    // in place: the window's bytes are what the GPU writes and reads, and nothing goes
    // through the ring. The upload cap is small here so a range inside the backing can
    // exceed it.
    [Fact]
    public void AnOversizedRangeBindsTheLargestWindowPrefixInPlace()
    {
        if (!RequireDevice())
        {
            return;
        }

        using var backing = new Backing();
        using var harness = new MetalTransferHarness(Device.Value);
        using var memory = Memory(backing, maxUploadBytes: 64 * 1024);

        // Starts half-way into granule 0 and runs three granules: no window may hold more
        // than two granules, so the prefix is two granules long.
        var address = GuestBase + (GranuleBytes / 2);
        var size = 3 * GranuleBytes;
        var covered = MaxWindowBytes;

        harness.Begin();
        var prefix = memory.Resolve(address, size);
        Assert.True(prefix.IsValid);
        Assert.Equal(covered, prefix.Size);
        Assert.Equal(1UL, memory.OversizedInPlace);
        Assert.Equal(0UL, memory.RingFallbacks);
        var write = MetalNative.Send(harness.CommandBuffer, SelBlit);
        Fill(write, prefix, 0xCD);
        MetalNative.SendVoid(write, SelEnd);
        harness.Submit();

        var bytes = new byte[covered];
        Assert.True(backing.TryReadBacking(address, bytes));
        Assert.All(bytes, value => Assert.Equal((byte)0xCD, value));
        var beyond = new byte[16];
        Assert.True(backing.TryReadBacking(address + covered, beyond));
        Assert.All(beyond, value => Assert.Equal((byte)0, value));
    }

    // A storage binding larger than the largest window and too large to stage binds the
    // prefix in place: the host treats it as guest memory, not as a ring copy to gather into
    // and write back, and the shader's bound is the prefix, so no access reaches past the
    // window into memory the binding does not mean.
    [Fact]
    public void AnOversizedStorageBindingIsGuestMemoryBoundedByItsPrefix()
    {
        if (!RequireDevice())
        {
            return;
        }

        using var backing = new Backing();
        using var memory = Memory(backing, maxUploadBytes: 64 * 1024);

        var address = GuestBase + GranuleBytes - (GranuleBytes / 4);
        var size = MaxWindowBytes + (GranuleBytes / 2);
        memory.BeginSubmission(1, 0);
        var staged = memory.Stage(address, size);

        Assert.True(staged.IsValid);
        Assert.True(memory.IsInPlace(staged));
        Assert.True(staged.Size < size);
        Assert.Equal((uint)staged.Size, MetalRenderHost.BoundBufferBytes(size, staged));
        Assert.Equal(0UL, memory.RingFallbacks);

        var ring = memory.Allocate(256, 256, out _);
        Assert.False(memory.IsInPlace(ring));
        Assert.Equal(256u, MetalRenderHost.BoundBufferBytes(size, ring));
        Assert.False(memory.IsInPlace(default));
    }

    private static void AssertNamed(MetalGuestMemory memory, int[] granules)
    {
        var covered = new MetalWindowSet();
        memory.AddCoveredGranules(memory.TouchedWindows, covered);
        Assert.Equal(granules, covered.ToArray());
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
            output.WriteLine("Guest window granules run on the device only behind the exclusive device gate.");
            return false;
        }

        Assert.True(OperatingSystem.IsMacOS());
        Assert.NotEqual(0, Device.Value);
        return true;
    }

    private static readonly Lazy<nint> Device = new(MetalNative.MTLCreateSystemDefaultDevice);
}

internal static class MetalWindowSetTestExtensions
{
    public static int[] ToArray(this MetalWindowSet set)
    {
        var ids = new List<int>();
        foreach (var id in set)
        {
            ids.Add(id);
        }

        return ids.ToArray();
    }
}
