// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using SharpEmu.Gpu.Metal.Tests.Scenes;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.HLE.Host;
using SharpEmu.Libs.Gpu.Rendering;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// Guest memory mapped a 64 KiB page at a time from scattered backing offsets, the way
/// Demon's Souls BatchMaps it: a range is contiguous to the guest and the host but not in
/// the backing, so no window holds it. A read binds a guest view, a no-copy buffer over the
/// guest mapping, whole and without a copy; a write is staged whole, however far past the
/// ring's 64 MiB cap, and its stores reach guest memory; a read behind a write recorded in
/// the same command buffer is staged with that write gathered in, since a view is another
/// GPU address for the bytes and would not see it. The backing here is a real shared-memory
/// object mapped twice, once whole (the canonical alias) and once scattered (the guest).
/// </summary>
public sealed unsafe class MetalGuestViewTests(ITestOutputHelper output)
{
    private const ulong Mib = 1024 * 1024;
    private const ulong BackingBytes = 160 * Mib;
    private const ulong ViewBytes = 96 * Mib;
    private const ulong GranuleBytes = 4 * Mib;
    private const ulong MaxWindowBytes = 128 * Mib;

    private static readonly nint SelBlit = MetalNative.Selector("blitCommandEncoder");
    private static readonly nint SelEnd = MetalNative.Selector("endEncoding");
    private static readonly nint SelNewFence = MetalNative.Selector("newFence");
    private static readonly nint SelUpdateFence = MetalNative.Selector("updateFence:");
    private static readonly nint SelWaitForFence = MetalNative.Selector("waitForFence:");
    private static readonly nint SelRelease = MetalNative.Selector("release");
    private static readonly nint SelCopy =
        MetalNative.Selector("copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:");

    // A read of the whole scattered range is one guest view: the bytes the GPU reads are the
    // guest's, page for page, nothing goes through the ring, and only the submission that
    // bound the view names it.
    [Fact]
    public void AScatteredRangePastTheRingCapIsReadWholeThroughAGuestView()
    {
        if (!RequireDevice())
        {
            return;
        }

        using var backing = new ShmScatteredBacking(BackingBytes, ViewBytes);
        backing.FillAliasPattern(0x9E3779B1u);
        using var harness = new MetalTransferHarness(Device.Value);
        using var memory = Memory(backing);

        harness.Begin();
        memory.BeginSubmission(1, 0);
        Assert.False(memory.TryResolveInPlace(backing.Base, ViewBytes, out _));
        Assert.True(memory.MayUseGuestView(ViewBytes));
        Assert.True(memory.TryResolveGuestView(backing.Base, ViewBytes, out var view));
        Assert.Equal(ViewBytes, view.Size);
        Assert.Contains(view.Buffer, memory.ResidentBuffers.ToArray());

        var copy = harness.Buffer(ViewBytes);
        var blit = MetalNative.Send(harness.CommandBuffer, SelBlit);
        MetalNative.SendCopyBuffer(blit, SelCopy, view.Buffer, (nuint)view.Offset, copy, 0, (nuint)ViewBytes);
        MetalNative.SendVoid(blit, SelEnd);
        harness.Submit();

        AssertGuestBytes(backing, backing.Base, harness.Read(copy, 0, ViewBytes));
        Assert.Equal(0UL, memory.RingFallbacks);
        Assert.Equal(0UL, memory.OversizedStaged);

        // A smaller binding inside it reuses the view; the next submission names it only if it binds it.
        memory.BeginSubmission(2, 1);
        Assert.DoesNotContain(view.Buffer, memory.ResidentBuffers.ToArray());
        Assert.True(memory.TryResolveGuestView(backing.Base + 3 * ShmScatteredBacking.Piece, 8 * Mib, out var inner));
        Assert.Equal(view.Buffer, inner.Buffer);
        Assert.Equal(3 * ShmScatteredBacking.Piece, inner.Offset);
        Assert.Equal(1L, memory.GuestViewCounts.Created);
    }

    // A written binding of the same range is staged whole - not its first 64 MiB - and what
    // the GPU stores into the copy lands in every scattered page of guest memory.
    [Fact]
    public void AScatteredRangePastTheRingCapIsStagedWholeAndItsStoresReachGuestMemory()
    {
        if (!RequireDevice())
        {
            return;
        }

        using var backing = new ShmScatteredBacking(BackingBytes, ViewBytes);
        backing.FillAliasPattern(0x9E3779B1u);
        using var harness = new MetalTransferHarness(Device.Value);
        using var memory = Memory(backing);
        var staged = new MetalStagedStorage(memory);

        harness.Begin();
        memory.BeginSubmission(1, 0);
        var copy = memory.Stage(backing.Base, ViewBytes);
        Assert.True(copy.IsValid);
        Assert.Equal(ViewBytes, copy.Size);
        Assert.False(memory.IsInPlace(copy));
        Assert.Equal(1UL, memory.OversizedStaged);
        Assert.Equal((uint)ViewBytes, MetalRenderHost.BoundBufferBytes(ViewBytes, copy));
        staged.Add(backing.Base, ViewBytes, copy, memory.Duplicate(copy), written: true);

        // The shader's stores: a pattern over every byte of the copy.
        var stores = MetalTileOracle.Pattern((int)ViewBytes, 0xC0FFEE);
        var source = harness.Buffer(stores);
        var fence = MetalNative.Send(Device.Value, SelNewFence);
        var store = MetalNative.Send(harness.CommandBuffer, SelBlit);
        MetalNative.SendCopyBuffer(store, SelCopy, source, 0, copy.Buffer, (nuint)copy.Offset, (nuint)ViewBytes);
        MetalNative.SendVoid(store, SelUpdateFence, fence);
        MetalNative.SendVoid(store, SelEnd);
        staged.Arm();
        var wait = MetalNative.Send(harness.CommandBuffer, SelBlit);
        MetalNative.SendVoid(wait, SelWaitForFence, fence);
        MetalNative.SendVoid(wait, SelEnd);
        Assert.Equal([(backing.Base, ViewBytes)], staged.EncodeWritebacks(harness.CommandBuffer));
        staged.MarkWrittenBack();
        harness.Submit();
        MetalNative.SendVoid(fence, SelRelease);

        var bytes = new byte[ViewBytes];
        Assert.True(backing.TryRead(backing.Base, bytes));
        AssertEqualBytes(stores, bytes);
    }

    // A large written range is staged by the GPU alone: the copy and its pristine twin are both
    // filled from the windows over the range's scattered runs, the CPU copies none of it, and
    // what the shader stores into the copy still reaches every scattered page. A range with a
    // page no window covers is left to the CPU.
    [Fact]
    public void AWrittenScatteredRangeIsStagedOnTheGpuAndItsStoresReachGuestMemory()
    {
        if (!RequireDevice())
        {
            return;
        }

        using var backing = new ShmScatteredBacking(BackingBytes, ViewBytes);
        backing.FillAliasPattern(0x2468ACE1u);
        using var harness = new MetalTransferHarness(Device.Value);
        using var memory = Memory(backing);
        var staged = new MetalStagedStorage(memory);

        harness.Begin();
        memory.BeginSubmission(1, 0);
        var copy = memory.ReserveWholeStaging(backing.Base, ViewBytes);
        var pristine = memory.Allocate(ViewBytes, 256, out _);
        Assert.True(copy.IsValid);
        Assert.Equal(ViewBytes, copy.Size);
        Assert.Equal(1UL, memory.OversizedStaged);
        Assert.True(staged.TryEncodeWholeGather(harness.CommandBuffer, backing.Base, copy, pristine));
        var tail = memory.Allocate(2 * ShmScatteredBacking.Piece, 256, out _);
        Assert.False(staged.TryEncodeWholeGather(
            harness.CommandBuffer, backing.Base + ViewBytes - ShmScatteredBacking.Piece, tail, tail));
        harness.Submit();
        AssertGuestBytes(backing, backing.Base, harness.Read(copy.Buffer, copy.Offset, ViewBytes));
        AssertGuestBytes(backing, backing.Base, harness.Read(pristine.Buffer, pristine.Offset, ViewBytes));

        harness.Begin();
        staged.Add(backing.Base, ViewBytes, copy, pristine, written: true);
        var stores = MetalTileOracle.Pattern((int)ViewBytes, 0xBEEF);
        var source = harness.Buffer(stores);
        var fence = MetalNative.Send(Device.Value, SelNewFence);
        var store = MetalNative.Send(harness.CommandBuffer, SelBlit);
        MetalNative.SendCopyBuffer(store, SelCopy, source, 0, copy.Buffer, (nuint)copy.Offset, (nuint)ViewBytes);
        MetalNative.SendVoid(store, SelUpdateFence, fence);
        MetalNative.SendVoid(store, SelEnd);
        staged.Arm();
        var wait = MetalNative.Send(harness.CommandBuffer, SelBlit);
        MetalNative.SendVoid(wait, SelWaitForFence, fence);
        MetalNative.SendVoid(wait, SelEnd);
        Assert.Equal([(backing.Base, ViewBytes)], staged.EncodeWritebacks(harness.CommandBuffer));
        staged.MarkWrittenBack();
        harness.Submit();
        MetalNative.SendVoid(fence, SelRelease);

        var bytes = new byte[ViewBytes];
        Assert.True(backing.TryRead(backing.Base, bytes));
        AssertEqualBytes(stores, bytes);
    }

    // Inside one command buffer a guest view would not see a write made through a window,
    // so a read of bytes this command buffer already wrote is staged with the write gathered
    // in, and sees it. The next command buffer reads the same bytes through a view.
    [Fact]
    public void AReadBehindAWriteInTheSameCommandBufferIsStagedAndSeesIt()
    {
        if (!RequireDevice())
        {
            return;
        }

        using var backing = new ShmScatteredBacking(BackingBytes, ViewBytes);
        backing.FillAliasPattern(0x9E3779B1u);
        using var rig = new HostRig(Device.Value, backing);
        var host = rig.Host;
        const ulong piece = ShmScatteredBacking.Piece;
        var address = backing.Base + (piece / 2);
        const ulong size = 1 * Mib;

        // One page of the range is written on the GPU, from a page outside it.
        var source = backing.Base + ViewBytes - piece;
        backing.TryWrite(source, Enumerable.Repeat((byte)0xAB, (int)piece).ToArray());
        var expected = new byte[size];
        Assert.True(backing.TryRead(address, expected));
        Array.Fill(expected, (byte)0xAB, (int)(piece / 2), (int)piece);
        var serial = host.SubmissionSerial;
        host.CopyGuestMemory(backing.Base + piece, source, piece, allowHost: false);

        // The GPU reads the binding in the same command buffer, after the write.
        using var harness = new MetalTransferHarness(Device.Value);
        var behind = host.ObtainBuffer(address, size, isWritten: false);
        Assert.False(host.GuestMemory.IsGuestView((nint)behind.Handle));
        var read = CopyOnHost(host, harness, behind, size);
        Assert.Equal(serial, host.SubmissionSerial);
        host.FlushAndWaitForGpu();
        AssertEqualBytes(expected, harness.Read(read, 0, size));

        var after = host.ObtainBuffer(address, size, isWritten: false);
        Assert.True(host.GuestMemory.IsGuestView((nint)after.Handle));
        read = CopyOnHost(host, harness, after, size);
        host.FlushAndWaitForGpu();
        AssertEqualBytes(expected, harness.Read(read, 0, size));
    }

    // A blit from a binding into a new shared buffer, recorded in the host's command buffer
    // the way its own transfers are.
    private static nint CopyOnHost(MetalRenderHost host, MetalTransferHarness harness, BufferBinding binding, ulong size)
    {
        var destination = harness.Buffer(size);
        var context = ((IMetalImageCacheOwner)host).BeginTransfer(MetalTransferCause.GuestCopy);
        var blit = MetalNative.Send(context.CommandBuffer, SelBlit);
        MetalNative.SendCopyBuffer(blit, SelCopy, (nint)binding.Handle, (nuint)binding.Offset, destination, 0, (nuint)size);
        MetalNative.SendVoid(blit, SelEnd);
        return destination;
    }

    // Scattered reads cost no staging: each binds a view, and neither the staged-range cache
    // nor the ring holds a byte of them.
    [Fact]
    public void ScatteredReadsLeaveTheStagedCacheAndTheRingEmpty()
    {
        if (!RequireDevice())
        {
            return;
        }

        using var backing = new ShmScatteredBacking(BackingBytes, ViewBytes);
        backing.FillAliasPattern(0x12345678u);
        using var rig = new HostRig(Device.Value, backing);
        var host = rig.Host;
        var stagedBefore = host.ImageCache.StagedRanges?.CacheBytes ?? 0;
        var ringBefore = host.GuestMemory.RingFallbackBytes;

        const ulong size = 4 * Mib;
        for (var offset = 0UL; offset + size <= ViewBytes; offset += size)
        {
            var binding = host.ObtainBuffer(backing.Base + offset + 4096, size - 8192, isWritten: false);
            Assert.True(host.GuestMemory.IsGuestView((nint)binding.Handle));
        }

        host.FlushAndWaitForGpu();
        Assert.Equal(stagedBefore, host.ImageCache.StagedRanges?.CacheBytes ?? 0);
        Assert.Equal(ringBefore, host.GuestMemory.RingFallbackBytes);
        Assert.True(host.GuestMemory.GuestViewCounts.Bytes <= ViewBytes);
    }

    // A view keeps the pages it was made over, so an unmap drops it, and a bind after the
    // guest maps other backing there reads the new pages through a new view.
    [Fact]
    public void AnUnmapDropsTheViewAndTheNextBindReadsTheNewMapping()
    {
        if (!RequireDevice())
        {
            return;
        }

        using var backing = new ShmScatteredBacking(BackingBytes, ViewBytes);
        backing.FillAliasPattern(0x9E3779B1u);
        using var harness = new MetalTransferHarness(Device.Value);
        using var memory = Memory(backing);
        const ulong size = 2 * Mib;

        memory.BeginSubmission(1, 0);
        Assert.True(memory.TryResolveGuestView(backing.Base, size, out var first));

        memory.OnGuestUnmap(backing.Base + ShmScatteredBacking.Piece, ShmScatteredBacking.Piece);
        backing.Remap(1, BackingBytes - ShmScatteredBacking.Piece);
        Assert.Equal(1L, memory.GuestViewCounts.Invalidated);

        Assert.True(memory.TryResolveGuestView(backing.Base, size, out var second));
        Assert.NotEqual(first.Buffer, second.Buffer);
        Assert.Equal(2L, memory.GuestViewCounts.Created);
        var expected = new byte[size];
        Assert.True(backing.TryRead(backing.Base, expected));
        Assert.Equal(expected, CopyOut(harness, second.Buffer, second.Offset, size));

        // The dropped view's buffer goes once the submission that bound it has completed.
        memory.NoteCompleted(1);
        Assert.Equal(1L, memory.GuestViewCounts.Released);
    }

    // The staged-range cache holds at most its budget, which is a quarter of what it was.
    [Fact]
    public void TheStagedRangeCacheStaysWithinItsBudget()
    {
        Assert.True(MetalStagedRangeCache.DefaultBudgetBytes <= 256 * Mib);
        var cache = new MetalStagedRangeCache(new CountingBackend(), MetalStagedRangeCache.DefaultBudgetBytes);
        var serial = 1UL;
        for (var address = 0x10_0000_0000UL; address < 0x10_0000_0000UL + 2048 * Mib; address += 64 * Mib)
        {
            cache.Acquire(address, MetalStagedRangeCache.MaxBytes, written: false, serial, serial - 1, out _);
            Assert.True(cache.CacheBytes <= MetalStagedRangeCache.DefaultBudgetBytes);
            serial++;
        }

        Assert.True(cache.TakeCounters().Evictions > 0);
    }

    private sealed class CountingBackend : IStagedRangeBackend
    {
        private nint _next = 0x1000;

        public (nint Handle, nint Contents, ulong GpuAddress) CreateBuffer(ulong capacity)
        {
            _next += 0x10;
            return (_next, 0, (ulong)_next);
        }

        public void ReleaseBuffer(nint handle)
        {
        }

        public void AddWatch(ulong address, ulong size)
        {
        }

        public void RemoveWatch(ulong address, ulong size)
        {
        }
    }

    private static MetalGuestMemory Memory(ShmScatteredBacking backing) =>
        new(Device.Value, backing, backing, GranuleBytes, MaxWindowBytes);

    private static byte[] CopyOut(MetalTransferHarness harness, nint buffer, ulong offset, ulong size)
    {
        harness.Begin();
        var copy = harness.Buffer(size);
        var blit = MetalNative.Send(harness.CommandBuffer, SelBlit);
        MetalNative.SendCopyBuffer(blit, SelCopy, buffer, (nuint)offset, copy, 0, (nuint)size);
        MetalNative.SendVoid(blit, SelEnd);
        harness.Submit();
        return harness.Read(copy, 0, size);
    }

    // What the guest reads at each page is the backing page it is mapped from.
    private static void AssertGuestBytes(ShmScatteredBacking backing, ulong address, byte[] actual)
    {
        var expected = new byte[actual.Length];
        Assert.True(backing.TryRead(address, expected));
        Assert.Equal(backing.AliasBytes(backing.BackingOffsetOf(address), 64), expected.AsSpan(0, 64).ToArray());
        AssertEqualBytes(expected, actual);
    }

    private static void AssertEqualBytes(byte[] expected, byte[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        var common = expected.AsSpan().CommonPrefixLength(actual);
        Assert.True(common == expected.Length, $"bytes differ first at 0x{common:X}");
    }

    private sealed class HostRig : IDisposable
    {
        private readonly nint _queue;
        private readonly PageGuard _pages = new(new SceneAddressSpace());

        public HostRig(nint device, ShmScatteredBacking backing)
        {
            _queue = MetalNative.Send(device, MetalNative.Selector("newCommandQueue"));
            Host = new MetalRenderHost(device, _queue, backing, backing, _pages, new MetalGpuWorkerRelay(static () => { }));
        }

        public MetalRenderHost Host { get; }

        public void Dispose()
        {
            Host.ImageCache.StopWatching();
            Host.Dispose();
            _pages.Dispose();
            MetalNative.SendVoid(_queue, MetalNative.Selector("release"));
        }
    }

    private bool RequireDevice()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Guest views run on the device only behind the exclusive device gate.");
            return false;
        }

        Assert.True(OperatingSystem.IsMacOS());
        Assert.NotEqual(0, Device.Value);
        return true;
    }

    private static readonly Lazy<nint> Device = new(MetalNative.MTLCreateSystemDefaultDevice);
}

/// <summary>
/// A shared-memory backing mapped whole once (the alias the windows slice) and once more as
/// guest memory, 64 KiB pages each from a scattered backing offset, at the host address the
/// guest sees it at, as backed views are.
/// </summary>
internal sealed unsafe class ShmScatteredBacking : IGuestBackedSpace, ICpuMemory, IDisposable
{
    public const ulong Piece = 64 * 1024;

    private const int ProtNone = 0;
    private const int ProtRead = 1;
    private const int ProtWrite = 2;
    private const int MapShared = 0x1;
    private const int MapPrivate = 0x2;
    private const int MapFixed = 0x10;
    private const int MapAnon = 0x1000;
    private const int ORdWr = 0x2;
    private const int OCreat = 0x200;
    private const int OExcl = 0x800;
    private static int s_count;

    private readonly int _descriptor;
    private readonly nint _alias;
    private readonly ulong _backingBytes;
    private readonly ulong _viewBytes;
    private readonly ulong[] _offsets;

    public ShmScatteredBacking(ulong backingBytes, ulong viewBytes)
    {
        _backingBytes = backingBytes;
        _viewBytes = viewBytes;
        var name = $"/se-gv-{Environment.ProcessId}-{Interlocked.Increment(ref s_count)}";
        _descriptor = shm_open(name, ORdWr | OCreat | OExcl, 0x180);
        if (_descriptor < 0)
        {
            throw new InvalidOperationException($"shm_open failed: errno={Marshal.GetLastPInvokeError()}");
        }

        shm_unlink(name);
        if (ftruncate(_descriptor, (long)backingBytes) != 0)
        {
            throw new InvalidOperationException($"ftruncate failed: errno={Marshal.GetLastPInvokeError()}");
        }

        _alias = mmap(0, (nuint)backingBytes, ProtRead | ProtWrite, MapShared, _descriptor, 0);
        Base = (ulong)mmap(0, (nuint)viewBytes, ProtNone, MapPrivate | MapAnon, -1, 0);
        if (_alias == -1 || Base == ulong.MaxValue)
        {
            throw new InvalidOperationException("mmap failed.");
        }

        var pieces = viewBytes / Piece;
        var backingPieces = backingBytes / Piece;
        _offsets = new ulong[pieces];
        for (var piece = 0UL; piece < pieces; piece++)
        {
            // 7919 is prime to the piece count, so no two pages share a backing page and no two
            // neighbours are neighbours in the backing.
            Remap((int)piece, (piece * 7919 + 13) % backingPieces * Piece);
        }
    }

    /// <summary>Where the guest sees the scattered pages; the host maps them at the same address.</summary>
    public ulong Base { get; }

    /// <summary>Maps one guest page from another backing offset, as a guest unmap and BatchMap would.</summary>
    public void Remap(int piece, ulong backingOffset)
    {
        var address = (nint)(Base + ((ulong)piece * Piece));
        if (mmap(address, (nuint)Piece, ProtRead | ProtWrite, MapShared | MapFixed, _descriptor, (long)backingOffset) != address)
        {
            throw new InvalidOperationException($"mmap of a guest page failed: errno={Marshal.GetLastPInvokeError()}");
        }

        _offsets[piece] = backingOffset;
    }

    /// <summary>Fills the backing through the alias with a word pattern that differs everywhere.</summary>
    public void FillAliasPattern(uint seed)
    {
        var words = new Span<uint>((void*)_alias, checked((int)(_backingBytes / 4)));
        for (var index = 0; index < words.Length; index++)
        {
            words[index] = (uint)index * seed + 0x5A5A5A5Au;
        }
    }

    public byte[] AliasBytes(ulong offset, int length) =>
        new ReadOnlySpan<byte>((byte*)_alias + offset, length).ToArray();

    public ulong BackingOffsetOf(ulong address) =>
        _offsets[(address - Base) / Piece] + ((address - Base) % Piece);

    public void Dispose()
    {
        munmap((nint)Base, (nuint)_viewBytes);
        munmap(_alias, (nuint)_backingBytes);
        close(_descriptor);
    }

    private bool Contains(ulong address, ulong size) =>
        size != 0 && address >= Base && address - Base <= _viewBytes && size <= _viewBytes - (address - Base);

    public bool TryHoldRange(ulong address, ulong size) => false;
    public bool TryHoldRangeAtOrAbove(ulong searchStart, ulong size, ulong alignment, out ulong address) { address = 0; return false; }
    public bool TryMapBacked(ulong address, ulong size, ulong backingOffset, GuestPageProtection protection, out HostViewFailure failure) { failure = HostViewFailure.None; return false; }
    public bool TryUnmapBacked(ulong address, ulong size) => false;
    public bool TryClearBacking(ulong offset, ulong size) => false;
    public bool IsBackedView(ulong address) => Contains(address, 1);
    public bool IsBackedRange(ulong address, ulong size) => Contains(address, size);
    public (ulong Base, ulong Size) BackingAliasRange => ((ulong)_alias, _backingBytes);

    // One alias address only for pages that are neighbours in the backing too.
    public bool TryResolveBackingAlias(ulong address, ulong size, out ulong alias)
    {
        alias = 0;
        if (!Contains(address, size))
        {
            return false;
        }

        var first = (address - Base) / Piece;
        var last = (address + size - 1 - Base) / Piece;
        for (var piece = first; piece < last; piece++)
        {
            if (_offsets[piece + 1] != _offsets[piece] + Piece)
            {
                return false;
            }
        }

        alias = (ulong)_alias + BackingOffsetOf(address);
        return true;
    }

    public bool TryResolveBackedHostView(ulong address, ulong size, out ulong host)
    {
        host = Contains(address, size) ? address : 0;
        return host != 0;
    }

    public bool TryWriteBacking(ulong address, ReadOnlySpan<byte> data) => TryWrite(address, data);

    public bool TryReadBacking(ulong address, Span<byte> data) => TryRead(address, data);

    public bool TryRead(ulong virtualAddress, Span<byte> destination)
    {
        if (!Contains(virtualAddress, (ulong)destination.Length))
        {
            return false;
        }

        new ReadOnlySpan<byte>((void*)virtualAddress, destination.Length).CopyTo(destination);
        return true;
    }

    public bool TryWrite(ulong virtualAddress, ReadOnlySpan<byte> source)
    {
        if (!Contains(virtualAddress, (ulong)source.Length))
        {
            return false;
        }

        source.CopyTo(new Span<byte>((void*)virtualAddress, source.Length));
        return true;
    }

    public bool CanRead(ulong address, ulong size) => Contains(address, size);

    [DllImport("libc", SetLastError = true)]
    private static extern nint mmap(nint address, nuint length, int protection, int flags, int descriptor, long offset);

    [DllImport("libc", SetLastError = true)]
    private static extern int munmap(nint address, nuint length);

    [DllImport("libc", SetLastError = true)]
    private static extern int ftruncate(int descriptor, long length);

    [DllImport("libc", SetLastError = true)]
    private static extern int close(int descriptor);

    [DllImport("libc", SetLastError = true)]
    private static extern int shm_open(string name, int flags, int mode);

    [DllImport("libc", SetLastError = true)]
    private static extern int shm_unlink(string name);
}
