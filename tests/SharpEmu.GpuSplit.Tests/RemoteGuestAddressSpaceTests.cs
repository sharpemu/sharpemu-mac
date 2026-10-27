// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.GpuRemote;
using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.HLE.GuestMemory;
using SharpEmu.HLE.Host;
using Xunit;

namespace SharpEmu.GpuSplit.Tests;

[Collection(SplitCollection.Name)]
public sealed class SplitSharedStateTests
{
    [MacOSFact]
    public void TheSummaryAnswersForEveryPageARangeTouches()
    {
        using var channel = GpuRemoteChannel.Create(SharedMemoryRegion.CreateUniqueName("sesplit"), SplitSharedState.Layout);
        var state = new SplitSharedState(channel);
        const ulong Page = 1UL << SplitSharedState.PageShift;
        Assert.Equal(TrackerLayout.PageBytes, Page);
        Assert.False(state.MayInterest(0, TrackerLayout.SpaceBytes));

        state.SetPage(0x40_0000 + 3 * Page + 0x123);
        Assert.True(state.MayInterest(0x40_0000 + 3 * Page, 8));
        Assert.True(state.MayInterest(0x40_0000 + 4 * Page - 1, 1));
        Assert.False(state.MayInterest(0x40_0000 + 4 * Page, Page));
        Assert.False(state.MayInterest(0x40_0000, 3 * Page));
        // Across words: 64 pages each.
        Assert.True(state.MayInterest(0x10_0000, 0x40_0000));
        Assert.True(state.MayInterest(0, TrackerLayout.SpaceBytes));
        Assert.False(state.MayInterest(TrackerLayout.SpaceBytes, 8));
        Assert.False(state.MayInterest(0x40_0000 + 3 * Page, 0));

        state.SetPage(TrackerLayout.SpaceBytes - 1);
        Assert.True(state.MayInterest(TrackerLayout.SpaceBytes - Page, ulong.MaxValue / 2));

        state.ClearPage(0x40_0000 + 3 * Page);
        Assert.False(state.MayInterest(0x40_0000, 0x40_0000));
    }

    [MacOSFact]
    public void AWriteToAnUnwatchedPageNextToAWatchedOneIsNotOfInterest()
    {
        using var channel = GpuRemoteChannel.Create(SharedMemoryRegion.CreateUniqueName("sesplit"), SplitSharedState.Layout);
        var state = new SplitSharedState(channel);
        const ulong Group = 0x10_0000;
        // One watched page in the middle of a 64 KiB group (the old summary's granularity).
        state.SetPage(Group + 5 * 0x1000);

        Assert.True(state.MayInterest(Group + 5 * 0x1000, 1));
        Assert.True(state.MayInterest(Group + 5 * 0x1000 + 0xFFF, 1));
        // The same 64 KiB group, every other page.
        for (ulong page = 0; page < 16; page++)
        {
            Assert.Equal(page == 5, state.MayInterest(Group + page * 0x1000, 8));
            Assert.Equal(page == 5, state.MayInterest(Group + page * 0x1000 + 0xFF8, 8));
        }

        // A write that straddles into the watched page is of interest, one that stops short is not.
        Assert.True(state.MayInterest(Group + 5 * 0x1000 - 4, 8));
        Assert.False(state.MayInterest(Group + 5 * 0x1000 - 8, 8));
        Assert.False(state.MayInterest(Group + 6 * 0x1000, 0x1000));
        Assert.False(state.MayInterest(Group, 5 * 0x1000));

        state.ClearPage(Group + 5 * 0x1000);
        Assert.False(state.MayInterest(Group, 16 * 0x1000));
    }

    [MacOSFact]
    public void EveryPageIsPendingUntilTheGpuSideKeepsTheSummary()
    {
        using var channel = GpuRemoteChannel.Create(SharedMemoryRegion.CreateUniqueName("sesplit"), SplitSharedState.Layout);
        var state = new SplitSharedState(channel);
        Assert.True(state.MayPendWriteback(0x10_0000, 8));

        var pending = new PendingWritebackPages(state);
        Assert.True(state.MayPendWriteback(0x10_0000, 8));
        pending.Activate();
        Assert.True(state.PendingLive);
        Assert.False(state.MayPendWriteback(0x10_0000, 8));
        Assert.False(state.MayPendWriteback(0, TrackerLayout.SpaceBytes));
    }

    [MacOSFact]
    public void APendingPageIsMarkedAtOnceAndUnmarkedOnlyWhenPublishedUnclaimed()
    {
        using var channel = GpuRemoteChannel.Create(SharedMemoryRegion.CreateUniqueName("sesplit"), SplitSharedState.Layout);
        var state = new SplitSharedState(channel);
        var pending = new PendingWritebackPages(state);
        pending.Activate();
        const ulong Image = 0x4_0000_0000 + 0x3F000; // straddles a summary word: pages 63 and 64 of the block
        const ulong Page = 0x1000;

        // An image of two pages and a writeback over its second page and the next.
        pending.Add(Image + 0x10, 2 * Page - 0x20);
        Assert.True(state.MayPendWriteback(Image, 1));
        Assert.True(state.MayPendWriteback(Image + Page + 0xFF8, 8));
        Assert.False(state.MayPendWriteback(Image - 8, 8));
        Assert.False(state.MayPendWriteback(Image + 2 * Page, 8));
        pending.Add(Image + Page, 2 * Page);
        Assert.Equal(2, pending.ClaimsAt(Image + Page));

        // The image's claim ends: nothing changes until published, and then only its own page goes.
        pending.Remove(Image + 0x10, 2 * Page - 0x20);
        Assert.True(state.MayPendWriteback(Image, 8));
        pending.Publish();
        Assert.False(state.MayPendWriteback(Image, Page));
        Assert.True(state.MayPendWriteback(Image + Page, 8));
        Assert.True(state.MayPendWriteback(Image + 2 * Page, 8));

        // Claimed again before the publish: the page stays marked.
        pending.Remove(Image + Page, 2 * Page);
        pending.Add(Image + 2 * Page, 8);
        pending.Publish();
        Assert.False(state.MayPendWriteback(Image, 2 * Page));
        Assert.True(state.MayPendWriteback(Image + 2 * Page, 1));
        pending.Remove(Image + 2 * Page, 8);
        pending.Publish();
        Assert.False(state.MayPendWriteback(Image - Page, 8 * Page));

        // A release of what was never claimed changes nothing.
        pending.Remove(Image, Page);
        pending.Publish();
        Assert.Equal(0, pending.ClaimsAt(Image));
    }

    [MacOSFact]
    public void AProcessWithoutTheBlockIsRefused()
    {
        using var channel = GpuRemoteChannel.Create(SharedMemoryRegion.CreateUniqueName("sesplit"));
        Assert.Throws<InvalidDataException>(() => new SplitSharedState(channel));
    }
}

/// <summary>The GPU process's guest address space over a real shareable backing, mirror and replica.</summary>
[Collection(SplitCollection.Name)]
public sealed unsafe class RemoteGuestAddressSpaceTests : IDisposable
{
    private const GuestPageProtection ReadWrite = GuestPageProtection.Read | GuestPageProtection.Write;
    private const ulong Marker = 0x5350_4C49_5447_5055;

    private readonly SharedGuestMemory _guest = new();
    private readonly HostBackingObject _mirror;
    private readonly HostSharedRegion _table;
    private readonly GuestReplica _replica;
    private readonly RecordingProtections _protections = new();
    private readonly RecordingSummary _summary = new();
    private readonly RemoteGuestAddressSpace _space;

    public RemoteGuestAddressSpaceTests()
    {
        (_mirror, _table, _replica) = _guest.MapLikeTheGpuProcess();
        _space = new RemoteGuestAddressSpace(
            _mirror, new GuestMirrorTranslator(_table, _mirror), _replica, _guest.Export.ChunkBytes, _guest.Export.ObjectBytes,
            _protections, _summary);
    }

    public void Dispose()
    {
        _replica.Dispose();
        _table.Dispose();
        _mirror.Dispose();
        _guest.Dispose();
    }

    [MacOSFact]
    public void BytesCrossBothWaysThroughTheMirror()
    {
        *_guest.Pointer(SharedGuestMemory.Chunk - 8) = Marker;
        *_guest.Pointer(SharedGuestMemory.Chunk) = ~Marker;
        Span<byte> read = stackalloc byte[16];
        Assert.True(_space.TryRead(_guest.Address + SharedGuestMemory.Chunk - 8, read));
        Assert.Equal(Marker, BitConverter.ToUInt64(read));
        Assert.Equal(~Marker, BitConverter.ToUInt64(read[8..]));
        Assert.True(_space.TryCompare(_guest.Address + SharedGuestMemory.Chunk - 8, read, out var equal));
        Assert.True(equal);

        Assert.True(_space.TryWrite(_guest.Address + 3 * SharedGuestMemory.Chunk, BitConverter.GetBytes(Marker ^ 0x55)));
        Assert.Equal(Marker ^ 0x55, *_guest.Pointer(3 * SharedGuestMemory.Chunk));
        Assert.True(_space.CanRead(_guest.Address, SharedGuestMemory.ViewBytes));
        Assert.True(_space.IsBackedRange(_guest.Address, SharedGuestMemory.ViewBytes));
        Assert.True(_space.IsBackedView(_guest.Address));
    }

    [MacOSFact]
    public void MemoryTheBackingDoesNotHoldFailsAndIsCounted()
    {
        Span<byte> read = stackalloc byte[8];
        Assert.False(_space.TryRead(_guest.Address + SharedGuestMemory.ViewBytes, read));
        Assert.False(_space.TryRead(_guest.Address + SharedGuestMemory.ViewBytes - 4, read));
        Assert.False(_space.TryWrite(0x1000, read));
        Assert.False(_space.CanRead(_guest.Address, SharedGuestMemory.ViewBytes + 1));
        Assert.Equal(3, _space.PrivateMemoryMisses);
        Assert.Contains("not in the shared backing", _space.DescribeReadRange(0x1000, 8));
    }

    [MacOSFact]
    public void AliasesAreMirrorAddressesAndHostViewsReplicaAddresses()
    {
        Assert.True(_space.TryResolveBackingAlias(_guest.Address + 0x40, 0x100, out var alias));
        Assert.Equal(_mirror.AliasBase + SharedGuestMemory.ViewOffset + 0x40, alias);
        Assert.Equal((_mirror.AliasBase, _space.BackingAliasRange.Size), _space.BackingAliasRange);
        Assert.False(_space.TryResolveBackingAlias(_guest.Address + SharedGuestMemory.ViewBytes - 8, 16, out _));

        *_guest.Pointer(0x80) = Marker;
        Assert.True(_space.TryResolveBackedHostView(_guest.Address, SharedGuestMemory.ViewBytes, out var host));
        Assert.Equal(_replica.HostAddress(_guest.Address), host);
        Assert.Equal(Marker, *(ulong*)(host + 0x80));
    }

    [MacOSFact]
    public void AHostViewSyncsTheReplicaToViewsMappedAfterIt()
    {
        // A second view the replica has not replayed yet: the table already has it.
        var second = _guest.Address + SharedGuestMemory.ViewBytes;
        var hole = _guest.Host.ReserveHole(second, SharedGuestMemory.Chunk);
        if (hole != second)
        {
            return;
        }

        try
        {
            Assert.True(_guest.Views.TryMapReservedRange(second, SharedGuestMemory.Chunk, 40 * SharedGuestMemory.Chunk,
                HostPageProtection.ReadWrite, out _));
            *(ulong*)second = Marker;
            var before = _replica.ReplayedSequence;
            Assert.True(_space.TryResolveBackedHostView(second, SharedGuestMemory.Chunk, out var host));
            Assert.True(_replica.ReplayedSequence > before);
            Assert.Equal(Marker, *(ulong*)host);
        }
        finally
        {
            _ = _guest.Views.Unmap(second, SharedGuestMemory.Chunk, out _);
            _ = _guest.Host.FreeHole(second, SharedGuestMemory.Chunk);
        }
    }

    [MacOSFact]
    public void ProtectionsGoToTheGuestAndOnlyTakingRightsWaits()
    {
        Assert.Equal(TrackerLayout.PageBytes, _space.ProtectionPageSize);
        Assert.True(_space.TryApplyDerivedProtection(_guest.Address, 0x2000, ReadWrite, GuestPageProtection.Read, ordinaryEnrollment: true));
        Assert.True(_space.TryApplyDerivedProtection(_guest.Address, 0x2000, ReadWrite, ReadWrite, ordinaryEnrollment: false));
        Assert.False(_space.TryApplyDerivedProtection(_guest.Address, 0x2000, GuestPageProtection.Read, ReadWrite, ordinaryEnrollment: false));
        Assert.Equal(
            new[]
            {
                (_guest.Address, 0x2000UL, ReadWrite, GuestPageProtection.Read, ProtectRequestFlags.Derived | ProtectRequestFlags.OrdinaryEnrollment),
                (_guest.Address, 0x2000UL, ReadWrite, ReadWrite, ProtectRequestFlags.Derived),
            },
            _protections.Requests);
        Assert.Equal(new ulong[] { 1 }, _protections.Waits);
    }

    [MacOSFact]
    public void WatchesOfTheGpuMemoryManagerKeepTheInterestSummary()
    {
        using var memory = new GuestGpuMemory(_space);
        memory.Register(_guest.Address, SharedGuestMemory.ViewBytes, ReadWrite);
        _protections.Requests.Clear();

        memory.Pages.AddWatch(_guest.Address, 0x1000, blockReads: false);
        memory.Pages.AddWatch(_guest.Address + 0x1000, 0x1000, blockReads: false);
        memory.Pages.AddWatch(_guest.Address + SharedGuestMemory.Chunk, 0x1000, blockReads: false);
        // One bit per watched page, not per 64 KiB group around it.
        Assert.Equal(new[] { _guest.Address, _guest.Address + 0x1000, _guest.Address + SharedGuestMemory.Chunk }, _summary.Set);
        // Each watch takes write access from the guest, and waits for the guest to apply it.
        Assert.Equal(3, _protections.Requests.Count(request => request.Access == GuestPageProtection.Read));
        Assert.Equal(3, _protections.Waits.Count);

        // A second watcher of a page keeps its bit until the last one goes.
        memory.Pages.AddWatch(_guest.Address, 0x1000, blockReads: false);
        memory.Pages.RemoveWatch(_guest.Address, 0x1000, blockReads: false);
        Assert.Empty(_summary.Cleared);
        memory.Pages.RemoveWatch(_guest.Address, 0x1000, blockReads: false);
        memory.Pages.RemoveWatch(_guest.Address + 0x1000, 0x1000, blockReads: false);
        memory.Pages.RemoveWatch(_guest.Address + SharedGuestMemory.Chunk, 0x1000, blockReads: false);
        Assert.Equal(new[] { _guest.Address, _guest.Address + 0x1000, _guest.Address + SharedGuestMemory.Chunk }, _summary.Cleared);
        Assert.Equal(3, _protections.Waits.Count);
    }

    [MacOSFact]
    public void ProtectionsOfOneWatchChangeAreConfirmedOnceAtItsEnd()
    {
        _space.BeginProtectionBatch();
        _space.BeginProtectionBatch();
        Assert.True(_space.TryApplyDerivedProtection(_guest.Address, 0x1000, ReadWrite, GuestPageProtection.Read, ordinaryEnrollment: false));
        Assert.True(_space.TryApplyDerivedProtection(_guest.Address + 0x4000, 0x1000, ReadWrite, GuestPageProtection.Read, ordinaryEnrollment: false));
        // Giving rights back never waits, batched or not.
        Assert.True(_space.TryApplyDerivedProtection(_guest.Address + 0x8000, 0x1000, ReadWrite, ReadWrite, ordinaryEnrollment: false));
        _space.EndProtectionBatch();
        Assert.Empty(_protections.Waits);
        Assert.Equal(3, _protections.Requests.Count);
        _space.EndProtectionBatch();
        // One wait for the last of them: the guest process applies them in order.
        Assert.Equal(new ulong[] { 2 }, _protections.Waits);

        // Nothing stays pending for the next change.
        _space.BeginProtectionBatch();
        _space.EndProtectionBatch();
        Assert.Single(_protections.Waits);

        // Outside a batch every protection waits at once.
        Assert.True(_space.TryApplyDerivedProtection(_guest.Address, 0x1000, ReadWrite, GuestPageProtection.Read, ordinaryEnrollment: false));
        Assert.Equal(new ulong[] { 2, 4 }, _protections.Waits);
    }

    [MacOSFact]
    public void AWatchChangeThatSentNothingStillWaitsForAnotherThreadsProtection()
    {
        using var sent = new ManualResetEventSlim(false);
        using var finish = new ManualResetEventSlim(false);
        var other = new Thread(() =>
        {
            _space.BeginProtectionBatch();
            _ = _space.TryApplyDerivedProtection(_guest.Address, 0x1000, ReadWrite, GuestPageProtection.Read, ordinaryEnrollment: false);
            sent.Set();
            finish.Wait(10_000);
            _space.EndProtectionBatch();
        });
        other.Start();
        Assert.True(sent.Wait(10_000));

        // This thread's watch found the page already watched by the other, so it sent nothing; the
        // other's protection may not be applied yet, and this thread must not return before it is.
        _space.BeginProtectionBatch();
        _space.EndProtectionBatch();
        Assert.Equal(new ulong[] { 1 }, _protections.Waits);

        finish.Set();
        other.Join();
    }

    [MacOSFact]
    public void ASparseWatchMaskWaitsOnceForAllItsRuns()
    {
        using var memory = new GuestGpuMemory(_space);
        memory.Register(_guest.Address, SharedGuestMemory.ViewBytes, ReadWrite);
        _protections.Requests.Clear();
        _protections.Waits.Clear();

        var blockBase = _guest.Address & ~(TrackerLayout.BlockBytes - 1);
        var first = (int)((_guest.Address - blockBase) / TrackerLayout.PageBytes);
        var mask = new PageMask();
        mask.Set(first);
        mask.Set(first + 2);
        mask.Set(first + 4);
        memory.Pages.AddWatchMask(blockBase, mask, blockReads: false);

        // Three runs, three requests, one wait for the last.
        Assert.Equal(3, _protections.Requests.Count(request => request.Access == GuestPageProtection.Read));
        Assert.Equal(new[] { _protections.LastRequested }, _protections.Waits);

        _protections.Waits.Clear();
        memory.Pages.RemoveWatchMask(blockBase, mask, blockReads: false);
        Assert.Empty(_protections.Waits);
    }

    [MacOSFact]
    public void TheGpuSideNeverAllocatesOrMaps()
    {
        Assert.Throws<NotSupportedException>(() => _space.AllocateAt(0x1000, 0x1000));
        Assert.False(_space.TryAllocateGuestMemory(0x1000, 0x1000, out _));
        Assert.False(_space.TryMapBacked(_guest.Address, 0x1000, 0, ReadWrite, out _));
        Assert.False(_space.TryUnmapBacked(_guest.Address, 0x1000));
        Assert.Null(((IGuestBackedSpace)_space).SingleCopy);
    }

    private sealed class RecordingProtections : IProtectionRequests
    {
        private ulong _last;

        public List<(ulong Address, ulong Size, GuestPageProtection Guest, GuestPageProtection Access, ProtectRequestFlags Flags)> Requests { get; } = new();

        public List<ulong> Waits { get; } = new();

        public ulong LastRequested => _last;

        public ulong Request(ulong address, ulong size, GuestPageProtection guestAccess, GuestPageProtection access, ProtectRequestFlags flags)
        {
            Requests.Add((address, size, guestAccess, access, flags));
            return ++_last;
        }

        public bool WaitApplied(ulong sequence)
        {
            Waits.Add(sequence);
            return true;
        }
    }

    private sealed class RecordingSummary : IInterestSummary
    {
        public List<ulong> Set { get; } = new();

        public List<ulong> Cleared { get; } = new();

        public void SetPage(ulong address) => Set.Add(address);

        public void ClearPage(ulong address) => Cleared.Add(address);
    }
}
