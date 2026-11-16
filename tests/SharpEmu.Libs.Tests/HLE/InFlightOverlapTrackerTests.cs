// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE.GpuMemory;
using Xunit;

namespace SharpEmu.Libs.Tests.HLE;

/// <summary>
/// The in-flight overlap diagnostic without a page guard or thread: a sampled frame's buffers are armed as they
/// commit (reads for stores, writes for loads), a fault is attributed by byte and counted as the race only between
/// the buffer's publication and its completion, a page touched before the publication is armed again at it, a
/// submission's command buffer is watched until it is walked, and every watch is given back.
/// </summary>
public sealed class InFlightOverlapTrackerTests
{
    private const ulong Page = TrackerLayout.PageBytes;

    private sealed class FakeWatches : IFrameOverlapWatches
    {
        public readonly Dictionary<(ulong Page, bool Reads), int> Counts = new();

        public bool IsReadWatched(ulong page) => false;

        public void Add(ulong blockBase, in PageMask pages, bool blockReads) => Change(blockBase, pages, blockReads, 1);

        public void Remove(ulong blockBase, in PageMask pages, bool blockReads) => Change(blockBase, pages, blockReads, -1);

        public bool StillBlocks(ulong page, FaultKind kind) =>
            Counts.GetValueOrDefault((page, true)) != 0 || (kind != FaultKind.Read && Counts.GetValueOrDefault((page, false)) != 0);

        private void Change(ulong blockBase, PageMask pages, bool reads, int delta)
        {
            for (var index = 0; index < TrackerLayout.PagesPerBlock; index++)
            {
                if (!pages.Get(index))
                {
                    continue;
                }

                var key = (blockBase + (ulong)index * Page, reads);
                var count = Counts.GetValueOrDefault(key) + delta;
                Assert.True(count >= 0, $"watch count below zero at 0x{key.Item1:X}");
                Counts[key] = count;
            }
        }

        public int Live => Counts.Values.Sum();
    }

    private long _now = 1;

    private InFlightOverlapTracker Create(FakeWatches watches, int every = 1) => new(watches, () => _now, every, 1000, armReads: true, startThread: false);

    // Present at serial 1, then buffer 2 reads a constant range and has an occlusion result copied into guest memory.
    private static void RecordFrame(InFlightOverlapTracker tracker)
    {
        tracker.OnCommit(1, present: true);
        tracker.MarkRead(0x100000, 0x100, OverlapTag.Constant);
        tracker.NoteWrite(0x200000, 0x40, OverlapWrite.Copy, OverlapTag.Occlusion, 0);
        tracker.OnCommit(2, present: false);
        tracker.OnCommit(3, present: false);
    }

    // The next present and its gate close the sampled window.
    private static void CloseWindow(InFlightOverlapTracker tracker)
    {
        tracker.OnCommit(4, present: true);
        tracker.OnCommit(5, present: false);
        tracker.OnCommit(6, present: false);
        tracker.OnRelease(6);
        tracker.OnCompleted(6);
    }

    [Fact]
    public void TheGateArmsReleasedBuffersStillRunningReadsForStoresAndWritesForLoads()
    {
        var watches = new FakeWatches();
        var tracker = Create(watches);
        RecordFrame(tracker);

        tracker.OnRelease(3);

        Assert.Equal(1, watches.Counts.GetValueOrDefault((0x100000UL, false)));
        Assert.Equal(1, watches.Counts.GetValueOrDefault((0x200000UL, true)));
        Assert.Equal(2, watches.Live);

        _now += 1000;
        Assert.True(tracker.TryHandleFault(FaultKind.Write, 0x100010, out var blocked));
        Assert.False(blocked);
        Assert.True(tracker.TryHandleFault(FaultKind.Read, 0x200008, out _));
        Assert.Equal(0, watches.Live);

        tracker.OnCompleted(3);
        CloseWindow(tracker);
        var line = tracker.TakeWindowLine();
        Assert.NotNull(line);
        Assert.Contains("store_rs_hit_pages=1.0/1", line);
        Assert.Contains("by_tag=[constant:1.0]", line);
        Assert.Contains("by_era=[p+1:1.0]", line);
        Assert.Contains("load_ws_hit_pages=1.0/1", line);
        Assert.Contains("by_origin=[copy:1.0] by_tag=[occlusion:1.0]", line);
        Assert.Equal(0, watches.Live);
    }

    [Fact]
    public void ABufferIsArmedAtCommitAndCompletionGivesEveryWatchBack()
    {
        var watches = new FakeWatches();
        var tracker = Create(watches);
        RecordFrame(tracker);
        Assert.Equal(2, watches.Live);

        tracker.OnCompleted(2);

        Assert.Equal(0, watches.Live);
    }

    [Fact]
    public void AStoreBeforeThePublicationIsNotTheRaceAndThePageIsArmedAgainAtIt()
    {
        var watches = new FakeWatches();
        var tracker = Create(watches);
        RecordFrame(tracker);

        Assert.True(tracker.TryHandleFault(FaultKind.Write, 0x100010, out _));
        Assert.Equal(0, watches.Counts.GetValueOrDefault((0x100000UL, false)));
        tracker.OnRelease(3);
        Assert.Equal(1, watches.Counts.GetValueOrDefault((0x100000UL, false)));
        Assert.True(tracker.TryHandleFault(FaultKind.Write, 0x100020, out _));

        tracker.OnCompleted(3);
        CloseWindow(tracker);
        var line = tracker.TakeWindowLine()!;
        Assert.Contains("pre_release stores=1.0/1 by_tag=[constant:1.0]", line);
        Assert.Contains("rearmed=1.0/1", line);
        Assert.Contains("store_rs_hit_pages=1.0/1", line);
    }

    [Fact]
    public void AStoreBesideTheRangeOnItsPageIsANeighbourNotAHit()
    {
        var watches = new FakeWatches();
        var tracker = Create(watches);
        RecordFrame(tracker);
        tracker.OnRelease(3);

        Assert.True(tracker.TryHandleFault(FaultKind.Write, 0x100800, out _));
        tracker.OnCompleted(3);
        CloseWindow(tracker);

        var line = tracker.TakeWindowLine()!;
        Assert.Contains("store_rs_hit_pages=0.0/0", line);
        Assert.Contains("ranges=0.0/0 hit_range_mb=0.000 nb_pages=1.0/1", line);
    }

    [Fact]
    public void ACommandBufferAdmittedAndNotWalkedIsWatchedUntilItIsWalked()
    {
        var watches = new FakeWatches();
        var tracker = Create(watches);
        tracker.OnAdmission(0x300000, 2 * Page);
        RecordFrame(tracker);
        tracker.OnRelease(3);
        Assert.Equal(1, watches.Counts.GetValueOrDefault((0x300000UL, false)));
        Assert.Equal(1, watches.Counts.GetValueOrDefault((0x300000UL + Page, false)));

        Assert.True(tracker.TryHandleFault(FaultKind.Write, 0x300010, out _));
        tracker.OnSubmissionWalked(0x300000, 2 * Page);
        Assert.Equal(0, watches.Counts.GetValueOrDefault((0x300000UL + Page, false)));

        tracker.OnCompleted(3);
        Assert.Equal(0, watches.Live);
        CloseWindow(tracker);
        Assert.Contains("store_cb_hit_pages=1.0/1", tracker.TakeWindowLine()!);
    }

    [Fact]
    public void AnUnsampledFrameArmsNothing()
    {
        var watches = new FakeWatches();
        var tracker = Create(watches, every: 4);
        RecordFrame(tracker);

        tracker.OnRelease(3);

        Assert.Equal(0, watches.Live);
    }

    [Fact]
    public void AFaultOnAPageAlreadyLetGoIsRetriedWhileNothingElseBlocksIt()
    {
        var watches = new FakeWatches();
        var tracker = Create(watches);
        RecordFrame(tracker);
        tracker.OnRelease(3);
        Assert.True(tracker.TryHandleFault(FaultKind.Write, 0x100010, out _));

        // A second thread's store that met the same protection before the release applied.
        Assert.True(tracker.TryHandleFault(FaultKind.Write, 0x100020, out _));
        // A page the diagnostic never armed is not its own.
        Assert.False(tracker.TryHandleFault(FaultKind.Write, 0x900000, out _));
    }

    [Fact]
    public void ACensusFindsALastWriterInAnUnsampledFrameThroughTheStreamHistory()
    {
        var watches = new FakeWatches();
        var tracker = Create(watches, every: 4);
        tracker.OnCommit(1, present: true);
        // Buffer 2 belongs to no sampled window: the previous frame's tail, storing 0x700000.
        tracker.NoteOpProgram(0xC0);
        tracker.NoteWrite(0x700000, 0x40, OverlapWrite.Storage, OverlapTag.StorageWritten, 0xC0);
        tracker.EndOp(dispatch: true, queue: 0);
        tracker.OnCommit(2, present: true);
        tracker.OnCommit(3, present: true);
        // Buffer 4 opens window 4 and reads it.
        tracker.NoteOpProgram(0xD0);
        tracker.MarkRead(0x700000, 0x40, OverlapTag.StorageRead);
        tracker.NoteWrite(0x200000, 0x40, OverlapWrite.Storage, OverlapTag.StorageWritten, 0xD0);
        tracker.EndOp(dispatch: true, queue: 0);
        tracker.OnCommit(4, present: false);

        var writer = tracker.LastWriterOf(4, 4, 0, 0x700000, 0x40);
        Assert.Contains("serial=2 ", writer);
        Assert.Contains("op=0/1 kind=dispatch queue=0 programs=[00000000000000C0]", writer);
    }

    [Fact]
    public void ACensusFindsTheWriterOpAndTheLastWriterOfItsInputsInStreamOrder()
    {
        var watches = new FakeWatches();
        var tracker = Create(watches);
        tracker.OnCommit(1, present: true);
        // Op 0, a draw: renders the image at 0x500000.
        tracker.NoteOpProgram(0xA0);
        tracker.NoteImage(0x500000, 0x1000, written: true);
        tracker.EndOp(dispatch: false, queue: 0);
        tracker.OnCommit(2, present: false);
        // Op 0 of buffer 3, a dispatch on queue 9: samples that image, reads a constant range, stores a result.
        tracker.NoteOpProgram(0xB0);
        tracker.NoteImage(0x500000, 0x1000, written: false);
        tracker.MarkRead(0x600000, 0x40, OverlapTag.Constant);
        tracker.NoteWrite(0x200000, 0x40, OverlapWrite.Storage, OverlapTag.StorageWritten, 0xB0);
        tracker.EndOp(dispatch: true, queue: 9);
        tracker.OnCommit(3, present: false);
        tracker.OnRelease(3);

        var image = tracker.LastWriterOf(1, 3, 0, 0x500000, 0x1000);
        Assert.Contains("serial=2 vs_present=1 op=0/1 kind=draw queue=0 programs=[00000000000000A0]", image);
        Assert.Contains("how=image_target", image);
        Assert.Contains("last_writer=none_in_history", tracker.LastWriterOf(1, 3, 0, 0x600000, 0x40));

        // The guest loads the result before buffer 3 completes: the census keeps the range for its value.
        Assert.True(tracker.TryHandleFault(FaultKind.Read, 0x200010, out _));
        Assert.Single(tracker.RecordOf(1, 3).CensusRanges!);
        tracker.OnCompleted(3);
        Assert.Equal(0, watches.Live);
    }

    [Fact]
    public void ErasPlaceABufferAgainstTheSampledFramesPresent()
    {
        var watches = new FakeWatches();
        var tracker = Create(watches);
        var window = new InFlightOverlapTracker.Window { PreviousPresentSerial = 10, PresentSerial = 40 };

        Assert.Equal("older", InFlightOverlapTracker.EraNames[tracker.Era(window, 9)]);
        Assert.Equal("f_first", InFlightOverlapTracker.EraNames[tracker.Era(window, 15)]);
        Assert.Equal("f_second", InFlightOverlapTracker.EraNames[tracker.Era(window, 25)]);
        Assert.Equal("f_last", InFlightOverlapTracker.EraNames[tracker.Era(window, 40)]);
        Assert.Equal("p+2", InFlightOverlapTracker.EraNames[tracker.Era(window, 42)]);
        Assert.Equal("next", InFlightOverlapTracker.EraNames[tracker.Era(window, 45)]);
    }
}
