// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE.GpuMemory;
using Xunit;

namespace SharpEmu.Libs.Tests.HLE;

/// <summary>
/// The in-flight overlap diagnostic without a page guard: a sampled frame's gate arms the released buffers the
/// GPU has not completed (reads for stores, writes for loads), each fault is attributed by byte and let go,
/// a submission's command buffer is watched until it is walked, and every watch is given back.
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

    private InFlightOverlapTracker Create(FakeWatches watches, int every = 1) => new(watches, () => _now, every, 1000, armReads: true);

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
    public void NothingIsArmedBeforeTheGateAndCompletionGivesEveryWatchBack()
    {
        var watches = new FakeWatches();
        var tracker = Create(watches);
        RecordFrame(tracker);

        // Present + 1 is not the gate.
        tracker.OnRelease(2);
        Assert.Equal(0, watches.Live);

        tracker.OnRelease(3);
        Assert.Equal(2, watches.Live);
        tracker.OnCompleted(3);
        Assert.Equal(0, watches.Live);
    }

    [Fact]
    public void ACompletedBufferIsNotArmed()
    {
        var watches = new FakeWatches();
        var tracker = Create(watches);
        RecordFrame(tracker);
        tracker.OnCompleted(2);

        tracker.OnRelease(3);

        Assert.Equal(0, watches.Live);
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
        Assert.Contains("ranges=0.0/0 hit_range_mb=0.000 page_range_mb=0.000 nb_pages=1.0/1", line);
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
        var tracker = Create(watches, every: 2);
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
