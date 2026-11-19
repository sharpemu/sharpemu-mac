// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using SharpEmu.HLE.GpuMemory;
using Xunit;

namespace SharpEmu.Libs.Tests.HLE;

/// <summary>
/// The frame-overlap diagnostic's bookkeeping without a page guard: which frame a mark belongs
/// to, which pages a completed frame arms and how, that each armed page counts its first fault
/// once and is let go, and that every watch it took is given back at the next boundary.
/// </summary>
public sealed class FrameOverlapTrackerTests
{
    private static readonly long Millisecond = Stopwatch.Frequency / 1000;

    private sealed class FakeWatches : IFrameOverlapWatches
    {
        public readonly Dictionary<(ulong Page, bool Reads), int> Counts = new();
        public readonly HashSet<ulong> ReadWatchedElsewhere = new();

        public bool IsReadWatched(ulong page) => ReadWatchedElsewhere.Contains(page);

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

                var key = (blockBase + (ulong)index * TrackerLayout.PageBytes, reads);
                var count = Counts.GetValueOrDefault(key) + delta;
                Assert.True(count >= 0, $"watch count below zero at 0x{key.Item1:X}");
                Counts[key] = count;
            }
        }

        public int Live => Counts.Values.Sum();
    }

    private long _now = 1;

    private FrameOverlapTracker Create(FakeWatches watches, int every = 1, int maxPages = 1000) =>
        new(watches, () => _now, every, maxPages, armReads: true, startThread: false);

    [Fact]
    public void ACompletedFrameArmsItsInPlaceReadsForWritesAndItsWritesForReads()
    {
        var watches = new FakeWatches();
        var tracker = Create(watches);
        tracker.MarkRead(0x10000, 0x2000, OverlapRead.InPlace);
        tracker.MarkRead(0x20000, 0x1000, OverlapRead.Staged);
        tracker.NoteWrite(0x30000, 0x1000, OverlapWrite.Storage);
        tracker.NoteWrite(0x40008, 8, OverlapWrite.EopDestination);
        tracker.OnAdmission(0x40000, 0x100);
        tracker.OnBoundaryAdmitted(unfinished: 0, blocked: 0);
        tracker.OnBoundaryWalked();
        _now = 10 * Millisecond;
        tracker.OnFrameCommitted(5);
        _now = 50 * Millisecond;
        tracker.OnCompleted(5);
        tracker.DrainRequests();

        Assert.Equal(1, watches.Counts[(0x10000, false)]);
        Assert.Equal(1, watches.Counts[(0x11000, false)]);
        Assert.Equal(1, watches.Counts[(0x30000, true)]);
        Assert.Equal(1, watches.Counts[(0x40000, true)]);
        Assert.False(watches.Counts.ContainsKey((0x20000, false)));
        Assert.Equal(4, watches.Live);

        // The first fault on a page is counted and lets the page go; a second one (another thread that
        // faulted before the release was applied) is retried without being counted again.
        Assert.True(tracker.TryHandleFault(FaultKind.Write, 0x10010, out var blocked));
        Assert.False(blocked);
        Assert.True(tracker.TryHandleFault(FaultKind.Write, 0x10020, out _));
        Assert.False(tracker.TryHandleFault(FaultKind.Write, 0x90000, out _));
        _now = 52 * Millisecond;
        tracker.OnAdmission(0x50000, 0x100);
        Assert.True(tracker.TryHandleFault(FaultKind.Read, 0x40010, out _));
        Assert.Equal(2, watches.Live);

        // The next frame's boundary ends the build: the rest is given back and the frame is counted.
        tracker.OnBoundaryAdmitted(unfinished: 1, blocked: 0);
        tracker.DrainRequests();
        Assert.Equal(0, watches.Live);

        var line = tracker.TakeWindowLine();
        Assert.Contains("frames=1 sampled=1 ", line);
        Assert.Contains("overlap_ms=40/40", line);
        Assert.Contains("complete_to_admit_ms=2/2", line);
        Assert.Contains("rs_written=1/1", line);
        Assert.Contains("ws_read=1/1", line);
        Assert.Contains("label_touched=1/1", line);
        Assert.Contains("cb_touched=1/1", line);
        Assert.Contains("labels_in_cb=1/1", line);
        Assert.Contains("first_fault_ms=0/0", line);
        Assert.Contains("trap_before_submit_frames=1", line);
        Assert.Contains("staged_mb=0/0", line);
        // The window starts over.
        Assert.Contains("frames=0 ", tracker.TakeWindowLine());
    }

    [Fact]
    public void MarksAfterACommitBelongToTheNextFrame()
    {
        var watches = new FakeWatches();
        var tracker = Create(watches, every: 2);
        tracker.MarkRead(0x10000, 0x1000, OverlapRead.InPlace);
        tracker.OnBoundaryAdmitted(0, 0);
        tracker.OnFrameCommitted(1);
        tracker.MarkRead(0x90000, 0x1000, OverlapRead.InPlace);
        tracker.OnCompleted(1);
        tracker.DrainRequests();
        // Frame 0 is sampled (every 2nd from 0): only its own page is armed.
        Assert.Equal(1, watches.Counts[(0x10000, false)]);
        Assert.False(watches.Counts.ContainsKey((0x90000, false)));

        tracker.OnBoundaryAdmitted(0, 0);
        tracker.OnFrameCommitted(2);
        tracker.OnCompleted(2);
        tracker.DrainRequests();
        // Frame 1 is not sampled, and frame 0 was given back at frame 1's boundary.
        Assert.Equal(0, watches.Live);
    }

    [Fact]
    public void ArmedAtCommitTheWatchesCoverTheGpuTailAndAreNotArmedTwice()
    {
        var watches = new FakeWatches();
        var tracker = new FrameOverlapTracker(watches, () => _now, 1, 1000, armReads: true, startThread: false) { ArmAtCommit = true };
        tracker.MarkRead(0x10000, 0x1000, OverlapRead.InPlace);
        tracker.OnBoundaryAdmitted(0, 0);
        _now = 10 * Millisecond;
        tracker.OnFrameCommitted(1);
        tracker.DrainRequests();
        Assert.Equal(1, watches.Live);

        // A guest store while the GPU still runs the frame: counted before the completion.
        _now = 20 * Millisecond;
        Assert.True(tracker.TryHandleFault(FaultKind.Write, 0x10000, out _));
        _now = 40 * Millisecond;
        tracker.OnCompleted(1);
        tracker.DrainRequests();
        Assert.Equal(0, watches.Live);
        tracker.OnBoundaryAdmitted(0, 0);
        tracker.DrainRequests();
        var line = tracker.TakeWindowLine();
        Assert.Contains("rs_written=1/1", line);
        Assert.Contains("first_fault_ms=-20/-20", line);
    }

    [Fact]
    public void AReadWatchedPageIsSkippedAndTheCapIsCounted()
    {
        var watches = new FakeWatches();
        watches.ReadWatchedElsewhere.Add(0x30000);
        var tracker = Create(watches, maxPages: 2);
        tracker.NoteWrite(0x30000, 0x3000, OverlapWrite.Writeback);
        tracker.OnBoundaryAdmitted(0, 0);
        tracker.OnFrameCommitted(1);
        tracker.OnCompleted(1);
        tracker.DrainRequests();

        // The one read bit belongs to someone else: write-watched instead, and two pages fit the cap.
        Assert.Equal(1, watches.Counts[(0x30000, false)]);
        Assert.Equal(1, watches.Counts[(0x31000, true)]);
        Assert.False(watches.Counts.ContainsKey((0x32000, true)));
        tracker.OnBoundaryAdmitted(0, 0);
        tracker.DrainRequests();
        Assert.Equal(0, watches.Live);
        var line = tracker.TakeWindowLine();
        Assert.Contains("cap_frames=1", line);
        Assert.Contains("skipped_read_watched=1", line);
    }

    [Fact]
    public void ManagedWritesAreCountedWithoutReleasingTheWatch()
    {
        var watches = new FakeWatches();
        var tracker = Create(watches);
        tracker.MarkRead(0x10000, 0x1000, OverlapRead.InPlace);
        tracker.OnBoundaryAdmitted(0, 0);
        tracker.OnFrameCommitted(1);
        tracker.OnCompleted(1);
        tracker.DrainRequests();
        tracker.NoteManagedWrite(0x10100, 4);
        tracker.NoteManagedWrite(0x10200, 4);
        Assert.Equal(1, watches.Live);
        tracker.OnBoundaryAdmitted(0, 0);
        tracker.DrainRequests();
        Assert.Contains("managed=1/1", tracker.TakeWindowLine());
    }

    [Fact]
    public void AFaultThatRacedTheDisarmIsRetriedOncePerPage()
    {
        var watches = new FakeWatches();
        var tracker = Create(watches);
        tracker.NoteWrite(0x40000, 8, OverlapWrite.CpWrite);
        tracker.OnBoundaryAdmitted(0, 0);
        tracker.OnFrameCommitted(1);
        tracker.OnCompleted(1);
        tracker.DrainRequests();
        tracker.OnBoundaryAdmitted(0, 0);
        tracker.DrainRequests();
        Assert.Equal(0, watches.Live);

        // The guest read the page before the given-back protection reached it (two threads at once).
        Assert.True(tracker.TryHandleFault(FaultKind.Read, 0x40000, out var blocked));
        Assert.False(blocked);
        Assert.True(tracker.TryHandleFault(FaultKind.Read, 0x40000, out _));
        Assert.False(tracker.TryHandleFault(FaultKind.Read, 0x50000, out _));
        Assert.Contains("late_faults=2", tracker.TakeWindowLine());
    }

    [Fact]
    public void ASecondThreadFaultingOnAPageAnotherThreadReleasedIsRetried()
    {
        var watches = new FakeWatches();
        var tracker = Create(watches);
        tracker.NoteWrite(0x40000, 8, OverlapWrite.Storage);
        tracker.OnBoundaryAdmitted(0, 0);
        tracker.OnFrameCommitted(1);
        tracker.OnCompleted(1);
        tracker.DrainRequests();
        Assert.True(tracker.TryHandleFault(FaultKind.Read, 0x40000, out _));
        Assert.True(tracker.TryHandleFault(FaultKind.Read, 0x40004, out _));
        tracker.OnBoundaryAdmitted(0, 0);
        tracker.DrainRequests();
        var line = tracker.TakeWindowLine();
        Assert.Contains("ws_read=1/1", line);
        Assert.Contains("late_faults=1", line);
    }

    [Fact]
    public void FaultsAreSplitIntoHitsNeighboursAndRecordTimeLabels()
    {
        var watches = new FakeWatches();
        var tracker = Create(watches);
        tracker.MarkRead(0x10000, 0x100, OverlapRead.InPlace);
        tracker.MarkRead(0x20000, 0x100, OverlapRead.InPlace);
        tracker.MarkRead(0x60000, 0x10, OverlapRead.DeviceAddress, program: 0xDA);
        tracker.NoteWrite(0x30000, 0x100, OverlapWrite.Storage, program: 0xABC);
        tracker.NoteWrite(0x31800, 0x10, OverlapWrite.Storage);
        tracker.NoteWrite(0x40008, 8, OverlapWrite.CpWrite);
        tracker.OnBoundaryAdmitted(0, 0);
        _now = 10 * Millisecond;
        tracker.OnFrameCommitted(1);
        _now = 50 * Millisecond;
        tracker.OnCompleted(1);
        tracker.DrainRequests();

        _now = 53 * Millisecond;
        Assert.True(tracker.TryHandleFault(FaultKind.Write, 0x10F00, out _)); // read set, other bytes of the page
        Assert.True(tracker.TryHandleFault(FaultKind.Write, 0x20010, out _)); // read set, inside the bound range
        Assert.True(tracker.TryHandleFault(FaultKind.Read, 0x30010, out _));  // GPU-valued store: a real wait
        Assert.True(tracker.TryHandleFault(FaultKind.Read, 0x31000, out _));  // shares the page only
        Assert.True(tracker.TryHandleFault(FaultKind.Read, 0x4000C, out _));  // a CP write: landed at publication
        Assert.True(tracker.TryHandleFault(FaultKind.Write, 0x60008, out _)); // device-address reach itself
        _now = 55 * Millisecond;
        tracker.OnAdmission(0x90000, 0x40);
        tracker.OnBoundaryAdmitted(0, 0);
        tracker.DrainRequests();

        var line = tracker.TakeWindowLine();
        Assert.Contains("armed_before_complete=0", line);
        Assert.Contains("rs_hit=1/1 rs_nb=1/1", line);
        Assert.Contains("ws_read label=1/1 gpu=1/1 nb=1/1", line);
        Assert.Contains("da_only hit=1/1 nb=0/0", line);
        Assert.Contains("real_trap_pages=3/3 real_wait_pages=1/1", line);
        Assert.Contains("k6_frames=1 k6_ms=3/3 k6_before_submit=1", line);
        Assert.Contains("overlap_until_wait_ms=3/3", line);
        var pages = tracker.TopRanges();
        Assert.Contains("0x30000 faults=1 ranges=[storage:0x30000+0x100:prog=0x0000000000000ABC] accesses=[0x010r=gpu_hit:storage:prog=0x0000000000000ABC]", pages);
        Assert.Contains("0x008w=da_only_hit:prog=0x00000000000000DA", pages);
    }

    [Fact]
    public void AGpuValuedRangeOutranksALabelOverTheSameBytes()
    {
        var ranges = new List<(ulong, ulong, int, ulong)>
        {
            (0x1000, 8, (int)OverlapWrite.EopLabel, 0),
            (0x1000, 8, (int)OverlapWrite.EopGpuValued, 0),
        };
        Assert.Equal(FrameOverlapTracker.Hit.GpuValued, FrameOverlapTracker.ClassifyWrite(ranges, 0x1004, out _, out _));
        Assert.Equal(FrameOverlapTracker.Hit.Label, FrameOverlapTracker.ClassifyWrite(ranges.Take(1).ToList(), 0x1004, out _, out _));
        Assert.Equal(FrameOverlapTracker.Hit.Neighbour, FrameOverlapTracker.ClassifyWrite(ranges, 0x1008, out _, out _));
    }

    [Fact]
    public void ThePageSetCountsDistinctPagesAcrossBlocks()
    {
        var set = new OverlapPageSet();
        Assert.Equal(3, set.Add(0x3FF000, 0x2001));
        Assert.Equal(0, set.Add(0x400000, 0x1000));
        Assert.True(set.Contains(0x401FFF));
        Assert.False(set.Contains(0x402000));
        Assert.True(set.ContainsAny(0x3FE000, 0x1001));
        Assert.Equal(3, set.Count);
    }

    [Fact]
    public void PercentilesIgnoreMissingSamples()
    {
        var culture = System.Globalization.CultureInfo.InvariantCulture;
        Assert.Equal("-", FrameOverlapTracker.Percentiles([double.NaN], culture));
        Assert.Equal("3/5", FrameOverlapTracker.Percentiles([1, 2, 3, 4, 5, double.NaN], culture));
    }
}
