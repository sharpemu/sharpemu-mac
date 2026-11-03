// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.VideoOut.Overlay;
using Xunit;

namespace SharpEmu.Libs.Tests.VideoOut.Overlay;

public sealed class OverlayAnalysisTests
{
    private static readonly string[] Names = ["", "GuestCode", "GpuGraphics", "FrameSlotWait", "Draw", "Shader"];

    private static FrameRecord Frame(long seq, float frameMs, float main = 0, float workers = 0, float render = 0, float gpu = 0,
        CostSample[]? costs = null, int activeWorkers = 4)
    {
        var kinds = new float[(int)CostKind.Count];
        kinds[(int)CostKind.Guest] = main + workers;
        kinds[(int)CostKind.Render] = render;
        kinds[(int)CostKind.Gpu] = gpu;
        return new FrameRecord
        {
            Sequence = seq,
            FrameMs = frameMs,
            GuestMainMs = main,
            GuestWorkersMs = workers,
            ActiveWorkers = activeWorkers,
            RenderBusyMs = render,
            GpuMs = gpu,
            KindMs = kinds,
            Costs = costs ?? [],
        };
    }

    private static OverlaySnapshot Snapshot(params FrameRecord[] frames) =>
        new(frames, new SystemStats { TargetFps = 60 }, id => id >= 0 && id < Names.Length ? Names[id] : "?");

    private static OverlayView Build(OverlaySnapshot snapshot, bool paused = false, long end = -1, long sel = -1)
    {
        var view = new OverlayView();
        new OverlayAnalyzer().Build(snapshot, paused, end, sel, view);
        return view;
    }

    [Fact]
    public void LowFpsIsTheMeanOfTheWorstSliceExpressedAsFps()
    {
        var ms = new float[100];
        for (var i = 0; i < 100; i++)
        {
            ms[i] = 10f;
        }

        ms[99] = 100f;
        Array.Sort(ms);
        Assert.Equal(10.0, OverlayAnalyzer.LowFps(ms, 100, 0.01), 3);
        Assert.Equal(1000.0 / ((100.0 + 10.0) / 2.0), OverlayAnalyzer.LowFps(ms, 100, 0.02), 3);
        Assert.Equal(0.0, OverlayAnalyzer.LowFps(ms, 0, 0.01));
    }

    [Fact]
    public void ViewFpsFollowsTheRecentFrameTimes()
    {
        var frames = Enumerable.Range(1, 40).Select(i => Frame(i, 20f)).ToArray();
        var view = Build(Snapshot(frames));
        Assert.Equal(50.0, view.Fps, 1);
        Assert.Equal(20.0, view.AvgFrameMs, 1);
        Assert.Equal(frames[^1].Sequence, view.Frame!.Sequence);
    }

    [Fact]
    public void PausedViewAnchorsToTheRequestedSequence()
    {
        var frames = Enumerable.Range(1, 30).Select(i => Frame(i, 10f + i)).ToArray();
        var view = Build(Snapshot(frames), paused: true, end: 20, sel: 12);
        Assert.Equal(12, view.Frame!.Sequence);
        Assert.Equal(11, view.SelIndex);
        Assert.Equal(20, view.Frames[view.EndIndex].Sequence);
    }

    [Fact]
    public void IndexOfSequenceClampsAndFindsExactMatches()
    {
        var frames = Enumerable.Range(10, 20).Select(i => Frame(i, 16f)).ToArray();
        Assert.Equal(5, OverlayAnalyzer.IndexOfSequence(frames, 15));
        Assert.Equal(0, OverlayAnalyzer.IndexOfSequence(frames, 1));
        Assert.Equal(19, OverlayAnalyzer.IndexOfSequence(frames, 999));
    }

    [Theory]
    [InlineData(40f, 5f, 8f, 38f, OverlayVerdictKind.Bound, OverlayBusyLane.Gpu)]
    [InlineData(40f, 36f, 8f, 4f, OverlayVerdictKind.Bound, OverlayBusyLane.GuestMain)]
    [InlineData(40f, 6f, 30f, 10f, OverlayVerdictKind.Sync, OverlayBusyLane.Render)]
    [InlineData(33f, 6f, 8f, 10f, OverlayVerdictKind.Limiter, OverlayBusyLane.Gpu)]
    public void VerdictPicksTheBusiestLaneAndOnlyCallsItBoundAtEightyFivePercent(
        float frameMs, float main, float render, float gpu, OverlayVerdictKind kind, OverlayBusyLane lane)
    {
        var agg = new FrameAggregate { Frames = 1, FrameMs = frameMs };
        agg.LaneMs[(int)OverlayBusyLane.GuestMain] = main;
        agg.LaneMs[(int)OverlayBusyLane.Render] = render;
        agg.LaneMs[(int)OverlayBusyLane.Gpu] = gpu;
        var verdict = OverlayAnalyzer.ComputeVerdict(agg, 1000f / 30f);
        Assert.Equal(kind, verdict.Kind);
        Assert.Equal(lane, verdict.Lane);
    }

    [Fact]
    public void VerdictFlagsStallsAndHandlesNoData()
    {
        var stall = new FrameAggregate { Frames = 1, FrameMs = 3000 };
        Assert.Equal(OverlayVerdictKind.Stall, OverlayAnalyzer.ComputeVerdict(stall, 33f).Kind);
        Assert.Equal(default, OverlayAnalyzer.ComputeVerdict(new FrameAggregate(), 33f));
    }

    [Fact]
    public void BudgetBarsUsePerLaneBusyTimeNotTheKindTotals()
    {
        // The prototype summed every kind into the "guest" bar; here the GPU lane is its own busy time.
        var frame = Frame(1, 40f, main: 10f, workers: 12f, render: 5f, gpu: 38f, activeWorkers: 4);
        var view = Build(Snapshot(frame));
        Assert.Equal(10f, view.Aggregate.Busy(OverlayBusyLane.GuestMain), 2);
        Assert.Equal(38f, view.Aggregate.Busy(OverlayBusyLane.Gpu), 2);
        Assert.Equal(5f, view.Aggregate.Busy(OverlayBusyLane.Render), 2);
        Assert.Equal(OverlayVerdictKind.Bound, view.Verdict.Kind);
        Assert.Equal(OverlayBusyLane.Gpu, view.Verdict.Lane);
    }

    [Fact]
    public void WorkerLaneIsTheBusiestWorkerNotTheSum()
    {
        var frame = new FrameRecord
        {
            Sequence = 1,
            FrameMs = 30f,
            GuestWorkersMs = 40f,
            ActiveWorkers = 4,
            Threads = [new ThreadSample(1, OverlayLane.GuestWorkers, 18f), new ThreadSample(1, OverlayLane.GuestWorkers, 7f)],
        };
        Assert.Equal(18f, OverlayAnalyzer.WorkerBusiestMs(frame), 3);
        var fallback = new FrameRecord { Sequence = 2, FrameMs = 30f, GuestWorkersMs = 40f, ActiveWorkers = 4 };
        Assert.Equal(10f, OverlayAnalyzer.WorkerBusiestMs(fallback), 3);
    }

    [Fact]
    public void TopCostsSortBySelfTimeDescendingAndHonourTopN()
    {
        var costs = new[]
        {
            new CostSample(4, CostKind.Render, OverlayLane.Render, 2f, 10),
            new CostSample(2, CostKind.Gpu, OverlayLane.Gpu, 30f, 1),
            new CostSample(1, CostKind.Guest, OverlayLane.GuestMain, 9f, 3),
            new CostSample(5, CostKind.Shader, OverlayLane.Render, 1f, 4),
        };
        var view = Build(Snapshot(Frame(1, 40f, costs: costs)));
        var result = new TopCostsResult();
        new OverlayAnalyzer().ComputeTopCosts(view,
            new TopCostOptions(3, TopCostSort.Self, TopCostSource.Frame, 12, TopCostThread.All, true), result);
        Assert.Equal([2, 1, 4], result.Rows.Select(r => r.NameId));
        Assert.Equal(30f, result.Rows[0].Milliseconds, 3);
    }

    [Fact]
    public void TopCostsCanSortByCallCountFilterByThreadAndDropWaits()
    {
        var costs = new[]
        {
            new CostSample(4, CostKind.Render, OverlayLane.Render, 2f, 100),
            new CostSample(2, CostKind.Gpu, OverlayLane.Gpu, 30f, 1),
            new CostSample(3, CostKind.Wait, OverlayLane.GuestMain, 20f, 5),
            new CostSample(1, CostKind.Guest, OverlayLane.GuestMain, 9f, 3),
        };
        var view = Build(Snapshot(Frame(1, 40f, costs: costs)));
        var analyzer = new OverlayAnalyzer();
        var byCount = new TopCostsResult();
        analyzer.ComputeTopCosts(view, new TopCostOptions(5, TopCostSort.Count, TopCostSource.Frame, 12, TopCostThread.All, true), byCount);
        Assert.Equal(4, byCount.Rows[0].NameId);

        var noWaits = new TopCostsResult();
        analyzer.ComputeTopCosts(view, new TopCostOptions(5, TopCostSort.Self, TopCostSource.Frame, 12, TopCostThread.All, false), noWaits);
        Assert.DoesNotContain(noWaits.Rows, r => r.Kind == CostKind.Wait);

        var guestOnly = new TopCostsResult();
        analyzer.ComputeTopCosts(view, new TopCostOptions(5, TopCostSort.Self, TopCostSource.Frame, 12, TopCostThread.GuestMain, true), guestOnly);
        Assert.All(guestOnly.Rows, r => Assert.Equal(OverlayLane.GuestMain, r.Lane));
    }

    [Fact]
    public void MemoryTrendKeepsSamplesAtTheRequestedInterval()
    {
        var analyzer = new OverlayAnalyzer();
        analyzer.AddMemorySample(1.0, 0, 100);
        analyzer.AddMemorySample(2.0, 50, 100);
        analyzer.AddMemorySample(3.0, 100, 100);
        Assert.Equal(2, analyzer.MemoryTrend.Length);
        Assert.Equal(3f, analyzer.MemoryTrend[^1], 3);
    }

    [Fact]
    public void AGpuBoundFrameStacksAsGpuThenWait()
    {
        var frame = Frame(1, 26f, main: 6f, workers: 8f, render: 2f, gpu: 21f);
        Span<float> kinds = stackalloc float[(int)CostKind.Count];
        Assert.Equal(OverlayBusyLane.Gpu, OverlayAnalyzer.BoundBreakdown(frame, kinds));
        Assert.Equal(21f, kinds[(int)CostKind.Gpu], 3);
        Assert.Equal(5f, kinds[(int)CostKind.Wait], 3);
        Assert.Equal(0f, kinds[(int)CostKind.Guest], 3);
    }

    [Fact]
    public void AGuestBoundFrameSplitsHleFromGuestCode()
    {
        var frame = Frame(1, 40f, main: 36f, gpu: 10f,
            costs: [new CostSample(OverlayTelemetry.HleNameBase + 1, CostKind.Hle, OverlayLane.GuestMain, 9f, 100),
                    new CostSample(OverlayTelemetry.HleNameBase + 2, CostKind.Hle, OverlayLane.GuestWorkers, 30f, 100)]);
        Span<float> kinds = stackalloc float[(int)CostKind.Count];
        Assert.Equal(OverlayBusyLane.GuestMain, OverlayAnalyzer.BoundBreakdown(frame, kinds));
        Assert.Equal(9f, kinds[(int)CostKind.Hle], 3);
        Assert.Equal(27f, kinds[(int)CostKind.Guest], 3);
        Assert.Equal(4f, kinds[(int)CostKind.Wait], 3);
    }
}
