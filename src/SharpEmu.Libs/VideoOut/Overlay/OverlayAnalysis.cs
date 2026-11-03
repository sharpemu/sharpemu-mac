// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.VideoOut.Overlay;

public enum OverlayVerdictKind : byte
{
    /// <summary>No frames yet.</summary>
    None,
    /// <summary>The frame took more than 1.5 s (loading, streaming).</summary>
    Stall,
    /// <summary>No lane is saturated and the frame met its budget: the frame limiter paces the game.</summary>
    Limiter,
    /// <summary>One lane is busy for at least 85% of the frame.</summary>
    Bound,
    /// <summary>No lane is saturated although the frame is late: it is waiting on something.</summary>
    Sync,
}

/// <summary>The four parallel "lanes" the budget bar compares against the frame budget.</summary>
public enum OverlayBusyLane : byte
{
    GuestMain,
    Workers,
    Render,
    Gpu,
    Count,
}

/// <summary>Per-lane busy ms averaged over a window (or one frame).</summary>
public sealed class FrameAggregate
{
    public int Frames;
    public float FrameMs;
    public readonly float[] LaneMs = new float[(int)OverlayBusyLane.Count];
    public readonly float[] KindMs = new float[(int)CostKind.Count];
    public float RenderWaitMs;
    public float Draws;
    public float Dispatches;
    public int ActiveWorkers;

    public float Busy(OverlayBusyLane lane) => LaneMs[(int)lane];
    public float Ratio(OverlayBusyLane lane) => FrameMs > 0 ? LaneMs[(int)lane] / FrameMs : 0f;
    public float KindTotal()
    {
        var sum = 0f;
        foreach (var v in KindMs)
        {
            sum += v;
        }

        return sum;
    }

    public void Clear()
    {
        Frames = 0;
        FrameMs = 0;
        RenderWaitMs = 0;
        Draws = 0;
        Dispatches = 0;
        ActiveWorkers = 0;
        Array.Clear(LaneMs);
        Array.Clear(KindMs);
    }
}

/// <summary>"Bound by X" decision computed from lane busy/frame ratios.</summary>
public readonly record struct OverlayVerdict(OverlayVerdictKind Kind, OverlayBusyLane Lane, float GuestRatio, float WorkerRatio, float RenderRatio, float GpuRatio);

/// <summary>What the widgets display: derived from one snapshot plus the pause/selection state.</summary>
public sealed class OverlayView
{
    public OverlaySnapshot Snapshot = OverlaySnapshot.Empty;
    public IReadOnlyList<FrameRecord> Frames = [];
    /// <summary>Last frame of the displayed window (the newest frame when live).</summary>
    public int EndIndex = -1;
    /// <summary>The frame inspected after a click or step, or -1.</summary>
    public int SelIndex = -1;
    public long SelSequence = -1;
    public long EndSequence = -1;
    public bool Paused;
    /// <summary>The selected frame, else the newest frame in the window.</summary>
    public FrameRecord? Frame;
    public float BudgetMs = 1000f / 60f;
    public double TargetFps = 60;
    public double Fps;
    public double AvgFrameMs;
    public double Low1Fps;
    public double Low01Fps;
    /// <summary>Selected frame time relative to the window average (1 = typical).</summary>
    public float FrameVsAverage = 1f;
    public readonly FrameAggregate Aggregate = new();
    public OverlayVerdict Verdict;
    public readonly List<int> HitchIndices = [];
    public string? LastHitchReason;
    public SystemStats System = SystemStats.Empty;

    public int Last => Frames.Count - 1;

    /// <summary>The last index of a window of <paramref name="n"/> frames that keeps a selected frame visible.</summary>
    public int WindowEnd(int n)
    {
        var end = EndIndex;
        if (SelIndex >= 0)
        {
            if (SelIndex > end)
            {
                end = SelIndex;
            }

            if (SelIndex < end - n + 1)
            {
                end = Math.Min(Last, SelIndex + n - 1);
            }
        }

        return end;
    }
}

public enum TopCostSort : byte { Self, Total, Count }
public enum TopCostSource : byte { Frame, Average }
public enum TopCostThread : byte { All, GuestMain, Workers, Render, Gpu }

public readonly record struct TopCostOptions(
    int MaxRows,
    TopCostSort Sort,
    TopCostSource Source,
    int AverageFrames,
    TopCostThread Thread,
    bool IncludeWaits);

/// <summary>Value is the sort metric per frame; Milliseconds the per-frame time (self per lane, or summed over lanes when sorting by total).</summary>
public readonly record struct TopCostRow(int NameId, CostKind Kind, OverlayLane Lane, float Value, float Milliseconds, float Count);

public sealed class TopCostsResult
{
    public readonly List<TopCostRow> Rows = [];
    /// <summary>Frame time the percentages refer to.</summary>
    public float FrameMs;
    /// <summary>Frames the figures were averaged over (1 for the frame source).</summary>
    public int Frames;
    public long FrameSequence;
}

/// <summary>
/// Turns snapshots into the numbers widgets draw: averages, lows, hitches, verdict, top costs and the
/// memory trend. Holds reusable scratch buffers, so it must be used from one thread.
/// </summary>
public sealed class OverlayAnalyzer
{
    public const int AggregateFrames = 12;
    public const int LowsWindow = 600;
    public const float BoundThreshold = 0.85f;
    public const float StallMs = 1500f;

    private float[] _scratch = new float[LowsWindow + 8];
    private readonly Dictionary<int, int> _costIndex = [];
    private readonly List<CostEntry> _costs = [];
    private float[] _sortKeys = new float[64];
    private int[] _sortItems = new int[64];
    private readonly float[] _trend = new float[240];
    private int _trendCount;
    private long _lastTrendTicks;

    private struct CostEntry
    {
        public int NameId;
        public CostKind Kind;
        public OverlayLane Lane;
        public float Ms;
        public float Count;
    }

    // ---------------------------------------------------------------- view

    /// <summary>Fills <paramref name="view"/> from a snapshot; selection and window are given as sequences.</summary>
    public void Build(OverlaySnapshot snapshot, bool paused, long endSequence, long selSequence, OverlayView view)
    {
        view.Snapshot = snapshot;
        view.Frames = snapshot.Frames;
        view.System = snapshot.System;
        view.Paused = paused;
        var frames = snapshot.Frames;
        var last = frames.Count - 1;
        view.TargetFps = snapshot.System.TargetFps > 0 ? snapshot.System.TargetFps : 60;
        view.BudgetMs = (float)(1000.0 / view.TargetFps);
        view.HitchIndices.Clear();
        view.LastHitchReason = null;
        view.Aggregate.Clear();
        if (last < 0)
        {
            view.EndIndex = -1;
            view.SelIndex = -1;
            view.SelSequence = -1;
            view.EndSequence = -1;
            view.Frame = null;
            view.Fps = snapshot.System.Fps;
            view.AvgFrameMs = view.Fps > 0 ? 1000.0 / view.Fps : 0;
            view.Low1Fps = view.Low01Fps = view.Fps;
            view.FrameVsAverage = 1f;
            view.Verdict = default;
            return;
        }

        var end = paused && endSequence >= 0 ? IndexOfSequence(frames, endSequence) : last;
        end = Math.Clamp(end, 0, last);
        var sel = -1;
        if (paused && selSequence >= 0)
        {
            sel = Math.Clamp(IndexOfSequence(frames, selSequence), 0, last);
        }

        view.EndIndex = end;
        view.SelIndex = sel;
        view.SelSequence = sel >= 0 ? frames[sel].Sequence : -1;
        view.EndSequence = frames[end].Sequence;
        view.Frame = frames[sel >= 0 ? sel : end];

        // Frame rate over about the last second of the displayed window.
        double sum = 0;
        var count = 0;
        for (var i = end; i >= 0 && (sum < 1000 || count < 2); i--)
        {
            sum += frames[i].FrameMs;
            count++;
            if (count >= 240)
            {
                break;
            }
        }

        view.AvgFrameMs = count > 0 ? sum / count : 0;
        view.Fps = sum > 0 ? count / (sum / 1000.0) : snapshot.System.Fps;

        // 1% / 0.1% lows over the last LowsWindow frames.
        var n = Math.Min(LowsWindow, end + 1);
        EnsureScratch(n);
        for (var i = 0; i < n; i++)
        {
            _scratch[i] = frames[end - i].FrameMs;
        }

        Array.Sort(_scratch, 0, n);
        view.Low1Fps = LowFps(_scratch, n, 0.01);
        view.Low01Fps = LowFps(_scratch, n, 0.001);

        // Aggregate: the selected frame alone, else the mean of the newest few.
        var aggregate = view.Aggregate;
        if (sel >= 0)
        {
            AddFrame(aggregate, frames[sel]);
        }
        else
        {
            for (var i = Math.Max(0, end - AggregateFrames + 1); i <= end; i++)
            {
                AddFrame(aggregate, frames[i]);
            }
        }

        Finish(aggregate);
        view.Verdict = ComputeVerdict(aggregate, view.BudgetMs);
        view.FrameVsAverage = view.AvgFrameMs > 0 ? (float)(view.Frame.FrameMs / view.AvgFrameMs) : 1f;

        var from = Math.Max(0, end - 239);
        for (var i = from; i <= end; i++)
        {
            if (frames[i].Hitch)
            {
                view.HitchIndices.Add(i);
                view.LastHitchReason = frames[i].HitchReason;
            }
        }
    }

    private void EnsureScratch(int n)
    {
        if (_scratch.Length < n)
        {
            _scratch = new float[n * 2];
        }
    }

    /// <summary>Index of the frame with this sequence, or the nearest one when it has scrolled out.</summary>
    public static int IndexOfSequence(IReadOnlyList<FrameRecord> frames, long sequence)
    {
        var lo = 0;
        var hi = frames.Count - 1;
        while (lo <= hi)
        {
            var mid = (lo + hi) >>> 1;
            var s = frames[mid].Sequence;
            if (s == sequence)
            {
                return mid;
            }

            if (s < sequence)
            {
                lo = mid + 1;
            }
            else
            {
                hi = mid - 1;
            }
        }

        return Math.Clamp(lo, 0, Math.Max(0, frames.Count - 1));
    }

    /// <summary>
    /// The frame rate of the slowest <paramref name="fraction"/> of frames (1% / 0.1% low): the mean
    /// of the worst <c>ceil(n*fraction)</c> frame times, as fps. <paramref name="ascendingMs"/> must be sorted.
    /// </summary>
    public static double LowFps(float[] ascendingMs, int count, double fraction)
    {
        if (count <= 0)
        {
            return 0;
        }

        var k = Math.Clamp((int)Math.Ceiling(count * fraction), 1, count);
        double sum = 0;
        for (var i = 0; i < k; i++)
        {
            sum += ascendingMs[count - 1 - i];
        }

        var mean = sum / k;
        return mean > 0 ? 1000.0 / mean : 0;
    }

    /// <summary>The busiest worker's CPU ms, or the aggregate divided by the active workers when per-thread data is absent.</summary>
    public static float WorkerBusiestMs(FrameRecord frame)
    {
        var best = 0f;
        var any = false;
        foreach (var t in frame.Threads)
        {
            if (t.Lane == OverlayLane.GuestWorkers)
            {
                any = true;
                best = MathF.Max(best, t.BusyMs);
            }
        }

        return any ? best : frame.GuestWorkersMs / Math.Max(1, frame.ActiveWorkers);
    }

    /// <summary>
    /// The lane that took longest this frame (guest main, busiest worker, render thread or GPU) and
    /// its busy ms; the frame cannot finish before that lane does, so it is what to chase.
    /// </summary>
    public static (OverlayBusyLane Lane, float BusyMs) BoundOf(FrameRecord frame)
    {
        var lane = OverlayBusyLane.GuestMain;
        var busy = frame.GuestMainMs;
        void Consider(OverlayBusyLane candidate, float ms)
        {
            if (ms > busy)
            {
                lane = candidate;
                busy = ms;
            }
        }

        Consider(OverlayBusyLane.Workers, WorkerBusiestMs(frame));
        Consider(OverlayBusyLane.Render, frame.RenderBusyMs);
        Consider(OverlayBusyLane.Gpu, frame.GpuMs);
        return (lane, MathF.Min(busy, frame.FrameMs));
    }

    /// <summary>
    /// Splits the bound lane's busy time by kind (HLE vs guest code on the guest lanes; render,
    /// shader and other work on the render thread; GPU on the GPU). Writes into <paramref name="kindMs"/>
    /// (cleared first) and returns the bound lane.
    /// </summary>
    public static OverlayBusyLane BoundBreakdown(FrameRecord frame, Span<float> kindMs)
    {
        kindMs.Clear();
        var (lane, busy) = BoundOf(frame);
        switch (lane)
        {
            case OverlayBusyLane.Gpu:
                kindMs[(int)CostKind.Gpu] = busy;
                break;
            case OverlayBusyLane.Render:
            {
                var k = frame.KindMs;
                var render = k[(int)CostKind.Render];
                var shader = k[(int)CostKind.Shader];
                var other = k[(int)CostKind.Other];
                var sum = render + shader + other;
                if (sum <= 0)
                {
                    kindMs[(int)CostKind.Render] = busy;
                    break;
                }

                kindMs[(int)CostKind.Render] = busy * render / sum;
                kindMs[(int)CostKind.Shader] = busy * shader / sum;
                kindMs[(int)CostKind.Other] = busy * other / sum;
                break;
            }
            default:
            {
                var target = lane == OverlayBusyLane.GuestMain ? OverlayLane.GuestMain : OverlayLane.GuestWorkers;
                var hle = 0f;
                foreach (var c in frame.Costs)
                {
                    if (c.Lane == target && c.Kind == CostKind.Hle)
                    {
                        hle += c.Milliseconds;
                    }
                }

                if (lane == OverlayBusyLane.Workers && frame.GuestWorkersMs > 0)
                {
                    // Worker HLE time is summed over every worker; take the busiest one's share.
                    hle *= busy / frame.GuestWorkersMs;
                }

                hle = Math.Clamp(hle, 0f, busy);
                kindMs[(int)CostKind.Hle] = hle;
                kindMs[(int)CostKind.Guest] = busy - hle;
                break;
            }
        }

        kindMs[(int)CostKind.Wait] = MathF.Max(0f, frame.FrameMs - busy);
        return lane;
    }

    public static string BusyLaneLabel(OverlayBusyLane lane) => lane switch
    {
        OverlayBusyLane.GuestMain => "guest main thread",
        OverlayBusyLane.Workers => "busiest guest worker",
        OverlayBusyLane.Render => "render thread",
        _ => "GPU",
    };

    private static void AddFrame(FrameAggregate a, FrameRecord f)
    {
        a.Frames++;
        a.FrameMs += f.FrameMs;
        a.LaneMs[(int)OverlayBusyLane.GuestMain] += f.GuestMainMs;
        a.LaneMs[(int)OverlayBusyLane.Workers] += WorkerBusiestMs(f);
        a.LaneMs[(int)OverlayBusyLane.Render] += f.RenderBusyMs;
        a.LaneMs[(int)OverlayBusyLane.Gpu] += f.GpuMs;
        a.RenderWaitMs += f.RenderWaitMs;
        a.Draws += f.Draws;
        a.Dispatches += f.Dispatches;
        a.ActiveWorkers = Math.Max(a.ActiveWorkers, f.ActiveWorkers);
        var kinds = f.KindMs;
        for (var k = 0; k < kinds.Length && k < a.KindMs.Length; k++)
        {
            a.KindMs[k] += kinds[k];
        }
    }

    private static void Finish(FrameAggregate a)
    {
        if (a.Frames <= 1)
        {
            return;
        }

        var inv = 1f / a.Frames;
        a.FrameMs *= inv;
        a.RenderWaitMs *= inv;
        a.Draws *= inv;
        a.Dispatches *= inv;
        for (var i = 0; i < a.LaneMs.Length; i++)
        {
            a.LaneMs[i] *= inv;
        }

        for (var i = 0; i < a.KindMs.Length; i++)
        {
            a.KindMs[i] *= inv;
        }
    }

    /// <summary>
    /// "Bound by X" is the lane with the highest busy/frame ratio when that ratio reaches 85%;
    /// otherwise the frame is stalled on sync / waits (or, when it met its budget, paced by the limiter).
    /// </summary>
    public static OverlayVerdict ComputeVerdict(FrameAggregate a, float budgetMs)
    {
        if (a.Frames == 0 || a.FrameMs <= 0)
        {
            return default;
        }

        var g = a.Ratio(OverlayBusyLane.GuestMain);
        var w = a.Ratio(OverlayBusyLane.Workers);
        var r = a.Ratio(OverlayBusyLane.Render);
        var p = a.Ratio(OverlayBusyLane.Gpu);
        var best = OverlayBusyLane.GuestMain;
        var bestRatio = g;
        if (w > bestRatio) { best = OverlayBusyLane.Workers; bestRatio = w; }
        if (r > bestRatio) { best = OverlayBusyLane.Render; bestRatio = r; }
        if (p > bestRatio) { best = OverlayBusyLane.Gpu; bestRatio = p; }

        if (a.FrameMs > StallMs)
        {
            return new(OverlayVerdictKind.Stall, best, g, w, r, p);
        }

        if (bestRatio >= BoundThreshold)
        {
            return new(OverlayVerdictKind.Bound, best, g, w, r, p);
        }

        return new(a.FrameMs <= budgetMs * 1.03f ? OverlayVerdictKind.Limiter : OverlayVerdictKind.Sync, best, g, w, r, p);
    }

    // ---------------------------------------------------------------- top costs

    public void ComputeTopCosts(OverlayView view, in TopCostOptions options, TopCostsResult result)
    {
        result.Rows.Clear();
        result.Frames = 0;
        result.FrameMs = 0;
        var frames = view.Frames;
        if (view.Frame is null || frames.Count == 0)
        {
            return;
        }

        _costIndex.Clear();
        _costs.Clear();
        int from, to;
        if (options.Source == TopCostSource.Average)
        {
            to = view.WindowEnd(options.AverageFrames);
            from = Math.Max(0, to - options.AverageFrames + 1);
        }
        else
        {
            to = view.SelIndex >= 0 ? view.SelIndex : view.EndIndex;
            from = to;
        }

        var span = to - from + 1;
        double frameMs = 0;
        var perName = options.Sort == TopCostSort.Total;
        for (var f = from; f <= to; f++)
        {
            var frame = frames[f];
            frameMs += frame.FrameMs;
            foreach (var c in frame.Costs)
            {
                if (c.Milliseconds <= 0 && c.Count <= 0)
                {
                    continue;
                }

                if (!options.IncludeWaits && c.Kind == CostKind.Wait)
                {
                    continue;
                }

                if (!LaneMatches(options.Thread, c.Lane))
                {
                    continue;
                }

                var key = c.NameId * 8 + (perName ? 0 : (int)c.Lane);
                if (_costIndex.TryGetValue(key, out var idx))
                {
                    var e = _costs[idx];
                    e.Ms += c.Milliseconds;
                    e.Count += c.Count;
                    _costs[idx] = e;
                }
                else
                {
                    _costIndex[key] = _costs.Count;
                    _costs.Add(new CostEntry { NameId = c.NameId, Kind = c.Kind, Lane = c.Lane, Ms = c.Milliseconds, Count = c.Count });
                }
            }
        }

        var inv = 1f / Math.Max(1, span);
        var n = _costs.Count;
        if (_sortKeys.Length < n)
        {
            _sortKeys = new float[n * 2];
            _sortItems = new int[n * 2];
        }

        for (var i = 0; i < n; i++)
        {
            var e = _costs[i];
            // Self is the exclusive time per (name, lane); Total sums a name over every lane it ran on.
            _sortKeys[i] = options.Sort == TopCostSort.Count ? e.Count : e.Ms;
            _sortItems[i] = i;
        }

        Array.Sort(_sortKeys, _sortItems, 0, n);
        var take = Math.Min(Math.Max(1, options.MaxRows), n);
        for (var r = 0; r < take; r++)
        {
            var e = _costs[_sortItems[n - 1 - r]];
            var value = (options.Sort == TopCostSort.Count ? e.Count : e.Ms) * inv;
            result.Rows.Add(new TopCostRow(e.NameId, e.Kind, e.Lane, value, e.Ms * inv, e.Count * inv));
        }

        result.Frames = span;
        result.FrameMs = (float)(frameMs / span);
        result.FrameSequence = frames[to].Sequence;
    }

    private static bool LaneMatches(TopCostThread filter, OverlayLane lane) => filter switch
    {
        TopCostThread.GuestMain => lane == OverlayLane.GuestMain,
        TopCostThread.Workers => lane == OverlayLane.GuestWorkers,
        TopCostThread.Render => lane == OverlayLane.Render,
        TopCostThread.Gpu => lane == OverlayLane.Gpu,
        _ => true,
    };

    // ---------------------------------------------------------------- threads

    private readonly Dictionary<int, int> _threadIndex = [];
    private readonly List<ThreadSample> _threadSums = [];

    /// <summary>
    /// Per-thread CPU ms per frame, averaged over the aggregate window (or the selected frame), busiest first.
    /// </summary>
    public void ComputeThreads(OverlayView view, List<ThreadSample> result)
    {
        result.Clear();
        var frames = view.Frames;
        if (view.Frame is null || frames.Count == 0)
        {
            return;
        }

        _threadIndex.Clear();
        _threadSums.Clear();
        int from, to = view.SelIndex >= 0 ? view.SelIndex : view.EndIndex;
        from = view.SelIndex >= 0 ? to : Math.Max(0, to - AggregateFrames + 1);
        for (var f = from; f <= to; f++)
        {
            foreach (var t in frames[f].Threads)
            {
                var key = t.NameId * 8 + (int)t.Lane;
                if (_threadIndex.TryGetValue(key, out var idx))
                {
                    var e = _threadSums[idx];
                    _threadSums[idx] = e with { BusyMs = e.BusyMs + t.BusyMs };
                }
                else
                {
                    _threadIndex[key] = _threadSums.Count;
                    _threadSums.Add(t);
                }
            }
        }

        var n = _threadSums.Count;
        if (_sortKeys.Length < n)
        {
            _sortKeys = new float[n * 2];
            _sortItems = new int[n * 2];
        }

        var inv = 1f / (to - from + 1);
        for (var i = 0; i < n; i++)
        {
            _sortKeys[i] = _threadSums[i].BusyMs;
            _sortItems[i] = i;
        }

        Array.Sort(_sortKeys, _sortItems, 0, n);
        for (var r = n - 1; r >= 0; r--)
        {
            var e = _threadSums[_sortItems[r]];
            result.Add(e with { BusyMs = e.BusyMs * inv });
        }
    }

    // ---------------------------------------------------------------- memory trend

    /// <summary>Appends a process-memory sample (GB) at most about twice a second.</summary>
    public void AddMemorySample(double gigabytes, long nowTicks, long minIntervalTicks)
    {
        if (!double.IsFinite(gigabytes) || gigabytes <= 0)
        {
            return;
        }

        if (_trendCount > 0 && nowTicks - _lastTrendTicks < minIntervalTicks)
        {
            return;
        }

        _lastTrendTicks = nowTicks;
        if (_trendCount == _trend.Length)
        {
            Array.Copy(_trend, 1, _trend, 0, _trend.Length - 1);
            _trendCount--;
        }

        _trend[_trendCount++] = (float)gigabytes;
    }

    public ReadOnlySpan<float> MemoryTrend => _trend.AsSpan(0, _trendCount);
}
