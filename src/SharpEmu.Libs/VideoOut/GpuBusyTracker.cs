// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.VideoOut;

/// <summary>
/// Measures how much of a wall-clock window the GPU spent executing, from the
/// [start, end] execution intervals of the command buffers it ran.
/// </summary>
/// <remarks>
/// Command buffers overlap (the next starts before the last one ends, or two queues run at
/// once), so summing their durations over-counts. The busy time is the length of the union of
/// the intervals, clipped to the window. Intervals are recorded once per command buffer, never
/// per draw. All times are seconds on one clock; the Metal host passes the GPUStartTime and
/// GPUEndTime of each command buffer, which share the CACurrentMediaTime clock.
/// </remarks>
internal sealed class GpuBusyTracker
{
    // A window rarely holds more than a few thousand command buffers; past this the list is
    // merged in place, and if even the merged list is full the interval is dropped.
    private const int MaxIntervals = 1 << 16;

    private readonly object _gate = new();
    private Interval[] _intervals = new Interval[256];
    private int _count;
    private double _countedUntil = double.NegativeInfinity;
    private bool _hasSamples;

    /// <summary>True once any interval has been recorded; false means the backend does not report GPU time.</summary>
    public bool HasSamples => Volatile.Read(ref _hasSamples);

    /// <summary>Records one command buffer's GPU execution interval.</summary>
    public void Record(double start, double end)
    {
        if (!double.IsFinite(start) || !double.IsFinite(end) || end <= start)
        {
            return;
        }

        lock (_gate)
        {
            if (_count == _intervals.Length)
            {
                if (_intervals.Length < MaxIntervals)
                {
                    Array.Resize(ref _intervals, _intervals.Length * 2);
                }
                else
                {
                    Merge();
                    if (_count == _intervals.Length)
                    {
                        return;
                    }
                }
            }

            _intervals[_count++] = new Interval(start, end);
            Volatile.Write(ref _hasSamples, true);
        }
    }

    /// <summary>
    /// Returns the GPU seconds recorded since the previous call, up to windowEnd, that no earlier
    /// call counted, at most the window's length. A command buffer reaches the tracker when its
    /// completion is handled, which can be seconds after it ran, so its time counts in the window
    /// it is recorded in rather than being clipped away. One still running at windowEnd keeps its
    /// remainder for the next window. Every instant counts once.
    /// </summary>
    public double TakeBusySeconds(double windowStart, double windowEnd)
    {
        if (!(windowEnd > windowStart))
        {
            return 0;
        }

        lock (_gate)
        {
            Merge();
            // The first window counts nothing from before it; later ones count from where the last left off.
            var countedUntil = double.IsNegativeInfinity(_countedUntil) ? windowStart : _countedUntil;
            var busy = 0.0;
            var kept = 0;
            for (var index = 0; index < _count; index++)
            {
                var interval = _intervals[index];
                busy += Math.Max(0, Math.Min(interval.End, windowEnd) - Math.Max(interval.Start, countedUntil));
                if (interval.End > windowEnd)
                {
                    _intervals[kept++] = new Interval(Math.Max(interval.Start, windowEnd), interval.End);
                }
                else
                {
                    countedUntil = Math.Max(countedUntil, interval.End);
                }
            }

            _count = kept;
            _countedUntil = countedUntil;
            return Math.Min(busy, windowEnd - windowStart);
        }
    }

    /// <summary>The busy fraction (0 to 1) of [windowStart, windowEnd]; see <see cref="TakeBusySeconds"/>.</summary>
    public double TakeBusyFraction(double windowStart, double windowEnd) =>
        windowEnd > windowStart ? TakeBusySeconds(windowStart, windowEnd) / (windowEnd - windowStart) : 0;

    // Sorts by start and folds overlapping or touching intervals together, in place.
    private void Merge()
    {
        if (_count < 2)
        {
            return;
        }

        Array.Sort(_intervals, 0, _count, StartComparer.Instance);
        var merged = 0;
        for (var index = 1; index < _count; index++)
        {
            ref var last = ref _intervals[merged];
            var next = _intervals[index];
            if (next.Start <= last.End)
            {
                if (next.End > last.End)
                {
                    last = new Interval(last.Start, next.End);
                }
            }
            else
            {
                _intervals[++merged] = next;
            }
        }

        _count = merged + 1;
    }

    private readonly record struct Interval(double Start, double End);

    private sealed class StartComparer : IComparer<Interval>
    {
        public static readonly StartComparer Instance = new();

        public int Compare(Interval left, Interval right) => left.Start.CompareTo(right.Start);
    }
}
