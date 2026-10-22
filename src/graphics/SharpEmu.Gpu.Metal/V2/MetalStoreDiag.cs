// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;

namespace SharpEmu.Gpu.Metal.V2;

// SHARPEMU_METAL_V2_STORE_DIAG=1: where the time of a store waiting for its writeback goes. The
// worker marks what it blocks on; a store reads the marks when it asks and when the worker takes
// the request. Off by default, and then every scope is the default struct.
internal static class MetalStoreDiag
{
    internal static readonly bool On = Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_STORE_DIAG") == "1";

    internal enum Activity
    {
        Recording,
        LibraryWait,
        LibraryWaitInGraphics,
        LibraryWaitInCompute,
        GraphicsPipelineBuild,
        ComputePipelineBuild,
        GpuWait,
        CommittedWait,
        Count,
    }

    private static readonly long[] Ticks = new long[(int)Activity.Count];
    private static int _current;
    private static long _currentStart = Stopwatch.GetTimestamp();
    private static long _boundaries;
    private static long _longestBoundaryGapTicks;
    private static long _lastBoundary = Stopwatch.GetTimestamp();

    internal readonly struct Scope : IDisposable
    {
        private readonly int _previous;
        private readonly bool _active;

        internal Scope(int previous)
        {
            _previous = previous;
            _active = true;
        }

        public void Dispose()
        {
            if (_active)
            {
                Switch(_previous);
            }
        }
    }

    // Worker only.
    internal static Scope Enter(Activity activity)
    {
        if (!On)
        {
            return default;
        }

        var previous = _current;
        if (activity == Activity.LibraryWait)
        {
            // Told apart by the build that asks, which is what could do without the wait.
            activity = (Activity)previous switch
            {
                Activity.GraphicsPipelineBuild => Activity.LibraryWaitInGraphics,
                Activity.ComputePipelineBuild => Activity.LibraryWaitInCompute,
                _ => Activity.LibraryWait,
            };
        }

        Switch((int)activity);
        return new Scope(previous);
    }

    private static void Switch(int next)
    {
        var now = Stopwatch.GetTimestamp();
        Volatile.Write(ref Ticks[_current], Ticks[_current] + (now - _currentStart));
        _currentStart = now;
        Volatile.Write(ref _current, next);
    }

    // Worker only, at each relay run (a packet boundary or a draw's start).
    internal static void NoteBoundary()
    {
        if (!On)
        {
            return;
        }

        var now = Stopwatch.GetTimestamp();
        var gap = now - _lastBoundary;
        _lastBoundary = now;
        _boundaries++;
        if (gap > Volatile.Read(ref _longestBoundaryGapTicks))
        {
            Volatile.Write(ref _longestBoundaryGapTicks, gap);
        }
    }

    internal readonly record struct Mark(long At, long[] Ticks, int Current, long CurrentSince, long Boundaries, long LastBoundary);

    // Any thread; the worker's counters are read without a lock, so a figure is an estimate.
    internal static Mark Snapshot()
    {
        var copy = new long[(int)Activity.Count];
        for (var index = 0; index < copy.Length; index++)
        {
            copy[index] = Volatile.Read(ref Ticks[index]);
        }

        var current = Volatile.Read(ref _current);
        var since = Volatile.Read(ref _currentStart);
        var now = Stopwatch.GetTimestamp();
        copy[current] += Math.Max(0, now - since);
        return new Mark(now, copy, current, since, Volatile.Read(ref _boundaries), Volatile.Read(ref _lastBoundary));
    }

    private static double Ms(long ticks) => ticks * 1000.0 / Stopwatch.Frequency;

    internal static string Describe(in Mark post, in Mark take)
    {
        var text = $"post_in={(Activity)post.Current}/{Ms(post.At - post.CurrentSince):F0}ms " +
            $"since_boundary={Ms(post.At - post.LastBoundary):F0}ms boundaries={take.Boundaries - post.Boundaries} between=[";
        for (var index = 0; index < (int)Activity.Count; index++)
        {
            text += $"{(Activity)index}={Ms(take.Ticks[index] - post.Ticks[index]):F0} ";
        }

        return text.TrimEnd() + "]";
    }

    private static int _lines;

    internal static void Report(string message)
    {
        if (Interlocked.Increment(ref _lines) <= 4000)
        {
            Console.Error.WriteLine($"[METAL-V2][STOREDIAG] {message}");
        }
    }
}
