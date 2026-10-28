// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Globalization;
using System.Text;

namespace SharpEmu.HLE.GpuMemory;

/// <summary>
/// The calls that would cross between the guest process and a separate GPU process (the two-process
/// split), counted where they happen today. Each kind is one protocol in the split's design; the
/// rates decide which crossings may stay synchronous and which need a shared-memory fast path.
/// </summary>
public enum SplitCrossing
{
    // Guest CPU accesses that fault on a page the GPU side watches (resolved in the signal handler).
    FaultWrite,
    FaultRead,
    FaultOther,
    // Faults the GPU-side stores resolved (a round trip to the GPU process in the split).
    FaultToStores,
    // Managed (HLE) writes reported to the GPU side, before and after the bytes land.
    ManagedWrite,
    ManagedWriteLanded,
    // Page watches the GPU side arms and drops (mprotect requests to the guest process).
    WatchAdd,
    WatchRemove,
    // Guest mapping changes the GPU side must mirror.
    MappingChange,
    // Guest -> GPU submissions.
    SubmitCommandStream,
    SubmitDone,
    FlipSubmit,
    RunAfterPending,
    // GPU -> guest events.
    Interrupt,
    FlipComplete,
    // Guest state the GPU side reads (AGC shader headers).
    ShaderHeaderLookup,
    // Split only. Managed writes under a page the GPU side watches (the interest summary says so),
    // the blocking CpuWrite calls those made (timed), and the waits for the guest process to
    // confirm a watch's protection (timed).
    ManagedWriteInterest,
    ManagedWriteCall,
    ProtectWait,
    // Split only. Watched managed writes with no pending writeback under them, sent as a record
    // instead of the call.
    ManagedWriteRecord,
}

public static class SplitCrossingStats
{
    // SHARPEMU_SPLIT_STATS=1: a [SPLIT-X] line every five seconds with each crossing's rate and the
    // time the GPU-side stores took inside faults (the latency a cross-process round trip adds to).
    public static readonly bool Enabled = string.Equals(
        Environment.GetEnvironmentVariable("SHARPEMU_SPLIT_STATS"), "1", StringComparison.Ordinal);

    private static readonly int KindCount = Enum.GetValues<SplitCrossing>().Length;
    private static readonly long[] Counts = new long[KindCount];
    private static readonly long[] Reported = new long[KindCount];
    private static readonly long[] TimeTicks = new long[KindCount];
    private static readonly long[] ReportedTimeTicks = new long[KindCount];
    private static long _storeFaultTicks;
    private static long _slowStoreFaults;
    private static long _maxStoreFaultTicks;
    private static Timer? _timer;
    private static long _lastReport = Stopwatch.GetTimestamp();

    // A fault the stores held longer than this would also block on the GPU process in the split.
    private static readonly long SlowFaultTicks = Stopwatch.Frequency / 20_000;

    public static void Count(SplitCrossing kind)
    {
        if (!Enabled)
        {
            return;
        }

        Interlocked.Increment(ref Counts[(int)kind]);
        EnsureTimer();
    }

    /// <summary>Adds the duration of one counted crossing of <paramref name="kind"/>; reported as its average.</summary>
    public static void Time(SplitCrossing kind, long ticks)
    {
        if (Enabled)
        {
            Interlocked.Add(ref TimeTicks[(int)kind], ticks);
        }
    }

    /// <summary>Records how long the GPU-side stores held one fault.</summary>
    public static void StoreFaultTime(long ticks)
    {
        if (!Enabled)
        {
            return;
        }

        Interlocked.Add(ref _storeFaultTicks, ticks);
        if (ticks > SlowFaultTicks)
        {
            Interlocked.Increment(ref _slowStoreFaults);
        }

        long max;
        while (ticks > (max = Interlocked.Read(ref _maxStoreFaultTicks)) &&
            Interlocked.CompareExchange(ref _maxStoreFaultTicks, ticks, max) != max)
        {
        }
    }

    private static void EnsureTimer()
    {
        if (Volatile.Read(ref _timer) is not null)
        {
            return;
        }

        var timer = new Timer(static _ => Report(), null, 5000, 5000);
        if (Interlocked.CompareExchange(ref _timer, timer, null) is not null)
        {
            timer.Dispose();
        }
    }

    private static void Report()
    {
        var now = Stopwatch.GetTimestamp();
        var seconds = Math.Max((now - Interlocked.Exchange(ref _lastReport, now)) / (double)Stopwatch.Frequency, 1e-3);
        var line = new StringBuilder("[SPLIT-X]");
        for (var kind = 0; kind < KindCount; kind++)
        {
            var total = Interlocked.Read(ref Counts[kind]);
            var delta = total - Reported[kind];
            Reported[kind] = total;
            line.Append(CultureInfo.InvariantCulture, $" {(SplitCrossing)kind}={delta / seconds:F0}/s");
            var timed = Interlocked.Read(ref TimeTicks[kind]);
            var timedDelta = timed - ReportedTimeTicks[kind];
            ReportedTimeTicks[kind] = timed;
            if (timed != 0)
            {
                line.Append(CultureInfo.InvariantCulture,
                    $"(avg {timedDelta * 1e6 / Stopwatch.Frequency / Math.Max(delta, 1):F1}us)");
            }
        }

        var faults = Math.Max(Interlocked.Read(ref Reported[(int)SplitCrossing.FaultToStores]), 1);
        line.Append(CultureInfo.InvariantCulture,
            $" store_us_avg={Interlocked.Read(ref _storeFaultTicks) * 1e6 / Stopwatch.Frequency / faults:F1}" +
            $" store_us_max={Interlocked.Exchange(ref _maxStoreFaultTicks, 0) * 1e6 / Stopwatch.Frequency:F0}" +
            $" slow_faults_total={Interlocked.Read(ref _slowStoreFaults)}");
        Console.Error.WriteLine(line.ToString());
    }
}
