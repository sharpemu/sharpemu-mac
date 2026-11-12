// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Generic;
using System.Linq;
using SharpEmu.HLE.Diagnostics;

namespace SharpEmu.ShaderCompiler.Resources;

public static class MaterializationTally
{
    public enum Outcome
    {
        FastHit,
        FastMiss,
        FastFallback,
        KeptRevalidated,
        KeptStale,
        RevalidateRefused,
        CacheHit,
        CacheRebaseHit,
        TableRefresh,
        Volatile,
        Full,
        Failed,
        Count,
    }

    private static readonly ThreadCounter[] Calls = Create();
    private static readonly ThreadCounter[] Ticks = Create();
    private static readonly ThreadCounter[] TableWords = Create();
    private static readonly ThreadCounter[] Images = Create();

    private static ThreadCounter[] Create() => Enumerable.Range(0, (int)Outcome.Count).Select(_ => new ThreadCounter()).ToArray();

    // The timings only feed the report: one call in TimingSample reads the clock and its ticks count TimingSample
    // times (the call counts stay exact). Two clock reads per stage per draw were a few percent of the resolving
    // thread. SHARPEMU_MATERIALIZATION_TIMING_SAMPLE=1 times every call.
    public static readonly int TimingSample =
        int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_MATERIALIZATION_TIMING_SAMPLE"), out var sample)
            ? (int)System.Numerics.BitOperations.RoundUpToPowerOf2((uint)Math.Clamp(sample, 1, 1024))
            : 16;

    [ThreadStatic]
    private static uint t_timingCalls;

    /// <summary>The timestamp to time this call from, or 0 when this call is not sampled.</summary>
    public static long StartTiming() =>
        TimingSample == 1 || (++t_timingCalls & (uint)(TimingSample - 1)) == 0 ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;

    /// <summary>The scaled ticks since <paramref name="started"/> (from <see cref="StartTiming"/>), or 0.</summary>
    public static long ElapsedTicks(long started) =>
        started == 0 ? 0 : (System.Diagnostics.Stopwatch.GetTimestamp() - started) * TimingSample;

    public static void Note(Outcome outcome, long ticks, ResourceSnapshot? snapshot)
    {
        var index = (int)outcome;
        Calls[index].Increment();
        Ticks[index].Add(ticks);
        if (snapshot is not null)
        {
            TableWords[index].Add(snapshot.FlattenedResourceTable.Length);
            Images[index].Add(snapshot.Images.Length);
        }
    }

    public static Outcome FromCache(ResourceMaterializationCache.MaterializeOutcome outcome) => outcome switch
    {
        ResourceMaterializationCache.MaterializeOutcome.Hit => Outcome.CacheHit,
        ResourceMaterializationCache.MaterializeOutcome.RebaseHit => Outcome.CacheRebaseHit,
        ResourceMaterializationCache.MaterializeOutcome.TableRefresh => Outcome.TableRefresh,
        ResourceMaterializationCache.MaterializeOutcome.Volatile or ResourceMaterializationCache.MaterializeOutcome.BecameVolatile => Outcome.Volatile,
        ResourceMaterializationCache.MaterializeOutcome.Failed => Outcome.Failed,
        _ => Outcome.Full,
    };

    public static string TakeReport()
    {
        var parts = new List<string>();
        var frequency = (double)System.Diagnostics.Stopwatch.Frequency;
        for (var index = 0; index < (int)Outcome.Count; index++)
        {
            var calls = Calls[index].Take();
            var ticks = Ticks[index].Take();
            var words = TableWords[index].Take();
            var images = Images[index].Take();
            if (calls == 0)
            {
                continue;
            }

            parts.Add(FormattableString.Invariant($"{(Outcome)index}={calls}/{ticks * 1000.0 / frequency:F1}ms/{ticks * 1e9 / frequency / calls:F0}ns/words{(double)words / calls:F1}/images{(double)images / calls:F1}"));
        }

        return "materialize=" + string.Join(",", parts);
    }
}
