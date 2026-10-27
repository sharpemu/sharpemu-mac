// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;

namespace SharpEmu.ShaderCompiler.Resources;

public static class ResourceMaterializationProfile
{
    // CacheKey, CacheFind, RecorderWrap and EntryBuild are the cache's own work
    // around a materialization: hashing the user data into a key, looking the key up
    // and comparing the entry, wrapping the three readers so the words are recorded,
    // and building the entry that records them. A miss measured about a millisecond
    // for a dozen guest words, so what the cache itself costs has to be separable
    // from what the walk costs.
    public enum Phase
    {
        Total,
        Snapshot,
        DeviceAddressRanges,
        Specialization,
        OutputAssembly,
        GuestRead,
        CleanGuestRead,
        CacheKey,
        CacheFind,
        RecorderWrap,
        EntryBuild,
        RecorderSort,
        // The three parts of the snapshot walk, which is where a miss spends almost all of
        // its time: the branch conditions that decide which descriptor sources are live, the
        // descriptor words themselves, and the flattened table, one graph value per word.
        ActiveSources,
        DescriptorWords,
        TableEvaluation,
        SourceEvaluation,
        BranchSelection,
        DescriptorPrefetch,
        DescriptorEvaluation,
        TablePrefetch,
        CompiledEvaluation,
        CompiledTableEvaluation,
        SpecializationGuardHit,
        CompiledOutputAllocation,
        CompiledBody,
        CompiledWordRead,
        CompiledGroupRead,
        ReadRecording,
        ConsumerLookup,
        ConsumerGraph,
        AddressRebasing,
        HitValidation,
        InPlaceComparison,
        HitSnapshot,
        Count,
    }

    public static readonly bool Enabled = IsEnabled(
        Environment.GetEnvironmentVariable("SHARPEMU_PROFILE_PERFORMANCE"),
        Environment.GetEnvironmentVariable("SHARPEMU_PROFILE_PERFORMANCE_FRAME_TRACE"),
        Environment.GetEnvironmentVariable("SHARPEMU_PROFILE_MATERIALIZATION"));
    private static readonly long[] _elapsedTicks = new long[(int)Phase.Count];
    private static readonly long[] _calls = new long[(int)Phase.Count];
    private static readonly long[] _allocatedBytesByPhase = new long[(int)Phase.Count];
    private static long _branchEvaluations;
    private static long _descriptorSources;
    private static long _inactiveSources;
    private static long _tableWords;
    private static long _descriptorWords;
    private static long _compiledEvaluators;
    private static long _interpretedEvaluators;

    internal enum ReadCounter
    {
        GroupAttempt, GroupIneligible, GroupRefused, GroupSucceeded, GroupWords,
        ScalarAddress, ScalarBuffer, OrderedReadLists, MergedReadLists, SortedReadLists, Count
    }
    private static readonly long[] _readCounts = new long[(int)ReadCounter.Count];

    internal static void CountRead(ReadCounter counter, int amount = 1)
    {
        if (Enabled) Interlocked.Add(ref _readCounts[(int)counter], amount);
    }

    // One tiny counter record per participating thread. Hit recording never walks ranges
    // and uses no atomics; reporting reads a cumulative, approximate live snapshot.
    private sealed class HitCounts
    {
        internal long Normal, Recall, RebasedNormal, RebasedRecall, Ranges, Bytes;
    }
    private static readonly System.Collections.Concurrent.ConcurrentBag<HitCounts> _hitCounts = new();
    [ThreadStatic] private static HitCounts? _threadHitCounts;

    internal static void RecordCacheHit(bool recalled, bool rebased, int ranges, int bytes)
    {
        if (!Enabled) return;
        var counts = _threadHitCounts;
        if (counts is null)
        {
            counts = _threadHitCounts = new HitCounts();
            _hitCounts.Add(counts);
        }
        if (recalled) counts.Recall++;
        else counts.Normal++;
        if (rebased)
        {
            if (recalled) counts.RebasedRecall++;
            else counts.RebasedNormal++;
        }
        counts.Ranges += ranges;
        counts.Bytes += bytes;
    }

    // The frame trace prints a line per guest flow and per mutex handoff, which on a
    // submission-heavy title costs more than the materializations being measured, so these
    // counters also answer to a switch of their own.
    internal static bool IsEnabled(string? performance, string? frameTrace, string? materialization) =>
        materialization == "1" || (performance == "1" && frameTrace == "1");

    public static Scope Measure(Phase phase) => Enabled ? new Scope(phase) : default;

    [ThreadStatic] private static int[]? _sampleCalls;
    internal const int SampleStride = 64;

    internal static Scope MeasureSampled(Phase phase)
    {
        if (!Enabled) return default;
        var calls = _sampleCalls ??= new int[(int)Phase.Count];
        return (unchecked(calls[(int)phase]++) & (SampleStride - 1)) == 0 ? new Scope(phase) : default;
    }

    public readonly struct Scope : IDisposable
    {
        private readonly Phase _phase;
        private readonly long _started;
        private readonly long _initialAllocatedBytes;

        internal Scope(Phase phase)
        {
            _phase = phase;
            _initialAllocatedBytes = phase is Phase.Total or Phase.RecorderWrap or Phase.EntryBuild
                ? GC.GetAllocatedBytesForCurrentThread()
                : 0;
            _started = Stopwatch.GetTimestamp();
        }

        public void Dispose()
        {
            if (_started == 0) return;
            Interlocked.Add(ref _elapsedTicks[(int)_phase], Stopwatch.GetTimestamp() - _started);
            Interlocked.Increment(ref _calls[(int)_phase]);
            if (_initialAllocatedBytes != 0)
                Interlocked.Add(
                    ref _allocatedBytesByPhase[(int)_phase],
                    GC.GetAllocatedBytesForCurrentThread() - _initialAllocatedBytes);
        }
    }

    // How many graph values one walk evaluates, so the measured milliseconds can be read per
    // word, and whether the plan had an IL-compiled evaluator or was interpreted.
    internal static void RecordWalk(int tableWords, int descriptorWords, bool compiled)
    {
        if (!Enabled) return;
        Interlocked.Add(ref _tableWords, tableWords);
        Interlocked.Add(ref _descriptorWords, descriptorWords);
        if (compiled) Interlocked.Increment(ref _compiledEvaluators);
        else Interlocked.Increment(ref _interpretedEvaluators);
    }

    internal static void RecordActivity(bool[] activeSources)
    {
        if (!Enabled) return;
        Interlocked.Increment(ref _branchEvaluations);
        Interlocked.Add(ref _descriptorSources, activeSources.Length);
        var inactiveCount = 0;
        foreach (var active in activeSources)
            if (!active) inactiveCount++;
        Interlocked.Add(ref _inactiveSources, inactiveCount);
    }

    // Cumulative inclusive times overlap: memory reads are also part of snapshot evaluation.
    public static void WriteReport() => WriteReport(Console.Error);

    internal static void WriteReport(TextWriter output)
    {
        if (!Enabled) return;
        var parts = new List<string>();
        for (var index = 0; index < (int)Phase.Count; index++)
            parts.Add(FormattableString.Invariant($"{(Phase)index}={Interlocked.Read(ref _elapsedTicks[index]) * 1000.0 / Stopwatch.Frequency:F3}ms/n{Interlocked.Read(ref _calls[index])}"));
        output.WriteLine($"[PERF][RESOURCE_MATERIALIZATION] cumulative=1 inclusive=1 sample_stride={SampleStride} sampled_phases=CompiledOutputAllocation,CompiledBody,CompiledWordRead,CompiledGroupRead,ReadRecording,ConsumerLookup,AddressRebasing,HitValidation,InPlaceComparison,HitSnapshot allocated_bytes={Interlocked.Read(ref _allocatedBytesByPhase[(int)Phase.Total])} wrap_bytes={Interlocked.Read(ref _allocatedBytesByPhase[(int)Phase.RecorderWrap])} build_bytes={Interlocked.Read(ref _allocatedBytesByPhase[(int)Phase.EntryBuild])} branch_evaluations={Interlocked.Read(ref _branchEvaluations)} descriptor_sources={Interlocked.Read(ref _descriptorSources)} inactive_sources={Interlocked.Read(ref _inactiveSources)} table_words={Interlocked.Read(ref _tableWords)} descriptor_words={Interlocked.Read(ref _descriptorWords)} compiled_walks={Interlocked.Read(ref _compiledEvaluators)} interpreted_walks={Interlocked.Read(ref _interpretedEvaluators)} {string.Join(" ", parts)}");
        long normal = 0, recalled = 0, rebasedNormal = 0, rebasedRecall = 0, ranges = 0, bytes = 0;
        foreach (var counts in _hitCounts)
        {
            normal += Volatile.Read(ref counts.Normal);
            recalled += Volatile.Read(ref counts.Recall);
            rebasedNormal += Volatile.Read(ref counts.RebasedNormal);
            rebasedRecall += Volatile.Read(ref counts.RebasedRecall);
            ranges += Volatile.Read(ref counts.Ranges);
            bytes += Volatile.Read(ref counts.Bytes);
        }
        output.WriteLine($"[PERF][RESOURCE_HIT_COUNTS] cumulative=1 sampled=0 live_snapshot=1 normal={normal} recalled={recalled} normal_active_rebase={rebasedNormal} recalled_active_rebase={rebasedRecall} recorded_ranges={ranges} recorded_bytes={bytes}");
        var reads = new List<string>();
        for (var index = 0; index < (int)ReadCounter.Count; index++)
            reads.Add($"{(ReadCounter)index}={Interlocked.Read(ref _readCounts[index])}");
        output.WriteLine($"[PERF][RESOURCE_READ_COUNTS] cumulative=1 sampled=0 {string.Join(" ", reads)}");
    }
}
