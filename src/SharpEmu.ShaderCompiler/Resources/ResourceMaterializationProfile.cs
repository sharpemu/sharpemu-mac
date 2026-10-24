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
        BatchedWalk,
        BatchedPendingProbe,
        BatchedResidentRead,
        BatchedScalarRead,
        Count,
    }

    public static readonly bool Enabled = IsEnabled(
        Environment.GetEnvironmentVariable("SHARPEMU_PROFILE_PERFORMANCE"),
        Environment.GetEnvironmentVariable("SHARPEMU_PROFILE_PERFORMANCE_FRAME_TRACE"),
        Environment.GetEnvironmentVariable("SHARPEMU_PROFILE_MATERIALIZATION"));
    private static readonly long[] _elapsedTicks = new long[(int)Phase.Count];
    private static readonly long[] _calls = new long[(int)Phase.Count];
    private static readonly long[] _allocatedBytesByPhase = new long[(int)Phase.Count];
    private static readonly long[] _workerElapsedTicks = new long[(int)Phase.Count];
    private static readonly long[] _workerCalls = new long[(int)Phase.Count];
    [ThreadStatic] private static bool _workerContext;

    // The caller already knows whether this call is inside the render worker's
    // profiling scopes. Keep its inclusive timers separate from prefetch threads.
    public static WorkerScope WorkerContext(bool worker)
    {
        if (!Enabled) return default;
        var previous = _workerContext;
        _workerContext = worker;
        return new(previous, true);
    }

    public readonly struct WorkerScope(bool previous, bool active) : IDisposable
    {
        public void Dispose() { if (active) _workerContext = previous; }
    }
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
        ScalarAddress, ScalarBuffer, Count
    }
    private static readonly long[] _readCounts = new long[(int)ReadCounter.Count];

    internal static void CountRead(ReadCounter counter, int amount = 1)
    {
        if (Enabled) Interlocked.Add(ref _readCounts[(int)counter], amount);
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
        private readonly bool _worker;

        internal Scope(Phase phase)
        {
            _phase = phase;
            _worker = _workerContext;
            _initialAllocatedBytes = phase is Phase.Total or Phase.RecorderWrap or Phase.EntryBuild
                ? GC.GetAllocatedBytesForCurrentThread()
                : 0;
            _started = Stopwatch.GetTimestamp();
        }

        public void Dispose()
        {
            if (_started == 0) return;
            var elapsed = Stopwatch.GetTimestamp() - _started;
            Interlocked.Add(ref _elapsedTicks[(int)_phase], elapsed);
            Interlocked.Increment(ref _calls[(int)_phase]);
            if (_worker)
            {
                Interlocked.Add(ref _workerElapsedTicks[(int)_phase], elapsed);
                Interlocked.Increment(ref _workerCalls[(int)_phase]);
            }
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
        output.WriteLine($"[PERF][RESOURCE_MATERIALIZATION] cumulative=1 inclusive=1 sample_stride={SampleStride} sampled_phases=CompiledOutputAllocation,CompiledBody,CompiledWordRead,CompiledGroupRead,ReadRecording,ConsumerLookup,BatchedPendingProbe,BatchedResidentRead,BatchedScalarRead allocated_bytes={Interlocked.Read(ref _allocatedBytesByPhase[(int)Phase.Total])} wrap_bytes={Interlocked.Read(ref _allocatedBytesByPhase[(int)Phase.RecorderWrap])} build_bytes={Interlocked.Read(ref _allocatedBytesByPhase[(int)Phase.EntryBuild])} branch_evaluations={Interlocked.Read(ref _branchEvaluations)} descriptor_sources={Interlocked.Read(ref _descriptorSources)} inactive_sources={Interlocked.Read(ref _inactiveSources)} table_words={Interlocked.Read(ref _tableWords)} descriptor_words={Interlocked.Read(ref _descriptorWords)} compiled_walks={Interlocked.Read(ref _compiledEvaluators)} interpreted_walks={Interlocked.Read(ref _interpretedEvaluators)} {string.Join(" ", parts)}");
        var reads = new List<string>();
        for (var index = 0; index < (int)ReadCounter.Count; index++)
            reads.Add($"{(ReadCounter)index}={Interlocked.Read(ref _readCounts[index])}");
        output.WriteLine($"[PERF][RESOURCE_READ_COUNTS] cumulative=1 sampled=0 {string.Join(" ", reads)}");
        var workerParts = new List<string>();
        for (var index = 0; index < (int)Phase.Count; index++)
            workerParts.Add(FormattableString.Invariant($"{(Phase)index}={Interlocked.Read(ref _workerElapsedTicks[index]) * 1000.0 / Stopwatch.Frequency:F3}ms/n{Interlocked.Read(ref _workerCalls[index])}"));
        output.WriteLine($"[PERF][RESOURCE_MATERIALIZATION_WORKER] cumulative=1 inclusive=1 sample_stride={SampleStride} {string.Join(" ", workerParts)}");
    }
}
