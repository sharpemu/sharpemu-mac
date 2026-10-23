// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Globalization;
using SharpEmu.HLE.GuestMemory;

namespace SharpEmu.Gpu.Metal.V2;

// Diagnostic fingerprints, never cache keys authorizing rendering. Two independent
// 64-bit accumulators keep a compact immutable capture; collisions remain possible.
internal record struct RecipeHash(ulong A, ulong B)
{
    public void Add(ulong word)
    {
        A = unchecked((A ^ word ^ 0x9e3779b97f4a7c15UL) * 1099511628211UL);
        B = unchecked(((B << 13) | (B >> 51)) + word + 0x517cc1b727220a95UL);
        B = unchecked(B * 0x94d049bb133111ebUL);
    }
    public override readonly string ToString() => $"{A:x16}{B:x16}";
}

// Explicit mapping geometry, usable by deterministic capture tests without a device.
internal static class RecipeAddressExpression
{
    internal static bool TryResolve(ReadOnlySpan<ViewRecord> views, ulong address, ulong size,
        out RecipeHash allocation, out ulong offset)
    {
        allocation = default; offset = 0;
        if (address == 0 || size == 0 || size > ulong.MaxValue - address) return false;
        var low = 0; var high = views.Length - 1;
        while (low <= high)
        {
            var middle = low + ((high - low) >> 1); var view = views[middle];
            if (address < view.Address) high = middle - 1;
            else if (address - view.Address >= view.Size) low = middle + 1;
            else
            {
                offset = address - view.Address;
                if (size > view.Size - offset) return false;
                allocation.Add(view.Address); allocation.Add(view.Size); allocation.Add(view.Offset);
                return true;
            }
        }
        return false;
    }
}

internal enum RecipeChange { Exact, Relocation, Semantic }

// Values contains every non-address input; Locations contains captured allocation +
// offset expressions. Allocation identities also enter Values so relocation can only
// change offsets inside the SAME known allocation. Unknown pointers enter Values verbatim.
internal readonly record struct RecipeSignature(RecipeHash Pass, RecipeHash Structure,
    RecipeHash Values, RecipeHash Locations, RecipeHash Dependencies);
internal readonly record struct RecipeObservation(RecipeSignature Signature, long MaterializationTicks,
    int Materializations, int Emissions, int UnknownAddresses, int UnobservedContents, long PassId,
    long FrameId = 0, long IssueId = 0, int QueueId = 0, bool Compute = false, bool Complete = true);

internal sealed class MetalRecipeClassifier
{
    internal const int HistoryFrames = 4;
    internal const int MaximumIssues = 65536;
    private const int MaximumReportedPasses = 1024;
    private readonly record struct PassKey(long Frame, long Id, bool Compute, RecipeHash Shape);
    private readonly record struct MatchKeys(RecipeHash Exact, RecipeHash Relocated, RecipeHash Values, bool Complete)
    {
        public static MatchKeys Of(in RecipeSignature s, bool complete)
        {
            var values = new RecipeHash();
            values.Add(s.Pass.A); values.Add(s.Pass.B); values.Add(s.Structure.A); values.Add(s.Structure.B);
            values.Add(s.Values.A); values.Add(s.Values.B);
            var relocated = values; relocated.Add(s.Dependencies.A); relocated.Add(s.Dependencies.B);
            var exact = relocated; exact.Add(s.Locations.A); exact.Add(s.Locations.B);
            return new(exact, relocated, values, complete);
        }
    }
    private sealed class Frame
    {
        public readonly List<RecipeObservation> Issues = new(1024);
        public readonly List<MatchKeys> Keys = new(1024);
        public long Id = -1;
    }
    private readonly Frame[] _frames = Enumerable.Range(0, HistoryFrames + 1).Select(_ => new Frame()).ToArray();
    private readonly Dictionary<RecipeHash, int> _exact = new(1024);
    private readonly Dictionary<RecipeHash, int> _relocated = new(1024);
    private readonly Dictionary<RecipeHash, int> _valueOnly = new(1024);
    private readonly Dictionary<PassKey, Counters[]> _passes = new();
    private readonly Counters[] _totals = NewCounters();
    private RecipeChange[] _classes = new RecipeChange[1024];
    private bool[] _valueCredits = new bool[1024];
    private int[] _costOrder = new int[1024];
    private sealed class CostComparer : IComparer<int>
    {
        public List<RecipeObservation> Issues = null!;
        public int Compare(int left, int right)
        {
            var cost = Issues[right].MaterializationTicks.CompareTo(Issues[left].MaterializationTicks);
            return cost != 0 ? cost : left.CompareTo(right);
        }
    }
    private readonly CostComparer _costComparer = new();
    private long _frame;
    private long _dropped;
    private long _captureTicks;
    private long _captureCalls;
    private long _matchingTicks;
    private long _reportTicks;
    private long _diagnosticBytes;
    private long _issues, _materializations, _emissions, _unknownAddresses, _unobservedContents, _incompleteIssues;
    private sealed class Counters
    {
        public readonly long[] Issues = new long[3];
        public readonly long[] Ticks = new long[3];
        public readonly long[] Calls = new long[3];
        public readonly long[] UnobservedIssues = new long[3];
        public long Comparisons, ValueRepeats, ValueRepeatTicks, ValueRepeatCalls, PassInstances;
        private long _lastFrame = -1, _lastPass = -1;
        public void Add(RecipeChange change, in RecipeObservation issue)
        {
            Issues[(int)change]++;
            Ticks[(int)change] += issue.MaterializationTicks;
            Calls[(int)change] += issue.Materializations;
            Comparisons++;
            if (issue.UnobservedContents != 0) UnobservedIssues[(int)change]++;
            if (issue.FrameId != _lastFrame || issue.PassId != _lastPass)
            {
                PassInstances++; _lastFrame = issue.FrameId; _lastPass = issue.PassId;
            }
        }
    }
    private static Counters[] NewCounters() => Enumerable.Range(0, HistoryFrames).Select(_ => new Counters()).ToArray();
    // Call only on the owning consumer, after an issued prefix acknowledgement.
    // A caller owns the returned immutable values; no executor/cache storage escapes.
    internal RecipeObservation[] CaptureCurrentFrame() => _frames[(int)(_frame % _frames.Length)].Issues.ToArray();
    internal long FrameId => _frame;
    internal long CaptureTicks => _captureTicks;
    internal void NoteCost(long start, long bytes) { _captureCalls++; _captureTicks += Stopwatch.GetTimestamp() - start; _diagnosticBytes += GC.GetAllocatedBytesForCurrentThread() - bytes; }
    internal void Record(in RecipeObservation issue)
    {
        if (!issue.Complete) _incompleteIssues++;
        _issues++; _materializations += issue.Materializations; _emissions += issue.Emissions;
        _unknownAddresses += issue.UnknownAddresses; _unobservedContents += issue.UnobservedContents;
        var frame = _frames[(int)(_frame % _frames.Length)];
        if (frame.Issues.Count == MaximumIssues) { _dropped++; return; }
        frame.Issues.Add(issue);
        frame.Keys.Add(MatchKeys.Of(issue.Signature, issue.Complete));
    }
    private static void Increment(Dictionary<RecipeHash, int> inventory, RecipeHash key)
    {
        ref var count = ref System.Runtime.InteropServices.CollectionsMarshal.GetValueRefOrAddDefault(inventory, key, out _);
        count++;
    }
    private static bool Take(Dictionary<RecipeHash, int> inventory, RecipeHash key)
    {
        ref var count = ref System.Runtime.InteropServices.CollectionsMarshal.GetValueRefOrNullRef(inventory, key);
        if (System.Runtime.CompilerServices.Unsafe.IsNullRef(ref count) || count == 0) return false;
        count--;
        return true;
    }
    // Compare multisets, with an exact-first sweep before allowing relocations to consume
    // the remaining inventory. Neither issue ordinal nor hash-set membership grants N:1 hits.
    internal static void Compare(ReadOnlySpan<RecipeObservation> current, ReadOnlySpan<RecipeObservation> previous,
        Span<RecipeChange> result)
    {
        var classifier = new MetalRecipeClassifier();
        var currentKeys = new MatchKeys[current.Length]; var previousKeys = new MatchKeys[previous.Length];
        for (var i = 0; i < current.Length; i++) currentKeys[i] = MatchKeys.Of(current[i].Signature, current[i].Complete);
        for (var i = 0; i < previous.Length; i++) previousKeys[i] = MatchKeys.Of(previous[i].Signature, previous[i].Complete);
        classifier.CompareInto(currentKeys, previousKeys, result);
    }
    private void CompareInto(ReadOnlySpan<MatchKeys> current, ReadOnlySpan<MatchKeys> previous,
        Span<RecipeChange> result)
    {
        _exact.Clear(); _relocated.Clear(); _valueOnly.Clear();
        foreach (ref readonly var issue in previous)
        {
            if (!issue.Complete) continue;
            Increment(_exact, issue.Exact);
            Increment(_relocated, issue.Relocated);
            Increment(_valueOnly, issue.Values);
        }
        for (var i = 0; i < current.Length; i++)
        {
            var key = current[i];
            result[i] = key.Complete && Take(_exact, key.Exact) ? RecipeChange.Exact : RecipeChange.Semantic;
            if (result[i] == RecipeChange.Exact) Take(_relocated, key.Relocated);
        }
        for (var i = 0; i < current.Length; i++)
            if (current[i].Complete && result[i] != RecipeChange.Exact && Take(_relocated, current[i].Relocated))
                result[i] = RecipeChange.Relocation;
    }
    internal void EndFrame()
    {
        var start = Stopwatch.GetTimestamp();
        var bytes = GC.GetAllocatedBytesForCurrentThread();
        var current = _frames[(int)(_frame % _frames.Length)];
        if (_classes.Length < current.Issues.Count)
        {
            Array.Resize(ref _classes, MaximumIssues);
            Array.Resize(ref _valueCredits, MaximumIssues);
            Array.Resize(ref _costOrder, MaximumIssues);
        }
        // The dependency-free time bound must select the most expensive eligible
        // occurrences. Ordinal selection could credit a cheap semantic candidate
        // and discard a later expensive exact candidate from the SAME inventory.
        for (var i = 0; i < current.Issues.Count; i++) _costOrder[i] = i;
        _costComparer.Issues = current.Issues;
        Array.Sort(_costOrder, 0, current.Issues.Count, _costComparer);
        var observations = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(current.Issues);
        var keys = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(current.Keys);
        for (var lag = 1; lag <= HistoryFrames && lag <= _frame; lag++)
        {
            var previous = _frames[(int)((_frame - lag) % _frames.Length)];
            CompareInto(keys, System.Runtime.InteropServices.CollectionsMarshal.AsSpan(previous.Keys), _classes);
            for (var rank = 0; rank < observations.Length; rank++)
            {
                var i = _costOrder[rank];
                _valueCredits[i] = keys[i].Complete && Take(_valueOnly, keys[i].Values);
            }
            for (var i = 0; i < observations.Length; i++)
            {
                ref readonly var issue = ref observations[i];
                var pass = new PassKey(issue.FrameId, issue.PassId, issue.Compute, issue.Signature.Pass);
                if (!_passes.TryGetValue(pass, out var counters))
                {
                    if (_passes.Count >= MaximumReportedPasses) pass = new(-1, -1, false, default);
                    if (!_passes.TryGetValue(pass, out counters)) _passes.Add(pass, counters = NewCounters());
                }
                counters[lag - 1].Add(_classes[i], issue);
                _totals[lag - 1].Add(_classes[i], issue);
                if (_valueCredits[i])
                {
                    counters[lag - 1].ValueRepeats++;
                    _totals[lag - 1].ValueRepeats++;
                    counters[lag - 1].ValueRepeatTicks += issue.MaterializationTicks;
                    _totals[lag - 1].ValueRepeatTicks += issue.MaterializationTicks;
                    counters[lag - 1].ValueRepeatCalls += issue.Materializations;
                    _totals[lag - 1].ValueRepeatCalls += issue.Materializations;
                }
            }
        }
        current.Id = _frame++;
        _frames[(int)(_frame % _frames.Length)].Issues.Clear();
        _frames[(int)(_frame % _frames.Length)].Keys.Clear();
        _matchingTicks += Stopwatch.GetTimestamp() - start;
        _diagnosticBytes += GC.GetAllocatedBytesForCurrentThread() - bytes;
    }
    internal void Report(TextWriter output)
    {
        var start = Stopwatch.GetTimestamp();
        var bytes = GC.GetAllocatedBytesForCurrentThread();
        output.WriteLine(FormattableString.Invariant($"[METAL-V2][RECIPE] schema=1 scope=observed_inputs dependency_guard=global candidate_only=1 cumulative=1 frames={_frame} partial_issues={_frames[(int)(_frame % _frames.Length)].Issues.Count} issues={_issues} emissions={_emissions} resmat_calls={_materializations} incomplete_issues={_incompleteIssues} unknown_addresses={_unknownAddresses} unobserved_contents={_unobservedContents} dropped={_dropped} history={HistoryFrames} capture_calls={_captureCalls} timing_probes={2 * _materializations} capture_ms={Ms(_captureTicks):F3} match_ms={Ms(_matchingTicks):F3} report_ms={Ms(_reportTicks):F3} diagnostic_bytes={_diagnosticBytes} actual_bypass=0 saved_ms=0"));
        for (var lag = 1; lag <= HistoryFrames; lag++)
        {
            WriteCounters(output, "all", lag, _totals[lag - 1]);
            foreach (var (pass, counters) in _passes)
                WriteCounters(output, $"{(pass.Frame < 0 ? "overflow" : pass.Shape.ToString())} frame={pass.Frame} id={pass.Id} kind={(pass.Compute ? "compute" : "render")} interval=1", lag, counters[lag - 1]);
        }
        _passes.Clear();
        _reportTicks += Stopwatch.GetTimestamp() - start;
        _diagnosticBytes += GC.GetAllocatedBytesForCurrentThread() - bytes;
    }
    private static double Ms(long ticks) => ticks * 1000.0 / Stopwatch.Frequency;
    private static void WriteCounters(TextWriter output, string pass, int lag, Counters c)
    {
        if (c.Comparisons == 0) return;
        output.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"[METAL-V2][RECIPE] pass={pass} lag={lag} n={c.Comparisons} exact={c.Issues[0]} relocation={c.Issues[1]} semantic={c.Issues[2]} value_repeat={c.ValueRepeats} exact_resmat_calls={c.Calls[0]} relocation_resmat_calls={c.Calls[1]} semantic_resmat_calls={c.Calls[2]} exact_candidate_ms={Ms(c.Ticks[0]):F3} relocation_candidate_ms={Ms(c.Ticks[1]):F3} semantic_resmat_ms={Ms(c.Ticks[2]):F3} value_repeat_calls={c.ValueRepeatCalls} value_repeat_candidate_ms={Ms(c.ValueRepeatTicks):F3} exact_unobserved={c.UnobservedIssues[0]} relocation_unobserved={c.UnobservedIssues[1]} semantic_unobserved={c.UnobservedIssues[2]} pass_instances={c.PassInstances}"));
    }
}
