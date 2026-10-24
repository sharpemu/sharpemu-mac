// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;

namespace SharpEmu.ShaderCompiler.Resources;

// The scalar evaluator's ordered, memoized walk, decoded once into array indices.
// Every memory instruction still runs on every use. Reads of the same handle and
// policy can span intervening total ALU operations. A refused span keeps scalar order.
// Structured resource selection stays on the original evaluator.
internal sealed class BatchedResourceWalk
{
    internal static readonly bool Enabled = PerfSwitches.Enabled("SHARPEMU_RESOURCE_FULLWALK");
    private const int MaximumGroupWords = 256;
    private enum Code { UserData, ShaderBase, Operation, Select, Read }
    private readonly record struct Op(Code Code, int Result, int A = 0, int B = 0, int C = 0, int D = 0,
        ScalarOperation Operation = default, ScalarValueKind Kind = default, long Immediate = 0,
        bool Clean = false, bool Absolute = false, int Group = 1, int[]? Pending = null, ScalarValue? Node = null,
        int PendingWords = 0, int[]? Members = null, int Ready = 0, int SkipWhen = 0);
    private readonly Op[] _ops;
    private readonly ulong[] _initial;
    private readonly int[][] _sources;
    private readonly int[] _table;
    private readonly int _tableStart;
    private readonly ShaderResourcePlan _plan;
    private readonly bool _requiresClean;
    private static long _walks, _groups, _words, _refused, _unsupported, _scalar, _structureFallback, _shapeFallback,
        _pendingFallback, _pendingSkipped, _pendingResident, _selectorHits, _selectorMisses, _typedCopies;
    internal static void RecordSelectorReuse(bool hit)
    {
        if (hit) Interlocked.Increment(ref _selectorHits);
        else Interlocked.Increment(ref _selectorMisses);
    }
    internal static void RecordTypedCopy() => Interlocked.Increment(ref _typedCopies);
    internal static void RecordFallback(ShaderResourcePlan plan)
    {
        if (RuntimeValueEvaluator.HasResourceStructure(plan) || plan.RequiresSpecializationMemory)
            Interlocked.Increment(ref _structureFallback);
        else Interlocked.Increment(ref _shapeFallback);
    }

    private BatchedResourceWalk(ShaderResourcePlan plan, Op[] ops, ulong[] initial, int[][] sources, int[] table, int tableStart)
    {
        _plan = plan; _ops = ops; _initial = initial; _sources = sources; _table = table; _tableStart = tableStart;
        _requiresClean = plan.CleanFlatSlots.Any(clean => clean != 0);
    }

    internal static string TakeReport() =>
        $" fullwalk_enabled={Enabled} fullwalk_walks={Interlocked.Exchange(ref _walks, 0)} fullwalk_groups={Interlocked.Exchange(ref _groups, 0)}" +
        $" fullwalk_words={Interlocked.Exchange(ref _words, 0)} fullwalk_span_refused={Interlocked.Exchange(ref _refused, 0)}" +
        $" fullwalk_unsupported_plans={Interlocked.Exchange(ref _unsupported, 0)} fullwalk_scalar={Interlocked.Exchange(ref _scalar, 0)}" +
        $" fullwalk_fallback_structure={Interlocked.Exchange(ref _structureFallback, 0)} fullwalk_fallback_shape={Interlocked.Exchange(ref _shapeFallback, 0)}" +
        $" fullwalk_pending_fallback={Interlocked.Exchange(ref _pendingFallback, 0)} fullwalk_pending_skipped={Interlocked.Exchange(ref _pendingSkipped, 0)}" +
        $" fullwalk_pending_resident={Interlocked.Exchange(ref _pendingResident, 0)}" +
        $" fullwalk_selector_hits={Interlocked.Exchange(ref _selectorHits, 0)} fullwalk_selector_misses={Interlocked.Exchange(ref _selectorMisses, 0)}" +
        $" fullwalk_typed_copies={Interlocked.Exchange(ref _typedCopies, 0)}";

    internal static BatchedResourceWalk? Build(ShaderResourcePlan plan)
    {
        if (RuntimeValueEvaluator.HasResourceStructure(plan) || plan.RequiresSpecializationMemory)
        { Interlocked.Increment(ref _unsupported); return null; }
        var ops = new List<Op>();
        var initial = new List<ulong> { 0 };
        var known = new Dictionary<(ScalarValue, bool), int>();
        var visiting = new HashSet<(ScalarValue, bool)>();
        var pendingByRead = new Dictionary<ScalarValue, int[]>();
        foreach (var (slot, indices) in plan.DeviceDataSlots)
            pendingByRead[plan.TableReads[slot].Value] = indices;

        int Compile(ScalarValue value, bool clean)
        {
            if (known.TryGetValue((value, clean), out var existing)) return existing;
            if (initial.Count >= 16384 || !visiting.Add((value, clean)) || visiting.Count > 64)
                throw new NotSupportedException();
            var result = initial.Count;
            initial.Add(0);
            int Operand(int index) => Compile(value.Operands[index], clean);
            switch (value.Kind)
            {
                case ScalarValueKind.Constant: initial[result] = value.Payload; break;
                case ScalarValueKind.MemoryAperture:
                    initial[result] = Gen5InlineConstants.DecodeAperture64((uint)value.Payload) >> 32; break;
                case ScalarValueKind.UserData:
                    if (value.UserDataRegister < plan.UserDataBase) throw new NotSupportedException();
                    ops.Add(new(Code.UserData, result, checked((int)(value.UserDataRegister - plan.UserDataBase)))); break;
                case ScalarValueKind.ShaderBase: ops.Add(new(Code.ShaderBase, result)); break;
                case ScalarValueKind.ResourceTableWord:
                    if (value.Payload >= (ulong)plan.TableReads.Count) throw new NotSupportedException();
                    var slot = (int)value.Payload;
                    result = Compile(plan.TableReads[slot].Value, clean || plan.CleanFlatSlots[slot] != 0); break;
                case ScalarValueKind.Phi:
                    var invariant = plan.Graph.ResolveInvariantPhi(value);
                    if (invariant is null || ReferenceEquals(invariant, value)) throw new NotSupportedException();
                    result = Compile(invariant, clean); break;
                case ScalarValueKind.Select:
                    if (value.Operands.Length != 3) throw new NotSupportedException();
                    ops.Add(new(Code.Select, result, Operand(0), Operand(1), Operand(2))); break;
                case ScalarValueKind.Operation:
                    if (!RuntimeValueValidator.IsUniformOperation(value.Operation) || value.Operands.Length > 4)
                        throw new NotSupportedException();
                    var arguments = new int[4];
                    // Missing operands are literal zero, never an unrelated value at index zero.
                    for (var index = 0; index < 4; index++)
                    {
                        if (index < value.Operands.Length) arguments[index] = Operand(index);
                        else arguments[index] = 0;
                    }
                    ops.Add(new(Code.Operation, result, arguments[0], arguments[1], arguments[2], arguments[3], value.Operation)); break;
                case ScalarValueKind.ScalarAddressWord:
                case ScalarValueKind.ScalarBufferWord:
                    if (value.Operands.Length < 2 || (uint)value.MemoryIndex >= plan.Memory.Count) throw new NotSupportedException();
                    var handle = value.Operands[0];
                    if (handle.Operands.Length < 2 || (value.Kind == ScalarValueKind.ScalarBufferWord && handle.Operands.Length != 4))
                        throw new NotSupportedException();
                    var low = Compile(handle.Operands[0], clean);
                    var high = Compile(handle.Operands[1], clean);
                    var offset = Operand(1);
                    var records = 0;
                    if (value.Kind == ScalarValueKind.ScalarBufferWord)
                    { records = Compile(handle.Operands[2], clean); Compile(handle.Operands[3], clean); }
                    pendingByRead.TryGetValue(value, out var pending);
                    ops.Add(new(Code.Read, result, low, high, offset, records,
                        Kind: value.Kind, Immediate: (int)plan.Memory[value.MemoryIndex].Offset, Clean: clean,
                        Absolute: RuntimeValueEvaluator.IsAbsoluteAddress(value.Kind, handle), Pending: clean ? null : pending, Node: value,
                        PendingWords: !clean && pending is not null ? 1 : 0)); break;
                default: throw new NotSupportedException();
            }
            visiting.Remove((value, clean));
            known[(value, clean)] = result;
            return result;
        }
        try
        {
            var sources = new int[plan.MaterializationSources.Count][];
            for (var source = 0; source < sources.Length; source++)
                sources[source] = plan.DescriptorSources[(int)plan.MaterializationSources[source]].Dwords.Select(word => Compile(word, false)).ToArray();
            var tableStart = ops.Count;
            var table = new int[plan.TableReads.Count];
            foreach (var read in plan.TableReads)
            {
                if (read.FlatOffset >= table.Length) return null;
                table[read.FlatOffset] = Compile(read.Value, plan.CleanFlatSlots[(int)read.FlatOffset] != 0);
            }
            // Prefetch through total, side-effect-free ALU only. A new handle, an
            // input access, or an operation that could fail ends the run. Results
            // are seeded ahead, but the ALU still executes in its original order.
            bool Total(Op op) => op.Code == Code.Select || op.Code == Code.ShaderBase ||
                op.Code == Code.Operation && op.Operation is
                    ScalarOperation.IAdd32 or ScalarOperation.IAdd64 or ScalarOperation.ISub32 or ScalarOperation.ISub64 or
                    ScalarOperation.IMul32 or ScalarOperation.IMul64 or ScalarOperation.Construct64 or
                    ScalarOperation.And32 or ScalarOperation.And64 or ScalarOperation.Or32 or ScalarOperation.Xor32 or
                    ScalarOperation.Not32 or ScalarOperation.ShiftLeft32 or ScalarOperation.ShiftLeft64 or
                    ScalarOperation.ShiftRightLogical32 or ScalarOperation.ShiftRightLogical64 or
                    ScalarOperation.ShiftRightArithmetic32 or ScalarOperation.ShiftRightArithmetic64;
            void Group(int first, int end)
            {
                for (var index = first; index < end; index++)
                {
                    var start = ops[index];
                    if (start.Code != Code.Read || start.SkipWhen != 0 || start.Node!.Operands[1].Kind != ScalarValueKind.Constant) continue;
                    long Relative(Op read) => read.Kind == ScalarValueKind.ScalarBufferWord
                        ? (read.Immediate + (uint)initial[read.C]) & ~3L
                        : (read.Immediate & ~3L) + ((uint)initial[read.C] & ~3U);
                    var relative = Relative(start);
                    var members = new List<int> { index };
                    var pendingWords = start.PendingWords;
                    for (var nextIndex = index + 1; nextIndex < end && members.Count < MaximumGroupWords; nextIndex++)
                    {
                        var next = ops[nextIndex];
                        if (Total(next)) continue;
                        if (next.Code != Code.Read || next.Kind != start.Kind || next.A != start.A || next.B != start.B ||
                            (next.Kind == ScalarValueKind.ScalarBufferWord && next.D != start.D) || next.Clean != start.Clean ||
                            next.Absolute != start.Absolute || next.Node!.Operands[1].Kind != ScalarValueKind.Constant ||
                            Relative(next) != relative + members.Count * 4) break;
                        pendingWords += next.PendingWords;
                        members.Add(nextIndex);
                    }
                    if (members.Count == 1) continue;
                    var ready = initial.Count;
                    initial.Add(0);
                    ops[index] = start with { Group = members.Count, PendingWords = pendingWords, Members = members.ToArray(), Ready = ready };
                    foreach (var member in members.Skip(1)) ops[member] = ops[member] with { SkipWhen = ready };
                }
            }
            Group(0, tableStart); Group(tableStart, ops.Count);
            return new(plan, ops.ToArray(), initial.ToArray(), sources, table, tableStart);
        }
        catch (NotSupportedException) { Interlocked.Increment(ref _unsupported); return null; }
    }

    // Only capacities live between calls. Inputs, guest words and output arrays do not.
    private sealed class Scratch
    {
        internal ulong[] Values = [];
        internal Scratch? Next;
    }
    [ThreadStatic] private static Scratch? _available;

    internal bool Evaluate(ResourceRuntimeInputs inputs, int additionalTableWords, out List<DescriptorWords> descriptors, out uint[] table)
    {
        using var profile = ResourceMaterializationProfile.Measure(ResourceMaterializationProfile.Phase.BatchedWalk);
        descriptors = []; table = [];
        if (_requiresClean && inputs.ReadCleanMemory is null) return false;
        var scratch = _available ?? new Scratch();
        _available = scratch.Next;
        scratch.Next = null;
        if (scratch.Values.Length < _initial.Length) scratch.Values = new ulong[Math.Max(_initial.Length, 256)];
        var values = scratch.Values;
        _initial.CopyTo(values, 0);
        var inTable = false;
        try
        {
            for (var index = 0; index < _ops.Length; index++)
            {
                if (!inTable && index >= _tableStart) { inTable = true; inputs.TablePhase?.Invoke(true); }
                ref readonly var op = ref _ops[index];
                switch (op.Code)
                {
                    case Code.UserData:
                        if ((uint)op.A >= (uint)inputs.UserData.Count) return false;
                        values[op.Result] = inputs.UserData[op.A]; break;
                    case Code.ShaderBase: values[op.Result] = inputs.ShaderBase; break;
                    case Code.Select: values[op.Result] = values[op.A] != 0 ? values[op.B] : values[op.C]; break;
                    case Code.Operation:
                        if (!ScalarOperationSemantics.TryEvaluateFixed(op.Operation, values[op.A], values[op.B], values[op.C], values[op.D], out values[op.Result])) return false;
                        break;
                    case Code.Read:
                        if (op.SkipWhen != 0 && values[op.SkipWhen] != 0)
                        {
                            if (values[op.SkipWhen] == 2) LearnPending(inputs, values, op);
                            break;
                        }
                        if (!Read(inputs, values, index, op.Group)) return false;
                        break;
                }
            }
            var output = new List<DescriptorWords>(_sources.Length);
            foreach (var source in _sources)
            {
                var words = new uint[source.Length];
                for (var word = 0; word < words.Length; word++) words[word] = (uint)values[source[word]];
                output.Add(new(words));
            }
            var flattened = new uint[checked(_table.Length + additionalTableWords)];
            for (var slot = 0; slot < _table.Length; slot++) flattened[slot] = (uint)values[_table[slot]];
            descriptors = output; table = flattened;
            Interlocked.Increment(ref _walks);
            return true;
        }
        finally
        {
            if (inTable) inputs.TablePhase?.Invoke(false);
            scratch.Next = _available;
            _available = scratch;
        }
    }

    private bool Read(ResourceRuntimeInputs inputs, ulong[] values, int index, int count)
    {
        ref readonly var firstOp = ref _ops[index];
        Span<byte> bytes = stackalloc byte[count * 4];
        var baseAddress = ((values[firstOp.B] << 32) | (uint)values[firstOp.A]) & 0x0000FFFFFFFFFFFFUL;
        var firstState = RuntimeValueEvaluator.ResolveRawAddress(firstOp.Kind, baseAddress, values[firstOp.B], values[firstOp.D],
            firstOp.Immediate, (uint)values[firstOp.C], out var firstAddress);
        // The plan proved a monotonic, contiguous run with a common handle. Its two
        // endpoints prove every intermediate raw-address/OOB check without repeating
        // that work for every word. Refusal keeps the scalar checks below.
        ref readonly var lastOp = ref _ops[firstOp.Members is { } members ? members[^1] : index];
        var lastState = RuntimeValueEvaluator.ResolveRawAddress(lastOp.Kind, baseAddress, values[lastOp.B], values[lastOp.D],
            lastOp.Immediate, (uint)values[lastOp.C], out var lastAddress);
        var contiguous = firstState == RuntimeValueEvaluator.RawAddress.Read && lastState == RuntimeValueEvaluator.RawAddress.Read &&
            lastAddress - firstAddress == (ulong)(count - 1) * 4 && (baseAddress & ~3UL) != 0;
        contiguous &= inputs.ReadResidentMemory is not null &&
            (firstOp.Clean ? inputs.ReadCleanMemory : inputs.ReadMemory) is not null;
        var resident = false;
        if (contiguous)
        {
            using var readProfile = ResourceMaterializationProfile.MeasureSampled(ResourceMaterializationProfile.Phase.BatchedResidentRead);
            resident = inputs.ReadResidentMemory!(firstAddress, bytes, firstOp.Clean || inputs.ReadsClean);
        }
        // Match RawReadPrefetch: an admitted resident span is read before the
        // table's precise data probes. A possible writer teaches device loads; it
        // must not force a synchronizing scalar read that the base prefetch avoids.
        // Refused spans still probe immediately before their original scalar read.
        var pending = inputs.DataWritePending;
        var probe = false;
        if (firstOp.PendingWords != 0 && pending is not null)
        {
            using var pendingProfile = ResourceMaterializationProfile.MeasureSampled(ResourceMaterializationProfile.Phase.BatchedPendingProbe);
            probe = !(contiguous && inputs.DataRangeWritePending is not null &&
                !inputs.DataRangeWritePending(firstAddress, (ulong)count * 4));
        }
        if (resident)
        {
            if (probe) Interlocked.Increment(ref _pendingResident);
            else if (firstOp.PendingWords != 0 && pending is not null) Interlocked.Add(ref _pendingSkipped, firstOp.PendingWords);
            Interlocked.Increment(ref _groups); Interlocked.Add(ref _words, count);
            for (var word = 0; word < count; word++)
            {
                ref readonly var op = ref _ops[firstOp.Members is { } indices ? indices[word] : index];
                values[op.Result] = BinaryPrimitives.ReadUInt32LittleEndian(bytes[(word * 4)..]);
                ResourceMaterializationCache.RecordReadConsumer(_plan, op.Node!, firstAddress + (ulong)word * 4);
            }
            if (firstOp.Ready != 0) values[firstOp.Ready] = probe ? 2UL : 1UL;
            if (probe) LearnPending(inputs, values, firstOp);
            return true;
        }
        if (contiguous) Interlocked.Increment(ref _refused);
        if (probe) Interlocked.Increment(ref _pendingFallback);
        if (!probe && firstOp.Pending is not null && pending is not null) Interlocked.Increment(ref _pendingSkipped);
        // Only this read falls back. Future group members retain their original
        // position, including precise pending probes and any earlier failure.
        ref readonly var scalar = ref _ops[index];
        var wordBase = ((values[scalar.B] << 32) | (uint)values[scalar.A]) & 0x0000FFFFFFFFFFFFUL;
        var state = RuntimeValueEvaluator.ResolveRawAddress(scalar.Kind, wordBase, values[scalar.B], values[scalar.D], scalar.Immediate, (uint)values[scalar.C], out var address);
        if (state == RuntimeValueEvaluator.RawAddress.Failed) return false;
        values[scalar.Result] = 0;
        if (state == RuntimeValueEvaluator.RawAddress.Zero) return true;
        if (probe && scalar.Pending is not null && pending!(address)) DeviceDataLoads.Learn(_plan.Hash, scalar.Pending);
        var reader = scalar.Clean ? inputs.ReadCleanMemory : inputs.ReadMemory;
        Interlocked.Increment(ref _scalar);
        uint value = 0;
        bool read;
        using (ResourceMaterializationProfile.MeasureSampled(ResourceMaterializationProfile.Phase.BatchedScalarRead))
            read = reader is not null && reader(address, out value);
        if (!read)
        {
            return (wordBase & ~3UL) == 0 || (scalar.Absolute && reader is not null && !scalar.Clean && !inputs.ReadsClean);
        }
        values[scalar.Result] = value;
        ResourceMaterializationCache.RecordReadConsumer(_plan, scalar.Node!, address);
        return true;
    }

    private void LearnPending(ResourceRuntimeInputs inputs, ulong[] values, in Op op)
    {
        if (op.Pending is null || inputs.DataWritePending is null) return;
        var baseAddress = ((values[op.B] << 32) | (uint)values[op.A]) & 0x0000FFFFFFFFFFFFUL;
        if (RuntimeValueEvaluator.ResolveRawAddress(op.Kind, baseAddress, values[op.B], values[op.D],
                op.Immediate, (uint)values[op.C], out var address) == RuntimeValueEvaluator.RawAddress.Read &&
            inputs.DataWritePending(address)) DeviceDataLoads.Learn(_plan.Hash, op.Pending);
    }
}
