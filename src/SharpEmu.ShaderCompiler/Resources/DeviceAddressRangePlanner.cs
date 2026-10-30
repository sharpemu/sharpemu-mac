// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Generic;
using System.Linq;

namespace SharpEmu.ShaderCompiler.Resources;

// One address handle's planned range: its base value and the signed byte span its
// accesses reach. A handle with a run-time offset is unbounded up to the cap.
public sealed record DeviceAddressRangePlan(
    uint Handle,
    ScalarValue BaseLow,
    ScalarValue BaseHigh,
    IReadOnlyList<int> MemoryIndices,
    bool Written,
    bool Bounded,
    long FirstByte,
    long EndByte,
    bool Plannable)
{
    public ulong Extent => Bounded && EndByte > FirstByte ? (ulong)(EndByte - FirstByte) : 0;
}

// A buffer store whose V# the program loads at a run-time offset: one the draw's own values
// give (Direct: the four words evaluate as they are), or through a pointer at one of a few
// values the program can only compute (Offsets), or at a counted loop's induction, offset =
// BaseOffset + index * Stride for index = Initial, Initial + Step, ... while the loop's guard
// against Limit holds. Words are the V#'s four scalar reads. With a RecordTable the V# is one
// field of a buffer of records the program indexes with a value it cannot follow: offset =
// index * RecordStride + FirstImmediate for every record the buffer holds.
public sealed record DeviceDescriptorTablePlan(
    bool Direct,
    ScalarValue? TableLow,
    ScalarValue? TableHigh,
    long FirstImmediate,
    IReadOnlyList<uint>? Offsets,
    uint BaseOffset,
    uint Stride,
    uint Initial,
    uint Step,
    ScalarValue? Limit,
    bool LimitInclusive,
    IReadOnlyList<ScalarValue> Words,
    ScalarValue? RecordTable = null,
    uint RecordStride = 0);

// Bounds every device-address handle after tracking. The pass is optional: an
// unplannable handle never fails the plan, it is reported for the hosts to decide.
public static class DeviceAddressRangePlanner
{
    public const ulong MaxRangeBytes = 16 * 1024 * 1024;
    private const ulong AddressMask = 0x0000_FFFF_FFFF_FFFFul;

    // The most descriptors a table's loop may select for one draw before its stores are left
    // unbounded again.
    public const int MaxDescriptorTableEntries = 64;

    // The most records a record table may hold for one draw before its stores are left unbounded.
    public const int MaxRecordTableEntries = 1024;

    // A store through a V# loaded from guest memory goes through the device-address page table,
    // and no handle range says where: the hosts record it as a store to all of guest memory, and
    // every later read waits for it. Ghost of Tsushima's light binning stores into one buffer per
    // bin, each bin's V# loaded with s_load_dwordx4 s[4:7], s[0:1], (bin << 4) + 16 for every bin
    // below a count it loads too - half the GPU worker's time went to the waits that caused - and
    // its tile classification picks one of four with (selector << 4) + 32. When the offset takes
    // only a few values, or is such a loop's induction, every V# it can select (and the loop's
    // bound) is read for each draw (Evaluate), and the stores are recorded against those buffers.
    // SHARPEMU_TRACE_DEVICE_DESCRIPTOR_TABLES=1: why each buffer store with a run-time V# was or was not planned.
    private static readonly bool TraceDescriptorTables =
        Environment.GetEnvironmentVariable("SHARPEMU_TRACE_DEVICE_DESCRIPTOR_TABLES") == "1";

    public static (IReadOnlyList<DeviceDescriptorTablePlan> Tables, bool Unplanned) PlanDescriptorTables(ShaderResourcePlan plan)
    {
        var tables = new List<DeviceDescriptorTablePlan>();
        var handles = new List<ScalarValue>();
        var unplanned = false;
        for (var index = 0; index < plan.Memory.Count; index++)
        {
            var memory = plan.Memory[index];
            if (TraceDescriptorTables && memory.Kind == MemoryResourceKind.Buffer &&
                memory.Access is MemoryAccess.Write or MemoryAccess.Atomic)
            {
                Console.Error.WriteLine(
                    $"[DD-TABLE] hash=0x{plan.Hash:X16} pc=0x{memory.Pc:X} {memory.Opcode} device_descriptor={memory.DeviceDescriptor} " +
                    $"provenance={memory.BufferDescriptor?.Provenance.ToString() ?? "none"} planning_only={memory.PlanningOnly}");
            }

            // Both lowerings of a V# the program loads itself go through the page table: a device
            // descriptor, and a run-time one no candidate table binds.
            if (memory.Kind != MemoryResourceKind.Buffer || memory.PlanningOnly ||
                memory.Access is not (MemoryAccess.Write or MemoryAccess.Atomic) ||
                !(memory.DeviceDescriptor || memory.BufferDescriptor is { Provenance: BufferDescriptorProvenance.Runtime }) ||
                plan.BufferCandidateTables.Any(candidates => candidates.MemoryIndices.Contains(index)) ||
                plan.Graph.Program.FunctionBufferAccesses.Contains(memory.Pc))
            {
                continue;
            }

            string? reason = null;
            if (plan.Accesses[index]?.Handle is not { Kind: ScalarValueKind.BufferHandle, Operands.Length: 4 } handle ||
                !TryPlanDescriptorTable(plan, handle, out var table, out reason))
            {
                if (TraceDescriptorTables)
                {
                    Console.Error.WriteLine($"[DD-TABLE] hash=0x{plan.Hash:X16} pc=0x{memory.Pc:X} unplanned: {reason ?? "no buffer handle"}");
                }

                unplanned = true;
                continue;
            }

            if (handles.Any(existing => plan.Graph.Equivalent(existing, handle)))
            {
                continue;
            }

            if (TraceDescriptorTables)
            {
                Console.Error.WriteLine($"[DD-TABLE] hash=0x{plan.Hash:X16} pc=0x{memory.Pc:X} planned: {reason ?? "loop"}");
            }

            handles.Add(handle);
            tables.Add(table);
        }

        return (tables, unplanned);
    }

    private static bool TryPlanDescriptorTable(
        ShaderResourcePlan plan, ScalarValue handle, out DeviceDescriptorTablePlan table, out string? reason)
    {
        table = null!;
        reason = null;

        // A V# read at an offset the draw fixes (a table word times a stride) is one buffer per
        // draw: its words evaluate like any descriptor's.
        if (handle.Operands.All(word => IsDrawUniform(plan, word, []) &&
                plan.ValidateRuntimeValue(word, out var controlDependent) && !controlDependent))
        {
            table = new DeviceDescriptorTablePlan(true, null, null, 0, null, 0, 0, 0, 0, null, false, handle.Operands);
            reason = "direct";
            return true;
        }

        if (RecordTableStores && TryPlanRecordTable(plan, handle, out table, out reason))
        {
            return true;
        }

        // Every dword is one read through the same pointer at the same run-time offset, at the
        // consecutive immediates one dwordx4 load gives its components.
        ScalarValue? address = null;
        ScalarValue? offset = null;
        long firstImmediate = 0;
        var words = new ScalarValue[4];
        for (var word = 0; word < 4; word++)
        {
            var read = handle.Operands[word];
            if (read.Kind != ScalarValueKind.ScalarAddressWord || (uint)read.MemoryIndex >= (uint)plan.Memory.Count ||
                read.Operands[0] is not { Kind: ScalarValueKind.AddressHandle, Operands.Length: 2 } pointer)
            {
                reason = $"word {word} is {read.Kind}, not a read through a pointer: {Describe(read, 0)}";
                return false;
            }

            var memory = plan.Memory[read.MemoryIndex];
            var immediate = (long)unchecked((int)memory.Offset);
            if (memory.Kind != MemoryResourceKind.ScalarAddress || memory.DataBits != 32 || memory.DataDwords != 1)
            {
                reason = $"word {word} reads {memory.Kind} {memory.DataBits}x{memory.DataDwords}";
                return false;
            }

            if (address is null)
            {
                address = pointer;
                offset = read.Operands[1];
                firstImmediate = immediate;
            }
            else if (!plan.Graph.Equivalent(address, pointer) || !plan.Graph.Equivalent(offset!, read.Operands[1]) ||
                immediate != firstImmediate + word * sizeof(uint))
            {
                reason = $"word {word} is not the next dword of one load";
                return false;
            }

            words[word] = read;
        }

        if (!plan.ValidateRuntimeValue(address!.Operands[0]) || !plan.ValidateRuntimeValue(address.Operands[1]))
        {
            reason = "table pointer not evaluable for a draw";
            return false;
        }

        // A handful of values computed from constants alone (selects of lane conditions).
        var values = new HashSet<uint>();
        if (TryEnumerateValues(offset!, values, [], plan.Graph) && values.Count != 0)
        {
            table = new DeviceDescriptorTablePlan(
                false, address.Operands[0], address.Operands[1], firstImmediate, [.. values.Order()], 0, 0, 0, 0, null, false, words);
            reason = $"offsets {string.Join(",", values.Order().Select(value => $"0x{value:X}"))}" +
                (TraceDescriptorTables ? $" from {Describe(offset!, 0)}" : string.Empty);
            return true;
        }

        // offset = base + (index << shift), or base + index * stride.
        var term = offset!;
        var baseOffset = 0u;
        if (term is { Kind: ScalarValueKind.Operation, Operation: ScalarOperation.IAdd32 })
        {
            if (term.Operands[1].IsConstant)
            {
                baseOffset = term.Operands[1].ConstantU32;
                term = term.Operands[0];
            }
            else if (term.Operands[0].IsConstant)
            {
                baseOffset = term.Operands[0].ConstantU32;
                term = term.Operands[1];
            }
        }

        ScalarValue index;
        uint stride;
        if (term is { Kind: ScalarValueKind.Operation, Operation: ScalarOperation.ShiftLeft32 } &&
            term.Operands[1].IsConstant && term.Operands[1].ConstantU32 < 32)
        {
            index = term.Operands[0];
            stride = 1u << (int)term.Operands[1].ConstantU32;
        }
        else if (term is { Kind: ScalarValueKind.Operation, Operation: ScalarOperation.IMul32 } &&
            (term.Operands[0].IsConstant || term.Operands[1].IsConstant))
        {
            index = term.Operands[0].IsConstant ? term.Operands[1] : term.Operands[0];
            stride = term.Operands[0].IsConstant ? term.Operands[0].ConstantU32 : term.Operands[1].ConstantU32;
        }
        else
        {
            reason = $"offset is {term.Kind} {term.Operation}, not an index times a stride: {Describe(offset!, 0)}";
            return false;
        }

        if (stride == 0 ||
            !BufferCandidateTablePlanner.TryProveInductionBound(plan, index, out var initial, out var step, out var limit, out var inclusive,
                exitGuards: true) ||
            limit is null || !initial.IsConstant)
        {
            reason = $"no proven loop bound (index {index.Kind} {index.Operation})";
            return false;
        }

        if (!limit.IsConstant && !plan.ValidateRuntimeValue(limit))
        {
            reason = $"loop limit ({limit.Kind}) not evaluable for a draw";
            return false;
        }

        table = new DeviceDescriptorTablePlan(
            false, address.Operands[0], address.Operands[1], firstImmediate, null, baseOffset, stride,
            initial.ConstantU32, step, limit, inclusive, words);
        return true;
    }

    // Ghost of Tsushima's Hi-Z culler (csA43A27DF) stores visibility through the V# at +0x190 of
    // the 504-byte record each wave picks with v_readlane, s_buffer_load_dwordx4 s[60:63], s[44:47],
    // s8 offset:0x190, under phis of the loop over its lanes' records. The index is lost, but the
    // record buffer is the draw's: every V# it can pick is one of its records', so the stores are
    // recorded against those buffers instead of all of guest memory (a quarter of the GPU worker's
    // time went to the waits that caused in its gorge).
    // SHARPEMU_RECORD_TABLE_STORES=0 leaves such stores unbounded again.
    private static readonly bool RecordTableStores =
        Environment.GetEnvironmentVariable("SHARPEMU_RECORD_TABLE_STORES") != "0";

    private static bool TryPlanRecordTable(
        ShaderResourcePlan plan, ScalarValue handle, out DeviceDescriptorTablePlan table, out string? reason)
    {
        table = null!;
        reason = null;
        ScalarValue? records = null;
        var recordStride = 0u;
        long firstImmediate = 0;
        var words = new ScalarValue[4];
        for (var word = 0; word < 4; word++)
        {
            var leaves = new List<ScalarValue>();
            if (!CollectPhiLeaves(handle.Operands[word], leaves, []) || leaves.Count == 0)
            {
                reason = $"record word {word}: phis too deep";
                return false;
            }

            foreach (var leaf in leaves)
            {
                if (leaf.Kind != ScalarValueKind.ScalarBufferWord || (uint)leaf.MemoryIndex >= (uint)plan.Memory.Count ||
                    leaf.Operands[0] is not { Kind: ScalarValueKind.BufferHandle, Operands.Length: 4 } buffer ||
                    !TrySplitRecordOffset(leaf.Operands[1], out var stride, out var fieldOffset))
                {
                    reason = $"record word {word} is {leaf.Kind}, not a buffer read at an index times a stride";
                    return false;
                }

                var memory = plan.Memory[leaf.MemoryIndex];
                var immediate = (long)unchecked((int)memory.Offset) + fieldOffset;
                if (memory.Kind != MemoryResourceKind.ScalarBuffer || memory.DataBits != 32 || memory.DataDwords != 1)
                {
                    reason = $"record word {word} reads {memory.Kind} {memory.DataBits}x{memory.DataDwords}";
                    return false;
                }

                if (records is null)
                {
                    if (word != 0 || stride < 4 * sizeof(uint) || immediate < 0 ||
                        !buffer.Operands.All(part => IsDrawUniform(plan, part, []) &&
                            plan.ValidateRuntimeValue(part, out var controlDependent) && !controlDependent))
                    {
                        reason = "record buffer not evaluable for a draw";
                        return false;
                    }

                    records = buffer;
                    recordStride = stride;
                    firstImmediate = immediate;
                }
                else if (!plan.Graph.Equivalent(records, buffer) || stride != recordStride ||
                    immediate != firstImmediate + word * sizeof(uint))
                {
                    reason = $"record word {word} is not the next dword of one record field";
                    return false;
                }
            }

            words[word] = leaves[0];
        }

        table = new DeviceDescriptorTablePlan(
            false, null, null, firstImmediate, null, 0, 0, 0, 0, null, false, words, records, recordStride);
        reason = $"record table, stride 0x{recordStride:X} field 0x{firstImmediate:X}";
        return true;
    }

    // The values a phi tree merges; undefined ones are lanes that do not run the access.
    private static bool CollectPhiLeaves(ScalarValue value, List<ScalarValue> leaves, HashSet<ScalarValue> visiting)
    {
        if (value.Kind == ScalarValueKind.Undefined)
        {
            return true;
        }

        if (value.Kind != ScalarValueKind.Phi)
        {
            if (!leaves.Contains(value))
            {
                leaves.Add(value);
            }

            return leaves.Count <= MaxDescriptorTableEntries;
        }

        if (!visiting.Add(value))
        {
            return true;
        }

        return value.Operands.All(operand => CollectPhiLeaves(operand, leaves, visiting));
    }

    // offset = index * stride (+ a constant field offset), index not a constant.
    private static bool TrySplitRecordOffset(ScalarValue offset, out uint stride, out long fieldOffset)
    {
        stride = 0;
        fieldOffset = 0;
        if (offset is { Kind: ScalarValueKind.Operation, Operation: ScalarOperation.IAdd32 } &&
            (offset.Operands[0].IsConstant || offset.Operands[1].IsConstant))
        {
            fieldOffset = offset.Operands[1].IsConstant ? offset.Operands[1].ConstantU32 : offset.Operands[0].ConstantU32;
            offset = offset.Operands[1].IsConstant ? offset.Operands[0] : offset.Operands[1];
        }

        if (offset is { Kind: ScalarValueKind.Operation, Operation: ScalarOperation.IMul32 } &&
            offset.Operands[0].IsConstant != offset.Operands[1].IsConstant)
        {
            stride = offset.Operands[0].IsConstant ? offset.Operands[0].ConstantU32 : offset.Operands[1].ConstantU32;
            return stride != 0;
        }

        if (offset is { Kind: ScalarValueKind.Operation, Operation: ScalarOperation.ShiftLeft32 } &&
            !offset.Operands[0].IsConstant && offset.Operands[1].IsConstant && offset.Operands[1].ConstantU32 < 32)
        {
            stride = 1u << (int)offset.Operands[1].ConstantU32;
            return true;
        }

        return false;
    }

    // A value every lane of every invocation of the draw computes alike: built only from user
    // data, the shader base, resource-table words, constants and memory read at such addresses.
    // A first lane, a select or a loop-carried phi can differ between invocations.
    private static bool IsDrawUniform(ShaderResourcePlan plan, ScalarValue value, HashSet<ScalarValue> visiting)
    {
        if (value.Kind == ScalarValueKind.Phi)
        {
            var invariant = plan.Graph.ResolveInvariantPhi(value);
            return invariant is not null && IsDrawUniform(plan, invariant, visiting);
        }

        if (value.IsConstant || value.Kind is ScalarValueKind.UserData or ScalarValueKind.ShaderBase or ScalarValueKind.ResourceTableWord)
        {
            return true;
        }

        if (value.Kind is not (ScalarValueKind.Operation or ScalarValueKind.ScalarBufferWord or ScalarValueKind.ScalarAddressWord or
                ScalarValueKind.BufferHandle or ScalarValueKind.AddressHandle) || !visiting.Add(value))
        {
            return false;
        }

        try
        {
            return value.Operands.All(operand => IsDrawUniform(plan, operand, visiting));
        }
        finally
        {
            visiting.Remove(value);
        }
    }

    // Every value a 32-bit expression of constants can take, through selects, phis and the first
    // lane of a vector value; false when something else feeds it or there are more than
    // MaxDescriptorTableEntries. An undefined operand is a lane that does not run the access.
    // Under a first lane, a select on that lane's active mask - how a vector write keeps an
    // inactive lane's old value - can only take its written arm: the first lane is an active
    // one. Ghost of Tsushima's tile classifier (cs9260AE70) picks its list V# as
    // [s0:s1 + 0x20 + bucket * 16] for buckets 0-4; the old values of the registers it reused
    // added 0x70 and 0x90, and the V# at 0x70 spans the whole 664 MB render-target heap, which
    // every dispatch then wrote back and invalidated.
    // SHARPEMU_FIRST_LANE_ACTIVE_ARMS=0 enumerates both arms of such selects again.
    private static readonly bool FirstLaneActiveArms =
        Environment.GetEnvironmentVariable("SHARPEMU_FIRST_LANE_ACTIVE_ARMS") != "0";

    private static bool TryEnumerateValues(
        ScalarValue value, HashSet<uint> values, HashSet<ScalarValue> visiting, ScalarValueGraph? graph = null,
        ScalarValue? active = null)
    {
        if (value.IsConstant)
        {
            values.Add(value.ConstantU32);
            return values.Count <= MaxDescriptorTableEntries;
        }

        if (value.Kind == ScalarValueKind.Undefined)
        {
            return true;
        }

        if (!visiting.Add(value))
        {
            return false;
        }

        try
        {
            switch (value.Kind)
            {
                case ScalarValueKind.Select when value.Operands.Length == 3:
                    if (FirstLaneActiveArms && active is not null && graph is not null && graph.Equivalent(value.Operands[0], active))
                    {
                        return TryEnumerateValues(value.Operands[1], values, visiting, graph, active);
                    }

                    return TryEnumerateValues(value.Operands[1], values, visiting, graph, active) &&
                        TryEnumerateValues(value.Operands[2], values, visiting, graph, active);
                case ScalarValueKind.Phi:
                    return value.Operands.All(operand => TryEnumerateValues(operand, values, visiting, graph, active));
                case ScalarValueKind.FirstLane when value.Operands.Length != 0:
                    return TryEnumerateValues(value.Operands[0], values, visiting, graph,
                        value.Operands.Length > 1 ? value.Operands[1] : active);
                case ScalarValueKind.Operation when value.Operands.Length == 2 && value.Operation is
                    ScalarOperation.IAdd32 or ScalarOperation.ISub32 or ScalarOperation.IMul32 or ScalarOperation.ShiftLeft32 or
                    ScalarOperation.ShiftRightLogical32 or ScalarOperation.And32 or ScalarOperation.Or32:
                {
                    var left = new HashSet<uint>();
                    var right = new HashSet<uint>();
                    if (!TryEnumerateValues(value.Operands[0], left, visiting, graph, active) ||
                        !TryEnumerateValues(value.Operands[1], right, visiting, graph, active))
                    {
                        return false;
                    }

                    Span<ulong> operands = stackalloc ulong[2];
                    foreach (var a in left)
                    {
                        foreach (var b in right)
                        {
                            operands[0] = a;
                            operands[1] = b;
                            if (!ScalarOperationSemantics.TryEvaluate(value.Operation, operands, out var result))
                            {
                                return false;
                            }

                            values.Add((uint)result);
                            if (values.Count > MaxDescriptorTableEntries)
                            {
                                return false;
                            }
                        }
                    }

                    return true;
                }
                default:
                    return false;
            }
        }
        finally
        {
            visiting.Remove(value);
        }
    }

    private static string Describe(ScalarValue value, int depth) =>
        depth > 7 ? "..." : value.IsConstant ? $"0x{value.Payload:X}" :
        $"{value.Kind}{(value.Kind == ScalarValueKind.Operation ? ":" + value.Operation : "")}({string.Join(", ", value.Operands.Select(operand => Describe(operand, depth + 1)))})";

    // Reads the V#s a table can select for this draw and adds the buffers they describe as
    // written ranges; false (the stores stay unbounded) when some value or word cannot be read,
    // a V# is swizzled, or a loop selects more than MaxDescriptorTableEntries.
    private static int _tracedTableFailures;

    private static bool TraceTableFailure(ulong hash, string reason)
    {
        if (TraceDescriptorTables && Interlocked.Increment(ref _tracedTableFailures) <= 32)
        {
            Console.Error.WriteLine($"[DD-TABLE] hash=0x{hash:X16} not bounded for this draw: {reason}");
        }

        return false;
    }

    private static bool TryEvaluateDescriptorTable(
        ulong hash, RuntimeValueEvaluator evaluator, DeviceDescriptorTablePlan table, List<DeviceAddressRange> ranges)
    {
        Span<ulong> words = stackalloc ulong[4];
        if (table.Direct)
        {
            for (var word = 0; word < 4; word++)
            {
                if (!evaluator.EvaluateWide(table.Words[word], out words[word]))
                {
                    return TraceTableFailure(hash, $"descriptor word {word}");
                }
            }

            AddDescriptorRange(words, ranges);
            return true;
        }

        if (table.RecordTable is { } recordTable)
        {
            return TryEvaluateRecordTable(hash, evaluator, table, recordTable, ranges);
        }

        if (table.TableLow is null || table.TableHigh is null ||
            !evaluator.EvaluateWide(table.TableLow, out var low) || !evaluator.EvaluateWide(table.TableHigh, out var high))
        {
            return TraceTableFailure(hash, "table pointer");
        }

        var count = 0ul;
        if (table.Offsets is { } offsets)
        {
            count = (ulong)offsets.Count;
        }
        else
        {
            if (table.Limit is null || !evaluator.Evaluate(table.Limit, out var limit))
            {
                return TraceTableFailure(hash, "loop limit");
            }

            var end = table.LimitInclusive ? (ulong)limit + 1 : limit;
            count = end > table.Initial ? (end - table.Initial + table.Step - 1) / table.Step : 0;
            if (count > MaxDescriptorTableEntries)
            {
                return TraceTableFailure(hash, $"{count} entries");
            }
        }

        var found = new List<DeviceAddressRange>((int)count);
        for (var entry = 0ul; entry < count; entry++)
        {
            var offset = table.Offsets is { } listed
                ? listed[(int)entry]
                : unchecked(table.BaseOffset + (uint)(table.Initial + entry * table.Step) * table.Stride);
            for (var word = 0; word < 4; word++)
            {
                if (!evaluator.ReadRawWord(ScalarValueKind.ScalarAddressWord, low, high, offset, 0,
                        table.FirstImmediate + word * sizeof(uint), false, out words[word], table.Words[word]))
                {
                    return TraceTableFailure(hash, $"read 0x{low | (high << 32):X}+0x{offset:X} word {word}");
                }
            }

            AddDescriptorRange(words, found);
        }

        foreach (var range in found)
        {
            ranges.Add(range with { Handle = (uint)ranges.Count });
        }

        return true;
    }

    // Reads the V# field of every record the buffer holds. Records tend to share their target
    // buffer or lie side by side in one, so overlapping and adjacent ranges are merged.
    private static bool TryEvaluateRecordTable(
        ulong hash, RuntimeValueEvaluator evaluator, DeviceDescriptorTablePlan table, ScalarValue recordTable,
        List<DeviceAddressRange> ranges)
    {
        if (!evaluator.TryEvaluateRawBase(recordTable, ScalarValueKind.ScalarBufferWord, out var recordBase, out var recordHigh,
                out var recordWords))
        {
            return TraceTableFailure(hash, "record buffer");
        }

        var bufferStride = ((uint)recordHigh >> 16) & 0x3FFFu;
        var bytes = bufferStride == 0 ? (ulong)(uint)recordWords : (ulong)bufferStride * (uint)recordWords;
        var fieldEnd = (ulong)table.FirstImmediate + 4 * sizeof(uint);
        var count = bytes >= fieldEnd ? (bytes - fieldEnd) / table.RecordStride + 1 : 0;
        if (count > MaxRecordTableEntries)
        {
            return TraceTableFailure(hash, $"{count} records");
        }

        Span<ulong> words = stackalloc ulong[4];
        var found = new List<DeviceAddressRange>((int)count);
        for (var entry = 0ul; entry < count; entry++)
        {
            for (var word = 0; word < 4; word++)
            {
                if (!evaluator.ReadRawWord(ScalarValueKind.ScalarBufferWord, recordBase, recordHigh, (uint)(entry * table.RecordStride),
                        recordWords, table.FirstImmediate + word * sizeof(uint), false, out words[word], table.Words[word]))
                {
                    return TraceTableFailure(hash, $"record {entry} word {word}");
                }
            }

            AddDescriptorRange(words, found);
        }

        found.Sort(static (left, right) => left.Base.CompareTo(right.Base));
        var merged = 0;
        for (var index = 1; index < found.Count; index++)
        {
            var last = found[merged];
            var next = found[index];
            if (next.Base <= last.Base + last.Size)
            {
                found[merged] = last with { Size = Math.Max(last.Size, next.Base + next.Size - last.Base) };
            }
            else
            {
                found[++merged] = next;
            }
        }

        if (TraceDescriptorTables && Interlocked.Increment(ref _tracedRecordTables) <= 16)
        {
            Console.Error.WriteLine(
                $"[DD-TABLE] hash=0x{hash:X16} record table 0x{recordBase:X}: {count} records -> " +
                $"{(found.Count == 0 ? 0 : merged + 1)} ranges " +
                string.Join(", ", found.Take(found.Count == 0 ? 0 : Math.Min(merged + 1, 8)).Select(static range => $"0x{range.Base:X}+0x{range.Size:X}")));
        }

        for (var index = 0; found.Count != 0 && index <= merged; index++)
        {
            ranges.Add(found[index] with { Handle = (uint)ranges.Count });
        }

        return true;
    }

    private static int _tracedRecordTables;

    // The bytes a buffer V# reaches, added as a written range. A swizzled V#, or one that adds
    // the thread id, interleaves its records in groups of up to 64, so it reaches to the end of
    // the last whole group. An offset only some lane condition the enumeration cannot see
    // selects can hold data rather than a V#; its base is then null or unmapped, which the host
    // skips, as the shader drops the store.
    private static void AddDescriptorRange(ReadOnlySpan<ulong> words, List<DeviceAddressRange> ranges)
    {
        var bufferBase = (words[0] | ((words[1] & 0xFFFFul) << 32)) & AddressMask;
        var bufferStride = (uint)(words[1] >> 16) & 0x3FFFu;
        var records = (ulong)(uint)words[2];
        if ((words[1] & 0x8000_0000u) != 0 || (words[3] & (1u << 23)) != 0)
        {
            records = (records + 63) & ~63ul;
        }

        var size = records * Math.Max(bufferStride, 1u);
        if (bufferBase != 0 && size != 0 && size <= AddressMask - bufferBase + 1)
        {
            ranges.Add(new DeviceAddressRange((uint)ranges.Count, bufferBase, size, true, true));
        }
    }

    public static IReadOnlyList<DeviceAddressRangePlan> Plan(ShaderResourcePlan plan)
    {
        var ranges = new List<DeviceAddressRangePlan>();
        var handles = new List<ScalarValue>();
        var slotsByHash = new Dictionary<int, List<int>>();
        for (var index = 0; index < plan.Memory.Count; index++)
        {
            var memory = plan.Memory[index];
            if (memory.Kind is not (MemoryResourceKind.Flat or MemoryResourceKind.Global or MemoryResourceKind.ScalarAddress) || memory.PlanningOnly)
            {
                continue;
            }

            if (plan.Accesses[index]?.Handle is not { Kind: ScalarValueKind.AddressHandle } handle || handle.Operands.Length != 2)
            {
                continue;
            }

            if (memory.Kind == MemoryResourceKind.Flat && memory.Offset == 0 &&
                (memory.Opcode.StartsWith("FlatLoadDword", StringComparison.Ordinal) ||
                 memory.Opcode.StartsWith("FlatStoreDword", StringComparison.Ordinal)))
            {
                memory.AddressSpace = ClassifyFlatAddress(plan, handle.Operands[1], plan.Accesses[index]!.Active);
                if (memory.AddressSpace != FlatAddressSpace.Global ||
                    ExcludesGlobalAddress(plan, memory.Pc, handle.Operands[1]))
                {
                    continue;
                }
            }

            var hash = plan.Graph.StructuralHash(handle);
            if (!slotsByHash.TryGetValue(hash, out var hashSlots))
            {
                hashSlots = [];
                slotsByHash.Add(hash, hashSlots);
            }

            var slot = -1;
            foreach (var existing in hashSlots)
            {
                if (plan.Graph.Equivalent(handles[existing], handle))
                {
                    slot = existing;
                    break;
                }
            }

            if (slot < 0)
            {
                slot = handles.Count;
                hashSlots.Add(slot);
                handles.Add(handle);
                ranges.Add(new DeviceAddressRangePlan((uint)slot, handle.Operands[0], handle.Operands[1], [], false, true, long.MaxValue, long.MinValue, false));
            }

            // The immediate offset is signed; the span keeps the lowest and highest byte.
            var written = memory.Access is MemoryAccess.Write or MemoryAccess.Atomic;
            var bytes = Math.Max((memory.DataBits + 7) / 8, 1u) * Math.Max(memory.DataDwords, 1u);
            var firstByte = (long)unchecked((int)memory.Offset);
            ranges[slot] = ranges[slot] with
            {
                MemoryIndices = [.. ranges[slot].MemoryIndices, index],
                Written = ranges[slot].Written || written,
                Bounded = ranges[slot].Bounded && IsBoundedOffset(plan, index),
                FirstByte = Math.Min(ranges[slot].FirstByte, firstByte),
                EndByte = Math.Max(ranges[slot].EndByte, firstByte + bytes),
            };
        }

        for (var slot = 0; slot < ranges.Count; slot++)
        {
            ranges[slot] = ranges[slot] with
            {
                Plannable = plan.ValidateRuntimeValue(ranges[slot].BaseLow) && plan.ValidateRuntimeValue(ranges[slot].BaseHigh),
            };
        }

        return ranges;
    }

    private static FlatAddressSpace ClassifyFlatAddress(ShaderResourcePlan plan, ScalarValue high, ScalarValue? active)
    {
        var range = UnsignedRange(plan, high, active, []);
        var (low, highBound) = IsEmpty(range) ? (0u, uint.MaxValue) : range;
        var lowShared = Gen5InlineConstants.IsSharedApertureHigh(low);
        var lowPrivate = Gen5InlineConstants.IsPrivateApertureHigh(low);
        var highShared = Gen5InlineConstants.IsSharedApertureHigh(highBound);
        var highPrivate = Gen5InlineConstants.IsPrivateApertureHigh(highBound);
        // This interval excludes global memory. Execution still checks the exact tag.
        if ((lowShared || lowPrivate) && (highShared || highPrivate))
        {
            return (lowShared || highShared, lowPrivate || highPrivate) switch
            {
                (true, true) => FlatAddressSpace.SharedOrPrivate,
                (true, false) => FlatAddressSpace.Shared,
                _ => FlatAddressSpace.Private,
            };
        }

        return ClassifyLocalAlternatives(plan.Graph, high, []);
    }

    // Unknown paths retain the full range. Only proven bounds can exclude global memory.
    private static readonly (uint Low, uint High) Empty = (uint.MaxValue, 0u);

    private static bool IsEmpty((uint Low, uint High) range) => range.Low > range.High;

    private static (uint Low, uint High) Union((uint Low, uint High) left, (uint Low, uint High) right) =>
        IsEmpty(left) ? right : IsEmpty(right) ? left : (Math.Min(left.Low, right.Low), Math.Max(left.High, right.High));

    private static (uint Low, uint High) UnsignedRange(ShaderResourcePlan plan, ScalarValue value, ScalarValue? active,
        HashSet<ScalarValue> visiting)
    {
        if (value.Kind == ScalarValueKind.Constant && value.Type == ScalarValueType.U32)
            return (value.ConstantU32, value.ConstantU32);
        if (value.Kind == ScalarValueKind.MemoryAperture)
        {
            var aperture = (uint)(Gen5InlineConstants.DecodeAperture64((uint)value.Payload) >> 32);
            return (aperture, aperture);
        }

        if (!visiting.Add(value))
            return (0, uint.MaxValue);
        try
        {
            switch (value.Kind)
            {
                case ScalarValueKind.Select when value.Operands.Length == 3 &&
                    active is not null && (ReferenceEquals(value.Operands[0], active) || plan.Graph.Equivalent(value.Operands[0], active)):
                    return UnsignedRange(plan, value.Operands[1], active, visiting);
                case ScalarValueKind.Select when value.Operands.Length == 3:
                    return Union(UnsignedRange(plan, value.Operands[1], active, visiting),
                        UnsignedRange(plan, value.Operands[2], active, visiting));
                case ScalarValueKind.Phi:
                {
                    var range = Empty;
                    foreach (var operand in value.Operands)
                        range = Union(range, UnsignedRange(plan, operand, active, visiting));
                    return range;
                }
                case ScalarValueKind.Operation when value.Operation is ScalarOperation.UMin32 or ScalarOperation.UMax32:
                {
                    var left = UnsignedRange(plan, value.Operands[0], active, visiting);
                    var right = UnsignedRange(plan, value.Operands[1], active, visiting);
                    if (IsEmpty(left) || IsEmpty(right))
                        return (0, uint.MaxValue);
                    return value.Operation == ScalarOperation.UMin32
                        ? (Math.Min(left.Low, right.Low), Math.Min(left.High, right.High))
                        : (Math.Max(left.Low, right.Low), Math.Max(left.High, right.High));
                }
                default:
                    return (0, uint.MaxValue);
            }
        }
        finally
        {
            visiting.Remove(value);
        }
    }

    private static bool ExcludesGlobalAddress(ShaderResourcePlan plan, uint pc, ScalarValue high)
    {
        var activeHigh = ActiveAddressHigh(plan, pc, high, out var minimumHigh);
        return minimumHigh >= 0x10000 || IsLocalAperture(plan.Graph, activeHigh, new HashSet<ScalarValue>());
    }

    private static ScalarValue ActiveAddressHigh(ShaderResourcePlan plan, uint pc, ScalarValue high, out uint minimumHigh)
    {
        minimumHigh = 0;
        var instructions = plan.Graph.Program.Instructions;
        var index = -1;
        for (var position = 0; position < instructions.Count; position++)
            if (instructions[position].Pc == pc) { index = position; break; }
        if (index < 0 || instructions[index].Control is not Gen5GlobalMemoryControl control) return high;
        var block = plan.Graph.ControlFlow.Blocks.First(block => pc >= block.StartPc && pc < block.EndPc);
        for (var position = index - 1; position >= 0 && instructions[position].Pc >= block.StartPc; position--)
        {
            var instruction = instructions[position];
            if (instruction.Opcode.StartsWith("VCmp", StringComparison.Ordinal) ||
                instruction.Destinations.Any(destination => destination.Kind == Gen5OperandKind.ScalarRegister && destination.Value >= 126)) break;
            if (instruction.Encoding is not (Gen5ShaderEncoding.Vop1 or Gen5ShaderEncoding.Vop2 or
                Gen5ShaderEncoding.Vop3 or Gen5ShaderEncoding.Vop3p) && instruction.Opcode is not ("SWaitcnt" or "SNop")) break;
            if (instruction.Destinations.Any(destination => destination.Kind == Gen5OperandKind.VectorRegister &&
                destination.Value == control.VectorAddress + 1))
            {
                // The address write and memory access use the same EXEC mask in this block.
                if (plan.Graph.UnsignedMedianSources.TryGetValue(instruction.Pc, out var sources))
                {
                    var constants = new List<uint>();
                    foreach (var source in sources)
                        if (TryConstant(plan.Graph, source, new HashSet<ScalarValue>(), out var constant))
                            constants.Add((uint)constant);
                    // Two constant operands bound an unsigned median from below.
                    if (constants.Count >= 2) minimumHigh = constants.Min();
                }
                return plan.Graph.ConditionalMaskResults.TryGetValue(instruction.Pc, out var result) ? result : high;
            }
        }
        return high;
    }

    private static bool IsLocalAperture(ScalarValueGraph graph, ScalarValue value, HashSet<ScalarValue> visiting) =>
        ClassifyLocalAlternatives(graph, value, visiting) != FlatAddressSpace.Global;

    private static FlatAddressSpace ClassifyLocalAlternatives(
        ScalarValueGraph graph, ScalarValue value, HashSet<ScalarValue> visiting)
    {
        value = graph.ResolveInvariantPhi(value) ?? value;
        if (TryConstant(graph, value, [], out var constant))
        {
            if (constant == Gen5InlineConstants.SharedApertureBase >> 32 ||
                constant == Gen5InlineConstants.SharedFlatApertureBase >> 32)
                return FlatAddressSpace.Shared;
            if (constant == Gen5InlineConstants.PrivateApertureBase >> 32 ||
                constant == Gen5InlineConstants.PrivateFlatApertureBase >> 32)
                return FlatAddressSpace.Private;
            return FlatAddressSpace.Global;
        }

        if (!visiting.Add(value)) return FlatAddressSpace.Global;
        try
        {
            var alternatives = value.Kind == ScalarValueKind.Select ? value.Operands.Skip(1) :
                value.Kind == ScalarValueKind.Phi ? value.Operands.AsEnumerable() : [];
            FlatAddressSpace? result = null;
            foreach (var alternative in alternatives)
            {
                var classification = ClassifyLocalAlternatives(graph, alternative, visiting);
                if (classification == FlatAddressSpace.Global) return FlatAddressSpace.Global;
                result = result is null || result == classification
                    ? classification : FlatAddressSpace.SharedOrPrivate;
            }
            return result ?? FlatAddressSpace.Global;
        }
        finally
        {
            visiting.Remove(value);
        }
    }

    private static bool TryConstant(ScalarValueGraph graph, ScalarValue value, HashSet<ScalarValue> visiting, out ulong result)
    {
        value = graph.ResolveInvariantPhi(value) ?? value;
        result = value.Payload;
        if (value.Kind == ScalarValueKind.MemoryAperture)
        {
            result = Gen5InlineConstants.DecodeAperture64((uint)value.Payload) >> 32;
            return true;
        }
        if (value.IsConstant) return true;
        if (value.Kind == ScalarValueKind.Phi && visiting.Add(value))
        {
            var pending = new Stack<ScalarValue>();
            var seen = new HashSet<ScalarValue>();
            pending.Push(value);
            ulong? candidate = null;
            var validPhi = true;
            while (pending.TryPop(out var current))
            {
                if (!seen.Add(current)) continue;
                if (current.Kind == ScalarValueKind.Phi)
                {
                    if (current.Operands.Length == 0) validPhi = false;
                    foreach (var incoming in current.Operands) pending.Push(incoming);
                }
                else if (!TryConstant(graph, current, visiting, out var incomingValue) ||
                    candidate.HasValue && candidate.Value != incomingValue) validPhi = false;
                else candidate = incomingValue;
            }
            visiting.Remove(value);
            result = candidate ?? 0;
            return validPhi && candidate.HasValue;
        }
        if (value.Kind != ScalarValueKind.Operation || !visiting.Add(value)) return false;
        var operands = new ulong[value.Operands.Length];
        var valid = true;
        for (var index = 0; index < operands.Length; index++)
            valid &= TryConstant(graph, value.Operands[index], visiting, out operands[index]);
        visiting.Remove(value);
        if (!valid || !ScalarOperationSemantics.TryEvaluate(value.Operation, operands, out result)) return false;
        if (value.Type == ScalarValueType.U32) result = (uint)result;
        return true;
    }

    // An access is bounded when its only run-time term is the record's immediate offset:
    // a scalar read, or a global access whose vector offset is a constant zero.
    private static bool IsBoundedOffset(ShaderResourcePlan plan, int memoryIndex)
    {
        var memory = plan.Memory[memoryIndex];
        if (memory.Kind == MemoryResourceKind.ScalarAddress)
        {
            var read = plan.Accesses[memoryIndex]?.Read;
            return read is not null && read.Operands[1].IsConstant;
        }

        var offset = plan.Accesses[memoryIndex]?.Offset;
        return offset is { IsConstant: true } && offset.ConstantU32 == 0;
    }

    // Evaluates the planned ranges for one draw. A bounded range spans its lowest byte to
    // its extent; an unbounded one starts at the lowest immediate and takes the cap.
    public static DeviceAddressRange[] Evaluate(ShaderResourcePlan plan, ResourceRuntimeInputs inputs)
    {
        var ranges = plan.DeviceAddressRanges;
        var result = new DeviceAddressRange[ranges.Count];
        using var scratch = RuntimeEvaluationScratch.Rent();
        var evaluator = new RuntimeValueEvaluator(scratch, plan, inputs);
        for (var index = 0; index < ranges.Count; index++)
        {
            var range = ranges[index];
            uint low = 0;
            uint high = 0;
            var planned = range.Plannable && evaluator.Evaluate(range.BaseLow, out low) && evaluator.Evaluate(range.BaseHigh, out high);
            var handleBase = (low | ((ulong)high << 32)) & AddressMask;
            var firstByte = range.Bounded ? range.FirstByte : Math.Min(range.FirstByte, 0);
            var baseAddress = planned ? unchecked(handleBase + (ulong)firstByte) & AddressMask : 0;
            var size = range.Bounded ? range.Extent : MaxRangeBytes;
            result[index] = new DeviceAddressRange(range.Handle, baseAddress, planned ? size : 0, planned, range.Written);
        }

        if (plan.Graph.Program.FunctionBufferAccesses.Count == 0 && plan.DeviceDescriptorTables.Count == 0) return result;
        var functionRanges = new List<DeviceAddressRange>(result);
        var seen = new HashSet<(ulong Base, ulong Size, bool Written)>();
        foreach (var memory in plan.Memory.Entries)
        {
            if (!memory.DeviceDescriptor || !plan.Graph.Program.FunctionBufferAccesses.Contains(memory.Pc)) continue;
            if (!plan.Memory.TryGetIndex(memory.Pc, memory.ComponentIndex, out var memoryIndex) ||
                plan.Accesses[memoryIndex]?.Handle is not { Kind: ScalarValueKind.BufferHandle, Operands.Length: 4 } handle)
                throw new ResourcePlanException("The shader function buffer has no descriptor.");
            var words = new uint[4];
            var resolved = true;
            for (var component = 0; component < 4 && resolved; component++)
                resolved = evaluator.Evaluate(handle.Operands[component], out words[component]);
            if (!resolved)
            {
                // A V# of a callee linked with run-time arguments: the host resolves it for each record (CallRecordCallee).
                if (plan.CallRecordCallees.Count != 0) continue;
                throw new ResourcePlanException("The shader function buffer range cannot be resolved.");
            }
            var address = words[0] | ((ulong)(words[1] & 0xFFFF) << 32);
            if ((words[1] & 0x80000000) != 0)
                throw new ResourcePlanException("The shader function uses an unsupported swizzled buffer.");
            var stride = (words[1] >> 16) & 0x3FFF;
            var size = (ulong)words[2] * Math.Max(stride, 1u);
            var written = memory.Access is MemoryAccess.Write or MemoryAccess.Atomic;
            if (size == 0) continue;
            if (size > AddressMask - address + 1)
                throw new ResourcePlanException("The shader function buffer range wraps the guest address space.");
            if (seen.Add((address, size, written)))
                functionRanges.Add(new((uint)functionRanges.Count, address, size, true, written));
        }

        // A table that cannot be bounded for this draw, or a device-loaded-descriptor store no
        // table covers, leaves a written handle unplanned: the hosts treat that as unbounded.
        var unboundedTable = false;
        foreach (var table in plan.DeviceDescriptorTables)
            unboundedTable |= !TryEvaluateDescriptorTable(plan.Hash, evaluator, table, functionRanges);
        if (unboundedTable || (plan.DeviceDescriptorTables.Count != 0 && plan.UnplannedDeviceDescriptorWrites))
            functionRanges.Add(new((uint)functionRanges.Count, 0, 0, false, true));
        return functionRanges.ToArray();
    }
}
