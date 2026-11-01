// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Generic;

namespace SharpEmu.ShaderCompiler.Resources;

// Recognises a runtime V# whose four dwords come from one static scalar buffer (the SRT)
// at a dynamic offset driven by a canonical unsigned induction loop, and proves that the
// loop's guard bounds the descriptors the access can select. The result is a pure
// description of the candidate set; no descriptor is invented and no memory bound is
// promoted into a semantic proof.
internal static class BufferCandidateTablePlanner
{
    public static bool TryPlan(ShaderResourcePlan plan, ScalarValue handle, int memoryIndex, out BufferCandidateTablePlan table)
    {
        table = null!;
        if (handle.Kind != ScalarValueKind.BufferHandle || handle.Operands.Length != 4)
        {
            return false;
        }


        // Every dword must be the same-width scalar-buffer read at a shared dynamic offset,
        // with the component immediates 0,4,8,12 that one dwordx4 read produces.
        ScalarValue? srt = null;
        ScalarValue? dynamicOffset = null;
        for (var word = 0; word < 4; word++)
        {
            var read = handle.Operands[word];
            if (read.Kind != ScalarValueKind.ScalarBufferWord || read.MemoryIndex < 0 || read.MemoryIndex >= plan.Memory.Count)
            {
                return false;
            }

            var memory = plan.Memory[read.MemoryIndex];
            if (memory.Kind != MemoryResourceKind.ScalarBuffer || memory.DataBits != 32 || memory.DataDwords != 1 ||
                memory.Offset != (uint)word * sizeof(uint))
            {
                return false;
            }

            if (srt is null)
            {
                srt = read.Operands[0];
            }
            else if (!plan.Graph.Equivalent(srt, read.Operands[0]))
            {
                return false;
            }

            if (dynamicOffset is null)
            {
                dynamicOffset = read.Operands[1];
            }
            else if (!plan.Graph.Equivalent(dynamicOffset, read.Operands[1]))
            {
                return false;
            }
        }

        // offset = base + index * stride
        var offset = dynamicOffset!;
        var baseOffset = 0u;
        if (offset.Kind == ScalarValueKind.Operation && offset.Operation == ScalarOperation.IAdd32)
        {
            if (offset.Operands[0].IsConstant)
            {
                baseOffset = offset.Operands[0].ConstantU32;
                offset = offset.Operands[1];
            }
            else if (offset.Operands[1].IsConstant)
            {
                baseOffset = offset.Operands[1].ConstantU32;
                offset = offset.Operands[0];
            }
        }

        if (offset.Kind != ScalarValueKind.Operation || offset.Operation != ScalarOperation.IMul32)
        {
            return false;
        }

        ScalarValue index;
        uint stride;
        if (offset.Operands[0].IsConstant)
        {
            stride = offset.Operands[0].ConstantU32;
            index = offset.Operands[1];
        }
        else if (offset.Operands[1].IsConstant)
        {
            stride = offset.Operands[1].ConstantU32;
            index = offset.Operands[0];
        }
        else
        {
            return false;
        }

        if (stride < BufferCandidateTablePlan.DescriptorByteSize)
        {
            return false;
        }

        if (!TryProveInductionBound(plan, index, out var initial, out var step, out var limit, out var inclusive) || limit is null)
        {
            return false;
        }

        if (!initial.IsConstant)
        {
            return false;
        }

        var srtExtent = SrtByteExtent(plan, srt!);
        var minOffset = unchecked(baseOffset + initial.ConstantU32 * stride);
        var candidate = new BufferCandidateTablePlan
        {
            SrtHandle = srt!,
            OffsetExpression = dynamicOffset!,
            IndexExpression = index,
            Initial = initial,
            Step = step,
            Limit = limit.IsConstant ? null : limit,
            LimitInclusive = inclusive,
            Stride = stride,
            BaseOffset = baseOffset,
            MinOffset = minOffset,
            SrtByteExtent = srtExtent,
            MemoryIndices = [memoryIndex],
        };

        int count;
        uint maxOffset;
        if (limit.IsConstant)
        {
            if (!candidate.TryResolveCount(limit.ConstantU32, out count))
            {
                return false;
            }

            if (count <= 0 || count > candidate.Cap)
            {
                return false;
            }

            maxOffset = unchecked(candidate.StaticCandidateOffset(count - 1) + BufferCandidateTablePlan.DescriptorByteSize);
        }
        else
        {
            if (!plan.ValidateRuntimeValue(limit))
            {
                return false;
            }

            // A run-time limit is capped and memory-checked at materialisation; the static
            // start of the range still has to stay inside the declared SRT extent.
            count = -1;
            maxOffset = uint.MaxValue;
        }

        if (maxOffset < minOffset || (maxOffset != uint.MaxValue && maxOffset > srtExtent))
        {
            return false;
        }

        table = new BufferCandidateTablePlan
        {
            SrtHandle = srt!,
            OffsetExpression = dynamicOffset!,
            IndexExpression = index,
            Initial = initial,
            Step = step,
            Limit = limit.IsConstant ? null : limit,
            LimitInclusive = inclusive,
            Stride = stride,
            BaseOffset = baseOffset,
            MinOffset = minOffset,
            MaxOffset = maxOffset,
            Count = count,
            SrtByteExtent = srtExtent,
            MemoryIndices = [memoryIndex],
            Cap = candidate.Cap,
        };
        return true;
    }

    // A canonical unsigned induction: the value is a phi with a constant initial and an
    // add-recurrence, and a branch guarded by an unsigned comparison against the phi
    // lives at the loop header or one of its predecessors. With exitGuards, a header whose
    // branch leaves the loop when the comparison fails (a top-tested loop: s_cmp_lt_u32 then
    // s_cbranch_scc0 past the latch) bounds it as well.
    internal static bool TryProveInductionBound(
        ShaderResourcePlan plan,
        ScalarValue index,
        out ScalarValue initial,
        out uint step,
        out ScalarValue? limit,
        out bool inclusive,
        bool exitGuards = false) =>
        TryProveInductionBound(plan, index, out initial, out step, out limit, out inclusive, out _, exitGuards, signedGuards: false);

    // The counter's start and step alone, for a loop whose bound the branch conditions do not tie to this phi (they were
    // recorded against the counter's earlier version in a nested loop).
    internal static bool TryProveInductionStep(ScalarValue index, out ScalarValue initial, out uint step)
    {
        initial = null!;
        step = 0;
        if (index.Kind != ScalarValueKind.Phi)
        {
            return false;
        }

        ScalarValue? found = null;
        var initialCount = 0;
        foreach (var operand in index.Operands)
        {
            if (operand is { Kind: ScalarValueKind.Operation, Operation: ScalarOperation.IAdd32 } &&
                (ReferenceEquals(operand.Operands[0], index) || ForwardsIndex(operand.Operands[0], index)) && operand.Operands[1].IsConstant)
            {
                step = operand.Operands[1].ConstantU32;
            }
            else if (!ReferenceEquals(operand, index))
            {
                found = operand;
                initialCount++;
            }
        }

        if (step == 0 || initialCount != 1 || found is null)
        {
            return false;
        }

        initial = found;
        return true;
    }

    // With signedGuards a signed comparison bounds the counter as well (s_cmp_lt_i32 against an int count, a C for
    // loop); signedLimit then tells the caller to order the counter and the bound as signed values, which a bound
    // below zero (a loop that never runs) needs.
    internal static bool TryProveInductionBound(
        ShaderResourcePlan plan,
        ScalarValue index,
        out ScalarValue initial,
        out uint step,
        out ScalarValue? limit,
        out bool inclusive,
        out bool signedLimit,
        bool exitGuards,
        bool signedGuards)
    {
        initial = null!;
        step = 0;
        limit = null;
        inclusive = false;
        signedLimit = false;
        if (index.Kind != ScalarValueKind.Phi)
        {
            return false;
        }

        // An inner loop carries the counter through a phi that only merges it with itself, and the increment at the
        // outer latch adds to that phi: it is the counter.
        bool IsIndex(ScalarValue value) => ReferenceEquals(value, index) || ForwardsIndex(value, index);

        ScalarValue? found = null;
        var initialCount = 0;
        foreach (var operand in index.Operands)
        {
            if (operand.Kind == ScalarValueKind.Operation && operand.Operation == ScalarOperation.IAdd32)
            {
                if (IsIndex(operand.Operands[0]) && operand.Operands[1].IsConstant)
                {
                    step = operand.Operands[1].ConstantU32;
                }
                else if (IsIndex(operand.Operands[1]) && operand.Operands[0].IsConstant)
                {
                    step = operand.Operands[0].ConstantU32;
                }
                else
                {
                    continue;
                }
            }
            else if (!ReferenceEquals(operand, index))
            {
                found = operand;
                initialCount++;
            }
        }

        if (step == 0 || initialCount != 1 || found is null)
        {
            return false;
        }

        initial = found;

        var flow = plan.Graph.ControlFlow;
        var guardBlock = (int?)null;
        foreach (var pair in plan.Graph.BranchConditions)
        {
            var condition = pair.Value;
            var exitGuard = false;
            if (exitGuards && condition is { Kind: ScalarValueKind.Operation, Operation: ScalarOperation.LogicalNot })
            {
                // s_cselect_b64 vcc, exec, 0 then s_cbranch_vccz: the loop stays only while the comparison holds.
                var stays = condition.Operands[0];
                if (stays is { Kind: ScalarValueKind.Select } && stays.Operands[2] is { IsConstant: true, ConstantU64: 0 })
                {
                    stays = stays.Operands[0];
                }

                if (stays.Kind == ScalarValueKind.Operation)
                {
                    condition = stays;
                    exitGuard = true;
                }
            }

            if (condition.Kind != ScalarValueKind.Operation)
            {
                continue;
            }

            ScalarValue? bound;
            bool boundInclusive;
            var boundSigned = condition.Operation is ScalarOperation.SLessThan32 or ScalarOperation.SLessThanEqual32 or
                ScalarOperation.SGreaterThan32 or ScalarOperation.SGreaterThanEqual32;
            if (boundSigned && !signedGuards)
            {
                continue;
            }

            switch (condition.Operation)
            {
                case ScalarOperation.ULessThan32 or ScalarOperation.SLessThan32
                    when MatchesIndex(plan, condition.Operands[0], index, step):
                    bound = condition.Operands[1];
                    boundInclusive = false;
                    break;
                case ScalarOperation.ULessThanEqual32 or ScalarOperation.SLessThanEqual32
                    when MatchesIndex(plan, condition.Operands[0], index, step):
                    bound = condition.Operands[1];
                    boundInclusive = true;
                    break;
                case ScalarOperation.UGreaterThan32 or ScalarOperation.SGreaterThan32
                    when MatchesIndex(plan, condition.Operands[1], index, step):
                    bound = condition.Operands[0];
                    boundInclusive = false;
                    break;
                case ScalarOperation.UGreaterThanEqual32 or ScalarOperation.SGreaterThanEqual32
                    when MatchesIndex(plan, condition.Operands[1], index, step):
                    bound = condition.Operands[0];
                    boundInclusive = true;
                    break;
                default:
                    continue;
            }

            if (!bound.IsConstant && !plan.ValidateRuntimeValue(bound))
            {
                continue;
            }

            var block = BlockOf(plan, pair.Key);
            if (block < 0)
            {
                continue;
            }

            // The guard must live at the loop header that owns the phi or one of the
            // blocks feeding it, so it really controls the recurrence. An exit guard runs
            // before every iteration: at the header or a block every iteration passes
            // through straight after it (a header that first tests exec).
            var header = index.PhiBlock;
            if (exitGuard ? !RunsEveryIteration(flow, block, header) || !BranchLeavesLoop(flow, block, header)
                : block != header && !flow.Predecessors[header].Contains(block))
            {
                continue;
            }

            if (limit is not null && (!plan.Graph.Equivalent(limit, bound) || signedLimit != boundSigned))
            {
                return false;
            }

            limit = bound;
            inclusive = boundInclusive;
            signedLimit = boundSigned;
            guardBlock = block;
        }

        return limit is not null && guardBlock is not null;
    }

    // The block is the header, or reached from it through loop blocks that each have one
    // successor inside the loop, so an iteration that stays in the loop runs it.
    private static bool RunsEveryIteration(Ir.IrControlFlowGraph flow, int block, int header)
    {
        var latch = -1;
        foreach (var predecessor in flow.Predecessors[header])
        {
            latch = Math.Max(latch, predecessor >= header ? predecessor : -1);
        }

        var current = header;
        while (current != block)
        {
            var next = -1;
            foreach (var successor in flow.Successors[current])
            {
                if (successor < header || successor > latch)
                {
                    continue;
                }

                if (next >= 0)
                {
                    return false;
                }

                next = successor;
            }

            if (next <= current)
            {
                return false;
            }

            current = next;
        }

        return true;
    }

    // The block's taken branch leaves the loop the header heads (past its last latch) and its
    // fall-through stays in it, so the loop runs only while the branch is not taken.
    private static bool BranchLeavesLoop(Ir.IrControlFlowGraph flow, int block, int header)
    {
        var latch = -1;
        foreach (var predecessor in flow.Predecessors[header])
        {
            latch = Math.Max(latch, predecessor >= header ? predecessor : -1);
        }

        var successors = flow.Successors[block];
        if (latch < 0 || successors.Count != 2 ||
            !flow.BlockByStartPc.TryGetValue(flow.Blocks[block].EndPc, out var fallThrough))
        {
            return false;
        }

        var taken = successors[0] == fallThrough ? successors[1] : successors[1] == fallThrough ? successors[0] : -1;
        return taken >= 0 && (taken > latch || taken < header) && fallThrough >= header && fallThrough <= latch;
    }

    // The guard compares either the induction value itself or its next value; both bound
    // the same executed set, so a one-step add of the phi is accepted.
    // Whether a phi only passes the counter on: every operand, through the phis between them, is the counter itself or
    // one of those phis - an inner loop's copy of the outer loop's counter, which the increment at the outer latch reads.
    internal static bool ForwardsIndex(ScalarValue value, ScalarValue index)
    {
        if (value.Kind != ScalarValueKind.Phi || ReferenceEquals(value, index))
        {
            return false;
        }

        var visited = new HashSet<ScalarValue> { value };
        var pending = new Stack<ScalarValue>(value.Operands);
        while (pending.TryPop(out var operand))
        {
            if (ReferenceEquals(operand, index) || !visited.Add(operand))
            {
                continue;
            }

            if (operand.Kind != ScalarValueKind.Phi || visited.Count > 64)
            {
                return false;
            }

            foreach (var inner in operand.Operands)
            {
                pending.Push(inner);
            }
        }

        return true;
    }

    private static bool MatchesIndex(ShaderResourcePlan plan, ScalarValue operand, ScalarValue index, uint step)
    {
        if (plan.Graph.Equivalent(operand, index) || ForwardsIndex(operand, index))
        {
            return true;
        }

        bool IsIndex(ScalarValue value) => plan.Graph.Equivalent(value, index) || ForwardsIndex(value, index);

        if (operand.Kind != ScalarValueKind.Operation || operand.Operation != ScalarOperation.IAdd32)
        {
            return false;
        }

        if (IsIndex(operand.Operands[0]) && operand.Operands[1].IsConstant)
        {
            return operand.Operands[1].ConstantU32 == step;
        }

        if (IsIndex(operand.Operands[1]) && operand.Operands[0].IsConstant)
        {
            return operand.Operands[0].ConstantU32 == step;
        }

        return false;
    }

    private static int BlockOf(ShaderResourcePlan plan, uint pc)
    {
        var blocks = plan.Graph.ControlFlow.Blocks;
        for (var index = 0; index < blocks.Count; index++)
        {
            if (pc >= blocks[index].StartPc && pc < blocks[index].EndPc)
            {
                return index;
            }
        }

        return -1;
    }

    // The SRT's declared byte size when both its stride and record count are constants.
    // This is a memory-safety bound, never evidence that any entry is a valid descriptor.
    private static uint SrtByteExtent(ShaderResourcePlan plan, ScalarValue srt)
    {
        if (srt.Operands.Length != 4)
        {
            return uint.MaxValue;
        }

        var strideWord = plan.Graph.ResolveInvariantPhi(srt.Operands[1]) ?? srt.Operands[1];
        var recordsWord = plan.Graph.ResolveInvariantPhi(srt.Operands[2]) ?? srt.Operands[2];
        if (!strideWord.IsConstant || !recordsWord.IsConstant)
        {
            return uint.MaxValue;
        }

        var stride = (strideWord.ConstantU32 >> 16) & 0x3FFFu;
        var records = recordsWord.ConstantU32;
        var size = stride == 0 ? records : (ulong)stride * records;
        return size > uint.MaxValue ? uint.MaxValue : (uint)size;
    }
}
