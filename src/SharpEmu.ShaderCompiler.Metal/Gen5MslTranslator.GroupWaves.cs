// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text;

namespace SharpEmu.ShaderCompiler.Metal;

// Where the two halves of a bridged wave64 meet, in a compute group of several waves
// (IsBridgedComputeWaves).
//
// Each guest wave runs as two 32-wide simdgroups whose exchanges meet at the wave's own pair
// barrier: a counter in threadgroup memory that each half counts into and then polls until its
// partner has counted in too. The waves of the group may take their own trip counts through
// loops, which is why a threadgroup barrier cannot stand in for it in general. But a half that
// polls keeps an Apple GPU core busy, and the core shares its issue slots among every resident
// simdgroup. GTA V's BVH-build radix sort (a onesweep scatter, compute 220AB61E9C35767C) has
// each workgroup spin on device memory until the workgroups before it post their digit
// counts; with its exchanges on pair barriers, the posting groups' halves sat polling for each
// other behind the spinning groups for whole milliseconds, the spinners ran out of steps, and
// the sort came out with keys on top of each other (10,734 duplicate ids in a 29,683-key sort).
// Three things keep the halves off the poll:
//
//  - An exchange in a block every wave of the group runs the same number of times, in the
//    same order, meets at a threadgroup barrier: the hardware parks the waiting simdgroups
//    instead of letting them poll. A block is such a block when no branch on a value that can
//    differ between the waves (EXEC, VCC, or SCC from a scalar that came from lane data)
//    decides whether, or how often, it runs (FindGroupUniformBlocks).
//
//  - A loop whose body touches nothing outside its own lanes (no cross-lane read, no LDS, no
//    scalar write but EXEC) and leaves on EXEC alone runs in each half on its own: the half
//    whose lanes are all done leaves, and the halves meet again at the first exchange after
//    the loop (FindHalfLocalLoops). A decoupled look-back's wait is such a loop; bridged, it
//    exchanged EXEC twice per poll.
//
//  - A pair barrier that still has to wait reads device memory between polls, which stalls
//    the polling simdgroup long enough for its partner to be scheduled (Wave64Barrier).
public static partial class Gen5MslTranslator
{
    private sealed partial class CompilationContext
    {
        /// <summary>An exchange of a group-uniform block: the bridged wave's halves meet with
        /// every other wave of the group. Spelled apart from S_BARRIER's statement so the
        /// structured body can be checked for a block it emitted twice.</summary>
        private const string GroupWaveBarrier = "{ threadgroup_barrier(mem_flags::mem_threadgroup); }";

        private const int SccSlotIndex = 128;
        private const int UniformSlots = 129;

        /// <summary>The blocks of loops that each half runs on its own (null when none).</summary>
        private bool[]? _halfLocalLoopBlocks;

        /// <summary>Set when a half-local loop still holds an exchange once the mask ballots are
        /// resolved; the program is then compiled again without half-local loops.</summary>
        internal bool HalfLocalLoopExchanged { get; private set; }

        private bool IsHalfLocalLoopBlock(int block) =>
            _halfLocalLoopBlocks is { } blocks && block < blocks.Length && blocks[block];

        /// <summary>The EXEC test of a half-local loop's branch: this half's lanes only.</summary>
        private static string HalfLocalExecCondition(bool zero) =>
            zero ? "(sharpemu_ballot(exec) == 0u)" : "(sharpemu_ballot(exec) != 0u)";

        /// <summary>Marks the loops each half of a bridged compute wave may run on its own: a
        /// back edge's span of blocks that is entered only at its header, branches only on
        /// EXEC, and holds only lane-local work (<see cref="IsHalfLocalInstruction"/>).</summary>
        private void FindHalfLocalLoops(IReadOnlyList<ShaderBlock> blocks)
        {
            if (!IsBridgedComputeWaves || !_options.HalfLocalWaveLoops)
            {
                return;
            }

            var instructions = _request.Program.Instructions;
            var targets = new int[blocks.Count];
            for (var block = 0; block < blocks.Count; block++)
            {
                var last = instructions[blocks[block].EndIndex - 1];
                targets[block] = last.Opcode.StartsWith("SCbranch", StringComparison.Ordinal) || last.Opcode == "SBranch"
                    ? (IsExitBranchTarget(instructions, last) ? ExitTarget
                        : TryGetBranchTargetBlock(blocks, last, out var target) ? target : int.MinValue)
                    : ExitTarget;
            }

            bool[]? marks = null;
            for (var latch = 0; latch < blocks.Count; latch++)
            {
                var header = targets[latch];
                if (header < 0 || header > latch || !IsHalfLocalLoop(blocks, targets, header, latch))
                {
                    continue;
                }

                marks ??= new bool[blocks.Count];
                for (var block = header; block <= latch; block++)
                {
                    marks[block] = true;
                }
            }

            _halfLocalLoopBlocks = marks;
        }

        /// <summary>A back edge's span that each half may run on its own. It must also leave
        /// to one place only: halves that left by different exits would run different code to
        /// their next exchange.</summary>
        private bool IsHalfLocalLoop(IReadOnlyList<ShaderBlock> blocks, int[] targets, int header, int latch)
        {
            var instructions = _request.Program.Instructions;
            int? exit = null;
            bool LeavesTo(int target)
            {
                if (target >= header && target <= latch)
                {
                    return true;
                }

                exit ??= target;
                return exit == target;
            }

            for (var block = header; block <= latch; block++)
            {
                var opcode = instructions[blocks[block].EndIndex - 1].Opcode;
                var next = block + 1 < blocks.Count ? block + 1 : ExitTarget;
                if ((opcode.StartsWith("SCbranch", StringComparison.Ordinal) || opcode == "SBranch") && !LeavesTo(targets[block]))
                {
                    return false;
                }

                if (opcode != "SBranch" && !LeavesTo(next))
                {
                    return false;
                }
            }

            for (var block = 0; block < blocks.Count; block++)
            {
                var inside = block >= header && block <= latch;
                if (targets[block] == int.MinValue ||
                    (!inside && targets[block] > header && targets[block] <= latch))
                {
                    // An unresolved branch, or a way into the body that skips the header.
                    return false;
                }

                if (!inside)
                {
                    continue;
                }

                for (var index = blocks[block].StartIndex; index < blocks[block].EndIndex; index++)
                {
                    if (!IsHalfLocalInstruction(_request.Program.Instructions[index]))
                    {
                        return false;
                    }
                }
            }

            return true;
        }

        /// <summary>Work a half can do without its partner: lane-local vector ALU and memory
        /// access, EXEC moves and mask ops, and branches on EXEC. Scalar state other than EXEC
        /// would come out of the loop different in the half that left first, and LDS or a
        /// cross-lane read would see the other half's lanes at another iteration.</summary>
        private static bool IsHalfLocalInstruction(Gen5ShaderInstruction instruction)
        {
            var opcode = instruction.Opcode;
            if (opcode is "SBranch" or "SCbranchExecz" or "SCbranchExecnz" or "SNop" or "SInstPrefetch" or "SClause" ||
                opcode.StartsWith("SWaitcnt", StringComparison.Ordinal))
            {
                return true;
            }

            if (opcode.StartsWith('S'))
            {
                // Mask moves and ops into EXEC or VCC, which the lanes hold as their own bits
                // (Gen5MslTranslator.LaneMasks.cs); a read of the pair as a scalar after the
                // loop would put its ballot in the loop, which HalfLocalLoopsHoldNoExchange
                // refuses.
                return instruction.Destinations is [{ Kind: Gen5OperandKind.ScalarRegister, Value: ExecLoRegister or VccLoRegister }] &&
                    opcode is "SMovB64" or "SAndB64" or "SAndn2B64" or "SOrB64" or "SOrn2B64" or "SXorB64";
            }

            if (opcode.StartsWith('V'))
            {
                return instruction.Control is not (Gen5DppControl or Gen5Dpp8Control) &&
                    !opcode.StartsWith("VReadlane", StringComparison.Ordinal) &&
                    !opcode.StartsWith("VReadfirstlane", StringComparison.Ordinal) &&
                    !opcode.StartsWith("VWritelane", StringComparison.Ordinal) &&
                    !opcode.StartsWith("VPermlane", StringComparison.Ordinal) &&
                    !opcode.StartsWith("VMbcnt", StringComparison.Ordinal);
            }

            return instruction.Control is Gen5BufferMemoryControl { Lds: false } ||
                opcode.StartsWith("Global", StringComparison.Ordinal) ||
                opcode.StartsWith("Image", StringComparison.Ordinal);
        }

        /// <summary>After the mask ballots are resolved: a half-local loop must hold no
        /// exchange, since its halves leave it after their own numbers of trips.</summary>
        private bool HalfLocalLoopsHoldNoExchange()
        {
            if (_halfLocalLoopBlocks is not { } blocks)
            {
                return true;
            }

            for (var block = 0; block < blocks.Length; block++)
            {
                if (blocks[block] &&
                    (_blockText[block].Contains(WaveSetFlip, StringComparison.Ordinal) ||
                     _blockText[block].Contains(Wave64Barrier, StringComparison.Ordinal) ||
                     _blockText[block].Contains("threadgroup_barrier", StringComparison.Ordinal)))
                {
                    HalfLocalLoopExchanged = true;
                    return false;
                }
            }

            return true;
        }

        /// <summary>The structured body of a bridged multi-wave compute program, with the
        /// exchanges of group-uniform blocks at threadgroup barriers. Falls back to the pair
        /// barriers everywhere when a converted block would be emitted twice (a duplicated
        /// latch or tail could put one wave's barrier on another call site than its
        /// neighbour's); the block text is left as it was when no structured body results.</summary>
        private bool TryBuildComputeStructuredBody(
            IReadOnlyList<ShaderBlock> blocks,
            [System.Diagnostics.CodeAnalysis.NotNullWhen(true)] out StringBuilder? body)
        {
            if (!IsBridgedComputeWaves || !_options.GroupUniformBarriers ||
                FindGroupUniformBlocks(blocks) is not { } uniform)
            {
                return TryBuildStructuredBody(blocks, out body);
            }

            var original = _blockText.ToArray();
            var expected = 0;
            for (var block = 0; block < blocks.Count; block++)
            {
                if (uniform[block])
                {
                    _blockText[block] = original[block].Replace(Wave64Barrier, GroupWaveBarrier, StringComparison.Ordinal);
                    expected += CountOccurrences(_blockText[block], GroupWaveBarrier);
                }
            }

            if (expected == 0)
            {
                return TryBuildStructuredBody(blocks, out body);
            }

            if (TryBuildStructuredBody(blocks, out body) &&
                CountOccurrences(body.ToString(), GroupWaveBarrier) == expected)
            {
                return true;
            }

            for (var block = 0; block < blocks.Count; block++)
            {
                _blockText[block] = original[block];
            }

            return TryBuildStructuredBody(blocks, out body);
        }

        private static int CountOccurrences(string text, string value)
        {
            var count = 0;
            for (var at = text.IndexOf(value, StringComparison.Ordinal); at >= 0;
                 at = text.IndexOf(value, at + value.Length, StringComparison.Ordinal))
            {
                count++;
            }

            return count;
        }

        /// <summary>The blocks every wave of the group runs equally often and in the same
        /// order: those no branch on a wave-varying value controls. A branch varies when it
        /// tests EXEC or VCC, or SCC computed from a varying scalar. A scalar varies when it
        /// comes from lane data (V_READLANE, V_CMP, a carry), from EXEC, from a wave's own
        /// hardware state (TG_SIZE's wave id, S_GETREG, a scalar atomic), from a varying
        /// scalar, or from a write under a varying branch that reaches past the branch's join.
        /// Null when the program indexes its scalar registers dynamically.</summary>
        private bool[]? FindGroupUniformBlocks(IReadOnlyList<ShaderBlock> blocks)
        {
            var instructions = _request.Program.Instructions;
            foreach (var instruction in instructions)
            {
                if (instruction.Opcode.StartsWith("SMovrel", StringComparison.Ordinal) ||
                    instruction.Opcode is "SSetpcB64" or "SSwappcB64" or "SRfeB64")
                {
                    return null;
                }
            }

            var count = blocks.Count;
            _successors = new List<int>[count];
            for (var block = 0; block < count; block++)
            {
                var terminator = _blockTerminators[block];
                _successors[block] = terminator.Kind == TerminatorKind.CondJump
                    ? [terminator.TrueTarget, terminator.FalseTarget]
                    : [terminator.TrueTarget];
            }

            var ipdom = PostDominators(count);
            var predecessors = EdgeLists(count, forward: false);
            var order = ReversePostorder(count, forward: true, out _);

            // The slots each block may write.
            var writes = new bool[count][];
            for (var block = 0; block < count; block++)
            {
                writes[block] = new bool[UniformSlots];
                for (var index = blocks[block].StartIndex; index < blocks[block].EndIndex; index++)
                {
                    UniformTransfer(instructions[index], writes[block], writes[block], markWrites: true);
                }
            }

            var entry = new bool[UniformSlots];
            entry[ExecLoRegister] = true;
            entry[ExecHiRegister] = true;
            if (_request.ComputeSystemRegisters is { ThreadGroupSizeRegister: { } waveId } && waveId < 128)
            {
                entry[waveId] = true;
            }

            var taint = new bool[count][];
            var inState = new bool[count][];
            var outState = new bool[count][];
            for (var block = 0; block < count; block++)
            {
                taint[block] = new bool[UniformSlots];
                inState[block] = new bool[UniformSlots];
                outState[block] = new bool[UniformSlots];
            }

            var divergent = new bool[count];
            for (var round = 0; round < 64; round++)
            {
                // Forward: a slot varies after a block when it varies on any way in.
                for (var pass = 0; pass < count + 2; pass++)
                {
                    var changed = false;
                    foreach (var block in order)
                    {
                        var state = block == 0 ? (bool[])entry.Clone() : new bool[UniformSlots];
                        foreach (var predecessor in predecessors[block])
                        {
                            Or(state, outState[predecessor]);
                        }

                        Or(state, taint[block]);
                        inState[block] = state;
                        var result = (bool[])state.Clone();
                        for (var index = blocks[block].StartIndex; index < blocks[block].EndIndex; index++)
                        {
                            UniformTransfer(instructions[index], result, result, markWrites: false);
                        }

                        result[ExecLoRegister] = true;
                        result[ExecHiRegister] = true;
                        if (!result.AsSpan().SequenceEqual(outState[block]))
                        {
                            outState[block] = result;
                            changed = true;
                        }
                    }

                    if (!changed)
                    {
                        break;
                    }
                }

                // Backward: the blocks a varying branch controls, and what they write reaching
                // past its join.
                var grew = false;
                for (var block = 0; block < count; block++)
                {
                    if (_blockTerminators[block].Kind != TerminatorKind.CondJump ||
                        !BranchVaries(instructions[blocks[block].EndIndex - 1], outState[block]))
                    {
                        continue;
                    }

                    var join = ipdom[block];
                    var region = new bool[count];
                    var pending = new Stack<int>();
                    foreach (var successor in _successors[block])
                    {
                        if (successor != ExitTarget && successor != join)
                        {
                            pending.Push(successor);
                        }
                    }

                    var regionWrites = new bool[UniformSlots];
                    while (pending.Count != 0)
                    {
                        var node = pending.Pop();
                        if (region[node])
                        {
                            continue;
                        }

                        region[node] = true;
                        Or(regionWrites, writes[node]);
                        if (!divergent[node])
                        {
                            divergent[node] = true;
                            grew = true;
                        }

                        foreach (var successor in _successors[node])
                        {
                            if (successor != ExitTarget && successor != join && !region[successor])
                            {
                                pending.Push(successor);
                            }
                        }
                    }

                    if (join >= 0 && join < count)
                    {
                        for (var slot = 0; slot < UniformSlots; slot++)
                        {
                            if (regionWrites[slot] && !taint[join][slot])
                            {
                                taint[join][slot] = true;
                                grew = true;
                            }
                        }
                    }
                }

                if (!grew)
                {
                    var reached = new bool[count];
                    foreach (var block in order)
                    {
                        reached[block] = true;
                    }

                    var uniform = new bool[count];
                    for (var block = 0; block < count; block++)
                    {
                        uniform[block] = reached[block] && !divergent[block];
                    }

                    return uniform;
                }
            }

            return null;
        }

        /// <summary>The vector ops that can write VCC without naming it: VOPC compares and the
        /// VOP2 carries (and, to be safe, their VOP3B forms and V_DIV_SCALE).</summary>
        private static bool WritesVccImplicitly(string opcode) =>
            opcode.StartsWith("VCmp", StringComparison.Ordinal) ||
            opcode.StartsWith("VDivScale", StringComparison.Ordinal) ||
            opcode is "VAddCoU32" or "VSubCoU32" or "VSubrevCoU32" or "VAddcU32" or "VSubbU32" or "VSubbrevU32" or
                "VAddCoCiU32" or "VSubCoCiU32" or "VSubrevCoCiU32" or "VMadU64U32" or "VMadI64I32";

        private static void Or(bool[] target, bool[] source)
        {
            for (var slot = 0; slot < target.Length; slot++)
            {
                target[slot] |= source[slot];
            }
        }

        /// <summary>Whether the branch ending a block can go different ways in different
        /// waves: EXEC and VCC are the waves' own, SCC varies with what it was computed from.</summary>
        private static bool BranchVaries(Gen5ShaderInstruction branch, bool[] state) => branch.Opcode switch
        {
            "SCbranchScc0" or "SCbranchScc1" => state[SccSlotIndex],
            "SCbranchCdbgsys" or "SCbranchCdbguser" or "SCbranchCdbgsysOrUser" or "SCbranchCdbgsysAndUser" => false,
            _ => true,
        };

        /// <summary>Applies one instruction to the varying set: reads <paramref name="state"/>
        /// and writes <paramref name="result"/> (the same array for the forward walk). With
        /// <paramref name="markWrites"/>, every slot the instruction may write is set.</summary>
        private static void UniformTransfer(Gen5ShaderInstruction instruction, bool[] state, bool[] result, bool markWrites)
        {
            var opcode = instruction.Opcode;
            var wide = opcode.Contains("64", StringComparison.Ordinal);

            void Write(uint register, bool varies, bool pair)
            {
                if (register < 128)
                {
                    result[register] = markWrites || varies;
                    if (pair && register + 1 < 128)
                    {
                        result[register + 1] = markWrites || varies;
                    }
                }
            }

            bool Reads(Gen5Operand operand, bool pair) =>
                operand.Kind == Gen5OperandKind.ScalarRegister && operand.Value < 128 &&
                (state[operand.Value] || (pair && operand.Value + 1 < 128 && state[operand.Value + 1]));

            if (!opcode.StartsWith('S'))
            {
                // Lane data reaching a scalar (V_READLANE, V_CMP's mask, a carry out), and the
                // VCC a compare or a carry writes without naming it.
                var namesScalar = false;
                foreach (var destination in instruction.Destinations)
                {
                    if (destination.Kind == Gen5OperandKind.ScalarRegister)
                    {
                        Write(destination.Value, varies: true, pair: true);
                        namesScalar = true;
                    }
                }

                if (!namesScalar && WritesVccImplicitly(opcode))
                {
                    Write(VccLoRegister, varies: true, pair: true);
                }

                return;
            }

            if (opcode.StartsWith("SLoad", StringComparison.Ordinal) ||
                opcode.StartsWith("SBufferLoad", StringComparison.Ordinal) ||
                opcode.StartsWith("SScratchLoad", StringComparison.Ordinal))
            {
                // A scalar load from a uniform address reads the same words in every wave.
                var addressVaries = false;
                for (var index = 0; index < instruction.Sources.Count; index++)
                {
                    addressVaries |= Reads(instruction.Sources[index], pair: index == 0);
                }

                foreach (var destination in instruction.Destinations)
                {
                    Write(destination.Value, addressVaries, pair: false);
                }

                return;
            }

            var waveState = opcode.StartsWith("SAtomic", StringComparison.Ordinal) ||
                opcode.StartsWith("SBufferAtomic", StringComparison.Ordinal) ||
                opcode.StartsWith("SGetreg", StringComparison.Ordinal) ||
                opcode is "SMemtime" or "SMemrealtime" or "SSendmsgRtnB32" or "SSendmsgRtnB64" ||
                opcode.Contains("Saveexec", StringComparison.Ordinal);
            var readsScc = opcode.StartsWith("SCselect", StringComparison.Ordinal) ||
                opcode.StartsWith("SCmov", StringComparison.Ordinal) ||
                opcode.StartsWith("SAddc", StringComparison.Ordinal) ||
                opcode.StartsWith("SSubb", StringComparison.Ordinal);
            var varies = waveState || (readsScc && state[SccSlotIndex]);
            foreach (var source in instruction.Sources)
            {
                varies |= Reads(source, wide);
            }

            foreach (var destination in instruction.Destinations)
            {
                if (destination.Kind == Gen5OperandKind.ScalarRegister)
                {
                    Write(destination.Value, varies, wide);
                }
            }

            // A compare sets SCC from its operands alone; any other scalar op may set SCC from
            // them or leave it, so the result varies when either does.
            result[SccSlotIndex] = markWrites ||
                (opcode.StartsWith("SCmp", StringComparison.Ordinal) || opcode.StartsWith("SBitcmp", StringComparison.Ordinal)
                    ? varies
                    : state[SccSlotIndex] || varies);
        }
    }
}
