// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.ShaderCompiler.Metal;

// EXPERIMENT (SHARPEMU_MSL_MULTIWAVE64_BY_NEED=1): which wave64 compute programs of several
// waves need each wave whole (IsBridgedComputeWaves).
//
// Bridged, a wave's two simdgroups meet at a pair barrier - a polled counter - for every
// exchange: each READFIRSTLANE, each EXEC or VCC a branch tests. Run as two wave32s, they
// exchange nothing. Every multi-wave group is bridged by default for GTA V's BVH-build radix
// sort, which needs its whole waves, and that cost Ghost of Tsushima 9% of its frame rate:
// its draw-list compute E0A3AEB0626C81E7 (four waves a group, 57 pair barriers) went from
// 4.7 to 12.7 ms a frame, a program whose lanes never look at each other.
//
// A program can tell its wave was run as two halves only through something that sees more
// than its own lane:
//  - a cross-lane read or write (V_READLANE/WRITELANE, PERMLANE64, MBCNT, DPP, a DS swizzle or
//    permute, DS_APPEND/CONSUME, GWS); PERMLANE16 and PERMLANEX16 stay within a half;
//  - LDS read where a store of the same wave may reach it with no S_BARRIER between (within
//    one wave the hardware needs none), or an LDS atomic that returns (its lanes get their
//    values in lane order). A group of several waves has to put a barrier between one wave's
//    stores and another's loads anyway; Ghost of Tsushima's 512-thread D54E1E3E does, and
//    its 64-thread twin 3E05938E, one wave, does not.
//  - a scalar op on a wave's own state (S_GETREG, S_SETREG, scalar stores and atomics,
//    relative scalar addressing, a message);
//  - a lane mask (EXEC, or what a V_CMP or a carry wrote) read as data: by anything but the
//    mask ops, the saves of EXEC, a branch, V_CNDMASK, a carry-in or a compare with zero (a
//    bit count or scan of one, say); or EXEC set to lanes picked by number (S_BFM_B64 32, 32).
//    A forward pass over the scalar registers and SCC follows which may hold a mask.
// Without any of these each half computes, lane for lane, what the whole wave would:
// READFIRSTLANE of a wave-uniform value is the same value in either half, and a waterfall
// loop over a non-uniform one still visits every value. Such a program runs as wave32s.
public static partial class Gen5MslTranslator
{
    private sealed partial class CompilationContext
    {
        private const int WholeWaveSccSlot = 128;
        private const int WholeWaveSlots = 129;

        private bool? _needsWholeWaves;
        private string _wholeWaveReason = string.Empty;

        private static readonly bool WholeWaveTrace =
            Environment.GetEnvironmentVariable("SHARPEMU_MSL_MULTIWAVE64_TRACE") == "1";

        /// <summary>Whether the program's results could tell its wave64 ran as two wave32s.</summary>
        private bool NeedsWholeWaves
        {
            get
            {
                if (_needsWholeWaves is { } known)
                {
                    return known;
                }

                var needs = FindWholeWaveUse(out _wholeWaveReason);
                _needsWholeWaves = needs;
                if (WholeWaveTrace)
                {
                    Console.Error.WriteLine(
                        $"[MSL][MULTIWAVE64] hash=0x{_request.Hash:X16} threads={ComputeThreads} " +
                        $"{(needs ? "bridged: " + _wholeWaveReason : "wave32")}");
                }

                return needs;
            }
        }

        private bool FindWholeWaveUse(out string reason)
        {
            var instructions = _request.Program.Instructions;
            if (ProgramUsesLds && LdsSeenWithinAWave(instructions, out reason))
            {
                return true;
            }

            foreach (var instruction in instructions)
            {
                if (SeesOtherLanes(instruction))
                {
                    reason = "op " + instruction.Opcode;
                    return true;
                }
            }

            return MaskReadAsData(instructions, out reason);
        }

        // Whether an LDS load may see a store of its own wave with no S_BARRIER between: a
        // forward pass with one bit, "a store since the last barrier".
        private bool LdsSeenWithinAWave(IReadOnlyList<Gen5ShaderInstruction> instructions, out string reason)
        {
            reason = string.Empty;
            if (_request.Memory.Entries.Any(static memory =>
                    memory.AddressSpace is FlatAddressSpace.Shared or FlatAddressSpace.SharedOrPrivate))
            {
                reason = "flat lds";
                return true;
            }

            var count = instructions.Count;
            var byPc = new Dictionary<uint, int>(count);
            for (var index = 0; index < count; index++)
            {
                byPc[instructions[index].Pc] = index;
            }

            var reached = new bool[count];
            var dirty = new bool[count];
            var work = new Queue<int>();
            reached[0] = true;
            work.Enqueue(0);
            Span<int> next = stackalloc int[2];
            while (work.Count != 0)
            {
                var index = work.Dequeue();
                var instruction = instructions[index];
                var opcode = instruction.Opcode;
                var state = dirty[index];
                var write = (opcode.StartsWith("Ds", StringComparison.Ordinal) &&
                        !opcode.StartsWith("DsRead", StringComparison.Ordinal) && opcode != "DsNop") ||
                    instruction.Control is Gen5BufferMemoryControl { Lds: true };
                var read = opcode.StartsWith("DsRead", StringComparison.Ordinal) ||
                    (opcode.StartsWith("Ds", StringComparison.Ordinal) && opcode.Contains("Rtn", StringComparison.Ordinal));
                if (opcode.StartsWith("Ds", StringComparison.Ordinal) && opcode.Contains("Rtn", StringComparison.Ordinal))
                {
                    reason = $"lds atomic {opcode}";
                    return true;
                }

                if (read && state)
                {
                    reason = $"lds read {opcode} at 0x{instruction.Pc:X4} after an unbarriered store";
                    return true;
                }

                state = opcode == "SBarrier" ? false : state || write;
                var nextCount = Successors(instructions, byPc, index, next);
                if (nextCount < 0)
                {
                    reason = "unresolved branch";
                    return true;
                }

                foreach (var successor in next[..nextCount])
                {
                    if (!reached[successor] || (state && !dirty[successor]))
                    {
                        reached[successor] = true;
                        dirty[successor] |= state;
                        work.Enqueue(successor);
                    }
                }
            }

            return false;
        }

        // The instructions control can go to after this one; -1 for a branch that cannot be resolved.
        private static int Successors(
            IReadOnlyList<Gen5ShaderInstruction> instructions, Dictionary<uint, int> byPc, int index, Span<int> next)
        {
            var instruction = instructions[index];
            var opcode = instruction.Opcode;
            var count = 0;
            if (opcode is not ("SBranch" or "SEndpgm" or "SSetpcB64") && index + 1 < instructions.Count)
            {
                next[count++] = index + 1;
            }

            if ((opcode == "SBranch" || opcode.StartsWith("SCbranch", StringComparison.Ordinal)) &&
                TryGetBranchTargetPc(instruction, out var targetPc))
            {
                if (byPc.TryGetValue(targetPc, out var target))
                {
                    next[count++] = target;
                }
                else if (!IsExitBranchTarget(instructions, instruction))
                {
                    return -1;
                }
            }

            return count;
        }

        private static bool SeesOtherLanes(Gen5ShaderInstruction instruction)
        {
            var opcode = instruction.Opcode;
            if (instruction.Control is Gen5DppControl or Gen5Dpp8Control)
            {
                return true;
            }

            return opcode.StartsWith("VReadlane", StringComparison.Ordinal) ||
                opcode.StartsWith("VWritelane", StringComparison.Ordinal) ||
                // V_PERMLANE16 and V_PERMLANEX16 move values within each half's 32 lanes.
                (opcode.StartsWith("VPermlane", StringComparison.Ordinal) &&
                    !opcode.StartsWith("VPermlane16", StringComparison.Ordinal) &&
                    !opcode.StartsWith("VPermlanex16", StringComparison.Ordinal)) ||
                opcode.StartsWith("VMbcnt", StringComparison.Ordinal) ||
                opcode.StartsWith("DsSwizzle", StringComparison.Ordinal) ||
                opcode.StartsWith("DsBpermute", StringComparison.Ordinal) ||
                opcode.StartsWith("DsPermute", StringComparison.Ordinal) ||
                opcode.StartsWith("DsAppend", StringComparison.Ordinal) ||
                opcode.StartsWith("DsConsume", StringComparison.Ordinal) ||
                opcode.StartsWith("DsGws", StringComparison.Ordinal) ||
                opcode.StartsWith("DsOrdered", StringComparison.Ordinal) ||
                opcode.StartsWith("SGetreg", StringComparison.Ordinal) ||
                opcode.StartsWith("SSetreg", StringComparison.Ordinal) ||
                opcode.StartsWith("SMovrel", StringComparison.Ordinal) ||
                opcode.StartsWith("SAtomic", StringComparison.Ordinal) ||
                opcode.StartsWith("SStore", StringComparison.Ordinal) ||
                opcode.StartsWith("SBufferStore", StringComparison.Ordinal) ||
                opcode.StartsWith("SBufferAtomic", StringComparison.Ordinal) ||
                opcode.StartsWith("SSendmsg", StringComparison.Ordinal) ||
                opcode is "SSetpcB64" or "SSwappcB64" or "SGetpcB64" or "SRfeB64";
        }

        // Ops whose result is a mask again when a source is one: kept in the lane form.
        private static bool IsMaskOp(string opcode) =>
            opcode is "SMovB64" or "SAndB64" or "SOrB64" or "SXorB64" or "SAndn2B64" or "SOrn2B64" or
                "SNandB64" or "SNorB64" or "SXnorB64" or "SNotB64" or "SCselectB64" or "SMovB32" or
                "SCselectB32" or "SAndB32" or "SOrB32" or "SXorB32" or "SAndn2B32" or "SOrn2B32" ||
            // Whole quad mode widens a mask quad by quad, and no quad crosses the halves.
            opcode is "SWqmB64" or "SWqmB32" ||
            opcode.Contains("Saveexec", StringComparison.Ordinal) ||
            opcode.Contains("Wrexec", StringComparison.Ordinal);

        // Ops that read a mask as lane bits.
        private static bool ReadsMaskAsLanes(string opcode) =>
            opcode.StartsWith("VCndmask", StringComparison.Ordinal) ||
            opcode.StartsWith("VAddc", StringComparison.Ordinal) ||
            opcode.StartsWith("VSubb", StringComparison.Ordinal) ||
            opcode.StartsWith("VDivFmas", StringComparison.Ordinal) ||
            opcode.StartsWith("SCbranch", StringComparison.Ordinal);

        // Ops that write a lane mask (a compare or a carry), into their destination or VCC.
        private static bool WritesLaneMask(string opcode) =>
            opcode.StartsWith("VCmp", StringComparison.Ordinal) ||
            (opcode.StartsWith('V') && (opcode.Contains("CoU32", StringComparison.Ordinal) ||
                opcode.Contains("CoCi", StringComparison.Ordinal))) ||
            opcode.StartsWith("VAddc", StringComparison.Ordinal) ||
            opcode.StartsWith("VSubb", StringComparison.Ordinal) ||
            opcode.StartsWith("VMadU64", StringComparison.Ordinal) ||
            opcode.StartsWith("VMadI64", StringComparison.Ordinal) ||
            opcode.StartsWith("VDivScale", StringComparison.Ordinal);

        private static bool Is64BitScalarOp(Gen5ShaderInstruction instruction) =>
            instruction.Opcode.StartsWith('S') && instruction.Opcode.Contains("64", StringComparison.Ordinal);

        private bool MaskReadAsData(IReadOnlyList<Gen5ShaderInstruction> instructions, out string reason)
        {
            reason = string.Empty;
            var count = instructions.Count;
            var byPc = new Dictionary<uint, int>(count);
            for (var index = 0; index < count; index++)
            {
                byPc[instructions[index].Pc] = index;
            }

            var input = new bool[count][];
            var queued = new bool[count];
            var work = new Queue<int>();
            input[0] = new bool[WholeWaveSlots];
            input[0][ExecLoRegister] = true;
            input[0][ExecHiRegister] = true;
            work.Enqueue(0);
            queued[0] = true;
            var state = new bool[WholeWaveSlots];
            Span<int> next = stackalloc int[2];
            while (work.Count != 0)
            {
                var index = work.Dequeue();
                queued[index] = false;
                Array.Copy(input[index], state, WholeWaveSlots);
                if (StepMask(instructions[index], state, out var dataRead))
                {
                    reason = $"mask read as data by {instructions[index].Opcode} at 0x{instructions[index].Pc:X4} ({dataRead})";
                    return true;
                }

                var nextCount = Successors(instructions, byPc, index, next);
                if (nextCount < 0)
                {
                    reason = "unresolved branch";
                    return true;
                }

                foreach (var successor in next[..nextCount])
                {
                    var changed = false;
                    if (input[successor] is not { } into)
                    {
                        input[successor] = into = new bool[WholeWaveSlots];
                        changed = true;
                    }

                    for (var slot = 0; slot < WholeWaveSlots; slot++)
                    {
                        if (state[slot] && !into[slot])
                        {
                            into[slot] = true;
                            changed = true;
                        }
                    }

                    if (changed && !queued[successor])
                    {
                        queued[successor] = true;
                        work.Enqueue(successor);
                    }
                }
            }

            return false;
        }

        // One instruction of the may-hold-a-mask pass: true when it reads a mask as data.
        private static bool StepMask(Gen5ShaderInstruction instruction, bool[] state, out string read)
        {
            read = string.Empty;
            var opcode = instruction.Opcode;
            var scalarOp = opcode.StartsWith('S');
            var wide = Is64BitScalarOp(instruction) ||
                opcode.Contains("F64", StringComparison.Ordinal) || opcode.Contains("U64", StringComparison.Ordinal) ||
                opcode.Contains("I64", StringComparison.Ordinal) || opcode.Contains("B64", StringComparison.Ordinal);
            // A compare of a mask with zero asks whether any lane is set, as a branch on it does.
            var anyTest = opcode is "SCmpEqU64" or "SCmpLgU64" &&
                instruction.Sources.Any(static source => source.Kind == Gen5OperandKind.EncodedConstant && source.Value == 128);
            var maskSource = false;
            foreach (var source in instruction.Sources)
            {
                if (source.Kind != Gen5OperandKind.ScalarRegister || source.Value >= 128)
                {
                    continue;
                }

                var held = state[source.Value] || (wide && source.Value + 1 < 128 && state[source.Value + 1]);
                if (!held)
                {
                    continue;
                }

                maskSource = true;
                if (!IsMaskOp(opcode) && !ReadsMaskAsLanes(opcode) && !anyTest)
                {
                    read = $"s{source.Value}";
                    return true;
                }
            }

            if (scalarOp && !maskSource && instruction.Destinations is [{ Kind: Gen5OperandKind.ScalarRegister, Value: ExecLoRegister }] &&
                !instruction.Sources.All(static source =>
                    (source.Kind == Gen5OperandKind.EncodedConstant && source.Value is 193 or 128) ||
                    (source.Kind == Gen5OperandKind.LiteralConstant && source.Value == uint.MaxValue)))
            {
                // EXEC set to lanes picked by number rather than by a mask.
                read = "exec set by number";
                return true;
            }

            var readsScc = opcode.StartsWith("SCselect", StringComparison.Ordinal) ||
                opcode.StartsWith("SAddc", StringComparison.Ordinal) || opcode.StartsWith("SSubb", StringComparison.Ordinal) ||
                opcode.StartsWith("SCmovk", StringComparison.Ordinal) || opcode.StartsWith("SCmov", StringComparison.Ordinal);
            if (readsScc && state[WholeWaveSccSlot])
            {
                if (!IsMaskOp(opcode))
                {
                    read = "scc";
                    return true;
                }

                maskSource = true;
            }

            // What the instruction writes now holds a mask or not.
            var producesMask = WritesLaneMask(opcode) || opcode.Contains("Saveexec", StringComparison.Ordinal) ||
                (IsMaskOp(opcode) && maskSource);
            var wroteDestination = false;
            foreach (var destination in instruction.Destinations)
            {
                if (destination.Kind != Gen5OperandKind.ScalarRegister || destination.Value >= 128)
                {
                    continue;
                }

                wroteDestination = true;
                state[destination.Value] = producesMask;
                if ((instruction.DestinationWidth == 2 || WritesLaneMask(opcode)) && destination.Value + 1 < 128)
                {
                    state[destination.Value + 1] = producesMask;
                }
            }

            if (WritesLaneMask(opcode) && !wroteDestination)
            {
                // The implicit destination: EXEC for V_CMPX, VCC for the rest.
                var implicitLo = opcode.StartsWith("VCmpx", StringComparison.Ordinal) ? ExecLoRegister : VccLoRegister;
                state[implicitLo] = true;
                state[implicitLo + 1] = true;
            }
            else if (opcode.StartsWith("VCmpx", StringComparison.Ordinal))
            {
                state[ExecLoRegister] = true;
                state[ExecHiRegister] = true;
            }

            // EXEC is a mask whatever was written into it.
            state[ExecLoRegister] = true;
            state[ExecHiRegister] = true;
            if (anyTest)
            {
                state[WholeWaveSccSlot] = true;
                return false;
            }

            if (scalarOp && !opcode.StartsWith("SCbranch", StringComparison.Ordinal) && opcode is not ("SBranch" or "SNop" or "SEndpgm") &&
                !opcode.StartsWith("SWaitcnt", StringComparison.Ordinal) && !opcode.StartsWith("SLoad", StringComparison.Ordinal) &&
                !opcode.StartsWith("SBufferLoad", StringComparison.Ordinal))
            {
                state[WholeWaveSccSlot] = IsMaskOp(opcode) && maskSource;
            }

            return false;
        }
    }
}
