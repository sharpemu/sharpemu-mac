// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Ir;

// The instructions that run with every lane of EXEC set. EXEC starts full; it stays full
// through S_WQM of a full EXEC and comes back when a saved copy of a full EXEC is moved or
// ORed back into it. Any other EXEC write (V_CMPX, S_AND_SAVEEXEC, ...) makes it unknown,
// and a join is full only when every path into it is. A backend can skip the EXEC guard of
// a vector register write there: no lane keeps its old value, so the old value need not
// stay live until the write.
//
// Under SGPR pressure the guest compiler parks the saved EXEC of an if in lanes of a VGPR
// (V_WRITELANE_B32 v27, s0, 34 / s1, 35) and brings it back with V_READLANE_B32 just before
// S_MOV_B64 exec, s[0:1] closes the if. With trackLaneSpills the analysis follows a full
// copy into such a lane and back: a constant-lane writelane of a full copy marks the lane,
// a readlane of a marked lane makes its scalar destination a full copy again, and any other
// write that may reach the VGPR (any lane, a relative move) unmarks it. Without it EXEC
// stays unknown from the first spilled if to the end of the program; a Demon's Souls
// lighting pixel shader runs 1800 of its 2000 vector instructions that way.
public static class Gen5ExecFullAnalysis
{
    private const uint ExecLow = 126;
    private const uint ExecHigh = 127;

    private readonly struct State(bool reached, bool execFull, UInt128 fullCopies, int[]? laneCopies = null)
    {
        public readonly bool Reached = reached;
        public readonly bool ExecFull = execFull;
        // SGPRs 0..127 holding (part of) a copy of a full EXEC.
        public readonly UInt128 FullCopies = fullCopies;
        // VGPR lanes (vector * 64 + lane, ascending) holding such a copy, under trackLaneSpills.
        public readonly int[] LaneCopies = laneCopies ?? [];

        public static State Meet(State left, State right) =>
            !left.Reached ? right
            : !right.Reached ? left
            : new State(true, left.ExecFull && right.ExecFull, left.FullCopies & right.FullCopies,
                left.LaneCopies.Length == 0 || right.LaneCopies.Length == 0 ? null
                : left.LaneCopies.Intersect(right.LaneCopies).ToArray());

        public bool SameAs(State other) =>
            Reached == other.Reached && ExecFull == other.ExecFull && FullCopies == other.FullCopies &&
            LaneCopies.AsSpan().SequenceEqual(other.LaneCopies);
    }

    // Pcs of the instructions that start with a full EXEC.
    public static IReadOnlySet<uint> Analyze(Gen5ShaderProgram program, bool wave32) =>
        Analyze(program, wave32, trackLaneSpills: false);

    // The same, following full copies through VGPR lane spills when trackLaneSpills is set.
    public static IReadOnlySet<uint> Analyze(Gen5ShaderProgram program, bool wave32, bool trackLaneSpills) =>
        Analyze(program, wave32, trackLaneSpills, trackWholeWave: false);

    // The same, and with trackWholeWave also inside a whole-wave-mode section: the guest
    // compiler enters one with S_OR_SAVEEXEC exec, -1 or S_ORN2_SAVEEXEC exec, exec (both set
    // every lane, whatever EXEC was) to run a reduction over the whole wave, and leaves it by
    // moving the saved mask back into EXEC.
    public static IReadOnlySet<uint> Analyze(Gen5ShaderProgram program, bool wave32, bool trackLaneSpills, bool trackWholeWave)
    {
        var instructions = program.Instructions;
        var result = new HashSet<uint>();
        if (instructions.Count == 0)
        {
            return result;
        }

        var indexByPc = new Dictionary<uint, int>(instructions.Count);
        for (var index = 0; index < instructions.Count; index++)
        {
            if (instructions[index].Opcode is "SSetpcB64" or "SSwappcB64" or "SRfeB64")
            {
                return result;
            }

            indexByPc[instructions[index].Pc] = index;
        }

        var states = new State[instructions.Count];
        states[0] = new State(true, true, 0);
        var pending = new Queue<int>();
        var queued = new bool[instructions.Count];
        pending.Enqueue(0);
        queued[0] = true;
        while (pending.TryDequeue(out var index))
        {
            queued[index] = false;
            var instruction = instructions[index];
            var output = Transfer(instruction, states[index], wave32, trackLaneSpills, trackWholeWave);
            void Flow(int successor)
            {
                var merged = State.Meet(states[successor], output);
                if (!merged.SameAs(states[successor]))
                {
                    states[successor] = merged;
                    if (!queued[successor])
                    {
                        queued[successor] = true;
                        pending.Enqueue(successor);
                    }
                }
            }

            if (Gen5IrBranchResolver.IsTerminator(instruction))
            {
                continue;
            }

            if (Gen5IrBranchResolver.Instance.TryGetBranchTarget(instruction, out var targetPc))
            {
                if (!indexByPc.TryGetValue(targetPc, out var target))
                {
                    // A branch out of the decoded range: give up on the whole program.
                    return new HashSet<uint>();
                }

                Flow(target);
                if (Gen5IrBranchResolver.IsUnconditionalBranch(instruction))
                {
                    continue;
                }
            }

            if (index + 1 < instructions.Count)
            {
                Flow(index + 1);
            }
        }

        for (var index = 0; index < instructions.Count; index++)
        {
            if (states[index].Reached && states[index].ExecFull)
            {
                result.Add(instructions[index].Pc);
            }
        }

        return result;
    }

    private static State Transfer(Gen5ShaderInstruction instruction, State input, bool wave32, bool trackLaneSpills, bool trackWholeWave)
    {
        if (!input.Reached)
        {
            return input;
        }

        var execFull = input.ExecFull;
        var copies = input.FullCopies;
        var opcode = instruction.Opcode;
        var sources = instruction.Sources;
        var destinations = instruction.Destinations;
        var wide = opcode.StartsWith('S') &&
            (opcode.EndsWith("B64", StringComparison.Ordinal) ||
             opcode.EndsWith("U64", StringComparison.Ordinal) ||
             opcode.EndsWith("I64", StringComparison.Ordinal));

        // Registers written, with EXEC treated separately.
        UInt128 written = 0;
        var writesExec = false;
        foreach (var destination in destinations)
        {
            if (destination.Kind != Gen5OperandKind.ScalarRegister)
            {
                continue;
            }

            written |= Bit(destination.Value);
            if (wide)
            {
                written |= Bit(destination.Value + 1);
            }
        }

        // GFX10 V_CMPX changes EXEC and ignores the legacy scalar destination.
        uint? scalarDestination = opcode.StartsWith("VCmpx", StringComparison.Ordinal) ? null : instruction.Control switch
        {
            Gen5Vop3Control control => control.ScalarDestination,
            Gen5SdwaControl control => control.ScalarDestination,
            _ => null,
        };
        if (scalarDestination is { } sdst)
        {
            written |= Bit(sdst) | Bit(sdst + 1);
        }

        if (opcode.StartsWith("VCmpx", StringComparison.Ordinal))
        {
            written |= Bit(ExecLow) | Bit(ExecHigh);
        }
        else if (opcode.StartsWith("VCmp", StringComparison.Ordinal) && scalarDestination is null)
        {
            // VOPC compares write VCC.
            written |= Bit(106) | Bit(107);
        }

        if (opcode.StartsWith('V') && opcode.Contains("Co", StringComparison.Ordinal) && scalarDestination is null)
        {
            // VOP2 carry forms write VCC without listing it.
            written |= Bit(106) | Bit(107);
        }

        if (opcode.Contains("Saveexec", StringComparison.Ordinal) || opcode.Contains("Wrexec", StringComparison.Ordinal) ||
            opcode.StartsWith("SMovrel", StringComparison.Ordinal))
        {
            // These also write EXEC, or registers chosen at run time.
            written |= Bit(ExecLow) | Bit(ExecHigh);
            if (opcode.StartsWith("SMovrel", StringComparison.Ordinal))
            {
                return new State(true, false, 0);
            }
        }

        writesExec = (written & (Bit(ExecLow) | Bit(ExecHigh))) != 0;
        var nextCopies = copies & ~written;
        var nextExecFull = execFull;
        if (writesExec)
        {
            nextExecFull = false;
            var onlyExecDestination = destinations.Count == 1 && destinations[0] is { Kind: Gen5OperandKind.ScalarRegister, Value: ExecLow };
            // In wave64 a 32-bit write leaves EXEC_HI as it was.
            if (onlyExecDestination && scalarDestination is null && (wide || wave32))
            {
                switch (opcode)
                {
                    case "SWqmB64" or "SWqmB32" when sources.Count == 1 && IsExec(sources[0]):
                        nextExecFull = execFull;
                        break;
                    case "SMovB64" or "SMovB32" when sources.Count == 1:
                        nextExecFull = IsFullCopy(sources[0], copies, execFull, wide, wave32);
                        break;
                    case "SOrB64" or "SOrB32" when sources.Count == 2:
                        nextExecFull = IsFullCopy(sources[0], copies, execFull, wide, wave32) ||
                            IsFullCopy(sources[1], copies, execFull, wide, wave32);
                        break;
                }
            }
            else if (trackWholeWave && EntersWholeWave(instruction, wave32))
            {
                nextExecFull = true;
            }
        }
        else if (execFull && opcode is "SMovB64" or "SMovB32" && destinations.Count == 1 &&
                 destinations[0].Kind == Gen5OperandKind.ScalarRegister && sources.Count == 1 && IsExec(sources[0]))
        {
            // A saved copy of the full EXEC.
            nextCopies |= Bit(destinations[0].Value);
            if (wide)
            {
                nextCopies |= Bit(destinations[0].Value + 1);
            }
        }

        if (opcode.Contains("Saveexec", StringComparison.Ordinal) && execFull && destinations.Count >= 1 &&
            destinations[0].Kind == Gen5OperandKind.ScalarRegister && destinations[0].Value < ExecLow)
        {
            // The saved destination receives the (full) EXEC from before the instruction.
            nextCopies |= Bit(destinations[0].Value);
            if (wide)
            {
                nextCopies |= Bit(destinations[0].Value + 1);
            }
        }

        var laneCopies = input.LaneCopies;
        if (trackLaneSpills)
        {
            (nextCopies, laneCopies) = TransferLaneCopies(instruction, execFull, copies, nextCopies, laneCopies);
        }

        return new State(true, nextExecFull, nextCopies, laneCopies);
    }

    // A full copy parked in a VGPR lane and read back (see the class comment). execFull and
    // copies are the state before the instruction, nextCopies after its scalar writes.
    private static (UInt128 Copies, int[] LaneCopies) TransferLaneCopies(
        Gen5ShaderInstruction instruction, bool execFull, UInt128 copies, UInt128 nextCopies, int[] laneCopies)
    {
        var opcode = instruction.Opcode;
        if (opcode == "VReadlaneB32")
        {
            if (laneCopies.Length != 0 &&
                Gen5LaneSpillAnalysis.TryGetSpillSlot(instruction, out var vector, out var lane) &&
                instruction.Destinations is [{ Kind: Gen5OperandKind.ScalarRegister, Value: < ExecLow } destination, ..] &&
                Array.BinarySearch(laneCopies, (int)(vector * 64 + lane)) >= 0)
            {
                nextCopies |= Bit(destination.Value);
            }

            return (nextCopies, laneCopies);
        }

        if (opcode == "VWritelaneB32")
        {
            if (Gen5LaneSpillAnalysis.TryGetSpillSlot(instruction, out var vector, out var lane))
            {
                var key = (int)(vector * 64 + lane);
                var full = instruction.Sources[0] is { Kind: Gen5OperandKind.ScalarRegister } source &&
                    (source.Value is ExecLow or ExecHigh ? execFull : (copies & Bit(source.Value)) != 0);
                return (nextCopies, full ? With(laneCopies, key) : Without(laneCopies, key, key + 1));
            }

            return (nextCopies, instruction.Destinations is [{ Kind: Gen5OperandKind.VectorRegister } dynamic, ..]
                ? Without(laneCopies, (int)dynamic.Value * 64, (int)dynamic.Value * 64 + 64)
                : laneCopies);
        }

        if (laneCopies.Length == 0)
        {
            return (nextCopies, laneCopies);
        }

        if (opcode.StartsWith("VMovreld", StringComparison.Ordinal) ||
            opcode.StartsWith("VMovrelsd", StringComparison.Ordinal) ||
            opcode.StartsWith("VSwaprel", StringComparison.Ordinal))
        {
            // The destination is relative to M0: any VGPR.
            return (nextCopies, []);
        }

        var width = (int)Gen5LaneSpillAnalysis.VectorWriteWidth(instruction);
        foreach (var destination in instruction.Destinations)
        {
            if (destination.Kind == Gen5OperandKind.VectorRegister)
            {
                laneCopies = Without(laneCopies, (int)destination.Value * 64, ((int)destination.Value + width) * 64);
            }
        }

        if (opcode == "VSwapB32" && instruction.Sources is [{ Kind: Gen5OperandKind.VectorRegister } swapped, ..])
        {
            laneCopies = Without(laneCopies, (int)swapped.Value * 64, (int)swapped.Value * 64 + 64);
        }

        return (nextCopies, laneCopies);
    }

    private static int[] With(int[] keys, int key)
    {
        var at = Array.BinarySearch(keys, key);
        if (at >= 0)
        {
            return keys;
        }

        at = ~at;
        var result = new int[keys.Length + 1];
        keys.AsSpan(0, at).CopyTo(result);
        result[at] = key;
        keys.AsSpan(at).CopyTo(result.AsSpan(at + 1));
        return result;
    }

    // keys without those in [first, end).
    private static int[] Without(int[] keys, int first, int end)
    {
        foreach (var key in keys)
        {
            if (key >= first && key < end)
            {
                return keys.Where(key => key < first || key >= end).ToArray();
            }
        }

        return keys;
    }

    // S_OR_SAVEEXEC with an inline -1, or S_ORN2_SAVEEXEC of EXEC itself (EXEC | ~EXEC): every
    // lane is set afterwards. A 32-bit form sets only EXEC_LO of a wave64.
    public static bool EntersWholeWave(Gen5ShaderInstruction instruction, bool wave32) =>
        (instruction.Opcode is "SOrSaveexecB64" or "SOrn2SaveexecB64" ||
         (wave32 && instruction.Opcode is "SOrSaveexecB32" or "SOrn2SaveexecB32")) &&
        instruction.Sources.Count == 1 &&
        (instruction.Opcode.StartsWith("SOrn2", StringComparison.Ordinal)
            ? IsExec(instruction.Sources[0])
            : instruction.Sources[0] is { Kind: Gen5OperandKind.EncodedConstant, Value: 193 });

    private static bool IsExec(Gen5Operand operand) =>
        operand.Kind == Gen5OperandKind.ScalarRegister && operand.Value == ExecLow;

    // True when the operand is a full EXEC or a saved copy of one.
    private static bool IsFullCopy(Gen5Operand operand, UInt128 copies, bool execFull, bool wide, bool wave32)
    {
        // Inline -1 sign-extends for a 64-bit scalar operation. A literal dword
        // zero-extends instead, so 0xffffffff covers only a 32-lane wave;
        // it must not prove a wave64 EXEC restore full.
        if (operand is { Kind: Gen5OperandKind.EncodedConstant, Value: 193 })
        {
            return true;
        }
        if ((!wide || wave32) && operand is { Kind: Gen5OperandKind.LiteralConstant, Value: uint.MaxValue })
        {
            return true;
        }

        if (operand.Kind != Gen5OperandKind.ScalarRegister)
        {
            return false;
        }

        if (operand.Value == ExecLow)
        {
            return execFull;
        }

        var mask = Bit(operand.Value) | (wide ? Bit(operand.Value + 1) : 0);
        return (copies & mask) == mask;
    }

    private static UInt128 Bit(uint register) => register < 128 ? UInt128.One << (int)register : 0;
}
