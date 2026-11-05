// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Ir;

/// <summary>
/// Which vector writes may also clobber the lanes EXEC has cleared: those no masked-off lane
/// reads back before writing it again. <see cref="Gen5VectorObservationAnalysis"/> answers
/// the same question but kills a register's liveness only where EXEC is provably full, so in
/// a program that narrows EXEC once at the top - the bounds check of nearly every compute
/// kernel - everything any lane reads later counts as observed, and every write stays guarded.
///
/// This one follows lanes instead. Backwards from the end it keeps:
///   Live: registers some lane active at this point reads before writing; a vector write
///         covers every active lane, so it kills the register here (unless it only merges).
///   Observed: registers some lane inactive here may read later. Each entry carries what is
///         known of those lanes: either nothing (Any), or that they are a subset of the
///         value an SGPR pair holds here - the mask whose restore wakes them up.
/// At a narrowing of EXEC the lanes it turns off are active before it, so what they observe
/// becomes Live there. An S_AND_SAVEEXEC also puts EXEC from before it into its destination,
/// so an entry tagged with that destination is about lanes that were active before it and
/// is dropped from Observed: this is what keeps an inner if's restore from leaking into the
/// writes around the if. A write of EXEC that may turn lanes on makes the registers Live
/// after it Observed before it, tagged with its source mask where it has one (S_MOV, S_OR,
/// S_XOR, S_ANDN2 and their SAVEEXEC forms), untagged otherwise. Any other write of a tag's
/// SGPRs turns the tag into Any. Cross-lane reads are observed in every lane. Where EXEC is
/// provably full (<see cref="Gen5ExecFullAnalysis"/>) no lane is inactive and Observed is
/// empty. A write is unobserved when its registers are not Observed right after it.
/// </summary>
public static class Gen5LaneObservationAnalysis
{
    private const uint ExecLow = 126;
    private const uint ExecHigh = 127;
    private const uint Vcc = 106;

    private readonly record struct Tag(uint Sgpr, VectorRegisterSet Registers);

    private readonly record struct State(VectorRegisterSet Live, VectorRegisterSet Any, Tag[]? Tags)
    {
        public Tag[] TagList => Tags ?? [];

        public VectorRegisterSet Observed()
        {
            var observed = Any;
            foreach (var tag in TagList)
            {
                observed = observed.Union(tag.Registers);
            }

            return observed;
        }

        public bool SameAs(State other) =>
            Live == other.Live && Any == other.Any && TagList.AsSpan().SequenceEqual(other.TagList);

        public static State Union(State left, State right) =>
            new(left.Live.Union(right.Live), left.Any.Union(right.Any), UnionTags(left.TagList, right.TagList));
    }

    private enum ExecKind : byte
    {
        None,
        // EXEC after is a subset of EXEC before.
        Narrow,
        // EXEC after may hold lanes that were off before.
        Reactivate,
    }

    public static Gen5VectorObservationPlan? Analyze(Gen5ShaderProgram program, bool wave32, bool trackLaneSpills)
    {
        var instructions = program.Instructions;
        if (instructions.Count == 0)
        {
            return null;
        }

        var effects = new Gen5VectorObservationAnalysis.Effect[instructions.Count];
        for (var index = 0; index < instructions.Count; index++)
        {
            if (!Gen5VectorObservationAnalysis.TryDescribe(instructions[index], out effects[index]))
            {
                return null;
            }
        }

        if (Gen5VectorObservationAnalysis.BuildSuccessors(instructions) is not { } successors)
        {
            return null;
        }

        var execFull = Gen5ExecFullAnalysis.Analyze(program, wave32, trackLaneSpills);
        var predecessors = new List<int>[instructions.Count];
        for (var index = 0; index < instructions.Count; index++)
        {
            predecessors[index] = [];
        }

        for (var index = 0; index < instructions.Count; index++)
        {
            foreach (var successor in successors[index])
            {
                predecessors[successor].Add(index);
            }
        }

        var states = new State[instructions.Count];
        var queued = new bool[instructions.Count];
        var work = new Queue<int>();
        for (var index = instructions.Count - 1; index >= 0; index--)
        {
            work.Enqueue(index);
            queued[index] = true;
        }

        // Every transfer is monotone over finite sets, so this settles; the budget only
        // guards against a mistake turning that into a hang.
        var budget = (long)instructions.Count * 512;
        while (work.TryDequeue(out var index))
        {
            if (--budget < 0)
            {
                return null;
            }

            queued[index] = false;
            var output = Out(index, successors, states);
            var input = Transfer(instructions[index], effects[index], execFull.Contains(instructions[index].Pc), wave32, output);
            if (input.SameAs(states[index]))
            {
                continue;
            }

            states[index] = input;
            foreach (var predecessor in predecessors[index])
            {
                if (!queued[predecessor])
                {
                    queued[predecessor] = true;
                    work.Enqueue(predecessor);
                }
            }
        }

        var unobserved = new Dictionary<uint, VectorRegisterSet>();
        for (var index = 0; index < instructions.Count; index++)
        {
            var writes = effects[index].Writes;
            if (writes.IsEmpty)
            {
                continue;
            }

            var dead = execFull.Contains(instructions[index].Pc)
                ? writes
                : writes.Except(Out(index, successors, states).Observed());
            if (!dead.IsEmpty)
            {
                unobserved[instructions[index].Pc] = dead;
            }
        }

        return new Gen5VectorObservationPlan(unobserved);
    }

    private static State Out(int index, int[][] successors, State[] states)
    {
        var output = default(State);
        foreach (var successor in successors[index])
        {
            output = State.Union(output, states[successor]);
        }

        return output;
    }

    private static State Transfer(Gen5ShaderInstruction instruction, in Gen5VectorObservationAnalysis.Effect effect,
        bool execFull, bool wave32, State output)
    {
        var live = output.Live;
        var any = output.Any;
        var tags = output.TagList;
        var written = ScalarWrites(instruction);
        var (kind, source, saved) = ClassifyExec(instruction, effect, wave32);

        if (kind != ExecKind.None)
        {
            // Lanes this turns off are active before it: what they read later is Live here.
            live = live.Union(output.Observed());
        }

        // A move between masks: lanes bounded by the destination after it are bounded by
        // the source before it.
        Tag? moved = null;
        if (kind == ExecKind.None && MaskMove(instruction, wave32) is var (moveDestination, moveSource) &&
            TryRemoveTag(ref tags, moveDestination, out var carried))
        {
            moved = new Tag(moveSource, carried);
        }

        // The saved destination holds EXEC from before this instruction: lanes bounded by it
        // are active before it, and already in Live.
        if (saved is { } savedRegister)
        {
            TryRemoveTag(ref tags, savedRegister, out _);
        }

        // A tag names the value its SGPRs hold after this instruction; one this overwrites
        // says nothing about the lanes before it.
        if (written != UInt128.Zero && tags.Length != 0)
        {
            var kept = new List<Tag>(tags.Length);
            foreach (var tag in tags)
            {
                if ((written & TagBits(tag.Sgpr)) != UInt128.Zero)
                {
                    any = any.Union(tag.Registers);
                }
                else
                {
                    kept.Add(tag);
                }
            }

            tags = [.. kept];
        }

        if (moved is { } move)
        {
            if ((written & TagBits(move.Sgpr)) != UInt128.Zero)
            {
                any = any.Union(move.Registers);
            }
            else
            {
                tags = WithTag(tags, move.Sgpr, move.Registers);
            }
        }

        if (kind == ExecKind.Reactivate)
        {
            // Lanes this turns on read what is Live after it.
            if (source is { } mask && (written & TagBits(mask)) == UInt128.Zero)
            {
                tags = WithTag(tags, mask, output.Live);
            }
            else
            {
                any = any.Union(output.Live);
            }
        }

        live = live.Except(effect.FullWrites).Union(effect.Reads).Union(effect.CrossReads);
        any = any.Union(effect.CrossReads);
        if (execFull)
        {
            // No lane is off before this instruction.
            return new State(live, default, null);
        }

        return new State(live, any, tags.Length == 0 ? null : tags);
    }

    // How a write of EXEC changes the active lanes, the SGPR pair bounding any lane it turns
    // on, and the destination of a SAVEEXEC.
    private static (ExecKind Kind, uint? Source, uint? Saved) ClassifyExec(
        Gen5ShaderInstruction instruction, in Gen5VectorObservationAnalysis.Effect effect, bool wave32)
    {
        if (effect.Exec == Gen5VectorObservationAnalysis.ExecEffect.None)
        {
            return (ExecKind.None, null, null);
        }

        var opcode = instruction.Opcode;
        uint? saved = opcode.Contains("Saveexec", StringComparison.Ordinal) &&
            instruction.Destinations is [{ Kind: Gen5OperandKind.ScalarRegister, Value: < ExecLow } destination, ..] &&
            (opcode.EndsWith("B32", StringComparison.Ordinal) ? wave32 : !wave32 || opcode.EndsWith("B64", StringComparison.Ordinal))
            ? destination.Value
            : null;
        if (opcode.StartsWith("SAndn1Saveexec", StringComparison.Ordinal) ||
            effect.Exec == Gen5VectorObservationAnalysis.ExecEffect.Narrowing)
        {
            // ~S0 & EXEC, S0 & EXEC, a compare or V_CMPX: a subset of EXEC.
            return (ExecKind.Narrow, null, saved);
        }

        // A full-width mask only: a 32-bit write in a wave64 keeps EXEC_HI.
        var fullWidth = wave32 || opcode.EndsWith("B64", StringComparison.Ordinal);
        var sources = instruction.Sources;
        uint? source = null;
        if (fullWidth)
        {
            if (opcode is "SOrSaveexecB64" or "SOrSaveexecB32" or "SXorSaveexecB64" or "SXorSaveexecB32" or
                    "SAndn2SaveexecB64" or "SAndn2SaveexecB32" &&
                sources is [{ Kind: Gen5OperandKind.ScalarRegister, Value: < ExecLow } maskSource])
            {
                // S0 | EXEC, S0 ^ EXEC, S0 & ~EXEC: a lane turned on is set in S0.
                source = maskSource.Value;
            }
            else if (instruction.Destinations is [{ Kind: Gen5OperandKind.ScalarRegister, Value: ExecLow }])
            {
                source = opcode switch
                {
                    "SMovB64" or "SMovB32" when sources is [{ Kind: Gen5OperandKind.ScalarRegister, Value: < ExecLow } moved] => moved.Value,
                    "SOrB64" or "SOrB32" or "SXorB64" or "SXorB32" when sources.Count == 2 => OtherThanExec(sources),
                    "SAndn2B64" or "SAndn2B32" when sources is [{ Kind: Gen5OperandKind.ScalarRegister, Value: < ExecLow } kept, { Kind: Gen5OperandKind.ScalarRegister, Value: ExecLow }] => kept.Value,
                    _ => null,
                };
            }
        }

        return (ExecKind.Reactivate, source, saved);
    }

    private static uint? OtherThanExec(IReadOnlyList<Gen5Operand> sources) =>
        sources switch
        {
            [{ Kind: Gen5OperandKind.ScalarRegister, Value: ExecLow }, { Kind: Gen5OperandKind.ScalarRegister, Value: < ExecLow } other] => other.Value,
            [{ Kind: Gen5OperandKind.ScalarRegister, Value: < ExecLow } other, { Kind: Gen5OperandKind.ScalarRegister, Value: ExecLow }] => other.Value,
            _ => null,
        };

    // S_MOV of a whole mask into another SGPR pair (not EXEC).
    private static (uint Destination, uint Source)? MaskMove(Gen5ShaderInstruction instruction, bool wave32) =>
        (instruction.Opcode == "SMovB64" || (wave32 && instruction.Opcode == "SMovB32")) &&
        instruction.Destinations is [{ Kind: Gen5OperandKind.ScalarRegister, Value: < ExecLow } destination] &&
        instruction.Sources is [{ Kind: Gen5OperandKind.ScalarRegister, Value: < ExecLow } source]
            ? (destination.Value, source.Value)
            : null;

    // The SGPRs this instruction writes, EXEC aside; all of them when the decoder does not
    // name its destination.
    private static UInt128 ScalarWrites(Gen5ShaderInstruction instruction)
    {
        var opcode = instruction.Opcode;
        if (opcode.StartsWith("SMovrel", StringComparison.Ordinal))
        {
            return UInt128.MaxValue;
        }

        var wide = opcode.StartsWith('S') &&
            (opcode.EndsWith("B64", StringComparison.Ordinal) ||
             opcode.EndsWith("U64", StringComparison.Ordinal) ||
             opcode.EndsWith("I64", StringComparison.Ordinal));
        UInt128 written = 0;
        foreach (var destination in instruction.Destinations)
        {
            if (destination.Kind == Gen5OperandKind.ScalarRegister)
            {
                written |= Bit(destination.Value) | (wide ? Bit(destination.Value + 1) : 0);
            }
        }

        if (opcode.StartsWith('V'))
        {
            var scalarDestination = opcode.StartsWith("VCmpx", StringComparison.Ordinal) ? null : instruction.Control switch
            {
                Gen5Vop3Control control => control.ScalarDestination,
                Gen5SdwaControl control => control.ScalarDestination,
                _ => null,
            };
            if (scalarDestination is { } sdst)
            {
                written |= Bit(sdst) | Bit(sdst + 1);
            }
            else if ((opcode.StartsWith("VCmp", StringComparison.Ordinal) && !opcode.StartsWith("VCmpx", StringComparison.Ordinal)) ||
                opcode.Contains("Co", StringComparison.Ordinal) ||
                opcode is "VAddcU32" or "VSubbU32" or "VSubbrevU32" ||
                opcode.StartsWith("VDivScale", StringComparison.Ordinal))
            {
                // VOPC compares and VOP2 carries write VCC without naming it.
                written |= Bit(Vcc) | Bit(Vcc + 1);
            }
        }

        return written & ~(Bit(ExecLow) | Bit(ExecHigh));
    }

    private static UInt128 TagBits(uint sgpr) => Bit(sgpr) | Bit(sgpr + 1);

    private static UInt128 Bit(uint register) => register < 128 ? UInt128.One << (int)register : 0;

    private static bool TryRemoveTag(ref Tag[] tags, uint sgpr, out VectorRegisterSet registers)
    {
        for (var index = 0; index < tags.Length; index++)
        {
            if (tags[index].Sgpr == sgpr)
            {
                registers = tags[index].Registers;
                tags = [.. tags[..index], .. tags[(index + 1)..]];
                return true;
            }
        }

        registers = default;
        return false;
    }

    private static Tag[] WithTag(Tag[] tags, uint sgpr, VectorRegisterSet registers)
    {
        if (registers.IsEmpty)
        {
            return tags;
        }

        for (var index = 0; index < tags.Length; index++)
        {
            if (tags[index].Sgpr == sgpr)
            {
                var copy = (Tag[])tags.Clone();
                copy[index] = new Tag(sgpr, tags[index].Registers.Union(registers));
                return copy;
            }

            if (tags[index].Sgpr > sgpr)
            {
                return [.. tags[..index], new Tag(sgpr, registers), .. tags[index..]];
            }
        }

        return [.. tags, new Tag(sgpr, registers)];
    }

    private static Tag[] UnionTags(Tag[] left, Tag[] right)
    {
        if (left.Length == 0)
        {
            return right;
        }

        var result = left;
        foreach (var tag in right)
        {
            result = WithTag(result, tag.Sgpr, tag.Registers);
        }

        return result;
    }
}
