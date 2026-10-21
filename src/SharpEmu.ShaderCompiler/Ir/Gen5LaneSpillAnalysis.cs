// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Ir;

/// <summary>The lane spills a program reloads: which V_WRITELANE_B32s also keep a
/// wave-uniform copy, and which V_READLANE_B32s can read that copy instead of the lane.</summary>
public sealed class Gen5LaneSpillPlan
{
    /// <summary>V_WRITELANE_B32 pc to the copy it updates.</summary>
    public required IReadOnlyDictionary<uint, int> CopyWrites { get; init; }

    /// <summary>V_READLANE_B32 pc to the copy it reads.</summary>
    public required IReadOnlyDictionary<uint, int> CopyReads { get; init; }

    public required int CopyCount { get; init; }
}

/// <summary>
/// The guest compiler spills SGPRs into lanes of a VGPR: V_WRITELANE_B32 vN, sX, K parks
/// a scalar in lane K and V_READLANE_B32 sY, vN, K brings it back. On a host that runs a
/// wave64 as two 32-wide halves the reload is a cross-half exchange with a barrier, and a
/// spill-heavy kernel pays hundreds of them. The parked value is a scalar, so every thread
/// already has it when it is written: a reload whose lane, on every path, still holds what
/// a constant-lane V_WRITELANE_B32 put there can read a per-thread copy taken at that
/// write instead. Anything else that writes the VGPR, a dynamic lane, or a relative move
/// drops the copies it may have overwritten, and the reload goes back to the lane.
/// </summary>
public static class Gen5LaneSpillAnalysis
{
    private const int VectorCount = 256;

    /// <summary>The plan, or null when the program has no copy-served reload or has
    /// dynamic control flow. VGPRs from <paramref name="firstUndefinedVector"/> up are not
    /// launch inputs and start undefined on the hardware, so no correct program depends on
    /// a lane of one that no writelane has reached yet, and the copy's initial 0 serves as
    /// well as anything there. Below it (the thread IDs a compute launch loads) a lane
    /// agrees with its copy only once a writelane has parked a value in it.</summary>
    public static Gen5LaneSpillPlan? Analyze(Gen5ShaderProgram program, uint firstUndefinedVector = VectorCount)
    {
        var instructions = program.Instructions;
        if (instructions.Count == 0)
        {
            return null;
        }

        // Every (VGPR, lane) a constant-lane writelane parks a value in gets a key.
        var keys = new Dictionary<(uint Vector, uint Lane), int>();
        var keysByVector = new List<int>[VectorCount];
        var hasReload = false;
        foreach (var instruction in instructions)
        {
            if (instruction.Opcode == "VReadlaneB32")
            {
                hasReload = true;
            }

            if (instruction.Opcode == "VWritelaneB32" &&
                TryGetSpillSlot(instruction, out var vector, out var lane) &&
                !keys.ContainsKey((vector, lane)))
            {
                keys[(vector, lane)] = keys.Count;
                (keysByVector[vector] ??= []).Add(keys[(vector, lane)]);
            }
        }

        if (keys.Count == 0 || !hasReload)
        {
            return null;
        }

        var successors = Gen5Wave64HalfMaskAnalysis.BuildSuccessors(instructions);
        if (successors is null)
        {
            return null;
        }

        // Forward must-analysis: a key is valid where the copy agrees with its lane on every
        // path, which a constant-lane writelane makes true and any other write of the VGPR
        // undoes. Null is "not reached yet", the top of the lattice.
        var words = (keys.Count + 63) / 64;
        var states = new ulong[]?[instructions.Count];
        states[0] = new ulong[words];
        foreach (var ((vector, _), key) in keys)
        {
            if (vector >= firstUndefinedVector)
            {
                states[0]![key >> 6] |= 1UL << (key & 63);
            }
        }

        var pending = new Queue<int>();
        var queued = new bool[instructions.Count];
        pending.Enqueue(0);
        queued[0] = true;
        while (pending.TryDequeue(out var index))
        {
            queued[index] = false;
            var output = Step(instructions[index], states[index]!, keys, keysByVector);
            foreach (var successor in successors[index])
            {
                var current = states[successor];
                if (current is null)
                {
                    states[successor] = (ulong[])output.Clone();
                }
                else if (!Intersect(current, output))
                {
                    continue;
                }

                if (!queued[successor])
                {
                    queued[successor] = true;
                    pending.Enqueue(successor);
                }
            }
        }

        var reads = new Dictionary<uint, int>();
        var used = new HashSet<int>();
        for (var index = 0; index < instructions.Count; index++)
        {
            var instruction = instructions[index];
            if (states[index] is { } state &&
                instruction.Opcode == "VReadlaneB32" &&
                TryGetSpillSlot(instruction, out var vector, out var lane) &&
                keys.TryGetValue((vector, lane), out var key) &&
                (state[key >> 6] & (1UL << (key & 63))) != 0)
            {
                reads[instruction.Pc] = key;
                used.Add(key);
            }
        }

        if (reads.Count == 0)
        {
            return null;
        }

        // Only the keys some reload reads need a copy; number them densely.
        var copyOf = new Dictionary<int, int>();
        foreach (var key in used.Order())
        {
            copyOf[key] = copyOf.Count;
        }

        var writes = new Dictionary<uint, int>();
        foreach (var instruction in instructions)
        {
            if (instruction.Opcode == "VWritelaneB32" &&
                TryGetSpillSlot(instruction, out var vector, out var lane) &&
                copyOf.TryGetValue(keys[(vector, lane)], out var copy))
            {
                writes[instruction.Pc] = copy;
            }
        }

        return new Gen5LaneSpillPlan
        {
            CopyWrites = writes,
            CopyReads = reads.ToDictionary(static pair => pair.Key, pair => copyOf[pair.Value]),
            CopyCount = copyOf.Count,
        };
    }

    private static ulong[] Step(
        Gen5ShaderInstruction instruction,
        ulong[] input,
        Dictionary<(uint Vector, uint Lane), int> keys,
        List<int>[] keysByVector)
    {
        var output = (ulong[])input.Clone();
        var opcode = instruction.Opcode;
        if (opcode == "VWritelaneB32")
        {
            if (TryGetSpillSlot(instruction, out var vector, out var lane))
            {
                var key = keys[(vector, lane)];
                output[key >> 6] |= 1UL << (key & 63);
            }
            else if (instruction.Destinations is [{ Kind: Gen5OperandKind.VectorRegister } dynamic, ..])
            {
                Kill(output, keysByVector, dynamic.Value, 1);
            }

            return output;
        }

        if (opcode.StartsWith("VMovreld", StringComparison.Ordinal) ||
            opcode.StartsWith("VMovrelsd", StringComparison.Ordinal) ||
            opcode.StartsWith("VSwaprel", StringComparison.Ordinal))
        {
            // The destination is relative to M0: any VGPR.
            Array.Clear(output);
            return output;
        }

        var width = VectorWriteWidth(instruction);
        foreach (var destination in instruction.Destinations)
        {
            if (destination.Kind == Gen5OperandKind.VectorRegister)
            {
                Kill(output, keysByVector, destination.Value, width);
            }
        }

        if (opcode == "VSwapB32" && instruction.Sources is [{ Kind: Gen5OperandKind.VectorRegister } swapped, ..])
        {
            Kill(output, keysByVector, swapped.Value, 1);
        }

        return output;
    }

    /// <summary>How many consecutive VGPRs a destination operand may cover, erring wide: a
    /// memory return gets one dword more than its components for a TFE status word, a
    /// 64-bit ALU result 2, a quad SAD 4, and a memory op this does not model 16.</summary>
    private static uint VectorWriteWidth(Gen5ShaderInstruction instruction)
    {
        var opcode = instruction.Opcode;
        switch (instruction.Control)
        {
            case Gen5ImageControl image:
            {
                var components = (uint)System.Numerics.BitOperations.PopCount(image.Dmask & 0xFu);
                if (opcode.Contains("Gather", StringComparison.Ordinal))
                {
                    components = 4;
                }

                return Math.Max(components, 1u) + 1;
            }
            case Gen5BufferMemoryControl buffer:
                return Math.Max(buffer.DwordCount, 1u) + 1;
            case Gen5GlobalMemoryControl global:
                return Math.Max(global.DwordCount, 1u) + 1;
            case Gen5DataShareControl:
            {
                var width = opcode.Contains("128", StringComparison.Ordinal) ? 4u
                    : opcode.Contains("B96", StringComparison.Ordinal) ? 3u
                    : opcode.Contains("64", StringComparison.Ordinal) && !opcode.Contains("St64B32", StringComparison.Ordinal) ? 2u
                    : 1u;
                return opcode.StartsWith("DsRead2", StringComparison.Ordinal) ||
                    opcode.Contains("xchg2", StringComparison.OrdinalIgnoreCase)
                    ? width * 2
                    : width;
            }
        }

        switch (instruction.Encoding)
        {
            case Gen5ShaderEncoding.Mubuf:
            case Gen5ShaderEncoding.Mtbuf:
            case Gen5ShaderEncoding.Mimg:
            case Gen5ShaderEncoding.Flat:
            case Gen5ShaderEncoding.Ds:
                return 16;
        }

        if (opcode.Contains("qsad", StringComparison.OrdinalIgnoreCase))
        {
            return 4;
        }

        return opcode.Contains("64", StringComparison.Ordinal) ? 2u : 1u;
    }

    private static void Kill(ulong[] state, List<int>[] keysByVector, uint first, uint width)
    {
        for (var vector = first; vector < first + width && vector < VectorCount; vector++)
        {
            if (keysByVector[vector] is { } vectorKeys)
            {
                foreach (var key in vectorKeys)
                {
                    state[key >> 6] &= ~(1UL << (key & 63));
                }
            }
        }
    }

    /// <summary>Meets <paramref name="state"/> with <paramref name="incoming"/> in place;
    /// true when it changed.</summary>
    private static bool Intersect(ulong[] state, ulong[] incoming)
    {
        var changed = false;
        for (var index = 0; index < state.Length; index++)
        {
            var merged = state[index] & incoming[index];
            if (merged != state[index])
            {
                state[index] = merged;
                changed = true;
            }
        }

        return changed;
    }

    /// <summary>The VGPR and lane of a V_WRITELANE_B32 / V_READLANE_B32 whose lane select
    /// is an inline or literal constant (a wave64 masks it to 0..63). A writelane also
    /// has to park a scalar or constant, which every thread holds.</summary>
    private static bool TryGetSpillSlot(Gen5ShaderInstruction instruction, out uint vector, out uint lane)
    {
        vector = 0;
        lane = 0;
        if (instruction.Sources.Count < 2 || !TryGetConstant(instruction.Sources[1], out var select))
        {
            return false;
        }

        lane = select & 63u;
        if (instruction.Opcode == "VWritelaneB32")
        {
            if (instruction.Sources[0].Kind == Gen5OperandKind.VectorRegister ||
                instruction.Destinations is not [{ Kind: Gen5OperandKind.VectorRegister } destination, ..])
            {
                return false;
            }

            vector = destination.Value;
            return true;
        }

        if (instruction.Sources[0].Kind != Gen5OperandKind.VectorRegister)
        {
            return false;
        }

        vector = instruction.Sources[0].Value;
        return true;
    }

    private static bool TryGetConstant(Gen5Operand operand, out uint value)
    {
        switch (operand.Kind)
        {
            case Gen5OperandKind.LiteralConstant:
                value = operand.Value;
                return true;
            case Gen5OperandKind.EncodedConstant when operand.Value is >= 128 and <= 192:
                value = operand.Value - 128;
                return true;
            case Gen5OperandKind.EncodedConstant when operand.Value is >= 193 and <= 208:
                value = unchecked((uint)(192 - (int)operand.Value));
                return true;
            default:
                value = 0;
                return false;
        }
    }
}
