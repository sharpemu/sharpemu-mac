// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Ir;

namespace SharpEmu.ShaderCompiler.Resources;

public enum VertexFetchIndex : byte { Constant, Vertex, Instance }

// (system id / divisor) + bias, in the guest's unsigned index arithmetic.
public readonly record struct AffineVertexIndex(VertexFetchIndex Kind, uint Divisor = 1, uint Bias = 0,
    int BiasScalarRegister = -1, uint BiasMask = uint.MaxValue);

public sealed record AffineVertexFetchLoad(uint Pc, uint Components, AffineVertexIndex Index, uint OffsetBytes, bool Raw = false);

public static partial class EmbeddedVertexFetchDetector
{
    // Straight-line prologs only. Stopping at the first transfer avoids accepting a
    // value whose other reaching definition disagrees, including loop-carried indices.
    // Descriptor words themselves are deliberately left to ScalarValueGraph: it proves
    // all four words, not just the low address word, through constants and lane spills.
    private static IEnumerable<AffineVertexFetchLoad> DetectAffine(Gen5ShaderProgram program, uint userDataBase, uint userDataCount)
    {
        var vectors = new AffineVertexIndex?[VectorRegisterCount];
        var scalars = new AffineVertexIndex?[ScalarRegisterCount];
        for (uint register = userDataBase; register < Math.Min(userDataBase + userDataCount, ScalarRegisterCount); register++)
            scalars[register] = new(VertexFetchIndex.Constant, BiasScalarRegister: (int)register);
        var lanes = new Dictionary<(uint, uint), AffineVertexIndex>();
        vectors[5] = new(VertexFetchIndex.Vertex);
        vectors[8] = new(VertexFetchIndex.Instance);
        var partial = new HashSet<uint>();
        // Partial EXEC is safe when the defining arithmetic and the fetch use the
        // same mask; only those lanes execute either instruction.
        var full = Gen5ExecFullAnalysis.Analyze(program, wave32: true);
        uint? Constant(Gen5Operand operand) => operand.Kind switch
        {
            Gen5OperandKind.LiteralConstant => operand.Value,
            Gen5OperandKind.EncodedConstant when operand.Value == 125 => 0,
            Gen5OperandKind.EncodedConstant when Gen5InlineConstants.TryDecode(operand.Value, out var value) => value,
            Gen5OperandKind.ScalarRegister when operand.Value < ScalarRegisterCount && scalars[operand.Value] is { Kind: VertexFetchIndex.Constant, BiasScalarRegister: -1 } known => known.Bias,
            _ => null,
        };
        AffineVertexIndex? Value(Gen5Operand operand) =>
            operand.Kind == Gen5OperandKind.VectorRegister && operand.Value < VectorRegisterCount ? vectors[operand.Value] :
            operand.Kind == Gen5OperandKind.ScalarRegister && operand.Value < ScalarRegisterCount ? scalars[operand.Value] :
            Constant(operand) is { } value ? new(VertexFetchIndex.Constant, 1, value) : null;

        foreach (var instruction in program.Instructions)
        {
            if (Gen5IrBranchResolver.Instance.TryGetBranchTarget(instruction, out _) ||
                Gen5IrBranchResolver.IsTerminator(instruction)) yield break;
            var sources = instruction.Sources;
            var raw = instruction.Opcode is "BufferLoadDword" or "BufferLoadDwordx2" or "BufferLoadDwordx3" or "BufferLoadDwordx4";
            if (instruction.Control is Gen5BufferMemoryControl buffer && (IsFetchBufferLoad(instruction.Opcode) || raw) &&
                !buffer.Typed && !buffer.Tfe && !buffer.Lds && !buffer.Glc && !buffer.Slc && !buffer.Dlc && !buffer.OffsetEnabled &&
                buffer.OffsetBytes >= 0 && sources.Count > 2 && Constant(sources[2]) is { } scalarOffset &&
                (ulong)scalarOffset + (uint)buffer.OffsetBytes <= uint.MaxValue)
            {
                var index = buffer.IndexEnabled && buffer.VectorAddress < VectorRegisterCount
                    ? vectors[buffer.VectorAddress] : !buffer.IndexEnabled ? new AffineVertexIndex(VertexFetchIndex.Constant) : null;
                if (index is { } known)
                    yield return new(instruction.Pc, buffer.DwordCount, known, scalarOffset + (uint)buffer.OffsetBytes, raw);
            }

            AffineVertexIndex? result = null;
            AffineVertexIndex? scalar = null;
            if (sources.Count > 0)
            {
                if (instruction.Opcode == "SMovB32") scalar = Value(sources[0]);
                if (instruction.Opcode == "SMovkI32") scalar = new(VertexFetchIndex.Constant, Bias: unchecked((uint)(short)sources[0].Value));
                if (instruction.Control is not Gen5SdwaControl &&
                    instruction.Control is not Gen5Vop3Control { AbsoluteMask: not 0 } &&
                    instruction.Control is not Gen5Vop3Control { NegateMask: not 0 } &&
                    instruction.Control is not Gen5Vop3Control { OutputModifier: not 0 } &&
                    instruction.Control is not Gen5Vop3Control { Clamp: true } &&
                    instruction.Control is not Gen5Vop3Control { OperandSelect: not 0 })
                {
                    if (instruction.Opcode == "VMovB32") result = Value(sources[0]);
                    if (sources.Count >= 2)
                    {
                        var left = Value(sources[0]);
                        var right = Value(sources[1]);
                        if (instruction.Opcode is "VAddI32" or "VAddU32" or "VAddNcU32")
                        {
                            result = Add(left, right) ?? Add(right, left);
                        }
                        // V_LSHRREV_B32's constant shift is the first operand.
                        if (instruction.Opcode == "VLshrrevB32" && Constant(sources[0]) is < 32 and var shift &&
                            right is { Bias: 0, BiasScalarRegister: -1 } divided && divided.Divisor <= (uint.MaxValue >> (int)shift))
                            result = divided with { Divisor = divided.Divisor << (int)shift };
                    }
                }
            }

            // A low-word scalar SDWA operand is common in embedded instance fetches.
            // Only the full-width, unsigned vector + low unsigned scalar form is affine.
            if ((instruction.Opcode is "VAddI32" or "VAddU32") &&
                instruction.Control is Gen5SdwaControl
                { DestinationSelect: 6, Source0Select: 6, Source1Select: 4,
                  Source0SignExtend: false, Source1SignExtend: false, AbsoluteMask: 0, NegateMask: 0,
                  OutputModifier: 0, Clamp: false } && sources.Count == 2 &&
                Value(sources[1]) is { Kind: VertexFetchIndex.Constant } bias &&
                (bias.BiasScalarRegister < 0 || bias.Bias == 0))
                result = Add(bias with { Bias = bias.Bias & 0xFFFF, BiasMask = bias.BiasMask & 0xFFFF }, Value(sources[0]));
            if (instruction.Opcode == "VWritelaneB32" && sources.Count >= 2 &&
                instruction.Destinations is [{ Kind: Gen5OperandKind.VectorRegister } laneVector, ..] &&
                Constant(sources[1]) is { } writeLane && Value(sources[0]) is { } saved)
                lanes[(laneVector.Value, writeLane & 31)] = saved;
            else
                foreach (var destination in instruction.Destinations.Where(d => d.Kind == Gen5OperandKind.VectorRegister))
                    foreach (var key in lanes.Keys.Where(key => key.Item1 == destination.Value).ToArray()) lanes.Remove(key);
            if (instruction.Opcode == "VReadlaneB32" && sources.Count >= 2 &&
                Constant(sources[1]) is { } readLane && lanes.TryGetValue((sources[0].Value, readLane & 31), out var restored))
                scalar = restored;
            if (instruction.Opcode.Contains("Movrel", StringComparison.OrdinalIgnoreCase))
            {
                Array.Clear(vectors);
                Array.Clear(scalars);
                lanes.Clear();
            }
            foreach (var destination in instruction.Destinations)
            {
                if (destination.Kind == Gen5OperandKind.VectorRegister && destination.Value < VectorRegisterCount)
                {
                    vectors[destination.Value] = result;
                    if (full.Contains(instruction.Pc)) partial.Remove(destination.Value);
                    else partial.Add(destination.Value);
                }
                if (destination.Kind == Gen5OperandKind.ScalarRegister && destination.Value < ScalarRegisterCount)
                    scalars[destination.Value] = scalar;
            }
            var scalarDestination = instruction.Control switch
            {
                Gen5Vop3Control control => control.ScalarDestination,
                Gen5SdwaControl control => control.ScalarDestination,
                _ => null,
            };
            if (scalarDestination is { } pair)
                for (uint register = pair; register <= pair + 1 && register < ScalarRegisterCount; register++) scalars[register] = null;
            // A predicated definition is valid at a later fetch only while EXEC is
            // unchanged. A newly enabled lane would otherwise see the old VGPR.
            if (instruction.Opcode.Contains("exec", StringComparison.OrdinalIgnoreCase) ||
                instruction.Opcode.Contains("Cmpx", StringComparison.Ordinal) ||
                instruction.Opcode.Contains("Cmpsx", StringComparison.Ordinal) ||
                instruction.Destinations.Any(d => d.Kind == Gen5OperandKind.ScalarRegister &&
                    d.Value <= 127 && d.Value + instruction.DestinationWidth > 126))
            {
                foreach (var register in partial) vectors[register] = null;
                partial.Clear();
            }
        }
    }

    private static AffineVertexIndex? Add(AffineVertexIndex? left, AffineVertexIndex? right)
    {
        if (left is not { Kind: VertexFetchIndex.Constant } constant || right is not { } affine ||
            (constant.BiasScalarRegister >= 0 && affine.BiasScalarRegister >= 0)) return null;
        return affine with
        {
            Bias = unchecked(affine.Bias + constant.Bias),
            BiasScalarRegister = constant.BiasScalarRegister >= 0 ? constant.BiasScalarRegister : affine.BiasScalarRegister,
            BiasMask = constant.BiasScalarRegister >= 0 ? constant.BiasMask : affine.BiasMask,
        };
    }
}
