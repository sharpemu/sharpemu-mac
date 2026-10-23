// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Ir;

namespace SharpEmu.ShaderCompiler.Resources;

public enum VertexFetchIndex : byte { Constant, Vertex, Instance }

// (system id / divisor) + bias, in the guest's unsigned index arithmetic.
public readonly record struct AffineVertexIndex(VertexFetchIndex Kind, uint Divisor = 1, uint Bias = 0);

public sealed record AffineVertexFetchLoad(uint Pc, uint Components, AffineVertexIndex Index, uint OffsetBytes, bool Raw = false);

public static partial class EmbeddedVertexFetchDetector
{
    // Straight-line prologs only. Stopping at the first transfer avoids accepting a
    // value whose other reaching definition disagrees, including loop-carried indices.
    // Descriptor words themselves are deliberately left to ScalarValueGraph: it proves
    // all four words, not just the low address word, through constants and lane spills.
    private static IEnumerable<AffineVertexFetchLoad> DetectAffine(Gen5ShaderProgram program)
    {
        var vectors = new AffineVertexIndex?[VectorRegisterCount];
        var scalars = new uint?[ScalarRegisterCount];
        vectors[5] = new(VertexFetchIndex.Vertex);
        vectors[8] = new(VertexFetchIndex.Instance);
        var full = Gen5ExecFullAnalysis.Analyze(program, wave32: true);
        uint? Constant(Gen5Operand operand) => operand.Kind switch
        {
            Gen5OperandKind.LiteralConstant => operand.Value,
            Gen5OperandKind.EncodedConstant when operand.Value == 125 => 0,
            Gen5OperandKind.EncodedConstant when Gen5InlineConstants.TryDecode(operand.Value, out var value) => value,
            Gen5OperandKind.ScalarRegister when operand.Value < ScalarRegisterCount => scalars[operand.Value],
            _ => null,
        };
        AffineVertexIndex? Value(Gen5Operand operand) =>
            operand.Kind == Gen5OperandKind.VectorRegister && operand.Value < VectorRegisterCount ? vectors[operand.Value] :
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
            uint? scalar = null;
            if (sources.Count > 0)
            {
                if (instruction.Opcode == "SMovB32") scalar = Constant(sources[0]);
                if (instruction.Opcode == "SMovkI32") scalar = unchecked((uint)(short)sources[0].Value);
                if (full.Contains(instruction.Pc) && instruction.Control is not Gen5SdwaControl &&
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
                            if (left is { Kind: VertexFetchIndex.Constant } constant && right is { } affine)
                                result = affine with { Bias = unchecked(affine.Bias + constant.Bias) };
                            else if (right is { Kind: VertexFetchIndex.Constant } other && left is { } value)
                                result = value with { Bias = unchecked(value.Bias + other.Bias) };
                        }
                        // V_LSHRREV_B32's constant shift is the first operand.
                        if (instruction.Opcode == "VLshrrevB32" && Constant(sources[0]) is < 32 and var shift &&
                            right is { Bias: 0 } divided && divided.Divisor <= (uint.MaxValue >> (int)shift))
                            result = divided with { Divisor = divided.Divisor << (int)shift };
                    }
                }
            }

            if (instruction.Opcode == "VMovreldB32") Array.Clear(vectors);
            foreach (var destination in instruction.Destinations)
            {
                if (destination.Kind == Gen5OperandKind.VectorRegister && destination.Value < VectorRegisterCount)
                    vectors[destination.Value] = result;
                if (destination.Kind == Gen5OperandKind.ScalarRegister && destination.Value < ScalarRegisterCount)
                    scalars[destination.Value] = scalar;
            }
        }
    }
}
