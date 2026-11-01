// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Vulkan;

public sealed partial class SpirvModuleBuilder
{
    private const uint GlslModf = 35;
    private const uint GlslFrexp = 51;

    private readonly Dictionary<uint, uint> _resultTypes = [];
    private readonly Dictionary<uint, (uint Width, bool Signed)> _integerTypeInfo = [];
    private readonly Dictionary<uint, ulong> _wideConstantValues = [];
    private readonly Dictionary<uint, uint> _floatConstantBits = [];
    private readonly Dictionary<InstructionKey, uint> _blockValues = [];
    private readonly HashSet<uint> _localVariables = [];
    private readonly HashSet<uint> _escapedVariables = [];
    private readonly Dictionary<uint, uint> _variableValues = [];
    private readonly HashSet<(uint Target, SpirvDecoration Decoration)> _plainDecorations = [];
    private readonly HashSet<uint> _namedTargets = [];

    private readonly struct InstructionKey : IEquatable<InstructionKey>
    {
        private readonly int _hash;

        public InstructionKey(SpirvOp opcode, uint type, uint[] operands)
        {
            Opcode = opcode;
            Type = type;
            Operands = operands;
            var hash = new HashCode();
            hash.Add(opcode);
            hash.Add(type);
            foreach (var operand in operands)
            {
                hash.Add(operand);
            }

            _hash = hash.ToHashCode();
        }

        public SpirvOp Opcode { get; }

        public uint Type { get; }

        public uint[] Operands { get; }

        public bool Equals(InstructionKey other) =>
            _hash == other._hash && Opcode == other.Opcode && Type == other.Type &&
            Operands.AsSpan().SequenceEqual(other.Operands);

        public override bool Equals(object? obj) => obj is InstructionKey other && Equals(other);

        public override int GetHashCode() => _hash;
    }

    public uint AddInstruction(SpirvOp opcode, uint resultType, params uint[] operands)
    {
        if (opcode == SpirvOp.Load)
        {
            return AddLoad(resultType, operands);
        }

        NoteVariableUses(operands);
        if (TryFold(opcode, resultType, operands, out var folded))
        {
            return folded;
        }

        if (!IsValueOperation(opcode, operands))
        {
            var effect = EmitResult(_functions, opcode, resultType, operands);
            if (opcode == SpirvOp.FunctionCall)
            {
                ResetMemoryState();
            }

            return effect;
        }

        var lookup = new InstructionKey(opcode, resultType, operands);
        if (_blockValues.TryGetValue(lookup, out var existing))
        {
            return existing;
        }

        var result = EmitResult(_functions, opcode, resultType, operands);
        _blockValues[new InstructionKey(opcode, resultType, operands.ToArray())] = result;
        return result;
    }

    public void AddStatement(SpirvOp opcode, params uint[] operands)
    {
        if (opcode == SpirvOp.Store)
        {
            AddStore(operands);
            return;
        }

        NoteVariableUses(operands);
        Emit(_functions, opcode, operands);
        if (opcode is SpirvOp.Branch or SpirvOp.BranchConditional or SpirvOp.Switch or SpirvOp.Return or
            SpirvOp.ReturnValue or SpirvOp.Kill or SpirvOp.Unreachable)
        {
            ResetBlockState();
        }
        else if (opcode == SpirvOp.FunctionCall)
        {
            ResetMemoryState();
        }
    }

    private uint AddLoad(uint resultType, uint[] operands)
    {
        var pointer = operands[0];
        if (operands.Length != 1 || !IsTrackedVariable(pointer))
        {
            return EmitResult(_functions, SpirvOp.Load, resultType, operands);
        }

        if (_variableValues.TryGetValue(pointer, out var known) && TypeOf(known) == resultType)
        {
            return known;
        }

        var result = EmitResult(_functions, SpirvOp.Load, resultType, operands);
        _variableValues[pointer] = result;
        return result;
    }

    private void AddStore(uint[] operands)
    {
        var pointer = operands[0];
        if (operands.Length != 2 || !IsTrackedVariable(pointer))
        {
            if (_localVariables.Contains(pointer))
            {
                _variableValues.Remove(pointer);
            }

            Emit(_functions, SpirvOp.Store, operands);
            return;
        }

        var value = operands[1];
        if (_variableValues.TryGetValue(pointer, out var current) && current == value)
        {
            return;
        }

        Emit(_functions, SpirvOp.Store, operands);
        _variableValues[pointer] = value;
    }

    private bool IsTrackedVariable(uint pointer) =>
        _localVariables.Contains(pointer) && !_escapedVariables.Contains(pointer);

    private void NoteVariableUses(uint[] operands)
    {
        foreach (var operand in operands)
        {
            if (_localVariables.Contains(operand) && _escapedVariables.Add(operand))
            {
                _variableValues.Remove(operand);
            }
        }
    }

    private void ResetMemoryState() => _variableValues.Clear();

    private void ResetBlockState()
    {
        ResetMemoryState();
        _blockValues.Clear();
    }

    private void NoteResultType(uint id, uint type) => _resultTypes[id] = type;

    private uint TypeOf(uint id) => _resultTypes.GetValueOrDefault(id);

    private uint IntegerWidth(uint type) => _integerTypeInfo.TryGetValue(type, out var info) ? info.Width : 0;

    private bool IsFloat32Type(uint type) => _floatTypes.TryGetValue(32, out var floatType) && floatType == type;

    private static bool IsValueOperation(SpirvOp opcode, uint[] operands) => opcode switch
    {
        SpirvOp.ExtInst => operands.Length >= 2 && operands[1] is not (GlslModf or GlslFrexp),
        SpirvOp.AccessChain or SpirvOp.ArrayLength or SpirvOp.VectorExtractDynamic or SpirvOp.VectorInsertDynamic or
            SpirvOp.VectorShuffle or SpirvOp.CompositeConstruct or SpirvOp.CompositeExtract or SpirvOp.CompositeInsert or
            SpirvOp.CopyObject or SpirvOp.ConvertFToU or SpirvOp.ConvertFToS or SpirvOp.ConvertSToF or SpirvOp.ConvertUToF or
            SpirvOp.ConvertUToPtr or SpirvOp.UConvert or SpirvOp.SConvert or SpirvOp.FConvert or SpirvOp.Bitcast or
            SpirvOp.SNegate or SpirvOp.FNegate or SpirvOp.IAdd or SpirvOp.FAdd or SpirvOp.ISub or SpirvOp.FSub or
            SpirvOp.IMul or SpirvOp.FMul or SpirvOp.UDiv or SpirvOp.SDiv or SpirvOp.FDiv or SpirvOp.UMod or SpirvOp.SRem or
            SpirvOp.SMod or SpirvOp.FRem or SpirvOp.FMod or SpirvOp.IAddCarry or SpirvOp.ISubBorrow or
            SpirvOp.UMulExtended or SpirvOp.SMulExtended or SpirvOp.Any or SpirvOp.All or SpirvOp.IsNan or SpirvOp.IsInf or
            SpirvOp.LogicalEqual or SpirvOp.LogicalNotEqual or SpirvOp.LogicalOr or SpirvOp.LogicalAnd or
            SpirvOp.LogicalNot or SpirvOp.Select or SpirvOp.IEqual or SpirvOp.INotEqual or SpirvOp.UGreaterThan or
            SpirvOp.SGreaterThan or SpirvOp.UGreaterThanEqual or SpirvOp.SGreaterThanEqual or SpirvOp.ULessThan or
            SpirvOp.SLessThan or SpirvOp.ULessThanEqual or SpirvOp.SLessThanEqual or SpirvOp.FOrdEqual or
            SpirvOp.FUnordEqual or SpirvOp.FOrdNotEqual or SpirvOp.FUnordNotEqual or SpirvOp.FOrdLessThan or
            SpirvOp.FUnordLessThan or SpirvOp.FOrdGreaterThan or SpirvOp.FUnordGreaterThan or
            SpirvOp.FOrdLessThanEqual or SpirvOp.FUnordLessThanEqual or SpirvOp.FOrdGreaterThanEqual or
            SpirvOp.FUnordGreaterThanEqual or SpirvOp.ShiftRightLogical or SpirvOp.ShiftRightArithmetic or
            SpirvOp.ShiftLeftLogical or SpirvOp.BitwiseOr or SpirvOp.BitwiseXor or SpirvOp.BitwiseAnd or SpirvOp.Not or
            SpirvOp.BitFieldInsert or SpirvOp.BitFieldSExtract or SpirvOp.BitFieldUExtract or SpirvOp.BitReverse or
            SpirvOp.BitCount => true,
        _ => false,
    };
}
