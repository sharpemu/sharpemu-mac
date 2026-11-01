// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests.Resources;

// s_cmpk_* (SOPK) compares the register its SDST field names with SIMM16 and writes SCC only;
// the decoder hands that register over as the instruction's destination.
public sealed class ScalarCompareImmediateTests
{
    // s5 = 8; s_cmpk_<op> s5, imm; s12 = scc ? 0x40 : 0x80; s13 = s12 + s5; s[20:23] = [s0:s1 + s13]; buffer load.
    private static ShaderResourcePlan Plan(string opcode, ushort immediate)
    {
        var compare = new Gen5ShaderInstruction(
            4, Gen5ShaderEncoding.Sopk, opcode, [0u], [new Gen5Operand(Gen5OperandKind.EncodedConstant, immediate)],
            [Gen5Operand.Scalar(5)], null);
        return Extract(Program(
            MoveScalar(0, 5, 8),
            compare,
            Sop2(8, "SCselectB32", 12, Operand(0x40), Operand(0x80)),
            Sop2(16, "SAddU32", 13, Gen5Operand.Scalar(12), Gen5Operand.Scalar(5)),
            ScalarLoad(20, 0, destination: 20, count: 4, dynamicOffsetRegister: 13),
            BufferLoad(28, 20),
            EndProgram(36)));
    }

    // The descriptor's first word, read from memory whose every word holds its own address.
    private static uint OffsetOf(ShaderResourcePlan plan)
    {
        var memory = new TestWordMemory { Base = 0x1000, Words = new uint[64] };
        for (var index = 0; index < memory.Words.Length; index++)
        {
            memory.Words[index] = 0x1000u + (uint)index * sizeof(uint);
        }

        foreach (var access in plan.Accesses)
        {
            if (access?.Handle is { Kind: ScalarValueKind.BufferHandle } handle)
            {
                Assert.True(new RuntimeValueEvaluator(plan, Inputs([0x1000, 0], memory.Read)).Evaluate(handle.Operands[0], out var word));
                return word - 0x1000u;
            }
        }

        throw new Xunit.Sdk.XunitException("the program has no buffer access");
    }

    [Theory]
    [InlineData("SCmpkGeU32", 3, 0x48u)]
    [InlineData("SCmpkGeU32", 9, 0x88u)]
    [InlineData("SCmpkEqU32", 8, 0x48u)]
    [InlineData("SCmpkLgU32", 8, 0x88u)]
    [InlineData("SCmpkGtI32", 0xFFFF, 0x48u)]
    [InlineData("SCmpkGtU32", 0xFFFF, 0x88u)]
    public void CompareSetsSccAndLeavesTheRegister(string opcode, ushort immediate, uint expected)
    {
        Assert.Equal(expected, OffsetOf(Plan(opcode, immediate)));
    }
}
