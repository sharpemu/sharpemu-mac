// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Metal;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests;

// The output modifiers of a float instruction (clamp and omod) belong to its result whatever the encoding or the width, and an SDWA word with a DWORD select writes all
// of the register: a 16-bit result is the half and zeros there, not the half over the old high half. The Vulkan translator is covered on the device
// (FloatOutputModifierDeviceTests); Metal cannot run here, so these look at the source it generates.
public sealed class Gen5FloatOutputModifierTests
{
    private static Gen5SdwaControl Sdwa(uint destination, uint outputModifier, bool clamp) =>
        new(destination, destination == 6 ? 0u : 2u, destination == 6 ? 6u : 4u, destination == 6 ? 6u : 4u, false, false, 0, 0, outputModifier, clamp, null);

    private static string Metal(Gen5ShaderInstruction instruction)
    {
        var (plan, resources, layout) = Prepare(Program(instruction, EndProgram(8)));
        var request = new ShaderCompileRequest(plan, resources, layout) { LocalSizeX = 1, ThreadCountX = 1, FloatMode = 0 };
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        return shader.Source;
    }

    private static int Count(string text, string pattern)
    {
        var count = 0;
        for (var index = text.IndexOf(pattern, StringComparison.Ordinal); index >= 0; index = text.IndexOf(pattern, index + pattern.Length, StringComparison.Ordinal))
        {
            count++;
        }

        return count;
    }

    private static Gen5ShaderInstruction Binary(string opcode, Gen5InstructionControl? control) =>
        Vop2(0, opcode, 3, Gen5Operand.Vector(0), Gen5Operand.Vector(1)) with { Control = control };

    private static Gen5ShaderInstruction Unary(string opcode, Gen5InstructionControl? control) =>
        Vop1(0, opcode, 3, Gen5Operand.Vector(0)) with { Control = control };

    [Theory]
    [InlineData("VMulF16", 4u)]
    [InlineData("VCvtF32F16", 6u)]
    [InlineData("VCvtF16F32", 6u)]
    [InlineData("VCvtF32I32", 6u)]
    public void SdwaModifiersReachTheResult(string opcode, uint destination)
    {
        Gen5ShaderInstruction Build(Gen5InstructionControl? control) => opcode == "VMulF16" ? Binary(opcode, control) : Unary(opcode, control);
        var plain = Metal(Build(Sdwa(destination, 0, false)));
        var modified = Metal(Build(Sdwa(destination, 1, true)));
        Assert.True(Count(modified, "* 2.0f") > Count(plain, "* 2.0f"), "omod");
        Assert.True(Count(modified, "clamp(") > Count(plain, "clamp("), "clamp");
    }

    [Theory]
    [InlineData("VAddF16", false)]
    [InlineData("VRcpF16", true)]
    public void AnSdwaDwordHalfResultDoesNotKeepTheHighHalf(string opcode, bool unary)
    {
        Gen5ShaderInstruction Build(Gen5InstructionControl? control) => unary ? Unary(opcode, control) : Binary(opcode, control);
        var baseline = Count(Metal(Unary("VMovB32", null)), "0xFFFF0000u");
        Assert.Equal(baseline, Count(Metal(Build(Sdwa(6, 0, false))), "0xFFFF0000u"));
        Assert.Equal(baseline + 1, Count(Metal(Build(new Gen5Vop3Control(0, 0, 0, false, 0, null))), "0xFFFF0000u"));
    }

    // The conversions with a 16-bit result write one half of the destination like the 16-bit arithmetic (HalfConversionDeviceTests).
    [Theory]
    [InlineData("VCvtU16F16")]
    [InlineData("VCvtI16F16")]
    [InlineData("VCvtF16F32")]
    public void APlainConversionToAHalfKeepsTheHighHalf(string opcode)
    {
        var baseline = Count(Metal(Unary("VMovB32", null)), "0xFFFF0000u");
        Assert.Equal(baseline + 1, Count(Metal(Unary(opcode, null)), "0xFFFF0000u"));
        Assert.Equal(baseline, Count(Metal(Unary(opcode, Sdwa(6, 0, false))), "0xFFFF0000u"));
    }
}
