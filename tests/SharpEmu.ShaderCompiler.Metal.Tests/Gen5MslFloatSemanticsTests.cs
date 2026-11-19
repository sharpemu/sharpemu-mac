// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>Bit-exact ALU semantics from the RDNA2 ISA (sections 6.2, 12.7-12.10) that the
/// Metal host does not give for free. Each case runs one inserted instruction on a single
/// lane with v0..v3 preset and stores v3; execution needs the exclusive device gate.</summary>
public sealed class Gen5MslFloatSemanticsTests(ITestOutputHelper output)
{
    // V_ADD/SUB/SUBREV_NC_U32 (VOP2 0x25-0x27) saturate as unsigned under an SDWA clamp; only
    // the VOP3-only V_ADD/SUB_NC_I32 saturate as signed.
    [Theory]
    [InlineData("VAddI32", 0xFFFFFFF0u, 0x20u, 0xFFFFFFFFu)]
    [InlineData("VSubI32", 3u, 5u, 0u)]
    [InlineData("VSubrevI32", 5u, 3u, 0u)]
    public void Vop2NoCarryAddSubClampSaturatesUnsigned(string opcode, uint a, uint b, uint expected)
    {
        var shader = Compile([
            new Gen5ShaderInstruction(28, Gen5ShaderEncoding.Vop2, opcode, [], [Gen5Operand.Vector(0), Gen5Operand.Vector(1)],
                [Gen5Operand.Vector(3)], new Gen5SdwaControl(6, 0, 6, 6, false, false, 0, 0, 0, true, null)),
        ], a, b);
        if (!RequireDevice()) return;
        Assert.Equal(expected, Execute(shader));
    }

    // A 16-bit float operand reads an integer inline constant as its bit pattern (129 is the
    // denormal 0x0001, 193 is 0xFFFF) and 248 as the f16 1/(2*pi), 0x3118.
    [Theory]
    [InlineData("VAddF16", 129u, 0x0001u, 0xABCD0002u)]
    [InlineData("VMulF16", 248u, 0x3C00u, 0xABCD3118u)]
    [InlineData("VMinF16", 130u, 0x0003u, 0xABCD0002u)]
    [InlineData("VMaxF16", 192u, 0x0003u, 0xABCD0040u)]
    public void HalfOperandsReadIntegerInlineConstantsAsBits(string opcode, uint constant, uint b, uint expected)
    {
        var shader = Compile([Vop3(opcode, [Gen5Operand.Source(constant), Gen5Operand.Vector(1)])], 0u, b, 0u, 0xABCD5678u);
        if (!RequireDevice()) return;
        Assert.Equal(expected, Execute(shader));
    }

    [Fact]
    public void FmaMixReadsAHalfSelectedInlineConstantAsF16()
    {
        // fma_mix(src0 = 1 as f16 (0x0001), src1 = v1 as f32 2^24, src2 = v2 as f32 0) = 1.0
        var shader = Compile([Vop3("VFmaMixF32", [Gen5Operand.Source(129), Gen5Operand.Vector(1), Gen5Operand.Vector(2)],
            new Gen5Vop3pControl(0, 1, 0, 0, false))], 0u, 0x4B800000u, 0u);
        if (!RequireDevice()) return;
        Assert.Equal(0x3F800000u, Execute(shader));
    }

    // V_CMP_CLASS_F32 tells a signalling NaN (bit 0) from a quiet one (bit 1) and reports an
    // f32 denormal as a subnormal (bits 4/7) although the host flushes it in float compares.
    [Theory]
    [InlineData(0x7F800001u, 0x001u, 1u)]
    [InlineData(0x7F800001u, 0x002u, 0u)]
    [InlineData(0x7FC00000u, 0x002u, 1u)]
    [InlineData(0x7FC00000u, 0x001u, 0u)]
    [InlineData(0x00400000u, 0x080u, 1u)]
    [InlineData(0x00400000u, 0x040u, 0u)]
    [InlineData(0x80000001u, 0x010u, 1u)]
    [InlineData(0x80000001u, 0x020u, 0u)]
    [InlineData(0x80000000u, 0x020u, 1u)]
    public void CompareClassF32ReadsTheBits(uint value, uint mask, uint expected)
    {
        var shader = Compile([
            new Gen5ShaderInstruction(28, Gen5ShaderEncoding.Vopc, "VCmpClassF32", [], [Gen5Operand.Vector(0), Gen5Operand.Vector(1)], [], null),
            CndmaskVcc(),
        ], value, mask);
        if (!RequireDevice()) return;
        Assert.Equal(expected, Execute(shader));
    }

    // V_MIN/V_MAX order -0 below +0 (ISA 12.7); V_MED3 is the max of the two operands other
    // than the one MAX3 matches; in IEEE mode a signalling NaN comes back quieted, and V_MIN3
    // then takes the min of that quiet NaN and src2, which is src2.
    [Theory]
    [InlineData("VMinF32", 0x00000000u, 0x80000000u, 0u, false, 0x80000000u)]
    [InlineData("VMinF32", 0x80000000u, 0x00000000u, 0u, false, 0x80000000u)]
    [InlineData("VMaxF32", 0x00000000u, 0x80000000u, 0u, false, 0x00000000u)]
    [InlineData("VMaxF32", 0x80000000u, 0x00000000u, 0u, false, 0x00000000u)]
    [InlineData("VMin3F32", 0x3F800000u, 0x00000000u, 0x80000000u, false, 0x80000000u)]
    [InlineData("VMax3F32", 0xBF800000u, 0x80000000u, 0x00000000u, false, 0x00000000u)]
    [InlineData("VMed3F32", 0x00000000u, 0x80000000u, 0x80000000u, false, 0x80000000u)]
    [InlineData("VMed3F32", 0x477FE000u, 0x7FC00000u, 0u, false, 0u)]
    [InlineData("VMinF32", 0x7F800001u, 0x3F800000u, 0u, false, 0x3F800000u)]
    [InlineData("VMinF32", 0x7F800001u, 0x3F800000u, 0u, true, 0x7FC00001u)]
    [InlineData("VMaxF32", 0x3F800000u, 0xFF800002u, 0u, true, 0xFFC00002u)]
    [InlineData("VMaxF32", 0x7FC00000u, 0x3F800000u, 0u, true, 0x3F800000u)]
    [InlineData("VMin3F32", 0x7F800001u, 0x3F800000u, 0x40000000u, true, 0x40000000u)]
    [InlineData("VMaxF16", 0x00000000u, 0x00008000u, 0u, false, 0xABCD0000u)]
    [InlineData("VMinF16", 0x00000000u, 0x00008000u, 0u, false, 0xABCD8000u)]
    [InlineData("VMed3F16", 0x00000000u, 0x00008000u, 0x00008000u, false, 0xABCD8000u)]
    [InlineData("VMaxF16", 0x00007C01u, 0x00003C00u, 0u, true, 0xABCD7E01u)]
    [InlineData("VMax3F16", 0x0000BC00u, 0x00008000u, 0x00000000u, false, 0xABCD0000u)]
    public void MinMaxFollowTheIsaZeroAndNanRules(string opcode, uint a, uint b, uint c, bool ieee, uint expected)
    {
        var old = opcode.EndsWith("F16", StringComparison.Ordinal) ? 0xABCD5678u : 0u;
        var shader = Compile([Vop3(opcode, V012)], a, b, c, old, ieee);
        if (!RequireDevice()) return;
        Assert.Equal(expected, Execute(shader));
    }

    [Theory]
    [InlineData("VPkMaxF16", 0x80000000u, 0x00008000u, 0x00000000u)]
    [InlineData("VPkMinF16", 0x80000000u, 0x00008000u, 0x80008000u)]
    public void PackedMinMaxOrderNegativeZeroBelowPositiveZero(string opcode, uint a, uint b, uint expected)
    {
        var shader = Compile([Vop3(opcode, [Gen5Operand.Vector(0), Gen5Operand.Vector(1)], new Gen5Vop3pControl(0, 3, 0, 0, false))], a, b);
        if (!RequireDevice()) return;
        Assert.Equal(expected, Execute(shader));
    }

    // (1 + 2^-23)^2 - (1 + 2^-22) is 2^-46 exactly: a fused multiply-add keeps it, a rounded
    // product loses it. V_MAD/MAC (and the legacy forms on gfx10.1) round the product.
    [Theory]
    [InlineData("VFmaF32", Gen5ShaderEncoding.Vop3, 0x28800000u)]
    [InlineData("VMadF32", Gen5ShaderEncoding.Vop3, 0u)]
    [InlineData("VMadLegacyF32", Gen5ShaderEncoding.Vop3, 0u)]
    [InlineData("VMacF32", Gen5ShaderEncoding.Vop2, 0u)]
    [InlineData("VFmacF32", Gen5ShaderEncoding.Vop2, 0x28800000u)]
    [InlineData("VMacLegacyF32", Gen5ShaderEncoding.Vop2, 0u)]
    public void MadFormsRoundTheProductBeforeTheAdd(string opcode, Gen5ShaderEncoding encoding, uint expected)
    {
        const uint Factor = 0x3F800001u, Addend = 0xBF800002u;
        var instruction = encoding == Gen5ShaderEncoding.Vop3
            ? Vop3(opcode, V012)
            : new Gen5ShaderInstruction(28, encoding, opcode, [], [Gen5Operand.Vector(0), Gen5Operand.Vector(1)], [Gen5Operand.Vector(3)], null);
        var exact = Gen5MslCodegenOptions.FromEnvironment with { FuseLegacyMultiplyAdd = false };
        Assert.True(TryCompile([instruction], Factor, Factor, Addend, Addend, out var shader, out var error, options: exact), error);
        if (!RequireDevice()) return;
        Assert.Equal(expected, Execute(shader!));
    }

    // Gen5MslCodegenOptions.FuseLegacyMultiplyAdd trades that rounding for one operation in a compute
    // program: the same inputs keep the 2^-46 a fused multiply-add keeps. Off, it is lost.
    [Theory]
    [InlineData("VMadF32", Gen5ShaderEncoding.Vop3)]
    [InlineData("VMacF32", Gen5ShaderEncoding.Vop2)]
    public void FusedLegacyMultiplyAddKeepsTheProductUnrounded(string opcode, Gen5ShaderEncoding encoding)
    {
        const uint Factor = 0x3F800001u, Addend = 0xBF800002u;
        var instruction = encoding == Gen5ShaderEncoding.Vop3
            ? Vop3(opcode, V012)
            : new Gen5ShaderInstruction(28, encoding, opcode, [], [Gen5Operand.Vector(0), Gen5Operand.Vector(1)], [Gen5Operand.Vector(3)], null);
        var options = Gen5MslCodegenOptions.FromEnvironment with { FuseLegacyMultiplyAdd = true };
        Assert.True(TryCompile([instruction], Factor, Factor, Addend, Addend, out var fused, out var error, options: options), error);
        Assert.Contains("fma(", fused!.Source);
        var exact = options with { FuseLegacyMultiplyAdd = false };
        Assert.True(TryCompile([instruction], Factor, Factor, Addend, Addend, out var unfused, out error, options: exact), error);
        if (!RequireDevice()) return;
        Assert.Equal(0x28800000u, Execute(fused));
        Assert.Equal(0u, Execute(unfused!));
    }

    // Fusing is the default; SHARPEMU_MSL_FUSE_MAD=0 (FuseLegacyMultiplyAdd false) restores the exact form.
    [Fact]
    public void LegacyMultiplyAddFusesByDefaultAndNotInExactMode()
    {
        Assert.True(new Gen5MslCodegenOptions().FuseLegacyMultiplyAdd);
        var instruction = Vop3("VMadF32", V012);
        Assert.True(TryCompile([instruction], 0u, 0u, 0u, 0u, out var byDefault, out var error, options: new Gen5MslCodegenOptions()), error);
        Assert.Contains("fma(", byDefault!.Source);
        var exact = new Gen5MslCodegenOptions { FuseLegacyMultiplyAdd = false };
        Assert.True(TryCompile([instruction], 0u, 0u, 0u, 0u, out var unfused, out error, options: exact), error);
        Assert.DoesNotContain("fma(", unfused!.Source);
    }

    // V_MAD_LEGACY_F32 flushes its denormal sources, so a denormal factor takes the DX9 zero
    // rule: denormal * inf + -0 is +0, not the NaN of 0 * inf.
    [Fact]
    public void MadLegacyTreatsADenormalFactorAsZero()
    {
        var shader = Compile([Vop3("VMadLegacyF32", V012)], 0x00000001u, 0x7F800000u, 0x80000000u);
        if (!RequireDevice()) return;
        Assert.Equal(0u, Execute(shader));
    }

    // gfx10 16-bit results keep the other half of the destination VGPR, in VOP1 and VOP3 forms.
    [Theory]
    [InlineData("VCvtF16F32", Gen5ShaderEncoding.Vop1, 0x3F800000u, 0xABCD3C00u)]
    [InlineData("VCvtF16F32", Gen5ShaderEncoding.Vop3, 0x3F800000u, 0xABCD3C00u)]
    [InlineData("VCvtU16F16", Gen5ShaderEncoding.Vop1, 0x00004500u, 0xABCD0005u)]
    [InlineData("VCvtU16F16", Gen5ShaderEncoding.Vop3, 0x00004500u, 0xABCD0005u)]
    public void HalfResultsKeepTheUpperHalf(string opcode, Gen5ShaderEncoding encoding, uint a, uint expected)
    {
        var shader = Compile([
            new Gen5ShaderInstruction(28, encoding, opcode, [], [Gen5Operand.Vector(0)], [Gen5Operand.Vector(3)],
                encoding == Gen5ShaderEncoding.Vop3 ? new Gen5Vop3Control(0, 0, 0, false, 0, null) : null),
        ], a, 0u, 0u, 0xABCD5678u);
        if (!RequireDevice()) return;
        Assert.Equal(expected, Execute(shader));
    }

    // omod applies only outside IEEE mode and while FLOAT_MODE flushes output denormals of
    // the result's precision (bit 5 for f32, bit 7 for f16/f64); it flushes -0 to +0.
    [Theory]
    [InlineData("VMulF32", 0x3FC00000u, 0x3F800000u, 0xC0u, 0x40400000u)] // 1.5 * 1, omod *2 = 3
    [InlineData("VMulF32", 0x3FC00000u, 0x3F800000u, 0xF0u, 0x3FC00000u)] // f32 output denormals kept
    [InlineData("VMulF32", 0xBF800000u, 0x00000000u, 0xC0u, 0x00000000u)] // -1 * 0, omod: +0
    [InlineData("VMulF16", 0x00003E00u, 0x00003C00u, 0xC0u, 0xABCD3E00u)] // f16 output denormals kept
    [InlineData("VMulF16", 0x00003E00u, 0x00003C00u, 0x00u, 0xABCD4200u)] // 1.5h * 1h, omod *2 = 3h
    [InlineData("VMulF16", 0x0000BC00u, 0x00000000u, 0x00u, 0xABCD0000u)] // -1h * 0h, omod: +0
    public void OutputModifierFollowsFloatMode(string opcode, uint a, uint b, uint floatMode, uint expected)
    {
        var shader = Compile([Vop3(opcode, [Gen5Operand.Vector(0), Gen5Operand.Vector(1)],
            new Gen5Vop3Control(0, 0, 1, false, 0, null))], a, b, 0u, 0xABCD5678u, floatMode: floatMode);
        if (!RequireDevice()) return;
        Assert.Equal(expected, Execute(shader));
    }

    [Theory]
    [InlineData(false, 0x40u, 0x40080000u)] // 1.5 * 1.0, omod *2 = 3.0
    [InlineData(true, 0x40u, 0x3FF80000u)]  // IEEE mode ignores omod
    [InlineData(false, 0xC0u, 0x3FF80000u)] // f64 output denormals kept
    public void Float64OutputModifierFollowsIeeeAndFloatMode(bool ieee, uint floatMode, uint expectedHigh)
    {
        var shader = Compile([
            new Gen5ShaderInstruction(28, Gen5ShaderEncoding.Vop3, "VMulF64", [], [Gen5Operand.Vector(0), Gen5Operand.Vector(2)],
                [Gen5Operand.Vector(4)], new Gen5Vop3Control(0, 0, 1, false, 0, null)),
            new Gen5ShaderInstruction(36, Gen5ShaderEncoding.Vop1, "VMovB32", [], [Gen5Operand.Vector(5)], [Gen5Operand.Vector(3)], null),
        ], 0u, 0x3FF80000u, 0u, 0x3FF00000u, ieee, floatMode);
        if (!RequireDevice()) return;
        Assert.Equal(expectedHigh, Execute(shader));
    }

    // A clamped NaN is 0 under DX10_CLAMP and stays NaN without it.
    [Theory]
    [InlineData("VAddF32", 0x7FC00000u, 0x3F800000u, true, 0x00000000u)]
    [InlineData("VAddF32", 0x7FC00000u, 0x3F800000u, false, 0x7FC00000u)]
    [InlineData("VAddF16", 0x00007E00u, 0x00003C00u, true, 0xABCD0000u)]
    [InlineData("VAddF16", 0x00007E00u, 0x00003C00u, false, 0xABCD7E00u)]
    [InlineData("VMaxF32", 0x7FC00000u, 0x7FC00000u, true, 0x00000000u)]
    [InlineData("VMaxF32", 0x7FC00000u, 0x7FC00000u, false, 0x7FC00000u)]
    [InlineData("VMaxF16", 0x00007E00u, 0x00007E00u, true, 0xABCD0000u)]
    [InlineData("VMaxF16", 0x00007E00u, 0x00007E00u, false, 0xABCD7E00u)]
    public void ClampOfNanFollowsDx10Clamp(string opcode, uint a, uint b, bool dx10Clamp, uint expected)
    {
        var shader = Compile([Vop3(opcode, [Gen5Operand.Vector(0), Gen5Operand.Vector(1)],
            new Gen5Vop3Control(0, 0, 0, true, 0, null))], a, b, 0u, 0xABCD5678u, dx10Clamp: dx10Clamp);
        if (!RequireDevice()) return;
        Assert.Equal(expected, Execute(shader));
    }

    [Theory]
    [InlineData(true, 0x00000000u)]
    [InlineData(false, 0x7FF80000u)]
    public void Float64ClampOfNanFollowsDx10Clamp(bool dx10Clamp, uint expectedHigh)
    {
        var shader = Compile([
            new Gen5ShaderInstruction(28, Gen5ShaderEncoding.Vop3, "VAddF64", [], [Gen5Operand.Vector(0), Gen5Operand.Vector(2)],
                [Gen5Operand.Vector(4)], new Gen5Vop3Control(0, 0, 0, true, 0, null)),
            new Gen5ShaderInstruction(36, Gen5ShaderEncoding.Vop1, "VMovB32", [], [Gen5Operand.Vector(5)], [Gen5Operand.Vector(3)], null),
        ], 0u, 0x7FF80000u, 0u, 0x3FF00000u, dx10Clamp: dx10Clamp);
        if (!RequireDevice()) return;
        Assert.Equal(expectedHigh, Execute(shader));
    }

    // The bit-exact min/max and frexp paths flush an f32 input denormal to a zero of its sign
    // while FLOAT_MODE flushes f32 input denormals (bit 4 clear), and keep it otherwise.
    [Theory]
    [InlineData("VMinF32", 0x00000001u, 0x3F800000u, 0xC0u, 0x00000000u)]
    [InlineData("VMinF32", 0x00000001u, 0x3F800000u, 0xF0u, 0x00000001u)]
    [InlineData("VMaxF32", 0x80000001u, 0xBF800000u, 0xC0u, 0x80000000u)]
    [InlineData("VFrexpMantF32", 0x00400000u, 0u, 0xC0u, 0x00000000u)]
    [InlineData("VFrexpMantF32", 0x00400000u, 0u, 0xF0u, 0x3F000000u)]
    [InlineData("VFrexpExpI32F32", 0x00400000u, 0u, 0xC0u, 0x00000000u)]
    [InlineData("VFrexpExpI32F32", 0x00400000u, 0u, 0xF0u, 0xFFFFFF82u)]
    public void BitExactPathsFlushF32InputDenormalsPerFloatMode(string opcode, uint a, uint b, uint floatMode, uint expected)
    {
        var shader = Compile([Vop3(opcode, [Gen5Operand.Vector(0), Gen5Operand.Vector(1)])], a, b, floatMode: floatMode);
        if (!RequireDevice()) return;
        Assert.Equal(expected, Execute(shader));
    }

    // v3 = VCC ? 1 : 0
    private static Gen5ShaderInstruction CndmaskVcc() =>
        new(28, Gen5ShaderEncoding.Vop2, "VCndmaskB32", [], [Gen5Operand.Source(128), Gen5Operand.Source(129)], [Gen5Operand.Vector(3)], null);

    private static Gen5ShaderInstruction Vop3(string opcode, Gen5Operand[] sources, Gen5InstructionControl? control = null) =>
        new(28, Gen5ShaderEncoding.Vop3, opcode, [], sources, [Gen5Operand.Vector(3)], control ?? new Gen5Vop3Control(0, 0, 0, false, 0, null));

    private static readonly Gen5Operand[] V012 = [Gen5Operand.Vector(0), Gen5Operand.Vector(1), Gen5Operand.Vector(2)];

    private bool RequireDevice()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Numerical execution requires the exclusive Metal device gate.");
            return false;
        }

        Assert.True(MetalNative.IsAvailable, "The required execution gate needs a Metal device.");
        return true;
    }

    private static uint Execute(Gen5MslShader shader)
    {
        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var error), error + shader.Source);
        var buffer = new byte[16];
        buffer.AsSpan().Fill(0xCD);
        Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, buffer,
            new byte[PushData.ByteSize], shader.ArgumentLayout!, 1, out var result, out error), error);
        return BinaryPrimitives.ReadUInt32LittleEndian(result);
    }

    private static Gen5MslShader Compile(Gen5ShaderInstruction[] inserted, uint a, uint b, uint c = 0, uint old = 0,
        bool ieee = false, uint floatMode = ShaderCompileRequest.DefaultFloatMode, bool dx10Clamp = true)
    {
        Assert.True(TryCompile(inserted, a, b, c, old, out var shader, out var error, ieee, floatMode, dx10Clamp), error);
        return shader!;
    }

    // v0 = a, v1 = b, v2 = c, v3 = old; the inserted instructions run before v3 is stored.
    private static bool TryCompile(Gen5ShaderInstruction[] inserted, uint a, uint b, uint c, uint old,
        out Gen5MslShader? shader, out string error, bool ieee = false,
        uint floatMode = ShaderCompileRequest.DefaultFloatMode, bool dx10Clamp = true,
        Gen5MslCodegenOptions? options = null)
    {
        var fixture = new Gen5ComputeFixture("float-semantics", [
            0x7E0002FF, a, 0x7E0202FF, b, 0x7E0402FF, c, 0x7E0602FF, old,
            0xE0700000, 0x80020300, 0xBF810000], 8, 16);
        var program = Gen5ComputeFixtures.DecodeOrThrow(fixture);
        var instructions = program.Instructions.ToList();
        instructions.InsertRange(4, inserted);
        var request = Gen5ComputeFixtures.RequestOrThrow(program with { Instructions = instructions }, ShaderStage.Compute, ieeeMode: ieee,
            floatMode: floatMode, dx10Clamp: dx10Clamp);
        var compiled = Gen5MslTranslator.TryCompileProgram(request, out var result, out error, options);
        shader = compiled ? result : null;
        return compiled;
    }
}
