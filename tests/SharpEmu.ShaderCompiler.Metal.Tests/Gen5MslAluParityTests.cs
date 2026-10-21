// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>Numerical fixtures based on RDNA2 ISA sections 12.7, 12.8 and 12.10.
/// Translation alone is checked everywhere; execution must be requested with the
/// exclusive device gate, and compares returned bits rather than generated strings.</summary>
public sealed class Gen5MslAluParityTests(ITestOutputHelper output)
{
    public static IEnumerable<object[]> Cases => new List<object[]>()
    {
        // opcode, source0/1/2, old destination, op_sel, op_sel_hi, neg, abs/high-neg, clamp, expected
        new object[] { "VFmaMixF32", 0xBE003E00u, 0x4080C000u, 0x41200000u, 0u, 0u, 3u, 0u, 0u, false, 0x40E00000u }, // 1.5 * -2 + 10 = 7
        new object[] { "VFmaMixF32", 0xBE003E00u, 0x40003C00u, 0x41200000u, 0u, 3u, 3u, 0u, 0u, false, 0x40E00000u }, // selected high halves: -1.5*2+10 = 7
        new object[] { "VFmaMixloF16", 0x00003E00u, 0x00004000u, 0x00003C00u, 0xABCD1234u, 0u, 7u, 0u, 0u, false, 0xABCD4400u }, // 1.5*2+1 = 4, preserve high
        new object[] { "VFmaMixhiF16", 0x00003E00u, 0x00004000u, 0x00003C00u, 0xABCD1234u, 0u, 7u, 0u, 0u, false, 0x44001234u },
        new object[] { "VFmaMixF32", 0x0000BE00u, 0x00004000u, 0x3F800000u, 0u, 0u, 3u, 1u, 1u, false, 0xC0000000u }, // abs then negate: -1.5*2+1 = -2
        new object[] { "VFmaMixloF16", 0x7FC00000u, 0x3F800000u, 0u, 0xABCDFFFFu, 0u, 0u, 0u, 0u, true, 0xABCD0000u }, // clamp NaN -> 0
        new object[] { "VPkFmaF16", 0x3E003E00u, 0x3C013C01u, 0x80018001u, 0u, 0u, 7u, 0u, 0u, false, 0x3E013E01u }, // exact result just below midpoint; float fma + half incorrectly gives 3E02
        new object[] { "VPkFmacF16", 0x3E003E00u, 0x3C013C01u, 0u, 0x80018001u, 0u, 0u, 0u, 0u, false, 0x3E013E01u },
        new object[] { "VPkAddF16", 0x40003C00u, 0x44004200u, 0u, 0u, 0u, 3u, 0u, 0u, false, 0x46004400u }, // (1+3, 2+4)
        new object[] { "VPkMulF16", 0x40003C00u, 0x44004200u, 0u, 0u, 0u, 3u, 0u, 0u, false, 0x48004200u }, // (1*3, 2*4)
        new object[] { "VPkAddI16", 0x7FFFFFFFu, 0x00010002u, 0u, 0u, 0u, 3u, 0u, 0u, false, 0x80000001u },
        new object[] { "VPkSubI16", 0x00000000u, 0x00010001u, 0u, 0u, 0u, 3u, 0u, 0u, false, 0xFFFFFFFFu },
        new object[] { "VDot2cF32F16", 0x40003C00u, 0x44004200u, 0u, 0x40A00000u, 0u, 0u, 0u, 0u, false, 0x41800000u }, // 1*3+2*4+5 = 16 (no clamp)
        new object[] { "VFfbhU32", 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, false, 0xFFFFFFFFu },
        new object[] { "VFfbhU32", 0x100000FFu, 0u, 0u, 0u, 0u, 0u, 0u, 0u, false, 3u },
        new object[] { "VFfbhI32", 0xFFFFFFFEu, 0u, 0u, 0u, 0u, 0u, 0u, 0u, false, 31u },
        new object[] { "VFfbhI32", 0xFFFFFFFFu, 0u, 0u, 0u, 0u, 0u, 0u, 0u, false, 0xFFFFFFFFu },
        new object[] { "VBfeI32", 0x000000F0u, 4u, 4u, 0u, 0u, 0u, 0u, 0u, false, 0xFFFFFFFFu },
        new object[] { "VMadI32I24", 0x00FFFFFFu, 2u, 3u, 0u, 0u, 0u, 0u, 0u, false, 1u },
        new object[] { "VAddNcU16", 0xFFFFu, 2u, 0u, 0xABCD5678u, 0u, 0u, 0u, 0u, false, 0xABCD0001u },
        new object[] { "VAshrrevI16", 4u, 0x8000u, 0u, 0xABCD5678u, 0u, 0u, 0u, 0u, false, 0xABCDF800u },
        new object[] { "VMaxI16", 0xFFFFu, 2u, 0u, 0xABCD5678u, 0u, 0u, 0u, 0u, false, 0xABCD0002u },
        // A NaN operand makes V_MED3 answer MIN3, whose min skips the NaN: the clamp
        // med3(65504, x, 0) flushes a NaN x to 0 rather than to 65504.
        new object[] { "VMed3F32", 0x477FE000u, 0x7FC00000u, 0u, 0u, 0u, 0u, 0u, 0u, false, 0u },
        new object[] { "VMed3F32", 0x7FC00000u, 0x40E00000u, 0x40A00000u, 0u, 0u, 0u, 0u, 0u, false, 0x40A00000u }, // min3(NaN, 7, 5) = 5
        new object[] { "VMed3F32", 0x477FE000u, 0x40000000u, 0u, 0u, 0u, 0u, 0u, 0u, false, 0x40000000u }, // med3(65504, 2, 0) = 2
        new object[] { "VMed3F16", 0x7E00u, 0x4700u, 0x4500u, 0xABCD5678u, 0u, 0u, 0u, 0u, false, 0xABCD4500u }, // min3(NaN, 7, 5) = 5

        // Integer clamp saturates (ISA 6.5): unsigned to 0..0xFFFFFFFF, signed to INT_MIN..INT_MAX.
        new object[] { "VAddU32", 0xFFFFFFF0u, 0x20u, 0u, 0u, 0u, 0u, 0u, 0u, true, 0xFFFFFFFFu },
        new object[] { "VAddU32", 0xFFFFFFF0u, 0x20u, 0u, 0u, 0u, 0u, 0u, 0u, false, 0x10u },
        new object[] { "VSubU32", 1u, 2u, 0u, 0u, 0u, 0u, 0u, 0u, true, 0u },
        new object[] { "VSubU32", 1u, 2u, 0u, 0u, 0u, 0u, 0u, 0u, false, 0xFFFFFFFFu },
        new object[] { "VSubrevU32", 2u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, true, 0u },
        new object[] { "VAddI32", 0x7FFFFFFFu, 1u, 0u, 0u, 0u, 0u, 0u, 0u, true, 0x7FFFFFFFu },
        new object[] { "VAddI32", 0x80000000u, 0xFFFFFFFFu, 0u, 0u, 0u, 0u, 0u, 0u, true, 0x80000000u },
        new object[] { "VSubI32", 0x80000000u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, true, 0x80000000u },
        // VSubrevI32 is VOP2 0x27, V_SUBREV_NC_U32 (gfx10 has no signed subrev): unsigned saturation.
        new object[] { "VSubrevI32", 1u, 0x80000000u, 0u, 0u, 0u, 0u, 0u, 0u, true, 0x7FFFFFFFu },
        new object[] { "VSubrevI32", 0xFFFFFFFFu, 0x7FFFFFFFu, 0u, 0u, 0u, 0u, 0u, 0u, true, 0u },
        new object[] { "VAddCoU32", 0xFFFFFFF0u, 0x20u, 0u, 0u, 0u, 0u, 0u, 0u, true, 0xFFFFFFFFu },
        new object[] { "VAddCoU32", 0xFFFFFFF0u, 0x20u, 0u, 0u, 0u, 0u, 0u, 0u, false, 0x10u },
        new object[] { "VSubCoU32", 3u, 5u, 0u, 0u, 0u, 0u, 0u, 0u, true, 0u },
        new object[] { "VSubCoU32", 3u, 5u, 0u, 0u, 0u, 0u, 0u, 0u, false, 0xFFFFFFFEu },
        new object[] { "VSubrevCoU32", 5u, 3u, 0u, 0u, 0u, 0u, 0u, 0u, true, 0u },
        // V_MUL_U32_U24 and V_MAD_U32_U24 read only bits 23:0 of each multiplicand.
        new object[] { "VMulU32U24", 0x01000003u, 2u, 0u, 0u, 0u, 0u, 0u, 0u, false, 6u },
        new object[] { "VMulU32U24", 0xFF000003u, 5u, 0u, 0u, 0u, 0u, 0u, 0u, false, 0xFu },
        new object[] { "VMadU32U24", 0x01000003u, 2u, 1u, 0u, 0u, 0u, 0u, 0u, false, 7u },
        // V_MAD_U32_U16 multiplies the op_sel-selected halves and adds the whole addend.
        new object[] { "VMadU32U16", 0x00030005u, 4u, 10u, 0u, 1u, 0u, 0u, 0u, false, 22u }, // op_sel:[1,0]: 3*4+10
        new object[] { "VMadU32U16", 0x00030005u, 0x00020000u, 10u, 0u, 3u, 0u, 0u, 0u, false, 16u }, // op_sel:[1,1]: 3*2+10
        new object[] { "VMadU32U16", 0x00030005u, 0x00040000u, 0x10000u, 0u, 2u, 0u, 0u, 0u, false, 0x10014u }, // op_sel:[0,1]: 5*4+0x10000
        new object[] { "VMadU32U16", 0xFFFFu, 0xFFFFu, 0xFFFFFFFFu, 0u, 0u, 0u, 0u, 0u, false, 0xFFFE0000u },
        new object[] { "VMadU32U16", 0xFFFFu, 0xFFFFu, 0xFFFFFFFFu, 0u, 0u, 0u, 0u, 0u, true, 0xFFFFFFFFu },
        // V_PK_ADD/SUB_I16 clamp saturates each lane to the signed 16-bit range.
        new object[] { "VPkAddI16", 0x7FFF7FFFu, 0x00010001u, 0u, 0u, 0u, 3u, 0u, 0u, true, 0x7FFF7FFFu },
        new object[] { "VPkSubI16", 0x80008000u, 0x00010001u, 0u, 0u, 0u, 3u, 0u, 0u, true, 0x80008000u },
        new object[] { "VPkAddI16", 0x80007FFFu, 0xFFFF0001u, 0u, 0u, 0u, 3u, 0u, 0u, true, 0x80007FFFu }, // lo 0x7FFF+1 saturates; hi -32768-1 saturates
        // Float to integer conversions saturate and map NaN to 0 (ISA 12.8).
        new object[] { "VCvtRpiI32F32", Bits(0.3f), 0u, 0u, 0u, 0u, 0u, 0u, 0u, false, 0u },
        new object[] { "VCvtRpiI32F32", Bits(2.2f), 0u, 0u, 0u, 0u, 0u, 0u, 0u, false, 2u },
        new object[] { "VCvtRpiI32F32", Bits(-1.7f), 0u, 0u, 0u, 0u, 0u, 0u, 0u, false, 0xFFFFFFFEu },
        new object[] { "VCvtRpiI32F32", Bits(0.5f), 0u, 0u, 0u, 0u, 0u, 0u, 0u, false, 1u },
        new object[] { "VCvtRpiI32F32", Bits(float.NaN), 0u, 0u, 0u, 0u, 0u, 0u, 0u, false, 0u },
        new object[] { "VCvtRpiI32F32", Bits(float.PositiveInfinity), 0u, 0u, 0u, 0u, 0u, 0u, 0u, false, 0x7FFFFFFFu },
        new object[] { "VCvtFlrI32F32", Bits(-0.5f), 0u, 0u, 0u, 0u, 0u, 0u, 0u, false, 0xFFFFFFFFu },
        new object[] { "VCvtFlrI32F32", Bits(float.NegativeInfinity), 0u, 0u, 0u, 0u, 0u, 0u, 0u, false, 0x80000000u },
        new object[] { "VCvtU32F32", Bits(-1.0f), 0u, 0u, 0u, 0u, 0u, 0u, 0u, false, 0u },
        new object[] { "VCvtU32F32", Bits(float.NaN), 0u, 0u, 0u, 0u, 0u, 0u, 0u, false, 0u },
        new object[] { "VCvtU32F32", Bits(float.PositiveInfinity), 0u, 0u, 0u, 0u, 0u, 0u, 0u, false, 0xFFFFFFFFu },
        new object[] { "VCvtU32F32", Bits(4294967296f), 0u, 0u, 0u, 0u, 0u, 0u, 0u, false, 0xFFFFFFFFu },
        new object[] { "VCvtU32F32", Bits(4294967040f), 0u, 0u, 0u, 0u, 0u, 0u, 0u, false, 0xFFFFFF00u },
        new object[] { "VCvtU32F32", Bits(3.9f), 0u, 0u, 0u, 0u, 0u, 0u, 0u, false, 3u },
        new object[] { "VCvtI32F32", Bits(3.0e9f), 0u, 0u, 0u, 0u, 0u, 0u, 0u, false, 0x7FFFFFFFu },
        new object[] { "VCvtI32F32", Bits(-3.0e9f), 0u, 0u, 0u, 0u, 0u, 0u, 0u, false, 0x80000000u },
        new object[] { "VCvtI32F32", Bits(float.NaN), 0u, 0u, 0u, 0u, 0u, 0u, 0u, false, 0u },
        new object[] { "VCvtI32F32", Bits(float.NegativeInfinity), 0u, 0u, 0u, 0u, 0u, 0u, 0u, false, 0x80000000u },
        new object[] { "VCvtI32F32", Bits(-2.9f), 0u, 0u, 0u, 0u, 0u, 0u, 0u, false, 0xFFFFFFFEu },
        // V_CVT_PK_U8_F32 saturates to 0..255, NaN is 0, and the byte lands at S1[1:0].
        new object[] { "VCvtPkU8F32", Bits(300f), 0u, 0u, 0u, 0u, 0u, 0u, 0u, false, 0xFFu },
        new object[] { "VCvtPkU8F32", Bits(-5f), 0u, 0u, 0u, 0u, 0u, 0u, 0u, false, 0u },
        new object[] { "VCvtPkU8F32", Bits(float.NaN), 0u, 0u, 0u, 0u, 0u, 0u, 0u, false, 0u },
        new object[] { "VCvtPkU8F32", Bits(float.PositiveInfinity), 1u, 0u, 0u, 0u, 0u, 0u, 0u, false, 0xFF00u },
        new object[] { "VCvtPkU8F32", Bits(3.7f), 1u, 0xAABBCCDDu, 0u, 0u, 0u, 0u, 0u, false, 0xAABB04DDu },
        // V_BFE: offset + width may pass 32, a zero width gives 0 (ISA 12.12).
        new object[] { "VBfeU32", 0xFFFFFFFFu, 28u, 8u, 0u, 0u, 0u, 0u, 0u, false, 0xFu },
        new object[] { "VBfeU32", 0xFFFFFFFFu, 31u, 31u, 0u, 0u, 0u, 0u, 0u, false, 1u },
        new object[] { "VBfeU32", 0xFFFFFFFFu, 4u, 0u, 0u, 0u, 0u, 0u, 0u, false, 0u },
        new object[] { "VBfeI32", 0xF0000000u, 28u, 8u, 0u, 0u, 0u, 0u, 0u, false, 0xFFFFFFFFu },
        new object[] { "VBfeI32", 0x70000000u, 28u, 8u, 0u, 0u, 0u, 0u, 0u, false, 7u },
        new object[] { "VBfeI32", 0xFFFFFFFFu, 4u, 0u, 0u, 0u, 0u, 0u, 0u, false, 0u },
    };

    public static IEnumerable<object[]> TranslationCases => Cases.Select(static row => row[..^1]);

    [Theory]
    [MemberData(nameof(TranslationCases))]
    public void EveryFixtureTranslates(string opcode, uint a, uint b, uint c, uint old,
        uint select, uint high, uint neg, uint abs, bool clamp)
    {
        var shader = Compile(opcode, a, b, c, old, select, high, neg, abs, clamp);
        Assert.NotEmpty(shader.Source);
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void EveryFixtureExecutesOnMetal(string opcode, uint a, uint b, uint c, uint old,
        uint select, uint high, uint neg, uint abs, bool clamp, uint expected)
    {
        if (!RequireDevice()) return;
        Assert.Equal(expected, Execute(Compile(opcode, a, b, c, old, select, high, neg, abs, clamp)));
    }

    // op_sel_hi marks a V_FMA_MIX operand as f16, literals included: 0x34CD is 0.3, not an
    // f32 denormal. Silent Hill weights a luminance with such literals and divides by it.
    [Theory]
    [InlineData(0x34CDu, 1u, 0x40000000u)]  // half-selected literal: f16 0.3 * 2
    [InlineData(0x3E800000u, 0u, 0x40000000u)] // f32 literal: 0.25 * 2
    public void FusedMixReadsHalfSelectedLiteralsAsHalfFloats(uint literal, uint high, uint factor)
    {
        var shader = Compile("VFmaMixF32", 0u, factor, 0u, 0u, 0u, high, 0u, 0u, false,
            new Gen5Operand(Gen5OperandKind.LiteralConstant, literal));
        if (!RequireDevice()) return;
        var weight = high != 0 ? (float)BitConverter.UInt16BitsToHalf((ushort)literal) : BitConverter.UInt32BitsToSingle(literal);
        Assert.Equal(
            BitConverter.SingleToUInt32Bits(MathF.FusedMultiplyAdd(weight, BitConverter.UInt32BitsToSingle(factor), 0f)),
            Execute(shader));
    }

    // V_CVT_PKRTZ_F16_F32 rounds toward zero: finite values past the f16 range saturate at
    // +-65504 instead of becoming infinity, and subnormal results keep whole 2^-24 steps.
    [Theory]
    [InlineData(5003.7f, -80000.3f)]
    [InlineData(70000f, -float.MaxValue)]
    [InlineData(65519.99f, -65504f)]
    [InlineData(3.3e-6f, -5e-5f)]
    [InlineData(5.9e-5f, -6.104e-5f)] // 989.86 steps of 2^-24: nearest would give 990
    [InlineData(1e-9f, -0f)]
    [InlineData(1.00097656f, -2.99f)]
    public void PackToHalfRoundsTowardZeroWithoutOverflowingToInfinity(float low, float high)
    {
        var shader = Compile("VCvtPkrtzF16F32", Bits(low), Bits(high), 0u, 0u, 0u, 0u, 0u, 0u, false);
        if (!RequireDevice()) return;
        Assert.Equal(HalfTowardZero(low) | HalfTowardZero(high) << 16, Execute(shader));
    }

    [Fact]
    public void PackToHalfKeepsNaNAndInfinity()
    {
        var nan = Compile("VCvtPkrtzF16F32", Bits(float.NaN), Bits(float.PositiveInfinity), 0u, 0u, 0u, 0u, 0u, 0u, false);
        var negative = Compile("VCvtPkrtzF16F32", Bits(float.NegativeInfinity), 0u, 0u, 0u, 0u, 0u, 0u, 0u, false);
        if (!RequireDevice()) return;
        var special = Execute(nan);
        Assert.True((special & 0x7C00) == 0x7C00 && (special & 0x3FF) != 0, $"0x{special:X8}");
        Assert.Equal(0x7C00u, special >> 16);
        Assert.Equal(0xFC00u, Execute(negative));
    }

    // V_CVT_F32_F16 reads a half, so its negate flips bit 15 of the selected half. Flipping
    // bit 31 of the register instead left the value positive: Silent Hill's motion blur
    // negates one velocity component this way and blurred the wrong way along x.
    [Theory]
    [InlineData(0x12343E00u, 6u, 1u, 0u, 0xBFC00000u)] // SDWA dword, negate: -1.5
    [InlineData(0x3E001234u, 5u, 1u, 0u, 0xBFC00000u)] // SDWA high word, negate: -1.5
    [InlineData(0x1234BE00u, 6u, 0u, 1u, 0x3FC00000u)] // SDWA dword, abs: 1.5
    public void HalfToFloatAppliesSdwaModifiersToTheHalf(uint source, uint select, uint neg, uint abs, uint expected)
    {
        var shader = Compile("VCvtF32F16", source, 0u, 0u, 0u, 0u, 0u, 0u, 0u, false,
            control: new Gen5SdwaControl(6, 0, select, 6, false, false, abs, neg, 0, false, null));
        Assert.DoesNotContain("^ 0x80000000u", shader.Source, StringComparison.Ordinal);
        if (!RequireDevice()) return;
        Assert.Equal(expected, Execute(shader));
    }

    // An SDWA V_CNDMASK_B32 writes only the selected word of its destination. GTA V's
    // upscale shader assembles packed f16 colour this way; writing the whole register put
    // green into the red half and cleared green, so the frame came out purple.
    [Theory]
    [InlineData(5u, 2u, 0x40001234u)] // word 1 written, word 0 preserved
    [InlineData(4u, 2u, 0xABCD4000u)] // word 0 written, word 1 preserved
    [InlineData(5u, 0u, 0x40000000u)] // word 1 written, the rest zeroed
    public void CndmaskWritesOnlyTheSdwaDestinationWord(uint destinationSelect, uint unused, uint expected)
    {
        // VCC is clear, so source 0's high word (0x4000) is the selected value.
        var fixture = new Gen5ComputeFixture("VCndmaskB32", [
            0x7E0002FF, 0x40003C00u, 0x7E0202FF, 0x00004400u, 0x7E0402FF, 0u, 0x7E0602FF, 0xABCD1234u,
            0xE0700000, 0x80020300, 0xBF810000], 8, 16);
        var program = Gen5ComputeFixtures.DecodeOrThrow(fixture);
        var instructions = program.Instructions.ToList();
        instructions.Insert(4, new Gen5ShaderInstruction(28, Gen5ShaderEncoding.Vop2, "VCndmaskB32",
            [], [Gen5Operand.Vector(0), Gen5Operand.Vector(1)], [Gen5Operand.Vector(3)],
            new Gen5SdwaControl(destinationSelect, unused, 5, 5, false, false, 0, 0, 0, false, null)));
        var request = Gen5ComputeFixtures.RequestOrThrow(program with { Instructions = instructions }, ShaderStage.Compute);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        if (!RequireDevice()) return;
        Assert.Equal(expected, Execute(shader));
    }

    // A 16-bit result under SDWA is the low word of a DWORD selection, and DST_UNUSED decides
    // the high word; a plain VOP1 write keeps it (GFX10). Black Myth: Wukong's BC6H encoder
    // takes V_CVT_F16_F32 into a register with UNUSED_PAD and reads it whole as an integer
    // next: keeping the high word made every block of a black card the same red garbage.
    [Theory]
    [InlineData(true, 6u, 0u, 0x00003E00u)] // DWORD, unused padded: the high word is zero
    [InlineData(true, 6u, 2u, 0xABCD3E00u)] // DWORD, unused preserved
    [InlineData(true, 4u, 2u, 0xABCD3E00u)] // word 0 written, word 1 preserved
    [InlineData(true, 5u, 2u, 0x3E001234u)] // word 1 written, word 0 preserved
    [InlineData(true, 5u, 0u, 0x3E000000u)] // word 1 written, the rest zeroed
    [InlineData(false, 0u, 0u, 0xABCD3E00u)] // plain VOP1: the high word is kept
    public void HalfConversionHonoursTheSdwaDestination(bool sdwa, uint destinationSelect, uint unused, uint expected)
    {
        // v1 = 1.5f; v3 starts as 0xABCD1234 and takes the converted half (0x3E00).
        var fixture = new Gen5ComputeFixture("VCvtF16F32", [
            0x7E0002FF, 0u, 0x7E0202FF, 0x3FC00000u, 0x7E0402FF, 0u, 0x7E0602FF, 0xABCD1234u,
            0xE0700000, 0x80020300, 0xBF810000], 8, 16);
        var program = Gen5ComputeFixtures.DecodeOrThrow(fixture);
        var instructions = program.Instructions.ToList();
        instructions.Insert(4, new Gen5ShaderInstruction(28, Gen5ShaderEncoding.Vop1, "VCvtF16F32",
            [], [Gen5Operand.Vector(1)], [Gen5Operand.Vector(3)],
            sdwa ? new Gen5SdwaControl(destinationSelect, unused, 6, 6, false, false, 0, 0, 0, false, null) : null));
        var request = Gen5ComputeFixtures.RequestOrThrow(program with { Instructions = instructions }, ShaderStage.Compute);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        if (!RequireDevice()) return;
        Assert.Equal(expected, Execute(shader));
    }

    [Theory]
    [InlineData(0x40001234u, 1u, 0u, 0x40000000u)] // op_sel picks the high half: 2.0
    [InlineData(0x12344000u, 0u, 1u, 0xC0000000u)] // VOP3 negate: -2.0
    public void HalfToFloatAppliesVop3SelectAndNegate(uint source, uint select, uint neg, uint expected)
    {
        var shader = Compile("VCvtF32F16", source, 0u, 0u, 0u, select, 0u, neg, 0u, false);
        if (!RequireDevice()) return;
        Assert.Equal(expected, Execute(shader));
    }

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
        Assert.Equal(buffer.AsSpan(4).ToArray(), result.AsSpan(4).ToArray());
        return BinaryPrimitives.ReadUInt32LittleEndian(result);
    }

    private static uint HalfTowardZero(float value)
    {
        var sign = value < 0 || (value == 0 && float.IsNegative(value)) ? 0x8000u : 0u;
        var magnitude = MathF.Abs(value);
        if (magnitude >= 65504f) return sign | 0x7BFF;
        if (magnitude < 6.10351562e-05f) return sign | (uint)Math.Floor(magnitude * 16777216.0);
        var bits = BitConverter.SingleToUInt32Bits(magnitude);
        return sign | ((((bits >> 23) - 127 + 15) << 10) | ((bits >> 13) & 0x3FF));
    }

    private static uint Bits(float value) => BitConverter.SingleToUInt32Bits(value);

    [Theory]
    [InlineData(0xCC201103u, 0x9C0A0300u, "VFmaMixF32")]
    [InlineData(0xCC214003u, 0x1C0A0300u, "VFmaMixloF16")]
    public void RealInstructionEncodingsReachTheLowering(uint first, uint second, string expected)
    {
        var fixture = new Gen5ComputeFixture(expected, [first, second, 0xBF810000], 0, 0);
        Assert.Equal(expected, Gen5ComputeFixtures.DecodeOrThrow(fixture).Instructions[0].Opcode);
        Assert.NotEmpty(Gen5ComputeFixtures.CompileRequestOrThrow(fixture).Source);
    }

    // The carry-in forms read VCC (clear here) with two explicit sources. Clamp saturates
    // the sum or difference; the carry out itself is unaffected.
    [Theory]
    [InlineData("VAddcU32", 0xFFFFFFFFu, 1u, true, 0xFFFFFFFFu)]
    [InlineData("VAddcU32", 0xFFFFFFFFu, 1u, false, 0u)]
    [InlineData("VAddCoCiU32", 0xFFFFFFF0u, 0x20u, true, 0xFFFFFFFFu)]
    [InlineData("VSubbU32", 0u, 1u, true, 0u)]
    [InlineData("VSubbU32", 0u, 1u, false, 0xFFFFFFFFu)]
    [InlineData("VSubbrevU32", 1u, 0u, true, 0u)]
    [InlineData("VSubCoCiU32", 3u, 5u, true, 0u)]
    [InlineData("VSubrevCoCiU32", 5u, 3u, true, 0u)]
    public void CarryFormsSaturateWithClamp(string opcode, uint a, uint b, bool clamp, uint expected)
    {
        var shader = CompileInserted(
            out var error,
            Vector3(opcode, [Gen5Operand.Vector(0), Gen5Operand.Vector(1)],
                new Gen5Vop3Control(0, 0, 0, clamp, 0, null)),
            a, b);
        Assert.True(shader is not null, error);
        if (!RequireDevice()) return;
        Assert.Equal(expected, Execute(shader!));
    }

    // Integer ops the clamp bit applies to but that are not implemented fail loudly instead
    // of wrapping silently, like the 16-bit forms.
    [Theory]
    [InlineData("VMulU32U24")]
    [InlineData("VMulI32I24")]
    [InlineData("VMadU32U24")]
    [InlineData("VMadI32I24")]
    [InlineData("VSadU32")]
    public void UnimplementedIntegerClampIsRejected(string opcode)
    {
        var shader = CompileInserted(
            out var error,
            Vector3(opcode, [Gen5Operand.Vector(0), Gen5Operand.Vector(1), Gen5Operand.Vector(2)],
                new Gen5Vop3Control(0, 0, 0, true, 0, null)),
            1u, 2u);
        Assert.Null(shader);
        Assert.Contains("clamped integer results are not supported", error, StringComparison.Ordinal);
    }

    // S_LSHLn_ADD_U32 sets SCC to the carry out of the 64-bit sum, which a following
    // S_CSELECT_B32 observes (7 when set, else 9). SCC is first put in the opposite state
    // by an S_CMP_EQ_U32, so a stale SCC gives the wrong answer either way.
    [Theory]
    [InlineData("SLshl2AddU32", 0x80000000u, 5u, false, 7u)]  // 2^33 + 5 carries
    [InlineData("SLshl1AddU32", 1u, 1u, true, 9u)]            // 3 does not
    [InlineData("SLshl4AddU32", 0x10000000u, 0xFu, false, 7u)] // 2^32 + 15 carries
    [InlineData("SLshl3AddU32", 0x1FFFFFFFu, 0xFFFFFFFFu, false, 7u)] // 0xFFFFFFF8 + 0xFFFFFFFF carries
    [InlineData("SLshl3AddU32", 0x1FFFFFFFu, 7u, true, 9u)]   // 0xFFFFFFF8 + 7 = 0xFFFFFFFF fits
    public void ShiftAddWritesTheCarryToScc(string opcode, uint left, uint right, bool sccBefore, uint expected)
    {
        var shader = CompileInserted(
            out var error,
            [
                new Gen5ShaderInstruction(0, Gen5ShaderEncoding.Sopc, "SCmpEqU32", [0u],
                    [Gen5Operand.Source(128), Gen5Operand.Source(sccBefore ? 128u : 129u)], [], null),
                ScalarOp(opcode,
                    [new Gen5Operand(Gen5OperandKind.LiteralConstant, left), new Gen5Operand(Gen5OperandKind.LiteralConstant, right)],
                    Gen5Operand.Scalar(10)),
                ScalarOp("SCselectB32", [Gen5Operand.Source(128 + 7), Gen5Operand.Source(128 + 9)], Gen5Operand.Scalar(11)),
                new Gen5ShaderInstruction(0, Gen5ShaderEncoding.Vop1, "VMovB32", [0u], [Gen5Operand.Scalar(11)], [Gen5Operand.Vector(3)], null),
            ],
            0u, 0u);
        Assert.True(shader is not null, error);
        if (!RequireDevice()) return;
        Assert.Equal(expected, Execute(shader!));
    }

    private static Gen5ShaderInstruction ScalarOp(string opcode, Gen5Operand[] sources, params Gen5Operand[] destinations) =>
        new(0, Gen5ShaderEncoding.Sop2, opcode, [0u], sources, destinations, null);

    private static Gen5ShaderInstruction Vector3(string opcode, Gen5Operand[] sources, Gen5InstructionControl control) =>
        new(28, Gen5ShaderEncoding.Vop3, opcode, [], sources, [Gen5Operand.Vector(3)], control);

    // v0 = a, v1 = b, v2 = 0; the inserted instructions run before the store of v3.
    private static Gen5MslShader? CompileInserted(out string error, Gen5ShaderInstruction inserted, uint a, uint b) =>
        CompileInserted(out error, [inserted], a, b);

    private static Gen5MslShader? CompileInserted(out string error, Gen5ShaderInstruction[] inserted, uint a, uint b)
    {
        var fixture = new Gen5ComputeFixture("inserted", [
            0x7E0002FF, a, 0x7E0202FF, b, 0x7E0402FF, 0u, 0x7E0602FF, 0u,
            0xE0700000, 0x80020300, 0xBF810000], 8, 16);
        var program = Gen5ComputeFixtures.DecodeOrThrow(fixture);
        var instructions = program.Instructions.ToList();
        instructions.InsertRange(4, inserted);
        var request = Gen5ComputeFixtures.RequestOrThrow(program with { Instructions = instructions }, ShaderStage.Compute);
        return Gen5MslTranslator.TryCompileProgram(request, out var shader, out error) ? shader : null;
    }

    private static Gen5MslShader Compile(string opcode, uint a, uint b, uint c, uint old,
        uint select, uint high, uint neg, uint abs, bool clamp, Gen5Operand? firstSource = null,
        Gen5InstructionControl? control = null)
    {
        // Actual decoded register initialization and store, with a direct IR opcode
        // insertion so each semantic case can exercise the control masks independently.
        var fixture = new Gen5ComputeFixture(opcode, [
            0x7E0002FF, a, 0x7E0202FF, b, 0x7E0402FF, c, 0x7E0602FF, old,
            0xE0700000, 0x80020300, 0xBF810000], 8, 16);
        var program = Gen5ComputeFixtures.DecodeOrThrow(fixture);
        var instructions = program.Instructions.ToList();
        control ??= opcode.StartsWith("VPk", StringComparison.Ordinal) && opcode != "VPkFmacF16" ||
            opcode.StartsWith("VFmaMix", StringComparison.Ordinal)
            ? new Gen5Vop3pControl(select, high, neg, abs, clamp)
            : new Gen5Vop3Control(abs, neg, 0, clamp, select, null);
        instructions.Insert(4, new Gen5ShaderInstruction(28, Gen5ShaderEncoding.Vop3, opcode,
            [], [firstSource ?? Gen5Operand.Vector(0), Gen5Operand.Vector(1), Gen5Operand.Vector(2)],
            [Gen5Operand.Vector(3)], control));
        program = program with { Instructions = instructions };
        var request = Gen5ComputeFixtures.RequestOrThrow(program, ShaderStage.Compute);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        return shader;
    }
}
