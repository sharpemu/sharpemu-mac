// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Numerics;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>
/// Integer, 16-bit float, packed, dot-product and scalar ALU ops (RDNA2 ISA chapters 5, 6,
/// 12). Each group is one straight-line compute program: every case loads its inputs with
/// v_mov/s_mov literals, runs the op and stores its result registers to consecutive dwords.
/// Expected values come from C# references of the ISA pseudocode. Translation is checked
/// everywhere; execution needs the exclusive Metal device gate.
/// </summary>
public sealed class Gen5MslIntegerAluGapTests(ITestOutputHelper output)
{
    private const uint Sentinel = 0xABCD1234;
    private const uint M0 = 124;
    private const uint ExecLo = 126;
    private const uint EndProgram = 0xBF810000;
    private const uint SccToS30 = 0x851E8081;      // s_cselect_b32 s30, 1, 0
    private const uint SetScc = 0xBF061F1F;        // s_cmp_eq_u32 s31, s31
    private const uint ClearScc = 0xBF071F1F;      // s_cmp_lg_u32 s31, s31
    private const uint RestoreExec = 0xBEFE03C1;   // s_mov_b32 exec_lo, -1
    private const uint ExecToS21 = 0xBE95037E;     // s_mov_b32 s21, exec_lo

    private sealed record Case(string Label, uint[] Words, uint[] Results, uint[] Expected, uint Mask = 0xFFFFFFFFu);

    // ---- encoding helpers ----

    private static uint[] V(uint register, uint value) => [0x7E0002FFu | (register << 17), value];

    private static uint[] S(uint register, uint value) => [0xBE8003FFu | (register << 16), value];

    private static uint VFromS(uint vector, uint scalar) => 0x7E000200u | (vector << 17) | scalar;

    /// <summary>v0.. = inputs, v8..v11 = sentinel.</summary>
    private static uint[] Inputs(params uint[] values)
    {
        var words = new List<uint>();
        for (var index = 0; index < values.Length; index++)
        {
            words.AddRange(V((uint)index, values[index]));
        }

        for (uint register = 8; register < 12; register++)
        {
            words.AddRange(V(register, Sentinel));
        }

        return [.. words];
    }

    private static Case Vector(string label, uint[] op, uint[] inputs, uint expected, uint mask = 0xFFFFFFFFu, uint result = 8) =>
        new(label, [.. Inputs(inputs), .. op], [result], [expected], mask);

    // ---- float helpers ----

    private static uint F(float value) => BitConverter.SingleToUInt32Bits(value);

    private static float Fv(uint bits) => BitConverter.UInt32BitsToSingle(bits);

    private static uint H(double value) => BitConverter.HalfToUInt16Bits((Half)value);

    private static double Hv(uint bits) => (double)BitConverter.UInt16BitsToHalf((ushort)bits);

    private static bool IsZero(uint bits) => (bits & 0x7FFFFFFFu) == 0;

    // ---- references ----

    // V_MUL/MAD/MAC_LEGACY_F32: 0 * anything is 0, and the MAD forms round the product before
    // the add (LLVM selects V_MAD_LEGACY_F32 for fadd(fmul_legacy)).
    private static uint LegacyMad(uint a, uint b, uint? c)
    {
        if (IsZero(a) || IsZero(b))
        {
            return F(0.0f + (c is { } addend ? Fv(addend) : 0.0f));
        }

        var product = Fv(a) * Fv(b);
        return F(c is { } value ? product + Fv(value) : product);
    }

    private static (uint Mantissa, uint Exponent) FrexpF32(uint bits)
    {
        if ((bits & 0x7F800000u) == 0x7F800000u || IsZero(bits))
        {
            return (bits, 0);
        }

        double value = Fv(bits);
        var exponent = Math.ILogB(value) + 1;
        return (F((float)Math.ScaleB(value, -exponent)), (uint)exponent);
    }

    private static (uint Mantissa, uint Exponent) FrexpF16(uint bits)
    {
        bits &= 0xFFFF;
        if ((bits & 0x7C00u) == 0x7C00u || (bits & 0x7FFFu) == 0)
        {
            return (bits, 0);
        }

        var value = Hv(bits);
        var exponent = Math.ILogB(value) + 1;
        return (H(Math.ScaleB(value, -exponent)), (uint)exponent & 0xFFFF);
    }

    private static uint CvtI16F16(uint bits)
    {
        var value = Hv(bits);
        if (double.IsNaN(value))
        {
            return 0;
        }

        return (uint)(ushort)(short)Math.Clamp(Math.Truncate(value), short.MinValue, short.MaxValue);
    }

    private static uint NormI16(uint bits)
    {
        var value = Hv(bits);
        return double.IsNaN(value)
            ? 0
            : (uint)(ushort)(short)Math.Round(Math.Clamp(value, -1.0, 1.0) * 32767.0, MidpointRounding.ToEven);
    }

    private static uint NormU16(uint bits)
    {
        var value = Hv(bits);
        return double.IsNaN(value)
            ? 0
            : (uint)Math.Round(Math.Clamp(value, 0.0, 1.0) * 65535.0, MidpointRounding.ToEven);
    }

    private static uint ClassF16(uint bits, uint mask)
    {
        bits &= 0xFFFF;
        var negative = (bits & 0x8000) != 0;
        var exponent = (bits >> 10) & 0x1F;
        var fraction = bits & 0x3FF;
        int kind;
        if (exponent == 0x1F)
        {
            kind = fraction == 0 ? (negative ? 2 : 9) : ((fraction & 0x200) != 0 ? 1 : 0);
        }
        else if (exponent == 0)
        {
            kind = fraction == 0 ? (negative ? 5 : 6) : (negative ? 4 : 7);
        }
        else
        {
            kind = negative ? 3 : 8;
        }

        return ((mask >> kind) & 1) != 0 ? 1u : 0u;
    }

    private static int Sext(uint value, int bits) => (int)(value << (32 - bits)) >> (32 - bits);

    private static uint Byte(uint value, int index) => (value >> (index * 8)) & 0xFF;

    private static uint ByteSad(uint window, uint reference, bool masked)
    {
        uint sum = 0;
        for (var index = 0; index < 4; index++)
        {
            var x = Byte(window, index);
            var y = Byte(reference, index);
            if (!masked || y != 0)
            {
                sum += x > y ? x - y : y - x;
            }
        }

        return sum;
    }

    private static uint Perm(uint s0, uint s1, uint selectors)
    {
        var bytes = ((ulong)s0 << 32) | s1;
        uint result = 0;
        for (var index = 0; index < 4; index++)
        {
            var selector = Byte(selectors, index);
            uint picked = selector switch
            {
                >= 13 => 0xFF,
                12 => 0,
                >= 8 => ((bytes >> (int)((selector - 8) * 16 + 15)) & 1) != 0 ? 0xFFu : 0u,
                _ => (uint)((bytes >> (int)(selector * 8)) & 0xFF),
            };
            result |= picked << (index * 8);
        }

        return result;
    }

    private static uint Lo(uint value) => value & 0xFFFF;

    private static uint Hi(uint value) => value >> 16;

    private static uint Pack(uint low, uint high) => (low & 0xFFFF) | ((high & 0xFFFF) << 16);

    private static uint SaturateI16(long value) => (uint)(ushort)(short)Math.Clamp(value, short.MinValue, short.MaxValue);

    private static uint SaturateU16(long value) => (uint)Math.Clamp(value, 0, ushort.MaxValue);

    private static uint SaturateI32(long value) => (uint)(int)Math.Clamp(value, int.MinValue, int.MaxValue);

    private static uint SaturateU32(long value) => (uint)Math.Clamp(value, 0, uint.MaxValue);

    private static long Median(long a, long b, long c) => Math.Max(Math.Min(a, b), Math.Min(Math.Max(a, b), c));

    // ---- groups ----

    [Fact]
    public void LegacyMultiplyAndMadMatchTheZeroRule()
    {
        uint[] mulLegacy = [0x0E100300];                 // v_mul_legacy_f32 v8, v0, v1
        uint[] mulLegacyNeg = [0xD5070008, 0x20020300];  // v_mul_legacy_f32 v8, -v0, v1
        uint[] mullit = [0xD5500008, 0x040A0300];        // v_mullit_f32 v8, v0, v1, v2
        uint[] madLegacy = [0xD5400008, 0x040A0300];     // v_mad_legacy_f32 v8, v0, v1, v2
        uint[] macLegacy = [0x0C040300];                 // v_mac_legacy_f32 v2, v0, v1
        var inf = F(float.PositiveInfinity);
        const uint nan = 0x7FC00000;
        (uint A, uint B, uint C)[] values =
        [
            (F(2), F(3), F(1)),
            (F(-2), F(3), F(-0.5f)),
            (0, inf, F(5)),
            (0x80000000, nan, F(-5)),
            (inf, 0x80000000, 0x80000000),
            (F(1.5f), F(-0.25f), F(0.125f)),
            (0x7F7FFFFF, F(2), 0),
            (0x3F800001, 0x3F7FFFFF, 0xBF800000), // the product rounds to 1: +0 (fused: 2^-24 - 2^-47)
        ];
        var cases = new List<Case>();
        foreach (var (a, b, c) in values)
        {
            var tag = $"{a:X8},{b:X8},{c:X8}";
            cases.Add(Vector($"mul_legacy {tag}", mulLegacy, [a, b], LegacyMad(a, b, null)));
            cases.Add(Vector($"mul_legacy -src0 {tag}", mulLegacyNeg, [a, b], LegacyMad(a ^ 0x80000000u, b, null)));
            cases.Add(Vector($"mullit {tag}", mullit, [a, b, c], LegacyMad(a, b, null)));
            cases.Add(Vector($"mad_legacy {tag}", madLegacy, [a, b, c], LegacyMad(a, b, c)));
            cases.Add(Vector($"mac_legacy {tag}", macLegacy, [a, b, c], LegacyMad(a, b, c), result: 2));
        }

        Run("legacy", cases);
    }

    [Fact]
    public void FrexpF32AndF16AreBitExact()
    {
        uint[] mantF32 = [0x7E108100];                     // v_frexp_mant_f32 v8, v0
        uint[] expF32 = [0x7E107F00];                      // v_frexp_exp_i32_f32 v8, v0
        uint[] mantF32NegAbs = [0xD5C00108, 0x20000100];   // v_frexp_mant_f32_e64 v8, -|v0|
        uint[] mantF32Clamp = [0xD5C08008, 0x00000100];    // v_frexp_mant_f32_e64 v8, v0 clamp
        uint[] mantF16 = [0x7E10B300];                     // v_frexp_mant_f16 v8, v0
        uint[] expF16 = [0x7E10B500];                      // v_frexp_exp_i16_f16 v8, v0
        uint[] mantF16E64 = [0xD5D90008, 0x00000100];      // v_frexp_mant_f16_e64 v8, v0
        var cases = new List<Case>();
        uint[] f32 =
        [
            F(8), F(-3), F(1), F(0.75f), 0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000,
            0x00000001, 0x00400000, 0x807FFFFF, 0x7F7FFFFF, F(1e-30f),
        ];
        foreach (var x in f32)
        {
            var (mantissa, exponent) = FrexpF32(x);
            cases.Add(Vector($"frexp_mant_f32 {x:X8}", mantF32, [x], mantissa));
            cases.Add(Vector($"frexp_exp_i32_f32 {x:X8}", expF32, [x], exponent));
        }

        foreach (var x in new[] { F(8), F(-3), F(0.75f), 0x00000001u })
        {
            cases.Add(Vector($"frexp_mant_f32 -|x| {x:X8}", mantF32NegAbs, [x], FrexpF32(x | 0x80000000u).Mantissa));
            var clamped = Math.Clamp(Fv(FrexpF32(x).Mantissa), 0f, 1f);
            cases.Add(Vector($"frexp_mant_f32 clamp {x:X8}", mantF32Clamp, [x], F(clamped)));
        }

        uint[] f16 = [0x4800, 0xC200, 0x3C00, 0x3A00, 0x0001, 0x0200, 0x83FF, 0x7BFF, 0x0000, 0x8000, 0x7C00, 0xFC00];
        foreach (var x in f16)
        {
            var (mantissa, exponent) = FrexpF16(x);
            var input = 0x5A5A0000u | x; // garbage high half must be ignored
            cases.Add(Vector($"frexp_mant_f16 {x:X4}", mantF16, [input], mantissa, 0xFFFF));
            cases.Add(Vector($"frexp_exp_i16_f16 {x:X4}", expF16, [input], exponent, 0xFFFF));
            cases.Add(Vector($"frexp_mant_f16_e64 {x:X4}", mantF16E64, [input], mantissa, 0xFFFF));
        }

        // FLOAT_MODE 0xF0 keeps f32 denormals, so a denormal input is normalized; the flushing
        // mode is covered by Gen5MslFloatSemanticsTests.
        Run("frexp", cases, floatMode: 0xF0);
    }

    [Fact]
    public void Half16ConversionsRoundAndSaturate()
    {
        uint[] f16FromU16 = [0x7E10A100];                // v_cvt_f16_u16 v8, v0
        uint[] f16FromI16 = [0x7E10A300];                // v_cvt_f16_i16 v8, v0
        uint[] f16FromU16E64 = [0xD5D00008, 0x00000100]; // v_cvt_f16_u16_e64 v8, v0
        uint[] i16FromF16 = [0x7E10A700];                // v_cvt_i16_f16 v8, v0
        uint[] normI16 = [0x7E10C700];                   // v_cvt_norm_i16_f16 v8, v0
        uint[] normU16 = [0x7E10C900];                   // v_cvt_norm_u16_f16 v8, v0
        uint[] satPk = [0x7E10C500];                     // v_sat_pk_u8_i16 v8, v0
        uint[] f16FromF32High = [0xD58A0008 | (8u << 11), 0x00000100]; // v_cvt_f16_f32_e64 v8, v0 op_sel:[0,1]
        var cases = new List<Case>();
        foreach (var x in new uint[] { 3, 0, 2049, 2051, 4097, 65504, 65519, 65520, 65535, 0x8000, 0xFFFD, 0x7FFF })
        {
            cases.Add(Vector($"cvt_f16_u16 {x}", f16FromU16, [0xDEAD0000u | x], H(x), 0xFFFF));
            cases.Add(Vector($"cvt_f16_u16_e64 {x}", f16FromU16E64, [0xDEAD0000u | x], H(x), 0xFFFF));
            cases.Add(Vector($"cvt_f16_i16 {x}", f16FromI16, [0xDEAD0000u | x], H((short)x), 0xFFFF));
        }

        uint[] halves =
        [
            0x4200, 0xC2CD, 0x3800, 0xB800, 0x3C00, 0xBC00, 0x4000, 0xC000, 0x3555, 0xB555, 0x0001, 0x8001,
            0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x7E00, 0x7C01, 0x77FF, 0x7800, 0xF800, 0x3BFF, 0x2E66,
        ];
        foreach (var x in halves)
        {
            var input = 0xBEEF0000u | x;
            cases.Add(Vector($"cvt_i16_f16 {x:X4}", i16FromF16, [input], CvtI16F16(x), 0xFFFF));
            cases.Add(Vector($"cvt_norm_i16_f16 {x:X4}", normI16, [input], NormI16(x), 0xFFFF));
            cases.Add(Vector($"cvt_norm_u16_f16 {x:X4}", normU16, [input], NormU16(x), 0xFFFF));
        }

        foreach (var x in new uint[] { 0x01FF0080, 0x8000FFFF, 0x00100020, 0x7FFF00FF, 0x00FF0100, 0xFF810000 })
        {
            var expected = (uint)Math.Clamp((int)(short)Lo(x), 0, 255) | ((uint)Math.Clamp((int)(short)Hi(x), 0, 255) << 8);
            cases.Add(Vector($"sat_pk_u8_i16 {x:X8}", satPk, [x], expected));
        }

        foreach (var x in new[] { 1.0f, -2.5f, 65520f, 1e-8f })
        {
            cases.Add(Vector($"cvt_f16_f32 dst-hi {x}", f16FromF32High, [F(x)], (H(x) << 16) | (Sentinel & 0xFFFF)));
        }

        Run("cvt16", cases);
    }

    [Fact]
    public void Half16ArithmeticMatchesReferences()
    {
        uint[] ldexp = [0x76100300];                     // v_ldexp_f16 v8, v0, v1
        uint[] fmac = [0x6C040300];                      // v_fmac_f16 v2, v0, v1
        uint[] cmpClass = [0x7D1E0300, 0xD5010008, 0x01A90280];  // v_cmp_class_f16 vcc, v0, v1; v_cndmask v8, 0, 1, vcc
        uint[] cmpClassNeg = [0xD48F0014, 0x20020300, 0xD5010008, 0x00510280]; // _e64 s20, -v0, v1; v_cndmask v8, 0, 1, s20
        var cases = new List<Case>();
        (uint Value, int Exponent)[] scales =
        [
            (0x3C00, 3), (0x3C00, -15), (0x3C00, 16), (0x3C00, -25), (0x3E00, -24), (0x3C00, 32767),
            (0x3C00, -32768), (0x7BFF, 1), (0x0001, 10), (0xC000, -3), (0x8000, 100), (0x7C00, -100),
            (0x3555, -14), (0x3555, -20),
        ];
        foreach (var (value, exponent) in scales)
        {
            var expected = H(Math.ScaleB(Hv(value), exponent));
            cases.Add(Vector($"ldexp_f16 {value:X4} {exponent}", ldexp, [0x12340000u | value, (uint)(ushort)(short)exponent], expected, 0xFFFF));
        }

        // High garbage in the exponent register is ignored (S1.i16).
        cases.Add(Vector("ldexp_f16 exp high garbage", ldexp, [0x3C00, 0xFFFF0003], H(8), 0xFFFF));

        (uint A, uint B, uint C)[] fmas =
        [
            (0x3E00, 0x3C01, 0x8001), // single rounding: 0x3E01 (double rounding gives 0x3E02)
            (0x4000, 0x4200, 0x3C00),
            (0x3C00, 0x3C00, 0xBC00),
            (0x7BFF, 0x4000, 0xFBFF),
            (0x0001, 0x3800, 0x0000),
            (0x3555, 0x3555, 0xB555),
        ];
        foreach (var (a, b, c) in fmas)
        {
            var expected = H(Hv(a) * Hv(b) + Hv(c));
            cases.Add(Vector($"fmac_f16 {a:X4},{b:X4},{c:X4}", fmac, [a, b, c], expected, 0xFFFF, result: 2));
        }

        (uint Value, uint Mask)[] classes =
        [
            (0x7C01, 1), (0x7C01, 2), (0x7E00, 2), (0x7E00, 1), (0xFC00, 4), (0xBC00, 8), (0x8001, 0x10),
            (0x8000, 0x20), (0x0000, 0x40), (0x0001, 0x80), (0x3C00, 0x100), (0x7C00, 0x200), (0x3C00, 0x200),
            (0x3C00, 0x3FF ^ 0x100), (0x0000, 0x20), (0x83FF, 0x3EF),
        ];
        foreach (var (value, mask) in classes)
        {
            cases.Add(Vector($"cmp_class_f16 {value:X4} {mask:X3}", cmpClass, [value, mask], ClassF16(value, mask)));
            cases.Add(Vector($"cmp_class_f16 -src0 {value:X4} {mask:X3}", cmpClassNeg, [value, mask], ClassF16(value ^ 0x8000, mask)));
        }

        // v_cmpx_class_f16 writes EXEC: v8 = 1 only when the class matches.
        foreach (var (value, mask) in classes.Take(6))
        {
            uint[] words = [.. Inputs(value, mask), 0x7D3E0300, 0x7E100281, RestoreExec];
            cases.Add(new Case($"cmpx_class_f16 {value:X4} {mask:X3}", words, [8], [ClassF16(value, mask) != 0 ? 1u : Sentinel]));
        }

        Run("f16", cases);
    }

    [Fact]
    public void VectorSwapsAndRelativeMovesUseM0()
    {
        var cases = new List<Case>();
        uint[] values = [0x10101010, 0x11111111, 0x22222222, 0x33333333, 0x44444444, 0x55555555];
        cases.Add(new Case("swap_b32", [.. Inputs(values), 0x7E02CB02], [1, 2], [values[2], values[1]]));
        cases.Add(new Case(
            "swaprel_b32 m0 dst+1 src+2",
            [.. Inputs(values), .. S(M0, 0x00010002), 0x7E02D102],
            [1, 2, 3, 4, 5],
            [values[1], values[4], values[3], values[2], values[5]]));
        cases.Add(new Case("movrels_b32_e64", [.. Inputs(values), .. S(M0, 2), 0xD5C30008, 0x00000101], [8], [values[3]]));
        cases.Add(new Case(
            "movreld_b32_e64",
            [.. Inputs(values), .. S(M0, 3), 0xD5C20001, 0x00000100],
            [1, 4],
            [values[1], values[0]]));
        Run("vmovrel", cases);
    }

    [Fact]
    public void IntegerVop2AndVop3OpsMatchReferences()
    {
        uint[] mulHi24 = [0x14100300];               // v_mul_hi_i32_i24 v8, v0, v1
        uint[] dot4c = [0x1A040300];                 // v_dot4c_i32_i8 v2, v0, v1
        uint[] lerp = [0xD54D0008, 0x040A0300];      // v_lerp_u8 v8, v0, v1, v2
        uint[] msad = [0xD5710008, 0x040A0300];      // v_msad_u8 v8, v0, v1, v2
        uint[] perm = [0xD7440008, 0x040A0300];      // v_perm_b32 v8, v0, v1, v2
        var cases = new List<Case>();
        (uint A, uint B, uint C)[] triples =
        [
            (0x007FFFFF, 0x007FFFFF, 0x00000000),
            (0x00800000, 0x007FFFFF, 0xFFFFFFFF),
            (0xFF800000, 0x12800000, 0x01010101),
            (0x80FF7F01, 0x7F80FF02, 0x7FFFFFF0),
            (0x01020304, 0x00000000, 0x80000005),
            (0xFFFFFFFF, 0xFFFFFFFF, 0x0C0D0E0F),
            (0x12345678, 0x9ABCDEF0, 0x08090A0B),
            (0x00FF0080, 0xFF0000FE, 0x07060504),
            (0xDEADBEEF, 0x01000100, 0x03020100),
        ];
        foreach (var (a, b, c) in triples)
        {
            var tag = $"{a:X8},{b:X8},{c:X8}";
            cases.Add(Vector($"mul_hi_i32_i24 {tag}", mulHi24, [a, b], (uint)((ulong)((long)Sext(a, 24) * Sext(b, 24)) >> 32)));
            var dot = (int)c;
            for (var index = 0; index < 4; index++)
            {
                dot = unchecked(dot + (sbyte)Byte(a, index) * (sbyte)Byte(b, index));
            }

            cases.Add(Vector($"dot4c_i32_i8 {tag}", dot4c, [a, b, c], (uint)dot, result: 2));
            uint lerped = 0;
            for (var index = 0; index < 4; index++)
            {
                lerped |= ((Byte(a, index) + Byte(b, index) + ((c >> (index * 8)) & 1)) >> 1) << (index * 8);
            }

            cases.Add(Vector($"lerp_u8 {tag}", lerp, [a, b, c], lerped));
            cases.Add(Vector($"msad_u8 {tag}", msad, [a, b, c], unchecked(c + ByteSad(a, b, masked: true))));
            cases.Add(Vector($"perm_b32 {tag}", perm, [a, b, c], Perm(a, b, c)));
        }

        Run("int32", cases);
    }

    [Fact]
    public void QuadSadsWriteEveryResultBeforeOverlap()
    {
        uint[] qsad = [0xD5720008, 0x04120500];      // v_qsad_pk_u16_u8 v[8:9], v[0:1], v2, v[4:5]
        uint[] mqsadPk = [0xD5730008, 0x04120500];   // v_mqsad_pk_u16_u8 v[8:9], v[0:1], v2, v[4:5]
        uint[] mqsad = [0xD5750008, 0x04120500];     // v_mqsad_u32_u8 v[8:11], v[0:1], v2, v[4:7]
        uint[] mqsadOverlap = [0xD5750004, 0x04120500]; // v_mqsad_u32_u8 v[4:7], v[0:1], v2, v[4:7]
        var cases = new List<Case>();
        (ulong Window, uint Reference, uint[] Accumulators)[] inputs =
        [
            (0x0011223344556677, 0x00FF7F10, [0x00010002, 0xFFFF0003, 0x00000004, 0xFFFFFFFF]),
            (0xFFFFFFFFFFFFFFFF, 0x00000000, [0xFFF0FFF0, 0x00100010, 0x80000000, 0x7FFFFFFF]),
            (0x8040201008040201, 0x01020408, [0x00000000, 0x00000000, 0x00000005, 0x00000006]),
            (0x0123456789ABCDEF, 0xFEDCBA98, [0xFE00FD00, 0x00FF00FE, 0xFFFFFF00, 0x00000100]),
        ];
        foreach (var (window, reference, accumulators) in inputs)
        {
            var tag = $"{window:X16},{reference:X8}";
            uint[] values = [(uint)window, (uint)(window >> 32), reference, 0, .. accumulators];
            uint[] Sads(bool masked) =>
                [.. Enumerable.Range(0, 4).Select(index => ByteSad((uint)(window >> (index * 8)), reference, masked))];
            uint PackedLane(uint[] sads, int index) =>
                (((accumulators[index / 2] >> (index % 2 * 16)) + sads[index]) & 0xFFFF);
            var plain = Sads(masked: false);
            var masked = Sads(masked: true);
            cases.Add(new Case($"qsad_pk_u16_u8 {tag}", [.. Inputs(values), .. qsad], [8, 9],
                [PackedLane(plain, 0) | (PackedLane(plain, 1) << 16), PackedLane(plain, 2) | (PackedLane(plain, 3) << 16)]));
            cases.Add(new Case($"mqsad_pk_u16_u8 {tag}", [.. Inputs(values), .. mqsadPk], [8, 9],
                [PackedLane(masked, 0) | (PackedLane(masked, 1) << 16), PackedLane(masked, 2) | (PackedLane(masked, 3) << 16)]));
            uint[] wide = [.. Enumerable.Range(0, 4).Select(index => unchecked(accumulators[index] + masked[index]))];
            cases.Add(new Case($"mqsad_u32_u8 {tag}", [.. Inputs(values), .. mqsad], [8, 9, 10, 11], wide));
            cases.Add(new Case($"mqsad_u32_u8 overlap {tag}", [.. Inputs(values), .. mqsadOverlap], [4, 5, 6, 7], wide));
        }

        Run("qsad", cases);
    }

    [Fact]
    public void SixtyFourBitMadAndShiftMatchReferences()
    {
        uint[] madS20 = [0xD5771408, 0x040A0300, VFromS(10, 20)]; // v_mad_i64_i32 v[8:9], s20, v0, v1, v[2:3]
        uint[] madVcc = [0xD5776A08, 0x040A0300, 0xD501000A, 0x01A90280]; // ..., vcc; v_cndmask v10, 0, 1, vcc
        uint[] ashr = [0xD7010008, 0x00020500];                    // v_ashrrev_i64 v[8:9], v0, v[2:3]
        var cases = new List<Case>();
        (int A, int B, long C)[] mads =
        [
            (2, 3, 4), (-1, 1, 0), (int.MinValue, int.MinValue, long.MaxValue), (int.MinValue, int.MaxValue, long.MinValue),
            (-5, 7, 100), (123456789, -987654321, 0x7FFFFFFFFFFFFFF0), (0, 0, -1), (1, 1, long.MaxValue), (-1, 1, long.MinValue),
        ];
        foreach (var (a, b, c) in mads)
        {
            var sum = (Int128)((long)a * b) + c;
            var low = (ulong)(sum & ulong.MaxValue);
            var carry = (uint)(sum >> 64) & 1;
            uint[] values = [(uint)a, (uint)b, (uint)c, (uint)((ulong)c >> 32)];
            var tag = $"{a},{b},{c}";
            cases.Add(new Case($"mad_i64_i32 s20 {tag}", [.. Inputs(values), .. madS20], [8, 9, 10],
                [(uint)low, (uint)(low >> 32), carry != 0 ? 0xFFFFFFFFu : 0u]));
            cases.Add(new Case($"mad_i64_i32 vcc {tag}", [.. Inputs(values), .. madVcc], [8, 9, 10],
                [(uint)low, (uint)(low >> 32), carry]));
        }

        foreach (var (shift, value) in new (uint, long)[] { (4, unchecked((long)0x8000000000000010)), (63, long.MinValue), (65, -3), (0, 12345), (32, 0x7FFFFFFF00000000), (31, -1) })
        {
            var result = (ulong)(value >> (int)(shift & 63));
            cases.Add(new Case($"ashrrev_i64 {shift} {value:X}", [.. Inputs(shift, 0, (uint)value, (uint)((ulong)value >> 32)), .. ashr],
                [8, 9], [(uint)result, (uint)(result >> 32)]));
        }

        Run("int64", cases);
    }

    [Fact]
    public void Int16Vop3OpsHonourOpSelAndClamp()
    {
        var cases = new List<Case>();
        (uint A, uint B, uint C)[] triples =
        [
            (0x00030005, 0x00070002, 0x00010009),
            (0x8000FFFF, 0x7FFF0002, 0x12348000),
            (0x1234FFFF, 0x5678FFFF, 0x9ABC7FFF),
            (0x00FF0100, 0x01000100, 0xFFFF0001),
            (0x7FFF7FFF, 0x7FFF7FFF, 0x80007FFF),
        ];
        foreach (var (a, b, c) in triples)
        {
            var tag = $"{a:X8},{b:X8},{c:X8}";
            uint[] values = [a, b, c];
            const uint W1 = 0x040A0300;
            void Add(string name, uint word0, uint expected, uint mask = 0xFFFF) =>
                cases.Add(Vector($"{name} {tag}", [word0, W1], values, expected, mask));

            long uProduct = Lo(a) * Lo(b);
            Add("mul_lo_u16", 0xD7050008, (uint)uProduct & 0xFFFF);
            Add("mul_lo_u16 clamp", 0xD7058008, SaturateU16(uProduct));
            Add("mad_u16", 0xD7400008, (uint)(uProduct + Lo(c)) & 0xFFFF);
            Add("mad_u16 clamp", 0xD7408008, SaturateU16(uProduct + Lo(c)));
            Add("mad_u16 op_sel:[1,0,1,1]", 0xD7406808, Pack(Sentinel, (Hi(a) * Lo(b)) + Hi(c)), 0xFFFFFFFF);
            long sProduct = (short)Lo(a) * (short)Lo(b);
            Add("mad_i16", 0xD75E0008, (uint)(sProduct + (short)Lo(c)) & 0xFFFF);
            Add("mad_i16 clamp", 0xD75E8008, SaturateI16(sProduct + (short)Lo(c)));
            Add("mad_i32_i16", 0xD7750008, unchecked((uint)(sProduct + (int)c)), 0xFFFFFFFF);
            Add("mad_i32_i16 clamp", 0xD7758008, SaturateI32(sProduct + (int)c), 0xFFFFFFFF);
            Add("mad_i32_i16 op_sel:[1,1,0,0]", 0xD7751808, unchecked((uint)((short)Hi(a) * (short)Hi(b) + (int)c)), 0xFFFFFFFF);
            long sa = (short)Lo(a), sb = (short)Lo(b), sc = (short)Lo(c);
            long ua = Lo(a), ub = Lo(b), uc = Lo(c);
            Add("min3_i16", 0xD7520008, (uint)Math.Min(sa, Math.Min(sb, sc)) & 0xFFFF);
            Add("min3_u16", 0xD7530008, (uint)Math.Min(ua, Math.Min(ub, uc)));
            Add("max3_i16", 0xD7550008, (uint)Math.Max(sa, Math.Max(sb, sc)) & 0xFFFF);
            Add("max3_u16", 0xD7560008, (uint)Math.Max(ua, Math.Max(ub, uc)));
            Add("med3_i16", 0xD7580008, (uint)Median(sa, sb, sc) & 0xFFFF);
            Add("med3_u16", 0xD7590008, (uint)Median(ua, ub, uc));
            Add("med3_i16 op_sel:[1,1,1,1]", 0xD7587808,
                Pack(Sentinel, (uint)Median((short)Hi(a), (short)Hi(b), (short)Hi(c))), 0xFFFFFFFF);
        }

        Run("int16", cases);
    }

    [Fact]
    public void PackedInt16OpsMatchPerLaneReferences()
    {
        var cases = new List<Case>();
        (uint A, uint B, uint C)[] triples =
        [
            (0x00030005, 0x00070002, 0x00010009),
            (0x8000FFFF, 0x7FFF0002, 0x12348000),
            (0x0011001F, 0xF00F8001, 0x7FFF7FFF),
            (0xFFFF0001, 0xFFFF0002, 0xFFFF0003),
            (0x01000004, 0x01000100, 0x00008000),
        ];
        foreach (var (a, b, c) in triples)
        {
            var tag = $"{a:X8},{b:X8},{c:X8}";
            uint[] values = [a, b, c];
            void Add(string name, uint[] op, Func<uint, uint, uint, uint> lane) =>
                cases.Add(Vector($"{name} {tag}", op, values,
                    Pack(lane(Lo(a), Lo(b), Lo(c)), lane(Hi(a), Hi(b), Hi(c)))));

            Add("pk_mad_i16", [0xCC004008, 0x1C0A0300], (x, y, z) => (uint)((short)x * (short)y + (short)z) & 0xFFFF);
            Add("pk_mad_i16 clamp", [0xCC00C008, 0x1C0A0300], (x, y, z) => SaturateI16((short)x * (short)y + (short)z));
            Add("pk_mul_lo_u16", [0xCC014008, 0x18020300], (x, y, _) => (x * y) & 0xFFFF);
            Add("pk_mul_lo_u16 clamp", [0xCC01C008, 0x18020300], (x, y, _) => SaturateU16((long)x * y));
            Add("pk_lshlrev_b16", [0xCC044008, 0x18020300], (x, y, _) => (y << (int)(x & 15)) & 0xFFFF);
            Add("pk_lshrrev_b16", [0xCC054008, 0x18020300], (x, y, _) => y >> (int)(x & 15));
            Add("pk_ashrrev_i16", [0xCC064008, 0x18020300], (x, y, _) => (uint)((short)y >> (int)(x & 15)) & 0xFFFF);
            Add("pk_max_i16", [0xCC074008, 0x18020300], (x, y, _) => (uint)Math.Max((short)x, (short)y) & 0xFFFF);
            Add("pk_min_i16", [0xCC084008, 0x18020300], (x, y, _) => (uint)Math.Min((short)x, (short)y) & 0xFFFF);
            Add("pk_mad_u16", [0xCC094008, 0x1C0A0300], (x, y, z) => (x * y + z) & 0xFFFF);
            Add("pk_mad_u16 clamp", [0xCC09C008, 0x1C0A0300], (x, y, z) => SaturateU16((long)x * y + z));
            Add("pk_add_u16", [0xCC0A4008, 0x18020300], (x, y, _) => (x + y) & 0xFFFF);
            Add("pk_add_u16 clamp", [0xCC0AC008, 0x18020300], (x, y, _) => SaturateU16((long)x + y));
            Add("pk_sub_u16", [0xCC0B4008, 0x18020300], (x, y, _) => (x - y) & 0xFFFF);
            Add("pk_sub_u16 clamp", [0xCC0BC008, 0x18020300], (x, y, _) => SaturateU16((long)x - y));
            Add("pk_max_u16", [0xCC0C4008, 0x18020300], (x, y, _) => Math.Max(x, y));
            Add("pk_min_u16", [0xCC0D4008, 0x18020300], (x, y, _) => Math.Min(x, y));
            // op_sel:[1,0] op_sel_hi:[0,1]: src0 halves swapped.
            cases.Add(Vector($"pk_max_u16 swapped {tag}", [0xCC0C4808, 0x10020300], values,
                Pack(Math.Max(Hi(a), Lo(b)), Math.Max(Lo(a), Hi(b)))));
            cases.Add(Vector($"pk_mad_i16 swapped {tag}", [0xCC004808, 0x140A0300], values,
                Pack((uint)((short)Hi(a) * (short)Lo(b) + (short)Lo(c)), (uint)((short)Lo(a) * (short)Hi(b) + (short)Hi(c)))));
        }

        Run("pk16", cases);
    }

    [Fact]
    public void DotProductsMatchReferences()
    {
        var cases = new List<Case>();
        (uint A, uint B, uint C)[] triples =
        [
            (0x01020304, 0x05060708, 0x00000010),
            (0x80FF7F01, 0x7F80FF02, 0x7FFFFFF0),
            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFF0),
            (0x80008000, 0x80008000, 0x80000005),
            (0x89ABCDEF, 0x76543210, 0x00000000),
            (0x7FFF8000, 0x80007FFF, 0x80000005),
        ];
        long SignedSum(uint a, uint b, int bits)
        {
            long sum = 0;
            for (var shift = 0; shift < 32; shift += bits)
            {
                sum += (long)Sext(a >> shift, bits) * Sext(b >> shift, bits);
            }

            return sum;
        }

        long UnsignedSum(uint a, uint b, int bits)
        {
            long sum = 0;
            var mask = (1u << bits) - 1;
            for (var shift = 0; shift < 32; shift += bits)
            {
                sum += (long)((a >> shift) & mask) * ((b >> shift) & mask);
            }

            return sum;
        }

        foreach (var (a, b, c) in triples)
        {
            var tag = $"{a:X8},{b:X8},{c:X8}";
            uint[] values = [a, b, c];
            foreach (var (name, opcode, bits, signed) in new (string, uint, int, bool)[]
            {
                ("dot2_i32_i16", 0x14, 16, true), ("dot2_u32_u16", 0x15, 16, false), ("dot4_i32_i8", 0x16, 8, true),
                ("dot4_u32_u8", 0x17, 8, false), ("dot8_i32_i4", 0x18, 4, true), ("dot8_u32_u4", 0x19, 4, false),
            })
            {
                var word0 = 0xCC004008 | (opcode << 16);
                var sum = signed ? SignedSum(a, b, bits) + (int)c : UnsignedSum(a, b, bits) + c;
                cases.Add(Vector($"{name} {tag}", [word0, 0x1C0A0300], values, unchecked((uint)sum)));
                cases.Add(Vector($"{name} clamp {tag}", [word0 | 0x8000, 0x1C0A0300], values,
                    signed ? SaturateI32(sum) : SaturateU32(sum)));
            }

            // op_sel:[1,0,0] op_sel_hi:[0,1,1]: element 0 takes a.hi, element 1 takes a.lo.
            cases.Add(Vector($"dot2_i32_i16 swapped {tag}", [0xCC144808, 0x140A0300], values,
                unchecked((uint)((short)Hi(a) * (short)Lo(b) + (short)Lo(a) * (short)Hi(b) + (int)c))));
        }

        (double A0, double A1, double B0, double B1, float C)[] floats =
        [
            (1.5, 2.0, 3.0, -0.5, 10f),
            (0.25, -4.0, 8.0, 0.125, -0.5f),
            (1.0, 1.0, 0.5, 0.25, -1f),
            (-2.0, 3.0, 0.125, 0.0625, 0.25f),
            (65504.0, 65504.0, 2.0, 2.0, 0f),
        ];
        foreach (var (a0, a1, b0, b1, c) in floats)
        {
            var a = Pack(H(a0), H(a1));
            var b = Pack(H(b0), H(b1));
            uint[] values = [a, b, F(c)];
            var tag = $"{a:X8},{b:X8},{c}";
            var sum = ((float)(a0 * b0) + (float)(a1 * b1)) + c;
            cases.Add(Vector($"dot2_f32_f16 {tag}", [0xCC134008, 0x1C0A0300], values, F(sum)));
            cases.Add(Vector($"dot2_f32_f16 clamp {tag}", [0xCC13C008, 0x1C0A0300], values, F(Math.Clamp(sum, 0f, 1f))));
            // op_sel:[1,0,0] op_sel_hi:[0,1,1] neg_lo:[1,0,0] neg_hi:[0,1,0]
            var modified = ((float)(-a1 * b0) + (float)(a0 * -b1)) + c;
            cases.Add(Vector($"dot2_f32_f16 mods {tag}", [0xCC134A08, 0x340A0300], values, F(modified)));
        }

        Run("dot", cases);
    }

    [Fact]
    public void PackF16AndPackNormMatchReferences()
    {
        var cases = new List<Case>();
        uint[] halves = [0x3C00, 0xBC00, 0x3800, 0x7E00, 0x7C01, 0x0001, 0x3555, 0xC000, 0x8000];
        for (var index = 0; index < halves.Length; index++)
        {
            var a = 0xA5A50000u | halves[index];
            var b = (halves[(index + 3) % halves.Length] << 16) | halves[(index + 1) % halves.Length];
            uint[] values = [a, b];
            var tag = $"{a:X8},{b:X8}";
            cases.Add(Vector($"pack_b32_f16 {tag}", [0xD7110008, 0x00020300], values, Pack(Lo(a), Lo(b))));
            // -v0, |v1|, op_sel:[1,0]
            cases.Add(Vector($"pack_b32_f16 mods {tag}", [0xD7110A08, 0x20020300], values,
                Pack(Hi(a) ^ 0x8000, Lo(b) & 0x7FFF)));
            cases.Add(Vector($"cvt_pknorm_i16_f16 {tag}", [0xD7120008, 0x00020300], values, Pack(NormI16(Lo(a)), NormI16(Lo(b)))));
            cases.Add(Vector($"cvt_pknorm_u16_f16 {tag}", [0xD7130008, 0x00020300], values, Pack(NormU16(Lo(a)), NormU16(Lo(b)))));
        }

        Run("pack", cases);
    }

    [Fact]
    public void ScalarBitOpsMatchReferences()
    {
        var cases = new List<Case>();
        // op reads s4 (or s[4:5]), writes s0 (or s[0:1]); v8 = s0, v9 = s1, v10 = SCC.
        Case Scalar(string label, uint[] op, uint s4, uint s5, uint s0, uint s1, uint scc, bool checkHigh, bool checkScc)
        {
            uint[] words =
            [
                .. Inputs(), .. S(0, 0x5EED0000), .. S(1, 0x5EED0001), .. S(4, s4), .. S(5, s5), ClearScc, .. op,
                SccToS30, VFromS(8, 0), VFromS(9, 1), VFromS(10, 30),
            ];
            var results = new List<uint> { 8 };
            var expected = new List<uint> { s0 };
            if (checkHigh)
            {
                results.Add(9);
                expected.Add(s1);
            }

            if (checkScc)
            {
                results.Add(10);
                expected.Add(scc);
            }

            return new Case(label, words, [.. results], [.. expected]);
        }

        (uint Lo, uint Hi)[] values =
        [
            (0, 0), (0xFFFFFFFF, 0xFFFFFFFF), (0x00000001, 0x80000000), (0x40000000, 0), (0x80000000, 0xFFFFFFFF),
            (0x0FFFFFFF, 0x12345678), (0xFFFF0000, 0x0000FFFF), (0x12345678, 0x9ABCDEF0), (0x000000FF, 0xFFFFFFFE), (0x7F, 0x80),
            (0xFFFFFFFE, 0x7FFFFFFF), (0x00008000, 0),
        ];
        foreach (var (lo, hi) in values)
        {
            var tag = $"{hi:X8}{lo:X8}";
            var wide = ((ulong)hi << 32) | lo;
            var reversed = ReverseBits(wide);
            cases.Add(Scalar($"s_brev_b64 {tag}", [0xBE800C04], lo, hi, (uint)reversed, (uint)(reversed >> 32), 0, true, false));
            var count32 = (uint)BitOperations.PopCount(~lo);
            cases.Add(Scalar($"s_bcnt0_i32_b32 {tag}", [0xBE800D04], lo, hi, count32, 0, count32 != 0 ? 1u : 0u, false, true));
            var count64 = (uint)BitOperations.PopCount(~wide);
            cases.Add(Scalar($"s_bcnt0_i32_b64 {tag}", [0xBE800E04], lo, hi, count64, 0, count64 != 0 ? 1u : 0u, false, true));
            var ff0 = ~lo == 0 ? 0xFFFFFFFFu : (uint)BitOperations.TrailingZeroCount(~lo);
            cases.Add(Scalar($"s_ff0_i32_b32 {tag}", [0xBE801104], lo, hi, ff0, 0, 0, false, false));
            var ff064 = ~wide == 0 ? 0xFFFFFFFFu : (uint)BitOperations.TrailingZeroCount(~wide);
            cases.Add(Scalar($"s_ff0_i32_b64 {tag}", [0xBE801204], lo, hi, ff064, 0, 0, false, false));
            var flipped = (lo & 0x80000000) != 0 ? ~lo : lo;
            var flbit = flipped == 0 ? 0xFFFFFFFFu : (uint)BitOperations.LeadingZeroCount(flipped);
            cases.Add(Scalar($"s_flbit_i32 {tag}", [0xBE801704], lo, hi, flbit, 0, 0, false, false));
            var flipped64 = (hi & 0x80000000) != 0 ? ~wide : wide;
            var flbit64 = flipped64 == 0 ? 0xFFFFFFFFu : (uint)BitOperations.LeadingZeroCount(flipped64);
            cases.Add(Scalar($"s_flbit_i32_i64 {tag}", [0xBE801804], lo, hi, flbit64, 0, 0, false, false));
            cases.Add(Scalar($"s_sext_i32_i8 {tag}", [0xBE801904], lo, hi, (uint)(sbyte)lo, 0, 0, false, false));
            cases.Add(Scalar($"s_sext_i32_i16 {tag}", [0xBE801A04], lo, hi, (uint)(short)lo, 0, 0, false, false));
            var difference = unchecked((int)lo - (int)hi);
            var absolute = unchecked((uint)(difference < 0 ? -difference : difference));
            cases.Add(Scalar($"s_absdiff_i32 {tag}", [0x96000504], lo, hi, absolute, 0, absolute != 0 ? 1u : 0u, false, true));
        }

        Run("sbits", cases);
    }

    [Fact]
    public void ScalarMovesSelectsAndExecWritesMatchReferences()
    {
        var cases = new List<Case>();
        uint[] Base() =>
        [
            .. Inputs(), .. S(0, 0x5EED0000), .. S(1, 0x5EED0001), .. S(4, 0x44444444), .. S(5, 0x55555555),
            .. S(6, 0x66666666), .. S(7, 0x77777777), .. S(20, 0x20202020), .. S(21, 0x21212121),
        ];

        cases.Add(new Case("s_cmovk_i32 scc=1", [.. Base(), SetScc, 0xB100FFFE, VFromS(8, 0)], [8], [0xFFFFFFFE]));
        cases.Add(new Case("s_cmovk_i32 scc=0", [.. Base(), ClearScc, 0xB100FFFE, VFromS(8, 0)], [8], [0x5EED0000]));
        cases.Add(new Case("s_cmov_b32 scc=1", [.. Base(), SetScc, 0xBE800504, VFromS(8, 0)], [8], [0x44444444]));
        cases.Add(new Case("s_cmov_b32 scc=0", [.. Base(), ClearScc, 0xBE800504, VFromS(8, 0)], [8], [0x5EED0000]));
        cases.Add(new Case("s_movrels_b32 m0=2", [.. Base(), .. S(M0, 2), 0xBE802E04, VFromS(8, 0)], [8], [0x66666666]));
        cases.Add(new Case("s_movrels_b64 m0=2", [.. Base(), .. S(M0, 2), 0xBE802F04, VFromS(8, 0), VFromS(9, 1)],
            [8, 9], [0x66666666, 0x77777777]));
        cases.Add(new Case("s_movreld_b32 m0=2", [.. Base(), .. S(M0, 2), 0xBE843014, VFromS(8, 6), VFromS(9, 4)],
            [8, 9], [0x20202020, 0x44444444]));
        cases.Add(new Case("s_movreld_b64 m0=2", [.. Base(), .. S(M0, 2), 0xBE843114, VFromS(8, 6), VFromS(9, 7), VFromS(10, 4)],
            [8, 9, 10], [0x20202020, 0x21212121, 0x44444444]));
        // m0 = 102 relocates s4 onto VCC_LO; the VCC view used by v_cndmask must follow.
        cases.Add(new Case("s_movreld_b32 into vcc_lo",
            [.. Base(), .. S(106, 0), .. S(20, 0xFFFFFFFF), .. S(M0, 102), 0xBE843014, 0xD5010008, 0x01A90280],
            [8], [1]));
        // m0 = (2 << 16) | 1: s[4 + 2] = s[6 + 1].
        cases.Add(new Case("s_movrelsd_2_b32", [.. Base(), .. S(M0, 0x00020001), 0xBE844906, VFromS(8, 6), VFromS(9, 7)],
            [8, 9], [0x77777777, 0x77777777]));

        // Wrexec: EXEC = 0x0000FFFF, s4 = 0x00FF00FF. v8 = D, v9 = SCC, v10 = EXEC after the op.
        // Every lane stores v8, so the case also restores EXEC before reading back.
        void Wrexec(string name, uint op, uint source, uint expected, bool scc)
        {
            uint[] words =
            [
                .. Base(), .. S(4, source), .. S(ExecLo, 0x0000FFFF), ClearScc, op, SccToS30, ExecToS21, RestoreExec,
                VFromS(8, 0), VFromS(9, 30), VFromS(10, 21),
            ];
            cases.Add(new Case($"{name} {source:X8}", words, [8, 9, 10], [expected, scc ? 1u : 0u, expected]));
        }

        foreach (var source in new uint[] { 0x00FF00FF, 0xFFFFFFFF, 0 })
        {
            Wrexec("s_andn1_wrexec_b32", 0xBE804604, source, ~source & 0xFFFF, (~source & 0xFFFF) != 0);
            Wrexec("s_andn2_wrexec_b32", 0xBE804704, source, source & 0xFFFF0000, (source & 0xFFFF0000) != 0);
            Wrexec("s_andn1_wrexec_b64", 0xBE803904, source, ~source & 0xFFFF, (~source & 0xFFFF) != 0);
            // The 64-bit form also ANDs s5 = 0x55555555 with ~EXEC_HI (0), so SCC stays set.
            Wrexec("s_andn2_wrexec_b64", 0xBE803A04, source, source & 0xFFFF0000, true);
        }

        // HW_REG_MODE reads the launch state: FLOAT_MODE 0xC0 in bits 7:0, DX10_CLAMP (on) in
        // bit 8, IEEE (off) in bit 9.
        cases.Add(new Case("s_getreg_b32 MODE[7:0]", [.. Base(), 0xB9003801, VFromS(8, 0)], [8], [0xC0]));
        cases.Add(new Case("s_getreg_b32 MODE[7:4]", [.. Base(), 0xB9001901, VFromS(8, 0)], [8], [0xC]));
        cases.Add(new Case("s_getreg_b32 MODE", [.. Base(), 0xB900F801, VFromS(8, 0)], [8], [0x1C0]));
        cases.Add(new Case("s_getreg_b32 MODE[31:9]", [.. Base(), 0xB900B241, VFromS(8, 0)], [8], [0]));
        Run("smov", cases);
    }

    [Theory]
    [InlineData("s_denorm_mode", new uint[] { 0xBFA50000 })]
    [InlineData("s_round_mode 0", new uint[] { 0xBFA40000 })]
    [InlineData("s_wakeup", new uint[] { 0xBF830000 })]
    [InlineData("s_setkill", new uint[] { 0xBF8B0000 })]
    [InlineData("s_sleep", new uint[] { 0xBF8E0001 })]
    [InlineData("s_icache_inv", new uint[] { 0xBF930000 })]
    [InlineData("s_incperflevel", new uint[] { 0xBF940001 })]
    [InlineData("s_decperflevel", new uint[] { 0xBF950001 })]
    [InlineData("s_wait_idle", new uint[] { 0xBFA20000 })]
    [InlineData("s_ttracedata_imm", new uint[] { 0xBFA80001 })]
    [InlineData("s_version", new uint[] { 0xB0800000 })]
    [InlineData("v_pipeflush", new uint[] { 0x7E003600 })]
    [InlineData("v_clrexcp", new uint[] { 0x7E008200 })]
    [InlineData("s_setreg_imm32_b32 MODE round", new uint[] { 0xBA801801, 0 })]
    public void StateOnlyInstructionsTranslate(string name, uint[] op)
    {
        var request = Gen5ComputeFixtures.CreateComputeRequest(new Gen5ComputeFixture(name, [.. op, EndProgram], 0, 0));
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out _, out var error), error);
    }

    // A launch FLOAT_MODE rounding other than to nearest even is refused, as S_ROUND_MODE is.
    [Fact]
    public void ANonNearestEvenLaunchRoundingIsRefused()
    {
        var request = Gen5ComputeFixtures.CreateComputeRequest(
            new Gen5ComputeFixture("float-mode round", [EndProgram], 0, 0), floatMode: 0xC1);
        Assert.False(Gen5MslTranslator.TryCompileProgram(request, out _, out var error));
        Assert.Contains("only round-to-nearest-even (FP_ROUND 0) is supported", error, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("s_round_mode 1", new uint[] { 0xBFA40001 }, "s_round_mode 0x1: only round-to-nearest-even (0) is supported")]
    [InlineData("s_sethalt", new uint[] { 0xBF8D0001 }, "SSethalt halts the wave or returns from a trap handler")]
    [InlineData("s_sendmsghalt", new uint[] { 0xBF910001 }, "SSendmsghalt halts the wave or returns from a trap handler")]
    [InlineData("s_endpgm_saved", new uint[] { 0xBF9B0000 }, "SEndpgmSaved halts the wave or returns from a trap handler")]
    [InlineData("s_rfe_b64", new uint[] { 0xBE802200 }, "SRfeB64 halts the wave or returns from a trap handler")]
    [InlineData("s_call_b64", new uint[] { 0xBB000000 }, "SCallB64 is subroutine/subvector control flow")]
    [InlineData("s_subvector_loop_begin", new uint[] { 0xBD800000 }, "SSubvectorLoopBegin is subroutine/subvector control flow")]
    [InlineData("s_subvector_loop_end", new uint[] { 0xBE000000 }, "SSubvectorLoopEnd is subroutine/subvector control flow")]
    [InlineData("s_setreg_imm32_b32 MODE.IEEE", new uint[] { 0xBA800241, 1 }, "s_setreg_imm32_b32 writes MODE.IEEE/DX10_CLAMP")]
    [InlineData("s_getreg_b32 TRAPSTS", new uint[] { 0xB900F803 }, "s_getreg_b32 of hardware register 3 is not supported")]
    [InlineData("v_dot4_i32_i8 op_sel", new uint[] { 0xCC164808, 0x1C0A0300 }, "VDot4I32I8: op_sel/op_sel_hi other than the default")]
    [InlineData("v_dot2_i32_i16 neg_lo", new uint[] { 0xCC144008, 0x3C0A0300 }, "VDot2I32I16: neg_lo/neg_hi on an integer dot product")]
    [InlineData("v_dot2_f32_f16 neg acc", new uint[] { 0xCC134008, 0x9C0A0300 }, "VDot2F32F16: op_sel/op_sel_hi/neg on the 32-bit accumulator")]
    [InlineData("v_msad_u8 clamp", new uint[] { 0xD5718008, 0x040A0300 }, "VMsadU8: clamped integer results are not supported")]
    [InlineData("v_mad_i64_i32 clamp", new uint[] { 0xD5779408, 0x040A0300 }, "VMadI64I32: clamped integer results are not supported")]
    [InlineData("v_mqsad_u32_u8 constant acc", new uint[] { 0xD5750008, 0x02020500 }, "VMqsadU32U8: a constant 128-bit accumulator (src2)")]
    [InlineData("v_fmac_f16_e64 neg acc", new uint[] { 0xD5360008, 0x80020300 }, "VFmacF16: modifiers or op_sel on the accumulator")]
    [InlineData("v_pack_b32_f16 clamp", new uint[] { 0xD7118008, 0x00020300 }, "VPackB32F16: clamp, omod and op_sel beyond the two sources")]
    [InlineData("v_cvt_pknorm_i16_f16 omod", new uint[] { 0xD7120008, 0x08020300 }, "VCvtPknormI16F16: omod and op_sel beyond the two sources")]
    public void UnsupportedFormsAreRefused(string name, uint[] op, string expectedError)
    {
        var request = Gen5ComputeFixtures.CreateComputeRequest(new Gen5ComputeFixture(name, [.. op, EndProgram], 0, 0));
        Assert.False(Gen5MslTranslator.TryCompileProgram(request, out _, out var error));
        Assert.Contains(expectedError, error);
    }

    // neg_lo/neg_hi only negate floating-point inputs: a packed integer op with them set
    // translates exactly as without them.
    [Fact]
    public void PackedIntegerOpsIgnoreNegateModifiers()
    {
        var plain = Gen5ComputeFixtures.CreateComputeRequest(
            new Gen5ComputeFixture("v_pk_add_u16", [0xCC0A4008, 0x18020300, EndProgram], 0, 0));
        var negated = Gen5ComputeFixtures.CreateComputeRequest(
            new Gen5ComputeFixture("v_pk_add_u16", [0xCC0A4008, 0x38020300, EndProgram], 0, 0));
        Assert.True(Gen5MslTranslator.TryCompileProgram(plain, out var expected, out var error), error);
        Assert.True(Gen5MslTranslator.TryCompileProgram(negated, out var actual, out error), error);
        Assert.Equal(expected.Source, actual.Source);
    }

    // ---- harness ----

    private static ulong ReverseBits(ulong value)
    {
        ulong result = 0;
        for (var bit = 0; bit < 64; bit++)
        {
            result |= ((value >> bit) & 1) << (63 - bit);
        }

        return result;
    }

    private void Run(string name, List<Case> cases, uint floatMode = ShaderCompileRequest.DefaultFloatMode)
    {
        var words = new List<uint>();
        var slot = 0u;
        foreach (var @case in cases)
        {
            Assert.Equal(@case.Results.Length, @case.Expected.Length);
            words.AddRange(@case.Words);
            foreach (var register in @case.Results)
            {
                Assert.True(slot < 1024, "too many result slots for 12-bit buffer offsets");
                words.AddRange([0xE0700000u | (slot * 4), 0x80020000u | (register << 8)]);
                slot++;
            }
        }

        words.Add(EndProgram);
        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(
            new Gen5ComputeFixture(name, [.. words], StoreScalarResourceBase: 8, StoreBackingBytes: (int)slot * 4),
            floatMode: floatMode);
        if (!RequireDevice())
        {
            return;
        }

        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var error), error);
        var buffer = new byte[slot * 4];
        buffer.AsSpan().Fill(0xCD);
        Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, buffer,
            new byte[PushData.ByteSize], shader.ArgumentLayout!, 32, out var result, out error), error);

        var failures = new List<string>();
        var index = 0;
        foreach (var @case in cases)
        {
            for (var result_ = 0; result_ < @case.Results.Length; result_++, index++)
            {
                var actual = BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan(index * 4)) & @case.Mask;
                var expected = @case.Expected[result_] & @case.Mask;
                if (actual != expected)
                {
                    failures.Add($"{@case.Label} v{@case.Results[result_]}: expected 0x{expected:X8}, got 0x{actual:X8}");
                }
            }
        }

        foreach (var failure in failures)
        {
            output.WriteLine(failure);
        }

        Assert.True(failures.Count == 0, $"{failures.Count} of {index} results differ:\n{string.Join("\n", failures)}");
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
}
