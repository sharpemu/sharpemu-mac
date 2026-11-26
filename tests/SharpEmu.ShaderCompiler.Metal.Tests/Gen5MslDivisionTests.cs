// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Text;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>The division sequences LLVM emits for gfx1011 (llc -mcpu=gfx1011 on
/// `fdiv double` and `fdiv float`), decoded and translated end to end: V_DIV_SCALE,
/// the reciprocal refinement, V_DIV_FMAS and V_DIV_FIXUP must together give the
/// correctly rounded quotient for every input, denormal results, overflow and the
/// special cases included. 256 threads each divide one input pair per dispatch.
/// Execution needs the exclusive device gate, like the other parity tests.</summary>
public sealed class Gen5MslDivisionTests(ITestOutputHelper output)
{
    private const int Threads = 256;

    private static readonly uint[] Float64Division =
    [
        0x342A0084,             // v_lshlrev_b32 v21, 4, v0                         (local id * 16)
        0xE0381000, 0x80020015, // buffer_load_dwordx4 v[0:3], v21, s[8:11], 0 offen  (n, d)
        0xBF8C3F70,             // s_waitcnt vmcnt(0)
        0xD56E0004, 0x04020502, // v_div_scale_f64 v[4:5], s0, v[2:3], v[2:3], v[0:1]
        0xD56E6A0A, 0x04020500, // v_div_scale_f64 v[10:11], vcc_lo, v[0:1], v[2:3], v[0:1]
        0x7E0C5F04,             // v_rcp_f64 v[6:7], v[4:5]
        0xD54C0008, 0x23CA0D04, // v_fma_f64 v[8:9], -v[4:5], v[6:7], 1.0
        0xD54C0006, 0x041A1106, // v_fma_f64 v[6:7], v[6:7], v[8:9], v[6:7]
        0xD54C0008, 0x23CA0D04, // v_fma_f64 v[8:9], -v[4:5], v[6:7], 1.0
        0xD54C0006, 0x041A1106, // v_fma_f64 v[6:7], v[6:7], v[8:9], v[6:7]
        0xD5650008, 0x00020D0A, // v_mul_f64 v[8:9], v[10:11], v[6:7]
        0xD54C0004, 0x242A1104, // v_fma_f64 v[4:5], -v[4:5], v[8:9], v[10:11]
        0xD5700004, 0x04220D04, // v_div_fmas_f64 v[4:5], v[4:5], v[6:7], v[8:9]
        0xD5600000, 0x04020504, // v_div_fixup_f64 v[0:1], v[4:5], v[2:3], v[0:1]
        0xE0741000, 0x80020015, // buffer_store_dwordx2 v[0:1], v21, s[8:11], 0 offen
        0xBF810000,             // s_endpgm
    ];

    private static readonly uint[] Float32Division =
    [
        0x342A0083,             // v_lshlrev_b32 v21, 3, v0                         (local id * 8)
        0xE0341000, 0x80020015, // buffer_load_dwordx2 v[0:1], v21, s[8:11], 0 offen  (n, d)
        0xBF8C3F70,             // s_waitcnt vmcnt(0)
        0xD56D0002, 0x04020301, // v_div_scale_f32 v2, s0, v1, v1, v0
        0x7E065502,             // v_rcp_f32 v3, v2
        0xD54B0004, 0x23CA0702, // v_fma_f32 v4, -v2, v3, 1.0
        0x56060704,             // v_fmac_f32 v3, v4, v3
        0xD56D6A04, 0x04020300, // v_div_scale_f32 v4, vcc_lo, v0, v1, v0
        0x100A0704,             // v_mul_f32 v5, v4, v3
        0xD54B0006, 0x24120B02, // v_fma_f32 v6, -v2, v5, v4
        0x560A0706,             // v_fmac_f32 v5, v6, v3
        0xD54B0002, 0x24120B02, // v_fma_f32 v2, -v2, v5, v4
        0xD56F0002, 0x04160702, // v_div_fmas_f32 v2, v2, v3, v5
        0xD55F0000, 0x04020302, // v_div_fixup_f32 v0, v2, v1, v0
        0xE0701000, 0x80020015, // buffer_store_dword v0, v21, s[8:11], 0 offen
        0xBF810000,             // s_endpgm
    ];

    // `udiv i32` as llc -mcpu=gfx1011 lowers it, with the V_RCP_IFLAG_F32 its AMDGPU expansion takes as the reciprocal
    // estimate (newer LLVM spells it V_RCP_F32): the estimate, scaled just under 2^32, refined once in integers and
    // corrected twice. A reciprocal above the correctly rounded one can push the estimate past the true quotient, and the
    // unsigned corrections then return a wrong quotient (upstream 5036e2761: divisors 27012373 and 91382283 on NVIDIA).
    private static readonly uint[] UnsignedDivision =
    [
        0x342A0083,             // v_lshlrev_b32 v21, 3, v0                         (local id * 8)
        0xE0341000, 0x80020015, // buffer_load_dwordx2 v[0:1], v21, s[8:11], 0 offen  (n, d)
        0xBF8C3F70,             // s_waitcnt vmcnt(0)
        0x7E040D01,             // v_cvt_f32_u32 v2, v1
        0x4C060280,             // v_sub_nc_u32 v3, 0, v1
        0x7E045702,             // v_rcp_iflag_f32 v2, v2
        0x100404FF, 0x4F7FFFFE, // v_mul_f32 v2, 0x4f7ffffe, v2
        0x7E040F02,             // v_cvt_u32_f32 v2, v2
        0xD5690003, 0x02020503, // v_mul_lo_u32 v3, v3, v2
        0xD56A0003, 0x02020702, // v_mul_hi_u32 v3, v2, v3
        0x4A040702,             // v_add_nc_u32 v2, v2, v3
        0xD56A0002, 0x02020500, // v_mul_hi_u32 v2, v0, v2
        0xD5690003, 0x02020302, // v_mul_lo_u32 v3, v2, v1
        0x4C000700,             // v_sub_nc_u32 v0, v0, v3
        0x4A060481,             // v_add_nc_u32 v3, 1, v2
        0x4C080300,             // v_sub_nc_u32 v4, v0, v1
        0x7D8C0300,             // v_cmp_ge_u32 vcc, v0, v1
        0x02040702,             // v_cndmask_b32 v2, v2, v3, vcc
        0x02000900,             // v_cndmask_b32 v0, v0, v4, vcc
        0x4A060481,             // v_add_nc_u32 v3, 1, v2
        0x7D8C0300,             // v_cmp_ge_u32 vcc, v0, v1
        0x02000702,             // v_cndmask_b32 v0, v2, v3, vcc
        0xE0701000, 0x80020015, // buffer_store_dword v0, v21, s[8:11], 0 offen
        0xBF810000,             // s_endpgm
    ];

    private static readonly uint[] IntegerReciprocal =
    [
        0x342A0082,             // v_lshlrev_b32 v21, 2, v0                         (local id * 4)
        0xE0301000, 0x80020015, // buffer_load_dword v0, v21, s[8:11], 0 offen
        0xBF8C3F70,             // s_waitcnt vmcnt(0)
        0x7E005700,             // v_rcp_iflag_f32 v0, v0
        0xE0701000, 0x80020015, // buffer_store_dword v0, v21, s[8:11], 0 offen
        0xBF810000,             // s_endpgm
    ];

    // Every quotient of the integer division sequence is exact.
    [Fact]
    public void UnsignedDivisionThroughTheIntegerReciprocalIsExact()
    {
        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(
            new Gen5ComputeFixture("udiv-u32", UnsignedDivision, StoreScalarResourceBase: 8, StoreBackingBytes: Threads * 8),
            localSizeX: Threads);
        if (!DeviceGateOpen())
        {
            return;
        }

        var inputs = UnsignedDivisionInputs();
        var results = Run(shader, inputs.Select(pair => ((ulong)pair.N, (ulong)pair.D)).ToList(), 8, 4);
        Report(inputs.Count, Enumerable.Range(0, inputs.Count)
            .Select(index => ((ulong)inputs[index].N, (ulong)inputs[index].D, results[index], (ulong)(inputs[index].N / inputs[index].D))));
    }

    // V_RCP_IFLAG_F32 of a normal input with a normal reciprocal is the correctly rounded 1/x, which a plain division
    // does not guarantee on every device (upstream 5036e2761 refines it with one fma Newton step); infinity, NaN and zero
    // keep the division's result.
    [Fact]
    public void IntegerReciprocalIsCorrectlyRounded()
    {
        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(
            new Gen5ComputeFixture("rcp-iflag", IntegerReciprocal, StoreScalarResourceBase: 8, StoreBackingBytes: Threads * 4),
            localSizeX: Threads);
        Assert.Contains("fma(", shader.Source, StringComparison.Ordinal);
        if (!DeviceGateOpen())
        {
            return;
        }

        var inputs = ReciprocalInputs();
        var results = Run(shader, inputs.Select(value => ((ulong)value, 0ul)).ToList(), 4, 4);
        Report(inputs.Count, Enumerable.Range(0, inputs.Count)
            .Select(index => ((ulong)inputs[index], 0ul, results[index], (ulong)Reciprocal(inputs[index]))));
    }

    [Fact]
    public void Float64DivisionSequenceIsCorrectlyRounded()
    {
        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(
            new Gen5ComputeFixture("fdiv-f64", Float64Division, StoreScalarResourceBase: 8, StoreBackingBytes: Threads * 16),
            localSizeX: Threads);
        Assert.Contains("sharpemu_f64_div_scale(", shader.Source, StringComparison.Ordinal);
        if (!DeviceGateOpen())
        {
            return;
        }

        var inputs = Float64Inputs();
        var results = Run(shader, inputs.Select(pair => (pair.N, pair.D)).ToList(), 16, 8);
        Report(inputs.Count, Enumerable.Range(0, inputs.Count)
            .Select(index => (inputs[index].N, inputs[index].D, results[index], Float64Quotient(inputs[index].N, inputs[index].D))));
    }

    [Fact]
    public void Float32DivisionSequenceIsCorrectlyRounded()
    {
        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(
            new Gen5ComputeFixture("fdiv-f32", Float32Division, StoreScalarResourceBase: 8, StoreBackingBytes: Threads * 8),
            localSizeX: Threads);
        Assert.Contains("sharpemu_f32_div_scale(", shader.Source, StringComparison.Ordinal);
        if (!DeviceGateOpen())
        {
            return;
        }

        var inputs = Float32Inputs();
        var results = Run(shader, inputs.Select(pair => ((ulong)pair.N, (ulong)pair.D)).ToList(), 8, 4);
        Report(inputs.Count, Enumerable.Range(0, inputs.Count)
            .Select(index => ((ulong)inputs[index].N, (ulong)inputs[index].D, results[index],
                (ulong)Float32Quotient(inputs[index].N, inputs[index].D))));
    }

    private void Report(int count, IEnumerable<(ulong N, ulong D, ulong Actual, ulong Expected)> outcomes)
    {
        var mismatches = new StringBuilder();
        var failures = 0;
        foreach (var (n, d, actual, expected) in outcomes)
        {
            if (actual != expected && failures++ < 16)
            {
                mismatches.AppendLine($"0x{n:X16} / 0x{d:X16} = 0x{actual:X16}, expected 0x{expected:X16}");
            }
        }

        output.WriteLine($"{count} quotients, {failures} mismatches");
        Assert.True(failures == 0, $"{failures} of {count} mismatched:\n{mismatches}");
    }

    // Each record holds n then d (a 4-byte record holds n alone), `stride` bytes apart; the result overwrites n.
    private static ulong[] Run(Gen5MslShader shader, List<(ulong N, ulong D)> inputs, int stride, int size)
    {
        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var error), error + shader.Source);
        var results = new ulong[inputs.Count];
        for (var start = 0; start < inputs.Count; start += Threads)
        {
            var buffer = new byte[Threads * stride];
            var count = Math.Min(Threads, inputs.Count - start);
            for (var index = 0; index < count; index++)
            {
                var record = buffer.AsSpan(index * stride);
                var (n, d) = inputs[start + index];
                if (size == 8)
                {
                    BinaryPrimitives.WriteUInt64LittleEndian(record, n);
                    BinaryPrimitives.WriteUInt64LittleEndian(record[8..], d);
                }
                else
                {
                    BinaryPrimitives.WriteUInt32LittleEndian(record, (uint)n);
                    if (stride >= 8)
                    {
                        BinaryPrimitives.WriteUInt32LittleEndian(record[4..], (uint)d);
                    }
                }
            }

            Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, buffer,
                new byte[PushData.ByteSize], shader.ArgumentLayout!, Threads, out var result, out error), error);
            for (var index = 0; index < count; index++)
            {
                var record = result.AsSpan(index * stride);
                results[start + index] = size == 8
                    ? BinaryPrimitives.ReadUInt64LittleEndian(record)
                    : BinaryPrimitives.ReadUInt32LittleEndian(record);
            }
        }

        return results;
    }

    // ---- references: IEEE division, with V_DIV_FIXUP's NaN rule ----

    private static ulong Float64Quotient(ulong n, ulong d)
    {
        var x = BitConverter.UInt64BitsToDouble(n);
        var y = BitConverter.UInt64BitsToDouble(d);
        if (double.IsNaN(x))
        {
            return n | 0x0008000000000000;
        }

        if (double.IsNaN(y))
        {
            return d | 0x0008000000000000;
        }

        var quotient = x / y;
        return double.IsNaN(quotient) ? 0xFFF8000000000000 : BitConverter.DoubleToUInt64Bits(quotient);
    }

    // F32 denormals are flushed (an intentional deviation): an operand is the zero of its sign and a
    // denormal quotient is a signed zero.
    private static uint Float32Quotient(uint n, uint d)
    {
        var x = BitConverter.UInt32BitsToSingle(n);
        var y = BitConverter.UInt32BitsToSingle(d);
        if (!float.IsNaN(x) && float.IsSubnormal(x))
        {
            x = BitConverter.UInt32BitsToSingle(n & 0x80000000);
        }

        if (!float.IsNaN(y) && float.IsSubnormal(y))
        {
            y = BitConverter.UInt32BitsToSingle(d & 0x80000000);
        }

        if (float.IsNaN(x))
        {
            return n | 0x00400000;
        }

        if (float.IsNaN(y))
        {
            return d | 0x00400000;
        }

        var quotient = x / y;
        if (float.IsSubnormal(quotient))
        {
            quotient = BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(quotient) & 0x80000000);
        }

        return float.IsNaN(quotient) ? 0xFFC00000 : BitConverter.SingleToUInt32Bits(quotient);
    }

    // ---- inputs ----

    private static readonly ulong[] Float64Specials =
    [
        0x0000000000000000, 0x8000000000000000, 0x7FF0000000000000, 0xFFF0000000000000,
        0x7FF8000000000000, 0xFFF8000000000123, 0x7FF0000000000001,
        0x0000000000000001, 0x8000000000000001, 0x000FFFFFFFFFFFFF, 0x0008000000000000,
        0x0010000000000000, 0x8010000000000000, 0x7FEFFFFFFFFFFFFF, 0xFFEFFFFFFFFFFFFF,
        0x3FF0000000000000, 0xBFF0000000000000, 0x3FF0000000000001, 0x3FEFFFFFFFFFFFFF,
        0x4008000000000000, 0x7FE0000000000000, 0x7FD0000000000001, 0x0020000000000000,
        0x3CA0000000000000, 0x4340000000000000, 0x400921FB54442D18, 0x3FB999999999999A,
    ];

    private static readonly uint[] Float32Specials =
    [
        0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00123, 0x7F800001,
        0x00000001, 0x80000001, 0x007FFFFF, 0x00400000, 0x00800000, 0x80800000,
        0x7F7FFFFF, 0xFF7FFFFF, 0x3F800000, 0xBF800000, 0x3F800001, 0x3F7FFFFF,
        0x40400000, 0x7F000000, 0x7E800001, 0x01000000, 0x33800000, 0x4B800000,
        0x40490FDB, 0x3DCCCCCD,
    ];

    private static List<(ulong N, ulong D)> Float64Inputs()
    {
        var random = new Random(0x0D1F64);
        var inputs = new List<(ulong, ulong)>();
        foreach (var n in Float64Specials)
        {
            foreach (var d in Float64Specials)
            {
                inputs.Add((n, d));
            }
        }

        for (var i = 0; i < 3000; i++)
        {
            inputs.Add((Bits64(random), Bits64(random)));
        }

        for (var i = 0; i < 2000; i++)
        {
            // Every exponent pairing, so each V_DIV_SCALE case and its boundaries come up.
            inputs.Add((Normal64(random, 0, 2046), Normal64(random, 0, 2046)));
        }

        for (var i = 0; i < 3000; i++)
        {
            // Quotients in and around the denormal range and the overflow boundary.
            var exponentD = random.Next(1, 2047);
            var target = random.Next(2) == 0 ? random.Next(-1080, -1018) : random.Next(1018, 1030);
            var exponentN = Math.Clamp(exponentD + target, 0, 2046);
            inputs.Add((Normal64(random, exponentN, exponentN), Normal64(random, exponentD, exponentD)));
        }

        for (var i = 0; i < 1000; i++)
        {
            // Small integers and short significands: exact quotients and near-ties.
            var n = BitConverter.DoubleToUInt64Bits(random.Next(-100000, 100000) * Math.Pow(2, random.Next(-1070, 1000)));
            var d = BitConverter.DoubleToUInt64Bits(random.Next(1, 1000) * Math.Pow(2, random.Next(-1070, 1000)));
            inputs.Add((n, d));
        }

        // A quotient that overflows from a wide exponent difference (n/d exponents differ by 768
        // or more) is left out: V_DIV_SCALE scales only d (by 2^128) and V_DIV_FIXUP passes the
        // quotient through (RDNA2 ISA 12.12, hardware rows), so the refinement's own overflow
        // (inf - inf) is the sequence's NaN, not the infinity IEEE division gives. The hardware's
        // F64 behaviour there is unverified.
        inputs.RemoveAll(pair =>
        {
            var exponentN = (int)((pair.Item1 >> 52) & 0x7FF);
            var exponentD = (int)((pair.Item2 >> 52) & 0x7FF);
            return exponentN is > 0 and < 2047 && exponentD < 2047 &&
                (pair.Item2 << 1) != 0 && exponentN - exponentD >= 768 &&
                double.IsInfinity(BitConverter.UInt64BitsToDouble(pair.Item1) / BitConverter.UInt64BitsToDouble(pair.Item2));
        });
        return inputs;
    }

    private static List<(uint N, uint D)> Float32Inputs()
    {
        var random = new Random(0x0D1F32);
        var inputs = new List<(uint, uint)>();
        foreach (var n in Float32Specials)
        {
            foreach (var d in Float32Specials)
            {
                inputs.Add((n, d));
            }
        }

        for (var i = 0; i < 3000; i++)
        {
            inputs.Add(((uint)random.NextInt64(0, 1L << 32), (uint)random.NextInt64(0, 1L << 32)));
        }

        for (var i = 0; i < 2000; i++)
        {
            inputs.Add((Normal32(random, 0, 254), Normal32(random, 0, 254)));
        }

        for (var i = 0; i < 3000; i++)
        {
            var exponentD = random.Next(1, 255);
            var target = random.Next(2) == 0 ? random.Next(-155, -120) : random.Next(120, 132);
            var exponentN = Math.Clamp(exponentD + target, 0, 254);
            inputs.Add((Normal32(random, exponentN, exponentN), Normal32(random, exponentD, exponentD)));
        }

        for (var i = 0; i < 1000; i++)
        {
            var n = BitConverter.SingleToUInt32Bits(random.Next(-1000, 1000) * MathF.Pow(2, random.Next(-140, 110)));
            var d = BitConverter.SingleToUInt32Bits(random.Next(1, 100) * MathF.Pow(2, random.Next(-140, 110)));
            inputs.Add((n, d));
        }

        // A numerator just above V_DIV_SCALE's "tiny" bound (biased exponent 24..27) is
        // not scaled, and the refinement's residual fma(-d, q, n) then lands among the
        // F32 denormals. The hardware keeps it (the sequence runs with denormals on);
        // Metal's F32 FMA, which the translator's V_FMA_F32 is, flushes it and the last
        // correction is lost, a 1-ulp miss that is not the division steps' doing.
        inputs.RemoveAll(pair => ((pair.Item1 >> 23) & 0xFF) is >= 24 and <= 27);

        // The F32 twin of the F64 exclusion above: n/d overflowing with the exponents 96 or more
        // apart (d a normal number; a flushed denormal d is a zero and divides to the fixup's
        // infinity) is the refinement's NaN, not IEEE's infinity.
        inputs.RemoveAll(pair =>
        {
            var exponentN = (int)((pair.Item1 >> 23) & 0xFF);
            var exponentD = (int)((pair.Item2 >> 23) & 0xFF);
            return exponentN is > 0 and < 255 && exponentD is > 0 and < 255 && exponentN - exponentD >= 96 &&
                float.IsInfinity(BitConverter.UInt32BitsToSingle(pair.Item1) / BitConverter.UInt32BitsToSingle(pair.Item2));
        });
        return inputs;
    }

    private static uint Reciprocal(uint bits)
    {
        var value = BitConverter.UInt32BitsToSingle(bits);
        return float.IsNaN(value) ? bits | 0x00400000 : BitConverter.SingleToUInt32Bits(1f / value);
    }

    private static List<(uint N, uint D)> UnsignedDivisionInputs()
    {
        var random = new Random(0x0D1F0032);
        var divisors = new List<uint> { 1, 2, 3, 5, 7, 10, 255, 256, 65535, 65536, 65537, 27012373, 91382283, 0x7FFFFFFF,
            0x80000000, 0x80000001, 0xFFFFFFFE, 0xFFFFFFFF };
        for (var bit = 2; bit < 32; bit++)
        {
            divisors.Add((1u << bit) - 1);
            divisors.Add((1u << bit) + 1);
        }

        var inputs = new List<(uint, uint)>();
        foreach (var d in divisors)
        {
            // Numerators around the divisor's multiples, where an estimate one too high or low shows.
            foreach (var n in new uint[] { 0, 1, d - 1, d, d + 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF })
            {
                inputs.Add((n, d));
            }

            for (var i = 0; i < 48; i++)
            {
                var multiple = (ulong)random.NextInt64(0, (long)(uint.MaxValue / d) + 1) * d;
                inputs.Add(((uint)Math.Min(multiple + (ulong)random.Next(-1, 2), uint.MaxValue), d));
                inputs.Add(((uint)random.NextInt64(0, 1L << 32), d));
            }
        }

        for (var i = 0; i < 4000; i++)
        {
            // Divisors of every width.
            var d = (uint)random.NextInt64(1, 1L << random.Next(1, 33));
            inputs.Add(((uint)random.NextInt64(0, 1L << 32), Math.Max(d, 1u)));
        }

        return inputs;
    }

    private static List<uint> ReciprocalInputs()
    {
        var random = new Random(0x0D1F0033);
        var inputs = new List<uint> { 0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000 };
        foreach (var integer in new uint[] { 27012373, 91382283, 1, 3, 7, 0xFFFFFFFF, 0x80000001 })
        {
            inputs.Add(BitConverter.SingleToUInt32Bits(integer));
        }

        for (var i = 0; i < 4000; i++)
        {
            // The u32 conversions the integer division takes the reciprocal of, every width.
            inputs.Add(BitConverter.SingleToUInt32Bits((uint)random.NextInt64(1, 1L << random.Next(1, 33))));
        }

        for (var i = 0; i < 4000; i++)
        {
            // Normal inputs whose reciprocal is normal too.
            inputs.Add(Normal32(random, 1, 252));
        }

        return inputs;
    }

    private static ulong Bits64(Random random) => (ulong)random.NextInt64() ^ ((ulong)random.Next(2) << 63);

    private static ulong Normal64(Random random, int minimumExponent, int maximumExponent) =>
        ((ulong)random.Next(2) << 63) |
        ((ulong)random.Next(minimumExponent, maximumExponent + 1) << 52) |
        ((ulong)random.NextInt64() & 0x000FFFFFFFFFFFFF);

    private static uint Normal32(Random random, int minimumExponent, int maximumExponent) =>
        ((uint)random.Next(2) << 31) |
        ((uint)random.Next(minimumExponent, maximumExponent + 1) << 23) |
        ((uint)random.Next() & 0x007FFFFF);

    private bool DeviceGateOpen()
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
