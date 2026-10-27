// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>
/// V_MIN/MAX/MIN3/MAX3/MED3_F32 lowered without the bit-level helpers
/// (<see cref="Gen5MslCodegenOptions.NativeFloatMinMax"/>) give the helpers' bits for every
/// pair of a sweep of special values: zeros of both signs, denormals, infinities, quiet and
/// signalling NaNs with payloads, and ordinary numbers. The operands come from a buffer, so
/// nothing is folded at compile time; each variant runs with the switch on and off, in the
/// flushing FLOAT_MODE and in the one that keeps f32 denormals, and the outputs must match.
/// </summary>
public sealed class MslNativeMinMaxSweepTests(ITestOutputHelper output)
{
    private const int Lanes = 1024;

    private static readonly uint[] Values =
    [
        0x00000000u, 0x80000000u, 0x3F800000u, 0xBF800000u, 0x3F000000u, 0xBF000000u, 0x7F800000u, 0xFF800000u,
        0x7F7FFFFFu, 0xFF7FFFFFu, 0x00800000u, 0x80800000u, 0x00000001u, 0x80000001u, 0x007FFFFFu, 0x807FFFFFu,
        0x7FC00000u, 0xFFC00000u, 0x7F800001u, 0xFF800002u, 0x7FC12345u, 0x7FFFFFFFu, 0x40000000u, 0xC0000000u,
        0x0DA24260u, 0x8DA24260u, 0x3F800001u, 0x3F7FFFFFu, 0x40490FDBu, 0xC0490FDBu, 0x4B000000u, 0x00400000u,
    ];

    // v1..v3 are read from the buffer, the inserted instruction writes v4 (see Fixture).
    public static readonly TheoryData<string, uint[]> Variants = new()
    {
        { "v_max_f32_e32 v4, v1, v2", [0x20080501u] },
        { "v_min_f32_e32 v4, v1, v2", [0x1E080501u] },
        { "v_max_f32_e64 v4, v1, v2 clamp", [0xD5108004u, 0x02020501u] },
        { "v_max_f32_e64 v4, -v1, |v2|", [0xD5100204u, 0x22020501u] },
        { "v_max3_f32 v4, v1, v2, v3", [0xD5540004u, 0x040E0501u] },
        { "v_min3_f32 v4, v1, v2, v3", [0xD5510004u, 0x040E0501u] },
        { "v_med3_f32 v4, v1, v2, v3", [0xD5570004u, 0x040E0501u] },
        { "v_max_f32_e32 v4, 0, v1", [0x20080280u] },
        { "v_min_f32_e32 v4, 0, v1", [0x1E080280u] },
        { "v_max_f32_e32 v4, 0x80000000, v1", [0x200802FFu, 0x80000000u] },
        { "v_min_f32_e32 v4, 0x80000000, v1", [0x1E0802FFu, 0x80000000u] },
        { "v_max_f32_e64 v4, neg(0), v1", [0xD5100004u, 0x22020280u] },
        { "v_min_f32_e64 v4, neg(0), v1", [0xD50F0004u, 0x22020280u] },
        { "v_max_f32_e32 v4, 1.0, v1", [0x200802F2u] },
        { "v_min_f32_e32 v4, 1.0, v1", [0x1E0802F2u] },
        { "v_max_f32_e32 v4, -1.0, v1", [0x200802F3u] },
        { "v_min_f32_e32 v4, -1.0, v1", [0x1E0802F3u] },
        { "v_max_f32_e32 v4, 0x7f800000, v1", [0x200802FFu, 0x7F800000u] },
        { "v_min_f32_e32 v4, 0x7f800000, v1", [0x1E0802FFu, 0x7F800000u] },
        { "v_max_f32_e32 v4, 0xff800000, v1", [0x200802FFu, 0xFF800000u] },
        { "v_min_f32_e32 v4, 0xff800000, v1", [0x1E0802FFu, 0xFF800000u] },
        { "v_max_f32_e32 v4, 1, v1", [0x20080281u] },
        { "v_min_f32_e32 v4, 0x80000001, v1", [0x1E0802FFu, 0x80000001u] },
        { "v_max_f32_e32 v4, 0x7f7fffff, v1", [0x200802FFu, 0x7F7FFFFFu] },
        { "v_max_f32_e64 v4, v1, 0", [0xD5100004u, 0x02010101u] },
        { "v_max_f32_e64 v4, |v1|, 0", [0xD5100104u, 0x02010101u] },
        { "v_min_f32_e64 v4, -v1, 0", [0xD50F0004u, 0x22010101u] },
        { "v_max_f32_e64 v4, 0, v1 clamp", [0xD5108004u, 0x02020280u] },
        { "v_max3_f32 v4, v1, v2, 0", [0xD5540004u, 0x02020501u] },
        { "v_min3_f32 v4, v1, 1.0, v2", [0xD5510004u, 0x0409E501u] },
        { "v_max3_f32 v4, 0, v1, -1.0", [0xD5540004u, 0x03CE0280u] },
        { "v_med3_f32 v4, v1, 0, 1.0", [0xD5570004u, 0x03C90101u] },
        { "v_med3_f32 v4, v1, -1.0, 1.0", [0xD5570004u, 0x03C9E701u] },
        { "v_med3_f32 v4, 0, v1, 1.0", [0xD5570004u, 0x03CA0280u] },
        { "v_med3_f32 v4, 1.0, 0, v1", [0xD5570004u, 0x040500F2u] },
        { "v_med3_f32 v4, v1, 0x80000000, 1.0", [0xD5570004u, 0x03C9FF01u, 0x80000000u] },
        { "v_med3_f32 v4, v1, -1.0, 0x80000000", [0xD5570004u, 0x03FDE701u, 0x80000000u] },
        { "v_med3_f32 v4, v1, 1.0, 1.0", [0xD5570004u, 0x03C9E501u] },
        { "v_med3_f32 v4, -v1, 0, 1.0", [0xD5570004u, 0x23C90101u] },
        { "v_med3_f32 v4, v1, 0.5, 2.0", [0xD5570004u, 0x03D1E101u] },
    };

    private static Gen5ComputeFixture Fixture(uint[] operation) => new(
        "minmax-sweep",
        [
            0x34140082u,              // v_lshlrev_b32 v10, 2, v0
            0xE0301000u, 0x8002010Au, // buffer_load_dword v1, v10, s[8:11], 0 offen
            0x4A1614FFu, 0x00001000u, // v_add_nc_u32 v11, 0x1000, v10
            0xE0301000u, 0x8002020Bu, // buffer_load_dword v2, v11, s[8:11], 0 offen
            0x4A1614FFu, 0x00002000u, // v_add_nc_u32 v11, 0x2000, v10
            0xE0301000u, 0x8002030Bu, // buffer_load_dword v3, v11, s[8:11], 0 offen
            0xBF8C3F70u,              // s_waitcnt vmcnt(0)
            .. operation,
            0x4A1614FFu, 0x00003000u, // v_add_nc_u32 v11, 0x3000, v10
            0xE0701000u, 0x8002040Bu, // buffer_store_dword v4, v11, s[8:11], 0 offen
            0xBF810000u,              // s_endpgm
        ],
        StoreScalarResourceBase: 8,
        StoreBackingBytes: 4 * Lanes * sizeof(uint));

    private static Gen5MslShader Compile(uint[] operation, uint floatMode, bool native)
    {
        var request = Gen5ComputeFixtures.CreateComputeRequest(Fixture(operation), localSizeX: Lanes, floatMode: floatMode);
        Assert.True(
            Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error, new Gen5MslCodegenOptions(NativeFloatMinMax: native)),
            error);
        return shader;
    }

    [Theory]
    [MemberData(nameof(Variants))]
    public void ConstantOperandsAndFlushedPairsSkipTheHelpers(string assembly, uint[] operation)
    {
        var native = Compile(operation, ShaderCompileRequest.DefaultFloatMode, native: true);
        var helper = Compile(operation, ShaderCompileRequest.DefaultFloatMode, native: false);
        // The prelude's own definitions and calls are in both sources alike.
        static int Calls(string source) =>
            source.Split("sharpemu_max_f32(").Length + source.Split("sharpemu_min_f32(").Length +
            source.Split("sharpemu_med3_f32(").Length;
        var keepsHelper = assembly is "v_med3_f32 v4, v1, v2, v3" or "v_med3_f32 v4, v1, 1.0, 1.0" or "v_med3_f32 v4, v1, -1.0, 0x80000000";
        if (keepsHelper)
        {
            Assert.Equal(Calls(helper.Source), Calls(native.Source));
        }
        else
        {
            Assert.True(Calls(native.Source) < Calls(helper.Source), assembly);
        }
    }

    [Theory]
    [MemberData(nameof(Variants))]
    public void NativeLoweringMatchesTheHelpersBitForBit(string assembly, uint[] operation)
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Numerical execution requires the exclusive Metal device gate.");
            return;
        }

        Assert.True(MetalNative.IsAvailable, "The required execution gate needs a Metal device.");
        var threeOperands = assembly.Contains("3_f32", StringComparison.Ordinal);
        foreach (var floatMode in new[] { ShaderCompileRequest.DefaultFloatMode, 0xF0u })
        {
            var native = Compile(operation, floatMode, native: true);
            var helper = Compile(operation, floatMode, native: false);
            for (var run = 0; run < (threeOperands ? Values.Length : 1); run++)
            {
                var buffer = new byte[4 * Lanes * sizeof(uint)];
                for (var lane = 0; lane < Lanes; lane++)
                {
                    BinaryPrimitives.WriteUInt32LittleEndian(buffer.AsSpan(lane * 4), Values[lane % Values.Length]);
                    BinaryPrimitives.WriteUInt32LittleEndian(buffer.AsSpan((Lanes + lane) * 4), Values[lane / Values.Length]);
                    BinaryPrimitives.WriteUInt32LittleEndian(buffer.AsSpan((2 * Lanes + lane) * 4), Values[(run + lane) % Values.Length]);
                }

                var expected = Execute(helper, buffer);
                var actual = Execute(native, buffer);
                for (var lane = 0; lane < Lanes; lane++)
                {
                    var want = BinaryPrimitives.ReadUInt32LittleEndian(expected.AsSpan((3 * Lanes + lane) * 4));
                    var got = BinaryPrimitives.ReadUInt32LittleEndian(actual.AsSpan((3 * Lanes + lane) * 4));
                    Assert.True(
                        want == got,
                        $"{assembly} float_mode=0x{floatMode:X2} a=0x{Values[lane % Values.Length]:X8} b=0x{Values[lane / Values.Length]:X8} " +
                        $"c=0x{Values[(run + lane) % Values.Length]:X8}: helper 0x{want:X8}, native 0x{got:X8}");
                }
            }
        }
    }

    public static readonly TheoryData<string, uint[]> Reciprocals = new()
    {
        { "v_rcp_f32 v4, v1", [0x7E085501u] },
        { "v_rsq_f32 v4, v1", [0x7E085D01u] },
        { "v_sqrt_f32 v4, v1", [0x7E086701u] },
    };

    // The fast forms (Gen5MslCodegenOptions.FastReciprocals) against the precise ones over
    // the special values and a spread of numbers of every exponent and sign: a special input
    // (zero, denormal, infinity, NaN, negative under a root) or a result that is not a normal
    // number must come out with the same bits, and a normal result within 1 ULP.
    [Theory]
    [MemberData(nameof(Reciprocals))]
    public void FastReciprocalsKeepTheSpecialResults(string assembly, uint[] operation)
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Numerical execution requires the exclusive Metal device gate.");
            return;
        }

        Assert.True(MetalNative.IsAvailable, "The required execution gate needs a Metal device.");
        foreach (var floatMode in new[] { ShaderCompileRequest.DefaultFloatMode, 0xF0u })
        {
            var request = Gen5ComputeFixtures.CreateComputeRequest(Fixture(operation), localSizeX: Lanes, floatMode: floatMode);
            Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var fast, out var error, new Gen5MslCodegenOptions(FastReciprocals: true)), error);
            Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var precise, out error, new Gen5MslCodegenOptions(FastReciprocals: false)), error);
            var squareRoot = assembly.StartsWith("v_sqrt", StringComparison.Ordinal);
            if (squareRoot)
                Assert.DoesNotContain("fast::sqrt(", fast.Source.Split("kernel void")[1], StringComparison.Ordinal);
            else
                Assert.Contains("fast::", fast.Source.Split("kernel void")[1], StringComparison.Ordinal);
            var worst = 0L;
            var state = 0x12345678u;
            for (var run = 0; run < 8; run++)
            {
                var inputs = new uint[Lanes];
                for (var lane = 0; lane < Lanes; lane++)
                {
                    state = state * 1664525u + 1013904223u;
                    inputs[lane] = run == 0 && lane < Values.Length ? Values[lane] : state;
                    // fast::sqrt gave 0x4EEFAD10 instead of 0x4EEFAD0E for this input.
                    if (run == 0 && lane == Values.Length)
                        inputs[lane] = 0x5E606496u;
                }

                var buffer = new byte[4 * Lanes * sizeof(uint)];
                for (var lane = 0; lane < Lanes; lane++)
                {
                    BinaryPrimitives.WriteUInt32LittleEndian(buffer.AsSpan(lane * 4), inputs[lane]);
                }

                var expected = Execute(precise, buffer);
                var actual = Execute(fast, buffer);
                for (var lane = 0; lane < Lanes; lane++)
                {
                    var want = BinaryPrimitives.ReadUInt32LittleEndian(expected.AsSpan((3 * Lanes + lane) * 4));
                    var got = BinaryPrimitives.ReadUInt32LittleEndian(actual.AsSpan((3 * Lanes + lane) * 4));
                    var input = inputs[lane];
                    var exponent = want & 0x7F800000u;
                    var special = (input & 0x7F800000u) is 0 or 0x7F800000u || exponent is 0 or 0x7F800000u ||
                        (input & 0x80000000u) != 0 && !assembly.StartsWith("v_rcp", StringComparison.Ordinal);
                    var ulps = Math.Abs((long)(int)(want & 0x7FFFFFFFu) - (int)(got & 0x7FFFFFFFu));
                    Assert.True(
                        special || squareRoot ? want == got : (want & 0x80000000u) == (got & 0x80000000u) && ulps <= 1,
                        $"{assembly} float_mode=0x{floatMode:X2} input=0x{input:X8}: precise 0x{want:X8}, fast 0x{got:X8}");
                    if (!special)
                    {
                        worst = Math.Max(worst, ulps);
                    }
                }
            }

            output.WriteLine($"{assembly} float_mode=0x{floatMode:X2}: worst {worst} ulp from the precise form");
        }
    }

    private static byte[] Execute(Gen5MslShader shader, byte[] buffer)
    {
        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var error), error);
        Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, buffer,
            new byte[PushData.ByteSize], shader.ArgumentLayout!, Lanes, out var result, out error), error);
        return result;
    }
}
