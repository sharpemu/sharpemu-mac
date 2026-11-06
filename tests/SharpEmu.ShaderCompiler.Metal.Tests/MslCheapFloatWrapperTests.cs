// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using Xunit;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

public sealed class MslCheapFloatWrapperTests
{
    // Each lane loads one f32, applies the opcode and stores the result over it.
    private static uint[] UnaryProgram(uint vop1) =>
    [
        0xD7650000, 0x000100C1, // v_mbcnt_lo_u32_b32 v0, -1, 0
        0xD7660000, 0x000200C1, // v_mbcnt_hi_u32_b32 v0, -1, v0
        0x34080082,             // v_lshlrev_b32 v4, 2, v0
        0xE0301000, 0x80020104, // buffer_load_dword v1, v4, s[8:11], 0 offen
        0xBF8C3F70,             // s_waitcnt vmcnt(0)
        vop1,                   // v_<op>_f32 v2, v1
        0xE0701000, 0x80020204, // buffer_store_dword v2, v4, s[8:11], 0 offen
        0xBF810000,             // s_endpgm
    ];

    private static readonly uint[] Inputs =
    [
        0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x40800000, 0xC0800000,
        0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000, 0x80800000,
        0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7FC00001, 0x7F800001,
        0xFF812345, 0x7F7FFFFF, 0xFF7FFFFF, 0x3E800000, 0xBE800000, 0x4B800000,
    ];

    [Theory]
    [InlineData("sqrt", 0x7E046701u)]
    [InlineData("rsq", 0x7E045D01u)]
    [InlineData("log", 0x7E044F01u)]
    public void CheapWrappers_GiveTheSameBitsAsThePreviousForms(string name, uint vop1)
    {
        var fixture = new Gen5ComputeFixture($"cheap-float-{name}", UnaryProgram(vop1), 8, 64 * 4);
        var request = Gen5ComputeFixtures.CreateComputeRequest(fixture, 64, 64);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var cheap, out var error,
            new Gen5MslCodegenOptions()), error);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var before, out error,
            new Gen5MslCodegenOptions { CheapFloatWrappers = false }), error);
        Assert.Contains("(bits & 0x7F800000u) == 0u", cheap.Source);
        Assert.DoesNotContain("sharpemu_invalid_nan_f32(", Body(cheap.Source));
        Assert.Contains("sharpemu_invalid_nan_f32(", Body(before.Source));
        Assert.DoesNotContain("(bits & 0x7F800000u) == 0u", before.Source);

        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            return;
        }

        Assert.True(MetalNative.IsAvailable, "A Metal device is required.");
        var input = new byte[fixture.StoreBackingBytes];
        for (var lane = 0; lane < 64; lane++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(input.AsSpan(lane * 4, 4), Inputs[lane % Inputs.Length]);
        }

        var expected = Execute(before);
        var actual = Execute(cheap);
        for (var lane = 0; lane < 64; lane++)
        {
            Assert.True(expected.AsSpan(lane * 4, 4).SequenceEqual(actual.AsSpan(lane * 4, 4)),
                $"{name}(0x{Inputs[lane % Inputs.Length]:X8}): before 0x{BinaryPrimitives.ReadUInt32LittleEndian(expected.AsSpan(lane * 4, 4)):X8}, " +
                $"cheap 0x{BinaryPrimitives.ReadUInt32LittleEndian(actual.AsSpan(lane * 4, 4)):X8}");
        }

        // The ISA's invalid result for a negative number, and -0 or a negative denormal kept a zero.
        Assert.Equal(0xFFC00000u, BinaryPrimitives.ReadUInt32LittleEndian(actual.AsSpan(3 * 4, 4)));
        if (name == "sqrt")
        {
            Assert.Equal(0x80000000u, BinaryPrimitives.ReadUInt32LittleEndian(actual.AsSpan(1 * 4, 4)));
            Assert.Equal(0x80000000u, BinaryPrimitives.ReadUInt32LittleEndian(actual.AsSpan(7 * 4, 4)));
        }

        byte[] Execute(Gen5MslShader shader)
        {
            Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var compileError), compileError);
            Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, input.ToArray(),
                new byte[Resources.PushData.ByteSize], shader.ArgumentLayout!, 64,
                out var result, out var runError), runError);
            return result;
        }
    }

    private static string Body(string source) => source[source.IndexOf(" gen5_cs(", StringComparison.Ordinal)..];
}
