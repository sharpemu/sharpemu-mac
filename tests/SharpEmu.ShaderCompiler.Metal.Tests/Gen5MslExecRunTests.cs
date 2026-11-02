// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>
/// Consecutive lane-local VALU instructions under a narrowed EXEC share one "if (exec)" block
/// instead of a guard per write (Gen5MslTranslator.ExecGuard.cs), and still write only the lanes
/// EXEC has set: the masked-off lanes keep what they held before.
/// </summary>
public sealed class Gen5MslExecRunTests(ITestOutputHelper output)
{
    // Lanes 0-15 take the masked arm: v1 = lane + 1, v2 = 3 * v1, v5 = 7. Lanes 16-31 keep v1 = 100,
    // v2 = 200. Each lane stores v1 and v2 at lane * 8.
    private static readonly Gen5ComputeFixture MaskedRun = new(
        "exec-run-masked",
        [
            0xD7650000, 0x000100C1, // v_mbcnt_lo_u32_b32 v0, -1, 0         lane 0..31
            0x7E0202FF, 100,        // v_mov_b32 v1, 100
            0x7E0402FF, 200,        // v_mov_b32 v2, 200
            0xBE8403FF, 0x0000FFFF, // s_mov_b32 s4, 0x0000ffff
            0xBE863C04,             // s_and_saveexec_b32 s6, s4
            0x4A020081,             // v_add_nc_u32 v1, 1, v0
            0x16040283,             // v_mul_u32_u24 v2, 3, v1
            0x7E0A0287,             // v_mov_b32 v5, 7
            0xBEFE0306,             // s_mov_b32 exec_lo, s6
            0x34080083,             // v_lshlrev_b32 v4, 3, v0
            0xE0701000, 0x80020104, // buffer_store_dword v1, v4, s[8:11], 0 offen
            0xE0701004, 0x80020204, // buffer_store_dword v2, v4, s[8:11], 0 offen offset:4
            0xBF810000,             // s_endpgm
        ],
        StoreScalarResourceBase: 8,
        StoreBackingBytes: 256);

    [Fact]
    public void LaneLocalWritesUnderAMaskShareOneRun()
    {
        var source = Compile(new Gen5MslCodegenOptions()).Source;
        Assert.Contains("if (exec)\n", source.Replace("\r\n", "\n"), StringComparison.Ordinal);
        Assert.DoesNotContain("if (exec) { v1 =", source, StringComparison.Ordinal);
        Assert.DoesNotContain("if (exec) { v2 =", source, StringComparison.Ordinal);
    }

    [Fact]
    public void WithoutRunsEachWriteKeepsItsGuard()
    {
        var source = Compile(new Gen5MslCodegenOptions() with { ExecRuns = false }).Source;
        Assert.Contains("if (exec) { v1 =", source, StringComparison.Ordinal);
        Assert.Contains("if (exec) { v2 =", source, StringComparison.Ordinal);
    }

    [Fact]
    public void ACompareEndsTheRunBeforeIt()
    {
        // v_cmp_eq_u32 vcc_lo, v1, v2 inside the masked arm writes VCC, which every lane must agree on.
        var words = MaskedRun.Words.ToList();
        words.Insert(words.IndexOf(0x7E0A0287u), 0x7D840501);
        var source = Gen5MslTranslatorTestSupport.CompileOrThrow(MaskedRun with { Words = [.. words] }, new Gen5MslCodegenOptions()).Source;
        var compare = source.IndexOf("vcc =", StringComparison.Ordinal);
        Assert.True(compare > 0, source);
        var openRun = source.LastIndexOf("if (exec)\n", compare, StringComparison.Ordinal);
        var closeRun = openRun < 0 ? -1 : source.IndexOf('}', openRun);
        Assert.True(openRun < 0 || closeRun < compare, "the compare landed inside a run:\n" + source);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void MaskedOffLanesKeepTheirValues(bool runs)
    {
        if (SkipWithoutMetalDevice("the EXEC run execution test")) return;

        var shader = Compile(new Gen5MslCodegenOptions() with { ExecRuns = runs });
        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var compileError),
            $"{compileError}\n{shader.Source}");
        Assert.True(
            MetalNative.TryExecuteWithArgumentBuffer(
                library,
                shader.EntryPoint,
                new byte[256],
                new byte[PushData.ByteSize],
                shader.ArgumentLayout!,
                32,
                out var result,
                out var runError),
            runError);
        for (var lane = 0u; lane < 32; lane++)
        {
            var v1 = BitConverter.ToUInt32(result, (int)lane * 8);
            var v2 = BitConverter.ToUInt32(result, (int)lane * 8 + 4);
            Assert.Equal(lane < 16 ? lane + 1 : 100u, v1);
            Assert.Equal(lane < 16 ? 3 * (lane + 1) : 200u, v2);
        }
    }

    private static Gen5MslShader Compile(Gen5MslCodegenOptions options) =>
        Gen5MslTranslatorTestSupport.CompileOrThrow(MaskedRun, options);

    private bool SkipWithoutMetalDevice(string activity)
    {
        if (MetalNative.IsAvailable) return false;
        Assert.False(
            OperatingSystem.IsMacOS() && Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") == "1",
            $"The required gate cannot run {activity} without a Metal device.");
        output.WriteLine($"No Metal device on this host; {activity} skipped.");
        return true;
    }
}

internal static class Gen5MslTranslatorTestSupport
{
    public static Gen5MslShader CompileOrThrow(Gen5ComputeFixture fixture, Gen5MslCodegenOptions options)
    {
        var request = Gen5ComputeFixtures.CreateComputeRequest(fixture);
        if (!Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error, options))
        {
            throw new InvalidOperationException($"[{fixture.Name}] MSL request emit failed: {error}");
        }

        return shader;
    }
}
