// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using Xunit;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

public sealed class MslHalfLocalBranchTests
{
    // An if over lanes below the threshold whose body only stores: each half of the wave64
    // skips it on its own lanes, without an exchange.
    private static uint[] StoreIfProgram(uint threshold) =>
    [
        0xD7650000, 0x000100C1,   // v_mbcnt_lo_u32_b32 v0, -1, 0
        0xD7660000, 0x000200C1,   // v_mbcnt_hi_u32_b32 v0, -1, v0
        0x34080082,               // v_lshlrev_b32 v4, 2, v0
        0x7D880080 | threshold,   // v_cmp_gt_u32 vcc, threshold, v0
        0xBE82246A,               // s_and_saveexec_b64 s[2:3], vcc
        0xBF880003,               // s_cbranch_execz join
        0x4A0A0085,               // v_add_nc_u32 v5, 5, v0
        0xE0701000, 0x80020504,   // buffer_store_dword v5, v4, s[8:11], 0 offen
        0xBEFE0402,               // join: s_mov_b64 exec, s[2:3]
        0xBF810000,               // s_endpgm
    ];

    // The same if, but its body leaves s5 behind for the code after it: the halves would
    // disagree on s5, so the branch keeps testing the whole wave.
    private static readonly uint[] LiveScalarProgram =
    [
        0xD7650000, 0x000100C1,   // v_mbcnt_lo_u32_b32 v0, -1, 0
        0xD7660000, 0x000200C1,   // v_mbcnt_hi_u32_b32 v0, -1, v0
        0x34080082,               // v_lshlrev_b32 v4, 2, v0
        0xBE850380,               // s_mov_b32 s5, 0
        0x7D880094,               // v_cmp_gt_u32 vcc, 20, v0
        0xBE82246A,               // s_and_saveexec_b64 s[2:3], vcc
        0xBF880001,               // s_cbranch_execz join
        0xBE850387,               // s_mov_b32 s5, 7
        0xBEFE0402,               // join: s_mov_b64 exec, s[2:3]
        0x7E0A0205,               // v_mov_b32 v5, s5
        0xE0701000, 0x80020504,   // buffer_store_dword v5, v4, s[8:11], 0 offen
        0xBF810000,               // s_endpgm
    ];

    // A body that reads another lane (v_readfirstlane) needs the whole wave.
    private static readonly uint[] CrossLaneProgram =
    [
        0xD7650000, 0x000100C1,   // v_mbcnt_lo_u32_b32 v0, -1, 0
        0xD7660000, 0x000200C1,   // v_mbcnt_hi_u32_b32 v0, -1, v0
        0x34080082,               // v_lshlrev_b32 v4, 2, v0
        0x7D880094,               // v_cmp_gt_u32 vcc, 20, v0
        0xBE82246A,               // s_and_saveexec_b64 s[2:3], vcc
        0xBF880003,               // s_cbranch_execz join
        0x7E0A0500,               // v_readfirstlane_b32 s5, v0
        0xE0701000, 0x80020004,   // buffer_store_dword v0, v4, s[8:11], 0 offen
        0xBEFE0402,               // join: s_mov_b64 exec, s[2:3]
        0xBF810000,               // s_endpgm
    ];

    private const string HalfLocalTest = "sharpemu_ballot(exec) == 0u";

    private static Gen5MslShader Compile(Gen5ComputeFixture fixture, Gen5MslCodegenOptions options)
    {
        var request = Gen5ComputeFixtures.CreateComputeRequest(fixture, 64, 64);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error, options), error);
        return shader;
    }

    private static int Count(string source, string text)
    {
        var count = 0;
        for (var at = source.IndexOf(text, StringComparison.Ordinal); at >= 0; at = source.IndexOf(text, at + 1, StringComparison.Ordinal))
        {
            count++;
        }

        return count;
    }

    [Theory]
    [InlineData(0u)]
    [InlineData(20u)]
    [InlineData(40u)]
    [InlineData(64u)]
    public void ALaneLocalIf_TestsEachHalfsLanes_AndStoresTheSame(uint threshold)
    {
        var fixture = new Gen5ComputeFixture($"half-branch-{threshold}", StoreIfProgram(threshold), 8, 64 * 4);
        var half = Compile(fixture, new Gen5MslCodegenOptions());
        var before = Compile(fixture, new Gen5MslCodegenOptions { HalfLocalExecBranches = false });
        var guarded = Compile(fixture, new Gen5MslCodegenOptions(ExecGuardElision: false) { HalfLocalExecBranches = false });

        Assert.Contains(HalfLocalTest, half.Source);
        Assert.DoesNotContain(HalfLocalTest, before.Source);
        Assert.True(Count(half.Source, "threadgroup_barrier") < Count(before.Source, "threadgroup_barrier"),
            $"{Count(half.Source, "threadgroup_barrier")} barriers with half-local branches, {Count(before.Source, "threadgroup_barrier")} before");

        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            return;
        }

        Assert.True(MetalNative.IsAvailable, "A Metal device is required.");
        var expected = Execute(fixture, guarded);
        Assert.Equal(expected, Execute(fixture, before));
        var actual = Execute(fixture, half);
        Assert.Equal(expected, actual);
        for (var lane = 0; lane < 64; lane++)
        {
            Assert.Equal(lane < threshold ? (uint)(lane + 5) : 0xCDCDCDCDu,
                BinaryPrimitives.ReadUInt32LittleEndian(actual.AsSpan(lane * 4, 4)));
        }
    }

    [Fact]
    public void ABodyThatLeavesALiveScalarBehind_KeepsTheWholeWaveTest()
    {
        var fixture = new Gen5ComputeFixture("half-branch-live-scalar", LiveScalarProgram, 8, 64 * 4);
        var shader = Compile(fixture, new Gen5MslCodegenOptions());
        Assert.DoesNotContain(HalfLocalTest, shader.Source);

        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            return;
        }

        Assert.True(MetalNative.IsAvailable, "A Metal device is required.");
        var actual = Execute(fixture, shader);
        for (var lane = 0; lane < 64; lane++)
        {
            Assert.Equal(7u, BinaryPrimitives.ReadUInt32LittleEndian(actual.AsSpan(lane * 4, 4)));
        }
    }

    [Fact]
    public void ABodyThatReadsAnotherLane_KeepsTheWholeWaveTest()
    {
        var fixture = new Gen5ComputeFixture("half-branch-cross-lane", CrossLaneProgram, 8, 64 * 4);
        Assert.DoesNotContain(HalfLocalTest, Compile(fixture, new Gen5MslCodegenOptions()).Source);
    }

    private static byte[] Execute(Gen5ComputeFixture fixture, Gen5MslShader shader)
    {
        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var compileError), compileError);
        var bytes = Enumerable.Repeat((byte)0xCD, fixture.StoreBackingBytes).ToArray();
        Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, bytes,
            new byte[Resources.PushData.ByteSize], shader.ArgumentLayout!, 64,
            out var result, out var runError), runError);
        return result;
    }
}
