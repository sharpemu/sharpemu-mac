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

    // Lanes 0..19 take an if whose body is `body` (the high half has none of them); `after` runs
    // at the join, then every lane stores v5. `pre` runs before the if.
    private static uint[] IfProgram(uint[] pre, uint[] body, uint[] after) =>
    [
        0xD7650000, 0x000100C1,   // v_mbcnt_lo_u32_b32 v0, -1, 0
        0xD7660000, 0x000200C1,   // v_mbcnt_hi_u32_b32 v0, -1, v0
        0x34080082,               // v_lshlrev_b32 v4, 2, v0
        0x7E0A0280,               // v_mov_b32 v5, 0
        0xBE840383,               // s_mov_b32 s4, 3
        0xBE8E0383,               // s_mov_b32 s14, 3
        .. pre,
        0x7D880094,               // v_cmp_gt_u32 vcc, 20, v0
        0xBE82246A,               // s_and_saveexec_b64 s[2:3], vcc
        0xBF880000 | (uint)body.Length, // s_cbranch_execz join
        .. body,
        0xBEFE0402,               // join: s_mov_b64 exec, s[2:3]
        .. after,
        0xE0701000, 0x80020504,   // buffer_store_dword v5, v4, s[8:11], 0 offen
        0xBF810000,               // s_endpgm
    ];

    private const uint AddFiveToV5 = 0x4A0A0085;        // v_add_nc_u32 v5, 5, v0

    public static TheoryData<string, uint[], bool> ProofShapes => new()
    {
        // The body writes s5 and the join reads only s4, as a 32-bit operand.
        { "s4-read-alone", IfProgram([], [0xBE850387, AddFiveToV5], [0x4A0A0A04]), true },
        // The join reads s[4:5] as a pair (s_mov_b64 s[6:7], s[4:5]), so s5 is live.
        { "s4-read-as-pair", IfProgram([], [0xBE850387, AddFiveToV5], [0xBE860404, 0x4A0A0A07]), false },
        // The body writes s12; the store after the join reads only its buffer resource s[8:11].
        { "beside-the-resource", IfProgram([], [0xBE8C0387, AddFiveToV5], []), true },
        // The body writes s11 (with its own value), inside that resource.
        { "inside-the-resource", IfProgram([], [0xBE8B030B, AddFiveToV5], []), false },
        // s_buffer_load_dwordx2 s[12:13] writes two registers; the join reads s14.
        { "past-the-load", IfProgram([], [0xF4240304, 0xFA000000, 0xBF8CC07F, AddFiveToV5], [0x4A0A0A0E]), true },
        // The join reads s13, which the load wrote.
        { "inside-the-load", IfProgram([], [0xF4240304, 0xFA000000, 0xBF8CC07F, AddFiveToV5], [0x4A0A0A0D]), false },
        // A nested if saves EXEC in s[4:5] and restores it before anything else writes s4; s4 is
        // written afterwards, which no longer hides the restore's subset.
        { "restore-then-overwrite", IfProgram([], [0x7D88008A, 0xBE84246A, AddFiveToV5, 0xBEFE0404, 0xBE840380], []), true },
        // s4 is overwritten between the save and the restore: the restore could turn lanes on.
        { "overwrite-then-restore", IfProgram([], [0x7D88008A, 0xBE84246A, AddFiveToV5, 0xBE840380, 0xBEFE0404], []), false },
        // v_mov_b32_dpp v5, v0 quad_perm:[1,0,3,2] reads a lane of its own quad.
        { "dpp-quad-perm", IfProgram([], [0x7E0A02FA, 0xFF00B100], []), true },
        // The same DPP, then v_readlane_b32 s5, v5, 40: a lane of the other half.
        { "dpp-then-readlane", IfProgram([], [0x7E0A02FA, 0xFF00B100, 0xD7600005, 0x00015105], []), false },
    };

    [Theory]
    [MemberData(nameof(ProofShapes))]
    public void TheBodyProof_TakesExactlyTheHalfLocalShapes(string name, uint[] program, bool halfLocal)
    {
        var fixture = new Gen5ComputeFixture($"half-branch-proof-{name}", program, 8, 64 * 4);
        var half = Compile(fixture, new Gen5MslCodegenOptions());
        if (halfLocal)
        {
            Assert.Contains(HalfLocalTest, half.Source);
        }
        else
        {
            Assert.DoesNotContain(HalfLocalTest, half.Source);
        }

        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            return;
        }

        Assert.True(MetalNative.IsAvailable, "A Metal device is required.");
        var before = Compile(fixture, new Gen5MslCodegenOptions { HalfLocalExecBranches = false });
        var guarded = Compile(fixture, new Gen5MslCodegenOptions(ExecGuardElision: false) { HalfLocalExecBranches = false });
        var expected = Execute(fixture, guarded);
        Assert.Equal(expected, Execute(fixture, before));
        Assert.Equal(expected, Execute(fixture, half));
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
