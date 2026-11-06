// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Text.RegularExpressions;
using Xunit;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

public sealed class MslExecLaneSpillTests
{
    // An if whose saved EXEC is parked in two lanes of v27 while s[0:1] are reused, then read
    // back and restored: the add after the restore runs with every lane set.
    private static readonly uint[] SpilledSaveexecProgram =
    [
        0xD7650000, 0x000100C1, // v_mbcnt_lo_u32_b32 v0, -1, 0
        0xD7660000, 0x000200C1, // v_mbcnt_hi_u32_b32 v0, -1, v0
        0x34080082,             // v_lshlrev_b32 v4, 2, v0
        0x7E040287,             // v_mov_b32 v2, 7
        0x7D880094,             // v_cmp_gt_u32 vcc, 20, v0
        0xBE80246A,             // s_and_saveexec_b64 s[0:1], vcc
        0xD761001B, 0x00014400, // v_writelane_b32 v27, s0, 34
        0xD761001B, 0x00014601, // v_writelane_b32 v27, s1, 35
        0xBE810380,             // s_mov_b32 s1, 0
        0xBE800380,             // s_mov_b32 s0, 0
        0x7E04028D,             // v_mov_b32 v2, 13 (lanes 0..19)
        0xD7600000, 0x0001451B, // v_readlane_b32 s0, v27, 34
        0xD7600001, 0x0001471B, // v_readlane_b32 s1, v27, 35
        0xBEFE0400,             // s_mov_b64 exec, s[0:1]
        0x4A040481,             // v_add_nc_u32 v2, 1, v2 (every lane)
        0xE0701000, 0x80020204, // buffer_store_dword v2, v4, s[8:11], 0 offen
        0xBF810000,             // s_endpgm
    ];

    [Fact]
    public void ReloadedSavedExec_RestoresUnguardedWrites_AndTheSwitchKeepsTheGuards()
    {
        var fixture = new Gen5ComputeFixture("exec-lane-spill", SpilledSaveexecProgram, 8, 64 * 4);
        var request = Gen5ComputeFixtures.CreateComputeRequest(fixture, 64, 64);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var tracked, out var error,
            new Gen5MslCodegenOptions()), error);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var untracked, out error,
            new Gen5MslCodegenOptions { ExecLaneSpills = false }), error);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var guarded, out error,
            new Gen5MslCodegenOptions(ExecGuardElision: false)), error);

        // The add and the store after the restore lose their guards; the masked move keeps its.
        Assert.True(GuardCount(tracked.Source) < GuardCount(untracked.Source),
            $"tracked {GuardCount(tracked.Source)} guards, untracked {GuardCount(untracked.Source)}");
        Assert.True(GuardCount(tracked.Source) >= 1);

        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            return;
        }

        Assert.True(MetalNative.IsAvailable, "A Metal device is required.");
        var expected = Execute(guarded);
        Assert.Equal(expected, Execute(untracked));
        var actual = Execute(tracked);
        Assert.Equal(expected, actual);
        for (var lane = 0; lane < 64; lane++)
        {
            Assert.Equal(lane < 20 ? 14u : 8u, BinaryPrimitives.ReadUInt32LittleEndian(actual.AsSpan(lane * 4, 4)));
        }

        byte[] Execute(Gen5MslShader shader)
        {
            Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var compileError), compileError);
            var bytes = Enumerable.Repeat((byte)0xCD, fixture.StoreBackingBytes).ToArray();
            Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, bytes,
                new byte[Resources.PushData.ByteSize], shader.ArgumentLayout!, 64,
                out var result, out var runError), runError);
            return result;
        }
    }

    private static int GuardCount(string source) => Regex.Matches(source, @"if \(exec\)").Count;
}
