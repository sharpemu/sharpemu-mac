// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Xunit;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>The half-local branch proof must refuse a body whose scalar write is read after the
/// join. These shapes hide that read behind instructions whose destination is also a source the
/// liveness pass does not record.</summary>
public sealed class MslHalfLocalLivenessSoundnessTests
{
    private const string HalfLocalTest = "sharpemu_ballot(exec) == 0u";
    private const uint AddFiveToV5 = 0x4A0A0085;        // v_add_nc_u32 v5, 5, v0

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

    private static void AssertRefused(string name, uint[] program)
    {
        var fixture = new Gen5ComputeFixture($"half-branch-liveness-{name}", program, 8, 64 * 4);
        var request = Gen5ComputeFixtures.CreateComputeRequest(fixture, 64, 64);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error, new Gen5MslCodegenOptions()), error);
        Assert.DoesNotContain(HalfLocalTest, shader.Source);
    }

    private static readonly uint[] BodyWritesS5 = [0xBE850387, AddFiveToV5];   // s_mov_b32 s5, 7

    // The body writes s5; after the join s_addk_i32 s5, 1 reads it (read-modify-write), then v5 = s5.
    [Fact]
    public void SAddk_AfterTheJoin_ReadsTheBodysWrite() =>
        AssertRefused("saddk", IfProgram([], BodyWritesS5, [0xB7850001, 0x7E0A0205]));

    // s_mulk_i32 s5, 3 likewise.
    [Fact]
    public void SMulk_AfterTheJoin_ReadsTheBodysWrite() =>
        AssertRefused("smulk", IfProgram([], BodyWritesS5, [0xB8050003, 0x7E0A0205]));

    // s_cmovk_i32 s5, 9 writes s5 only when SCC is set, otherwise keeps the body's value.
    // The body writes s5 with a scalar load, so it writes no SCC and SCC being live (the cmovk reads it)
    // cannot refuse it by itself.
    [Fact]
    public void SCmovk_AfterTheJoin_KeepsTheBodysWriteWhenNotTaken() =>
        AssertRefused("scmovk", IfProgram([], [0xF4200144, 0xFA000000, 0xBF8CC07F, AddFiveToV5], [0xB1050009, 0x7E0A0205]));

    // s_bitset0_b32 s5, 1 clears one bit of s5 and keeps the rest (read-modify-write).
    [Fact]
    public void SBitset0_AfterTheJoin_ReadsTheBodysWrite() =>
        AssertRefused("sbitset0", IfProgram([], BodyWritesS5, [0xBE851B81, 0x7E0A0205]));
}
