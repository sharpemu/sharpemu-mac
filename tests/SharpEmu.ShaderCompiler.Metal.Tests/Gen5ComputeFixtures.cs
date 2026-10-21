// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

internal sealed record Gen5ComputeFixture(
    string Name,
    uint[] Words,
    uint StoreScalarResourceBase,
    int StoreBackingBytes);

// Synthetic instruction programs for decoder and compile-request tests.
internal static class Gen5ComputeFixtures
{
    public const ulong ProgramAddress = 0x100000;

    // Straight-line ALU: VOP2 fmac + fmamk/fmaak literals + the VOP3 fmac form.
    public static readonly Gen5ComputeFixture Fmac = new(
        "fmac",
        [
            0x560A0501,             // v_fmac_f32 v5, v1, v2
            0x580A0501, 0x42280000, // v_fmamk_f32 v5, v1, 42.0, v2
            0x5A0A0501, 0x42280000, // v_fmaak_f32 v5, v1, v2, 42.0
            0xD52B0005, 0x00020501, // v_fmac_f32_e64 v5, v1, v2
            0xBF810000,             // s_endpgm
        ],
        StoreScalarResourceBase: 0,
        StoreBackingBytes: 0);

    // VOP3 integer multiplies, low and high halves, signed and unsigned.
    public static readonly Gen5ComputeFixture Muls = new(
        "muls",
        [
            0xD5690005, 0x00020501, // v_mul_lo_u32 v5, v1, v2
            0xD56A0005, 0x00020501, // v_mul_hi_u32 v5, v1, v2
            0xD56B0005, 0x00020501, // v_mul_lo_i32 v5, v1, v2
            0xD56C0005, 0x00020501, // v_mul_hi_i32 v5, v1, v2
            0xBF810000,             // s_endpgm
        ],
        StoreScalarResourceBase: 0,
        StoreBackingBytes: 0);

    // End-to-end executable program: real ALU results stored to buffer 0 at dword
    // offsets 0/4/8, then proof that a store with EXEC=0 does not land (offset 12
    // keeps its sentinel) and that stores work again once EXEC is restored (16).
    public static readonly Gen5ComputeFixture ExecStore = new(
        "exec-store",
        [
            0xBFA10001,             // s_clause 0x1 (scheduling hint)
            0x7E0002FF, 0x3FC00000, // v_mov_b32 v0, 1.5f
            0x7E0202FF, 0x40100000, // v_mov_b32 v1, 2.25f
            0x7E0402FF, 0x41200000, // v_mov_b32 v2, 10.0f
            0x56040300,             // v_fmac_f32 v2, v0, v1      -> v2 = fma(1.5, 2.25, 10.0)
            0x7E0602FF, 0x7FFFFFFF, // v_mov_b32 v3, 0x7FFFFFFF
            0x7E0802FF, 0x00010003, // v_mov_b32 v4, 0x00010003
            0xD56C0005, 0x00020903, // v_mul_hi_i32 v5, v3, v4
            0xD56B0006, 0x00020903, // v_mul_lo_i32 v6, v3, v4
            0xE0700000, 0x80020200, // buffer_store_dword v2, off, s[8:11], 0
            0xE0700004, 0x80020500, // buffer_store_dword v5, off, s[8:11], 0 offset:4
            0xE0700008, 0x80020600, // buffer_store_dword v6, off, s[8:11], 0 offset:8
            0xBEFE0380,             // s_mov_b32 exec_lo, 0       -> lane inactive
            0xE070000C, 0x80020200, // buffer_store_dword v2 offset:12 (masked, must not land)
            0xBEFE03C1,             // s_mov_b32 exec_lo, -1      -> lane active again
            0xE0700010, 0x80020000, // buffer_store_dword v0 offset:16
            0xBF810000,             // s_endpgm
        ],
        StoreScalarResourceBase: 8,
        StoreBackingBytes: 64);

    // Control flow through the PC dispatcher: a scalar loop counting down from 5,
    // accumulating 5+4+3+2+1 = 15 into v1, with the result stored to buffer 0.
    // (s_sub_i32's SCC is signed overflow, so the loop condition needs an
    // explicit s_cmp_lg_u32 — which also exercises the scalar-compare path.)
    public static readonly Gen5ComputeFixture Loop = new(
        "loop",
        [
            0xBE800385,             // s_mov_b32 s0, 5
            0x7E020280,             // v_mov_b32 v1, 0
            // loop: (pc=0x8)
            0x4A020200,             // v_add_nc_u32 v1, s0, v1
            0x81808100,             // s_sub_i32 s0, s0, 1 (inline +1 = 0x81)
            0xBF078000,             // s_cmp_lg_u32 s0, 0
            0xBF85FFFC,             // s_cbranch_scc1 loop
            0xE0700000, 0x80020100, // buffer_store_dword v1, off, s[8:11], 0
            0xBF810000,             // s_endpgm
        ],
        StoreScalarResourceBase: 8,
        StoreBackingBytes: 16);

    // A loop that continues from three places through one short latch, the shape of
    // UE5's persistent culling loop. Odd i runs X and X2, i == 4 continues straight to
    // the latch, i == 6 runs Y, and i == 9 leaves from inside X. s1 counts X (+1) and
    // X2 (+16), s2 counts Y and s3 the latch; the store packs them as
    // s1 | s2 << 8 | s3 << 16, which is 103 | 1 << 8 | 8 << 16 = 0x00080167.
    public static readonly Gen5ComputeFixture SharedLatchLoop = new(
        "shared-latch-loop",
        [
            0xBE800380,             // s_mov_b32 s0, 0
            0xBE810380,             // s_mov_b32 s1, 0
            0xBE820380,             // s_mov_b32 s2, 0
            0xBE830380,             // s_mov_b32 s3, 0
            0x80008100,             // H (0x10): s_add_u32 s0, s0, 1
            0xBF0D8000,             // s_bitcmp1_b32 s0, 0
            0xBF850004,             // s_cbranch_scc1 X
            0xBF068400,             // s_cmp_eq_u32 s0, 4
            0xBF850008,             // s_cbranch_scc1 L
            0xBF068600,             // s_cmp_eq_u32 s0, 6
            0xBF850005,             // s_cbranch_scc1 Y
            0x80018101,             // X (0x2C): s_add_u32 s1, s1, 1
            0xBF098900,             // s_cmp_ge_u32 s0, 9
            0xBF850005,             // s_cbranch_scc1 DONE
            0x80019001,             // X2: s_add_u32 s1, s1, 16
            0xBF820001,             // s_branch L
            0x80028102,             // Y (0x40): s_add_u32 s2, s2, 1
            0x80038103,             // L (0x44): s_add_u32 s3, s3, 1
            0xBF82FFF1,             // s_branch H
            0x8F028802,             // DONE (0x4C): s_lshl_b32 s2, s2, 8
            0x8F039003,             // s_lshl_b32 s3, s3, 16
            0x88010201,             // s_or_b32 s1, s1, s2
            0x88010301,             // s_or_b32 s1, s1, s3
            0x7E020201,             // v_mov_b32 v1, s1
            0xE0700000, 0x80020100, // buffer_store_dword v1, off, s[8:11], 0
            0xBF810000,             // s_endpgm
        ],
        StoreScalarResourceBase: 8,
        StoreBackingBytes: 16);

    // A branch inside a loop whose two arms each either run a shared tail T or skip
    // straight to the join J, the shape EXEC-skip branches leave in UE5's cloud
    // kernels. For i = 1..8, odd i counts s1 and runs T unless i == 5, even i runs T
    // unless i == 4, and J counts every trip in s2. T counts in s3, so the store packs
    // s1 | s2 << 8 | s3 << 16 = 4 | 8 << 8 | 6 << 16 = 0x00060804.
    public static readonly Gen5ComputeFixture SharedTailLoop = new(
        "shared-tail-loop",
        [
            0xBE800380,             // s_mov_b32 s0, 0
            0xBE810380,             // s_mov_b32 s1, 0
            0xBE820380,             // s_mov_b32 s2, 0
            0xBE830380,             // s_mov_b32 s3, 0
            0x80008100,             // H (0x10): s_add_u32 s0, s0, 1
            0xBF0D8000,             // s_bitcmp1_b32 s0, 0
            0xBF850003,             // s_cbranch_scc1 P
            0xBF068400,             // s_cmp_eq_u32 s0, 4
            0xBF850005,             // s_cbranch_scc1 J
            0xBF820003,             // s_branch T
            0x80018101,             // P (0x28): s_add_u32 s1, s1, 1
            0xBF068500,             // s_cmp_eq_u32 s0, 5
            0xBF850001,             // s_cbranch_scc1 J
            0x80038103,             // T (0x34): s_add_u32 s3, s3, 1
            0x80028102,             // J (0x38): s_add_u32 s2, s2, 1
            0xBF0A8800,             // s_cmp_lt_u32 s0, 8
            0xBF85FFF3,             // s_cbranch_scc1 H
            0x8F028802,             // s_lshl_b32 s2, s2, 8
            0x8F039003,             // s_lshl_b32 s3, s3, 16
            0x88010201,             // s_or_b32 s1, s1, s2
            0x88010301,             // s_or_b32 s1, s1, s3
            0x7E020201,             // v_mov_b32 v1, s1
            0xE0700000, 0x80020100, // buffer_store_dword v1, off, s[8:11], 0
            0xBF810000,             // s_endpgm
        ],
        StoreScalarResourceBase: 8,
        StoreBackingBytes: 16);

    // A loop with two ways out, A and B, that rejoin at J: for i = 1.. it leaves through A when
    // i == 3 (s2 = 10) or through B when i == 5 (s2 = 11), counting the full trips in s1, and
    // stores s1 | s2 << 8. This one leaves through A: 2 | 10 << 8 = 0x00000A02.
    public static readonly Gen5ComputeFixture MultiExitLoopFirst = new(
        "multi-exit-loop-first",
        [
            0xBE800380,             // s_mov_b32 s0, 0
            0xBE810380,             // s_mov_b32 s1, 0
            0xBE820380,             // s_mov_b32 s2, 0
            0x80008100,             // H (0x0C): s_add_u32 s0, s0, 1
            0xBF068300,             // s_cmp_eq_u32 s0, 3
            0xBF850004,             // s_cbranch_scc1 A
            0xBF068500,             // s_cmp_eq_u32 s0, 5
            0xBF850004,             // s_cbranch_scc1 B
            0x80018101,             // s_add_u32 s1, s1, 1
            0xBF82FFF9,             // s_branch H
            0xBE82038A,             // A (0x28): s_mov_b32 s2, 10
            0xBF820001,             // s_branch J
            0xBE82038B,             // B (0x30): s_mov_b32 s2, 11
            0x8F028802,             // J (0x34): s_lshl_b32 s2, s2, 8
            0x88010201,             // s_or_b32 s1, s1, s2
            0x7E020201,             // v_mov_b32 v1, s1
            0xE0700000, 0x80020100, // buffer_store_dword v1, off, s[8:11], 0
            0xBF810000,             // s_endpgm
        ],
        StoreScalarResourceBase: 8,
        StoreBackingBytes: 16);

    // The same loop with A at i == 7, so it leaves through B: 4 | 11 << 8 = 0x00000B04.
    public static readonly Gen5ComputeFixture MultiExitLoopSecond = new(
        "multi-exit-loop-second",
        [
            0xBE800380,             // s_mov_b32 s0, 0
            0xBE810380,             // s_mov_b32 s1, 0
            0xBE820380,             // s_mov_b32 s2, 0
            0x80008100,             // H (0x0C): s_add_u32 s0, s0, 1
            0xBF068700,             // s_cmp_eq_u32 s0, 7
            0xBF850004,             // s_cbranch_scc1 A
            0xBF068500,             // s_cmp_eq_u32 s0, 5
            0xBF850004,             // s_cbranch_scc1 B
            0x80018101,             // s_add_u32 s1, s1, 1
            0xBF82FFF9,             // s_branch H
            0xBE82038A,             // A (0x28): s_mov_b32 s2, 10
            0xBF820001,             // s_branch J
            0xBE82038B,             // B (0x30): s_mov_b32 s2, 11
            0x8F028802,             // J (0x34): s_lshl_b32 s2, s2, 8
            0x88010201,             // s_or_b32 s1, s1, s2
            0x7E020201,             // v_mov_b32 v1, s1
            0xE0700000, 0x80020100, // buffer_store_dword v1, off, s[8:11], 0
            0xBF810000,             // s_endpgm
        ],
        StoreScalarResourceBase: 8,
        StoreBackingBytes: 16);

    // A branch with an arm that can end the program: B0 goes to J or B1, B1 to an early
    // S_ENDPGM at X or on to B2, and B2 falls into J. Nothing post-dominates B0, so its
    // arms used to be emitted apart and met J twice. s0 = 3 runs J alone (1), 5 leaves
    // through X (9) and 7 runs B2 then J (3).
    public static Gen5ComputeFixture ExitArmJoin(uint selector) => new(
        $"exit-arm-join-{selector}",
        [
            0xBE800380 | selector,  // s_mov_b32 s0, selector
            0xBE810380,             // s_mov_b32 s1, 0
            0xBF068300,             // B0: s_cmp_eq_u32 s0, 3
            0xBF850003,             // s_cbranch_scc1 J
            0xBF068500,             // B1 (0x10): s_cmp_eq_u32 s0, 5
            0xBF850006,             // s_cbranch_scc1 X
            0xBE810382,             // B2 (0x18): s_mov_b32 s1, 2
            0x80018101,             // J (0x1C): s_add_u32 s1, s1, 1
            0x7E020201,             // v_mov_b32 v1, s1
            0xE0700000, 0x80020100, // buffer_store_dword v1, off, s[8:11], 0
            0xBF810000,             // s_endpgm
            0xBE810389,             // X (0x30): s_mov_b32 s1, 9
            0x7E020201,             // v_mov_b32 v1, s1
            0xE0700000, 0x80020100, // buffer_store_dword v1, off, s[8:11], 0
            0xBF810000,             // s_endpgm
        ],
        StoreScalarResourceBase: 8,
        StoreBackingBytes: 16);

    // LDS round trip: write a literal through workgroup-shared memory, barrier,
    // read it back, and store it to buffer 0.
    public static readonly Gen5ComputeFixture Lds = new(
        "lds",
        [
            0x7E000280,             // v_mov_b32 v0, 0            (LDS byte address)
            0x7E0402FF, 0x00001234, // v_mov_b32 v2, 0x1234
            0xD8340000, 0x00000200, // ds_write_b32 v0, v2
            0xBF8A0000,             // s_barrier
            0xD8D80000, 0x03000000, // ds_read_b32 v3, v0
            0xE0700000, 0x80020300, // buffer_store_dword v3, off, s[8:11], 0
            0xBF810000,             // s_endpgm
        ],
        StoreScalarResourceBase: 8,
        StoreBackingBytes: 16);

    // Wave64 cross-lane: v_readfirstlane routes through the translator's
    // two-half threadgroup-scratch broadcast bridge. All lanes hold the same
    // value (42), so the broadcast result is 42 on every lane regardless of
    // which lane is "first" — the fixture's job is to prove the emitted wave64
    // MSL compiles and that a full 64-lane dispatch runs through the bridge
    // barriers without deadlocking, returning the broadcast value.
    public static readonly Gen5ComputeFixture Wave64Broadcast = new(
        "wave64-broadcast",
        [
            0x7E0002FF, 0x0000002A, // v_mov_b32 v0, 42
            0x7E000500,             // v_readfirstlane_b32 s0, v0
            0x7E020200,             // v_mov_b32 v1, s0
            0xE0700000, 0x80020100, // buffer_store_dword v1, off, s[8:11], 0
            0xBF810000,             // s_endpgm
        ],
        StoreScalarResourceBase: 8,
        StoreBackingBytes: 16);

    // Typed load: the instruction's two 32-bit floats win over the descriptor's four
    // unorm bytes in s11, so the two raw dwords at offset 0 land at offsets 16 and 20.
    public static readonly Gen5ComputeFixture TypedLoad = new(
        "typed-load",
        [
            0xBE8B03FF, 0x00038FAC, // s_mov_b32 s11, 0x38FAC (descriptor format 56, swizzle xyzw)
            0xEA010000, 0x80020200, // tbuffer_load_format_xy v[2:3], off, s[8:11], 0 format:64
            0xE0700010, 0x80020200, // buffer_store_dword v2, off, s[8:11], 0 offset:16
            0xE0700014, 0x80020300, // buffer_store_dword v3, off, s[8:11], 0 offset:20
            0xBF810000,             // s_endpgm
        ],
        StoreScalarResourceBase: 8,
        StoreBackingBytes: 32);

    // Typed store with format 8_8 unorm: 1.0 in v2 stores 0xFF at offset 16 and 0.5 in v3
    // stores 0x80 at offset 17; the other two bytes of that dword keep their value.
    public static readonly Gen5ComputeFixture TypedStore = new(
        "typed-store",
        [
            0xBE8B03FF, 0x00038FAC, // s_mov_b32 s11, 0x38FAC (descriptor format 56, swizzle xyzw)
            0x7E0402FF, 0x3F800000, // v_mov_b32 v2, 1.0
            0x7E0602FF, 0x3F000000, // v_mov_b32 v3, 0.5
            0xE8750010, 0x80020200, // tbuffer_store_format_xy v[2:3], off, s[8:11], 0 format:14 offset:16
            0xBF810000,             // s_endpgm
        ],
        StoreScalarResourceBase: 8,
        StoreBackingBytes: 32);

    public static readonly Gen5ComputeFixture[] All = [Fmac, Muls, ExecStore, Loop, Lds, TypedLoad, TypedStore];

    // Minimal pixel program: two interpolated attribute channels plus two
    // inline constants exported to MRT0 (done+vm).
    public static readonly uint[] PixelWords =
    [
        0xC8020002,             // v_interp_mov_f32 v0, p0, attr0.x
        0xC8060102,             // v_interp_mov_f32 v1, p0, attr0.y
        0x7E0402F2,             // v_mov_b32 v2, 1.0
        0x7E0602F0,             // v_mov_b32 v3, 0.5
        0xF800180F, 0x03020100, // exp mrt0 v0, v1, v2, v3 done vm
        0xBF810000,             // s_endpgm
    ];

    // Minimal vertex program: constant position export plus one param output.
    public static readonly uint[] VertexWords =
    [
        0x7E000280,             // v_mov_b32 v0, 0
        0x7E0202F2,             // v_mov_b32 v1, 1.0
        0xF80008CF, 0x01000000, // exp pos0 v0, v0, v0, v1 done
        0xF800020F, 0x01000101, // exp param0 v1, v1, v0, v1
        0xBF810000,             // s_endpgm
    ];

    public static Gen5MslShader CompileVertexOrThrow(int requiredVertexOutputCount = 0) =>
        CompileStageRequestOrThrow(VertexWords, ShaderStage.Vertex, requiredVertexOutputCount);

    public static Gen5MslShader CompilePixelOrThrow(
        Gen5PixelOutputKind outputKind = Gen5PixelOutputKind.Float,
        Gen5ColorComponentMapping componentMapping = default)
    {
        var program = DecodeOrThrow(new Gen5ComputeFixture("pixel", PixelWords, 0, 0));
        var request = RequestOrThrow(program, ShaderStage.Pixel,
            pixelOutputs: [new Gen5PixelOutputBinding(0, 0, outputKind, componentMapping)]);
        if (!Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error))
        {
            throw new InvalidOperationException($"[pixel] MSL request emit failed: {error}");
        }
        return shader;
    }

    // Per-lane wave64 programs. Every lane stores its results at lane * 8 (v4), so a
    // test can read each of the 64 lanes back. v0 = the full 0..63 lane index.
    private static readonly uint[] Wave64LaneIndex =
    [
        0xD7650000, 0x000100C1, // v_mbcnt_lo_u32_b32 v0, -1, 0
        0xD7660000, 0x000200C1, // v_mbcnt_hi_u32_b32 v0, -1, v0        lane 0..63
    ];

    // v1 = 3 * lane + 7; every lane reads lane 63 and lane 5 of the whole wave.
    public static readonly Gen5ComputeFixture Wave64ReadLane = new(
        "wave64-readlane",
        [
            .. Wave64LaneIndex,
            0xD5430001, 0x021D0700, // v_mad_u32_u24 v1, v0, 3, 7
            0xD7600000, 0x00017F01, // v_readlane_b32 s0, v1, 63
            0xD7600001, 0x00010B01, // v_readlane_b32 s1, v1, 5
            0x7E040200,             // v_mov_b32 v2, s0
            0x7E060201,             // v_mov_b32 v3, s1
            0x34080083,             // v_lshlrev_b32 v4, 3, v0
            0xE0701000, 0x80020204, // buffer_store_dword v2, v4, s[8:11], 0 offen
            0xE0701004, 0x80020304, // buffer_store_dword v3, v4, s[8:11], 0 offen offset:4
            0xBF810000,             // s_endpgm
        ],
        StoreScalarResourceBase: 8,
        StoreBackingBytes: 512);

    private static readonly uint[] Wave64LaneSpillPark =
    [
        0xBE8203FF, 0x00001234, // s_mov_b32 s2, 0x1234
        0xBE8303FF, 0x0000004D, // s_mov_b32 s3, 77
        0x7E020300,             // v_mov_b32 v1, v0
        0xD7610001, 0x00014A02, // v_writelane_b32 v1, s2, 37
        0xD7610001, 0x00010603, // v_writelane_b32 v1, s3, 3
        0xBE820380,             // s_mov_b32 s2, 0
        0xBE830380,             // s_mov_b32 s3, 0
    ];

    private static readonly uint[] Wave64LaneSpillStore =
    [
        0x7E040200,             // v_mov_b32 v2, s0
        0x7E060201,             // v_mov_b32 v3, s1
        0x34080083,             // v_lshlrev_b32 v4, 3, v0
        0xE0701000, 0x80020204, // buffer_store_dword v2, v4, s[8:11], 0 offen
        0xE0701004, 0x80020304, // buffer_store_dword v3, v4, s[8:11], 0 offen offset:4
        0xBF810000,             // s_endpgm
    ];

    // An SGPR spill and reload through VGPR lanes, the guest compiler's pattern: s2 and s3
    // are parked in lanes 37 and 3 of v1 (which otherwise holds the lane index), clobbered,
    // and read back. Every lane stores 0x1234 and 77.
    public static readonly Gen5ComputeFixture Wave64LaneSpill = new(
        "wave64-lane-spill",
        [
            .. Wave64LaneIndex,
            .. Wave64LaneSpillPark,
            0xD7600000, 0x00014B01, // v_readlane_b32 s0, v1, 37
            0xD7600001, 0x00010701, // v_readlane_b32 s1, v1, 3
            .. Wave64LaneSpillStore,
        ],
        StoreScalarResourceBase: 8,
        StoreBackingBytes: 512);

    // The same, but v1 is incremented between the spill and the reload, so the lanes no
    // longer hold what was parked and the reload has to read them: 0x1235 and 78.
    public static readonly Gen5ComputeFixture Wave64LaneSpillOverwritten = new(
        "wave64-lane-spill-overwritten",
        [
            .. Wave64LaneIndex,
            .. Wave64LaneSpillPark,
            0x4A020281,             // v_add_nc_u32 v1, 1, v1
            0xD7600000, 0x00014B01, // v_readlane_b32 s0, v1, 37
            0xD7600001, 0x00010701, // v_readlane_b32 s1, v1, 3
            .. Wave64LaneSpillStore,
        ],
        StoreScalarResourceBase: 8,
        StoreBackingBytes: 512);

    // A spill parked on one arm of a branch only: the path that skips the writelane would
    // read a lane nothing defined, which a correct program never relies on, so the reload
    // still reads the copy. At run time the branch falls through and every lane stores
    // 0x1234 twice.
    public static readonly Gen5ComputeFixture Wave64LaneSpillOneArm = new(
        "wave64-lane-spill-one-arm",
        [
            .. Wave64LaneIndex,
            0xBE8203FF, 0x00001234, // s_mov_b32 s2, 0x1234
            0xBF068002,             // s_cmp_eq_u32 s2, 0
            0xBF850002,             // s_cbranch_scc1 skip
            0xD7610005, 0x00014A02, // v_writelane_b32 v5, s2, 37
            0xBE820380,             // skip: s_mov_b32 s2, 0
            0xD7600000, 0x00014B05, // v_readlane_b32 s0, v5, 37
            0xBE810300,             // s_mov_b32 s1, s0
            .. Wave64LaneSpillStore,
        ],
        StoreScalarResourceBase: 8,
        StoreBackingBytes: 512);

    // v1 = 3 * lane + 7. Two V_READFIRSTLANEs back to back under narrowed EXECs: the first
    // active lane is 44 (upper half only), then 8 (lower half only). Every lane, with EXEC
    // restored, stores both: 139 and 31.
    public static readonly Gen5ComputeFixture Wave64ReadFirstLane = new(
        "wave64-readfirstlane",
        [
            .. Wave64LaneIndex,
            0xD5430001, 0x021D0700, // v_mad_u32_u24 v1, v0, 3, 7
            0xBEFE0380,             // s_mov_b32 exec_lo, 0
            0xBEFF03FF, 0xF0F0F000, // s_mov_b32 exec_hi, 0xf0f0f000     lanes 44.. of 32..63
            0x7E000501,             // v_readfirstlane_b32 s0, v1
            0xBEFE03FF, 0x00000100, // s_mov_b32 exec_lo, 0x100          lane 8
            0xBEFF0380,             // s_mov_b32 exec_hi, 0
            0x7E020501,             // v_readfirstlane_b32 s1, v1
            0xBEFE04C1,             // s_mov_b64 exec, -1
            0x7E040200,             // v_mov_b32 v2, s0
            0x7E060201,             // v_mov_b32 v3, s1
            0x34080083,             // v_lshlrev_b32 v4, 3, v0
            0xE0701000, 0x80020204, // buffer_store_dword v2, v4, s[8:11], 0 offen
            0xE0701004, 0x80020304, // buffer_store_dword v3, v4, s[8:11], 0 offen offset:4
            0xBF810000,             // s_endpgm
        ],
        StoreScalarResourceBase: 8,
        StoreBackingBytes: 512);

    // v1 = 3 * lane + 7. With EXEC empty in both halves the first lane is lane 0 (ISA
    // V_READFIRSTLANE_B32), so every lane stores 7.
    public static readonly Gen5ComputeFixture Wave64ReadFirstLaneEmptyExec = new(
        "wave64-readfirstlane-empty-exec",
        [
            .. Wave64LaneIndex,
            0xD5430001, 0x021D0700, // v_mad_u32_u24 v1, v0, 3, 7
            0xBEFE0480,             // s_mov_b64 exec, 0
            0x7E000501,             // v_readfirstlane_b32 s0, v1
            0xBEFE04C1,             // s_mov_b64 exec, -1
            0x7E040200,             // v_mov_b32 v2, s0
            0x34080083,             // v_lshlrev_b32 v4, 3, v0
            0xE0701000, 0x80020204, // buffer_store_dword v2, v4, s[8:11], 0 offen
            0xBF810000,             // s_endpgm
        ],
        StoreScalarResourceBase: 8,
        StoreBackingBytes: 512);

    // A loop of 64 trips, each with two exchanges: V_READLANE of lane i (3i + 7) and the
    // 64-bit VCC of lane > i (63 - i lanes, counted by S_BCNT1). The sum, 6496 + 2016 =
    // 8512, is only right if no trip reads a slot the next exchange has overwritten.
    public static readonly Gen5ComputeFixture Wave64ExchangeLoop = new(
        "wave64-exchange-loop",
        [
            .. Wave64LaneIndex,
            0xD5430001, 0x021D0700, // v_mad_u32_u24 v1, v0, 3, 7
            0xBE820380,             // s_mov_b32 s2, 0                   sum
            0xBE830380,             // s_mov_b32 s3, 0                   i
            0xD7600000, 0x00000701, // loop: v_readlane_b32 s0, v1, s3
            0x80020002,             // s_add_u32 s2, s2, s0
            0x7D820003,             // v_cmp_lt_u32 vcc, s3, v0
            0xBE84106A,             // s_bcnt1_i32_b64 s4, vcc
            0x80020402,             // s_add_u32 s2, s2, s4
            0x80038103,             // s_add_u32 s3, s3, 1
            0xBF0AC003,             // s_cmp_lt_u32 s3, 64
            0xBF85FFF7,             // s_cbranch_scc1 loop
            0x7E040202,             // v_mov_b32 v2, s2
            0x34080082,             // v_lshlrev_b32 v4, 2, v0
            0xE0701000, 0x80020204, // buffer_store_dword v2, v4, s[8:11], 0 offen
            0xBF810000,             // s_endpgm
        ],
        StoreScalarResourceBase: 8,
        StoreBackingBytes: 256);

    // v1 = lane, then v_writelane_b32 v1, 55, 40 must change only lane 40.
    public static readonly Gen5ComputeFixture Wave64WriteLane = new(
        "wave64-writelane",
        [
            .. Wave64LaneIndex,
            0x7E020300,             // v_mov_b32 v1, v0
            0xD7610001, 0x000150B7, // v_writelane_b32 v1, 55, 40
            0x34080082,             // v_lshlrev_b32 v4, 2, v0
            0xE0701000, 0x80020104, // buffer_store_dword v1, v4, s[8:11], 0 offen
            0xBF810000,             // s_endpgm
        ],
        StoreScalarResourceBase: 8,
        StoreBackingBytes: 256);

    // No s_barrier: lane l writes l + 1 to LDS slot l, then reads slot (l + 32) & 63, which the
    // other 32-wide half wrote. A guest wave's LDS ops are in order, so this must see l' + 1.
    public static readonly Gen5ComputeFixture Wave64LdsExchange = new(
        "wave64-lds-exchange",
        [
            .. Wave64LaneIndex,
            0xD5430001, 0x02050300, // v_mad_u32_u24 v1, v0, 1, 1           lane + 1
            0x34080082,             // v_lshlrev_b32 v4, 2, v0
            0xD8340000, 0x00000104, // ds_write_b32 v4, v1
            0xD5430005, 0x02810300, // v_mad_u32_u24 v5, v0, 1, 32          lane + 32
            0x360A0ABF,             // v_and_b32 v5, 63, v5
            0x340A0A82,             // v_lshlrev_b32 v5, 2, v5
            0xD8D80000, 0x03000005, // ds_read_b32 v3, v5
            0xE0701000, 0x80020304, // buffer_store_dword v3, v4, s[8:11], 0 offen
            0xBF810000,             // s_endpgm
        ],
        StoreScalarResourceBase: 8,
        StoreBackingBytes: 256);

    // s[4:5] = 0xF0F0F0F0_0000FFFF. v2 = the lane's bit of s[4:5] through a VOP3
    // v_cndmask_b32; v3 = 1 only on lanes that survive s_and_saveexec_b64 s[6:7], s[4:5].
    public static readonly Gen5ComputeFixture Wave64MaskBits = new(
        "wave64-mask-bits",
        [
            .. Wave64LaneIndex,
            0xBE8403FF, 0x0000FFFF, // s_mov_b32 s4, 0x0000ffff
            0xBE8503FF, 0xF0F0F0F0, // s_mov_b32 s5, 0xf0f0f0f0
            0xD5010002, 0x00110280, // v_cndmask_b32 v2, 0, 1, s[4:5]
            0x7E060280,             // v_mov_b32 v3, 0
            0xBE862404,             // s_and_saveexec_b64 s[6:7], s[4:5]
            0x7E060281,             // v_mov_b32 v3, 1
            0xBEFE0406,             // s_mov_b64 exec, s[6:7]
            0x34080083,             // v_lshlrev_b32 v4, 3, v0
            0xE0701000, 0x80020204, // buffer_store_dword v2, v4, s[8:11], 0 offen
            0xE0701004, 0x80020304, // buffer_store_dword v3, v4, s[8:11], 0 offen offset:4
            0xBF810000,             // s_endpgm
        ],
        StoreScalarResourceBase: 8,
        StoreBackingBytes: 512);

    // A VOP3-encoded V_CMP names its mask SGPR pair: s[4:5] = (lane & 3) == 0, with VCC
    // preset to 0x0f0f0f0f_0f0f0f0f and left alone. Per lane at lane * 32: the s[4:5]
    // bit (v_cndmask), the VCC bit, s4, s5, vcc_lo, and whether the lane survives
    // s_and_saveexec_b64 s[6:7], s[4:5].
    public static readonly Gen5ComputeFixture Wave64Vop3CompareSgpr = new(
        "wave64-vop3-compare-sgpr",
        [
            .. Wave64LaneIndex,
            0xBEEA03FF, 0x0F0F0F0F, // s_mov_b32 vcc_lo, 0x0f0f0f0f
            0xBEEB03FF, 0x0F0F0F0F, // s_mov_b32 vcc_hi, 0x0f0f0f0f
            0x36040083,             // v_and_b32 v2, 3, v0
            0xD4C20004, 0x00010102, // v_cmp_eq_u32_e64 s[4:5], v2, 0
            0xD5010001, 0x00110280, // v_cndmask_b32 v1, 0, 1, s[4:5]
            0xD5010002, 0x01A90280, // v_cndmask_b32 v2, 0, 1, vcc
            0x7E060204,             // v_mov_b32 v3, s4
            0x7E0A0205,             // v_mov_b32 v5, s5
            0x7E0C026A,             // v_mov_b32 v6, vcc_lo
            0x7E0E0280,             // v_mov_b32 v7, 0
            0xBE862404,             // s_and_saveexec_b64 s[6:7], s[4:5]
            0x7E0E0281,             // v_mov_b32 v7, 1
            0xBEFE0406,             // s_mov_b64 exec, s[6:7]
            0x34080085,             // v_lshlrev_b32 v4, 5, v0
            0xE0701000, 0x80020104, // buffer_store_dword v1, v4, s[8:11], 0 offen
            0xE0701004, 0x80020204, // buffer_store_dword v2, v4, s[8:11], 0 offen offset:4
            0xE0701008, 0x80020304, // buffer_store_dword v3, v4, s[8:11], 0 offen offset:8
            0xE070100C, 0x80020504, // buffer_store_dword v5, v4, s[8:11], 0 offen offset:12
            0xE0701010, 0x80020604, // buffer_store_dword v6, v4, s[8:11], 0 offen offset:16
            0xE0701014, 0x80020704, // buffer_store_dword v7, v4, s[8:11], 0 offen offset:20
            0xBF810000,             // s_endpgm
        ],
        StoreScalarResourceBase: 8,
        StoreBackingBytes: 2048);

    // GLC buffer accesses: each lane reads its dword coherently and writes it back 128
    // bytes on, also coherently. Nanite polls node data written by other groups this way.
    public static readonly Gen5ComputeFixture CoherentBufferCopy = new(
        "coherent-buffer-copy",
        [
            0xD7650000, 0x000100C1, // v_mbcnt_lo_u32_b32 v0, -1, 0         lane 0..31
            0x34000082,             // v_lshlrev_b32 v0, 2, v0
            0xE0305000, 0x80020100, // buffer_load_dword v1, v0, s[8:11], 0 offen glc
            0xBF8C3F70,             // s_waitcnt vmcnt(0)
            0xE0705080, 0x80020100, // buffer_store_dword v1, v0, s[8:11], 0 offen offset:128 glc
            0xBF810000,             // s_endpgm
        ],
        StoreScalarResourceBase: 8,
        StoreBackingBytes: 256);

    // A loop in a polling program (it has a GLC load): 5+4+3+2+1 = 15 per lane, stored at
    // offset 128.
    public static readonly Gen5ComputeFixture PollingLoop = new(
        "polling-loop",
        [
            0xD7650000, 0x000100C1, // v_mbcnt_lo_u32_b32 v0, -1, 0         lane 0..31
            0x34000082,             // v_lshlrev_b32 v0, 2, v0
            0xE0305000, 0x80020200, // buffer_load_dword v2, v0, s[8:11], 0 offen glc
            0xBF8C3F70,             // s_waitcnt vmcnt(0)
            0xBE800385,             // s_mov_b32 s0, 5
            0x7E020280,             // v_mov_b32 v1, 0
            0x4A020200,             // loop: v_add_nc_u32 v1, s0, v1
            0x81808100,             // s_sub_i32 s0, s0, 1
            0xBF078000,             // s_cmp_lg_u32 s0, 0
            0xBF85FFFC,             // s_cbranch_scc1 loop
            0xE0701080, 0x80020100, // buffer_store_dword v1, v0, s[8:11], 0 offen offset:128
            0xBF810000,             // s_endpgm
        ],
        StoreScalarResourceBase: 8,
        StoreBackingBytes: 256);

    // The shape of a persistent-threads kernel: one loop over most of the program that polls
    // coherent memory and writes device memory, and nothing after it but the end.
    public static readonly Gen5ComputeFixture PersistentQueueLoop = new(
        "persistent-queue-loop",
        [
            0xD7650000, 0x000100C1, // v_mbcnt_lo_u32_b32 v0, -1, 0         lane 0..31
            0x34000082,             // v_lshlrev_b32 v0, 2, v0
            0xBE800385,             // s_mov_b32 s0, 5
            0xE0305000, 0x80020200, // loop: buffer_load_dword v2, v0, s[8:11], 0 offen glc
            0xBF8C3F70,             // s_waitcnt vmcnt(0)
            0xBF800000,             // s_nop 0
            0xBF800000,             // s_nop 0
            0xBF800000,             // s_nop 0
            0xBF800000,             // s_nop 0
            0xBF800000,             // s_nop 0
            0xBF800000,             // s_nop 0
            0x4A040400,             // v_add_nc_u32 v2, s0, v2
            0xE0701080, 0x80020200, // buffer_store_dword v2, v0, s[8:11], 0 offen offset:128
            0x81808100,             // s_sub_i32 s0, s0, 1
            0xBF078000,             // s_cmp_lg_u32 s0, 0
            0xBF85FFF1,             // s_cbranch_scc1 loop
            0xBF810000,             // s_endpgm
        ],
        StoreScalarResourceBase: 8,
        StoreBackingBytes: 256);

    // Legacy MAD rounds the product, FMA does not. With a = 1 + 2^-12 and c = -(1 + 2^-11),
    // a*a = 1 + 2^-11 + 2^-24 rounds to 1 + 2^-11, so V_MAD_F32 gives exactly 0 while
    // V_FMA_F32 keeps the 2^-24. Each lane stores both.
    public static readonly Gen5ComputeFixture MadVersusFma = new(
        "mad-versus-fma",
        [
            0xD7650000, 0x000100C1, // v_mbcnt_lo_u32_b32 v0, -1, 0         lane 0..31
            0xBE8003FF, 0x3F800800, // s_mov_b32 s0, 1 + 2^-12
            0xBE8103FF, 0xBF801000, // s_mov_b32 s1, -(1 + 2^-11)
            0x7E040200,             // v_mov_b32 v2, s0
            0xD5410001, 0x00060502, // v_mad_f32 v1, v2, v2, s1
            0xD54B0003, 0x00060502, // v_fma_f32 v3, v2, v2, s1
            0x34080083,             // v_lshlrev_b32 v4, 3, v0
            0xE0701000, 0x80020104, // buffer_store_dword v1, v4, s[8:11], 0 offen
            0xE0701004, 0x80020304, // buffer_store_dword v3, v4, s[8:11], 0 offen offset:4
            0xBF810000,             // s_endpgm
        ],
        StoreScalarResourceBase: 8,
        StoreBackingBytes: 256);

    // The wave32 form: the compare's mask is the single SGPR s4, so s5 keeps 0xa5a5a5a5.
    public static readonly Gen5ComputeFixture Wave32Vop3CompareSgpr = new(
        "wave32-vop3-compare-sgpr",
        [
            0xD7650000, 0x000100C1, // v_mbcnt_lo_u32_b32 v0, -1, 0         lane 0..31
            0xBEEA03FF, 0x0F0F0F0F, // s_mov_b32 vcc_lo, 0x0f0f0f0f
            0xBE8503FF, 0xA5A5A5A5, // s_mov_b32 s5, 0xa5a5a5a5
            0x36040083,             // v_and_b32 v2, 3, v0
            0xD4C20004, 0x00010102, // v_cmp_eq_u32_e64 s4, v2, 0
            0xD5010001, 0x00110280, // v_cndmask_b32 v1, 0, 1, s4
            0xD5010002, 0x01A90280, // v_cndmask_b32 v2, 0, 1, vcc_lo
            0x7E060204,             // v_mov_b32 v3, s4
            0x7E0A0205,             // v_mov_b32 v5, s5
            0x7E0C026A,             // v_mov_b32 v6, vcc_lo
            0x7E0E0280,             // v_mov_b32 v7, 0
            0xBE863C04,             // s_and_saveexec_b32 s6, s4
            0x7E0E0281,             // v_mov_b32 v7, 1
            0xBEFE0306,             // s_mov_b32 exec_lo, s6
            0x34080085,             // v_lshlrev_b32 v4, 5, v0
            0xE0701000, 0x80020104, // buffer_store_dword v1, v4, s[8:11], 0 offen
            0xE0701004, 0x80020204, // buffer_store_dword v2, v4, s[8:11], 0 offen offset:4
            0xE0701008, 0x80020304, // buffer_store_dword v3, v4, s[8:11], 0 offen offset:8
            0xE070100C, 0x80020504, // buffer_store_dword v5, v4, s[8:11], 0 offen offset:12
            0xE0701010, 0x80020604, // buffer_store_dword v6, v4, s[8:11], 0 offen offset:16
            0xE0701014, 0x80020704, // buffer_store_dword v7, v4, s[8:11], 0 offen offset:20
            0xBF810000,             // s_endpgm
        ],
        StoreScalarResourceBase: 8,
        StoreBackingBytes: 1024);

    // VOP3 compares into NULL (writes nothing) and into EXEC. Per lane at lane * 16: the
    // EXEC bit after the NULL compare (all lanes), the VCC bit (preset, untouched), and 1
    // only on lanes the EXEC compare kept ((lane & 3) == 0).
    public static readonly Gen5ComputeFixture Wave64Vop3CompareNullAndExec = new(
        "wave64-vop3-compare-null-exec",
        [
            .. Wave64LaneIndex,
            0xBEEA03FF, 0x0F0F0F0F, // s_mov_b32 vcc_lo, 0x0f0f0f0f
            0xBEEB03FF, 0x0F0F0F0F, // s_mov_b32 vcc_hi, 0x0f0f0f0f
            0x36040083,             // v_and_b32 v2, 3, v0
            0x7E0E0280,             // v_mov_b32 v7, 0
            0xD4C2007D, 0x00010102, // v_cmp_eq_u32_e64 null, v2, 0
            0xD5010001, 0x01F90280, // v_cndmask_b32 v1, 0, 1, exec
            0xD4C2007E, 0x00010102, // v_cmp_eq_u32_e64 exec, v2, 0
            0x7E0E0281,             // v_mov_b32 v7, 1
            0xBEFE04C1,             // s_mov_b64 exec, -1
            0xD5010002, 0x01A90280, // v_cndmask_b32 v2, 0, 1, vcc
            0x34080084,             // v_lshlrev_b32 v4, 4, v0
            0xE0701000, 0x80020104, // buffer_store_dword v1, v4, s[8:11], 0 offen
            0xE0701004, 0x80020204, // buffer_store_dword v2, v4, s[8:11], 0 offen offset:4
            0xE0701008, 0x80020704, // buffer_store_dword v7, v4, s[8:11], 0 offen offset:8
            0xBF810000,             // s_endpgm
        ],
        StoreScalarResourceBase: 8,
        StoreBackingBytes: 1024);

    // VOP3B carry-out in wave32: v_add_co_u32 v3, s4, v0, -16 carries on lanes >= 16 into
    // the single SGPR s4 (s5 keeps 0xa5a5a5a5); a second add sends its carry to NULL and
    // must leave EXEC and VCC alone. Per lane at lane * 32: the s4 bit, s4, s5, vcc_lo,
    // exec_lo, the sum.
    public static readonly Gen5ComputeFixture Wave32Vop3CarryOutSgpr = new(
        "wave32-vop3-carry-out-sgpr",
        [
            0xD7650000, 0x000100C1, // v_mbcnt_lo_u32_b32 v0, -1, 0         lane 0..31
            0xBEEA03FF, 0x0F0F0F0F, // s_mov_b32 vcc_lo, 0x0f0f0f0f
            0xBE8503FF, 0xA5A5A5A5, // s_mov_b32 s5, 0xa5a5a5a5
            0xD70F0403, 0x0001A100, // v_add_co_u32 v3, s4, v0, -16
            0xD70F7D08, 0x0001A100, // v_add_co_u32 v8, null, v0, -16
            0xD5010001, 0x00110280, // v_cndmask_b32 v1, 0, 1, s4
            0x7E040204,             // v_mov_b32 v2, s4
            0x7E0A0205,             // v_mov_b32 v5, s5
            0x7E0C026A,             // v_mov_b32 v6, vcc_lo
            0x7E0E027E,             // v_mov_b32 v7, exec_lo
            0x34080085,             // v_lshlrev_b32 v4, 5, v0
            0xE0701000, 0x80020104, // buffer_store_dword v1, v4, s[8:11], 0 offen
            0xE0701004, 0x80020204, // buffer_store_dword v2, v4, s[8:11], 0 offen offset:4
            0xE0701008, 0x80020504, // buffer_store_dword v5, v4, s[8:11], 0 offen offset:8
            0xE070100C, 0x80020604, // buffer_store_dword v6, v4, s[8:11], 0 offen offset:12
            0xE0701010, 0x80020704, // buffer_store_dword v7, v4, s[8:11], 0 offen offset:16
            0xE0701014, 0x80020804, // buffer_store_dword v8, v4, s[8:11], 0 offen offset:20
            0xBF810000,             // s_endpgm
        ],
        StoreScalarResourceBase: 8,
        StoreBackingBytes: 1024);

    // The request path: the fixture's own resource plan, default specialization and layout.
    private const ulong RequestHash = 0x5348_4152_5045_4D55;
    private const uint UserDataCount = 16;

    public static ShaderCompileRequest RequestOrThrow(Gen5ShaderProgram program, ShaderStage stage, uint waveLaneCount = 32, uint localSizeX = 32, int requiredVertexOutputCount = 0,
        IReadOnlyList<Gen5PixelOutputBinding>? pixelOutputs = null, ResourceRuntimeInputs? runtimeInputs = null,
        Gen5NggStageInfo? primitiveShader = null, uint userDataBase = 0, uint scratchDwords = 0,
        uint localDataShareDwords = 0, uint pushDataStartDword = 0, uint positionExportControl = 0,
        bool supportsClipDistance = true, bool enableGraphicsSubgroupOperations = true,
        ShaderClipSpaceTransform clipSpace = default, bool pixelDepthExportEnable = false,
        ulong excludedUserDataRegisters = 0, bool ieeeMode = false, Gen5ColorExportFormat? pixelDepthExportFormat = null,
        bool pixelStencilExportEnable = false, bool pixelSampleMaskExportEnable = false,
        uint floatMode = ShaderCompileRequest.DefaultFloatMode, bool dx10Clamp = true, bool pixelGuestWave32 = false,
        uint meshPerVertexParameterMask = 0, IReadOnlyList<uint>? vertexInterpolants = null)
    {
        var plan = ShaderResourcePlan.Extract(program, stage, RequestHash, userDataBase, UserDataCount,
            excludedUserDataRegisters: excludedUserDataRegisters);
        var specialization = ResourceSpecialization.Default(plan.Info);
        if (runtimeInputs is not null)
        {
            var snapshot = new ResourceSnapshot();
            if (!ResourceMaterializer.Materialize(plan, runtimeInputs, ref snapshot, ref specialization, out var failure))
                throw new InvalidOperationException($"Fixture resource resolution failed: {failure}");
        }
        var resources = ResourceMaterializer.ApplyTo(plan, specialization);
        var layout = BindingLayout.Allocate(
            resources.Info,
            BindingLayout.CollectUserDataRegisters(program, userDataBase, UserDataCount, excludedUserDataRegisters),
            BindingLayout.UsesGlobalDataShare(program),
            ShaderCompileRequest.RequiresFlattenedTable(plan, resources),
            BindingLayout.ReadsShaderBase(program),
            pushDataStartDword);
        return new ShaderCompileRequest(plan, resources, layout)
        {
            WaveSize = waveLaneCount,
            IeeeMode = ieeeMode,
            FloatMode = floatMode,
            Dx10Clamp = dx10Clamp,
            PixelGuestWave32 = pixelGuestWave32,
            LocalSizeX = stage == ShaderStage.Compute ? localSizeX : 1,
            PixelOutputs = pixelOutputs ?? (stage == ShaderStage.Pixel ? [new Gen5PixelOutputBinding(0, 0, Gen5PixelOutputKind.Float)] : []),
            RequiredVertexOutputCount = requiredVertexOutputCount,
            PrimitiveShader = primitiveShader,
            MeshPerVertexParameterMask = meshPerVertexParameterMask,
            VertexInterpolants = vertexInterpolants ?? [],
            ScratchDwords = scratchDwords,
            LocalDataShareDwords = localDataShareDwords,
            PositionExportControl = positionExportControl,
            SupportsClipDistance = supportsClipDistance,
            EnableGraphicsSubgroupOperations = enableGraphicsSubgroupOperations,
            ClipSpace = clipSpace,
            PixelDepthExportEnable = pixelDepthExportEnable,
            PixelDepthExportFormat = pixelDepthExportFormat,
            PixelStencilExportEnable = pixelStencilExportEnable,
            PixelSampleMaskExportEnable = pixelSampleMaskExportEnable,
        };
    }

    public static Gen5MslShader CompileRequestOrThrow(Gen5ComputeFixture fixture, uint waveLaneCount = 32, uint localSizeX = 32,
        uint localDataShareDwords = 0, uint pushDataStartDword = 0, uint floatMode = ShaderCompileRequest.DefaultFloatMode)
    {
        var request = CreateComputeRequest(fixture, waveLaneCount, localSizeX, localDataShareDwords, pushDataStartDword, floatMode);
        if (!Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error))
        {
            throw new InvalidOperationException($"[{fixture.Name}] MSL request emit failed: {error}");
        }

        return shader;
    }

    public static ShaderCompileRequest CreateComputeRequest(Gen5ComputeFixture fixture, uint waveLaneCount = 32, uint localSizeX = 32,
        uint localDataShareDwords = 0, uint pushDataStartDword = 0, uint floatMode = ShaderCompileRequest.DefaultFloatMode)
    {
        var program = DecodeOrThrow(fixture);
        var hasBufferDescriptor = program.Instructions.Any(instruction => instruction.Control is Gen5BufferMemoryControl);
        var userData = new uint[UserDataCount];
        if (hasBufferDescriptor && fixture.StoreBackingBytes != 0)
        {
            userData[fixture.StoreScalarResourceBase] = 0x200000;
            userData[fixture.StoreScalarResourceBase + 2] = (uint)fixture.StoreBackingBytes;
            userData[fixture.StoreScalarResourceBase + 3] = DescriptorConstants.IdentityDestinationSelect;
        }
        return RequestOrThrow(program, ShaderStage.Compute, waveLaneCount, localSizeX,
            runtimeInputs: hasBufferDescriptor ? new ResourceRuntimeInputs { UserData = userData } : null,
            localDataShareDwords: localDataShareDwords,
            pushDataStartDword: pushDataStartDword,
            floatMode: floatMode);
    }

    public static Gen5MslShader CompileStageRequestOrThrow(uint[] words, ShaderStage stage, int requiredVertexOutputCount = 0)
    {
        var memory = new FakeGuestMemory();
        memory.AddRegion(ProgramAddress, words);
        var ctx = new CpuContext(memory, Generation.Gen5);
        if (!Gen5ShaderTranslator.TryDecodeProgram(ctx, ProgramAddress, out var program, out var decodeError))
        {
            throw new InvalidOperationException($"[{stage}] decode failed: {decodeError}");
        }

        var request = RequestOrThrow(program!, stage, requiredVertexOutputCount: requiredVertexOutputCount);
        if (!Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error))
        {
            throw new InvalidOperationException($"[{stage}] MSL request emit failed: {error}");
        }

        return shader;
    }

    public static Gen5ShaderProgram DecodeOrThrow(Gen5ComputeFixture fixture)
    {
        var memory = new FakeGuestMemory();
        memory.AddRegion(ProgramAddress, fixture.Words);
        var ctx = new CpuContext(memory, Generation.Gen5);

        if (!Gen5ShaderTranslator.TryDecodeProgram(ctx, ProgramAddress, out var program, out var decodeError))
        {
            throw new InvalidOperationException($"[{fixture.Name}] decode failed: {decodeError}");
        }

        return program!;
    }
}
