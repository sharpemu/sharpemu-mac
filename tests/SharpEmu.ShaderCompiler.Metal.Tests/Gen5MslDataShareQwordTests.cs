// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

// The 64-bit LDS atomics in compute (threadgroup LDS, the qword locks of
// Gen5MslTranslator.EmitLdsLocked), DS_CONDXCHG32_RTN_B64, and the GDS-only
// DS_ORDERED_COUNT and DS_GWS_BARRIER, run on the device against C# models of the
// RDNA2 ISA 12.13 pseudo-code. Instruction words are llvm-mc -mcpu=gfx1030 output.
public sealed class Gen5MslDataShareQwordTests(ITestOutputHelper output)
{
    private const int Lanes = 32;
    private const int OutputOffset = 1024;
    private const int BufferBytes = 2048;

    public static TheoryData<uint> QwordOpcodes()
    {
        var data = new TheoryData<uint>();
        // Non-returning 0x40-0x53, returning 0x60-0x73 (gaps are other opcodes).
        foreach (var opcode in new uint[]
                 {
                     0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x4B, 0x4C, 0x50, 0x51, 0x52, 0x53,
                     0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x6B, 0x6C, 0x6D, 0x70, 0x71, 0x72, 0x73,
                     0x7E,
                 })
        {
            data.Add(opcode);
        }

        return data;
    }

    // Each lane seeds its own LDS qword with ds_write_b64, runs the atomic with DATA0 and
    // DATA1 from the buffer, and stores the qword and what the atomic returned.
    [Theory]
    [MemberData(nameof(QwordOpcodes))]
    public void EachSharedLdsQwordAtomicMatchesTheIsa(uint opcode)
    {
        var returns = opcode >= 0x60;
        var fixture = new Gen5ComputeFixture($"ds-qword-{opcode:X2}",
        [
            0x34020083,                         // v_lshlrev_b32 v1, 3, v0      LDS qword of the lane
            0x34040085,                         // v_lshlrev_b32 v2, 5, v0      its inputs
            0x34060084,                         // v_lshlrev_b32 v3, 4, v0      its outputs
            0xE0381000, 0x80020402,             // buffer_load_dwordx4 v[4:7], v2, s[8:11], 0 offen   memory, DATA0
            0xE0341010, 0x80020802,             // buffer_load_dwordx2 v[8:9], v2, s[8:11], 0 offen offset:16   DATA1
            0x7E1402FF, 0xA5A5A5A5,             // v_mov_b32 v10, 0xa5a5a5a5
            0x7E1602FF, 0xA5A5A5A5,             // v_mov_b32 v11, 0xa5a5a5a5
            0xBF8C3F70,                         // s_waitcnt vmcnt(0)
            0xD9340000, 0x00000401,             // ds_write_b64 v1, v[4:5]
            0xD8000000u | opcode << 18, opcode == 0x7E ? 0x0A000601u : 0x0A080601u, // ds_* [v[10:11]], v1, v[6:7][, v[8:9]]
            0xBF8CC07F,                         // s_waitcnt lgkmcnt(0)
            0xD9D80000, 0x0C000001,             // ds_read_b64 v[12:13], v1
            0xBF8CC07F,                         // s_waitcnt lgkmcnt(0)
            0xE0741400, 0x80020C03,             // buffer_store_dwordx2 v[12:13], v3, s[8:11], 0 offen offset:1024
            0xE0741408, 0x80020A03,             // buffer_store_dwordx2 v[10:11], v3, s[8:11], 0 offen offset:1032
            0xBF810000,                         // s_endpgm
        ], 8, BufferBytes);
        var shader = Compile(fixture, Lanes);
        var splits = !returns && opcode is 0x40 or 0x41 or 0x49 or 0x4A or 0x4B;
        Assert.Equal(!splits && opcode != 0x7E, shader.UsesPairLocks);
        if (shader.UsesPairLocks) Assert.True(shader.WritesStepReport);
        Assert.Equal(!splits && opcode != 0x7E, shader.Source.Contains("sharpemu_lds_try_lock(", StringComparison.Ordinal));
        if (!RequireDevice())
        {
            return;
        }

        var (olds, data0, data1) = Operands(opcode);
        var input = new byte[BufferBytes];
        for (var lane = 0; lane < Lanes; lane++)
        {
            BinaryPrimitives.WriteUInt64LittleEndian(input.AsSpan(lane * 32), olds[lane]);
            BinaryPrimitives.WriteUInt64LittleEndian(input.AsSpan(lane * 32 + 8), data0[lane]);
            BinaryPrimitives.WriteUInt64LittleEndian(input.AsSpan(lane * 32 + 16), data1[lane]);
        }

        var result = Execute(shader, input, Lanes, 1, out var stepCapHits);
        Assert.Equal(0u, stepCapHits);
        for (var lane = 0; lane < Lanes; lane++)
        {
            var expected = Reference(opcode, olds[lane], data0[lane], data1[lane]);
            var memory = BinaryPrimitives.ReadUInt64LittleEndian(result.AsSpan(OutputOffset + lane * 16));
            var returned = BinaryPrimitives.ReadUInt64LittleEndian(result.AsSpan(OutputOffset + lane * 16 + 8));
            var expectedReturn = returns ? olds[lane] : 0xA5A5A5A5_A5A5A5A5ul;
            if (memory != expected || returned != expectedReturn)
            {
                output.WriteLine($"lane {lane}: old 0x{olds[lane]:X} d0 0x{data0[lane]:X} d1 0x{data1[lane]:X} -> 0x{memory:X} (expected 0x{expected:X}), returned 0x{returned:X}");
            }

            Assert.Equal(expected, memory);
            Assert.Equal(expectedReturn, returned);
        }
    }

    // 256 lanes (eight SIMD-groups) on one LDS qword: even lanes DS_INC_RTN_U64 it, odd
    // lanes DS_ADD_RTN_U32 its high dword. The INC is a locked load, compute and store;
    // were a 32-bit add to land between them it would be lost. With the low dword starting
    // at 0 no INC carries, so the qword must end at (128, high + 128) and both kinds of
    // lane must have seen every count exactly once.
    [Fact]
    public void ContendedQwordAndDwordAtomicsOnOneLdsQwordAreAtomic()
    {
        const uint high = 0x1234_0000;
        var fixture = new Gen5ComputeFixture("ds-qword-mixed",
        [
            0x7E020280,                         // v_mov_b32 v1, 0
            0x34060083,                         // v_lshlrev_b32 v3, 3, v0
            0x7E080280,                         // v_mov_b32 v4, 0
            0x7E0A02FF, high,                   // v_mov_b32 v5, high
            0xD9340000, 0x00000401,             // ds_write_b64 v1, v[4:5]
            0xBF8CC07F,                         // s_waitcnt lgkmcnt(0)
            0xBF8A0000,                         // s_barrier
            0x7E0C02C1,                         // v_mov_b32 v6, -1
            0x7E0E02C1,                         // v_mov_b32 v7, -1
            0x7E1C0284,                         // v_mov_b32 v14, 4
            0x7E1E0281,                         // v_mov_b32 v15, 1
            0x36200081,                         // v_and_b32 v16, 1, v0
            0x7D842080,                         // v_cmp_eq_u32 vcc_lo, 0, v16
            0xBE953C6A,                         // s_and_saveexec_b32 s21, vcc_lo
            0xD98C0000, 0x0A000601,             // ds_inc_rtn_u64 v[10:11], v1, v[6:7]      even lanes
            0x8A7E6A15,                         // s_andn2_b32 exec_lo, s21, vcc_lo
            0xD8800000, 0x0A000F0E,             // ds_add_rtn_u32 v10, v14, v15             odd lanes
            0xBEFE0315,                         // s_mov_b32 exec_lo, s21
            0xBF8CC07F,                         // s_waitcnt lgkmcnt(0)
            0xBF8A0000,                         // s_barrier
            0xD9D80000, 0x0C000001,             // ds_read_b64 v[12:13], v1
            0xBF8CC07F,                         // s_waitcnt lgkmcnt(0)
            0xE0741040, 0x80020A03,             // buffer_store_dwordx2 v[10:11], v3, s[8:11], 0 offen offset:64
            0xE0740000, 0x80020C00,             // buffer_store_dwordx2 v[12:13], off, s[8:11], 0
            0xBF810000,                         // s_endpgm
        ], 8, 64 + 256 * 8);
        var shader = Compile(fixture, 256);
        Assert.Contains("sharpemu_lds_try_lock(", shader.Source, StringComparison.Ordinal);
        if (!RequireDevice())
        {
            return;
        }

        var result = Execute(shader, new byte[64 + 256 * 8], 256, 1, out var stepCapHits);
        Assert.Equal(0u, stepCapHits);
        Assert.Equal(128ul | ((ulong)(high + 128) << 32), BinaryPrimitives.ReadUInt64LittleEndian(result));
        var incs = new List<ulong>();
        var adds = new List<uint>();
        for (var lane = 0; lane < 256; lane++)
        {
            var returned = BinaryPrimitives.ReadUInt64LittleEndian(result.AsSpan(64 + lane * 8));
            if ((lane & 1) == 0)
            {
                incs.Add(returned & 0xFFFF_FFFF);
            }
            else
            {
                adds.Add((uint)returned);
            }
        }

        incs.Sort();
        adds.Sort();
        Assert.Equal(Enumerable.Range(0, 128).Select(value => (ulong)value), incs);
        Assert.Equal(Enumerable.Range(0, 128).Select(value => high + (uint)value), adds);
    }

    // A lock-free 64-bit counter built from DS_CMPST_RTN_B64 by 256 lanes: each retries
    // old -> old + 0xFFFFFFFF (a carry every time) with the value it got back until its
    // swap wins, then stores the value it replaced.
    [Fact]
    public void ContendedLdsCompareStoreLoopsCountExactly()
    {
        const ulong initial = 0x0000_0007_FFFF_FF00;
        const ulong step = 0xFFFF_FFFF;
        var fixture = new Gen5ComputeFixture("ds-cmpst-loop",
        [
            0x7E020280,                         // v_mov_b32 v1, 0
            0x34060083,                         // v_lshlrev_b32 v3, 3, v0
            0x7E0802FF, unchecked((uint)initial), // v_mov_b32 v4, initial low
            0x7E0A02FF, (uint)(initial >> 32),  // v_mov_b32 v5, initial high
            0xD9340000, 0x00000401,             // ds_write_b64 v1, v[4:5]
            0xBF8CC07F,                         // s_waitcnt lgkmcnt(0)
            0xBF8A0000,                         // s_barrier
            0xD9D80000, 0x0C000001,             // ds_read_b64 v[12:13], v1
            0xBF8CC07F,                         // s_waitcnt lgkmcnt(0)
            0xBE96037E,                         // s_mov_b32 s22, exec_lo
            // retry:
            0xD70F6A08, 0x0001830C,             // v_add_co_u32 v8, vcc_lo, v12, -1
            0x50121A80,                         // v_add_co_ci_u32 v9, vcc_lo, 0, v13, vcc_lo
            0x7E0C030C,                         // v_mov_b32 v6, v12
            0x7E0E030D,                         // v_mov_b32 v7, v13
            0xD9C00000, 0x0A080601,             // ds_cmpst_rtn_b64 v[10:11], v1, v[6:7], v[8:9]
            0xBF8CC07F,                         // s_waitcnt lgkmcnt(0)
            0x7DC40D0A,                         // v_cmp_eq_u64 vcc_lo, v[10:11], v[6:7]
            0xBE953C6A,                         // s_and_saveexec_b32 s21, vcc_lo
            0xE0741040, 0x80020A03,             // buffer_store_dwordx2 v[10:11], v3, s[8:11], 0 offen offset:64
            0x8A7E6A15,                         // s_andn2_b32 exec_lo, s21, vcc_lo
            0x7E18030A,                         // v_mov_b32 v12, v10
            0x7E1A030B,                         // v_mov_b32 v13, v11
            0xBF89FFF0,                         // s_cbranch_execnz retry
            0xBEFE0316,                         // s_mov_b32 exec_lo, s22
            0xBF8A0000,                         // s_barrier
            0xD9D80000, 0x0C000001,             // ds_read_b64 v[12:13], v1
            0xBF8CC07F,                         // s_waitcnt lgkmcnt(0)
            0xE0740000, 0x80020C00,             // buffer_store_dwordx2 v[12:13], off, s[8:11], 0
            0xBF810000,                         // s_endpgm
        ], 8, 64 + 256 * 8);
        var shader = Compile(fixture, 256);
        if (!RequireDevice())
        {
            return;
        }

        var result = Execute(shader, new byte[64 + 256 * 8], 256, 1, out var stepCapHits);
        Assert.Equal(0u, stepCapHits);
        Assert.Equal(initial + 256 * step, BinaryPrimitives.ReadUInt64LittleEndian(result));
        var seen = Enumerable.Range(0, 256)
            .Select(lane => BinaryPrimitives.ReadUInt64LittleEndian(result.AsSpan(64 + lane * 8)))
            .Order()
            .ToArray();
        Assert.Equal(Enumerable.Range(0, 256).Select(index => initial + (ulong)index * step), seen);
    }

    // 64 workgroups of one wave each add 32 to GDS dword 1 with DS_ORDERED_COUNT; every lane
    // stores base + lane at that index. The waves' ranges must tile [0, 2048) exactly.
    [Fact]
    public void OrderedCountGivesEachWaveADisjointRange()
    {
        const uint groups = 64;
        var fixture = new Gen5ComputeFixture("ds-ordered-count",
        [
            0xBEFC0380,                         // s_mov_b32 m0, 0
            0x7E0402A0,                         // v_mov_b32 v2, 32
            0xD8FE0004, 0x03000002,             // ds_ordered_count v3, v2 offset:4 gds
            0xBF8CC07F,                         // s_waitcnt lgkmcnt(0)
            0x4A080103,                         // v_add_nc_u32 v4, v3, v0
            0x340A0882,                         // v_lshlrev_b32 v5, 2, v4
            0xE0701000, 0x80020405,             // buffer_store_dword v4, v5, s[8:11], 0 offen
            0xBF810000,                         // s_endpgm
        ], 8, (int)groups * Lanes * 4);
        var shader = Compile(fixture, Lanes);
        Assert.Contains("atomic_fetch_add_explicit((device atomic_uint*)(sharpemu_gds + ", shader.Source, StringComparison.Ordinal);
        if (!RequireDevice())
        {
            return;
        }

        var gds = MetalNative.CreateTable(DescriptorBindingKind.GlobalDataShare, new byte[256], 64);
        var result = Execute(shader, new byte[groups * Lanes * 4], Lanes, groups, out _, [gds]);
        for (var slot = 0; slot < groups * Lanes; slot++)
        {
            Assert.Equal((uint)slot, BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan(slot * 4)));
        }

        Assert.Equal(groups * Lanes, MetalNative.ReadWords(gds.Buffer, 2)[1]);
    }

    [Fact]
    public void OrderedSwapExchangesTheCounter()
    {
        // ds_ordered_count v3, v2 offset:4100 gds: OFFSET1 bit 4 selects swap.
        var fixture = new Gen5ComputeFixture("ds-ordered-swap",
        [
            0xBEFC0380, 0x7E0402A0, 0xD8FE1004, 0x03000002, 0xBF8CC07F, 0xBF810000,
        ], 8, 64);
        var shader = Compile(fixture, Lanes);
        Assert.Contains("atomic_exchange_explicit((device atomic_uint*)(sharpemu_gds + ", shader.Source, StringComparison.Ordinal);
    }

    // Nine one-lane workgroups count their arrival, meet at DS_GWS_BARRIER with DATA 7 and
    // then read the count. The table starts at zero, so the first wave to arrive finds the
    // barrier open and arms it for eight; the other eight wait for the last of them, so
    // at least eight waves read all nine arrivals.
    [Fact]
    public void GwsBarrierHoldsWavesUntilTheCountArrives()
    {
        const uint groups = 9;
        var fixture = new Gen5ComputeFixture("ds-gws-barrier",
        [
            0xBEFE0381,                         // s_mov_b32 exec_lo, 1
            0x7E0C0281,                         // v_mov_b32 v6, 1
            0xE0C84008, 0x80020600,             // buffer_atomic_add v6, off, s[8:11], 0 offset:8 glc
            0xBF8C3F70,                         // s_waitcnt vmcnt(0)
            0xBEFC0380,                         // s_mov_b32 m0, 0
            0x7E040287,                         // v_mov_b32 v2, 7
            0xD8760000, 0x00000002,             // ds_gws_barrier v2 gds
            0xBF8CC07F,                         // s_waitcnt lgkmcnt(0)
            0xE0304008, 0x80020700,             // buffer_load_dword v7, off, s[8:11], 0 offset:8 glc
            0xBF8C3F70,                         // s_waitcnt vmcnt(0)
            0x34100C82,                         // v_lshlrev_b32 v8, 2, v6
            0xE0701040, 0x80020708,             // buffer_store_dword v7, v8, s[8:11], 0 offen offset:64
            0xBF810000,                         // s_endpgm
        ], 8, 128);
        var shader = Compile(fixture, Lanes);
        Assert.Contains("sharpemu_gws_barrier(", shader.Source, StringComparison.Ordinal);
        if (!RequireDevice())
        {
            return;
        }

        var gds = MetalNative.CreateTable(DescriptorBindingKind.GlobalDataShare, new byte[256], 64);
        var result = Execute(shader, new byte[128], Lanes, groups, out var stepCapHits, [gds]);
        Assert.Equal(0u, stepCapHits);
        var seen = Enumerable.Range(0, (int)groups)
            .Select(ticket => BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan(64 + ticket * 4)))
            .ToArray();
        Assert.True(seen.Count(value => value == groups) >= groups - 1, string.Join(", ", seen));
    }

    [Theory]
    [InlineData(0x18u, "sharpemu_gws_sema_release_all(")]
    [InlineData(0x19u, "sharpemu_gws_init(")]
    [InlineData(0x1Au, "sharpemu_gws_sema_release(")]
    [InlineData(0x1Bu, "sharpemu_gws_sema_release(")]
    [InlineData(0x1Cu, "sharpemu_gws_sema_p(")]
    public void EveryGwsOperationCompiles(uint opcode, string call)
    {
        var fixture = new Gen5ComputeFixture("ds-gws",
        [
            0xBEFC0380, 0x7E040287, 0xD8020003u | opcode << 18, 0x00000002, 0xBF8CC07F, 0xBF810000,
        ], 8, 64);
        var shader = Compile(fixture, Lanes);
        Assert.Contains(call, shader.Source, StringComparison.Ordinal);
    }

    private static (ulong[] Olds, ulong[] Data0, ulong[] Data1) Operands(uint opcode)
    {
        var floating = opcode is 0x51 or 0x52 or 0x53 or 0x71 or 0x72 or 0x73;
        ulong[] values = floating
            ?
            [
                Bits(0.0), Bits(-0.0), Bits(1.5), Bits(-2.25), Bits(double.PositiveInfinity), Bits(double.NegativeInfinity),
                0x7FF8_0000_0000_0000, 0x7FF0_0000_0000_0001, 1, 0x8000_0000_0000_0001, Bits(double.MaxValue),
                0x0010_0000_0000_0000, Bits(1e300), Bits(-1e-300), Bits(1.5000000000000002), Bits(3.0),
            ]
            : opcode == 0x7E
                ? [0, 0x8000_0000, 0x8000_0001_FFFF_FFFF, 0x1234_5678_8765_4321, 0xFFFF_FFFF_FFFF_FFFF, 0x7FFF_FFFF_8000_0000, 1, 0x8000_0000_0000_0000]
                :
                [
                    0, 1, 2, 0xFFFF_FFFF, 0x1_0000_0000, 0x7FFF_FFFF_FFFF_FFFF, 0x8000_0000_0000_0000,
                    0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_0000_0000, 0x1234_5678_9ABC_DEF0, 0x0000_0001_FFFF_FFFF,
                    0xFFFF_FFFE_FFFF_FFFF, 1000, 999, 0x00FF_00FF_00FF_00FF, 0xF0F0_F0F0_F0F0_F0F0,
                ];
        var olds = new ulong[Lanes];
        var data0 = new ulong[Lanes];
        var data1 = new ulong[Lanes];
        for (var lane = 0; lane < Lanes; lane++)
        {
            olds[lane] = values[lane % values.Length];
            data0[lane] = values[(lane * 7 + 3) % values.Length];
            data1[lane] = values[(lane * 5 + 1) % values.Length];
            if ((lane & 1) == 0 && opcode is 0x50 or 0x51 or 0x70 or 0x71)
            {
                data0[lane] = olds[lane];   // DS_CMPST compares memory with DATA0
            }
        }

        if (floating)
        {
            data0[1] = Bits(-0.0);                      // memory +0.0, -0.0 compares equal
            olds[3] = 0x7FF8_0000_0000_0000;            // a NaN equals nothing, itself included
            data0[3] = olds[3];
        }

        return (olds, data0, data1);
    }

    // RDNA2 ISA 12.13; DS_MIN/MAX_F64 take the min/max of DATA0 and memory (see
    // Gen5MslTranslator.DataShareFloatMinMax), DS_CONDXCHG32_RTN_B64 the GFX9 ISA's
    // per-dword conditional exchange.
    private static ulong Reference(uint opcode, ulong old, ulong data0, ulong data1) => (opcode & ~0x20u) switch
    {
        0x40 => old + data0,
        0x41 => old - data0,
        0x42 => data0 - old,
        0x43 => old >= data0 ? 0 : old + 1,
        0x44 => old == 0 || old > data0 ? data0 : old - 1,
        0x45 => (ulong)Math.Min((long)old, (long)data0),
        0x46 => (ulong)Math.Max((long)old, (long)data0),
        0x47 => Math.Min(old, data0),
        0x48 => Math.Max(old, data0),
        0x49 => old & data0,
        0x4A => old | data0,
        0x4B => old ^ data0,
        0x4C => (old & ~data0) | data1,
        0x4D => data0,
        0x50 => old == data0 ? data1 : old,
        0x51 => D(old) == D(data0) ? data1 : old,
        0x52 => D(data0) < D(old) ? data0 : old,
        0x53 => D(data0) > D(old) ? data0 : old,
        0x5E => CondExchange((uint)old, (uint)data0) | ((ulong)CondExchange((uint)(old >> 32), (uint)(data0 >> 32)) << 32),
        _ => throw new ArgumentOutOfRangeException(nameof(opcode)),
    };

    private static uint CondExchange(uint old, uint data) => (data & 0x8000_0000) != 0 ? data & 0x7FFF_FFFF : old;

    private static double D(ulong bits) => BitConverter.UInt64BitsToDouble(bits);

    private static ulong Bits(double value) => BitConverter.DoubleToUInt64Bits(value);

    private bool RequireDevice()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Atomic execution requires the exclusive Metal device gate.");
            return false;
        }

        Assert.True(MetalNative.IsAvailable, "The required execution gate needs a Metal device.");
        return true;
    }

    private static Gen5MslShader Compile(Gen5ComputeFixture fixture, uint localSizeX)
    {
        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(fixture, localSizeX: localSizeX);
        if (MetalNative.IsAvailable)
        {
            Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var error), error + "\n" + shader.Source);
            if (library != 0) MetalNative.ReleaseCollisionLibrary(library);
        }

        return shader;
    }

    private static byte[] Execute(Gen5MslShader shader, byte[] initial, uint threads, uint groups, out uint stepCapHits, IReadOnlyList<DeviceTable>? tables = null)
    {
        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var compileError), compileError);
        try
        {
            Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, initial,
                new byte[PushData.ByteSize], shader.ArgumentLayout!, threads, out var result, out stepCapHits, out var runError,
                threadgroupCount: groups, tables: tables), runError);
            return result;
        }
        finally
        {
            MetalNative.ReleaseCollisionLibrary(library);
        }
    }
}
