// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

// LDS atomics over threadgroup memory. Each fixture seeds one LDS word, runs
// the atomic on it and stores both the returned value and the word the atomic
// left behind, so the oracle covers the operation and the return contract.
public sealed class Gen5MslDataShareAtomicTests(ITestOutputHelper output)
{
    private const uint Sentinel = 0xA5A5A5A5;
    private const uint Initial = 0x00001234;
    private const uint Operand = 0x00000011;
    private const uint Compare = 0x0000ABCD;

    [Theory]
    // Returning forms: VDST takes the pre-operation word.
    [InlineData("DsAddRtnU32", 0x20u, Initial, Operand, Compare, Initial, 0x00001245u)]
    [InlineData("DsSubRtnU32", 0x21u, Initial, Operand, Compare, Initial, 0x00001223u)]
    // INC/DEC wrap against DATA0 (RDNA2 ISA DS_INC_U32/DS_DEC_U32): Initial >= Operand, so INC wraps to 0
    // and DEC (Initial > Operand) takes the limit; below the limit they step by one.
    [InlineData("DsIncRtnU32", 0x23u, Initial, Operand, Compare, Initial, 0u)]
    [InlineData("DsIncRtnU32-below", 0x23u, 5u, Operand, Compare, 5u, 6u)]
    [InlineData("DsDecRtnU32-below", 0x24u, 5u, Operand, Compare, 5u, 4u)]
    [InlineData("DsDecRtnU32-zero", 0x24u, 0u, Operand, Compare, 0u, Operand)]
    [InlineData("DsIncRtnU32-equal", 0x23u, Operand, Operand, Compare, Operand, 0u)]
    [InlineData("DsDecRtnU32", 0x24u, Initial, Operand, Compare, Initial, Operand)]
    [InlineData("DsMinRtnI32", 0x25u, Initial, Operand, Compare, Initial, Operand)]
    [InlineData("DsMaxRtnI32", 0x26u, Initial, Operand, Compare, Initial, Initial)]
    [InlineData("DsMinRtnU32", 0x27u, Initial, Operand, Compare, Initial, Operand)]
    [InlineData("DsMaxRtnU32", 0x28u, Initial, Operand, Compare, Initial, Initial)]
    [InlineData("DsAndRtnB32", 0x29u, Initial, Operand, Compare, Initial, 0x00000010u)]
    [InlineData("DsOrRtnB32", 0x2Au, Initial, Operand, Compare, Initial, 0x00001235u)]
    [InlineData("DsXorRtnB32", 0x2Bu, Initial, Operand, Compare, Initial, 0x00001225u)]
    [InlineData("DsWrxchgRtnB32", 0x2Du, Initial, Operand, Compare, Initial, Operand)]
    // The sign of the operands separates the signed from the unsigned minimum.
    [InlineData("DsMinRtnI32-negative", 0x25u, 0x80000000u, Operand, Compare, 0x80000000u, 0x80000000u)]
    [InlineData("DsMinRtnU32-negative", 0x27u, 0x80000000u, Operand, Compare, 0x80000000u, Operand)]
    [InlineData("DsMaxRtnI32-negative", 0x26u, 0x80000000u, Operand, Compare, 0x80000000u, Operand)]
    [InlineData("DsMaxRtnU32-negative", 0x28u, 0x80000000u, Operand, Compare, 0x80000000u, 0x80000000u)]
    // DS_CMPST reads DATA0 as the comparator and DATA1 as the new value.
    [InlineData("DsCmpstRtnB32-miss", 0x30u, Initial, Operand, Compare, Initial, Initial)]
    [InlineData("DsCmpstRtnB32-hit", 0x30u, Initial, Initial, Compare, Initial, Compare)]
    // Non-returning forms leave VDST alone.
    [InlineData("DsAddU32", 0x00u, Initial, Operand, Compare, Sentinel, 0x00001245u)]
    [InlineData("DsSubU32", 0x01u, Initial, Operand, Compare, Sentinel, 0x00001223u)]
    [InlineData("DsIncU32", 0x03u, Initial, Operand, Compare, Sentinel, 0u)]
    [InlineData("DsDecU32", 0x04u, Initial, Operand, Compare, Sentinel, Operand)]
    [InlineData("DsMinI32", 0x05u, Initial, Operand, Compare, Sentinel, Operand)]
    [InlineData("DsMaxI32", 0x06u, Initial, Operand, Compare, Sentinel, Initial)]
    [InlineData("DsMinU32", 0x07u, Initial, Operand, Compare, Sentinel, Operand)]
    [InlineData("DsMaxU32", 0x08u, Initial, Operand, Compare, Sentinel, Initial)]
    [InlineData("DsAndB32", 0x09u, Initial, Operand, Compare, Sentinel, 0x00000010u)]
    [InlineData("DsOrB32", 0x0Au, Initial, Operand, Compare, Sentinel, 0x00001235u)]
    [InlineData("DsXorB32", 0x0Bu, Initial, Operand, Compare, Sentinel, 0x00001225u)]
    [InlineData("DsCmpstB32-hit", 0x10u, Initial, Initial, Compare, Sentinel, Compare)]
    // DS_MIN/MAX_F32 keep the smaller/larger of DATA0 and memory; DATA1 is not compared
    // (see Gen5MslTranslator.DataShareFloatMinMax), so it cannot change the outcome.
    [InlineData("DsMinF32-replace", 0x12u, 0x40000000u, 0x3F800000u, 0x3F000000u, Sentinel, 0x3F800000u)]
    [InlineData("DsMinF32-data1-ignored", 0x12u, 0x40000000u, 0x3F800000u, 0x40800000u, Sentinel, 0x3F800000u)]
    [InlineData("DsMinF32-keep", 0x12u, 0x3F000000u, 0x3F800000u, 0x3E800000u, Sentinel, 0x3F000000u)]
    [InlineData("DsMaxF32-replace", 0x13u, 0x3F000000u, 0x3F800000u, 0x40800000u, Sentinel, 0x3F800000u)]
    [InlineData("DsMaxF32-data1-ignored", 0x13u, 0x40000000u, 0x3F800000u, 0x3F000000u, Sentinel, 0x40000000u)]
    [InlineData("DsMaxF32-keep", 0x13u, 0x40000000u, 0x3F800000u, 0x40800000u, Sentinel, 0x40000000u)]
    [InlineData("DsMinRtnF32-replace", 0x32u, 0x40000000u, 0x3F800000u, 0x3F000000u, 0x40000000u, 0x3F800000u)]
    [InlineData("DsMaxRtnF32-keep", 0x33u, 0x40000000u, 0x3F800000u, 0x40800000u, 0x40000000u, 0x40000000u)]
    // DS_RSUB subtracts memory from DATA0.
    [InlineData("DsRsubRtnU32", 0x22u, Initial, Operand, Compare, Initial, 0xFFFFEDDDu)]
    [InlineData("DsRsubU32", 0x02u, Initial, Operand, Compare, Sentinel, 0xFFFFEDDDu)]
    // DS_MSKOR: (memory & ~DATA0) | DATA1.
    [InlineData("DsMskorRtnB32", 0x2Cu, Initial, Operand, Compare, Initial, 0x0000BBEDu)]
    [InlineData("DsMskorB32", 0x0Cu, Initial, Operand, Compare, Sentinel, 0x0000BBEDu)]
    // DS_CMPST_F32 compares as floats: DATA0 the comparator, DATA1 the new value.
    [InlineData("DsCmpstRtnF32-hit", 0x31u, 0x3F800000u, 0x3F800000u, 0x40000000u, 0x3F800000u, 0x40000000u)]
    [InlineData("DsCmpstRtnF32-miss", 0x31u, 0x3F800000u, 0x40400000u, 0x40000000u, 0x3F800000u, 0x3F800000u)]
    [InlineData("DsCmpstF32-hit", 0x11u, 0x3F800000u, 0x3F800000u, 0x40000000u, Sentinel, 0x40000000u)]
    // DS_ADD_F32: 1.0 + 2.0.
    [InlineData("DsAddF32", 0x15u, 0x3F800000u, 0x40000000u, Compare, Sentinel, 0x40400000u)]
    [InlineData("DsAddRtnF32", 0x55u, 0x3F800000u, 0x40000000u, Compare, 0x3F800000u, 0x40400000u)]
    // DS_WRAP_RTN: memory >= DATA0 ? memory - DATA0 : memory + DATA1.
    [InlineData("DsWrapRtnB32-wrap", 0x34u, Initial, Operand, Compare, Initial, 0x00001223u)]
    [InlineData("DsWrapRtnB32-add", 0x34u, 5u, Operand, Compare, 5u, 0x0000ABD2u)]
    public void OneWordAtomicsReturnThePreOperationWordAndLeaveTheOperationResult(
        string name, uint opcode, uint initial, uint data0, uint data1, uint expectedReturn, uint expectedMemory)
    {
        var fixture = OneWordFixture(name, opcode, initial, data0, data1);
        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(fixture);
        Assert.Contains("sharpemu_lds", shader.Source);
        if (!RequireDevice()) return;

        var result = Execute(fixture, shader, 16);
        Assert.Equal(expectedReturn, BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan(0)));
        Assert.Equal(expectedMemory, BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan(4)));
    }

    [Theory]
    // ds_add_u64 folds the low dword's carry into the high one.
    [InlineData("DsAddU64", 0x40u, 0x00000001u, 0x00000000u, 0xFFFFFFFFu, 0x00000001u, 0x00000000u, 0x00000002u)]
    [InlineData("DsAddU64-nocarry", 0x40u, 0x00000001u, 0x00000010u, 0x00000002u, 0x00000001u, 0x00000003u, 0x00000011u)]
    // ds_or_b64 is bitwise-independent, so the halves never influence each other.
    [InlineData("DsOrB64", 0x4Au, 0x0000000Fu, 0x000000F0u, 0x00000030u, 0x0000000Cu, 0x0000003Fu, 0x000000FCu)]
    // ds_sub_u64 borrows from the high dword.
    [InlineData("DsSubU64", 0x41u, 0x00000000u, 0x00000001u, 0x00000001u, 0x00000000u, 0xFFFFFFFFu, 0x00000000u)]
    [InlineData("DsXorB64", 0x4Bu, 0x0000FFFFu, 0xFFFF0000u, 0x000000FFu, 0xFF000000u, 0x0000FF00u, 0x00FF0000u)]
    public void PairAtomicsApplyTheOperationAcrossBothDwords(
        string name, uint opcode, uint initialLow, uint initialHigh,
        uint low, uint high, uint expectedLow, uint expectedHigh)
    {
        var fixture = PairFixture(name, opcode, initialLow, initialHigh, low, high);
        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(fixture);
        if (!RequireDevice()) return;

        var result = Execute(fixture, shader, 16);
        Assert.Equal(expectedLow, BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan(0)));
        Assert.Equal(expectedHigh, BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan(4)));
    }

    [Theory]
    // ds_read_i8 sign-extends the byte the low two address bits select.
    [InlineData(0u, 0xFFFFFF81u)]
    [InlineData(1u, 0xFFFFFFFFu)]
    [InlineData(2u, 0x00000012u)]
    [InlineData(3u, 0x00000034u)]
    public void SignedByteReadSelectsAndSignExtendsTheAddressedByte(uint byteOffset, uint expected)
    {
        var fixture = new Gen5ComputeFixture($"ds-read-i8-{byteOffset}",
        [
            0x7E000280,                           // v_mov_b32 v0, 0
            MovLiteral(2), 0x3412FF81,            // v_mov_b32 v2, 0x3412FF81
            DsWord0(0x0D), DsWord1(0, 2, 0, 0),   // ds_write_b32 v0, v2
            0xBF8A0000,                           // s_barrier
            DsWord0(0x39, byteOffset), DsWord1(0, 0, 0, 3), // ds_read_i8 v3, v0 offset:byteOffset
            0xE0700000, 0x80020300,               // buffer_store_dword v3, off, s[8:11], 0
            0xBF810000,                           // s_endpgm
        ], 8, 16);
        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(fixture);
        if (!RequireDevice()) return;

        var result = Execute(fixture, shader, 16);
        Assert.Equal(expected, BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan(0)));
    }

    [Fact]
    public void AtomicsInAGraphicsStageUseThePrivateLdsScratchModel()
    {
        // A pixel shader's LDS is a per-invocation array, so there is no
        // threadgroup atomic: the single logical lane reads and writes it back.
        var words = new uint[]
        {
            0x7E000280,                           // v_mov_b32 v0, 0
            MovLiteral(2), Initial,               // v_mov_b32 v2, Initial
            DsWord0(0x0D), DsWord1(0, 2, 0, 0),   // ds_write_b32 v0, v2
            MovLiteral(4), Operand,               // v_mov_b32 v4, Operand
            DsWord0(0x20), DsWord1(0, 4, 0, 3),   // ds_add_rtn_u32 v3, v0, v4
            0xF800180F, 0x03030303,               // exp mrt0 v3, v3, v3, v3 done vm
            0xBF810000,                           // s_endpgm
        };
        var shader = Gen5ComputeFixtures.CompileStageRequestOrThrow(words, ShaderStage.Pixel);
        Assert.Contains("thread uint sharpemu_lds", shader.Source);
        Assert.DoesNotContain("threadgroup atomic_uint*", shader.Source);
        if (!RequireDevice()) return;
        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out _, out var error), $"{error}\n{shader.Source}");
    }

    [Fact]
    public void GraphicsStageLdsIsNotZeroFilledPerInvocation()
    {
        // Hardware LDS a wave never wrote holds stale data, so the per-invocation
        // stand-in is not zero-filled on every fragment.
        var words = new uint[]
        {
            0x7E000280,                           // v_mov_b32 v0, 0
            MovLiteral(2), Initial,               // v_mov_b32 v2, Initial
            DsWord0(0x0D), DsWord1(0, 2, 0, 0),   // ds_write_b32 v0, v2
            DsWord0(0x36), DsWord1(0, 0, 0, 3),   // ds_read_b32 v3, v0
            0xF800180F, 0x03030303,               // exp mrt0 v3, v3, v3, v3 done vm
            0xBF810000,                           // s_endpgm
        };
        var shader = Gen5ComputeFixtures.CompileStageRequestOrThrow(words, ShaderStage.Pixel);
        Assert.Contains("    thread uint sharpemu_lds[2048];", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("sharpemu_lds[2048] = {}", shader.Source, StringComparison.Ordinal);
        if (!RequireDevice()) return;
        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out _, out var error), $"{error}\n{shader.Source}");
    }

    [Fact]
    public void PixelShaderWithoutDataShareOpsDeclaresNoLdsArray()
    {
        // DS_SWIZZLE is a cross-lane op that never touches LDS memory, so it must
        // not pull the per-invocation LDS stand-in in either.
        var words = new uint[]
        {
            MovLiteral(2), Initial,               // v_mov_b32 v2, Initial
            DsWord0(0x35, 0x1F, 0x80), DsWord1(2, 0, 0, 3), // ds_swizzle_b32 v3, v2 offset:0x801F
            0xF800180F, 0x03030303,               // exp mrt0 v3, v3, v3, v3 done vm
            0xBF810000,                           // s_endpgm
        };
        var shader = Gen5ComputeFixtures.CompileStageRequestOrThrow(words, ShaderStage.Pixel);
        Assert.DoesNotContain("sharpemu_lds", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("sharpemu_scratch", shader.Source, StringComparison.Ordinal);
        if (!RequireDevice()) return;
        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out _, out var error), $"{error}\n{shader.Source}");
    }

    [Fact]
    public void ComputeAtomicsUseThreadgroupAtomicsAndCompileOnTheDevice()
    {
        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(
            OneWordFixture("add-rtn", 0x20, Initial, Operand, Compare));
        Assert.Contains("atomic_fetch_add_explicit((threadgroup atomic_uint*)", shader.Source);
        var compareSwap = Gen5ComputeFixtures.CompileRequestOrThrow(
            OneWordFixture("cmpst-rtn", 0x30, Initial, Operand, Compare));
        Assert.Contains("atomic_compare_exchange_weak_explicit", compareSwap.Source);
        if (!RequireDevice()) return;
        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out _, out var error), $"{error}\n{shader.Source}");
        Assert.True(MetalNative.TryCompileLibrary(compareSwap.Source, out _, out var swapError), swapError);
    }

    [Theory]
    // 64-bit LDS atomics a pair of 32-bit threadgroup atomics cannot make exact run under
    // the LDS qword locks (Gen5MslDataShareQwordTests runs them on the device).
    [InlineData(0x47u, "DsMinU64")]
    [InlineData(0x50u, "DsCmpstB64")]
    [InlineData(0x60u, "DsAddRtnU64")]
    [InlineData(0x6Du, "DsWrxchgRtnB64")]
    [InlineData(0x52u, "DsMinF64")]
    public void Unsplittable64BitSharedLdsAtomicsTakeTheQwordLock(uint opcode, string name)
    {
        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(PairFixture(name, opcode, 0, 0, 0, 0));
        Assert.Contains("sharpemu_lds_try_lock(", shader.Source);
        Assert.Equal(name == "DsMinF64", shader.Source.Contains("sharpemu_f64_lt(", StringComparison.Ordinal));
    }

    [Fact]
    public void GraphicsStage64BitReturningAtomicsUseThePrivateLdsWord()
    {
        // The per-invocation LDS has one writer, so the ulong read-modify-write is exact.
        var words = new uint[]
        {
            0x7E000280,                           // v_mov_b32 v0, 0
            MovLiteral(2), Initial,               // v_mov_b32 v2, Initial
            MovLiteral(3), Operand,               // v_mov_b32 v3, Operand
            DsWord0(0x67), DsWord1(0, 2, 0, 4),   // ds_min_rtn_u64 v[4:5], v0, v[2:3]
            0xF800180F, 0x05040504,               // exp mrt0 v4, v5, v4, v5 done vm
            0xBF810000,                           // s_endpgm
        };
        var shader = Gen5ComputeFixtures.CompileStageRequestOrThrow(words, ShaderStage.Pixel);
        Assert.Contains("thread uint sharpemu_lds", shader.Source);
        if (!RequireDevice()) return;
        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out _, out var error), $"{error}\n{shader.Source}");
    }

    [Theory]
    // The seeded word is 0x3412FF81; DATA0 is 0xAABBCCDD.
    [InlineData("DsWriteB8", 0x1Eu, 1u, 0x3412DD81u)]
    [InlineData("DsWriteB8-top", 0x1Eu, 3u, 0xDD12FF81u)]
    [InlineData("DsWriteB16", 0x1Fu, 2u, 0xCCDDFF81u)]
    [InlineData("DsWriteB16-low", 0x1Fu, 0u, 0x3412CCDDu)]
    // The D16_HI forms store from DATA0's high half.
    [InlineData("DsWriteB8D16Hi", 0xA0u, 0u, 0x3412FFBBu)]
    [InlineData("DsWriteB16D16Hi", 0xA1u, 2u, 0xAABBFF81u)]
    public void SubwordWritesMergeIntoTheAddressedBytes(string name, uint opcode, uint byteOffset, uint expected)
    {
        var fixture = new Gen5ComputeFixture($"ds-{name}",
        [
            0x7E000280,                           // v_mov_b32 v0, 0
            MovLiteral(2), 0x3412FF81,            // v_mov_b32 v2, 0x3412FF81
            DsWord0(0x0D), DsWord1(0, 2, 0, 0),   // ds_write_b32 v0, v2
            0xBF8A0000,                           // s_barrier
            MovLiteral(4), 0xAABBCCDD,            // v_mov_b32 v4, 0xAABBCCDD
            DsWord0(opcode, byteOffset), DsWord1(0, 4, 0, 0), // the write under test
            0xBF8A0000,                           // s_barrier
            DsWord0(0x36), DsWord1(0, 0, 0, 3),   // ds_read_b32 v3, v0
            0xE0700000, 0x80020300,               // buffer_store_dword v3, off, s[8:11], 0
            0xBF810000,                           // s_endpgm
        ], 8, 16);
        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(fixture);
        if (!RequireDevice()) return;

        var result = Execute(fixture, shader, 16);
        Assert.Equal(expected, BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan(0)));
    }

    [Theory]
    // The seeded word is 0x3412FF81; VDST starts as the sentinel 0xA5A5A5A5.
    [InlineData("DsReadU8", 0x3Au, 0u, 0x00000081u)]
    [InlineData("DsReadU8-top", 0x3Au, 3u, 0x00000034u)]
    [InlineData("DsReadU16", 0x3Cu, 0u, 0x0000FF81u)]
    [InlineData("DsReadI16", 0x3Bu, 0u, 0xFFFFFF81u)]
    [InlineData("DsReadI16-positive", 0x3Bu, 2u, 0x00003412u)]
    // D16 reads replace one half of VDST and keep the other.
    [InlineData("DsReadU8D16", 0xA2u, 0u, 0xA5A50081u)]
    [InlineData("DsReadU8D16Hi", 0xA3u, 0u, 0x0081A5A5u)]
    [InlineData("DsReadI8D16", 0xA4u, 0u, 0xA5A5FF81u)]
    [InlineData("DsReadI8D16Hi", 0xA5u, 0u, 0xFF81A5A5u)]
    [InlineData("DsReadU16D16", 0xA6u, 2u, 0xA5A53412u)]
    [InlineData("DsReadU16D16Hi", 0xA7u, 0u, 0xFF81A5A5u)]
    public void SubwordReadsExtendOrMergeTheAddressedBytes(string name, uint opcode, uint byteOffset, uint expected)
    {
        var fixture = new Gen5ComputeFixture($"ds-{name}",
        [
            0x7E000280,                           // v_mov_b32 v0, 0
            MovLiteral(2), 0x3412FF81,            // v_mov_b32 v2, 0x3412FF81
            DsWord0(0x0D), DsWord1(0, 2, 0, 0),   // ds_write_b32 v0, v2
            0xBF8A0000,                           // s_barrier
            MovLiteral(3), Sentinel,              // v_mov_b32 v3, sentinel
            DsWord0(opcode, byteOffset), DsWord1(0, 0, 0, 3), // the read under test
            0xE0700000, 0x80020300,               // buffer_store_dword v3, off, s[8:11], 0
            0xBF810000,                           // s_endpgm
        ], 8, 16);
        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(fixture);
        if (!RequireDevice()) return;

        var result = Execute(fixture, shader, 16);
        Assert.Equal(expected, BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan(0)));
    }

    [Fact]
    public void Wrxchg2ExchangesBothWordsAndReturnsThePair()
    {
        // ds_wrxchg2_rtn_b32 v[3:4], v0, v5, v6 offset0:0 offset1:1 swaps LDS words 0 and 1.
        var fixture = new Gen5ComputeFixture("ds-wrxchg2",
        [
            0x7E000280,                           // v_mov_b32 v0, 0
            MovLiteral(2), Initial,               // v_mov_b32 v2, Initial
            DsWord0(0x0D), DsWord1(0, 2, 0, 0),   // ds_write_b32 v0, v2
            MovLiteral(2), Compare,               // v_mov_b32 v2, Compare
            DsWord0(0x0D, 4), DsWord1(0, 2, 0, 0), // ds_write_b32 v0, v2 offset:4
            0xBF8A0000,                           // s_barrier
            MovLiteral(5), 0x11111111,            // v_mov_b32 v5, 0x11111111
            MovLiteral(6), 0x22222222,            // v_mov_b32 v6, 0x22222222
            DsWord0(0x2E, 0, 1), DsWord1(0, 5, 6, 3), // ds_wrxchg2_rtn_b32 v[3:4], v0, v5, v6 offset1:1
            0xBF8A0000,                           // s_barrier
            DsWord0(0x76), DsWord1(0, 0, 0, 7),   // ds_read_b64 v[7:8], v0
            0xE0700000, 0x80020300,               // buffer_store_dword v3, off, s[8:11], 0
            0xE0700004, 0x80020400,               // buffer_store_dword v4, off, s[8:11], 0 offset:4
            0xE0700008, 0x80020700,               // buffer_store_dword v7, off, s[8:11], 0 offset:8
            0xE070000C, 0x80020800,               // buffer_store_dword v8, off, s[8:11], 0 offset:12
            0xBF810000,                           // s_endpgm
        ], 8, 16);
        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(fixture);
        if (!RequireDevice()) return;

        var result = Execute(fixture, shader, 16);
        Assert.Equal(Initial, BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan(0)));
        Assert.Equal(Compare, BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan(4)));
        Assert.Equal(0x11111111u, BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan(8)));
        Assert.Equal(0x22222222u, BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan(12)));
    }

    [Theory]
    // Every lane sends its index to lane + 1, so each lane receives its predecessor's.
    [InlineData("next", false)]
    // Every lane sends to lane 0: the highest sender wins and the rest receive zero.
    [InlineData("all-to-zero", true)]
    public void PermuteForwardsEachLanesDataToTheLaneItAddresses(string name, bool allToZero)
    {
        var fixture = new Gen5ComputeFixture($"ds-permute-{name}",
        [
            0xD7650000, 0x000100C1,               // v_mbcnt_lo_u32_b32 v0, -1, 0        lane 0..31
            allToZero ? 0x7E040280u : 0x34040082u, // v_mov_b32 v2, 0 | v_lshlrev_b32 v2, 2, v0
            DsWord0(0xB2, allToZero ? 0u : 4u), DsWord1(2, 0, 0, 3), // ds_permute_b32 v3, v2, v0 offset:0|4
            0xBF8C0000,                           // s_waitcnt 0
            0x34080082,                           // v_lshlrev_b32 v4, 2, v0
            0xE0701000, 0x80020304,               // buffer_store_dword v3, v4, s[8:11], 0 offen
            0xBF810000,                           // s_endpgm
        ], 8, 128);
        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(fixture);
        Assert.DoesNotContain("sharpemu_lds", shader.Source);
        if (!RequireDevice()) return;

        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var compileError),
            $"{compileError}\n{shader.Source}");
        Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, new byte[128],
            new byte[PushData.ByteSize], shader.ArgumentLayout!, 32, out var result, out var runError), runError);
        for (var lane = 0u; lane < 32; lane++)
        {
            var expected = allToZero ? (lane == 0 ? 31u : 0u) : (lane + 31) & 31;
            Assert.Equal(expected, BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan((int)lane * 4)));
        }
    }

    // DS: [31:26] = 0b110110, offset0 [7:0], offset1 [15:8], GDS [17], op [25:18].
    private static uint DsWord0(uint opcode, uint offset0 = 0, uint offset1 = 0) =>
        0xD8000000u | opcode << 18 | offset1 << 8 | offset0;

    private static uint DsWord1(uint address, uint data0, uint data1, uint destination) =>
        destination << 24 | data1 << 16 | data0 << 8 | address;

    // VOP1 v_mov_b32 with a 32-bit literal following.
    private static uint MovLiteral(uint register) => 0x7E0002FFu | register << 17;

    private static Gen5ComputeFixture OneWordFixture(
        string name, uint opcode, uint initial, uint data0, uint data1) =>
        new($"ds-{name}",
        [
            0x7E000280,                           // v_mov_b32 v0, 0            (LDS byte address)
            MovLiteral(2), initial,               // v_mov_b32 v2, initial
            DsWord0(0x0D), DsWord1(0, 2, 0, 0),   // ds_write_b32 v0, v2
            0xBF8A0000,                           // s_barrier
            MovLiteral(3), Sentinel,              // v_mov_b32 v3, sentinel     (return register)
            MovLiteral(4), data0,                 // v_mov_b32 v4, data0
            MovLiteral(6), data1,                 // v_mov_b32 v6, data1
            DsWord0(opcode), DsWord1(0, 4, 6, 3), // the atomic under test
            0xBF8A0000,                           // s_barrier
            DsWord0(0x36), DsWord1(0, 0, 0, 5),   // ds_read_b32 v5, v0
            0xE0700000, 0x80020300,               // buffer_store_dword v3, off, s[8:11], 0
            0xE0700004, 0x80020500,               // buffer_store_dword v5, off, s[8:11], 0 offset:4
            0xBF810000,                           // s_endpgm
        ], 8, 16);

    // DS_READ_I8 and DS_READ_U8 pick the byte the offset address names out of its word;
    // the I8 form sign-extends it and the U8 form zero-extends it.
    [Theory]
    [InlineData(0u, 0xFFFFFF81u, 0x00000081u)]
    [InlineData(1u, 0x0000007Fu, 0x0000007Fu)]
    [InlineData(2u, 0xFFFFFFC3u, 0x000000C3u)]
    [InlineData(3u, 0x00000012u, 0x00000012u)]
    public void ByteReadsSelectTheAddressedByteAndExtendIt(uint offset, uint expectedSigned, uint expectedUnsigned)
    {
        var fixture = new Gen5ComputeFixture($"ds-read-byte-{offset}",
        [
            0x7E000280,                                  // v_mov_b32 v0, 0
            MovLiteral(2), 0x12C37F81u,                  // v_mov_b32 v2, bytes 81 7F C3 12
            DsWord0(0x0D), DsWord1(0, 2, 0, 0),          // ds_write_b32 v0, v2
            0xBF8A0000,                                  // s_barrier
            DsWord0(0x39, offset), DsWord1(0, 0, 0, 3),  // ds_read_i8 v3, v0 offset:n
            DsWord0(0x3A, offset), DsWord1(0, 0, 0, 5),  // ds_read_u8 v5, v0 offset:n
            0xE0700000, 0x80020300,                      // buffer_store_dword v3, off, s[8:11], 0
            0xE0700004, 0x80020500,                      // buffer_store_dword v5, off, s[8:11], 0 offset:4
            0xBF810000,                                  // s_endpgm
        ], 8, 16);
        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(fixture);
        if (!RequireDevice()) return;

        var result = Execute(fixture, shader, 16);
        Assert.Equal(expectedSigned, BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan(0)));
        Assert.Equal(expectedUnsigned, BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan(4)));
    }

    private static Gen5ComputeFixture PairFixture(
        string name, uint opcode, uint initialLow, uint initialHigh, uint low, uint high) =>
        new($"ds-{name}",
        [
            0x7E000280,                              // v_mov_b32 v0, 0
            MovLiteral(2), initialLow,               // v_mov_b32 v2, initialLow
            MovLiteral(3), initialHigh,              // v_mov_b32 v3, initialHigh
            DsWord0(0x4D), DsWord1(0, 2, 0, 0),      // ds_write_b64 v0, v[2:3]
            0xBF8A0000,                              // s_barrier
            MovLiteral(4), low,                      // v_mov_b32 v4, low
            MovLiteral(5), high,                     // v_mov_b32 v5, high
            DsWord0(opcode), DsWord1(0, 4, 0, 0),    // the 64-bit atomic under test
            0xBF8A0000,                              // s_barrier
            DsWord0(0x76), DsWord1(0, 0, 0, 6),      // ds_read_b64 v[6:7], v0
            0xE0700000, 0x80020600,                  // buffer_store_dword v6, off, s[8:11], 0
            0xE0700004, 0x80020700,                  // buffer_store_dword v7, off, s[8:11], 0 offset:4
            0xBF810000,                              // s_endpgm
        ], 8, 16);

    private static byte[] Execute(Gen5ComputeFixture fixture, Gen5MslShader shader, int bytes)
    {
        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var compileError),
            $"[{fixture.Name}] {compileError}\n{shader.Source}");
        Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, new byte[bytes],
            new byte[PushData.ByteSize], shader.ArgumentLayout!, 1, out var result, out var runError),
            $"[{fixture.Name}] {runError}");
        return result;
    }

    private bool RequireDevice()
    {
        if (!MetalNative.IsAvailable)
        {
            Assert.False(
                OperatingSystem.IsMacOS() && Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") == "1",
                "The required gate cannot run the LDS atomic execution test without a Metal device.");
            output.WriteLine("No Metal device on this host; the LDS atomic execution check is skipped.");
            return false;
        }

        return true;
    }
}
