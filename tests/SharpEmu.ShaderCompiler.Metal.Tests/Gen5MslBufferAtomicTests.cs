// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

public sealed class Gen5MslBufferAtomicTests(ITestOutputHelper output)
{
    private const uint Sentinel = 0xBADCAFE;
    private const uint Before = 0xFEDCBA98;
    private const uint Replacement = 0x11223344;

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void SwapDecoderAndDenseBufferCodegenPreserveItsOperandContract(bool glc)
    {
        var fixture = Fixture(glc, 1, 8, 4, indexed: true);
        var program = Gen5ComputeFixtures.DecodeOrThrow(fixture);
        var instruction = Assert.Single(program.Instructions.Where(static instruction => instruction.Opcode == "BufferAtomicSwap"));
        var control = Assert.IsType<Gen5BufferMemoryControl>(instruction.Control);
        Assert.Equal(1u, control.DwordCount);
        Assert.Equal(4u, control.VectorData);
        Assert.Equal(0u, control.VectorAddress);
        Assert.Equal(8u, control.ScalarResource);
        Assert.Equal(4, control.OffsetBytes);
        Assert.True(control.IndexEnabled);
        Assert.True(control.OffsetEnabled);
        Assert.Equal(glc, control.Glc);
        var shader = Compile(fixture, 16);
        Assert.False(shader.UsesPairLocks);
        Assert.False(shader.WritesStepReport);
        Assert.Contains("atomic_exchange_explicit", shader.Source);
        Assert.Contains("v4, memory_order_relaxed", shader.Source);
        Assert.Contains("(v0 * 0x10u)", shader.Source);
        Assert.Contains("4u <= sharpemu_resources.buffer_bytes[0]", shader.Source);
        Assert.Contains("sharpemu_resources.buffer_bytes[0] - 4u", shader.Source);
        Assert.DoesNotContain("+ 4u <= sharpemu_resources.buffer_bytes", shader.Source);
    }

    [Theory]
    // An unnamed MUBUF opcode still fails loudly rather than emitting nothing (0x4D-0x4F
    // are unassigned on gfx10.3).
    [InlineData(0x4Du, "MubufRaw4D")]
    public void UndecodedBufferOpcodesRemainExplicitlyUnsupported(uint opcode, string name)
    {
        var fixture = Fixture(true, 0, 0, 0, indexed: false, opcode);
        var request = Request(fixture, 0);
        Assert.False(Gen5MslTranslator.TryCompileProgram(request, out _, out var error));
        Assert.Contains(name, error);
        Assert.Contains("unsupported memory opcode", error);
    }

    [Theory]
    [InlineData(0x30u, Replacement, Before)]            // swap
    [InlineData(0x32u, 0x0FFEEDDCu, Before)]            // add
    [InlineData(0x33u, 0xEDBA8754u, Before)]            // sub
    [InlineData(0x35u, Before, Before)]                 // signed min keeps the negative word
    [InlineData(0x36u, Replacement, Before)]            // unsigned min takes the operand
    [InlineData(0x37u, Replacement, Before)]            // signed max takes the operand
    [InlineData(0x38u, Before, Before)]                 // unsigned max keeps the larger word
    [InlineData(0x39u, 0x10003200u, Before)]            // and
    [InlineData(0x3Au, 0xFFFEBBDCu, Before)]            // or
    [InlineData(0x3Bu, 0xEFFE89DCu, Before)]            // xor
    [InlineData(0x3Cu, 0u, Before)]                     // inc: Before >= DATA wraps to 0
    [InlineData(0x3Du, Replacement, Before)]            // dec: Before > DATA wraps to DATA
    // As floats Before is -1.47e38 and Replacement 1.28e-28.
    [InlineData(0x3Fu, Before, Before)]                 // fmin keeps the negative word
    [InlineData(0x40u, Replacement, Before)]            // fmax takes the operand
    public void EveryBufferAtomicAppliesItsOperationAndReturnsThePreOperationWord(
        uint opcode, uint expectedMemory, uint expectedReturn)
    {
        var fixture = Fixture(true, 0, 32, 0, indexed: false, opcode);
        var shader = Compile(fixture, 0);
        if (!RequireDevice()) return;

        var expected = SentinelBuffer(32);
        BinaryPrimitives.WriteUInt32LittleEndian(expected.AsSpan(32), expectedMemory);
        BinaryPrimitives.WriteUInt32LittleEndian(expected.AsSpan(48), expectedReturn);
        Assert.Equal(expected, Execute(shader, SentinelBuffer(32)));
    }

    [Theory]
    // CMPSWAP reads the new value from VDATA and the comparator from the next
    // register, so the swap lands only when the comparator matches memory.
    [InlineData(Before, Replacement)]
    [InlineData(0u, Before)]
    public void CompareSwapOnlyReplacesTheWordItsComparatorMatches(uint comparator, uint expectedMemory)
    {
        var fixture = CompareSwapFixture(comparator);
        var shader = Compile(fixture, 0);
        if (!RequireDevice()) return;

        var expected = SentinelBuffer(32);
        BinaryPrimitives.WriteUInt32LittleEndian(expected.AsSpan(32), expectedMemory);
        BinaryPrimitives.WriteUInt32LittleEndian(expected.AsSpan(48), Before);
        Assert.Equal(expected, Execute(shader, SentinelBuffer(32)));
    }

    [Theory]
    // The X2 forms run over a dword pair and return both halves into VDATA.
    [InlineData(0x50u, Replacement, 0u)]
    [InlineData(0x5Au, 0xFFFEBBDCu, Sentinel)]
    public void PairWideAtomicsCoverBothDwordsOfTheAccess(uint opcode, uint expectedLow, uint expectedHigh)
    {
        var fixture = PairFixture(opcode);
        var shader = Compile(fixture, 0);
        if (!RequireDevice()) return;

        var expected = SentinelBuffer(32);
        BinaryPrimitives.WriteUInt32LittleEndian(expected.AsSpan(32), expectedLow);
        BinaryPrimitives.WriteUInt32LittleEndian(expected.AsSpan(36), expectedHigh);
        // The returned pair replaces v[4:5], which the fixture stores at 48/52.
        BinaryPrimitives.WriteUInt32LittleEndian(expected.AsSpan(48), Before);
        BinaryPrimitives.WriteUInt32LittleEndian(expected.AsSpan(52), Sentinel);
        Assert.Equal(expected, Execute(shader, SentinelBuffer(32)));
    }

    [Fact]
    public void DecodedUnsignedMaxUsesTheUnsignedNativeAtomic()
    {
        var fixture = Fixture(true, 1, 8, 4, indexed: true, opcode: 0x38);
        var instruction = Assert.Single(Gen5ComputeFixtures.DecodeOrThrow(fixture).Instructions
            .Where(static candidate => candidate.Opcode == "BufferAtomicUmax"));
        Assert.IsType<Gen5BufferMemoryControl>(instruction.Control);
        Assert.Contains("atomic_fetch_max_explicit((device atomic_uint*)", Compile(fixture, 16).Source);
        if (!RequireDevice()) return;
        var initial = SentinelBuffer(32);
        var expected = initial.ToArray();
        // Before has its sign bit set: signed max would incorrectly select
        // Replacement, while unsigned max must leave Before in memory.
        BinaryPrimitives.WriteUInt32LittleEndian(expected.AsSpan(48), Before);
        Assert.Equal(expected, Execute(Compile(fixture, 16), initial));
    }

    [Fact]
    public void SwapCapturesAliasedAddressAndReplacementBeforeWritingReturnedVdata()
    {
        var fixture = new Gen5ComputeFixture("atomic-aliased-vdata",
            [0x7E0802FF, 32, 0xE0C05000, 0x80020404, 0xE0700030, 0x80020400, 0xBF810000], 8, 64);
        var instruction = Assert.Single(Gen5ComputeFixtures.DecodeOrThrow(fixture).Instructions
            .Where(static candidate => candidate.Opcode == "BufferAtomicSwap"));
        var control = Assert.IsType<Gen5BufferMemoryControl>(instruction.Control);
        Assert.Equal(4u, control.VectorData);
        Assert.Equal(4u, control.VectorAddress);
        Assert.True(control.Glc);
        var shader = Compile(fixture, 0);
        if (!RequireDevice()) return;
        var initial = SentinelBuffer(32);
        var expected = initial.ToArray();
        BinaryPrimitives.WriteUInt32LittleEndian(expected.AsSpan(32), 32u);
        BinaryPrimitives.WriteUInt32LittleEndian(expected.AsSpan(48), Before);
        Assert.Equal(expected, Execute(shader, initial));
    }

    [Theory]
    [InlineData(true, 1u, 8u, 4u, true, true, 32, true)]
    [InlineData(false, 1u, 8u, 4u, true, true, 32, true)]
    [InlineData(true, 0u, 60u, 0u, false, true, 60, true)]
    [InlineData(true, 0u, 64u, 0u, false, true, 0, false)]
    [InlineData(true, 0u, 0xFFFFFFFCu, 0u, false, true, 0, false)]
    [InlineData(true, 0u, 3u, 0u, false, true, 0, false)]
    [InlineData(true, 1u, 8u, 4u, true, false, 32, false)]
    public void SwapExecutesWithIndependentAddressReturnAndSentinelOracle(bool glc, uint index,
        uint offset, uint immediate, bool indexed, bool active, int target, bool writes)
    {
        // Do not touch a device outside the shared exclusive test/game window.
        if (!RequireDevice()) return;
        var fixture = Fixture(glc, index, offset, immediate, indexed, active: active);
        var shader = Compile(fixture, indexed ? 16u : 0u);
        var initial = SentinelBuffer(target);
        var expected = initial.ToArray();
        if (writes) BinaryPrimitives.WriteUInt32LittleEndian(expected.AsSpan(target), Replacement);
        // GLC returns the pre-operation word into VDATA; disabled EXEC leaves it
        // untouched, while an out-of-range active returning access reads zero.
        var returned = !active || !glc ? Replacement : writes ? Before : 0u;
        BinaryPrimitives.WriteUInt32LittleEndian(expected.AsSpan(48), returned);
        Assert.Equal(expected, Execute(shader, initial));
    }

    [Theory]
    // ATOMIC_INC stores (old >= DATA) ? 0 : old + 1 and ATOMIC_DEC stores
    // (old == 0 || old > DATA) ? DATA : old - 1; the pre-operation word is returned either way.
    [InlineData(0x3Cu, 3u, 3u, 0u)]                     // inc at the limit wraps to 0
    [InlineData(0x3Cu, 2u, 3u, 3u)]                     // inc below the limit steps
    [InlineData(0x3Du, 0u, 7u, 7u)]                     // dec from 0 takes the limit
    [InlineData(0x3Du, 9u, 7u, 7u)]                     // dec above the limit takes the limit
    [InlineData(0x3Du, 5u, 7u, 4u)]                     // dec within the limit steps
    public void IncAndDecWrapAgainstTheDataOperand(uint opcode, uint memory, uint data, uint expectedMemory)
    {
        var fixture = Fixture(true, 0, 32, 0, indexed: false, opcode);
        // The fixture loads Replacement into v4; patch the literal to the wrap limit under test.
        var words = fixture.Words.ToArray();
        words[5] = data;
        var shader = Compile(fixture with { Words = words }, 0);
        if (!RequireDevice()) return;

        var initial = SentinelBuffer(32);
        BinaryPrimitives.WriteUInt32LittleEndian(initial.AsSpan(32), memory);
        var expected = initial.ToArray();
        BinaryPrimitives.WriteUInt32LittleEndian(expected.AsSpan(32), expectedMemory);
        BinaryPrimitives.WriteUInt32LittleEndian(expected.AsSpan(48), memory);
        Assert.Equal(expected, Execute(shader, initial));
    }

    private bool RequireDevice()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Atomic readback requires the exclusive Metal device gate.");
            return false;
        }
        Assert.True(MetalNative.IsAvailable);
        return true;
    }

    private static byte[] SentinelBuffer(int target)
    {
        var initial = new byte[64];
        for (var byteIndex = 0; byteIndex < initial.Length; byteIndex += 4)
            BinaryPrimitives.WriteUInt32LittleEndian(initial.AsSpan(byteIndex), Sentinel);
        BinaryPrimitives.WriteUInt32LittleEndian(initial.AsSpan(target), Before);
        return initial;
    }

    private static byte[] Execute(Gen5MslShader shader, byte[] initial)
    {
        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var compileError), compileError);
        Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, initial,
            new byte[PushData.ByteSize], shader.ArgumentLayout!, 1, out var result, out var runError), runError);
        return result;
    }

    private static Gen5MslShader Compile(Gen5ComputeFixture fixture, uint stride)
    {
        Assert.True(Gen5MslTranslator.TryCompileProgram(Request(fixture, stride), out var shader, out var error), error);
        return shader!;
    }

    private static ShaderCompileRequest Request(Gen5ComputeFixture fixture, uint stride)
    {
        var userData = new uint[16];
        userData[8] = 0x200000;
        userData[9] = stride << 16;
        userData[10] = 64;
        userData[11] = DescriptorConstants.IdentityDestinationSelect;
        return Gen5ComputeFixtures.RequestOrThrow(Gen5ComputeFixtures.DecodeOrThrow(fixture), ShaderStage.Compute,
            runtimeInputs: new ResourceRuntimeInputs { UserData = userData });
    }

    // One MUBUF atomic at byte 32 of the dense buffer with GLC set, so VDATA
    // comes back holding the pre-operation word, stored at byte 48.
    private static Gen5ComputeFixture CompareSwapFixture(uint comparator) =>
        new("buffer-atomic-cmpswap",
        [
            0x7E0202FF, 32,                                  // v_mov_b32 v1, 32
            0x7E0802FF, Replacement,                         // v_mov_b32 v4, replacement
            0x7E0A02FF, comparator,                          // v_mov_b32 v5, comparator
            0xE0000000u | 0x31u << 18 | 1u << 14 | 1u << 12, // buffer_atomic_cmpswap v[4:5], v1, s[8:11] offen glc
            0x80020401,
            0xE0700030, 0x80020400,                          // buffer_store_dword v4, off, s[8:11], 0 offset:48
            0xBF810000,                                      // s_endpgm
        ], 8, 64);

    private static Gen5ComputeFixture PairFixture(uint opcode) =>
        new("buffer-atomic-x2",
        [
            0x7E0202FF, 32,                                   // v_mov_b32 v1, 32
            0x7E0802FF, Replacement,                          // v_mov_b32 v4, replacement
            0x7E0A0280,                                       // v_mov_b32 v5, 0
            0xE0000000u | opcode << 18 | 1u << 14 | 1u << 12, // the X2 atomic under test
            0x80020401,
            0xE0700030, 0x80020400,                           // buffer_store_dword v4 offset:48
            0xE0700034, 0x80020500,                           // buffer_store_dword v5 offset:52
            0xBF810000,                                       // s_endpgm
        ], 8, 64);

    private static Gen5ComputeFixture Fixture(bool glc, uint index, uint offset,
        uint immediate, bool indexed, uint opcode = 0x30, bool active = true)
    {
        var words = new List<uint>
        {
            0x7E0002FF, index,       // v_mov_b32 v0, index
            0x7E0202FF, offset,      // v_mov_b32 v1, offset
            0x7E0802FF, Replacement, // v_mov_b32 v4, replacement
        };
        if (!active) words.Add(0xBEFE0380); // s_mov_b32 exec_lo, 0
        // RDNA2 MUBUF: GLC bit14; IDXEN bit13; OFFEN bit12. For indexed
        // fixture addr=imm4 + scalar4 + vectorOffset8 + index1*stride16=32.
        words.Add(0xE0000000u | opcode << 18 | (glc ? 1u << 14 : 0) |
            (indexed ? 1u << 13 : 0) | 1u << 12 | immediate);
        words.Add(((indexed ? 0x84u : 0x80u) << 24) | 2u << 16 | 4u << 8 | (indexed ? 0u : 1u));
        if (!active) words.Add(0xBEFE03C1); // restore EXEC for result store
        words.AddRange([0xE0700030, 0x80020400, 0xBF810000]); // buffer_store_dword v4 at48; endpgm
        return new("buffer-atomic", words.ToArray(), 8, 64);
    }
}
