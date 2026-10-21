// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>
/// Structural checks over the emitted MSL — these run on every platform because
/// translation is pure text generation; only the runtime tests need a Metal device.
/// </summary>
public sealed class MslTranslationTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ShaderCallsReportTheBackendLimitation(bool linked)
    {
        var program = new Gen5ShaderProgram(0x1000, [
            new(0, Gen5ShaderEncoding.Sop1, "SSwappcB64", [0xBE8E2100],
                [Gen5Operand.Scalar(0)], [Gen5Operand.Scalar(14)], null),
            new(4, Gen5ShaderEncoding.Sopp, "SEndpgm", [0xBF810000], [], [], null)]);
        if (linked)
        {
            var function = new Gen5ShaderProgram(0x2000, [
                new(0, Gen5ShaderEncoding.Sop1, "SSetpcB64", [0xBE80200E],
                    [Gen5Operand.Scalar(14)], [], null)]);
            program = Gen5ShaderCallLinker.Link(program, [new ShaderCallSite(0, 0, 2, 14, 0x1004,
                [new ShaderCallTarget(0x2000, 0, function)])]);
        }
        var request = Gen5ComputeFixtures.RequestOrThrow(program, ShaderStage.Compute);
        // An unlinked S_SWAPPC_B64 is the caller compiled alone (the call is skipped): the
        // pipeline cache only links calls for a backend that supports them, which Metal does not.
        if (!linked)
        {
            Assert.True(Gen5MslTranslator.TryCompileProgram(request, out _, out var unlinkedError), unlinkedError);
            return;
        }
        Assert.False(Gen5MslTranslator.TryCompileProgram(request, out _, out var error));
        Assert.Equal("Metal does not support linked shader calls.", error);
    }

    [Fact]
    public void ComputeFixturesResolveDescriptorFormatsBeforeCompilation()
    {
        foreach (var fixture in new[] { Gen5ComputeFixtures.TypedLoad, Gen5ComputeFixtures.TypedStore })
        {
            var request = Gen5ComputeFixtures.CreateComputeRequest(fixture);
            Assert.NotEmpty(request.Resources.Info.Buffers);
            Assert.All(request.Resources.Info.Buffers, buffer =>
            {
                Assert.Equal(56u, buffer.DescriptorFormat);
                Assert.Equal(DescriptorConstants.IdentityDestinationSelect, buffer.DescriptorSwizzle);
            });
            _ = Gen5ComputeFixtures.CompileRequestOrThrow(fixture);
        }
    }

    [Fact]
    public void SadU32UsesUnsignedAbsoluteDifferenceAndAccumulator()
    {
        var sad = new Gen5ShaderInstruction(
            0,
            Gen5ShaderEncoding.Vop3,
            "VSadU32",
            [0, 0],
            [Gen5Operand.Vector(0), Gen5Operand.Vector(1), Gen5Operand.Vector(2)],
            [Gen5Operand.Vector(3)],
            null);
        var end = new Gen5ShaderInstruction(
            8,
            Gen5ShaderEncoding.Sopp,
            "SEndpgm",
            [0xBF810000],
            [],
            [],
            null);
        var request = Gen5ComputeFixtures.RequestOrThrow(
            new Gen5ShaderProgram(0, [sad, end]), ShaderStage.Compute, localSizeX: 1);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        Assert.Contains("((max(v0, v1) - min(v0, v1)) + (v2))", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void FlbitI32B64CompilesWithWordSizedScansAndZeroFallback()
    {
        var scan = new Gen5ShaderInstruction(
            0, Gen5ShaderEncoding.Sop1, "SFlbitI32B64", [0xBE821600],
            [Gen5Operand.Scalar(0)], [Gen5Operand.Scalar(2)], null);
        var end = new Gen5ShaderInstruction(
            4, Gen5ShaderEncoding.Sopp, "SEndpgm", [0xBF810000], [], [], null);
        var request = Gen5ComputeFixtures.RequestOrThrow(
            new Gen5ShaderProgram(0, [scan, end]), ShaderStage.Compute, localSizeX: 1);

        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        Assert.Contains("clz(", shader.Source, StringComparison.Ordinal);
        Assert.Contains(">> 32", shader.Source, StringComparison.Ordinal);
        Assert.Contains("32u + clz(", shader.Source, StringComparison.Ordinal);
        Assert.Contains("0xFFFFFFFFu", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void F16CompareUsesHalfOperands()
    {
        var compare = new Gen5ShaderInstruction(
            0,
            Gen5ShaderEncoding.Vopc,
            "VCmpLtF16",
            [],
            [Gen5Operand.Vector(0), Gen5Operand.Vector(1)],
            [],
            null);
        var end = new Gen5ShaderInstruction(
            4,
            Gen5ShaderEncoding.Sopp,
            "SEndpgm",
            [0xBF810000],
            [],
            [],
            null);
        var request = Gen5ComputeFixtures.RequestOrThrow(
            new Gen5ShaderProgram(0, [compare, end]), ShaderStage.Compute, localSizeX: 1);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        Assert.Contains("as_type<half>", shader.Source, StringComparison.Ordinal);
        Assert.Contains(" < ", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void EveryFixtureTranslates()
    {
        foreach (var fixture in Gen5ComputeFixtures.All)
        {
            var shader = Gen5ComputeFixtures.CompileRequestOrThrow(fixture);
            Assert.Equal(Gen5MslStage.Compute, shader.Stage);
            Assert.Equal("gen5_cs", shader.EntryPoint);
            Assert.Contains("kernel void gen5_cs(", shader.Source, StringComparison.Ordinal);

            // Structured control flow, so no PC dispatcher survives in the output.
            Assert.DoesNotContain("switch (pc)", shader.Source, StringComparison.Ordinal);
        }
    }

    /// <summary>The per-lane EXEC guard is dropped only where the program proves every
    /// lane is set. A program that narrows EXEC keeps its guards from that point on, and
    /// turning the elision off brings them all back — which is what the switch is for.</summary>
    [Fact]
    public void ExecGuardsSurviveWhereExecIsNarrowed()
    {
        // The fixture moves 0 into EXEC_LO, stores, then restores it: everything before
        // the first EXEC write is unguarded, everything after it is guarded.
        var request = Gen5ComputeFixtures.CreateComputeRequest(Gen5ComputeFixtures.ExecStore);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var elided, out var error), error);
        var body = elided.Source;
        var firstExecWrite = body.IndexOf("exec = ((", StringComparison.Ordinal);
        Assert.True(firstExecWrite > 0, "the fixture is supposed to write EXEC");
        Assert.DoesNotContain("if (exec)", body[..firstExecWrite], StringComparison.Ordinal);
        Assert.Contains("if (exec)", body[firstExecWrite..], StringComparison.Ordinal);

        Assert.True(
            Gen5MslTranslator.TryCompileProgram(
                request,
                out var guarded,
                out error,
                new Gen5MslCodegenOptions(ExecGuardElision: false)),
            error);
        Assert.Contains("if (exec) { v0 = ", guarded.Source, StringComparison.Ordinal);
        Assert.True(
            CountOf(guarded.Source, "if (exec)") > CountOf(body, "if (exec)"),
            "the switch has to put the guards back");
    }

    private static int CountOf(string source, string value)
    {
        var count = 0;
        for (var index = source.IndexOf(value, StringComparison.Ordinal); index >= 0;
            index = source.IndexOf(value, index + 1, StringComparison.Ordinal))
        {
            count++;
        }

        return count;
    }

    /// <summary>The register file is uint, so an instruction that reads a float and
    /// writes back raw bits used to emit as_type&lt;uint&gt;(as_type&lt;float&gt;(v0)) — a
    /// no-op, but one the Metal compiler had to be handed first. V_CVT_PKRTZ_F16_F32
    /// truncates its operands' mantissas before converting, which is where it showed up.
    /// The packed halves still have to come out right: 1.5 is 0x3E00 and 2.0 is 0x4000.</summary>
    [Fact]
    public void PackedHalfConversionDoesNotRoundTripItsBitcasts()
    {
        var fixture = new Gen5ComputeFixture(
            "pkrtz",
            [
                0x7E0002FF, 0x3FC00000, // v_mov_b32 v0, 1.5f
                0x7E0202FF, 0x40000000, // v_mov_b32 v1, 2.0f
                0x5E0A0300,             // v_cvt_pkrtz_f16_f32 v5, v0, v1
                0xE0700000, 0x80020500, // buffer_store_dword v5, off, s[8:11], 0
                0xBF810000,             // s_endpgm
            ],
            StoreScalarResourceBase: 8,
            StoreBackingBytes: 16);

        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(fixture);
        Assert.Contains("& 0xFFFFE000u", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("as_type<uint>(as_type<float>(", shader.Source, StringComparison.Ordinal);

        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            return;
        }

        Assert.True(MetalNative.IsAvailable, "The required execution gate needs a Metal device.");
        Assert.True(
            MetalNative.TryCompileLibrary(shader.Source, out var library, out var metalError),
            metalError);
        Assert.True(
            MetalNative.TryExecuteWithArgumentBuffer(
                library,
                shader.EntryPoint,
                new byte[fixture.StoreBackingBytes],
                new byte[Resources.PushData.ByteSize],
                shader.ArgumentLayout!,
                threadsPerThreadgroup: 1,
                out var result,
                out var runError),
            runError);
        Assert.Equal(
            0x40003E00u,
            System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(result));
    }

    /// <summary>Threadgroup memory is a hard occupancy limit on Apple GPUs: a
    /// threadgroup that asks for all 32 KB is the only one that fits on a core. The
    /// declaration therefore follows what the dispatch allocated, rounded up to a power
    /// of two because LDS addresses are masked into bounds rather than clamped — and the
    /// mask has to follow the size, or an address would escape the array.</summary>
    [Theory]
    [InlineData(0u, 8192u)]      // the dispatch did not say: assume the whole 32 KB
    [InlineData(1u, 1u)]
    [InlineData(100u, 128u)]
    [InlineData(256u, 256u)]
    [InlineData(1024u, 1024u)]
    [InlineData(8192u, 8192u)]
    [InlineData(99999u, 8192u)]  // more than the hardware has: clamp to the ceiling
    public void ComputeThreadgroupMemoryFollowsTheDispatchAllocation(uint declared, uint expected)
    {
        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(
            Gen5ComputeFixtures.Lds,
            localDataShareDwords: declared);

        Assert.Contains(
            $"threadgroup uint sharpemu_lds[{expected}];",
            shader.Source,
            StringComparison.Ordinal);
        Assert.Contains($"& {expected - 1}u", shader.Source, StringComparison.Ordinal);
    }

    /// <summary>The wave64 ballot bridge only has to share the LDS allocation when the
    /// dispatch asked for all of it; with room to spare it gets its own eight dwords and
    /// cannot collide with guest data.</summary>
    [Theory]
    [InlineData(1024u, false)]
    [InlineData(8192u, true)]
    public void TheWave64BridgeOnlySharesLdsWhenThereIsNoRoom(uint declared, bool aliased)
    {
        // v_readfirstlane is what makes a wave64 program need the bridge at all.
        var fixture = new Gen5ComputeFixture(
            "wave64-bridge-lds",
            [
                0x7E000280,             // v_mov_b32 v0, 0
                0x7E0402FF, 0x00001234, // v_mov_b32 v2, 0x1234
                0xD8340000, 0x00000200, // ds_write_b32 v0, v2
                0x7E000500,             // v_readfirstlane_b32 s0, v0
                0xBF810000,             // s_endpgm
            ],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);
        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(
            fixture,
            waveLaneCount: 64,
            localSizeX: 64,
            localDataShareDwords: declared);

        Assert.Equal(
            aliased,
            shader.Source.Contains(
                "threadgroup uint* sharpemu_wave_scratch = &sharpemu_lds[",
                StringComparison.Ordinal));
        Assert.Equal(
            !aliased,
            shader.Source.Contains(
                "threadgroup uint sharpemu_wave_scratch[8];",
                StringComparison.Ordinal));
    }

    /// <summary>The user-data words are a read-only table the host fills before the
    /// encode and nothing on the GPU writes, and every read of one is at an index the
    /// program knows — so the pointer belongs in the constant address space, which is
    /// Apple's uniform datapath. Read from "device" instead, a value that is the same in
    /// every lane costs a per-lane memory access: the heaviest Octopath vertex shader
    /// reads 32 of them in its preamble, once per wave on hardware and once per vertex
    /// here, 12.5 million times a frame.
    ///
    /// Pushing the start cursor to the end of push data is what makes a program take the
    /// shader-data descriptor instead, which is the path a graphics stage takes.</summary>
    [Fact]
    public void UserDataReadsGoThroughTheConstantAddressSpace()
    {
        var pushed = Gen5ComputeFixtures.CompileRequestOrThrow(Gen5ComputeFixtures.ExecStore);
        Assert.Contains("constant uint* sharpemu_push_data", pushed.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("shader_data", pushed.Source, StringComparison.Ordinal);

        var tabled = Gen5ComputeFixtures.CompileRequestOrThrow(
            Gen5ComputeFixtures.ExecStore,
            pushDataStartDword: Resources.PushData.DwordCount);
        Assert.Contains("constant uint* shader_data", tabled.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("device const uint* shader_data", tabled.Source, StringComparison.Ordinal);

        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            return;
        }

        // Both layouts have to compile and run, and leave the same bytes behind: the
        // address space is the only thing that differs between them.
        Assert.True(MetalNative.IsAvailable, "The required execution gate needs a Metal device.");
        Assert.Equal(Execute(pushed), Execute(tabled));

        static byte[] Execute(Gen5MslShader shader)
        {
            Assert.True(
                MetalNative.TryCompileLibrary(shader.Source, out var library, out var metalError),
                $"{metalError}\n{shader.Source}");
            Assert.True(
                MetalNative.TryExecuteWithArgumentBuffer(
                    library,
                    shader.EntryPoint,
                    SentinelBuffer(),
                    new byte[Resources.PushData.ByteSize],
                    shader.ArgumentLayout!,
                    threadsPerThreadgroup: 1,
                    out var result,
                    out var runError),
                runError);
            return result;
        }

        static byte[] SentinelBuffer()
        {
            var buffer = new byte[64];
            for (var offset = 0; offset < buffer.Length; offset += sizeof(uint))
            {
                System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(
                    buffer.AsSpan(offset), 0xDEADBEEFu);
            }

            return buffer;
        }
    }

    /// <summary>A 64-lane guest wave runs as two 32-wide Apple simdgroups, so its lane
    /// index goes up to 63 while each half of EXEC is a 32-bit word. Shifting a uint by
    /// 32 or more is undefined in MSL — the count wraps in practice, so lanes 32..63 were
    /// reading the low half and seeing another lane's EXEC bit. The half has to be picked
    /// before the shift.
    ///
    /// The program masks EXEC down to its high half, so only lanes 32..63 write v1, then
    /// restores EXEC and has every lane add v1 into one word. Thirty-two lanes wrote a
    /// one, so the sum is 32; reading the wrong half gives 0 or 64.</summary>
    [Fact]
    public void Wave64TakesEachLanesExecBitFromTheRightHalf()
    {
        var fixture = new Gen5ComputeFixture(
            "wave64-exec-halves",
            [
                0xBEFE0380,             // s_mov_b32 exec_lo, 0
                0xBEFF03C1,             // s_mov_b32 exec_hi, -1
                0x7E020281,             // v_mov_b32 v1, 1      (lanes 32..63 only)
                0xBEFE03C1,             // s_mov_b32 exec_lo, -1
                0xBEFF03C1,             // s_mov_b32 exec_hi, -1
                0xE0C80000, 0x80020100, // buffer_atomic_add v1, off, s[8:11], 0
                0xBF810000,             // s_endpgm
            ],
            StoreScalarResourceBase: 8,
            StoreBackingBytes: 16);

        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(
            fixture, waveLaneCount: 64, localSizeX: 64);

        Assert.Contains("sharpemu_lane < 32u ?", shader.Source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_lane & 31u", shader.Source, StringComparison.Ordinal);

        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            return;
        }

        Assert.True(MetalNative.IsAvailable, "The required execution gate needs a Metal device.");
        Assert.True(
            MetalNative.TryCompileLibrary(shader.Source, out var library, out var metalError),
            $"{metalError}\n{shader.Source}");
        Assert.True(
            MetalNative.TryExecuteWithArgumentBuffer(
                library,
                shader.EntryPoint,
                new byte[fixture.StoreBackingBytes],
                new byte[Resources.PushData.ByteSize],
                shader.ArgumentLayout!,
                threadsPerThreadgroup: 64,
                out var result,
                out var runError),
            runError);
        Assert.Equal(
            32u,
            System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(result));
    }

    /// <summary>A guest wave64's 64-bit mask spans two Apple simdgroups, so a V_CMP has
    /// to exchange halves through threadgroup memory — a barrier, and a threadgroup
    /// barrier makes every simdgroup wait. The per-lane vcc bool needs no exchange at
    /// all, so the barriers are only worth paying when the program reads the pair as
    /// data. Here nothing does: v0 == v0 is balloted into VCC and never looked at.</summary>
    [Fact]
    public void AWave64BallotNothingReadsCostsNoBarriers()
    {
        var shader = CompileWave64Compare(readsTheMask: false);

        Assert.DoesNotContain("threadgroup_barrier", Body(shader), StringComparison.Ordinal);
        Assert.DoesNotContain("sharpemu_wave_scratch", Body(shader), StringComparison.Ordinal);
        // The per-lane view is still set, which is what the EXEC guards read.
        Assert.Contains("vcc = ", shader.Source, StringComparison.Ordinal);

        // And it is the elision that removed them, not the program being trivial.
        var materialized = CompileWave64Compare(
            readsTheMask: false,
            new Gen5MslCodegenOptions(DeadMaskBallotElision: false));
        Assert.Contains("threadgroup_barrier", Body(materialized), StringComparison.Ordinal);

        // A VOPC names VCC implicitly, so nothing in this program looks like a cross-lane
        // operation and the pre-pass that decides whether to declare the bridge used to
        // say no — leaving the exchange referring to an array that was never declared, so
        // the library failed to compile and the dispatch was dropped. The declaration now
        // follows the emitted body.
        Assert.Contains(
            "threadgroup uint sharpemu_wave_scratch[8];",
            materialized.Source,
            StringComparison.Ordinal);
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") == "1")
        {
            Assert.True(MetalNative.IsAvailable, "The required execution gate needs a Metal device.");
            Assert.True(
                MetalNative.TryCompileLibrary(materialized.Source, out _, out var error),
                error + materialized.Source);
        }
    }

    /// <summary>And when the program does read the pair, the exchange has to be there and
    /// has to be right. v0 == v0 holds in every lane, so the mask is all ones across both
    /// halves and VCC_HI reads 0xFFFFFFFF — which only the exchange can produce, because
    /// lanes 0..31 cannot see the other simdgroup's ballot.</summary>
    [Fact]
    public void AWave64BallotSomethingReadsKeepsItsBarriers()
    {
        var shader = CompileWave64Compare(readsTheMask: true);

        Assert.Equal(1, Occurrences(Body(shader), "threadgroup_barrier"));
        Assert.Contains("sharpemu_wave_scratch[sharpemu_wave_set]", shader.Source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_wave_scratch[sharpemu_wave_set + 1u]", shader.Source, StringComparison.Ordinal);

        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            return;
        }

        Assert.True(MetalNative.IsAvailable, "The required execution gate needs a Metal device.");
        Assert.True(
            MetalNative.TryCompileLibrary(shader.Source, out var library, out var metalError),
            $"{metalError}\n{shader.Source}");
        Assert.True(
            MetalNative.TryExecuteWithArgumentBuffer(
                library,
                shader.EntryPoint,
                new byte[16],
                new byte[Resources.PushData.ByteSize],
                shader.ArgumentLayout!,
                threadsPerThreadgroup: 64,
                out var result,
                out var runError),
            runError);
        Assert.Equal(
            0xFFFFFFFFu,
            System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(result));
    }

    /// <summary>The shape the hot Octopath dispatches actually have: a few V_CMPX whose
    /// EXEC an S_CBRANCH_EXECZ reads, mixed with V_CMPs whose VCC nothing reads. The two
    /// pairs are tracked separately, so the EXEC exchanges stay and the VCC ones go — one
    /// exchange for the one live ballot, not one per comparison.</summary>
    [Fact]
    public void OnlyTheLiveHalfOfAMixedWave64ProgramKeepsItsExchange()
    {
        var words = new List<uint>
        {
            0x7D840100, // v_cmp_eq_u32 vcc, v0, v0    -> dead
            0x7D840100, // v_cmp_eq_u32 vcc, v0, v0    -> dead
            0x7DA40100, // v_cmpx_eq_u32 exec, v0, v0  -> read by the branch below
            0xBF880000, // s_cbranch_execz +0
            0x7D840100, // v_cmp_eq_u32 vcc, v0, v0    -> dead
            0x7E0202FF, 0x5A5A5A5Au, // v_mov_b32 v1, 0x5A5A5A5A
            0xE0700000, 0x80020100,  // buffer_store_dword v1, off, s[8:11], 0
            0xBF810000,              // s_endpgm
        };
        var fixture = new Gen5ComputeFixture(
            "wave64-mixed-ballots", [.. words], StoreScalarResourceBase: 8, StoreBackingBytes: 16);
        var program = Gen5ComputeFixtures.DecodeOrThrow(fixture);
        Assert.Equal("VCmpEqU32", program.Instructions[0].Opcode);
        Assert.Equal("VCmpxEqU32", program.Instructions[2].Opcode);
        Assert.Equal("SCbranchExecz", program.Instructions[3].Opcode);

        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(
            fixture, waveLaneCount: 64, localSizeX: 64);
        var body = Body(shader);

        // Four comparisons, one live, so one exchange rather than four. The starting
        // 64-lane EXEC is a constant: a bridged wave is one full threadgroup.
        Assert.Equal(1, Occurrences(body, "sharpemu_wave_scratch[(sharpemu_lane >> 5)"));
        Assert.Contains("s126 = 0xFFFFFFFFu;", body, StringComparison.Ordinal);
        Assert.Contains("s127 = 0xFFFFFFFFu;", body, StringComparison.Ordinal);
        // The register allocator has scalarized the file by now, so EXEC_LO reads s126.
        Assert.Contains("s126 = sharpemu_wave_scratch[sharpemu_wave_set];", body, StringComparison.Ordinal);
        Assert.DoesNotContain("s106 = sharpemu_wave_scratch[sharpemu_wave_set];", body, StringComparison.Ordinal);

        // Materializing everything is what it used to cost.
        var request = Gen5ComputeFixtures.CreateComputeRequest(
            fixture, waveLaneCount: 64, localSizeX: 64);
        Assert.True(
            Gen5MslTranslator.TryCompileProgram(
                request,
                out var every,
                out var error,
                new Gen5MslCodegenOptions(DeadMaskBallotElision: false)),
            error);
        Assert.Equal(4, Occurrences(Body(every), "sharpemu_wave_scratch[(sharpemu_lane >> 5)"));
    }

    /// <summary>A wave64 exchange — V_READFIRSTLANE, V_READLANE, a ballot the program reads —
    /// costs one threadgroup barrier: the slots are double-buffered, so the barrier that
    /// used to follow each read to protect it from the next write is gone. Each exchange
    /// flips the slot set once, so flips and barriers match.</summary>
    [Theory]
    [InlineData("readfirstlane", 2)]
    [InlineData("readlane", 2)]
    [InlineData("loop", 2)]
    public void EveryWave64ExchangeCostsOneBarrier(string program, int exchanges)
    {
        var fixture = program switch
        {
            "readfirstlane" => Gen5ComputeFixtures.Wave64ReadFirstLane,
            "readlane" => Gen5ComputeFixtures.Wave64ReadLane,
            _ => Gen5ComputeFixtures.Wave64ExchangeLoop,
        };
        var body = Body(Gen5ComputeFixtures.CompileRequestOrThrow(fixture, waveLaneCount: 64, localSizeX: 64));

        Assert.Equal(exchanges, Occurrences(body, "threadgroup_barrier"));
        Assert.Equal(exchanges, Occurrences(body, "sharpemu_wave_set ^= 4u;"));
    }

    /// <summary>DS_APPEND counts the wave's active lanes in one exchange and broadcasts the
    /// one atomic's result in a second: two barriers, where it used to take five.</summary>
    [Fact]
    public void AWave64DataShareCounterCostsTwoBarriers()
    {
        var fixture = new Gen5ComputeFixture(
            "wave64-data-share-counter",
            [
                0xD8F80014, 0x07000000, // ds_append v7 offset:20
                0xBF810000,             // s_endpgm
            ],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);
        var body = Body(Gen5ComputeFixtures.CompileRequestOrThrow(fixture, waveLaneCount: 64, localSizeX: 64));

        Assert.Equal(2, Occurrences(body, "sharpemu_wave_set ^= 4u;"));
        Assert.Equal(2, Occurrences(body, "threadgroup_barrier"));
        Assert.Equal(1, Occurrences(body, "atomic_fetch_add_explicit"));
    }

    /// <summary>The loop's exchanges sit in a block the dispatcher re-enters, so the slot
    /// set cannot be chosen when the MSL is emitted: it is a variable every exchange
    /// flips, declared once before the first block, never reset inside the loop.</summary>
    [Fact]
    public void TheWave64SlotSetIsCarriedAcrossLoopTripsAtRuntime()
    {
        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(
            Gen5ComputeFixtures.Wave64ExchangeLoop, waveLaneCount: 64, localSizeX: 64);

        Assert.Equal(1, Occurrences(shader.Source, "uint sharpemu_wave_set = 0u;"));
        Assert.DoesNotContain("sharpemu_wave_set = 0u;", Body(shader).Replace(
            "uint sharpemu_wave_set = 0u;", string.Empty, StringComparison.Ordinal), StringComparison.Ordinal);
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") == "1")
        {
            Assert.True(MetalNative.IsAvailable, "The required execution gate needs a Metal device.");
            Assert.True(MetalNative.TryCompileLibrary(shader.Source, out _, out var error), error + shader.Source);
        }
    }

    private static int Occurrences(string source, string value)
    {
        var count = 0;
        for (var index = source.IndexOf(value, StringComparison.Ordinal); index >= 0;
            index = source.IndexOf(value, index + 1, StringComparison.Ordinal))
        {
            count++;
        }

        return count;
    }

    private static Gen5MslShader CompileWave64Compare(
        bool readsTheMask,
        Gen5MslCodegenOptions? options = null)
    {
        var words = new List<uint>
        {
            0x7D840100, // v_cmp_eq_u32 vcc, v0, v0   -> every lane, both halves
        };
        if (readsTheMask)
        {
            words.Add(0xBE80036B); // s_mov_b32 s0, vcc_hi
            words.Add(0x7E020200); // v_mov_b32 v1, s0
        }
        else
        {
            words.Add(0x7E0202FF); // v_mov_b32 v1, 0x5A5A5A5A
            words.Add(0x5A5A5A5Au);
        }

        words.Add(0xE0700000); // buffer_store_dword v1, off, s[8:11], 0
        words.Add(0x80020100);
        words.Add(0xBF810000); // s_endpgm

        var fixture = new Gen5ComputeFixture(
            readsTheMask ? "wave64-ballot-read" : "wave64-ballot-dead",
            [.. words],
            StoreScalarResourceBase: 8,
            StoreBackingBytes: 16);
        var program = Gen5ComputeFixtures.DecodeOrThrow(fixture);
        Assert.Equal("VCmpEqU32", program.Instructions[0].Opcode);
        if (options is null)
        {
            return Gen5ComputeFixtures.CompileRequestOrThrow(
                fixture, waveLaneCount: 64, localSizeX: 64);
        }

        var request = Gen5ComputeFixtures.CreateComputeRequest(
            fixture, waveLaneCount: 64, localSizeX: 64);
        Assert.True(
            Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error, options),
            error);
        return shader;
    }

    /// <summary>The translated function, without the shared prelude that every shader
    /// carries (and whose helpers contain barriers of their own).</summary>
    private static string Body(Gen5MslShader shader)
    {
        var entry = shader.Source.IndexOf("kernel void ", StringComparison.Ordinal);
        return entry < 0 ? shader.Source : shader.Source[entry..];
    }

    [Fact]
    public void ExecMaskedStoresAreGuarded()
    {
        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(Gen5ComputeFixtures.ExecStore);

        // Every buffer store past the first EXEC write must sit behind the per-lane
        // guard, and a whole-dword store goes out as one dword, not four bytes.
        Assert.Contains("if (exec)", shader.Source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_store_dword(b0,", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("sharpemu_store_bytes(b0,", shader.Source, StringComparison.Ordinal);

        // s_mov_b32 exec_lo, 0 / -1 must drive the per-lane bool.
        Assert.Contains("exec = ((", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void LoopFixtureBecomesARealLoop()
    {
        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(Gen5ComputeFixtures.Loop);

        // The backward branch is a while (true) whose back edge is a continue and
        // whose exit is a break, with the scalar condition driving both.
        Assert.Contains("while (true)", shader.Source, StringComparison.Ordinal);
        Assert.Contains("if (scc)", shader.Source, StringComparison.Ordinal);
        Assert.Contains("continue;", shader.Source, StringComparison.Ordinal);
        Assert.Contains("break;", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("switch (pc)", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void LoopsAreBoundedByDefault()
    {
        // The step cap rides on the back edge now that there is no dispatcher to
        // charge per block, so a loop carries it and straight-line code does not.
        var looping = Gen5ComputeFixtures.CompileRequestOrThrow(Gen5ComputeFixtures.Loop);
        Assert.Contains("if (++steps >=", looping.Source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_note_step_cap(", looping.Source, StringComparison.Ordinal);

        var straightLine = Gen5ComputeFixtures.CompileRequestOrThrow(Gen5ComputeFixtures.Fmac);
        Assert.DoesNotContain("++steps", straightLine.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void OnlyAPersistentThreadKernelGetsTheLargerStepBudget()
    {
        // A persistent-thread kernel spins on a GLC load until other workgroups publish
        // work, so its trip count follows the whole dispatch. Cutting it off early leaves
        // the queue it shares with later dispatches corrupt.
        var persistent = Gen5ComputeFixtures.CompileRequestOrThrow(Gen5ComputeFixtures.PersistentQueueLoop).Source;
        Assert.Contains("if (++steps >= 6400000u)", persistent, StringComparison.Ordinal);

        // Any other loop that polls keeps the plain cap: one whose wait never ends (GTA V has
        // one) would otherwise hold the queue 64 times as long on every dispatch.
        var polling = new Gen5ComputeFixture(
            "polling-loop",
            [
                0xBE800385,             // s_mov_b32 s0, 5
                0xE0305000, 0x80020100, // loop: buffer_load_dword v1, v0, s[8:11], 0 offen glc
                0xBF8C3F70,             // s_waitcnt vmcnt(0)
                0x81808100,             // s_sub_i32 s0, s0, 1
                0xBF078000,             // s_cmp_lg_u32 s0, 0
                0xBF85FFFA,             // s_cbranch_scc1 loop
                0xBF810000,             // s_endpgm
            ],
            StoreScalarResourceBase: 8,
            StoreBackingBytes: 256);
        var pollingSource = Gen5ComputeFixtures.CompileRequestOrThrow(polling).Source;
        Assert.Contains("if (++steps >= 100000u)", pollingSource, StringComparison.Ordinal);

        var plain = Gen5ComputeFixtures.CompileRequestOrThrow(Gen5ComputeFixtures.Loop).Source;
        Assert.Contains("if (++steps >= 100000u)", plain, StringComparison.Ordinal);
    }

    [Fact]
    public void APersistentThreadKernelsRecordsAreWrittenAndReadWholeUnderTheirLock()
    {
        // The persistent queue loop with its dword load and store made a GLC LOAD_DWORDX3 of
        // v2..v4 and a STORE_DWORDX3 of the same: a record another workgroup reads.
        var words = Gen5ComputeFixtures.PersistentQueueLoop.Words.ToArray();
        static void Replace(uint[] program, uint word0, uint word1, uint newWord0)
        {
            var at = Array.FindIndex(program, word => word == word0);
            Assert.True(at >= 0 && program[at + 1] == word1);
            program[at] = newWord0;
        }

        Replace(words, 0xE0305000, 0x80020200, 0xE03C5000); // buffer_load_dwordx3 v[2:4], v0, s[8:11], 0 offen glc
        Replace(words, 0xE0701080, 0x80020200, 0xE07C1080); // buffer_store_dwordx3 v[2:4], v0, s[8:11], 0 offen offset:128
        var fixture = Gen5ComputeFixtures.PersistentQueueLoop with { Name = "persistent-queue-records", Words = words };
        static string Body(string source) => source[source.IndexOf("kernel void ", StringComparison.Ordinal)..];

        var locked = Body(Gen5ComputeFixtures.CompileRequestOrThrow(fixture).Source);
        Assert.Contains("sharpemu_load_record_coherent(", locked, StringComparison.Ordinal);
        Assert.Contains("sharpemu_pair_try_lock(", locked, StringComparison.Ordinal);

        Assert.True(Gen5MslTranslator.TryCompileProgram(
            Gen5ComputeFixtures.CreateComputeRequest(fixture), out var unlocked, out var error,
            new Gen5MslCodegenOptions(PersistentRecordLocks: false)), error);
        Assert.DoesNotContain("sharpemu_load_record_coherent(", Body(unlocked.Source), StringComparison.Ordinal);
        Assert.DoesNotContain("sharpemu_pair_try_lock(", Body(unlocked.Source), StringComparison.Ordinal);
    }

    [Fact]
    public void APersistentThreadKernelsWaitsAreDeviceFencesAndNoOtherKernelsAre()
    {
        // Workgroups of a persistent-thread kernel hand work to each other through memory, and
        // the guest's waits are what make a store visible before the atomic that publishes it.
        const string Fence = "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);";
        static string Body(string source) => source[source.IndexOf("kernel void ", StringComparison.Ordinal)..];
        var persistent = Body(Gen5ComputeFixtures.CompileRequestOrThrow(Gen5ComputeFixtures.PersistentQueueLoop).Source);
        Assert.Contains(Fence, persistent, StringComparison.Ordinal);

        Assert.True(Gen5MslTranslator.TryCompileProgram(
            Gen5ComputeFixtures.CreateComputeRequest(Gen5ComputeFixtures.PersistentQueueLoop),
            out var unfenced, out var error, new Gen5MslCodegenOptions(PersistentWaitFences: false)), error);
        Assert.DoesNotContain(Fence, Body(unfenced.Source), StringComparison.Ordinal);

        // A kernel that only polls (its own wait, no handing over) keeps S_WAITCNT a no-op.
        var polling = new Gen5ComputeFixture(
            "polling-loop-fence",
            [
                0xBE800385,             // s_mov_b32 s0, 5
                0xE0305000, 0x80020100, // loop: buffer_load_dword v1, v0, s[8:11], 0 offen glc
                0xBF8C3F70,             // s_waitcnt vmcnt(0)
                0x81808100,             // s_sub_i32 s0, s0, 1
                0xBF078000,             // s_cmp_lg_u32 s0, 0
                0xBF85FFFA,             // s_cbranch_scc1 loop
                0xBF810000,             // s_endpgm
            ],
            StoreScalarResourceBase: 8,
            StoreBackingBytes: 256);
        Assert.DoesNotContain(Fence, Body(Gen5ComputeFixtures.CompileRequestOrThrow(polling).Source), StringComparison.Ordinal);
    }

    [Fact]
    public void ArgumentBufferCarriesBufferLengthsAndUsesSeparatePushData()
    {
        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(Gen5ComputeFixtures.ExecStore);
        Assert.NotNull(shader.ArgumentLayout);
        Assert.Contains("buffer_bytes [[id(", shader.Source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_resources [[buffer(0)]]", shader.Source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_push_data [[buffer(1)]]", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("SharpEmuUniforms", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void PixelStageEmitsFragmentInterface()
    {
        var shader = Gen5ComputeFixtures.CompilePixelOrThrow();

        Assert.Equal(Gen5MslStage.Pixel, shader.Stage);
        Assert.Equal("gen5_ps", shader.EntryPoint);
        Assert.Equal(1u, shader.AttributeCount);
        Assert.Contains("fragment Gen5PsOut gen5_ps(", shader.Source, StringComparison.Ordinal);
        // The fixture reads attr0 through V_INTERP_MOV's P0 selector, which is
        // the provoking vertex's parameter -- exactly a flat input.
        Assert.Contains("float4 attr0 [[user(locn0), flat]];", shader.Source, StringComparison.Ordinal);
        Assert.Contains("[[color(0)]]", shader.Source, StringComparison.Ordinal);
        Assert.Contains("[[position]]", shader.Source, StringComparison.Ordinal);

        // Interpolation reads land in VGPRs; the export writes MRT0 and inactive
        // lanes discard at the end. This program never narrows EXEC, so the export
        // assigns outright — ExecGuardsSurviveWhereExecIsNarrowed covers the masked form.
        Assert.Contains("as_type<uint>(sharpemu_in.attr0[0])", shader.Source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_out.mrt0 = vec<float, 4>(", shader.Source, StringComparison.Ordinal);
        Assert.Contains("discard_fragment();", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void NullValidMaskExportControlsFragmentDiscard()
    {
        var export = new Gen5ShaderInstruction(
            0,
            Gen5ShaderEncoding.Exp,
            "Exp",
            [],
            [
                Gen5Operand.Vector(0),
                Gen5Operand.Vector(1),
                Gen5Operand.Vector(2),
                Gen5Operand.Vector(3),
            ],
            [],
            new Gen5ExportControl(9, 0, false, true, true));
        var end = new Gen5ShaderInstruction(
            8,
            Gen5ShaderEncoding.Sopp,
            "SEndpgm",
            [0xBF810000],
            [],
            [],
            null);
        var request = Gen5ComputeFixtures.RequestOrThrow(
            new Gen5ShaderProgram(0, [export, end]), ShaderStage.Pixel, pixelOutputs: []);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        Assert.Contains(
            "bool pixel_valid_mask_active = true;",
            shader.Source,
            StringComparison.Ordinal);
        Assert.Contains(
            "pixel_valid_mask_active = exec;",
            shader.Source,
            StringComparison.Ordinal);
        Assert.Contains(
            "if (!pixel_valid_mask_active)",
            shader.Source,
            StringComparison.Ordinal);
    }

    [Fact]
    public void PixelOutputKindsSelectTheAttachmentType()
    {
        var uintShader = Gen5ComputeFixtures.CompilePixelOrThrow(Gen5PixelOutputKind.Uint);
        Assert.Contains("uint4 mrt0 [[color(0)]];", uintShader.Source, StringComparison.Ordinal);

        var sintShader = Gen5ComputeFixtures.CompilePixelOrThrow(Gen5PixelOutputKind.Sint);
        Assert.Contains("int4 mrt0 [[color(0)]];", sintShader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void IdentityPixelOutputKeepsGuestComponentOrder()
    {
        var shader = Gen5ComputeFixtures.CompilePixelOrThrow();

        Assert.Contains(
            "vec<float, 4>(as_type<float>(v0), as_type<float>(v1), " +
            "as_type<float>(v2), as_type<float>(v3))",
            shader.Source,
            StringComparison.Ordinal);
    }

    [Fact]
    public void BgraPixelOutputMapsGuestComponentsToPhysicalOrder()
    {
        var shader = Gen5ComputeFixtures.CompilePixelOrThrow(
            componentMapping: new Gen5ColorComponentMapping(0xC6));

        Assert.Contains(
            "vec<float, 4>(as_type<float>(v2), as_type<float>(v1), " +
            "as_type<float>(v0), as_type<float>(v3))",
            shader.Source,
            StringComparison.Ordinal);
    }

    [Fact]
    public void VertexStageEmitsVertexInterface()
    {
        var shader = Gen5ComputeFixtures.CompileVertexOrThrow();

        Assert.Equal(Gen5MslStage.Vertex, shader.Stage);
        Assert.Equal("gen5_vs", shader.EntryPoint);
        Assert.Equal(1u, shader.AttributeCount);
        Assert.Contains("vertex Gen5VsOut gen5_vs(", shader.Source, StringComparison.Ordinal);
        Assert.Contains("float4 sharpemu_position [[position, invariant]];", shader.Source, StringComparison.Ordinal);
        Assert.Contains("float4 param0 [[user(locn0)]];", shader.Source, StringComparison.Ordinal);
        Assert.Contains("uint sharpemu_vertex_id [[vertex_id]],", shader.Source, StringComparison.Ordinal);
        Assert.Contains("v5 = sharpemu_vertex_id;", shader.Source, StringComparison.Ordinal);
        Assert.Contains("v8 = sharpemu_instance_id;", shader.Source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_out.sharpemu_position = float4(", shader.Source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_out.param0 = float4(", shader.Source, StringComparison.Ordinal);
        Assert.Contains("return sharpemu_out;", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void RequiredVertexOutputsAreZeroFilledDeclarations()
    {
        // The paired fragment shader reads locations 0..2; the program only
        // exports param0, so 1 and 2 must still be declared (zero-filled).
        var shader = Gen5ComputeFixtures.CompileVertexOrThrow(requiredVertexOutputCount: 3);
        Assert.Equal(3u, shader.AttributeCount);
        Assert.Contains("float4 param1 [[user(locn1)]];", shader.Source, StringComparison.Ordinal);
        Assert.Contains("float4 param2 [[user(locn2)]];", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void FixedShadersCoverThePresenterSurface()
    {
        var fullscreen = MslFixedShaders.CreateFullscreenVertex(2);
        Assert.Contains("vertex FullscreenOut fullscreen_vs(", fullscreen, StringComparison.Ordinal);
        Assert.Contains("float4 attr1 [[user(locn1)]];", fullscreen, StringComparison.Ordinal);

        Assert.Contains("tex0.sample(smp0, in.attr0.xy)", MslFixedShaders.CreateCopyFragment(), StringComparison.Ordinal);
        Assert.Contains("float4(1.0f, 0.0f, 1.0f, 1.0f)", MslFixedShaders.CreateSolidFragment(1f, 0f, 1f, 1f), StringComparison.Ordinal);
        Assert.Contains("return in.attr3;", MslFixedShaders.CreateAttributeFragment(3), StringComparison.Ordinal);
        Assert.Contains("fragment void depth_only_fs()", MslFixedShaders.CreateDepthOnlyFragment(), StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(0xFACu, "as_type<float>(v4), as_type<float>(v5), as_type<float>(v6), as_type<float>(v7)")]
    [InlineData(0x9F5u, "as_type<float>(v7), as_type<float>(v4), as_type<float>(v5), as_type<float>(v6)")]
    [InlineData(0xF2Eu, "as_type<float>(v6), as_type<float>(v5), as_type<float>(v4), as_type<float>(v7)")]
    [InlineData(0x3ACu, "as_type<float>(v4), as_type<float>(v5), as_type<float>(v6), 0.0f")]
    [InlineData(0xFA4u, "as_type<float>(v4), 0.0f, as_type<float>(v6), as_type<float>(v7)")]
    public void ImageStoreAppliesInverseDescriptorSwizzle(
        uint dstSelect,
        string expectedComponents)
    {
        var shader = CompileImageStore(dstSelect, dmask: 0xF);

        Assert.Contains(
            $".write(vec<float, 4>({expectedComponents}),",
            shader.Source,
            StringComparison.Ordinal);
    }

    [Fact]
    public void ImageStoreTreatsZeroDmaskAsX()
    {
        var shader = CompileImageStore(
            Gen5ShaderTranslator.IdentityImageDstSelect,
            dmask: 0);

        Assert.Contains(
            ".write(vec<float, 4>(as_type<float>(v4), 0.0f, 0.0f, 0.0f),",
            shader.Source,
            StringComparison.Ordinal);
    }

    [Fact]
    public void ImageStoreMipAppliesInverseDescriptorSwizzle()
    {
        var shader = CompileImageStore(
            0x9F5u,
            dmask: 0xF,
            opcode: "ImageStoreMip");

        Assert.Contains(
            ".write(vec<float, 4>(as_type<float>(v7), as_type<float>(v4), as_type<float>(v5), as_type<float>(v6)),",
            shader.Source,
            StringComparison.Ordinal);
    }

    [Fact]
    public void UintImageStoreUsesUintTextureAndInverseDescriptorSwizzle()
    {
        var shader = CompileImageStore(
            0xF2Eu,
            dmask: 0xF,
            unifiedFormat: 69u); // FORMAT_16_16_16_16_UINT

        Assert.Contains(
            "texture2d<uint, access::write>",
            shader.Source,
            StringComparison.Ordinal);
        Assert.Contains(
            ".write(vec<uint, 4>(v6, v5, v4, v7),",
            shader.Source,
            StringComparison.Ordinal);
    }

    [Fact]
    public void UnsupportedOpcodeFailsLoudlyWithPc()
    {
        // s_sethalt is real but has no host equivalent: the translator must
        // name the opcode and pc instead of emitting wrong code.
        var fixture = new Gen5ComputeFixture(
            "unsupported",
            [
                0xBF8D0001,             // s_sethalt 1
                0xBF810000,             // s_endpgm
            ],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);
        var exception = Assert.Throws<InvalidOperationException>(
            () => Gen5ComputeFixtures.CompileRequestOrThrow(fixture));
        Assert.Contains("pc=0x", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void RelativeVectorSourceUsesM0ForDynamicRead()
    {
        var fixture = new Gen5ComputeFixture(
            "relative-vector-source",
            [
                0x7E1802FF, 0x11111111, // v_mov_b32 v12, 0x11111111
                0x7E1A02FF, 0x22222222, // v_mov_b32 v13, 0x22222222
                0x7E6E870C,             // v_movrels_b32 v55, v12
                0xBF810000,             // s_endpgm
            ],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(fixture);

        // The relative number is still the encoded one plus M0, wrapped into the file,
        // but nothing indexes the register file with it: the read is a choice among the
        // registers this program touches, so they all stay in hardware registers. A
        // register the program never writes holds undefined data in hardware, which is
        // why only its own registers have to be reachable.
        Assert.Contains("(12u + ", shader.Source, StringComparison.Ordinal);
        Assert.Contains("s124", shader.Source, StringComparison.Ordinal);
        Assert.Contains(") & 255u", shader.Source, StringComparison.Ordinal);
        Assert.Matches(@"case 12u: t\d+ = v12; break;", shader.Source);
        Assert.Matches(@"case 13u: t\d+ = v13; break;", shader.Source);
        Assert.Matches(@"default: t\d+ = v\d+; break; }", shader.Source);
        Assert.DoesNotContain("uint v[256]", shader.Source, StringComparison.Ordinal);
    }

    /// <summary>A program past 64 VGPRs still resolves a run-time read: the switch costs a
    /// few compares however many registers it chooses among, so the file never falls back
    /// to a thread-private array (Silent Hill's FFT bloom touches 124).</summary>
    [Fact]
    public void RelativeVectorSourceResolvesPastSixtyFourRegisters()
    {
        var words = new List<uint>();
        for (var register = 1u; register <= 80u; register++)
        {
            words.Add(0x7E0002FFu | (register << 17)); // v_mov_b32 vN, literal
            words.Add(register);
        }

        words.Add(0x7E00870Cu | (100u << 17)); // v_movrels_b32 v100, v12
        words.Add(0xBF810000);                 // s_endpgm
        var fixture = new Gen5ComputeFixture(
            "relative-vector-source-wide",
            [.. words],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(fixture);

        Assert.DoesNotContain("uint v[256]", shader.Source, StringComparison.Ordinal);
        Assert.Matches(@"case 80u: t\d+ = v80; break;", shader.Source);
    }

    /// <summary>When straight-line scalar code produces M0, the move reaches a known set
    /// of registers and resolves to them directly — no run-time number at all.</summary>
    [Fact]
    public void RelativeVectorSourceWithAKnownM0ResolvesToOneRegister()
    {
        var fixture = new Gen5ComputeFixture(
            "relative-vector-source-constant",
            [
                0xBEFC0382,             // s_mov_b32 m0, 2
                0x7E1C02FF, 0x33333333, // v_mov_b32 v14, 0x33333333
                0x7E6E870C,             // v_movrels_b32 v55, v12
                0xBF810000,             // s_endpgm
            ],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(fixture);

        // 12 + 2: the source is v14 outright, with no register number to resolve.
        Assert.Contains("= v14;", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("& 255u", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("== 14u ?", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("uint v[256]", shader.Source, StringComparison.Ordinal);

        // With the switch off the same program indexes the register file at run time,
        // and the file has to stay an array for it.
        var request = Gen5ComputeFixtures.CreateComputeRequest(fixture);
        Assert.True(
            Gen5MslTranslator.TryCompileProgram(
                request,
                out var unresolved,
                out var error,
                new Gen5MslCodegenOptions(BoundedRelativeMoves: false)),
            error);
        Assert.Contains("(12u + ", unresolved.Source, StringComparison.Ordinal);
        Assert.Matches(@"v\[t\d+\]", unresolved.Source);
        Assert.Contains("uint v[256] = {};", unresolved.Source, StringComparison.Ordinal);
    }

    [Theory]
    // V_MOVRELD writes the relative destination from an ordinary src0, V_MOVRELSD
    // makes both ends relative, and V_MOVRELSD2 splits M0 into two offsets.
    [InlineData(0x7E6E850Cu, "VMovreldB32")]
    [InlineData(0x7E6E890Cu, "VMovrelsdB32")]
    [InlineData(0x7E6E910Cu, "VMovrelsd2B32")]
    public void RelativeMoveFormsAddressTheRegisterFileThroughM0(uint word, string opcode)
    {
        var fixture = new Gen5ComputeFixture(
            $"relative-{opcode}",
            [word, 0xBF810000],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);
        var program = Gen5ComputeFixtures.DecodeOrThrow(fixture);
        Assert.Equal(opcode, program.Instructions[0].Opcode);

        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(fixture);
        Assert.Contains("s124", shader.Source, StringComparison.Ordinal);
        // The destination is the relative one for every form but V_MOVRELS, and
        // the relative number wraps into the register file.
        Assert.Contains("(55u + ", shader.Source, StringComparison.Ordinal);
        Assert.Contains("& 255u", shader.Source, StringComparison.Ordinal);

        // These forms write the register M0 names, so the file stays an array. Resolving
        // a read over the program's own registers is sound because a register it never
        // writes holds undefined data; a write lands on one the program does mean to
        // write and may read back through the same number later, so it cannot be dropped
        // for being outside the set.
        Assert.Contains("uint v[256] = {};", shader.Source, StringComparison.Ordinal);

        // V_MOVRELSD2 reads M0's two halves as independent offsets.
        Assert.Equal(
            opcode == "VMovrelsd2B32",
            shader.Source.Contains(">> 16u) & 0x3FFu", StringComparison.Ordinal));
    }

    [Fact]
    public void ScalarBlockerOpcodesCompileWithRdna2Semantics()
    {
        var fixture = new Gen5ComputeFixture(
            "scalar-blockers",
            [
                0xBF130200, // s_cmp_lg_u64 s[0:1], s[2:3]
                0xBE861404, // s_ff1_i32_b64 s6, s[4:5]
                0xBEEB106A, // s_bcnt1_i32_b64 s107, s[106:107]
                0xBE890908, // s_wqm_b32 s9, s8
                0xBF810000, // s_endpgm
            ],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(fixture);

        Assert.Contains(" != ", shader.Source, StringComparison.Ordinal);
        Assert.Contains("(uint)ctz(", shader.Source, StringComparison.Ordinal);
        Assert.Contains("(uint)popcount(", shader.Source, StringComparison.Ordinal);
        Assert.Contains("& 0x11111111u", shader.Source, StringComparison.Ordinal);
        Assert.Contains("* 0xFu", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void DataShareWaveCountersUseOneAtomicPerWave()
    {
        var fixture = new Gen5ComputeFixture(
            "data-share-wave-counters",
            [
                0xD8F80014, 0x07000000, // ds_append v7 offset:20
                0xD8F40014, 0x08000000, // ds_consume v8 offset:20
                0xBF810000,             // s_endpgm
            ],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(fixture);

        Assert.Contains("popcount(", shader.Source, StringComparison.Ordinal);
        Assert.Contains(">> 16u", shader.Source, StringComparison.Ordinal);
        Assert.Contains("& 0xFFFFu", shader.Source, StringComparison.Ordinal);
        Assert.Contains("atomic_fetch_add_explicit", shader.Source, StringComparison.Ordinal);
        Assert.Contains("atomic_fetch_sub_explicit", shader.Source, StringComparison.Ordinal);
        Assert.Contains("simd_broadcast", shader.Source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(0xBF970001u)]
    [InlineData(0xBF980001u)]
    [InlineData(0xBF990001u)]
    [InlineData(0xBF9A0001u)]
    public void DebugConditionBranchesFallThroughWithoutShaderDebugger(uint branch)
    {
        var fixture = new Gen5ComputeFixture(
            "debug-condition-branch",
            [
                branch,
                0xBF800000, // s_nop 0
                0xBF810000, // s_endpgm
            ],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(fixture);

        Assert.Contains("if (!(false))", shader.Source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(true, 22u)]
    [InlineData(false, 0u)]
    public void TypedBufferLoadUsesTheInstructionFormatAndUntypedKeepsTheDescriptorTable(bool typed, uint typedFormat)
    {
        var load = new Gen5ShaderInstruction(
            0,
            typed ? Gen5ShaderEncoding.Mtbuf : Gen5ShaderEncoding.Mubuf,
            typed ? "TBufferLoadFormatXyzw" : "BufferLoadFormatXyzw",
            [0, 0],
            [Gen5Operand.Vector(0), Gen5Operand.Scalar(8), Gen5Operand.Source(128, null)],
            [Gen5Operand.Vector(4)],
            new Gen5BufferMemoryControl(
                4,
                0,
                4,
                8,
                0,
                IndexEnabled: false,
                OffsetEnabled: false,
                Glc: false,
                Slc: false,
                Typed: typed,
                TypedFormat: typedFormat));
        var end = new Gen5ShaderInstruction(8, Gen5ShaderEncoding.Sopp, "SEndpgm", [0xBF810000], [], [], null);
        var scalars = new uint[256];
        scalars[8] = 0x2000;
        scalars[10] = 64;
        // The untyped load takes the descriptor's format. 8_8 is not whole dwords, so it keeps
        // the table form (a whole-dword literal format loads as one run, tested separately).
        scalars[11] = (typed ? 77u : 14u) << 12;
        var shader = CompileMaterialized(new Gen5ShaderProgram(0, [load, end]), scalars[..12]);

        if (typed)
        {
            Assert.DoesNotContain("sharpemu_gfx10_formats[(", shader.Source, StringComparison.Ordinal);
            Assert.Contains("sharpemu_load_words<1>(b0, ", shader.Source, StringComparison.Ordinal);
        }
        else
        {
            Assert.Contains("sharpemu_gfx10_formats[(", shader.Source, StringComparison.Ordinal);
            Assert.Contains("== 7u ?", shader.Source, StringComparison.Ordinal);
            Assert.Contains(": true);", shader.Source, StringComparison.Ordinal);
            Assert.DoesNotContain("sharpemu_format_bytes", shader.Source, StringComparison.Ordinal);
        }
    }

    [Theory]
    [InlineData("TBufferLoadFormatX", 1u, 22u)]
    [InlineData("TBufferLoadFormatXy", 2u, 64u)]
    [InlineData("TBufferLoadFormatXyzw", 4u, 77u)]
    public void TypedLoadOfWholeDwordComponentsReadsOneRunFromOneAddress(string opcode, uint count, uint format)
    {
        // Unified formats 22, 64 and 77 are 32, 32_32 and 32_32_32_32.
        var shader = CompileTypedBufferAccess(opcode, count, format);

        Assert.Contains($"sharpemu_load_words<{count}>(b0, ", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("sharpemu_load_word(b0, ", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void TypedLoadOfSubDwordComponentsKeepsThePerComponentPath()
    {
        // Format 14 is 8_8 unorm.
        var shader = CompileTypedBufferAccess("TBufferLoadFormatXy", 2, typedFormat: 14);

        Assert.DoesNotContain("sharpemu_load_words<", shader.Source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_load_word(b0, ", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void TypedBufferStoreEncodesAndPlacesEachComponentInOneElement()
    {
        // Format 14 is 8_8 unorm: both registers are encoded, packed into one element
        // with a two-byte mask, and stored under the descriptor and element bounds.
        var shader = CompileTypedBufferAccess("TBufferStoreFormatXy", dwordCount: 2, typedFormat: 14);

        Assert.Contains("sharpemu_format_encode(v4, 8u, 0u, 3u)", shader.Source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_format_encode(v5, 8u, 0u, 3u)", shader.Source, StringComparison.Ordinal);
        Assert.Contains("<< 8u), 0u, 0u, 0u), uint4(0xFFFFu, 0x0u, 0x0u, 0x0u));", shader.Source, StringComparison.Ordinal);
        Assert.Contains("2u <= sharpemu_resources.buffer_bytes[0] && ", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("sharpemu_store_bytes(b", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void TypedBufferStoreOfWholeDwordsKeepsTheRegisterBits()
    {
        // Format 77 is 32x4 float: four aligned dword stores, no encoding, one bounds test.
        var shader = CompileTypedBufferAccess("TBufferStoreFormatXyzw", dwordCount: 4, typedFormat: 77);

        Assert.Contains(", v4);", shader.Source, StringComparison.Ordinal);
        Assert.Contains("+ 12u), v7);", shader.Source, StringComparison.Ordinal);
        Assert.Contains("16u <= sharpemu_resources.buffer_bytes[0] && ", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("sharpemu_format_encode(v[", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("sharpemu_store_element(b", shader.Source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("TBufferLoadFormatX", 1u, 4u)]
    [InlineData("TBufferLoadFormatXy", 2u, 8u)]
    [InlineData("TBufferStoreFormatX", 1u, 4u)]
    [InlineData("TBufferStoreFormatXy", 2u, 8u)]
    public void TypedBufferBoundsUseOnlyTransferredComponents(string opcode, uint componentCount, uint accessBytes)
    {
        var shader = CompileTypedBufferAccess(opcode, componentCount, typedFormat: 77);
        Assert.Contains($"{accessBytes}u <= sharpemu_resources.buffer_bytes[0] && ", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("16u <= sharpemu_resources.buffer_bytes[0] && ", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void FormattedBufferStoreUsesDescriptorFormatForPackedNormals()
    {
        var shader = CompileTypedBufferAccess("BufferStoreFormatXyzw", 4, 0, typed: false, descriptorFormat: 50);

        // 2_10_10_10 is R10G10B10A2: ten-bit X, Y and Z, then the two-bit W.
        Assert.Contains("sharpemu_format_encode(v4, 10u, 0u, 9u)", shader.Source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_format_encode(v5, 10u, 0u, 9u)", shader.Source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_format_encode(v6, 10u, 0u, 9u)", shader.Source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_format_encode(v7, 2u, 0u, 9u)", shader.Source, StringComparison.Ordinal);
        Assert.Contains("uint4(0xFFFFFFFFu, 0x0u, 0x0u, 0x0u)", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("sharpemu_store_bytes(b", shader.Source, StringComparison.Ordinal);
    }

    [Theory]
    // opcode, registers, components, typed format (8_8_8_8 unorm) or the descriptor's (32_32_32_32 float)
    [InlineData("TBufferLoadFormatD16Xyzw", 2u, 4, true)]
    [InlineData("TBufferLoadFormatD16Xyz", 2u, 3, true)]
    [InlineData("BufferLoadFormatD16Xy", 1u, 2, false)]
    [InlineData("BufferLoadFormatD16HiX", 1u, 1, false)]
    public void D16FormattedLoadPacksTwoHalvesPerRegister(string opcode, uint registers, int components, bool typed)
    {
        var shader = CompileTypedBufferAccess(opcode, registers, typedFormat: 56, typed: typed);

        Assert.Equal(components, CountOccurrences(shader.Source, "= sharpemu_format_d16("));
        // D16 never takes the whole-dword run: a 32-bit channel still narrows to a half.
        Assert.DoesNotContain("sharpemu_load_words<", shader.Source, StringComparison.Ordinal);
        if (opcode.EndsWith("HiX", StringComparison.Ordinal))
            Assert.Contains("v4 = (v4 & 0x0000FFFFu) | (", shader.Source, StringComparison.Ordinal);
        if (components == 3)
            Assert.Contains("v5 = (v5 & 0xFFFF0000u) | ", shader.Source, StringComparison.Ordinal);
        if (MetalNative.IsAvailable)
            Assert.True(MetalNative.TryCompileLibrary(shader.Source, out _, out var error), error);
    }

    [Fact]
    public void D16FormattedStoreWidensEachHalfBeforeEncoding()
    {
        // Format 14 is 8_8 unorm; X is v4's low half and Y its high half.
        var shader = CompileTypedBufferAccess("TBufferStoreFormatD16Xy", dwordCount: 1, typedFormat: 14);

        Assert.Contains("sharpemu_format_d16_expand((v4 & 0xFFFFu), 0u)", shader.Source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_format_d16_expand((v4 >> 16u), 0u)", shader.Source, StringComparison.Ordinal);
        if (MetalNative.IsAvailable)
            Assert.True(MetalNative.TryCompileLibrary(shader.Source, out _, out var error), error);
    }

    private static int CountOccurrences(string text, string value)
    {
        var count = 0;
        for (var index = text.IndexOf(value, StringComparison.Ordinal); index >= 0;
             index = text.IndexOf(value, index + value.Length, StringComparison.Ordinal))
        {
            count++;
        }

        return count;
    }

    private static Gen5MslShader CompileTypedBufferAccess(string opcode, uint dwordCount, uint typedFormat,
        bool typed = true, uint descriptorFormat = 77)
    {
        var data = new Gen5Operand[dwordCount];
        for (var index = 0; index < data.Length; index++)
        {
            data[index] = Gen5Operand.Vector(4 + (uint)index);
        }

        var access = new Gen5ShaderInstruction(
            0,
            typed ? Gen5ShaderEncoding.Mtbuf : Gen5ShaderEncoding.Mubuf,
            opcode,
            [0, 0],
            [Gen5Operand.Vector(0), Gen5Operand.Scalar(8), Gen5Operand.Source(128, null)],
            data,
            new Gen5BufferMemoryControl(
                dwordCount,
                0,
                4,
                8,
                0,
                IndexEnabled: false,
                OffsetEnabled: false,
                Glc: false,
                Slc: false,
                Typed: typed,
                TypedFormat: typedFormat));
        var end = new Gen5ShaderInstruction(8, Gen5ShaderEncoding.Sopp, "SEndpgm", [0xBF810000], [], [], null);
        var scalars = new uint[256];
        scalars[8] = 0x2000;
        scalars[10] = 64;
        scalars[11] = descriptorFormat << 12;
        var shader = CompileMaterialized(new Gen5ShaderProgram(0, [access, end]), scalars[..12]);
        return shader;
    }

    [Fact]
    public void FormattedVertexFetchZeroFillsMissingComponents()
    {
        var fetch = new Gen5ShaderInstruction(
            0,
            Gen5ShaderEncoding.Mubuf,
            "BufferLoadFormatXyz",
            [],
            [],
            [],
            new Gen5BufferMemoryControl(
                3,
                5,
                0,
                0,
                0,
                IndexEnabled: true,
                OffsetEnabled: false,
                Glc: false,
                Slc: false));
        var end = new Gen5ShaderInstruction(
            4,
            Gen5ShaderEncoding.Sopp,
            "SEndpgm",
            [],
            [],
            [],
            null);
        var program = new Gen5ShaderProgram(0, [fetch, end]);
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Vertex, 1, 0, 0, new HashSet<uint> { 0 });
        var resources = ResourceMaterializer.ApplyTo(plan, ResourceSpecialization.Default(plan.Info));
        var layout = BindingLayout.Allocate(resources.Info, [], false, false, false);
        var request = new ShaderCompileRequest(plan, resources, layout)
        {
            VertexInputs = [new ShaderVertexInput(0, 0, 2, 7, false, [])],
        };
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        Assert.Contains("v2 = 0u;", shader.Source, StringComparison.Ordinal);
    }

    private static Gen5MslShader CompileImageStore(
        uint dstSelect,
        uint dmask,
        string opcode = "ImageStore",
        uint unifiedFormat = 71u)
    {
        var control = new Gen5ImageControl(
            Dmask: dmask,
            VectorAddress: 0,
            AddressRegisters: [0, 1],
            VectorData: 4,
            ScalarResource: 8,
            ScalarSampler: 16,
            Dimension: 1,
            IsArray: false,
            Glc: false,
            Slc: false,
            A16: false,
            D16: false);
        var store = new Gen5ShaderInstruction(
            0,
            Gen5ShaderEncoding.Mimg,
            opcode,
            [],
            [],
            [],
            control);
        var end = new Gen5ShaderInstruction(
            8,
            Gen5ShaderEncoding.Sopp,
            "SEndpgm",
            [0xBF810000],
            [],
            [],
            null);
        var userData = new uint[16];
        userData[8] = 0x20;
        userData[9] = unifiedFormat << 20;
        userData[11] = (9u << 28) | dstSelect;
        return CompileMaterialized(new Gen5ShaderProgram(0x1_0000_C000, [store, end]), userData);
    }
    private static Gen5MslShader CompileMaterialized(Gen5ShaderProgram program, uint[] userData)
    {
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, 1, 0, (uint)userData.Length);
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, new ResourceRuntimeInputs { UserData = userData },
            ref snapshot, ref specialization));
        var resources = ResourceMaterializer.ApplyTo(plan, specialization);
        var layout = BindingLayout.Allocate(resources.Info,
            BindingLayout.CollectUserDataRegisters(program, 0, (uint)userData.Length), false,
            ShaderCompileRequest.RequiresFlattenedTable(plan, resources), false);
        var request = new ShaderCompileRequest(plan, resources, layout);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        return shader;
    }

}
