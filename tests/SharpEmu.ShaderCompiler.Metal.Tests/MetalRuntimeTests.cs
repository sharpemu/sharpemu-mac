// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>
/// Tier 2 and 3: the emitted MSL must compile with the OS runtime Metal compiler, and
/// the executable fixtures must produce bit-exact results on the GPU, including EXEC
/// masking and dispatcher control flow. These tests no-op (with a note) on hosts
/// without a Metal device so the suite stays green on Windows/Linux CI; the golden
/// and structural tiers still run everywhere.
/// </summary>
public sealed class MetalRuntimeTests(ITestOutputHelper output)
{
    [Fact]
    public void AllFixturesCompileWithTheRuntimeMetalCompiler()
    {
        if (SkipWithoutMetalDevice("compile validation")) return;

        foreach (var fixture in Gen5ComputeFixtures.All)
        {
            var shader = Gen5ComputeFixtures.CompileRequestOrThrow(fixture);
            Assert.True(
                MetalNative.TryCompileLibrary(shader.Source, out _, out var error),
                $"[{fixture.Name}] Metal rejected the emitted MSL: {error}\n{shader.Source}");
        }
    }

    [Fact]
    public void ExecStoreProgramExecutesWithExecMasking()
    {
        if (SkipWithoutMetalDevice("the execution test")) return;

        // Sentinel-filled buffer: any dword the program does not store must survive.
        const uint Sentinel = 0xDEADBEEFu;
        var buffer = new byte[64];
        for (var offset = 0; offset < buffer.Length; offset += sizeof(uint))
        {
            BinaryPrimitives.WriteUInt32LittleEndian(buffer.AsSpan(offset), Sentinel);
        }

        var result = ExecuteRequestOrThrow(Gen5ComputeFixtures.ExecStore, buffer);

        // Reference results computed with the same semantics the program encodes.
        var fmac = BitConverter.SingleToUInt32Bits(MathF.FusedMultiplyAdd(1.5f, 2.25f, 10.0f));
        var mulHiSigned = (uint)(((long)0x7FFFFFFF * 0x00010003) >> 32);
        var mulLoSigned = unchecked(0x7FFFFFFFu * 0x00010003u);
        var movBits = BitConverter.SingleToUInt32Bits(1.5f);

        Assert.Equal(fmac, ReadDword(result, 0));
        Assert.Equal(mulHiSigned, ReadDword(result, 4));
        Assert.Equal(mulLoSigned, ReadDword(result, 8));
        Assert.Equal(Sentinel, ReadDword(result, 12)); // EXEC=0: the store must not land.
        Assert.Equal(movBits, ReadDword(result, 16));
        for (var offset = 20; offset < result.Length; offset += sizeof(uint))
        {
            Assert.Equal(Sentinel, ReadDword(result, offset));
        }
    }

    [Fact]
    public void LoopProgramIteratesThroughTheDispatcher()
    {
        if (SkipWithoutMetalDevice("the execution test")) return;

        var result = ExecuteRequestOrThrow(Gen5ComputeFixtures.Loop, new byte[16]);

        // 5 + 4 + 3 + 2 + 1, accumulated across five dispatcher round trips.
        Assert.Equal(15u, ReadDword(result, 0));
    }

    [Fact]
    public void ALoopContinuingThroughASharedLatchIsStructuredAndRunsCorrectly()
    {
        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(Gen5ComputeFixtures.SharedLatchLoop);
        Assert.DoesNotContain("switch (pc)", shader.Source, StringComparison.Ordinal);

        if (SkipWithoutMetalDevice("the execution test")) return;

        var result = ExecuteRequestOrThrow(Gen5ComputeFixtures.SharedLatchLoop, new byte[16]);
        Assert.Equal(0x00080167u, ReadDword(result, 0));
    }

    [Fact]
    public void ATailSharedByBothArmsOfABranchInALoopIsStructuredAndRunsCorrectly()
    {
        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(Gen5ComputeFixtures.SharedTailLoop);
        Assert.DoesNotContain("switch (pc)", shader.Source, StringComparison.Ordinal);

        if (SkipWithoutMetalDevice("the execution test")) return;

        var result = ExecuteRequestOrThrow(Gen5ComputeFixtures.SharedTailLoop, new byte[16]);
        Assert.Equal(0x00060804u, ReadDword(result, 0));
    }

    // A loop with two exits used to send the whole shader to the dispatcher. Each exit now
    // records its number and breaks, and the code after the loop runs that exit's region.
    [Theory]
    [InlineData(true, 0x00000A02u)]
    [InlineData(false, 0x00000B04u)]
    public void ALoopWithTwoExitsIsStructuredAndRunsTheExitItLeftThrough(bool first, uint expected)
    {
        var fixture = first ? Gen5ComputeFixtures.MultiExitLoopFirst : Gen5ComputeFixtures.MultiExitLoopSecond;
        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(fixture);
        Assert.DoesNotContain("switch (pc)", shader.Source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_loop_exit_", shader.Source, StringComparison.Ordinal);

        if (SkipWithoutMetalDevice("the execution test")) return;

        var result = ExecuteRequestOrThrow(fixture, new byte[16]);
        Assert.Equal(expected, ReadDword(result, 0));
    }

    // A branch whose arm can end the program has no post-dominator; the arms that go on
    // now rejoin where they meet instead of sending the shader to the dispatcher.
    [Theory]
    [InlineData(3u, 1u)]
    [InlineData(5u, 9u)]
    [InlineData(7u, 3u)]
    public void ABranchWithAnExitingArmIsStructuredAndRunsEachPath(uint selector, uint expected)
    {
        var fixture = Gen5ComputeFixtures.ExitArmJoin(selector);
        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(fixture);
        Assert.DoesNotContain("switch (pc)", shader.Source, StringComparison.Ordinal);

        if (SkipWithoutMetalDevice("the execution test")) return;

        var result = ExecuteRequestOrThrow(fixture, new byte[16]);
        Assert.Equal(expected, ReadDword(result, 0));
    }

    [Fact]
    public void PixelShaderCompilesWithTheRuntimeMetalCompiler()
    {
        if (SkipWithoutMetalDevice("compile validation")) return;

        var shader = Gen5ComputeFixtures.CompilePixelOrThrow();
        Assert.True(
            MetalNative.TryCompileLibrary(shader.Source, out _, out var error),
            $"[pixel] Metal rejected the emitted MSL: {error}\n{shader.Source}");
    }

    [Fact]
    public void VertexShaderCompilesWithTheRuntimeMetalCompiler()
    {
        if (SkipWithoutMetalDevice("compile validation")) return;

        var shader = Gen5ComputeFixtures.CompileVertexOrThrow(requiredVertexOutputCount: 2);
        Assert.True(
            MetalNative.TryCompileLibrary(shader.Source, out _, out var error),
            $"[vertex] Metal rejected the emitted MSL: {error}\n{shader.Source}");
    }

    [Fact]
    public void FixedShadersCompileWithTheRuntimeMetalCompiler()
    {
        if (SkipWithoutMetalDevice("compile validation")) return;

        var sources = new (string Name, string Source)[]
        {
            ("fullscreen", MslFixedShaders.CreateFullscreenVertex(3)),
            ("copy", MslFixedShaders.CreateCopyFragment()),
            ("present", MslFixedShaders.CreatePresentFragment()),
            ("solid", MslFixedShaders.CreateSolidFragment(0.25f, 0.5f, 0.75f, 1f)),
            ("attribute", MslFixedShaders.CreateAttributeFragment(1)),
            ("depth-only", MslFixedShaders.CreateDepthOnlyFragment()),
        };
        foreach (var (name, source) in sources)
        {
            Assert.True(
                MetalNative.TryCompileLibrary(source, out _, out var error),
                $"[{name}] Metal rejected the fixed shader: {error}\n{source}");
        }
    }

    [Fact]
    public void LdsRoundTripExecutesThroughThreadgroupMemory()
    {
        if (SkipWithoutMetalDevice("the execution test")) return;

        var result = ExecuteRequestOrThrow(Gen5ComputeFixtures.Lds, new byte[16]);

        // Written to LDS, barriered, read back, stored to the buffer.
        Assert.Equal(0x1234u, ReadDword(result, 0));
    }

    [Fact]
    public void Wave64CrossLaneEmitsValidMsl()
    {
        if (SkipWithoutMetalDevice("compile validation")) return;

        // A wave64 program with a cross-lane op emits the threadgroup-scratch
        // bridge and its barriers; the runtime Metal compiler must accept it.
        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(
            Gen5ComputeFixtures.Wave64Broadcast, waveLaneCount: 64, localSizeX: 64);
        Assert.Contains("sharpemu_wave_scratch", shader.Source);
        Assert.Contains("threadgroup_barrier", shader.Source);
        Assert.True(
            MetalNative.TryCompileLibrary(shader.Source, out _, out var error),
            $"Metal rejected the wave64 MSL: {error}\n{shader.Source}");
    }

    [Fact]
    public void Wave64BroadcastExecutesAcrossBothHalvesWithoutDeadlock()
    {
        if (SkipWithoutMetalDevice("the execution test")) return;

        // 64 threads = one guest wave (two 32-wide simdgroups). The read-first-
        // lane bridge takes a threadgroup_barrier reached by all 64 lanes; if
        // the two halves did not rendezvous this would deadlock (the command
        // buffer would never complete). Every lane holds 42, so the broadcast
        // result is 42 — proving the bridge runs to completion and returns the
        // published value.
        var result = ExecuteRequestOrThrow(Gen5ComputeFixtures.Wave64Broadcast, new byte[16],
            waveLaneCount: 64, localSizeX: 64, threadsPerThreadgroup: 64);

        Assert.Equal(42u, ReadDword(result, 0));
    }

    [Fact]
    public void Wave64LdsExchangeIsOrderedAcrossHalvesWithoutABarrier()
    {
        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(
            Gen5ComputeFixtures.Wave64LdsExchange, waveLaneCount: 64, localSizeX: 64);
        var write = shader.Source.IndexOf("sharpemu_lds[", StringComparison.Ordinal);
        Assert.True(write >= 0, shader.Source);
        var barrier = shader.Source.IndexOf("threadgroup_barrier", write, StringComparison.Ordinal);
        Assert.True(barrier > write, $"no threadgroup_barrier after the LDS write\n{shader.Source}");
        var read = shader.Source.IndexOf("sharpemu_lds[", barrier, StringComparison.Ordinal);
        Assert.True(read > barrier, $"no LDS read after the barrier\n{shader.Source}");
    }

    [Fact]
    public void Wave64LdsExchangeSeesTheOtherHalfsWrites()
    {
        if (SkipWithoutMetalDevice("the execution test")) return;

        for (var round = 0; round < 20; round++)
        {
            var result = ExecuteRequestOrThrow(Gen5ComputeFixtures.Wave64LdsExchange, new byte[256],
                waveLaneCount: 64, localSizeX: 64, threadsPerThreadgroup: 64);
            for (var lane = 0; lane < 64; lane++)
            {
                Assert.Equal((uint)(((lane + 32) & 63) + 1), ReadDword(result, lane * 4));
            }
        }
    }

    [Fact]
    public void Wave64ReadLaneReadsTheFullWaveInBothHalves()
    {
        if (SkipWithoutMetalDevice("the execution test")) return;

        var result = ExecuteRequestOrThrow(Gen5ComputeFixtures.Wave64ReadLane, new byte[512],
            waveLaneCount: 64, localSizeX: 64, threadsPerThreadgroup: 64);

        // Lane l holds 3 * l + 7: every lane, in either half, sees lane 63's and lane 5's.
        for (var lane = 0; lane < 64; lane++)
        {
            Assert.Equal(3u * 63u + 7u, ReadDword(result, lane * 8));
            Assert.Equal(3u * 5u + 7u, ReadDword(result, lane * 8 + 4));
        }
    }

    [Fact]
    public void Wave64SpillReloadsReadACopyInsteadOfTheExchange()
    {
        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(
            Gen5ComputeFixtures.Wave64LaneSpill, waveLaneCount: 64, localSizeX: 64);
        Assert.Contains("sharpemu_spill1", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("threadgroup_barrier", shader.Source, StringComparison.Ordinal);

        if (SkipWithoutMetalDevice("the execution test")) return;

        var result = ExecuteRequestOrThrow(Gen5ComputeFixtures.Wave64LaneSpill, new byte[512],
            waveLaneCount: 64, localSizeX: 64, threadsPerThreadgroup: 64);
        for (var lane = 0; lane < 64; lane++)
        {
            Assert.Equal(0x1234u, ReadDword(result, lane * 8));
            Assert.Equal(77u, ReadDword(result, lane * 8 + 4));
        }
    }

    [Fact]
    public void Wave64SpillParkedOnOneArmIsStillReloadedFromTheCopy()
    {
        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(
            Gen5ComputeFixtures.Wave64LaneSpillOneArm, waveLaneCount: 64, localSizeX: 64);
        Assert.Contains("sharpemu_spill0", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("threadgroup_barrier", shader.Source, StringComparison.Ordinal);

        if (SkipWithoutMetalDevice("the execution test")) return;

        var result = ExecuteRequestOrThrow(Gen5ComputeFixtures.Wave64LaneSpillOneArm, new byte[512],
            waveLaneCount: 64, localSizeX: 64, threadsPerThreadgroup: 64);
        for (var lane = 0; lane < 64; lane++)
        {
            Assert.Equal(0x1234u, ReadDword(result, lane * 8));
            Assert.Equal(0x1234u, ReadDword(result, lane * 8 + 4));
        }
    }

    [Fact]
    public void Wave64SpillReloadAfterTheVgprChangedReadsTheLane()
    {
        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(
            Gen5ComputeFixtures.Wave64LaneSpillOverwritten, waveLaneCount: 64, localSizeX: 64);
        Assert.DoesNotContain("sharpemu_spill", shader.Source, StringComparison.Ordinal);
        Assert.Contains("threadgroup_barrier", shader.Source, StringComparison.Ordinal);

        if (SkipWithoutMetalDevice("the execution test")) return;

        var result = ExecuteRequestOrThrow(Gen5ComputeFixtures.Wave64LaneSpillOverwritten, new byte[512],
            waveLaneCount: 64, localSizeX: 64, threadsPerThreadgroup: 64);
        for (var lane = 0; lane < 64; lane++)
        {
            Assert.Equal(0x1235u, ReadDword(result, lane * 8));
            Assert.Equal(78u, ReadDword(result, lane * 8 + 4));
        }
    }

    [Fact]
    public void Wave64ReadFirstLaneTakesTheFirstActiveLaneOfEitherHalf()
    {
        if (SkipWithoutMetalDevice("the execution test")) return;

        var result = ExecuteRequestOrThrow(Gen5ComputeFixtures.Wave64ReadFirstLane, new byte[512],
            waveLaneCount: 64, localSizeX: 64, threadsPerThreadgroup: 64);

        // Lane 44 when only the upper half has EXEC, lane 8 when only the lower half does.
        for (var lane = 0; lane < 64; lane++)
        {
            Assert.Equal(3u * 44u + 7u, ReadDword(result, lane * 8));
            Assert.Equal(3u * 8u + 7u, ReadDword(result, lane * 8 + 4));
        }
    }

    [Fact]
    public void Wave64ReadFirstLaneWithEmptyExecReadsLaneZero()
    {
        if (SkipWithoutMetalDevice("the execution test")) return;

        var result = ExecuteRequestOrThrow(Gen5ComputeFixtures.Wave64ReadFirstLaneEmptyExec, new byte[512],
            waveLaneCount: 64, localSizeX: 64, threadsPerThreadgroup: 64);

        for (var lane = 0; lane < 64; lane++)
        {
            Assert.Equal(7u, ReadDword(result, lane * 8));
        }
    }

    [Fact]
    public void Wave64ExchangesInALoopNeverReadAnOverwrittenSlot()
    {
        if (SkipWithoutMetalDevice("the execution test")) return;

        // 128 exchanges back to back at one barrier each: a trip that read a slot the next
        // exchange had already rewritten would add another trip's value into the sum.
        for (var round = 0; round < 20; round++)
        {
            var result = ExecuteRequestOrThrow(Gen5ComputeFixtures.Wave64ExchangeLoop, new byte[256],
                waveLaneCount: 64, localSizeX: 64, threadsPerThreadgroup: 64);
            for (var lane = 0; lane < 64; lane++)
            {
                Assert.Equal(8512u, ReadDword(result, lane * 4));
            }
        }
    }

    [Fact]
    public void Wave64WriteLaneChangesOnlyTheAddressedLane()
    {
        if (SkipWithoutMetalDevice("the execution test")) return;

        var result = ExecuteRequestOrThrow(Gen5ComputeFixtures.Wave64WriteLane, new byte[256],
            waveLaneCount: 64, localSizeX: 64, threadsPerThreadgroup: 64);

        // Lane 40 is in the upper half; lane 8 (its twin in the lower half) must not change.
        for (var lane = 0; lane < 64; lane++)
        {
            Assert.Equal(lane == 40 ? 55u : (uint)lane, ReadDword(result, lane * 4));
        }
    }

    [Fact]
    public void Wave64MaskBitsAreReadAtTheFullLaneIndex()
    {
        if (SkipWithoutMetalDevice("the execution test")) return;

        var result = ExecuteRequestOrThrow(Gen5ComputeFixtures.Wave64MaskBits, new byte[512],
            waveLaneCount: 64, localSizeX: 64, threadsPerThreadgroup: 64);

        const ulong Mask = 0xF0F0F0F0_0000FFFFul;
        for (var lane = 0; lane < 64; lane++)
        {
            var expected = (uint)((Mask >> lane) & 1ul);
            Assert.Equal(expected, ReadDword(result, lane * 8));      // SGPR mask bit
            Assert.Equal(expected, ReadDword(result, lane * 8 + 4));  // saveexec EXEC bit
        }
    }

    [Fact]
    public void Vop3CompareDecodesItsScalarDestination()
    {
        var compare = Gen5ComputeFixtures.DecodeOrThrow(Gen5ComputeFixtures.Wave64Vop3CompareSgpr)
            .Instructions.Single(static instruction => instruction.Opcode == "VCmpEqU32");
        Assert.Equal(Gen5ShaderEncoding.Vop3, compare.Encoding);
        Assert.Equal([Gen5Operand.Scalar(4)], compare.Destinations);
    }

    [Fact]
    public void Vop3ClassCompareWritesTheScalarRegisterItNames()
    {
        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(new Gen5ComputeFixture(
            "vop3-cmp-class-sgpr",
            [
                0xD4880004, 0x00020300, // v_cmp_class_f32_e64 s4, v0, v1
                0xBF810000,             // s_endpgm
            ],
            0,
            0));

        // Registers may be emitted as s[4] or promoted to a local s4.
        Assert.Matches(@"s\[?4\]? = sharpemu_ballot\(", shader.Source);
        Assert.DoesNotMatch(@"s\[?106\]? = sharpemu_ballot\(", shader.Source);
    }

    [Theory]
    [InlineData(64u)]
    [InlineData(32u)]
    public void Vop3CompareWritesTheScalarRegisterItNamesAndLeavesVccAlone(uint waveLanes)
    {
        if (SkipWithoutMetalDevice("the execution test")) return;

        var wave64 = waveLanes == 64;
        var fixture = wave64
            ? Gen5ComputeFixtures.Wave64Vop3CompareSgpr
            : Gen5ComputeFixtures.Wave32Vop3CompareSgpr;
        var result = ExecuteRequestOrThrow(fixture, new byte[fixture.StoreBackingBytes],
            waveLaneCount: waveLanes, localSizeX: waveLanes, threadsPerThreadgroup: waveLanes);

        for (var lane = 0; lane < (int)waveLanes; lane++)
        {
            var hit = (lane & 3) == 0 ? 1u : 0u;
            var offset = lane * 32;
            Assert.Equal(hit, ReadDword(result, offset));                         // s[4:5] bit
            Assert.Equal((0x0F0F0F0Fu >> (lane & 31)) & 1u, ReadDword(result, offset + 4)); // VCC bit
            Assert.Equal(0x11111111u, ReadDword(result, offset + 8));            // s4
            Assert.Equal(wave64 ? 0x11111111u : 0xA5A5A5A5u, ReadDword(result, offset + 12)); // s5
            Assert.Equal(0x0F0F0F0Fu, ReadDword(result, offset + 16));           // vcc_lo
            Assert.Equal(hit, ReadDword(result, offset + 20));                    // saveexec
        }
    }

    // A queue-draining loop over the whole program is a persistent-threads kernel; a loop that
    // does not poll, and a program with no loop, are not.
    [Fact]
    public void OnlyAQueueDrainingLoopIsMarkedPersistentThreads()
    {
        Assert.True(Gen5ComputeFixtures.CompileRequestOrThrow(Gen5ComputeFixtures.PersistentQueueLoop).PersistentThreads);
        Assert.False(Gen5ComputeFixtures.CompileRequestOrThrow(Gen5ComputeFixtures.PollingLoop).PersistentThreads);
        Assert.False(Gen5ComputeFixtures.CompileRequestOrThrow(Gen5ComputeFixtures.CoherentBufferCopy).PersistentThreads);
        Assert.False(Gen5ComputeFixtures.CompileRequestOrThrow(Gen5ComputeFixtures.Loop).PersistentThreads);
    }

    // V_MAD_F32 rounds its product like a separate multiply and add; V_FMA_F32 does not.
    [Fact]
    public void LegacyMadRoundsTheProductAndFmaDoesNot()
    {
        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(Gen5ComputeFixtures.MadVersusFma);
        Assert.Contains("#pragma clang fp contract(off)", shader.Source, StringComparison.Ordinal);

        if (SkipWithoutMetalDevice("the execution test")) return;

        var result = ExecuteRequestOrThrow(Gen5ComputeFixtures.MadVersusFma, new byte[256],
            waveLaneCount: 32, localSizeX: 32, threadsPerThreadgroup: 32);
        for (var lane = 0; lane < 32; lane++)
        {
            Assert.Equal(0u, ReadDword(result, lane * 8));
            Assert.Equal(0x33800000u, ReadDword(result, lane * 8 + 4));
        }
    }

    [Fact]
    public void Vop3CompareIntoNullWritesNothingAndIntoExecMasksTheWave()
    {
        if (SkipWithoutMetalDevice("the execution test")) return;

        var fixture = Gen5ComputeFixtures.Wave64Vop3CompareNullAndExec;
        var result = ExecuteRequestOrThrow(fixture, new byte[fixture.StoreBackingBytes],
            waveLaneCount: 64, localSizeX: 64, threadsPerThreadgroup: 64);

        for (var lane = 0; lane < 64; lane++)
        {
            Assert.Equal(1u, ReadDword(result, lane * 16));                           // EXEC intact
            Assert.Equal((0x0F0F0F0Fu >> (lane & 31)) & 1u, ReadDword(result, lane * 16 + 4)); // VCC
            Assert.Equal((lane & 3) == 0 ? 1u : 0u, ReadDword(result, lane * 16 + 8)); // EXEC = mask
        }
    }

    [Fact]
    public void Vop3CarryOutWritesOnlyTheWave32ScalarAndNothingForNull()
    {
        if (SkipWithoutMetalDevice("the execution test")) return;

        var fixture = Gen5ComputeFixtures.Wave32Vop3CarryOutSgpr;
        var result = ExecuteRequestOrThrow(fixture, new byte[fixture.StoreBackingBytes],
            waveLaneCount: 32, localSizeX: 32, threadsPerThreadgroup: 32);

        for (var lane = 0; lane < 32; lane++)
        {
            var offset = lane * 32;
            Assert.Equal(lane >= 16 ? 1u : 0u, ReadDword(result, offset));  // s4 carry bit
            Assert.Equal(0xFFFF0000u, ReadDword(result, offset + 4));       // s4
            Assert.Equal(0xA5A5A5A5u, ReadDword(result, offset + 8));       // s5
            Assert.Equal(0x0F0F0F0Fu, ReadDword(result, offset + 12));      // vcc_lo
            Assert.Equal(0xFFFFFFFFu, ReadDword(result, offset + 16));      // exec_lo
            Assert.Equal(unchecked((uint)lane - 16u), ReadDword(result, offset + 20)); // sum
        }
    }

    [Fact]
    public void Wave64InALargerWorkgroupTranslatesWithoutTheThreadgroupBridge()
    {
        // The SPIR-V translator bridges only a 64-thread (one-wave) workgroup; a 256-thread
        // group would share the half-indexed scratch between its four waves.
        var bridged = Gen5ComputeFixtures.CompileRequestOrThrow(
            Gen5ComputeFixtures.Wave64Broadcast, waveLaneCount: 64, localSizeX: 64);
        Assert.Contains("sharpemu_wave_scratch", bridged.Source);

        foreach (var fixture in new[] { Gen5ComputeFixtures.Wave64Broadcast, Gen5ComputeFixtures.Wave64ReadLane })
        {
            var shader = Gen5ComputeFixtures.CompileRequestOrThrow(fixture, waveLaneCount: 64, localSizeX: 256);
            Assert.DoesNotContain("sharpemu_wave_scratch", shader.Source);
            Assert.DoesNotContain("threadgroup_barrier", shader.Source);
        }
    }

    [Fact]
    public void TypedLoadFixtureReturnsInstructionFormatComponents()
    {
        if (SkipWithoutMetalDevice("the execution test")) return;

        const uint Sentinel = 0xDEADBEEFu;
        var buffer = new byte[32];
        for (var offset = 0; offset < buffer.Length; offset += sizeof(uint))
        {
            BinaryPrimitives.WriteUInt32LittleEndian(buffer.AsSpan(offset), Sentinel);
        }

        var first = BitConverter.SingleToUInt32Bits(1.5f);
        var second = BitConverter.SingleToUInt32Bits(-2.25f);
        BinaryPrimitives.WriteUInt32LittleEndian(buffer.AsSpan(0), first);
        BinaryPrimitives.WriteUInt32LittleEndian(buffer.AsSpan(4), second);

        var result = ExecuteRequestOrThrow(Gen5ComputeFixtures.TypedLoad, buffer);

        // The descriptor says unorm bytes; the instruction's 32_32 float format wins.
        Assert.Equal(first, ReadDword(result, 16));
        Assert.Equal(second, ReadDword(result, 20));
        Assert.Equal(Sentinel, ReadDword(result, 8));
        Assert.Equal(Sentinel, ReadDword(result, 12));
        Assert.Equal(Sentinel, ReadDword(result, 24));
        Assert.Equal(Sentinel, ReadDword(result, 28));
    }

    [Fact]
    public void TypedStoreFixtureEncodesComponentsAtTheirFormatOffsets()
    {
        if (SkipWithoutMetalDevice("the execution test")) return;

        const uint Sentinel = 0xDEADBEEFu;
        var buffer = new byte[32];
        for (var offset = 0; offset < buffer.Length; offset += sizeof(uint))
        {
            BinaryPrimitives.WriteUInt32LittleEndian(buffer.AsSpan(offset), Sentinel);
        }

        var result = ExecuteRequestOrThrow(Gen5ComputeFixtures.TypedStore, buffer);

        // Format 8_8 unorm: 1.0 and 0.5 encode to 0xFF and 0x80 in their own bytes; the rest is untouched.
        Assert.Equal(0xDEAD80FFu, ReadDword(result, 16));
        for (var offset = 0; offset < result.Length; offset += sizeof(uint))
        {
            if (offset != 16) Assert.Equal(Sentinel, ReadDword(result, offset));
        }
    }

    [Theory]
    [InlineData(true, false, 1u, 28u, 0xFACu, true)]
    [InlineData(true, false, 2u, 24u, 0xFACu, true)]
    [InlineData(true, false, 2u, 28u, 0xFACu, false)]
    [InlineData(false, false, 1u, 28u, 0xFACu, true)]
    [InlineData(false, false, 2u, 28u, 4u | (4u << 3), true)]
    [InlineData(false, false, 2u, 28u, 0xFACu, false)]
    [InlineData(true, true, 1u, 28u, 0xFACu, true)]
    [InlineData(true, true, 2u, 24u, 0xFACu, true)]
    [InlineData(true, true, 2u, 28u, 0xFACu, false)]
    public void FormattedAccessBoundsUseOnlyRequiredComponents(
        bool typed, bool store, uint componentCount, uint offset, uint selectors, bool inBounds)
    {
        if (SkipWithoutMetalDevice("the component bounds test")) return;

        var buffer = new byte[32];
        for (var position = 0; position < buffer.Length; position += 4)
            BinaryPrimitives.WriteUInt32LittleEndian(buffer.AsSpan(position), 0xDEADBEEF);
        var expected = (byte[])buffer.Clone();
        var words = new List<uint> { 0xBE8B03FF, (77u << 12) | selectors };
        if (store)
        {
            words.AddRange([0x7E0402FF, 0x3F800000, 0x7E0602FF, 0x40000000]);
        }
        var opcode = componentCount - 1 + (store ? 4u : 0u);
        words.Add(typed ? 0xE8000000 | (77u << 19) | (opcode << 16) | offset
            : 0xE0000000 | (opcode << 18) | offset);
        words.Add(0x80020200);
        for (uint component = 0; component < componentCount; component++)
        {
            if (store)
            {
                if (inBounds)
                    BinaryPrimitives.WriteUInt32LittleEndian(expected.AsSpan((int)(offset + component * 4)),
                        component == 0 ? 0x3F800000u : 0x40000000u);
            }
            else
            {
                words.Add(0xE0700000 | (component * 4));
                words.Add(0x80020000 | ((2 + component) << 8));
                BinaryPrimitives.WriteUInt32LittleEndian(expected.AsSpan((int)component * 4), inBounds ? 0xDEADBEEFu : 0u);
            }
        }
        words.Add(0xBF810000);
        var fixture = new Gen5ComputeFixture("component-bounds", words.ToArray(), StoreScalarResourceBase: 8, StoreBackingBytes: 32);
        Assert.Equal(expected, ExecuteRequestOrThrow(fixture, buffer));
    }

    // ---- the request path: the argument buffer replaces the uniforms and per-slot buffers ----

    [Fact]
    public void AllFixturesCompileThroughTheArgumentBuffer()
    {
        if (SkipWithoutMetalDevice("compile validation")) return;

        foreach (var fixture in Gen5ComputeFixtures.All)
        {
            var shader = Gen5ComputeFixtures.CompileRequestOrThrow(fixture);
            Assert.True(
                MetalNative.TryCompileLibrary(shader.Source, out _, out var error),
                $"[{fixture.Name}] Metal rejected the request-path MSL: {error}\n{shader.Source}");
        }

        var pixel = Gen5ComputeFixtures.CompileStageRequestOrThrow(Gen5ComputeFixtures.PixelWords, Resources.ShaderStage.Pixel);
        Assert.True(MetalNative.TryCompileLibrary(pixel.Source, out _, out var pixelError), $"[pixel] {pixelError}\n{pixel.Source}");
        var vertex = Gen5ComputeFixtures.CompileStageRequestOrThrow(Gen5ComputeFixtures.VertexWords, Resources.ShaderStage.Vertex, requiredVertexOutputCount: 2);
        Assert.True(MetalNative.TryCompileLibrary(vertex.Source, out _, out var vertexError), $"[vertex] {vertexError}\n{vertex.Source}");
    }

    [Fact]
    public void ExecStoreProgramExecutesThroughTheArgumentBuffer()
    {
        if (SkipWithoutMetalDevice("the execution test")) return;

        const uint Sentinel = 0xDEADBEEFu;
        var buffer = new byte[64];
        for (var offset = 0; offset < buffer.Length; offset += sizeof(uint))
        {
            BinaryPrimitives.WriteUInt32LittleEndian(buffer.AsSpan(offset), Sentinel);
        }

        var result = ExecuteRequestOrThrow(Gen5ComputeFixtures.ExecStore, buffer);

        var fmac = BitConverter.SingleToUInt32Bits(MathF.FusedMultiplyAdd(1.5f, 2.25f, 10.0f));
        var mulHiSigned = (uint)(((long)0x7FFFFFFF * 0x00010003) >> 32);
        var mulLoSigned = unchecked(0x7FFFFFFFu * 0x00010003u);
        var movBits = BitConverter.SingleToUInt32Bits(1.5f);

        Assert.Equal(fmac, ReadDword(result, 0));
        Assert.Equal(mulHiSigned, ReadDword(result, 4));
        Assert.Equal(mulLoSigned, ReadDword(result, 8));
        Assert.Equal(Sentinel, ReadDword(result, 12));
        Assert.Equal(movBits, ReadDword(result, 16));
    }

    [Fact]
    public void LoopProgramIteratesThroughTheArgumentBuffer()
    {
        if (SkipWithoutMetalDevice("the execution test")) return;

        var result = ExecuteRequestOrThrow(Gen5ComputeFixtures.Loop, new byte[16]);
        Assert.Equal(15u, ReadDword(result, 0));
    }

    [Fact]
    public void TypedLoadFixtureReturnsInstructionFormatComponentsThroughTheArgumentBuffer()
    {
        if (SkipWithoutMetalDevice("the execution test")) return;

        const uint Sentinel = 0xDEADBEEFu;
        var buffer = new byte[32];
        for (var offset = 0; offset < buffer.Length; offset += sizeof(uint))
        {
            BinaryPrimitives.WriteUInt32LittleEndian(buffer.AsSpan(offset), Sentinel);
        }

        var first = BitConverter.SingleToUInt32Bits(1.5f);
        var second = BitConverter.SingleToUInt32Bits(-2.25f);
        BinaryPrimitives.WriteUInt32LittleEndian(buffer.AsSpan(0), first);
        BinaryPrimitives.WriteUInt32LittleEndian(buffer.AsSpan(4), second);

        var result = ExecuteRequestOrThrow(Gen5ComputeFixtures.TypedLoad, buffer);

        Assert.Equal(first, ReadDword(result, 16));
        Assert.Equal(second, ReadDword(result, 20));
        Assert.Equal(Sentinel, ReadDword(result, 8));
    }

    private static byte[] ExecuteRequestOrThrow(Gen5ComputeFixture fixture, byte[] buffer,
        uint waveLaneCount = 32, uint localSizeX = 32, uint threadsPerThreadgroup = 1)
    {
        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(fixture, waveLaneCount, localSizeX);
        if (!MetalNative.TryCompileLibrary(shader.Source, out var library, out var compileError))
        {
            throw new InvalidOperationException($"[{fixture.Name}] {compileError}\n{shader.Source}");
        }

        // The user data registers hold zero; bounds come from the argument buffer lengths.
        var pushData = new byte[Resources.PushData.ByteSize];
        if (!MetalNative.TryExecuteWithArgumentBuffer(
                library,
                shader.EntryPoint,
                buffer,
                pushData,
                shader.ArgumentLayout ?? throw new InvalidOperationException($"[{fixture.Name}] no argument layout"),
                threadsPerThreadgroup,
                out var result,
                out var runError))
        {
            throw new InvalidOperationException($"[{fixture.Name}] {runError}");
        }

        return result;
    }

    // Without a Metal device the test skips, unless the required gate runs on a Mac.
    private bool SkipWithoutMetalDevice(string activity)
    {
        if (MetalNative.IsAvailable) return false;
        Assert.False(
            OperatingSystem.IsMacOS() && Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") == "1",
            $"The required gate cannot run {activity} without a Metal device.");
        output.WriteLine($"No Metal device on this host; {activity} skipped.");
        return true;
    }

    private static uint ReadDword(byte[] buffer, int offset) =>
        BinaryPrimitives.ReadUInt32LittleEndian(buffer.AsSpan(offset));
}
