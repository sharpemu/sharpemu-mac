// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using Xunit;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>A wave64 compute workgroup of several waves whose program reads a lane of the upper
/// half by number bridges each wave's two simdgroups (Gen5MslTranslator.IsBridgedComputeWaves):
/// Ghost of Tsushima's tiled light culler totals a wave from lanes 31 and 63, and run as wave32s
/// each half read its own lane 31 twice.</summary>
public sealed class Gen5MslBridgedWavesTests
{
    private const uint Threads = 128;

    private static Gen5MslShader Compile(string name, uint lane)
    {
        var words = new uint[]
        {
            0xD7600014, 0x00010100 | ((128u + lane) << 9), // v_readlane_b32 s20, v0, lane
            0x7E020214,                                    // v_mov_b32 v1, s20
            0x34040082,                                    // v_lshlrev_b32 v2, 2, v0
            0xE0701000, 0x80020102,                        // buffer_store_dword v1, v2, s[8:11], 0 offen
            0xBF810000,                                    // s_endpgm
        };
        var fixture = new Gen5ComputeFixture(name, words, StoreScalarResourceBase: 8, StoreBackingBytes: (int)Threads * 4);
        Gen5ComputeFixtures.DecodeOrThrow(fixture);
        var request = Gen5ComputeFixtures.CreateComputeRequest(fixture, waveLaneCount: 64, localSizeX: Threads);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        return shader;
    }

    [Fact]
    public void EachWaveReadsItsOwnUpperLane()
    {
        var shader = Compile("bridged-waves-lane63", 63);
        Assert.Contains("sharpemu_wave_pair", shader.Source, StringComparison.Ordinal);
        if (!MetalNative.IsAvailable)
        {
            Assert.False(
                OperatingSystem.IsMacOS() && Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") == "1",
                "The required gate cannot run the bridged wave test without a Metal device.");
            return;
        }

        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var libraryError), libraryError);
        Assert.True(
            MetalNative.TryExecuteWithArgumentBuffer(
                library, shader.EntryPoint, new byte[Threads * 4], new byte[Resources.PushData.ByteSize],
                shader.ArgumentLayout!, threadsPerThreadgroup: Threads, out var result, out var runError),
            runError);
        for (var thread = 0; thread < Threads; thread++)
        {
            Assert.Equal(thread < 64 ? 63u : 127u, BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan(thread * 4)));
        }
    }

    [Fact]
    public void AProgramThatReadsOnlyTheLowerHalfKeepsOneSimdgroupAWave()
    {
        var shader = Compile("bridged-waves-lane5", 5);
        Assert.DoesNotContain("sharpemu_wave_pair", shader.Source, StringComparison.Ordinal);
    }
}
