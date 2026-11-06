// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>Wave64 mask producers and consumers keep per-lane bools and exchange the two
/// simdgroups' halves only where a scalar reader sees the pair (Gen5MslTranslator.LaneMasks.cs).</summary>
public sealed class Gen5MslLaneMaskTests
{
    private const uint VCmpGtU32To_S2 = 0xD4C40002; // v_cmp_gt_u32 s[2:3], ...
    private const uint VCmpLtU32To_S2 = 0xD4C10002; // v_cmp_lt_u32 s[2:3], ...
    private const uint AndSaveexecS4S2 = 0xBE842402; // s_and_saveexec_b64 s[4:5], s[2:3]
    private const uint OrExecExecS4 = 0x88FE047E;    // s_or_b64 exec, exec, s[4:5]
    private const uint ExecZeroSkipsStore = 0xBF880002; // s_cbranch_execz +2
    private const uint StoreV0 = 0xE0700000;          // buffer_store_dword v0, off, s[8:11], 0
    private const uint StoreV0Word1 = 0x80020000;
    private const uint EndProgram = 0xBF810000;

    private static string Body(Gen5MslShader shader)
    {
        var entry = shader.Source.IndexOf("kernel void ", StringComparison.Ordinal);
        return entry < 0 ? shader.Source : shader.Source[entry..];
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

    private static Gen5MslShader Compile(string name, uint[] words, Gen5MslCodegenOptions? options = null)
    {
        var fixture = new Gen5ComputeFixture(name, words, StoreScalarResourceBase: 8, StoreBackingBytes: 16);
        Gen5ComputeFixtures.DecodeOrThrow(fixture);
        var request = Gen5ComputeFixtures.CreateComputeRequest(fixture, waveLaneCount: 64, localSizeX: 64);
        Assert.True(
            Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error, options),
            error);
        return shader;
    }

    private static uint[] Program(uint compare, uint compareOperands, bool skipOnExecz, bool restore)
    {
        var words = new List<uint> { compare, compareOperands, AndSaveexecS4S2 };
        if (skipOnExecz)
        {
            words.Add(ExecZeroSkipsStore);
        }

        words.Add(StoreV0);
        words.Add(StoreV0Word1);
        if (restore)
        {
            words.Add(OrExecExecS4);
        }

        words.Add(EndProgram);
        return [.. words];
    }

    // v_cmp s[2:3]; s_and_saveexec s[4:5], s[2:3]; store: nothing reads a pair, so no exchange.
    private static readonly uint[] Upper = Program(VCmpGtU32To_S2, 0x00017700, skipOnExecz: false, restore: false);

    [Fact]
    public void ACompareFeedingOnlyPerLaneConsumersCostsNoBarrier()
    {
        var shader = Compile("lane-mask-chain", Upper);
        Assert.Equal(0, Occurrences(Body(shader), "threadgroup_barrier"));

        // Before, the compare's pair was read by the saveexec and so exchanged.
        var every = Compile("lane-mask-chain-every", Upper, new Gen5MslCodegenOptions(DeadMaskBallotElision: false));
        Assert.True(Occurrences(Body(every), "threadgroup_barrier") >= 1);
    }

    [Fact]
    public void AnExecBranchMaterializesTheNewExecPairOnce()
    {
        var shader = Compile(
            "lane-mask-execz",
            Program(VCmpGtU32To_S2, 0x00017700, skipOnExecz: true, restore: true),
            new Gen5MslCodegenOptions { HalfLocalExecBranches = false });
        Assert.Equal(1, Occurrences(Body(shader), "threadgroup_barrier"));
    }

    [Fact]
    public void AMaskReadAsDataStillExchanges()
    {
        // v_cmp s[2:3]; s_and_saveexec s[4:5], s[2:3]; s_mov_b32 s0, s3 ; the pair s[2:3] is read as data.
        var words = new uint[]
        {
            VCmpGtU32To_S2, 0x00017700,
            AndSaveexecS4S2,
            0xBE800303, // s_mov_b32 s0, s3
            0x7E020200, // v_mov_b32 v1, s0
            0xE0700000, 0x80020100, // buffer_store_dword v1
            EndProgram,
        };
        var shader = Compile("lane-mask-data", words);
        Assert.Equal(1, Occurrences(Body(shader), "threadgroup_barrier"));
    }

    [Fact]
    public void ACndmaskReadsTheLaneBoolInsteadOfThePair()
    {
        // v_cmp s[2:3]; v_cndmask_b32 v1, v1, v0, s[2:3]; store
        var words = new uint[]
        {
            VCmpGtU32To_S2, 0x00017700,
            0xD5010001, 0x000A0101, // v_cndmask_b32 v1, v1, v0, s[2:3]
            0xE0700000, 0x80020100,
            EndProgram,
        };
        var shader = Compile("lane-mask-cndmask", words);
        Assert.Equal(0, Occurrences(Body(shader), "threadgroup_barrier"));
    }

    [Fact]
    public void AJoinWhereOnePathOverwritesThePairReadsTheDwords()
    {
        // v_cmp s[2:3]; s_cbranch_scc0 +1; s_mov_b32 s2, 0; v_cndmask ... s[2:3]
        var words = new uint[]
        {
            VCmpGtU32To_S2, 0x00017700,
            0xBF840001,             // s_cbranch_scc0 +1
            0xBE820380,             // s_mov_b32 s2, 0
            0xD5010001, 0x000A0101, // v_cndmask_b32 v1, v1, v0, s[2:3]
            0xE0700000, 0x80020100,
            EndProgram,
        };
        var body = Body(Compile("lane-mask-join", words));
        Assert.Equal(1, Occurrences(body, "threadgroup_barrier"));
        Assert.Contains("sharpemu_lane < 32u ? (s2) : (s3)", body, StringComparison.Ordinal);

        // Without the overwrite every path reaches the join with the lane variable intact.
        words[3] = 0xBF800000; // s_nop
        var intact = Body(Compile("lane-mask-join-intact", words));
        Assert.Equal(0, Occurrences(intact, "threadgroup_barrier"));
        Assert.DoesNotContain("sharpemu_lane < 32u ? (s2)", intact, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(true, false, false)]  // lanes 60..63: only the upper simdgroup runs the store
    [InlineData(false, false, false)] // lanes 0..3: only the lower one
    [InlineData(true, true, true)]    // no lane passes: EXECZ skips the store
    public void TheExecMaskMatchesTheGuestAcrossBothSimdgroups(bool upper, bool none, bool skip)
    {
        var compare = upper ? VCmpGtU32To_S2 : VCmpLtU32To_S2;
        // v0 > 59, v0 < 4, v0 > 63
        var operands = none ? 0x00017F00u : upper ? 0x00017700u : 0x00010900u;
        var words = Program(compare, operands, skipOnExecz: skip, restore: true);
        var whole = Compile("lane-mask-exec", words, new Gen5MslCodegenOptions { HalfLocalExecBranches = false });
        Assert.Equal(skip ? 1 : 0, Occurrences(Body(whole), "threadgroup_barrier"));
        // Each half takes the EXECZ skip over the lane-local store on its own lanes, without the exchange.
        var half = Compile("lane-mask-exec-half", words);
        Assert.Equal(0, Occurrences(Body(half), "threadgroup_barrier"));
        if (!MetalNative.IsAvailable)
        {
            return;
        }

        Run(whole, upper, none);
        Run(half, upper, none);
    }

    private static void Run(Gen5MslShader shader, bool upper, bool none)
    {
        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var libraryError), libraryError + shader.Source);
        var input = new byte[16];
        BinaryPrimitives.WriteUInt32LittleEndian(input, 0xFFFFFFFFu);
        Assert.True(
            MetalNative.TryExecuteWithArgumentBuffer(
                library,
                shader.EntryPoint,
                input,
                new byte[Resources.PushData.ByteSize],
                shader.ArgumentLayout!,
                threadsPerThreadgroup: 64,
                out var result,
                out var runError),
            runError);
        var stored = BinaryPrimitives.ReadUInt32LittleEndian(result);
        if (none)
        {
            Assert.Equal(0xFFFFFFFFu, stored);
        }
        else if (upper)
        {
            Assert.InRange(stored, 60u, 63u);
        }
        else
        {
            Assert.InRange(stored, 0u, 3u);
        }
    }
}
