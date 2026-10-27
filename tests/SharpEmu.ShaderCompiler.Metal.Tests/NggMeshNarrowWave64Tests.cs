// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Xunit;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>Structural checks over the narrow-wave64 lowering of mesh programs.</summary>
public sealed class NggMeshNarrowWave64Tests
{
    private static Gen5MslShader Compile(
        Gen5ShaderProgram program, Gen5NggStageInfo? stage, bool structured, bool meshExitJoin = true)
    {
        var request = Gen5NggFixtures.Request(program, stage);
        Assert.True(
            Gen5MslTranslator.TryCompileProgram(
                request, out var shader, out var error,
                new Gen5MslCodegenOptions(StructuredControlFlow: structured) { MeshExitJoin = meshExitJoin }),
            error);
        return shader;
    }

    [Fact]
    public void ATopLevelBarrierIsEmittedInPlaceWithoutTheRendezvous()
    {
        // Two waves, one barrier outside any branch: every thread runs it exactly once.
        var shader = Compile(Gen5NggFixtures.CullingTriangle(), null, structured: true);
        Assert.Equal(2u, Assert.IsType<Gen5MslMeshInfo>(shader.MeshInfo).WavesPerThreadgroup);
        Assert.DoesNotContain("while (active && !sharpemu_mesh_wait)", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("sharpemu_mesh_wait = true", shader.Source, StringComparison.Ordinal);
        Assert.Contains("threadgroup_barrier(mem_flags::mem_threadgroup);", shader.Source, StringComparison.Ordinal);

        var dispatched = Compile(Gen5NggFixtures.CullingTriangle(), null, structured: false);
        Assert.Contains("while (active && !sharpemu_mesh_wait)", dispatched.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void ABarrierInsideABranchKeepsTheDispatcherWhenThreadsSpanWaves()
    {
        var shader = Compile(Gen5NggFixtures.BarrierInDivergentBranch(), null, structured: true);
        Assert.Equal(2u, Assert.IsType<Gen5MslMeshInfo>(shader.MeshInfo).WavesPerThreadgroup);
        Assert.Contains("while (active && !sharpemu_mesh_wait)", shader.Source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_mesh_wait = true;", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void ABarrierInsideABranchIsEmittedInPlaceInAOneWaveGroup()
    {
        // One wave: the pc is uniform, so a barrier under a scalar branch is reached by all.
        var stage = Gen5NggFixtures.StageInfo with
        {
            VerticesPerSubgroup = 32,
            PrimitivesPerSubgroup = 1,
            MaxOutputPerSubgroup = 32,
        };
        var shader = Compile(Gen5NggFixtures.BarrierInDivergentBranch(), stage, structured: true);
        Assert.Equal(1u, Assert.IsType<Gen5MslMeshInfo>(shader.MeshInfo).WavesPerThreadgroup);
        Assert.DoesNotContain("while (active && !sharpemu_mesh_wait)", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void AControlFlowTheStructuredFormCannotNestTakesTheLinearForm()
    {
        var stage = Gen5NggFixtures.StageInfo with
        {
            VerticesPerSubgroup = 32,
            PrimitivesPerSubgroup = 1,
            MaxOutputPerSubgroup = 32,
        };
        var shader = Compile(Gen5NggFixtures.BarrierLoopWithTwoExits(), stage, structured: true, meshExitJoin: false);
        Assert.DoesNotContain("while (active && !sharpemu_mesh_wait)", shader.Source, StringComparison.Ordinal);
        Assert.Contains("while (active && pc >= 1u && pc <= 3u)", shader.Source, StringComparison.Ordinal);
        Assert.Contains("if (active && pc == 2u)", shader.Source, StringComparison.Ordinal);
        Assert.Contains("threadgroup_barrier(mem_flags::mem_threadgroup);", shader.Source, StringComparison.Ordinal);

        // Joining the exits now lets this one-wave loop use structured control flow;
        // the kill switch above still exercises the linear fallback itself.
        var joined = Compile(Gen5NggFixtures.BarrierLoopWithTwoExits(), stage, structured: true);
        Assert.DoesNotContain("pc ==", joined.Source, StringComparison.Ordinal);
        Assert.Contains("while (true)", joined.Source, StringComparison.Ordinal);

        // Several waves around a barrier inside a loop: back to the dispatcher.
        var wide = Compile(Gen5NggFixtures.BarrierLoopWithTwoExits(), null, structured: true);
        Assert.Contains("while (active && !sharpemu_mesh_wait)", wide.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void ALinearBarrierOfSeveralWavesStandsOutsideItsPcGuard()
    {
        // A wave that runs out of steps clears active; a barrier under that guard would be
        // skipped by it while the other waves wait. So it follows the guarded block.
        var shader = Compile(Gen5NggFixtures.BarrierAfterLoopWithTwoExits(), null, structured: true, meshExitJoin: false);
        Assert.Equal(2u, Assert.IsType<Gen5MslMeshInfo>(shader.MeshInfo).WavesPerThreadgroup);
        Assert.Contains("while (active && pc >= ", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("while (active && !sharpemu_mesh_wait)", shader.Source, StringComparison.Ordinal);

        var lines = shader.Source.Split('\n');
        var body = Array.FindIndex(lines, static line => line.Contains("if (active && pc == 0u)", StringComparison.Ordinal));
        var barrier = Array.FindIndex(lines, body, static line => line.Trim() == "threadgroup_barrier(mem_flags::mem_threadgroup);");
        Assert.True(barrier > 0, "no in-place barrier was emitted");
        Assert.Equal("}", lines[barrier - 1].Trim());
        Assert.Equal(lines[barrier - 1].Length - lines[barrier - 1].TrimStart().Length,
            lines[barrier].Length - lines[barrier].TrimStart().Length);

        // The same shape with one wave keeps it inside the guard.
        var stage = Gen5NggFixtures.StageInfo with
        {
            VerticesPerSubgroup = 32,
            PrimitivesPerSubgroup = 1,
            MaxOutputPerSubgroup = 32,
        };
        var oneWave = Compile(Gen5NggFixtures.BarrierAfterLoopWithTwoExits(), stage, structured: true, meshExitJoin: false);
        var oneLines = oneWave.Source.Split('\n');
        var oneBody = Array.FindIndex(oneLines, static line => line.Contains("if (active && pc == 0u)", StringComparison.Ordinal));
        var oneBarrier = Array.FindIndex(oneLines, oneBody, static line => line.Trim() == "threadgroup_barrier(mem_flags::mem_threadgroup);");
        Assert.True(oneBarrier > 0);
        Assert.NotEqual("}", oneLines[oneBarrier - 1].Trim());
    }

    [Fact]
    public void CompareIntoAScalarPairClearsTheHighDword()
    {
        // A wave64 v_cmp into s[0:1] defines both dwords; the upper lanes do not exist, so
        // s1 is 0 rather than whatever the guest left there, and s_bcnt1_i32_b64 counts
        // only the real survivors.
        var shader = Gen5NggFixtures.CompileOrThrow(
            Gen5NggFixtures.Wave64MaskPairProbe(), Gen5NggFixtures.Wave64StageInfo);
        Assert.True(Assert.IsType<Gen5MslMeshInfo>(shader.MeshInfo).NarrowWave64);
        Assert.Matches(@"s0 = sharpemu_ballot\([^;]*\);\s+s1 = 0u;", shader.Source);

        // The same program as wave32 keeps its one-dword mask semantics.
        var wave32 = Gen5NggFixtures.CompileOrThrow(Gen5NggFixtures.Wave64MaskPairProbe());
        Assert.DoesNotMatch(@"s0 = sharpemu_ballot\([^;]*\);\s+s1 = 0u;", wave32.Source);
    }
}
