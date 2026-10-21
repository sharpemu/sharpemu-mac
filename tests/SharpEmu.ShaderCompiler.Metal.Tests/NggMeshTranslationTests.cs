// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>
/// Structural checks over the mesh emission of NGG primitive shaders. Pure text generation,
/// so these run everywhere; the Metal-compiler and device checks live in
/// <see cref="NggMeshRuntimeTests"/>.
/// </summary>
public sealed class NggMeshTranslationTests
{
    [Fact]
    public void PassthroughPrimitiveShaderKeepsThePerVertexPath()
    {
        var program = Gen5NggFixtures.Passthrough();
        Assert.False(ShaderCompileRequest.RequiresPrimitiveShaderWorkgroup(program));

        var shader = Gen5NggFixtures.CompileOrThrow(program);
        Assert.Equal(Gen5MslStage.Vertex, shader.Stage);
        Assert.Equal("gen5_vs", shader.EntryPoint);
        Assert.Null(shader.MeshInfo);
    }

    [Fact]
    public void PrimitiveShaderWithoutTheStageMaskKeepsThePerVertexPath()
    {
        // The same program without the draw's primitive-shader state must not change
        // backends: only a request that says the draw selected the NGG stage may.
        var request = Gen5ComputeFixtures.RequestOrThrow(
            Gen5NggFixtures.CullingTriangle(),
            ShaderStage.Vertex,
            userDataBase: Gen5NggFixtures.VertexUserDataBase);
        Assert.False(Gen5MslTranslator.RequiresMeshEmission(request));
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        Assert.Equal(Gen5MslStage.Vertex, shader.Stage);
    }

    [Fact]
    public void CullingPrimitiveShaderEmitsAMeshFunction()
    {
        var shader = Gen5NggFixtures.CompileOrThrow(Gen5NggFixtures.CullingTriangle());
        Assert.Equal(Gen5MslStage.Mesh, shader.Stage);
        Assert.Equal("gen5_ms", shader.EntryPoint);
        Assert.Contains("[[mesh, max_total_threads_per_threadgroup(64)]]", shader.Source, StringComparison.Ordinal);
        Assert.Contains("void gen5_ms(", shader.Source, StringComparison.Ordinal);
        Assert.Contains(
            "using Gen5MeshOutput = mesh<Gen5MeshVertex, void, 64, 1, topology::triangle>;",
            shader.Source,
            StringComparison.Ordinal);
    }

    [Fact]
    public void MeshInterfaceDescribesTheSubgroupLaunch()
    {
        var shader = Gen5NggFixtures.CompileOrThrow(Gen5NggFixtures.CullingTriangle());
        var mesh = Assert.IsType<Gen5MslMeshInfo>(shader.MeshInfo);
        Assert.Equal(64u, mesh.ThreadsPerThreadgroup);
        Assert.Equal(2u, mesh.WavesPerThreadgroup);
        Assert.Equal(Gen5NggFixtures.VertexCount, mesh.MaxVertices);
        Assert.Equal(Gen5NggFixtures.PrimitiveCount, mesh.MaxPrimitives);
        Assert.Equal(3u, mesh.VerticesPerPrimitive);
        Assert.Equal(Gen5MslMeshTopology.Triangle, mesh.Topology);
        Assert.Equal(2u, mesh.DrawInfoBufferIndex);
        Assert.Equal(3u, mesh.IndexBufferIndex);

        // The threadgroup size the host launches with has to match the interface.
        Assert.Equal(mesh.ThreadsPerThreadgroup, shader.ThreadgroupSizeX);
    }

    [Fact]
    public void MeshVertexPayloadMatchesTheVertexStageInterface()
    {
        // The pixel program is translated against user(locn*) params, so a mesh pipeline
        // can only reuse it when the mesh vertex struct declares the same locations.
        var mesh = Gen5NggFixtures.CompileOrThrow(Gen5NggFixtures.CullingTriangle(), requiredVertexOutputCount: 3);
        var vertex = Gen5NggFixtures.CompileOrThrow(Gen5NggFixtures.Passthrough(), requiredVertexOutputCount: 3);
        for (var location = 0; location < 3; location++)
        {
            var field = $"float4 param{location} [[user(locn{location})]];";
            Assert.Contains(field, mesh.Source, StringComparison.Ordinal);
            Assert.Contains(field, vertex.Source, StringComparison.Ordinal);
        }

        Assert.Contains("float4 sharpemu_position [[position, invariant]];", mesh.Source, StringComparison.Ordinal);
        Assert.Equal(3u, mesh.AttributeCount);
    }

    [Fact]
    public void LocalDataShareBecomesThreadgroupMemory()
    {
        var shader = Gen5NggFixtures.CompileOrThrow(Gen5NggFixtures.CullingTriangle());

        // Not the per-invocation scratch array the vertex path uses.
        Assert.Contains("threadgroup uint sharpemu_lds[", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("thread uint sharpemu_lds[", shader.Source, StringComparison.Ordinal);
        var mesh = Assert.IsType<Gen5MslMeshInfo>(shader.MeshInfo);
        Assert.Contains($"threadgroup uint sharpemu_lds[{mesh.ThreadgroupMemoryBytes / 4}]", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void BarrierParksTheWaveInsteadOfBarrieringInPlace()
    {
        var request = Gen5NggFixtures.Request(Gen5NggFixtures.CullingTriangle());
        Assert.True(
            Gen5MslTranslator.TryCompileProgram(
                request, out var shader, out var error, new Gen5MslCodegenOptions(StructuredControlFlow: false)),
            error);

        // S_BARRIER must end its block and park the wave; the dispatcher's loop body is
        // the only place a threadgroup_barrier may appear, because two waves of the same
        // subgroup can sit at different pc values.
        Assert.Contains("sharpemu_mesh_wait = true;", shader.Source, StringComparison.Ordinal);
        Assert.Contains("while (active && !sharpemu_mesh_wait)", shader.Source, StringComparison.Ordinal);
        Assert.Contains("case 1u:", shader.Source, StringComparison.Ordinal);
        Assert.Contains("atomic_fetch_add_explicit(&sharpemu_mesh_parked", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void AllocationRequestDrivesThePrimitiveCount()
    {
        var shader = Gen5NggFixtures.CompileOrThrow(Gen5NggFixtures.CullingTriangle());

        // M0[9:0] is the vertex count and M0[22:12] the primitive count.
        Assert.Contains("sharpemu_mesh_alloc[0] = min(((s124 >> 0u) & 0x3FFu)", shader.Source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_mesh_alloc[1] = min(((s124 >> 12u) & 0x7FFu)", shader.Source, StringComparison.Ordinal);
        Assert.Contains("set_primitive_count(min(sharpemu_mesh_alloc[1]", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void PrimitiveExportDecodesTenBitSlots()
    {
        var shader = Gen5NggFixtures.CompileOrThrow(Gen5NggFixtures.CullingTriangle());
        Assert.Contains("sharpemu_mesh_prim[0] = (", shader.Source, StringComparison.Ordinal);
        Assert.Contains(">> 0u) & 0x1FFu;", shader.Source, StringComparison.Ordinal);
        Assert.Contains(">> 10u) & 0x1FFu;", shader.Source, StringComparison.Ordinal);
        Assert.Contains(">> 20u) & 0x1FFu;", shader.Source, StringComparison.Ordinal);
        Assert.Contains(">> 31u) & 1u) != 0u;", shader.Source, StringComparison.Ordinal);

        // A culled primitive becomes degenerate; Metal cannot drop one in place.
        Assert.Contains("sharpemu_mesh_culled ? uint3(0u) : sharpemu_mesh_prim", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void MergedWaveInfoSeedsTheGuestExecRebuild()
    {
        var shader = Gen5NggFixtures.CompileOrThrow(Gen5NggFixtures.CullingTriangle());
        Assert.Contains("s3 = sharpemu_mesh_wave_verts |", shader.Source, StringComparison.Ordinal);
        Assert.Contains("(sharpemu_mesh_wave_prims << 8u)", shader.Source, StringComparison.Ordinal);
        Assert.Contains("(2u << 28u)", shader.Source, StringComparison.Ordinal);
        Assert.Contains("s2 =", shader.Source, StringComparison.Ordinal);

        // The ES/VS vector inputs stay where the per-vertex path puts them.
        Assert.Contains("v5 = sharpemu_mesh_vertex_index;", shader.Source, StringComparison.Ordinal);
        Assert.Contains("v8 = sharpemu_mesh_instance_index;", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void SystemScalarsThatAreAlsoUserDataAreLeftAlone()
    {
        // A program whose user data starts at s0 has no system scalars to seed; writing
        // them would clobber a resource pointer.
        var request = Gen5ComputeFixtures.RequestOrThrow(
            Gen5NggFixtures.CullingTriangle(),
            ShaderStage.Vertex,
            primitiveShader: Gen5NggFixtures.StageInfo,
            userDataBase: 0);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        Assert.Equal(Gen5MslStage.Mesh, shader.Stage);
        Assert.DoesNotContain("s3 = sharpemu_mesh_wave_verts", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void ExcludedSystemScalarsAreSeededWhenTheUserDataStartsAtS0()
    {
        // A fused program's user data starts at s0 so the geometry user-data pointer can
        // occupy s[0:1]; s2..s7 are excluded from it and are still the launch state.
        var request = Gen5ComputeFixtures.RequestOrThrow(
            Gen5NggFixtures.CullingTriangle(),
            ShaderStage.Vertex,
            primitiveShader: Gen5NggFixtures.StageInfo,
            userDataBase: 0,
            excludedUserDataRegisters: 0xFC);
        Assert.Equal(0xFCUL, request.ExcludedUserDataRegisters);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        Assert.Equal(Gen5MslStage.Mesh, shader.Stage);
        Assert.Contains("s3 = sharpemu_mesh_wave_verts", shader.Source, StringComparison.Ordinal);
        Assert.Contains("s2 =", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void SubgroupBudgetIsClampedToWhatAMeshThreadgroupCanCarry()
    {
        // A title that programs more than the hardware's 256 output vertices, or leaves the
        // on-chip registers at zero, still has to produce a compilable pipeline.
        var huge = Gen5NggFixtures.StageInfo with
        {
            VerticesPerSubgroup = 4096,
            PrimitivesPerSubgroup = 4096,
        };
        var clamped = Gen5NggFixtures.CompileOrThrow(Gen5NggFixtures.CullingTriangle(), huge);
        var mesh = Assert.IsType<Gen5MslMeshInfo>(clamped.MeshInfo);
        Assert.Equal(256u, mesh.MaxVertices);
        Assert.Equal(256u, mesh.MaxPrimitives);
        Assert.Equal(256u, mesh.ThreadsPerThreadgroup);

        var empty = Gen5NggFixtures.StageInfo with
        {
            VerticesPerSubgroup = 0,
            PrimitivesPerSubgroup = 0,
            MaxOutputPerSubgroup = 0,
        };
        var fallback = Gen5NggFixtures.CompileOrThrow(Gen5NggFixtures.CullingTriangle(), empty);
        var fallbackMesh = Assert.IsType<Gen5MslMeshInfo>(fallback.MeshInfo);
        Assert.Equal(32u, fallbackMesh.ThreadsPerThreadgroup);
        Assert.Equal(1u, fallbackMesh.WavesPerThreadgroup);
    }

    [Fact]
    public void MeshStageKeepsItsOwnVertexFetchBinding()
    {
        // The ES part of a real primitive shader fetches vertices itself, through a V# its
        // user data carries. A mesh function has no [[stage_in]] to substitute that with, so
        // the fetch has to reach a real buffer binding.
        var runtimeInputs = new ResourceRuntimeInputs { UserData = Gen5NggFixtures.PositionUserData() };
        var request = Gen5NggFixtures.Request(
            Gen5NggFixtures.BufferFetchTriangle(), runtimeInputs: runtimeInputs);
        Assert.Empty(request.VertexInputs);
        var buffer = Assert.Single(request.Resources.Info.Buffers);
        Assert.Equal(77u, buffer.DescriptorFormat);

        var shader = Gen5NggFixtures.CompileOrThrow(
            Gen5NggFixtures.BufferFetchTriangle(), runtimeInputs: runtimeInputs);
        Assert.Equal(Gen5MslStage.Mesh, shader.Stage);

        // The fetch reads the bound buffer, not a vertex attribute.
        Assert.Contains("b0", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("[[stage_in]]", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("[[attribute(", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void EmbeddedVertexFetchIsRefusedForAWorkgroupPrimitiveShader()
    {
        // The substitution rewrites the program's own fetch into a host vertex attribute and
        // drops the scalar loads behind its V#. For the merged ES/GS workgroup stage that
        // leaves the fetch with nothing to read, so it must not be applied.
        var culling = Gen5NggFixtures.CullingTriangle();
        Assert.False(ShaderCompileRequest.AllowsEmbeddedVertexFetch(culling, Gen5NggFixtures.StageInfo));

        // Without the draw's primitive-shader state the program is an ordinary vertex
        // shader again, and the substitution is allowed.
        Assert.True(ShaderCompileRequest.AllowsEmbeddedVertexFetch(culling, null));

        // A passthrough primitive shader stays on the per-vertex path, so it keeps it too.
        Assert.True(ShaderCompileRequest.AllowsEmbeddedVertexFetch(
            Gen5NggFixtures.Passthrough(), Gen5NggFixtures.StageInfo));
    }

    [Fact]
    public void LineAndPointOutputPrimitivesChangeTheTopology()
    {
        var line = Gen5NggFixtures.CompileOrThrow(
            Gen5NggFixtures.CullingTriangle(),
            Gen5NggFixtures.StageInfo with { OutputPrimitiveType = Gen5NggExports.OutputPrimitiveLineStrip });
        Assert.Equal(Gen5MslMeshTopology.Line, Assert.IsType<Gen5MslMeshInfo>(line.MeshInfo).Topology);
        Assert.Contains("topology::line", line.Source, StringComparison.Ordinal);

        var point = Gen5NggFixtures.CompileOrThrow(
            Gen5NggFixtures.CullingTriangle(),
            Gen5NggFixtures.StageInfo with { OutputPrimitiveType = Gen5NggExports.OutputPrimitivePointList });
        Assert.Equal(Gen5MslMeshTopology.Point, Assert.IsType<Gen5MslMeshInfo>(point.MeshInfo).Topology);
        Assert.Contains("topology::point", point.Source, StringComparison.Ordinal);
    }
    [Fact]
    public void AGsHalfAssemblesItsInputPrimitivesFromTheDrawTopology()
    {
        // A fused program's GS half reads its input vertices back from LDS through the GS
        // vertex offsets: the guest takes bits 2-15 of each half as the ES vertex, so the
        // offsets carry the vertex index shifted by two, and the subgroup takes its share
        // of the draw's input primitives rather than a full budget.
        var shader = Gen5NggFixtures.CompileOrThrow(
            Gen5NggFixtures.CullingTriangle(),
            Gen5NggFixtures.StageInfo with { GeometryVerticesOut = 3 });
        var mesh = Assert.IsType<Gen5MslMeshInfo>(shader.MeshInfo);
        Assert.True(mesh.HasGeometryHalf);
        Assert.Contains("v0 = ((sharpemu_gs_v0 << 2u) & 0xFFFFu) |", shader.Source, StringComparison.Ordinal);
        Assert.Contains("(((sharpemu_gs_v1 << 2u) & 0xFFFFu) << 16u);", shader.Source, StringComparison.Ordinal);
        Assert.Contains("v1 = (sharpemu_gs_v2 << 2u) & 0xFFFFu;", shader.Source, StringComparison.Ordinal);
        Assert.Contains(
            "const uint sharpemu_mesh_vertex_slot = sharpemu_gs_vertex_start + sharpemu_tg_index;",
            shader.Source,
            StringComparison.Ordinal);
        Assert.Contains("sharpemu_draw.input_vertex_count", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("sharpemu_mesh_prim_base", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void APrimitiveShaderWithoutAGsHalfKeepsTheListOffsets()
    {
        var shader = Gen5NggFixtures.CompileOrThrow(Gen5NggFixtures.CullingTriangle());
        Assert.False(Assert.IsType<Gen5MslMeshInfo>(shader.MeshInfo).HasGeometryHalf);
        Assert.Contains("const uint sharpemu_mesh_prim_base = sharpemu_tg_index * 3u;", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("sharpemu_gs_v0", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void AProgramWithoutAGsHalfReadsCachedMeshletsWhenTheDrawAsksForThem()
    {
        var shader = Gen5NggFixtures.CompileOrThrow(Gen5NggFixtures.CullingTriangle());
        Assert.Contains("uint meshlet_count;", shader.Source, StringComparison.Ordinal);
        Assert.Contains("device const uint* sharpemu_meshlets [[buffer(4)]]", shader.Source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_meshlets_on = sharpemu_draw.meshlet_count != 0u;", shader.Source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_mesh_alloc[0] = min(sharpemu_ml.w & 0xFFFFu,", shader.Source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_meshlets[sharpemu_draw.meshlet_vertex_base + sharpemu_ml.x + sharpemu_tg_index]", shader.Source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_meshlets_on ? sharpemu_ml.z + sharpemu_tg_index", shader.Source, StringComparison.Ordinal);
        Assert.Equal(80u, Gen5MslMeshInfo.DrawInfoByteSize);

        // A GS half assembles its own input primitives and takes no meshlets.
        var fused = Gen5NggFixtures.CompileOrThrow(
            Gen5NggFixtures.CullingTriangle(), Gen5NggFixtures.StageInfo with { GeometryVerticesOut = 3 });
        Assert.DoesNotContain("sharpemu_meshlets", fused.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void ALocalPrimitiveSeedsItsVerticesInTheFieldsTheGuestDecodes()
    {
        // The guest reads bfe(v0, 2, 14), v0 >> 18 and bfe(v1, 2, 14): the local vertex index
        // sits in bits 2-15 of each 16-bit field, exactly as the GS half packs it.
        var shader = Gen5NggFixtures.CompileOrThrow(Gen5NggFixtures.CullingTriangle());
        Assert.Contains(
            ": (((sharpemu_mesh_prim_base << 2u) & 0xFFFFu) |", shader.Source, StringComparison.Ordinal);
        Assert.Contains(
            "((((sharpemu_mesh_prim_base + 1u) << 2u) & 0xFFFFu) << 16u));", shader.Source, StringComparison.Ordinal);
        Assert.Contains(
            ": (((sharpemu_mesh_prim_base + 2u) << 2u) & 0xFFFFu);", shader.Source, StringComparison.Ordinal);

        // A cached meshlet's triangle (three 8-bit local indices) is seeded in the same fields.
        Assert.Contains(
            "(((sharpemu_mesh_prim_packed & 0xFFu) << 2u) | (((sharpemu_mesh_prim_packed >> 8u) & 0xFFu) << 18u))",
            shader.Source, StringComparison.Ordinal);
        Assert.Contains(
            "(((sharpemu_mesh_prim_packed >> 16u) & 0xFFu) << 2u)", shader.Source, StringComparison.Ordinal);

        foreach (var lane in new[] { 0u, 1u, 9u, 10u, 31u })
        {
            var @base = lane * 3;
            var v0 = ((@base << 2) & 0xFFFFu) | ((((@base + 1) << 2) & 0xFFFFu) << 16);
            var v1 = ((@base + 2) << 2) & 0xFFFFu;
            Assert.Equal(@base, (v0 >> 2) & 0x3FFFu);
            Assert.Equal(@base + 1, v0 >> 18);
            Assert.Equal(@base + 2, (v1 >> 2) & 0x3FFFu);
        }
    }

    [Theory]
    [InlineData(64u, 64u, 3u, 21u, 63u)]
    [InlineData(64u, 10u, 3u, 10u, 30u)]
    [InlineData(30u, 64u, 3u, 10u, 30u)]
    [InlineData(64u, 64u, 2u, 32u, 64u)]
    [InlineData(64u, 64u, 1u, 64u, 64u)]
    [InlineData(2u, 64u, 3u, 1u, 3u)]
    public void ASubgroupWithoutVertexReuseHoldsWholePrimitives(
        uint maxVertices, uint maxPrimitives, uint perPrimitive, uint primitives, uint vertices)
    {
        var mesh = new Gen5MslMeshInfo(
            ThreadsPerThreadgroup: 64, WavesPerThreadgroup: 2, MaxVertices: maxVertices, MaxPrimitives: maxPrimitives,
            VerticesPerPrimitive: perPrimitive, Topology: Gen5MslMeshTopology.Triangle, ThreadgroupMemoryBytes: 4096,
            DrawInfoBufferIndex: 2, IndexBufferIndex: 3, ReadsIndexBuffer: true);
        Assert.Equal(primitives, mesh.PrimitivesPerSubgroup);
        Assert.Equal(vertices, mesh.VerticesPerSubgroup);

        // Every subgroup starts on a primitive boundary: its first vertex slot is a multiple
        // of the primitive size, so a triangle is never split across two subgroups.
        for (var subgroup = 0u; subgroup < 8; subgroup++)
        {
            Assert.Equal(0u, subgroup * mesh.VerticesPerSubgroup % perPrimitive);
            Assert.Equal(subgroup * mesh.PrimitivesPerSubgroup * perPrimitive, subgroup * mesh.VerticesPerSubgroup);
        }
    }

    [Fact]
    public void AWave64PrimitiveShaderRunsAsOneNarrowWave()
    {
        var shader = Gen5NggFixtures.CompileOrThrow(
            Gen5NggFixtures.Wave64Triangles(), Gen5NggFixtures.Wave64StageInfo);
        var mesh = Assert.IsType<Gen5MslMeshInfo>(shader.MeshInfo);
        Assert.True(mesh.NarrowWave64);
        Assert.Equal(32u, mesh.ThreadsPerThreadgroup);
        Assert.Equal(1u, mesh.WavesPerThreadgroup);
        // One lane per vertex and one per primitive: the pipeline declares the whole wave of
        // each, and the whole-triangle path takes 10 triangles / 30 vertices of it per draw.
        Assert.Equal(32u, mesh.MaxVertices);
        Assert.Equal(32u, mesh.MaxPrimitives);
        Assert.Equal(10u, mesh.PrimitivesPerSubgroup);
        Assert.Equal(30u, mesh.VerticesPerSubgroup);
        Assert.Contains("mesh<Gen5MeshVertex, void, 32, 32, topology::triangle>", shader.Source, StringComparison.Ordinal);

        // The export gating follows the allocation, not the declared extent.
        Assert.Contains("if (sharpemu_tg_index < sharpemu_mesh_out_verts)", shader.Source, StringComparison.Ordinal);
        Assert.Contains("if (sharpemu_tg_index < sharpemu_mesh_out_prims)", shader.Source, StringComparison.Ordinal);
        Assert.Contains("max_total_threads_per_threadgroup(32)", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void AWave32OrFusedPrimitiveShaderKeepsItsLaunch()
    {
        var wave32 = Gen5NggFixtures.CompileOrThrow(Gen5NggFixtures.Wave64Triangles());
        Assert.False(Assert.IsType<Gen5MslMeshInfo>(wave32.MeshInfo).NarrowWave64);

        var fused = Gen5NggFixtures.CompileOrThrow(
            Gen5NggFixtures.Wave64Triangles(), Gen5NggFixtures.Wave64StageInfo with { GeometryVerticesOut = 3 });
        Assert.False(Assert.IsType<Gen5MslMeshInfo>(fused.MeshInfo).NarrowWave64);
    }

    [Fact]
    public void NarrowWave64LowersTheUpperHalfAsInactive()
    {
        var shader = Gen5NggFixtures.CompileOrThrow(
            Gen5NggFixtures.Wave64LaneProbe(), Gen5NggFixtures.Wave64StageInfo);
        Assert.True(Assert.IsType<Gen5MslMeshInfo>(shader.MeshInfo).NarrowWave64);

        // v_readlane of lane 63 reads an inactive lane: zero, not lane 31.
        Assert.Contains(" >= 32u ? 0u : simd_shuffle(", shader.Source, StringComparison.Ordinal);
        Assert.Contains("& 63u", shader.Source, StringComparison.Ordinal);

        // v_mbcnt_hi has no lanes to count: the addend passes through.
        Assert.DoesNotContain("popcount", shader.Source, StringComparison.Ordinal);

        // An all-ones EXEC_HI write leaves the upper half empty.
        Assert.Matches(@"uint (t\d+) = 0u;\s+s127 = \1;", shader.Source);
    }

    [Fact]
    public void ALargeLocalDataShareLivesInASlotOfTheDevicePool()
    {
        var shader = Gen5NggFixtures.CompileOrThrow(
            Gen5NggFixtures.CullingTriangle(),
            Gen5NggFixtures.StageInfo with { LocalDataShareBytes = 15 * 1024 });
        var mesh = Assert.IsType<Gen5MslMeshInfo>(shader.MeshInfo);
        Assert.True(mesh.DeviceLds);
        Assert.Equal(0u, mesh.ThreadgroupMemoryBytes);
        Assert.DoesNotContain("threadgroup uint sharpemu_lds[", shader.Source, StringComparison.Ordinal);
        Assert.Contains("threadgroup uint sharpemu_lds_slot;", shader.Source, StringComparison.Ordinal);
        Assert.Contains("device atomic_uint* sharpemu_lds_pool [[buffer(5)]]", shader.Source, StringComparison.Ordinal);
        Assert.Contains(
            "device uint* sharpemu_lds = (device uint*)sharpemu_lds_pool + 4096u + sharpemu_lds_slot * 4096u;",
            shader.Source,
            StringComparison.Ordinal);
        Assert.Contains("atomic_compare_exchange_weak_explicit(&sharpemu_lds_pool[1u + sharpemu_slot]", shader.Source, StringComparison.Ordinal);
        Assert.Contains("atomic_store_explicit(&sharpemu_lds_pool[1u + sharpemu_lds_slot], 0u", shader.Source, StringComparison.Ordinal);

        // Addresses stay inside the slot by mask alone.

        // Every workgroup barrier orders device memory too, where the data now lives.
        Assert.DoesNotContain("threadgroup_barrier(mem_flags::mem_threadgroup);", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void ALocalDataShareUnderTheThresholdStaysInThreadgroupMemory()
    {
        var shader = Gen5NggFixtures.CompileOrThrow(
            Gen5NggFixtures.CullingTriangle(),
            Gen5NggFixtures.StageInfo with { LocalDataShareBytes = 12 * 1024 });
        Assert.False(Assert.IsType<Gen5MslMeshInfo>(shader.MeshInfo).DeviceLds);
        Assert.DoesNotContain("sharpemu_lds_pool", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void LocalDataShareIsDeclaredAtTheGuestSize()
    {
        // 9 KB (the shape of a culling template's allocation) is 2304 dwords, not 4096.
        var shader = Gen5NggFixtures.CompileOrThrow(
            Gen5NggFixtures.CullingTriangle(),
            Gen5NggFixtures.StageInfo with { LocalDataShareBytes = 9216 });
        Assert.Contains("threadgroup uint sharpemu_lds[2304];", shader.Source, StringComparison.Ordinal);
        Assert.Equal(9216u, Assert.IsType<Gen5MslMeshInfo>(shader.MeshInfo).ThreadgroupMemoryBytes);

        // Addresses mask to the covering power of two and clamp to the last declared dword.
        Assert.Contains("& 4095u), 2303u)", shader.Source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(Gen5MslMeshInputTopology.TriangleList, 3u, 10u)]
    [InlineData(Gen5MslMeshInputTopology.TriangleStrip, 3u, 10u)]
    [InlineData(Gen5MslMeshInputTopology.PointList, 3u, 10u)]
    [InlineData(Gen5MslMeshInputTopology.TriangleStrip, 4u, 8u)]
    [InlineData(Gen5MslMeshInputTopology.TriangleStrip, 1u, 30u)]
    public void AGsSubgroupTakesWhatItsInputsAndOutputsFit(
        Gen5MslMeshInputTopology input, uint verticesOut, uint expected)
    {
        var mesh = new Gen5MslMeshInfo(
            ThreadsPerThreadgroup: 64, WavesPerThreadgroup: 2, MaxVertices: 64, MaxPrimitives: 64,
            VerticesPerPrimitive: 3, Topology: Gen5MslMeshTopology.Triangle, ThreadgroupMemoryBytes: 4096,
            DrawInfoBufferIndex: 2, IndexBufferIndex: 3, ReadsIndexBuffer: true, GeometryVerticesOut: verticesOut);
        Assert.Equal(expected, mesh.InputPrimitivesPerSubgroup(input));
    }

    [Theory]
    [InlineData(Gen5MslMeshInputTopology.TriangleStrip, 4u, 2u)]
    [InlineData(Gen5MslMeshInputTopology.TriangleStrip, 2u, 0u)]
    [InlineData(Gen5MslMeshInputTopology.TriangleList, 6u, 2u)]
    [InlineData(Gen5MslMeshInputTopology.PointList, 5u, 5u)]
    public void AGsDrawCountsItsInputPrimitivesByTopology(Gen5MslMeshInputTopology input, uint count, uint expected) =>
        Assert.Equal(expected, Gen5MslMeshInfo.InputPrimitiveCount(input, count));
}
