// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>
/// Tier 2 and 3 for the NGG mesh path: the emitted MSL must compile with the OS Metal
/// compiler, and a synthetic primitive shader must rasterize through a real mesh render
/// pipeline. These no-op (with a note) on hosts without a Metal device, like the compute
/// runtime tests, and the Void capture case additionally no-ops when the preserved capture
/// is not on this machine — no guest shader bytes live in the repository.
/// </summary>
public sealed class NggMeshRuntimeTests(ITestOutputHelper output)
{
    private const uint TargetSize = 32;

    // The preserved rejected-draw capture, under the checkout's untracked artifacts directory.
    private static readonly string VoidCaptureDirectory = Path.Combine(
        FindCheckoutRoot(), "artifacts", "validation", "void-skipped-shader-400e5c64", "shader-capture");

    private static string FindCheckoutRoot()
    {
        for (var directory = new DirectoryInfo(AppContext.BaseDirectory); directory is not null; directory = directory.Parent)
        {
            if (File.Exists(Path.Combine(directory.FullName, "SharpEmu.slnx")))
            {
                return directory.FullName;
            }
        }

        return AppContext.BaseDirectory;
    }

    [Fact]
    public void SyntheticPrimitiveShadersCompileWithTheRuntimeMetalCompiler()
    {
        if (SkipWithoutMetalDevice("mesh compile validation")) return;

        foreach (var (name, shader) in SyntheticShaders())
        {
            Assert.True(
                MetalNative.TryCompileLibrary(shader.Source, out _, out var error),
                $"[{name}] Metal rejected the emitted mesh MSL: {error}\n{shader.Source}");
        }
    }

    [Fact]
    public void SyntheticPrimitiveShaderRasterizesThroughAMeshPipeline()
    {
        if (SkipWithoutMetalDevice("the mesh execution test")) return;

        var pixels = RenderOrThrow(Gen5NggFixtures.CullingTriangle());

        // The subgroup allocated one triangle over vertices 0/1/2, whose positions the
        // program only has because the LDS round trip and the barrier both worked:
        // (-0.8,-0.8), (0,0.8), (0.8,-0.8). The center is inside it.
        Assert.Equal(1.0f, Green(pixels, TargetSize / 2, TargetSize / 2), 3);

        // param0.x carries the exchanged lane value, so the center interpolates the three
        // exported values 0/1/2 at barycentrics (0.25, 0.5, 0.25) — exactly 1.
        Assert.Equal(1.0f, Red(pixels, TargetSize / 2, TargetSize / 2), 1);

        // Outside the triangle the clear survives, so the primitive count really was one
        // and the remaining 63 primitive slots stayed degenerate.
        Assert.Equal(0.0f, Green(pixels, 1, 1), 3);
        Assert.Equal(0.0f, Green(pixels, TargetSize - 2, 1), 3);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void APassthroughProgramThatHandsVertexValuesIsAMeshFunction(bool reciprocalW)
    {
        var plain = Gen5NggFixtures.CompileOrThrow(Gen5NggFixtures.Passthrough());
        Assert.Equal(Gen5MslStage.Vertex, plain.Stage);

        var shader = Gen5NggFixtures.CompileOrThrow(Gen5NggFixtures.Passthrough(), meshPerVertexParameterMask: 1,
            clipSpace: new ShaderClipSpaceTransform { WReciprocal = reciprocalW });
        Assert.Equal(Gen5MslStage.Mesh, shader.Stage);
        Assert.Contains("float3 sharpemu_vertex_w [[user(sharpemu_vertex_w)]];", shader.Source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_mesh_vertex_w[sharpemu_tg_index] = " +
            (reciprocalW ? "1.0f / " : string.Empty) + "sharpemu_out.sharpemu_position.w;", shader.Source, StringComparison.Ordinal);
        for (var vertex = 0; vertex < 3; vertex++)
        {
            Assert.Contains($"float4 param0_v{vertex} [[user(locn0_v{vertex})]];", shader.Source, StringComparison.Ordinal);
            Assert.Contains($"primitive.param0_v{vertex} = sharpemu_mesh_param0[min(indices[{vertex}]", shader.Source,
                StringComparison.Ordinal);
        }

        if (SkipWithoutMetalDevice("the per-vertex mesh compile")) return;
        Assert.True(
            MetalNative.TryCompileLibrary(shader.Source, out _, out var error),
            $"Metal rejected the emitted mesh MSL: {error}\n{shader.Source}");
    }

    [Fact]
    public void HandedVertexValuesShrinkTheMeshToTheOutputBudget()
    {
        // 256 vertices of eight params and 256 primitives of three handed params would
        // declare 112 KB of mesh output; Apple GPUs refuse anything above 32 KB.
        var stageInfo = Gen5NggFixtures.StageInfo with
        {
            VerticesPerSubgroup = 256,
            PrimitivesPerSubgroup = 256,
            MaxOutputPerSubgroup = 256,
            OutputPrimitiveType = Gen5NggExports.OutputPrimitiveTriangleStrip,
        };
        var shader = Gen5NggFixtures.CompileOrThrow(
            Gen5NggFixtures.Passthrough(), stageInfo, requiredVertexOutputCount: 8, meshPerVertexParameterMask: 0b111);
        var mesh = shader.MeshInfo!;
        const uint vertexBytes = 16 + (8 * 16) + 48;
        const uint primitiveBytes = (3 * 48) + 16 + 16;
        Assert.True((mesh.MaxVertices * vertexBytes) + (mesh.MaxPrimitives * primitiveBytes) <= 32768,
            $"max_v={mesh.MaxVertices} max_p={mesh.MaxPrimitives}");
        Assert.True(mesh.MaxVertices >= 3);
        Assert.Equal(mesh.ThreadsPerThreadgroup, ((Math.Max(mesh.MaxVertices, mesh.MaxPrimitives) + 31) / 32) * 32);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void HandedVertexValuesReachTheFragmentInPrimitiveOrder(bool passthrough)
    {
        if (SkipWithoutMetalDevice("the per-vertex mesh execution test")) return;

        // The triangle's vertices 0/1/2 export param0.x = 0/1/2. Each fragment must see
        // exactly those three values in the primitive's vertex order, and the barycentric
        // components must weight them in the same order: their dot product is the value
        // Metal interpolates for param0 itself. The passthrough program exports the
        // primitive the prologue packed into v0.
        var shader = passthrough
            ? Gen5NggFixtures.CompileOrThrow(
                Gen5NggFixtures.PassthroughTriangle(), Gen5NggFixtures.PassthroughStageInfo, meshPerVertexParameterMask: 1)
            : Gen5NggFixtures.CompileOrThrow(Gen5NggFixtures.CullingTriangle(), meshPerVertexParameterMask: 1);
        Assert.Equal(Gen5MslStage.Mesh, shader.Stage);
        var mesh = shader.MeshInfo!;
        var source = shader.Source + PerVertexFragmentSource;
        if (!MetalNative.TryCompileLibrary(source, out var library, out var compileError))
        {
            throw new InvalidOperationException($"Metal rejected the mesh MSL: {compileError}\n{source}");
        }

        Assert.True(
            MetalNative.TryRenderMesh(
                library,
                shader.EntryPoint,
                "gen5_mesh_per_vertex_fs",
                [
                    (BuildArgumentBuffer(shader), 0u),
                    (new byte[64], 1u),
                    (BuildDrawInfo(mesh), mesh.DrawInfoBufferIndex),
                    (new byte[16], mesh.IndexBufferIndex),
                ],
                threadgroups: 1,
                mesh.ThreadsPerThreadgroup,
                TargetSize,
                TargetSize,
                out var pixels,
                out var renderError),
            $"the mesh draw failed: {renderError}");

        var center = (((TargetSize / 2) * TargetSize) + (TargetSize / 2)) * 4;
        Assert.Equal(0.0f, pixels[center]);
        Assert.Equal(1.0f, pixels[center + 1]);
        Assert.Equal(2.0f, pixels[center + 2]);
        Assert.Equal(0.0f, pixels[center + 3], 4);

        // Outside the triangle the clear survives.
        Assert.Equal(0.0f, Green(pixels, 1, 1), 3);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DeviceLdsPrimitiveShaderExchangesThroughTheSlotAndBarrier(bool nullPrimitive)
    {
        if (SkipWithoutMetalDevice("the device LDS mesh execution test")) return;

        // The same program as the threadgroup-memory case, with its LDS in the device pool:
        // the exchange across both waves only survives if the slot is acquired before the
        // first access and every barrier orders device memory.
        var shader = Gen5NggFixtures.CompileOrThrow(
            Gen5NggFixtures.CullingTriangle(nullPrimitive),
            Gen5NggFixtures.StageInfo with { LocalDataShareBytes = 15 * 1024 });
        var mesh = shader.MeshInfo!;
        Assert.True(mesh.DeviceLds);
        var source = shader.Source + FragmentSource;
        if (!MetalNative.TryCompileLibrary(source, out var library, out var compileError))
        {
            throw new InvalidOperationException($"Metal rejected the mesh MSL: {compileError}\n{source}");
        }

        Assert.True(
            MetalNative.TryRenderMesh(
                library,
                shader.EntryPoint,
                "gen5_mesh_test_fs",
                [
                    (BuildArgumentBuffer(shader), 0u),
                    (new byte[64], 1u),
                    (BuildDrawInfo(mesh), mesh.DrawInfoBufferIndex),
                    (new byte[16], mesh.IndexBufferIndex),
                    (new byte[Gen5MslMeshInfo.LdsPoolByteSize], Gen5MslMeshInfo.LdsPoolBufferIndex),
                ],
                threadgroups: 4,
                mesh.ThreadsPerThreadgroup,
                TargetSize,
                TargetSize,
                out var pixels,
                out var renderError),
            $"the mesh draw failed: {renderError}");

        Assert.Equal(nullPrimitive ? 0.0f : 1.0f, Green(pixels, TargetSize / 2, TargetSize / 2), 3);
        Assert.Equal(0.0f, Green(pixels, 1, 1), 3);
    }

    [Fact]
    public void ABarrierEmittedInPlaceDrawsTheSamePixelsAsTheDispatcher()
    {
        if (SkipWithoutMetalDevice("the barrier form comparison")) return;

        float[] Render(bool structured)
        {
            var request = Gen5NggFixtures.Request(Gen5NggFixtures.CullingTriangle());
            Assert.True(
                Gen5MslTranslator.TryCompileProgram(
                    request, out var shader, out var error, new Gen5MslCodegenOptions(StructuredControlFlow: structured)),
                error);
            var mesh = shader.MeshInfo!;
            Assert.Equal(structured, !shader.Source.Contains("while (active && !sharpemu_mesh_wait)", StringComparison.Ordinal));
            var source = shader.Source + FragmentSource;
            if (!MetalNative.TryCompileLibrary(source, out var library, out var compileError))
            {
                throw new InvalidOperationException($"Metal rejected the mesh MSL: {compileError}\n{source}");
            }

            Assert.True(
                MetalNative.TryRenderMesh(
                    library,
                    shader.EntryPoint,
                    "gen5_mesh_test_fs",
                    [
                        (BuildArgumentBuffer(shader), 0u),
                        (new byte[64], 1u),
                        (BuildDrawInfo(mesh), mesh.DrawInfoBufferIndex),
                        (new byte[16], mesh.IndexBufferIndex),
                    ],
                    threadgroups: 1,
                    mesh.ThreadsPerThreadgroup,
                    TargetSize,
                    TargetSize,
                    out var pixels,
                    out var renderError),
                $"the mesh draw failed: {renderError}");
            return pixels;
        }

        var dispatched = Render(structured: false);
        var inPlace = Render(structured: true);
        Assert.Equal(1.0f, Green(inPlace, TargetSize / 2, TargetSize / 2), 3);
        Assert.Equal(dispatched, inPlace);
    }

    [Fact]
    public void LinearAndDispatchedBarrierLoopsCompileWithTheRuntimeMetalCompiler()
    {
        if (SkipWithoutMetalDevice("the barrier loop compile")) return;

        var stage = Gen5NggFixtures.StageInfo with
        {
            VerticesPerSubgroup = 32,
            PrimitivesPerSubgroup = 1,
            MaxOutputPerSubgroup = 32,
        };
        foreach (var (name, program, info) in new[]
        {
            ("linear", Gen5NggFixtures.BarrierLoopWithTwoExits(), stage),
            ("dispatched", Gen5NggFixtures.BarrierLoopWithTwoExits(), Gen5NggFixtures.StageInfo),
            ("branch", Gen5NggFixtures.BarrierInDivergentBranch(), Gen5NggFixtures.StageInfo),
        })
        {
            var shader = Gen5NggFixtures.CompileOrThrow(program, info);
            Assert.True(
                MetalNative.TryCompileLibrary(shader.Source, out _, out var error),
                $"[{name}] Metal rejected the emitted mesh MSL: {error}\n{shader.Source}");
        }
    }

    [Fact]
    public void CachedMeshletsDrawTheSamePixelsAsWholeTriangleSubgroups()
    {
        if (SkipWithoutMetalDevice("the meshlet mesh execution test")) return;

        // Two triangles sharing vertices 0-2, drawn indexed. As whole-triangle subgroups each
        // triangle owns three vertex slots; as one cached meshlet they share three vertices
        // and name them by local index. The rasterized result has to be identical.
        var shader = Gen5NggFixtures.CompileOrThrow(
            Gen5NggFixtures.Wave64Triangles(), Gen5NggFixtures.Wave64StageInfo);
        var mesh = shader.MeshInfo!;
        var source = shader.Source + FragmentSource;
        if (!MetalNative.TryCompileLibrary(source, out var library, out var compileError))
        {
            throw new InvalidOperationException($"Metal rejected the mesh MSL: {compileError}\n{source}");
        }

        uint[] indices = [0, 1, 2, 2, 1, 0];
        var indexBytes = new byte[indices.Length * sizeof(uint)];
        Buffer.BlockCopy(indices, 0, indexBytes, 0, indexBytes.Length);

        // One meshlet: header {vertexOffset, primOffset, firstTriangle, 3 | 2 << 16}, then the
        // vertex list (raw index values), then one packed local triple per triangle.
        uint[] meshlets =
        [
            0, 0, 0, 3 | (2u << 16),
            0, 1, 2,
            0 | (1u << 8) | (2u << 16),
            2 | (1u << 8) | (0u << 16),
        ];
        var meshletBytes = new byte[meshlets.Length * sizeof(uint)];
        Buffer.BlockCopy(meshlets, 0, meshletBytes, 0, meshletBytes.Length);

        float[] Render(bool useMeshlets)
        {
            var words = new uint[Gen5MslMeshInfo.DrawInfoDwordCount];
            words[0] = mesh.VerticesPerSubgroup;
            words[1] = mesh.PrimitivesPerSubgroup;
            words[2] = (uint)indices.Length;
            words[3] = 2;
            words[6] = 1;
            words[15] = (uint)indices.Length;
            if (useMeshlets)
            {
                words[16] = 1;
                words[17] = 4;
                words[18] = 7;
            }

            var drawInfo = new byte[words.Length * sizeof(uint)];
            Buffer.BlockCopy(words, 0, drawInfo, 0, drawInfo.Length);
            Assert.True(
                MetalNative.TryRenderMesh(
                    library,
                    shader.EntryPoint,
                    "gen5_mesh_test_fs",
                    [
                        (BuildArgumentBuffer(shader), 0u),
                        (new byte[64], 1u),
                        (drawInfo, mesh.DrawInfoBufferIndex),
                        (indexBytes, mesh.IndexBufferIndex),
                        (useMeshlets ? meshletBytes : new byte[16], Gen5MslMeshInfo.MeshletBufferIndex),
                    ],
                    threadgroups: 1,
                    mesh.ThreadsPerThreadgroup,
                    TargetSize,
                    TargetSize,
                    out var pixels,
                    out var renderError),
                $"the mesh draw failed: {renderError}");
            return pixels;
        }

        var plain = Render(useMeshlets: false);
        var cached = Render(useMeshlets: true);
        Assert.Equal(1.0f, Green(plain, TargetSize / 2, TargetSize / 2), 3);
        Assert.Equal(plain, cached);
    }

    [Fact]
    public void IndexedWave64PrimitiveShaderDrawsTrianglesFromEverySubgroup()
    {
        if (SkipWithoutMetalDevice("the narrow wave64 mesh execution test")) return;

        // 12 triangles, 36 indices, two subgroups of 10 and 2 triangles. Triangles 0-9 use
        // index 3 for every corner, so they are degenerate; triangle 10 is the first of the
        // second subgroup and uses indices 0/1/2, the triangle through the screen center.
        // It only appears if the seeded local vertices decode, the second subgroup starts on
        // a triangle boundary (slot 30, not 64) and the narrow wave launches.
        var shader = Gen5NggFixtures.CompileOrThrow(
            Gen5NggFixtures.Wave64Triangles(), Gen5NggFixtures.Wave64StageInfo);
        var mesh = shader.MeshInfo!;
        Assert.Equal(32u, mesh.ThreadsPerThreadgroup);
        var source = shader.Source + FragmentSource;
        if (!MetalNative.TryCompileLibrary(source, out var library, out var compileError))
        {
            throw new InvalidOperationException($"Metal rejected the mesh MSL: {compileError}\n{source}");
        }

        var indices = new uint[36];
        for (var i = 0; i < indices.Length; i++)
        {
            indices[i] = 3;
        }

        indices[30] = 0;
        indices[31] = 1;
        indices[32] = 2;
        var indexBytes = new byte[indices.Length * sizeof(uint)];
        Buffer.BlockCopy(indices, 0, indexBytes, 0, indexBytes.Length);

        var words = new uint[Gen5MslMeshInfo.DrawInfoDwordCount];
        words[0] = mesh.VerticesPerSubgroup;
        words[1] = mesh.PrimitivesPerSubgroup;
        words[2] = (uint)indices.Length;
        words[3] = 2; // 32-bit indices.
        words[6] = 1;
        words[15] = (uint)indices.Length;
        var drawInfo = new byte[words.Length * sizeof(uint)];
        Buffer.BlockCopy(words, 0, drawInfo, 0, drawInfo.Length);

        Assert.True(
            MetalNative.TryRenderMesh(
                library,
                shader.EntryPoint,
                "gen5_mesh_test_fs",
                [
                    (BuildArgumentBuffer(shader), 0u),
                    (new byte[64], 1u),
                    (drawInfo, mesh.DrawInfoBufferIndex),
                    (indexBytes, mesh.IndexBufferIndex),
                ],
                threadgroups: 2,
                mesh.ThreadsPerThreadgroup,
                TargetSize,
                TargetSize,
                out var pixels,
                out var renderError),
            $"the mesh draw failed: {renderError}");

        Assert.Equal(1.0f, Green(pixels, TargetSize / 2, TargetSize / 2), 3);
        Assert.Equal(0.0f, Green(pixels, 1, 1), 3);
        Assert.Equal(0.0f, Green(pixels, TargetSize - 2, 1), 3);
    }

    [Fact]
    public void BufferFetchedVerticesRasterizeThroughAMeshPipeline()
    {
        if (SkipWithoutMetalDevice("the mesh vertex-fetch test")) return;

        // The ES part fetches its position with buffer_load_format_xyzw over a V# in the
        // geometry stage's user data - the fetch the vertex-fetch substitution used to
        // swallow, leaving the mesh stage with no buffer binding. Positions come from the
        // buffer, so a pixel only appears if that binding reached the GPU.
        var runtimeInputs = new ResourceRuntimeInputs { UserData = Gen5NggFixtures.PositionUserData() };
        var shader = Gen5NggFixtures.CompileOrThrow(
            Gen5NggFixtures.BufferFetchTriangle(), runtimeInputs: runtimeInputs);
        var mesh = shader.MeshInfo!;
        var source = shader.Source + FragmentSource;
        if (!MetalNative.TryCompileLibrary(source, out var library, out var compileError))
        {
            throw new InvalidOperationException($"Metal rejected the mesh MSL: {compileError}\n{source}");
        }

        if (!MetalNative.TryRenderMeshWithArgumentBuffer(
                library,
                shader.EntryPoint,
                "gen5_mesh_test_fs",
                PositionBuffer(),
                new byte[SharpEmu.ShaderCompiler.Resources.PushData.ByteSize],
                BuildDrawInfo(mesh),
                shader.ArgumentLayout!,
                mesh.DrawInfoBufferIndex,
                mesh.IndexBufferIndex,
                threadgroups: 1,
                mesh.ThreadsPerThreadgroup,
                TargetSize,
                TargetSize,
                out var pixels,
                out var renderError))
        {
            throw new InvalidOperationException($"the mesh draw failed: {renderError}");
        }

        // Same triangle as the arithmetic fixture, so the same pixels have to light up.
        Assert.Equal(1.0f, Green(pixels, TargetSize / 2, TargetSize / 2), 3);
        Assert.Equal(1.0f, Red(pixels, TargetSize / 2, TargetSize / 2), 1);
        Assert.Equal(0.0f, Green(pixels, 1, 1), 3);
        Assert.Equal(0.0f, Green(pixels, TargetSize - 2, 1), 3);
    }

    // Four float4 records; the primitive export references the first three.
    private static byte[] PositionBuffer()
    {
        var positions = new[]
        {
            -0.8f, -0.8f, 0.0f, 1.0f,
             0.0f,  0.8f, 0.0f, 1.0f,
             0.8f, -0.8f, 0.0f, 1.0f,
             0.0f,  0.0f, 0.0f, 1.0f,
        };
        var bytes = new byte[Gen5NggFixtures.PositionStrideBytes * Gen5NggFixtures.PositionRecordCount];
        Buffer.BlockCopy(positions, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    [Fact]
    public void NullPrimitiveBitCullsTheTriangle()
    {
        if (SkipWithoutMetalDevice("the mesh culling test")) return;

        var pixels = RenderOrThrow(Gen5NggFixtures.CullingTriangle(nullPrimitive: true));
        for (var y = 0u; y < TargetSize; y++)
        {
            for (var x = 0u; x < TargetSize; x++)
            {
                Assert.Equal(0.0f, Green(pixels, x, y), 3);
            }
        }
    }

    [Fact]
    public void PreservedVoidCaptureCompilesAsAMeshShader()
    {
        if (SkipWithoutMetalDevice("the captured-shader compile")) return;
        if (!TryLoadVoidCapture(out var program, out var stageInfo, out var userDataBase, out var reason))
        {
            output.WriteLine($"Void capture unavailable ({reason}); the captured-shader compile is skipped.");
            return;
        }

        Assert.True(ShaderCompileRequest.RequiresPrimitiveShaderWorkgroup(program));

        // The captured program is exactly the shape the vertex-fetch substitution must not
        // touch: its ES part fetches vertices with buffer_load_format over a V# its user
        // data carries, and substituting that drops the scalar loads the fetch needs, so
        // the mesh stage is left with "missing buffer-memory binding".
        Assert.False(ShaderCompileRequest.AllowsEmbeddedVertexFetch(program, stageInfo));
        Assert.Contains(program.Instructions, instruction =>
            instruction.Control is Gen5BufferMemoryControl &&
            instruction.Opcode.StartsWith("BufferLoadFormat", StringComparison.Ordinal));

        var request = Gen5ComputeFixtures.RequestOrThrow(
            program,
            ShaderStage.Vertex,
            requiredVertexOutputCount: 4,
            primitiveShader: stageInfo,
            userDataBase: userDataBase);
        Assert.True(Gen5MslTranslator.RequiresMeshEmission(request));

        // Every one of those fetches has to reach a buffer binding, or translation fails.
        Assert.NotEmpty(request.Resources.Info.Buffers);
        Assert.True(
            Gen5MslTranslator.TryCompileProgram(request, out var shader, out var translateError),
            $"the captured primitive shader did not translate: {translateError}");
        Assert.Equal(Gen5MslStage.Mesh, shader.Stage);
        Assert.True(
            MetalNative.TryCompileLibrary(shader.Source, out _, out var compileError),
            $"Metal rejected the captured primitive shader's MSL: {compileError}");
        output.WriteLine(
            $"Captured primitive shader translated: {program.Instructions.Count} instructions, " +
            $"{shader.MeshInfo!.ThreadsPerThreadgroup} threads/threadgroup, " +
            $"{shader.MeshInfo.MaxVertices} vertices, {shader.MeshInfo.MaxPrimitives} primitives.");
    }

    private static IEnumerable<(string Name, Gen5MslShader Shader)> SyntheticShaders()
    {
        yield return ("culling triangle", Gen5NggFixtures.CompileOrThrow(Gen5NggFixtures.CullingTriangle()));
        yield return ("null primitive", Gen5NggFixtures.CompileOrThrow(Gen5NggFixtures.CullingTriangle(nullPrimitive: true)));
        yield return ("line output", Gen5NggFixtures.CompileOrThrow(
            Gen5NggFixtures.CullingTriangle(),
            Gen5NggFixtures.StageInfo with { OutputPrimitiveType = Gen5NggExports.OutputPrimitiveLineStrip }));
        yield return ("point output", Gen5NggFixtures.CompileOrThrow(
            Gen5NggFixtures.CullingTriangle(),
            Gen5NggFixtures.StageInfo with { OutputPrimitiveType = Gen5NggExports.OutputPrimitivePointList }));
        yield return ("clamped subgroup", Gen5NggFixtures.CompileOrThrow(
            Gen5NggFixtures.CullingTriangle(),
            Gen5NggFixtures.StageInfo with { VerticesPerSubgroup = 4096, PrimitivesPerSubgroup = 4096 }));
        yield return ("device lds", Gen5NggFixtures.CompileOrThrow(
            Gen5NggFixtures.CullingTriangle(),
            Gen5NggFixtures.StageInfo with { LocalDataShareBytes = 15 * 1024 }));
        yield return ("device lds null primitive", Gen5NggFixtures.CompileOrThrow(
            Gen5NggFixtures.CullingTriangle(nullPrimitive: true),
            Gen5NggFixtures.StageInfo with { LocalDataShareBytes = 15 * 1024 }));
        yield return ("narrow wave64", Gen5NggFixtures.CompileOrThrow(
            Gen5NggFixtures.Wave64Triangles(), Gen5NggFixtures.Wave64StageInfo));
        yield return ("narrow wave64 lane probe", Gen5NggFixtures.CompileOrThrow(
            Gen5NggFixtures.Wave64LaneProbe(), Gen5NggFixtures.Wave64StageInfo));
        yield return ("gs half", Gen5NggFixtures.CompileOrThrow(
            Gen5NggFixtures.CullingTriangle(),
            Gen5NggFixtures.StageInfo with { GeometryVerticesOut = 3 }));
    }

    private float[] RenderOrThrow(Gen5ShaderProgram program)
    {
        var shader = Gen5NggFixtures.CompileOrThrow(program);
        var mesh = shader.MeshInfo!;
        var source = shader.Source + FragmentSource;
        if (!MetalNative.TryCompileLibrary(source, out var library, out var compileError))
        {
            throw new InvalidOperationException($"Metal rejected the mesh MSL: {compileError}\n{source}");
        }

        if (!MetalNative.TryRenderMesh(
                library,
                shader.EntryPoint,
                "gen5_mesh_test_fs",
                [
                    (BuildArgumentBuffer(shader), 0u),
                    (new byte[64], 1u),
                    (BuildDrawInfo(mesh), mesh.DrawInfoBufferIndex),
                    (new byte[16], mesh.IndexBufferIndex),
                ],
                threadgroups: 1,
                mesh.ThreadsPerThreadgroup,
                TargetSize,
                TargetSize,
                out var pixels,
                out var renderError))
        {
            throw new InvalidOperationException($"the mesh draw failed: {renderError}");
        }

        return pixels;
    }

    // The stage argument buffer; the synthetic programs touch no resources, so zeros are
    // enough — only its size has to match the layout.
    private static byte[] BuildArgumentBuffer(Gen5MslShader shader) =>
        new byte[Math.Max(shader.ArgumentLayout?.ByteSize ?? 0, 8)];

    // Gen5MeshDrawInfo, in the order docs/mac/ngg-mesh-shaders.md records.
    private static byte[] BuildDrawInfo(Gen5MslMeshInfo mesh)
    {
        var words = new uint[Gen5MslMeshInfo.DrawInfoDwordCount];
        words[0] = mesh.MaxVertices;
        words[1] = mesh.MaxPrimitives;
        words[2] = 0; // index_count: an automatic draw, so the lane index is the vertex.
        words[3] = 0; // index_type: none.
        words[6] = 1; // instance_count.
        words[8] = mesh.MaxVertices;
        words[9] = mesh.MaxPrimitives;
        var bytes = new byte[words.Length * sizeof(uint)];
        Buffer.BlockCopy(words, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    // Stands in for the translated pixel program: it reads the same position/user(locn0)
    // interface the vertex path produces, which is the point of the shared payload struct.
    private const string FragmentSource = """

        struct Gen5MeshTestIn
        {
            float4 position [[position]];
            float4 param0 [[user(locn0)]];
        };

        fragment float4 gen5_mesh_test_fs(Gen5MeshTestIn in [[stage_in]])
        {
            return float4(in.param0.x, 1.0f, 0.0f, 1.0f);
        }
        """;

    // Reads param0's three vertex values the way a translated pixel program built with
    // ShaderCompileRequest.PixelPerVertexInputsFromPrimitive declares them.
    private const string PerVertexFragmentSource = """

        struct Gen5MeshPerVertexIn
        {
            float4 position [[position]];
            float4 param0 [[user(locn0)]];
            float4 param0_v0 [[user(locn0_v0)]];
            float4 param0_v1 [[user(locn0_v1)]];
            float4 param0_v2 [[user(locn0_v2)]];
            float3 vertex_w [[user(sharpemu_vertex_w), flat]];
            float3 bary [[barycentric_coord, center_perspective]];
        };

        fragment float4 gen5_mesh_per_vertex_fs(Gen5MeshPerVertexIn in [[stage_in]])
        {
            const float3 values = float3(in.param0_v0.x, in.param0_v1.x, in.param0_v2.x);
            return float4(values, dot(in.bary, values) - in.param0.x + length(in.vertex_w - float3(1.0)));
        }
        """;

    private static float Red(float[] pixels, uint x, uint y) => pixels[((y * TargetSize) + x) * 4];

    private static float Green(float[] pixels, uint x, uint y) => pixels[(((y * TargetSize) + x) * 4) + 1];

    // The capture is a decoded-program JSON plus the raw code range beside it. Only the
    // range bytes are read, and only when the directory exists on this machine.
    private static bool TryLoadVoidCapture(
        out Gen5ShaderProgram program,
        out Gen5NggStageInfo stageInfo,
        out uint userDataBase,
        out string reason)
    {
        program = null!;
        stageInfo = default;
        userDataBase = Gen5NggFixtures.VertexUserDataBase;
        if (!Directory.Exists(VoidCaptureDirectory))
        {
            reason = "the capture directory is not present";
            return false;
        }

        foreach (var json in Directory.GetFiles(VoidCaptureDirectory, "skipped-export-*.json"))
        {
            var range = Path.ChangeExtension(json, null) + ".range-0.bin";
            if (!File.Exists(range))
            {
                continue;
            }

            var bytes = File.ReadAllBytes(range);
            if (bytes.Length < sizeof(uint) || !TryReadCaptureRegisters(json, out var captured))
            {
                continue;
            }

            var words = new uint[bytes.Length / sizeof(uint)];
            Buffer.BlockCopy(bytes, 0, words, 0, words.Length * sizeof(uint));
            var memory = new FakeGuestMemory();
            memory.AddRegion(0x1000, words);
            var ctx = new CpuContext(memory, Generation.Gen5);
            if (!Gen5ShaderTranslator.TryDecodeProgram(ctx, 0x1000, out var decoded, out _))
            {
                continue;
            }

            if (!ShaderCompileRequest.RequiresPrimitiveShaderWorkgroup(decoded))
            {
                continue;
            }

            program = decoded;
            stageInfo = captured;
            reason = string.Empty;
            return true;
        }

        reason = "no capture in the directory decoded into a workgroup primitive shader";
        return false;
    }

    // Only the geometry-engine scalars are taken from the capture's JSON; no draw payload,
    // resource descriptor or user-data value is read.
    private static bool TryReadCaptureRegisters(string path, out Gen5NggStageInfo stageInfo)
    {
        stageInfo = default;
        try
        {
            using var document = System.Text.Json.JsonDocument.Parse(File.ReadAllText(path));
            var root = document.RootElement;
            var maxOutput = ReadUInt(root, "maxOutput");
            var ldsSize = root.TryGetProperty("geometryResource2", out var resource2) &&
                resource2.TryGetProperty("LocalDataShareSize", out var lds)
                    ? lds.GetUInt32()
                    : 0u;
            stageInfo = new Gen5NggStageInfo(
                VerticesPerSubgroup: maxOutput,
                PrimitivesPerSubgroup: maxOutput,
                MaxOutputPerSubgroup: maxOutput,
                OutputPrimitiveType: ReadUInt(root, "outputPrimitive"),
                PrimitiveAmplificationFactor: ReadUInt(root, "subgroupControl"),

                // GFX10 programs LDS_SIZE in 128-dword granules.
                LocalDataShareBytes: ldsSize * 128 * sizeof(uint),
                VertexGroupSize: maxOutput,
                PrimitiveGroupSize: maxOutput);
            return true;
        }
        catch (Exception exception) when (exception is IOException or System.Text.Json.JsonException)
        {
            return false;
        }
    }

    private static uint ReadUInt(System.Text.Json.JsonElement root, string name) =>
        root.TryGetProperty(name, out var value) && value.TryGetUInt32(out var parsed) ? parsed : 0u;

    private bool SkipWithoutMetalDevice(string activity)
    {
        if (MetalNative.IsAvailable) return false;
        Assert.False(
            OperatingSystem.IsMacOS() && Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") == "1",
            $"The required gate cannot run {activity} without a Metal device.");
        output.WriteLine($"No Metal device on this host; {activity} skipped.");
        return true;
    }
}
