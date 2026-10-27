// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>
/// The indirect launch of a primitive shader: <see cref="Gen5MslMeshLaunch.KernelSource"/>
/// has to write, from a guest indirect record on the GPU, exactly the draw-info block and the
/// launch the render host's DrawMesh writes on the CPU from the same counts
/// (<see cref="Gen5MslMeshLaunch.SubgroupCount"/>, <see cref="Gen5MslMeshLaunch.FillDrawInfo"/>),
/// and a draw launched from that has to rasterize what the CPU draw does.
/// </summary>
public sealed class NggMeshIndirectLaunchTests(ITestOutputHelper output)
{
    private const uint TargetSize = 32;
    private const ulong MaxThreadgroups = 1UL << 26;
    private const int RecordBytes = 32;

    private static readonly uint[] Counts = [0, 1, 2, 3, 29, 30, 31, 36, 64, 65, 1000, 0xFFFF_FFFF];
    private static readonly uint[] Instances = [0, 1, 3];
    private static readonly uint[] IndexLimits = [uint.MaxValue, 31];

    [Fact]
    public void TheLaunchKernelWritesWhatTheCpuDrawWrites()
    {
        if (SkipWithoutMetalDevice("the indirect launch kernel")) return;

        foreach (var (name, mesh) in Meshes())
        {
            var cases = new List<LaunchCase>();
            foreach (var input in Enum.GetValues<Gen5MslMeshInputTopology>())
            {
                foreach (var count in Counts)
                {
                    foreach (var instances in Instances)
                    {
                        cases.Add(new LaunchCase(input, Indexed: false, count, instances, 0));
                        foreach (var limit in IndexLimits)
                        {
                            cases.Add(new LaunchCase(input, Indexed: true, count, instances, limit));
                        }
                    }
                }
            }

            var records = new byte[cases.Count * RecordBytes];
            var slots = new byte[cases.Count * (int)Gen5MslMeshLaunch.SlotBytes];
            for (var index = 0; index < cases.Count; index++)
            {
                var launch = cases[index];
                WriteRecord(records.AsSpan(index * RecordBytes), launch);
                Gen5MslMeshLaunch.WriteMeshParameters(
                    Parameters(slots, index),
                    mesh,
                    launch.Input,
                    (ulong)(index * RecordBytes),
                    launch.Indexed,
                    IndexType32,
                    launch.IndexLimit,
                    MaxThreadgroups);
            }

            Assert.True(
                MetalNative.TryRunMeshLaunchKernel(records, slots, (uint)cases.Count, out var written, out var error),
                $"[{name}] the launch kernel failed: {error}");
            for (var index = 0; index < cases.Count; index++)
            {
                var launch = cases[index];
                var slot = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, uint>(
                    written.AsSpan(index * (int)Gen5MslMeshLaunch.SlotBytes, (int)Gen5MslMeshLaunch.SlotBytes));
                var (expectedGrid, expectedWords) = CpuLaunch(mesh, launch);
                var label = $"[{name}] {launch}";
                Assert.True(
                    expectedGrid.AsSpan().SequenceEqual(slot.Slice((int)Gen5MslMeshLaunch.GridByteOffset / 4, 3)),
                    $"{label}: grid {string.Join(",", slot.Slice((int)Gen5MslMeshLaunch.GridByteOffset / 4, 3).ToArray())} " +
                    $"!= {string.Join(",", expectedGrid)}");
                if (expectedWords is not null)
                {
                    var actual = slot[..(int)Gen5MslMeshInfo.DrawInfoDwordCount].ToArray();
                    Assert.True(
                        expectedWords.AsSpan().SequenceEqual(actual),
                        $"{label}: draw info {string.Join(",", actual)} != {string.Join(",", expectedWords)}");
                }
            }

            output.WriteLine($"[{name}] {cases.Count} launches match DrawMesh.");
        }
    }

    [Fact]
    public void TheLaunchKernelWritesARectangleListsPatchDraw()
    {
        if (SkipWithoutMetalDevice("the indirect launch kernel")) return;

        // (vertexCount, instanceCount, firstVertex, firstInstance): three rectangles from
        // vertex 4, twice from instance 1; then an empty instance count draws nothing.
        uint[] records = [9, 2, 4, 1, 0, 0, 0, 0, 9, 0, 4, 1, 0, 0, 0, 0];
        var recordBytes = new byte[records.Length * sizeof(uint)];
        Buffer.BlockCopy(records, 0, recordBytes, 0, recordBytes.Length);
        var slots = new byte[2 * Gen5MslMeshLaunch.SlotBytes];
        Gen5MslMeshLaunch.WriteRectangleParameters(Parameters(slots, 0), 0, MaxThreadgroups);
        Gen5MslMeshLaunch.WriteRectangleParameters(Parameters(slots, 1), RecordBytes, MaxThreadgroups);

        Assert.True(MetalNative.TryRunMeshLaunchKernel(recordBytes, slots, 2, out var written, out var error), error);
        var words = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, uint>(written);
        var grid = (int)Gen5MslMeshLaunch.GridByteOffset / 4;
        Assert.Equal([4u, 0u, 0u, 0u], words[..4].ToArray());
        Assert.Equal([3u, 2u, 0u, 1u], words.Slice(grid, 4).ToArray());
        var second = (int)Gen5MslMeshLaunch.SlotBytes / 4;
        Assert.Equal(0u, words[second + grid]);
    }

    // The CPU path reads the record back and draws through DrawMesh: an indexed draw binds
    // the indices from its first index on, a non-indexed one has no start DrawMesh uses. The
    // GPU path binds the indices from the start and launches from what the kernel wrote. The
    // narrow wave64 program draws 12 triangles whose corners are all index 3 (degenerate) but
    // triangle 10's, 0/1/2 through the screen center, as in the narrow wave64 runtime test;
    // the GS half assembles 64 strip or list vertices into one input primitive per subgroup.
    [Theory]
    [InlineData("narrow wave64", true, Gen5MslMeshInputTopology.TriangleList, 36u)]
    [InlineData("gs half", false, Gen5MslMeshInputTopology.TriangleStrip, 64u)]
    [InlineData("gs half", false, Gen5MslMeshInputTopology.TriangleList, 64u)]
    public void AnIndirectDrawWithGpuWrittenArgumentsRendersLikeTheCpuDraw(
        string program, bool indexed, Gen5MslMeshInputTopology input, uint count)
    {
        if (SkipWithoutMetalDevice("the indirect mesh draw")) return;

        var shader = program == "narrow wave64"
            ? Gen5NggFixtures.CompileOrThrow(Gen5NggFixtures.Wave64Triangles(), Gen5NggFixtures.Wave64StageInfo)
            : Gen5NggFixtures.CompileOrThrow(
                Gen5NggFixtures.CullingTriangle(), Gen5NggFixtures.StageInfo with { GeometryVerticesOut = 3 });
        var mesh = shader.MeshInfo!;
        Assert.True(program == "narrow wave64" ? mesh.NarrowWave64 : mesh.HasGeometryHalf);
        var source = shader.Source + FragmentSource;
        Assert.True(MetalNative.TryCompileLibrary(source, out var library, out var compileError), compileError);

        const uint firstIndex = 6;
        var drawn = new uint[indexed ? count : 4];
        Array.Fill(drawn, 3u);
        if (indexed)
        {
            drawn[30] = 0;
            drawn[31] = 1;
            drawn[32] = 2;
        }

        var guestIndices = new uint[firstIndex + drawn.Length];
        Array.Fill(guestIndices, 3u);
        drawn.CopyTo(guestIndices, (int)firstIndex);

        // The CPU draw: the counts read back, the indices bound from the first index on.
        var words = new uint[Gen5MslMeshInfo.DrawInfoDwordCount];
        Gen5MslMeshLaunch.FillDrawInfo(mesh, input, words, count, 1, indexed, IndexType32, 0, 0, 0);
        var subgroups = Gen5MslMeshLaunch.SubgroupCount(mesh, input, count);
        var cpu = Render(library, shader, mesh, Bytes(words), Bytes(drawn), subgroups, indirect: null);

        // The GPU draw: the guest record, the whole index buffer, the kernel's launch.
        var slot = new byte[Gen5MslMeshLaunch.SlotBytes];
        Gen5MslMeshLaunch.WriteMeshParameters(
            Parameters(slot, 0), mesh, input, 0, indexed, IndexType32, uint.MaxValue, MaxThreadgroups);
        uint[] record = indexed ? [count, 1, firstIndex, 0, 0] : [count, 1, firstIndex, 0];
        var launch = new MeshIndirectLaunch(Bytes(record), slot, mesh.DrawInfoBufferIndex);
        var gpu = Render(
            library, shader, mesh, new byte[Gen5MslMeshInfo.DrawInfoByteSize],
            Bytes(indexed ? guestIndices : drawn), 0, launch);

        var lit = cpu.Where((value, index) => index % 4 == 1 && value != 0).Count();
        output.WriteLine($"[{program} {input}] subgroups={subgroups} lit pixels={lit}");
        Assert.True(lit > 0, $"[{program}] the CPU draw rasterized nothing, so the comparison would prove nothing");
        Assert.Equal(cpu, gpu);
        var written = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, uint>(launch.Written);
        Assert.Equal([subgroups, 1u, 1u], written.Slice((int)Gen5MslMeshLaunch.GridByteOffset / 4, 3).ToArray());
        // Word for word what the CPU draw bound, but for the first index the GPU path keeps.
        var block = written[..(int)Gen5MslMeshInfo.DrawInfoDwordCount].ToArray();
        if (indexed)
        {
            Assert.Equal(firstIndex, block[4]);
            block[4] = 0;
        }

        Assert.Equal(words, block);
    }

    private float[] Render(
        nint library,
        Gen5MslShader shader,
        Gen5MslMeshInfo mesh,
        byte[] drawInfo,
        byte[] indices,
        uint threadgroups,
        MeshIndirectLaunch? indirect)
    {
        Assert.True(
            MetalNative.TryRenderMesh(
                library,
                shader.EntryPoint,
                "gen5_mesh_test_fs",
                [
                    (new byte[Math.Max(shader.ArgumentLayout?.ByteSize ?? 0, 8)], 0u),
                    (new byte[64], 1u),
                    (drawInfo, mesh.DrawInfoBufferIndex),
                    (indices, mesh.IndexBufferIndex),
                ],
                threadgroups,
                mesh.ThreadsPerThreadgroup,
                TargetSize,
                TargetSize,
                out var pixels,
                out var error,
                indirect: indirect),
            $"the mesh draw failed: {error}");
        return pixels;
    }

    private const uint IndexType32 = 2;

    private readonly record struct LaunchCase(
        Gen5MslMeshInputTopology Input, bool Indexed, uint Count, uint Instances, uint IndexLimit)
    {
        // Fixed starts, so the words that only carry them are checked too.
        public const uint FirstIndexOrVertex = 7;
        public const int BaseVertex = -5;
        public const uint FirstInstance = 2;
    }

    private static void WriteRecord(Span<byte> destination, LaunchCase launch)
    {
        var words = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, uint>(destination[..RecordBytes]);
        words[0] = launch.Count;
        words[1] = launch.Instances;
        words[2] = LaunchCase.FirstIndexOrVertex;
        if (launch.Indexed)
        {
            words[3] = unchecked((uint)LaunchCase.BaseVertex);
            words[4] = LaunchCase.FirstInstance;
        }
        else
        {
            words[3] = LaunchCase.FirstInstance;
        }
    }

    // What DrawMesh launches and writes for the record's counts read back on the CPU: a grid
    // of (subgroups, instances, 1), or nothing (an empty grid) when it would draw nothing.
    private static (uint[] Grid, uint[]? Words) CpuLaunch(Gen5MslMeshInfo mesh, LaunchCase launch)
    {
        var count = launch.Indexed ? Math.Min(launch.Count, launch.IndexLimit) : launch.Count;
        var subgroups = Gen5MslMeshLaunch.SubgroupCount(mesh, launch.Input, count);
        if (subgroups == 0 || launch.Instances == 0 || (ulong)subgroups * launch.Instances > MaxThreadgroups)
        {
            return ([0, 0, 1], null);
        }

        var words = new uint[Gen5MslMeshInfo.DrawInfoDwordCount];
        Gen5MslMeshLaunch.FillDrawInfo(
            mesh,
            launch.Input,
            words,
            count,
            launch.Instances,
            launch.Indexed,
            IndexType32,
            LaunchCase.FirstIndexOrVertex,
            launch.Indexed ? LaunchCase.BaseVertex : 0,
            LaunchCase.FirstInstance);
        return ([subgroups, launch.Instances, 1], words);
    }

    private static IEnumerable<(string Name, Gen5MslMeshInfo Mesh)> Meshes()
    {
        yield return ("narrow wave64", Gen5NggFixtures.CompileOrThrow(
            Gen5NggFixtures.Wave64Triangles(), Gen5NggFixtures.Wave64StageInfo).MeshInfo!);
        yield return ("wave32 culling", Gen5NggFixtures.CompileOrThrow(Gen5NggFixtures.CullingTriangle()).MeshInfo!);
        yield return ("gs half", Gen5NggFixtures.CompileOrThrow(
            Gen5NggFixtures.CullingTriangle(), Gen5NggFixtures.StageInfo with { GeometryVerticesOut = 3 }).MeshInfo!);
        yield return ("gs half strip of four", new Gen5MslMeshInfo(
            ThreadsPerThreadgroup: 64, WavesPerThreadgroup: 2, MaxVertices: 64, MaxPrimitives: 64,
            VerticesPerPrimitive: 3, Topology: Gen5MslMeshTopology.Triangle, ThreadgroupMemoryBytes: 4096,
            DrawInfoBufferIndex: 2, IndexBufferIndex: 3, ReadsIndexBuffer: true, GeometryVerticesOut: 4));
        yield return ("geometry separate input and output capacities", new Gen5MslMeshInfo(
            ThreadsPerThreadgroup: 128, WavesPerThreadgroup: 2, MaxVertices: 96, MaxPrimitives: 88,
            VerticesPerPrimitive: 3, Topology: Gen5MslMeshTopology.Triangle, ThreadgroupMemoryBytes: 4096,
            DrawInfoBufferIndex: 2, IndexBufferIndex: 3, ReadsIndexBuffer: true, GeometryVerticesOut: 24,
            Wave64: true, GeometryInputVertices: 24, GeometryInputPrimitives: 4));
        yield return ("points", new Gen5MslMeshInfo(
            ThreadsPerThreadgroup: 32, WavesPerThreadgroup: 1, MaxVertices: 32, MaxPrimitives: 32,
            VerticesPerPrimitive: 1, Topology: Gen5MslMeshTopology.Point, ThreadgroupMemoryBytes: 4096,
            DrawInfoBufferIndex: 2, IndexBufferIndex: 3, ReadsIndexBuffer: true));
    }

    private static Span<uint> Parameters(byte[] slots, int index) =>
        System.Runtime.InteropServices.MemoryMarshal.Cast<byte, uint>(slots.AsSpan(
            (index * (int)Gen5MslMeshLaunch.SlotBytes) + (int)Gen5MslMeshLaunch.ParametersByteOffset,
            Gen5MslMeshLaunch.ParameterDwords * sizeof(uint)));

    private static byte[] Bytes(uint[] words)
    {
        var bytes = new byte[words.Length * sizeof(uint)];
        Buffer.BlockCopy(words, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    // As the runtime tests: the program's param0.x in red, coverage in green.
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
