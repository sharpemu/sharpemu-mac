// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

public sealed class MetalMeshletBuilderTests
{
    // Rebuilds the index stream from the buffer the way the mesh prologue reads it.
    private static uint[] Reassemble(MeshletBuilder.Result result, uint maxVertices, uint maxPrimitives)
    {
        var indices = new List<uint>();
        var nextTriangle = 0u;
        for (var meshlet = 0u; meshlet < result.MeshletCount; meshlet++)
        {
            var vertexOffset = result.Words[(meshlet * 4) + 0];
            var primitiveOffset = result.Words[(meshlet * 4) + 1];
            var firstTriangle = result.Words[(meshlet * 4) + 2];
            var counts = result.Words[(meshlet * 4) + 3];
            var vertexCount = counts & 0xFFFFu;
            var primitiveCount = counts >> 16;

            Assert.True(vertexCount <= maxVertices, $"meshlet {meshlet} has {vertexCount} vertices");
            Assert.True(primitiveCount <= maxPrimitives, $"meshlet {meshlet} has {primitiveCount} triangles");
            Assert.Equal(nextTriangle, firstTriangle);
            nextTriangle += primitiveCount;

            for (var primitive = 0u; primitive < primitiveCount; primitive++)
            {
                var packed = result.Words[result.PrimitiveBase + primitiveOffset + primitive];
                foreach (var shift in new[] { 0, 8, 16 })
                {
                    var local = (packed >> shift) & 0xFFu;
                    Assert.True(local < vertexCount);
                    indices.Add(result.Words[result.VertexBase + vertexOffset + local]);
                }
            }
        }

        return [.. indices];
    }

    // The builder before its vertex table: a linear search of the current meshlet per lookup.
    private static uint[] ReferenceWords(uint[] indices, uint maxVertices, uint maxPrimitives)
    {
        maxVertices = Math.Clamp(maxVertices, 3, MeshletBuilder.MaxLocalVertices);
        maxPrimitives = Math.Clamp(maxPrimitives, 1, MeshletBuilder.MaxLocalVertices);
        var headers = new List<uint>();
        var vertices = new List<uint>();
        var primitives = new List<uint>();
        var current = new List<uint>();
        uint vertexStart = 0, primitiveStart = 0, firstTriangle = 0, primitiveCount = 0;
        void Flush()
        {
            if (primitiveCount == 0)
            {
                return;
            }

            headers.AddRange([vertexStart, primitiveStart, firstTriangle, (uint)current.Count | (primitiveCount << 16)]);
            vertices.AddRange(current);
            vertexStart += (uint)current.Count;
            primitiveStart += primitiveCount;
            current.Clear();
            primitiveCount = 0;
        }

        uint Local(uint value)
        {
            var found = current.IndexOf(value);
            if (found >= 0)
            {
                return (uint)found;
            }

            current.Add(value);
            return (uint)current.Count - 1;
        }

        for (var triangle = 0; triangle < indices.Length / 3; triangle++)
        {
            var a = indices[triangle * 3];
            var b = indices[(triangle * 3) + 1];
            var c = indices[(triangle * 3) + 2];
            var added = (current.Contains(a) ? 0u : 1u) + (b != a && !current.Contains(b) ? 1u : 0u) +
                (c != a && c != b && !current.Contains(c) ? 1u : 0u);
            if (primitiveCount == maxPrimitives || (uint)current.Count + added > maxVertices)
            {
                Flush();
            }

            if (primitiveCount == 0)
            {
                firstTriangle = (uint)triangle;
            }

            var la = Local(a);
            var lb = Local(b);
            var lc = Local(c);
            primitives.Add(la | (lb << 8) | (lc << 16));
            primitiveCount++;
        }

        Flush();
        return [.. headers, .. vertices, .. primitives];
    }

    [Theory]
    [InlineData(30u, 10u, 200)]
    [InlineData(64u, 126u, 100000)]
    [InlineData(128u, 256u, 4000)]
    [InlineData(255u, 255u, 600)]
    public void TheVertexTableBuildsWhatTheLinearSearchBuilt(uint maxVertices, uint maxPrimitives, int vertexRange)
    {
        var random = new Random(4321);
        var scratch = new MeshletBuilder.Scratch();
        // One scratch across builds, as the cache keeps it: each build starts from the last one's table.
        for (var build = 0; build < 3; build++)
        {
            var indices = new uint[3 * 20000];
            for (var i = 0; i < indices.Length; i++)
            {
                indices[i] = (uint)random.Next(0, vertexRange);
            }

            MeshletBuilder.BuildInto(indices, maxVertices, maxPrimitives, scratch);
            var words = new uint[scratch.WordCount];
            scratch.CopyTo(words);
            Assert.Equal(ReferenceWords(indices, maxVertices, maxPrimitives), words);
        }
    }

    [Theory]
    [InlineData(30u, 10u)]
    [InlineData(32u, 32u)]
    [InlineData(64u, 64u)]
    [InlineData(255u, 255u)]
    public void RandomTrianglesReassembleToTheInputInOrder(uint maxVertices, uint maxPrimitives)
    {
        var random = new Random(1234);
        var indices = new uint[3 * 5000];
        for (var i = 0; i < indices.Length; i++)
        {
            // A small vertex range makes reuse (and degenerate triangles) common.
            indices[i] = (uint)random.Next(0, 200);
        }

        var result = MeshletBuilder.Build(indices, maxVertices, maxPrimitives);
        Assert.Equal(indices, Reassemble(result, maxVertices, maxPrimitives));
    }

    [Fact]
    public void DegenerateTrianglesAreKept()
    {
        uint[] indices = [3, 3, 3, 1, 2, 2, 5, 6, 7];
        var result = MeshletBuilder.Build(indices, 30, 10);
        Assert.Equal(indices, Reassemble(result, 30, 10));
        Assert.Equal(1u, result.MeshletCount);
    }

    [Fact]
    public void ARaggedIndexCountDropsTheIncompleteTriangle()
    {
        uint[] indices = [0, 1, 2, 3, 4];
        var result = MeshletBuilder.Build(indices, 30, 10);
        Assert.Equal<uint>([0, 1, 2], Reassemble(result, 30, 10));
    }

    [Fact]
    public void AnEmptyDrawBuildsNoMeshlets()
    {
        Assert.Equal(0u, MeshletBuilder.Build([], 30, 10).MeshletCount);
    }

    [Theory]
    [InlineData(32u, 32u)]
    [InlineData(30u, 10u)]
    public void AGridPacksNearThePrimitiveLimit(uint maxVertices, uint maxPrimitives)
    {
        const int Cells = 64;
        var indices = new List<uint>();
        for (var y = 0; y < Cells; y++)
        {
            for (var x = 0; x < Cells; x++)
            {
                var a = (uint)((y * (Cells + 1)) + x);
                var b = a + 1;
                var c = a + Cells + 1;
                var d = c + 1;
                indices.AddRange([a, b, c, b, d, c]);
            }
        }

        var result = MeshletBuilder.Build([.. indices], maxVertices, maxPrimitives);
        Assert.Equal([.. indices], Reassemble(result, maxVertices, maxPrimitives));

        var triangles = indices.Count / 3;
        var ideal = (triangles + (int)maxPrimitives - 1) / (int)maxPrimitives;
        if (maxVertices == 32)
        {
            Assert.True(
                result.MeshletCount <= ideal * 1.2,
                $"{result.MeshletCount} meshlets for {triangles} triangles (ideal {ideal})");
        }
        else
        {
            // 10-triangle meshlets are bound by the triangle limit; the vertex limit never bites.
            Assert.True(result.MeshletCount <= ideal * 1.2);
        }
    }
}
