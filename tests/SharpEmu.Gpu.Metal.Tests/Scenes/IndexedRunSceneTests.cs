// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

/// <summary>
/// Indexed draw runs on the Metal host (upstream c30081163/cb790c35e): consecutive DRAW_INDEX_INDIRECT packets under the
/// same registers whose records lie in one page reach the translation as one deferred draw carrying the records' offsets
/// (MetalRenderHost.TryDrawIndexedRun), resolved once; the recorder binds once and issues one indirect draw per record
/// when the host shares the bindings (CanShareIndexedBindings). The picture is the one the packets draw one by one.
/// </summary>
public sealed class IndexedRunSceneTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private const uint Width = 16;
    private const int Stripes = 4;

    // MRT0 = the four user-data words s[0:3] as floats.
    private static readonly uint[] ConstantColor =
    [
        0x7E040200,                // v_mov_b32_e32 v2, s0
        0x7E060201,                // v_mov_b32_e32 v3, s1
        0x7E080202,                // v_mov_b32_e32 v4, s2
        0x7E0A0203,                // v_mov_b32_e32 v5, s3
        0xF800180F, 0x05040302,    // exp mrt0 v2, v3, v4, v5 done vm
        0xBF810000,                // s_endpgm
    ];

    private static readonly uint[] Green = [SceneState.F(0), SceneState.F(1), SceneState.F(0), SceneState.F(1)];

    // Draws stripes 0, 1 and 3 of four vertical stripes, each through its own indexed indirect record, after one
    // learning draw of stripe 0 in a submission of its own (the translation groups only programs it has seen resolve
    // without stores). Stripe 2 is left clear.
    private byte[] DrawRun(bool runs, out (long Taken, long Draws, long Shared, long Refused) counts)
    {
        using var rig = CreateRig()!;
        rig.Host.IndexedRunsEnabled = runs;
        var vertices = new List<SceneVertex>();
        var indices = new List<ushort>();
        for (var stripe = 0; stripe < Stripes; stripe++)
        {
            var left = -1f + (2f * stripe / Stripes);
            var right = left + (2f / Stripes);
            var first = (ushort)vertices.Count;
            vertices.AddRange(
            [
                SceneVertex.At(left, -1, 0.5f, 0, 0, 0), SceneVertex.At(right, -1, 0.5f, 0, 0, 0),
                SceneVertex.At(left, 1, 0.5f, 0, 0, 0), SceneVertex.At(right, 1, 0.5f, 0, 0, 0),
            ]);
            indices.AddRange([first, (ushort)(first + 1), (ushort)(first + 2), (ushort)(first + 2), (ushort)(first + 1), (ushort)(first + 3)]);
        }

        var mesh = rig.UploadMesh(vertices);
        var indexAddress = rig.Memory.Upload<ushort>(indices.ToArray());
        // Records (indexCount, instanceCount, firstIndex, vertexOffset, firstInstance), 20 bytes apart in one page.
        var arguments = rig.Memory.Allocate(0x1000, 0x1000);
        rig.Memory.Write<uint>(arguments, [6, 1, 0, 0, 0, 6, 1, 6, 0, 0, 6, 1, 18, 0, 0]);
        var target = rig.ColorTarget(Width, Width, 16);
        var vertex = rig.Program(ScenePrograms.FetchVertex);
        var pixel = rig.Program(ConstantColor);

        ScenePm4 Begin()
        {
            var pm4 = new ScenePm4();
            new SceneState(Width, Width).ColorTarget(0, target, Width, Width, format: 14, numberType: 7)
                .VertexShader(vertex, mesh).PixelShader(pixel, Green).Emit(pm4);
            pm4.NumInstances(1);
            pm4.Raw(ScenePm4.Header(4, ScenePm4.OpSetBase), 1u, (uint)arguments, (uint)(arguments >> 32));
            pm4.IndexType(0).Packet(ScenePm4.OpIndexBase, (uint)indexAddress, (uint)(indexAddress >> 32))
                .Packet(ScenePm4.OpIndexBufferSize, (uint)indices.Count);
            return pm4;
        }

        rig.Submit(Begin().Packet(ScenePm4.OpDrawIndexIndirect, 0u, 0u, 0u, 2u));
        var run = Begin();
        foreach (var offset in new uint[] { 0, 20, 40 })
        {
            run.Packet(ScenePm4.OpDrawIndexIndirect, offset, 0u, 0u, 2u);
        }

        rig.Submit(run);
        counts = rig.Host.IndexedRunCounts;
        Output.WriteLine($"runs={runs}: taken={counts.Taken} draws={counts.Draws} shared={counts.Shared} refused={counts.Refused}");
        Assert.Empty(rig.Fatals);
        return rig.ReadImage(target, Width, Width, 16);
    }

    [Fact]
    public void ARunOfIndexedIndirectDrawsSharesItsBindingsAndDrawsEachRecord()
    {
        if (CreateRig() is not { } probe) return;
        probe.Dispose();
        var with = DrawRun(runs: true, out var counts);
        var without = DrawRun(runs: false, out var none);
        Assert.Equal(1, counts.Taken);
        Assert.Equal(3, counts.Draws);
        Assert.True(counts.Shared >= 1, "the run's bindings were not shared");
        Assert.Equal(0, counts.Refused);
        Assert.Equal(0, none.Taken);
        Assert.Equal(without, with);
        var pixels = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(with);
        for (var stripe = 0; stripe < Stripes; stripe++)
        {
            var x = (stripe * (int)Width / Stripes) + 1;
            var green = pixels[(int)(((8 * Width) + (uint)x) * 4) + 1];
            Assert.Equal(stripe == 2 ? 0f : 1f, green);
        }
    }
}
