// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Reflection;
using System.Runtime.InteropServices;
using SharpEmu.Gpu.Metal.V2;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

/// <summary>
/// Binding packets (MetalBindingPacket) change where the binding decode runs, not what is bound: the same stream of
/// textured draws and of dispatches that sample a texture and store into an image through a table renders the same
/// bytes with the packets taken and with them turned off, and with them off no packet is built or taken.
/// </summary>
public sealed class BindingPacketSceneTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private const uint Rgba8 = 56;
    private const uint Edge = 16;
    private const uint TargetSize = 32;

    private static readonly MethodInfo PrefetchCounters =
        typeof(MetalRenderHost).GetMethod("FormatPrefetchCounters", BindingFlags.Instance | BindingFlags.NonPublic)!;

    private static long Counter(string counters, string kind, string name)
    {
        var start = counters.IndexOf($"{kind}_packets=(", StringComparison.Ordinal);
        Assert.True(start >= 0, counters);
        var at = counters.IndexOf($"{name}=", start, StringComparison.Ordinal) + name.Length + 1;
        var end = at;
        while (char.IsAsciiDigit(counters[end])) end++;
        return long.Parse(counters.AsSpan(at, end - at));
    }

    private (ulong Checksum, string Counters)? Run(bool packets)
    {
        using var rig = CreateRig();
        if (rig is null) return null;
        rig.Host.DrawPacketsEnabled = packets;
        rig.Host.DispatchPacketsEnabled = packets;
        var images = Enumerable.Range(0, 8).Select(_ => rig.ColorTarget(Edge, Edge, 4)).ToArray();
        for (var index = 0; index < images.Length; index++)
        {
            rig.Memory.Write<uint>(images[index], Enumerable.Range(0, (int)(Edge * Edge)).Select(texel => (uint)(texel * 7 + index)).ToArray());
        }

        var target = rig.ColorTarget(TargetSize, TargetSize, 16, 0);
        var vertexProgram = rig.Program(ScenePrograms.FetchVertex);
        var pixelProgram = rig.Program(DrawRecordBenchTests.SampledThree);
        var mesh = rig.UploadMesh(SceneDraws.TexturedQuad());
        var pm4 = new ScenePm4();
        for (var round = 0; round < 3; round++)
        {
            for (var dispatch = 0; dispatch < 12; dispatch++)
            {
                var source = images[dispatch % images.Length];
                var destination = images[(dispatch + 1) % images.Length];
                var table = rig.Memory.Upload<uint>([.. Descriptors.Texture(destination, 22, Edge, Edge), Edge, Edge, Edge / 2, Edge / 2], 16);
                var state = new SceneState(1, 1);
                state.Context.Clear();
                state.Uconfig.Clear();
                state.ComputeShader(rig.Program(ComputeScenePrograms.HzbReduceMax), 8, 8, 1,
                    [.. Descriptors.Texture(source, 22, Edge, Edge), Descriptors.Lo(table), Descriptors.Hi(table)], true, true, false, 1, 0);
                state.Emit(pm4);
                pm4.DispatchDirect(1, 1, 1);
            }

            for (var draw = 0; draw < 12; draw++)
            {
                uint[] pixelData =
                [
                    .. Descriptors.Texture(images[draw % images.Length], Rgba8, Edge, Edge),
                    .. Descriptors.Texture(images[(draw + 3) % images.Length], Rgba8, Edge, Edge),
                    .. Descriptors.Texture(images[(draw + 5) % images.Length], Rgba8, Edge, Edge),
                    .. Descriptors.Sampler(),
                ];
                new SceneState(TargetSize, TargetSize)
                    .ColorTarget(0, target, TargetSize, TargetSize, format: 14, numberType: 7)
                    .VertexShader(vertexProgram, mesh)
                    .PixelShader(pixelProgram, pixelData)
                    .Emit(pm4);
                pm4.NumInstances(1);
                pm4.DrawAuto(6);
            }
        }

        _ = PrefetchCounters.Invoke(rig.Host, null);
        rig.Submit(pm4);
        var counters = (string)PrefetchCounters.Invoke(rig.Host, null)!;
        Assert.Empty(rig.Fatals);
        var checksum = 0xCBF29CE484222325UL;
        foreach (var word in images.SelectMany(image => rig.Read<uint>(image, (int)(Edge * Edge)))
                     .Concat(rig.Read<uint>(target, (int)(TargetSize * TargetSize * 4))))
        {
            checksum = (checksum ^ word) * 0x100000001B3UL;
        }

        return (checksum, counters);
    }

    [Fact]
    public void TheSameStreamBindsTheSameWithAndWithoutPackets()
    {
        if (Run(packets: false) is not { } off || Run(packets: true) is not { } on) return;
        Output.WriteLine($"off: {off.Counters}");
        Output.WriteLine($"on: {on.Counters}");
        Assert.Equal(off.Checksum, on.Checksum);
        foreach (var kind in new[] { "draw", "dispatch" })
        {
            Assert.Equal(0, Counter(off.Counters, kind, "built") + Counter(off.Counters, kind, "taken"));
            Assert.True(Counter(on.Counters, kind, "taken") > 0, on.Counters);
            Assert.Equal(0, Counter(on.Counters, kind, "program") + Counter(on.Counters, kind, "base"));
        }
    }
}
