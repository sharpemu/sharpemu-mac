// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using SharpEmu.Gpu.Metal.V2;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

/// <summary>
/// A stage whose snapshot arrays were prepared before takes its texture memo entries and samplers from that
/// preparation (the stage binding plan, MetalRenderHost.StagePlan.cs). Every draw renders a stripe of one target with
/// the texture it binds, under =verify (each planned answer checked against a fresh lookup), and each case that
/// retires a plan or one of its answers must render what the full path renders.
/// </summary>
public sealed class StagePlanSceneTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private const uint Rgba8 = 56;
    private const uint Size = 8;
    private const uint Width = 16;
    private const uint Red = 0xFF0000FF;
    private const uint Green = 0xFF00FF00;

    private sealed class Scene
    {
        private readonly SceneRig _rig;
        private readonly ulong _target;
        private readonly SceneState _state;
        private readonly SceneVertex[] _quad =
        [
            SceneVertex.At(-1, -1, 0.5f, 0, 1, 0, 1), SceneVertex.At(1, -1, 0.5f, 1, 1, 0, 1),
            SceneVertex.At(-1, 1, 0.5f, 0, 0, 0, 1), SceneVertex.At(-1, 1, 0.5f, 0, 0, 0, 1),
            SceneVertex.At(1, -1, 0.5f, 1, 1, 0, 1), SceneVertex.At(1, 1, 0.5f, 1, 0, 0, 1),
        ];

        public Scene(SceneRig rig, bool plan)
        {
            _rig = rig;
            rig.Host.StagePlanEnabled = plan;
            rig.Host.StagePlanVerify = plan;
            _target = rig.ColorTarget(Width, Width, 16, 0xFF);
            _state = new SceneState(Width, Width).ColorTarget(0, _target, Width, Width, format: 14, numberType: 7);
        }

        public ulong Solid(uint abgr)
        {
            var texels = new uint[Size * 64];
            Array.Fill(texels, abgr);
            var address = _rig.Memory.Allocate(Size * 64 * 4, 0x10000);
            _rig.Memory.Write(address, MemoryMarshal.AsBytes(texels.AsSpan()));
            return address;
        }

        // One draw over columns [left, right) sampling the texture through a T# of the given extent.
        public void Stripe(ScenePm4 pm4, int left, int right, ulong texture, uint extent = Size)
        {
            _state.ScreenScissor(left, 0, right, (int)Width);
            uint[] data = [.. Descriptors.Texture(texture, Rgba8, extent, extent), .. Descriptors.Sampler()];
            _rig.DrawMesh(pm4, _state, _quad, ScenePrograms.SampledTexture, data);
        }

        public void Submit(params Action<ScenePm4>[] stripes)
        {
            var pm4 = new ScenePm4();
            foreach (var stripe in stripes) stripe(pm4);
            _rig.Submit(pm4);
        }

        public void AssertColour(int x, uint abgr)
        {
            var pixels = MemoryMarshal.Cast<byte, float>(_rig.ReadImage(_target, Width, Width, 16));
            var actual = pixels.Slice((int)((8 * Width + (uint)x) * 4), 4).ToArray();
            float[] expected =
                [(abgr & 0xFF) / 255f, ((abgr >> 8) & 0xFF) / 255f, ((abgr >> 16) & 0xFF) / 255f, ((abgr >> 24) & 0xFF) / 255f];
            for (var channel = 0; channel < 4; channel++)
            {
                Assert.True(MathF.Abs(actual[channel] - expected[channel]) <= 0.01f,
                    $"column {x}: channel {channel} is {actual[channel]}, expected {expected[channel]}");
            }
        }
    }

    // A material bound again takes the plan: its first draw stores it, the next finds its texture's memo entry
    // (made by the first draw's binding), and the ones after take the entry from the plan.
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void RepeatedMaterialsAreAnsweredFromTheirPlanAndRenderAsWithout(bool plan)
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var scene = new Scene(rig, plan);
        var red = scene.Solid(Red);
        var green = scene.Solid(Green);
        scene.Submit(
            pm4 => scene.Stripe(pm4, 0, 3, red),
            pm4 => scene.Stripe(pm4, 3, 6, red),
            pm4 => scene.Stripe(pm4, 6, 9, green),
            pm4 => scene.Stripe(pm4, 9, 12, red),
            pm4 => scene.Stripe(pm4, 12, 16, green));
        scene.AssertColour(1, Red);
        scene.AssertColour(4, Red);
        scene.AssertColour(7, Green);
        scene.AssertColour(10, Red);
        scene.AssertColour(14, Green);
        var counts = rig.Host.StagePlanCounts;
        output.WriteLine($"plan={plan} {counts}");
        if (plan)
        {
            Assert.True(counts.Hits >= 3, $"the repeated materials took {counts.Hits} plans");
            Assert.True(counts.ImageLookups >= 1, "no plan took the memo entry its first binding made");
            Assert.True(counts.Verified > 0, "no planned answer was verified");
        }
        else
        {
            Assert.Equal(0L, counts.Hits);
            Assert.Equal(0L, counts.Stores);
        }

        Assert.Empty(rig.Fatals);
    }

    // Another image over a planned texture's bytes (the same address through a T# of another extent) moves the owner
    // versions its memo stamp holds: the planned binding is resolved again and memoised as on a lookup miss.
    [Fact]
    public void AnImageAddedOverAPlannedTextureSendsItsBindingDownTheFullPath()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var scene = new Scene(rig, plan: true);
        var red = scene.Solid(Red);
        scene.Submit(
            pm4 => scene.Stripe(pm4, 0, 4, red),
            pm4 => scene.Stripe(pm4, 4, 8, red),
            pm4 => scene.Stripe(pm4, 8, 12, red));
        var before = rig.Host.StagePlanCounts;
        Assert.Equal(0L, before.StaleImages);

        scene.Submit(
            pm4 => scene.Stripe(pm4, 0, 8, red, extent: Size / 2),
            pm4 => scene.Stripe(pm4, 8, 16, red));
        var after = rig.Host.StagePlanCounts;
        output.WriteLine($"{before} -> {after}");
        Assert.True(after.Hits > before.Hits, "the original material took no plan after the other image came");
        Assert.True(after.StaleImages > before.StaleImages, "the planned binding's stamp held over a new image");
        scene.AssertColour(12, Red);
        Assert.Empty(rig.Fatals);
    }

    // Clearing the texture memo retires every plan made before it: the next binding looks its texture up again.
    [Fact]
    public void AClearedTextureMemoRetiresThePlansMadeBeforeIt()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var scene = new Scene(rig, plan: true);
        var red = scene.Solid(Red);
        var green = scene.Solid(Green);
        scene.Submit(pm4 => scene.Stripe(pm4, 0, 8, red), pm4 => scene.Stripe(pm4, 8, 16, red));
        var before = rig.Host.StagePlanCounts;
        Assert.True(before.Hits >= 1);

        rig.Host.ClearTextureMemo();
        scene.Submit(pm4 => scene.Stripe(pm4, 0, 8, red), pm4 => scene.Stripe(pm4, 8, 16, green));
        var after = rig.Host.StagePlanCounts;
        output.WriteLine($"{before} -> {after}");
        Assert.Equal(before.MemoCleared + 1, after.MemoCleared);
        Assert.Equal(before.Hits, after.Hits);
        scene.AssertColour(4, Red);
        scene.AssertColour(12, Green);
        Assert.Empty(rig.Fatals);
    }
}
