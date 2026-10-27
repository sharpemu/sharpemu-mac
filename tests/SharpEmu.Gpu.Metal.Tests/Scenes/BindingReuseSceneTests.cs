// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using SharpEmu.Gpu.Metal.V2;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

/// <summary>
/// Draws that bind the same textures in one command buffer reuse the ring range their
/// argument buffer was written to; a draw whose binding changed does not, and renders what it
/// binds. Every colour is read back from the target the draw rendered to.
/// </summary>
public sealed class BindingReuseSceneTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private const uint Rgba8 = 56;
    private const uint Size = 8;

    private static readonly uint[] Sample =
    [
        0xBEFC0310,                // s_mov_b32 m0, s16
        0xC8080000,                // v_interp_p1_f32_e32 v2, v0, attr0.x
        0xC8090001,                // v_interp_p2_f32_e32 v2, v1, attr0.x
        0xC80C0100,                // v_interp_p1_f32_e32 v3, v0, attr0.y
        0xC80D0101,                // v_interp_p2_f32_e32 v3, v1, attr0.y
        0xC8100200,                // v_interp_p1_f32_e32 v4, v0, attr0.z
        0xC8110201,                // v_interp_p2_f32_e32 v4, v1, attr0.z
        0xF0800F08, 0x00400802,    // image_sample v[8:11], v[2:3], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C3F70,                // s_waitcnt vmcnt(0)
        0xF800180F, 0x0B0A0908,    // exp mrt0 v8, v9, v10, v11 done vm
        0xBF810000,                // s_endpgm
    ];

    private static ulong SolidTexture(SceneRig rig, uint abgr)
    {
        var texels = new uint[Size * 64];
        Array.Fill(texels, abgr);
        var address = rig.Memory.Allocate(Size * 64 * 4, 0x10000);
        rig.Memory.Write(address, MemoryMarshal.AsBytes(texels.AsSpan()));
        return address;
    }

    private static (ulong Target, uint Pitch) DrawTextured(SceneRig rig, ScenePm4 pm4, ulong texture)
    {
        const uint width = 16;
        var target = rig.ColorTarget(width, width, 16, 0xFF);
        var state = new SceneState(width, width).ColorTarget(0, target, width, width, format: 14, numberType: 7);
        static SceneVertex V(float x, float y, float u, float v) => SceneVertex.At(x, y, 0.5f, u, v, 0, 1);
        SceneVertex[] quad =
        [
            V(-1, -1, 0, 1), V(1, -1, 1, 1), V(-1, 1, 0, 0),
            V(-1, 1, 0, 0), V(1, -1, 1, 1), V(1, 1, 1, 0),
        ];
        uint[] userData =
        [
            .. Descriptors.Texture(texture, Rgba8, Size, Size), .. Descriptors.Sampler(),
            SceneState.F(0), SceneState.F(0), SceneState.F(0), SceneState.F(0),
        ];
        rig.DrawMesh(pm4, state, quad, Sample, userData);
        return (target, width);
    }

    private static float[] Centre(SceneRig rig, (ulong Target, uint Pitch) draw)
    {
        var pixels = MemoryMarshal.Cast<byte, float>(rig.ReadImage(draw.Target, draw.Pitch, draw.Pitch, 16));
        return pixels.Slice((int)(((8 * draw.Pitch) + 8) * 4), 4).ToArray();
    }

    private static void AssertColour(string what, float[] actual, uint abgr)
    {
        float[] expected =
            [(abgr & 0xFF) / 255f, ((abgr >> 8) & 0xFF) / 255f, ((abgr >> 16) & 0xFF) / 255f, ((abgr >> 24) & 0xFF) / 255f];
        for (var channel = 0; channel < 4; channel++)
        {
            Assert.True(
                MathF.Abs(actual[channel] - expected[channel]) <= 0.01f,
                $"{what}: channel {channel} is {actual[channel]}, expected {expected[channel]}");
        }
    }

    [Fact]
    public void DrawsWithTheSameBindingsShareArgumentBuffersAndAChangedBindingIsSeen()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        const uint red = 0xFF0000FF;
        const uint green = 0xFF00FF00;
        var textures = new Dictionary<uint, ulong> { [red] = SolidTexture(rig, red), [green] = SolidTexture(rig, green) };

        // The binding changes after a reuse (red, red, green) and changes back (green, red).
        uint[] sequence = [red, red, red, green, green, red, green];
        var pm4 = new ScenePm4();
        var draws = sequence.Select(colour => DrawTextured(rig, pm4, textures[colour])).ToArray();
        var reusesBefore = rig.Host.ArgumentBufferReuses;
        rig.Submit(pm4);

        for (var index = 0; index < draws.Length; index++)
        {
            AssertColour($"draw {index}", Centre(rig, draws[index]), sequence[index]);
        }

        var reuses = rig.Host.ArgumentBufferReuses - reusesBefore;
        var writes = rig.Host.ArgumentBufferWrites;
        output.WriteLine($"{sequence.Length} draws: {reuses} argument/push ranges reused, {writes} written");
        Assert.True(reuses > 0, "no draw reused an earlier draw's argument buffer");
        Assert.True(writes > 0, "no argument buffer was written");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FixedAttachmentDrawsFollowChangedArgumentOffsetsAndEncoderRestart(bool reuse)
    {
        using var rig = CreateRig();
        if (rig is null) return;
        rig.Host.GraphicsBindingReuseEnabled = reuse;
        const uint red = 0xFF0000FF;
        const uint green = 0xFF00FF00;
        const uint width = 16;
        var redTexture = SolidTexture(rig, red);
        var greenTexture = SolidTexture(rig, green);
        var target = rig.ColorTarget(width, width, 16, 0xFF);
        var state = new SceneState(width, width).ColorTarget(0, target, width, width, format: 14, numberType: 7);
        SceneVertex[] quad =
        [
            SceneVertex.At(-1, -1, 0.5f, 0, 1, 0, 1), SceneVertex.At(1, -1, 0.5f, 1, 1, 0, 1),
            SceneVertex.At(-1, 1, 0.5f, 0, 0, 0, 1), SceneVertex.At(-1, 1, 0.5f, 0, 0, 0, 1),
            SceneVertex.At(1, -1, 0.5f, 1, 1, 0, 1), SceneVertex.At(1, 1, 0.5f, 1, 0, 0, 1),
        ];
        void Stripe(ScenePm4 stream, int left, int right, ulong texture)
        {
            state.ScreenScissor(left, 0, right, (int)width);
            uint[] data = [.. Descriptors.Texture(texture, Rgba8, Size, Size), .. Descriptors.Sampler(), 0, 0, 0, 0];
            rig.DrawMesh(stream, state, quad, Sample, data);
        }
        float[] Pixel(int x)
        {
            var pixels = MemoryMarshal.Cast<byte, float>(rig.ReadImage(target, width, width, 16));
            return pixels.Slice((int)((8 * width + (uint)x) * 4), 4).ToArray();
        }

        var pm4 = new ScenePm4();
        Stripe(pm4, 0, 4, redTexture);
        Stripe(pm4, 4, 8, redTexture); // Same fragment arguments: native state can survive.
        Stripe(pm4, 8, 12, greenTexture); // Same ring buffer, different argument content/range.
        Stripe(pm4, 12, 16, redTexture); // Return to the earlier argument range.
        rig.Submit(pm4);
        AssertColour("first red", Pixel(2), red);
        AssertColour("repeated red", Pixel(6), red);
        AssertColour("changed green", Pixel(10), green);
        AssertColour("returned red", Pixel(14), red);
        var first = rig.Host.GraphicsBindingCounters(MetalGraphicsBindingStage.Fragment);
        Assert.True(first.Attempted > 0);
        if (reuse) Assert.True(first.Skipped > 0, "No repeated fragment binding was skipped inside the fixed render pass.");
        else Assert.Equal(0L, first.Skipped);

        // Submit drained and ended the old encoder. The first draw of the next one
        // must bind every valid fragment slot even when its material was seen before.
        var restart = new ScenePm4();
        Stripe(restart, 0, 16, redTexture);
        rig.Submit(restart);
        var after = rig.Host.GraphicsBindingCounters(MetalGraphicsBindingStage.Fragment);
        Assert.True(after.Attempted > first.Attempted);
        Assert.Equal(first.Skipped, after.Skipped);
        AssertColour("restarted encoder", Pixel(8), red);
        Assert.Empty(rig.Fatals);
        output.WriteLine($"graphics binding reuse={reuse} attempted={rig.Host.GraphicsBindingAttempts} skipped={rig.Host.GraphicsBindingSkipped}");
    }

}
