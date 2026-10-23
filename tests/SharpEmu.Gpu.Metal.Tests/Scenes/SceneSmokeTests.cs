// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

/// <summary>The scene harness itself: one PM4 draw reaches the target through every layer.</summary>
public sealed class SceneSmokeTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private const uint Size = 64;

    [Fact]
    public void NativeIssueConsumerAndSerialIssuePathRenderTheSameMixedPm4Tape()
    {
        (byte[] Pixels, uint[] Compute, uint[] Tape)? Replay(bool split)
        {
            using var rig = CreateRig(split);
            if (rig is null) return null;
            var target = rig.ColorTarget(Size, Size);
            var buffer = rig.Memory.Allocate(64 * 4, 256);
            var state = new SceneState(Size, Size).ColorTarget(0, target, Size, Size);
            var pm4 = new ScenePm4();
            for (var index = 0; index < 8; index++)
            {
                rig.DrawMesh(pm4, state, SceneDraws.Quad(0.5f, index / 7f, 0.25f, 1f));
                rig.Dispatch(pm4, ScenePrograms.StoreGlobalId, Descriptors.Buffer(buffer, 4, 64, format: 20), 1);
                // A real WAIT_REG_MEM on the compute shader's output must commit its
                // prefix, suspend if necessary, and observe the same bytes in both modes.
                pm4.Packet(0x3C, 0x13u, (uint)buffer, (uint)(buffer >> 32), 0x5A5Au, 0xFFFFFFFFu, 0);
            }
            var tape = pm4.ToArray();
            rig.Submit(tape);
            Assert.Empty(rig.Fatals);
            return (rig.ReadImage(target, Size, Size), rig.Read<uint>(buffer, 64), tape);
        }
        var serial = Replay(false);
        if (serial is null) return;
        var split = Replay(true)!.Value;
        Assert.Equal(serial.Value.Tape, split.Tape);
        Assert.Equal(serial.Value.Pixels, split.Pixels);
        Assert.Equal(serial.Value.Compute, split.Compute);
        Assert.Equal(0x5A5Au, split.Compute[0]);
        Assert.Equal(63u ^ 0x5A5Au, split.Compute[63]);
    }

    [Fact]
    public void ATriangleFromThePacketStreamLandsOnItsPixels()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var target = rig.ColorTarget(Size, Size);
        var state = new SceneState(Size, Size).ColorTarget(0, target, Size, Size);
        // Lower-left half of the target (NDC y up): (-1,-1), (1,-1), (-1,1).
        var pm4 = rig.DrawMesh(new ScenePm4(), state,
        [
            SceneVertex.At(-1, -1, 0.5f, 1, 0, 0),
            SceneVertex.At(1, -1, 0.5f, 1, 0, 0),
            SceneVertex.At(-1, 1, 0.5f, 1, 0, 0),
        ]);
        rig.Submit(pm4);
        var pixels = rig.ReadImage(target, Size, Size);
        // Bottom-left texel is inside, top-right is outside.
        Assert.Equal(0xFF0000FFu, SceneDraws.Pixel(pixels, Size, 2, Size - 3));
        Assert.Equal(0u, SceneDraws.Pixel(pixels, Size, Size - 3, 2));
    }

    [Fact]
    public void ASampledLinearTextureLandsTexelForPixel()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var texels = new byte[Size * Size * 4];
        for (var index = 0; index < Size * Size; index++)
        {
            BitConverter.TryWriteBytes(texels.AsSpan(index * 4), 0xFF000000u | (uint)(index * 2654435761u >> 8 & 0xFFFFFF));
        }

        var image = rig.LinearImage(Size, Size, 4, texels);
        var target = rig.ColorTarget(Size, Size);
        var state = new SceneState(Size, Size).ColorTarget(0, target, Size, Size);
        uint[] userData = [.. Descriptors.Texture(image, 56, Size, Size), .. Descriptors.Sampler()];
        rig.Submit(rig.DrawMesh(new ScenePm4(), state, SceneDraws.TexturedQuad(), ScenePrograms.SampledTexture, userData));
        var pixels = rig.ReadImage(target, Size, Size);
        Assert.Equal(texels, pixels);
    }

    [Fact]
    public void DepthWrittenByTheDepthBlockReadsBackThroughATexture()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var depth = rig.Memory.Allocate(Size * Size * 4 + 0x10000, 0x10000);
        var state = new SceneState(Size, Size).ColorTarget(0, rig.ColorTarget(Size, Size), Size, Size)
            .DepthTarget(depth, Size, Size, zFormat: 3).DepthControl(test: true, write: true, CompareFunc.Always);
        var pm4 = rig.DrawMesh(new ScenePm4(), state, SceneDraws.Quad(0.25f, 1, 1, 1, right: 0f));

        var readback = rig.ColorTarget(Size, Size);
        var copy = new SceneState(Size, Size).ColorTarget(0, readback, Size, Size, format: 4, numberType: 7);
        uint[] userData = [.. Descriptors.Texture(depth, 22, Size, Size, tileMode: 24), .. Descriptors.Sampler()];
        rig.DrawMesh(pm4, copy, SceneDraws.TexturedQuad(), ScenePrograms.SampledTexture, userData);
        rig.Submit(pm4);
        var values = rig.ReadImage<float>(readback, Size, Size);
        Assert.Equal(0.25f, values[5 * Size + 3]);
        Assert.Equal(0f, values[5 * Size + Size - 3]);
    }

    [Fact]
    public void AComputeDispatchStoresEveryThread()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        const uint groups = 4;
        var buffer = rig.Memory.Allocate(groups * 64 * 4, 256);
        rig.Submit(rig.Dispatch(new ScenePm4(), ScenePrograms.StoreGlobalId, Descriptors.Buffer(buffer, 4, groups * 64, format: 20), groups));
        var values = rig.Read<uint>(buffer, (int)(groups * 64));
        for (var index = 0u; index < values.Length; index++)
        {
            Assert.Equal(index ^ 0x5A5Au, values[index]);
        }
    }
}
