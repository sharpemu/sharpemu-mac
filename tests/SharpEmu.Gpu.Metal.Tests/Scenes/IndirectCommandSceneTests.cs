// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

public sealed class IndirectCommandSceneTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private const uint Size = 32;
    private static readonly uint[] UniformColor =
    [
        0x7E000200, // v_mov_b32 v0, s0
        0x7E020201, // v_mov_b32 v1, s1
        0x7E040202, // v_mov_b32 v2, s2
        0x7E060203, // v_mov_b32 v3, s3
        0xF800180F, 0x03020100, // exp mrt0 v0, v1, v2, v3 done vm
        0xBF810000,
    ];
    private static uint[] Color(float r, float g, float b) =>
        new[] { r, g, b, 1f }.Select(BitConverter.SingleToUInt32Bits).ToArray();

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void ReexecutionPatchesArgumentsOnGpuAndPreservesDrawOrder(bool indexed, bool split)
    {
        using var rig = CreateRig(split); if (rig is null) return;
        rig.Host.IcbEnabled = true;
        var target = rig.ColorTarget(Size, Size);
        var vertexData = rig.UploadMesh(SceneDraws.Quad(0.5f, 1, 1, 1));
        var indexData = rig.Memory.Upload<ushort>([0, 1, 2, 3, 4, 5], 256);
        ScenePm4 Tape(bool alternate)
        {
            var state = new SceneState(Size, Size).ColorTarget(0, target, Size, Size)
                .VertexShader(rig.Program(ScenePrograms.FetchVertex), vertexData)
                .PixelShader(rig.Program(UniformColor), Color(0, 0, 1));
            var pm4 = new ScenePm4();
            for (var tile = 0; tile < 8; tile++)
            {
                state.ScreenScissor(tile * 4, 0, tile * 4 + 4, (int)Size).Emit(pm4);
                pm4.NumInstances(1);
                if (indexed) pm4.IndexType(0).DrawIndexed(indexData, 6, 6); else pm4.DrawAuto(6);
                var red = (tile % 2 == 0) ^ alternate;
                pm4.Shader(Sh.UserDataPs0, Color(red ? 1 : 0, red ? 0 : 1, 0));
                if (indexed) pm4.DrawIndexed(indexData, 6, 6); else pm4.DrawAuto(6);
            }
            return pm4;
        }
        var first = Tape(false);
        var changed = Tape(true);
        rig.Submit(first); rig.Submit(first); // All pipelines and recipes ready.
        var before = rig.Host.IcbCounters;
        var firstPass = rig.Host.RenderPassSerial;
        // Multiple retained banks in one pass; changed serial-local argument
        // records across frames must not rebuild their native command structures.
        for (var frame = 0; frame < 8; frame++) rig.Submit(frame % 2 == 0 ? first : changed);
        var after = rig.Host.IcbCounters;
        Assert.Equal(8, rig.Host.RenderPassSerial - firstPass);
        Output.WriteLine($"indexed={indexed} split={split} before={before} after={after} gpu_patches={rig.Host.IcbPatchedRecords}");
        Assert.True(after.Reused > before.Reused, $"No persistent reuse: {before} -> {after}");
        Assert.Equal(before.Built, after.Built);
        Assert.True(after.Draws - before.Draws >= 128, $"Draws escaped ICB: {before} -> {after}");
        Assert.True(after.Runs - before.Runs < after.Draws - before.Draws, $"Draws were not batched: {before} -> {after}");
        Assert.True(rig.Host.IcbPatchedRecords > 0, "The GPU argument patch path was not exercised.");
        var actual = rig.ReadImage(target, Size, Size);
        for (uint tile = 0; tile < 8; tile++)
            Assert.Equal(tile % 2 == 0 ? 0xFF00FF00u : 0xFF0000FFu, SceneDraws.Pixel(actual, Size, tile * 4 + 2, 16));
        rig.Host.IcbEnabled = false;
        rig.Submit(changed);
        Assert.Equal(actual, rig.ReadImage(target, Size, Size));
        Assert.Empty(rig.Fatals);
    }

    [Fact]
    public void ChangedCommandRecipeBuildsNewCommandsAndMatchesDirectPixels()
    {
        using var rig = CreateRig(); if (rig is null) return;
        rig.Host.IcbEnabled = true;
        var target = rig.ColorTarget(Size, Size);
        var state = new SceneState(Size, Size).ColorTarget(0, target, Size, Size)
            .VertexShader(rig.Program(ScenePrograms.FetchVertex), rig.UploadMesh(SceneDraws.Quad(0.5f, 1, 1, 1)))
            .PixelShader(rig.Program(UniformColor), Color(1, 0, 0));
        ScenePm4 Tape(uint count) => state.Emit(new ScenePm4()).NumInstances(1).DrawAuto(count);
        rig.Submit(Tape(6)); rig.Submit(Tape(6));
        var before = rig.Host.IcbCounters;
        // SceneMemory's raw writes do not publish image-cache CPU-store events.
        // Use fresh zeroed attachments for an independent partial-coverage oracle.
        var changedTarget = rig.ColorTarget(Size, Size);
        state.ColorTarget(0, changedTarget, Size, Size);
        rig.Submit(Tape(3));
        Assert.True(rig.Host.IcbCounters.Built > before.Built, "Changed draw arguments reused the old recipe.");
        var actual = rig.ReadImage(changedTarget, Size, Size);
        Assert.Contains(actual, b => b != 0);
        var pixels = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, uint>(actual).ToArray();
        Assert.Contains(0u, pixels);
        Assert.Contains(0xFF0000FFu, pixels);
        rig.Host.IcbEnabled = false;
        var referenceTarget = rig.ColorTarget(Size, Size);
        state.ColorTarget(0, referenceTarget, Size, Size);
        rig.Submit(Tape(3));
        Assert.Equal(actual, rig.ReadImage(referenceTarget, Size, Size));
        Assert.Empty(rig.Fatals);
    }

    [Fact]
    public void DifferentPipelinesAndBindingsStayInOneOrderedIcbRun()
    {
        using var rig = CreateRig(); if (rig is null) return;
        rig.Host.IcbEnabled = true;
        var target = rig.ColorTarget(Size, Size);
        var vertex = rig.Program(ScenePrograms.FetchVertex);
        var data = rig.UploadMesh(SceneDraws.Quad(0.5f, 0, 1, 0));
        var state = new SceneState(Size, Size).ColorTarget(0, target, Size, Size)
            .VertexShader(vertex, data).PixelShader(rig.Program(UniformColor), Color(0, 0, 1));
        var tape = state.Emit(new ScenePm4()).NumInstances(1).DrawAuto(6);
        state.PixelShader(rig.Program(ScenePrograms.InterpolatedColor)).Emit(tape).DrawAuto(6);
        rig.Submit(tape); rig.Submit(tape);
        var before = rig.Host.IcbCounters;
        var firstPass = rig.Host.RenderPassSerial;
        rig.Submit(tape);
        var after = rig.Host.IcbCounters;
        Assert.Equal(1, rig.Host.RenderPassSerial - firstPass);
        Output.WriteLine($"heterogeneous before={before} after={after}");
        Assert.Equal(2, after.Draws - before.Draws);
        Assert.Equal(1, after.Runs - before.Runs);
        Assert.Equal(before.Built, after.Built);
        Assert.True(after.Reused > before.Reused);
        var actual = rig.ReadImage(target, Size, Size);
        Assert.Equal(0xFF00FF00u, SceneDraws.Pixel(actual, Size, 16, 16));
        rig.Host.IcbEnabled = false;
        rig.Submit(tape);
        Assert.Equal(actual, rig.ReadImage(target, Size, Size));
        Assert.Empty(rig.Fatals);
    }

    [Fact]
    public void CommonBindingsCanBeInheritedAcrossFramesAndReplayCanBeDisabledIndependently()
    {
        using var rig = CreateRig(); if (rig is null) return;
        rig.Host.IcbEnabled = true;
        rig.Host.RecipeReplayEnabled = false;
        var target = rig.ColorTarget(Size, Size);
        var state = new SceneState(Size, Size).ColorTarget(0, target, Size, Size);
        var tape = rig.DrawMesh(new ScenePm4(), state, SceneDraws.Quad(0.5f, 0.25f, 0.5f, 1));
        rig.Submit(tape); rig.Submit(tape);
        var changed = rig.DrawMesh(new ScenePm4(), new SceneState(Size, Size).ColorTarget(0, target, Size, Size),
            SceneDraws.Quad(0.5f, 1, 0, 0));
        var before = rig.Host.IcbCounters;
        rig.Submit(changed);
        var after = rig.Host.IcbCounters;
        Assert.Equal(1, after.Draws - before.Draws);
        Assert.True(after.Reused > before.Reused);
        Assert.Equal(before.Built, after.Built);
        Assert.Equal(0, rig.Host.IcbPatchedRecords); // The parent supplies current relocated argument/push buffers.
        var actual = rig.ReadImage(target, Size, Size);
        Assert.Equal(0xFF0000FFu, SceneDraws.Pixel(actual, Size, 16, 16));
        rig.Host.IcbEnabled = false;
        rig.Submit(changed);
        Assert.Equal(actual, rig.ReadImage(target, Size, Size));
        Assert.Empty(rig.Fatals);
    }
}
