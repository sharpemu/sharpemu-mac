// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

/// <summary>
/// The orderings the host leaves to Metal: dispatches in one serial compute encoder with no
/// barrier between them, and guest cache flushes inside a render pass that holds no stores.
/// </summary>
public sealed class BarrierSceneTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private const uint BufferFormat = 20;
    private const uint Size = 32;

    /// <summary>
    /// Eight dispatches with no guest flush between them, each loading through a guest pointer
    /// what the one before stored and storing the value plus 1000 one KiB further on.
    /// </summary>
    [Fact]
    public void ADispatchReadsWhatTheDispatchBeforeItStoredInTheSameEncoder()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        const uint threads = 128;
        const int steps = 8;
        var data = rig.Memory.Upload<uint>(
            Enumerable.Range(0, 256 * (steps + 1)).Select(i => i < 256 ? (uint)i * 3 : 0xFFFFFFFFu).ToArray(), 256);
        var returned = rig.Memory.Upload<uint>(Enumerable.Repeat(0xFFFFFFFFu, (int)threads).ToArray(), 256);
        var pm4 = new ScenePm4();
        for (var step = 0; step < steps; step++)
        {
            var source = data + (ulong)step * 1024;
            var state = new SceneState(1, 1);
            state.Context.Clear();
            state.Uconfig.Clear();
            state.ComputeShader(rig.Program(ComputeScenePrograms.GlobalLoadStore), 64, 1, 1,
                [Descriptors.Lo(source), Descriptors.Hi(source), 0, 0, .. Descriptors.Buffer(returned, 4, threads, BufferFormat)], true);
            state.Emit(pm4).DispatchDirect(threads / 64, 1, 1);
        }

        rig.Submit(pm4);
        for (var step = 1; step <= steps; step++)
        {
            var values = rig.Read<uint>(data + (ulong)step * 1024, (int)threads);
            for (var index = 0; index < threads; index++)
            {
                Assert.Equal((uint)(index * 3 + 1000 * step), values[index]);
            }
        }

        var last = rig.Read<uint>(returned, (int)threads);
        for (var index = 0; index < threads; index++)
        {
            Assert.Equal((uint)(index * 3 + 1000 * steps), last[index]);
        }
    }

    /// <summary>
    /// A release and an acquire between two draws into one target: the pass holds no shader
    /// stores, so the host records no barrier for them, and both draws land.
    /// </summary>
    [Fact]
    public void AGuestFlushBetweenDrawsWithoutStoresKeepsBothDraws()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var target = rig.ColorTarget(Size, Size);
        var state = new SceneState(Size, Size).ColorTarget(0, target, Size, Size);
        var before = rig.Host.GlobalBarrierCounts;
        // Lower-left half red, then the upper-right half green.
        var pm4 = rig.DrawMesh(new ScenePm4(), state,
        [
            SceneVertex.At(-1, -1, 0.5f, 1, 0, 0),
            SceneVertex.At(1, -1, 0.5f, 1, 0, 0),
            SceneVertex.At(-1, 1, 0.5f, 1, 0, 0),
        ]);
        pm4.Event(ScenePm4.EventPsPartialFlush).Event(ScenePm4.EventCacheFlushAndInvalidate);
        rig.DrawMesh(pm4, state,
        [
            SceneVertex.At(1, -1, 0.5f, 0, 1, 0),
            SceneVertex.At(1, 1, 0.5f, 0, 1, 0),
            SceneVertex.At(-1, 1, 0.5f, 0, 1, 0),
        ]);
        rig.Submit(pm4);
        var after = rig.Host.GlobalBarrierCounts;
        var pixels = rig.ReadImage(target, Size, Size);

        Assert.Equal(0xFF0000FFu, SceneDraws.Pixel(pixels, Size, 2, Size - 3));
        Assert.Equal(0xFF00FF00u, SceneDraws.Pixel(pixels, Size, Size - 3, 2));
        if (SharpEmu.Gpu.Metal.V2.MetalV2Diagnostics.ElideGlobalBarriers)
        {
            Assert.Equal(before.Recorded, after.Recorded);
            Assert.True(after.Elided > before.Elided, $"elided {before.Elided} -> {after.Elided}");
        }
    }
}
