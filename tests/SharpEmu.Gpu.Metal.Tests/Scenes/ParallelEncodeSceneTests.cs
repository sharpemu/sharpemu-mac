// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.GpuCommands;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

public sealed class ParallelEncodeSceneTests(ITestOutputHelper output) : SceneTestBase(output)
{
    [Fact]
    public void ShortPassesMatchSerialPixelsWithoutExtraNativeChildrenPerDraw()
    {
        const uint size = 16;
        const int passes = 64;
        byte[][]? serial = null;
        foreach (var parallel in new[] { false, true })
        {
            using var rig = CreateRig();
            if (rig is null) return;
            rig.Host.ParallelEncode = parallel;
            rig.Host.ParallelWorkerCount = 4;
            var targets = new[] { rig.ColorTarget(size, size), rig.ColorTarget(size, size) };
            var pm4 = new ScenePm4();
            for (var i = 0; i < passes; i++)
            {
                var state = new SceneState(size, size).ColorTarget(0, targets[i & 1], size, size);
                rig.DrawMesh(pm4, state, SceneDraws.Quad(.5f, i / 63f, .25f, .75f));
            }
            rig.Submit(pm4);
            var pixels = targets.Select(target => rig.ReadImage(target, size, size)).ToArray();
            Assert.Empty(rig.Fatals);
            if (!parallel) serial = pixels;
            else
            {
                for (var i = 0; i < targets.Length; i++) Assert.Equal(serial![i], pixels[i]);
                Assert.Equal(passes, rig.Host.ParallelCounts.Queued);
                Assert.Equal(0, rig.Host.ParallelCounts.Batches);
                // A singleton pass needs its initial child only. The old small-
                // batch path allocated two more children for every draw.
                Assert.Equal(passes, rig.Host.ParallelNativeChildCount);
            }
        }
    }

    [Fact]
    public void ReversedChunkCompletionMatchesSerialPixelsAcrossStateChangesAndBatchRollover()
    {
        const uint size = 64;
        byte[]? serial = null;
        foreach (var parallel in new[] { false, true })
        {
            using var rig = CreateRig();
            if (rig is null) return;
            rig.Host.ParallelEncode = parallel;
            rig.Host.ParallelWorkerCount = 4;
            using var lastChildEnded = new ManualResetEventSlim(false);
            var held = 0;
            if (parallel)
            {
                rig.Host.BeforeParallelChunk = worker =>
                {
                    if (worker == 0 && Interlocked.CompareExchange(ref held, 1, 0) == 0)
                        Assert.True(lastChildEnded.Wait(TimeSpan.FromSeconds(10)));
                };
                rig.Host.AfterParallelChunk = worker => { if (worker == 3) lastChildEnded.Set(); };
            }
            var target = rig.ColorTarget(size, size);
            var textureA = rig.LinearImage(1, 1, 4, [255, 17, 93, 255]);
            var textureB = rig.LinearImage(1, 1, 4, [31, 239, 71, 255]);
            uint[][] textures =
            [
                [.. Descriptors.Texture(textureA, 56, 1, 1), .. Descriptors.Sampler()],
                [.. Descriptors.Texture(textureB, 56, 1, 1), .. Descriptors.Sampler()],
            ];
            var pm4 = new ScenePm4();
            for (var i = 0; i < 257; i++)
            {
                var tile = i % 16;
                var left = tile % 4 * 16;
                var top = tile / 4 * 16;
                var state = new SceneState(size, size).ColorTarget(0, target, size, size)
                    .ScreenScissor(left, top, left + 16, top + 16);
                if ((i & 1) != 0)
                    rig.DrawMesh(pm4, state, SceneDraws.TexturedQuad(), ScenePrograms.SampledTexture, textures[(i >> 1) & 1]);
                else
                    rig.DrawMesh(pm4, state, SceneDraws.Quad(.5f,
                        ((i * 37) % 251) / 250f, ((i * 71) % 251) / 250f, ((i * 113) % 251) / 250f));
            }
            rig.Submit(pm4);
            var pixels = rig.ReadImage(target, size, size);
            Assert.True(pixels.Any(value => value != 0));
            Assert.Empty(rig.Fatals);
            if (!parallel) serial = pixels;
            else
            {
                Assert.Equal(serial, pixels);
                Assert.True(rig.Host.ParallelCounts.Batches > 0);
                Assert.True(rig.Host.ParallelCounts.Queued > 64);
                Assert.Equal(1, held);
                Assert.True(lastChildEnded.IsSet);
            }
        }
    }

    [Fact]
    public void DrawBarrierComputeLabelAndCpuWaitKeepTheSameOrderAsTheSerialStream()
    {
        const uint size = 64;
        byte[]? serialPixels = null;
        uint[]? serialWords = null;
        foreach (var parallel in new[] { false, true })
        {
            using var rig = CreateRig();
            if (rig is null) return;
            rig.Host.ParallelEncode = parallel;
            rig.Host.ParallelWorkerCount = 4;
            var target = rig.ColorTarget(size, size);
            var label = rig.Memory.Allocate(256, 256);
            var words = rig.Memory.Allocate(64 * sizeof(uint), 256);
            var state = new SceneState(size, size).ColorTarget(0, target, size, size);
            var pm4 = new ScenePm4();
            for (var i = 0; i < 64; i++)
                rig.DrawMesh(pm4, state, SceneDraws.Quad(.5f, i / 63f, .25f, .5f));
            pm4.Event(ScenePm4.EventCacheFlushAndInvalidate);
            rig.Dispatch(pm4, ScenePrograms.StoreGlobalId, Descriptors.Buffer(words, 4, 64, format: 20), 1);
            pm4.Packet(PacketOpcode.ReleaseMemory, 0x28u | (5u << 8), 1u << 29,
                (uint)label, (uint)(label >> 32), 1, 0, 0);
            pm4.Packet(PacketOpcode.WaitRegisterMemory, 0x13u, (uint)label, (uint)(label >> 32), 1, uint.MaxValue, 0);
            for (var i = 0; i < 64; i++)
                rig.DrawMesh(pm4, state, SceneDraws.Quad(.5f, .25f, i / 63f, .75f));
            rig.Submit(pm4);
            var pixels = rig.ReadImage(target, size, size);
            var stored = rig.Read<uint>(words, 64);
            Assert.Equal(1u, rig.Read<uint>(label, 1)[0]);
            Assert.Equal(Enumerable.Range(0, 64).Select(i => (uint)i ^ 0x5A5Au), stored);
            Assert.Empty(rig.Fatals);
            if (!parallel) { serialPixels = pixels; serialWords = stored; }
            else
            {
                Assert.Equal(serialPixels, pixels);
                Assert.Equal(serialWords, stored);
                Assert.True(rig.Host.ParallelCounts.Batches > 0);
            }
        }
    }

}
