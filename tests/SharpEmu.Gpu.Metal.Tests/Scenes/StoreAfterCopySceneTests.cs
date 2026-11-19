// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

/// <summary>
/// Upstream f484338f5 ("block CPU reads of earlier GPU writes before a command-stream store lands"): a store the guest
/// polls after a command-processor copy tells it the copy is done, so a CPU read that sees the store must see the
/// copied bytes. The Vulkan host defers the read protection of GPU-written pages and applies it before the store; the
/// Metal host has no such step to defer, because the copy writes guest memory itself, in stream order ahead of the
/// store. This pins that order as the guest observes it: raw guest memory, polled while the GPU runs the stream.
/// </summary>
public sealed class StoreAfterCopySceneTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private const uint Words = 256;

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AStoreAfterACopyIsNeverSeenBeforeTheCopiedBytes(bool queued)
    {
        using var rig = CreateRig();
        if (rig is null) return;
        rig.Host.QueueCommandProcessorWrites = queued;
        var source = rig.Memory.Allocate(Words * 4);
        var destination = rig.Memory.Allocate(Words * 4);
        var flag = rig.Memory.Allocate(16);
        var pattern = Enumerable.Range(0, (int)Words).Select(word => 0xC0DE0000u | (uint)word).ToArray();
        rig.Memory.Write<uint>(source, pattern);
        rig.Memory.Write(destination, new byte[Words * 4]);
        rig.Memory.Write32(flag, 0);

        var stream = new ScenePm4()
            // DMA_DATA, memory to memory, the whole source to the destination.
            .Packet(0x50, 0u, (uint)source, (uint)(source >> 32), (uint)destination, (uint)(destination >> 32), Words * 4)
            // WRITE_DATA of the flag the guest polls.
            .Packet(ScenePm4.OpWriteData, 1u << 8, (uint)flag, (uint)(flag >> 32), 1u);
        rig.Enqueue(stream.ToArray());
        rig.RecordPending();

        var deadline = Environment.TickCount64 + 10_000;
        while (BitConverter.ToUInt32(rig.Memory.ReadRaw(flag, 4)) != 1)
        {
            Assert.True(Environment.TickCount64 < deadline, "The store never landed.");
            rig.Host.FlushAndWaitForGpu();
            rig.Host.PollGpuCompletions();
        }

        var copied = rig.Memory.ReadRaw(destination, (int)Words * 4);
        Assert.Equal(pattern, Enumerable.Range(0, (int)Words).Select(word => BitConverter.ToUInt32(copied, word * 4)));
        rig.Drain();
        Assert.Empty(rig.Fatals);
    }
}
