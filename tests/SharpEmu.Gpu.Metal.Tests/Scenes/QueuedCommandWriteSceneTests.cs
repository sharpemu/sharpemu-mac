// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

public sealed class QueuedCommandWriteSceneTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private static void Write(ScenePm4 stream, ulong address, params uint[] words) =>
        stream.Packet(ScenePm4.OpWriteData, [1u << 8, (uint)address, (uint)(address >> 32), .. words]);

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RewrittenIndirectPacketsAreReadAfterTheQueuedWrite(bool queued)
    {
        using var rig = CreateRig();
        if (rig is null) return;
        rig.Host.QueueCommandProcessorWrites = queued;
        var outputWord = rig.Memory.Allocate(16);
        var stale = new ScenePm4(); Write(stale, outputWord, 0xBAD);
        var child = rig.Memory.Upload<uint>(stale.ToArray());
        var parent = new ScenePm4();
        // Rewrite the packet's payload, then follow it without a GPU completion wait.
        // The parser must see the worker's pending CP overlay rather than stale RAM.
        Write(parent, child + 16, 0xC0FFEE);
        parent.Packet(0x3F, (uint)child, (uint)(child >> 32), (uint)stale.Count);
        Write(parent, outputWord + 4, 0x1234);
        rig.Submit(parent);
        Assert.Equal(new uint[] { 0xC0FFEE, 0x1234 }, rig.Read<uint>(outputWord, 2));
        Assert.Empty(rig.Fatals);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void OverlappingQueuedWritesPreserveBytesAndPredicationOrder(bool queued)
    {
        using var rig = CreateRig();
        if (rig is null) return;
        rig.Host.QueueCommandProcessorWrites = queued;
        var query = rig.Memory.Allocate(16);
        var outputWord = rig.Memory.Allocate(16);
        var stream = new ScenePm4();
        Write(stream, query, 0, 0, 0xAA, 0xBB);
        Write(stream, query + 4, 1);
        // BOOL64 is nonzero only because the second write changes the high word.
        stream.SetPredication(3, true, false, false, query);
        stream.Packet(ScenePm4.OpWriteData, true, 1u << 8, (uint)outputWord, (uint)(outputWord >> 32), 0x55);
        stream.SetPredication(0, false, false, false, 0);
        Write(stream, query, 0, 0);
        stream.SetPredication(3, true, false, false, query);
        stream.Packet(ScenePm4.OpWriteData, true, 1u << 8, (uint)outputWord, (uint)(outputWord >> 32), 0xBAD);
        stream.SetPredication(0, false, false, false, 0);
        rig.Submit(stream);
        Assert.Equal(0x55u, rig.Read<uint>(outputWord, 1)[0]);
        Assert.Equal(new uint[] { 0, 0, 0xAA, 0xBB }, rig.Read<uint>(query, 4));
        Assert.Empty(rig.Fatals);
    }
}
