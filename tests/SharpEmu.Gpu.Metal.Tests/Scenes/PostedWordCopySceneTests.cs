// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Rendering;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

/// <summary>
/// IRenderHost.TryPostWordCopy on Metal (upstream 61e9ec8f3/5d4937c46): the executor, resolving a bounded word copy
/// on the worker at its stream position, hands the copy to the host as a command-processor store and drops the
/// dispatch. The store lands in stream order with the writes around it, and reads before it lands see it.
/// </summary>
public sealed class PostedWordCopySceneTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private const int SourceWords = 4;
    private const int Words = 1024;

    private static uint[] Replicated(uint[] source, int words) =>
        Enumerable.Range(0, words).Select(word => source[word % source.Length]).ToArray();

    private static uint[] Words32(byte[] bytes) =>
        Enumerable.Range(0, bytes.Length / 4).Select(word => BitConverter.ToUInt32(bytes, word * 4)).ToArray();

    [Fact]
    public void APostedCopyLandsInStreamOrderBetweenTheStoresAroundIt()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var host = rig.Host;
        var source = rig.Memory.Allocate(SourceWords * 4);
        var destination = rig.Memory.Allocate(Words * 4);
        uint[] pattern = [0x11111111, 0x22222222, 0x33333333, 0x44444444];
        rig.Memory.Write<uint>(source, pattern);
        rig.Memory.Write(destination, new byte[Words * 4]);

        // A store before the copy is overwritten by it; a store after it overwrites it.
        host.RecordCommandProcessorWrite(destination, BitConverter.GetBytes(0xDEADBEEFu), "test");
        Assert.True(((IRenderHost)host).TryPostWordCopy(destination, source, SourceWords, Words));
        host.RecordCommandProcessorWrite(destination + 8, BitConverter.GetBytes(0xFEEDFACEu), "test");
        Assert.Equal(1, host.PostedWordCopies(MetalRenderHost.PostedWordCopyRefusal.None));

        var expected = Replicated(pattern, Words);
        expected[2] = 0xFEEDFACE;
        // Before the GPU ran it, a stream read sees the store as written.
        Assert.Equal(expected, Words32(rig.Read(destination, Words * 4)));

        rig.Drain();
        Assert.Equal(expected, Words32(rig.Memory.ReadRaw(destination, Words * 4)));
        Assert.Empty(rig.Fatals);
    }

    [Fact]
    public void ACopyIsRefusedAndLeftToItsDispatchWhenItsSourceIsNotFinalOrItsRangeIsUnsupported()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var host = rig.Host;
        var source = rig.Memory.Allocate(SourceWords * 4);
        var destination = rig.Memory.Allocate(Words * 4);
        rig.Memory.Write(destination, new byte[Words * 4]);

        // Source bytes a recorded store still owes: the kernel would read them on the GPU, the host cannot yet.
        host.RecordCommandProcessorWrite(source, BitConverter.GetBytes(0x12345678u), "test");
        Assert.False(((IRenderHost)host).TryPostWordCopy(destination, source, SourceWords, Words));
        Assert.Equal(1, host.PostedWordCopies(MetalRenderHost.PostedWordCopyRefusal.Input));

        // Overlapping, unaligned and oversized copies are not taken either.
        Assert.False(((IRenderHost)host).TryPostWordCopy(destination, destination + 4, SourceWords, Words));
        Assert.False(((IRenderHost)host).TryPostWordCopy(destination + 2, source, SourceWords, Words));
        Assert.False(((IRenderHost)host).TryPostWordCopy(destination, source, SourceWords, 4 * 1024 * 1024 + 1));
        Assert.Equal(3, host.PostedWordCopies(MetalRenderHost.PostedWordCopyRefusal.Unsupported));

        rig.Drain();
        // Nothing was stored for the refused copies.
        Assert.Equal(new uint[Words], Words32(rig.Memory.ReadRaw(destination, Words * 4)));
        Assert.Empty(rig.Fatals);
    }
}
