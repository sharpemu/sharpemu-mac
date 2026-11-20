// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Rendering;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

/// <summary>
/// IRenderHost.TryPostWordCopy on Metal (upstream 61e9ec8f3/5d4937c46): the executor, resolving a bounded word copy
/// on the worker at its stream position, hands the copy to the host and drops the dispatch. By default the host stores
/// it at once when no GPU work can touch the destination and refuses it as busy otherwise; with the store class off it
/// is an ordered command-processor store that lands in stream order with the writes around it, read as written before.
/// </summary>
public sealed class PostedWordCopySceneTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private const int SourceWords = 4;
    private const int Words = 1024;

    private static uint[] Replicated(uint[] source, int words) =>
        Enumerable.Range(0, words).Select(word => source[word % source.Length]).ToArray();

    private static uint[] Words32(byte[] bytes) =>
        Enumerable.Range(0, bytes.Length / 4).Select(word => BitConverter.ToUInt32(bytes, word * 4)).ToArray();

    // The held class (SHARPEMU_METAL_V2_POSTED_WORD_COPY_HOST_STORE=0): every copy is an ordered command-processor write.
    [Fact]
    public void APostedCopyLandsInStreamOrderBetweenTheStoresAroundIt()
    {
        var saved = MetalRenderHost.PostedWordCopyHostStore;
        MetalRenderHost.PostedWordCopyHostStore = false;
        try
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
        finally
        {
            MetalRenderHost.PostedWordCopyHostStore = saved;
        }
    }

    // The store class (default): with no GPU work recorded or in flight the copy is stored at once and logged, so a
    // look-ahead resolution that read the bytes before it does not stand; with work recorded it is refused as busy and
    // left to its dispatch, which stores nothing here.
    [Fact]
    public void AnIdleGpuTakesTheCopyAsAHostStoreAndABusyOneLeavesItToItsDispatch()
    {
        var saved = MetalRenderHost.PostedWordCopyHostStore;
        MetalRenderHost.PostedWordCopyHostStore = true;
        try
        {
            using var rig = CreateRig();
            if (rig is null) return;
            var host = rig.Host;
            var source = rig.Memory.Allocate(SourceWords * 4);
            var destination = rig.Memory.Allocate(Words * 4);
            var busyDestination = rig.Memory.Allocate(Words * 4);
            uint[] pattern = [0x11111111, 0x22222222, 0x33333333, 0x44444444];
            rig.Memory.Write<uint>(source, pattern);
            rig.Memory.Write(destination, new byte[Words * 4]);
            rig.Memory.Write(busyDestination, new byte[Words * 4]);
            rig.Drain();

            var concurrent = (SharpEmu.Libs.Gpu.Rendering.IConcurrentResolveHost)host;
            concurrent.BeginConcurrentResolve();
            object assumptions;
            try
            {
                Assert.True(host.TryReadGuestWord(destination + 8, out var before));
                Assert.Equal(0u, before);
            }
            finally
            {
                assumptions = concurrent.EndConcurrentResolve();
            }

            try
            {
                Assert.True(concurrent.IsStillCurrent(assumptions));
                Assert.True(((IRenderHost)host).TryPostWordCopy(destination, source, SourceWords, Words));
                Assert.Equal(1, host.PostedWordCopyHostStores);
                // Stored now, with nothing submitted, and the resolution that read the old bytes no longer stands.
                Assert.Equal(Replicated(pattern, Words), Words32(rig.Memory.ReadRaw(destination, Words * 4)));
                Assert.False(concurrent.IsStillCurrent(assumptions));
            }
            finally
            {
                concurrent.ReleaseAssumptions(assumptions);
            }

            // Recorded work that could still read the destination: refused, nothing stored.
            host.RecordCommandProcessorWrite(rig.Memory.Allocate(sizeof(uint)), BitConverter.GetBytes(0xDEADBEEFu), "test");
            Assert.False(((IRenderHost)host).TryPostWordCopy(busyDestination, source, SourceWords, Words));
            Assert.Equal(1, host.PostedWordCopies(MetalRenderHost.PostedWordCopyRefusal.Busy));
            Assert.Equal(1, host.PostedWordCopies(MetalRenderHost.PostedWordCopyRefusal.None));
            rig.Drain();
            Assert.Equal(new uint[Words], Words32(rig.Memory.ReadRaw(busyDestination, Words * 4)));
            Assert.Empty(rig.Fatals);
        }
        finally
        {
            MetalRenderHost.PostedWordCopyHostStore = saved;
        }
    }

    [Fact]
    public void ACopyIsRefusedAndLeftToItsDispatchWhenItsSourceIsNotFinalOrItsRangeIsUnsupported()
    {
        // The held class: under the store class a recorded write ahead of the copy refuses it as busy first.
        var saved = MetalRenderHost.PostedWordCopyHostStore;
        MetalRenderHost.PostedWordCopyHostStore = false;
        try
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
        finally
        {
            MetalRenderHost.PostedWordCopyHostStore = saved;
        }
    }

    // Upstream 90d306cb1: the image refusal asks whether an image's bytes lie in the destination, not whether an image
    // shares its pages. A copy into the bytes right after a render target, on the target's last page, is posted; one
    // whose first word is the target's last is refused; with the exact check off, both are refused.
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void OnlyACopyThatMeetsAnImagesBytesIsRefusedForTheImage(bool exact)
    {
        // The image refusal comes before the class choice; the held class posts the copy whatever the GPU is doing.
        var saved = MetalRenderHost.PostedWordCopyExactImageBytes;
        var savedStore = MetalRenderHost.PostedWordCopyHostStore;
        MetalRenderHost.PostedWordCopyExactImageBytes = exact;
        MetalRenderHost.PostedWordCopyHostStore = false;
        try
        {
            using var rig = CreateRig();
            if (rig is null) return;
            var host = rig.Host;
            var cache = host.ImageCache;
            const uint Side = 60;
            var target = rig.Memory.Allocate(0x20000, 0x10000);
            rig.Memory.Write(target, new byte[0x20000]);
            var state = new SceneState(Side, Side).ColorTarget(0, target, Side, Side);
            SceneVertex[] quad =
            [
                SceneVertex.At(-1, -1, 0.5f, 1, 0, 0, 1), SceneVertex.At(1, -1, 0.5f, 1, 0, 0, 1),
                SceneVertex.At(-1, 1, 0.5f, 1, 0, 0, 1), SceneVertex.At(-1, 1, 0.5f, 1, 0, 0, 1),
                SceneVertex.At(1, -1, 0.5f, 1, 0, 0, 1), SceneVertex.At(1, 1, 0.5f, 1, 0, 0, 1),
            ];
            var draw = new ScenePm4();
            rig.DrawMesh(draw, state, quad);
            rig.Submit(draw);

            // Where the target's bytes end: the first word after them that no image holds.
            var end = target;
            while (cache.HasImageBytes(end, sizeof(uint)))
            {
                end += sizeof(uint);
            }

            const int CopyWords = 16;
            const ulong CopyBytes = CopyWords * sizeof(uint);
            Assert.True(end > target && end < target + 0x10000);
            // The boundary: the bytes after the target share its page, the bytes one word earlier hold its last word.
            Assert.True(cache.MayOwn(end, CopyBytes));
            Assert.False(cache.HasImageBytes(end, CopyBytes));
            Assert.True(cache.HasImageBytes(end - sizeof(uint), CopyBytes));

            var source = rig.Memory.Allocate(SourceWords * 4);
            uint[] pattern = [0xC0FFEE01, 0xC0FFEE02, 0xC0FFEE03, 0xC0FFEE04];
            rig.Memory.Write<uint>(source, pattern);

            Assert.False(((IRenderHost)host).TryPostWordCopy(end - sizeof(uint), source, SourceWords, CopyWords));
            Assert.Equal(exact, ((IRenderHost)host).TryPostWordCopy(end, source, SourceWords, CopyWords));
            Assert.Equal(exact ? 1 : 2, host.PostedWordCopies(MetalRenderHost.PostedWordCopyRefusal.Image));
            Assert.Equal(exact ? 1 : 0, host.PostedWordCopies(MetalRenderHost.PostedWordCopyRefusal.None));

            // The target reads as drawn (its first texel opaque; its last word is pitch padding the draw leaves), and the
            // posted bytes beside it survive its read-back.
            rig.Drain();
            var expected = exact ? Replicated(pattern, CopyWords) : new uint[CopyWords];
            Assert.Equal(0xFF, rig.Read(target, sizeof(uint))[3]);
            Assert.Equal(expected, Words32(rig.Read(end, (int)CopyBytes)));
            Assert.Equal(expected, Words32(rig.Memory.ReadRaw(end, (int)CopyBytes)));
            Assert.Empty(rig.Fatals);
        }
        finally
        {
            MetalRenderHost.PostedWordCopyExactImageBytes = saved;
            MetalRenderHost.PostedWordCopyHostStore = savedStore;
        }
    }
}
