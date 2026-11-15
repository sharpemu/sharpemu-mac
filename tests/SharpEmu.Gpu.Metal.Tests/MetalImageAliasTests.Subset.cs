// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Images;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

// Two arrays over the same guest bytes that lay their layers out the same way (Demon's Souls' 64- and 72-layer
// 160x90 arrays on one base) hand the layers they share over by a slice copy between their textures
// (SHARPEMU_METAL_V2_ALIAS_SUBSET_DEVICE). Every case runs with the copy on and off and must end bit-identical:
// both textures read back, and guest memory after every GPU write is flushed.
public sealed unsafe partial class MetalImageAliasTests
{
    private const uint SubsetSide = 128;

    // A 128x128 RGBA8 array of the given tile mode: one 64 KiB block (or 64 KiB of linear rows) per layer.
    // A width under 128 leaves pitch padding in every row, as Demon's Souls' 160-wide arrays have (pitch 256).
    private static ImageRequest SubsetArray(
        ulong address, uint layers, GuestTileMode tile, ImageRole role, uint levels = 1, uint width = SubsetSide)
    {
        var request = Rig.ArrayRequest(address, SubsetSide, layers, role);
        var description = request.Description;
        var size = TileGeometry.TextureTotalSize(
            GuestPixelFormat.Bits8_8_8_8UNorm, width, SubsetSide, layers, levels, tile, volume: false).Size;
        description.Data = new GuestSpan(address, size);
        description.Extent = new GpuExtent3D(width, SubsetSide, 1);
        description.Pitch = TileGeometry.TexturePitch(GuestPixelFormat.Bits8_8_8_8UNorm, width, tile);
        description.TileMode = tile;
        description.Resources = new SubresourceCount(levels, layers);
        // One layer's bytes, as a colour target's description carries them: the cache then keeps two such arrays
        // over one base as two images (FindMatchingArraySlice does not match them), as it does Demon's Souls'.
        description.MipLayout[0] = new MipLevelLayout
        {
            Offset = 0, Size = size / layers, Pitch = description.Pitch, Height = SubsetSide,
        };
        return new ImageRequest(description, request.View with { LevelCount = levels }, role);
    }

    private static ulong LayerBytes(GuestTileMode tile) =>
        TileGeometry.TextureTotalSize(GuestPixelFormat.Bits8_8_8_8UNorm, SubsetSide, SubsetSide, 1, 1, tile, volume: false).Size;

    // Every layer of the image's texture, retiled to its guest layout.
    private static byte[] ReadLayers(Rig rig, MetalCachedImage image)
    {
        var description = image.Description;
        var size = description.Data.Size;
        var buffer = rig.Harness.Buffer(size);
        Assert.True(rig.Harness.Transfer.EncodeTextureToGuest(
            rig.Harness, in description, new SubresourceRange(0, description.Resources.Levels, 0, description.TransferLayers),
            image.Texture, new MetalGuestRange(buffer, 0, size, 0), default, 0, size, 0));
        rig.Harness.Submit();
        rig.Harness.Begin();
        return rig.Harness.Read(buffer, 0, size);
    }

    private sealed record SubsetOutcome(byte[] First, byte[] Second, byte[] Guest, long Copies, long Writebacks, string Text);

    // The first array is written by the GPU (its texture holds `written`, guest memory something else), then the
    // second is acquired with the given role. Then both are read back and every GPU write is flushed.
    private static SubsetOutcome RunSubset(
        bool subset, ImageRequest first, ImageRequest second, ImageRole secondRole, bool verify = false)
    {
        using var rig = new Rig(Device.Value);
        rig.Cache.AliasSubsetDeviceEnabled = subset;
        rig.Cache.AliasSubsetVerify = verify;
        var start = Math.Min(first.Description.Data.Address, second.Description.Data.Address);
        var end = Math.Max(first.Description.Data.End, second.Description.Data.End);
        var span = end - start;
        rig.WriteGuest(start, MetalTileOracle.Pattern((int)span, 0x5A));
        var firstImage = rig.Cache.GetImage(rig.Acquire(first with { Role = ImageRole.StorageImage }));
        Assert.True(firstImage.IsGpuModified);
        rig.Harness.Submit();
        rig.Harness.Begin();
        rig.WriteGuest(start, MetalTileOracle.Pattern((int)span, 0xC3));
        _ = rig.Cache.TakeAliasTransitionText();
        var writebacks = rig.Cache.Counters.Writebacks;
        var secondImage = rig.Cache.GetImage(rig.Acquire(second with { Role = secondRole }));
        rig.Harness.Submit();
        rig.Harness.Begin();
        var copies = rig.Cache.AliasSubsetCounters.Copies;
        writebacks = rig.Cache.Counters.Writebacks - writebacks;
        if (verify)
        {
            rig.Cache.DrainAliasSubsetChecks();
            var counters = rig.Cache.AliasSubsetCounters;
            Assert.Equal((copies, 0L), (counters.Verified, counters.Unverified));
            Assert.Equal(0, counters.Mismatches);
        }

        Assert.NotSame(firstImage, secondImage);
        var text = $"first_dirty={firstImage.IsCpuDirty}:0x{firstImage.DirtyStart:X}-0x{firstImage.DirtyEnd:X} " +
            $"first_gpu={firstImage.IsGpuModified} " + rig.Cache.TakeAliasTransitionText();

        // The first image read again: a write through the second left it stale over the shared layers.
        rig.Acquire(first with { Role = ImageRole.Texture });
        rig.Harness.Submit();
        rig.Harness.Begin();
        var firstBytes = ReadLayers(rig, firstImage);
        var secondBytes = ReadLayers(rig, secondImage);
        rig.Cache.FlushGpuWrites(start, span, MetalWritebackCause.HostRead);
        rig.Harness.Submit();
        rig.Harness.Begin();
        return new SubsetOutcome(firstBytes, secondBytes, rig.ReadGuest(start, span), copies, writebacks, text);
    }

    private static void AssertSameOutcome(SubsetOutcome on, SubsetOutcome off)
    {
        Assert.Equal(off.First, on.First);
        Assert.Equal(off.Second, on.Second);
        Assert.Equal(off.Guest, on.Guest);
    }

    // 8 layers written, then 9 sampled from the same base, and the reverse; in the slice-swizzled RenderTarget mode
    // (Demon's Souls' arrays), the 64 KiB standard mode and linear rows.
    [Theory]
    [InlineData(GuestTileMode.RenderTarget, 8u, 9u, ImageRole.Texture, 128u)]
    [InlineData(GuestTileMode.RenderTarget, 9u, 8u, ImageRole.StorageImage, 128u)]
    [InlineData(GuestTileMode.RenderTarget, 9u, 8u, ImageRole.Texture, 128u)]
    [InlineData(GuestTileMode.RenderTarget, 8u, 9u, ImageRole.StorageImage, 128u)]
    [InlineData(GuestTileMode.RenderTarget, 8u, 9u, ImageRole.Texture, 120u)]
    [InlineData(GuestTileMode.RenderTarget, 9u, 8u, ImageRole.StorageImage, 120u)]
    [InlineData(GuestTileMode.Standard64KB, 8u, 9u, ImageRole.Texture, 128u)]
    [InlineData(GuestTileMode.Standard64KB, 9u, 8u, ImageRole.StorageImage, 120u)]
    [InlineData(GuestTileMode.Linear, 8u, 9u, ImageRole.Texture, 128u)]
    [InlineData(GuestTileMode.Linear, 9u, 8u, ImageRole.StorageImage, 128u)]
    public void ALayerSubsetOnTheSameBaseIsCopiedOnTheDeviceAndMatchesTheRoundTrip(
        GuestTileMode tile, uint firstLayers, uint secondLayers, ImageRole role, uint width)
    {
        if (!RequireDevice()) return;
        var first = SubsetArray(ImageAddress, firstLayers, tile, ImageRole.StorageImage, width: width);
        var second = SubsetArray(ImageAddress, secondLayers, tile, role, width: width);
        var on = RunSubset(subset: true, first, second, role, verify: true);
        var off = RunSubset(subset: false, first, second, role);
        output.WriteLine($"{tile} {firstLayers}->{secondLayers} {role}: on copies={on.Copies} wb={on.Writebacks}, " +
            $"off copies={off.Copies} wb={off.Writebacks}\n on: {on.Text}\n off: {off.Text}");
        Assert.Equal(1, on.Copies);
        Assert.Equal(0, on.Writebacks);
        Assert.Equal(0, off.Copies);
        Assert.NotEqual(0, off.Writebacks);
        AssertSameOutcome(on, off);
    }

    // A two-layer image placed on layers 3 and 4 of a nine-layer array: a whole number of layers in, in a mode whose
    // swizzle does not depend on the slice index.
    [Theory]
    [InlineData(GuestTileMode.Standard64KB, ImageRole.Texture)]
    [InlineData(GuestTileMode.Standard64KB, ImageRole.StorageImage)]
    [InlineData(GuestTileMode.Linear, ImageRole.Texture)]
    [InlineData(GuestTileMode.Linear, ImageRole.StorageImage)]
    public void ASubsetAtANonZeroBaseLayerIsCopiedOnTheDeviceAndMatchesTheRoundTrip(GuestTileMode tile, ImageRole role)
    {
        if (!RequireDevice()) return;
        var first = SubsetArray(ImageAddress, 9, tile, ImageRole.StorageImage);
        var second = SubsetArray(ImageAddress + 3 * LayerBytes(tile), 2, tile, role);
        var on = RunSubset(subset: true, first, second, role, verify: true);
        var off = RunSubset(subset: false, first, second, role);
        output.WriteLine($"on: {on.Text}");
        output.WriteLine($"off: {off.Text}");
        Assert.Equal(1, on.Copies);
        Assert.Equal(0, on.Writebacks);
        Assert.Equal(0, off.Copies);
        AssertSameOutcome(on, off);
    }

    // Refused, and the round trip taken as before: a RenderTarget array a layer in (its swizzle takes the slice index,
    // so the same bytes are different texels in each), a layer offset that is not whole, and a mip chain.
    public static TheoryData<string> SubsetRefusals => ["rt-shifted", "half-layer", "mips"];

    [Theory]
    [MemberData(nameof(SubsetRefusals))]
    public void ALayerSubsetTheLayoutsDoNotShareIsRefused(string name)
    {
        if (!RequireDevice()) return;
        var (first, second) = name switch
        {
            "rt-shifted" => (SubsetArray(ImageAddress, 9, GuestTileMode.RenderTarget, ImageRole.StorageImage),
                SubsetArray(ImageAddress + LayerBytes(GuestTileMode.RenderTarget), 2, GuestTileMode.RenderTarget, ImageRole.Texture)),
            "half-layer" => (SubsetArray(ImageAddress, 9, GuestTileMode.Linear, ImageRole.StorageImage),
                SubsetArray(ImageAddress + LayerBytes(GuestTileMode.Linear) / 2, 2, GuestTileMode.Linear, ImageRole.Texture)),
            _ => (SubsetArray(ImageAddress, 4, GuestTileMode.Standard64KB, ImageRole.StorageImage, levels: 2),
                SubsetArray(ImageAddress, 2, GuestTileMode.Standard64KB, ImageRole.Texture)),
        };
        var on = RunSubset(subset: true, first, second, ImageRole.Texture);
        var off = RunSubset(subset: false, first, second, ImageRole.Texture);
        Assert.Equal(0, on.Copies);
        AssertSameOutcome(on, off);
    }

    // The planning side of the refusals, without a cache: the slice-swizzled mode matches only at the same layer index.
    [Fact]
    public void TheLayerMatchFollowsTheSliceSwizzle()
    {
        if (!RequireDevice()) return;
        using var transfer = new MetalImageTransfer(Device.Value);
        foreach (var tile in new[] { GuestTileMode.RenderTarget, GuestTileMode.Standard64KB, GuestTileMode.Linear })
        {
            var array = SubsetArray(ImageAddress, 9, tile, ImageRole.Texture).Description;
            var same = SubsetArray(ImageAddress, 8, tile, ImageRole.Texture).Description;
            var shifted = SubsetArray(ImageAddress + 3 * LayerBytes(tile), 2, tile, ImageRole.Texture).Description;
            Assert.True(transfer.TryMatchLayerSubset(in array, in same, out var sourceFirst, out var targetFirst, out var layers));
            Assert.Equal((0u, 0u, 8u), (sourceFirst, targetFirst, layers));
            Assert.True(transfer.TryMatchLayerSubset(in same, in array, out sourceFirst, out targetFirst, out layers));
            Assert.Equal((0u, 0u, 8u), (sourceFirst, targetFirst, layers));
            var matched = transfer.TryMatchLayerSubset(in array, in shifted, out sourceFirst, out targetFirst, out layers);
            Assert.Equal(tile != GuestTileMode.RenderTarget, matched);
            if (matched)
            {
                Assert.Equal((3u, 0u, 2u), (sourceFirst, targetFirst, layers));
            }
        }
    }

    // The source already stale inside the run (a one-layer alias took its layer 2): copying it would carry its old
    // bytes there, so the array is refused (the alias, a subset itself, may still be copied). Same bytes either way.
    [Fact]
    public void ASourceStaleInsideTheSubsetIsRefused()
    {
        if (!RequireDevice()) return;
        var outcomes = new List<(byte[] Second, byte[] Guest, long Copies)>();
        foreach (var subset in new[] { true, false })
        {
            using var rig = new Rig(Device.Value);
            rig.Cache.AliasSubsetDeviceEnabled = subset;
            var first = SubsetArray(ImageAddress, 9, GuestTileMode.Standard64KB, ImageRole.StorageImage);
            var size = first.Description.Data.Size;
            var layer = LayerBytes(GuestTileMode.Standard64KB);
            rig.WriteGuest(ImageAddress, MetalTileOracle.Pattern((int)size, 0x5A));
            var firstImage = rig.Cache.GetImage(rig.Acquire(first));
            rig.Harness.Submit();
            rig.Harness.Begin();
            rig.WriteGuest(ImageAddress, MetalTileOracle.Pattern((int)size, 0xC3));
            rig.Acquire(SubsetArray(ImageAddress + 2 * layer, 1, GuestTileMode.Standard64KB, ImageRole.StorageImage));
            rig.Harness.Submit();
            rig.Harness.Begin();
            Assert.True(firstImage.IsGpuModified);
            Assert.True(firstImage.IsCpuDirty);
            Assert.Equal((2 * layer, 3 * layer), (firstImage.DirtyStart, firstImage.DirtyEnd));
            var copies = rig.Cache.AliasSubsetCounters.Copies;
            var second = rig.Cache.GetImage(rig.Acquire(
                SubsetArray(ImageAddress, 8, GuestTileMode.Standard64KB, ImageRole.Texture)));
            rig.Harness.Submit();
            rig.Harness.Begin();
            copies = rig.Cache.AliasSubsetCounters.Copies - copies;
            var secondBytes = ReadLayers(rig, second);
            rig.Cache.FlushGpuWrites(ImageAddress, size, MetalWritebackCause.HostRead);
            rig.Harness.Submit();
            rig.Harness.Begin();
            outcomes.Add((secondBytes, rig.ReadGuest(ImageAddress, size), copies));
        }

        // On: one copy at most, the alias's own layer; never the array.
        Assert.True(outcomes[0].Copies <= 1);
        Assert.Equal(0, outcomes[1].Copies);
        Assert.Equal(outcomes[1].Second, outcomes[0].Second);
        Assert.Equal(outcomes[1].Guest, outcomes[0].Guest);
    }
}
