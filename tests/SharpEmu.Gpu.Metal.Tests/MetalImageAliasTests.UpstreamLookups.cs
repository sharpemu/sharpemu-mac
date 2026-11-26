// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Images;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

// Lookups ported from upstream's GuestImageCacheTests (ImageTypes 957d1dc90, BlockAlias cb790c35e).
public sealed unsafe partial class MetalImageAliasTests
{
    // A program can read a two-dimensional texture as a volume. Its image is not 3D, and an image made from the
    // texture's description would not be either: the lookup finds the same image again instead of making a new one
    // each time.
    [Fact]
    public void ATwoDimensionalTextureReadAsAVolumeIsFoundAgain()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var request = VolumeViewOfTexture(ImageAddress);
        var identifier = rig.Cache.FindImage(ref request);
        var images = rig.Cache.ImageCount;

        for (var lookup = 0; lookup < 3; lookup++)
        {
            var again = VolumeViewOfTexture(ImageAddress);
            Assert.Equal(identifier, rig.Cache.FindImage(ref again));
        }

        Assert.Equal(images, rig.Cache.ImageCount);
        Assert.False(rig.Cache.GetImage(identifier).SupportsViewType(request.View));
    }

    private static ImageRequest VolumeViewOfTexture(ulong address)
    {
        var request = Rig.Request(address, 64, ImageRole.Texture);
        request.View = request.View with { Type = GpuImageViewType.Type3D };
        return request;
    }

    // A linear 64x64 BC1 texture with two levels and a one-level 16x16 R32G32Uint image (texels as large as the chain's
    // blocks, as many as its level 0 holds), laid out as the console lays them out (TileGeometry, which the transfer
    // follows): rows padded to the linear pitch.
    private static (ulong Size, TileLevelSpan[] Levels) ChainLayout()
    {
        var levels = new TileLevelSpan[TiledSurfaceLayout.MaxLevels];
        Assert.True(TileGeometry.TryGetTextureSize(
            GuestPixelFormat.Bc1Srgb, 64, 64, 2, GuestTileMode.Linear, out var total, levels, null));
        return (total.Size, levels);
    }

    private static ulong BlockSizedImageBytes =>
        TileGeometry.TextureTotalSize(GuestPixelFormat.Bits32_32UInt, 16, 16, 1, 1, GuestTileMode.Linear, volume: false).Size;

    private static ImageRequest MippedCompressedTexture(ulong address)
    {
        var (size, levels) = ChainLayout();
        var description = ImageDescription.Create();
        description.Data = new GuestSpan(address, size);
        description.PixelFormat = GpuFormat.BC1RgbaSrgbBlock;
        description.GuestFormat = GuestPixelFormat.Bc1Srgb;
        description.Type = GuestImageType.Color2D;
        description.Extent = new GpuExtent3D(64, 64, 1);
        description.Resources = new SubresourceCount(2, 1);
        description.Pitch = TileGeometry.TexturePitch(GuestPixelFormat.Bc1Srgb, 64, GuestTileMode.Linear);
        description.BytesPerBlock = 8;
        description.Samples = 1;
        description.TileMode = GuestTileMode.Linear;
        description.MipLayout[0] = new MipLevelLayout { Offset = levels[0].Offset, Size = levels[0].Size, Pitch = description.Pitch, Height = 64 };
        description.MipLayout[1] = new MipLevelLayout { Offset = levels[1].Offset, Size = levels[1].Size, Pitch = description.Pitch / 2, Height = 32 };
        var view = ImageViewDescription.Default with
        {
            Format = GpuFormat.BC1RgbaSrgbBlock,
            Type = GpuImageViewType.Type2D,
            Aspect = GpuImageAspect.ColorBit,
            LevelCount = 2,
            LayerCount = 1,
            Usage = GpuImageUsage.SampledBit,
        };
        return new ImageRequest(description, view, ImageRole.Texture);
    }

    private static ImageRequest BlockSizedImage(ulong address)
    {
        var size = BlockSizedImageBytes;
        var description = ImageDescription.Create();
        description.Data = new GuestSpan(address, size);
        description.PixelFormat = GpuFormat.R32G32Uint;
        description.GuestFormat = GuestPixelFormat.Bits32_32UInt;
        description.Type = GuestImageType.Color2D;
        description.Extent = new GpuExtent3D(16, 16, 1);
        description.Resources = new SubresourceCount(1, 1);
        description.Pitch = TileGeometry.TexturePitch(GuestPixelFormat.Bits32_32UInt, 16, GuestTileMode.Linear);
        description.BytesPerBlock = 8;
        description.Samples = 1;
        description.TileMode = GuestTileMode.Linear;
        description.MipLayout[0] = new MipLevelLayout { Offset = 0, Size = size, Pitch = description.Pitch, Height = 16 };
        var view = ImageViewDescription.Default with
        {
            Format = GpuFormat.R32G32Uint,
            Type = GpuImageViewType.Type2D,
            Aspect = GpuImageAspect.ColorBit,
            LayerCount = 1,
            Usage = GpuImageUsage.SampledBit,
        };
        return new ImageRequest(description, view, ImageRole.Texture);
    }

    // Guest bytes whose row padding is zero, as a texture read back into guest layout leaves it: 16 rows of 16
    // eight-byte elements (texels, or a BC1 level's blocks) from offset, each row at its padded pitch.
    private static byte[] PaddedRows(int length, uint seed, ulong offset, ulong levelBytes)
    {
        var bytes = MetalTileOracle.Pattern(length, seed);
        var pitch = (int)(levelBytes / 16);
        for (var row = 0; row < 16; row++)
        {
            Array.Clear(bytes, (int)offset + (row * pitch) + (16 * 8), pitch - (16 * 8));
        }

        return bytes;
    }

    // The memory of a texture the guest gave up now holds a one-level image whose texels are as large as the texture's
    // blocks (a 16x16 R32G32 state where a 64x64 BC1 chain was). The texture is neither viewed in the other format nor
    // copied across: the image is its own, made from guest memory, and the texture stays registered beside it unless
    // it is past the removal window.
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AnImageOfTheOtherBlockClassOverAMipChainIsItsOwn(bool aged)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var (chainBytes, _) = ChainLayout();
        rig.WriteGuest(ImageAddress, MetalTileOracle.Pattern((int)chainBytes, 0x31));
        var textureId = rig.Acquire(MippedCompressedTexture(ImageAddress));
        rig.Cache.GetImage(textureId).MarkGpuModified();
        rig.Harness.Submit();
        rig.Harness.Begin();
        if (aged)
        {
            for (var tick = 0; tick < 40; tick++)
            {
                rig.Cache.Collect();
            }
        }

        // The guest's store reaches guest memory after the texture's GPU bytes there are written back.
        var state = PaddedRows((int)BlockSizedImageBytes, 0x77, 0, BlockSizedImageBytes);
        Assert.True(((IGuestImageStore)rig.Cache).MarkCpuWrite(ImageAddress, BlockSizedImageBytes));
        rig.WriteGuest(ImageAddress, state);
        var imageId = rig.Acquire(BlockSizedImage(ImageAddress));
        rig.Harness.Submit();
        rig.Harness.Begin();

        Assert.NotEqual(textureId, imageId);
        var image = rig.Cache.GetImage(imageId);
        Assert.False(image.Description.IsBlock);
        Assert.Equal(state, ReadTexture(rig, image));
        Assert.Equal(!aged, rig.Cache.TryGetImage(textureId, out var texture) && texture.Description.IsBlock);
    }

    // The other way: a compressed chain requested where a one-level image of its block size lies. The chain is its own
    // image, made from the guest bytes, not from the image's texels.
    [Fact]
    public void ACompressedChainOverAnImageOfItsBlockSizeIsMadeFromGuestMemory()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var imageId = rig.Acquire(BlockSizedImage(ImageAddress));
        rig.Harness.Submit();
        rig.Harness.Begin();

        var (chainBytes, levels) = ChainLayout();
        var blocks = PaddedRows((int)chainBytes, 0x5C, levels[0].Offset, levels[0].Size);
        Assert.True(((IGuestImageStore)rig.Cache).MarkCpuWrite(ImageAddress, chainBytes));
        rig.WriteGuest(ImageAddress, blocks);
        var textureId = rig.Acquire(MippedCompressedTexture(ImageAddress));
        rig.Harness.Submit();
        rig.Harness.Begin();

        Assert.NotEqual(imageId, textureId);
        var texture = rig.Cache.GetImage(textureId);
        Assert.True(texture.Description.IsBlock);
        var level0 = new Range((int)levels[0].Offset, (int)(levels[0].Offset + levels[0].Size));
        Assert.Equal(blocks[level0], ReadTexture(rig, texture)[level0]);
    }
}
