// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Images;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.Images;

// A one-level image a title places on a mip-tail slot is that level of the mip-chained image: Ghost of Tsushima's
// reflection probes downsample each 256x256 RGBA16F cube face and write its 16x16 to 2x2 levels through one-level
// images at +0x800, +0x600, +0x500 and +0x400 of the face's 4 KiB tail block, where the block's swizzle puts those
// levels' first texels.
public sealed class MipTailLevelMatchTests
{
    private const ulong Base = 0x8_5000_0000;

    private static readonly ShaderImageShape Sampled = new(false, true, false, false, TextureNumericClass.Float);
    private static readonly ShaderImageShape Storage = new(false, false, true, false, TextureNumericClass.Float);

    private static ImageDescription Cube() =>
        ImageRequestBuilders.Texture(
            RegisterWords.Texture(Base, GuestPixelFormat.Bits16_16_16_16Float, 256, 256, GuestImageType.Cube,
                GuestTileMode.Standard4KB, lastLevel: 7, maxMip: 7, layers: 6),
            Sampled).Request.Description;

    private static ImageDescription Level(ulong address, uint size) =>
        ImageRequestBuilders.Texture(
            RegisterWords.Texture(address, GuestPixelFormat.Bits16_16_16_16Float, size, size, GuestImageType.Color2D,
                GuestTileMode.Standard4KB),
            Storage).Request.Description;

    [Theory]
    [InlineData(16u, 0x800u, 4)]
    [InlineData(8u, 0x600u, 5)]
    [InlineData(4u, 0x500u, 6)]
    [InlineData(2u, 0x400u, 7)]
    public void AOneLevelImageOnATailSlotIsThatLevelOfItsFace(uint size, uint slot, int level)
    {
        var cube = Cube();
        Assert.Equal(8u, cube.Resources.Levels);
        var face = cube.Data.Size / 6;
        Assert.Equal(0ul, cube.MipLayout[level].Offset);
        Assert.Equal(slot, cube.MipLayout[level].TailOffset);

        for (uint layer = 0; layer < 6; layer++)
        {
            var image = Level(Base + layer * face + slot, size);
            Assert.Equal(level, image.FindMatchingMipLevel(cube));
            Assert.Equal((int)layer, image.FindMatchingArraySlice(cube, level));
        }
    }

    [Fact]
    public void AOneLevelImageAtTheTailBlocksStartIsNotItsFirstLevel()
    {
        var cube = Cube();
        Assert.Equal(-1, Level(Base, 16).FindMatchingArraySlice(cube, 4));
    }

    [Fact]
    public void LevelsAboveTheTailKeepTheirOffsets()
    {
        var cube = Cube();
        Assert.Equal(0ul, cube.MipLayout[3].TailOffset);
        Assert.Equal(3, Level(Base + cube.MipLayout[3].Offset, 32).FindMatchingMipLevel(cube));
    }
}
