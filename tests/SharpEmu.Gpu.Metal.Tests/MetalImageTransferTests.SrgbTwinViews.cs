// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Images;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

// Unreal samples a virtual texture physical pool through an sRGB and a linear descriptor of
// the same memory. The cache makes one image per memory, in the format of whichever
// descriptor reached it first, and the other descriptor needs a view in its own encoding.
// A compressed texture used to be created without the pixel-format-view usage, so that
// view was refused and the image bound as it was: in about half the Black Myth: Wukong runs
// the armour's base colour pool was created linear, its sRGB texels were read undecoded,
// and the gold armour came out pale silver.
public sealed partial class MetalImageTransferTests
{
    private static readonly nint SelTexturePixelFormat = MetalNative.Selector("pixelFormat");

    private static ImageDescription BlockDescription(GpuFormat format, GuestPixelFormat guestFormat)
    {
        var description = ImageDescription.Create();
        description.PixelFormat = format;
        description.GuestFormat = guestFormat;
        description.BytesPerBlock = 16;
        description.Pitch = 64;
        description.Type = GuestImageType.Color2D;
        description.Extent = new GpuExtent3D(64, 64, 1);
        return description;
    }

    [Fact]
    public void ABlockFormatWithAnSrgbTwinIsCreatedViewableAndOthersAreNot()
    {
        var bc3 = BlockDescription(GpuFormat.BC3UnormBlock, GuestPixelFormat.Bc3UNorm);
        var bc5 = BlockDescription(GpuFormat.BC5UnormBlock, GuestPixelFormat.Bc5UNorm);
        Assert.NotEqual(0u, (uint)(MetalImageTransfer.UsageOf(bc3, MtlPixelFormat.Bc3Rgba) & MtlTextureUsage.PixelFormatView));
        Assert.Equal(0u, (uint)(MetalImageTransfer.UsageOf(bc5, MtlPixelFormat.Bc5RgUnorm) & MtlTextureUsage.PixelFormatView));
        Assert.True(MetalGuestFormats.AreSrgbTwins(MtlPixelFormat.Bc3Rgba, MtlPixelFormat.Bc3RgbaSrgb));
        Assert.True(MetalGuestFormats.AreSrgbTwins(MtlPixelFormat.Bc7RgbaUnormSrgb, MtlPixelFormat.Bc7RgbaUnorm));
        Assert.False(MetalGuestFormats.AreSrgbTwins(MtlPixelFormat.Bc1Rgba, MtlPixelFormat.Bc2RgbaSrgb));
        Assert.False(MetalGuestFormats.AreSrgbTwins(MtlPixelFormat.Bc3Rgba, MtlPixelFormat.Bc3Rgba));
    }

    [Theory]
    [InlineData(GpuFormat.BC3UnormBlock, GuestPixelFormat.Bc3UNorm, GpuFormat.BC3SrgbBlock, MtlFormat.Bc3RgbaSrgb)]
    [InlineData(GpuFormat.BC3SrgbBlock, GuestPixelFormat.Bc3Srgb, GpuFormat.BC3UnormBlock, MtlFormat.Bc3Rgba)]
    [InlineData(GpuFormat.BC1RgbaUnormBlock, GuestPixelFormat.Bc1UNorm, GpuFormat.BC1RgbaSrgbBlock, MtlFormat.Bc1RgbaSrgb)]
    [InlineData(GpuFormat.BC7UnormBlock, GuestPixelFormat.Bc7UNorm, GpuFormat.BC7SrgbBlock, MtlFormat.Bc7RgbaUnormSrgb)]
    public void TheOtherEncodingOfABlockImageGetsARealView(
        GpuFormat imageFormat, GuestPixelFormat guestFormat, GpuFormat viewFormat, ulong expectedPixelFormat)
    {
        if (!RequireDevice()) return;
        using var harness = new MetalTransferHarness(Device.Value);
        var description = BlockDescription(imageFormat, guestFormat);
        var pixelFormat = MetalV2Formats.PixelFormat(imageFormat);
        var shape = new MetalTextureShape(
            pixelFormat, Mtl.TextureType2D, 64, 64, 1, 1, 1, 1,
            (ulong)MetalImageTransfer.UsageOf(description, (MtlPixelFormat)pixelFormat));
        var texture = harness.Texture(shape);
        var image = new MetalCachedImage(in description, in shape, texture);

        var request = new ImageViewDescription(
            viewFormat, GpuImageViewType.Type2D, GpuImageAspect.ColorBit, 0, 1, 0, 1, default, GpuImageUsage.SampledBit);
        var view = image.GetOrCreateView(request);
        Assert.NotEqual(texture, view);
        Assert.Equal(expectedPixelFormat, (ulong)MetalNative.Send(view, SelTexturePixelFormat));

        // A reinterpretation Metal does not allow for a compressed texture still binds the
        // image rather than raising.
        var foreign = request with { Format = GpuFormat.R32G32B32A32Uint };
        Assert.Equal(texture, image.GetOrCreateView(foreign));
        output.WriteLine($"{imageFormat} viewed as {viewFormat} -> MTLPixelFormat {expectedPixelFormat}");
    }
}
