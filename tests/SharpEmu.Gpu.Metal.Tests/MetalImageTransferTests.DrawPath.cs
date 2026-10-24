// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Images;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

public sealed partial class MetalImageTransferTests
{
    [Fact]
    public void DrawPathViewReuseKeepsSameFormatSwizzlePolicyLive()
    {
        if (!RequireDevice()) return;
        using var harness = new MetalTransferHarness(Device.Value);
        var description = ImageDescription.Create();
        description.PixelFormat = GpuFormat.D32Sfloat;
        description.Extent = new GpuExtent3D(64, 64, 1);
        var shape = new MetalTextureShape((ulong)MtlPixelFormat.Depth32Float, Mtl.TextureType2D,
            64, 64, 1, 1, 1, 1, Mtl.TextureUsageShaderRead | Mtl.TextureUsageRenderTarget);
        var texture = harness.Texture(shape);
        MetalNative.Send(texture, MetalNative.Selector("retain"));
        using var image = new MetalCachedImage(in description, in shape, texture);
        var request = new ImageViewDescription(description.PixelFormat, GpuImageViewType.Type2D,
            GpuImageAspect.DepthBit, 0, 1, 0, 1,
            new GpuComponentMapping(GpuComponentSwizzle.R, GpuComponentSwizzle.R, GpuComponentSwizzle.R, GpuComponentSwizzle.One),
            GpuImageUsage.SampledBit);
        var swizzled = image.GetOrCreateView(request, out _, sameFormatSwizzle: true);
        Assert.NotEqual(texture, swizzled);
        Assert.Equal(swizzled, image.GetOrCreateView(request, out _, sameFormatSwizzle: true));
        // Without the policy exception the original whole-texture path must still run.
        Assert.Equal(texture, image.GetOrCreateView(request, out _, sameFormatSwizzle: false));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void DrawPathViewReuseFollowsTextureReplacementAndResidentLevels(int replacement)
    {
        if (!RequireDevice()) return;
        using var harness = new MetalTransferHarness(Device.Value);
        var description = Color2D(GuestPixelFormat.Bits8_8_8_8UNorm, GpuFormat.R8G8B8A8Unorm,
            64, 64, 3, 1, GuestTileMode.Linear);
        var shape = harness.Transfer.ShapeFor(description);
        nint OwnedTexture(in MetalTextureShape requested) =>
            MetalNative.Send(harness.Texture(requested), MetalNative.Selector("retain"));
        using var image = new MetalCachedImage(in description, in shape, OwnedTexture(shape));
        var request = new ImageViewDescription(description.PixelFormat, GpuImageViewType.Type2D,
            GpuImageAspect.ColorBit, 1, 1, 0, 1, default, GpuImageUsage.SampledBit);
        var old = image.GetOrCreateView(request, out var oldId);
        Assert.Equal(old, image.GetOrCreateView(request, out var sameId));
        Assert.Equal(oldId, sameId);
        var other = image.GetOrCreateView(request with { BaseLevel = 0 });
        Assert.NotEqual(old, other);
        Assert.Equal(old, image.GetOrCreateView(request));
        MetalCachedImage? retired = null;
        var newShape = replacement == 2 ? harness.Transfer.ShapeFor(description, 1) : shape;
        var texture = OwnedTexture(newShape);
        try
        {
            switch (replacement)
            {
                case 0: image.ReplaceTexture(texture, newShape); break;
                case 1: retired = image.ReplaceWithCopy(texture, newShape, 1); break;
                case 2: retired = image.ReplaceLevels(texture, newShape, 1, 0, description.Data.Size); break;
                case 3: retired = image.PromoteToOwned(texture, newShape); break;
            }
            var current = image.GetOrCreateView(request, out var newId);
            // Metal may recycle a released view's resource ID. Its actual parent
            // proves that replacement did not reuse a stale owned view.
            Assert.NotEqual(0ul, newId);
            Assert.Equal(texture, MetalNative.Send(current, MetalNative.Selector("parentTexture")));
            Assert.Equal(32ul, (ulong)MetalNative.Send(current, MetalNative.Selector("width")));
            Assert.Equal(current, image.GetOrCreateView(request));
        }
        finally { retired?.Dispose(); }
    }
}
