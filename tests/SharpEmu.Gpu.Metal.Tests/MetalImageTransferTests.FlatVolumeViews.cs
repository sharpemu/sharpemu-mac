// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Images;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

// The reverse of a volume's array view: a binding that names a volume over a flat texture (the null image a table's
// empty candidate binds through a volume access, as Ghost of Tsushima's cloud noise does). Metal has no 3D view of a
// 2D texture and raises an Objective-C assertion that ends the GPU process.
public sealed partial class MetalImageTransferTests
{
    [Fact]
    public void AVolumeViewOfAFlatTextureBindsTheTextureRatherThanRaising()
    {
        if (!RequireDevice()) return;
        using var harness = new MetalTransferHarness(Device.Value);
        var shape = new MetalTextureShape(
            MtlFormat.R8Unorm, Mtl.TextureType2D, 1, 1, 1, 1, 1, 1,
            Mtl.TextureUsageShaderRead | Mtl.TextureUsagePixelFormatView);
        var texture = harness.Texture(shape);
        var description = ImageDescription.Create();
        description.PixelFormat = GpuFormat.R8Unorm;
        description.BytesPerBlock = 1;
        description.Pitch = 1;
        description.Type = GuestImageType.Color2D;
        description.Extent = new GpuExtent3D(1, 1, 1);
        var image = new MetalCachedImage(in description, in shape, texture);

        var volume = new ImageViewDescription(
            GpuFormat.R8Unorm, GpuImageViewType.Type3D, GpuImageAspect.ColorBit,
            0, 1, 0, 1, default, GpuImageUsage.SampledBit);
        Assert.Equal(texture, image.GetOrCreateView(volume));
        Assert.False(image.SupportsViewType(volume));
        output.WriteLine("a volume view of a flat texture bound the texture itself");
    }
}
