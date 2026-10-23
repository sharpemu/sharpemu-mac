// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Images;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

// GPU time of the transfers Demon's Souls spends its frame on, printed for a profile
// read-out; nothing is asserted beyond the transfer completing.
public sealed partial class MetalImageTransferTests
{
    private static readonly nint SelGpuStartTime = MetalNative.Selector("GPUStartTime");
    private static readonly nint SelGpuEndTime = MetalNative.Selector("GPUEndTime");

    [Theory]
    [InlineData("depth 1024x1024x4", 0u)]
    [InlineData("rgba16f 1280x720 RT", 1u)]
    [InlineData("rgba16f 3840x2160 RT", 2u)]
    [InlineData("r32f 2560x1440 depth-tiled", 3u)]
    public void TransferGpuTime(string label, uint which)
    {
        if (!RequireDevice()) return;
        using var harness = new MetalTransferHarness(Device.Value);
        var description = which switch
        {
            0 => Depth(GuestDepthFormat.Z32Float, false, 1024, 1024, 4),
            1 => Color2D(GuestPixelFormat.Bits16_16_16_16Float, GpuFormat.R16G16B16A16Sfloat, 1280, 720, 1, 1, GuestTileMode.RenderTarget),
            2 => Color2D(GuestPixelFormat.Bits16_16_16_16Float, GpuFormat.R16G16B16A16Sfloat, 3840, 2160, 1, 1, GuestTileMode.RenderTarget),
            _ => Color2D(GuestPixelFormat.Bits32Float, GpuFormat.R32Sfloat, 2560, 1440, 1, 1, GuestTileMode.Depth),
        };
        var layers = description.Resources.Layers;
        var size = description.Data.Size;
        var guest = harness.Buffer(size);
        var texture = harness.Texture(harness.Transfer.ShapeFor(description));
        var range = new SubresourceRange(0, 1, 0, layers);
        double bestUp = double.MaxValue, bestDown = double.MaxValue;
        for (var rep = 0; rep < 5; rep++)
        {
            harness.Begin();
            var buffer = harness.CommandBuffer;
            harness.Transfer.EncodeGuestToTexture(harness, description, range,
                new MetalGuestRange(guest, 0, size, 0), default, texture, 0, size);
            MetalNative.SendVoid(buffer, MetalNative.Selector("retain"));
            harness.Submit();
            bestUp = Math.Min(bestUp, MetalNative.SendDouble(buffer, SelGpuEndTime) - MetalNative.SendDouble(buffer, SelGpuStartTime));
            MetalNative.SendVoid(buffer, MetalNative.Selector("release"));

            harness.Begin();
            buffer = harness.CommandBuffer;
            harness.Transfer.EncodeTextureToGuest(harness, description, range, texture,
                new MetalGuestRange(guest, 0, size, 0), default, 0, size);
            MetalNative.SendVoid(buffer, MetalNative.Selector("retain"));
            harness.Submit();
            bestDown = Math.Min(bestDown, MetalNative.SendDouble(buffer, SelGpuEndTime) - MetalNative.SendDouble(buffer, SelGpuStartTime));
            MetalNative.SendVoid(buffer, MetalNative.Selector("release"));
        }

        output.WriteLine($"TIMING {label}: bytes=0x{size:x} upload={bestUp * 1000:F3} ms writeback={bestDown * 1000:F3} ms");
    }
}

// GPU time of an alias transition fused on the device against the guest-memory round trip
// it replaces (a retile of the source out, a detile into the alias), for the shapes the
// fused path takes. Printed, with the fused result checked against the round trip's.
public sealed partial class MetalImageTransferTests
{
    [Theory]
    [InlineData("rgba16f 1920x1080 RT -> rg32ui view (blit)", 0u)]
    [InlineData("rgba16f 1920x1088 64KB_S -> RT (gather)", 1u)]
    [InlineData("rgba8 2048x1024 64KB_S -> rgba16f 1024x1024 RT (gather)", 2u)]
    public void AliasTransitionGpuTime(string label, uint which)
    {
        if (!RequireDevice()) return;
        using var harness = new MetalTransferHarness(Device.Value);
        var (source, target) = which switch
        {
            0 => (Color2D(GuestPixelFormat.Bits16_16_16_16Float, GpuFormat.R16G16B16A16Sfloat, 1920, 1080, 1, 1, GuestTileMode.RenderTarget),
                  Color2D(GuestPixelFormat.Bits16_16_16_16Float, GpuFormat.R32G32Uint, 1920, 1080, 1, 1, GuestTileMode.RenderTarget)),
            1 => (AliasDescription(GuestTileMode.Standard64KB, 1920, 1088, 8), AliasDescription(GuestTileMode.RenderTarget, 1920, 1088, 8)),
            _ => (AliasDescription(GuestTileMode.Standard64KB, 2048, 1024, 4), AliasDescription(GuestTileMode.RenderTarget, 1024, 1024, 8)),
        };
        Assert.Equal(source.Data.Size, target.Data.Size);
        var size = source.Data.Size;
        var guest = harness.Buffer(BuildGuest(source, 41));
        var from = harness.Texture(harness.Transfer.ShapeFor(source));
        var fused = harness.Texture(harness.Transfer.ShapeFor(target));
        var roundTrip = harness.Texture(harness.Transfer.ShapeFor(target));
        harness.Begin();
        Assert.True(harness.Transfer.EncodeGuestToTexture(harness, source, SubresourceRange.First,
            new MetalGuestRange(guest, 0, size, 0), default, from, 0, size));
        harness.Submit();
        var published = harness.Buffer(size);
        double bestFused = double.MaxValue, bestRoundTrip = double.MaxValue;
        for (var rep = 0; rep < 5; rep++)
        {
            harness.Begin();
            var buffer = harness.CommandBuffer;
            Assert.True(harness.Transfer.TryEncodeFusedAlias(harness, source, from, target, fused, out var refusal), refusal);
            MetalNative.SendVoid(buffer, MetalNative.Selector("retain"));
            harness.Submit();
            bestFused = Math.Min(bestFused, MetalNative.SendDouble(buffer, SelGpuEndTime) - MetalNative.SendDouble(buffer, SelGpuStartTime));
            MetalNative.SendVoid(buffer, MetalNative.Selector("release"));

            harness.Begin();
            buffer = harness.CommandBuffer;
            Assert.True(harness.Transfer.EncodeTextureToGuest(harness, source, SubresourceRange.First, from,
                new MetalGuestRange(published, 0, size, 0), default, 0, size));
            Assert.True(harness.Transfer.EncodeGuestToTexture(harness, target, SubresourceRange.First,
                new MetalGuestRange(published, 0, size, 0), default, roundTrip, 0, size));
            MetalNative.SendVoid(buffer, MetalNative.Selector("retain"));
            harness.Submit();
            bestRoundTrip = Math.Min(bestRoundTrip, MetalNative.SendDouble(buffer, SelGpuEndTime) - MetalNative.SendDouble(buffer, SelGpuStartTime));
            MetalNative.SendVoid(buffer, MetalNative.Selector("release"));
        }

        Assert.Equal(ReadTexels(harness, target, roundTrip), ReadTexels(harness, target, fused));
        output.WriteLine($"TIMING alias {label}: bytes=0x{size:x} fused={bestFused * 1000:F3} ms round-trip={bestRoundTrip * 1000:F3} ms");
    }
}
