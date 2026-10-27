// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
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

    private static ImageDescription TimingCase(uint which) => which switch
    {
        0 => Depth(GuestDepthFormat.Z32Float, false, 1024, 1024, 4),
        1 => Color2D(GuestPixelFormat.Bits16_16_16_16Float, GpuFormat.R16G16B16A16Sfloat, 1280, 720, 1, 1, GuestTileMode.RenderTarget),
        2 => Color2D(GuestPixelFormat.Bits16_16_16_16Float, GpuFormat.R16G16B16A16Sfloat, 3840, 2160, 1, 1, GuestTileMode.RenderTarget),
        3 => Color2D(GuestPixelFormat.Bits32Float, GpuFormat.R32Sfloat, 2560, 1440, 1, 1, GuestTileMode.Depth),
        4 => Color2D(GuestPixelFormat.Bc7UNorm, GpuFormat.BC7UnormBlock, 2048, 2048, 12, 1, GuestTileMode.Standard64KB),
        5 => Color2D(GuestPixelFormat.Bits8_8_8_8Srgb, GpuFormat.R8G8B8A8Srgb, 2048, 2048, 12, 1, GuestTileMode.Standard64KB),
        6 => Color2D(GuestPixelFormat.Bits8UNorm, GpuFormat.R8Unorm, 2048, 2048, 12, 1, GuestTileMode.Standard64KB),
        7 => Color2D(GuestPixelFormat.Bits8_8_8_8UNorm, GpuFormat.R8G8B8A8Unorm, 2048, 2048, 12, 1, GuestTileMode.Standard64KB),
        _ => Color2D(GuestPixelFormat.Bits8_8_8_8UNorm, GpuFormat.R8G8B8A8Unorm, 256, 256, 9, 1, GuestTileMode.Standard64KB),
    };

    [Theory]
    [InlineData("depth 1024x1024x4", 0u)]
    [InlineData("rgba16f 1280x720 RT", 1u)]
    [InlineData("rgba16f 3840x2160 RT", 2u)]
    [InlineData("r32f 2560x1440 depth-tiled", 3u)]
    [InlineData("bc7 2048x2048 12 mips 64KB", 4u)]
    [InlineData("rgba8 srgb 2048x2048 12 mips 64KB", 5u)]
    [InlineData("r8 2048x2048 12 mips 64KB", 6u)]
    [InlineData("rgba8 2048x2048 12 mips 64KB", 7u)]
    public void TransferGpuTime(string label, uint which)
    {
        if (!RequireDevice()) return;
        using var harness = new MetalTransferHarness(Device.Value);
        var description = TimingCase(which);
        var layers = description.Resources.Layers;
        var size = description.Data.Size;
        var guest = harness.Buffer(size);
        var texture = harness.Texture(harness.Transfer.ShapeFor(description));
        var range = new SubresourceRange(0, description.Resources.Levels, 0, layers);
        double bestUp = double.MaxValue, bestDown = double.MaxValue;
        int encodersUp = 0, encodersDown = 0;
        for (var rep = 0; rep < 5; rep++)
        {
            harness.Begin();
            var buffer = harness.CommandBuffer;
            var before = harness.EncodersStarted;
            harness.Transfer.EncodeGuestToTexture(harness, description, range,
                new MetalGuestRange(guest, 0, size, 0), default, texture, 0, size);
            encodersUp = harness.EncodersStarted - before;
            MetalNative.SendVoid(buffer, MetalNative.Selector("retain"));
            harness.Submit();
            bestUp = Math.Min(bestUp, MetalNative.SendDouble(buffer, SelGpuEndTime) - MetalNative.SendDouble(buffer, SelGpuStartTime));
            MetalNative.SendVoid(buffer, MetalNative.Selector("release"));

            harness.Begin();
            buffer = harness.CommandBuffer;
            before = harness.EncodersStarted;
            harness.Transfer.EncodeTextureToGuest(harness, description, range, texture,
                new MetalGuestRange(guest, 0, size, 0), default, 0, size);
            encodersDown = harness.EncodersStarted - before;
            MetalNative.SendVoid(buffer, MetalNative.Selector("retain"));
            harness.Submit();
            bestDown = Math.Min(bestDown, MetalNative.SendDouble(buffer, SelGpuEndTime) - MetalNative.SendDouble(buffer, SelGpuStartTime));
            MetalNative.SendVoid(buffer, MetalNative.Selector("release"));
        }

        output.WriteLine(
            $"TIMING {label}: bytes=0x{size:x} upload={bestUp * 1000:F3} ms ({encodersUp} encoders) " +
            $"writeback={bestDown * 1000:F3} ms ({encodersDown} encoders)");
    }

    // The CPU side of one transfer: what encoding it costs the thread that records it,
    // as the median of batches of uploads and writebacks into one command buffer.
    [Theory]
    [InlineData("rgba16f 1280x720 RT", 1u, false)]
    [InlineData("rgba16f 1280x720 RT", 1u, true)]
    [InlineData("rgba8 256x256 9 mips 64KB", 8u, false)]
    [InlineData("rgba8 256x256 9 mips 64KB", 8u, true)]
    [InlineData("r8 2048x2048 12 mips 64KB", 6u, false)]
    [InlineData("r8 2048x2048 12 mips 64KB", 6u, true)]
    public void TransferCpuTime(string label, uint which, bool cached)
    {
        if (!RequireDevice()) return;
        using var harness = new MetalTransferHarness(Device.Value);
        var description = TimingCase(which);
        var size = description.Data.Size;
        var guest = harness.Buffer(size);
        var texture = harness.Texture(harness.Transfer.ShapeFor(description));
        if (cached) MetalTileViews.Reference(texture);
        try
        {
            var range = new SubresourceRange(0, description.Resources.Levels, 0, description.Resources.Layers);
            const int Batch = 32;
            var up = new List<double>();
            var down = new List<double>();
            for (var rep = 0; rep < 12; rep++)
            {
                harness.Begin();
                var watch = Stopwatch.StartNew();
                for (var index = 0; index < Batch; index++)
                {
                    harness.Transfer.EncodeGuestToTexture(harness, description, range,
                        new MetalGuestRange(guest, 0, size, 0), default, texture, 0, size);
                }

                up.Add(watch.Elapsed.TotalMicroseconds / Batch);
                watch.Restart();
                for (var index = 0; index < Batch; index++)
                {
                    harness.Transfer.EncodeTextureToGuest(harness, description, range, texture,
                        new MetalGuestRange(guest, 0, size, 0), default, 0, size);
                }

                down.Add(watch.Elapsed.TotalMicroseconds / Batch);
                harness.Submit();
            }

            // The first batches pay for pipeline builds and tier-up.
            up = up.Skip(4).Order().ToList();
            down = down.Skip(4).Order().ToList();
            output.WriteLine(
                $"CPUTIME {label} cached={cached}: upload={up[up.Count / 2]:F2} us writeback={down[down.Count / 2]:F2} us per transfer " +
                $"(min {up[0]:F2} / {down[0]:F2})");
        }
        finally
        {
            if (cached) MetalTileViews.Unreference(texture);
        }
    }
}
