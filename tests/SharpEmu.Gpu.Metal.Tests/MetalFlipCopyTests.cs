// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Abstractions;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// SHARPEMU_METAL_V2_PRESENT_SNAPSHOT / SHARPEMU_FLIP_AHEAD: the frame a flip hands the presenter is a copy taken in the stream at the
/// flip, so a draw into the display image recorded after the flip does not reach it.
/// </summary>
public sealed class MetalFlipCopyTests(ITestOutputHelper output)
{
    private const uint Side = 64;
    private const ulong Address = MetalPassRig.GuestBase + 0x10000;

    [Fact]
    public void ADrawIntoTheDisplayImageRecordedAfterTheFlipDoesNotChangeTheCapturedFrame()
    {
        if (!RequireDevice()) return;
        using var rig = new MetalPassRig(Device.Value, 1024 * 1024);
        var host = rig.Host;
        var formats = new[] { GpuFormat.R8G8B8A8Unorm };
        var red = rig.Pipeline(formats, ["1.0f, 0.0f, 0.0f, 1.0f"]);
        var green = rig.Pipeline(formats, ["0.0f, 1.0f, 0.0f, 1.0f"]);
        var target = rig.MakeTarget(Address, GpuFormat.R8G8B8A8Unorm, Side, Side);

        using (host.BeginPreparation())
        {
            rig.Draw(red, [target]);
            host.EndRendering();
        }

        var display = host.ImageCache.GetImage(target.State.Image).Texture;
        Assert.True(host.TryCopyFlipFrame(display, out var captured));
        Assert.NotEqual(display, captured);

        // The next frame, recorded after the flip, renders into the same display image.
        using (host.BeginPreparation())
        {
            rig.Draw(green, [target]);
            host.EndRendering();
        }

        host.FlushAndWaitForGpu(MetalWaitReason.Synchronize);
        var frame = ReadTexture(Device.Value, captured);
        var next = ReadTexture(Device.Value, host.ImageCache.GetImage(target.State.Image).Texture);
        output.WriteLine($"captured=0x{BitConverter.ToUInt32(frame, 0):X8} display=0x{BitConverter.ToUInt32(next, 0):X8}");
        for (var texel = 0; texel < frame.Length; texel += 4)
        {
            Assert.Equal<byte>([0xFF, 0x00, 0x00, 0xFF], frame.AsSpan(texel, 4).ToArray());
            Assert.Equal<byte>([0x00, 0xFF, 0x00, 0xFF], next.AsSpan(texel, 4).ToArray());
        }
    }

    // SHARPEMU_METAL_V2_PRESENT_SNAPSHOT (default on): any flip, a CPU flip too, hands the presenter
    // a copy, so the guest redrawing the buffer still on screen does not reach the presented frame.
    [Fact]
    public void TheSnapshotHandsThePresenterAFrameARedrawOfTheDisplayBufferDoesNotChange()
    {
        if (!RequireDevice()) return;
        if (!MetalV2Diagnostics.PresentSnapshot)
        {
            output.WriteLine("SHARPEMU_METAL_V2_PRESENT_SNAPSHOT=0: flips are presented live.");
            return;
        }

        using var rig = new MetalPassRig(Device.Value, 1024 * 1024);
        var host = rig.Host;
        var formats = new[] { GpuFormat.R8G8B8A8Unorm };
        var red = rig.Pipeline(formats, ["1.0f, 0.0f, 0.0f, 1.0f"]);
        var green = rig.Pipeline(formats, ["0.0f, 1.0f, 0.0f, 1.0f"]);
        var target = rig.MakeTarget(Address, GpuFormat.R8G8B8A8Unorm, Side, Side);

        using (host.BeginPreparation())
        {
            rig.Draw(red, [target]);
            host.EndRendering();
        }

        var display = host.ImageCache.GetImage(target.State.Image).Texture;
        var presented = host.FlipFrameForPresenter(display, gpuQueued: false, out var copied);
        Assert.True(copied);
        Assert.NotEqual(display, presented);

        using (host.BeginPreparation())
        {
            rig.Draw(green, [target]);
            host.EndRendering();
        }

        host.FlushAndWaitForGpu(MetalWaitReason.Synchronize);
        var frame = ReadTexture(Device.Value, presented);
        var next = ReadTexture(Device.Value, host.ImageCache.GetImage(target.State.Image).Texture);
        output.WriteLine($"presented=0x{BitConverter.ToUInt32(frame, 0):X8} display=0x{BitConverter.ToUInt32(next, 0):X8}");
        for (var texel = 0; texel < frame.Length; texel += 4)
        {
            Assert.Equal<byte>([0xFF, 0x00, 0x00, 0xFF], frame.AsSpan(texel, 4).ToArray());
            Assert.Equal<byte>([0x00, 0xFF, 0x00, 0xFF], next.AsSpan(texel, 4).ToArray());
        }
    }

    private static byte[] ReadTexture(nint device, nint texture)
    {
        const nuint bytes = Side * Side * 4;
        var queue = MetalNative.Send(device, MetalNative.Selector("newCommandQueue"));
        var buffer = MetalNative.SendNewBuffer(device, MetalNative.Selector("newBufferWithLength:options:"), bytes, 0);
        try
        {
            var commandBuffer = MetalNative.Send(queue, MetalNative.Selector("commandBuffer"));
            var blit = MetalNative.Send(commandBuffer, MetalNative.Selector("blitCommandEncoder"));
            MetalNative.SendCopyTextureToBufferOptions(
                blit,
                MetalNative.Selector(
                    "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:" +
                    "toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:options:"),
                texture, 0, 0, default, new MtlSize { Width = Side, Height = Side, Depth = 1 },
                buffer, 0, Side * 4, bytes, 0);
            MetalNative.SendVoid(blit, MetalNative.Selector("endEncoding"));
            MetalNative.SendVoid(commandBuffer, MetalNative.Selector("commit"));
            MetalNative.SendVoid(commandBuffer, MetalNative.Selector("waitUntilCompleted"));
            var result = new byte[bytes];
            Marshal.Copy(MetalNative.Send(buffer, MetalNative.Selector("contents")), result, 0, result.Length);
            return result;
        }
        finally
        {
            MetalNative.SendVoid(buffer, MetalNative.Selector("release"));
            MetalNative.SendVoid(queue, MetalNative.Selector("release"));
        }
    }

    private bool RequireDevice()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("The flip copy is encoded on the device only behind the exclusive device gate.");
            return false;
        }

        Assert.True(OperatingSystem.IsMacOS());
        Assert.NotEqual(0, Device.Value);
        return true;
    }

    private static readonly Lazy<nint> Device = new(MetalNative.MTLCreateSystemDefaultDevice);
}
