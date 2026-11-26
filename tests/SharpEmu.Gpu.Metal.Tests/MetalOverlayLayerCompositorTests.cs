// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Overlay;
using Xunit;
using Xunit.Abstractions;
using Native = SharpEmu.Gpu.Metal.MetalNative;

namespace SharpEmu.Gpu.Metal.Tests;

// The in-game overlay layers on the v2 presenter: a layer's painted pixels land on the presented
// image at the layer's place, 1:1, blended premultiplied over the frame, and nothing outside it moves.
public sealed class MetalOverlayLayerCompositorTests(ITestOutputHelper output)
{
    private const int SurfaceWidth = 32;
    private const int SurfaceHeight = 16;

    // The frame under the layers: BGRA (40, 80, 120, 255).
    private const byte FrameB = 40, FrameG = 80, FrameR = 120;

    [Fact]
    public void ALayersPixelsLandAtItsPlaceBlendedOverTheFrame()
    {
        if (!RequireDevice()) return;

        // A 6x3 layer at (10, 5): an opaque row, a half-transparent row, a transparent row; each
        // row's first pixel is marked so a flip or an offset shows.
        const int left = 10, top = 5, width = 6, height = 3;
        var pixels = new byte[width * height * 4];
        for (var x = 0; x < width; x++)
        {
            Put(pixels, width, x, 0, 200, 10, 30, 255);
            Put(pixels, width, x, 1, 100, 0, 0, 128);
            Put(pixels, width, x, 2, 0, 0, 0, 0);
        }

        Put(pixels, width, 0, 0, 1, 2, 3, 255);
        var frame = new OverlayLayerFrame(0, 1, SurfaceWidth, SurfaceHeight, left, top, width, height, pixels);

        var image = Composite([frame], out var drawn);
        Assert.Equal(1, drawn);

        for (var y = 0; y < SurfaceHeight; y++)
        {
            for (var x = 0; x < SurfaceWidth; x++)
            {
                var inside = x >= left && x < left + width && y >= top && y < top + height;
                var (b, g, r, a) = Get(image, x, y);
                if (!inside)
                {
                    Assert.True((b, g, r, a) == (FrameB, FrameG, FrameR, 255), $"({x}, {y}) outside the layer changed: {(b, g, r, a)}");
                    continue;
                }

                var row = y - top;
                var column = x - left;
                var (sb, sg, sr, sa) = (pixels[(row * width + column) * 4], pixels[(row * width + column) * 4 + 1],
                    pixels[(row * width + column) * 4 + 2], pixels[(row * width + column) * 4 + 3]);
                // Premultiplied source-over: out = src + dst * (1 - src.a).
                AssertNear(sb + FrameB * (255 - sa) / 255.0, b, $"({x}, {y}) blue");
                AssertNear(sg + FrameG * (255 - sa) / 255.0, g, $"({x}, {y}) green");
                AssertNear(sr + FrameR * (255 - sa) / 255.0, r, $"({x}, {y}) red");
            }
        }

        Assert.Equal((1, 2, 3), (Get(image, left, top).B, Get(image, left, top).G, Get(image, left, top).R));
    }

    [Fact]
    public void ALayerPaintedForAnotherSurfaceWaitsForItsRepaint()
    {
        if (!RequireDevice()) return;

        var pixels = new byte[4 * 4 * 4];
        Array.Fill(pixels, (byte)255);
        var stale = new OverlayLayerFrame(0, 1, SurfaceWidth * 2, SurfaceHeight * 2, 0, 0, 4, 4, pixels);
        var image = Composite([stale], out var drawn);
        Assert.Equal(0, drawn);
        Assert.Equal((FrameB, FrameG, FrameR, (byte)255), Get(image, 0, 0));
    }

    [Fact]
    public void LayersDrawInOrderAndARepaintReplacesTheImage()
    {
        if (!RequireDevice()) return;

        var below = Solid(0, 1, 2, 2, 8, 8, 0, 0, 255, 255);
        var above = Solid(1, 1, 6, 6, 8, 8, 255, 0, 0, 255);
        var device = Device.Value;
        using var compositor = new MetalOverlayLayerCompositor(device);
        var image = Composite(compositor, [below, above], out var drawn);
        Assert.Equal(2, drawn);
        Assert.Equal(((byte)0, (byte)0, (byte)255, (byte)255), Get(image, 3, 3));
        Assert.Equal(((byte)255, (byte)0, (byte)0, (byte)255), Get(image, 7, 7));

        // The same layer repainted green: the compositor uploads the new frame, not the cached one.
        var repainted = Solid(1, 2, 6, 6, 8, 8, 0, 255, 0, 255);
        image = Composite(compositor, [below, repainted], out drawn);
        Assert.Equal(2, drawn);
        Assert.Equal(((byte)0, (byte)255, (byte)0, (byte)255), Get(image, 7, 7));
    }

    private static OverlayLayerFrame Solid(int id, long revision, int left, int top, int width, int height,
        byte b, byte g, byte r, byte a)
    {
        var pixels = new byte[width * height * 4];
        for (var index = 0; index < width * height; index++)
        {
            pixels[index * 4] = b;
            pixels[index * 4 + 1] = g;
            pixels[index * 4 + 2] = r;
            pixels[index * 4 + 3] = a;
        }

        return new OverlayLayerFrame(id, revision, SurfaceWidth, SurfaceHeight, left, top, width, height, pixels);
    }

    private static void Put(byte[] pixels, int width, int x, int y, byte b, byte g, byte r, byte a)
    {
        var offset = (y * width + x) * 4;
        pixels[offset] = b;
        pixels[offset + 1] = g;
        pixels[offset + 2] = r;
        pixels[offset + 3] = a;
    }

    private static (byte B, byte G, byte R, byte A) Get(byte[] image, int x, int y)
    {
        var offset = (y * SurfaceWidth + x) * 4;
        return (image[offset], image[offset + 1], image[offset + 2], image[offset + 3]);
    }

    private static void AssertNear(double expected, byte actual, string what) =>
        Assert.True(Math.Abs(expected - actual) <= 1.5, $"{what}: expected {expected:F1}, got {actual}");

    private static byte[] Composite(IReadOnlyList<OverlayLayerFrame> frames, out int drawn)
    {
        using var compositor = new MetalOverlayLayerCompositor(Device.Value);
        return Composite(compositor, frames, out drawn);
    }

    // Clears a BGRA8 target to the frame colour, composites the layers in the same render pass, as
    // the presenter does after drawing the frame, and reads the target back.
    private static unsafe byte[] Composite(MetalOverlayLayerCompositor compositor, IReadOnlyList<OverlayLayerFrame> frames, out int drawn)
    {
        var device = Device.Value;
        var pool = Native.objc_autoreleasePoolPush();
        nint target = 0, buffer = 0, queue = 0;
        try
        {
            var descriptor = Native.SendTextureDescriptor(
                Native.Class("MTLTextureDescriptor"),
                Native.Selector("texture2DDescriptorWithPixelFormat:width:height:mipmapped:"),
                (nuint)MtlFormat.Bgra8Unorm,
                SurfaceWidth,
                SurfaceHeight,
                mipmapped: false);
            Native.SendVoidNUInt(descriptor, Native.Selector("setUsage:"), 4);
            target = Native.Send(device, Native.Selector("newTextureWithDescriptor:"), descriptor);
            Assert.NotEqual(0, target);
            const int bytes = SurfaceWidth * SurfaceHeight * 4;
            buffer = Native.SendNewBuffer(device, Native.Selector("newBufferWithLength:options:"), bytes, 0);
            queue = Native.Send(device, Native.Selector("newCommandQueue"));
            var command = Native.Send(queue, Native.Selector("commandBuffer"));

            var pass = Native.Send(Native.Class("MTLRenderPassDescriptor"), Native.Selector("renderPassDescriptor"));
            var attachment = Native.SendAtIndex(
                Native.Send(pass, Native.Selector("colorAttachments")), Native.Selector("objectAtIndexedSubscript:"), 0);
            Native.SendVoid(attachment, Native.Selector("setTexture:"), target);
            Native.SendVoidNUInt(attachment, Native.Selector("setLoadAction:"), (nuint)Mtl.LoadActionClear);
            Native.SendVoidNUInt(attachment, Native.Selector("setStoreAction:"), (nuint)Mtl.StoreActionStore);
            Native.SendVoidClearColor(
                attachment,
                Native.Selector("setClearColor:"),
                new MtlClearColor { Red = FrameR / 255.0, Green = FrameG / 255.0, Blue = FrameB / 255.0, Alpha = 1 });
            var encoder = Native.Send(command, Native.Selector("renderCommandEncoderWithDescriptor:"), pass);
            compositor.Encode(encoder, frames, SurfaceWidth, SurfaceHeight, hdr: false, headroom: 1);
            drawn = compositor.DrawnLayers;
            Native.SendVoid(encoder, Native.Selector("endEncoding"));

            var blit = Native.Send(command, Native.Selector("blitCommandEncoder"));
            Native.SendCopyTextureToBuffer(
                blit,
                Native.Selector(
                    "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:" +
                    "destinationOffset:destinationBytesPerRow:destinationBytesPerImage:"),
                target,
                0,
                0,
                default,
                new MtlSize { Width = SurfaceWidth, Height = SurfaceHeight, Depth = 1 },
                buffer,
                0,
                SurfaceWidth * 4,
                bytes);
            Native.SendVoid(blit, Native.Selector("endEncoding"));
            Native.SendVoid(command, Native.Selector("commit"));
            Native.SendVoid(command, Native.Selector("waitUntilCompleted"));

            var result = new byte[bytes];
            new ReadOnlySpan<byte>((void*)Native.Send(buffer, Native.Selector("contents")), bytes).CopyTo(result);
            return result;
        }
        finally
        {
            foreach (var handle in new[] { target, buffer, queue })
            {
                if (handle != 0)
                {
                    Native.SendVoid(handle, Native.Selector("release"));
                }
            }

            Native.objc_autoreleasePoolPop(pool);
        }
    }

    private bool RequireDevice()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("The overlay layer compositor runs on the device only behind the exclusive device gate.");
            return false;
        }

        Assert.True(OperatingSystem.IsMacOS());
        Assert.NotEqual(0, Device.Value);
        return true;
    }

    private static readonly Lazy<nint> Device = new(Native.MTLCreateSystemDefaultDevice);
}
