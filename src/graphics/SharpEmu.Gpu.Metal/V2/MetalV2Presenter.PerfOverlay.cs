// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using SharpEmu.Libs.VideoOut;

namespace SharpEmu.Gpu.Metal.V2;

// The on-screen performance display. PerfOverlay rasterizes its panel into a BGRA
// buffer; this uploads it to a small texture and draws it as the last quad of the
// present pass, clipped to the rectangle PerfOverlay.GetRectangle asks for. The
// buffer is refreshed at most every 100 ms, so the cost per present is one draw.
internal static partial class MetalV2Presenter
{
    private const long OverlayRefreshMilliseconds = 100;

    private static nint _overlayTexture;
    private static nint _overlaySampler;
    private static byte[]? _overlayPixels;
    private static long _overlayFilledAt;
    private static int _overlayFilledHeight;
    private static bool _overlayDrawn;

    // True when an unchanged guest frame must still be presented again: the panel is due a
    // refresh, or it was just shown or hidden.
    private static bool PerfOverlayNeedsPresent()
    {
        var shown = PerfOverlay.Enabled && PerfOverlay.DrawOnScreen;
        if (shown != _overlayDrawn)
        {
            return true;
        }

        return shown &&
            (Stopwatch.GetTimestamp() - _overlayFilledAt) * 1000 >= OverlayRefreshMilliseconds * Stopwatch.Frequency;
    }

    // Runs on the window thread, inside the present encoder, after the frame is drawn.
    private static unsafe void EncodePerfOverlay(nint encoder, double contentsScale)
    {
        _overlayDrawn = false;
        if (!PerfOverlay.Enabled || !PerfOverlay.DrawOnScreen || _drawableWidth < 1 || _drawableHeight < 1)
        {
            return;
        }

        if (_overlayTexture == 0)
        {
            var descriptor = MetalNative.SendTextureDescriptor(
                MetalNative.Class("MTLTextureDescriptor"),
                MetalNative.Selector("texture2DDescriptorWithPixelFormat:width:height:mipmapped:"),
                (nuint)PixelFormatBgra8Unorm,
                PerfOverlay.PixelBufferWidth,
                PerfOverlay.PixelBufferHeight,
                mipmapped: false);
            _overlayTexture = MetalNative.Send(
                _device, MetalNative.Selector("newTextureWithDescriptor:"), descriptor);
            if (_overlayTexture == 0)
            {
                return;
            }

            // Nearest filtering is the sampler default; it keeps the glyph edges crisp when
            // the panel is drawn at the display's pixel scale.
            var samplerDescriptor = MetalNative.Send(
                MetalNative.Send(MetalNative.Class("MTLSamplerDescriptor"), MetalNative.Selector("alloc")),
                MetalNative.Selector("init"));
            _overlaySampler = MetalNative.Send(
                _device, MetalNative.Selector("newSamplerStateWithDescriptor:"), samplerDescriptor);
            MetalNative.SendVoid(samplerDescriptor, MetalNative.Selector("release"));
            if (_overlaySampler == 0)
            {
                ReleasePerfOverlayResources();
                return;
            }
            _overlayFilledAt = 0;
        }

        var now = Stopwatch.GetTimestamp();
        var displayHeight = PerfOverlay.DisplayHeight;
        if (_overlayFilledAt == 0 ||
            displayHeight != _overlayFilledHeight ||
            (now - _overlayFilledAt) * 1000 >= OverlayRefreshMilliseconds * Stopwatch.Frequency)
        {
            _overlayPixels ??= new byte[PerfOverlay.PixelBufferWidth * PerfOverlay.PixelBufferHeight * 4];
            PerfOverlay.Fill(_overlayPixels, 0, 0);
            fixed (byte* pixels = _overlayPixels)
            {
                MetalNative.SendReplaceRegion(
                    _overlayTexture,
                    MetalNative.Selector("replaceRegion:mipmapLevel:withBytes:bytesPerRow:"),
                    new MtlRegion
                    {
                        Width = PerfOverlay.PixelBufferWidth,
                        Height = PerfOverlay.PixelBufferHeight,
                        Depth = 1,
                    },
                    0,
                    (nint)pixels,
                    (nuint)(PerfOverlay.PixelBufferWidth * 4));
            }

            _overlayFilledAt = now;
            _overlayFilledHeight = displayHeight;
        }

        // The panel is laid out in logical points; the drawable is in pixels.
        var scale = contentsScale <= 0 ? 1.0 : contentsScale;
        var rectangle = PerfOverlay.GetRectangle(
            (int)Math.Floor(_drawableWidth / scale), (int)Math.Floor(_drawableHeight / scale));
        var left = Math.Min(Math.Round(rectangle.Left * scale), _drawableWidth - 1);
        var top = Math.Min(Math.Round(rectangle.Top * scale), _drawableHeight - 1);
        var width = Math.Min(Math.Round(rectangle.Width * scale), _drawableWidth - left);
        var height = Math.Min(Math.Round(rectangle.Height * scale), _drawableHeight - top);
        if (width < 1 || height < 1)
        {
            return;
        }

        if (_layerHdrActive)
        {
            EncodeHdrPresentState(encoder, _hdrPresentPipeline, false, _edrCurrentHeadroom);
        }
        else
        {
            MetalNative.SendVoid(encoder, MetalNative.Selector("setRenderPipelineState:"), _presentPipeline);
        }

        MetalNative.SendVoidViewport(
            encoder,
            MetalNative.Selector("setViewport:"),
            new MtlViewport
            {
                OriginX = left,
                OriginY = top,
                Width = PerfOverlay.PixelBufferWidth * scale,
                Height = PerfOverlay.PixelBufferHeight * scale,
                ZNear = 0,
                ZFar = 1,
            });
        MetalNative.SendVoidScissor(
            encoder,
            MetalNative.Selector("setScissorRect:"),
            new MtlScissorRect { X = (nuint)left, Y = (nuint)top, Width = (nuint)width, Height = (nuint)height });
        MetalNative.SendSetAtIndex(
            encoder, MetalNative.Selector("setFragmentTexture:atIndex:"), _overlayTexture, 0);
        MetalNative.SendSetAtIndex(
            encoder, MetalNative.Selector("setFragmentSamplerState:atIndex:"), _overlaySampler, 0);
        MetalNative.SendDrawPrimitives(
            encoder, MetalNative.Selector("drawPrimitives:vertexStart:vertexCount:"), Mtl.PrimitiveTriangle, 0, 3);
        _overlayDrawn = true;
    }
    private static void ReleasePerfOverlayResources()
    {
        if (_overlayTexture != 0) MetalNative.SendVoid(_overlayTexture, MetalNative.Selector("release"));
        if (_overlaySampler != 0) MetalNative.SendVoid(_overlaySampler, MetalNative.Selector("release"));
        _overlayTexture = 0;
        _overlaySampler = 0;
        _overlayPixels = null;
        _overlayFilledAt = 0;
        _overlayFilledHeight = 0;
        _overlayDrawn = false;
    }

}
