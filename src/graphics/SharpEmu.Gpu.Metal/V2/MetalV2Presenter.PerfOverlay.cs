// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using SharpEmu.Libs.VideoOut;
using SharpEmu.Libs.VideoOut.Overlay;
using SharpEmu.ShaderCompiler.Metal;

namespace SharpEmu.Gpu.Metal.V2;

// The on-screen performance overlay. PerfOverlay's own thread rasterizes its widgets into a
// premultiplied BGRA image the size of the drawable; this uploads the regions that hold widgets
// into a texture of the same size and blends them over the presented frame, one scissored
// fullscreen triangle per region, as the last draws of the present pass.
internal static partial class MetalV2Presenter
{
    // The overlay redraws at up to 30 Hz; an unchanged guest frame is presented again at that
    // rate while the overlay is up so its graphs keep moving during a stall.
    private const long OverlayRefreshMilliseconds = 33;

    private static nint _overlayTexture;
    private static int _overlayTextureWidth;
    private static int _overlayTextureHeight;
    private static nint _overlaySampler;
    private static nint _overlayPipeline;
    private static nint _overlayHdrPipeline;
    private static bool _overlayPipelinesFailed;
    private static OverlayFrame? _overlayUploadedFrame;
    private static long _overlayUploadedVersion = -1;
    private static long _overlayFilledAt;
    private static bool _overlayDrawn;

    // True when an unchanged guest frame must still be presented again: the overlay is due a
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

        if (!EnsureOverlayPipelines())
        {
            return;
        }

        var scale = contentsScale <= 0 ? 1.0 : contentsScale;
        var frame = PerfOverlay.AcquireFrame((int)_drawableWidth, (int)_drawableHeight, scale);
        // Counted as a refresh even when no frame is ready yet, so a slow first render does not
        // force a present per display refresh.
        _overlayFilledAt = Stopwatch.GetTimestamp();
        if (frame is null)
        {
            return;
        }

        try
        {
            if (frame.Width != (int)_drawableWidth || frame.Height != (int)_drawableHeight || frame.Regions.Count == 0)
            {
                // Rendered for the previous drawable size; the next one will match.
                return;
            }

            if (!EnsureOverlayTexture(frame.Width, frame.Height))
            {
                return;
            }

            if (!ReferenceEquals(frame, _overlayUploadedFrame) || frame.Version != _overlayUploadedVersion)
            {
                fixed (byte* pixels = frame.Pixels)
                {
                    var rowBytes = (nuint)(frame.Width * 4);
                    foreach (var region in frame.Regions)
                    {
                        MetalNative.SendReplaceRegion(
                            _overlayTexture,
                            MetalNative.Selector("replaceRegion:mipmapLevel:withBytes:bytesPerRow:"),
                            new MtlRegion
                            {
                                X = (nuint)region.Left,
                                Y = (nuint)region.Top,
                                Width = (nuint)region.Width,
                                Height = (nuint)region.Height,
                                Depth = 1,
                            },
                            0,
                            (nint)(pixels + ((long)region.Top * frame.Width + region.Left) * 4),
                            rowBytes);
                    }
                }

                _overlayUploadedFrame = frame;
                _overlayUploadedVersion = frame.Version;
            }

            if (_layerHdrActive)
            {
                EncodeHdrPresentState(encoder, _overlayHdrPipeline, false, _edrCurrentHeadroom);
            }
            else
            {
                MetalNative.SendVoid(encoder, MetalNative.Selector("setRenderPipelineState:"), _overlayPipeline);
            }

            MetalNative.SendVoidViewport(
                encoder,
                MetalNative.Selector("setViewport:"),
                new MtlViewport
                {
                    OriginX = 0,
                    OriginY = 0,
                    Width = _drawableWidth,
                    Height = _drawableHeight,
                    ZNear = 0,
                    ZFar = 1,
                });
            MetalNative.SendSetAtIndex(
                encoder, MetalNative.Selector("setFragmentTexture:atIndex:"), _overlayTexture, 0);
            MetalNative.SendSetAtIndex(
                encoder, MetalNative.Selector("setFragmentSamplerState:atIndex:"), _overlaySampler, 0);
            foreach (var region in frame.Regions)
            {
                MetalNative.SendVoidScissor(
                    encoder,
                    MetalNative.Selector("setScissorRect:"),
                    new MtlScissorRect
                    {
                        X = (nuint)region.Left,
                        Y = (nuint)region.Top,
                        Width = (nuint)region.Width,
                        Height = (nuint)region.Height,
                    });
                MetalNative.SendDrawPrimitives(
                    encoder, MetalNative.Selector("drawPrimitives:vertexStart:vertexCount:"), Mtl.PrimitiveTriangle, 0, 3);
            }

            // Later draws in this encoder must not inherit the last region's scissor.
            MetalNative.SendVoidScissor(
                encoder,
                MetalNative.Selector("setScissorRect:"),
                new MtlScissorRect { X = 0, Y = 0, Width = (nuint)_drawableWidth, Height = (nuint)_drawableHeight });
            _overlayDrawn = true;
        }
        finally
        {
            PerfOverlay.ReleaseFrame(frame);
        }
    }

    private static bool EnsureOverlayPipelines()
    {
        if (_overlayPipeline != 0 && _overlayHdrPipeline != 0)
        {
            return true;
        }

        if (_overlayPipelinesFailed)
        {
            return false;
        }

        var source = MslFixedShaders.CreateOverlayFragment();
        if (!TryCreatePresentPipeline(_device, out _overlayPipeline, out var error, source, "overlay_fs",
                PixelFormatBgra8Unorm, premultipliedBlend: true) ||
            !TryCreatePresentPipeline(_device, out _overlayHdrPipeline, out error, source, "overlay_hdr_fs",
                MetalV2HdrOutput.PixelFormat, premultipliedBlend: true))
        {
            Console.Error.WriteLine($"[METAL-V2][ERROR] The performance overlay pipeline failed: {error}");
            _overlayPipelinesFailed = true;
            return false;
        }

        // Nearest filtering is the sampler default; the image is drawn 1:1 with the drawable.
        var samplerDescriptor = MetalNative.Send(
            MetalNative.Send(MetalNative.Class("MTLSamplerDescriptor"), MetalNative.Selector("alloc")),
            MetalNative.Selector("init"));
        _overlaySampler = MetalNative.Send(
            _device, MetalNative.Selector("newSamplerStateWithDescriptor:"), samplerDescriptor);
        MetalNative.SendVoid(samplerDescriptor, MetalNative.Selector("release"));
        if (_overlaySampler == 0)
        {
            _overlayPipelinesFailed = true;
            return false;
        }

        return true;
    }

    private static bool EnsureOverlayTexture(int width, int height)
    {
        if (_overlayTexture != 0 && _overlayTextureWidth == width && _overlayTextureHeight == height)
        {
            return true;
        }

        if (_overlayTexture != 0)
        {
            MetalNative.SendVoid(_overlayTexture, MetalNative.Selector("release"));
            _overlayTexture = 0;
        }

        var descriptor = MetalNative.SendTextureDescriptor(
            MetalNative.Class("MTLTextureDescriptor"),
            MetalNative.Selector("texture2DDescriptorWithPixelFormat:width:height:mipmapped:"),
            (nuint)PixelFormatBgra8Unorm,
            (nuint)width,
            (nuint)height,
            mipmapped: false);
        _overlayTexture = MetalNative.Send(_device, MetalNative.Selector("newTextureWithDescriptor:"), descriptor);
        _overlayTextureWidth = _overlayTexture != 0 ? width : 0;
        _overlayTextureHeight = _overlayTexture != 0 ? height : 0;
        _overlayUploadedFrame = null;
        _overlayUploadedVersion = -1;
        return _overlayTexture != 0;
    }

    private static void ReleasePerfOverlayResources()
    {
        foreach (var handle in new[] { _overlayTexture, _overlaySampler, _overlayPipeline, _overlayHdrPipeline })
        {
            if (handle != 0) MetalNative.SendVoid(handle, MetalNative.Selector("release"));
        }

        _overlayTexture = 0;
        _overlaySampler = 0;
        _overlayPipeline = 0;
        _overlayHdrPipeline = 0;
        _overlayTextureWidth = 0;
        _overlayTextureHeight = 0;
        _overlayPipelinesFailed = false;
        _overlayUploadedFrame = null;
        _overlayUploadedVersion = -1;
        _overlayFilledAt = 0;
        _overlayDrawn = false;
    }
}
