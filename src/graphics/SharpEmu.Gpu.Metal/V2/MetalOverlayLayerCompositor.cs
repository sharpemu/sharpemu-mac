// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Overlay;
using SharpEmu.ShaderCompiler.Metal;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// Draws the in-game overlay layers (<see cref="OverlayLayer"/>: the IME keyboard, the save data
/// list) over the presented frame, the Metal counterpart of the Vulkan presenter's
/// RecordOverlayLayers. Each layer's painted BGRA rows live in a texture of their own size, drawn
/// 1:1 at the layer's place with one fullscreen triangle under a viewport and scissor of that
/// rectangle, blended premultiplied over the frame (the panels are opaque, so this is the Vulkan
/// copy; on an extended-range drawable the sRGB pixels are decoded as the performance overlay's are).
/// </summary>
/// <remarks>
/// A repainted layer gets a new texture instead of a write into the one an earlier command buffer may
/// still be sampling; the command buffer keeps the old one alive until it completes.
/// </remarks>
internal sealed class MetalOverlayLayerCompositor : IDisposable
{
    private readonly nint _device;
    private readonly Dictionary<int, (nint Texture, OverlayLayerFrame Frame)> _textures = new();
    private readonly HashSet<int> _drawn = new();
    private nint _pipeline;
    private nint _hdrPipeline;
    private nint _sampler;
    private bool _failed;
    private int _refusals;

    public MetalOverlayLayerCompositor(nint device)
    {
        _device = device;
    }

    /// <summary>Layers drawn by the last <see cref="Encode"/>.</summary>
    public int DrawnLayers { get; private set; }

    /// <summary>
    /// Draws <paramref name="frames"/> into the render pass of <paramref name="encoder"/>, whose target
    /// is <paramref name="surfaceWidth"/> x <paramref name="surfaceHeight"/>. A frame painted for
    /// another size (a resize the painting side has not caught up with) is skipped until its repaint.
    /// </summary>
    public unsafe bool Encode(
        nint encoder, IReadOnlyList<OverlayLayerFrame> frames, int surfaceWidth, int surfaceHeight, bool hdr, double headroom)
    {
        DrawnLayers = 0;
        _drawn.Clear();
        if (frames.Count == 0 || surfaceWidth < 1 || surfaceHeight < 1 || !EnsurePipelines())
        {
            ReleaseUndrawn();
            return false;
        }

        var bound = false;
        foreach (var frame in frames)
        {
            if (!frame.Fits(surfaceWidth, surfaceHeight))
            {
                if (frame.SurfaceWidth == surfaceWidth && frame.SurfaceHeight == surfaceHeight)
                {
                    Refuse($"layer {frame.Id} at {frame.Left},{frame.Top} {frame.Width}x{frame.Height} reaches past the " +
                        $"{surfaceWidth}x{surfaceHeight} surface it was painted for");
                }

                continue;
            }

            var texture = TextureFor(frame);
            if (texture == 0)
            {
                continue;
            }

            if (!bound)
            {
                if (hdr)
                {
                    MetalV2Presenter.EncodeHdrPresentState(encoder, _hdrPipeline, false, headroom);
                }
                else
                {
                    MetalNative.SendVoid(encoder, MetalNative.Selector("setRenderPipelineState:"), _pipeline);
                }

                MetalNative.SendSetAtIndex(
                    encoder, MetalNative.Selector("setFragmentSamplerState:atIndex:"), _sampler, 0);
                bound = true;
            }

            MetalNative.SendVoidViewport(
                encoder,
                MetalNative.Selector("setViewport:"),
                new MtlViewport
                {
                    OriginX = frame.Left,
                    OriginY = frame.Top,
                    Width = frame.Width,
                    Height = frame.Height,
                    ZNear = 0,
                    ZFar = 1,
                });
            MetalNative.SendVoidScissor(
                encoder,
                MetalNative.Selector("setScissorRect:"),
                new MtlScissorRect
                {
                    X = (nuint)frame.Left,
                    Y = (nuint)frame.Top,
                    Width = (nuint)frame.Width,
                    Height = (nuint)frame.Height,
                });
            MetalNative.SendSetAtIndex(encoder, MetalNative.Selector("setFragmentTexture:atIndex:"), texture, 0);
            MetalNative.SendDrawPrimitives(
                encoder, MetalNative.Selector("drawPrimitives:vertexStart:vertexCount:"), Mtl.PrimitiveTriangle, 0, 3);
            _drawn.Add(frame.Id);
            DrawnLayers++;
        }

        if (bound)
        {
            // Later draws in this encoder (the performance overlay) must not inherit the last layer's rectangle.
            MetalNative.SendVoidViewport(
                encoder,
                MetalNative.Selector("setViewport:"),
                new MtlViewport { Width = surfaceWidth, Height = surfaceHeight, ZNear = 0, ZFar = 1 });
            MetalNative.SendVoidScissor(
                encoder,
                MetalNative.Selector("setScissorRect:"),
                new MtlScissorRect { Width = (nuint)surfaceWidth, Height = (nuint)surfaceHeight });
        }

        ReleaseUndrawn();
        return DrawnLayers != 0;
    }

    private unsafe nint TextureFor(OverlayLayerFrame frame)
    {
        if (_textures.TryGetValue(frame.Id, out var cached) && ReferenceEquals(cached.Frame, frame))
        {
            return cached.Texture;
        }

        var descriptor = MetalNative.SendTextureDescriptor(
            MetalNative.Class("MTLTextureDescriptor"),
            MetalNative.Selector("texture2DDescriptorWithPixelFormat:width:height:mipmapped:"),
            (nuint)MtlFormat.Bgra8Unorm,
            (nuint)frame.Width,
            (nuint)frame.Height,
            mipmapped: false);
        var texture = MetalNative.Send(_device, MetalNative.Selector("newTextureWithDescriptor:"), descriptor);
        if (texture == 0)
        {
            Refuse($"no texture for layer {frame.Id} ({frame.Width}x{frame.Height})");
            return 0;
        }

        fixed (byte* pixels = frame.Pixels)
        {
            MetalNative.SendReplaceRegion(
                texture,
                MetalNative.Selector("replaceRegion:mipmapLevel:withBytes:bytesPerRow:"),
                new MtlRegion { Width = (nuint)frame.Width, Height = (nuint)frame.Height, Depth = 1 },
                0,
                (nint)pixels,
                (nuint)(frame.Width * 4));
        }

        if (cached.Texture != 0)
        {
            MetalNative.SendVoid(cached.Texture, MetalNative.Selector("release"));
        }

        _textures[frame.Id] = (texture, frame);
        return texture;
    }

    // A layer that was not drawn this time (hidden, or waiting for its repaint at a new size) gives
    // its texture back; it is uploaded again when it returns.
    private void ReleaseUndrawn()
    {
        if (_textures.Count == _drawn.Count)
        {
            return;
        }

        foreach (var id in _textures.Keys.Where(id => !_drawn.Contains(id)).ToList())
        {
            MetalNative.SendVoid(_textures[id].Texture, MetalNative.Selector("release"));
            _textures.Remove(id);
        }
    }

    private bool EnsurePipelines()
    {
        if (_pipeline != 0 && _hdrPipeline != 0 && _sampler != 0)
        {
            return true;
        }

        if (_failed)
        {
            return false;
        }

        var source = MslFixedShaders.CreateOverlayFragment();
        if (!MetalV2Presenter.TryCreatePresentPipeline(_device, out _pipeline, out var error, source, "overlay_fs",
                MtlFormat.Bgra8Unorm, premultipliedBlend: true) ||
            !MetalV2Presenter.TryCreatePresentPipeline(_device, out _hdrPipeline, out error, source, "overlay_hdr_fs",
                MetalV2HdrOutput.PixelFormat, premultipliedBlend: true))
        {
            Console.Error.WriteLine($"[METAL-V2][ERROR] The overlay layer pipeline failed; overlay layers are not drawn: {error}");
            _failed = true;
            return false;
        }

        // Nearest filtering (the sampler default): every layer is drawn 1:1.
        var samplerDescriptor = MetalNative.Send(
            MetalNative.Send(MetalNative.Class("MTLSamplerDescriptor"), MetalNative.Selector("alloc")),
            MetalNative.Selector("init"));
        _sampler = MetalNative.Send(_device, MetalNative.Selector("newSamplerStateWithDescriptor:"), samplerDescriptor);
        MetalNative.SendVoid(samplerDescriptor, MetalNative.Selector("release"));
        if (_sampler == 0)
        {
            Console.Error.WriteLine("[METAL-V2][ERROR] The overlay layer sampler failed; overlay layers are not drawn.");
            _failed = true;
            return false;
        }

        return true;
    }

    private void Refuse(string what)
    {
        if (++_refusals <= 8)
        {
            Console.Error.WriteLine($"[METAL-V2][WARN] Overlay layer not drawn: {what}.");
        }
    }

    public void Dispose()
    {
        foreach (var (texture, _) in _textures.Values)
        {
            MetalNative.SendVoid(texture, MetalNative.Selector("release"));
        }

        _textures.Clear();
        foreach (var handle in new[] { _pipeline, _hdrPipeline, _sampler })
        {
            if (handle != 0)
            {
                MetalNative.SendVoid(handle, MetalNative.Selector("release"));
            }
        }

        _pipeline = 0;
        _hdrPipeline = 0;
        _sampler = 0;
    }
}
