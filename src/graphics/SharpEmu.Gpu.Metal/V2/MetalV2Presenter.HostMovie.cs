// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Media;

namespace SharpEmu.Gpu.Metal.V2;

// The AvPlayer host fallback frame, presented in place of the guest's flips: the v2
// twin of the Vulkan presenter's TryReplaceWithHostMovieFrame and
// TryTakeHostMovieOnlyPresentation. Runs on the window thread only.
internal static partial class MetalV2Presenter
{
    private static nint _movieFallbackTexture;
    private static uint _movieFallbackWidth;
    private static uint _movieFallbackHeight;
    private static long _movieFallbackSerial;
    private static bool _movieFallbackShown;
    private static long _movieFallbackGuestFrames;
    private static long _tracedAvPlayerFallbackSerial;
    private static long _avPlayerFallbackPresentationCount;

    /// <summary>
    /// AvPlayer titles whose guest texture allocators reject the decoded movie surface
    /// have no sampled image to draw, so the movie would never become visible; the
    /// AvPlayer HLE then keeps a host-decoded BGRA frame, and that frame is shown in
    /// place of the guest image. It follows the host clock, not the title's flips,
    /// which are far slower than the movie's rate. Like a Vulkan presentation, the
    /// movie frame stays on screen until a guest flip reaches the display while no
    /// movie frame is available to stand in for it. True when the movie frame, which is
    /// SDR, replaced the guest's.
    /// </summary>
    private static bool SelectHostMovieFallback(ref nint texture, ref uint width, ref uint height)
    {
        var guestFrames = Interlocked.Read(ref _adoptedGuestFrameSequence);
        if (MoviePresentationServices.TryGetFallbackPresentationFrame(
                out var pixels, out var frameWidth, out var frameHeight, out var serial))
        {
            UploadMovieFallback(pixels, frameWidth, frameHeight, serial);
            _movieFallbackShown = _movieFallbackTexture != 0;
        }
        else if (guestFrames != _movieFallbackGuestFrames && _movieFallbackShown)
        {
            _movieFallbackShown = false;
            // The present command buffers that sampled it hold their own reference.
            MetalNative.SendVoid(_movieFallbackTexture, MetalNative.Selector("release"));
            _movieFallbackTexture = 0;
            _movieFallbackSerial = 0;
        }

        _movieFallbackGuestFrames = guestFrames;
        if (_movieFallbackShown)
        {
            texture = _movieFallbackTexture;
            width = _movieFallbackWidth;
            height = _movieFallbackHeight;
        }

        return _movieFallbackShown;
    }

    // A new texture per movie frame: the drawables still in flight keep sampling the
    // one before, and the command buffers that bound it keep it alive until they finish.
    private static unsafe void UploadMovieFallback(byte[] pixels, uint width, uint height, long serial)
    {
        if (serial == _movieFallbackSerial && _movieFallbackTexture != 0)
        {
            return;
        }

        var descriptor = MetalNative.SendTextureDescriptor(
            MetalNative.Class("MTLTextureDescriptor"),
            MetalNative.Selector("texture2DDescriptorWithPixelFormat:width:height:mipmapped:"),
            (nuint)PixelFormatBgra8Unorm,
            width,
            height,
            mipmapped: false);
        var texture = descriptor == 0
            ? 0
            : MetalNative.Send(_device, MetalNative.Selector("newTextureWithDescriptor:"), descriptor);
        if (texture == 0)
        {
            return;
        }

        fixed (byte* source = pixels)
        {
            MetalNative.SendReplaceRegion(
                texture,
                MetalNative.Selector("replaceRegion:mipmapLevel:withBytes:bytesPerRow:"),
                new MtlRegion { Width = width, Height = height, Depth = 1 },
                0,
                (nint)source,
                width * 4);
        }

        if (_movieFallbackTexture != 0)
        {
            MetalNative.SendVoid(_movieFallbackTexture, MetalNative.Selector("release"));
        }

        _movieFallbackTexture = texture;
        _movieFallbackWidth = width;
        _movieFallbackHeight = height;
        _movieFallbackSerial = serial;
        if (Interlocked.Exchange(ref _tracedAvPlayerFallbackSerial, serial) != serial)
        {
            var frameCount = Interlocked.Increment(ref _avPlayerFallbackPresentationCount);
            if (frameCount <= 4 || frameCount % 30 == 0)
            {
                Console.Error.WriteLine(
                    "[VIDEOOUT][INFO] AvPlayer host fallback frame presented: " +
                    $"frame={frameCount} serial={serial} size={width}x{height}.");
            }
        }
    }
}
