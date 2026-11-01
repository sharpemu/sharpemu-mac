// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Globalization;
using SharpEmu.Gpu.ShaderProgress;
using SharpEmu.Libs.VideoOut;

namespace SharpEmu.Gpu.Metal.V2;

// The shader warm-up progress: while the prewarm holds the guest and no guest frame has been
// presented, the title's splash image (black when it has none) is shown with the panel the
// Vulkan presenter draws over its splash (SharpEmu.Gpu.ShaderProgress). The splash is
// composed on the CPU at its own size, uploaded to a texture of its own and presented in
// place of the host frame; it is redrawn ten times a second while the prewarm runs, and
// released, with the presenter back on its ordinary frames, once the prewarm is over.
// SHARPEMU_PREWARM_PROGRESS=0 turns it off.
internal static partial class MetalV2Presenter
{
    private const double WarmupRefreshMilliseconds = 100;

    private static readonly bool PrewarmProgressEnabled =
        !string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_PREWARM_PROGRESS"), "0", StringComparison.Ordinal);

    private static uint _requestedWidth;
    private static uint _requestedHeight;
    private static nint _warmupTexture;
    private static uint _warmupWidth;
    private static uint _warmupHeight;
    private static byte[]? _warmupBackground;
    private static byte[]? _warmupPixels;
    private static long _warmupLastDraw;

    /// <summary>
    /// Window thread, before the present encoder: the texture to present while the warm-up
    /// progress shows, or false when it does not (the ordinary frame is then presented).
    /// </summary>
    private static bool PrepareWarmupSplash(out nint texture, out uint width, out uint height)
    {
        texture = 0;
        width = height = 0;
        var gate = MetalShaderPrewarm.Gate;
        if (!PrewarmProgressEnabled || _splashHidden || gate.Total <= 0 || !gate.IsRunning)
        {
            ReleaseWarmupSplash();
            return false;
        }

        if (_warmupBackground is null && !LoadWarmupBackground())
        {
            return false;
        }

        if (_warmupTexture == 0 || _warmupWidth != _warmupBackgroundWidth || _warmupHeight != _warmupBackgroundHeight)
        {
            ReleaseWarmupTexture();
            var descriptor = MetalNative.SendTextureDescriptor(
                MetalNative.Class("MTLTextureDescriptor"),
                MetalNative.Selector("texture2DDescriptorWithPixelFormat:width:height:mipmapped:"),
                (nuint)PixelFormatBgra8Unorm,
                _warmupBackgroundWidth,
                _warmupBackgroundHeight,
                mipmapped: false);
            _warmupTexture = MetalNative.Send(
                _device, MetalNative.Selector("newTextureWithDescriptor:"), descriptor);
            if (_warmupTexture == 0)
            {
                return false;
            }

            _warmupWidth = _warmupBackgroundWidth;
            _warmupHeight = _warmupBackgroundHeight;
            _warmupLastDraw = 0;
        }

        if (_warmupLastDraw == 0 ||
            Stopwatch.GetElapsedTime(_warmupLastDraw).TotalMilliseconds >= WarmupRefreshMilliseconds)
        {
            DrawWarmupSplash(gate);
        }

        texture = _warmupTexture;
        width = _warmupWidth;
        height = _warmupHeight;
        return true;
    }

    private static uint _warmupBackgroundWidth;
    private static uint _warmupBackgroundHeight;

    // The splash image as the Vulkan presenter loads it, else a black frame of the size the
    // guest's video out asked for.
    private static bool LoadWarmupBackground()
    {
        if (PngSplashLoader.TryLoad(out var pixels, out var width, out var height) &&
            pixels.Length == checked((int)(width * height * 4)))
        {
            _warmupBackground = pixels;
            _warmupBackgroundWidth = width;
            _warmupBackgroundHeight = height;
            return true;
        }

        width = _requestedWidth != 0 ? _requestedWidth : 1920;
        height = _requestedHeight != 0 ? _requestedHeight : 1080;
        var black = new byte[checked((int)(width * height * 4))];
        for (var offset = 3; offset < black.Length; offset += 4)
        {
            black[offset] = 255;
        }

        _warmupBackground = black;
        _warmupBackgroundWidth = width;
        _warmupBackgroundHeight = height;
        return true;
    }

    private static unsafe void DrawWarmupSplash(MetalShaderPrewarmGate gate)
    {
        var background = _warmupBackground!;
        var pixels = _warmupPixels is { } existing && existing.Length == background.Length
            ? existing
            : _warmupPixels = new byte[background.Length];
        background.CopyTo(pixels, 0);
        ShaderCacheProgressOverlay.Draw(pixels, (int)_warmupWidth, (int)_warmupHeight, MetalWarmupView.Build(gate));
        fixed (byte* data = pixels)
        {
            MetalNative.SendReplaceRegion(
                _warmupTexture,
                MetalNative.Selector("replaceRegion:mipmapLevel:withBytes:bytesPerRow:"),
                new MtlRegion { Width = _warmupWidth, Height = _warmupHeight, Depth = 1 },
                0,
                (nint)data,
                _warmupWidth * 4);
        }

        _warmupLastDraw = Stopwatch.GetTimestamp();
    }

    private static void ReleaseWarmupTexture()
    {
        if (_warmupTexture != 0)
        {
            MetalNative.SendVoid(_warmupTexture, MetalNative.Selector("release"));
            _warmupTexture = 0;
        }
    }

    private static void ReleaseWarmupSplash()
    {
        ReleaseWarmupTexture();
        _warmupBackground = null;
        _warmupPixels = null;
    }
}

/// <summary>
/// The warm-up counts the Metal prewarm keeps, as the panel the Vulkan presenter shows: the
/// recorded compute and graphics programs the guest is held for, then the library list and
/// the static plan as a row of their own while those phases run.
/// </summary>
internal static class MetalWarmupView
{
    public static ShaderCacheProgressView Build(MetalShaderPrewarmGate gate)
    {
        var rows = new List<ShaderCacheProgressRow>(3)
        {
            new("Compute pipelines", gate.ComputeDone, gate.ComputeTotal),
            new("Graphics pipelines", gate.GraphicsDone, gate.GraphicsTotal),
        };
        var phase = gate.Phase;
        var failed = MetalShaderPrewarm.FailedPrograms;
        var done = gate.Banked + gate.Progress;
        if (phase == MetalWarmupPhase.Programs)
        {
            done = gate.ComputeDone + gate.GraphicsDone;
        }
        else
        {
            rows.Add(new ShaderCacheProgressRow(
                phase == MetalWarmupPhase.Libraries ? "Shader libraries" : "Programs from the game's files",
                gate.Progress, gate.Total));
            failed += MetalLibraryPrewarm.Failed;
            if (phase == MetalWarmupPhase.Static)
            {
                failed += MetalStaticPrecompile.Counters.Failed;
            }
        }

        var seconds = gate.WarmupStarted == 0 ? 0 : Stopwatch.GetElapsedTime(gate.WarmupStarted).TotalSeconds;
        return new ShaderCacheProgressView(
            "Warming up shaders",
            "Rebuilding pipelines recorded in earlier sessions",
            rows,
            [
                $"Warmed {Math.Max(0, done - failed)} / Failed {failed}",
                string.Create(CultureInfo.InvariantCulture,
                    $"{MetalShaderPrewarm.WorkerCount} workers / {seconds:F0} seconds"),
            ],
            seconds);
    }
}
