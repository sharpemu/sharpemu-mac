// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.VideoOut;

public enum HostWindowMode
{
    Windowed,
    Borderless,
    ExclusiveFullscreen,
}

public enum HostScalingMode
{
    Fit,
    Cover,
    Stretch,
    Integer,
}

public enum HostHdrMode
{
    Auto,
    On,
    Off,
}

/// <summary>
/// The display resolution the guest is told it is driving. A title asks the system what the
/// television is and sizes its render targets from the answer, so this is the number that
/// decides how much pixel work it does. <see cref="Auto"/>, the default, is the output
/// resolution the user picked (<see cref="HostVideoOptions.Width"/> x
/// <see cref="HostVideoOptions.Height"/>, <c>--resolution</c>); the others pin a size whatever
/// that resolution is.
/// </summary>
public enum GuestDisplayResolution
{
    /// <summary>1920x1080.</summary>
    Hd,

    /// <summary>2560x1440.</summary>
    Qhd,

    /// <summary>3840x2160.</summary>
    Uhd,

    /// <summary>1280x720: for titles whose render sizes follow the display (see
    /// TitleDisplayPatches) a way to cut their pixel work below 1080p. A title that registers
    /// its own fixed-size buffers keeps them, and the presenter scales them as a PS5 does.</summary>
    Hd720,

    /// <summary>The largest console output mode that fits the chosen output resolution; see
    /// <see cref="HostVideoOptions.EffectiveDisplayResolution"/>. Last so the values the
    /// split GPU process receives keep their meaning.</summary>
    Auto,
}

public enum PerformanceOverlayCorner
{
    TopLeft,
    TopRight,
    BottomRight,
    BottomLeft,
}

public enum PerformanceOverlayMode
{
    Full,
    Minimal,
    TitleBar,
}

public sealed record HostVideoOptions
{
    public static HostVideoOptions Default { get; } = new();

    public HostWindowMode WindowMode { get; init; } = HostWindowMode.Windowed;

    public HostScalingMode ScalingMode { get; init; } = HostScalingMode.Fit;

    public int Width { get; init; } = 1920;

    public int Height { get; init; } = 1080;

    public int DisplayIndex { get; init; }

    public int RefreshRate { get; init; }

    public bool VSync { get; init; } = true;

    public HostHdrMode HdrMode { get; init; } = HostHdrMode.Auto;

    /// <summary>What the guest is told the display is; <see cref="GuestDisplayResolution.Auto"/>
    /// follows <see cref="Width"/> x <see cref="Height"/>. See
    /// <see cref="EffectiveDisplayResolution"/>.</summary>
    public GuestDisplayResolution DisplayResolution { get; init; } = GuestDisplayResolution.Auto;

    public bool OverlayEnabled { get; init; } = true;
    public PerformanceOverlayCorner OverlayCorner { get; init; } = PerformanceOverlayCorner.TopRight;
    public PerformanceOverlayMode OverlayMode { get; init; } = PerformanceOverlayMode.Full;

    // Whether HdrMode.Auto may turn HDR on: a presenter capability, not an OS property. The
    // Metal presenter registers false: Auto stays on SDR there, and only an explicit
    // --hdr=on presents HDR.
    internal static bool AutoHdrAllowed => HostVideoHost.AutoHdrAllowed;

    /// <summary>The console output mode the guest is told it drives: <see cref="DisplayResolution"/>
    /// when one is pinned, otherwise the largest of 2160p, 1440p, 1080p and 720p that fits
    /// inside the chosen output resolution in both dimensions (720p below that, the console's
    /// smallest mode). A PS5 only drives these modes, and titles size their render targets
    /// from the mode, so a 1920x1200 or 2560x1600 output gets 1080p or 1440p and the presenter
    /// scales it to the window.</summary>
    public GuestDisplayResolution EffectiveDisplayResolution =>
        DisplayResolution != GuestDisplayResolution.Auto && Enum.IsDefined(DisplayResolution)
            ? DisplayResolution
            : Width >= 3840 && Height >= 2160
                ? GuestDisplayResolution.Uhd
                : Width >= 2560 && Height >= 1440
                    ? GuestDisplayResolution.Qhd
                    : Width >= 1920 && Height >= 1080
                        ? GuestDisplayResolution.Hd
                        : GuestDisplayResolution.Hd720;

    /// <summary>The display size every guest query reports, from
    /// <see cref="EffectiveDisplayResolution"/>. One place, so the answers cannot disagree.</summary>
    public (uint Width, uint Height) GuestDisplaySize => EffectiveDisplayResolution switch
    {
        GuestDisplayResolution.Uhd => (3840u, 2160u),
        GuestDisplayResolution.Qhd => (2560u, 1440u),
        GuestDisplayResolution.Hd720 => (1280u, 720u),
        _ => (1920u, 1080u),
    };

    /// <summary>The resolution class sceVideoOutGetOutputStatus reports: 2 is the 4K class,
    /// 1 everything below it.</summary>
    public int GuestResolutionClass =>
        EffectiveDisplayResolution == GuestDisplayResolution.Uhd ? 2 : 1;

    public bool CanUseHdr(bool displayHdrEnabled, bool surfaceSupportsHdr) =>
        CanUseHdr(displayHdrEnabled, surfaceSupportsHdr, AutoHdrAllowed);

    internal bool CanUseHdr(bool displayHdrEnabled, bool surfaceSupportsHdr, bool autoAllowed) =>
        surfaceSupportsHdr && (HdrMode == HostHdrMode.On || HdrMode == HostHdrMode.Auto && autoAllowed && displayHdrEnabled);

    public HostVideoOptions Normalize() => this with
    {
        Width = Math.Clamp(Width, 640, 16384),
        Height = Math.Clamp(Height, 360, 16384),
        DisplayIndex = Math.Max(0, DisplayIndex),
        RefreshRate = Math.Clamp(RefreshRate, 0, 1000),
        HdrMode = Enum.IsDefined(HdrMode) ? HdrMode : HostHdrMode.Auto,
        DisplayResolution = Enum.IsDefined(DisplayResolution)
            ? DisplayResolution
            : GuestDisplayResolution.Auto,
        OverlayCorner = Enum.IsDefined(OverlayCorner) ? OverlayCorner : PerformanceOverlayCorner.TopRight,
        OverlayMode = Enum.IsDefined(OverlayMode) ? OverlayMode : PerformanceOverlayMode.Full,
    };
}

/// <summary>
/// The presenter-configuration seam. The composed graphics backend registers its presenter's
/// window configuration and HDR capability here at startup, so neither the video-out exports
/// nor this class names a concrete presenter or checks the host OS to find one.
/// </summary>
public static class HostVideoHost
{
    private static HostVideoOptions _currentOptions = HostVideoOptions.Default;
    private static bool _videoConfigured;
    private static Func<HostVideoOptions, bool>? _configurePresenter;
    private static Func<bool> _queryHdrOutputSupport = static () => false;
    private static bool _autoHdrAllowed = true;

    public static HostVideoOptions CurrentOptions => Volatile.Read(ref _currentOptions);

    /// <summary>
    /// Installs the presenter behind <see cref="TryConfigureVideo"/>. Called once from the
    /// application's composition root, next to <c>GuestGpu.Register</c>.
    /// </summary>
    /// <param name="configure">Applies the window/display options to the presenter before it starts.</param>
    /// <param name="queryHdrOutputSupport">Whether the presenter can actually drive an HDR
    /// swapchain right now; only consulted once <paramref name="configure"/> has succeeded.</param>
    /// <param name="autoHdrAllowed">Whether <see cref="HostHdrMode.Auto"/> may enable HDR at all
    /// on this presenter.</param>
    public static void RegisterPresenter(
        Func<HostVideoOptions, bool> configure,
        Func<bool> queryHdrOutputSupport,
        bool autoHdrAllowed)
    {
        ArgumentNullException.ThrowIfNull(configure);
        ArgumentNullException.ThrowIfNull(queryHdrOutputSupport);
        _configurePresenter = configure;
        _queryHdrOutputSupport = queryHdrOutputSupport;
        _autoHdrAllowed = autoHdrAllowed;
    }

    internal static bool AutoHdrAllowed => Volatile.Read(ref _autoHdrAllowed);

    // Public for the two-process split's GPU process, which answers the guest process's question with it.
    public static bool IsHdrOutputSupported =>
        Volatile.Read(ref _videoConfigured) && CurrentOptions.HdrMode != HostHdrMode.Off &&
        _queryHdrOutputSupport();

    public static bool TryConfigureVideo(HostVideoOptions options)
    {
        var normalized = options.Normalize();
        Volatile.Write(ref _currentOptions, normalized);
        var presenter = _configurePresenter ?? throw new InvalidOperationException(
            "No presenter was registered. The application's composition root must call " +
            "HostVideoHost.RegisterPresenter before configuring video output.");
        var configured = presenter(normalized);
        if (configured)
            Volatile.Write(ref _videoConfigured, true);
        return configured;
    }
}
