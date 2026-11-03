// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Fonts.Abstractions;
using SharpEmu.Libs.Font;
using SharpEmu.Libs.VideoOut.Overlay;
using SharpEmu.Logging;

namespace SharpEmu.Libs.VideoOut;

// The interactive, widget-based overlay. The graphics backend asks for a finished frame on its
// present thread (AcquireFrame / ReleaseFrame), the host window forwards pointer and key events,
// and a background thread does the capture, layout and rasterization.
public static partial class PerfOverlay
{
    private static readonly object HostLock = new();
    private static OverlayHost? _host;
    private static IOverlayDataSource? _attachedSource;
    private static IFontFactory? _attachedFonts;

    private static OverlayHost Host
    {
        get
        {
            if (_host is { } existing)
            {
                return existing;
            }

            lock (HostLock)
            {
                if (_host is not null)
                {
                    return _host;
                }

                OverlayHost.DefaultFontProvider ??= static () => GuestFontServices.Factory;
                var preset = DisplayState.Mode == PerformanceOverlayMode.Minimal
                    ? OverlayLayoutModel.MinimalPreset
                    : OverlayLayoutModel.InspectorPreset;
                var source = _attachedSource ?? SimulatedOverlayDataSource.FromEnvironment();
                _host = new OverlayHost(source, _attachedFonts, static () => DisplayState.DrawOnScreen,
                    new OverlayLayoutStore(SharpEmuPaths.UserPath("overlay")), preset);
                return _host;
            }
        }
    }

    /// <summary>Supplies the snapshots the widgets draw (OverlayTelemetry in the emulator).</summary>
    public static void AttachDataSource(IOverlayDataSource source)
    {
        ArgumentNullException.ThrowIfNull(source);
        lock (HostLock)
        {
            _attachedSource = source;
            _host?.AttachDataSource(source);
        }
    }

    /// <summary>Sets the font factory used to rasterize overlay text; without one, the guest font services are tried and a built-in bitmap font is the last resort.</summary>
    public static void SetFontFactory(IFontFactory factory)
    {
        ArgumentNullException.ThrowIfNull(factory);
        lock (HostLock)
        {
            _attachedFonts = factory;
            _host?.SetFontFactory(factory);
        }
    }

    /// <summary>
    /// Returns the newest finished overlay frame for a surface of this pixel size, or null when the
    /// overlay is hidden or not ready. Call <see cref="ReleaseFrame"/> after uploading it. The size
    /// passed here also drives the next render.
    /// </summary>
    public static OverlayFrame? AcquireFrame(int surfacePixelWidth, int surfacePixelHeight, double pixelScale) =>
        DisplayState.DrawOnScreen ? Host.AcquireFrame(surfacePixelWidth, surfacePixelHeight, pixelScale) : null;

    public static void ReleaseFrame(OverlayFrame frame) => _host?.ReleaseFrame(frame);

    /// <summary>Forwards a pointer event (surface points); true when the overlay consumed it.</summary>
    public static bool HandlePointer(OverlayPointerEvent e) => DisplayState.DrawOnScreen && Host.HandlePointer(e);

    /// <summary>Forwards an overlay key; F1 toggles visibility, the rest edit the overlay. True when consumed.</summary>
    public static bool HandleKey(OverlayKey key)
    {
        if (key == OverlayKey.ToggleVisible)
        {
            Toggle();
            return true;
        }

        return DisplayState.DrawOnScreen && Host.HandleKey(key);
    }

    /// <summary>True in edit mode or while the pointer is over a widget: show a cursor and keep clicks away from the guest.</summary>
    public static bool WantsPointer => DisplayState.DrawOnScreen && _host is { WantsPointer: true };
}
