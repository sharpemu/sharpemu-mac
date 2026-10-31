// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.VideoOut;

/// <summary>
/// The presentation window the Metal presenter drives. Keeps the presenter free of
/// any windowing-toolkit reference: on macOS the implementation is the native
/// AppKit <c>MacHostWindow</c>, and the contract is deliberately the subset of the
/// old SDL host window the Metal path actually used, with identical semantics so the
/// render cadence does not change.
/// </summary>
public interface IMetalHostWindow : IDisposable
{
    /// <summary>Backing-store size of the drawable surface in device pixels
    /// (Retina aware). Never smaller than 1x1.</summary>
    (int Width, int Height) PixelSize { get; }

    /// <summary>Drawable pixels per logical point, used to size the on-screen
    /// performance display. 1 for a window with no scale factor.</summary>
    double ContentsScale => 1.0;

    /// <summary>True while the window is miniaturized; the presenter skips
    /// acquiring drawables and throttles instead.</summary>
    bool IsMinimized { get; }

    /// <summary>True once the user closed the window (or asked the app to quit),
    /// as opposed to the emulator asking the loop to stop.</summary>
    bool ClosedByUser { get; }

    /// <summary>
    /// Creates (once) and returns the CAMetalLayer backing the window's content
    /// view. The caller configures device, pixel format and display sync.
    /// </summary>
    nint CreateMetalLayer();

    /// <summary>
    /// The extended-dynamic-range headroom of the screen the window is on, as
    /// multiples of SDR white: <c>Current</c> is what the display can show right
    /// now (it rises once a layer asks for EDR), <c>Potential</c> the most it can
    /// ever reach. Both are 1.0 on a display without EDR. Main thread only.
    /// </summary>
    (double Current, double Potential) EdrHeadroom { get; }

    /// <summary>
    /// Reads and clears the "surface geometry changed" latch — resize, display
    /// change, backing-scale change, fullscreen transition or restore. The
    /// presenter uses it to force a drawable-size resync.
    /// </summary>
    bool ConsumeSurfaceRestore();

    /// <summary>Replaces the base window title (the overlay summary is appended
    /// by the window itself).</summary>
    void SetTitle(string title);

    /// <summary>Asks <see cref="Run"/> to return at the next loop boundary.</summary>
    void Close();

    /// <summary>
    /// Runs the window/event loop on the calling thread until <see cref="Close"/>
    /// or a user close. Calls <paramref name="initialize"/> once first,
    /// <paramref name="render"/> once per iteration with the elapsed seconds
    /// since the previous call, <paramref name="idle"/> after each rendered
    /// iteration, and <paramref name="closing"/> exactly once on the way out.
    /// </summary>
    void Run(Action initialize, Action<double> render, Action closing, Action? idle = null);
}
