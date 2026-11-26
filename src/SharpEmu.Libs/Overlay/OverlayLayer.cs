// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SkiaSharp;

namespace SharpEmu.Libs.Overlay;

public abstract class OverlayLayer
{
    private long _revision;

    public long Revision => Interlocked.Read(ref _revision);

    public abstract bool IsVisible { get; }

    public abstract SKRectI Bounds(int surfaceWidth, int surfaceHeight);

    public abstract int Render(SKCanvas canvas, int width, int height, float unit);

    protected void Invalidate()
    {
        Interlocked.Increment(ref _revision);
        OverlayLayers.NoteChanged();
    }

    protected static SKRectI Centered(int surfaceWidth, int surfaceHeight, int width, int height) =>
        SKRectI.Create((surfaceWidth - width) / 2, (surfaceHeight - height) / 2, width, height);
}

public static class OverlayLayers
{
    private static readonly object Gate = new();
    private static OverlayLayer[] _layers = [];
    private static long _version;

    /// <summary>Moves whenever a layer is registered or changes (shown, hidden or redrawn).</summary>
    public static long Version => Interlocked.Read(ref _version);

    /// <summary>Raised after <see cref="Version"/> moved, on the thread that changed the layer.</summary>
    public static event Action? Changed;

    public static T Register<T>(T layer) where T : OverlayLayer
    {
        lock (Gate)
        {
            _layers = [.. _layers, layer];
        }

        NoteChanged();
        return layer;
    }

    public static OverlayLayer[] Snapshot() => Volatile.Read(ref _layers);

    internal static void NoteChanged()
    {
        Interlocked.Increment(ref _version);
        Changed?.Invoke();
    }
}
