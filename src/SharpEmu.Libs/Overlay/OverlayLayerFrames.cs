// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.Overlay;

/// <summary>
/// One overlay layer painted for a presentation surface: <see cref="Height"/> rows of
/// <see cref="Width"/> premultiplied BGRA pixels, to be blended over the presented frame with their
/// top-left corner at (<see cref="Left"/>, <see cref="Top"/>). Immutable once made; a presenter can
/// tell a repaint from the frame it already uploaded by reference.
/// </summary>
public sealed class OverlayLayerFrame
{
    public OverlayLayerFrame(
        int id, long revision, int surfaceWidth, int surfaceHeight, int left, int top, int width, int height, byte[] pixels)
    {
        ArgumentNullException.ThrowIfNull(pixels);
        if (width <= 0 || height <= 0 || pixels.Length != checked(width * height * 4))
        {
            throw new ArgumentException("An overlay layer frame's pixels must match its size.", nameof(pixels));
        }

        Id = id;
        Revision = revision;
        SurfaceWidth = surfaceWidth;
        SurfaceHeight = surfaceHeight;
        Left = left;
        Top = top;
        Width = width;
        Height = height;
        Pixels = pixels;
    }

    /// <summary>The layer's index in <see cref="OverlayLayers.Snapshot"/>: its draw order.</summary>
    public int Id { get; }

    public long Revision { get; }

    public int SurfaceWidth { get; }

    public int SurfaceHeight { get; }

    public int Left { get; }

    public int Top { get; }

    public int Width { get; }

    public int Height { get; }

    public byte[] Pixels { get; }

    /// <summary>Whether the frame was painted for this surface and lies inside it.</summary>
    public bool Fits(int surfaceWidth, int surfaceHeight) =>
        SurfaceWidth == surfaceWidth && SurfaceHeight == surfaceHeight && Left >= 0 && Top >= 0 &&
        (long)Left + Width <= surfaceWidth && (long)Top + Height <= surfaceHeight;
}

/// <summary>Where a presenter takes the overlay layers it draws over each presented frame.</summary>
public interface IOverlayLayerSource
{
    /// <summary>Moves whenever a layer appears, changes or hides: a presenter that saw it last can skip a present.</summary>
    long Version { get; }

    /// <summary>
    /// Adds the visible layers painted for a surface of this size to <paramref name="frames"/>, in
    /// draw order. A layer that has nothing to draw at this size is left out.
    /// </summary>
    void Collect(int surfaceWidth, int surfaceHeight, List<OverlayLayerFrame> frames);
}

/// <summary>
/// Paints the registered <see cref="OverlayLayer"/>s in this process, repainting a layer only when
/// its revision or the surface size changed. A layer is placed as the Vulkan presenter places it:
/// its bounds for the surface, the rows its render reports used centred vertically in them.
/// </summary>
public sealed class OverlayLayerPainter : IOverlayLayerSource
{
    private readonly object _gate = new();
    private readonly Dictionary<int, OverlayLayerFrame> _frames = new();
    private readonly Func<OverlayLayer[]> _layers;

    public OverlayLayerPainter()
        : this(OverlayLayers.Snapshot)
    {
    }

    public OverlayLayerPainter(Func<OverlayLayer[]> layers)
    {
        _layers = layers;
    }

    public long Version => OverlayLayers.Version;

    public void Collect(int surfaceWidth, int surfaceHeight, List<OverlayLayerFrame> frames)
    {
        lock (_gate)
        {
            var layers = _layers();
            for (var id = 0; id < layers.Length; id++)
            {
                var layer = layers[id];
                // Read before the visibility, as the Vulkan presenter does: a change in between is
                // painted again on the next collection.
                var revision = layer.Revision;
                if (!layer.IsVisible)
                {
                    _frames.Remove(id);
                    continue;
                }

                if (_frames.TryGetValue(id, out var frame) && frame.Revision == revision &&
                    frame.SurfaceWidth == surfaceWidth && frame.SurfaceHeight == surfaceHeight)
                {
                    frames.Add(frame);
                    continue;
                }

                var painted = Paint(id, layer, revision, surfaceWidth, surfaceHeight);
                if (painted is null)
                {
                    _frames.Remove(id);
                    continue;
                }

                _frames[id] = painted;
                frames.Add(painted);
            }

            if (_frames.Count > layers.Length)
            {
                foreach (var stale in _frames.Keys.Where(id => id >= layers.Length).ToList())
                {
                    _frames.Remove(stale);
                }
            }
        }
    }

    /// <summary>Paints one layer for a surface, or null when it has no room on it (a surface too small for it).</summary>
    public static OverlayLayerFrame? Paint(int id, OverlayLayer layer, long revision, int surfaceWidth, int surfaceHeight)
    {
        var bounds = layer.Bounds(surfaceWidth, surfaceHeight);
        if (bounds.Width <= 0 || bounds.Height <= 0 || bounds.Left < 0 || bounds.Top < 0 ||
            bounds.Right > surfaceWidth || bounds.Bottom > surfaceHeight)
        {
            return null;
        }

        var pixels = new byte[checked(bounds.Width * bounds.Height * 4)];
        var unit = OverlayStyle.UnitFor(surfaceHeight);
        var used = OverlayPainter.RenderInto(
            pixels, bounds.Width, bounds.Height, canvas => layer.Render(canvas, bounds.Width, bounds.Height, unit));
        var rows = used == bounds.Height ? pixels : pixels.AsSpan(0, bounds.Width * used * 4).ToArray();
        return new OverlayLayerFrame(
            id, revision, surfaceWidth, surfaceHeight, bounds.Left, bounds.Top + (bounds.Height - used) / 2,
            bounds.Width, used, rows);
    }
}

/// <summary>
/// The presenter's side of the overlay layers: where it takes them from, and the size of the surface
/// it presents to, which the painting side needs. By default the layers of this process are painted
/// on the presenting thread; the split's GPU process takes them from the guest process instead.
/// </summary>
public static class OverlayLayerOutput
{
    private static IOverlayLayerSource _source = new OverlayLayerPainter();
    private static long _surface;

    public static IOverlayLayerSource Source
    {
        get => Volatile.Read(ref _source);
        set => Volatile.Write(ref _source, value ?? throw new ArgumentNullException(nameof(value)));
    }

    /// <summary>The presented surface's size in pixels; (0, 0) before the presenter reported one.</summary>
    public static (int Width, int Height) Surface
    {
        get
        {
            var packed = Interlocked.Read(ref _surface);
            return ((int)(packed >> 32), (int)(uint)packed);
        }
    }

    /// <summary>Raised when <see cref="ReportSurface"/> changed the size.</summary>
    public static event Action<int, int>? SurfaceChanged;

    /// <summary>Called by the presenter when its drawable takes a new size.</summary>
    public static void ReportSurface(int width, int height)
    {
        var packed = (long)(uint)width << 32 | (uint)height;
        if (Interlocked.Exchange(ref _surface, packed) != packed)
        {
            SurfaceChanged?.Invoke(width, height);
        }
    }
}
