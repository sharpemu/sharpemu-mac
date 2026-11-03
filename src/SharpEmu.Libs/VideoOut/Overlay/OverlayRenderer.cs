// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Fonts.Abstractions;

namespace SharpEmu.Libs.VideoOut.Overlay;

/// <summary>The colours of one overlay theme.</summary>
public readonly record struct OverlayPalette(
    OverlayColor Panel,
    OverlayColor Border,
    OverlayColor Text,
    OverlayColor Dim,
    OverlayColor Track,
    OverlayColor Chip,
    OverlayColor ChartLine)
{
    public static OverlayPalette For(OverlayTheme theme) => theme switch
    {
        OverlayTheme.Oled => new(
            OverlayColor.Hex(0x000000, 0.92f), OverlayColor.Hex(0xFFFFFF, 0.15f), OverlayColor.Hex(0xF4F5F7),
            OverlayColor.Hex(0x8A8F99), OverlayColor.Hex(0xFFFFFF, 0.10f), OverlayColor.Hex(0xFFFFFF, 0.10f),
            OverlayColor.Hex(0xFFFFFF, 0.22f)),
        // Glass cannot blur the game frame, so it is a more transparent panel with a lighter border.
        OverlayTheme.Glass => new(
            OverlayColor.Hex(0x161A20, 0.46f), OverlayColor.Hex(0xFFFFFF, 0.34f), OverlayColor.Hex(0xFFFFFF),
            OverlayColor.Hex(0xE6EAF0, 0.78f), OverlayColor.Hex(0xFFFFFF, 0.18f), OverlayColor.Hex(0xFFFFFF, 0.20f),
            OverlayColor.Hex(0xFFFFFF, 0.40f)),
        _ => new(
            OverlayColor.Hex(0x0E1014, 0.78f), OverlayColor.Hex(0xFFFFFF, 0.09f), OverlayColor.Hex(0xE8EAEE),
            OverlayColor.Hex(0x8B93A1), OverlayColor.Hex(0xFFFFFF, 0.08f), OverlayColor.Hex(0xFFFFFF, 0.09f),
            OverlayColor.Hex(0xFFFFFF, 0.20f)),
    };
}

/// <summary>Colours of the cost kinds and the frame-rate traffic light.</summary>
public static class OverlayColors
{
    public static readonly OverlayColor Guest = OverlayColor.Hex(0x5FB3FF);
    public static readonly OverlayColor Hle = OverlayColor.Hex(0x9AD0FF);
    public static readonly OverlayColor Render = OverlayColor.Hex(0xB48CFF);
    public static readonly OverlayColor Gpu = OverlayColor.Hex(0x4AD991);
    public static readonly OverlayColor Wait = OverlayColor.Hex(0xFFB547);
    public static readonly OverlayColor Shader = OverlayColor.Hex(0xFF6B8A);
    public static readonly OverlayColor Other = OverlayColor.Hex(0x6B7280);
    public static readonly OverlayColor Good = OverlayColor.Hex(0x7EE2A8);
    public static readonly OverlayColor Mid = OverlayColor.Hex(0xFFB547);
    public static readonly OverlayColor Bad = OverlayColor.Hex(0xFF6B8A);

    public static OverlayColor OfKind(CostKind kind) => kind switch
    {
        CostKind.Guest => Guest,
        CostKind.Hle => Hle,
        CostKind.Render => Render,
        CostKind.Shader => Shader,
        CostKind.Gpu => Gpu,
        CostKind.Wait => Wait,
        _ => Other,
    };

    public static OverlayColor OfLane(OverlayBusyLane lane) => lane switch
    {
        OverlayBusyLane.GuestMain => Guest,
        OverlayBusyLane.Workers => Guest.WithAlpha(0.6f),
        OverlayBusyLane.Render => Render,
        _ => Gpu,
    };

    public static string KindLabel(CostKind kind) => kind switch
    {
        CostKind.Guest => "Guest",
        CostKind.Hle => "HLE",
        CostKind.Render => "Render",
        CostKind.Shader => "Shader",
        CostKind.Gpu => "GPU",
        CostKind.Wait => "Wait",
        _ => "GC / IO",
    };

    /// <summary>Green at the target, amber below 93%, red below 70% (the prototype's traffic light).</summary>
    public static OverlayColor Fps(double fps, double target) =>
        fps >= target * 0.93 ? Good : fps >= target * 0.7 ? Mid : Bad;
}

/// <summary>
/// Rasterizes the overlay (widgets, edit chrome, popovers, tooltips) into an <see cref="OverlayFrame"/>.
/// One instance per render thread; it keeps scratch buffers and the glyph cache between redraws.
/// </summary>
public sealed partial class OverlayRenderer : IDisposable
{
    private readonly OverlayController _ctl;
    private readonly OverlayAnalyzer _analyzer;
    private readonly OverlayCanvas _c = new();
    private readonly IFontFactory? _fonts;
    private OverlayTypography _t;
    private readonly TextBuf _tb = new();
    private readonly TextBuf _tb2 = new();
    private readonly List<OverlayRectI> _regions = [];
    private readonly TopCostsResult _top = new();
    private OverlayPoint[] _pts = new OverlayPoint[512];
    private float[] _scratchF = new float[64];
    private int[] _scratchI = new int[64];

    // Per-redraw state.
    private OverlayView _v = new();
    private OverlayPalette _pal;
    private OverlayGlobals _g = new();
    private OverlayMetrics _m;
    private float _fs = 11f;
    private float _ps = 1f;
    private long _now;
    private OverlayFrame? _frame;

    public OverlayRenderer(OverlayController controller, OverlayAnalyzer analyzer, IFontFactory? fonts)
    {
        _ctl = controller;
        _analyzer = analyzer;
        _fonts = fonts;
        _t = new OverlayTypography(_c, fonts, controller.Model.Globals.Font);
    }

    public bool UsesBuiltInFont => _t.UsesBuiltInFont;

    public void Dispose() => _t.Dispose();

    /// <summary>The composite regions of the last <see cref="Render"/>, in pixels.</summary>
    public IReadOnlyList<OverlayRectI> LastRegions => _regions;

    private OverlayPoint[] Points(int n)
    {
        if (_pts.Length < n)
        {
            _pts = new OverlayPoint[Math.Max(n, _pts.Length * 2)];
        }

        return _pts;
    }

    private float[] FloatScratch(int n)
    {
        if (_scratchF.Length < n)
        {
            _scratchF = new float[Math.Max(n, _scratchF.Length * 2)];
        }

        return _scratchF;
    }

    private int[] IntScratch(int n)
    {
        if (_scratchI.Length < n)
        {
            _scratchI = new int[Math.Max(n, _scratchI.Length * 2)];
        }

        return _scratchI;
    }

    // ------------------------------------------------------------------ entry point

    /// <summary>
    /// Draws one overlay frame. The caller holds <see cref="OverlayController.Gate"/>. Pixels inside the
    /// resulting regions are rewritten completely; pixels outside are left alone.
    /// </summary>
    public void Render(OverlayFrame frame, OverlayView view, double pixelScale, long nowTicks)
    {
        _frame = frame;
        _v = view;
        _now = nowTicks;
        _ps = (float)Math.Max(0.5, pixelScale);
        _g = _ctl.Model.Globals;
        _g.Clamp();
        _pal = OverlayPalette.For(_g.Theme);
        if (_g.Font != _t.Font && !_t.UsesBuiltInFont)
        {
            // The typeface changed (settings, or the layout file reloaded): rebuild the glyph cache.
            _t.Dispose();
            _t = new OverlayTypography(_c, _fonts, _g.Font);
        }

        _fs = _g.FontSize;
        _m = new OverlayMetrics(frame.Width / _ps, frame.Height / _ps, _ps, _g.Scale);
        _ctl.UpdateSurface(_m);
        _ctl.Frames = view.Frames;
        PrepareChrome();
        _ctl.BeginHits();
        _c.Attach(frame.Pixels, frame.Width, frame.Height);

        ComputeRegions();
        BuildOverwrittenRects();
        foreach (var r in _regions)
        {
            _c.ClearPixelsExcept(r.X, r.Y, r.Width, r.Height, System.Runtime.InteropServices.CollectionsMarshal.AsSpan(_overwritten));
        }

        _c.Opacity = 1f;
        if (_ctl.Edit)
        {
            DrawEditBackdrop();
        }

        var widgets = _ctl.Model.Widgets;
        _c.Opacity = _g.Opacity;
        for (var i = 0; i < widgets.Count; i++)
        {
            DrawWidget(widgets[i], i);
        }

        _c.Opacity = 1f;
        _c.ResetState();
        DrawChrome();
        frame.Regions.Clear();
        foreach (var r in _regions)
        {
            frame.Regions.Add(new OverlayRectangle(r.X, r.Y, r.Width, r.Height));
        }

        frame.Version++;
        _ctl.RefreshHover();
    }

    // ------------------------------------------------------------------ regions

    private void AddRegionPx(float x, float y, float w, float h)
    {
        var x0 = Math.Max(0, (int)MathF.Floor(x));
        var y0 = Math.Max(0, (int)MathF.Floor(y));
        var x1 = Math.Min(_frame!.Width, (int)MathF.Ceiling(x + w));
        var y1 = Math.Min(_frame.Height, (int)MathF.Ceiling(y + h));
        if (x1 > x0 && y1 > y0)
        {
            _regions.Add(new OverlayRectI(x0, y0, x1 - x0, y1 - y0));
        }
    }

    private void AddRegionPt(RectF r, float inflate = 0f) =>
        AddRegionPx((r.X - inflate) * _ps, (r.Y - inflate) * _ps, (r.W + 2 * inflate) * _ps, (r.H + 2 * inflate) * _ps);

    private readonly List<OverlayRectI> _overwritten = [];

    /// <summary>
    /// The solid interior of each widget panel is stored (not blended) by <see cref="DrawWidget"/>, so those
    /// pixels need no clear beforehand. Only done outside edit mode, where widgets are laid out apart.
    /// </summary>
    private void BuildOverwrittenRects()
    {
        _overwritten.Clear();
        if (_ctl.Edit)
        {
            return;
        }

        var widgets = _ctl.Model.Widgets;
        var inset = (int)MathF.Ceiling(PanelRadius * _m.ContentScale) + 4;
        for (var i = 0; i < widgets.Count; i++)
        {
            var box = _m.BoxOf(widgets[i]);
            var x0 = (int)MathF.Ceiling(box.X * _ps) + inset;
            var y0 = (int)MathF.Ceiling(box.Y * _ps) + inset;
            var x1 = (int)MathF.Floor(box.Right * _ps) - inset;
            var y1 = (int)MathF.Floor(box.Bottom * _ps) - inset;
            if (x1 > x0 && y1 > y0)
            {
                _overwritten.Add(new OverlayRectI(x0, y0, x1 - x0, y1 - y0));
            }
        }
    }

    private void ComputeRegions()
    {
        _regions.Clear();
        if (_ctl.Edit)
        {
            _regions.Add(new OverlayRectI(0, 0, _frame!.Width, _frame.Height));
            return;
        }

        var widgets = _ctl.Model.Widgets;
        for (var i = 0; i < widgets.Count; i++)
        {
            AddRegionPt(_m.BoxOf(widgets[i]), 2f);
        }

        AddChromeRegions();
        MergeRegions();
    }

    /// <summary>Overlapping regions would be blended twice by a backend, so overlapping ones are united.</summary>
    private void MergeRegions()
    {
        var merged = true;
        while (merged)
        {
            merged = false;
            for (var i = 0; i < _regions.Count && !merged; i++)
            {
                for (var j = i + 1; j < _regions.Count; j++)
                {
                    var a = _regions[i];
                    var b = _regions[j];
                    if (a.X < b.X + b.Width && a.X + a.Width > b.X && a.Y < b.Y + b.Height && a.Y + a.Height > b.Y)
                    {
                        var x0 = Math.Min(a.X, b.X);
                        var y0 = Math.Min(a.Y, b.Y);
                        var x1 = Math.Max(a.X + a.Width, b.X + b.Width);
                        var y1 = Math.Max(a.Y + a.Height, b.Y + b.Height);
                        _regions[i] = new OverlayRectI(x0, y0, x1 - x0, y1 - y0);
                        _regions.RemoveAt(j);
                        merged = true;
                        break;
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------ shared drawing helpers

    private float Txt(ReadOnlySpan<char> s, float x, float y, float size, OverlayColor color, bool bold = false,
        OverlayTextAlign align = OverlayTextAlign.Left, float max = 0f, bool upper = false, float spacing = 0f) =>
        _t.Draw(s, x, y, size, color, new OverlayTextStyle(bold, upper, spacing, align, max));

    private float Measure(ReadOnlySpan<char> s, float size, bool bold = false, float spacing = 0f) =>
        _t.Measure(s, size, new OverlayTextStyle(bold, false, spacing));

    private float Label(ReadOnlySpan<char> s, float x, float y, float max = 0f) =>
        Txt(s, x, y, _fs * 0.86f, _pal.Dim, true, OverlayTextAlign.Left, max, true, 0.08f);

    private static float LineH(float size) => OverlayTypography.LineHeight(size);

    private RectF ToPoints(float x, float y, float w, float h) =>
        new((_c.OriginX + x * _c.Scale) / _ps, (_c.OriginY + y * _c.Scale) / _ps, w * _c.Scale / _ps, h * _c.Scale / _ps);

    /// <summary>The pointer in the current transform's units, or false when it is not over the surface.</summary>
    private bool LocalPointer(out float x, out float y)
    {
        x = (_ctl.PointerX * _ps - _c.OriginX) / _c.Scale;
        y = (_ctl.PointerY * _ps - _c.OriginY) / _c.Scale;
        return _ctl.PointerInside && !_ctl.Edit && !_ctl.Dragging;
    }
}

/// <summary>An integer pixel rectangle (the renderer's region type; converted to OverlayRectangle for backends).</summary>
public readonly record struct OverlayRectI(int X, int Y, int Width, int Height);
