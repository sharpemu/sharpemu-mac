// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace SharpEmu.Libs.VideoOut.Overlay;

/// <summary>A straight-alpha colour; the canvas premultiplies while blending.</summary>
public readonly record struct OverlayColor(byte R, byte G, byte B, byte A = 255)
{
    public static readonly OverlayColor Transparent = new(0, 0, 0, 0);

    /// <summary>0xRRGGBB with an optional alpha 0..1.</summary>
    public static OverlayColor Hex(int rgb, float alpha = 1f) => new(
        (byte)((rgb >> 16) & 0xFF), (byte)((rgb >> 8) & 0xFF), (byte)(rgb & 0xFF),
        (byte)Math.Clamp((int)MathF.Round(alpha * 255f), 0, 255));

    public OverlayColor WithAlpha(float alpha) =>
        this with { A = (byte)Math.Clamp((int)MathF.Round(alpha * 255f), 0, 255) };

    /// <summary>Multiplies the alpha channel.</summary>
    public OverlayColor Scaled(float factor) =>
        this with { A = (byte)Math.Clamp((int)MathF.Round(A * factor), 0, 255) };

    public static OverlayColor Lerp(OverlayColor a, OverlayColor b, float t)
    {
        t = Math.Clamp(t, 0f, 1f);
        return new((byte)(a.R + (b.R - a.R) * t), (byte)(a.G + (b.G - a.G) * t),
            (byte)(a.B + (b.B - a.B) * t), (byte)(a.A + (b.A - a.A) * t));
    }
}

public readonly record struct OverlayPoint(float X, float Y);

/// <summary>
/// A premultiplied BGRA8 software rasterizer with anti-aliasing. Coordinates passed to the drawing
/// calls are in the current transform space (<see cref="SetTransform"/>); clip rectangles are kept in
/// pixels. Nothing here allocates once the polygon scratch buffer has grown to its working size.
/// </summary>
public sealed class OverlayCanvas
{
    private const int MaxClips = 24;

    private byte[] _bytes = [];
    private readonly int[] _clipStack = new int[MaxClips * 4];
    private int _clipDepth;
    private int _cx0, _cy0, _cx1, _cy1;
    private float[] _accum = new float[4096];

    public int Width { get; private set; }
    public int Height { get; private set; }
    public float OriginX { get; private set; }
    public float OriginY { get; private set; }
    public float Scale { get; private set; } = 1f;

    /// <summary>Multiplies the alpha of everything drawn (the overlay's global opacity), 0..1.</summary>
    public float Opacity { get; set; } = 1f;

    /// <summary>Points the canvas at a pixel buffer (premultiplied BGRA8, stride width*4).</summary>
    public void Attach(byte[] pixels, int width, int height)
    {
        if (pixels.Length < width * height * 4)
        {
            throw new ArgumentException("Pixel buffer is smaller than width*height*4.", nameof(pixels));
        }

        _bytes = pixels;
        Width = width;
        Height = height;
        ResetState();
    }

    private Span<uint> Px => MemoryMarshal.Cast<byte, uint>(_bytes.AsSpan(0, Width * Height * 4));

    public void ResetState()
    {
        _clipDepth = 0;
        _cx0 = 0;
        _cy0 = 0;
        _cx1 = Width;
        _cy1 = Height;
        OriginX = 0;
        OriginY = 0;
        Scale = 1f;
    }

    public void SetTransform(float originX, float originY, float scale)
    {
        OriginX = originX;
        OriginY = originY;
        Scale = scale;
    }

    // ---------------------------------------------------------------- clip

    public void PushClip(float x, float y, float w, float h)
    {
        if (_clipDepth >= MaxClips)
        {
            throw new InvalidOperationException("Overlay clip stack overflow.");
        }

        var o = _clipDepth * 4;
        _clipStack[o] = _cx0;
        _clipStack[o + 1] = _cy0;
        _clipStack[o + 2] = _cx1;
        _clipStack[o + 3] = _cy1;
        _clipDepth++;
        _cx0 = Math.Max(_cx0, (int)MathF.Floor(OriginX + x * Scale));
        _cy0 = Math.Max(_cy0, (int)MathF.Floor(OriginY + y * Scale));
        _cx1 = Math.Min(_cx1, (int)MathF.Ceiling(OriginX + (x + w) * Scale));
        _cy1 = Math.Min(_cy1, (int)MathF.Ceiling(OriginY + (y + h) * Scale));
        if (_cx1 < _cx0) { _cx1 = _cx0; }
        if (_cy1 < _cy0) { _cy1 = _cy0; }
    }

    public void PopClip()
    {
        if (_clipDepth == 0)
        {
            return;
        }

        _clipDepth--;
        var o = _clipDepth * 4;
        _cx0 = _clipStack[o];
        _cy0 = _clipStack[o + 1];
        _cx1 = _clipStack[o + 2];
        _cy1 = _clipStack[o + 3];
    }

    // ---------------------------------------------------------------- blending

    private const uint Lanes = 0x00FF00FFu;

    /// <summary>Per-lane (two 8-bit channels at a time) x * f / 255, rounded.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint MulLanes(uint lanes, uint f)
    {
        var t = lanes * f + 0x00800080u;
        return ((t + ((t >> 8) & Lanes)) >> 8) & Lanes;
    }

    /// <summary>Source-over of a premultiplied colour scaled by <paramref name="cov"/> (0..255).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Blend(ref uint dst, uint pb, uint pg, uint pr, uint pa, uint cov)
        => BlendPacked(ref dst, (pa << 24) | (pr << 16) | (pg << 8) | pb, cov);

    /// <summary>Source-over of a packed premultiplied BGRA colour scaled by <paramref name="cov"/> (0..255).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void BlendPacked(ref uint dst, uint src, uint cov)
    {
        if (cov < 255)
        {
            src = MulLanes(src & Lanes, cov) | (MulLanes((src >> 8) & Lanes, cov) << 8);
        }

        var sa = src >> 24;
        var d = dst;
        if (sa == 255 || d == 0)
        {
            dst = src;
            return;
        }

        var inv = 255 - sa;
        var rb = MulLanes(d & Lanes, inv);
        var ag = MulLanes((d >> 8) & Lanes, inv);
        dst = src + (rb | (ag << 8));
    }

    private readonly struct Pre(OverlayColor c, float opacity)
    {
        public readonly uint A = (uint)Math.Clamp((int)(c.A * opacity + 0.5f), 0, 255);
        public readonly uint B = ((uint)c.B * (uint)Math.Clamp((int)(c.A * opacity + 0.5f), 0, 255) + 127) / 255;
        public readonly uint G = ((uint)c.G * (uint)Math.Clamp((int)(c.A * opacity + 0.5f), 0, 255) + 127) / 255;
        public readonly uint R = ((uint)c.R * (uint)Math.Clamp((int)(c.A * opacity + 0.5f), 0, 255) + 127) / 255;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint ToCoverage(float c) => (uint)(Math.Clamp(c, 0f, 1f) * 255f + 0.5f);

    /// <summary>
    /// When set, solid interior spans store their colour instead of blending over what is there. Only valid
    /// where the caller knows the destination is transparent or about to be fully covered.
    /// </summary>
    public bool ReplaceSolid { get; set; }

    private void SolidSpan(Span<uint> px, int y, int x0, int x1, in Pre p)
    {
        if (x1 <= x0)
        {
            return;
        }

        var row = px.Slice(y * Width + x0, x1 - x0);
        if (p.A == 255 || ReplaceSolid)
        {
            row.Fill(p.A == 255 ? 0xFF000000u | (p.R << 16) | (p.G << 8) | p.B : (p.A << 24) | (p.R << 16) | (p.G << 8) | p.B);
            return;
        }

        var src = (p.A << 24) | (p.R << 16) | (p.G << 8) | p.B;
        var i = 0;
        if (Vector128.IsHardwareAccelerated)
        {
            // Over untouched (transparent) pixels a source-over is a plain store; do that four at a time.
            var vsrc = Vector128.Create(src);
            ref var first = ref MemoryMarshal.GetReference(row);
            for (; i + 4 <= row.Length; i += 4)
            {
                ref var cell = ref Unsafe.Add(ref first, i);
                var d = Vector128.LoadUnsafe(ref cell);
                if (d == Vector128<uint>.Zero)
                {
                    vsrc.StoreUnsafe(ref cell);
                }
                else
                {
                    BlendPacked(ref cell, src, 255);
                    BlendPacked(ref Unsafe.Add(ref cell, 1), src, 255);
                    BlendPacked(ref Unsafe.Add(ref cell, 2), src, 255);
                    BlendPacked(ref Unsafe.Add(ref cell, 3), src, 255);
                }
            }
        }

        for (; i < row.Length; i++)
        {
            BlendPacked(ref row[i], src, 255);
        }
    }

    // ---------------------------------------------------------------- primitives

    /// <summary>
    /// Clears a pixel rectangle except the given rectangles (which the caller overwrites completely).
    /// </summary>
    public void ClearPixelsExcept(int x, int y, int w, int h, ReadOnlySpan<OverlayRectI> keep)
    {
        var x0 = Math.Max(0, x);
        var y0 = Math.Max(0, y);
        var x1 = Math.Min(Width, x + w);
        var y1 = Math.Min(Height, y + h);
        if (x1 <= x0 || y1 <= y0)
        {
            return;
        }

        if (keep.Length == 0 || keep.Length > 32)
        {
            ClearPixels(x, y, w, h);
            return;
        }

        var px = Px;
        Span<int> from = stackalloc int[32];
        Span<int> to = stackalloc int[32];
        for (var yy = y0; yy < y1; yy++)
        {
            var n = 0;
            foreach (var k in keep)
            {
                if (yy < k.Y || yy >= k.Y + k.Height || k.X >= x1 || k.X + k.Width <= x0)
                {
                    continue;
                }

                // Insertion sort by start; at most a few spans per row.
                var s = Math.Max(x0, k.X);
                var e = Math.Min(x1, k.X + k.Width);
                var at = n++;
                while (at > 0 && from[at - 1] > s)
                {
                    from[at] = from[at - 1];
                    to[at] = to[at - 1];
                    at--;
                }

                from[at] = s;
                to[at] = e;
            }

            var cursor = x0;
            for (var i = 0; i < n; i++)
            {
                if (from[i] > cursor)
                {
                    px.Slice(yy * Width + cursor, from[i] - cursor).Clear();
                }

                cursor = Math.Max(cursor, to[i]);
            }

            if (cursor < x1)
            {
                px.Slice(yy * Width + cursor, x1 - cursor).Clear();
            }
        }
    }

    /// <summary>Writes zero (transparent) over a pixel rectangle, clipped to the canvas.</summary>
    public void ClearPixels(int x, int y, int w, int h)
    {
        var x0 = Math.Max(0, x);
        var y0 = Math.Max(0, y);
        var x1 = Math.Min(Width, x + w);
        var y1 = Math.Min(Height, y + h);
        if (x1 <= x0)
        {
            return;
        }

        var px = Px;
        for (var yy = y0; yy < y1; yy++)
        {
            px.Slice(yy * Width + x0, x1 - x0).Clear();
        }
    }

    /// <summary>Axis-aligned rectangle with fractional edge coverage.</summary>
    public void FillRect(float x, float y, float w, float h, OverlayColor color)
    {
        if (color.A == 0 || w <= 0 || h <= 0)
        {
            return;
        }

        FillRectPixels(OriginX + x * Scale, OriginY + y * Scale, OriginX + (x + w) * Scale, OriginY + (y + h) * Scale, color);
    }

    private void FillRectPixels(float px0, float py0, float px1, float py1, OverlayColor color)
    {
        var fx0 = (int)MathF.Floor(px0);
        var fx1 = (int)MathF.Ceiling(px1);
        var iy0 = Math.Max(_cy0, (int)MathF.Floor(py0));
        var iy1 = Math.Min(_cy1, (int)MathF.Ceiling(py1));
        if (fx1 <= Math.Max(_cx0, fx0) || iy1 <= iy0 || fx0 >= _cx1)
        {
            return;
        }

        var p = new Pre(color, Opacity);
        var px = Px;
        for (var iy = iy0; iy < iy1; iy++)
        {
            var cy = MathF.Min(iy + 1, py1) - MathF.Max(iy, py0);
            if (cy <= 0.002f)
            {
                continue;
            }

            var rowBase = iy * Width;
            if (fx1 - fx0 == 1)
            {
                if (fx0 >= _cx0 && fx0 < _cx1)
                {
                    Blend(ref px[rowBase + fx0], p.B, p.G, p.R, p.A, ToCoverage((px1 - px0) * cy));
                }

                continue;
            }

            // Left edge column, solid interior, right edge column.
            var cxl = fx0 + 1 - px0;
            if (fx0 >= _cx0 && fx0 < _cx1)
            {
                Blend(ref px[rowBase + fx0], p.B, p.G, p.R, p.A, ToCoverage(cxl * cy));
            }

            var a = Math.Max(_cx0, fx0 + 1);
            var b = Math.Min(_cx1, fx1 - 1);
            if (b > a)
            {
                if (cy >= 0.998f)
                {
                    SolidSpan(px, iy, a, b, p);
                }
                else
                {
                    var cov = ToCoverage(cy);
                    for (var x = a; x < b; x++)
                    {
                        Blend(ref px[rowBase + x], p.B, p.G, p.R, p.A, cov);
                    }
                }
            }

            var cxr = px1 - (fx1 - 1);
            if (fx1 - 1 >= _cx0 && fx1 - 1 < _cx1)
            {
                Blend(ref px[rowBase + fx1 - 1], p.B, p.G, p.R, p.A, ToCoverage(cxr * cy));
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float RoundRectDistance(float x, float y, float cx, float cy, float hx, float hy, float r)
    {
        var qx = MathF.Abs(x - cx) - (hx - r);
        var qy = MathF.Abs(y - cy) - (hy - r);
        var ox = MathF.Max(qx, 0f);
        var oy = MathF.Max(qy, 0f);
        return MathF.Sqrt(ox * ox + oy * oy) + MathF.Min(MathF.Max(qx, qy), 0f) - r;
    }

    public void FillRoundRect(float x, float y, float w, float h, float radius, OverlayColor color)
    {
        if (color.A == 0 || w <= 0 || h <= 0)
        {
            return;
        }

        var px0 = OriginX + x * Scale;
        var py0 = OriginY + y * Scale;
        var px1 = OriginX + (x + w) * Scale;
        var py1 = OriginY + (y + h) * Scale;
        var r = Math.Clamp(radius * Scale, 0f, MathF.Min(px1 - px0, py1 - py0) / 2f);
        if (r < 0.25f)
        {
            FillRectPixels(px0, py0, px1, py1, color);
            return;
        }

        var ix0 = Math.Max(_cx0, (int)MathF.Floor(px0));
        var iy0 = Math.Max(_cy0, (int)MathF.Floor(py0));
        var ix1 = Math.Min(_cx1, (int)MathF.Ceiling(px1));
        var iy1 = Math.Min(_cy1, (int)MathF.Ceiling(py1));
        if (ix1 <= ix0 || iy1 <= iy0)
        {
            return;
        }

        var p = new Pre(color, Opacity);
        var px = Px;
        float cx = (px0 + px1) / 2f, cy = (py0 + py1) / 2f, hx = (px1 - px0) / 2f, hy = (py1 - py0) / 2f;
        var cornerBand = (int)MathF.Ceiling(r) + 2;
        var yTop = (int)MathF.Floor(py0);
        var yBot = (int)MathF.Ceiling(py1);
        var xLeft = (int)MathF.Floor(px0);
        var xRight = (int)MathF.Ceiling(px1);
        for (var iy = iy0; iy < iy1; iy++)
        {
            var inCorner = iy < yTop + cornerBand || iy >= yBot - cornerBand;
            var edge = inCorner ? cornerBand : 2;
            var leftEnd = Math.Min(ix1, xLeft + edge);
            var rightStart = Math.Max(ix0, xRight - edge);
            var ry = iy + 0.5f;
            var row = iy * Width;
            for (var ix = ix0; ix < leftEnd; ix++)
            {
                var cov = ToCoverage(0.5f - RoundRectDistance(ix + 0.5f, ry, cx, cy, hx, hy, r));
                if (cov != 0)
                {
                    Blend(ref px[row + ix], p.B, p.G, p.R, p.A, cov);
                }
            }

            // Between the corner bands the edge is straight, so coverage depends on the row only.
            var midStart = Math.Max(ix0, leftEnd);
            var midEnd = Math.Max(midStart, rightStart);
            if (midStart < midEnd)
            {
                var midCov = ToCoverage(0.5f - RoundRectDistance(cx, ry, cx, cy, hx, hy, r));
                if (midCov >= 255)
                {
                    SolidSpan(px, iy, midStart, midEnd, p);
                }
                else if (midCov != 0)
                {
                    var src = (p.A << 24) | (p.R << 16) | (p.G << 8) | p.B;
                    for (var ix = midStart; ix < midEnd; ix++)
                    {
                        BlendPacked(ref px[row + ix], src, midCov);
                    }
                }
            }

            for (var ix = midEnd; ix < ix1; ix++)
            {
                var cov = ToCoverage(0.5f - RoundRectDistance(ix + 0.5f, ry, cx, cy, hx, hy, r));
                if (cov != 0)
                {
                    Blend(ref px[row + ix], p.B, p.G, p.R, p.A, cov);
                }
            }
        }
    }

    /// <summary>Inside-aligned rounded-rectangle outline of the given line width.</summary>
    public void StrokeRoundRect(float x, float y, float w, float h, float radius, float lineWidth, OverlayColor color)
    {
        if (color.A == 0 || w <= 0 || h <= 0 || lineWidth <= 0)
        {
            return;
        }

        var px0 = OriginX + x * Scale;
        var py0 = OriginY + y * Scale;
        var px1 = OriginX + (x + w) * Scale;
        var py1 = OriginY + (y + h) * Scale;
        var lw = MathF.Max(lineWidth * Scale, 0.5f);
        var r = Math.Clamp(radius * Scale, 0f, MathF.Min(px1 - px0, py1 - py0) / 2f);
        var ix0 = Math.Max(_cx0, (int)MathF.Floor(px0));
        var iy0 = Math.Max(_cy0, (int)MathF.Floor(py0));
        var ix1 = Math.Min(_cx1, (int)MathF.Ceiling(px1));
        var iy1 = Math.Min(_cy1, (int)MathF.Ceiling(py1));
        if (ix1 <= ix0 || iy1 <= iy0)
        {
            return;
        }

        var p = new Pre(color, Opacity);
        var px = Px;
        float cx = (px0 + px1) / 2f, cy = (py0 + py1) / 2f, hx = (px1 - px0) / 2f, hy = (py1 - py0) / 2f;
        var band = (int)MathF.Ceiling(lw) + 2;
        var corner = Math.Max(band, (int)MathF.Ceiling(r) + 2);
        var yTop = (int)MathF.Floor(py0);
        var yBot = (int)MathF.Ceiling(py1);
        var xLeft = (int)MathF.Floor(px0);
        var xRight = (int)MathF.Ceiling(px1);
        for (var iy = iy0; iy < iy1; iy++)
        {
            var wide = iy < yTop + corner || iy >= yBot - corner;
            var ry = iy + 0.5f;
            var row = iy * Width;
            if (wide)
            {
                // Away from the corners the distance only depends on the row, so one coverage serves the span.
                var midStart = Math.Min(ix1, Math.Max(ix0, xLeft + corner));
                var midEnd = Math.Max(midStart, Math.Min(ix1, xRight - corner));
                StrokeRun(px, row, ix0, midStart, ry, cx, cy, hx, hy, r, lw, p);
                if (midStart < midEnd)
                {
                    var d = RoundRectDistance(cx, ry, cx, cy, hx, hy, r);
                    var c = ToCoverage(Math.Clamp(0.5f - d, 0f, 1f) - Math.Clamp(0.5f - (d + lw), 0f, 1f));
                    if (c != 0)
                    {
                        var src = (p.A << 24) | (p.R << 16) | (p.G << 8) | p.B;
                        for (var ix = midStart; ix < midEnd; ix++)
                        {
                            BlendPacked(ref px[row + ix], src, c);
                        }
                    }
                }

                StrokeRun(px, row, midEnd, ix1, ry, cx, cy, hx, hy, r, lw, p);
            }
            else
            {
                StrokeRun(px, row, ix0, Math.Min(ix1, xLeft + band), ry, cx, cy, hx, hy, r, lw, p);
                StrokeRun(px, row, Math.Max(ix0, xRight - band), ix1, ry, cx, cy, hx, hy, r, lw, p);
            }
        }
    }

    private static void StrokeRun(Span<uint> px, int row, int x0, int x1, float ry, float cx, float cy, float hx, float hy,
        float r, float lw, in Pre p)
    {
        for (var ix = x0; ix < x1; ix++)
        {
            var d = RoundRectDistance(ix + 0.5f, ry, cx, cy, hx, hy, r);
            var cov = Math.Clamp(0.5f - d, 0f, 1f) - Math.Clamp(0.5f - (d + lw), 0f, 1f);
            var c = ToCoverage(cov);
            if (c != 0)
            {
                Blend(ref px[row + ix], p.B, p.G, p.R, p.A, c);
            }
        }
    }

    /// <summary>A line of the given width with round caps; axis-aligned lines take the rectangle path.</summary>
    public void Line(float x0, float y0, float x1, float y1, float width, OverlayColor color)
    {
        if (color.A == 0 || width <= 0)
        {
            return;
        }

        if (x0 == x1 || y0 == y1)
        {
            var hw = width / 2f;
            FillRect(MathF.Min(x0, x1) - (y0 == y1 ? 0 : hw), MathF.Min(y0, y1) - (x0 == x1 ? 0 : hw),
                x0 == x1 ? width : MathF.Abs(x1 - x0), y0 == y1 ? width : MathF.Abs(y1 - y0), color);
            return;
        }

        var ax = OriginX + x0 * Scale;
        var ay = OriginY + y0 * Scale;
        var bx = OriginX + x1 * Scale;
        var by = OriginY + y1 * Scale;
        var half = MathF.Max(width * Scale, 0.6f) / 2f;
        var ix0 = Math.Max(_cx0, (int)MathF.Floor(MathF.Min(ax, bx) - half - 1));
        var iy0 = Math.Max(_cy0, (int)MathF.Floor(MathF.Min(ay, by) - half - 1));
        var ix1 = Math.Min(_cx1, (int)MathF.Ceiling(MathF.Max(ax, bx) + half + 1));
        var iy1 = Math.Min(_cy1, (int)MathF.Ceiling(MathF.Max(ay, by) + half + 1));
        if (ix1 <= ix0 || iy1 <= iy0)
        {
            return;
        }

        var dx = bx - ax;
        var dy = by - ay;
        var len2 = dx * dx + dy * dy;
        var p = new Pre(color, Opacity);
        var px = Px;
        for (var iy = iy0; iy < iy1; iy++)
        {
            var py = iy + 0.5f;
            for (var ix = ix0; ix < ix1; ix++)
            {
                var qx = ix + 0.5f - ax;
                var qy = py - ay;
                var t = len2 > 0 ? Math.Clamp((qx * dx + qy * dy) / len2, 0f, 1f) : 0f;
                var ex = qx - dx * t;
                var ey = qy - dy * t;
                var cov = ToCoverage(half + 0.5f - MathF.Sqrt(ex * ex + ey * ey));
                if (cov != 0)
                {
                    Blend(ref px[iy * Width + ix], p.B, p.G, p.R, p.A, cov);
                }
            }
        }
    }

    /// <summary>A horizontal dashed line (dash and gap lengths in transform units).</summary>
    public void DashedHLine(float x0, float x1, float y, float width, float dash, float gap, OverlayColor color)
    {
        for (var x = x0; x < x1; x += dash + gap)
        {
            FillRect(x, y - width / 2f, MathF.Min(dash, x1 - x), width, color);
        }
    }

    /// <summary>A joined polyline; segments are drawn individually with round caps.</summary>
    public void Polyline(ReadOnlySpan<OverlayPoint> points, float width, OverlayColor color)
    {
        for (var i = 1; i < points.Length; i++)
        {
            Line(points[i - 1].X, points[i - 1].Y, points[i].X, points[i].Y, width, color);
        }
    }

    /// <summary>Anti-aliased polygon fill (signed-area accumulation), clipped to the clip rect.</summary>
    public void FillPolygon(ReadOnlySpan<OverlayPoint> points, OverlayColor color)
    {
        if (color.A == 0 || points.Length < 3)
        {
            return;
        }

        float minX = float.MaxValue, minY = float.MaxValue, maxX = float.MinValue, maxY = float.MinValue;
        for (var i = 0; i < points.Length; i++)
        {
            var x = OriginX + points[i].X * Scale;
            var y = OriginY + points[i].Y * Scale;
            minX = MathF.Min(minX, x);
            maxX = MathF.Max(maxX, x);
            minY = MathF.Min(minY, y);
            maxY = MathF.Max(maxY, y);
        }

        var bx0 = Math.Max(_cx0, (int)MathF.Floor(minX));
        var by0 = Math.Max(_cy0, (int)MathF.Floor(minY));
        var bx1 = Math.Min(_cx1, (int)MathF.Ceiling(maxX));
        var by1 = Math.Min(_cy1, (int)MathF.Ceiling(maxY));
        var bw = bx1 - bx0;
        var bh = by1 - by0;
        if (bw <= 0 || bh <= 0)
        {
            return;
        }

        var stride = bw + 2;
        var need = stride * bh + 2;
        if (_accum.Length < need)
        {
            _accum = new float[Math.Max(need, _accum.Length * 2)];
        }
        else
        {
            Array.Clear(_accum, 0, need);
        }

        for (var i = 0; i < points.Length; i++)
        {
            var a = points[i];
            var b = points[(i + 1) % points.Length];
            AccumulateEdge(
                OriginX + a.X * Scale - bx0, OriginY + a.Y * Scale - by0,
                OriginX + b.X * Scale - bx0, OriginY + b.Y * Scale - by0, bw, bh, stride);
        }

        var p = new Pre(color, Opacity);
        var px = Px;
        for (var y = 0; y < bh; y++)
        {
            var acc = 0f;
            var line = y * stride;
            for (var x = 0; x < bw; x++)
            {
                acc += _accum[line + x];
                var cov = ToCoverage(MathF.Abs(acc));
                if (cov != 0)
                {
                    Blend(ref px[(by0 + y) * Width + bx0 + x], p.B, p.G, p.R, p.A, cov);
                }
            }
        }
    }

    private void AccumulateEdge(float x0, float y0, float x1, float y1, int bw, int bh, int stride)
    {
        if (y0 == y1)
        {
            return;
        }

        var dir = 1f;
        if (y0 > y1)
        {
            (x0, y0, x1, y1) = (x1, y1, x0, y0);
            dir = -1f;
        }

        var dxdy = (x1 - x0) / (y1 - y0);
        var x = x0;
        var yStart = (int)MathF.Max(0f, MathF.Floor(y0));
        if (y0 < 0)
        {
            x -= y0 * dxdy;
        }

        var yEnd = Math.Min(bh, (int)MathF.Ceiling(y1));
        for (var y = yStart; y < yEnd; y++)
        {
            var line = y * stride;
            var dy = MathF.Min(y + 1, y1) - MathF.Max(y, y0);
            var xnext = x + dxdy * dy;
            var d = dy * dir;
            var xa = x;
            var xb = xnext;
            if (xa > xb)
            {
                (xa, xb) = (xb, xa);
            }

            xa = Math.Clamp(xa, 0f, bw);
            xb = Math.Clamp(xb, 0f, bw);
            var x0floor = MathF.Floor(xa);
            var x0i = (int)x0floor;
            var x1ceil = MathF.Ceiling(xb);
            var x1i = (int)x1ceil;
            if (x1i <= x0i + 1)
            {
                var xmf = 0.5f * (xa + xb) - x0floor;
                _accum[line + x0i] += d - d * xmf;
                _accum[line + x0i + 1] += d * xmf;
            }
            else
            {
                var s = 1f / (xb - xa);
                var x0f = xa - x0floor;
                var a0 = 0.5f * s * (1f - x0f) * (1f - x0f);
                var x1f = xb - x1ceil + 1f;
                var am = 0.5f * s * x1f * x1f;
                _accum[line + x0i] += d * a0;
                if (x1i == x0i + 2)
                {
                    _accum[line + x0i + 1] += d * (1f - a0 - am);
                }
                else
                {
                    var a1 = s * (1.5f - x0f);
                    _accum[line + x0i + 1] += d * (a1 - a0);
                    for (var xi = x0i + 2; xi < x1i - 1; xi++)
                    {
                        _accum[line + xi] += d * s;
                    }

                    var a2 = a1 + (x1i - x0i - 3) * s;
                    _accum[line + x1i - 1] += d * (1f - a2 - am);
                }

                _accum[line + x1i] += d * am;
            }

            x = xnext;
        }
    }

    /// <summary>
    /// A donut of consecutive segments starting at 12 o'clock and running clockwise. Segment sizes
    /// are normalised by <paramref name="total"/> when positive, otherwise by their sum; any
    /// remainder shows the track colour.
    /// </summary>
    public void Donut(float cx, float cy, float outerRadius, float thickness, ReadOnlySpan<float> values,
        ReadOnlySpan<OverlayColor> colors, float total, OverlayColor track)
    {
        var count = Math.Min(values.Length, colors.Length);
        if (count == 0 || outerRadius <= 0)
        {
            return;
        }

        var norm = total;
        if (norm <= 0)
        {
            norm = 0;
            for (var i = 0; i < count; i++)
            {
                norm += MathF.Max(0f, values[i]);
            }

            if (norm <= 0)
            {
                norm = 1f;
            }
        }

        var pcx = OriginX + cx * Scale;
        var pcy = OriginY + cy * Scale;
        var ro = outerRadius * Scale;
        var ri = MathF.Max(0f, ro - thickness * Scale);
        var ix0 = Math.Max(_cx0, (int)MathF.Floor(pcx - ro - 1));
        var iy0 = Math.Max(_cy0, (int)MathF.Floor(pcy - ro - 1));
        var ix1 = Math.Min(_cx1, (int)MathF.Ceiling(pcx + ro + 1));
        var iy1 = Math.Min(_cy1, (int)MathF.Ceiling(pcy + ro + 1));
        var px = Px;
        const float twoPi = MathF.PI * 2f;
        for (var iy = iy0; iy < iy1; iy++)
        {
            var qy = iy + 0.5f - pcy;
            for (var ix = ix0; ix < ix1; ix++)
            {
                var qx = ix + 0.5f - pcx;
                var d = MathF.Sqrt(qx * qx + qy * qy);
                var radial = MathF.Min(ro + 0.5f - d, d - ri + 0.5f);
                if (radial <= 0)
                {
                    continue;
                }

                var t = MathF.Atan2(qx, -qy) / twoPi;
                if (t < 0)
                {
                    t += 1f;
                }

                var pos = t * norm;
                var acc = 0f;
                var seg = -1;
                var segLo = 0f;
                var segHi = 0f;
                for (var s = 0; s < count; s++)
                {
                    var v = MathF.Max(0f, values[s]);
                    if (pos < acc + v)
                    {
                        seg = s;
                        segLo = acc;
                        segHi = acc + v;
                        break;
                    }

                    acc += v;
                }

                var c = track;
                if (seg >= 0)
                {
                    c = colors[seg];
                    var arcPx = twoPi * d / norm;
                    var toLo = (pos - segLo) * arcPx;
                    var toHi = (segHi - pos) * arcPx;
                    if (toLo < 0.5f)
                    {
                        c = OverlayColor.Lerp(seg > 0 ? colors[seg - 1] : track, c, 0.5f + toLo);
                    }
                    else if (toHi < 0.5f)
                    {
                        c = OverlayColor.Lerp(seg + 1 < count ? colors[seg + 1] : track, c, 0.5f + toHi);
                    }
                }

                if (c.A == 0)
                {
                    continue;
                }

                var p = new Pre(c, Opacity);
                Blend(ref px[iy * Width + ix], p.B, p.G, p.R, p.A, ToCoverage(radial));
            }
        }
    }

    /// <summary>A filled disc with anti-aliased edge.</summary>
    public void FillCircle(float cx, float cy, float radius, OverlayColor color) =>
        FillRoundRect(cx - radius, cy - radius, radius * 2f, radius * 2f, radius, color);

    // ---------------------------------------------------------------- masks

    /// <summary>Blends a tightly packed Alpha8 coverage mask (a glyph) at an integer pixel position.</summary>
    public void BlendMask(int x, int y, int w, int h, ReadOnlySpan<byte> alpha, OverlayColor color)
    {
        if (color.A == 0 || w <= 0 || h <= 0)
        {
            return;
        }

        var ix0 = Math.Max(_cx0, x);
        var iy0 = Math.Max(_cy0, y);
        var ix1 = Math.Min(_cx1, x + w);
        var iy1 = Math.Min(_cy1, y + h);
        if (ix1 <= ix0 || iy1 <= iy0)
        {
            return;
        }

        var p = new Pre(color, Opacity);
        var packed = (p.A << 24) | (p.R << 16) | (p.G << 8) | p.B;
        var px = Px;
        for (var iy = iy0; iy < iy1; iy++)
        {
            var src = (iy - y) * w - x;
            var row = iy * Width;
            for (var ix = ix0; ix < ix1; ix++)
            {
                var a = alpha[src + ix];
                if (a != 0)
                {
                    BlendPacked(ref px[row + ix], packed, a);
                }
            }
        }
    }

    /// <summary>Reads back one pixel (premultiplied, packed as 0xAARRGGBB).</summary>
    public uint GetPixel(int x, int y) => Px[y * Width + x];
}
