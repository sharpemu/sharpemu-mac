// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Fonts.Abstractions;

namespace SharpEmu.Libs.VideoOut.Overlay;

public enum OverlayTextAlign : byte { Left, Center, Right }

/// <summary>How a run of text is drawn. Sizes are in the canvas transform's units.</summary>
public readonly record struct OverlayTextStyle(
    bool Bold = false,
    bool Upper = false,
    float Spacing = 0f,
    OverlayTextAlign Align = OverlayTextAlign.Left,
    float MaxWidth = 0f);

/// <summary>
/// Glyph cache and text layout over an <see cref="IFontFace"/> pair (regular + bold), with tabular
/// digits and ellipsis. When no font factory is available (or it throws) the pre-existing 5x7 bitmap
/// font is used so the overlay still shows text.
/// </summary>
public sealed class OverlayTypography : IDisposable
{
    private const int MaxCachedGlyphs = 6000;
    private const float LineHeightEm = 1.25f;

    private readonly OverlayCanvas _canvas;
    private IFontFace? _regular;
    private IFontFace? _bold;
    private readonly Dictionary<long, Glyph> _glyphs = [];
    private readonly Dictionary<long, SizeMetrics> _sizes = [];
    private readonly byte[] _coverageLut = new byte[256];

    public OverlayTypography(OverlayCanvas canvas, IFontFactory? factory, OverlayFont font = OverlayFont.Inter)
    {
        _canvas = canvas;
        Font = font;
        for (var i = 0; i < 256; i++)
        {
            // A slight boost keeps light text on dark panels from looking thin.
            _coverageLut[i] = (byte)Math.Clamp((int)MathF.Round(MathF.Pow(i / 255f, 0.82f) * 255f), 0, 255);
        }

        try
        {
            if (factory is not null)
            {
                if (BundledFaceNames(font) is var (regularName, boldName))
                {
                    _regular = CreateBundledFace(factory, regularName);
                    _bold = CreateBundledFace(factory, boldName);
                    if (_regular is null || _bold is null)
                    {
                        DisposeFaces();
                    }
                }

                _regular ??= factory.CreateSystemFace(FontStyle.Normal);
                _bold ??= factory.CreateSystemFace(FontStyle.Bold);
            }
        }
        catch (Exception)
        {
            DisposeFaces();
        }

        if (_regular is null || _bold is null)
        {
            DisposeFaces();
            var fallback = new BuiltInBitmapFontFactory();
            _regular = fallback.CreateSystemFace(FontStyle.Normal);
            _bold = fallback.CreateSystemFace(FontStyle.Bold);
            UsesBuiltInFont = true;
        }
    }

    public bool UsesBuiltInFont { get; }

    /// <summary>The typeface asked for; a bundled one falls back to the system face if it cannot load.</summary>
    public OverlayFont Font { get; }

    private static (string Regular, string Bold)? BundledFaceNames(OverlayFont font) => font switch
    {
        OverlayFont.Inter => ("Inter-Regular.ttf", "Inter-SemiBold.ttf"),
        OverlayFont.JetBrainsMono => ("JetBrainsMono-Regular.ttf", "JetBrainsMono-Bold.ttf"),
        _ => null,
    };

    private static readonly Dictionary<string, byte[]?> BundledFontBytes = new(StringComparer.Ordinal);

    /// <summary>The bytes of a font embedded in this assembly, or null when it is missing.</summary>
    internal static byte[]? BundledFont(string fileName)
    {
        lock (BundledFontBytes)
        {
            if (BundledFontBytes.TryGetValue(fileName, out var cached))
            {
                return cached;
            }

            using var stream = typeof(OverlayTypography).Assembly.GetManifestResourceStream("SharpEmu.Overlay.Fonts." + fileName);
            byte[]? bytes = null;
            if (stream is not null)
            {
                bytes = new byte[stream.Length];
                stream.ReadExactly(bytes);
            }

            BundledFontBytes[fileName] = bytes;
            return bytes;
        }
    }

    private static IFontFace? CreateBundledFace(IFontFactory factory, string fileName) =>
        BundledFont(fileName) is { } bytes ? factory.CreateMemoryFace(bytes, 0) : null;

    private sealed class Glyph(byte[] coverage, int width, int height, float bearingX, float bearingY, float advance)
    {
        public readonly byte[] Coverage = coverage;
        public readonly int Width = width;
        public readonly int Height = height;
        public readonly float BearingX = bearingX;
        public readonly float BearingY = bearingY;
        public readonly float Advance = advance;
    }

    private readonly record struct SizeMetrics(float Ascent, float Descent, float DigitAdvance);

    /// <summary>Line box height for a font size.</summary>
    public static float LineHeight(float size) => size * LineHeightEm;

    private static long SizeKey(float pixelSize, bool bold) => ((long)MathF.Round(pixelSize * 4f) << 1) | (bold ? 1L : 0L);

    private SizeMetrics MetricsFor(float pixelSize, bool bold)
    {
        var key = SizeKey(pixelSize, bold);
        if (_sizes.TryGetValue(key, out var cached))
        {
            return cached;
        }

        var px = (key >> 1) / 4f;
        var face = bold ? _bold! : _regular!;
        var m = face.GetMetrics(px, px);
        var digit = 0f;
        for (var d = '0'; d <= '9'; d++)
        {
            digit = MathF.Max(digit, GlyphFor(d, px, bold).Advance);
        }

        var result = new SizeMetrics(-m.Ascent, m.Descent, digit);
        _sizes[key] = result;
        return result;
    }

    private Glyph GlyphFor(int codePoint, float pixelSize, bool bold)
    {
        var sizeKey = SizeKey(pixelSize, bold);
        var key = (sizeKey << 24) | (uint)(codePoint & 0xFFFFFF);
        if (_glyphs.TryGetValue(key, out var glyph))
        {
            return glyph;
        }

        if (_glyphs.Count >= MaxCachedGlyphs)
        {
            _glyphs.Clear();
            _sizes.Clear();
        }

        var px = (sizeKey >> 1) / 4f;
        var face = bold ? _bold! : _regular!;
        FontGlyphBitmap bitmap;
        try
        {
            bitmap = face.GetGlyph(FontCharacter.FromUnicode((uint)codePoint), px, px);
        }
        catch (Exception)
        {
            bitmap = face.GetGlyph(FontCharacter.FromUnicode('?'), px, px);
        }

        var src = bitmap.Coverage.Span;
        var lut = new byte[src.Length];
        for (var i = 0; i < src.Length; i++)
        {
            lut[i] = _coverageLut[src[i]];
        }

        glyph = new Glyph(lut, bitmap.Width, bitmap.Height, bitmap.BearingX, bitmap.BearingY, bitmap.Advance);
        _glyphs[key] = glyph;
        return glyph;
    }

    private static int Normalize(char c, bool upper)
    {
        if (upper)
        {
            c = char.ToUpperInvariant(c);
        }

        return c;
    }

    private float AdvanceOf(char c, float px, bool bold, bool upper, float spacingPx, bool tabular = false)
    {
        var cp = Normalize(c, upper);
        if (tabular)
        {
            return MetricsFor(px, bold).DigitAdvance + spacingPx;
        }

        return GlyphFor(cp, px, bold).Advance + spacingPx;
    }

    private static bool IsNumeric(char c) => c is >= '0' and <= '9' or '.' or ',';

    /// <summary>
    /// Digits inside a number take one shared width so changing figures do not jitter; a digit
    /// standing alone in a label ("1% low") keeps its own width.
    /// </summary>
    private static bool Tabular(ReadOnlySpan<char> text, int index) =>
        text[index] is >= '0' and <= '9' &&
        ((index > 0 && IsNumeric(text[index - 1])) || (index + 1 < text.Length && IsNumeric(text[index + 1])));

    private float MeasurePixels(ReadOnlySpan<char> text, float px, bool bold, bool upper, float spacingPx)
    {
        var width = 0f;
        for (var i = 0; i < text.Length; i++)
        {
            width += AdvanceOf(text[i], px, bold, upper, spacingPx, Tabular(text, i));
        }

        return width;
    }

    /// <summary>Distance from a line box's top to its text baseline, in transform units.</summary>
    public float BaselineOffset(float size, bool bold = false)
    {
        var scale = _canvas.Scale;
        var m = MetricsFor(MathF.Max(1f, size * scale), bold);
        return ((size * LineHeightEm * scale - (m.Ascent + m.Descent)) / 2f + m.Ascent) / scale;
    }

    /// <summary>Width of a text run in transform units.</summary>
    public float Measure(ReadOnlySpan<char> text, float size, OverlayTextStyle style = default) =>
        MeasurePixels(text, size * _canvas.Scale, style.Bold, style.Upper, style.Spacing * size * _canvas.Scale)
        / _canvas.Scale;

    /// <summary>
    /// Draws text with its line box's top at <paramref name="y"/>; returns the drawn width in transform
    /// units. Text wider than <see cref="OverlayTextStyle.MaxWidth"/> is cut with an ellipsis.
    /// </summary>
    public float Draw(ReadOnlySpan<char> text, float x, float y, float size, OverlayColor color, OverlayTextStyle style = default)
    {
        if (text.IsEmpty || color.A == 0)
        {
            return 0f;
        }

        var scale = _canvas.Scale;
        var px = MathF.Max(1f, size * scale);
        var spacingPx = style.Spacing * px;
        var bold = style.Bold;
        var upper = style.Upper;
        var total = MeasurePixels(text, px, bold, upper, spacingPx);
        var limit = style.MaxWidth > 0 ? style.MaxWidth * scale : float.MaxValue;
        var count = text.Length;
        var ellipsis = false;
        var drawn = total;
        if (total > limit)
        {
            var ellipsisWidth = AdvanceOf('…', px, bold, false, 0f);
            var budget = limit - ellipsisWidth;
            var acc = 0f;
            count = 0;
            while (count < text.Length)
            {
                var adv = AdvanceOf(text[count], px, bold, upper, spacingPx, Tabular(text, count));
                if (acc + adv > budget)
                {
                    break;
                }

                acc += adv;
                count++;
            }

            ellipsis = true;
            drawn = acc + ellipsisWidth;
        }

        var left = _canvas.OriginX + x * scale;
        var penX = style.Align switch
        {
            OverlayTextAlign.Center => left - drawn / 2f,
            OverlayTextAlign.Right => left - drawn,
            _ => left,
        };

        var metrics = MetricsFor(px, bold);
        var lineBox = size * LineHeightEm * scale;
        var baseline = MathF.Round(_canvas.OriginY + y * scale + (lineBox - (metrics.Ascent + metrics.Descent)) / 2f + metrics.Ascent);
        for (var i = 0; i < count; i++)
        {
            penX = DrawGlyph(Normalize(text[i], upper), penX, baseline, px, bold, spacingPx, color, Tabular(text, i));
        }

        if (ellipsis)
        {
            DrawGlyph('…', penX, baseline, px, bold, 0f, color, false);
        }

        return drawn / scale;
    }

    private float DrawGlyph(int codePoint, float penX, float baseline, float px, bool bold, float spacingPx, OverlayColor color,
        bool tabular)
    {
        var glyph = GlyphFor(codePoint, px, bold);
        var advance = tabular ? MetricsFor(px, bold).DigitAdvance : glyph.Advance;
        // A narrow digit sits in the middle of its tabular cell, as a font's own tabular figures do.
        var offset = tabular ? (advance - glyph.Advance) / 2f : 0f;
        if (glyph.Width > 0 && glyph.Height > 0)
        {
            _canvas.BlendMask((int)MathF.Round(penX + offset + glyph.BearingX), (int)(baseline - glyph.BearingY + 0.5f),
                glyph.Width, glyph.Height, glyph.Coverage, color);
        }

        return penX + advance + spacingPx;
    }

    public void Dispose() => DisposeFaces();

    private void DisposeFaces()
    {
        _regular?.Dispose();
        _bold?.Dispose();
        _regular = null;
        _bold = null;
    }
}

/// <summary>The 5x7 bitmap face used when no system font factory is usable.</summary>
internal sealed class BuiltInBitmapFontFactory : IFontFactory
{
    public string BackendName => "builtin";
    public IFontFace CreateSystemFace(FontStyle style) => new Face((style & FontStyle.Bold) != 0);
    public IFontFace? CreateMemoryFace(ReadOnlySpan<byte> data, uint collectionIndex) => null;

    private sealed class Face(bool bold) : IFontFace
    {
        private const int Columns = 5;
        private const int Rows = 7;

        private static int PixelsPerDot(float pixelHeight) => Math.Max(1, (int)MathF.Round(pixelHeight / 9f));

        public FontMetrics GetMetrics(float pixelWidth, float pixelHeight)
        {
            var s = PixelsPerDot(pixelHeight);
            return new FontMetrics(-Rows * s, 2 * s, 0);
        }

        public FontGlyphBitmap GetGlyph(FontCharacter character, float pixelWidth, float pixelHeight)
        {
            var s = PixelsPerDot(pixelHeight);
            var c = character.IsGlyphId ? '?' : (char)Math.Min(character.Value, 0xFFFF);
            c = c switch { '×' => 'X', '·' => '-', '…' => '.', _ => char.ToUpperInvariant(c) };
            var advance = (Columns + 1) * s;
            if (c < ' ' || c > 'Z')
            {
                c = '?';
            }

            var width = Columns * s + (bold ? 1 : 0);
            var height = Rows * s;
            var coverage = new byte[width * height];
            var glyph = PerfOverlay.Font.Slice((c - ' ') * Columns, Columns);
            for (var col = 0; col < Columns; col++)
            {
                for (var row = 0; row < Rows; row++)
                {
                    if ((glyph[col] & (1 << row)) == 0)
                    {
                        continue;
                    }

                    for (var dy = 0; dy < s; dy++)
                    {
                        for (var dx = 0; dx < s + (bold ? 1 : 0); dx++)
                        {
                            var x = col * s + dx;
                            if (x < width)
                            {
                                coverage[(row * s + dy) * width + x] = 255;
                            }
                        }
                    }
                }
            }

            return new FontGlyphBitmap(width, height, 0, Rows * s, advance, coverage, 0, true);
        }

        public void Dispose()
        {
        }
    }
}
