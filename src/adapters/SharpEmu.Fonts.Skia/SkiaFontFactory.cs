// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
using System.Runtime.InteropServices;
using SharpEmu.Fonts.Abstractions;
using SkiaSharp;

namespace SharpEmu.Fonts.Skia;

public sealed class SkiaFontFactory : IFontFactory
{
    public string BackendName => "skia";
    public IFontFace CreateSystemFace(FontStyle style)
    {
        if ((style & ~(FontStyle.Bold | FontStyle.Italic)) != 0) throw new ArgumentOutOfRangeException(nameof(style));
        var skStyle = style switch
        {
            FontStyle.Bold => SKFontStyle.Bold,
            FontStyle.Italic => SKFontStyle.Italic,
            FontStyle.Bold | FontStyle.Italic => SKFontStyle.BoldItalic,
            _ => SKFontStyle.Normal,
        };
        return Own(SKTypeface.FromFamilyName(null, skStyle)
            ?? throw new InvalidOperationException("Skia could not create the requested system font."));
    }
    public IFontFace? CreateMemoryFace(ReadOnlySpan<byte> bytes, uint collectionIndex)
    {
        if (bytes.IsEmpty || collectionIndex > int.MaxValue) return null;
        using var data = SKData.CreateCopy(bytes);
        var typeface = SKTypeface.FromData(data, (int)collectionIndex);
        return typeface is null ? null : Own(typeface);
    }

    private static IFontFace Own(SKTypeface typeface)
    {
        try { return new Face(typeface); }
        catch { typeface.Dispose(); throw; }
    }

    private sealed class Face(SKTypeface typeface) : IFontFace
    {
        private readonly object _gate = new();
        private SKTypeface? _typeface = typeface;
        private readonly Dictionary<(FontCharacter Character, float Width, float Height), FontGlyphBitmap> _glyphs = [];
        private long _cachedBytes;
        private readonly Lazy<TrueTypeMetrics?> _metrics = new(() => TrueTypeMetrics.TryCreate(typeface));
        public FontGlyphMetrics GetGlyphMetrics(FontCharacter character, float width, float height)
        {
            FontScale.Validate(width, height);
            lock (_gate)
            {
                ObjectDisposedException.ThrowIf(_typeface is null, this);
                if (_glyphs.TryGetValue((character, width, height), out var cached)) return cached.Metrics;
                using var font = CreateFont(width, height);
                var glyph = character.IsGlyphId
                    ? character.Value < _typeface.GlyphCount ? (ushort)character.Value : (ushort)0
                    : font.GetGlyph((int)character.Value);
                SKTypeface? fallback = null;
                try
                {
                    if (glyph == 0 && !character.IsGlyphId)
                    {
                        fallback = SKFontManager.Default.MatchCharacter(_typeface.FamilyName, _typeface.FontStyle, null, (int)character.Value);
                        if (fallback is not null) { font.Typeface = fallback; glyph = font.GetGlyph((int)character.Value); }
                    }
                    var tables = fallback is null ? _metrics.Value : TrueTypeMetrics.TryCreate(fallback);
                    if (tables?.TryMeasure(glyph, width, height, out var measured) == true) return measured;
                    ReadOnlySpan<ushort> glyphs = [glyph];
                    var advance = font.MeasureText(glyphs, out var bounds);
                    var left = (int)MathF.Floor(bounds.Left); var top = (int)MathF.Floor(bounds.Top);
                    var bitmapWidth = (int)MathF.Ceiling(bounds.Right) - left;
                    var bitmapHeight = (int)MathF.Ceiling(bounds.Bottom) - top;
                    return new(Math.Max(bitmapWidth, 0), Math.Max(bitmapHeight, 0), left, -top, advance);
                }
                finally { fallback?.Dispose(); }
            }
        }
        public FontMetrics GetMetrics(float width, float height)
        {
            FontScale.Validate(width, height);
            lock (_gate)
            {
                using var font = CreateFont(width, height);
                var metrics = font.Metrics;
                return new(metrics.Ascent, metrics.Descent, metrics.Leading);
            }
        }
        public FontGlyphBitmap GetGlyph(FontCharacter character, float width, float height)
        {
            FontScale.Validate(width, height);
            lock (_gate)
            {
                ObjectDisposedException.ThrowIf(_typeface is null, this);
                var key = (character, width, height);
                if (_glyphs.TryGetValue(key, out var cached)) return cached;
                var glyph = Rasterize(character, width, height);
                if (_glyphs.Count >= 1024 || _cachedBytes + glyph.Coverage.Length > 16 * 1024 * 1024)
                { _glyphs.Clear(); _cachedBytes = 0; }
                _glyphs.Add(key, glyph); _cachedBytes += glyph.Coverage.Length;
                return glyph;
            }
        }
        private SKFont CreateFont(float width, float height)
        {
            ObjectDisposedException.ThrowIf(_typeface is null, this);
            return new(_typeface, height, width / height) { Edging = SKFontEdging.Antialias, Subpixel = false };
        }
        private FontGlyphBitmap Rasterize(FontCharacter character, float width, float height)
        {
            using var font = CreateFont(width, height);
            var glyph = character.IsGlyphId
                ? character.Value < _typeface!.GlyphCount ? (ushort)character.Value : (ushort)0
                : font.GetGlyph((int)character.Value);
            SKTypeface? fallback = null;
            try
            {
                if (glyph == 0 && !character.IsGlyphId)
                {
                    fallback = SKFontManager.Default.MatchCharacter(_typeface!.FamilyName, _typeface.FontStyle, null, (int)character.Value);
                    if (fallback is not null) { font.Typeface = fallback; glyph = font.GetGlyph((int)character.Value); }
                }
                ReadOnlySpan<ushort> glyphs = [glyph];
                var advance = font.MeasureText(glyphs, out var bounds);
                var left = (int)MathF.Floor(bounds.Left); var top = (int)MathF.Floor(bounds.Top);
                var bitmapWidth = (int)MathF.Ceiling(bounds.Right) - left;
                var bitmapHeight = (int)MathF.Ceiling(bounds.Bottom) - top;
                if (bitmapWidth <= 0 || bitmapHeight <= 0)
                    return new(0, 0, left, -top, advance, [], glyph, glyph != 0, fallback is not null);
                if ((long)bitmapWidth * bitmapHeight > 16 * 1024 * 1024)
                    throw new NotSupportedException("Font glyph coverage exceeds the bounded bitmap budget.");
                using var bitmap = new SKBitmap(bitmapWidth, bitmapHeight, SKColorType.Alpha8, SKAlphaType.Premul);
                using var canvas = new SKCanvas(bitmap);
                using var paint = new SKPaint { Color = SKColors.White, IsAntialias = true };
                using var blob = SKTextBlob.Create(MemoryMarshal.AsBytes(glyphs), SKTextEncoding.GlyphId, font);
                canvas.Clear(SKColors.Transparent); canvas.DrawText(blob!, -left, -top, paint);
                var coverage = new byte[checked(bitmapWidth * bitmapHeight)];
                for (var row = 0; row < bitmapHeight; row++)
                    Marshal.Copy(bitmap.GetPixels() + row * bitmap.RowBytes, coverage, row * bitmapWidth, bitmapWidth);
                return new(bitmapWidth, bitmapHeight, left, -top, advance, coverage, glyph, glyph != 0, fallback is not null);
            }
            finally { fallback?.Dispose(); }
        }
        public void Dispose()
        {
            lock (_gate) { _typeface?.Dispose(); _typeface = null; _glyphs.Clear(); _cachedBytes = 0; }
        }
    }
}
