// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
using System.Text;

namespace SharpEmu.Fonts.Abstractions;

[Flags]
public enum FontStyle : byte { Normal = 0, Bold = 1, Italic = 2 }

/// <summary>A Unicode scalar or a glyph ID belonging to the selected face.</summary>
public readonly record struct FontCharacter
{
    public uint Value { get; }
    public bool IsGlyphId { get; }
    private FontCharacter(uint value, bool glyphId) { Value = value; IsGlyphId = glyphId; }
    public static FontCharacter FromUnicode(uint scalar) => Rune.IsValid((int)scalar)
        ? new(scalar, false) : throw new ArgumentOutOfRangeException(nameof(scalar));
    public static FontCharacter FromGlyphId(ushort glyphId) => new(glyphId, true);
}

/// <summary>Pixel metrics: ascent is negative, descent is positive, and leading is extra line spacing.</summary>
public readonly record struct FontMetrics(float Ascent, float Descent, float Leading);

/// <summary>Pixel bounds, bearings and horizontal advance without raster coverage.</summary>
public readonly record struct FontGlyphMetrics(int Width, int Height, float BearingX, float BearingY, float Advance);

/// <summary>
/// Immutable, tightly packed, top-to-bottom Alpha8 coverage. Construction copies
/// the supplied bytes; the result remains valid after its face is disposed.
/// Bearings locate the left edge and top above the baseline; advance is horizontal.
/// Missing glyphs may render the face's .notdef glyph, with HasGlyph false.
/// </summary>
public sealed class FontGlyphBitmap
{
    private readonly byte[] _coverage;
    public int Width { get; }
    public int Height { get; }
    public float BearingX { get; }
    public float BearingY { get; }
    public float Advance { get; }
    public ushort GlyphId { get; }
    public bool HasGlyph { get; }
    public bool UsedFallback { get; }
    public ReadOnlyMemory<byte> Coverage => _coverage;
    public FontGlyphMetrics Metrics => new(Width, Height, BearingX, BearingY, Advance);
    public FontGlyphBitmap(int width, int height, float bearingX, float bearingY, float advance,
        ReadOnlySpan<byte> coverage, ushort glyphId, bool hasGlyph, bool usedFallback = false)
    {
        if (width < 0 || height < 0 || (long)width * height > 16 * 1024 * 1024 ||
            coverage.Length != (long)width * height ||
            !float.IsFinite(bearingX) || !float.IsFinite(bearingY) || !float.IsFinite(advance))
            throw new ArgumentException("Font glyph geometry and owned coverage must agree.");
        Width = width; Height = height; BearingX = bearingX; BearingY = bearingY; Advance = advance;
        GlyphId = glyphId; HasGlyph = hasGlyph; UsedFallback = usedFallback;
        _coverage = coverage.ToArray();
    }
}

/// <summary>
/// Owns one face. Calls and disposal must be serialized by its owner. Pixel width
/// scales horizontal geometry independently of pixel height; no shaping is implied.
/// Glyph-ID requests never select a fallback face. Scalar requests may use coverage
/// fallback, reported on their bitmap. No backend handles escape this boundary.
/// </summary>
public interface IFontFace : IDisposable
{
    FontMetrics GetMetrics(float pixelWidth, float pixelHeight);
    FontGlyphBitmap GetGlyph(FontCharacter character, float pixelWidth, float pixelHeight);
    FontGlyphMetrics GetGlyphMetrics(FontCharacter character, float pixelWidth, float pixelHeight) =>
        GetGlyph(character, pixelWidth, pixelHeight).Metrics;
}

/// <summary>Creates owned faces; font bytes are copied or retained independently before returning.</summary>
public interface IFontFactory
{
    string BackendName { get; }
    IFontFace CreateSystemFace(FontStyle style);
    /// <summary>Returns null for invalid data or an unsupported collection index, never a system substitute.</summary>
    IFontFace? CreateMemoryFace(ReadOnlySpan<byte> data, uint collectionIndex);
}

public static class FontScale
{
    public static void Validate(float width, float height)
    {
        if (!float.IsFinite(width) || !float.IsFinite(height) || width <= 0 || height <= 0 || width > 4096 || height > 4096)
            throw new ArgumentOutOfRangeException(nameof(width), "Font pixel dimensions must be finite and in (0, 4096].");
    }
}
