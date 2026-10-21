// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
using SharpEmu.Fonts.Abstractions;

namespace SharpEmu.Libs.Font;

// All access is under FontExports.FaceGate. Each guest handle owns one
// reference, so closing a source does not invalidate its existing instances.
internal sealed class FontFace(IFontFace face)
{
    private int references = 1;
    public void Retain() { if (references == 0) throw new ObjectDisposedException(nameof(FontFace)); references++; }
    public void Release() { if (--references == 0) face.Dispose(); }
    public FontMetrics GetMetrics(float width, float height) => face.GetMetrics(width, height);
    public FontGlyphMetrics GetGlyphMetrics(uint code, float width, float height) => face.GetGlyphMetrics(
        (code & 0xffff0000) == 0x80000000 ? FontCharacter.FromGlyphId((ushort)code) : FontCharacter.FromUnicode(code), width, height);
    public FontGlyphBitmap GetGlyph(uint code, float width, float height) => face.GetGlyph(
        (code & 0xffff0000) == 0x80000000 ? FontCharacter.FromGlyphId((ushort)code) : FontCharacter.FromUnicode(code), width, height);
}
