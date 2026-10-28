// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.CompilerServices;

namespace SharpEmu.Libs.Audio;

// Lets AjmExports.Trace($"...") skip building the message (and evaluating its holes) when
// SHARPEMU_LOG_AJM is off. Every AJM call traces, so an eagerly formatted string is a per-call allocation.
[InterpolatedStringHandler]
internal ref struct AjmTraceInterpolatedStringHandler
{
    private DefaultInterpolatedStringHandler _inner;
    private readonly bool _enabled;

    public AjmTraceInterpolatedStringHandler(int literalLength, int formattedCount, out bool shouldAppend)
    {
        _enabled = AjmExports.TraceEnabled;
        shouldAppend = _enabled;
        _inner = _enabled ? new DefaultInterpolatedStringHandler(literalLength, formattedCount) : default;
    }

    public void AppendLiteral(string value) => _inner.AppendLiteral(value);

    public void AppendFormatted<T>(T value) => _inner.AppendFormatted(value);

    public void AppendFormatted<T>(T value, string? format) => _inner.AppendFormatted(value, format);

    public void AppendFormatted<T>(T value, int alignment) => _inner.AppendFormatted(value, alignment);

    public void AppendFormatted<T>(T value, int alignment, string? format) =>
        _inner.AppendFormatted(value, alignment, format);

    public void AppendFormatted(string? value) => _inner.AppendFormatted(value);

    internal bool Enabled => _enabled;

    internal string ToStringAndClear() => _enabled ? _inner.ToStringAndClear() : string.Empty;
}
