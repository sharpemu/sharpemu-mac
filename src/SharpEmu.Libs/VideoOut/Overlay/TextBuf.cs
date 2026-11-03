// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;

namespace SharpEmu.Libs.VideoOut.Overlay;

/// <summary>
/// A reusable character buffer with invariant-culture number formatting, so the widgets can
/// build their strings on every redraw without allocating.
/// </summary>
internal sealed class TextBuf
{
    private static readonly string[] FixedFormats = ["F0", "F1", "F2", "F3"];
    private char[] _chars = new char[256];
    private int _length;

    public ReadOnlySpan<char> Span => _chars.AsSpan(0, _length);
    public int Length => _length;

    public TextBuf Clear()
    {
        _length = 0;
        return this;
    }

    private void Ensure(int extra)
    {
        if (_length + extra > _chars.Length)
        {
            Array.Resize(ref _chars, Math.Max(_chars.Length * 2, _length + extra));
        }
    }

    public TextBuf Append(ReadOnlySpan<char> text)
    {
        Ensure(text.Length);
        text.CopyTo(_chars.AsSpan(_length));
        _length += text.Length;
        return this;
    }

    public TextBuf Append(string text) => Append(text.AsSpan());

    public TextBuf Append(char c)
    {
        Ensure(1);
        _chars[_length++] = c;
        return this;
    }

    public TextBuf AppendFixed(double value, int decimals)
    {
        if (!double.IsFinite(value))
        {
            return Append('-');
        }

        Ensure(32);
        if (value.TryFormat(_chars.AsSpan(_length), out var written, FixedFormats[Math.Clamp(decimals, 0, 3)],
                CultureInfo.InvariantCulture))
        {
            _length += written;
        }

        return this;
    }

    public TextBuf AppendInt(long value)
    {
        Ensure(24);
        if (value.TryFormat(_chars.AsSpan(_length), out var written, default, CultureInfo.InvariantCulture))
        {
            _length += written;
        }

        return this;
    }

    /// <summary>Two digits, zero padded.</summary>
    public TextBuf AppendTwo(long value)
    {
        value = Math.Clamp(value, 0, 99);
        Append((char)('0' + value / 10));
        return Append((char)('0' + value % 10));
    }

    /// <summary>Milliseconds with the prototype's precision rules; seconds from 1000 ms.</summary>
    public TextBuf AppendMs(double ms, bool unit = false)
    {
        if (!double.IsFinite(ms))
        {
            return Append('-');
        }

        if (ms >= 1000)
        {
            AppendFixed(ms / 1000.0, 2);
            return Append(" s");
        }

        AppendFixed(ms, ms >= 100 ? 0 : ms >= 10 ? 1 : 2);
        return unit ? Append(" ms") : this;
    }

    public TextBuf AppendFps(double fps)
    {
        if (!double.IsFinite(fps))
        {
            return Append('-');
        }

        return AppendFixed(fps, fps >= 100 ? 0 : fps >= 10 ? 1 : 2);
    }

    public TextBuf AppendPercent(double percent) =>
        double.IsFinite(percent) ? AppendFixed(Math.Clamp(percent, 0, 999), 0).Append('%') : Append('-');

    /// <summary>"6.1 GB" from 1 GiB up, "640 MB" below.</summary>
    public TextBuf AppendBytes(ulong bytes)
    {
        const double MiB = 1024.0 * 1024.0;
        const double GiB = 1024.0 * MiB;
        return bytes >= GiB ? AppendFixed(bytes / GiB, 1).Append(" GB") : AppendFixed(bytes / MiB, 0).Append(" MB");
    }

    /// <summary>"1.2k" style counts.</summary>
    public TextBuf AppendCount(double value)
    {
        if (value >= 1000)
        {
            return AppendFixed(value / 1000.0, 1).Append('k');
        }

        return AppendFixed(value, 0);
    }

    /// <summary>"mm:ss" or "h:mm:ss".</summary>
    public TextBuf AppendElapsed(long seconds)
    {
        seconds = Math.Max(0, seconds);
        if (seconds >= 3600)
        {
            AppendInt(seconds / 3600).Append(':');
        }

        AppendTwo(seconds / 60 % 60).Append(':');
        return AppendTwo(seconds % 60);
    }
}
