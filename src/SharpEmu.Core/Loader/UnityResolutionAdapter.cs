// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Globalization;
using System.Text;
using SharpEmu.HLE;

namespace SharpEmu.Core.Loader;

/// <summary>
/// Unity on PS5 sizes its display from <c>/app0/Media/boot.config</c>, not from the output status:
/// ScreenManagerPS5 takes <c>platform-ps5-video-out-width</c> (height = width x 9 / 16), and
/// SetResolutionImmediate keeps it at least 1920 and at most 3840 on a 4K output; below 4K the
/// most is 1920 when <c>platform-ps5-initialization-flags</c> has bit 1 (value 2) set, 3840
/// otherwise. Superliminal ships 2560 with that bit set, so it renders 1920x1080 on every
/// display short of 4K. This rewrites the width in the file as the title reads it, clears the
/// cap bit when the width is above 1920 (the bit has no other reader), and for a width below
/// 1920 lowers the minimum the executable checks (<c>cmp edx, 0x77F</c>).
/// </summary>
internal static class UnityResolutionAdapter
{
    internal const string WidthKey = "platform-ps5-video-out-width";
    internal const string FlagsKey = "platform-ps5-initialization-flags";
    internal const uint CapToHdFlag = 2;

    // cmp edx, 0x77F; jg +rel8; mov esi, 0x780: SetResolutionImmediate's "width below 1920" check.
    private static readonly byte[] MinimumWidthSignature = [0x81, 0xFA, 0x7F, 0x07, 0x00, 0x00, 0x7F];
    private static readonly byte[] MinimumWidthTail = [0xBE, 0x80, 0x07, 0x00, 0x00];

    internal static void Apply(
        ICpuMemory memory,
        IReadOnlyList<EngineResolution.Segment> segments,
        string? app0Root,
        uint width,
        uint height)
    {
        if (string.IsNullOrEmpty(app0Root))
        {
            return;
        }

        var path = Path.Combine(app0Root, "Media", "boot.config");
        if (EngineResolution.TryReadApp0File(path) is not { } original ||
            RewriteBootConfig(original, width) is not { } rewritten)
        {
            return;
        }

        if (width < 1920)
        {
            var site = FindMinimumWidthCheck(segments);
            if (site is null)
            {
                EngineResolution.Warn("Unity", $"no minimum-width check found; keeping 1920x1080 for a {width}x{height} display.");
                return;
            }

            Span<byte> immediate = stackalloc byte[4];
            BinaryPrimitives.WriteUInt32LittleEndian(immediate, width - 1);
            if (!memory.TryWrite(site.Value + 2, immediate))
            {
                EngineResolution.Warn("Unity", "the code segment refused the minimum-width patch.");
                return;
            }
        }

        if (!rewritten.AsSpan().SequenceEqual(original) &&
            !EngineResolution.TryReplaceApp0File(path, rewritten, "Unity boot.config video-out width"))
        {
            EngineResolution.Warn("Unity", "boot.config on disk cannot change length; left as shipped.");
            return;
        }

        EngineResolution.Report("Unity", "boot.config", width, width * 9 / 16,
            $"{WidthKey}={width}" + (width < 1920 ? $", minimum width {width}" : string.Empty));
    }

    /// <summary>boot.config with the video-out width set to <paramref name="width"/> and, above
    /// 1920, the 1080p cap bit cleared. Null when the file is not a Unity PS5 boot.config.</summary>
    internal static byte[]? RewriteBootConfig(byte[] original, uint width)
    {
        var text = Encoding.UTF8.GetString(original);
        var lines = text.Split('\n');
        var foundWidth = false;
        for (var index = 0; index < lines.Length; index++)
        {
            var line = lines[index];
            var trailing = line.EndsWith('\r') ? "\r" : string.Empty;
            var body = line.TrimEnd('\r');
            var equals = body.IndexOf('=');
            if (equals <= 0)
            {
                continue;
            }

            var key = body[..equals].Trim();
            var value = body[(equals + 1)..].Trim();
            if (string.Equals(key, WidthKey, StringComparison.Ordinal))
            {
                foundWidth = true;
                lines[index] = $"{body[..(equals + 1)]}{width.ToString(CultureInfo.InvariantCulture)}{trailing}";
            }
            else if (string.Equals(key, FlagsKey, StringComparison.Ordinal) && width > 1920 &&
                uint.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var flags) &&
                (flags & CapToHdFlag) != 0)
            {
                lines[index] = $"{body[..(equals + 1)]}{(flags & ~CapToHdFlag).ToString(CultureInfo.InvariantCulture)}{trailing}";
            }
        }

        return foundWidth ? Encoding.UTF8.GetBytes(string.Join('\n', lines)) : null;
    }

    /// <summary>The address of the <c>cmp edx, 0x77F</c> SetResolutionImmediate uses to refuse a
    /// width below 1920, when exactly one executable site matches.</summary>
    internal static ulong? FindMinimumWidthCheck(IReadOnlyList<EngineResolution.Segment> segments)
    {
        ulong? found = null;
        foreach (var segment in segments)
        {
            if (!segment.Executable)
            {
                continue;
            }

            var span = segment.Bytes.AsSpan();
            var from = 0;
            while (from < span.Length)
            {
                var hit = span[from..].IndexOf(MinimumWidthSignature);
                if (hit < 0)
                {
                    break;
                }

                var offset = from + hit;
                var tail = offset + MinimumWidthSignature.Length + 1;
                if (tail + MinimumWidthTail.Length <= span.Length && span.Slice(tail, MinimumWidthTail.Length).SequenceEqual(MinimumWidthTail))
                {
                    if (found is not null)
                    {
                        return null;
                    }

                    found = segment.Start + (ulong)offset;
                }

                from = offset + 1;
            }
        }

        return found;
    }
}
