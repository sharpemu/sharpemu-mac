// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.HLE;

namespace SharpEmu.Core.Loader;

/// <summary>
/// Nightdive's KEX engine (Quake) picks its render size from the output class alone: 3840x2160
/// on a 4K output, 1920x1080 otherwise, written to the render-size console variables, and the
/// display buffers follow the render target. So a 1440p or 720p display still gets 1080p. The
/// selection is one sequence, <c>cmp dword [rbp-0x60], 2; mov edi, 2160; mov ebx, 1080;
/// mov eax, 3840; mov esi, 1920; cmove ...</c>; this sets both outcomes to the advertised size.
/// </summary>
internal static class KexResolutionPatch
{
    internal static readonly byte[] Signature =
    [
        0x83, 0x7D, 0xA0, 0x02,
        0xBF, 0x70, 0x08, 0x00, 0x00,
        0xBB, 0x38, 0x04, 0x00, 0x00,
        0xB8, 0x00, 0x0F, 0x00, 0x00,
        0xBE, 0x80, 0x07, 0x00, 0x00,
    ];

    // Immediate offsets inside the signature: 4K height, HD height, 4K width, HD width.
    private const int UhdHeight = 5;
    private const int HdHeight = 10;
    private const int UhdWidth = 15;
    private const int HdWidth = 20;

    internal static void Apply(ICpuMemory memory, IReadOnlyList<EngineResolution.Segment> segments, uint width, uint height)
    {
        if (Find(segments) is not { } site)
        {
            return;
        }

        if (width == 1920 && height == 1080)
        {
            EngineResolution.Report("KEX", "render size", width, height, "the title's own size");
            return;
        }

        var bytes = BuildReplacement(width, height);
        if (!memory.TryWrite(site, bytes))
        {
            EngineResolution.Warn("KEX", "the code segment refused the write.");
            return;
        }

        EngineResolution.Report("KEX", "render size", width, height, $"output-class selection at 0x{site:X}");
    }

    /// <summary>The signature with both outcomes set to <paramref name="width"/> x <paramref name="height"/>.</summary>
    internal static byte[] BuildReplacement(uint width, uint height)
    {
        var bytes = (byte[])Signature.Clone();
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(UhdHeight), height);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(HdHeight), height);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(UhdWidth), width);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(HdWidth), width);
        return bytes;
    }

    /// <summary>The address of the selection when exactly one executable site carries it.</summary>
    internal static ulong? Find(IReadOnlyList<EngineResolution.Segment> segments)
    {
        ulong? found = null;
        foreach (var segment in segments)
        {
            if (!segment.Executable)
            {
                continue;
            }

            var span = segment.Bytes.AsSpan();
            var first = span.IndexOf(Signature);
            if (first < 0)
            {
                continue;
            }

            if (found is not null || span[(first + 1)..].IndexOf(Signature) >= 0)
            {
                return null;
            }

            found = segment.Start + (ulong)first;
        }

        return found;
    }
}
