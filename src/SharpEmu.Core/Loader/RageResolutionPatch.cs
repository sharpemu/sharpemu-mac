// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.HLE;

namespace SharpEmu.Core.Loader;

/// <summary>
/// Rockstar's RAGE (GTA V) sizes its back buffer from the video settings' ScreenWidth and
/// ScreenHeight: a preset loaded for each settings slot, then overridden by the <c>-width</c> and
/// <c>-height</c> parameters. On PS5 those parameters cannot be passed: the shipped build keeps
/// the parameter objects but nothing ever walks them against argv, so their value pointers stay
/// null whatever the process is started with. The override is the one place that sets the size,
/// once per slot: <c>mov rdi, [param]; test rdi, rdi; je; cmp byte [rdi], 0; je; xor esi, esi;
/// xor edx, edx; call strtol; mov [rbx+0x1A8], eax</c> (0x1AC for the height). This takes the
/// branches out and replaces the strtol call with <c>mov eax, size</c>, as if
/// <c>-width=W -height=H</c> had been given.
/// </summary>
internal static class RageResolutionPatch
{
    // From the width test to the height store; null bytes are wildcards (call targets, RIP displacements).
    private static readonly byte?[] Signature =
    [
        0x48, 0x85, 0xFF, 0x74, 0x1B, 0x80, 0x3F, 0x00, 0x74, 0x16, 0x31, 0xF6, 0x31, 0xD2,
        0xE8, null, null, null, null,
        0x89, 0x83, 0xA8, 0x01, 0x00, 0x00,
        0xC6, 0x05, null, null, null, null, 0x01,
        0x48, 0x8B, 0x3D, null, null, null, null,
        0x48, 0x85, 0xFF, 0x74, 0x1B, 0x80, 0x3F, 0x00, 0x74, 0x16, 0x31, 0xF6, 0x31, 0xD2,
        0xE8, null, null, null, null,
        0x89, 0x83, 0xAC, 0x01, 0x00, 0x00,
    ];

    private const int WidthBlock = 0;
    private const int HeightBlock = 39;

    internal static void Apply(ICpuMemory memory, IReadOnlyList<EngineResolution.Segment> segments, uint width, uint height)
    {
        if (Find(segments) is not { } site)
        {
            return;
        }

        var original = new byte[Signature.Length];
        if (!memory.TryRead(site, original))
        {
            return;
        }

        if (!memory.TryWrite(site, BuildReplacement(original, width, height)))
        {
            EngineResolution.Warn("RAGE", "the code segment refused the write.");
            return;
        }

        EngineResolution.Report("RAGE", "video settings -width/-height", width, height, $"parameter override at 0x{site:X}");
    }

    /// <summary><paramref name="original"/> with both parameter tests taken out and the strtol
    /// calls replaced by the size.</summary>
    internal static byte[] BuildReplacement(ReadOnlySpan<byte> original, uint width, uint height)
    {
        var bytes = original.ToArray();
        Rewrite(bytes.AsSpan(WidthBlock), width);
        Rewrite(bytes.AsSpan(HeightBlock), height);
        return bytes;
    }

    // test rdi, rdi (kept); je -> 2-byte nop; cmp byte [rdi], 0 -> 3-byte nop; je -> 2-byte nop;
    // xor esi, esi; xor edx, edx (kept); call strtol -> mov eax, value.
    private static void Rewrite(Span<byte> block, uint value)
    {
        block[3] = 0x66;
        block[4] = 0x90;
        block[5] = 0x0F;
        block[6] = 0x1F;
        block[7] = 0x00;
        block[8] = 0x66;
        block[9] = 0x90;
        block[14] = 0xB8;
        BinaryPrimitives.WriteUInt32LittleEndian(block[15..], value);
    }

    /// <summary>The address of the override when exactly one executable site matches.</summary>
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
            var anchor = Signature.TakeWhile(value => value is not null).Select(value => value!.Value).ToArray();
            for (var from = 0; from < span.Length;)
            {
                var hit = span[from..].IndexOf(anchor);
                if (hit < 0)
                {
                    break;
                }

                var offset = from + hit;
                if (Matches(span[offset..]))
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

    private static bool Matches(ReadOnlySpan<byte> code)
    {
        if (code.Length < Signature.Length)
        {
            return false;
        }

        for (var index = 0; index < Signature.Length; index++)
        {
            if (Signature[index] is { } expected && code[index] != expected)
            {
                return false;
            }
        }

        return true;
    }
}
