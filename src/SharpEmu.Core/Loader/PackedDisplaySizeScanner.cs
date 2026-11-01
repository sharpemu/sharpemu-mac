// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using Iced.Intel;
using SharpEmu.HLE;

namespace SharpEmu.Core.Loader;

/// <summary>
/// Small custom engines (NIS's NMPL in Void Terrarium, the Construct runtime in Dreaming Sarah)
/// never ask the display: the 3840x2160 they register is a compiled constant, a
/// <c>movabs r64, 0x0000087000000F00</c> packing width and height into one immediate. In a small
/// executable that constant is precise: it is the display size and nothing else. This finds those
/// sites and, in the functions that hold them, the related 32-bit immediates (<c>mov r32</c>,
/// <c>mov m32</c>, <c>cmp</c> with 3840 or 2160, such as the literals passed to
/// sceVideoOutSetBufferAttribute2), plus the read-only <c>{3840.0f, 2160.0f}</c> render-target
/// size vector, and rewrites them together to the advertised display. When those functions use
/// the size in a way a rewrite cannot keep consistent (a scalar 3840.0/2160.0 literal, an offset
/// or tolerance around it), nothing is written: half a rewrite would misplace the image.
/// </summary>
internal static class PackedDisplaySizeScanner
{
    internal const uint SourceWidth = 3840;
    internal const uint SourceHeight = 2160;
    internal const ulong Packed = SourceWidth | ((ulong)SourceHeight << 32);

    /// <summary>Above this much code a packed 3840x2160 is no longer specific to the display.</summary>
    internal const int MaxCodeBytes = 8 << 20;

    private const int MaxPackedSites = 8;
    private const int FunctionSearchLimit = 0x20000;
    private static readonly byte[] Prologue = [0x55, 0x48, 0x89, 0xE5];

    internal enum SiteKind
    {
        PackedImmediate,
        WidthImmediate,
        HeightImmediate,
        FloatPair,
    }

    internal readonly record struct Site(ulong Address, SiteKind Kind);

    internal sealed record Plan(IReadOnlyList<Site> Sites, string? Refusal);

    internal static void Apply(ICpuMemory memory, IReadOnlyList<EngineResolution.Segment> segments, uint width, uint height)
    {
        if (Scan(segments) is not { } plan)
        {
            return;
        }

        if (plan.Refusal is not null)
        {
            EngineResolution.Warn(
                "small engine",
                $"a compiled 3840x2160 display was found but left as shipped: {plan.Refusal}. The presenter scales it.");
            return;
        }

        if (width == SourceWidth && height == SourceHeight)
        {
            return;
        }

        foreach (var site in plan.Sites)
        {
            if (!memory.TryWrite(site.Address, Encode(site.Kind, width, height)))
            {
                EngineResolution.Warn("small engine", $"the segment refused the write at 0x{site.Address:X}.");
                return;
            }
        }

        SharpEmu.Libs.VideoOut.VideoOutExports.ExpectDisplayBuffer(width, height, "the packed display-size patch");
        EngineResolution.Report(
            "small engine",
            "packed display size",
            width,
            height,
            $"{plan.Sites.Count(site => site.Kind == SiteKind.PackedImmediate)} packed, " +
            $"{plan.Sites.Count(site => site.Kind is SiteKind.WidthImmediate or SiteKind.HeightImmediate)} immediates, " +
            $"{plan.Sites.Count(site => site.Kind == SiteKind.FloatPair)} float vectors");
    }

    internal static byte[] Encode(SiteKind kind, uint width, uint height)
    {
        var bytes = new byte[kind is SiteKind.PackedImmediate or SiteKind.FloatPair ? 8 : 4];
        switch (kind)
        {
            case SiteKind.PackedImmediate:
                BinaryPrimitives.WriteUInt64LittleEndian(bytes, width | ((ulong)height << 32));
                break;
            case SiteKind.WidthImmediate:
                BinaryPrimitives.WriteUInt32LittleEndian(bytes, width);
                break;
            case SiteKind.HeightImmediate:
                BinaryPrimitives.WriteUInt32LittleEndian(bytes, height);
                break;
            case SiteKind.FloatPair:
                BinaryPrimitives.WriteSingleLittleEndian(bytes, width);
                BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(4), height);
                break;
        }

        return bytes;
    }

    /// <summary>The sites to rewrite, or a refusal; null when the image carries no packed
    /// display size or is too large for the constant to be specific.</summary>
    internal static Plan? Scan(IReadOnlyList<EngineResolution.Segment> segments)
    {
        var code = segments.Where(segment => segment.Executable).ToList();
        if (code.Count == 0 || code.Sum(segment => (long)segment.Bytes.Length) > MaxCodeBytes)
        {
            return null;
        }

        var packedSites = new List<(EngineResolution.Segment Segment, int Offset)>();
        Span<byte> needle = stackalloc byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(needle, Packed);
        foreach (var segment in code)
        {
            var span = segment.Bytes.AsSpan();
            for (var from = 0; from < span.Length;)
            {
                var hit = span[from..].IndexOf(needle);
                if (hit < 0)
                {
                    break;
                }

                var immediate = from + hit;
                // movabs r64, imm64: REX.W (48 or 49) then B8+r.
                if (immediate >= 2 && span[immediate - 2] is 0x48 or 0x49 && span[immediate - 1] is >= 0xB8 and <= 0xBF)
                {
                    packedSites.Add((segment, immediate - 2));
                }

                from = immediate + 1;
            }
        }

        if (packedSites.Count == 0)
        {
            return null;
        }

        if (packedSites.Count > MaxPackedSites)
        {
            return new Plan([], $"{packedSites.Count} packed sites is too many to be the display alone");
        }

        var sites = new List<Site>();
        var functions = new HashSet<(EngineResolution.Segment Segment, int Start, int End)>();
        foreach (var (segment, offset) in packedSites)
        {
            var span = segment.Bytes.AsSpan();
            var searchFrom = Math.Max(0, offset - FunctionSearchLimit);
            var start = span[searchFrom..(offset + 1)].LastIndexOf(Prologue);
            start = start < 0 ? searchFrom : searchFrom + start;
            var searchTo = Math.Min(span.Length, offset + FunctionSearchLimit);
            var end = span[(offset + 1)..searchTo].IndexOf(Prologue);
            end = end < 0 ? searchTo : offset + 1 + end;
            functions.Add((segment, start, end));
        }

        foreach (var (segment, start, end) in functions)
        {
            if (ScanFunction(segment, start, end, segments, sites) is { } refusal)
            {
                return new Plan([], refusal);
            }
        }

        sites.AddRange(FindFloatPairs(segments));
        return new Plan([.. sites.Distinct().OrderBy(site => site.Address)], null);
    }

    private static string? ScanFunction(
        EngineResolution.Segment segment,
        int start,
        int end,
        IReadOnlyList<EngineResolution.Segment> segments,
        List<Site> sites)
    {
        var bytes = segment.Bytes.AsSpan(start, end - start).ToArray();
        var decoder = Iced.Intel.Decoder.Create(64, new ByteArrayCodeReader(bytes), segment.Start + (ulong)start);
        var endAddress = segment.Start + (ulong)end;
        while (decoder.IP < endAddress)
        {
            decoder.Decode(out var instruction);
            if (instruction.IsInvalid)
            {
                continue;
            }

            if (instruction.Code == Code.Mov_r64_imm64 && instruction.Immediate64 == Packed)
            {
                sites.Add(new Site(instruction.IP + 2, SiteKind.PackedImmediate));
                continue;
            }

            for (var operand = 0; operand < instruction.OpCount; operand++)
            {
                var kind = instruction.GetOpKind(operand);
                if (kind is OpKind.Immediate32 or OpKind.Immediate8to32 or OpKind.Immediate32to64 or OpKind.Immediate8to64 or OpKind.Immediate16 or OpKind.Immediate64)
                {
                    var value = (long)instruction.GetImmediate(operand);
                    if (value is SourceWidth or SourceHeight)
                    {
                        // A mask (and/or/test/xor) with the same bits is not the size.
                        if (instruction.Mnemonic is Mnemonic.And or Mnemonic.Or or Mnemonic.Test or Mnemonic.Xor)
                        {
                            continue;
                        }

                        if (kind != OpKind.Immediate32 || instruction.Mnemonic is not (Mnemonic.Mov or Mnemonic.Cmp))
                        {
                            return $"{instruction.Mnemonic.ToString().ToLowerInvariant()} with {value} at 0x{instruction.IP:X}";
                        }

                        // The imm32 is the instruction's last four bytes for mov r32/m32 and cmp r/m32.
                        sites.Add(new Site(
                            instruction.IP + (ulong)instruction.Length - 4,
                            value == SourceWidth ? SiteKind.WidthImmediate : SiteKind.HeightImmediate));
                    }
                    else if (Math.Abs(value - SourceWidth) <= 2 || Math.Abs(value - SourceHeight) <= 2)
                    {
                        return $"a tolerance around the size ({value}) at 0x{instruction.IP:X}";
                    }
                }
                else if (kind == OpKind.Memory)
                {
                    if (instruction.IsIPRelativeMemoryOperand)
                    {
                        if (ReadsScalarSize(segments, instruction.IPRelativeMemoryAddress))
                        {
                            return $"a 3840.0/2160.0 literal read at 0x{instruction.IP:X}";
                        }
                    }
                    else if (instruction.Mnemonic == Mnemonic.Lea &&
                        (long)instruction.MemoryDisplacement64 is -(long)SourceWidth or -(long)SourceHeight)
                    {
                        return $"an offset from the size at 0x{instruction.IP:X}";
                    }
                }
            }
        }

        return null;
    }

    /// <summary>Whether a scalar 3840.0 or 2160.0 (single or double) that is not the first half of
    /// a <c>{3840.0f, 2160.0f}</c> vector sits at <paramref name="address"/>.</summary>
    private static bool ReadsScalarSize(IReadOnlyList<EngineResolution.Segment> segments, ulong address)
    {
        Span<byte> value = stackalloc byte[8];
        if (!TryRead(segments, address, value))
        {
            return false;
        }

        var single = BinaryPrimitives.ReadSingleLittleEndian(value);
        var second = BinaryPrimitives.ReadSingleLittleEndian(value[4..]);
        if (single == SourceWidth && second == SourceHeight)
        {
            return false;
        }

        var dbl = BinaryPrimitives.ReadDoubleLittleEndian(value);
        return single is SourceWidth or SourceHeight || dbl is SourceWidth or SourceHeight;
    }

    private static bool TryRead(IReadOnlyList<EngineResolution.Segment> segments, ulong address, Span<byte> destination)
    {
        foreach (var segment in segments)
        {
            if (address >= segment.Start && address + (ulong)destination.Length <= segment.Start + (ulong)segment.Bytes.Length)
            {
                segment.Bytes.AsSpan((int)(address - segment.Start), destination.Length).CopyTo(destination);
                return true;
            }
        }

        return false;
    }

    /// <summary>Every 8-byte-aligned <c>{3840.0f, 2160.0f}</c> in the data segments.</summary>
    private static List<Site> FindFloatPairs(IReadOnlyList<EngineResolution.Segment> segments)
    {
        var needle = Encode(SiteKind.FloatPair, SourceWidth, SourceHeight);
        var sites = new List<Site>();
        foreach (var segment in segments)
        {
            if (segment.Executable)
            {
                continue;
            }

            var span = segment.Bytes.AsSpan();
            for (var from = 0; from < span.Length;)
            {
                var hit = span[from..].IndexOf(needle);
                if (hit < 0)
                {
                    break;
                }

                var offset = from + hit;
                if (((segment.Start + (ulong)offset) & 7) == 0)
                {
                    sites.Add(new Site(segment.Start + (ulong)offset, SiteKind.FloatPair));
                }

                from = offset + 1;
            }
        }

        return sites;
    }
}
