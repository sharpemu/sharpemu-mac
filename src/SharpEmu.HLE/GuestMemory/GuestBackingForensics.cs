// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
using System.Runtime.InteropServices;

namespace SharpEmu.HLE.GuestMemory;

/// <summary>
/// Fault-time forensics over the shared guest backing, for the fault dump (SHARPEMU_FAULT_DUMP):
/// which backing bytes a guest range maps and which other guest ranges map the same bytes
/// (<see cref="DescribeAliases(ulong, ulong, int)"/>), and where else in mapped guest memory a run
/// of bytes appears (<see cref="FindBytes(ReadOnlySpan{byte}, int, bool)"/>). Read only: it takes
/// a copy of the view records and reads the canonical alias, and changes nothing.
/// </summary>
public static unsafe class GuestBackingForensics
{
    private static SharedBackingViews? _views;

    // The newest backing views constructed in this process: the guest's, in a running game.
    internal static void Register(SharedBackingViews views)
    {
        Volatile.Write(ref _views, views);
        if (GpuWriteRing.HleEnabled)
        {
            // SHARPEMU_HLE_WRITE_RING=1: each recorded write keeps the backing its address maps now.
            GpuWriteRing.BackingResolver = address =>
                views.AliasBase is var aliasBase and not 0 && views.TryResolveAlias(address, 1, out var alias)
                    ? alias - aliasBase
                    : ulong.MaxValue;
        }
    }

    /// <summary>
    /// For each view the guest range [address, address + size) lies in: its backing bytes, and
    /// every other view that maps any of those same backing bytes.
    /// </summary>
    public static List<string> DescribeAliases(ulong address, ulong size, int limit = 32)
    {
        if (Volatile.Read(ref _views) is not { } views || !views.TrySnapshotViews(out var records))
        {
            return ["no backing views to inspect"];
        }

        return DescribeAliases(records, address, size, limit);
    }

    /// <summary>The backing bytes the guest range [address, address + size) maps now, one run per view.</summary>
    public static List<(ulong Start, ulong Size)> BackingSpansOf(ulong address, ulong size)
    {
        var spans = new List<(ulong Start, ulong Size)>();
        if (Volatile.Read(ref _views) is not { } views || !views.TrySnapshotViews(out var records))
        {
            return spans;
        }

        var end = address + size;
        foreach (var view in records)
        {
            if (view.Address + view.Size <= address || view.Address >= end)
            {
                continue;
            }

            var from = Math.Max(address, view.Address);
            spans.Add((view.Offset + (from - view.Address), Math.Min(end, view.Address + view.Size) - from));
        }

        return spans;
    }

    internal static List<string> DescribeAliases(IReadOnlyList<ViewRecord> records, ulong address, ulong size, int limit)
    {
        var lines = new List<string>();
        var end = address + size;
        foreach (var view in records)
        {
            if (view.Address + view.Size <= address || view.Address >= end)
            {
                continue;
            }

            var from = Math.Max(address, view.Address);
            var to = Math.Min(end, view.Address + view.Size);
            var backingStart = view.Offset + (from - view.Address);
            var backingEnd = backingStart + (to - from);
            lines.Add(
                $"guest [0x{from:X16}, 0x{to:X16}) = backing [0x{backingStart:X}, 0x{backingEnd:X}) " +
                $"in view guest=0x{view.Address:X16} size=0x{view.Size:X} offset=0x{view.Offset:X} prot={view.Protection}");
            var others = 0;
            foreach (var other in records)
            {
                if (other.Address == view.Address ||
                    other.Offset + other.Size <= backingStart || other.Offset >= backingEnd)
                {
                    continue;
                }

                var shared = Math.Max(backingStart, other.Offset);
                var sharedEnd = Math.Min(backingEnd, other.Offset + other.Size);
                lines.Add(
                    $"  ALIAS backing [0x{shared:X}, 0x{sharedEnd:X}) is also guest " +
                    $"[0x{other.Address + (shared - other.Offset):X16}, 0x{other.Address + (sharedEnd - other.Offset):X16}) " +
                    $"in view guest=0x{other.Address:X16} size=0x{other.Size:X} offset=0x{other.Offset:X} prot={other.Protection}");
                if (++others >= limit)
                {
                    lines.Add("  ALIAS list cut short");
                    break;
                }
            }

            if (others == 0)
            {
                lines.Add("  no other view maps these backing bytes");
            }
        }

        if (lines.Count == 0)
        {
            lines.Add($"guest [0x{address:X16}, 0x{end:X16}) is in no backing view");
        }

        return lines;
    }

    /// <summary>
    /// Every place in the backing bytes some view maps that holds <paramref name="needle"/>, as
    /// "backing offset → each guest address mapping it", newest view order. Pages the host reports
    /// as not resident are skipped (they cannot hold bytes anyone wrote) unless
    /// <paramref name="scanAll"/>; the last line says how much was scanned and skipped.
    /// </summary>
    public static List<string> FindBytes(ReadOnlySpan<byte> needle, int limit = 32, bool scanAll = false)
    {
        if (Volatile.Read(ref _views) is not { } views || !views.TrySnapshotViews(out var records) ||
            views.AliasBase == 0)
        {
            return ["no backing views to search"];
        }

        return FindBytes(records, (byte*)views.AliasBase, views.Size, needle, limit,
            scanAll || OperatingSystem.IsWindows() ? null : IsResidentPosix);
    }

    // residency(start, length, pageSize, vector) fills one byte per page, nonzero when resident;
    // false when it cannot tell, and the range is then scanned anyway.
    internal delegate bool ResidencyQuery(byte* start, ulong length, ulong pageSize, byte[] vector);

    internal static List<string> FindBytes(
        IReadOnlyList<ViewRecord> records, byte* alias, ulong backingSize, ReadOnlySpan<byte> needle, int limit,
        ResidencyQuery? residency)
    {
        var lines = new List<string>();
        if (needle.IsEmpty)
        {
            return ["empty needle"];
        }

        // The backing bytes some view maps, merged.
        var spans = new List<(ulong Start, ulong End)>();
        foreach (var view in records)
        {
            var start = Math.Min(view.Offset, backingSize);
            var end = Math.Min(view.Offset + view.Size, backingSize);
            if (end > start)
            {
                spans.Add((start, end));
            }
        }

        spans.Sort((a, b) => a.Start.CompareTo(b.Start));
        var merged = new List<(ulong Start, ulong End)>();
        foreach (var span in spans)
        {
            if (merged.Count > 0 && span.Start <= merged[^1].End)
            {
                merged[^1] = (merged[^1].Start, Math.Max(merged[^1].End, span.End));
            }
            else
            {
                merged.Add(span);
            }
        }

        var pageSize = (ulong)Math.Max(Environment.SystemPageSize, 4096);
        const ulong Batch = 256UL * 1024 * 1024;
        var vector = residency is null ? [] : new byte[Batch / pageSize + 2];
        ulong scanned = 0, skipped = 0, hits = 0;
        var overlap = (ulong)needle.Length - 1;
        foreach (var (spanStart, spanEnd) in merged)
        {
            for (var batchStart = spanStart; batchStart < spanEnd; batchStart += Batch)
            {
                var batchEnd = Math.Min(spanEnd, batchStart + Batch);
                var runs = new List<(ulong Start, ulong End)>();
                var pageStart = batchStart / pageSize * pageSize;
                if (residency is not null &&
                    residency(alias + pageStart, batchEnd - pageStart, pageSize, vector))
                {
                    for (var page = pageStart; page < batchEnd; page += pageSize)
                    {
                        var index = (int)((page - pageStart) / pageSize);
                        var from = Math.Max(page, batchStart);
                        var to = Math.Min(page + pageSize, batchEnd);
                        if (vector[index] == 0)
                        {
                            skipped += to - from;
                        }
                        else if (runs.Count > 0 && runs[^1].End == from)
                        {
                            runs[^1] = (runs[^1].Start, to);
                        }
                        else
                        {
                            runs.Add((from, to));
                        }
                    }
                }
                else
                {
                    runs.Add((batchStart, batchEnd));
                }

                foreach (var (runStart, runEnd) in runs)
                {
                    // A match may start in this run and end in the resident bytes after it.
                    var scanEnd = Math.Min(spanEnd, runEnd + overlap);
                    var cursor = runStart;
                    while (cursor + (ulong)needle.Length <= scanEnd && lines.Count < limit)
                    {
                        var window = new ReadOnlySpan<byte>(alias + cursor, (int)Math.Min(scanEnd - cursor, int.MaxValue));
                        var found = window.IndexOf(needle);
                        if (found < 0 || cursor + (ulong)found >= runEnd)
                        {
                            break;
                        }

                        var offset = cursor + (ulong)found;
                        hits++;
                        lines.Add(DescribeHit(records, offset));
                        cursor = offset + 1;
                    }

                    scanned += runEnd - runStart;
                }
            }
        }

        lines.Add(
            string.Create(CultureInfo.InvariantCulture,
                $"searched {needle.Length} bytes: hits={hits}{(lines.Count >= limit ? "+" : string.Empty)} " +
                $"scanned_mb={scanned / 1048576.0:F1} skipped_not_resident_mb={skipped / 1048576.0:F1} views={records.Count}"));
        return lines;
    }

    private static string DescribeHit(IReadOnlyList<ViewRecord> records, ulong offset)
    {
        var text = new System.Text.StringBuilder($"backing 0x{offset:X} =");
        foreach (var view in records)
        {
            if (offset >= view.Offset && offset < view.Offset + view.Size)
            {
                text.Append(CultureInfo.InvariantCulture,
                    $" guest 0x{view.Address + (offset - view.Offset):X16} (view guest=0x{view.Address:X16} size=0x{view.Size:X} offset=0x{view.Offset:X})");
            }
        }

        return text.ToString();
    }

    [DllImport("libc", EntryPoint = "mincore", SetLastError = true)]
    private static extern int MinCore(void* address, nuint length, byte* vector);

    private static bool IsResidentPosix(byte* start, ulong length, ulong pageSize, byte[] vector)
    {
        Array.Clear(vector);
        try
        {
            fixed (byte* output = vector)
            {
                return MinCore(start, (nuint)length, output) == 0;
            }
        }
        catch (Exception exception) when (exception is DllNotFoundException or EntryPointNotFoundException)
        {
            return false;
        }
    }
}
