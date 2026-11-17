// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Threading;

namespace SharpEmu.HLE;

/// <summary>
/// The last 1 Mi writes (SHARPEMU_GPU_WRITE_RING_ENTRIES, a power of two) of 16 bytes or more that
/// the host made or encoded into guest memory on the GPU's behalf: image
/// texels written back (with the image's description), declared GPU stores, fills, copies, and
/// the host-side fills and copies. A guest fault over bytes that hold something the guest never
/// wrote asks this ring which of those writes covered them.
///
/// Off unless <c>SHARPEMU_GPU_WRITE_RING=1</c>. A record is an interlocked add and a struct store;
/// the strings are references the caller already holds. Writes under 16 bytes (labels, write-data
/// dwords) are not kept: they are most of the traffic (Demon's Souls boot: ~18k writes a second in
/// all) and cannot be the multi-texel run this ring is for, so leaving them out keeps about a minute.
///
/// The ring lives in whichever process encodes GPU work. In one process the fault dump queries it
/// directly. In the split, the guest process's fault dump writes the ranges it dumped to
/// <see cref="FaultRangesPath"/> (keyed by its pid) and the GPU host, once it sees that process
/// gone, prints the entries that intersect them (<see cref="ReportFaultRangesOf"/>) into the same
/// log.
///
/// <c>SHARPEMU_HLE_WRITE_RING=1</c> records the guest process's own bulk HLE writes into the same
/// ring (<see cref="RecordHle"/>): file reads into guest memory (AMPR, read, pread) with the file
/// and its offset, and the libc memcpy/memmove import with its source. Those are the host writes a
/// fault over foreign bytes has to tell apart from GPU work, and the guest process's fault dump
/// prints them directly.
/// </summary>
public static class GpuWriteRing
{
    private const ulong MinimumSize = 16;
    private static readonly int Capacity = ReadCapacity();
    private const int MaxReportedPerRange = 64;

    private struct Entry
    {
        public long Sequence;
        public long Ticks;
        public ulong Address;
        public ulong Size;
        public ulong Serial;
        public ulong Source;
        public int Thread;
        public string? Kind;
        public string? ProducerA;
        public string? ProducerB;
    }

    private static Entry[]? _entries;
    private static Entry[] Entries => _entries ??= AllocateEntries();

    private static Entry[] AllocateEntries()
    {
        var entries = new Entry[Capacity];
        Interlocked.CompareExchange(ref _entries, entries, null);
        return _entries!;
    }

    private static int ReadCapacity()
    {
        var text = Environment.GetEnvironmentVariable("SHARPEMU_GPU_WRITE_RING_ENTRIES");
        return int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out var entries) &&
               entries >= 1024 && entries <= 1 << 24 && (entries & (entries - 1)) == 0
            ? entries
            : 1 << 20;
    }
    private static long _sequence;

    /// <summary>True while writes are recorded.</summary>
    public static bool Enabled { get; set; } = string.Equals(
        Environment.GetEnvironmentVariable("SHARPEMU_GPU_WRITE_RING"), "1", StringComparison.Ordinal);

    /// <summary>True while the guest process's bulk HLE writes are recorded (<see cref="RecordHle"/>).</summary>
    public static bool HleEnabled { get; set; } = string.Equals(
        Environment.GetEnvironmentVariable("SHARPEMU_HLE_WRITE_RING"), "1", StringComparison.Ordinal);

    /// <summary>How many writes have been recorded since startup.</summary>
    public static long Recorded => Volatile.Read(ref _sequence);

    /// <summary>
    /// Notes a write of [address, address + size). <paramref name="kind"/> says what wrote it and
    /// whether it is a GPU command encoded for later or a host store made now; the two producer
    /// strings (either may be null) name the image or the draw, and are joined only when printed.
    /// </summary>
    public static void Record(ulong address, ulong size, ulong serial, string kind, string? producerA, string? producerB = null)
    {
        if (!Enabled || size < MinimumSize)
        {
            return;
        }

        Store(address, size, serial, source: 0, kind, producerA, producerB);
    }

    /// <summary>
    /// Notes an HLE write of [address, address + size) made by the guest process itself:
    /// <paramref name="source"/> is the guest source address of a copy or the file offset of a
    /// read, and <paramref name="producer"/> the file read (null for a copy).
    /// </summary>
    public static void RecordHle(ulong address, ulong size, ulong source, string kind, string? producer)
    {
        if (!HleEnabled || size < MinimumSize)
        {
            return;
        }

        Store(address, size, serial: 0, source, kind, producer, producerB: null);
    }

    private static void Store(ulong address, ulong size, ulong serial, ulong source, string kind, string? producerA, string? producerB)
    {
        var sequence = Interlocked.Increment(ref _sequence);
        ref var entry = ref Entries[(int)(sequence & (Capacity - 1))];
        // Sequence last: a reader that sees it sees the rest (or skips an entry being rewritten).
        Volatile.Write(ref entry.Sequence, 0);
        entry.Ticks = Stopwatch.GetTimestamp();
        entry.Address = address;
        entry.Size = size;
        entry.Serial = serial;
        entry.Source = source;
        entry.Thread = Environment.CurrentManagedThreadId;
        entry.Kind = kind;
        entry.ProducerA = producerA;
        entry.ProducerB = producerB;
        Volatile.Write(ref entry.Sequence, sequence);
    }

    /// <summary>
    /// The recorded writes intersecting [start, end), newest first, at most
    /// <paramref name="limit"/>, each formatted on one line with its age at the time of the call.
    /// </summary>
    public static List<string> Describe(ulong start, ulong end, int limit = MaxReportedPerRange)
    {
        var lines = new List<string>();
        var newest = Volatile.Read(ref _sequence);
        var now = Stopwatch.GetTimestamp();
        for (var sequence = newest; sequence > 0 && sequence > newest - Capacity && lines.Count < limit; sequence--)
        {
            ref var entry = ref Entries[(int)(sequence & (Capacity - 1))];
            if (Volatile.Read(ref entry.Sequence) != sequence)
            {
                continue;
            }

            var entryEnd = entry.Address + entry.Size;
            if (entryEnd <= start || entry.Address >= end)
            {
                continue;
            }

            var ageMs = (now - entry.Ticks) * 1000.0 / Stopwatch.Frequency;
            var producer = entry.ProducerB is null ? entry.ProducerA : $"{entry.ProducerA} {entry.ProducerB}";
            lines.Add(
                $"#{sequence} age={ageMs.ToString("F1", CultureInfo.InvariantCulture)}ms " +
                $"[0x{entry.Address:X16}, 0x{entryEnd:X16}) size=0x{entry.Size:X} serial={entry.Serial} " +
                $"src=0x{entry.Source:X} tid={entry.Thread} kind={entry.Kind} producer={producer ?? "?"}");
        }

        return lines;
    }

    /// <summary>The file a guest process with this pid hands its fault ranges over in.</summary>
    public static string FaultRangesPath(int processId) =>
        Path.Combine(Path.GetTempPath(), $"sharpemu-fault-ranges-{processId}.txt");

    /// <summary>
    /// Prints the entries intersecting each range here, and (with the GPU ring on) leaves the
    /// ranges for a GPU host process to answer: one "start size label" line per range.
    /// </summary>
    public static void ReportFaultRanges(IReadOnlyList<(ulong Start, ulong Size, string Label)> ranges)
    {
        if ((!Enabled && !HleEnabled) || ranges.Count == 0)
        {
            return;
        }

        Print("local", ranges);
        if (!Enabled)
        {
            return;
        }

        try
        {
            var text = new StringBuilder();
            foreach (var (start, size, label) in ranges)
            {
                text.Append(CultureInfo.InvariantCulture, $"{start:X} {size:X} {label}\n");
            }

            File.WriteAllText(FaultRangesPath(Environment.ProcessId), text.ToString());
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"[GPU-WRITE-RING][WARN] could not hand the fault ranges to the GPU host: {exception.Message}");
        }
    }

    /// <summary>
    /// In the GPU host, after its guest process <paramref name="processId"/> exited: prints the
    /// entries intersecting the ranges that process's fault dump left, then removes the file.
    /// </summary>
    public static void ReportFaultRangesOf(int processId)
    {
        if (!Enabled)
        {
            return;
        }

        var path = FaultRangesPath(processId);
        if (!File.Exists(path))
        {
            Console.Error.WriteLine($"[GPU-WRITE-RING][INFO] guest process {processId} left no fault ranges; recorded={Recorded}.");
            return;
        }

        var ranges = new List<(ulong Start, ulong Size, string Label)>();
        foreach (var line in File.ReadAllLines(path))
        {
            var parts = line.Split(' ', 3);
            if (parts.Length >= 2 &&
                ulong.TryParse(parts[0], NumberStyles.HexNumber, CultureInfo.InvariantCulture, out var start) &&
                ulong.TryParse(parts[1], NumberStyles.HexNumber, CultureInfo.InvariantCulture, out var size))
            {
                ranges.Add((start, size, parts.Length > 2 ? parts[2] : string.Empty));
            }
        }

        Print($"gpu-host for pid {processId}", ranges);
        try { File.Delete(path); } catch (IOException) { }
    }

    private static void Print(string side, IReadOnlyList<(ulong Start, ulong Size, string Label)> ranges)
    {
        foreach (var (start, size, label) in ranges)
        {
            var lines = Describe(start, start + size);
            Console.Error.WriteLine(
                $"[GPU-WRITE-RING][INFO] {side}: {label} [0x{start:X16}, 0x{start + size:X16}) " +
                $"writes={lines.Count}{(lines.Count == MaxReportedPerRange ? "+" : string.Empty)} recorded={Recorded}");
            foreach (var line in lines)
            {
                Console.Error.WriteLine($"[GPU-WRITE-RING][INFO]   {line}");
            }
        }

        Console.Error.Flush();
    }
}
