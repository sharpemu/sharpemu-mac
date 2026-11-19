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

    // SHARPEMU_HLE_WRITE_RING_MIN=<bytes>: the smallest HLE write kept. The default keeps every
    // libc copy, which costs enough to move the timing of a race; 65536 keeps file reads and
    // bulk copies only.
    private static readonly ulong HleMinimumSize =
        ulong.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_HLE_WRITE_RING_MIN"), out var minimum) && minimum > MinimumSize
            ? minimum
            : MinimumSize;
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
        public ulong Backing;
        public ulong SourceBacking;
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

    private const ulong NoBacking = ulong.MaxValue;

    /// <summary>
    /// The backing offset a guest address resolves to in this process when a write is recorded
    /// (ulong.MaxValue for none), set by the owner of the views. A write lands where its address
    /// resolved then, which is no longer the address's bytes once the guest has moved the mapping,
    /// so a fault range is also matched by the backing bytes it maps now.
    /// </summary>
    public static Func<ulong, ulong>? BackingResolver { get; set; }

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

        Store(address, size, serial, source: 0, NoBacking, kind, producerA, producerB);
    }

    /// <summary>
    /// Notes an HLE write of [address, address + size) made by the guest process itself:
    /// <paramref name="source"/> is the guest source address of a copy or the file offset of a
    /// read, and <paramref name="producer"/> the file read (null for a copy).
    /// </summary>
    public static void RecordHle(ulong address, ulong size, ulong source, string kind, string? producer)
    {
        if (!HleEnabled || size < HleMinimumSize)
        {
            return;
        }

        // A copy's source is a guest address, and where it was mapped is known only now.
        var sourceBacking = ReferenceEquals(kind, CopyKind) && BackingResolver is { } resolver ? resolver(source) : NoBacking;
        Store(address, size, serial: 0, source, sourceBacking, kind, producer, producerB: null);
    }

    /// <summary>Where in a write a sample is taken, and how many bytes of it.</summary>
    public const int SampleOffset = 0x20000;

    public const int SampleLength = 16;

    /// <summary>
    /// Notes what the bytes at <see cref="SampleOffset"/> of an HLE write's destination hold at
    /// this moment (before or after the write): which of two racing writers came last is what
    /// those bytes say when both are recorded.
    /// </summary>
    public static void RecordHleSample(ulong address, ReadOnlySpan<byte> destination, string kind)
    {
        if (!HleEnabled || (ulong)destination.Length < HleMinimumSize || destination.Length < SampleOffset + SampleLength)
        {
            return;
        }

        Store(address, (ulong)destination.Length, serial: 0, source: 0, NoBacking, kind,
            "sample=" + Convert.ToHexString(destination.Slice(SampleOffset, SampleLength)), producerB: null);
    }

    /// <summary>The GPU process's sample of the same bytes, with what it knows of the wait.</summary>
    public static void RecordSample(ulong address, ulong size, ulong serial, string kind, ReadOnlySpan<byte> sample, string detail)
    {
        if (!Enabled)
        {
            return;
        }

        Store(address, size, serial, source: 0, NoBacking, kind, "sample=" + Convert.ToHexString(sample), detail);
    }

    /// <summary>The kind a libc copy is recorded under; its source is a guest address.</summary>
    public const string CopyKind = "hle:memcpy";

    /// <summary>
    /// The source bytes of each recorded copy that wrote into [start, end), newest first: the
    /// guest range the intersecting part was copied from and the backing it mapped then
    /// (ulong.MaxValue when unknown). Bytes a copy brought were written at its source.
    /// </summary>
    public static List<(ulong Start, ulong Size, ulong Backing)> CopySourcesOf(ulong start, ulong end, int limit = 4)
    {
        var sources = new List<(ulong Start, ulong Size, ulong Backing)>();
        var newest = Volatile.Read(ref _sequence);
        for (var sequence = newest; sequence > 0 && sequence > newest - Capacity && sources.Count < limit; sequence--)
        {
            ref var entry = ref Entries[(int)(sequence & (Capacity - 1))];
            if (Volatile.Read(ref entry.Sequence) != sequence || !ReferenceEquals(entry.Kind, CopyKind) ||
                entry.Address + entry.Size <= start || entry.Address >= end)
            {
                continue;
            }

            var from = Math.Max(start, entry.Address);
            var delta = from - entry.Address;
            sources.Add((
                entry.Source + delta,
                Math.Min(end, entry.Address + entry.Size) - from,
                entry.SourceBacking == NoBacking ? NoBacking : entry.SourceBacking + delta));
        }

        return sources;
    }

    private static void Store(
        ulong address, ulong size, ulong serial, ulong source, ulong sourceBacking, string kind, string? producerA, string? producerB)
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
        entry.Backing = BackingResolver is { } resolver ? resolver(address) : NoBacking;
        entry.SourceBacking = sourceBacking;
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
    public static List<string> Describe(ulong start, ulong end, int limit = MaxReportedPerRange) =>
        Describe(start, end, byBacking: false, limit);

    /// <summary>
    /// As <see cref="Describe(ulong, ulong, int)"/>; with <paramref name="byBacking"/> the range is
    /// backing offsets and a write matches by where its first byte resolved when it was recorded
    /// (taken as one run of the backing, which a write across two mappings is not).
    /// </summary>
    public static List<string> Describe(ulong start, ulong end, bool byBacking, int limit = MaxReportedPerRange)
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
            if (byBacking
                    ? entry.Backing == NoBacking || entry.Backing + entry.Size <= start || entry.Backing >= end
                    : entryEnd <= start || entry.Address >= end)
            {
                continue;
            }

            lines.Add(Format(sequence, in entry, now));
        }

        return lines;
    }

    // t is the raw monotonic timestamp (ns on macOS and Linux), comparable between the guest
    // process and the GPU host, whose ages are taken at different moments.
    private static string Format(long sequence, in Entry entry, long now)
    {
        var entryEnd = entry.Address + entry.Size;
        var ageMs = (now - entry.Ticks) * 1000.0 / Stopwatch.Frequency;
        var producer = entry.ProducerB is null ? entry.ProducerA : $"{entry.ProducerA} {entry.ProducerB}";
        return
                $"#{sequence} t={entry.Ticks} age={ageMs.ToString("F1", CultureInfo.InvariantCulture)}ms " +
                $"[0x{entry.Address:X16}, 0x{entryEnd:X16}) size=0x{entry.Size:X} " +
                $"backing={(entry.Backing == NoBacking ? "?" : $"0x{entry.Backing:X}")} serial={entry.Serial} " +
                $"src=0x{entry.Source:X}{(entry.SourceBacking == NoBacking ? string.Empty : $" src_backing=0x{entry.SourceBacking:X}")} " +
                $"tid={entry.Thread} kind={entry.Kind} producer={producer ?? "?"}";
    }

    /// <summary>The newest <paramref name="count"/> recorded writes, newest first.</summary>
    public static List<string> Tail(int count)
    {
        var lines = new List<string>();
        var newest = Volatile.Read(ref _sequence);
        var now = Stopwatch.GetTimestamp();
        for (var sequence = newest; sequence > 0 && sequence > newest - Capacity && lines.Count < count; sequence--)
        {
            ref var entry = ref Entries[(int)(sequence & (Capacity - 1))];
            if (Volatile.Read(ref entry.Sequence) == sequence)
            {
                lines.Add(Format(sequence, in entry, now));
            }
        }

        return lines;
    }

    /// <summary>
    /// Each recorded write whose producer names <paramref name="text"/> (a file, for a read),
    /// newest first: its range and the backing it mapped then (ulong.MaxValue when unknown).
    /// </summary>
    public static List<(ulong Start, ulong Size, ulong Backing)> RangesOfProducer(string text, int limit = 8)
    {
        var found = new List<(ulong Start, ulong Size, ulong Backing)>();
        var newest = Volatile.Read(ref _sequence);
        for (var sequence = newest; sequence > 0 && sequence > newest - Capacity && found.Count < limit; sequence--)
        {
            ref var entry = ref Entries[(int)(sequence & (Capacity - 1))];
            if (Volatile.Read(ref entry.Sequence) == sequence && entry.ProducerA is { } producer &&
                producer.Contains(text, StringComparison.OrdinalIgnoreCase) && !found.Contains((entry.Address, entry.Size, entry.Backing)))
            {
                found.Add((entry.Address, entry.Size, entry.Backing));
            }
        }

        return found;
    }

    /// <summary>The file a guest process with this pid hands its fault ranges over in.</summary>
    public static string FaultRangesPath(int processId) =>
        Path.Combine(Path.GetTempPath(), $"sharpemu-fault-ranges-{processId}.txt");

    /// <summary>
    /// Prints the entries intersecting each range here, and (with the GPU ring on) leaves the
    /// ranges for a GPU host process to answer: one "start size label" line per range.
    /// </summary>
    public static void ReportFaultRanges(
        IReadOnlyList<(ulong Start, ulong Size, string Label)> ranges,
        IReadOnlyList<(ulong Start, ulong Size, string Label)>? backingRanges = null)
    {
        if ((!Enabled && !HleEnabled) || ranges.Count == 0)
        {
            return;
        }

        Print("local", ranges, byBacking: false);
        if (backingRanges is not null)
        {
            Print("local by backing", backingRanges, byBacking: true);
        }

        // Bytes a recorded copy brought into a range were written at the copy's source: its
        // address and the backing it mapped then join the ranges, here and for the GPU host.
        var followed = new List<(ulong Start, ulong Size, string Label)>();
        var followedBacking = new List<(ulong Start, ulong Size, string Label)>();
        foreach (var (start, size, label) in ranges)
        {
            foreach (var (sourceStart, sourceSize, sourceBacking) in CopySourcesOf(start, start + size))
            {
                followed.Add((sourceStart, sourceSize, $"copy source of {label}"));
                if (sourceBacking != NoBacking)
                {
                    followedBacking.Add((sourceBacking, sourceSize, $"copy source of {label}"));
                }
            }
        }

        // SHARPEMU_FAULT_RING_PRODUCER=<text>: the writes of that producer (a file's reads) join the
        // ranges too, for bytes that reached a fault range by a copy nothing recorded.
        if (Environment.GetEnvironmentVariable("SHARPEMU_FAULT_RING_PRODUCER") is { Length: > 0 } producerText)
        {
            foreach (var (producerStart, producerSize, producerBacking) in RangesOfProducer(producerText))
            {
                followed.Add((producerStart, producerSize, $"write of {producerText}"));
                if (producerBacking != NoBacking)
                {
                    followedBacking.Add((producerBacking, producerSize, $"write of {producerText}"));
                }
            }
        }

        // SHARPEMU_WRITE_RING_TAIL=<n>: the newest n records of this process, whatever they wrote.
        if (int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_WRITE_RING_TAIL"), out var tail) && tail > 0)
        {
            foreach (var line in Tail(tail))
            {
                Console.Error.WriteLine($"[GPU-WRITE-RING][INFO] local tail: {line}");
            }
        }

        if (followed.Count != 0)
        {
            Print("local", followed, byBacking: false);
            Print("local by backing", followedBacking, byBacking: true);
            ranges = [.. ranges, .. followed];
            backingRanges = [.. backingRanges ?? [], .. followedBacking];
        }

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

            // '@' lines are backing offsets: the bytes the ranges above map at the fault.
            foreach (var (start, size, label) in backingRanges ?? [])
            {
                text.Append(CultureInfo.InvariantCulture, $"@{start:X} {size:X} {label}\n");
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
        var backingRanges = new List<(ulong Start, ulong Size, string Label)>();
        foreach (var line in File.ReadAllLines(path))
        {
            var byBacking = line.StartsWith('@');
            var parts = (byBacking ? line[1..] : line).Split(' ', 3);
            if (parts.Length >= 2 &&
                ulong.TryParse(parts[0], NumberStyles.HexNumber, CultureInfo.InvariantCulture, out var start) &&
                ulong.TryParse(parts[1], NumberStyles.HexNumber, CultureInfo.InvariantCulture, out var size))
            {
                (byBacking ? backingRanges : ranges).Add((start, size, parts.Length > 2 ? parts[2] : string.Empty));
            }
        }

        Print($"gpu-host for pid {processId}", ranges, byBacking: false);
        Print($"gpu-host for pid {processId} by backing", backingRanges, byBacking: true);
        try { File.Delete(path); } catch (IOException) { }
    }

    private static void Print(string side, IReadOnlyList<(ulong Start, ulong Size, string Label)> ranges, bool byBacking)
    {
        foreach (var (start, size, label) in ranges)
        {
            var lines = Describe(start, start + size, byBacking);
            Console.Error.WriteLine(
                $"[GPU-WRITE-RING][INFO] {side}: {label} {(byBacking ? "backing " : string.Empty)}[0x{start:X16}, 0x{start + size:X16}) " +
                $"writes={lines.Count}{(lines.Count == MaxReportedPerRange ? "+" : string.Empty)} recorded={Recorded}");
            foreach (var line in lines)
            {
                Console.Error.WriteLine($"[GPU-WRITE-RING][INFO]   {line}");
            }
        }

        Console.Error.Flush();
    }
}
