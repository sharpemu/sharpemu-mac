// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using SharpEmu.HLE.Host;

namespace SharpEmu.HLE.GuestMemory;

/// <summary>A guest virtual range of the <see cref="GuestViewTable"/>: the backing bytes behind it and its host protection.</summary>
public readonly record struct GuestViewEntry(ulong Address, ulong Size, ulong Offset, HostPageProtection Protection)
{
    public ulong End => Address + Size;
}

/// <summary>
/// The guest view table: where in the backing each guest address a backing view maps lives, kept in
/// shared memory so another process (the two-process split's GPU process, which maps the backing as
/// an offset mirror at an address of its own) turns a guest address into a mirror pointer without a
/// call (<see cref="GuestMirrorTranslator"/>), and a log of every change to it, so that process
/// can replicate every view at the guest address plus a constant of its own (<see cref="GuestReplica"/>).
/// Only a shareable backing has one (<see cref="HostBackingObject.Export"/>, <c>SHARPEMU_GPU_PROCESS=1</c>).
///
/// <para>
/// <see cref="SharedBackingViews"/> writes it at every view record change and
/// <see cref="GuestSpaceOwner"/> at every protection change of a backed range, each after the host
/// mapping changed. Entries are ordered by address and never overlap. Neighbours whose addresses and
/// offsets both continue, at one protection, are one entry.
/// </para>
///
/// <para>
/// Protection changes always go to the log. The entries follow them only when the table was made
/// with <c>trackProtection</c> (<see cref="ProtectionTrackedFlag"/>): a change over part of a view
/// then splits its entry, and the change back joins it again. Splitting moves the entries above it,
/// which measured 0.6 µs a change at 1,000 entries, 3.5 µs at 10,000 and 20 µs at 50,000 (arm64,
/// M4), on the owner's write-fault path, where page watches change protections thousands of times
/// a second. A shareable backing's table leaves it off: the replica keeps every view read-write
/// and needs only addresses and offsets. Without it each entry keeps its view's mapping protection.
/// </para>
///
/// <para>
/// Layout (little-endian; both processes are 64-bit): a 64-byte header (<see cref="GuestViewTableHeader"/>),
/// then <c>Capacity</c> entries of 32 bytes: address, size, offset, protection
/// (<see cref="HostPageProtection"/>). Readers copy what they need between two reads of
/// <c>Sequence</c>, which is odd while a change is being made, and retry when it moved.
/// <c>MappingGeneration</c> moves only when an address stops meaning the offset it meant (map, unmap,
/// remap), never for a protection change: page watches change protection thousands of times a
/// second, and a reader's cached translations outlive them. A change that does not fit sets
/// <see cref="IncompleteFlag"/> for good: the table may then miss ranges (never keep one that was
/// unmapped) or hold stale protections, and a reader must ask the owner instead.
/// </para>
///
/// <para>
/// The view log follows the entries: a ring of <c>LogCapacity</c> records of 40 bytes (sequence,
/// <see cref="GuestViewLogKind"/>, protection, address, size, offset), one per view mapped,
/// unmapped or protected, in the order the owner made them. Record <c>n</c> (counting from 0) sits in
/// slot <c>n % LogCapacity</c> and carries <c>n + 1</c> once written; <c>LogSequence</c> in the
/// header counts the records published. It is written in the same change as the entries, so a
/// consistent read of the table also says which log record it is current to: a reader that fell a
/// whole ring behind starts again from the entries and replays from there. Nothing waits for readers.
/// </para>
/// </summary>
public sealed unsafe class GuestViewTable : IDisposable
{
    public const uint Magic = 0x5456_4553; // "SEVT"
    public const uint LayoutVersion = 1;
    public const int HeaderBytes = 64;
    public const int EntryBytes = 32;
    public const uint IncompleteFlag = 1;
    public const uint ProtectionTrackedFlag = 2;

    public const int LogRecordBytes = 40;

    // 8 MiB of entries, of which only the pages in use are ever touched: thousands of views plus the
    // splits page watches make need a small part of it.
    public const int DefaultCapacity = 1 << 18;

    // 2.5 MiB of log. Views change about once a second in Grand Theft Auto V and almost never in
    // Black Myth: Wukong; protection changes are logged too and come thousands of times a second
    // under page watches, so a reader that syncs less than every few seconds starts again from the
    // entries instead (GuestReplica does that by itself).
    public const int DefaultLogCapacity = 1 << 16;

    private readonly HostSharedRegion _region;
    private readonly GuestViewTableHeader* _header;
    private readonly GuestViewTableEntry* _entries;
    private readonly GuestViewLogRecord* _log;
    private readonly object _gate = new();
    private int _overflowReports;
    private readonly bool _traceMappings = Environment.GetEnvironmentVariable("SHARPEMU_SPLIT_MISS_DETAILS") == "1";

    /// <summary>
    /// Builds an empty table over a writable shared region, with a log of
    /// <paramref name="logCapacity"/> records (a power of two, or 0 for none) at its end and entries
    /// in the rest; the table owns the region from here on. With <paramref name="trackProtection"/>
    /// the entries follow protection changes too (see the class summary).
    /// </summary>
    public GuestViewTable(HostSharedRegion region, ulong backingSize, int logCapacity = DefaultLogCapacity,
        bool trackProtection = true)
    {
        ArgumentNullException.ThrowIfNull(region);
        if (logCapacity < 0 || (logCapacity & (logCapacity - 1)) != 0)
            throw new ArgumentOutOfRangeException(nameof(logCapacity), "The log holds a power of two of records, or none.");
        if (!region.IsWritable || region.Size < RegionBytes(1, logCapacity))
            throw new ArgumentException("The view table needs a writable region of at least one entry and its log.", nameof(region));
        _region = region;
        LogCapacity = logCapacity;
        Capacity = (int)Math.Min(int.MaxValue, (region.Size - HeaderBytes - (ulong)logCapacity * LogRecordBytes) / EntryBytes);
        _header = (GuestViewTableHeader*)region.Address;
        _entries = (GuestViewTableEntry*)(region.Address + HeaderBytes);
        _log = (GuestViewLogRecord*)(region.Address + HeaderBytes + (ulong)Capacity * EntryBytes);
        *_header = default;
        _header->LayoutVersion = LayoutVersion;
        _header->Capacity = (uint)Capacity;
        _header->LogCapacity = (uint)logCapacity;
        _header->BackingSize = backingSize;
        _header->Flags = trackProtection ? ProtectionTrackedFlag : 0;
        TracksProtection = trackProtection;
        // A reader that sees the magic sees the rest of the header.
        Volatile.Write(ref _header->Magic, Magic);
    }

    /// <summary>A new table in a fresh shared region of <paramref name="host"/>; null when the host cannot make one.</summary>
    public static GuestViewTable? TryCreate(IHostGuestMirrorMemory host, ulong backingSize, int capacity = DefaultCapacity,
        int logCapacity = DefaultLogCapacity, bool trackProtection = true)
    {
        ArgumentNullException.ThrowIfNull(host);
        ArgumentOutOfRangeException.ThrowIfLessThan(capacity, 1);
        if (!host.TryCreateSharedRegion(RegionBytes(capacity, logCapacity), out var region) || region is null) return null;
        try { return new GuestViewTable(region, backingSize, logCapacity, trackProtection); }
        catch
        {
            region.Dispose();
            throw;
        }
    }

    public static ulong RegionBytes(int capacity, int logCapacity = DefaultLogCapacity) =>
        HeaderBytes + (ulong)capacity * EntryBytes + (ulong)logCapacity * LogRecordBytes;

    /// <summary>The shared region; another process maps its <see cref="HostSharedRegion.Descriptor"/> read-only.</summary>
    public HostSharedRegion Region => _region;

    public int Capacity { get; }

    public int LogCapacity { get; }

    /// <summary>Whether entries follow protection changes, or keep each view's mapping protection.</summary>
    public bool TracksProtection { get; }

    /// <summary>Log records published so far.</summary>
    public ulong LogSequence => Volatile.Read(ref _header->LogSequence);

    public int Count => (int)Volatile.Read(ref _header->Count);

    public bool IsComplete => (Volatile.Read(ref _header->Flags) & IncompleteFlag) == 0;

    public ulong Sequence => Volatile.Read(ref _header->Sequence);

    public ulong MappingGeneration => Volatile.Read(ref _header->MappingGeneration);

    /// <summary>[address, address + size) now maps backing offset <paramref name="offset"/>, replacing whatever the table held there.</summary>
    public void Map(ulong address, ulong size, ulong offset, HostPageProtection protection)
    {
        if (IsValidRange(address, size)) ChangeMapping(0, 0, Entry(address, size, offset, protection));
    }

    /// <summary>A record replaced in place: the old range goes and the new one comes in one change, so no reader sees neither.</summary>
    public void Remap(ulong oldAddress, ulong oldSize, ulong address, ulong size, ulong offset, HostPageProtection protection)
    {
        if (!IsValidRange(address, size)) Unmap(oldAddress, oldSize);
        else ChangeMapping(oldAddress, IsValidRange(oldAddress, oldSize) ? oldSize : 0, Entry(address, size, offset, protection));
    }

    /// <summary>[address, address + size) maps nothing any more.</summary>
    public void Unmap(ulong address, ulong size)
    {
        if (IsValidRange(address, size)) ChangeMapping(address, size, null);
    }

    /// <summary>
    /// The host protection of every mapped address in [address, address + size) is now
    /// <paramref name="protection"/>; unmapped addresses stay unmapped. Always logged; when the
    /// entries track protection, no change and no sequence step when nothing mapped there differs.
    /// </summary>
    public void Protect(ulong address, ulong size, HostPageProtection protection)
    {
        if (!IsValidRange(address, size)) return;
        var end = address + size;
        var raw = (uint)protection;
        lock (_gate)
        {
            if (!TracksProtection)
            {
                // No entry changes, so no table change either: a consistent read that sees this
                // record in LogSequence sees entries that are just as current.
                AppendLocked(GuestViewLogKind.Protect, address, size, 0, raw);
                return;
            }

            var count = (int)_header->Count;
            var first = FirstEndingAfterLocked(address);
            var last = first;
            var differs = false;
            for (; last < count && _entries[last].Address < end; last++) differs |= _entries[last].Protection != raw;
            if (!differs) return;

            BeginWrite(mapping: false);
            try
            {
                AppendLocked(GuestViewLogKind.Protect, address, size, 0, raw);
                // Split only an end entry the range cuts and whose protection changes; the
                // join pass below puts back together whatever ends up alike.
                if (_entries[first].Address < address && _entries[first].Protection != raw)
                {
                    if (!SplitLocked(first, address)) { NoteOverflowLocked(); return; }
                    first++;
                    last++;
                }

                if (_entries[last - 1].End > end && _entries[last - 1].Protection != raw && !SplitLocked(last - 1, end))
                {
                    NoteOverflowLocked();
                    return;
                }

                for (var index = first; index < last; index++) _entries[index].Protection = raw;
                JoinLocked(first - 1, last + 1);
            }
            finally { EndWrite(); }
        }
    }

    /// <summary>The entry holding <paramref name="address"/>, read as any process would read it.</summary>
    public bool TryFind(ulong address, out GuestViewEntry entry) =>
        GuestViewTableReader.TryFind(_header, _entries, Capacity, address, joinRun: false, out entry, out _);

    /// <summary>Every entry, in address order, as of one consistent moment.</summary>
    public GuestViewEntry[] Snapshot() => GuestViewTableReader.Snapshot(_header, _entries, Capacity);

    /// <summary>Log record <paramref name="number"/> (from 0), while the ring still holds it.</summary>
    public bool TryReadLog(ulong number, out GuestViewLogEntry entry)
    {
        entry = default;
        return number < LogSequence && GuestViewTableReader.TryReadLog(_log, LogCapacity, number, out entry);
    }

    public void Dispose() => _region.Dispose();

    private void ChangeMapping(ulong cutAddress, ulong cutSize, GuestViewTableEntry? added)
    {
        lock (_gate)
        {
            BeginWrite(mapping: true);
            try
            {
                if (cutSize != 0) AppendLocked(GuestViewLogKind.Unmap, cutAddress, cutSize, 0, 0);
                if (added is { } logged)
                    AppendLocked(GuestViewLogKind.Map, logged.Address, logged.Size, logged.Offset, logged.Protection);
                var fits = cutSize == 0 || CutLocked(cutAddress, cutAddress + cutSize);
                if (added is { } entry)
                    fits = CutLocked(entry.Address, entry.End) && InsertLocked(entry) && fits;
                if (!fits) NoteOverflowLocked();
            }
            finally { EndWrite(); }
            // Log after publishing the even sequence so diagnostic I/O does not hold readers
            // in the seqlock retry loop. Mapping-only history survives protection-log churn.
            if (_traceMappings)
            {
                if (cutSize != 0)
                    Console.Error.WriteLine($"[GPU-VIEW][UNMAP] generation={MappingGeneration} address=0x{cutAddress:X16} size=0x{cutSize:X}");
                if (added is { } traced)
                    Console.Error.WriteLine($"[GPU-VIEW][MAP] generation={MappingGeneration} address=0x{traced.Address:X16} size=0x{traced.Size:X} offset=0x{traced.Offset:X}");
            }
        }
    }

    // The sequence is odd from here to EndWrite, and the mapping generation moves first for a change
    // of what an address means: a reader holding a cached translation drops it before any entry moves.
    private void BeginWrite(bool mapping)
    {
        Volatile.Write(ref _header->Sequence, _header->Sequence + 1);
        if (mapping) Volatile.Write(ref _header->MappingGeneration, _header->MappingGeneration + 1);
        // No entry store may become visible before the odd sequence.
        Interlocked.MemoryBarrier();
    }

    // A release store: every entry store is visible before the even sequence.
    private void EndWrite() => Volatile.Write(ref _header->Sequence, _header->Sequence + 1);

    // Inside a change, so LogSequence moves with the entries it describes. The slot's sequence is
    // cleared before its fields change: a reader still on the previous lap sees it move under it.
    private void AppendLocked(GuestViewLogKind kind, ulong address, ulong size, ulong offset, uint protection)
    {
        if (LogCapacity == 0) return;
        var number = _header->LogSequence;
        ref var record = ref _log[(int)(number & (ulong)(LogCapacity - 1))];
        Volatile.Write(ref record.Sequence, 0);
        Interlocked.MemoryBarrier();
        record.Kind = (uint)kind;
        record.Protection = protection;
        record.Address = address;
        record.Size = size;
        record.Offset = offset;
        Volatile.Write(ref record.Sequence, number + 1);
        Volatile.Write(ref _header->LogSequence, number + 1);
    }

    // The first entry whose end lies above address: the one holding it, or the next one up.
    private int FirstEndingAfterLocked(ulong address)
    {
        int low = 0, high = (int)_header->Count;
        while (low < high)
        {
            var middle = low + ((high - low) >> 1);
            if (_entries[middle].End <= address) low = middle + 1;
            else high = middle;
        }

        return low;
    }

    // Takes [start, end) out of every entry: entries inside it go, entries reaching into it are
    // trimmed, and one holding all of it splits in two. False when that split does not fit.
    private bool CutLocked(ulong start, ulong end)
    {
        var count = (int)_header->Count;
        var first = FirstEndingAfterLocked(start);
        if (first == count || _entries[first].Address >= end) return true;
        var head = _entries[first];
        if (head.Address < start && head.End > end)
        {
            if (SplitLocked(first, end)) return TrimEndLocked(first, start);
            // No room for the far side: drop the whole entry. A missing range sends a reader to
            // the owner; a stale one would hand it the wrong bytes.
            RemoveLocked(first, 1);
            return false;
        }

        var removeFrom = first;
        if (head.Address < start)
        {
            TrimEndLocked(first, start);
            removeFrom++;
        }

        // Every entry from here starts at or above start.
        var removeTo = removeFrom;
        while (removeTo < count && _entries[removeTo].End <= end) removeTo++;
        if (removeTo < count && _entries[removeTo].Address < end)
        {
            ref var tail = ref _entries[removeTo];
            var cut = end - tail.Address;
            tail.Address = end;
            tail.Offset += cut;
            tail.Size -= cut;
        }

        RemoveLocked(removeFrom, removeTo - removeFrom);
        return true;
    }

    private bool TrimEndLocked(int index, ulong end)
    {
        _entries[index].Size = end - _entries[index].Address;
        return true;
    }

    // Splits the entry at index into [Address, at) and [at, End). False when the table is full.
    private bool SplitLocked(int index, ulong at)
    {
        var entry = _entries[index];
        var high = Entry(at, entry.End - at, entry.Offset + (at - entry.Address), (HostPageProtection)entry.Protection);
        if (!InsertAtLocked(index + 1, high)) return false;
        _entries[index].Size = at - entry.Address;
        return true;
    }

    // Inserts an entry that overlaps none, then joins it with its neighbours.
    private bool InsertLocked(GuestViewTableEntry entry)
    {
        var index = FirstEndingAfterLocked(entry.Address);
        if (!InsertAtLocked(index, entry)) return false;
        JoinLocked(index - 1, index + 2);
        return true;
    }

    private bool InsertAtLocked(int index, GuestViewTableEntry entry)
    {
        var count = (int)_header->Count;
        if (count >= Capacity) return false;
        if (index < count)
            Buffer.MemoryCopy(_entries + index, _entries + index + 1, (long)(Capacity - index - 1) * EntryBytes,
                (long)(count - index) * EntryBytes);
        _entries[index] = entry;
        _header->Count = (uint)(count + 1);
        return true;
    }

    private void RemoveLocked(int index, int removed)
    {
        if (removed == 0) return;
        var count = (int)_header->Count;
        if (index + removed < count)
            Buffer.MemoryCopy(_entries + index + removed, _entries + index, (long)(Capacity - index) * EntryBytes,
                (long)(count - index - removed) * EntryBytes);
        _header->Count = (uint)(count - removed);
    }

    // Joins each pair of neighbours in [from, to) whose addresses and offsets both continue at one protection.
    private void JoinLocked(int from, int to)
    {
        var index = Math.Max(from, 0);
        var stop = Math.Min(to, (int)_header->Count);
        while (index + 1 < stop)
        {
            ref var low = ref _entries[index];
            var high = _entries[index + 1];
            if (low.End == high.Address && low.Offset + low.Size == high.Offset && low.Protection == high.Protection)
            {
                low.Size += high.Size;
                RemoveLocked(index + 1, 1);
                stop--;
            }
            else
            {
                index++;
            }
        }
    }

    private void NoteOverflowLocked()
    {
        Interlocked.Or(ref _header->Flags, IncompleteFlag);
        if (++_overflowReports <= 4)
        {
            Console.Error.WriteLine(
                $"[VMEM][WARN] the shared guest view table is full ({Capacity} entries); it is incomplete from now on " +
                "and another process must ask the owner for translations.");
            Console.Error.Flush();
        }
    }

    private static GuestViewTableEntry Entry(ulong address, ulong size, ulong offset, HostPageProtection protection) =>
        new() { Address = address, Size = size, Offset = offset, Protection = (uint)protection };

    private static bool IsValidRange(ulong address, ulong size) => size != 0 && size <= ulong.MaxValue - address;
}

/// <summary>The shared header of a <see cref="GuestViewTable"/>; offsets are the cross-process layout.</summary>
[StructLayout(LayoutKind.Explicit, Size = GuestViewTable.HeaderBytes)]
internal struct GuestViewTableHeader
{
    [FieldOffset(0)] public uint Magic;
    [FieldOffset(4)] public uint LayoutVersion;
    [FieldOffset(8)] public uint Capacity;
    [FieldOffset(12)] public uint Count;
    [FieldOffset(16)] public ulong Sequence;
    [FieldOffset(24)] public ulong MappingGeneration;
    [FieldOffset(32)] public ulong BackingSize;
    [FieldOffset(40)] public uint Flags;
    [FieldOffset(44)] public uint LogCapacity;
    [FieldOffset(48)] public ulong LogSequence;
}

/// <summary>What a <see cref="GuestViewTable"/> log record says happened.</summary>
public enum GuestViewLogKind : uint
{
    /// <summary>[Address, Address + Size) now maps backing Offset, replacing what was there.</summary>
    Map = 1,

    /// <summary>[Address, Address + Size) maps nothing any more.</summary>
    Unmap = 2,

    /// <summary>The mapped addresses of [Address, Address + Size) now have host protection Protection.</summary>
    Protect = 3,
}

/// <summary>One log record of a view table, as it is read out of the ring.</summary>
public readonly record struct GuestViewLogEntry(ulong Number, GuestViewLogKind Kind, ulong Address, ulong Size, ulong Offset,
    HostPageProtection Protection);

[StructLayout(LayoutKind.Explicit, Size = GuestViewTable.LogRecordBytes)]
internal struct GuestViewLogRecord
{
    [FieldOffset(0)] public ulong Sequence;
    [FieldOffset(8)] public uint Kind;
    [FieldOffset(12)] public uint Protection;
    [FieldOffset(16)] public ulong Address;
    [FieldOffset(24)] public ulong Size;
    [FieldOffset(32)] public ulong Offset;
}

[StructLayout(LayoutKind.Explicit, Size = GuestViewTable.EntryBytes)]
internal struct GuestViewTableEntry
{
    [FieldOffset(0)] public ulong Address;
    [FieldOffset(8)] public ulong Size;
    [FieldOffset(16)] public ulong Offset;
    [FieldOffset(24)] public uint Protection;

    public readonly ulong End => Address + Size;
}

/// <summary>Lock-free reads of a view table, in the owner or in another process mapping it read-only.</summary>
internal static unsafe class GuestViewTableReader
{
    // The longest run of continuing neighbours one lookup joins: protection splits break a view into
    // a few entries, never into more than this many that a lookup has to walk.
    private const int MaximumRunEntries = 64;

    /// <summary>
    /// The parts of a view table another process mapped at <paramref name="address"/>. Capacities
    /// never reach past the <paramref name="bytes"/> mapped here, whatever the header claims.
    /// </summary>
    public static void Open(ulong address, ulong bytes, out GuestViewTableHeader* header, out GuestViewTableEntry* entries,
        out int capacity, out GuestViewLogRecord* log, out int logCapacity)
    {
        if (address == 0 || bytes < GuestViewTable.RegionBytes(1, 0))
            throw new ArgumentOutOfRangeException(nameof(bytes), "The view table region is too small.");
        header = (GuestViewTableHeader*)address;
        if (Volatile.Read(ref header->Magic) != GuestViewTable.Magic || header->LayoutVersion != GuestViewTable.LayoutVersion)
            throw new InvalidOperationException("The region does not hold a guest view table of this layout.");
        var logBytes = (ulong)header->LogCapacity * GuestViewTable.LogRecordBytes;
        var entryBytes = (ulong)header->Capacity * GuestViewTable.EntryBytes;
        if ((header->LogCapacity & (header->LogCapacity - 1)) != 0 || header->Capacity == 0 ||
            GuestViewTable.HeaderBytes + entryBytes + logBytes > bytes)
            throw new InvalidOperationException("The view table header does not fit the region mapped here.");
        entries = (GuestViewTableEntry*)(address + GuestViewTable.HeaderBytes);
        capacity = (int)header->Capacity;
        log = (GuestViewLogRecord*)(address + GuestViewTable.HeaderBytes + entryBytes);
        logCapacity = (int)header->LogCapacity;
    }

    /// <summary>
    /// The entry holding address as of one consistent moment, with the mapping generation of that
    /// moment. With joinRun, the entry grows over neighbours whose addresses and offsets both
    /// continue (whatever their protection), so it covers the whole translation run around address.
    /// </summary>
    public static bool TryFind(GuestViewTableHeader* header, GuestViewTableEntry* entries, int capacity, ulong address,
        bool joinRun, out GuestViewEntry entry, out ulong generation)
    {
        var spinner = new SpinWait();
        while (true)
        {
            var before = Volatile.Read(ref header->Sequence);
            if ((before & 1) == 0)
            {
                generation = Volatile.Read(ref header->MappingGeneration);
                var found = Search(entries, (int)Math.Min(header->Count, (uint)capacity), address, joinRun, out var copy);
                // Every entry load completes before the sequence is read again.
                Interlocked.MemoryBarrier();
                if (Volatile.Read(ref header->Sequence) == before)
                {
                    entry = found ? new GuestViewEntry(copy.Address, copy.Size, copy.Offset, (HostPageProtection)copy.Protection) : default;
                    return found;
                }
            }

            spinner.SpinOnce(sleep1Threshold: -1);
        }
    }

    public static GuestViewEntry[] Snapshot(GuestViewTableHeader* header, GuestViewTableEntry* entries, int capacity) =>
        Snapshot(header, entries, capacity, out _);

    /// <summary>Every entry, and the number of log records they are current to, as of one consistent moment.</summary>
    public static GuestViewEntry[] Snapshot(GuestViewTableHeader* header, GuestViewTableEntry* entries, int capacity,
        out ulong logSequence)
    {
        var spinner = new SpinWait();
        while (true)
        {
            var before = Volatile.Read(ref header->Sequence);
            if ((before & 1) == 0)
            {
                logSequence = Volatile.Read(ref header->LogSequence);
                var count = (int)Math.Min(header->Count, (uint)capacity);
                var copy = new GuestViewEntry[count];
                for (var index = 0; index < count; index++)
                {
                    var raw = entries[index];
                    copy[index] = new GuestViewEntry(raw.Address, raw.Size, raw.Offset, (HostPageProtection)raw.Protection);
                }

                Interlocked.MemoryBarrier();
                if (Volatile.Read(ref header->Sequence) == before) return copy;
            }

            spinner.SpinOnce(sleep1Threshold: -1);
        }
    }

    /// <summary>
    /// Log record <paramref name="number"/>, or false when the ring has moved past it (or a writer is
    /// overwriting it right now): the reader is a whole ring behind and must start again from the
    /// entries. The caller reads only numbers below the published LogSequence.
    /// </summary>
    public static bool TryReadLog(GuestViewLogRecord* log, int logCapacity, ulong number, out GuestViewLogEntry entry)
    {
        entry = default;
        if (logCapacity == 0) return false;
        ref var record = ref log[(int)(number & (ulong)(logCapacity - 1))];
        if (Volatile.Read(ref record.Sequence) != number + 1) return false;
        var copy = record;
        // Every field load completes before the slot's sequence is read again.
        Interlocked.MemoryBarrier();
        if (Volatile.Read(ref record.Sequence) != number + 1) return false;
        entry = new GuestViewLogEntry(number, (GuestViewLogKind)copy.Kind, copy.Address, copy.Size, copy.Offset,
            (HostPageProtection)copy.Protection);
        return true;
    }

    // Never trusts what it reads: a racing writer can leave any bytes behind, and the sequence check
    // throws the answer away afterwards. Indices stay inside [0, count).
    private static bool Search(GuestViewTableEntry* entries, int count, ulong address, bool joinRun, out GuestViewTableEntry found)
    {
        int low = 0, high = count - 1, index = -1;
        while (low <= high)
        {
            var middle = low + ((high - low) >> 1);
            if (entries[middle].Address <= address)
            {
                index = middle;
                low = middle + 1;
            }
            else
            {
                high = middle - 1;
            }
        }

        found = default;
        if (index < 0) return false;
        found = entries[index];
        if (address - found.Address >= found.Size) return false;
        if (!joinRun) return true;

        for (var below = index - 1; below >= 0 && index - below <= MaximumRunEntries; below--)
        {
            var neighbour = entries[below];
            if (neighbour.End != found.Address || neighbour.Offset + neighbour.Size != found.Offset) break;
            found.Address = neighbour.Address;
            found.Offset = neighbour.Offset;
            found.Size += neighbour.Size;
        }

        for (var above = index + 1; above < count && above - index <= MaximumRunEntries; above++)
        {
            var neighbour = entries[above];
            if (found.End != neighbour.Address || found.Offset + found.Size != neighbour.Offset) break;
            found.Size += neighbour.Size;
        }

        return true;
    }
}
