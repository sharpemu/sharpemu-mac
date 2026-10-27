// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE.Host;

namespace SharpEmu.HLE.GuestMemory;

/// <summary>
/// Turns guest virtual addresses into pointers into an offset mirror of the guest backing, through
/// a <see cref="GuestViewTable"/> another process owns: the two-process split's GPU process cannot
/// map guest memory at the guest addresses (macOS 27 guards 0xFC0000000-0x70_0000_0000, and the
/// process has a layout of its own), so it maps the backing wherever the host puts it
/// (<see cref="IHostGuestMirrorMemory.TryMapMirror"/>) and finds each guest address here, without
/// a call. (<see cref="GuestReplica"/> is the other way: every view mapped again at the guest address
/// plus a constant, so the GPU backend needs no lookup at all.)
///
/// <para>
/// A hit costs one load of the table's mapping generation and one cached range check. Each of the
/// <see cref="CacheSlots"/> slots, picked by a hash of the address's 64 KiB block, holds one range
/// under a sequence of its own, so threads share the cache without a lock and a miss allocates
/// nothing; a slot is good while the mapping generation it was read under is current, and
/// protection changes do not move that generation. Behind the slots each thread keeps the last
/// range it read from the table, which serves the other blocks of a range the thread walks through.
/// A miss reads the table under its sequence and caches the whole run of entries around the
/// address whose addresses and offsets both continue.
/// </para>
///
/// <para>
/// A translation is as current as the table was when it was read: an unmap the owner makes right
/// after it is not seen by the caller, exactly as a host pointer taken before an unmap in one
/// process. The split orders mapping changes against GPU work at its own sync points.
/// Addresses the backing does not hold (executable module pages, stacks, TLS, the HLE libc heap:
/// private memory of the owner) are never in the table; <see cref="IsFullyBacked"/> says whether a
/// range is wholly in it.
/// </para>
/// </summary>
public sealed unsafe class GuestMirrorTranslator
{
    public const int CacheSlots = 1 << CacheBits;

    private const int CacheBits = 10;

    // Slots are picked by a hash of the 64 KiB block of the address, the guest's own mapping
    // granule: buffers placed at a power-of-two stride would otherwise all land in one slot.
    private const int CacheShift = 16;

    private readonly GuestViewTableHeader* _header;
    private readonly GuestViewTableEntry* _entries;
    private readonly int _capacity;
    private readonly GuestViewLogRecord* _log;
    private readonly int _logCapacity;
    private readonly Slot[] _cache = new Slot[CacheSlots];
    private long _hits;
    private long _misses;

    // The last range the thread read from the table, for whichever translator it read it from.
    [ThreadStatic]
    private static LastRange t_last;

    // One cached range under a sequence: odd while a writer fills it, so a reader that sees the
    // same even sequence before and after its loads read one writer's range. An empty slot
    // (End 0) holds no address.
    private struct Slot
    {
        public long Sequence;
        public ulong Start;
        public ulong End;
        public ulong Offset;
        public ulong Generation;
    }

    private struct LastRange
    {
        public GuestMirrorTranslator? Owner;
        public CachedRange Range;
    }

    private readonly record struct CachedRange(ulong Start, ulong End, ulong Offset, ulong Generation);

    /// <summary>
    /// A translator over a mapped view table (often read-only) and a mirror of the backing it
    /// describes. Both mappings must outlive the translator.
    /// </summary>
    public GuestMirrorTranslator(ulong tableAddress, ulong tableBytes, ulong mirrorBase, ulong mirrorBytes)
    {
        GuestViewTableReader.Open(tableAddress, tableBytes, out _header, out _entries, out _capacity, out _log, out _logCapacity);
        MirrorBase = mirrorBase;
        MirrorBytes = Math.Min(mirrorBytes, _header->BackingSize);
    }

    public GuestMirrorTranslator(HostSharedRegion table, HostBackingObject mirror)
        : this(table.Address, table.Size, mirror.AliasBase, mirror.Size)
    {
    }

    /// <summary>Host address of backing offset 0 in this process.</summary>
    public ulong MirrorBase { get; }

    /// <summary>Backing bytes the mirror covers; no translation reaches past them.</summary>
    public ulong MirrorBytes { get; }

    /// <summary>False once the owner could not fit a change: translations must then come from the owner.</summary>
    public bool IsComplete => (Volatile.Read(ref _header->Flags) & GuestViewTable.IncompleteFlag) == 0;

    public ulong MappingGeneration => Volatile.Read(ref _header->MappingGeneration);

    /// <summary>The sequence after the last published mapping/protection change, for diagnostics.</summary>
    public ulong ChangeSequence => Volatile.Read(ref _header->LogSequence);

    /// <summary>A published change while the owner's ring still retains it. Reading it
    /// does not advance the replica or alter translation caches.</summary>
    public bool TryReadChange(ulong number, out GuestViewLogEntry entry)
    {
        entry = default;
        return number < ChangeSequence && GuestViewTableReader.TryReadLog(_log, _logCapacity, number, out entry);
    }

    /// <summary>
    /// Whether translations count their cache hits and misses (<see cref="CacheHits"/>,
    /// <see cref="CacheMisses"/>). Off unless SHARPEMU_SPLIT_STATS=1: the worker, the program
    /// resolvers and the ring threads all translate, and a counter they share is a cache line
    /// bouncing between cores on every address of every descriptor, shader read and staged range.
    /// </summary>
    public bool CountCacheAccesses { get; set; } = GpuMemory.SplitCrossingStats.Enabled;

    // Diagnostics only: plain increments, so counts from racing threads may be lost.
    public long CacheHits => Volatile.Read(ref _hits);

    public long CacheMisses => Volatile.Read(ref _misses);

    /// <summary>
    /// The mirror address of guest [address, address + size), when one run of the table holds all
    /// of it; false when any of it is unmapped, private to the owner, or split over runs whose
    /// offsets do not continue (the caller then translates piece by piece).
    /// </summary>
    public bool TryTranslate(ulong address, ulong size, out ulong mirrorAddress)
    {
        mirrorAddress = 0;
        if (size == 0 || size > ulong.MaxValue - address || !TryGetRange(address, out var range) || address + size > range.End)
            return false;
        mirrorAddress = MirrorBase + range.Offset + (address - range.Start);
        return true;
    }

    /// <summary>
    /// The backing offset of guest <paramref name="address"/> and how many bytes from there continue
    /// in the backing (to the end of its run).
    /// </summary>
    public bool TryGetOffset(ulong address, out ulong offset, out ulong contiguousBytes)
    {
        offset = contiguousBytes = 0;
        if (!TryGetRange(address, out var range)) return false;
        offset = range.Offset + (address - range.Start);
        contiguousBytes = range.End - address;
        return true;
    }

    /// <summary>
    /// Whether every byte of guest [address, address + size) is a view of the backing, however many
    /// views it crosses and wherever in the backing they lie. False names the first byte that is not:
    /// memory private to the owner (module code, stacks, TLS, the HLE libc heap) or unmapped, which
    /// this process must ask the owner about instead of reading it (docs/macos/split-shared-memory.md).
    /// Reads the table, not the cache, so it costs one lookup per run of views.
    /// </summary>
    public bool IsFullyBacked(ulong address, ulong size, out ulong firstUnbacked)
    {
        firstUnbacked = address;
        if (size == 0 || size > ulong.MaxValue - address) return false;
        var end = address + size;
        while (firstUnbacked < end)
        {
            if (!GuestViewTableReader.TryFind(_header, _entries, _capacity, firstUnbacked, joinRun: true, out var entry, out _) ||
                entry.End <= firstUnbacked)
                return false;
            firstUnbacked = entry.End;
        }

        firstUnbacked = 0;
        return true;
    }

    /// <summary>
    /// The table entry holding <paramref name="address"/>, never cached. Its protection is the view's
    /// mapping protection unless the table tracks protection changes (<see cref="GuestViewTable.TracksProtection"/>).
    /// </summary>
    public bool TryGetEntry(ulong address, out GuestViewEntry entry) =>
        GuestViewTableReader.TryFind(_header, _entries, _capacity, address, joinRun: false, out entry, out _);

    /// <summary>Every entry, in address order, as of one consistent moment.</summary>
    public GuestViewEntry[] Snapshot() => GuestViewTableReader.Snapshot(_header, _entries, _capacity);

    private bool TryGetRange(ulong address, out CachedRange range)
    {
        var generation = Volatile.Read(ref _header->MappingGeneration);
        ref var slot = ref _cache[SlotIndex(address)];
        var sequence = Volatile.Read(ref slot.Sequence);
        if ((sequence & 1) == 0)
        {
            // Acquire loads: none moves above the sequence read, and the second read stays below them.
            range = new CachedRange(Volatile.Read(ref slot.Start), Volatile.Read(ref slot.End),
                Volatile.Read(ref slot.Offset), Volatile.Read(ref slot.Generation));
            if (Volatile.Read(ref slot.Sequence) == sequence && range.Generation == generation &&
                address >= range.Start && address < range.End)
            {
                if (CountCacheAccesses) _hits++;
                return true;
            }
        }

        ref var last = ref t_last;
        if (ReferenceEquals(last.Owner, this) && last.Range.Generation == generation &&
            address >= last.Range.Start && address < last.Range.End)
        {
            if (CountCacheAccesses) _hits++;
            range = last.Range;
            // The slot answers the next address of this block without the thread's lookup.
            Store(ref slot, range);
            return true;
        }

        if (CountCacheAccesses) _misses++;
        range = default;
        if (!GuestViewTableReader.TryFind(_header, _entries, _capacity, address, joinRun: true, out var entry, out var readGeneration))
            return false;
        // A run reaching past the mirror is cut at its end; an address past it has no translation.
        if (entry.Offset >= MirrorBytes) return false;
        var size = Math.Min(entry.Size, MirrorBytes - entry.Offset);
        if (address - entry.Address >= size) return false;
        range = new CachedRange(entry.Address, entry.Address + size, entry.Offset, readGeneration);
        last.Owner = this;
        last.Range = range;
        Store(ref slot, range);
        return true;
    }

    private static int SlotIndex(ulong address) =>
        (int)(((address >> CacheShift) * 0x9E3779B97F4A7C15UL) >> (64 - CacheBits));

    // A writer that finds the slot being filled leaves it to the other one.
    private static void Store(ref Slot slot, in CachedRange range)
    {
        var sequence = Volatile.Read(ref slot.Sequence);
        if ((sequence & 1) != 0 || Interlocked.CompareExchange(ref slot.Sequence, sequence + 1, sequence) != sequence)
            return;
        // The exchange is a full barrier: these stores are not seen before the odd sequence.
        slot.Start = range.Start;
        slot.End = range.End;
        slot.Offset = range.Offset;
        slot.Generation = range.Generation;
        Volatile.Write(ref slot.Sequence, sequence + 2);
    }
}
