// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// The bytes of command-processor writes the GPU has been told to make but has not made
/// yet, by guest address, so that what runs at parse time reads them as written.
/// </summary>
/// <remarks>
/// The hardware's command processor sees its own writes at once; on this host they land
/// when the GPU reaches them, in order with the labels around them. Until then a read
/// that overlaps one takes its bytes from here instead of waiting for the GPU or reading
/// what memory held before. An entry stops answering for bytes any later write covers -
/// a shader store, a fill, a copy, a label, a host write - because from then on program
/// order says those bytes are someone else's; and it is dropped once its submission has
/// completed, when memory itself holds it (and the guest may since have written over it).
/// </remarks>
internal sealed class MetalPendingWriteOverlay
{
    private const int PageShift = 12;

    // An entry is published under the lock; reads may come from a prefetch thread.
    private readonly object _gate = new();
    private readonly Dictionary<ulong, List<Entry>> _pages = new();
    private const int MaxSpareLists = 256;
    private readonly Stack<List<Entry>> _spareLists = new();
    private readonly Queue<Entry> _order = new();
    private ulong _sequence;
    private int _count;
    private long _version;

    // Every read the command stream and the resource walks make asks this overlay first (each
    // packet header, every word a cached materialization validates), and almost none of them
    // lands on a page a pending write is on. A count per hashed page of the entries listed on it
    // answers that without the lock: a range none of whose pages counts anything overlaps no
    // entry. The counts change under the lock with the page lists; a read that races an add
    // sees the overlay either before or after it, as it would by taking the lock first or second.
    // A collision only sends the read on to the locked walk.
    private const int FilterBits = 14;
    private const ulong FilterMaxPages = 64;
    private readonly int[]? _pageFilter;

    /// <param name="pageFilter">False takes the lock for every read, as before the page filter.</param>
    public MetalPendingWriteOverlay(bool pageFilter = true)
    {
        _pageFilter = pageFilter ? new int[1 << FilterBits] : null;
    }

    private sealed class Entry
    {
        public ulong Address;
        public byte[] Bytes = [];
        public ulong Serial;
        public ulong Sequence;
        public byte[]? Baseline;
        public bool[]? Dead;
        public bool Removed;

        public ulong End => Address + (ulong)Bytes.Length;
    }

    // A command-processor write is almost always a dword or a qword, and one entry with its
    // byte array per write was garbage the worker made at every draw's occlusion dump. Entries
    // of up to MaxPooledBytes are kept, by size, once they have left the page lists and the
    // order queue, and reused by the next write of that size. Guarded by the gate.
    private const int MaxPooledBytes = 16;
    private const int MaxPooledPerSize = 256;
    private readonly Stack<Entry>[] _spareEntries = [new(), new(), new(), new(), new()];

    private Entry RentEntry(int length)
    {
        if (length <= MaxPooledBytes && (length & 3) == 0 && _spareEntries[length >> 2] is { Count: > 0 } spare)
        {
            return spare.Pop();
        }

        return new Entry { Bytes = new byte[length] };
    }

    // Only an entry dequeued from the order queue and removed: nothing lists it any more.
    private void RecycleEntry(Entry entry)
    {
        var length = entry.Bytes.Length;
        if (length == 0 || length > MaxPooledBytes || (length & 3) != 0 ||
            _spareEntries[length >> 2].Count >= MaxPooledPerSize)
        {
            return;
        }

        entry.Baseline = null;
        entry.Dead = null;
        entry.Removed = false;
        _spareEntries[length >> 2].Push(entry);
    }

    /// <summary>Entries not yet dropped. Zero is the fast path every read checks first.</summary>
    public int Count => Volatile.Read(ref _count);

    /// <summary>
    /// Changes on every add, forget and clear (a forget stands for a newer write over the
    /// range); retiring landed entries leaves it alone.
    /// </summary>
    public long Version => Volatile.Read(ref _version);

    /// <summary>Reads guest memory as it stands, without waiting for anything.</summary>
    public delegate bool RawBytesReader(ulong address, Span<byte> destination);

    /// <summary>
    /// Records the bytes a write recorded into <paramref name="serial"/> will land.
    /// <paramref name="baseline"/>, when given, is what memory held over them at that point:
    /// <see cref="DropChangedBaselines"/> uses it to tell that someone else has written there since.
    /// </summary>
    public void Add(ulong address, ReadOnlySpan<byte> bytes, ulong serial, ReadOnlySpan<byte> baseline = default)
    {
        if (bytes.IsEmpty)
        {
            return;
        }

        lock (_gate)
        {
            AddLocked(address, bytes, serial, baseline);
        }
    }

    /// <summary>
    /// <see cref="Add"/> for <paramref name="count"/> eight-byte entries holding one value,
    /// <paramref name="strideBytes"/> apart, under one hold of the gate. Each is its own entry, so a
    /// later write over one of them forgets that one only, exactly as if added one by one.
    /// </summary>
    public void AddQwords(ulong address, int count, ulong strideBytes, ulong value, ulong serial)
    {
        if (count <= 0)
        {
            return;
        }

        Span<byte> bytes = stackalloc byte[sizeof(ulong)];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt64LittleEndian(bytes, value);
        lock (_gate)
        {
            for (var index = 0; index < count; index++)
            {
                AddLocked(address + (ulong)index * strideBytes, bytes, serial, default);
            }
        }
    }

    private void AddLocked(ulong address, ReadOnlySpan<byte> bytes, ulong serial, ReadOnlySpan<byte> baseline)
    {
        Interlocked.Increment(ref _version);
        var entry = RentEntry(bytes.Length);
        entry.Address = address;
        bytes.CopyTo(entry.Bytes);
        entry.Serial = serial;
        entry.Sequence = ++_sequence;
        if (baseline.Length == bytes.Length)
        {
            entry.Baseline = baseline.ToArray();
        }

        for (var page = address >> PageShift; page <= (entry.End - 1) >> PageShift; page++)
        {
            if (!_pages.TryGetValue(page, out var list))
            {
                _pages[page] = list = _spareLists.Count != 0 ? _spareLists.Pop() : new List<Entry>(2);
            }

            list.Add(entry);
            if (_pageFilter is not null)
            {
                Interlocked.Increment(ref _pageFilter[FilterSlot(page)]);
            }
        }

        _order.Enqueue(entry);
        _count++;
    }

    /// <summary>Drops every entry whose submission has completed.</summary>
    public void Retire(ulong completedSerial)
    {
        lock (_gate)
        {
            while (_order.Count != 0 && (_order.Peek().Serial <= completedSerial || _order.Peek().Removed))
            {
                var entry = _order.Dequeue();
                RemoveLocked(entry);
                RecycleEntry(entry);
            }
        }
    }

    /// <summary>A later write covers this range: no pending entry speaks for it any more.</summary>
    public void Forget(ulong address, ulong size)
    {
        Interlocked.Increment(ref _version);
        if (size == 0 || Count == 0 || !MayOverlap(address, size))
        {
            return;
        }

        lock (_gate)
        {
            var forget = new ForgetVisitor(this, address, size);
            ForEachOverlapping(address, size, ref forget);
        }
    }

    /// <summary>
    /// Drops the entries over the range whose baseline memory no longer holds: the guest CPU
    /// wrote there after the entry was recorded, and the entry no longer speaks for it. A
    /// read that fails also drops the entry, since nothing is known about those bytes.
    /// </summary>
    public void DropChangedBaselines(ulong address, ulong size, RawBytesReader read)
    {
        if (size == 0 || Count == 0 || !MayOverlap(address, size))
        {
            return;
        }

        lock (_gate)
        {
            var visit = new BaselineVisitor(this, read);
            ForEachOverlapping(address, size, ref visit);
        }
    }

    /// <summary>Drops everything: some write of unknown extent was recorded.</summary>
    public void Clear()
    {
        Interlocked.Increment(ref _version);
        lock (_gate)
        {
            while (_order.Count != 0)
            {
                var entry = _order.Dequeue();
                RemoveLocked(entry);
                RecycleEntry(entry);
            }
        }
    }

    /// <summary>
    /// How much of a range pending writes cover, after dropping those whose submission has
    /// completed. <paramref name="completedSerial"/> is asked for only when the range
    /// overlaps an entry at all, since asking the GPU is a message send.
    /// </summary>
    public MetalOverlayCoverage Coverage(ulong address, ulong size, Func<ulong> completedSerial)
    {
        if (size == 0 || Count == 0 || !MayOverlap(address, size))
        {
            return MetalOverlayCoverage.None;
        }

        lock (_gate)
        {
            if (!AnyOverlapLocked(address, size))
            {
                return MetalOverlayCoverage.None;
            }

            RetireOverlappingLocked(address, size, completedSerial());
            if (size > 64 * 1024)
            {
                return AnyOverlapLocked(address, size) ? MetalOverlayCoverage.Partial : MetalOverlayCoverage.None;
            }

            // Rented rather than allocated: this answers every word-line the command
            // processor parses while any write is pending.
            var covered = System.Buffers.ArrayPool<bool>.Shared.Rent((int)size);
            try
            {
                Array.Clear(covered, 0, (int)size);
                var coverage = new CoverageVisitor(address, size, covered);
                ForEachOverlapping(address, size, ref coverage);
                if (!coverage.Any)
                {
                    return MetalOverlayCoverage.None;
                }

                return Array.IndexOf(covered, false, 0, (int)size) < 0
                    ? MetalOverlayCoverage.Full
                    : MetalOverlayCoverage.Partial;
            }
            finally
            {
                System.Buffers.ArrayPool<bool>.Shared.Return(covered);
            }
        }
    }
    /// <summary>Writes the pending bytes over <paramref name="destination"/>, oldest first.</summary>
    public void Apply(ulong address, Span<byte> destination)
    {
        if (destination.IsEmpty || Count == 0 || !MayOverlap(address, (ulong)destination.Length))
        {
            return;
        }

        lock (_gate)
        {
            var size = (ulong)destination.Length;
            List<Entry>? entries = null;
            var collect = new CollectVisitor();
            ForEachOverlapping(address, size, ref collect);
            entries = collect.Entries;
            if (entries is null)
            {
                return;
            }

            entries.Sort(static (left, right) => left.Sequence.CompareTo(right.Sequence));
            foreach (var entry in entries)
            {
                var start = Math.Max(address, entry.Address);
                var end = Math.Min(address + size, entry.End);
                for (var at = start; at < end; at++)
                {
                    if (entry.Dead is null || !entry.Dead[at - entry.Address])
                    {
                        destination[(int)(at - address)] = entry.Bytes[at - entry.Address];
                    }
                }
            }
        }
    }

    // False only when no entry overlaps the range; true may be a collision or a long range.
    private bool MayOverlap(ulong address, ulong size)
    {
        if (_pageFilter is null)
        {
            return true;
        }

        var firstPage = address >> PageShift;
        var lastPage = (address + size - 1) >> PageShift;
        if (lastPage - firstPage >= FilterMaxPages)
        {
            return true;
        }

        for (var page = firstPage; page <= lastPage; page++)
        {
            if (Volatile.Read(ref _pageFilter[FilterSlot(page)]) != 0)
            {
                return true;
            }
        }

        return false;
    }

    private static int FilterSlot(ulong page) => (int)((page * 0x9E3779B97F4A7C15UL) >> (64 - FilterBits));

    private bool AnyOverlapLocked(ulong address, ulong size)
    {
        var any = new AnyVisitor();
        ForEachOverlapping(address, size, ref any);
        return any.Found;
    }

    private void RetireOverlappingLocked(ulong address, ulong size, ulong completedSerial)
    {
        var retire = new RetireVisitor(completedSerial);
        ForEachOverlapping(address, size, ref retire);
        var landed = retire.Landed;

        if (landed is not null)
        {
            foreach (var entry in landed)
            {
                RemoveLocked(entry);
            }
        }
    }

    // Visits each live entry overlapping the range once, in no particular order.
    private void ForEachOverlapping<TVisitor>(ulong address, ulong size, ref TVisitor visit)
        where TVisitor : struct, IEntryVisitor
    {
        if (_pages.Count == 0)
        {
            return;
        }

        var end = address + size;
        var firstPage = address >> PageShift;
        var lastPage = (end - 1) >> PageShift;
        Entry? previous = null;
        HashSet<Entry>? seen = null;
        if (lastPage - firstPage >= (ulong)_pages.Count)
        {
            // A range longer than the pages entries are on: only those pages are walked. The
            // order queue was walked here before, and it holds every entry until its submission
            // retires, forgotten ones too: GTA V's thousands of occlusion-dump qwords made each
            // fill, copy and written binding over 256 KiB scan them all (15% of its GPU worker).
            // A visit may drop a page's list, so the pages are taken first.
            _longRangePages.Clear();
            foreach (var page in _pages.Keys)
            {
                if (page >= firstPage && page <= lastPage)
                {
                    _longRangePages.Add(page);
                }
            }

            foreach (var page in _longRangePages)
            {
                if (_pages.TryGetValue(page, out var list))
                {
                    VisitPage(list, address, end, ref previous, ref seen, ref visit);
                }
            }

            return;
        }

        for (var page = firstPage; page <= lastPage; page++)
        {
            if (_pages.TryGetValue(page, out var list))
            {
                VisitPage(list, address, end, ref previous, ref seen, ref visit);
            }
        }
    }

    // The long-range walk's pages; under the gate.
    private readonly List<ulong> _longRangePages = new();

    private static void VisitPage<TVisitor>(
        List<Entry> list, ulong address, ulong end, ref Entry? previous, ref HashSet<Entry>? seen, ref TVisitor visit)
        where TVisitor : struct, IEntryVisitor
    {
        // Backwards, so a visit that removes the entry it was given keeps the walk valid.
        for (var index = list.Count - 1; index >= 0; index--)
        {
            if (index >= list.Count)
            {
                continue;
            }

            var entry = list[index];
            if (entry.Removed || entry.Address >= end || address >= entry.End)
            {
                continue;
            }

            // An entry spanning pages is listed on each; visit it once.
            if (entry.End - 1 >> PageShift != entry.Address >> PageShift)
            {
                if (entry == previous || !(seen ??= new HashSet<Entry>()).Add(entry))
                {
                    continue;
                }
            }

            previous = entry;
            visit.Visit(entry);
        }
    }

    // Struct visitors, so a probe allocates neither a closure nor a delegate.
    private interface IEntryVisitor
    {
        void Visit(Entry entry);
    }

    private struct AnyVisitor : IEntryVisitor
    {
        public bool Found;

        public void Visit(Entry entry) => Found = true;
    }

    private struct CollectVisitor : IEntryVisitor
    {
        public List<Entry>? Entries;

        public void Visit(Entry entry) => (Entries ??= new List<Entry>(2)).Add(entry);
    }

    private struct RetireVisitor(ulong completedSerial) : IEntryVisitor
    {
        public List<Entry>? Landed;

        public void Visit(Entry entry)
        {
            if (entry.Serial <= completedSerial)
            {
                (Landed ??= new List<Entry>(2)).Add(entry);
            }
        }
    }

    private struct CoverageVisitor(ulong address, ulong size, bool[] covered) : IEntryVisitor
    {
        public bool Any;

        public void Visit(Entry entry)
        {
            var start = Math.Max(address, entry.Address);
            var end = Math.Min(address + size, entry.End);
            for (var at = start; at < end; at++)
            {
                if (entry.Dead is null || !entry.Dead[at - entry.Address])
                {
                    covered[at - address] = true;
                    Any = true;
                }
            }
        }
    }

    private struct BaselineVisitor(MetalPendingWriteOverlay owner, RawBytesReader read) : IEntryVisitor
    {
        public void Visit(Entry entry)
        {
            if (entry.Baseline is null)
            {
                return;
            }

            Span<byte> now = stackalloc byte[entry.Baseline.Length];
            if (!read(entry.Address, now) || !now.SequenceEqual(entry.Baseline))
            {
                owner.RemoveLocked(entry);
            }
        }
    }

    private struct ForgetVisitor(MetalPendingWriteOverlay owner, ulong address, ulong size) : IEntryVisitor
    {
        public void Visit(Entry entry)
        {
            var start = Math.Max(address, entry.Address);
            var end = Math.Min(address + size, entry.End);
            if (start <= entry.Address && end >= entry.End)
            {
                owner.RemoveLocked(entry);
                return;
            }

            entry.Dead ??= new bool[entry.Bytes.Length];
            entry.Dead.AsSpan((int)(start - entry.Address), (int)(end - start)).Fill(true);
        }
    }

    private void RemoveLocked(Entry entry)
    {
        if (entry.Removed)
        {
            return;
        }

        entry.Removed = true;
        _count--;
        for (var page = entry.Address >> PageShift; page <= (entry.End - 1) >> PageShift; page++)
        {
            if (_pages.TryGetValue(page, out var list))
            {
                // A forget walks a page's list from its end, so the entry it drops is usually
                // the last one; a retire drops the oldest, which List.Remove finds first.
                if (list.Count != 0 && list[^1] == entry)
                {
                    list.RemoveAt(list.Count - 1);
                }
                else
                {
                    list.Remove(entry);
                }

                if (list.Count == 0)
                {
                    _pages.Remove(page);
                    // Reused by the next write's page: one list per page per write was garbage.
                    if (_spareLists.Count < MaxSpareLists)
                    {
                        _spareLists.Push(list);
                    }
                }
            }

            if (_pageFilter is not null)
            {
                Interlocked.Decrement(ref _pageFilter[FilterSlot(page)]);
            }
        }
    }
}

internal enum MetalOverlayCoverage
{
    None,
    Partial,
    Full,
}
