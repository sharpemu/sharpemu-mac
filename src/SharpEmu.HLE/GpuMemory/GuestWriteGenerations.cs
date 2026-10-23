// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;
using System.Numerics;

namespace SharpEmu.HLE.GpuMemory;

// Durable content versions, separate from upload dirty bits. A sparse 2 MiB directory
// skips unwatched granules; bitmaps inside live granules select only watched 4 KiB pages.
// Enrollment publishes the bit before sampling the version and reading bytes. Stores
// never take an observer lock, allocate, or issue atomics for unwatched pages.
public sealed class GuestWriteGenerations : IDisposable
{
    private const ulong GranuleBytes = 2 * 1024 * 1024;
    private const int PagesPerGranule = (int)(GranuleBytes / TrackerLayout.PageBytes);
    private readonly WatchedGranule?[] _granules = new WatchedGranule[TrackerLayout.SpaceBytes / GranuleBytes];
    private sealed class WatchedGranule
    {
        public readonly long[] Versions = new long[PagesPerGranule];
        public readonly long[] Watched = new long[PagesPerGranule / 64];
    }
    private readonly ConcurrentDictionary<ulong, PageWatch> _pages = new();
    private readonly PageGuard? _guard;
    private long _epoch;

    private sealed class PageWatch(ulong address)
    {
        public readonly ulong Address = address;
        // Cold enrollment and native fault recovery serialize only this page.
        public int Gate;
        public int Armed;
    }

    public sealed class Token
    {
        private readonly GuestWriteGenerations _owner;
        private readonly long _epoch;
        private readonly int[] _indices;
        private readonly WatchedGranule[] _granules;
        private readonly long[] _versions;
        private readonly PageWatch[] _pages;

        private Token(GuestWriteGenerations owner, long epoch, int[] indices, WatchedGranule[] granules, long[] versions, PageWatch[] pages)
        {
            _owner = owner;
            _epoch = epoch;
            _indices = indices;
            _granules = granules;
            _versions = versions;
            _pages = pages;
        }

        public long RetainedBytes => 224L + _indices.Length * (sizeof(int) + sizeof(long) + 2 * IntPtr.Size);

        public bool IsCurrent
        {
            get
            {
                if (Volatile.Read(ref _owner._epoch) != _epoch) return false;
                for (var i = 0; i < _indices.Length; i++)
                    if (Volatile.Read(ref _granules[i].Versions[_indices[i]]) != _versions[i] ||
                        (_owner._guard is not null && Volatile.Read(ref _pages[i].Armed) == 0)) return false;
                return Volatile.Read(ref _owner._epoch) == _epoch;
            }
        }

        internal static Token Capture(GuestWriteGenerations owner, ulong address, ulong size)
        {
            var first = address / TrackerLayout.PageBytes;
            var count = checked((int)((address + size - 1) / TrackerLayout.PageBytes - first + 1));
            var epoch = Volatile.Read(ref owner._epoch);
            var indices = new int[count];
            var granules = new WatchedGranule[count];
            var versions = new long[count];
            var pages = new PageWatch[count];
            for (var i = 0; i < count; i++)
            {
                var page = (first + (ulong)i) * TrackerLayout.PageBytes;
                var slot = page / GranuleBytes;
                var granule = Volatile.Read(ref owner._granules[slot]);
                if (granule is null)
                {
                    var created = new WatchedGranule();
                    granule = Interlocked.CompareExchange(ref owner._granules[slot], created, null) ?? created;
                }
                var index = (int)((page / TrackerLayout.PageBytes) % PagesPerGranule);
                Interlocked.Or(ref granule.Watched[index / 64], 1L << (index % 64));
                indices[i] = index;
                granules[i] = granule;
                // Sample before arming; a fault in progress must invalidate this capture.
                versions[i] = Volatile.Read(ref granule.Versions[index]);
                if (owner._guard is null) continue;
                var watch = owner._pages.GetOrAdd(page, static p => new PageWatch(p));
                pages[i] = watch;
                Enter(watch);
                try
                {
                    if (watch.Armed == 0)
                    {
                        owner._guard.AddWatch(page, TrackerLayout.PageBytes, blockReads: false);
                        Volatile.Write(ref watch.Armed, 1);
                    }
                }
                finally { Volatile.Write(ref watch.Gate, 0); }
            }
            return new Token(owner, epoch, indices, granules, versions, pages);
        }
    }

    public GuestWriteGenerations(PageGuard? guard = null) => _guard = guard;

    // Register before reading bytes, and validate the SAME token after reading and on reuse.
    // A null guard is for managed-only memory whose every write is explicitly notified.
    public Token Register(ulong address, ulong size)
    {
        if (size == 0 || !new GuestSpan(address, size).IsValid) throw new ArgumentOutOfRangeException(nameof(size));
        return Token.Capture(this, address, size);
    }

    public void NoteWrite(ulong address, ulong size)
    {
        if (size == 0) return;
        if (!new GuestSpan(address, size).IsValid) { NoteUnboundedWrite(); return; }
        var first = address / TrackerLayout.PageBytes;
        var last = (address + size - 1) / TrackerLayout.PageBytes;
        // Bound work for unbounded/huge notifications just as the former striped table did.
        if (last - first >= 65536) { NoteUnboundedWrite(); return; }
        for (var slot = first / PagesPerGranule; slot <= last / PagesPerGranule; slot++)
        {
            var granule = Volatile.Read(ref _granules[slot]);
            if (granule is null) continue;
            var start = (int)(Math.Max(first, slot * PagesPerGranule) - slot * PagesPerGranule);
            var end = (int)(Math.Min(last, (slot + 1) * PagesPerGranule - 1) - slot * PagesPerGranule);
            for (var word = start / 64; word <= end / 64; word++)
            {
                var bits = (ulong)Volatile.Read(ref granule.Watched[word]);
                if (word == start / 64) bits &= ulong.MaxValue << (start % 64);
                if (word == end / 64) bits &= ulong.MaxValue >> (63 - end % 64);
                while (bits != 0)
                {
                    var bit = BitOperations.TrailingZeroCount(bits);
                    Interlocked.Increment(ref granule.Versions[word * 64 + bit]);
                    bits &= bits - 1;
                }
            }
        }
    }

    public void NoteUnboundedWrite() => Interlocked.Increment(ref _epoch);

    // Native stores fault only once until observation is armed again. Disarming the
    // observer is fault recovery, not part of the managed-store notification path.
    public bool ReleaseFaultWatch(ulong address)
    {
        var page = address & ~(TrackerLayout.PageBytes - 1);
        if (_guard is null || !_pages.TryGetValue(page, out var watch)) return false;
        Enter(watch);
        try
        {
            if (watch.Armed == 0) return false;
            Volatile.Write(ref watch.Armed, 0);
            _guard.RemoveWatch(page, TrackerLayout.PageBytes, blockReads: false);
            NoteWrite(page, TrackerLayout.PageBytes);
            return true;
        }
        finally { Volatile.Write(ref watch.Gate, 0); }
    }

    private static void Enter(PageWatch watch)
    {
        var spin = new SpinWait();
        while (Interlocked.CompareExchange(ref watch.Gate, 1, 0) != 0) spin.SpinOnce();
    }

    public void NoteRemap(ulong address, ulong size)
    {
        NoteUnboundedWrite();
        if (_guard is null || size == 0 || !new GuestSpan(address, size).IsValid) return;
        foreach (var page in _pages.Keys)
            if (page < address + size && address < page + TrackerLayout.PageBytes) ReleaseFaultWatch(page);
    }

    public void Dispose()
    {
        NoteUnboundedWrite();
        foreach (var page in _pages.Keys) ReleaseFaultWatch(page);
    }
}
