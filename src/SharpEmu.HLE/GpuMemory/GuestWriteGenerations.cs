// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;

namespace SharpEmu.HLE.GpuMemory;

// Durable content versions, separate from upload dirty bits. Hash collisions only reject
// an otherwise usable token. A normal store performs one interlocked increment, without
// looking up registered ranges or taking an observer lock.
public sealed class GuestWriteGenerations : IDisposable
{
    private const int StripeCount = 65536;
    private readonly long[] _versions = new long[StripeCount];
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
        private readonly int[] _stripes;
        private readonly long[] _versions;
        private readonly PageWatch[] _pages;

        private Token(GuestWriteGenerations owner, long epoch, int[] stripes, long[] versions, PageWatch[] pages)
        {
            _owner = owner;
            _epoch = epoch;
            _stripes = stripes;
            _versions = versions;
            _pages = pages;
        }

        public bool IsCurrent
        {
            get
            {
                if (Volatile.Read(ref _owner._epoch) != _epoch) return false;
                for (var i = 0; i < _stripes.Length; i++)
                    if (Volatile.Read(ref _owner._versions[_stripes[i]]) != _versions[i] ||
                        (_owner._guard is not null && Volatile.Read(ref _pages[i].Armed) == 0)) return false;
                return Volatile.Read(ref _owner._epoch) == _epoch;
            }
        }

        internal static Token Capture(GuestWriteGenerations owner, ulong address, ulong size)
        {
            var first = address / TrackerLayout.PageBytes;
            var count = checked((int)((address + size - 1) / TrackerLayout.PageBytes - first + 1));
            var epoch = Volatile.Read(ref owner._epoch);
            var stripes = new int[count];
            var versions = new long[count];
            var pages = new PageWatch[count];
            for (var i = 0; i < count; i++)
            {
                var page = (first + (ulong)i) * TrackerLayout.PageBytes;
                stripes[i] = Stripe(page);
                // Sample before arming; a fault in progress must invalidate this capture.
                versions[i] = Volatile.Read(ref owner._versions[stripes[i]]);
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
            return new Token(owner, epoch, stripes, versions, pages);
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

    private static int Stripe(ulong address) => (int)((address / TrackerLayout.PageBytes) % StripeCount);

    public void NoteWrite(ulong address, ulong size)
    {
        if (size == 0) return;
        if (!new GuestSpan(address, size).IsValid) { NoteUnboundedWrite(); return; }
        var last = (address + size - 1) / TrackerLayout.PageBytes;
        if (last - address / TrackerLayout.PageBytes >= StripeCount) { NoteUnboundedWrite(); return; }
        for (var page = address / TrackerLayout.PageBytes; page <= last; page++)
            Interlocked.Increment(ref _versions[page % StripeCount]);
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
