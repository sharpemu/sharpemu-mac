// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Metal.V2;

// A rejection hint, never permission to copy bytes. The fence list and its watches
// remain authoritative. Addresses are offsets in one cache's canonical backing,
// so different guest VAs of the same bytes use the same word. Leaves live until
// the cache dies; readers neither allocate nor take a Monitor.
internal sealed class MetalFencePageIndex
{
    internal const int PageBits = 12;
    internal const ulong AddressSpaceSize = 1UL << 40;
    private readonly long[]?[]?[] _root = new long[]?[]?[4096];

    internal static bool Valid(ulong address, ulong size) =>
        size != 0 && address < AddressSpaceSize && size <= AddressSpaceSize - address;

    internal static bool TryResolveBackingRange(SharpEmu.HLE.IGuestBackedSpace backing,
        ulong address, ulong size, out ulong offset)
    {
        offset = 0;
        if (size == 0 || size > ulong.MaxValue - address) return false;
        var snapshot = backing.BackingAliasSnapshot;
        var (start, capacity) = backing.BackingAliasRange;
        if (snapshot is null || !backing.TryResolveBackingAliasSpan(address, size, out var alias) ||
            alias < start || alias - start >= capacity || size > capacity - (alias - start) ||
            !ReferenceEquals(snapshot, backing.BackingAliasSnapshot)) return false;
        offset = alias - start;
        // Tracker-page queries and backing-page queries must have the same
        // boundaries. Unusual sub-page mappings stay on the authoritative path.
        return Valid(offset, size) && (offset & 4095) == (address & 4095);
    }

    // Serial is also the generation: submissions never reuse a serial. Overflow
    // is refused, and the caller retains an unindexed (always slow) fence instead.
    public bool Publish(ulong address, ulong size, ulong serial)
    {
        if (!Valid(address, size) || serial == 0 || serial > long.MaxValue) return false;
        var last = (address + size - 1) >> PageBits;
        for (var page = address >> PageBits; page <= last; page++)
        {
            ref var middleSlot = ref _root[page >> 16];
            var middle = Volatile.Read(ref middleSlot);
            if (middle is null)
            {
                var created = new long[]?[256];
                middle = Interlocked.CompareExchange(ref middleSlot, created, null) ?? created;
            }

            ref var leafSlot = ref middle[(page >> 8) & 255];
            var leaf = Volatile.Read(ref leafSlot);
            if (leaf is null)
            {
                var created = new long[256];
                leaf = Interlocked.CompareExchange(ref leafSlot, created, null) ?? created;
            }

            ref var word = ref leaf[page & 255];
            var old = Volatile.Read(ref word);
            while ((ulong)old < serial)
            {
                var seen = Interlocked.CompareExchange(ref word, (long)serial, old);
                if (seen == old) break;
                MetalStoreFastPathCounters.Count(MetalStoreFastPathCounters.Kind.CasRetry);
                old = seen;
            }
        }

        return true;
    }

    public bool MayOverlap(ulong address, ulong size, ulong completed = 0)
    {
        if (!Valid(address, size)) return true;
        var last = (address + size - 1) >> PageBits;
        for (var page = address >> PageBits; page <= last; page++)
        {
            var middle = Volatile.Read(ref _root[page >> 16]);
            var leaf = middle is null ? null : Volatile.Read(ref middle[(page >> 8) & 255]);
            if (leaf is not null && (ulong)Volatile.Read(ref leaf[page & 255]) > completed) return true;
        }

        return false;
    }

    // Only completed fences may retire. In particular a compatibility timeout
    // removing a watch must leave its hint positive until actual completion.
    public void Retire(ulong address, ulong size, ulong serial, ulong completed)
    {
        if (!Valid(address, size) || serial == 0 || serial > completed || serial > long.MaxValue) return;
        var last = (address + size - 1) >> PageBits;
        for (var page = address >> PageBits; page <= last; page++)
        {
            var middle = Volatile.Read(ref _root[page >> 16]);
            var leaf = middle is null ? null : Volatile.Read(ref middle[(page >> 8) & 255]);
            if (leaf is not null)
            {
                // Never clear a newer publication, even if it raced retirement.
                Interlocked.CompareExchange(ref leaf[page & 255], 0, (long)serial);
            }
        }
    }
}
