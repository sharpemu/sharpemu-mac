// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Metal.V2;

// The worker owns the list; completion callbacks only publish their record's final
// state. GPU event completion alone does not land a host-written label or CP store.
internal sealed class MetalHostPublications
{
    internal sealed class Record(ulong address, ulong size, ulong serial)
    {
        internal readonly ulong Address = address, Size = size, Serial = serial;
        private int _complete;
        internal bool Completed => Volatile.Read(ref _complete) != 0;
        internal void Complete() => Volatile.Write(ref _complete, 1);
    }

    private readonly List<Record> _records = [];

    internal ulong Version { get; private set; }

    internal Record Add(ulong address, ulong size, ulong serial)
    {
        Version++;
        _records.RemoveAll(static record => record.Completed);
        var record = new Record(address, size, serial);
        _records.Add(record);
        return record;
    }

    internal long AliasFallbacks { get; private set; }

    internal ulong Pending(ulong address, ulong size, SharpEmu.HLE.IGuestBackedSpace? backing = null)
    {
        ulong serial = 0, all = 0;
        var snapshot = backing?.BackingAliasSnapshot;
        var queryOffset = 0UL;
        var queryResolved = backing is not null && MetalFencePageIndex.TryResolveBackingRange(backing, address, size, out queryOffset);
        foreach (var record in _records)
        {
            if (record.Completed) continue;
            all = Math.Max(all, record.Serial);
            var overlap = Overlaps(address, size, record.Address, record.Size);
            if (!overlap && backing is not null)
            {
                // The callback writes through its VA when it runs, so resolve both
                // sides against today's mapping, including remapped/aliased tables.
                if (!queryResolved || !MetalFencePageIndex.TryResolveBackingRange(backing, record.Address, record.Size, out var offset))
                {
                    overlap = true;
                    AliasFallbacks++;
                }
                else overlap = Overlaps(queryOffset, size, offset, record.Size);
            }
            if (overlap) serial = Math.Max(serial, record.Serial);
        }
        // A mapping publication racing the query can only demand a wider wait.
        return backing is not null && !ReferenceEquals(snapshot, backing.BackingAliasSnapshot) ? all : serial;
    }

    private static bool Overlaps(ulong address, ulong size, ulong other, ulong otherSize) =>
        size != 0 && otherSize != 0 &&
        (address <= other ? other - address < size : address - other < otherSize);
}
