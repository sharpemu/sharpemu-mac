// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// The window range a read-only guest-range binding resolved to (MetalRenderHost.ResolveStorageBuffer through
/// MetalGuestMemory.TryResolveInPlace), kept for the bindings of the same range after it. The worker's alone: no
/// locks. An entry is kept only for a range that was mapped whole, over which the flush found no GPU-modified image,
/// and that resolved in place to a window; it answers while its <see cref="Stamp"/> equals the current one and the
/// range still translates to the same backing alias. What the caller checks live on every hit (no deferred fill or
/// label copy, no staged copy, no trace) is not part of the stamp.
/// </summary>
internal sealed class MetalReadBindingMemo
{
    /// <summary>
    /// What the skipped chain read that can change between two bindings of one range, other than the mapping
    /// (checked through the alias) and what the caller checks live.
    /// </summary>
    /// <param name="Serial">The guest memory's submission: windows are pinned and touched per command buffer.</param>
    /// <param name="Partition">The window cuts; a window's buffer is replaced only with a cut change.</param>
    /// <param name="Trap">Single-copy holes and stale windows (MetalGuestMemory's trap version).</param>
    /// <param name="ImageSet">Images added or dropped: which images the flush could find over the range.</param>
    /// <param name="GpuModified">An image going GPU-modified: which of them the flush would write back.</param>
    internal readonly record struct Stamp(ulong Serial, long Partition, long Trap, long ImageSet, long GpuModified);

    private struct Entry
    {
        public ulong Address;
        public ulong Size;
        public ulong Alias;
        public bool Compute;
        public bool Valid;
        public Stamp Stamp;
        public MetalGuestRange Range;
    }

    // Two ways per set, the more recently used first: a frame's command buffer binds a few thousand ranges, which
    // direct-mapped slots lost to collisions (most of the DsDrawCostBench draws' misses before).
    private const int Sets = 2048;
    private readonly Entry[] _entries = new Entry[Sets * 2];

    public long Hits;
    public long Misses;
    public long Stores;
    public long Verified;

    // Why lookups missed: an empty set, other ranges' entries in it (collisions or evictions), or this range's
    // entry with another alias or stamp (each field that moved counts, so they can add up past the misses); refused
    // are the caller's live refusals and windows that no longer stand. Evictions are stores over another range's entry.
    public long MissEmpty, MissOtherKey, MissAlias, MissSerial, MissPartition, MissTrap, MissImageSet, MissGpuModified;
    public long MissRefused, Evictions;

    private static int SetOf(ulong address, ulong size, bool compute)
    {
        var hash = ((address >> 4) * 0x9E3779B97F4A7C15UL) ^ (size * 0xC2B2AE3D27D4EB4FUL) ^ (compute ? 0x165667B19E3779F9UL : 0UL);
        return (int)((hash ^ (hash >> 29)) >> 40) & (Sets - 1);
    }

    private static bool Holds(in Entry entry, ulong address, ulong size, bool compute) =>
        entry.Valid && entry.Address == address && entry.Size == size && entry.Compute == compute;

    /// <summary>The range kept for this binding at this stamp and alias, if one is.</summary>
    public bool TryGet(ulong address, ulong size, bool compute, ulong alias, in Stamp stamp, out MetalGuestRange range)
    {
        var way = SetOf(address, size, compute) * 2;
        ref var first = ref _entries[way];
        if (!Holds(in first, address, size, compute))
        {
            ref var second = ref _entries[way + 1];
            if (Holds(in second, address, size, compute))
            {
                (first, second) = (second, first);
            }
            else
            {
                MissEmpty += !first.Valid && !second.Valid ? 1 : 0;
                MissOtherKey += first.Valid || second.Valid ? 1 : 0;
                range = default;
                return false;
            }
        }

        if (first.Alias == alias && first.Stamp == stamp)
        {
            range = first.Range;
            return true;
        }

        NoteStale(in first, alias, in stamp);
        range = default;
        return false;
    }

    private void NoteStale(in Entry entry, ulong alias, in Stamp stamp)
    {
        if (entry.Alias != alias) MissAlias++;
        if (entry.Stamp.Serial != stamp.Serial) MissSerial++;
        if (entry.Stamp.Partition != stamp.Partition) MissPartition++;
        if (entry.Stamp.Trap != stamp.Trap) MissTrap++;
        if (entry.Stamp.ImageSet != stamp.ImageSet) MissImageSet++;
        if (entry.Stamp.GpuModified != stamp.GpuModified) MissGpuModified++;
    }

    public void Store(ulong address, ulong size, bool compute, ulong alias, in Stamp stamp, in MetalGuestRange range)
    {
        var way = SetOf(address, size, compute) * 2;
        if (!Holds(in _entries[way], address, size, compute))
        {
            if (!Holds(in _entries[way + 1], address, size, compute))
            {
                // The less recently used way goes; the new entry takes the front.
                Evictions += _entries[way + 1].Valid ? 1 : 0;
            }

            _entries[way + 1] = _entries[way];
        }

        ref var entry = ref _entries[way];
        entry.Address = address;
        entry.Size = size;
        entry.Alias = alias;
        entry.Compute = compute;
        entry.Stamp = stamp;
        entry.Range = range;
        entry.Valid = true;
        Stores++;
    }

    /// <summary>hits, misses (and why), stores, evictions and verified hits since the last call; resets them.</summary>
    public string TakeCounters()
    {
        var text = $"hits={Hits} misses={Misses} stores={Stores} verified={Verified} " +
            $"miss_by=(empty={MissEmpty},other_key={MissOtherKey},alias={MissAlias},serial={MissSerial}," +
            $"partition={MissPartition},trap={MissTrap},image_set={MissImageSet},gpu_modified={MissGpuModified}," +
            $"refused={MissRefused}) evictions={Evictions}";
        Hits = Misses = Stores = Verified = 0;
        MissEmpty = MissOtherKey = MissAlias = MissSerial = MissPartition = MissTrap = MissImageSet = MissGpuModified = 0;
        MissRefused = Evictions = 0;
        return text;
    }
}
