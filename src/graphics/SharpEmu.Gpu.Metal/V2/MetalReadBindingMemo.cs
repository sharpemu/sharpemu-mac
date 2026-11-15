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

    /// <summary>Ways per set, the most recently used first (move-to-front); the last one is evicted.</summary>
    internal const int Ways = 4;

    /// <summary>
    /// Sets in the table. A Demon's Souls frame binds some 20k distinct read-only ranges, which come back a frame later;
    /// 2048 sets of two ways lost nearly all of them to evictions. 16384 sets of four hold 65536 entries (about 0.3-0.5
    /// of them in use), so few sets overflow. SHARPEMU_METAL_V2_READ_BINDING_MEMO_SETS=&lt;power of two&gt; (diagnostic).
    /// </summary>
    internal static readonly int DefaultSets = ResolveSets(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_READ_BINDING_MEMO_SETS"));

    internal static int ResolveSets(string? configured)
    {
        const int sets = 16384;
        if (configured is null)
        {
            return sets;
        }

        if (int.TryParse(configured, out var value) && value >= 1 && value <= (1 << 22) && System.Numerics.BitOperations.IsPow2(value))
        {
            return value;
        }

        MetalV2Diagnostics.WarnOnce(
            "read-binding-memo-sets",
            $"SHARPEMU_METAL_V2_READ_BINDING_MEMO_SETS={configured} is not a power of two from 1 to 4194304; using {sets}.");
        return sets;
    }

    private readonly int _setMask;
    private readonly Entry[] _entries;

    public MetalReadBindingMemo(int sets = 0)
    {
        sets = sets == 0 ? DefaultSets : sets;
        _setMask = sets - 1;
        _entries = new Entry[sets * Ways];
    }

    public long Hits;
    public long Misses;
    public long Stores;
    public long Verified;

    // Why lookups missed: an empty set, other ranges' entries in it (collisions or evictions), or this range's
    // entry with another alias or stamp (each field that moved counts, so they can add up past the misses).
    // Evictions are stores over another range's entry.
    public long MissEmpty, MissOtherKey, MissAlias, MissSerial, MissPartition, MissTrap, MissImageSet, MissGpuModified;
    public long Evictions;

    // The caller's live refusals, one per refused lookup, the first that applied: a trace on, the range not mapped
    // whole, a deferred fill, a held label copy or a staged copy at all, or a matched entry whose window no longer
    // stands. StoreAfterRefusal: stores after a lookup refused for pending work.
    public long RefusedTrace, RefusedUnmapped, RefusedFill, RefusedLabel, RefusedStaged, RefusedWindow;
    public long StoreAfterRefusal;

    internal int SetOf(ulong address, ulong size, bool compute) => SetIndex(address, size, compute, _setMask);

    internal static int SetIndex(ulong address, ulong size, bool compute, int setMask)
    {
        var hash = ((address >> 4) * 0x9E3779B97F4A7C15UL) ^ (size * 0xC2B2AE3D27D4EB4FUL) ^ (compute ? 0x165667B19E3779F9UL : 0UL);
        return (int)((hash ^ (hash >> 29)) >> 40) & setMask;
    }

    private static bool Holds(in Entry entry, ulong address, ulong size, bool compute) =>
        entry.Valid && entry.Address == address && entry.Size == size && entry.Compute == compute;

    // The way holding the range moved to the front of its set; false when no way does.
    private bool TryFront(int first, ulong address, ulong size, bool compute)
    {
        var set = _entries.AsSpan(first, Ways);
        if (Holds(in set[0], address, size, compute))
        {
            return true;
        }

        for (var way = 1; way < Ways; way++)
        {
            if (Holds(in set[way], address, size, compute))
            {
                var found = set[way];
                set[..way].CopyTo(set[1..]);
                set[0] = found;
                return true;
            }
        }

        return false;
    }

    /// <summary>The range kept for this binding at this stamp and alias, if one is.</summary>
    public bool TryGet(ulong address, ulong size, bool compute, ulong alias, in Stamp stamp, out MetalGuestRange range)
    {
        var first = SetOf(address, size, compute) * Ways;
        if (!TryFront(first, address, size, compute))
        {
            var any = false;
            for (var way = 0; way < Ways; way++)
            {
                any |= _entries[first + way].Valid;
            }

            MissEmpty += any ? 0 : 1;
            MissOtherKey += any ? 1 : 0;
            range = default;
            return false;
        }

        ref var front = ref _entries[first];
        if (front.Alias == alias && front.Stamp == stamp)
        {
            range = front.Range;
            return true;
        }

        NoteStale(in front, alias, in stamp);
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
        var first = SetOf(address, size, compute) * Ways;
        if (!TryFront(first, address, size, compute))
        {
            // The least recently used way goes; the new entry takes the front.
            var set = _entries.AsSpan(first, Ways);
            Evictions += set[Ways - 1].Valid ? 1 : 0;
            set[..(Ways - 1)].CopyTo(set[1..]);
        }

        ref var entry = ref _entries[first];
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
        var text = $"hits={Hits} misses={Misses} stores={Stores} verified={Verified} evictions={Evictions} " +
            $"miss_by=(empty={MissEmpty},other_key={MissOtherKey},alias={MissAlias},serial={MissSerial}," +
            $"partition={MissPartition},trap={MissTrap},image_set={MissImageSet},gpu_modified={MissGpuModified}) " +
            $"refused_by=(trace={RefusedTrace},unmapped={RefusedUnmapped},window={RefusedWindow},fill={RefusedFill}," +
            $"label={RefusedLabel},staged={RefusedStaged}) store_after_refusal={StoreAfterRefusal}";
        Hits = Misses = Stores = Verified = Evictions = 0;
        MissEmpty = MissOtherKey = MissAlias = MissSerial = MissPartition = MissTrap = MissImageSet = MissGpuModified = 0;
        RefusedTrace = RefusedUnmapped = RefusedFill = RefusedLabel = RefusedStaged = RefusedWindow = StoreAfterRefusal = 0;
        return text;
    }
}
