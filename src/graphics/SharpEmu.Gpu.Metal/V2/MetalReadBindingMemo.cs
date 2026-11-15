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

    private const int Slots = 1024;
    private readonly Entry[] _entries = new Entry[Slots];

    public long Hits;
    public long Misses;
    public long Stores;
    public long Verified;

    private static int SlotOf(ulong address, ulong size, bool compute) =>
        (int)((((address >> 4) * 0x9E3779B97F4A7C15UL) ^ (size * 0xC2B2AE3D27D4EB4FUL) ^ (compute ? 1UL : 0UL)) >> 54) & (Slots - 1);

    /// <summary>The range kept for this binding at this stamp and alias, if one is.</summary>
    public bool TryGet(ulong address, ulong size, bool compute, ulong alias, in Stamp stamp, out MetalGuestRange range)
    {
        ref var entry = ref _entries[SlotOf(address, size, compute)];
        if (entry.Valid && entry.Address == address && entry.Size == size && entry.Compute == compute &&
            entry.Alias == alias && entry.Stamp == stamp)
        {
            range = entry.Range;
            return true;
        }

        range = default;
        return false;
    }

    public void Store(ulong address, ulong size, bool compute, ulong alias, in Stamp stamp, in MetalGuestRange range)
    {
        ref var entry = ref _entries[SlotOf(address, size, compute)];
        entry.Address = address;
        entry.Size = size;
        entry.Alias = alias;
        entry.Compute = compute;
        entry.Stamp = stamp;
        entry.Range = range;
        entry.Valid = true;
        Stores++;
    }

    /// <summary>hits, misses, stores and verified hits since the last call; resets them.</summary>
    public string TakeCounters()
    {
        var text = $"hits={Hits} misses={Misses} stores={Stores} verified={Verified}";
        Hits = Misses = Stores = Verified = 0;
        return text;
    }
}
