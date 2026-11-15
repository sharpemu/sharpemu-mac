// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// The window range a read-only guest-range binding resolved to, kept per range with what makes it stand
/// (MetalRenderHost.ReadBindingMemo.cs, SHARPEMU_METAL_V2_READ_BINDING_MEMO_RANGES). Unlike
/// <see cref="MetalReadBindingMemo"/>, whose entries answer only while nothing anywhere moved (the submission, the window
/// cuts, the image set, any image going GPU-modified), an entry here is checked against its own range: the window
/// object it was placed in is still in the partition with the same buffer, and the caller checks the rest per range.
/// The worker's alone: no locks.
/// </summary>
internal sealed class MetalReadBindingMemoV2
{
    internal const int Ways = MetalReadBindingMemo.Ways;

    private const byte FlagValid = 1;
    private const byte FlagCompute = 2;

    /// <summary>What the image flush over the range was answered by when the entry was stored.</summary>
    internal enum ImageTier : byte
    {
        /// <summary>No image owns a page of the range: the flush returns at its page test. Checked live.</summary>
        NoOwnerPages,

        /// <summary>Owner pages, but no image overlaps the bytes, at image-set version <see cref="Entry.ImageSet"/>.</summary>
        NoImage,

        /// <summary>One or two images overlap the bytes, none GPU-modified, at <see cref="Entry.ImageSet"/>.</summary>
        Held,
    }

    internal struct Entry
    {
        public ulong Address;
        public uint Size;
        public byte Flags;
        public ImageTier Tier;
        public byte HeldImages;
        public ulong Alias;
        public MetalWindowPartition.Window? Window;
        public nint Buffer;
        public ulong Offset;
        public ulong GpuAddress;
        public long ImageSet;
    }

    // Beside each entry: what is read only on some lookups.
    internal struct Side
    {
        // The single-copy trap version the window was checked at.
        public long Trap;

        // The submission the entry last answered in (hits_new_cb).
        public ulong Serial;

        // Tier Held: the images over the range.
        public MetalCachedImage? Image0;
        public MetalCachedImage? Image1;
    }

    private readonly int _setMask;
    private readonly Entry[] _entries;
    private readonly Side[] _sides;

    public MetalReadBindingMemoV2(int sets = 0)
    {
        sets = sets == 0 ? MetalReadBindingMemo.DefaultSets : sets;
        _setMask = sets - 1;
        _entries = new Entry[sets * Ways];
        _sides = new Side[sets * Ways];
    }

    public long Hits, Misses, Stores, Verified, Evictions;

    // One per miss, the first check that failed (they add up to Misses).
    public long MissEmpty, MissOtherKey, MissAlias, MissWindowRetired, MissWindowBuffer, MissTrap;
    public long MissImagePages, MissImageSet, MissImageGpuModified;
    public long RefusedTrace, RefusedUnmapped, RefusedWindow, RefusedFill, RefusedLabel, RefusedStaged;

    // What hits did: re-checked a moved trap or image-set version for the range, took the window lock to pin and
    // touch the window in this submission, answered in another submission than the last, answered while fills,
    // label copies or staged copies were pending (all of which the v1 memo refused).
    public long RevalidatedTrap, RevalidatedImageSet, WindowTouches, HitsNewCommandBuffer, HitsWithPending;

    // Chains whose answer was not kept: an image flushed, not a window, the window left the partition while the
    // chain ran, a range of 4 GiB or more, more than two images over the range.
    public long StoreSkipFlushed, StoreSkipNotInPlace, StoreSkipRetiredAfter, StoreSkipLarge, StoreSkipImageMany;

    private static byte FlagsOf(bool compute) => (byte)(FlagValid | (compute ? FlagCompute : 0));

    /// <summary>The entry at an index <see cref="Find"/> or <see cref="Insert"/> returned.</summary>
    public ref Entry At(int index) => ref _entries[index];

    public ref Side SideAt(int index) => ref _sides[index];

    /// <summary>
    /// The index of the range's entry, moved to the front of its set; -1 when the set holds none (the miss is
    /// counted as empty or other_key).
    /// </summary>
    public int Find(ulong address, ulong size, bool compute)
    {
        var first = MetalReadBindingMemo.SetIndex(address, size, compute, _setMask) * Ways;
        var flags = FlagsOf(compute);
        var any = false;
        for (var way = 0; way < Ways; way++)
        {
            ref var entry = ref _entries[first + way];
            if (entry.Flags == flags && entry.Address == address && entry.Size == size)
            {
                if (way != 0)
                {
                    MoveToFront(first, way);
                }

                return first;
            }

            any |= entry.Flags != 0;
        }

        Misses++;
        if (any)
        {
            MissOtherKey++;
        }
        else
        {
            MissEmpty++;
        }

        return -1;
    }

    /// <summary>
    /// The index of a front entry for the range, cleared but for its key: the range's entry if it had one, otherwise
    /// a new one, the set's least recently used entry evicted.
    /// </summary>
    public int Insert(ulong address, uint size, bool compute)
    {
        var first = MetalReadBindingMemo.SetIndex(address, size, compute, _setMask) * Ways;
        var flags = FlagsOf(compute);
        var found = -1;
        for (var way = 0; way < Ways && found < 0; way++)
        {
            ref var entry = ref _entries[first + way];
            if (entry.Flags == flags && entry.Address == address && entry.Size == size)
            {
                found = way;
            }
        }

        if (found < 0)
        {
            Evictions += _entries[first + Ways - 1].Flags != 0 ? 1 : 0;
            found = Ways - 1;
        }

        if (found != 0)
        {
            MoveToFront(first, found);
        }

        _entries[first] = new Entry { Address = address, Size = size, Flags = flags };
        _sides[first] = default;
        Stores++;
        return first;
    }

    private void MoveToFront(int first, int way)
    {
        var entry = _entries[first + way];
        var side = _sides[first + way];
        _entries.AsSpan(first, way).CopyTo(_entries.AsSpan(first + 1, way));
        _sides.AsSpan(first, way).CopyTo(_sides.AsSpan(first + 1, way));
        _entries[first] = entry;
        _sides[first] = side;
    }

    /// <summary>The counters since the last call (the PERF line's read_binding_memo); resets them.</summary>
    public string TakeCounters()
    {
        var text = $"ranges=1 hits={Hits} misses={Misses} stores={Stores} verified={Verified} evictions={Evictions} " +
            $"miss_by=(empty={MissEmpty},other_key={MissOtherKey},alias={MissAlias},window_retired={MissWindowRetired}," +
            $"window_buffer={MissWindowBuffer},trap={MissTrap},image_pages={MissImagePages},image_set={MissImageSet}," +
            $"image_gpu_modified={MissImageGpuModified}) " +
            $"refused_by=(trace={RefusedTrace},unmapped={RefusedUnmapped},window={RefusedWindow},fill={RefusedFill}," +
            $"label={RefusedLabel},staged={RefusedStaged}) " +
            $"revalidated=(trap={RevalidatedTrap},image_set={RevalidatedImageSet}) window_touches={WindowTouches} " +
            $"hits_new_cb={HitsNewCommandBuffer} hits_with_pending={HitsWithPending} " +
            $"store_skip=(flushed={StoreSkipFlushed},image_many={StoreSkipImageMany},not_in_place={StoreSkipNotInPlace}," +
            $"retired_after={StoreSkipRetiredAfter},large={StoreSkipLarge})";
        Hits = Misses = Stores = Verified = Evictions = 0;
        MissEmpty = MissOtherKey = MissAlias = MissWindowRetired = MissWindowBuffer = MissTrap = 0;
        MissImagePages = MissImageSet = MissImageGpuModified = 0;
        RefusedTrace = RefusedUnmapped = RefusedWindow = RefusedFill = RefusedLabel = RefusedStaged = 0;
        RevalidatedTrap = RevalidatedImageSet = WindowTouches = HitsNewCommandBuffer = HitsWithPending = 0;
        StoreSkipFlushed = StoreSkipNotInPlace = StoreSkipRetiredAfter = StoreSkipLarge = StoreSkipImageMany = 0;
        return text;
    }
}
