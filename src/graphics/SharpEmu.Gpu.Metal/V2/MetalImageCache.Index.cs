// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Scheduling;

namespace SharpEmu.Gpu.Metal.V2;

// Generational slots for cached images: the shared engine only ever holds
// ResourceSlotIdentifier values, so a slot can be erased while an identifier for it
// is still in flight and the stale identifier simply stops resolving.
internal sealed class MetalImageSlots
{
    private readonly List<MetalCachedImage?> _entries = new();
    private readonly List<uint> _generations = new();
    private readonly Stack<int> _free = new();

    public int Count { get; private set; }

    public MetalCachedImage this[ResourceSlotIdentifier identifier] =>
        TryGet(identifier) ?? throw GpuFatal.Fatal(
            $"The image slot is no longer live: index={identifier.Index} generation={identifier.Generation}.");

    public MetalCachedImage? TryGet(ResourceSlotIdentifier identifier)
    {
        var index = (int)identifier.Index;
        if (identifier.Generation == 0 || index >= _entries.Count || _generations[index] != identifier.Generation)
        {
            return null;
        }

        return _entries[index];
    }

    public ResourceSlotIdentifier Insert(MetalCachedImage image)
    {
        if (_free.TryPop(out var index))
        {
            _entries[index] = image;
            Count++;
            return new ResourceSlotIdentifier((uint)index, _generations[index]);
        }

        index = _entries.Count;
        // Generation 0 never identifies a live slot, so a fresh slot starts at 1.
        _entries.Add(image);
        _generations.Add(1);
        Count++;
        return new ResourceSlotIdentifier((uint)index, 1);
    }

    public void Erase(ResourceSlotIdentifier identifier)
    {
        var index = (int)identifier.Index;
        if (index >= _entries.Count || _generations[index] != identifier.Generation || _entries[index] is null)
        {
            return;
        }

        _entries[index] = null;
        _generations[index] = _generations[index] + 1 == 0 ? 1u : _generations[index] + 1;
        _free.Push(index);
        Count--;
    }

    public void ForEach(Action<ResourceSlotIdentifier, MetalCachedImage> visit)
    {
        for (var index = 0; index < _entries.Count; index++)
        {
            if (_entries[index] is { } image)
            {
                visit(new ResourceSlotIdentifier((uint)index, _generations[index]), image);
            }
        }
    }
}

// The images that own one 1 MiB guest page, in a sparse two-level lookup that a query
// never allocates a bucket for, and beside it a count of the live owners of each
// 64 KiB page, which lets a CPU write on the faulting thread skip the cache lock when
// no image covers the bytes it touches.
//
// The lists are read and written only under the image cache lock. The counts are
// written only under it too, but read without it, and they stay correct for such a
// reader because of the order Add and Remove keep:
//
//  * Add raises the counts before the image goes into any owner list. Nothing finds an
//    image except through those lists, and the cache arms an image's page watch only
//    after it is registered, so by the time a lookup can return the image or a store
//    can fault on its watch, every count over its bytes is already non-zero. Each
//    increment is a full fence, so no later store can be seen ahead of it.
//  * Remove takes the image out of every list before it lowers the counts, and the
//    cache drops the page watch before either. An image a count no longer shows is
//    one no lookup can find and no watch can report.
//
// So a reader that sees zero over a range sees what a locked lookup at that moment
// would have found: no image over those bytes. The count only gates the lock; the
// locked path still asks the lists and the byte and page overlap tests. A count page
// spans whole 4 KiB tracker pages, so the page-granular overlap a fault uses can never
// reach an image the count does not show.
internal sealed class MetalImagePageOwners
{
    public const int PageBits = 20;
    public const int AddressSpaceBits = 40;
    public const int FirstLevelBits = 10;
    public const int SecondLevelBits = AddressSpaceBits - FirstLevelBits - PageBits;
    public const int BucketEntries = 1 << SecondLevelBits;
    public const ulong PageCount = 1UL << (AddressSpaceBits - PageBits);
    public const ulong AddressSpaceSize = 1UL << AddressSpaceBits;

    // Finer than the owner lists: a command buffer the guest builds next to a texture
    // shares its 1 MiB page often enough to send every store through the lock.
    public const int CountPageBits = 16;
    private const int CountLeafBits = 12;
    private const int CountRootBits = AddressSpaceBits - CountPageBits - CountLeafBits;
    private const ulong CountLeafMask = (1UL << CountLeafBits) - 1;

    private readonly List<ResourceSlotIdentifier>?[]?[] _firstLevel =
        new List<ResourceSlotIdentifier>?[]?[1 << FirstLevelBits];

    // How many times each page's owner list changed. A memo of a lookup over a range is stale
    // once the sum over its pages moved (they only grow, so an equal sum is no change).
    private readonly long[]?[] _versions = new long[]?[1 << FirstLevelBits];

    // Live owners per 64 KiB page. A leaf, once published, is never replaced or freed,
    // so a lock-free reader that finds it may keep reading it.
    private readonly int[]?[] _liveOwners = new int[]?[1 << CountRootBits];

    // Images smaller than an owner page are listed per 64 KiB fine page instead (SHARPEMU_METAL_V2_IMAGE_FINE_PAGES=0 lists
    // every image per 1 MiB page, as before): Ghost of Tsushima's texture pool packs hundreds of small images into each
    // 1 MiB page, and a lookup there walked all of them; a large image, listed per 1 MiB page, is still walked once per
    // page of a large range. _fineMasks marks, per 1 MiB page, which of its sixteen fine pages list an owner, so a range -
    // a render target's, say, over pages a few small images share - walks only those. Versions stay per 1 MiB page.
    public static readonly bool FinePagesEnabled =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_IMAGE_FINE_PAGES") != "0";
    public const int FinePageBits = CountPageBits;
    private const int FineSecondLevelBits = AddressSpaceBits - FirstLevelBits - FinePageBits;
    private const int FineBucketEntries = 1 << FineSecondLevelBits;
    private const ulong FineMaxBytes = 1UL << PageBits;
    private const int FinePagesPerPage = 1 << (PageBits - FinePageBits);
    private readonly List<ResourceSlotIdentifier>?[]?[] _fineFirstLevel =
        new List<ResourceSlotIdentifier>?[]?[1 << FirstLevelBits];
    private readonly ushort[]?[] _fineMasks = new ushort[]?[1 << FirstLevelBits];

    public static bool IsFine(ulong size) => FinePagesEnabled && size < FineMaxBytes;

    /// <summary>The fine owners listed over a 64 KiB page (images smaller than an owner page).</summary>
    public List<ResourceSlotIdentifier>? FindFine(ulong finePage) =>
        finePage < (1UL << (AddressSpaceBits - FinePageBits)) && _fineFirstLevel[finePage >> FineSecondLevelBits] is { } bucket
            ? bucket[finePage & (FineBucketEntries - 1)]
            : null;

    /// <summary>Which fine pages of this 1 MiB page list an image smaller than an owner page: bit i, the i-th 64 KiB.</summary>
    public ushort FineMask(ulong page) =>
        IsValidPage(page) && _fineMasks[page >> SecondLevelBits] is { } bucket ? bucket[page & (BucketEntries - 1)] : (ushort)0;

    /// <summary>Whether any image smaller than an owner page is listed inside this 1 MiB page.</summary>
    public bool HasFineOwners(ulong page) => FineMask(page) != 0;

    private void AddFine(ulong address, ulong size, ResourceSlotIdentifier imageIdentifier, int delta)
    {
        var first = address >> FinePageBits;
        var last = (address + size - 1) >> FinePageBits;
        for (var finePage = first; finePage <= last; finePage++)
        {
            ref var bucket = ref _fineFirstLevel[finePage >> FineSecondLevelBits];
            var page = finePage >> (PageBits - FinePageBits);
            ref var masks = ref _fineMasks[page >> SecondLevelBits];
            var bit = (ushort)(1 << (int)(finePage & (FinePagesPerPage - 1)));
            if (delta > 0)
            {
                bucket ??= new List<ResourceSlotIdentifier>?[FineBucketEntries];
                (bucket[finePage & (FineBucketEntries - 1)] ??= new List<ResourceSlotIdentifier>(4)).Add(imageIdentifier);
                masks ??= new ushort[BucketEntries];
                masks[page & (BucketEntries - 1)] |= bit;
            }
            else if (bucket?[finePage & (FineBucketEntries - 1)] is { } owners && owners.Remove(imageIdentifier) &&
                owners.Count == 0 && masks is not null)
            {
                masks[page & (BucketEntries - 1)] &= (ushort)~bit;
            }
        }

        for (var page = address >> PageBits; page <= (address + size - 1) >> PageBits; page++)
        {
            BumpVersion(page);
        }
    }

    public static bool IsValidPage(ulong page) => page < PageCount;

    public List<ResourceSlotIdentifier>? Find(ulong page) =>
        IsValidPage(page) && _firstLevel[page >> SecondLevelBits] is { } bucket
            ? bucket[page & (BucketEntries - 1)]
            : null;

    /// <summary>
    /// Registers an image over <c>[address, address + size)</c>: counts first, then the
    /// owner lists. Caller holds the image cache lock.
    /// </summary>
    public void Add(ulong address, ulong size, ResourceSlotIdentifier imageIdentifier)
    {
        if (!TryGetPageRange(address, size, out var first, out var lastExclusive))
        {
            throw GpuFatal.Fatal(
                $"The owner-table range is outside the guest address space: address=0x{address:X16} size=0x{size:X}.");
        }

        AdjustLiveOwners(address, size, 1);
        if (IsFine(size))
        {
            AddFine(address, size, imageIdentifier, 1);
            return;
        }

        for (var page = first; page < lastExclusive; page++)
        {
            GetOrCreate(page).Add(imageIdentifier);
            BumpVersion(page);
        }
    }

    /// <summary>
    /// Unregisters an image added over the same range: the owner lists first, then the
    /// counts. Caller holds the image cache lock.
    /// </summary>
    public void Remove(ulong address, ulong size, ResourceSlotIdentifier imageIdentifier)
    {
        if (!TryGetPageRange(address, size, out var first, out var lastExclusive))
        {
            return;
        }

        if (IsFine(size))
        {
            AddFine(address, size, imageIdentifier, -1);
        }
        else
        {
            for (var page = first; page < lastExclusive; page++)
            {
                Find(page)?.Remove(imageIdentifier);
                BumpVersion(page);
            }
        }

        AdjustLiveOwners(address, size, -1);
    }

    private void BumpVersion(ulong page)
    {
        ref var bucket = ref _versions[page >> SecondLevelBits];
        bucket ??= new long[BucketEntries];
        Volatile.Write(ref bucket[page & (BucketEntries - 1)], bucket[page & (BucketEntries - 1)] + 1);
    }

    /// <summary>
    /// The sum of the owner-list change counts over the pages of the range: equal sums mean no
    /// image was added to or removed from any of them in between. Changes happen under the image
    /// cache lock; a reader without it may see an older sum, never a newer one paired with older lists.
    /// </summary>
    public long VersionSum(ulong address, ulong size)
    {
        if (!TryGetPageRange(address, size, out var first, out var lastExclusive))
        {
            return 0;
        }

        long sum = 0;
        for (var page = first; page < lastExclusive; page++)
        {
            if (Volatile.Read(ref _versions[page >> SecondLevelBits]) is { } bucket)
            {
                sum += Volatile.Read(ref bucket[page & (BucketEntries - 1)]);
            }
        }

        return sum;
    }

    private List<ResourceSlotIdentifier> GetOrCreate(ulong page)
    {
        ref var bucket = ref _firstLevel[page >> SecondLevelBits];
        bucket ??= new List<ResourceSlotIdentifier>?[BucketEntries];
        return bucket[page & (BucketEntries - 1)] ??= new List<ResourceSlotIdentifier>(4);
    }

    // Caller holds the image cache lock, so the counts have one writer at a time.
    private void AdjustLiveOwners(ulong address, ulong size, int delta)
    {
        var first = address >> CountPageBits;
        var last = (address + size - 1) >> CountPageBits;
        for (var page = first; page <= last; page++)
        {
            ref var leaf = ref _liveOwners[page >> CountLeafBits];
            if (leaf is null)
            {
                if (delta < 0)
                {
                    throw GpuFatal.Fatal($"An image left a page it was never counted on: page=0x{page:X}.");
                }

                Volatile.Write(ref leaf, new int[1 << CountLeafBits]);
            }

            if (Interlocked.Add(ref leaf[page & CountLeafMask], delta) < 0)
            {
                throw GpuFatal.Fatal($"An image left a page it was never counted on: page=0x{page:X}.");
            }
        }
    }

    /// <summary>
    /// Whether a live image covers any 64 KiB page of the range. Lock free; see the
    /// class comment for why a false answer is as good as a locked lookup.
    /// </summary>
    public bool MayHaveOwners(ulong address, ulong size)
    {
        if (!TryGetPageRange(address, size, out _, out _))
        {
            return false;
        }

        var page = address >> CountPageBits;
        var last = (address + size - 1) >> CountPageBits;
        while (page <= last)
        {
            var leaf = Volatile.Read(ref _liveOwners[page >> CountLeafBits]);
            if (leaf is null)
            {
                // No image has ever been counted anywhere in this leaf.
                page = (page | CountLeafMask) + 1;
                continue;
            }

            var leafLast = Math.Min(last, page | CountLeafMask);
            for (; page <= leafLast; page++)
            {
                if (Volatile.Read(ref leaf[page & CountLeafMask]) != 0)
                {
                    return true;
                }
            }
        }

        return false;
    }

    public static bool TryGetPageRange(ulong address, ulong size, out ulong first, out ulong lastExclusive)
    {
        first = 0;
        lastExclusive = 0;
        if (size == 0 || address >= AddressSpaceSize || size > AddressSpaceSize - address)
        {
            return false;
        }

        first = address >> PageBits;
        lastExclusive = ((address + size - 1) >> PageBits) + 1;
        return true;
    }
}
