// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Scheduling;

namespace SharpEmu.Libs.Gpu.GpuCommands;

// The queued stores, fills and copies by guest page (upstream's "Index queued stores by page for overlay
// reads" and "Read guest memory without the store lock when the page has no pending store", ported from the
// Vulkan presenter's command thread). Every read the stream makes asks whether a queued item writes its
// bytes (ReadOverlay, TryReadQueuedFill, WaitForPendingRange); before the index each such read took the
// write lock the worker also takes for every item it retires, and walked every queued item. Now a read
// whose pages no queued item touches answers without the lock, and a read inside one page looks only at the
// items on that page.
//
// What is indexed: a store (PostWrite of bytes) and a range item a stream read waits for or is answered by
// (PostRange, PostFill, PostWrite of known bytes, PostQwords). Labels are not: no stream read waits for one.
// An item over more than MaxIndexedPages pages is listed as wide; while one is queued every read takes the
// locked walk, as before the index.
//
// Ordering: items enter the index under _writeLock when they are queued and leave it under the same lock
// once the worker has run them, so the page lists keep the queue's order (PostOrder). A read on the
// stream's own thread sees every item the stream queued before it; a stale count (an item the worker is
// retiring) only sends the read to the locked walk, which answers as before.
public sealed partial class DeferredCommandThread
{
    private const int PendingPageShift = 12;
    private const ulong MaxIndexedPages = 64;
    private const int PendingPageFilterBits = 14;
    private const int MaxSparePageLists = 4096;

    // Indexed items per hashed page; changed under _writeLock, read without it. A collision only sends a
    // read to the locked walk.
    private readonly int[] _pendingPageFilter = new int[1 << PendingPageFilterBits];
    private readonly Dictionary<ulong, List<Item>> _pendingByPage = new();
    private readonly Stack<List<Item>> _sparePageLists = new();
    private readonly List<Item> _widePending = new();
    private volatile int _widePendingCount;
    private long _postOrder;
    private long _lockedPendingReads;

    /// <summary>Stream reads that had to take the write lock to find the queued items over their bytes.</summary>
    internal long LockedPendingReads => Interlocked.Read(ref _lockedPendingReads);

    private static int PendingPageSlot(ulong page) =>
        (int)((page * 0x9E3779B97F4A7C15UL) >> (64 - PendingPageFilterBits));

    private static ulong IndexedSize(Item item) => item.Kind == Kind.Write ? (ulong)item.Data!.Length : item.RangeSize;

    private static bool IsIndexed(Item item) => item.Kind == Kind.Write ? item.Data!.Length != 0 : item.RangeSize != 0 && !item.IsLabel;

    // Under _writeLock.
    private void IndexPending(Item item, bool add)
    {
        if (!IsIndexed(item))
        {
            return;
        }

        if (add)
        {
            item.PostOrder = ++_postOrder;
        }

        var first = item.Address >> PendingPageShift;
        var last = (item.Address + IndexedSize(item) - 1) >> PendingPageShift;
        if (last - first >= MaxIndexedPages)
        {
            if (add)
            {
                _widePending.Add(item);
            }
            else
            {
                _widePending.Remove(item);
            }

            _widePendingCount = _widePending.Count;
            return;
        }

        for (var page = first; page <= last; page++)
        {
            ref var count = ref _pendingPageFilter[PendingPageSlot(page)];
            if (add)
            {
                if (!_pendingByPage.TryGetValue(page, out var queued))
                {
                    queued = _sparePageLists.Count != 0 ? _sparePageLists.Pop() : new List<Item>();
                    _pendingByPage.Add(page, queued);
                }

                queued.Add(item);
                Volatile.Write(ref count, count + 1);
                continue;
            }

            if (_pendingByPage.TryGetValue(page, out var listed) && listed.Remove(item))
            {
                Volatile.Write(ref count, count - 1);
                if (listed.Count == 0)
                {
                    _pendingByPage.Remove(page);
                    if (_sparePageLists.Count < MaxSparePageLists)
                    {
                        _sparePageLists.Push(listed);
                    }
                }
            }
            else
            {
                throw GpuFatal.Fatal("The command stream's queued item index lost an item.");
            }
        }
    }

    // Without the lock: whether a queued store, fill or copy may write a byte of the range.
    private bool MayHavePendingItem(ulong address, ulong size)
    {
        if (size == 0)
        {
            return false;
        }

        if (_widePendingCount != 0)
        {
            return true;
        }

        var first = address >> PendingPageShift;
        var last = (address + size - 1) >> PendingPageShift;
        if (last - first >= MaxIndexedPages)
        {
            return true;
        }

        for (var page = first; page <= last; page++)
        {
            if (Volatile.Read(ref _pendingPageFilter[PendingPageSlot(page)]) != 0)
            {
                return true;
            }
        }

        return false;
    }

    // Under _writeLock: the indexed items listed on the read's single page, or null when the read
    // spans pages (the caller walks every queued item then). Wide items are not in the list.
    private List<Item>? PendingOnPage(ulong address, ulong size)
    {
        _lockedPendingReads++;
        var page = address >> PendingPageShift;
        if (page != (address + size - 1) >> PendingPageShift)
        {
            return null;
        }

        return _pendingByPage.TryGetValue(page, out var queued) ? queued : EmptyItems;
    }

    private static readonly List<Item> EmptyItems = new();
}
