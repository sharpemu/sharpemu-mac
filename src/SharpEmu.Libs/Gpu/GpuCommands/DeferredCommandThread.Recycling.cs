// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;

namespace SharpEmu.Libs.Gpu.GpuCommands;

public sealed partial class DeferredCommandThread
{
    private static readonly bool RecycleAsyncItemsEnabled =
        Environment.GetEnvironmentVariable("SHARPEMU_COMMAND_ITEM_REUSE") != "0";

    // A return channel, opposite to the work queue: only the render thread publishes items;
    // only the interpreter takes them. The existing producer/consumer ownership also applies
    // during Stop, whose runItems callback runs on the consumer thread.
    private readonly Item?[]? _recycledItems;
    private RecycleCursors _recycleCursors;

    // Separate the two writers' counters so returning an item does not invalidate the producer's
    // write counter's cache line. The extra spacing also tolerates a non-aligned containing object.
    [StructLayout(LayoutKind.Explicit, Size = 256)]
    private struct RecycleCursors
    {
        [FieldOffset(0)] public long Read;
        [FieldOffset(8)] public long ReadAvailable;
        [FieldOffset(128)] public long Write;
        [FieldOffset(136)] public long WriteLimit;
    }

    private Item RentAsyncItem(Kind kind)
    {
        var slots = _recycledItems;
        if (slots is not null)
        {
            var read = _recycleCursors.Read;
            // Refresh the remote cursor only after consuming the published batch. Reading it
            // on every item would repeatedly transfer the consumer's cache line between cores.
            if (read == _recycleCursors.ReadAvailable)
                _recycleCursors.ReadAvailable = Volatile.Read(ref _recycleCursors.Write);
            if (read != _recycleCursors.ReadAvailable)
            {
                var index = (int)(read & (MaxOutstanding - 1));
                var item = slots[index]!;
                slots[index] = null;
                // Publish the empty slot after taking its item; the consumer may then refill it.
                Volatile.Write(ref _recycleCursors.Read, read + 1);
                item.Kind = kind;
                return item;
            }
        }

        return new Item { Kind = kind };
    }

    private void RecycleAsyncItem(Item item)
    {
        var slots = _recycledItems;
        if (slots is null)
            return;

        var write = _recycleCursors.Write;
        if (write == _recycleCursors.WriteLimit)
        {
            _recycleCursors.WriteLimit = Volatile.Read(ref _recycleCursors.Read) + slots.Length;
            if (write == _recycleCursors.WriteLimit)
                return;
        }

        // Do not retain a draw's banks, closures, or write buffers while the queue is idle.
        item.Work = null;
        item.Action = null;
        item.Data = null;
        item.Failure = null;
        item.IsDraw = false;
        item.IsDispatch = false;
        item.OwnsLastJob = false;
        item.Ungated = false;
        item.IsFill = false;
        item.IsLabel = false;
        item.FillValue = 0;
        item.RangeBytes = null;
        item.IsQwords = false;
        item.QwordValue = 0;
        item.QwordStride = 0;
        item.Address = 0;
        item.RangeSize = 0;
        item.JobsBefore = 0;
        item.Done = false;
        slots[(int)(write & (MaxOutstanding - 1))] = item;
        Volatile.Write(ref _recycleCursors.Write, write + 1);
    }
}
