// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Metal.V2;

// Registration precedes observer publication; removal follows it. Mutations are
// serialized by the owning cache. Stable sparse leaves allow lock-free rejection
// at 4 KiB rather than sending every neighbour in a 1 MiB owner bucket to a lock.
internal sealed class MetalStorePageCounts
{
    private readonly int[]?[] _leaves = new int[]?[1 << 16];

    public void Add(ulong address, ulong size) => Adjust(address, size, 1);
    public void Remove(ulong address, ulong size) => Adjust(address, size, -1);

    private void Adjust(ulong address, ulong size, int delta)
    {
        if (!MetalFencePageIndex.Valid(address, size)) throw new ArgumentOutOfRangeException(nameof(size));
        var last = (address + size - 1) >> 12;
        for (var page = address >> 12; page <= last; page++)
        {
            ref var slot = ref _leaves[page >> 12];
            var leaf = Volatile.Read(ref slot);
            if (leaf is null)
            {
                if (delta < 0) throw new InvalidOperationException("Removing an unregistered store observer.");
                Volatile.Write(ref slot, leaf = new int[4096]);
            }
            if (Interlocked.Add(ref leaf[page & 4095], delta) < 0)
                throw new InvalidOperationException("Store observer count underflow.");
        }
    }

    public bool MayOverlap(ulong address, ulong size)
    {
        if (!MetalFencePageIndex.Valid(address, size)) return true;
        // Bound common scalar-store work; large writes use the existing range scan.
        if (size > 8192) return true;
        var last = (address + size - 1) >> 12;
        for (var page = address >> 12; page <= last; page++)
        {
            var leaf = Volatile.Read(ref _leaves[page >> 12]);
            if (leaf is not null && Volatile.Read(ref leaf[page & 4095]) != 0) return true;
        }
        return false;
    }
}
