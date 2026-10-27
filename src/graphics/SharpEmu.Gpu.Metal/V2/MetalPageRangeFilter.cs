// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Metal.V2;

// A bounded negative filter, never an overlap proof. Hash collisions and large ranges
// fall back to the caller's exact range list. The owner serializes Add/MayOverlap/Clear.
internal sealed class MetalPageRangeFilter
{
    private const int PageShift = 12;
    private const int MaxPages = 64;
    private readonly ulong[] _bits = new ulong[64];
    private bool _unbounded;

    public void Add(ulong address, ulong size, ulong space = 0)
    {
        if (size == 0 || _unbounded) return;
        if (!TryPages(address, size, out var first, out var last))
        {
            _unbounded = true;
            return;
        }

        for (var page = first; page <= last; page++)
        {
            var bit = Bit(page, space);
            _bits[bit >> 6] |= 1UL << (bit & 63);
        }
    }

    public bool MayOverlap(ulong address, ulong size, ulong space = 0)
    {
        if (size == 0) return false;
        if (_unbounded || !TryPages(address, size, out var first, out var last)) return true;
        for (var page = first; page <= last; page++)
        {
            var bit = Bit(page, space);
            if ((_bits[bit >> 6] & (1UL << (bit & 63))) != 0) return true;
        }

        return false;
    }

    public void Clear()
    {
        Array.Clear(_bits);
        _unbounded = false;
    }

    private static bool TryPages(ulong address, ulong size, out ulong first, out ulong last)
    {
        first = address >> PageShift;
        last = 0;
        if (size - 1 > ulong.MaxValue - address) return false;
        last = (address + size - 1) >> PageShift;
        return last - first < MaxPages;
    }

    private static int Bit(ulong page, ulong space)
    {
        var hash = unchecked(page ^ (space * 0x9E3779B97F4A7C15UL));
        hash ^= hash >> 33;
        hash = unchecked(hash * 0xFF51AFD7ED558CCDUL);
        hash ^= hash >> 33;
        return (int)(hash & 4095);
    }
}
