// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.CompilerServices;
using SharpEmu.HLE.GpuMemory;

namespace SharpEmu.GpuSplit;

/// <summary>
/// B's pending-writeback summary (<see cref="SplitSharedState"/>): a claim count per 4 KiB page,
/// with the page's shared bit set while the count is above zero.
/// </summary>
/// <remarks>
/// Counts live in 1 GiB chunks made on first use (1 MiB each), so a claim costs one add per page;
/// a render target of a few thousand pages is written back many times a frame. A page whose count
/// drops to zero keeps its bit until <see cref="Publish"/>, which clears it only if the count is
/// still zero then. Everything happens under one lock, which nests inside the image cache's.
/// SHARPEMU_SPLIT_WRITEBACK_SUMMARY=0 keeps no summary: A then calls for every watched write, as before.
/// </remarks>
internal sealed class PendingWritebackPages : IPendingWritebackPages
{
    public static readonly bool Enabled =
        Environment.GetEnvironmentVariable("SHARPEMU_SPLIT_WRITEBACK_SUMMARY") != "0";

    private const int PageShift = SplitSharedState.PageShift;
    private const int ChunkPageShift = 18;
    private const ulong ChunkPageMask = (1UL << ChunkPageShift) - 1;

    private readonly SplitSharedState _state;
    private readonly object _gate = new();
    private readonly int[]?[] _chunks = new int[(TrackerLayout.SpaceBytes >> PageShift) >> ChunkPageShift][];
    // Page runs whose count reached zero since the last publish (first page, page count).
    private readonly List<(ulong First, ulong Count)> _released = [];
    private int _releasedRuns;

    public PendingWritebackPages(SplitSharedState state) => _state = state;

    public void Activate() => _state.SetPendingLive();

    public void Add(ulong address, ulong size)
    {
        if (!TryPages(address, size, out var first, out var count))
        {
            return;
        }

        lock (_gate)
        {
            var word = first >> 6;
            ulong mask = 0;
            for (var page = first; page < first + count; page++)
            {
                if ((page >> 6) != word)
                {
                    if (mask != 0) _state.SetPendingBits(word, mask);
                    word = page >> 6;
                    mask = 0;
                }

                ref var slot = ref Count(page, create: true);
                if (slot++ == 0)
                {
                    mask |= 1UL << (int)(page & 63);
                }
            }

            if (mask != 0) _state.SetPendingBits(word, mask);
        }
    }

    public void Remove(ulong address, ulong size)
    {
        if (!TryPages(address, size, out var first, out var count))
        {
            return;
        }

        lock (_gate)
        {
            var reachedZero = false;
            for (var page = first; page < first + count; page++)
            {
                ref var slot = ref Count(page, create: false);
                if (Unsafe.IsNullRef(ref slot) || slot == 0)
                {
                    continue;
                }

                if (--slot == 0)
                {
                    reachedZero = true;
                }
            }

            if (reachedZero)
            {
                _released.Add((first, count));
                Volatile.Write(ref _releasedRuns, _released.Count);
            }
        }
    }

    public void Publish()
    {
        if (Volatile.Read(ref _releasedRuns) == 0)
        {
            return;
        }

        lock (_gate)
        {
            foreach (var (first, count) in _released)
            {
                var word = first >> 6;
                ulong mask = 0;
                for (var page = first; page < first + count; page++)
                {
                    if ((page >> 6) != word)
                    {
                        if (mask != 0) _state.ClearPendingBits(word, mask);
                        word = page >> 6;
                        mask = 0;
                    }

                    ref var slot = ref Count(page, create: false);
                    if (Unsafe.IsNullRef(ref slot) || slot == 0)
                    {
                        mask |= 1UL << (int)(page & 63);
                    }
                }

                if (mask != 0) _state.ClearPendingBits(word, mask);
            }

            _released.Clear();
            Volatile.Write(ref _releasedRuns, 0);
        }
    }

    /// <summary>The claims on the page holding <paramref name="address"/> (tests).</summary>
    internal int ClaimsAt(ulong address)
    {
        lock (_gate)
        {
            ref var slot = ref Count(address >> PageShift, create: false);
            return Unsafe.IsNullRef(ref slot) ? 0 : slot;
        }
    }

    private ref int Count(ulong page, bool create)
    {
        var index = page >> ChunkPageShift;
        var chunk = _chunks[index];
        if (chunk is null)
        {
            if (!create)
            {
                return ref Unsafe.NullRef<int>();
            }

            chunk = _chunks[index] = new int[1 << ChunkPageShift];
        }

        return ref chunk[page & ChunkPageMask];
    }

    private static bool TryPages(ulong address, ulong size, out ulong first, out ulong count)
    {
        first = count = 0;
        if (size == 0 || address >= TrackerLayout.SpaceBytes)
        {
            return false;
        }

        var end = size > TrackerLayout.SpaceBytes - address ? TrackerLayout.SpaceBytes : address + size;
        first = address >> PageShift;
        count = ((end - 1) >> PageShift) - first + 1;
        return true;
    }
}
