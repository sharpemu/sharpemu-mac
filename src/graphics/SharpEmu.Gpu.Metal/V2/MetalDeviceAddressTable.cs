// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.ShaderCompiler.Metal;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>Where the device-address table finds the window bytes of a guest run.</summary>
internal interface IMetalGranuleRuns
{
    /// <summary>
    /// The longest run from <paramref name="gpuVa"/>, in whole pages and at most
    /// <paramref name="size"/>, that is contiguous in the backing and inside one granule,
    /// with the backing offset of its first byte; false when its first page is not backed.
    /// </summary>
    bool TryResolveGranuleRun(ulong gpuVa, ulong size, ulong page, out ulong backingOffset, out ulong length, out int granule);

    /// <summary>Changes whenever the windows are cut differently.</summary>
    long PartitionVersion { get; }

    /// <summary>
    /// The window holding a backing offset: its id, the offset it ends at, and the GPU
    /// address of the offset in it.
    /// </summary>
    (int Id, ulong End, ulong GpuAddress) WindowAt(ulong backingOffset);

    /// <summary>The chunk size of the trap map (single-copy chunks); 0 when there is none.</summary>
    ulong TrapChunkBytes => 0;

    /// <summary>How many chunks the trap map has.</summary>
    int TrapChunkCount => 0;

    /// <summary>Changes whenever the trap map does.</summary>
    long TrapVersion => 0;

    /// <summary>
    /// Copies the trap map, one byte per backing chunk (<see cref="TrapChunkCount"/> of them):
    /// nonzero for a chunk whose bytes the windows do not hold - a single-copy hole, or one the
    /// shadow mode would have made - so a device-address access to it is reported. Returns how
    /// many chunks are set.
    /// </summary>
    int CopyTrapChunks(Span<byte> chunks) => 0;
}

/// <summary>
/// The sorted range table a device-address program binary-searches
/// (sharpemu_resolve_device_address in resource_prelude.msl): one entry per run of a mapped
/// guest span that is contiguous in the backing and lies in one granule and one window,
/// giving the GPU address its first page lives at. The shader computes base + (page -
/// firstPage) * pageSize + pageOffset, which is exact because the run is contiguous in the
/// backing as well as in guest space.
///
/// <para>
/// The mapped spans are resolved into backing runs cut at granule edges when the spans
/// change (<see cref="Rebuild"/>), and those are cut again at window edges into the entries
/// whenever the windows are cut differently (<see cref="Recut"/>), which needs no lookup of
/// the guest mappings. An entry in one window is named by naming that window, and an entry
/// in one granule is flagged and learned by the reach set (MetalRenderHost.DeviceAddressReach.cs)
/// one granule at a time. The cuts cost one entry per granule or window edge a run crosses:
/// Demon's Souls in-world maps ~11.2 GiB in ~1750 spans, which made ~2550 entries over 1 GiB
/// windows and makes ~2900 over 32 MiB granules (one more per 32 MiB of span; a window edge
/// moved off a granule edge adds at most one more), against the 16384 the shader's search
/// covers. The per-invocation range cache still hits for every access that stays inside one
/// run, which a granule of 32 MiB leaves almost all of.
/// </para>
///
/// <para>
/// An entry's page count carries the reach flag in its upper half: bit 63 marks a run in a
/// window the submission did not name and bits 32..62 its granule, so the shader can report
/// it; bits 0..31 are the page count (a run is at most one granule).
/// </para>
///
/// <para>
/// Single-copy trap (artifacts/memopt/single-copy-device-address.md, option (a)): runs are
/// also cut at the edges of the chunks the trap map sets (<see cref="IMetalGranuleRuns.CopyTrapChunks"/>),
/// and an entry over such chunks has bit 62 set and its first backing page in bits 32..61
/// instead: the shader counts every access to it and reports the page
/// (MetalDeviceAddressReach). A trap entry is never flagged unnamed: its access does not land
/// (or, in the shadow mode, lands without learning a granule).
/// </para>
/// </summary>
internal sealed class MetalDeviceAddressTable
{
    internal const int EntryBytes = 24;
    internal const ulong UnnamedFlag = 1UL << 63;
    internal const ulong TrapFlag = 1UL << 62;
    internal const int GranuleShift = 32;
    internal const ulong PageCountMask = 0xFFFF_FFFFUL;
    internal const ulong MaxGranule = 0x3FFF_FFFFUL;
    internal const ulong MaxTrapPage = 0x3FFF_FFFFUL;

    private readonly List<(ulong FirstPage, ulong PageCount, ulong Offset, int Granule)> _runs = new();
    private readonly List<(ulong FirstPage, ulong PageCount, ulong Base)> _entries = new();
    private int[] _entryGranules = [];
    private int[] _entryWindows = [];

    // The first backing page of each trap entry, -1 for the others.
    private long[] _entryTrapPages = [];
    private byte[] _trapChunks = [];
    private uint[] _index = [];
    private int _maxEntries = (int)Gen5MslTranslator.MaxAddressRangeCount;

    /// <summary>The entries in table order.</summary>
    public IReadOnlyList<(ulong FirstPage, ulong PageCount, ulong Base)> Entries => _entries;

    /// <summary>The granule of each entry, in table order.</summary>
    public ReadOnlySpan<int> EntryGranules => _entryGranules.AsSpan(0, _entries.Count);

    /// <summary>The window of each entry, in table order.</summary>
    public ReadOnlySpan<int> EntryWindows => _entryWindows.AsSpan(0, _entries.Count);

    /// <summary>Every granule some entry lies in.</summary>
    public MetalWindowSet Granules { get; } = new();

    /// <summary>Every window some entry lies in.</summary>
    public MetalWindowSet Windows { get; } = new();

    /// <summary>The window cut the entries were made for.</summary>
    public long PartitionVersion { get; private set; } = long.MinValue;

    /// <summary>The trap map the entries were made for.</summary>
    public long TrapVersion { get; private set; } = long.MinValue;

    /// <summary>Entries the trap flags, and the bytes they cover.</summary>
    public int TrapEntryCount { get; private set; }

    public ulong TrapBytes { get; private set; }

    /// <summary>Trap pieces left unflagged: a run whose backing is not page-aligned, or a table with no room for the cuts.</summary>
    public int TrapUnflagged { get; private set; }

    /// <summary>Whether the last cut dropped every trap flag because the cuts did not fit the table.</summary>
    public bool TrapOverflowed { get; private set; }

    public int SpanCount { get; private set; }

    /// <summary>Bytes the entries cover.</summary>
    public ulong SpanBytes { get; private set; }

    public int Count => _entries.Count;

    /// <summary>Bytes <see cref="Write"/> needs: the entries, then the granule index.</summary>
    public int ByteSize => Math.Max((_entries.Count * EntryBytes) + (_index.Length * sizeof(uint)), EntryBytes);

    /// <summary>
    /// Rebuilds the entries from the mapped guest spans. Pages with no window are left out,
    /// never copied, because a shader store through a copied page would not reach guest
    /// memory.
    /// </summary>
    public void Rebuild(IMetalGranuleRuns memory, Action<Action<ulong, ulong>> forEachSpan, int maxEntries)
    {
        _runs.Clear();
        Granules.Clear();
        SpanBytes = 0;
        _maxEntries = maxEntries;
        var spanCount = 0;
        forEachSpan((address, size) =>
        {
            spanCount++;
            var page = DeviceAddressPaging.PageSize;
            var start = (address + page - 1) & ~(page - 1);
            var end = (address + size) & ~(page - 1);
            while (start < end)
            {
                if (!memory.TryResolveGranuleRun(start, end - start, page, out var offset, out var length, out var granule))
                {
                    start += page;
                    continue;
                }

                if (_runs.Count >= maxEntries)
                {
                    throw Full(maxEntries, spanCount, start >> DeviceAddressPaging.PageBits);
                }

                _runs.Add((start >> DeviceAddressPaging.PageBits, length >> DeviceAddressPaging.PageBits, offset, granule));
                Granules.Add(granule);
                SpanBytes += length;
                start += length;
            }
        });

        SpanCount = spanCount;
        _runs.Sort(static (left, right) => left.FirstPage.CompareTo(right.FirstPage));
        Recut(memory);
    }

    /// <summary>
    /// Cuts the runs into the entries at the window edges as they are now, with the GPU
    /// address of each in its window.
    /// </summary>
    public void Recut(IMetalGranuleRuns memory)
    {
        PartitionVersion = memory.PartitionVersion;
        TrapVersion = memory.TrapVersion;
        var chunkBytes = memory.TrapChunkBytes;
        var trapped = 0;
        if (chunkBytes != 0)
        {
            if (_trapChunks.Length != memory.TrapChunkCount)
            {
                _trapChunks = new byte[memory.TrapChunkCount];
            }

            trapped = memory.CopyTrapChunks(_trapChunks);
        }

        TrapOverflowed = false;
        if (!TryCut(memory, trapped == 0 ? 0 : chunkBytes, throwWhenFull: trapped == 0))
        {
            // The trap cuts do not fit the table: no entry is flagged (the host gives every
            // chunk back when it sees this, MetalRenderHost.UploadDeviceAddressTable).
            TryCut(memory, 0, throwWhenFull: true);
            TrapOverflowed = true;
            TrapUnflagged = trapped;
        }

        _index = MetalDeviceAddressIndex.Build(_entries);
    }

    // The entries for the runs, cut at window edges and, given a chunk size, where the trap
    // map changes. False when the table has no room and the caller asked not to throw.
    private bool TryCut(IMetalGranuleRuns memory, ulong chunkBytes, bool throwWhenFull)
    {
        _entries.Clear();
        Windows.Clear();
        TrapEntryCount = 0;
        TrapBytes = 0;
        TrapUnflagged = 0;
        var pageBits = DeviceAddressPaging.PageBits;
        foreach (var (firstPage, pageCount, offset, granule) in _runs)
        {
            var page = firstPage;
            var left = pageCount;
            var at = offset;
            while (left != 0)
            {
                var (window, end, gpuAddress) = memory.WindowAt(at);
                // Window edges are page-aligned; a run that is not keeps its pages whole.
                var pages = Math.Max(1UL, Math.Min(left, (end - at) >> pageBits));
                var pieceEnd = at + (pages << pageBits);
                if (chunkBytes == 0 || !AnyTrapped(at, pieceEnd, chunkBytes))
                {
                    if (!TryAdd(page, pages, gpuAddress, granule, window, -1, throwWhenFull))
                    {
                        return false;
                    }
                }
                else if (at % DeviceAddressPaging.PageSize != 0)
                {
                    // Backing that is not page-aligned cannot put a chunk edge between whole
                    // pages: the piece stays unflagged.
                    TrapUnflagged++;
                    if (!TryAdd(page, pages, gpuAddress, granule, window, -1, throwWhenFull))
                    {
                        return false;
                    }
                }
                else
                {
                    for (var cursor = at; cursor < pieceEnd;)
                    {
                        var chunk = cursor / chunkBytes;
                        var trap = IsTrapped(chunk);
                        var next = Math.Min(pieceEnd, (chunk + 1) * chunkBytes);
                        while (next < pieceEnd && IsTrapped(next / chunkBytes) == trap)
                        {
                            next = Math.Min(pieceEnd, next + chunkBytes);
                        }

                        var trapPage = trap ? (long)(cursor >> pageBits) : -1;
                        if (trapPage > (long)MaxTrapPage)
                        {
                            TrapUnflagged++;
                            trapPage = -1;
                        }

                        var subPage = page + ((cursor - at) >> pageBits);
                        if (!TryAdd(subPage, (next - cursor) >> pageBits, gpuAddress + (cursor - at), granule, window, trapPage, throwWhenFull))
                        {
                            return false;
                        }

                        cursor = next;
                    }
                }

                page += pages;
                at += pages << pageBits;
                left -= pages;
            }
        }

        return true;
    }

    private bool IsTrapped(ulong chunk) => chunk < (ulong)_trapChunks.Length && _trapChunks[chunk] != 0;

    // Whether a chunk overlapping backing bytes [start, end) is trapped.
    private bool AnyTrapped(ulong start, ulong end, ulong chunkBytes)
    {
        var first = start / chunkBytes;
        if (first >= (ulong)_trapChunks.Length)
        {
            return false;
        }

        var last = Math.Min((end - 1) / chunkBytes, (ulong)_trapChunks.Length - 1);
        return _trapChunks.AsSpan((int)first, (int)(last - first + 1)).IndexOfAnyExcept((byte)0) >= 0;
    }

    private bool TryAdd(ulong page, ulong pages, ulong gpuAddress, int granule, int window, long trapPage, bool throwWhenFull)
    {
        if (_entries.Count >= _maxEntries)
        {
            if (throwWhenFull)
            {
                throw Full(_maxEntries, SpanCount, page);
            }

            return false;
        }

        if (_entryGranules.Length <= _entries.Count)
        {
            var length = Math.Max(_entries.Count + 1, _entryGranules.Length * 2);
            Array.Resize(ref _entryGranules, length);
            Array.Resize(ref _entryWindows, length);
            Array.Resize(ref _entryTrapPages, length);
        }

        _entryGranules[_entries.Count] = granule;
        _entryWindows[_entries.Count] = window;
        _entryTrapPages[_entries.Count] = trapPage;
        _entries.Add((page, pages, gpuAddress));
        Windows.Add(window);
        if (trapPage >= 0)
        {
            TrapEntryCount++;
            TrapBytes += pages << DeviceAddressPaging.PageBits;
        }

        return true;
    }

    /// <summary>Adds the windows of the entries that lie in <paramref name="granules"/> to <paramref name="windows"/>.</summary>
    public void AddWindowsOfGranules(MetalWindowSet granules, MetalWindowSet windows)
    {
        for (var index = 0; index < _entries.Count; index++)
        {
            if (granules.Contains(_entryGranules[index]))
            {
                windows.Add(_entryWindows[index]);
            }
        }
    }

    /// <summary>Adds all windows intersecting an exact captured guest byte range.</summary>
    public bool AddWindowsOfRange(ulong address, ulong size, MetalWindowSet windows)
    {
        if (size == 0 || size > ulong.MaxValue - address || _entries.Count == 0) return false;
        var first = address >> DeviceAddressPaging.PageBits;
        var end = ((address + size - 1) >> DeviceAddressPaging.PageBits) + 1;
        var low = 0;
        var high = _entries.Count;
        while (low < high)
        {
            var middle = low + ((high - low) >> 1);
            if (_entries[middle].FirstPage <= first) low = middle + 1;
            else high = middle;
        }
        var added = false;
        for (var index = Math.Max(low - 1, 0); index < _entries.Count && _entries[index].FirstPage < end; index++)
            if (_entries[index].FirstPage + _entries[index].PageCount > first)
                added |= windows.Add(_entryWindows[index]);
        return added;
    }

    /// <summary>
    /// Writes the table and its granule index. An entry in a window outside
    /// <paramref name="namedWindows"/> is flagged with its granule; null flags none.
    /// </summary>
    public void Write(Span<byte> span, MetalWindowSet? namedWindows)
    {
        span.Clear();
        for (var index = 0; index < _entries.Count; index++)
        {
            var (firstPage, pageCount, address) = _entries[index];
            var entry = span[(index * EntryBytes)..];
            var named = namedWindows is null || namedWindows.Contains(_entryWindows[index]);
            var trapPage = _entryTrapPages[index];
            BinaryPrimitives.WriteUInt64LittleEndian(entry, firstPage);
            BinaryPrimitives.WriteUInt64LittleEndian(entry[8..],
                trapPage >= 0 ? Trapped(pageCount, (ulong)trapPage) : named ? pageCount : Flagged(pageCount, _entryGranules[index]));
            BinaryPrimitives.WriteUInt64LittleEndian(entry[16..], address);
        }

        System.Runtime.InteropServices.MemoryMarshal.AsBytes(_index.AsSpan()).CopyTo(span[(_entries.Count * EntryBytes)..]);
    }

    /// <summary>The page count an entry carries: flagged with its granule when that is not named.</summary>
    internal static ulong FlaggedPageCount(ulong pageCount, int granule, MetalWindowSet? named) =>
        named is null || granule < 0 || named.Contains(granule) ? pageCount : Flagged(pageCount, granule);

    private static ulong Flagged(ulong pageCount, int granule) =>
        (pageCount & PageCountMask) | UnnamedFlag | (((ulong)granule & MaxGranule) << GranuleShift);

    /// <summary>The page count of a trap entry whose first page is backing page <paramref name="backingPage"/>.</summary>
    internal static ulong Trapped(ulong pageCount, ulong backingPage) =>
        (pageCount & PageCountMask) | TrapFlag | ((backingPage & MaxTrapPage) << GranuleShift);

    /// <summary>The first backing page of each entry the trap flags, -1 for the others, in table order.</summary>
    public ReadOnlySpan<long> EntryTrapPages => _entryTrapPages.AsSpan(0, _entries.Count);

    /// <summary>
    /// Adds one run to a table. A span the table has no room for resolves to null in every
    /// shader, so its loads read zero and its stores are dropped without a word; a full
    /// table therefore ends the run instead of truncating.
    /// </summary>
    internal static void AddEntry(
        List<(ulong FirstPage, ulong PageCount, ulong Base)> entries,
        (ulong FirstPage, ulong PageCount, ulong Base) entry,
        int spanCount,
        int maxEntries = (int)Gen5MslTranslator.MaxAddressRangeCount)
    {
        if (entries.Count >= maxEntries)
        {
            throw Full(maxEntries, spanCount, entry.FirstPage);
        }

        entries.Add(entry);
    }

    private static Exception Full(int maxEntries, int spanCount, ulong firstPage) =>
        GpuFatal.Fatal(
            $"The device-address table is full and later guest spans are not addressable: " +
            $"limit={maxEntries} spans_so_far={spanCount} page=0x{firstPage:X}.");
}
