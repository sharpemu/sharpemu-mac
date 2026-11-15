// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// The cut of the guest backing into windows: disjoint, page-aligned intervals that together
/// cover it, each one no-copy buffer once something uses it (MetalGuestMemory). The windows
/// start out one per granule and the cuts between them move so that every range a binding
/// uses lies inside one window.
///
/// <para>
/// Disjoint because Metal does not keep two buffers over the same pages coherent inside a
/// command buffer. Measured on an M4 (MetalGuestWindowGranuleTests): a blit or a kernel that
/// writes bytes through one no-copy buffer and a later blit or kernel in the same command
/// buffer that reads them through another buffer over the same pages reads the old bytes -
/// with a fence between them, with a buffer-scope memory barrier between them in one
/// encoder, and in every blit/compute combination - while the same read through the same
/// buffer, or through the other buffer from the next command buffer, sees the write. The GPU
/// caches by virtual address, and a buffer is a virtual address range. So a guest byte has to
/// have one GPU address for the life of a command buffer, which overlapping windows (a seam
/// shared by two windows, or a span window over granules that also have their own windows)
/// cannot give.
/// </para>
///
/// <para>
/// A range that crosses a cut moves the cut rather than merging the windows it touches.
/// Merging whole windows chains: in a heap packed with allocations, one straddles each
/// granule edge, and every two of them share the granule between their edges, so merging
/// would grow one window across the whole heap. Moving the cut to the end of the range
/// (rightwards; leftwards to its start, or around it on both sides, when that would make
/// the window larger than the cap) changes only the two windows beside it, and only ranges
/// that really overlap each other end up in one window.
/// </para>
///
/// <para>
/// A move replaces the windows it changes with new buffers at new GPU addresses, so it may
/// not touch a window the command buffer being recorded has already used (the caller's
/// pinned set): <see cref="Place"/> says so, the caller stages that range for this command
/// buffer and moves the cut at the start of the next one. Windows nothing has used for a
/// while are cut back to one per granule (<see cref="Normalize"/>), so the layout follows
/// what is in use rather than everything that ever was.
/// </para>
/// </summary>
internal sealed class MetalWindowPartition
{
    /// <summary>One window: the backing bytes [Start, End).</summary>
    internal sealed class Window
    {
        public Window(int id, ulong start, ulong end)
        {
            Id = id;
            Start = start;
            End = end;
        }

        public int Id { get; }

        public ulong Start { get; }

        public ulong End { get; }

        public ulong Size => End - Start;

        // The last submission that used the window.
        public ulong LastUsedSerial;

        // The owner's buffer over the window, zero until something uses it.
        public nint Buffer;

        public ulong GpuAddress;

        // Set once the window leaves the partition (Replace: a move, a cut back to granules, a renewal); it never
        // comes back. Written under the owner's window lock, read without it by the read-binding memo on the worker,
        // where every placement runs.
        public bool Retired;
    }

    internal enum Placement
    {
        /// <summary>The range lies in one window already.</summary>
        Fits,

        /// <summary>A cut was moved; the returned window is new.</summary>
        Moved,

        /// <summary>A cut has to move, but a window it would replace is pinned.</summary>
        Pinned,

        /// <summary>No window up to the cap can hold the range.</summary>
        TooLarge,
    }

    private readonly List<Window> _windows = new();
    private readonly Stack<int> _freeIds = new();
    private readonly List<Window> _pieces = new();
    private readonly List<ulong> _cuts = new();
    private Window?[] _byId = new Window?[64];
    private int _nextId;
    private readonly ulong _total;
    private readonly ulong _granule;
    private readonly ulong _max;
    private readonly ulong _page;

    public MetalWindowPartition(ulong totalBytes, ulong granuleBytes, ulong maxWindowBytes, ulong pageBytes)
    {
        if (totalBytes == 0 || !System.Numerics.BitOperations.IsPow2(pageBytes) ||
            granuleBytes < pageBytes || granuleBytes % pageBytes != 0 || maxWindowBytes < granuleBytes)
        {
            throw new ArgumentException(
                $"Bad window layout: total=0x{totalBytes:X} granule=0x{granuleBytes:X} max=0x{maxWindowBytes:X} page=0x{pageBytes:X}.");
        }

        _total = totalBytes;
        _granule = granuleBytes;
        _max = maxWindowBytes;
        _page = pageBytes;
        for (var start = 0UL; start < totalBytes; start += granuleBytes)
        {
            _windows.Add(NewWindow(start, Math.Min(start + granuleBytes, totalBytes)));
        }
    }

    /// <summary>Bumped whenever a cut moves, so a table of window addresses knows to re-cut.</summary>
    public long Version { get; private set; }

    /// <summary>The windows in backing order.</summary>
    public IReadOnlyList<Window> Windows => _windows;

    /// <summary>Window ids are below this.</summary>
    public int IdLimit => _nextId;

    public Window? ById(int id) => (uint)id < (uint)_byId.Length ? _byId[id] : null;

    /// <summary>The window holding a backing offset.</summary>
    public Window At(ulong offset) => _windows[IndexOf(offset)];

    private int IndexOf(ulong offset)
    {
        var low = 0;
        var high = _windows.Count - 1;
        while (low < high)
        {
            var middle = (low + high + 1) / 2;
            if (_windows[middle].Start <= offset)
            {
                low = middle;
            }
            else
            {
                high = middle - 1;
            }
        }

        return low;
    }

    /// <summary>
    /// The window that holds the backing bytes [<paramref name="start"/>, <paramref name="end"/>),
    /// moving the cuts inside the range out of it when it crosses one. Windows a move replaces
    /// are appended to <paramref name="retired"/>; the caller releases their buffers once the
    /// GPU is done with them.
    /// </summary>
    public Placement Place(ulong start, ulong end, MetalWindowSet pinned, List<Window> retired, out Window window)
    {
        if (end <= start || end > _total)
        {
            throw new ArgumentOutOfRangeException(nameof(end), $"0x{start:X}..0x{end:X} of 0x{_total:X}");
        }

        var first = IndexOf(start);
        var last = IndexOf(end - 1);
        window = _windows[first];
        if (first == last)
        {
            return Placement.Fits;
        }

        var low = start & ~(_page - 1);
        var high = Math.Min((end + _page - 1) & ~(_page - 1), _total);
        var outerStart = _windows[first].Start;
        var outerEnd = _windows[last].End;
        ulong from;
        ulong to;
        if (high - outerStart <= _max)
        {
            // The cuts inside move right, to the range's end.
            (from, to) = (outerStart, high);
        }
        else if (outerEnd - low <= _max)
        {
            (from, to) = (low, outerEnd);
        }
        else if (high - low <= _max)
        {
            (from, to) = (low, high);
        }
        else
        {
            return Placement.TooLarge;
        }

        for (var index = first; index <= last; index++)
        {
            if (pinned.Contains(_windows[index].Id))
            {
                return Placement.Pinned;
            }
        }

        _pieces.Clear();
        if (from > outerStart)
        {
            _pieces.Add(NewWindow(outerStart, from));
        }

        var placed = NewWindow(from, to);
        _pieces.Add(placed);
        if (to < outerEnd)
        {
            _pieces.Add(NewWindow(to, outerEnd));
        }

        Replace(first, last, retired);
        window = placed;
        return Placement.Moved;
    }

    /// <summary>
    /// Cuts every run of windows that are idle - unused since <paramref name="idleSerial"/>
    /// and not pinned - back to one window per granule, keeping the run's outer edges (a busy
    /// neighbour's ranges may end there). An end piece shorter than half a granule joins its
    /// neighbour in the run. True when anything changed.
    /// </summary>
    public bool Normalize(ulong idleSerial, MetalWindowSet pinned, List<Window> retired)
    {
        var changed = false;
        var index = 0;
        while (index < _windows.Count)
        {
            if (!IsIdle(_windows[index], idleSerial, pinned))
            {
                index++;
                continue;
            }

            var runEnd = index;
            while (runEnd + 1 < _windows.Count && IsIdle(_windows[runEnd + 1], idleSerial, pinned))
            {
                runEnd++;
            }

            if (HomeCut(_windows[index].Start, _windows[runEnd].End, index, runEnd))
            {
                Replace(index, runEnd, retired);
                changed = true;
                index += _pieces.Count;
            }
            else
            {
                index = runEnd + 1;
            }
        }

        return changed;
    }

    /// <summary>
    /// Replaces a window with a new one over the same bytes, so its buffer is made again over
    /// what the host mapping holds now (a buffer keeps the pages it was made over). The old
    /// window goes to <paramref name="retired"/>; the caller must not have it pinned.
    /// </summary>
    public Window Renew(Window window, List<Window> retired)
    {
        var index = IndexOf(window.Start);
        if (!ReferenceEquals(_windows[index], window))
        {
            throw new InvalidOperationException($"Window {window.Id} at 0x{window.Start:X} is not in the partition.");
        }

        _pieces.Clear();
        var renewed = NewWindow(window.Start, window.End);
        renewed.LastUsedSerial = window.LastUsedSerial;
        _pieces.Add(renewed);
        Replace(index, index, retired);
        return renewed;
    }

    private static bool IsIdle(Window window, ulong idleSerial, MetalWindowSet pinned) =>
        window.LastUsedSerial <= idleSerial && !pinned.Contains(window.Id);

    // Fills _pieces with the granule cut of [start, end); false when windows first..last are already it.
    private bool HomeCut(ulong start, ulong end, int first, int last)
    {
        var cuts = _cuts;
        cuts.Clear();
        for (var cut = (start / _granule + 1) * _granule; cut < end; cut += _granule)
        {
            cuts.Add(cut);
        }

        if (cuts.Count != 0 && cuts[0] - start < _granule / 2)
        {
            cuts.RemoveAt(0);
        }

        if (cuts.Count != 0 && end - cuts[^1] < _granule / 2)
        {
            cuts.RemoveAt(cuts.Count - 1);
        }

        if (cuts.Count == last - first)
        {
            var same = true;
            for (var index = 0; index < cuts.Count && same; index++)
            {
                same = _windows[first + index + 1].Start == cuts[index];
            }

            if (same)
            {
                return false;
            }
        }

        _pieces.Clear();
        var pieceStart = start;
        foreach (var cut in cuts)
        {
            _pieces.Add(NewWindow(pieceStart, cut));
            pieceStart = cut;
        }

        _pieces.Add(NewWindow(pieceStart, end));
        return true;
    }

    // Replaces windows first..last with _pieces.
    private void Replace(int first, int last, List<Window> retired)
    {
        for (var index = first; index <= last; index++)
        {
            var old = _windows[index];
            Volatile.Write(ref old.Retired, true);
            retired.Add(old);
            _byId[old.Id] = null;
            _freeIds.Push(old.Id);
        }

        _windows.RemoveRange(first, last - first + 1);
        _windows.InsertRange(first, _pieces);
        Version++;
    }

    private Window NewWindow(ulong start, ulong end)
    {
        var id = _freeIds.Count != 0 ? _freeIds.Pop() : _nextId++;
        if (id >= _byId.Length)
        {
            Array.Resize(ref _byId, Math.Max(id + 1, _byId.Length * 2));
        }

        var window = new Window(id, start, end);
        _byId[id] = window;
        return window;
    }
}
