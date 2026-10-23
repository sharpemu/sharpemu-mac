// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;
using SharpEmu.HLE;
using SharpEmu.HLE.GuestMemory;
using SharpEmu.HLE.Host;
using SharpEmu.Libs.Gpu.Scheduling;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// Windows over a GPU alias with holes (single-copy textures, stage 3 of
/// artifacts/memopt/single-copy-textures.md; MetalSingleCopy has the protocol).
///
/// <para>
/// A command buffer that names a window wires and faults in every page of it, so a window
/// over a chunk whose guest copy was emptied fills it with zeros again and keeps it resident
/// (§4 a2). With single-copy textures on the windows are therefore made over a second
/// mapping of the backing (<see cref="IHostBackingGpuAlias"/>) instead of the canonical alias:
/// the same chunk objects, except that a released chunk is punched out and one shared dummy
/// chunk is mapped there. Host code keeps the canonical alias and its complete bytes.
/// </para>
///
/// <para>
/// A buffer keeps the objects it was made over (§4 c), so every change to the GPU alias
/// renews the windows over the chunk: a new window over the same bytes, with its own
/// buffer, and the old one retired until the submissions that used it complete. A window
/// the command buffer being recorded already used may not change (a byte has one GPU address
/// per command buffer), so:
/// <list type="bullet">
/// <item>punches wait for the next submission, when nothing is pinned yet. The old buffers
/// still read the real chunk, whose bytes stay until it is emptied, and it is emptied only
/// once the punch's submission has completed (MetalSingleCopy);</item>
/// <item>a restored chunk is filled back at once, but a pinned window over it keeps reading
/// the dummy until the next submission renews it: a range over those bytes is refused that
/// window for the rest of this command buffer (staged for a read, written another way),
/// like a cut that cannot move yet.</item>
/// </list>
/// A range a binding resolves never covers a released chunk: it is restored first.
/// </para>
/// </summary>
internal sealed partial class MetalGuestMemory
{
    private GuestSingleCopy? _singleCopy;
    private IHostBackingGpuAlias? _gpuAlias;

    // Where window buffers point: the GPU alias, or the canonical alias without single copy.
    private ulong _windowBase;
    private ulong _chunkBytes;

    // Under _windowLock: chunks the GPU alias maps the dummy over. In the shadow mode
    // (MetalSingleCopy.ShadowMode) the chunks it would map the dummy over: nothing is punched.
    private byte[]? _holes;
    private int _holeCount;
    private bool _singleCopyShadow;

    // Under _windowLock: changes whenever the holes or the stale windows do (the trap map).
    private long _trapVersion;
    private readonly List<(int Chunk, ulong Address)> _pendingPunches = new();
    private readonly List<ulong> _punchedViewAddresses = new();

    // Windows whose buffers still map the dummy over bytes that hold data again, and those
    // byte ranges: renewed as the next submission begins.
    private readonly Dictionary<MetalWindowPartition.Window, List<(ulong Start, ulong End)>> _staleWindows = new();

    // From any thread: chunks restored since the last submission began.
    private readonly ConcurrentQueue<int> _pendingFills = new();

    // To the manager: chunks punched, and the submission that first used windows without them.
    private readonly ConcurrentQueue<(int Chunk, ulong Serial)> _punched = new();
    private long _holePunches;
    private long _holeFills;
    private long _windowRenewals;
    private long _staleRefusals;

    /// <summary>Chunks the GPU alias maps the dummy over now (unlocked).</summary>
    internal int SingleCopyHoleCount => Volatile.Read(ref _holeCount);

    /// <summary>The GPU alias windows are made over, or null when single-copy textures are off.</summary>
    internal IHostBackingGpuAlias? GpuAlias => _gpuAlias;

    /// <summary>Punches and fills of the GPU alias, windows renewed for them, ranges refused a stale window, holes now.</summary>
    internal (long Punches, long Fills, long Renewals, long StaleRefusals, int Holes) SingleCopyWindowCounts
    {
        get
        {
            using (SharpEmu.Libs.VideoOut.GpuLockProfile.Enter(_windowLock, SharpEmu.Libs.VideoOut.GpuLockProfile.Gate.Windows))
            {
                return (_holePunches, _holeFills, _windowRenewals, _staleRefusals, _holeCount);
            }
        }
    }

    private void InitializeSingleCopyWindows(IGuestBackedSpace backing, bool? singleCopy)
    {
        _windowBase = _aliasBase;
        if (!(singleCopy ?? MetalSingleCopy.Enabled) || backing.SingleCopy is not { ChunkBytes: > 0 } state)
        {
            return;
        }

        var gpuAlias = state.CreateGpuAlias();
        if (gpuAlias is null || gpuAlias.Size < _aliasSize)
        {
            gpuAlias?.Dispose();
            if (!MetalSingleCopy.Requested && singleCopy is null)
            {
                return;
            }

            Console.Error.WriteLine(
                "[METAL-V2][WARN] single-copy textures are off: the guest backing has no GPU alias " +
                "(SHARPEMU_METAL_V2_SINGLE_COPY=1 needs SHARPEMU_GUEST_BACKING=chunked).");
            return;
        }

        if (MetalSingleCopy.SentinelMode)
        {
            gpuAlias.FillDummy(MetalSingleCopy.DummySentinel);
        }

        _gpuAlias = gpuAlias;
        _windowBase = gpuAlias.Base;
        _singleCopy = state;
        _chunkBytes = state.ChunkBytes;
        _holes = new byte[state.ChunkCount];
        _singleCopyShadow = MetalSingleCopy.ShadowMode;
    }

    /// <summary>Whether the holes are only recorded, never punched (the shadow mode).</summary>
    internal bool IsSingleCopyShadow => _singleCopyShadow;

    // The trap map of the device-address table (IMetalGranuleRuns): the holes, and the chunks
    // under a window that still maps the dummy over bytes that hold data again.

    public ulong TrapChunkBytes => _holes is null ? 0 : _chunkBytes;

    public int TrapChunkCount => _holes?.Length ?? 0;

    public long TrapVersion
    {
        get
        {
            if (_holes is null)
            {
                return 0;
            }

            using (SharpEmu.Libs.VideoOut.GpuLockProfile.Enter(_windowLock, SharpEmu.Libs.VideoOut.GpuLockProfile.Gate.Windows))
            {
                return _trapVersion;
            }
        }
    }

    public int CopyTrapChunks(Span<byte> chunks)
    {
        if (_holes is not { } holes)
        {
            return 0;
        }

        using (SharpEmu.Libs.VideoOut.GpuLockProfile.Enter(_windowLock, SharpEmu.Libs.VideoOut.GpuLockProfile.Gate.Windows))
        {
            var count = Math.Min(chunks.Length, holes.Length);
            holes.AsSpan(0, count).CopyTo(chunks);
            chunks[count..].Clear();
            var set = _holeCount;
            foreach (var ranges in _staleWindows.Values)
            {
                foreach (var (start, end) in ranges)
                {
                    for (var chunk = start / _chunkBytes; chunk < (ulong)count && chunk * _chunkBytes < end; chunk++)
                    {
                        if (chunks[(int)chunk] == 0)
                        {
                            chunks[(int)chunk] = 1;
                            set++;
                        }
                    }
                }
            }

            return set;
        }
    }

    /// <summary>Punches a released chunk out of the GPU alias as the next submission begins.</summary>
    internal void QueueSingleCopyPunch(int chunk, ulong address)
    {
        using (SharpEmu.Libs.VideoOut.GpuLockProfile.Enter(_windowLock, SharpEmu.Libs.VideoOut.GpuLockProfile.Gate.Windows))
        {
            _pendingPunches.Add((chunk, address));
        }
    }

    /// <summary>A chunk holds its bytes again: the GPU alias maps it back by the next submission.</summary>
    internal void QueueSingleCopyFill(int chunk)
    {
        if (_holes is not null)
        {
            _pendingFills.Enqueue(chunk);
        }
    }

    /// <summary>Fills every queued hole now (the GPU is about to reach those bytes by address).</summary>
    internal void ApplySingleCopyFills()
    {
        if (_holes is null || _pendingFills.IsEmpty)
        {
            return;
        }

        using (SharpEmu.Libs.VideoOut.GpuLockProfile.Enter(_windowLock, SharpEmu.Libs.VideoOut.GpuLockProfile.Gate.Windows))
        {
            DrainFillsLocked();
            RetireReplaced();
        }
    }

    /// <summary>Whether a window this command buffer used still maps a GPU alias hole over bytes that hold data again.</summary>
    internal bool HasStaleSingleCopyWindows
    {
        get
        {
            if (_holes is null)
            {
                return false;
            }

            using (SharpEmu.Libs.VideoOut.GpuLockProfile.Enter(_windowLock, SharpEmu.Libs.VideoOut.GpuLockProfile.Gate.Windows))
            {
                return _staleWindows.Count != 0;
            }
        }
    }

    /// <summary>Whether backing bytes [offset, offset + size) lie under such a window.</summary>
    internal bool IsStaleSingleCopyRange(ulong offset, ulong size)
    {
        using (SharpEmu.Libs.VideoOut.GpuLockProfile.Enter(_windowLock, SharpEmu.Libs.VideoOut.GpuLockProfile.Gate.Windows))
        {
            foreach (var ranges in _staleWindows.Values)
            {
                foreach (var (start, end) in ranges)
                {
                    if (start < offset + size && offset < end)
                    {
                        return true;
                    }
                }
            }

            return false;
        }
    }

    internal bool TryTakeSingleCopyPunched(out (int Chunk, ulong Serial) punched) => _punched.TryDequeue(out punched);

    /// <summary>Whether the GPU alias maps the dummy over a chunk now.</summary>
    internal bool IsSingleCopyHole(int chunk)
    {
        using (SharpEmu.Libs.VideoOut.GpuLockProfile.Enter(_windowLock, SharpEmu.Libs.VideoOut.GpuLockProfile.Gate.Windows))
        {
            return _holes is { } holes && (uint)chunk < (uint)holes.Length && holes[chunk] != 0;
        }
    }

    private void RestoreSingleCopyRange(ulong offset, ulong size)
    {
        if (_singleCopy is { } state && state.AnyReleased(offset, size))
        {
            state.Handler?.Restore(offset, size, GuestSingleCopyAccess.GpuBinding);
        }
    }

    // Caller holds _windowLock. The holes in the range whose chunk holds data again.
    private void FillSingleCopyHolesLocked(ulong offset, ulong size)
    {
        if (_holes is not { } holes || _holeCount == 0 || size == 0)
        {
            return;
        }

        var first = (int)(offset / _chunkBytes);
        var last = (int)Math.Min((offset + size - 1) / _chunkBytes, (ulong)holes.Length - 1);
        if (holes.AsSpan(first, last - first + 1).IndexOfAnyExcept((byte)0) < 0)
        {
            return;
        }

        for (var chunk = first; chunk <= last; chunk++)
        {
            if (holes[chunk] != 0 && !_singleCopy!.IsReleased(chunk))
            {
                FillHoleLocked(chunk);
            }
        }

        RetireReplaced();
    }

    // Caller holds _windowLock. Called as a submission begins, when nothing is pinned.
    private void ApplySingleCopyHolesLocked()
    {
        if (_holes is not { } holes)
        {
            return;
        }

        DrainFillsLocked();
        foreach (var (chunk, address) in _pendingPunches)
        {
            if (!_singleCopy!.IsReleased(chunk))
            {
                continue;
            }

            if (holes[chunk] == 0)
            {
                var (offset, length) = ChunkRange(chunk);
                if (!_singleCopyShadow && !_gpuAlias!.TryPunch(offset, length))
                {
                    throw GpuFatal.Fatal($"A released guest chunk could not be punched out of the GPU alias: offset=0x{offset:X}.");
                }

                holes[chunk] = 1;
                _holeCount++;
                _holePunches++;
                _trapVersion++;
                if (!_singleCopyShadow)
                {
                    RenewWindowsOverLocked(offset, offset + length, refill: false);
                }

                if (!_singleCopy.IsReleased(chunk))
                {
                    // Restored while it was being punched; its fill may already have been drained.
                    FillHoleLocked(chunk);
                    continue;
                }
            }

            if (!_singleCopyShadow)
            {
                _punchedViewAddresses.Add(address);
            }

            _punched.Enqueue((chunk, _submissionSerial));
        }

        _pendingPunches.Clear();
        if (_staleWindows.Count != 0)
        {
            _trapVersion++;
            foreach (var window in _staleWindows.Keys.ToArray())
            {
                if (ReferenceEquals(_partition.ById(window.Id), window))
                {
                    RenewLocked(window);
                }
            }

            _staleWindows.Clear();
        }
    }

    // Guest views (buffers over the guest's own mapping) over a punched chunk would keep
    // reading the real chunk: they go, and none is made over it while it is watched.
    private void DropSingleCopyGuestViews()
    {
        if (_punchedViewAddresses.Count == 0)
        {
            return;
        }

        foreach (var address in _punchedViewAddresses)
        {
            OnGuestUnmap(address, _chunkBytes);
        }

        _punchedViewAddresses.Clear();
    }

    // Caller holds _windowLock.
    private void DrainFillsLocked()
    {
        while (_pendingFills.TryDequeue(out var chunk))
        {
            if (_holes![chunk] != 0 && !_singleCopy!.IsReleased(chunk))
            {
                FillHoleLocked(chunk);
            }
        }
    }

    // Caller holds _windowLock.
    private void FillHoleLocked(int chunk)
    {
        var (offset, length) = ChunkRange(chunk);
        if (!_singleCopyShadow && !_gpuAlias!.TryFill(offset, length))
        {
            throw GpuFatal.Fatal($"A restored guest chunk could not be mapped back into the GPU alias: offset=0x{offset:X}.");
        }

        _holes![chunk] = 0;
        _holeCount--;
        _holeFills++;
        _trapVersion++;
        if (!_singleCopyShadow)
        {
            RenewWindowsOverLocked(offset, offset + length, refill: true);
        }
    }

    private (ulong Offset, ulong Length) ChunkRange(int chunk)
    {
        var offset = (ulong)chunk * _chunkBytes;
        return (offset, Math.Min(_chunkBytes, _gpuAlias!.Size - offset));
    }

    // Caller holds _windowLock. Every window over [start, end) that has a buffer gets a new
    // one; a pinned one cannot, and after a fill it is stale over those bytes until the next
    // submission. (A punch only ever runs with nothing pinned.)
    private void RenewWindowsOverLocked(ulong start, ulong end, bool refill)
    {
        end = Math.Min(end, _aliasSize);
        for (var cursor = start; cursor < end;)
        {
            var window = _partition.At(cursor);
            cursor = window.End;
            if (window.Buffer == 0)
            {
                continue;
            }

            if (!_pinnedWindows.Contains(window.Id))
            {
                RenewLocked(window);
                continue;
            }

            if (refill)
            {
                if (!_staleWindows.TryGetValue(window, out var ranges))
                {
                    _staleWindows[window] = ranges = new List<(ulong, ulong)>();
                }

                ranges.Add((Math.Max(start, window.Start), Math.Min(end, window.End)));
                _trapVersion++;
            }
        }
    }

    // Caller holds _windowLock; the caller retires _retiredScratch.
    private MetalWindowPartition.Window RenewLocked(MetalWindowPartition.Window window)
    {
        if (_staleWindows.Remove(window))
        {
            _trapVersion++;
        }

        _windowRenewals++;
        return _partition.Renew(window, _retiredScratch);
    }

    // Caller holds _windowLock. False when the window is pinned and its buffer still maps the
    // dummy over part of the range; a window that is not pinned yet is renewed instead.
    private bool TryUseSingleCopyWindowLocked(ref MetalWindowPartition.Window window, ulong offset, ulong size)
    {
        if (_staleWindows.Count == 0 || !_staleWindows.TryGetValue(window, out var ranges))
        {
            return true;
        }

        if (!_pinnedWindows.Contains(window.Id))
        {
            window = RenewLocked(window);
            RetireReplaced();
            return true;
        }

        foreach (var (start, end) in ranges)
        {
            if (start < offset + size && offset < end)
            {
                _staleRefusals++;
                MetalV2Diagnostics.InfoOnce(
                    "single-copy-stale-window",
                    "A restored guest chunk lies under a window this command buffer already used, whose buffer still " +
                    $"maps the GPU alias hole; the range has no window until the next command buffer: offset=0x{offset:X} " +
                    $"size=0x{size:X} window={window.Id}.");
                return false;
            }
        }

        return true;
    }
}
