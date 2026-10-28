// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE.GpuMemory;

namespace SharpEmu.Gpu.Metal.V2;

// In the split GPU process the guest's managed writes reach this cache across processes, and the
// call that runs MarkPreciseCpuWrite before a write lands costs the guest thread a round trip.
// That call does something before the write only where this cache holds bytes guest memory lacks:
// an image the GPU wrote and has not written back (it is written back first), or a recorded
// writeback whose submission has not completed (the write waits for it). Everything else the call
// does, the write's ordered record (MarkCpuWriteLanded) does too. So the cache keeps those pages
// in the summary the guest process reads (IPendingWritebackPages, SplitSharedState), and the
// guest calls only where one is marked.
//
// A clear page must mean a call made at that moment would find nothing to publish or wait for.
// A call sees this cache's state as of a release of the lock, so the summary follows the same
// states:
//  - a GPU-modified image claims its pages before it is marked GPU-modified, and gives them up
//    when it is no longer (MetalCachedImage.PendingWritebacks), or when it leaves the cache;
//  - a writeback claims its range as it is fenced, and gives it up once its submission completed
//    (a call does not wait for a completed one);
//  - a page given up is unmarked only as the lock is let go, and only if nothing claims it then.
//    A writeback ends the image's claim before it starts its own, inside one hold of the lock.
// Marking is immediate and a full barrier, unmarking happens under the lock: so a page is marked
// whenever a call would act on it, and a guest write that saw it clear lands as one would after a
// call that found nothing (the GPU may still write the image after either; that race is the
// guest's, as on the hardware).
internal sealed partial class MetalImageCache
{
    private IPendingWritebackPages? _pendingWritebacks;
    private readonly object _pendingFenceGate = new();
    private readonly Dictionary<(ulong Address, ulong Size), ulong> _pendingFences = new();
    private ulong _pendingFenceMinSerial = ulong.MaxValue;
    private int _pendingFenceCount;

    /// <summary>
    /// The split's pending-writeback summary, kept from here on (null in one process). Set it
    /// before the first image is made.
    /// </summary>
    internal IPendingWritebackPages? PendingWritebacks
    {
        get => _pendingWritebacks;
        set
        {
            _pendingWritebacks = value;
            _lock.BeforeRelease = value is null ? null : PublishPendingWritebacks;
            value?.Activate();
        }
    }

    // A recorded writeback's range stays pending until its submission completes.
    private void ClaimPendingFence(ulong address, ulong size, ulong serial)
    {
        if (_pendingWritebacks is not { } pending)
        {
            return;
        }

        lock (_pendingFenceGate)
        {
            if (_pendingFences.TryGetValue((address, size), out var held))
            {
                _pendingFences[(address, size)] = Math.Max(held, serial);
                return;
            }

            pending.Add(address, size);
            _pendingFences.Add((address, size), serial);
            _pendingFenceMinSerial = Math.Min(_pendingFenceMinSerial, serial);
            Volatile.Write(ref _pendingFenceCount, _pendingFences.Count);
        }
    }

    // On every release of the lock (RegionLock.BeforeRelease), with the lock still held.
    private void PublishPendingWritebacks()
    {
        var pending = _pendingWritebacks!;
        if (Volatile.Read(ref _pendingFenceCount) != 0)
        {
            var completed = _owner.CompletedSerial;
            lock (_pendingFenceGate)
            {
                if (completed >= _pendingFenceMinSerial)
                {
                    ReleaseCompletedFencesLocked(pending, completed);
                }
            }
        }

        pending.Publish();
    }

    private void ReleaseCompletedFencesLocked(IPendingWritebackPages pending, ulong completed)
    {
        List<(ulong Address, ulong Size)>? done = null;
        var minimum = ulong.MaxValue;
        foreach (var (range, serial) in _pendingFences)
        {
            if (serial <= completed)
            {
                (done ??= []).Add(range);
            }
            else
            {
                minimum = Math.Min(minimum, serial);
            }
        }

        if (done is not null)
        {
            foreach (var range in done)
            {
                _pendingFences.Remove(range);
                pending.Remove(range.Address, range.Size);
            }
        }

        _pendingFenceMinSerial = minimum;
        Volatile.Write(ref _pendingFenceCount, _pendingFences.Count);
    }
}
