// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Images;

namespace SharpEmu.Gpu.Metal.V2;

// A writeback is the GPU storing an image's bytes into guest memory, and it runs when the GPU
// reaches it - up to the GPU's lag after it was recorded, which in Silent Hill reached 200 ms.
// The guest decides when memory is free from labels, and a writeback is recorded behind the
// labels that report the image's last use: an image evicted for the budget (or written back
// because something new overlaps it) is published at the point the stream has reached, not
// where the guest last used it. In that window the guest may already have freed the block and
// built something else in it. Its stores no longer fault - a released image has no watch - so
// the writeback then lands over them: Silent Hill's emulation thread called through a vtable
// that had become a released render target's pixels (0xFF405460FF00010D).
//
// So the range a writeback stores into stays write-watched until the submission carrying it
// completes, and a guest store that faults there waits for that completion before it lands.
// The store then follows the writeback, as it followed the GPU's write on the hardware.
//
// A range written back again before the GPU reached the last writeback - a target flushed for
// a binding on every draw - extends its fence to the newer submission rather than adding one,
// and the fenced ranges share one page-guard watch per page (MetalPageWatchSet): the guard
// counts at most 127 watches a page, and Silent Hill, 400 ms behind on the GPU, reached that.
//
// The wait never runs on the GPU worker (it would be waiting for itself), holds no cache lock,
// asks the worker to commit a submission it is still recording, and gives up after
// SHARPEMU_METAL_V2_WRITEBACK_FENCE_TIMEOUT_MS (250 ms) with a warning - a store that cannot
// wait lands as it did before. SHARPEMU_METAL_V2_WRITEBACK_FENCE=0 keeps no watch at all.
internal sealed partial class MetalImageCache
{
    private static readonly bool WritebackFenceDefault =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_WRITEBACK_FENCE") != "0";

    private static readonly int WritebackFenceTimeoutDefault =
        int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_WRITEBACK_FENCE_TIMEOUT_MS"), out var ms) && ms > 0
            ? ms
            : 250;

    private const int MaxFenceTimeoutReports = 8;

    private readonly object _fenceGate = new();
    private readonly List<FencedWriteback> _fencedWritebacks = [];
    private MetalPageWatchSet? _fencedPages;
    private int _fencedWritebackCount;
    private long _fencedWritebacksAdded;
    private long _fenceWaits;
    private long _fenceWaitTicks;
    private long _fenceTimeouts;
    private int _fenceTimeoutReports;

    private readonly record struct FencedWriteback(ulong Address, ulong Size, ulong Serial, MetalWritebackCause Cause);

    /// <summary>Whether a recorded writeback keeps its guest range watched until it completes.</summary>
    internal bool WritebackFenceEnabled { get; set; } = WritebackFenceDefault;

    /// <summary>How long a faulting guest store waits for a writeback's submission.</summary>
    internal int WritebackFenceTimeoutMilliseconds { get; set; } = WritebackFenceTimeoutDefault;

    /// <summary>Writebacks whose range is still watched for their submission.</summary>
    internal int FencedWritebackCount => Volatile.Read(ref _fencedWritebackCount);

    /// <summary>Guest stores that waited for a writeback, and the waits that gave up.</summary>
    internal (long Waits, long Timeouts) FenceWaitCounts =>
        (Interlocked.Read(ref _fenceWaits), Interlocked.Read(ref _fenceTimeouts));

    /// <summary>
    /// The page guard's write watches held now (every watcher's, one per page each), and the
    /// highest count one page reached since the last call; the guard fails past 127.
    /// </summary>
    public (long Live, int PeakPerPage) TakePageWatchCounters() =>
        (_pages.LiveWriteWatches, _pages.TakePeakPageWriteWatches());

    /// <summary>The CHURN line's fence counters since the last call.</summary>
    public string TakeWritebackFenceCounters()
    {
        var added = Interlocked.Exchange(ref _fencedWritebacksAdded, 0);
        var waits = Interlocked.Exchange(ref _fenceWaits, 0);
        var ticks = Interlocked.Exchange(ref _fenceWaitTicks, 0);
        var timeouts = Interlocked.Exchange(ref _fenceTimeouts, 0);
        var waitMs = ticks * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
        return $" wb_fenced={added} wb_fence_live={FencedWritebackCount} wb_fence_waits={waits} " +
            $"wb_fence_wait_ms={waitMs:F1} wb_fence_timeouts={timeouts} " +
            $"wb_before_store_total={StorePublishCounts.Writebacks} wb_before_store_timeouts_total={StorePublishCounts.Timeouts}";
    }

    // The writeback of these bytes was just recorded into the submission with this serial.
    // Caller holds the lock, on the GPU worker.
    private void FenceWriteback(ulong address, ulong size, ulong serial, MetalWritebackCause cause)
    {
        if (!WritebackFenceEnabled || _watchingStopped || size == 0 || serial <= _owner.CompletedSerial)
        {
            return;
        }

        lock (_fenceGate)
        {
            PruneFencedWritebacksLocked(_owner.CompletedSerial);
            // Serials complete in order, so waiting for the newer writeback covers the older.
            var index = _fencedWritebacks.Count - 1;
            while (index >= 0 && (_fencedWritebacks[index].Address != address || _fencedWritebacks[index].Size != size))
            {
                index--;
            }

            if (index >= 0)
            {
                var entry = _fencedWritebacks[index];
                _fencedWritebacks[index] = entry with { Serial = Math.Max(entry.Serial, serial), Cause = cause };
            }
            else
            {
                (_fencedPages ??= new MetalPageWatchSet(_pages)).Add(address, size);
                _fencedWritebacks.Add(new FencedWriteback(address, size, serial, cause));
                Volatile.Write(ref _fencedWritebackCount, _fencedWritebacks.Count);
            }
        }

        Interlocked.Increment(ref _fencedWritebacksAdded);
    }

    // Drops the watch of every writeback the GPU has finished.
    private void PruneFencedWritebacks()
    {
        if (Volatile.Read(ref _fencedWritebackCount) == 0)
        {
            return;
        }

        lock (_fenceGate)
        {
            PruneFencedWritebacksLocked(_owner.CompletedSerial);
        }
    }

    private void PruneFencedWritebacksLocked(ulong completed)
    {
        var kept = 0;
        for (var index = 0; index < _fencedWritebacks.Count; index++)
        {
            var entry = _fencedWritebacks[index];
            if (entry.Serial <= completed)
            {
                _fencedPages!.Remove(entry.Address, entry.Size);
                continue;
            }

            _fencedWritebacks[kept++] = entry;
        }

        _fencedWritebacks.RemoveRange(kept, _fencedWritebacks.Count - kept);
        Volatile.Write(ref _fencedWritebackCount, _fencedWritebacks.Count);
    }

    // Removes the watches of the fenced writebacks sharing a page with the range that are no
    // newer than the serial the store waited for (or gave up on). One recorded while it waited
    // keeps its watch, and the retried store faults again and waits for that one too.
    private void ReleaseFencesOverLocked(ulong address, ulong size, ulong upToSerial)
    {
        var kept = 0;
        for (var index = 0; index < _fencedWritebacks.Count; index++)
        {
            var entry = _fencedWritebacks[index];
            if (entry.Serial <= upToSerial && GuestRangeOverlap.Pages(entry.Address, entry.Size, address, size))
            {
                _fencedPages!.Remove(entry.Address, entry.Size);
                continue;
            }

            _fencedWritebacks[kept++] = entry;
        }

        _fencedWritebacks.RemoveRange(kept, _fencedWritebacks.Count - kept);
        Volatile.Write(ref _fencedWritebackCount, _fencedWritebacks.Count);
    }

    /// <summary>
    /// A guest store faulted in this range. When a recorded writeback still stores into its
    /// page, waits for that writeback's submission so the store lands after it, then drops the
    /// watch. True when a fence covered the page (the store can be retried). Called without the
    /// cache lock.
    /// </summary>
    private bool WaitForFencedWritebacks(ulong address, ulong size)
    {
        ulong pending = 0;
        MetalWritebackCause cause = default;
        var covered = false;
        lock (_fenceGate)
        {
            var completed = _owner.CompletedSerial;
            foreach (var entry in _fencedWritebacks)
            {
                if (!GuestRangeOverlap.Pages(entry.Address, entry.Size, address, size))
                {
                    continue;
                }

                covered = true;
                if (entry.Serial > completed && entry.Serial > pending)
                {
                    pending = entry.Serial;
                    cause = entry.Cause;
                }
            }

            if (!covered)
            {
                return false;
            }

            if (pending == 0)
            {
                ReleaseFencesOverLocked(address, size, completed);
                return true;
            }
        }

        var started = System.Diagnostics.Stopwatch.GetTimestamp();
        var landed = _owner.WaitForSubmissionOffWorker(pending, WritebackFenceTimeoutMilliseconds);
        Interlocked.Add(ref _fenceWaitTicks, System.Diagnostics.Stopwatch.GetTimestamp() - started);
        Interlocked.Increment(ref _fenceWaits);
        if (!landed)
        {
            Interlocked.Increment(ref _fenceTimeouts);
            if (Interlocked.Increment(ref _fenceTimeoutReports) <= MaxFenceTimeoutReports)
            {
                Console.Error.WriteLine(
                    $"[METAL-V2][WARN] A guest store into a recorded writeback did not wait for it: " +
                    $"address=0x{address:X16} serial={pending} completed={_owner.CompletedSerial} cause={cause} " +
                    $"timeout_ms={WritebackFenceTimeoutMilliseconds}. The writeback may land over the store.");
            }
        }

        lock (_fenceGate)
        {
            ReleaseFencesOverLocked(address, size, Math.Max(pending, _owner.CompletedSerial));
        }

        return true;
    }

    // Shutdown: the page guard refuses to be freed with a watch outstanding.
    private void ReleaseAllFencedWritebacks()
    {
        lock (_fenceGate)
        {
            _fencedPages?.Clear();
            _fencedWritebacks.Clear();
            Volatile.Write(ref _fencedWritebackCount, 0);
        }
    }
}
