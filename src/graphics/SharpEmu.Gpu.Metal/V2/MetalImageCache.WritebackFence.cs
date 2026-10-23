// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
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
// asks the worker to commit a submission it is still recording, and reports a stall every
// SHARPEMU_METAL_V2_WRITEBACK_FENCE_TIMEOUT_MS (250 ms). A timeout never admits a store:
// only completion or session shutdown ends the wait. SHARPEMU_METAL_V2_WRITEBACK_FENCE=0
// keeps no watch at all.
//
// A managed (HLE) write never faults, so it asks before it lands instead: it waits for a
// fenced writeback over its exact bytes (WaitForFencedWritebacksUnder). On the worker, which
// may be such a writer, that wait is a commit and a wait for the GPU.
internal sealed partial class MetalImageCache
{
    private static readonly bool WritebackFenceDefault =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_WRITEBACK_FENCE") != "0";

    private static readonly int WritebackFenceTimeoutDefault =
        int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_WRITEBACK_FENCE_TIMEOUT_MS"), out var ms) && ms > 0
            ? ms
            : 250;

    private const int MaxFenceTimeoutReports = 8;

    internal bool StoreFastPathEnabled { get; set; } = MetalV2Diagnostics.StoreFastPath;
    private readonly MetalFencePageIndex _fenceIndex = new();
    private int _unindexedFences;
    private long _fenceCompletionFloor;
    private long _fenceIndexVersion;
    private object? _fenceMappingSnapshot;
    private readonly object _fenceGate = new();
    private readonly List<FencedWriteback> _fencedWritebacks = [];
    private MetalPageWatchSet? _fencedPages;
    private int _fencedWritebackCount;
    private long _fencedWritebacksAdded;
    private long _fenceWaits;
    private long _fenceWaitTicks;
    private long _fenceTimeouts;
    private int _fenceTimeoutReports;

    private readonly record struct FencedWriteback(
        ulong Address, ulong Size, ulong Serial, MetalWritebackCause Cause, ulong BackingAddress, bool Indexed);

    // A mapping snapshot change or a scattered/unmapped range is uncertainty, never
    // a negative answer. The captured backing offset also survives guest VA reuse.
    private bool TryFenceBackingRange(ulong address, ulong size, out ulong offset) =>
        MetalFencePageIndex.TryResolveBackingRange(_backing, address, size, out offset);

    private ulong SampleFenceCompletion()
    {
        var completed = _owner.CompletedSerial; // Never called with _fenceGate held.
        if (completed <= long.MaxValue)
        {
            var old = Volatile.Read(ref _fenceCompletionFloor);
            while ((ulong)old < completed)
            {
                var seen = Interlocked.CompareExchange(ref _fenceCompletionFloor, (long)completed, old);
                if (seen == old) break;
                old = seen;
            }
        }
        return completed;
    }

    private bool RejectFenceQuery(ulong address, ulong size, bool exact, out ulong backing, out bool resolved, out object? snapshot)
    {
        var version = Volatile.Read(ref _fenceIndexVersion);
        snapshot = _backing.BackingAliasSnapshot;
        resolved = TryFenceBackingRange(address, size, out backing) &&
            ReferenceEquals(snapshot, Volatile.Read(ref _fenceMappingSnapshot));
        if (!StoreFastPathEnabled || size > 8192 || !resolved || (version & 1) != 0 ||
            Volatile.Read(ref _unindexedFences) != 0)
        {
            MetalStoreFastPathCounters.Count(MetalStoreFastPathCounters.Kind.Fallback);
            return false;
        }
        // Page/fault queries must still remove completed watches. Only the precise
        // pre-write query may reject against a cached completion floor.
        var completed = exact ? (ulong)Volatile.Read(ref _fenceCompletionFloor) : 0;
        if (_fenceIndex.MayOverlap(backing, size, completed))
        {
            MetalStoreFastPathCounters.Count(MetalStoreFastPathCounters.Kind.FenceReject);
            return false;
        }
        // A multi-page scan cannot reject across a concurrent publication or
        // mapping change. This validates the lookup only, not the later memcpy.
        if (version != Volatile.Read(ref _fenceIndexVersion) ||
            !ReferenceEquals(snapshot, _backing.BackingAliasSnapshot))
        {
            resolved = false;
            MetalStoreFastPathCounters.Count(MetalStoreFastPathCounters.Kind.Fallback);
            return false;
        }
        MetalStoreFastPathCounters.Count(MetalStoreFastPathCounters.Kind.FenceFast);
        return true;
    }

    private static bool FenceOverlaps(FencedWriteback entry, ulong address, ulong size,
        ulong backing, bool resolved, bool exact)
    {
        // Unknown mappings conservatively wait. Otherwise test both VA and backing:
        // exact managed bytes must not inherit a neighbouring byte's wait, while
        // native/landed notifications retain tracker-page semantics.
        if (!resolved || !entry.Indexed) return true;
        return exact
            ? GuestRangeOverlap.Bytes(entry.Address, entry.Size, address, size) ||
                GuestRangeOverlap.Bytes(entry.BackingAddress, entry.Size, backing, size)
            : GuestRangeOverlap.Pages(entry.Address, entry.Size, address, size) ||
                GuestRangeOverlap.Pages(entry.BackingAddress, entry.Size, backing, size);
    }

    private void RetireFenceHint(FencedWriteback entry, ulong completed)
    {
        if (entry.Indexed) _fenceIndex.Retire(entry.BackingAddress, entry.Size, entry.Serial, completed);
        else Interlocked.Decrement(ref _unindexedFences);
    }

    /// <summary>Whether a recorded writeback keeps its guest range watched until it completes.</summary>
    internal bool WritebackFenceEnabled { get; set; } = WritebackFenceDefault;

    /// <summary>Interval for reporting a store still waiting for a writeback submission.</summary>
    internal int WritebackFenceTimeoutMilliseconds { get; set; } = WritebackFenceTimeoutDefault;

    /// <summary>Writebacks whose range is still watched for their submission.</summary>
    internal int FencedWritebackCount => Volatile.Read(ref _fencedWritebackCount);

    /// <summary>
    /// Writeback runs staged because no window held them whole, writebacks that left some
    /// bytes behind, and writebacks that recorded nothing.
    /// </summary>
    /// <summary>Upload pieces the transfer refused, which reached nothing.</summary>
    internal long RefusedUploads => Interlocked.Read(ref _refusedUploads);

    internal (long StagedRuns, long Partial, long Unrecorded) WritebackCarryCounts =>
        (Interlocked.Read(ref _stagedWritebackRuns), Interlocked.Read(ref _shortWritebacks),
            Interlocked.Read(ref _unrecordedWritebacks));

    /// <summary>Store waits for a writeback, and expired diagnostic wait intervals.</summary>
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
        FlushUnwrittenWritebackReports();
        var added = Interlocked.Exchange(ref _fencedWritebacksAdded, 0);
        var waits = Interlocked.Exchange(ref _fenceWaits, 0);
        var ticks = Interlocked.Exchange(ref _fenceWaitTicks, 0);
        var timeouts = Interlocked.Exchange(ref _fenceTimeouts, 0);
        var waitMs = ticks * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
        return $" wb_fenced={added} wb_fence_live={FencedWritebackCount} wb_fence_waits={waits} " +
            $"wb_fence_wait_ms={waitMs:F1} wb_fence_timeouts={timeouts} " +
            $"wb_staged_runs_total={Interlocked.Read(ref _stagedWritebackRuns)} " +
            $"wb_partial_total={Interlocked.Read(ref _shortWritebacks)} " +
            $"wb_unrecorded_total={Interlocked.Read(ref _unrecordedWritebacks)} " +
            $"upload_refused_total={Interlocked.Read(ref _refusedUploads)}" + StorePublishCounterText();
    }

    // The writeback of these bytes was just recorded into the submission with this serial.
    // Caller holds the lock, on the GPU worker.
    private void FenceWriteback(ulong address, ulong size, ulong serial, MetalWritebackCause cause, object? mappingSnapshot)
    {
        if (!WritebackFenceEnabled || _watchingStopped || size == 0 || serial <= (ulong)Volatile.Read(ref _fenceCompletionFloor))
        {
            return;
        }

        var completed = SampleFenceCompletion();
        if (serial <= completed) return;
        var resolved = TryFenceBackingRange(address, size, out var backing) &&
            ReferenceEquals(mappingSnapshot, _backing.BackingAliasSnapshot);
        lock (_fenceGate)
        {
            Interlocked.Increment(ref _fenceIndexVersion);
            try
            {
                PruneFencedWritebacksLocked(completed);
                if (_fencedWritebacks.Count == 0)
                {
                    // Captured before resolving any GPU destination. A mapping
                    // change anywhere through encoding vetoes negative queries
                    // until this batch retires, even if its old VA was reused.
                    Volatile.Write(ref _fenceMappingSnapshot, mappingSnapshot);
                }
                var index = _fencedWritebacks.Count - 1;
                while (index >= 0 && (_fencedWritebacks[index].Address != address || _fencedWritebacks[index].Size != size ||
                    _fencedWritebacks[index].Indexed != resolved ||
                    (resolved && _fencedWritebacks[index].BackingAddress != backing)))
                {
                    index--;
                }

                // Repeated writebacks in one submission already published these words.
                // Extending to a newer serial publishes every positive before the list,
                // watch and live count, and before the caller encodes a GPU store.
                var indexed = resolved && (index >= 0 && _fencedWritebacks[index].Serial >= serial ||
                    _fenceIndex.Publish(backing, size, serial));
                if (index >= 0 && _fencedWritebacks[index].Indexed != indexed) index = -1;
                if (index >= 0)
                {
                    var entry = _fencedWritebacks[index];
                    _fencedWritebacks[index] = entry with { Serial = Math.Max(entry.Serial, serial), Cause = cause };
                }
                else
                {
                    // Unknown mapping/capacity/serial must veto rejection before exposure.
                    if (!indexed) Interlocked.Increment(ref _unindexedFences);
                    (_fencedPages ??= new MetalPageWatchSet(_pages)).Add(address, size);
                    _fencedWritebacks.Add(new FencedWriteback(address, size, serial, cause, backing, indexed));
                    Volatile.Write(ref _fencedWritebackCount, _fencedWritebacks.Count);
                }
            }
            finally
            {
                Interlocked.Increment(ref _fenceIndexVersion);
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

        var completed = SampleFenceCompletion();
        lock (_fenceGate)
        {
            Interlocked.Increment(ref _fenceIndexVersion);
            try { PruneFencedWritebacksLocked(completed); }
            finally { Interlocked.Increment(ref _fenceIndexVersion); }
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
                RetireFenceHint(entry, completed);
                continue;
            }

            _fencedWritebacks[kept++] = entry;
        }

        _fencedWritebacks.RemoveRange(kept, _fencedWritebacks.Count - kept);
        Volatile.Write(ref _fencedWritebackCount, _fencedWritebacks.Count);
    }

    // Removes the watches of the fenced writebacks sharing a page with the range that are no
    // newer than the completed serial the store waited for. One recorded while it waited
    // keeps its watch, and the retried store faults again and waits for that one too.
    private void ReleaseFencesOverLocked(ulong address, ulong size, ulong upToSerial, ulong completed, ulong backing, bool resolved)
    {
        Interlocked.Increment(ref _fenceIndexVersion);
        try
        {
            var kept = 0;
            for (var index = 0; index < _fencedWritebacks.Count; index++)
            {
                var entry = _fencedWritebacks[index];
                if (entry.Serial <= upToSerial && FenceOverlaps(entry, address, size, backing, resolved, exact: false))
                {
                    _fencedPages!.Remove(entry.Address, entry.Size);
                    RetireFenceHint(entry, completed);
                    continue;
                }

                _fencedWritebacks[kept++] = entry;
            }

            _fencedWritebacks.RemoveRange(kept, _fencedWritebacks.Count - kept);
            Volatile.Write(ref _fencedWritebackCount, _fencedWritebacks.Count);
        }
        finally
        {
            Interlocked.Increment(ref _fenceIndexVersion);
        }
    }

    /// <summary>
    /// A guest store faulted in this range. When a recorded writeback still stores into its
    /// page, waits for that writeback's submission so the store lands after it, then drops the
    /// watch. True when a fence covered the page (the store can be retried). Called without the
    /// cache lock.
    /// </summary>
    private bool WaitForFencedWritebacks(ulong address, ulong size, out bool slow)
    {
        slow = false;
        if (RejectFenceQuery(address, size, exact: false, out var backing, out var resolved, out var snapshot)) return false;
        slow = true;
        var completed = SampleFenceCompletion();
        ulong pending = 0;
        MetalWritebackCause cause = default;
        var covered = false;
        var byteOverlap = false;
        MetalStoreFastPathCounters.Count(MetalStoreFastPathCounters.Kind.FenceGate);
        lock (_fenceGate)
        {
            resolved &= ReferenceEquals(snapshot, _backing.BackingAliasSnapshot) &&
                ReferenceEquals(snapshot, _fenceMappingSnapshot);
            foreach (var entry in _fencedWritebacks)
            {
                if (!FenceOverlaps(entry, address, size, backing, resolved, exact: false))
                {
                    continue;
                }

                covered = true;
                if (entry.Serial > completed && resolved && entry.Indexed &&
                    FenceOverlaps(entry, address, size, backing, resolved, exact: true)) byteOverlap = true;
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
                ReleaseFencesOverLocked(address, size, completed, completed, backing, resolved);
                return true;
            }
        }

        if (byteOverlap) MetalStoreFastPathCounters.Count(MetalStoreFastPathCounters.Kind.TrueOverlap);
        var started = System.Diagnostics.Stopwatch.GetTimestamp();
        WaitForFencedSubmission(pending, address, cause, managed: false);
        Interlocked.Add(ref _fenceWaitTicks, System.Diagnostics.Stopwatch.GetTimestamp() - started);
        Interlocked.Increment(ref _fenceWaits);

        completed = SampleFenceCompletion();
        lock (_fenceGate)
        {
            ReleaseFencesOverLocked(address, size, completed, completed, backing, resolved);
        }

        return true;
    }

    /// <summary>
    /// A managed write of exactly these bytes, about to land: waits for the newest recorded
    /// writeback still storing into them, which would otherwise complete after the write and
    /// land over it. A managed write never faults, so nothing retries it and no watch is
    /// dropped here; a fence ends when its submission completes. Off the worker the wait is
    /// a faulting store's, with the same diagnostic interval. The worker cannot wait for a
    /// submission it may still be recording, so it commits and waits for the GPU, as it does
    /// for the writeback it records before its own managed write (AwaitStoreRequest). Called
    /// without the cache lock.
    /// </summary>
    private bool WaitForFencedWritebacksUnder(ulong address, ulong size)
    {
        if (RejectFenceQuery(address, size, exact: true, out var backing, out var resolved, out var snapshot)) return false;
        var completed = SampleFenceCompletion();
        MetalStoreFastPathCounters.Count(MetalStoreFastPathCounters.Kind.FenceGate);
        ulong pending = 0;
        MetalWritebackCause cause = default;
        var byteOverlap = false;
        lock (_fenceGate)
        {
            resolved &= ReferenceEquals(snapshot, _backing.BackingAliasSnapshot) &&
                ReferenceEquals(snapshot, _fenceMappingSnapshot);
            foreach (var entry in _fencedWritebacks)
            {
                if (entry.Serial > completed && entry.Serial > pending &&
                    FenceOverlaps(entry, address, size, backing, resolved, exact: true))
                {
                    byteOverlap |= resolved && entry.Indexed;
                    pending = entry.Serial;
                    cause = entry.Cause;
                }
            }
        }

        if (pending == 0)
        {
            return true;
        }

        if (byteOverlap) MetalStoreFastPathCounters.Count(MetalStoreFastPathCounters.Kind.TrueOverlap);
        var started = System.Diagnostics.Stopwatch.GetTimestamp();
        if (_owner.IsWorkerThread)
        {
            _owner.WaitForGpu();
        }
        else
        {
            WaitForFencedSubmission(pending, address, cause, managed: true);
        }

        Interlocked.Add(ref _fenceWaitTicks, System.Diagnostics.Stopwatch.GetTimestamp() - started);
        Interlocked.Increment(ref _fenceWaits);
        return true;
    }

    // An encoded writeback cannot be withdrawn. Letting the store through on a
    // timeout lets that GPU write corrupt reused guest heap. Match the publication
    // path's completion/shutdown rule, without holding either cache or fence lock.
    private void WaitForFencedSubmission(ulong serial, ulong address, MetalWritebackCause cause, bool managed)
    {
        while (!_owner.WaitForSubmissionOffWorker(serial, WritebackFenceTimeoutMilliseconds))
        {
            if (_watchingStopped || HostSessionControl.IsShutdownRequested) return;
            Interlocked.Increment(ref _fenceTimeouts);
            if (Interlocked.Increment(ref _fenceTimeoutReports) <= MaxFenceTimeoutReports)
            {
                Console.Error.WriteLine(
                    $"[METAL-V2][WARN] A {(managed ? "managed write" : "guest store")} is still waiting for a recorded writeback: " +
                    $"address=0x{address:X16} serial={serial} completed={_owner.CompletedSerial} cause={cause} " +
                    $"interval_ms={WritebackFenceTimeoutMilliseconds}. The store remains blocked.");
            }
        }
    }

    // Shutdown: the page guard refuses to be freed with a watch outstanding.
    private void ReleaseAllFencedWritebacks()
    {
        lock (_fenceGate)
        {
            Interlocked.Increment(ref _fenceIndexVersion);
            try
            {
                _fencedPages?.Clear();
                _fencedWritebacks.Clear();
                Volatile.Write(ref _unindexedFences, 0);
                Volatile.Write(ref _fencedWritebackCount, 0);
            }
            finally
            {
                Interlocked.Increment(ref _fenceIndexVersion);
            }
        }
    }
}
