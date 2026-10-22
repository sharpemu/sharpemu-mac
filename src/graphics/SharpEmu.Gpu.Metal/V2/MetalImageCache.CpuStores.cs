// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using SharpEmu.HLE;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Images;

namespace SharpEmu.Gpu.Metal.V2;

// Guest stores into an image whose bytes only its texture holds.
//
// A GPU-written image is published to guest memory lazily, so until then guest memory under
// it holds whatever was there before. A store that faults on the image's page watch gives the
// image to guest memory - marks it stale, drops the GPU's claim, reloads it on the next use -
// because the fault cannot say which bytes were stored, and later stores into the page go
// unseen. Without more, that replaces everything the GPU wrote with those older bytes, where
// on the PS5 the store changes the bytes it writes and nothing else. Silent Hill: The Short
// Message filters its reflection captures into a cube-array pool of up to 100 MB on the GPU
// and then stores the first texel of each cube it places there from the CPU; every such store
// turned the whole pool into stale memory, and every surface reflecting it sampled NaN and
// values in the thousands (the saturated "neon" scenes).
//
// So before such a store lands, the GPU worker writes the image back, and the store waits for
// that writeback to complete. Guest memory then holds the GPU's bytes, the store lands over
// them, and the image goes to guest memory as before, now losing nothing. Three rules keep it
// exact:
//  - A writeback never lands after the store. The game reuses a transient image's memory for
//    heap and command data, and GPU bytes landing late over a heap header corrupt it. A
//    request the worker has taken is waited for until its submission completes, however long
//    that is: a committed command buffer completes without the CPU's help.
//  - Nothing the GPU writes is dropped. The worker keeps recording while the store waits, and
//    an image it writes again is GPU-written again; the check that nothing over the range is,
//    and the store's own invalidation, happen under one hold of the lock, so another round
//    publishes what was written meanwhile.
//  - A request is the store's own. The worker takes requests between packets (the relay runs
//    at every packet boundary), so it is late only by one packet's work, which while shaders
//    compile cold is hundreds of milliseconds. A request still not taken after
//    StorePublishTakeTimeoutMilliseconds (as it was when the request was made) means a hung
//    worker: it is withdrawn under the lock, so it cannot be written back after the store,
//    and the store takes the old path.
// Every store that takes the old path anyway is counted by reason and reported once the lock
// is let go, and the time faulting threads spend blocked here is counted. A managed write is
// published before it lands (MarkPreciseCpuWrite) and never after (MarkCpuWriteLanded).
//
// SHARPEMU_METAL_V2_PUBLISH_BEFORE_STORE=0 gives every store the old path.
internal sealed partial class MetalImageCache
{
    private const int StoreRequestCount = 64;
    private const int MaxImagesPerStoreRequest = 8;
    private const int MaxStorePublishRounds = 8;
    private const int StoreCompletionReportMilliseconds = 1000;
    private const int MaxStorePublishReports = 16;
    private const int StoreShutdownPollMilliseconds = 250;

    private static readonly bool PublishBeforeStoreDefault =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_PUBLISH_BEFORE_STORE") != "0";

    /// <summary>Why a store took the old path, losing what the GPU wrote.</summary>
    internal enum StorePublishFallback
    {
        // No worker could be handed the request: the store came from the worker itself, which
        // cannot wait for its own writeback in the middle of a packet, or the worker is gone.
        Worker,
        // The worker did not take the request in time.
        Hung,
        // Every request was in use.
        Full,
        // The writeback could not be recorded.
        Failed,
        // A writeback left some bytes behind: no run could carry them. The rounds go on, and a
        // store counts this once however many of its images fell short.
        Partial,
        // The GPU kept writing the range for every round.
        Rounds,
        // The host is shutting down. Counted, but not as a loss: nothing reads the image again.
        Shutdown,
    }

    private static class StoreRequestState
    {
        public const int Free = 0;
        public const int Pending = 1;
        public const int Recorded = 2;
        public const int Failed = 3;
    }

    // Owned by the store that fills it; the worker only moves a pending one on, under the lock.
    private sealed class StoreRequest
    {
        public int State;
        public ulong Serial;
        public int Count;
        public int TakeTimeoutMilliseconds;
        // A failed request's image, whose writeback recorded nothing.
        public ImageDescription Unwritten;
        // The images whose writebacks left bytes behind.
        public int PartialCount;
        public readonly ImageDescription[] Partials = new ImageDescription[MaxImagesPerStoreRequest];
        public readonly ResourceSlotIdentifier[] Images = new ResourceSlotIdentifier[MaxImagesPerStoreRequest];
    }

    /// <summary>
    /// A store that lost some of what the GPU wrote, reported once the cache lock is let go:
    /// the reason it took the old path, if it did, the image that recorded nothing, and the
    /// images whose writebacks left bytes behind.
    /// </summary>
    internal readonly record struct StoreFallback(
        StorePublishFallback? Reason, ulong Address, ImageDescription? Unwritten, List<ImageDescription>? Partials);

    private readonly StoreRequest[] _storeRequests = CreateStoreRequests();
    private readonly object _storeRequestGate = new();
    private readonly long[] _storeFallbacks = new long[Enum.GetValues<StorePublishFallback>().Length];
    private int _storeRequestsPending;
    private Action? _serviceStoreRequests;
    private long _storePublishWritebacks;
    private long _storeCompletionStalls;
    private long _storeTakeMaxTicks;
    private long _storeBlockedTicks;
    private long _storeBlockedMaxTicks;
    private int _storePublishReports;

    /// <summary>Whether a guest store into a GPU-written image waits for the image's writeback.</summary>
    internal bool PublishBeforeStore { get; set; } = PublishBeforeStoreDefault;

    /// <summary>
    /// How long a store waits for the worker to take its request before it calls the worker
    /// hung. Read when the request is made, so a change applies to stores made after it.
    /// </summary>
    internal int StorePublishTakeTimeoutMilliseconds { get; set; } = 10_000;

    /// <summary>Requests waiting for the worker to take them.</summary>
    internal int PendingStoreRequests => Volatile.Read(ref _storeRequestsPending);

    /// <summary>What publishing before stores has done since the cache was made.</summary>
    internal StorePublishCounters StorePublishCounts => new(
        Interlocked.Read(ref _storePublishWritebacks),
        Interlocked.Read(ref _storeFallbacks[(int)StorePublishFallback.Worker]),
        Interlocked.Read(ref _storeFallbacks[(int)StorePublishFallback.Hung]),
        Interlocked.Read(ref _storeFallbacks[(int)StorePublishFallback.Full]),
        Interlocked.Read(ref _storeFallbacks[(int)StorePublishFallback.Failed]),
        Interlocked.Read(ref _storeFallbacks[(int)StorePublishFallback.Partial]),
        Interlocked.Read(ref _storeFallbacks[(int)StorePublishFallback.Rounds]),
        Interlocked.Read(ref _storeFallbacks[(int)StorePublishFallback.Shutdown]),
        Interlocked.Read(ref _storeCompletionStalls));

    internal readonly record struct StorePublishCounters(
        long Writebacks, long Worker, long Hung, long Full, long Failed, long Partial, long Rounds, long Shutdown,
        long CompletionStalls)
    {
        // Shutdown is left out: a store cut short by teardown loses nothing anyone reads.
        public long Fallbacks => Worker + Hung + Full + Failed + Partial + Rounds;
    }

    private string StorePublishCounterText()
    {
        var counts = StorePublishCounts;
        return $" wb_before_store_total={counts.Writebacks} wb_before_store_fallbacks_total={counts.Fallbacks} " +
            $"(worker={counts.Worker} hung={counts.Hung} full={counts.Full} failed={counts.Failed} " +
            $"partial={counts.Partial} rounds={counts.Rounds}) wb_before_store_shutdown_total={counts.Shutdown} " +
            $"wb_before_store_stalls_total={counts.CompletionStalls} " +
            $"wb_before_store_take_max_ms={Milliseconds(Interlocked.Read(ref _storeTakeMaxTicks)):F1} " +
            $"wb_before_store_blocked_ms_total={Milliseconds(Interlocked.Read(ref _storeBlockedTicks)):F1} " +
            $"wb_before_store_blocked_max_ms={Milliseconds(Interlocked.Read(ref _storeBlockedMaxTicks)):F1}";
    }

    private static double Milliseconds(long ticks) => ticks * 1000.0 / Stopwatch.Frequency;

    private static StoreRequest[] CreateStoreRequests()
    {
        var requests = new StoreRequest[StoreRequestCount];
        for (var index = 0; index < requests.Length; index++)
        {
            requests[index] = new StoreRequest();
        }

        return requests;
    }

    // Before a store into [address, address + size) lands. Called and returning with the lock
    // held; it is let go while the worker and the GPU are waited for. On return no image over
    // the range is GPU-written, unless the store took the old path, which the caller reports
    // (ReportStoreFallback) once it has let the lock go. Exact asks only for the images over
    // the stored bytes - a managed write to backed memory, which invalidates no more - and
    // otherwise for every image on their pages, as a fault does.
    private StoreFallback? PublishBeforeCpuStoreLocked(ulong address, ulong size, bool managedWrite, bool exact)
    {
        if (!PublishBeforeStore || !WritebackFenceEnabled || !HasGpuWrittenImageLocked(address, size, exact))
        {
            return null;
        }

        var started = Stopwatch.GetTimestamp();
        try
        {
            return PublishRoundsLocked(address, size, managedWrite, exact);
        }
        finally
        {
            NoteStoreBlocked(Stopwatch.GetTimestamp() - started);
        }
    }

    private StoreFallback? PublishRoundsLocked(ulong address, ulong size, bool managedWrite, bool exact)
    {
        List<ImageDescription>? partials = null;
        for (var round = 0; !_watchingStopped; round++)
        {
            if (!HasGpuWrittenImageLocked(address, size, exact))
            {
                break;
            }

            if (round == MaxStorePublishRounds)
            {
                return NoteStoreFallback(StorePublishFallback.Rounds, address, null, partials);
            }

            if (RequestStorePublishLocked(address, size, exact) is not { } request)
            {
                return NoteStoreFallback(StorePublishFallback.Full, address, null, partials);
            }

            var completed = false;
            var withdrawn = StorePublishFallback.Worker;
            var returned = false;
            _lock.Exit();
            try
            {
                completed = AwaitStoreRequest(request, managedWrite, out withdrawn);
                returned = true;
            }
            finally
            {
                _lock.Enter();
                if (!returned)
                {
                    // Thrown out of the wait: the slot goes back, and a pending request with it.
                    FreeStoreRequestLocked(request);
                }
            }

            var state = request.State;
            if (state == StoreRequestState.Pending)
            {
                // Withdrawn under the lock: the worker can no longer write it back after the store.
                FreeStoreRequestLocked(request);
                return NoteStoreFallback(withdrawn, address, null, partials);
            }

            if (!completed)
            {
                // Taken after the wait gave up on it: its writeback is recorded, and lands first.
                var waited = false;
                _lock.Exit();
                try
                {
                    WaitForStoreWriteback(request.Serial);
                    waited = true;
                }
                finally
                {
                    _lock.Enter();
                    if (!waited)
                    {
                        FreeStoreRequestLocked(request);
                    }
                }
            }

            for (var index = 0; index < request.PartialCount; index++)
            {
                (partials ??= []).Add(request.Partials[index]);
            }

            var unwritten = request.Unwritten;
            FreeStoreRequestLocked(request);
            if (state == StoreRequestState.Failed)
            {
                return NoteStoreFallback(StorePublishFallback.Failed, address, unwritten, partials);
            }
        }

        return NoteStoreFallback(null, address, null, partials);
    }

    private bool HasGpuWrittenImageLocked(ulong address, ulong size, bool exact)
    {
        foreach (var imageIdentifier in FindImagesInRange(address, size, pageOverlap: !exact))
        {
            if (_slots.TryGet(imageIdentifier) is { IsGpuModified: true })
            {
                return true;
            }
        }

        return false;
    }

    // A free request filled with the GPU-written images over the range, at most a request's
    // worth; the next round takes the rest. Null when every request is in use.
    private StoreRequest? RequestStorePublishLocked(ulong address, ulong size, bool exact)
    {
        StoreRequest? request = null;
        foreach (var candidate in _storeRequests)
        {
            if (candidate.State == StoreRequestState.Free)
            {
                request = candidate;
                break;
            }
        }

        if (request is null)
        {
            return null;
        }

        var count = 0;
        foreach (var imageIdentifier in FindImagesInRange(address, size, pageOverlap: !exact))
        {
            if (count < MaxImagesPerStoreRequest && _slots.TryGet(imageIdentifier) is { IsGpuModified: true })
            {
                request.Images[count++] = imageIdentifier;
            }
        }

        request.Count = count;
        request.Serial = 0;
        request.TakeTimeoutMilliseconds = StorePublishTakeTimeoutMilliseconds;
        Volatile.Write(ref request.State, StoreRequestState.Pending);
        _storeRequestsPending++;
        return request;
    }

    private void FreeStoreRequestLocked(StoreRequest request)
    {
        if (request.State == StoreRequestState.Pending)
        {
            _storeRequestsPending--;
        }

        request.Count = 0;
        request.Unwritten = default;
        request.PartialCount = 0;
        Volatile.Write(ref request.State, StoreRequestState.Free);
    }

    // Outside the lock. True when the request was taken and its writeback has completed; false
    // when it is still pending (withdrawn by the caller) or was taken too late to be waited for here.
    private bool AwaitStoreRequest(StoreRequest request, bool managedWrite, out StorePublishFallback withdrawn)
    {
        // Why the store takes the old path if the request is still pending on return.
        withdrawn = StorePublishFallback.Worker;
        if (_owner.IsWorkerThread)
        {
            // A managed write the worker makes itself, between packets: it can record the
            // writeback and wait for the GPU like any read of GPU-written bytes there. A fault
            // on the worker can be anywhere, so that store keeps its old path.
            if (!managedWrite)
            {
                return false;
            }

            withdrawn = StorePublishFallback.Hung;
            ServiceStoreRequests();
            if (Volatile.Read(ref request.State) != StoreRequestState.Pending)
            {
                _owner.WaitForGpu();
                return true;
            }

            return false;
        }

        var started = Stopwatch.GetTimestamp();
        if (!_owner.TryRunOnWorker(_serviceStoreRequests ??= ServiceStoreRequests))
        {
            return false;
        }

        withdrawn = StorePublishFallback.Hung;
        var deadline = Environment.TickCount64 + request.TakeTimeoutMilliseconds;
        lock (_storeRequestGate)
        {
            while (Volatile.Read(ref request.State) == StoreRequestState.Pending)
            {
                if (_watchingStopped || HostSessionControl.IsShutdownRequested)
                {
                    // The worker is going away, not hung.
                    withdrawn = StorePublishFallback.Shutdown;
                    return false;
                }

                var remaining = deadline - Environment.TickCount64;
                if (remaining <= 0)
                {
                    return false;
                }

                Monitor.Wait(_storeRequestGate, (int)Math.Min(remaining, StoreShutdownPollMilliseconds));
            }
        }

        NoteStoreTakeLatency(Stopwatch.GetTimestamp() - started);
        WaitForStoreWriteback(request.Serial);
        return true;
    }

    // Until the submission holding the writeback completes. Nothing bounds this but shutdown:
    // the store landing first is what this exists to prevent, and a committed command buffer
    // needs nothing from the CPU to complete.
    private void WaitForStoreWriteback(ulong serial)
    {
        var stalled = false;
        while (!_owner.WaitForSubmissionOffWorker(serial, StoreCompletionReportMilliseconds))
        {
            if (_watchingStopped || HostSessionControl.IsShutdownRequested)
            {
                return;
            }

            if (!stalled)
            {
                stalled = true;
                Interlocked.Increment(ref _storeCompletionStalls);
                ReportStorePublish(
                    $"A guest store is still waiting for the writeback under it: serial={serial} " +
                    $"completed={_owner.CompletedSerial}.");
            }
        }
    }

    // On the GPU worker, between packets: every pending request's images go back to guest memory.
    private void ServiceStoreRequests()
    {
        if (Volatile.Read(ref _storeRequestsPending) == 0)
        {
            return;
        }

        var recorded = false;
        using (var held = _lock.Hold())
        {
            foreach (var request in _storeRequests)
            {
                if (request.State != StoreRequestState.Pending)
                {
                    continue;
                }

                var failed = false;
                var wroteBack = false;
                for (var index = 0; index < request.Count; index++)
                {
                    if (_slots.TryGet(request.Images[index]) is not { IsGpuModified: true } image)
                    {
                        continue;
                    }

                    var outcome = WriteBackToGuest(image, MetalWritebackCause.HostRead);
                    wroteBack = true;
                    if (outcome == MetalWritebackOutcome.Complete)
                    {
                        Interlocked.Increment(ref _storePublishWritebacks);
                        continue;
                    }

                    if (outcome == MetalWritebackOutcome.Partial)
                    {
                        // Recorded, with bytes left behind: counted, and the rounds go on.
                        Interlocked.Increment(ref _storePublishWritebacks);
                        request.Partials[request.PartialCount++] = image.Description;
                        continue;
                    }

                    // Recorded nothing: the image stays GPU-written.
                    if (!failed)
                    {
                        request.Unwritten = image.Description;
                    }

                    failed = true;
                }

                request.Serial = wroteBack ? _owner.CurrentSerial : 0;
                recorded |= wroteBack;
                Volatile.Write(ref request.State, failed ? StoreRequestState.Failed : StoreRequestState.Recorded);
                _storeRequestsPending--;
            }
        }

        if (recorded)
        {
            _owner.CommitRecorded();
        }

        lock (_storeRequestGate)
        {
            Monitor.PulseAll(_storeRequestGate);
        }

        FlushUnwrittenWritebackReports();
    }

    private void NoteStoreTakeLatency(long ticks)
    {
        var max = Interlocked.Read(ref _storeTakeMaxTicks);
        while (ticks > max)
        {
            var seen = Interlocked.CompareExchange(ref _storeTakeMaxTicks, ticks, max);
            if (seen == max)
            {
                break;
            }

            max = seen;
        }
    }

    private void NoteStoreBlocked(long ticks)
    {
        Interlocked.Add(ref _storeBlockedTicks, ticks);
        var max = Interlocked.Read(ref _storeBlockedMaxTicks);
        while (ticks > max)
        {
            var seen = Interlocked.CompareExchange(ref _storeBlockedMaxTicks, ticks, max);
            if (seen == max)
            {
                break;
            }

            max = seen;
        }
    }

    // Counted where it happens, under the lock; the report waits for the lock to be let go.
    // Null when the store lost nothing.
    private StoreFallback? NoteStoreFallback(
        StorePublishFallback? reason, ulong address, ImageDescription? unwritten, List<ImageDescription>? partials)
    {
        if (reason is { } why)
        {
            Interlocked.Increment(ref _storeFallbacks[(int)why]);
        }

        if (partials != null)
        {
            Interlocked.Increment(ref _storeFallbacks[(int)StorePublishFallback.Partial]);
        }

        return reason is null && partials is null ? null : new StoreFallback(reason, address, unwritten, partials);
    }

    // Without the cache lock: it formats and writes to the console.
    private void ReportStoreFallback(in StoreFallback? fallback)
    {
        FlushUnwrittenWritebackReports();
        if (fallback is not { } report)
        {
            return;
        }

        var unwritten = report.Unwritten is { } image ? $" unwritten={ProfileLabel("writeback", in image)}" : "";
        var partial = "";
        if (report.Partials is { } partials)
        {
            var labels = new string[partials.Count];
            for (var index = 0; index < labels.Length; index++)
            {
                labels[index] = ProfileLabel("writeback", partials[index]);
            }

            partial = $" partial=[{string.Join("; ", labels)}]";
        }

        ReportStorePublish(report.Reason is { } reason
            ? $"A guest store into a GPU-written image took the old path ({reason}): " +
              $"address=0x{report.Address:X16}{unwritten}{partial}. " +
              "The image reloads from guest memory, which does not hold what the GPU wrote."
            : $"A guest store landed over GPU-written images whose writebacks left bytes behind (Partial): " +
              $"address=0x{report.Address:X16}{partial}. Those bytes reload from guest memory.");
    }

    private void ReportStorePublish(string message)
    {
        if (Interlocked.Increment(ref _storePublishReports) <= MaxStorePublishReports)
        {
            Console.Error.WriteLine($"[METAL-V2][WARN] {message}");
        }
    }
}
