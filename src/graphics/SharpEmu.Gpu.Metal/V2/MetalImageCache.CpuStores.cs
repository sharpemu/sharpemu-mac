// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using SharpEmu.HLE;
using SharpEmu.Libs.Gpu.Buffers;

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
//    StorePublishTakeTimeoutMilliseconds means a hung worker: it is withdrawn under the lock,
//    so it cannot be written back after the store, and the store takes the old path.
// Every store that takes the old path anyway is counted by reason and reported. A managed
// write is published before it lands (MarkPreciseCpuWrite) and never after (MarkCpuWriteLanded).
//
// SHARPEMU_METAL_V2_PUBLISH_BEFORE_STORE=0 gives every store the old path.
internal sealed partial class MetalImageCache
{
    private const int StoreRequestCount = 64;
    private const int MaxImagesPerStoreRequest = 8;
    private const int MaxStorePublishRounds = 8;
    private const int StoreCompletionReportMilliseconds = 1000;
    private const int MaxStorePublishReports = 16;

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
        // The GPU kept writing the range for every round.
        Rounds,
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
        public string? Unwritten;
        public readonly ResourceSlotIdentifier[] Images = new ResourceSlotIdentifier[MaxImagesPerStoreRequest];
    }

    private readonly StoreRequest[] _storeRequests = CreateStoreRequests();
    private readonly object _storeRequestGate = new();
    private readonly long[] _storeFallbacks = new long[Enum.GetValues<StorePublishFallback>().Length];
    private int _storeRequestsPending;
    private Action? _serviceStoreRequests;
    private long _storePublishWritebacks;
    private long _storeCompletionStalls;
    private long _storeTakeMaxTicks;
    private int _storePublishReports;

    /// <summary>Whether a guest store into a GPU-written image waits for the image's writeback.</summary>
    internal bool PublishBeforeStore { get; set; } = PublishBeforeStoreDefault;

    /// <summary>How long a store waits for the worker to take its request before it calls the worker hung.</summary>
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
        Interlocked.Read(ref _storeFallbacks[(int)StorePublishFallback.Rounds]),
        Interlocked.Read(ref _storeCompletionStalls));

    internal readonly record struct StorePublishCounters(
        long Writebacks, long Worker, long Hung, long Full, long Failed, long Rounds, long CompletionStalls)
    {
        public long Fallbacks => Worker + Hung + Full + Failed + Rounds;
    }

    private string StorePublishCounterText()
    {
        var counts = StorePublishCounts;
        var takeMaxMs = Interlocked.Read(ref _storeTakeMaxTicks) * 1000.0 / Stopwatch.Frequency;
        return $" wb_before_store_total={counts.Writebacks} wb_before_store_fallbacks_total={counts.Fallbacks} " +
            $"(worker={counts.Worker} hung={counts.Hung} full={counts.Full} failed={counts.Failed} rounds={counts.Rounds}) " +
            $"wb_before_store_stalls_total={counts.CompletionStalls} wb_before_store_take_max_ms={takeMaxMs:F1}";
    }

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
    // the range is GPU-written, unless the store took the old path.
    private void PublishBeforeCpuStoreLocked(ulong address, ulong size, bool managedWrite)
    {
        if (!PublishBeforeStore || !WritebackFenceEnabled)
        {
            return;
        }

        for (var round = 0; !_watchingStopped; round++)
        {
            if (!HasGpuWrittenImageLocked(address, size))
            {
                return;
            }

            if (round == MaxStorePublishRounds)
            {
                NoteStoreFallback(StorePublishFallback.Rounds, address);
                return;
            }

            if (RequestStorePublishLocked(address, size) is not { } request)
            {
                NoteStoreFallback(StorePublishFallback.Full, address);
                return;
            }

            bool completed;
            bool handedOver;
            _lock.Exit();
            try
            {
                completed = AwaitStoreRequest(request, managedWrite, out handedOver);
            }
            finally
            {
                _lock.Enter();
            }

            var state = request.State;
            if (state == StoreRequestState.Pending)
            {
                // Withdrawn under the lock: the worker can no longer write it back after the store.
                FreeStoreRequestLocked(request);
                NoteStoreFallback(handedOver ? StorePublishFallback.Hung : StorePublishFallback.Worker, address);
                return;
            }

            if (!completed)
            {
                // Taken after the wait gave up on it: its writeback is recorded, and lands first.
                _lock.Exit();
                try
                {
                    WaitForStoreWriteback(request.Serial);
                }
                finally
                {
                    _lock.Enter();
                }
            }

            var unwritten = request.Unwritten;
            FreeStoreRequestLocked(request);
            if (state == StoreRequestState.Failed)
            {
                NoteStoreFallback(StorePublishFallback.Failed, address, unwritten);
                return;
            }
        }
    }

    private bool HasGpuWrittenImageLocked(ulong address, ulong size)
    {
        foreach (var imageIdentifier in FindImagesInRange(address, size, pageOverlap: true))
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
    private StoreRequest? RequestStorePublishLocked(ulong address, ulong size)
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
        foreach (var imageIdentifier in FindImagesInRange(address, size, pageOverlap: true))
        {
            if (count < MaxImagesPerStoreRequest && _slots.TryGet(imageIdentifier) is { IsGpuModified: true })
            {
                request.Images[count++] = imageIdentifier;
            }
        }

        request.Count = count;
        request.Serial = 0;
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
        request.Unwritten = null;
        Volatile.Write(ref request.State, StoreRequestState.Free);
    }

    // Outside the lock. True when the request was taken and its writeback has completed; false
    // when it is still pending (withdrawn by the caller) or was taken too late to be waited for here.
    private bool AwaitStoreRequest(StoreRequest request, bool managedWrite, out bool handedOver)
    {
        handedOver = false;
        if (_owner.IsWorkerThread)
        {
            // A managed write the worker makes itself, between packets: it can record the
            // writeback and wait for the GPU like any read of GPU-written bytes there. A fault
            // on the worker can be anywhere, so that store keeps its old path.
            if (!managedWrite)
            {
                return false;
            }

            handedOver = true;
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

        handedOver = true;
        var deadline = Environment.TickCount64 + StorePublishTakeTimeoutMilliseconds;
        lock (_storeRequestGate)
        {
            while (Volatile.Read(ref request.State) == StoreRequestState.Pending)
            {
                var remaining = deadline - Environment.TickCount64;
                if (remaining <= 0)
                {
                    return false;
                }

                Monitor.Wait(_storeRequestGate, (int)Math.Min(remaining, int.MaxValue));
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
                    if (_slots.TryGet(request.Images[index]) is { IsGpuModified: true } image)
                    {
                        WriteBackToGuest(image, MetalWritebackCause.HostRead);
                        wroteBack = true;
                        if (image.IsGpuModified)
                        {
                            // A writeback that could not be recorded leaves the image GPU-written.
                            failed = true;
                            request.Unwritten ??= ProfileLabel("writeback", in image.Description);
                            continue;
                        }

                        Interlocked.Increment(ref _storePublishWritebacks);
                    }
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

    private void NoteStoreFallback(StorePublishFallback reason, ulong address, string? image = null)
    {
        Interlocked.Increment(ref _storeFallbacks[(int)reason]);
        ReportStorePublish(
            $"A guest store into a GPU-written image took the old path ({reason}): address=0x{address:X16}" +
            (image != null ? $" unwritten={image}" : "") + ". " +
            "The image reloads from guest memory, which does not hold what the GPU wrote.");
    }

    private void ReportStorePublish(string message)
    {
        if (Interlocked.Increment(ref _storePublishReports) <= MaxStorePublishReports)
        {
            Console.Error.WriteLine($"[METAL-V2][WARN] {message}");
        }
    }
}
