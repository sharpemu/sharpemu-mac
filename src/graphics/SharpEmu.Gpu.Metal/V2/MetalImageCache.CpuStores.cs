// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

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
// So the faulting thread first has the GPU worker write the image back, and waits for that
// writeback to complete. Guest memory then holds the GPU's bytes, the store lands over them, and
// the reload gives the texture both. A writeback must never land after the store: the game
// reuses a transient image's memory for its own data, and GPU bytes landing late over a heap
// header or a command buffer corrupt it. The wait for the worker to take the request is bounded
// by the writeback fence's timeout (MetalImageCache.WritebackFence.cs); a request it does not
// take in time (the worker stuck behind this very thread), or cannot be given at all (the
// worker itself storing), is withdrawn under the lock and leaves the store the old path. A writeback the worker has taken is committed, so only the GPU stands between it and
// the store, and that wait is allowed longer.
//
// SHARPEMU_METAL_V2_PUBLISH_BEFORE_STORE=0 gives every faulting store the old path.
internal sealed partial class MetalImageCache
{
    private const int MaxStorePublishes = 16;
    private const int StorePublishCompletionMilliseconds = 5000;

    private static readonly bool PublishBeforeStoreDefault =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_PUBLISH_BEFORE_STORE") != "0";

    // Images a store is waiting on, filled under the lock without allocating. The request
    // counts and the serial change under the lock; the gate only wakes the waiting stores.
    private readonly ResourceSlotIdentifier[] _storePublishes = new ResourceSlotIdentifier[MaxStorePublishes];
    private readonly object _storePublishGate = new();
    private int _storePublishCount;
    private ulong _storePublishRequested;
    private ulong _storePublishTaken;
    private ulong _storePublishSerial;
    private Action? _serviceStorePublishes;
    private long _storePublishWritebacks;
    private long _storePublishTimeouts;

    /// <summary>Whether a guest store into a GPU-written image waits for the image's writeback.</summary>
    internal bool PublishBeforeStore { get; set; } = PublishBeforeStoreDefault;

    /// <summary>Writebacks recorded for faulting guest stores, and stores that stopped waiting for one.</summary>
    internal (long Writebacks, long Timeouts) StorePublishCounts =>
        (Interlocked.Read(ref _storePublishWritebacks), Interlocked.Read(ref _storePublishTimeouts));

    // On the faulting thread, before the lock is taken for the store.
    private void PublishBeforeCpuStore(ulong address, ulong size)
    {
        if (!PublishBeforeStore || !WritebackFenceEnabled)
        {
            return;
        }

        ulong ticket;
        using (var held = _lock.Hold())
        {
            if (_watchingStopped || !RequestStorePublishesLocked(address, size))
            {
                return;
            }

            ticket = _storePublishRequested;
        }

        if (!_owner.TryRunOnWorker(_serviceStorePublishes ??= ServiceStorePublishes))
        {
            using var held = _lock.Hold();
            _storePublishCount = 0;
            return;
        }

        WaitForStorePublishes(ticket);
        ulong serial;
        using (var held = _lock.Hold())
        {
            if (_storePublishTaken < ticket)
            {
                // Not taken in time: withdrawn, so no writeback can be recorded after the store.
                _storePublishCount = 0;
                Interlocked.Increment(ref _storePublishTimeouts);
                return;
            }

            serial = _storePublishSerial;
        }

        // The writeback is committed: only the GPU is left to wait for.
        if (!_owner.WaitForSubmissionOffWorker(serial, StorePublishCompletionMilliseconds))
        {
            Interlocked.Increment(ref _storePublishTimeouts);
        }

        if (Volatile.Read(ref _fencedWritebackCount) != 0)
        {
            WaitForFencedWritebacks(address, size);
        }
    }

    private bool RequestStorePublishesLocked(ulong address, ulong size)
    {
        var requested = false;
        foreach (var imageIdentifier in FindImagesInRange(address, size, pageOverlap: true))
        {
            if (_slots.TryGet(imageIdentifier) is not { IsGpuModified: true } image || image.Texture == 0)
            {
                continue;
            }

            if (Array.IndexOf(_storePublishes, imageIdentifier, 0, _storePublishCount) < 0)
            {
                if (_storePublishCount == MaxStorePublishes)
                {
                    continue;
                }

                _storePublishes[_storePublishCount++] = imageIdentifier;
            }

            requested = true;
        }

        if (requested)
        {
            _storePublishRequested++;
        }

        return requested;
    }

    private void WaitForStorePublishes(ulong ticket)
    {
        var deadline = Environment.TickCount64 + WritebackFenceTimeoutMilliseconds;
        lock (_storePublishGate)
        {
            while (Volatile.Read(ref _storePublishTaken) < ticket)
            {
                var remaining = deadline - Environment.TickCount64;
                if (remaining <= 0)
                {
                    return;
                }

                Monitor.Wait(_storePublishGate, (int)remaining);
            }
        }
    }

    // On the GPU worker: every requested image still GPU-written goes back to guest memory.
    private void ServiceStorePublishes()
    {
        using (var held = _lock.Hold())
        {
            for (var index = 0; index < _storePublishCount; index++)
            {
                if (_slots.TryGet(_storePublishes[index]) is { IsGpuModified: true } image)
                {
                    WriteBackToGuest(image, MetalWritebackCause.HostRead);
                    _storePublishSerial = Math.Max(_storePublishSerial, _owner.CurrentSerial);
                    Interlocked.Increment(ref _storePublishWritebacks);
                }
            }

            _storePublishCount = 0;
            Volatile.Write(ref _storePublishTaken, _storePublishRequested);
        }

        lock (_storePublishGate)
        {
            Monitor.PulseAll(_storePublishGate);
        }
    }
}
