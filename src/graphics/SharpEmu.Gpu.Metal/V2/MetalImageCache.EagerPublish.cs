// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using SharpEmu.HLE;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Images;

namespace SharpEmu.Gpu.Metal.V2;

// Publishing ahead of a store an image is known to take.
//
// A store into a GPU-written image waits for the image's writeback (MetalImageCache.CpuStores.cs),
// and that writeback is recorded when the store asks for it, so it completes behind every
// submission the worker committed before it. With the GPU seconds behind on a cold shader cache,
// Silent Hill's stores into its small render targets waited up to a second for that backlog,
// although the GPU had finished writing the image long before.
//
// So an image a store has needed written back is store-prone, and while it is, the commit after
// the GPU writes it records its writeback as well (MetalRenderHost.FlushCommandBuffer). The store
// that follows then usually finds the image already in guest memory and nothing to publish; one
// that comes while that writeback is still in flight waits for it as it would for its own:
//  - A writeback never lands after a store. One recorded ahead of a store is waited for without
//    a bound, apart from shutdown, by every store over its pages - faulting, managed, or into an
//    image released meanwhile - before the store's own publishing and again under the lock
//    that invalidates the image, since another one can be recorded in between. It is committed
//    as soon as it is recorded, so the wait needs nothing from the worker; on the worker itself
//    only a committed one can be waited for, and one that is not yet is counted.
//  - Nothing is published into memory the image no longer owns. The mark is the image's: it goes
//    with the image when it is released or evicted, and an image described anew is a new image.
//  - The cost is bounded. The mark decays after SHARPEMU_METAL_V2_EAGER_PUBLISH_FRAMES (60)
//    frames without a store into the image, and an image over
//    SHARPEMU_METAL_V2_EAGER_PUBLISH_MAX_MB (32) is written ahead only while a store came in this
//    frame or the last: a large pool is written back every frame only when it is stored into
//    every frame. At most MaxStoreProneImages are marked at once.
// The CHURN line counts the writebacks made ahead, their bytes, the ones the GPU overwrote before
// any store came (spent for nothing), the stores that found one, and the stores that waited for one.
//
// SHARPEMU_METAL_V2_EAGER_PUBLISH=0 writes nothing ahead.
internal sealed partial class MetalImageCache
{
    private const int MaxStoreProneImages = 64;

    private static readonly bool EagerPublishDefault =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_EAGER_PUBLISH") != "0";

    private static readonly int EagerPublishFramesDefault =
        int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_EAGER_PUBLISH_FRAMES"), out var frames) &&
        frames > 0
            ? frames
            : 60;

    private static readonly ulong EagerPublishMaxBytesDefault =
        (ulong.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_EAGER_PUBLISH_MAX_MB"), out var mb) && mb > 0
            ? mb
            : 32) << 20;

    // Under the lock; an image is listed exactly while its StoreFrame is not zero, or until a
    // pass finds it released.
    private readonly List<ResourceSlotIdentifier> _storeProneImages = [];
    // Under the fence gate: the writebacks recorded ahead whose submission has not completed.
    private readonly List<EagerWriteback> _eagerWritebacks = [];
    private int _storeProneCount;
    private int _eagerWritebackCount;
    // Under the lock: advanced by MarkFrame.
    private long _cacheFrame = 1;
    private long _eagerWritebacksRecorded;
    private long _eagerWritebackBytes;
    private long _eagerWritebacksOverwritten;
    private long _eagerStoreWaits;
    private long _eagerStoreWaitTicks;
    private long _eagerStoreWaitMaxTicks;
    private long _eagerStoreHits;
    private long _eagerUnwaited;

    private readonly record struct EagerWriteback(ulong Address, ulong Size, ulong Serial);

    /// <summary>Whether the commit after the GPU writes a store-prone image writes it back.</summary>
    internal bool EagerPublish { get; set; } = EagerPublishDefault;

    /// <summary>Frames without a store after which an image is no longer store-prone.</summary>
    internal int EagerPublishFrames { get; set; } = EagerPublishFramesDefault;

    /// <summary>Images larger than this are written ahead only while stored into every frame.</summary>
    internal ulong EagerPublishMaxBytes { get; set; } = EagerPublishMaxBytesDefault;

    /// <summary>Images marked store-prone now.</summary>
    internal int StoreProneImageCount => Volatile.Read(ref _storeProneCount);

    /// <summary>
    /// The writebacks recorded ahead of stores, their bytes, the ones overwritten by the GPU
    /// before any store, the stores that waited for one, and the waits the worker could not make.
    /// </summary>
    internal (long Writebacks, long Bytes, long Overwritten, long StoreWaits, long Unwaited) EagerPublishCounts =>
        (Interlocked.Read(ref _eagerWritebacksRecorded), Interlocked.Read(ref _eagerWritebackBytes),
            Interlocked.Read(ref _eagerWritebacksOverwritten), Interlocked.Read(ref _eagerStoreWaits),
            Interlocked.Read(ref _eagerUnwaited));

    private string EagerPublishCounterText()
    {
        var counts = EagerPublishCounts;
        return $" wb_eager_total={counts.Writebacks} wb_eager_mb_total={counts.Bytes / (1024.0 * 1024.0):F1} " +
            $"wb_eager_overwritten_total={counts.Overwritten} wb_eager_store_waits_total={counts.StoreWaits} " +
            $"wb_eager_store_wait_ms_total={Milliseconds(Interlocked.Read(ref _eagerStoreWaitTicks)):F1} " +
            $"wb_eager_store_wait_max_ms={Milliseconds(Interlocked.Read(ref _eagerStoreWaitMaxTicks)):F1} " +
            $"wb_eager_hits_total={Interlocked.Read(ref _eagerStoreHits)} " +
            $"wb_eager_unwaited_total={counts.Unwaited} wb_eager_prone={StoreProneImageCount}";
    }

    // A store needed this image written back. Caller holds the lock.
    private void NoteStorePublishedLocked(ResourceSlotIdentifier imageIdentifier, MetalCachedImage image)
    {
        if (!EagerPublish)
        {
            return;
        }

        if (image.StoreFrame == 0)
        {
            if (_storeProneImages.Count >= MaxStoreProneImages)
            {
                return;
            }

            _storeProneImages.Add(imageIdentifier);
            Volatile.Write(ref _storeProneCount, _storeProneImages.Count);
        }

        image.StoreFrame = _cacheFrame;
        image.EagerUnstored = false;
    }

    // A CPU store reached the image. Caller holds the lock.
    private void NoteStoreLocked(MetalCachedImage image)
    {
        if (image.StoreFrame != 0)
        {
            if (image.EagerUnstored)
            {
                Interlocked.Increment(ref _eagerStoreHits);
            }

            image.StoreFrame = _cacheFrame;
            image.EagerUnstored = false;
        }
    }

    /// <summary>
    /// On the worker at a commit, outside a preparation: records the writeback of every
    /// store-prone image the GPU has written since its last one, into the submission being
    /// committed. Skipped inside a cache call (the commit comes from it); the next commit
    /// records them.
    /// </summary>
    public void RecordEagerWritebacks()
    {
        if (Volatile.Read(ref _storeProneCount) == 0 || _lock.IsHeldByCurrentThread)
        {
            return;
        }

        using var held = _lock.Hold();
        for (var index = _storeProneImages.Count - 1; index >= 0; index--)
        {
            var image = _slots.TryGet(_storeProneImages[index]);
            if (image is null || !EagerPublish || !WritebackFenceEnabled || _watchingStopped ||
                _cacheFrame - image.StoreFrame > EagerPublishFrames)
            {
                if (image is not null)
                {
                    image.StoreFrame = 0;
                }

                _storeProneImages.RemoveAt(index);
                continue;
            }

            ref readonly var description = ref image.Description;
            if (!image.IsGpuModified || (description.Data.Size > EagerPublishMaxBytes && _cacheFrame - image.StoreFrame > 1))
            {
                continue;
            }

            if (WriteBackToGuest(image, MetalWritebackCause.Eager) == MetalWritebackOutcome.Failed)
            {
                continue;
            }

            if (image.EagerUnstored)
            {
                Interlocked.Increment(ref _eagerWritebacksOverwritten);
            }

            image.EagerUnstored = true;
            var serial = _owner.CurrentSerial;
            NoteEagerWriteback(description.Data.Address, description.Data.Size, serial);
            if (description.HasStencil)
            {
                NoteEagerWriteback(description.Stencil.Address, description.Stencil.Size, serial);
            }

            Interlocked.Increment(ref _eagerWritebacksRecorded);
            Interlocked.Add(ref _eagerWritebackBytes, (long)(description.Data.Size + description.Stencil.Size));
        }

        Volatile.Write(ref _storeProneCount, _storeProneImages.Count);
    }

    private void NoteEagerWriteback(ulong address, ulong size, ulong serial)
    {
        if (size == 0)
        {
            return;
        }

        lock (_fenceGate)
        {
            PruneEagerWritebacksLocked(_owner.CompletedSerial);
            _eagerWritebacks.Add(new EagerWriteback(address, size, serial));
            Volatile.Write(ref _eagerWritebackCount, _eagerWritebacks.Count);
        }
    }

    private void PruneEagerWritebacksLocked(ulong completed)
    {
        _eagerWritebacks.RemoveAll(entry => entry.Serial <= completed);
        Volatile.Write(ref _eagerWritebackCount, _eagerWritebacks.Count);
    }

    // The newest writeback recorded ahead over the range's pages that has not completed; zero for none.
    private ulong PendingEagerWriteback(ulong address, ulong size)
    {
        if (Volatile.Read(ref _eagerWritebackCount) == 0)
        {
            return 0;
        }

        ulong pending = 0;
        lock (_fenceGate)
        {
            PruneEagerWritebacksLocked(_owner.CompletedSerial);
            foreach (var entry in _eagerWritebacks)
            {
                if (entry.Serial > pending && GuestRangeOverlap.Pages(entry.Address, entry.Size, address, size))
                {
                    pending = entry.Serial;
                }
            }
        }

        return pending;
    }

    // Before a store into the range lands, without the lock: waits for the writebacks recorded
    // ahead over its pages. True when there was one.
    private bool WaitForEagerWritebacks(ulong address, ulong size)
    {
        var pending = PendingEagerWriteback(address, size);
        if (pending == 0)
        {
            return false;
        }

        WaitForEagerWriteback(pending);
        return true;
    }

    // The same with the lock held, which it lets go while it waits. True when it waited, after
    // which an image over the range may be GPU-written again and has to be published first.
    private bool WaitForEagerWritebacksLocked(ulong address, ulong size)
    {
        var pending = PendingEagerWriteback(address, size);
        if (pending == 0)
        {
            return false;
        }

        _lock.Exit();
        try
        {
            return WaitForEagerWriteback(pending);
        }
        finally
        {
            _lock.Enter();
        }
    }

    // False when it could not wait: on the worker, for one it has not committed yet, or at shutdown.
    private bool WaitForEagerWriteback(ulong serial)
    {
        var started = Stopwatch.GetTimestamp();
        var waited = true;
        if (!_owner.IsWorkerThread)
        {
            WaitForStoreWriteback(serial);
            waited = _owner.CompletedSerial >= serial;
        }
        else if (!_owner.WaitForCommittedSubmissionOnWorker(serial))
        {
            waited = false;
            Interlocked.Increment(ref _eagerUnwaited);
        }

        var ticks = Stopwatch.GetTimestamp() - started;
        Interlocked.Add(ref _eagerStoreWaitTicks, ticks);
        Interlocked.Increment(ref _eagerStoreWaits);
        for (var max = Interlocked.Read(ref _eagerStoreWaitMaxTicks);
             ticks > max && Interlocked.CompareExchange(ref _eagerStoreWaitMaxTicks, ticks, max) != max;
             max = Interlocked.Read(ref _eagerStoreWaitMaxTicks))
        {
        }
        return waited;
    }
}
