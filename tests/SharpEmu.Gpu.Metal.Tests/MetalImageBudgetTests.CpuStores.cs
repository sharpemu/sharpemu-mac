// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Images;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// A guest store into an image whose bytes only its texture holds changes the bytes it
/// writes and nothing else (MetalImageCache.CpuStores.cs). Silent Hill's reflection-cube
/// pool lost everything the GPU had filtered into it to one such store, and every
/// reflection after it sampled NaN.
/// </summary>
public sealed partial class MetalImageBudgetTests
{
    private const ulong StoreOffset = 0x1040;
    private const ulong OtherImage = GuestBase + 0x20000;

    [Fact]
    public void AGuestStoreIntoAGpuWrittenImageLandsOverTheGpusBytes()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var gpu = AGpuWrittenImageOverStaleMemory(rig, GuestBase, 0x5A, out var identifier);
        var worker = 0;
        rig.CacheOwner.RunOnWorker = work =>
        {
            worker++;
            work();
            return true;
        };
        rig.CacheOwner.WaitOffWorker = SubmitAndComplete(rig);

        // The store waits for the image's writeback, so guest memory holds the GPU's bytes...
        Assert.True(((IGuestImageStore)rig.Cache).MarkCpuWrite(GuestBase + StoreOffset, 8));
        Assert.Equal(1, worker);
        Assert.Equal((1L, 0L), (rig.Cache.StorePublishCounts.Writebacks, rig.Cache.StorePublishCounts.Fallbacks));
        Assert.Equal(gpu, rig.ReadGuest(GuestBase, ImageSize));

        // ...and once it has landed over them, the texture takes both.
        var store = new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        rig.WriteGuest(GuestBase + StoreOffset, store);
        store.CopyTo(gpu, (int)StoreOffset);
        Assert.Equal(gpu, TextureBytes(rig, identifier));
    }

    [Fact]
    public void AStoreWithNoWorkerToTakeItGivesTheImageToGuestMemory()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var gpu = AGpuWrittenImageOverStaleMemory(rig, GuestBase, 0x5A, out var identifier);
        var stale = rig.ReadGuest(GuestBase, ImageSize);

        Assert.True(((IGuestImageStore)rig.Cache).MarkCpuWrite(GuestBase + StoreOffset, 8));
        Assert.Equal((0L, 1L), (rig.Cache.StorePublishCounts.Writebacks, rig.Cache.StorePublishCounts.Worker));
        Assert.False(rig.Cache.GetImage(identifier).IsGpuModified);
        Assert.Equal(stale, rig.ReadGuest(GuestBase, ImageSize));
        Assert.NotEqual(gpu, stale);
    }

    [Fact]
    public void ARequestTheWorkerTakesTooLateWritesNothingAfterTheStore()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        _ = AGpuWrittenImageOverStaleMemory(rig, GuestBase, 0x5A, out var identifier);
        var stale = rig.ReadGuest(GuestBase, ImageSize);
        Action? late = null;
        rig.CacheOwner.RunOnWorker = work =>
        {
            late = work;
            return true;
        };
        rig.Cache.StorePublishTakeTimeoutMilliseconds = 1;

        Assert.True(((IGuestImageStore)rig.Cache).MarkCpuWrite(GuestBase + StoreOffset, 8));
        Assert.Equal((0L, 1L), (rig.Cache.StorePublishCounts.Writebacks, rig.Cache.StorePublishCounts.Hung));
        Assert.Equal(0, rig.Cache.PendingStoreRequests);
        Assert.False(rig.Cache.GetImage(identifier).IsGpuModified);

        // The store landed; the worker getting round to the request now must not write over it.
        var store = new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        rig.WriteGuest(GuestBase + StoreOffset, store);
        store.CopyTo(stale, (int)StoreOffset);
        Assert.NotNull(late);
        late();
        rig.Harness.Submit();
        rig.Harness.Begin();
        Assert.Equal(0L, rig.Cache.StorePublishCounts.Writebacks);
        Assert.Equal(stale, rig.ReadGuest(GuestBase, ImageSize));
    }

    [Fact]
    public void AStoreGivingUpOnItsRequestLeavesAnotherStoresRequestPending()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        _ = AGpuWrittenImageOverStaleMemory(rig, GuestBase, 0x5A, out var first);
        var other = AGpuWrittenImageOverStaleMemory(rig, OtherImage, 0x6B, out _);
        var stale = rig.ReadGuest(GuestBase, ImageSize);
        var posted = new List<Action>();
        rig.CacheOwner.RunOnWorker = work =>
        {
            lock (posted)
            {
                posted.Add(work);
            }

            return true;
        };
        rig.CacheOwner.WaitOffWorker = (serial, timeout) =>
            SpinWait.SpinUntil(() => rig.CacheOwner.CompletedSerial >= serial, timeout);

        // Another thread's store waits for the worker...
        rig.Cache.StorePublishTakeTimeoutMilliseconds = 30_000;
        var otherStored = false;
        var otherThread = new Thread(() =>
            otherStored = ((IGuestImageStore)rig.Cache).MarkCpuWrite(OtherImage + StoreOffset, 8));
        otherThread.Start();
        Assert.True(SpinWait.SpinUntil(() =>
        {
            lock (posted)
            {
                return posted.Count == 1;
            }
        }, 10_000));

        // ...while this one gives up on its own request and takes the old path.
        rig.Cache.StorePublishTakeTimeoutMilliseconds = 1;
        Assert.True(((IGuestImageStore)rig.Cache).MarkCpuWrite(GuestBase + StoreOffset, 8));
        Assert.Equal(1L, rig.Cache.StorePublishCounts.Hung);
        Assert.False(rig.Cache.GetImage(first).IsGpuModified);

        // The worker taking the first request still publishes the other store's image, and only that.
        posted[0]();
        rig.Harness.Submit();
        rig.Harness.Begin();
        Assert.True(otherThread.Join(10_000));
        Assert.True(otherStored);
        Assert.Equal((1L, 1L), (rig.Cache.StorePublishCounts.Writebacks, rig.Cache.StorePublishCounts.Fallbacks));
        Assert.Equal(other, rig.ReadGuest(OtherImage, ImageSize));
        Assert.Equal(stale, rig.ReadGuest(GuestBase, ImageSize));
    }

    [Fact]
    public void AnImageTheGpuWritesWhileTheStoreWaitsIsPublishedAgain()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        _ = AGpuWrittenImageOverStaleMemory(rig, GuestBase, 0x5A, out var identifier);
        var rewritten = MetalTileOracle.Pattern((int)ImageSize, 0x2B);
        var takes = 0;
        rig.CacheOwner.RunOnWorker = work =>
        {
            work();
            if (takes++ == 0)
            {
                // The worker goes on recording once it has taken the request, and draws into
                // the image again: its texture now holds bytes guest memory has never seen.
                rig.Harness.Submit();
                rig.Harness.Begin();
                rig.WriteGuest(GuestBase, rewritten);
                rig.Cache.InvalidateMemory(GuestBase, ImageSize);
                Assert.Equal(identifier, rig.Acquire(GuestBase, ImageRole.ColorTarget));
                rig.Harness.Submit();
                rig.Harness.Begin();
                rig.WriteGuest(GuestBase, MetalTileOracle.Pattern((int)ImageSize, 0x3C));
            }

            return true;
        };
        rig.CacheOwner.WaitOffWorker = SubmitAndComplete(rig);

        Assert.True(((IGuestImageStore)rig.Cache).MarkCpuWrite(GuestBase + StoreOffset, 8));
        Assert.Equal(2, takes);
        Assert.Equal((2L, 0L), (rig.Cache.StorePublishCounts.Writebacks, rig.Cache.StorePublishCounts.Fallbacks));
        Assert.False(rig.Cache.GetImage(identifier).IsGpuModified);
        Assert.Equal(rewritten, rig.ReadGuest(GuestBase, ImageSize));
    }

    [Fact]
    public void AStoreWaitsForAWritebackTheGpuIsSlowToComplete()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var gpu = AGpuWrittenImageOverStaleMemory(rig, GuestBase, 0x5A, out _);
        rig.CacheOwner.RunOnWorker = work =>
        {
            work();
            return true;
        };
        var refusals = 0;
        var complete = SubmitAndComplete(rig);
        rig.CacheOwner.WaitOffWorker = (serial, timeout) => refusals++ >= 3 && complete(serial, timeout);

        Assert.True(((IGuestImageStore)rig.Cache).MarkCpuWrite(GuestBase + StoreOffset, 8));
        Assert.Equal(1L, rig.Cache.StorePublishCounts.CompletionStalls);
        Assert.Equal(0L, rig.Cache.StorePublishCounts.Fallbacks);
        Assert.Equal(gpu, rig.ReadGuest(GuestBase, ImageSize));
    }

    [Fact]
    public void AManagedWriteIsPublishedBeforeItLandsAndNeverAfter()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var gpu = AGpuWrittenImageOverStaleMemory(rig, GuestBase, 0x5A, out var identifier);
        rig.CacheOwner.RunOnWorker = work =>
        {
            work();
            return true;
        };
        rig.CacheOwner.WaitOffWorker = SubmitAndComplete(rig);
        IGuestImageStore store = rig.Cache;

        Assert.True(store.MarkPreciseCpuWrite(GuestBase + StoreOffset, 8));
        Assert.Equal(1L, rig.Cache.StorePublishCounts.Writebacks);
        Assert.Equal(gpu, rig.ReadGuest(GuestBase, ImageSize));

        // Written by the GPU again before the write is reported landed: nothing goes under it now.
        Assert.Equal(identifier, rig.Acquire(GuestBase, ImageRole.ColorTarget));
        rig.Harness.Submit();
        rig.Harness.Begin();
        var landed = rig.ReadGuest(GuestBase, ImageSize);
        Assert.True(store.MarkCpuWriteLanded(GuestBase + StoreOffset, 8));
        rig.Harness.Submit();
        rig.Harness.Begin();
        Assert.Equal(1L, rig.Cache.StorePublishCounts.Writebacks);
        Assert.Equal(landed, rig.ReadGuest(GuestBase, ImageSize));
    }

    [Fact]
    public void AWritebackThatCannotBeRecordedIsNotCountedAsOne()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        _ = AGpuWrittenImageOverStaleMemory(rig, GuestBase, 0x5A, out var identifier);
        rig.CacheOwner.RunOnWorker = work =>
        {
            work();
            return true;
        };
        rig.CacheOwner.WaitOffWorker = SubmitAndComplete(rig);
        rig.GuestBacking.NoWindows = true;

        Assert.True(((IGuestImageStore)rig.Cache).MarkCpuWrite(GuestBase + StoreOffset, 8));
        Assert.Equal((0L, 1L), (rig.Cache.StorePublishCounts.Writebacks, rig.Cache.StorePublishCounts.Failed));
        Assert.Equal(0, rig.Cache.PendingStoreRequests);
        Assert.False(rig.Cache.GetImage(identifier).IsGpuModified);
    }

    private static Func<ulong, int, bool> SubmitAndComplete(Rig rig) => (_, _) =>
    {
        rig.Harness.Submit();
        rig.Harness.Begin();
        return true;
    };

    // A texture the "GPU" wrote over guest memory that never got its bytes.
    private static byte[] AGpuWrittenImageOverStaleMemory(Rig rig, ulong address, uint seed, out ResourceSlotIdentifier identifier)
    {
        var gpu = MetalTileOracle.Pattern((int)ImageSize, seed);
        rig.WriteGuest(address, gpu);
        identifier = rig.Acquire(address, ImageRole.ColorTarget);
        Assert.True(rig.Cache.GetImage(identifier).IsGpuModified);
        rig.Harness.Submit();
        rig.Harness.Begin();
        rig.WriteGuest(address, MetalTileOracle.Pattern((int)ImageSize, seed ^ 0xFF));
        return gpu;
    }

    // What the texture holds: bound as a target again, then retired and written back over
    // guest memory scribbled with something else.
    private static byte[] TextureBytes(Rig rig, ResourceSlotIdentifier identifier)
    {
        Assert.Equal(identifier, rig.Acquire(GuestBase, ImageRole.ColorTarget));
        rig.Harness.Submit();
        rig.Harness.Begin();
        rig.WriteGuest(GuestBase, MetalTileOracle.Pattern((int)ImageSize, 0x77));
        rig.Configure(retireFrames: 2, retireSeconds: 1);
        for (var frame = 0; frame < 4 && rig.Cache.TryGetImage(identifier, out _); frame++)
        {
            rig.Frame(seconds: 1, submit: false);
        }

        Assert.False(rig.Cache.TryGetImage(identifier, out _));
        rig.Harness.Submit();
        rig.Harness.Begin();
        return rig.ReadGuest(GuestBase, ImageSize);
    }
}
