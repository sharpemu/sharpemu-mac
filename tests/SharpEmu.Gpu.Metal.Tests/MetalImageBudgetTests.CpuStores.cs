// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Abstractions;
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
        using var release = new ManualResetEventSlim();
        rig.CacheOwner.RunOnWorker = work =>
        {
            int count;
            lock (posted)
            {
                posted.Add(work);
                count = posted.Count;
            }

            // The other store is held here, its request made, until this one has given up: its
            // wait for the worker begins after the timeout is lowered and keeps its request's own.
            if (count == 1)
            {
                Assert.True(release.Wait(10_000));
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
        release.Set();
        // Far past the lowered timeout, which the other store's wait must not use.
        Thread.Sleep(200);

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

    [Fact]
    public void AStoreWhoseRequestIsTakenJustAfterItsWaitGaveUpStillWaitsForTheWriteback()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var gpu = AGpuWrittenImageOverStaleMemory(rig, GuestBase, 0x5A, out _);
        Thread? worker = null;
        using var recording = new ManualResetEventSlim();
        rig.CacheOwner.RunOnWorker = work =>
        {
            worker = new Thread(() => work());
            worker.Start();
            Assert.True(recording.Wait(10_000));
            return true;
        };
        rig.CacheOwner.OnBeginTransfer = () =>
        {
            // The worker has taken the request and records its writeback, under the cache
            // lock, while the store's wait for it gives up.
            recording.Set();
            Thread.Sleep(200);
        };
        rig.CacheOwner.WaitOffWorker = SubmitAndComplete(rig);
        rig.Cache.StorePublishTakeTimeoutMilliseconds = 0;

        Assert.True(((IGuestImageStore)rig.Cache).MarkCpuWrite(GuestBase + StoreOffset, 8));
        Assert.True(worker!.Join(10_000));
        Assert.Equal((1L, 0L), (rig.Cache.StorePublishCounts.Writebacks, rig.Cache.StorePublishCounts.Fallbacks));
        Assert.Equal(gpu, rig.ReadGuest(GuestBase, ImageSize));
    }

    [Fact]
    public void AShadowMapScatteredOverTheBackingIsWrittenBackBeforeAStore()
    {
        if (!RequireDevice()) return;
        var size = D16TargetRequest().Description.Data.Size;
        var whole = ShadowMapAfterAStore(layout: null, out _, out _);
        Assert.NotEqual(MetalTileOracle.Pattern((int)size, 0xB2), whole);
        var scattered = ShadowMapAfterAStore(
            backing =>
            {
                backing.ScatterStart = ShadowMap;
                backing.ScatterBytes = size;
            },
            out var carry, out var counts);

        // Its one slice is in 64 KB pieces no window holds together: staged, and carried whole.
        Assert.Equal((1L, 0L), (counts.Writebacks, counts.Fallbacks));
        Assert.True(carry.StagedRuns > 0);
    }

    [Fact]
    public void AWritebackThatLeavesBytesBehindIsAPartialFallback()
    {
        if (!RequireDevice()) return;
        const ulong hole = 0x2000;
        using var rig = new Rig(Device.Value, layout: backing =>
        {
            backing.HoleStart = GuestBase + hole;
            backing.HoleBytes = 0x1000;
        });
        var gpu = AGpuWrittenImageOverStaleMemory(rig, GuestBase, 0x5A, out var identifier);
        var stale = rig.ReadGuest(GuestBase, ImageSize);
        rig.CacheOwner.RunOnWorker = work =>
        {
            work();
            return true;
        };
        rig.CacheOwner.WaitOffWorker = SubmitAndComplete(rig);

        Assert.True(((IGuestImageStore)rig.Cache).MarkCpuWrite(GuestBase + StoreOffset, 8));
        Assert.Equal((1L, 1L), (rig.Cache.StorePublishCounts.Writebacks, rig.Cache.StorePublishCounts.Partial));
        Assert.Equal((1L, 1L), (rig.Cache.StorePublishCounts.Fallbacks, rig.Cache.WritebackCarryCounts.Partial));
        Assert.False(rig.Cache.GetImage(identifier).IsGpuModified);

        // Every byte a window holds is the GPU's; the page no window holds keeps the old ones.
        var expected = (byte[])gpu.Clone();
        Array.Copy(stale, (int)hole, expected, (int)hole, 0x1000);
        Assert.Equal(expected, rig.ReadGuest(GuestBase, ImageSize));
    }

    [Fact]
    public void APartialWritebackLetsTheRoundsGoOnForWhatTheGpuWritesMeanwhile()
    {
        if (!RequireDevice()) return;
        const ulong hole = 0x2000;
        using var rig = new Rig(Device.Value, layout: backing =>
        {
            backing.HoleStart = GuestBase + hole;
            backing.HoleBytes = 0x1000;
        });
        _ = AGpuWrittenImageOverStaleMemory(rig, GuestBase, 0x5A, out var identifier);
        var stale = rig.ReadGuest(GuestBase, ImageSize);
        var rewritten = MetalTileOracle.Pattern((int)ImageSize, 0x2B);
        var takes = 0;
        rig.CacheOwner.RunOnWorker = work =>
        {
            work();
            if (takes++ == 0)
            {
                // Written again by the GPU while the store waits for the first, partial writeback.
                rig.Harness.Submit();
                rig.Harness.Begin();
                rig.WriteGuest(GuestBase, rewritten);
                rig.Cache.InvalidateMemory(GuestBase, ImageSize);
                Assert.Equal(identifier, rig.Acquire(GuestBase, ImageRole.ColorTarget));
                rig.Harness.Submit();
                rig.Harness.Begin();
                rig.WriteGuest(GuestBase, stale);
            }

            return true;
        };
        rig.CacheOwner.WaitOffWorker = SubmitAndComplete(rig);

        Assert.True(((IGuestImageStore)rig.Cache).MarkCpuWrite(GuestBase + StoreOffset, 8));
        Assert.Equal(2, takes);
        Assert.Equal((2L, 1L), (rig.Cache.StorePublishCounts.Writebacks, rig.Cache.StorePublishCounts.Partial));
        Assert.Equal((1L, 2L), (rig.Cache.StorePublishCounts.Fallbacks, rig.Cache.WritebackCarryCounts.Partial));
        Assert.False(rig.Cache.GetImage(identifier).IsGpuModified);
        var expected = (byte[])rewritten.Clone();
        Array.Copy(stale, (int)hole, expected, (int)hole, 0x1000);
        Assert.Equal(expected, rig.ReadGuest(GuestBase, ImageSize));
    }

    [Fact]
    public void AStagedWritebackKeepsTheGuestBytesTheRetileDoesNotWrite()
    {
        if (!RequireDevice()) return;
        var description = Request(MipChain, 0).Description;
        var whole = MipChainAfterAStore(layout: null, out var stale, out _);
        var scattered = MipChainAfterAStore(
            backing =>
            {
                backing.ScatterStart = MipChain;
                backing.ScatterBytes = description.Data.Size;
                backing.ScatterChunk = 0x1000;
            },
            out _, out var carry);

        // No 64 KB block fits a 4 KB piece, so every run is staged; the slack in the mip
        // tail is no texel's, and keeps what guest memory held.
        Assert.True(carry.StagedRuns > 0);
        Assert.Equal((0L, 0L), (carry.Partial, carry.Unrecorded));
        Assert.Equal(whole, scattered);
        var covered = MetalImageTransferTests.FidelityCoverage(in description, 0).Data;
        var slack = Enumerable.Range(0, covered.Length).Where(i => !covered[i]).ToArray();
        Assert.NotEmpty(slack);
        Assert.True(slack.All(i => scattered[i] == stale[i]));
        Assert.NotEqual(stale, scattered);
    }

    [Fact]
    public void ABackedManagedWritePublishesOnlyTheImagesOverItsBytes()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        // Two images sharing the page 0x4000-0x5000 without sharing a byte.
        var written = AGpuWrittenImageOverStaleMemory(rig, GuestBase + 0x800, 0x5A, out var first);
        _ = AGpuWrittenImageOverStaleMemory(rig, GuestBase + 0x4800, 0x6B, out var second);
        rig.CacheOwner.RunOnWorker = work =>
        {
            work();
            return true;
        };
        rig.CacheOwner.WaitOffWorker = SubmitAndComplete(rig);

        Assert.True(((IGuestImageStore)rig.Cache).MarkPreciseCpuWrite(GuestBase + 0x4700, 8));
        Assert.Equal((1L, 0L), (rig.Cache.StorePublishCounts.Writebacks, rig.Cache.StorePublishCounts.Fallbacks));
        Assert.Equal(written, rig.ReadGuest(GuestBase + 0x800, ImageSize));
        Assert.False(rig.Cache.GetImage(first).IsGpuModified);
        Assert.True(rig.Cache.GetImage(second).IsGpuModified);
    }

    private const ulong MipChain = GuestBase + 0x100000;

    // The guest bytes under a GPU-written 256² RGBA8 nine-level chain in 64 KB tiles once a
    // store into it has landed, its backing laid out as given; and what guest memory held
    // before the writeback.
    private byte[] MipChainAfterAStore(
        Action<Backing>? layout, out byte[] stale, out (long StagedRuns, long Partial, long Unrecorded) carry)
    {
        using var rig = new Rig(Device.Value, layout: layout);
        var (identifier, _) = AcquireMipped(rig, MipChain, 0, 0x3D);
        rig.Cache.MarkGpuWritten(identifier);
        rig.Harness.Submit();
        rig.Harness.Begin();
        var size = (int)rig.Cache.GetImage(identifier).Description.Data.Size;
        stale = MetalTileOracle.Pattern(size, 0xC2);
        rig.WriteGuest(MipChain, stale);
        rig.CacheOwner.RunOnWorker = work =>
        {
            work();
            return true;
        };
        rig.CacheOwner.WaitOffWorker = SubmitAndComplete(rig);

        Assert.True(((IGuestImageStore)rig.Cache).MarkCpuWrite(MipChain + StoreOffset, 8));
        Assert.Equal((1L, 0L), (rig.Cache.StorePublishCounts.Writebacks, rig.Cache.StorePublishCounts.Fallbacks));
        carry = rig.Cache.WritebackCarryCounts;
        return rig.ReadGuest(MipChain, (ulong)size);
    }

    private const ulong ShadowMap = GuestBase + 0x100000;
    private const uint ShadowMapSide = 1024;

    // The guest bytes under a GPU-written 1024² D16 shadow map once a store into it has
    // landed, its backing laid out as given.
    private byte[] ShadowMapAfterAStore(
        Action<Backing>? layout, out (long StagedRuns, long Partial, long Unrecorded) carry,
        out MetalImageCache.StorePublishCounters counts)
    {
        using var rig = new Rig(Device.Value, layout: layout);
        var request = D16TargetRequest();
        var size = request.Description.Data.Size;
        rig.WriteGuest(ShadowMap, MetalTileOracle.Pattern((int)size, 0x4D));
        var identifier = rig.Cache.FindImage(ref request);
        rig.Cache.AcquireDepthTargetView(identifier, in request);
        Assert.True(rig.Cache.GetImage(identifier).IsGpuModified);
        rig.Harness.Submit();
        rig.Harness.Begin();
        rig.WriteGuest(ShadowMap, MetalTileOracle.Pattern((int)size, 0xB2));
        rig.CacheOwner.RunOnWorker = work =>
        {
            work();
            return true;
        };
        rig.CacheOwner.WaitOffWorker = SubmitAndComplete(rig);

        Assert.True(((IGuestImageStore)rig.Cache).MarkCpuWrite(ShadowMap + StoreOffset, 8));
        carry = rig.Cache.WritebackCarryCounts;
        counts = rig.Cache.StorePublishCounts;
        return rig.ReadGuest(ShadowMap, size);
    }

    private static ImageRequest D16TargetRequest()
    {
        Assert.True(TileGeometry.TryGetDepthSize(ShadowMapSide, ShadowMapSide, 0, GuestDepthFormat.Z16,
            GuestStencilFormat.Invalid, false, out _, out _, out var depthSize));
        var size = (ulong)depthSize.Size;
        var pitch = TileGeometry.DepthPitch(ShadowMapSide, sizeof(ushort));
        var description = ImageDescription.Create();
        description.Data = new GuestSpan(ShadowMap, size);
        description.PixelFormat = GpuFormat.D16Unorm;
        description.GuestFormat = DepthFormatRule.Find(GuestDepthFormat.Z16)!.GuestFormat;
        description.Type = GuestImageType.Color2D;
        description.Extent = new GpuExtent3D(ShadowMapSide, ShadowMapSide, 1);
        description.Resources = new SubresourceCount(1, 1);
        description.Pitch = pitch;
        description.BytesPerBlock = sizeof(ushort);
        description.Samples = 1;
        description.TileMode = GuestTileMode.Depth;
        description.MipLayout[0] = new MipLevelLayout { Offset = 0, Size = size, Pitch = pitch, Height = ShadowMapSide };
        var view = ImageViewDescription.Default with
        {
            Format = GpuFormat.D16Unorm,
            Type = GpuImageViewType.Type2D,
            Aspect = GpuImageAspect.DepthBit,
            LayerCount = 1,
            Usage = GpuImageUsage.DepthStencilAttachmentBit,
        };
        return new ImageRequest(description, view, ImageRole.DepthTarget);
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
