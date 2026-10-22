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

    [Fact]
    public void AGuestStoreIntoAGpuWrittenImageLandsOverTheGpusBytes()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var gpu = AGpuWrittenImageOverStaleMemory(rig, out var identifier);
        var worker = 0;
        rig.CacheOwner.RunOnWorker = work =>
        {
            worker++;
            work();
            return true;
        };
        rig.CacheOwner.WaitOffWorker = (_, _) =>
        {
            rig.Harness.Submit();
            rig.Harness.Begin();
            return true;
        };

        // The store waits for the image's writeback, so guest memory holds the GPU's bytes...
        Assert.True(((IGuestImageStore)rig.Cache).MarkCpuWrite(GuestBase + StoreOffset, 8));
        Assert.Equal(1, worker);
        Assert.Equal((1L, 0L), rig.Cache.StorePublishCounts);
        Assert.Equal(gpu, rig.ReadGuest(GuestBase, ImageSize));

        // ...and once it has landed over them, the texture takes both.
        var store = new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        rig.WriteGuest(GuestBase + StoreOffset, store);
        store.CopyTo(gpu, (int)StoreOffset);
        Assert.Equal(gpu, TextureBytes(rig, identifier));
    }

    [Fact]
    public void AStoreTheWorkerCannotTakeGivesTheImageToGuestMemory()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var gpu = AGpuWrittenImageOverStaleMemory(rig, out var identifier);
        var stale = rig.ReadGuest(GuestBase, ImageSize);

        Assert.True(((IGuestImageStore)rig.Cache).MarkCpuWrite(GuestBase + StoreOffset, 8));
        Assert.Equal((0L, 0L), rig.Cache.StorePublishCounts);
        Assert.False(rig.Cache.GetImage(identifier).IsGpuModified);
        Assert.Equal(stale, rig.ReadGuest(GuestBase, ImageSize));
        Assert.NotEqual(gpu, stale);
    }

    [Fact]
    public void ARequestTheWorkerTakesTooLateWritesNothingAfterTheStore()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        _ = AGpuWrittenImageOverStaleMemory(rig, out var identifier);
        var stale = rig.ReadGuest(GuestBase, ImageSize);
        Action? late = null;
        rig.CacheOwner.RunOnWorker = work =>
        {
            late = work;
            return true;
        };
        rig.Cache.WritebackFenceTimeoutMilliseconds = 1;

        Assert.True(((IGuestImageStore)rig.Cache).MarkCpuWrite(GuestBase + StoreOffset, 8));
        Assert.Equal((0L, 1L), rig.Cache.StorePublishCounts);
        Assert.False(rig.Cache.GetImage(identifier).IsGpuModified);

        // The store landed; the worker getting round to the request now must not write over it.
        var store = new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        rig.WriteGuest(GuestBase + StoreOffset, store);
        store.CopyTo(stale, (int)StoreOffset);
        Assert.NotNull(late);
        late();
        rig.Harness.Submit();
        rig.Harness.Begin();
        Assert.Equal((0L, 1L), rig.Cache.StorePublishCounts);
        Assert.Equal(stale, rig.ReadGuest(GuestBase, ImageSize));
    }

    // A texture the "GPU" wrote over guest memory that never got its bytes.
    private static byte[] AGpuWrittenImageOverStaleMemory(Rig rig, out ResourceSlotIdentifier identifier)
    {
        var gpu = MetalTileOracle.Pattern((int)ImageSize, 0x5A);
        rig.WriteGuest(GuestBase, gpu);
        identifier = rig.Acquire(GuestBase, ImageRole.ColorTarget);
        Assert.True(rig.Cache.GetImage(identifier).IsGpuModified);
        rig.Harness.Submit();
        rig.Harness.Begin();
        rig.WriteGuest(GuestBase, MetalTileOracle.Pattern((int)ImageSize, 0xC3));
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
