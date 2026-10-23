// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Images;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// An image a guest store has needed written back is written back at the commit after the GPU
/// writes it again, so the next store finds it in guest memory instead of waiting behind the
/// GPU's backlog (MetalImageCache.EagerPublish.cs).
/// </summary>
public sealed partial class MetalImageBudgetTests
{
    [Fact]
    public void AStoreProneImageIsWrittenBackAtTheCommitAfterTheGpuWritesIt()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var identifier = AStoreProneImage(rig);
        // An image no store has needed written back stays the GPU's.
        _ = AGpuWrittenImageOverStaleMemory(rig, OtherImage, 0x44, out var other);
        var rewritten = TheGpuWritesItAgain(rig, identifier, 0x2B);

        rig.Cache.RecordEagerWritebacks();
        rig.Harness.Submit();
        rig.Harness.Begin();

        Assert.Equal(1L, rig.Cache.EagerPublishCounts.Writebacks);
        Assert.Equal((long)ImageSize, rig.Cache.EagerPublishCounts.Bytes);
        Assert.False(rig.Cache.GetImage(identifier).IsGpuModified);
        Assert.Equal(rewritten, rig.ReadGuest(GuestBase, ImageSize));
        Assert.True(rig.Cache.GetImage(other).IsGpuModified);
    }

    [Fact]
    public void AStoreAfterACompletedEagerWritebackDoesNotWait()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var identifier = AStoreProneImage(rig);
        var rewritten = TheGpuWritesItAgain(rig, identifier, 0x2B);
        rig.Cache.RecordEagerWritebacks();
        rig.Harness.Submit();
        rig.Harness.Begin();
        var takes = 0;
        var waits = 0;
        rig.CacheOwner.RunOnWorker = work =>
        {
            takes++;
            work();
            return true;
        };
        rig.CacheOwner.WaitOffWorker = (_, _) =>
        {
            waits++;
            return true;
        };
        var published = rig.Cache.StorePublishCounts.Writebacks;

        Assert.True(((IGuestImageStore)rig.Cache).MarkCpuWrite(GuestBase + StoreOffset, 8));

        Assert.Equal((0, 0), (takes, waits));
        Assert.Equal(published, rig.Cache.StorePublishCounts.Writebacks);
        Assert.Equal(0L, rig.Cache.EagerPublishCounts.StoreWaits);
        Assert.Equal(rewritten, rig.ReadGuest(GuestBase, ImageSize));
    }

    [Fact]
    public void AStoreWaitsWithoutABoundForAnEagerWritebackStillInFlight()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var identifier = AStoreProneImage(rig);
        var rewritten = TheGpuWritesItAgain(rig, identifier, 0x2B);
        rig.Cache.RecordEagerWritebacks();
        var serial = rig.Harness.SubmissionSerial;
        rig.Harness.Submit();
        rig.Harness.Begin();
        // The GPU has not reached the writeback yet, and reaches it only after the store has
        // waited past the fence's bound.
        rig.CacheOwner.CompletedOverride = serial - 1;
        rig.Cache.WritebackFenceTimeoutMilliseconds = 1;
        var waits = 0;
        rig.CacheOwner.WaitOffWorker = (_, _) =>
        {
            if (++waits < 3)
            {
                return false;
            }

            rig.CacheOwner.CompletedOverride = null;
            return true;
        };

        Assert.True(((IGuestImageStore)rig.Cache).MarkCpuWrite(GuestBase + StoreOffset, 8));

        Assert.Equal(3, waits);
        Assert.Equal(1L, rig.Cache.EagerPublishCounts.StoreWaits);
        Assert.Equal((0L, 0L), rig.Cache.FenceWaitCounts);
        Assert.Equal(rewritten, rig.ReadGuest(GuestBase, ImageSize));
    }

    [Fact]
    public void AStoreProneMarkDecaysAfterFramesWithoutAStore()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Cache.EagerPublishFrames = 3;
        var identifier = AStoreProneImage(rig);

        // A store two frames in keeps it marked four frames after its publish...
        MarkFrames(rig, 2);
        Assert.True(((IGuestImageStore)rig.Cache).MarkCpuWrite(GuestBase + StoreOffset, 8));
        MarkFrames(rig, 2);
        _ = TheGpuWritesItAgain(rig, identifier, 0x2B);
        rig.Cache.RecordEagerWritebacks();
        Assert.Equal(1L, rig.Cache.EagerPublishCounts.Writebacks);

        // ...and past three frames without one it is no longer written ahead.
        MarkFrames(rig, 4);
        _ = TheGpuWritesItAgain(rig, identifier, 0x3C);
        rig.Cache.RecordEagerWritebacks();
        Assert.Equal(1L, rig.Cache.EagerPublishCounts.Writebacks);
        Assert.Equal(0, rig.Cache.StoreProneImageCount);
        Assert.True(rig.Cache.GetImage(identifier).IsGpuModified);
    }

    [Fact]
    public void AReleasedStoreProneImageIsNotWrittenBackAhead()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var identifier = AStoreProneImage(rig);
        _ = TheGpuWritesItAgain(rig, identifier, 0x2B);

        // The guest unmaps the image and builds a new one, with the GPU's bytes, in its memory.
        ((IGuestImageStore)rig.Cache).Unregister(GuestBase, ImageSize);
        Assert.False(rig.Cache.TryGetImage(identifier, out _));
        var reused = AGpuWrittenImageOverStaleMemory(rig, GuestBase, 0x61, out var successor);
        var scribbled = rig.ReadGuest(GuestBase, ImageSize);

        rig.Cache.RecordEagerWritebacks();
        rig.Harness.Submit();
        rig.Harness.Begin();

        Assert.Equal(0L, rig.Cache.EagerPublishCounts.Writebacks);
        Assert.Equal(0, rig.Cache.StoreProneImageCount);
        Assert.True(rig.Cache.GetImage(successor).IsGpuModified);
        Assert.Equal(scribbled, rig.ReadGuest(GuestBase, ImageSize));
        Assert.NotEqual(reused, scribbled);
    }

    // A GPU-written image a store has needed written back, with that store landed over it.
    private static ResourceSlotIdentifier AStoreProneImage(Rig rig)
    {
        _ = AGpuWrittenImageOverStaleMemory(rig, GuestBase, 0x5A, out var identifier);
        rig.CacheOwner.RunOnWorker = work =>
        {
            work();
            return true;
        };
        rig.CacheOwner.WaitOffWorker = SubmitAndComplete(rig);
        Assert.True(((IGuestImageStore)rig.Cache).MarkCpuWrite(GuestBase + StoreOffset, 8));
        Assert.Equal(1L, rig.Cache.StorePublishCounts.Writebacks);
        Assert.Equal(1, rig.Cache.StoreProneImageCount);
        rig.WriteGuest(GuestBase + StoreOffset, [1, 2, 3, 4, 5, 6, 7, 8]);
        return identifier;
    }

    // Drawn into again: the texture holds these bytes, guest memory something else.
    private static byte[] TheGpuWritesItAgain(Rig rig, ResourceSlotIdentifier identifier, uint seed)
    {
        var bytes = MetalTileOracle.Pattern((int)ImageSize, seed);
        rig.WriteGuest(GuestBase, bytes);
        rig.Cache.InvalidateMemory(GuestBase, ImageSize);
        Assert.Equal(identifier, rig.Acquire(GuestBase, ImageRole.ColorTarget));
        Assert.True(rig.Cache.GetImage(identifier).IsGpuModified);
        rig.Harness.Submit();
        rig.Harness.Begin();
        rig.WriteGuest(GuestBase, MetalTileOracle.Pattern((int)ImageSize, seed ^ 0xFF));
        return bytes;
    }

    private static void MarkFrames(Rig rig, int frames)
    {
        for (var frame = 0; frame < frames; frame++)
        {
            rig.Cache.MarkFrame();
        }
    }
}
