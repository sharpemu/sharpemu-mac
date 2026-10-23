// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Reflection;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Images;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

public sealed partial class MetalImageBudgetTests
{
    private static readonly FieldInfo FenceGateField = typeof(MetalImageCache)
        .GetField("_fenceGate", BindingFlags.NonPublic | BindingFlags.Instance)!;
    private static readonly MethodInfo FenceMethod = typeof(MetalImageCache)
        .GetMethod("FenceWriteback", BindingFlags.NonPublic | BindingFlags.Instance)!;

    [Fact]
    public void CanonicalAliasResolutionAndUncertaintyAreDeviceIndependent()
    {
        using var backing = new Backing { MirrorStart = GuestBase + 0x800000 };
        Assert.True(MetalFencePageIndex.TryResolveBackingRange(backing, GuestBase + 4095, 2, out var original));
        Assert.True(MetalFencePageIndex.TryResolveBackingRange(backing, backing.MirrorStart + 4095, 2, out var alias));
        Assert.Equal(original, alias);
        var index = new MetalFencePageIndex();
        index.Publish(original, 2, 7);
        Assert.True(index.MayOverlap(alias, 2));
        backing.NoWindows = true;
        Assert.False(MetalFencePageIndex.TryResolveBackingRange(backing, GuestBase, 8, out _));
        Assert.False(MetalFencePageIndex.TryResolveBackingRange(backing, ulong.MaxValue, 8, out _));
    }

    [Fact]
    public void UnrelatedPairedStoresAvoidALiveFenceGateAndCompletionQuery()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        ReleaseAGpuWrittenImage(rig);
        rig.Acquire(GuestBase + 0x60000, ImageRole.Texture);
        rig.Cache.StoreFastPathEnabled = true;
        var gate = FenceGateField.GetValue(rig.Cache)!;
        var imageGate = (RegionLock)typeof(MetalImageCache).GetField("_lock",
            BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(rig.Cache)!;
        var queries = 0;
        rig.CacheOwner.OnCompletedQuery = () =>
        {
            Assert.False(Monitor.IsEntered(gate));
            Interlocked.Increment(ref queries);
        };
        Task store;
        lock (gate)
        {
            using var held = imageGate.Hold();
            store = Task.Factory.StartNew(() =>
            {
                IGuestImageStore images = rig.Cache;
                Assert.False(images.MarkPreciseCpuWrite(GuestBase + 0x40000, 8));
                rig.WriteGuest(GuestBase + 0x40000, new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 });
                Assert.False(images.MarkCpuWriteLanded(GuestBase + 0x40000, 8));
            }, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);
            Assert.True(store.Wait(TimeSpan.FromSeconds(10)));
        }
        store.GetAwaiter().GetResult();
        Assert.Equal(0, queries);
        Assert.Equal(1, rig.Cache.FencedWritebackCount);
        rig.CacheOwner.OnCompletedQuery = null;
    }

    [Fact]
    public void DisablingRejectionRestoresTheUnrelatedStoreFenceGateLookup()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        ReleaseAGpuWrittenImage(rig);
        rig.Cache.StoreFastPathEnabled = false;
        using var started = new ManualResetEventSlim();
        Task store;
        lock (FenceGateField.GetValue(rig.Cache)!)
        {
            store = Task.Factory.StartNew(() =>
            {
                started.Set();
                ((IGuestImageStore)rig.Cache).MarkPreciseCpuWrite(GuestBase + 0x40000, 8);
            }, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);
            Assert.True(started.Wait(TimeSpan.FromSeconds(10)));
            Assert.False(store.Wait(TimeSpan.FromMilliseconds(100)));
        }
        Assert.True(store.Wait(TimeSpan.FromSeconds(10)));
        Assert.Equal((0L, 0L), rig.Cache.FenceWaitCounts);
    }

    [Fact]
    public void AnUncertainPublicationBatchCannotGrantANegativeLookup()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        ReleaseAGpuWrittenImage(rig);
        rig.Cache.StoreFastPathEnabled = true;
        var version = typeof(MetalImageCache).GetField("_fenceIndexVersion",
            BindingFlags.NonPublic | BindingFlags.Instance)!;
        var before = (long)version.GetValue(rig.Cache)!;
        using var started = new ManualResetEventSlim();
        Task store;
        bool rejected;
        lock (FenceGateField.GetValue(rig.Cache)!)
        {
            version.SetValue(rig.Cache, before + 1); // Publisher paused inside its batch.
            store = Task.Factory.StartNew(() =>
            {
                started.Set();
                ((IGuestImageStore)rig.Cache).MarkPreciseCpuWrite(GuestBase + 0x40000, 8);
            }, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);
            Assert.True(started.Wait(TimeSpan.FromSeconds(10)));
            rejected = !store.Wait(TimeSpan.FromMilliseconds(100));
            version.SetValue(rig.Cache, before + 2);
        }
        Assert.True(store.Wait(TimeSpan.FromSeconds(10)));
        Assert.True(rejected);
    }

    [Fact]
    public void StoresThroughAnotherGuestVaWaitForTheSameBackingFence()
    {
        if (!RequireDevice()) return;
        const ulong mirror = GuestBase + 0x800000;
        using var rig = new Rig(Device.Value, layout: backing => backing.MirrorStart = mirror);
        var (_, written) = ReleaseAGpuWrittenImage(rig);
        rig.Cache.StoreFastPathEnabled = true;
        var gate = FenceGateField.GetValue(rig.Cache)!;
        rig.CacheOwner.OnCompletedQuery = () => Assert.False(Monitor.IsEntered(gate));
        var waits = 0;
        rig.CacheOwner.WaitOffWorker = (serial, _) =>
        {
            Assert.False(Monitor.IsEntered(gate));
            waits++;
            rig.Harness.Submit();
            rig.Harness.Begin();
            return rig.Harness.CompletedSerial >= serial;
        };
        IGuestImageStore images = rig.Cache;
        images.MarkPreciseCpuWrite(mirror + 0x40, 8);
        Assert.Equal(1, waits);
        var patch = new byte[] { 9, 8, 7, 6, 5, 4, 3, 2 };
        rig.WriteGuest(mirror + 0x40, patch);
        images.MarkCpuWriteLanded(mirror + 0x40, 8);
        patch.CopyTo(written, 0x40);
        Assert.Equal(written, rig.ReadGuest(GuestBase, ImageSize));
        rig.CacheOwner.OnCompletedQuery = null;
    }

    [Fact]
    public void AChangedMappingSnapshotCannotRejectAnOutstandingFence()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        ReleaseAGpuWrittenImage(rig);
        rig.Cache.StoreFastPathEnabled = true;
        rig.GuestBacking.AliasSnapshot = new object();
        var waits = 0;
        rig.CacheOwner.WaitOffWorker = (serial, _) =>
        {
            waits++;
            rig.Harness.Submit();
            rig.Harness.Begin();
            return rig.Harness.CompletedSerial >= serial;
        };
        // Unknown mapping identity must take the authoritative path even for
        // an address whose newly resolved backing page has no index entry.
        ((IGuestImageStore)rig.Cache).MarkPreciseCpuWrite(GuestBase + 0x40000, 8);
        Assert.Equal(1, waits);
    }

    [Fact]
    public void RetiringAWaitedFenceKeepsANewerFenceAndItsWatch()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        ReleaseAGpuWrittenImage(rig);
        rig.Cache.StoreFastPathEnabled = true;
        var serial = rig.Harness.SubmissionSerial;
        rig.CacheOwner.WaitOffWorker = (waited, _) =>
        {
            Assert.Equal(serial, waited);
            FenceMethod.Invoke(rig.Cache, new object[] { GuestBase, ImageSize, serial + 1, MetalWritebackCause.Release, rig.GuestBacking.BackingAliasSnapshot! });
            rig.CacheOwner.CompletedOverride = serial;
            return true;
        };
        Assert.True(((IGuestImageStore)rig.Cache).MarkCpuWrite(GuestBase + 0x40, 8));
        Assert.Equal(1, rig.Cache.FencedWritebackCount);
        Assert.False(rig.Pages.Allows(GuestBase, FaultKind.Write));
        ulong newest = 0;
        rig.CacheOwner.WaitOffWorker = (waited, _) =>
        {
            newest = waited;
            rig.CacheOwner.CompletedOverride = waited;
            return true;
        };
        ((IGuestImageStore)rig.Cache).MarkPreciseCpuWrite(GuestBase + 0x40, 8);
        Assert.Equal(serial + 1, newest);
    }
}
