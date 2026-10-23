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

    private sealed class RollingWritebackContext(MetalTransferHarness harness) : IMetalSubmissionContext
    {
        public nint Device => harness.Device;
        public nint CommandBuffer => harness.CommandBuffer;
        public ulong SubmissionSerial => harness.SubmissionSerial;
        public ulong CompletedSerial => harness.CompletedSerial;
        public bool WantsEncodedLabels => true;
        public bool Rolled { get; private set; }
        public void ProfileEncoded(string label)
        {
            if (Rolled) return;
            harness.Submit();
            harness.Begin();
            Rolled = true;
        }
    }

    [Fact]
    public void ATransferThatRollsItsSubmissionRetainsTheNewestFenceSerial()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Acquire(GuestBase, ImageRole.ColorTarget);
        var initial = rig.Harness.SubmissionSerial;
        var rolling = new RollingWritebackContext(rig.Harness);
        rig.CacheOwner.TransferContext = rolling;
        rig.CacheOwner.CompletedOverride = 0;
        rig.Cache.FlushGpuWrites(GuestBase, ImageSize, MetalWritebackCause.StorageDescriptor);
        Assert.True(rolling.Rolled);
        Assert.True(rolling.SubmissionSerial > initial);
        ulong waited = 0;
        rig.CacheOwner.WaitOffWorker = (serial, _) =>
        {
            waited = serial;
            rig.CacheOwner.CompletedOverride = serial;
            return true;
        };
        ((IGuestImageStore)rig.Cache).MarkPreciseCpuWrite(GuestBase, 8);
        Assert.Equal(rolling.SubmissionSerial, waited);
    }

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
    public void UnrelatedStoresAfterAMappingChangeDoNotWait()
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
        // An unrelated mapping publication does not change the encoded destination.
        ((IGuestImageStore)rig.Cache).MarkPreciseCpuWrite(GuestBase + 0x40000, 8);
        ((IGuestImageStore)rig.Cache).MarkCpuWriteLanded(GuestBase + 0x40000, 8);
        Assert.Equal(0, waits);
        Assert.Equal(1, rig.Cache.FencedWritebackCount);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ARemappedVaStillWaitsForTheCapturedDestinationThroughItsOldAlias(bool fastPath)
    {
        if (!RequireDevice()) return;
        const ulong mirror = GuestBase + 0x800000;
        using var rig = new Rig(Device.Value, layout: backing => backing.MirrorStart = mirror);
        var written = MetalTileOracle.Pattern((int)ImageSize, 0x5A);
        rig.WriteGuest(GuestBase, written);
        rig.Acquire(GuestBase, ImageRole.ColorTarget);
        rig.Harness.Submit();
        rig.Harness.Begin();
        rig.Cache.StoreFastPathEnabled = fastPath;
        // Remap after ResolveWritable, before the writeback publishes its fence.
        rig.CacheOwner.OnBeginTransfer = () =>
        {
            rig.GuestBacking.RemapOffset = 0x100000;
            rig.GuestBacking.AliasSnapshot = new object();
        };
        rig.Cache.FlushGpuWrites(GuestBase, ImageSize, MetalWritebackCause.HostRead);
        rig.CacheOwner.OnBeginTransfer = null;
        var waits = 0;
        rig.CacheOwner.WaitOffWorker = (serial, _) =>
        {
            waits++;
            rig.Harness.Submit();
            rig.Harness.Begin();
            return rig.Harness.CompletedSerial >= serial;
        };
        IGuestImageStore store = rig.Cache;
        store.MarkPreciseCpuWrite(GuestBase + 0x40000, 8);
        Assert.Equal(0, waits);
        store.MarkPreciseCpuWrite(mirror + 0x40, 8);
        Assert.Equal(1, waits);
        var patch = new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        rig.WriteGuest(mirror + 0x40, patch);
        store.MarkCpuWriteLanded(mirror + 0x40, 8);
        patch.CopyTo(written, 0x40);
        Assert.Equal(written, rig.ReadGuest(mirror, ImageSize));
        Assert.Contains("wb_fence_waits_byte_overlap=1", rig.Cache.TakeWritebackFenceCounters());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ScatteredWritebackRangesDoNotMakeUnrelatedStoresWait(bool fastPath)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value, layout: backing =>
        {
            backing.ScatterStart = GuestBase;
            backing.ScatterBytes = ImageSize;
            backing.ScatterChunk = 4096;
        });
        ReleaseAGpuWrittenImage(rig);
        rig.Cache.StoreFastPathEnabled = fastPath;
        rig.GuestBacking.AliasSnapshot = new object();
        var waits = 0;
        rig.CacheOwner.WaitOffWorker = (serial, _) =>
        {
            waits++;
            rig.Harness.Submit();
            rig.Harness.Begin();
            return rig.Harness.CompletedSerial >= serial;
        };
        IGuestImageStore store = rig.Cache;
        store.MarkPreciseCpuWrite(GuestBase + 0x40000, 8);
        store.MarkCpuWriteLanded(GuestBase + 0x40000, 8);
        Assert.Equal(0, waits);
        // A query crossing scattered pages must still find its actual overlap.
        store.MarkPreciseCpuWrite(GuestBase + 4092, 8);
        Assert.Equal(1, waits);
        var counters = rig.Cache.TakeWritebackFenceCounters();
        Assert.Contains("wb_fence_waits_byte_overlap=1", counters);
        Assert.Contains("wb_fence_waits_partial_run=1", counters);
        Assert.Contains("wb_fence_waits_mapping_uncertainty=0", counters);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void MappingChangesKeepExactBytesDistinctFromTrackerPageOverlap(bool fastPath)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Cache.StoreFastPathEnabled = fastPath;
        FenceMethod.Invoke(rig.Cache, new object?[] { GuestBase + 0x40, 8UL,
            rig.Harness.SubmissionSerial, MetalWritebackCause.Release, (ulong?)0x40, false });
        rig.GuestBacking.AliasSnapshot = new object();
        var waits = 0;
        rig.CacheOwner.WaitOffWorker = (serial, _) =>
        {
            waits++;
            rig.CacheOwner.CompletedOverride = serial;
            return true;
        };
        IGuestImageStore store = rig.Cache;
        store.MarkPreciseCpuWrite(GuestBase + 0x80, 8);
        Assert.Equal(0, waits);
        // Landed/fault notifications deliberately keep the baseline's tracker-page semantics.
        store.MarkCpuWriteLanded(GuestBase + 0x80, 8);
        Assert.Equal(1, waits);
        var counters = rig.Cache.TakeWritebackFenceCounters();
        Assert.Contains("wb_fence_waits_page_overlap=1", counters);
        Assert.Contains("wb_fence_waits_byte_overlap=0", counters);
        Assert.Contains("wb_fence_waits_mapping_uncertainty=0", counters);
    }

    [Fact]
    public void AnUnindexedKnownDestinationDoesNotMakeUnrelatedStoresWait()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        // Deliberately outside the hint's capacity, but still an exact physical identity.
        FenceMethod.Invoke(rig.Cache, new object?[] { GuestBase, ImageSize,
            rig.Harness.SubmissionSerial, MetalWritebackCause.Release, (ulong?)(1UL << 40), false });
        ((IGuestImageStore)rig.Cache).MarkPreciseCpuWrite(GuestBase + 0x40000, 8);
        Assert.Equal((0L, 0L), rig.Cache.FenceWaitCounts);
        rig.CacheOwner.WaitOffWorker = (serial, _) =>
        {
            Thread.Sleep(2);
            rig.CacheOwner.CompletedOverride = serial;
            return true;
        };
        ((IGuestImageStore)rig.Cache).MarkPreciseCpuWrite(GuestBase, 8);
        var counters = rig.Cache.TakeWritebackFenceCounters();
        Assert.Contains("wb_fence_waits_byte_overlap=1", counters);
        Assert.Contains("wb_fence_waits_unindexed=1", counters);
        Assert.Contains("wb_fence_waits_mapping_uncertainty=0", counters);
        Assert.True(CounterValue(counters, "wb_fence_wait_ms_unindexed") > 0);
        Assert.Contains("wb_fence_waits_unindexed=0", rig.Cache.TakeWritebackFenceCounters());
    }

    [Fact]
    public void AnUnknownDestinationRetainsItsWaitAndReportsUncertainty()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        FenceMethod.Invoke(rig.Cache, new object?[] { GuestBase, ImageSize,
            rig.Harness.SubmissionSerial, MetalWritebackCause.Release, null, false });
        rig.CacheOwner.WaitOffWorker = (serial, _) =>
        {
            rig.CacheOwner.CompletedOverride = serial;
            return true;
        };
        ((IGuestImageStore)rig.Cache).MarkPreciseCpuWrite(GuestBase + 0x40000, 8);
        var counters = rig.Cache.TakeWritebackFenceCounters();
        Assert.Contains("wb_fence_waits_mapping_uncertainty=1", counters);
        Assert.Contains("wb_fence_waits_unindexed=1", counters);
        Assert.Contains("wb_fence_waits_byte_overlap=0", counters);
    }

    private static double CounterValue(string counters, string name) =>
        double.Parse(counters.Split(' ', StringSplitOptions.RemoveEmptyEntries)
            .Single(value => value.StartsWith(name + "=", StringComparison.Ordinal))[(name.Length + 1)..],
            System.Globalization.CultureInfo.CurrentCulture);

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void APartialRunMustCompleteBeforeAnOverlappingStore(bool managed, bool fastPath)
    {
        if (!RequireDevice()) return;
        const ulong mirror = GuestBase + 0x800000;
        using var rig = new Rig(Device.Value, layout: backing => backing.MirrorStart = mirror);
        rig.Cache.StoreFastPathEnabled = fastPath;
        var written = MetalTileOracle.Pattern((int)ImageSize, 0x5A);
        rig.WriteGuest(GuestBase, written);
        var image = rig.Cache.GetImage(rig.Acquire(GuestBase, ImageRole.ColorTarget));
        rig.Harness.Submit();
        rig.Harness.Begin();
        rig.WriteGuest(GuestBase, MetalTileOracle.Pattern((int)ImageSize, 0xC3));
        var method = typeof(MetalImageCache).GetMethod("TryWriteBackRun", BindingFlags.NonPublic | BindingFlags.Instance)!;
        var args = new object?[] { image, 0UL, 4096UL, MetalWritebackCause.ImageOverlap, null };
        Assert.True((bool)method.Invoke(rig.Cache, args)!);
        Assert.Equal(1, rig.Cache.FencedWritebackCount);
        // The image owner index is keyed by guest VA. This other mapping reaches
        // the same bytes without finding that owner: the fence is independently needed.
        Assert.True(image.IsGpuModified);
        var waits = 0;
        rig.CacheOwner.WaitOffWorker = (serial, _) =>
        {
            waits++;
            rig.Harness.Submit();
            rig.Harness.Begin();
            return rig.Harness.CompletedSerial >= serial;
        };
        IGuestImageStore store = rig.Cache;
        store.MarkPreciseCpuWrite(mirror + 8192, 8);
        Assert.Equal(0, waits);
        if (managed) store.MarkPreciseCpuWrite(mirror + 0x40, 8);
        else Assert.True(store.MarkCpuWrite(mirror + 0x40, 8));
        Assert.Equal(1, waits);
        var patch = new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        rig.WriteGuest(mirror + 0x40, patch);
        if (managed) store.MarkCpuWriteLanded(mirror + 0x40, 8);
        rig.Harness.Submit();
        rig.Harness.Begin();
        patch.CopyTo(written, 0x40);
        Assert.Equal(written.AsSpan(0, 4096).ToArray(), rig.ReadGuest(GuestBase, 4096));
        var counters = rig.Cache.TakeWritebackFenceCounters();
        Assert.Contains("wb_fence_waits_partial_run=1", counters);
        Assert.Contains("wb_fence_waits_byte_overlap=1", counters);
        Assert.Equal(CounterValue(counters, "wb_fence_wait_ms"), CounterValue(counters, "wb_fence_wait_ms_partial_run"));
    }

    [Fact]
    public void BaselinesUnfencedPartialRunCanOverwriteALaterCpuStore()
    {
        if (!RequireDevice()) return;
        const ulong mirror = GuestBase + 0x800000;
        using var rig = new Rig(Device.Value, layout: backing => backing.MirrorStart = mirror);
        rig.Cache.WritebackFenceEnabled = false;
        var written = MetalTileOracle.Pattern((int)ImageSize, 0x5A);
        rig.WriteGuest(GuestBase, written);
        var image = rig.Cache.GetImage(rig.Acquire(GuestBase, ImageRole.ColorTarget));
        rig.Harness.Submit();
        rig.Harness.Begin();
        var method = typeof(MetalImageCache).GetMethod("TryWriteBackRun", BindingFlags.NonPublic | BindingFlags.Instance)!;
        Assert.True((bool)method.Invoke(rig.Cache, new object?[] {
            image, 0UL, 4096UL, MetalWritebackCause.ImageOverlap, null })!);
        Assert.Equal(0, rig.Cache.FencedWritebackCount);
        // Reproduce the baseline's partial-run ordering: the CPU lands before
        // the recorded GPU retile. Its patch is demonstrably overwritten.
        var patch = new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        Assert.NotEqual(patch, written.AsSpan(0x40, 8).ToArray());
        IGuestImageStore store = rig.Cache;
        store.MarkPreciseCpuWrite(mirror + 0x40, 8);
        rig.WriteGuest(mirror + 0x40, patch);
        store.MarkCpuWriteLanded(mirror + 0x40, 8);
        Assert.Equal((0L, 0L), rig.Cache.FenceWaitCounts);
        rig.Harness.Submit();
        rig.Harness.Begin();
        Assert.Equal(written.AsSpan(0x40, 8).ToArray(), rig.ReadGuest(GuestBase + 0x40, 8));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ARefusedWritebackHasNoFenceAsInTheBaseline(bool fastPath)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Cache.StoreFastPathEnabled = fastPath;
        var image = rig.Cache.GetImage(rig.Acquire(GuestBase, ImageRole.ColorTarget, samples: 2));
        var method = typeof(MetalImageCache).GetMethod("TryWriteBackRun", BindingFlags.NonPublic | BindingFlags.Instance)!;
        Assert.False((bool)method.Invoke(rig.Cache, new object?[] {
            image, 0UL, 4096UL, MetalWritebackCause.ImageOverlap, null })!);
        Assert.Equal(0, rig.Cache.FencedWritebackCount);
        rig.Cache.FlushGpuWrites(GuestBase, ImageSize, MetalWritebackCause.HostRead);
        Assert.Equal(0, rig.Cache.FencedWritebackCount);
        Assert.True(image.IsGpuModified);
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
            FenceMethod.Invoke(rig.Cache, new object[] { GuestBase, ImageSize, serial + 1, MetalWritebackCause.Release, (ulong?)0, false });
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
