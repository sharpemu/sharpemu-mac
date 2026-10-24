// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Reflection;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Images;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// A recorded writeback runs when the GPU reaches it, and the guest may have reused the bytes
/// by then: Silent Hill rebuilt an object in a released render target's memory and the late
/// writeback turned its vtable into pixels. The range stays watched until the submission
/// carrying the writeback completes, and a guest store that faults there waits for it.
/// </summary>
public sealed partial class MetalImageBudgetTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void UnrelatedStoresDoNotWaitBehindTheFenceMetadataGate(bool precise)
    {
        if (!RequireDevice() || Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_FENCE_PAGE_FILTER") == "0") return;
        using var rig = new Rig(Device.Value);
        ReleaseAGpuWrittenImage(rig);
        Assert.Equal(1, rig.Cache.FencedWritebackCount);
        var gate = typeof(MetalImageCache).GetField("_fenceGate", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(rig.Cache)!;
        using var entered = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        var holder = Task.Run(() =>
        {
            lock (gate)
            {
                entered.Set();
                Assert.True(release.Wait(TimeSpan.FromSeconds(10)));
            }
        });
        Assert.True(entered.Wait(TimeSpan.FromSeconds(10)));
        Task<bool>? store = null;
        try
        {
            store = Task.Run(() => precise
                ? ((IGuestImageStore)rig.Cache).MarkPreciseCpuWrite(GuestBase + 0x40000, 8)
                : ((IGuestImageStore)rig.Cache).MarkCpuWrite(GuestBase + 0x40000, 8));
            Assert.True(store.Wait(TimeSpan.FromSeconds(2)));
            Assert.False(store.Result);
            Assert.Equal(1, rig.Cache.FencedWritebackCount);
        }
        finally
        {
            release.Set();
            Assert.True(holder.Wait(TimeSpan.FromSeconds(10)));
            if (store is not null) Assert.True(store.Wait(TimeSpan.FromSeconds(10)));
        }
    }

    [Fact]
    public void AReleasedImagesWritebackKeepsItsRangeWatchedUntilItsSubmissionCompletes()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var (identifier, _) = ReleaseAGpuWrittenImage(rig);

        Assert.False(rig.Cache.TryGetImage(identifier, out _));
        Assert.Equal(1, rig.Cache.FencedWritebackCount);
        Assert.False(rig.Pages.Allows(GuestBase, FaultKind.Write));
        Assert.False(rig.Pages.Allows(GuestBase + ImageSize - 1, FaultKind.Write));

        // Still recorded: collecting keeps the watch.
        rig.Cache.Collect();
        Assert.Equal(1, rig.Cache.FencedWritebackCount);

        rig.Harness.Submit();
        rig.Harness.Begin();
        rig.Cache.Collect();
        Assert.Equal(0, rig.Cache.FencedWritebackCount);
        Assert.True(rig.Pages.Allows(GuestBase, FaultKind.Write));
    }

    [Fact]
    public void WithTheFenceOffAReleasedImageLeavesNoWatch()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Cache.WritebackFenceEnabled = false;
        var (identifier, _) = ReleaseAGpuWrittenImage(rig);

        Assert.False(rig.Cache.TryGetImage(identifier, out _));
        Assert.Equal(0, rig.Cache.FencedWritebackCount);
        Assert.True(rig.Pages.Allows(GuestBase, FaultKind.Write));
    }

    // The store faults while the writeback is recorded and not run, waits for exactly that
    // submission, and lands after the writeback instead of under it.
    [Fact]
    public void AGuestStoreIntoARecordedWritebackWaitsForItsSubmission()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var (_, written) = ReleaseAGpuWrittenImage(rig);
        var writebackSerial = rig.Harness.SubmissionSerial;

        using var waiting = new ManualResetEventSlim();
        using var completed = new ManualResetEventSlim();
        ulong waitedFor = 0;
        rig.CacheOwner.WaitOffWorker = (serial, timeout) =>
        {
            waitedFor = serial;
            waiting.Set();
            return completed.Wait(timeout) && rig.Harness.CompletedSerial >= serial;
        };
        rig.Cache.WritebackFenceTimeoutMilliseconds = 10_000;

        var store = Task.Run(() => ((IGuestImageStore)rig.Cache).MarkCpuWrite(GuestBase + 0x40, 8));
        Assert.True(waiting.Wait(TimeSpan.FromSeconds(10)));
        Assert.Equal(writebackSerial, waitedFor);
        Assert.False(store.Wait(TimeSpan.FromMilliseconds(100)));

        rig.Harness.Submit();
        rig.Harness.Begin();
        completed.Set();
        Assert.True(store.Wait(TimeSpan.FromSeconds(10)));
        Assert.True(store.Result);
        Assert.Equal(0, rig.Cache.FencedWritebackCount);
        Assert.True(rig.Pages.Allows(GuestBase, FaultKind.Write));
        Assert.Equal((1L, 0L), rig.Cache.FenceWaitCounts);

        // The writeback has landed; the store the guest retries now lands after it.
        Assert.Equal(written, rig.ReadGuest(GuestBase, ImageSize));
        var reused = MetalTileOracle.Pattern((int)ImageSize, 0x3C);
        rig.WriteGuest(GuestBase, reused);
        rig.Cache.Collect();
        Assert.Equal(reused, rig.ReadGuest(GuestBase, ImageSize));
    }

    // A managed write never faults, so it waits before it lands: the writeback recorded
    // before it completes first, and the write lands over the GPU's bytes and stays there.
    [Fact]
    public void AManagedWriteIntoARecordedWritebackWaitsForItsSubmission()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var (_, written) = ReleaseAGpuWrittenImage(rig);
        var writebackSerial = rig.Harness.SubmissionSerial;
        ulong waitedFor = 0;
        rig.CacheOwner.WaitOffWorker = (serial, _) =>
        {
            waitedFor = waitedFor == 0 ? serial : waitedFor;
            rig.Harness.Submit();
            rig.Harness.Begin();
            return rig.Harness.CompletedSerial >= serial;
        };

        var expected = ManagedWriteOverAFencedWriteback(rig, written);
        Assert.Equal(writebackSerial, waitedFor);
        Assert.Equal((1L, 0L), rig.Cache.FenceWaitCounts);
        Assert.Equal(expected, rig.ReadGuest(GuestBase, ImageSize));
    }

    // The GPU worker making a managed write cannot wait for a submission it may still be
    // recording: it commits and waits for the GPU instead, and never deadlocks on itself.
    [Fact]
    public void AManagedWriteOnTheWorkerCommitsAndWaitsForTheWriteback()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var (_, written) = ReleaseAGpuWrittenImage(rig);
        var offWorker = 0;
        var gpuWaits = 0;
        rig.CacheOwner.WaitOffWorker = (_, _) => { offWorker++; return false; };
        rig.CacheOwner.OnWaitForGpu = () =>
        {
            gpuWaits++;
            rig.Harness.Submit();
            rig.Harness.Begin();
        };
        rig.CacheOwner.OnWorker = true;

        var expected = ManagedWriteOverAFencedWriteback(rig, written);
        Assert.Equal((1, 0), (gpuWaits, offWorker));
        Assert.Equal((1L, 0L), rig.Cache.FenceWaitCounts);
        Assert.Equal(expected, rig.ReadGuest(GuestBase, ImageSize));
    }

    // A managed write past the fenced bytes does not wait.
    [Fact]
    public void AManagedWritePastAFencedWritebacksBytesDoesNotWait()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        _ = ReleaseAGpuWrittenImage(rig);
        var waits = 0;
        rig.CacheOwner.WaitOffWorker = (_, _) => { waits++; return true; };

        _ = ((IGuestImageStore)rig.Cache).MarkPreciseCpuWrite(GuestBase + ImageSize, 8);
        Assert.Equal(0, waits);
        Assert.Equal(1, rig.Cache.FencedWritebackCount);
    }

    // A managed write of eight bytes into the released image, made the way the HLE makes it:
    // marked, written, marked landed. Returns what guest memory should hold once it has.
    private static byte[] ManagedWriteOverAFencedWriteback(Rig rig, byte[] written)
    {
        Assert.Equal(1, rig.Cache.FencedWritebackCount);
        IGuestImageStore store = rig.Cache;
        var bytes = new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        _ = store.MarkPreciseCpuWrite(GuestBase + 0x40, (ulong)bytes.Length);
        rig.WriteGuest(GuestBase + 0x40, bytes);
        _ = store.MarkCpuWriteLanded(GuestBase + 0x40, (ulong)bytes.Length);
        rig.Harness.Submit();
        rig.Harness.Begin();
        var expected = (byte[])written.Clone();
        bytes.CopyTo(expected, 0x40);
        return expected;
    }

    // A fault outside every fenced page is none of the fence's business and costs no wait.
    [Fact]
    public void AStoreBesideAFencedWritebackDoesNotWait()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        _ = ReleaseAGpuWrittenImage(rig);
        var waits = 0;
        rig.CacheOwner.WaitOffWorker = (_, _) => { waits++; return true; };

        Assert.False(((IGuestImageStore)rig.Cache).MarkCpuWrite(GuestBase + 0x40000, 8));
        Assert.Equal(0, waits);
        Assert.Equal(1, rig.Cache.FencedWritebackCount);
    }

    // A store that cannot wait - the timeout, or the GPU worker itself - drops the fence and
    // proceeds as it did before, counted.
    [Fact]
    public void AStoreThatCannotWaitDropsTheFenceAndIsCounted()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        _ = ReleaseAGpuWrittenImage(rig);
        rig.CacheOwner.WaitOffWorker = (_, _) => false;

        Assert.True(((IGuestImageStore)rig.Cache).MarkCpuWrite(GuestBase, 8));
        Assert.Equal((1L, 1L), rig.Cache.FenceWaitCounts);
        Assert.Equal(0, rig.Cache.FencedWritebackCount);
        Assert.True(rig.Pages.Allows(GuestBase, FaultKind.Write));
    }

    // Silent Hill flushed a target for a storage binding on draw after draw while the GPU ran
    // 400 ms behind, and every writeback took its own watch on the target's pages until the
    // 128th found the page guard's 7-bit count full. The repeats share one fence and one watch.
    [Fact]
    public void ATargetWrittenBackOnEveryDrawHoldsOneWatchPerPage()
    {
        if (!RequireDevice()) return;
        var fatals = RecordPageGuardFatals(out var previous);
        try
        {
            // Four watches a page instead of 127, so a few draws reach the limit.
            using var rig = new Rig(Device.Value, writeWatchLimit: 4);
            rig.WriteGuest(GuestBase, MetalTileOracle.Pattern((int)ImageSize, 0x5A));
            long? watches = null;
            for (var draw = 0; draw < 32; draw++)
            {
                var identifier = rig.Acquire(GuestBase, ImageRole.ColorTarget);
                Assert.True(rig.Cache.GetImage(identifier).IsGpuModified);
                rig.Cache.FlushGpuWrites(GuestBase, ImageSize, MetalWritebackCause.StorageDescriptor);
                watches ??= rig.Pages.LiveWriteWatches;
                Assert.Equal(watches, rig.Pages.LiveWriteWatches);
            }

            Assert.Empty(fatals);
            Assert.Equal(1, rig.Cache.FencedWritebackCount);
            Assert.False(rig.Pages.Allows(GuestBase, FaultKind.Write));

            rig.Harness.Submit();
            rig.Harness.Begin();
            rig.Cache.Collect();
            Assert.Equal(0, rig.Cache.FencedWritebackCount);
            Assert.Equal(watches - (long)(ImageSize / TrackerLayout.PageBytes), rig.Pages.LiveWriteWatches);
        }
        finally
        {
            RestorePageGuardFatals(previous);
        }
    }

    // The same range written back in submission after submission the GPU has not reached is one
    // fence, and it holds until the latest of them: the store still lands after every one.
    [Fact]
    public void WritebacksAcrossLaggingSubmissionsShareAFenceThatWaitsForTheLatest()
    {
        if (!RequireDevice()) return;
        var fatals = RecordPageGuardFatals(out var previous);
        try
        {
            using var rig = new Rig(Device.Value, writeWatchLimit: 4);
            rig.WriteGuest(GuestBase, MetalTileOracle.Pattern((int)ImageSize, 0x5A));
            var serials = new List<ulong>();
            for (var submission = 0; submission < 8; submission++)
            {
                rig.Acquire(GuestBase, ImageRole.ColorTarget);
                rig.Cache.FlushGpuWrites(GuestBase, ImageSize, MetalWritebackCause.StorageDescriptor);
                serials.Add(rig.Harness.SubmissionSerial);
                rig.CacheOwner.CompletedOverride = serials[0] - 1;
                rig.Harness.Submit();
                rig.Harness.Begin();
            }

            Assert.Empty(fatals);
            Assert.Equal(1, rig.Cache.FencedWritebackCount);

            // The GPU reaches the middle writeback: the later ones still hold the pages.
            rig.CacheOwner.CompletedOverride = serials[3];
            rig.Cache.Collect();
            Assert.Equal(1, rig.Cache.FencedWritebackCount);
            Assert.False(rig.Pages.Allows(GuestBase, FaultKind.Write));

            ulong waitedFor = 0;
            rig.CacheOwner.WaitOffWorker = (serial, _) =>
            {
                waitedFor = serial;
                rig.CacheOwner.CompletedOverride = serial;
                return true;
            };
            Assert.True(((IGuestImageStore)rig.Cache).MarkCpuWrite(GuestBase + 0x40, 8));
            Assert.Equal(serials[^1], waitedFor);
            Assert.Equal(0, rig.Cache.FencedWritebackCount);
        }
        finally
        {
            RestorePageGuardFatals(previous);
        }
    }

    // Ranges that repeat, overlap and cross a block boundary: each page carries one watch while
    // any range covers it, so even a limit of one is never reached.
    [Fact]
    public void OverlappingFencedRangesHoldOneWatchPerPage()
    {
        const ulong Page = TrackerLayout.PageBytes;
        var fatals = RecordPageGuardFatals(out var previous);
        try
        {
            var pages = NewPageGuard(new AddressSpace(), writeWatchLimit: 1);
            var set = new MetalPageWatchSet(pages);
            for (var repeat = 0; repeat < 200; repeat++)
            {
                set.Add(GuestBase + Page, 2 * Page);
            }

            set.Add(GuestBase, 2 * Page);
            set.Add(GuestBase + 2 * Page, TrackerLayout.BlockBytes);
            var watched = 2 + TrackerLayout.BlockBytes / Page;
            Assert.Equal((long)watched, set.WatchedPages);
            Assert.Equal((long)watched, pages.LiveWriteWatches);
            Assert.Equal(1, pages.TakePeakPageWriteWatches());

            for (var repeat = 0; repeat < 200; repeat++)
            {
                set.Remove(GuestBase + Page, 2 * Page);
            }

            // Page 1 is still the second range's, page 2 the block-crossing one's.
            Assert.False(pages.Allows(GuestBase + Page, FaultKind.Write));
            Assert.False(pages.Allows(GuestBase + 2 * Page, FaultKind.Write));

            set.Remove(GuestBase, 2 * Page);
            Assert.True(pages.Allows(GuestBase, FaultKind.Write));
            Assert.True(pages.Allows(GuestBase + Page, FaultKind.Write));
            Assert.False(pages.Allows(GuestBase + TrackerLayout.BlockBytes + Page, FaultKind.Write));

            set.Clear();
            Assert.Equal(0, set.WatchedPages);
            Assert.Equal(0, pages.LiveWriteWatches);
            Assert.True(pages.Allows(GuestBase + TrackerLayout.BlockBytes + Page, FaultKind.Write));
            pages.Dispose();
            Assert.Empty(fatals);
        }
        finally
        {
            RestorePageGuardFatals(previous);
        }
    }

    // The page guard's limit and fatal hook are internal to SharpEmu.HLE, which grants this suite
    // no friendship.
    private static readonly FieldInfo PageGuardFatal =
        typeof(PageGuard).GetField("OnFatal", BindingFlags.NonPublic | BindingFlags.Static)!;

    private static PageGuard NewPageGuard(IGuestAddressSpace space, int writeWatchLimit)
    {
        var pages = new PageGuard(space);
        typeof(PageGuard).GetProperty("WriteWatchLimit", BindingFlags.NonPublic | BindingFlags.Instance)!
            .SetValue(pages, writeWatchLimit);
        return pages;
    }

    // Records what would have failed the process, so a test can assert none happened.
    private static List<string> RecordPageGuardFatals(out Action<string> previous)
    {
        var fatals = new List<string>();
        previous = (Action<string>)PageGuardFatal.GetValue(null)!;
        PageGuardFatal.SetValue(null, (Action<string>)(message =>
        {
            lock (fatals) fatals.Add(message);
        }));
        return fatals;
    }

    private static void RestorePageGuardFatals(Action<string> previous) => PageGuardFatal.SetValue(null, previous);

    // A GPU-written image retired with its writeback recorded into the open submission.
    private static (ResourceSlotIdentifier Identifier, byte[] Written) ReleaseAGpuWrittenImage(Rig rig)
    {
        rig.Configure(retireFrames: 2, retireSeconds: 1);
        var written = MetalTileOracle.Pattern((int)ImageSize, 0x5A);
        rig.WriteGuest(GuestBase, written);
        var identifier = rig.Acquire(GuestBase, ImageRole.ColorTarget);
        Assert.True(rig.Cache.GetImage(identifier).IsGpuModified);
        rig.Harness.Submit();
        rig.Harness.Begin();
        rig.WriteGuest(GuestBase, MetalTileOracle.Pattern((int)ImageSize, 0xC3));
        for (var frame = 0; frame < 4 && rig.Cache.TryGetImage(identifier, out _); frame++)
        {
            rig.Frame(seconds: 1, submit: false);
        }

        Assert.False(rig.Cache.TryGetImage(identifier, out _));
        return (identifier, written);
    }
}
