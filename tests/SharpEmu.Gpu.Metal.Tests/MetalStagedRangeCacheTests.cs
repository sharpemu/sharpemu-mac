// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
using SharpEmu.Gpu.Metal.V2;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// The rules a staged range cache keeps without a GPU: what makes an entry stale, what
/// never is dropped, and when a buffer may be written again.
/// </summary>
public sealed class MetalStagedRangeCacheTests
{
    private const ulong Mib = 1024 * 1024;
    private const ulong Range = 0x10_0000_0000;
    private const ulong Size = 512 * 1024;

    private sealed class FakeBackend : IStagedRangeBackend
    {
        public int Created;
        public int Released;
        public readonly List<(ulong Address, ulong Size)> Watches = new();
        public Action? BeforeRemoveWatch;
        private nint _next = 0x1000;

        public (nint Handle, nint Contents, ulong GpuAddress) CreateBuffer(ulong capacity)
        {
            Created++;
            _next += 0x10;
            return (_next, 0, (ulong)_next);
        }

        public void ReleaseBuffer(nint handle) => Released++;

        public void AddWatch(ulong address, ulong size) => Watches.Add((address, size));

        public void RemoveWatch(ulong address, ulong size)
        {
            BeforeRemoveWatch?.Invoke();
            Assert.True(Watches.Remove((address, size)));
        }
    }

    private static MetalStagedRangeCache Cache(FakeBackend backend, ulong budgetMiB = 64) =>
        new(backend, budgetMiB * Mib);

    [Fact]
    public void RemovalKeepsRejectionPositiveUntilTheWatchIsGone()
    {
        var backend = new FakeBackend();
        var cache = Cache(backend);
        cache.StoreFastPathEnabled = true;
        cache.Acquire(Range, Size, written: false, 1, 0, out _);
        using var removing = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        backend.BeforeRemoveWatch = () =>
        {
            removing.Set();
            Assert.True(release.Wait(TimeSpan.FromSeconds(10)));
        };
        var removal = Task.Factory.StartNew(() => cache.OnUnmap(Range, Size),
            CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);
        try
        {
            Assert.True(removing.Wait(TimeSpan.FromSeconds(10)));
            Assert.True(cache.MayCover(Range, 8));
        }
        finally
        {
            release.Set();
            Assert.True(removal.Wait(TimeSpan.FromSeconds(10)));
            cache.Dispose();
        }
        Assert.False(cache.MayCover(Range, 8));
    }

    [Fact]
    public void AStoreOnAnUnrelatedPageOfALiveBucketDoesNotTakeTheStagedGate()
    {
        var backend = new FakeBackend();
        var cache = Cache(backend);
        cache.StoreFastPathEnabled = true;
        cache.Acquire(Range, Size, written: false, 1, 0, out var entry);
        var gate = typeof(MetalStagedRangeCache).GetField("_gate",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!.GetValue(cache)!;
        lock (gate)
        {
            var stores = Task.Factory.StartNew(() =>
            {
                Assert.False(cache.MayCover(Range + Size, 8));
                Assert.False(cache.OnPreciseCpuWrite(Range + Size, 8, backed: true));
                Assert.False(cache.OnCpuWrite(Range + Size, 8));
            }, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);
            Assert.True(stores.Wait(TimeSpan.FromSeconds(10)));
        }
        Assert.Equal(0, entry!.Stale);
        Assert.Single(backend.Watches);
        cache.OnUnmap(Range, Size);
        Assert.False(cache.MayCover(Range, 8));
        cache.Dispose();
    }

    [Fact]
    public void ASecondBindOfTheSameRangeHitsAndTakesNoBuffer()
    {
        var backend = new FakeBackend();
        var cache = Cache(backend);
        Assert.False(cache.Acquire(Range, Size, written: false, 1, 0, out var first));
        Assert.True(cache.Acquire(Range, Size, written: false, 2, 1, out var second));
        Assert.Same(first, second);
        Assert.Equal(1, backend.Created);
        Assert.Single(backend.Watches);
    }

    [Fact]
    public void ARangeBelowTheThresholdIsNotCached()
    {
        var cache = Cache(new FakeBackend());
        Assert.False(cache.Acquire(Range, 4096, written: false, 1, 0, out var entry));
        Assert.Null(entry);
    }

    [Fact]
    public void EveryKindOfWriteMarksOverlappingEntriesStaleAndDropsTheirWatch()
    {
        foreach (var write in new Action<MetalStagedRangeCache>[]
        {
            c => Assert.True(c.OnCpuWrite(Range + 100, 8)),
            c => c.OnPreciseCpuWrite(Range + 100, 8, backed: true),
            c => c.OnHostWrite(Range + 100, 8),
            c => c.OnGpuWrite(Range + 100, 8, MetalGpuWriteOrigin.Fill),
        })
        {
            var backend = new FakeBackend();
            var cache = Cache(backend);
            cache.Acquire(Range, Size, written: false, 1, 0, out var entry);
            write(cache);
            Assert.Equal(1, entry!.Stale);
            Assert.Empty(backend.Watches);
            Assert.False(cache.Acquire(Range, Size, written: false, 2, 2, out _));
            Assert.Single(backend.Watches);
        }
    }

    [Fact]
    public void AWriteBesideAnEntryLeavesItCurrentButAFaultOnItsPageDoesNot()
    {
        var backend = new FakeBackend();
        var cache = Cache(backend);
        cache.Acquire(Range + 100, Size, written: false, 1, 0, out var entry);
        cache.OnHostWrite(Range, 100);
        cache.OnPreciseCpuWrite(Range + 100 + Size, 4, backed: true);
        Assert.Equal(0, entry!.Stale);
        // A fault is page-sized: the store can only land once the page's watch is gone.
        Assert.True(cache.OnCpuWrite(Range + 100 + Size, 4));
        Assert.Equal(1, entry.Stale);
    }

    [Fact]
    public void AnEntryWithItsOwnStoresIsKeptUntilTheyAreWrittenBack()
    {
        var backend = new FakeBackend();
        var cache = Cache(backend);
        cache.Acquire(Range, Size, written: true, 1, 0, out var entry);
        entry!.Dirty = true;
        cache.OnGpuWrite(Range, 16, MetalGpuWriteOrigin.Fill);
        cache.OnCpuWrite(Range, 8);
        Assert.Equal(0, entry.Stale);
        Assert.True(cache.Acquire(Range, Size, written: true, 2, 1, out _));
        cache.OnWrittenBack(entry);
        Assert.Equal(1, entry.Stale);
        Assert.False(cache.Acquire(Range, Size, written: false, 3, 3, out _));
    }

    [Fact]
    public void AnEntrysOwnStoresAndWritebackDoNotStaleIt()
    {
        var cache = Cache(new FakeBackend());
        cache.Acquire(Range, Size, written: true, 1, 0, out var entry);
        entry!.Dirty = true;
        cache.OnGpuWrite(Range, Size, MetalGpuWriteOrigin.StorageBuffer);
        cache.MarkSelfWriteback(entry);
        cache.OnGpuWrite(Range, Size, MetalGpuWriteOrigin.StorageBuffer);
        cache.OnWrittenBack(entry);
        Assert.Equal(0, entry.Stale);
        Assert.True(cache.Acquire(Range, Size, written: false, 2, 1, out _));
    }

    [Fact]
    public void ABufferAnInFlightSubmissionReadsIsNeverRewritten()
    {
        var backend = new FakeBackend();
        var cache = Cache(backend);
        cache.Acquire(Range, Size, written: false, 5, 0, out var entry);
        var before = entry!.Staging.Buffer;
        cache.OnHostWrite(Range, 8);

        // Submission 5 has not completed: the restaged entry gets another buffer.
        cache.Acquire(Range, Size, written: false, 6, 4, out _);
        Assert.NotEqual(before, entry.Staging.Buffer);
        Assert.Equal(1, cache.RetiredCount);

        // Once 5 completes its buffer is reusable: submission 6's restage retires, 7's reuses.
        cache.OnHostWrite(Range, 8);
        cache.Acquire(Range, Size, written: false, 7, 6, out _);
        Assert.Equal(2, backend.Created);
    }

    [Fact]
    public void OverBudgetTheLeastRecentlyUsedIdleEntryGoesAndNothingInFlightDoes()
    {
        var backend = new FakeBackend();
        var cache = Cache(backend, budgetMiB: 2);
        cache.Acquire(Range, Size, written: false, 1, 0, out var oldest);
        cache.Acquire(Range + 8 * Mib, Size, written: false, 2, 0, out var newer);

        // Both are still in flight: the cache is full and says so.
        Assert.False(cache.Acquire(Range + 16 * Mib, Size, written: false, 3, 0, out var refused));
        Assert.Null(refused);

        // With submission 1 done, the oldest goes.
        Assert.False(cache.Acquire(Range + 16 * Mib, Size, written: false, 3, 1, out var third));
        Assert.NotNull(third);
        Assert.True(oldest!.Removed);
        Assert.False(newer!.Removed);
        Assert.Equal(1, cache.TakeCounters().Evictions);
    }

    [Fact]
    public void ADirtyEntryIsNotEvicted()
    {
        var cache = Cache(new FakeBackend(), budgetMiB: 2);
        cache.Acquire(Range, Size, written: true, 1, 0, out var dirty);
        dirty!.Dirty = true;
        Assert.False(cache.Acquire(Range + 8 * Mib, Size, written: false, 2, 5, out var other));
        Assert.Null(other);
        Assert.False(dirty.Removed);
    }

    [Fact]
    public void UnmapDropsEntriesWithTheirWatchAndRetiresTheirBuffersBySerial()
    {
        var backend = new FakeBackend();
        var cache = Cache(backend);
        cache.Acquire(Range, Size, written: true, 4, 0, out var entry);
        entry!.Dirty = true;
        cache.OnUnmap(Range, Size);
        Assert.True(entry.Removed);
        Assert.Empty(backend.Watches);
        Assert.Equal(0, cache.EntryCount);
        Assert.Equal(2, cache.RetiredCount);

        // A new entry cannot take those buffers until submission 4 completes.
        cache.Acquire(Range, Size, written: false, 5, 3, out _);
        Assert.Equal(3, backend.Created);
    }

    [Fact]
    public void EveryLiveEntryHoldsExactlyOneWatchAndAFaultOnADirtyOneReleasesItWithoutStalingIt()
    {
        var backend = new FakeBackend();
        var cache = Cache(backend);
        cache.Acquire(Range, Size, written: false, 1, 0, out _);
        cache.Acquire(Range + 8 * Mib, Size, written: true, 1, 0, out var dirty);
        dirty!.Dirty = true;
        Assert.Equal((2, 2, 1, 0, 0), cache.Gauges());
        Assert.Equal(2, backend.Watches.Count);

        // The store has to be able to land, so the watch goes; the stores keep the entry.
        Assert.True(cache.OnCpuWrite(Range + 8 * Mib, 8));
        Assert.Equal((2, 1, 1, 0, 0), cache.Gauges());
        Assert.Equal(0, dirty.Stale);
        Assert.Single(backend.Watches);
    }

    private static long Sampled(byte[] cached, byte[] guest, bool sample, uint seed) =>
        MetalStagedRangeCache.FirstMismatch(
            cached, (offset, destination) => guest.AsSpan((int)offset, destination.Length).CopyTo(destination),
            sample, seed);

    [Fact]
    public void ASampleSeesAChangeInEitherEdgeAndReportsTheFirstOffset()
    {
        var cached = new byte[2 * Mib];
        Assert.Equal(-1, Sampled(cached, (byte[])cached.Clone(), sample: true, seed: 1));

        var firstEdge = (byte[])cached.Clone();
        firstEdge[100] = 1;
        Assert.Equal(100, Sampled(cached, firstEdge, sample: true, seed: 1));

        var lastEdge = (byte[])cached.Clone();
        lastEdge[^5] = 1;
        lastEdge[^1000] = 1;
        Assert.Equal(cached.Length - 1000, Sampled(cached, lastEdge, sample: true, seed: 1));
    }

    [Fact]
    public void ASampleOfAMiddleChangeFindsItOnlyWhenAChosenPageHoldsIt()
    {
        var cached = new byte[2 * Mib];
        var found = 0;
        for (uint seed = 1; seed <= 64; seed++)
        {
            var guest = (byte[])cached.Clone();
            guest[Mib] = 1;
            var offset = Sampled(cached, guest, sample: true, seed);
            Assert.True(offset == -1 || offset == (long)Mib);
            found += offset >= 0 ? 1 : 0;
        }

        // Four 4 KB pages of 512 slots: some seeds hit, most do not.
        Assert.InRange(found, 0, 32);
        Assert.Equal(Mib, (ulong)Sampled(cached, MiddleChanged(cached), sample: false, seed: 1));
    }

    private static byte[] MiddleChanged(byte[] cached)
    {
        var guest = (byte[])cached.Clone();
        guest[Mib] = 1;
        return guest;
    }

    [Fact]
    public void EntriesLeftStaleAreUnwatchedAndCountedAsStaleNeverAsCurrentWithoutAWatch()
    {
        var backend = new FakeBackend();
        var cache = Cache(backend);
        for (ulong index = 0; index < 5; index++)
        {
            cache.Acquire(Range + index * 8 * Mib, Size, written: false, 1, 0, out _);
        }

        // Three are written over and never bound again, as a loading screen's data is.
        cache.OnHostWrite(Range, 8);
        cache.OnCpuWrite(Range + 8 * Mib, 8);
        cache.OnGpuWrite(Range + 16 * Mib, 8, MetalGpuWriteOrigin.Fill);
        Assert.Equal((5, 2, 0, 3, 0), cache.Gauges());
        Assert.Equal(2, backend.Watches.Count);

        // Binding one again re-arms it; the others stay stale until they are bound.
        Assert.False(cache.Acquire(Range, Size, written: false, 2, 1, out _));
        Assert.Equal((5, 3, 0, 2, 0), cache.Gauges());
    }

    [Fact]
    public void AReadOnlyEntryIsRestagedBeforeAWrittenBindingUsesIt()
    {
        var cache = Cache(new FakeBackend());
        cache.Acquire(Range, Size, written: false, 1, 0, out var entry);
        Assert.False(entry!.Pristine.IsValid);
        Assert.False(cache.Acquire(Range, Size, written: true, 2, 1, out _));
        Assert.True(entry.Pristine.IsValid);
    }
}
