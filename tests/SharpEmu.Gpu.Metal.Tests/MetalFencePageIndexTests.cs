// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

public sealed class MetalFencePageIndexTests
{
    [Fact]
    public void InvalidRangesAndSerialOverflowRefusePublicationAndNeverRejectAQuery()
    {
        var index = new MetalFencePageIndex();
        foreach (var (address, size) in new[] { (0UL, 0UL), (ulong.MaxValue, 2UL), (1UL << 40, 1UL), ((1UL << 40) - 1, 2UL) })
        {
            Assert.False(index.Publish(address, size, 1));
            Assert.True(index.MayOverlap(address, size));
        }
        Assert.False(index.Publish(0, 1, ulong.MaxValue));
        Assert.False(index.Publish(0, 1, 0));
    }

    [Fact]
    public void BoundariesMissingLeavesAndUnrelatedLiveFences()
    {
        var index = new MetalFencePageIndex();
        Assert.False(index.MayOverlap(0, 8));
        Assert.True(index.Publish(4095, 2, 10));
        Assert.True(index.MayOverlap(0, 1));
        Assert.True(index.MayOverlap(8191, 1));
        Assert.False(index.MayOverlap(8192, 1));
        Assert.False(index.MayOverlap(0, 8192, completed: 10));
        Assert.True(index.Publish((1UL << 40) - 1, 1, 11));
        Assert.True(index.MayOverlap((1UL << 40) - 4096, 4096));
    }

    [Fact]
    public void RetirementNeverClearsANewerGenerationOrAnIncompleteFence()
    {
        var index = new MetalFencePageIndex();
        index.Publish(4095, 2, 10);
        index.Publish(4096, 4096, 11);
        index.Retire(4095, 2, 10, completed: 10);
        Assert.False(index.MayOverlap(0, 1));
        Assert.True(index.MayOverlap(4096, 1));
        index.Retire(4096, 4096, 11, completed: 10);
        Assert.True(index.MayOverlap(4096, 1));
        index.Retire(4096, 4096, 11, completed: 11);
        Assert.False(index.MayOverlap(4096, 1));
    }

    [Fact]
    public void RejectionQueriesAllocateNoMemory()
    {
        var index = new MetalFencePageIndex();
        index.Publish(4096, 8, 1);
        for (var i = 0; i < 100; i++) _ = index.MayOverlap(8192, 8);
        var before = GC.GetAllocatedBytesForCurrentThread();
        var hits = 0;
        for (var i = 0; i < 10000; i++) if (index.MayOverlap(8192, 8)) hits++;
        var allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.Equal(0, hits);
        Assert.Equal(0, allocated);
    }

    // Enumerate all orders of old retirement, newer publication and queries.
    // The submission generation supplies the ABA protection; no timing assumption.
    [Fact]
    public void EverySmallStateRetirePublishOrderingPreservesThePendingMaximum()
    {
        foreach (var order in new[] { new[] { 0, 1, 2 }, new[] { 0, 2, 1 }, new[] { 1, 0, 2 }, new[] { 1, 2, 0 }, new[] { 2, 0, 1 }, new[] { 2, 1, 0 } })
        {
            var index = new MetalFencePageIndex();
            index.Publish(0, 4096, 1);
            var published = false;
            foreach (var step in order)
            {
                if (step == 0) index.Retire(0, 4096, 1, completed: 1);
                if (step == 1) { index.Publish(0, 4096, 2); published = true; }
                if (published) Assert.True(index.MayOverlap(0, 4096, completed: 1));
            }
        }
    }

    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(2)] [InlineData(3)] [InlineData(4)]
    [InlineData(5)] [InlineData(6)] [InlineData(7)] [InlineData(8)] [InlineData(9)]
    [InlineData(10)] [InlineData(11)] [InlineData(12)] [InlineData(13)] [InlineData(14)]
    [InlineData(15)] [InlineData(16)] [InlineData(17)] [InlineData(18)] [InlineData(19)]
    public void ThirteenProducersWorkerAndCompletionStress(int seed)
    {
        const int operations = 1_000_000;
        var index = new MetalFencePageIndex();
        var generations = new long[32];
        long completed = 0;
        var history = new long[64];
        using var start = new Barrier(15);
        using var cancel = new CancellationTokenSource(TimeSpan.FromSeconds(60));
        Task Run(Action action) => Task.Factory.StartNew(action, cancel.Token,
            TaskCreationOptions.LongRunning, TaskScheduler.Default);
        var tasks = new List<Task>();
        // A single fake worker publishes before making each writeback observable.
        tasks.Add(Run(() =>
        {
            start.SignalAndWait(cancel.Token);
            for (var i = 1; i <= operations / 15; i++)
            {
                if ((i & 1023) == 0) cancel.Token.ThrowIfCancellationRequested();
                var slot = (i + seed) & 31;
                var serial = (long)i;
                index.Publish((ulong)slot * 4096, (i & 1) == 0 ? 4097UL : 8UL, (ulong)serial);
                Volatile.Write(ref history[i & 63], (serial << 8) | (uint)slot);
                Volatile.Write(ref generations[slot], serial);
                if (((i * 17 + seed) & 127) == 0) Thread.Yield();
            }
        }));
        tasks.Add(Run(() =>
        {
            start.SignalAndWait(cancel.Token);
            for (var i = 0; i < operations / 15; i++)
            {
                var slot = (i * 7 + seed) & 31;
                var serial = Volatile.Read(ref generations[slot]);
                // Completion is real logical time, and cannot retire a newer serial.
                Volatile.Write(ref completed, Math.Max(Volatile.Read(ref completed), serial));
                index.Retire((ulong)slot * 4096, 4096, (ulong)serial, (ulong)serial);
                if ((i & 127) == 0) Thread.Yield();
            }
        }));
        for (var producer = 0; producer < 13; producer++)
        {
            var id = producer;
            tasks.Add(Run(() =>
            {
                var random = new Random(seed * 31 + id);
                start.SignalAndWait(cancel.Token);
                for (var i = 0; i < operations / 13; i++)
                {
                    if ((i & 1023) == 0) cancel.Token.ThrowIfCancellationRequested();
                    // Hot page, neighbours sharing cache lines, disjoint pages and
                    // opposite-order cross-page queries (including alias-equivalent offsets).
                    var slot = i % 4 == 0 ? 0 : random.Next(32);
                    var before = Volatile.Read(ref generations[slot]);
                    var overlaps = index.MayOverlap((ulong)slot * 4096, 8);
                    var after = Volatile.Read(ref generations[slot]);
                    var floor = Volatile.Read(ref completed);
                    if (before == after && after > floor && !overlaps)
                        throw new InvalidOperationException($"seed={seed} producer={id} slot={slot} serial={after} completed={floor} history={string.Join(',', history)}");
                    if ((i & 127) == 0) Thread.Yield();
                }
            }));
        }
        Assert.True(Task.WaitAll(tasks.ToArray(), TimeSpan.FromSeconds(65)), $"seed={seed} deadline history={string.Join(',', history)}");
        // Retire the actual maximum after all publishers quiesce.
        index.Retire(0, 33 * 4096, (ulong)(operations / 15), (ulong)(operations / 15));
        Assert.False(index.MayOverlap(0x100000, 8));
    }
}
