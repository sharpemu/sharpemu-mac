// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.GpuCommands;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.GpuCommands;

public sealed class CommandIssueRingTests
{
    private sealed class Item { internal int Value; }

    [Fact]
    public void FullRingBackpressuresUntilConsumerReleasesItsOldestBlock()
    {
        var ring = new CommandIssueRing<Item>(2, 8, () => new());
        for (var i = 0; i < 2; i++)
        {
            var block = ring.Acquire(); block.Count = 8; block.LastSequence = i + 1; ring.Publish(block);
        }
        using var entered = new ManualResetEventSlim();
        using var acquired = new ManualResetEventSlim();
        var producer = new Thread(() => { entered.Set(); ring.Acquire(); acquired.Set(); });
        producer.Start();
        Assert.True(entered.Wait(TimeSpan.FromSeconds(5)));
        Assert.True(SpinWait.SpinUntil(() => Interlocked.Read(ref ring.FullStalls) != 0, TimeSpan.FromSeconds(5)));
        Assert.False(acquired.IsSet);
        ring.Release(ring.Take()!);
        Assert.True(acquired.Wait(TimeSpan.FromSeconds(5)));
        Assert.True(producer.Join(5000));
        ring.Stop();
    }

    [Fact]
    public void EmptyFullAndAcknowledgementTransitionsNeverLoseAWakeup()
    {
        const int count = 100_000;
        var ring = new CommandIssueRing<Item>(3, 7, () => new());
        var expected = 0;
        Exception? failure = null;
        var consumer = new Thread(() =>
        {
            try
            {
                while (ring.Take() is { } block)
                {
                    for (var i = 0; i < block.Count; i++)
                    {
                        Assert.Equal(expected++, block.Items[i].Value);
                        if ((expected & 127) == 0) Thread.Yield();
                    }
                    ring.Release(block);
                }
            }
            catch (Exception error) { failure = error; ring.Fail(error); }
        });
        consumer.Start();
        try
        {
            for (var next = 0; next < count;)
            {
                var block = ring.Acquire();
                var n = Math.Min(1 + next % 7, count - next);
                block.Count = n;
                for (var i = 0; i < n; i++) block.Items[i].Value = next++;
                block.LastSequence = next;
                ring.Publish(block);
                if ((next & 31) == 0) { Thread.Yield(); ring.WaitFor(next); }
            }
            ring.WaitFor(count);
        }
        finally { ring.Stop(); }
        Assert.True(consumer.Join(10000));
        Assert.Null(failure);
        Assert.Equal(count, expected);
    }

    [Fact]
    public void PublishingWhileConsumerIsBusyDoesNotWakeItPerBlock()
    {
        var ring = new CommandIssueRing<Item>(4, 16, () => new());
        // No consumer is parked. The release/acquire indices are sufficient.
        for (var index = 0; index < 4; index++)
        {
            var block = ring.Acquire();
            block.Count = 16;
            block.LastSequence = (index + 1) * 16;
            ring.Publish(block);
        }
        Assert.Equal(4, ring.PublishedBlocks);
        Assert.Equal(0, ring.Wakeups);
        for (var index = 0; index < 4; index++) ring.Release(ring.Take()!);
        Assert.False(ring.WaitFor(64));
        ring.Stop();
    }

    [Fact]
    public void StopReleasesAProducerBlockedOnTheFullRing()
    {
        var ring = new CommandIssueRing<Item>(2, 1, () => new());
        for (var index = 0; index < 2; index++)
        {
            var block = ring.Acquire(); block.Count = 1; block.LastSequence = index + 1; ring.Publish(block);
        }
        var producer = Task.Run(() => Assert.Throws<OperationCanceledException>(() => ring.Acquire()));
        Assert.True(SpinWait.SpinUntil(() => Interlocked.Read(ref ring.FullStalls) != 0, TimeSpan.FromSeconds(5)));
        ring.Stop();
        Assert.True(producer.Wait(TimeSpan.FromSeconds(5)));
    }

    [Fact]
    public void FailureReleasesPrefixAndBackpressureWaitersWithTheOriginalCause()
    {
        var ring = new CommandIssueRing<Item>(2, 1, () => new());
        var error = new InvalidOperationException("consumer failed before acknowledgement");
        var waiter = Task.Run(() => Assert.Same(error, Assert.Throws<InvalidOperationException>(() => ring.WaitFor(1))));
        Assert.True(SpinWait.SpinUntil(() => Interlocked.Read(ref ring.SyncStalls) != 0, TimeSpan.FromSeconds(5)));
        ring.Fail(error);
        Assert.True(waiter.Wait(TimeSpan.FromSeconds(5)));
        Assert.Same(error, Assert.Throws<InvalidOperationException>(() => ring.Acquire()));
    }

    [Fact]
    public void StopDoesNotAcknowledgeAnUnexecutedPrefix()
    {
        var ring = new CommandIssueRing<Item>(2, 1, () => new());
        ring.Stop();
        Assert.Throws<OperationCanceledException>(() => ring.WaitFor(1));
    }
}
