// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Xunit;

namespace SharpEmu.GpuSplit.Tests;

/// <summary>
/// The queue the events thread takes the GPU process's events from: in order, parked without a
/// spin, and ended by Complete the way CompleteAdding ended the collection before it.
/// </summary>
public sealed class SplitEventQueueTests
{
    private static readonly TimeSpan Wait = TimeSpan.FromSeconds(30);

    [Fact]
    public void KeepsTheOrderUnderAConcurrentProducer()
    {
        const int Count = 50_000;
        var queue = new SplitEventQueue();
        var seen = new List<int>(Count);
        var consumer = Task.Factory.StartNew(() =>
        {
            while (queue.Take(out var work))
            {
                work();
            }
        }, TaskCreationOptions.LongRunning);
        var producer = Task.Factory.StartNew(() =>
        {
            for (var index = 0; index < Count; index++)
            {
                var value = index;
                Assert.True(queue.Post(() => seen.Add(value)));
            }

            queue.Complete();
        }, TaskCreationOptions.LongRunning);

        Assert.True(Task.WaitAll([producer, consumer], Wait));
        Assert.Equal(Count, seen.Count);
        for (var index = 0; index < Count; index++)
        {
            Assert.Equal(index, seen[index]);
        }
    }

    [Fact]
    public void ABlockedConsumerWakesOnPost()
    {
        var queue = new SplitEventQueue();
        using var ran = new ManualResetEventSlim(false);
        var consumer = Task.Factory.StartNew(() =>
        {
            Assert.True(queue.Take(out var work));
            work();
        }, TaskCreationOptions.LongRunning);

        // The consumer is, or soon is, parked on an empty queue.
        Thread.Sleep(100);
        Assert.False(consumer.IsCompleted);
        Assert.True(queue.Post(ran.Set));

        Assert.True(ran.Wait(Wait));
        Assert.True(consumer.Wait(Wait));
    }

    [Fact]
    public void CompleteWakesABlockedConsumerAndLaterPostsAreDropped()
    {
        var queue = new SplitEventQueue();
        var consumer = Task.Factory.StartNew(() => queue.Take(out _), TaskCreationOptions.LongRunning);

        Thread.Sleep(100);
        Assert.False(consumer.IsCompleted);
        queue.Complete();

        Assert.True(consumer.Wait(Wait));
        Assert.False(consumer.Result);
        var ran = false;
        Assert.False(queue.Post(() => ran = true));
        Assert.False(queue.Take(out _));
        Assert.False(ran);
    }

    [Fact]
    public void WhatWasPostedBeforeCompleteIsStillTaken()
    {
        var queue = new SplitEventQueue();
        var seen = new List<int>();
        for (var index = 0; index < 100; index++)
        {
            var value = index;
            Assert.True(queue.Post(() => seen.Add(value)));
        }

        queue.Complete();
        Assert.False(queue.Post(() => seen.Add(-1)));

        // A consumer that only starts after the end still gets all of it, then ends.
        var taken = 0;
        while (queue.Take(out var work))
        {
            work();
            taken++;
        }

        Assert.Equal(100, taken);
        Assert.Equal(Enumerable.Range(0, 100), seen);
    }

    [Fact]
    public void PostsRacingCompleteAreEitherTakenOrRefused()
    {
        for (var round = 0; round < 50; round++)
        {
            var queue = new SplitEventQueue();
            var accepted = 0;
            var ran = 0;
            var consumer = Task.Factory.StartNew(() =>
            {
                while (queue.Take(out var work))
                {
                    work();
                }
            }, TaskCreationOptions.LongRunning);
            var producers = Enumerable.Range(0, 3).Select(_ => Task.Factory.StartNew(() =>
            {
                for (var index = 0; index < 2_000; index++)
                {
                    if (queue.Post(() => Interlocked.Increment(ref ran)))
                    {
                        Interlocked.Increment(ref accepted);
                    }
                }
            }, TaskCreationOptions.LongRunning)).ToArray();

            Thread.Sleep(1);
            queue.Complete();

            Assert.True(Task.WaitAll([.. producers, consumer], Wait));
            Assert.Equal(accepted, ran);
        }
    }

    [Fact]
    public void AnEventPostedFromTheConsumerRunsAfterTheOnesAlreadyThere()
    {
        var queue = new SplitEventQueue();
        var seen = new List<string>();
        Assert.True(queue.Post(() =>
        {
            seen.Add("first");
            Assert.True(queue.Post(() => seen.Add("third")));
        }));
        Assert.True(queue.Post(() => seen.Add("second")));

        // Ends after the nested one: a Post from an event is taken while the queue is open.
        Assert.True(queue.Post(() => queue.Post(queue.Complete)));
        while (queue.Take(out var work))
        {
            work();
        }

        Assert.Equal(["first", "second", "third"], seen);
    }
}
