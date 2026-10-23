// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Core.Cpu.Native;
using Xunit;

namespace SharpEmu.Libs.Tests.Cpu;

// The ready dispatcher parks on its own event; producers wake it under the gate. A wake must
// never be lost, whichever side reaches the gate first.
public sealed class ReadyDispatchSignalTests
{
    [Fact]
    public void ASignalBeforeThePark_IsNotLost()
    {
        var gate = new object();
        var signal = new ReadyDispatchSignal();
        lock (gate)
        {
            signal.MarkParkedLocked();
        }

        lock (gate)
        {
            Assert.True(signal.TrySignalLocked());
            Assert.False(signal.IsParkedLocked);
            // Only the first enqueuer after the park pays for the event.
            Assert.False(signal.TrySignalLocked());
        }

        // The dispatcher reaches its wait after the signal landed: it returns at once.
        Assert.True(signal.Park(10_000));
    }

    [Fact]
    public void ADispatcherThatIsNotParkedIsNotSignalled()
    {
        var gate = new object();
        var signal = new ReadyDispatchSignal();
        lock (gate)
        {
            Assert.False(signal.TrySignalLocked());
        }

        Assert.False(signal.Park(0));
    }

    [Fact]
    public void AParkedWaiterWakesPromptlyOnASignal()
    {
        var gate = new object();
        var signal = new ReadyDispatchSignal();
        var queue = new Queue<int>();
        var got = Task.Run(() => Consume(gate, signal, queue, 1, Timeout.Infinite).Count);
        Thread.Sleep(50);
        var started = System.Diagnostics.Stopwatch.GetTimestamp();
        Produce(gate, signal, queue, 7);
        Assert.True(got.Wait(TimeSpan.FromSeconds(20)), "the parked dispatcher never woke");
        Assert.Equal(1, got.Result);
        Assert.True(System.Diagnostics.Stopwatch.GetElapsedTime(started) < TimeSpan.FromSeconds(5));
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(4, 2)]
    [InlineData(8, 3)]
    public void ProducersAndConsumersLoseNoWakeups(int producers, int pollers)
    {
        // One dispatcher that parks (it waits with no timeout, so a lost wake hangs it), plus
        // pollers that take items without parking, like the Pump path draining the same queue.
        const int PerProducer = 20_000;
        var total = producers * PerProducer;
        var gate = new object();
        var signal = new ReadyDispatchSignal();
        var queue = new Queue<int>();
        var consumed = 0;
        var done = new ManualResetEventSlim(false);

        void Note(int count)
        {
            if (Interlocked.Add(ref consumed, count) == total)
            {
                done.Set();
            }
        }

        var dispatcher = new Thread(() =>
        {
            while (!done.IsSet)
            {
                bool wait = false;
                var taken = 0;
                lock (gate)
                {
                    while (queue.TryDequeue(out _))
                    {
                        taken++;
                    }

                    if (taken == 0)
                    {
                        signal.MarkParkedLocked();
                        wait = true;
                    }
                }

                if (taken != 0)
                {
                    Note(taken);
                }

                if (wait)
                {
                    _ = signal.Park(Timeout.Infinite);
                    lock (gate)
                    {
                        signal.ClearParkedLocked();
                    }
                }
            }
        }) { IsBackground = true };

        var pollerThreads = Enumerable.Range(0, pollers).Select(_ => new Thread(() =>
        {
            while (!done.IsSet)
            {
                var taken = 0;
                lock (gate)
                {
                    if (queue.TryDequeue(out _))
                    {
                        taken = 1;
                    }
                }

                if (taken != 0)
                {
                    Note(taken);
                }
                else
                {
                    Thread.Yield();
                }
            }
        }) { IsBackground = true }).ToArray();

        var producerThreads = Enumerable.Range(0, producers).Select(p => new Thread(() =>
        {
            for (var i = 0; i < PerProducer; i++)
            {
                Produce(gate, signal, queue, i);
                if (i % 64 == 0)
                {
                    // Let the dispatcher drain and park, so parks and signals interleave.
                    Thread.Sleep(0);
                }
            }
        }) { IsBackground = true }).ToArray();

        dispatcher.Start();
        foreach (var poller in pollerThreads)
        {
            poller.Start();
        }

        foreach (var producer in producerThreads)
        {
            producer.Start();
        }

        var finished = done.Wait(TimeSpan.FromSeconds(60));
        if (!finished)
        {
            // Release the stuck threads so the failure reports instead of leaking them.
            signal.SignalAlways();
        }

        Assert.True(finished, $"only {Volatile.Read(ref consumed)} of {total} items were taken: a wake was lost");
        signal.SignalAlways();
        Assert.True(dispatcher.Join(TimeSpan.FromSeconds(20)));
        foreach (var poller in pollerThreads)
        {
            Assert.True(poller.Join(TimeSpan.FromSeconds(20)));
        }

        foreach (var producer in producerThreads)
        {
            Assert.True(producer.Join(TimeSpan.FromSeconds(20)));
        }

        Assert.Equal(total, consumed);
    }

    private static void Produce(object gate, ReadyDispatchSignal signal, Queue<int> queue, int item)
    {
        lock (gate)
        {
            queue.Enqueue(item);
            _ = signal.TrySignalLocked();
        }
    }

    private static List<int> Consume(object gate, ReadyDispatchSignal signal, Queue<int> queue, int want, int timeout)
    {
        var items = new List<int>();
        while (items.Count < want)
        {
            var wait = false;
            lock (gate)
            {
                while (queue.TryDequeue(out var item))
                {
                    items.Add(item);
                }

                if (items.Count < want)
                {
                    signal.MarkParkedLocked();
                    wait = true;
                }
            }

            if (wait)
            {
                _ = signal.Park(timeout);
                lock (gate)
                {
                    signal.ClearParkedLocked();
                }
            }
        }

        return items;
    }
}
