// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.GpuCommands;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.GpuCommands;

// Labels the guest's CPU writes reach no notification; a parked wait on one is re-read.
public sealed class CommandStreamQueueWatchTests
{
    private const ulong Label = StreamRunner.LabelAddress;
    private const ulong Graphics = StreamRunner.CommandAddress;

    private static uint[] Instances(uint count) => StreamRunner.Packet(PacketOpcode.NumInstances, count);

    private static uint[] WaitEqual(ulong address, uint reference) =>
        StreamRunner.Packet(PacketOpcode.WaitRegisterMemory, 0x13u, StreamRunner.Low(address), StreamRunner.High(address), reference, 0xFFFF_FFFFu, 0);

    private static void Enqueue(RecordingCommandStreamHost host, CommandStreamQueue queue, ulong submissionId, params uint[][] packets)
    {
        var words = StreamRunner.Concat(packets);
        host.WriteWords(Graphics, words);
        queue.EnqueueGraphics(Graphics, (uint)words.Length, submissionId, null);
    }

    [Fact]
    public void ACpuStoreThatPassesTheCompareWakesTheHead()
    {
        var host = new RecordingCommandStreamHost();
        var queue = new CommandStreamQueue(host);
        host.WriteDword(Label, 0);
        Enqueue(host, queue, 1, WaitEqual(Label, 2), Instances(5));
        Assert.Equal(SliceResult.BlockedWithoutProgress, queue.ProcessOne());

        Assert.False(queue.RecheckWatched());

        // A change that still fails the compare wakes nothing.
        host.WriteDword(Label, 1);
        Assert.False(queue.RecheckWatched());
        Assert.Equal(SliceResult.AllBlocked, queue.ProcessOne());

        host.WriteDword(Label, 2);
        Assert.True(queue.RecheckWatched());
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.Equal(5u, queue.GetInterpreter(0).InstanceCount);
        Assert.Equal(1UL, queue.WatchWakes);
    }

    [Fact]
    public void WaitForWorkReturnsWithinTheWatchIntervalOfTheStore()
    {
        var host = new RecordingCommandStreamHost();
        var queue = new CommandStreamQueue(host);
        host.WriteDword(Label, 0);
        Enqueue(host, queue, 1, WaitEqual(Label, 1), Instances(5));
        Assert.Equal(SliceResult.BlockedWithoutProgress, queue.ProcessOne());

        // The store comes from its own thread and the latency is taken from the store, not from the
        // start: a pool timer can fire seconds late on a loaded runner, which is not the watch
        // being slow. The wait timeout only guards against a watch that never wakes.
        long storedAt = 0;
        var writer = new Thread(() =>
        {
            Thread.Sleep(20);
            host.WriteDword(Label, 1);
            Volatile.Write(ref storedAt, System.Diagnostics.Stopwatch.GetTimestamp());
        }) { IsBackground = true };
        writer.Start();
        Assert.True(queue.WaitForWork(10_000));
        var woke = System.Diagnostics.Stopwatch.GetTimestamp();
        writer.Join();
        var latencyMilliseconds = (woke - Volatile.Read(ref storedAt)) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
        Assert.True(latencyMilliseconds < CommandStreamQueue.AllBlockedRetryMilliseconds * 5, $"woke {latencyMilliseconds:F1} ms after the store");
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
    }

    [Fact]
    public void ASemaphoreSignalFromTheCpuWakesTheWaiter()
    {
        var host = new RecordingCommandStreamHost();
        var queue = new CommandStreamQueue(host);
        host.WriteQword(Label, 0);
        // MEM_SEMAPHORE wait (select 7 in bits 29-31).
        var wait = StreamRunner.Packet(PacketOpcode.MemorySemaphore, StreamRunner.Low(Label), StreamRunner.High(Label), 7u << 29);
        Enqueue(host, queue, 1, wait, Instances(4));
        Assert.Equal(SliceResult.BlockedWithoutProgress, queue.ProcessOne());

        host.WriteQword(Label, 1);
        Assert.True(queue.RecheckWatched());
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.Equal(0UL, host.ReadQword(Label));
    }

    [Fact]
    public void AGpuTickLeavesEngineCounterAndPipelineWaitsAsleep()
    {
        var host = new RecordingCommandStreamHost();
        var queue = new CommandStreamQueue(host);
        // WAIT_ON_DE_COUNTER_DIFF of zero: the counters are level, so it parks.
        Enqueue(host, queue, 1, StreamRunner.Packet(PacketOpcode.WaitOnDeCounterDiff, 0), Instances(4));
        Assert.Equal(SliceResult.BlockedWithoutProgress, queue.ProcessOne());

        var skipped = queue.TargetedWakesSkipped;
        queue.NotifyCompletedGpuTick(1);
        Assert.Equal(skipped + 1, queue.TargetedWakesSkipped);
        Assert.Equal(SliceResult.AllBlocked, queue.ProcessOne());
    }

    [Fact]
    public void AGpuCompletionSignalWakesAParkedWatchedWaitWithoutTheWatchTick()
    {
        var host = new RecordingCommandStreamHost();
        var queue = new CommandStreamQueue(host) { WatchIntervalOverrideMilliseconds = 30_000 };
        host.WriteDword(Label, 0);
        Enqueue(host, queue, 1, WaitEqual(Label, 1), Instances(5));
        Assert.Equal(SliceResult.BlockedWithoutProgress, queue.ProcessOne());

        // Nothing is parked yet: the callback costs nothing and signals nothing.
        queue.NotifyGpuCompletionObserved();
        Assert.Equal(0, queue.GpuCompletionWakes);

        var parked = Task.Run(() => queue.WaitForWork(60_000));
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        while (!queue.ParkedOnWatch && stopwatch.Elapsed < TimeSpan.FromSeconds(20))
        {
            Thread.Sleep(1);
        }

        Assert.True(queue.ParkedOnWatch);
        // Let it get from raising the flag to sleeping: a label that lands during its first read
        // would end the wait without any signal, which is the other path.
        Thread.Sleep(200);
        Assert.False(parked.IsCompleted);

        // The label lands while the worker sleeps, then the completion callback fires.
        host.WriteDword(Label, 1);
        queue.NotifyGpuCompletionObserved();

        Assert.True(parked.Wait(TimeSpan.FromSeconds(20)), "the parked wait never woke: only the 30 s watch tick was left");
        Assert.True(parked.Result);
        Assert.Equal(1, queue.GpuCompletionWakes);
        Assert.False(queue.ParkedOnWatch);
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
    }

    [Fact]
    public void ACompletionThatLandedBeforeTheWaitParkedIsSeenByTheFirstRead()
    {
        var host = new RecordingCommandStreamHost();
        var queue = new CommandStreamQueue(host) { WatchIntervalOverrideMilliseconds = 30_000 };
        host.WriteDword(Label, 0);
        Enqueue(host, queue, 1, WaitEqual(Label, 1), Instances(5));
        Assert.Equal(SliceResult.BlockedWithoutProgress, queue.ProcessOne());

        // The callback found the worker still running and signalled nothing; the label is already there.
        host.WriteDword(Label, 1);
        queue.NotifyGpuCompletionObserved();
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        Assert.True(queue.WaitForWork(60_000));
        Assert.True(stopwatch.Elapsed < TimeSpan.FromSeconds(5));
    }
}
