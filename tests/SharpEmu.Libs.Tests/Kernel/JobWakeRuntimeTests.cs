// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Diagnostics;
using System.Runtime.InteropServices;
using SharpEmu.Libs.Kernel;
using Xunit;

namespace SharpEmu.Libs.Tests.Kernel;

public sealed class JobWakeRuntimeTests
{
    private readonly struct FixedWork(ulong mask) : IJobWorkProbe
    {
        public ulong WorkMask() => mask;
    }

    private static readonly FixedWork Everyone = new(ulong.MaxValue);

    [Fact]
    public void AMovedGenerationOrQueuedWorkReturnsBeforeSleeping()
    {
        if (!Initialize())
        {
            return;
        }

        var before = JobWakeRuntime.Snapshot();
        var stale = Generation() - 1;
        Assert.Equal(JobWakeRuntime.WaitOutcome.NoBlock, JobWakeRuntime.Wait<FixedWork>(40, stale, 5_000_000, null));
        Assert.Equal(JobWakeRuntime.WaitOutcome.NoBlock, JobWakeRuntime.Wait(40, Generation(), 5_000_000, (FixedWork?)new FixedWork(1UL << 40)));
        var after = JobWakeRuntime.Snapshot();
        Assert.True(after.NoBlock - before.NoBlock >= 2);
        Assert.Equal(0UL, Sleepers() & (1UL << 40));
    }

    [Fact]
    public void AWaitWithoutAWakeTimesOut()
    {
        if (!Initialize())
        {
            return;
        }

        var watch = Stopwatch.StartNew();
        Assert.Equal(JobWakeRuntime.WaitOutcome.TimedOut, JobWakeRuntime.Wait<FixedWork>(41, Generation(), 20_000, null));
        Assert.True(watch.Elapsed >= TimeSpan.FromMilliseconds(19));
        Assert.Equal(0UL, (Sleepers() | Pollers()) & (1UL << 41));
    }

    [Fact]
    public void AWakeEndsTheWaitLongBeforeItsTimeoutAndOnlyForEligibleSleepers()
    {
        if (!Initialize())
        {
            return;
        }

        const int Worker = 42;
        var bit = 1UL << Worker;
        var watch = Stopwatch.StartNew();
        var waiter = Task.Run(() => JobWakeRuntime.Wait<FixedWork>(Worker, Generation(), 10_000_000, null));
        WaitForSleeper(bit);
        Assert.Equal(bit, Pollers() & bit);

        // Outside the mask, or without queued work: left asleep.
        Assert.Equal(0, JobWakeRuntime.Wake(~bit, 8, Everyone));
        Assert.Equal(0, JobWakeRuntime.Wake(bit, 8, new FixedWork(~bit)));
        Assert.Equal(0, JobWakeRuntime.Wake(bit, 0, Everyone));
        Assert.False(waiter.IsCompleted);

        Assert.Equal(1, JobWakeRuntime.Wake(bit, 8, Everyone));
        Assert.Equal(JobWakeRuntime.WaitOutcome.Woken, waiter.Wait(TimeSpan.FromSeconds(5)) ? waiter.Result : default);
        Assert.True(watch.Elapsed < TimeSpan.FromSeconds(5));
        Assert.Equal(0UL, Sleepers() & bit);
        Assert.Equal(0UL, Pollers() & bit);

        // Already woken: a second wake finds no one.
        Assert.Equal(0, JobWakeRuntime.Wake(bit, 8, Everyone));
    }

    [Fact]
    public void AQueueCheckingSleeperIsNotAPoller()
    {
        if (!Initialize())
        {
            return;
        }

        const int Worker = 45;
        var bit = 1UL << Worker;
        var waiter = Task.Run(() => JobWakeRuntime.Wait(Worker, Generation(), 10_000_000, (FixedWork?)new FixedWork(0)));
        WaitForSleeper(bit);
        Assert.Equal(0UL, Pollers() & bit);
        Assert.Equal(1, JobWakeRuntime.Wake(bit, 1, Everyone));
        Assert.True(waiter.Wait(TimeSpan.FromSeconds(5)));
        Assert.Equal(JobWakeRuntime.WaitOutcome.Woken, waiter.Result);
    }

    [Fact]
    public void AWakeStopsAtItsCount()
    {
        if (!Initialize())
        {
            return;
        }

        const ulong Pair = (1UL << 43) | (1UL << 44);
        var waiters = new[] { 43, 44 }
            .Select(worker => Task.Run(() => JobWakeRuntime.Wait<FixedWork>(worker, Generation(), 10_000_000, null)))
            .ToArray();
        WaitForSleeper(Pair);

        Assert.Equal(1, JobWakeRuntime.Wake(Pair, 1, Everyone));
        Assert.True(Task.WaitAny(waiters, TimeSpan.FromSeconds(5)) >= 0);
        Assert.Equal(1, BitCount(Sleepers() & Pair));
        Assert.Equal(1, JobWakeRuntime.Wake(Pair, 1, Everyone));
        Assert.True(Task.WaitAll(waiters, TimeSpan.FromSeconds(5)));
        Assert.All(waiters, waiter => Assert.Equal(JobWakeRuntime.WaitOutcome.Woken, waiter.Result));
    }

    [Fact]
    public void CountersFormatAsDeltas()
    {
        var line = JobWakeRuntime.Format(new(10, 9, 8, 7, 6, 5, 4, 3), new(1, 1, 1, 1, 1, 1, 1, 1));
        Assert.Equal(
            "[LOADER][INFO] job wake: signals=9 wake_calls=8 woke=7 waits=6 woken_early=5 timed_out=4 no_block=3 ab_naps=2",
            line);
    }

    [Fact]
    public void GuestQueuesReportWorkersWithAPositiveCountOrAMailbox()
    {
        using var memory = new SharpEmu.Core.Memory.PhysicalVirtualMemory();
        Assert.True(memory.TryAllocateAtOrAbove(0x300000000UL, 0x40000, executable: false, 0x4000, out var region));
        var scheduler = region;
        var groups = region + 0x10000;
        Write64(memory, scheduler + GuestJobQueues.WorkerCountOffset, 13);
        Write32(memory, scheduler + GuestJobQueues.GroupCountOffset, 2);
        Write64(memory, scheduler + GuestJobQueues.GroupsOffset, groups);
        Write64(memory, scheduler + GuestJobQueues.MailboxOffset + (3 * 8), 0xABC);
        Write32(memory, groups + GuestJobQueues.CountsOffset + (7 * 4), unchecked((uint)-1));
        Write32(memory, groups + GuestJobQueues.GroupStride + GuestJobQueues.CountsOffset + (5 * 4), 2);

        // Worker 13 is past the pool, so its count is not read.
        Write32(memory, groups + GuestJobQueues.CountsOffset + (13 * 4), 1);

        Assert.Equal((1UL << 3) | (1UL << 5), new GuestJobQueues(memory, scheduler).WorkMask());
        Assert.Equal(0UL, new GuestJobQueues(memory, 0x10).WorkMask());
    }

    private static bool Initialize() => JobWakeRuntime.IsSupported && JobWakeRuntime.TryInitialize(out _);

    private static long Generation() => Marshal.ReadInt64((nint)JobWakeRuntime.GenerationAddress);

    private static ulong Sleepers() => (ulong)Marshal.ReadInt64((nint)JobWakeRuntime.SleepersAddress);

    private static ulong Pollers() => (ulong)Marshal.ReadInt64((nint)JobWakeRuntime.PollersAddress);

    private static int BitCount(ulong value) => System.Numerics.BitOperations.PopCount(value);

    private static void WaitForSleeper(ulong bits)
    {
        var watch = Stopwatch.StartNew();
        while ((Sleepers() & bits) != bits)
        {
            Assert.True(watch.Elapsed < TimeSpan.FromSeconds(5), "the waiter never went to sleep");
            Thread.Sleep(1);
        }

        // Past the bit: give the waiter time to reach the kernel wait (a wake before it gets
        // there is not lost either, the word just moves first).
        Thread.Sleep(20);
    }

    private static void Write64(SharpEmu.Core.Memory.PhysicalVirtualMemory memory, ulong address, ulong value)
    {
        var bytes = new byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(bytes, value);
        Assert.True(memory.TryWrite(address, bytes));
    }

    private static void Write32(SharpEmu.Core.Memory.PhysicalVirtualMemory memory, ulong address, uint value)
    {
        var bytes = new byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes, value);
        Assert.True(memory.TryWrite(address, bytes));
    }
}
