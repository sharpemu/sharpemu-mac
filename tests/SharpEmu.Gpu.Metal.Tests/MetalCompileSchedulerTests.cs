// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using SharpEmu.Gpu.Metal.V2;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

// The game's compiles go ahead of the background prewarm's: a prewarm compile starts only
// while none of the game's is pending and none just finished, at most its share at once,
// and a prewarm held back for the starvation limit still gets one compile through at a time.
public sealed class MetalCompileSchedulerTests
{
    private static readonly TimeSpan Held = TimeSpan.FromMilliseconds(150);
    private static readonly TimeSpan Released = TimeSpan.FromSeconds(10);
    private static readonly TimeSpan Forever = TimeSpan.FromHours(1);

    [Fact]
    public void APrewarmCompileStartsAtOnceWhileTheGameCompilesNothing()
    {
        var scheduler = new MetalCompileScheduler(2, TimeSpan.Zero, Forever);
        using var prewarm = scheduler.EnterPrewarm(static () => false);
        Assert.True(prewarm.Entered);
        Assert.Equal(1, scheduler.PrewarmInFlight);
    }

    [Fact]
    public void APendingRuntimeCompileHoldsEveryPrewarmCompileBackUntilItIsDone()
    {
        var scheduler = new MetalCompileScheduler(4, TimeSpan.Zero, Forever);
        scheduler.RuntimeStarted();
        var prewarm = Enter(scheduler);
        Assert.False(prewarm.Wait(Held));

        scheduler.RuntimeDone();
        Assert.True(prewarm.Wait(Released));
        Assert.True(scheduler.TakeCounters().HeldMilliseconds > 0);
    }

    [Fact]
    public void ARuntimeScopeCountsAsPendingUntilItIsDisposed()
    {
        var scheduler = new MetalCompileScheduler(4, TimeSpan.Zero, Forever);
        var runtime = scheduler.Runtime();
        Assert.Equal(1, scheduler.RuntimePending);
        var prewarm = Enter(scheduler);
        Assert.False(prewarm.Wait(Held));

        runtime.Dispose();
        Assert.Equal(0, scheduler.RuntimePending);
        Assert.True(prewarm.Wait(Released));
    }

    // The worker's misses come in bursts: the gap between two of them must not let the
    // prewarm refill the driver.
    [Fact]
    public void APrewarmCompileWaitsForTheQuietPeriodAfterTheLastRuntimeCompile()
    {
        var quiet = TimeSpan.FromMilliseconds(300);
        var scheduler = new MetalCompileScheduler(4, quiet, Forever);
        scheduler.RuntimeStarted();
        // Taken before the quiet period starts, so a descheduled thread cannot shorten the elapsed time.
        var started = Stopwatch.GetTimestamp();
        scheduler.RuntimeDone();
        using var prewarm = scheduler.EnterPrewarm(static () => false);
        Assert.True(prewarm.Entered);
        Assert.True(Stopwatch.GetElapsedTime(started) >= quiet - TimeSpan.FromMilliseconds(20));
    }

    [Fact]
    public void ThePrewarmKeepsNoMoreThanItsShareInTheDriver()
    {
        var scheduler = new MetalCompileScheduler(2, TimeSpan.Zero, Forever);
        var first = scheduler.EnterPrewarm(static () => false);
        using var second = scheduler.EnterPrewarm(static () => false);
        var third = Enter(scheduler);
        Assert.False(third.Wait(Held));
        Assert.Equal(2, scheduler.PrewarmInFlight);

        first.Dispose();
        Assert.True(third.Wait(Released));
        third.Result.Dispose();
    }

    // The game compiling without pause must not stop the prewarm for good: past the
    // starvation limit one prewarm compile at a time goes through.
    [Fact]
    public void AStarvedPrewarmTricklesOneCompileAtATime()
    {
        var limit = TimeSpan.FromMilliseconds(200);
        var scheduler = new MetalCompileScheduler(4, TimeSpan.Zero, limit);
        scheduler.RuntimeStarted();
        var waiters = Enumerable.Range(0, 3).Select(_ => Enter(scheduler)).ToArray();

        var released = new HashSet<Task<MetalCompileScheduler.PrewarmScope>>();
        for (var through = 1; through <= waiters.Length; through++)
        {
            Assert.True(SpinWait.SpinUntil(() => waiters.Count(waiter => waiter.IsCompleted) == through, Released));
            // One, and only one, while it is in the driver.
            Thread.Sleep(Held);
            Assert.Equal(through, waiters.Count(waiter => waiter.IsCompleted));
            Assert.Equal(1, scheduler.PrewarmInFlight);
            var entered = waiters.Single(waiter => waiter.IsCompleted && !released.Contains(waiter));
            Assert.True(entered.Result.Entered);
            entered.Result.Dispose();
            released.Add(entered);
        }

        Assert.Equal(0, scheduler.PrewarmInFlight);
        Assert.Equal(3, scheduler.TakeCounters().Trickled);
        scheduler.RuntimeDone();
    }

    [Fact]
    public void AStoppingPrewarmLeavesWithoutASlot()
    {
        var scheduler = new MetalCompileScheduler(4, TimeSpan.Zero, Forever);
        scheduler.RuntimeStarted();
        var stopping = false;
        var prewarm = Task.Run(() => scheduler.EnterPrewarm(() => Volatile.Read(ref stopping)));
        Assert.False(prewarm.Wait(Held));

        Volatile.Write(ref stopping, true);
        Assert.True(prewarm.Wait(Released));
        Assert.False(prewarm.Result.Entered);
        Assert.Equal(0, scheduler.PrewarmInFlight);
    }

    // A compile the worker needs before a pool thread took it runs on the worker at once,
    // ahead of the queue, and only once.
    [Fact]
    public void AQueuedLibraryCompileRunsOnTheCallerAheadOfTheQueue()
    {
        var pool = new MetalLibraryCompilePool(1);
        using var firstStarted = new ManualResetEventSlim(false);
        using var blocker = new ManualResetEventSlim(false);
        var order = new List<string>();
        var first = pool.Run(() =>
        {
            firstStarted.Set();
            blocker.Wait();
            lock (order)
            {
                order.Add("first");
            }
        });
        var runs = 0;
        var caller = Environment.CurrentManagedThreadId;
        var ranOn = 0;
        var needed = pool.Run(() =>
        {
            Interlocked.Increment(ref runs);
            ranOn = Environment.CurrentManagedThreadId;
            lock (order)
            {
                order.Add("needed");
            }
        });

        Assert.True(firstStarted.Wait(Released));
        Assert.False(pool.TryRunHere(first));
        Assert.True(pool.TryRunHere(needed));
        Assert.True(needed.Task.IsCompletedSuccessfully);
        Assert.Equal(caller, ranOn);

        blocker.Set();
        Assert.True(first.Task.Wait(Released));
        // The pool thread reaches the job the caller took and skips it.
        var after = pool.Run(static () => { });
        Assert.True(after.Task.Wait(Released));
        Assert.Equal(1, Volatile.Read(ref runs));
        Assert.Equal(["needed", "first"], order);
    }

    [Fact]
    public void ALibraryCompileAPoolThreadHasIsWaitedForNotRunAgain()
    {
        var pool = new MetalLibraryCompilePool(1);
        using var started = new ManualResetEventSlim(false);
        using var release = new ManualResetEventSlim(false);
        var runs = 0;
        var job = pool.Run(() =>
        {
            Interlocked.Increment(ref runs);
            started.Set();
            release.Wait();
        });

        Assert.True(started.Wait(Released));
        Assert.False(pool.TryRunHere(job));
        release.Set();
        Assert.True(job.Task.Wait(Released));
        Assert.Equal(1, Volatile.Read(ref runs));
    }

    [Fact]
    public void AFailedCompileTheCallerRanFailsItsTask()
    {
        var pool = new MetalLibraryCompilePool(1);
        using var busyStarted = new ManualResetEventSlim(false);
        using var blocker = new ManualResetEventSlim(false);
        _ = pool.Run(() =>
        {
            busyStarted.Set();
            blocker.Wait();
        });
        var failing = pool.Run(static () => throw new InvalidOperationException("compile failed"));
        Assert.True(busyStarted.Wait(Released));

        Assert.True(pool.TryRunHere(failing));
        var exception = Assert.Throws<InvalidOperationException>(() => failing.Task.GetAwaiter().GetResult());
        Assert.Equal("compile failed", exception.Message);
        blocker.Set();
    }

    // A runtime compile collects, on its own thread, what it waited for the prewarm's compile
    // of the same source; another thread's waits are not its own.
    [Fact]
    public void PrewarmWaitsAreCollectedOnTheThreadThatWaited()
    {
        _ = MetalCompileScheduler.TakeThreadPrewarmWait();
        MetalCompileScheduler.NotePrewarmWait(100);
        var other = new Thread(static () => MetalCompileScheduler.NotePrewarmWait(1000));
        other.Start();
        other.Join();
        MetalCompileScheduler.NotePrewarmWait(20);

        Assert.Equal(120, MetalCompileScheduler.TakeThreadPrewarmWait());
        Assert.Equal(0, MetalCompileScheduler.TakeThreadPrewarmWait());
    }

    private static Task<MetalCompileScheduler.PrewarmScope> Enter(MetalCompileScheduler scheduler) =>
        Task.Factory.StartNew(
            () => scheduler.EnterPrewarm(static () => false),
            CancellationToken.None,
            TaskCreationOptions.LongRunning,
            TaskScheduler.Default);

    [Fact]
    public void AnUnusedDeferredLibraryCanBeCancelledWithoutRunningIt()
    {
        var runs = 0;
        var job = new MetalLibraryCompilePool.Job(() => runs++);
        Assert.True(job.CancelIfQueued());
        Assert.False(job.TryRun());
        Assert.False(job.CancelIfQueued());
        Assert.True(job.Task.IsCanceled);
        Assert.Equal(0, runs);
    }

    [Fact]
    public void AClaimedDeferredLibraryCannotBeCancelled()
    {
        using var entered = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        var job = new MetalLibraryCompilePool.Job(() => { entered.Set(); release.Wait(); });
        var running = Task.Run(() => job.TryRun());
        try
        {
            Assert.True(entered.Wait(Released));
            Assert.False(job.CancelIfQueued());
        }
        finally { release.Set(); }
        Assert.True(running.Wait(Released));
        Assert.True(job.Task.IsCompletedSuccessfully);
    }
}
