// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Reflection;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.Libs.Tests.Gpu.Scheduling;
using Xunit;
using static SharpEmu.Libs.Tests.Gpu.Rendering.RenderExecutorFixtures;

namespace SharpEmu.Libs.Tests.Gpu.Rendering;

[Collection(SchedulingStateCollection.Name)]
public sealed partial class ParallelProgramPrefetchCallerTests
{
    private sealed class ConcurrentHost : IConcurrentResolveHost
    {
        internal bool Current = true;
        internal int Active, MaximumActive, Begins, Ends, Releases;
        public void BeginConcurrentResolve()
        {
            var active = Interlocked.Increment(ref Active);
            int old;
            do { old = Volatile.Read(ref MaximumActive); }
            while (active > old && Interlocked.CompareExchange(ref MaximumActive, active, old) != old);
            Interlocked.Increment(ref Begins);
        }
        public object EndConcurrentResolve()
        {
            Interlocked.Increment(ref Ends);
            Interlocked.Decrement(ref Active);
            return this;
        }
        public bool IsStillCurrent(object assumptions) => ReferenceEquals(this, assumptions) && Current;
        public void ReleaseAssumptions(object assumptions)
        {
            Assert.Same(this, assumptions);
            Interlocked.Increment(ref Releases);
        }
    }

    private sealed class Fixture : IDisposable
    {
        private readonly FatalScope _fatal = new();
        internal readonly RecordingRenderHost Host = new();
        internal readonly FakePipelineProvider Pipelines = new();
        internal readonly ConcurrentHost Concurrent = new();
        internal readonly ParallelProgramPrefetch Prefetch;
        // Holding the queue monitor on this test thread makes the unclaimed-head
        // case deterministic, without sleeps or a production scheduler test hook.
        internal readonly object QueueLock;
        internal Fixture(bool enabled = true)
        {
            Prefetch = new(new RenderExecutor(Host, Pipelines), Host, Concurrent, enabled);
            QueueLock = typeof(ParallelProgramPrefetch).GetField("_lock", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(Prefetch)!;
        }
        internal bool Take(RegisterBanks banks, ParallelProgramPrefetch.Job job, bool indexed = false)
        {
            Prefetch.Current = job;
            var inputs = RenderExecutor.ProgramInputsOf(banks, Host.FormatSupport, Host.Fatal, indexed);
            return Prefetch.TryTakeGraphics(banks, in inputs, out _);
        }
        internal void Idle() => Assert.True(Task.Run(Prefetch.WaitIdle).Wait(TimeSpan.FromSeconds(10)), "Resolver did not become idle.");
        public void Dispose() { Prefetch.Dispose(); _fatal.Dispose(); }
    }

    [Fact]
    public void CallerTakesOnlyTheHeadAndCompletesItsPrefixExactlyOnce()
    {
        using var f = new Fixture();
        var first = Banks(); var second = Banks();
        lock (f.QueueLock)
        {
            var a = f.Prefetch.Submit(first); var b = f.Prefetch.Submit(second);
            Assert.False(f.Prefetch.TryResolveOnCaller(b));
            Assert.Equal(0, f.Concurrent.Begins);
            Assert.True(f.Take(first, a));
            Assert.False(f.Prefetch.TryResolveOnCaller(a));
            Assert.Equal(0, b.State);
            f.Prefetch.WaitFinished(1);
            Assert.True(f.Take(second, b));
            f.Prefetch.WaitFinished(2);
        }
        f.Idle();
        Assert.Equal((2, 2, 2, 0, 1), (f.Concurrent.Begins, f.Concurrent.Ends, f.Concurrent.Releases, f.Concurrent.Active, f.Concurrent.MaximumActive));
        Assert.Equal(2, f.Prefetch.TakeCallerResolveCounters().Count);
        Assert.Equal(2, f.Prefetch.TakeCounters().Taken);
    }

    [Fact]
    public void PauseAndDependencyGateBothPreventCallerResolution()
    {
        using var f = new Fixture();
        lock (f.QueueLock)
        {
            f.Prefetch.Pause();
            var banks = Banks(); var job = f.Prefetch.Submit(banks, gate: 2);
            Assert.False(f.Prefetch.TryResolveOnCaller(job));
            f.Prefetch.Resume();
            Assert.False(f.Prefetch.TryResolveOnCaller(job));
            f.Prefetch.AdvanceGate(1);
            Assert.False(f.Prefetch.TryResolveOnCaller(job));
            f.Prefetch.AdvanceGate(2);
            f.Prefetch.Pause();
            Assert.False(f.Prefetch.TryResolveOnCaller(job));
            f.Prefetch.Resume();
            Assert.True(f.Take(banks, job));
        }
        f.Idle();
        Assert.Equal(1, f.Concurrent.Begins);
    }

    [Fact]
    public void TheCallerTakesAHeadTheWorkerClaimedOnceItsGateOpens()
    {
        using var f = new Fixture();
        var banks = Banks();
        var job = f.Prefetch.Submit(banks, gate: 1);
        var follower = Banks();
        var next = f.Prefetch.Submit(follower, gate: 1);
        var claim = typeof(ParallelProgramPrefetch).GetField("_workerClaimed", BindingFlags.Instance | BindingFlags.NonPublic)!;
        Assert.True(SpinWait.SpinUntil(() => { lock (f.QueueLock) return (bool)claim.GetValue(f.Prefetch)!; }, TimeSpan.FromSeconds(10)));
        lock (f.QueueLock)
        {
            // Closed gate: still nobody's to resolve.
            Assert.False(f.Prefetch.TryResolveOnCaller(job));
            f.Prefetch.AdvanceGate(1);
            // The worker sleeps on the gate behind this lock: the caller takes its head.
            Assert.True(f.Take(banks, job));
            Assert.Equal(1, f.Prefetch.TakeClaimSteals());
            Assert.Equal(0, next.State);
        }

        // The worker finds its head gone; the next job still resolves exactly once, on either thread.
        Assert.True(f.Take(follower, next));
        f.Idle();
        Assert.Equal((2, 2, 1), (f.Concurrent.Begins, f.Concurrent.Releases, f.Concurrent.MaximumActive));
        Assert.InRange(f.Prefetch.TakeCallerResolveCounters().Count, 1, 2);
        Assert.Equal(2, f.Prefetch.TakeCounters().Taken);
    }

    [Theory]
    [InlineData(false, false, true, true)]
    [InlineData(true, true, true, true)]
    [InlineData(true, false, true, false)]
    [InlineData(false, true, true, false)]
    [InlineData(false, false, false, false)]
    public void CallerResultsStillRequireMatchingInputsAndCurrentOwnership(bool indexed, bool requested, bool current, bool accepted)
    {
        using var f = new Fixture();
        f.Concurrent.Current = current;
        var banks = Banks();
        lock (f.QueueLock)
            Assert.Equal(accepted, f.Take(banks, f.Prefetch.Submit(banks, indexed: indexed), requested));
        f.Idle();
        Assert.Equal(1, f.Concurrent.Releases);
        Assert.Equal(1, f.Prefetch.TakeCallerResolveCounters().Count);
        Assert.Null(f.Prefetch.Current);
    }

    [Fact]
    public void PauseWaitsForAnActiveCallerAndHoldsTheFollowingJob()
    {
        using var f = new Fixture();
        Task? pause = null;
        var pausedField = typeof(ParallelProgramPrefetch).GetField("_paused", BindingFlags.Instance | BindingFlags.NonPublic)!;
        ParallelProgramPrefetch.Job? second = null;
        var invocations = 0;
        f.Pipelines.BeforeGraphicsResolve = _ =>
        {
            if (++invocations != 1) return;
            // Release the test's outer monitor while the caller is inside resolution.
            // Production releases its own claim lock before invoking this callback.
            Monitor.Exit(f.QueueLock);
            try
            {
                pause = Task.Run(f.Prefetch.Pause);
                Assert.True(SpinWait.SpinUntil(() => { lock (f.QueueLock) return (bool)pausedField.GetValue(f.Prefetch)!; }, TimeSpan.FromSeconds(10)));
                Assert.False(pause.IsCompleted);
                Assert.Equal(0, second!.State);
            }
            finally { Monitor.Enter(f.QueueLock); }
        };
        var firstBanks = Banks(); var secondBanks = Banks();
        lock (f.QueueLock)
        {
            var first = f.Prefetch.Submit(firstBanks);
            second = f.Prefetch.Submit(secondBanks);
            Assert.True(f.Take(firstBanks, first));
        }
        Assert.NotNull(pause);
        Assert.True(pause.Wait(TimeSpan.FromSeconds(10)));
        lock (f.QueueLock)
        {
            Assert.Equal(0, second.State);
            Assert.False(f.Prefetch.TryResolveOnCaller(second));
        }
        f.Prefetch.Resume();
        Assert.True(f.Take(secondBanks, second));
        f.Idle();
        Assert.Equal(1, f.Concurrent.MaximumActive);
        Assert.Equal(2, f.Concurrent.Releases);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SpeculativeFailureAbandonsOnceAndRestoresCallerFatalPolicy(bool alreadySpeculative)
    {
        using var f = new Fixture();
        f.Pipelines.BeforeGraphicsResolve = _ => throw GpuFatal.Fatal("synthetic speculative failure");
        var previous = GpuFatal.ThrowOnFatal;
        try
        {
            GpuFatal.ThrowOnFatal = alreadySpeculative;
            var banks = Banks();
            lock (f.QueueLock)
                Assert.False(f.Take(banks, f.Prefetch.Submit(banks)));
            Assert.Equal(alreadySpeculative, GpuFatal.ThrowOnFatal);
        }
        finally { GpuFatal.ThrowOnFatal = previous; }
        f.Idle();
        Assert.Equal(1, f.Prefetch.TakeCounters().Abandoned);
        Assert.Equal((1, 1, 1), (f.Concurrent.Begins, f.Concurrent.Ends, f.Concurrent.Releases));
    }

    [Fact]
    public void DisabledAndStoppedQueuesRefuseCallerClaims()
    {
        using var f = new Fixture(false);
        lock (f.QueueLock)
            Assert.False(f.Prefetch.TryResolveOnCaller(f.Prefetch.Submit(Banks())));
        f.Idle();
        using var stopped = new Fixture();
        lock (stopped.QueueLock)
        {
            var job = stopped.Prefetch.Submit(Banks());
            stopped.Prefetch.Dispose();
            Assert.False(stopped.Prefetch.TryResolveOnCaller(job));
        }
    }

    [Fact]
    public void ConcurrentClaimsRetainFifoAndNeverResolveTwoJobsAtOnce()
    {
        using var f = new Fixture();
        var resolved = new List<ulong>();
        f.Pipelines.BeforeGraphicsResolve = vertex => { resolved.Add(vertex.ExportAddress); Thread.SpinWait(128); };
        const int count = 2048;
        var banks = Enumerable.Range(0, count).Select(i => { var b = Banks(); b.Shader.Vertex.ExportAddress = (ulong)(i + 1) * 256; return b; }).ToArray();
        var jobs = new ParallelProgramPrefetch.Job[count];
        for (var start = 0; start < count; start += 8)
        {
            // Guarantee at least one caller claim, then let both threads race for the rest.
            lock (f.QueueLock)
            {
                for (var i = start; i < start + 8; i++) jobs[i] = f.Prefetch.Submit(banks[i]);
                if (start == 0) Assert.True(f.Take(banks[0], jobs[0]));
            }
            for (var i = start == 0 ? 1 : start; i < start + 8; i++) Assert.True(f.Take(banks[i], jobs[i]));
            f.Prefetch.WaitFinished(start + 8);
        }
        f.Idle();
        Assert.Equal(banks.Select(b => b.Shader.Vertex.ExportAddress), resolved);
        Assert.Equal((count, count, count, 0, 1), (f.Concurrent.Begins, f.Concurrent.Ends, f.Concurrent.Releases, f.Concurrent.Active, f.Concurrent.MaximumActive));
        Assert.True(f.Prefetch.TakeCallerResolveCounters().Count > 0);
    }

    [Theory]
    [InlineData(8, true)]
    [InlineData(9, false)]
    public void CallerClaimsAreBoundedToShortQueues(int count, bool claimed)
    {
        using var f = new Fixture();
        var banks = Banks();
        ParallelProgramPrefetch.Job[] jobs;
        lock (f.QueueLock)
        {
            jobs = Enumerable.Range(0, count).Select(_ => f.Prefetch.Submit(banks)).ToArray();
            Assert.Equal(claimed, f.Prefetch.TryResolveOnCaller(jobs[0]));
        }
        f.Idle();
        foreach (var job in jobs) Assert.True(f.Take(banks, job));
        Assert.Equal(count, f.Concurrent.Begins);
        Assert.Equal(count, f.Concurrent.Releases);
    }

    [Fact]
    public void TakingACompletedJobReportsNoWaitAndAnUnfinishedOneDoes()
    {
        using var f = new Fixture(false);
        var banks = Banks();
        var done = f.Prefetch.Submit(banks);
        f.Idle();
        Assert.True(f.Take(banks, done));
        Assert.Equal(0, f.Prefetch.TakeCounters().WaitMilliseconds);
        Task<bool> waiter;
        Monitor.Enter(f.QueueLock);
        try
        {
            // The worker cannot start the job while this thread holds the queue lock, so the take waits for it.
            var late = f.Prefetch.Submit(banks);
            waiter = Task.Run(() => f.Take(banks, late));
            Assert.False(waiter.Wait(50));
        }
        finally { Monitor.Exit(f.QueueLock); }

        Assert.True(waiter.Wait(TimeSpan.FromSeconds(10)) && waiter.Result);
        Assert.True(f.Prefetch.TakeCounters().WaitMilliseconds > 0);
    }
}
