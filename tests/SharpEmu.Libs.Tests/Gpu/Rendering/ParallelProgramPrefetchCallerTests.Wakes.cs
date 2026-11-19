// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Reflection;
using SharpEmu.Libs.Gpu.Rendering;
using Xunit;
using static SharpEmu.Libs.Tests.Gpu.Rendering.RenderExecutorFixtures;

namespace SharpEmu.Libs.Tests.Gpu.Rendering;

// FinishJob wakes only registered waiters and the worker idle in its queue wait: none of them may be left asleep.
public sealed partial class ParallelProgramPrefetchCallerTests
{
    private static int Waiters(Fixture f) =>
        (int)typeof(ParallelProgramPrefetch).GetField("_waiters", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(f.Prefetch)!;

    [Fact]
    public void TheWorkerIdleBehindACallerResolutionResolvesTheNextJobOnceItFinishes()
    {
        for (var round = 0; round < 50; round++)
        {
            using var f = new Fixture();
            var firstBanks = Banks(); var secondBanks = Banks();
            ParallelProgramPrefetch.Job second;
            lock (f.QueueLock)
            {
                var first = f.Prefetch.Submit(firstBanks);
                second = f.Prefetch.Submit(secondBanks);
                Assert.True(f.Take(firstBanks, first));
            }

            // Nobody waits on the second job: only the caller's FinishJob can wake a worker idle behind it.
            Assert.True(SpinWait.SpinUntil(() => second.State != 0, TimeSpan.FromSeconds(10)), "The worker was left asleep.");
            f.Idle();
            lock (f.QueueLock)
            {
                Assert.Equal(0, Waiters(f));
            }
        }
    }

    [Fact]
    public void EveryWaiterWakesWhenTheWorkerFinishesAndNoneIsLeftCounted()
    {
        using var f = new Fixture();
        var gate = new ManualResetEventSlim();
        var entered = new ManualResetEventSlim();
        f.Pipelines.BeforeGraphicsResolve = _ =>
        {
            entered.Set();
            gate.Wait(TimeSpan.FromSeconds(10));
        };
        var banks = Banks();
        var job = f.Prefetch.Submit(banks);
        Assert.True(entered.Wait(TimeSpan.FromSeconds(10)));
        var waits = new[] { Task.Run(f.Prefetch.WaitIdle), Task.Run(() => f.Prefetch.WaitFinished(1)), Task.Run(f.Prefetch.Pause) };
        Assert.True(SpinWait.SpinUntil(() => { lock (f.QueueLock) return Waiters(f) == 3; }, TimeSpan.FromSeconds(10)));
        gate.Set();
        Assert.True(Task.WaitAll(waits, TimeSpan.FromSeconds(10)));
        f.Prefetch.Resume();
        Assert.Equal(1, job.State);
        lock (f.QueueLock)
        {
            Assert.Equal(0, Waiters(f));
        }
    }
}
