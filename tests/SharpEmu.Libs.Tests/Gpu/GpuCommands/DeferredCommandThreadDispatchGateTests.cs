// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Agc;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Tests.Gpu.Rendering;
using SharpEmu.Libs.Tests.Gpu.Scheduling;
using Xunit;
using static SharpEmu.Libs.Tests.Gpu.Rendering.RenderExecutorFixtures;

namespace SharpEmu.Libs.Tests.Gpu.GpuCommands;

// A dispatch's prefetch job may resolve once the non-draw items before it that are not dispatches ran (DispatchGate),
// and the dispatch item waits for the jobs before its own, not for its own.
[Collection(SchedulingStateCollection.Name)]
public sealed class DeferredCommandThreadDispatchGateTests
{
    private sealed record Work : AgcExports.DeferredWork;

    private sealed class ConcurrentHost : IConcurrentResolveHost
    {
        public void BeginConcurrentResolve() { }
        public object EndConcurrentResolve() => this;
        public bool IsStillCurrent(object assumptions) => true;
    }

    [Fact]
    public void DispatchesLeaveTheDispatchGateWhereItIsAndOtherItemsMoveIt()
    {
        var host = new RecordingCommandStreamHost();
        var queue = new CommandStreamQueue(host);
        var producer = new DeferredCommandThread(queue, host.Memory, _ => { }, null, () => { }, false);
        using var completed = new ManualResetEventSlim();
        var gates = new List<(long Gate, long DispatchGate)>();
        Assert.True(queue.TryEnqueueControlBarrier(() =>
        {
            producer.Post(() => { });
            gates.Add((producer.Gate, producer.DispatchGate));
            producer.EnqueueDeferredDispatch(new Work(), hasJob: false);
            producer.EnqueueDeferredDispatch(new Work(), hasJob: false);
            gates.Add((producer.Gate, producer.DispatchGate));
            producer.PostOrdering(() => { });
            gates.Add((producer.Gate, producer.DispatchGate));
            producer.Post(() => { });
            gates.Add((producer.Gate, producer.DispatchGate));
            completed.Set();
        }, () => { }));
        producer.Start();
        try
        {
            Assert.True(SpinWait.SpinUntil(() =>
            {
                producer.RunBatch(long.MaxValue, 64, () => { });
                return completed.IsSet && !producer.HasWork;
            }, TimeSpan.FromSeconds(10)));
            producer.ThrowIfFailed();
            Assert.Equal(new[] { (1L, 1L), (3L, 1L), (3L, 1L), (4L, 4L) }, gates);
        }
        finally
        {
            producer.Stop(() => producer.RunBatch(long.MaxValue, 64, () => { }));
        }
    }

    // Its own job's gate never opens here: a dispatch item that waited for it would never run.
    [Fact]
    public void ADispatchItemDoesNotWaitForItsOwnJob()
    {
        using var fatal = new FatalScope();
        var renderHost = new RecordingRenderHost();
        using var prefetch = new ParallelProgramPrefetch(new RenderExecutor(renderHost, new FakePipelineProvider()), renderHost,
            new ConcurrentHost(), allowCallerResolution: false);
        var host = new RecordingCommandStreamHost();
        var queue = new CommandStreamQueue(host);
        var ran = 0;
        var producer = new DeferredCommandThread(queue, host.Memory, _ => Interlocked.Increment(ref ran), prefetch, () => { }, false);
        Assert.True(queue.TryEnqueueControlBarrier(() =>
        {
            prefetch.SubmitCompute(Banks().Snapshot(), long.MaxValue, 0x41, 1, 1, 1, false);
            producer.EnqueueDeferredDispatch(new Work(), hasJob: true);
        }, () => { }));
        producer.Start();
        try
        {
            Assert.True(SpinWait.SpinUntil(() =>
            {
                producer.RunBatch(long.MaxValue, 64, () => { });
                return Volatile.Read(ref ran) == 1;
            }, TimeSpan.FromSeconds(10)));
            producer.ThrowIfFailed();
        }
        finally
        {
            producer.Stop(() => producer.RunBatch(long.MaxValue, 64, () => { }));
        }
    }
}
