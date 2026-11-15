// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.Libs.Tests.Gpu.GpuCommands;
using SharpEmu.Libs.Tests.Gpu.Scheduling;
using Xunit;
using static SharpEmu.Libs.Tests.Gpu.Rendering.RenderExecutorFixtures;

namespace SharpEmu.Libs.Tests.Gpu.Rendering;

// A draw's job may resolve before the dispatches just before the draw ran (SubmitDrawAhead). Its take accepts the
// resolution only when none of the dispatches that ran after it began did more than encode GPU work and the host's
// answers still hold across them (IsStillCurrentAcrossDispatches); otherwise the draw resolves its programs itself.
// The host's collector call after a submission is looked ahead of the same way (NoteCollect). Every other gated item
// still holds the draw's resolution.
[Collection(SchedulingStateCollection.Name)]
public sealed class ParallelProgramPrefetchDrawLookaheadTests : IDisposable
{
    private readonly RecordingRenderHost _host = new();
    private readonly FakePipelineProvider _pipelines = new();
    private readonly FatalScope _fatal = new();
    private readonly ConcurrentHost _concurrent = new();
    private readonly ParallelProgramPrefetch _prefetch;

    public ParallelProgramPrefetchDrawLookaheadTests()
    {
        var executor = new RenderExecutor(_host, _pipelines);
        _prefetch = new ParallelProgramPrefetch(executor, _host, _concurrent, allowCallerResolution: false)
        {
            LookaheadMode = ParallelProgramPrefetch.DrawLookaheadMode.On,
        };
        executor.Prefetch = _prefetch;
    }

    public void Dispose()
    {
        _prefetch.Dispose();
        _fatal.Dispose();
    }

    private sealed class ConcurrentHost : IConcurrentResolveHost
    {
        internal bool Current = true;
        internal bool CurrentAcrossDispatches = true;
        internal int Checks, CrossChecks;
        public bool PrefetchedDrawsCrossDispatches => true;
        public void BeginConcurrentResolve() { }
        public object EndConcurrentResolve() => this;
        public bool IsStillCurrent(object assumptions)
        {
            Interlocked.Increment(ref Checks);
            return Current;
        }

        public bool IsStillCurrentAcrossDispatches(object assumptions)
        {
            Interlocked.Increment(ref CrossChecks);
            return CurrentAcrossDispatches && IsStillCurrent(assumptions);
        }
    }

    // A draw job resolved once the gated items run so far (at least `gate` of them) ran, with `fullGate` items ahead of its draw.
    private ParallelProgramPrefetch.Job Resolved(RegisterBanks banks, long gate, long fullGate)
    {
        var job = _prefetch.SubmitDrawAhead(banks, gate, fullGate);
        _prefetch.WaitIdle();
        Assert.Equal(1, job.State);
        return job;
    }

    private bool Take(RegisterBanks banks, ParallelProgramPrefetch.Job job)
    {
        _prefetch.Current = job;
        var inputs = RenderExecutor.ProgramInputsOf(banks, _host.FormatSupport, _host.Fatal);
        return _prefetch.TryTakeGraphics(banks, in inputs, out _);
    }

    private (long Jobs, long Taken, long[] Refused, long Verified) Counters() => _prefetch.TakeDrawLookaheadCounters();

    [Fact]
    public void ADrawResolvedAheadOfADispatchThatChangedNothingIsTaken()
    {
        var banks = Banks().Snapshot();
        var job = Resolved(banks, gate: 0, fullGate: 1);
        Assert.Equal(0, job.ExecutedAtStart);
        // The dispatch runs, encoding GPU work only.
        _prefetch.AdvanceGate(1);
        Assert.True(Take(banks, job));
        Assert.Equal(1, _concurrent.CrossChecks);
        var counters = Counters();
        Assert.Equal((1L, 1L), (counters.Jobs, counters.Taken));
        Assert.All(counters.Refused, count => Assert.Equal(0, count));
    }

    // Resolved once the dispatches had run, the job looked ahead of nothing: its take is a draw's as before.
    [Fact]
    public void AJobResolvedAfterTheDispatchesRanIsCheckedAsBefore()
    {
        var banks = Banks().Snapshot();
        _prefetch.AdvanceGate(2);
        var job = Resolved(banks, gate: 1, fullGate: 2);
        _concurrent.CurrentAcrossDispatches = false;
        Assert.True(Take(banks, job));
        Assert.Equal((0, 1), (_concurrent.CrossChecks, _concurrent.Checks));
        Assert.Equal(0, Counters().Jobs);
    }

    [Fact]
    public void ADrawWhoseAnswersChangedAcrossTheDispatchesIsRefused()
    {
        var banks = Banks().Snapshot();
        var job = Resolved(banks, gate: 0, fullGate: 1);
        _prefetch.AdvanceGate(1);
        _concurrent.CurrentAcrossDispatches = false;
        Assert.False(Take(banks, job));
        Assert.Equal(1, Counters().Refused[(int)ParallelProgramPrefetch.DrawLookaheadRefusal.Ownership]);
        Assert.Equal(1, _prefetch.TakeCounters().Refused);
    }

    [Fact]
    public void ADispatchThatDidMoreThanEncodeRefusesTheDrawsResolvedAheadOfItOnly()
    {
        var banks = Banks().Snapshot();
        // Three dispatches before the draws. `ahead` resolved before any ran; the first is consumed by a host fill.
        var ahead = Resolved(banks, gate: 0, fullGate: 3);
        _prefetch.NoteDispatchEffect();
        _prefetch.AdvanceGate(1);
        // `behind` resolved after the first ran, ahead of the other two, which only encode.
        var behind = Resolved(banks, gate: 1, fullGate: 3);
        Assert.Equal(1, behind.ExecutedAtStart);
        _prefetch.AdvanceGate(3);
        Assert.False(Take(banks, ahead));
        Assert.True(Take(banks, behind));
        var counters = Counters();
        Assert.Equal((2L, 1L), (counters.Jobs, counters.Taken));
        Assert.Equal(1, counters.Refused[(int)ParallelProgramPrefetch.DrawLookaheadRefusal.Effects]);
    }

    [Fact]
    public void AnEffectOfTheLastDispatchAheadRefusesEveryDrawResolvedBeforeIt()
    {
        var banks = Banks().Snapshot();
        var first = Resolved(banks, gate: 0, fullGate: 2);
        _prefetch.AdvanceGate(1);
        var second = Resolved(banks, gate: 1, fullGate: 2);
        // The second dispatch (gated item 2) copies an image.
        _prefetch.NoteDispatchEffect();
        _prefetch.AdvanceGate(2);
        Assert.False(Take(banks, first));
        Assert.False(Take(banks, second));
        Assert.Equal(2, Counters().Refused[(int)ParallelProgramPrefetch.DrawLookaheadRefusal.Effects]);
    }

    [Fact]
    public void OtherInputsAreRefused()
    {
        var banks = Banks().Snapshot();
        var job = Resolved(banks, gate: 0, fullGate: 1);
        _prefetch.AdvanceGate(1);
        _prefetch.Current = job;
        var inputs = RenderExecutor.ProgramInputsOf(banks, _host.FormatSupport, _host.Fatal, indexed: true);
        Assert.False(_prefetch.TryTakeGraphics(banks, in inputs, out _));
        Assert.Equal(1, Counters().Refused[(int)ParallelProgramPrefetch.DrawLookaheadRefusal.Inputs]);
    }

    // =verify: the worker resolves an accepted look-ahead again, and a resolution that does not match is fatal.
    [Fact]
    public void VerifyResolvesAnAcceptedLookaheadAgainAndFailsOnADifference()
    {
        _prefetch.LookaheadMode = ParallelProgramPrefetch.DrawLookaheadMode.Verify;
        var banks = Banks().Snapshot();
        var job = Resolved(banks, gate: 0, fullGate: 1);
        _prefetch.AdvanceGate(1);
        Assert.True(Take(banks, job));
        Assert.Equal(1, Counters().Verified);

        // The host wrongly stands for a resolution the dispatch changed: the provider now resolves other programs.
        _prefetch.NoteGateItem(2, ParallelProgramPrefetch.GateItemKind.Dispatch);
        job = Resolved(banks, gate: 1, fullGate: 2);
        _pipelines.Graphics = Programs(vertexBuffers: [new VertexInputBuffer(VertexBase, 16, 4)]);
        _prefetch.AdvanceGate(2);
        var fatal = Assert.Throws<RenderExecutorFatalException>(() => Take(banks, job));
        Assert.Equal(1, Counters().Verified);
        // The fatal names every differing group, both resolutions, whether the worker resolves the same again, and the
        // items the resolution looked over, so a verify run's failure can be told apart without another run.
        Assert.Contains("differences=[vertex.buffers,", fatal.Message);
        Assert.Contains("ahead=[available=True", fatal.Message);
        Assert.Contains("worker=[available=True", fatal.Message);
        Assert.Contains("worker_again=same", fatal.Message);
        Assert.Contains("looked_over=[2:dispatch=0x0000000000000000]", fatal.Message);
        Assert.Contains("last_dispatch_effect=-1", fatal.Message);
    }

    // A gated item other than a dispatch (here an interrupt) still holds the resolution of the draw after it, while the
    // dispatch between them does not.
    [Fact]
    public void AnInterruptBeforeTheDispatchStillHoldsTheDraw()
    {
        var host = new RecordingCommandStreamHost();
        var queue = new CommandStreamQueue(host);
        var producer = new DeferredCommandThread(queue, host.Memory, _ => { }, _prefetch, () => { }, false);
        ParallelProgramPrefetch.Job? job = null;
        var stateWhenTheInterruptRan = -1;
        var done = 0;
        Assert.True(queue.TryEnqueueControlBarrier(() =>
        {
            producer.Post(() => stateWhenTheInterruptRan = job!.State, ParallelProgramPrefetch.GateItemKind.Event);
            producer.EnqueueDeferredDispatch(new Work(), hasJob: false);
            job = _prefetch.SubmitDrawAhead(Banks().Snapshot(), producer.DispatchGate, producer.Gate);
            producer.EnqueueDeferred(new Work(), isDraw: true);
            producer.Post(() => Volatile.Write(ref done, 1));
        }, () => { }));
        producer.Start();
        try
        {
            Assert.True(SpinWait.SpinUntil(() =>
            {
                producer.RunBatch(long.MaxValue, 64, () => { });
                return Volatile.Read(ref done) == 1;
            }, TimeSpan.FromSeconds(10)));
            producer.ThrowIfFailed();
        }
        finally
        {
            producer.Stop(() => producer.RunBatch(long.MaxValue, 64, () => { }));
        }

        _prefetch.WaitIdle();
        Assert.Equal((1L, 2L), (job!.Gate, job.FullGate));
        Assert.Equal(0, stateWhenTheInterruptRan);
        Assert.True(job.ExecutedAtStart >= 1);
    }

    // A draw job submitted after a collector call queued as gated item `collectGate`, resolved once `executed` items ran.
    private ParallelProgramPrefetch.Job ResolvedAfterCollect(RegisterBanks banks, long collectGate, long gate, long fullGate)
    {
        _prefetch.NoteGateItem(collectGate, ParallelProgramPrefetch.GateItemKind.Collect);
        var job = Resolved(banks, gate, fullGate);
        Assert.Equal(collectGate, job.LastCollectGate);
        return job;
    }

    // The collector call after a submission released nothing, disposed nothing and armed nothing: a draw resolved
    // before it ran is taken, and counted as a look-ahead over it.
    [Fact]
    public void ADrawResolvedAheadOfACollectorCallThatDidNothingIsTaken()
    {
        var banks = Banks().Snapshot();
        var job = ResolvedAfterCollect(banks, collectGate: 1, gate: 0, fullGate: 1);
        Assert.Equal(0, job.ExecutedAtStart);
        // The worker runs the call (gated item 1).
        _prefetch.NoteCollect(effect: false);
        _prefetch.AdvanceGate(1);
        Assert.True(Take(banks, job));
        Assert.Equal(1, _concurrent.CrossChecks);
        var counters = Counters();
        Assert.Equal((1L, 1L), (counters.Jobs, counters.Taken));
        Assert.All(counters.Refused, count => Assert.Equal(0, count));
        Assert.Equal((1L, 1L, 0L, 1L, 0L), _prefetch.TakeCollectLookaheadCounters());
    }

    // A collector call that noted an effect (an image released, single-copy work) refuses the draws resolved before it
    // ran, and only those: one resolved after it ran, still ahead of a dispatch, is taken and not counted over the call.
    [Fact]
    public void ACollectorCallThatNotedAnEffectRefusesTheDrawsResolvedAheadOfItOnly()
    {
        var banks = Banks().Snapshot();
        // Gated item 1 is the collector call, item 2 a dispatch.
        var ahead = ResolvedAfterCollect(banks, collectGate: 1, gate: 0, fullGate: 2);
        _prefetch.NoteCollect(effect: true);
        _prefetch.AdvanceGate(1);
        var behind = Resolved(banks, gate: 1, fullGate: 2);
        Assert.Equal(1, behind.ExecutedAtStart);
        _prefetch.AdvanceGate(2);
        Assert.False(Take(banks, ahead));
        Assert.True(Take(banks, behind));
        var counters = Counters();
        Assert.Equal((2L, 1L), (counters.Jobs, counters.Taken));
        Assert.Equal(1, counters.Refused[(int)ParallelProgramPrefetch.DrawLookaheadRefusal.CollectEffects]);
        Assert.Equal(0, counters.Refused[(int)ParallelProgramPrefetch.DrawLookaheadRefusal.Effects]);
        Assert.Equal((1L, 0L, 1L, 1L, 1L), _prefetch.TakeCollectLookaheadCounters());
    }

    // =verify covers a look-ahead over a collector call: the worker resolves it again, and a difference is fatal.
    [Fact]
    public void VerifyResolvesALookaheadOverACollectorCallAgainAndFailsOnADifference()
    {
        _prefetch.LookaheadMode = ParallelProgramPrefetch.DrawLookaheadMode.Verify;
        var banks = Banks().Snapshot();
        var job = ResolvedAfterCollect(banks, collectGate: 1, gate: 0, fullGate: 1);
        _prefetch.NoteCollect(effect: false);
        _prefetch.AdvanceGate(1);
        Assert.True(Take(banks, job));
        Assert.Equal(1, Counters().Verified);
        Assert.Equal(1, _prefetch.TakeCollectLookaheadCounters().Taken);

        // The collector wrongly reported nothing for a change the resolution read: the provider now resolves other programs.
        job = ResolvedAfterCollect(banks, collectGate: 2, gate: 1, fullGate: 2);
        _pipelines.Graphics = Programs(vertexBuffers: [new VertexInputBuffer(VertexBase, 16, 4)]);
        _prefetch.NoteCollect(effect: false);
        _prefetch.AdvanceGate(2);
        Assert.Throws<RenderExecutorFatalException>(() => Take(banks, job));
    }

    // On the command thread: with look-ahead on, a collector call leaves the dispatch gate where it is, so the draw after
    // it resolves before the call runs (the call waits here until it did); with SHARPEMU_PREFETCH_DRAW_LOOKAHEAD=0 the
    // call moves the gate and still holds the draw, as with SHARPEMU_PREFETCH_GC_LOOKAHEAD=0 alone.
    [Theory]
    [InlineData(ParallelProgramPrefetch.DrawLookaheadMode.On, true)]
    [InlineData(ParallelProgramPrefetch.DrawLookaheadMode.Verify, true)]
    [InlineData(ParallelProgramPrefetch.DrawLookaheadMode.Off, true)]
    [InlineData(ParallelProgramPrefetch.DrawLookaheadMode.On, false)]
    public void ACollectorCallHoldsTheDrawAfterItOnlyWithLookaheadOff(ParallelProgramPrefetch.DrawLookaheadMode mode, bool collectLookahead)
    {
        _prefetch.LookaheadMode = mode;
        _prefetch.CollectLookahead = collectLookahead;
        var lookahead = mode != ParallelProgramPrefetch.DrawLookaheadMode.Off && collectLookahead;
        Assert.Equal(lookahead, _prefetch.LooksAheadOfCollects);
        var host = new RecordingCommandStreamHost();
        var queue = new CommandStreamQueue(host);
        var producer = new DeferredCommandThread(queue, host.Memory, _ => { }, _prefetch, () => { }, false);
        ParallelProgramPrefetch.Job? job = null;
        var resolvedBeforeTheCall = false;
        var done = 0;
        Assert.True(queue.TryEnqueueControlBarrier(() =>
        {
            producer.Post(() =>
            {
                // Held, the job cannot resolve before this call returns: wait briefly; ahead, it resolves now.
                // The command thread may not have submitted the job yet when the worker gets here.
                resolvedBeforeTheCall = SpinWait.SpinUntil(() => Volatile.Read(ref job) is { State: not 0 },
                    TimeSpan.FromMilliseconds(lookahead ? 10_000 : 200));
                _prefetch.NoteCollect(effect: false);
            }, ParallelProgramPrefetch.GateItemKind.Collect);
            job = _prefetch.SubmitDrawAhead(Banks().Snapshot(), producer.DispatchGate, producer.Gate);
            producer.EnqueueDeferred(new Work(), isDraw: true);
            producer.Post(() => Volatile.Write(ref done, 1));
        }, () => { }));
        producer.Start();
        try
        {
            Assert.True(SpinWait.SpinUntil(() =>
            {
                producer.RunBatch(long.MaxValue, 64, () => { });
                return Volatile.Read(ref done) == 1;
            }, TimeSpan.FromSeconds(20)));
            producer.ThrowIfFailed();
        }
        finally
        {
            producer.Stop(() => producer.RunBatch(long.MaxValue, 64, () => { }));
        }

        _prefetch.WaitIdle();
        Assert.Equal((lookahead ? 0L : 1L, 1L, 1L), (job!.Gate, job.FullGate, job.LastCollectGate));
        Assert.Equal(lookahead, resolvedBeforeTheCall);
        Assert.True(lookahead ? job.ExecutedAtStart == 0 : job.ExecutedAtStart >= 1);
    }

    private sealed record Work : SharpEmu.Libs.Agc.AgcExports.DeferredWork;
}
