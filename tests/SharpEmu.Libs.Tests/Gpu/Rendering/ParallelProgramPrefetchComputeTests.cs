// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.Libs.Tests.Gpu.Scheduling;
using Xunit;
using static SharpEmu.Libs.Tests.Gpu.Rendering.RenderExecutorFixtures;

namespace SharpEmu.Libs.Tests.Gpu.Rendering;

// A dispatch takes the program its prefetch job resolved only when the job stands for it: same banks and arguments,
// a result the provider stood for, nothing done by the dispatches it was resolved ahead of, and ownership answers
// that still hold across them. Otherwise it resolves the program itself, as before.
[Collection(SchedulingStateCollection.Name)]
public sealed class ParallelProgramPrefetchComputeTests : IDisposable
{
    private const uint Initiator = 0x41;

    private readonly RecordingRenderHost _host = new();
    private readonly FakePipelineProvider _pipelines = new();
    private readonly RenderExecutor _executor;
    private readonly FatalScope _fatal = new();
    private readonly ConcurrentHost _concurrent = new();
    private readonly ParallelProgramPrefetch _prefetch;

    public ParallelProgramPrefetchComputeTests()
    {
        _executor = new RenderExecutor(_host, _pipelines);
        _prefetch = new ParallelProgramPrefetch(_executor, _host, _concurrent, allowCallerResolution: false);
        _executor.Prefetch = _prefetch;
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
        // The answers move after the take's check: the verify's check (the second for the job) sees them moved.
        internal bool Raced;
        private int _crossChecksSinceRace;
        internal int Checks, CrossChecks;
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
            if (Raced && ++_crossChecksSinceRace % 2 == 0)
            {
                return false;
            }

            return CurrentAcrossDispatches && IsStillCurrent(assumptions);
        }
    }

    private ParallelProgramPrefetch.Job Resolved(RegisterBanks banks, long gate = 0, uint x = 4, uint y = 2, uint z = 1, bool indirect = false)
    {
        var job = _prefetch.SubmitCompute(banks, gate, Initiator, x, y, z, indirect);
        _prefetch.AdvanceGate(gate);
        _prefetch.WaitIdle();
        Assert.Equal(1, job.State);
        return job;
    }

    private bool Take(RegisterBanks banks, ParallelProgramPrefetch.Job job, uint x = 4, uint y = 2, uint z = 1, bool indirect = false)
    {
        _prefetch.Current = job;
        return _prefetch.TryTakeCompute(banks, Initiator, x, y, z, indirect, out _);
    }

    private long Refused(ParallelProgramPrefetch.ComputeRefusal reason) => _prefetch.TakeComputeCounters().Refused[(int)reason];

    [Fact]
    public void ADispatchTakesTheProgramItsJobResolvedFromTheSameBanksAndArguments()
    {
        var banks = Banks().Snapshot();
        var job = Resolved(banks);
        _prefetch.Current = job;
        Assert.True(_prefetch.TryTakeCompute(banks, Initiator, 4, 2, 1, false, out var program));
        Assert.Same(_pipelines.Compute, program);
        Assert.Equal(1, _concurrent.CrossChecks);
        var counters = _prefetch.TakeComputeCounters();
        Assert.Equal((1L, 1L), (counters.Submitted, counters.Taken));
        Assert.All(counters.Refused, count => Assert.Equal(0, count));
    }

    [Fact]
    public void OtherBanksOrArgumentsAreRefused()
    {
        var banks = Banks().Snapshot();
        Assert.False(Take(Banks().Snapshot(), Resolved(banks)));
        Assert.Equal(1, Refused(ParallelProgramPrefetch.ComputeRefusal.Inputs));
        Assert.False(Take(banks, Resolved(banks), x: 8));
        Assert.Equal(1, Refused(ParallelProgramPrefetch.ComputeRefusal.Inputs));
        Assert.False(Take(banks, Resolved(banks), indirect: true));
        Assert.Equal(1, Refused(ParallelProgramPrefetch.ComputeRefusal.Inputs));
    }

    [Fact]
    public void AResolutionWhoseOwnershipAnswersChangedAcrossTheDispatchesIsRefused()
    {
        var banks = Banks().Snapshot();
        _concurrent.CurrentAcrossDispatches = false;
        Assert.False(Take(banks, Resolved(banks)));
        Assert.Equal(1, Refused(ParallelProgramPrefetch.ComputeRefusal.Ownership));
        _concurrent.CurrentAcrossDispatches = true;
        _concurrent.Current = false;
        Assert.False(Take(banks, Resolved(banks)));
        Assert.Equal(1, Refused(ParallelProgramPrefetch.ComputeRefusal.Ownership));
    }

    [Theory]
    [InlineData(ComputePrefetchRefusal.Frontend)]
    [InlineData(ComputePrefetchRefusal.Prediction)]
    [InlineData(ComputePrefetchRefusal.Unavailable)]
    public void AResultTheProviderDidNotStandForIsRefusedWithItsReason(ComputePrefetchRefusal reason)
    {
        _pipelines.AheadRefusal = reason;
        var banks = Banks().Snapshot();
        Assert.False(Take(banks, Resolved(banks)));
        Assert.Equal(1, Refused(ParallelProgramPrefetch.ComputeRefusal.Provider));
        Assert.Equal(1, _prefetch.TakeProviderRefusals()[(int)reason]);
        Assert.Equal(0, _concurrent.CrossChecks);
    }

    [Fact]
    public void AnAbandonedResolutionIsRefused()
    {
        _pipelines.BeforeComputeAhead = () => throw GpuFatal.Fatal("synthetic speculative failure");
        var banks = Banks().Snapshot();
        var job = _prefetch.SubmitCompute(banks, 0, Initiator, 4, 2, 1, false);
        _prefetch.WaitIdle();
        Assert.Equal(2, job.State);
        Assert.False(Take(banks, job));
        Assert.Equal(1, Refused(ParallelProgramPrefetch.ComputeRefusal.Abandoned));
    }

    // A dispatch that did more than encode GPU work refuses the jobs resolved ahead of it (gate at or below the item
    // it ran as), not the jobs that waited for it.
    [Fact]
    public void ADispatchEffectRefusesOnlyTheJobsResolvedAheadOfIt()
    {
        var banks = Banks().Snapshot();
        var ahead = Resolved(banks, gate: 3);
        var behind = _prefetch.SubmitCompute(banks, 4, Initiator, 4, 2, 1, false);
        // Item 3 (the fourth gated non-draw) is a dispatch that was consumed.
        _prefetch.NoteDispatchEffect();
        _prefetch.AdvanceGate(4);
        _prefetch.WaitIdle();
        Assert.False(Take(banks, ahead));
        Assert.Equal(1, Refused(ParallelProgramPrefetch.ComputeRefusal.Effects));
        Assert.True(Take(banks, behind));
    }

    // A collector call that noted an effect refuses the dispatch jobs resolved ahead of it as a dispatch's effect does;
    // one that did nothing refuses none.
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ACollectorCallEffectRefusesOnlyTheJobsResolvedAheadOfIt(bool effect)
    {
        var banks = Banks().Snapshot();
        var ahead = Resolved(banks, gate: 3);
        var behind = _prefetch.SubmitCompute(banks, 4, Initiator, 4, 2, 1, false);
        // Item 3 (the fourth gated non-draw) is the collector call.
        _prefetch.NoteCollect(effect);
        _prefetch.AdvanceGate(4);
        _prefetch.WaitIdle();
        Assert.Equal(!effect, Take(banks, ahead));
        Assert.Equal(effect ? 1 : 0, Refused(ParallelProgramPrefetch.ComputeRefusal.Effects));
        Assert.True(Take(banks, behind));
    }

    [Fact]
    public void ADrawDoesNotTakeADispatchJobNorADispatchADrawJob()
    {
        var banks = Banks().Snapshot();
        var dispatch = Resolved(banks);
        _prefetch.Current = dispatch;
        var inputs = RenderExecutor.ProgramInputsOf(banks, _host.FormatSupport, _host.Fatal);
        Assert.False(_prefetch.TryTakeGraphics(banks, in inputs, out _));
        Assert.Same(dispatch, _prefetch.Current);
        Assert.True(_prefetch.TryTakeCompute(banks, Initiator, 4, 2, 1, false, out _));

        var draw = _prefetch.Submit(banks);
        _prefetch.WaitIdle();
        _prefetch.Current = draw;
        Assert.False(_prefetch.TryTakeCompute(banks, Initiator, 4, 2, 1, false, out _));
        Assert.Same(draw, _prefetch.Current);
    }

    // RenderExecutor.Dispatch: a taken program stands for the provider's own resolution; a refused one falls back to it.
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void TheExecutorResolvesADispatchItselfOnlyWhenItsJobIsRefused(bool current)
    {
        var banks = Banks().Snapshot();
        _concurrent.CurrentAcrossDispatches = current;
        _prefetch.Current = Resolved(banks);
        _executor.Dispatch(5, banks, 4, 2, 1, Initiator);
        Assert.Equal(current ? 0 : 1, _pipelines.Calls.Count(call => call.StartsWith("get_compute_program", StringComparison.Ordinal)));
        Assert.Equal(current ? 1 : 0, _prefetch.TakeComputeCounters().Taken);
        Assert.Null(_prefetch.Current);
    }

    // A job resolved before the program cache replaced an entry (its stamp moved) may hold an entry a lookup made now
    // would not find: refused as replanned, as a draw's.
    [Fact]
    public void AJobResolvedBeforeTheProgramCacheMovedIsRefused()
    {
        var banks = Banks().Snapshot();
        var job = Resolved(banks);
        _pipelines.ProgramState = new ProgramStateStamp(1, 0, 0);
        Assert.False(Take(banks, job));
        Assert.Equal(1, Refused(ParallelProgramPrefetch.ComputeRefusal.Replanned));
        Assert.True(Take(banks, Resolved(banks)));
    }

    // =verify: the worker resolves every taken dispatch job again and compares; a difference under answers that held
    // is fatal, one under answers that moved is a race the dispatch resolves itself.
    [Fact]
    public void VerifyResolvesATakenJobAgainAndCountsOrFailsOnADifference()
    {
        _prefetch.LookaheadMode = ParallelProgramPrefetch.DrawLookaheadMode.Verify;
        var banks = Banks().Snapshot();
        var job = Resolved(banks);
        var resolutions = _pipelines.ComputeAheadResolutions;
        Assert.True(Take(banks, job));
        Assert.Equal(resolutions + 1, _pipelines.ComputeAheadResolutions);
        Assert.Equal("dispatch_verify=(verified=1,raced=0,failed=0)", _prefetch.TakeComputeReport());

        // The worker now resolves another program while the job's answers moved: a race, refused and counted.
        job = Resolved(banks);
        var original = _pipelines.Compute;
        _pipelines.Compute = new ComputeProgram { Program = new ShaderProgram(0x44), Input = original.Input };
        _concurrent.Raced = true;
        Assert.False(Take(banks, job));
        Assert.Equal("dispatch_verify=(verified=1,raced=1,failed=0)", _prefetch.TakeComputeReport());

        // The same difference under answers that still hold is a hole: fatal, naming the job and the difference.
        _pipelines.Compute = original;
        job = Resolved(banks);
        _concurrent.Raced = false;
        _pipelines.Compute = new ComputeProgram { Program = new ShaderProgram(0x44), Input = original.Input };
        var fatal = Assert.Throws<RenderExecutorFatalException>(() => Take(banks, job));
        Assert.Contains("compute_program", fatal.Message);
        Assert.Contains($"job {job.Index}", fatal.Message);
        Assert.Contains("looked_over=[", fatal.Message);
        Assert.Contains("stamp_ahead=", fatal.Message);
        Assert.Equal("dispatch_verify=(verified=1,raced=0,failed=1)", _prefetch.TakeComputeReport());
    }

    // A program that consumed its dispatch counts as an effect for the jobs resolved ahead of it.
    [Fact]
    public void AConsumedDispatchNotesAnEffect()
    {
        var banks = Banks().Snapshot();
        _pipelines.Compute = new ComputeProgram { Consumed = true };
        _prefetch.AdvanceGate(2);
        var ahead = _prefetch.SubmitCompute(banks, 2, Initiator, 4, 2, 1, false);
        _prefetch.WaitIdle();
        _executor.Dispatch(5, banks, 4, 2, 1, Initiator);
        Assert.False(Take(banks, ahead));
        Assert.Equal(1, Refused(ParallelProgramPrefetch.ComputeRefusal.Effects));
    }
}
