// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.Libs.Tests.Gpu.Scheduling;
using Xunit;
using static SharpEmu.Libs.Tests.Gpu.Rendering.RenderExecutorFixtures;

namespace SharpEmu.Libs.Tests.Gpu.Rendering;

// After a job resolved, the resolving thread hands the host each stage it will bind (IConcurrentResolveHost.PrepareBindingsAhead):
// a draw's vertex stage, its pixel stage when the pixel stage runs, and a dispatch's compute stage when the provider stood for a
// program the dispatch would record. Nothing is handed over for a result the dispatch would not bind.
[Collection(SchedulingStateCollection.Name)]
public sealed class ParallelProgramPrefetchBindingTests : IDisposable
{
    private const uint Initiator = 0x41;

    private readonly RecordingRenderHost _host = new();
    private readonly FakePipelineProvider _pipelines = new();
    private readonly RenderExecutor _executor;
    private readonly FatalScope _fatal = new();
    private readonly ConcurrentHost _concurrent = new();
    private readonly ParallelProgramPrefetch _prefetch;

    public ParallelProgramPrefetchBindingTests()
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
        internal readonly List<(ShaderStageKind Stage, bool Dispatch)> Prepared = [];
        internal Action? OnPrepare;
        public void BeginConcurrentResolve() { }
        public object EndConcurrentResolve() => this;
        public bool IsStillCurrent(object assumptions) => true;

        public void PrepareBindingsAhead(in ShaderStageResources stage, bool dispatch)
        {
            lock (Prepared)
            {
                Prepared.Add((stage.Program!.Stage, dispatch));
            }

            OnPrepare?.Invoke();
        }
    }

    private ParallelProgramPrefetch.Job ResolvedDispatch()
    {
        var job = _prefetch.SubmitCompute(Banks().Snapshot(), 0, Initiator, 4, 2, 1, false);
        _prefetch.WaitIdle();
        return job;
    }

    [Fact]
    public void ADispatchJobPreparesItsComputeStageInTheSameJob()
    {
        var job = ResolvedDispatch();
        Assert.Equal(1, job.State);
        Assert.Equal([(ShaderStageKind.Compute, true)], _concurrent.Prepared);
    }

    [Theory]
    [InlineData(ComputePrefetchRefusal.Frontend)]
    [InlineData(ComputePrefetchRefusal.Unavailable)]
    public void ADispatchResultTheProviderDidNotStandForPreparesNothing(ComputePrefetchRefusal reason)
    {
        _pipelines.AheadRefusal = reason;
        ResolvedDispatch();
        Assert.Empty(_concurrent.Prepared);
    }

    [Fact]
    public void AConsumedOrUnavailableDispatchProgramPreparesNothing()
    {
        _pipelines.Compute = new ComputeProgram { Consumed = true };
        ResolvedDispatch();
        _pipelines.Compute = new ComputeProgram { Available = false };
        ResolvedDispatch();
        Assert.Empty(_concurrent.Prepared);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ADrawJobPreparesItsVertexStageAndItsPixelStageOnlyWhenThePixelStageRuns(bool withPixel)
    {
        var job = _prefetch.Submit(Banks(withPixel: withPixel).Snapshot());
        _prefetch.WaitIdle();
        Assert.Equal(1, job.State);
        Assert.Equal(withPixel
            ? [(ShaderStageKind.Vertex, false), (ShaderStageKind.Pixel, false)]
            : [(ShaderStageKind.Vertex, false)], _concurrent.Prepared);
    }

    // A host whose preparation throws (it must not) still leaves the job resolved for its draw.
    [Fact]
    public void APreparationFailureAbandonsTheJobAndTheDrawResolvesItself()
    {
        _concurrent.OnPrepare = () => throw GpuFatal.Fatal("synthetic preparation failure");
        var banks = Banks().Snapshot();
        var job = _prefetch.Submit(banks);
        _prefetch.WaitIdle();
        Assert.Equal(2, job.State);
        _prefetch.Current = job;
        var inputs = RenderExecutor.ProgramInputsOf(banks, _host.FormatSupport, _host.Fatal);
        Assert.False(_prefetch.TryTakeGraphics(banks, in inputs, out _));
    }
}
