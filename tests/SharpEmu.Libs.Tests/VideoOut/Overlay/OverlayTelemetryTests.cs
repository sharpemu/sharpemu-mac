// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using SharpEmu.Libs.VideoOut;
using SharpEmu.Libs.VideoOut.Overlay;
using Xunit;

namespace SharpEmu.Libs.Tests.VideoOut.Overlay;

[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class OverlayTelemetryCollection
{
    public const string Name = "OverlayTelemetry";
}

[Collection(OverlayTelemetryCollection.Name)]
public sealed class OverlayTelemetryTests : IDisposable
{
    private static readonly long Ms = Stopwatch.Frequency / 1000;

    public OverlayTelemetryTests()
    {
        OverlayTelemetry.ManualBuildForTests = true;
        OverlayTelemetry.ResetForTests();
        OverlayTelemetry.DataSource.SetCaptureDemand(OverlayCaptureDemand.RenderPhases | OverlayCaptureDemand.Timeline);
    }

    public void Dispose()
    {
        OverlayTelemetry.DataSource.SetCaptureDemand(OverlayCaptureDemand.None);
        RenderPhaseProfile.SetOverlayCapture(false, false);
        OverlayTelemetry.ResetForTests();
    }

    [Theory]
    [InlineData("IdleNoQueuedWork", CostKind.Wait)]
    [InlineData("FrameSlotWait", CostKind.Wait)]
    [InlineData("ProgramPrefetchDrainWait", CostKind.Wait)]
    [InlineData("ProgramCompile", CostKind.Shader)]
    [InlineData("PipelineCreation", CostKind.Shader)]
    [InlineData("CommandDrawTranslation", CostKind.Render)]
    [InlineData("WindowEventPolling", CostKind.Other)]
    [InlineData("Unattributed", CostKind.Other)]
    public void PhasesAreClassifiedByWhatTheyDo(string phase, CostKind expected) =>
        Assert.Equal(expected, OverlayTelemetry.ClassifyPhase(phase));

    [Theory]
    [InlineData("libKernel:sceKernelWaitEqueue", true)]
    [InlineData("libKernel:scePthreadCondTimedwait", true)]
    [InlineData("libKernel:sceKernelUsleep", true)]
    [InlineData("libKernel:scePthreadMutexLock", true)]
    [InlineData("libKernel:scePthreadMutexUnlock", false)]
    [InlineData("libKernel:scePthreadJoin", true)]
    [InlineData("LibKernel:IdleNap", true)]
    [InlineData("libSceAgc:sceAgcDcbDrawIndex", false)]
    [InlineData("libSceVideoOut:sceVideoOutSubmitFlip", false)]
    public void BlockingExportsAreWaits(string name, bool expected) =>
        Assert.Equal(expected, OverlayTelemetry.IsWaitName(name));

    [Fact]
    public void SegmentBufferCoalescesConsecutivePhasesOfOneKind()
    {
        var buffer = new RenderSegmentBuffer();
        var draw = (int)RenderPhaseProfile.Phase.CommandDrawTranslation;
        var bind = (int)RenderPhaseProfile.Phase.ResourceBinding;
        var idle = (int)RenderPhaseProfile.Phase.IdleNoQueuedWork;
        buffer.Append(draw, 0, 10);
        buffer.Append(bind, 10, 40);
        buffer.Append(idle, 40, 50);
        buffer.Append(draw, 50, 55);

        Assert.Equal(3, buffer.Count);
        Assert.Equal((0L, 40L), (buffer.Start[0], buffer.End[0]));
        // The segment is named after its longest piece.
        Assert.Equal(bind, buffer.Phase[0]);
        Assert.Equal((byte)CostKind.Wait, buffer.Kind[1]);
    }

    [Fact]
    public void FramesCarryRenderPhasesTimelineAndGpuUnion()
    {
        var start = Stopwatch.GetTimestamp();
        OverlayTelemetry.EndRenderFrame(start, new long[(int)RenderPhaseProfile.Phase.Count], null);

        var phases = new long[(int)RenderPhaseProfile.Phase.Count];
        phases[(int)RenderPhaseProfile.Phase.CommandDrawTranslation] = 6 * Ms;
        phases[(int)RenderPhaseProfile.Phase.ProgramCompile] = 2 * Ms;
        phases[(int)RenderPhaseProfile.Phase.FrameSlotWait] = 8 * Ms;
        var segments = new RenderSegmentBuffer();
        segments.Append((int)RenderPhaseProfile.Phase.CommandDrawTranslation, start, start + 6 * Ms);
        segments.Append((int)RenderPhaseProfile.Phase.FrameSlotWait, start + 6 * Ms, start + 14 * Ms);

        // Two overlapping command buffers (union 7 ms) and one past the frame's end.
        OverlayTelemetry.RecordGpuInterval(start + 1 * Ms, start + 6 * Ms);
        OverlayTelemetry.RecordGpuInterval(start + 4 * Ms, start + 8 * Ms);
        OverlayTelemetry.RecordGpuInterval(start + 15 * Ms, start + 20 * Ms);

        var end = start + 16 * Ms;
        OverlayTelemetry.EndRenderFrame(end, phases, segments);
        OverlayTelemetry.BuildPendingForTests();

        var frame = Assert.Single(OverlayTelemetry.DataSource.Capture().Frames);
        Assert.Equal(16f, frame.FrameMs, 2);
        Assert.Equal(8f, frame.RenderBusyMs, 2);
        Assert.Equal(8f, frame.RenderWaitMs, 2);
        Assert.Equal(2f, frame.KindMs[(int)CostKind.Shader], 2);
        Assert.Equal(8f, frame.GpuMs, 2); // 7 ms union + 1 ms of the carried buffer
        Assert.Contains(frame.Costs, cost => cost.NameId == (int)RenderPhaseProfile.Phase.ProgramCompile && cost.Kind == CostKind.Shader);
        var render = frame.Segments.Where(segment => segment.Lane == OverlayLane.Render).ToArray();
        Assert.Equal(2, render.Length);
        Assert.Equal(6f, render[0].EndMs, 2);
        Assert.Equal(CostKind.Wait, render[1].Kind);
        Assert.Contains(frame.Segments, segment => segment.Lane == OverlayLane.Gpu);
        Assert.Equal("ProgramCompile", OverlayTelemetry.DataSource.Capture().NameOf((int)RenderPhaseProfile.Phase.ProgramCompile));

        // The buffer still running at the frame's end is counted again in the next frame.
        OverlayTelemetry.EndRenderFrame(end + 16 * Ms, null, null);
        OverlayTelemetry.BuildPendingForTests();
        Assert.Equal(4f, OverlayTelemetry.DataSource.Capture().Frames[^1].GpuMs, 2);
    }

    [Fact]
    public void HleCallsBecomeCostsWithWaitsSeparated()
    {
        OverlayTelemetry.DataSource.SetCaptureDemand(OverlayCaptureDemand.HleCalls);
        OverlayTelemetry.SetHleNameResolver(index => index switch
        {
            3 => "libKernel:sceKernelWaitEqueue",
            4 => "libSceAgc:sceAgcSubmit",
            _ => null,
        });
        var start = Stopwatch.GetTimestamp();
        OverlayTelemetry.EndRenderFrame(start, null, null);
        OverlayTelemetry.BuildPendingForTests();

        OverlayTelemetry.RecordHleCall(3, start, start + 5 * Ms);
        OverlayTelemetry.RecordHleCall(4, start, start + 1 * Ms);
        OverlayTelemetry.RecordHleCall(4, start, start + 1 * Ms);
        OverlayTelemetry.EndRenderFrame(start + 16 * Ms, null, null);
        OverlayTelemetry.BuildPendingForTests();

        var snapshot = OverlayTelemetry.DataSource.Capture();
        var frame = snapshot.Frames[^1];
        Assert.Equal(5f, frame.KindMs[(int)CostKind.Wait], 2);
        Assert.Equal(2f, frame.KindMs[(int)CostKind.Hle], 2);
        var submit = Assert.Single(frame.Costs, cost => snapshot.NameOf(cost.NameId) == "libSceAgc:sceAgcSubmit");
        Assert.Equal(2, submit.Count);
        Assert.Equal(CostKind.Hle, submit.Kind);
        Assert.Contains(frame.Costs, cost => snapshot.NameOf(cost.NameId) == "libKernel:sceKernelWaitEqueue" && cost.Kind == CostKind.Wait);

        // Nothing new: the next frame has no HLE cost.
        OverlayTelemetry.EndRenderFrame(start + 32 * Ms, null, null);
        OverlayTelemetry.BuildPendingForTests();
        Assert.DoesNotContain(OverlayTelemetry.DataSource.Capture().Frames[^1].Costs, cost => cost.NameId >= 1 << 16 && cost.NameId < 1 << 24);
    }

    [Fact]
    public void ALongFrameAmongSteadyOnesIsAHitch()
    {
        var timestamp = Stopwatch.GetTimestamp();
        OverlayTelemetry.EndRenderFrame(timestamp, null, null);
        for (var index = 0; index < 20; index++)
        {
            timestamp += 16 * Ms;
            OverlayTelemetry.EndRenderFrame(timestamp, null, null);
            OverlayTelemetry.BuildPendingForTests();
        }

        timestamp += 60 * Ms;
        OverlayTelemetry.EndRenderFrame(timestamp, null, null);
        OverlayTelemetry.BuildPendingForTests();
        var frames = OverlayTelemetry.DataSource.Capture().Frames;
        Assert.True(frames[^1].Hitch);
        Assert.NotNull(frames[^1].HitchReason);
        Assert.DoesNotContain(frames.Take(frames.Count - 1), frame => frame.Hitch);
    }

    [Fact]
    public void ThreadSamplerSeesTheCallingThread()
    {
        var sampler = OverlayThreadSampler.CreateForCurrentProcess();
        Assert.NotNull(sampler);
        var threads = new List<OverlayThreadSampler.ThreadCpu>();
        var spinUntil = Stopwatch.GetTimestamp() + 20 * Ms;
        while (Stopwatch.GetTimestamp() < spinUntil) { }
        Assert.True(sampler!.Sample(threads));
        var self = OverlayThreadSampler.CurrentThreadId();
        var mine = Assert.Single(threads, thread => thread.ThreadId == self);
        Assert.True(mine.CpuNanoseconds > 10_000_000, $"cpu ns {mine.CpuNanoseconds}");
    }

    [Fact]
    public void DrmFdInfoSumsEnginesOncePerClient()
    {
        var clients = new HashSet<string>();
        var totals = new Dictionary<string, long>();
        const string info = "pos:\t0\ndrm-driver:\tamdgpu\ndrm-client-id:\t7\ndrm-engine-gfx:\t1000 ns\ndrm-engine-compute:\t50 ns\ndrm-engine-capacity-gfx:\t1\n";
        LinuxDrmGpuUsage.ParseFdInfo(info, clients, totals);
        LinuxDrmGpuUsage.ParseFdInfo(info, clients, totals); // the same client through a dup'd fd
        Assert.Equal(1000, totals["gfx"]);
        Assert.Equal(50, totals["compute"]);
        Assert.False(totals.ContainsKey("capacity-gfx"));
    }
}
