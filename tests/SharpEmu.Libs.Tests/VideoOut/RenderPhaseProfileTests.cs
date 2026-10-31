// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.VideoOut;
using Xunit;

namespace SharpEmu.Libs.Tests.VideoOut;

public sealed class RenderPhaseProfileTests
{
    [Fact]
    public void ResolveWorkers_ReportEveryScopeOnceAcrossThreads()
    {
        if (!RenderPhaseProfile.Enabled) return;

        var originalError = Console.Error;
        using var output = new StringWriter(System.Globalization.CultureInfo.InvariantCulture);
        using var start = new ManualResetEventSlim();
        try
        {
            Console.SetError(output);
            RenderPhaseProfile.ReportResolveWorker(1, 1);
            output.GetStringBuilder().Clear();
            var threads = Enumerable.Range(0, 4).Select(_ => new Thread(() =>
            {
                RenderPhaseProfile.AttachResolveWorker();
                start.Wait();
                for (var index = 0; index < 1000; index++)
                {
                    using var scope = RenderPhaseProfile.Measure(RenderPhaseProfile.Phase.ProgramPreparation);
                    using var detail = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ResourceMaterialization);
                }
            }) { IsBackground = true }).ToArray();

            foreach (var thread in threads) thread.Start();
            start.Set();
            foreach (var thread in threads) Assert.True(thread.Join(TimeSpan.FromSeconds(30)));

            RenderPhaseProfile.ReportResolveWorker(1, 1);
            var report = output.ToString();
            Assert.Matches(@"ProgramPreparation=[0-9.]+ms/n4000\b", report);
            Assert.Matches(@"ResourceMaterialization=[0-9.]+ms/n4000\b", report);

            output.GetStringBuilder().Clear();
            RenderPhaseProfile.ReportResolveWorker(1, 1);
            Assert.DoesNotContain("ms/n", output.ToString());
        }
        finally
        {
            start.Set();
            Console.SetError(originalError);
        }
    }

    [Theory]
    [InlineData((int)RenderPhaseProfile.Phase.ProgramTranslation)]
    [InlineData((int)RenderPhaseProfile.Phase.ProgramLayoutRequest)]
    public void PreparationDetail_RestoresTheOuterScopeAfterFailure(int phase)
    {
        using (RenderPhaseProfile.Measure(RenderPhaseProfile.Phase.ProgramPreparation))
        {
            Assert.Throws<InvalidOperationException>((Action)(() =>
            {
                using var detail = RenderPhaseProfile.MeasureDetail((RenderPhaseProfile.Phase)phase);
                throw new InvalidOperationException();
            }));
            Assert.Equal(RenderPhaseProfile.Enabled, RenderPhaseProfile.DetailMeasurementsEnabled);
        }
        Assert.False(RenderPhaseProfile.DetailMeasurementsEnabled);
    }

    [Fact]
    public void DisabledDetailMeasurement_DoesNotAllocateOnANewThread()
    {
        if (RenderPhaseProfile.Enabled) return;

        long allocated = -1;
        var thread = new Thread(() =>
        {
            var before = GC.GetAllocatedBytesForCurrentThread();
            using (RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ImageTracking))
            {
            }
            allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        }) { IsBackground = true };
        thread.Start();
        Assert.True(thread.Join(TimeSpan.FromSeconds(30)));
        Assert.Equal(0, allocated);
    }

    [Fact]
    public void DetailMeasurement_PreservesTheOuterScope()
    {
        Assert.False(RenderPhaseProfile.DetailMeasurementsEnabled);
        using (RenderPhaseProfile.Measure(RenderPhaseProfile.Phase.Draw))
        {
            Assert.Equal(RenderPhaseProfile.Enabled, RenderPhaseProfile.DetailMeasurementsEnabled);
            using (RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ImageTracking))
                Assert.Equal(RenderPhaseProfile.Enabled, RenderPhaseProfile.DetailMeasurementsEnabled);
            Assert.Equal(RenderPhaseProfile.Enabled, RenderPhaseProfile.DetailMeasurementsEnabled);
        }
        Assert.False(RenderPhaseProfile.DetailMeasurementsEnabled);
    }

    // Needs SHARPEMU_PROFILE_RENDER=1, which the static switch reads once at startup.
    [Fact]
    public void CommandStreamSubPhases_AreExclusiveAndTheParkIsIdleNotUnattributed()
    {
        if (!RenderPhaseProfile.Enabled) return;

        var stream = TimeSpan.FromMilliseconds(20);
        var detail = TimeSpan.FromMilliseconds(30);
        var failure = (Exception?)null;
        long streamTicks = 0, registerTicks = 0, readTicks = 0, idleTicks = 0, unattributedTicks = 0;
        var thread = new Thread(() =>
        {
            try
            {
                // The first charge on a thread has no earlier timestamp to measure from.
                using (RenderPhaseProfile.Measure(RenderPhaseProfile.Phase.Unattributed)) { }
                var streamBefore = RenderPhaseProfile.PhaseTicks(RenderPhaseProfile.Phase.CommandStream);
                var registerBefore = RenderPhaseProfile.PhaseTicks(RenderPhaseProfile.Phase.CommandRegisterWrite);
                var readBefore = RenderPhaseProfile.PhaseTicks(RenderPhaseProfile.Phase.CommandPacketRead);
                var idleBefore = RenderPhaseProfile.PhaseTicks(RenderPhaseProfile.Phase.Idle);
                var unattributedBefore = RenderPhaseProfile.PhaseTicks(RenderPhaseProfile.Phase.Unattributed);
                using (RenderPhaseProfile.Measure(RenderPhaseProfile.Phase.Idle))
                {
                    Thread.Sleep(detail);
                }

                using (RenderPhaseProfile.Measure(RenderPhaseProfile.Phase.CommandStream))
                {
                    Thread.Sleep(stream);
                    using (RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.CommandRegisterWrite))
                    {
                        Thread.Sleep(detail);
                    }

                    using (RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.CommandPacketRead))
                    {
                        Thread.Sleep(detail);
                    }
                }

                streamTicks = RenderPhaseProfile.PhaseTicks(RenderPhaseProfile.Phase.CommandStream) - streamBefore;
                registerTicks = RenderPhaseProfile.PhaseTicks(RenderPhaseProfile.Phase.CommandRegisterWrite) - registerBefore;
                readTicks = RenderPhaseProfile.PhaseTicks(RenderPhaseProfile.Phase.CommandPacketRead) - readBefore;
                idleTicks = RenderPhaseProfile.PhaseTicks(RenderPhaseProfile.Phase.Idle) - idleBefore;
                unattributedTicks = RenderPhaseProfile.PhaseTicks(RenderPhaseProfile.Phase.Unattributed) - unattributedBefore;
            }
            catch (Exception exception)
            {
                failure = exception;
            }
        }) { IsBackground = true };
        thread.Start();
        Assert.True(thread.Join(TimeSpan.FromSeconds(30)));
        Assert.Null(failure);

        double Milliseconds(long ticks) => ticks * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
        Assert.InRange(Milliseconds(registerTicks), detail.TotalMilliseconds - 1, double.MaxValue);
        Assert.InRange(Milliseconds(readTicks), detail.TotalMilliseconds - 1, double.MaxValue);
        // The slice's own time excludes both nested sub-phases (20 ms against 80 ms inclusive).
        Assert.InRange(Milliseconds(streamTicks), stream.TotalMilliseconds - 1, stream.TotalMilliseconds + detail.TotalMilliseconds - 1);
        Assert.InRange(Milliseconds(idleTicks), detail.TotalMilliseconds - 1, double.MaxValue);
        Assert.True(Milliseconds(unattributedTicks) < detail.TotalMilliseconds - 1,
            $"The park leaked into Unattributed: {Milliseconds(unattributedTicks):F1} ms.");
    }
}
