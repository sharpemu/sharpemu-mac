// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.VideoOut;
using Xunit;

namespace SharpEmu.Libs.Tests.VideoOut;

public sealed class GpuBusyTrackerTests
{
    [Fact]
    public void DisjointIntervalsAreSummed()
    {
        var tracker = new GpuBusyTracker();
        tracker.Record(0.0, 0.1);
        tracker.Record(0.5, 0.7);
        Assert.Equal(0.3, tracker.TakeBusySeconds(0, 1), 9);
    }

    [Fact]
    public void OverlappingAndNestedIntervalsAreNotCountedTwice()
    {
        var tracker = new GpuBusyTracker();
        tracker.Record(0.0, 0.4);
        tracker.Record(0.3, 0.6); // overlaps the first
        tracker.Record(0.1, 0.2); // nested in the first
        tracker.Record(0.6, 0.7); // touches the second
        Assert.Equal(0.7, tracker.TakeBusySeconds(0, 1), 9);
    }

    [Fact]
    public void IntervalsRecordedOutOfOrderGiveTheSameUnion()
    {
        var tracker = new GpuBusyTracker();
        tracker.Record(10, 20);
        tracker.Record(5, 25);
        tracker.Record(30, 31);
        Assert.Equal(21, tracker.TakeBusySeconds(0, 100), 9);
    }

    [Fact]
    public void IntervalsAreClippedToTheWindow()
    {
        var tracker = new GpuBusyTracker();
        tracker.Record(0.9, 1.3);
        Assert.Equal(0.1, tracker.TakeBusySeconds(1.0 - 0.1, 1.0), 9);
    }

    [Fact]
    public void ABufferStillRunningAtTheWindowEndCountsItsRemainderInTheNextWindow()
    {
        var tracker = new GpuBusyTracker();
        tracker.Record(0.8, 1.4);
        Assert.Equal(0.2, tracker.TakeBusySeconds(0, 1), 9);
        Assert.Equal(0.4, tracker.TakeBusySeconds(1, 2), 9);
        Assert.Equal(0, tracker.TakeBusySeconds(2, 3), 9);
    }

    [Fact]
    public void FractionIsBusyTimeOverWindowAndNeverExceedsOne()
    {
        var tracker = new GpuBusyTracker();
        tracker.Record(0, 0.65);
        Assert.Equal(0.65, tracker.TakeBusyFraction(0, 1), 9);
        for (var index = 0; index < 10; index++)
        {
            tracker.Record(1.0 + index * 0.1 - 0.05, 1.0 + (index + 1) * 0.1 + 0.05);
        }

        Assert.Equal(1.0, tracker.TakeBusyFraction(1, 2), 9);
    }

    [Fact]
    public void EmptyAndDegenerateInputsReadAsIdle()
    {
        var tracker = new GpuBusyTracker();
        Assert.False(tracker.HasSamples);
        Assert.Equal(0, tracker.TakeBusyFraction(0, 1));
        tracker.Record(1, 1);
        tracker.Record(2, 1);
        tracker.Record(double.NaN, 3);
        Assert.False(tracker.HasSamples);
        tracker.Record(0, 0.5);
        Assert.True(tracker.HasSamples);
        Assert.Equal(0, tracker.TakeBusyFraction(1, 1));
        Assert.Equal(0, tracker.TakeBusyFraction(2, 1));
    }

    [Fact]
    public void ManyIntervalsStayBoundedAndExact()
    {
        var tracker = new GpuBusyTracker();
        // 200000 back-to-back 2 ms buffers, each overlapping the next by 1 ms: one continuous run.
        for (var index = 0; index < 200_000; index++)
        {
            tracker.Record(index * 0.001, index * 0.001 + 0.002);
        }

        Assert.Equal(200.001, tracker.TakeBusySeconds(0, 300), 6);
    }
}
