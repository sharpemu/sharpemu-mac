// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.GpuCommands;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.GpuCommands;

public sealed class DeferredCommandThreadTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ProducerIdentityAndOrderedConsumerWorkSurviveThreadSelection(bool useHostQos)
    {
        var host = new RecordingCommandStreamHost();
        var queue = new CommandStreamQueue(host);
        var producer = new DeferredCommandThread(queue, host.Memory, _ => { }, null, () => { }, useHostQos);
        using var entered = new ManualResetEventSlim();
        using var completed = new ManualResetEventSlim();
        bool producerIdentity = false, consumerIdentity = true;
        var values = new List<int>();
        Assert.True(queue.TryEnqueueControlBarrier(() =>
        {
            producerIdentity = producer.IsProducerThread;
            producer.Post(() => values.Add(1));
            entered.Set();
            producer.Run(() => { consumerIdentity = producer.IsProducerThread; values.Add(2); });
            completed.Set();
        }, () => { }));
        producer.Start();
        try
        {
            Assert.False(producer.IsProducerThread);
            Assert.True(entered.Wait(TimeSpan.FromSeconds(10)));
            Assert.True(producer.IsAlive);
            Assert.Throws<ThreadStateException>(producer.Start);
            Assert.True(SpinWait.SpinUntil(() =>
            {
                producer.RunBatch(long.MaxValue, 64, () => { });
                return completed.IsSet;
            }, TimeSpan.FromSeconds(10)));
            producer.ThrowIfFailed();
            Assert.True(producerIdentity);
            Assert.False(consumerIdentity);
            Assert.Equal(new[] { 1, 2 }, values);
        }
        finally
        {
            producer.Stop(() => producer.RunBatch(long.MaxValue, 64, () => { }));
        }
        Assert.False(producer.IsAlive);
    }

    // A barrier or pass end only orders GPU commands: it runs in stream order like any other
    // item, but the draws after it do not wait for it to resolve their programs.
    [Fact]
    public void AnOrderingItemRunsInOrderWithoutHoldingTheGate()
    {
        var host = new RecordingCommandStreamHost();
        var queue = new CommandStreamQueue(host);
        var producer = new DeferredCommandThread(queue, host.Memory, _ => { }, null, () => { }, false);
        using var completed = new ManualResetEventSlim();
        var values = new List<int>();
        long gateAfterPost = -1, gateAfterOrdering = -1;
        Assert.True(queue.TryEnqueueControlBarrier(() =>
        {
            producer.Post(() => values.Add(1));
            gateAfterPost = producer.Gate;
            producer.PostOrdering(() => values.Add(2));
            gateAfterOrdering = producer.Gate;
            producer.Run(() => values.Add(3));
            completed.Set();
        }, () => { }));
        producer.Start();
        try
        {
            Assert.True(SpinWait.SpinUntil(() =>
            {
                producer.RunBatch(long.MaxValue, 64, () => { });
                return completed.IsSet;
            }, TimeSpan.FromSeconds(10)));
            producer.ThrowIfFailed();
            Assert.Equal(new[] { 1, 2, 3 }, values);
            Assert.Equal(gateAfterPost, gateAfterOrdering);
            Assert.Equal(gateAfterPost + 1, producer.Gate);
        }
        finally
        {
            producer.Stop(() => producer.RunBatch(long.MaxValue, 64, () => { }));
        }
    }

    [Fact]
    public void AnUngatedWriteHoldsOnlyItsRangeForTheResolutionsAfterIt()
    {
        var host = new RecordingCommandStreamHost();
        var queue = new CommandStreamQueue(host);
        var producer = new DeferredCommandThread(queue, host.Memory, _ => { }, null, () => { }, false);
        using var queued = new ManualResetEventSlim();
        long gateBefore = -1, gateAfter = -1;
        bool heldInside = false, heldOutside = true, heldEarlierJob = true, heldGated = true;
        Assert.True(queue.TryEnqueueControlBarrier(() =>
        {
            gateBefore = producer.Gate;
            producer.PostRange(() => { }, 0x1000, 0x100, ungated: true);
            gateAfter = producer.Gate;
            producer.PostRange(() => { }, 0x8000, 0x100);
            heldInside = producer.PendingRangeHolds(0, 0x10F0, 0x20);
            heldOutside = producer.PendingRangeHolds(0, 0x1100, 0x20);
            heldEarlierJob = producer.PendingRangeHolds(-1, 0x1000, 0x10);
            heldGated = producer.PendingRangeHolds(0, 0x8000, 0x10);
            queued.Set();
        }, () => { }));
        producer.Start();
        try
        {
            Assert.True(queued.Wait(TimeSpan.FromSeconds(10)));
            Assert.Equal(gateBefore, gateAfter);
            Assert.True(heldInside);
            Assert.False(heldOutside);
            Assert.False(heldEarlierJob);
            Assert.False(heldGated);
            Assert.True(SpinWait.SpinUntil(() =>
            {
                producer.RunBatch(long.MaxValue, 64, () => { });
                return !producer.PendingRangeHolds(0, 0x1000, 0x100);
            }, TimeSpan.FromSeconds(10)));
            producer.ThrowIfFailed();
        }
        finally
        {
            producer.Stop(() => producer.RunBatch(long.MaxValue, 64, () => { }));
        }
    }

    [Fact]
    public void AQueuedFillAnswersReadsOfTheBytesItCoversUntilAWriteQueuedAfterItOverlaps()
    {
        var host = new RecordingCommandStreamHost();
        var queue = new CommandStreamQueue(host);
        var producer = new DeferredCommandThread(queue, host.Memory, _ => { }, null, () => { }, false);
        using var queued = new ManualResetEventSlim();
        var inside = new byte[4];
        var shifted = new byte[2];
        bool readInside = false, readShifted = false, readPartial = true, readCovered = true, readBehindLabel = false;
        Assert.True(queue.TryEnqueueControlBarrier(() =>
        {
            producer.PostFill(() => { }, 0x1000, 0x10, 0x44332211u, ungated: true);
            readInside = producer.TryReadQueuedFill(0x1004, inside);
            readShifted = producer.TryReadQueuedFill(0x1001, shifted);
            readPartial = producer.TryReadQueuedFill(0x100E, new byte[4]);
            producer.PostLabel(() => { }, 0x1000, 0x8, ungated: true);
            readBehindLabel = producer.TryReadQueuedFill(0x1000, new byte[4]);
            producer.PostRange(() => { }, 0x1008, 0x4, ungated: true);
            readCovered = producer.TryReadQueuedFill(0x1008, new byte[4]);
            queued.Set();
        }, () => { }));
        producer.Start();
        try
        {
            Assert.True(queued.Wait(TimeSpan.FromSeconds(10)));
            Assert.True(readInside);
            Assert.Equal(new byte[] { 0x11, 0x22, 0x33, 0x44 }, inside);
            Assert.True(readShifted);
            Assert.Equal(new byte[] { 0x22, 0x33 }, shifted);
            Assert.False(readPartial);
            Assert.True(readBehindLabel);
            Assert.False(readCovered);
            Assert.True(SpinWait.SpinUntil(() =>
            {
                producer.RunBatch(long.MaxValue, 64, () => { });
                return !producer.TryReadQueuedFill(0x1000, new byte[4]);
            }, TimeSpan.FromSeconds(10)));
            producer.ThrowIfFailed();
        }
        finally
        {
            producer.Stop(() => producer.RunBatch(long.MaxValue, 64, () => { }));
        }
    }

    [Fact]
    public void QueuedWritesOfKnownBytesAnswerReadsTheyFullyCoverWithoutDraining()
    {
        var host = new RecordingCommandStreamHost();
        var queue = new CommandStreamQueue(host);
        var producer = new DeferredCommandThread(queue, host.Memory, _ => { }, null, () => { }, false);
        using var queued = new ManualResetEventSlim();
        var written = new byte[4];
        var qword = new byte[8];
        bool readWrite = false, readPastWrite = true, readQword = false, readGap = true, readShadowed = true;
        Assert.True(queue.TryEnqueueControlBarrier(() =>
        {
            // An 8-byte command-processor write: reads inside it are its bytes.
            producer.PostWrite(() => { }, 0x2000, [1, 2, 3, 4, 5, 6, 7, 8], ungated: true);
            readWrite = producer.TryReadQueuedFill(0x2002, written);
            readPastWrite = producer.TryReadQueuedFill(0x2006, new byte[4]);
            // Two qwords 16 bytes apart: the gap between them is not this item's to answer.
            producer.PostQwords(() => { }, 0x3000, 0x18, 0x1122334455667788ul, 0x10, ungated: true);
            readQword = producer.TryReadQueuedFill(0x3010, qword);
            readGap = producer.TryReadQueuedFill(0x3008, new byte[4]);
            // An unknown range queued after the write decides its bytes again.
            producer.PostRange(() => { }, 0x2000, 0x4, ungated: true);
            readShadowed = producer.TryReadQueuedFill(0x2000, new byte[4]);
            queued.Set();
        }, () => { }));
        producer.Start();
        try
        {
            Assert.True(queued.Wait(TimeSpan.FromSeconds(10)));
            Assert.True(readWrite);
            Assert.Equal(new byte[] { 3, 4, 5, 6 }, written);
            Assert.False(readPastWrite);
            Assert.True(readQword);
            Assert.Equal(BitConverter.GetBytes(0x1122334455667788ul), qword);
            Assert.False(readGap);
            Assert.False(readShadowed);
            producer.ThrowIfFailed();
        }
        finally
        {
            producer.Stop(() => producer.RunBatch(long.MaxValue, 64, () => { }));
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ProducerFailureIsReportedAndThreadCanBeStopped(bool useHostQos)
    {
        var host = new RecordingCommandStreamHost();
        var queue = new CommandStreamQueue(host);
        using var failed = new ManualResetEventSlim();
        var expected = new InvalidOperationException("producer failure");
        var producer = new DeferredCommandThread(queue, host.Memory, _ => { }, null, failed.Set, useHostQos);
        Assert.True(queue.TryEnqueueControlBarrier(() => throw expected, () => { }));
        producer.Start();
        try
        {
            Assert.True(failed.Wait(TimeSpan.FromSeconds(10)));
            Assert.Same(expected, Assert.Throws<InvalidOperationException>(producer.ThrowIfFailed).InnerException);
        }
        finally
        {
            producer.Stop(() => { });
        }
        Assert.False(producer.IsAlive);
    }
}
