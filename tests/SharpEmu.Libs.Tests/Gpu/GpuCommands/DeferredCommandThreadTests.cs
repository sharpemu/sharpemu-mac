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
