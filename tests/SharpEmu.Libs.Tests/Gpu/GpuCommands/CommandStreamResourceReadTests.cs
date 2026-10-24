// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.GpuCommands;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.GpuCommands;

public sealed class CommandStreamResourceReadTests
{
    private static void Enqueue(RecordingCommandStreamHost host, CommandStreamQueue queue,
        ulong address, bool compute, params uint[][] packets)
    {
        var words = StreamRunner.Concat(packets);
        host.WriteWords(address, words);
        if (compute) queue.EnqueueCompute(0x20, address, (uint)words.Length, 1, null);
        else queue.EnqueueGraphics(address, (uint)words.Length, 2, null);
    }

    [Fact]
    public void AResourceReadParksOnlyItsQueueAndResumesAtItsProducerSerial()
    {
        var ready = false;
        var host = new RecordingCommandStreamHost { ResourceReadDependency = () => ready ? 0UL : 9UL };
        var queue = new CommandStreamQueue(host);
        Enqueue(host, queue, StreamRunner.CommandAddress, true,
            StreamRunner.Packet(PacketOpcode.DispatchDirect, 1, 1, 1, 1),
            StreamRunner.Packet(PacketOpcode.NumInstances, 7));
        Assert.Equal(SliceResult.BlockedWithoutProgress, queue.ProcessOne());
        Assert.Equal(1u, queue.GetInterpreter(1).InstanceCount);
        Assert.Equal(CommandWaitKind.GpuSerial, Assert.Single(queue.TakeNewStalls(0)).Wait.Kind);

        // Another queue runs, and its completion does not retry the dependent packet.
        Enqueue(host, queue, StreamRunner.CommandAddress + 0x1000, false,
            StreamRunner.Packet(PacketOpcode.NumInstances, 5));
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.Equal(5u, queue.GetInterpreter(0).InstanceCount);
        Assert.Equal(SliceResult.AllBlocked, queue.ProcessOne());
        queue.NotifyResourceReadReady(8);
        queue.NotifyPipelineReady();
        queue.NotifyCompletedGpuTick(8);
        Assert.Equal(SliceResult.AllBlocked, queue.ProcessOne());
        queue.RetryBlocked();
        queue.NotifyPublicationCompleted();
        Assert.Equal(SliceResult.AllBlocked, queue.ProcessOne());

        ready = true;
        queue.NotifyResourceReadReady(9);
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.Equal(7u, queue.GetInterpreter(1).InstanceCount);
        Assert.False(queue.HasPending);
    }

    [Fact]
    public void AResourceCompletionBetweenTheReadAndParkIsNotLost()
    {
        var host = new RecordingCommandStreamHost();
        var queue = new CommandStreamQueue(host);
        var first = true;
        host.ResourceReadDependency = () =>
        {
            if (!first) return 0;
            first = false;
            // The completion arrives before SuspendOn installs the wait condition.
            queue.NotifyResourceReadReady(12);
            return 12;
        };
        Enqueue(host, queue, StreamRunner.CommandAddress, true,
            StreamRunner.Packet(PacketOpcode.DispatchDirect, 1, 1, 1, 1),
            StreamRunner.Packet(PacketOpcode.NumInstances, 6));
        Assert.Equal(SliceResult.BlockedWithoutProgress, queue.ProcessOne());
        Assert.True(queue.HasUnblockedPending);
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.Equal(6u, queue.GetInterpreter(1).InstanceCount);
    }

    [Fact]
    public void AnUnrelatedWakeRacingTheReadDoesNotRetryThePendingProducer()
    {
        var host = new RecordingCommandStreamHost();
        var queue = new CommandStreamQueue(host);
        var ready = false;
        host.ResourceReadDependency = () =>
        {
            if (ready) return 0;
            queue.NotifyPipelineReady();
            return 12;
        };
        Enqueue(host, queue, StreamRunner.CommandAddress, true,
            StreamRunner.Packet(PacketOpcode.DispatchDirect, 1, 1, 1, 1),
            StreamRunner.Packet(PacketOpcode.NumInstances, 8));
        Assert.Equal(SliceResult.BlockedWithoutProgress, queue.ProcessOne());
        Assert.False(queue.HasUnblockedPending);
        Assert.Equal(SliceResult.AllBlocked, queue.ProcessOne());

        ready = true;
        queue.NotifyResourceReadReady(12);
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.Equal(8u, queue.GetInterpreter(1).InstanceCount);
    }
}
