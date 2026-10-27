// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.GpuCommands;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.GpuCommands;

[CollectionDefinition(CommandStreamQueueStateCollection.Name, DisableParallelization = true)]
public sealed class CommandStreamQueueStateCollection
{
    public const string Name = "CommandStreamQueueState";
}

[Collection(CommandStreamQueueStateCollection.Name)]
public sealed class CommandStreamQueueTests
{
    private const ulong Label = StreamRunner.LabelAddress;
    private const ulong Graphics = StreamRunner.CommandAddress;
    private const ulong Compute = StreamRunner.DataAddress;

    private static uint[] CreateInstanceCountPacket(uint count) => StreamRunner.Packet(PacketOpcode.NumInstances, count);

    private static uint[] WaitEqual(ulong address, uint reference) =>
        StreamRunner.Packet(PacketOpcode.WaitRegisterMemory, 0x13u, StreamRunner.Low(address), StreamRunner.High(address), reference, 0xFFFF_FFFFu, 0);

    private static uint[] WriteLabel(ulong address, uint value) =>
        StreamRunner.Packet(PacketOpcode.WriteData, 0x00000500u, StreamRunner.Low(address), StreamRunner.High(address), value);

    private static (RecordingCommandStreamHost Host, CommandStreamQueue Queue) NewQueue()
    {
        var host = new RecordingCommandStreamHost();
        return (host, new CommandStreamQueue(host));
    }

    [Theory]
    [InlineData(0UL)]
    [InlineData(4UL)]
    public void APartialLabelReplacementCannotAnswerWithTheOldQword(ulong offset)
    {
        var (host, queue) = NewQueue();
        host.PendingLabelTick = 5;
        var interpreter = queue.GetInterpreter(0);
        interpreter.PendingLabelRecorded!(Label, 8, 0x22222222_11111111UL);
        interpreter.PendingLabelRecorded!(Label + offset, 4, 0x33333333);
        Assert.False(interpreter.PendingLabelLookup!(Label, true, out _));
        Assert.True(interpreter.PendingLabelLookup!(Label + offset, false, out var replacement));
        Assert.Equal(0x33333333UL, replacement);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AQwordReplacementRemovesAnOlderUpperDword(bool batch)
    {
        var (host, queue) = NewQueue();
        host.PendingLabelTick = 5;
        var interpreter = queue.GetInterpreter(0);
        interpreter.PendingLabelRecorded!(Label + 4, 4, 0xBAD);
        if (batch) interpreter.PendingQwordsRecorded!(Label, 2, 16, 0x22222222_11111111UL);
        else interpreter.PendingLabelRecorded!(Label, 8, 0x22222222_11111111UL);
        Assert.True(interpreter.PendingLabelLookup!(Label + 4, false, out var high));
        Assert.Equal(0x22222222UL, high);
        queue.NotifyCompletedGpuTick(5);
        Assert.False(interpreter.PendingLabelLookup!(Label, true, out _));
    }

    [Fact]
    public void AnUnalignedReplacementRefusesStaleCachedAnswers()
    {
        var (host, queue) = NewQueue();
        host.PendingLabelTick = 5;
        var interpreter = queue.GetInterpreter(0);
        interpreter.PendingLabelRecorded!(Label, 8, 0x22222222_11111111UL);
        interpreter.PendingLabelRecorded!(Label + 2, 4, 0x33333333);
        Assert.False(interpreter.PendingLabelLookup!(Label, true, out _));
        Assert.False(interpreter.PendingLabelLookup!(Label + 2, false, out _));
    }

    private static void Enqueue(RecordingCommandStreamHost host, CommandStreamQueue queue, ulong address, ulong submissionId, params uint[][] packets)
    {
        var words = StreamRunner.Concat(packets);
        host.WriteWords(address, words);
        queue.EnqueueGraphics(address, (uint)words.Length, submissionId, null);
    }

    private static void EnqueueCompute(RecordingCommandStreamHost host, CommandStreamQueue queue, uint computeQueue, ulong address, ulong submissionId, params uint[][] packets)
    {
        var words = StreamRunner.Concat(packets);
        host.WriteWords(address, words);
        queue.EnqueueCompute(computeQueue, address, (uint)words.Length, submissionId, null);
    }

    [Fact]
    public void GraphicsSubmission_RunsBeginsCollectsAndFlushes()
    {
        var (host, queue) = NewQueue();
        Enqueue(host, queue, Graphics, 7, CreateInstanceCountPacket(3));

        Assert.Equal(SliceResult.Completed, queue.ProcessOne());

        Assert.Equal(new[] { "begin 0 7", "gc", "flush" }, host.Calls);
        Assert.Equal(3u, queue.GetInterpreter(0).InstanceCount);
        Assert.Equal(1UL, queue.GetInterpreter(0).SubmitId);
        Assert.Equal(SliceResult.NoWork, queue.ProcessOne());
        Assert.Equal(IdleOutcome.Completed, queue.WaitForIdle());
    }

    private static uint[] ReleaseMemoryInterrupt(ulong label, uint value) =>
        StreamRunner.Packet(
            PacketOpcode.ReleaseMemory,
            0x28u | (5u << 8),
            (2u << 24) | (1u << 29),
            StreamRunner.Low(label), StreamRunner.High(label),
            value, 0,
            0);

    // Release-memory packets do not submit on their own; the slice end does, so the
    // interrupt they queue always reaches the GPU once the slice is over.
    [Fact]
    public void ReleaseMemoryInterrupt_IsSubmittedAtTheEndOfTheSlice()
    {
        var (host, queue) = NewQueue();
        Enqueue(host, queue, Graphics, 1, ReleaseMemoryInterrupt(Label, 5), CreateInstanceCountPacket(2), ReleaseMemoryInterrupt(Label, 6));

        Assert.Equal(SliceResult.Completed, queue.ProcessOne());

        Assert.Equal(new[] { "begin 0 1", "eop Interrupt32", "eop Interrupt32", "gc", "flush" }, host.Calls);
    }

    // A slice that blocks after a release still submits it, so a wait that depends on
    // the guest reacting to that interrupt cannot hold the interrupt back.
    [Fact]
    public void ReleaseMemoryInterrupt_IsSubmittedWhenTheSliceBlocks()
    {
        var (host, queue) = NewQueue();
        Enqueue(host, queue, Graphics, 1, ReleaseMemoryInterrupt(Label, 5), WaitEqual(Label + 8, 1));

        Assert.Equal(SliceResult.Progressed, queue.ProcessOne());

        var release = host.Calls.IndexOf("eop Interrupt32");
        Assert.True(release >= 0 && host.Calls.IndexOf("flush") > release, string.Join(", ", host.Calls));
    }

    // A video-out export flip captures after the draws submitted before it, never ahead of them.
    [Fact]
    public void FlipPreparation_RunsAfterTheGraphicsSubmissionsBeforeIt()
    {
        var (host, queue) = NewQueue();
        Enqueue(host, queue, Graphics, 1, StreamRunner.Packet(PacketOpcode.DrawIndexAuto, 3, 0));
        Assert.True(queue.TryEnqueueFlipPreparation(1, 2, 77));

        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.Equal(SliceResult.NoWork, queue.ProcessOne());

        var draw = host.Calls.IndexOf("draw_auto 1 3");
        var flip = host.Calls.IndexOf("cpu_flip 1 2 77");
        Assert.True(draw >= 0 && flip > draw, string.Join(", ", host.Calls));
        Assert.Equal("flush", host.Calls[flip + 1]);

        queue.StopAccepting();
        Assert.False(queue.TryEnqueueFlipPreparation(1, 2, 78));
    }

    [Fact]
    public void CpuFlipCapturesInOrderWithoutWaitingForPublication()
    {
        var (host, queue) = NewQueue();
        var pending = true;
        host.PendingPublications = _ => pending;
        Assert.True(queue.TryEnqueueFlipPreparation(1, 2, 77));
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.Contains("cpu_flip 1 2 77", host.Calls);
    }

    [Fact]
    public void SubmitIds_AreSharedAcrossQueuesAndEventIdsFollowTheQueue()
    {
        var (host, queue) = NewQueue();
        Enqueue(host, queue, Graphics, 1, CreateInstanceCountPacket(1));
        EnqueueCompute(host, queue, 0x21, Compute, 2, CreateInstanceCountPacket(1));

        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());

        Assert.Equal(1UL, queue.GetInterpreter(0).SubmitId);
        Assert.Equal(2UL, queue.GetInterpreter(2).SubmitId);
        Assert.Equal(0, queue.GetInterpreter(0).InterruptEventId);
        Assert.Equal(0x21, queue.GetInterpreter(2).InterruptEventId);
        Assert.Equal(new[] { "begin 0 1", "gc", "flush", "begin 2 2", "gc", "flush" }, host.Calls);
        Assert.Contains("compute queue is outside", Assert.Throws<CommandStreamFatalException>(() => queue.EnqueueCompute(0x58, Compute, 2, 3, null)).Message);
        Assert.Contains("compute submission is empty", Assert.Throws<CommandStreamFatalException>(() => queue.EnqueueCompute(0x20, Compute, 0, 3, null)).Message);
    }

    [Fact]
    public void Queues_AlternateRoundRobin()
    {
        var (host, queue) = NewQueue();
        Enqueue(host, queue, Graphics, 1, CreateInstanceCountPacket(1));
        Enqueue(host, queue, Graphics + 0x100, 2, CreateInstanceCountPacket(1));
        EnqueueCompute(host, queue, 0x20, Compute, 3, CreateInstanceCountPacket(1));

        for (var slice = 0; slice < 3; slice++)
        {
            Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        }

        Assert.Equal(new[] { "begin 0 1", "begin 1 3", "begin 0 2" }, host.Calls.Where(call => call.StartsWith("begin", StringComparison.Ordinal)));
    }

    [Fact]
    public void ControlBarrier_RunsBetweenTheAcceptedSliceAndLaterAdmissions()
    {
        var (host, queue) = NewQueue();
        Enqueue(host, queue, Graphics, 1, CreateInstanceCountPacket(1));
        var cancelled = false;
        host.PendingCommands.Enqueue(() =>
        {
            Assert.True(queue.TryEnqueueControlBarrier(
                () => host.Calls.Add("mapping_change"),
                () => cancelled = true));
            Enqueue(host, queue, Graphics + 0x100, 2, CreateInstanceCountPacket(2));
        });

        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.DoesNotContain("mapping_change", host.Calls);
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());

        var first = host.Calls.IndexOf("begin 0 1");
        var barrier = host.Calls.IndexOf("mapping_change");
        var second = host.Calls.IndexOf("begin 0 2");
        Assert.True(first >= 0 && barrier > first && second > barrier, string.Join(", ", host.Calls));
        Assert.False(cancelled);
    }

    [Fact]
    public void ControlBarrier_PreservesCrossQueueProgressWhileAnOlderHeadIsBlocked()
    {
        var (host, queue) = NewQueue();
        host.WriteDword(Label, 0);
        Enqueue(host, queue, Graphics, 1, WaitEqual(Label, 1));
        Assert.Equal(SliceResult.BlockedWithoutProgress, queue.ProcessOne());
        Assert.True(queue.TryEnqueueControlBarrier(() => host.Calls.Add("mapping_change"), () => { }));
        EnqueueCompute(host, queue, 0x20, Compute, 2, CreateInstanceCountPacket(2));

        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.DoesNotContain("mapping_change", host.Calls);
        host.WriteDword(Label, 1);
        queue.RetryBlocked();
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());

        var compute = host.Calls.IndexOf("begin 1 2");
        var retriedGraphics = host.Calls.FindLastIndex(call => call == "begin 0 1");
        var barrier = host.Calls.IndexOf("mapping_change");
        Assert.True(compute >= 0 && retriedGraphics > compute && barrier > retriedGraphics, string.Join(", ", host.Calls));
    }

    [Fact]
    public void ControlBarrier_IsCancelledWhenTheQueueIsDiscarded()
    {
        var (_, queue) = NewQueue();
        var ran = false;
        var cancelled = false;
        Assert.True(queue.TryEnqueueControlBarrier(() => ran = true, () => cancelled = true));

        queue.DiscardAll();

        Assert.False(ran);
        Assert.True(cancelled);
        Assert.False(queue.HasPending);
    }

    [Fact]
    public void ShutdownDrain_RunsAnAcceptedControlBarrier()
    {
        var (_, queue) = NewQueue();
        var ran = false;
        Assert.True(queue.TryEnqueueControlBarrier(() => ran = true, () => { }));
        queue.StopAccepting();

        Assert.Equal(IdleOutcome.Completed, queue.DrainForShutdown(cancelBlockedOnNoProgress: true));
        Assert.True(ran);
        Assert.False(queue.HasPending);
    }

    [Fact]
    public void BlockedHead_IsRequeuedAndRetriedWithoutStoppingOtherQueues()
    {
        var (host, queue) = NewQueue();
        host.WriteDword(Label, 0);
        Enqueue(host, queue, Graphics, 1, WaitEqual(Label, 1), CreateInstanceCountPacket(9));
        EnqueueCompute(host, queue, 0x20, Compute, 2, CreateInstanceCountPacket(5));

        Assert.Equal(SliceResult.BlockedWithoutProgress, queue.ProcessOne());
        Assert.Equal(1, queue.BlockedQueueCount);
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        // The compute submission finishing says nothing about the label the graphics
        // head is comparing, so that head is left asleep instead of re-running a
        // slice to read the same value and suspend again.
        Assert.Equal(1, queue.BlockedQueueCount);
        Assert.Equal(SliceResult.AllBlocked, queue.ProcessOne());

        // A write the queue cannot observe — here the test host's own — is what the
        // blocked-retry fallback is for.
        host.WriteDword(Label, 1);
        queue.RetryBlocked();
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.Equal(9u, queue.GetInterpreter(0).InstanceCount);
        Assert.Equal(1UL, queue.BlockedRetries);
    }

    /// <summary>
    /// A head waiting on guest memory wakes when those bytes are written and stays
    /// asleep when other bytes are. Waking every head on every event is what made
    /// Demon's Souls run 952,889 slices for 371,022 submissions.
    /// </summary>
    /// <summary>
    /// Async compute waiting on a graphics fence the GPU has yet to write. Every queue
    /// encodes into one GPU stream in order, so the waiter may go on as soon as the label
    /// is recorded; waiting until the GPU wrote it and the guest was told let the guest
    /// recycle the fence slot first, and SILENT HILL's compute queue then waited for ever
    /// on bytes that had become a pool pointer.
    /// </summary>
    [Fact]
    public void AnotherQueuesRecordedLabelSatisfiesAWaitUntilItLands()
    {
        var (host, queue) = NewQueue();
        host.DefersEndOfPipeLabels = true;
        host.PendingLabelTick = 5;
        host.WriteDword(Label, 0);
        EnqueueCompute(host, queue, 0x20, Compute, 1, WaitEqual(Label, 1), WriteLabel(Label + 8, 7));
        Assert.Equal(SliceResult.BlockedWithoutProgress, queue.ProcessOne());

        Enqueue(host, queue, Graphics, 2, ReleaseMemoryInterrupt(Label, 1));
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());

        // The GPU has not written the label, yet the compute head goes on.
        Assert.Equal(0u, host.ReadDword(Label));
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.Equal(7u, host.ReadDword(Label + 8));

        // Once that tick completes the label is memory's business again: the slot still
        // holds 0, so a later wait on it sleeps instead of seeing the old value.
        queue.NotifyCompletedGpuTick(5);
        EnqueueCompute(host, queue, 0x20, Compute, 3, WaitEqual(Label, 1), WriteLabel(Label + 8, 9));
        Assert.Equal(SliceResult.BlockedWithoutProgress, queue.ProcessOne());
        Assert.Equal(7u, host.ReadDword(Label + 8));
    }

    /// <summary>
    /// A wait on this queue's own recorded label is satisfied by the label, so nothing is
    /// committed for it: a commit would end the open render pass and store every attachment
    /// for a wait that never sleeps. Only a wait nothing recorded satisfies commits.
    /// </summary>
    [Fact]
    public void AWaitItsOwnRecordedLabelSatisfiesCommitsNothing()
    {
        var (host, queue) = NewQueue();
        host.DefersEndOfPipeLabels = true;
        host.PendingLabelTick = 5;
        host.WriteDword(Label, 0);
        Enqueue(host, queue, Graphics, 1, ReleaseMemoryInterrupt(Label, 1), WaitEqual(Label, 1), WriteLabel(Label + 8, 7));

        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.Equal(0u, host.ReadDword(Label));
        Assert.Equal(7u, host.ReadDword(Label + 8));
        Assert.Empty(host.PollCommits);

        Enqueue(host, queue, Graphics, 2, WaitEqual(Label, 2), WriteLabel(Label + 8, 9));
        Assert.Equal(SliceResult.BlockedWithoutProgress, queue.ProcessOne());
        Assert.Equal(new[] { Label }, host.PollCommits);
        Assert.Equal(7u, host.ReadDword(Label + 8));
    }

    /// <summary>
    /// Demon's Souls' drain shape: a label, a wait on it, then a store the guest recycles
    /// command memory on. The wait is met at parse by the recorded label, and the store used
    /// to be written then too, long before the GPU wrote the label - which then landed in
    /// memory the guest had already reused. On a host that lands command-processor writes in
    /// order the store is recorded behind the label instead, and a wait on it is still met.
    /// </summary>
    [Fact]
    public void AStoreAfterAWaitOnARecordedLabelIsRecordedBehindTheLabel()
    {
        var (host, queue) = NewQueue();
        host.DefersEndOfPipeLabels = true;
        host.DefersCommandProcessorWrites = true;
        host.PendingLabelTick = 5;
        var signal = Label + 0x10;
        host.WriteDword(Label, 0);
        host.WriteDword(signal, 0);
        Enqueue(host, queue, Graphics, 1, ReleaseMemoryInterrupt(Label, 1), WaitEqual(Label, 1), WriteLabel(signal, 0x5A17));

        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.Equal(0u, host.ReadDword(Label));
        Assert.Equal(0u, host.ReadDword(signal));
        var label = host.Calls.FindIndex(call => call.StartsWith("eop ", StringComparison.Ordinal));
        var store = host.Calls.IndexOf($"cp_write WRITE_DATA {signal:X} 4");
        Assert.True(label >= 0 && store > label, string.Join(", ", host.Calls));
        Assert.Empty(host.PollCommits);
        var entry = Assert.Single(CommandProcessorWriteHistory.Find(signal, sizeof(uint)), e => e.Kind == "WRITE_DATA" && e.Value == 0x5A17);
        Assert.True(entry.Deferred);

        // Waits on the store, this queue's and another's, read it as written.
        Enqueue(host, queue, Graphics, 2, WaitEqual(signal, 0x5A17), WriteLabel(signal + 8, 9));
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        EnqueueCompute(host, queue, 0x20, Compute, 3, WaitEqual(signal, 0x5A17), WriteLabel(signal + 0x10, 3));
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.Empty(host.PollCommits);

        // The compute queue's store is recorded in order too.
        Assert.Equal(0u, host.ReadDword(signal + 0x10));
        Assert.Equal(0u, host.ReadDword(signal + 8));
        Assert.Contains($"cp_write WRITE_DATA {signal + 0x10:X} 4", host.Calls);
    }

    private static uint[] WaitNotEqual(ulong address, uint reference) =>
        StreamRunner.Packet(PacketOpcode.WaitRegisterMemory, 0x14u, StreamRunner.Low(address), StreamRunner.High(address), reference, 0xFFFF_FFFFu, 0);

    /// <summary>
    /// The synthetic occlusion dump is sixteen qword stores 16 bytes apart. A host that takes
    /// them as a run gets one call; either way the host sees the same stores in the same
    /// order, another queue's wait on one of them is met by the recorded label at once, a
    /// wait on the bytes between them is not woken, and the labels are forgotten when their
    /// tick lands.
    /// </summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void AnOcclusionDumpIsRecordedInOrderAndAnotherQueuesWaitSeesItUntilItLands(bool batch)
    {
        var (host, queue) = NewQueue();
        host.DefersEndOfPipeLabels = true;
        host.DefersCommandProcessorWrites = true;
        host.PublishesSyntheticOcclusionDumps = true;
        host.RecordsQwordRuns = true;
        host.PendingLabelTick = 5;
        var dump = Label + 0x200;
        var seen = Label + 0x10;
        var untouched = Label + 0x18;
        host.WriteDword(dump + 48, 0);
        host.WriteDword(dump + 56, 0);
        host.WriteDword(seen, 0);
        host.WriteDword(untouched, 0);
        EnqueueCompute(host, queue, 0x20, Compute, 1, WaitNotEqual(dump + 48, 0), WriteLabel(seen, 7));
        Assert.Equal(SliceResult.BlockedWithoutProgress, queue.ProcessOne());
        EnqueueCompute(host, queue, 0x21, Compute + 0x1000, 2, WaitNotEqual(dump + 56, 0), WriteLabel(untouched, 9));
        Assert.Equal(SliceResult.BlockedWithoutProgress, queue.ProcessOne());
        Assert.Equal(2, queue.BlockedQueueCount);

        var interpreter = queue.GetInterpreter(0);
        interpreter.BatchCommandProcessorWrites = batch;
        interpreter.RaiseEvent(0x39, 1, dump);

        var stores = host.CommandProcessorWrites.Where(write => write.Kind == "ZPASS_DUMP").ToArray();
        Assert.Equal(Enumerable.Range(0, 16).Select(index => dump + (ulong)index * 16), stores.Select(write => write.Address));
        var value = BitConverter.ToUInt64(stores[0].Bytes);
        Assert.Equal(1UL << 63, value & (1UL << 63));
        Assert.All(stores, write => Assert.Equal(value, BitConverter.ToUInt64(write.Bytes)));
        Assert.Equal(batch ? 1 : 0, host.Calls.Count(call => call.StartsWith("cp_qwords ", StringComparison.Ordinal)));
        Assert.Equal(batch ? 0 : 16, host.Calls.Count(call => call.StartsWith("cp_write ZPASS_DUMP ", StringComparison.Ordinal)));

        // The wait on a stored qword is met by the recorded label; the one on the bytes
        // between two stores is not woken by them.
        Assert.Equal(1, queue.BlockedQueueCount);
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.Contains($"cp_write WRITE_DATA {seen:X} 4", host.Calls);
        Assert.DoesNotContain($"cp_write WRITE_DATA {untouched:X} 4", host.Calls);

        // Once that tick completes the labels are memory's business again.
        queue.NotifyCompletedGpuTick(5);
        EnqueueCompute(host, queue, 0x22, Compute + 0x2000, 3, WaitNotEqual(dump + 96, 0), WriteLabel(seen + 0x40, 3));
        Assert.Equal(SliceResult.BlockedWithoutProgress, queue.ProcessOne());
    }

    [Fact]
    public void AWaitMetByARecordedLabelDoesNotNeedTheBlockedHeadHintToBeRaised()
    {
        // Nothing is blocked when the label is recorded, so its wake is skipped; a wait that
        // starts afterwards still reads the label, which was published before it.
        var (host, queue) = NewQueue();
        host.DefersEndOfPipeLabels = true;
        host.PendingLabelTick = 5;
        host.WriteDword(Label, 0);
        Enqueue(host, queue, Graphics, 1, ReleaseMemoryInterrupt(Label, 1));
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.Equal(0, queue.BlockedQueueCount);

        EnqueueCompute(host, queue, 0x20, Compute, 2, WaitEqual(Label, 1), WriteLabel(Label + 8, 7));
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.Equal(7u, host.ReadDword(Label + 8));
        Assert.Equal(0, CommandStreamQueue.LabelWakeVerifyMismatches);
    }

    [Fact]
    public void MemoryWaitWakesOnlyForAWriteThatTouchesItsOwnBytes()
    {
        var (host, queue) = NewQueue();
        host.WriteDword(Label, 0);
        Enqueue(host, queue, Graphics, 1, WaitEqual(Label, 1), CreateInstanceCountPacket(9));
        Assert.Equal(SliceResult.BlockedWithoutProgress, queue.ProcessOne());
        Assert.Equal(1, queue.BlockedQueueCount);

        // Somewhere else entirely: the head has no reason to look again.
        queue.NotifyGuestMemoryWritten(Label + 0x1000, sizeof(uint));
        Assert.Equal(1, queue.BlockedQueueCount);
        Assert.Equal(SliceResult.AllBlocked, queue.ProcessOne());

        // Its own bytes: it looks again, and the value is still wrong, so it sleeps.
        queue.NotifyGuestMemoryWritten(Label, sizeof(uint));
        Assert.Equal(0, queue.BlockedQueueCount);
        Assert.Equal(SliceResult.BlockedWithoutProgress, queue.ProcessOne());

        host.WriteDword(Label, 1);
        queue.NotifyGuestMemoryWritten(Label, sizeof(uint));
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.Equal(9u, queue.GetInterpreter(0).InstanceCount);
    }

    /// <summary>
    /// A wake that arrives while the slice is running reaches a head that has not parked
    /// yet, so parking it afterwards would sleep until the retry timer. A command
    /// processor polling a label the GPU is about to write is exactly that race, and the
    /// timer is a hundred milliseconds against a submission that finishes in one.
    /// </summary>
    [Fact]
    public void AWakeDuringTheSliceLeavesTheHeadRunnable()
    {
        var (host, queue) = NewQueue();
        host.WriteDword(Label, 0);
        Enqueue(host, queue, Graphics, 1, WaitEqual(Label, 1), CreateInstanceCountPacket(9));
        host.BeforeGuestRead = address =>
        {
            if (address != Label)
            {
                return;
            }

            // The submission completes between the poll and the park.
            host.BeforeGuestRead = null;
            queue.NotifyCompletedGpuTick(1);
        };

        Assert.Equal(SliceResult.BlockedWithoutProgress, queue.ProcessOne());
        Assert.Equal(0, queue.BlockedQueueCount);
        Assert.Equal(1ul, queue.WakesRacedWithSlice);

        // No further event is needed to look again; the value is still wrong, and with
        // nothing racing this time the head does park.
        Assert.Equal(SliceResult.BlockedWithoutProgress, queue.ProcessOne());
        Assert.Equal(1, queue.BlockedQueueCount);

        host.WriteDword(Label, 1);
        queue.NotifyCompletedGpuTick(2);
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.Equal(9u, queue.GetInterpreter(0).InstanceCount);
    }

    [Fact]
    public void CompletedGpuTick_RetriesBlockedWaitWithoutTheTimeout()
    {
        var (host, queue) = NewQueue();
        host.WriteDword(Label, 0);
        Enqueue(host, queue, Graphics, 1, WaitEqual(Label, 1), CreateInstanceCountPacket(9));
        Assert.Equal(SliceResult.BlockedWithoutProgress, queue.ProcessOne());

        queue.NotifyCompletedGpuTick(1);
        Assert.Equal(SliceResult.BlockedWithoutProgress, queue.ProcessOne());
        host.WriteDword(Label, 1);
        queue.NotifyCompletedGpuTick(1);
        Assert.Equal(SliceResult.AllBlocked, queue.ProcessOne());

        queue.NotifyCompletedGpuTick(2);
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.Equal(9u, queue.GetInterpreter(0).InstanceCount);
    }

    [Fact]
    public void PresentationProgress_RetriesBlockedFlipWait()
    {
        var (host, queue) = NewQueue();
        host.FlipDone = false;
        Enqueue(host, queue, Graphics, 1,
            StreamRunner.CustomPacket(PacketOpcode.Nop, PacketCustomCode.WaitFlipDone, 1, 0, 0, 0, 0, 0));
        Assert.Equal(SliceResult.BlockedWithoutProgress, queue.ProcessOne());
        host.FlipDone = true;
        queue.RetryBlocked();
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
    }

    [Fact]
    public async Task WaitForIdle_IsNotReleasedByStopAccepting()
    {
        var (host, queue) = NewQueue();
        host.WriteDword(Label, 0);
        Enqueue(host, queue, Graphics, 1, WaitEqual(Label, 1));
        Assert.Equal(SliceResult.BlockedWithoutProgress, queue.ProcessOne());
        var waiter = Task.Run(queue.WaitForIdle);

        queue.StopAccepting();
        await Task.WhenAny(waiter, Task.Delay(200));
        Assert.False(waiter.IsCompleted);
        Assert.Throws<OperationCanceledException>(() => queue.EnqueueGraphics(Graphics, 1, 2, null));
        Assert.Throws<OperationCanceledException>(() => queue.EnqueueCompute(GpuCommandInterpreter.ComputeQueueBase, Compute, 1, 3, null));

        host.WriteDword(Label, 1);
        queue.RetryBlocked();
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.Equal(IdleOutcome.Completed, await waiter.WaitAsync(TimeSpan.FromSeconds(10)));
    }

    [Fact]
    public void DrainForShutdown_WaitsTheRetryIntervalBeforeCancelling()
    {
        var (host, queue) = NewQueue();
        host.WriteDword(Label, 0);
        Enqueue(host, queue, Graphics, 1, WaitEqual(Label, 1), CreateInstanceCountPacket(9));
        queue.StopAccepting();
        var outcome = IdleOutcome.Cancelled;
        var drainThread = new Thread(() => outcome = queue.DrainForShutdown(cancelBlockedOnNoProgress: true));
        drainThread.Start();
        try
        {
            Assert.True(SpinWait.SpinUntil(
                () => (drainThread.ThreadState & ThreadState.WaitSleepJoin) != 0,
                TimeSpan.FromSeconds(10)));
            host.WriteDword(Label, 1);
        }
        finally
        {
            Assert.True(drainThread.Join(TimeSpan.FromSeconds(10)));
        }

        Assert.Equal(IdleOutcome.Completed, outcome);
        Assert.Equal(9u, queue.GetInterpreter(0).InstanceCount);
    }

    [Fact]
    public void DrainForShutdown_CancelsOnlyHeadsThatMakeNoProgressAndTheOutcomeSticks()
    {
        var (host, queue) = NewQueue();
        host.WriteDword(Label, 0);
        Enqueue(host, queue, Graphics, 1, WaitEqual(Label, 1), CreateInstanceCountPacket(9));
        Enqueue(host, queue, Graphics + 0x100, 2, CreateInstanceCountPacket(4));
        EnqueueCompute(host, queue, 0x20, Compute, 3, CreateInstanceCountPacket(5));
        queue.StopAccepting();

        var outcome = queue.DrainForShutdown(cancelBlockedOnNoProgress: true);

        Assert.Equal(IdleOutcome.Cancelled, outcome);
        Assert.Equal(5u, queue.GetInterpreter(1).InstanceCount);
        Assert.Equal(1u, queue.GetInterpreter(0).InstanceCount);
        Assert.False(queue.HasPending);
        Assert.Equal(IdleOutcome.Cancelled, queue.WaitForIdle());
        Assert.Equal(IdleOutcome.Cancelled, queue.Outcome);
    }

    [Fact]
    public async Task DrainForShutdown_WithoutCancellationKeepsRetrying()
    {
        var (host, queue) = NewQueue();
        host.WriteDword(Label, 0);
        Enqueue(host, queue, Graphics, 1, WaitEqual(Label, 1), CreateInstanceCountPacket(9));
        queue.StopAccepting();
        var drain = Task.Run(() => queue.DrainForShutdown(cancelBlockedOnNoProgress: false));

        await Task.WhenAny(drain, Task.Delay(350));
        Assert.False(drain.IsCompleted);
        host.WriteDword(Label, 1);
        Assert.Equal(IdleOutcome.Completed, await drain.WaitAsync(TimeSpan.FromSeconds(10)));
        Assert.Equal(9u, queue.GetInterpreter(0).InstanceCount);
    }

    [Fact]
    public async Task HostFailure_MarksTheQueueFailedAndReleasesWaiters()
    {
        var (host, queue) = NewQueue();
        host.WriteWords(Graphics, StreamRunner.Packet(0x41, 0));
        queue.EnqueueGraphics(Graphics, 2, 1, null);
        var waiter = Task.Run(queue.WaitForIdle);

        Assert.Throws<CommandStreamFatalException>(() => queue.ProcessOne());

        Assert.Equal(IdleOutcome.Failed, await waiter.WaitAsync(TimeSpan.FromSeconds(10)));
        Assert.False(queue.HasPending);
        Assert.Contains("no longer accepts", Assert.Throws<CommandStreamFatalException>(() => queue.EnqueueGraphics(Graphics, 2, 2, null)).Message);
    }

    [Fact]
    public async Task ErrorWriterFailure_PreservesTheHostExceptionAndReleasesWaiters()
    {
        var (host, queue) = NewQueue();
        var hostException = new CommandStreamFatalException("The command stream host failed.");
        host.PendingCommands.Enqueue(() => throw hostException);
        Enqueue(host, queue, Graphics, 1, CreateInstanceCountPacket(1));
        Enqueue(host, queue, Graphics + 0x100, 2, CreateInstanceCountPacket(2));
        var waiter = Task.Run(queue.WaitForIdle);
        var previousErrorWriter = Console.Error;
        using var failingErrorWriter = new FailingErrorWriter();

        try
        {
            Exception? actualException;
            try
            {
                Console.SetError(failingErrorWriter);
                actualException = Record.Exception(() => queue.ProcessOne());
            }
            finally
            {
                Console.SetError(previousErrorWriter);
            }

            Assert.True(failingErrorWriter.WriteAttempted);
            Assert.Same(hostException, actualException);
            Assert.Equal(IdleOutcome.Failed, await waiter.WaitAsync(TimeSpan.FromSeconds(10)));
            Assert.False(queue.HasPending);
            Assert.Contains("no longer accepts", Assert.Throws<CommandStreamFatalException>(() =>
                queue.EnqueueGraphics(Graphics, 2, 3, null)).Message);
        }
        finally
        {
            queue.Fail();
            await waiter.WaitAsync(TimeSpan.FromSeconds(10));
        }
    }

    private sealed class FailingErrorWriter : StringWriter
    {
        public bool WriteAttempted { get; private set; }

        public override void WriteLine(string? value)
        {
            WriteAttempted = true;
            throw new IOException("The error output is unavailable.");
        }
    }

    [Fact]
    public async Task Done_QueuesTheBoundaryAndOnlyWaitsForASecondOne()
    {
        var host = new RecordingCommandStreamHost();
        var queue = new CommandStreamQueue(host, boundariesInFlight: 1);
        Enqueue(host, queue, Graphics, 1, CreateInstanceCountPacket(1));

        // The first suspend point returns without draining the accepted submissions.
        Assert.Equal(IdleOutcome.Completed, await Task.Run(queue.Done).WaitAsync(TimeSpan.FromSeconds(10)));
        Assert.Equal(0, queue.FrameNumber);
        Assert.True(queue.HasPending);

        // The second one waits until the first boundary has been processed.
        var second = Task.Run(queue.Done);
        await Task.WhenAny(second, Task.Delay(100));
        Assert.False(second.IsCompleted);
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.Equal(IdleOutcome.Completed, await second.WaitAsync(TimeSpan.FromSeconds(10)));
        Assert.Equal(1, queue.FrameNumber);

        // The next graphics submission starts from a reset processor.
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.Equal(2, queue.FrameNumber);
        Enqueue(host, queue, Graphics, 2, CreateInstanceCountPacket(2));
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.Equal(2UL, queue.GetInterpreter(0).SubmitId);
    }

    [Fact]
    public async Task Done_WaitsForItsOwnBoundaryWithoutRunAhead()
    {
        var host = new RecordingCommandStreamHost();
        var queue = new CommandStreamQueue(host, boundariesInFlight: 0);
        Enqueue(host, queue, Graphics, 1, CreateInstanceCountPacket(1));

        var done = Task.Run(queue.Done);
        await Task.WhenAny(done, Task.Delay(100));
        Assert.False(done.IsCompleted);

        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        await Task.WhenAny(done, Task.Delay(100));
        Assert.False(done.IsCompleted);

        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.Equal(IdleOutcome.Completed, await done.WaitAsync(TimeSpan.FromSeconds(10)));
        Assert.Equal(1, queue.FrameNumber);
        Assert.False(queue.HasPending);
    }

    [Fact]
    public async Task Done_KeepsALabelRewrittenAfterItAwayFromAnUnfinishedFrame()
    {
        var host = new RecordingCommandStreamHost();
        var queue = new CommandStreamQueue(host, boundariesInFlight: 0);
        void WriteLabel(uint value) => Assert.True(host.GuestMemory.TryWrite(StreamRunner.DataAddress, BitConverter.GetBytes(value)));
        WriteLabel(0);
        Enqueue(host, queue, Graphics, 1,
            StreamRunner.Packet(PacketOpcode.WriteData, 0x00000500u, StreamRunner.Low(StreamRunner.DataAddress), StreamRunner.High(StreamRunner.DataAddress), 1),
            WaitEqual(StreamRunner.DataAddress, 1));

        var guest = Task.Run(() =>
        {
            var outcome = queue.Done();
            WriteLabel(0);
            return outcome;
        });

        await Task.WhenAny(guest, Task.Delay(100));
        Assert.False(guest.IsCompleted);
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.Equal(IdleOutcome.Completed, await guest.WaitAsync(TimeSpan.FromSeconds(10)));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task PairedLabelWait_SurvivesTheNextFramesResetWithRunAhead(bool is64Bit)
    {
        var host = new RecordingCommandStreamHost();
        var queue = new CommandStreamQueue(host, boundariesInFlight: 1);
        var packets = new List<uint[]>();
        for (var pair = 0; pair < 7; pair++)
        {
            var address = Label + (ulong)pair * 8;
            packets.Add(is64Bit
                ? StreamRunner.Packet(PacketOpcode.WriteData, 0x00000500u,
                    StreamRunner.Low(address), StreamRunner.High(address), 1, 2)
                : WriteLabel(address, 1));
            packets.Add(is64Bit
                ? StreamRunner.Packet(PacketOpcode.WaitRegisterMemory64, 0x13u,
                    StreamRunner.Low(address), StreamRunner.High(address), 1, 2, uint.MaxValue, uint.MaxValue, 0)
                : WaitEqual(address, 1));
        }
        Enqueue(host, queue, Graphics, 1, packets.ToArray());
        host.BeforeGuestRead = address =>
        {
            if (address == Label + 6 * 8) host.WriteQword(address, 0);
        };

        Assert.Equal(IdleOutcome.Completed, await Task.Run(queue.Done).WaitAsync(TimeSpan.FromSeconds(10)));
        Assert.Equal(0, queue.FrameNumber); // CPU/GPU overlap remains enabled.
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.Equal(0UL, host.ReadQword(Label + 6 * 8)); // The CPU reset was not overwritten.
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
        Assert.Equal(1, queue.FrameNumber);
        Assert.Equal(0, queue.BlockedQueueCount);
    }

    [Theory]
    [InlineData(0)] // A CPU-produced label must still be read from live memory.
    [InlineData(1)] // Never forward a value from the preceding submission.
    [InlineData(2)] // Never forward through work that may write the label on the GPU.
    public void LabelWait_DoesNotForwardAnUnrelatedOrOlderStore(int scenario)
    {
        var (host, queue) = NewQueue();
        host.WriteDword(Label, 0);
        if (scenario == 0)
        {
            Enqueue(host, queue, Graphics, 1, WriteLabel(Label + 8, 1), WaitEqual(Label, 1));
        }
        else if (scenario == 1)
        {
            Enqueue(host, queue, Graphics, 1, WriteLabel(Label, 1));
            Assert.Equal(SliceResult.Completed, queue.ProcessOne());
            host.WriteDword(Label, 0);
            Enqueue(host, queue, Graphics + 0x100, 2, WaitEqual(Label, 1));
        }
        else
        {
            Enqueue(host, queue, Graphics, 1, WriteLabel(Label, 1),
                StreamRunner.Packet(PacketOpcode.DrawIndexAuto, 3, 0), WaitEqual(Label, 1));
            host.BeforeGuestRead = address =>
            {
                if (address == Label) host.WriteDword(Label, 0);
            };
        }
        Assert.Equal(scenario == 1 ? SliceResult.BlockedWithoutProgress : SliceResult.Progressed, queue.ProcessOne());
        host.BeforeGuestRead = null;
        host.WriteDword(Label, 1);
        queue.RetryBlocked();
        Assert.Equal(SliceResult.Completed, queue.ProcessOne());
    }
}
