// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using SharpEmu.HLE;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.Scheduling;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.GpuCommands;

/// <summary>
/// SHARPEMU_FRAME_OVERLAP_PROBE's publication rules on the managed host, with a fake GPU timeline
/// standing in for the Metal completion queue: a buffer's publications land at its commit, in
/// stream order, while no flip but the newest committed one is still on the GPU; the bound holds
/// the rest until that flip completes; a refused buffer and everything behind it wait for their
/// real completion; real-completion-only work never runs early.
/// </summary>
public sealed class FrameOverlapProbeTests
{
    // Runs work once its serial is signalled: in serial order, and in queue order among equals.
    private sealed class FakeGpu
    {
        private readonly List<(ulong Serial, Action Work)> _pending = new();

        public ulong Signaled { get; private set; }

        public void Enqueue(ulong serial, Action work) => _pending.Add((serial, work));

        public void Signal(ulong value)
        {
            Signaled = Math.Max(Signaled, value);
            Pump();
        }

        // The completion thread waking: everything already signalled runs.
        public void Pump()
        {
            while (true)
            {
                var index = -1;
                for (var candidate = 0; candidate < _pending.Count; candidate++)
                {
                    if (_pending[candidate].Serial <= Signaled &&
                        (index < 0 || _pending[candidate].Serial < _pending[index].Serial))
                    {
                        index = candidate;
                    }
                }

                if (index < 0)
                {
                    return;
                }

                var work = _pending[index].Work;
                _pending.RemoveAt(index);
                work();
            }
        }
    }

    private sealed class Log : IEndOfPipeSink, ICommandStreamFlipTarget
    {
        public List<string> Entries { get; } = new();

        public void TriggerInterrupt(int eventId, uint contextId) => Entries.Add($"interrupt {eventId}");

        public ulong Prepare(int handle, int index, int flipMode, long flipArgument) => 0;

        public void Complete(ulong requestId) => Entries.Add($"flip {requestId}");

        public void Presented(ulong requestId)
        {
        }

        public bool IsDone(int handle, int index) => true;
    }

    private sealed class NullMemory : ICpuMemory
    {
        public bool TryRead(ulong virtualAddress, Span<byte> destination) => true;

        public bool TryWrite(ulong virtualAddress, ReadOnlySpan<byte> source) => true;
    }

    // The managed host, with its end-of-pipe work queued for a GPU completion through the probe
    // the way the Metal host queues it, and serials advancing at every commit.
    private sealed class ProbeHost : ManagedCommandStreamHost
    {
        public ProbeHost(FakeGpu gpu, Log log, Func<long>? clock = null, FrameOverlapProbe.ArmTrigger? trigger = null)
            : base(new NullMemory(), log, log)
        {
            Gpu = gpu;
            Probe = new FrameOverlapProbe(gpu.Enqueue, () => gpu.Signaled, clock, trigger);
        }

        public FakeGpu Gpu { get; }

        public FrameOverlapProbe Probe { get; }

        public ulong OpenSerial { get; private set; } = 1;

        public override void RecordEndOfPipe(in EndOfPipeWrite write)
        {
            var copy = write;
            Gpu.Enqueue(OpenSerial, Probe.Track(() => base.RecordEndOfPipe(copy), address: copy.Destination,
                bytes: copy.Destination == 0 ? 0UL : sizeof(uint)));
        }

        // A flip whose own completion stays with the GPU, as the Metal host records it under the probe.
        public void RecordFlip(ulong requestId, int eventId)
        {
            Probe.NoteFlip();
            Gpu.Enqueue(OpenSerial, Probe.Track(
                () => TriggerInterrupt(eventId, 0),
                atRealCompletion: () => base.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Flip, 0, FlipRequestId: requestId))));
        }

        public void Interrupt(int eventId, ulong destination = 0) =>
            RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.InterruptOnly, 0, Destination: destination, EventId: eventId));

        public override void Flush() => Commit();

        // A command buffer is committed with the open serial.
        public void Commit()
        {
            var serial = OpenSerial++;
            Probe.OnCommit(serial);
        }
    }

    private readonly FakeGpu _gpu = new();
    private readonly Log _log = new();

    private static string Counter(string line, string name)
    {
        var start = line.IndexOf(" " + name + "=", StringComparison.Ordinal);
        Assert.True(start >= 0, $"{name} missing from: {line}");
        start += name.Length + 2;
        var end = line.IndexOf(' ', start);
        return end < 0 ? line[start..] : line[start..end];
    }

    // One frame: `draws` draws spread over two buffers, the second carrying the flip, each completed at once.
    private static void Frame(ProbeHost host, int draws, ref ulong flipId)
    {
        for (var draw = 0; draw < draws; draw++)
        {
            host.Probe.NoteDraw();
            if (draw == draws / 2)
            {
                host.Interrupt(1);
                host.Commit();
            }
        }

        host.RecordFlip(++flipId, 2);
        host.Commit();
        host.Gpu.Pump();
        host.Gpu.Signal(host.OpenSerial - 1);
    }

    [Fact]
    public void TheDrawTriggerArmsAfterEnoughHeavyFramesInARowAndThenTheSettleDelay()
    {
        long now = 0;
        var host = new ProbeHost(_gpu, _log, () => now, new FrameOverlapProbe.ArmTrigger(Draws: 10, Flips: 3, SettleSeconds: 1));
        ulong flipId = 0;

        // Two heavy frames, a light one (menu or loading) resets the run, then two heavy again: not armed.
        Frame(host, 12, ref flipId);
        Frame(host, 12, ref flipId);
        Frame(host, 4, ref flipId);
        Frame(host, 12, ref flipId);
        Frame(host, 12, ref flipId);
        var line = host.Probe.TakeWindowLine();
        Assert.Equal("0", Counter(line, "published_early"));

        // The third heavy frame in a row arms it; within the settle delay nothing lands early yet.
        Frame(host, 12, ref flipId);
        now += Stopwatch.Frequency / 2;
        Frame(host, 12, ref flipId);
        line = host.Probe.TakeWindowLine();
        Assert.Equal("0", Counter(line, "published_early"));

        // Past the delay buffers land early again, and a later light frame does not disarm it.
        now += Stopwatch.Frequency;
        Frame(host, 4, ref flipId);
        Frame(host, 4, ref flipId);
        line = host.Probe.TakeWindowLine();
        Assert.NotEqual("0", Counter(line, "published_early"));
    }

    [Fact]
    public void ABufferLandsAtItsCommitWhileNoEarlierFlipIsOutstanding()
    {
        var host = new ProbeHost(_gpu, _log);
        host.Interrupt(1);
        host.Commit();
        _gpu.Pump();
        Assert.Equal(["interrupt 1"], _log.Entries);

        host.Interrupt(2);
        host.Commit();
        _gpu.Pump();
        Assert.Equal(["interrupt 1", "interrupt 2"], _log.Entries);

        // The real completions find them published and do not publish them again.
        _gpu.Signal(2);
        Assert.Equal(["interrupt 1", "interrupt 2"], _log.Entries);
        var line = host.Probe.TakeWindowLine();
        Assert.Equal("2", Counter(line, "at_commit"));
        Assert.Equal("2", Counter(line, "published_early"));
        Assert.Equal("0", Counter(line, "held_by_bound"));
    }

    [Fact]
    public void TheBoundHoldsWorkWhileTheFlipBeforeTheNewestIsStillOnTheGpu()
    {
        var host = new ProbeHost(_gpu, _log);
        host.RecordFlip(requestId: 7, eventId: 10);
        host.Commit();                   // serial 1: flip 1
        host.Interrupt(1);
        host.Commit();                   // serial 2: after flip 1, nothing older outstanding
        _gpu.Pump();
        Assert.Equal(["interrupt 10", "interrupt 1"], _log.Entries);

        host.RecordFlip(requestId: 8, eventId: 11);
        host.Commit();                   // serial 3: flip 2, while flip 1 still runs
        host.Interrupt(2);
        host.Commit();                   // serial 4: after flip 2
        _gpu.Pump();

        // Flip 1 is still on the GPU: what follows flip 2 waits for it.
        Assert.Equal(["interrupt 10", "interrupt 1"], _log.Entries);

        // Flip 1 completes: its own real-completion work, then the held buffers in stream order.
        _gpu.Signal(1);
        Assert.Equal(["interrupt 10", "interrupt 1", "flip 7", "interrupt 11", "interrupt 2"], _log.Entries);
        var line = host.Probe.TakeWindowLine();
        Assert.Equal("2", Counter(line, "held_by_bound"));
        Assert.Equal("4", Counter(line, "published_early"));
        Assert.Equal("1", Counter(line, "flips_outstanding_max"));

        _gpu.Signal(4);
        Assert.Equal(["interrupt 10", "interrupt 1", "flip 7", "interrupt 11", "interrupt 2", "flip 8"], _log.Entries);
    }

    [Fact]
    public void ARefusedBufferAndTheBuffersBehindItPublishAtTheirRealCompletionInStreamOrder()
    {
        var host = new ProbeHost(_gpu, _log);
        host.Interrupt(1);
        host.Probe.Refuse(FrameOverlapRefusal.GdsStore, 0x1000);
        host.Interrupt(2);
        host.Commit();
        host.Interrupt(3);
        host.Commit();
        _gpu.Pump();
        Assert.Empty(_log.Entries);

        _gpu.Signal(1);
        Assert.Equal(["interrupt 1", "interrupt 2"], _log.Entries);
        _gpu.Signal(2);
        Assert.Equal(["interrupt 1", "interrupt 2", "interrupt 3"], _log.Entries);
        var line = host.Probe.TakeWindowLine();
        Assert.Equal("1", Counter(line, "refused"));
        Assert.Equal("1", Counter(line, "behind_refusal"));
        Assert.Equal("1", Counter(line, "gds_store"));
    }

    [Fact]
    public void RealCompletionWorkNeverRunsEarly()
    {
        var host = new ProbeHost(_gpu, _log);
        host.Interrupt(1);
        host.RecordFlip(requestId: 7, eventId: 2);
        host.Commit();
        _gpu.Pump();

        // The flip's interrupt goes out with the buffer; the flip's GPU completion does not.
        Assert.Equal(["interrupt 1", "interrupt 2"], _log.Entries);
        _gpu.Signal(1);
        Assert.Equal(["interrupt 1", "interrupt 2", "flip 7"], _log.Entries);
        var line = host.Probe.TakeWindowLine();
        Assert.Equal("1", Counter(line, "flips_early"));
    }

    [Fact]
    public void AParkOnBytesACompletionStillOwesRefusesTheOpenBuffer()
    {
        var host = new ProbeHost(_gpu, _log);
        host.Interrupt(1, destination: 0x2000);
        host.Commit();
        host.Interrupt(2, destination: 0x2100);
        host.Probe.NoteSuspendedWalk(CommandWaitCondition.Memory(0x2000, 4));
        host.Probe.NoteSuspendedWalk(CommandWaitCondition.Memory(0x3000, 4));
        host.Commit();
        _gpu.Pump();
        Assert.Equal(["interrupt 1"], _log.Entries);

        _gpu.Signal(2);
        Assert.Equal(["interrupt 1", "interrupt 2"], _log.Entries);
        var line = host.Probe.TakeWindowLine();
        Assert.Equal("1", Counter(line, "suspended_walk"));
        Assert.Equal("1", Counter(line, "suspended_other"));
        Assert.Equal("1", Counter(line, "refused"));
    }

    // The probe's queue side: a pending label noted with no completion tick (the Metal host's tick under
    // the probe) answers waits and parse-time reads until its publication lands it, not when a tick passes.
    [Fact]
    public void AQueuedPublicationAnswersWaitsUntilItLandsNotWhenATickPasses()
    {
        const ulong label = StreamRunner.LabelAddress;
        var host = new RecordingCommandStreamHost { PendingLabelTick = ulong.MaxValue };
        var queue = new CommandStreamQueue(host);
        var interpreter = queue.GetInterpreter(0);
        interpreter.PendingLabelRecorded!(label, 4, 7);

        // Every submission completes on the GPU: the label is still only queued, and still answers.
        queue.NotifyCompletedGpuTick(1_000_000);
        Assert.True(interpreter.PendingLabelLookup!(label, false, out var queued));
        Assert.Equal(7UL, queued);

        // Its publication has not written memory yet: landing keeps it.
        queue.LandPendingLabels(label, 4, host.Memory);
        Assert.Equal(1, queue.PendingLabelCount);

        // Published: memory holds the value, and memory answers from now on.
        Assert.True(host.Memory.TryWrite(label, BitConverter.GetBytes(7u)));
        queue.LandPendingLabels(label, 4, host.Memory);
        Assert.Equal(0, queue.PendingLabelCount);
        Assert.False(interpreter.PendingLabelLookup!(label, false, out _));
    }

    [Fact]
    public void LandingAnOlderPublicationKeepsTheNewerQueuedValue()
    {
        const ulong label = StreamRunner.LabelAddress;
        var host = new RecordingCommandStreamHost { PendingLabelTick = ulong.MaxValue };
        var queue = new CommandStreamQueue(host);
        var interpreter = queue.GetInterpreter(0);
        interpreter.PendingLabelRecorded!(label, 8, 1);
        interpreter.PendingLabelRecorded!(label, 8, 2);

        // The first publication wrote 1; the stream's newest write (2) is still queued.
        Assert.True(host.Memory.TryWrite(label, BitConverter.GetBytes(1UL)));
        queue.LandPendingLabels(label, 8, host.Memory);
        Assert.True(interpreter.PendingLabelLookup!(label, true, out var newest));
        Assert.Equal(2UL, newest);

        Assert.True(host.Memory.TryWrite(label, BitConverter.GetBytes(2UL)));
        queue.LandPendingLabels(label, 8, host.Memory);
        Assert.False(interpreter.PendingLabelLookup!(label, true, out _));
    }

    [Fact]
    public void QueuedBytesStayOwedUntilTheyArePublished()
    {
        var host = new ProbeHost(_gpu, _log);
        host.Interrupt(1, destination: 0x2000);
        host.Probe.Refuse(FrameOverlapRefusal.GdsStore);
        host.Commit();
        host.Interrupt(2, destination: 0x3000);
        host.Commit();
        _gpu.Pump();

        // A copy reading either range must still be queued behind the write.
        Assert.True(host.Probe.Owes(0x2000, 4));
        Assert.True(host.Probe.Owes(0x3000, 4));
        _gpu.Signal(1);
        Assert.False(host.Probe.Owes(0x2000, 4));
        Assert.True(host.Probe.Owes(0x3000, 4));
        _gpu.Signal(2);
        Assert.False(host.Probe.Owes(0x3000, 4));
        Assert.Equal(["interrupt 1", "interrupt 2"], _log.Entries);
    }
}
