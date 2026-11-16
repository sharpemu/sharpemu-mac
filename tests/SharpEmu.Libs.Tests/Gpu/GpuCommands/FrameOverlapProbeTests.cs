// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.Scheduling;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.GpuCommands;

/// <summary>
/// SHARPEMU_FRAME_OVERLAP_PROBE's publication rules on the managed host, with a fake GPU timeline
/// standing in for the Metal completion queue: a frame publishes at its commit once the frame
/// before it really completed, in stream order behind anything older; never two frames ahead;
/// a refused frame waits for its real completion; real-completion-only work never runs early.
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
        public ProbeHost(FakeGpu gpu, Log log) : base(new NullMemory(), log, log)
        {
            Gpu = gpu;
            Probe = new FrameOverlapProbe(gpu.Enqueue, () => gpu.Signaled);
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

        public override void Flush() => OpenSerial++;

        // FlushFrame: the frame's last buffer is committed with the open serial.
        public void CommitFrame()
        {
            var last = OpenSerial++;
            Probe.OnFrameCommitted(last);
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

    [Fact]
    public void AFramePublishesAtItsCommitOnceThePreviousFrameIsDone()
    {
        var host = new ProbeHost(_gpu, _log);
        host.Interrupt(1);
        host.Flush();
        host.Interrupt(2);
        host.CommitFrame();
        _gpu.Pump();

        Assert.Equal(["interrupt 1", "interrupt 2"], _log.Entries);

        // The real completion finds them published and does not publish them again.
        _gpu.Signal(2);
        Assert.Equal(["interrupt 1", "interrupt 2"], _log.Entries);
        var line = host.Probe.TakeWindowLine();
        Assert.Equal("1", Counter(line, "published_early"));
        Assert.Equal("2", Counter(line, "items_early"));
        Assert.Equal("1", Counter(line, "depth_max"));
    }

    [Fact]
    public void TheNextFrameWaitsForThePublishedFrameToReallyComplete()
    {
        var host = new ProbeHost(_gpu, _log);
        host.Interrupt(1);
        host.CommitFrame();
        _gpu.Pump();
        host.Interrupt(2);
        host.CommitFrame();
        _gpu.Pump();

        // Frame 0 is published and still on the GPU: frame 1 is not published, the guest is one ahead.
        Assert.Equal(["interrupt 1"], _log.Entries);

        // Frame 0 completes after frame 1's commit: frame 1 publishes at that completion, still early.
        _gpu.Signal(1);
        Assert.Equal(["interrupt 1", "interrupt 2"], _log.Entries);
        var line = host.Probe.TakeWindowLine();
        Assert.Equal("2", Counter(line, "published_early"));
        Assert.Equal("1", Counter(line, "prev_running_at_commit"));
        Assert.Equal("1", Counter(line, "depth_max"));

        _gpu.Signal(2);
        Assert.Equal(["interrupt 1", "interrupt 2"], _log.Entries);
    }

    [Fact]
    public void ARefusedFramePublishesAtItsRealCompletionAndStreamOrderHolds()
    {
        var host = new ProbeHost(_gpu, _log);
        host.Interrupt(1);
        host.Probe.Refuse(FrameOverlapRefusal.GdsStore, 0x1000);
        host.Interrupt(2);
        host.CommitFrame();
        _gpu.Pump();
        Assert.Empty(_log.Entries);

        host.Interrupt(3);
        host.Interrupt(4);
        host.CommitFrame();
        _gpu.Pump();
        Assert.Empty(_log.Entries);

        // Frame 0 completes: its own completions run, then frame 1 publishes behind them.
        _gpu.Signal(1);
        Assert.Equal(["interrupt 1", "interrupt 2", "interrupt 3", "interrupt 4"], _log.Entries);
        var line = host.Probe.TakeWindowLine();
        Assert.Equal("1", Counter(line, "refused_frames"));
        Assert.Equal("1", Counter(line, "published_early"));
        Assert.Equal("1", Counter(line, "gds_store"));
    }

    [Fact]
    public void RealCompletionWorkNeverRunsEarly()
    {
        var host = new ProbeHost(_gpu, _log);
        host.Interrupt(1);
        host.RecordFlip(requestId: 7, eventId: 2);
        host.CommitFrame();
        _gpu.Pump();

        // The flip's interrupt goes out with the frame; the flip's GPU completion does not.
        Assert.Equal(["interrupt 1", "interrupt 2"], _log.Entries);
        _gpu.Signal(1);
        Assert.Equal(["interrupt 1", "interrupt 2", "flip 7"], _log.Entries);
        var line = host.Probe.TakeWindowLine();
        Assert.Equal("1", Counter(line, "flips_early"));
    }

    [Fact]
    public void AParkOnBytesACompletionStillOwesRefusesTheFrame()
    {
        var host = new ProbeHost(_gpu, _log);
        host.Interrupt(1, destination: 0x2000);
        host.Probe.NoteSuspendedWalk(CommandWaitCondition.Memory(0x3000, 4));
        host.CommitFrame();
        _gpu.Pump();
        Assert.Equal(["interrupt 1"], _log.Entries);

        host.Interrupt(2, destination: 0x2100);
        host.Probe.NoteSuspendedWalk(CommandWaitCondition.Memory(0x2100, 4));
        host.CommitFrame();
        _gpu.Signal(1);
        Assert.Empty(_log.Entries.Skip(1));

        _gpu.Signal(2);
        Assert.Equal(["interrupt 1", "interrupt 2"], _log.Entries);
        var line = host.Probe.TakeWindowLine();
        Assert.Equal("1", Counter(line, "suspended_walk"));
        Assert.Equal("1", Counter(line, "suspended_other"));
        Assert.Equal("1", Counter(line, "refused_frames"));
    }
}
