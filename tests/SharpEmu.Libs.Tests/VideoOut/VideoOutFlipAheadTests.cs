// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Diagnostics;
using SharpEmu.HLE;
using SharpEmu.Libs.Kernel;
using SharpEmu.Libs.Tests.Gpu.Scheduling;
using SharpEmu.Libs.VideoOut;
using Xunit;

namespace SharpEmu.Libs.Tests.VideoOut;

// SHARPEMU_FLIP_AHEAD: a flip the GPU host recorded (NoteFlipRecorded) is published to the guest
// once the flip before it really completed on the GPU; the request stays for the pacer and its
// real completion releases the GPU slot without publishing again.
[Collection(SchedulingStateCollection.Name)]
public sealed class VideoOutFlipAheadTests : IDisposable
{
    private const ulong MemoryBase = 0x1_0000_0000;
    private const ulong StatusAddress = MemoryBase + 0x100;
    private const ulong ScratchAddress = MemoryBase + 0x200;
    private const ulong EventAddress = MemoryBase + 0x300;
    private const ulong OutCountAddress = MemoryBase + 0x400;
    private const ulong TimeoutAddress = MemoryBase + 0x480;
    private const int TimedOut = unchecked((int)0x8002003C);

    private readonly FakeCpuMemory _memory = new(MemoryBase, 0x1000);
    private readonly CpuContext _context;
    private readonly int _handle;
    private ulong _queue;

    public VideoOutFlipAheadTests()
    {
        _context = new CpuContext(_memory, Generation.Gen5);
        _context[CpuRegister.Rdi] = 0;
        _context[CpuRegister.Rsi] = 0;
        _context[CpuRegister.Rdx] = 0;
        _context[CpuRegister.Rcx] = 0;
        _handle = VideoOutExports.VideoOutOpen(_context);
        Assert.True(_handle > 0);
        VideoOutExports.ResetFlipAheadForTests(enabled: true);
    }

    public void Dispose()
    {
        if (_queue != 0)
        {
            _context[CpuRegister.Rdi] = _queue;
            _ = KernelEventQueueCompatExports.KernelDeleteEqueue(_context);
        }

        _context[CpuRegister.Rdi] = unchecked((ulong)_handle);
        _ = VideoOutExports.VideoOutClose(_context);
        VideoOutExports.ResetFlipAheadForTests(enabled: false);
    }

    private ulong Reserve(long flipArg, bool gpuQueued = true)
    {
        Assert.Equal(0, VideoOutExports.TryReserveFlipRequest(_handle, -1, 2, flipArg, gpuQueued, out var requestId));
        return requestId;
    }

    private (ulong Count, long FlipArg, int GpuQueue, int Pending) Status()
    {
        _context[CpuRegister.Rdi] = unchecked((ulong)_handle);
        _context[CpuRegister.Rsi] = StatusAddress;
        Assert.Equal(0, VideoOutExports.VideoOutGetFlipStatus(_context));
        Span<byte> status = stackalloc byte[0x40];
        Assert.True(_memory.TryRead(StatusAddress, status));
        return (BinaryPrimitives.ReadUInt64LittleEndian(status),
            BinaryPrimitives.ReadInt64LittleEndian(status[0x18..]),
            BinaryPrimitives.ReadInt32LittleEndian(status[0x30..]),
            BinaryPrimitives.ReadInt32LittleEndian(status[0x34..]));
    }

    private void AddFlipEvent()
    {
        _context[CpuRegister.Rdi] = ScratchAddress;
        _context[CpuRegister.Rsi] = 0;
        Assert.Equal(0, KernelEventQueueCompatExports.KernelCreateEqueue(_context));
        Assert.True(_context.TryReadUInt64(ScratchAddress, out _queue));
        _context[CpuRegister.Rdi] = _queue;
        _context[CpuRegister.Rsi] = unchecked((ulong)_handle);
        _context[CpuRegister.Rdx] = 0x55;
        Assert.Equal(0, VideoOutExports.VideoOutAddFlipEvent(_context));
    }

    private int WaitOne(uint timeoutUsec)
    {
        Assert.True(_memory.TryWrite(TimeoutAddress, BitConverter.GetBytes(timeoutUsec)));
        _context[CpuRegister.Rdi] = _queue;
        _context[CpuRegister.Rsi] = EventAddress;
        _context[CpuRegister.Rdx] = 1;
        _context[CpuRegister.Rcx] = OutCountAddress;
        _context[CpuRegister.R8] = TimeoutAddress;
        return KernelEventQueueCompatExports.KernelWaitEqueue(_context);
    }

    [Fact]
    public void AheadPublicationPublishesOnceAndTheRealCompletionOnlyReleasesTheGpuSlot()
    {
        AddFlipEvent();
        var before = Status();
        var request = Reserve(101);
        Assert.Equal(before.GpuQueue + 1, Status().GpuQueue);

        VideoOutExports.NoteFlipRecorded(request);
        var ahead = Status();
        Assert.Equal(before.Count + 1, ahead.Count);
        Assert.Equal(101, ahead.FlipArg);
        Assert.Equal(before.Pending, ahead.Pending);
        // The GPU still holds the flip.
        Assert.Equal(before.GpuQueue + 1, ahead.GpuQueue);
        Assert.Equal(0, WaitOne(100_000));
        // The pacer still has the request.
        Assert.True(VideoOutExports.IsFlipPresentationPending(request));
        Assert.Equal(VideoOutExports.FlipOutcome.Pending, VideoOutExports.GetFlipOutcomeForTests(request));

        VideoOutExports.CompleteFlipOnGpu(request);
        VideoOutExports.CompleteFlipOnGpu(request);
        var done = Status();
        Assert.Equal(ahead.Count, done.Count);
        Assert.Equal(before.GpuQueue, done.GpuQueue);
        Assert.Equal(TimedOut, WaitOne(1_000));
        Assert.True(VideoOutExports.CanPresentFlip(request, Stopwatch.GetTimestamp()));

        VideoOutExports.MarkFlipPresented(request);
        Assert.Equal(ahead.Count, Status().Count);
        Assert.Equal(TimedOut, WaitOne(1_000));
        Assert.Null(VideoOutExports.GetFlipOutcomeForTests(request));

        var counters = VideoOutExports.PeekFlipAheadCountersForTests();
        Assert.Equal(1, counters.AtRecord);
        Assert.Equal(0, counters.AtPrevious);
        Assert.Equal(1, counters.LeadSamples);
    }

    [Fact]
    public void TheNextFlipWaitsForThePreviousRealCompletion()
    {
        var before = Status();
        var first = Reserve(111);
        var second = Reserve(112);

        VideoOutExports.NoteFlipRecorded(first);
        VideoOutExports.NoteFlipRecorded(second);
        Assert.Equal(before.Count + 1, Status().Count);
        Assert.Equal(111, Status().FlipArg);

        VideoOutExports.CompleteFlipOnGpu(first);
        Assert.Equal(before.Count + 2, Status().Count);
        Assert.Equal(112, Status().FlipArg);

        var counters = VideoOutExports.PeekFlipAheadCountersForTests();
        Assert.Equal(1, counters.AtRecord);
        Assert.Equal(1, counters.AtPrevious);

        VideoOutExports.CompleteFlipOnGpu(second);
        VideoOutExports.MarkFlipPresented(first);
        VideoOutExports.MarkFlipPresented(second);
        Assert.Equal(before.Count + 2, Status().Count);
        Assert.Equal(before.GpuQueue, Status().GpuQueue);
        Assert.Equal(before.Pending, Status().Pending);
        Assert.Null(VideoOutExports.GetFlipOutcomeForTests(first));
        Assert.Null(VideoOutExports.GetFlipOutcomeForTests(second));
    }

    // A flip whose predecessor was not published ahead goes no later than today: when the
    // predecessor is shown, never before it.
    [Fact]
    public void AFlipIsNeverPublishedBeforeTheOneBeforeIt()
    {
        var before = Status();
        var first = Reserve(121);
        var second = Reserve(122);

        VideoOutExports.NoteFlipRecorded(second);
        Assert.Equal(before.Count, Status().Count);

        VideoOutExports.CompleteFlipOnGpu(first);
        Assert.Equal(before.Count, Status().Count);

        VideoOutExports.MarkFlipPresented(first);
        Assert.Equal(before.Count + 2, Status().Count);
        Assert.Equal(122, Status().FlipArg);

        VideoOutExports.CompleteFlipOnGpu(second);
        VideoOutExports.MarkFlipPresented(second);
        Assert.Equal(before.Count + 2, Status().Count);
        Assert.Equal(before.GpuQueue, Status().GpuQueue);
    }

    [Fact]
    public void ThePacerStillPresentsAFlipPublishedAheadInOrder()
    {
        var before = Status();
        var first = Reserve(131);
        VideoOutExports.NoteFlipRecorded(first);
        var second = Reserve(132);
        // The guest's next flip arrives before the first was shown; both stay with the pacer.
        Assert.False(VideoOutExports.IsFlipDone(_handle, -1));
        Assert.True(VideoOutExports.IsFlipPresentationPending(first));
        Assert.True(VideoOutExports.IsFlipPresentationPending(second));

        // Shown before its real completion: the request outlives the presentation until then.
        VideoOutExports.MarkFlipPresented(first);
        Assert.Equal(VideoOutExports.FlipOutcome.Presented, VideoOutExports.GetFlipOutcomeForTests(first));
        Assert.Equal(before.GpuQueue + 2, Status().GpuQueue);
        VideoOutExports.CompleteFlipOnGpu(first);
        Assert.Null(VideoOutExports.GetFlipOutcomeForTests(first));
        Assert.Equal(before.GpuQueue + 1, Status().GpuQueue);

        VideoOutExports.NoteFlipRecorded(second);
        Assert.Equal(before.Count + 2, Status().Count);
        VideoOutExports.CompleteFlipOnGpu(second);
        VideoOutExports.DiscardFlip(second);
        Assert.Equal(before.Count + 2, Status().Count);
        Assert.True(VideoOutExports.IsFlipDone(_handle, -1));
        Assert.Null(VideoOutExports.GetFlipOutcomeForTests(second));
        Assert.Equal(before.GpuQueue, Status().GpuQueue);
        Assert.Equal(before.Pending, Status().Pending);
    }

    [Fact]
    public void ALateNoticeAndCancelledOrCpuFlipsKeepTodaysTiming()
    {
        var before = Status();

        // Real completion first: the notice is ignored and presentation publishes.
        var late = Reserve(141);
        VideoOutExports.CompleteFlipOnGpu(late);
        VideoOutExports.NoteFlipRecorded(late);
        Assert.Equal(before.Count, Status().Count);
        VideoOutExports.MarkFlipPresented(late);
        Assert.Equal(before.Count + 1, Status().Count);

        // Cancelled: nothing is published.
        var cancelled = Reserve(142);
        VideoOutExports.CancelFlip(cancelled);
        VideoOutExports.NoteFlipRecorded(cancelled);
        Assert.Equal(before.Count + 1, Status().Count);

        // A CPU flip is not GPU-queued: a notice for it is ignored.
        var cpu = Reserve(143, gpuQueued: false);
        VideoOutExports.NoteFlipRecorded(cpu);
        Assert.Equal(before.Count + 1, Status().Count);

        // An unpublished CPU flip before a GPU flip holds the GPU flip back.
        var gpu = Reserve(144);
        VideoOutExports.NoteFlipRecorded(gpu);
        Assert.Equal(before.Count + 1, Status().Count);
        VideoOutExports.CompleteFlip(cpu);
        Assert.Equal(before.Count + 3, Status().Count);
        Assert.Equal(144, Status().FlipArg);
        VideoOutExports.MarkFlipPresented(cpu);

        // Cancelling a flip published ahead gives its GPU slot back.
        Assert.Equal(before.GpuQueue + 1, Status().GpuQueue);
        VideoOutExports.CancelFlip(gpu);
        Assert.Equal(before.GpuQueue, Status().GpuQueue);
        Assert.Equal(before.Pending, Status().Pending);
        Assert.Equal(before.Count + 3, Status().Count);

        Assert.Equal(3, VideoOutExports.PeekFlipAheadCountersForTests().LateNotices);
    }

    [Fact]
    public void WithTheSwitchOffTheNoticeChangesNothing()
    {
        VideoOutExports.ResetFlipAheadForTests(enabled: false);
        AddFlipEvent();
        var before = Status();
        var request = Reserve(151);

        VideoOutExports.NoteFlipRecorded(request);
        Assert.Equal(before.Count, Status().Count);
        Assert.Equal(TimedOut, WaitOne(1_000));

        VideoOutExports.CompleteFlipOnGpu(request);
        Assert.Equal(before.Count, Status().Count);
        Assert.Equal(before.GpuQueue, Status().GpuQueue);
        Assert.Equal(before.Pending + 1, Status().Pending);

        VideoOutExports.MarkFlipPresented(request);
        Assert.Equal(before.Count + 1, Status().Count);
        Assert.Equal(before.Pending, Status().Pending);
        Assert.Equal(0, WaitOne(100_000));
        Assert.Null(VideoOutExports.GetFlipOutcomeForTests(request));
        Assert.Equal(default, VideoOutExports.PeekFlipAheadCountersForTests());
    }
}
