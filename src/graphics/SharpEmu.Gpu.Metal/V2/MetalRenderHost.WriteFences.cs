// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;

namespace SharpEmu.Gpu.Metal.V2;

// A host read of guest bytes recorded work still writes waits for the submission serial of
// that work, and a serial is signalled only at the end of its command buffer. The coalescer
// keeps one command buffer open across guest submissions, so a word an early submission
// wrote was waited for together with everything recorded after it: the later submissions'
// whole GPU time, on top of the GPU work queued ahead.
//
// A write fence is a second shared event the command buffer signals between two guest
// submissions. It is planted only where the coalescer chose not to commit - a point where a
// commit was legal, so every write noted so far has its work encoded ahead of the signal -
// only when no render pass is open (ending one would store and reload its attachments), and
// only when a write since the last fence touched guest memory the CPU has waited on before
// (64 KiB blocks learned from those waits). Each live write range of that command buffer
// takes the fence's value (MetalGpuWriteRanges.StampFence); a range noted later starts over.
//
// A read waits for the fence when every pending write over its bytes in the newest pending
// serial carries one, and for the serial before it to complete. The guest sees the same bytes
// it would after the whole buffer: the writes it waits for are exactly the ones over its
// bytes, and the older serial covers every write over them from earlier buffers. Ordering with
// CPU writes is unchanged: pending command-processor and label bytes are still applied by the
// callers, and every other user of the write ranges still sees the serial pending.
//
// Eager fences: a block is learned only after a read of it has waited for the whole buffer, so
// the first read of a word, and every read of a word a ring buffer moves to a new block, paid the
// full wait. By default any recorded write since the last fence asks for one, so the first read
// waits for its writer too. The fence is still planted only where a commit was legal and no render
// pass is open, so what it orders is exactly what the learned fence orders; only how often one is
// planted changes (at most once per coalesced submission, CoalesceMaxSubmissions per buffer).
//
// SHARPEMU_METAL_V2_WRITE_FENCES=0 plants none and waits for whole command buffers again.
// SHARPEMU_METAL_V2_EAGER_WRITE_FENCES=0 plants them only for blocks reads have waited on.
internal sealed partial class MetalRenderHost
{
    private static readonly bool WriteFencesDefault =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_WRITE_FENCES") != "0";

    private static readonly bool EagerWriteFencesDefault =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_EAGER_WRITE_FENCES") != "0";

    private static readonly nint SelNotifyListener = MetalNative.Selector("notifyListener:atValue:block:");

    private const int ObservedBlockShift = 16;
    private const int MaxObservedBlocks = 256;

    // A read wider than this is a table or a buffer, not a polled word or an operand.
    private const ulong MaxObservedReadBlocks = 4;

    // How long a fence wait goes before saying, once, that it is still waiting.
    private const int WriteFenceStallMilliseconds = 5000;

    private readonly HashSet<ulong> _observedReadBlocks = [];
    private bool _writeFenceWanted;
    private nint _writeFenceEvent;
    private nint _writeFenceListener;
    private bool _writeFenceUnavailable;
    private ulong _writeFencePlanted;
    private long _writeFencesPlanted;
    private long _writeFenceWaits;
    private long _writeFenceReads;
    private long _writeFenceRenderSkips;

    /// <summary>Whether reads may wait for a write fence instead of the whole command buffer.</summary>
    internal bool WriteFences { get; set; } = WriteFencesDefault;

    /// <summary>Whether any recorded write asks for a fence, not only one into a block a read waited on.</summary>
    internal bool EagerWriteFences { get; set; } = EagerWriteFencesDefault;

    /// <summary>Tests only: runs with the command buffer just after a fence is planted.</summary>
    internal Action<nint>? WriteFencePlantedHook { get; set; }

    /// <summary>Host waits for the GPU since the last PERF report (tests read it).</summary>
    internal long HostGpuWaits => Interlocked.Read(ref _gpuWaits);

    internal (long Planted, long Waits, long Reads, long RenderSkips) WriteFenceCounters =>
        (Interlocked.Read(ref _writeFencesPlanted), Interlocked.Read(ref _writeFenceWaits),
            Interlocked.Read(ref _writeFenceReads), Interlocked.Read(ref _writeFenceRenderSkips));

    private string TakeWriteFenceCounters() =>
        $" write_fences={Interlocked.Exchange(ref _writeFencesPlanted, 0)}" +
        $" write_fence_waits={Interlocked.Exchange(ref _writeFenceWaits, 0)}" +
        $" write_fence_reads={Interlocked.Exchange(ref _writeFenceReads, 0)}" +
        $" write_fence_render_skips={Interlocked.Exchange(ref _writeFenceRenderSkips, 0)}";

    // A read that had to wait for recorded work: its bytes are worth a fence next time.
    private void NoteObservedRead(ulong address, ulong size)
    {
        if (!WriteFences || _writeFenceUnavailable || size == 0 || address > ulong.MaxValue - size)
        {
            return;
        }

        var first = address >> ObservedBlockShift;
        var last = (address + size - 1) >> ObservedBlockShift;
        if (last - first >= MaxObservedReadBlocks)
        {
            return;
        }

        if (_observedReadBlocks.Count + (int)(last - first + 1) > MaxObservedBlocks)
        {
            _observedReadBlocks.Clear();
        }

        for (var block = first; block <= last; block++)
        {
            _observedReadBlocks.Add(block);
        }
    }

    // A recorded write: any one (eager), or one into a block the CPU has waited on, asks for a fence
    // at the next boundary.
    private void NoteFencedWrite(ulong address, ulong size)
    {
        if (_writeFenceWanted || !WriteFences || size == 0 || address > ulong.MaxValue - size)
        {
            return;
        }

        if (EagerWriteFences && !_writeFenceUnavailable)
        {
            _writeFenceWanted = true;
            return;
        }

        if (_observedReadBlocks.Count == 0)
        {
            return;
        }

        var first = address >> ObservedBlockShift;
        var last = (address + size - 1) >> ObservedBlockShift;
        if ((ulong)_observedReadBlocks.Count <= last - first)
        {
            foreach (var block in _observedReadBlocks)
            {
                if (block >= first && block <= last)
                {
                    _writeFenceWanted = true;
                    return;
                }
            }

            return;
        }

        for (var block = first; block <= last; block++)
        {
            if (_observedReadBlocks.Contains(block))
            {
                _writeFenceWanted = true;
                return;
            }
        }
    }

    /// <summary>
    /// A guest submission ended and the coalescer keeps the command buffer open: plants a write
    /// fence when a write since the last one touched memory the CPU reads.
    /// </summary>
    private void PlantWriteFenceIfWanted()
    {
        if (!_writeFenceWanted || !WriteFences || _commandBuffer == 0)
        {
            return;
        }

        if (_renderEncoder != 0)
        {
            // Ending the pass costs more than the narrower wait saves; the want stays for the
            // next boundary, and a read before it waits for the whole buffer as before.
            Interlocked.Increment(ref _writeFenceRenderSkips);
            return;
        }

        if (!EnsureWriteFenceEvent())
        {
            _writeFenceWanted = false;
            return;
        }

        // The signal goes between encoders, after the staged stores and label copies the
        // encoders owe, which ending them encodes.
        EndEncoders(PassEndCause.Other);
        if (_renderEncoder != 0 || _computeEncoder != 0)
        {
            return;
        }

        var value = ++_writeFencePlanted;
        MetalV2Native.SendEventValue(_commandBuffer, SelEncodeSignalEvent, _writeFenceEvent, value);
        _gpuWrites.StampFence(_submissionSerial, value);
        _writeFenceWanted = false;
        Interlocked.Increment(ref _writeFencesPlanted);
        WriteFencePlantedHook?.Invoke(_commandBuffer);
    }

    private bool EnsureWriteFenceEvent()
    {
        if (_writeFenceEvent != 0)
        {
            return true;
        }

        if (_writeFenceUnavailable)
        {
            return false;
        }

        // A fence is signalled in the middle of a command buffer, where no completion handler
        // runs; without a listener to wake the waiter it would be found by a timed poll, which
        // is slower than waiting for the buffer.
        var block = MetalCompletionSignal.Block;
        var listenerClass = MetalNative.Class("MTLSharedEventListener");
        if (!_completions.WantsCompletionHandler || block == 0 || listenerClass == 0)
        {
            _writeFenceUnavailable = true;
            return false;
        }

        var listener = MetalNative.Send(MetalNative.Send(listenerClass, MetalNative.Selector("alloc")), MetalNative.Selector("init"));
        var sharedEvent = MetalNative.Send(_device, SelNewSharedEvent);
        if (listener == 0 || sharedEvent == 0)
        {
            if (listener != 0)
            {
                MetalNative.SendVoid(listener, SelRelease);
            }

            if (sharedEvent != 0)
            {
                MetalNative.SendVoid(sharedEvent, SelRelease);
            }

            _writeFenceUnavailable = true;
            return false;
        }

        _writeFenceListener = listener;
        _writeFenceEvent = sharedEvent;
        return true;
    }

    private bool WriteFenceReached(ulong value) =>
        MetalV2Native.SendValue64(_writeFenceEvent, SelSignaledValue) >= value;

    /// <summary>
    /// Waits for the write fence that covers every pending write over the range in
    /// <paramref name="pending"/>, the newest serial writing it, and for the serial before it.
    /// False when no fence covers them all; the caller then waits for the serial.
    /// </summary>
    private bool TryWaitForWriteFence(ulong address, ulong size, ulong pending)
    {
        if (!WriteFences || _writeFencePlanted == 0 || _writeFenceEvent == 0)
        {
            return false;
        }

        var fence = _gpuWrites.WriteFence(address, size, pending);
        if (fence == 0)
        {
            return false;
        }

        // The fence was planted in the serial's own command buffer: it signals once that runs.
        if (_commandBuffer != 0 && pending >= _submissionSerial)
        {
            FlushCommandBuffer(MetalFlushCause.WaitGpu);
        }

        var older = pending - 1;
        bool Landed() => (WriteFenceReached(fence) && _completions.SignaledValue >= older) ||
            _completions.SignaledValue >= pending;
        if (Landed())
        {
            Interlocked.Increment(ref _writeFenceReads);
            return true;
        }

        using var waitProfile = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(
            SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.CommandGpuWait);
        var started = System.Diagnostics.Stopwatch.GetTimestamp();
        // One notification for this value; the block is the completion handler's, so it wakes
        // the same waiters a finished command buffer does.
        SendNotifyListener(_writeFenceEvent, SelNotifyListener, _writeFenceListener, fence, MetalCompletionSignal.Block);
        while (!_completions.WaitUntil(Landed, WriteFenceStallMilliseconds))
        {
            MetalV2Diagnostics.WarnOnce(
                "write-fence-stall",
                $"A write-fence wait has not finished in {WriteFenceStallMilliseconds} ms and is still waiting: " +
                $"fence={fence} reached={MetalV2Native.SendValue64(_writeFenceEvent, SelSignaledValue)} serial={pending} " +
                $"completed={CompletedSerial}.");
        }

        var buffer = InFlightBufferFor(pending, out var commitTime);
        var signalled = System.Diagnostics.Stopwatch.GetTimestamp() - started;
        ReportWait(pending, started, started, buffer, commitTime, new MetalCompletionWait(signalled, 0),
            MetalWaitReason.ReadRange, partial: true);
        Interlocked.Increment(ref _writeFenceWaits);
        Interlocked.Increment(ref _partialWaits);
        var completed = CompletedSerial;
        _guestMemory.NoteCompleted(completed);
        _gpuWrites.Retire(completed);
        RetirePendingCopies(completed);
        _cpOverlay.Retire(completed);
        _labelOverlay.Retire(completed);
        InvalidateWordLines();
        return true;
    }

    private void DisposeWriteFences()
    {
        if (_writeFenceEvent != 0)
        {
            MetalNative.SendVoid(_writeFenceEvent, SelRelease);
            _writeFenceEvent = 0;
        }

        if (_writeFenceListener != 0)
        {
            MetalNative.SendVoid(_writeFenceListener, SelRelease);
            _writeFenceListener = 0;
        }
    }

    // -notifyListener:atValue:block:
    [LibraryImport("/usr/lib/libobjc.A.dylib", EntryPoint = "objc_msgSend")]
    private static partial void SendNotifyListener(nint receiver, nint selector, nint listener, ulong value, nint block);
}
