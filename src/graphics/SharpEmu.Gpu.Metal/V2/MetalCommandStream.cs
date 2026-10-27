// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.HLE.DsTrace;
using SharpEmu.Libs.Agc;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.VideoOut;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// The command-stream half of the render host: the packet interpreter's view of the
/// backend. The command translation layer builds its RenderExecutor and
/// ShaderPipelineCache only when one object implements ICommandStreamHost,
/// IRenderHost and IShaderPipelineHost at once, which is why this is a partial of
/// MetalRenderHost rather than a type of its own.
///
/// Everything here runs on the GPU worker thread. Completion is a shared event:
/// a committed command buffer signals its serial, and the completion thread runs
/// the interrupts and flip completions of that submission in order.
/// </summary>
internal sealed partial class MetalRenderHost : ICommandStreamHost
{
    private CommandStreamQueue? _commandStream;
    private AgcExports.CommandStreamTranslation? _translation;
    private IMetalFlipTarget? _flipTarget;
    private long _lastProgressTicks;
    private ulong _submissionsStarted;
    private ulong _endOfPipeRecords;
    private ulong _frameBoundaries;

    private static readonly long _blockedRetryTicks =
        CommandStreamQueue.AllBlockedRetryMilliseconds * System.Diagnostics.Stopwatch.Frequency / 1000L;

    public CommandStreamQueue CommandStream =>
        _commandStream ?? throw Fatal("The Metal command stream has not been created.");

    /// <summary>Creates the interpreter queue and the draw translation over this host.</summary>
    public void InitializeCommandStream(IMetalFlipTarget flipTarget)
    {
        _flipTarget = flipTarget;
        ApplyTranslationSwitches();
        _commandStream = new CommandStreamQueue(this);
        _translation = AgcBackendServices.CreateCommandStreamTranslation(_memory, this);
        _lastProgressTicks = System.Diagnostics.Stopwatch.GetTimestamp();
    }

    ICpuMemory ICommandStreamHost.Memory => _memory;

    public void RunPendingCommands()
    {
        // With the stream on its own thread the worker runs them between its items.
        if (OnCommandThread)
        {
            return;
        }

        if (_relay.HasPendingCommands)
        {
            DrainDraws();
        }

        _relay.RunPendingCommands();
    }

    public void BeginSubmission(int queueId, ulong submissionId, object? geometrySnapshots)
    {
        if (OnCommandThread)
        {
            // The translation state belongs to the interpreting thread; the command buffer to the worker.
            _submissionsStarted++;
            _translation!.BeginSubmission(queueId, submissionId, geometrySnapshots, CommandStream.GetInterpreter(queueId));
            _commandThread!.Post(() =>
            {
                NativeOcclusionBeginQueue(queueId);
                CommitCoalescedIfOverdue();
                _ = EnsureCommandBuffer();
            });
            return;
        }

        DrainDraws();
        NativeOcclusionBeginQueue(queueId);
        _submissionsStarted++;
        CommitCoalescedIfOverdue();
        _ = EnsureCommandBuffer();
        _translation!.BeginSubmission(queueId, submissionId, geometrySnapshots, CommandStream.GetInterpreter(queueId));
    }

    void ICommandStreamHost.Flush()
    {
        if (PostToWorker(() => ((ICommandStreamHost)this).Flush()))
        {
            return;
        }

        DrainDraws();
        _frameBoundaries++;
        FlushStream();
    }

    // A frame boundary is the guest's own suspend point: what the frame recorded goes
    // out now, whatever the coalescing limits say. It is at most one commit a frame.
    void ICommandStreamHost.FlushFrame()
    {
        // The guest reuses the frame's memory once the boundary returns: the frame is recorded first.
        if (RunOnWorker(() => ((ICommandStreamHost)this).FlushFrame()))
        {
            return;
        }

        DrainDraws();
        _frameBoundaries++;
        FlushCommandBuffer(MetalFlushCause.Stream);
    }

    // ---- command-buffer coalescing (SHARPEMU_METAL_V2_FLUSH_COALESCE) ----
    //
    // The stream asks for a flush at the end of every guest submission that made progress.
    // Committing there costs the GPU an idle gap per submission (commit-to-schedule latency
    // plus however long the worker takes to record the next buffer), so a Stream flush only
    // marks the buffer as wanting a commit and the commit happens at the first of:
    //  - the stream has nothing else runnable (the worker is about to park anyway);
    //  - the buffer holds enough work (submissions or encoders over the limit);
    //  - the first deferred submission is older than the time limit, checked at every
    //    submission start and at draw/dispatch preparations outside a render pass as well,
    //    so a label a guest spins on is never held for long while the worker keeps recording;
    //  - any other commit cause (present, budget, host wait, completion, poll, labels),
    //    which commits the same buffer and clears the request.

    // When the first deferred Stream flush of the open buffer happened; 0 when none is owed.
    private long _commitWantedSince;
    private int _coalescedSubmissions;
    private long _deferredStreamFlushes;

    private static readonly long CoalesceMaxTicks =
        MetalV2Diagnostics.CoalesceMaxMicroseconds * System.Diagnostics.Stopwatch.Frequency / 1_000_000L;

    /// <summary>Stream flushes that left the buffer open instead of committing it.</summary>
    public long TakeDeferredStreamFlushes() => Interlocked.Exchange(ref _deferredStreamFlushes, 0);

    /// <summary>
    /// Whether a Stream flush commits now. Pure, so the policy is testable without a device.
    /// </summary>
    internal static bool ShouldCommitStreamFlush(
        bool coalesce,
        bool streamIdle,
        int submissions,
        int encoders,
        long wantedTicks,
        int maxSubmissions,
        int maxEncoders,
        long maxTicks) =>
        !coalesce || streamIdle || submissions >= maxSubmissions || encoders >= maxEncoders ||
        wantedTicks >= maxTicks;

    private void FlushStream()
    {
        if (_commandBuffer == 0)
        {
            // Nothing is open, so there is nothing to commit or to owe.
            return;
        }

        var now = System.Diagnostics.Stopwatch.GetTimestamp();
        if (_commitWantedSince == 0)
        {
            _commitWantedSince = now;
        }

        _coalescedSubmissions++;
        var coalesce = MetalV2Diagnostics.CoalesceFlushes;
        // Read only when the cheaper limits leave the decision open: it takes the queue lock.
        var commit = ShouldCommitStreamFlush(
            coalesce, streamIdle: false, _coalescedSubmissions, _encodersInBuffer, now - _commitWantedSince,
            MetalV2Diagnostics.CoalesceMaxSubmissions, MetalV2Diagnostics.CoalesceMaxEncoders, CoalesceMaxTicks) ||
            // With a producer, an empty packet queue can still have many worker items
            // behind this flush. Its ordered idle callback commits after those items;
            // testing the producer queue here would commit each parsed submission.
            (!UsesCommandThread && !(_commandStream?.HasUnblockedPending ?? false));
        if (commit)
        {
            FlushCommandBuffer(MetalFlushCause.Stream);
            return;
        }

        // A commit would have been legal here, so a write fence covers everything recorded.
        PlantWriteFenceIfWanted();
        Interlocked.Increment(ref _deferredStreamFlushes);
    }

    /// <summary>Commits the buffer a deferred Stream flush left open, if one is owed.</summary>
    internal void CommitCoalescedFlush()
    {
        if (_commitWantedSince != 0)
        {
            FlushCommandBuffer(MetalFlushCause.Stream);
        }
    }

    // The time limit, checked where the worker is between packets and outside a resource
    // preparation: a submission start and a draw or dispatch about to prepare.
    private void CommitCoalescedIfOverdue()
    {
        if (_commitWantedSince != 0 &&
            System.Diagnostics.Stopwatch.GetTimestamp() - _commitWantedSince >= CoalesceMaxTicks)
        {
            FlushCommandBuffer(MetalFlushCause.Stream);
        }
    }

    void ICommandStreamHost.FlushAndWait()
    {
        if (RunOnWorker(() => ((ICommandStreamHost)this).FlushAndWait()))
        {
            return;
        }

        DrainDraws();
        FlushAndWaitForGpu(MetalWaitReason.CommandProcessor);
    }

    public void SynchronizeGpu()
    {
        if (RunOnWorker(SynchronizeGpu))
        {
            return;
        }

        DrainDraws();
        _imageCache.FlushAllGpuWrites();
        FlushAndWaitForGpu(MetalWaitReason.Synchronize);
    }

    public void RunGarbageCollector()
    {
        if (PostToWorker(RunGarbageCollector))
        {
            return;
        }

        DrainDraws();
        _imageCache.Collect();
    }

    // SHARPEMU_TRACE_GDS=1: the first data share fills and copies, offsets and guest addresses.
    private static readonly bool TraceGds = Environment.GetEnvironmentVariable("SHARPEMU_TRACE_GDS") == "1";
    private int _tracedGds;

    private void TraceGdsTransfer(string what)
    {
        if (TraceGds && Interlocked.Increment(ref _tracedGds) <= 600)
            Console.Error.WriteLine($"[METAL-V2][GDS] {what}");
    }

    // SHARPEMU_TRACE_DMA_RANGE=<lo>-<hi> (hex): guest memory fills and copies whose destination
    // overlaps the range, as the data share trace prints them.
    private static readonly (ulong Low, ulong High) TraceDmaRange = ParseTraceDmaRange(
        Environment.GetEnvironmentVariable("SHARPEMU_TRACE_DMA_RANGE"));

    private static (ulong, ulong) ParseTraceDmaRange(string? text)
    {
        var parts = (text ?? string.Empty).Split('-');
        return parts.Length == 2 &&
            ulong.TryParse(parts[0].Replace("0x", string.Empty), System.Globalization.NumberStyles.HexNumber, null, out var low) &&
            ulong.TryParse(parts[1].Replace("0x", string.Empty), System.Globalization.NumberStyles.HexNumber, null, out var high)
            ? (low, high)
            : (0, 0);
    }

    private void TraceDmaTransfer(ulong destination, ulong size, string what)
    {
        if (TraceDmaRange.High != 0 && destination < TraceDmaRange.High && TraceDmaRange.Low < destination + size &&
            Interlocked.Increment(ref _tracedGds) <= 600)
            Console.Error.WriteLine($"[METAL-V2][DMA] {what}");
    }

    public void FillBuffer(ulong address, ulong size, uint value, bool isGds)
    {
        if (OnCommandThread)
        {
            _commandThread!.PostRange(() => FillBuffer(address, size, value, isGds), address, isGds ? 0 : size);
            return;
        }

        DrainDraws();
        if (size == 0)
        {
            return;
        }

        if (isGds)
        {
            TraceGdsTransfer($"fill gds=0x{address:X} size=0x{size:X} value=0x{value:X}");
            if (address > MetalGuestMemory.GdsBytes || size > MetalGuestMemory.GdsBytes - address)
            {
                throw Fatal($"The GDS fill range is outside the buffer: offset=0x{address:X} size=0x{size:X}.");
            }

            if (TryFillGdsOnGpu(address, size, value))
            {
                return;
            }

            // The data share is a shared-storage buffer, so the host fill is visible to
            // the GPU without a transfer; a shader write to it is ordered by Metal.
            SynchronizeGpu();
            var words = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, uint>(
                _guestMemory.GdsWords().Slice((int)address, (int)size));
            words.Fill(value);
            return;
        }

        TraceDmaTransfer(address, size, $"fill mem=0x{address:X} size=0x{size:X} value=0x{value:X}");
        FillGuestMemory(address, size, value);
    }

    public void CopyBuffer(ulong destination, ulong source, ulong size, bool destinationIsGds, bool sourceIsGds)
    {
        if (OnCommandThread)
        {
            _commandThread!.PostRange(
                () => CopyBuffer(destination, source, size, destinationIsGds, sourceIsGds), destination, destinationIsGds ? 0 : size);
            return;
        }

        DrainDraws();
        if (size == 0)
        {
            return;
        }

        if (destinationIsGds && sourceIsGds)
        {
            throw Fatal($"A GDS-to-GDS copy is not supported: destination=0x{destination:X} source=0x{source:X} size=0x{size:X}.");
        }

        if (!destinationIsGds && !sourceIsGds)
        {
            TraceDmaTransfer(destination, size, $"copy mem=0x{source:X} -> mem=0x{destination:X} size=0x{size:X}");
            CopyGuestMemory(destination, source, size);
            return;
        }

        TraceGdsTransfer(sourceIsGds
            ? $"copy gds=0x{source:X} -> mem=0x{destination:X} size=0x{size:X}"
            : $"copy mem=0x{source:X} -> gds=0x{destination:X} size=0x{size:X}");
        if (sourceIsGds
                ? source > MetalGuestMemory.GdsBytes || size > MetalGuestMemory.GdsBytes - source
                : destination > MetalGuestMemory.GdsBytes || size > MetalGuestMemory.GdsBytes - destination)
        {
            throw Fatal(
                $"The GDS copy is outside the buffer: destination=0x{destination:X} source=0x{source:X} size=0x{size:X} from_gds={sourceIsGds}.");
        }

        if (TryCopyGdsOnGpu(destination, source, size, destinationIsGds))
        {
            return;
        }

        SynchronizeGpu();
        var gds = _guestMemory.GdsWords();
        if (sourceIsGds)
        {

            var watch = _guard?.Begin(destination, size, "gds-copy") ?? -1;
            var wrote = _memory.TryWrite(destination, gds.Slice((int)source, (int)size));
            _guard?.End(watch);
            if (!wrote)
            {
                throw Fatal($"The GDS copy cannot write guest memory: address=0x{destination:X16} size=0x{size:X}.");
            }

            _imageCache.InvalidateMemory(destination, size);
            return;
        }

        if (destination > MetalGuestMemory.GdsBytes || size > MetalGuestMemory.GdsBytes - destination)
        {
            throw Fatal($"The GDS copy destination is outside the buffer: offset=0x{destination:X} size=0x{size:X}.");
        }

        if (!TryReadGuest(source, gds.Slice((int)destination, (int)size)))
        {
            throw Fatal($"The GDS copy cannot read guest memory: address=0x{source:X16} size=0x{size:X}.");
        }
    }

    public void ReadGds(Span<uint> destination, uint wordOffset, uint wordCount)
    {
        if (OnCommandThread)
        {
            var words = new uint[destination.Length];
            RunTagged(() => ReadGds(words, wordOffset, wordCount));
            words.CopyTo(destination);
            return;
        }

        DrainDraws();
        SynchronizeGpu();
        GlobalDataShare.ReadWords(_guestMemory.GdsWords(), destination, wordOffset, wordCount);
    }

    // A label is published after the work it reports, as the hardware publishes it. The
    // GPU writes it itself where the packet sits in the stream, so no completion and no
    // flush are needed for the bytes; only an interrupt or a flip, which the host has to
    // deliver, still waits for the shared event — and that completion then also carries
    // the label whenever the GPU could not write it.
    public void RecordEndOfPipe(in EndOfPipeWrite write)
    {
        if (OnCommandThread)
        {
            var queued = write;
            _commandThread!.Post(() =>
            {
                if (PendingLabelsVisible && EndOfPipeLabelBytes(queued.Kind) != 0)
                {
                    CommandStream.UpdatePendingLabelTick(queued.Destination, SubmissionSerial);
                }

                RecordEndOfPipe(in queued);
            });
            return;
        }

        // Behind deferred draws the label waits in their queue, as on the command thread: it lands
        // after their work either way, and the drain records it with the serial it rides.
        if (MetalV2Diagnostics.DeferEndOfPipe)
        {
            var deferred = write;
            if (TryDeferBehindDraws(() =>
                {
                    if (PendingLabelsVisible && EndOfPipeLabelBytes(deferred.Kind) != 0)
                    {
                        CommandStream.UpdatePendingLabelTick(deferred.Destination, SubmissionSerial);
                    }

                    RecordEndOfPipe(in deferred);
                }, write.Destination, write.Destination == 0 ? 0 : Math.Max(sizeof(ulong), (ulong)EndOfPipeLabelBytes(write.Kind))))
            {
                return;
            }
        }

        DrainDraws();
        using var endOfPipeScope = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(
            SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.CommandEndOfPipe);
        _endOfPipeRecords++;
        // Recorded ahead of the label, so its bytes land before the guest is told the
        // work is done.
        _imageCache.PublishLinearGpuWrites();
        var label = write;
        var eventId = write.EventId;
        var contextId = write.ContextId;
        var requestId = write.FlipRequestId;
        var encoded = TryEncodeEndOfPipeLabel(in write);
        if (DsTrace.Gpu && DsTrace.Armed)
        {
            // DSTRACE §4.7 LENC. deferred: 0 copy encoded at the packet, 1 copy deferred to
            // the next encoder boundary, 2 not encoded (published by the interpreter or a completion).
            DsTrace.Emit(DsEv.LENC, encoded ? _dsLabelSerial : _submissionSerial, write.Destination, write.Value,
                encoded ? _dsLabelDeferred : 2UL, (ulong)write.Kind);
        }
        if (EndOfPipeTrace && !encoded)
        {
            Console.Error.WriteLine(
                $"[METAL-V2][EOP] host kind={write.Kind} address=0x{write.Destination:X16} value=0x{write.Value:X16} " +
                $"deferred={DefersEndOfPipeLabel(write.Kind)} event={write.EventId} flip={write.FlipRequestId}" +
                ClockSamplingTrace(in write));
        }

        // A label the GPU could not write is owed until its completion publishes it, and
        // while one is owed every later label waits behind it (see OrderedEndOfPipe).
        var owed = !encoded && EndOfPipeLabelBytes(write.Kind) != 0 && DefersEndOfPipeLabel(write.Kind);
        var lazy = false;
        if (owed)
        {
            // A clock owed here is sampled again by its completion (ClockSampling).
            label = write with { ReferenceClock = SamplesClockAtCompletion(in write) };
            lazy = RidesNextCommit(label.ReferenceClock);
            if (label.ReferenceClock)
            {
                Interlocked.Increment(ref _owedClockLabels);
            }

            Interlocked.Increment(ref _owedHostLabels);
        }
        else
        {
            label = write with { ReferenceClock = false };
        }

        switch (write.Kind)
        {
            case EndOfPipeWriteKind.InterruptOnly:
            case EndOfPipeWriteKind.Interrupt32:
            case EndOfPipeWriteKind.Interrupt64:
            case EndOfPipeWriteKind.InterruptWriteBack32:
            case EndOfPipeWriteKind.InterruptWriteBack64:
                QueueCompletion(() =>
                {
                    if (!encoded) PublishHostLabel(in label, owed);
                    TriggerInterrupt(eventId, contextId);
                });
                break;
            case EndOfPipeWriteKind.Flip:
            case EndOfPipeWriteKind.FlipWithWrite32:
                QueueCompletion(() =>
                {
                    if (!encoded) PublishHostLabel(in label, owed);
                    VideoOutBackendServices.CompleteFlipOnGpu(requestId);
                    MetalV2Presenter.SignalFlipPacer();
                });
                break;
            case EndOfPipeWriteKind.FlipWithInterruptWriteBack32:
                QueueCompletion(() =>
                {
                    if (!encoded) PublishHostLabel(in label, owed);
                    VideoOutBackendServices.CompleteFlipOnGpu(requestId);
                    MetalV2Presenter.SignalFlipPacer();
                    TriggerInterrupt(eventId, 0);
                });
                break;
            case EndOfPipeWriteKind.GdsWrite32:
                RecordEndOfPipeGdsWrite(in write);
                break;
            case EndOfPipeWriteKind.AtomicAdd32:
                RecordEndOfPipeAtomicAdd(in write);
                break;
            default:
                // A plain label or clock write has no interrupt behind it, but it is
                // still an end-of-pipe write: the guest polls it to learn that the work
                // recorded before it has finished, and reuses what that work reads once
                // it sees it. The GPU writing it needs nothing further; a label outside
                // every window has to be published by a completion, which costs a flush.
                if (!encoded)
                {
                    QueueCompletion(() => PublishHostLabel(in label, owed),
                        lazy ? (label.Destination, EndOfPipeLabelBytes(label.Kind)) : null);
                }

                break;
        }
    }

    /// <summary>The interpreter leaves end-of-pipe GDS stores to this host (RecordEndOfPipeGdsWrite).</summary>
    public bool WritesGdsAtEndOfPipe => true;

    /// <summary>
    /// An end-of-pipe GDS store: the words are copied out of the data share where the packet
    /// sits in the stream, after the work that wrote them, instead of the worker waiting for
    /// the GPU and copying them itself. Where the GPU cannot write the destination, or an
    /// older label is still owed by a completion, a completion copies them in order instead.
    /// </summary>
    private void RecordEndOfPipeGdsWrite(in EndOfPipeWrite write)
    {
        var bytes = (ulong)write.GdsWordCount * sizeof(uint);
        var source = (ulong)write.GdsWordOffset * sizeof(uint);
        if (bytes == 0)
        {
            return;
        }

        if (source > MetalGuestMemory.GdsBytes || bytes > MetalGuestMemory.GdsBytes - source ||
            write.Destination == 0 || bytes > ulong.MaxValue - write.Destination)
        {
            throw Fatal(
                $"An end-of-pipe GDS store is out of range: gds=0x{source:X}+0x{bytes:X} address=0x{write.Destination:X16}.");
        }

        if (GpuTimedLabels && MayEncodeLabelOnGpu(OrderedEndOfPipe, Volatile.Read(ref _owedHostLabels)))
        {
            var destination = _guestMemory.ResolveWritable(write.Destination, bytes);
            if (destination.IsValid)
            {
                var copy = new DeferredLabelCopy(
                    _guestMemory.GdsBuffer, source, destination.Buffer, destination.Offset, (int)bytes, write.Destination);
                _ = EncodeOrderedCopy(in copy, write.Destination, null, 0,
                    MetalGpuWriteOrigin.EndOfPipeLabel, "eop gds", null, bytes, sourceIsFinal: false);
                Interlocked.Increment(ref _gpuTimedLabels);
                return;
            }
        }

        var destinationAddress = write.Destination;
        var wordOffset = write.GdsWordOffset;
        var wordCount = write.GdsWordCount;
        Interlocked.Increment(ref _owedHostLabels);
        QueueCompletion(() =>
        {
            try
            {
                var words = new uint[wordCount];
                GlobalDataShare.ReadWords(_guestMemory.GdsWords(), words, wordOffset, wordCount);
                if (!_memory.TryWrite(destinationAddress, System.Runtime.InteropServices.MemoryMarshal.AsBytes<uint>(words)))
                {
                    throw Fatal(
                        $"An end-of-pipe GDS store cannot write guest memory: address=0x{destinationAddress:X16} bytes={bytes}.");
                }

                _commandStream?.NotifyGuestMemoryWritten(destinationAddress, bytes);
            }
            finally
            {
                Interlocked.Decrement(ref _owedHostLabels);
            }
        });
    }

    /// <summary>
    /// RELEASE_MEM data select 7: the add happens once the work before it has finished, from
    /// that work's completion, and like an owed label it keeps later labels behind it.
    /// </summary>
    private void RecordEndOfPipeAtomicAdd(in EndOfPipeWrite write)
    {
        if (!DefersEndOfPipeLabel(EndOfPipeWriteKind.AtomicAdd32))
        {
            // The interpreter added as it parsed the packet.
            return;
        }

        var address = write.Destination;
        var addend = (uint)write.Value;
        Interlocked.Increment(ref _owedHostLabels);
        QueueCompletion(() =>
        {
            try
            {
                if (!SharpEmu.Libs.Gpu.GpuCommands.Packets.AtomicMemoryPacket.TryAdd32(_memory, address, addend))
                {
                    throw Fatal($"An end-of-pipe atomic add cannot update guest memory: address=0x{address:X16}.");
                }

                _commandStream?.NotifyGuestMemoryWritten(address, sizeof(uint));
            }
            finally
            {
                Interlocked.Decrement(ref _owedHostLabels);
            }
        });
    }

    /// <summary>
    /// Encodes a label's bytes into the command buffer where its packet sits, so the GPU
    /// writes the guest window itself once it reaches them.
    /// </summary>
    /// <remarks>
    /// This is what the label means: the work recorded before it has finished. Handing the
    /// bytes to a completion says the same thing but costs a command buffer flush per label
    /// and then waits for the host to be told the submission finished, which on a title that
    /// writes a label per render pass was measured at 24 fps down to 9.5. A copy from a
    /// staging ring into the window has neither cost: blit encoders run in the order they
    /// were created, after the encoders of the work the label reports, which is the same
    /// ordering the untracked windows already rely on everywhere else on this host.
    ///
    /// The bytes land when the command buffer is committed, so the label is recorded as GPU
    /// work: a command processor read of those bytes waits for the submission, and the
    /// worker commits what it has before it parks (<see cref="FlushPendingLabelWrites"/>).
    /// </remarks>
    private bool TryEncodeEndOfPipeLabel(in EndOfPipeWrite write)
    {
        var bytes = EndOfPipeLabelBytes(write.Kind);
        if (bytes == 0)
        {
            // A GDS write was copied where the data share lives, and an interrupt or a
            // bare flip carries no payload at all.
            return false;
        }

        if (write.Destination == 0 || (ulong)bytes > ulong.MaxValue - write.Destination)
        {
            throw Fatal(
                $"An end-of-pipe label has an invalid range: address=0x{write.Destination:X16} bytes={bytes}.");
        }

        if (!GpuTimedLabels || !DefersEndOfPipeLabel(write.Kind))
        {
            // The interpreter published these bytes as it parsed the packet.
            return false;
        }

        if (SamplesClockAtCompletion(in write))
        {
            // The GPU would copy the clock as it was parsed; only the completion of the work
            // before it knows when that work finished (ClockSampling).
            return false;
        }

        if (!MayEncodeLabelOnGpu(OrderedEndOfPipe, Volatile.Read(ref _owedHostLabels)))
        {
            // An older label is still owed by a completion. Written by the GPU now, this one
            // would land first, and a guest that reuses memory on the newer signal then has
            // the older label land in it. Its completion is queued behind the owed one's.
            Interlocked.Increment(ref _orderedHostLabels);
            CountClockForcedLabel();
            return false;
        }

        var destination = _guestMemory.ResolveWritable(write.Destination, (ulong)bytes);
        if (!destination.IsValid)
        {
            Interlocked.Increment(ref _labelsWithoutWindow);
            return false;
        }

        var staging = _guestMemory.Allocate((ulong)bytes, sizeof(ulong), out var payload);
        if (!staging.IsValid)
        {
            return false;
        }

        Span<byte> value = stackalloc byte[sizeof(ulong)];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt64LittleEndian(value, write.Value);
        value[..bytes].CopyTo(payload);

        var copy = new DeferredLabelCopy(
            staging.Buffer, staging.Offset, destination.Buffer, destination.Offset, bytes, write.Destination);
        var serial = EncodeOrderedCopy(in copy, write.Destination,
            CommandProcessorWriteHistory.Enabled ? GpuClobberDiagnostics.LandedKind(write.Kind) : null,
            bytes == sizeof(uint) ? (uint)write.Value : write.Value,
            MetalGpuWriteOrigin.EndOfPipeLabel, "eop label", null, (ulong)bytes);
        Interlocked.Increment(ref _gpuTimedLabels);

        // The resource materializer reads the label as written until it lands (LabelOverlay).
        NoteEndOfPipeLabelPending(write.Destination, value[..bytes], serial);
        if (EndOfPipeTrace)
        {
            var address = write.Destination;
            var kind = write.Kind;
            var labelValue = write.Value;
            var queued = System.Diagnostics.Stopwatch.GetTimestamp();
            Console.Error.WriteLine(
                $"[METAL-V2][EOP] encode kind={kind} address=0x{address:X16} bytes={bytes} " +
                $"value=0x{labelValue:X16} serial={serial}" + ClockSamplingTrace(in write));
            _completions.Enqueue(serial, () => Console.Error.WriteLine(
                $"[METAL-V2][EOP] landed address=0x{address:X16} serial={serial} waited_ms=" +
                ((System.Diagnostics.Stopwatch.GetTimestamp() - queued) * 1000.0 / System.Diagnostics.Stopwatch.Frequency)
                    .ToString("F1", System.Globalization.CultureInfo.InvariantCulture)));
        }
        return true;
    }

    /// <summary>
    /// Encodes a copy into guest memory where the stream has reached, the way a label's bytes
    /// are: at once between encoders, or at the next encoder boundary while a render pass or
    /// a compute encoder is open. Either way it lands after every write recorded before it
    /// and before every one recorded after it. Returns the submission that carries it.
    /// </summary>
    /// <remarks>
    /// A deferred copy lands after every draw or dispatch recorded into the same encoder
    /// after it, not only the ones before it. A store the command processor makes and a
    /// later draw then reads on the GPU - a WRITE_DATA into a buffer a shader reads - is
    /// read before the bytes land, and a shader store to those bytes is overwritten by them.
    /// The render-pass deferral has always accepted this for draws: nothing checks the
    /// copy's destination against what the pass binds. A compute encoder stays open over
    /// copies too, because the encoder split it avoids is the frame's largest GPU cost, but
    /// a dispatch that binds a copy's bytes ends it first (OrderAfterDeferredCopies): a
    /// compute queue writes a table and dispatches over it at once, and GTA V's bounds
    /// pass atomically updates the bytes its WRITE_DATA has just initialised. What reads the
    /// bytes at parse time is not affected (MetalPendingWriteOverlay), nor is an image made
    /// from them: the write invalidates it, and the upload that replaces it is a transfer,
    /// which ends the encoder and lands the copies first.
    /// SHARPEMU_METAL_V2_DEFER_COMPUTE_COPIES=0 ends the compute encoder for every copy again.
    /// </remarks>
    private ulong EncodeOrderedCopy(
        in DeferredLabelCopy copy,
        ulong guestAddress,
        string? landedKind,
        ulong landedValue,
        MetalGpuWriteOrigin origin,
        string profileWhat,
        string? profileKind,
        ulong profileBytes,
        bool sourceIsFinal = true) =>
        EncodeOrderedCopies(new ReadOnlySpan<DeferredLabelCopy>(in copy), guestAddress, landedKind, landedValue,
            origin, profileWhat, profileKind, profileBytes, sourceIsFinal);

    /// <summary>
    /// <see cref="EncodeOrderedCopy"/> for several copies recorded back to back: they take the
    /// same place in the stream, in the order given, with nothing between them. A diagnostic
    /// <paramref name="landedKind"/> describes one copy, so it goes with a single one.
    /// </summary>
    private ulong EncodeOrderedCopies(
        ReadOnlySpan<DeferredLabelCopy> copies,
        ulong guestAddress,
        string? landedKind,
        ulong landedValue,
        MetalGpuWriteOrigin origin,
        string profileWhat,
        string? profileKind,
        ulong profileBytes,
        bool sourceIsFinal = true,
        bool deferStableGpuResult = false)
    {
        ulong serial;
        // A copy whose source later work in the same encoder may still change (the GDS a
        // pixel or compute shader appends to) cannot wait for the boundary: it would copy
        // what that later work left there.
        // A GPU-produced snapshot may wait for its producer encoder to end, but must
        // never be read on the CPU for label replay before that producer has executed.
        if ((sourceIsFinal || deferStableGpuResult) &&
            (LabelsAtCommit ||
             (MetalV2Diagnostics.DeferLabelWrites && _renderEncoder != 0) ||
             (DefersCopiesInCompute && _computeEncoder != 0)))
        {
            // Held to the commit, the copies need a command buffer to land in.
            if (LabelsAtCommit)
            {
                EnsureCommandBuffer();
            }

            // Four bytes are not worth a framebuffer. Ending the open pass to copy them
            // stores and reloads every attachment, which at 3840x2160 is tens of
            // megabytes for a label the guest polls, so the copy waits for the next
            // boundary between encoders instead — one the frame reaches anyway. It then
            // lands after more work than the packet named, and a label that lands late
            // still only ever says that the work before it has finished. A compute
            // encoder is held open the same way: ending it per copy split each frame
            // into ~114 encoders, and the gaps between them outcost the dispatches.
            HoldLabelCopies(copies);
            Interlocked.Add(ref _deferredLabelWrites, copies.Length);
            NoteRecordedWork();
            serial = SubmissionSerial;
            if (DsTrace.Gpu)
            {
                _dsLabelDeferred = 1;
            }
        }
        else
        {
            // A copy that cannot wait lands after the held ones: copies land in stream order.
            if (LabelsAtCommit && _deferredLabelCopies.Count != 0)
            {
                OrderAfterDeferredCopies(0, ulong.MaxValue);
            }

            var context = ((IMetalImageCacheOwner)this).BeginTransfer(MetalTransferCause.EndOfPipeLabel);
            EncodeLabelCopies(context.CommandBuffer, copies);

            // Beginning the transfer marked this submission as carrying GPU work, which is
            // what keeps a completion queued behind it — the interrupt of this same packet
            // — from running inline ahead of the bytes.
            if (context.WantsEncodedLabels)
            {
                context.ProfileEncoded(profileKind is null
                    ? $"{profileWhat} bytes={profileBytes}"
                    : $"{profileWhat} {profileKind} bytes={profileBytes}");
            }
            serial = context.SubmissionSerial;
            if (DsTrace.Gpu)
            {
                _dsLabelDeferred = 0;
            }
        }

        if (DsTrace.Gpu)
        {
            _dsLabelSerial = serial;
        }

        if (sourceIsFinal)
        {
            KeepLabelReplays(copies, serial);
        }

        // A command processor polling these bytes suspends until something says they
        // changed, and the only thing that knows is the completion of this submission. The
        // worker reports completions when it next flushes, which is never while every head
        // is suspended, so one wake per submission is queued here: it costs no flush, and
        // without it a poll waits out the blocked-retry timer instead of a millisecond.
        if (_labelWakeSerial != serial)
        {
            _labelWakeSerial = serial;
            _completions.Enqueue(serial, () =>
            {
                if (DsTrace.Gpu)
                {
                    DsNoteLabelsLanded(serial);
                }

                _commandStream?.NotifyCompletedGpuTick(serial);
            });
        }

        if (landedKind is not null && CommandProcessorWriteHistory.Enabled)
        {
            // SHARPEMU_GPU_CLOBBER_DIAG / SHARPEMU_CP_HISTORY: the bytes are in guest memory once this serial completes.
            var bytes = copies[0].Bytes;
            _completions.Enqueue(serial, () => CommandProcessorWriteHistory.Record(
                -1, guestAddress, bytes, landedValue, landedKind, deferred: true));
        }

        foreach (ref readonly var copy in copies)
        {
            if (origin != MetalGpuWriteOrigin.EndOfPipeLabel)
            {
                // A command-processor store can land in bytes an image was made from; a label
                // never has, and checking for one per label is not free.
                _imageCache.InvalidateMemoryFromGpu(copy.GuestAddress, (ulong)copy.Bytes);
            }
            else
            {
                // Nor does it re-arm a DCC claim; one standing over the label is read again.
                _imageCache.ForgetStandingDccClaims(copy.GuestAddress, (ulong)copy.Bytes);
            }

            using (SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(
                SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.CommandCpBookkeeping))
            {
                NoteGuestMemoryWrite(copy.GuestAddress, (ulong)copy.Bytes, origin);
            }

            _pendingLabelWrites++;
        }

        return serial;
    }

    /// <summary>
    /// Writes command-processor bytes so they land in record order with the deferred labels:
    /// copied by the GPU where the stream has reached, or - when the GPU cannot write them,
    /// or a label is still owed by a completion - published by a completion behind it.
    /// </summary>
    internal bool QueueCommandProcessorWrites { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_QUEUE_CP_WRITES") == "1";

    public void RecordCommandProcessorWrite(ulong address, ReadOnlySpan<byte> source, string kind)
    {
        if (OnCommandThread)
        {
            if (source.IsEmpty) return;
            if (address == 0 || (ulong)source.Length > ulong.MaxValue - address)
                throw Fatal($"A command-processor write has an invalid range: address=0x{address:X16} bytes={source.Length}.");
            var queued = source.ToArray();
            if (QueueCommandProcessorWrites)
            {
                // Reads overlapping this queued write wait until the worker installed
                // its overlay. Unrelated packets can continue; prefetch's non-draw gate
                // still orders future resource reads after the write.
                _commandThread!.PostRange(() => RecordCommandProcessorWrite(address, queued, kind), address, (ulong)queued.Length);
            }
            else
            {
                RunTagged(() => RecordCommandProcessorWrite(address, queued, kind));
            }
            return;
        }

        DrainDraws();
        if (source.IsEmpty)
        {
            return;
        }

        if (address == 0 || (ulong)source.Length > ulong.MaxValue - address)
        {
            throw Fatal($"A command-processor write has an invalid range: address=0x{address:X16} bytes={source.Length}.");
        }

        if (TryEncodeCommandProcessorWrite(address, source, kind))
        {
            Interlocked.Increment(ref _gpuTimedCommandProcessorWrites);
            return;
        }

        Interlocked.Increment(ref _hostCommandProcessorWrites);
        var bytes = source.ToArray();
        var landedKind = CommandProcessorWriteHistory.Enabled ? "landed:" + kind : null;
        var lazy = RidesNextCommit(clock: false);
        Interlocked.Increment(ref _owedHostLabels);
        QueueCompletion(() =>
        {
            try
            {
                if (!_memory.TryWrite(address, bytes))
                {
                    throw Fatal(
                        $"A command-processor write cannot be written to guest memory: address=0x{address:X16} bytes={bytes.Length}.");
                }

                _imageCache.InvalidateMemory(address, (ulong)bytes.Length);
                if (landedKind is not null && CommandProcessorWriteHistory.Enabled)
                {
                    CommandProcessorWriteHistory.Record(-1, address, bytes.Length, FirstValue(bytes), landedKind, deferred: true);
                }
            }
            finally
            {
                Interlocked.Decrement(ref _owedHostLabels);
            }
        }, lazy ? (address, bytes.Length) : null);
    }

    private bool TryEncodeCommandProcessorWrite(ulong address, ReadOnlySpan<byte> source, string kind)
    {
        using var phaseScope = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.CommandProcessorWrite);
        if (!GpuTimedLabels)
        {
            return false;
        }

        if (!MayEncodeLabelOnGpu(OrderedEndOfPipe, Volatile.Read(ref _owedHostLabels)))
        {
            CountClockForcedLabel();
            return false;
        }

        var size = (ulong)source.Length;
        var destination = _guestMemory.ResolveWritable(address, size);
        if (!destination.IsValid)
        {
            MetalV2Diagnostics.WarnOnce("cp-write-without-window",
                $"A command-processor write has no window, so a completion publishes it: address=0x{address:X16} bytes={size} kind={kind}.");
            return false;
        }

        var staging = _guestMemory.Allocate(size, sizeof(ulong), out var payload);
        if (!staging.IsValid)
        {
            return false;
        }

        source.CopyTo(payload);
        var copy = new DeferredLabelCopy(
            staging.Buffer, staging.Offset, destination.Buffer, destination.Offset, source.Length, address);
        var recordHistory = CommandProcessorWriteHistory.Enabled;
        var serial = EncodeOrderedCopy(in copy, address, recordHistory ? "landed:" + kind : null,
            recordHistory ? FirstValue(source) : 0, MetalGpuWriteOrigin.Copy, "cp write", kind, size);

        // Parse-time readers see the bytes as written until they land (MetalPendingWriteOverlay).
        using var bookkeepingScope = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(
            SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.CommandCpBookkeeping);
        if (_cpOverlay.Count >= MaxOverlayEntriesBeforeRetire)
        {
            _cpOverlay.Retire(CompletedSerial);
        }

        _cpOverlay.Add(address, source, serial);
        return true;
    }

    /// <summary>
    /// <paramref name="count"/> qword stores of one value, <paramref name="strideBytes"/> apart,
    /// recorded as one write: the same bytes land at the same places, in the same order, as
    /// <paramref name="count"/> calls to <see cref="RecordCommandProcessorWrite"/>.
    /// </summary>
    /// <remarks>
    /// Nothing is recorded between the stores, so they sit back to back at the one place in the
    /// stream a single store would (one staging block, one set of blits, one overlay hold). Any
    /// case the batch does not cover - a range with no window, no staging room, a label still owed
    /// by a completion, the clobber diagnostic - records them one by one before anything has been
    /// recorded. SHARPEMU_GPU_CP_WRITE_BATCH=0 always does.
    /// </remarks>
    public void RecordCommandProcessorQwords(ulong address, int count, ulong strideBytes, ulong value, string kind)
    {
        if (count <= 0) return;
        if (OnCommandThread)
        {
            // Publish the whole conservative hull before the producer advances. Reads in a
            // sparse gap may wait unnecessarily, but no following overlapping read can miss
            // the worker installing these writes. Invalid/wrapping runs keep the synchronous
            // path and its original store-by-store failure/partial-write behavior.
            if (QueueCommandProcessorWrites && address != 0 && address <= ulong.MaxValue - sizeof(ulong) &&
                (count == 1 || strideBytes <= (ulong.MaxValue - address - sizeof(ulong)) / (ulong)(count - 1)))
            {
                var size = (ulong)(count - 1) * strideBytes + sizeof(ulong);
                _commandThread!.PostRange(() => RecordCommandProcessorQwords(address, count, strideBytes, value, kind), address, size);
            }
            else
            {
                RunTagged(() => RecordCommandProcessorQwords(address, count, strideBytes, value, kind));
            }
            return;
        }

        DrainDraws();

        if (count > 1 && strideBytes >= sizeof(ulong) && address != 0 && CommandProcessorWriteBatching.Enabled &&
            !GpuClobberDiagnostics.Enabled &&
            strideBytes <= (ulong.MaxValue - address - sizeof(ulong)) / (ulong)(count - 1) &&
            TryEncodeCommandProcessorQwords(address, count, strideBytes, value, kind))
        {
            Interlocked.Add(ref _gpuTimedCommandProcessorWrites, count);
            _batchedCommandProcessorRuns++;
            return;
        }

        Span<byte> bytes = stackalloc byte[sizeof(ulong)];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt64LittleEndian(bytes, value);
        for (var index = 0; index < count; index++)
        {
            RecordCommandProcessorWrite(address + (ulong)index * strideBytes, bytes, kind);
        }
    }

    private DeferredLabelCopy[] _cpBatchCopies = new DeferredLabelCopy[32];

    private long _batchedCommandProcessorRuns;

    /// <summary>Qword runs recorded as one write rather than store by store.</summary>
    internal long BatchedCommandProcessorRuns => _batchedCommandProcessorRuns;

    private bool TryEncodeCommandProcessorQwords(ulong address, int count, ulong strideBytes, ulong value, string kind)
    {
        using var phaseScope = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.CommandProcessorWrite);
        // Declined, not counted: the one-by-one path that follows counts each store it forces.
        if (!GpuTimedLabels || !MayEncodeLabelOnGpu(OrderedEndOfPipe, Volatile.Read(ref _owedHostLabels)))
        {
            return false;
        }

        if (_cpBatchCopies.Length < count)
        {
            _cpBatchCopies = new DeferredLabelCopy[count];
        }

        var copies = _cpBatchCopies.AsSpan(0, count);
        for (var index = 0; index < count; index++)
        {
            var destination = _guestMemory.ResolveWritable(address + (ulong)index * strideBytes, sizeof(ulong));
            if (!destination.IsValid)
            {
                return false;
            }

            copies[index] = new DeferredLabelCopy(
                0, 0, destination.Buffer, destination.Offset, sizeof(ulong), address + (ulong)index * strideBytes);
        }

        var staging = _guestMemory.Allocate((ulong)count * sizeof(ulong), sizeof(ulong), out var payload);
        if (!staging.IsValid)
        {
            return false;
        }

        for (var index = 0; index < count; index++)
        {
            System.Buffers.Binary.BinaryPrimitives.WriteUInt64LittleEndian(payload[(index * sizeof(ulong))..], value);
            copies[index] = copies[index] with
            {
                StagingBuffer = staging.Buffer,
                StagingOffset = staging.Offset + (ulong)index * sizeof(ulong),
            };
        }

        var serial = EncodeOrderedCopies(copies, address, null, 0, MetalGpuWriteOrigin.Copy,
            "cp write", kind, (ulong)count * sizeof(ulong));

        // Parse-time readers see the bytes as written until they land (MetalPendingWriteOverlay).
        using (SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.CommandCpBookkeeping))
        {
            if (_cpOverlay.Count >= MaxOverlayEntriesBeforeRetire)
            {
                _cpOverlay.Retire(CompletedSerial);
            }

            _cpOverlay.AddQwords(address, count, strideBytes, value, serial);
        }

        if (CommandProcessorWriteBatching.Verify)
        {
            VerifyCommandProcessorQwords(address, count, strideBytes, value);
        }

        return true;
    }

    /// <summary>Stores whose overlay readback disagreed with what was recorded; zero is the invariant.</summary>
    internal static long CommandProcessorBatchVerifyMismatches;

    // SHARPEMU_GPU_CP_WRITE_BATCH_VERIFY: a parse-time read of each store sees the staged value.
    private void VerifyCommandProcessorQwords(ulong address, int count, ulong strideBytes, ulong value)
    {
        Span<byte> read = stackalloc byte[sizeof(ulong)];
        for (var index = 0; index < count; index++)
        {
            var at = address + (ulong)index * strideBytes;
            read.Clear();
            _cpOverlay.Apply(at, read);
            if (System.Buffers.Binary.BinaryPrimitives.ReadUInt64LittleEndian(read) != value)
            {
                Interlocked.Increment(ref CommandProcessorBatchVerifyMismatches);
                Console.Error.WriteLine(
                    $"[METAL-V2][VERIFY] A batched command-processor write is not in the overlay: address=0x{at:X16} expected=0x{value:X16}.");
            }
        }
    }

    private const int MaxOverlayEntriesBeforeRetire = 256;

    internal bool RenderEncoderOpen => _renderEncoder != 0;

    internal bool ComputeEncoderOpen => _computeEncoder != 0;

    /// <summary>
    /// Whether a copy recorded while a compute encoder is open waits for its end instead of
    /// ending it (SHARPEMU_METAL_V2_DEFER_COMPUTE_COPIES).
    /// </summary>
    internal bool DefersCopiesInCompute { get; set; } = MetalV2Diagnostics.DeferComputeCopies;

    internal int DeferredLabelCopyCount => _deferredLabelCopies.Count;

    // The bytes of command-processor writes encoded but not landed, by address.
    private readonly MetalPendingWriteOverlay _cpOverlay = new(MetalV2Diagnostics.OverlayPageFilter);

    /// <summary>
    /// A host write made at parse time (the interpreter's own stores on a queue that writes
    /// as parsed) is newer than any command-processor bytes still owed there.
    /// </summary>
    public bool TryWriteGuest(ulong address, ReadOnlySpan<byte> source, string kind = "WRITE_BYTES")
    {
        if (OnCommandThread)
        {
            var bytes = source.ToArray();
            var written = false;
            RunTagged(() => written = TryWriteGuest(address, bytes, kind));
            return written;
        }

        DrainDraws();
        _ = kind;
        _cpOverlay.Forget(address, (ulong)source.Length);
        _labelOverlay.Forget(address, (ulong)source.Length);
        return _memory.TryWrite(address, source);
    }

    private static ulong FirstValue(ReadOnlySpan<byte> bytes) => bytes.Length switch
    {
        >= sizeof(ulong) when bytes.Length != sizeof(uint) => System.Buffers.Binary.BinaryPrimitives.ReadUInt64LittleEndian(bytes),
        >= sizeof(uint) => System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(bytes),
        _ => 0,
    };

    // A fill this small is a fence reset or a slot clear, and is staged and copied like a
    // label so it neither ends the open pass nor lands out of order. A larger one is a GPU
    // fill, which ends the pass but is still recorded where the stream has reached; the
    // host fast path, which writes as parsed, is what the ordered path leaves out.
    private const ulong MaxStagedCommandProcessorFillBytes = 4096;

    /// <summary>A DMA fill of guest memory that lands in record order with the labels.</summary>
    public void FillBufferInOrder(ulong address, ulong size, uint value)
    {
        if (OnCommandThread)
        {
            _commandThread!.PostRange(() => FillBufferInOrder(address, size, value), address, size);
            return;
        }

        DrainDraws();
        if (size == 0)
        {
            return;
        }

        if (size <= MaxStagedCommandProcessorFillBytes && size % sizeof(uint) == 0)
        {
            Span<uint> pattern = stackalloc uint[(int)(MaxStagedCommandProcessorFillBytes / sizeof(uint))];
            pattern = pattern[..(int)(size / sizeof(uint))];
            pattern.Fill(value);
            RecordCommandProcessorWrite(address, System.Runtime.InteropServices.MemoryMarshal.AsBytes(pattern), "DMA_FILL");
            return;
        }

        FillGuestMemory(address, size, value, allowHost: false);
    }

    /// <summary>A DMA copy between guest ranges that lands in record order with the labels.</summary>
    public void CopyBufferInOrder(ulong destination, ulong source, ulong size)
    {
        if (OnCommandThread)
        {
            _commandThread!.PostRange(() => CopyBufferInOrder(destination, source, size), destination, size);
            return;
        }

        DrainDraws();
        if (size != 0)
        {
            CopyGuestMemory(destination, source, size, allowHost: false);
        }
    }

    /// <summary>Whether the queues' command-processor writes are recorded here.</summary>
    public bool DefersCommandProcessorWrites => DefersEndOfPipeLabels && !CommandProcessorWritesAtParse;

    // SHARPEMU_METAL_V2_CP_WRITES=parse writes WRITE_DATA, DMA fills and copies and COPY_DATA
    // immediates into guest memory as the queues parse them, as before. They then
    // reach the guest ahead of every label still owed, which is what let Demon's Souls
    // recycle command memory a late label then landed in.
    internal bool CommandProcessorWritesAtParse { get; set; } = CommandProcessorWritesAtParseDefault;

    private static readonly bool CommandProcessorWritesAtParseDefault =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_CP_WRITES") == "parse";

    private long _gpuTimedCommandProcessorWrites;
    private long _hostCommandProcessorWrites;

    /// <summary>Command-processor writes the GPU made in order, and ones a completion made.</summary>
    public (long GpuTimed, long Host) TakeCommandProcessorWriteCounters() =>
        (Interlocked.Exchange(ref _gpuTimedCommandProcessorWrites, 0), Interlocked.Exchange(ref _hostCommandProcessorWrites, 0));

    private static int EndOfPipeLabelBytes(EndOfPipeWriteKind kind) => kind switch
    {
        EndOfPipeWriteKind.Write32 or EndOfPipeWriteKind.WriteBack32 or
        EndOfPipeWriteKind.Interrupt32 or EndOfPipeWriteKind.InterruptWriteBack32 or
        EndOfPipeWriteKind.FlipWithWrite32 or EndOfPipeWriteKind.FlipWithInterruptWriteBack32 => 4,
        EndOfPipeWriteKind.Write64 or EndOfPipeWriteKind.WriteBack64 or
        EndOfPipeWriteKind.Interrupt64 or EndOfPipeWriteKind.InterruptWriteBack64 or
        EndOfPipeWriteKind.ClockWrite or EndOfPipeWriteKind.ClockWriteBack => 8,
        _ => 0,
    };

    /// <summary>
    /// Commits a command buffer that carries label bytes nothing else will flush.
    /// </summary>
    /// <remarks>
    /// A label written by the GPU lands when its command buffer is committed, and the guest
    /// thread polling it is not something this host can see. The worker calls this when it
    /// runs out of runnable work: a guest waiting on a label is exactly the case where the
    /// stream stops advancing, so one commit per park publishes every label at once instead
    /// of one commit per label.
    /// </remarks>
    private void FlushPendingLabelWrites()
    {
        if (_pendingLabelWrites == 0)
        {
            return;
        }

        FlushCommandBuffer(MetalFlushCause.Labels);
    }

    // A label copy waiting for a boundary between encoders. Its staging allocation lives
    // in this submission's ring, so the copy has to be encoded before the command buffer
    // is committed, which is the last boundary there is.
    private readonly record struct DeferredLabelCopy(
        nint StagingBuffer,
        ulong StagingOffset,
        nint DestinationBuffer,
        ulong DestinationOffset,
        int Bytes,
        ulong GuestAddress);

    private readonly List<DeferredLabelCopy> _deferredLabelCopies = [];
    private bool _encodingDeferredLabels;
    private static readonly bool DeferredLabelFilterEnabled =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_LABEL_PAGE_FILTER") != "0";
    private readonly MetalPageRangeFilter _deferredGuestPages = new();
    private readonly MetalPageRangeFilter _deferredBufferPages = new();
    private long _labelLookupQueries, _labelHullRejects, _labelPageRejects, _labelExactScans, _labelEntriesInspected;
    private int _labelMaxHeldCopies;

    private void NoteLabelScan(int entries)
    {
        if (!SharpEmu.Libs.VideoOut.RenderPhaseProfile.Enabled) return;
        _labelExactScans++;
        _labelEntriesInspected += entries;
    }

    private string TakeDeferredLabelLookupStats()
    {
        if (!SharpEmu.Libs.VideoOut.RenderPhaseProfile.Enabled) return "disabled";
        var result = $"queries:{_labelLookupQueries},hull_rejects:{_labelHullRejects},page_rejects:{_labelPageRejects}," +
            $"scans:{_labelExactScans},entries:{_labelEntriesInspected},max_held:{_labelMaxHeldCopies}";
        _labelLookupQueries = _labelHullRejects = _labelPageRejects = _labelExactScans = _labelEntriesInspected = 0;
        _labelMaxHeldCopies = _deferredLabelCopies.Count;
        return result;
    }

    // The span of guest bytes, and of destination bytes when they all go to one buffer
    // (_deferredLabelBuffer, -1 for several), the held copies land in: every storage binding
    // asks DeferredLabelCopiesTouch, and with copies held to the commit a scan of all of them
    // per binding grew with the command buffer. Bindings outside the span need no scan.
    private ulong _deferredLabelLow = ulong.MaxValue;
    private ulong _deferredLabelHigh;
    private nint _deferredLabelBuffer;
    private ulong _deferredLabelDestinationLow = ulong.MaxValue;
    private ulong _deferredLabelDestinationHigh;

    private void HoldLabelCopies(ReadOnlySpan<DeferredLabelCopy> copies)
    {
        foreach (ref readonly var copy in copies)
        {
            _deferredLabelLow = Math.Min(_deferredLabelLow, copy.GuestAddress);
            _deferredLabelHigh = Math.Max(_deferredLabelHigh, copy.GuestAddress + (ulong)copy.Bytes);
            _deferredLabelBuffer = _deferredLabelCopies.Count == 0 && _deferredLabelBuffer == 0
                ? copy.DestinationBuffer
                : _deferredLabelBuffer == copy.DestinationBuffer ? _deferredLabelBuffer : -1;
            _deferredLabelDestinationLow = Math.Min(_deferredLabelDestinationLow, copy.DestinationOffset);
            _deferredLabelDestinationHigh = Math.Max(_deferredLabelDestinationHigh, copy.DestinationOffset + (ulong)copy.Bytes);
            _deferredLabelCopies.Add(copy);
            if (DeferredLabelFilterEnabled)
            {
                _deferredGuestPages.Add(copy.GuestAddress, (ulong)copy.Bytes);
                _deferredBufferPages.Add(copy.DestinationOffset, (ulong)copy.Bytes, unchecked((ulong)copy.DestinationBuffer));
            }
            if (SharpEmu.Libs.VideoOut.RenderPhaseProfile.Enabled)
                _labelMaxHeldCopies = Math.Max(_labelMaxHeldCopies, _deferredLabelCopies.Count);
        }
    }

    private void ClearHeldLabelCopies()
    {
        _deferredLabelCopies.Clear();
        if (DeferredLabelFilterEnabled)
        {
            _deferredGuestPages.Clear();
            _deferredBufferPages.Clear();
        }
        _deferredLabelLow = ulong.MaxValue;
        _deferredLabelHigh = 0;
        _deferredLabelBuffer = 0;
        _deferredLabelDestinationLow = ulong.MaxValue;
        _deferredLabelDestinationHigh = 0;
    }

    // Label copies wait for the command buffer's commit instead of the next encoder boundary,
    // and land there behind one barrier. The barrier is a shared event (EncodeLabelBarrier),
    // which publishes the earlier stores before the label but costs far more than the encoder
    // boundary it sits at: GTA V's prologue wrote about 170 labels a frame, each behind its own
    // barrier, and ran at 16 fps against 22.5 with no barrier at all. Work that binds a held
    // copy's bytes lands it first (OrderAfterDeferredCopies), as does a depth-only join or a
    // transfer sharing the compute encoder over them, and a commit lands whatever is left. A
    // label that lands later still only says that the work before it has finished.
    // SHARPEMU_METAL_V2_LABELS_AT_COMMIT=0 lands them at each encoder boundary again.
    internal static readonly bool DefaultLabelsAtCommit =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_LABELS_AT_COMMIT") != "0";

    internal bool LabelsAtCommit { get; set; } = DefaultLabelsAtCommit;

    // Set by whatever needs the held copies in the stream now; the next encode lands them.
    private bool _landDeferredLabels;
    private long _deferredLabelWrites;

    /// <summary>Labels whose bytes waited for a boundary instead of ending a pass.</summary>
    public long TakeDeferredLabelWrites() => Interlocked.Exchange(ref _deferredLabelWrites, 0);

    // SHARPEMU_METAL_V2_LABEL_BARRIER=0 encodes label copies with no barrier ahead of them;
    // =event uses a device-only MTLEvent, which orders the copy after the work but does not
    // make that work's stores visible to the CPU first (see EncodeLabelBarrier).
    private static readonly string? LabelBarrierMode =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_LABEL_BARRIER");

    private static readonly bool LabelBarrierEnabled = LabelBarrierMode != "0";

    private static readonly bool LabelBarrierShared = LabelBarrierMode != "event";

    private static readonly nint SelEncodeWaitForEvent = MetalNative.Selector("encodeWaitForEvent:value:");
    private static readonly nint SelNewEvent = MetalNative.Selector("newEvent");

    private nint _labelBarrierEvent;
    private ulong _labelBarrierValue;
    private long _labelBarriers;

    // The command buffer the current event belongs to, its serial, and the events of earlier
    // command buffers, released once those are done.
    private nint _labelBarrierCommandBuffer;
    private ulong _labelBarrierSerial;
    private readonly List<(nint Event, ulong Serial)> _retiredLabelBarrierEvents = [];
    private static readonly nint SelReleaseLabelEvent = MetalNative.Selector("release");

    /// <summary>Barriers encoded ahead of label copies since the last call.</summary>
    internal long TakeLabelBarriers() => Interlocked.Exchange(ref _labelBarriers, 0);

    /// <summary>The event label barriers signal and wait on; zero until the first one.</summary>
    internal nint LabelBarrierEvent => _labelBarrierEvent;

    // The windows are untracked, so Metal orders nothing between the encoders of a
    // command buffer: Apple silicon starts an encoder while the ones before it are still
    // running (the pass profile shows compute encoders starting more than a millisecond
    // before the render pass ahead of them ends). A label copy is an encoder of its own,
    // and without this it could land while the work it reports was still storing. A guest
    // that frees memory on the label then has those stores land in what it freed: Silent
    // Hill's virtual-texture feedback copy wrote 0xFFFFFFFF over a freed block's free-list
    // link and the allocator crashed on the render thread. Signalling an event once all
    // the earlier work has finished and waiting for it makes the copy wait for that work.
    //
    // The event is a shared one because finishing is not the same as being seen. A shader's
    // stores reach memory the CPU reads tens to hundreds of microseconds after the work
    // ends, and a device-only MTLEvent lets the copy run in that gap: measured on an M-series
    // GPU, a label copied behind one was in memory before a fragment shader's earlier store
    // in a quarter of tries and before a compute store in one in twenty, and every CPU store
    // made on seeing the label was then overwritten by the late one. Silent Hill's render
    // thread freed a block that way and its free-list link became feedback-buffer words
    // (0x0400E00EFFFFFFFF). Signalling an MTLSharedEvent - one the CPU can observe - makes
    // the earlier stores visible first; none landed late in hundreds of tries.
    private void EncodeLabelBarrier(nint commandBuffer)
    {
        if (!LabelBarrierEnabled)
        {
            return;
        }

        EncodeEventBarrier(commandBuffer);
        Interlocked.Increment(ref _labelBarriers);
    }

    // Work encoded after this point starts only when everything encoded before it in the
    // command buffer has finished and its stores are visible.
    private void EncodeEventBarrier(nint commandBuffer)
    {
        // One event per command buffer. Command buffers overlap on the GPU, and with one event
        // across all of them a later buffer's signal of a lower value could land between an
        // earlier one's signal and its wait, which then waited for a value no one signals
        // again: Black Myth: Wukong's save load stopped with every guest thread waiting on a
        // label and the GPU idle. Within one command buffer the barriers run in order.
        if (_labelBarrierEvent == 0 || commandBuffer != _labelBarrierCommandBuffer)
        {
            ReleaseCompletedLabelBarrierEvents();
            if (_labelBarrierEvent != 0)
            {
                _retiredLabelBarrierEvents.Add((_labelBarrierEvent, _labelBarrierSerial));
            }

            _labelBarrierEvent = MetalNative.Send(_device, LabelBarrierShared ? SelNewSharedEvent : SelNewEvent);
            if (_labelBarrierEvent == 0)
            {
                throw GpuFatal.Fatal("Metal v2 needs an MTLEvent to order work after the work before it.");
            }

            _labelBarrierCommandBuffer = commandBuffer;
            _labelBarrierSerial = SubmissionSerial;
            _labelBarrierValue = 0;
        }

        var value = ++_labelBarrierValue;
        MetalV2Native.SendEventValue(commandBuffer, SelEncodeSignalEvent, _labelBarrierEvent, value);
        MetalV2Native.SendEventValue(commandBuffer, SelEncodeWaitForEvent, _labelBarrierEvent, value);
    }

    private void ReleaseCompletedLabelBarrierEvents()
    {
        if (_retiredLabelBarrierEvents.Count == 0)
        {
            return;
        }

        var completed = CompletedSerial;
        _retiredLabelBarrierEvents.RemoveAll(retired =>
        {
            if (retired.Serial > completed)
            {
                return false;
            }

            MetalNative.SendVoid(retired.Event, SelReleaseLabelEvent);
            return true;
        });
    }

    private void EncodeLabelCopies(nint commandBuffer, ReadOnlySpan<DeferredLabelCopy> copies)
    {
        using var encodeScope = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(
            SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.CommandLabelEncode);
        EncodeLabelBarrier(commandBuffer);
        var encoder = MetalNative.Send(commandBuffer, SelBlitEncoder);
        MetalMemoryTelemetry.BlitEncoders.Add();
        if (encoder == 0)
        {
            throw GpuFatal.Fatal("A Metal blit encoder could not be created for an end-of-pipe label.");
        }

        WaitForPreviousEncoder(encoder, render: false);
        try
        {
            foreach (ref readonly var copy in copies)
            {
                MetalNative.SendCopyBuffer(
                    encoder,
                    SelCopyBuffer,
                    copy.StagingBuffer,
                    (nuint)copy.StagingOffset,
                    copy.DestinationBuffer,
                    (nuint)copy.DestinationOffset,
                    (nuint)copy.Bytes);
            }
        }
        finally
        {
            UpdateEncoderFence(encoder, render: false);
            MetalNative.SendVoid(encoder, SelEndEncoding);
        }
    }

    /// <summary>
    /// Copies every label that waited for a boundary. Called where no encoder is open and
    /// before the next one is created, so the bytes land after the work the labels report
    /// and before whatever follows them.
    /// </summary>
    // Whether a copy waiting for the end of the open encoder lands in these guest bytes.
    private bool DeferredLabelCopiesTouch(ulong address, ulong size)
    {
        if (SharpEmu.Libs.VideoOut.RenderPhaseProfile.Enabled) _labelLookupQueries++;
        if (address >= _deferredLabelHigh || address + size <= _deferredLabelLow)
        {
            if (SharpEmu.Libs.VideoOut.RenderPhaseProfile.Enabled) _labelHullRejects++;
            return false;
        }
        if (DeferredLabelFilterEnabled && !_deferredGuestPages.MayOverlap(address, size))
        {
            if (SharpEmu.Libs.VideoOut.RenderPhaseProfile.Enabled) _labelPageRejects++;
            return false;
        }

        var copies = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(_deferredLabelCopies);
        for (var index = 0; index < copies.Length; index++)
        {
            ref readonly var copy = ref copies[index];
            if (copy.GuestAddress < address + size && address < copy.GuestAddress + (ulong)copy.Bytes)
            {
                NoteLabelScan(index + 1);
                return true;
            }
        }

        NoteLabelScan(copies.Length);
        return false;
    }

    // Buffer identity is part of the page key: aliases with different guest addresses
    // must still land before a binding to the same native buffer bytes.
    private bool DeferredLabelCopiesTouch(in MetalGuestRange range)
    {
        if (SharpEmu.Libs.VideoOut.RenderPhaseProfile.Enabled) _labelLookupQueries++;
        if (_deferredLabelBuffer != -1 &&
            (range.Buffer != _deferredLabelBuffer ||
             range.Offset >= _deferredLabelDestinationHigh || range.Offset + range.Size <= _deferredLabelDestinationLow))
        {
            if (SharpEmu.Libs.VideoOut.RenderPhaseProfile.Enabled) _labelHullRejects++;
            return false;
        }
        if (DeferredLabelFilterEnabled &&
            !_deferredBufferPages.MayOverlap(range.Offset, range.Size, unchecked((ulong)range.Buffer)))
        {
            if (SharpEmu.Libs.VideoOut.RenderPhaseProfile.Enabled) _labelPageRejects++;
            return false;
        }

        var copies = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(_deferredLabelCopies);
        for (var index = 0; index < copies.Length; index++)
        {
            ref readonly var copy = ref copies[index];
            if (copy.DestinationBuffer == range.Buffer &&
                copy.DestinationOffset < range.Offset + range.Size &&
                range.Offset < copy.DestinationOffset + (ulong)copy.Bytes)
            {
                NoteLabelScan(index + 1);
                return true;
            }
        }

        NoteLabelScan(copies.Length);
        return false;
    }

    /// <summary>
    /// A dispatch about to bind these guest bytes runs after the command-processor writes
    /// recorded before it: one still waiting for the open compute encoder to end would land
    /// after the dispatch instead, so the encoder ends here and the copies are encoded at
    /// that boundary. Draws need nothing here; beginning a pass ends the compute encoder.
    /// </summary>
    internal void OrderAfterDeferredCopies(ulong address, ulong size)
    {
        if (_deferredLabelCopies.Count == 0 || (_computeEncoder == 0 && !LabelsAtCommit) ||
            (size != ulong.MaxValue && !DeferredLabelCopiesTouch(address, size)))
        {
            return;
        }

        // Held to the commit, a copy whose bytes this work binds lands here instead: the work
        // reads or writes them, and must see the copy, or overwrite it, in stream order.
        _landDeferredLabels = true;
        if (_computeEncoder != 0 || _renderEncoder != 0)
        {
            EndEncoders(PassEndCause.Other);
        }

        EncodeDeferredLabelCopies();
    }

    private void EncodeDeferredLabelCopies()
    {
        if (_deferredLabelCopies.Count == 0 || _encodingDeferredLabels)
        {
            return;
        }

        if (LabelsAtCommit && !_landDeferredLabels)
        {
            return;
        }

        _landDeferredLabels = false;

        _encodingDeferredLabels = true;
        try
        {
            var copies = _deferredLabelCopies.Count;
            EncodeLabelCopies(
                EnsureCommandBuffer(),
                System.Runtime.InteropServices.CollectionsMarshal.AsSpan(_deferredLabelCopies));
            ClearHeldLabelCopies();
            // The barrier and the blit encoder are gap the pass profile cannot sample.
            if (PassProfileEnabled)
            {
                PassProfileNoteBetween($"label-copies n={copies}");
            }
        }
        finally
        {
            _encodingDeferredLabels = false;
        }
    }

    private ulong _pendingLabelWrites;
    private ulong _labelWakeSerial;

    // DSTRACE §4.7 LENC: how the label just recorded was encoded, and into which serial.
    private ulong _dsLabelDeferred;
    private ulong _dsLabelSerial;

    // DSTRACE §4.7 LLAND: the highest serial the host has been told completed, from either the
    // completion thread (label wake) or the worker (completion poll). One record per advance:
    // a label recorded into serial s has landed at the first LLAND whose serial is >= s.
    private long _dsLandedSerial;

    private void DsNoteLabelsLanded(ulong serial)
    {
        if (!DsTrace.Armed)
        {
            return;
        }

        var landed = (long)serial;
        var previous = Volatile.Read(ref _dsLandedSerial);
        while (landed > previous)
        {
            var seen = Interlocked.CompareExchange(ref _dsLandedSerial, landed, previous);
            if (seen == previous)
            {
                DsTrace.Emit(DsEv.LLAND, serial);
                return;
            }

            previous = seen;
        }
    }
    private long _gpuTimedLabels;
    private long _labelsWithoutWindow;

    /// <summary>Labels the GPU wrote itself, and labels with no window to write.</summary>
    public (long GpuTimed, long WithoutWindow) TakeLabelCounters() =>
        (Interlocked.Exchange(ref _gpuTimedLabels, 0), Interlocked.Exchange(ref _labelsWithoutWindow, 0));

    /// <summary>
    /// Writes an end-of-pipe label's bytes into guest memory. The interpreter leaves
    /// them to the host (<see cref="DefersEndOfPipeLabels"/>) because it runs ahead of
    /// the GPU by design: publishing a label when its packet is parsed tells the guest
    /// that work still queued has finished, and the guest then overwrites the constants
    /// and vertex data that work has yet to read. Superliminal's level rendered from
    /// whatever had replaced its view-projection matrix by the time the GPU got to it,
    /// which came out as a degenerate clip position and an empty G-buffer.
    /// </summary>
    private void PublishEndOfPipeLabel(in EndOfPipeWrite write)
    {
        var bytes = EndOfPipeLabelBytes(write.Kind);
        if (bytes == 0 || !DefersEndOfPipeLabel(write.Kind))
        {
            // No payload, or the interpreter already wrote these bytes.
            return;
        }

        if (write.Destination == 0 || (ulong)bytes > ulong.MaxValue - write.Destination)
        {
            throw Fatal(
                $"An end-of-pipe label has an invalid range: address=0x{write.Destination:X16} bytes={bytes}.");
        }

        Span<byte> payload = stackalloc byte[8];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt64LittleEndian(payload, write.Value);
        if (!_memory.TryWrite(write.Destination, payload[..bytes]))
        {
            throw Fatal(
                $"An end-of-pipe label cannot be written to guest memory: " +
                $"address=0x{write.Destination:X16} bytes={bytes}.");
        }

        if (CommandProcessorWriteHistory.Enabled)
        {
            CommandProcessorWriteHistory.Record(-1, write.Destination, bytes,
                bytes == sizeof(uint) ? (uint)write.Value : write.Value, GpuClobberDiagnostics.LandedKind(write.Kind), deferred: true);
        }
    }

    /// <summary>
    /// Publishes a label the GPU did not write, from the completion that reports its work,
    /// and stops owing it.
    /// </summary>
    private void PublishHostLabel(in EndOfPipeWrite label, bool owed)
    {
        try
        {
            if (label.ReferenceClock)
            {
                PublishClockAtCompletion(in label);
            }
            else
            {
                PublishEndOfPipeLabel(in label);
            }
        }
        finally
        {
            if (owed)
            {
                if (label.ReferenceClock)
                {
                    Interlocked.Decrement(ref _owedClockLabels);
                }

                Interlocked.Decrement(ref _owedHostLabels);
            }
        }
    }

    /// <summary>
    /// Publishes a clock label with the reference clock as it is now, from the completion
    /// of the work recorded before it.
    /// </summary>
    /// <remarks>
    /// The completion runs once the GPU signalled that submission, so the value is at least
    /// the time that work finished, plus the time the host took to notice. Completions run
    /// one at a time in submission order, so clocks published here never go backwards.
    /// </remarks>
    private void PublishClockAtCompletion(in EndOfPipeWrite label)
    {
        var sampled = label with { Value = GpuReferenceClock.Read() };
        PublishEndOfPipeLabel(in sampled);
        Interlocked.Increment(ref _clockCompletionLabels);
        if (EndOfPipeTrace)
        {
            Console.Error.WriteLine(
                $"[METAL-V2][EOP] clock kind={label.Kind} address=0x{label.Destination:X16} " +
                $"value=0x{sampled.Value:X16} parsed=0x{label.Value:X16} " +
                $"late_us={(long)(sampled.Value - label.Value) / 100} sampling=completion");
        }
    }

    // Labels a completion still has to publish; nonzero sends every later label there too.
    private long _owedHostLabels;
    private long _orderedHostLabels;

    /// <summary>Labels published by a completion only to stay behind an older owed one.</summary>
    public long TakeOrderedHostLabels() => Interlocked.Exchange(ref _orderedHostLabels, 0);

    /// <summary>
    /// End-of-pipe writes of a stream land in the order they were recorded.
    /// </summary>
    /// <remarks>
    /// A label the GPU cannot write - no window covers it, or the GPU copy was refused - is
    /// published by the completion of its submission, which runs after the GPU finished it
    /// and the host noticed. A later label the GPU writes itself lands as soon as the GPU
    /// reaches it, which is earlier, so without this the newer signal is in memory first:
    /// a guest that recycles memory on it (Demon's Souls gates command memory on GPU-written
    /// signals) then has the older label land in what it already reused. Completions run in
    /// submission order, so while any label is owed, later ones are published there as well.
    /// SHARPEMU_METAL_V2_EOP_ORDERED=0 lets a later label be written by the GPU regardless.
    /// </remarks>
    internal bool OrderedEndOfPipe { get; set; } = OrderedEndOfPipeDefault;

    private static readonly bool OrderedEndOfPipeDefault =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_EOP_ORDERED") != "0";

    internal static bool MayEncodeLabelOnGpu(bool ordered, long owedHostLabels) => !ordered || owedHostLabels == 0;

    // SHARPEMU_METAL_V2_GPU_LABELS=0 goes back to publishing every label from the
    // completion that reports its work. That is correct, but a
    // completion costs a command buffer flush and then waits to be told the submission
    // finished, so a title that writes a label per render pass pays both per pass:
    // measured at 9.5 fps against 24. The GPU writing the label where its packet sits
    // says the same thing and costs neither, which is why it is the default.
    private static readonly bool GpuTimedLabels =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_GPU_LABELS") != "0";

    /// <summary>
    /// The interpreter must never publish label bytes itself on this host: it parses a
    /// packet long before the GPU runs the work the label reports, and a guest told that
    /// work has finished reuses the memory it still has to read. See
    /// <see cref="TryEncodeEndOfPipeLabel"/> and <see cref="PublishEndOfPipeLabel"/>.
    /// </summary>
    public bool DefersEndOfPipeLabels => !PublishLabelsWhenParsed;

    // Keep conservative pairs by default. Precise GPU snapshots are opt-in while
    // native query ordering and runtime behavior are validated; the audit changes no dumps.
    public bool PublishesSyntheticOcclusionDumps => !NativeOcclusionEnabled;

    // A deferred label is encoded into the command buffer being recorded, before it is
    // committed, so it has landed once that submission completes.
    // On the command thread the buffer the label will ride is not open yet: the worker's serial is a
    // provisional tick (too early only makes a wait read memory again), and the worker gives the real
    // one when it records the label (RecordEndOfPipe).
    public ulong PendingLabelTick => !PendingLabelsVisible ? 0 : OnCommandThread ? Volatile.Read(ref _submissionSerial) : SubmissionSerial;

    private static readonly bool PendingLabelsVisible =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_PENDING_LABELS") != "0";

    /// <summary>
    /// Every end-of-pipe write waits for the work it reports, a clock included.
    /// </summary>
    /// <remarks>
    /// End-of-pipe writes of a queue retire in order, so a clock lands with the labels before
    /// it. Guests (Demon's Souls) gate memory reuse on it: published at parse time it let a
    /// command buffer be recycled while an older deferred label to it was still owed.
    /// SHARPEMU_METAL_V2_EOP_CLOCKS=parse opts back into publishing clocks as parsed.
    /// </remarks>
    public bool DefersEndOfPipeLabel(EndOfPipeWriteKind kind) =>
        DefersLabelKind(kind, DefersEndOfPipeLabels, ClockSampling == EndOfPipeClockSampling.Parse);

    internal static bool DefersLabelKind(EndOfPipeWriteKind kind, bool defersLabels, bool publishClocksWhenParsed) =>
        defersLabels &&
        (!publishClocksWhenParsed || kind is not (EndOfPipeWriteKind.ClockWrite or EndOfPipeWriteKind.ClockWriteBack));

    /// <summary>
    /// When a clock label's value is sampled: SHARPEMU_METAL_V2_EOP_CLOCKS=deferred (the
    /// default), =completion, =completion-strict or =parse.
    /// </summary>
    /// <remarks>
    /// An end-of-pipe timestamp is the time the work before it finished. The interpreter
    /// samples the clock as it parses the packet, which on this host is well before the GPU
    /// runs that work, so a frame bracketed by two such stamps measures how long the stream
    /// took to parse rather than to render. Demon's Souls feeds them to its dynamic
    /// resolution: it saw a GPU frame far below the real ~38 ms and stayed at 3840x2160.
    /// Completion leaves the label to the completion of the work before it, which samples
    /// the clock again. That costs what any host-published label costs: a command buffer
    /// flush unless the work already went out, the wait for the host to notice, and every
    /// later label and command-processor write sent through completions while the clock is
    /// owed, so a guest that sees a later fence never reads an older clock.
    /// Deferred lands the parse-time value in order with the labels, as the GPU reaches it;
    /// parse writes it as parsed. Measured on Demon's Souls (2026-10-15, runs
    /// goal-phase-9-clock-*): completion did not change the game's render resolution in
    /// the cinematic, title or character creation (the scene stays 2560x1440 and character
    /// creation 3840x2160 in both modes) while the worker's host waits doubled (host_wait_ms
    /// 1380 vs 730 per 2 s) and the title lost ~20% frame rate, so deferred is the default
    /// until gameplay shows the scaler reacting to completion-time stamps.
    /// </remarks>
    internal EndOfPipeClockSampling ClockSampling { get; set; } = ClockSamplingDefault;

    internal static EndOfPipeClockSampling ParseClockSampling(string? value) => value switch
    {
        "completion" => EndOfPipeClockSampling.Completion,
        "completion-strict" => EndOfPipeClockSampling.CompletionStrict,
        "parse" => EndOfPipeClockSampling.Parse,
        _ => EndOfPipeClockSampling.Deferred,
    };

    private static readonly EndOfPipeClockSampling ClockSamplingDefault =
        ParseClockSampling(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_EOP_CLOCKS"));

    internal static bool IsClockLabel(in EndOfPipeWrite write) =>
        write.ReferenceClock || write.Kind is EndOfPipeWriteKind.ClockWrite or EndOfPipeWriteKind.ClockWriteBack;

    private bool SamplesClockAtCompletion(in EndOfPipeWrite write) =>
        ClockSampling is EndOfPipeClockSampling.Completion or EndOfPipeClockSampling.CompletionStrict &&
        IsClockLabel(in write) &&
        EndOfPipeLabelBytes(write.Kind) != 0 && DefersEndOfPipeLabel(write.Kind);

    private string ClockSamplingTrace(in EndOfPipeWrite write) => IsClockLabel(in write)
        ? " sampling=" + (SamplesClockAtCompletion(in write) ? "completion"
            : DefersEndOfPipeLabel(write.Kind) ? "deferred" : "parse")
        : string.Empty;

    /// <summary>
    /// Whether a host label's completion may wait for the next commit instead of forcing one.
    /// </summary>
    /// <remarks>
    /// In completion mode a clock never commits the buffer: its completion rides the next
    /// commit that carries the work before it, and its value is when the GPU finished that
    /// whole submission. That can overstate the work by the rest of the buffer, but the
    /// frame's first and last stamps move the same way, so the gap between them still
    /// tracks the GPU frame. A label held behind an owed clock cannot land before the clock
    /// does, so committing for it buys nothing either. Commits still come from the frame
    /// flush, coalescing limits, the worker parking, and a wait packet polling the bytes.
    /// </remarks>
    private bool RidesNextCommit(bool clock) =>
        ClockSampling == EndOfPipeClockSampling.Completion &&
        (clock || Volatile.Read(ref _owedClockLabels) != 0);

    // Host labels in the open buffer whose completions wait for its commit; cleared by it.
    private readonly List<(ulong Address, int Bytes)> _lazyLabelRanges = [];
    private long _lazyLabelCompletions;

    /// <summary>Host label completions that rode the next commit instead of forcing one.</summary>
    public long TakeLazyLabelCompletions() => Interlocked.Exchange(ref _lazyLabelCompletions, 0);

    private bool CoversLazyLabel(ulong address, ulong size)
    {
        foreach (var (start, bytes) in _lazyLabelRanges)
        {
            if (address < start + (ulong)bytes && start < address + size)
            {
                return true;
            }
        }

        return false;
    }

    // Clock labels a completion owes; nonzero means a label refused the GPU is held behind one.
    private long _owedClockLabels;
    private long _clockCompletionLabels;
    private long _clockForcedLabels;

    private void CountClockForcedLabel()
    {
        if (Volatile.Read(ref _owedClockLabels) != 0)
        {
            Interlocked.Increment(ref _clockForcedLabels);
        }
    }

    /// <summary>
    /// Clock labels sampled and written at completion, and labels or command-processor
    /// writes sent through completions because a clock was owed.
    /// </summary>
    public (long Completion, long Forced) TakeClockLabelCounters() =>
        (Interlocked.Exchange(ref _clockCompletionLabels, 0), Interlocked.Exchange(ref _clockForcedLabels, 0));

    // SHARPEMU_METAL_V2_EOP_TRACE=1 names every label: what it is, where it goes, which
    // submission carries it and how long that submission took to land. A title that gives
    // up waiting for one is only diagnosable against that list.
    private static readonly bool EndOfPipeTrace =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_EOP_TRACE") == "1";

    // SHARPEMU_METAL_V2_EOP_IMMEDIATE=1 restores publishing a label as its packet is
    // parsed. It is wrong — it tells the guest that work still queued has finished — and
    // it is here to tell a title that depends on the early value apart from one that
    // merely runs slower without it.
    private static readonly bool PublishLabelsWhenParsed =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_EOP_IMMEDIATE") == "1";

    // The completion runs once the GPU finishes everything recorded so far, so the
    // command buffer is committed right away instead of waiting for the next flush.
    // An end-of-pipe write the guest waits on only has to wait for the GPU work
    // recorded before it. When every submission that carries work has already
    // finished, "after the GPU drains" is true now, and running the completion here
    // saves the guest a whole submission round trip: on this host that is tens of
    // milliseconds even when nothing is wrong, and far more when something is.
    // Ordering is kept by only doing it while no earlier completion is outstanding.
    private static readonly bool InlineCompletions =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_INLINE_EOP") != "0";

    private long _inlineCompletionCount;

    /// <summary>End-of-pipe completions that ran without a GPU round trip.</summary>
    public long TakeInlineCompletions() => Interlocked.Exchange(ref _inlineCompletionCount, 0);

    /// <param name="lazyLabel">
    /// A label whose completion may ride the next commit instead of forcing one (see
    /// <see cref="RidesNextCommit"/>): the worker commits it when it parks, and a wait
    /// packet polling these bytes commits it through the Poll cause.
    /// </param>
    private void QueueCompletion(Action work, (ulong Address, int Bytes)? lazyLabel = null)
    {
        if (InlineCompletions && LastWorkSerial <= CompletedSerial && _completions.PendingCount == 0)
        {
            Interlocked.Increment(ref _inlineCompletionCount);
            work();
            return;
        }

        // Nothing has been recorded since the last flush, so everything this completion
        // waits for is in the submission already committed: it rides that one's serial
        // and needs no flush of its own. A run of end-of-pipe labels with no work
        // between them is the common shape, and flushing once per label turned a frame
        // into hundreds of empty submissions.
        var pending = LastWorkSerial;
        var queued = System.Diagnostics.Stopwatch.GetTimestamp();
        if (pending != 0 && pending <= _previousSerial)
        {
            Enqueue(pending);
            return;
        }

        var serial = SubmissionSerial;
        Enqueue(serial);
        if (lazyLabel is { } open && pending == serial)
        {
            // The work before it is in the open buffer, which some commit takes out anyway.
            _lazyLabelRanges.Add(open);
            _pendingLabelWrites++;
            Interlocked.Increment(ref _lazyLabelCompletions);
            return;
        }

        FlushCommandBuffer(MetalFlushCause.Completion);
        return;

        void Enqueue(ulong on) => _completions.Enqueue(on, () =>
        {
            var ran = System.Diagnostics.Stopwatch.GetTimestamp();
            Interlocked.Add(ref _completionLatencyTicks, ran - queued);
            Interlocked.Increment(ref _completionCount);
            ReportSlowCompletion(on, queued, ran);
            work();
        });
    }

    // An end-of-pipe completion the guest is waiting on that took long enough to cost
    // frames. The two halves are what separate the causes: the GPU not having
    // finished the submission, and the host not having noticed that it had.
    private void ReportSlowCompletion(ulong serial, long queued, long ran)
    {
        var frequency = System.Diagnostics.Stopwatch.Frequency;
        if (ran - queued <= frequency / 20 || _slowCompletionReports >= 4)
        {
            return;
        }

        Interlocked.Increment(ref _slowCompletionReports);
        var observed = _completions.LastSignalObservedTimestamp;
        var noticedMs = observed == 0 ? -1.0 : (ran - observed) * 1000.0 / frequency;
        Console.Error.WriteLine(
            $"[METAL-V2][SLOW-EOP] serial={serial} latency_ms={(ran - queued) * 1000.0 / frequency:F1} " +
            $"gpu_ms={(observed == 0 ? -1.0 : (observed - queued) * 1000.0 / frequency):F1} " +
            $"noticed_ms={noticedMs:F1} signalled={_completions.SignaledValue} " +
            $"pending={_completions.PendingCount}");
    }

    private int _slowCompletionReports;

    /// <summary>Resets the per-window budget for slow end-of-pipe reports.</summary>
    public void ResetSlowCompletionReports() => Interlocked.Exchange(ref _slowCompletionReports, 0);

    private long _completionLatencyTicks;
    private long _completionCount;

    /// <summary>How long end-of-pipe completions waited for the GPU since the last read.</summary>
    public (long Count, double Milliseconds) TakeCompletionLatency()
    {
        var count = Interlocked.Exchange(ref _completionCount, 0);
        var ticks = Interlocked.Exchange(ref _completionLatencyTicks, 0);
        return (count, ticks * 1000.0 / System.Diagnostics.Stopwatch.Frequency);
    }

    private readonly EndOfPipeEvents _endOfPipeEvents = new();

    public void TriggerInterrupt(int eventId, uint contextId)
    {
        if (PostToWorker(() => TriggerInterrupt(eventId, contextId)))
        {
            return;
        }

        DrainDraws();
        _endOfPipeEvents.TriggerInterrupt(eventId, contextId);
    }

    public bool HasFlipSlot()
    {
        if (OnCommandThread)
        {
            var hasSlot = false;
            RunTagged(() => hasSlot = HasFlipSlot());
            return hasSlot;
        }

        return _flipTarget?.HasFlipSlot() ?? true;
    }

    // Where each flip this interval ended up. Every flip that reaches CaptureFlip counts
    // as an attempt and then lands in exactly one of the three outcomes, so attempts
    // staying at zero means the guest never asked for a flip at all.
    private long _flipAttempts;
    private long _flipsWithoutBuffer;
    private long _flipsDiscarded;
    private long _flipsPublished;
    private long _flipsRefreshed;

    internal readonly record struct FlipCounters(
        long Attempts, long WithoutBuffer, long Discarded, long Published, long Refreshed);

    internal FlipCounters TakeFlipCounters() => new(
        Interlocked.Exchange(ref _flipAttempts, 0),
        Interlocked.Exchange(ref _flipsWithoutBuffer, 0),
        Interlocked.Exchange(ref _flipsDiscarded, 0),
        Interlocked.Exchange(ref _flipsPublished, 0),
        Interlocked.Exchange(ref _flipsRefreshed, 0));

    public ulong PrepareFlip(int handle, int index, int flipMode, long flipArgument)
    {
        if (OnCommandThread)
        {
            var request = 0UL;
            RunTagged(() => request = PrepareFlip(handle, index, flipMode, flipArgument));
            return request;
        }

        DrainDraws();
        var result = VideoOutBackendServices.TryReserveFlipRequest(
            handle, index, flipMode, flipArgument, gpuQueued: true, out var requestId);
        if (result != 0)
        {
            throw Fatal(
                $"Could not submit the GPU flip: result=0x{result:X8} handle={handle} index={index} mode={flipMode} arg={flipArgument}.");
        }

        if (requestId == 0)
        {
            throw Fatal("The GPU flip submission returned a request id of zero.");
        }

        try
        {
            CaptureFlip(handle, index, requestId, flipMode, flipArgument);
        }
        catch
        {
            VideoOutBackendServices.CancelFlip(requestId);
            throw;
        }

        return requestId;
    }

    public bool IsFlipDone(int handle, int index) => VideoOutBackendServices.IsFlipDone(handle, index);

    public void PrepareCpuFlip(int handle, int index, ulong requestId)
    {
        if (RunOnWorker(() => PrepareCpuFlip(handle, index, requestId)))
        {
            return;
        }

        DrainDraws();
        CaptureFlip(handle, index, requestId, flipMode: 0, flipArg: 0);
    }

    // The display surface is a cached image like any other; the flip hands its
    // MTLTexture to the presenter, so no pixel ever travels through the CPU.
    // Where a flip stops before it reaches the presenter. A title that shows a
    // frame while guest_fps stays zero is presenting through the CPU path, and
    // these say which step dropped the GPU one.
    internal static long FlipsCaptured;
    internal static long FlipsWithoutBuffer;
    internal static long FlipsDiscarded;
    internal static long FlipsPublished;

    private void CaptureFlip(int handle, int index, ulong requestId, int flipMode, long flipArg)
    {
        Interlocked.Increment(ref _flipAttempts);
        Interlocked.Increment(ref FlipsCaptured);
        NoteFlipForCompilingDraws();
        // The one point in the frame that is outside a flush, so the safe place to read
        // back everything the passes since the last flip wrote.
        DrainPendingDumps();
        if (VideoOutBackendServices.ReleaseNoBufferFlip(index, requestId))
        {
            // The guest flipped without naming a new buffer. Legitimate on its own, but a
            // title whose every flip goes this way never publishes a frame, and that used
            // to look exactly like no flip arriving at all.
            Interlocked.Increment(ref _flipsWithoutBuffer);
            Interlocked.Increment(ref FlipsWithoutBuffer);
            MetalV2Diagnostics.InfoOnce(
                "flip-no-buffer",
                $"A flip named no display buffer and was released without a frame: " +
                $"handle={handle} index={index} request={requestId}.");
            return;
        }

        if (!VideoOutBackendServices.TryGetDisplayBufferInfo(handle, index, out var displayBuffer))
        {
            throw Fatal(
                $"The flip names a display buffer that is not registered: handle={handle} index={index} request={requestId}.");
        }

        var surface = new DisplaySurfaceWords(
            displayBuffer.Address,
            0,
            displayBuffer.PixelFormat,
            displayBuffer.Width,
            displayBuffer.Height,
            displayBuffer.TilingMode,
            displayBuffer.Option,
            0,
            0,
            false);
        var request = ImageRequestBuilders.DisplaySurface(surface);
        var imageIdentifier = _imageCache.FindImage(ref request);
        var image = _imageCache.GetImage(imageIdentifier);
        image.Uses.VideoOut = true;
        // A display surface the GPU just composited into must not be refilled from guest
        // memory on the way to the screen: those bytes are whatever was there before the
        // composite, and uploading them replaces the frame with the old one. Report it
        // when it happens rather than presenting the result silently.
        if (image.IsCpuDirty)
        {
            Interlocked.Increment(ref _flipsRefreshed);
            MetalV2Diagnostics.WarnOnce(
                "flip-refresh",
                $"A flip's display surface was CPU-dirty, so it is uploaded from guest memory before " +
                $"presenting: cause={image.DirtyCause} address=0x{displayBuffer.Address:X16} " +
                $"{displayBuffer.Width}x{displayBuffer.Height} gpuWritten={image.IsGpuModified}.");
        }

        // Before and after, so an upload that replaces a composited frame with the bytes
        // that were in guest memory before it shows up as the two differing.
        if (_dumpThisFlip) { DumpTexture(image.Texture, "flip-before-refresh"); }
        _imageCache.RefreshImage(imageIdentifier);
        if (_dumpThisFlip) { DumpTexture(image.Texture, "flip-after-refresh"); }
        SampleFlipLuma(image.Texture);
        var texture = PresentationViewOf(imageIdentifier, image, in request);
        var serial = SubmissionSerial;
        FlushCommandBuffer(PassEndCause.Present);
        if (_flipTarget is null || texture == 0)
        {
            // The display surface resolved to no presentable texture, so there is nothing
            // to hand the presenter. Silent until now, which made it indistinguishable
            // from a title that never flips at all.
            Interlocked.Increment(ref _flipsDiscarded);
            Interlocked.Increment(ref FlipsDiscarded);
            MetalV2Diagnostics.WarnOnce(
                "flip-discarded",
                $"A flip had no presentable texture and was discarded: handle={handle} index={index} " +
                $"request={requestId} target={(_flipTarget is null ? "none" : "present")} " +
                $"address=0x{displayBuffer.Address:X16} {displayBuffer.Width}x{displayBuffer.Height} " +
                $"format={displayBuffer.PixelFormat} tiling={displayBuffer.TilingMode} " +
                $"view={request.View.Format}.");
            VideoOutBackendServices.DiscardFlip(requestId);
            return;
        }

        Interlocked.Increment(ref _flipsPublished);
        Interlocked.Increment(ref FlipsPublished);
        ReportFlipSurface(in displayBuffer, in request, imageIdentifier, image, texture);
        _flipTarget.PublishGuestFrame(
            texture,
            Math.Max(displayBuffer.Width, 1),
            Math.Max(displayBuffer.Height, 1),
            requestId,
            serial,
            VideoOutBackendServices.IsHdrPixelFormat(displayBuffer.PixelFormat));
        VideoOutBackendServices.TraceGpuFlip(
            _memory, handle, index, flipMode, flipArg, displayBuffer.Address, requestId);
        // The shared phase profile measures the whole draw translation path, which v2
        // runs on this thread exactly as the Vulkan render loop does on its own. A
        // flip is this host's frame boundary, so SHARPEMU_PROFILE_RENDER=1 reports the
        // same breakdown here without any Metal-specific instrumentation.
        SharpEmu.Libs.VideoOut.RenderPhaseProfile.RecordFrame();
        SharpEmu.Libs.VideoOut.PresentTimeLog.Record();
    }

    // What the flip actually resolved to. A title whose frames come out black has either
    // drawn nothing or had its flip resolve to a different cached image from the one the
    // render passes wrote, and the flip counters alone cannot tell those apart: this says
    // which image, which view, and whether the GPU ever wrote it.
    private void ReportFlipSurface(
        in VideoOutExports.DisplayBufferInfo displayBuffer,
        in ImageRequest request,
        SharpEmu.Libs.Gpu.Buffers.ResourceSlotIdentifier identifier,
        MetalCachedImage image,
        nint texture)
    {
        MetalV2Diagnostics.InfoOnce(
            "flip-surface",
            $"The first published flip resolved to image={identifier} texture=0x{texture:X} " +
            $"address=0x{displayBuffer.Address:X16} {displayBuffer.Width}x{displayBuffer.Height} " +
            $"guestFormat={displayBuffer.PixelFormat} tiling={displayBuffer.TilingMode} " +
            $"hdr={VideoOutBackendServices.IsHdrPixelFormat(displayBuffer.PixelFormat)} " +
            $"description={image.Description.PixelFormat} tile={image.Description.TileMode} " +
            $"extent={image.Description.Extent.Width}x{image.Description.Extent.Height} " +
            $"view={request.View.Format} gpuWritten={image.IsGpuModified} renderTarget={image.Uses.RenderTarget}.");
        if (_dumpThisFlip) { DumpTexture(texture, "flip"); }
    }

    // The presenter samples the flip texture and writes the result into a plain
    // BGRA8 drawable, so an sRGB surface has to be read through a linear view:
    // sampling it as sRGB would convert to linear and show the frame far too dark.
    private nint PresentationViewOf(
        ResourceSlotIdentifier imageIdentifier, MetalCachedImage image, in ImageRequest request)
    {
        var view = request.View;
        var linear = MetalV2Formats.LinearCounterpart(view.Format);
        if (linear != view.Format)
        {
            view = view with { Format = linear };
        }

        view = view with { Mapping = PresentSwizzle(request.Description.Bgra16, image.Description.Bgra16) };
        return _imageCache.AcquirePresentationView(imageIdentifier, in view);
    }

    // A Bgra16 image has its guest bytes swapped on upload and download, so its texture
    // holds logical R,G,B,A. A render target in the same B-first order is not swapped: its
    // texture keeps memory order, channel 0 being blue. The flip's request resolves to
    // either one (HasSameBacking ignores Bgra16), and the presenter must sample logical
    // R,G,B,A from both, so a request and an image that disagree on the order read through
    // a B,G,R,A view. The exchange is its own inverse, so the same view serves either
    // direction of the disagreement.
    internal static GpuComponentMapping PresentSwizzle(bool requestBgra16, bool imageBgra16) =>
        requestBgra16 != imageBgra16
            ? new GpuComponentMapping(
                GpuComponentSwizzle.B, GpuComponentSwizzle.G, GpuComponentSwizzle.R, GpuComponentSwizzle.A)
            : default;

    public void DrawIndexed(ulong submitId, in DrawIndexedArguments arguments)
    {
        using var entryScope = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(
            SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.CommandDrawEntry);
        _translation!.DrawIndexed(submitId, in arguments);
    }

    public void DrawAuto(ulong submitId, in DrawAutoArguments arguments)
    {
        using var entryScope = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(
            SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.CommandDrawEntry);
        _translation!.DrawAuto(submitId, in arguments);
    }

    public void DispatchDirect(
        ulong submitId,
        uint groupsX,
        uint groupsY,
        uint groupsZ,
        uint dispatchInitiator,
        ulong indirectArgumentsAddress = 0)
    {
        DrainDraws();
        _translation!.Dispatch(submitId, groupsX, groupsY, groupsZ, dispatchInitiator, indirectArgumentsAddress);
    }

    // The dispatches after one that waits on its pipeline compile beside it (GpuCommandInterpreter.PrefetchPipelinesAhead).
    public bool PrefetchesPipelines => SupportsPipelineSuspend;

    public void PrefetchDispatchPipeline(
        SharpEmu.Libs.Gpu.GpuCommands.Registers.RegisterBanks banks,
        uint groupsX,
        uint groupsY,
        uint groupsZ,
        uint dispatchInitiator)
    {
        _prefetchingPipeline = true;
        var abandoned = Interlocked.Read(ref _speculativeWaitsAbandoned);
        try
        {
            _translation?.PrefetchDispatchPipeline(banks, groupsX, groupsY, groupsZ, dispatchInitiator);
        }
        catch (SharpEmu.Libs.Gpu.Scheduling.GpuFatal.AbandonedWorkException) when (
            Interlocked.Read(ref _speculativeWaitsAbandoned) != abandoned)
        {
            // This dispatch reads bytes the GPU is still writing (AbandonSpeculativeWait): its
            // pipeline is left to the dispatch itself, and the look-ahead goes on to the next one.
        }
        finally
        {
            _prefetchingPipeline = false;
        }
    }

    public bool ResolvesIndirectDispatchOnGpu => true;

    public bool ResolvesIndirectDrawOnGpu => true;

    public void OnQueueReset(int queueId)
    {
        // The translation state belongs to the interpreting thread.
        DrainDraws();
        _translation!.QueueReset(queueId);
    }

    public void ReleaseShaderWrites()
    {
        // Ordered behind the draws before it, and nothing the stream waits for: queued, not waited on.
        if (PostToWorker(ReleaseShaderWrites))
        {
            return;
        }

        if (TryDeferBehindDraws(_releaseOrderedShaderWrites ??= () => _translation?.ReleaseShaderWrites()))
        {
            return;
        }

        DrainDraws();
        _translation?.ReleaseShaderWrites();
    }

    private Action? _releaseOrderedShaderWrites;

    /// <summary>
    /// Runs interpreter slices until nothing is runnable or the budget ends; blocked
    /// heads retry on the shared interval, exactly as the Vulkan worker does.
    /// </summary>
    public void RunCommandStreamSlices(int maxSlices)
    {
        if (UsesCommandThread)
        {
            RunCommandThreadBatch();
            return;
        }

        var stream = CommandStream;
        // A title stuck before its first flip never reaches the frame boundary below, and
        // that is the title whose breakdown is wanted: the report is driven from here too.
        SharpEmu.Libs.VideoOut.RenderPhaseProfile.RecordProgress();
        for (var slice = 0; slice < maxSlices; slice++)
        {
            // A stopping session gets its teardown now, not after the rest of the budget:
            // while a title loads, one slice can spend seconds building pipelines, and
            // the worker's drain has to fit in the host's bounded shutdown window.
            if (HostSessionControl.IsShutdownRequested)
            {
                break;
            }

            // The shared profile only charges its detail phases inside an enclosing
            // scope, so that cache work on other threads stays out of the render
            // thread's counters. This is the v2 worker's equivalent of the Vulkan
            // render loop's outer scope, and it is what makes
            // SHARPEMU_PROFILE_RENDER=1 attribute the draw path on this backend.
            using var sliceScope = SharpEmu.Libs.VideoOut.RenderPhaseProfile.Measure(
                SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.CommandStream);
            var result = stream.ProcessOne();
            // A slice that ends suspended or between submissions leaves no draw unrecorded.
            DrainDraws();
            switch (result)
            {
                case SliceResult.NoWork:
                    CommitCoalescedFlush();
                    FlushPendingLabelWrites();
                    return;
                case SliceResult.AllBlocked:
                    RetryBlockedIfDue();
                    CommitCoalescedFlush();
                    CommitRecordedWhileBlocked();
                    FlushPendingLabelWrites();
                    return;
                case SliceResult.BlockedWithoutProgress:
                    if (!stream.HasUnblockedPending)
                    {
                        RetryBlockedIfDue();
                        CommitCoalescedFlush();
                        CommitRecordedWhileBlocked();
                        FlushPendingLabelWrites();
                        return;
                    }

                    break;
                default:
                    _lastProgressTicks = System.Diagnostics.Stopwatch.GetTimestamp();
                    break;
            }
        }

        // The batch can also end with its slice budget spent, and the worker then parks
        // without coming back through the cases above. A label left in an uncommitted
        // command buffer at that point is one the GPU never writes and the guest waits
        // for: Quake's engine gave up and called abort. A deferred Stream flush is held
        // only while more work is runnable; the worker commits it before it parks.
        CommitCoalescedIfOverdue();
        FlushPendingLabelWrites();
    }

    // The presenter freed the flip slot: retry suspended heads now instead of at the
    // next blocked-retry interval. Runs on the GPU worker thread.
    internal void RetryBlockedFlips()
    {
        CommandStream.NotifyFlipSlotAvailable();
        _lastProgressTicks = System.Diagnostics.Stopwatch.GetTimestamp();
    }

    /// <summary>
    /// The GPU finished everything up to this serial, so a compare against bytes a
    /// shader, a fill or an image writeback produced may pass now. Without this a
    /// wait on GPU-written memory would only be retried by the blocked-retry timer.
    /// </summary>
    internal void NotifyGpuCompleted(ulong completedSerial)
    {
        if (DsTrace.Gpu)
        {
            DsNoteLabelsLanded(completedSerial);
        }

        if (_commandStream is { } stream)
        {
            stream.NotifyCompletedGpuTick(completedSerial);
        }
    }

    // Every head waits, and what they wait for may be work this buffer already holds: a
    // dispatch on one queue whose shader stores the flag another queue's WAIT_REG_MEM reads.
    // Only a completion re-checks such a wait, so a buffer left open here is never run and
    // the waits pass only when something else commits it (Ghost of Tsushima's title screen:
    // one frame every few seconds, every queue parked on another's flag).
    private void CommitRecordedWhileBlocked()
    {
        if (_commandBuffer != 0 && _encodersInBuffer != 0)
        {
            FlushCommandBuffer(MetalFlushCause.WaitUncommitted);
        }
    }

    private void RetryBlockedIfDue()
    {
        var now = System.Diagnostics.Stopwatch.GetTimestamp();
        if (now - _lastProgressTicks < _blockedRetryTicks)
        {
            return;
        }

        CommandStream.RetryBlocked();
        _lastProgressTicks = now;
    }

    public ulong SubmissionsStarted => _submissionsStarted;
}

// What a flip needs from the presenter: a slot check and the published frame.
internal interface IMetalFlipTarget
{
    bool HasFlipSlot();

    // The texture is presented once the submission serial completes; the presenter
    // retains it until the frame it belongs to has been shown.
    void PublishGuestFrame(nint texture, uint width, uint height, ulong requestId, ulong serial, bool isHdr);
}

/// <summary>When a clock label's value is sampled (see MetalRenderHost.ClockSampling).</summary>
internal enum EndOfPipeClockSampling
{
    /// <summary>
    /// Sampled by the completion of the work before it and published there, riding the next
    /// commit that carries that work.
    /// </summary>
    Completion,

    /// <summary>As Completion, but the clock commits the buffer it is in at once.</summary>
    CompletionStrict,

    /// <summary>Sampled as parsed and landed in order with the labels, as the GPU reaches it.</summary>
    Deferred,

    /// <summary>Sampled and written as parsed.</summary>
    Parse,
}
