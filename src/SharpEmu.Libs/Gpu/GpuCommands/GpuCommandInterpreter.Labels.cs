// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.GpuCommands.Packets;
using SharpEmu.Libs.Gpu.Scheduling;

namespace SharpEmu.Libs.Gpu.GpuCommands;

public sealed partial class GpuCommandInterpreter
{
    private const uint GcrGl2MetadataInvalidate = 1u << 1;
    private const uint GcrGl0VectorInvalidate = 1u << 2;
    private const uint GcrGl1Invalidate = 1u << 3;
    private const uint GcrGl2Unshared = 1u << 4;
    private const uint GcrGl2Invalidate = 1u << 8;
    private const uint GcrGl2Writeback = 1u << 9;
    private const uint GcrOrder012 = 1u << 10;
    private const uint GcrOrder210 = 2u << 10;
    private const uint GcrKnownMask = GcrGl2MetadataInvalidate | GcrGl0VectorInvalidate | GcrGl1Invalidate |
        GcrGl2Unshared | GcrGl2Invalidate | GcrGl2Writeback | GcrOrder012 | GcrOrder210;

    private static int _unknownGcrWarnings;

    private bool WaitForPublication()
    {
        if (!_host.HasPendingPublications(QueueId)) return false;
        Suspend();
        return true;
    }

    // What an end-of-pipe or release write stores, decoded from the packet's data select.
    internal enum EndOfPipeData : byte
    {
        None,
        Value32,
        Value64,
        Clock,
        Gds,
        AtomicAdd32,
    }

    // The VGT event types that complete as a timestamp event (end of pipe or end of shader).
    private static bool IsTimestampEvent(uint eventType) =>
        eventType is 0x04 or 0x14 or 0x28 or 0x29 or 0x2B or 0x2D or 0x2F or 0x30;

    // EVENT_WRITE_EOP cache action bits 12-17: TCL1_VOL, TC_VOL, (reserved), TC_WB, TCL1, TC.
    private const uint EndOfPipeCacheActionKnownMask = 0x3Bu;

    private static int _unknownCacheActionWarnings;

    private static void WarnUnknownEndOfPipeCacheAction(uint cacheAction)
    {
        var unknown = cacheAction & ~EndOfPipeCacheActionKnownMask;
        if (unknown != 0 && Interlocked.Increment(ref _unknownCacheActionWarnings) <= 16)
        {
            Console.Error.WriteLine($"[LOADER][WARN] command_stream.eop_unknown_cache_action bits=0x{unknown:X2}");
        }
    }

    // EVENT_WRITE_EOP and EVENT_WRITE_EOS. Each field is decoded on its own: the event type
    // says the write is a timestamp event, the data select what is written, the interrupt
    // select whether the host is told, and the cache action only whether it is a write-back.
    // The cache policy is an L2 residency hint and the event index follows the event type;
    // neither changes what lands.
    internal void WriteEndOfPipe(
        bool is64Bit,
        uint cachePolicy,
        uint eventWriteDestination,
        uint eopEventType,
        uint cacheAction,
        uint eventIndex,
        uint eventWriteSource,
        ulong destination,
        ulong value,
        uint interruptSelector,
        uint interruptContextId)
    {
        _ = cachePolicy;
        if (eventWriteDestination > 1)
        {
            throw _host.Fatal($"The end-of-pipe write destination is not supported: destination={eventWriteDestination} address=0x{destination:X16}.");
        }

        if (!IsTimestampEvent(eopEventType))
        {
            throw _host.Fatal(
                $"The end-of-pipe event type is unknown: source={eventWriteSource} type=0x{eopEventType:X2} action=0x{cacheAction:X2} " +
                $"index={eventIndex} bits={(is64Bit ? 64 : 32)} interrupt={interruptSelector} destination=0x{destination:X16}.");
        }

        // EOP: 0 discard, 1 low 32 bits, 2 64 bits, 3 GPU clock, 4 system clock, 5 GDS.
        // EOS: 1 GDS, 2 32 bits (0 is the append count, which nothing here keeps).
        var data = is64Bit
            ? eventWriteSource switch
            {
                0 => EndOfPipeData.None,
                1 => EndOfPipeData.Value32,
                2 => EndOfPipeData.Value64,
                3 or 4 => EndOfPipeData.Clock,
                5 => EndOfPipeData.Gds,
                _ => throw _host.Fatal($"The end-of-pipe data selection is not supported: selection={eventWriteSource} address=0x{destination:X16}."),
            }
            : eventWriteSource switch
            {
                1 => EndOfPipeData.Gds,
                2 => EndOfPipeData.Value32,
                _ => throw _host.Fatal($"The end-of-shader data selection is not supported: selection={eventWriteSource} address=0x{destination:X16}."),
            };

        WarnUnknownEndOfPipeCacheAction(cacheAction);
        CompleteEndOfPipe(data, writeBack: cacheAction != 0, destination, value, interruptSelector, interruptContextId);
    }

    // Records one end-of-pipe completion: the data it stores and the interrupt it raises.
    // Interrupt select 0 is none, and 3 sends the data with a write confirm, which every
    // label here already has (it lands in order behind the work): neither raises one.
    // 1 on the graphics queue is the interrupt alone; 2, and 4 (with its context id),
    // interrupt once the data is written.
    private void CompleteEndOfPipe(EndOfPipeData data, bool writeBack, ulong destination, ulong value, uint interruptSelector, uint interruptContextId)
    {
        var withInterrupt = false;
        switch (interruptSelector)
        {
            case 0:
            case 3:
                break;
            case 1:
                if (!IsComputeQueue)
                {
                    QueueInterrupt(interruptContextId);
                    return;
                }

                withInterrupt = true;
                break;
            case 2:
            case 4:
                withInterrupt = true;
                break;
            default:
                throw _host.Fatal($"The end-of-pipe interrupt selector is unknown: selector={interruptSelector} destination=0x{destination:X16}.");
        }

        if (data != EndOfPipeData.None && destination == 0)
        {
            throw _host.Fatal($"The end-of-pipe destination is zero: data={data} interrupt={interruptSelector}.");
        }

        var eventId = InterruptEventId;
        switch (data)
        {
            case EndOfPipeData.None:
                if (withInterrupt)
                {
                    QueueInterrupt(interruptContextId);
                }

                return;
            case EndOfPipeData.Value32:
                Write32(destination, (uint)value, writeBack, withInterrupt, eventId, interruptContextId);
                return;
            case EndOfPipeData.Value64:
                Write64(destination, value, writeBack, withInterrupt, eventId, interruptContextId);
                return;
            case EndOfPipeData.Clock:
            {
                var clock = GpuReferenceClock.Read();
                var kind = withInterrupt
                    ? (writeBack ? EndOfPipeWriteKind.InterruptWriteBack64 : EndOfPipeWriteKind.Interrupt64)
                    : (writeBack ? EndOfPipeWriteKind.ClockWriteBack : EndOfPipeWriteKind.ClockWrite);
                PublishLabel(new EndOfPipeWrite(kind, SubmitId, destination, clock, eventId, interruptContextId, ReferenceClock: true), true);
                return;
            }
            case EndOfPipeData.Gds:
                WriteGdsWords(destination, value, withInterrupt, interruptContextId);
                return;
            case EndOfPipeData.AtomicAdd32:
                AtomicAddAtEndOfPipe(destination, (uint)value, withInterrupt, eventId, interruptContextId);
                return;
        }
    }

    private void WriteEndOfPipeDword(ulong destination, uint value, EndOfPipeWriteKind kind)
    {
        if (_host.DefersEndOfPipeLabel(kind))
        {
            if (CommandProcessorWriteHistory.Enabled)
            {
                CommandProcessorWriteHistory.Record(QueueId, destination, sizeof(uint), value, kind.ToString(), deferred: true);
            }

            PendingLabelRecorded?.Invoke(destination, sizeof(uint), value);
            _lastWriteLength = 0;
        }
        else WriteDword(destination, value);
    }

    private void WriteEndOfPipeQword(ulong destination, ulong value, EndOfPipeWriteKind kind)
    {
        if (_host.DefersEndOfPipeLabel(kind))
        {
            if (CommandProcessorWriteHistory.Enabled)
            {
                CommandProcessorWriteHistory.Record(QueueId, destination, sizeof(ulong), value, kind.ToString(), deferred: true);
            }

            PendingLabelRecorded?.Invoke(destination, sizeof(ulong), value);
            _lastWriteLength = 0;
        }
        else WriteQword(destination, value);
    }

    private void Write32(ulong destination, uint value, bool withWriteBack, bool withInterrupt, int eventId, uint contextId)
    {
        var kind = withInterrupt
            ? (withWriteBack ? EndOfPipeWriteKind.InterruptWriteBack32 : EndOfPipeWriteKind.Interrupt32)
            : (withWriteBack ? EndOfPipeWriteKind.WriteBack32 : EndOfPipeWriteKind.Write32);
        PublishLabel(new EndOfPipeWrite(kind, SubmitId, destination, value, eventId, contextId), false);
    }

    private void Write64(ulong destination, ulong value, bool withWriteBack, bool withInterrupt, int eventId, uint contextId)
    {
        var kind = withInterrupt
            ? (withWriteBack ? EndOfPipeWriteKind.InterruptWriteBack64 : EndOfPipeWriteKind.Interrupt64)
            : (withWriteBack ? EndOfPipeWriteKind.WriteBack64 : EndOfPipeWriteKind.Write64);
        PublishLabel(new EndOfPipeWrite(kind, SubmitId, destination, value, eventId, contextId), true);
    }

    // A host that publishes labels after its work (TryDeferEndOfPipe) owns the write; a host
    // that defers one kind itself (DefersEndOfPipeLabel) records the pending value; any other
    // host takes the write now.
    private void PublishLabel(in EndOfPipeWrite write, bool is64Bit)
    {
        var completion = write with { WaitForGpu = CompletionLabelsRequireGpu };
        if (_host.TryDeferEndOfPipe(QueueId, in completion))
        {
            _lastWriteLength = 0;
            return;
        }

        if (completion.WaitForGpu) _host.SynchronizeGpu();
        if (is64Bit) WriteEndOfPipeQword(write.Destination, write.Value, write.Kind);
        else WriteEndOfPipeDword(write.Destination, (uint)write.Value, write.Kind);
        RecordHostEndOfPipe(in write);
    }

    // GDS words land after the work before them, which on the hardware is what makes them
    // final. A host that copies them out of its GDS in stream order takes the write as it
    // is; any other host reads the GDS once the GPU has finished.
    private void WriteGdsWords(ulong destination, ulong value, bool withInterrupt, uint contextId)
    {
        if (WaitForPublication()) return;
        var wordOffset = (uint)(value & 0xFFFFu);
        var wordCount = (uint)(value >> 16);
        var write = new EndOfPipeWrite(EndOfPipeWriteKind.GdsWrite32, SubmitId, destination, GdsWordOffset: wordOffset, GdsWordCount: wordCount);
        if (_host.WritesGdsAtEndOfPipe)
        {
            RecordHostEndOfPipe(in write);
            _lastWriteLength = 0;
            if (withInterrupt)
            {
                QueueInterrupt(contextId);
            }

            return;
        }

        _host.SynchronizeGpu();
        var words = new uint[wordCount];
        _host.ReadGds(words, wordOffset, wordCount);
        WriteBytes(destination, System.Runtime.InteropServices.MemoryMarshal.AsBytes<uint>(words));
        RecordHostEndOfPipe(in write);
        if (withInterrupt)
        {
            _host.TriggerInterrupt(InterruptEventId, contextId);
        }
    }

    // RELEASE_MEM data select 7 adds the low 32 bits of the data to the dword at the
    // destination once the work before it has finished. A host that defers labels does the
    // add from the completion of that work; an immediate host has already finished it.
    private void AtomicAddAtEndOfPipe(ulong destination, uint addend, bool withInterrupt, int eventId, uint contextId)
    {
        if ((destination & 0x3u) != 0)
        {
            throw _host.Fatal($"The end-of-pipe atomic add is not dword aligned: address=0x{destination:X16}.");
        }

        var kind = EndOfPipeWriteKind.AtomicAdd32;
        if (!_host.DefersEndOfPipeLabel(kind))
        {
            NotePacketBlockWrite();
            _host.ExecuteCommandMemoryOperation(CommandMemoryOperation.Atomic(Packets.AtomicMemoryPacket.Add32(destination, addend)));
            GuestMemoryWritten?.Invoke(destination, sizeof(uint));
        }

        _lastWriteLength = 0;
        RecordHostEndOfPipe(new EndOfPipeWrite(kind, SubmitId, destination, addend, eventId, contextId));
        if (withInterrupt)
        {
            QueueInterrupt(contextId);
        }
    }

    // Release-memory completions stay in the recording command buffer instead of
    // submitting it at every packet: immediate hosts already wrote their labels,
    // deferred hosts own the payload+acknowledgement, and RunSlice submits when
    // the slice ends or blocks, so every deferred interrupt still reaches the GPU.
    // Submitting per packet produced ~430 tiny submissions per Astro Bot frame.
    // SHARPEMU_DEFER_RELEASE_FLUSH=0 restores the per-packet submit.
    private static readonly bool DeferReleaseFlush = !string.Equals(
        Environment.GetEnvironmentVariable("SHARPEMU_DEFER_RELEASE_FLUSH"), "0", StringComparison.Ordinal);

    private void FlushReleaseMemory()
    {
        if (!DeferReleaseFlush)
        {
            _host.Flush();
        }
    }

    internal void QueueInterrupt(uint interruptContextId)
    {
        var write = new EndOfPipeWrite(EndOfPipeWriteKind.InterruptOnly, SubmitId, EventId: InterruptEventId, ContextId: interruptContextId);
        if (!_host.TryDeferEndOfPipe(QueueId, in write)) RecordHostEndOfPipe(in write);
    }

    private static bool ReleaseMemGcrNeedsBarrier(uint eopEventType, uint gcrControl) =>
        eopEventType != 0x28 ||
        (gcrControl & (GcrGl2MetadataInvalidate | GcrGl0VectorInvalidate | GcrGl1Invalidate | GcrGl2Invalidate | GcrGl2Writeback)) != 0;

    private static void WarnUnknownReleaseMemGcr(uint gcrControl)
    {
        var unknown = gcrControl & ~GcrKnownMask;
        if (unknown != 0 && Interlocked.Increment(ref _unknownGcrWarnings) <= 16)
        {
            Console.Error.WriteLine($"[LOADER][WARN] command_stream.release_mem_unknown_gcr bits=0x{unknown:X4}");
        }
    }

    // Native and wrapped packets use the same release-memory operation. The fields are
    // decoded independently: the event type and GCR bits decide the barrier, the data
    // select what is written, and the interrupt select what the host is told.
    internal void ReleaseMemory(
        uint cachePolicy,
        uint eopEventType,
        uint eventIndex,
        uint gcrControl,
        uint releaseDestination,
        uint dataSelection,
        uint interruptSelector,
        ulong destination,
        ulong value,
        uint interruptContextId)
    {
        _ = cachePolicy;
        _ = eventIndex;
        if (releaseDestination > 1)
        {
            throw _host.Fatal($"The release-memory destination is not supported: destination={releaseDestination} address=0x{destination:X16}.");
        }

        // 0 none, 1 low 32 bits, 2 64 bits, 3 GPU clock, 4 system clock (both the reference
        // clock here), 5 GDS words, 7 atomic add of the low 32 bits.
        var data = dataSelection switch
        {
            0 => EndOfPipeData.None,
            1 => EndOfPipeData.Value32,
            2 => EndOfPipeData.Value64,
            3 or 4 => EndOfPipeData.Clock,
            5 => EndOfPipeData.Gds,
            7 => EndOfPipeData.AtomicAdd32,
            _ => throw _host.Fatal($"The release-memory data selection is not supported: selection={dataSelection} address=0x{destination:X16}."),
        };

        WarnUnknownReleaseMemGcr(gcrControl);

        // Selectors 5 and 6 raise the context-id interrupt only when the label at the
        // destination is at or below the data (a 32- or 64-bit compare), and write nothing.
        if (interruptSelector is 5 or 6)
        {
            var conditionReadable = destination != 0;
            var conditionValue = 0UL;
            if (conditionReadable)
            {
                conditionValue = interruptSelector == 5 ? ReadDword(destination) : ReadQword(destination);
            }

            var decision = InterruptDecision.Evaluate(interruptSelector, IsComputeQueue, dataSelection, conditionReadable, conditionValue, value);
            data = EndOfPipeData.None;
            interruptSelector = decision.RaisesInterrupt ? 4u : 0u;
        }

        if (data == EndOfPipeData.None || (releaseDestination == 0 && destination == 0))
        {
            if (eopEventType != 0x28 || gcrControl != 0)
            {
                Diagnostics.DispatchRunTally.NoteSync();
                _host.EmitGlobalBarrier();
                _host.ReleaseShaderWrites();
            }

            TriggerReleaseInterrupt(interruptSelector, interruptContextId);
            return;
        }

        if (ReleaseMemGcrNeedsBarrier(eopEventType, gcrControl))
        {
            Diagnostics.DispatchRunTally.NoteSync();
            _host.EmitGlobalBarrier();
            _host.ReleaseShaderWrites();
        }

        // A GL2 write-back makes the label a write-back write, as the EOP cache action does.
        var writeBack = (gcrControl & GcrGl2Writeback) != 0 && data != EndOfPipeData.Gds;
        CompleteEndOfPipe(data, writeBack, destination, value, interruptSelector, interruptContextId);
        if (data == EndOfPipeData.Value32 || (data == EndOfPipeData.Gds && interruptSelector == 1))
        {
            FlushReleaseMemory();
        }
    }

    private void TriggerReleaseInterrupt(uint interruptSelector, uint interruptContextId)
    {
        switch (interruptSelector)
        {
            case 0:
            case 3:
                return;
            case 1:
            case 2:
            case 4:
                QueueInterrupt(interruptContextId);
                FlushReleaseMemory();
                return;
            default:
                throw _host.Fatal($"The release-memory interrupt selector is unknown: selector={interruptSelector}.");
        }
    }

    // The events that wait for pixel-shader stores to land: PS_PARTIAL_FLUSH waits for the
    // pixel stage to go idle and CACHE_FLUSH_AND_INV_EVENT flushes every cache after it. The
    // compute and vertex partial flushes and the CB/DB-only flushes do not order shader stores.
    private const uint PsPartialFlushEvent = 0x10;
    private const uint CacheFlushAndInvalidateEvent = 0x16;

    internal void RaiseEvent(uint eventType, uint eventIndex, ulong eventAddress)
    {
        var validCacheEventIndex = eventIndex is 0 or 7;
        if (eventType == 0x39)
        {
            // A query's begin dump writes the first qword of a 16-byte depth-block pair and its
            // end dump the second: open from the one to the other. A pair seen out of step
            // only keeps draws running that could have been left out.
            _occlusionQueryOpen = (eventAddress & 0x8) == 0;
        }

        switch (eventType)
        {
            case 0x07:
            case 0x0F:
            case 0x10:
                Diagnostics.DispatchRunTally.NoteSync();
                _host.EmitGlobalBarrier();
                if (eventType == PsPartialFlushEvent)
                {
                    _host.ReleaseShaderWrites();
                }

                break;
            case 0x16:
            case 0x31:
            case 0x2A:
            case 0x2C:
            case 0x2E:
                if (!validCacheEventIndex)
                {
                    throw FatalUnknownEvent(eventType, eventIndex);
                }

                Diagnostics.DispatchRunTally.NoteSync();
                _host.EmitGlobalBarrier();
                if (eventType == CacheFlushAndInvalidateEvent)
                {
                    _host.ReleaseShaderWrites();
                }

                break;
            case 0x0D:
            case 0x0E:
            case 0x12:
            case 0x17:
            case 0x18:
            case 0x19:
            case 0x1A:
            case 0x1B:
                break;
            case 0x38 or 0x3A when _host.PublishesSyntheticOcclusionDumps:
                break;
            case 0x39 when _host.PublishesSyntheticOcclusionDumps:
                WriteSyntheticOcclusionDump(eventIndex, eventAddress);
                break;
            case 0x38:
            {
                var counterId = (eventAddress >> 3) & 0x3Fu;
                if (eventIndex != 1 || counterId >= (ulong)_syntheticOcclusionCounters.Length)
                {
                    throw _host.Fatal($"The occlusion-counter selection is not supported: index=0x{eventIndex:X8} counter={counterId}.");
                }

                // The address field holds packed control data, not a guest pointer.
                _selectedOcclusionCounter = (int)counterId;
                _occlusionDumpStrideBytes = 4u << (int)((eventAddress >> 9) & 0x3u);
                _occlusionDumpInstanceMask = (uint)((eventAddress >> 11) & 0xFF_FFFFu);
                Interlocked.Increment(ref _pixelPipeStatControls);
                break;
            }
            case 0x3A:
                if (eventIndex != 0)
                {
                    throw _host.Fatal($"The occlusion-counter reset index is invalid: index=0x{eventIndex:X8}.");
                }

                _syntheticOcclusionCounters[_selectedOcclusionCounter] = 0;
                if (_selectedOcclusionCounter == 0 && !_host.ResetOcclusionCounter(QueueId) &&
                    _host.TryReadOcclusionCounter(QueueId, out var resetCount))
                {
                    _occlusionCounterStart = resetCount;
                }
                Interlocked.Increment(ref _pixelPipeStatResets);
                break;
            case 0x39:
            {
                if (eventIndex != 1 || eventAddress == 0 || (eventAddress & 0x7u) != 0)
                {
                    throw _host.Fatal($"The occlusion-counter dump is invalid: index=0x{eventIndex:X8} address=0x{eventAddress:X16}.");
                }

                NotePacketBlockWrite();
                if (_selectedOcclusionCounter == 0 &&
                    _host.DumpOcclusionCounter(QueueId, eventAddress, _occlusionDumpInstanceMask, _occlusionDumpStrideBytes))
                {
                    if (VisibilityResultTrace.Enabled)
                        VisibilityResultTrace.Occlusion($"dump queue={QueueId} submit={SubmitId} address=0x{eventAddress:X16} counter=0 deferred=True mask=0x{_occlusionDumpInstanceMask:X6} stride={_occlusionDumpStrideBytes}");
                    break;
                }

                const ulong readyBit = 1UL << 63;
                var hostCount = 0UL;
                var aggregate = _selectedOcclusionCounter == 0 && _host.TryReadOcclusionCounter(QueueId, out hostCount);
                var count = aggregate ? (hostCount - _occlusionCounterStart) & (readyBit - 1) : SyntheticOcclusionCounter;
                var result = readyBit | count;
                if (VisibilityResultTrace.Enabled)
                    VisibilityResultTrace.Occlusion($"dump queue={QueueId} submit={SubmitId} address=0x{eventAddress:X16} counter={_selectedOcclusionCounter} aggregate={aggregate} host={hostCount} resetBase={_occlusionCounterStart} value={count} mask=0x{_occlusionDumpInstanceMask:X6} stride={_occlusionDumpStrideBytes}");
                for (var depthBlock = 0u; depthBlock < 24u; depthBlock++)
                {
                    if ((_occlusionDumpInstanceMask & (1u << (int)depthBlock)) != 0)
                    {
                        var destination = eventAddress + ((ulong)depthBlock * _occlusionDumpStrideBytes);
                        // One virtual depth block holds the host sum. The other blocks are empty.
                        // The DB writes the counters when the pipeline reaches the event, so they land in
                        // order with the command-processor stores recorded before it: UE4 zeroes each
                        // query block with a WRITE_DATA right before its begin dump, and dumps written as
                        // parsed were wiped when that deferred clear landed, so every query read occluded.
                        WriteCommandProcessorQword(destination, aggregate && depthBlock != 0 ? readyBit : result, "ZPASS_DUMP");
                    }
                }

                if (_occlusionDumpInstanceMask != 0 && VisibilityResultTrace.Enabled)
                    VisibilityResultTrace.Published(QueueId, SubmitId, eventAddress, result,
                        _occlusionDumpInstanceMask, _occlusionDumpStrideBytes, _host.Memory as SharpEmu.HLE.IGuestBackedSpace, aggregate);
                if (!aggregate)
                    _syntheticOcclusionCounters[_selectedOcclusionCounter] = (SyntheticOcclusionCounter + 1) & (readyBit - 1);
                Interlocked.Increment(ref _occlusionWrites);
                break;
            }

            default:
                throw FatalUnknownEvent(eventType, eventIndex);
        }
    }

    // See ICommandStreamHost.PublishesSyntheticOcclusionDumps. The per-queue counter bank,
    // the select's block mask and stride, and the reset model the hardware counters a host
    // reads. With no such counter behind them, Demon's Souls on Metal drew hard-edged black
    // areas in the Outpost Passage, which the synthetic pairs below never did.
    private void WriteSyntheticOcclusionDump(uint eventIndex, ulong eventAddress)
    {
        using var phaseScope = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.OcclusionDump);
        if (eventIndex != 1 || eventAddress == 0 || (eventAddress & 0x7u) != 0)
        {
            throw _host.Fatal($"The occlusion-counter dump is invalid: index=0x{eventIndex:X8} address=0x{eventAddress:X16}.");
        }

        const ulong readyBit = 1UL << 63;
        var result = readyBit | ((ulong)Interlocked.Increment(ref _hostlessOcclusionCounter) & (readyBit - 1));
        // The DB writes the counters when the pipeline reaches the event, so they land in
        // order with the command-processor stores recorded before it: UE4 zeroes each
        // query block with a WRITE_DATA right before its begin dump, and dumps written as
        // parsed were wiped when that deferred clear landed, so every query read occluded.
        WriteCommandProcessorQwords(eventAddress, 16, 2 * sizeof(ulong), result, "ZPASS_DUMP");
    }

    // Flip packets first need a queue slot; without one the submission retries later.
    private bool TryReserveFlipSlot()
    {
        if (_host.HasFlipSlot())
        {
            return true;
        }

        SuspendOn(new CommandWaitCondition(CommandWaitKind.FlipSlot));
        return false;
    }

    internal void SubmitFlip()
    {
        if (!TryReserveFlipSlot())
        {
            return;
        }

        var flip = PendingFlip;
        var requestId = _host.PrepareFlip(QueueId, flip.Handle, flip.Index, flip.FlipMode, flip.FlipArgument);
        var write = new EndOfPipeWrite(
            EndOfPipeWriteKind.Flip, SubmitId, FlipHandle: flip.Handle, FlipIndex: flip.Index, FlipMode: flip.FlipMode, FlipArgument: flip.FlipArgument, FlipRequestId: requestId);
        if (!_host.TryDeferEndOfPipe(QueueId, in write)) RecordHostEndOfPipe(in write);
        _host.Flush();
    }

    internal void SubmitFlipWithLabel(ulong destination, uint value)
    {
        if (!TryReserveFlipSlot())
        {
            return;
        }

        var deferred = _host.HasPendingPublications(QueueId);
        if (!deferred) WriteEndOfPipeDword(destination, value, EndOfPipeWriteKind.FlipWithWrite32);
        var flip = PendingFlip;
        var requestId = _host.PrepareFlip(QueueId, flip.Handle, flip.Index, flip.FlipMode, flip.FlipArgument);
        var write = new EndOfPipeWrite(
            EndOfPipeWriteKind.FlipWithWrite32, SubmitId, destination, value,
            FlipHandle: flip.Handle, FlipIndex: flip.Index, FlipMode: flip.FlipMode, FlipArgument: flip.FlipArgument, FlipRequestId: requestId);
        if (deferred) PublishLabel(in write, false);
        else RecordHostEndOfPipe(in write);
        _host.Flush();
    }

    internal void SubmitFlipWithInterrupt(uint eopEventType, uint cacheAction, ulong destination, uint value)
    {
        if (eopEventType != 0x04 || cacheAction != 0x38)
        {
            throw _host.Fatal($"The flip event type is unknown: type=0x{eopEventType:X8} action=0x{cacheAction:X8} destination=0x{destination:X16}.");
        }

        if (!TryReserveFlipSlot())
        {
            return;
        }

        var deferred = _host.HasPendingPublications(QueueId);
        if (!deferred) WriteEndOfPipeDword(destination, value, EndOfPipeWriteKind.FlipWithInterruptWriteBack32);
        var flip = PendingFlip;
        var requestId = _host.PrepareFlip(QueueId, flip.Handle, flip.Index, flip.FlipMode, flip.FlipArgument);
        var write = new EndOfPipeWrite(
            EndOfPipeWriteKind.FlipWithInterruptWriteBack32, SubmitId, destination, value, InterruptEventId,
            FlipHandle: flip.Handle, FlipIndex: flip.Index, FlipMode: flip.FlipMode, FlipArgument: flip.FlipArgument, FlipRequestId: requestId);
        if (deferred) PublishLabel(in write, false);
        else RecordHostEndOfPipe(in write);
        _host.Flush();
    }

    internal void WaitForFlip(uint videoOutHandle, uint displayBufferIndex)
    {
        _host.Flush();
        if (!_host.IsFlipDone((int)videoOutHandle, (int)displayBufferIndex))
        {
            SuspendOn(CommandWaitCondition.Flip((int)videoOutHandle, (int)displayBufferIndex));
        }
    }
}
