// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Runtime.InteropServices;
using SharpEmu.Libs.Gpu.GpuCommands.Packets;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.Libs.VideoOut;

namespace SharpEmu.Libs.Gpu.GpuCommands;

public sealed partial class GpuCommandInterpreter
{
    private const uint DmaSkippedDestination = 0x3022C;

    // Destination selectors 1, 2, 4 and 5 are memory; registers and GDS are not supported.
    internal void WriteData(ulong destination, ReadOnlySpan<uint> source, uint writeControl)
    {
        var control = WriteDataControl.DecodeNative(writeControl);
        if (control.Destination is not (1 or 2 or 4 or 5))
        {
            throw _host.Fatal($"The write-data destination selector is not supported: selector=0x{control.Destination:X2} address=0x{destination:X16}.");
        }

        WriteDataValues(destination, source, !control.IncrementAddress);
    }

    private void WriteDataValues(ulong destination, ReadOnlySpan<uint> source, bool writeOneAddress)
    {
        if (source.IsEmpty)
        {
            return;
        }

        if (writeOneAddress)
        {
            WriteCommandProcessorBytes(destination, MemoryMarshal.AsBytes(source[^1..]), "WRITE_DATA");
            return;
        }

        WriteCommandProcessorBytes(destination, MemoryMarshal.AsBytes(source), "WRITE_DATA");
    }

    /// <summary>
    /// Whether this queue's command-processor writes land in record order with its deferred
    /// labels and the work recorded before them, rather than as parsed. Every queue: a compute
    /// queue's write made at parse time reached guest memory while dispatches recorded ahead
    /// of it were still running, and GTA V's culling queue rewrites a descriptor table with
    /// WRITE_DATA in memory an earlier dispatch of the same queue stores vertices into. The
    /// stores then landed over the table, and the next dispatch read floats as its V#.
    /// </summary>
    private bool DefersCommandProcessorWrites => _host.DefersCommandProcessorWrites;

    // Pending values noted for waits are capped: a wait polls a fence, not a table.
    private const int MaxNotedPendingBytes = 64;

    /// <summary>
    /// A command-processor store. On a host that defers labels the queue hands it to the host
    /// to land in record order: written as parsed, it reached the guest ahead of
    /// labels recorded before it, and Demon's Souls recycled command memory on such a write
    /// while a label into that memory was still owed (a WAIT_REG_MEM between them is met at
    /// parse by the pending label, so the parser runs on past it). This queue's own waits see
    /// the value through the pending lookup and the own-write check; every other read of the
    /// bytes is a read of a recorded GPU write, which the host waits for.
    /// SHARPEMU_METAL_V2_CP_WRITES=parse writes them as parsed again.
    /// </summary>
    internal void WriteCommandProcessorBytes(ulong destination, ReadOnlySpan<byte> source, string kind)
    {
        NotePacketBlockWrite();
        if (!DefersCommandProcessorWrites)
        {
            WriteBytes(destination, source, kind);
            return;
        }

        var first = source.Length >= sizeof(ulong)
            ? BinaryPrimitives.ReadUInt64LittleEndian(source)
            : source.Length >= sizeof(uint) ? BinaryPrimitives.ReadUInt32LittleEndian(source) : 0UL;
        using (RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.CommandCpBookkeeping))
        {
            if (CommandProcessorWriteHistory.Enabled)
            {
                CommandProcessorWriteHistory.Record(QueueId, destination, source.Length,
                    source.Length == sizeof(uint) ? (uint)first : first, kind, deferred: true);
            }

            NotePendingBytes(destination, source);
        }

        _host.RecordCommandProcessorWrite(destination, source, kind);

        using var trailingScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.CommandCpBookkeeping);
        _lastWriteLength = 0;
        if (source.Length is sizeof(uint) or sizeof(ulong))
        {
            _lastWriteAddress = destination;
            _lastWriteValue = source.Length == sizeof(uint) ? (uint)first : first;
            _lastWriteLength = source.Length;
            _lastWritePacket = _packetSerial;
        }
    }

    private void WriteCommandProcessorDword(ulong destination, uint value, string kind)
    {
        Span<byte> bytes = stackalloc byte[sizeof(uint)];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes, value);
        WriteCommandProcessorBytes(destination, bytes, kind);
    }

    private void WriteCommandProcessorQword(ulong destination, ulong value, string kind)
    {
        Span<byte> bytes = stackalloc byte[sizeof(ulong)];
        BinaryPrimitives.WriteUInt64LittleEndian(bytes, value);
        WriteCommandProcessorBytes(destination, bytes, kind);
    }

    /// <summary>
    /// Whether runs of identical qword stores (the synthetic occlusion dump) are handed to the host
    /// as one call. SHARPEMU_GPU_CP_WRITE_BATCH=0 records each store on its own again.
    /// </summary>
    internal bool BatchCommandProcessorWrites { get; set; } = CommandProcessorWriteBatching.Enabled;

    /// <summary>
    /// <paramref name="count"/> qword stores of <paramref name="value"/>, <paramref name="strideBytes"/>
    /// apart. They land exactly as <paramref name="count"/> calls to
    /// <see cref="WriteCommandProcessorQword"/> would: in order, with nothing recorded between them.
    /// </summary>
    private void WriteCommandProcessorQwords(ulong destination, int count, ulong strideBytes, ulong value, string kind)
    {
        if (count <= 0)
        {
            return;
        }

        if (!BatchCommandProcessorWrites || !DefersCommandProcessorWrites || strideBytes < sizeof(ulong) ||
            count == 1 || PendingLabelRecorded is null)
        {
            for (var index = 0; index < count; index++)
            {
                WriteCommandProcessorQword(destination + (ulong)index * strideBytes, value, kind);
            }

            return;
        }

        using (RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.CommandCpBookkeeping))
        {
            if (CommandProcessorWriteHistory.Enabled)
            {
                for (var index = 0; index < count; index++)
                {
                    CommandProcessorWriteHistory.Record(QueueId, destination + (ulong)index * strideBytes,
                        sizeof(ulong), value, kind, deferred: true);
                }
            }

            if (PendingQwordsRecorded is { } noteRun)
            {
                noteRun(destination, count, strideBytes, value);
            }
            else
            {
                for (var index = 0; index < count; index++)
                {
                    PendingLabelRecorded(destination + (ulong)index * strideBytes, sizeof(ulong), value);
                }
            }
        }

        NotePacketBlockWrite();
        _host.RecordCommandProcessorQwords(destination, count, strideBytes, value, kind);

        _lastWriteAddress = destination + (ulong)(count - 1) * strideBytes;
        _lastWriteValue = value;
        _lastWriteLength = sizeof(ulong);
        _lastWritePacket = _packetSerial;
    }

    // Tells the queue what the bytes will hold once they land, so a wait on them - this
    // queue's or another's - reads them as written, as it does a deferred label.
    private void NotePendingBytes(ulong destination, ReadOnlySpan<byte> bytes)
    {
        if (PendingLabelRecorded is not { } note)
        {
            return;
        }

        var length = Math.Min(bytes.Length, MaxNotedPendingBytes);
        var unit = bytes.Length % sizeof(ulong) == 0 ? sizeof(ulong) : sizeof(uint);
        for (var offset = 0; offset + unit <= length; offset += unit)
        {
            var value = unit == sizeof(ulong)
                ? BinaryPrimitives.ReadUInt64LittleEndian(bytes[offset..])
                : BinaryPrimitives.ReadUInt32LittleEndian(bytes[offset..]);
            note(destination + (ulong)offset, unit, value);
        }
    }

    // SHARPEMU_TRACE_GPU_CLOCKS=<n>: the first n clock values the stream samples, with the gap since the last one, so the
    // durations a title measures from its timestamps can be read off.
    private static readonly int TracedClockLimit =
        int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_TRACE_GPU_CLOCKS"), out var tracedClocks) ? tracedClocks : 0;
    private static int _tracedClocks;
    private static ulong _lastTracedClock;

    private void TraceClock(string source, ulong destination, ulong clock)
    {
        if (TracedClockLimit == 0 || Interlocked.Increment(ref _tracedClocks) > TracedClockLimit)
        {
            return;
        }

        var last = Interlocked.Exchange(ref _lastTracedClock, clock);
        Console.Error.WriteLine(
            $"[GPU][CLOCK] {source} submit={SubmitId} dst=0x{destination:X} value={clock} since_last_us={(last == 0 ? 0 : (long)(clock - last) / 100.0):F1}");
    }

    internal void WriteReferenceClock(ulong destination, uint byteCount)
    {
        if (destination == 0 || byteCount is not (sizeof(uint) or sizeof(ulong)) || (destination & (byteCount - 1u)) != 0)
        {
            throw _host.Fatal($"The reference-clock copy is invalid: destination=0x{destination:X16} size={byteCount}.");
        }

        var clock = GpuReferenceClock.Read();
        TraceClock("copy", destination, clock);
        if (_host.DefersEndOfPipeLabel(EndOfPipeWriteKind.ClockWrite) &&
            (byteCount == sizeof(ulong) || _host.DefersEndOfPipeLabel(EndOfPipeWriteKind.Write32)))
        {
            RecordReferenceClock(destination, byteCount, clock);
            return;
        }

        if (byteCount == sizeof(uint))
        {
            WriteDword(destination, (uint)clock);
        }
        else
        {
            WriteQword(destination, clock);
        }
    }

    /// <summary>
    /// Hands a reference-clock copy to the host like an end-of-pipe clock: the value is the
    /// clock as the packet was parsed, and it lands after the labels recorded before it.
    /// </summary>
    /// <remarks>
    /// Demon's Souls recycles command memory once a GPU timestamp it reads back passes the
    /// one it recorded. Written at parse time, the timestamp overtook end-of-pipe labels the
    /// GPU still owed, and a late 64-bit label then landed in a recycled command buffer.
    /// Same gate as end-of-pipe clocks: SHARPEMU_METAL_V2_EOP_CLOCKS=parse writes it at parse.
    /// A four-byte copy rides a 32-bit label so the host writes exactly the bytes asked for.
    /// </remarks>
    private void RecordReferenceClock(ulong destination, uint byteCount, ulong clock)
    {
        var is64Bit = byteCount == sizeof(ulong);
        var value = is64Bit ? clock : (uint)clock;
        if (CommandProcessorWriteHistory.Enabled)
        {
            CommandProcessorWriteHistory.Record(QueueId, destination, (int)byteCount, value, "COPY_DATA_CLOCK", deferred: true);
        }

        PendingLabelRecorded?.Invoke(destination, (int)byteCount, value);
        _lastWriteLength = 0;
        RecordHostEndOfPipe(new EndOfPipeWrite(
            is64Bit ? EndOfPipeWriteKind.ClockWrite : EndOfPipeWriteKind.Write32, SubmitId, destination, value,
            ReferenceClock: true));
    }

    // Selectors 0 and 3 are memory, 1 is GDS, 2 is immediate data (source) or nowhere (destination).
    internal void TransferData(
        uint engine,
        uint destinationSelect,
        uint destinationCachePolicy,
        ulong destination,
        uint sourceSelect,
        uint sourceCachePolicy,
        ulong sourceOrImmediate,
        uint byteCount,
        uint waitForPrevious,
        uint writeConfirm,
        uint blockEngine)
    {
        if (engine > 1)
        {
            throw _host.Fatal($"The DMA engine is not supported: engine={engine} destination=0x{destination:X16}.");
        }

        if (byteCount == 0)
        {
            return;
        }

        if (destinationCachePolicy > 3 || sourceCachePolicy > 3 || waitForPrevious > 1 || writeConfirm > 1 || blockEngine > 1)
        {
            throw _host.Fatal(
                $"The DMA control fields are out of range: dstCache={destinationCachePolicy} srcCache={sourceCachePolicy} " +
                $"wait={waitForPrevious} confirm={writeConfirm} block={blockEngine} destination=0x{destination:X16}.");
        }

        if ((uint)destination == DmaSkippedDestination)
        {
            return;
        }

        if (destinationSelect == 2)
        {
            if (sourceSelect != 3)
            {
                throw _host.Fatal($"The DMA nowhere source selector is not supported: selector=0x{sourceSelect:X2}.");
            }

            return;
        }

        if (!TryDecodeGds(destinationSelect, out var destinationIsGds))
        {
            throw _host.Fatal($"The DMA destination selector is not supported: selector=0x{destinationSelect:X2} destination=0x{destination:X16}.");
        }

        if (sourceSelect == 2)
        {
            if (!destinationIsGds && DefersCommandProcessorWrites)
            {
                var fill = (uint)sourceOrImmediate;
                if (CommandProcessorWriteHistory.Enabled)
                {
                    CommandProcessorWriteHistory.Record(QueueId, destination, (int)Math.Min(byteCount, int.MaxValue),
                        fill, "DMA_FILL", deferred: true);
                }

                if (byteCount <= MaxNotedPendingBytes && byteCount % sizeof(uint) == 0)
                {
                    Span<uint> pattern = stackalloc uint[MaxNotedPendingBytes / sizeof(uint)];
                    pattern.Fill(fill);
                    NotePendingBytes(destination, MemoryMarshal.AsBytes(pattern)[..(int)byteCount]);
                }

                NotePacketBlockWrite();
                _host.FillBufferInOrder(destination, byteCount, fill);
                return;
            }

            if (!destinationIsGds && CommandProcessorWriteHistory.Enabled)
            {
                CommandProcessorWriteHistory.Record(QueueId, destination, (int)Math.Min(byteCount, int.MaxValue),
                    (uint)sourceOrImmediate, "DMA_FILL", deferred: false);
            }

            NotePacketBlockWrite();
            _host.FillBuffer(destination, byteCount, (uint)sourceOrImmediate, destinationIsGds);
            if (!destinationIsGds)
            {
                GuestMemoryWritten?.Invoke(destination, byteCount);
            }
            return;
        }

        if (!TryDecodeGds(sourceSelect, out var sourceIsGds))
        {
            throw _host.Fatal($"The DMA source selector is not supported: selector=0x{sourceSelect:X2} source=0x{sourceOrImmediate:X16}.");
        }

        if (sourceIsGds && destinationIsGds)
        {
            throw _host.Fatal($"A GDS-to-GDS DMA copy is not supported: destination=0x{destination:X16} source=0x{sourceOrImmediate:X16} size={byteCount}.");
        }

        var deferredCopy = !destinationIsGds && !sourceIsGds && DefersCommandProcessorWrites;
        if (!destinationIsGds && CommandProcessorWriteHistory.Enabled)
        {
            // The value is the source's, read now for the record only: a copy may land later.
            ulong copied = 0;
            if (!sourceIsGds && byteCount is 4 or 8)
            {
                Span<byte> bytes = stackalloc byte[8];
                if (_host.Memory.TryRead(sourceOrImmediate, bytes[..(int)byteCount]))
                {
                    copied = byteCount == 4 ? BinaryPrimitives.ReadUInt32LittleEndian(bytes) : BinaryPrimitives.ReadUInt64LittleEndian(bytes);
                }
            }

            CommandProcessorWriteHistory.Record(QueueId, destination, (int)Math.Min(byteCount, int.MaxValue),
                copied, $"DMA_COPY from=0x{sourceOrImmediate:X}", deferred: deferredCopy);
        }

        if (deferredCopy)
        {
            // A copy of a value a wait could also read as pending - a label or another
            // deferred store - is pending at its destination too. Anything else is read by
            // the GPU when it gets there, and a wait on it waits for that.
            if (byteCount is sizeof(uint) or sizeof(ulong) && PendingLabelLookup is { } lookup &&
                lookup(sourceOrImmediate, byteCount == sizeof(ulong), out var pendingValue))
            {
                PendingLabelRecorded?.Invoke(destination, (int)byteCount, pendingValue);
            }

            NotePacketBlockWrite();
            _host.CopyBufferInOrder(destination, sourceOrImmediate, byteCount);
            return;
        }

        NotePacketBlockWrite();
        _host.CopyBuffer(destination, sourceOrImmediate, byteCount, destinationIsGds, sourceIsGds);
        if (!destinationIsGds)
        {
            GuestMemoryWritten?.Invoke(destination, byteCount);
        }
    }

    private static bool TryDecodeGds(uint selector, out bool isGds)
    {
        switch (selector)
        {
            case 0:
            case 3:
                isGds = false;
                return true;
            case 1:
                isGds = true;
                return true;
            default:
                isGds = false;
                return false;
        }
    }

    internal void WriteConstantRam(uint byteOffset, ReadOnlySpan<uint> source) =>
        source.CopyTo(_constantRam.AsSpan((int)(byteOffset / 4)));

    internal void DumpConstantRam(ulong destination, uint byteOffset, uint dwordCount) =>
        WriteBytes(destination, MemoryMarshal.AsBytes(_constantRam.AsSpan((int)(byteOffset / 4), (int)dwordCount)));

    internal uint WriteDataPacket(in PacketContext packet, ReadOnlySpan<uint> payload)
    {
        var dwordCount = (packet.Header >> 16) & 0x3FFFu;
        var destination = Address(payload[1], payload[2]);
        WriteData(destination, payload.Slice(3, (int)dwordCount - 2), payload[0]);
        return 1 + dwordCount;
    }

    internal uint DmaDataPacket(in PacketContext packet, ReadOnlySpan<uint> payload)
    {
        if (packet.Header != PacketHeader.Make(7, PacketOpcode.DmaData))
        {
            throw _host.Fatal($"The DMA packet header is not supported: header=0x{packet.Header:X8} address=0x{packet.PacketAddress:X16}.");
        }

        var control = payload[0];
        var control2 = payload[5];
        var source = Address(payload[1], payload[2]);
        var destination = Address(payload[3], payload[4]);
        if (control == 0x6000_0000u && destination == DmaSkippedDestination && (control2 >> 21) == 0x141u)
        {
            return 6;
        }

        TransferData(
            engine: control & 0x1u,
            destinationSelect: ((control >> 20) & 0x3u) | ((control2 >> 25) & 0x4u) | ((control2 >> 26) & 0x8u),
            destinationCachePolicy: (control >> 25) & 0x3u,
            destination,
            sourceSelect: ((control >> 29) & 0x3u) | ((control2 >> 24) & 0x4u) | ((control2 >> 25) & 0x8u),
            sourceCachePolicy: (control >> 13) & 0x3u,
            source,
            byteCount: control2 & 0x03FF_FFFFu,
            waitForPrevious: (control2 >> 30) & 0x1u,
            writeConfirm: (control2 >> 31) & 0x1u,
            blockEngine: (control >> 31) & 0x1u);
        return 6;
    }

    private uint CopyDataDestinationToDma(uint destinationSelect) => destinationSelect switch
    {
        2 or 4 or 5 => 3,
        3 or 6 or 7 => 1,
        _ => throw _host.Fatal($"The copy-data destination selector is not supported: selector=0x{destinationSelect:X2}."),
    };

    private uint CopyDataSourceToDma(uint sourceSelect) => sourceSelect switch
    {
        2 or 4 or 5 => 3,
        3 or 6 or 7 => 1,
        10 or 11 => 2,
        _ => throw _host.Fatal($"The copy-data source selector is not supported: selector=0x{sourceSelect:X2}."),
    };

    internal uint CopyDataPacketHandler(in PacketContext packet, ReadOnlySpan<uint> payload)
    {
        if (packet.Header != PacketHeader.Make(6, PacketOpcode.CopyData))
        {
            throw _host.Fatal($"The copy-data packet header is not supported: header=0x{packet.Header:X8} address=0x{packet.PacketAddress:X16}.");
        }

        var control = payload[0];
        var byteCount = ((control >> 16) & 0x1u) != 0 ? 8u : 4u;
        var source = Address(payload[1], payload[2]);
        var destination = Address(payload[3], payload[4]);
        var sourceCachePolicy = (control >> 13) & 0x3u;
        var destinationCachePolicy = (control >> 25) & 0x3u;
        var writeConfirm = (control >> 20) & 0x1u;

        // Compute queues carry the raw selectors; the graphics queue folds the engine bit in.
        if (IsComputeQueue)
        {
            CopyDataOnComputeQueue(control, source, destination, byteCount, sourceCachePolicy, destinationCachePolicy, writeConfirm);
            return 5;
        }

        var sourceSelect = ((control & 0xFu) << 1) | ((control >> 30) & 0x1u);
        var destinationSelect = ((control >> 8) & 0xFu) << 1;
        var referenceClockDestination = sourceSelect switch
        {
            9u => 2u,
            18u => 4u,
            _ => 0u,
        };
        if (referenceClockDestination != 0)
        {
            if (destinationSelect != referenceClockDestination || destination == 0 || (destination & (byteCount - 1u)) != 0)
            {
                throw _host.Fatal($"The reference-clock copy-data is not supported: srcSel=0x{sourceSelect:X2} dstSel=0x{destinationSelect:X2} destination=0x{destination:X16} size={byteCount}.");
            }

            WriteReferenceClock(destination, byteCount);
            return 5;
        }

        if (sourceSelect == 12)
        {
            CopyAtomicReturn(destinationSelect, destination, byteCount, usePfp: false);
            return 5;
        }

        var dmaSource = CopyDataSourceToDma(sourceSelect);
        if (dmaSource == 2 && byteCount == 8)
        {
            // An 8-byte immediate is a plain label store, not a 32-bit fill.
            WriteCommandProcessorQword(destination, source, "COPY_DATA_IMM");
            return 5;
        }

        TransferData(0, CopyDataDestinationToDma(destinationSelect), destinationCachePolicy, destination, dmaSource, sourceCachePolicy, source, byteCount, 1, writeConfirm, 1);
        return 5;
    }

    private void CopyDataOnComputeQueue(uint control, ulong source, ulong destination, uint byteCount, uint sourceCachePolicy, uint destinationCachePolicy, uint writeConfirm)
    {
        var sourceSelect = control & 0xFu;
        var destinationSelect = (control >> 8) & 0xFu;
        var dmaDestination = destinationSelect switch
        {
            1 or 2 => 3u,
            3 => 1u,
            _ => throw _host.Fatal($"The copy-data destination selector is not supported: selector=0x{destinationSelect:X2} queue={QueueId}."),
        };
        switch (sourceSelect)
        {
            case 1:
            case 2:
                TransferData(0, dmaDestination, destinationCachePolicy, destination, 3, sourceCachePolicy, source, byteCount, 1, writeConfirm, 1);
                return;
            case 3:
                TransferData(0, dmaDestination, destinationCachePolicy, destination, 1, sourceCachePolicy, source, byteCount, 1, writeConfirm, 1);
                return;
            case 5:
                if (byteCount == 8)
                {
                    WriteQword(destination, source);
                    return;
                }

                TransferData(0, dmaDestination, destinationCachePolicy, destination, 2, sourceCachePolicy, source, byteCount, 1, writeConfirm, 1);
                return;
            case 6:
                CopyAtomicReturn(destinationSelect, destination, byteCount, usePfp: false);
                return;
            default:
                throw _host.Fatal($"The copy-data source selector is not supported: selector=0x{sourceSelect:X2} queue={QueueId}.");
        }
    }

    // The atomic return latch is the value the last returning atomic read before it wrote.
    private void CopyAtomicReturn(uint destinationSelect, ulong destination, uint byteCount, bool usePfp)
    {
        var valid = usePfp ? AtomicReturnPfpValid : AtomicReturnMeValid;
        if (!valid)
        {
            throw _host.Fatal($"The atomic return value is not available: destination=0x{destination:X16} queue={QueueId}.");
        }

        if (destination == 0 || (destination & (byteCount - 1u)) != 0)
        {
            throw _host.Fatal($"The copy-data destination is invalid: dstSel=0x{destinationSelect:X2} destination=0x{destination:X16} size={byteCount}.");
        }

        var data = usePfp ? AtomicReturnPfpData : AtomicReturnMeData;
        if (byteCount == 8)
        {
            WriteCommandProcessorQword(destination, data, "COPY_DATA_ATOMIC");
        }
        else
        {
            WriteCommandProcessorDword(destination, (uint)data, "COPY_DATA_ATOMIC");
        }
    }

    internal uint WriteConstantRamPacket(in PacketContext packet, ReadOnlySpan<uint> payload)
    {
        var dwordCount = (packet.Header >> 16) & 0x3FFFu;
        var offset = payload[0];
        if (dwordCount >= ConstantRamDwords || offset > 0xBFFC || (offset & 0x3u) != 0)
        {
            throw _host.Fatal($"The constant RAM write is out of range: offset=0x{offset:X} dwords={dwordCount}.");
        }

        WriteConstantRam(offset, payload.Slice(1, (int)dwordCount));
        return 1 + dwordCount;
    }

    internal uint DumpConstantRamPacket(in PacketContext packet, ReadOnlySpan<uint> payload)
    {
        if (packet.Header != PacketHeader.Make(5, PacketOpcode.DumpConstantRam))
        {
            throw _host.Fatal($"The constant RAM dump header is not supported: header=0x{packet.Header:X8}.");
        }

        var offset = payload[0];
        var dwordCount = payload[1];
        var destination = Address(payload[2], payload[3]);
        if (dwordCount >= ConstantRamDwords || offset > 0xBFFC || (offset & 0x3u) != 0)
        {
            throw _host.Fatal($"The constant RAM dump is out of range: offset=0x{offset:X} dwords={dwordCount} destination=0x{destination:X16}.");
        }

        DumpConstantRam(destination, offset, dwordCount);
        return 4;
    }

    internal uint GetLodStatsPacket(in PacketContext packet, ReadOnlySpan<uint> payload)
    {
        if (packet.Header != PacketHeader.Make(5, PacketOpcode.GetLodStats))
        {
            throw _host.Fatal($"The LOD statistics header is not supported: header=0x{packet.Header:X8}.");
        }

        // The title reads LOD statistics: a host that left the counters out of its pipelines
        // until now compiles them in from the next pipeline on.
        MipStatistics.NoteTitleUsesCounters();
        var bufferSize = payload[0];
        var destination = (payload[1] & 0xFFFF_FFC0u) | ((ulong)payload[2] << 32);
        var slotsBound = SharpEmu.Libs.Gpu.Images.LodStatsTelemetry.Enabled
            ? MipStatistics.Shared.BoundSlotCount
            : 0;
        // Control bit 18 asks for a reset and bit 19 enables the counters again
        // (sceAgcDcbGetLodStats). Either one starts a new measurement window: a title that
        // hands its counter ids to other textures every report (Demon's Souls passes
        // enable without reset) reads the min of unrelated textures otherwise.
        var resetCounters = (payload[3] & ((1u << 18) | (1u << 19))) != 0;
        NotePacketBlockWrite();
        if (destination != 0 && bufferSize != 0 && _host.TryRecordLodStatistics(destination, bufferSize, resetCounters))
        {
            SharpEmu.Libs.Gpu.Images.LodStatsTelemetry.NotePacket(bufferSize, destination, hostRecorded: true, slotsBound);
            return 4;
        }

        SharpEmu.Libs.Gpu.Images.LodStatsTelemetry.NotePacket(bufferSize, destination, hostRecorded: false, slotsBound);

        if (destination != 0 && bufferSize != 0)
        {
            var report = new byte[bufferSize];
            MipStatistics.Shared.WriteReport(report);
            SharpEmu.Libs.Gpu.Images.LodStatsTelemetry.NoteReport(report);
            WriteBytes(destination, report);
        }

        return 4;
    }
}
