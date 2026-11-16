// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Runtime.InteropServices;

namespace SharpEmu.Libs.Gpu.GpuCommands;

public sealed partial class GpuCommandInterpreter
{
    // Runs only on a fatal: the stream around the cursor, the cursor stack, and a re-read of
    // the packet the walk started from. Words that changed since that read prove a rewrite.
    private void ReportStreamContext(string reason)
    {
        _streamContextReportedPacket = _packetSerial + 1;
        var execution = _execution;
        if (execution is null || execution.IsEmpty)
        {
            Console.Error.WriteLine($"[GPU][FATAL] {reason}: no cursor is loaded (queue={QueueId}).");
            DumpPacketTrace();
            return;
        }

        var topIndex = execution.Depth - 1;
        var cursor = execution.At(topIndex);
        DumpUnknownPacket(cursor, cursor.Offset);
        DumpPacketTrace();

        var packetAddress = cursor.Address + ((ulong)cursor.Offset * sizeof(uint));
        Console.Error.WriteLine(
            $"[GPU][FATAL] {reason}: queue={QueueId} depth={execution.Depth} kind={DescribeCursorKind(execution, topIndex)} " +
            $"buffer=0x{cursor.Address:X16} dwords={cursor.DwordCount} offset=0x{cursor.Offset:X5} " +
            $"packet=0x{packetAddress:X16} ringChunkBase=0x{cursor.RingChunkBase:X16}");
        for (var index = 0; index <= topIndex; index++)
        {
            var frame = execution.At(index);
            Console.Error.WriteLine(
                $"\tcursor[{index}] kind={DescribeCursorKind(execution, index)} buffer=0x{frame.Address:X16} " +
                $"dwords={frame.DwordCount} offset=0x{frame.Offset:X5}");
        }

        if (topIndex >= _payloadScratchByDepth.Count)
        {
            return;
        }

        // The window read when the walk reached this packet is still in the depth's scratch.
        var scratch = _payloadScratchByDepth[topIndex];
        var header = scratch[0];
        var length = PacketHeader.Length(header);
        var covered = (int)Math.Min(Math.Min(length, PacketWindowDwords), cursor.Remaining);
        var reread = new uint[covered];
        if (!_host.TryReadGuest(packetAddress, MemoryMarshal.AsBytes(reread.AsSpan())))
        {
            Console.Error.WriteLine($"[GPU][FATAL] The packet cannot be re-read: address=0x{packetAddress:X16}.");
            return;
        }

        var changed = 0;
        var text = new System.Text.StringBuilder();
        for (var index = 0; index < covered; index++)
        {
            text.Append(index == 0 ? " " : ", ").Append($"{scratch[index]:X8}");
            if (scratch[index] != reread[index])
            {
                changed++;
                text.Append($"->{reread[index]:X8}");
            }
        }

        Console.Error.WriteLine(
            $"[GPU][FATAL] Packet as read: header=0x{header:X8} type={PacketHeader.PacketType(header)} " +
            $"opcode=0x{PacketHeader.Opcode(header):X2} length={length} payloadDwords={covered - 1} " +
            $"dwords={text}");
        Console.Error.WriteLine(
            changed == 0
                ? "[GPU][FATAL] Re-read matches: the words did not change under the worker."
                : $"[GPU][FATAL] The guest rewrote this packet while it was being read: {changed} of {covered} dwords changed.");
    }

    // SHARPEMU_GPU_CLOBBER_DIAG: the packet a stream-context report was printed for (serial
    // plus one, zero for none), and the register table the current packet pointed at.
    private ulong _streamContextReportedPacket;
    private ulong _clobberTablePacket;
    private ulong _clobberTableAddress;
    private uint _clobberTableCount;

    private void NoteClobberRegisterTable(ulong tableAddress, uint count)
    {
        _clobberTablePacket = _packetSerial + 1;
        _clobberTableAddress = tableAddress;
        _clobberTableCount = count;
    }

    /// <summary>
    /// SHARPEMU_GPU_CLOBBER_DIAG: what a late write into recycled command memory leaves
    /// behind, printed by the queue before a fatal ends the process. A submit-time table
    /// address that differs from the packet now, or a landed label at the packet's bytes,
    /// is the clobber; the same bad words at submit mean the guest wrote them.
    /// </summary>
    /// <summary>Where the walk stands (SHARPEMU_CP_HISTORY=reuse): the next packet of the innermost buffer, 0 when idle.</summary>
    internal ulong ReuseWalkPosition() =>
        _execution is { IsEmpty: false } execution
            ? execution.Top.Address + ((ulong)execution.Top.Offset * sizeof(uint))
            : 0;

    internal void ReportClobber(CommandSubmission submission, string queues)
    {
        var now = System.Diagnostics.Stopwatch.GetTimestamp();
        var execution = _execution;
        var packetAddress = 0UL;
        var remaining = 0u;
        if (execution is { IsEmpty: false })
        {
            ref var top = ref execution.Top;
            packetAddress = top.Address + ((ulong)top.Offset * sizeof(uint));
            remaining = top.Remaining;
        }

        Console.Error.WriteLine(
            $"[GPU][CLOBBER] packet=0x{packetAddress:X16} queue={QueueId} submission={submission.SubmissionId} " +
            $"submitId={SubmitId} enqueueAgeMs={GpuClobberDiagnostics.AgeMilliseconds(submission.EnqueuedTicks, now)} " +
            $"dcb=0x{submission.Address:X16} dcbDwords={submission.DwordCount} depth={execution?.Depth ?? 0}");
        Console.Error.WriteLine($"[GPU][CLOBBER] queues {queues}");
        if (_streamContextReportedPacket != _packetSerial + 1)
        {
            ReportStreamContext("Clobber report");
        }

        if (packetAddress == 0)
        {
            return;
        }

        var covered = (int)Math.Min(PacketWindowDwords, Math.Max(remaining, 1u));
        Span<uint> words = stackalloc uint[(int)PacketWindowDwords];
        if (_host.TryReadGuest(packetAddress, MemoryMarshal.AsBytes(words[..covered])))
        {
            var length = (int)Math.Min(Math.Max(PacketHeader.Length(words[0]), 1u), (uint)covered);
            var text = new System.Text.StringBuilder();
            for (var index = 0; index < length; index++)
            {
                text.Append(index == 0 ? "" : ",").Append($"{words[index]:X8}");
            }

            Console.Error.WriteLine(
                $"[GPU][CLOBBER] reread packet=0x{packetAddress:X16} header=0x{words[0]:X8} " +
                $"opcode=0x{PacketHeader.Opcode(words[0]):X2} length={PacketHeader.Length(words[0])} dwords={text}");
        }
        else
        {
            Console.Error.WriteLine($"[GPU][CLOBBER] reread packet=0x{packetAddress:X16} unreadable=1");
        }

        var tables = GpuClobberDiagnostics.FindSubmittedTables(packetAddress, 4);
        if (tables.Count == 0)
        {
            Console.Error.WriteLine($"[GPU][CLOBBER] submitTable packet=0x{packetAddress:X16} recorded=0");
        }

        foreach (var table in tables)
        {
            Console.Error.WriteLine(
                $"[GPU][CLOBBER] submitTable packet=0x{packetAddress:X16} table=0x{table.TableAddress:X16} " +
                $"count={table.Count} register=0x{table.Register:X4} ageMs={GpuClobberDiagnostics.AgeMilliseconds(table.Ticks, now)}");
        }

        var parseTable = _clobberTablePacket == _packetSerial + 1;
        if (parseTable)
        {
            Console.Error.WriteLine(
                $"[GPU][CLOBBER] parseTable packet=0x{packetAddress:X16} table=0x{_clobberTableAddress:X16} count={_clobberTableCount}");
        }

        const ulong Margin = 64;
        var packetBytes = (ulong)covered * sizeof(uint);
        ReportClobberHistory("packet", packetAddress, packetAddress >= Margin ? packetAddress - Margin : 0, packetBytes + 2 * Margin, now);
        if (parseTable && _clobberTableAddress != 0)
        {
            var tableBytes = Math.Min((ulong)_clobberTableCount * 2 * sizeof(uint), 4096UL);
            ReportClobberHistory("table", _clobberTableAddress,
                _clobberTableAddress >= Margin ? _clobberTableAddress - Margin : 0, tableBytes + 2 * Margin, now);
        }
    }

    private static void ReportClobberHistory(string near, ulong origin, ulong address, ulong size, long now)
    {
        const int Limit = 32;
        var entries = CommandProcessorWriteHistory.Find(address, size);
        Console.Error.WriteLine(
            $"[GPU][CLOBBER] history near={near} origin=0x{origin:X16} from=0x{address:X16} bytes={size} " +
            $"entries={entries.Count} shown={Math.Min(entries.Count, Limit)}");
        for (var index = Math.Max(0, entries.Count - Limit); index < entries.Count; index++)
        {
            var entry = entries[index];
            var delta = (long)(entry.Address - origin);
            Console.Error.WriteLine(
                $"[GPU][CLOBBER] history near={near} ageMs={GpuClobberDiagnostics.AgeMilliseconds(entry.Ticks, now)} " +
                $"queue={entry.QueueId} address=0x{entry.Address:X16} delta={(delta < 0 ? "-" : "+")}0x{Math.Abs(delta):X} " +
                $"size={entry.Size} value=0x{entry.Value:X16} kind={entry.Kind} deferred={(entry.Deferred ? 1 : 0)}");
        }
    }

    private static string DescribeCursorKind(PacketCursorStack execution, int index)
    {
        if (execution.At(index).FollowedChunkAdvance)
        {
            return "followed-ring-chunk";
        }

        return index == 0 ? "top-level-dcb" : "indirect-buffer";
    }

    private Exception FatalUnknownEvent(uint eventType, uint eventIndex)
    {
        ReportStreamContext("Unknown event");
        return _host.Fatal($"The event type is unknown: type=0x{eventType:X8} index=0x{eventIndex:X8}.");
    }
}
