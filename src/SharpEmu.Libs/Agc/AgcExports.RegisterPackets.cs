// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.Libs.Diagnostics;
using SharpEmu.Libs.Kernel;

namespace SharpEmu.Libs.Agc;

public static partial class AgcExports
{
    // This partial constructs AGC register-programming command packets.

    private const uint CbSetShRegisterRangeMarker = 0x6875000D;

    // SHARPEMU_AGC_COALESCE=0 emits CbSetShRegistersDirect's runs one allocation and write at a
    // time again. Not readonly so the differential test can compare both paths in one process.
    internal static bool CoalesceShRegisterRuns =
        Environment.GetEnvironmentVariable("SHARPEMU_AGC_COALESCE") != "0";

    [SysAbiExport(
        Nid = "UZbQjYAwwXM",
        ExportName = "sceAgcCbSetShRegistersDirect",
        Target = Generation.Gen5,
        LibraryName = "libSceAgc")]
    public static int CbSetShRegistersDirect(CpuContext ctx)
    {
        var commandBufferAddress = ctx[CpuRegister.Rdi];
        var registersAddress = ctx[CpuRegister.Rsi];
        var registerCount = (uint)ctx[CpuRegister.Rdx];
        if (registerCount == 0)
        {
            if (Diagnostics.HleCallShapeProfile.Enabled) Diagnostics.HleCallShapeProfile.RecordShRegisters(0, false, false, 0, false, Diagnostics.HleCallShapeProfile.ShRegistersOutcome.Invalid);
            return ReturnPointer(ctx, 0);
        }

        if (commandBufferAddress == 0 || registersAddress == 0 || registerCount > 4096)
        {
            if (Diagnostics.HleCallShapeProfile.Enabled) Diagnostics.HleCallShapeProfile.RecordShRegisters(registerCount, false, false, 0, false, Diagnostics.HleCallShapeProfile.ShRegistersOutcome.Invalid);
            return ReturnPointer(ctx, 0);
        }

        // One bulk read of the (offset, value) list and one write per contiguous run, instead of
        // two reads per register and a write per dword: each guest access pays the memory
        // admission machinery. The register array is rented, not allocated, because this runs
        // every frame.
        var registerPool = System.Buffers.ArrayPool<RegisterDefaultValue>.Shared;
        var registers = registerPool.Rent((int)registerCount);
        uint[]? packet = null;
        // SHARPEMU_PERF_HLE only: the call's shape, recorded once on the way out.
        var profile = Diagnostics.HleCallShapeProfile.Enabled;
        var unsorted = false;
        var duplicates = false;
        var sourceOverlap = false;
        var runs = 0;
        var outcome = Diagnostics.HleCallShapeProfile.ShRegistersOutcome.ReadFailed;
        try
        {
            if (!ctx.Memory.TryRead(
                    registersAddress,
                    System.Runtime.InteropServices.MemoryMarshal.AsBytes(registers.AsSpan(0, (int)registerCount))))
            {
                return ReturnPointer(ctx, 0);
            }

            if (profile)
            {
                for (var index = 1; index < (int)registerCount; index++)
                {
                    unsorted |= registers[index].Offset < registers[index - 1].Offset;
                }
            }

            registers.AsSpan(0, (int)registerCount).Sort(static (left, right) => left.Offset.CompareTo(right.Offset));
            if (profile)
            {
                for (var index = 1; index < (int)registerCount; index++)
                {
                    duplicates |= registers[index].Offset == registers[index - 1].Offset;
                }
            }

            var count = (int)registerCount;
            // Every run's packet lands at the next cursor position, so when the buffer already has
            // room for all of them they are one reservation, one state read and one write (cursor
            // and packets together). Without that room the runs go one at a time below, each
            // reaching the buffer's full callback with its own size, exactly as before.
            if (CoalesceShRegisterRuns)
            {
                uint totalDwords = 0;
                var runCount = 0;
                for (var start = 0; start < count; runCount++)
                {
                    var end = start + 1;
                    while (end < count && registers[end].Offset == registers[end - 1].Offset + 1) end++;
                    totalDwords += (uint)(end - start) + 2; // at most 3 x 4096 dwords
                    start = end;
                }

                using var appendScope = new CommandBufferAppendScope(ctx, commandBufferAddress);
                if (TryReserveCommandDwordsInPlace(ctx, commandBufferAddress, totalDwords, out var batchAddress, out var nextCursor))
                {
                    runs = runCount;
                    if (profile)
                    {
                        sourceOverlap = batchAddress < registersAddress + registerCount * 8UL &&
                            registersAddress < batchAddress + totalDwords * 4UL;
                    }

                    packet = System.Buffers.ArrayPool<uint>.Shared.Rent((int)totalDwords);
                    var cursor = 0;
                    for (var start = 0; start < count;)
                    {
                        var end = start + 1;
                        while (end < count && registers[end].Offset == registers[end - 1].Offset + 1) end++;
                        packet[cursor++] = Pm4((uint)(end - start) + 2, ItSetShReg, 0);
                        packet[cursor++] = registers[start].Offset & 0xFFFFu;
                        for (var index = start; index < end; index++) packet[cursor++] = registers[index].Value;
                        start = end;
                    }

                    if (!TryCommitCommandDwords(ctx, commandBufferAddress, nextCursor, batchAddress, packet.AsSpan(0, cursor)))
                    {
                        outcome = Diagnostics.HleCallShapeProfile.ShRegistersOutcome.WriteFailed;
                        return ReturnPointer(ctx, 0);
                    }

                    outcome = Diagnostics.HleCallShapeProfile.ShRegistersOutcome.Written;
                    return ReturnPointer(ctx, batchAddress);
                }
            }

            ulong firstCommandAddress = 0;
            var startIndex = 0;
            while (startIndex < count)
            {
                var endIndex = startIndex + 1;
                while (endIndex < count &&
                       registers[endIndex].Offset == registers[endIndex - 1].Offset + 1)
                {
                    endIndex++;
                }

                var valueCount = (uint)(endIndex - startIndex);
                var packetDwords = valueCount + 2;
                runs++;
                if (!TryAllocateCommandDwords(ctx, commandBufferAddress, packetDwords, out var commandAddress))
                {
                    outcome = Diagnostics.HleCallShapeProfile.ShRegistersOutcome.NoRoom;
                    return ReturnPointer(ctx, 0);
                }

                if (profile)
                {
                    sourceOverlap |= commandAddress < registersAddress + registerCount * 8UL &&
                        registersAddress < commandAddress + packetDwords * 4UL;
                }

                packet ??= System.Buffers.ArrayPool<uint>.Shared.Rent(count + 2);
                packet[0] = Pm4(packetDwords, ItSetShReg, 0);
                packet[1] = registers[startIndex].Offset & 0xFFFFu;
                for (var index = startIndex; index < endIndex; index++)
                {
                    packet[2 + index - startIndex] = registers[index].Value;
                }

                if (!TryWriteCommandDwords(ctx, commandAddress, packet.AsSpan(0, (int)packetDwords)))
                {
                    outcome = Diagnostics.HleCallShapeProfile.ShRegistersOutcome.WriteFailed;
                    return ReturnPointer(ctx, 0);
                }

                firstCommandAddress = firstCommandAddress == 0 ? commandAddress : firstCommandAddress;
                startIndex = endIndex;
            }

            outcome = Diagnostics.HleCallShapeProfile.ShRegistersOutcome.Written;
            return ReturnPointer(ctx, firstCommandAddress);
        }
        finally
        {
            if (profile)
            {
                Diagnostics.HleCallShapeProfile.RecordShRegisters(registerCount, unsorted, duplicates, runs, sourceOverlap, outcome);
            }

            registerPool.Return(registers);
            if (packet is not null)
            {
                System.Buffers.ArrayPool<uint>.Shared.Return(packet);
            }
        }
    }

    // Payloads up to this many dwords are read and written with the packet's head in one write.
    private const uint InlinePayloadDwords = 64;

    [SysAbiExport(
        Nid = "n2fD4A+pb+g",
        ExportName = "sceAgcCbSetShRegisterRangeDirect",
        Target = Generation.Gen5,
        LibraryName = "libSceAgc")]
    public static int CbSetShRegisterRangeDirect(CpuContext ctx)
    {
        using var emitterMeasurement = AgcRegisterPacketProfile.Measure(AgcRegisterPacketProfile.Phase.Emitter);
        var commandBufferAddress = ctx[CpuRegister.Rdi];
        var offset = (uint)ctx[CpuRegister.Rsi];
        var valuesAddress = ctx[CpuRegister.Rdx];
        var valueCount = (uint)ctx[CpuRegister.Rcx];
        if (commandBufferAddress == 0 ||
            offset == 0 ||
            offset > 0x3FF ||
            !TryGetCbSetShRegisterRangeDirectLayout(valueCount, out var packetDwords, out _))
        {
            AgcRegisterPacketProfile.RecordResult(false);
            return ReturnPointer(ctx, 0);
        }

        ulong commandAddress = 0;
        var payloadWritten = false;
        using (AgcRegisterPacketProfile.Measure(AgcRegisterPacketProfile.Phase.PacketSetup))
        {
            // With the room for both, the marker NOP and the packet are one allocation and their four
            // leading dwords one write - with a small payload, read first, in it too: each guest write
            // and cursor update is a backing admission, and this runs once per draw on GTA V's render
            // threads.
            bool reserved;
            using (new CommandBufferAppendScope(ctx, commandBufferAddress))
            {
                reserved = TryReserveCommandDwordsInPlace(ctx, commandBufferAddress, 2 + packetDwords, out var markerAddress, out var nextCursor);
                if (reserved)
                {
                    commandAddress = markerAddress + 2 * sizeof(uint);
                    Span<uint> packet = stackalloc uint[4 + (int)Math.Min(valueCount, InlinePayloadDwords)];
                    packet[0] = Pm4(2, ItNop, RZero);
                    packet[1] = CbSetShRegisterRangeMarker;
                    packet[2] = Pm4(packetDwords, ItSetShReg, 0);
                    packet[3] = offset;
                    payloadWritten = valuesAddress != 0 && valueCount <= InlinePayloadDwords &&
                        (valuesAddress >= markerAddress + (2 + (ulong)packetDwords) * sizeof(uint) ||
                         valuesAddress + (ulong)valueCount * sizeof(uint) <= markerAddress) &&
                        ctx.Memory.TryRead(valuesAddress, System.Runtime.InteropServices.MemoryMarshal.AsBytes(packet[4..]));
                    if (!TryCommitCommandDwords(ctx, commandBufferAddress, nextCursor, markerAddress, payloadWritten ? packet : packet[..4]))
                    {
                        AgcRegisterPacketProfile.RecordResult(false);
                        return ReturnPointer(ctx, 0);
                    }
                }
            }

            // Without the room the fallback runs outside the append scope: the full callback must run without the lock.
            if (!reserved && TryDeclineLeaf(ctx))
            {
                // The marker and the packet would be allocated one after the other: a leaf backs out
                // before either is written.
                return ReturnPointer(ctx, 0);
            }

            if (!reserved && (!TryEmitCommandDwords(ctx, commandBufferAddress, out _, Pm4(2, ItNop, RZero), CbSetShRegisterRangeMarker) ||
                !TryAllocateCommandDwords(ctx, commandBufferAddress, packetDwords, out commandAddress) ||
                !TryWriteCommandDwords(ctx, commandAddress, Pm4(packetDwords, ItSetShReg, 0), offset)))
            {
                AgcRegisterPacketProfile.RecordResult(false);
                return ReturnPointer(ctx, 0);
            }
        }

        var payloadAddress = commandAddress + 8;
        var payloadSize = (ulong)valueCount * sizeof(uint);
        var validRanges = valuesAddress <= ulong.MaxValue - payloadSize &&
            payloadAddress <= ulong.MaxValue - payloadSize;
        var separateRanges = validRanges &&
            (valuesAddress + payloadSize <= payloadAddress || payloadAddress + payloadSize <= valuesAddress);
        var copied = payloadWritten;
        if (valuesAddress != 0 && separateRanges && !payloadWritten)
        {
            using (AgcRegisterPacketProfile.Measure(AgcRegisterPacketProfile.Phase.BulkCopy))
                copied = ctx.Memory.TryCopy(payloadAddress, valuesAddress, payloadSize);
            AgcRegisterPacketProfile.RecordPath(copied
                ? AgcRegisterPacketProfile.PayloadPath.BulkCopied
                : AgcRegisterPacketProfile.PayloadPath.BulkDeclined, valueCount);
        }
        else if (!payloadWritten)
        {
            AgcRegisterPacketProfile.RecordPath(valuesAddress == 0
                ? AgcRegisterPacketProfile.PayloadPath.NullSource
                : validRanges ? AgcRegisterPacketProfile.PayloadPath.Overlap
                : AgcRegisterPacketProfile.PayloadPath.WrappedRange, valueCount);
        }
        // A null source reserves the payload for the caller to fill later.
        // Keep forward word-copy behavior when a bulk copy is not available.
        if (valuesAddress != 0 && !copied)
        {
            using var payloadMeasurement = AgcRegisterPacketProfile.Measure(AgcRegisterPacketProfile.Phase.ScalarCopy);
            for (uint valueIndex = 0; valueIndex < valueCount; valueIndex++)
            {
                if (!TryReadUInt32(ctx, valuesAddress + (valueIndex * sizeof(uint)), out var value))
                {
                    AgcRegisterPacketProfile.RecordResult(false);
                    return ReturnPointer(ctx, 0);
                }

                if (!TryWriteUInt32(ctx, payloadAddress + (valueIndex * sizeof(uint)), value))
                {
                    AgcRegisterPacketProfile.RecordResult(false);
                    return ReturnPointer(ctx, 0);
                }
            }
        }

        TraceAgc($"agc.cb_set_sh_range buf=0x{commandBufferAddress:X16} cmd=0x{commandAddress:X16} offset=0x{offset:X8} count={valueCount}");
        AgcRegisterPacketProfile.RecordResult(true);
        return ReturnPointer(ctx, commandAddress);
    }

    [SysAbiExport(
        Nid = "bxGoVxpdSPQ",
        ExportName = "sceAgcCbSetShRegisterRangeDirectGetSize",
        Target = Generation.Gen5,
        LibraryName = "libSceAgc")]
    public static int CbSetShRegisterRangeDirectGetSize(CpuContext ctx)
    {
        var valueCount = (uint)ctx[CpuRegister.Rdi];
        if (!TryGetCbSetShRegisterRangeDirectLayout(valueCount, out _, out var sizeBytes))
        {
            ctx[CpuRegister.Rax] = 0;
            return 0;
        }

        ctx[CpuRegister.Rax] = sizeBytes;
        return (int)sizeBytes;
    }

    private static bool TryGetCbSetShRegisterRangeDirectLayout(
        uint valueCount,
        out uint packetDwords,
        out uint sizeBytes)
    {
        // The PM4 count field can encode at most 0x4001 dwords. This packet
        // uses two header dwords. The SharpEmu marker uses two more dwords.
        if (valueCount == 0 || valueCount > 0x3FFF)
        {
            packetDwords = 0;
            sizeBytes = 0;
            return false;
        }

        packetDwords = valueCount + 2;
        sizeBytes = (valueCount + 4) * sizeof(uint);
        return true;
    }

    [SysAbiExport(
        Nid = "ZvwO9euwYzc",
        ExportName = "sceAgcDcbSetCxRegistersIndirect",
        Target = Generation.Gen5,
        LibraryName = "libSceAgc")]
    public static int DcbSetCxRegistersIndirect(CpuContext ctx) =>
        DcbSetRegistersIndirect(ctx, RCxRegsIndirect, "cx");

    [SysAbiExport(
        Nid = "GBCh3zCihoU",
        ExportName = "sceAgcDcbSetCxRegistersIndirectGetSize",
        Target = Generation.Gen5,
        LibraryName = "libSceAgc")]
    public static int DcbSetCxRegistersIndirectGetSize(CpuContext ctx) =>
        DcbSetRegistersIndirectGetSize(ctx);

    [SysAbiExport(
        Nid = "-HOOCn0JY48",
        ExportName = "sceAgcDcbSetShRegistersIndirect",
        Target = Generation.Gen5,
        LibraryName = "libSceAgc")]
    public static int DcbSetShRegistersIndirect(CpuContext ctx) =>
        DcbSetRegistersIndirect(ctx, RShRegsIndirect, "sh");

    [SysAbiExport(
        Nid = "nNlUtdDDvZ0",
        ExportName = "sceAgcDcbSetShRegistersIndirectGetSize",
        Target = Generation.Gen5,
        LibraryName = "libSceAgc")]
    public static int DcbSetShRegistersIndirectGetSize(CpuContext ctx) =>
        DcbSetRegistersIndirectGetSize(ctx);

    [SysAbiExport(
        Nid = "pFLArOT53+w",
        ExportName = "sceAgcDcbSetShRegisterDirect",
        Target = Generation.Gen5,
        LibraryName = "libSceAgc")]
    public static int SetShaderRegisterDirect(CpuContext context) =>
        DcbSetRegisterDirect(context, ItSetShReg, "sh");

    [SysAbiExport(
        Nid = "QhPDD513V0w",
        ExportName = "sceAgcDcbSetShRegisterDirectGetSize",
        Target = Generation.Gen5,
        LibraryName = "libSceAgc")]
    public static int GetShaderRegisterDirectSize(CpuContext context)
    {
        context[CpuRegister.Rax] = 3u * sizeof(uint);
        return (int)context[CpuRegister.Rax];
    }

    [SysAbiExport(
        Nid = "hvUfkUIQcOE",
        ExportName = "sceAgcDcbSetUcRegistersIndirect",
        Target = Generation.Gen5,
        LibraryName = "libSceAgc")]
    public static int DcbSetUcRegistersIndirect(CpuContext ctx) =>
        DcbSetRegistersIndirect(ctx, RUcRegsIndirect, "uc");

    [SysAbiExport(
        Nid = "UQGTw4xRlcM",
        ExportName = "sceAgcDcbSetUcRegistersIndirectGetSize",
        Target = Generation.Gen5,
        LibraryName = "libSceAgc")]
    public static int DcbSetUcRegistersIndirectGetSize(CpuContext ctx) =>
        DcbSetRegistersIndirectGetSize(ctx);

    [SysAbiExport(
        Nid = "w4-d0n60hdo",
        ExportName = "sceAgcDcbSetUcRegisterDirect",
        Target = Generation.Gen5,
        LibraryName = "libSceAgc")]
    public static int DcbSetUcRegisterDirect(CpuContext ctx) =>
        DcbSetRegisterDirect(ctx, ItSetUconfigReg, "uc");

    [SysAbiExport(
        Nid = "aP1Ki9G3++4",
        ExportName = "sceAgcDcbSetUcRegisterDirectGetSize",
        Target = Generation.Gen5,
        LibraryName = "libSceAgc")]
    public static int DcbSetUcRegisterDirectGetSize(CpuContext ctx)
    {
        // SET_UCONFIG_REG header + offset + value.
        ctx[CpuRegister.Rax] = 3u * sizeof(uint);
        return (int)ctx[CpuRegister.Rax];
    }

    // DcbSetRegistersIndirect writes the same five dwords for every register space.
    private static int DcbSetRegistersIndirectGetSize(CpuContext ctx) => ctx.SetReturn(5 * sizeof(uint));

    private static int DcbSetRegistersIndirect(CpuContext ctx, uint packetRegister, string registerSpace)
    {
        var commandBufferAddress = ctx[CpuRegister.Rdi];
        var registersAddress = ctx[CpuRegister.Rsi];
        var registerCount = (uint)ctx[CpuRegister.Rdx];
        if (commandBufferAddress == 0)
        {
            return ReturnPointer(ctx, 0);
        }


        var op = packetRegister switch
        {
            RCxRegsIndirect => ItSetContextRegIndirect,
            RShRegsIndirect => ItSetShRegIndirect,
            RUcRegsIndirect => ItSetUconfigRegIndirect,
            _ => 0u,
        };
        if (op == 0 ||
            !TryEmitCommandDwords(ctx, commandBufferAddress, out var commandAddress, Pm4(5, op, 0), (uint)(registersAddress & 0xFFFF_FFFCUL), (uint)(registersAddress >> 32), 0x8000_0000u, registerCount & 0x3FFFu))
        {
            return ReturnPointer(ctx, 0);
        }

        TraceAgc($"agc.dcb_set_{registerSpace}_indirect buf=0x{commandBufferAddress:X16} cmd=0x{commandAddress:X16} regs=0x{registersAddress:X16} count={registerCount}");
        if (_traceAgc && (registerCount & 0x3FFFu) != 0)
        {
            Span<byte> tableSnapshot = stackalloc byte[(int)Math.Min(registerCount & 0x3FFFu, 16u) * 8];
            var tableAddress = registersAddress & ~3UL;
            var captured = ctx.Memory.TryRead(tableAddress, tableSnapshot);
            TraceAgc($"agc.register_table_created cmd=0x{commandAddress:X16} bank={registerSpace} table=0x{tableAddress:X16} count={registerCount & 0x3FFFu} captured={captured} bytes={(captured ? Convert.ToHexString(tableSnapshot) : string.Empty)}");
        }
        return ReturnPointer(ctx, commandAddress);
    }

    private static int DcbSetRegisterDirect(CpuContext ctx, uint op, string registerSpace)
    {
        var commandBufferAddress = ctx[CpuRegister.Rdi];
        // Uc/Cx/Sh register is passed by value as {u32 offset, u32 value} in RSI.
        var packedRegister = ctx[CpuRegister.Rsi];
        var registerOffset = (uint)(packedRegister & 0xFFFF_FFFFUL);
        var registerValue = (uint)(packedRegister >> 32);
        if (commandBufferAddress == 0)
        {
            return ReturnPointer(ctx, 0);
        }

        // The three dwords are one allocation and one guest write, like the other fixed-size builders.
        const uint packetDwords = 3;
        if (!TryEmitCommandDwords(ctx, commandBufferAddress, out var commandAddress, Pm4(packetDwords, op, 0), registerOffset & 0xFFFFu, registerValue))
        {
            return ReturnPointer(ctx, 0);
        }

        TraceAgc(
            $"agc.dcb_set_{registerSpace}_direct buf=0x{commandBufferAddress:X16} cmd=0x{commandAddress:X16} offset=0x{registerOffset:X4} value=0x{registerValue:X8}");
        return ReturnPointer(ctx, commandAddress);
    }

    // Sony SetCf* range writer — SET_CONTEXT_REG packet (same shape as SH range).
    [SysAbiExport(
        Nid = "BVFg3CWU6Eo",
        ExportName = "sceAgcDcbSetCfRegisterRangeDirect",
        Target = Generation.Gen5,
        LibraryName = "libSceAgc")]
    public static int DcbSetCfRegisterRangeDirect(CpuContext ctx) =>
        DcbSetRegisterRangeDirect(ctx, ItSetContextReg, "cf");

    // Logged unresolved as LHFXRrlTPD8 during North Yankton load.
    [SysAbiExport(
        Nid = "LHFXRrlTPD8",
        ExportName = "sceAgcDcbSetCxRegisterDirect",
        Target = Generation.Gen5,
        LibraryName = "libSceAgc")]
    public static int DcbSetCxRegisterDirect(CpuContext ctx) =>
        DcbSetRegisterDirect(ctx, ItSetContextReg, "cx");

    // Matches the 3-dword SET_CONTEXT_REG packet DcbSetCxRegisterDirect writes.
    [SysAbiExport(
        Nid = "1DeUNpRIDDA",
        ExportName = "sceAgcDcbSetCxRegisterDirectGetSize",
        Target = Generation.Gen5,
        LibraryName = "libSceAgc")]
    public static int DcbSetCxRegisterDirectGetSize(CpuContext ctx)
    {
        ctx[CpuRegister.Rax] = 3u * sizeof(uint);
        return (int)ctx[CpuRegister.Rax];
    }

    private static int DcbSetRegisterRangeDirect(CpuContext ctx, uint op, string registerSpace)
    {
        var commandBufferAddress = ctx[CpuRegister.Rdi];
        var offset = (uint)ctx[CpuRegister.Rsi];
        var valuesAddress = ctx[CpuRegister.Rdx];
        var valueCount = (uint)ctx[CpuRegister.Rcx];
        if (commandBufferAddress == 0 || valueCount == 0 || valueCount > 0x3FFE)
        {
            return ReturnPointer(ctx, 0);
        }

        var packetDwords = valueCount + 2;
        if (!TryAllocateCommandDwords(ctx, commandBufferAddress, packetDwords, out var commandAddress))
        {
            return ReturnPointer(ctx, 0);
        }

        // The header, the offset and the payload are one guest write: read the values first (a null
        // source reserves the payload as zeros), then write the whole packet.
        uint[]? rented = null;
        try
        {
            Span<uint> packet = packetDwords <= 2 + InlinePayloadDwords
                ? stackalloc uint[(int)packetDwords]
                : (rented = System.Buffers.ArrayPool<uint>.Shared.Rent((int)packetDwords)).AsSpan(0, (int)packetDwords);
            packet[0] = Pm4(packetDwords, op, 0);
            packet[1] = offset & 0xFFFFu;
            var payload = packet[2..];
            if (valuesAddress != 0)
            {
                if (!ctx.Memory.TryRead(valuesAddress, System.Runtime.InteropServices.MemoryMarshal.AsBytes(payload)))
                {
                    return ReturnPointer(ctx, 0);
                }
            }
            else
            {
                payload.Clear();
            }

            if (!TryWriteCommandDwords(ctx, commandAddress, packet))
            {
                return ReturnPointer(ctx, 0);
            }
        }
        finally
        {
            if (rented is not null)
            {
                System.Buffers.ArrayPool<uint>.Shared.Return(rented);
            }
        }

        TraceAgc(
            $"agc.dcb_set_{registerSpace}_range buf=0x{commandBufferAddress:X16} cmd=0x{commandAddress:X16} offset=0x{offset:X4} count={valueCount}");
        return ReturnPointer(ctx, commandAddress);
    }
}
