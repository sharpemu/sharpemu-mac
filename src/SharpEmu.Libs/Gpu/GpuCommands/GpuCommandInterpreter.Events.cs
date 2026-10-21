// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.GpuCommands.Packets;

namespace SharpEmu.Libs.Gpu.GpuCommands;

public sealed partial class GpuCommandInterpreter
{
    private const uint MarkerMagic = 0x6875_0000u;
    private const uint UserDataMarkerBufferResource = 1;
    private const uint UserDataMarkerRegion = 2;

    internal uint EventWritePacket(in PacketContext packet, ReadOnlySpan<uint> payload)
    {
        var eventIndex = (payload[0] >> 8) & 0x7u;
        var eventType = payload[0] & 0x3Fu;
        ulong eventAddress = 0;
        if (eventType is 0x38 or 0x39)
        {
            if (packet.Length != 4)
            {
                throw _host.Fatal($"The occlusion event packet length is not supported: length={packet.Length} address=0x{packet.PacketAddress:X16}.");
            }

            eventAddress = Address(payload[1], payload[2]);
        }

        if (eventType == 0x3A && packet.Length != 2)
        {
            throw _host.Fatal($"The occlusion reset packet length is not supported: length={packet.Length} address=0x{packet.PacketAddress:X16}.");
        }

        RaiseEvent(eventType, eventIndex, eventAddress);
        return packet.Length - 1;
    }

    internal uint EventWriteEndOfPipePacket(in PacketContext packet, ReadOnlySpan<uint> payload)
    {
        RequireHeader(packet, PacketHeader.Make(6, PacketOpcode.EventWriteEndOfPipe));
        var control = payload[0];
        var destination = payload[1] | ((ulong)(payload[2] & 0xFFFFu) << 32);
        WriteEndOfPipe(
            is64Bit: true,
            cachePolicy: (control >> 25) & 0x3u,
            eventWriteDestination: ((control >> 23) & 0x10u) | ((payload[2] >> 16) & 0x01u),
            eopEventType: control & 0x3Fu,
            cacheAction: (control >> 12) & 0x3Fu,
            eventIndex: (control >> 8) & 0x7u,
            eventWriteSource: (payload[2] >> 29) & 0x7u,
            destination,
            value: Address(payload[3], payload[4]),
            interruptSelector: (payload[2] >> 24) & 0x7u,
            interruptContextId: 0);
        return 5;
    }

    internal uint EventWriteEndOfShaderPacket(in PacketContext packet, ReadOnlySpan<uint> payload)
    {
        // The builder sets header bit 1, so the whole header is compared.
        RequireHeader(packet, 0xC003_4802u);
        var control = payload[0];
        var destination = payload[1] | ((ulong)(payload[2] & 0xFFFFu) << 32);
        WriteEndOfPipe(
            is64Bit: false,
            cachePolicy: (control >> 25) & 0x3u,
            eventWriteDestination: 0,
            eopEventType: control & 0x3Fu,
            cacheAction: (control >> 12) & 0x3Fu,
            eventIndex: (control >> 8) & 0x7u,
            eventWriteSource: (payload[2] >> 29) & 0x7u,
            destination,
            value: payload[3],
            interruptSelector: (payload[2] >> 24) & 0x7u,
            interruptContextId: 0);
        return 4;
    }

    // Native release-memory body: event control, then destination and selectors.
    internal uint ReleaseMemoryNativePacket(in PacketContext packet, ReadOnlySpan<uint> payload)
    {
        RequireHeader(packet, PacketHeader.Make(8, PacketOpcode.ReleaseMemory));
        ReleaseMemoryFromNativeFields(payload);
        return 7;
    }

    private void ReleaseMemoryFromNativeFields(ReadOnlySpan<uint> payload)
    {
        var eventControl = payload[0];
        var control = payload[1];
        ReleaseMemory(
            cachePolicy: (eventControl >> 25) & 0x3u,
            eopEventType: eventControl & 0x3Fu,
            eventIndex: (eventControl >> 8) & 0x7u,
            gcrControl: (eventControl >> 12) & 0xFFFu,
            releaseDestination: (control >> 16) & 0x3u,
            dataSelection: (control >> 29) & 0x7u,
            interruptSelector: (control >> 24) & 0x7u,
            destination: Address(payload[2], payload[3]),
            value: Address(payload[4], payload[5]),
            interruptContextId: payload[6] & 0x07FF_FFFFu);
    }

    // The library builder packs the wrapped form as action | policy << 8, then
    // gcr | selection << 16 | interrupt << 24; the event index follows the action.
    internal uint ReleaseMemoryWrappedPacket(in PacketContext packet, ReadOnlySpan<uint> payload)
    {
        RequireHeader(packet, PacketHeader.Make(8, PacketOpcode.Nop, PacketCustomCode.ReleaseMemory));
        var action = payload[0] & 0x3Fu;
        var control = payload[1];
        var gcrControl = control & 0xFFFFu;
        var dataSelection = (control >> 16) & 0xFFu;
        var interruptSelector = (control >> 24) & 0xFFu;
        var destination = Address(payload[2], payload[3]);
        var data = Address(payload[4], payload[5]);
        var contextId = payload[6] & 0x07FF_FFFFu;
        var eventIndex = action >= 0x2F ? 6u : 5u;

        // Selectors 5 and 6 raise only when the label already reached the data; the shared
        // release path compares it, as it does for the native packet.
        ReleaseMemory(0, action, eventIndex, gcrControl, 0, dataSelection, interruptSelector, destination, data, contextId);
        return 7;
    }

    // CP_COHER_CNTL cache actions: TC_NC/TC_WC/TC_INV_METADATA (3-5), TCL1_VOL (15),
    // TC_WB (18), TCL1 (22), TC (23), CB (25), DB (26), SH_KCACHE/_VOL (27-28),
    // SH_ICACHE (29), SH_KCACHE_WB (30). The DEST_BASE enables (6-14) only name targets.
    private const uint CoherCacheActionMask =
        (0x7u << 3) | (1u << 15) | (1u << 18) | (1u << 22) | (1u << 23) | (0x3Fu << 25);

    // GCR_CNTL of a gfx10 ACQUIRE_MEM: GLI_INV (0-1), GL1_RANGE (2-3), GLM_WB/INV (4-5),
    // GLK_WB/INV (6-7), GLV_INV (8), GL1_INV (9), GL2_US (10), GL2_RANGE (11-12),
    // GL2_DISCARD (13), GL2_INV (14), GL2_WB (15), SEQ (16-17).
    private const uint AcquireGcrKnownMask = 0x3FFFFu;
    private const uint AcquireGcrCacheActionMask = 0x3u | (0xFFu << 4) | (0x7u << 13);

    private static int _unknownAcquireGcrWarnings;

    /// <summary>Acquires since startup that invalidated or wrote back a cache and so ordered memory.</summary>
    public static long AcquireBarriers => Interlocked.Read(ref _acquireBarriers);

    private static long _acquireBarriers;

    internal uint AcquireMemoryPacket(in PacketContext packet, ReadOnlySpan<uint> payload)
    {
        // The library's wrapped form and the gfx10 packet share one body: COHER_CNTL with the
        // engine in bit 31, the size and base in 256-byte units, the poll interval, then
        // GCR_CNTL. The gfx9 packet is the same body without GCR_CNTL.
        if (packet.Header == PacketHeader.Make(8, PacketOpcode.Nop, PacketCustomCode.AcquireMemory) ||
            packet.Header == PacketHeader.Make(8, PacketOpcode.AcquireMemory))
        {
            AcquireMemory(payload[0] & 0x7FFF_FFFFu, payload[6]);
            return 7;
        }

        if (packet.Header == PacketHeader.Make(7, PacketOpcode.AcquireMemory))
        {
            AcquireMemory(payload[0] & 0x7FFF_FFFFu, 0);
            return 6;
        }

        throw _host.Fatal($"The acquire-memory header is not supported: header=0x{packet.Header:X8} address=0x{packet.PacketAddress:X16}.");
    }

    // An acquire makes earlier stores visible to later work. When it invalidates or writes
    // back a cache the shaders read through, that is a memory barrier between the work on
    // either side; either way the later work may read what a fragment shader stored in a
    // pass the executor left open. The base and size narrow it to a range on the hardware;
    // a Metal barrier is scoped by resource kind, not address, so the whole range is
    // ordered, which is never weaker.
    internal void AcquireMemory(uint coherControl, uint gcrControl)
    {
        var unknown = gcrControl & ~AcquireGcrKnownMask;
        if (unknown != 0 && Interlocked.Increment(ref _unknownAcquireGcrWarnings) <= 16)
        {
            Console.Error.WriteLine($"[LOADER][WARN] command_stream.acquire_mem_unknown_gcr bits=0x{unknown:X8}");
        }

        if ((coherControl & CoherCacheActionMask) != 0 || (gcrControl & AcquireGcrCacheActionMask) != 0)
        {
            Interlocked.Increment(ref _acquireBarriers);
            _host.EmitGlobalBarrier();
        }

        _host.ReleaseShaderWrites();
    }

    // SURFACE_SYNC is the gfx9 acquire: COHER_CNTL, size, base and poll interval.
    internal uint SurfaceSyncPacket(in PacketContext packet, ReadOnlySpan<uint> payload)
    {
        if (packet.Length != 5)
        {
            throw _host.Fatal($"The surface-sync packet length is not supported: length={packet.Length} address=0x{packet.PacketAddress:X16}.");
        }

        AcquireMemory(payload[0] & 0x7FFF_FFFFu, 0);
        return 4;
    }

    // CONTEXT_CONTROL enables register loading and shadowing from memory images, which the
    // command processor does for a preempted context. Nothing here is preempted mid-buffer:
    // every register write lands in this queue's banks as its packet runs, so the enables
    // select nothing to load. A load or shadow request is reported once.
    private static int _contextControlWarnings;

    internal uint ContextControlPacket(in PacketContext packet, ReadOnlySpan<uint> payload)
    {
        if (packet.Length != 3)
        {
            throw _host.Fatal($"The context-control packet length is not supported: length={packet.Length} address=0x{packet.PacketAddress:X16}.");
        }

        if (((payload[0] | payload[1]) & 0x7FFF_FFFFu) != 0 && Interlocked.Increment(ref _contextControlWarnings) <= 4)
        {
            Console.Error.WriteLine(
                $"[LOADER][WARN] command_stream.context_control load=0x{payload[0]:X8} shadow=0x{payload[1]:X8} (register images are not loaded)");
        }

        return 2;
    }

    // INDEX_ATTRIBUTES_INDIRECT loads the index base, the index buffer size and the index
    // type from a 16-byte record at attribute base + index * 16, an operand earlier work
    // may write (an ExecuteIndirect-style argument buffer).
    internal uint IndexAttributesIndirectPacket(in PacketContext packet, ReadOnlySpan<uint> payload)
    {
        if (packet.Length != 4 || (payload[0] & 0xFu) != 0 || (payload[2] >> 16) != 0)
        {
            throw _host.Fatal(
                $"The index-attributes packet is not supported: length={packet.Length} control=0x{payload[0]:X8} " +
                $"index=0x{(packet.Length > 3 ? payload[2] : 0):X8} address=0x{packet.PacketAddress:X16}.");
        }

        var record = Address(payload[0], payload[1]) + ((ulong)(payload[2] & 0xFFFFu) * 16);
        Span<uint> words = stackalloc uint[4];
        if (!TryReadOperandOrSuspend(record, System.Runtime.InteropServices.MemoryMarshal.AsBytes(words)))
        {
            return 3;
        }

        if (words[3] > 3)
        {
            throw _host.Fatal($"The index-attributes record has an unknown index type: type=0x{words[3]:X8} record=0x{record:X16}.");
        }

        SetIndexBaseAddress(Address(words[0], words[1]));
        SetIndexBufferSize(words[2]);
        SetIndexType(words[3]);
        return 3;
    }

    internal uint NopPacket(in PacketContext packet, ReadOnlySpan<uint> payload)
    {
        var customCode = packet.CustomCode;
        if (customCode == PacketCustomCode.Zero)
        {
            if ((payload[0] & 0xFFFF_0000u) == MarkerMagic)
            {
                return MarkerPacket(packet, payload);
            }

            return packet.Length - 1;
        }

        var handler = PacketDispatchTable.CustomCodes[customCode];
        if (handler is null)
        {
            throw _host.Fatal($"The custom packet code is unknown: offset=0x{packet.Offset:X5} code=0x{customCode:X2} header=0x{packet.Header:X8} address=0x{packet.PacketAddress:X16}.");
        }

        return handler(this, in packet, payload);
    }

    private uint MarkerPacket(in PacketContext packet, ReadOnlySpan<uint> payload)
    {
        var markerId = payload[0] & 0xFFFu;
        switch (markerId)
        {
            case 0x0:
                break;
            case 0x4:
                UserDataMarker = UserDataMarkerBufferResource;
                break;
            case 0xD:
                UserDataMarker = UserDataMarkerRegion;
                break;
            case 0x777:
                SubmitFlip();
                break;
            case 0x778:
                SubmitFlipWithLabel(Address(payload[1], payload[2]), payload[3]);
                break;
            case 0x781:
                SubmitFlipWithInterrupt(payload[4], payload[5], Address(payload[1], payload[2]), payload[3]);
                break;
            default:
                // A marker id this interpreter has no side effect for (profiling/debug annotations such as
                // 0x6EC and 0x80C in Demon's Souls). The packet carries no work, so it is consumed; the id is
                // reported once so a marker that later turns out to matter is still visible.
                NoteUnknownMarker(markerId, packet);
                break;
        }

        return packet.Length - 1;
    }

    private static readonly System.Collections.Concurrent.ConcurrentDictionary<uint, bool> _reportedUnknownMarkers = new();

    private static void NoteUnknownMarker(uint markerId, in PacketContext packet)
    {
        if (_reportedUnknownMarkers.TryAdd(markerId, true))
        {
            Console.Error.WriteLine(
                $"[LOADER][WARN] command_stream.marker_ignored marker=0x{markerId:X} offset=0x{packet.Offset:X5} address=0x{packet.PacketAddress:X16}");
        }
    }

    internal uint PushMarkerPacket(in PacketContext packet, ReadOnlySpan<uint> payload) => packet.Length - 1;

    internal uint PopMarkerPacket(in PacketContext packet, ReadOnlySpan<uint> payload) => packet.Length - 1;

    internal uint FlipPacket(in PacketContext packet, ReadOnlySpan<uint> payload)
    {
        RequireHeader(packet, PacketHeader.Make(6, PacketOpcode.Nop, PacketCustomCode.Flip));
        SetFlip(new FlipRequest((int)payload[0], (int)payload[1], (int)payload[2], (long)Address(payload[3], payload[4])));
        SubmitFlip();
        return 5;
    }

    internal uint WaitFlipDonePacket(in PacketContext packet, ReadOnlySpan<uint> payload)
    {
        RequireHeader(packet, PacketHeader.Make(7, PacketOpcode.Nop, PacketCustomCode.WaitFlipDone));
        WaitForFlip(payload[0], payload[1]);
        return 6;
    }

    internal uint ClearStatePacket(in PacketContext packet, ReadOnlySpan<uint> payload)
    {
        RequireHeader(packet, PacketHeader.Make(2, PacketOpcode.ClearState));
        if ((payload[0] & ~0xFu) != 0)
        {
            throw _host.Fatal($"The clear-state flags are not supported: flags=0x{payload[0]:X8} address=0x{packet.PacketAddress:X16}.");
        }

        var before = TypedRegisters.Context;
        TypedRegisters.ApplyContextState(ContextStateOperation.Clear);
        if (Rendering.ScissorTrace.Instance is { Active: true } trace)
            trace.ContextOperation(SubmitId, QueueId, packet.PacketAddress, $"ClearState(flags=0x{payload[0]:X8})", TypedRegisters, before);
        return 1;
    }

    internal uint ContextStatePacket(in PacketContext packet, ReadOnlySpan<uint> payload)
    {
        if (packet.Length is not (3 or 5) || payload[0] > (uint)ContextStateOperation.PushClear)
        {
            throw _host.Fatal($"The context-state packet is not supported: length={packet.Length} operation={payload[0]} address=0x{packet.PacketAddress:X16}.");
        }

        var before = TypedRegisters.Context;
        var operation = (ContextStateOperation)payload[0];
        TypedRegisters.ApplyContextState(operation);
        if (Rendering.ScissorTrace.Instance is { Active: true } trace)
            trace.ContextOperation(SubmitId, QueueId, packet.PacketAddress, operation.ToString(), TypedRegisters, before);
        return packet.Length - 1;
    }

    internal uint DispatchResetPacket(in PacketContext packet, ReadOnlySpan<uint> payload)
    {
        RequireHeader(packet, PacketHeader.Make(2, PacketOpcode.Nop, PacketCustomCode.DispatchReset));
        var before = TypedRegisters.Context;
        ResetForQueueReset();
        if (Rendering.ScissorTrace.Instance is { Active: true } trace)
            trace.ContextOperation(SubmitId, QueueId, packet.PacketAddress, "DispatchReset", TypedRegisters, before);
        return 1;
    }

    internal uint DrawResetPacket(in PacketContext packet, ReadOnlySpan<uint> payload)
    {
        var before = TypedRegisters.Context;
        ResetForQueueReset();
        if (Rendering.ScissorTrace.Instance is { Active: true } trace)
            trace.ContextOperation(SubmitId, QueueId, packet.PacketAddress, "DrawReset", TypedRegisters, before);
        return packet.Length - 1;
    }

    internal uint PfpSyncMePacket(in PacketContext packet, ReadOnlySpan<uint> payload)
    {
        RequireHeader(packet, PacketHeader.Make(2, PacketOpcode.PfpSyncMe));
        return 1;
    }
}
