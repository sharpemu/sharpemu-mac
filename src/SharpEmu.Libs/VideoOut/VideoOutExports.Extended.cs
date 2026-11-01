// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using SharpEmu.HLE;
using SharpEmu.Libs.Kernel;

namespace SharpEmu.Libs.VideoOut;

// The libSceVideoOut exports beyond the core open/register/flip path: event deletion,
// event counts, the resolution and VRR status queries, the buffer label array, flip
// master links, and the colour and buffer-attribute helpers without a size argument.
public static partial class VideoOutExports
{
    private const int OrbisVideoOutErrorUnsupportedOperation = unchecked((int)0x8029001C);
    // 0x80291000 | ENOMEM, as kytyps5's errno table names it.
    private const int OrbisVideoOutErrorNoMemory = unchecked((int)0x8029100C);
    private const int VideoOutResolutionStatusSize = 0x30;
    private const int VideoOutOutputStatusWithVrrSize = 0x50;
    private const int VideoOutBufferAttribute2OptionOffset = 0x18;
    private const int VideoOutBufferLabelSize = sizeof(ulong);
    // Not raised by this host (VRR is not modelled); distinct from the flip (6),
    // output mode (8) and vblank (0x40) idents.
    private const ulong SceVideoOutInternalEventVrrStatus = 0x10;

    private static readonly object _bufferLabelAllocationGate = new();
    private static readonly ConditionalWeakTable<VideoOutPortState, VideoOutPortExtendedState> _portExtendedState = new();

    private sealed class VideoOutPortExtendedState
    {
        public List<FlipEventRegistration> VrrStatusEvents { get; } = new();
        // Guest address of this port's VIDEO_OUT_BUFFER_NUM_MAX 64-bit labels; 0 until first queried.
        public ulong BufferLabelAddress;
        public ICpuMemory? BufferLabelMemory;
        // Handle of the port whose flips this port's flips follow; 0 when unlinked.
        public int FlipMasterHandle;
    }

    private static VideoOutPortExtendedState GetExtendedStateLocked(VideoOutPortState port) =>
        _portExtendedState.GetValue(port, static _ => new VideoOutPortExtendedState());

    [SysAbiExport(
        Nid = "-Ozn0F1AFRg",
        ExportName = "sceVideoOutDeleteFlipEvent",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceVideoOut")]
    public static int VideoOutDeleteFlipEvent(CpuContext ctx) =>
        DeleteVideoOutEvent(ctx, static port => port.FlipEvents, SceVideoOutInternalEventFlip);

    [SysAbiExport(
        Nid = "vwtVpHQKelI",
        ExportName = "sceVideoOutDeleteOutputModeEvent",
        Target = Generation.Gen5,
        LibraryName = "libSceVideoOut")]
    public static int VideoOutDeleteOutputModeEvent(CpuContext ctx) =>
        DeleteVideoOutEvent(ctx, static port => port.OutputModeEvents, SceVideoOutInternalEventOutputMode);

    #pragma warning disable SHEM006
    // The name hashes to the NID titles import but is absent from the catalog.
    [SysAbiExport(
        Nid = "LibwuIonIBw",
        ExportName = "sceVideoOutAddVrrActiveStatusEvent",
        Target = Generation.Gen5,
        LibraryName = "libSceVideoOut")]
    public static int VideoOutAddVrrActiveStatusEvent(CpuContext ctx)
    {
        var equeue = ctx[CpuRegister.Rdi];
        var handle = unchecked((int)ctx[CpuRegister.Rsi]);
        var userData = ctx[CpuRegister.Rdx];
        if (!TryGetPort(handle, out var port))
        {
            return OrbisVideoOutErrorInvalidHandle;
        }

        if (!KernelEventQueueCompatExports.IsValidEqueue(equeue))
        {
            return OrbisVideoOutErrorInvalidEventQueue;
        }

        // The console signals this event when the output's VRR state changes. The host
        // output always runs at a fixed rate, so a registration never fires; it is kept so
        // that sceVideoOutDeleteVrrActiveStatusEvent removes exactly what was added.
        lock (_stateGate)
        {
            var events = GetExtendedStateLocked(port).VrrStatusEvents;
            var existingIndex = events.FindIndex(registration => registration.Equeue == equeue);
            if (existingIndex >= 0)
            {
                events[existingIndex] = new FlipEventRegistration(equeue, userData);
            }
            else
            {
                events.Add(new FlipEventRegistration(equeue, userData));
            }
        }

        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "-q-vbO859Tw",
        ExportName = "sceVideoOutDeleteVrrActiveStatusEvent",
        Target = Generation.Gen5,
        LibraryName = "libSceVideoOut")]
    public static int VideoOutDeleteVrrActiveStatusEvent(CpuContext ctx) =>
        DeleteVideoOutEvent(ctx, static port => GetExtendedStateLocked(port).VrrStatusEvents, SceVideoOutInternalEventVrrStatus);

    // int sceVideoOutGetOutputStatusWithVrr(int handle, void* status): an 80-byte status whose
    // first 48 bytes are sceVideoOutGetOutputStatus's. The console fills bytes 48..53 from a
    // VRR query whose fields are not established; with no VRR on this host they stay zero.
    [SysAbiExport(
        Nid = "gWT7X8H0bYs",
        ExportName = "sceVideoOutGetOutputStatusWithVrr",
        Target = Generation.Gen5,
        LibraryName = "libSceVideoOut")]
    public static int VideoOutGetOutputStatusWithVrr(CpuContext ctx)
    {
        var handle = unchecked((int)ctx[CpuRegister.Rdi]);
        var statusAddress = ctx[CpuRegister.Rsi];
        if (statusAddress == 0)
        {
            return OrbisVideoOutErrorInvalidAddress;
        }

        if (!TryGetPort(handle, out var port))
        {
            return OrbisVideoOutErrorInvalidHandle;
        }

        NoteDisplayQueried();
        Span<byte> status = stackalloc byte[VideoOutOutputStatusWithVrrSize];
        status.Clear();
        WriteOutputStatus(status[..VideoOutOutputStatusSize], HostVideoHost.CurrentOptions.GuestResolutionClass,
            port.RefreshRate, HostVideoHost.IsHdrOutputSupported);
        return ctx.Memory.TryWrite(statusAddress, status)
            ? (int)OrbisGen2Result.ORBIS_GEN2_OK
            : (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
    }
    #pragma warning restore SHEM006

    // int sceVideoOutGetEventCount(const SceKernelEvent* ev): bits 12..15 of the event data,
    // the number of times the event fired since the queue last delivered it (saturating at 15).
    [SysAbiExport(
        Nid = "Mt4QHHkxkOc",
        ExportName = "sceVideoOutGetEventCount",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceVideoOut")]
    public static int VideoOutGetEventCount(CpuContext ctx)
    {
        var eventAddress = ctx[CpuRegister.Rdi];
        if (eventAddress == 0)
        {
            return OrbisVideoOutErrorInvalidAddress;
        }

        // SceKernelEvent: ident u64 @0x00, filter s16 @0x08, flags u16 @0x0A,
        // fflags u32 @0x0C, data s64 @0x10, udata @0x18.
        if (!TryReadInt16(ctx, eventAddress + 0x08, out var filter) ||
            !ctx.TryReadUInt64(eventAddress + 0x10, out var data))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        if (filter != OrbisKernelEventFilterVideoOut)
        {
            return OrbisVideoOutErrorInvalidEvent;
        }

        return (int)((data >> 12) & 0xFUL);
    }

    // int sceVideoOutGetResolutionStatus(int handle, SceVideoOutResolutionStatus* status)
    //   0x00 u32 fullWidth      0x04 u32 fullHeight
    //   0x08 u32 paneWidth      0x0C u32 paneHeight
    //   0x10 u64 refreshRate    (SCE_VIDEO_OUT_REFRESH_RATE_*, as sceVideoOutGetOutputStatus)
    //   0x18 f32 screenSizeInInch
    //   0x1C u16 flags          0x1E u16 reserved0
    //   0x20 u32 reserved1[3]   (0x2C..0x30 padding) — 48 bytes.
    [SysAbiExport(
        Nid = "6kPnj51T62Y",
        ExportName = "sceVideoOutGetResolutionStatus",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceVideoOut")]
    public static int VideoOutGetResolutionStatus(CpuContext ctx)
    {
        var handle = unchecked((int)ctx[CpuRegister.Rdi]);
        var statusAddress = ctx[CpuRegister.Rsi];
        if (statusAddress == 0)
        {
            return OrbisVideoOutErrorInvalidAddress;
        }

        if (!TryGetPort(handle, out var port))
        {
            return OrbisVideoOutErrorInvalidHandle;
        }

        uint width;
        uint height;
        uint refreshRate;
        lock (_stateGate)
        {
            width = port.OutputWidth;
            height = port.OutputHeight;
            refreshRate = port.RefreshRate;
        }

        NoteDisplayQueried();
        Span<byte> status = stackalloc byte[VideoOutResolutionStatusSize];
        status.Clear();
        BinaryPrimitives.WriteUInt32LittleEndian(status[0x00..], width);
        BinaryPrimitives.WriteUInt32LittleEndian(status[0x04..], height);
        // The pane is the full output: sceVideoOutSetWindowModeMargins shapes flip timing
        // here, not the presented area.
        BinaryPrimitives.WriteUInt32LittleEndian(status[0x08..], width);
        BinaryPrimitives.WriteUInt32LittleEndian(status[0x0C..], height);
        BinaryPrimitives.WriteUInt64LittleEndian(status[0x10..], refreshRate >= 119
            ? SceVideoOutRefreshRate119_88Hz
            : SceVideoOutRefreshRate59_94Hz);
        // screenSizeInInch stays 0.0, EDID's value for an undefined size: the host display's
        // physical size is not read. flags stays 0; its bits are not established.
        return ctx.Memory.TryWrite(statusAddress, status)
            ? (int)OrbisGen2Result.ORBIS_GEN2_OK
            : (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
    }

    // int sceVideoOutGetBufferLabelAddress(int handle, uintptr_t* address): stores the address
    // of the port's array of VIDEO_OUT_BUFFER_NUM_MAX 64-bit labels (label i at base + i * 8)
    // and returns the label count. A label is cleared when its buffer is registered and when a
    // completed flip replaces it as the displayed buffer.
    [SysAbiExport(
        Nid = "OcQybQejHEY",
        ExportName = "sceVideoOutGetBufferLabelAddress",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceVideoOut")]
    public static int VideoOutGetBufferLabelAddress(CpuContext ctx)
    {
        var handle = unchecked((int)ctx[CpuRegister.Rdi]);
        var outputAddress = ctx[CpuRegister.Rsi];
        if (outputAddress == 0)
        {
            return OrbisVideoOutErrorInvalidAddress;
        }

        if (!TryGetPort(handle, out var port))
        {
            return OrbisVideoOutErrorInvalidHandle;
        }

        ulong labelAddress;
        lock (_bufferLabelAllocationGate)
        {
            lock (_stateGate)
            {
                labelAddress = GetExtendedStateLocked(port).BufferLabelAddress;
            }

            if (labelAddress == 0)
            {
                const int labelBytes = MaxDisplayBuffers * VideoOutBufferLabelSize;
                if (!TryAllocateBufferLabels(ctx, labelBytes, out labelAddress) ||
                    !ctx.Memory.TryWrite(labelAddress, stackalloc byte[labelBytes]))
                {
                    return OrbisVideoOutErrorNoMemory;
                }

                lock (_stateGate)
                {
                    if (!_ports.ContainsKey(handle))
                    {
                        return OrbisVideoOutErrorInvalidHandle;
                    }

                    var state = GetExtendedStateLocked(port);
                    state.BufferLabelAddress = labelAddress;
                    state.BufferLabelMemory = ctx.Memory;
                }
            }
        }

        return ctx.TryWriteUInt64(outputAddress, labelAddress)
            ? MaxDisplayBuffers
            : (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
    }

    // The label array is guest memory the title and its GPU commands read and write.
    private static bool TryAllocateBufferLabels(CpuContext ctx, ulong length, out ulong address)
    {
        var allocator = ctx.Memory as IGuestMemoryAllocator;
        if (allocator is null && ctx.Memory is ICpuMemoryWrapper { Inner: IGuestMemoryAllocator inner })
        {
            allocator = inner;
        }

        if (allocator is not null && allocator.TryAllocateGuestMemory(length, VideoOutBufferLabelSize, out address))
        {
            return true;
        }

        return KernelMemoryCompatExports.TryAllocateHleData(ctx, length, VideoOutBufferLabelSize, out address);
    }

    // Registration starts a buffer's label at zero. Called with _stateGate held.
    private static void ResetBufferLabelsLocked(VideoOutPortState port, int startIndex, int count)
    {
        if (!_portExtendedState.TryGetValue(port, out var state) ||
            state.BufferLabelAddress == 0 || state.BufferLabelMemory is null)
        {
            return;
        }

        for (var i = 0; i < count; i++)
        {
            StoreBufferLabel(state, startIndex + i);
        }
    }

    // A completed flip releases the buffer it replaced on screen. Called with _stateGate held,
    // before the port's current buffer is updated.
    private static void ReleaseBufferLabelLocked(VideoOutPortState port, int previousBuffer)
    {
        if (previousBuffer < 0 ||
            !_portExtendedState.TryGetValue(port, out var state) ||
            state.BufferLabelAddress == 0 || state.BufferLabelMemory is null)
        {
            return;
        }

        StoreBufferLabel(state, previousBuffer);
    }

    private static void StoreBufferLabel(VideoOutPortExtendedState state, int index)
    {
        if ((uint)index >= MaxDisplayBuffers)
        {
            return;
        }

        _ = state.BufferLabelMemory!.TryWrite(
            state.BufferLabelAddress + (ulong)(index * VideoOutBufferLabelSize),
            stackalloc byte[VideoOutBufferLabelSize]);
    }

    // int sceVideoOutSetFlipMaster(int slaveHandle, int masterHandle). Validation and results
    // follow kytyps5's driver: both ports open, distinct, neither already part of another
    // link, and no flip pending on either. Only main-bus ports open on this host and they all
    // present to the one host output, so the link is recorded but does not regroup flips.
    [SysAbiExport(
        Nid = "WkYtyOg30do",
        ExportName = "sceVideoOutSetFlipMaster",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceVideoOut")]
    public static int VideoOutSetFlipMaster(CpuContext ctx)
    {
        var slaveHandle = unchecked((int)ctx[CpuRegister.Rdi]);
        var masterHandle = unchecked((int)ctx[CpuRegister.Rsi]);
        lock (_stateGate)
        {
            if (!_ports.TryGetValue(slaveHandle, out var slave) ||
                !_ports.TryGetValue(masterHandle, out var master))
            {
                return OrbisVideoOutErrorInvalidHandle;
            }

            if (ReferenceEquals(slave, master))
            {
                return OrbisVideoOutErrorUnsupportedOperation;
            }

            if (GetFlipMasterLocked(slave) != 0 || HasFlipSlavesLocked(slaveHandle) || GetFlipMasterLocked(master) != 0)
            {
                return OrbisVideoOutErrorUnsupportedOperation;
            }

            if (slave.FlipPendingCount != 0 || master.FlipPendingCount != 0)
            {
                return OrbisVideoOutErrorResourceBusy;
            }

            GetExtendedStateLocked(slave).FlipMasterHandle = masterHandle;
        }

        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    // int sceVideoOutClearFlipMaster(int slaveHandle). Neither reference establishes the
    // signature; it is taken as the slave handle sceVideoOutSetFlipMaster linked. Clearing a
    // port without a link succeeds, as deleting an absent event does.
    [SysAbiExport(
        Nid = "T0ynQY3mH-0",
        ExportName = "sceVideoOutClearFlipMaster",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceVideoOut")]
    public static int VideoOutClearFlipMaster(CpuContext ctx)
    {
        var slaveHandle = unchecked((int)ctx[CpuRegister.Rdi]);
        lock (_stateGate)
        {
            if (!_ports.TryGetValue(slaveHandle, out var slave))
            {
                return OrbisVideoOutErrorInvalidHandle;
            }

            if (_portExtendedState.TryGetValue(slave, out var state))
            {
                state.FlipMasterHandle = 0;
            }
        }

        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    // A link whose master has closed no longer counts.
    private static int GetFlipMasterLocked(VideoOutPortState port) =>
        _portExtendedState.TryGetValue(port, out var state) &&
        state.FlipMasterHandle != 0 &&
        _ports.ContainsKey(state.FlipMasterHandle)
            ? state.FlipMasterHandle
            : 0;

    private static bool HasFlipSlavesLocked(int handle)
    {
        foreach (var port in _ports.Values)
        {
            if (GetFlipMasterLocked(port) == handle)
            {
                return true;
            }
        }

        return false;
    }

    internal static int GetFlipMasterForTests(int slaveHandle)
    {
        lock (_stateGate)
        {
            return _ports.TryGetValue(slaveHandle, out var port) ? GetFlipMasterLocked(port) : 0;
        }
    }

    // int sceVideoOutColorSettingsSetGamma(SceVideoOutColorSettings* settings, float gamma):
    // the sized variant without its size argument; same registers, same checks.
    // SceVideoOutColorSettings: f32 gamma @0x00, u32 reserved[3] @0x04 — 16 bytes.
    [SysAbiExport(
        Nid = "7wEmHVceFNg",
        ExportName = "sceVideoOutColorSettingsSetGamma",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceVideoOut")]
    public static int VideoOutColorSettingsSetGammaUnsized(CpuContext ctx) => VideoOutColorSettingsSetGamma(ctx);

    // int sceVideoOutAdjustColor(int handle, const SceVideoOutColorSettings* settings).
    [SysAbiExport(
        Nid = "ipf9hrW6kBY",
        ExportName = "sceVideoOutAdjustColor",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceVideoOut")]
    public static int VideoOutAdjustColorUnsized(CpuContext ctx) => VideoOutAdjustColor(ctx);

    // void sceVideoOutAddBufferAttributeOption(SceVideoOutBufferAttribute2* attribute, u64 option):
    // ORs the option into the attribute's option word (offset 0x18).
    [SysAbiExport(
        Nid = "MBlv8YWoyOk",
        ExportName = "sceVideoOutAddBufferAttributeOption",
        Target = Generation.Gen5,
        LibraryName = "libSceVideoOut")]
    public static int VideoOutAddBufferAttributeOption(CpuContext ctx)
    {
        var attributeAddress = ctx[CpuRegister.Rdi];
        var option = ctx[CpuRegister.Rsi];
        if (attributeAddress == 0)
        {
            return OrbisVideoOutErrorInvalidAddress;
        }

        var optionAddress = attributeAddress + VideoOutBufferAttribute2OptionOffset;
        return ctx.TryReadUInt64(optionAddress, out var current) &&
            ctx.TryWriteUInt64(optionAddress, current | option)
            ? (int)OrbisGen2Result.ORBIS_GEN2_OK
            : (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
    }

    // Shared by the Delete*Event exports: (SceKernelEqueue eq, int handle). An unknown handle
    // fails before the queue is looked at, as in sceVideoOutAdd*Event. Deleting an event the
    // port never added (or already deleted) returns SCE_OK, as the kernel's ENOENT is mapped to
    // success by both reference drivers.
    private static int DeleteVideoOutEvent(
        CpuContext ctx,
        Func<VideoOutPortState, List<FlipEventRegistration>> selectEvents,
        ulong ident)
    {
        var equeue = ctx[CpuRegister.Rdi];
        var handle = unchecked((int)ctx[CpuRegister.Rsi]);
        if (!TryGetPort(handle, out var port))
        {
            return OrbisVideoOutErrorInvalidHandle;
        }

        if (!KernelEventQueueCompatExports.IsValidEqueue(equeue))
        {
            return OrbisVideoOutErrorInvalidEventQueue;
        }

        bool lastRegistration;
        lock (_stateGate)
        {
            if (selectEvents(port).RemoveAll(registration => registration.Equeue == equeue) == 0)
            {
                return (int)OrbisGen2Result.ORBIS_GEN2_OK;
            }

            // Ports share the queue's (ident, filter) slot, so an occurrence another port
            // still feeds stays queued.
            lastRegistration = !_ports.Values.Any(other =>
                selectEvents(other).Exists(registration => registration.Equeue == equeue));
        }

        if (lastRegistration)
        {
            DiscardQueuedVideoOutEvent(equeue, ident);
        }

        TraceVideoOut($"videoout.delete_event eq=0x{equeue:X16} handle={handle} ident=0x{ident:X}");
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    // Removing a kevent also drops its undelivered occurrence. VideoOut events are queued
    // directly rather than through a queue registration, so a transient registration of the
    // same (ident, filter) is made and deleted, which drops the queued occurrence with it.
    private static void DiscardQueuedVideoOutEvent(ulong equeue, ulong ident)
    {
        if (KernelEventQueueCompatExports.RegisterEvent(equeue, ident, OrbisKernelEventFilterVideoOut, 0))
        {
            _ = KernelEventQueueCompatExports.DeleteRegisteredEvent(equeue, ident, OrbisKernelEventFilterVideoOut);
        }
    }
}
