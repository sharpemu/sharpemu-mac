// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Metal.V2;

internal sealed partial class MetalRenderHost
{
    private ulong _gdsSerial;
    private long _gdsGpuFills, _gdsGpuCopies, _gdsHostFallbacks;

    private long _owedGdsSnapshots, _gdsAlignmentFallbacks, _gdsBackingFallbacks, _gdsDisabledFallbacks, _gdsSnapshotWaits;
    private ulong _gdsHostSnapshotSerial;

    private void WaitForGdsSnapshot()
    {
        if (Volatile.Read(ref _owedGdsSnapshots) == 0) return;
        _gdsSnapshotWaits++;
        WaitForCommittedSerial(_gdsHostSnapshotSerial, MetalWaitReason.HostPublication);
    }

    private void WaitForCpuGds()
    {
        WaitForGdsSnapshot();
        if (_gdsSerial != 0) WaitForCommittedSerial(_gdsSerial, MetalWaitReason.GdsRead, gpuBytesOnly: true);
    }

    private bool GdsOnGpu()
    {
        if (GpuWordsEnabled) { WaitForGdsSnapshot(); return true; }
        _gdsHostFallbacks++;
        _gdsDisabledFallbacks++;
        return false;
    }

    private void FillGdsOnGpu(ulong address, ulong size, uint value)
    {
        FlushComputeBatch("gds-fill");
        _ = EnsureCommandBuffer();
        var pattern = _guestMemory.Allocate(size, 256, out var bytes);
        System.Runtime.InteropServices.MemoryMarshal.Cast<byte, uint>(bytes).Fill(value);
        CopyGdsOnGpu(pattern.Buffer, pattern.Offset, _guestMemory.GdsBuffer, address, size);
        _gdsGpuFills++;
    }

    private bool TryCopyGdsOnGpu(ulong destination, ulong source, ulong size, bool sourceIsGds)
    {
        if (!GdsOnGpu()) return false;
        if (((destination | source | size) & 3) != 0) { _gdsHostFallbacks++; _gdsAlignmentFallbacks++; return false; }
        FlushComputeBatch("gds-copy");
        var address = sourceIsGds ? destination : source;
        // Resolve an actual guest window; staging here would read a GPU-produced
        // source on the CPU before its writer had completed.
        if (!_backing.IsBackedRange(address, size)) { _gdsHostFallbacks++; _gdsBackingFallbacks++; return false; }
        if (!_guestMemory.TryResolveInPlace(address, size, out var range) || !range.IsValid || range.Size < size) { _gdsHostFallbacks++; _gdsBackingFallbacks++; return false; }
        WaitForHostPublication(address, size);
        OrderAfterDeferredCopies(address, size);
        if (!sourceIsGds) FlushGpuWrites(address, size, MetalWritebackCause.GuestCopy);
        CopyGdsOnGpu(sourceIsGds ? _guestMemory.GdsBuffer : range.Buffer,
            sourceIsGds ? source : range.Offset,
            sourceIsGds ? range.Buffer : _guestMemory.GdsBuffer,
            sourceIsGds ? range.Offset : destination, size);
        if (sourceIsGds)
        {
            _imageCache.InvalidateMemoryFromGpu(destination, size);
            NoteGuestMemoryWrite(destination, size, MetalGpuWriteOrigin.Copy);
        }
        _gdsGpuCopies++;
        return true;
    }

    private void CopyGdsOnGpu(nint source, ulong sourceOffset, nint destination, ulong destinationOffset, ulong size)
    {
        var context = ((IMetalImageCacheOwner)this).BeginTransfer(MetalTransferCause.GuestCopy);
        var encoder = MetalNative.Send(context.CommandBuffer, SelBlitEncoder);
        MetalMemoryTelemetry.BlitEncoders.Add();
        if (encoder == 0) throw Fatal("A Metal blit encoder could not be created for GDS.");
        try
        {
            MetalNative.SendCopyBuffer(encoder, SelCopyBuffer, source, (nuint)sourceOffset,
                destination, (nuint)destinationOffset, (nuint)size);
        }
        finally { MetalNative.SendVoid(encoder, SelEndEncoding); }
        _gdsSerial = context.SubmissionSerial;
    }
}
