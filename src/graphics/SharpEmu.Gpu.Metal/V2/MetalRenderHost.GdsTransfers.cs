// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Scheduling;

namespace SharpEmu.Gpu.Metal.V2;

// DMA_DATA fills and copies that name the global data share, as blits in stream order. The
// data share is a shared buffer the shaders address, so a blit is ordered with the work around
// it as the command processor orders the packet. Draining the GPU and doing them on the CPU
// instead left the GPU idle through every one: Ghost of Tsushima's menu issues about sixty a
// frame. SHARPEMU_METAL_V2_GDS_ON_GPU=0 restores the drain for comparison.
internal sealed partial class MetalRenderHost
{
    private static readonly bool GdsOnGpu = Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_GDS_ON_GPU") != "0";

    private long _gpuGdsTransfers;

    /// <summary>Data share fills and copies encoded on the GPU since the last call.</summary>
    internal long TakeGpuGdsTransfers() => Interlocked.Exchange(ref _gpuGdsTransfers, 0);

    // A fill of data share bytes [offset, offset + size), already range checked.
    private bool TryFillGdsOnGpu(ulong offset, ulong size, uint value)
    {
        if (!GdsOnGpu)
        {
            return false;
        }

        var gds = _guestMemory.GdsBuffer;
        if (CopiesShareCompute && (size & 3) == 0 && (offset & 3) == 0 && size / sizeof(uint) is > 0 and <= uint.MaxValue &&
            ComputeEncoderForTransfer() is var shared and not 0)
        {
            _transferCauses[(int)MetalTransferCause.GuestFill]++;
            EncodeFillWords(shared, gds, offset, size, value);
            Interlocked.Increment(ref _gpuGdsTransfers);
            return true;
        }

        var context = ((IMetalImageCacheOwner)this).BeginTransfer(MetalTransferCause.GuestFill);
        var encoder = MetalNative.Send(context.CommandBuffer, SelBlitEncoder);
        MetalMemoryTelemetry.BlitEncoders.Add();
        if (encoder == 0)
        {
            throw GpuFatal.Fatal("A Metal blit encoder could not be created for a GDS fill.");
        }

        WaitForPreviousEncoder(encoder, render: false);
        try
        {
            var bytes = (byte)value;
            if (value == bytes * 0x01010101u)
            {
                MetalNative.SendFillBuffer(
                    encoder, SelFillBuffer, gds, new NsRange { Location = (nuint)offset, Length = (nuint)size }, bytes);
            }
            else
            {
                var pattern = _guestMemory.Allocate(Math.Min(size, 64 * 1024), 256, out var span);
                System.Runtime.InteropServices.MemoryMarshal.Cast<byte, uint>(span).Fill(value);
                for (var written = 0UL; written < size;)
                {
                    var chunk = Math.Min(pattern.Size, size - written);
                    MetalNative.SendCopyBuffer(
                        encoder, SelCopyBuffer, pattern.Buffer, (nuint)pattern.Offset, gds, (nuint)(offset + written), (nuint)chunk);
                    written += chunk;
                }
            }
        }
        finally
        {
            UpdateEncoderFence(encoder, render: false);
            MetalNative.SendVoid(encoder, SelEndEncoding);
        }

        if (context.WantsEncodedLabels)
        {
            context.ProfileEncoded($"gds fill bytes=0x{size:X}");
        }

        Interlocked.Increment(ref _gpuGdsTransfers);
        return true;
    }

    // A copy between the data share and guest memory, its data share side already range
    // checked; false when the guest side has no window, which the drain then copies.
    private bool TryCopyGdsOnGpu(ulong destination, ulong source, ulong size, bool destinationIsGds)
    {
        if (!GdsOnGpu)
        {
            return false;
        }

        var gds = _guestMemory.GdsBuffer;
        nint sourceBuffer, destinationBuffer;
        ulong sourceOffset, destinationOffset;
        if (destinationIsGds)
        {
            var origin = _guestMemory.Resolve(source, size);
            if (!origin.IsValid || origin.Size < size)
            {
                return false;
            }

            FlushGpuWrites(source, size, MetalWritebackCause.GuestCopy);
            OrderAfterDeferredCopies(source, size);
            (sourceBuffer, sourceOffset, destinationBuffer, destinationOffset) = (origin.Buffer, origin.Offset, gds, destination);
        }
        else
        {
            var target = _guestMemory.ResolveWritable(destination, size);
            if (!target.IsValid)
            {
                return false;
            }

            OrderAfterDeferredCopies(destination, size);
            (sourceBuffer, sourceOffset, destinationBuffer, destinationOffset) = (gds, source, target.Buffer, target.Offset);
        }

        var guestAddress = destinationIsGds ? source : destination;
        if (CopiesShareCompute && WordCopyFits(destinationBuffer, destinationOffset, sourceBuffer, sourceOffset, size) &&
            !DeferredLabelCopiesTouch(guestAddress, size) &&
            !(_stagedStorageCopies is { Count: > 0 } staged && staged.HasRecordedStoresOver(guestAddress, size)) &&
            ComputeEncoderForTransfer() is var shared and not 0)
        {
            _transferCauses[(int)MetalTransferCause.GuestCopy]++;
            EncodeCopyWords(shared, sourceBuffer, sourceOffset, destinationBuffer, destinationOffset, size);
            if (!destinationIsGds)
            {
                _imageCache.InvalidateMemoryFromGpu(destination, size);
                NoteGuestMemoryWrite(destination, size, MetalGpuWriteOrigin.Copy);
            }

            Interlocked.Increment(ref _gpuGdsTransfers);
            return true;
        }

        var context = ((IMetalImageCacheOwner)this).BeginTransfer(MetalTransferCause.GuestCopy);
        var encoder = MetalNative.Send(context.CommandBuffer, SelBlitEncoder);
        MetalMemoryTelemetry.BlitEncoders.Add();
        if (encoder == 0)
        {
            throw GpuFatal.Fatal("A Metal blit encoder could not be created for a GDS copy.");
        }

        WaitForPreviousEncoder(encoder, render: false);
        try
        {
            MetalNative.SendCopyBuffer(
                encoder, SelCopyBuffer, sourceBuffer, (nuint)sourceOffset, destinationBuffer, (nuint)destinationOffset, (nuint)size);
        }
        finally
        {
            UpdateEncoderFence(encoder, render: false);
            MetalNative.SendVoid(encoder, SelEndEncoding);
        }

        if (context.WantsEncodedLabels)
        {
            context.ProfileEncoded($"gds copy bytes=0x{size:X}");
        }

        if (!destinationIsGds)
        {
            _imageCache.InvalidateMemoryFromGpu(destination, size);
            NoteGuestMemoryWrite(destination, size, MetalGpuWriteOrigin.Copy);
        }

        Interlocked.Increment(ref _gpuGdsTransfers);
        return true;
    }
}
