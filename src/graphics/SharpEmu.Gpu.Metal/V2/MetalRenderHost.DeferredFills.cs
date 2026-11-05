// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Scheduling;

namespace SharpEmu.Gpu.Metal.V2;

// A guest fill recorded while a render pass is open lands when that pass ends.
//
// Demon's Souls fills a few small buffers (0x6000-0x14000 bytes) between the draws of its
// G-buffer and lighting passes. A fill is a blit, which no render encoder can record, so each
// one ended the pass and the next draw began another: all its attachments stored and loaded
// again for a few kilobytes (~0.45 ms of GPU a frame in the PASS-PROF split analysis). The
// draws recorded before the fill read the bytes as they were, which landing it after the pass
// keeps; the draws after it are the ones that must see it. So the fill waits in the open pass,
// and lands as soon as anything that may see its bytes comes: a draw that binds them (a buffer,
// a vertex or index buffer, an indirect argument record, a texture over them), anything that
// orders itself after held command-processor writes (device-address ranges, transfers, image
// uploads and reads, copies, dispatches), an image appearing over them, or the pass ending for
// any reason, the commit included.
//
// Only a fill nothing else holds bytes in deferred: none over an image the cache knows of, none
// over a held label copy or a staged store waiting for the encoder boundary, at most 1 MiB, and
// at most 64 waiting.
// SHARPEMU_METAL_V2_DEFER_PASS_FILLS=0 ends the pass for every fill, as before.
internal sealed partial class MetalRenderHost
{
    private const ulong MaxDeferredFillBytes = 1UL << 20;
    private const int MaxDeferredFills = 64;

    /// <summary>Whether a fill recorded in an open render pass lands at its end (SHARPEMU_METAL_V2_DEFER_PASS_FILLS).</summary>
    internal bool DeferPassFills { get; set; } = MetalV2Diagnostics.DeferPassFills;

    private readonly record struct DeferredFill(ulong Address, ulong Size, uint Value, nint Buffer, ulong Offset);

    private readonly List<DeferredFill> _deferredFills = [];
    private bool _encodingDeferredFills;

    private long _fillsDeferred;
    private long _deferredFillLandings;
    private long _deferredFillsLandedEarly;

    internal int DeferredFillCount => _deferredFills.Count;

    internal (long Deferred, long Landings, long Early) TakeDeferredFills() =>
        (Interlocked.Exchange(ref _fillsDeferred, 0), Interlocked.Exchange(ref _deferredFillLandings, 0),
         Interlocked.Exchange(ref _deferredFillsLandedEarly, 0));

    private string FormatDeferredFills()
    {
        var (deferred, landings, early) = TakeDeferredFills();
        return $"fills_deferred=[fills:{deferred},landings:{landings},early:{early}]";
    }

    // Holds the fill for the end of the open pass; false leaves it to be encoded now.
    private bool TryDeferFillToPassEnd(ulong address, ulong size, uint value, in MetalGuestRange range)
    {
        if (!DeferPassFills || !_renderingActive || _renderEncoder == 0 || size > MaxDeferredFillBytes ||
            _deferredFills.Count >= MaxDeferredFills || DeferredLabelCopiesTouch(address, size) ||
            (_stagedStorageCopies is { Count: > 0 } staged && staged.HasRecordedStoresOver(address, size)) ||
            _imageCache.MayOwn(address, size))
        {
            return false;
        }

        _deferredFills.Add(new DeferredFill(address, size, value, range.Buffer, range.Offset));
        _transferCauses[(int)MetalTransferCause.GuestFill]++;
        _fillsDeferred++;
        NoteRecordedWork();
        if (PassProfileEnabled)
        {
            PassProfileNoteWork(nameof(MetalTransferCause.GuestFill), "deferred");
        }

        return true;
    }

    private bool DeferredFillsTouch(ulong address, ulong size)
    {
        foreach (var fill in _deferredFills)
        {
            if (fill.Address < address + size && address < fill.Address + fill.Size)
            {
                return true;
            }
        }

        return false;
    }

    private bool DeferredFillsTouch(in MetalGuestRange range)
    {
        foreach (var fill in _deferredFills)
        {
            if ((fill.Buffer == range.Buffer && fill.Offset < range.Offset + range.Size && range.Offset < fill.Offset + fill.Size) ||
                (range.GpuAddress != 0 && fill.Address < range.GpuAddress + range.Size && range.GpuAddress < fill.Address + fill.Size))
            {
                return true;
            }
        }

        return false;
    }

    // A binding of a buffer from this offset on, its end unknown (a vertex or index stream).
    private bool DeferredFillsFrom(nint buffer, ulong offset)
    {
        foreach (var fill in _deferredFills)
        {
            if (fill.Buffer == buffer && offset < fill.Offset + fill.Size)
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Lands the waiting fills before work over these bytes: the open pass ends, and its end
    /// encodes them. ulong.MaxValue for a size lands them whatever the address.
    /// </summary>
    private void LandDeferredFillsOver(ulong address, ulong size)
    {
        if (_deferredFills.Count == 0 || (size != ulong.MaxValue && !DeferredFillsTouch(address, size)))
        {
            return;
        }

        _deferredFillsLandedEarly++;
        EndEncoders(PassEndCause.Other);
    }

    /// <summary>
    /// Whether the draw BeginRendering is for may see a waiting fill: it binds its bytes, or an
    /// image has appeared over them since it was held.
    /// </summary>
    private bool UpcomingDrawSeesDeferredFill()
    {
        foreach (var fill in _deferredFills)
        {
            if (_imageCache.MayOwn(fill.Address, fill.Size))
            {
                return true;
            }
        }

        for (var index = 0; index < _vertexBufferCount; index++)
        {
            if (DeferredFillsFrom(_vertexBuffers[index], _vertexOffsets[index]))
            {
                return true;
            }
        }

        if (_indexBuffer != 0 &&
            (_indexGuestSize != 0 ? DeferredFillsTouch(_indexGuestAddress, _indexGuestSize) : DeferredFillsFrom(_indexBuffer, _indexOffset)))
        {
            return true;
        }

        foreach (var stage in _joinStages)
        {
            if (stage is null)
            {
                continue;
            }

            for (var index = 0; index < stage.BufferCount; index++)
            {
                if (stage.Buffers[index].IsValid && DeferredFillsTouch(in stage.Buffers[index]))
                {
                    return true;
                }
            }

            for (var index = 0; index < stage.ImageCount && index < stage.Requests.Length; index++)
            {
                var data = stage.Requests[index].Description.Data;
                if (data.Size != 0 && DeferredFillsTouch(data.Address, data.Size))
                {
                    return true;
                }
            }
        }

        return false;
    }

    // The open pass has ended: the fills it held are encoded, in the order they were recorded.
    private void EncodeDeferredFills()
    {
        if (_deferredFills.Count == 0 || _encodingDeferredFills)
        {
            return;
        }

        _encodingDeferredFills = true;
        try
        {
            var encoder = MetalNative.Send(EnsureCommandBuffer(), SelBlitEncoder);
            MetalMemoryTelemetry.BlitEncoders.Add();
            if (encoder == 0)
            {
                throw GpuFatal.Fatal("A Metal blit encoder could not be created for the fills a pass held.");
            }

            WaitForPreviousEncoder(encoder, render: false);
            try
            {
                foreach (var fill in _deferredFills)
                {
                    EncodeBlitFill(encoder, fill.Buffer, fill.Offset, fill.Size, fill.Value);
                }
            }
            finally
            {
                UpdateEncoderFence(encoder, render: false);
                MetalNative.SendVoid(encoder, SelEndEncoding);
            }

            _deferredFillLandings++;
            if (PassProfileEnabled)
            {
                PassProfileNoteBetween($"deferred-fills n={_deferredFills.Count}");
            }

            _deferredFills.Clear();
        }
        finally
        {
            _encodingDeferredFills = false;
        }
    }
}
