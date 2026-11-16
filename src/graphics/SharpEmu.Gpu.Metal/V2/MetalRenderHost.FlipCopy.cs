// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.VideoOut;

namespace SharpEmu.Gpu.Metal.V2;

// SHARPEMU_FLIP_AHEAD: a GPU flip's frame is copied, in the stream at the flip, into a texture
// only the presenter reads (MetalFlipCopyPool), and video-out is told once the flip is recorded
// and the buffer carrying it committed, so it may publish the flip to the guest before the GPU
// finishes it (VideoOutExports.FlipAhead.cs).
internal sealed partial class MetalRenderHost
{
    private static readonly nint SelMipmapLevelCount = MetalNative.Selector("mipmapLevelCount");
    private static readonly nint SelArrayLength = MetalNative.Selector("arrayLength");
    private static readonly nint SelAllocatedSize = MetalNative.Selector("allocatedSize");

    // The flips whose frame was copied while the frame before them was a copy too: the only
    // ones whose recorded notice is sent. Worker thread; locked for the command thread's sake.
    private readonly List<ulong> _flipAheadEligible = new();

    // Whether the last frame handed to the presenter was a copy. A frame handed over live (a CPU
    // flip, a refused copy) is the display image itself, which stays on screen until the next
    // frame is promoted: the guest must not hear of that next flip before it is shown.
    private bool _previousFlipFrameCopied = true;

    // Copies the flip's presented view into a pool texture at this point of the stream. False
    // (refused, logged once and counted) leaves the frame to be presented live, as today.
    internal bool TryCopyFlipFrame(nint source, out nint presented)
    {
        presented = source;
        var type = (ulong)MetalV2Native.SendValue(source, SelTextureType);
        if (type != Mtl.TextureType2D)
        {
            return RefuseFlipCopy(MetalFlipCopyPool.Refusal.TextureType, $"type={type}");
        }

        var samples = (ulong)MetalV2Native.SendValue(source, SelSampleCount);
        if (samples > 1)
        {
            return RefuseFlipCopy(MetalFlipCopyPool.Refusal.Multisampled, $"samples={samples}");
        }

        var levels = (ulong)MetalV2Native.SendValue(source, SelMipmapLevelCount);
        var layers = (ulong)MetalV2Native.SendValue(source, SelArrayLength);
        if (levels != 1 || layers != 1)
        {
            return RefuseFlipCopy(MetalFlipCopyPool.Refusal.Layout, $"levels={levels} layers={layers}");
        }

        var pixelFormat = (ulong)MetalV2Native.SendValue(source, SelPixelFormat);
        var width = (uint)MetalV2Native.SendValue(source, SelWidth);
        var height = (uint)MetalV2Native.SendValue(source, SelHeight);
        var swizzle = MetalFlipCopyPool.SwizzleOf(source);
        if (!MetalFlipCopyPool.TryAcquire(_device, pixelFormat, width, height, swizzle,
                out var destination, out var copy, out var refusal))
        {
            return RefuseFlipCopy(refusal, $"format={pixelFormat} {width}x{height} capacity={MetalFlipCopyPool.Capacity}");
        }

        EndEncoders(PassEndCause.Present);
        var encoder = MetalNative.Send(EnsureCommandBuffer(), SelBlitEncoder);
        MetalMemoryTelemetry.BlitEncoders.Add();
        if (encoder == 0)
        {
            return RefuseFlipCopy(MetalFlipCopyPool.Refusal.Encoder, $"format={pixelFormat} {width}x{height}");
        }

        // Fenced like every other encoder of the stream: after the passes that drew the frame,
        // before anything recorded after the flip.
        WaitForPreviousEncoder(encoder, render: false);
        try
        {
            MetalNative.SendCopyTextureSlice(
                encoder,
                SelCopyTextureSlice,
                source,
                0,
                0,
                default,
                new MtlSize { Width = width, Height = height, Depth = 1 },
                destination,
                0,
                0,
                default);
        }
        finally
        {
            UpdateEncoderFence(encoder, render: false);
            MetalNative.SendVoid(encoder, SelEndEncoding);
        }

        NoteRecordedWork();
        MetalFlipCopyPool.NoteCopy((ulong)MetalV2Native.SendValue(destination, SelAllocatedSize));
        presented = copy;
        return true;
    }

    private static bool RefuseFlipCopy(MetalFlipCopyPool.Refusal refusal, string detail)
    {
        MetalFlipCopyPool.NoteRefusal(refusal);
        MetalV2Diagnostics.WarnOnce(
            $"flip-copy-{MetalFlipCopyPool.RefusalName(refusal)}",
            $"Flip ahead: a flip's frame could not be copied for the presenter ({MetalFlipCopyPool.RefusalName(refusal)}: " +
            $"{detail}); it is presented live and published to the guest at presentation, as without the switch.");
        return false;
    }

    // After the frame went to the presenter: copied flips whose previous frame was a copy too
    // may be published ahead once their end-of-pipe completion is queued.
    private void NoteFlipFrameCaptured(ulong requestId, bool copied)
    {
        var eligible = copied && _previousFlipFrameCopied;
        _previousFlipFrameCopied = copied;
        if (!eligible)
        {
            return;
        }

        lock (_flipAheadEligible)
        {
            // A flip whose end-of-pipe write never reaches RecordEndOfPipe (a label published by
            // the interpreter) keeps today's timing; its entry is dropped here.
            if (_flipAheadEligible.Count >= 8)
            {
                _flipAheadEligible.RemoveAt(0);
            }

            _flipAheadEligible.Add(requestId);
        }
    }

    // RecordEndOfPipe queued the flip's completion: the buffer carrying it is committed (or the
    // completion already ran inline, in which case video-out ignores the late notice).
    private void NoteFlipRecordedIfEligible(ulong requestId)
    {
        if (!MetalV2Diagnostics.FlipAhead || requestId == 0)
        {
            return;
        }

        bool eligible;
        lock (_flipAheadEligible)
        {
            eligible = _flipAheadEligible.Remove(requestId);
        }

        if (eligible)
        {
            VideoOutBackendServices.NoteFlipRecorded(requestId);
        }
    }
}
