// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Numerics;
using System.Text;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// Why an image upload needed while a render pass is open could not be encoded ahead of the open
/// command buffer (the upload prelude). An upload is counted under every condition it fails.
/// </summary>
[Flags]
internal enum MetalPreludeBlock : uint
{
    None = 0,
    // The image went stale through an alias hand-over: its source bytes are normally written back
    // by a writeback earlier in the same command buffer.
    CauseAlias = 1 << 0,
    // Guest memory under the image was written by the GPU (a store, fill or copy the cache saw).
    CauseGpu = 1 << 1,
    // The texture is bound, written or named by a transfer in the open command buffer: work
    // recorded before the upload would see the new texels if the upload ran first.
    UsedInBuffer = 1 << 2,
    // A clear of the texture waits for its pass in the open command buffer.
    PendingClear = 1 << 3,
    // A GPU store over the source bytes is recorded in the open command buffer (a dispatch, a
    // writeback, a fill, a GuestCopy, or a staged store that lands at the encoder's end).
    PendingGpuWrite = 1 << 4,
    // The source is a ring copy the GPU must gather behind pending writes, in the open buffer.
    StagedGather = 1 << 5,
    // A command-processor copy held for the commit lands in the source bytes.
    HeldCopy = 1 << 6,
    // A guest fill held by the open pass lands in the source bytes.
    HeldFill = 1 << 7,
    // A whole upload of an image a device copy wrote since its last refresh.
    DeviceCopy = 1 << 8,
    // A sparse texture maps its tiles in the open command buffer first.
    Sparse = 1 << 9,
    // Depth and stencil uploads are left in place (they encode render passes of their own).
    DepthStencil = 1 << 10,
    // The upload is cut into pieces (larger than one transfer piece).
    Pieces = 1 << 11,
    // The open buffer's transient staging is past its budget: the normal path commits it.
    Budget = 1 << 12,
    // The encoders are not chained by the encoder fence, which is what orders the prelude.
    Ordering = 1 << 13,
    // The source range resolves to no window and no ring copy.
    Unmapped = 1 << 14,
}

// The upload prelude counters: every image upload that met an open render pass, by cause, and by
// each condition of the prelude it passes or fails. Printed on the TBDR line as
// prelude=[eligible,refused,<reason>=n...] pass_end_upload=[cpu,invalidated,created,gpu,alias].
internal sealed partial class MetalRenderHost
{
    private static readonly string[] PreludeBlockNames =
    [
        "cause_alias", "cause_gpu", "used", "clear", "gpu_write", "gather", "held_copy", "held_fill",
        "device_copy", "sparse", "depth", "pieces", "budget", "ordering", "unmapped",
    ];

    private readonly long[] _preludeBlocked = new long[PreludeBlockNames.Length];
    private readonly long[] _passEndUploads = new long[Enum.GetValues<MetalImageDirtyCause>().Length];
    private long _preludeEligible;
    private long _preludeRefused;

    // The textures a transfer of the open command buffer named, read or written. A bound texture is
    // known by its image's last-use serial; a transfer names the texture itself (a device alias
    // copy, a usage promotion, a sparse remap, a clear).
    private readonly HashSet<nint> _transferTexturesInBuffer = [];

    private void NoteTransferTextures(nint texture, nint otherTexture)
    {
        if (texture != 0)
        {
            _transferTexturesInBuffer.Add(texture);
        }

        if (otherTexture != 0)
        {
            _transferTexturesInBuffer.Add(otherTexture);
        }
    }

    // A transfer begun now ends an open render pass.
    bool IMetalImageCacheOwner.UploadWouldEndRenderPass => _commandBuffer != 0 && _renderEncoder != 0;

    // What the host knows of the conditions for encoding an upload of this texture from these guest
    // bytes ahead of the open command buffer.
    MetalPreludeBlock IMetalImageCacheOwner.PreludeBlocks(nint texture, ulong address, ulong size)
    {
        var blocks = MetalPreludeBlock.None;
        if (_transferTexturesInBuffer.Contains(texture))
        {
            blocks |= MetalPreludeBlock.UsedInBuffer;
        }

        if (PendingClearTouches(texture))
        {
            blocks |= MetalPreludeBlock.PendingClear;
        }

        if ((_gpuWrites.MayCover(address, size) && _gpuWrites.PendingSerial(address, size, CompletedSerial) >= _submissionSerial) ||
            (_stagedStorageCopies is { Count: > 0 } staged && staged.HasRecordedStoresOver(address, size)))
        {
            blocks |= MetalPreludeBlock.PendingGpuWrite;
        }

        if (DeferredLabelCopiesTouch(address, size))
        {
            blocks |= MetalPreludeBlock.HeldCopy;
        }

        if (_deferredFills.Count != 0 && DeferredFillsTouch(address, size))
        {
            blocks |= MetalPreludeBlock.HeldFill;
        }

        if (RecordingTransientBytes >= TransientCommitBytes)
        {
            blocks |= MetalPreludeBlock.Budget;
        }

        if (EncoderOrdering != EncoderOrderingMode.Fence)
        {
            blocks |= MetalPreludeBlock.Ordering;
        }

        return blocks;
    }

    void IMetalImageCacheOwner.NotePassEndingUpload(MetalImageDirtyCause cause, MetalPreludeBlock blocks)
    {
        _passEndUploads[(int)cause]++;
        if (blocks == MetalPreludeBlock.None)
        {
            _preludeEligible++;
            return;
        }

        _preludeRefused++;
        for (var bits = (uint)blocks; bits != 0; bits &= bits - 1)
        {
            _preludeBlocked[BitOperations.TrailingZeroCount(bits)]++;
        }
    }

    /// <summary>The prelude counters as the TBDR line prints them, reset.</summary>
    internal string TakePreludeText()
    {
        var builder = new StringBuilder("prelude=[eligible=");
        builder.Append(_preludeEligible).Append(" refused=").Append(_preludeRefused);
        for (var index = 0; index < _preludeBlocked.Length; index++)
        {
            if (_preludeBlocked[index] != 0)
            {
                builder.Append(' ').Append(PreludeBlockNames[index]).Append('=').Append(_preludeBlocked[index]);
            }
        }

        builder.Append("] pass_end_upload=[cpu=").Append(_passEndUploads[(int)MetalImageDirtyCause.CpuWrite])
            .Append(" invalidated=").Append(_passEndUploads[(int)MetalImageDirtyCause.Invalidated])
            .Append(" created=").Append(_passEndUploads[(int)MetalImageDirtyCause.Created])
            .Append(" gpu=").Append(_passEndUploads[(int)MetalImageDirtyCause.GpuWrite])
            .Append(" alias=").Append(_passEndUploads[(int)MetalImageDirtyCause.AliasWrite]).Append(']');
        _preludeEligible = 0;
        _preludeRefused = 0;
        Array.Clear(_preludeBlocked);
        Array.Clear(_passEndUploads);
        return builder.ToString();
    }

    /// <summary>The prelude counters since the last call, for tests: (eligible, refused, refusals by reason name).</summary>
    internal (long Eligible, long Refused, Dictionary<string, long> Blocked) TakePreludeCounters()
    {
        var blocked = new Dictionary<string, long>();
        for (var index = 0; index < _preludeBlocked.Length; index++)
        {
            blocked[PreludeBlockNames[index]] = _preludeBlocked[index];
        }

        var counters = (_preludeEligible, _preludeRefused, blocked);
        _ = TakePreludeText();
        return counters;
    }
}
