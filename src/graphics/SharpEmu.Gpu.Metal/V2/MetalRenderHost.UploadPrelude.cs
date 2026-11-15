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

// The upload prelude. An image upload the CPU caused (stale causes CpuWrite, Invalidated or
// Created only), needed while a render pass is open, is encoded into a command buffer of its own
// that is committed at once, ahead of the open one, instead of ending the pass: on a tile-based
// GPU ending and resuming Demon's Souls' 7-attachment G-buffer pass stores and reloads 22-26 MB.
// SHARPEMU_METAL_V2_UPLOAD_PRELUDE=0 ends the pass for every upload, as before.
//
// Order. Metal runs a queue's command buffers in the order they are enqueued, and commit enqueues
// a buffer not enqueued yet; nothing in this backend calls enqueue, and the open buffer is
// committed only by FlushCommandBuffer. So the prelude's commit (EndUploadPrelude) puts it after
// every buffer committed before it and before the open buffer, whenever that one is committed and
// however long the stream holds it (MetalCommandStream HoldForOpenPass): nothing committed after
// the prelude can run ahead of it. Its encoders take the same encoder fence as every other
// encoder (EncoderOrdering = Fence; any other mode refuses, counted as ordering): Metal resolves a
// fence across command buffers of one queue in commit order, so the prelude's first encoder waits
// for the last encoder of the buffer committed before it, and the open buffer's first encoder -
// and through the chain the open render pass - waits for the prelude's last. The texture is
// hazard-tracked as well (private textures are created tracked), so a draw of the open pass that
// samples it after the upload waits for the prelude's write either way.
//
// What moves. The upload now runs before everything the open buffer recorded ahead of it, so it
// may move only when none of that touches the texture or writes the bytes it reads: the texture
// has no use in the open buffer (its image's last-use and GPU-write serials, the textures the open
// buffer's transfers named, a clear waiting for its pass), and no GPU write over the source is
// recorded in it (MetalGpuWriteRanges at the open serial: dispatch stores, image writebacks, fills
// and copies; staged stores landing at an encoder end; a held label copy or a held fill over the
// bytes). An alias-caused or GPU-caused upload never moves: its source is normally produced by a
// writeback earlier in the same buffer. A staged source that needs a GPU gather, a sparse texture,
// a depth/stencil image, an upload in pieces, and a whole upload of an image a device copy wrote
// since its last refresh are refused too. Each refusal is counted and the upload ends the pass as
// before. Work recorded after the upload (draws sampling the texture, later writes over the
// source, other committed buffers) runs after the prelude, as it ran after the upload before.
//
// Lifetime. The prelude borrows staging, ring copies and window pins under the open buffer's
// serial (its context's SubmissionSerial), so they return to their pools only when the open buffer
// completes. Its signal is encoded after all of its encoders, whose fence chain starts behind the
// prelude's last encoder, so that serial completes only after the prelude has. The image notes the
// open serial as a use, as the binding after the upload would, so its texture outlives the prelude
// and a second upload of it in the same open buffer stays in that buffer. A prelude buffer is kept
// until the open serial completes, then its status is read: one that failed is counted and reported.
//
// Residency. A residency set (other titles) is attached to the queue, so the prelude has it as the
// open buffer does; without one (Demon's Souls) the transfer binds its window, ring and staging
// buffers directly to its encoders, which makes them resident for the prelude buffer.
//
// The counters: every image upload that met an open render pass, by cause, and by each condition
// of the prelude it passes or fails, printed on the TBDR line as
// prelude=[eligible,hoisted,refused,failed,<reason>=n...] pass_end_upload=[cpu,invalidated,created,gpu,alias];
// eligible counts uploads that passed every condition with the switch off.
internal sealed partial class MetalRenderHost
{
    /// <summary>Whether an upload the CPU caused may run ahead of an open render pass (SHARPEMU_METAL_V2_UPLOAD_PRELUDE).</summary>
    internal bool UploadPreludeEnabled { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_UPLOAD_PRELUDE") != "0";

    bool IMetalImageCacheOwner.UploadPreludeEnabled => UploadPreludeEnabled;

    private long _preludeHoisted;
    private long _preludeFailed;

    // Committed prelude buffers, with the serial whose completion implies theirs.
    private readonly List<(nint Buffer, ulong Serial)> _preludeBuffers = [];
    private UploadPreludeContext? _preludeContext;

    IMetalSubmissionContext? IMetalImageCacheOwner.BeginUploadPrelude()
    {
        if (!UploadPreludeEnabled || _commandBuffer == 0 || _renderEncoder == 0 ||
            EncoderOrdering != EncoderOrderingMode.Fence)
        {
            return null;
        }

        var buffer = NewCommandBuffer();
        if (buffer == 0)
        {
            throw Fatal("A Metal command buffer could not be created for an upload prelude.");
        }

        // Kept until its status is read; the queue hands it back autoreleased.
        MetalNative.SendVoid(buffer, SelRetain);
        MetalMemoryTelemetry.CommandBuffers.Add();
        // An end-of-pipe completion of the open serial waits for the GPU: the prelude is its work too.
        NoteRecordedWork();
        var context = _preludeContext ??= new UploadPreludeContext(this);
        context.CommandBuffer = buffer;
        return context;
    }

    void IMetalImageCacheOwner.EndUploadPrelude(IMetalSubmissionContext prelude)
    {
        var context = (UploadPreludeContext)prelude;
        var buffer = context.CommandBuffer;
        context.CommandBuffer = 0;
        MetalNative.SendVoid(buffer, SelCommit);
        MetalMemoryTelemetry.Commits.Add();
        _preludeBuffers.Add((buffer, _submissionSerial));
        _preludeHoisted++;
    }

    // Reads the status of each prelude whose open serial has completed, and lets it go.
    private void RetireUploadPreludes(ulong completed, bool all = false)
    {
        if (_preludeBuffers.Count == 0)
        {
            return;
        }

        var kept = 0;
        for (var index = 0; index < _preludeBuffers.Count; index++)
        {
            var (buffer, serial) = _preludeBuffers[index];
            if (!all && serial > completed)
            {
                _preludeBuffers[kept++] = _preludeBuffers[index];
                continue;
            }

            if (MetalV2Native.SendValue(buffer, MetalNative.Selector("status")) == CommandBufferStatusError)
            {
                _preludeFailed++;
                var error = MetalNative.Send(buffer, MetalNative.Selector("error"));
                Console.Error.WriteLine(
                    $"[METAL-V2][WARN] An upload prelude failed, so a texture it uploaded does not hold the guest's " +
                    $"bytes: serial={serial} {MetalNative.DescribeError(error)}");
            }

            MetalNative.SendVoid(buffer, SelRelease);
        }

        _preludeBuffers.RemoveRange(kept, _preludeBuffers.Count - kept);
    }

    /// <summary>Prelude buffers whose status has not been read yet.</summary>
    internal int PendingUploadPreludes => _preludeBuffers.Count;

    // The submission context of a prelude buffer. It borrows under the open buffer's serial and
    // orders its encoders with the host's encoder fence; it shares no encoder of the open buffer.
    private sealed class UploadPreludeContext(MetalRenderHost host) : IMetalSubmissionContext
    {
        public nint CommandBuffer { get; set; }

        public nint Device => host._device;

        public ulong SubmissionSerial => host._submissionSerial;

        public ulong CompletedSerial => host.CompletedSerial;

        public void EncoderStarted(nint encoder, bool render) => host.WaitForPreviousEncoder(encoder, render);

        public void EncoderEnding(nint encoder, bool render) => host.UpdateEncoderFence(encoder, render);
    }

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
        builder.Append(_preludeEligible).Append(" hoisted=").Append(_preludeHoisted)
            .Append(" refused=").Append(_preludeRefused).Append(" failed=").Append(_preludeFailed);
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
        _preludeHoisted = 0;
        _preludeRefused = 0;
        _preludeFailed = 0;
        Array.Clear(_preludeBlocked);
        Array.Clear(_passEndUploads);
        return builder.ToString();
    }

    /// <summary>The prelude counters since the last call, for tests.</summary>
    internal (long Eligible, long Hoisted, long Refused, long Failed, Dictionary<string, long> Blocked) TakePreludeCounters()
    {
        var blocked = new Dictionary<string, long>();
        for (var index = 0; index < _preludeBlocked.Length; index++)
        {
            blocked[PreludeBlockNames[index]] = _preludeBlocked[index];
        }

        var counters = (_preludeEligible, _preludeHoisted, _preludeRefused, _preludeFailed, blocked);
        _ = TakePreludeText();
        return counters;
    }
}
