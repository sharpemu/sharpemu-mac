// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
using System.Runtime.InteropServices;
using SharpEmu.Libs.Gpu.Images;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// An alias that shares whole layers with a GPU-written image, laid out the same way (MetalImageTransfer.LayerSubset),
/// takes those layers by a slice copy between the two textures instead of a writeback of the one and an upload of the
/// other. Demon's Souls keeps a 64-layer and a 72-layer 160x90 array on one base: each use of either moved 16-18 MiB
/// out through guest memory and back, every frame. The copy leaves the target holding exactly the texels the round
/// trip would have; guest memory is not written, and does not need to be: the source stays GPU-modified (a read) or
/// the target becomes the owner (a write), so the lazy writeback rules publish the bytes when anything asks for them.
/// </summary>
internal sealed partial class MetalImageCache
{
    // SHARPEMU_METAL_V2_ALIAS_SUBSET_DEVICE (on; =0 sends such aliases through guest memory again; =verify, or
    // SHARPEMU_PERF_VERIFY=1, also retiles both sides of every copy into scratch and compares them once the command
    // buffer completed, logging [METAL-V2][ALIAS-VERIFY] and counting a mismatch).
    private static readonly string? AliasSubsetSetting =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_ALIAS_SUBSET_DEVICE");

    internal bool AliasSubsetDeviceEnabled { get; set; } = AliasSubsetSetting != "0";

    internal bool AliasSubsetVerify { get; set; } =
        AliasSubsetSetting == "verify" || Environment.GetEnvironmentVariable("SHARPEMU_PERF_VERIFY") == "1";

    private long _aliasSubsetCopies, _aliasSubsetBytes;
    private long _aliasSubsetVerified, _aliasSubsetMismatches, _aliasSubsetUnverified;

    private sealed record AliasSubsetCheck(ulong Serial, nint Copied, nint RoundTripped, ulong Size, string Pair);

    private readonly List<AliasSubsetCheck> _aliasSubsetChecks = [];

    private static readonly nint SelSubsetNewBuffer = MetalNative.Selector("newBufferWithLength:options:");
    private static readonly nint SelSubsetContents = MetalNative.Selector("contents");
    private static readonly nint SelSubsetRelease = MetalNative.Selector("release");

    /// <summary>Copies, bytes, verified copies, verify mismatches and copies the verify could not retile.</summary>
    internal (long Copies, long Bytes, long Verified, long Mismatches, long Unverified) AliasSubsetCounters =>
        (Interlocked.Read(ref _aliasSubsetCopies), Interlocked.Read(ref _aliasSubsetBytes),
         Interlocked.Read(ref _aliasSubsetVerified), Interlocked.Read(ref _aliasSubsetMismatches),
         Interlocked.Read(ref _aliasSubsetUnverified));

    // The other image is GPU-modified and not stale over the run the two share (the loop checked both). Caller holds
    // the lock. False, with nothing encoded and neither image changed, unless the pair is one MetalImageTransfer
    // matches layer for layer and the copy leaves the target current over the whole run.
    private bool TryPublishAliasLayerSubset(
        MetalCachedImage image, MetalCachedImage other, bool takesGpuOwnership, ulong otherStart, ulong otherEnd)
    {
        if (!AliasSubsetDeviceEnabled || image.Texture == 0 || other.Texture == 0 || image.Texture == other.Texture ||
            image.IsWindowLinearView || other.IsWindowLinearView || image.FirstLevel != 0 || other.FirstLevel != 0 ||
            SparseOf(image) is not null || SparseOf(other) is not null ||
            image.Shape.PixelFormat != other.Shape.PixelFormat || image.Shape.SampleCount != other.Shape.SampleCount ||
            image.Shape.Width != other.Shape.Width || image.Shape.Height != other.Shape.Height ||
            image.Shape.MipLevels != 1 || other.Shape.MipLevels != 1 ||
            !_transfer.TryMatchLayerSubset(
                in other.Description, in image.Description, out var sourceFirst, out var targetFirst, out var layers))
        {
            return false;
        }

        // The matched layers are the whole of what the two share: both sizes are whole numbers of layers.
        var stride = image.Description.Data.Size / image.Description.Resources.Layers;
        var sourceStart = sourceFirst * stride;
        var sourceEnd = sourceStart + layers * stride;
        var targetStart = targetFirst * stride;
        var targetEnd = targetStart + layers * stride;
        var otherInside = sourceStart == 0 && sourceEnd == other.Description.Data.Size;
        if (sourceStart != otherStart || sourceEnd != otherEnd ||
            (other.IsCpuDirty && other.StaleOverlaps(sourceStart, sourceEnd)) ||
            // Whether marking the run current leaves none of it stale: a refresh of what stays stale would put the
            // guest's older bytes back over the copy.
            !image.CanCutStale(targetStart, targetEnd) ||
            // A write leaves the other image stale over the run and GPU-modified for the rest; one stale interval
            // cannot hold the run apart from one it already has, and widening it would cover bytes only its texture
            // holds. Run sets hold it apart while the set has room.
            (takesGpuOwnership && !otherInside && other.IsCpuDirty &&
             (other.RunCapacity > 1
                 ? !other.CanAddStaleRun(sourceStart, sourceEnd)
                 : other.DirtyEnd < sourceStart || other.DirtyStart > sourceEnd)))
        {
            return false;
        }

        if (LaterOwnerOverlaps(image, other))
        {
            RefuseAliasTransition("subset-later-owner");
            return false;
        }

        if (!RestReloadsAround(image, targetStart, targetEnd))
        {
            RefuseAliasTransition("subset-rest-reload");
            return false;
        }

        if (takesGpuOwnership && !otherInside)
        {
            // The other image stays GPU-modified and stale over the run, and only TryRefreshDirtyRun reloads a stale run
            // of such an image (a whole refresh would replace bytes only its texture holds). Unless that reload is
            // certain to make exactly this run, its next use would keep its older texels there (alias_run_refresh_refused).
            var (staleStart, staleEnd) = other.MergedStaleRun(sourceStart, sourceEnd);
            if (!CanReloadRunExactly(other, staleStart, staleEnd))
            {
                RefuseAliasTransition("subset-stale-run-inexact");
                return false;
            }
        }

        EnsureSparseBacked(image);
        var context = _owner.BeginTransfer(MetalTransferCause.Writeback, other.Texture, image.Texture);
        if (!_transfer.TryEncodeLayerSubsetCopy(
                context, other.Texture, sourceFirst, image.Texture, targetFirst, layers, image.Shape.Width, image.Shape.Height))
        {
            return false;
        }

        if (context.WantsEncodedLabels)
        {
            context.ProfileEncoded(ProfileLabel("alias-subset", in image.Description));
        }

        if (AliasSubsetVerify)
        {
            EncodeAliasSubsetCheck(context, image, other, targetStart, sourceStart, sourceEnd - sourceStart);
        }

        // The texture changed whether or not it was stale there; then it is current over the run. It now holds the
        // other image's bytes there, which guest memory does not (the other still owes them): a run this image had
        // published is no longer its bytes in guest memory, and would let a later hand-over or consumer skip them.
        image.NoteContentChanged();
        image.RefreshComplete(targetStart, targetEnd);
        image.ForgetPublished(targetStart, targetEnd);
        if (takesGpuOwnership)
        {
            if (otherInside)
            {
                other.ClearGpuModified();
                other.MarkCpuDirty(MetalImageDirtyCause.AliasWrite);
            }
            else
            {
                // As a narrow hand-over leaves it, without the publication: the bytes went to this image's texture,
                // which owns them from this write on.
                other.MarkCpuDirty(MetalImageDirtyCause.AliasWrite, sourceStart, sourceEnd);
            }
        }

        _aliasSubsetCopies++;
        _aliasSubsetBytes += (long)(sourceEnd - sourceStart);
        _aliasMovedBlitBytes += (long)(sourceEnd - sourceStart);
        return true;
    }

    // Whether an alias this sync has not reached yet still owes bytes it shares with this image: on its own turn it
    // publishes them and marks this image stale over all it shares with it (MarkAliasRunDirty). Over the run, the
    // reload would replace the copy with guest bytes the source has not given guest memory; anywhere else, it can
    // tip what is left stale past the narrow refresh (RestReloadsAround). One that has nothing to give (stale over all
    // it shares) is skipped by the sync, and one already reached marked this image before the copy, which the copy's
    // cut then clears. Caller holds the lock, inside SynchronizeAliases.
    private bool LaterOwnerOverlaps(MetalCachedImage image, MetalCachedImage other)
    {
        var address = image.Description.Data.Address;
        var size = image.Description.Data.Size;
        foreach (var identifier in FindImagesInRange(address, size, pageOverlap: false))
        {
            if (_slots.TryGet(identifier) is not { } owner || owner == image || owner == other ||
                !owner.IsGpuModified || !owner.Overlaps(address, size) || _syncedScratch.Contains(owner))
            {
                continue;
            }

            var (shareStart, shareEnd) = owner.OverlapWithin(address, size);
            if (shareEnd > shareStart && owner.IsStaleOver(shareStart, shareEnd))
            {
                continue;
            }

            return true;
        }

        return false;
    }

    // Whether what stays stale on this image once [start, end) is marked current is reloaded without touching the
    // run: by TryRefreshDirtyRun, each run alone (the half rule met, as the image is now), over a covering run that
    // does not reach into the copied one, read in place. Otherwise the reload is a whole one, and it would put guest
    // bytes - which do not have the source's yet - over the copy.
    private bool RestReloadsAround(MetalCachedImage image, ulong start, ulong end)
    {
        if (!image.IsCpuDirty)
        {
            return true;
        }

        Span<(ulong Start, ulong End)> left = stackalloc (ulong, ulong)[MetalCachedImage.MaxRuns + 1];
        var count = MetalCachedImage.SubtractRun(image.StaleRuns, start, end, left);
        ulong stale = 0;
        foreach (var run in left[..count])
        {
            stale += run.End - run.Start;
        }

        if (count == 0)
        {
            return true;
        }

        ref readonly var description = ref image.Description;
        if (!SubRangeTransfersEnabled || description.HasStencil ||
            (stale > description.Data.Size / 2 && !NarrowRefreshPastHalf(image)))
        {
            return false;
        }

        foreach (var run in left[..count])
        {
            if (!TryCoveringRun(in description, run.Start, run.End, out var runStart, out var runEnd) ||
                (runStart < end && runEnd > start) ||
                !_guestMemory.TryResolveInPlace(description.Data.Address + runStart, runEnd - runStart, out _))
            {
                return false;
            }
        }

        return true;
    }

    // Verify mode: what the round trip would have published (the source's run, retiled) and what the copy put in the
    // target (the target's run, retiled the same way) go to two scratch buffers, compared once the command buffer
    // completed. Identical layouts retile identical texels to identical bytes. Caller holds the lock.
    private void EncodeAliasSubsetCheck(
        IMetalSubmissionContext context, MetalCachedImage image, MetalCachedImage other,
        ulong targetStart, ulong sourceStart, ulong size)
    {
        var copied = NewSubsetScratch(size);
        var roundTripped = NewSubsetScratch(size);
        if (copied == 0 || roundTripped == 0 ||
            !_transfer.EncodeTextureToGuest(
                context, in other.Description,
                new SubresourceRange(0, other.Description.Resources.Levels, 0, other.Description.TransferLayers),
                other.Texture, new MetalGuestRange(roundTripped, 0, size, 0), default, sourceStart, size) ||
            !_transfer.EncodeTextureToGuest(
                context, in image.Description,
                new SubresourceRange(0, image.Description.Resources.Levels, 0, image.Description.TransferLayers),
                image.Texture, new MetalGuestRange(copied, 0, size, 0), default, targetStart, size))
        {
            ReleaseSubsetScratch(copied);
            ReleaseSubsetScratch(roundTripped);
            _aliasSubsetUnverified++;
            return;
        }

        _aliasSubsetChecks.Add(new AliasSubsetCheck(
            context.SubmissionSerial, copied, roundTripped, size,
            $"target=<{AliasImageText(in image.Description)}> source=<{AliasImageText(in other.Description)}> " +
            $"target_run=0x{targetStart:X} source_run=0x{sourceStart:X}+0x{size:X}"));
    }

    private unsafe nint NewSubsetScratch(ulong size)
    {
        var buffer = MetalNative.SendNewBuffer(_owner.Device, SelSubsetNewBuffer, (nuint)size, 0);
        if (buffer != 0)
        {
            NativeMemory.Clear((void*)MetalNative.Send(buffer, SelSubsetContents), (nuint)size);
        }

        return buffer;
    }

    private static void ReleaseSubsetScratch(nint buffer)
    {
        if (buffer != 0)
        {
            MetalNative.SendVoid(buffer, SelSubsetRelease);
        }
    }

    /// <summary>Compares every verify-mode copy whose command buffer completed. Caller holds the lock.</summary>
    internal unsafe void DrainAliasSubsetChecksLocked()
    {
        if (_aliasSubsetChecks.Count == 0)
        {
            return;
        }

        var completed = _owner.CompletedSerial;
        for (var index = 0; index < _aliasSubsetChecks.Count;)
        {
            var check = _aliasSubsetChecks[index];
            if (check.Serial > completed)
            {
                index++;
                continue;
            }

            _aliasSubsetChecks.RemoveAt(index);
            var copied = new ReadOnlySpan<byte>((void*)MetalNative.Send(check.Copied, SelSubsetContents), (int)check.Size);
            var roundTripped = new ReadOnlySpan<byte>(
                (void*)MetalNative.Send(check.RoundTripped, SelSubsetContents), (int)check.Size);
            var first = copied.CommonPrefixLength(roundTripped);
            Interlocked.Increment(ref _aliasSubsetVerified);
            if (first != copied.Length)
            {
                if (Interlocked.Increment(ref _aliasSubsetMismatches) <= 16)
                {
                    Console.Error.WriteLine(
                        $"[METAL-V2][ALIAS-VERIFY] a layer-subset copy differs from the round trip at run byte " +
                        $"0x{first.ToString("X", CultureInfo.InvariantCulture)}: {check.Pair}");
                }
            }

            ReleaseSubsetScratch(check.Copied);
            ReleaseSubsetScratch(check.RoundTripped);
        }
    }

    /// <summary>Test hook: compares the completed verify-mode copies.</summary>
    internal void DrainAliasSubsetChecks()
    {
        using var held = _lock.Hold();
        DrainAliasSubsetChecksLocked();
    }
}
