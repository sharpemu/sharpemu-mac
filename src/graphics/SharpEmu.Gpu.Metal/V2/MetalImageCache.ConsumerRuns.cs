// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Metal.V2;

// A buffer binding, a storage descriptor or a host read that names a few KiB of a large
// GPU-written image publishes the run it names, not the image. A whole writeback retiles
// all of a 16-33 MiB render target for a 4 KiB V# inside it, on every draw that binds it.
//
// The image stays GPU-modified: its texture still owns every byte outside the published
// run, and the run is recorded (MetalCachedImage.MarkPublished) so the next consumer of the
// same bytes publishes nothing. A later reader of the rest, a guest store, a release or a
// drain still writes the whole image back.
//
// A consumer that also writes the bytes is followed by InvalidateMemoryFromGpu over them.
// That used to give the whole image to guest memory, which is only right once all of it is
// there. Here a GPU write into an image that stays GPU-modified marks just the written bytes
// stale, when the narrow reload that follows (TryRefreshDirtyRun) is certain to read only
// bytes guest memory holds for the image: ones published, or ones already stale. A written
// consumer publishes its run only when that will hold, and otherwise the whole image, as
// before.
//
// SHARPEMU_METAL_V2_SUBRANGE_CONSUMERS=0 writes every consumer's images back whole again.
internal sealed partial class MetalImageCache
{
    private static readonly bool SubRangeConsumersDefault =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_SUBRANGE_CONSUMERS") != "0";

    /// <summary>Whether a consumer of part of a GPU-written image publishes only that part.</summary>
    internal bool SubRangeConsumers { get; set; } = SubRangeConsumersDefault;

    private long _consumerRuns, _consumerRunBytes, _consumerRunsSkipped, _gpuWritesKeptOwned;
    private long _writebackBytes;

    /// <summary>Image bytes every writeback, whole or run, has retiled into guest memory.</summary>
    internal long WritebackBytes => Interlocked.Read(ref _writebackBytes);

    internal (long Runs, long RunBytes, long AlreadyPublished, long GpuWritesKept) TakeConsumerRuns() =>
        (Interlocked.Exchange(ref _consumerRuns, 0), Interlocked.Exchange(ref _consumerRunBytes, 0),
         Interlocked.Exchange(ref _consumerRunsSkipped, 0), Interlocked.Exchange(ref _gpuWritesKeptOwned, 0));

    // Caller holds the lock. True when guest memory now holds what the consumer reads of
    // the image, without the image being written back whole.
    private bool TryPublishConsumerRunLocked(
        MetalCachedImage image, ulong address, ulong size, MetalWritebackCause cause, bool written)
    {
        if (!SubRangeConsumers || !SubRangeTransfersEnabled ||
            cause is not (MetalWritebackCause.HostRead or MetalWritebackCause.BufferBinding
                or MetalWritebackCause.StorageDescriptor) ||
            !IsRunShaped(image))
        {
            return false;
        }

        var (start, end) = image.OverlapWithin(address, size);
        if (end <= start || (start == 0 && end >= image.Description.Data.Size))
        {
            return false;
        }

        if (image.IsCpuDirty)
        {
            // Stale over these bytes already: guest memory, or the alias this same loop
            // publishes, holds the newer copy, so the image has nothing of them to give.
            // Anything else keeps the whole writeback, which reloads the stale run first.
            if (start < image.DirtyStart || end > image.DirtyEnd ||
                (written && !CanKeepGpuOwnership(image, start, end)))
            {
                return false;
            }

            _consumerRunsSkipped++;
            return true;
        }

        if (written)
        {
            // Published here, stale once the write lands (TryKeepGpuOwnershipOverGpuWriteLocked):
            // the write keeps the texture the owner of the rest only if the reload of the run
            // reads nothing guest memory lacks, so that is settled before anything is skipped.
            if (!TryNarrowReloadRun(image, start, end, out var runStart, out var runEnd))
            {
                return false;
            }

            if (!image.IsPublished(runStart, runEnd))
            {
                if (!TryWriteBackRun(image, runStart, runEnd, cause, out _))
                {
                    return false;
                }

                NoteConsumerRun(runEnd - runStart);
            }
            else
            {
                _consumerRunsSkipped++;
            }

            // A run recorded but not kept as published (a longer one is) leaves the write
            // with no proof; the whole writeback then lands over it, in order.
            return CanKeepGpuOwnership(image, start, end);
        }

        if (image.IsPublished(start, end))
        {
            _consumerRunsSkipped++;
            return true;
        }

        if (!TryWriteBackRun(image, start, end, cause, out var refusal))
        {
            if (cause == MetalWritebackCause.HostRead)
            {
                RefuseAliasTransition($"cpu-read-{refusal}");
            }

            return false;
        }

        if (cause == MetalWritebackCause.HostRead)
        {
            _aliasCpuReadRuns++;
            _aliasCpuReadBytes += (long)(end - start);
        }

        NoteConsumerRun(end - start);
        return true;
    }

    private void NoteConsumerRun(ulong bytes)
    {
        _consumerRuns++;
        _consumerRunBytes += (long)bytes;
    }

    // A run of the image may be published, and reloaded, on its own: one surface plane,
    // the whole chain in an owned texture, fully resident and exact.
    private static bool IsRunShaped(MetalCachedImage image) =>
        image.Texture != 0 && !image.Description.HasStencil && image.FirstLevel == 0 &&
        image.Sparse is null && !image.IsWindowLinearView && !image.Shape.Lossy;

    // Caller holds the lock. A GPU write into an image the GPU also wrote: instead of
    // giving all of it to guest memory, mark the written bytes stale and keep the texture
    // the owner of the rest, when the narrow reload of the stale bytes will read nothing
    // but bytes guest memory holds for the image.
    private bool TryKeepGpuOwnershipOverGpuWriteLocked(MetalCachedImage image, ulong address, ulong size)
    {
        if (!SubRangeConsumers || !SubRangeTransfersEnabled || !image.IsGpuModified)
        {
            return false;
        }

        var (start, end) = image.OverlapWithin(address, size);
        if (!CanKeepGpuOwnership(image, start, end))
        {
            return false;
        }

        image.MarkCpuDirty(MetalImageDirtyCause.GpuWrite, start, end);
        _gpuWritesKeptOwned++;
        return true;
    }

    // Whether a GPU write of [start, end) may leave the image GPU-modified and stale over
    // just those bytes. Changes nothing, so a written consumer asks it before the write
    // and gets the answer the write's invalidation gets after.
    private bool CanKeepGpuOwnership(MetalCachedImage image, ulong start, ulong end)
    {
        if (end <= start || !image.IsGpuModified || !IsRunShaped(image))
        {
            return false;
        }

        // One stale interval: a write apart from the one already stale would merge the
        // bytes between them, which only the texture holds.
        var (staleStart, staleEnd) = (start, end);
        if (image.IsCpuDirty)
        {
            if (start > image.DirtyEnd || end < image.DirtyStart)
            {
                return false;
            }

            staleStart = Math.Min(staleStart, image.DirtyStart);
            staleEnd = Math.Max(staleEnd, image.DirtyEnd);
        }

        return TryNarrowReloadRun(image, staleStart, staleEnd, out var runStart, out var runEnd) &&
            GuestHoldsImageBytes(image, runStart, runEnd, staleStart, staleEnd);
    }

    // The run TryRefreshDirtyRun reloads for a stale interval, when it reloads just that run:
    // the same conditions, in the same order, so the answer here is the answer there.
    private bool TryNarrowReloadRun(MetalCachedImage image, ulong staleStart, ulong staleEnd,
        out ulong runStart, out ulong runEnd)
    {
        runStart = runEnd = 0;
        ref readonly var description = ref image.Description;
        if (staleEnd <= staleStart || staleEnd - staleStart > description.Data.Size / 2 ||
            !TryCoveringRun(in description, staleStart, staleEnd, out runStart, out runEnd))
        {
            return false;
        }

        var source = _guestMemory.Resolve(description.Data.Address + runStart, runEnd - runStart);
        return source.IsValid && source.Size >= runEnd - runStart;
    }

    // Every byte of [runStart, runEnd) is one guest memory holds for the image: inside the
    // published run, or inside the stale interval, where guest memory is the newer copy.
    private static bool GuestHoldsImageBytes(
        MetalCachedImage image, ulong runStart, ulong runEnd, ulong staleStart, ulong staleEnd)
    {
        if (runStart >= staleStart && runEnd <= staleEnd)
        {
            return true;
        }

        var (publishedStart, publishedEnd) = (image.PublishedStart, image.PublishedEnd);
        if (publishedEnd <= publishedStart)
        {
            return false;
        }

        if (publishedStart <= staleEnd && staleStart <= publishedEnd)
        {
            // Touching or overlapping: together one interval.
            publishedStart = Math.Min(publishedStart, staleStart);
            publishedEnd = Math.Max(publishedEnd, staleEnd);
        }

        return runStart >= publishedStart && runEnd <= publishedEnd;
    }
}
