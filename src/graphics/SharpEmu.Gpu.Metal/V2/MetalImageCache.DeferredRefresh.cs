// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Images;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// A render target the guest wrote with the CPU is uploaded before a pass renders into
/// it, but a pass that clears the whole target, or whose first draw overwrites all of it
/// (MetalPassLoads), never looks at what the upload put there. On Demon's Souls' title
/// screen that is ~4 ms of GPU time a frame. Which of the two a pass is only becomes
/// known when it begins, so an attachment acquire leaves the upload, and the GPU-write
/// bookkeeping that follows it, to BeginRendering (ResolveDeferredRefresh), which either
/// drops the upload or makes it.
///
/// While it waits the image is in an ordinary state - CPU-dirty and not GPU-modified -
/// and every other path that uses its texture uploads it first, as for any dirty image.
/// What the wait changes is only that the image is not yet marked as written by the
/// pass, so anything that could look at it differently because of that resolves the
/// wait first: a texture acquire, a copy's refresh, a clear encoded by the host, another
/// attachment over the same bytes, the end of the draw's preparation, a binding reset
/// and a command-buffer flush.
/// </summary>
internal sealed partial class MetalImageCache
{
    // The view a pass will name the attachment by, the image, and the GPU-write marking
    // the acquire left for later: a colour target, or a depth target and whether it
    // writes only the stencil plane.
    private readonly record struct DeferredRefresh(nint View, ResourceSlotIdentifier Identifier,
        MetalCachedImage Image, bool Depth, bool StencilOnly, bool AliasesPending);

    private readonly List<DeferredRefresh> _deferredRefreshes = [];

    // Mirrors _deferredRefreshes.Count so the recording thread can ask without the lock.
    private int _deferredRefreshCount;

    private long _refreshesSkipped;
    private long _refreshSkippedBytes;

    /// <summary>Whether an attachment acquire left an upload for the next pass to decide.</summary>
    public bool HasDeferredRefreshes => Volatile.Read(ref _deferredRefreshCount) != 0;

    /// <summary>Uploads dropped because the pass overwrote the whole image, and their guest bytes.</summary>
    public (long Refreshes, long Bytes) TakeSkippedRefreshes() =>
        (Interlocked.Exchange(ref _refreshesSkipped, 0), Interlocked.Exchange(ref _refreshSkippedBytes, 0));

    // Whether an attachment acquire may leave this image's upload to the pass. Only an
    // image that is dirty and holds nothing the GPU wrote: a GPU-modified one refreshes a
    // run at most (RefreshFromGuest), and its texture is the truth everywhere else.
    private static bool CanDeferRefresh(MetalCachedImage image) =>
        MetalV2Diagnostics.SkipOverwrittenRefresh && image.IsCpuDirty && !image.IsGpuModified &&
        // An array is never skipped (CoversWholeImage), and a sparse one maps its layers in the
        // upload, which has to happen before its views are handed out.
        SparseOf(image) is null &&
        image.Texture != 0 && !ImageDescription.IsEmptyRange(image.Description.Data);

    // Caller holds the lock.
    private void DeferRefreshLocked(nint view, ResourceSlotIdentifier identifier, MetalCachedImage image,
        bool depth, bool stencilOnly, bool aliasesPending)
    {
        _deferredRefreshes.Add(new DeferredRefresh(view, identifier, image, depth, stencilOnly, aliasesPending));
        Volatile.Write(ref _deferredRefreshCount, _deferredRefreshes.Count);
    }

    // Takes the entry at index out of the list before it is resolved: the upload can end
    // up flushing the command buffer, and the flush drains whatever is left.
    private DeferredRefresh TakeDeferredLocked(int index)
    {
        var entry = _deferredRefreshes[index];
        _deferredRefreshes.RemoveAt(index);
        Volatile.Write(ref _deferredRefreshCount, _deferredRefreshes.Count);
        return entry;
    }

    /// <summary>
    /// Resolves, by uploading, every waiting attachment that is this image or shares guest
    /// bytes with it. An attachment acquire calls it first: synchronizing this image's
    /// aliases has to see the other one as the GPU-written target it is about to be.
    /// Caller holds the lock.
    /// </summary>
    private void DrainOverlappingDeferredRefreshesLocked(MetalCachedImage image)
    {
        if (_deferredRefreshes.Count == 0)
        {
            return;
        }

        var address = image.Description.Data.Address;
        var size = image.Description.Data.Size;
        for (var index = 0; index < _deferredRefreshes.Count;)
        {
            var waiting = _deferredRefreshes[index].Image;
            if (waiting == image || waiting.Overlaps(address, size) || image.Overlaps(
                    waiting.Description.Data.Address, waiting.Description.Data.Size))
            {
                CompleteDeferredLocked(TakeDeferredLocked(index), overwritten: false);
                index = 0;
                continue;
            }

            index++;
        }
    }

    /// <summary>Uploads every waiting attachment and marks it written, as its acquire would have.</summary>
    public void DrainDeferredRefreshes()
    {
        if (!HasDeferredRefreshes)
        {
            return;
        }

        // A flush or a host clear can run inside a cache call that already holds the lock.
        if (_lock.IsHeldByCurrentThread)
        {
            DrainDeferredRefreshesLocked();
            return;
        }

        using var held = _lock.Hold();
        DrainDeferredRefreshesLocked();
    }

    /// <summary>Uploads, and marks written, the waiting attachments whose texture is one of these.</summary>
    public void DrainDeferredRefreshesOf(ReadOnlySpan<nint> textures)
    {
        if (!HasDeferredRefreshes || textures.IsEmpty)
        {
            return;
        }

        if (_lock.IsHeldByCurrentThread)
        {
            DrainDeferredRefreshesOfLocked(textures);
            return;
        }

        using var held = _lock.Hold();
        DrainDeferredRefreshesOfLocked(textures);
    }

    private void DrainDeferredRefreshesOfLocked(ReadOnlySpan<nint> textures)
    {
        for (var index = 0; index < _deferredRefreshes.Count;)
        {
            if (textures.Contains(_deferredRefreshes[index].Image.Texture))
            {
                CompleteDeferredLocked(TakeDeferredLocked(index), overwritten: false);
                index = 0;
                continue;
            }

            index++;
        }
    }

    // Caller holds the lock.
    private void DrainDeferredRefreshesLocked()
    {
        if (_deferredRefreshes.Count == 0)
        {
            return;
        }

        while (_deferredRefreshes.Count != 0)
        {
            CompleteDeferredLocked(TakeDeferredLocked(0), overwritten: false);
        }

        Epochs.NoteBindingChanged();
    }

    /// <summary>
    /// Resolves the waiting attachment a pass about to begin names by view. Overwritten
    /// says the pass replaces every texel of the attachment (a clear load action or an
    /// elided load); the colour or depth planes it covers are the caller's to judge, the
    /// image's subresources are judged here. Called before the pass's encoder exists.
    /// </summary>
    public void ResolveDeferredRefresh(nint view, bool overwritten, bool stencilOverwritten = false)
    {
        if (!HasDeferredRefreshes || view == 0)
        {
            return;
        }

        if (_lock.IsHeldByCurrentThread)
        {
            ResolveDeferredRefreshLocked(view, overwritten, stencilOverwritten);
            return;
        }

        using var held = _lock.Hold();
        ResolveDeferredRefreshLocked(view, overwritten, stencilOverwritten);
    }

    private void ResolveDeferredRefreshLocked(nint view, bool overwritten, bool stencilOverwritten)
    {
        for (var index = 0; index < _deferredRefreshes.Count; index++)
        {
            if (_deferredRefreshes[index].View != view)
            {
                continue;
            }

            var entry = TakeDeferredLocked(index);
            CompleteDeferredLocked(
                entry, overwritten && CoversWholeImage(entry, stencilOverwritten));
            return;
        }
    }

    // Whether a pass that overwrites the attachment overwrites every byte the image owns:
    // one subresource, so the attachment is all of it, and for a depth target every plane
    // the image has.
    private static bool CoversWholeImage(in DeferredRefresh entry, bool stencilOverwritten)
    {
        var image = entry.Image;
        var shape = image.Shape;
        if (shape.MipLevels != 1 || shape.ArrayLength != 1 || image.Description.IsVolume ||
            shape.TextureType == Mtl.TextureType3D)
        {
            return false;
        }

        if (!entry.Depth)
        {
            return true;
        }

        var aspects = ViewFormatRules.DepthAspects(image.Description.PixelFormat);
        if ((aspects & GpuImageAspect.DepthBit) == 0)
        {
            // A stencil-only target is not worth the cases it would take.
            return false;
        }

        var hasStencil = (aspects & GpuImageAspect.StencilBit) != 0 || image.Description.HasStencil;
        return !hasStencil || stencilOverwritten;
    }

    // Makes or drops the upload, then marks the image written by the pass exactly as the
    // acquire used to right after its refresh. Caller holds the lock.
    private void CompleteDeferredLocked(in DeferredRefresh entry, bool overwritten)
    {
        var image = entry.Image;
        if (entry.AliasesPending && (!overwritten || !TryDiscardOverwrittenAliases(entry.Identifier)))
        {
            SynchronizeAliases(entry.Identifier, takesGpuOwnership: true);
        }
        if (overwritten && image.IsCpuDirty)
        {
            Interlocked.Increment(ref _refreshesSkipped);
            Interlocked.Add(
                ref _refreshSkippedBytes,
                (long)(image.Description.Data.Size + (image.Description.HasStencil ? image.Description.Stencil.Size : 0)));
            image.RefreshComplete();
        }
        else
        {
            RefreshFromGuest(image);
        }

        if (entry.Depth)
        {
            image.MarkGpuModified(depthPlaneChanged: !entry.StencilOnly);
            NoteLinearGpuWrite(image);
            image.GpuWriteSerial = _owner.CurrentSerial;
            image.NoteUse(image.GpuWriteSerial);
            return;
        }

        image.MarkGpuModified();
        NoteLinearGpuWrite(image);
        image.GpuWriteSerial = _owner.CurrentSerial;
        image.NoteUse(image.GpuWriteSerial);
    }

    // A released image's upload no longer has a texture to land in. Caller holds the lock.
    private void ForgetDeferredRefreshesLocked(MetalCachedImage image)
    {
        for (var index = _deferredRefreshes.Count - 1; index >= 0; index--)
        {
            if (_deferredRefreshes[index].Image == image)
            {
                _ = TakeDeferredLocked(index);
            }
        }
    }
}
