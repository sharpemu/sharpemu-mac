// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Images;

namespace SharpEmu.Gpu.Metal.V2;

// A depth surface whose HTILE a compute pass cleared keeps that clear pending until the surface is next bound as a depth
// target, whose load then clears it (MetalRenderHost.AcquireDepthAttachment). A surface sampled before that bind kept its
// older contents, where the console's texture unit reads the cleared value through the HTILE. A game that clears an
// occlusion depth this way each frame and builds its culling pyramid from it then culled with the occluders of the last
// frame that drew any (Demon's Souls, upstream cb790c35e).
//
// A sample of a one-layer, one-level depth surface with a pending clear now clears it first, with the clear value of the
// surface's last depth-target bind (the value that bind would load), and consumes the clear. A layered surface keeps its
// pending clear for its next bind, as its clear bits are not exact per layer. The stencil plane is left as it is.
// SHARPEMU_METAL_V2_SAMPLED_DEPTH_CLEAR=0 leaves every pending clear to the next bind again.
internal sealed partial class MetalImageCache
{
    private static readonly bool SampledDepthClearDefault =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_SAMPLED_DEPTH_CLEAR") != "0";

    /// <summary>Whether a sample takes a depth surface's pending HTILE clear (test hook; see the file comment).</summary>
    internal bool SampledDepthClears { get; set; } = SampledDepthClearDefault;

    // Moves whenever a sampled depth surface may have a clear to take that it had not before: an HTILE slice marked
    // cleared, a new registration, or a first clear value. A memoised sampled binding of such a surface is stamped
    // with it (TryStampTextureLocked), so the next bind after a clear takes the full acquire, which records the clear.
    private long _sampledDepthClearVersion;

    private long _sampledDepthClearsTaken;

    /// <summary>Pending HTILE clears a sample recorded since the last call.</summary>
    public long TakeSampledDepthClears() => Interlocked.Exchange(ref _sampledDepthClearsTaken, 0);

    private void NoteHtileClearPendingLocked() => Interlocked.Increment(ref _sampledDepthClearVersion);

    /// <summary>
    /// The depth clear value (DB_DEPTH_CLEAR) of a depth-target bind with HTILE: what a pending HTILE clear loads at
    /// the next bind, and what a sample reads before that bind.
    /// </summary>
    public void NoteDepthClearValue(ulong htileAddress, float value)
    {
        using var held = _lock.Hold();
        if (_surfaceMetadata.TryGetValue(htileAddress, out var found) && found.Kind == SurfaceMetadataKind.HTile)
        {
            if (!found.HasDepthClearValue)
            {
                NoteHtileClearPendingLocked();
            }

            found.DepthClearValue = value;
            found.HasDepthClearValue = true;
        }
    }

    // The stamp value for a sampled binding of the image: the version above when the image is the shape whose samples
    // can take a pending clear, otherwise -1 (never compared). Caller holds the lock.
    private long SampledDepthClearStamp(MetalCachedImage image) =>
        SampledDepthClears && IsSampledDepthClearShape(in image.Description) ? Volatile.Read(ref _sampledDepthClearVersion) : -1;

    private bool SampledDepthClearStampHolds(long stamped) =>
        stamped < 0 || stamped == Volatile.Read(ref _sampledDepthClearVersion);

    private static bool IsSampledDepthClearShape(in ImageDescription description) =>
        description.IsDepth && description.Resources.Levels == 1 && description.Resources.Layers == 1 &&
        description.Metadata.Kind == MetadataKind.Htile;

    // The registration whose pending clear a sample of the image takes, or null. Caller holds the lock.
    private SurfaceMetadata? SampleClearsDepthLocked(MetalCachedImage image)
    {
        ref readonly var description = ref image.Description;
        return SampledDepthClears && image.Texture != 0 && IsSampledDepthClearShape(in description) &&
            _surfaceMetadata.TryGetValue(description.Metadata.Range.Address, out var metadata) &&
            metadata is { Kind: SurfaceMetadataKind.HTile, Invalidated: false, HasDepthClearValue: true } &&
            metadata.IsSliceClear(0)
                ? metadata
                : null;
    }

    // Records the clear the next depth-target bind would load, ahead of a sample, and consumes it. A write like any
    // pass's: the image takes the bytes from its aliases and owes guest memory a writeback. Its stencil plane, which the
    // clear leaves alone, is brought up to date first, since taking ownership drops what the image still owed it.
    // Caller holds the lock.
    private void ClearDepthBeforeSampleLocked(ResourceSlotIdentifier imageIdentifier, MetalCachedImage image, SurfaceMetadata metadata)
    {
        PromoteLinearView(image);
        MapSparseLayers(image, 0, 1);
        WatchImage(imageIdentifier);
        DrainOverlappingDeferredRefreshesLocked(image);
        SynchronizeAliases(imageIdentifier, takesGpuOwnership: true, role: "depth");
        if (image.Description.HasStencil)
        {
            RefreshFromGuest(image);
        }

        var range = new SubresourceRange(0, 1, 0, 1);
        _owner.ClearTextureDepthStencil(
            image.Texture, image.Shape.PixelFormat, in range,
            image.Description.Extent.Width, image.Description.Extent.Height,
            metadata.DepthClearValue, 0, clearDepth: true, clearStencil: false);
        NoteRefusedRewrite(image, whole: !image.Description.HasStencil);
        image.RefreshComplete();
        image.MarkGpuModified();
        NoteLinearGpuWrite(image);
        image.GpuWriteSerial = _owner.CurrentSerial;
        metadata.SetSliceClear(0, false);
        _sampledDepthClearsTaken++;
        if (SharpEmu.Libs.Gpu.Rendering.RenderExecutor.TraceHtile)
        {
            Console.Error.WriteLine(
                $"[METAL-V2][HTILE] sampled-clear depth=0x{image.Description.Data.Address:X16} " +
                $"htile=0x{image.Description.Metadata.Range.Address:X16} value={metadata.DepthClearValue}");
        }
    }
}
