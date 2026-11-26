// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Metal.V2;

// Where an overlapping request decides whether a cached image is safe to release (ResolveOverlap), an image the current
// frame used counts as in use, and so does a recurring alias: one used in each of the two frames before this one
// (upstream c30081163, GuestImageCache.UsedThisFrameForOverlap). Collection ticks advance per completed submission and a
// frame spans hundreds of them, so the tick window alone released an alias the frame had used early on, and two aliases
// of the same memory that a title uses every frame swapped at each one's first use in a frame: each request released the
// other (a writeback when the GPU wrote it) and the next frame made it again from guest memory. An image the frame
// before used but the one before it did not is released as before, so the one-off images a title places in the same
// memory frame after frame do not pile up. Before the first presented frame no frame is known, and the tick window
// decides alone, as it did. SHARPEMU_METAL_V2_FRAME_ALIASES=0 restores the tick window alone.
internal sealed partial class MetalImageCache
{
    private static readonly bool FrameAliasesDefault =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_FRAME_ALIASES") != "0";

    /// <summary>Whether overlap resolution keeps aliases this frame or a recurring pair uses (test hook; see above).</summary>
    internal bool FrameAliases { get; set; } = FrameAliasesDefault;

    // The collection tick at which the frame before the previous one began (MarkFrame).
    private ulong _secondPreviousFrameStartTick;

    // Stamps a use of the image: the first use in a frame keeps the tick of the use before it. Caller holds the lock.
    private void TouchImage(MetalCachedImage image)
    {
        if (image.LastAccessTick < _frameStartTick)
        {
            image.PreviousUseTick = image.LastAccessTick;
        }

        image.LastAccessTick = _collectionTick;
    }

    // Whether an overlapping cached image counts as in use for overlap resolution. Caller holds the lock.
    private bool UsedThisFrameForOverlap(MetalCachedImage cached)
    {
        if (!FrameAliases || _idleClock.Frames == 0)
        {
            return false;
        }

        var lastUse = cached.LastAccessTick;
        if (lastUse >= _frameStartTick)
        {
            return true;
        }

        var previous = cached.PreviousUseTick;
        return _idleClock.Frames >= 2 && lastUse >= _previousFrameStartTick &&
            previous != ulong.MaxValue && previous >= _secondPreviousFrameStartTick && previous < _previousFrameStartTick;
    }
}
