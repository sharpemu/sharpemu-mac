// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Images;

namespace SharpEmu.Gpu.Metal.V2;

// Sampled textures held from their T# BASE_LEVEL.
//
// On the PS5 the texture unit never reads a level finer than BASE_LEVEL (the SDK's
// Texture::setMipLevelRange "redefines the size of the base map to that at baseMipLevel";
// LAST_LEVEL bounds the other end), and a title's texture streamer uses exactly that: Demon's
// Souls keeps only some levels of each streamed texture resident, decommits the guest memory
// of the finer ones, and writes BASE_LEVEL = the resident top level into the T#. A texture
// allocated with the whole chain spends three quarters of its bytes or more on levels the GPU
// cannot sample, and its upload reads guest pages the title has released.
//
// So a sampled-only image gets a texture of the guest levels from BASE_LEVEL down (to the
// chain's last level, which keeps a different LAST_LEVEL from changing the texture over
// levels a quarter the size of the one above them). The image's description stays the full
// guest chain: identity, the page index and every guest offset are the guest's, and only
// the texture, the bytes uploaded into it and the page watch are cut to those levels.
// Shaders are unchanged: they sample through a view whose level 0 is BASE_LEVEL either way
// (MetalCachedImage.GetOrCreateView).
//
// The texture follows the binding:
//  - a binding that needs a finer level gets a texture with it at once, filled from guest
//    memory (the streamer only lowers BASE_LEVEL after the level is resident);
//  - a binding that needs only coarser levels keeps the texture through a view until no
//    binding for a whole window of presented frames (ShrinkWindowFrames) needed the finer
//    ones, and then shrinks it to the finest level any binding in that window needed -
//    not to the level of the binding that happens to look - because every level dropped
//    gives back about three quarters of what is left and the coarser levels that are
//    uploaded again are only a third of the level above them. The window is per image and
//    per level: two descriptors of one texture with different BASE_LEVELs keep it at the
//    finer one, and a texture that had to grow back soon after a shrink doubles its window
//    (up to 2^MaxShrinkBackoff), so a descriptor that comes back every so often cannot make
//    it trade reallocations. Frames, not seconds or submissions, because a title presenting
//    twice a second still binds its textures once per frame. A texture whose finer levels
//    hold less than ShrinkMinimumBytes is not worth the reallocation;
//  - anything that writes the image (render target, storage, clear, copy, kernel fill)
//    gets the whole chain first (PromoteLinearView), so a cut texture is never written
//    back and write-back and retiling of GPU-written images are as they were.
//
// Only a binding tells the cache a texture's BASE_LEVEL (the T# lives in guest memory), so a
// texture the title coarsens and stops binding keeps its levels until age retirement or the
// budget takes the whole image.
//
// SHARPEMU_METAL_V2_BASE_LEVEL=0 allocates every texture with its whole chain again, except where upstream's
// resident levels keep the levels before BASE_LEVEL out because the guest has not mapped them
// (MetalImageCache.ResidentLevels.cs).
internal sealed partial class MetalImageCache
{
    // Finer levels that hold less than this are kept through a view rather than reallocated.
    private const ulong ShrinkMinimumBytes = 256 * 1024;

    // Presented frames the finer levels must go unneeded before the texture is cut down, and
    // how many times a reversed shrink may double that (8 << 5 = 256 frames at most).
    private const long ShrinkWindowFrames = 8;
    private const int MaxShrinkBackoff = 5;

    /// <summary>Whether sampled textures are allocated from their BASE_LEVEL; tests switch it.</summary>
    internal static bool BaseLevelAllocation { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_BASE_LEVEL") != "0";

    private long _baseLevelCreated;
    private long _baseLevelGrows;
    private long _baseLevelShrinks;
    private long _baseLevelPromotions;
    private long _baseLevelRegrows;
    private long _baseLevelRegrowsReported;
    private (long Created, long Grows, long Shrinks, long Promotions) _baseLevelReported;

    /// <summary>Texture reshapes since the cache was made: created cut, grown, shrunk, given the whole chain.</summary>
    public (long Created, long Grows, long Shrinks, long Promotions) BaseLevelCounters =>
        (Volatile.Read(ref _baseLevelCreated), Volatile.Read(ref _baseLevelGrows),
            Volatile.Read(ref _baseLevelShrinks), Volatile.Read(ref _baseLevelPromotions));

    /// <summary>
    /// The guest level a sampled texture of this description is allocated from when its
    /// binding's BASE_LEVEL is <paramref name="baseLevel"/>: the level itself for a plain
    /// mipped colour texture, zero for anything with a layout the cut does not cover (a
    /// volume, multisample, depth, stencil, DCC metadata, the byte-swapped BGRA16 path).
    /// A block-compressed texture starts no later than its last level of whole blocks each
    /// way, so the base of the Metal texture is block-aligned as a whole-chain one is.
    /// </summary>
    internal static uint DesiredFirstLevel(in ImageDescription description, uint baseLevel)
    {
        if (!BaseLevelAllocation || baseLevel == 0 || ImageDescription.IsEmptyRange(description.Data) ||
            description.Resources.Levels <= 1 || description.Resources.Levels > ImageDescription.MaxLevels ||
            description.IsVolume || description.Samples != 1 || description.IsDepth || description.HasMetadata ||
            description.HasStencil || description.Bgra16)
        {
            return 0;
        }

        var level = Math.Min(baseLevel, description.Resources.Levels - 1);
        if (description.IsBlock)
        {
            while (level > 0 &&
                   ((description.Extent.Width >> (int)level) % 4 != 0 || (description.Extent.Height >> (int)level) % 4 != 0))
            {
                level--;
            }
        }

        return level;
    }

    // The same for an image the cache holds: only one that has been sampled and nothing else.
    private static uint DesiredFirstLevel(MetalCachedImage image, uint baseLevel) =>
        HoldsGpuContents(image) ? 0 : DesiredFirstLevel(in image.Description, baseLevel);

    // An image that is a window view, or that something other than sampling used, holds its whole chain.
    private static bool HoldsGpuContents(MetalCachedImage image)
    {
        var uses = image.Uses;
        return image.IsWindowLinearView || image.IsGpuModified ||
            uses.Storage || uses.RenderTarget || uses.DepthTarget || uses.VideoOut;
    }

    // The guest levels the image's texture holds, in the description's own level numbers.
    private static SubresourceRange ResidentRange(MetalCachedImage image) =>
        new(image.FirstLevel, image.Description.Resources.Levels - image.FirstLevel,
            0, image.Description.TransferLayers);

    // Guest bytes of levels [from, to): what a texture from `to` leaves out next to one from `from`.
    private static ulong LevelBytesBetween(in ImageDescription description, uint from, uint to)
    {
        ulong bytes = 0;
        for (var level = from; level < to && level < ImageDescription.MaxLevels; level++)
        {
            bytes += description.MipLayout[(int)level].Size;
        }

        return bytes;
    }

    /// <summary>
    /// Fits the texture of a sampled image to a binding's BASE_LEVEL before the binding takes
    /// its view: a finer level than the texture holds grows it at once; once no binding for a
    /// window of presented frames needed its finer levels, it shrinks to the finest level they
    /// did need. Caller holds the lock.
    /// </summary>
    private void FitResidentLevelsLocked(MetalCachedImage image, in ImageViewDescription view)
    {
        if (image.Texture == 0 || image.IsWindowLinearView)
        {
            return;
        }

        var frame = _idleClock.Frames;
        var window = ShrinkWindowFrames << image.ShrinkBackoff;
        var desired = DesiredFirstLevel(image, view.BaseLevel);
        if (desired < image.FirstLevel)
        {
            // Never grown into levels before the binding's that the guest has not mapped (ResidentLevels.cs).
            desired = ResidentFloor(in image.Description, view.BaseLevel, desired, HoldsGpuContents(image));
        }

        NoteNeededLevel(image, desired, frame, window);
        if (desired < image.FirstLevel)
        {
            if (image.LevelsLastShrunk && frame - image.LevelsChangedFrame < ShrinkWindowFrames << MaxShrinkBackoff)
            {
                // The shrink was premature: wait longer before the next one.
                image.ShrinkBackoff = (byte)Math.Min(image.ShrinkBackoff + 1, MaxShrinkBackoff);
                _baseLevelRegrows++;
            }

            ReshapeLevelsLocked(image, desired);
            _baseLevelGrows++;
            image.LevelsChangedFrame = frame;
            image.LevelsLastShrunk = false;
            return;
        }

        // Shrinking is the memory policy's; the resident-levels floor only ever keeps levels out.
        var needed = Math.Min(image.NeededLevel, image.NeededLevelBefore);
        if (!BaseLevelAllocation || needed <= image.FirstLevel || frame - image.LevelsChangedFrame < window ||
            LevelBytesBetween(in image.Description, image.FirstLevel, needed) < ShrinkMinimumBytes)
        {
            return;
        }

        ReshapeLevelsLocked(image, needed);
        _baseLevelShrinks++;
        image.LevelsChangedFrame = frame;
        image.LevelsLastShrunk = true;
    }

    // Folds one binding's level into the image's two windows. A window's bindings all fall
    // in its first `window` frames (a later one starts the next window), so a window dropped
    // two windows on holds nothing from the last `window` frames, and the two kept hold at
    // least that span. The window only ever widens, which keeps that true across a backoff.
    private static void NoteNeededLevel(MetalCachedImage image, uint level, long frame, long window)
    {
        var age = frame - image.NeededWindowFrame;
        if (age >= window)
        {
            image.NeededLevelBefore = age < 2 * window ? image.NeededLevel : MetalCachedImage.NoNeededLevel;
            image.NeededLevel = MetalCachedImage.NoNeededLevel;
            image.NeededWindowFrame = frame;
        }

        image.NeededLevel = Math.Min(image.NeededLevel, level);
    }

    /// <summary>
    /// Gives a sampled-only image a new texture holding the guest levels from
    /// <paramref name="firstLevel"/>, refilled from guest memory on its next use; the old
    /// texture waits for the submissions that bound it. Caller holds the lock.
    /// </summary>
    private void ReshapeLevelsLocked(MetalCachedImage image, uint firstLevel, bool dropLossy = false)
    {
        // Without dropLossy a lossy texture stays one; with it the new texture has the full
        // usage of the format and is lossless, and an image made again over the same bytes
        // starts that way too. It refills from guest memory, which holds what the lossy
        // texture was only an approximation of.
        var shape = _transfer.ShapeFor(image.Description, firstLevel) with { Usage = image.Shape.Usage };
        if (image.Shape.Lossy)
        {
            if (dropLossy)
            {
                shape = _transfer.ShapeFor(image.Description, firstLevel);
                ForgetLossy(image);
            }
            else
            {
                shape = shape with { Lossy = true };
            }
        }

        var texture = CreateTexture(in shape, image.Description);
        var (start, end) = _transfer.LevelBytes(image.Description, firstLevel);
        RetireImage(image.ReplaceLevels(texture, in shape, firstLevel, start, end));
        RecountAllocatedLocked(image);
        if (image.IsWatched)
        {
            // The watch follows the bytes the texture now reads.
            WatchImageLocked(image);
        }
    }

    // Census of the cut textures: how many, what they hold, and what the levels they leave
    // out would have cost. Caller holds the lock.
    private void CountResidentLevels(MetalCachedImage image, ref long images, ref ulong allocated, ref ulong dropped)
    {
        if (image.FirstLevel == 0)
        {
            return;
        }

        images++;
        allocated += image.AllocatedBytes;
        dropped += LevelBytesBetween(in image.Description, 0, image.FirstLevel);
    }

    private string DescribeBaseLevels(long images, ulong allocated, ulong dropped)
    {
        const double Mb = 1048576.0;
        var current = BaseLevelCounters;
        var last = _baseLevelReported;
        _baseLevelReported = current;
        var regrows = Volatile.Read(ref _baseLevelRegrows);
        var lastRegrows = _baseLevelRegrowsReported;
        _baseLevelRegrowsReported = regrows;
        return string.Create(CultureInfo.InvariantCulture,
            $"base_level_images={images} base_level_allocated_mb={allocated / Mb:F0} " +
            $"base_level_dropped_mb={dropped / Mb:F0} " +
            $"base_level_created={current.Created}(+{current.Created - last.Created}) " +
            $"base_level_grows={current.Grows}(+{current.Grows - last.Grows}) " +
            $"base_level_shrinks={current.Shrinks}(+{current.Shrinks - last.Shrinks}) " +
            $"base_level_regrows={regrows}(+{regrows - lastRegrows}) " +
            $"base_level_promotions={current.Promotions}(+{current.Promotions - last.Promotions})");
    }
}
