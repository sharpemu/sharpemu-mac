// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Images;

namespace SharpEmu.Gpu.Metal.V2;

// Upstream's resident-mip images (ImageDescription.FirstLevel / HostExtent, ImageRequestBuilders.PartialMipChains,
// ResidentImageBacking), in the cache's one mechanism for a texture that holds part of its chain
// (MetalImageCache.BaseLevel.cs).
//
// Upstream describes a sampled texture whose T# starts past mip 0 from that level: Resources.Levels counts the levels
// from FirstLevel, the host image is HostExtent (Extent >> FirstLevel) with those levels, and a view's levels are
// host levels. A streamed texture maps only its resident mips (a tiled chain lays the small levels first and level 0
// last, so they are a mapped prefix of the range), and its upload reads only their bytes, through ResidentImageBacking.
// Vulkan keeps such an image apart from the whole chain until the guest has mapped all of it (or another image over
// the range holds GPU contents), and makes a new one when the T# starts at a finer level.
//
// The Metal cache already holds a sampled texture from its BASE_LEVEL: the image's description stays the whole guest
// chain and only the texture, its upload and its page watch are cut (MetalCachedImage.FirstLevel, ResidentStart/End).
// That covers what upstream's image does - a texture of HostExtent with the levels from FirstLevel, viewed with guest
// level L as texture level L - FirstLevel, grown when a finer level is bound, never written back while cut - and more
// (a texture whose guest memory is all mapped is cut too, and a coarser binding shrinks it after a window of frames).
// So a resident-levels request is taken as the whole chain it is part of (ImageRequestBuilders.WithWholeMipChain:
// the view rebased onto guest levels), which keys it with every other request for those guest bytes, and the levels
// upstream would leave out become a floor on where the texture may start (ResidentFloor): while the guest has not
// mapped the chain, the texture never holds a level before the binding's BASE_LEVEL, whatever the memory policy
// (SHARPEMU_METAL_V2_BASE_LEVEL=0, a block-compressed base the policy would move to a block-aligned level) would
// choose. One mechanism decides the texture's levels; the request's FirstLevel is one of its inputs.
//
// SHARPEMU_METAL_V2_PARTIAL_MIP_CHAINS=0 (default on) leaves the request builders on whole chains, drops the floor and
// reads image bytes as before (a readable prefix, zeros after it) instead of through ResidentImageBacking.
internal sealed partial class MetalImageCache
{
    /// <summary>
    /// Whether the cache takes upstream's resident-levels textures (and the backend asks the request builders for them:
    /// MetalGpuBackend.ConfigureSharedGpuCode) and reads image bytes through <see cref="ResidentImageBacking"/>.
    /// Tests switch it.
    /// </summary>
    internal static bool PartialMipChains { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_PARTIAL_MIP_CHAINS") != "0";

    private long _residentRequests;
    private long _residentFloors;
    private MetalGuestMemory.StagingRead? _imageRead;

    /// <summary>Requests described from a later level, and textures held from BASE_LEVEL because the chain is not mapped.</summary>
    public (long Requests, long Floors) ResidentLevelCounters =>
        (Volatile.Read(ref _residentRequests), Volatile.Read(ref _residentFloors));

    // The staged read of image bytes: ResidentImageBacking's policy, or the binding's prefix read with the switch off.
    private MetalGuestMemory.StagingRead? ImageRead => PartialMipChains ? _imageRead ??= _guestMemory.ReadImageBacking : null;

    /// <summary>
    /// Whether the texels of an image over this range could be read from the guest: the range is valid and its first
    /// page is mapped (upstream's GuestImageCache.IsBackedImageRange). A descriptor read from a table the guest no longer
    /// uses may name memory it has unmapped since.
    /// </summary>
    public bool IsBackedImageRange(GuestSpan data) => ImageDescription.IsValidRange(data) && _backing.IsBackedRange(data.Address, 1);

    // A request described from its first resident level, as the whole guest chain it is part of: the same image as every
    // other request for those bytes, viewed from the same guest level (BaseLevel.cs holds the texture from there).
    private void TakeWholeChain(ref ImageRequest request)
    {
        if (request.Description.FirstLevel == 0)
        {
            return;
        }

        request = ImageRequestBuilders.WithWholeMipChain(request);
        Interlocked.Increment(ref _residentRequests);
    }

    /// <summary>
    /// Upstream's resident levels as a floor: a sampled texture whose binding starts at guest level
    /// <paramref name="baseLevel"/> holds no level before it while the guest has not mapped the chain's whole range and
    /// no image over it holds GPU contents (GuestImageCache.HoldsWholeChain), for any request the builder would describe
    /// from that level (ImageRequestBuilders.Texture: no volume, multisample, metadata or depth). Returns
    /// <paramref name="level"/>, the memory policy's choice (DesiredFirstLevel), raised to the base level when that holds.
    /// Asked only when the policy chose a finer level than the binding's, which with the base-level cut on is a
    /// block-compressed base it moved to a block-aligned level, and with it off is every texture bound past mip 0.
    /// </summary>
    private uint ResidentFloor(in ImageDescription description, uint baseLevel, uint level, bool gpuContents)
    {
        if (level >= baseLevel || !PartialMipChains || gpuContents || baseLevel >= description.Resources.Levels ||
            description.Resources.Levels > ImageDescription.MaxLevels || !ImageDescription.IsValidRange(description.Data) ||
            description.IsVolume || description.Samples != 1 || description.HasMetadata || description.IsDepth ||
            _backing.IsBackedRange(description.Data.Address, description.Data.Size))
        {
            return level;
        }

        Interlocked.Increment(ref _residentFloors);
        return baseLevel;
    }

    // Whether any image over a new texture's range holds what guest memory does not: upstream then takes the whole
    // chain, sharing that image rather than reading guest memory into another. Asked before the overlaps are resolved,
    // as upstream does, since resolving them may release those images.
    private bool CandidatesHoldGpuContents(in ImageQueryResults candidates)
    {
        foreach (var candidate in candidates)
        {
            if (_slots.TryGet(candidate) is { } image &&
                (image.IsGpuModified || image.Uses.RenderTarget || image.Uses.Storage || image.Uses.DepthTarget))
            {
                return true;
            }
        }

        return false;
    }

    // The level a new sampled texture starts at: the memory policy, floored at the binding's base level while the
    // guest has not mapped the chain.
    private uint InsertFirstLevel(in ImageRequest request, bool gpuContents)
    {
        var baseLevel = request.View.BaseLevel;
        var level = DesiredFirstLevel(in request.Description, baseLevel);
        return level < baseLevel ? ResidentFloor(in request.Description, baseLevel, level, gpuContents) : level;
    }
}
