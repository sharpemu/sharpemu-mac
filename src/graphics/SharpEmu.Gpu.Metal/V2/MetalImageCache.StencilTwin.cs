// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Images;

namespace SharpEmu.Gpu.Metal.V2;

// A multisampled depth-stencil surface whose stencil plane a shader reads through an 8-bit integer
// view binds a stencil view (X32_Stencil8) of the depth texture, as a depth surface sampled as its
// colour twin binds the depth texture (MetalImageCache.DepthTwin.cs). Nothing else can serve that
// read: the hand-over between a depth image and an 8-bit image over its stencil bytes goes through a
// buffer (MetalImageTransfer.TryEncodeStencilFromColor) or guest memory, and a multisampled texture
// has neither a blit to a buffer nor an upload. Ghost of Tsushima copies its two-sample checkerboard
// G-buffer depth and stencil into the depth target its terrain draws against; the stencil came through
// an 8-bit image nothing had filled, so every object's stencil reference read zero and the terrain
// material passes treated the objects' pixels by stale stencil.
//
// Only sampling goes this way; an image the guest stores into keeps its own texture.
internal sealed partial class MetalImageCache
{
    // SHARPEMU_METAL_V2_STENCIL_TWIN=0 gives such a read its own (unfilled) 8-bit texture again.
    private static readonly bool StencilTwinsEnabled =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_STENCIL_TWIN") != "0";

    private long _stencilTwinBinds;

    // SHARPEMU_METAL_V2_TRACE_STENCIL_TWIN=1 (diagnostic): each multisampled 8-bit integer request once per address, with
    // what each stencil owner over its bytes would need to serve it.
    internal static readonly bool TraceStencilTwins = Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_TRACE_STENCIL_TWIN") == "1";
    private readonly Dictionary<ulong, int> _tracedStencilTwins = new();

    internal void TraceStencilTwin(in ImageRequest request, bool storage = false)
    {
        ref readonly var requested = ref request.Description;
        var key = requested.Data.Address ^ ((ulong)request.Role << 56);
        var seen = _tracedStencilTwins.GetValueOrDefault(key) + 1;
        _tracedStencilTwins[key] = seen;
        if (_tracedStencilTwins.Count > 64 || seen is not (1 or 50 or 500 or 5000))
        {
            return;
        }

        var owners = new System.Text.StringBuilder();
        using (_lock.Hold())
        {
            foreach (var ownerIdentifier in _stencilOwners)
            {
                if (_slots.TryGet(ownerIdentifier) is not { } depth || !depth.Description.HasStencil)
                {
                    continue;
                }

                ref readonly var described = ref depth.Description;
                if (described.Stencil.Address != requested.Data.Address &&
                    !GuestRangeOverlap.Bytes(described.Stencil.Address, described.Stencil.Size, requested.Data.Address, requested.Data.Size))
                {
                    continue;
                }

                owners.Append($" owner@0x{described.Data.Address:X} stencil=0x{described.Stencil.Address:X}+0x{described.Stencil.Size:X} " +
                    $"fmt={depth.Shape.PixelFormat} samples={described.Samples} extent={described.Extent.Width}x{described.Extent.Height} " +
                    $"level0={depth.FirstLevel == 0} view_usage={(depth.Shape.Usage & Mtl.TextureUsagePixelFormatView) != 0} tex={depth.Texture != 0};");
            }
        }

        Console.Error.WriteLine(
            $"[METAL-V2][STENCIL-TWIN] request@0x{requested.Data.Address:X}+0x{requested.Data.Size:X} role={request.Role} storage={storage} " +
            $"view={request.View.Format} samples={requested.Samples} extent={requested.Extent.Width}x{requested.Extent.Height} " +
            $"levels={requested.Resources.Levels} layers={requested.TransferLayers}{(owners.Length == 0 ? " no stencil owner" : owners.ToString())}" +
            (owners.Length == 0 ? AllStencilOwners() : string.Empty));
    }

    private string AllStencilOwners()
    {
        var all = new System.Text.StringBuilder(" all:");
        using (_lock.Hold())
        {
            foreach (var ownerIdentifier in _stencilOwners)
            {
                if (_slots.TryGet(ownerIdentifier) is { } depth)
                {
                    ref readonly var described = ref depth.Description;
                    all.Append($" [0x{described.Data.Address:X}+0x{described.Data.Size:X} stencil=0x{described.Stencil.Address:X}+0x{described.Stencil.Size:X} " +
                        $"samples={described.Samples} {described.Extent.Width}x{described.Extent.Height}]");
                }
            }
        }

        return all.ToString();
    }

    /// <summary>Sampled stencil reads served by a depth texture's stencil view since the last read.</summary>
    public long TakeStencilTwinBinds() => Interlocked.Exchange(ref _stencilTwinBinds, 0);

    /// <summary>
    /// The multisampled depth image whose stencil plane <paramref name="request"/> samples as 8-bit
    /// integers, if one is cached: the caller binds it, and its view of the request is the stencil view.
    /// </summary>
    public bool TryFindStencilTwin(in ImageRequest request, out ResourceSlotIdentifier depthImage)
    {
        depthImage = ResourceSlotIdentifier.Invalid;
        ref readonly var requested = ref request.Description;
        if (TraceStencilTwins && requested.PixelFormat == GpuFormat.R8Uint && requested.Samples > 1)
        {
            TraceStencilTwin(in request);
        }

        if (!StencilTwinsEnabled || request.Role != ImageRole.Texture || requested.Samples <= 1 ||
            requested.PixelFormat != GpuFormat.R8Uint || request.View.Format != GpuFormat.R8Uint ||
            requested.Resources.Levels != 1 || requested.TransferLayers != 1)
        {
            return false;
        }

        using var held = _lock.Hold();
        foreach (var ownerIdentifier in _stencilOwners)
        {
            if (_slots.TryGet(ownerIdentifier) is not { } depth || !depth.Description.HasStencil ||
                depth.Description.Stencil.Address != requested.Data.Address || depth.Texture == 0 ||
                depth.FirstLevel != 0 || depth.Shape.PixelFormat != MtlFormat.Depth32FloatStencil8 ||
                (depth.Shape.Usage & Mtl.TextureUsagePixelFormatView) == 0 ||
                depth.Description.Samples != requested.Samples ||
                depth.Description.Extent.Width != requested.Extent.Width ||
                depth.Description.Extent.Height != requested.Extent.Height)
            {
                continue;
            }

            _stencilTwinBinds++;
            depthImage = ownerIdentifier;
            return true;
        }

        return false;
    }
}
