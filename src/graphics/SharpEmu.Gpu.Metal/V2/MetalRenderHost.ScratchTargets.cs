// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Rendering;

namespace SharpEmu.Gpu.Metal.V2;

// A colour target the draw writes nothing to, whose texture its pixel program reads or
// stores through a resource.
//
// Demon's Souls composites each character-creation face layer with a pixel program
// (0xF48B3044F3DACA7A) that binds the 2048x2048 face as its colour target with every
// channel masked off, only for the raster area, then samples the face and stores
// the result into it through an image. The GPU on the PS5 writes nothing back for a
// masked target. A Metal pass on a tile-based GPU stores the tile it loaded at the end of
// the pass, over every image store the program made, so each layer was lost and the face
// came out black, magenta or checkered. Sampling a texture that is also an attachment of
// the open pass is undefined on Metal in any case.
//
// Such an attachment is replaced in the pass by a memoryless texture of the same format
// and size. It is never loaded or stored, so the guest's texture is only a resource and
// the pass never writes it. A pass with a stand-in only continues into a draw that wants
// the same stand-ins, so a later draw that does write the target gets a pass with the real
// texture.
internal sealed partial class MetalRenderHost
{
    // SHARPEMU_METAL_V2_SCRATCH_TARGETS=0 keeps the guest's texture as the attachment of such
    // a pass, as before.
    private static readonly bool ScratchTargetsEnabled =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_SCRATCH_TARGETS") != "0";

    private static readonly nint SelScratchPixelFormat = MetalNative.Selector("pixelFormat");
    private static readonly nint SelScratchSetStorageMode = MetalNative.Selector("setStorageMode:");

    // The colour attachments of the open pass that are stand-ins, one bit each.
    private uint _passScratchMask;

    private readonly Dictionary<(ulong Format, uint Width, uint Height), nint> _scratchTargets = new();

    /// <summary>
    /// The colour attachments of the upcoming draw that it writes nothing to but whose
    /// texture its pixel stage binds, one bit each.
    /// </summary>
    private uint ScratchAttachmentMask(in RenderingState state, MetalPipelineEntry? upcoming)
    {
        if (!ScratchTargetsEnabled || upcoming?.StaticParameters is not { } parameters ||
            (_preparedStageMask & (1 << StageFragment)) == 0 || state.Samples > 1 || state.Layers > 1)
        {
            return 0;
        }

        var stage = _stageBindings[StageFragment];
        var count = Math.Min(Math.Min((int)state.ColorAttachmentCount, _attachmentCount), MaxColorAttachments);
        uint mask = 0;
        for (var index = 0; index < count; index++)
        {
            var target = _attachmentTextures[index];
            if (target == 0 || state.ColorAttachments[index].View.Handle == 0 || parameters.GetColorMask(index) != 0)
            {
                continue;
            }

            for (var image = 0; image < stage.ImageCount && image < stage.Images.Length; image++)
            {
                if (_imageCache.TryGetImage(stage.Images[image], out var cached) && cached.Texture == target)
                {
                    mask |= 1u << index;
                    break;
                }
            }
        }

        return mask;
    }

    /// <summary>A memoryless colour texture to stand in for an attachment of this view's format.</summary>
    private nint ScratchTarget(nint view, uint width, uint height)
    {
        var format = (ulong)MetalV2Native.SendValue(view, SelScratchPixelFormat);
        var key = (format, width, height);
        if (_scratchTargets.TryGetValue(key, out var texture))
        {
            return texture;
        }

        var descriptor = MetalNative.SendTextureDescriptor(
            MetalNative.Class("MTLTextureDescriptor"), SelTexture2DDescriptor, (nuint)format, width, height,
            mipmapped: false);
        if (descriptor == 0)
        {
            throw Fatal($"A stand-in colour target could not be described: format={format} size={width}x{height}.");
        }

        MetalV2Native.SendVoidValue(descriptor, SelMovieSetUsage, (nuint)Mtl.TextureUsageRenderTarget);
        MetalV2Native.SendVoidValue(descriptor, SelScratchSetStorageMode, Mtl.StorageModeMemoryless);
        texture = MetalNative.Send(_device, SelMovieNewTexture, descriptor);
        if (texture == 0)
        {
            throw Fatal($"A stand-in colour target could not be created: format={format} size={width}x{height}.");
        }

        _scratchTargets[key] = texture;
        return texture;
    }
}
