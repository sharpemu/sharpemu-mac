// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Images;

namespace SharpEmu.Gpu.Metal.V2;

// Texture usage that keeps a render target losslessly compressed.
//
// Apple GPUs store a texture compressed only when it has neither the shader-write nor the
// pixel-format-view usage. A colour target without compression costs every render pass
// that loads, stores or samples it the full bytes: a 1920x1080 RGBA16F full-screen pass
// measured 2.2-3.4x the GPU time on an M4. Most colour targets are only ever rendered to
// and sampled in their own format (or its sRGB twin, which Metal views without the usage),
// so an image first asked for as a colour target is created with ShaderRead |
// RenderTarget, and gains the rest of what ShapeFor gives the format the first time a
// shader stores to it or a view reinterprets its format. Gaining usage means a new texture
// with a device copy of the old one; the guest address and format are remembered, so an
// image made again over the same bytes starts with the usage it needed.
//
// The tiler's one-pass kernels write and read through a reinterpreting view and so skip a
// texture without those usages; its transfers then take the staged path, which is exact.
internal sealed partial class MetalImageCache
{
    private const ulong CompressionBreakingUsage =
        Mtl.TextureUsageShaderWrite | Mtl.TextureUsagePixelFormatView | (ulong)MtlTextureUsage.ShaderAtomic;

    private readonly HashSet<(ulong Address, GpuFormat Format)> _fullUsageImages = new();
    private long _usagePromotions;

    /// <summary>Compressible images that gained usage since the last read.</summary>
    public long TakeUsagePromotions() => Interlocked.Exchange(ref _usagePromotions, 0);

    /// <summary>
    /// The shape an image is created with: what the transfer gives the format, without the
    /// usages that break compression when the image is made for a colour target and has
    /// not needed them before. Caller holds the lock.
    /// </summary>
    private MetalTextureShape CreationShape(in ImageDescription description, ImageRole role)
    {
        var shape = _transfer.ShapeFor(in description);
        if (!MetalV2Diagnostics.CompressibleTargets || role != ImageRole.ColorTarget ||
            (shape.Usage & Mtl.TextureUsageRenderTarget) == 0 ||
            _fullUsageImages.Contains((description.Data.Address, description.PixelFormat)))
        {
            return shape;
        }

        return shape with { Usage = shape.Usage & ~CompressionBreakingUsage };
    }

    /// <summary>
    /// The view the presenter samples a display surface through. Its linear format and
    /// channel exchange are a view like any other, so the texture gains the usage the view
    /// needs first: without it Metal would hand back the surface with its swizzle dropped.
    /// </summary>
    public nint AcquirePresentationView(ResourceSlotIdentifier imageIdentifier, in ImageViewDescription view)
    {
        using var held = _lock.Hold();
        var image = _slots[imageIdentifier];
        EnsureViewUsageLocked(image, in view);
        return image.GetOrCreateView(view);
    }

    /// <summary>Makes sure the image's texture can be bound for a shader store. Caller holds the lock.</summary>
    private void EnsureStorageUsageLocked(MetalCachedImage image) =>
        EnsureTextureUsageLocked(image, Mtl.TextureUsageShaderWrite);

    /// <summary>Makes sure the image's texture can give this view. Caller holds the lock.</summary>
    private void EnsureViewUsageLocked(MetalCachedImage image, in ImageViewDescription view) =>
        EnsureTextureUsageLocked(image, image.UsageNeededFor(in view));

    /// <summary>
    /// Gives an image created compressible the full usage of its format when something
    /// needs a usage it lacks: a new texture, a device copy of every subresource into it,
    /// and the old texture retired once the copy and everything that bound it are done.
    /// A usage the format cannot have at all (an sRGB format off an Apple GPU is not writable) is left
    /// to the caller, which reports it as it always has. Caller holds the lock.
    /// </summary>
    private void EnsureTextureUsageLocked(MetalCachedImage image, ulong needed)
    {
        if (needed == 0 || image.Texture == 0 || image.IsWindowLinearView || (image.Shape.Usage & needed) == needed)
        {
            return;
        }

        if (image.Shape.Lossy)
        {
            // Lossy textures hold ShaderRead alone: any usage they lack means a lossless one.
            ReshapeLevelsLocked(image, image.FirstLevel, dropLossy: true);
            return;
        }

        var full = _transfer.ShapeFor(image.Description);
        var shape = image.Shape with { Usage = image.Shape.Usage | full.Usage };
        if ((shape.Usage & needed) != needed || shape.Usage == image.Shape.Usage)
        {
            return;
        }

        _fullUsageImages.Add((image.Description.Data.Address, image.Description.PixelFormat));
        // An upload still waiting for its pass would land in the texture being replaced.
        DrainOverlappingDeferredRefreshesLocked(image);
        var texture = CreateTexture(in shape, image.Description);
        var context = _owner.BeginTransfer(MetalTransferCause.ImageCopy, image.Texture, texture);
        _transfer.EncodeWholeTextureCopy(context, image.Texture, texture);
        RetireImage(image.ReplaceWithCopy(texture, in shape, _owner.CurrentSerial));
        RecountAllocatedLocked(image);
        _usagePromotions++;
    }
}
