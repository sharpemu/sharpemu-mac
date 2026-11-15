// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Images;

namespace SharpEmu.Gpu.Metal.V2;

// Two cached images over the same guest bytes that lay their layers out identically - a 64-layer and a 72-layer
// array from one base, or a one-layer surface placed on a layer of an array - map every shared guest byte to the
// same texel of the same format. A copy of those slices between their textures is then exactly what a writeback of
// one and an upload of the other through guest memory would leave in the target.
internal sealed partial class MetalImageTransfer
{
    private static readonly nint SelCopyTextureSlice = MetalNative.Selector(
        "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:" +
        "toTexture:destinationSlice:destinationLevel:destinationOrigin:");

    /// <summary>
    /// Whether target layers [<paramref name="targetFirstLayer"/>, + <paramref name="layers"/>) lie on exactly the
    /// guest bytes of source layers [<paramref name="sourceFirstLayer"/>, + <paramref name="layers"/>), byte for byte
    /// the same texels, and no other target layer touches a source byte. False for anything else: a depth or stencil
    /// plane, a volume, a mip chain, multisampling, a different format, extent, pitch or tile mode, a layer stride that
    /// differs or is not a whole multiple of the address difference, a guest size that is not a whole number of
    /// layers, and a tile mode that swizzles by slice index (RenderTarget and Depth 64 KiB blocks) at a layer offset
    /// other than zero, since the same bytes then hold different texels in each.
    /// </summary>
    public bool TryMatchLayerSubset(
        in ImageDescription source, in ImageDescription target,
        out uint sourceFirstLayer, out uint targetFirstLayer, out uint layers)
    {
        sourceFirstLayer = 0;
        targetFirstLayer = 0;
        layers = 0;
        if (_disposed || !SameLayerShape(in source, in target))
        {
            return false;
        }

        var sourcePlan = PlanFor(source);
        var targetPlan = PlanFor(target);
        if (sourcePlan.DepthStencil || targetPlan.DepthStencil || sourcePlan.Volume || targetPlan.Volume ||
            sourcePlan.Tiled != targetPlan.Tiled || sourcePlan.Swap != targetPlan.Swap ||
            sourcePlan.Format != targetPlan.Format ||
            !TryLayerStride(sourcePlan, source.Resources.Layers, out var stride) ||
            !TryLayerStride(targetPlan, target.Resources.Layers, out var targetStride) || stride != targetStride ||
            source.Data.Size != stride * source.Resources.Layers || target.Data.Size != stride * target.Resources.Layers)
        {
            return false;
        }

        // Target layer t sits on source layer t + shift.
        long shift;
        if (target.Data.Address >= source.Data.Address)
        {
            var delta = target.Data.Address - source.Data.Address;
            if (delta % stride != 0 || delta / stride >= source.Resources.Layers)
            {
                return false;
            }

            shift = (long)(delta / stride);
        }
        else
        {
            var delta = source.Data.Address - target.Data.Address;
            if (delta % stride != 0 || delta / stride >= target.Resources.Layers)
            {
                return false;
            }

            shift = -(long)(delta / stride);
        }

        var first = shift < 0 ? -shift : 0;
        var end = Math.Min((long)target.Resources.Layers, source.Resources.Layers - shift);
        if (end <= first)
        {
            return false;
        }

        for (var layer = first; layer < end; layer++)
        {
            if (!SameLayerLayout(sourcePlan, (int)(layer + shift), targetPlan, (int)layer))
            {
                return false;
            }
        }

        sourceFirstLayer = (uint)(first + shift);
        targetFirstLayer = (uint)first;
        layers = (uint)(end - first);
        return true;
    }

    /// <summary>
    /// The guest bytes, as offsets into the image, of layers [<paramref name="baseLayer"/>, + <paramref name="count"/>)
    /// of a one-level, non-volume image whose guest size is a whole number of equal layers: every byte of those layers
    /// and no byte of any other. False for anything else.
    /// </summary>
    public bool TryLayerBytes(in ImageDescription description, uint baseLayer, uint count, out ulong start, out ulong end)
    {
        start = 0;
        end = 0;
        var layers = description.Resources.Layers;
        if (_disposed || count == 0 || layers == 0 || baseLayer >= layers || count > layers - baseLayer ||
            description.IsVolume || description.Resources.Levels != 1 || ImageDescription.IsEmptyRange(description.Data))
        {
            return false;
        }

        var plan = PlanFor(description);
        ulong stride;
        if (plan.DepthStencil)
        {
            stride = plan.SliceSize;
            if (plan.Layers != layers)
            {
                return false;
            }
        }
        else if (plan.Volume || !TryLayerStride(plan, layers, out stride))
        {
            return false;
        }

        if (stride == 0 || stride * layers != description.Data.Size)
        {
            return false;
        }

        start = baseLayer * stride;
        end = start + count * stride;
        return true;
    }

    private static bool SameLayerShape(in ImageDescription source, in ImageDescription target) =>
        source.Type is GuestImageType.Color2D or GuestImageType.Color2DArray &&
        target.Type is GuestImageType.Color2D or GuestImageType.Color2DArray &&
        !source.IsDepth && !target.IsDepth && !source.HasStencil && !target.HasStencil &&
        source.Stencil.Size == 0 && target.Stencil.Size == 0 &&
        source.Samples == 1 && target.Samples == 1 &&
        source.Resources.Levels == 1 && target.Resources.Levels == 1 &&
        source.Resources.Layers != 0 && target.Resources.Layers != 0 &&
        source.GuestFormat == target.GuestFormat && source.PixelFormat == target.PixelFormat &&
        source.TileMode == target.TileMode && source.Bgra16 == target.Bgra16 &&
        source.Extent.Width == target.Extent.Width && source.Extent.Height == target.Extent.Height &&
        source.Extent.Depth <= 1 && target.Extent.Depth <= 1 &&
        source.Pitch == target.Pitch && source.BytesPerBlock == target.BytesPerBlock &&
        !ImageDescription.IsEmptyRange(source.Data) && !ImageDescription.IsEmptyRange(target.Data);

    // The guest distance between consecutive layers, confirmed against every layer's transfer: one tile transfer or
    // one copy per layer (one level), each at a whole number of strides from the first.
    private static bool TryLayerStride(TransferPlan plan, uint layers, out ulong stride)
    {
        stride = 0;
        if (plan.Tiled)
        {
            stride = plan.Layout.SourceSliceStride != 0 ? plan.Layout.SourceSliceStride : plan.Layout.Surface.BlockSliceSize;
            if (stride == 0 || plan.Tiles.Count != layers)
            {
                return false;
            }

            for (var layer = 0; layer < plan.Tiles.Count; layer++)
            {
                if (plan.Tiles[layer].Slice != (uint)layer || plan.Tiles[layer].Level != 0 ||
                    plan.Tiles[layer].TiledOffset != plan.Tiles[0].TiledOffset + (ulong)layer * stride ||
                    plan.Tiles[layer].TiledOffset + plan.Tiles[layer].TiledSize > ((ulong)layer + 1) * stride)
                {
                    return false;
                }
            }

            return true;
        }

        stride = plan.Layout.SliceStride;
        if (stride == 0 || plan.Copies.Count != layers)
        {
            return false;
        }

        for (var layer = 0; layer < plan.Copies.Count; layer++)
        {
            if (plan.Copies[layer].Slice != (uint)layer || plan.Copies[layer].Level != 0 ||
                plan.Copies[layer].BufferOffset != plan.Copies[0].BufferOffset + (ulong)layer * stride ||
                CopyEnd(plan, plan.Copies[layer]) > ((ulong)layer + 1) * stride)
            {
                return false;
            }
        }

        return true;
    }

    // One layer of each plan, compared field by field except where in the image it sits (its offsets and slice): the
    // swizzle's slice term (SurfaceZ) is compared, so a slice-swizzled mode matches only at the same layer index.
    private static bool SameLayerLayout(TransferPlan sourcePlan, int sourceLayer, TransferPlan targetPlan, int targetLayer)
    {
        if (sourcePlan.Tiled)
        {
            var a = sourcePlan.Tiles[sourceLayer];
            var b = targetPlan.Tiles[targetLayer];
            return a.Kind == b.Kind && a.BytesPerElement == b.BytesPerElement && a.TiledSize == b.TiledSize &&
                a.LinearSize == b.LinearSize && a.LinearSliceStride == b.LinearSliceStride &&
                a.Width == b.Width && a.Height == b.Height && a.Depth == b.Depth && a.Pitch == b.Pitch &&
                a.TailX == b.TailX && a.TailY == b.TailY && a.Tail == b.Tail &&
                a.TiledWidth == b.TiledWidth && a.TiledHeight == b.TiledHeight &&
                a.SurfaceZ == b.SurfaceZ && a.BaseY == b.BaseY &&
                // TryLayerStride put every layer at its first layer's offset plus whole strides.
                sourcePlan.Tiles[0].TiledOffset == targetPlan.Tiles[0].TiledOffset;
        }

        var c = sourcePlan.Copies[sourceLayer];
        var d = targetPlan.Copies[targetLayer];
        return c.BufferRowLength == d.BufferRowLength && c.BufferImageHeight == d.BufferImageHeight &&
            c.OriginZ == d.OriginZ && c.OriginY == d.OriginY && c.Width == d.Width && c.Height == d.Height &&
            c.Depth == d.Depth && sourcePlan.Copies[0].BufferOffset == targetPlan.Copies[0].BufferOffset;
    }

    /// <summary>
    /// Copies source slices [<paramref name="sourceFirstLayer"/>, + <paramref name="layers"/>) into target slices
    /// from <paramref name="targetFirstLayer"/>, level 0, whole slices, in one blit pass. False, with nothing
    /// encoded, when the textures cannot be blitted between (different formats, extents or sample counts).
    /// </summary>
    public bool TryEncodeLayerSubsetCopy(
        IMetalSubmissionContext context, nint sourceTexture, uint sourceFirstLayer,
        nint targetTexture, uint targetFirstLayer, uint layers, uint width, uint height)
    {
        if (_disposed || context.CommandBuffer == 0 || sourceTexture == 0 || targetTexture == 0 ||
            sourceTexture == targetTexture || layers == 0 || width == 0 || height == 0)
        {
            return false;
        }

        var blit = _tiler.BeginBlit(context);
        var size = new MtlSize { Width = width, Height = height, Depth = 1 };
        for (var layer = 0u; layer < layers; layer++)
        {
            MetalNative.SendCopyTextureSlice(
                blit, SelCopyTextureSlice, sourceTexture, sourceFirstLayer + layer, 0, default, size,
                targetTexture, targetFirstLayer + layer, 0, default);
        }

        MetalTiler.EndEncoding(context, blit);
        return true;
    }
}
