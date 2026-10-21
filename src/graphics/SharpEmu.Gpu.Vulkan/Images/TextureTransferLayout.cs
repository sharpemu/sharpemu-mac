// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Scheduling;
using Silk.NET.Vulkan;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Gpu.Vulkan.Scheduling;

namespace SharpEmu.Gpu.Vulkan.Images;

// One tiler dispatch: a linear span, a tiled span and the surface window between them.
public struct TileTransfer
{
    public TileBlockKind Kind;
    public uint BytesPerElement;
    public ulong LinearOffset;
    public ulong LinearSize;
    public ulong TiledOffset;
    public ulong TiledSize;
    public ulong LinearSliceStride;
    public uint Width;
    public uint Height;
    public uint Depth;
    public uint Pitch;
    public uint TailX;
    public uint TailY;
    public bool Tail;
    public uint TiledWidth;
    public uint TiledHeight;
    public uint SurfaceZ;
}

public struct TransferMipLayout
{
    public ulong Offset;
    public ulong Size;
    public uint RowLength;
    public uint ImageHeight;
}

// The linear staging layout of a guest texture and its tiled surface layout.
public sealed class TextureTransferLayout
{
    public uint Pitch;
    public ulong SliceStride;
    public ulong SourceSliceStride;
    public TiledSurfaceLayout Surface = new();
    public TileElementLayout Element;
    public readonly TransferMipLayout[] Mips = new TransferMipLayout[TiledSurfaceLayout.MaxLevels];

    private static uint LevelDepth(uint depth, uint level, bool volume) => volume ? Math.Max(depth >> (int)level, 1) : depth;

    private static int RegionCount(uint depth, uint levels, bool volume)
    {
        var count = 0;
        for (uint level = 0; level < levels; level++)
        {
            count += (int)LevelDepth(depth, level, volume);
        }

        return count;
    }

    private static ulong SliceStrideOf(TransferMipLayout[] mips, uint levels, ulong totalSize, uint depth)
    {
        ulong stride = 0;
        for (uint levelIndex = 0; levelIndex < levels; levelIndex++)
        {
            stride = Math.Max(stride, mips[levelIndex].Offset + mips[levelIndex].Size);
        }

        if (depth > 1 && totalSize != 0 && totalSize % depth == 0)
        {
            var guestStride = totalSize / depth;
            if (guestStride >= stride)
            {
                stride = guestStride;
            }
        }

        return stride;
    }

    private static ulong LinearLevelSize(in TileElementLayout element, uint pitch, uint height)
    {
        var elementsWide = Math.Max((pitch + element.TexelWidth - 1) / element.TexelWidth, 1);
        var elementsHigh = Math.Max((height + element.TexelHeight - 1) / element.TexelHeight, 1);
        return (ulong)elementsWide * elementsHigh * element.Bytes;
    }

    private static ulong SetLinearLevels(TransferMipLayout[] mips, in TileElementLayout element, uint height, uint levels, uint basePitch)
    {
        ulong offset = 0;
        var pitch = basePitch;
        var h = height;
        for (uint levelIndex = 0; levelIndex < levels; levelIndex++)
        {
            var size = LinearLevelSize(element, pitch, h);
            mips[levelIndex].Size = size;
            mips[levelIndex].Offset = offset;
            offset += size;
            if (pitch > 1)
            {
                pitch /= 2;
            }

            if (h > 1)
            {
                h /= 2;
            }
        }

        return offset;
    }

    public static TextureTransferLayout Compute(
        GuestPixelFormat format,
        uint width,
        uint height,
        uint levels,
        uint depth,
        GuestTileMode tile,
        ulong uploadSize,
        bool allowDepthTile,
        bool volume,
        string owner)
    {
        var layout = new TextureTransferLayout();
        var description = new TiledSurfaceDescription(
            format,
            tile,
            volume ? TileSurfaceDimension.Volume3D : TileSurfaceDimension.Flat2D,
            width,
            height,
            volume ? depth : 1,
            levels,
            volume ? 1 : depth);
        layout.Surface.Description = description;
        var element = default(TileElementLayout);

        if (format == GuestPixelFormat.Invalid)
        {
            throw SubmissionScheduler.Fatal($"{owner}: the texture upload format is unknown: format=0 tile={(uint)tile} size={uploadSize} extent={width}x{height} levels={levels}.");
        }

        switch (tile)
        {
            case GuestTileMode.Linear:
                if (!TileGeometry.TryGetElementLayout(format, out element))
                {
                    throw SubmissionScheduler.Fatal($"{owner}: the linear texture format is not supported: format={(uint)format}.");
                }

                break;
            case GuestTileMode.Depth when !allowDepthTile:
                throw UnsupportedTiledUpload(owner, format, tile, uploadSize, width, height, levels);
            default:
                if (!TileGeometry.TryGetTiledTextureLayout(description, out var surface))
                {
                    throw UnsupportedTiledUpload(owner, format, tile, uploadSize, width, height, levels);
                }

                layout.Surface = surface;
                break;
        }

        if (tile != GuestTileMode.Linear)
        {
            var texture = layout.Surface.Texture;
            element = new TileElementLayout(texture.Block.BytesPerElement, texture.TexelWidth, texture.TexelHeight);
        }

        layout.Pitch = TileGeometry.TexturePitch(format, width, tile);
        if (tile == GuestTileMode.Linear)
        {
            var (levelSpans, paddedSizes) = MipLayoutScratch.Rent();
            TileGeometry.TryGetTextureSize(format, width, height, levels, tile, out _, levelSpans, paddedSizes);
            for (uint level = 0; level < levels; level++)
            {
                layout.Mips[level] = new TransferMipLayout
                {
                    Offset = levelSpans[level].Offset,
                    Size = levelSpans[level].Size,
                    RowLength = paddedSizes[level].Width,
                    ImageHeight = paddedSizes[level].Height,
                };
            }
        }
        else if (volume)
        {
            layout.SliceStride = SetLinearLevels(layout.Mips, element, height, levels, width);
        }
        else
        {
            layout.SourceSliceStride = layout.Surface.BlockSliceSize;
            if (depth > 1 && uploadSize != 0 && uploadSize % depth == 0)
            {
                layout.SourceSliceStride = Math.Max(layout.SourceSliceStride, uploadSize / depth);
            }

            SetLinearLevels(layout.Mips, element, height, levels, layout.Pitch);
        }

        if (!volume || tile == GuestTileMode.Linear)
        {
            layout.SliceStride = SliceStrideOf(layout.Mips, levels, uploadSize, depth);
        }

        layout.Element = element;
        return layout;
    }

    // The last byte a copy reaches in its buffer, plus one, following Vulkan's buffer addressing: rows advance
    // by BufferRowLength and slices by BufferImageHeight.
    public static ulong CopyFootprint(IReadOnlyList<BufferImageCopy> regions, TileElementLayout element)
    {
        static ulong Blocks(uint texels, uint block) => (texels + (ulong)block - 1) / block;
        ulong end = 0;
        foreach (var region in regions)
        {
            var extent = region.ImageExtent;
            if (extent.Width == 0 || extent.Height == 0 || extent.Depth == 0 || region.ImageSubresource.LayerCount == 0)
            {
                continue;
            }

            var rowBlocks = Blocks(region.BufferRowLength == 0 ? extent.Width : region.BufferRowLength, element.TexelWidth);
            var sliceRows = Blocks(region.BufferImageHeight == 0 ? extent.Height : region.BufferImageHeight, element.TexelHeight);
            var slices = (ulong)extent.Depth * region.ImageSubresource.LayerCount;
            var lastBlock = ((slices - 1) * sliceRows + Blocks(extent.Height, element.TexelHeight) - 1) * rowBlocks +
                Blocks(extent.Width, element.TexelWidth);
            end = Math.Max(end, region.BufferOffset + lastBlock * element.Bytes);
        }

        return end;
    }

    private static Exception UnsupportedTiledUpload(string owner, GuestPixelFormat format, GuestTileMode tile, ulong uploadSize, uint width, uint height, uint levels) =>
        SubmissionScheduler.Fatal($"{owner}: the tiled texture upload is not supported: format={(uint)format} tile={(uint)tile} size={uploadSize} extent={width}x{height} levels={levels}.");

    public List<BufferImageCopy> BuildCopies()
    {
        var description = Surface.Description;
        var volume = description.Dimension == TileSurfaceDimension.Volume3D;
        var depth = volume ? description.Depth : description.Layers;
        var mipWidth = description.Width;
        var mipHeight = description.Height;
        var mipPitch = volume && description.TileMode != GuestTileMode.Linear ? description.Width : Pitch;
        var linear = description.TileMode == GuestTileMode.Linear;

        var regions = new List<BufferImageCopy>(RegionCount(depth, description.Levels, volume));
        for (uint level = 0; level < description.Levels; level++)
        {
            if (Mips[level].Size == 0)
            {
                throw SubmissionScheduler.Fatal($"A mip level of the transfer layout has no size: level={level}.");
            }

            var mipDepth = LevelDepth(depth, level, volume);
            for (uint depthSlice = 0; depthSlice < mipDepth; depthSlice++)
            {
                var region = new BufferImageCopy
                {
                    BufferOffset = Mips[level].Offset + depthSlice * SliceStride,
                    ImageSubresource = new ImageSubresourceLayers(ImageAspectFlags.ColorBit, level, volume ? 0 : depthSlice, 1),
                    ImageOffset = new Offset3D(0, 0, volume ? (int)depthSlice : 0),
                    ImageExtent = new Extent3D(mipWidth, mipHeight, 1),
                };
                if (linear)
                {
                    region.BufferRowLength = Mips[level].RowLength;
                    region.BufferImageHeight = Mips[level].ImageHeight;
                }
                else
                {
                    var texelWidth = Surface.Texture.TexelWidth;
                    static uint Align(uint value, uint block) => (value + block - 1) / block * block;
                    var pitch = Align(mipPitch, texelWidth);
                    region.BufferRowLength = pitch > Align(mipWidth, texelWidth) ? pitch : 0;
                }

                regions.Add(region);
            }

            if (mipWidth > 1)
            {
                mipWidth /= 2;
            }

            if (mipHeight > 1)
            {
                mipHeight /= 2;
            }

            if (mipPitch > 1)
            {
                mipPitch /= 2;
            }
        }

        return regions;
    }

    private static bool TrySetSpan(ulong offset, ulong length, ulong capacity, out ulong size)
    {
        size = 0;
        if (offset > capacity || length > capacity - offset)
        {
            return false;
        }

        size = length;
        return true;
    }

    // One tile transfer per copy region, or false when the layout cannot be tiled on the GPU.
    public bool TryBuildTileTransfers(ulong tiledSize, List<BufferImageCopy> regions, uint levels, out List<TileTransfer> transfers)
    {
        transfers = null!;
        var description = Surface.Description;
        var volume = description.Dimension == TileSurfaceDimension.Volume3D;
        var depth = volume ? description.Depth : description.Layers;
        if (tiledSize == 0 || levels == 0 || levels > TiledSurfaceLayout.MaxLevels || depth == 0 ||
            regions.Count != RegionCount(depth, levels, volume) || GuestPixelFormats.IsFmaskFormat(description.Format))
        {
            return false;
        }

        var surface = Surface;
        var texture = surface.Texture;
        var block = texture.Block;
        if (surface.Description.Levels < levels || !block.IsValid)
        {
            return false;
        }

        var result = new List<TileTransfer>(regions.Count);
        if (volume)
        {
            var regionBase = 0;
            for (uint level = 0; level < levels; level++)
            {
                var mipDepth = LevelDepth(depth, level, true);
                var tail = level >= surface.FirstTailLevel;
                var linearStride = SliceStride;
                ref var mip = ref surface.Mips[level];
                for (uint depthSlice = 0; depthSlice < mipDepth; depthSlice += block.BlockDepth)
                {
                    var copyDepth = Math.Min(block.BlockDepth, mipDepth - depthSlice);
                    var region = regions[regionBase + (int)depthSlice];
                    var pitch = region.BufferRowLength != 0 ? region.BufferRowLength : region.ImageExtent.Width;
                    var logicalHeight = region.BufferImageHeight != 0 ? region.BufferImageHeight : region.ImageExtent.Height;
                    var transfer = new TileTransfer
                    {
                        Kind = block.Kind,
                        BytesPerElement = block.BytesPerElement,
                        LinearOffset = region.BufferOffset,
                        TiledOffset = (ulong)(depthSlice / block.BlockDepth) * surface.BlockSliceSize + mip.Offset,
                    };
                    var linearSpan = (ulong)(copyDepth - 1) * linearStride + Mips[level].Size;
                    if (!TrySetSpan(transfer.LinearOffset, linearSpan, ulong.MaxValue, out transfer.LinearSize) ||
                        !TrySetSpan(transfer.TiledOffset, mip.Size, tiledSize, out transfer.TiledSize))
                    {
                        return false;
                    }

                    transfer.LinearSliceStride = linearStride;
                    transfer.Width = Math.Max((region.ImageExtent.Width + texture.TexelWidth - 1) / texture.TexelWidth, 1);
                    transfer.Height = Math.Max((logicalHeight + texture.TexelHeight - 1) / texture.TexelHeight, 1);
                    transfer.Depth = copyDepth;
                    transfer.SurfaceZ = block.BlockDepth == 1 ? (uint)region.ImageOffset.Z : 0;
                    transfer.Pitch = Math.Max((pitch + texture.TexelWidth - 1) / texture.TexelWidth, 1);
                    transfer.TailX = tail ? mip.TailX : 0;
                    transfer.TailY = tail ? mip.TailY : 0;
                    transfer.Tail = tail;
                    transfer.TiledWidth = mip.PaddedWidth;
                    transfer.TiledHeight = mip.PaddedHeight;
                    result.Add(transfer);
                }

                regionBase += (int)mipDepth;
            }
        }
        else
        {
            var regionIndex = 0;
            for (uint level = 0; level < levels; level++)
            {
                var levelSize = Mips[level];
                ref var mip = ref surface.Mips[level];
                var tail = level >= surface.FirstTailLevel;
                var levelDepth = LevelDepth(depth, level, false);
                for (uint depthSlice = 0; depthSlice < levelDepth; depthSlice++)
                {
                    var region = regions[regionIndex++];
                    var pitch = region.BufferRowLength != 0 ? region.BufferRowLength : region.ImageExtent.Width;
                    var logicalHeight = region.BufferImageHeight != 0 ? region.BufferImageHeight : region.ImageExtent.Height;
                    var transfer = new TileTransfer
                    {
                        Kind = block.Kind,
                        BytesPerElement = block.BytesPerElement,
                        LinearOffset = region.BufferOffset,
                    };
                    var sourceStride = SourceSliceStride != 0 ? SourceSliceStride : surface.BlockSliceSize;
                    if (depthSlice > (ulong.MaxValue - mip.Offset) / sourceStride)
                    {
                        return false;
                    }

                    transfer.TiledOffset = mip.Offset + depthSlice * sourceStride;
                    if (!TrySetSpan(transfer.LinearOffset, levelSize.Size, ulong.MaxValue, out transfer.LinearSize) ||
                        !TrySetSpan(transfer.TiledOffset, mip.Size, tiledSize, out transfer.TiledSize))
                    {
                        return false;
                    }

                    transfer.Width = Math.Max((region.ImageExtent.Width + texture.TexelWidth - 1) / texture.TexelWidth, 1);
                    transfer.Height = Math.Max((logicalHeight + texture.TexelHeight - 1) / texture.TexelHeight, 1);
                    transfer.Depth = 1;
                    transfer.SurfaceZ = block.Kind is TileBlockKind.RenderTarget64KB or TileBlockKind.RenderTarget64KBGen5 or TileBlockKind.Depth64KB
                        ? region.ImageSubresource.BaseArrayLayer
                        : 0;
                    transfer.Pitch = Math.Max((pitch + texture.TexelWidth - 1) / texture.TexelWidth, 1);
                    transfer.Tail = tail;
                    transfer.TailX = tail ? mip.TailX : 0;
                    transfer.TailY = tail ? mip.TailY : 0;
                    transfer.TiledWidth = mip.PaddedWidth;
                    transfer.TiledHeight = mip.PaddedHeight;
                    result.Add(transfer);
                }
            }
        }

        if (result.Count == 0)
        {
            return false;
        }

        transfers = result;
        return true;
    }
}
