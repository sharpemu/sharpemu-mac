// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Images;

namespace SharpEmu.Gpu.Metal.V2;

internal sealed partial class MetalImageTransfer
{
    public bool HasDenseAliasCoverage(in ImageDescription description) =>
        !description.IsDepth && !description.IsBlock && !description.HasStencil &&
        description.Samples == 1 && description.Resources.Levels == 1 &&
        TryDenseAliasSurface(description, PlanFor(description), out _);

    public bool CanDiscardAliasForOverwrite(in ImageDescription target, in ImageDescription source)
    {
        if (HasDenseAliasCoverage(target)) return true;
        if (target.IsDepth || source.IsDepth || target.HasStencil || source.HasStencil ||
            target.IsVolume || source.IsVolume || target.IsBlock || source.IsBlock ||
            target.Resources.Levels != 1 || source.Resources.Levels != 1 ||
            target.TransferLayers != 1 || source.TransferLayers != 1 ||
            target.BytesPerBlock != source.BytesPerBlock || target.Samples != 1 || source.Samples != 1)
            return false;
        if (source.Data.Address == target.Data.Address && source.Data.Size == target.Data.Size &&
            SameTransferGrid(source, target) && source.TileMode == target.TileMode && source.Pitch == target.Pitch)
            return true;
        var from = PlanFor(source);
        var to = PlanFor(target);
        if (from.Tiles.Count != 1 || to.Tiles.Count != 1 || !from.Tiled || !to.Tiled) return false;
        var a = from.Tiles[0];
        var b = to.Tiles[0];
        if (a.Tail || b.Tail || a.TiledOffset != 0 || b.TiledOffset != 0 ||
            a.SurfaceZ != 0 || b.SurfaceZ != 0 || a.Kind != b.Kind || from.ElementBytes != to.ElementBytes ||
            !TileGeometry.TryGetBlockLayout(a.Kind, from.ElementBytes, out var block) || block.BlockDepth != 1 ||
            source.Data.Address < target.Data.Address) return false;
        var offset = source.Data.Address - target.Data.Address;
        if (offset % block.BlockSize != 0) return false;
        var sourceColumns = ((a.TiledWidth != 0 ? a.TiledWidth : a.Pitch) + block.BlockWidth - 1) / block.BlockWidth;
        var targetColumns = ((b.TiledWidth != 0 ? b.TiledWidth : b.Pitch) + block.BlockWidth - 1) / block.BlockWidth;
        if (sourceColumns == 0 || targetColumns == 0) return false;
        // A full destination block kills every byte a smaller/padded source block
        // could own. Padding and incomplete destination edge blocks never qualify.
        for (uint y = 0; y < a.Height; y += block.BlockHeight)
        for (uint x = 0; x < a.Width; x += block.BlockWidth)
        {
            var index = offset / block.BlockSize + (ulong)(y / block.BlockHeight) * sourceColumns + x / block.BlockWidth;
            var tx = (index % targetColumns) * block.BlockWidth;
            var ty = (index / targetColumns) * block.BlockHeight;
            if (tx + block.BlockWidth > b.Width || ty + block.BlockHeight > b.Height) return false;
        }
        return true;
    }

    public bool TryEncodeFusedAlias(IMetalSubmissionContext context,
        in ImageDescription sourceDescription, nint sourceTexture,
        in ImageDescription targetDescription, nint targetTexture, out string refusal)
    {
        refusal = "shape";
        if (_disposed || context.CommandBuffer == 0 || sourceTexture == 0 || targetTexture == 0 ||
            sourceTexture == targetTexture || sourceDescription.IsDepth || targetDescription.IsDepth ||
            sourceDescription.HasStencil || targetDescription.HasStencil ||
            sourceDescription.IsBlock || targetDescription.IsBlock ||
            sourceDescription.IsVolume || targetDescription.IsVolume ||
            sourceDescription.Samples != 1 || targetDescription.Samples != 1 ||
            sourceDescription.Data.Size != targetDescription.Data.Size ||
            sourceDescription.Resources.Levels != 1 || targetDescription.Resources.Levels != 1)
            return false;

        var source = PlanFor(sourceDescription);
        var target = PlanFor(targetDescription);
        // Matching texel mappings can use a blit, including compression-friendly render
        // targets: do not add shader-write/view usage just to make a transfer cheaper.
        if (!source.Swap && !target.Swap && source.ElementBytes == target.ElementBytes &&
            SameTransferGrid(sourceDescription, targetDescription) &&
            sourceDescription.TileMode == targetDescription.TileMode &&
            sourceDescription.Pitch == targetDescription.Pitch &&
            sourceDescription.MipLayout[0].Equals(targetDescription.MipLayout[0]))
        {
            if (source.Format == target.Format)
            {
                EncodeWholeTextureCopy(context, sourceTexture, targetTexture);
                refusal = "none";
                return true;
            }
            // Reinterpreting views make a bitwise blit possible even for padded images;
            // both sides address the same guest texel, so padding is never observed.
            var fromView = AliasView(sourceTexture, source.Format, source.ElementBytes, writing: false, sourceDescription.TransferLayers);
            if (fromView != 0)
            {
                try
                {
                    var toView = AliasView(targetTexture, target.Format, target.ElementBytes, writing: false, targetDescription.TransferLayers);
                    if (toView != 0)
                    {
                        try { EncodeWholeTextureCopy(context, fromView, toView); }
                        finally { MetalNative.SendVoid(toView, SelReleaseView); }
                        refusal = "none";
                        return true;
                    }
                }
                finally { MetalNative.SendVoid(fromView, SelReleaseView); }
            }
        }

        refusal = "padding-or-subresources";
        if (!TryDenseAliasSurface(sourceDescription, source, out var from) ||
            !TryDenseAliasSurface(targetDescription, target, out var to)) return false;
        refusal = "format-or-usage";
        var sourceView = AliasView(sourceTexture, source.Format, source.ElementBytes, writing: false);
        if (sourceView == 0) return false;
        try
        {
            var targetView = AliasView(targetTexture, target.Format, target.ElementBytes, writing: true);
            if (targetView == 0) return false;
            try
            {
                var compute = _tiler.BeginCompute(context);
                _tiler.EncodeAlias(compute, sourceView, targetView, from, to);
                MetalTiler.EndEncoding(compute);
                refusal = "none";
                return true;
            }
            finally { MetalNative.SendVoid(targetView, SelReleaseView); }
        }
        finally { MetalNative.SendVoid(sourceView, SelReleaseView); }
    }

    private static bool TryDenseAliasSurface(in ImageDescription description, TransferPlan plan,
        out MetalAliasSurface surface)
    {
        surface = default;
        var width = description.Extent.Width;
        var height = description.Extent.Height;
        if (plan.Swap || plan.Volume || plan.DepthStencil || description.TransferLayers != 1 ||
            plan.ElementBytes is not (4 or 8 or 16) || width == 0 || height == 0 ||
            plan.TexelWidth != 1 || plan.TexelHeight != 1 ||
            description.Data.Size != (ulong)width * height * plan.ElementBytes ||
            description.Data.Size > uint.MaxValue || description.Pitch != width ||
            description.MipLayout[0].Offset != 0) return false;
        TileBlockKind? kind = null;
        if (plan.Tiled)
        {
            if (plan.Tiles.Count != 1) return false;
            var tile = plan.Tiles[0];
            if (tile.Tail || tile.BaseY != 0 || tile.SurfaceZ != 0 || tile.TiledOffset != 0 ||
                tile.Width != width || tile.Height != height || tile.Depth != 1 ||
                tile.TiledSize != description.Data.Size ||
                (tile.TiledWidth != 0 && tile.TiledWidth != width) ||
                (tile.TiledHeight != 0 && tile.TiledHeight != height) ||
                !TileGeometry.TryGetBlockLayout(tile.Kind, plan.ElementBytes, out var block) ||
                block.BlockDepth != 1 || width % block.BlockWidth != 0 || height % block.BlockHeight != 0)
                return false;
            kind = tile.Kind;
        }
        surface = new MetalAliasSurface(width, height, plan.ElementBytes, kind);
        return true;
    }

    private static nint AliasView(nint texture, MtlPixelFormat format, uint bytes, bool writing, uint layers = 1)
    {
        var raw = TextureTileViewFormat(format);
        var usage = (MtlTextureUsage)(uint)MetalV2Native.SendValue(texture, SelUsage);
        var needed = writing ? MtlTextureUsage.ShaderWrite : MtlTextureUsage.ShaderRead;
        var type = (MtlTextureType)(uint)MetalV2Native.SendValue(texture, SelTextureType);
        if (raw == MtlPixelFormat.Invalid || ViewElementBytes(raw) != bytes ||
            (usage & needed) == 0 || (raw != format && (usage & MtlTextureUsage.PixelFormatView) == 0) ||
            type is not (MtlTextureType.Type2D or MtlTextureType.Type2DArray)) return 0;
        return MetalV2Native.SendNewTextureView(texture, SelNewTextureView, (nuint)raw,
            (nuint)MtlTextureType.Type2DArray,
            new NsRange { Location = 0, Length = 1 }, new NsRange { Location = 0, Length = layers });
    }
}
