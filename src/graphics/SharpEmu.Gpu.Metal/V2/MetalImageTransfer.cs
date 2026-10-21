// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Scheduling;

namespace SharpEmu.Gpu.Metal.V2;

// MTLTextureType, only the members a guest image can need. Cube faces arrive as array
// layers because the MSL translator declares every cube binding Dim2DArray
// (Gen5MslArgumentLayout.TextureType), so a cube is allocated as a 2D array.
internal enum MtlTextureType : uint
{
    Type1D = 0,
    Type1DArray = 1,
    Type2D = 2,
    Type2DArray = 3,
    Type2DMultisample = 4,
    Type3D = 7,
    Type2DMultisampleArray = 8,
}

[Flags]
internal enum MtlTextureUsage : uint
{
    Unknown = 0,
    ShaderRead = 1,
    ShaderWrite = 2,
    RenderTarget = 4,
    PixelFormatView = 16,
    // macOS 14 and later. Metal allows a texture atomic only on a texture
    // created with this bit, and only on R32Uint, R32Sint or R64Uint.
    ShaderAtomic = 32,
}

[Flags]
internal enum MtlBlitOption : uint
{
    None = 0,
    DepthFromDepthStencil = 1,
    StencilFromDepthStencil = 2,
}

// Moves image contents between guest memory and Metal textures entirely on the GPU.
//
// Every tiled surface goes through the tiler compute kernels into a linear staging
// buffer and then through one MTLBlitCommandEncoder copy per subresource, which is the
// one path that also covers depth, stencil and block-compressed formats. A linear
// surface skips the compute pass whenever its guest offset and row pitch are something
// a buffer-to-texture blit can address, which for the guest's 256-byte-aligned linear
// pitches is the normal case.
//
// Called from the GPU worker thread only: the selection scratch lists and the plan cache
// are not synchronised, matching how the Vulkan GuestImageCache records transfers.
internal sealed partial class MetalImageTransfer : IMetalImageTransfer, IDisposable
{
    // Conservative blit alignment. Metal wants a buffer-to-texture offset and row pitch
    // that are multiples of the texel block size; four bytes on top of that keeps the
    // sub-dword formats off the slow path only when the guest layout is genuinely odd.
    private const uint BlitAlignment = 4;

    private const ulong StagingAlignment = 16;

    private static readonly nint SelCopyBufferToTexture = MetalNative.Selector(
        "copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:" +
        "toTexture:destinationSlice:destinationLevel:destinationOrigin:options:");

    private static readonly nint SelCopyTextureToBuffer = MetalNative.Selector(
        "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:" +
        "toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:options:");

    private static readonly nint SelCopyWholeTexture = MetalNative.Selector("copyFromTexture:toTexture:");

    // Everything about one description that changes the staged bytes. Two images that
    // agree on all of it share a plan, so a steady-state frame builds none.
    private readonly record struct PlanKey(
        GuestPixelFormat GuestFormat,
        GpuFormat PixelFormat,
        GuestTileMode TileMode,
        GuestImageType Type,
        uint Width,
        uint Height,
        uint Depth,
        uint Levels,
        uint Layers,
        uint Pitch,
        uint BytesPerBlock,
        ulong DataSize,
        ulong StencilSize,
        bool Bgra16);

    private sealed class TransferPlan
    {
        public MetalTextureLayout Layout = null!;
        public readonly List<MetalSubresourceCopy> Copies = [];
        public readonly List<MetalTileTransfer> Tiles = [];
        public bool Tiled;
        public bool Volume;
        public bool Swap;
        public uint ElementBytes;
        public uint TexelWidth = 1;
        public uint TexelHeight = 1;
        public MtlPixelFormat Format;
        public ulong LinearSize;

        // Depth and stencil images carry their own per-layer plane transfers instead of
        // a mip chain: a depth surface is one level with one 64 KiB block layout per layer.
        public bool DepthStencil;
        public uint Layers;
        public ulong SliceSize;
        public uint TransferBytes;
        public bool StencilPlane;
        public TileBlockKind BlockKind = TileBlockKind.Depth64KB;
        public uint StencilPitch;
        public ulong StencilSliceSize;
        public readonly List<MetalTileTransfer> StencilTiles = [];
    }

    private readonly nint _device;
    private readonly MetalTiler _tiler;
    private const int MaxPlans = 2048;
    private readonly Dictionary<PlanKey, TransferPlan> _plans = [];
    private readonly List<MetalSubresourceCopy> _copies = [];
    private readonly List<MetalTileTransfer> _tiles = [];
    private readonly List<MetalTileTransfer> _stencilTiles = [];
    private readonly List<ulong> _stagedOffsets = [];
    private bool _disposed;

    public MetalImageTransfer(nint device)
    {
        if (device == 0)
        {
            throw GpuFatal.Fatal("The Metal image transfer needs a device.");
        }

        _device = device;
        _tiler = new MetalTiler(device);
    }

    public MetalTiler Tiler => _tiler;

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _tiler.Dispose();
        _plans.Clear();
    }

    /// <summary>
    /// Hands one cached surface's contents to another that occupies the same guest bytes,
    /// without going through guest memory. Two surfaces that alias each other every frame
    /// otherwise cost a retile out to guest memory and a detile back in, and neither
    /// result is any different from what the device could have copied directly.
    ///
    /// The bytes travel through a linear staging buffer, because the one pair that
    /// matters most - a depth-stencil target and its single-channel float alias - has no
    /// blit between the textures themselves. That route is bit-exact: it is the same
    /// MTLBlitOptionDepthFromDepthStencil plane the tiler already writes guest memory
    /// from. False when the pair is not one this can carry, and the caller publishes
    /// through guest memory as before.
    /// </summary>
    public void EncodeWholeTextureCopy(IMetalSubmissionContext context, nint sourceTexture, nint targetTexture)
    {
        if (_disposed || context.CommandBuffer == 0 || sourceTexture == 0 || targetTexture == 0)
        {
            throw GpuFatal.Fatal(
                $"A whole-texture copy has nothing to copy with: source=0x{sourceTexture:X} target=0x{targetTexture:X}.");
        }

        var blit = _tiler.BeginBlit(context);
        MetalNative.SendVoid(blit, SelCopyWholeTexture, sourceTexture, targetTexture);
        MetalTiler.EndEncoding(blit);
    }

    public bool TryEncodeTextureToTexture(
        IMetalSubmissionContext context,
        in ImageDescription sourceDescription,
        nint sourceTexture,
        in ImageDescription targetDescription,
        nint targetTexture)
    {
        if (_disposed || context.CommandBuffer == 0 || sourceTexture == 0 || targetTexture == 0 ||
            sourceTexture == targetTexture)
        {
            return false;
        }

        var source = PlanFor(sourceDescription);
        var target = PlanFor(targetDescription);
        if (BlockAliasOnDevice &&
            (IsBlockWordAlias(sourceDescription, source, targetDescription, target) ||
             IsBlockWordAlias(targetDescription, target, sourceDescription, source)))
        {
            return TryEncodeBlockWordCopy(context, in sourceDescription, sourceTexture, in targetDescription, targetTexture);
        }

        // One plane of equally sized texels, one mip, the same grid, and no multisampling:
        // anything else is a reinterpretation this cannot claim to preserve.
        // Only the pairs that were measured to matter: a depth target read through a
        // single-channel float alias, and a 16-bit shadow map and its R16 alias in either
        // direction. Depth is the side with no blit to a colour texture, which is why the
        // pair costs a round trip through guest memory at all; every other pair either
        // already has a texture view or is not what anything spends its time on, and
        // claiming them without a device test behind each one is how a sampled surface
        // quietly comes back wrong.
        if (!SameTransferGrid(sourceDescription, targetDescription) ||
            !(IsDepth32ToColorAlias(sourceDescription, source, targetDescription, target) ||
              (ColorToDepthDeviceCopyEnabled &&
               IsDepth32ToColorAlias(targetDescription, target, sourceDescription, source)) ||
              (Depth16AliasOnDevice &&
               (IsDepth16ColorAlias(sourceDescription, source, targetDescription, target) ||
                IsDepth16ColorAlias(targetDescription, target, sourceDescription, source)))) ||
            // Texel (x, y) of each surface has to come from the same guest bytes, which
            // takes the same tile mode and the same pitch over the same footprint.
            sourceDescription.TileMode != targetDescription.TileMode ||
            sourceDescription.Pitch != targetDescription.Pitch ||
            sourceDescription.Data.Size != targetDescription.Data.Size ||
            sourceDescription.Resources.Levels != 1 || targetDescription.Resources.Levels != 1 ||
            // A colour side with a stencil plane is no single-channel alias; a D32S8 depth
            // side takes only its depth plane (DepthFromDepthStencil below).
            (targetDescription.HasStencil && !targetDescription.IsDepth) ||
            (sourceDescription.HasStencil && !sourceDescription.IsDepth))
        {
            return false;
        }

        var texelBytes = targetDescription.BytesPerBlock;
        var layers = targetDescription.TransferLayers;
        var rowBytes = (ulong)targetDescription.Extent.Width * texelBytes;
        var sliceBytes = rowBytes * targetDescription.Extent.Height;
        if (layers == 0 || rowBytes == 0 || sliceBytes == 0 ||
            sliceBytes > ulong.MaxValue / layers)
        {
            return false;
        }

        var staging = _tiler.Acquire(context, sliceBytes * layers);
        if (!staging.IsValid)
        {
            return false;
        }

        var size = new MtlSize
        {
            Width = targetDescription.Extent.Width,
            Height = targetDescription.Extent.Height,
            Depth = 1,
        };
        var readOption = MetalGuestFormats.HasStencilPlane(source.Format)
            ? MtlBlitOption.DepthFromDepthStencil
            : MtlBlitOption.None;
        var writeOption = MetalGuestFormats.HasStencilPlane(target.Format)
            ? MtlBlitOption.DepthFromDepthStencil
            : MtlBlitOption.None;
        var blit = _tiler.BeginBlit(context);
        for (uint layer = 0; layer < layers; layer++)
        {
            var offset = staging.Offset + (sliceBytes * layer);
            MetalNative.SendCopyTextureToBufferOptions(blit, SelCopyTextureToBuffer,
                sourceTexture, layer, 0, default, size,
                staging.Buffer, (nuint)offset, (nuint)rowBytes, (nuint)sliceBytes, (nuint)readOption);
            MetalNative.SendCopyBufferToTextureOptions(blit, SelCopyBufferToTexture,
                staging.Buffer, (nuint)offset, (nuint)rowBytes, (nuint)sliceBytes,
                size, targetTexture, layer, 0, default, (nuint)writeOption);
        }

        MetalTiler.EndEncoding(blit);
        return true;
    }

    /// <summary>
    /// Whether the depth image's own texture, sampled in its own format, reads texel for
    /// texel what a texture of the colour description over the same guest bytes would hold
    /// after an upload: a 32-bit float depth plane against R32 float, or a 16-bit unorm
    /// depth plane against R16 unorm. Both sides must take their texels from the same
    /// guest bytes untouched (no widening, no channel swap) in the same tile mode, pitch and
    /// footprint, the same pair <see cref="TryEncodeTextureToTexture"/> copies bit for bit.
    /// </summary>
    public bool SamplesAsDepthTwin(in ImageDescription depthDescription, in ImageDescription colorDescription)
    {
        var pairs = (depthDescription.PixelFormat, colorDescription.PixelFormat) is
            (GpuFormat.D32Sfloat or GpuFormat.D32SfloatS8Uint, GpuFormat.R32Sfloat) or
            (GpuFormat.D16Unorm, GpuFormat.R16Unorm);
        if (_disposed || !pairs || !depthDescription.IsDepth || colorDescription.IsDepth ||
            !SameTransferGrid(depthDescription, colorDescription) ||
            depthDescription.TileMode != colorDescription.TileMode ||
            depthDescription.Pitch != colorDescription.Pitch ||
            depthDescription.Data.Size != colorDescription.Data.Size ||
            depthDescription.BytesPerBlock != colorDescription.BytesPerBlock ||
            depthDescription.Resources != colorDescription.Resources ||
            depthDescription.Resources.Levels != 1 ||
            colorDescription.HasStencil || colorDescription.HasMetadata)
        {
            return false;
        }

        var depth = PlanFor(depthDescription);
        var color = PlanFor(colorDescription);
        return depth.TransferBytes == depthDescription.BytesPerBlock && !color.Swap;
    }

    // A single 32-bit channel alias handing back to its 32-bit depth plane: UE5's Nanite depth
    // export writes the scene depth through an R32 view of it, and the round trip through
    // guest memory left Black Myth: Wukong's base pass without its prepass depth (black
    // characters). SHARPEMU_METAL_V2_ALIAS_COLOR_TO_DEPTH=0 sends it through guest memory again.
    internal static bool ColorToDepthDeviceCopyEnabled =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_ALIAS_COLOR_TO_DEPTH") != "0";

    // An 8-bit view of a depth image's stencil plane into that image's stencil, through a
    // staging buffer: a combined depth-stencil texture takes stencil only from a buffer, with
    // MTLBlitOptionStencilFromDepthStencil. The texels are the stencil values whatever the
    // guest tiling, because both textures hold them detiled.
    public bool TryEncodeStencilFromColor(
        IMetalSubmissionContext context,
        in ImageDescription colorDescription,
        nint colorTexture,
        in ImageDescription depthDescription,
        nint depthTexture)
    {
        if (_disposed || context.CommandBuffer == 0 || colorTexture == 0 || depthTexture == 0 ||
            colorDescription.IsBlock || colorDescription.BytesPerBlock != 1 ||
            colorDescription.Extent.Width != depthDescription.Extent.Width ||
            colorDescription.Extent.Height != depthDescription.Extent.Height ||
            colorDescription.Resources.Levels != 1 || colorDescription.TransferLayers != 1 ||
            depthDescription.TransferLayers != 1 ||
            colorDescription.Samples != 1 || depthDescription.Samples != 1 ||
            !MetalGuestFormats.TryDecodeHostFormat(depthDescription.PixelFormat, out var depthFormat) ||
            !MetalGuestFormats.HasStencilPlane(depthFormat))
        {
            return false;
        }

        var width = depthDescription.Extent.Width;
        var height = depthDescription.Extent.Height;
        var rowBytes = ((ulong)width + 255) & ~255UL;
        var sliceBytes = rowBytes * height;
        var staging = _tiler.Acquire(context, sliceBytes);
        if (!staging.IsValid)
        {
            return false;
        }

        var blit = _tiler.BeginBlit(context);
        MetalNative.SendCopyTextureToBufferOptions(blit, SelCopyTextureToBuffer,
            colorTexture, 0, 0, default, new MtlSize { Width = width, Height = height, Depth = 1 },
            staging.Buffer, (nuint)staging.Offset, (nuint)rowBytes, (nuint)sliceBytes, (nuint)MtlBlitOption.None);
        MetalNative.SendCopyBufferToTextureOptions(blit, SelCopyBufferToTexture,
            staging.Buffer, (nuint)staging.Offset, (nuint)rowBytes, (nuint)sliceBytes,
            new MtlSize { Width = width, Height = height, Depth = 1 },
            depthTexture, 0, 0, default, (nuint)MtlBlitOption.StencilFromDepthStencil);
        MetalTiler.EndEncoding(blit);
        return true;
    }

    private static bool IsRawSingleChannel32(MtlPixelFormat format) =>
        format is MtlPixelFormat.R32Float or MtlPixelFormat.R32Uint or MtlPixelFormat.R32Sint;

    private static bool IsRawSingleChannel16(MtlPixelFormat format) =>
        format is MtlPixelFormat.R16Unorm or MtlPixelFormat.R16Uint or MtlPixelFormat.R16Sint;

    // SHARPEMU_METAL_V2_D16_ALIAS_DEVICE=0 sends a 16-bit depth plane and its R16 alias
    // through guest memory again, in both directions.
    private static readonly bool Depth16AliasDefault =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_D16_ALIAS_DEVICE") != "0";

    /// <summary>Whether a 16-bit depth plane and its R16 alias are handed over on the device.</summary>
    public bool Depth16AliasOnDevice { get; set; } = Depth16AliasDefault;

    // A 32-bit depth plane (D32, or the depth of D32S8) into a single 32-bit channel. The
    // other way is not claimed: nothing was measured spending time on it.
    private static bool IsDepth32ToColorAlias(
        in ImageDescription depthDescription, TransferPlan depth,
        in ImageDescription colorDescription, TransferPlan color) =>
        depthDescription.IsDepth && !colorDescription.IsDepth &&
        depth.TransferBytes == sizeof(float) &&
        // A depth surface the tiler widens or narrows on its way through guest memory
        // does not hold the guest's bits in its texels, so the alias reading those bits
        // cannot be handed the texture's.
        depth.TransferBytes == depthDescription.BytesPerBlock &&
        // The colour side uploads guest bytes into its texels untouched only for a single
        // 32-bit channel; every other format reorders or reinterprets them.
        IsRawSingleChannel32(color.Format) && colorDescription.BytesPerBlock == sizeof(float) &&
        !color.Swap;

    // A 16-bit depth plane and a single 16-bit channel over the same bytes, either of them
    // the source. SILENT HILL renders its shadow maps as D16 and also reads and writes them
    // through an R16 alias, and the round trip moved the whole map through guest memory on
    // every switch: 32 MiB out and 32 MiB back in for the 4096x4096 map, hundreds of times
    // a minute. Both sides take their texels from the guest bytes untouched - Depth16Unorm
    // is never widened here, an R16 upload swaps nothing - and both are addressed by the
    // 64 KiB depth block of their 2-byte elements, so texel (x, y) of one is texel (x, y)
    // of the other (MetalImageTransferTests.Depth16Alias proves it against the round trip,
    // both ways). Any other tile mode is left to the round trip, untested. The depth side
    // carries no stencil plane, so a copy into it replaces everything the texture holds,
    // and a DCC colour side is refused: its guest bytes mean nothing without the metadata.
    private static bool IsDepth16ColorAlias(
        in ImageDescription depthDescription, TransferPlan depth,
        in ImageDescription colorDescription, TransferPlan color) =>
        depthDescription.IsDepth && !colorDescription.IsDepth &&
        depthDescription.PixelFormat == GpuFormat.D16Unorm && !depthDescription.HasStencil &&
        depth.TransferBytes == sizeof(ushort) && depthDescription.BytesPerBlock == sizeof(ushort) &&
        IsRawSingleChannel16(color.Format) && colorDescription.BytesPerBlock == sizeof(ushort) &&
        !color.Swap && !colorDescription.HasMetadata &&
        depthDescription.TileMode == GuestTileMode.Depth && colorDescription.TileMode == GuestTileMode.Depth;

    // SHARPEMU_METAL_V2_BLOCK_ALIAS_DEVICE=0 sends a block-compressed surface and the
    // integer view of its blocks through guest memory again.
    private static readonly bool BlockAliasDefault =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_BLOCK_ALIAS_DEVICE") != "0";

    /// <summary>Whether a block-compressed surface and the integer view of its blocks are handed over on the device.</summary>
    public bool BlockAliasOnDevice { get; set; } = BlockAliasDefault;

    // A block-compressed surface and an integer view of its blocks over the same bytes, either
    // of them the source: UE5 writes its virtual texture pool's BC blocks through an RGBA32Uint
    // view (SILENT HILL: a 7800x7800 BC3 pool, 60 MiB, out to guest memory and back in on
    // every frame that streamed a tile). Block (x, y) of one is texel (x, y) of the other, the
    // same 8 or 16 bytes untouched on both sides - a BC texture holds the guest's blocks, and
    // a two- or four-channel 32-bit integer upload converts nothing - so the blocks are copied
    // across in a linear staging buffer and neither tiler is involved.
    private static bool IsBlockWordAlias(
        in ImageDescription blockDescription, TransferPlan block,
        in ImageDescription wordDescription, TransferPlan word)
    {
        var bytes = blockDescription.BytesPerBlock;
        return blockDescription.IsBlock && !wordDescription.IsBlock &&
            BlockBytes(block.Format) == bytes && wordDescription.BytesPerBlock == bytes &&
            WordBytes(word.Format) == bytes &&
            !block.Swap && !word.Swap && !block.DepthStencil && !word.DepthStencil &&
            !blockDescription.HasMetadata && !wordDescription.HasMetadata &&
            blockDescription.Extent.Width == wordDescription.Extent.Width * 4 &&
            blockDescription.Extent.Height == wordDescription.Extent.Height * 4 &&
            blockDescription.Extent.Depth == 1 && wordDescription.Extent.Depth == 1 &&
            blockDescription.TransferLayers == wordDescription.TransferLayers &&
            blockDescription.Samples == 1 && wordDescription.Samples == 1 &&
            !blockDescription.IsVolume && !wordDescription.IsVolume &&
            blockDescription.TileMode == wordDescription.TileMode &&
            blockDescription.BlockExtent.Width == wordDescription.Pitch &&
            blockDescription.Data.Size == wordDescription.Data.Size &&
            blockDescription.Resources.Levels == 1 && wordDescription.Resources.Levels == 1;
    }

    private static uint BlockBytes(MtlPixelFormat format) => format switch
    {
        MtlPixelFormat.Bc1Rgba or MtlPixelFormat.Bc1RgbaSrgb or
            MtlPixelFormat.Bc4RUnorm or MtlPixelFormat.Bc4RSnorm => 8u,
        MtlPixelFormat.Bc2Rgba or MtlPixelFormat.Bc2RgbaSrgb or
            MtlPixelFormat.Bc3Rgba or MtlPixelFormat.Bc3RgbaSrgb or
            MtlPixelFormat.Bc5RgUnorm or MtlPixelFormat.Bc5RgSnorm or
            MtlPixelFormat.Bc6HRgbFloat or MtlPixelFormat.Bc6HRgbUfloat or
            MtlPixelFormat.Bc7RgbaUnorm or MtlPixelFormat.Bc7RgbaUnormSrgb => 16u,
        _ => 0u,
    };

    private static uint WordBytes(MtlPixelFormat format) => format switch
    {
        MtlPixelFormat.Rg32Uint or MtlPixelFormat.Rg32Sint => 8u,
        MtlPixelFormat.Rgba32Uint or MtlPixelFormat.Rgba32Sint => 16u,
        _ => 0u,
    };

    // A row of blocks is a row of words: the same bytes per row and per layer on both sides,
    // each side copied at its own texel extent.
    private bool TryEncodeBlockWordCopy(
        IMetalSubmissionContext context,
        in ImageDescription sourceDescription,
        nint sourceTexture,
        in ImageDescription targetDescription,
        nint targetTexture)
    {
        var words = sourceDescription.IsBlock ? targetDescription : sourceDescription;
        var rowBytes = (ulong)words.Extent.Width * words.BytesPerBlock;
        var sliceBytes = rowBytes * words.Extent.Height;
        var layers = words.TransferLayers;
        var staging = _tiler.Acquire(context, sliceBytes * layers);
        if (!staging.IsValid)
        {
            return false;
        }

        var sourceSize = new MtlSize { Width = sourceDescription.Extent.Width, Height = sourceDescription.Extent.Height, Depth = 1 };
        var targetSize = new MtlSize { Width = targetDescription.Extent.Width, Height = targetDescription.Extent.Height, Depth = 1 };
        var blit = _tiler.BeginBlit(context);
        for (uint layer = 0; layer < layers; layer++)
        {
            var offset = staging.Offset + (sliceBytes * layer);
            MetalNative.SendCopyTextureToBufferOptions(blit, SelCopyTextureToBuffer,
                sourceTexture, layer, 0, default, sourceSize,
                staging.Buffer, (nuint)offset, (nuint)rowBytes, (nuint)sliceBytes, (nuint)MtlBlitOption.None);
            MetalNative.SendCopyBufferToTextureOptions(blit, SelCopyBufferToTexture,
                staging.Buffer, (nuint)offset, (nuint)rowBytes, (nuint)sliceBytes,
                targetSize, targetTexture, layer, 0, default, (nuint)MtlBlitOption.None);
        }

        MetalTiler.EndEncoding(blit);
        return true;
    }

    // The two surfaces cover the same texel grid, so a copy between them is a copy and
    // not a resample.
    private static bool SameTransferGrid(in ImageDescription left, in ImageDescription right) =>
        left.Extent.Width == right.Extent.Width &&
        left.Extent.Height == right.Extent.Height &&
        left.Extent.Depth == right.Extent.Depth &&
        left.TransferLayers == right.TransferLayers &&
        left.Samples == 1 && right.Samples == 1 &&
        !left.IsVolume && !right.IsVolume;

    public MetalTextureShape ShapeFor(in ImageDescription description, uint firstLevel = 0)
    {
        if (!MetalGuestFormats.TryDecodeHostFormat(description.PixelFormat, out var format))
        {
            throw GpuFatal.Fatal(
                $"No Metal pixel format carries this image: hostFormat={(int)description.PixelFormat} " +
                $"guestFormat={(uint)description.GuestFormat} bytesPerBlock={description.BytesPerBlock}.");
        }

        if (firstLevel != 0 &&
            (firstLevel >= description.Resources.Levels || description.IsVolume || description.Samples != 1))
        {
            throw GpuFatal.Fatal(
                $"A texture cannot start at this guest level: firstLevel={firstLevel} " +
                $"levels={description.Resources.Levels} volume={description.IsVolume} samples={description.Samples}.");
        }

        // Metal sizes level i of a texture as max(1, width >> i), as the guest does, so a
        // texture from firstLevel has exactly the extents of the chain's levels from there.
        return new MetalTextureShape(
            (ulong)format,
            (ulong)TextureTypeOf(description),
            Math.Max(description.Extent.Width >> (int)firstLevel, 1),
            Math.Max(description.Extent.Height >> (int)firstLevel, 1),
            description.IsVolume ? description.Extent.Depth : 1,
            description.Resources.Levels - firstLevel,
            description.IsVolume ? 1 : description.Resources.Layers,
            description.Samples,
            (ulong)UsageOf(description, format));
    }

    public (ulong Start, ulong End) LevelBytes(in ImageDescription description, uint firstLevel)
    {
        if (firstLevel == 0 || ImageDescription.IsEmptyRange(description.Data))
        {
            return (0, description.Data.Size);
        }

        // From the plan an upload selects from, so the run holds every byte the selected
        // tiles or copies read. The guest chain keeps its full-chain offsets whatever the
        // texture holds: the T# base address is mip 0's chain, and a coarser level sits
        // where the full layout puts it.
        var plan = PlanFor(description);
        var start = ulong.MaxValue;
        var end = 0UL;
        if (plan.Tiled)
        {
            foreach (var transfer in plan.Tiles)
            {
                if (transfer.Level >= firstLevel)
                {
                    start = Math.Min(start, transfer.TiledOffset);
                    end = Math.Max(end, transfer.TiledOffset + transfer.TiledSize);
                }
            }
        }
        else if (!plan.DepthStencil)
        {
            foreach (var copy in plan.Copies)
            {
                if (copy.Level >= firstLevel)
                {
                    start = Math.Min(start, copy.BufferOffset);
                    end = Math.Max(end, CopyEnd(plan, copy));
                }
            }
        }

        if (start == ulong.MaxValue || start >= description.Data.Size)
        {
            return (0, description.Data.Size);
        }

        // A subresource's tiled extent may run past the bytes the description claims; the
        // piece that reaches the image's end owns whatever is there (see Select).
        return (start, Math.Min(end, description.Data.Size));
    }

    private static MtlTextureType TextureTypeOf(in ImageDescription description)
    {
        if (description.IsVolume)
        {
            return MtlTextureType.Type3D;
        }

        var arrayed = description.Resources.Layers > 1;
        if (description.Type == GuestImageType.Color1D)
        {
            return arrayed ? MtlTextureType.Type1DArray : MtlTextureType.Type1D;
        }

        if (description.Samples > 1)
        {
            return arrayed ? MtlTextureType.Type2DMultisampleArray : MtlTextureType.Type2DMultisample;
        }

        return arrayed ? MtlTextureType.Type2DArray : MtlTextureType.Type2D;
    }

    internal static MtlTextureUsage UsageOf(in ImageDescription description, MtlPixelFormat format)
    {
        if (MetalGuestFormats.IsDepthStencil(format))
        {
            return MtlTextureUsage.ShaderRead | MtlTextureUsage.RenderTarget;
        }

        // A compressed texture is only ever sampled: Metal rejects a render-target or
        // writable compressed texture. The one reinterpretation the cache makes of it is
        // the sRGB/linear twin (GetOrCreateView): a guest that reads one block memory
        // through both an sRGB and a linear descriptor (Unreal's virtual texture pools)
        // otherwise gets whichever encoding created the image first. On Black Myth: Wukong
        // that turned the armour's base colour pool linear in about half the runs, which
        // washed the gold out to silver.
        if (description.IsBlock)
        {
            return MetalGuestFormats.HasSrgbTwin(format)
                ? MtlTextureUsage.ShaderRead | MtlTextureUsage.PixelFormatView
                : MtlTextureUsage.ShaderRead;
        }

        var usage = MtlTextureUsage.ShaderRead | MtlTextureUsage.RenderTarget | MtlTextureUsage.PixelFormatView;
        if (description.Samples == 1 && !MetalGuestFormats.IsSrgb(format))
        {
            usage |= MtlTextureUsage.ShaderWrite;

            // A shader that runs IMAGE_ATOMIC_* on this image needs the atomic
            // usage bit, and Metal only accepts it on the single-channel 32-bit
            // integer formats and on RG32Uint, which holds the 64-bit texels of
            // a 64-bit atomic. Those are exactly the formats a guest atomic
            // image has (BindingLayout gives an atomic binding to a uint image
            // only), so the bit goes on by format rather than by waiting for a
            // shader to be compiled against the texture the cache already made.
            if (format is MtlPixelFormat.R32Uint or MtlPixelFormat.R32Sint or MtlPixelFormat.Rg32Uint)
            {
                usage |= MtlTextureUsage.ShaderAtomic;
            }
        }

        return usage;
    }

    public void EncodeGuestToTexture(IMetalSubmissionContext context, in ImageDescription description,
        SubresourceRange range, MetalGuestRange source, MetalGuestRange stencilSource, nint texture,
        ulong guestBase, ulong pieceSize, uint textureFirstLevel = 0)
    {
        EncodeGuestToTextureCore(
            context, description, range, source, stencilSource, texture, guestBase, pieceSize, textureFirstLevel);
        if (textureFirstLevel == 0)
        {
            MetalBlockCubeDump.Capture(context, _tiler, description, source, texture, guestBase, pieceSize);
        }
    }

    private void EncodeGuestToTextureCore(IMetalSubmissionContext context, in ImageDescription description,
        SubresourceRange range, MetalGuestRange source, MetalGuestRange stencilSource, nint texture,
        ulong guestBase, ulong pieceSize, uint textureFirstLevel)
    {
        // Per-sample tiled layouts are not decoded on either backend; the Vulkan cache
        // leaves a multisampled image untouched too.
        if (!Ready(context, description, source, texture, guestBase, pieceSize))
        {
            return;
        }

        var plan = PlanFor(description);
        if (plan.DepthStencil)
        {
            if (textureFirstLevel != 0)
            {
                throw GpuFatal.Fatal(
                    $"A depth image has no texture from a later level: firstLevel={textureFirstLevel} " +
                    $"address=0x{description.Data.Address:X16}.");
            }

            EncodeDepthUpload(context, description, plan, range, source, stencilSource, texture, guestBase, pieceSize);
            return;
        }

        Select(plan, description, range, guestBase, pieceSize, textureFirstLevel);
        if (_copies.Count == 0)
        {
            return;
        }

        if (TryEncodeTextureTiles(context, plan, texture, description.Data.Address + guestBase,
                new MetalBufferSpan(source.Buffer, source.Offset, TiledCapacity(source)), MetalTileDirection.Detile))
        {
            return;
        }

        var linear = StageUpload(context, plan, source, out var tight);
        var encoder = _tiler.BeginBlit(context);
        for (var index = 0; index < _copies.Count; index++)
        {
            var copy = _copies[index];
            var (offset, rowBytes, rows) = BlitRegion(plan, index, linear, tight);
            MetalNative.SendCopyBufferToTextureOptions(encoder, SelCopyBufferToTexture,
                linear.Buffer, (nuint)offset, rowBytes, (nuint)((ulong)rowBytes * rows),
                new MtlSize { Width = copy.Width, Height = copy.Height, Depth = copy.Depth },
                texture, copy.Slice, copy.Level,
                new MtlOrigin { X = 0, Y = copy.OriginY, Z = copy.OriginZ }, (nuint)MtlBlitOption.None);
        }

        MetalTiler.EndEncoding(encoder);
    }

    // Where one subresource's rows sit in the buffer a blit reads or writes. A tight
    // staging buffer holds the mip's own width per row; everything else keeps the guest
    // layout, where a padded pitch becomes the row stride.
    private (ulong Offset, uint RowBytes, uint Rows) BlitRegion(TransferPlan plan, int index, MetalBufferSpan linear, bool tight)
    {
        var copy = _copies[index];
        if (tight)
        {
            return (linear.Offset + _stagedOffsets[index], RowBytes(plan, copy.Width), BlockRows(plan, copy.Height));
        }

        var rowTexels = copy.BufferRowLength != 0 ? copy.BufferRowLength : copy.Width;
        var rows = BlockRows(plan, copy.BufferImageHeight != 0 ? copy.BufferImageHeight : copy.Height);
        return (linear.Offset + copy.BufferOffset, RowBytes(plan, rowTexels), rows);
    }

    public void EncodeTextureToGuest(IMetalSubmissionContext context, in ImageDescription description,
        SubresourceRange range, nint texture, MetalGuestRange destination, MetalGuestRange stencilDestination,
        ulong guestBase, ulong pieceSize, uint textureFirstLevel = 0)
    {
        if (!Ready(context, description, destination, texture, guestBase, pieceSize))
        {
            return;
        }

        var plan = PlanFor(description);
        if (plan.DepthStencil)
        {
            if (textureFirstLevel != 0)
            {
                throw GpuFatal.Fatal(
                    $"A depth image has no texture from a later level: firstLevel={textureFirstLevel} " +
                    $"address=0x{description.Data.Address:X16}.");
            }

            EncodeDepthDownload(context, description, plan, range, texture, destination, stencilDestination, guestBase, pieceSize);
            return;
        }

        Select(plan, description, range, guestBase, pieceSize, textureFirstLevel);
        if (_copies.Count == 0)
        {
            return;
        }

        if (TryEncodeTextureTiles(context, plan, texture, description.Data.Address + guestBase,
                new MetalBufferSpan(destination.Buffer, destination.Offset, pieceSize), MetalTileDirection.Tile))
        {
            return;
        }

        // A readback lands in staging first whenever the retile reads it, the channels
        // need swapping, or the guest offset and pitch are not something a blit can write.
        var staged = BuildStagedOffsets(plan);
        var direct = !plan.Tiled && !plan.Swap && IsDirectlyBlittable(plan, destination.Offset);
        var tight = !direct && !plan.Tiled;
        var linear = direct
            ? new MetalBufferSpan(destination.Buffer, destination.Offset, pieceSize)
            : _tiler.Acquire(context, plan.Tiled ? plan.LinearSize : staged);

        var blit = _tiler.BeginBlit(context);
        for (var index = 0; index < _copies.Count; index++)
        {
            var copy = _copies[index];
            var (offset, rowBytes, rows) = BlitRegion(plan, index, linear, tight);
            MetalNative.SendCopyTextureToBufferOptions(blit, SelCopyTextureToBuffer,
                texture, copy.Slice, copy.Level,
                new MtlOrigin { X = 0, Y = copy.OriginY, Z = copy.OriginZ },
                new MtlSize { Width = copy.Width, Height = copy.Height, Depth = copy.Depth },
                linear.Buffer, (nuint)offset, rowBytes, (nuint)((ulong)rowBytes * rows),
                (nuint)MtlBlitOption.None);
        }

        MetalTiler.EndEncoding(blit);
        if (direct)
        {
            return;
        }

        var compute = _tiler.BeginCompute(context);
        var source = linear;
        if (plan.Swap)
        {
            var swapped = _tiler.Acquire(context, plan.Tiled ? plan.LinearSize : staged);
            _tiler.EncodeSwapBgra16(compute, new MetalBufferSpan(linear.Buffer, linear.Offset, swapped.Size), swapped);
            source = swapped;
        }

        var guest = new MetalBufferSpan(destination.Buffer, destination.Offset, pieceSize);
        if (plan.Tiled)
        {
            _tiler.EncodeTile(compute, new MetalBufferSpan(source.Buffer, source.Offset, plan.LinearSize), guest, _tiles);
        }
        else
        {
            RepackSubresources(compute, plan, guest, source, toGuest: true);
        }

        MetalTiler.EndEncoding(compute);
    }

    // SHARPEMU_METAL_V2_TEXTURE_TILER=0 stages every tiled colour transfer through a
    // linear buffer and a blit again, for a comparison or a bisect.
    private static readonly bool TextureTilerEnabled =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_TEXTURE_TILER") != "0";

    private static readonly nint SelPixelFormat = MetalNative.Selector("pixelFormat");
    private static readonly nint SelUsage = MetalNative.Selector("usage");
    private static readonly nint SelTextureType = MetalNative.Selector("textureType");
    private static readonly nint SelMipmapLevelCount = MetalNative.Selector("mipmapLevelCount");
    private static readonly nint SelArrayLength = MetalNative.Selector("arrayLength");
    private static readonly nint SelNewTextureView =
        MetalNative.Selector("newTextureViewWithPixelFormat:textureType:levels:slices:");
    private static readonly nint SelReleaseView = MetalNative.Selector("release");

    /// <summary>
    /// The unsigned-integer view format whose texel is exactly one element of this colour
    /// format, or Invalid when the format is not one a texture block copy may carry: an
    /// sRGB format (no shader writes, and a read would decode it), a shared-exponent or
    /// compressed one, or one whose texel is not 4, 8 or 16 bytes.
    /// </summary>
    internal static MtlPixelFormat TextureTileViewFormat(MtlPixelFormat format) => format switch
    {
        MtlPixelFormat.R32Uint or MtlPixelFormat.R32Sint or MtlPixelFormat.R32Float or
        MtlPixelFormat.Rg16Unorm or MtlPixelFormat.Rg16Snorm or MtlPixelFormat.Rg16Uint or
        MtlPixelFormat.Rg16Sint or MtlPixelFormat.Rg16Float or
        MtlPixelFormat.Rgba8Unorm or MtlPixelFormat.Rgba8Snorm or MtlPixelFormat.Rgba8Uint or
        MtlPixelFormat.Rgba8Sint or MtlPixelFormat.Bgra8Unorm or
        MtlPixelFormat.Rgb10A2Unorm or MtlPixelFormat.Rgb10A2Uint or MtlPixelFormat.Bgr10A2Unorm or
        MtlPixelFormat.Rg11B10Float => MtlPixelFormat.R32Uint,
        MtlPixelFormat.Rg32Uint or MtlPixelFormat.Rg32Sint or MtlPixelFormat.Rg32Float or
        MtlPixelFormat.Rgba16Unorm or MtlPixelFormat.Rgba16Snorm or MtlPixelFormat.Rgba16Uint or
        MtlPixelFormat.Rgba16Sint or MtlPixelFormat.Rgba16Float => MtlPixelFormat.Rg32Uint,
        MtlPixelFormat.Rgba32Uint or MtlPixelFormat.Rgba32Sint or MtlPixelFormat.Rgba32Float => MtlPixelFormat.Rgba32Uint,
        _ => MtlPixelFormat.Invalid,
    };

    private static uint ViewElementBytes(MtlPixelFormat view) => view switch
    {
        MtlPixelFormat.R32Uint => 4,
        MtlPixelFormat.Rg32Uint => 8,
        MtlPixelFormat.Rgba32Uint => 16,
        _ => 0,
    };

    // A tiled colour transfer moved in one pass between the guest surface and the texture,
    // instead of through a linear staging buffer and a blit: half the bytes moved, and on
    // a frame that ping-pongs aliased render targets through guest memory the transfers
    // are most of the GPU's time. Only what the one-pass kernels address exactly the way
    // the staged path does qualifies - whole-element texels of a plain 2D or 2D-array
    // texture, no mip tail, no channel swap - and the staged path keeps the rest.
    private bool TryEncodeTextureTiles(
        IMetalSubmissionContext context, TransferPlan plan, nint texture, ulong guestAddress, MetalBufferSpan guest,
        MetalTileDirection direction)
    {
        if (!TextureTilerEnabled || !plan.Tiled || plan.Swap || plan.Volume || plan.TexelWidth != 1 ||
            plan.TexelHeight != 1 || _tiles.Count == 0 || guest.Offset % 4 != 0)
        {
            return false;
        }

        var format = (MtlPixelFormat)(uint)MetalV2Native.SendValue(texture, SelPixelFormat);
        var viewFormat = TextureTileViewFormat(format);
        if (viewFormat == MtlPixelFormat.Invalid || ViewElementBytes(viewFormat) != plan.ElementBytes)
        {
            return false;
        }

        var usage = (MtlTextureUsage)(uint)MetalV2Native.SendValue(texture, SelUsage);
        var needed = direction == MetalTileDirection.Detile ? MtlTextureUsage.ShaderWrite : MtlTextureUsage.ShaderRead;
        if ((usage & needed) == 0 || (format != viewFormat && (usage & MtlTextureUsage.PixelFormatView) == 0))
        {
            return false;
        }

        var type = (MtlTextureType)(uint)MetalV2Native.SendValue(texture, SelTextureType);
        if (type is not (MtlTextureType.Type2D or MtlTextureType.Type2DArray))
        {
            return false;
        }

        var levels = MetalV2Native.SendValue(texture, SelMipmapLevelCount);
        var slices = MetalV2Native.SendValue(texture, SelArrayLength);
        foreach (var transfer in _tiles)
        {
            if (transfer.Tail || transfer.Depth != 1 || transfer.BytesPerElement != plan.ElementBytes ||
                transfer.Level >= levels || transfer.Slice >= slices)
            {
                return false;
            }
        }

        var view = MetalV2Native.SendNewTextureView(
            texture, SelNewTextureView, (nuint)viewFormat, (nuint)MtlTextureType.Type2DArray,
            new NsRange { Location = 0, Length = levels }, new NsRange { Location = 0, Length = slices });
        if (view == 0)
        {
            throw GpuFatal.Fatal(
                $"The texture tiler could not make its view: format={format} view={viewFormat} type={type} " +
                $"levels={levels} slices={slices}.");
        }

        try
        {
            // Nothing but dispatches: the open compute encoder takes them when the host
            // leaves it to this transfer, which saves the encoder boundary.
            var shared = context.TryShareComputeEncoder(guestAddress, guest.Size);
            var compute = shared != 0 ? shared : _tiler.BeginCompute(context);
            _tiler.EncodeTextureBlockCopies(compute, guest, view, _tiles, direction);
            if (shared != 0)
            {
                context.EndSharedCompute();
            }
            else
            {
                MetalTiler.EndEncoding(compute);
            }
        }
        finally
        {
            // The command buffer holds the view for as long as the encoded work needs it.
            MetalNative.SendVoid(view, SelReleaseView);
        }

        return true;
    }

    private bool Ready(
        IMetalSubmissionContext context, in ImageDescription description, in MetalGuestRange guest, nint texture,
        ulong guestBase, ulong pieceSize)
    {
        if (_disposed || context.CommandBuffer == 0 || texture == 0 || !guest.IsValid ||
            ImageDescription.IsEmptyRange(description.Data) || description.Samples != 1 ||
            description.PixelFormat == GpuFormat.Undefined)
        {
            return false;
        }

        if (guestBase > description.Data.Size || pieceSize > description.Data.Size - guestBase)
        {
            throw GpuFatal.Fatal(
                $"An image piece falls outside its image: base=0x{guestBase:X} piece=0x{pieceSize:X} " +
                $"image=0x{description.Data.Size:X}.");
        }

        if (guest.Size < pieceSize)
        {
            // The resolver could not map even this piece. A copy sized from the
            // description would run off the end of what it did map, so this piece is
            // dropped and the texture keeps whatever it already holds there. The caller
            // splits a large image into pieces precisely so that one unmappable piece
            // costs only its own subresources; never fatal, and never silent.
            MetalV2Diagnostics.WarnOnce(
                "short-image-range",
                $"A guest range is smaller than the image piece, so the piece is skipped: " +
                $"range=0x{guest.Size:x} piece=0x{pieceSize:x} at 0x{guestBase:x} of 0x{description.Data.Size:x} " +
                $"address=0x{description.Data.Address:X16} format={description.PixelFormat} " +
                $"type={description.Type} extent={description.Extent.Width}x{description.Extent.Height}x" +
                $"{description.Extent.Depth} levels={description.Resources.Levels} " +
                $"layers={description.TransferLayers} tile={description.TileMode}.");
            return false;
        }

        return true;
    }

    // Staging for an upload: the detiled bytes, the swapped bytes, or nothing at all when
    // the guest range is already something the blit can read. `tight` reports whether the
    // result holds one tightly packed subresource per copy rather than the guest layout.
    private MetalBufferSpan StageUpload(IMetalSubmissionContext context, TransferPlan plan, MetalGuestRange source, out bool tight)
    {
        tight = false;
        var staged = BuildStagedOffsets(plan);
        if (!plan.Tiled && !plan.Swap && IsDirectlyBlittable(plan, source.Offset))
        {
            return new MetalBufferSpan(source.Buffer, source.Offset, source.Size);
        }

        var guest = new MetalBufferSpan(source.Buffer, source.Offset, source.Size);
        var compute = _tiler.BeginCompute(context);
        MetalBufferSpan linear;
        if (plan.Tiled)
        {
            linear = _tiler.Acquire(context, plan.LinearSize);
            _tiler.EncodeDetile(compute, new MetalBufferSpan(guest.Buffer, guest.Offset, TiledCapacity(source)), linear, _tiles);
        }
        else
        {
            tight = true;
            linear = _tiler.Acquire(context, staged);
            RepackSubresources(compute, plan, guest, linear, toGuest: false);
        }

        if (plan.Swap)
        {
            var swapped = _tiler.Acquire(context, linear.Size);
            _tiler.EncodeSwapBgra16(compute, linear, swapped);
            linear = swapped;
        }

        MetalTiler.EndEncoding(compute);
        return linear;
    }

    // How much of the mapped guest piece the selected tiles actually reach. The selection
    // is already rebased onto the piece, so this is the piece's own extent, not the
    // image's.
    private ulong TiledCapacity(in MetalGuestRange source)
    {
        ulong capacity = 0;
        foreach (var transfer in _tiles)
        {
            capacity = Math.Max(capacity, transfer.TiledOffset + transfer.TiledSize);
        }

        return Math.Min(Math.Max(capacity, 4), source.Size);
    }

    // A linear subresource staged tightly has rows of exactly the mip's own width, which
    // is what the repack kernel writes and what the blit then reads. Fills the per-copy
    // offsets and returns the total.
    private ulong BuildStagedOffsets(TransferPlan plan)
    {
        _stagedOffsets.Clear();
        ulong offset = 0;
        foreach (var copy in _copies)
        {
            _stagedOffsets.Add(offset);
            offset += StagedSize(plan, copy);
        }

        return Math.Max(offset, 4);
    }

    private static ulong StagedSize(TransferPlan plan, in MetalSubresourceCopy copy)
    {
        var size = (ulong)RowBytes(plan, copy.Width) * BlockRows(plan, copy.Height) * Math.Max(copy.Depth, 1);
        return (size + StagingAlignment - 1) & ~(StagingAlignment - 1);
    }

    private void RepackSubresources(nint encoder, TransferPlan plan, MetalBufferSpan guest, MetalBufferSpan staging, bool toGuest)
    {
        for (var index = 0; index < _copies.Count; index++)
        {
            var copy = _copies[index];
            var rowTexels = copy.BufferRowLength != 0 ? copy.BufferRowLength : copy.Width;
            var guestRowBytes = RowBytes(plan, rowTexels);
            var tightRowBytes = RowBytes(plan, copy.Width);
            var rows = BlockRows(plan, copy.Height);
            var guestSpan = new MetalBufferSpan(guest.Buffer, guest.Offset + copy.BufferOffset, guest.Size - copy.BufferOffset);
            var stagedSpan = new MetalBufferSpan(staging.Buffer, staging.Offset + _stagedOffsets[index], StagedSize(plan, copy));
            if (toGuest)
            {
                _tiler.EncodeRepackRows(encoder, stagedSpan, guestSpan, tightRowBytes, rows, 1,
                    tightRowBytes, guestRowBytes, 0, 0);
            }
            else
            {
                _tiler.EncodeRepackRows(encoder, guestSpan, stagedSpan, tightRowBytes, rows, 1,
                    guestRowBytes, tightRowBytes, 0, 0);
            }
        }
    }

    private bool IsDirectlyBlittable(TransferPlan plan, ulong guestOffset)
    {
        var alignment = Math.Max(plan.ElementBytes, BlitAlignment);
        if (guestOffset % alignment != 0)
        {
            return false;
        }

        foreach (var copy in _copies)
        {
            var rowTexels = copy.BufferRowLength != 0 ? copy.BufferRowLength : copy.Width;
            if (copy.BufferOffset % alignment != 0 || RowBytes(plan, rowTexels) % alignment != 0)
            {
                return false;
            }
        }

        return true;
    }

    private static uint RowBytes(TransferPlan plan, uint texels) =>
        Math.Max((texels + plan.TexelWidth - 1) / plan.TexelWidth, 1) * plan.ElementBytes;

    private static uint BlockRows(TransferPlan plan, uint texels) =>
        Math.Max((texels + plan.TexelHeight - 1) / plan.TexelHeight, 1);

    // Narrows the plan to the subresources the caller asked for. A volume's layers are
    // depth slices of one texture, so only its levels are selected.
    //
    // guestBase is where the guest range the caller resolved starts inside the image, so a
    // caller that could only map part of a very large image hands over that part and the
    // subresources inside it. Every selected offset is rebased onto it: the guest offsets
    // become relative to the piece that was mapped, and the linear offsets to the start of
    // the piece's own staging, so the staging a transfer asks for is the piece's size and
    // never the whole image's.
    //
    // textureFirstLevel moves every selected level down by that much, onto a texture that
    // holds only the guest levels from there on (ShapeFor). The guest offsets stay the full
    // chain's: the guest lays the chain out from mip 0 whatever the texture keeps of it.
    private void Select(
        TransferPlan plan, in ImageDescription description, SubresourceRange range, ulong guestBase, ulong pieceSize,
        uint textureFirstLevel = 0)
    {
        _copies.Clear();
        _tiles.Clear();
        if (textureFirstLevel != 0 && (range.LevelCount == 0 || range.BaseLevel < textureFirstLevel))
        {
            throw GpuFatal.Fatal(
                $"A transfer names guest levels the texture does not hold: levels={range.BaseLevel}+{range.LevelCount} " +
                $"firstLevel={textureFirstLevel} address=0x{description.Data.Address:X16}.");
        }

        // The guest bytes this transfer may touch, as offsets into the image. A caller
        // that could only map part of the image hands over that part, and a subresource
        // that straddles the end of it is cut down to the block rows inside rather than
        // dropped: a run of whole block rows is a contiguous run of tiled bytes, so it
        // is the finest cut whose two sides are both still one range.
        // A piece that reaches the end of the image has no upper limit: a subresource's
        // tiled extent is its block grid, which can run past the bytes the description
        // claims, and the last piece owns whatever is there. Only a piece with more of
        // the image after it has to stop where it stops.
        var end = pieceSize > ulong.MaxValue - guestBase ? ulong.MaxValue : guestBase + pieceSize;
        var limit = pieceSize == 0 || end >= description.Data.Size ? ulong.MaxValue : end;
        var lastLevel = range.LevelCount == 0 ? 0 : range.BaseLevel + range.LevelCount - 1;
        var lastLayer = range.LayerCount == 0 ? 0 : range.BaseLayer + range.LayerCount - 1;
        if (!plan.Tiled)
        {
            foreach (var copy in plan.Copies)
            {
                if (!Selected(plan, copy, range, lastLevel, lastLayer) ||
                    !TryClipRows(plan, copy, guestBase, limit, out var clipped))
                {
                    continue;
                }

                var rebased = clipped;
                rebased.BufferOffset = Rebase(clipped.BufferOffset, guestBase, "a subresource copy");
                rebased.Level -= textureFirstLevel;
                _copies.Add(rebased);
            }

            return;
        }

        // A tiled piece names tiled bytes, but its copies and its staging are linear, and
        // the two are different offsets: a band past the first starts at its own linear
        // row, and a subresource's tiled extent is padded to whole blocks where its linear
        // one is not. So the piece first picks the block rows whose tiled bytes it holds;
        // those rows, in linear offsets, are what the copies are cut to and what the
        // staging is rebased onto.
        var linearBase = ulong.MaxValue;
        foreach (var transfer in plan.Tiles)
        {
            if (!Selected(plan, transfer, range, lastLevel, lastLayer) || !Overlaps(transfer, guestBase, limit))
            {
                continue;
            }

            AddBandsWithin(transfer, guestBase, limit, textureFirstLevel, ref linearBase);
        }

        if (linearBase == ulong.MaxValue)
        {
            plan.LinearSize = 4;
            return;
        }

        foreach (var copy in plan.Copies)
        {
            if (!Selected(plan, copy, range, lastLevel, lastLayer) ||
                !TryClipToBands(plan, copy, textureFirstLevel, out var clipped))
            {
                continue;
            }

            var rebased = clipped;
            rebased.BufferOffset = Rebase(clipped.BufferOffset, linearBase, "a tiled subresource copy");
            rebased.Level -= textureFirstLevel;
            _copies.Add(rebased);
        }

        ulong linear = 0;
        for (var index = 0; index < _tiles.Count; index++)
        {
            var rebased = _tiles[index];
            rebased.LinearOffset -= linearBase;
            _tiles[index] = rebased;
            linear = Math.Max(linear, rebased.LinearOffset + rebased.LinearSize);
        }

        plan.LinearSize = Math.Max(linear, 4);
    }

    /// <summary>
    /// The largest offset at or below <paramref name="end"/> that no part of this image's
    /// transfer plan straddles, so a run ending there loses nothing: a transfer the run
    /// would cut is either pulled back to a block-row boundary inside it or excluded
    /// entirely. Zero means nothing of the image starts inside the run.
    /// </summary>
    public ulong AlignRunEnd(in ImageDescription description, ulong start, ulong end)
    {
        if (end <= start)
        {
            return start;
        }

        var plan = PlanFor(description);
        if (plan.DepthStencil)
        {
            // The depth kernels walk a layer at a time, so a run of one ends where a
            // layer does.
            return plan.SliceSize == 0 ? start : Math.Max(start, end / plan.SliceSize * plan.SliceSize);
        }

        var aligned = end;
        foreach (var copy in RunCopies(plan))
        {
            if (copy.BufferOffset >= aligned || CopyEnd(plan, copy) <= aligned)
            {
                continue;
            }

            var rowTexels = copy.BufferRowLength != 0 ? copy.BufferRowLength : copy.Width;
            var rowBytes = (ulong)RowBytes(plan, rowTexels);
            aligned = rowBytes != 0 && copy.Depth == 1
                ? Math.Max(start, copy.BufferOffset + ((aligned - copy.BufferOffset) / rowBytes * rowBytes))
                : Math.Max(start, copy.BufferOffset);
        }

        // The stencil plane needs no pass of its own: only a depth plan has one, and
        // those ended above - and its offsets are in the stencil plane's bytes anyway,
        // which are not the bytes a run is named in.
        foreach (var transfer in plan.Tiles)
        {
            aligned = AlignToBands(transfer, start, aligned);
        }

        return aligned;
    }

    /// <summary>
    /// The first offset at or after <paramref name="offset"/> that a restricted transfer
    /// can begin at. A caller steps over guest bytes no window covers and lands wherever
    /// the window ended, which may be inside a block row or a texel row whose other half
    /// it just stepped past: that row carries nothing now, so the run after it begins
    /// past the row rather than inside it, and every byte a run is given is a byte the
    /// run really writes.
    /// </summary>
    public ulong AlignRunStart(in ImageDescription description, ulong offset)
    {
        var plan = PlanFor(description);
        if (plan.DepthStencil)
        {
            return plan.SliceSize == 0
                ? offset
                : (offset + plan.SliceSize - 1) / plan.SliceSize * plan.SliceSize;
        }

        // Pushing past one transfer can land inside the next, so the push repeats until
        // it stands still. Two kinds of transfer and no overlap between them means it
        // settles at once in practice; the bound is only there so it always ends.
        var aligned = offset;
        for (var pass = 0; pass < 4; pass++)
        {
            var moved = aligned;
            foreach (var copy in RunCopies(plan))
            {
                var rowTexels = copy.BufferRowLength != 0 ? copy.BufferRowLength : copy.Width;
                moved = AdvancePast(moved, copy.BufferOffset, CopyEnd(plan, copy),
                    copy.Depth == 1 ? (ulong)RowBytes(plan, rowTexels) : 0);
            }

            foreach (var transfer in plan.Tiles)
            {
                var rowBytes = TryBandBytes(transfer, out var bandBytes, out _) ? bandBytes : 0;
                moved = AdvancePast(moved, transfer.TiledOffset, transfer.TiledOffset + transfer.TiledSize, rowBytes);
            }

            if (moved == aligned)
            {
                break;
            }

            aligned = moved;
        }

        return aligned;
    }

    private static readonly List<MetalSubresourceCopy> NoCopies = [];

    // The copies whose guest bytes bound a run: a linear plan's, and none of a tiled
    // plan's, whose copies are named in linear staging offsets and cut from its bands.
    private static List<MetalSubresourceCopy> RunCopies(TransferPlan plan) => plan.Tiled ? NoCopies : plan.Copies;

    // Moves an offset that falls inside a transfer forward to the next row of it, or past
    // it when it has no rows of its own; an offset outside the transfer stands still.
    private static ulong AdvancePast(ulong offset, ulong from, ulong to, ulong rowBytes)
    {
        if (offset <= from || offset >= to)
        {
            return offset;
        }

        if (rowBytes == 0)
        {
            return to;
        }

        var rows = (offset - from + rowBytes - 1) / rowBytes;
        return Math.Min(to, from + (rows * rowBytes));
    }

    // Where a run may end inside one tile transfer: at a block-row boundary when the
    // transfer can be cut into rows, and otherwise before the transfer begins.
    private static ulong AlignToBands(in MetalTileTransfer transfer, ulong start, ulong end)
    {
        var transferEnd = transfer.TiledOffset + transfer.TiledSize;
        if (transfer.TiledOffset >= end || transferEnd <= end)
        {
            return end;
        }

        if (!TryBandBytes(transfer, out var rowBytes, out _) || transfer.TiledOffset >= end)
        {
            return Math.Max(start, Math.Min(end, transfer.TiledOffset));
        }

        var rows = (end - transfer.TiledOffset) / rowBytes;
        var band = transfer.TiledOffset + (rows * rowBytes);
        return Math.Max(start, Math.Min(end, band));
    }

    // The bytes one block row of a tile transfer occupies, and the surface rows it
    // carries. A mip tail is one block the layout places by hand and a volume's blocks
    // span depth, so neither has rows that are their own contiguous run.
    private static bool TryBandBytes(in MetalTileTransfer transfer, out ulong rowBytes, out uint rowHeight)
    {
        rowBytes = 0;
        rowHeight = 0;
        if (transfer.Tail || transfer.Depth != 1 || transfer.TiledWidth == 0 ||
            !TileGeometry.TryGetBlockLayout(transfer.Kind, transfer.BytesPerElement, out var block) ||
            block.BlockDepth != 1 || block.BlockHeight == 0 || block.BlockWidth == 0)
        {
            return false;
        }

        var columns = ((ulong)transfer.TiledWidth + block.BlockWidth - 1) / block.BlockWidth;
        rowBytes = columns * block.BlockSize;
        rowHeight = block.BlockHeight;
        return rowBytes != 0;
    }

    /// <summary>
    /// A subresource copy cut down to the rows whose guest bytes lie inside a run. A
    /// linear row is one contiguous run of guest bytes, so a copy that only partly fits
    /// keeps the rows that do; one that fits whole is kept as it is, and one whose rows
    /// are not its own runs (a depth slice stack) is dropped.
    /// </summary>
    private static bool TryClipRows(
        TransferPlan plan, in MetalSubresourceCopy copy, ulong from, ulong to, out MetalSubresourceCopy clipped)
    {
        clipped = copy;
        var end = CopyEnd(plan, copy);
        if (copy.BufferOffset >= from && end <= to)
        {
            return true;
        }

        if (end <= from || copy.BufferOffset >= to || copy.Depth != 1 || copy.Height == 0)
        {
            return false;
        }

        var rowTexels = copy.BufferRowLength != 0 ? copy.BufferRowLength : copy.Width;
        var rowBytes = (ulong)RowBytes(plan, rowTexels);
        var blockRows = BlockRows(plan, copy.Height);
        if (rowBytes == 0 || blockRows == 0 || plan.TexelHeight == 0)
        {
            return false;
        }

        var firstRow = copy.BufferOffset >= from ? 0UL : (from - copy.BufferOffset + rowBytes - 1) / rowBytes;
        var lastRow = Math.Min(blockRows, (ulong)((to - copy.BufferOffset) / rowBytes));
        if (firstRow >= lastRow)
        {
            return false;
        }

        clipped.BufferOffset = copy.BufferOffset + (firstRow * rowBytes);
        clipped.OriginY = copy.OriginY + (uint)(firstRow * plan.TexelHeight);
        // The last block row of a block-compressed copy can run past the texels it has.
        clipped.Height = (uint)Math.Min((lastRow - firstRow) * plan.TexelHeight, copy.Height - (firstRow * plan.TexelHeight));
        clipped.BufferImageHeight = clipped.Height;
        return true;
    }

    // A tiled subresource copy cut down to the block rows the selected bands carry. The
    // bands still hold their offsets in the whole image's linear staging, as the copy
    // does, so the rows of the copy a band overlaps are the rows it carries.
    //
    // Only a band of the copy's own subresource counts. A volume's staging interleaves its
    // levels slice by slice (slice z of every level sits in z's stride), so a band of one
    // level spans the other levels' slices too: matched by bytes alone, a piece holding
    // only level 0's tiles took level 1's copies with it and blitted whatever its staging
    // held there over level 1.
    private bool TryClipToBands(
        TransferPlan plan, in MetalSubresourceCopy copy, uint textureFirstLevel, out MetalSubresourceCopy clipped)
    {
        var copyEnd = CopyEnd(plan, copy);
        var from = ulong.MaxValue;
        ulong to = 0;
        foreach (var band in _tiles)
        {
            if (band.Level + textureFirstLevel != copy.Level ||
                (plan.Volume
                    ? copy.OriginZ < band.Slice || copy.OriginZ - band.Slice >= Math.Max(band.Depth, 1u)
                    : band.Slice != copy.Slice))
            {
                continue;
            }

            var pitchBytes = (ulong)band.Pitch * band.BytesPerElement;
            var sliceBytes = band.LinearSliceStride != 0 ? band.LinearSliceStride : pitchBytes * band.Height;
            var bandEnd = band.LinearOffset + (sliceBytes * (Math.Max(band.Depth, 1u) - 1)) + (pitchBytes * band.Height);
            if (band.LinearOffset >= copyEnd || bandEnd <= copy.BufferOffset)
            {
                continue;
            }

            from = Math.Min(from, band.LinearOffset);
            to = Math.Max(to, bandEnd);
        }

        clipped = copy;
        return from != ulong.MaxValue && TryClipRows(plan, copy, from, to, out clipped);
    }

    private static bool Selected(
        TransferPlan plan, in MetalSubresourceCopy copy, SubresourceRange range, uint lastLevel, uint lastLayer) =>
        range.LevelCount != 0 &&
        copy.Level >= range.BaseLevel && copy.Level <= lastLevel &&
        (plan.Volume || (range.LayerCount != 0 && copy.Slice >= range.BaseLayer && copy.Slice <= lastLayer));

    /// <summary>The end of a subresource copy's guest bytes.</summary>
    private static ulong CopyEnd(TransferPlan plan, in MetalSubresourceCopy copy)
    {
        var rowTexels = copy.BufferRowLength != 0 ? copy.BufferRowLength : copy.Width;
        var rows = BlockRows(plan, copy.BufferImageHeight != 0 ? copy.BufferImageHeight : copy.Height);
        return copy.BufferOffset + ((ulong)RowBytes(plan, rowTexels) * rows * Math.Max(copy.Depth, 1));
    }

    private static bool Overlaps(in MetalTileTransfer transfer, ulong from, ulong to) =>
        transfer.TiledOffset < to && transfer.TiledOffset + transfer.TiledSize > from;

    /// <summary>
    /// The parts of a tile transfer whose tiled bytes lie inside a guest run. One whole
    /// block row is one contiguous run of tiled bytes - the block index walks x before
    /// y - so a transfer that only partly fits is cut at block-row boundaries. Anything
    /// that cannot be cut that way (a mip tail, which is one block the layout places by
    /// hand, or a volume whose blocks span depth) is kept whole or dropped whole.
    /// </summary>
    // Each band is rebased onto the piece (guest offsets from guestBase) and added to
    // _tiles, and linearBase drops to the lowest linear offset added. Not an iterator: it
    // runs for every tile of every transfer, and an iterator allocated each time.
    private void AddBandsWithin(in MetalTileTransfer transfer, ulong from, ulong to, uint textureFirstLevel, ref ulong linearBase)
    {
        if (transfer.TiledOffset >= from && transfer.TiledOffset + transfer.TiledSize <= to)
        {
            AddBand(transfer, from, textureFirstLevel, ref linearBase);
            return;
        }

        var pitchBytes = (ulong)transfer.Pitch * transfer.BytesPerElement;
        if (!TryBandBytes(transfer, out var rowBytes, out var rowHeight) || pitchBytes == 0)
        {
            return;
        }

        var transferEnd = transfer.TiledOffset + transfer.TiledSize;
        for (uint row = 0; (ulong)row * rowHeight < transfer.Height; row++)
        {
            var tiledOffset = transfer.TiledOffset + (row * rowBytes);
            if (tiledOffset < from || tiledOffset + rowBytes > to || tiledOffset + rowBytes > transferEnd)
            {
                continue;
            }

            var band = transfer;
            band.BaseY = transfer.BaseY + (row * rowHeight);
            band.Height = Math.Min(rowHeight, transfer.Height - (row * rowHeight));
            band.TiledOffset = tiledOffset;
            band.TiledSize = rowBytes;
            band.TiledHeight = rowHeight;
            band.LinearOffset = transfer.LinearOffset + (row * rowHeight * pitchBytes);
            band.LinearSize = ((ulong)(band.Height - 1) * pitchBytes) + ((ulong)transfer.Width * transfer.BytesPerElement);
            AddBand(band, from, textureFirstLevel, ref linearBase);
        }
    }

    private void AddBand(in MetalTileTransfer band, ulong guestBase, uint textureFirstLevel, ref ulong linearBase)
    {
        var rebased = band;
        rebased.TiledOffset = Rebase(band.TiledOffset, guestBase, "a tile transfer");
        rebased.Level -= textureFirstLevel;
        _tiles.Add(rebased);
        linearBase = Math.Min(linearBase, band.LinearOffset);
    }

    private static bool Selected(
        TransferPlan plan, in MetalTileTransfer transfer, SubresourceRange range, uint lastLevel, uint lastLayer) =>
        range.LevelCount != 0 &&
        transfer.Level >= range.BaseLevel && transfer.Level <= lastLevel &&
        (plan.Volume || (range.LayerCount != 0 && transfer.Slice >= range.BaseLayer && transfer.Slice <= lastLayer));

    private static ulong Rebase(ulong offset, ulong guestBase, string what) =>
        offset >= guestBase
            ? offset - guestBase
            : throw GpuFatal.Fatal(
                $"{what} starts before the mapped piece of its image: offset=0x{offset:X} base=0x{guestBase:X}.");

    private TransferPlan PlanFor(in ImageDescription description)
    {
        var key = new PlanKey(
            description.GuestFormat,
            description.PixelFormat,
            description.TileMode,
            description.Type,
            description.Extent.Width,
            description.Extent.Height,
            description.Extent.Depth,
            description.Resources.Levels,
            description.Resources.Layers,
            description.Pitch,
            description.BytesPerBlock,
            description.Data.Size,
            description.Stencil.Size,
            description.Bgra16);
        if (_plans.TryGetValue(key, out var plan))
        {
            return plan;
        }

        plan = description.IsDepth ? PlanDepth(description) : PlanColor(description);
        // A plan is a pure function of its key, so dropping them all only costs replanning.
        // Every distinct shape a title streams (and every resize) adds one, each with a copy
        // per level and layer.
        if (_plans.Count >= MaxPlans)
        {
            _plans.Clear();
        }

        _plans[key] = plan;
        return plan;
    }

    private static TransferPlan PlanColor(in ImageDescription description)
    {
        if (!MetalGuestFormats.TryDecodeHostFormat(description.PixelFormat, out var format))
        {
            throw GpuFatal.Fatal($"No Metal pixel format carries this image: hostFormat={(int)description.PixelFormat}.");
        }

        // A render-target or depth-tiled surface is addressed by element size, not by its
        // sampled channel layout, exactly as the Vulkan ColorTarget role does.
        var tilerFormat = description.TileMode is GuestTileMode.RenderTarget or GuestTileMode.Depth
            ? ImageDescription.RenderTargetTransferFormat(description.BytesPerBlock)
            : description.GuestFormat;
        var plan = new TransferPlan
        {
            Format = format,
            Volume = description.IsVolume,
            Swap = description.Bgra16,
            Tiled = description.IsTiled,
        };
        plan.Layout = MetalTextureLayout.Compute(
            tilerFormat,
            description.Extent.Width,
            description.Extent.Height,
            description.Resources.Levels,
            description.TransferLayers,
            description.TileMode,
            description.Data.Size,
            allowDepthTile: true,
            volume: description.IsVolume,
            owner: "image transfer");
        plan.Layout.BuildCopies(plan.Copies);
        if (plan.Tiled)
        {
            var texture = plan.Layout.Surface.Texture;
            plan.ElementBytes = texture.Block.BytesPerElement;
            plan.TexelWidth = texture.TexelWidth;
            plan.TexelHeight = texture.TexelHeight;
            if (!plan.Layout.TryBuildTileTransfers(description.Data.Size, plan.Copies, description.Resources.Levels, plan.Tiles))
            {
                throw GpuFatal.Fatal(
                    $"The tiled image cannot be transferred on the GPU: address=0x{description.Data.Address:X16} " +
                    $"size=0x{description.Data.Size:x} format={(uint)tilerFormat} tile={(uint)description.TileMode} " +
                    $"extent={description.Extent.Width}x{description.Extent.Height}x{description.Extent.Depth} " +
                    $"levels={description.Resources.Levels} layers={description.Resources.Layers}.");
            }
        }
        else
        {
            if (!TileGeometry.TryGetElementLayout(tilerFormat, out var element))
            {
                throw GpuFatal.Fatal($"The linear image format is not supported: format={(uint)tilerFormat}.");
            }

            plan.ElementBytes = element.Bytes;
            plan.TexelWidth = element.TexelWidth;
            plan.TexelHeight = element.TexelHeight;
        }

        return plan;
    }

    private static TransferPlan PlanDepth(in ImageDescription description)
    {
        if (!MetalGuestFormats.TryDecodeHostFormat(description.PixelFormat, out var format))
        {
            throw GpuFatal.Fatal($"No Metal depth format carries this image: hostFormat={(int)description.PixelFormat}.");
        }

        var layers = description.Resources.Layers;
        if (layers == 0 || description.Data.Size % layers != 0 || description.Resources.Levels != 1 ||
            GuestPixelFormats.BytesPerElement(description.GuestFormat) != description.BytesPerBlock)
        {
            throw GpuFatal.Fatal(
                $"The depth transfer is invalid: address=0x{description.Data.Address:X16} size=0x{description.Data.Size:x} " +
                $"layers={layers} levels={description.Resources.Levels} guestFormat={(uint)description.GuestFormat} " +
                $"bytesPerBlock={description.BytesPerBlock}.");
        }

        var plan = new TransferPlan
        {
            Format = format,
            DepthStencil = true,
            Tiled = description.IsTiled,
            Layers = layers,
            SliceSize = description.Data.Size / layers,
            ElementBytes = description.BytesPerBlock,
            TransferBytes = MetalGuestFormats.DepthTransferBytes(format),
            StencilPlane = description.HasStencil && MetalGuestFormats.HasStencilPlane(format),
        };
        if (plan.TransferBytes == 0 ||
            (plan.TransferBytes != description.BytesPerBlock && (description.BytesPerBlock != 2 || plan.TransferBytes != 4)))
        {
            throw GpuFatal.Fatal(
                $"The depth transfer conversion is not supported: bytesPerBlock={description.BytesPerBlock} transferBytes={plan.TransferBytes}.");
        }

        if (plan.Tiled)
        {
            // A depth image is addressed by the block layout of its own tile mode: a
            // comparison-sampled cube is Standard256B, not the 64 KiB depth block.
            var tilerFormat = description.TileMode is GuestTileMode.RenderTarget or GuestTileMode.Depth
                ? ImageDescription.RenderTargetTransferFormat(description.BytesPerBlock)
                : description.GuestFormat;
            if (!TileGeometry.TryGetTextureBlockLayout(tilerFormat, description.TileMode, false, out var texture) ||
                !TileGeometry.TryGetTiledTextureLayout(
                    new TiledSurfaceDescription(tilerFormat, description.TileMode, TileSurfaceDimension.Flat2D,
                        description.Extent.Width, description.Extent.Height), out var surface))
            {
                throw GpuFatal.Fatal(
                    $"The depth tile layout is not supported: format={(uint)tilerFormat} tile={(uint)description.TileMode}.");
            }

            if (plan.SliceSize != surface.BlockSliceSize)
            {
                throw GpuFatal.Fatal(
                    $"The depth slice does not match its tile layout: slice=0x{plan.SliceSize:x} " +
                    $"layout=0x{surface.BlockSliceSize:x} tile={(uint)description.TileMode}.");
            }

            plan.BlockKind = texture.Block.Kind;
            AddPlaneTransfers(plan.Tiles, plan.BlockKind, description.BytesPerBlock, description.Pitch,
                description.Extent.Width, description.Extent.Height, layers, plan.SliceSize,
                surface.Mips[0].PaddedWidth, surface.Mips[0].PaddedHeight);
        }

        if (plan.StencilPlane)
        {
            if (description.TileMode != GuestTileMode.Depth)
            {
                throw GpuFatal.Fatal($"A stencil plane needs the depth tile mode: tile={(uint)description.TileMode}.");
            }

            if (description.Stencil.Size % layers != 0)
            {
                throw GpuFatal.Fatal(
                    $"The stencil plane does not divide by its layers: size=0x{description.Stencil.Size:x} layers={layers}.");
            }

            plan.StencilSliceSize = description.Stencil.Size / layers;
            plan.StencilPitch = plan.Tiled ? TileGeometry.DepthPitch(description.Extent.Width, 1) : description.Pitch;
            if (plan.Tiled)
            {
                AddPlaneTransfers(plan.StencilTiles, TileBlockKind.Depth64KB, 1, plan.StencilPitch,
                    description.Extent.Width, description.Extent.Height, layers, plan.StencilSliceSize, 0, 0);
            }
        }

        return plan;
    }

    // One 64 KiB depth block transfer per array layer; the layer rides in SurfaceZ because
    // the depth kind folds it into every element offset.
    private static void AddPlaneTransfers(List<MetalTileTransfer> transfers, TileBlockKind kind, uint bytes, uint pitch, uint width, uint height, uint layers, ulong sliceSize, uint paddedWidth, uint paddedHeight)
    {
        if (!TileGeometry.TryGetBlockLayout(kind, bytes, out _))
        {
            throw GpuFatal.Fatal($"The depth tile layout is not supported: bytesPerElement={bytes}.");
        }

        transfers.Clear();
        for (uint layer = 0; layer < layers; layer++)
        {
            transfers.Add(new MetalTileTransfer
            {
                Kind = kind,
                BytesPerElement = bytes,
                LinearOffset = sliceSize * layer,
                LinearSize = sliceSize,
                TiledOffset = sliceSize * layer,
                TiledSize = sliceSize,
                Width = width,
                Height = height,
                Depth = 1,
                Pitch = pitch,
                SurfaceZ = kind is TileBlockKind.Depth64KB or TileBlockKind.RenderTarget64KB or TileBlockKind.RenderTarget64KBGen5 ? layer : 0,
                TiledWidth = paddedWidth,
                TiledHeight = paddedHeight,
                Slice = layer,
            });
        }
    }

    // The layers of one piece of a depth image. A depth plan is always a single level
    // with equal-sized layers (PlanDepth enforces both), so a piece is a run of layers and
    // guestBase has to be exactly where that run starts.
    // Which layers of a depth surface a piece carries. A depth plane is one 64 KiB
    // block layout per layer and the kernels walk it a layer at a time, so a piece is
    // named by its bytes and the layers are read back out of them - a caller that cuts
    // by subresource and one that cuts by guest run then agree.
    private static (uint Base, uint Count) DepthLayers(
        TransferPlan plan, SubresourceRange range, ulong guestBase, ulong pieceSize)
    {
        if (plan.SliceSize == 0)
        {
            return (0, 0);
        }

        if (guestBase % plan.SliceSize != 0)
        {
            throw GpuFatal.Fatal(
                $"A depth piece does not start on a layer boundary: base=0x{guestBase:X} " +
                $"sliceSize=0x{plan.SliceSize:X}.");
        }

        var first = Math.Max(Math.Min(range.BaseLayer, plan.Layers), (uint)(guestBase / plan.SliceSize));
        if (first >= plan.Layers)
        {
            return (first, 0);
        }

        var lastInRange = range.LayerCount == 0
            ? first
            : Math.Min(plan.Layers, range.BaseLayer + range.LayerCount);
        var count = first < lastInRange ? lastInRange - first : 0;
        if (pieceSize != 0)
        {
            // Rounded up: a whole-surface piece whose size falls a little short of the
            // last layer's padding still carries that layer, which is what naming the
            // layers by subresource alone used to do.
            count = Math.Min(count, (uint)((pieceSize + plan.SliceSize - 1) / plan.SliceSize));
        }

        return (first, count);
    }

    // The stencil plane has its own tile list; rebase it onto the piece the way Select
    // rebases the depth plane's.
    private List<MetalTileTransfer> SelectStencilTiles(TransferPlan plan, uint firstLayer, uint layerCount)
    {
        _stencilTiles.Clear();
        var guestBase = plan.StencilSliceSize * firstLayer;
        var lastLayer = firstLayer + layerCount - 1;
        var linearBase = ulong.MaxValue;
        foreach (var transfer in plan.StencilTiles)
        {
            if (transfer.Slice >= firstLayer && transfer.Slice <= lastLayer)
            {
                linearBase = Math.Min(linearBase, transfer.LinearOffset);
            }
        }

        if (linearBase == ulong.MaxValue)
        {
            return _stencilTiles;
        }

        foreach (var transfer in plan.StencilTiles)
        {
            if (transfer.Slice < firstLayer || transfer.Slice > lastLayer)
            {
                continue;
            }

            var rebased = transfer;
            rebased.TiledOffset = Rebase(transfer.TiledOffset, guestBase, "a stencil tile transfer");
            rebased.LinearOffset = transfer.LinearOffset - linearBase;
            _stencilTiles.Add(rebased);
        }

        return _stencilTiles;
    }

    private void EncodeDepthUpload(IMetalSubmissionContext context, in ImageDescription description, TransferPlan plan,
        SubresourceRange range, MetalGuestRange source, MetalGuestRange stencilSource, nint texture, ulong guestBase,
        ulong pieceSize)
    {
        var (firstLayer, layerCount) = DepthLayers(plan, range, guestBase, pieceSize);
        if (layerCount == 0)
        {
            return;
        }

        Select(plan, description, range, guestBase, pieceSize);
        var depthPiece = plan.SliceSize * layerCount;
        var stencilPiece = plan.StencilSliceSize * layerCount;
        var depthPlane = new MetalBufferSpan(source.Buffer, source.Offset, depthPiece);
        var stencilPlane = plan.StencilPlane && stencilSource.IsValid && stencilSource.Size >= stencilPiece
            ? new MetalBufferSpan(stencilSource.Buffer, stencilSource.Offset, stencilPiece)
            : default;
        nint compute = 0;
        if (plan.Tiled)
        {
            compute = _tiler.BeginCompute(context);
            var detiled = _tiler.Acquire(context, depthPiece);
            _tiler.EncodeDetile(compute, depthPlane, detiled, _tiles);
            depthPlane = detiled;
            if (stencilPlane.IsValid)
            {
                var stencilTiles = SelectStencilTiles(plan, firstLayer, layerCount);
                var detiledStencil = _tiler.Acquire(context, stencilPiece);
                _tiler.EncodeDetile(compute, stencilPlane, detiledStencil, stencilTiles);
                stencilPlane = detiledStencil;
            }
        }

        var depthRowBytes = (ulong)description.Pitch * plan.TransferBytes;
        var depthSliceBytes = depthRowBytes * description.Extent.Height;
        if (plan.TransferBytes != description.BytesPerBlock)
        {
            compute = compute != 0 ? compute : _tiler.BeginCompute(context);
            var widened = _tiler.Acquire(context, depthSliceBytes * layerCount);
            _tiler.EncodeDepthConversion(compute, depthPlane, widened, MetalDepthConversion.Widen, float32: true,
                new MetalDepthConversionLayout
                {
                    Width = description.Extent.Width,
                    Height = description.Extent.Height,
                    Layers = layerCount,
                    SourceRowStride = (ulong)description.Pitch * description.BytesPerBlock,
                    TargetRowStride = depthRowBytes,
                    SourceSliceStride = plan.SliceSize,
                    TargetSliceStride = depthSliceBytes,
                });
            depthPlane = widened;
        }
        else
        {
            depthSliceBytes = plan.SliceSize;
        }

        if (compute != 0)
        {
            MetalTiler.EndEncoding(compute);
        }

        var blit = _tiler.BeginBlit(context);
        var depthOption = MetalGuestFormats.HasStencilPlane(plan.Format)
            ? MtlBlitOption.DepthFromDepthStencil
            : MtlBlitOption.None;
        for (uint layer = 0; layer < layerCount; layer++)
        {
            MetalNative.SendCopyBufferToTextureOptions(blit, SelCopyBufferToTexture,
                depthPlane.Buffer, (nuint)(depthPlane.Offset + depthSliceBytes * layer),
                (nuint)depthRowBytes, (nuint)depthSliceBytes,
                new MtlSize { Width = description.Extent.Width, Height = description.Extent.Height, Depth = 1 },
                texture, firstLayer + layer, 0, default, (nuint)depthOption);
        }

        if (stencilPlane.IsValid)
        {
            for (uint layer = 0; layer < layerCount; layer++)
            {
                MetalNative.SendCopyBufferToTextureOptions(blit, SelCopyBufferToTexture,
                    stencilPlane.Buffer, (nuint)(stencilPlane.Offset + plan.StencilSliceSize * layer),
                    plan.StencilPitch, (nuint)plan.StencilSliceSize,
                    new MtlSize { Width = description.Extent.Width, Height = description.Extent.Height, Depth = 1 },
                    texture, firstLayer + layer, 0, default, (nuint)MtlBlitOption.StencilFromDepthStencil);
            }
        }

        MetalTiler.EndEncoding(blit);
    }

    // The way out of a tiled depth surface: one compute pass reads the depth texture and
    // tiles the guest plane straight out of it, so the plane never lands in a staging
    // buffer and the widening the upload did is undone inside the same kernel. The
    // stencil plane still takes a blit, because its only route out of a combined
    // depth-stencil texture is MTLBlitOptionStencilFromDepthStencil: reading it in a
    // shader would need an X32_Stencil8 view, and the PixelFormatView usage that view
    // asks for is what turns off the attachment's lossless compression while it renders.
    private void EncodeTiledDepthDownload(IMetalSubmissionContext context, in ImageDescription description,
        TransferPlan plan, SubresourceRange range, nint texture, MetalGuestRange destination,
        MetalGuestRange stencilDestination, ulong guestBase, ulong pieceSize)
    {
        var (firstLayer, layerCount) = DepthLayers(plan, range, guestBase, pieceSize);
        if (layerCount == 0)
        {
            return;
        }

        // The selected tiles carry the texture slice they came from and guest offsets
        // rebased onto this piece, so the kernel reads the right array slices and writes
        // inside the piece that was mapped.
        Select(plan, description, range, guestBase, pieceSize);
        var depthPiece = plan.SliceSize * layerCount;
        var stencilPiece = plan.StencilSliceSize * layerCount;
        var stencilWanted = plan.StencilPlane && stencilDestination.IsValid && stencilDestination.Size >= stencilPiece;
        var stencilPlane = stencilWanted ? _tiler.Acquire(context, stencilPiece) : default;
        if (stencilWanted)
        {
            var blit = _tiler.BeginBlit(context);
            for (uint layer = 0; layer < layerCount; layer++)
            {
                MetalNative.SendCopyTextureToBufferOptions(blit, SelCopyTextureToBuffer,
                    texture, firstLayer + layer, 0, default,
                    new MtlSize { Width = description.Extent.Width, Height = description.Extent.Height, Depth = 1 },
                    stencilPlane.Buffer, (nuint)(stencilPlane.Offset + plan.StencilSliceSize * layer),
                    plan.StencilPitch, (nuint)plan.StencilSliceSize, (nuint)MtlBlitOption.StencilFromDepthStencil);
            }

            MetalTiler.EndEncoding(blit);
        }

        if (plan.BlockKind != TileBlockKind.Depth64KB)
        {
            EncodeStagedTiledDepthDownload(context, description, plan, texture, destination, firstLayer, layerCount);
            return;
        }

        var compute = _tiler.BeginCompute(context);
        _tiler.EncodeDepthTextureWriteback(compute, texture,
            new MetalBufferSpan(destination.Buffer, destination.Offset, depthPiece),
            description.BytesPerBlock, plan.Layers > 1, _tiles);
        if (stencilWanted)
        {
            _tiler.EncodeTile(compute, stencilPlane,
                new MetalBufferSpan(stencilDestination.Buffer, stencilDestination.Offset, stencilPiece),
                SelectStencilTiles(plan, firstLayer, layerCount));
        }

        MetalTiler.EndEncoding(compute);
    }

    // The fused writeback only has Depth64KB offset functions, so any other tile mode
    // goes out the way the linear download does: texture to a linear staging plane, then
    // tiled out of it.
    private void EncodeStagedTiledDepthDownload(IMetalSubmissionContext context, in ImageDescription description,
        TransferPlan plan, nint texture, MetalGuestRange destination, uint firstLayer, uint layerCount)
    {
        var depthPiece = plan.SliceSize * layerCount;
        var widening = plan.TransferBytes != description.BytesPerBlock;
        var depthRowBytes = (ulong)description.Pitch * plan.TransferBytes;
        var depthSliceBytes = widening ? depthRowBytes * description.Extent.Height : plan.SliceSize;
        var staging = _tiler.Acquire(context, depthSliceBytes * layerCount);
        var blit = _tiler.BeginBlit(context);
        var depthOption = MetalGuestFormats.HasStencilPlane(plan.Format)
            ? MtlBlitOption.DepthFromDepthStencil
            : MtlBlitOption.None;
        for (uint layer = 0; layer < layerCount; layer++)
        {
            MetalNative.SendCopyTextureToBufferOptions(blit, SelCopyTextureToBuffer,
                texture, firstLayer + layer, 0, default,
                new MtlSize { Width = description.Extent.Width, Height = description.Extent.Height, Depth = 1 },
                staging.Buffer, (nuint)(staging.Offset + depthSliceBytes * layer),
                (nuint)depthRowBytes, (nuint)depthSliceBytes, (nuint)depthOption);
        }

        MetalTiler.EndEncoding(blit);
        var compute = _tiler.BeginCompute(context);
        if (widening)
        {
            var narrowed = _tiler.Acquire(context, depthPiece);
            _tiler.EncodeDepthConversion(compute, staging, narrowed, MetalDepthConversion.Narrow, float32: true,
                new MetalDepthConversionLayout
                {
                    Width = description.Extent.Width,
                    Height = description.Extent.Height,
                    Layers = layerCount,
                    SourceRowStride = depthRowBytes,
                    TargetRowStride = (ulong)description.Pitch * description.BytesPerBlock,
                    SourceSliceStride = depthSliceBytes,
                    TargetSliceStride = plan.SliceSize,
                });
            staging = narrowed;
        }

        _tiler.EncodeTile(compute, staging,
            new MetalBufferSpan(destination.Buffer, destination.Offset, depthPiece), _tiles);
        MetalTiler.EndEncoding(compute);
    }

    private void EncodeDepthDownload(IMetalSubmissionContext context, in ImageDescription description, TransferPlan plan,
        SubresourceRange range, nint texture, MetalGuestRange destination, MetalGuestRange stencilDestination,
        ulong guestBase, ulong pieceSize)
    {
        var (firstLayer, layerCount) = DepthLayers(plan, range, guestBase, pieceSize);
        if (layerCount == 0)
        {
            return;
        }

        if (plan.Tiled)
        {
            EncodeTiledDepthDownload(
                context, description, plan, range, texture, destination, stencilDestination, guestBase, pieceSize);
            return;
        }

        var depthPiece = plan.SliceSize * layerCount;
        var stencilPiece = plan.StencilSliceSize * layerCount;
        var widening = plan.TransferBytes != description.BytesPerBlock;
        var depthRowBytes = (ulong)description.Pitch * plan.TransferBytes;
        var depthSliceBytes = widening ? depthRowBytes * description.Extent.Height : plan.SliceSize;
        var direct = !widening && destination.Offset % plan.TransferBytes == 0;
        var depthPlane = direct
            ? new MetalBufferSpan(destination.Buffer, destination.Offset, depthPiece)
            : _tiler.Acquire(context, depthSliceBytes * layerCount);
        var stencilWanted = plan.StencilPlane && stencilDestination.IsValid && stencilDestination.Size >= stencilPiece;
        var stencilPlane = stencilWanted
            ? new MetalBufferSpan(stencilDestination.Buffer, stencilDestination.Offset, stencilPiece)
            : default;

        var blit = _tiler.BeginBlit(context);
        var depthOption = MetalGuestFormats.HasStencilPlane(plan.Format)
            ? MtlBlitOption.DepthFromDepthStencil
            : MtlBlitOption.None;
        for (uint layer = 0; layer < layerCount; layer++)
        {
            MetalNative.SendCopyTextureToBufferOptions(blit, SelCopyTextureToBuffer,
                texture, firstLayer + layer, 0, default,
                new MtlSize { Width = description.Extent.Width, Height = description.Extent.Height, Depth = 1 },
                depthPlane.Buffer, (nuint)(depthPlane.Offset + depthSliceBytes * layer),
                (nuint)depthRowBytes, (nuint)depthSliceBytes, (nuint)depthOption);
        }

        if (stencilPlane.IsValid)
        {
            for (uint layer = 0; layer < layerCount; layer++)
            {
                MetalNative.SendCopyTextureToBufferOptions(blit, SelCopyTextureToBuffer,
                    texture, firstLayer + layer, 0, default,
                    new MtlSize { Width = description.Extent.Width, Height = description.Extent.Height, Depth = 1 },
                    stencilPlane.Buffer, (nuint)(stencilPlane.Offset + plan.StencilSliceSize * layer),
                    plan.StencilPitch, (nuint)plan.StencilSliceSize, (nuint)MtlBlitOption.StencilFromDepthStencil);
            }
        }

        MetalTiler.EndEncoding(blit);
        if (direct)
        {
            return;
        }

        var compute = _tiler.BeginCompute(context);
        if (widening)
        {
            var narrowed = _tiler.Acquire(context, depthPiece);
            _tiler.EncodeDepthConversion(compute, depthPlane, narrowed, MetalDepthConversion.Narrow, float32: true,
                new MetalDepthConversionLayout
                {
                    Width = description.Extent.Width,
                    Height = description.Extent.Height,
                    Layers = layerCount,
                    SourceRowStride = depthRowBytes,
                    TargetRowStride = (ulong)description.Pitch * description.BytesPerBlock,
                    SourceSliceStride = depthSliceBytes,
                    TargetSliceStride = plan.SliceSize,
                });
            depthPlane = narrowed;
        }

        _tiler.EncodeRepackRows(compute, depthPlane,
            new MetalBufferSpan(destination.Buffer, destination.Offset, depthPiece),
            (uint)((ulong)description.Pitch * description.BytesPerBlock), description.Extent.Height, layerCount,
            (ulong)description.Pitch * description.BytesPerBlock,
            (ulong)description.Pitch * description.BytesPerBlock,
            plan.SliceSize, plan.SliceSize);
        MetalTiler.EndEncoding(compute);
    }
}
