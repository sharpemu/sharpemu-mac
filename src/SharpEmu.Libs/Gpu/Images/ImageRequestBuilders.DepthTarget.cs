// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Scheduling;

namespace SharpEmu.Libs.Gpu.Images;

// A resolved depth target: the request plus the planes, sizes and clear enables of the attachment.
public readonly record struct DepthTargetResolution(
    ImageRequest Request,
    GpuFormat Format,
    uint Width,
    uint Height,
    uint Samples,
    bool HasStencil,
    bool HasHtile,
    ulong DepthAddress,
    ulong DepthSize,
    ulong StencilAddress,
    ulong StencilSize,
    ulong HtileAddress,
    ulong HtileSize,
    bool DepthClearEnabled,
    bool StencilClearEnabled,
    bool DepthWriteDisabled,
    bool StencilWriteDisabled);

public static partial class ImageRequestBuilders
{
    public static GpuImageUsage DepthTargetUsage =>
        GpuImageUsage.DepthStencilAttachmentBit | GpuImageUsage.SampledBit | GpuImageUsage.TransferSrcBit | GpuImageUsage.TransferDstBit;

    // The first host format of the rule the device supports at the sample count, or Undefined.
    private static GpuFormat HostDepthAttachmentFormat(IImageFormatSupport device, DepthFormatRule rule, bool hasStencil, uint samples)
    {
        var required = ImageDescription.VulkanSampleCount(samples);
        bool Supports(GpuFormat format) =>
            format != GpuFormat.Undefined &&
            device.TryGetImageFormatProperties(format, GpuImageType.Type2D, GpuImageTiling.Optimal, DepthTargetUsage, 0, out var properties) &&
            (properties.SampleCounts & required) != 0;

        if (!hasStencil)
        {
            return Supports(rule.DepthAttachmentFormat) ? rule.DepthAttachmentFormat : GpuFormat.Undefined;
        }

        foreach (var format in rule.StencilAttachmentFormats)
        {
            if (Supports(format))
            {
                return format;
            }
        }

        return GpuFormat.Undefined;
    }

    // A guest that leaves DB_STENCIL_READ_BASE at zero while the write base is set reads stencil
    // through the write-base plane, as it does for depth, whose read and write bases are equal.
    // Demon's Souls ships a register table with no read-base entry.
    private static bool ReadsThroughWriteBase(in DepthTargetWords depthWords) =>
        depthWords.StencilReadBase == 0 &&
        depthWords.StencilWriteBase != 0 &&
        !depthWords.StencilWriteDisabled;

    private static int _stencilReadBaseMissingTraced;

    // Builds the request for the bound depth target. Null when no depth or stencil state is active.
    // Reused while the words and the device are the same, as ColorTarget's requests are.
    public static DepthTargetResolution? DepthTarget(in DepthTargetWords depthWords, IImageFormatSupport device)
    {
        if (Rendering.RenderTrace.Enabled)
        {
            return DepthTargetCore(in depthWords, device, copyMode: false, writeBuffer: false);
        }

        if (!SharpEmu.ShaderCompiler.WorkerScratch.Enabled)
        {
            if (_depthTargetMemo is { } memo && ReferenceEquals(memo.Device, device) && memo.Words.Equals(depthWords))
            {
                return memo.Result;
            }

            var built = DepthTargetCore(in depthWords, device, copyMode: false, writeBuffer: false);
            _depthTargetMemo = new DepthTargetMemo(depthWords, device, built);
            return built;
        }

        // A frame alternates between a few depth states (passes), which a one-entry memo missed
        // and refilled with a new object each time; the slots are structs, as ColorTarget's are.
        var memos = _depthTargetMemos ??= new DepthTargetSlot[DepthMemoCount];
        ref var slot = ref memos[(int)((depthWords.ZReadBase >> 12) ^ (depthWords.ZWriteBase >> 12) ^ (depthWords.DepthControl * 31u)) & (DepthMemoCount - 1)];
        if (slot.Valid && ReferenceEquals(slot.Device, device) && slot.Words.Equals(depthWords))
        {
            return slot.Result;
        }

        var result = DepthTargetCore(in depthWords, device, copyMode: false, writeBuffer: false);
        slot = new DepthTargetSlot { Valid = true, Words = depthWords, Device = device, Result = result };
        return result;
    }

    private sealed record DepthTargetMemo(DepthTargetWords Words, IImageFormatSupport Device, DepthTargetResolution? Result);

    [ThreadStatic]
    private static DepthTargetMemo? _depthTargetMemo;

    private struct DepthTargetSlot
    {
        public bool Valid;
        public DepthTargetWords Words;
        public IImageFormatSupport Device;
        public DepthTargetResolution? Result;
    }

    private const int DepthMemoCount = 8;

    [ThreadStatic]
    private static DepthTargetSlot[]? _depthTargetMemos;

    // Builds the source or destination image used by DB_RENDER_OVERRIDE depth/stencil copies.
    public static DepthTargetResolution? DepthTargetCopy(in DepthTargetWords depthWords, IImageFormatSupport device, bool writeBuffer) =>
        DepthTargetCore(in depthWords, device, copyMode: true, writeBuffer);

    private static DepthTargetResolution? DepthTargetCore(in DepthTargetWords depthWords, IImageFormatSupport device, bool copyMode, bool writeBuffer)
    {
        var hasStencil = depthWords.StencilFormat != GuestStencilFormat.Invalid;
        if (!copyMode)
        {
            var depthActive = depthWords.DepthTestEnabled || depthWords.DepthWriteEnabled || depthWords.DepthBoundsEnabled || depthWords.DepthClearEnabled || depthWords.CopyDepthToColor;
            var stencilActive = hasStencil && (depthWords.StencilTestEnabled || depthWords.StencilClearEnabled || depthWords.CopyStencilToColor);
            if (!depthActive && !stencilActive)
            {
                if (Rendering.RenderTrace.Enabled)
                {
                    Rendering.RenderTrace.Write("DepthAttachmentRejected reason=depth-and-stencil-inactive");
                }
                return null;
            }
        }

        var depthAddress = copyMode && writeBuffer ? depthWords.ZWriteBase : depthWords.ZReadBase;
        var stencilAddress = copyMode && writeBuffer ? depthWords.StencilWriteBase : depthWords.StencilReadBase;
        var stencilReadsThroughWriteBase = !copyMode && hasStencil && ReadsThroughWriteBase(in depthWords);
        if (stencilReadsThroughWriteBase)
        {
            if (Rendering.RenderTrace.Enabled && System.Threading.Interlocked.Exchange(ref _stencilReadBaseMissingTraced, 1) == 0)
            {
                Rendering.RenderTrace.Write($"StencilReadBaseMissing write=0x{depthWords.StencilWriteBase:X}");
            }
            stencilAddress = depthWords.StencilWriteBase;
        }

        // The size register is independent state; a zero encoding alone must not produce an attachment.
        var attachmentUnbound =
            depthWords.DepthFormat == GuestDepthFormat.Invalid && !hasStencil && depthWords.SamplesLog2 == 0 && !depthWords.ZExpClear && !depthWords.ZPartiallyResident && depthWords.MaxMip == 0 &&
            !depthWords.StencilExpClear && !depthWords.StencilPartiallyResident && depthWords.SliceStart == 0 && depthWords.SliceMax == 0 && depthWords.ViewMipLevel == 0 &&
            !depthWords.DepthWriteDisabled && !depthWords.StencilWriteDisabled && depthWords.ZReadBase == 0 && depthWords.ZWriteBase == 0 && depthWords.StencilReadBase == 0 &&
            depthWords.StencilWriteBase == 0 && depthWords.HtileBase == 0 && !depthWords.HtileAcceleration && depthWords.ShadingRateEncoding == 0 && depthWords.XMax == 0 && depthWords.YMax == 0;
        if (attachmentUnbound)
        {
            if (Rendering.RenderTrace.Enabled)
            {
                Rendering.RenderTrace.Write("DepthAttachmentRejected reason=unbound-register-state");
            }
            return null;
        }

        var hasHtile = depthWords.HtileAcceleration;
        var unsupportedShadingRate = depthWords.ShadingRateEncoding > 1 || (depthWords.ShadingRateEncoding != 0 && !hasHtile);
        var samples = SampleCount(depthWords.SamplesLog2);
        if (samples == 0)
        {
            throw GpuFatal.Fatal($"The depth fragment count is not supported: samples={depthWords.SamplesLog2}.");
        }

        if (!TargetViewRange.TryResolve(depthWords.SliceStart, depthWords.SliceMax, 0, out var view))
        {
            throw GpuFatal.Fatal($"The depth view is invalid: base={depthWords.SliceStart} last={depthWords.SliceMax}.");
        }

        // EXPCLEAR only changes how the hardware reads the state of a tile in HTILE; the image cache keeps a clear flag per
        // slice and no per-tile HTILE state, so the bits do not change what is rendered.
        if (depthWords.CopyDepthToColor || depthWords.CopyStencilToColor || depthWords.CopyCentroid || depthWords.CopySample != 0 ||
            depthWords.MaxMip != 0 || depthWords.ViewMipLevel != 0 || unsupportedShadingRate ||
            depthAddress == 0 || (!copyMode && !depthWords.DepthWriteDisabled && depthWords.ZWriteBase != depthWords.ZReadBase) || (depthAddress & 0xFFFF) != 0 || depthWords.DepthCompare > 7)
        {
            throw GpuFatal.Fatal(
                $"The depth register state is not supported: zInfo=0x{depthWords.ZInfo:X8} stencilInfo=0x{depthWords.StencilInfo:X8} view=0x{depthWords.DepthView:X8} " +
                $"renderControl=0x{depthWords.RenderControl:X8} shadingRate={depthWords.ShadingRateEncoding} read=0x{depthWords.ZReadBase:X16} write=0x{depthWords.ZWriteBase:X16}.");
        }

        if (hasStencil)
        {
            if (depthWords.StencilFormat != GuestStencilFormat.Stencil8UInt ||
                stencilAddress == 0 || (!copyMode && !stencilReadsThroughWriteBase && !depthWords.StencilWriteDisabled && depthWords.StencilWriteBase != depthWords.StencilReadBase) || (stencilAddress & 0xFFFF) != 0)
            {
                throw GpuFatal.Fatal(
                    $"The stencil attachment state is not supported: stencilInfo=0x{depthWords.StencilInfo:X8} htile={hasHtile} read=0x{depthWords.StencilReadBase:X16} write=0x{depthWords.StencilWriteBase:X16} " +
                    $"depthControl=0x{depthWords.DepthControl:X8} renderControl=0x{depthWords.RenderControl:X8} view=0x{depthWords.DepthView:X8}.");
            }
        }
        else if (!copyMode && (depthWords.StencilReadBase != 0 || depthWords.StencilWriteBase != 0))
        {
            throw GpuFatal.Fatal($"Stencil state is set without an active stencil attachment: read=0x{depthWords.StencilReadBase:X16} write=0x{depthWords.StencilWriteBase:X16}.");
        }

        if (hasHtile)
        {
            if (depthWords.HtileBase == 0 || (depthWords.HtileBase & 0x7FFF) != 0)
            {
                throw GpuFatal.Fatal($"The HTile metadata address is invalid: htile=0x{depthWords.HtileBase:X16}.");
            }

            if (depthWords.SliceMax >= 32)
            {
                throw GpuFatal.Fatal($"HTile clear tracking supports at most 32 slices: last={depthWords.SliceMax}.");
            }
        }

        if (!depthWords.DepthSizeValid)
        {
            throw GpuFatal.Fatal("The depth extent is missing.");
        }

        var width = depthWords.XMax + 1;
        var height = depthWords.YMax + 1;
        if (width > 16384 || height > 16384)
        {
            throw GpuFatal.Fatal($"The depth extent is invalid: extent={width}x{height}.");
        }

        var rule = DepthFormatRule.Find(depthWords.DepthFormat) ?? throw GpuFatal.Fatal($"The depth/stencil format pair is not supported: zFormat={(uint)depthWords.DepthFormat} stencil={hasStencil}.");
        var format = HostDepthAttachmentFormat(device, rule, hasStencil, samples);
        if (format == GpuFormat.Undefined)
        {
            throw GpuFatal.Fatal($"No host depth/stencil format supports the required usage: ideal={(int)rule.AttachmentFormat(hasStencil)} samples={samples}.");
        }

        var bytes = rule.BytesPerElement;
        var pitch = TileGeometry.DepthPitch(width, bytes, depthWords.SamplesLog2);
        if (!TileGeometry.TryGetDepthSize(width, height, 0, depthWords.DepthFormat, depthWords.StencilFormat, hasHtile, out var stencilSize, out var htileSize, out var depthSize, depthWords.SamplesLog2) ||
            depthSize.Align != 65536 || depthSize.Size == 0 ||
            hasStencil != (stencilSize.Align == 65536 && stencilSize.Size != 0) ||
            hasHtile != (htileSize.Align == 32768 && htileSize.Size != 0))
        {
            throw GpuFatal.Fatal(
                $"The depth/stencil/HTile footprint is not supported: extent={width}x{height} zFormat={(uint)depthWords.DepthFormat} stencil={hasStencil} htile={hasHtile} samplesLog2={depthWords.SamplesLog2}.");
        }

        if (depthSize.Size > ulong.MaxValue / view.ImageLayers || stencilSize.Size > ulong.MaxValue / view.ImageLayers || htileSize.Size > ulong.MaxValue / view.ImageLayers)
        {
            throw GpuFatal.Fatal($"The layered depth footprint overflows: layers={view.ImageLayers}.");
        }

        var depthBackingSize = (ulong)depthSize.Size * view.ImageLayers;
        var stencilBackingSize = (ulong)stencilSize.Size * view.ImageLayers;
        var htileBackingSize = (ulong)htileSize.Size * view.ImageLayers;
        if (!new GuestSpan(depthAddress, depthBackingSize).IsValid ||
            (hasStencil && !new GuestSpan(stencilAddress, stencilBackingSize).IsValid) ||
            (hasHtile && !new GuestSpan(depthWords.HtileBase, htileBackingSize).IsValid))
        {
            throw GpuFatal.Fatal(
                $"The layered depth backing range is invalid: depth=0x{depthAddress:X16}+0x{depthBackingSize:X} stencil=0x{stencilAddress:X16}+0x{stencilBackingSize:X} htile=0x{depthWords.HtileBase:X16}+0x{htileBackingSize:X}.");
        }

        stencilAddress = hasStencil ? stencilAddress : 0;
        stencilBackingSize = hasStencil ? stencilBackingSize : 0;
        var htileAddress = hasHtile ? depthWords.HtileBase : 0;
        htileBackingSize = hasHtile ? htileBackingSize : 0;
        var description = ImageDescription.Create();
        description.Data = new GuestSpan(depthAddress, depthBackingSize);
        description.Stencil = new GuestSpan(stencilAddress, stencilBackingSize);
        description.PixelFormat = format;
        description.GuestFormat = rule.GuestFormat;
        description.Type = GuestImageType.Color2D;
        description.Extent = new GpuExtent3D(width, height, 1);
        description.Resources = new SubresourceCount(1, view.ImageLayers);
        description.Pitch = pitch;
        description.BytesPerBlock = bytes;
        description.Samples = samples;
        description.TileMode = GuestTileMode.Depth;
        description.MipLayout[0] = new MipLevelLayout { Offset = 0, Size = depthBackingSize, Pitch = pitch, Height = height };
        description.Metadata.Range = new GuestSpan(htileAddress, htileBackingSize);
        description.Metadata.Kind = hasHtile ? MetadataKind.Htile : MetadataKind.None;
        description.Metadata.StencilCompressed = hasStencil && hasHtile && !depthWords.HtileStencilDisabled;
        var viewDescription = new ImageViewDescription(
            format, view.LayerCount == 1 ? GpuImageViewType.Type2D : GpuImageViewType.Type2DArray, ViewFormatRules.DepthAspects(format),
            0, 1, view.BaseLayer, view.LayerCount, default, GpuImageUsage.DepthStencilAttachmentBit);
        var request = new ImageRequest(description, viewDescription, ImageRole.DepthTarget);
        return new DepthTargetResolution(
            request, format, width, height, samples, hasStencil, hasHtile,
            depthAddress, depthBackingSize, stencilAddress, stencilBackingSize, htileAddress, htileBackingSize,
            depthWords.DepthClearEnabled, hasStencil && depthWords.StencilClearEnabled && !depthWords.StencilWriteDisabled, depthWords.DepthWriteDisabled, depthWords.StencilWriteDisabled);
    }

    // Builds the request for a video-out surface registered by the guest.
    public static ImageRequest DisplaySurface(in DisplaySurfaceWords surface)
    {
        const ulong SupportedOptions = (1UL << 3) | (1UL << 5); // Strict colorimetry and premultiplied alpha.
        var compression = ImageDescription.ClassifyDisplayCompression(surface.Compressed, surface.MetadataAddress, surface.DccControl, surface.DccClearColor);
        if (surface.Width == 0 || surface.Height == 0 || surface.Width > 16384 || surface.Height > 16384 ||
            (surface.Option & ~SupportedOptions) != 0 || surface.TilingMode != 0 || surface.DataAddress == 0 ||
            compression == DisplayCompression.Unsupported)
        {
            throw GpuFatal.Fatal(
                $"The video-out surface attributes are not supported: address=0x{surface.DataAddress:X16} extent={surface.Width}x{surface.Height} tiling={surface.TilingMode} " +
                $"option=0x{surface.Option:X} compressed={surface.Compressed} metadata=0x{surface.MetadataAddress:X16} dccControl=0x{surface.DccControl:X8}.");
        }

        if (!DisplayFormatRule.TryDecode(surface.PixelFormat, out var pixelFormat))
        {
            throw GpuFatal.Fatal($"The video-out pixel format is not supported: format=0x{surface.PixelFormat:X16}.");
        }

        const GuestTileMode tileMode = GuestTileMode.RenderTarget;
        var pitch = TileGeometry.TexturePitch(pixelFormat.GuestFormat, surface.Width, tileMode);
        var total = TileGeometry.TextureTotalSize(pixelFormat.GuestFormat, surface.Width, surface.Height, 1, 1, tileMode, false);
        if (total.Size == 0 || total.Align != 65536 || (surface.DataAddress & (total.Align - 1UL)) != 0)
        {
            throw GpuFatal.Fatal($"The video-out surface footprint or alignment is invalid: address=0x{surface.DataAddress:X16} size=0x{total.Size:X} align=0x{total.Align:X}.");
        }

        var description = ImageDescription.Create();
        description.Data = new GuestSpan(surface.DataAddress, total.Size);
        description.PixelFormat = pixelFormat.HostFormat;
        description.GuestFormat = pixelFormat.GuestFormat;
        description.Type = GuestImageType.Color2D;
        description.Extent = new GpuExtent3D(surface.Width, surface.Height, 1);
        description.Resources = SubresourceCount.Single;
        description.Pitch = pitch;
        description.BytesPerBlock = pixelFormat.BytesPerElement;
        description.Samples = 1;
        description.TileMode = tileMode;
        description.Bgra16 = pixelFormat.Bgra16;
        description.MipLayout[0] = new MipLevelLayout { Offset = 0, Size = total.Size, Pitch = pitch, Height = surface.Height };
        if (compression != DisplayCompression.Uncompressed)
        {
            description.Metadata.Range = new GuestSpan(surface.MetadataAddress, 0);
            description.Metadata.Kind = MetadataKind.Dcc;
            description.Metadata.Control = surface.DccControl;
            description.Metadata.Compression = compression;
        }

        description.Validate();
        if (!DisplayFormatRule.Supports(description))
        {
            throw GpuFatal.Fatal($"The normalized video-out format is not supported: format={(int)description.PixelFormat} guestFormat={(uint)description.GuestFormat}.");
        }

        var view = new ImageViewDescription(
            description.PixelFormat, GpuImageViewType.Type2D, GpuImageAspect.ColorBit, 0, 1, 0, 1, default, GpuImageUsage.TransferSrcBit);
        return new ImageRequest(description, view, ImageRole.DisplaySurface);
    }
}
