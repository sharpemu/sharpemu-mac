// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler;
using SharpEmu.Libs.Gpu;
using SharpEmu.Libs.Gpu.Abstractions;

namespace SharpEmu.Gpu.Metal;

/// <summary>
/// MTLPixelFormat raw values — only the formats the backend maps. Declared here
/// rather than pulled from a binding package: the Metal backend talks to the OS
/// exclusively through objc_msgSend, so ABI constants are owned locally.
/// </summary>
internal enum MtlPixelFormat : uint
{
    Invalid = 0,
    A8Unorm = 1,
    R8Unorm = 10,
    R8UnormSrgb = 11,
    R8Snorm = 12,
    R8Uint = 13,
    R8Sint = 14,
    R16Unorm = 20,
    R16Snorm = 22,
    R16Uint = 23,
    R16Sint = 24,
    R16Float = 25,
    Rg8Unorm = 30,
    Rg8UnormSrgb = 31,
    Rg8Snorm = 32,
    Rg8Uint = 33,
    Rg8Sint = 34,
    B5G6R5Unorm = 40,
    A1Bgr5Unorm = 41,
    Abgr4Unorm = 42,
    Bgr5A1Unorm = 43,
    R32Uint = 53,
    R32Sint = 54,
    R32Float = 55,
    Rg16Unorm = 60,
    Rg16Snorm = 62,
    Rg16Uint = 63,
    Rg16Sint = 64,
    Rg16Float = 65,
    Rgba8Unorm = 70,
    Rgba8UnormSrgb = 71,
    Rgba8Snorm = 72,
    Rgba8Uint = 73,
    Rgba8Sint = 74,
    Bgra8Unorm = 80,
    Bgra8UnormSrgb = 81,
    Rgb10A2Unorm = 90,
    Rgb10A2Uint = 91,
    Rg11B10Float = 92,
    Rgb9E5Float = 93,
    Bgr10A2Unorm = 94,
    Rg32Uint = 103,
    Rg32Sint = 104,
    Rg32Float = 105,
    Rgba16Unorm = 110,
    Rgba16Snorm = 112,
    Rgba16Uint = 113,
    Rgba16Sint = 114,
    Rgba16Float = 115,
    Rgba32Uint = 123,
    Rgba32Sint = 124,
    Rgba32Float = 125,
    Bc1Rgba = 130,
    Bc1RgbaSrgb = 131,
    Bc2Rgba = 132,
    Bc2RgbaSrgb = 133,
    Bc3Rgba = 134,
    Bc3RgbaSrgb = 135,
    Bc4RUnorm = 140,
    Bc4RSnorm = 141,
    Bc5RgUnorm = 142,
    Bc5RgSnorm = 143,
    Bc6HRgbFloat = 150,
    Bc6HRgbUfloat = 151,
    Bc7RgbaUnorm = 152,
    Bc7RgbaUnormSrgb = 153,
    Depth16Unorm = 250,
    Depth32Float = 252,
    Stencil8 = 253,
    Depth32FloatStencil8 = 260,
    X32Stencil8 = 261,
}

/// <summary>A sampled-texture format: the Metal pixel format plus the byte
/// layout the upload path needs. <see cref="BlockBytes"/> is nonzero for
/// block-compressed formats (bytes per 4x4 block); otherwise
/// <see cref="BytesPerPixel"/> applies.</summary>
internal readonly record struct MetalTextureFormat(
    MtlPixelFormat Format,
    uint BytesPerPixel,
    uint BlockBytes)
{
    public bool IsBlockCompressed => BlockBytes != 0;
}

internal readonly record struct MetalRenderTargetFormat(
    MtlPixelFormat Format,
    Gen5PixelOutputKind OutputKind,
    Gen5ColorComponentMapping ExportMapping)
{
    public MetalRenderTargetFormat(
        MtlPixelFormat format,
        Gen5PixelOutputKind outputKind)
        : this(format, outputKind, Gen5ColorComponentMapping.Identity)
    {
    }

    public static uint GetBytesPerPixel(MtlPixelFormat format) =>
        format switch
        {
            MtlPixelFormat.R8Unorm or MtlPixelFormat.R8Snorm or MtlPixelFormat.R8Uint => 1,
            MtlPixelFormat.Rg8Unorm or MtlPixelFormat.Rg8Snorm => 2,
            MtlPixelFormat.Rg32Float => 8,
            MtlPixelFormat.Rgba16Unorm or MtlPixelFormat.Rgba16Snorm or MtlPixelFormat.Rgba16Uint or
                MtlPixelFormat.Rgba16Sint or MtlPixelFormat.Rgba16Float => 8,
            MtlPixelFormat.Rgba32Float => 16,
            _ => 4,
        };
}

/// <summary>
/// Guest texture-descriptor codes to Metal formats, mirroring the Vulkan
/// backend's table case for case so both backends accept the same guest
/// formats. Guest format 9 (2:10:10:10) maps to BGR10A2 — the bit layout that
/// matches Vulkan's A2R10G10B10 pack.
/// </summary>
internal static class MetalGuestFormats
{
    /// <summary>Guest sampled-texture format to Metal, mirroring the Vulkan
    /// backend's implemented mappings. Unknown codes fail explicitly instead
    /// of reinterpreting unrelated source bytes as RGBA8. BC formats upload
    /// raw blocks — Mac-family GPUs decode them
    /// natively.</summary>
    public static MetalTextureFormat DecodeTextureFormat(uint dataFormat, uint numberType)
    {
        var format = (dataFormat, numberType) switch
        {
            (1, 0) => MtlPixelFormat.R8Unorm,
            (1, 1) => MtlPixelFormat.R8Snorm,
            (1, 4) => MtlPixelFormat.R8Uint,
            (1, 5) => MtlPixelFormat.R8Sint,
            (2, 0) => MtlPixelFormat.R16Unorm,
            (2, 1) => MtlPixelFormat.R16Snorm,
            (2, 4) => MtlPixelFormat.R16Uint,
            (2, 5) => MtlPixelFormat.R16Sint,
            (2, 7) => MtlPixelFormat.R16Float,
            (3, 0) => MtlPixelFormat.Rg8Unorm,
            (3, 1) => MtlPixelFormat.Rg8Snorm,
            (3, 4) => MtlPixelFormat.Rg8Uint,
            (3, 5) => MtlPixelFormat.Rg8Sint,
            (4, 4) => MtlPixelFormat.R32Uint,
            (4, 5) => MtlPixelFormat.R32Sint,
            (4, 7) => MtlPixelFormat.R32Float,
            (5, 0) => MtlPixelFormat.Rg16Unorm,
            (5, 4) => MtlPixelFormat.Rg16Uint,
            (5, 5) => MtlPixelFormat.Rg16Sint,
            (5, 7) => MtlPixelFormat.Rg16Float,
            (6, 7) or (7, 7) => MtlPixelFormat.Rg11B10Float,
            (8, _) or (9, _) => MtlPixelFormat.Bgr10A2Unorm,
            (10, 0) => MtlPixelFormat.Rgba8Unorm,
            (10, 4) => MtlPixelFormat.Rgba8Uint,
            (10, 5) => MtlPixelFormat.Rgba8Sint,
            (10, 9) => MtlPixelFormat.Rgba8UnormSrgb,
            (11, 4) => MtlPixelFormat.Rg32Uint,
            (11, 5) => MtlPixelFormat.Rg32Sint,
            (11, 7) => MtlPixelFormat.Rg32Float,
            (12, 0) => MtlPixelFormat.Rgba16Unorm,
            (12, 4) => MtlPixelFormat.Rgba16Uint,
            (12, 5) => MtlPixelFormat.Rgba16Sint,
            (12, 7) => MtlPixelFormat.Rgba16Float,
            // Guest RGB32 is 12 bytes, not Metal RGBA32's 16. It requires
            // an explicit repack before a native upload can preserve pixels.
            (14, 4) => MtlPixelFormat.Rgba32Uint,
            (14, 5) => MtlPixelFormat.Rgba32Sint,
            (14, _) => MtlPixelFormat.Rgba32Float,
            (16, 0) => MtlPixelFormat.B5G6R5Unorm,
            (34, 7) => MtlPixelFormat.Rgb9E5Float,
            (169, _) => MtlPixelFormat.Bc1Rgba,
            (170, _) => MtlPixelFormat.Bc1RgbaSrgb,
            (171, _) => MtlPixelFormat.Bc2Rgba,
            (172, _) => MtlPixelFormat.Bc2RgbaSrgb,
            (173, _) => MtlPixelFormat.Bc3Rgba,
            (174, _) => MtlPixelFormat.Bc3RgbaSrgb,
            (175, 1) or (176, _) => MtlPixelFormat.Bc4RSnorm,
            (175, _) => MtlPixelFormat.Bc4RUnorm,
            (177, 1) or (178, _) => MtlPixelFormat.Bc5RgSnorm,
            (177, _) => MtlPixelFormat.Bc5RgUnorm,
            (179, _) => MtlPixelFormat.Bc6HRgbUfloat,
            (180, _) => MtlPixelFormat.Bc6HRgbFloat,
            (181, _) => MtlPixelFormat.Bc7RgbaUnorm,
            (182, _) => MtlPixelFormat.Bc7RgbaUnormSrgb,
            _ => throw new NotSupportedException($"Metal guest texture format {dataFormat}/{numberType} is unsupported."),
        };

        var blockBytes = format switch
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

        var bytesPerPixel = format switch
        {
            MtlPixelFormat.R8Unorm or MtlPixelFormat.R8Snorm or
                MtlPixelFormat.R8Uint or MtlPixelFormat.R8Sint => 1u,
            MtlPixelFormat.R16Unorm or MtlPixelFormat.R16Snorm or
                MtlPixelFormat.R16Uint or MtlPixelFormat.R16Sint or
                MtlPixelFormat.R16Float or MtlPixelFormat.Rg8Unorm or
                MtlPixelFormat.Rg8Snorm or MtlPixelFormat.Rg8Uint or
                MtlPixelFormat.Rg8Sint or MtlPixelFormat.B5G6R5Unorm => 2u,
            MtlPixelFormat.Rg32Uint or MtlPixelFormat.Rg32Sint or
                MtlPixelFormat.Rg32Float or MtlPixelFormat.Rgba16Unorm or
                MtlPixelFormat.Rgba16Uint or MtlPixelFormat.Rgba16Sint or
                MtlPixelFormat.Rgba16Float => 8u,
            MtlPixelFormat.Rgba32Uint or MtlPixelFormat.Rgba32Sint or
                MtlPixelFormat.Rgba32Float => 16u,
            _ => 4u,
        };

        return new MetalTextureFormat(format, bytesPerPixel, blockBytes);
    }

    /// <summary>Source byte footprint of a sampled texture, block-aware —
    /// the same math the AGC layer uses to size the texel copy it ships.</summary>
    public static ulong GetTextureByteCount(in MetalTextureFormat format, uint width, uint height) =>
        format.IsBlockCompressed
            ? checked(((ulong)width + 3) / 4 * (((ulong)height + 3) / 4) * format.BlockBytes)
            : checked((ulong)width * height * format.BytesPerPixel);

    /// <summary>
    /// The neutral host format the shared image cache names, as a Metal pixel format.
    /// <see cref="GpuFormat"/> carries VkFormat's numeric values, so this is the Metal
    /// half of what the Vulkan backend gets from a plain cast; the image cache and the
    /// tiler both resolve a texture's format through here. Only the formats the engine
    /// actually names are listed, and an unlisted one fails rather than reinterpreting
    /// unrelated bytes.
    /// </summary>
    public static bool TryDecodeHostFormat(GpuFormat format, out MtlPixelFormat result)
    {
        result = format switch
        {
            GpuFormat.R8Unorm => MtlPixelFormat.R8Unorm,
            GpuFormat.R8SNorm => MtlPixelFormat.R8Snorm,
            GpuFormat.R8Uint => MtlPixelFormat.R8Uint,
            GpuFormat.R8Sint => MtlPixelFormat.R8Sint,
            GpuFormat.R8Srgb => MtlPixelFormat.R8UnormSrgb,
            GpuFormat.R8G8Unorm => MtlPixelFormat.Rg8Unorm,
            GpuFormat.R8G8SNorm => MtlPixelFormat.Rg8Snorm,
            GpuFormat.R8G8Uint => MtlPixelFormat.Rg8Uint,
            GpuFormat.R8G8Sint => MtlPixelFormat.Rg8Sint,
            GpuFormat.R8G8Srgb => MtlPixelFormat.Rg8UnormSrgb,
            GpuFormat.R8G8B8A8Unorm => MtlPixelFormat.Rgba8Unorm,
            GpuFormat.R8G8B8A8SNorm => MtlPixelFormat.Rgba8Snorm,
            GpuFormat.R8G8B8A8Uint => MtlPixelFormat.Rgba8Uint,
            GpuFormat.R8G8B8A8Sint => MtlPixelFormat.Rgba8Sint,
            GpuFormat.R8G8B8A8Srgb => MtlPixelFormat.Rgba8UnormSrgb,
            GpuFormat.B8G8R8A8Unorm => MtlPixelFormat.Bgra8Unorm,
            GpuFormat.B8G8R8A8Srgb => MtlPixelFormat.Bgra8UnormSrgb,
            // Metal names the 16-bit packed formats by memory order, so R5G6B5 pack16
            // and B5G6R5 pack16 are the same bits under the two spellings.
            GpuFormat.R5G6B5UnormPack16 or GpuFormat.B5G6R5UnormPack16 => MtlPixelFormat.B5G6R5Unorm,
            GpuFormat.A1R5G5B5UnormPack16 => MtlPixelFormat.Bgr5A1Unorm,
            GpuFormat.R5G5B5A1UnormPack16 => MtlPixelFormat.A1Bgr5Unorm,
            GpuFormat.R4G4B4A4UnormPack16 => MtlPixelFormat.Abgr4Unorm,
            GpuFormat.R16Unorm => MtlPixelFormat.R16Unorm,
            GpuFormat.R16SNorm => MtlPixelFormat.R16Snorm,
            GpuFormat.R16Uint => MtlPixelFormat.R16Uint,
            GpuFormat.R16Sint => MtlPixelFormat.R16Sint,
            GpuFormat.R16Sfloat => MtlPixelFormat.R16Float,
            GpuFormat.R16G16Unorm => MtlPixelFormat.Rg16Unorm,
            GpuFormat.R16G16SNorm => MtlPixelFormat.Rg16Snorm,
            GpuFormat.R16G16Uint => MtlPixelFormat.Rg16Uint,
            GpuFormat.R16G16Sint => MtlPixelFormat.Rg16Sint,
            GpuFormat.R16G16Sfloat => MtlPixelFormat.Rg16Float,
            GpuFormat.R16G16B16A16Unorm => MtlPixelFormat.Rgba16Unorm,
            GpuFormat.R16G16B16A16SNorm => MtlPixelFormat.Rgba16Snorm,
            GpuFormat.R16G16B16A16Uint => MtlPixelFormat.Rgba16Uint,
            GpuFormat.R16G16B16A16Sint => MtlPixelFormat.Rgba16Sint,
            GpuFormat.R16G16B16A16Sfloat => MtlPixelFormat.Rgba16Float,
            GpuFormat.R32Uint => MtlPixelFormat.R32Uint,
            GpuFormat.R32Sint => MtlPixelFormat.R32Sint,
            GpuFormat.R32Sfloat => MtlPixelFormat.R32Float,
            GpuFormat.R32G32Uint => MtlPixelFormat.Rg32Uint,
            GpuFormat.R32G32Sint => MtlPixelFormat.Rg32Sint,
            GpuFormat.R32G32Sfloat => MtlPixelFormat.Rg32Float,
            GpuFormat.R32G32B32A32Uint => MtlPixelFormat.Rgba32Uint,
            GpuFormat.R32G32B32A32Sint => MtlPixelFormat.Rgba32Sint,
            GpuFormat.R32G32B32A32Sfloat => MtlPixelFormat.Rgba32Float,
            GpuFormat.A2B10G10R10UnormPack32 => MtlPixelFormat.Rgb10A2Unorm,
            GpuFormat.A2B10G10R10UintPack32 => MtlPixelFormat.Rgb10A2Uint,
            GpuFormat.A2R10G10B10UnormPack32 => MtlPixelFormat.Bgr10A2Unorm,
            GpuFormat.B10G11R11UfloatPack32 => MtlPixelFormat.Rg11B10Float,
            GpuFormat.E5B9G9R9UfloatPack32 => MtlPixelFormat.Rgb9E5Float,
            GpuFormat.D16Unorm => MtlPixelFormat.Depth16Unorm,
            GpuFormat.D32Sfloat => MtlPixelFormat.Depth32Float,
            GpuFormat.S8Uint => MtlPixelFormat.Stencil8,
            // Apple GPUs have no 16- or 24-bit combined depth-stencil format, so every
            // guest depth format that carries stencil lands on the 32-bit float plane
            // and the tiler widens or narrows the depth bytes around the blit.
            GpuFormat.D16UnormS8Uint or GpuFormat.D24UnormS8Uint or GpuFormat.D32SfloatS8Uint =>
                MtlPixelFormat.Depth32FloatStencil8,
            GpuFormat.BC1RgbUnormBlock or GpuFormat.BC1RgbaUnormBlock => MtlPixelFormat.Bc1Rgba,
            GpuFormat.BC1RgbSrgbBlock or GpuFormat.BC1RgbaSrgbBlock => MtlPixelFormat.Bc1RgbaSrgb,
            GpuFormat.BC2UnormBlock => MtlPixelFormat.Bc2Rgba,
            GpuFormat.BC2SrgbBlock => MtlPixelFormat.Bc2RgbaSrgb,
            GpuFormat.BC3UnormBlock => MtlPixelFormat.Bc3Rgba,
            GpuFormat.BC3SrgbBlock => MtlPixelFormat.Bc3RgbaSrgb,
            GpuFormat.BC4UnormBlock => MtlPixelFormat.Bc4RUnorm,
            GpuFormat.BC4SNormBlock => MtlPixelFormat.Bc4RSnorm,
            GpuFormat.BC5UnormBlock => MtlPixelFormat.Bc5RgUnorm,
            GpuFormat.BC5SNormBlock => MtlPixelFormat.Bc5RgSnorm,
            GpuFormat.BC6HUfloatBlock => MtlPixelFormat.Bc6HRgbUfloat,
            GpuFormat.BC6HSfloatBlock => MtlPixelFormat.Bc6HRgbFloat,
            GpuFormat.BC7UnormBlock => MtlPixelFormat.Bc7RgbaUnorm,
            GpuFormat.BC7SrgbBlock => MtlPixelFormat.Bc7RgbaUnormSrgb,
            _ => MtlPixelFormat.Invalid,
        };
        return result != MtlPixelFormat.Invalid;
    }

    /// <summary>True for the formats whose textures are depth, stencil or both.</summary>
    public static bool IsDepthStencil(MtlPixelFormat format) => format is
        MtlPixelFormat.Depth16Unorm or MtlPixelFormat.Depth32Float or MtlPixelFormat.Stencil8 or
        MtlPixelFormat.Depth32FloatStencil8 or MtlPixelFormat.X32Stencil8;

    /// <summary>True when the format carries a separate stencil plane a blit must
    /// select with MTLBlitOptionStencilFromDepthStencil.</summary>
    public static bool HasStencilPlane(MtlPixelFormat format) =>
        format is MtlPixelFormat.Depth32FloatStencil8;

    /// <summary>Bytes one depth element occupies in a linear buffer that a
    /// buffer-to-texture blit reads; zero when the format has no depth plane.</summary>
    public static uint DepthTransferBytes(MtlPixelFormat format) => format switch
    {
        MtlPixelFormat.Depth16Unorm => 2,
        MtlPixelFormat.Depth32Float or MtlPixelFormat.Depth32FloatStencil8 => 4,
        _ => 0,
    };

    /// <summary>True for the sRGB formats, which Metal refuses to bind writable.</summary>
    public static bool IsSrgb(MtlPixelFormat format) => format is
        MtlPixelFormat.R8UnormSrgb or MtlPixelFormat.Rg8UnormSrgb or MtlPixelFormat.Rgba8UnormSrgb or
        MtlPixelFormat.Bgra8UnormSrgb or MtlPixelFormat.Bc1RgbaSrgb or MtlPixelFormat.Bc2RgbaSrgb or
        MtlPixelFormat.Bc3RgbaSrgb or MtlPixelFormat.Bc7RgbaUnormSrgb;

    /// <summary>
    /// The linear format of an sRGB one, and any other format itself. Metal numbers each
    /// sRGB format one past its linear twin, and a view that only moves between the two
    /// needs no pixel-format-view usage on the texture.
    /// </summary>
    public static MtlPixelFormat LinearOf(MtlPixelFormat format) =>
        IsSrgb(format) ? (MtlPixelFormat)((uint)format - 1) : format;

    /// <summary>
    /// True for the block formats that come in an sRGB and a linear flavour. A guest samples
    /// one memory through both (Unreal's virtual texture physical pools have an sRGB and a
    /// linear SRV), and Metal allows exactly this reinterpretation of a compressed texture.
    /// </summary>
    public static bool HasSrgbTwin(MtlPixelFormat format) => format is
        MtlPixelFormat.Bc1Rgba or MtlPixelFormat.Bc1RgbaSrgb or MtlPixelFormat.Bc2Rgba or MtlPixelFormat.Bc2RgbaSrgb or
        MtlPixelFormat.Bc3Rgba or MtlPixelFormat.Bc3RgbaSrgb or MtlPixelFormat.Bc7RgbaUnorm or MtlPixelFormat.Bc7RgbaUnormSrgb;

    /// <summary>True when the two block formats differ only in sRGB encoding (BC1, BC2, BC3, BC7).</summary>
    public static bool AreSrgbTwins(MtlPixelFormat a, MtlPixelFormat b) =>
        a != b && HasSrgbTwin(a) && HasSrgbTwin(b) && ((uint)a & ~1u) == ((uint)b & ~1u);

    public static bool TryDecodeRenderTargetFormat(
        uint dataFormat,
        uint numberType,
        out MetalRenderTargetFormat result) =>
        TryDecodeRenderTargetFormat(
            dataFormat,
            numberType,
            componentSwap: 0,
            out result);

    public static bool TryDecodeRenderTargetFormat(
        uint dataFormat,
        uint numberType,
        uint componentSwap,
        out MetalRenderTargetFormat result)
    {
        var format = (dataFormat, numberType, componentSwap) switch
        {
            // Early G-buffer / scene targets (R16 + RG32). The same guest encodings are
            // decoded by the Vulkan backend's TryDecodeRenderTargetFormat; keep both in sync
            // (GuestSurfaceFormats in the shared engine is the backend-neutral half).
            (2, 0, _) => MtlPixelFormat.R16Unorm,
            (2, 1, _) => MtlPixelFormat.R16Snorm,
            (2, 4, _) => MtlPixelFormat.R16Uint,
            (2, 5, _) => MtlPixelFormat.R16Sint,
            (2, 7, _) => MtlPixelFormat.R16Float,
            (4, 4, _) => MtlPixelFormat.R32Uint,
            (4, 5, _) => MtlPixelFormat.R32Sint,
            (4, 7, _) => MtlPixelFormat.R32Float,
            (5, 1, _) => MtlPixelFormat.Rg16Snorm,
            (5, 4, _) => MtlPixelFormat.Rg16Uint,
            (5, 5, _) => MtlPixelFormat.Rg16Sint,
            (5, 7, _) => MtlPixelFormat.Rg16Float,
            (6, 7, _) or (7, 7, _) => MtlPixelFormat.Rg11B10Float,
            (9, _, 1) => MtlPixelFormat.Bgr10A2Unorm,
            (9, _, _) => MtlPixelFormat.Rgb10A2Unorm,
            (10, 1, _) => MtlPixelFormat.Rgba8Snorm,
            (10, 4, _) => MtlPixelFormat.Rgba8Uint,
            (10, 5, _) => MtlPixelFormat.Rgba8Sint,
            (10, 6 or 9, 1) => MtlPixelFormat.Bgra8UnormSrgb,
            (10, 6 or 9, _) => MtlPixelFormat.Rgba8UnormSrgb,
            (10, 0, 1) => MtlPixelFormat.Bgra8Unorm,
            (10, _, _) => MtlPixelFormat.Rgba8Unorm,
            (11, 4, _) => MtlPixelFormat.Rg32Uint,
            (11, 5, _) => MtlPixelFormat.Rg32Sint,
            (11, 7, _) => MtlPixelFormat.Rg32Float,
            (12, 1, _) => MtlPixelFormat.Rgba16Snorm,
            (12, 4, _) => MtlPixelFormat.Rgba16Uint,
            (12, 5, _) => MtlPixelFormat.Rgba16Sint,
            (12, 7, _) => MtlPixelFormat.Rgba16Float,
            (14, 4, _) => MtlPixelFormat.Rgba32Uint,
            (13, 7, _) or (14, 7, _) => MtlPixelFormat.Rgba32Float,
            (20, 0, _) => MtlPixelFormat.R32Uint,
            (29, 0, _) or (4, 0, _) => MtlPixelFormat.R32Float,
            (1, 0, _) or (36, 0, _) => MtlPixelFormat.R8Unorm,
            (1, 4, _) or (49, 0, _) => MtlPixelFormat.R8Uint,
            (1, 1, _) => MtlPixelFormat.R8Snorm,
            (3, 0, _) => MtlPixelFormat.Rg8Unorm,
            (3, 1, _) => MtlPixelFormat.Rg8Snorm,
            (5, 0, _) => MtlPixelFormat.Rg16Unorm,
            (7, 0, _) => MtlPixelFormat.Rg11B10Float,
            (12, 0, _) => MtlPixelFormat.Rgba16Unorm,
            (13, 0, _) or (14, 0, _) => MtlPixelFormat.Rgba32Float,
            (22, 0, _) or (71, 0, _) => MtlPixelFormat.Rgba16Float,
            (56, 0, _) or (62, 0, _) or (64, 0, _) => MtlPixelFormat.Rgba8Unorm,
            (75, 0, _) => MtlPixelFormat.Rg32Float,
            _ => MtlPixelFormat.Invalid,
        };

        if (format == MtlPixelFormat.Invalid ||
            !TryGetRenderTargetComponentCount(dataFormat, out var componentCount) ||
            !Gen5ColorComponentMapping.TryResolveRenderTarget(
                componentSwap,
                componentCount,
                out var orderMapping))
        {
            result = default;
            return false;
        }

        var outputKind = format switch
        {
            MtlPixelFormat.R8Uint or MtlPixelFormat.R16Uint or MtlPixelFormat.R32Uint or
                MtlPixelFormat.Rg16Uint or MtlPixelFormat.Rg32Uint or MtlPixelFormat.Rgba8Uint or
                MtlPixelFormat.Rgba16Uint or MtlPixelFormat.Rgba32Uint => Gen5PixelOutputKind.Uint,
            MtlPixelFormat.R16Sint or MtlPixelFormat.R32Sint or MtlPixelFormat.Rg16Sint or
                MtlPixelFormat.Rg32Sint or MtlPixelFormat.Rgba8Sint or MtlPixelFormat.Rgba16Sint =>
                Gen5PixelOutputKind.Sint,
            _ => Gen5PixelOutputKind.Float,
        };
        var hostToStorage = dataFormat switch
        {
            9 when componentSwap == 1 => new Gen5ColorComponentMapping(0xC6),
            10 when componentSwap == 1 && numberType is 0 or 6 or 9 =>
                new Gen5ColorComponentMapping(0xC6),
            _ => Gen5ColorComponentMapping.Identity,
        };
        result = new MetalRenderTargetFormat(
            format,
            outputKind,
            hostToStorage.Then(orderMapping));
        return true;
    }

    private static bool TryGetRenderTargetComponentCount(
        uint dataFormat,
        out uint componentCount)
    {
        componentCount = dataFormat switch
        {
            1 or 2 or 4 or 20 or 29 or 36 or 49 => 1,
            3 or 5 or 11 or 75 => 2,
            6 or 7 => 3,
            9 or 10 or 12 or 13 or 14 or 22 or 56 or 62 or 64 or 71 => 4,
            _ => 0,
        };
        return componentCount != 0;
    }
}
