// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler.Metal;

namespace SharpEmu.Gpu.Metal.V2;

// The MTLPixelFormat raw values v2 needs beyond MtlPixelFormat's set; the enum in
// the first Metal path covers only what its guest-code tables mapped.
internal static class MtlFormat
{
    public const ulong Invalid = 0;
    public const ulong A8Unorm = 1;
    public const ulong R8Unorm = 10;
    public const ulong R8UnormSrgb = 11;
    public const ulong R8Snorm = 12;
    public const ulong R8Uint = 13;
    public const ulong R8Sint = 14;
    public const ulong R16Unorm = 20;
    public const ulong R16Snorm = 22;
    public const ulong R16Uint = 23;
    public const ulong R16Sint = 24;
    public const ulong R16Float = 25;
    public const ulong Rg8Unorm = 30;
    public const ulong Rg8UnormSrgb = 31;
    public const ulong Rg8Snorm = 32;
    public const ulong Rg8Uint = 33;
    public const ulong Rg8Sint = 34;
    public const ulong B5G6R5Unorm = 40;
    public const ulong A1Bgr5Unorm = 41;
    public const ulong Abgr4Unorm = 42;
    public const ulong Bgr5A1Unorm = 43;
    public const ulong R32Uint = 53;
    public const ulong R32Sint = 54;
    public const ulong R32Float = 55;
    public const ulong Rg16Unorm = 60;
    public const ulong Rg16Snorm = 62;
    public const ulong Rg16Uint = 63;
    public const ulong Rg16Sint = 64;
    public const ulong Rg16Float = 65;
    public const ulong Rgba8Unorm = 70;
    public const ulong Rgba8UnormSrgb = 71;
    public const ulong Rgba8Snorm = 72;
    public const ulong Rgba8Uint = 73;
    public const ulong Rgba8Sint = 74;
    public const ulong Bgra8Unorm = 80;
    public const ulong Bgra8UnormSrgb = 81;
    public const ulong Rgb10A2Unorm = 90;
    public const ulong Rgb10A2Uint = 91;
    public const ulong Rg11B10Float = 92;
    public const ulong Rgb9E5Float = 93;
    public const ulong Bgr10A2Unorm = 94;
    public const ulong Rg32Uint = 103;
    public const ulong Rg32Sint = 104;
    public const ulong Rg32Float = 105;
    public const ulong Rgba16Unorm = 110;
    public const ulong Rgba16Snorm = 112;
    public const ulong Rgba16Uint = 113;
    public const ulong Rgba16Sint = 114;
    public const ulong Rgba16Float = 115;
    public const ulong Rgba32Uint = 123;
    public const ulong Rgba32Sint = 124;
    public const ulong Rgba32Float = 125;
    public const ulong Bc1Rgba = 130;
    public const ulong Bc1RgbaSrgb = 131;
    public const ulong Bc2Rgba = 132;
    public const ulong Bc2RgbaSrgb = 133;
    public const ulong Bc3Rgba = 134;
    public const ulong Bc3RgbaSrgb = 135;
    public const ulong Bc4RUnorm = 140;
    public const ulong Bc4RSnorm = 141;
    public const ulong Bc5RgUnorm = 142;
    public const ulong Bc5RgSnorm = 143;
    public const ulong Bc6HRgbFloat = 150;
    public const ulong Bc6HRgbUfloat = 151;
    public const ulong Bc7RgbaUnorm = 152;
    public const ulong Bc7RgbaUnormSrgb = 153;
    public const ulong Depth16Unorm = 250;
    public const ulong Depth32Float = 252;
    public const ulong Stencil8 = 253;
    public const ulong Depth32FloatStencil8 = 260;
    public const ulong X32Stencil8 = 261;
}

// Translates the shared neutral vocabulary (GpuFormat, blend and stencil codes,
// topologies) into Metal. The neutral enums carry Vulkan's numeric values, so none
// of them can be cast; every mapping is explicit.
internal static class MetalV2Formats
{
    private static readonly ConcurrentDictionary<GpuFormat, byte> _reportedFormats = new();

    // The pixel format a view or an attachment uses. The mapping is the image
    // transfer's (MetalGuestFormats), so a texture and the bytes staged into it never
    // disagree; an unmapped format falls back to four-byte RGBA and logs once, because
    // a texture is never worth a fatal error.
    public static ulong PixelFormat(GpuFormat format)
    {
        if (MetalGuestFormats.TryDecodeHostFormat(format, out var mapped))
        {
            return (ulong)mapped;
        }

        if (_reportedFormats.TryAdd(format, 0))
        {
            Console.Error.WriteLine(
                $"[METAL-V2][WARN] The guest pixel format has no Metal format; using RGBA8: format={(int)format}.");
        }

        return MtlFormat.Rgba8Unorm;
    }

    public static bool IsSupported(GpuFormat format) =>
        MetalGuestFormats.TryDecodeHostFormat(format, out _);

    // The format an unmapped image falls back to: same bytes per block, so the tiler's
    // element size and the guest footprint still agree.
    public static GpuFormat SubstituteFor(uint bytesPerBlock) => bytesPerBlock switch
    {
        1 => GpuFormat.R8Unorm,
        2 => GpuFormat.R16Unorm,
        8 => GpuFormat.R16G16B16A16Unorm,
        16 => GpuFormat.R32G32B32A32Sfloat,
        _ => GpuFormat.R8G8B8A8Unorm,
    };

    public static bool IsDepthFormat(ulong format) =>
        format is MtlFormat.Depth16Unorm or MtlFormat.Depth32Float or MtlFormat.Depth32FloatStencil8;

    public static bool HasStencil(ulong format) =>
        format is MtlFormat.Depth32FloatStencil8 or MtlFormat.Stencil8 or MtlFormat.X32Stencil8;

    // The non-sRGB format with the same bits, for a view that must not convert.
    public static GpuFormat LinearCounterpart(GpuFormat format) => format switch
    {
        GpuFormat.R8Srgb => GpuFormat.R8Unorm,
        GpuFormat.R8G8Srgb => GpuFormat.R8G8Unorm,
        GpuFormat.R8G8B8A8Srgb => GpuFormat.R8G8B8A8Unorm,
        GpuFormat.B8G8R8A8Srgb => GpuFormat.B8G8R8A8Unorm,
        GpuFormat.A8B8G8R8SrgbPack32 => GpuFormat.A8B8G8R8UnormPack32,
        _ => format,
    };

    public static ulong TextureType(GpuImageType type, uint layers, uint samples) => (type, layers, samples) switch
    {
        (GpuImageType.Type3D, _, _) => Mtl.TextureType3D,
        (GpuImageType.Type1D, 1, _) => Mtl.TextureType1D,
        (GpuImageType.Type1D, _, _) => Mtl.TextureType1DArray,
        (_, 1, 1) => Mtl.TextureType2D,
        (_, 1, _) => Mtl.TextureType2DMultisample,
        (_, _, 1) => Mtl.TextureType2DArray,
        _ => Mtl.TextureType2DMultisampleArray,
    };

    // The texture type a shader view asks for. Guest cube maps bind as 2D arrays
    // (see Gen5MslArgumentLayout.TextureType), so no cube type is produced here.
    public static ulong ViewTextureType(GpuImageViewType type, uint samples) => type switch
    {
        GpuImageViewType.Type1D => Mtl.TextureType1D,
        GpuImageViewType.Type1DArray => Mtl.TextureType1DArray,
        GpuImageViewType.Type3D => Mtl.TextureType3D,
        GpuImageViewType.Type2D => samples > 1 ? Mtl.TextureType2DMultisample : Mtl.TextureType2D,
        _ => samples > 1 ? Mtl.TextureType2DMultisampleArray : Mtl.TextureType2DArray,
    };

    public static nuint Topology(GpuPrimitiveTopology topology) => topology switch
    {
        GpuPrimitiveTopology.PointList => Mtl.PrimitivePoint,
        GpuPrimitiveTopology.LineList => Mtl.PrimitiveLine,
        GpuPrimitiveTopology.LineStrip => Mtl.PrimitiveLineStrip,
        GpuPrimitiveTopology.TriangleStrip => Mtl.PrimitiveTriangleStrip,
        // Metal has no fan primitive; the host answers DrawsTriangleFans false, so the
        // executor draws every fan as a triangle list and none may reach a pipeline.
        GpuPrimitiveTopology.TriangleFan => throw new InvalidOperationException(
            "A triangle-fan pipeline reached Metal, which has no fan primitive; the executor converts fans to triangle lists."),
        // A guest rectangle list draws as tessellated patches, whose output is triangles;
        // see MetalRenderHost.DrawRectangles.
        GpuPrimitiveTopology.PatchList => Mtl.PrimitiveTriangle,
        _ => Mtl.PrimitiveTriangle,
    };

    // The rectangles a rectangle-list draw of this many vertices (or indices) makes:
    // every complete group of three is one rectangle, and a trailing incomplete group
    // draws nothing, as the hardware's primitive assembly discards it.
    public static uint RectanglePatchCount(uint vertexCount) =>
        vertexCount / Gen5MslRectangleList.VerticesPerRectangle;

    // MTLPrimitiveTopologyClass of the primitive type Topology picks.
    public static nuint TopologyClass(GpuPrimitiveTopology topology) => Topology(topology) switch
    {
        Mtl.PrimitivePoint => 1,
        Mtl.PrimitiveLine or Mtl.PrimitiveLineStrip => 2,
        _ => 3,
    };

    public static nuint IndexType(GpuIndexType type) =>
        type == GpuIndexType.Uint32 ? Mtl.IndexTypeUInt32 : Mtl.IndexTypeUInt16;

    // MTLCompareFunction and MTLStencilOperation happen to share Vulkan's values,
    // but the cast is spelled out so a future divergence is a compile error here.
    public static nuint CompareFunction(GpuCompareOp compare) => compare switch
    {
        GpuCompareOp.Never => 0,
        GpuCompareOp.Less => 1,
        GpuCompareOp.Equal => 2,
        GpuCompareOp.LessOrEqual => 3,
        GpuCompareOp.Greater => 4,
        GpuCompareOp.NotEqual => 5,
        GpuCompareOp.GreaterOrEqual => 6,
        _ => 7,
    };

    public static nuint StencilOperation(GpuStencilOp operation) => operation switch
    {
        GpuStencilOp.Keep => 0,
        GpuStencilOp.Zero => 1,
        GpuStencilOp.Replace => 2,
        GpuStencilOp.IncrementAndClamp => 3,
        GpuStencilOp.DecrementAndClamp => 4,
        GpuStencilOp.Invert => 5,
        GpuStencilOp.IncrementAndWrap => 6,
        _ => 7,
    };

    // Guest CB_BLEND factor codes to MTLBlendFactor, matching the Vulkan mapping.
    public static nint BlendFactor(uint factor) => factor switch
    {
        0 => 0,
        1 => 1,
        2 => 2,
        3 => 3,
        4 => 4,
        5 => 5,
        6 => 8,
        7 => 9,
        8 => 6,
        9 => 7,
        10 => 10,
        13 => 11,
        14 => 12,
        15 => 15,
        16 => 16,
        17 => 17,
        18 => 18,
        19 => 13,
        20 => 14,
        _ => 1,
    };

    // Guest COMB_FCN codes to MTLBlendOperation.
    public static nint BlendOperation(uint function) => function switch
    {
        0 => Mtl.BlendOperationAdd,
        1 => Mtl.BlendOperationSubtract,
        2 => Mtl.BlendOperationMin,
        3 => Mtl.BlendOperationMax,
        4 => Mtl.BlendOperationReverseSubtract,
        _ => Mtl.BlendOperationAdd,
    };

    // Guest CB write-mask bits are R=1,G=2,B=4,A=8; MTLColorWriteMask reverses them.
    public static nint ColorWriteMask(uint guestMask) =>
        ((guestMask & 1) != 0 ? Mtl.ColorWriteMaskRed : 0) |
        ((guestMask & 2) != 0 ? Mtl.ColorWriteMaskGreen : 0) |
        ((guestMask & 4) != 0 ? Mtl.ColorWriteMaskBlue : 0) |
        ((guestMask & 8) != 0 ? Mtl.ColorWriteMaskAlpha : 0);

    // The vertex format of a buffer descriptor, narrowed to the components the
    // program fetches. Mirrors the Vulkan VertexAttributeFormats table.
    public static MtlVertexFormat VertexFormat(in BufferDescriptorWords descriptor, uint usedComponents)
    {
        var (format, size) = descriptor.Format switch
        {
            77 or 113 => (MtlVertexFormat.Float4, 4u),
            76 => (MtlVertexFormat.Int4, 4u),
            75 => (MtlVertexFormat.UInt4, 4u),
            74 => (MtlVertexFormat.Float3, 3u),
            73 => (MtlVertexFormat.Int3, 3u),
            72 => (MtlVertexFormat.UInt3, 3u),
            71 => (MtlVertexFormat.Half4, 4u),
            70 => (MtlVertexFormat.Short4, 4u),
            69 => (MtlVertexFormat.UShort4, 4u),
            68 => (MtlVertexFormat.Short4, 4u),
            67 => (MtlVertexFormat.UShort4, 4u),
            66 => (MtlVertexFormat.Short4Normalized, 4u),
            65 => (MtlVertexFormat.UShort4Normalized, 4u),
            64 => (MtlVertexFormat.Float2, 2u),
            63 => (MtlVertexFormat.Int2, 2u),
            62 => (MtlVertexFormat.UInt2, 2u),
            60 => (MtlVertexFormat.UChar4, 4u),
            61 => (MtlVertexFormat.Char4, 4u),
            59 => (MtlVertexFormat.Char4, 4u),
            58 => (MtlVertexFormat.UChar4, 4u),
            57 => (MtlVertexFormat.Char4Normalized, 4u),
            56 => (MtlVertexFormat.UChar4Normalized, 4u),
            51 => (MtlVertexFormat.Int1010102Normalized, 4u),
            50 => (MtlVertexFormat.UInt1010102Normalized, 4u),
            36 => (MtlVertexFormat.FloatRG11B10, 3u),
            29 or 121 => (MtlVertexFormat.Half2, 2u),
            28 => (MtlVertexFormat.Short2, 2u),
            27 => (MtlVertexFormat.UShort2, 2u),
            26 => (MtlVertexFormat.Short2, 2u),
            25 => (MtlVertexFormat.UShort2, 2u),
            24 => (MtlVertexFormat.Short2Normalized, 2u),
            23 => (MtlVertexFormat.UShort2Normalized, 2u),
            22 => (MtlVertexFormat.Float, 1u),
            21 => (MtlVertexFormat.Int, 1u),
            20 => (MtlVertexFormat.UInt, 1u),
            19 => (MtlVertexFormat.Char2, 2u),
            18 => (MtlVertexFormat.UChar2, 2u),
            17 => (MtlVertexFormat.Char2, 2u),
            16 => (MtlVertexFormat.UChar2, 2u),
            15 => (MtlVertexFormat.Char2Normalized, 2u),
            14 => (MtlVertexFormat.UChar2Normalized, 2u),
            13 => (MtlVertexFormat.Half, 1u),
            12 => (MtlVertexFormat.Short, 1u),
            11 => (MtlVertexFormat.UShort, 1u),
            10 => (MtlVertexFormat.Short, 1u),
            9 => (MtlVertexFormat.UShort, 1u),
            8 => (MtlVertexFormat.ShortNormalized, 1u),
            7 => (MtlVertexFormat.UShortNormalized, 1u),
            6 => (MtlVertexFormat.Char, 1u),
            5 => (MtlVertexFormat.UChar, 1u),
            4 => (MtlVertexFormat.Char, 1u),
            3 => (MtlVertexFormat.UChar, 1u),
            2 => (MtlVertexFormat.CharNormalized, 1u),
            1 => (MtlVertexFormat.UCharNormalized, 1u),
            _ => (MtlVertexFormat.Invalid, 4u),
        };
        if (format == MtlVertexFormat.Invalid)
        {
            return usedComponents switch
            {
                1 => MtlVertexFormat.Float,
                2 => MtlVertexFormat.Float2,
                3 => MtlVertexFormat.Float3,
                _ => MtlVertexFormat.Float4,
            };
        }

        return Narrow(format, size, usedComponents);
    }

    // Fewer components than the format holds select the matching narrower format.
    private static MtlVertexFormat Narrow(MtlVertexFormat format, uint size, uint usedComponents)
    {
        if (usedComponents == 0 || usedComponents >= size)
        {
            return format;
        }

        return (format, usedComponents) switch
        {
            (MtlVertexFormat.Float4, 1) => MtlVertexFormat.Float,
            (MtlVertexFormat.Float4, 2) => MtlVertexFormat.Float2,
            (MtlVertexFormat.Float4, 3) => MtlVertexFormat.Float3,
            (MtlVertexFormat.Float3, 1) => MtlVertexFormat.Float,
            (MtlVertexFormat.Float3, 2) => MtlVertexFormat.Float2,
            (MtlVertexFormat.Int4, 1) => MtlVertexFormat.Int,
            (MtlVertexFormat.Int4, 2) => MtlVertexFormat.Int2,
            (MtlVertexFormat.Int4, 3) => MtlVertexFormat.Int3,
            (MtlVertexFormat.UInt4, 1) => MtlVertexFormat.UInt,
            (MtlVertexFormat.UInt4, 2) => MtlVertexFormat.UInt2,
            (MtlVertexFormat.UInt4, 3) => MtlVertexFormat.UInt3,
            (MtlVertexFormat.Half4, 1) => MtlVertexFormat.Half,
            (MtlVertexFormat.Half4, 2) => MtlVertexFormat.Half2,
            (MtlVertexFormat.Half4, 3) => MtlVertexFormat.Half3,
            (MtlVertexFormat.Short4, 2) => MtlVertexFormat.Short2,
            (MtlVertexFormat.UShort4, 2) => MtlVertexFormat.UShort2,
            (MtlVertexFormat.Short4Normalized, 2) => MtlVertexFormat.Short2Normalized,
            (MtlVertexFormat.UShort4Normalized, 2) => MtlVertexFormat.UShort2Normalized,
            (MtlVertexFormat.UChar4, 1) => MtlVertexFormat.UChar,
            (MtlVertexFormat.UChar4, 2) => MtlVertexFormat.UChar2,
            (MtlVertexFormat.UChar4, 3) => MtlVertexFormat.UChar3,
            (MtlVertexFormat.Char4, 1) => MtlVertexFormat.Char,
            (MtlVertexFormat.Char4, 2) => MtlVertexFormat.Char2,
            (MtlVertexFormat.UChar4Normalized, 1) => MtlVertexFormat.UCharNormalized,
            (MtlVertexFormat.UChar4Normalized, 2) => MtlVertexFormat.UChar2Normalized,
            (MtlVertexFormat.Char4Normalized, 2) => MtlVertexFormat.Char2Normalized,
            _ => format,
        };
    }

    // Guest sampler clamp codes to MTLSamplerAddressMode. MIRROR_ONCE_BORDER (7) samples
    // mirror-once-to-edge and the shader substitutes the border past the mirrored copy; the
    // half-border modes take their nearest host mode, a deviation the sampler reports.
    public static nint SamplerAddressMode(uint mode) => mode switch
    {
        0 => Mtl.SamplerAddressRepeat,
        1 => Mtl.SamplerAddressMirrorRepeat,
        2 => Mtl.SamplerAddressClampToEdge,
        3 or 5 or 7 => Mtl.SamplerAddressMirrorClampToEdge,
        4 or 6 => Mtl.SamplerAddressClampToBorderColor,
        _ => Mtl.SamplerAddressClampToEdge,
    };
}
