// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

/// <summary>Context register dword offsets (mm* / 4 - 0xA000), gfx10.3.</summary>
internal static class Cx
{
    public const uint DbRenderControl = 0x000;
    public const uint DbCountControl = 0x001;
    public const uint DbDepthView = 0x002;
    public const uint DbRenderOverride = 0x003;
    public const uint DbHtileDataBase = 0x005;
    public const uint DbDepthSizeXy = 0x007;
    public const uint DbDepthBoundsMin = 0x008;
    public const uint DbDepthBoundsMax = 0x009;
    public const uint DbStencilClear = 0x00A;
    public const uint DbDepthClear = 0x00B;
    public const uint PaScScreenScissorTl = 0x00C;
    public const uint PaScScreenScissorBr = 0x00D;
    public const uint DbZInfo = 0x010;
    public const uint DbStencilInfo = 0x011;
    public const uint DbZReadBase = 0x012;
    public const uint DbStencilReadBase = 0x013;
    public const uint DbZWriteBase = 0x014;
    public const uint DbStencilWriteBase = 0x015;
    public const uint DbZReadBaseHi = 0x01A;
    public const uint DbStencilReadBaseHi = 0x01B;
    public const uint DbZWriteBaseHi = 0x01C;
    public const uint DbStencilWriteBaseHi = 0x01D;
    public const uint PaScWindowOffset = 0x080;
    public const uint PaScWindowScissorTl = 0x081;
    public const uint PaScWindowScissorBr = 0x082;
    public const uint CbTargetMask = 0x08E;
    public const uint CbShaderMask = 0x08F;
    public const uint PaScGenericScissorTl = 0x090;
    public const uint PaScGenericScissorBr = 0x091;
    public const uint PaScVportScissor0Tl = 0x094;
    public const uint PaScVportScissor0Br = 0x095;
    public const uint PaScVportZMin0 = 0x0B4;
    public const uint PaScVportZMax0 = 0x0B5;
    public const uint CbBlendRed = 0x105;
    public const uint DbStencilControl = 0x10B;
    public const uint DbStencilRefMask = 0x10C;
    public const uint DbStencilRefMaskBf = 0x10D;
    public const uint PaClVportXScale = 0x10F;
    public const uint PaClVportXOffset = 0x110;
    public const uint PaClVportYScale = 0x111;
    public const uint PaClVportYOffset = 0x112;
    public const uint PaClVportZScale = 0x113;
    public const uint PaClVportZOffset = 0x114;
    public const uint PaClUcp0X = 0x16F;
    public const uint SpiPsInputCntl0 = 0x191;
    public const uint SpiVsOutConfig = 0x1B1;
    public const uint SpiPsInputEna = 0x1B3;
    public const uint SpiPsInputAddr = 0x1B4;
    public const uint SpiPsInControl = 0x1B6;
    public const uint SpiBarycCntl = 0x1B8;
    public const uint SpiShaderPosFormat = 0x1C3;
    public const uint SpiShaderZFormat = 0x1C4;
    public const uint SpiShaderColFormat = 0x1C5;
    public const uint CbBlend0Control = 0x1E0;
    public const uint DbDepthControl = 0x200;
    public const uint DbEqaa = 0x201;
    public const uint CbColorControl = 0x202;
    public const uint DbShaderControl = 0x203;
    public const uint PaClClipCntl = 0x204;
    public const uint PaSuScModeCntl = 0x205;
    public const uint PaClVteCntl = 0x206;
    public const uint PaClVsOutCntl = 0x207;
    public const uint PaSuLineCntl = 0x282;
    public const uint PaScModeCntl0 = 0x292;
    public const uint VgtShaderStagesEn = 0x2D5;
    public const uint DbAlphaToMask = 0x2DC;
    public const uint PaSuPolyOffsetDbFmtCntl = 0x2DE;
    public const uint PaSuPolyOffsetClamp = 0x2DF;
    public const uint PaSuPolyOffsetFrontScale = 0x2E0;
    public const uint PaSuPolyOffsetFrontOffset = 0x2E1;
    public const uint PaSuPolyOffsetBackScale = 0x2E2;
    public const uint PaSuPolyOffsetBackOffset = 0x2E3;
    public const uint PaScAaConfig = 0x2F8;
    public const uint PaClGbVertClipAdj = 0x2FA;
    public const uint PaClGbVertDiscAdj = 0x2FB;
    public const uint PaClGbHorzClipAdj = 0x2FC;
    public const uint PaClGbHorzDiscAdj = 0x2FD;
    public const uint CbColor0Base = 0x318;
    public const uint CbColor0View = 0x31B;
    public const uint CbColor0Info = 0x31C;
    public const uint CbColor0Attrib = 0x31D;
    public const uint CbColor0BaseExt = 0x390;
    public const uint CbColor0Attrib2 = 0x3B0;
    public const uint CbColor0Attrib3 = 0x3B8;
    public const uint CbColorSlotStride = 15;
}

/// <summary>Shader register dword offsets (mm* / 4 - 0x2C00).</summary>
internal static class Sh
{
    public const uint PgmLoPs = 0x008;
    public const uint PgmHiPs = 0x009;
    public const uint PgmRsrc1Ps = 0x00A;
    public const uint PgmRsrc2Ps = 0x00B;
    public const uint UserDataPs0 = 0x00C;
    public const uint PgmRsrc1Gs = 0x08A;
    public const uint PgmRsrc2Gs = 0x08B;
    public const uint UserDataGs0 = 0x08C;
    public const uint PgmLoEs = 0x0C8;
    public const uint PgmHiEs = 0x0C9;
    public const uint ComputePgmLo = 0x20C;
    public const uint ComputePgmHi = 0x20D;
    public const uint ComputePgmRsrc1 = 0x212;
    public const uint ComputePgmRsrc2 = 0x213;
    public const uint ComputeNumThreadX = 0x207;
    public const uint ComputeNumThreadY = 0x208;
    public const uint ComputeNumThreadZ = 0x209;
    public const uint ComputeUserData0 = 0x240;
}

/// <summary>User-config register dword offsets (mm* / 4 - 0xC000).</summary>
internal static class Uc
{
    public const uint VgtPrimitiveType = 0x242;
    public const uint VgtIndexType = 0x243;
}

/// <summary>VGT_PRIMITIVE_TYPE values.</summary>
internal static class Prim
{
    public const uint PointList = 1;
    public const uint LineList = 2;
    public const uint LineStrip = 3;
    public const uint TriangleList = 4;
    public const uint TriangleFan = 5;
    public const uint TriangleStrip = 6;
    public const uint RectangleList = 17;
}

/// <summary>Compare functions as DB_DEPTH_CONTROL and the S# encode them.</summary>
internal enum CompareFunc : uint
{
    Never = 0,
    Less = 1,
    Equal = 2,
    LessEqual = 3,
    Greater = 4,
    NotEqual = 5,
    GreaterEqual = 6,
    Always = 7,
}

/// <summary>
/// The register state of one draw, emitted as one SET_*_REG packet per register so every
/// write goes through the interpreter's own writer for it. Defaults are a minimal valid
/// pipeline: no depth, no culling, D3D clip space, one 32-bit-per-channel colour export.
/// </summary>
internal sealed class SceneState
{
    public SortedDictionary<uint, uint> Context { get; } = new();
    public SortedDictionary<uint, uint> Shader { get; } = new();
    public SortedDictionary<uint, uint> Uconfig { get; } = new();

    public SceneState(uint width, uint height)
    {
        Context[Cx.VgtShaderStagesEn] = 0;
        Context[Cx.CbTargetMask] = 0xF;
        Context[Cx.CbShaderMask] = 0xF;
        Context[Cx.SpiShaderColFormat] = 9;
        Context[Cx.SpiShaderZFormat] = 0;
        Context[Cx.SpiShaderPosFormat] = 4;
        Context[Cx.SpiPsInputEna] = 0x2;
        Context[Cx.SpiPsInputAddr] = 0x2;
        Context[Cx.SpiPsInControl] = 1;
        Context[Cx.SpiPsInputCntl0] = 0;
        Context[Cx.SpiVsOutConfig] = 0;
        Context[Cx.PaClClipCntl] = 1u << 19;
        Context[Cx.PaSuScModeCntl] = 0;
        Context[Cx.PaClVteCntl] = 0x43F;
        Context[Cx.DbDepthControl] = 0;
        Context[Cx.DbRenderControl] = 0;
        Context[Cx.DbShaderControl] = 0x10;
        Context[Cx.CbColorControl] = (1u << 4) | (0xCCu << 16);
        Context[Cx.CbBlend0Control] = 0;
        Context[Cx.PaScAaConfig] = 0;
        Context[Cx.DbEqaa] = 0;
        Uconfig[Uc.VgtPrimitiveType] = Prim.TriangleList;
        Viewport(0, 0, width, height);
        ScreenScissor(0, 0, (int)width, (int)height);
    }

    public static uint F(float value) => BitConverter.SingleToUInt32Bits(value);

    /// <summary>A D3D-style viewport: NDC y up maps to the top row, z [0,1] to [minZ, maxZ].</summary>
    public SceneState Viewport(float x, float y, float width, float height, float minZ = 0f, float maxZ = 1f)
    {
        Context[Cx.PaClVportXScale] = F(width / 2);
        Context[Cx.PaClVportXOffset] = F(x + (width / 2));
        Context[Cx.PaClVportYScale] = F(-height / 2);
        Context[Cx.PaClVportYOffset] = F(y + (height / 2));
        Context[Cx.PaClVportZScale] = F(maxZ - minZ);
        Context[Cx.PaClVportZOffset] = F(minZ);
        Context[Cx.PaScVportZMin0] = F(Math.Min(minZ, maxZ));
        Context[Cx.PaScVportZMax0] = F(Math.Max(minZ, maxZ));
        return this;
    }

    public SceneState ScreenScissor(int left, int top, int right, int bottom)
    {
        Context[Cx.PaScScreenScissorTl] = (uint)(left & 0xFFFF) | ((uint)(top & 0xFFFF) << 16);
        Context[Cx.PaScScreenScissorBr] = (uint)(right & 0xFFFF) | ((uint)(bottom & 0xFFFF) << 16);
        return this;
    }

    /// <summary>A colour target in slot <paramref name="slot"/>. Format and number type as CB_COLOR_INFO encodes them.</summary>
    public SceneState ColorTarget(uint slot, ulong address, uint width, uint height, uint format = 10, uint numberType = 0,
        uint tileMode = 0, uint compSwap = 0, uint samplesLog2 = 0, uint lastSlice = 0, uint mipLevel = 0, uint maxMip = 0,
        bool blendBypass = false, bool blendClamp = false)
    {
        var block = slot * Cx.CbColorSlotStride;
        Context[Cx.CbColor0Base + block] = (uint)(address >> 8);
        Context[Cx.CbColor0BaseExt + slot] = (uint)(address >> 40);
        Context[Cx.CbColor0View + block] = (lastSlice << 13) | (mipLevel << 26);
        Context[Cx.CbColor0Info + block] = (format << 2) | (numberType << 8) | (compSwap << 11) |
            (blendClamp ? 1u << 15 : 0u) | (blendBypass ? 1u << 16 : 0u);
        Context[Cx.CbColor0Attrib + block] = (samplesLog2 << 12) | (samplesLog2 << 15);
        Context[Cx.CbColor0Attrib2 + slot] = (height - 1) | ((width - 1) << 14) | (maxMip << 28);
        Context[Cx.CbColor0Attrib3 + slot] = (tileMode << 14) | (1u << 24);
        return this;
    }

    /// <summary>
    /// A depth (and optional stencil) target. <paramref name="zFormat"/> is DB_Z_INFO.FORMAT:
    /// 1 = Z16, 3 = Z32_FLOAT. Bases must be 64 KiB aligned; the layout is 64KB_Z_X (SW mode 24).
    /// </summary>
    public SceneState DepthTarget(ulong depthAddress, uint width, uint height, uint zFormat, ulong stencilAddress = 0)
    {
        Context[Cx.DbZInfo] = zFormat | (24u << 4);
        Context[Cx.DbStencilInfo] = stencilAddress != 0 ? 1u | (24u << 4) : 0u;
        Context[Cx.DbZReadBase] = (uint)(depthAddress >> 8);
        Context[Cx.DbZWriteBase] = (uint)(depthAddress >> 8);
        Context[Cx.DbZReadBaseHi] = (uint)(depthAddress >> 40);
        Context[Cx.DbZWriteBaseHi] = (uint)(depthAddress >> 40);
        Context[Cx.DbStencilReadBase] = (uint)(stencilAddress >> 8);
        Context[Cx.DbStencilWriteBase] = (uint)(stencilAddress >> 8);
        Context[Cx.DbStencilReadBaseHi] = (uint)(stencilAddress >> 40);
        Context[Cx.DbStencilWriteBaseHi] = (uint)(stencilAddress >> 40);
        Context[Cx.DbDepthSizeXy] = (width - 1) | ((height - 1) << 16);
        Context[Cx.DbDepthView] = 0;
        Context[Cx.PaSuPolyOffsetDbFmtCntl] = zFormat == 3 ? (1u << 8) | (unchecked((uint)-23) & 0xFF) : unchecked((uint)-16) & 0xFF;
        return this;
    }

    public SceneState DepthControl(bool test, bool write, CompareFunc func, bool bounds = false, bool stencil = false,
        CompareFunc stencilFunc = CompareFunc.Always, bool backFace = false, CompareFunc stencilFuncBack = CompareFunc.Always)
    {
        Context[Cx.DbDepthControl] = (stencil ? 1u : 0u) | (test ? 2u : 0u) | (write ? 4u : 0u) | (bounds ? 8u : 0u) |
            ((uint)func << 4) | (backFace ? 1u << 7 : 0u) | ((uint)stencilFunc << 8) | ((uint)stencilFuncBack << 20);
        return this;
    }

    /// <summary>The vertex program at the ES address with its user data in GS user SGPRs (s8 onward).</summary>
    public SceneState VertexShader(ulong address, params uint[] userData)
    {
        Shader[Sh.PgmLoEs] = (uint)(address >> 8);
        Shader[Sh.PgmHiEs] = (uint)(address >> 40);
        // VGPRS = 63 blocks of 4 (wave32 granule 8: generous), SGPRS generous.
        Shader[Sh.PgmRsrc1Gs] = 0x3F | (0xF << 6);
        Shader[Sh.PgmRsrc2Gs] = ((uint)userData.Length & 0x1F) << 1 | ((uint)userData.Length >> 5) << 27;
        for (var index = 0u; index < userData.Length; index++)
        {
            Shader[Sh.UserDataGs0 + index] = userData[index];
        }

        return this;
    }

    /// <summary>The pixel program with its user data in s0 onward; the primitive mask follows it.</summary>
    public SceneState PixelShader(ulong address, params uint[] userData)
    {
        Shader[Sh.PgmLoPs] = (uint)(address >> 8);
        Shader[Sh.PgmHiPs] = (uint)(address >> 40);
        Shader[Sh.PgmRsrc1Ps] = 0x3F | (0xF << 6);
        Shader[Sh.PgmRsrc2Ps] = ((uint)userData.Length & 0x1F) << 1 | ((uint)userData.Length >> 5) << 27;
        for (var index = 0u; index < userData.Length; index++)
        {
            Shader[Sh.UserDataPs0 + index] = userData[index];
        }

        return this;
    }

    /// <summary>
    /// The compute program with its user data in s0 onward. Thread-group ids X/Y/Z follow
    /// the user data when enabled; local ids arrive in v0 (and v1/v2 for 2D/3D groups).
    /// </summary>
    public SceneState ComputeShader(ulong address, uint threadsX, uint threadsY, uint threadsZ, uint[] userData,
        bool groupIdX = true, bool groupIdY = false, bool groupIdZ = false, uint threadIdComponents = 0, uint ldsBytes = 0)
    {
        Shader[Sh.ComputePgmLo] = (uint)(address >> 8);
        Shader[Sh.ComputePgmHi] = (uint)(address >> 40);
        Shader[Sh.ComputePgmRsrc1] = 0x3F | (0xF << 6);
        Shader[Sh.ComputePgmRsrc2] = (((uint)userData.Length & 0x1F) << 1) | (groupIdX ? 1u << 7 : 0u) |
            (groupIdY ? 1u << 8 : 0u) | (groupIdZ ? 1u << 9 : 0u) | (threadIdComponents << 11) |
            (((ldsBytes + 511) / 512) << 15);
        Shader[Sh.ComputeNumThreadX] = threadsX;
        Shader[Sh.ComputeNumThreadY] = threadsY;
        Shader[Sh.ComputeNumThreadZ] = threadsZ;
        for (var index = 0u; index < userData.Length; index++)
        {
            Shader[Sh.ComputeUserData0 + index] = userData[index];
        }

        return this;
    }

    /// <summary>Writes every register, one packet each, in bank order.</summary>
    public ScenePm4 Emit(ScenePm4 pm4)
    {
        foreach (var (offset, value) in Uconfig)
        {
            pm4.Uconfig(offset, value);
        }

        foreach (var (offset, value) in Shader)
        {
            pm4.Shader(offset, value);
        }

        foreach (var (offset, value) in Context)
        {
            pm4.Context(offset, value);
        }

        return pm4;
    }
}

/// <summary>Resource descriptor words, encoded as the gfx10 hardware reads them.</summary>
internal static class Descriptors
{
    public const uint IdentitySelect = 0xFAC;

    /// <summary>A V# for structured (idxen) loads: <paramref name="stride"/> bytes per record.</summary>
    public static uint[] Buffer(ulong address, uint stride, uint records, uint format = 77, uint oobSelect = 0) =>
    [
        (uint)address,
        ((uint)(address >> 32) & 0xFFFF) | (stride << 16),
        records,
        IdentitySelect | (format << 12) | (1u << 24) | (oobSelect << 28),
    ];

    /// <summary>
    /// A T# over a linear or tiled image. <paramref name="type"/> 9 = 2D, 13 = 2D array, 11 = cube, 10 = 3D. Word 5 MAX_MIP is
    /// the resource's mip count minus one, independent of the view's LAST_LEVEL; <paramref name="maxMip"/> defaults to LAST_LEVEL.
    /// </summary>
    public static uint[] Texture(ulong address, uint format, uint width, uint height, uint type = 9, uint tileMode = 0,
        uint baseLevel = 0, uint lastLevel = 0, uint depthOrLastLayer = 0, uint baseArray = 0, uint dstSelect = IdentitySelect,
        uint minLodFixed = 0, uint? maxMip = null)
    {
        var shifted = address >> 8;
        var lastX = width - 1;
        var lastY = height - 1;
        return
        [
            (uint)shifted,
            ((uint)(shifted >> 32) & 0xFF) | (minLodFixed << 8) | (format << 20) | ((lastX & 0x3) << 30),
            ((lastX >> 2) & 0xFFF) | (lastY << 14),
            dstSelect | (baseLevel << 12) | (lastLevel << 16) | (tileMode << 20) | (type << 28),
            depthOrLastLayer | (baseArray << 16),
            (maxMip ?? lastLevel) << 4,
            0,
            0,
        ];
    }

    /// <summary>
    /// An S#. Clamp 0 = wrap, 1 = mirror, 2 = clamp-to-last, 4 = clamp-half-border, 6 = clamp-to-border; filters 0 = point, 1 = bilinear;
    /// mip filter 0 = none, 1 = point, 2 = linear. LOD fields are unsigned 4.8 fixed point; bias is signed 6.8.
    /// </summary>
    public static uint[] Sampler(uint clampX = 2, uint clampY = 2, uint clampZ = 2, CompareFunc depthCompare = CompareFunc.Never,
        uint magFilter = 0, uint minFilter = 0, uint mipFilter = 0, float minLod = 0, float maxLod = 15, float lodBias = 0,
        uint borderColorType = 0, uint maxAniso = 0, bool forceUnnormalized = false, uint zFilter = 0, bool forceDegamma = false,
        uint filterMode = 0)
    {
        static uint Fixed(float value, int bits) => (uint)Math.Clamp((int)MathF.Round(value * 256f), 0, (1 << bits) - 1);
        var bias = (uint)((int)MathF.Round(lodBias * 256f) & 0x3FFF);
        return
        [
            clampX | (clampY << 3) | (clampZ << 6) | (maxAniso << 9) | ((uint)depthCompare << 12) | (forceUnnormalized ? 1u << 15 : 0u) |
                (forceDegamma ? 1u << 20 : 0u) | (filterMode << 29),
            Fixed(minLod, 12) | (Fixed(maxLod, 12) << 12),
            bias | (magFilter << 20) | (minFilter << 22) | (zFilter << 24) | (mipFilter << 26),
            borderColorType << 30,
        ];
    }

    public static uint Lo(ulong address) => (uint)address;

    public static uint Hi(ulong address) => (uint)(address >> 32);
}
