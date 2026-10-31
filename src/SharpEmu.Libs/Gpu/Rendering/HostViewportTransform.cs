// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.ShaderCompiler;

namespace SharpEmu.Libs.Gpu.Rendering;

/// <summary>
/// The guest's path from the exported position to window coordinates (PA_CL_CLIP_CNTL,
/// PA_CL_VTE_CNTL and viewport 0) split between the host's fixed-function viewport and the
/// conversion the vertex program applies. The pipeline cache keys the program on
/// <see cref="ClipSpace"/> and the draw state sets the rest, so both come from this one
/// resolution.
/// </summary>
public readonly record struct HostViewportTransform(
    ClipSpaceTransform ClipSpace,
    float X,
    float Y,
    float Width,
    float Height,
    float MinDepth,
    float MaxDepth,
    bool DepthClip)
{
    public const uint MaxViewportDimension = 16384;

    // PA_CL_VTE_CNTL: VPORT_{X,Y,Z}_{SCALE,OFFSET}_ENA in bits 0-5, then VTX_XY_FMT (x and y
    // are already divided by w), VTX_Z_FMT (z is) and VTX_W0_FMT (w is w, not 1/w).
    private const uint XScaleEnable = 1u << 0;
    private const uint XOffsetEnable = 1u << 1;
    private const uint YScaleEnable = 1u << 2;
    private const uint YOffsetEnable = 1u << 3;
    private const uint ZScaleEnable = 1u << 4;
    private const uint ZOffsetEnable = 1u << 5;
    private const uint XyFormat = 1u << 8;
    private const uint ZFormat = 1u << 9;
    private const uint W0Format = 1u << 10;
    private const uint XyEnables = XScaleEnable | XOffsetEnable | YScaleEnable | YOffsetEnable;

    private static int _unclampedFloatWarningShown;
    private static int _clampRangeWarningShown;

    /// <param name="maxWidth">The largest host viewport width, at most <see cref="MaxViewportDimension"/>.</param>
    /// <param name="maxHeight">The largest host viewport height, at most <see cref="MaxViewportDimension"/>.</param>
    public static HostViewportTransform Resolve(ContextRegisters context, uint maxWidth, uint maxHeight)
    {
        var clip = context.Clip;
        var control = context.ScreenViewport.TransformControl;
        ref readonly var viewport = ref context.ScreenViewport.Primary;
        float x;
        float y;
        float width;
        float height;
        ClipSpaceTransform clipSpace;
        if (clip.ClipDisable || (control & XyEnables) != XyEnables)
        {
            // Window coordinates: the program applies the viewport transform against the
            // largest host viewport, at the origin and y-down. A disabled scale is 1 and a
            // disabled offset 0, which no host viewport rectangle expresses.
            width = Math.Min(maxWidth, MaxViewportDimension);
            height = Math.Min(maxHeight, MaxViewportDimension);
            x = 0;
            y = 0;
            clipSpace = new ClipSpaceTransform(
                true,
                (control & XScaleEnable) != 0 ? viewport.XScale : 1f,
                (control & YScaleEnable) != 0 ? viewport.YScale : 1f,
                (control & XOffsetEnable) != 0 ? viewport.XOffset : 0f,
                (control & YOffsetEnable) != 0 ? viewport.YOffset : 0f,
                width * 0.5f,
                height * 0.5f)
            {
                PositionYDown = true,
            };
        }
        else
        {
            // The host takes the guest's rectangle. A negative x scale mirrors it, which no
            // host viewport width expresses: the rectangle keeps its pixels with a positive
            // width and the program negates x. A positive y scale is a y-down viewport,
            // which a y-up host reaches by negating y.
            x = viewport.XOffset - Math.Abs(viewport.XScale);
            y = viewport.YOffset - viewport.YScale;
            width = Math.Abs(viewport.XScale) * 2f;
            height = viewport.YScale * 2f;
            clipSpace = new ClipSpaceTransform(false, 0, 0, 0, 0, 0, 0)
            {
                PositionXMirror = viewport.XScale < 0f,
                PositionYDown = viewport.YScale > 0f,
            };
        }

        var depth = ResolveDepth(context);
        clipSpace = clipSpace with
        {
            XyPreDivided = (control & XyFormat) != 0,
            ZPreDivided = (control & ZFormat) != 0,
            WReciprocal = (control & W0Format) == 0,
            DepthNegativeOneToOne = depth.NegativeOneToOne,
            DepthRemap = depth.Remap,
            DepthScale = depth.Scale,
            DepthOffset = depth.Offset,
            DepthPlanes = depth.Planes,
            UserClipPlanes = EnabledUserClipPlanes(context),
            UserClipPlanesCullOnly = clip.UserClipPlaneCullOnly,
        };
        return new HostViewportTransform(clipSpace, x, y, width, height, depth.MinDepth, depth.MaxDepth, depth.Clip);
    }

    // PA_CL_CLIP_CNTL.UCP_ENA_n selects the planes, in the guest's clip space; CLIP_DISABLE
    // turns the clipper off, planes included.
    private static float[]? EnabledUserClipPlanes(ContextRegisters context)
    {
        var enabled = context.Clip.ClipDisable ? 0 : context.Clip.UserClipPlanes & 0x3F;
        if (enabled == 0)
        {
            return null;
        }

        var planes = new float[System.Numerics.BitOperations.PopCount((uint)enabled) * 4];
        var next = 0;
        for (var plane = 0; plane < ContextRegisters.UserClipPlaneCount; plane++)
        {
            if ((enabled & (1 << plane)) != 0)
            {
                Array.Copy(context.UserClipPlanes, plane * 4, planes, next, 4);
                next += 4;
            }
        }

        return planes;
    }

    /// <summary>Whether the host clips depth to its viewport range (else it clamps), as <see cref="Resolve"/> decides.</summary>
    public static bool DepthClipOf(ContextRegisters context) => ResolveDepth(context).Clip;

    private readonly record struct DepthTransform(
        float MinDepth,
        float MaxDepth,
        bool Clip,
        ShaderDepthPlanes Planes,
        bool NegativeOneToOne,
        bool Remap = false,
        float Scale = 0f,
        float Offset = 0f);

    // The guest's depth is z / w * scale + offset (PA_CL_VPORT_ZSCALE/ZOFFSET), clipped in
    // its clip space at z = 0 (z = -w for a [-w, w] depth) and z = w. The host's is
    // near + z' / w' * (far - near), clipped at z' = 0 and z' = w' or clamped to [near, far].
    private static DepthTransform ResolveDepth(ContextRegisters context)
    {
        var clip = context.Clip;
        var control = context.ScreenViewport.TransformControl;
        ref readonly var viewport = ref context.ScreenViewport.Primary;
        var near = !clip.NearZClipDisable;
        var far = !clip.FarZClipDisable;
        var planes = (near ? clip.DirectXClipSpace ? ShaderDepthPlanes.NearAtZero : ShaderDepthPlanes.NearAtNegativeW : 0) |
            (far ? ShaderDepthPlanes.Far : 0);
        var identity = (control & (ZScaleEnable | ZOffsetEnable | ZFormat | W0Format)) == (ZScaleEnable | ZOffsetEnable | W0Format);
        var minDepth = viewport.ZOffset - (clip.DirectXClipSpace ? 0f : viewport.ZScale);
        var maxDepth = viewport.ZScale + viewport.ZOffset;
        if (identity && ClampMatchesTransform(context, Math.Min(minDepth, maxDepth), Math.Max(minDepth, maxDepth), near, far))
        {
            // The host's viewport range is the guest's transform, and z' is z (or (z + w) / 2
            // for a [-w, w] depth), so the host's planes are the guest's.
            if (near == far)
            {
                return new DepthTransform(minDepth, maxDepth, near, ShaderDepthPlanes.None, !clip.DirectXClipSpace);
            }

            // ZCLIP_NEAR_DISABLE and ZCLIP_FAR_DISABLE are independent, the host's depth clip
            // is not: it clamps, and the program clips the plane that is still enabled.
            return new DepthTransform(minDepth, maxDepth, false, planes, !clip.DirectXClipSpace);
        }

        // Anything else (a disabled z scale or offset, VTX_Z_FMT, a reciprocal w, or a depth
        // clamp the transform's range does not do) the program maps: z' = z_w * scale' + w' * offset', where z_w is z
        // (z * w' when VTX_Z_FMT says z is already divided), puts the guest's depth on the
        // host's viewport range, which is the guest's own depth clamp. The host clamps to
        // it and the program clips the guest's enabled planes from the exported position.
        var (clampMin, clampMax) = DepthClampRange(context);
        var span = clampMax - clampMin;
        var scale = (control & ZScaleEnable) != 0 ? viewport.ZScale : 1f;
        var offset = (control & ZOffsetEnable) != 0 ? viewport.ZOffset : 0f;
        return new DepthTransform(
            clampMin,
            clampMax,
            false,
            planes,
            false,
            true,
            span == 0f ? 0f : scale / span,
            span == 0f ? 0f : (offset - clampMin) / span);
    }

    // Whether the host's viewport range [low, high] (the guest's transform) also does the
    // guest's depth clamp. The DB clamps every fragment's depth, interpolated or exported, to
    // PA_SC_VPORT_ZMIN/ZMAX. Clipped at both planes the depth stays inside the transform,
    // so a clamp range around it never bites; any unclipped side reaches the host's clamp,
    // which is to its viewport range, so the two ranges have to be the same.
    private static bool ClampMatchesTransform(ContextRegisters context, float low, float high, bool near, bool far)
    {
        if (!(low >= 0f && high <= 1f))
        {
            // The host's viewport depth is inside [0, 1].
            return false;
        }

        ref readonly var viewport = ref context.ScreenViewport.Primary;
        if (context.DepthRenderOverride.DisableViewportClamp)
        {
            return (near && far) ||
                (low == 0f && high == 1f && context.DepthTarget.DepthFormat != GuestDepthFormat.Z32Float);
        }

        return near && far
            ? viewport.MinDepth <= low && high <= viewport.MaxDepth
            : viewport.MinDepth == low && high == viewport.MaxDepth;
    }

    // The range the DB clamps the guest's depth to: PA_SC_VPORT_ZMIN/ZMAX, or with
    // DB_RENDER_OVERRIDE.DISABLE_VIEWPORT_CLAMP only the depth format's own range.
    private static (float Min, float Max) DepthClampRange(ContextRegisters context)
    {
        ref readonly var viewport = ref context.ScreenViewport.Primary;
        if (context.DepthRenderOverride.DisableViewportClamp)
        {
            // A unorm depth holds [0, 1] whatever the clamp; a float one keeps any value,
            // which no host viewport range does: the host's [0, 1] is the nearest.
            if (context.DepthTarget.DepthFormat == GuestDepthFormat.Z32Float && Interlocked.Exchange(ref _unclampedFloatWarningShown, 1) == 0)
            {
                Console.Error.WriteLine(
                    "[GPU][WARN] An unclamped 32-bit float depth (DB_RENDER_OVERRIDE.DISABLE_VIEWPORT_CLAMP) is clamped to the " +
                    $"host's [0, 1] viewport depth: PA_CL_VTE_CNTL=0x{context.ScreenViewport.TransformControl:X8} " +
                    $"zScale={viewport.ZScale:R} zOffset={viewport.ZOffset:R} (reported once).");
            }

            return (0f, 1f);
        }

        if (viewport.MinDepth >= 0f && viewport.MinDepth <= viewport.MaxDepth && viewport.MaxDepth <= 1f)
        {
            return (viewport.MinDepth, viewport.MaxDepth);
        }

        // The host's viewport depth is an ordered range inside [0, 1]; the nearest to the
        // guest's is its ends clamped into [0, 1] and put in order.
        if (Interlocked.Exchange(ref _clampRangeWarningShown, 1) == 0)
        {
            Console.Error.WriteLine(
                "[GPU][WARN] The depth clamp range is not an ordered range inside [0, 1]; the host clamps to the nearest one: " +
                $"PA_SC_VPORT_ZMIN={viewport.MinDepth:R} PA_SC_VPORT_ZMAX={viewport.MaxDepth:R} (reported once).");
        }

        var low = Math.Clamp(viewport.MinDepth, 0f, 1f);
        var high = Math.Clamp(viewport.MaxDepth, 0f, 1f);
        return float.IsNaN(low) || float.IsNaN(high) ? (0f, 1f) : (Math.Min(low, high), Math.Max(low, high));
    }
}
