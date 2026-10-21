// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.VideoOut;

namespace SharpEmu.Libs.Gpu.Rendering;

public sealed partial class RenderExecutor
{

    // A guest bias for a fixed-point depth format scales by the host depth bits.
    //
    // A fixed-point guest depth (a Z24 surface the host keeps as D32 float) has a constant unit
    // of 2^NEG_NUM_DB_BITS, while the host's float unit is 2^(exponent(max z) - 23). The two are
    // equal for primitives whose depth lies in [0.5, 1) once the factor is scaled to 2^-24 units;
    // deeper below 0.5 the host offset is smaller by 2^(-1 - exponent). This is a documented
    // deviation: the host has no constant-unit bias on a float attachment, and refusing would
    // stop every title whose Z24 targets land on D32 float.
    public static float DepthBiasConstantFactor(float guestFactor, in PolygonOffsetRegisters offset, GpuFormat hostDepthFormat)
    {
        if (offset.DepthIsFloat)
        {
            return guestFactor;
        }

        var hostDepthBits = hostDepthFormat switch
        {
            GpuFormat.D16Unorm or GpuFormat.D16UnormS8Uint => 16,
            GpuFormat.D24UnormS8Uint => 24,
            _ => FloatHostUnitBits,
        };
        return MathF.ScaleB(guestFactor, hostDepthBits + offset.NegativeDepthBits);
    }

    // The float host unit that equals a 24-bit fixed-point one: 2^(exponent - 23) for z in [0.5, 1).
    private const int FloatHostUnitBits = 24;

    // The dynamic state of a draw from the context bank and the resolved targets.
    private DynamicDrawState BuildDynamicState(RegisterBanks banks, in DrawState state, ulong submitId, ulong packetAddress)
    {
        using var profileScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.DrawDynamicStatePreparation);
        var context = banks.Context;
        var viewportRegisters = context.ScreenViewport;
        var limits = _host.Limits;
        uint framebufferWidth;
        uint framebufferHeight;
        if (state.ColorCount > 0 && state.Colors[0].Image.IsValid)
        {
            framebufferWidth = state.Colors[0].Resolution.Extent.Width;
            framebufferHeight = state.Colors[0].Resolution.Extent.Height;
        }
        else if (state.Depth.Image.IsValid)
        {
            framebufferWidth = state.Depth.Target.Target.Width;
            framebufferHeight = state.Depth.Target.Target.Height;
        }
        else
        {
            framebufferWidth = limits.MaxFramebufferWidth;
            framebufferHeight = limits.MaxFramebufferHeight;
        }

        var scissor = ResolveScissor(viewportRegisters, context.ScanMode, framebufferWidth, framebufferHeight, recoverStaleWindow: true);
        // The pipeline cache keyed the vertex program on the other half of this transform.
        var viewport = HostViewportTransform.Resolve(context, limits.MaxViewportWidth, limits.MaxViewportHeight);

        // The host draws lines one pixel wide; validation warned once about a wider guest line.
        const float lineWidth = 1f;

        var depthTarget = state.Depth.HasTarget ? state.Depth.Target : default;
        var depthState = depthTarget.State;
        var depthFormat = depthTarget.Target.Format;
        var mode = context.RasterMode;
        var polygonOffset = context.PolygonOffset;
        // A front-facing triangle takes the front offset under POLY_OFFSET_FRONT_ENABLE, a
        // back-facing one the back offset under POLY_OFFSET_BACK_ENABLE, and points and lines
        // the front offset under POLY_OFFSET_PARA_ENABLE; a face whose enable is off is not
        // biased. A host sets one bias per draw, so a draw whose two visible faces differ is
        // marked per-face and the host draws each face with its own.
        var frontOffset = mode.PolygonOffsetFrontEnable || mode.PolygonOffsetParaEnable
            ? (DepthBiasConstantFactor(polygonOffset.FrontOffset, in polygonOffset, depthFormat), polygonOffset.FrontScale / 16f)
            : (0f, 0f);
        var front = mode.PolygonOffsetFrontEnable ? frontOffset : (0f, 0f);
        var back = mode.PolygonOffsetBackEnable
            ? (DepthBiasConstantFactor(polygonOffset.BackOffset, in polygonOffset, depthFormat), polygonOffset.BackScale / 16f)
            : (0f, 0f);
        var pointLine = mode.PolygonOffsetParaEnable ? frontOffset : (0f, 0f);
        var frontVisible = !mode.CullFront;
        var backVisible = !mode.CullBack;
        var depthBiasPerFace = frontVisible && backVisible && front != back;
        var (depthBiasConstant, depthBiasSlope) = frontVisible ? front : back;
        var depthBiasEnabled = (frontVisible && mode.PolygonOffsetFrontEnable) ||
            (backVisible && mode.PolygonOffsetBackEnable) || mode.PolygonOffsetParaEnable;

        byte colorWriteMask = 0;
        for (var i = 0; i < state.ColorCount; i++)
        {
            if (context.RenderTargetMaskForSlot(state.Colors[i].Slot) != 0)
            {
                colorWriteMask |= (byte)(1 << i);
            }
        }

        var blend = context.BlendColor;
        var dynamicState = new DynamicDrawState(
            viewport.X,
            viewport.Y,
            viewport.Width,
            viewport.Height,
            viewport.MinDepth,
            viewport.MaxDepth,
            scissor,
            lineWidth,
            blend.Red,
            blend.Green,
            blend.Blue,
            blend.Alpha,
            depthState.DepthTestEnabled,
            // The DB writes depth only when it tests it: Z_WRITE_ENABLE without Z_ENABLE
            // leaves the buffer alone, which a host that writes under an always-pass compare
            // would not.
            depthState.DepthTestEnabled && depthState.DepthWriteEnabled && !depthState.DepthClearEnabled,
            depthState.DepthCompare,
            depthBiasEnabled,
            depthBiasConstant,
            depthBiasEnabled ? polygonOffset.Clamp : 0f,
            depthBiasSlope,
            // The bounds test reads the stored depth, so without a depth target there is
            // nothing to test (depthState is the default then).
            depthState.DepthBoundsTestEnabled,
            depthState.DepthMinBounds,
            depthState.DepthMaxBounds,
            depthState.StencilTestEnabled,
            depthState.FrontMasks,
            depthState.BackMasks,
            state.ColorCount,
            colorWriteMask,
            depthBiasPerFace,
            depthBiasPerFace ? back.Item1 : 0f,
            depthBiasPerFace ? back.Item2 : 0f,
            pointLine.Item1,
            pointLine.Item2,
            context.DepthCountControl);
        if (ScissorTrace.Instance is { Active: true } trace)
            trace.Draw(banks, submitId, packetAddress, framebufferWidth, framebufferHeight,
                state.ColorCount > 0 ? state.Colors[0].Resolution.BaseAddress : 0,
                state.Depth.HasTarget ? state.Depth.Target.Target.DepthAddress : 0, in dynamicState);
        return dynamicState;
    }

    // The screen, window, generic, viewport and clip rectangles intersected and clamped to the extent.
    public static ScissorRectangle ResolveScissor(ScreenViewportRegisters viewport, in ScanModeRegisters scanMode, uint width, uint height,
        bool recoverStaleWindow = false)
    {
        var screen = new ScissorRectangle(viewport.ScreenScissorLeft, viewport.ScreenScissorTop, viewport.ScreenScissorRight, viewport.ScreenScissorBottom);
        var final = screen;
        if (!screen.IsSet)
        {
            final = new ScissorRectangle(0, 0, (int)width, (int)height);
            if (RenderTrace.Enabled && RenderTrace.ScissorDefault())
            {
                RenderTrace.Write($"Defaulting an unset screen scissor to the framebuffer extent: width={width} height={height}");
            }
        }

        var window = new ScissorRectangle(viewport.WindowScissorLeft, viewport.WindowScissorTop, viewport.WindowScissorRight, viewport.WindowScissorBottom);
        // Some AGC submissions restore a full-target viewport after a smaller offscreen pass
        // without rewriting PA_SC_WINDOW_SCISSOR. The stale window can be smaller in only
        // one dimension (e.g. 75% of the target height), so check both axes independently.
        if (window.IsSet && !(recoverStaleWindow && IsStaleOffscreenWindow(viewport, in scanMode, width, height, in window)))
        {
            if (viewport.WindowScissorWindowOffsetEnable)
            {
                window = window.Offset(viewport.WindowOffsetX, viewport.WindowOffsetY);
            }

            final = final.Intersect(in window);
        }

        var generic = new ScissorRectangle(viewport.GenericScissorLeft, viewport.GenericScissorTop, viewport.GenericScissorRight, viewport.GenericScissorBottom);
        if (generic.IsSet)
        {
            if (viewport.GenericScissorWindowOffsetEnable)
            {
                generic = generic.Offset(viewport.WindowOffsetX, viewport.WindowOffsetY);
            }

            final = final.Intersect(in generic);
        }

        ref readonly var first = ref viewport.Viewports[0];
        var viewportScissor = new ScissorRectangle(first.ScissorLeft, first.ScissorTop, first.ScissorRight, first.ScissorBottom);
        if (scanMode.ViewportScissorEnable && viewportScissor.IsSet)
        {
            if (first.ScissorWindowOffsetEnable)
            {
                viewportScissor = viewportScissor.Offset(viewport.WindowOffsetX, viewport.WindowOffsetY);
            }

            final = final.Intersect(in viewportScissor);
        }

        if (viewport.ClipRectangleRule == 0)
        {
            final = new ScissorRectangle(0, 0, 0, 0);
        }
        else if (viewport.ClipRectangleRule != 0xFFFF)
        {
            if (TryClipRuleToIntersectionMask(viewport.ClipRectangleRule, out var clipMask))
            {
                for (var i = 0; i < ScreenViewportRegisters.ClipRectangleCount; i++)
                {
                    if ((clipMask & (1 << i)) == 0)
                    {
                        continue;
                    }

                    var clip = new ScissorRectangle(viewport.ClipRectangleLeft[i], viewport.ClipRectangleTop[i], viewport.ClipRectangleRight[i], viewport.ClipRectangleBottom[i]);
                    if (viewport.ClipRectangleWindowOffsetEnable[i])
                    {
                        clip = clip.Offset(viewport.WindowOffsetX, viewport.WindowOffsetY);
                    }

                    final = final.Intersect(in clip);
                }
            }
            else if (RenderTrace.Enabled && RenderTrace.ClipRuleWarning())
            {
                RenderTrace.Write($"Leaving the scissor unchanged for an unsupported clip rectangle rule: rule=0x{viewport.ClipRectangleRule:X4}");
            }
        }

        return ClampScissor(final, width, height);
    }

    private static bool IsStaleOffscreenWindow(ScreenViewportRegisters viewport, in ScanModeRegisters scanMode,
        uint width, uint height, in ScissorRectangle window)
    {
        if (!scanMode.ViewportScissorEnable || viewport.WindowScissorWindowOffsetEnable ||
            width == 0 || height == 0 || window.Left != 0 || window.Top != 0 ||
            window.Right <= 0 || window.Bottom <= 0 ||
            window.Right >= width && window.Bottom >= height)
        {
            return false;
        }

        ref readonly var active = ref viewport.Viewports[0];
        return !active.ScissorWindowOffsetEnable &&
            active.ScissorLeft == 0 && active.ScissorTop == 0 &&
            active.ScissorRight == width && active.ScissorBottom == height &&
            MathF.Abs(active.XScale * 2f) >= width && MathF.Abs(active.YScale * 2f) >= height &&
            MathF.Abs(active.XOffset - active.XScale) < 1f &&
            MathF.Abs(active.YOffset - MathF.Abs(active.YScale)) < 1f;
    }

    private static ScissorRectangle ClampScissor(ScissorRectangle rectangle, uint width, uint height)
    {
        var maxRight = (int)width;
        var maxBottom = (int)height;
        var clamped = new ScissorRectangle(
            Math.Clamp(rectangle.Left, 0, maxRight),
            Math.Clamp(rectangle.Top, 0, maxBottom),
            Math.Clamp(rectangle.Right, 0, maxRight),
            Math.Clamp(rectangle.Bottom, 0, maxBottom));
        return clamped.IsValid ? clamped : clamped with { Right = clamped.Left, Bottom = clamped.Top };
    }

    // A rule is the set of rectangle combinations that pass; only "all of a subset" rules are supported.
    private static bool TryClipRuleToIntersectionMask(ushort rule, out byte mask)
    {
        for (var candidate = 0u; candidate < 16; candidate++)
        {
            ushort combinations = 0;
            for (var combination = 0u; combination < 16; combination++)
            {
                if ((combination & candidate) == candidate)
                {
                    combinations |= (ushort)(1u << (int)combination);
                }
            }

            if (combinations == rule)
            {
                mask = (byte)candidate;
                return true;
            }
        }

        mask = 0;
        return false;
    }
}
