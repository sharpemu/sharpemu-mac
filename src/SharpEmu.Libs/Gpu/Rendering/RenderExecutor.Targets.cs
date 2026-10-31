// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.VideoOut;
using SharpEmu.Libs.Gpu.Pipelines;

namespace SharpEmu.Libs.Gpu.Rendering;

public sealed partial class RenderExecutor
{
    internal static readonly bool DropMixedSampleDepth =
        Environment.GetEnvironmentVariable("SHARPEMU_MIXED_SAMPLES") == "drop-depth";

    // Every draw resolves its targets against layer zero of the view range.
    private const uint DrawLayerOffset = 0;
    private static int _targetVersions;

    private bool TryResolveDrawTargets(RegisterBanks banks, in DrawCall draw, ref DrawState state, ResolvedDraw record)
    {
        using var profileScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.DrawTargetResolution);
        if (Diagnostics.DccWriterTrace.Enabled)
        {
            Diagnostics.DccWriterTrace.CurrentKind = "draw";
            Diagnostics.DccWriterTrace.CurrentProgram = 0;
        }

        var context = banks.Context;
        if (TryResolveMultisampleColor(context, record))
        {
            TraceDrawDisposition(banks, in draw, "multisample-color-resolve");
            return false;
        }

        var layout = ResolveTargetLayout(banks);
        for (var index = 0; index < layout.ColorCount; index++)
        {
            ref readonly var resolution = ref layout.Colors[index];
            if (MeshDrawTrace.Active) MeshDrawTrace.Write("color-request", $"slot={layout.ColorSlots[index]} address=0x{resolution.Request.Description.Data.Address:X16} size=0x{resolution.Request.Description.Data.Size:X} view={resolution.Request.View}");
            state.Colors[(int)state.ColorCount++] = new ColorTargetState(in resolution, layout.ColorSlots[index], ResourceSlotIdentifier.Invalid);
        }

        if (layout.Depth is { } depthTarget)
        {
            if (MeshDrawTrace.Active) MeshDrawTrace.Write("depth-request", $"address=0x{depthTarget.Target.Request.Description.Data.Address:X16} size=0x{depthTarget.Target.Request.Description.Data.Size:X} view={depthTarget.Target.Request.View}");
            state.Depth = new DepthAttachmentState(in depthTarget, ResourceSlotIdentifier.Invalid);
            state.DepthResolved = true;
        }

        state.PixelActive = layout.PixelActive;
        if (state.ColorCount == 0 && !state.Depth.HasTarget && !state.PixelActive &&
            VertexStageWritesMemory(banks, ref state))
        {
            TraceDrawDisposition(banks, in draw, "vertex-stores-only");
            return true;
        }

        if (state.ColorCount == 0 && !state.Depth.HasTarget && !state.PixelActive &&
            banks.Shader.Pixel.Address != 0 && PixelStageWritesMemory(banks, ref state))
        {
            // No attachment, but the pixel program stores to memory (a storage image, a
            // written buffer or a device address): a lookup-table bake or similar pass whose
            // only output is what it stores. It runs for those stores alone.
            TraceDrawDisposition(banks, in draw, "pixel-stores-only");
            return true;
        }

        if (state.ColorCount == 0 && !state.Depth.HasTarget && !state.PixelActive)
        {
            TraceDrawDisposition(banks, in draw, "no-framebuffer");
            if (RenderTrace.Enabled && RenderTrace.FramebufferSkip())
            {
                RenderTrace.Write(
                    $"Skipping a draw without a framebuffer: name={draw.Name} count={draw.Count} targetMask=0x{context.RenderTargetMask:X8} " +
                    $"pixel=0x{banks.Shader.Pixel.Address:X16} colorShaderMask=0x{context.ShaderInterface.ColorShaderMask:X8}");
            }

            return false;
        }

        return true;
    }

    internal bool DrawHasTargets(RegisterBanks banks)
    {
        if (ConsumesColorMetadataOperation(banks.Context))
        {
            return false;
        }

        var layout = ResolveTargetLayout(banks);
        return layout.ColorCount != 0 || layout.Depth is not null;
    }

    private DrawResolution ResolveTargetLayout(RegisterBanks banks)
    {
        var resolution = _drawResolution;
        var contextStamp = banks.Stamp(RegisterStamp.ColorTargets, RegisterStamp.DepthTargets, RegisterStamp.ShaderInterface);
        var programsStamp = banks.Stamp(RegisterStamp.ShaderPrograms);
        if (resolution.TargetContextStamp == contextStamp && resolution.TargetProgramsStamp == programsStamp)
        {
            return resolution;
        }

        var context = banks.Context;
        resolution.ColorCount = 0;
        for (var slot = 0u; slot < ContextRegisters.ColorTargetCount; slot++)
        {
            if (slot != 0 && (context.RenderTargetMaskForSlot(slot) == 0 || context.ColorTargets[slot].BaseAddress == 0))
            {
                continue;
            }

            if (ColorTargetResolver.Resolve(context, slot, DrawLayerOffset, ignoreTargetMask: false, out var resolvedSlot) is not { } color)
            {
                continue;
            }

            resolution.Colors[resolution.ColorCount] = color;
            resolution.ColorSlots[resolution.ColorCount++] = resolvedSlot;
        }

        resolution.Depth = DepthTargetResolver.Resolve(context, _host.FormatSupport, HostFatal);
        resolution.PixelActive = HasActivePixelShader(banks);
        resolution.TargetContextStamp = contextStamp;
        resolution.TargetProgramsStamp = programsStamp;
        resolution.TargetVersion = (uint)Interlocked.Increment(ref _targetVersions);
        return resolution;
    }

    private bool VertexStageWritesMemory(RegisterBanks banks, ref DrawState state)
    {
        if (banks.Shader.Vertex.ExportAddress == 0)
        {
            return false;
        }

        ResolveShaderPrograms(banks, ref state);
        var stage = state.Programs.VertexInput.Stage;
        return stage.Program is { } program && (WritesStorageImage(program) || HasBufferWrites(stage));
    }

    // Whether the pixel program of an attachment-less draw stores to memory. The pixel stage
    // only decodes once it is active, so the program is compiled to find out; the answer is
    // remembered per pixel program address because such draws arrive by the thousand. On a
    // positive answer the pixel stage stays active; otherwise the state is left as it was.
    private bool PixelStageWritesMemory(RegisterBanks banks, ref DrawState state)
    {
        var address = banks.Shader.Pixel.Address;
        if (_pixelStoreDecisions.TryGetValue(address, out var known))
        {
            state.PixelActive = known;
            return known;
        }

        state.PixelActive = true;
        state.Programs = null!;
        ResolveShaderPrograms(banks, ref state);
        var writes = state.Programs.Available && DrawWritesMemory(state.Programs.PixelInput.Stage);
        state.PixelActive = writes;
        // The programs were resolved with the pixel stage on; a draw that does not store resolves them again.
        if (!writes)
        {
            state.Programs = null!;
        }

        _pixelStoreDecisions[address] = writes;
        return writes;
    }

    private static bool TryResolveMultisampleColor(ContextRegisters context, ResolvedDraw record)
    {
        if (context.ColorControl.Mode != ColorModeResolve)
        {
            return false;
        }

        if (context.ColorTargets[0].BaseAddress == 0 || context.ColorTargets[1].BaseAddress == 0)
        {
            return false;
        }

        var source = ResolveMultisampleSlot(context, 0);
        var destination = ResolveMultisampleSlot(context, 1);
        if (source is not { } from || destination is not { } to)
        {
            return false;
        }

        record.Kind = ResolvedDrawKind.MultisampleResolve;
        record.ResolveSource = from;
        record.ResolveDestination = to;
        return true;
    }

    private bool TryResolveDepthStencilCopy(ContextRegisters context, ResolvedDraw record)
    {
        if (context.ColorControl.Mode != 0)
        {
            return false;
        }

        ref readonly var words = ref context.DepthTarget;
        var renderOverride = context.DepthRenderOverride;
        var depthCopy =
            renderOverride.ForceZDirty &&
            renderOverride.ForceZValid &&
            words.DepthFormat != GuestDepthFormat.Invalid &&
            words.ZReadBase != 0 &&
            words.ZWriteBase != 0 &&
            words.ZReadBase != words.ZWriteBase;
        var stencilCopy =
            renderOverride.ForceStencilDirty &&
            renderOverride.ForceStencilValid &&
            words.StencilFormat != GuestStencilFormat.Invalid &&
            words.StencilReadBase != 0 &&
            words.StencilWriteBase != 0 &&
            words.StencilReadBase != words.StencilWriteBase;
        if (!depthCopy && !stencilCopy)
        {
            return false;
        }

        var read = ImageRequestBuilders.DepthTargetCopy(in words, _host.FormatSupport, writeBuffer: false)
            ?? throw _host.Fatal("A depth/stencil copy has no readable depth target.");
        var write = ImageRequestBuilders.DepthTargetCopy(in words, _host.FormatSupport, writeBuffer: true)
            ?? throw _host.Fatal("A depth/stencil copy has no writable depth target.");
        record.Kind = ResolvedDrawKind.DepthStencilCopy;
        record.CopyRead = read.Request;
        record.CopyWrite = write.Request;
        record.CopyReadFormat = read.Format;
        record.CopyWriteFormat = write.Format;
        record.CopyExtent = new GpuExtent3D(write.Width, write.Height, 1);
        record.CopyAspects = (depthCopy ? GpuImageAspect.DepthBit : 0) | (stencilCopy ? GpuImageAspect.StencilBit : 0);
        return true;
    }

    private static ColorTargetState? ResolveMultisampleSlot(ContextRegisters context, uint slot)
    {
        if (ColorTargetResolver.Resolve(context, slot, DrawLayerOffset, ignoreTargetMask: true, out var resolvedSlot) is not { } resolution)
        {
            return null;
        }

        return new ColorTargetState(in resolution, resolvedSlot, ResourceSlotIdentifier.Invalid);
    }

    // A depth pass can leave a color target bound that it never writes (target mask 0, or a
    // slot the pixel program does not export). Hardware bounds such a draw only by the
    // scissor, but a host attachment also bounds the render area: a 1024x1024 color target
    // left bound by Astro Bot's depth clear kept the clear out of most of a 1080p depth buffer.
    // The pixel outputs drop the same slots (ShaderPipelineCache.ResolveBoundTargets), so host
    // locations stay dense and aligned with the remaining attachments.
    // The provider also asks for the drop when it leaves a single target for dual-source
    // alpha blending (ShaderPipelineCache.LeavesSingleDualSourceTarget).
    internal static void DropUnwrittenColorTargets(ContextRegisters context, ref DrawState state, ShaderProgramInfo? pixelProgram)
    {
        if (!state.Depth.HasTarget && !state.Programs.DropUnwrittenColorTargets)
        {
            return;
        }

        var exportMasks = pixelProgram?.PixelColorExportMasks ?? 0u;
        var kept = 0u;
        for (var i = 0; i < state.ColorCount; i++)
        {
            ref readonly var target = ref state.Colors[i];
            if (IsUnwrittenColorTarget(context, target.Slot, target.Resolution.ExportMapping, exportMasks))
            {
                continue;
            }

            state.Colors[(int)kept++] = target;
        }

        state.ColorCount = kept;
    }

    internal static uint WrittenColorTargets(ContextRegisters context, in DrawState state, ShaderProgramInfo? pixelProgram)
    {
        var all = (1u << (int)state.ColorCount) - 1;
        if (!state.Depth.HasTarget && !state.Programs.DropUnwrittenColorTargets)
        {
            return all;
        }

        var exportMasks = pixelProgram?.PixelColorExportMasks ?? 0u;
        var written = 0u;
        for (var i = 0; i < state.ColorCount; i++)
        {
            ref readonly var target = ref state.Colors[i];
            if (!IsUnwrittenColorTarget(context, target.Slot, target.Resolution.ExportMapping, exportMasks))
            {
                written |= 1u << i;
            }
        }

        return written;
    }

    // The export that reaches the slot is the one routed to it, not the export of the same
    // number: exports are compacted onto the slots CB_SHADER_MASK enables.
    internal static bool IsUnwrittenColorTarget(ContextRegisters context, uint slot, ColorComponentMap exportMapping, uint pixelColorExportMasks)
    {
        if (exportMapping.ApplyMask(context.RenderTargetMaskForSlot(slot)) == 0)
        {
            return true;
        }

        var export = PixelExportRouting.ExportForSlot(context.ShaderInterface, slot);
        return export < 0 || ((pixelColorExportMasks >> (export * 4)) & 0xFu) == 0;
    }
}
