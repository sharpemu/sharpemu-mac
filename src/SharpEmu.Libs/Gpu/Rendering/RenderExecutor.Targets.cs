// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE.Diagnostics;
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
            // Written in place, field by field, as the constructor would: no stack copy of the resolution.
            ref var color = ref state.Colors[(int)state.ColorCount++];
            color.Resolution = resolution;
            color.Slot = layout.ColorSlots[index];
            color.Image = ResourceSlotIdentifier.Invalid;
            color.View = default;
        }

        if (layout.Depth.HasValue)
        {
            ref readonly var depthTarget = ref Nullable.GetValueRefOrDefaultRef(in layout.Depth);
            if (MeshDrawTrace.Active) MeshDrawTrace.Write("depth-request", $"address=0x{depthTarget.Target.Request.Description.Data.Address:X16} size=0x{depthTarget.Target.Request.Description.Data.Size:X} view={depthTarget.Target.Request.View}");
            state.Depth.Target = depthTarget;
            state.Depth.Image = ResourceSlotIdentifier.Invalid;
            state.Depth.View = default;
            state.Depth.MetadataClear = false;
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

    // A draw's targets are kept across program changes that leave its target registers alone, and
    // the recorder reuses the previous draw's attachments for them (RenderRecorder.RepeatsPreviousTargets).
    // SHARPEMU_WIDE_TARGET_REUSE=0 re-resolves the targets whenever the programs change.
    internal static readonly bool WideTargetReuse = Environment.GetEnvironmentVariable("SHARPEMU_WIDE_TARGET_REUSE") != "0";

    private DrawResolution ResolveTargetLayout(RegisterBanks banks)
    {
        var resolution = _drawResolution;
        var contextStamp = banks.Stamp(RegisterStamp.ColorTargets, RegisterStamp.DepthTargets, RegisterStamp.ShaderInterface);
        var programsStamp = banks.Stamp(RegisterStamp.ShaderPrograms);
        if (resolution.TargetContextStamp == contextStamp && resolution.TargetProgramsStamp == programsStamp)
        {
            return resolution;
        }

        // The programs decide only whether a pixel stage runs: with the target registers unchanged,
        // a new program pair keeps the targets, and their version, unless that flips.
        if (WideTargetReuse && _host.ReusesTargetsAcrossPrograms && resolution.TargetContextStamp == contextStamp &&
            resolution.TargetVersion != 0)
        {
            resolution.TargetProgramsStamp = programsStamp;
            var pixelActive = HasActivePixelShader(banks);
            if (pixelActive == resolution.PixelActive)
            {
                return resolution;
            }

            resolution.PixelActive = pixelActive;
            resolution.TargetVersion = (uint)Interlocked.Increment(ref _targetVersions);
            return resolution;
        }

        var context = banks.Context;
        if (TargetValueReuse && resolution.TargetVersion != 0 && resolution.Inputs.Matches(context))
        {
            // Target registers written again with the values they held (a material or program bind rewrites the
            // shader interface, the depth control or the target mask with each draw): the same targets, under the
            // same version, so the recorder keeps their acquisitions (RenderRecorder.RepeatsPreviousTargets).
            if (TargetValueReuseVerify)
            {
                VerifyTargetLayout(context, resolution);
            }

            resolution.TargetContextStamp = contextStamp;
            resolution.TargetProgramsStamp = programsStamp;
            var pixelActive = HasActivePixelShader(banks);
            if (pixelActive == resolution.PixelActive)
            {
                _targetValuesKept.Increment();
                return resolution;
            }

            _targetValuesPixelStage.Increment();
            resolution.PixelActive = pixelActive;
            resolution.TargetVersion = (uint)Interlocked.Increment(ref _targetVersions);
            return resolution;
        }

        _targetValuesResolved.Increment();
        resolution.ColorCount = ResolveColorTargets(context, resolution.Colors, resolution.ColorSlots);
        resolution.Depth = DepthTargetResolver.Resolve(context, _host.FormatSupport, HostFatal);
        resolution.Inputs.Capture(context);
        resolution.PixelActive = HasActivePixelShader(banks);
        resolution.TargetContextStamp = contextStamp;
        resolution.TargetProgramsStamp = programsStamp;
        resolution.TargetVersion = (uint)Interlocked.Increment(ref _targetVersions);
        return resolution;
    }

    internal static int ResolveColorTargets(ContextRegisters context, ColorTargetResolution[] colors, uint[] slots)
    {
        var count = 0;
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

            colors[count] = color;
            slots[count++] = resolvedSlot;
        }

        return count;
    }

    // A draw whose target registers changed stamp but not value keeps the resolved targets and their version
    // (TargetInputs). SHARPEMU_TARGET_VALUE_REUSE=0 resolves them again and gives them a new version whenever a
    // target register is written; =verify (or SHARPEMU_PERF_VERIFY=1) resolves them again on every kept draw as
    // well and stops on a difference.
    internal static readonly bool TargetValueReuse = Environment.GetEnvironmentVariable("SHARPEMU_TARGET_VALUE_REUSE") != "0";

    internal static readonly bool TargetValueReuseVerify =
        TargetValueReuse && (Environment.GetEnvironmentVariable("SHARPEMU_TARGET_VALUE_REUSE") == "verify" ||
                             Environment.GetEnvironmentVariable("SHARPEMU_PERF_VERIFY") == "1");

    private static readonly ThreadCounter _targetValuesKept = new();
    private static readonly ThreadCounter _targetValuesPixelStage = new();
    private static readonly ThreadCounter _targetValuesResolved = new();

    /// <summary>Target layouts kept on equal values, kept with only the pixel stage flipped, and resolved again.</summary>
    internal static string TakeTargetValueReport() =>
        FormattableString.Invariant(
            $"target_values=(kept={_targetValuesKept.Take()},pixel_stage={_targetValuesPixelStage.Take()},resolved={_targetValuesResolved.Take()}) ");

    private readonly ColorTargetResolution[] _verifyColors = new ColorTargetResolution[RenderingState.ColorAttachmentCapacity];
    private readonly uint[] _verifySlots = new uint[RenderingState.ColorAttachmentCapacity];

    private void VerifyTargetLayout(ContextRegisters context, DrawResolution resolution)
    {
        var count = ResolveColorTargets(context, _verifyColors, _verifySlots);
        var depth = DepthTargetResolver.Resolve(context, _host.FormatSupport, HostFatal);
        var same = count == resolution.ColorCount && SameDepth(depth, resolution.Depth);
        for (var index = 0; same && index < count; index++)
        {
            same = _verifySlots[index] == resolution.ColorSlots[index] && SameColor(in _verifyColors[index], in resolution.Colors[index]);
        }

        if (!same)
        {
            throw HostFatal(
                $"A target layout kept on equal register values resolves differently: colors={resolution.ColorCount}->{count} " +
                $"depth={resolution.Depth.HasValue}->{depth.HasValue} version={resolution.TargetVersion}.");
        }
    }

    internal static bool SameColor(in ColorTargetResolution left, in ColorTargetResolution right) =>
        SameRequest(left.Request, right.Request) && left with { Request = default } == right with { Request = default };

    internal static bool SameDepth(DepthTargetState? left, DepthTargetState? right)
    {
        if (left is not { } first || right is not { } second)
        {
            return left.HasValue == right.HasValue;
        }

        return SameRequest(first.Target.Request, second.Target.Request) &&
            first.Target with { Request = default } == second.Target with { Request = default } && first.State == second.State;
    }

    // An image request field for field; the mip layouts by their levels, not by the array that holds them.
    private static bool SameRequest(ImageRequest left, ImageRequest right)
    {
        for (var level = 0; level < ImageDescription.MaxLevels; level++)
        {
            if (!left.Description.MipLayout[level].Equals(right.Description.MipLayout[level]))
            {
                return false;
            }
        }

        var first = left;
        var second = right;
        first.Description.MipLayout = default;
        second.Description.MipLayout = default;
        return first.Description.Equals(second.Description) && first.View == second.View && first.Role == second.Role &&
            first.TraceTextureMetadataAddress == second.TraceTextureMetadataAddress &&
            string.Equals(first.TraceTextureDescriptor, second.TraceTextureDescriptor, StringComparison.Ordinal) &&
            first.TraceMetadataCompress == second.TraceMetadataCompress && first.TraceWriteCompress == second.TraceWriteCompress;
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
            // Programs the vertex-store check resolved with the pixel stage off have no pixel
            // program; a draw whose pixel stage turns on resolves them again.
            if (known)
            {
                state.Programs = null!;
            }

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
        record.TargetCopyFieldsWritten = true;
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
        record.TargetCopyFieldsWritten = true;
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
        var routing = PixelExportRouting.Routing.Of(context.ShaderInterface);
        var kept = 0u;
        for (var i = 0; i < state.ColorCount; i++)
        {
            ref readonly var target = ref state.Colors[i];
            if (IsUnwrittenColorTarget(context, in routing, target.Slot, target.Resolution.ExportMapping, exportMasks))
            {
                continue;
            }

            // Until the first dropped target every kept one is already in its place.
            if (kept != i)
            {
                state.Colors[(int)kept] = target;
            }

            kept++;
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
        var routing = PixelExportRouting.Routing.Of(context.ShaderInterface);
        var written = 0u;
        for (var i = 0; i < state.ColorCount; i++)
        {
            ref readonly var target = ref state.Colors[i];
            if (!IsUnwrittenColorTarget(context, in routing, target.Slot, target.Resolution.ExportMapping, exportMasks))
            {
                written |= 1u << i;
            }
        }

        return written;
    }

    // The export that reaches the slot is the one routed to it, not the export of the same
    // number: exports are compacted onto the slots CB_SHADER_MASK enables.
    internal static bool IsUnwrittenColorTarget(ContextRegisters context, uint slot, ColorComponentMap exportMapping, uint pixelColorExportMasks) =>
        IsUnwrittenColorTarget(context, PixelExportRouting.Routing.Of(context.ShaderInterface), slot, exportMapping, pixelColorExportMasks);

    // As above, with the registers' export routing counted once by a caller that asks for several slots.
    internal static bool IsUnwrittenColorTarget(
        ContextRegisters context, in PixelExportRouting.Routing routing, uint slot, ColorComponentMap exportMapping, uint pixelColorExportMasks)
    {
        if (exportMapping.ApplyMask(context.RenderTargetMaskForSlot(slot)) == 0)
        {
            return true;
        }

        var export = routing.ExportForSlot(slot);
        return export < 0 || ((pixelColorExportMasks >> (export * 4)) & 0xFu) == 0;
    }
}
