// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Images;

namespace SharpEmu.Libs.Gpu.Rendering;

// The draw paths kept beside the ported executor: they act on the resolved programs before the draw records.
public sealed partial class RenderExecutor
{
    private const uint DccFastClearVertexCount = 4;
    private const byte BlendFactorOne = 1;
    private const byte BlendFactorOneMinusSourceAlpha = 5;
    private const byte BlendFunctionAdd = 0;
    private const float ClipSpaceTolerance = 0.001f;

    // False when the draw does not record: no programs, a solid clear, a metadata-only quad or a retained draw.
    private bool ApplyProgramAdaptations(RegisterBanks banks, in DrawCall draw, ref DrawState state, in TargetlessDrawArguments arguments, ResolvedDraw record)
    {
        var programs = state.Programs;
        if (!programs.Available)
        {
            if (RenderTrace.Enabled)
            {
                RenderTrace.Write($"Skipping a draw without programs: name={draw.Name} export=0x{banks.Shader.Vertex.ExportAddress:X16} pixel=0x{banks.Shader.Pixel.Address:X16}");
            }

            return false;
        }

        if (programs.SolidClear is { } clear && state.ColorCount != 0)
        {
            TraceDrawDisposition(banks, in draw, "color-clear");
            record.Kind = ResolvedDrawKind.ClearColorTargets;
            record.Clear = clear;
            return false;
        }

        if (IsMetadataClearQuad(banks, in draw, ref state))
        {
            TraceDrawDisposition(banks, in draw, "metadata-clear-quad");
            return false;
        }

        if (DepthBoundsDiscardEverything(ref state))
        {
            // Every fragment fails the bounds test and the draw stores nothing, so it has no
            // effect at all; a host test cannot take an empty range (Metal's needs min <= max).
            TraceDrawDisposition(banks, in draw, "empty-depth-bounds");
            return false;
        }

        if (MayRetainTargetlessDraw(ref state) && _host.TryRetainTargetlessDraw(banks, programs, in arguments))
        {
            TraceTargetlessAttachments(banks, arguments.SubmitId);
            TraceDrawDisposition(banks, in draw, "retained-targetless-draw");
            return false;
        }

        return true;
    }

    // Whether the draw tests depth bounds that hold no stored depth and has no effect
    // besides its fragments. The DB passes a fragment whose stored depth d satisfies
    // min <= d <= max; a stored depth is within [0, 1], so a range that holds none of
    // [0, 1] (min > max, min > 1 or max < 0) discards every fragment. The vertex stage
    // still runs then and the pixel stage may run before the discard, so a draw whose
    // programs store to memory records as usual and the host has to take the range.
    // A clearing draw records as usual too.
    private bool DepthBoundsDiscardEverything(ref DrawState state)
    {
        if (!state.Depth.HasTarget)
        {
            return false;
        }

        ref readonly var target = ref state.Depth.Target;
        var depth = target.State;
        if (!depth.DepthBoundsTestEnabled || depth.DepthClearEnabled || depth.StencilClearEnabled ||
            (ViewFormatRules.DepthAspects(target.Target.Format) & GpuImageAspect.DepthBit) == 0 ||
            float.IsNaN(depth.DepthMinBounds) || float.IsNaN(depth.DepthMaxBounds) ||
            Math.Max(depth.DepthMinBounds, 0f) <= Math.Min(depth.DepthMaxBounds, 1f))
        {
            return false;
        }

        return !(state.PixelActive && DrawWritesMemory(state.Programs.PixelInput.Stage)) &&
            !DrawWritesMemory(state.Programs.VertexInput.Stage);
    }

    // A targetless draw whose pixel program samples images and stores nothing may be
    // retained and replayed later, so it needs its counts. Later draws can read stores
    // before flip, so a draw that stores is never deferred.
    private bool MayRetainTargetlessDraw(ref DrawState state) =>
        state.ColorCount == 0 && !state.Depth.HasTarget && state.PixelActive &&
        state.Programs.PixelInput.Stage.Program is { Images.Length: > 0 } &&
        !DrawWritesMemory(state.Programs.PixelInput.Stage) && !DrawWritesMemory(state.Programs.VertexInput.Stage);

    private static void TraceDrawDisposition(RegisterBanks banks, in DrawCall draw, string reason)
    {
        if (!RenderTrace.Enabled)
        {
            return;
        }

        ref readonly var viewport = ref banks.Context.ScreenViewport.Primary;
        RenderTrace.Write(
            $"DrawDisposition name={draw.Name} reason={reason} count={draw.Count} " +
            $"export=0x{banks.Shader.Vertex.ExportAddress:X16} pixel=0x{banks.Shader.Pixel.Address:X16} " +
            $"colorMode={banks.Context.ColorControl.Mode} targetMask=0x{banks.Context.RenderTargetMask:X8} " +
            $"zScale={viewport.ZScale} zOffset={viewport.ZOffset}");
    }

    private static void TraceTargetlessAttachments(RegisterBanks banks, ulong submitId)
    {
        if (!RenderTrace.Enabled)
        {
            return;
        }

        var context = banks.Context;
        RenderTrace.Write($"TargetlessDepth submit={submitId} registers={context.DepthTarget}");
        for (var slot = 0; slot < ContextRegisters.ColorTargetCount; slot++)
        {
            RenderTrace.Write(
                $"TargetlessColor submit={submitId} slot={slot} mask=0x{context.RenderTargetMaskForSlot((uint)slot):X} " +
                $"registers={context.ColorTargets[slot]}");
        }
    }

    // Whether a draw with these programs may store to guest memory, from the programs alone: any
    // written buffer (whatever its range), storage image or device address, or a mesh stage.
    public static bool ProgramsMayWriteMemory(GraphicsPrograms programs) =>
        programs.Available &&
        (programs.MeshInput is not null || StageMayWriteMemory(programs.VertexInput.Stage.Program) ||
            StageMayWriteMemory(programs.PixelInput.Stage.Program));

    private static bool StageMayWriteMemory(ShaderProgramInfo? program)
    {
        if (program is null)
        {
            return false;
        }

        if (program.WritesDeviceAddresses || WritesStorageImage(program))
        {
            return true;
        }

        foreach (var buffer in program.Buffers)
        {
            if (buffer.Written)
            {
                return true;
            }
        }

        return false;
    }

    internal static bool WritesStorageImage(ShaderProgramInfo program)
    {
        foreach (var image in program.Images)
        {
            if (image.Class == ImageResourceClass.Storage && image.Written)
            {
                return true;
            }
        }

        return false;
    }

    private static bool IsPremultipliedFillBlend(in BlendRegisters blend) =>
        blend.Enable && blend.ColorSourceFactor == BlendFactorOne && blend.ColorDestinationFactor == BlendFactorOneMinusSourceAlpha && blend.ColorFunction == BlendFunctionAdd;

    // A covering quad that only programs the DCC clear codes; its shaded output is discarded.
    private bool IsMetadataClearQuad(RegisterBanks banks, in DrawCall draw, ref DrawState state)
    {
        var programs = state.Programs;
        var context = banks.Context;
        if (draw.Count != DccFastClearVertexCount || (GuestPrimitiveType)banks.UserConfig.PrimitiveType != GuestPrimitiveType.TriangleStrip ||
            state.ColorCount == 0 || programs.PixelInput.Stage.Program is not { Images.Length: 0 } || programs.VertexInput.Stage.Program is not { Images.Length: 0 })
        {
            return false;
        }

        foreach (ref readonly var target in DrawState.BoundColors(ref state))
        {
            if (!IsPremultipliedFillBlend(in context.BlendControls[target.Slot]))
            {
                return false;
            }
        }

        var slot = state.Colors[0].Slot;
        var words = context.ColorTargets[slot];
        return words.DccEnabled && words.ClearWord0 == 0 && context.ColorClearWord1[slot] == 0 && CoversClipSpace(programs.PositionStream, draw.Count);
    }

    // The quad covers the clip volume when its positions span [-1, 1] on both axes.
    private bool CoversClipSpace(VertexPositionStream? stream, uint vertexCount)
    {
        if (stream is not { } positions)
        {
            return false;
        }

        var stride = positions.Stride == 0 ? 12u : positions.Stride;
        Span<byte> vertex = stackalloc byte[8];
        float minX = float.MaxValue, maxX = float.MinValue, minY = float.MaxValue, maxY = float.MinValue;
        for (var index = 0u; index < vertexCount; index++)
        {
            var address = positions.Address + positions.OffsetBytes + (ulong)index * stride;
            if (!_host.TryReadGuest(address, vertex))
            {
                return false;
            }

            var x = BinaryPrimitives.ReadSingleLittleEndian(vertex);
            var y = BinaryPrimitives.ReadSingleLittleEndian(vertex[4..]);
            if (!float.IsFinite(x) || !float.IsFinite(y))
            {
                return false;
            }

            minX = Math.Min(minX, x);
            maxX = Math.Max(maxX, x);
            minY = Math.Min(minY, y);
            maxY = Math.Max(maxY, y);
        }

        return minX <= -1f + ClipSpaceTolerance && maxX >= 1f - ClipSpaceTolerance && minY <= -1f + ClipSpaceTolerance && maxY >= 1f - ClipSpaceTolerance;
    }
}
