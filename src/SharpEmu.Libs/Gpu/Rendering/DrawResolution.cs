// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.ShaderCompiler;

namespace SharpEmu.Libs.Gpu.Rendering;

internal sealed class DrawResolution
{
    public ulong TargetContextStamp;
    public ulong TargetProgramsStamp;
    public uint TargetVersion;
    public int ColorCount;
    public readonly ColorTargetResolution[] Colors = new ColorTargetResolution[RenderingState.ColorAttachmentCapacity];
    public readonly uint[] ColorSlots = new uint[RenderingState.ColorAttachmentCapacity];
    public DepthTargetState? Depth;
    public bool PixelActive;
    // The context fields the colours and depth above were resolved from (TargetInputs.Matches).
    public TargetInputs Inputs;

    public ulong ValidatedContextStamp;
    public ulong ValidatedUserConfigStamp;

    public DynamicStateKey DynamicKey;
    public DynamicDrawState Dynamic;
}

internal readonly record struct DynamicStateKey(
    ulong ContextStamp,
    ulong DynamicStamp,
    uint TargetVersion,
    uint ColorCount,
    uint BoundSlots,
    bool FirstColorFound,
    bool DepthFound);

internal sealed class KeptPipeline
{
    public bool Valid;
    public PipelineKey Key;
    public RenderingState Rendering;
    public PipelineHandle Handle;
}

internal readonly record struct PipelineKey(
    ulong ContextStamp,
    ulong DynamicStamp,
    uint TargetVersion,
    uint ColorCount,
    uint BoundSlots,
    bool DepthBound,
    VertexAttributeResource[] VertexAttributes,
    VertexInputBuffer[] VertexBuffers,
    uint[]? PixelInterpolators,
    ulong VertexProgram,
    ulong PixelProgram,
    int Topology,
    bool PrimitiveRestart,
    bool DisableBlending)
{
    // SHARPEMU_PIPELINE_MEMO_BY_CONTENT=0 keeps the previous draw's pipeline only for the very same vertex
    // tables and attachments.
    internal static readonly bool ByContent =
        Environment.GetEnvironmentVariable("SHARPEMU_PIPELINE_MEMO_BY_CONTENT") != "0";

    // The same pipeline as the other key's draw: the vertex tables are compared by what a pipeline takes
    // from them (each buffer's stride and step rate, each attribute's offset and buffer; ShaderPipelineCache.
    // FillVertexInputState), since every draw rebuilds them with its own buffer addresses.
    public bool SamePipelineAs(in PipelineKey other) =>
        ContextStamp == other.ContextStamp && DynamicStamp == other.DynamicStamp && TargetVersion == other.TargetVersion &&
        ColorCount == other.ColorCount && BoundSlots == other.BoundSlots && DepthBound == other.DepthBound &&
        VertexProgram == other.VertexProgram && PixelProgram == other.PixelProgram && Topology == other.Topology &&
        PrimitiveRestart == other.PrimitiveRestart && DisableBlending == other.DisableBlending &&
        SameBuffers(VertexBuffers, other.VertexBuffers) && SameAttributes(VertexAttributes, other.VertexAttributes) &&
        SameInterpolators(PixelInterpolators, other.PixelInterpolators);

    // What a pipeline takes from the attachments: their count, formats and sample count, not the views.
    public static bool SamePipelineRendering(in RenderingState first, in RenderingState second, uint colorCount)
    {
        if (first.ColorAttachmentCount != second.ColorAttachmentCount || first.Samples != second.Samples ||
            first.DepthFormat != second.DepthFormat || first.StencilFormat != second.StencilFormat)
        {
            return false;
        }

        for (var index = 0; index < colorCount; index++)
        {
            if (first.ColorAttachments[index].Format != second.ColorAttachments[index].Format)
            {
                return false;
            }
        }

        return true;
    }

    private static bool SameBuffers(VertexInputBuffer[] first, VertexInputBuffer[] second)
    {
        if (ReferenceEquals(first, second))
        {
            return true;
        }

        if (first.Length != second.Length)
        {
            return false;
        }

        for (var index = 0; index < first.Length; index++)
        {
            if (first[index].Stride != second[index].Stride || first[index].PerInstance != second[index].PerInstance)
            {
                return false;
            }
        }

        return true;
    }

    private static bool SameAttributes(VertexAttributeResource[] first, VertexAttributeResource[] second)
    {
        if (ReferenceEquals(first, second))
        {
            return true;
        }

        if (first.Length != second.Length)
        {
            return false;
        }

        for (var index = 0; index < first.Length; index++)
        {
            if (first[index].OffsetBytes != second[index].OffsetBytes || first[index].BufferIndex != second[index].BufferIndex)
            {
                return false;
            }
        }

        return true;
    }

    private static bool SameInterpolators(uint[]? first, uint[]? second) =>
        ReferenceEquals(first, second) ||
        (first is not null && second is not null && first.AsSpan().SequenceEqual(second));
}

// Everything ColorTargetResolver and DepthTargetResolver read from the context bank (the colour target words, the
// target mask, the depth words and the depth/stencil state fields beside them); the resolvers are pure functions of
// these, so a draw whose target registers were written again with the same values resolves the same targets. Floats
// are compared by their bits (a -0 clear value is not a +0 one). DrawResolutionInputsTests writes every context
// register and checks that one these fields do not cover leaves the resolved targets alone.
internal struct TargetInputs
{
    [InlineArray(ContextRegisters.ColorTargetCount)]
    internal struct ColorWords
    {
        private ColorTargetWords _first;
    }

    public ColorWords Colors;
    public uint RenderTargetMask;
    public DepthTargetWords Depth;
    public uint DepthClearBits;
    public uint DepthBoundsMinBits;
    public uint DepthBoundsMaxBits;
    public byte StencilClearValue;
    public StencilControlRegisters StencilControl;
    public StencilMaskRegisters StencilMask;
    public bool Captured;

    public void Capture(ContextRegisters context)
    {
        context.ColorTargets.AsSpan(0, ContextRegisters.ColorTargetCount).CopyTo(Colors);
        RenderTargetMask = context.RenderTargetMask;
        Depth = context.DepthTarget;
        DepthClearBits = BitConverter.SingleToUInt32Bits(context.DepthClearValue);
        DepthBoundsMinBits = BitConverter.SingleToUInt32Bits(context.DepthBoundsMin);
        DepthBoundsMaxBits = BitConverter.SingleToUInt32Bits(context.DepthBoundsMax);
        StencilClearValue = context.StencilClearValue;
        StencilControl = context.StencilControl;
        StencilMask = context.StencilMask;
        Captured = true;
    }

    public readonly bool Matches(ContextRegisters context)
    {
        if (!Captured || RenderTargetMask != context.RenderTargetMask || Depth != context.DepthTarget ||
            DepthClearBits != BitConverter.SingleToUInt32Bits(context.DepthClearValue) ||
            DepthBoundsMinBits != BitConverter.SingleToUInt32Bits(context.DepthBoundsMin) ||
            DepthBoundsMaxBits != BitConverter.SingleToUInt32Bits(context.DepthBoundsMax) ||
            StencilClearValue != context.StencilClearValue ||
            !SameBytes(in StencilControl, in context.StencilControl) || !SameBytes(in StencilMask, in context.StencilMask))
        {
            return false;
        }

        ReadOnlySpan<ColorTargetWords> colors = Colors;
        var current = context.ColorTargets;
        for (var slot = 0; slot < colors.Length; slot++)
        {
            if (colors[slot] != current[slot])
            {
                return false;
            }
        }

        return true;
    }

    // Both register structs are bytes only: no padding, so their bytes are their value.
    private static bool SameBytes<T>(in T left, in T right) where T : unmanaged =>
        MemoryMarshal.AsBytes(new ReadOnlySpan<T>(in left)).SequenceEqual(MemoryMarshal.AsBytes(new ReadOnlySpan<T>(in right)));
}
