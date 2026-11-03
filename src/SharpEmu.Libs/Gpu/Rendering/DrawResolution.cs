// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

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
