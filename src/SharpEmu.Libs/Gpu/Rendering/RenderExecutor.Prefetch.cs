// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Images;

namespace SharpEmu.Libs.Gpu.Rendering;

// What a draw's program resolution reads besides the stage and context registers: the export
// mapping of every bound color slot, whether the pixel stage runs, whether a depth target is
// bound and whether that target has a depth aspect, and whether the draw is indexed. All
// follow from the register banks and the draw packet alone.
public struct GraphicsProgramInputs : IEquatable<GraphicsProgramInputs>
{
    [System.Runtime.CompilerServices.InlineArray(RenderingState.ColorAttachmentCapacity)]
    public struct ExportMappings
    {
        private ColorComponentMap _element0;
    }

    public ExportMappings Mapping;
    public bool PixelActive;
    public bool DepthBound;
    public bool DepthAspectBound;
    public bool StencilAspectBound;

    // VGT_PRIMITIVE_TYPE is a triangle list (see IShaderPipelineProvider.GetGraphicsPrograms).
    public bool TriangleList;
    public bool Indexed;

    // The vertex user data register that carries the draw's first instance, or -1 (the draw packet's).
    public int StartInstanceUserData;

    public static GraphicsProgramInputs Create()
    {
        var inputs = new GraphicsProgramInputs();
        ((Span<ColorComponentMap>)inputs.Mapping).Fill(ColorComponentMap.Identity);
        inputs.StartInstanceUserData = -1;
        return inputs;
    }

    public readonly bool Equals(GraphicsProgramInputs other) =>
        PixelActive == other.PixelActive && DepthBound == other.DepthBound && DepthAspectBound == other.DepthAspectBound &&
        StencilAspectBound == other.StencilAspectBound && TriangleList == other.TriangleList && Indexed == other.Indexed &&
        StartInstanceUserData == other.StartInstanceUserData &&
        ((ReadOnlySpan<ColorComponentMap>)Mapping).SequenceEqual(other.Mapping);

    public override readonly bool Equals(object? obj) => obj is GraphicsProgramInputs other && Equals(other);

    public override readonly int GetHashCode()
    {
        var hash = new HashCode();
        foreach (var map in (ReadOnlySpan<ColorComponentMap>)Mapping)
        {
            hash.Add(map.Packed);
        }

        hash.Add(PixelActive);
        hash.Add(DepthBound);
        hash.Add(DepthAspectBound);
        hash.Add(StencilAspectBound);
        hash.Add(TriangleList);
        hash.Add(Indexed);
        hash.Add(StartInstanceUserData);
        return hash.ToHashCode();
    }
}

// The host side of resolving programs on a thread other than the render thread. The resolving
// thread records what it assumed of the host's GPU-ownership state; the render thread checks,
// when it takes the result, that the assumptions still hold.
public interface IConcurrentResolveHost
{
    // False keeps a host that can answer reads from another thread resolving on its own thread,
    // so it can be turned on and off at run time.
    bool ResolvesConcurrently => true;

    // Opt-in: the host's speculative-read context may also run on the drawing thread.
    // The queue still permits only one resolution at a time, including that caller.
    bool AllowsCallerResolution => false;

    // On the resolving thread, around one resolution.
    void BeginConcurrentResolve();

    object EndConcurrentResolve();

    // On the render thread.
    bool IsStillCurrent(object assumptions);

    // On the render thread, once a taken resolution's assumptions were checked: the host may reuse
    // them for a later resolution.
    void ReleaseAssumptions(object assumptions)
    {
    }
}

// Program resolutions computed ahead of their draws. A draw takes one only when it was resolved
// from the same register banks with the same inputs; otherwise it resolves its programs itself.
public interface IProgramPrefetch
{
    bool TryTakeGraphics(RegisterBanks banks, in GraphicsProgramInputs inputs, out GraphicsPrograms programs);
}

public sealed partial class RenderExecutor
{
    public IProgramPrefetch? Prefetch { get; set; }

    // The inputs ResolveShaderPrograms derives from the targets TryResolveDrawTargets binds, from the
    // registers alone: the target requests and the pixel and depth activity, without the images.
    public static GraphicsProgramInputs ProgramInputsOf(RegisterBanks banks, IImageFormatSupport formats, Func<string, Exception> fatal, bool indexed = false,
        int startInstanceUserData = -1)
    {
        var inputs = GraphicsProgramInputs.Create();
        var context = banks.Context;
        Span<ColorComponentMap> mapping = inputs.Mapping;
        for (var slot = 0u; slot < ContextRegisters.ColorTargetCount; slot++)
        {
            if (slot != 0 && (context.RenderTargetMaskForSlot(slot) == 0 || context.ColorTargets[slot].BaseAddress == 0))
            {
                continue;
            }

            if (ColorTargetResolver.Resolve(context, slot, DrawLayerOffset, ignoreTargetMask: false, out var resolvedSlot) is { } resolution)
            {
                mapping[(int)resolvedSlot] = resolution.ExportMapping;
            }
        }

        var depth = DepthTargetResolver.Resolve(context, formats, fatal);
        inputs.DepthBound = depth is not null;
        inputs.DepthAspectBound = depth is { } target &&
            (ViewFormatRules.FullAspects(target.Target.Format) & GpuImageAspect.DepthBit) != 0;
        // The aspect the attachment binds (RenderExecutor.AcquireAttachments).
        inputs.StencilAspectBound = depth is { } stencilTarget &&
            (ViewFormatRules.DepthAspects(stencilTarget.Target.Format) & GpuImageAspect.StencilBit) != 0;
        inputs.PixelActive = HasActivePixelShader(banks);
        inputs.TriangleList = AssemblesTriangleList(banks);
        inputs.Indexed = indexed;
        inputs.StartInstanceUserData = startInstanceUserData;
        return inputs;
    }

    // The programs of a draw with these banks and inputs, as ResolveShaderPrograms resolves them.
    public GraphicsPrograms ResolveGraphicsPrograms(RegisterBanks banks, in GraphicsProgramInputs inputs)
    {
        var context = banks.Context;
        if (UsesMeshExecution(banks))
        {
            var userConfig = banks.UserConfig;
            // Indexed strips are expanded to lists. Lists already have the right topology;
            // the pipeline resolver only reads these banks, so they need no private copy.
            if (inputs.Indexed && userConfig.PrimitiveType == (uint)GuestPrimitiveType.TriangleStrip)
            {
                userConfig = userConfig.Copy();
                userConfig.PrimitiveType = (uint)GuestPrimitiveType.TriangleList;
            }

            return _pipelines.GetMeshGraphicsPrograms(
                banks, userConfig, inputs.Mapping, inputs.PixelActive, inputs.DepthBound);
        }

        return _pipelines.GetGraphicsPrograms(
            banks,
            inputs.Mapping,
            inputs.PixelActive,
            inputs.DepthBound,
            inputs.DepthAspectBound,
            inputs.StencilAspectBound,
            inputs.TriangleList,
            inputs.StartInstanceUserData);
    }

    private static bool AssemblesTriangleList(RegisterBanks banks) =>
        (GuestPrimitiveType)banks.UserConfig.PrimitiveType == GuestPrimitiveType.TriangleList;
}
