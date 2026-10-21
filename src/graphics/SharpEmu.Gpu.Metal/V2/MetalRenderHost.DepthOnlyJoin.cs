// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;

namespace SharpEmu.Gpu.Metal.V2;

// A depth-only draw into the depth attachment of the open pass stays in that pass.
//
// Deferred decals and stencil masks alternate one draw into depth and stencil alone (the
// stencil mark) with one draw into a set of colour targets over the same depth (the decal
// that tests it). On the guest a target change between the two costs nothing; here each is a
// render pass of its own, and Silent Hill: The Short Message's cutscenes do it about four
// hundred times a frame, every pass paying a pass start, a store and a reload of the whole
// attachment set for one draw. A Metal pass cannot drop colour attachments for a draw, but a
// pipeline can leave them unwritten: the depth-only draw is drawn with a pipeline made for the
// pass's colour formats and a zero write mask on every one, so the pass ends only when the
// colour targets do change.
//
// Only what keeps the result the same joins: the same depth (and stencil) subresource, render
// area, layers and samples, no clear, no colour output from the pixel program, and nothing
// the draw samples is one of the pass's colour attachments - in its own pass it would read
// them as stored, here it would read memory the pass has not stored yet.
// SHARPEMU_METAL_V2_JOIN_DEPTH_ONLY=0 turns it off.
internal sealed partial class MetalRenderHost
{
    // The graphics stages the next draw committed, for the sampling check.
    private readonly MetalStageBindings?[] _joinStages = new MetalStageBindings?[2];

    // The draw BeginRendering joined to the open pass: its own pipeline and the one it is
    // drawn with there. BindPipeline takes them.
    private MetalPipelineEntry? _joinedPipeline;
    private MetalPipelineEntry? _joinedVariant;

    // Variant pipelines by depth-only pipeline and the colour formats of the pass.
    private readonly Dictionary<(ulong Pipeline, int Formats), ulong> _joinVariants = new();

    private long _depthOnlyJoins;
    private long _depthOnlyJoinsSampled;
    private long _depthOnlyJoinsDeferredCopies;

    internal (long Joined, long Sampled, long DeferredCopies) TakeDepthOnlyJoins() =>
        (Interlocked.Exchange(ref _depthOnlyJoins, 0),
         Interlocked.Exchange(ref _depthOnlyJoinsSampled, 0),
         Interlocked.Exchange(ref _depthOnlyJoinsDeferredCopies, 0));

    internal void NoteJoinStages(ReadOnlySpan<IPreparedBindings> stages)
    {
        _joinStages[0] = stages.Length > 0 ? stages[0] as MetalStageBindings : null;
        _joinStages[1] = stages.Length > 1 ? stages[1] as MetalStageBindings : null;
    }

    /// <summary>
    /// Whether a depth-only draw is drawn inside the open pass rather than in one of its own.
    /// On true the open pass stays as it is and the draw's pipeline is swapped at BindPipeline.
    /// </summary>
    private bool TryJoinDepthOnly(in RenderingState state, MetalPipelineEntry? upcoming)
    {
        if (!MetalV2Diagnostics.JoinDepthOnlyDraws || upcoming is null || !_renderingActive || _renderEncoder == 0)
        {
            return false;
        }

        ref readonly var open = ref _renderingState;
        if (state.ColorAttachmentCount != 0 || open.ColorAttachmentCount == 0 ||
            open.ColorAttachmentCount > MaxColorAttachments || _passAttachmentCount != open.ColorAttachmentCount ||
            state.Width != open.Width || state.Height != open.Height ||
            state.Layers != open.Layers || state.Samples != open.Samples)
        {
            return false;
        }

        ref readonly var depth = ref state.DepthStencilAttachment;
        ref readonly var openDepth = ref open.DepthStencilAttachment;
        if (depth.View.Handle == 0 || depth.View.Handle != openDepth.View.Handle ||
            depth.DepthClear || depth.StencilClear || depth.Format != openDepth.Format ||
            depth.HasDepth != openDepth.HasDepth || depth.HasStencil != openDepth.HasStencil ||
            _depthAttachmentLevel != _passDepthLevel || _depthAttachmentSlice != _passDepthSlice)
        {
            return false;
        }

        if (upcoming.Mesh is not null || upcoming.RectangleStates is not null ||
            upcoming.GraphicsDescription is not { } description || description.Rendering.ColorCount != 0 ||
            description.StaticParameters.Samples != open.Samples)
        {
            return false;
        }

        if (description.PixelStage is not null && description.PixelProgram.IsValid &&
            LookupModule(description.PixelProgram).Shader.ColorOutputMask != 0)
        {
            return false;
        }

        if (SamplesOpenColorAttachment())
        {
            _depthOnlyJoinsSampled++;
            return false;
        }

        // A copy deferred to the next encoder boundary lands there. Beginning a pass of its own
        // is that boundary; joining skips it, and the draw would read the bytes before the copy.
        if (_deferredLabelCopies.Count != 0 && BindsDeferredCopyDestination())
        {
            // The pass this begins is the boundary the copies land at, even when they wait for the commit.
            _landDeferredLabels = true;
            _depthOnlyJoinsDeferredCopies++;
            return false;
        }

        var variant = DepthOnlyJoinVariant(upcoming, description, in open);
        if (variant is null)
        {
            return false;
        }

        _joinedPipeline = upcoming;
        _joinedVariant = variant;
        _depthOnlyJoins++;
        PassProfileNoteJoin();
        return true;
    }

    // Whether the draw reads a buffer a deferred copy writes. A program that reaches guest
    // memory through device addresses names no ranges, so any waiting copy may be one it reads.
    private bool BindsDeferredCopyDestination()
    {
        foreach (var stage in _joinStages)
        {
            if (stage is null)
            {
                continue;
            }

            if (stage.Program.UsesDeviceAddresses)
            {
                return true;
            }

            for (var index = 0; index < stage.BufferCount; index++)
            {
                if (stage.Buffers[index].IsValid && DeferredLabelCopiesTouch(in stage.Buffers[index]))
                {
                    return true;
                }
            }
        }

        return false;
    }

    // Whether a texture the draw binds is one of the open pass's colour attachments, or a
    // view of one. A program that reaches images through bindless tables could reach any.
    private bool SamplesOpenColorAttachment()
    {
        foreach (var stage in _joinStages)
        {
            if (stage is null)
            {
                continue;
            }

            if (stage.Program.Bindings is { BindlessImageTableCount: > 0 })
            {
                return true;
            }

            for (var index = 0; index < stage.ImageCount; index++)
            {
                if (IsOpenColorAttachment(stage.Textures[index]))
                {
                    return true;
                }

                if (stage.MipTextures[index] is { } mips)
                {
                    foreach (var mip in mips)
                    {
                        if (IsOpenColorAttachment(mip))
                        {
                            return true;
                        }
                    }
                }
            }
        }

        return false;
    }

    private bool IsOpenColorAttachment(nint texture)
    {
        if (texture == 0)
        {
            return false;
        }

        var parent = MetalNative.Send(texture, SelParentTexture);
        for (var index = 0; index < _renderingState.ColorAttachmentCount; index++)
        {
            var attachment = (nint)_renderingState.ColorAttachments[index].View.Handle;
            var root = _passTextures[index];
            if (texture == attachment || texture == root ||
                (parent != 0 && (parent == attachment || parent == root)))
            {
                return true;
            }
        }

        return false;
    }

    // The depth-only pipeline remade for the open pass's colour formats, writing none of them;
    // null while it compiles (the draw then opens its own pass, as before).
    private MetalPipelineEntry? DepthOnlyJoinVariant(
        MetalPipelineEntry entry, GraphicsPipelineDescription description, in RenderingState open)
    {
        var formats = new HashCode();
        formats.Add(open.ColorAttachmentCount);
        for (var index = 0; index < open.ColorAttachmentCount; index++)
        {
            formats.Add(open.ColorAttachments[index].Format);
        }

        var key = (entry.Id, formats.ToHashCode());
        if (_joinVariants.TryGetValue(key, out var id) && _pipelineEntries.TryGetValue(id, out var known) &&
            MatchesColorFormats(known, in open))
        {
            return known;
        }

        var joined = JoinedDescription(description, in open);
        PipelineHandle handle;
        if (AsynchronousGraphics)
        {
            if (!Compiler.TryTake(
                    GraphicsCompileKey(joined),
                    () => BuildGraphicsPipelineStates(joined, onWorker: false),
                    out var compiled))
            {
                return null;
            }

            handle = RegisterGraphicsPipeline(joined, compiled!);
        }
        else
        {
            handle = CreateGraphicsPipeline(joined);
        }

        _joinVariants[key] = handle.Pipeline;
        return _pipelineEntries[handle.Pipeline];
    }

    private static bool MatchesColorFormats(MetalPipelineEntry entry, in RenderingState open)
    {
        if (entry.GraphicsDescription is not { } description ||
            description.Rendering.ColorCount != open.ColorAttachmentCount)
        {
            return false;
        }

        for (var index = 0; index < open.ColorAttachmentCount; index++)
        {
            if (description.Rendering.ColorFormats[index] != open.ColorAttachments[index].Format)
            {
                return false;
            }
        }

        return true;
    }

    internal static GraphicsPipelineDescription JoinedDescription(
        GraphicsPipelineDescription description, in RenderingState open)
    {
        var rendering = new PipelineRenderingState
        {
            DepthFormat = description.Rendering.DepthFormat,
            StencilFormat = description.Rendering.StencilFormat,
            ColorCount = open.ColorAttachmentCount,
        };
        for (var index = 0; index < open.ColorAttachmentCount; index++)
        {
            rendering.ColorFormats[index] = open.ColorAttachments[index].Format;
        }

        var parameters = PipelineStaticParameters.FromBytes(description.StaticParameters.Bytes);
        parameters.ColorCount = open.ColorAttachmentCount;
        for (var index = 0; index < PipelineStaticParameters.ColorAttachmentCount; index++)
        {
            parameters.SetColorMask(index, 0);
            parameters.SetBlendEnable(index, false);
        }

        return new GraphicsPipelineDescription
        {
            Rendering = rendering,
            VertexInput = description.VertexInput,
            VertexInfo = description.VertexInfo,
            VertexProgram = description.VertexProgram,
            VertexStage = description.VertexStage,
            PixelInfo = description.PixelInfo,
            PixelProgram = description.PixelProgram,
            PixelStage = description.PixelStage,
            StaticParameters = parameters,
        };
    }
}
