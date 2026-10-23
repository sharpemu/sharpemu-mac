// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;

namespace SharpEmu.Gpu.Metal.V2;

// A draw with a slot-preserving subset of the open pass's colour attachments can
// stay in tile memory. Its pipeline keeps the pass's formats, preserves the draw's
// masks/blends, and disables writes in the extra slots. Compacted/reordered subsets
// cannot join: the fragment outputs would address different attachments.
// Clears, render-area/subresource changes and reads of any open attachment still
// need a boundary, as do copies whose destination the draw binds.
internal sealed partial class MetalRenderHost
{
    // The graphics stages the next draw committed, for the sampling check.
    private readonly MetalStageBindings?[] _joinStages = new MetalStageBindings?[2];

    // The draw BeginRendering joined to the open pass: its own pipeline and the one it is
    // drawn with there. BindPipeline takes them.
    private MetalPipelineEntry? _joinedPipeline;
    private MetalPipelineEntry? _joinedVariant;

    // Variant pipelines by draw pipeline and the colour formats of the pass.
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
    /// Whether a subset draw is drawn inside the open pass rather than in one of its own.
    /// On true the open pass stays as it is and the draw's pipeline is swapped at BindPipeline.
    /// </summary>
    private bool TryJoinSubset(in RenderingState state, MetalPipelineEntry? upcoming)
    {
        if (!_renderingActive || _renderEncoder == 0) return false;
        var enabled = state.ColorAttachmentCount == 0
            ? MetalV2Diagnostics.JoinDepthOnlyDraws : MetalV2Diagnostics.JoinSubsetDraws;
        if (!enabled) return RefuseJoin(JoinRefusal.Disabled);
        if (upcoming is null) return RefuseJoin(JoinRefusal.Pipeline);

        ref readonly var open = ref _renderingState;
        if (state.ColorAttachmentCount >= open.ColorAttachmentCount || open.ColorAttachmentCount == 0 ||
            open.ColorAttachmentCount > MaxColorAttachments || _passAttachmentCount != open.ColorAttachmentCount)
            return RefuseJoin(JoinRefusal.ColorCount);
        if (state.Width != open.Width || state.Height != open.Height) return RefuseJoin(JoinRefusal.Size);
        if (state.Layers != open.Layers) return RefuseJoin(JoinRefusal.Layers);
        if (state.Samples != open.Samples) return RefuseJoin(JoinRefusal.Samples);

        ref readonly var depth = ref state.DepthStencilAttachment;
        ref readonly var openDepth = ref open.DepthStencilAttachment;
        if (depth.DepthClear || depth.StencilClear) return RefuseJoin(JoinRefusal.Clear);
        if (depth.View.Handle != openDepth.View.Handle ||
            depth.Format != openDepth.Format ||
            depth.HasDepth != openDepth.HasDepth || depth.HasStencil != openDepth.HasStencil ||
            (depth.View.Handle != 0 && (_depthAttachmentLevel != _passDepthLevel || _depthAttachmentSlice != _passDepthSlice)))
        {
            return RefuseJoin(JoinRefusal.Depth);
        }

        if (upcoming.RectangleStates is not null) return RefuseJoin(JoinRefusal.Rectangle);
        // The variant compiler preserves the mesh stage and its launch interface.
        if (upcoming.GraphicsDescription is not { } description || description.Rendering.ColorCount != state.ColorAttachmentCount ||
            description.StaticParameters.Samples != open.Samples)
        {
            return RefuseJoin(JoinRefusal.Pipeline);
        }

        if (description.PixelStage is not null && description.PixelProgram.IsValid &&
            (LookupModule(description.PixelProgram).Shader.ColorOutputMask >> (int)state.ColorAttachmentCount) != 0)
        {
            return RefuseJoin(JoinRefusal.Exports);
        }

        for (var index = 0; index < state.ColorAttachmentCount; index++)
        {
            ref readonly var next = ref state.ColorAttachments[index];
            ref readonly var previous = ref open.ColorAttachments[index];
            if (next.IsClear) return RefuseJoin(JoinRefusal.Clear);
            if (next.View.Handle != previous.View.Handle) return RefuseJoin(JoinRefusal.Texture);
            if (next.Format != previous.Format) return RefuseJoin(JoinRefusal.Format);
            if (index >= _attachmentCount || _attachmentLevels[index] != _passLevels[index] ||
                _attachmentSlices[index] != _passSlices[index] ||
                _attachmentDepthPlanes[index] != _passDepthPlanes[index])
                return RefuseJoin(JoinRefusal.Subresource);
            if (_attachmentWidths[index] != _passWidths[index] || _attachmentHeights[index] != _passHeights[index])
                return RefuseJoin(JoinRefusal.Size);
        }

        if (JoinBindingsNeedBoundary(state.ColorAttachmentCount == 0))
        {
            return false;
        }

        var variant = SubsetJoinVariant(upcoming, description, in open);
        if (variant is null)
        {
            return RefuseJoin(JoinRefusal.Compiling);
        }

        _joinedPipeline = upcoming;
        _joinedVariant = variant;
        if (state.ColorAttachmentCount == 0)
        {
            _depthOnlyJoins++;
            PassProfileNoteJoin();
        }
        else
        {
            NoteSubsetTransition(in state);
        }
        return true;
    }

    // Returning from a subset to the full set also skips a boundary the old path
    // would have made. Apply the same sampling and deferred-copy guards there.
    private bool JoinBindingsNeedBoundary(bool depthOnly)
    {
        if (SamplesOpenAttachment() is { } sampled)
        {
            if (depthOnly) _depthOnlyJoinsSampled++;
            RefuseJoin(sampled);
            return true;
        }

        // A copy deferred to the next encoder boundary lands there. Beginning a pass of its own
        // is that boundary; joining skips it, and the draw would read the bytes before the copy.
        if (_deferredLabelCopies.Count != 0 && BindsDeferredCopyDestination())
        {
            // The pass this begins is the boundary the copies land at, even when they wait for the commit.
            _landDeferredLabels = true;
            if (depthOnly) _depthOnlyJoinsDeferredCopies++;
            RefuseJoin(JoinRefusal.Copies);
            return true;
        }

        return false;
    }

    private uint _lastDrawColorCount;
    private long _subsetJoins;
    private long _subsetLoadAvoidedBytes;

    internal (long Joined, long Bytes) TakeSubsetJoins() =>
        (Interlocked.Exchange(ref _subsetJoins, 0), Interlocked.Exchange(ref _subsetLoadAvoidedBytes, 0));

    // Count avoided pass openings, rather than every draw using the subset variant.
    private void NoteSubsetTransition(in RenderingState state)
    {
        if (state.ColorAttachmentCount == _lastDrawColorCount)
        {
            return;
        }
        _lastDrawColorCount = state.ColorAttachmentCount;
        _subsetJoins++;
        long bytes = 0;
        for (var index = 0; index < state.ColorAttachmentCount; index++)
        {
            bytes += AttachmentBytes(in state, _passTexelBytes[index]);
        }
        if (state.DepthStencilAttachment.View.Handle != 0)
        {
            if (state.DepthStencilAttachment.HasDepth)
            {
                bytes += AttachmentBytes(in state, _passDepthTexelBytes);
            }
            if (state.DepthStencilAttachment.HasStencil)
            {
                bytes += AttachmentBytes(in state, 1);
            }
        }
        _subsetLoadAvoidedBytes += bytes;
        PassProfileNoteJoin(bytes);
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

    // Whether a texture the draw binds is one of the open pass's attachments, or a
    // view of one. A program that reaches images through bindless tables could reach any.
    private JoinRefusal? SamplesOpenAttachment()
    {
        foreach (var stage in _joinStages)
        {
            if (stage is null)
            {
                continue;
            }

            if (stage.Program.Bindings is { BindlessImageTableCount: > 0 })
            {
                return JoinRefusal.Bindless;
            }

            for (var index = 0; index < stage.ImageCount; index++)
            {
                if (IsOpenAttachment(stage.Textures[index]))
                {
                    return JoinRefusal.Sampled;
                }

                if (stage.MipTextures[index] is { } mips)
                {
                    foreach (var mip in mips)
                    {
                        if (IsOpenAttachment(mip))
                        {
                            return JoinRefusal.Sampled;
                        }
                    }
                }
            }
        }

        return null;
    }

    private bool IsOpenAttachment(nint texture)
    {
        if (texture == 0)
        {
            return false;
        }

        var parent = MetalNative.Send(texture, SelParentTexture);
        var depth = (nint)_renderingState.DepthStencilAttachment.View.Handle;
        if (depth != 0 && (texture == depth || texture == _passDepthTexture ||
            (parent != 0 && (parent == depth || parent == _passDepthTexture))))
        {
            return true;
        }
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

    // The draw's pipeline remade for the open pass's formats, writing only its own slots;
    // null while it compiles (the draw then opens its own pass, as before).
    private MetalPipelineEntry? SubsetJoinVariant(
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
        for (var index = (int)description.Rendering.ColorCount; index < PipelineStaticParameters.ColorAttachmentCount; index++)
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
            LodStatistics = description.LodStatistics,
            StaticParameters = parameters,
        };
    }
}
