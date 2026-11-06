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
    // memory through device addresses names no bindings, so without its ordered ranges any
    // waiting copy may be one it reads; with them, only a copy over one of them is.
    private bool BindsDeferredCopyDestination()
    {
        foreach (var stage in _joinStages)
        {
            if (stage is null)
            {
                continue;
            }

            if (stage.Program.UsesDeviceAddresses && HeldCopiesMayReachDeviceRanges(stage))
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

    // ---- the device-address ranges a join checks the held copies against (SHARPEMU_METAL_V2_JOIN_DEVICE_RANGES) ----
    //
    // A join skips the encoder boundary held label copies would land at, so a draw that may read
    // one is refused and its pass is that boundary. A device-address program binds no buffers,
    // so every such draw was refused while any copy was held, and with labels held to the commit
    // Demon's Souls' G-buffer always holds some: each six-to-five step began a pass again. Yet
    // binding has already ordered every range such a program reaches against the copies held
    // then (BindDeviceAddressRanges: a planned handle's base and extent, an unplanned one's
    // whole space): a copy over one of them landed there, the open encoder ended first, which is
    // all a draw continuing the pass gets as well. What a join can still skip are the copies that
    // ordering did not cover: those held since, or all of them when copies land at encoder
    // boundaries (bind time then orders none). Only one of those over a range the program reaches
    // refuses the join. The TBDR line's join_device_ranges=[cleared,overlap,unordered] counts the
    // device-address stages a join checked: let through, refused for a copy over a range, refused
    // with no ordered ranges. SHARPEMU_METAL_V2_JOIN_DEVICE_RANGES=0 refuses them all again.
    internal bool JoinChecksDeviceRanges { get; set; } = MetalV2Diagnostics.JoinChecksDeviceRanges;

    private long _deviceRangeJoinsCleared;
    private long _deviceRangeJoinsOverlap;
    private long _deviceRangeJoinsUnordered;

    internal (long Cleared, long Overlap, long Unordered) TakeDeviceRangeJoinChecks() =>
        (Interlocked.Exchange(ref _deviceRangeJoinsCleared, 0),
         Interlocked.Exchange(ref _deviceRangeJoinsOverlap, 0),
         Interlocked.Exchange(ref _deviceRangeJoinsUnordered, 0));

    private string FormatDeviceRangeJoinChecks()
    {
        var (cleared, overlap, unordered) = TakeDeviceRangeJoinChecks();
        return $"cleared:{cleared},overlap:{overlap},unordered:{unordered}";
    }

    // A stage's binding starts: nothing is ordered for it yet. Called again where its ranges are
    // ordered, so the held copies counted are those every range was ordered against.
    internal void BeginDeviceRangeOrdering(MetalStageBindings stage)
    {
        stage.DeviceRangeCount = 0;
        stage.DeviceRangesOrdered = false;
        stage.DeviceRangesEpoch = _labelScanEpoch;
        stage.DeviceRangesHeld = _deferredLabelCopies.Count;
    }

    /// <summary>A range the stage's device-address program reaches, ordered after the held copies over it.</summary>
    internal void OrderDeviceRange(MetalStageBindings stage, ulong address, ulong size)
    {
        OrderAfterDeferredCopies(address, size);
        if (stage.DeviceRangeCount == stage.DeviceRanges.Length)
        {
            Array.Resize(ref stage.DeviceRanges, Math.Max(4, stage.DeviceRanges.Length * 2));
        }

        stage.DeviceRanges[stage.DeviceRangeCount++] = (address, size);
    }

    /// <summary>Every range of the stage is ordered: the copies held since BeginDeviceRangeOrdering were checked against each.</summary>
    internal void EndDeviceRangeOrdering(MetalStageBindings stage)
    {
        // Copies are only appended until they land, and landing them starts a new epoch: in the
        // same epoch the copies held at the start were there for every range's check. Held to
        // encoder boundaries instead, no copy was checked at all.
        if (!LabelsAtCommit || stage.DeviceRangesEpoch != _labelScanEpoch)
        {
            stage.DeviceRangesEpoch = _labelScanEpoch;
            stage.DeviceRangesHeld = 0;
        }

        stage.DeviceRangesOrdered = true;
    }

    // Whether a copy held now, and not ordered against the stage's ranges at bind time, may write
    // bytes its device-address program reads.
    private bool HeldCopiesMayReachDeviceRanges(MetalStageBindings stage)
    {
        if (!JoinChecksDeviceRanges)
        {
            return true;
        }

        if (!stage.DeviceRangesOrdered)
        {
            _deviceRangeJoinsUnordered++;
            return true;
        }

        var copies = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(_deferredLabelCopies);
        var from = stage.DeviceRangesEpoch == _labelScanEpoch ? Math.Min(stage.DeviceRangesHeld, copies.Length) : 0;
        for (var index = from; index < copies.Length; index++)
        {
            var low = copies[index].GuestAddress;
            var high = low + (ulong)copies[index].Bytes;
            for (var range = 0; range < stage.DeviceRangeCount; range++)
            {
                var (address, size) = stage.DeviceRanges[range];
                var end = size > ulong.MaxValue - address ? ulong.MaxValue : address + size;
                if (low < end && address < high)
                {
                    _deviceRangeJoinsOverlap++;
                    return true;
                }
            }
        }

        _deviceRangeJoinsCleared++;
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
