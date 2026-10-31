// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text.RegularExpressions;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;

namespace SharpEmu.Gpu.Metal.V2;

// A draw into some of the open pass's colour attachments stays in that pass.
//
// Grand Theft Auto V's G-buffer alternates draws into five targets with draws into three of the
// same five, over the same depth, some forty-five times a frame. On the guest a target change
// costs nothing; here each one ended a pass and began another, storing and reloading every
// attachment - about 100 GB/s of attachment traffic each way in its street - and made an encoder
// on the GPU worker. As a depth-only draw is (DepthOnlyJoin.cs), such a draw is drawn in the open
// pass with a pipeline made for the pass's colour formats: its own attachments keep their write
// masks and blending at the pass's indices of them, the others are not written, and its pixel
// program's [[color(n)]] outputs move to those indices (a library of its own, made once per
// program and remap; none when the draw's attachments are the pass's first ones in order).
//
// The depth-only join's conditions hold: the same depth subresource, render area, layers and
// samples, no clear on any attachment of the draw, nothing it samples is one of the pass's
// attachments, and no copy waits at the encoder boundary for something it binds. Besides, every
// colour attachment of the draw is one of the pass's at the same subresource and format, its
// pixel program blends with no second source and writes no sample mask, and alpha to coverage
// keeps its first output where it was.
// SHARPEMU_METAL_V2_JOIN_SUBSETS=0 turns it off.
internal sealed partial class MetalRenderHost
{
    // Variant pipelines by pipeline, colour formats of the pass and placement of the draw's attachments.
    private readonly Dictionary<(ulong Pipeline, int Formats, uint Placement), ulong> _subsetJoinVariants = new();

    // Whether a pixel module's colour outputs can move, by module (CanRemapColorOutputs).
    private readonly Dictionary<ulong, bool> _pixelOutputsMovable = new();

    private long _subsetJoins;
    private long _subsetJoinsSampled;

    // SHARPEMU_METAL_V2_JOIN_REASONS=1: why draws did not join, in the TBDR line (subset_refused).
    private static readonly bool CountJoinRefusals = Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_JOIN_REASONS") == "1";

    // SHARPEMU_METAL_V2_JOIN_EXCLUDE=<pixel hash>[,...]: draws with these pixel programs never join,
    // for finding the draw a join changes.
    private static readonly HashSet<ulong> JoinExcludedPixelPrograms = ParseDumpBufferHashesFrom("SHARPEMU_METAL_V2_JOIN_EXCLUDE");
    private readonly Dictionary<string, long> _subsetRefusals = new();

    private bool RefuseSubsetJoin(string reason)
    {
        if (CountJoinRefusals)
        {
            ref var count = ref System.Runtime.InteropServices.CollectionsMarshal.GetValueRefOrAddDefault(_subsetRefusals, reason, out _);
            count++;
        }

        return false;
    }

    internal string TakeSubsetRefusals()
    {
        if (_subsetRefusals.Count == 0)
        {
            return string.Empty;
        }

        var text = string.Join(",", _subsetRefusals.OrderByDescending(pair => pair.Value).Select(pair => $"{pair.Key}:{pair.Value}"));
        _subsetRefusals.Clear();
        return $" subset_refused=({text})";
    }

    internal (long Joined, long Sampled) TakeSubsetJoins() =>
        (Interlocked.Exchange(ref _subsetJoins, 0), Interlocked.Exchange(ref _subsetJoinsSampled, 0));

    /// <summary>
    /// Whether a draw into a subset of the open pass's colour attachments is drawn inside that
    /// pass. On true the open pass stays as it is and the draw's pipeline is swapped at BindPipeline.
    /// </summary>
    private bool TryJoinColorSubset(in RenderingState state, MetalPipelineEntry? upcoming)
    {
        if (!MetalV2Diagnostics.JoinColorSubsets || upcoming is null || !_renderingActive || _renderEncoder == 0)
        {
            return RefuseSubsetJoin("inactive");
        }

        ref readonly var open = ref _renderingState;
        var count = (int)state.ColorAttachmentCount;
        if (count == 0 || count > (int)open.ColorAttachmentCount || open.ColorAttachmentCount > MaxColorAttachments ||
            _passAttachmentCount != open.ColorAttachmentCount || _attachmentCount < count ||
            state.Width != open.Width || state.Height != open.Height ||
            state.Layers != open.Layers || state.Samples != open.Samples)
        {
            return RefuseSubsetJoin("shape");
        }

        ref readonly var depth = ref state.DepthStencilAttachment;
        ref readonly var openDepth = ref open.DepthStencilAttachment;
        if (depth.DepthClear || depth.StencilClear || depth.View.Handle != openDepth.View.Handle ||
            depth.Format != openDepth.Format || depth.HasDepth != openDepth.HasDepth || depth.HasStencil != openDepth.HasStencil ||
            (depth.View.Handle != 0 && (_depthAttachmentLevel != _passDepthLevel || _depthAttachmentSlice != _passDepthSlice)))
        {
            return RefuseSubsetJoin("depth");
        }

        // Four bits per attachment of the draw: one more than the pass's index of it.
        var placement = 0u;
        var taken = 0u;
        var inOrder = true;
        for (var index = 0; index < count; index++)
        {
            ref readonly var next = ref state.ColorAttachments[index];
            if (next.IsClear || next.View.Handle == 0)
            {
                return RefuseSubsetJoin("clear");
            }

            var found = -1;
            for (var slot = 0; slot < (int)open.ColorAttachmentCount; slot++)
            {
                ref readonly var candidate = ref open.ColorAttachments[slot];
                if ((taken & (1u << slot)) == 0 && candidate.View.Handle == next.View.Handle && candidate.Format == next.Format &&
                    _attachmentLevels[index] == _passLevels[slot] && _attachmentSlices[index] == _passSlices[slot] &&
                    _attachmentDepthPlanes[index] == _passDepthPlanes[slot])
                {
                    found = slot;
                    break;
                }
            }

            if (found < 0)
            {
                return RefuseSubsetJoin("unmatched");
            }

            taken |= 1u << found;
            inOrder &= found == index;
            placement |= (uint)(found + 1) << (index * 4);
        }

        if (inOrder && count == (int)open.ColorAttachmentCount)
        {
            // The pass's own attachments: CanContinuePass refused the draw for another reason.
            return RefuseSubsetJoin("same-set");
        }

        if (upcoming.Mesh is not null || upcoming.RectangleStates is not null ||
            upcoming.GraphicsDescription is not { } description || description.Rendering.ColorCount != count ||
            description.StaticParameters.Samples != open.Samples || description.ColorRemap != 0)
        {
            return RefuseSubsetJoin("pipeline");
        }

        var remapsOutputs = !inOrder && description.PixelStage is not null && description.PixelProgram.IsValid;
        if (JoinExcludedPixelPrograms.Count != 0 && description.PixelStage is { } pixelStage &&
            JoinExcludedPixelPrograms.Contains(pixelStage.Hash))
        {
            return RefuseSubsetJoin("excluded");
        }

        if (MovesCoverageSource(description.StaticParameters, placement))
        {
            return RefuseSubsetJoin("alpha-to-coverage");
        }

        // A pixel program that writes its own sample mask opens its own pass. Grand Theft Auto V's
        // hair normals (one draw into the normal, flag and velocity targets, coverage from the hair
        // texture) drawn joined left a shell of the last frame's head lit around Michael's hair in
        // its opening cutscene; drawn in a pass of its own, as before the join, they do not.
        if (description.PixelInfo is { SampleMaskExportEnable: true })
        {
            return RefuseSubsetJoin("coverage");
        }

        if (SamplesOpenColorAttachment())
        {
            _subsetJoinsSampled++;
            return RefuseSubsetJoin("sampled");
        }

        // A copy deferred to the next encoder boundary lands there (see TryJoinDepthOnly).
        if (_deferredLabelCopies.Count != 0 && BindsDeferredCopyDestination())
        {
            _landDeferredLabels = true;
            return RefuseSubsetJoin("copies");
        }

        if (remapsOutputs && !PixelOutputsMovable(description.PixelProgram))
        {
            return RefuseSubsetJoin("dual-source");
        }

        var variant = SubsetJoinVariant(upcoming, description, in open, remapsOutputs ? placement : 0, placement);
        if (variant is null)
        {
            return RefuseSubsetJoin("compiling");
        }

        _joinedPipeline = upcoming;
        _joinedVariant = variant;
        _subsetJoins++;
        PassProfileNoteJoin();
        return true;
    }

    // Metal takes alpha to coverage from whatever [[color(0)]] returns: with the draw's first output
    // moved to another attachment of the pass, its coverage would come from an output it does not have.
    internal static bool MovesCoverageSource(PipelineStaticParameters parameters, uint placement) =>
        parameters.AlphaToCoverageEnable && (placement & 0xF) != 1;

    // The draw's pipeline remade for the open pass's colour formats; null while it compiles (the
    // draw then opens its own pass, as before).
    private MetalPipelineEntry? SubsetJoinVariant(
        MetalPipelineEntry entry, GraphicsPipelineDescription description, in RenderingState open, uint remap, uint placement)
    {
        var formats = new HashCode();
        formats.Add(open.ColorAttachmentCount);
        for (var index = 0; index < open.ColorAttachmentCount; index++)
        {
            formats.Add(open.ColorAttachments[index].Format);
        }

        var key = (entry.Id, formats.ToHashCode(), placement);
        if (_subsetJoinVariants.TryGetValue(key, out var id) && _pipelineEntries.TryGetValue(id, out var known) &&
            MatchesColorFormats(known, in open))
        {
            return known;
        }

        var joined = SubsetJoinedDescription(description, in open, remap, placement);
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

        _subsetJoinVariants[key] = handle.Pipeline;
        return _pipelineEntries[handle.Pipeline];
    }

    internal static GraphicsPipelineDescription SubsetJoinedDescription(
        GraphicsPipelineDescription description, in RenderingState open, uint remap, uint placement)
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

        var own = description.StaticParameters;
        var parameters = PipelineStaticParameters.FromBytes(own.Bytes);
        parameters.ColorCount = open.ColorAttachmentCount;
        for (var slot = 0; slot < PipelineStaticParameters.ColorAttachmentCount; slot++)
        {
            parameters.SetColorMask(slot, 0);
            parameters.SetBlendEnable(slot, false);
        }

        for (var index = 0; index < description.Rendering.ColorCount; index++)
        {
            var slot = (int)((placement >> (index * 4)) & 0xF) - 1;
            parameters.SetColorMask(slot, own.GetColorMask(index));
            parameters.SetBlendEnable(slot, own.GetBlendEnable(index));
            parameters.SetBlendBypass(slot, own.GetBlendBypass(index));
            parameters.SetSeparateAlphaBlend(slot, own.GetSeparateAlphaBlend(index));
            parameters.SetColorSourceBlend(slot, own.GetColorSourceBlend(index));
            parameters.SetColorBlendFunction(slot, own.GetColorBlendFunction(index));
            parameters.SetColorDestinationBlend(slot, own.GetColorDestinationBlend(index));
            parameters.SetAlphaSourceBlend(slot, own.GetAlphaSourceBlend(index));
            parameters.SetAlphaBlendFunction(slot, own.GetAlphaBlendFunction(index));
            parameters.SetAlphaDestinationBlend(slot, own.GetAlphaDestinationBlend(index));
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
            LodStatistics = description.LodStatistics,
            ColorRemap = remap,
        };
    }

    private static readonly Regex ColorAttribute = new(@"\[\[color\((\d+)\)", RegexOptions.CultureInvariant);

    // A second blend source is declared [[color(0), index(1)]]: such outputs stay where they are.
    private bool PixelOutputsMovable(in ShaderProgram program)
    {
        if (!_pixelOutputsMovable.TryGetValue(program.Module, out var movable))
        {
            movable = !LookupModule(program).Compiled.AcquireShader().Source.Contains(", index(1)", StringComparison.Ordinal);
            _pixelOutputsMovable[program.Module] = movable;
        }

        return movable;
    }

    // The pixel function with every [[color(n)]] moved where the remap sends output n, with its LOD
    // statistics counters in or out as PixelFunctionOf would choose them.
    private (nint Function, UInt128 ArchiveKey) RequireRemappedPixelFunction(MetalShaderModule module, uint remap, bool lodStatistics)
    {
        lock (module)
        {
            module.RemappedFunctions ??= new();
            var counters = lodStatistics && module.Shader.UsesMipStatisticsConstant;
            if (module.RemappedFunctions.TryGetValue((remap, counters), out var known))
            {
                return (known.Function, known.ArchiveKey);
            }

            var shader = module.Compiled.AcquireShader();
            var source = ColorAttribute.Replace(shader.Source, match =>
            {
                var output = int.Parse(match.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture);
                var slot = output < 8 ? (int)((remap >> (output * 4)) & 0xF) - 1 : -1;
                return slot < 0 ? match.Value : $"[[color({slot})";
            });
            var library = CompileLibrary(source, module.Hash, shader.UsesPerVertexInputs, prewarmable: true, out var archiveKey, out _);
            nint error = 0;
            var function = shader.UsesMipStatisticsConstant
                ? NewMipStatisticsFunction(library, shader.EntryPoint, enabled: counters, out error)
                : MetalNative.Send(library, MetalNative.Selector("newFunctionWithName:"), MetalNative.NsString(shader.EntryPoint));
            if (function == 0)
            {
                ReleaseLibrary(library);
                throw Fatal(
                    $"The remapped pixel library has no entry point: entry={shader.EntryPoint} hash=0x{module.Hash:X16} remap=0x{remap:X8}" +
                    (error != 0 ? $" {MetalNative.DescribeError(error)}." : "."));
            }

            if (shader.UsesMipStatisticsConstant && !counters)
            {
                // As in PixelFunctionOf: samples drawn through this pipeline never reach the bank.
                Volatile.Write(ref _lodStatsCompiledOut, true);
            }

            module.RemappedFunctions[(remap, counters)] = (library, function, archiveKey);
            return (function, archiveKey);
        }
    }

    private void ReleaseRemappedFunctions(MetalShaderModule module)
    {
        if (module.RemappedFunctions is not { } remapped)
        {
            return;
        }

        foreach (var (library, function, _) in remapped.Values)
        {
            MetalNative.SendVoid(function, SelRelease);
            ReleaseLibrary(library);
        }

        module.RemappedFunctions = null;
    }
}
