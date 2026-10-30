// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Agc;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Scheduling;

namespace SharpEmu.Libs.Gpu.Rendering;

// A tessellated draw on a host whose hull stage is a compute kernel (IRenderHost.SupportsTessellationEmulation).
//
// Grand Theft Auto V tessellates the near version of its palm trunks and other meshes (curved
// PN triangles with a displacement texture); the shadow passes draw them untessellated. The merged
// local/hull program runs as a compute kernel over the patches (one guest wave64 per threadgroup, as
// many whole patches as fit), which hands each patch to the domain stage through the guest's off-chip
// buffer and writes its tessellation factors to the guest's factor ring; the host turns those into its
// tessellator's factors, and the patches are drawn through that tessellator with the domain program as
// the post-tessellation vertex function and the pixel program as usual (Gen5TessellationLaunch).
//
// Only triangle and quad patches with as many output control points as input ones are taken (Ghost of
// Tsushima tessellates quads, drawn automatically rather than indexed), and only as many
// patches as the guest's off-chip buffer and factor ring hold at once; anything else is skipped as
// before. SHARPEMU_TESSELLATION=0 skips every tessellated draw.
public sealed partial class RenderExecutor
{
    // VGT_SHADER_STAGES_EN.LS_EN (bits 0-1, on) and HS_EN (bit 2).
    private const uint LocalStageEnable = 0x1;
    private const uint HullStageEnable = 0x4;

    // DI_PT_PATCH, and VGT_TF_PARAM.TYPE's triangle and quad domains and TOPOLOGY's two triangle windings.
    private const uint PatchPrimitiveType = 9;
    private const uint TriangleTessellationDomain = 1;
    private const uint QuadTessellationDomain = 2;
    private const uint TriangleClockwiseTopology = 2;
    private const uint TriangleCounterClockwiseTopology = 3;

    private static readonly bool EmulateTessellation =
        Environment.GetEnvironmentVariable("SHARPEMU_TESSELLATION") != "0";

    private static int _tessellationRefusalWarnings;
    private bool _tessellationAnnounced;

    // Whether this draw is a tessellated one the host runs, how many control points its patches have
    // and whether they are quads.
    private bool IsEmulatedTessellation(RegisterBanks banks, out uint controlPoints, out bool quad)
    {
        controlPoints = 0;
        quad = false;
        var stages = banks.Context.ShaderStages;
        if (!EmulateTessellation || !_host.SupportsTessellationEmulation ||
            (stages & LocalStageEnable) == 0 || (stages & HullStageEnable) == 0 ||
            banks.Shader.Vertex.LocalAddress == 0 || banks.UserConfig.PrimitiveType != PatchPrimitiveType)
        {
            return false;
        }

        // VGT_LS_HS_CONFIG: HS_NUM_INPUT_CP [13:8], HS_NUM_OUTPUT_CP [19:14].
        var configuration = banks.Context.ShaderInterface.LocalHullConfiguration;
        var inputPoints = (configuration >> 8) & 0x3F;
        var outputPoints = (configuration >> 14) & 0x3F;
        var parameter = banks.Context.ShaderInterface.TessellationFactorParameter;
        var domain = parameter & 0x3;
        var partition = (parameter >> 2) & 0x7;
        var topology = (parameter >> 5) & 0x7;
        if (inputPoints == 0 || inputPoints != outputPoints || inputPoints > 32 ||
            domain is not (TriangleTessellationDomain or QuadTessellationDomain) ||
            partition > 3 || (topology != TriangleClockwiseTopology && topology != TriangleCounterClockwiseTopology))
        {
            WarnTessellationRefused(
                $"control points {inputPoints} in, {outputPoints} out, domain {domain}, partition {partition}, topology {topology} " +
                $"(VGT_LS_HS_CONFIG=0x{configuration:X8} VGT_TF_PARAM=0x{parameter:X8})");
            return false;
        }

        controlPoints = inputPoints;
        quad = domain == QuadTessellationDomain;
        return true;
    }

    // The draw a tessellated draw's hull stage runs over: an indexed draw's index buffer, or an automatic
    // draw's vertices numbered from zero (Ghost of Tsushima draws its tessellated meshes automatically).
    private readonly record struct TessellatedDraw(uint VertexCount, uint InstanceCount, uint FirstInstance,
        int VertexOffset, ulong IndexAddress = 0, uint IndexTypeAndSize = 0, bool Indexed = false);

    private static void WarnTessellationRefused(string reason)
    {
        if (Interlocked.Increment(ref _tessellationRefusalWarnings) <= 4)
        {
            Console.Error.WriteLine($"[GPU][WARN] A tessellated draw is skipped: {reason}.");
        }
    }

    private static readonly bool ReverseTessellationBatches = Environment.GetEnvironmentVariable("SHARPEMU_TESS_BATCHES_REVERSED") == "1";

    private void DrawTessellated(ulong submitId, RegisterBanks banks, in TessellatedDraw arguments, uint controlPoints, bool quad)
    {
        var patchCount = arguments.VertexCount / controlPoints;
        if (patchCount == 0 || arguments.InstanceCount == 0)
        {
            return;
        }

        var patchesPerGroup = ShaderPipelineCache.TessellationPatchesPerGroup(controlPoints);
        // VGT_HS_OFFCHIP_PARAM.OFFCHIP_BUFFERING [8:0] is one less than the off-chip blocks the guest
        // allocated; the factor ring holds its size over the bytes of one patch.
        var offchipBlocks = (TessellationRings.OffchipParameter & 0x1FF) + 1;
        var factorPatches = TessellationRings.TessFactorRingSize / ShaderPipelineCache.TessellationFactorBytesPerPatch(quad);
        if (TessellationRings.TessFactorRing == 0)
        {
            WarnTessellationRefused("the guest has set no tessellation-factor ring");
            return;
        }

        // Every instance's patches go through the hull stage, as many whole instances at a time as the
        // guest's rings hold (Ghost of Tsushima draws its grass as one patch of thousands of instances).
        var ringPatches = Math.Min(offchipBlocks * patchesPerGroup, factorPatches);
        if (patchCount > ringPatches)
        {
            WarnTessellationRefused($"{patchCount} patches, the guest's rings hold {offchipBlocks} groups of {patchesPerGroup} and {factorPatches} patches");
            return;
        }

        var batchInstances = ringPatches / patchCount;
        var traceStarted = TraceTessellationTimes ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        // The batches share every register, so their programs are resolved once: the lookup materializes the
        // pixel program's resources, which for Ghost of Tsushima's particles (a 138-image table) cost ~3.5 ms a batch.
        TessellationPrograms? programs = null;
        var batches = (arguments.InstanceCount + batchInstances - 1) / batchInstances;
        for (var batch = 0u; batch < batches; batch++)
        {
            // SHARPEMU_TESS_BATCHES_REVERSED=1 (diagnostic): the last batch first, for telling a batch that draws
            // wrong because of its place in the draw from one whose instances are wrong.
            var done = (ReverseTessellationBatches ? batches - 1 - batch : batch) * batchInstances;
            var instances = Math.Min(batchInstances, arguments.InstanceCount - done);
            if (!DrawTessellatedInstances(submitId, banks, in arguments, controlPoints, quad, patchCount, instances,
                    arguments.FirstInstance + done, ref programs))
            {
                return;
            }
        }

        if (TraceTessellationTimes && _tracedTessellationTimes++ < 400)
        {
            Console.Error.WriteLine(
                $"[GPU][TESS-TIME] instances={arguments.InstanceCount} batches={(arguments.InstanceCount + batchInstances - 1) / batchInstances} " +
                $"total_ms={System.Diagnostics.Stopwatch.GetElapsedTime(traceStarted).TotalMilliseconds:F2} " +
                $"targets_ms={_tessTargetsTicks * 1000.0 / System.Diagnostics.Stopwatch.Frequency:F2} " +
                $"programs_ms={_tessProgramsTicks * 1000.0 / System.Diagnostics.Stopwatch.Frequency:F2} " +
                $"hull_ms={_tessHullTicks * 1000.0 / System.Diagnostics.Stopwatch.Frequency:F2} " +
                $"record_ms={_tessRecordTicks * 1000.0 / System.Diagnostics.Stopwatch.Frequency:F2}");
            _tessTargetsTicks = _tessProgramsTicks = _tessHullTicks = _tessRecordTicks = 0;
        }

        if (!_tessellationAnnounced)
        {
            _tessellationAnnounced = true;
            Console.Error.WriteLine(
                $"[GPU][INFO] Tessellated draws run as a hull kernel and a patch draw through the host's tessellator: " +
                $"first draw {patchCount} {(quad ? "quad" : "triangle")} patches of {controlPoints} control points, " +
                $"{arguments.InstanceCount} instances, {batchInstances} a batch, " +
                $"local=0x{banks.Shader.Vertex.LocalAddress:X16} domain=0x{banks.Shader.Vertex.ExportAddress:X16}.");
        }
    }

    // One batch of a tessellated draw's instances: the hull kernel over each instance's patches, then
    // the patches of those instances through the host's tessellator. False when the draw is given up.
    private bool DrawTessellatedInstances(ulong submitId, RegisterBanks banks, in TessellatedDraw arguments,
        uint controlPoints, bool quad, uint patchCount, uint instances, uint firstInstance, ref TessellationPrograms? resolved)
    {
        var draw = new DrawCall("DrawTessellated", RecordedOperation.DrawIndex, patchCount, instances, firstInstance);
        var phase = System.Diagnostics.Stopwatch.GetTimestamp();
        ref var state = ref BeginDrawState();
        if (!TryResolveDrawTargets(banks, in draw, ref state))
        {
            _host.ResetBindings();
            return false;
        }
        _tessTargetsTicks += NextTessellationPhase(ref phase);

        var inputs = GraphicsProgramInputs.Create();
        Span<ColorComponentMap> mapping = inputs.Mapping;
        foreach (ref readonly var color in BoundColors(ref state))
        {
            mapping[(int)color.Slot] = color.Resolution.ExportMapping;
        }

        var depthAspect = state.Depth.HasTarget &&
            (ViewFormatRules.FullAspects(state.Depth.Target.Target.Format) & GpuImageAspect.DepthBit) != 0;
        var stencilAspect = state.Depth.HasTarget &&
            (ViewFormatRules.DepthAspects(state.Depth.Target.Target.Format) & GpuImageAspect.StencilBit) != 0;
        var programs = resolved ??= _pipelines.GetTessellationPrograms(
            banks.Shader.Vertex, banks.Shader.Pixel, banks.Context.ShaderInterface, banks.Context, mapping,
            state.PixelActive, state.Depth.HasTarget, depthAspect, stencilAspect, controlPoints, quad);
        if (programs is not { Graphics.Available: true, Hull: { Available: true } hull })
        {
            _host.ResetBindings();
            return false;
        }

        state.Programs = programs.Graphics;
        _tessProgramsTicks += NextTessellationPhase(ref phase);
        if (!TryRunHull(hull, banks, in arguments, controlPoints, quad, patchCount, instances, firstInstance))
        {
            _host.ResetBindings();
            return false;
        }
        _tessHullTicks += NextTessellationPhase(ref phase);

        // Every patch through the host's tessellator, with the factors TryRunHull had it make.
        var emission = new DrawEmission(false, 0, 0, firstInstance, PatchControlPoints: controlPoints);
        var indexSource = new IndexSource(false, 0, null, 0, GpuIndexType.Uint16);
        _ = RecordDraw(submitId, banks, in draw, ref state, GpuPrimitiveTopology.TriangleList, in emission, in indexSource,
            primitiveRestart: false, setBindDebug: true, setAutoDebug: false);
        _host.ResetBindings();
        _tessRecordTicks += NextTessellationPhase(ref phase);
        return true;
    }

    // SHARPEMU_TRACE_TESSELLATION_TIMES=1: what each tessellated draw costs the worker, by phase.
    private static readonly bool TraceTessellationTimes =
        Environment.GetEnvironmentVariable("SHARPEMU_TRACE_TESSELLATION_TIMES") == "1";
    private int _tracedTessellationTimes;
    private long _tessTargetsTicks, _tessProgramsTicks, _tessHullTicks, _tessRecordTicks;

    private static long NextTessellationPhase(ref long phase)
    {
        var now = System.Diagnostics.Stopwatch.GetTimestamp();
        var elapsed = now - phase;
        phase = now;
        return elapsed;
    }

    // Runs the hull kernel over every patch of the batch's instances, instance after instance; false
    // while its pipeline still compiles (the draw is left out this time, before anything of it was recorded).
    private bool TryRunHull(ComputeProgram hull, RegisterBanks banks, in TessellatedDraw arguments,
        uint controlPoints, bool quad, uint patchCount, uint instances, uint firstInstance)
    {
        var hullPatches = patchCount * instances;
        var patchesPerGroup = ShaderPipelineCache.TessellationPatchesPerGroup(controlPoints);
        var groups = (hullPatches + patchesPerGroup - 1) / patchesPerGroup;
        var input = hull.Input;
        if (!_pipelines.TryCreateComputePipeline(input, hull.Program, out var pipeline))
        {
            return false;
        }

        // The local half's vertex of each control point, as the geometry engine would hand it over:
        // the index (or an automatic draw's vertex number) with the draw's vertex offset.
        var indices = arguments.Indexed
            ? ReadGuestIndices(arguments.IndexAddress, patchCount * controlPoints, (GuestIndexType)arguments.IndexTypeAndSize)
            : new uint[patchCount * controlPoints];
        for (var index = 0; index < indices.Length; index++)
        {
            indices[index] = unchecked((uint)((int)(arguments.Indexed ? indices[index] : (uint)index) + arguments.VertexOffset));
        }

        ReleaseFragmentWrites();
        _host.EndRendering("tessellation");
        using (_host.BeginPreparation())
        {
            var bindings = PrepareBindings(input.Stage);
            _host.BindResources(bindings);
            Span<IPreparedBindings> stages = [bindings];
            _host.CommitBindings(GpuBindPoint.Compute, in pipeline, stages);
            // The off-chip blocks are the ones the last tessellated draw's domain stage read.
            _host.ShaderWriteHazardBarrier();
            _host.BindPipeline(GpuBindPoint.Compute, in pipeline);
            _host.BindHullLaunch(indices, hullPatches, firstInstance, patchCount);
            var imageSnapshot = _host.BeginComputeImageSnapshot(bindings);
            _host.Dispatch(groups, 1, 1);
            _host.EndComputeImageSnapshot(imageSnapshot, bindings);
            _host.ShaderAccessBarrier();
            var shaderInterface = banks.Context.ShaderInterface;
            _host.PrepareTessellationFactors(
                TessellationRings.TessFactorRing, hullPatches, quad,
                BitConverter.UInt32BitsToSingle(shaderInterface.MinTessellationLevel),
                BitConverter.UInt32BitsToSingle(shaderInterface.MaxTessellationLevel));
        }

        _host.ResetBindings();
        return true;
    }
}
