// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Pipelines;

namespace SharpEmu.Libs.Gpu.Rendering;

// The hull stage of a tessellated draw on a host that runs it as a compute kernel
// (RenderExecutor.TessellationEmulation.cs).
internal sealed partial class RenderRecorder
{
    // Runs the hull kernel over every patch of the batch's instances, instance after instance; false
    // while its pipeline still compiles (the draw is left out this time, before anything of it was recorded).
    // indices: the local half's vertex of each control point, with the draw's vertex offset.
    internal bool RunTessellationHull(ComputeProgram hull, ShaderInterfaceRegisters shaderInterface, uint[] indices,
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

        _inPlace.Valid = false;
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
            _host.PrepareTessellationFactors(
                Agc.TessellationRings.TessFactorRing, hullPatches, quad,
                BitConverter.UInt32BitsToSingle(shaderInterface.MinTessellationLevel),
                BitConverter.UInt32BitsToSingle(shaderInterface.MaxTessellationLevel));
        }

        _host.ResetBindings();
        return true;
    }
}
