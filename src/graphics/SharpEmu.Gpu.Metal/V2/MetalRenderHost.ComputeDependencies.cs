// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Rendering;
using System.Globalization;
using SharpEmu.ShaderCompiler.Metal;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Gpu.Metal.V2;

internal sealed partial class MetalRenderHost
{
    internal bool ComputeChainEnabled { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_COMPUTE_CHAIN") != "0";

    // 0 restores the serial encoder and the executor's original pre/post barriers.
    internal bool ConcurrentComputeEnabled { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_CONCURRENT_COMPUTE") != "0";

    public bool TracksComputeDependencies => ConcurrentComputeEnabled;

    bool IMetalSubmissionContext.SharedComputeIsConcurrent => ConcurrentComputeEnabled;

    private readonly MetalComputeDependencies _computeDependencies = new();
    private readonly MetalComputeFootprint _computeFootprint = new();
    private nint _computeBoundaryEvent;
    private ulong _computeBoundaryValue;
    private ulong _computeBoundarySerial;
    private ulong _lastConcurrentComputeSerial;
    internal long ComputeDispatches { get; private set; }
    internal long ComputeScopeBarriers { get; private set; }
    internal long ComputeEncoders { get; private set; }
    internal long ComputeDependencyBarriers { get; private set; }
    internal long ComputeGuestBarriers { get; private set; }
    internal long ComputeIndependentDispatches { get; private set; }
    internal long ComputeResourceBarriers { get; private set; }
    internal long ComputeIndirectDispatches { get; private set; }
    private readonly Dictionary<string, long> _computeDependencyReasons = new(StringComparer.Ordinal);
    private readonly Dictionary<string, long> _computeEncoderEnds = new(StringComparer.Ordinal);
    private sealed class ComputeChainShaderCounts
    {
        internal long Dispatches, Unknown, Dependencies;
    }
    private readonly Dictionary<ulong, ComputeChainShaderCounts> _computeChainShaders = new();
    private static readonly nint SelComputeResourceBarrier = MetalNative.Selector("memoryBarrierWithResources:count:");

    private static readonly nint SelComputeEncoderDispatchType =
        MetalNative.Selector("computeCommandEncoderWithDispatchType:");
    private static readonly nint SelSetComputeDispatchType = MetalNative.Selector("setDispatchType:");

    // CP copies and staged stores keep the same range-specific materialization as
    // serial recording. Draining all held writes before every dispatch splits the
    // encoder even for unrelated labels; scope barriers only order actual GPU work.
    // Captured only for compute, while BindResources already has the resolved ranges.
    // Render preparation and CommitBindings do no dependency work. Keep the storage
    // across dispatches, including those that reuse the committed bindings.
    private void BeginComputeFootprint(MetalStageBindings stage)
    {
        var footprint = _computeFootprint;
        footprint.Clear();
        footprint.Accesses.EnsureCapacity(stage.BufferCount * 4 + stage.ImageCount * 9 + 8);
        var program = stage.Program;
        var info = program.Resources!.Info;
        var layout = program.Bindings!;
        footprint.UnknownRead = program.UsesDeviceAddresses || info.UsesDeviceAddresses;
        footprint.UnknownWrite = program.WritesDeviceAddresses || info.WritesDeviceAddresses;
        if (ComputeChainEnabled && footprint.UnknownRead)
            CaptureComputeDeviceRanges(stage);
        if (layout.BindlessImageTableCount != 0 || stage.StageResources.Resources.BindlessImages.Length != 0)
            footprint.UnknownRead = footprint.UnknownWrite = true;
        for (var index = 0; index < layout.Descriptors.Count; index++)
        {
            var binding = layout.Descriptors[index];
            if (binding.Kind is DescriptorBindingKind.GlobalDataShare or DescriptorBindingKind.MipStatistics ||
                (binding.Kind == DescriptorBindingKind.FaultBuffer &&
                 (!ComputeChainEnabled || _singleCopy is not null)))
                footprint.UnknownRead = footprint.UnknownWrite = true;
            // fault_word_count is zero. Without single-copy traps this pointer
            // only receives host-private reach feedback: atomic OR/add with unused
            // return values. No shader consumes it and the CPU uses atomic exchange;
            // it adds no guest ordering edge. Trap/control feedback stays unknown.
        }
    }

    private void NoteComputeBuffer(ulong address, uint bytes, in MetalGuestRange range, bool written)
    {
        if (!range.IsValid) return;
        NoteComputeGuestRange(address, bytes, written, range.Buffer);
        _computeFootprint.Accesses.Add(new(MetalComputeAddressSpace.Buffer,
            (ulong)range.Buffer, range.Offset, bytes, written, range.Buffer));
        _computeFootprint.Accesses.Add(new(MetalComputeAddressSpace.Device,
            0, range.GpuAddress, bytes, written, range.Buffer));
    }

    private void NoteComputeImage(MetalStageBindings stage, int index)
    {
        var imageInfo = stage.Program.Resources!.Info.Images[index];
        var written = imageInfo.Written || imageInfo.Atomic;
        if (!_imageCache.TryGetImage(stage.Images[index], out var image) ||
            index == stage.MovieLuma || index == stage.MovieChroma)
        {
            _computeFootprint.UnknownRead = true;
            _computeFootprint.UnknownWrite |= written;
            return;
        }

        // Backing extents cover format and subresource aliases, not just the bound view.
        var description = image.Description;
        NoteComputeGuestRange(description.Data.Address, description.Data.Size, written, image.Texture);
        NoteComputeGuestRange(description.Stencil.Address, description.Stencil.Size, written, image.Texture);
        _computeFootprint.Accesses.Add(new(MetalComputeAddressSpace.Texture, (ulong)image.Texture, 0, 1, written, image.Texture));
    }

    private void CaptureComputeDeviceRanges(MetalStageBindings stage)
    {
        var ranges = stage.StageResources.Resources.DeviceAddressRanges;
        // Planned means the base can be evaluated. An unbounded handle is merely
        // capped to 16 MiB by the residency planner: it is NOT an independence proof.
        if (!stage.Program.Resources!.Info.DeviceAddressRangesComplete || ranges.Length == 0) return;
        for (var index = 0; index < ranges.Length; index++)
            if (!ranges[index].Planned || !ranges[index].Bounded) return;
        _computeFootprint.UnknownRead = _computeFootprint.UnknownWrite = false;
        foreach (var range in ranges)
            NoteComputeGuestRange(range.Base, range.Size, range.Written);
    }

    private void OrderComputeDispatch(ulong indirectArguments = 0, MetalGuestRange indirectRange = default)
    {
        ComputeDispatches++;
        if (!ConcurrentComputeEnabled) return;
        var footprint = _computeFootprint;
        var accessCount = footprint.Accesses.Count;
        var unknownRead = footprint.UnknownRead;
        var unknownWrite = footprint.UnknownWrite;
        var stage = _stageBindings[StageCompute];
        if ((_preparedStageMask & (1 << StageCompute)) == 0 || !stage.Committed)
        {
            // Native/direct callers without a committed resource snapshot have no proof.
            footprint.UnknownRead = footprint.UnknownWrite = true;
        }
        else
        {
            var program = stage.Program;
            // Shader-internal diagnostics and pair locks are real device writes too.
            // Reflection records whether emission actually uses these trailing pointers.
            if (_pendingComputePipeline is not { } pipeline)
            {
                footprint.UnknownRead = footprint.UnknownWrite = true;
            }
            else
            {
                if (pipeline.WritesStepReport && _stepReportSlotByHash.TryGetValue(program.Hash, out var slot))
                    footprint.Accesses.Add(new(MetalComputeAddressSpace.Device, 0,
                        _stepReportGpuAddress + (ulong)(slot * StepReportStride * sizeof(uint)),
                        StepReportStride * sizeof(uint), true));
                if (pipeline.UsesPairLocks)
                    footprint.Accesses.Add(new(MetalComputeAddressSpace.Device, 0,
                        _guestMemory.PairLockTableGpuAddress, Gen5MslArgumentLayout.PairLockTableWords * sizeof(uint), true));
            }
        }

        if (indirectArguments != 0)
        {
            ComputeIndirectDispatches++;
            NoteComputeGuestRange(indirectArguments, 3 * sizeof(uint), written: false, indirectRange.Buffer);
            footprint.Accesses.Add(new(MetalComputeAddressSpace.Buffer,
                (ulong)indirectRange.Buffer, indirectRange.Offset, 3 * sizeof(uint), false, indirectRange.Buffer));
            footprint.Accesses.Add(new(MetalComputeAddressSpace.Device,
                0, indirectRange.GpuAddress, 3 * sizeof(uint), false, indirectRange.Buffer));
        }

        var reason = _computeDependencies.BarrierReason(footprint);
        if (ComputeChainEnabled && _pendingComputePipeline is { } active)
        {
            if (!_computeChainShaders.TryGetValue(active.VertexHash, out var counts))
                _computeChainShaders.Add(active.VertexHash, counts = new());
            counts.Dispatches++;
            if (footprint.UnknownRead || footprint.UnknownWrite) counts.Unknown++;
            if (reason != "none") counts.Dependencies++;
        }
        if (reason != "none")
        {
            ComputeDependencyBarriers++;
            _computeDependencyReasons[reason] = _computeDependencyReasons.GetValueOrDefault(reason) + 1;
            FlushComputeBatch("dependency");
            Span<nint> resources = stackalloc nint[64];
            if (ComputeChainEnabled && _computeDependencies.TryBarrierResources(footprint, resources, out var count))
            {
                ComputeResourceBarrier(resources[..count]);
                _computeDependencies.RetireResources(resources[..count]);
            }
            else ComputeScopeBarrier();
        }
        else if (_computeDependencies.HasDispatch)
        {
            ComputeIndependentDispatches++;
        }

        _computeDependencies.Record(footprint);
        // Auxiliary and indirect accesses belong to this dispatch, not its bindings.
        footprint.Accesses.RemoveRange(accessCount, footprint.Accesses.Count - accessCount);
        footprint.UnknownRead = unknownRead;
        footprint.UnknownWrite = unknownWrite;
    }

    private void NoteComputeGuestRange(ulong address, ulong size, bool written, nint resource = 0)
    {
        if (size == 0) return;
        var previousCount = _computeFootprint.Accesses.Count;
        _computeFootprint.Guest(address, size, written);
        if (_computeFootprint.Accesses.Count != previousCount)
            _computeFootprint.Accesses[^1] = _computeFootprint.Accesses[^1] with { NativeResource = resource };
        if (_backing.TryResolveBackingAliasSpan(address, size, out var alias))
        {
            // Distinct guest virtual addresses can map the very same physical bytes.
            _computeFootprint.Accesses.Add(new(MetalComputeAddressSpace.Backing, 0, alias, size, written, resource));
        }
        else
        {
            // Scattered, partial or unresolvable backing has no canonical extent.
            // Do not assume that different virtual addresses mean different storage.
            _computeFootprint.UnknownRead = true;
            _computeFootprint.UnknownWrite |= written;
        }
    }

    private unsafe void ComputeResourceBarrier(ReadOnlySpan<nint> resources)
    {
        fixed (nint* pointer = resources)
            MetalV2Native.SendComputeResourceBarrier(_computeEncoder, SelComputeResourceBarrier,
                (nint)pointer, (nuint)resources.Length);
        ComputeResourceBarriers++;
    }

    private (ulong Frames, long Batched, long Independent, long Barriers, long Encoders) _computeReportPrevious;

    private string ComputeChainCounters()
    {
        var frames = _bindingPresentId - _computeReportPrevious.Frames;
        string Rate(long current, long previous) =>
            (frames == 0 ? 0 : (double)(current - previous) / frames).ToString("F2", CultureInfo.InvariantCulture);
        var rates = $"compute_window_frames={frames} " +
            $"compute_batched_pf={Rate(ComputeBatchedDispatches, _computeReportPrevious.Batched)} " +
            $"compute_independent_pf={Rate(ComputeIndependentDispatches, _computeReportPrevious.Independent)} " +
            $"compute_barriers_pf={Rate(ComputeScopeBarriers + ComputeResourceBarriers, _computeReportPrevious.Barriers)} " +
            $"compute_encoders_pf={Rate(ComputeEncoders, _computeReportPrevious.Encoders)} ";
        _computeReportPrevious = (_bindingPresentId, ComputeBatchedDispatches, ComputeIndependentDispatches,
            ComputeScopeBarriers + ComputeResourceBarriers, ComputeEncoders);
        return rates + $"compute_chain={(ComputeChainEnabled ? 1 : 0)} compute_frames={_bindingPresentId} " +
        $"compute_dispatches={ComputeDispatches} compute_concurrent_independent={ComputeIndependentDispatches} " +
        $"compute_indirect_gpu={ComputeIndirectDispatches} compute_scope_barriers={ComputeScopeBarriers} " +
        $"compute_resource_barriers={ComputeResourceBarriers} compute_guest_barriers={ComputeGuestBarriers} " +
        $"compute_dependency_reasons=[{string.Join(',', _computeDependencyReasons.Select(p => $"{p.Key}:{p.Value}"))}] " +
        $"compute_encoder_ends=[{string.Join(',', _computeEncoderEnds.Select(p => $"{p.Key}:{p.Value}"))}] " +
        $"compute_cp_copies={ComputeCommandProcessorCopies} " +
        $"compute_cp_refused=[{string.Join(',', _computeCopyRefusals.Select(p => $"{p.Key}:{p.Value}"))}] " +
        $"compute_chain_shaders=[{string.Join(',', _computeChainShaders.OrderByDescending(p => p.Value.Dispatches).Take(12)
            .Select(p => $"{p.Key:X16}:dispatch={p.Value.Dispatches}/unknown={p.Value.Unknown}/dependency={p.Value.Dependencies}"))}] ";
    }

    // Reset only after encoding the barrier, including guest and shared-transfer cuts.
    private void ComputeScopeBarrier(bool force = false)
    {
        FlushComputeBatch("barrier");
        if (ComputeChainEnabled && !force && !_computeDependencies.HasDispatch) return;
        ComputeScopeBarriers++;
        MetalV2Native.SendMemoryBarrierScope(
            _computeEncoder, SelComputeMemoryBarrier, Mtl.BarrierScopeBuffers | Mtl.BarrierScopeTextures);
        _computeDependencies.Reset();
    }

    // Guest windows are untracked. Explicitly join all preceding work before and
    // after a concurrent encoder, including CP labels, staged stores and transfers.
    // MTLEvent signal completes prior commands; the wait gates all subsequent passes.
    // This is GPU-only and does not commit or wait on the CPU.
    private void OrderComputeEncoderBoundary(bool starting = false)
    {
        _computeDependencies.Reset();
        if (!ConcurrentComputeEnabled) return;
        if (starting)
        {
            if (_submissionSerial > 1)
                MetalV2Native.SendEventValue(_commandBuffer, SelEncodeWaitForEvent, _sharedEvent, _submissionSerial - 1);
            _lastConcurrentComputeSerial = _submissionSerial;
        }
        if (_computeBoundaryEvent == 0 || _computeBoundarySerial != _submissionSerial)
        {
            // A command buffer retains its events. Give each submission its own
            // sequence so overlapping buffers cannot signal this event backwards.
            if (_computeBoundaryEvent != 0) MetalNative.SendVoid(_computeBoundaryEvent, SelRelease);
            _computeBoundaryEvent = MetalNative.Send(_device, SelNewSharedEvent);
            if (_computeBoundaryEvent == 0) throw Fatal("A Metal compute boundary event could not be created.");
            _computeBoundarySerial = _submissionSerial;
            _computeBoundaryValue = 0;
        }

        var value = ++_computeBoundaryValue;
        MetalV2Native.SendEventValue(_commandBuffer, SelEncodeSignalEvent, _computeBoundaryEvent, value);
        MetalV2Native.SendEventValue(_commandBuffer, SelEncodeWaitForEvent, _computeBoundaryEvent, value);
    }
}
