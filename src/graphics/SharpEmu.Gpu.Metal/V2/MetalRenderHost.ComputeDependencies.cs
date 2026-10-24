// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler.Metal;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Gpu.Metal.V2;

internal sealed partial class MetalRenderHost
{
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
    internal long ComputeTransferBarriers { get; private set; }
    internal long ComputeConcurrentDispatches { get; private set; }
    private long _computeOverlapBarriers, _computeUnknownBarriers, _computeCapacityBarriers, _computeFirstDispatches;
    private readonly Dictionary<ulong, ComputeDispatchShape> _computeShapes = new();

    private sealed class ComputeDispatchShape(MetalPipelineEntry pipeline)
    {
        internal readonly ulong Hash = pipeline.VertexHash;
        internal readonly MtlSize Threads = pipeline.ThreadsPerThreadgroup;
        internal long Count, Indirect, OneGroup, Unknown, Dependencies;
        internal uint MinX = uint.MaxValue, MinY = uint.MaxValue, MinZ = uint.MaxValue;
        internal uint MaxX, MaxY, MaxZ;
        internal string Grid => Count == Indirect ? "grid=indirect" :
            string.Create(CultureInfo.InvariantCulture, $"grid_min={MinX}x{MinY}x{MinZ}/grid_max={MaxX}x{MaxY}x{MaxZ}");
    }

    private string ComputeDispatchCounters() =>
        $"compute_concurrent={(ConcurrentComputeEnabled ? 1 : 0)} compute_dispatches={ComputeDispatches} " +
        $"compute_dispatches_concurrent={ComputeConcurrentDispatches} compute_encoders={ComputeEncoders} " +
        $"compute_barriers={ComputeScopeBarriers} compute_barriers_overlap={_computeOverlapBarriers} " +
        $"compute_barriers_unknown={_computeUnknownBarriers} compute_barriers_capacity={_computeCapacityBarriers} " +
        $"compute_barriers_guest={ComputeGuestBarriers} compute_barriers_transfer={ComputeTransferBarriers} " +
        $"compute_barriers_elided_independent={ComputeIndependentDispatches} compute_barriers_elided_first={_computeFirstDispatches} " +
        $"compute_dispatches_coalesced=0 compute_shapes=[{string.Join(';', _computeShapes.Values.OrderByDescending(p => p.Count).Take(12).Select(p =>
            string.Create(CultureInfo.InvariantCulture, $"{p.Hash:X16}:n={p.Count}/indirect={p.Indirect}/one_group={p.OneGroup}/{p.Grid}/threads={p.Threads.Width}x{p.Threads.Height}x{p.Threads.Depth}/unknown={p.Unknown}/dependent={p.Dependencies}")))}] ";

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
        if (layout.BindlessImageTableCount != 0 || stage.StageResources.Resources.BindlessImages.Length != 0)
            footprint.UnknownRead = footprint.UnknownWrite = true;
        for (var index = 0; index < layout.Descriptors.Count; index++)
        {
            var binding = layout.Descriptors[index];
            if (binding.Kind is DescriptorBindingKind.GlobalDataShare or
                DescriptorBindingKind.MipStatistics or DescriptorBindingKind.FaultBuffer)
                footprint.UnknownRead = footprint.UnknownWrite = true;
        }
    }

    private void NoteComputeBuffer(ulong address, uint bytes, in MetalGuestRange range, bool written)
    {
        if (!range.IsValid) return;
        NoteComputeGuestRange(address, bytes, written);
        _computeFootprint.Accesses.Add(new(MetalComputeAddressSpace.Buffer,
            (ulong)range.Buffer, range.Offset, bytes, written));
        _computeFootprint.Accesses.Add(new(MetalComputeAddressSpace.Device,
            0, range.GpuAddress, bytes, written));
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
        NoteComputeGuestRange(description.Data.Address, description.Data.Size, written);
        NoteComputeGuestRange(description.Stencil.Address, description.Stencil.Size, written);
        _computeFootprint.Accesses.Add(new(MetalComputeAddressSpace.Texture, (ulong)image.Texture, 0, 1, written));
    }

    private void OrderComputeDispatch(ulong indirectArguments = 0, MetalGuestRange indirectRange = default,
        uint groupsX = 0, uint groupsY = 0, uint groupsZ = 0)
    {
        ComputeDispatches++;
        ComputeDispatchShape? shape = null;
        if (_pendingComputePipeline is { } active)
        {
            if (!_computeShapes.TryGetValue(active.Id, out shape))
                _computeShapes.Add(active.Id, shape = new(active));
            shape.Count++;
            if (indirectArguments != 0) shape.Indirect++;
            else
            {
                shape.MinX = Math.Min(shape.MinX, groupsX); shape.MaxX = Math.Max(shape.MaxX, groupsX);
                shape.MinY = Math.Min(shape.MinY, groupsY); shape.MaxY = Math.Max(shape.MaxY, groupsY);
                shape.MinZ = Math.Min(shape.MinZ, groupsZ); shape.MaxZ = Math.Max(shape.MaxZ, groupsZ);
                if (groupsX == 1 && groupsY == 1 && groupsZ == 1) shape.OneGroup++;
            }
        }
        if (!ConcurrentComputeEnabled) return;
        ComputeConcurrentDispatches++;
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
            NoteComputeGuestRange(indirectArguments, 3 * sizeof(uint), written: false);
            footprint.Accesses.Add(new(MetalComputeAddressSpace.Buffer,
                (ulong)indirectRange.Buffer, indirectRange.Offset, 3 * sizeof(uint), false));
            footprint.Accesses.Add(new(MetalComputeAddressSpace.Device,
                0, indirectRange.GpuAddress, 3 * sizeof(uint), false));
        }

        var reason = _computeDependencies.BarrierReason(footprint);
        if (shape is not null && (footprint.UnknownRead || footprint.UnknownWrite)) shape.Unknown++;
        if (reason != MetalComputeBarrierReason.None)
        {
            if (shape is not null) shape.Dependencies++;
            switch (reason)
            {
                case MetalComputeBarrierReason.Overlap: _computeOverlapBarriers++; break;
                case MetalComputeBarrierReason.Unknown: _computeUnknownBarriers++; break;
                case MetalComputeBarrierReason.Capacity: _computeCapacityBarriers++; break;
            }
            ComputeDependencyBarriers++;
            ComputeScopeBarrier();
        }
        else if (_computeDependencies.HasDispatch)
        {
            ComputeIndependentDispatches++;
        }
        else _computeFirstDispatches++;

        _computeDependencies.Record(footprint);
        // Auxiliary and indirect accesses belong to this dispatch, not its bindings.
        footprint.Accesses.RemoveRange(accessCount, footprint.Accesses.Count - accessCount);
        footprint.UnknownRead = unknownRead;
        footprint.UnknownWrite = unknownWrite;
    }

    private void NoteComputeGuestRange(ulong address, ulong size, bool written)
    {
        if (size == 0) return;
        _computeFootprint.Guest(address, size, written);
        if (_backing.TryResolveBackingAliasSpan(address, size, out var alias))
        {
            // Distinct guest virtual addresses can map the very same physical bytes.
            _computeFootprint.Accesses.Add(new(MetalComputeAddressSpace.Backing, 0, alias, size, written));
        }
        else
        {
            // Scattered, partial or unresolvable backing has no canonical extent.
            // Do not assume that different virtual addresses mean different storage.
            _computeFootprint.UnknownRead = true;
            _computeFootprint.UnknownWrite |= written;
        }
    }

    // Reset only after encoding the barrier, including guest and shared-transfer cuts.
    private void ComputeScopeBarrier()
    {
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
