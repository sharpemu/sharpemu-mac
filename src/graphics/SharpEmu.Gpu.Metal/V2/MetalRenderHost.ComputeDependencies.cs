// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

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
    internal long ComputeDependencyBarriers { get; private set; }
    internal long ComputeGuestBarriers { get; private set; }
    internal long ComputeIndependentDispatches { get; private set; }

    private static readonly nint SelComputeEncoderDispatchType =
        MetalNative.Selector("computeCommandEncoderWithDispatchType:");
    private static readonly nint SelSetComputeDispatchType = MetalNative.Selector("setDispatchType:");

    private void OrderComputeExternalWrites()
    {
        if (!ConcurrentComputeEnabled) return;
        // CP copies and staged writebacks have footprints outside the dispatch tracker.
        // Land them before resolving bindings as well as before native dispatches:
        // virtual-range disjointness alone cannot exclude a physical backing alias.
        OrderAfterDeferredCopies(0, ulong.MaxValue);
        OrderAfterStagedStores(0, ulong.MaxValue);
    }

    private void OrderComputeDispatch(ulong indirectArguments = 0, MetalGuestRange indirectRange = default)
    {
        if (!ConcurrentComputeEnabled) return;
        var footprint = _computeFootprint;
        footprint.Clear();
        var stage = _stageBindings[StageCompute];
        if ((_preparedStageMask & (1 << StageCompute)) == 0 || !stage.Committed)
        {
            // Native/direct callers without a committed resource snapshot have no proof.
            footprint.UnknownRead = footprint.UnknownWrite = true;
        }
        else
        {
            var program = stage.Program;
            var info = program.Resources!.Info;
            var layout = program.Bindings!;
            footprint.UnknownRead = program.UsesDeviceAddresses || info.UsesDeviceAddresses;
            footprint.UnknownWrite = program.WritesDeviceAddresses || info.WritesDeviceAddresses;
            if (layout.BindlessImageTableCount != 0 || stage.StageResources.Resources.BindlessImages.Length != 0 ||
                layout.Descriptors.Any(binding => binding.Kind is DescriptorBindingKind.GlobalDataShare or
                    DescriptorBindingKind.MipStatistics or DescriptorBindingKind.FaultBuffer))
            {
                // Runtime-selected resources and auxiliary atomic feedback have no bounded
                // binding footprint here. Never infer independence from shader identity.
                footprint.UnknownRead = footprint.UnknownWrite = true;
            }

            for (var index = 0; index < stage.BufferCount; index++)
            {
                var range = stage.Buffers[index];
                if (!range.IsValid) continue; // null descriptor: shader cannot access it
                var descriptor = BufferDescriptorWords.From(stage.StageResources.Resources.Buffers[index]);
                var written = info.Buffers[index].Written || info.Buffers[index].Atomic;
                NoteComputeGuestRange(descriptor.Address, stage.BufferLengths[index], written);
                footprint.Accesses.Add(new(MetalComputeAddressSpace.Buffer,
                    (ulong)range.Buffer, range.Offset, stage.BufferLengths[index], written));
                footprint.Accesses.Add(new(MetalComputeAddressSpace.Device,
                    0, range.GpuAddress, stage.BufferLengths[index], written));
            }

            for (var index = 0; index < stage.ImageCount; index++)
            {
                var written = info.Images[index].Written || info.Images[index].Atomic;
                if (!_imageCache.TryGetImage(stage.Images[index], out var image) ||
                    index == stage.MovieLuma || index == stage.MovieChroma)
                {
                    footprint.UnknownRead = true;
                    footprint.UnknownWrite |= written;
                    continue;
                }

                // Use the entire cached backing image, not a texture-view ID or guessed
                // mip extent. This also covers differently formatted/subresource aliases.
                var description = image.Description;
                NoteComputeGuestRange(description.Data.Address, description.Data.Size, written);
                NoteComputeGuestRange(description.Stencil.Address, description.Stencil.Size, written);
                footprint.Accesses.Add(new(MetalComputeAddressSpace.Texture, (ulong)image.Texture, 0, 1, written));
            }

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

        if (_computeDependencies.NeedsBarrier(footprint))
        {
            ComputeDependencyBarriers++;
            ComputeScopeBarrier();
        }
        else if (_computeDependencies.HasDispatch)
        {
            ComputeIndependentDispatches++;
        }

        _computeDependencies.Record(footprint);
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
