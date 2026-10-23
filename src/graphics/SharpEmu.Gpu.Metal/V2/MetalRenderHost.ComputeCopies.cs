// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Runtime.InteropServices;

namespace SharpEmu.Gpu.Metal.V2;

internal sealed partial class MetalRenderHost
{
    private nint _computeCopyPipeline;
    internal long ComputeCommandProcessorCopies { get; private set; }
    internal long ComputeCommandProcessorCopyRefusals(string reason) => _computeCopyRefusals.GetValueOrDefault(reason);
    private readonly Dictionary<string, long> _computeCopyRefusals = new(StringComparer.Ordinal);

    private bool TryEncodeComputeCommandProcessorCopies()
    {
        if (!ComputeChainEnabled || _computeEncoder == 0 || _encodingDeferredLabels) return false;
        var copies = CollectionsMarshal.AsSpan(_deferredLabelCopies);
        long bytes = 0;
        foreach (ref readonly var copy in copies)
        {
            // An EOP label promises CPU visibility of *all* earlier work. A device
            // barrier cannot provide that promise. Mixed sets keep their original order.
            if (!copy.CommandProcessor) return Refuse("eop");
            bytes += copy.Bytes;
        }
        if (copies.Length > 256 || bytes > 65536) return Refuse("capacity");
        FlushComputeBatch("cp-copies");
        if (_computeCopyPipeline == 0)
        {
            const string source = """
                #include <metal_stdlib>
                using namespace metal;
                struct Copy { device const uchar* source; device uchar* destination; ulong bytes; };
                kernel void cp_copies(device const Copy* copies [[buffer(0)]], constant uint& count [[buffer(1)]])
                {
                    // One invocation preserves overlapping writes and partial dwords exactly.
                    for (uint i = 0; i < count; ++i)
                        for (ulong j = 0; j < copies[i].bytes; ++j)
                            copies[i].destination[j] = copies[i].source[j];
                }
                """;
            var library = TryCompileLibrary(_device, source, false, out var error);
            if (library == 0) throw Fatal($"The CP copy kernel did not compile: {error}");
            nint function = 0;
            try
            {
                function = MetalNative.Send(library, MetalNative.Selector("newFunctionWithName:"), MetalNative.NsString("cp_copies"));
                nint pipelineError = 0;
                _computeCopyPipeline = MetalNative.Send(_device,
                    MetalNative.Selector("newComputePipelineStateWithFunction:error:"), function, ref pipelineError);
                if (_computeCopyPipeline == 0) throw Fatal("The CP copy pipeline could not be created.");
            }
            finally
            {
                if (function != 0) MetalNative.SendVoid(function, SelRelease);
                MetalNative.SendVoid(library, SelRelease);
            }
        }

        var table = _guestMemory.Allocate((ulong)copies.Length * 24, 256, out var payload);
        for (var index = 0; index < copies.Length; index++)
        {
            ref readonly var copy = ref copies[index];
            var row = payload.Slice(index * 24, 24);
            BinaryPrimitives.WriteUInt64LittleEndian(row,
                MetalNative.SendGpuAddress(copy.StagingBuffer, SelGpuAddress) + copy.StagingOffset);
            BinaryPrimitives.WriteUInt64LittleEndian(row[8..],
                MetalNative.SendGpuAddress(copy.DestinationBuffer, SelGpuAddress) + copy.DestinationOffset);
            BinaryPrimitives.WriteUInt64LittleEndian(row[16..], (ulong)copy.Bytes);
        }

        // Prior shaders finish before these stores; later shaders see them. No CPU
        // wait, shared event or encoder boundary. Sources are immutable staging bytes.
        ComputeScopeBarrier();
        MetalNative.SendVoid(_computeEncoder, SelSetComputePipelineState, _computeCopyPipeline);
        MetalNative.SendSetBuffer(_computeEncoder, SelSetComputeBuffer, table.Buffer, (nuint)table.Offset, 0);
        unsafe
        {
            var count = (uint)copies.Length;
            MetalNative.SendSetBytes(_computeEncoder, MetalNative.Selector("setBytes:length:atIndex:"),
                (nint)(&count), sizeof(uint), 1);
        }
        foreach (ref readonly var copy in copies)
        {
            MetalNative.SendUseResourceUsage(_computeEncoder, MetalNative.Selector("useResource:usage:"),
                copy.StagingBuffer, Mtl.ResourceUsageRead);
            MetalNative.SendUseResourceUsage(_computeEncoder, MetalNative.Selector("useResource:usage:"),
                copy.DestinationBuffer, Mtl.ResourceUsageWrite);
        }
        MetalNative.SendDispatch(_computeEncoder, SelDispatchThreadgroups,
            new MtlSize { Width = 1, Height = 1, Depth = 1 }, new MtlSize { Width = 1, Height = 1, Depth = 1 });
        ComputeScopeBarrier(force: true);
        _boundComputePipeline = 0;
        ComputeCommandProcessorCopies += copies.Length;
        ClearHeldLabelCopies();
        if (PassProfileEnabled) PassProfileNoteWork("cp-copies", "transfer");
        return true;
    }

    private bool Refuse(string reason)
    {
        _computeCopyRefusals[reason] = _computeCopyRefusals.GetValueOrDefault(reason) + 1;
        return false;
    }
}
