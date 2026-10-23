// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;

namespace SharpEmu.Gpu.Metal.V2;

internal sealed partial class MetalRenderHost
{
    // Copy only small argument/push records. Bulk constants/vertices stay in guest
    // no-copy windows. A rasterization-disabled vertex producer can run inside the
    // existing pass (as the rectangle rebase producer does); a compute patch encoder
    // here would end the pass and undo the Batch 1 tile/load/store joins.
    private const string IcbPatchSource = """
        #include <metal_stdlib>
        using namespace metal;
        struct Patch { device const uint* source; device uint* destination; uint words; uint pad; };
        vertex void sharpemu_icb_patch(uint id [[vertex_id]], device const Patch* patches [[buffer(30)]])
        {
            const Patch patch = patches[id];
            for (uint i = 0; i < patch.words; ++i) patch.destination[i] = patch.source[i];
        }
        """;
    private nint _icbPatchPipeline;
    private bool _icbPatchUnavailable;
    private long _icbPatchRecords, _icbPatchBytes, _totalIcbPatchRecords;
    internal long IcbPatchedRecords => _totalIcbPatchRecords;
    private readonly Dictionary<nint, ulong> _icbPatchAddresses = new();

    private bool PatchIcbArguments(MetalIndirectCommands.Entry entry, ReadOnlySpan<MetalRecipeBuffer> bindings)
    {
        if (entry.PatchBuffer == 0) return true;
        var pipeline = IcbPatchPipeline();
        if (pipeline == 0) return false;
        var count = 0;
        foreach (var binding in bindings) if (binding.Index < 2) count++;
        var records = _guestMemory.Allocate((ulong)count * 24, 256, out var bytes);
        if (!records.IsValid) return false;
        var words = MemoryMarshal.Cast<byte, ulong>(bytes);
        _icbPatchAddresses.Clear(); // Native handles can be recycled between submissions.
        var cursor = 0;
        for (var i = 0; i < bindings.Length; i++)
        {
            var binding = bindings[i];
            if (binding.Index >= 2) continue;
            if (!_icbPatchAddresses.TryGetValue(binding.Buffer, out var address))
                _icbPatchAddresses[binding.Buffer] = address = MetalNative.SendGpuAddress(binding.Buffer, MetalNative.Selector("gpuAddress"));
            words[cursor++] = address + binding.Offset;
            words[cursor++] = entry.PatchAddress + entry.PatchOffsets[i];
            words[cursor++] = binding.ByteSize / 4; // uint words, uint pad=0
            _icbPatchRecords++; _totalIcbPatchRecords++; _icbPatchBytes += binding.ByteSize;
        }
        // This bank is written only once per encoder. The cache selects another
        // bank if this ICB shape recurs in the same pass, avoiding a fragment ->
        // vertex WAR dependency unsupported by TBDR. Across encoders/submissions,
        // tracked storage and useResource order GPU writes and consumers; the CPU
        // never modifies this private buffer or waits for an old reader.
        MetalNative.SendUseResource(_renderEncoder, SelUseResourceStages, entry.PatchBuffer,
            Mtl.ResourceUsageRead | Mtl.ResourceUsageWrite, Mtl.RenderStageVertex | Mtl.RenderStageFragment);
        MetalNative.SendVoid(_renderEncoder, SelSetRenderPipelineState, pipeline);
        MetalNative.SendSetBuffer(_renderEncoder, SelSetVertexBuffer, records.Buffer, (nuint)records.Offset, 30);
        MetalV2Native.SendDrawPrimitivesBaseInstance(_renderEncoder, SelDrawPrimitives, Mtl.PrimitivePoint, 0, (nuint)count, 1, 0);
        MetalV2Native.SendMemoryBarrierStages(_renderEncoder, SelRenderMemoryBarrier, Mtl.BarrierScopeBuffers,
            Mtl.RenderStageVertex, Mtl.RenderStageVertex | Mtl.RenderStageFragment);
        return true;
    }

    private nint IcbPatchPipeline()
    {
        if (_icbPatchPipeline != 0) return _icbPatchPipeline;
        if (_icbPatchUnavailable) return 0;
        _icbPatchUnavailable = true;
        var library = TryCompileLibrary(_device, IcbPatchSource, usesPerVertexInputs: false, out var error, out _, askArchive: false);
        if (library == 0) return 0;
        var function = MetalNative.Send(library, MetalNative.Selector("newFunctionWithName:"), MetalNative.NsString("sharpemu_icb_patch"));
        MetalNative.SendVoid(library, SelRelease);
        if (function == 0) return 0;
        var descriptor = MetalNative.Send(MetalNative.Send(MetalNative.Class("MTLRenderPipelineDescriptor"),
            MetalNative.Selector("alloc")), MetalNative.Selector("init"));
        try
        {
            MetalNative.SendVoid(descriptor, MetalNative.Selector("setVertexFunction:"), function);
            MetalNative.SendVoidBool(descriptor, MetalNative.Selector("setRasterizationEnabled:"), false);
            nint nativeError = 0;
            _icbPatchPipeline = MetalNative.Send(_device, MetalNative.Selector("newRenderPipelineStateWithDescriptor:error:"), descriptor, ref nativeError);
            // An unavailable helper refuses this run; ordinary native draws remain valid.
            _icbPatchUnavailable = _icbPatchPipeline == 0;
            return _icbPatchPipeline;
        }
        finally
        {
            MetalNative.SendVoid(function, SelRelease);
            MetalNative.SendVoid(descriptor, SelRelease);
        }
    }
}
