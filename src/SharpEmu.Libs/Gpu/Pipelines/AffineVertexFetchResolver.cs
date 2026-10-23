// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Libs.Gpu.Pipelines;

// Backend opt-in: Metal 3.2 has no robust vertex fetch enabled here. Prove the
// complete native fetch range before replacing any individual shader load.
public enum VertexFetchRefusal { None, Descriptor, Format, Selector, Index, BoundsUnknown, BoundsRange, Footprint, Alignment, Resource, Mapping, Capacity, Eligibility, NoAffineIndex }

public static class AffineVertexFetchResolver
{
    public static bool Enabled { get; set; }

    public static bool TryResolve(AffineVertexFetchLoad load, BufferDescriptorWords descriptor,
        VertexFetchBounds bounds, out VertexInputBuffer buffer, out VertexAttributeResource attribute)
        => TryResolve(load, descriptor, bounds, out buffer, out attribute, out _);

    public static bool TryResolve(AffineVertexFetchLoad load, BufferDescriptorWords descriptor,
        VertexFetchBounds bounds, out VertexInputBuffer buffer, out VertexAttributeResource attribute, out VertexFetchRefusal refusal)
    {
        refusal = VertexFetchRefusal.Descriptor;
        buffer = default;
        attribute = default;
        if (load.Raw)
        {
            var format = load.Components switch { 1 => 20u, 2 => 62u, 3 => 72u, _ => 75u };
            descriptor = descriptor with { Word3 = (descriptor.Word3 & ~0x7FFFFu) | format << 12 | 0xFACu };
        }
        if (load.Components is 0 or > 4 || descriptor.Type != 0 || descriptor.SwizzleEnabled || descriptor.AddThreadId ||
            descriptor.Address == 0) return false;
        refusal = VertexFetchRefusal.Format;
        if (!Gfx10UnifiedFormat.TryDecode(descriptor.Format, out var data, out var number) ||
            !NativeFormat(descriptor.Format) || number is 2 or 3) return false;
        var bytes = Gfx10UnifiedFormat.GetAccessByteSize(data, 4);
        if (bytes == 0) return false;
        refusal = VertexFetchRefusal.Selector;
        for (var component = 0; component < load.Components; component++)
        {
            var selector = (descriptor.DestinationSelectXYZW >> (component * 3)) & 7;
            if (selector is 2 or 3) return false;
        }

        refusal = VertexFetchRefusal.Index;
        var index = load.Index;
        if (index.BiasScalarRegister >= 0) return false;
        if (index.Divisor == 0 || (index.Kind == VertexFetchIndex.Vertex && index.Divisor != 1)) return false;
        // Metal adds baseInstance after stepping, while the guest divides the full
        // instance id. Without rebasing outside the guest allocation these differ.
        if (descriptor.Stride != 0 && index.Kind == VertexFetchIndex.Instance && index.Divisor != 1 &&
            (!bounds.InstancesKnown || bounds.FirstInstance != 0)) return false;
        ulong first = 0, last = 0;
        var constant = descriptor.Stride == 0 || index.Kind == VertexFetchIndex.Constant;
        if (!constant)
        {
            var known = index.Kind == VertexFetchIndex.Vertex ? bounds.VerticesKnown : bounds.InstancesKnown;
            var start = index.Kind == VertexFetchIndex.Vertex ? bounds.FirstVertex : bounds.FirstInstance;
            var count = index.Kind == VertexFetchIndex.Vertex ? bounds.VertexCount : bounds.InstanceCount;
            refusal = VertexFetchRefusal.BoundsUnknown;
            if (!known) return false;
            refusal = VertexFetchRefusal.BoundsRange;
            if (count == 0 || (ulong)start + count - 1 > uint.MaxValue) return false;
            first = start / index.Divisor;
            last = ((ulong)start + count - 1) / index.Divisor;
        }
        refusal = VertexFetchRefusal.BoundsRange;
        first += index.Bias;
        last += index.Bias;
        if (last > uint.MaxValue) return false;
        var offset = (ulong)load.OffsetBytes;
        var stride = (ulong)descriptor.Stride;
        var end = last * stride + offset + bytes;
        refusal = VertexFetchRefusal.Footprint;
        var footprint = descriptor.Footprint();
        if (footprint is null || end > footprint || end > uint.MaxValue ||
            (descriptor.Address + end) > 0x1_0000_0000_0000UL) return false;
        // These tests are deliberately stronger than the guest's individual bounds
        // modes. No native attribute may read an extra byte that shader fetch zeros.
        if (descriptor.OutOfBounds == 0 && (last >= descriptor.RecordCount || offset + bytes > stride)) return false;
        if (descriptor.OutOfBounds == 1 && (constant && index.Kind != VertexFetchIndex.Constant || last >= descriptor.RecordCount)) return false;
        if (descriptor.OutOfBounds == 2 && descriptor.RecordCount == 0) return false;
        if (descriptor.OutOfBounds == 3 && stride == 0 && offset + bytes > descriptor.RecordCount) return false;
        refusal = VertexFetchRefusal.Alignment;
        if (stride != 0 && (offset + bytes > stride || (stride & 3) != 0)) return false;
        var address = descriptor.Address + offset + index.Bias * stride;
        if ((address & 3) != 0) return false;
        var nativeStride = constant ? 0u : descriptor.Stride;
        var nativeDescriptor = descriptor.WithAddress(address) with
        {
            Word1 = (descriptor.WithAddress(address).Word1 & ~0x3FFF0000u) | (nativeStride << 16),
            Word2 = constant ? bytes : descriptor.RecordCount - index.Bias,
        };
        buffer = new(address, nativeStride, nativeDescriptor.RecordCount,
            index.Kind == VertexFetchIndex.Instance && !constant, index.Divisor,
            // The native stage never indexes beyond the proved last record. Binding
            // the unused descriptor tail would add mapping checks/copies for no fetch.
            constant ? bytes : end - offset - index.Bias * stride);
        attribute = new(nativeDescriptor, 0, (int)load.Components, 0,
            index.Kind == VertexFetchIndex.Instance && !constant ? 1u : 0u, 0, 0);
        refusal = VertexFetchRefusal.None;
        return true;
    }

    // Exact native equivalents only. Packed integer/scaled and reversed packed
    // floats have no matching Metal vertex format and keep their shader decoder.
    private static bool NativeFormat(uint format) => format is
        1 or 2 or 5 or 6 or 7 or 8 or 11 or 12 or 13 or 14 or 15 or 18 or 19 or
        20 or 21 or 22 or 23 or 24 or 27 or 28 or 29 or 36 or 50 or 51 or
        56 or 57 or 60 or 61 or 62 or 63 or 64 or 65 or 66 or 69 or 70 or 71 or
        72 or 73 or 74 or 75 or 76 or 77;
}
