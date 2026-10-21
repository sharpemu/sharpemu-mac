// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler.Metal;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// An indexed indirect rectangle list whose counts the GPU has yet to write, drawn without
/// reading them on the CPU.
/// </summary>
/// <remarks>
/// <para>
/// A rectangle list draws one three-control-point patch per rectangle (DrawRectangles), so its
/// patch count is the index count divided by three, which Metal cannot read from the guest's
/// argument record (indexCount, instanceCount, firstIndex, vertexOffset, firstInstance); and a
/// patch draw has no base vertex, so the vertex offset has to reach the control-point indices.
/// Reading the record on the CPU waits for the queue to drain: Black Myth: Wukong draws ~50 of
/// these a frame from counts a compute pass has just written, mostly empty, and its GPU worker
/// spent ~24% of its time in that one wait a frame.
/// </para>
/// <para>
/// So the pass builds the patch draw itself. A vertex function with rasterization off is
/// launched as a point draw from the record: Metal reads it as (vertexCount, instanceCount,
/// vertexStart, baseInstance), so there is one invocation per index from vertex firstIndex,
/// with the vertex offset for the base instance. Each writes its index plus the vertex offset
/// as a 32-bit control-point index, and the first writes MTLDrawPatchIndirectArguments
/// (indexCount / 3, instanceCount, 0, firstInstance). After a vertex-to-vertex buffer barrier,
/// the patch draw reads both: indices, per-vertex buffers at their bound offsets and the first
/// instance come out as DrawRectangles makes them. A draw the GPU gave no indices or instances
/// leaves the arguments as the CPU cleared them, so it draws nothing. Nothing ends the pass.
/// </para>
/// <para>
/// Measured on an M5 Pro: inside one pass, with and without multisampling and under the Metal
/// validation layer, the patch draw's indirect arguments and control-point fetch see the
/// rasterization-off draw's stores behind the barrier, over 1500 draws of random counts,
/// first indices, vertex offsets and instance counts. The pipeline names no attachment, and
/// the validation layer accepts it in any pass.
/// SHARPEMU_METAL_V2_RECT_INDIRECT_GPU=0 reads the record on the CPU as before.
/// </para>
/// </remarks>
internal sealed partial class MetalRenderHost
{
    internal static readonly bool IndexedRectanglesOnGpu =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_RECT_INDIRECT_GPU") != "0";

    // The vertex-stage slots the rebase draw uses: record, guest indices, control points,
    // patch arguments and its block. The patch draw rebinds the last two as its own index
    // and draw-block slots; a rectangle list's vertex buffers stay below the first.
    internal const int RectangleRebaseFirstSlot = 26;
    private const int RebaseRecordSlot = 26;
    private const int RebaseIndexSlot = 27;
    private const int RebaseControlPointSlot = 28;
    private const int RebaseArgumentsSlot = 29;
    private const int RebaseBlockSlot = 30;

    // Control points are made for at most this many indices (4 MiB of ring a draw); a draw
    // whose bound index range is larger, or unbounded, is read on the CPU.
    private const uint MaxRectangleRebaseIndices = 1u << 20;

    private const string RectangleRebaseFunction = "sharpemu_rect_rebase";

    private const string RectangleRebaseSource = """
        #include <metal_stdlib>
        using namespace metal;

        struct SharpEmuRectRebase
        {
            uint index_limit;
            uint index_type;
        };

        vertex void sharpemu_rect_rebase(
            uint vertex_id [[vertex_id]],
            uint instance_id [[instance_id]],
            uint base_instance [[base_instance]],
            device const uint* record [[buffer(26)]],
            device const uint* indices [[buffer(27)]],
            device uint* control_points [[buffer(28)]],
            device uint* patch_arguments [[buffer(29)]],
            constant SharpEmuRectRebase& draw [[buffer(30)]])
        {
            if (instance_id != base_instance)
            {
                return;
            }

            const uint count = min(record[0], draw.index_limit);
            const uint local = vertex_id - record[2];
            if (local >= count)
            {
                return;
            }

            const uint index = draw.index_type == 1u
                ? (indices[vertex_id >> 1u] >> ((vertex_id & 1u) * 16u)) & 0xFFFFu
                : indices[vertex_id];
            control_points[local] = index + base_instance;
            if (local == 0u)
            {
                patch_arguments[0] = count / 3u;
                patch_arguments[1] = record[1];
                patch_arguments[2] = 0u;
                patch_arguments[3] = record[4];
            }
        }
        """;

    private static readonly nint SelDrawIndexedPatchesIndirect = MetalNative.Selector(
        "drawIndexedPatches:patchIndexBuffer:patchIndexBufferOffset:controlPointIndexBuffer:" +
        "controlPointIndexBufferOffset:indirectBuffer:indirectBufferOffset:");

    private nint _rectangleRebasePipeline;
    private uint _rectangleRebaseLimit;

    /// <summary>
    /// Whether an indexed indirect rectangle list keeps its counts on the GPU: when pending GPU
    /// work writes the record (otherwise reading it costs no drain, and a draw the CPU finds
    /// empty is left out whole) and its bound index range is small enough to rebase.
    /// </summary>
    public bool TakesIndirectIndexedRectangleListArguments(
        VertexInputInfo vertexInput, ulong argumentsAddress, ulong argumentsSize, uint indexLimit) =>
        IndexedRectanglesOnGpu && indexLimit <= MaxRectangleRebaseIndices &&
        TakesLaunchArguments(argumentsAddress, argumentsSize);

    private void DrawRectanglesIndexedIndirect(MetalPipelineEntry entry, BufferBinding arguments)
    {
        var limit = _rectangleRebaseLimit;
        if (limit > MaxRectangleRebaseIndices)
        {
            throw Fatal($"An indirect rectangle list reached the rebase with an index range too large for it: limit={limit}.");
        }

        var controlPoints = _guestMemory.Allocate(Math.Max(limit, 1u) * (ulong)sizeof(uint), 256, out _);
        var patchArguments = _guestMemory.Allocate(4 * sizeof(uint), 256, out var patchArgumentBytes);
        if (!controlPoints.IsValid || !patchArguments.IsValid)
        {
            throw Fatal($"An indirect rectangle list has no ring space for its control points: limit={limit}.");
        }

        patchArgumentBytes.Clear();
        var rebase = RectangleRebasePipeline();
        MetalNative.SendVoid(_renderEncoder, SelSetRenderPipelineState, rebase);
        _boundRenderState = rebase;
        MetalNative.SendSetBuffer(
            _renderEncoder, SelSetVertexBuffer, (nint)arguments.Handle, (nuint)arguments.Offset, RebaseRecordSlot);
        MetalNative.SendSetBuffer(_renderEncoder, SelSetVertexBuffer, _indexBuffer, (nuint)_indexOffset, RebaseIndexSlot);
        MetalNative.SendSetBuffer(
            _renderEncoder, SelSetVertexBuffer, controlPoints.Buffer, (nuint)controlPoints.Offset, RebaseControlPointSlot);
        MetalNative.SendSetBuffer(
            _renderEncoder, SelSetVertexBuffer, patchArguments.Buffer, (nuint)patchArguments.Offset, RebaseArgumentsSlot);
        Span<uint> block = [limit, _indexType == Mtl.IndexTypeUInt32 ? 2u : 1u];
        unsafe
        {
            fixed (uint* words = block)
            {
                MetalV2Native.SendSetBytes(_renderEncoder, SelSetVertexBytes, (nint)words, (nuint)(block.Length * sizeof(uint)), RebaseBlockSlot);
            }
        }

        for (var slot = RectangleRebaseFirstSlot; slot <= RebaseBlockSlot; slot++)
        {
            _boundVertexBuffers[slot - VertexBufferSlotBase] = -1;
        }

        MetalV2Native.SendDrawIndirect(
            _renderEncoder, SelDrawIndirect, Mtl.PrimitivePoint, (nint)arguments.Handle, (nuint)arguments.Offset);
        MetalV2Native.SendMemoryBarrierStages(
            _renderEncoder, SelRenderMemoryBarrier, Mtl.BarrierScopeBuffers, Mtl.RenderStageVertex, Mtl.RenderStageVertex);

        // The patch draw: DrawRectangles' bindings for a 32-bit indexed draw from index 0 with
        // no vertex offset, the control points as its index buffer.
        var state = RectangleState(entry, 2);
        MetalNative.SendVoid(_renderEncoder, SelSetRenderPipelineState, state);
        _boundRenderState = state;
        Span<uint> draw = stackalloc uint[(int)(Gen5MslRectangleList.DrawInfoByteSize / sizeof(uint))];
        draw.Clear();
        draw[2] = 2;
        unsafe
        {
            fixed (uint* words = draw)
            {
                MetalV2Native.SendSetBytes(
                    _renderEncoder,
                    SelSetVertexBytes,
                    (nint)words,
                    Gen5MslRectangleList.DrawInfoByteSize,
                    Gen5MslRectangleList.DrawInfoBufferIndex);
            }
        }

        MetalNative.SendSetBuffer(
            _renderEncoder,
            SelSetVertexBuffer,
            controlPoints.Buffer,
            (nuint)controlPoints.Offset,
            Gen5MslRectangleList.IndexBufferIndex);
        var factors = _guestMemory.Allocate(6 * sizeof(ushort), 256, out var factorBytes);
        System.Runtime.InteropServices.MemoryMarshal.Cast<byte, ushort>(factorBytes).Fill(0x3C00);
        MetalNative.SendSetBuffer(_renderEncoder, SelSetTessellationFactorBuffer, factors.Buffer, (nuint)factors.Offset, 0);
        MetalV2Native.SendDrawIndexedPatchesIndirect(
            _renderEncoder,
            SelDrawIndexedPatchesIndirect,
            Gen5MslRectangleList.VerticesPerRectangle,
            0,
            0,
            controlPoints.Buffer,
            (nuint)controlPoints.Offset,
            patchArguments.Buffer,
            (nuint)patchArguments.Offset);
        Interlocked.Increment(ref _indirectGpuDraws);
        Interlocked.Increment(ref _indirectRectangleDraws);
    }

    private nint RectangleRebasePipeline()
    {
        if (_rectangleRebasePipeline != 0)
        {
            return _rectangleRebasePipeline;
        }

        var pool = MetalNative.objc_autoreleasePoolPush();
        try
        {
            var options = MetalNative.Send(
                MetalNative.Send(MetalNative.Class("MTLCompileOptions"), MetalNative.Selector("alloc")),
                MetalNative.Selector("init"));
            MetalNative.ApplyMslCompileOptions(options);
            nint error = 0;
            var library = MetalNative.Send(
                _device, MetalNative.Selector("newLibraryWithSource:options:error:"),
                MetalNative.NsString(RectangleRebaseSource), options, ref error);
            MetalNative.SendVoid(options, SelRelease);
            if (library == 0)
            {
                throw Fatal($"The rectangle rebase function did not compile: {MetalNative.DescribeError(error)}.");
            }

            var function = MetalNative.Send(
                library, MetalNative.Selector("newFunctionWithName:"), MetalNative.NsString(RectangleRebaseFunction));
            MetalNative.SendVoid(library, SelRelease);
            if (function == 0)
            {
                throw Fatal("The rectangle rebase function is missing.");
            }

            var descriptor = MetalNative.Send(
                MetalNative.Send(MetalNative.Class("MTLRenderPipelineDescriptor"), MetalNative.Selector("alloc")),
                MetalNative.Selector("init"));
            MetalNative.SendVoid(descriptor, MetalNative.Selector("setVertexFunction:"), function);
            MetalNative.SendVoidBool(descriptor, MetalNative.Selector("setRasterizationEnabled:"), false);
            error = 0;
            var pipeline = MetalNative.Send(
                _device, MetalNative.Selector("newRenderPipelineStateWithDescriptor:error:"), descriptor, ref error);
            MetalNative.SendVoid(descriptor, SelRelease);
            MetalNative.SendVoid(function, SelRelease);
            if (pipeline == 0)
            {
                throw Fatal($"The rectangle rebase pipeline did not build: {MetalNative.DescribeError(error)}.");
            }

            _rectangleRebasePipeline = pipeline;
            return pipeline;
        }
        finally
        {
            MetalNative.objc_autoreleasePoolPop(pool);
        }
    }
}
