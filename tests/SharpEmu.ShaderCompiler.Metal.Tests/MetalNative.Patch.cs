// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>
/// Rectangle-list draws over the same objc_msgSend harness, set up the way the v2 render
/// host draws them: the program's post-tessellation form (Gen5MslRectangleList) behind a
/// tessellation pipeline, one three-control-point patch per rectangle at factor 1, the
/// draw block and index buffer in their slots, into an RGBA32Float 2D target.
/// </summary>
internal static partial class MetalNative
{
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    private static partial void SendDrawPatches(nint receiver, nint selector, nuint controlPoints, nuint patchStart,
        nuint patchCount, nint patchIndexBuffer, nuint patchIndexBufferOffset, nuint instanceCount, nuint baseInstance);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    private static partial void SendDrawIndexedPatches(nint receiver, nint selector, nuint controlPoints, nuint patchStart,
        nuint patchCount, nint patchIndexBuffer, nuint patchIndexBufferOffset, nint controlPointIndexBuffer,
        nuint controlPointIndexBufferOffset, nuint instanceCount, nuint baseInstance);

    /// <summary>Draws <paramref name="patchCount"/> rectangles. <paramref name="indices"/>
    /// makes it an indexed draw with 16-bit control-point indices; <paramref name="positions"/>
    /// (x, y per vertex) is bound as a float2 attribute 0 at buffer 2, stepped per control
    /// point; <paramref name="vertexBase"/> is the automatic draw's first vertex.</summary>
    public static bool TryRenderRectanglePatches(
        nint library,
        string fragmentEntryPoint,
        uint resourceBytes,
        uint patchCount,
        uint vertexBase,
        ushort[]? indices,
        float[]? positions,
        uint width,
        uint height,
        out float[] pixels,
        out string error)
    {
        pixels = [];
        error = string.Empty;

        var vertexFunction = Send(library, Selector("newFunctionWithName:"), NsString(Gen5MslRectangleList.EntryPoint));
        var fragmentFunction = Send(library, Selector("newFunctionWithName:"), NsString(fragmentEntryPoint));
        if (vertexFunction == 0 || fragmentFunction == 0)
        {
            error = $"entry points '{Gen5MslRectangleList.EntryPoint}'/'{fragmentEntryPoint}' not both found in the library";
            return false;
        }

        var descriptor = Send(Send(objc_getClass("MTLRenderPipelineDescriptor"), Selector("alloc")), Selector("init"));
        SendVoid(descriptor, Selector("setVertexFunction:"), vertexFunction);
        SendVoid(descriptor, Selector("setFragmentFunction:"), fragmentFunction);
        var attachment = SendIndexed(Send(descriptor, Selector("colorAttachments")), Selector("objectAtIndexedSubscript:"), 0);
        SendInteger(attachment, Selector("setPixelFormat:"), ColorPixelFormat);
        SendInteger(descriptor, Selector("setTessellationFactorStepFunction:"), 0); // Constant
        SendInteger(descriptor, Selector("setTessellationPartitionMode:"), 1); // Integer
        SendInteger(descriptor, Selector("setTessellationOutputWindingOrder:"), 1); // CounterClockwise
        SendInteger(descriptor, Selector("setTessellationControlPointIndexType:"), indices is null ? 0u : 1u); // None / UInt16
        if (positions is not null)
        {
            var vertexDescriptor = Send(Send(objc_getClass("MTLVertexDescriptor"), Selector("alloc")), Selector("init"));
            var attribute = SendIndexed(Send(vertexDescriptor, Selector("attributes")), Selector("objectAtIndexedSubscript:"), 0);
            SendInteger(attribute, Selector("setFormat:"), 29); // MTLVertexFormatFloat2
            SendInteger(attribute, Selector("setOffset:"), 0);
            SendInteger(attribute, Selector("setBufferIndex:"), 2);
            var layout = SendIndexed(Send(vertexDescriptor, Selector("layouts")), Selector("objectAtIndexedSubscript:"), 2);
            SendInteger(layout, Selector("setStride:"), 8);
            SendInteger(layout, Selector("setStepFunction:"), 4); // PerPatchControlPoint
            SendVoid(descriptor, Selector("setVertexDescriptor:"), vertexDescriptor);
        }

        nint nsError = 0;
        var pipeline = Send(Device.Value, Selector("newRenderPipelineStateWithDescriptor:error:"), descriptor, ref nsError);
        if (pipeline == 0)
        {
            error = $"render pipeline creation failed: {DescribeError(nsError)}";
            return false;
        }

        var textureDescriptor = Send(Send(objc_getClass("MTLTextureDescriptor"), Selector("alloc")), Selector("init"));
        SendInteger(textureDescriptor, Selector("setPixelFormat:"), ColorPixelFormat);
        SendInteger(textureDescriptor, Selector("setTextureType:"), 2); // MTLTextureType2D
        SendInteger(textureDescriptor, Selector("setWidth:"), width);
        SendInteger(textureDescriptor, Selector("setHeight:"), height);
        SendInteger(textureDescriptor, Selector("setStorageMode:"), 0); // Shared, for readback
        SendInteger(textureDescriptor, Selector("setUsage:"), 5); // RenderTarget | ShaderRead
        var target = Send(Device.Value, Selector("newTextureWithDescriptor:"), textureDescriptor);
        if (target == 0)
        {
            error = "failed to create the render target";
            return false;
        }

        var pass = Send(objc_getClass("MTLRenderPassDescriptor"), Selector("renderPassDescriptor"));
        var passAttachment = SendIndexed(Send(pass, Selector("colorAttachments")), Selector("objectAtIndexedSubscript:"), 0);
        SendVoid(passAttachment, Selector("setTexture:"), target);
        SendInteger(passAttachment, Selector("setLoadAction:"), 2); // MTLLoadActionClear
        SendInteger(passAttachment, Selector("setStoreAction:"), 1); // MTLStoreActionStore
        SendClearColor(passAttachment, Selector("setClearColor:"), default);

        var queue = Send(Device.Value, Selector("newCommandQueue"));
        var commandBuffer = Send(queue, Selector("commandBuffer"));
        var encoder = Send(commandBuffer, Selector("renderCommandEncoderWithDescriptor:"), pass);
        if (encoder == 0)
        {
            error = "failed to create the render command encoder";
            return false;
        }

        var setVertexBuffer = Selector("setVertexBuffer:offset:atIndex:");
        SendVoid(encoder, Selector("setRenderPipelineState:"), pipeline);
        SendSetBuffer(encoder, setVertexBuffer, SharedBuffer(new byte[Math.Max(resourceBytes, 16u)]), 0, 0);
        if (positions is not null)
        {
            var positionBytes = new byte[positions.Length * sizeof(float)];
            Buffer.BlockCopy(positions, 0, positionBytes, 0, positionBytes.Length);
            SendSetBuffer(encoder, setVertexBuffer, SharedBuffer(positionBytes), 0, 2);
        }

        // Gen5RectDraw: vertex_base, vertex_offset, index_type, first_index, base_vertex.
        var drawBlock = new byte[Gen5MslRectangleList.DrawInfoByteSize];
        BitConverter.TryWriteBytes(drawBlock.AsSpan(0), indices is null ? vertexBase : 0u);
        BitConverter.TryWriteBytes(drawBlock.AsSpan(8), indices is null ? 0u : 1u);
        BitConverter.TryWriteBytes(drawBlock.AsSpan(16), indices is null ? vertexBase : 0u);
        SendSetBuffer(encoder, setVertexBuffer, SharedBuffer(drawBlock), 0, Gen5MslRectangleList.DrawInfoBufferIndex);
        var indexBytes = new byte[Math.Max(((indices?.Length ?? 0) * sizeof(ushort) + 3) & ~3, 16)];
        if (indices is not null)
        {
            Buffer.BlockCopy(indices, 0, indexBytes, 0, indices.Length * sizeof(ushort));
        }

        var indexBuffer = SharedBuffer(indexBytes);
        SendSetBuffer(encoder, setVertexBuffer, indexBuffer, 0, Gen5MslRectangleList.IndexBufferIndex);

        // MTLQuadTessellationFactorsHalf, every factor 1.0.
        var factors = new byte[12];
        for (var index = 0; index < factors.Length; index += 2)
        {
            BitConverter.TryWriteBytes(factors.AsSpan(index), (ushort)0x3C00);
        }

        SendSetBuffer(encoder, Selector("setTessellationFactorBuffer:offset:instanceStride:"), SharedBuffer(factors), 0, 0);
        if (indices is null)
        {
            SendDrawPatches(encoder, Selector(
                    "drawPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:instanceCount:baseInstance:"),
                3, 0, patchCount, 0, 0, 1, 0);
        }
        else
        {
            SendDrawIndexedPatches(encoder, Selector(
                    "drawIndexedPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:" +
                    "controlPointIndexBuffer:controlPointIndexBufferOffset:instanceCount:baseInstance:"),
                3, 0, patchCount, 0, 0, indexBuffer, 0, 1, 0);
        }

        SendVoid(encoder, Selector("endEncoding"));
        SendVoid(commandBuffer, Selector("commit"));
        SendVoid(commandBuffer, Selector("waitUntilCompleted"));

        var commandError = Send(commandBuffer, Selector("error"));
        if (commandError != 0)
        {
            error = "GPU execution failed: " + DescribeError(commandError);
            return false;
        }

        var bytes = ReadFloatTexture(target, width, height, 1, 1);
        pixels = new float[width * height * 4];
        Buffer.BlockCopy(bytes, 0, pixels, 0, bytes.Length);
        return true;
    }
}
