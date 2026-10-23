// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

internal static partial class MetalNative
{
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    private static partial void DrawFetch(nint receiver, nint selector, nuint primitive,
        nuint first, nuint count, nuint instances, nuint baseInstance);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    private static partial void DrawIndexedFetch(nint receiver, nint selector, nuint primitive,
        nuint count, nuint indexType, nint indices, nuint offset, nuint instances, nint baseVertex, nuint baseInstance);

    // Capture a translated vertex export directly to uint4 storage, bypassing
    // raster/interpolation/float render-target conversion. Every output bit counts.
    public static bool TryCaptureVertexFetch(Gen5MslShader shader, byte[] data, uint nativeFormat,
        uint stride, bool instance, uint divisor, uint firstVertex, uint baseInstance, bool indexed,
        out byte[] result, out string error)
    {
        result = [];
        var source = shader.Source.Replace("    uint sharpemu_vertex_id [[vertex_id]],",
            "    device uint4* captured [[buffer(4)]],\n    uint sharpemu_vertex_id [[vertex_id]],", StringComparison.Ordinal)
            .Replace("return sharpemu_out;",
                "captured[sharpemu_instance_id * 32u + sharpemu_vertex_id] = as_type<uint4>(sharpemu_out.param0);\n    return;",
                StringComparison.Ordinal)
            .Replace("vertex Gen5VsOut ", "vertex void ", StringComparison.Ordinal);
        if (!TryCompileLibrary(source, out var library, out error)) return false;
        var owned = new List<nint> { library };
        nint Own(nint handle) { if (handle != 0) owned.Add(handle); return handle; }
        try
        {
            var function = Own(Send(library, Selector("newFunctionWithName:"), NsString(shader.EntryPoint)));
            var descriptor = Own(Send(Send(objc_getClass("MTLRenderPipelineDescriptor"), Selector("alloc")), Selector("init")));
            SendVoid(descriptor, Selector("setVertexFunction:"), function);
            SendVoidBool(descriptor, Selector("setRasterizationEnabled:"), false);
            if (nativeFormat != 0)
            {
                var vertex = Own(Send(Send(objc_getClass("MTLVertexDescriptor"), Selector("alloc")), Selector("init")));
                var attribute = SendIndexed(Send(vertex, Selector("attributes")), Selector("objectAtIndexedSubscript:"), 0);
                SendInteger(attribute, Selector("setFormat:"), nativeFormat);
                SendInteger(attribute, Selector("setBufferIndex:"), 2);
                var layout = SendIndexed(Send(vertex, Selector("layouts")), Selector("objectAtIndexedSubscript:"), 2);
                SendInteger(layout, Selector("setStride:"), Math.Max(stride, 4));
                SendInteger(layout, Selector("setStepFunction:"), stride == 0 ? 0u : instance ? 2u : 1u);
                SendInteger(layout, Selector("setStepRate:"), stride == 0 ? 0 : divisor);
                SendVoid(descriptor, Selector("setVertexDescriptor:"), vertex);
            }
            nint nsError = 0;
            var pipeline = Own(Send(Device.Value, Selector("newRenderPipelineStateWithDescriptor:error:"), descriptor, ref nsError));
            if (pipeline == 0) { error = DescribeError(nsError); return false; }
            var input = Own(SharedBuffer(data));
            var captured = Own(SharedBuffer(new byte[16 * 32 * 16]));
            var step = Own(SharedBuffer(new byte[4]));
            var arguments = BuildArgumentBuffer(shader.ArgumentLayout!, SendULong(input, Selector("gpuAddress")),
                (uint)data.Length, SendULong(step, Selector("gpuAddress")));
            var argumentBuffer = Own(SharedBuffer(arguments));
            var push = Own(SharedBuffer(new byte[4096]));
            var indices = Own(SharedBuffer(new byte[] { 0, 0, 2, 0, 1, 0, 3, 0 }));
            var queue = Own(Send(Device.Value, Selector("newCommandQueue")));
            var command = Send(queue, Selector("commandBuffer"));
            var pass = Send(objc_getClass("MTLRenderPassDescriptor"), Selector("renderPassDescriptor"));
            SendInteger(pass, Selector("setRenderTargetWidth:"), 1);
            SendInteger(pass, Selector("setRenderTargetHeight:"), 1);
            SendInteger(pass, Selector("setDefaultRasterSampleCount:"), 1);
            var encoder = Send(command, Selector("renderCommandEncoderWithDescriptor:"), pass);
            SendVoid(encoder, Selector("setRenderPipelineState:"), pipeline);
            foreach (var (buffer, slot) in new[] { (argumentBuffer, 0u), (push, 1u), (input, 2u), (captured, 4u) })
                SendSetBuffer(encoder, Selector("setVertexBuffer:offset:atIndex:"), buffer, 0, slot);
            SendUseResource(encoder, Selector("useResource:usage:"), input, 1);
            SendUseResource(encoder, Selector("useResource:usage:"), step, 3);
            if (indexed)
                DrawIndexedFetch(encoder, Selector("drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:baseInstance:"),
                    0, 4, 0, indices, 0, 3, (nint)firstVertex, baseInstance);
            else
                DrawFetch(encoder, Selector("drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:"),
                    0, firstVertex, 4, 3, baseInstance);
            SendVoid(encoder, Selector("endEncoding"));
            SendVoid(command, Selector("commit"));
            SendVoid(command, Selector("waitUntilCompleted"));
            var commandError = Send(command, Selector("error"));
            if (commandError != 0) { error = DescribeError(commandError); return false; }
            result = new byte[16 * 32 * 16];
            Marshal.Copy(Send(captured, Selector("contents")), result, 0, result.Length);
            return true;
        }
        finally
        {
            for (var index = owned.Count - 1; index >= 0; index--) SendVoid(owned[index], Selector("release"));
        }
    }
}
