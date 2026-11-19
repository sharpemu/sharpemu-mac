// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>
/// Vertex render pipelines over the same objc_msgSend harness: a translated vertex function
/// plus a fragment function draw one non-indexed triangle list into an RGBA32Float 2D array
/// target, optionally across several viewports, and every layer is read back.
/// </summary>
internal static partial class MetalNative
{
    [StructLayout(LayoutKind.Sequential)]
    public struct MtlViewport
    {
        public double OriginX, OriginY, Width, Height, ZNear, ZFar;
    }

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    private static partial void SendDrawPrimitives(nint receiver, nint selector, nuint primitiveType,
        nuint vertexStart, nuint vertexCount);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    private static partial void SendArray(nint receiver, nint selector, nint items, nuint count);

    /// <summary>Draws <paramref name="vertexCount"/> vertices as a triangle list into a
    /// width x height x layers RGBA32Float array target and returns its texels, layer after
    /// layer. The stage resources at buffer 0 are zero-filled; the viewports default to the
    /// whole target. <paramref name="draw"/>, when given, encodes the draw instead.</summary>
    public static bool TryRenderVertices(
        nint library,
        string vertexEntryPoint,
        string fragmentEntryPoint,
        uint resourceBytes,
        uint vertexCount,
        uint width,
        uint height,
        uint layers,
        IReadOnlyList<MtlViewport>? viewports,
        out float[] pixels,
        out string error,
        Action<nint>? draw = null)
    {
        pixels = [];
        error = string.Empty;

        var vertexFunction = Send(library, Selector("newFunctionWithName:"), NsString(vertexEntryPoint));
        var fragmentFunction = Send(library, Selector("newFunctionWithName:"), NsString(fragmentEntryPoint));
        if (vertexFunction == 0 || fragmentFunction == 0)
        {
            error = $"entry points '{vertexEntryPoint}'/'{fragmentEntryPoint}' not both found in the library";
            return false;
        }

        var descriptor = Send(Send(objc_getClass("MTLRenderPipelineDescriptor"), Selector("alloc")), Selector("init"));
        SendVoid(descriptor, Selector("setVertexFunction:"), vertexFunction);
        SendVoid(descriptor, Selector("setFragmentFunction:"), fragmentFunction);
        var attachment = SendIndexed(Send(descriptor, Selector("colorAttachments")), Selector("objectAtIndexedSubscript:"), 0);
        SendInteger(attachment, Selector("setPixelFormat:"), ColorPixelFormat);
        SendInteger(descriptor, Selector("setInputPrimitiveTopology:"), 3); // MTLPrimitiveTopologyClassTriangle

        nint nsError = 0;
        var pipeline = Send(Device.Value, Selector("newRenderPipelineStateWithDescriptor:error:"), descriptor, ref nsError);
        if (pipeline == 0)
        {
            error = $"render pipeline creation failed: {DescribeError(nsError)}";
            return false;
        }

        var textureDescriptor = Send(Send(objc_getClass("MTLTextureDescriptor"), Selector("alloc")), Selector("init"));
        SendInteger(textureDescriptor, Selector("setPixelFormat:"), ColorPixelFormat);
        SendInteger(textureDescriptor, Selector("setTextureType:"), 3); // MTLTextureType2DArray
        SendInteger(textureDescriptor, Selector("setWidth:"), width);
        SendInteger(textureDescriptor, Selector("setHeight:"), height);
        SendInteger(textureDescriptor, Selector("setArrayLength:"), layers);
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
        SendInteger(pass, Selector("setRenderTargetArrayLength:"), layers);

        var queue = Send(Device.Value, Selector("newCommandQueue"));
        var commandBuffer = Send(queue, Selector("commandBuffer"));
        var encoder = Send(commandBuffer, Selector("renderCommandEncoderWithDescriptor:"), pass);
        if (encoder == 0)
        {
            error = "failed to create the render command encoder";
            return false;
        }

        SendVoid(encoder, Selector("setRenderPipelineState:"), pipeline);
        var resources = SharedBuffer(new byte[Math.Max(resourceBytes, 16u)]);
        SendSetBuffer(encoder, Selector("setVertexBuffer:offset:atIndex:"), resources, 0, 0);
        if (viewports is { Count: > 0 })
        {
            var array = viewports.ToArray();
            unsafe
            {
                fixed (MtlViewport* items = array)
                {
                    SendArray(encoder, Selector("setViewports:count:"), (nint)items, (nuint)array.Length);
                }
            }
        }

        if (draw is null)
        {
            SendDrawPrimitives(encoder, Selector("drawPrimitives:vertexStart:vertexCount:"), 3, 0, vertexCount); // Triangle
        }
        else
        {
            draw(encoder);
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

        var bytes = ReadFloatTexture(target, width, height, layers, 1);
        pixels = new float[width * height * layers * 4];
        Buffer.BlockCopy(bytes, 0, pixels, 0, bytes.Length);
        return true;
    }
}
