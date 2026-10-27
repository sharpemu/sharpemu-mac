// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>
/// Mesh render pipelines over the same objc_msgSend harness: build an
/// MTLMeshRenderPipelineDescriptor from a translated primitive shader plus a fragment
/// function, launch it with drawMeshThreadgroups into a small float render target and read
/// the pixels back. Same dependency-free style as the compute side; lifetimes lean on
/// process teardown, which is fine for a test host.
/// </summary>
internal static partial class MetalNative
{
    [StructLayout(LayoutKind.Sequential)]
    private struct MtlClearColor
    {
        public double Red, Green, Blue, Alpha;
    }

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    private static partial nint SendMeshPipeline(nint receiver, nint selector, nint descriptor,
        nuint options, nint reflection, ref nint error);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    private static partial nint SendIndexed(nint receiver, nint selector, nuint index);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    private static partial void SendClearColor(nint receiver, nint selector, MtlClearColor color);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    private static partial void SendDrawMesh(nint receiver, nint selector, MtlSize threadgroups,
        MtlSize threadsPerObjectThreadgroup, MtlSize threadsPerMeshThreadgroup);

    // MTLPixelFormatRGBA32Float, so the readback is exact.
    private const nuint ColorPixelFormat = 125;

    /// <summary>Renders one mesh draw into a width x height RGBA32Float target and returns
    /// its texels. <paramref name="meshBuffers"/> is bound to the mesh stage at the given
    /// buffer indices; each entry is raw bytes the helper uploads. With
    /// <paramref name="indirect"/> the draw launches from the grid the launch kernel writes
    /// instead of <paramref name="threadgroups"/>.</summary>
    public static bool TryRenderMesh(
        nint library,
        string meshEntryPoint,
        string fragmentEntryPoint,
        IReadOnlyList<(byte[] Bytes, uint Index)> meshBuffers,
        uint threadgroups,
        uint threadsPerThreadgroup,
        uint width,
        uint height,
        out float[] pixels,
        out string error,
        IReadOnlyList<nint>? extraResidentBuffers = null,
        uint layers = 1,
        bool volume = false,
        MeshIndirectLaunch? indirect = null,
        IDictionary<uint, byte[]>? readbackBuffers = null)
    {
        pixels = [];
        error = string.Empty;

        // The meshlet slot is always bound (the null buffer when a draw uses none), as the
        // host does; a test that does not supply cached meshlets gets an empty one.
        if (!meshBuffers.Any(static buffer => buffer.Index == Gen5MslMeshInfo.MeshletBufferIndex))
        {
            meshBuffers = [.. meshBuffers, (new byte[16], Gen5MslMeshInfo.MeshletBufferIndex)];
        }

        var meshFunction = Send(library, Selector("newFunctionWithName:"), NsString(meshEntryPoint));
        var fragmentFunction = Send(library, Selector("newFunctionWithName:"), NsString(fragmentEntryPoint));
        if (meshFunction == 0 || fragmentFunction == 0)
        {
            error = $"entry points '{meshEntryPoint}'/'{fragmentEntryPoint}' not both found in the library";
            return false;
        }

        var descriptorClass = objc_getClass("MTLMeshRenderPipelineDescriptor");
        if (descriptorClass == 0)
        {
            error = "MTLMeshRenderPipelineDescriptor is unavailable on this OS";
            return false;
        }

        var descriptor = Send(Send(descriptorClass, Selector("alloc")), Selector("init"));
        SendVoid(descriptor, Selector("setMeshFunction:"), meshFunction);
        SendVoid(descriptor, Selector("setFragmentFunction:"), fragmentFunction);
        var attachments = Send(descriptor, Selector("colorAttachments"));
        var attachment = SendIndexed(attachments, Selector("objectAtIndexedSubscript:"), 0);
        SendInteger(attachment, Selector("setPixelFormat:"), ColorPixelFormat);
        SendInteger(descriptor, Selector("setMaxTotalThreadsPerMeshThreadgroup:"), threadsPerThreadgroup);

        nint nsError = 0;
        var pipeline = SendMeshPipeline(
            Device.Value,
            Selector("newRenderPipelineStateWithMeshDescriptor:options:reflection:error:"),
            descriptor,
            0,
            0,
            ref nsError);
        if (pipeline == 0)
        {
            error = $"mesh pipeline creation failed: {DescribeError(nsError)}";
            return false;
        }

        // A volume target is one 3D texture whose depth planes are the layers; the pass
        // still needs renderTargetArrayLength set to that depth for a layered draw.
        var target = CreateRenderTarget(width, height, layers, volume: volume);
        if (target == 0)
        {
            error = "failed to create the render target";
            return false;
        }

        var pass = Send(objc_getClass("MTLRenderPassDescriptor"), Selector("renderPassDescriptor"));
        var passAttachment = SendIndexed(
            Send(pass, Selector("colorAttachments")), Selector("objectAtIndexedSubscript:"), 0);
        SendVoid(passAttachment, Selector("setTexture:"), target);
        SendInteger(passAttachment, Selector("setLoadAction:"), 2); // MTLLoadActionClear
        SendInteger(passAttachment, Selector("setStoreAction:"), 1); // MTLStoreActionStore
        SendClearColor(passAttachment, Selector("setClearColor:"), default);
        if (layers > 1)
        {
            SendInteger(pass, Selector("setRenderTargetArrayLength:"), layers);
        }

        var queue = Send(Device.Value, Selector("newCommandQueue"));
        var commandBuffer = Send(queue, Selector("commandBuffer"));
        nint launchSlot = 0;
        nint launchFence = 0;
        if (indirect is not null)
        {
            launchSlot = EncodeMeshLaunch(commandBuffer, indirect, out launchFence, out error);
            if (launchSlot == 0)
            {
                return false;
            }
        }

        var encoder = Send(commandBuffer, Selector("renderCommandEncoderWithDescriptor:"), pass);
        if (encoder == 0)
        {
            error = "failed to create the render command encoder";
            return false;
        }

        SendVoid(encoder, Selector("setRenderPipelineState:"), pipeline);
        var residents = new List<nint>();
        var bound = new List<(nint Buffer, uint Index, int Length)>();
        foreach (var (bytes, index) in meshBuffers)
        {
            var buffer = SharedBuffer(bytes.Length == 0 ? new byte[4] : bytes);
            if (buffer == 0)
            {
                error = $"failed to create the mesh buffer at index {index}";
                return false;
            }

            residents.Add(buffer);
            bound.Add((buffer, index, bytes.Length));
            SendSetBuffer(encoder, Selector("setMeshBuffer:offset:atIndex:"), buffer, 0, index);
        }

        foreach (var resident in extraResidentBuffers ?? [])
        {
            residents.Add(resident);
        }

        foreach (var resident in residents)
        {
            SendUseResource(encoder, Selector("useResource:usage:"), resident, ReadWriteUsage);
        }

        var one = new MtlSize { Width = 1, Height = 1, Depth = 1 };
        if (indirect is not null)
        {
            DrawMeshLaunch(encoder, indirect, launchSlot, launchFence, threadsPerThreadgroup);
        }
        else
        {
            SendDrawMesh(
                encoder,
                Selector("drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:"),
                new MtlSize { Width = threadgroups, Height = 1, Depth = 1 },
                one,
                new MtlSize { Width = threadsPerThreadgroup, Height = 1, Depth = 1 });
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

        if (indirect is not null)
        {
            indirect.Written = ReadSharedBuffer(launchSlot, indirect.Slot.Length);
        }

        if (readbackBuffers is not null)
            foreach (var (buffer, index, length) in bound)
                readbackBuffers[index] = ReadSharedBuffer(buffer, length);

        var bytesOut = volume
            ? ReadFloatTexture(target, width, height, 1, layers)
            : ReadFloatTexture(target, width, height, layers, 1);
        pixels = new float[width * height * layers * 4];
        Buffer.BlockCopy(bytesOut, 0, pixels, 0, bytesOut.Length);
        return true;
    }

    /// <summary>Renders a mesh draw whose shader reads resources through its stage argument
    /// buffer. The argument buffer is filled from the layout the way the compute harness does
    /// it - every buffer field points at the one data buffer, with its byte length - so a
    /// program that fetches vertices through a V# reads <paramref name="bufferContents"/>.</summary>
    public static bool TryRenderMeshWithArgumentBuffer(
        nint library,
        string meshEntryPoint,
        string fragmentEntryPoint,
        byte[] bufferContents,
        byte[] pushData,
        byte[] drawInfo,
        Gen5MslArgumentLayout layout,
        uint drawInfoBufferIndex,
        uint indexBufferIndex,
        uint threadgroups,
        uint threadsPerThreadgroup,
        uint width,
        uint height,
        out float[] pixels,
        out string error)
    {
        pixels = [];
        var data = SharedBuffer(bufferContents.Length == 0 ? new byte[4] : bufferContents);
        if (data == 0)
        {
            error = "failed to create the data buffer";
            return false;
        }

        var address = SendULong(data, Selector("gpuAddress"));
        var arguments = new byte[Math.Max(layout.ByteSize, 8)];
        foreach (var field in layout.Fields)
        {
            for (var element = 0; element < field.Count; element++)
            {
                switch (field.FieldKind)
                {
                    case MslArgumentFieldKind.BufferPointers:
                        BitConverter.TryWriteBytes(arguments.AsSpan((int)field.ByteOffset + (element * 8)), address);
                        break;
                    case MslArgumentFieldKind.BufferByteCounts:
                        BitConverter.TryWriteBytes(
                            arguments.AsSpan((int)field.ByteOffset + (element * 4)), (uint)bufferContents.Length);
                        break;
                }
            }
        }

        return TryRenderMesh(
            library,
            meshEntryPoint,
            fragmentEntryPoint,
            [
                (arguments, (uint)Gen5MslArgumentLayout.ResourcesBufferIndex),
                (pushData, (uint)Gen5MslArgumentLayout.PushDataBufferIndex),
                (drawInfo, drawInfoBufferIndex),
                (new byte[16], indexBufferIndex),
            ],
            threadgroups,
            threadsPerThreadgroup,
            width,
            height,
            out pixels,
            out error,
            [data]);
    }

    private static nint CreateRenderTarget(uint width, uint height, uint layers = 1, nuint pixelFormat = ColorPixelFormat,
        bool volume = false)
    {
        var descriptor = Send(Send(objc_getClass("MTLTextureDescriptor"), Selector("alloc")), Selector("init"));
        SendInteger(descriptor, Selector("setPixelFormat:"), pixelFormat);
        // MTLTextureType3D / 2DArray / 2D
        SendInteger(descriptor, Selector("setTextureType:"), volume ? 7u : layers > 1 ? 3u : 2u);
        SendInteger(descriptor, Selector("setWidth:"), width);
        SendInteger(descriptor, Selector("setHeight:"), height);
        SendInteger(descriptor, Selector("setDepth:"), volume ? layers : 1u);
        SendInteger(descriptor, Selector("setArrayLength:"), volume ? 1u : layers);
        SendInteger(descriptor, Selector("setStorageMode:"), 0); // Shared, for readback
        SendInteger(descriptor, Selector("setUsage:"), 5); // RenderTarget | ShaderRead
        return Send(Device.Value, Selector("newTextureWithDescriptor:"), descriptor);
    }
}
