// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>
/// The draw calls that carry a base vertex and a base instance, directly and through an
/// indirect argument buffer, for <see cref="TryRenderVertices"/>'s draw callback.
/// </summary>
internal static partial class MetalNative
{
    private const nuint TrianglePrimitive = 3;
    private const nuint IndexTypeUInt16 = 0;

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    private static partial void SendDrawPrimitivesBaseInstance(nint receiver, nint selector, nuint primitiveType,
        nuint vertexStart, nuint vertexCount, nuint instanceCount, nuint baseInstance);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    private static partial void SendDrawIndexedBaseVertex(nint receiver, nint selector, nuint primitiveType,
        nuint indexCount, nuint indexType, nint indexBuffer, nuint indexBufferOffset, nuint instanceCount,
        nint baseVertex, nuint baseInstance);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    private static partial void SendDrawIndirect(nint receiver, nint selector, nuint primitiveType,
        nint indirectBuffer, nuint indirectBufferOffset);

    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    private static partial void SendDrawIndexedIndirect(nint receiver, nint selector, nuint primitiveType,
        nuint indexType, nint indexBuffer, nuint indexBufferOffset, nint indirectBuffer, nuint indirectBufferOffset);

    /// <summary>A triangle list of <paramref name="vertexCount"/> vertices from
    /// <paramref name="vertexStart"/>, one instance from <paramref name="baseInstance"/>.</summary>
    public static Action<nint> DrawAuto(uint vertexStart, uint vertexCount, uint baseInstance) => encoder =>
        SendDrawPrimitivesBaseInstance(encoder, Selector("drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:"),
            TrianglePrimitive, vertexStart, vertexCount, 1, baseInstance);

    /// <summary>An indexed triangle list over 16-bit <paramref name="indices"/>, one instance.</summary>
    public static Action<nint> DrawIndexed(ushort[] indices, int baseVertex, uint baseInstance) => encoder =>
        SendDrawIndexedBaseVertex(encoder,
            Selector("drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:baseInstance:"),
            TrianglePrimitive, (nuint)indices.Length, IndexTypeUInt16, IndexBuffer(indices), 0, 1, baseVertex, baseInstance);

    /// <summary>The guest's automatic indirect record (vertexCount, instanceCount, firstVertex,
    /// firstInstance), which is MTLDrawPrimitivesIndirectArguments.</summary>
    public static Action<nint> DrawAutoIndirect(uint vertexStart, uint vertexCount, uint baseInstance) => encoder =>
        SendDrawIndirect(encoder, Selector("drawPrimitives:indirectBuffer:indirectBufferOffset:"), TrianglePrimitive,
            SharedBuffer(Words(vertexCount, 1, vertexStart, baseInstance)), 0);

    /// <summary>The guest's indexed indirect record (indexCount, instanceCount, firstIndex,
    /// vertexOffset, firstInstance), which is MTLDrawIndexedPrimitivesIndirectArguments.</summary>
    public static Action<nint> DrawIndexedIndirect(ushort[] indices, int baseVertex, uint baseInstance) => encoder =>
        SendDrawIndexedIndirect(encoder,
            Selector("drawIndexedPrimitives:indexType:indexBuffer:indexBufferOffset:indirectBuffer:indirectBufferOffset:"),
            TrianglePrimitive, IndexTypeUInt16, IndexBuffer(indices), 0,
            SharedBuffer(Words((uint)indices.Length, 1, 0, unchecked((uint)baseVertex), baseInstance)), 0);

    private static nint IndexBuffer(ushort[] indices)
    {
        var bytes = new byte[Math.Max((indices.Length * sizeof(ushort) + 3) & ~3, 16)];
        Buffer.BlockCopy(indices, 0, bytes, 0, indices.Length * sizeof(ushort));
        return SharedBuffer(bytes);
    }

    private static byte[] Words(params uint[] words)
    {
        var bytes = new byte[words.Length * sizeof(uint)];
        Buffer.BlockCopy(words, 0, bytes, 0, bytes.Length);
        return bytes;
    }
}
