// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Runtime.InteropServices;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

internal static partial class MetalNative
{
    internal static bool ExecuteComputeBatch(Gen5MslShader shader, IReadOnlyList<byte[]> arguments,
        IReadOnlyList<byte[]> pushes, IReadOnlyList<uint> groups, IReadOnlyList<int> lengths,
        out byte[][] result, out string error)
    {
        result = [];
        if (!TryCompileLibrary(shader.Source, out var library, out error)) return false;
        var residents = new List<nint>();
        var outputs = new List<nint>();
        var table = new byte[Gen5MslArgumentLayout.ComputeBatchHeaderBytes +
            groups.Count * Gen5MslArgumentLayout.ComputeBatchRequestBytes];
        BinaryPrimitives.WriteUInt32LittleEndian(table, (uint)groups.Count);
        uint first = 0;
        for (var index = 0; index < groups.Count; index++)
        {
            var data = SharedBuffer(new byte[lengths[index]]);
            outputs.Add(data);
            var bytes = arguments[index].ToArray();
            foreach (var field in shader.ArgumentLayout!.Fields)
            {
                if (field.FieldKind == MslArgumentFieldKind.BufferPointers)
                    BinaryPrimitives.WriteUInt64LittleEndian(bytes.AsSpan((int)field.ByteOffset), SendULong(data, Selector("gpuAddress")));
                if (field.FieldKind == MslArgumentFieldKind.BufferByteCounts)
                    BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan((int)field.ByteOffset), (uint)lengths[index]);
            }
            var argument = SharedBuffer(bytes);
            var push = SharedBuffer(pushes[index]);
            residents.AddRange([data, argument, push]);
            var record = table.AsSpan(Gen5MslArgumentLayout.ComputeBatchHeaderBytes +
                index * Gen5MslArgumentLayout.ComputeBatchRequestBytes);
            BinaryPrimitives.WriteUInt64LittleEndian(record, SendULong(argument, Selector("gpuAddress")));
            BinaryPrimitives.WriteUInt64LittleEndian(record[8..], SendULong(push, Selector("gpuAddress")));
            BinaryPrimitives.WriteUInt32LittleEndian(record[16..], first);
            first += groups[index];
            BinaryPrimitives.WriteUInt32LittleEndian(record[20..], first);
            BinaryPrimitives.WriteUInt32LittleEndian(record[24..], groups[index]);
            BinaryPrimitives.WriteUInt32LittleEndian(record[28..], 1);
        }
        var batch = SharedBuffer(table);
        residents.Add(batch);
        if (!RunKernel(library, shader.ComputeBatchEntryPoint!, [(batch, 0)], residents,
            shader.ThreadgroupSizeX, outputs[0], lengths[0], out _, out error, first)) return false;
        result = outputs.Select((buffer, index) =>
        {
            var bytes = new byte[lengths[index]];
            Marshal.Copy(Send(buffer, Selector("contents")), bytes, 0, bytes.Length);
            return bytes;
        }).ToArray();
        return true;
    }
}
