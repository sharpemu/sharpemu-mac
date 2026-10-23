// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Runtime.InteropServices;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>A table the host binds for one pointer field of the argument buffer (the
/// device-address ranges, the fault bitmap): its buffer, the count the paired word field
/// holds, and the buffers it points into, which have to be resident too.</summary>
internal sealed record DeviceTable(DescriptorBindingKind Kind, nint Buffer, uint Count, IReadOnlyList<nint> Residents);

internal static partial class MetalNative
{
    private static DeviceTable? FindTable(IReadOnlyList<DeviceTable>? tables, DescriptorBindingKind kind) =>
        tables?.FirstOrDefault(table => table.Kind == kind);

    /// <summary>The address-range table and fault bitmap mapping one guest range onto a
    /// shared buffer holding <paramref name="guestBytes"/>, as the v2 host builds them
    /// (Gen5AddressRange: first page, page count, host pointer; 16 KB pages).
    /// <paramref name="trailingRanges"/> one-page entries on the pages after the guest
    /// range follow it, so the guest range is the first of that many more entries.</summary>
    public static (IReadOnlyList<DeviceTable> Tables, nint Faults) CreateGuestMapping(
        ulong guestBase, byte[] guestBytes, uint faultWords, uint trailingRanges = 0, bool indexed = false)
    {
        var pageBytes = 1ul << DeviceAddressPaging.PageBits;
        if (guestBase % pageBytes != 0 || (ulong)guestBytes.Length % pageBytes != 0)
            throw new ArgumentException("the guest range has to cover whole pages");

        var memory = SharedBuffer(guestBytes);
        var gpuAddress = SendULong(memory, Selector("gpuAddress"));
        var firstPage = guestBase >> DeviceAddressPaging.PageBits;
        var pageCount = (ulong)guestBytes.Length / pageBytes;
        var table = new byte[24 * (trailingRanges + 1)];
        BinaryPrimitives.WriteUInt64LittleEndian(table, firstPage);
        BinaryPrimitives.WriteUInt64LittleEndian(table.AsSpan(8), pageCount);
        BinaryPrimitives.WriteUInt64LittleEndian(table.AsSpan(16), gpuAddress);
        for (var index = 1u; index <= trailingRanges; index++)
        {
            var entry = table.AsSpan((int)(index * 24));
            BinaryPrimitives.WriteUInt64LittleEndian(entry, firstPage + pageCount + index);
            BinaryPrimitives.WriteUInt64LittleEndian(entry[8..], 1);
            BinaryPrimitives.WriteUInt64LittleEndian(entry[16..], gpuAddress);
        }

        var count = trailingRanges + 1;
        if (indexed)
        {
            var entries = new List<(ulong FirstPage, ulong PageCount, ulong Base)>();
            for (var index = 0; index < count; index++)
            {
                entries.Add((BinaryPrimitives.ReadUInt64LittleEndian(table.AsSpan(index * 24)), 0, 0));
            }

            var words = SharpEmu.Gpu.Metal.V2.MetalDeviceAddressIndex.Build(entries);
            Array.Resize(ref table, table.Length + words.Length * sizeof(uint));
            MemoryMarshal.AsBytes(words.AsSpan()).CopyTo(table.AsSpan((int)count * 24));
            count |= SharpEmu.Gpu.Metal.V2.MetalDeviceAddressIndex.IndexedFlag;
        }

        var ranges = SharedBuffer(table);
        var faults = SharedBuffer(new byte[faultWords * sizeof(uint)]);
        return (
            [
                new DeviceTable(DescriptorBindingKind.DeviceAddressPageTable, ranges, count, [memory]),
                new DeviceTable(DescriptorBindingKind.FaultBuffer, faults, faultWords, []),
            ],
            faults);
    }

    /// <summary>A shared buffer holding <paramref name="contents"/> bound for one table kind
    /// (the global data share, say) with <paramref name="count"/> in its word field.</summary>
    public static DeviceTable CreateTable(DescriptorBindingKind kind, byte[] contents, uint count) =>
        new(kind, SharedBuffer(contents), count, []);

    public static uint[] ReadWords(nint buffer, int count)
    {
        var contents = Send(buffer, Selector("contents"));
        var words = new int[count];
        Marshal.Copy(contents, words, 0, count);
        return Array.ConvertAll(words, static word => unchecked((uint)word));
    }
    [LibraryImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
    private static partial void SendCopyBindingWords(nint encoder, nint selector, nint source, nuint sourceOffset,
        nint destination, nuint destinationOffset, nuint length);

    public static void CopyBindingWords(nint commandBuffer, nint memory, nuint source, nuint destination, nuint length)
    {
        var blit = Send(commandBuffer, Selector("blitCommandEncoder"));
        SendCopyBindingWords(blit, Selector("copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:"),
            memory, source, memory, destination, length);
        SendVoid(blit, Selector("endEncoding"));
    }

    public static void CompleteBindingWriter(Action<nint> encode)
    {
        var command = Send(SharedQueue.Value, Selector("commandBuffer"));
        encode(command);
        SendVoid(command, Selector("commit"));
        SendVoid(command, Selector("waitUntilCompleted"));
    }

}
