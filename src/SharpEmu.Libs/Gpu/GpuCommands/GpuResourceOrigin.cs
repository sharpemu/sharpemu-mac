// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Libs.Gpu.GpuCommands;

// Historical packet provenance, never a cache dependency or an authorization to read a descriptor.
internal static class GpuResourceOrigin
{
    internal static readonly bool Enabled = Environment.GetEnvironmentVariable("SHARPEMU_GPU_RESOURCE_ORIGIN") == "1";
    private static readonly GpuResourceOriginRing? History = Enabled ? new(128) : null;
    internal const uint PointerLow = ShaderRegisterOffset.SpiShaderUserDataPs0 + 28;
    internal const uint PointerHigh = PointerLow + 1;

    internal static void Record(int queue, ulong submit, in PacketContext packet, bool table,
        ulong sourceAddress, uint offset, uint rawOffset, uint value, uint entry, uint count, RegisterBanks banks)
    {
        if (!Enabled || offset < PointerLow || offset > PointerHigh) return;
        History!.Add(Capture(queue, submit, in packet, table, sourceAddress, offset, rawOffset, value, entry, count, banks));
    }

    internal static GpuResourceOriginWrite Capture(int queue, ulong submit, in PacketContext packet, bool table,
        ulong sourceAddress, uint offset, uint rawOffset, uint value, uint entry, uint count, RegisterBanks banks)
    {
        var pixel = banks.Shader.Pixel;
        return new(0, queue, submit, packet.PacketAddress, packet.Header, table, sourceAddress,
            offset, rawOffset, value, entry, count, pixel.Address, pixel.UserScalars.Values[28], pixel.UserScalars.Values[29]);
    }

    internal static void ReportFailure(ShaderSource source, ResidentGuestBytesReader reader, TextWriter output)
    {
        if (!Enabled || source.Stage != ShaderStage.Pixel) return;
        var lowIndex = 28L - source.UserDataBase;
        var captured = lowIndex >= 0 && lowIndex + 1 < source.UserData.Length;
        var low = captured ? source.UserData[(int)lowIndex] : 0;
        var high = captured ? source.UserData[(int)lowIndex + 1] : 0;
        output.WriteLine($"[GPU][RESOURCE_ORIGIN] historical=1 not_current_proof=1 capacity=128 hash=0x{source.Hash:X16} " +
            $"shader=0x{source.Address:X16} captured_pointer_known={captured} captured_pointer=0x{((ulong)high << 32 | low):X16}");
        Span<byte> bytes = stackalloc byte[sizeof(uint)];
        foreach (var write in History!.Snapshot())
        {
            var readable = reader(write.SourceAddress, bytes, false);
            var current = readable ? BinaryPrimitives.ReadUInt32LittleEndian(bytes) : 0;
            var matches = captured && write.ShaderAddress == source.Address && write.PointerLow == low && write.PointerHigh == high;
            output.WriteLine($"[GPU][RESOURCE_ORIGIN] historical=1 seq={write.Sequence} queue={write.Queue} submit={write.Submit} " +
                $"packet=0x{write.PacketAddress:X16} header=0x{write.Header:X8} source={(write.Table ? "table" : "direct")} " +
                $"value_address=0x{write.SourceAddress:X16} offset=0x{write.Offset:X4} raw_offset=0x{write.RawOffset:X8} " +
                $"value=0x{write.Value:X8} entry={write.Entry} count={write.Count} shader=0x{write.ShaderAddress:X16} " +
                $"pointer=0x{((ulong)write.PointerHigh << 32 | write.PointerLow):X16} matches_captured_inputs={matches} " +
                $"reread_readable={readable} reread_value=0x{current:X8} reread_changed={(readable && current != write.Value)}");
        }
    }
}

internal readonly record struct GpuResourceOriginWrite(long Sequence, int Queue, ulong Submit,
    ulong PacketAddress, uint Header, bool Table, ulong SourceAddress, uint Offset, uint RawOffset,
    uint Value, uint Entry, uint Count, ulong ShaderAddress, uint PointerLow, uint PointerHigh);

// Pure bounded storage so wraparound and ordering can be checked without a GPU or process flags.
internal sealed class GpuResourceOriginRing
{
    private readonly object _gate = new();
    private readonly GpuResourceOriginWrite[] _records;
    private int _next;
    private int _count;
    private long _sequence;

    internal GpuResourceOriginRing(int capacity)
    {
        if (capacity <= 0) throw new ArgumentOutOfRangeException(nameof(capacity));
        _records = new GpuResourceOriginWrite[capacity];
    }

    internal void Add(in GpuResourceOriginWrite record)
    {
        lock (_gate)
        {
            _records[_next] = record with { Sequence = ++_sequence };
            _next = (_next + 1) % _records.Length;
            _count = Math.Min(_count + 1, _records.Length);
        }
    }

    internal GpuResourceOriginWrite[] Snapshot()
    {
        lock (_gate)
        {
            var result = new GpuResourceOriginWrite[_count];
            var first = (_next - _count + _records.Length) % _records.Length;
            for (var index = 0; index < result.Length; index++)
                result[index] = _records[(first + index) % _records.Length];
            return result;
        }
    }
}
