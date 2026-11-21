// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Gpu.Metal.V2;

// The call records of a program linked with run-time record arguments: the port of the Vulkan presenter's preparation
// (VulkanVideoPresenter.CallRecords.cs and .CallRecordSpans.cs). For every record of a dispatch's tables, the image and
// sampler fields its callee reads through the record's argument get a heap element (the host's Element callback, from
// BindlessHeap.CallRecordFirstIndex on), and the words table the shader's call entries search is written
// (BindlessHeap.CallRecordKeyWords). The memory a callee reads through its record, its own reads and its buffers, comes back
// as device-address ranges the host orders and flushes as a planned range. Guest memory is read through the callbacks, so
// the preparation runs without a device in tests.
//
// Call record spans (upstream's SHARPEMU_CALL_RECORD_SPANS, on by default; "0" or any value but "1" turns them off): each
// record is read once over the span of its callee's fields and V#s, with the per-field preparation's rows, elements, words
// and merged ranges.
internal sealed class MetalCallRecordPreparation(
    MetalCallRecordPreparation.GuestRead tryRead,
    Func<ulong, ulong, bool> canRead,
    MetalCallRecordPreparation.ElementOf element)
{
    internal delegate bool GuestRead(ulong address, Span<byte> destination);

    // The heap element of a field whose descriptor words were read: eight words, a sampler's last four zero.
    internal delegate uint ElementOf(CallRecordField field, ReadOnlySpan<uint> words);

    internal static readonly bool ConfiguredSpans =
        Environment.GetEnvironmentVariable("SHARPEMU_CALL_RECORD_SPANS") is null or "" or "1";

    // Tests: the mode of this instance, whatever the configuration.
    internal bool Spans { get; init; } = ConfiguredSpans;

    // Rows past the shader's search depth and records left out because a buffer they write describes no buffer.
    internal long DroppedRows, LeftOutRecords;
    private int _reports;

    private void Warn(string message)
    {
        if (_reports++ < 8)
        {
            Console.Error.WriteLine(message);
        }
    }

    internal (uint[] Words, DeviceAddressRange[] Ranges) Prepare(ShaderProgramInfo program, ShaderStageResources stage) =>
        Spans ? PrepareSpanned(program, stage) : PrepareOriginal(program, stage);

    // The preparation without call record spans, which is also their reference.
    internal (uint[] Words, DeviceAddressRange[] Ranges) PrepareOriginal(ShaderProgramInfo program, ShaderStageResources stage)
    {
        var callees = program.CallRecordCallees;
        var targets = stage.CallTargets ?? [];
        var slotOf = new Dictionary<ulong, int>(targets.Length);
        for (var slot = 0; slot < targets.Length; slot++)
        {
            slotOf.TryAdd(targets[slot], slot);
        }

        var rows = new List<(uint Low, uint High, uint Callee, ulong Argument)>(stage.CallRecords?.Length ?? 0);
        foreach (var (target, argument) in stage.CallRecords ?? [])
        {
            if (slotOf.TryGetValue(target, out var slot) && slot < callees.Count &&
                !LeavesRecordOut(callees[slot], target, argument, program.Hash))
            {
                rows.Add(((uint)argument, (uint)(argument >> 32), (uint)slot, argument));
            }
        }

        rows.Sort(static (left, right) =>
            left.High != right.High ? left.High.CompareTo(right.High) :
            left.Low != right.Low ? left.Low.CompareTo(right.Low) : left.Callee.CompareTo(right.Callee));
        if (rows.Count > BindlessHeap.CallRecordMaximumRows)
        {
            DroppedRows += rows.Count - BindlessHeap.CallRecordMaximumRows;
            Warn($"[METAL-V2][WARN] A dispatch has more call records than the shader searches: records={rows.Count} " +
                $"searched={BindlessHeap.CallRecordMaximumRows} hash=0x{program.Hash:X16}.");
            rows.RemoveRange((int)BindlessHeap.CallRecordMaximumRows, rows.Count - (int)BindlessHeap.CallRecordMaximumRows);
        }

        var fieldWords = 0;
        foreach (var row in rows)
        {
            fieldWords += callees[(int)row.Callee].Fields.Length;
        }

        var keyWords = (int)BindlessHeap.CallRecordKeyWords;
        var rowStarts = 1 + rows.Count * keyWords;
        var words = new uint[Math.Max(rowStarts + rows.Count + fieldWords, (int)BindlessHeap.CallRecordMinimumWords)];
        words[0] = (uint)rows.Count;
        var cursor = rowStarts + rows.Count;
        var ranges = new List<DeviceAddressRange>();
        Span<uint> buffer = stackalloc uint[4];
        Span<uint> descriptor = stackalloc uint[8];
        for (var index = 0; index < rows.Count; index++)
        {
            var row = rows[index];
            words[1 + index * keyWords] = row.Low;
            words[2 + index * keyWords] = row.High;
            words[3 + index * keyWords] = row.Callee;
            words[rowStarts + index] = (uint)cursor;
            var callee = callees[(int)row.Callee];
            foreach (var field in callee.Fields)
            {
                descriptor.Clear();
                words[cursor++] = tryRead(row.Argument + field.Immediate, MemoryMarshal.AsBytes(descriptor[..(field.Sampler ? 4 : 8)]))
                    ? element(field, descriptor)
                    : BindlessHeap.CallRecordNullIndex;
            }

            if (callee.ArgumentBytes != 0)
            {
                ranges.Add(new DeviceAddressRange(uint.MaxValue, row.Argument, callee.ArgumentBytes, true, false));
            }

            foreach (var field in callee.Buffers)
            {
                if (!tryRead(row.Argument + field.Immediate, MemoryMarshal.AsBytes(buffer)))
                {
                    continue;
                }

                var address = buffer[0] | ((ulong)(buffer[1] & 0xFFFF) << 32);
                var stride = (buffer[1] >> 16) & 0x3FFF;
                var size = (ulong)buffer[2] * Math.Max(stride, 1u);
                if (CallRecordBufferPolicy.Resident(address, size, field.Written))
                {
                    ranges.Add(new DeviceAddressRange(uint.MaxValue, address, size, true, field.Written));
                }
            }
        }

        ranges.AddRange(stage.CallRecordRanges ?? []);
        return (words, MergeRanges(ranges));
    }

    // Whether a record is left out because a buffer its callee writes holds no buffer descriptor
    // (CallRecordBufferPolicy): its call returns at once instead of storing through it.
    private bool LeavesRecordOut(CallRecordCallee callee, ulong target, ulong argument, ulong hash)
    {
        Span<uint> buffer = stackalloc uint[4];
        foreach (var field in callee.Buffers)
        {
            if (!tryRead(argument + field.Immediate, MemoryMarshal.AsBytes(buffer)))
            {
                continue;
            }

            if (LeavesOut(buffer, field, target, argument, hash))
            {
                return true;
            }
        }

        return false;
    }

    private bool LeavesOut(ReadOnlySpan<uint> buffer, CallRecordBuffer field, ulong target, ulong argument, ulong hash)
    {
        var address = buffer[0] | ((ulong)(buffer[1] & 0xFFFF) << 32);
        var size = (ulong)buffer[2] * Math.Max((buffer[1] >> 16) & 0x3FFF, 1u);
        if (!CallRecordBufferPolicy.LeavesRecordOut(address, size, field.Written))
        {
            return false;
        }

        LeftOutRecords++;
        Warn($"[METAL-V2][WARN][LINKED_PROGRAM] A call record's buffer describes no buffer; its call returns at once: " +
            $"hash=0x{hash:X16} target=0x{target:X} argument=0x{argument:X} base=0x{address:X} size=0x{size:X}.");
        return true;
    }

    // Ranges that overlap or meet, or lie apart only by readable memory within a page, become one, so a dispatch makes few
    // ranges resident for many records.
    private DeviceAddressRange[] MergeRanges(List<DeviceAddressRange> ranges)
    {
        if (ranges.Count == 0)
        {
            return [];
        }

        ranges.Sort(static (left, right) => left.Written != right.Written ? left.Written.CompareTo(right.Written) : left.Base.CompareTo(right.Base));
        var merged = new List<DeviceAddressRange>(ranges.Count);
        var current = ranges[0];
        for (var index = 1; index < ranges.Count; index++)
        {
            var range = ranges[index];
            var end = current.Base + current.Size;
            if (range.Written == current.Written &&
                (range.Base <= end || (range.Base - end <= 4096 && canRead(end, range.Base - end))))
            {
                current = current with { Size = Math.Max(end, range.Base + range.Size) - current.Base };
                continue;
            }

            merged.Add(current);
            current = range;
        }

        merged.Add(current);
        return [.. merged];
    }

    // The bytes past a record's argument that its callee's fields (32 bytes for an image, 16 for a sampler) and buffer V#s
    // (16 bytes) occupy: from the lowest immediate, Length bytes. Zero length when it reads none, or when they lie more than
    // MaxSpan bytes apart; such records are read field by field.
    private readonly record struct Span(uint First, int Length);

    private const int MaxSpan = 512;

    private static readonly ConditionalWeakTable<CallRecordCallee, StrongBox<Span>> SpanTable = new();

    private static Span SpanOf(CallRecordCallee callee) => SpanTable.GetValue(callee, static callee =>
    {
        var first = ulong.MaxValue;
        var end = 0ul;
        foreach (var field in callee.Fields)
        {
            first = Math.Min(first, field.Immediate);
            end = Math.Max(end, (ulong)field.Immediate + (field.Sampler ? 16u : 32u));
        }

        foreach (var buffer in callee.Buffers)
        {
            first = Math.Min(first, buffer.Immediate);
            end = Math.Max(end, (ulong)buffer.Immediate + 16u);
        }

        return new StrongBox<Span>(first == ulong.MaxValue || end - first > MaxSpan
            ? default : new Span((uint)first, (int)(end - first)));
    }).Value;

    // A record the preparation keeps, with the offset of its span's bytes in the scratch (-1: read field by field).
    private readonly record struct Row(uint Low, uint High, uint Callee, ulong Argument, int Bytes, uint First);

    private readonly Dictionary<ulong, int> _slotOf = new();
    private readonly List<Row> _rows = new();
    private byte[] _bytes = new byte[4096];
    private int _byteCount;

    // PrepareOriginal with each record read once over its span.
    internal (uint[] Words, DeviceAddressRange[] Ranges) PrepareSpanned(ShaderProgramInfo program, ShaderStageResources stage)
    {
        var callees = program.CallRecordCallees;
        var targets = stage.CallTargets ?? [];
        _slotOf.Clear();
        for (var slot = 0; slot < targets.Length; slot++)
        {
            _slotOf.TryAdd(targets[slot], slot);
        }

        var rows = _rows;
        rows.Clear();
        _byteCount = 0;
        foreach (var (target, argument) in stage.CallRecords ?? [])
        {
            if (!_slotOf.TryGetValue(target, out var slot) || slot >= callees.Count)
            {
                continue;
            }

            var callee = callees[slot];
            var span = SpanOf(callee);
            var bytes = -1;
            if (span.Length != 0)
            {
                if (_bytes.Length - _byteCount < span.Length)
                {
                    Array.Resize(ref _bytes, Math.Max(_bytes.Length * 2, _byteCount + span.Length));
                }

                if (tryRead(argument + span.First, _bytes.AsSpan(_byteCount, span.Length)))
                {
                    bytes = _byteCount;
                    _byteCount += span.Length;
                }
            }

            var row = new Row((uint)argument, (uint)(argument >> 32), (uint)slot, argument, bytes, span.First);
            if (!LeavesRecordOut(callee, target, program.Hash, row))
            {
                rows.Add(row);
            }
        }

        rows.Sort(static (left, right) =>
            left.High != right.High ? left.High.CompareTo(right.High) :
            left.Low != right.Low ? left.Low.CompareTo(right.Low) : left.Callee.CompareTo(right.Callee));
        if (rows.Count > BindlessHeap.CallRecordMaximumRows)
        {
            DroppedRows += rows.Count - BindlessHeap.CallRecordMaximumRows;
            Warn($"[METAL-V2][WARN] A dispatch has more call records than the shader searches: records={rows.Count} " +
                $"searched={BindlessHeap.CallRecordMaximumRows} hash=0x{program.Hash:X16}.");
            rows.RemoveRange((int)BindlessHeap.CallRecordMaximumRows, rows.Count - (int)BindlessHeap.CallRecordMaximumRows);
        }

        var fieldWords = 0;
        foreach (var row in rows)
        {
            fieldWords += callees[(int)row.Callee].Fields.Length;
        }

        var keyWords = (int)BindlessHeap.CallRecordKeyWords;
        var rowStarts = 1 + rows.Count * keyWords;
        var words = new uint[Math.Max(rowStarts + rows.Count + fieldWords, (int)BindlessHeap.CallRecordMinimumWords)];
        words[0] = (uint)rows.Count;
        var cursor = rowStarts + rows.Count;
        var ranges = new List<DeviceAddressRange>();
        Span<uint> descriptor = stackalloc uint[8];
        var argumentBase = 0ul;
        var argumentEnd = 0ul;
        var argumentOpen = false;
        for (var index = 0; index < rows.Count; index++)
        {
            var row = rows[index];
            words[1 + index * keyWords] = row.Low;
            words[2 + index * keyWords] = row.High;
            words[3 + index * keyWords] = row.Callee;
            words[rowStarts + index] = (uint)cursor;
            var callee = callees[(int)row.Callee];
            foreach (var field in callee.Fields)
            {
                // As the per-field preparation: the words past a sampler's four stay zero in the key.
                descriptor.Clear();
                words[cursor++] = ReadRecordWords(row, field.Immediate, descriptor[..(field.Sampler ? 4 : 8)])
                    ? element(field, descriptor)
                    : BindlessHeap.CallRecordNullIndex;
            }

            if (callee.ArgumentBytes != 0)
            {
                // An argument range that overlaps or meets the open one (rows ascend by argument) joins it, as the merge
                // joins them; one whose end wraps is added on its own.
                var end = row.Argument + callee.ArgumentBytes;
                if (argumentOpen && row.Argument <= argumentEnd && end > row.Argument)
                {
                    argumentEnd = Math.Max(argumentEnd, end);
                }
                else
                {
                    if (argumentOpen)
                    {
                        ranges.Add(new DeviceAddressRange(uint.MaxValue, argumentBase, argumentEnd - argumentBase, true, false));
                        argumentOpen = false;
                    }

                    if (end > row.Argument)
                    {
                        (argumentBase, argumentEnd, argumentOpen) = (row.Argument, end, true);
                    }
                    else
                    {
                        ranges.Add(new DeviceAddressRange(uint.MaxValue, row.Argument, callee.ArgumentBytes, true, false));
                    }
                }
            }

            foreach (var field in callee.Buffers)
            {
                var buffer = descriptor[..4];
                if (!ReadRecordWords(row, field.Immediate, buffer))
                {
                    continue;
                }

                var address = buffer[0] | ((ulong)(buffer[1] & 0xFFFF) << 32);
                var stride = (buffer[1] >> 16) & 0x3FFF;
                var size = (ulong)buffer[2] * Math.Max(stride, 1u);
                if (CallRecordBufferPolicy.Resident(address, size, field.Written))
                {
                    ranges.Add(new DeviceAddressRange(uint.MaxValue, address, size, true, field.Written));
                }
            }
        }

        if (argumentOpen)
        {
            ranges.Add(new DeviceAddressRange(uint.MaxValue, argumentBase, argumentEnd - argumentBase, true, false));
        }

        ranges.AddRange(stage.CallRecordRanges ?? []);
        return (words, MergeRanges(ranges));
    }

    private bool LeavesRecordOut(CallRecordCallee callee, ulong target, ulong hash, in Row row)
    {
        Span<uint> buffer = stackalloc uint[4];
        foreach (var field in callee.Buffers)
        {
            if (ReadRecordWords(row, field.Immediate, buffer) && LeavesOut(buffer, field, target, row.Argument, hash))
            {
                return true;
            }
        }

        return false;
    }

    // The words at an immediate past a record's argument: from its span when it was read whole, else read now.
    private bool ReadRecordWords(in Row row, uint immediate, Span<uint> destination)
    {
        var target = MemoryMarshal.AsBytes(destination);
        if (row.Bytes < 0)
        {
            return tryRead(row.Argument + immediate, target);
        }

        _bytes.AsSpan(row.Bytes + (int)(immediate - row.First), target.Length).CopyTo(target);
        return true;
    }
}
