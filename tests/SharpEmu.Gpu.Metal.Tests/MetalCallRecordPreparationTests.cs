// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

// The call record preparation of the Metal host (MetalCallRecordPreparation), without a device: the rows and words table a
// linked program's call entries search, the buffer policy (a record whose written buffer describes no buffer is left out;
// a buffer it describes is a resident range), and call record spans preparing what the per-field preparation prepares. The
// Metal twins of tests/SharpEmu.Libs.Tests/VideoOut/CallRecordPreparationTests.cs and CallRecordSpanTests.cs.
public class MetalCallRecordPreparationTests
{
    // Off: the per-field preparation. The derived class runs every case with call record spans.
    protected virtual bool Spans => false;

    private const ulong Target = 0x2000;
    private const ulong Argument = 0x1000;

    // Guest memory of one mapped region.
    internal sealed class Memory(ulong start, int size)
    {
        private readonly byte[] _bytes = new byte[size];

        public bool CanRead(ulong address, ulong length) =>
            address >= start && length <= (ulong)_bytes.Length && address - start <= (ulong)_bytes.Length - length;

        public bool TryRead(ulong address, Span<byte> destination)
        {
            if (!CanRead(address, (ulong)destination.Length)) return false;
            _bytes.AsSpan((int)(address - start), destination.Length).CopyTo(destination);
            return true;
        }

        public void Write(ulong address, ReadOnlySpan<byte> source)
        {
            Assert.True(CanRead(address, (ulong)source.Length));
            source.CopyTo(_bytes.AsSpan((int)(address - start)));
        }
    }

    internal sealed record Prepared(uint[] Words, DeviceAddressRange[] Ranges, List<string> Elements);

    // The elements stand in for the bindless heap (which needs a device): each call is listed with its slot and words and
    // gets the next number.
    internal static Prepared Prepare(bool spans, Memory memory, CallRecordCallee[] callees, ulong[] targets,
        (ulong Target, ulong Argument)[] records, DeviceAddressRange[]? stageRanges = null)
    {
        var elements = new List<string>();
        var preparation = new MetalCallRecordPreparation(memory.TryRead, memory.CanRead, (field, words) =>
        {
            elements.Add($"{field.Slot}:{string.Join(",", words.ToArray())}");
            return (uint)elements.Count;
        })
        { Spans = spans };
        var program = new ShaderProgramInfo { CallRecordCallees = callees };
        var stage = new ShaderStageResources(program, new ResourceSnapshot())
        {
            CallTargets = targets,
            CallRecords = records,
            CallRecordRanges = stageRanges,
        };
        var (words, ranges) = preparation.Prepare(program, stage);
        return new(words, ranges, elements);
    }

    private (uint[] Words, DeviceAddressRange[] Ranges) Prepare(Memory memory, CallRecordBuffer[] buffers,
        params (ulong Target, ulong Argument)[] records)
    {
        var prepared = Prepare(Spans, memory, [new CallRecordCallee([], buffers, 64)], [Target], records);
        return (prepared.Words, prepared.Ranges);
    }

    internal static void Descriptor(Memory memory, ulong at, ulong address, uint records, uint stride = 0)
    {
        Span<byte> bytes = stackalloc byte[16];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes, (uint)address);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes[4..], (uint)(address >> 32) | stride << 16);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes[8..], records);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes[12..], 0);
        memory.Write(at, bytes);
    }

    [Fact]
    public void ARecordWithoutBufferFieldsKeepsItsKeyAndArgumentRange()
    {
        var memory = new Memory(Argument, 0x2000);
        var result = Prepare(memory, [], (Target, Argument));
        Assert.Equal(1u, result.Words[0]);
        Assert.Equal((uint)Argument, result.Words[1]);
        Assert.Equal(0u, result.Words[2]);
        Assert.Equal(0u, result.Words[3]);
        Assert.Contains(result.Ranges, range => range.Base == Argument && range.Size == 64 && !range.Written);
    }

    [Theory]
    [InlineData(0x8000ul, 64u, false, true, true)]
    [InlineData(0x8000ul, 64u, true, true, true)]
    [InlineData(1ul << 40, 64u, false, true, false)]
    [InlineData(1ul << 40, 64u, true, false, false)]
    [InlineData(0ul, 64u, false, true, false)]
    [InlineData(0ul, 64u, true, true, false)]
    [InlineData(0x8000ul, 0u, false, true, false)]
    [InlineData(0x8000ul, 0u, true, true, false)]
    public void BufferPolicyControlsTheUploadedRowAndResidentRange(ulong address, uint size, bool written, bool keep, bool resident)
    {
        var memory = new Memory(Argument, 0x2000);
        Descriptor(memory, Argument, address, size);
        var result = Prepare(memory, [new CallRecordBuffer(0, written)], (Target, Argument));
        Assert.Equal(keep ? 1u : 0u, result.Words[0]);
        if (!keep)
        {
            Assert.Empty(result.Ranges);
            return;
        }

        Assert.Equal((uint)Argument, result.Words[1]);
        Assert.Contains(result.Ranges, range => range.Base == Argument && range.Size == 64 && !range.Written);
        Assert.Equal(resident, result.Ranges.Any(range => range.Base == address && range.Size == size && range.Written == written));
    }

    [Fact]
    public void AnInvalidWritableRecordDoesNotRemoveTheValidRecordBesideIt()
    {
        var memory = new Memory(Argument, 0x2000);
        Descriptor(memory, Argument, 0x8000, 64);
        Descriptor(memory, Argument + 0x100, 1ul << 40, 64);
        var result = Prepare(memory, [new CallRecordBuffer(0, true)], (Target, Argument + 0x100), (Target, Argument));
        Assert.Equal(1u, result.Words[0]);
        Assert.Equal((uint)Argument, result.Words[1]);
        Assert.Contains(result.Ranges, range => range.Base == 0x8000 && range.Written);
        Assert.DoesNotContain(result.Ranges, range => range.Base == Argument + 0x100);
    }

    [Fact]
    public void AnUnreadableFieldPreservesTheExistingKeepRecordPolicy()
    {
        var memory = new Memory(Argument, 0x2000);
        var result = Prepare(memory, [new CallRecordBuffer(0, true)], (Target, 0x5000));
        Assert.Equal(1u, result.Words[0]);
        Assert.Equal(0x5000u, result.Words[1]);
        Assert.Equal(0x5000ul, Assert.Single(result.Ranges).Base);
    }

    // Rows sort by argument (high, low) and callee; each row's first field index points at its fields' elements, which the
    // shader reads as call_records[call_records[1 + rows * 3 + row] + field].
    [Fact]
    public void RowsSortByArgumentAndCalleeAndNameTheirFieldElements()
    {
        var memory = new Memory(Argument, 0x2000);
        CallRecordCallee[] callees = [new([new CallRecordField(0, false, 0), new CallRecordField(32, true, 8)], [], 0), new([], [], 0)];
        var prepared = Prepare(Spans, memory, callees, [0x2000, 0x3000],
            [(0x2000, Argument + 0x40), (0x3000, Argument), (0x2000, Argument), (0x9000, Argument)]);
        Assert.Equal(3u, prepared.Words[0]);
        Assert.Equal(new uint[] { (uint)Argument, 0, 0, (uint)Argument, 0, 1, (uint)Argument + 0x40, 0, 0 }, prepared.Words[1..10]);
        var starts = prepared.Words[10..13];
        Assert.Equal(new uint[] { 1, 2 }, prepared.Words[(int)starts[0]..((int)starts[0] + 2)]);
        Assert.Equal(starts[0] + 2, starts[1]);
        Assert.Equal(new uint[] { 3, 4 }, prepared.Words[(int)starts[2]..((int)starts[2] + 2)]);
    }
}

public sealed class SpannedMetalCallRecordPreparationTests : MetalCallRecordPreparationTests
{
    protected override bool Spans => true;
}

// Call record spans: on the same memory the preparation hands the heap the same descriptor words in the same order, and
// writes the same words table and merged ranges, as the per-field preparation.
public sealed class MetalCallRecordSpanTests
{
    private const ulong Base = 0x10000;
    private const int Size = 0x10000;

    // Callees whose fields and V#s lie within one span, one whose span is too wide to read whole, one that reads nothing.
    private static readonly CallRecordCallee[] Callees =
    [
        new([new CallRecordField(0, false, 1), new CallRecordField(32, true, 7)], [new CallRecordBuffer(48, false)], 64),
        new([], [new CallRecordBuffer(0, true), new CallRecordBuffer(16, false)], 32),
        new([new CallRecordField(0x100, false, 2)], [new CallRecordBuffer(0, false)], 0),
        new([new CallRecordField(0, false, 3)], [new CallRecordBuffer(0x300, false)], 16),
        new([], [], 16),
    ];

    private static readonly ulong[] Targets = [0x2000, 0x3000, 0x4000, 0x5000, 0x6000];

    private static void Same(MetalCallRecordPreparationTests.Prepared expected, MetalCallRecordPreparationTests.Prepared actual)
    {
        Assert.Equal(expected.Words, actual.Words);
        Assert.Equal(expected.Ranges, actual.Ranges);
        Assert.Equal(expected.Elements, actual.Elements);
    }

    // Random memory with valid, null and invalid V#s, and records whose spans lie inside the region, cross its end or lie past
    // it, with unknown targets and repeated arguments.
    private static (MetalCallRecordPreparationTests.Memory Memory, (ulong, ulong)[] Records) RandomRecords(int seed, int count)
    {
        var random = new Random(seed);
        var memory = new MetalCallRecordPreparationTests.Memory(Base, Size);
        var noise = new byte[Size];
        random.NextBytes(noise);
        memory.Write(Base, noise);
        var records = new List<(ulong, ulong)>();
        for (var index = 0; index < count; index++)
        {
            var argument = Base + (ulong)random.Next(0, Size + 0x400) / 16 * 16;
            if (random.Next(4) == 0 && records.Count != 0) argument = records[random.Next(records.Count)].Item2;
            var target = random.Next(12) == 0 ? 0x9000ul : Targets[random.Next(Targets.Length)];
            if (random.Next(3) == 0)
            {
                foreach (var immediate in new uint[] { 0, 16, 48, 0x300 })
                {
                    if (argument + immediate + 16 <= Base + Size)
                        MetalCallRecordPreparationTests.Descriptor(memory, argument + immediate, 0x8000 + (ulong)random.Next(0x1000) * 16,
                            (uint)random.Next(1, 64), 16);
                }
            }
            else if (random.Next(3) == 0 && argument + 32 <= Base + Size)
            {
                MetalCallRecordPreparationTests.Descriptor(memory, argument, 0, 0);
                MetalCallRecordPreparationTests.Descriptor(memory, argument + 16, 1ul << 40, 64);
            }

            if (!records.Contains((target, argument))) records.Add((target, argument));
        }

        return (memory, [.. records]);
    }

    [Theory]
    [InlineData(1, 8)]
    [InlineData(2, 64)]
    [InlineData(3, 300)]
    [InlineData(4, 2000)]
    public void SpansPrepareWhatTheOriginalPrepares(int seed, int count)
    {
        var (memory, records) = RandomRecords(seed, count);
        DeviceAddressRange[] stageRanges = [new(3, 0x8000, 0x40, true, false), new(4, 0x9000, 0x20, true, true)];
        var original = MetalCallRecordPreparationTests.Prepare(false, memory, Callees, Targets, records, stageRanges);
        Same(original, MetalCallRecordPreparationTests.Prepare(true, memory, Callees, Targets, records, stageRanges));
        Assert.NotEmpty(original.Elements);
        Assert.True(original.Words[0] > 0);
    }

    // A record whose span crosses the end of mapped memory is read field by field: the fields inside still read, and the
    // ones past the end take the null element, as in the original.
    [Fact]
    public void ASpanPastMappedMemoryReadsFieldByField()
    {
        var memory = new MetalCallRecordPreparationTests.Memory(Base, Size);
        var argument = Base + Size - 40;
        MetalCallRecordPreparationTests.Descriptor(memory, argument + 16, 0x8000, 4, 16);
        var records = new[] { (Targets[1], argument), (Targets[0], argument) };
        var original = MetalCallRecordPreparationTests.Prepare(false, memory, Callees, Targets, records);
        var spanned = MetalCallRecordPreparationTests.Prepare(true, memory, Callees, Targets, records);
        Same(original, spanned);
        Assert.Contains(spanned.Ranges, range => range.Base == 0x8000 && range.Size == 64);
        Assert.Single(spanned.Elements);
        Assert.Equal(BindlessHeap.CallRecordNullIndex, spanned.Words[^1]);
    }

    [Fact]
    public void SpansAreOnUnlessConfigured()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_CALL_RECORD_SPANS") is not null) return;
        Assert.True(MetalCallRecordPreparation.ConfiguredSpans);
    }
}
