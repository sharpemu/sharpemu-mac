// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests.Resources;

public sealed class BatchedResourceWalkTests
{
    private static ShaderResourcePlan Plan(int count, bool nested, ulong hash = Hash)
    {
        var instructions = new List<Gen5ShaderInstruction>();
        uint pc = 0;
        if (nested) { instructions.Add(ScalarLoad(pc, 0, 10, 2)); pc += 8; }
        instructions.Add(ScalarLoad(pc, nested ? 10u : 0u, 4, 4, 2048)); pc += 8;
        for (var index = 0; index < count; index++)
        {
            instructions.Add(ScalarLoad(pc, nested ? 10u : 0u, 16, immediateOffset: 64 + index * 4)); pc += 8;
            instructions.Add(MoveVectorFromScalar(pc, 4, 16)); pc += 4;
            instructions.Add(BufferStore(pc, 4, index * 4)); pc += 8;
        }
        instructions.Add(EndProgram(pc));
        return ShaderResourcePlan.Extract(Program([.. instructions]), ShaderStage.Compute, hash, 0, 2);
    }

    private sealed class Memory
    {
        internal readonly List<string> Trace = [];
        internal ulong Failure = ulong.MaxValue;
        internal uint Salt, Pointer = 0x2000, Stride = 4u << 16, Format = 20u << 12;
        internal bool RefuseSpan, RefuseClean, InTable;
        internal int SpanCalls;
        internal uint Word(ulong address) => address switch
        {
            0x1000 => Pointer, 0x1004 => 0,
            0x1800 or 0x2800 or 0x3800 => 0x5000 + Salt * 0x1000,
            0x1804 or 0x2804 or 0x3804 => Stride,
            0x1808 or 0x2808 or 0x3808 => 256,
            0x180C or 0x280C or 0x380C => Format,
            _ => (uint)address + Salt,
        };
        private bool Read(ulong address, bool clean, out uint word)
        {
            Trace.Add($"R:{address}:{clean}:{InTable}"); word = Word(address);
            return address != Failure && !(clean && RefuseClean);
        }
        internal bool ReadGeneral(ulong address, out uint word) => Read(address, false, out word);
        private bool ReadClean(ulong address, out uint word) => Read(address, true, out word);
        internal bool Span(ulong address, Span<byte> bytes, bool clean)
        {
            SpanCalls++;
            if (RefuseSpan || (clean && RefuseClean) || (Failure >= address && Failure - address < (ulong)bytes.Length)) return false;
            for (var offset = 0; offset < bytes.Length; offset += 4)
            {
                Read(address + (ulong)offset, clean, out var word);
                BitConverter.TryWriteBytes(bytes[offset..], word);
            }
            return true;
        }
        internal ResourceRuntimeInputs Inputs(bool resident) => new()
        {
            UserData = [0x1000, 0], ReadMemory = ReadGeneral, ReadCleanMemory = ReadClean,
            ReadResidentMemory = resident ? Span : null, TablePhase = state => InTable = state,
        };
    }

    private static bool Reference(ShaderResourcePlan plan, ResourceRuntimeInputs inputs,
        out List<DescriptorWords> descriptors, out uint[] table) => RuntimeValueEvaluator.EvaluateSources(
            plan, plan.MaterializationSources.ToArray(), inputs, plan.CleanFlatSlots, true, out descriptors, out table);

    [Theory]
    [InlineData(4, false, false)]
    [InlineData(32, true, false)]
    [InlineData(128, true, true)]
    public void FreshGroupedWalkMatchesReferenceReadOrderAndKeepsPriorOutputs(int count, bool nested, bool clean)
    {
        var plan = Plan(count, nested);
        if (clean) Array.Fill((byte[])plan.CleanFlatSlots, (byte)1);
        var walk = Assert.IsType<BatchedResourceWalk>(BatchedResourceWalk.Build(plan));
        var memory = new Memory();
        Assert.True(Reference(plan, memory.Inputs(false), out var expectedDescriptors, out var expectedTable));
        var trace = memory.Trace.ToArray(); memory.Trace.Clear();
        Assert.True(walk.Evaluate(memory.Inputs(true), 3, out var descriptors, out var table));
        Assert.Equal(trace, memory.Trace);
        Assert.Equal(expectedTable, table[..^3]);
        Assert.Equal(new uint[3], table[^3..]);
        Assert.Equal(expectedDescriptors.SelectMany(value => value.Dwords), descriptors.SelectMany(value => value.Dwords));
        Assert.True(memory.SpanCalls > 0);
        var savedTable = table.ToArray();
        var savedDescriptors = descriptors.SelectMany(value => value.Dwords).ToArray();
        memory.Salt++; memory.Pointer = 0x3000;
        Assert.True(Reference(plan, memory.Inputs(false), out expectedDescriptors, out expectedTable));
        Assert.True(walk.Evaluate(memory.Inputs(true), 0, out var changedDescriptors, out var changedTable));
        Assert.Equal(expectedTable, changedTable);
        Assert.Equal(expectedDescriptors.SelectMany(value => value.Dwords), changedDescriptors.SelectMany(value => value.Dwords));
        Assert.Equal(savedTable, table);
        Assert.Equal(savedDescriptors, descriptors.SelectMany(value => value.Dwords));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RefusedSpansRetainScalarFailureOrderAndPublishNothing(bool clean)
    {
        var plan = Plan(8, true);
        if (clean) Array.Fill((byte[])plan.CleanFlatSlots, (byte)1);
        var walk = Assert.IsType<BatchedResourceWalk>(BatchedResourceWalk.Build(plan));
        var memory = new Memory { RefuseSpan = true };
        Assert.True(Reference(plan, memory.Inputs(false), out _, out _));
        var addresses = memory.Trace.Select(line => ulong.Parse(line.Split(':')[1])).Distinct().ToArray();
        foreach (var address in addresses)
        {
            memory.Failure = address; memory.Trace.Clear();
            Assert.False(Reference(plan, memory.Inputs(false), out _, out _));
            var expectedTrace = memory.Trace.ToArray(); memory.Trace.Clear();
            Assert.False(walk.Evaluate(memory.Inputs(true), 0, out var descriptors, out var table));
            Assert.Empty(descriptors); Assert.Empty(table);
            Assert.False(memory.InTable);
            Assert.Equal(expectedTrace, memory.Trace);
        }
        memory.Failure = ulong.MaxValue; memory.RefuseClean = clean;
        Assert.Equal(!clean, walk.Evaluate(memory.Inputs(true), 0, out _, out _));
    }

    [Fact]
    public void PendingRangeUsesPreciseProbeBeforeEachScalarRead()
    {
        const ulong hash = 0xB3F0_1100_0000_0001;
        var plan = Plan(4, false, hash);
        var walk = Assert.IsType<BatchedResourceWalk>(BatchedResourceWalk.Build(plan));
        var memory = new Memory();
        var probes = new List<ulong>();
        var inputs = new ResourceRuntimeInputs
        {
            UserData = [0x1000, 0], ReadMemory = memory.ReadGeneral, ReadResidentMemory = memory.Span,
            DataRangeWritePending = (_, _) => true,
            DataWritePending = address =>
            {
                probes.Add(address);
                var dataAddresses = new ulong[] { 0x1040, 0x1044, 0x1048, 0x104C };
                Assert.Equal(probes.Count - 1, memory.Trace.Count(line => dataAddresses.Contains(ulong.Parse(line.Split(':')[1]))));
                Assert.DoesNotContain(memory.Trace, line => line.StartsWith($"R:{address}:"));
                return true;
            },
        };
        Assert.True(walk.Evaluate(inputs, 0, out _, out _));
        Assert.Equal(plan.DeviceDataSlots.Count, probes.Count);
        Assert.All(plan.DeviceDataSlots.Values.SelectMany(indices => indices), index => Assert.True(DeviceDataLoads.Contains(hash, index)));
    }

    [Fact]
    public void DisjointRangeEliminatesPerWordProbesAndStillReadsFreshWords()
    {
        var plan = Plan(32, false, 0xB3F0_1100_0000_0002);
        var walk = Assert.IsType<BatchedResourceWalk>(BatchedResourceWalk.Build(plan));
        var memory = new Memory();
        var inputs = new ResourceRuntimeInputs
        {
            UserData = [0x1000, 0], ReadMemory = memory.ReadGeneral, ReadResidentMemory = memory.Span,
            DataRangeWritePending = (_, _) => false,
            DataWritePending = _ => throw new InvalidOperationException("range proves no pending word"),
        };
        Assert.True(walk.Evaluate(inputs, 0, out _, out var first));
        Assert.True(memory.SpanCalls > 0);
        memory.Salt++;
        Assert.True(walk.Evaluate(inputs, 0, out _, out var second));
        Assert.NotEqual(first, second);
    }

    [Fact]
    public void DescriptorPayloadAndSpecializationChangesPassThroughTheCache()
    {
        var plan = Plan(4, false);
        var memory = new Memory();
        var cache = new ResourceMaterializationCache();
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        for (var pass = 0; pass < 4; pass++)
        {
            memory.Salt++;
            memory.Stride = (uint)(pass + 1) * 4 << 16;
            memory.Format = (pass % 2 == 0 ? 20u : 77u) << 12;
            Assert.True(cache.Materialize(plan, memory.Inputs(true), memory.Span, ref snapshot, ref specialization, out _));
            var reference = new ResourceSnapshot();
            var referenceSpecialization = new ResourceSpecialization();
            Assert.True(ResourceMaterializer.Materialize(plan, memory.Inputs(false), ref reference, ref referenceSpecialization));
            Assert.Equal(reference.Buffers.SelectMany(words => words), snapshot.Buffers.SelectMany(words => words));
            Assert.Equal(reference.FlattenedResourceTable, snapshot.FlattenedResourceTable);
            Assert.Equal(reference.DeviceAddressRanges, snapshot.DeviceAddressRanges);
            Assert.Equal(referenceSpecialization, specialization);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void MissingAbsoluteAndNullAddressesKeepReferenceZeroSemantics(bool absolute)
    {
        var plan = Extract(Program(MoveScalar(0, 4, absolute ? 0xE0040000u : 0), MoveScalar(8, 5, absolute ? 0xFu : 0),
            ScalarLoad(16, 4, 8, 4), BufferLoad(24, 8), EndProgram(32)));
        var walk = Assert.IsType<BatchedResourceWalk>(BatchedResourceWalk.Build(plan));
        var inputs = Inputs([], (ulong _, out uint word) => { word = 0xDEADBEEF; return false; });
        Assert.True(Reference(plan, inputs, out var expectedDescriptors, out var expectedTable));
        Assert.True(walk.Evaluate(inputs, 0, out var descriptors, out var table));
        Assert.Equal(expectedTable, table);
        Assert.Equal(expectedDescriptors.SelectMany(value => value.Dwords), descriptors.SelectMany(value => value.Dwords));
    }
}
