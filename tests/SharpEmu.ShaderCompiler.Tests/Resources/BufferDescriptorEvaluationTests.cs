// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests.Resources;

public sealed class BufferDescriptorEvaluationTests(ITestOutputHelper output)
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SubsetReadsMatchMaterializationAcrossChangingBranchActivity(bool cleanReader)
    {
        var program = Program(
            Sopc(0, "SCmpEqU32", Gen5Operand.Scalar(8), Operand(1)),
            Branch(4, "SCbranchScc1", 3), BufferLoad(8, 0), Branch(16, "SBranch", 2),
            BufferLoad(20, 4), EndProgram(28));
        var plan = Extract(program);
        Assert.NotEmpty(plan.ResourceBranches);
        uint[] data = [0x1000, 0, 256, 0, 0x2000, 0, 256, 0, 0];
        bool Read(ulong address, out uint word) { word = 0; return false; }
        var inputs = Inputs(data, readCleanMemory: cleanReader ? Read : null);
        uint[] sources = plan.Info.Buffers.Select(buffer => buffer.Source).ToArray();
        Span<uint> words = stackalloc uint[sources.Length * 4];
        RuntimeValueEvaluator.BufferDescriptorEvaluationState? state = null;
        foreach (var condition in new uint[] { 0, 1, 0 })
        {
            data[8] = condition;
            inputs.UserData = data.ToArray();
            Assert.True(RuntimeValueEvaluator.EvaluateBufferDescriptorWords(plan, sources, inputs, words, ref state));
            var snapshot = new ResourceSnapshot();
            var specialization = new ResourceSpecialization();
            Assert.True(ResourceMaterializer.Materialize(plan, inputs, ref snapshot, ref specialization));
            for (var index = 0; index < snapshot.Buffers.Length; index++)
                Assert.True(words.Slice(index * 4, 4).SequenceEqual(snapshot.Buffers[index]));
        }
    }

    [Fact]
    public void SubsetRereadsGuestTableDescriptorWordsAndPointers()
    {
        var plan = Extract(Program(ScalarLoad(0, 0, 8, 4), BufferLoad(8, 8), EndProgram(16)));
        uint[] data = [0x1000, 0];
        var table = new Dictionary<ulong, uint>
        {
            [0x1000] = 0x2000, [0x1004] = 16u << 16, [0x1008] = 64, [0x100C] = 77u << 12 | 0xFAC,
            [0x1100] = 0x3000, [0x1104] = 32u << 16, [0x1108] = 32, [0x110C] = 75u << 12 | 0xFAC,
        };
        bool Read(ulong address, out uint word) => table.TryGetValue(address, out word);
        var inputs = Inputs(data, Read, Read);
        uint[] sources = [Assert.Single(plan.Info.Buffers).Source];
        Span<uint> words = stackalloc uint[4];
        RuntimeValueEvaluator.BufferDescriptorEvaluationState? state = null;
        Assert.True(RuntimeValueEvaluator.EvaluateBufferDescriptorWords(plan, sources, inputs, words, ref state));
        Assert.Equal(0x2000u, words[0]);
        Assert.NotNull(state);
        Assert.NotSame(inputs, state.Inputs);
        Assert.Empty(state.Inputs.UserData);
        Assert.Empty(state.CleanInputs.UserData);
        table[0x1000] = 0x4000;
        table[0x1008] = 7;
        Assert.True(RuntimeValueEvaluator.EvaluateBufferDescriptorWords(plan, sources, inputs, words, ref state));
        Assert.Equal(0x4000u, words[0]);
        Assert.Equal(7u, words[2]);
        data[0] = 0x1100;
        Assert.True(RuntimeValueEvaluator.EvaluateBufferDescriptorWords(plan, sources, inputs, words, ref state));
        Assert.Equal(0x3000u, words[0]);
        Assert.Equal(32u, words[2]);
        table.Remove(0x1100);
        Assert.False(RuntimeValueEvaluator.EvaluateBufferDescriptorWords(plan, sources, inputs, words, ref state));
        table[0x1100] = 0x5000;
        Assert.True(RuntimeValueEvaluator.EvaluateBufferDescriptorWords(plan, sources, inputs, words, ref state));
        Assert.Equal(0x5000u, words[0]);

        // Warm compiled evaluation and storage, then prove that retaining the
        // evaluator doesn't retain any draw's values or allocate on every read.
        plan.CompileEvaluatorNow();
        for (var iteration = 0; iteration < 100; iteration++)
            Assert.True(RuntimeValueEvaluator.EvaluateBufferDescriptorWords(plan, sources, inputs, words, ref state));
        _ = Stopwatch.GetTimestamp();
        var allocated = GC.GetAllocatedBytesForCurrentThread();
        var start = Stopwatch.GetTimestamp();
        for (var iteration = 0; iteration < 1000; iteration++)
        {
            table[0x1100] = (uint)iteration;
            if (!RuntimeValueEvaluator.EvaluateBufferDescriptorWords(plan, sources, inputs, words, ref state) || words[0] != iteration)
                throw new InvalidOperationException("The reused evaluator retained a stale descriptor word.");
        }
        var elapsed = Stopwatch.GetElapsedTime(start);
        allocated = GC.GetAllocatedBytesForCurrentThread() - allocated;
        Assert.Equal(0, allocated);
        output.WriteLine($"fresh guest-table descriptor n=1000 ns/call={elapsed.TotalNanoseconds / 1000:F1} bytes/call={(double)allocated / 1000:F1}");
    }
}
