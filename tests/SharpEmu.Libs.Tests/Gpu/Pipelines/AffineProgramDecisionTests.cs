// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Tests.Gpu.Scheduling;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.Libs.Tests.Gpu.Pipelines;

[Collection(SchedulingStateCollection.Name)]
public sealed class AffineProgramDecisionTests(ITestOutputHelper output)
{
    private static readonly VertexFetchBounds Bounds = new(2, 4, 0, 1, true, true);
    private const ulong Address = PipelineTestGuest.MemoryBase + 0x40000;

    private static (AffineProgramDecision Decision, ShaderSource Source, ResourceRuntimeInputs Inputs, FakePipelineHost Host,
        ShaderResourcePlan Plan) Create(int biasScalar = -1)
    {
        var guest = new PipelineTestGuest();
        guest.RegisterProgram(PipelineTestGuest.MemoryBase + 0x1000, PipelineTestGuest.MemoryBase + 0x8000, PipelineTestGuest.EndProgram);
        var data = new uint[16];
        PipelineTestGuest.BufferDescriptor(Address, 16, 8, 77).CopyTo(data, 8);
        var source = guest.Source(PipelineTestGuest.MemoryBase + 0x1000, ShaderStage.Vertex, data);
        var program = Program(BufferAccess(0, "BufferLoadFormatXyzw", 8, dwords: 4, indexEnabled: true, vectorAddress: 5), EndProgram(8));
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Vertex, source.Hash, 0, 16, waveSize: 32);
        var entry = new ProgramSourceEntry
        {
            Plan = plan, Program = program, HasBitwiseExclusiveOr = false, HasShaderCalls = false,
            AffineFetches = [new(0, 4, new(VertexFetchIndex.Vertex, BiasScalarRegister: biasScalar, BiasMask: 0xFFFF), 0)],
        };
        return (AffineProgramDecision.Create(entry), source, new() { UserData = data }, guest.Host, plan);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void EveryDescriptorWordInvalidatesPositiveAndNegativeDecisions(int word)
    {
        var (decision, source, inputs, host, plan) = Create();
        VertexInputInfo Resolve()
        {
            Assert.True(RuntimeValueEvaluator.EvaluateBufferDescriptorWords(plan, decision.Sources, inputs, decision.FreshWords));
            var info = new VertexInputInfo { FetchBounds = Bounds };
            decision.Resolve(source, info, host, true);
            return info;
        }
        _ = VertexFetchCounters.TakeReport();
        var first = Resolve();
        Assert.Single(first.Attributes);
        Assert.Same(first.Attributes, Resolve().Attributes);
        var original = source.UserData[8 + word];
        source.UserData[8 + word] = word switch
        {
            0 => original + 16,
            1 => original ^ (1u << 31), // Unsupported swizzle, then recover.
            2 => 1, // Insufficient records, then recover.
            _ => 0, // Unsupported format/selectors, then recover.
        };
        var changed = Resolve();
        if (word == 0) Assert.Equal(Address + 16, Assert.Single(changed.Buffers).Address);
        else Assert.Empty(changed.Attributes);
        Assert.Same(changed.Attributes, Resolve().Attributes);
        source.UserData[8 + word] = original;
        Assert.Single(Resolve().Attributes);
        var report = VertexFetchCounters.TakeReport();
        Assert.Contains("affine_cache_hits=2 ", report);
        Assert.Contains("affine_cache_misses=3 ", report);
    }

    [Fact]
    public void BoundsBiasCapacityAndMappingArePartOfTheDecision()
    {
        var (decision, source, inputs, host, plan) = Create(biasScalar: 4);
        VertexInputInfo Resolve(VertexFetchBounds bounds)
        {
            Assert.True(RuntimeValueEvaluator.EvaluateBufferDescriptorWords(plan, decision.Sources, inputs, decision.FreshWords));
            var info = new VertexInputInfo { FetchBounds = bounds };
            decision.Resolve(source, info, host, true);
            return info;
        }
        var first = Resolve(Bounds);
        Assert.Same(first.Attributes, Resolve(Bounds).Attributes);
        Assert.Equal(1, host.VertexFetchMappingQueries);
        source.UserData[0] = 55; // Unrelated words cannot disturb the affine key.
        Assert.Same(first.Attributes, Resolve(Bounds).Attributes);
        source.UserData[4] = 1;
        Assert.Equal(Address + 16, Assert.Single(Resolve(Bounds).Buffers).Address);
        source.UserData[4] |= 0xFFFF0000; // Only the symbolic bias mask is consumed.
        Assert.Equal(Address + 16, Assert.Single(Resolve(Bounds).Buffers).Address);
        Assert.Equal(2, host.VertexFetchMappingQueries);
        Assert.Empty(Resolve(Bounds with { VertexCount = 8 }).Attributes);
        Assert.Empty(Resolve(default).Attributes);
        Assert.Single(Resolve(Bounds).Attributes);
        host.MaxFixedFunctionVertexBuffers = 0;
        Assert.Empty(Resolve(Bounds).Attributes);
        host.MaxFixedFunctionVertexBuffers = 24;
        Assert.Single(Resolve(Bounds).Attributes);
        host.VertexFetchMapped = false;
        host.VertexFetchMappingGeneration++;
        Assert.Empty(Resolve(Bounds).Attributes);
        var queries = host.VertexFetchMappingQueries;
        Assert.Empty(Resolve(Bounds).Attributes); // Negative mapping decisions are versioned too.
        Assert.Equal(queries, host.VertexFetchMappingQueries);
        host.VertexFetchMapped = true;
        host.VertexFetchMappingGeneration++;
        Assert.Single(Resolve(Bounds).Attributes);
        host.VertexFetchMappingGeneration = null;
        Assert.Single(Resolve(Bounds).Attributes);
        queries = host.VertexFetchMappingQueries;
        Assert.Single(Resolve(Bounds).Attributes);
        Assert.Equal(queries + 1, host.VertexFetchMappingQueries);
        VertexInputBuffer[] originalBuffers = [new(Address, 16, 8, Divisor: 7)];
        var fallbackInfo = new VertexInputInfo { Buffers = originalBuffers };
        decision.Resolve(source, fallbackInfo, host, true);
        Assert.Same(originalBuffers, fallbackInfo.Buffers);
        decision.Resolve(source, fallbackInfo, host, true);
        Assert.Same(originalBuffers, fallbackInfo.Buffers);
        Assert.Empty(fallbackInfo.Attributes);
    }

    [Fact]
    public void RepeatedDecisionTimingAndAllocation()
    {
        const int count = 20000;
        var (decision, source, inputs, host, plan) = Create();
        foreach (var bounds in new[] { Bounds, default(VertexFetchBounds) })
        {
            var info = new VertexInputInfo { FetchBounds = bounds };
            void Run(int iterations, bool cached)
            {
                for (var iteration = 0; iteration < iterations; iteration++)
                {
                    if (!RuntimeValueEvaluator.EvaluateBufferDescriptorWords(plan, decision.Sources, inputs, decision.FreshWords))
                        throw new InvalidOperationException("Descriptor read failed.");
                    decision.Resolve(source, info, host, cached);
                }
            }
            Run(2000, true);
            foreach (var cached in new[] { false, true })
            {
                Run(1000, cached);
                var allocated = GC.GetAllocatedBytesForCurrentThread();
                var start = Stopwatch.GetTimestamp();
                Run(count, cached);
                var elapsed = Stopwatch.GetElapsedTime(start);
                allocated = GC.GetAllocatedBytesForCurrentThread() - allocated;
                output.WriteLine($"affine decision bounds_known={bounds.VerticesKnown} cache={cached} n={count} ns/call={elapsed.TotalNanoseconds / count:F1} bytes/call={(double)allocated / count:F1}");
                if (cached) Assert.Equal(0, allocated);
            }
        }
    }
}
