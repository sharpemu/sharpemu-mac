// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceDependencyBypassTests;
using Stream = SharpEmu.ShaderCompiler.Tests.Resources.ResourceDependencyBypassTests.Stream;

namespace SharpEmu.ShaderCompiler.Tests.Resources;

public sealed class ResourceReplayValidationTests
{
    [Fact]
    public void WarmValidationCostAndAllocationsAreReported()
    {
        var memory = TableMemory(); memory.ByteValidation = true;
        var stream = new Stream(TablePlan(), memory); stream.Cache.CaptureReplayValidation = true;
        var roots = Roots(); stream.Run(roots);
        var proof = Assert.IsType<ResourceReplayValidation>(stream.Cache.ReplayValidation);
        for (var i = 0; i < 10000; i++) Assert.True(proof.TryReplay(memory, roots, out _, out _, out _));
        var allocated = GC.GetAllocatedBytesForCurrentThread();
        var started = System.Diagnostics.Stopwatch.GetTimestamp();
        var hits = 0;
        for (var i = 0; i < 90460; i++) if (proof.TryReplay(memory, roots, out _, out _, out _)) hits++;
        var elapsed = System.Diagnostics.Stopwatch.GetElapsedTime(started);
        var bytes = GC.GetAllocatedBytesForCurrentThread() - allocated;
        Assert.Equal(90460, hits);
        Assert.Equal(0, bytes);
        // Calibration only: no timing threshold in the correctness suite.
        System.Console.WriteLine($"resource_replay_validation n={hits} us_per_stage={elapsed.TotalMicroseconds / hits:F3} bytes={bytes} includes_renderer=0");
    }
    [Fact]
    public void ExactAndPatchedReplayEqualFullMaterializationWithoutEnteringItsCache()
    {
        var memory = TableMemory(); memory.ByteValidation = true;
        var plan = TablePlan(); var stream = new Stream(plan, memory);
        stream.Cache.CaptureReplayValidation = true;
        stream.Run(Roots());
        var proof = Assert.IsType<ResourceReplayValidation>(stream.Cache.ReplayValidation);
        var before = stream.Cache.FullPathCalls + stream.Cache.BypassedCalls + stream.Cache.PatchedCalls;
        foreach (var address in new uint[] { 0x10000, 0x18000, 0x24000, 0x10000 })
        {
            var roots = Roots(cb: address);
            Assert.True(proof.TryReplay(memory, roots, out var snapshot, out var patched, out var status));
            Assert.Equal(ResourceDependencyStatus.Current, status);
            Assert.Equal(address != 0x10000, patched);
            var full = new ResourceSnapshot(); var specialization = new ResourceSpecialization();
            Assert.True(ResourceMaterializer.Materialize(plan, new() { UserData = roots, ReadMemory = memory.Read }, ref full, ref specialization));
            Assert.Equal(full.UserData, snapshot.UserData);
            Assert.Equal(full.FlattenedResourceTable, snapshot.FlattenedResourceTable);
            Assert.Equal(full.Buffers.SelectMany(words => words), snapshot.Buffers.SelectMany(words => words));
        }
        Assert.Equal(before, stream.Cache.FullPathCalls + stream.Cache.BypassedCalls + stream.Cache.PatchedCalls);
    }

    [Theory]
    [InlineData("cpu")]
    [InlineData("alias")]
    [InlineData("pending_gpu")]
    [InlineData("gpu_changed")]
    [InlineData("overlay")]
    [InlineData("publication")]
    [InlineData("mapping_aba")]
    [InlineData("policy")]
    [InlineData("source")]
    [InlineData("semantic_root")]
    public void EveryUnprovedInputRefuses(string change)
    {
        var memory = TableMemory(); memory.ByteValidation = true;
        var stream = new Stream(TablePlan(), memory); stream.Cache.CaptureReplayValidation = true; stream.Run(Roots());
        var proof = Assert.IsType<ResourceReplayValidation>(stream.Cache.ReplayValidation);
        IResourceDependencySource source = memory; var roots = Roots();
        switch (change)
        {
            case "cpu": memory.Write(0x1040, 12); break;
            case "alias": memory.Words[0x1040] = 12; break; // raw alias: no VA fault/generation
            case "pending_gpu": memory.Pending = true; memory.Epoch = memory.Epoch with { GpuWrites = 1 }; break;
            case "gpu_changed": memory.Words[0x1040] = 12; memory.Epoch = memory.Epoch with { GpuWrites = 1 }; break;
            case "overlay": memory.Pending = true; memory.Epoch = memory.Epoch with { Overlay = 1 }; break;
            case "publication": memory.PublicationsPending = true; break;
            case "mapping_aba": memory.Epoch = memory.Epoch with { Mapping = 2 }; break;
            case "policy": memory.Policy++; break;
            case "source": source = TableMemory(); break;
            case "semantic_root": roots[6]++; break;
        }
        Assert.False(proof.TryReplay(source, roots, out _, out _, out _));
    }

    [Fact]
    public void UnrelatedGlobalWritersRequireCurrentBytesAndDoNotPoisonTheRecipe()
    {
        var memory = TableMemory(); memory.ByteValidation = true;
        var stream = new Stream(TablePlan(), memory); stream.Cache.CaptureReplayValidation = true; stream.Run(Roots());
        var proof = Assert.IsType<ResourceReplayValidation>(stream.Cache.ReplayValidation);
        memory.Epoch = new(19, 22, 0, 16, 33);
        var validations = memory.ByteValidations;
        Assert.True(proof.TryReplay(memory, Roots(), out _, out _, out _));
        Assert.Equal(validations + 1, memory.ByteValidations);
        memory.Words[0x1040]++;
        Assert.False(proof.TryReplay(memory, Roots(), out _, out _, out _));
    }

    [Fact]
    public void PublicationDuringValidationCannotCertifyOldBytes()
    {
        var memory = TableMemory(); memory.ByteValidation = true;
        var source = new RacingSource(memory);
        var stream = new Stream(TablePlan(), memory); stream.Inputs.DependencySource = source;
        stream.Cache.CaptureReplayValidation = true;
        stream.Run(Roots());
        var proof = Assert.IsType<ResourceReplayValidation>(stream.Cache.ReplayValidation);
        source.Race = true;
        Assert.False(proof.TryReplay(source, Roots(), out _, out _, out _));
    }

    private sealed class RacingSource(Memory memory) : IResourceDependencySource
    {
        internal bool Race;
        public ResourceDependencyEpoch DependencyEpoch => memory.Epoch;
        public bool DependencyReadsReady => !memory.PublicationsPending;
        public bool UsesDependencyByteValidation => true;
        public bool TryCaptureDependency(ulong address, ReadOnlySpan<byte> expected, bool clean, out ResourceDependency? dependency)
        { dependency = null; return false; }
        public bool TryValidateDependencyBytes(ReadOnlySpan<ulong> addresses, ReadOnlySpan<int> lengths, ReadOnlySpan<int> offsets, ReadOnlySpan<byte> expected)
        {
            var equal = memory.TryValidateDependencyBytes(addresses, lengths, offsets, expected);
            if (Race) memory.Epoch = memory.Epoch with { Overlay = memory.Epoch.Overlay + 1 };
            return equal;
        }
    }
}
