// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Tests.Gpu.Scheduling;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.Pipelines;

[Collection(SchedulingStateCollection.Name)]
public sealed class PreparedProgramReplayTests
{
    [Fact]
    public void ReplayedSpecializationBindingsAndPatchedScalarsEqualTheReferencePath()
    {
        var guest = new PipelineTestGuest(); var host = new ReplayHost(guest);
        var cache = new ShaderProgramCache(guest.Context, guest.Compiler, host);
        const ulong code = PipelineTestGuest.MemoryBase + 0x1000, header = PipelineTestGuest.MemoryBase + 0x8000;
        guest.RegisterProgram(code, header, PipelineTestGuest.FormatLoadProgram);
        var options = new StageCompileOptions { ComputeInfo = new ComputeInputInfo { ThreadsX = 64, ThreadsY = 1, ThreadsZ = 1 } };
        ShaderProgram Bind(uint address, out ShaderStageResources stage)
        {
            var source = guest.Source(code, ShaderStage.Compute, PipelineTestGuest.BufferDescriptor(address, 16, 8, 77));
            var cursor = 0u;
            return cache.GetOrCompile(source, options, ref cursor, out stage);
        }
        var initial = Bind(0x10000, out var first);
        var calls = cache.Materializations!.FullPathCalls + cache.Materializations.BypassedCalls + cache.Materializations.PatchedCalls;
        Assert.Equal(initial, Bind(0x10000, out var exact));
        Assert.Equal(initial, Bind(0x18000, out var patched));
        Assert.Equal(1, host.Exact); Assert.Equal(1, host.Patched);
        Assert.Equal(calls, cache.Materializations.FullPathCalls + cache.Materializations.BypassedCalls + cache.Materializations.PatchedCalls);
        Assert.Same(first.Program, exact.Program); Assert.Same(first.Program, patched.Program);
        host.Enabled = false;
        Assert.Equal(initial, Bind(0x18000, out var reference));
        Assert.Equal(reference.Resources.UserData, patched.Resources.UserData);
        Assert.Equal(reference.Resources.Buffers.SelectMany(words => words), patched.Resources.Buffers.SelectMany(words => words));
        Assert.Equal(reference.Program!.Bindings, patched.Program!.Bindings);
        Assert.Single(guest.Compiler.Requests);
        host.Enabled = true; host.Pending = true;
        Bind(0x18000, out _);
        Assert.Equal(1, host.Patched); // A pending host publication is a full-path refusal, even with no descriptor reads.
    }

    [Fact]
    public void ContentIdentityStaticOptionsAndSemanticDescriptorsCannotReuseOldSpecialization()
    {
        var guest = new PipelineTestGuest(); var host = new ReplayHost(guest);
        var cache = new ShaderProgramCache(guest.Context, guest.Compiler, host);
        const ulong code = PipelineTestGuest.MemoryBase + 0x1000, header = PipelineTestGuest.MemoryBase + 0x8000;
        guest.RegisterProgram(code, header, PipelineTestGuest.FormatLoadProgram);
        var words = PipelineTestGuest.BufferDescriptor(0x10000, 16, 8, 77);
        var source = guest.Source(code, ShaderStage.Compute, words);
        ShaderProgram Bind(ShaderSource input, uint threads)
        {
            var cursor = 0u;
            return cache.GetOrCompile(input, new() { ComputeInfo = new() { ThreadsX = threads, ThreadsY = 1, ThreadsZ = 1 } }, ref cursor, out _);
        }
        var a = Bind(source, 64);
        Assert.NotEqual(a, Bind(source with { ContentHash = source.Identity + 1 }, 64));
        Assert.NotEqual(a, Bind(source, 32));
        words = PipelineTestGuest.BufferDescriptor(0x10000, 32, 8, 77);
        Bind(source with { UserData = words }, 64);
        Assert.Equal(0, host.Exact + host.Patched);
    }

    private sealed class ReplayHost(PipelineTestGuest guest) : IShaderPipelineHost, IResourceDependencySource, IRecipeReplaySink
    {
        internal bool Enabled = true, Pending;
        internal int Exact, Patched;
        public IRecipeReplaySink? RecipeReplay => Enabled ? this : null;
        public ResourceDependencyEpoch DependencyEpoch => default;
        public bool DependencyReadsReady => !Pending;
        public bool UsesDependencyByteValidation => true;
        public bool TryCaptureDependency(ulong address, ReadOnlySpan<byte> expected, bool clean, out ResourceDependency? dependency)
        { dependency = null; return false; }
        public bool TryValidateDependencyBytes(ReadOnlySpan<ulong> addresses, ReadOnlySpan<int> lengths, ReadOnlySpan<int> offsets, ReadOnlySpan<byte> expected)
        {
            if (Pending) return false;
            for (var i = 0; i < addresses.Length; i++)
            {
                Span<byte> actual = stackalloc byte[lengths[i]];
                if (!guest.Memory.TryRead(addresses[i], actual) || !actual.SequenceEqual(expected.Slice(offsets[i], lengths[i]))) return false;
            }
            return true;
        }
        public void ProgramReplay(bool hit, bool relocated, ResourceDependencyStatus reason, long ticks, long saved)
        { if (hit) { if (relocated) Patched++; else Exact++; } }
        public void PipelineReplay(bool hit) { }
        public uint MaxPushDescriptors => guest.Host.MaxPushDescriptors;
        public bool ComputeWave64Supported => guest.Host.ComputeWave64Supported;
        public bool GraphicsSubgroupOperationsEnabled => guest.Host.GraphicsSubgroupOperationsEnabled;
        public bool SharedInt64AtomicsEnabled => guest.Host.SharedInt64AtomicsEnabled;
        public RenderHostLimits Limits => guest.Host.Limits;
        public GpuSampleCount NoAttachmentSampleCounts => guest.Host.NoAttachmentSampleCounts;
        public bool TryResolveColorOutput(uint a, uint b, uint c, out Gen5PixelOutputKind kind, out Gen5ColorComponentMapping map) => guest.Host.TryResolveColorOutput(a, b, c, out kind, out map);
        public bool TryReadGuestWord(ulong address, out uint word) => guest.Host.TryReadGuestWord(address, out word);
        public bool TryReadCleanGuestWord(ulong address, out uint word) => guest.Host.TryReadCleanGuestWord(address, out word);
        public ulong CreateShaderModule(IGuestCompiledShader shader, ShaderStage stage, ulong hash, ulong programId) => guest.Host.CreateShaderModule(shader, stage, hash, programId);
        public PipelineHandle CreateGraphicsPipeline(GraphicsPipelineDescription description) => guest.Host.CreateGraphicsPipeline(description);
        public PipelineHandle CreateComputePipeline(ComputePipelineDescription description) => guest.Host.CreateComputePipeline(description);
    }
}
