// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.IO.Hashing;
using System.Runtime.InteropServices;
using System.Text;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.HLE.Host;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Metal;
using SharpEmu.ShaderCompiler.Metal.Tests;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// A translated shader's MSL source is only held until its library is built: the render
/// host drops it, keeping the reflection data, and a later consumer gets the same text
/// back by translating the request again. Device work is behind the exclusive device gate.
/// </summary>
public sealed unsafe class MetalShaderSourceReleaseTests(ITestOutputHelper output)
{
    private const ulong GuestBase = 0x2_0000_0000;

    [Fact]
    public void AReleasedSourceKeepsTheReflectionAndTranslatesAgainToTheSameText()
    {
        var request = Gen5ComputeFixtures.CreateComputeRequest(Gen5ComputeFixtures.ExecStore);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var original, out var error), error);
        var metal = new MetalCompiledGuestShader(original, request);

        metal.ReleaseSource();

        Assert.True(metal.SourceReleased);
        Assert.Empty(metal.Shader.Source);
        Assert.Equal(original with { Source = string.Empty }, metal.Shader);
        Assert.Equal(original.Source, metal.AcquireShader().Source);
        Assert.Equal(Encoding.UTF8.GetBytes(original.Source), metal.Payload);
    }

    // The compute prewarm dedupes and matches modules by this identity; it is the same
    // before and after the text is dropped, and the same for two translations of one request.
    [Fact]
    public void ThePayloadIdentityIsTheHashOfThePayloadWhetherOrNotTheSourceIsHeld()
    {
        var request = Gen5ComputeFixtures.CreateComputeRequest(Gen5ComputeFixtures.ExecStore);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var first, out var error), error);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var second, out error), error);
        var expected = XxHash3.HashToUInt64(Encoding.UTF8.GetBytes(first.Source));
        var held = new MetalCompiledGuestShader(first, request);
        var released = new MetalCompiledGuestShader(second, request);

        released.ReleaseSource();

        Assert.Equal(expected, held.PayloadHash);
        Assert.Equal(expected, released.PayloadHash);
        Assert.Equal(MetalShaderPrewarm.Identity(held), MetalShaderPrewarm.Identity(released));
    }

    [Fact]
    public void AShaderWithoutItsRequestKeepsItsSource()
    {
        var shader = new Gen5MslShader("kernel void k() {}", "k", Gen5MslStage.Compute, AttributeCount: 0);
        var metal = new MetalCompiledGuestShader(shader);

        metal.ReleaseSource();

        Assert.False(metal.SourceReleased);
        Assert.Same(shader, metal.Shader);
    }

    // The rectangle-list form is built from the vertex program's text the first time a
    // rectangle-list pipeline uses it, which can be long after the text was dropped.
    [Fact]
    public void TheRectangleListFormOfAReleasedVertexProgramIsTheSame()
    {
        var memory = new FakeGuestMemory();
        memory.AddRegion(Gen5ComputeFixtures.ProgramAddress, Gen5ComputeFixtures.VertexWords);
        Assert.True(Gen5ShaderTranslator.TryDecodeProgram(
            new CpuContext(memory, Generation.Gen5), Gen5ComputeFixtures.ProgramAddress, out var program, out var decodeError), decodeError);
        var request = Gen5ComputeFixtures.RequestOrThrow(program!, ShaderStage.Vertex);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var original, out var error), error);
        var expected = Gen5MslRectangleList.BuildSource(original);
        var metal = new MetalCompiledGuestShader(original, request);

        metal.ReleaseSource();

        Assert.Equal(expected, Gen5MslRectangleList.BuildSource(metal.AcquireShader()));
    }

    [Fact]
    public void TheRenderHostDropsAModuleSourceOnceItsLibraryIsBuilt()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var request = Gen5ComputeFixtures.CreateComputeRequest(Gen5ComputeFixtures.ExecStore);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var original, out var error), error);
        var metal = new MetalCompiledGuestShader(original, request);

        var first = Pipeline(rig.Host, metal, 1);

        Assert.True(metal.SourceReleased);
        Assert.Empty(metal.Shader.Source);

        // A second module of the same shader builds from the text translated again.
        var second = Pipeline(rig.Host, metal, 2);
        Assert.NotEqual(first.Pipeline, second.Pipeline);
    }

    // A module the shader cache bound released is built again from the kept request alone:
    // AcquireShader checks the translation against the hash of the text it dropped, so a
    // rebuild that succeeds compiled the same MSL.
    [Fact]
    public void AModuleReleasedByTheShaderCacheBoundIsBuiltAgainFromItsRequest()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Host.ShaderCacheLimits = MetalShaderCacheLimits.Unbounded;
        var request = Gen5ComputeFixtures.CreateComputeRequest(Gen5ComputeFixtures.ExecStore);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var original, out var error), error);
        var expected = XxHash3.HashToUInt64(Encoding.UTF8.GetBytes(original.Source));
        var metal = new MetalCompiledGuestShader(original, request);
        var program = new ShaderProgram(1, rig.Host.CreateShaderModule(metal, ShaderStage.Compute, 0x5EED, 1));
        var description = new ComputePipelineDescription
        {
            Input = new ComputeInputInfo { ThreadsX = 32, ThreadsY = 1, ThreadsZ = 1 },
            Program = program,
            Stage = new ShaderProgramInfo { Stage = ShaderStageKind.Compute, Hash = 0x5EED },
        };
        rig.Host.CreateComputePipeline(description);

        var (_, modules) = rig.Host.TrimShaderCache(
            rig.Host.CompletedSerial,
            System.Diagnostics.Stopwatch.GetTimestamp() + 3600 * System.Diagnostics.Stopwatch.Frequency,
            new MetalShaderCacheLimits(0, 1, 0, 1, 0));
        Assert.Equal(1, modules);
        Assert.Equal(0, rig.Host.ShaderCacheCounters.ResidentModules);

        rig.Host.CreateComputePipeline(description);

        Assert.Equal(1, rig.Host.ShaderCacheCounters.ModuleRebuilds);
        Assert.Equal(1, rig.Host.ShaderCacheCounters.ResidentModules);
        Assert.True(metal.SourceReleased);
        Assert.Equal(expected, metal.PayloadHash);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ADeferredModuleCompilesOnlyIfAPipelineRequestsIt(bool requested)
    {
        if (!RequireDevice()) return;
        var compiles = 0;
        using (var rig = new Rig(Device.Value))
        {
            rig.Host.DeferLibraryCompiles = true;
            rig.Host.BeforeLibraryCompile = () => Interlocked.Increment(ref compiles);
            var request = Gen5ComputeFixtures.CreateComputeRequest(Gen5ComputeFixtures.ExecStore);
            Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
            var metal = new MetalCompiledGuestShader(shader, request);
            var program = new ShaderProgram(1, rig.Host.CreateShaderModule(metal, ShaderStage.Compute, 0x5EED, 1));
            Assert.Equal(0, Volatile.Read(ref compiles));
            if (requested)
            {
                rig.Host.CreateComputePipeline(new ComputePipelineDescription
                {
                    Input = new ComputeInputInfo { ThreadsX = 32, ThreadsY = 1, ThreadsZ = 1 },
                    Program = program,
                    Stage = new ShaderProgramInfo { Stage = ShaderStageKind.Compute, Hash = 0x5EED },
                });
                Assert.True(metal.SourceReleased);
                Assert.Equal(1, Volatile.Read(ref compiles));
            }
        }
        Assert.Equal(requested ? 1 : 0, Volatile.Read(ref compiles));
    }

    private static PipelineHandle Pipeline(MetalRenderHost host, MetalCompiledGuestShader shader, ulong id)
    {
        var program = new ShaderProgram(id, host.CreateShaderModule(shader, ShaderStage.Compute, 0x5EED + id, id));
        return host.CreateComputePipeline(new ComputePipelineDescription
        {
            Input = new ComputeInputInfo { ThreadsX = 32, ThreadsY = 1, ThreadsZ = 1 },
            Program = program,
            Stage = new ShaderProgramInfo { Stage = ShaderStageKind.Compute, Hash = 0x5EED + id },
        });
    }

    private sealed class AddressSpace : IGuestAddressSpace
    {
        public ulong ProtectionPageSize => 4096;
        public bool TryProtect(ulong address, ulong size, GuestPageProtection protection) => true;
        public ulong AllocateAt(ulong desiredAddress, ulong size, bool executable = true, bool allowAlternative = true) => throw new NotSupportedException();
        public bool TryBackFixedRange(ulong address, ulong size, bool executable) => throw new NotSupportedException();
        public bool TryAllocateAtOrAbove(ulong desiredAddress, ulong size, bool executable, ulong alignment, out ulong actualAddress) => throw new NotSupportedException();
        public bool TryEnsureRangeCommitted(ulong address, ulong size) => throw new NotSupportedException();
        public bool TryAllocateGuestMemory(ulong size, ulong alignment, out ulong address) => throw new NotSupportedException();
        public bool TryFreeGuestMemory(ulong address) => throw new NotSupportedException();
    }

    // One page-aligned native block seen as guest memory at GuestBase.
    private sealed class Backing : IGuestBackedSpace, ICpuMemory, IDisposable
    {
        public const ulong Size = 1024 * 1024;
        private readonly void* _block = NativeMemory.AlignedAlloc((nuint)Size, 16384);

        public Backing() => NativeMemory.Clear(_block, (nuint)Size);

        public void Dispose() => NativeMemory.AlignedFree(_block);

        private bool Contains(ulong address, ulong length) => address >= GuestBase && address - GuestBase + length <= Size;

        public bool TryHoldRange(ulong address, ulong size) => false;
        public bool TryHoldRangeAtOrAbove(ulong searchStart, ulong size, ulong alignment, out ulong address) { address = 0; return false; }
        public bool TryMapBacked(ulong address, ulong size, ulong backingOffset, GuestPageProtection protection, out HostViewFailure failure) { failure = HostViewFailure.None; return false; }
        public bool TryUnmapBacked(ulong address, ulong size) => false;
        public bool TryClearBacking(ulong offset, ulong size) => false;
        public bool IsBackedView(ulong address) => Contains(address, 1);
        public bool IsBackedRange(ulong address, ulong size) => Contains(address, size);
        public (ulong Base, ulong Size) BackingAliasRange => ((ulong)_block, Size);

        public bool TryResolveBackingAlias(ulong address, ulong size, out ulong alias)
        {
            alias = Contains(address, size) ? (ulong)_block + (address - GuestBase) : 0;
            return alias != 0;
        }

        public bool TryWriteBacking(ulong address, ReadOnlySpan<byte> data) => TryWrite(address, data);

        public bool TryReadBacking(ulong address, Span<byte> data) => TryRead(address, data);

        public bool TryRead(ulong virtualAddress, Span<byte> destination)
        {
            if (!Contains(virtualAddress, (ulong)destination.Length)) return false;
            new ReadOnlySpan<byte>((byte*)_block + (virtualAddress - GuestBase), destination.Length).CopyTo(destination);
            return true;
        }

        public bool TryWrite(ulong virtualAddress, ReadOnlySpan<byte> source)
        {
            if (!Contains(virtualAddress, (ulong)source.Length)) return false;
            source.CopyTo(new Span<byte>((byte*)_block + (virtualAddress - GuestBase), source.Length));
            return true;
        }

        public bool CanRead(ulong address, ulong size) => Contains(address, size);
    }

    private sealed class Rig : IDisposable
    {
        private readonly nint _queue;
        private readonly AddressSpace _space = new();
        private readonly PageGuard _pages;
        private readonly Backing _backing = new();

        public Rig(nint device)
        {
            _queue = MetalNative.Send(device, MetalNative.Selector("newCommandQueue"));
            _pages = new PageGuard(_space);
            Host = new MetalRenderHost(device, _queue, _backing, _backing, _pages, new MetalGpuWorkerRelay(static () => { }));
        }

        public MetalRenderHost Host { get; }

        public void Dispose()
        {
            Host.ImageCache.StopWatching();
            Host.Dispose();
            _pages.Dispose();
            _backing.Dispose();
            MetalNative.SendVoid(_queue, MetalNative.Selector("release"));
        }
    }

    private bool RequireDevice()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Module libraries are built on the device only behind the exclusive device gate.");
            return false;
        }

        Assert.True(OperatingSystem.IsMacOS());
        Assert.NotEqual(0, Device.Value);
        return true;
    }

    private static readonly Lazy<nint> Device = new(MetalNative.MTLCreateSystemDefaultDevice);
}
