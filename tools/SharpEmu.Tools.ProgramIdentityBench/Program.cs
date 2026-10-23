// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Reflection;
using SharpEmu.Libs.Gpu;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;
using System.Runtime.InteropServices;
using System.Buffers.Binary;
using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Pipelines;

const int count = 2_000_000;
var memory = new BenchMemory();
using var generations = new GuestWriteGenerations();
var registry = new ShaderHeaderRegistry(new CpuContext(memory, Generation.Gen5), _ => 0x20000,
    _ => null, generations, static (_, _) => { });
var cache = new ShaderIdentityCache(memory, static (_, _) => { }, generations);
var providerType = typeof(ShaderIdentity).Assembly.GetType("SharpEmu.Libs.Gpu.Pipelines.ShaderPipelineCache")!;
var provider = (IShaderPipelineProvider)providerType.GetConstructors().Single().Invoke(
    [new CpuContext(memory, Generation.Gen5), new BenchHost(), DispatchProxy.Create<IGuestGpuBackend, UnusedBackend>(), registry, generations]);
var vertex = new VertexInputInfo { Stage = new ShaderStageResources(new ShaderProgramInfo { Stage = ShaderStageKind.Vertex }, null!) };
var context = new ContextRegisters();
var rendering = new RenderingState { Samples = 1 };
ulong SourceLookup() => cache.Get(registry.Require(0x10000, "bench"), "bench").ContentHash;
ulong PipelineLookup() => provider.CreateGraphicsPipeline([], default, vertex, null, context, rendering,
    GpuPrimitiveTopology.TriangleList, false, false, new ShaderProgram(1), default).Pipeline;
Measure("header+code", SourceLookup);
Measure("pipeline", PipelineLookup);
Measure("combined", () => SourceLookup() ^ PipelineLookup());

void Measure(string label, Func<ulong> lookup)
{
    ulong sink = 0;
    for (var i = 0; i < 50_000; i++) sink ^= lookup();
    var allocations = GC.GetAllocatedBytesForCurrentThread();
    var before = Stopwatch.GetTimestamp();
    for (var i = 0; i < count; i++) sink ^= lookup();
    var elapsed = Stopwatch.GetElapsedTime(before);
    allocations = GC.GetAllocatedBytesForCurrentThread() - allocations;
    Console.WriteLine($"path={label} arch={RuntimeInformation.ProcessArchitecture} intern={Environment.GetEnvironmentVariable("SHARPEMU_PROGRAM_INTERN") ?? "1"} draws={count} code_bytes={BenchMemory.CodeSize} ns/draw={elapsed.TotalNanoseconds / count:F1} bytes/draw={(double)allocations / count:F3} sink={sink}");
}

public class UnusedBackend : DispatchProxy
{
    protected override object? Invoke(MethodInfo? targetMethod, object?[]? args) => throw new InvalidOperationException("The warm benchmark must never compile a shader.");
}

sealed class BenchHost : IShaderPipelineHost
{
    public uint MaxPushDescriptors => 32;
    public bool ComputeWave64Supported => true;
    public bool GraphicsSubgroupOperationsEnabled => false;
    public bool SharedInt64AtomicsEnabled => false;
    public RenderHostLimits Limits => default;
    public GpuSampleCount NoAttachmentSampleCounts => GpuSampleCount.Count1Bit;
    public bool TryResolveColorOutput(uint format, uint type, uint swap, out Gen5PixelOutputKind kind, out Gen5ColorComponentMapping mapping)
    { kind = default; mapping = default; return false; }
    public bool TryReadGuestWord(ulong address, out uint word) { word = 0; return false; }
    public bool TryReadCleanGuestWord(ulong address, out uint word) { word = 0; return false; }
    public ulong CreateShaderModule(IGuestCompiledShader shader, ShaderStage stage, ulong hash, ulong id) => throw new InvalidOperationException();
    public PipelineHandle CreateGraphicsPipeline(GraphicsPipelineDescription description) => new(1, 0, false);
    public PipelineHandle CreateComputePipeline(ComputePipelineDescription description) => throw new InvalidOperationException();
}

sealed class BenchMemory : ICpuMemory
{
    public const int CodeSize = 16 * 1024;
    private readonly byte[] _code = new byte[CodeSize];
    private readonly byte[] _header = new byte[0x60];
    public BenchMemory()
    {
        new Random(1).NextBytes(_code);
        BinaryPrimitives.WriteUInt32LittleEndian(_code, 0xBF810000);
        BinaryPrimitives.WriteUInt32LittleEndian(_header.AsSpan(0x44), CodeSize);
    }
    public bool TryRead(ulong address, Span<byte> destination)
    {
        var bytes = address >= 0x20000 ? _header : _code;
        var offset = (int)(address >= 0x20000 ? address - 0x20000 : address - 0x10000);
        if (offset < 0 || destination.Length > bytes.Length - offset) return false;
        bytes.AsSpan(offset, destination.Length).CopyTo(destination);
        return true;
    }
    public bool TryWrite(ulong address, ReadOnlySpan<byte> source) => false;
}
