// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Images;

namespace SharpEmu.Libs.Gpu.Rendering;

// A compiled shader module the pipeline provider hands back; zero means no program.
public readonly record struct ShaderProgram(ulong Id, ulong Module = 0)
{
    public bool IsValid => Id != 0;
}

public readonly record struct PipelineHandle(ulong Pipeline, ulong Layout, bool UsesPushDescriptors);

// A clear the provider recognised in the program pair; the host clears the targets instead of drawing.
public readonly record struct SolidColorClear(float Red, float Green, float Blue, float Alpha);

// The float32x3 position stream of the vertex program, when it has one.
public readonly record struct VertexPositionStream(ulong Address, uint Stride, uint OffsetBytes);

public sealed class GraphicsPrograms
{
    // A natively tessellated draw's configuration and hull program (hosts with SupportsTessellation).
    public TessellationDrawPrograms? Tessellation { get; init; }

    public GraphicsPrograms()
    {
    }

    public GraphicsPrograms(VertexInputInfo vertexInput, PixelInputInfo pixelInput)
    {
        _vertexInput = vertexInput;
        _pixelInput = pixelInput;
    }

    public ShaderProgram Vertex { get; init; }
    public ShaderProgram Pixel { get; init; }
    // A default is made on first read: an initializer made one for every draw only for the
    // resolver to replace it.
    public VertexInputInfo VertexInput { get => _vertexInput ??= new(); init => _vertexInput = value; }
    public MeshDrawConfiguration? MeshInput { get; init; }
    public PixelInputInfo PixelInput { get => _pixelInput ??= new(); init => _pixelInput = value; }
    private VertexInputInfo? _vertexInput;
    private PixelInputInfo? _pixelInput;

    // False when the provider could not build the programs; the executor skips the draw.
    public bool Available { get; init; } = true;

    public SolidColorClear? SolidClear { get; init; }

    public VertexPositionStream? PositionStream { get; init; }

    // Blending is off for a fill the provider recognised as an overwrite.
    public bool DisableBlending { get; init; }

    // The pixel program was compiled without the color targets it never writes,
    // so the executor drops them too and attachments stay aligned with its outputs.
    public bool DropUnwrittenColorTargets { get; init; }
}

// A tessellated draw's programs: the domain and pixel programs it draws with, and the hull kernel
// that runs before it (ShaderPipelineCache.GetTessellationPrograms).
public sealed record TessellationPrograms(GraphicsPrograms Graphics, ComputeProgram Hull);

public sealed record TessellationDrawPrograms(SharpEmu.ShaderCompiler.Gen5TessellationInfo Configuration,
    SharpEmu.ShaderCompiler.Gen5TessellationHullInfo HullConfiguration, ComputeProgram Hull);
// Learns when a known write of one producer is read.
internal interface IKnownWriteDemand
{
    void NoteRead();
}

public sealed class KnownGpuWrite(ulong address, uint[] buffer, int count, bool pooled = false)
{
    private readonly object _gate = new();
    private bool _released;
    private int _read;

    internal IKnownWriteDemand? Demand { get; init; }

    public ulong Address { get; } = address;

    internal void NoteRead()
    {
        if (Demand is { } demand && Interlocked.Exchange(ref _read, 1) == 0)
        {
            demand.NoteRead();
        }
    }

    public int ByteCount => count * sizeof(uint);

    public ReadOnlySpan<uint> Words => buffer.AsSpan(0, count);

    public bool TryCopyWords(int offsetBytes, Span<byte> destination)
    {
        lock (_gate)
        {
            if (_released)
            {
                return false;
            }

            System.Runtime.InteropServices.MemoryMarshal.AsBytes(buffer.AsSpan(0, count)).Slice(offsetBytes, destination.Length).CopyTo(destination);
            return true;
        }
    }

    public void Release()
    {
        lock (_gate)
        {
            if (pooled && !_released)
            {
                _released = true;
                System.Buffers.ArrayPool<uint>.Shared.Return(buffer);
            }
        }
    }
}

public sealed record FrontendImageCopy(ImageRequest Source, ImageRequest Destination, Func<ComputeProgram> Fallback);

public sealed class ComputeProgram
{
    public FrontendImageCopy? ImageCopy { get; init; }
    public KnownGpuWrite? KnownWrite { get; init; }

    public ShaderProgram Program { get; init; }
    // Made on first read, like GraphicsPrograms' inputs.
    public ComputeInputInfo Input { get => _input ??= new(); init => _input = value; }
    private ComputeInputInfo? _input;

    // False when the provider could not build the program; the executor skips the dispatch.
    public bool Available { get; init; } = true;

    // True when the provider ran the kernel itself; the executor records nothing.
    public bool Consumed { get; init; }
}

// The shader and pipeline caches behind the executor.
public interface IShaderPipelineProvider
{
    // True when the backend can run an NGG primitive shader as a mesh pipeline. The
    // executor then stops skipping culling primitive-shader draws; see IGuestGpuBackend.
    bool SupportsPrimitiveShaderMeshPipelines => false;

    // Optional capture of rejected stage inputs; it never admits or executes the draw.
    void RecordUnsupportedGraphics(VertexStageRegisters vertex, ContextRegisters context) { }

    // Original packet and primitive-assembly state at the rejection boundary.
    // Legacy diagnostic implementers retain their existing optional callback.
    void RecordUnsupportedGraphics(VertexStageRegisters vertex, ContextRegisters context,
        UserConfigRegisters userConfig, in TargetlessDrawArguments draw,
        DrawIndexedArguments? originalIndirectIndexedDraw = null) =>
        RecordUnsupportedGraphics(vertex, context);

    bool MeshShadersSupported => false;

    void DumpRejectedGraphics(RegisterBanks banks, bool autoDraw) { }

    // The programs of a draw. triangleList: VGT_PRIMITIVE_TYPE is a triangle list; only then can a
    // passthrough primitive shader run as a mesh function that hands its pixel program the three
    // vertex values a host without per-vertex pixel inputs cannot give it. startInstanceUserData:
    // the vertex user data register that carries the draw's first instance, or -1.
    GraphicsPrograms GetGraphicsPrograms(
        RegisterBanks banks,
        ReadOnlySpan<ColorComponentMap> targetExportMapping,
        bool pixelActive,
        bool depthBound,
        bool depthAspectBound,
        bool stencilAspectBound,
        bool triangleList,
        int startInstanceUserData);

    GraphicsPrograms GetMeshGraphicsPrograms(
        RegisterBanks banks,
        UserConfigRegisters userConfig,
        ReadOnlySpan<ColorComponentMap> targetExportMapping,
        bool pixelActive,
        bool depthBound) => new() { Available = false };

    PipelineHandle CreateGraphicsPipeline(
        ReadOnlySpan<ColorTargetState> colors,
        in DepthAttachmentState depth,
        VertexInputInfo vertexInput,
        PixelInputInfo? pixelInput,
        ContextRegisters context,
        in RenderingState rendering,
        GpuPrimitiveTopology topology,
        bool primitiveRestartEnabled,
        bool disableBlending,
        ShaderProgram vertexProgram,
        ShaderProgram pixelProgram);

    // False means the host is still compiling this pipeline and the draw may be left out;
    // asking again returns it once the compile finishes, without starting a second one.
    bool TryCreateGraphicsPipeline(
        ReadOnlySpan<ColorTargetState> colors,
        in DepthAttachmentState depth,
        VertexInputInfo vertexInput,
        PixelInputInfo? pixelInput,
        ContextRegisters context,
        in RenderingState rendering,
        GpuPrimitiveTopology topology,
        bool primitiveRestartEnabled,
        bool disableBlending,
        ShaderProgram vertexProgram,
        ShaderProgram pixelProgram,
        out PipelineHandle handle)
    {
        handle = CreateGraphicsPipeline(colors, in depth, vertexInput, pixelInput, context, in rendering, topology,
            primitiveRestartEnabled, disableBlending, vertexProgram, pixelProgram);
        return true;
    }

    // The programs of a tessellated draw run without a tessellator, or null when the provider cannot
    // build them (the executor then skips the draw, as before).
    TessellationPrograms? GetTessellationPrograms(
        RegisterBanks banks,
        ReadOnlySpan<ColorComponentMap> targetExportMapping,
        bool pixelActive,
        bool depthBound,
        bool depthAspectBound,
        bool stencilAspectBound,
        uint controlPoints,
        bool quad) => null;

    // The dispatch dimensions are groups, or threads when the initiator counts threads.
    ComputeProgram GetComputeProgram(
        ComputeStageRegisters compute,
        ShaderInterfaceRegisters shaderInterface,
        uint dispatchInitiator,
        uint dimensionX,
        uint dimensionY,
        uint dimensionZ);

    // The same for a dispatch that may suspend on its pipeline: retry keeps the prepared program
    // for the attempt that runs it, which takes it back while the guest words it read still
    // hold (ComputeProgramRetry). A provider that keeps nothing prepares it every time.
    ComputeProgram GetComputeProgram(
        ComputeStageRegisters compute,
        ShaderInterfaceRegisters shaderInterface,
        uint dispatchInitiator,
        uint dimensionX,
        uint dimensionY,
        uint dimensionZ,
        Pipelines.ComputeProgramRetry retry) =>
        GetComputeProgram(compute, shaderInterface, dispatchInitiator, dimensionX, dimensionY, dimensionZ);

    // The program of a dispatch the stream has not reached, prepared as that dispatch will
    // prepare it but consuming nothing (no adaptation kernel is submitted), so its pipeline can
    // be asked for early. A provider that cannot prepare ahead says it is not available.
    ComputeProgram PrepareComputeProgramAhead(
        ComputeStageRegisters compute,
        uint dispatchInitiator,
        uint dimensionX,
        uint dimensionY,
        uint dimensionZ) => new() { Available = false };

    ComputeProgram GetComputeProgram(RegisterBanks banks, uint dispatchInitiator, uint dimensionX, uint dimensionY, uint dimensionZ) =>
        GetComputeProgram(banks.Shader.Compute, banks.Context.ShaderInterface, dispatchInitiator, dimensionX, dimensionY, dimensionZ);

    // An indirect dispatch reads its dimensions on the GPU, so the provider cannot know its result.
    ComputeProgram GetComputeProgram(RegisterBanks banks, uint dispatchInitiator, uint dimensionX, uint dimensionY, uint dimensionZ, bool indirect) =>
        GetComputeProgram(banks, dispatchInitiator, dimensionX, dimensionY, dimensionZ);

    // Both: a dispatch that may suspend on its pipeline keeps its prepared program in retry.
    ComputeProgram GetComputeProgram(RegisterBanks banks, uint dispatchInitiator, uint dimensionX, uint dimensionY, uint dimensionZ, bool indirect,
        Pipelines.ComputeProgramRetry? retry) =>
        retry is null
            ? GetComputeProgram(banks, dispatchInitiator, dimensionX, dimensionY, dimensionZ, indirect)
            : GetComputeProgram(banks.Shader.Compute, banks.Context.ShaderInterface, dispatchInitiator, dimensionX, dimensionY, dimensionZ, retry);

    PipelineHandle CreateComputePipeline(ComputeInputInfo input, ShaderProgram program);

    Task? GetComputePipelineCompletion(ShaderProgram program) => null;

    // False means the host is still compiling this pipeline; asking again returns it once
    // the compile finishes, without starting a second one.
    bool TryCreateComputePipeline(ComputeInputInfo input, ShaderProgram program, out PipelineHandle handle)
    {
        handle = CreateComputePipeline(input, program);
        return true;
    }
}
