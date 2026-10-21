// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Libs.Gpu.Pipelines;

// Everything a host pipeline creation reads for one graphics pipeline.
public sealed class GraphicsPipelineDescription
{
    public required PipelineRenderingState Rendering { get; init; }
    public required PipelineVertexInputState VertexInput { get; init; }
    public required VertexInputInfo VertexInfo { get; init; }
    public required ShaderProgram VertexProgram { get; init; }
    public required ShaderProgramInfo VertexStage { get; init; }
    public PixelInputInfo? PixelInfo { get; init; }
    public ShaderProgram PixelProgram { get; init; }
    public ShaderProgramInfo? PixelStage { get; init; }
    public required PipelineStaticParameters StaticParameters { get; init; }

    // The pixel program's LOD statistics counters are compiled in (the title uses LOD
    // statistics, MipStatistics.TitleUsesCounters, and the host asked for the split in
    // IShaderPipelineHost.PixelPipelinesFollowLodStatistics). Part of the pipeline key.
    public bool LodStatistics { get; init; }
}

public sealed class ComputePipelineDescription
{
    public required ComputeInputInfo Input { get; init; }
    public required ShaderProgram Program { get; init; }
    public required ShaderProgramInfo Stage { get; init; }
}

public readonly record struct MeshShaderLimits(
    uint MaxInvocations,
    uint OutputVertexCapacity,
    uint OutputPrimitiveCapacity,
    uint MaxSharedMemoryBytes,
    uint MaxGroupCountX,
    uint MaxGroupCountY,
    uint MaxGroupTotalCount,
    uint MaxWorkGroupSizeX,
    uint MaxOutputMemoryBytes,
    uint MaxPayloadAndOutputMemoryBytes,
    uint OutputPerVertexGranularity,
    uint OutputPerPrimitiveGranularity,
    uint MaxOutputComponents);

// The host objects the shader and pipeline caches need: modules, pipelines, limits and guest readers.
public interface IShaderPipelineHost
{
    uint MaxPushDescriptors { get; }

    bool ComputeWave64Supported { get; }

    // True when a 64-invocation wave64 compute workgroup runs as one 64-lane host subgroup.
    bool ComputeWave64SubgroupNative => false;

    bool ComputeRunsOn32LaneSubgroups(ComputeInputInfo input) => false;

    bool MipStatisticsSubgroupReduction => false;

    bool GraphicsSubgroupOperationsEnabled { get; }

    // The device supports shaderSharedInt64Atomics, so LDS 64-bit atomics can be
    // emitted as real 64-bit atomics instead of a non-atomic 32-bit pair.
    bool SharedInt64AtomicsEnabled { get; }

    // The device supports shaderImageInt64Atomics, so an atomic on a 64-bit image texel
    // (a 32_32 UINT image) can be compiled; without it such a program is rejected.
    bool ImageInt64AtomicsEnabled => false;
    bool ShaderSignedZeroInfNanPreserveFloat32Supported => false;
    bool ExecGuardElisionEnabled => true;
    ShaderPrewarmList? ShaderPrewarm => null;
    GraphicsPrewarmList? GraphicsPrewarm => null;
    // The host makes a pixel pipeline with or without the LOD statistics counters, by whether
    // the title uses them yet (MipStatistics.TitleUsesCounters): a pixel pipeline made before
    // is not the one wanted after, so the pipeline cache keys on it. A host whose pipelines
    // never differ answers false and keeps one pipeline.
    bool PixelPipelinesFollowLodStatistics => false;

    bool PerVertexPixelInputsSupported => true;
    bool ClipDistanceEnabled => false;

    bool MeshShadersSupported => false;

    MeshShaderLimits MeshLimits => default;
    uint MeshSubgroupSize => ComputeWave64Supported ? 64u : 32u;

    RenderHostLimits Limits { get; }

    // The device can blend with a second fragment output (dualSrcBlend).
    bool SupportsDualSourceBlend => false;

    // The depth bounds test is part of the host's pipeline object (min, max and enable). A host
    // that sets it as encoder state, or has no such test, answers false, and the bounds then stay
    // out of the pipeline key instead of splitting one pipeline per bounds a draw happens to use.
    bool DepthBoundsInPipelineState => true;

    // The sample counts a pipeline without attachments can rasterize at.
    GpuSampleCount NoAttachmentSampleCounts { get; }

    bool TryResolveColorOutput(uint dataFormat, uint numberType, uint componentSwap, out Gen5PixelOutputKind outputKind, out Gen5ColorComponentMapping componentMapping);

    // Reads one guest dword the CPU may see; a range the GPU wrote is downloaded first.
    bool TryReadGuestWord(ulong address, out uint word);

    // Bracket one resource materialization's TryReadGuestWord calls on the calling thread.
    // Inside, a host may answer a word from guest bytes it read and checked once, when that
    // is the answer a word read would give; outside, every word is read afresh.
    void BeginGuestWordReads() { }

    void EndGuestWordReads() { }

    // Waits for recorded GPU work still writing these bytes, before shader code is hashed
    // and decoded out of guest memory: a title that moves its shader code into place with
    // a DMA copy otherwise has the decoder read whatever the bytes held before the copy.
    void SynchronizeShaderCode(ulong address, ulong size) { }

    // Reads one guest dword only when no GPU work may still own the range.
    bool TryReadCleanGuestWord(ulong address, out uint word);

    // Copies guest bytes the CPU already holds, without synchronizing. False when the GPU
    // may own the range (or, for a clean read, when a clean word read would be refused);
    // the resource cache then re-materializes instead of trusting a stale copy.
    bool TryReadResidentGuestBytes(ulong address, Span<byte> destination, bool clean) => false;

    // TryReadResidentGuestBytes for each range, compared with the recorded bytes where they
    // lie instead of copied out. True only when every range passed the same checks and is
    // equal; false when one differs or the host cannot compare in place.
    bool TryCompareResidentGuestBytes(ReadOnlySpan<ulong> addresses, ReadOnlySpan<int> lengths, ReadOnlySpan<int> offsets,
        ReadOnlySpan<byte> recorded) => false;

    // Whether recorded GPU work may still write this dword, answered without waiting. A
    // host that returns a delegate lets a shader whose data-only table words the GPU writes
    // be re-planned to load them on the device (DeviceDataLoads); null keeps them flattened.
    Func<ulong, bool>? DataWritePending => null;

    // Creates the host module of one compiled permutation and returns its handle.
    ulong CreateShaderModule(IGuestCompiledShader shader, ShaderStage stage, ulong hash, ulong programId);

    PipelineHandle CreateGraphicsPipeline(GraphicsPipelineDescription description);

    // Creates the graphics pipeline without blocking the command stream on the host shader
    // compiler: false means the compile is still running, and carries on, and the caller
    // leaves the draw out; true hands over the finished pipeline. Asked only for a draw the
    // executor judged safe to leave out (RenderExecutor.MayLeaveOutWhileCompiling); a
    // backend that compiles cheaply keeps the blocking behaviour.
    bool TryCreateGraphicsPipeline(GraphicsPipelineDescription description, out PipelineHandle handle)
    {
        handle = CreateGraphicsPipeline(description);
        return true;
    }

    PipelineHandle CreateComputePipeline(ComputePipelineDescription description);

    Task? GetComputePipelineCompletion(ShaderProgram program) => null;

    // Creates the compute pipeline without blocking the command stream on the host
    // shader compiler: false means the compile is still running and the caller should ask
    // again, true hands over the finished pipeline. A backend that compiles cheaply keeps
    // the blocking behaviour.
    bool TryCreateComputePipeline(ComputePipelineDescription description, out PipelineHandle handle)
    {
        handle = CreateComputePipeline(description);
        return true;
    }
}
