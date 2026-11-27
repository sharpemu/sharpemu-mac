// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.CompilerServices;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.Libs.Gpu.Abstractions;

namespace SharpEmu.Libs.Gpu.Rendering;

internal readonly record struct DrawCall(string Name, RecordedOperation Operation, uint Count, uint InstanceCount, uint FirstInstance);

// IndirectArgumentsAddress: the GPU reads the draw's counts from guest memory there, and
// IndirectIndexLimit bounds an indexed draw's count as the bound index range does.
// TriangleListIndices: nonzero when the draw was converted to an indexed triangle list of
// that many indices (TriangulateFans), which EmitDraw then draws whatever the guest type.
// PatchControlPoints: the draw is that many patches of this many control points through the
// host's tessellator, whatever the guest primitive type (RenderExecutor.Tessellation.cs).
internal readonly record struct DrawEmission(bool Indexed, int VertexOffset, uint FirstVertex, uint FirstInstance,
    ulong IndirectArgumentsAddress = 0, uint IndirectIndexLimit = uint.MaxValue, uint TriangleListIndices = 0,
    uint PatchControlPoints = 0);

internal readonly record struct IndexSource(bool Enabled, ulong Address, byte[]? HostData, ulong Size, GpuIndexType Type);

internal readonly record struct DebugInformation(RecordedOperation Operation, ulong SubmitId, uint Argument0, uint Argument1, uint Argument2, uint Argument3, ulong Argument4);

[InlineArray(RenderingState.ColorAttachmentCapacity)]
internal struct ColorTargetStates
{
    private ColorTargetState _element0;
}

internal struct DrawState
{
    public ColorTargetStates Colors;
    public uint ColorCount;
    public DepthAttachmentState Depth;
    public bool DepthResolved;
    public bool PixelActive;
    public RenderingState Rendering;
    public GraphicsPrograms Programs;
    public int StartInstanceUserData;
    public int BaseVertexUserData;

    public static DrawState Create() => new() { PixelActive = true, Programs = null!, StartInstanceUserData = -1, BaseVertexUserData = -1 };

    public void Begin(int startInstanceUserData, int baseVertexUserData = -1)
    {
        BaseVertexUserData = baseVertexUserData;
        ColorCount = 0;
        Depth = default;
        DepthResolved = false;
        PixelActive = true;
        Rendering.ColorAttachmentCount = 0;
        Programs = null!;
        StartInstanceUserData = startInstanceUserData;
    }

    public static ReadOnlySpan<ColorTargetState> BoundColors(ref DrawState state) =>
        ((ReadOnlySpan<ColorTargetState>)state.Colors)[..(int)state.ColorCount];
}

internal enum ResolvedDrawKind
{
    Skipped,
    Draw,
    ClearColorTargets,
    MultisampleResolve,
    DepthStencilCopy,
    Tessellation,
}

internal interface IResolvedCommandSink
{
    long NextSequence();

    void Record(ResolvedDraw record);

    void Record(ResolvedDispatch record);
}

/// <summary>A tessellated draw's hull kernel, run where its patch draw is recorded when the executor hands its
/// records to a sink that records them later (the Metal translation): the kernel has to land between the records
/// before and after the draw, as RenderRecorder.RunTessellationHull does for a direct recorder.</summary>
internal sealed record TessellationHullLaunch(
    ComputeProgram Hull, uint MinTessellationLevel, uint MaxTessellationLevel, uint[] Indices, uint ControlPoints,
    bool Quad, uint PatchCount, uint Instances, uint FirstInstance);

internal sealed class ResolvedDraw
{
    public long Sequence;
    public TessellationHullLaunch? TessellationHull;
    public ResolvedDrawKind Kind;
    public bool Announce;
    public DebugInformation Debug;
    public bool ResetBindings;
    public bool TargetsResolved;
    public ulong SubmitId;
    public DrawCall Draw;
    public DrawState State = DrawState.Create();
    public GpuPrimitiveTopology Topology;
    public bool PrimitiveRestart;
    public GuestPrimitiveType PrimitiveType;
    public DrawEmission Emission;
    public uint[]? IndirectOffsets;
    public uint IndirectBytes;
    public IndexSource Index;
    public ulong TessellationIndexAddress;
    public uint TessellationIndexSize;
    public int TessellationVertexOffset;
    public ContextRegisters Context = null!;
    public ulong ContextStamp;
    public ulong DynamicStamp;
    public uint TargetVersion;
    public DynamicDrawState Dynamic;
    public bool WritesMemory;
    public GpuPipelineStage WriteStages;
    public bool SetBindDebug;
    public bool SetAutoDebug;
    // The draw packet asked to run even while its pipeline compiles (DrawIndexedArguments.MustExecute).
    public bool MustExecute;
    public GpuCommands.IndexedDrawTrace.Sample? IndexedTrace;
    public ulong PacketAddress;
    public SolidColorClear Clear;
    public ColorTargetState ResolveSource;
    public ColorTargetState ResolveDestination;
    public ImageRequest CopyRead;
    public ImageRequest CopyWrite;
    public GpuFormat CopyReadFormat;
    public GpuFormat CopyWriteFormat;
    public GpuExtent3D CopyExtent;
    public GpuImageAspect CopyAspects;

    // A multisample resolve or depth/stencil copy wrote the fields above since the last Reset; until one does they
    // hold their defaults, and Reset leaves them alone (two 328-byte targets and two image requests every draw).
    public bool TargetCopyFieldsWritten;

    // An indexed indirect draw resolved on the plain path (no host index data, no strip, mesh or tessellation): the next draw
    // under the same registers can take this record with only its own arguments (RenderExecutor.TryRepeatIndexed).
    public bool RepeatableArguments;

    // The arguments were read on the CPU (DrawIndexedWithCpuArguments); a repeat reads its own the same way.
    public bool RepeatsCpuArguments;

    public void CopyFrom(ResolvedDraw source)
    {
        Sequence = source.Sequence;
        Kind = source.Kind;
        Announce = source.Announce;
        Debug = source.Debug;
        ResetBindings = source.ResetBindings;
        TargetsResolved = source.TargetsResolved;
        SubmitId = source.SubmitId;
        Draw = source.Draw;
        State = source.State;
        Topology = source.Topology;
        PrimitiveRestart = source.PrimitiveRestart;
        PrimitiveType = source.PrimitiveType;
        Emission = source.Emission;
        Index = source.Index;
        TessellationIndexAddress = source.TessellationIndexAddress;
        TessellationIndexSize = source.TessellationIndexSize;
        TessellationVertexOffset = source.TessellationVertexOffset;
        Context = source.Context;
        ContextStamp = source.ContextStamp;
        DynamicStamp = source.DynamicStamp;
        TargetVersion = source.TargetVersion;
        Dynamic = source.Dynamic;
        WritesMemory = source.WritesMemory;
        WriteStages = source.WriteStages;
        SetBindDebug = source.SetBindDebug;
        SetAutoDebug = source.SetAutoDebug;
        IndexedTrace = source.IndexedTrace;
        PacketAddress = source.PacketAddress;
        Clear = source.Clear;
        ResolveSource = source.ResolveSource;
        ResolveDestination = source.ResolveDestination;
        CopyRead = source.CopyRead;
        CopyWrite = source.CopyWrite;
        CopyReadFormat = source.CopyReadFormat;
        CopyWriteFormat = source.CopyWriteFormat;
        CopyExtent = source.CopyExtent;
        CopyAspects = source.CopyAspects;
        TargetCopyFieldsWritten = source.TargetCopyFieldsWritten;
        TessellationHull = source.TessellationHull;
        MustExecute = source.MustExecute;
        RepeatableArguments = source.RepeatableArguments;
        RepeatsCpuArguments = source.RepeatsCpuArguments;
    }

    public void Reset()
    {
        var previous = Kind;
        TessellationHull = null;
        RepeatableArguments = false;
        RepeatsCpuArguments = false;
        Sequence = 0;
        Kind = ResolvedDrawKind.Skipped;
        Announce = false;
        Debug = default;
        ResetBindings = false;
        TargetsResolved = false;
        SubmitId = 0;
        Draw = default;
        State.Begin(-1);
        Topology = default;
        PrimitiveRestart = false;
        PrimitiveType = default;
        Emission = default;
        IndirectOffsets = null;
        IndirectBytes = 0;
        Index = default;
        TessellationIndexAddress = 0;
        TessellationIndexSize = 0;
        TessellationVertexOffset = 0;
        Context = null!;
        ContextStamp = 0;
        DynamicStamp = 0;
        TargetVersion = 0;
        Dynamic = default;
        WritesMemory = false;
        WriteStages = default;
        SetBindDebug = false;
        SetAutoDebug = false;
        MustExecute = false;
        IndexedTrace = null;
        PacketAddress = 0;
        if (previous == ResolvedDrawKind.ClearColorTargets)
        {
            Clear = default;
        }

        // The resolve and copy fields: written by those kinds only, and flagged when they are (TargetCopyFieldsWritten).
        if (TargetCopyFieldsWritten)
        {
            ResolveSource = default;
            ResolveDestination = default;
            CopyRead = default;
            CopyWrite = default;
            CopyReadFormat = default;
            CopyWriteFormat = default;
            CopyExtent = default;
            CopyAspects = default;
            TargetCopyFieldsWritten = false;
        }
    }
}

internal readonly record struct MetadataBufferProbe(ulong Address, bool Written, bool Read);

internal readonly record struct ConstantFillCandidate(ulong Address, ulong Size, uint Value);

internal readonly record struct BoundedCopyCandidate(ulong Destination, ulong Source, ulong Modulus, ulong Words);

internal sealed class ResolvedDispatch
{
    public long Sequence;
    public DebugInformation Debug;
    public bool Skipped;
    public ulong SubmitId;
    public ulong ShaderAddress;
    public ComputeProgram Program = null!;
    public uint GroupsX;
    public uint GroupsY;
    public uint GroupsZ;
    public ulong IndirectArgumentsAddress;

    // Thread counts left in guest memory that the GPU converts to group counts
    // (IRenderHost.TryDispatchThreadsIndirect); the group counts are then zero.
    public bool ThreadsOnGpu;

    // The stream put nothing between this dispatch and the one before it that orders them
    // (RenderExecutor.DispatchMayOverlapBit): the host may run the two side by side.
    public bool MayOverlap;

    // The indirect arguments address the packet carried, before thread dimensions cleared it:
    // a suspended dispatch is matched to its retry by it.
    public ulong PacketIndirectArgumentsAddress;

    // The resolver already checked the recording, ran the host's pending operations and set the
    // debug information (RenderExecutor.RecordsDirectly), so the recorder does not again.
    public bool HostPrepared;

    // A dispatch that may suspend on its pipeline: what it prepared, and where the recorder keeps
    // it for the packet's next run. Null when the dispatch waits for its pipeline instead.
    public Pipelines.ComputeProgramRetry? Retry;
    public ComputeDispatchSuspensions? Suspensions;

    // Resolved from a suspended attempt's program; the host consumers declined it then.
    public bool Retried;

    // The first MetadataClearCount entries are the buffers of a metadata clear candidate. The
    // array stays with the pooled record.
    public MetadataClearProbe[] MetadataClearBuffer = [];
    public int MetadataClearCount;

    // The indirect arguments count threads the GPU converts to groups (IRenderHost.PrepareThreadDispatch, the Vulkan
    // host's form; ThreadsOnGpu is the Metal host's).
    public bool GpuThreadCounts;
    public ConstantFillCandidate? ConstantFill;
    public KernelFillCandidate? BoundedFill;
    public BoundedCopyCandidate? BoundedCopy;
    public ComputeImageClear? ImageClear;
    public bool HasStorageWrites;

    public ReadOnlySpan<MetadataClearProbe> MetadataClear => MetadataClearBuffer.AsSpan(0, MetadataClearCount);

    public void Reset()
    {
        Sequence = 0;
        Debug = default;
        Skipped = false;
        SubmitId = 0;
        ShaderAddress = 0;
        Program = null!;
        GroupsX = 0;
        GroupsY = 0;
        GroupsZ = 0;
        IndirectArgumentsAddress = 0;
        ThreadsOnGpu = false;
        MayOverlap = false;
        PacketIndirectArgumentsAddress = 0;
        HostPrepared = false;
        Retry = null;
        Suspensions = null;
        Retried = false;
        MetadataClearCount = 0;
        GpuThreadCounts = false;
        ConstantFill = null;
        BoundedFill = null;
        BoundedCopy = null;
        ImageClear = null;
        HasStorageWrites = false;
    }
}
