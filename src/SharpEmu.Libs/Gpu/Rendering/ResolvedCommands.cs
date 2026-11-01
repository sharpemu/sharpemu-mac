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

    public static DrawState Create() => new() { PixelActive = true, Programs = null!, StartInstanceUserData = -1 };

    public void Begin(int startInstanceUserData)
    {
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

internal sealed class ResolvedDraw
{
    public long Sequence;
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

    public void Reset()
    {
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
        Clear = default;
        ResolveSource = default;
        ResolveDestination = default;
        CopyRead = default;
        CopyWrite = default;
        CopyReadFormat = default;
        CopyWriteFormat = default;
        CopyExtent = default;
        CopyAspects = default;
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
        PacketIndirectArgumentsAddress = 0;
        HostPrepared = false;
        Retry = null;
        Suspensions = null;
        Retried = false;
        MetadataClearCount = 0;
        ConstantFill = null;
        BoundedFill = null;
        BoundedCopy = null;
        ImageClear = null;
        HasStorageWrites = false;
    }
}
