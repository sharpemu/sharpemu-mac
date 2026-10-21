// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.Libs.VideoOut;
using ResourceSnapshot = SharpEmu.ShaderCompiler.Resources.ResourceSnapshot;

namespace SharpEmu.Libs.Gpu.Rendering;

public enum GuestPrimitiveType : uint
{
    None = 0,
    PointList = 1,
    LineList = 2,
    LineStrip = 3,
    TriangleList = 4,
    TriangleFan = 5,
    TriangleStrip = 6,
    RectangleList = 7,
    RectangleListLegacy = 17,
    QuadListLegacy = 19,
    Polygon = 21,
}

public enum GuestIndexType : uint
{
    Index16 = 0,
    Index32 = 1,
    Index8 = 2,
}

// Resolves draw and dispatch state from the register banks and records it through the host.
public sealed partial class RenderExecutor
{
    // The stage-mask readings live in VgtShaderStages, so the pipeline cache compiles the
    // vertex program the same way this decides whether the draw runs.
    private static bool IsPrimitiveShaderStageMask(uint stages) =>
        VgtShaderStages.IsPrimitiveShader(stages);

    private const uint MaxOutputPerSubgroupLimit = 0x40;
    private static int _geometryWarningShown;

    private readonly IRenderHost _host;
    private readonly IShaderPipelineProvider _pipelines;
    private readonly bool _strictDrawResources;
    private readonly bool _dumpSkippedGraphics = Environment.GetEnvironmentVariable("SHARPEMU_DUMP_SKIPPED_GRAPHICS") == "1";
    private readonly HashSet<(ulong ShaderHash, GpuImageType ImageType, GpuImageViewType ViewType)> _reportedDrawImageTypeMismatches = [];

    public RenderExecutor(IRenderHost host, IShaderPipelineProvider pipelines)
        : this(host, pipelines, Environment.GetEnvironmentVariable("SHARPEMU_STRICT_COMPUTE") != "0")
    {
    }

    internal RenderExecutor(IRenderHost host, IShaderPipelineProvider pipelines, bool strictDrawResources)
    {
        _host = host;
        _pipelines = pipelines;
        _strictDrawResources = strictDrawResources;
    }

    private readonly record struct DrawCall(string Name, RecordedOperation Operation, uint Count, uint InstanceCount, uint FirstInstance);

    // IndirectArgumentsAddress: the GPU reads the draw's counts from guest memory there, and
    // IndirectIndexLimit bounds an indexed draw's count as the bound index range does.
    // TriangleListIndices: nonzero when the draw was converted to an indexed triangle list of
    // that many indices (TriangulateFans), which EmitDraw then draws whatever the guest type.
    private readonly record struct DrawEmission(bool Indexed, int VertexOffset, uint FirstVertex, uint FirstInstance,
        ulong IndirectArgumentsAddress = 0, uint IndirectIndexLimit = uint.MaxValue, uint TriangleListIndices = 0);

    private readonly record struct IndexSource(bool Enabled, ulong Address, byte[]? HostData, ulong Size, GpuIndexType Type);

    [System.Runtime.CompilerServices.InlineArray(RenderingState.ColorAttachmentCapacity)]
    private struct ColorTargetStates
    {
        private ColorTargetState _element0;
    }

    private readonly Dictionary<ulong, bool> _pixelStoreDecisions = new();

    // A draw stored to memory from its fragment stage and, because the host defers it, the
    // render pass it drew in (_fragmentWritesPass, the host's pass serial) was left open.
    private bool _fragmentWritesPending;
    private long _fragmentWritesPass;

    // The buffer ranges those deferred stores may have written, or untracked when one of the
    // storing draws wrote through a device address or a storage image.
    private readonly List<(ulong Start, ulong End)> _fragmentWriteRanges = new();
    private bool _fragmentWritesUntracked;

    /// <summary>
    /// A consumer of earlier shader stores can follow: the guest flushed or invalidated its
    /// caches. Ends the render pass that holds deferred fragment-stage stores, if any.
    /// </summary>
    public void ReleaseShaderWrites() => ReleaseFragmentWrites();

    // Ends the pass a deferred fragment store was made in and records the barrier the storing
    // draw would have recorded. A pass the host has ended since needs neither: the encoder
    // boundary orders its stores, and a barrier now would land in whatever pass is open.
    private void ReleaseFragmentWrites()
    {
        if (!_fragmentWritesPending)
        {
            return;
        }

        _fragmentWritesPending = false;
        if (_host.RenderPassSerial != _fragmentWritesPass)
        {
            return;
        }

        _host.EndRendering("release");
        _host.ShaderWriteBarrier(GpuPipelineStage.FragmentShaderBit);
    }

    // Whether a deferred fragment store may have written the range: always when a storing
    // draw wrote memory no buffer range names.
    private bool FragmentWritesMayCover(ulong address, ulong size)
    {
        if (!_fragmentWritesPending)
        {
            return false;
        }

        if (_fragmentWritesUntracked || _host.RenderPassSerial != _fragmentWritesPass)
        {
            return true;
        }

        var end = address + size;
        foreach (var (start, stop) in _fragmentWriteRanges)
        {
            if (start < end && address < stop)
            {
                return true;
            }
        }

        return false;
    }

    // Notes what a draw whose fragment stores are deferred may write.
    private void NoteFragmentWrites(ShaderStageResources stage)
    {
        if (!_fragmentWritesPending || _host.RenderPassSerial != _fragmentWritesPass)
        {
            _fragmentWriteRanges.Clear();
            _fragmentWritesUntracked = false;
        }

        var program = stage.Program!;
        if (program.WritesDeviceAddresses || WritesStorageImage(program) || _fragmentWriteRanges.Count >= MaxFragmentWriteRanges)
        {
            _fragmentWritesUntracked = true;
            return;
        }

        var descriptors = stage.Resources.Buffers;
        for (var i = 0; i < program.Buffers.Length; i++)
        {
            if (!program.Buffers[i].Written || descriptors[i].Length < 4)
            {
                continue;
            }

            var descriptor = BufferDescriptorWords.From(descriptors[i]);
            if (descriptor.Footprint() is not { } footprint || descriptor.Address > ulong.MaxValue - footprint)
            {
                _fragmentWritesUntracked = true;
                return;
            }

            if (descriptor.Address != 0 && footprint != 0)
            {
                _fragmentWriteRanges.Add((descriptor.Address, descriptor.Address + footprint));
            }
        }
    }

    private const int MaxFragmentWriteRanges = 64;

    private struct DrawState
    {
        public ColorTargetStates Colors;
        public uint ColorCount;
        public DepthAttachmentState Depth;
        public bool PixelActive;
        public RenderingState Rendering;
        public GraphicsPrograms Programs;

        public static DrawState Create() => new() { PixelActive = true, Programs = null! };
    }

    private static ReadOnlySpan<ColorTargetState> BoundColors(ref DrawState state) =>
        ((ReadOnlySpan<ColorTargetState>)state.Colors)[..(int)state.ColorCount];

    public void DrawIndexed(ulong submitId, RegisterBanks banks, in DrawIndexedArguments arguments)
    {
        _drawMustExecute = arguments.MustExecute;
        using var meshTrace = MeshDrawTrace.Begin(submitId, banks.Shader.Vertex.ExportAddress, banks.Context.ShaderStages);
        try
        {
            if (arguments.IndirectArgumentsAddress != 0 &&
                (UsesMeshExecution(banks)
                    ? !CanDrawMeshIndirectOnGpu(banks, in arguments)
                    : !CanDrawIndirectOnGpu(banks, in arguments) || TriangulatesFans(banks.UserConfig)))
            {
                DrawIndexedWithCpuArguments(submitId, banks, in arguments);
                return;
            }

            DrawIndexedCore(submitId, banks, in arguments);
        }
        catch (Exception exception) when (meshTrace is not null)
        {
            meshTrace?.Complete("failed");
            if (MeshDrawTrace.Active) MeshDrawTrace.Write("failure", $"type={exception.GetType().Name} message={exception.Message}");
            throw;
        }
    }

    // The GPU reads the arguments of an indirect draw unless the draw is emulated from
    // its counts: strips (metadata clear quads), legacy primitives, 8-bit indices and
    // restart indices converted on the CPU.
    private static bool CanDrawIndirectOnGpu(RegisterBanks banks, in DrawIndexedArguments arguments)
    {
        switch ((GuestPrimitiveType)banks.UserConfig.PrimitiveType)
        {
            case GuestPrimitiveType.PointList:
            case GuestPrimitiveType.LineList:
            case GuestPrimitiveType.LineStrip:
            case GuestPrimitiveType.TriangleList:
            case GuestPrimitiveType.TriangleFan:
            case GuestPrimitiveType.Polygon:
            case GuestPrimitiveType.RectangleList:
            case GuestPrimitiveType.RectangleListLegacy:
                break;
            default:
                return false;
        }

        var index32 = (GuestIndexType)arguments.IndexTypeAndSize == GuestIndexType.Index32;
        if (!index32 && (GuestIndexType)arguments.IndexTypeAndSize != GuestIndexType.Index16)
        {
            return false;
        }

        if ((banks.UserConfig.PrimitiveResetControl & 0x1) != 0)
        {
            var indexMask = index32 ? uint.MaxValue : 0xFFFFu;
            if ((banks.Context.PrimitiveResetIndex & indexMask) != indexMask)
            {
                return false;
            }
        }

        return (ulong)arguments.IndexCount * (index32 ? 4ul : 2ul) <= MaxIndirectIndexBufferBytes;
    }

    private const ulong MaxIndirectIndexBufferBytes = 64ul << 20;

    private const ulong AutoIndirectArgumentsSize = 16;

    // List inputs ignore restart and use the learned index extent as their capacity.
    // Other mesh inputs keep the CPU path.
    private bool CanDrawMeshIndirectOnGpu(RegisterBanks banks, in DrawIndexedArguments arguments)
    {
        if (!_host.SupportsIndirectMeshDraw ||
            arguments.IndexCount is 0 or > IndirectMeshDrawRequest.MaximumIndexCapacity ||
            (GuestPrimitiveType)banks.UserConfig.PrimitiveType is not (GuestPrimitiveType.PointList or GuestPrimitiveType.TriangleList) ||
            (GuestIndexType)arguments.IndexTypeAndSize is not (GuestIndexType.Index16 or GuestIndexType.Index32))
            return false;
        var elementSize = (GuestIndexType)arguments.IndexTypeAndSize == GuestIndexType.Index16 ? 2ul : 4ul;
        var start = arguments.IndexAddress & ~3ul;
        var size = ((arguments.IndexAddress - start + arguments.IndexCount * elementSize) + 3) & ~3ul;
        if ((arguments.IndirectArgumentsAddress & 3) != 0 || (arguments.IndexAddress & (elementSize - 1)) != 0 ||
            start > ulong.MaxValue - size || arguments.IndirectArgumentsAddress > ulong.MaxValue - IndexedIndirectArgumentsSize)
            return false;
        return _host.ClampMappedSize(start, size) == size &&
            _host.ClampMappedSize(arguments.IndirectArgumentsAddress, IndexedIndirectArgumentsSize) == IndexedIndirectArgumentsSize;
    }

    // Reads the indirect arguments now and draws from them, as the interpreter would have.
    private void DrawIndexedWithCpuArguments(ulong submitId, RegisterBanks banks, in DrawIndexedArguments arguments)
    {
        Span<byte> bytes = stackalloc byte[(int)IndexedIndirectArgumentsSize];
        _host.NoteIndirectArgumentsReadOnCpu(arguments.IndirectArgumentsAddress, IndexedIndirectArgumentsSize);
        if (!_host.TryReadGuest(arguments.IndirectArgumentsAddress, bytes))
        {
            throw _host.Fatal($"The indirect draw arguments are unreadable: address=0x{arguments.IndirectArgumentsAddress:X16}.");
        }

        var words = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, uint>(bytes);
        var elementSize = (GuestIndexType)arguments.IndexTypeAndSize switch
        {
            GuestIndexType.Index16 => 2ul,
            GuestIndexType.Index32 => 4ul,
            _ => 1ul,
        };
        var resolved = arguments with
        {
            IndexCount = arguments.UnboundedIndexBuffer ? words[0] : Math.Min(words[0], arguments.IndexCount),
            InstanceCount = words[1],
            IndexAddress = arguments.IndexAddress + (words[2] * elementSize),
            BaseVertex = unchecked((int)words[3]),
            FirstInstance = words[4],
            IndirectArgumentsAddress = 0,
            UnboundedIndexBuffer = false,
        };
        if (resolved.IndexCount == 0 || resolved.InstanceCount == 0)
        {
            _host.ResetBindings();
            return;
        }

        DrawIndexedCore(submitId, banks, in resolved, arguments);
    }

    private void DrawIndexedCore(ulong submitId, RegisterBanks banks, in DrawIndexedArguments arguments,
        DrawIndexedArguments? originalIndirectIndexedDraw = null)
    {
        if (MeshDrawTrace.Active) MeshDrawTrace.Write("selection", $"indexed=true count={arguments.IndexCount}");
        using var profileScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.DrawExecutor);
        var traceSample = IndexedDrawTrace.Enabled && _host is ICommandStreamHost traceHost
            ? IndexedDrawTrace.Capture(arguments.IndexCount, arguments.IndexTypeAndSize, arguments.IndexAddress, traceHost.Memory.TryRead)
            : null;
        var tracePacket = arguments.PacketAddress;
        void TraceDisposition(string disposition)
        {
            IndexedDrawTrace.Write(traceSample, disposition == "recorded" ? "record-path-returned" : disposition, submitId, tracePacket);
            TraceDrawTargets(banks, disposition);
        }
        IndexedDrawTrace.Write(traceSample, "executor", submitId, tracePacket);
        if (!_host.IsRecording)
        {
            throw _host.Fatal("An indexed draw has no recording command buffer.");
        }

        if (arguments.OffsetSource == DrawOffsetSource.Packet && arguments.FirstInstance != 0)
        {
            throw _host.Fatal($"A packet indexed draw carries a first instance: firstInstance={arguments.FirstInstance}.");
        }

        _host.RunPendingOperations();
        var userConfig = banks.UserConfig;
        var shader = banks.Shader;
        _host.SetDebugInformation(RecordedOperation.DrawIndex, submitId, arguments.IndexCount, 0, 1, arguments.InstanceCount, arguments.IndexAddress);
        if (arguments.IndexCount == 0 || arguments.InstanceCount == 0)
        {
            TraceDisposition("zero-count");
            return;
        }

        if (ConsumesColorMetadataOperation(banks.Context) || TryDepthStencilCopy(banks.Context))
        {
            TraceDisposition("metadata-operation");
            _host.ResetBindings();
            return;
        }

        if (!HasValidVertexShader(shader))
        {
            TraceDisposition("missing-vertex-shader");
            return;
        }

        if (IsUnsupportedGeometryStage(banks,
            new TargetlessDrawArguments(submitId, true, arguments, default), originalIndirectIndexedDraw))
        {
            TraceDisposition("unsupported-geometry-stage");
            return;
        }

        if (RenderTrace.Enabled)
        {
            RenderTrace.Write(
                $"DrawIndexed submit={submitId} indexTypeAndSize=0x{arguments.IndexTypeAndSize:X8} count=0x{arguments.IndexCount:X8} " +
                $"indexAddress=0x{arguments.IndexAddress:X16} instances=0x{arguments.InstanceCount:X8} baseVertex=0x{(uint)arguments.BaseVertex:X8} firstInstance=0x{arguments.FirstInstance:X8}");
        }

        ValidateDrawRegisters(banks);
        if (!ResolveTopology(userConfig, out var topology))
        {
            TraceDisposition("no-topology");
            return;
        }

        var primitiveRestart = ResolvePrimitiveRestart(banks, topology, arguments.IndexTypeAndSize);
        GpuIndexType indexType;
        ulong indexSize;
        var expandIndex8 = false;
        switch ((GuestIndexType)arguments.IndexTypeAndSize)
        {
            case GuestIndexType.Index16:
                indexType = GpuIndexType.Uint16;
                indexSize = 2ul * arguments.IndexCount;
                break;
            case GuestIndexType.Index32:
                indexType = GpuIndexType.Uint32;
                indexSize = 4ul * arguments.IndexCount;
                break;
            case GuestIndexType.Index8:
                indexType = GpuIndexType.Uint16;
                indexSize = arguments.IndexCount;
                expandIndex8 = true;
                break;
            default:
                throw _host.Fatal($"The index type and size is unknown: indexTypeAndSize={arguments.IndexTypeAndSize}.");
        }

        var draw = new DrawCall("DrawIndexed", RecordedOperation.DrawIndex, arguments.IndexCount, arguments.InstanceCount, arguments.FirstInstance);
        byte[]? expandedIndices = null;
        var triangleListIndices = 0u;
        if (TriangulatesFans(userConfig))
        {
            // Every index is read so restarts split the fans where the hardware does; the
            // list's indices are the guest's own, so the vertex offset applies unchanged.
            var restartIndex = primitiveRestart
                ? banks.Context.PrimitiveResetIndex & IndexMask((GuestIndexType)arguments.IndexTypeAndSize)
                : (uint?)null;
            var vertices = ReadGuestIndices(arguments.IndexAddress, arguments.IndexCount, (GuestIndexType)arguments.IndexTypeAndSize);
            var list = TriangulateFans((GuestPrimitiveType)userConfig.PrimitiveType, vertices, restartIndex);
            if (list.Length == 0)
            {
                return;
            }

            expandedIndices = ListIndexBytes(list, wide: true);
            triangleListIndices = (uint)list.Length;
            indexType = GpuIndexType.Uint32;
            topology = GpuPrimitiveTopology.TriangleList;
            primitiveRestart = false;
        }
        else if (expandIndex8)
        {
            expandedIndices = ExpandIndex8(arguments.IndexAddress, arguments.IndexCount, primitiveRestart, banks.Context.PrimitiveResetIndex);
        }
        else if (primitiveRestart)
        {
            var indexMask = indexType == GpuIndexType.Uint16 ? 0xFFFFu : uint.MaxValue;
            var restartIndex = banks.Context.PrimitiveResetIndex & indexMask;
            if (restartIndex != indexMask)
            {
                expandedIndices = ConvertRestartIndices(arguments.IndexAddress, arguments.IndexCount, indexType, restartIndex);
                indexType = GpuIndexType.Uint32;
            }
        }

        var indexSource = new IndexSource(
            true,
            arguments.IndexAddress,
            expandedIndices,
            expandedIndices is null ? indexSize : (ulong)expandedIndices.Length,
            indexType);
        var state = DrawState.Create();
        if (!TryResolveDrawTargets(banks, in draw, ref state))
        {
            TraceDisposition("no-targets");
            _host.ResetBindings();
            return;
        }

        if (arguments.IndirectArgumentsAddress != 0 && !_host.TakesTargetlessIndirectDrawArguments &&
            state.ColorCount == 0 && !state.Depth.HasTarget)
        {
            // A targetless draw may be retained and replayed later; it needs its counts.
            DrawIndexedWithCpuArguments(submitId, banks, in arguments);
            return;
        }

        ResolveShaderPrograms(banks, ref state, indexed: true);
        if (arguments.IndirectArgumentsAddress != 0 && _host.TakesTargetlessIndirectDrawArguments &&
            MayRetainTargetlessDraw(ref state))
        {
            // A retained draw is replayed later; it needs its counts.
            DrawIndexedWithCpuArguments(submitId, banks, in arguments);
            return;
        }

        if (arguments.IndirectArgumentsAddress != 0 &&
            (!_host.TakesIndirectDrawArguments(
                    state.Programs.Vertex, arguments.IndirectArgumentsAddress, IndexedIndirectArgumentsSize) ||
                (IsRectangleList(banks.UserConfig.PrimitiveType) &&
                    !_host.TakesIndirectIndexedRectangleListArguments(
                        state.Programs.VertexInput,
                        arguments.IndirectArgumentsAddress,
                        IndexedIndirectArgumentsSize,
                        arguments.UnboundedIndexBuffer ? uint.MaxValue : arguments.IndexCount))))
        {
            // The backend runs this program as a mesh pipeline, or this rectangle list as
            // patches, whose launch is sized from the very counts the GPU would have read;
            // they are read here instead, before any pass is open, so the draw still happens.
            DrawIndexedWithCpuArguments(submitId, banks, in arguments);
            return;
        }

        if (!ApplyProgramAdaptations(banks, in draw, ref state, new TargetlessDrawArguments(submitId, true, arguments, default)))
        {
            TraceDisposition("adapted");
            _host.ResetBindings();
            return;
        }

        TraceDrawState(submitId, banks, in draw, in state);
        var indirect = arguments.OffsetSource == DrawOffsetSource.IndirectArguments;
        var vertexOffset = indirect
            ? arguments.BaseVertex
            : ResolveVertexOffset(userConfig.IndexOffset, state.Programs.VertexInput) + arguments.BaseVertex;
        var emission = new DrawEmission(
            true,
            vertexOffset,
            0,
            indirect ? arguments.FirstInstance : ResolveInstanceOffset(state.Programs.VertexInput),
            arguments.IndirectArgumentsAddress,
            arguments.UnboundedIndexBuffer ? uint.MaxValue : arguments.IndexCount,
            triangleListIndices);
        var disposition = RecordDraw(submitId, banks, in draw, ref state, topology, in emission, in indexSource, primitiveRestart,
            setBindDebug: true, setAutoDebug: false, indexedTrace: traceSample, packetAddress: tracePacket);
        TraceDisposition(disposition);
        _host.ResetBindings();
    }

    public void DrawAuto(ulong submitId, RegisterBanks banks, in DrawAutoArguments arguments)
    {
        _drawMustExecute = arguments.MustExecute;
        using var meshTrace = MeshDrawTrace.Begin(submitId, banks.Shader.Vertex.ExportAddress, banks.Context.ShaderStages);
        try
        {
            if (arguments.IndirectArgumentsAddress != 0 &&
                (!CanDrawAutoIndirectOnGpu(banks, in arguments) || TriangulatesFans(banks.UserConfig)))
            {
                DrawAutoWithCpuArguments(submitId, banks, in arguments);
                return;
            }

            DrawAutoCore(submitId, banks, in arguments);
        }
        catch (Exception exception) when (meshTrace is not null)
        {
            meshTrace?.Complete("failed");
            if (MeshDrawTrace.Active) MeshDrawTrace.Write("failure", $"type={exception.GetType().Name} message={exception.Message}");
            throw;
        }
    }

    // A non-indexed draw takes its arguments on the GPU for the same primitives as an
    // indexed one; strips (metadata clear quads), legacy primitives and misaligned
    // arguments are drawn from their counts.
    private static bool CanDrawAutoIndirectOnGpu(RegisterBanks banks, in DrawAutoArguments arguments)
    {
        if ((arguments.IndirectArgumentsAddress & 0x3) != 0)
        {
            return false;
        }

        switch ((GuestPrimitiveType)banks.UserConfig.PrimitiveType)
        {
            case GuestPrimitiveType.PointList:
            case GuestPrimitiveType.LineList:
            case GuestPrimitiveType.LineStrip:
            case GuestPrimitiveType.TriangleList:
            case GuestPrimitiveType.TriangleFan:
            case GuestPrimitiveType.Polygon:
            case GuestPrimitiveType.RectangleList:
            case GuestPrimitiveType.RectangleListLegacy:
                return true;
            default:
                return false;
        }
    }

    // Reads the indirect arguments now and draws from them, as the interpreter would have.
    private void DrawAutoWithCpuArguments(ulong submitId, RegisterBanks banks, in DrawAutoArguments arguments)
    {
        Span<byte> bytes = stackalloc byte[(int)AutoIndirectArgumentsSize];
        _host.NoteIndirectArgumentsReadOnCpu(arguments.IndirectArgumentsAddress, AutoIndirectArgumentsSize);
        if (!_host.TryReadGuest(arguments.IndirectArgumentsAddress, bytes))
        {
            throw _host.Fatal($"The indirect draw arguments are unreadable: address=0x{arguments.IndirectArgumentsAddress:X16}.");
        }

        var words = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, uint>(bytes);
        var resolved = arguments with
        {
            VertexCount = words[0],
            InstanceCount = words[1],
            FirstVertex = words[2],
            FirstInstance = words[3],
            IndirectArgumentsAddress = 0,
        };
        if (resolved.VertexCount == 0 || resolved.InstanceCount == 0)
        {
            _host.ResetBindings();
            return;
        }

        DrawAutoCore(submitId, banks, in resolved);
    }

    private void DrawAutoCore(ulong submitId, RegisterBanks banks, in DrawAutoArguments arguments)
    {
        if (MeshDrawTrace.Active)
            MeshDrawTrace.Write("selection", $"count={arguments.VertexCount} instances={arguments.InstanceCount} firstVertex={arguments.FirstVertex} firstInstance={arguments.FirstInstance} primitive={banks.UserConfig.PrimitiveType} geometry=0x{banks.Shader.Vertex.GeometryAddress:X16} hull=0x{banks.Shader.Vertex.HullAddress:X16} local=0x{banks.Shader.Vertex.LocalAddress:X16}");
        using var profileScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.DrawExecutor);
        if (!_host.IsRecording)
        {
            throw _host.Fatal("An automatic draw has no recording command buffer.");
        }

        if (arguments.OffsetSource == DrawOffsetSource.Packet && arguments.FirstInstance != 0)
        {
            throw _host.Fatal($"A packet automatic draw carries a first instance: firstInstance={arguments.FirstInstance}.");
        }

        _host.RunPendingOperations();
        var userConfig = banks.UserConfig;
        var shader = banks.Shader;
        _host.SetDebugInformation(RecordedOperation.DrawIndexAuto, submitId, arguments.VertexCount, 0, arguments.FirstVertex, arguments.InstanceCount, arguments.FirstInstance);
        if (arguments.VertexCount == 0 || arguments.InstanceCount == 0)
        {
            TraceDrawTargets(banks, "zero-count");
            return;
        }

        if (ConsumesColorMetadataOperation(banks.Context) || TryDepthStencilCopy(banks.Context))
        {
            TraceDrawTargets(banks, "metadata-operation");
            _host.ResetBindings();
            return;
        }

        if (!HasValidVertexShader(shader))
        {
            TraceDrawTargets(banks, "missing-vertex-shader");
            return;
        }

        if (IsUnsupportedGeometryStage(banks, new TargetlessDrawArguments(submitId, false, default, arguments)))
        {
            TraceDrawTargets(banks, "unsupported-geometry-stage");
            return;
        }

        if (RenderTrace.Enabled)
        {
            RenderTrace.Write(
                $"DrawAuto submit={submitId} count=0x{arguments.VertexCount:X8} instances=0x{arguments.InstanceCount:X8} " +
                $"firstVertex=0x{arguments.FirstVertex:X8} firstInstance=0x{arguments.FirstInstance:X8}");
        }

        ValidateDrawRegisters(banks);
        var draw = new DrawCall("DrawAuto", RecordedOperation.DrawIndexAuto, arguments.VertexCount, arguments.InstanceCount, arguments.FirstInstance);
        var state = DrawState.Create();
        if (!TryResolveDrawTargets(banks, in draw, ref state))
        {
            TraceDrawTargets(banks, "no-targets");
            _host.ResetBindings();
            return;
        }

        if (!ResolveTopology(userConfig, out var topology))
        {
            TraceDrawDisposition(banks, in draw, "no-primitive-topology");
            TraceDrawTargets(banks, "no-topology");
            _host.ResetBindings();
            return;
        }

        ResolveShaderPrograms(banks, ref state);
        if (arguments.IndirectArgumentsAddress != 0 &&
            (_host.TakesTargetlessIndirectDrawArguments
                ? MayRetainTargetlessDraw(ref state)
                : state.ColorCount == 0 && !state.Depth.HasTarget))
        {
            // A retained draw is replayed later; it needs its counts.
            DrawAutoWithCpuArguments(submitId, banks, in arguments);
            return;
        }

        if (arguments.IndirectArgumentsAddress != 0 &&
            (!_host.TakesIndirectDrawArguments(
                    state.Programs.Vertex, arguments.IndirectArgumentsAddress, AutoIndirectArgumentsSize) ||
                (IsRectangleList(banks.UserConfig.PrimitiveType) &&
                    !_host.TakesIndirectAutoRectangleListArguments(
                        state.Programs.VertexInput, arguments.IndirectArgumentsAddress, AutoIndirectArgumentsSize))))
        {
            // A mesh pipeline or a rectangle list drawn as patches may be sized from the
            // counts, so they are read here, before any pass is open.
            DrawAutoWithCpuArguments(submitId, banks, in arguments);
            return;
        }

        var vertexInput = state.Programs.VertexInput;
        var pixelInput = state.Programs.PixelInput;
        var rectangleList = topology == GpuPrimitiveTopology.PatchList;
        if (rectangleList && vertexInput.Buffers.Length == 0 && vertexInput.Stage.Program?.ParameterExportMask == 0 && pixelInput.InputCount != 0)
        {
            if (RenderTrace.Enabled)
            {
                RenderTrace.Write(
                    $"DrawAuto skipped a rectangle list with no vertex parameter exports and pixel inputs: pixelInputs={pixelInput.InputCount} " +
                    $"pixel=0x{shader.Pixel.Address:X16} export=0x{shader.Vertex.ExportAddress:X16} geometry=0x{shader.Vertex.GeometryAddress:X16}");
            }

            TraceDrawTargets(banks, "rectangle-list-without-exports");
            _host.ResetBindings();
            return;
        }

        if (!ApplyProgramAdaptations(banks, in draw, ref state, new TargetlessDrawArguments(submitId, false, default, arguments)))
        {
            TraceDrawTargets(banks, "adapted");
            _host.ResetBindings();
            return;
        }

        TraceDrawState(submitId, banks, in draw, in state);
        var indirect = arguments.OffsetSource == DrawOffsetSource.IndirectArguments;
        var vertexOffset = indirect
            ? (int)arguments.FirstVertex
            : ResolveVertexOffset(userConfig.IndexOffset, vertexInput) + (int)arguments.FirstVertex;
        var firstInstance = indirect ? arguments.FirstInstance : ResolveInstanceOffset(vertexInput);
        if (TriangulatesFans(userConfig))
        {
            // The vertices are numbered from zero and the first vertex becomes the indexed
            // draw's vertex offset, so every vertex keeps its vertex id.
            if (arguments.IndirectArgumentsAddress != 0)
            {
                throw _host.Fatal("A triangle fan to convert to a triangle list still has its counts in guest memory.");
            }

            var vertices = new uint[arguments.VertexCount];
            for (var index = 0u; index < arguments.VertexCount; index++)
            {
                vertices[index] = index;
            }

            var list = TriangulateFans((GuestPrimitiveType)userConfig.PrimitiveType, vertices, restartIndex: null);
            if (list.Length == 0)
            {
                _host.ResetBindings();
                return;
            }

            var wide = arguments.VertexCount > ushort.MaxValue;
            var bytes = ListIndexBytes(list, wide);
            var listSource = new IndexSource(true, 0, bytes, (ulong)bytes.Length, wide ? GpuIndexType.Uint32 : GpuIndexType.Uint16);
            var listEmission = new DrawEmission(true, vertexOffset, 0, firstInstance, TriangleListIndices: (uint)list.Length);
            RecordDraw(submitId, banks, in draw, ref state, GpuPrimitiveTopology.TriangleList, in listEmission, in listSource,
                primitiveRestart: false, setBindDebug: false, setAutoDebug: true);
            _host.ResetBindings();
            return;
        }

        var emission = new DrawEmission(
            false,
            0,
            (uint)vertexOffset,
            firstInstance,
            arguments.IndirectArgumentsAddress);
        var disposition = RecordDraw(submitId, banks, in draw, ref state, topology, in emission, default, primitiveRestart: false, setBindDebug: false, setAutoDebug: true);
        TraceDrawTargets(banks, disposition);
        _host.ResetBindings();
    }

    // Triangle fans, polygons and legacy quad lists, which a host without fans draws as
    // triangle lists (IRenderHost.DrawsTriangleFans).
    private bool TriangulatesFans(UserConfigRegisters userConfig) =>
        !_host.DrawsTriangleFans &&
        (GuestPrimitiveType)userConfig.PrimitiveType is
            GuestPrimitiveType.TriangleFan or GuestPrimitiveType.Polygon or GuestPrimitiveType.QuadListLegacy;

    // The triangle list the primitive assembler makes of these vertices, in its vertex order:
    // fan triangle i is (v[i+1], v[i+2], v[0]), the order DI_PT_TRIFAN assembles (the
    // first-vertex provoking vertex is v[i+1], as radv's VK fans report it), and a restart
    // index ends the fan, the next index starting a new one. A polygon is one fan. Each
    // quad of a quad list is (q0, q1, q2) and (q0, q2, q3), both taking q0 as their
    // provoking vertex; a count that is not a whole number of quads is refused.
    private uint[] TriangulateFans(GuestPrimitiveType type, uint[] vertices, uint? restartIndex)
    {
        if (type == GuestPrimitiveType.QuadListLegacy)
        {
            if ((vertices.Length & 0x3) != 0)
            {
                throw _host.Fatal($"A legacy quad list count is not a multiple of four: count={vertices.Length}.");
            }

            var quads = new uint[checked((vertices.Length / 4) * 6)];
            for (int quad = 0, output = 0; quad < vertices.Length; quad += 4, output += 6)
            {
                quads[output] = vertices[quad];
                quads[output + 1] = vertices[quad + 1];
                quads[output + 2] = vertices[quad + 2];
                quads[output + 3] = vertices[quad];
                quads[output + 4] = vertices[quad + 2];
                quads[output + 5] = vertices[quad + 3];
            }

            return quads;
        }

        var triangles = new List<uint>(Math.Max(0, (vertices.Length - 2) * 3));
        var hub = 0;
        for (var index = 0; index < vertices.Length; index++)
        {
            if (restartIndex is { } restart && vertices[index] == restart)
            {
                hub = index + 1;
                continue;
            }

            if (index - hub >= 2)
            {
                triangles.Add(vertices[index - 1]);
                triangles.Add(vertices[index]);
                triangles.Add(vertices[hub]);
            }
        }

        return [.. triangles];
    }

    private byte[] ListIndexBytes(uint[] list, bool wide)
    {
        var bytes = new byte[checked(list.Length * (wide ? sizeof(uint) : sizeof(ushort)))];
        for (var index = 0; index < list.Length; index++)
        {
            var value = list[index];
            if (wide)
            {
                // Metal reserves the all-ones index for primitive restart.
                if (value == uint.MaxValue)
                {
                    throw _host.Fatal("A triangle fan converted to a triangle list has a 32-bit vertex index of 0xFFFFFFFF.");
                }

                BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(index * sizeof(uint)), value);
            }
            else
            {
                BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(index * sizeof(ushort)), checked((ushort)value));
            }
        }

        return bytes;
    }

    private static uint IndexMask(GuestIndexType type) => type switch
    {
        GuestIndexType.Index8 => 0xFFu,
        GuestIndexType.Index16 => 0xFFFFu,
        _ => uint.MaxValue,
    };

    // The draw's guest indices, widened to 32 bits.
    private uint[] ReadGuestIndices(ulong indexAddress, uint indexCount, GuestIndexType type)
    {
        var stride = type switch
        {
            GuestIndexType.Index8 => 1,
            GuestIndexType.Index16 => 2,
            GuestIndexType.Index32 => 4,
            _ => throw _host.Fatal($"The index type and size is unknown: indexTypeAndSize={(uint)type}."),
        };
        var source = new byte[checked((int)indexCount * stride)];
        if (indexAddress == 0 || !_host.TryReadGuest(indexAddress, source))
        {
            throw _host.Fatal($"The fan index data is unreadable: address=0x{indexAddress:X16} count={indexCount}.");
        }

        var indices = new uint[indexCount];
        for (var index = 0; index < indices.Length; index++)
        {
            indices[index] = stride switch
            {
                1 => source[index],
                2 => BinaryPrimitives.ReadUInt16LittleEndian(source.AsSpan(index * 2)),
                _ => BinaryPrimitives.ReadUInt32LittleEndian(source.AsSpan(index * 4)),
            };
        }

        return indices;
    }

    private byte[] ConvertRestartIndices(ulong indexAddress, uint indexCount, GpuIndexType indexType, uint restartIndex)
    {
        var sourceStride = indexType == GpuIndexType.Uint16 ? sizeof(ushort) : sizeof(uint);
        var source = new byte[checked((int)indexCount * sourceStride)];
        if (indexAddress == 0 || !_host.TryReadGuest(indexAddress, source))
        {
            throw _host.Fatal($"The restart index data is unreadable: address=0x{indexAddress:X16} count={indexCount}.");
        }

        // Widen 16-bit indices so their maximum value remains an ordinary vertex.
        var converted = new byte[checked((int)indexCount * sizeof(uint))];
        for (var index = 0; index < (int)indexCount; index++)
        {
            var value = sourceStride == sizeof(ushort)
                ? BinaryPrimitives.ReadUInt16LittleEndian(source.AsSpan(index * sourceStride))
                : BinaryPrimitives.ReadUInt32LittleEndian(source.AsSpan(index * sourceStride));
            if (value == uint.MaxValue)
            {
                throw _host.Fatal("A custom restart index conflicts with a 32-bit vertex index of 0xFFFFFFFF.");
            }

            BinaryPrimitives.WriteUInt32LittleEndian(converted.AsSpan(index * sizeof(uint)),
                value == restartIndex ? uint.MaxValue : value);
        }

        return converted;
    }

    private byte[] ExpandIndex8(ulong indexAddress, uint indexCount, bool primitiveRestart, uint restartIndex)
    {
        if (indexAddress == 0)
        {
            throw _host.Fatal($"An 8-bit index draw has no index address: count={indexCount}.");
        }

        var source = new byte[indexCount];
        if (!_host.TryReadGuest(indexAddress, source))
        {
            throw _host.Fatal($"The 8-bit index data is unreadable: address=0x{indexAddress:X16} count={indexCount}.");
        }

        var expanded = new byte[indexCount * 2];
        for (var i = 0; i < indexCount; i++)
        {
            var value = source[i];
            ushort index = primitiveRestart && value == (restartIndex & 0xFF) ? (ushort)0xFFFF : value;
            expanded[i * 2] = (byte)index;
            expanded[(i * 2) + 1] = (byte)(index >> 8);
        }

        return expanded;
    }

    // Vertex offset: the index offset register, or the embedded fetch scalar when that register is zero.
    public static int ResolveVertexOffset(uint indexOffset, VertexInputInfo vertexInput)
    {
        if (indexOffset != 0 || !vertexInput.FetchEmbedded)
        {
            return (int)indexOffset;
        }

        var program = vertexInput.Stage.Program ?? throw GpuFatal.Fatal("The embedded fetch vertex input has no program.");
        return (int)ReadUserDataScalar(program, vertexInput.Stage.Resources, program.VertexOffsetScalarRegister);
    }

    public static uint ResolveInstanceOffset(VertexInputInfo vertexInput)
    {
        if (!vertexInput.FetchEmbedded)
        {
            return 0;
        }

        var program = vertexInput.Stage.Program ?? throw GpuFatal.Fatal("The embedded fetch vertex input has no program.");
        return ReadUserDataScalar(program, vertexInput.Stage.Resources, program.InstanceOffsetScalarRegister);
    }

    private static uint ReadUserDataScalar(ShaderProgramInfo program, ResourceSnapshot resources, int scalarRegister)
    {
        if (scalarRegister >= (int)program.UserDataBase)
        {
            var index = (uint)scalarRegister - program.UserDataBase;
            if (index < resources.UserData.Length)
            {
                return resources.UserData[index];
            }
        }

        return 0;
    }

    private static bool HasValidVertexShader(ShaderProgramRegisters shader) => shader.Vertex.ExportAddress != 0;

    // Records active color-target bases with the draw's outcome for the image-clear trace.
    private void TraceDrawTargets(RegisterBanks banks, string disposition)
    {
        if (!Images.ImageClearTrace.Enabled) return;
        var context = banks.Context;
        for (var slot = 0u; slot < ContextRegisters.ColorTargetCount; slot++)
        {
            if (context.RenderTargetMaskForSlot(slot) == 0) continue;
            var address = context.ColorTargets[slot].BaseAddress;
            if (address != 0)
                _host.TraceDrawTarget(disposition, address, banks.Shader.Pixel.Address, banks.Shader.Vertex.ExportAddress);
        }
    }

    private static bool IsKnownGeometryOutputPrimitiveType(uint value) => value <= 4;

    // Only the plain vertex path and the primitive-shader vertex path with default geometry state run.
    private bool IsUnsupportedGeometryStage(RegisterBanks banks, in TargetlessDrawArguments draw,
        DrawIndexedArguments? originalIndirectIndexedDraw = null)
    {
        var context = banks.Context;
        var shaderInterface = context.ShaderInterface;
        var vertex = banks.Shader.Vertex;
        var stages = context.ShaderStages;
        var autoDraw = !draw.Indexed;
        if (UsesMeshExecution(banks))
        {
            if (MeshDrawTrace.Active) MeshDrawTrace.Write("gate", "accepted=true");
            return false;
        }
        var primitiveShaderVertexPath =
            IsPrimitiveShaderStageMask(stages) && vertex.ExportAddress != 0 &&
            (vertex.GeometryAddress == 0 ||
             (shaderInterface.GeometryMaxVerticesOut == 0 && IsKnownGeometryOutputPrimitiveType(shaderInterface.GeometryOutputPrimitiveType)));
        // A backend with mesh pipelines runs the merged ES/GS stage as it really is, so the
        // NGG setup registers that defeat the per-vertex path stop being rejections: the
        // primitive shader itself consumes the subgroup control and the output budget. That
        // includes a draw with the geometry stage enabled: Gen5 has no separate geometry
        // program, the GS half is the continuation of the merged program at the export
        // address, so a stored geometry address or output budget is no reason to refuse it.
        var primitiveShaderMeshPath =
            _pipelines.SupportsPrimitiveShaderMeshPipelines &&
            vertex.ExportAddress != 0 &&
            VgtShaderStages.SelectsPrimitiveShaderOnly(stages);
        var unsupportedStageMask = stages != 0 && !IsPrimitiveShaderStageMask(stages) && !primitiveShaderMeshPath;
        var unsupportedGeometryStage = vertex.ExportAddress != 0 && vertex.GeometryAddress != 0 &&
            !primitiveShaderVertexPath && !primitiveShaderMeshPath;
        var geometryRegisters = !primitiveShaderVertexPath && !primitiveShaderMeshPath &&
            ((shaderInterface.PrimitiveShaderSubgroupControl != 0 && shaderInterface.PrimitiveShaderSubgroupControl != 1) ||
             shaderInterface.GeometryMaxVerticesOut != 0 ||
             !IsKnownGeometryOutputPrimitiveType(shaderInterface.GeometryOutputPrimitiveType) ||
             shaderInterface.MaxOutputPerSubgroup > MaxOutputPerSubgroupLimit);
        if (!unsupportedStageMask && !unsupportedGeometryStage && !geometryRegisters)
        {
            return false;
        }

        _pipelines.DumpRejectedGraphics(banks, autoDraw);
        if (Interlocked.Exchange(ref _geometryWarningShown, 1) == 0)
        {
            Console.Error.WriteLine(
                "Warning: the title uses unsupported graphics pipelines; some draw calls were skipped. " +
                $"autoDraw={autoDraw} meshSupported={_pipelines.MeshShadersSupported} " +
                $"stages=0x{stages:X8} subgroup=0x{shaderInterface.PrimitiveShaderSubgroupControl:X8} " +
                $"maxOutput=0x{shaderInterface.MaxOutputPerSubgroup:X8} " +
                $"maxVerticesOut=0x{shaderInterface.GeometryMaxVerticesOut:X8} " +
                $"outputPrimitive=0x{shaderInterface.GeometryOutputPrimitiveType:X8} " +
                $"export=0x{vertex.ExportAddress:X16} geometry=0x{vertex.GeometryAddress:X16} " +
                $"hull=0x{vertex.HullAddress:X16} local=0x{vertex.LocalAddress:X16} " +
                $"unsupportedMask={unsupportedStageMask} unsupportedGeometry={unsupportedGeometryStage} " +
                $"unsupportedRegisters={geometryRegisters}.");
        }

        if (_dumpSkippedGraphics)
            _pipelines.RecordUnsupportedGraphics(vertex, context, banks.UserConfig, in draw, originalIndirectIndexedDraw);

        if (RenderTrace.Enabled && RenderTrace.GeometryStageSkip())
        {
            var geometryControl = banks.UserConfig.GeometryEngineControl;
            RenderTrace.Write(
                $"Skipping an unsupported geometry stage draw: stages=0x{stages:X8} primitiveGroup=0x{geometryControl.PrimitiveGroupSize:X4} " +
                $"vertexGroup=0x{geometryControl.VertexGroupSize:X4} subgroupControl=0x{shaderInterface.PrimitiveShaderSubgroupControl:X8} " +
                $"maxOutput=0x{shaderInterface.MaxOutputPerSubgroup:X8} maxVerticesOut=0x{shaderInterface.GeometryMaxVerticesOut:X8} " +
                $"outputPrimitive=0x{shaderInterface.GeometryOutputPrimitiveType:X8} export=0x{vertex.ExportAddress:X16} geometry=0x{vertex.GeometryAddress:X16}");
        }

        if (MeshDrawTrace.Active) MeshDrawTrace.Write("gate", $"accepted=false auto={autoDraw} supported={_pipelines.MeshShadersSupported}");
        return true;
    }

    private static bool PixelShaderHasDepthOrCoverageSideEffects(ShaderInterfaceRegisters shaderInterface)
    {
        var control = shaderInterface.DepthShaderControl;
        // The export format describes data; it does not enable shader execution.
        return control.KillEnable || control.DepthExportEnable || control.StencilExportEnable ||
               control.MaskExportEnable || control.DualExportEnable || control.ExecuteOnNoop;
    }

    private static bool HasActivePixelShader(RegisterBanks banks)
    {
        var context = banks.Context;
        var shaderInterface = context.ShaderInterface;
        var hasColorOutput = (context.RenderTargetMask & shaderInterface.ColorShaderMask) != 0;
        return banks.Shader.Pixel.Address != 0 && (hasColorOutput || PixelShaderHasDepthOrCoverageSideEffects(shaderInterface));
    }

    private const byte ColorModeEliminateFastClear = 2;
    private const byte ColorModeResolve = 3;
    private const byte ColorModeFmaskDecompress = 5;
    private const byte ColorModeDccDecompress = 6;

    // These modes run color metadata operations; the shader output is not a normal draw.
    public static bool ConsumesColorMetadataOperation(ContextRegisters context)
    {
        var mode = context.ColorControl.Mode;
        return mode is ColorModeEliminateFastClear or ColorModeFmaskDecompress or ColorModeDccDecompress;
    }

    // DI_PT_2D_RECTANGLE (7) and DI_PT_RECTLIST (17) are both rectangle lists: three
    // vertices per rectangle, the fourth corner completed by the rasterizer. The AGC
    // names them kRectList and kRectListLegacy, and they draw through the same path.
    private static bool IsRectangleList(uint primitiveType) =>
        (GuestPrimitiveType)primitiveType is GuestPrimitiveType.RectangleList or GuestPrimitiveType.RectangleListLegacy;

    public bool ResolveTopology(UserConfigRegisters userConfig, out GpuPrimitiveTopology topology)
    {
        topology = GpuPrimitiveTopology.PointList;
        switch ((GuestPrimitiveType)userConfig.PrimitiveType)
        {
            case GuestPrimitiveType.None:
                return false;
            case GuestPrimitiveType.PointList:
                topology = GpuPrimitiveTopology.PointList;
                break;
            case GuestPrimitiveType.LineList:
                topology = GpuPrimitiveTopology.LineList;
                break;
            case GuestPrimitiveType.LineStrip:
                topology = GpuPrimitiveTopology.LineStrip;
                break;
            case GuestPrimitiveType.TriangleList:
                topology = GpuPrimitiveTopology.TriangleList;
                break;
            case GuestPrimitiveType.TriangleFan:
            case GuestPrimitiveType.Polygon:
                topology = GpuPrimitiveTopology.TriangleFan;
                break;
            case GuestPrimitiveType.TriangleStrip:
                topology = GpuPrimitiveTopology.TriangleStrip;
                break;
            case GuestPrimitiveType.RectangleList:
            case GuestPrimitiveType.RectangleListLegacy:
                topology = GpuPrimitiveTopology.PatchList;
                break;
            case GuestPrimitiveType.QuadListLegacy:
                topology = GpuPrimitiveTopology.TriangleFan;
                break;
            default:
                throw _host.Fatal($"The primitive type is unknown: primitiveType={userConfig.PrimitiveType}.");
        }

        return true;
    }

    public bool ResolvePrimitiveRestart(RegisterBanks banks, GpuPrimitiveTopology topology, uint indexTypeAndSize)
    {
        var userConfig = banks.UserConfig;
        var control = userConfig.PrimitiveResetControl;
        if ((control & ~0x3u) != 0)
        {
            throw _host.Fatal($"The primitive reset control has unsupported bits: control=0x{control:X8}.");
        }

        if ((control & 0x1) == 0)
        {
            return false;
        }

        switch ((GuestPrimitiveType)userConfig.PrimitiveType)
        {
            case GuestPrimitiveType.LineStrip:
            case GuestPrimitiveType.TriangleFan:
            case GuestPrimitiveType.TriangleStrip:
                break;
            default:
                return false;
        }

        if (topology is not (GpuPrimitiveTopology.LineStrip or GpuPrimitiveTopology.TriangleStrip or GpuPrimitiveTopology.TriangleFan))
        {
            return false;
        }

        var indexMask = (GuestIndexType)indexTypeAndSize switch
        {
            GuestIndexType.Index8 => 0xFFu,
            GuestIndexType.Index16 => 0xFFFFu,
            GuestIndexType.Index32 => 0xFFFF_FFFFu,
            _ => throw _host.Fatal($"The index type and size is unknown: indexTypeAndSize={indexTypeAndSize}."),
        };
        var resetIndex = banks.Context.PrimitiveResetIndex;
        if ((control & 0x2) != 0 && (resetIndex & ~indexMask) != 0)
        {
            return false;
        }

        return true;
    }

    private static bool RequiresMeshExecution(uint stages) =>
        (stages & 0x20u) != 0 || (stages & 0x0200203Fu) == 0x2000u;

    private bool UsesMeshExecution(RegisterBanks banks) =>
        _pipelines.MeshShadersSupported && RequiresMeshExecution(banks.Context.ShaderStages) &&
        banks.Shader.Vertex.ExportAddress != 0 &&
        banks.Shader.Vertex.HullAddress == 0 && banks.Shader.Vertex.LocalAddress == 0;

    private void ResolveShaderPrograms(RegisterBanks banks, ref DrawState state, bool indexed = false)
    {
        using var profileScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.DrawProgramResolution);
        var inputs = GraphicsProgramInputs.Create();
        Span<ColorComponentMap> targetExportMapping = inputs.Mapping;
        foreach (ref readonly var color in BoundColors(ref state))
        {
            targetExportMapping[(int)color.Slot] = color.Resolution.ExportMapping;
        }

        inputs.Indexed = indexed;
        inputs.PixelActive = state.PixelActive;
        inputs.TriangleList = AssemblesTriangleList(banks);
        inputs.DepthBound = state.Depth.HasTarget;
        inputs.DepthAspectBound = state.Depth.HasTarget &&
            (ViewFormatRules.FullAspects(state.Depth.Target.Target.Format) & GpuImageAspect.DepthBit) != 0;
        inputs.StencilAspectBound = state.Depth.HasTarget &&
            (ViewFormatRules.DepthAspects(state.Depth.Target.Target.Format) & GpuImageAspect.StencilBit) != 0;
        if (Prefetch is { } prefetch && prefetch.TryTakeGraphics(banks, in inputs, out var prefetched))
        {
            state.Programs = prefetched;
            return;
        }

        using var resolveScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ProgramSynchronousResolution);
        state.Programs = ResolveGraphicsPrograms(banks, in inputs);
    }

    private static void TraceDrawState(ulong submitId, RegisterBanks banks, in DrawCall draw, in DrawState state)
    {
        if (!RenderTrace.Enabled)
        {
            return;
        }

        var shader = banks.Shader;
        var colorAddress = state.ColorCount != 0 ? state.Colors[0].Resolution.BaseAddress : 0;
        var colorWidth = state.ColorCount != 0 ? state.Colors[0].Resolution.Extent.Width : 0;
        var colorHeight = state.ColorCount != 0 ? state.Colors[0].Resolution.Extent.Height : 0;
        var depth = state.Depth.HasTarget ? state.Depth.Target : default;
        ref readonly var viewport = ref banks.Context.ScreenViewport.Viewports[0];
        var clip = banks.Context.Clip;
        RenderTrace.Write(
            $"DrawState seq={RenderTrace.NextSequence()} submit={submitId} name={draw.Name} export=0x{shader.Vertex.ExportAddress:X16} " +
            $"pixel=0x{shader.Pixel.Address:X16} pixelActive={state.PixelActive} count={draw.Count} instances={draw.InstanceCount} " +
            $"colors={state.ColorCount} color=0x{colorAddress:X10}:{colorWidth}x{colorHeight} targetMask=0x{banks.Context.RenderTargetMask:X8} " +
            $"depth=0x{depth.Target.DepthAddress:X10}:{depth.Target.Width}x{depth.Target.Height}:{(int)depth.Target.Format} " +
            $"depthState={(depth.State.DepthTestEnabled ? 1 : 0)}/{(depth.State.DepthWriteEnabled ? 1 : 0)}/{(int)depth.State.DepthCompare} " +
            $"viewportTransform=0x{banks.Context.ScreenViewport.TransformControl:X8} directXClip={clip.DirectXClipSpace} " +
            $"clipDisabled={clip.ClipDisable} nearClipDisabled={clip.NearZClipDisable} farClipDisabled={clip.FarZClipDisable} " +
            $"zScale={viewport.ZScale:R} zOffset={viewport.ZOffset:R} zBounds={viewport.MinDepth:R}/{viewport.MaxDepth:R} " +
            $"hostDepthRange={viewport.ZOffset - (clip.DirectXClipSpace ? 0f : viewport.ZScale):R}/{viewport.ZScale + viewport.ZOffset:R}");
    }
}
