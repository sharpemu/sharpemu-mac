// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.Libs.VideoOut;
using SharpEmu.ShaderCompiler;
using WorkerScratch = SharpEmu.ShaderCompiler.WorkerScratch;
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
    private static int _tessellationWarningShown;

    private readonly IRenderHost _host;
    private readonly DrawResolution _drawResolution = new();
    private readonly IShaderPipelineProvider _pipelines;
    private readonly IResolvedCommandSink _recorder;
    private readonly RecordPool _pool;
    // Forming the host's Fatal method group allocated a delegate on every draw; one serves them all.
    private readonly Func<string, Exception> _fatal;
    private Func<string, Exception> HostFatal => WorkerScratch.Enabled ? _fatal : _host.Fatal;
    private readonly bool _dumpSkippedGraphics = Environment.GetEnvironmentVariable("SHARPEMU_DUMP_SKIPPED_GRAPHICS") == "1";

    public RenderExecutor(IRenderHost host, IShaderPipelineProvider pipelines)
        : this(host, pipelines, DefaultStrictDrawResources)
    {
    }

    internal RenderExecutor(IRenderHost host, IShaderPipelineProvider pipelines, bool strictDrawResources)
        : this(host, pipelines, new PendingGpuWrites(), new RecordPool(), strictDrawResources)
    {
    }

    private RenderExecutor(IRenderHost host, IShaderPipelineProvider pipelines, PendingGpuWrites pendingWrites, RecordPool pool, bool strictDrawResources)
        : this(host, pipelines, new RenderRecorder(host, pipelines, strictDrawResources, pendingWrites, pool), pendingWrites, pool)
    {
    }

    internal RenderExecutor(IRenderHost host, IShaderPipelineProvider pipelines, IResolvedCommandSink recorder, PendingGpuWrites? pendingWrites = null, RecordPool? pool = null)
    {
        _host = host;
        _fatal = host.Fatal;
        _pipelines = pipelines;
        _recorder = recorder;
        PendingWrites = pendingWrites ?? new PendingGpuWrites();
        _pool = pool ?? new RecordPool();
    }

    internal static bool DefaultStrictDrawResources => Environment.GetEnvironmentVariable("SHARPEMU_STRICT_COMPUTE") != "0";

    // The recorder that records each draw as it is resolved, on this thread (every host but the
    // Vulkan presenter's slice workers). Record-side state the stream asks about lives there.
    private RenderRecorder? DirectRecorder => _recorder as RenderRecorder;

    /// <summary>
    /// Whether the open render pass holds shader stores no barrier has ordered yet: fragment stores
    /// whose pass end is deferred, or vertex stores the draw's own barrier does not cover. A guest
    /// cache flush or invalidate in the pass has nothing else to make visible to later draws.
    /// </summary>
    public bool HasUnorderedStoresInOpenPass => DirectRecorder?.HasUnorderedStoresInOpenPass ?? false;

    /// <summary>
    /// A consumer of earlier shader stores can follow: the guest flushed or invalidated its
    /// caches. Ends the render pass that holds deferred fragment-stage stores, if any.
    /// </summary>
    public void ReleaseShaderWrites() => DirectRecorder?.ReleaseShaderWrites();

    // Host index data a record carries: a scratch buffer the next draw reuses when the record is
    // recorded at once, a copy of it otherwise.
    private byte[] RecordIndexData(byte[] scratch, int length) =>
        DirectRecorder is not null ? scratch : scratch.AsSpan(0, length).ToArray();

    private readonly Dictionary<ulong, bool> _pixelStoreDecisions = new();

    // The draw packet asked to be executed even when its pipeline is still compiling.
    private bool _drawMustExecute;

    public void DrawIndexed(ulong submitId, RegisterBanks banks, in DrawIndexedArguments arguments)
    {
        _drawMustExecute = arguments.MustExecute;
        using var meshTrace = MeshDrawTrace.Begin(submitId, banks.Shader.Vertex.ExportAddress, banks.Context.ShaderStages);
        try
        {
            if (arguments.IndirectArgumentsAddress != 0 &&
                (TracesIndirectArguments(banks) ||
                 (UsesMeshExecution(banks)
                    ? !CanDrawMeshIndirectOnGpu(banks, in arguments)
                    : !CanDrawIndirectOnGpu(banks, in arguments) || TriangulatesFans(banks.UserConfig))))
            {
                NoteCpuIndirectArguments(banks, in arguments);
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

    // The index range binds where it lies (a window or a guest view of guest memory), so its
    // size is not a copy. Ghost of Tsushima draws from a 330 MiB shared index pool with the
    // whole pool as each draw's range; at a 64 MiB cap every such draw read its arguments on
    // the CPU, which waits for the GPU work that wrote them, ~70 round trips a frame.
    private const ulong MaxIndirectIndexBufferBytes = 1ul << 30;

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

    // SHARPEMU_TRACE_INDIRECT_SHADER=<hex>: the indirect draws of the vertex program at that address
    // (export or geometry stage) read their arguments on the CPU and print them, so a count the
    // GPU writes can be followed from frame to frame. A diagnostic: each read waits for the GPU.
    private static readonly ulong TracedIndirectShader = ulong.TryParse(
        (Environment.GetEnvironmentVariable("SHARPEMU_TRACE_INDIRECT_SHADER") ?? string.Empty)
            .Replace("0x", string.Empty, StringComparison.OrdinalIgnoreCase),
        System.Globalization.NumberStyles.HexNumber, null, out var tracedShader) ? tracedShader : 0;

    private long _tracedIndirectDraws;

    private static bool TracesIndirectArguments(RegisterBanks banks) =>
        TracedIndirectShader != 0 &&
        (banks.Shader.Vertex.ExportAddress == TracedIndirectShader || banks.Shader.Vertex.GeometryAddress == TracedIndirectShader);

    private void TraceIndirectArguments(RegisterBanks banks, ulong submitId, ulong address, ReadOnlySpan<uint> words)
    {
        if (!TracesIndirectArguments(banks) || Interlocked.Increment(ref _tracedIndirectDraws) % 8 != 1)
            return;

        Console.Error.WriteLine(
            $"[GPU][INDIRECT] submit={submitId} args=0x{address:X16} words=[{string.Join(' ', words.ToArray().Select(word => word.ToString()))}]");
    }

    // Each distinct shape of indirect indexed draw whose arguments are read on the CPU, named once:
    // the read waits for the GPU work that wrote them.
    private readonly HashSet<(bool Mesh, uint Primitive, uint IndexType, bool Restart, bool Unbounded)> _cpuIndirectShapes = [];

    private void NoteCpuIndirectArguments(RegisterBanks banks, in DrawIndexedArguments arguments)
    {
        var shape = (UsesMeshExecution(banks), banks.UserConfig.PrimitiveType, arguments.IndexTypeAndSize,
            (banks.UserConfig.PrimitiveResetControl & 0x1) != 0, arguments.UnboundedIndexBuffer);
        if (_cpuIndirectShapes.Count < 32 && _cpuIndirectShapes.Add(shape))
        {
            Console.Error.WriteLine(
                $"[GPU][INFO] An indirect indexed draw reads its arguments on the CPU: mesh={shape.Item1} primitive={shape.Item2} " +
                $"index_type={shape.Item3} restart={shape.Item4} restart_index=0x{banks.Context.PrimitiveResetIndex:X} " +
                $"unbounded={shape.Item5} index_count={arguments.IndexCount} mesh_gpu={_host.SupportsIndirectMeshDraw}.");
        }
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
        TraceIndirectArguments(banks, submitId, arguments.IndirectArgumentsAddress, words);
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
            var skipped = _pool.RentDraw();
            skipped.ResetBindings = true;
            Record(skipped);
            return;
        }

        DrawIndexedCore(submitId, banks, in resolved, arguments);
    }

    private void DrawIndexedCore(ulong submitId, RegisterBanks banks, in DrawIndexedArguments arguments,
        DrawIndexedArguments? originalIndirectIndexedDraw = null)
    {
        if (MeshDrawTrace.Active) MeshDrawTrace.Write("selection", $"indexed=true count={arguments.IndexCount}");
        using var profileScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.DrawExecutor);
        var record = _pool.RentDraw();
        record.Announce = true;
        record.Debug = new DebugInformation(RecordedOperation.DrawIndex, submitId, arguments.IndexCount, 0, 1, arguments.InstanceCount, arguments.IndexAddress);
        record.SubmitId = submitId;
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
        if (arguments.OffsetSource == DrawOffsetSource.Packet && arguments.FirstInstance != 0)
        {
            throw _host.Fatal($"A packet indexed draw carries a first instance: firstInstance={arguments.FirstInstance}.");
        }

        var userConfig = banks.UserConfig;
        var shader = banks.Shader;
        if (arguments.IndexCount == 0 || arguments.InstanceCount == 0)
        {
            TraceDisposition("zero-count");
            Record(record);
            return;
        }

        if (ConsumesColorMetadataOperation(banks.Context) || TryResolveDepthStencilCopy(banks.Context, record))
        {
            TraceDisposition("metadata-operation");
            record.ResetBindings = true;
            Record(record);
            return;
        }

        if (HasValidVertexShader(shader) && IsEmulatedTessellation(banks, out var controlPoints, out var quadPatches))
        {
            // The host runs the hull stage as a kernel and draws the patches itself
            // (RenderExecutor.TessellationEmulation.cs); the record only announces the draw.
            Record(record);
            var tessellated = new TessellatedDraw(arguments.IndexCount, arguments.InstanceCount, arguments.FirstInstance,
                (int)userConfig.IndexOffset + arguments.BaseVertex, arguments.IndexAddress, arguments.IndexTypeAndSize, Indexed: true);
            DrawTessellated(submitId, banks, in tessellated, controlPoints, quadPatches);
            TraceDisposition("tessellated");
            return;
        }

        if (Pipelines.ShaderPipelineCache.IsMergedTessellationMask(banks.Context.ShaderStages))
        {
            if (IsSupportedTessellation(banks))
            {
                DrawTessellationIndexed(submitId, banks, arguments, record);
            }
            else Record(record);

            TraceDisposition("tessellation");
            return;
        }

        if (!HasValidVertexShader(shader))
        {
            TraceDisposition("missing-vertex-shader");
            Record(record);
            return;
        }

        if (IsUnsupportedGeometryStage(banks,
            new TargetlessDrawArguments(submitId, true, arguments, default), originalIndirectIndexedDraw))
        {
            TraceDisposition("unsupported-geometry-stage");
            Record(record);
            return;
        }

        if (RenderTrace.Enabled)
        {
            RenderTrace.Write(
                $"DrawIndexed submit={submitId} indexTypeAndSize=0x{arguments.IndexTypeAndSize:X8} count=0x{arguments.IndexCount:X8} " +
                $"indexAddress=0x{arguments.IndexAddress:X16} instances=0x{arguments.InstanceCount:X8} baseVertex=0x{(uint)arguments.BaseVertex:X8} firstInstance=0x{arguments.FirstInstance:X8}");
        }

        ValidateChangedDrawRegisters(banks);
        if (!ResolveTopology(userConfig, out var topology))
        {
            TraceDisposition("no-topology");
            Record(record);
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
        var expandedLength = 0;
        var triangleListIndices = 0u;
        if (TriangulatesFans(userConfig))
        {
            // Every index is read so restarts split the fans where the hardware does; the
            // list's indices are the guest's own, so the vertex offset applies unchanged.
            var restartIndex = primitiveRestart
                ? banks.Context.PrimitiveResetIndex & IndexMask((GuestIndexType)arguments.IndexTypeAndSize)
                : (uint?)null;
            var vertices = ReadGuestIndicesScratch(arguments.IndexAddress, arguments.IndexCount, (GuestIndexType)arguments.IndexTypeAndSize);
            var list = TriangulateFansScratch((GuestPrimitiveType)userConfig.PrimitiveType, vertices, restartIndex);
            if (list.Length == 0)
            {
                Record(record);
                return;
            }

            expandedIndices = ListIndexBytesScratch(list, wide: true, out expandedLength);
            triangleListIndices = (uint)list.Length;
            indexType = GpuIndexType.Uint32;
            topology = GpuPrimitiveTopology.TriangleList;
            primitiveRestart = false;
        }
        else if (expandIndex8)
        {
            expandedIndices = ExpandIndex8Scratch(arguments.IndexAddress, arguments.IndexCount, primitiveRestart, banks.Context.PrimitiveResetIndex, out expandedLength);
        }
        else if (primitiveRestart)
        {
            var indexMask = indexType == GpuIndexType.Uint16 ? 0xFFFFu : uint.MaxValue;
            var restartIndex = banks.Context.PrimitiveResetIndex & indexMask;
            if (restartIndex != indexMask)
            {
                expandedIndices = ConvertRestartIndicesScratch(arguments.IndexAddress, arguments.IndexCount, indexType, restartIndex, out expandedLength);
                indexType = GpuIndexType.Uint32;
            }
        }

        var indexSource = new IndexSource(
            true,
            arguments.IndexAddress,
            expandedIndices is null ? null : RecordIndexData(expandedIndices, expandedLength),
            expandedIndices is null ? indexSize : (ulong)expandedLength,
            indexType);
        ref var state = ref record.State;
        state.Begin(arguments.StartInstanceUserData);
        if (!TryResolveDrawTargets(banks, in draw, ref state, record))
        {
            TraceDisposition("no-targets");
            record.ResetBindings = true;
            Record(record);
            return;
        }

        record.TargetsResolved = true;
        if (arguments.IndirectArgumentsAddress != 0 && !_host.TakesTargetlessIndirectDrawArguments &&
            state.ColorCount == 0 && !state.Depth.HasTarget)
        {
            // A targetless draw may be retained and replayed later; it needs its counts.
            Record(record);
            DrawIndexedWithCpuArguments(submitId, banks, in arguments);
            return;
        }

        ResolveShaderPrograms(banks, ref state, indexed: true);
        if (arguments.IndirectArgumentsAddress != 0 && _host.TakesTargetlessIndirectDrawArguments &&
            MayRetainTargetlessDraw(ref state))
        {
            // A retained draw is replayed later; it needs its counts.
            Record(record);
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
            Record(record);
            DrawIndexedWithCpuArguments(submitId, banks, in arguments);
            return;
        }
        if (!ApplyProgramAdaptations(banks, in draw, ref state, new TargetlessDrawArguments(submitId, true, arguments, default), record))
        {
            TraceDisposition("adapted");
            record.ResetBindings = true;
            Record(record);
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
        record.IndexedTrace = traceSample;
        record.PacketAddress = tracePacket;
        FillDraw(record, banks, in draw, ref state, topology, in emission, in indexSource, primitiveRestart, setBindDebug: true, setAutoDebug: false);
        TraceDisposition("recorded");
        Record(record);
    }

    public void DrawAuto(ulong submitId, RegisterBanks banks, in DrawAutoArguments arguments)
    {
        _drawMustExecute = arguments.MustExecute;
        using var meshTrace = MeshDrawTrace.Begin(submitId, banks.Shader.Vertex.ExportAddress, banks.Context.ShaderStages);
        try
        {
            if (arguments.IndirectArgumentsAddress != 0 &&
                (TracesIndirectArguments(banks) ||
                 !CanDrawAutoIndirectOnGpu(banks, in arguments) || TriangulatesFans(banks.UserConfig)))
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
    // indexed one; strips (metadata clear quads), legacy primitives, misaligned or unmapped
    // arguments and natively tessellated draws are drawn from their counts.
    private bool CanDrawAutoIndirectOnGpu(RegisterBanks banks, in DrawAutoArguments arguments)
    {
        var argumentsAddress = arguments.IndirectArgumentsAddress;
        if (Pipelines.ShaderPipelineCache.IsMergedTessellationMask(banks.Context.ShaderStages) ||
            (argumentsAddress & 0x3) != 0 || argumentsAddress > ulong.MaxValue - AutoIndirectArgumentsSize ||
            _host.ClampMappedSize(argumentsAddress, AutoIndirectArgumentsSize) != AutoIndirectArgumentsSize)
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
        TraceIndirectArguments(banks, submitId, arguments.IndirectArgumentsAddress, words);
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
            var skipped = _pool.RentDraw();
            skipped.ResetBindings = true;
            Record(skipped);
            return;
        }

        DrawAutoCore(submitId, banks, in resolved);
    }

    private void DrawAutoCore(ulong submitId, RegisterBanks banks, in DrawAutoArguments arguments)
    {
        if (MeshDrawTrace.Active)
            MeshDrawTrace.Write("selection", $"count={arguments.VertexCount} instances={arguments.InstanceCount} firstVertex={arguments.FirstVertex} firstInstance={arguments.FirstInstance} primitive={banks.UserConfig.PrimitiveType} geometry=0x{banks.Shader.Vertex.GeometryAddress:X16} hull=0x{banks.Shader.Vertex.HullAddress:X16} local=0x{banks.Shader.Vertex.LocalAddress:X16}");
        using var profileScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.DrawExecutor);
        var record = _pool.RentDraw();
        record.Announce = true;
        record.Debug = new DebugInformation(RecordedOperation.DrawIndexAuto, submitId, arguments.VertexCount, 0, arguments.FirstVertex, arguments.InstanceCount, arguments.FirstInstance);
        record.SubmitId = submitId;
        if (arguments.OffsetSource == DrawOffsetSource.Packet && arguments.FirstInstance != 0)
        {
            throw _host.Fatal($"A packet automatic draw carries a first instance: firstInstance={arguments.FirstInstance}.");
        }

        var userConfig = banks.UserConfig;
        var shader = banks.Shader;
        if (arguments.VertexCount == 0 || arguments.InstanceCount == 0)
        {
            TraceDrawTargets(banks, "zero-count");
            Record(record);
            return;
        }

        if (ConsumesColorMetadataOperation(banks.Context) || TryResolveDepthStencilCopy(banks.Context, record))
        {
            TraceDrawTargets(banks, "metadata-operation");
            record.ResetBindings = true;
            Record(record);
            return;
        }

        if (HasValidVertexShader(shader) && IsEmulatedTessellation(banks, out var controlPoints, out var quadPatches))
        {
            if (arguments.IndirectArgumentsAddress != 0)
            {
                // The hull kernel's patches are counted from the draw's counts.
                _pool.Return(record);
                DrawAutoWithCpuArguments(submitId, banks, in arguments);
                return;
            }

            // The host runs the hull stage as a kernel and draws the patches itself
            // (RenderExecutor.TessellationEmulation.cs); the record only announces the draw.
            Record(record);
            var patchVertexOffset = (int)arguments.FirstVertex +
                (arguments.OffsetSource == DrawOffsetSource.IndirectArguments ? 0 : (int)userConfig.IndexOffset);
            var tessellated = new TessellatedDraw(arguments.VertexCount, arguments.InstanceCount, arguments.FirstInstance, patchVertexOffset);
            DrawTessellated(submitId, banks, in tessellated, controlPoints, quadPatches);
            TraceDrawTargets(banks, "tessellated");
            return;
        }

        if (Pipelines.ShaderPipelineCache.IsMergedTessellationMask(banks.Context.ShaderStages))
        {
            var tessellationVertexOffset = unchecked((int)arguments.FirstVertex +
                (arguments.OffsetSource == DrawOffsetSource.IndirectArguments ? 0 : (int)userConfig.IndexOffset));
            if (IsSupportedTessellation(banks))
            {
                DrawTessellation(submitId, banks, arguments.VertexCount, arguments.InstanceCount, 0, 0, tessellationVertexOffset, arguments.FirstInstance, record);
            }
            else Record(record);

            TraceDrawTargets(banks, "tessellation");
            return;
        }

        if (!HasValidVertexShader(shader))
        {
            TraceDrawTargets(banks, "missing-vertex-shader");
            Record(record);
            return;
        }

        if (IsUnsupportedGeometryStage(banks, new TargetlessDrawArguments(submitId, false, default, arguments)))
        {
            TraceDrawTargets(banks, "unsupported-geometry-stage");
            Record(record);
            return;
        }

        if (RenderTrace.Enabled)
        {
            RenderTrace.Write(
                $"DrawAuto submit={submitId} count=0x{arguments.VertexCount:X8} instances=0x{arguments.InstanceCount:X8} " +
                $"firstVertex=0x{arguments.FirstVertex:X8} firstInstance=0x{arguments.FirstInstance:X8}");
        }

        ValidateChangedDrawRegisters(banks);
        var draw = new DrawCall("DrawAuto", RecordedOperation.DrawIndexAuto, arguments.VertexCount, arguments.InstanceCount, arguments.FirstInstance);
        ref var state = ref record.State;
        state.Begin(arguments.StartInstanceUserData);
        if (!TryResolveDrawTargets(banks, in draw, ref state, record))
        {
            TraceDrawTargets(banks, "no-targets");
            record.ResetBindings = true;
            Record(record);
            return;
        }

        record.TargetsResolved = true;
        if (!ResolveTopology(userConfig, out var topology))
        {
            TraceDrawDisposition(banks, in draw, "no-primitive-topology");
            TraceDrawTargets(banks, "no-topology");
            record.ResetBindings = true;
            Record(record);
            return;
        }

        ResolveShaderPrograms(banks, ref state);
        if (arguments.IndirectArgumentsAddress != 0 &&
            (_host.TakesTargetlessIndirectDrawArguments
                ? MayRetainTargetlessDraw(ref state)
                : state.ColorCount == 0 && !state.Depth.HasTarget))
        {
            // A retained draw is replayed later; it needs its counts.
            Record(record);
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
            Record(record);
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
            record.ResetBindings = true;
            Record(record);
            return;
        }

        if (!ApplyProgramAdaptations(banks, in draw, ref state, new TargetlessDrawArguments(submitId, false, default, arguments), record))
        {
            TraceDrawTargets(banks, "adapted");
            record.ResetBindings = true;
            Record(record);
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

            var vertices = SequentialVerticesScratch(arguments.VertexCount);
            var list = TriangulateFansScratch((GuestPrimitiveType)userConfig.PrimitiveType, vertices, restartIndex: null);
            if (list.Length == 0)
            {
                record.ResetBindings = true;
                Record(record);
                return;
            }

            var wide = arguments.VertexCount > ushort.MaxValue;
            var bytes = ListIndexBytesScratch(list, wide, out var byteCount);
            var listSource = new IndexSource(true, 0, RecordIndexData(bytes, byteCount), (ulong)byteCount, wide ? GpuIndexType.Uint32 : GpuIndexType.Uint16);
            var listEmission = new DrawEmission(true, vertexOffset, 0, firstInstance, TriangleListIndices: (uint)list.Length);
            FillDraw(record, banks, in draw, ref state, GpuPrimitiveTopology.TriangleList, in listEmission, in listSource,
                primitiveRestart: false, setBindDebug: false, setAutoDebug: true);
            TraceDrawTargets(banks, "recorded");
            Record(record);
            return;
        }

        var emission = new DrawEmission(
            false,
            0,
            (uint)vertexOffset,
            firstInstance,
            arguments.IndirectArgumentsAddress);
        record.PacketAddress = arguments.PacketAddress;
        FillDraw(record, banks, in draw, ref state, topology, in emission, default, primitiveRestart: false, setBindDebug: false, setAutoDebug: true);
        TraceDrawTargets(banks, "recorded");
        Record(record);
    }

    private void FillDraw(ResolvedDraw record, RegisterBanks banks, in DrawCall draw, ref DrawState state, GpuPrimitiveTopology topology,
        in DrawEmission emission, in IndexSource indexSource, bool primitiveRestart, bool setBindDebug, bool setAutoDebug)
    {
        record.Kind = ResolvedDrawKind.Draw;
        var context = banks.ShareContext();
        var vertexInput = state.Programs.VertexInput;
        var pixelInput = state.Programs.PixelInput;
        var pixelProgram = state.PixelActive ? pixelInput.Stage.Program : null;
        var writeStages = GpuPipelineStage.None;
        if (HasBufferWrites(vertexInput.Stage))
        {
            writeStages |= state.Programs.MeshInput is not null ? GpuPipelineStage.MeshShaderBitExt
                : vertexInput.Tessellation is not null ? GpuPipelineStage.TessellationEvaluationShaderBit : GpuPipelineStage.VertexShaderBit;
        }

        // Storage images and device-address stores of the fragment stage count too: a tile-based
        // host defers their pass end (RenderRecorder.NoteFragmentWrites).
        if (state.PixelActive && DrawWritesMemory(pixelInput.Stage))
        {
            writeStages |= GpuPipelineStage.FragmentShaderBit;
        }

        record.Draw = draw;
        record.Topology = topology;
        record.PrimitiveRestart = primitiveRestart;
        record.PrimitiveType = (GuestPrimitiveType)banks.UserConfig.PrimitiveType;
        record.Emission = emission;
        record.Index = indexSource;
        record.Context = context;
        record.ContextStamp = banks.ContextStamp;
        record.DynamicStamp = banks.Stamp(RegisterStamp.ContextDynamic);
        record.TargetVersion = _drawResolution.TargetVersion;
        record.Dynamic = ResolveDynamicState(banks, in state, WrittenColorTargets(context, in state, pixelProgram), record.SubmitId, record.PacketAddress);
        if (record.Dynamic.DepthBiasPerFace && DrawWritesMemory(vertexInput.Stage))
        {
            // The host draws each face separately, which runs the vertex stage twice.
            throw _host.Fatal(
                "A draw whose front and back faces take different polygon offsets writes memory from its vertex stage: " +
                $"vertex=0x{vertexInput.Stage.Program?.Hash ?? 0:X16}.");
        }

        record.WritesMemory = DrawWritesMemory(vertexInput.Stage) || (state.PixelActive && DrawWritesMemory(pixelInput.Stage));
        record.WriteStages = writeStages;
        record.SetBindDebug = setBindDebug;
        record.SetAutoDebug = setAutoDebug;
        record.MustExecute = _drawMustExecute;
        record.ResetBindings = true;
    }

    // Triangle fans, polygons and legacy quad lists, which a host without fans draws as
    // triangle lists (IRenderHost.DrawsTriangleFans).
    private bool TriangulatesFans(UserConfigRegisters userConfig) =>
        !_host.DrawsTriangleFans &&
        (GuestPrimitiveType)userConfig.PrimitiveType is
            GuestPrimitiveType.TriangleFan or GuestPrimitiveType.Polygon or GuestPrimitiveType.QuadListLegacy;

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

    // A tessellation configuration the bridge cannot express skips the draw, as an
    // unsupported geometry stage does, instead of stopping the emulator.
    private bool IsSupportedTessellation(RegisterBanks banks)
    {
        var shaderInterface = banks.Context.ShaderInterface;
        string? error = null;
        if (!_host.SupportsTessellation)
        {
            error = "the render host does not support native tessellation";
        }
        else if (!Gen5TessellationInfo.TryDecode(shaderInterface.TessellationFactorParameter, out var tessellation, out var factorError))
        {
            error = factorError;
        }
        else if (tessellation.Spacing == Gen5TessellationSpacing.PowerOfTwo)
        {
            error = "power-of-two partitioning needs factor conversion";
        }
        else if (!Gen5TessellationHullInfo.TryDecode(shaderInterface.LocalHullConfiguration, 0, out _, out var hullError))
        {
            error = hullError;
        }

        if (error is null)
        {
            return true;
        }

        if (Interlocked.Exchange(ref _tessellationWarningShown, 1) == 0)
        {
            Console.Error.WriteLine(
                $"Warning: the title uses an unsupported tessellation configuration; those draw calls are skipped. {error} " +
                $"tf_param=0x{shaderInterface.TessellationFactorParameter:X8} ls_hs_config=0x{shaderInterface.LocalHullConfiguration:X8}");
        }

        return false;
    }

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
               control.MaskExportEnable || control.ExecuteOnHierFail || control.ExecuteOnNoop;
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

    private const uint TessellationStageBits = 0x7u;

    private bool UsesMeshExecution(RegisterBanks banks) =>
        _pipelines.MeshShadersSupported && RequiresMeshExecution(banks.Context.ShaderStages) &&
        banks.Shader.Vertex.ExportAddress != 0 &&
        (banks.Context.ShaderStages & TessellationStageBits) == 0;

    private void ValidateChangedDrawRegisters(RegisterBanks banks)
    {
        var resolution = _drawResolution;
        var contextStamp = banks.ContextStamp;
        var userConfigStamp = banks.Stamp(RegisterStamp.UserConfig);
        if (resolution.ValidatedContextStamp == contextStamp && resolution.ValidatedUserConfigStamp == userConfigStamp)
        {
            return;
        }

        ValidateDrawRegisters(banks);
        resolution.ValidatedContextStamp = contextStamp;
        resolution.ValidatedUserConfigStamp = userConfigStamp;
    }

    private void ResolveShaderPrograms(RegisterBanks banks, ref DrawState state, bool indexed = false)
    {
        if (state.Programs is not null)
        {
            return;
        }

        using var profileScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.DrawProgramResolution);
        var inputs = GraphicsProgramInputs.Create();
        Span<ColorComponentMap> targetExportMapping = inputs.Mapping;
        foreach (ref readonly var color in DrawState.BoundColors(ref state))
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
        inputs.StartInstanceUserData = state.StartInstanceUserData;
        if (Prefetch is { } prefetch && prefetch.TryTakeGraphics(banks, in inputs, out var prefetched))
        {
            state.Programs = prefetched;
            return;
        }

        using var resolveScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ProgramSynchronousResolution);
        var resolveStarted = System.Diagnostics.Stopwatch.GetTimestamp();
        state.Programs = ResolveGraphicsPrograms(banks, in inputs);
        Interlocked.Add(ref _synchronousResolveTicks, System.Diagnostics.Stopwatch.GetTimestamp() - resolveStarted);
        Interlocked.Increment(ref _synchronousResolves);
    }

    private static long _synchronousResolveTicks;
    private static long _synchronousResolves;

    /// <summary>Graphics program resolutions the drawing thread made itself, and their time, since the last call.</summary>
    public static (long Count, double Milliseconds) TakeSynchronousResolveCounters() =>
        (Interlocked.Exchange(ref _synchronousResolves, 0),
            Interlocked.Exchange(ref _synchronousResolveTicks, 0) * 1000.0 / System.Diagnostics.Stopwatch.Frequency);

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
        ref readonly var viewport = ref banks.Context.ScreenViewport.Primary;
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
