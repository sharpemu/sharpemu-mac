// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler.Metal;
using SharpEmu.ShaderCompiler.Resources;
using ImageResourceClass = SharpEmu.Libs.Gpu.Rendering.ImageResourceClass;

namespace SharpEmu.Gpu.Metal.V2;

// Experimental frozen-native-input backend. Guest reads, image ownership and
// materialization still execute in order; only immutable native draw inputs cross
// this boundary. This is not a raw-register parallel resource materializer.
internal sealed partial class MetalRenderHost
{
    internal bool ParallelEncode { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_PARALLEL_ENCODE") == "1";
    private static readonly int ConfiguredParallelWorkerCount =
        int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_PARALLEL_WORKERS"), out var workers)
            ? Math.Clamp(workers, 1, 10) : 4;
    internal int ParallelWorkerCount { get; set; } = ConfiguredParallelWorkerCount;
    private static readonly nint SelParallelRenderEncoder = MetalNative.Selector("parallelRenderCommandEncoderWithDescriptor:");
    private static readonly nint SelParallelChild = MetalNative.Selector("renderCommandEncoder");
    private nint _parallelRenderEncoder;
    private readonly FrozenDraw?[] _parallelDraws = new FrozenDraw[64];
    private readonly HashSet<nint> _parallelRetained = new();
    private readonly Dictionary<nint, nuint> _parallelResidency = new();
    private bool _parallelWindowsNamed;
    private nint[] _parallelReadResources = new nint[64], _parallelReadWriteResources = new nint[64];
    private int _parallelReadCount, _parallelReadWriteCount;
    private readonly bool[] _parallelChildEnded = new bool[10];
    private readonly nint[] _parallelChildren = new nint[10];
    private MetalParallelBatchWorkers? _parallelWorkers;
    private int _parallelDrawCount;
    private long _parallelQueued, _parallelBatches, _parallelChunks, _parallelSmallBatches, _parallelRefused;
    private long _parallelCaptureTicks, _parallelNativeTicks, _parallelJoinTicks;
    private long _parallelRefusedWriter, _parallelRefusedShape, _parallelRefusedDynamic, _parallelRefusedPass;
    private long _parallelBarrierPassEnds;

    internal Action<int>? BeforeParallelChunk { get; set; }
    internal Action<int>? AfterParallelChunk { get; set; }
    internal (long Queued, long Batches, long Chunks) ParallelCounts => (_parallelQueued, _parallelBatches, _parallelChunks);

    // Native encoder ordering alone does not order arbitrary shader stores. Only
    // attachment writes and commutative mip-statistics atomics are admitted here.
    internal static bool ParallelReadOnly(ShaderProgramInfo? program)
    {
        if (program is null) return true;
        if (program.WritesDeviceAddresses || program.ScratchDwords != 0) return false;
        foreach (var buffer in program.Buffers)
            if (buffer.Written || buffer.Atomic) return false;
        foreach (var image in program.Images)
            if (image.Written || image.Class == ImageResourceClass.Storage) return false;
        if (program.Bindings is { } bindings)
            foreach (var descriptor in bindings.Descriptors)
                if (descriptor.Kind == DescriptorBindingKind.GlobalDataShare) return false;
        return true;
    }

    private bool CanParallelPipeline(MetalPipelineEntry? entry)
    {
        if (!ParallelEncode || PassProfileEnabled ||
            entry is not { Mesh: null, RectangleStates: null, GraphicsDescription: { } description }) return false;
        if (entry.ParallelDrawCompatible is { } compatible) return compatible;
        compatible = ParallelReadOnly(description.VertexStage) && ParallelReadOnly(description.PixelStage);
        entry.ParallelDrawCompatible = compatible;
        return compatible;
    }

    private nint NewRenderEncoder(nint buffer, nint descriptor, MetalPipelineEntry? upcoming)
    {
        if (!CanParallelPipeline(upcoming)) return MetalNative.Send(buffer, SelRenderEncoder, descriptor);
        _parallelRenderEncoder = RetainEncoder(MetalNative.Send(buffer, SelParallelRenderEncoder, descriptor));
        MetalMemoryTelemetry.RenderEncoders.Add();
        if (_parallelRenderEncoder == 0) throw Fatal("A parallel render encoder could not be created.");
        return MetalNative.Send(_parallelRenderEncoder, SelParallelChild);
    }

    private void HoldParallel(nint handle)
    {
        if (handle != 0 && _parallelRetained.Add(handle)) MetalNative.SendVoid(handle, SelRetain);
    }

    private void ResidentParallel(nint handle, nuint usage)
    {
        if (handle == 0) return;
        _parallelResidency.TryGetValue(handle, out var previous);
        _parallelResidency[handle] = previous | usage;
        HoldParallel(handle);
    }

    private bool TryQueueParallelDraw(bool indexed, uint count, uint instances, uint first, int baseVertex, uint baseInstance,
        BufferBinding indirect = default)
    {
        var entry = _pendingGraphicsPipeline;
        if (_parallelRenderEncoder == 0 || !CanParallelPipeline(entry) || _dynamicState.DepthBiasPerFace ||
            _dynamicState.DepthBoundsTestEnabled || _tinyPass)
        {
            if (ParallelEncode)
            {
                _parallelRefused++;
                if (entry is { Mesh: not null } or { RectangleStates: not null }) _parallelRefusedShape++;
                else if (entry?.GraphicsDescription is { } description &&
                    (!ParallelReadOnly(description.VertexStage) || !ParallelReadOnly(description.PixelStage))) _parallelRefusedWriter++;
                else if (_dynamicState.DepthBiasPerFace || _dynamicState.DepthBoundsTestEnabled)
                    _parallelRefusedDynamic++;
                else _parallelRefusedPass++;
            }
            FlushParallelDraws();
            return false;
        }
        if (_parallelDrawCount == _parallelDraws.Length) FlushParallelDraws();
        var captureStarted = System.Diagnostics.Stopwatch.GetTimestamp();
        if (indexed && _indexBuffer == 0) throw Fatal("An indexed parallel draw has no index buffer.");
        UsePipelineState(entry!);
        var draw = _parallelDraws[_parallelDrawCount] ??= new FrozenDraw();
        draw.Pipeline = entry!.State;
        draw.DepthStencil = GetDepthStencilState(entry, in _dynamicState);
        draw.Primitive = entry.PrimitiveType;
        draw.Cull = (nuint)(MetalV2Diagnostics.IgnoreFragmentTests ? Mtl.CullModeNone : entry.CullMode);
        draw.Winding = (nuint)entry.Winding;
        draw.Clip = (nuint)entry.DepthClipMode;
        draw.Fill = (nuint)entry.FillMode;
        draw.CullsEverything = entry.CullsEverything && !MetalV2Diagnostics.IgnoreFragmentTests;
        draw.AllViewports = entry.WritesViewportIndex;
        draw.Triangles = RasterizesTriangles(entry);
        draw.Dynamic = _dynamicState;
        draw.Width = Math.Max(_renderingState.Width, 1u);
        draw.Height = Math.Max(_renderingState.Height, 1u);
        draw.Indexed = indexed;
        draw.Count = count;
        draw.Instances = Math.Max(instances, 1);
        draw.First = first;
        draw.BaseVertex = baseVertex;
        draw.BaseInstance = baseInstance;
        draw.IndexBuffer = _indexBuffer;
        draw.IndexOffset = _indexOffset;
        draw.IndexType = _indexType;
        draw.Indirect = (nint)indirect.Handle;
        draw.IndirectOffset = indirect.Offset;
        draw.VertexCount = _vertexBufferCount;
        Array.Copy(_vertexBuffers, draw.Vertices, draw.VertexCount);
        Array.Copy(_vertexOffsets, draw.VertexOffsets, draw.VertexCount);
        for (var stage = 0; stage < 2; stage++)
        {
            draw.Arguments[stage] = _stageArgumentValid[stage] ? _stageArgumentBuffers[stage] : 0;
            draw.ArgumentOffsets[stage] = _stageArgumentOffsets[stage];
            draw.Push[stage] = _hasPushData && (_preparedStageMask & (1 << stage)) != 0 ? _pushBuffer : 0;
            HoldParallel(draw.Arguments[stage]);
            HoldParallel(draw.Push[stage]);
        }
        draw.PushOffset = _pushOffset;
        HoldParallel(draw.Pipeline);
        HoldParallel(draw.DepthStencil);
        HoldParallel(draw.IndexBuffer);
        HoldParallel(draw.Indirect);
        for (var i = 0; i < draw.VertexCount; i++) HoldParallel(draw.Vertices[i]);
        foreach (var buffer in _guestMemory.ResidentBuffers)
            ResidentParallel(buffer, Mtl.ResourceUsageRead | Mtl.ResourceUsageWrite);
        for (var i = 0; i < _usedResourceCount; i++) ResidentParallel(_usedResources[i], _usedResourceUsage[i]);
        if (_stageUsesDeviceAddresses[StageVertex] || _stageUsesDeviceAddresses[StageFragment])
        {
            if (_deviceAddressTableSerial != _submissionSerial) UploadDeviceAddressTable();
            if (!_parallelWindowsNamed)
            {
                var named = DeviceAddressNamedWindows();
                if (SharpEmu.Logging.MemoryTelemetry.Enabled) _commandBufferReachWindows.UnionWith(named);
                foreach (var id in named)
                    ResidentParallel(_guestMemory.WindowBuffer(id), Mtl.ResourceUsageRead | Mtl.ResourceUsageWrite);
                _parallelWindowsNamed = true;
            }
        }
        foreach (var table in _bindlessDeclarations)
            foreach (var texture in table.Textures) ResidentParallel(texture, Mtl.ResourceUsageRead);
        _boundGraphicsPipeline = entry;
        _parallelDrawCount++;
        _parallelQueued++;
        _parallelCaptureTicks += System.Diagnostics.Stopwatch.GetTimestamp() - captureStarted;
        return true;
    }

    private void FlushParallelDraws(bool createTail = true)
    {
        if (_parallelDrawCount == 0) return;
        var joinStarted = System.Diagnostics.Stopwatch.GetTimestamp();
        _parallelReadCount = _parallelReadWriteCount = 0;
        foreach (var (resource, usage) in _parallelResidency)
        {
            ref var resources = ref ((usage & Mtl.ResourceUsageWrite) != 0 ? ref _parallelReadWriteResources : ref _parallelReadResources);
            ref var count = ref ((usage & Mtl.ResourceUsageWrite) != 0 ? ref _parallelReadWriteCount : ref _parallelReadCount);
            if (count == resources.Length) Array.Resize(ref resources, count * 2);
            resources[count++] = resource;
        }
        // The serial prefix precedes every child; it cannot inherit queued state.
        EndAndReleaseEncoder(_renderEncoder);
        _renderEncoder = 0;
        var chunks = _parallelDrawCount >= Math.Min(ParallelWorkerCount * 8, _parallelDraws.Length) ? ParallelWorkerCount : 1;
        try
        {
            // Creation order, not worker completion order, fixes draw order.
            for (var i = 0; i < chunks; i++)
            {
                _parallelChildEnded[i] = false;
                _parallelChildren[i] = RetainEncoder(MetalNative.Send(_parallelRenderEncoder, SelParallelChild));
                MetalMemoryTelemetry.RenderEncoders.Add();
                if (_parallelChildren[i] == 0) throw Fatal("A parallel render child could not be created.");
            }
            if (chunks == 1)
            {
                _parallelSmallBatches++;
                EncodeParallelChunk(0, 1);
            }
            else
            {
                _parallelWorkers ??= new MetalParallelBatchWorkers(ParallelWorkerCount,
                    worker => EncodeParallelChunk(worker, ParallelWorkerCount));
                _parallelWorkers.Execute();
                _parallelBatches++;
            }
            _parallelChunks += chunks;
        }
        finally
        {
            for (var i = 0; i < chunks; i++)
            {
                if (_parallelChildren[i] != 0)
                {
                    if (!_parallelChildEnded[i]) MetalNative.SendVoid(_parallelChildren[i], SelEndEncoding);
                    MetalNative.SendVoid(_parallelChildren[i], SelRelease);
                }
                _parallelChildren[i] = 0;
            }
            foreach (var handle in _parallelRetained) MetalNative.SendVoid(handle, SelRelease);
            _parallelRetained.Clear();
            _parallelResidency.Clear();
            _parallelWindowsNamed = false;
            _parallelDrawCount = 0;
        }
        if (createTail)
        {
            _renderEncoder = RetainEncoder(MetalNative.Send(_parallelRenderEncoder, SelParallelChild));
            MetalMemoryTelemetry.RenderEncoders.Add();
            if (_renderEncoder == 0) throw Fatal("A parallel render tail could not be created.");
        }
        ForgetEncoderState();
        _parallelJoinTicks += System.Diagnostics.Stopwatch.GetTimestamp() - joinStarted;
    }

    private unsafe void EncodeParallelChunk(int worker, int chunks)
    {
        var nativeStarted = System.Diagnostics.Stopwatch.GetTimestamp();
        var pool = MetalNative.objc_autoreleasePoolPush();
        var encoder = _parallelChildren[worker];
        try
        {
            if (chunks > 1) BeforeParallelChunk?.Invoke(worker);
            const nuint stages = Mtl.RenderStageVertex | Mtl.RenderStageFragment;
            fixed (nint* resources = _parallelReadResources)
                if (_parallelReadCount != 0)
                    MetalNative.SendUseResourcesStages(encoder, SelUseResourcesStages, resources,
                        (nuint)_parallelReadCount, Mtl.ResourceUsageRead, stages);
            fixed (nint* resources = _parallelReadWriteResources)
                if (_parallelReadWriteCount != 0)
                    MetalNative.SendUseResourcesStages(encoder, SelUseResourcesStages, resources,
                        (nuint)_parallelReadWriteCount, Mtl.ResourceUsageRead | Mtl.ResourceUsageWrite, stages);
            var (start, end) = MetalParallelBatchWorkers.Chunk(_parallelDrawCount, chunks, worker);
            for (var i = start; i < end; i++) _parallelDraws[i]!.Encode(encoder);
        }
        finally
        {
            MetalNative.SendVoid(encoder, SelEndEncoding);
            _parallelChildEnded[worker] = true;
            try { if (chunks > 1) AfterParallelChunk?.Invoke(worker); }
            finally
            {
                MetalNative.objc_autoreleasePoolPop(pool);
                Interlocked.Add(ref _parallelNativeTicks, System.Diagnostics.Stopwatch.GetTimestamp() - nativeStarted);
            }
        }
    }

    private nint StoreActionEncoder => _parallelRenderEncoder != 0 ? _parallelRenderEncoder : _renderEncoder;

    private static string ParallelMilliseconds(long ticks) =>
        (ticks * 1000d / System.Diagnostics.Stopwatch.Frequency).ToString("F3", System.Globalization.CultureInfo.InvariantCulture);

    private void ReportParallelEncode()
    {
        if (!ParallelEncode) return;
        Console.Error.WriteLine($"[METAL-V2][PARALLEL] workers={ParallelWorkerCount} queued={_parallelQueued} " +
            $"batches={_parallelBatches} chunks={_parallelChunks} small_batches={_parallelSmallBatches} refused={_parallelRefused} " +
            $"native_qos_workers={_parallelWorkers?.NativeQosWorkers ?? 0} " +
            $"refused_writer={_parallelRefusedWriter} refused_shape={_parallelRefusedShape} " +
            $"refused_dynamic={_parallelRefusedDynamic} refused_pass={_parallelRefusedPass} barrier_pass_ends={_parallelBarrierPassEnds} " +
            $"capture_wall_ms={ParallelMilliseconds(_parallelCaptureTicks)} " +
            $"native_wall_ms={ParallelMilliseconds(Interlocked.Read(ref _parallelNativeTicks))} " +
            $"join_wall_ms={ParallelMilliseconds(_parallelJoinTicks)} " +
            "guest_prepare=ordered frozen_native_encode=1 compute=ordered");
    }

    private sealed class FrozenDraw
    {
        internal nint Pipeline, DepthStencil, IndexBuffer, Indirect;
        internal nuint Primitive, Cull, Winding, Clip, Fill, IndexType;
        internal ulong IndexOffset, IndirectOffset, PushOffset;
        internal uint Count, Instances, First, BaseInstance, Width, Height;
        internal int BaseVertex, VertexCount;
        internal bool Indexed, CullsEverything, AllViewports, Triangles;
        internal DynamicDrawState Dynamic;
        internal readonly nint[] Vertices = new nint[MaxVertexBufferSlots];
        internal readonly ulong[] VertexOffsets = new ulong[MaxVertexBufferSlots];
        internal readonly nint[] Arguments = new nint[2], Push = new nint[2];
        internal readonly ulong[] ArgumentOffsets = new ulong[2];

        internal unsafe void Encode(nint encoder)
        {
            MetalNative.SendVoid(encoder, SelSetRenderPipelineState, Pipeline);
            MetalNative.SendVoid(encoder, SelSetDepthStencilState, DepthStencil);
            MetalV2Native.SendVoidValue(encoder, SelSetCullMode, Cull);
            MetalV2Native.SendVoidValue(encoder, SelSetFrontFacingWinding, Winding);
            MetalV2Native.SendVoidValue(encoder, SelSetDepthClipMode, Clip);
            MetalV2Native.SendVoidValue(encoder, SelSetTriangleFillMode, Fill);
            var viewport = HostViewport(in Dynamic);
            var left = (uint)Math.Clamp(Dynamic.Scissor.Left, 0, (int)Width);
            var top = (uint)Math.Clamp(Dynamic.Scissor.Top, 0, (int)Height);
            var right = (uint)Math.Clamp(Dynamic.Scissor.Right, 0, (int)Width);
            var bottom = (uint)Math.Clamp(Dynamic.Scissor.Bottom, 0, (int)Height);
            var scissor = new MtlScissorRect
            {
                X = CullsEverything ? 0 : left,
                Y = CullsEverything ? 0 : top,
                Width = CullsEverything ? 1 : Math.Max(right > left ? right - left : 0, 1),
                Height = CullsEverything ? 1 : Math.Max(bottom > top ? bottom - top : 0, 1),
            };
            if (AllViewports)
            {
                var viewports = stackalloc MtlViewport[ViewportSlotCount];
                var scissors = stackalloc MtlScissorRect[ViewportSlotCount];
                for (var i = 0; i < ViewportSlotCount; i++) { viewports[i] = viewport; scissors[i] = scissor; }
                MetalV2Native.SendVoidArray(encoder, SelSetViewports, (nint)viewports, ViewportSlotCount);
                MetalV2Native.SendVoidArray(encoder, SelSetScissorRects, (nint)scissors, ViewportSlotCount);
            }
            else
            {
                MetalNative.SendVoidViewport(encoder, SelSetViewport, viewport);
                MetalNative.SendVoidScissor(encoder, SelSetScissorRect, scissor);
            }
            MetalNative.SendVoidBlendColor(encoder, SelSetBlendColor, Dynamic.BlendRed, Dynamic.BlendGreen, Dynamic.BlendBlue, Dynamic.BlendAlpha);
            MetalV2Native.SendDepthBias(encoder, SelSetDepthBias,
                !Dynamic.DepthBiasEnabled ? 0 : Triangles ? Dynamic.DepthBiasConstantFactor : Dynamic.DepthBiasPointLineConstantFactor,
                !Dynamic.DepthBiasEnabled ? 0 : Triangles ? Dynamic.DepthBiasSlopeFactor : Dynamic.DepthBiasPointLineSlopeFactor,
                Dynamic.DepthBiasEnabled ? Dynamic.DepthBiasClamp : 0);
            if (Dynamic.StencilTestEnabled)
                MetalV2Native.SendStencilReferences(encoder, SelSetStencilReferences, Dynamic.FrontStencil.Reference, Dynamic.BackStencil.Reference);
            for (var i = 0; i < VertexCount; i++)
                MetalNative.SendSetBuffer(encoder, SelSetVertexBuffer, Vertices[i], (nuint)VertexOffsets[i], (nuint)(VertexBufferSlotBase + i));
            for (var stage = 0; stage < 2; stage++)
            {
                var selector = stage == StageVertex ? SelSetVertexBuffer : SelSetFragmentBuffer;
                if (Arguments[stage] != 0)
                    MetalNative.SendSetBuffer(encoder, selector, Arguments[stage], (nuint)ArgumentOffsets[stage], Gen5MslArgumentLayout.ResourcesBufferIndex);
                if (Push[stage] != 0)
                    MetalNative.SendSetBuffer(encoder, selector, Push[stage], (nuint)PushOffset, Gen5MslArgumentLayout.PushDataBufferIndex);
            }
            if (Indirect != 0)
            {
                if (Indexed)
                    MetalV2Native.SendDrawIndexedIndirect(encoder, SelDrawIndexedIndirect, Primitive, IndexType, IndexBuffer,
                        (nuint)IndexOffset, Indirect, (nuint)IndirectOffset);
                else MetalV2Native.SendDrawIndirect(encoder, SelDrawIndirect, Primitive, Indirect, (nuint)IndirectOffset);
            }
            else if (Indexed)
                MetalV2Native.SendDrawIndexedBaseVertex(encoder, SelDrawIndexedPrimitives, Primitive, Count, IndexType, IndexBuffer,
                    (nuint)(IndexOffset + First * (IndexType == Mtl.IndexTypeUInt32 ? 4UL : 2UL)), Instances, BaseVertex, BaseInstance);
            else MetalV2Native.SendDrawPrimitivesBaseInstance(encoder, SelDrawPrimitives, Primitive, First, Count, Instances, BaseInstance);
        }
    }
}
