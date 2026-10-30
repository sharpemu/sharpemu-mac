// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler.Metal;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// Indirect draws whose counts stay in guest memory, read by the GPU.
/// </summary>
/// <remarks>
/// <para>
/// Reading an indirect draw's arguments on the CPU waits for the GPU that wrote them - on
/// Silent Hill's title, the compute pass that culls the scene - and drains the whole queue
/// with it. A plain indirect draw hands the guest record to Metal's indirect draws. A
/// primitive shader runs here as a mesh pipeline, whose launch grid and draw-info block
/// (<see cref="DrawMesh"/>) are sized from the counts, and a non-indexed rectangle list
/// draws one patch per rectangle; for those a compute kernel (<see cref="Gen5MslMeshLaunch"/>)
/// reads the guest record and writes the block and the grid the CPU draw would have
/// written, and the draw launches from them with drawMeshThreadgroupsWithIndirectBuffer or
/// drawPatches:...indirectBuffer:.
/// </para>
/// <para>
/// The kernel has to run where no pass is open, so the launches are encoded in
/// <see cref="BeginRendering"/>, between the pass that ends and the one that begins. When
/// the draw would have continued the open pass, that pass ends for the kernel (counted as
/// end_indirect). To keep that to the first indirect draw of a pass, one dispatch also
/// writes the launches of the indirect draws that followed this one the last time it was
/// drawn; while the pass they were written for stays open nothing can write the guest
/// records (every GPU write outside a draw needs an encoder of its own, and a draw that
/// stores to memory ends the pass), so a later draw of the pass that finds its launch there
/// uses it without ending the pass.
/// </para>
/// <para>
/// Arguments no GPU work is still writing are read on the CPU as before: the draw then
/// needs no kernel, and an indexed triangle list can still use cached meshlets.
/// SHARPEMU_METAL_V2_INDIRECT_GPU=0 reads every indirect draw's arguments on the CPU.
/// </para>
/// </remarks>
internal sealed partial class MetalRenderHost
{
    // Off with =0: every indirect draw of a mesh pipeline or a rectangle list, and every
    // non-indexed indirect draw, is read on the CPU, as before the GPU path existed.
    internal static readonly bool IndirectArgumentsOnGpu =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_INDIRECT_GPU") != "0";

    // The launches one dispatch writes: the draw's own and those predicted to follow it.
    private const int MaxIndirectLaunchBatch = 32;

    // Successors remembered; the map is dropped and relearnt past this.
    private const int MaxIndirectLaunchSuccessors = 8192;

    // Launches past this many threadgroups are not a real draw but arguments the GPU never
    // wrote; they are dropped rather than handed to the GPU as a grid it could not finish.
    private const ulong MaxIndirectLaunchThreadgroups = 1UL << 26;

    private static readonly nint SelDrawPatchesIndirect = MetalNative.Selector(
        "drawPatches:patchIndexBuffer:patchIndexBufferOffset:indirectBuffer:indirectBufferOffset:");
    private static readonly nint SelNewFence = MetalNative.Selector("newFence");
    private static readonly nint SelUpdateFence = MetalNative.Selector("updateFence:");
    private static readonly nint SelWaitForFenceBeforeStages = MetalNative.Selector("waitForFence:beforeStages:");
    private static readonly nint SelDrawMeshIndirect = MetalNative.Selector(
        "drawMeshThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:");

    // One launch: the pipeline, the guest record (its address and the binding it resolved
    // to) and what the kernel needs besides. Two draws with equal keys get equal launches
    // from equal guest bytes.
    private readonly record struct IndirectLaunchKey(
        MetalPipelineEntry Entry,
        ulong Address,
        ulong Buffer,
        ulong Offset,
        bool Indexed,
        uint IndexType,
        uint IndexLimit);

    private nint _indirectLaunchPipeline;
    private nint _indirectLaunchFence;

    // The launch PrepareIndirectDraw asked for, for the BeginRendering that follows it.
    private IndirectLaunchKey? _requestedLaunch;

    // The launch the next indirect draw of a mesh pipeline or rectangle list uses.
    private IndirectLaunchKey? _launchForDrawKey;
    private MetalGuestRange _launchForDraw;

    // The launches the last dispatch wrote, valid while the pass they were written for is open.
    private readonly Dictionary<IndirectLaunchKey, MetalGuestRange> _preparedLaunches = new();
    private long _preparedLaunchPass = -1;

    // Which launch followed which, the last time: the prediction a dispatch writes ahead.
    private readonly Dictionary<IndirectLaunchKey, IndirectLaunchKey> _launchSuccessors = new();
    private readonly List<IndirectLaunchKey> _launchBatch = new(MaxIndirectLaunchBatch);
    private readonly HashSet<IndirectLaunchKey> _launchBatchSeen = new();
    private IndirectLaunchKey? _lastLaunch;

    // Bumped whenever a render pass ends; a prepared launch is only good for its own pass.
    private long _renderPassSerial;

    private long _indirectGpuDraws;
    private long _indirectRectangleDraws;
    private long _indirectCpuReads;
    private long _indirectCpuWaits;
    private long _indirectLaunchBatches;
    private long _indirectLaunchesWritten;
    private long _indirectLaunchesReused;
    private long _passEndIndirect;
    private long _passEndRelease;
    private long _passEndBarrier;
    private long _passEndDispatch;

    /// <summary>The indirect-draw counters of the PERF line; reported and reset.</summary>
    internal string TakeIndirectCounters() =>
        $"indirect_gpu={Interlocked.Exchange(ref _indirectGpuDraws, 0)} " +
        $"indirect_rect_gpu={Interlocked.Exchange(ref _indirectRectangleDraws, 0)} " +
        $"indirect_cpu={Interlocked.Exchange(ref _indirectCpuReads, 0)} " +
        $"indirect_cpu_waits={Interlocked.Exchange(ref _indirectCpuWaits, 0)} " +
        $"indirect_batches={Interlocked.Exchange(ref _indirectLaunchBatches, 0)} " +
        $"indirect_written={Interlocked.Exchange(ref _indirectLaunchesWritten, 0)} " +
        $"indirect_reused={Interlocked.Exchange(ref _indirectLaunchesReused, 0)}";

    /// <summary>Passes an indirect launch ended that would otherwise have continued; reported and reset.</summary>
    internal long TakeIndirectPassEnds() => Interlocked.Exchange(ref _passEndIndirect, 0);

    /// <summary>Passes the executor ended itself, as release/barrier/dispatch; reported and reset.</summary>
    internal string TakeExecutorPassEnds() =>
        $"end_release={Interlocked.Exchange(ref _passEndRelease, 0)} " +
        $"end_barrier={Interlocked.Exchange(ref _passEndBarrier, 0)} " +
        $"end_dispatch={Interlocked.Exchange(ref _passEndDispatch, 0)}";

    /// <summary>
    /// Whether recorded GPU work may still write these bytes, so that a CPU read of them now
    /// would wait: the same test <see cref="WaitForGuestBytes"/> waits on.
    /// </summary>
    public bool HasPendingGpuWrite(ulong address, ulong size)
    {
        if (OnCommandThread)
        {
            var pending = false;
            RunTagged(() => pending = HasPendingGpuWrite(address, size));
            return pending;
        }

        // A deferred draw may be what writes the bytes.
        DrainDraws();
        using var phaseScope = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.PendingGpuWriteQuery);
        if (size == 0 || (!_gpuWrites.MayCover(address, size) && !_imageCache.MayOwn(address, size)))
        {
            return false;
        }

        return _imageCache.HasGpuModifiedImageBytes(address, size) ||
            PendingSerialHinted(address, size) != 0;
    }

    public void NoteIndirectArgumentsReadOnCpu(ulong address, ulong size)
    {
        Interlocked.Increment(ref _indirectCpuReads);
        if (HasPendingGpuWrite(address, size))
        {
            Interlocked.Increment(ref _indirectCpuWaits);
        }
    }

    public bool ResolvesIndirectAutoDrawOnGpu => IndirectArgumentsOnGpu;

    public bool TakesTargetlessIndirectDrawArguments => IndirectArgumentsOnGpu;

    /// <summary>
    /// Whether an indirect draw with this vertex program can be handed over with its counts
    /// still in guest memory. A plain program always can. A primitive shader runs here as a
    /// mesh pipeline, whose launch the kernel then has to write, so it takes them only while
    /// GPU work still writes them; final counts are cheaper read now.
    /// </summary>
    public bool TakesIndirectDrawArguments(ShaderProgram vertexProgram, ulong argumentsAddress, ulong argumentsSize)
    {
        lock (_modules)
        {
            var index = (int)vertexProgram.Module - 1;
            // An unknown module is not a mesh program as far as anything here knows; the
            // draw path reports it where it can say which program it was.
            if (index < 0 || index >= _modules.Count || _modules[index].Shader.Stage != Gen5MslStage.Mesh)
            {
                return true;
            }
        }

        return TakesLaunchArguments(argumentsAddress, argumentsSize);
    }

    /// <summary>
    /// Whether a non-indexed rectangle list with this vertex input can take its counts on
    /// the GPU: its first vertex then reaches the program through a GPU-written draw block,
    /// which is enough unless a per-vertex buffer has to be rebound at it.
    /// </summary>
    public bool TakesIndirectAutoRectangleListArguments(VertexInputInfo vertexInput, ulong argumentsAddress, ulong argumentsSize)
    {
        foreach (var buffer in vertexInput.Buffers)
        {
            if (buffer.Stride != 0 && !buffer.PerInstance)
            {
                return false;
            }
        }

        return TakesLaunchArguments(argumentsAddress, argumentsSize);
    }

    // The kernel reads the record as dwords.
    private bool TakesLaunchArguments(ulong address, ulong size) =>
        IndirectArgumentsOnGpu && (address & 0x3) == 0 && HasPendingGpuWrite(address, size);

    public void PrepareIndirectDraw(in PipelineHandle pipeline, ulong argumentsAddress, BufferBinding arguments, bool indexed, uint indexLimit)
    {
        _requestedLaunch = null;
        _tracedIndirectArguments = argumentsAddress;
        var entry = RequirePipelineEntry(in pipeline);
        if (entry.RectangleStates is not null && indexed)
        {
            // Built in the pass from the record; only the bound index count is needed.
            _rectangleRebaseLimit = indexLimit;
            return;
        }

        var rectangles = entry.RectangleStates is not null && !indexed;
        if (entry.Mesh is null && !rectangles)
        {
            return;
        }

        if (arguments.Handle == 0)
        {
            throw Fatal($"An indirect launch has no argument buffer: pipeline={entry.Id} {entry.Profile}.");
        }

        var key = new IndirectLaunchKey(
            entry,
            argumentsAddress,
            arguments.Handle,
            arguments.Offset,
            indexed,
            indexed ? _indexType == Mtl.IndexTypeUInt32 ? 2u : 1u : 0u,
            indexed ? indexLimit : 0);
        if (_lastLaunch is { } last && last != key)
        {
            if (_launchSuccessors.Count >= MaxIndirectLaunchSuccessors)
            {
                _launchSuccessors.Clear();
            }

            _launchSuccessors[last] = key;
        }

        _lastLaunch = key;
        _requestedLaunch = key;
    }

    /// <summary>
    /// Called by <see cref="BeginRendering"/>: true when the launch the draw asked for was
    /// already written for the open pass, which the draw may then continue.
    /// </summary>
    private bool TryTakePreparedLaunch(IndirectLaunchKey key)
    {
        if (!_renderingActive || _preparedLaunchPass != _renderPassSerial ||
            !_preparedLaunches.TryGetValue(key, out var slot))
        {
            return false;
        }

        _launchForDrawKey = key;
        _launchForDraw = slot;
        Interlocked.Increment(ref _indirectLaunchesReused);
        return true;
    }

    /// <summary>
    /// Called by <see cref="BeginRendering"/> where no encoder is open, before the pass the
    /// draw opens: writes the draw's launch, and the launches predicted to follow it in the
    /// same pass, with one dispatch.
    /// </summary>
    private void EncodeIndirectLaunches(IndirectLaunchKey key)
    {
        var batch = _launchBatch;
        var seen = _launchBatchSeen;
        batch.Clear();
        seen.Clear();
        batch.Add(key);
        seen.Add(key);
        var next = key;
        for (var step = 0; step < MaxIndirectLaunchBatch * 2 && batch.Count < MaxIndirectLaunchBatch; step++)
        {
            if (!_launchSuccessors.TryGetValue(next, out var successor) || !seen.Add(successor))
            {
                break;
            }

            next = successor;
            if (CanWriteLaunchAhead(key, successor))
            {
                batch.Add(successor);
            }
        }

        var slots = _guestMemory.Allocate((ulong)batch.Count * Gen5MslMeshLaunch.SlotBytes, Gen5MslMeshLaunch.SlotBytes, out var span);
        span.Clear();
        _preparedLaunches.Clear();
        for (var index = 0; index < batch.Count; index++)
        {
            var launch = batch[index];
            var slotOffset = index * (int)Gen5MslMeshLaunch.SlotBytes;
            var parameters = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, uint>(
                span.Slice(slotOffset + (int)Gen5MslMeshLaunch.ParametersByteOffset, Gen5MslMeshLaunch.ParameterDwords * sizeof(uint)));
            if (launch.Entry.Mesh is { } mesh)
            {
                Gen5MslMeshLaunch.WriteMeshParameters(
                    parameters,
                    mesh,
                    MeshInputTopology(launch.Entry.StaticParameters?.Topology),
                    launch.Offset,
                    launch.Indexed,
                    launch.IndexType,
                    launch.IndexLimit,
                    MaxIndirectLaunchThreadgroups);
            }
            else
            {
                Gen5MslMeshLaunch.WriteRectangleParameters(parameters, launch.Offset, MaxIndirectLaunchThreadgroups);
            }

            _preparedLaunches[launch] = slots with
            {
                Offset = slots.Offset + (ulong)slotOffset,
                Size = Gen5MslMeshLaunch.SlotBytes,
            };
        }

        var encoder = MetalNative.Send(EnsureCommandBuffer(), SelComputeEncoder);
        MetalMemoryTelemetry.ComputeEncoders.Add();
        if (encoder == 0)
        {
            throw Fatal("A Metal compute encoder could not be created for an indirect launch.");
        }

        WaitForPreviousEncoder(encoder, render: false);

        MetalNative.SendVoid(encoder, SelSetComputePipelineState, IndirectLaunchPipeline());
        // Every record of the batch is in this buffer (CanWriteLaunchAhead), at its own offset.
        MetalNative.SendSetBuffer(encoder, SelSetComputeBuffer, (nint)key.Buffer, 0, 0);
        MetalNative.SendSetBuffer(encoder, SelSetComputeBuffer, slots.Buffer, (nuint)slots.Offset, 1);
        MetalNative.SendDispatch(
            encoder,
            SelDispatchThreadgroups,
            new MtlSize { Width = 1, Height = 1, Depth = 1 },
            new MtlSize { Width = (nuint)batch.Count, Height = 1, Depth = 1 });
        // The slots live in an untracked ring: the pass that draws from them waits on this.
        MetalNative.SendVoid(encoder, SelUpdateFence, IndirectLaunchFence());
        UpdateEncoderFence(encoder, render: false);
        MetalNative.SendVoid(encoder, SelEndEncoding);
        NoteRecordedWork();
        if (PassProfileEnabled)
        {
            PassProfileNoteBetween($"indirect-launch n={batch.Count}");
        }

        _preparedLaunchPass = _renderPassSerial;
        _launchForDrawKey = key;
        _launchForDraw = _preparedLaunches[key];
        Interlocked.Increment(ref _indirectLaunchBatches);
        Interlocked.Add(ref _indirectLaunchesWritten, batch.Count);
    }

    // A launch written ahead reads its record now, so the record has to be guest memory in
    // place in the same buffer as the draw's own (one binding serves the dispatch), and not
    // bytes an image still holds, which only the draw's own binding would publish.
    private bool CanWriteLaunchAhead(IndirectLaunchKey key, IndirectLaunchKey ahead)
    {
        if (ahead.Buffer != key.Buffer || (ahead.Address & 0x3) != 0)
        {
            return false;
        }

        var size = ahead.Indexed ? 20UL : 16UL;
        return !_imageCache.MayOwn(ahead.Address, size) &&
            _guestMemory.TryResolveInPlace(ahead.Address, size, out var range) &&
            (ulong)range.Buffer == ahead.Buffer && range.Offset == ahead.Offset;
    }

    /// <summary>Called by <see cref="BeginRendering"/> once the pass that draws from fresh launches exists.</summary>
    private void WaitForIndirectLaunches() =>
        MetalV2Native.SendWaitForFence(
            _renderEncoder,
            SelWaitForFenceBeforeStages,
            IndirectLaunchFence(),
            Mtl.RenderStageVertex | Mtl.RenderStageObject | Mtl.RenderStageMesh);

    private MetalGuestRange TakeLaunchForDraw(MetalPipelineEntry entry, BufferBinding arguments, string kind)
    {
        if (_launchForDrawKey is not { } key || !ReferenceEquals(key.Entry, entry) ||
            key.Buffer != arguments.Handle || key.Offset != arguments.Offset)
        {
            throw Fatal(
                $"An indirect {kind} arrived without the launch prepared for it: " +
                $"pipeline={entry.Id} {entry.Profile}.");
        }

        _launchForDrawKey = null;
        return _launchForDraw;
    }

    // The draw half of a non-indexed rectangle list; see DrawRectangles, whose block and
    // patch count the kernel wrote instead.
    private void DrawRectanglesIndirect(MetalPipelineEntry entry, BufferBinding arguments)
    {
        var slot = TakeLaunchForDraw(entry, arguments, "rectangle list");
        foreach (var stride in entry.RectangleVertexStrides!)
        {
            if (stride != 0)
            {
                throw Fatal(
                    "An indirect rectangle list has a per-vertex buffer to rebind at its first vertex: " +
                    $"pipeline={entry.Id} {entry.Profile}.");
            }
        }

        var state = entry.RectangleStates![0];
        if (_boundRenderState != state)
        {
            MetalNative.SendVoid(_renderEncoder, SelSetRenderPipelineState, state);
            _boundRenderState = state;
        }

        MetalNative.SendSetBuffer(
            _renderEncoder, SelSetVertexBuffer, slot.Buffer, (nuint)slot.Offset, Gen5MslRectangleList.DrawInfoBufferIndex);
        MetalNative.SendSetBuffer(
            _renderEncoder, SelSetVertexBuffer, _guestMemory.NullBuffer, 0, Gen5MslRectangleList.IndexBufferIndex);
        _boundVertexBuffers[Gen5MslRectangleList.IndexBufferIndex - VertexBufferSlotBase] = -1;
        _boundVertexBuffers[Gen5MslRectangleList.DrawInfoBufferIndex - VertexBufferSlotBase] = -1;
        var factors = _guestMemory.Allocate(6 * sizeof(ushort), 256, out var factorBytes);
        System.Runtime.InteropServices.MemoryMarshal.Cast<byte, ushort>(factorBytes).Fill(0x3C00);
        MetalNative.SendSetBuffer(_renderEncoder, SelSetTessellationFactorBuffer, factors.Buffer, (nuint)factors.Offset, 0);
        MetalV2Native.SendDrawPatchesIndirect(
            _renderEncoder,
            SelDrawPatchesIndirect,
            Gen5MslRectangleList.VerticesPerRectangle,
            0,
            0,
            slot.Buffer,
            (nuint)(slot.Offset + Gen5MslMeshLaunch.GridByteOffset));
        Interlocked.Increment(ref _indirectGpuDraws);
    }

    // The draw half of an indirect draw of a mesh pipeline: DrawMesh's bindings, with the
    // draw-info block and the grid the kernel wrote. No cached meshlets: the kernel cannot
    // build them, so the program runs whole subgroups (meshlet_count 0).
    private void DrawMeshIndirect(MetalPipelineEntry entry, Gen5MslMeshInfo mesh, BufferBinding arguments, bool indexed)
    {
        var slot = TakeLaunchForDraw(entry, arguments, "mesh draw");
        MetalNative.SendSetBuffer(
            _renderEncoder, SelSetMeshBuffer, slot.Buffer, (nuint)slot.Offset, (nuint)mesh.DrawInfoBufferIndex);
        var indexBuffer = indexed && _indexBuffer != 0 ? _indexBuffer : _guestMemory.NullBuffer;
        var indexOffset = indexed && _indexBuffer != 0 ? _indexOffset : 0;
        MetalNative.SendSetBuffer(
            _renderEncoder, SelSetMeshBuffer, indexBuffer, (nuint)indexOffset, (nuint)mesh.IndexBufferIndex);
        if (!mesh.HasGeometryHalf)
        {
            MetalNative.SendSetBuffer(
                _renderEncoder, SelSetMeshBuffer, _guestMemory.NullBuffer, 0, Gen5MslMeshInfo.MeshletBufferIndex);
        }

        if (mesh.DeviceLds)
        {
            MetalNative.SendSetBuffer(
                _renderEncoder, SelSetMeshBuffer, _guestMemory.MeshLdsPool, 0, Gen5MslMeshInfo.LdsPoolBufferIndex);
        }

        var passes = FacePassCount(entry);
        for (var pass = 0; pass < passes; pass++)
        {
            if (passes > 1)
            {
                SelectFacePass(pass);
            }

            MetalV2Native.SendDrawMeshIndirect(
                _renderEncoder,
                SelDrawMeshIndirect,
                slot.Buffer,
                (nuint)(slot.Offset + Gen5MslMeshLaunch.GridByteOffset),
                new MtlSize { Width = 1, Height = 1, Depth = 1 },
                new MtlSize { Width = Math.Max(mesh.ThreadsPerThreadgroup, 1), Height = 1, Depth = 1 });
        }

        EndFacePasses(entry, passes);
        Interlocked.Increment(ref _indirectGpuDraws);
    }

    private nint IndirectLaunchFence()
    {
        if (_indirectLaunchFence == 0)
        {
            _indirectLaunchFence = MetalNative.Send(_device, SelNewFence);
            if (_indirectLaunchFence == 0)
            {
                throw Fatal("A Metal fence could not be created for indirect launches.");
            }
        }

        return _indirectLaunchFence;
    }

    private nint IndirectLaunchPipeline()
    {
        if (_indirectLaunchPipeline != 0)
        {
            return _indirectLaunchPipeline;
        }

        var pool = MetalNative.objc_autoreleasePoolPush();
        try
        {
            var options = MetalNative.Send(
                MetalNative.Send(MetalNative.Class("MTLCompileOptions"), MetalNative.Selector("alloc")),
                MetalNative.Selector("init"));
            MetalNative.ApplyMslCompileOptions(options);
            nint error = 0;
            var library = MetalNative.Send(
                _device, MetalNative.Selector("newLibraryWithSource:options:error:"),
                MetalNative.NsString(Gen5MslMeshLaunch.KernelSource), options, ref error);
            MetalNative.SendVoid(options, SelRelease);
            if (library == 0)
            {
                throw Fatal($"The indirect launch kernel did not compile: {MetalNative.DescribeError(error)}.");
            }

            var function = MetalNative.Send(
                library, MetalNative.Selector("newFunctionWithName:"), MetalNative.NsString(Gen5MslMeshLaunch.KernelFunction));
            MetalNative.SendVoid(library, SelRelease);
            if (function == 0)
            {
                throw Fatal("The indirect launch function is missing.");
            }

            error = 0;
            var pipeline = MetalNative.Send(
                _device, MetalNative.Selector("newComputePipelineStateWithFunction:error:"), function, ref error);
            MetalNative.SendVoid(function, SelRelease);
            if (pipeline == 0)
            {
                throw Fatal($"The indirect launch pipeline did not build: {MetalNative.DescribeError(error)}.");
            }

            _indirectLaunchPipeline = pipeline;
            return pipeline;
        }
        finally
        {
            MetalNative.objc_autoreleasePoolPop(pool);
        }
    }
}
