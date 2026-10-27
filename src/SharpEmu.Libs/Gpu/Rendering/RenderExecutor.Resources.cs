// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.Libs.VideoOut;
using WorkerScratch = SharpEmu.ShaderCompiler.WorkerScratch;

namespace SharpEmu.Libs.Gpu.Rendering;

public sealed partial class RenderExecutor
{
    private const uint TransientIndexAlignment = 16;

    private struct VertexBufferRange
    {
        public ulong BaseAddress;
        public ulong RequestedEnd;
        public ulong AcquiredEnd;
        public BufferBinding Binding;

        public readonly ulong RequestedSize => RequestedEnd - BaseAddress;
    }

    private readonly record struct PreparedIndexBuffer(BufferBinding Binding, ulong Size, GpuIndexType Type);

    // The bytes one fetch of a descriptor format reads (formats 113 and 121 are the two the
    // vertex input accepts outside the unified table).
    private uint ConstantAttributeByteSize(in BufferDescriptorWords descriptor)
    {
        if (descriptor.Format == 113) return 16;
        if (descriptor.Format == 121) return 4;
        if (SharpEmu.ShaderCompiler.Gfx10UnifiedFormat.TryDecode(descriptor.Format, out var dataFormat, out _))
        {
            var bytes = SharpEmu.ShaderCompiler.Gfx10UnifiedFormat.GetAccessByteSize(dataFormat, 4);
            if (bytes != 0) return bytes;
        }

        throw _host.Fatal($"The vertex buffer format is unknown: format={descriptor.Format}.");
    }

    // Merges the vertex ranges, obtains one host buffer per merged range and offsets every slot into it.
    // The bindings of one draw, handed to BindVertexBuffers as a span and not kept past that call
    // (the Metal host copies them into its slot tables), so the executor's draw thread reuses one array.
    private BufferBinding[]? _vertexBindingScratch;

    private ReadOnlySpan<BufferBinding> AcquireVertexBuffers(VertexInputInfo vertexInput)
    {
        using var profileScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.DrawVertexBufferAcquisition);
        var buffers = vertexInput.Buffers;
        if (buffers.Length > VertexInputInfo.MaxBuffers)
        {
            throw _host.Fatal($"The vertex input has too many buffers: count={buffers.Length} max={VertexInputInfo.MaxBuffers}.");
        }

        // Sized to this draw's buffers: a MaxBuffers span is over a kilobyte, zeroed on every draw.
        Span<VertexBufferRange> ranges = stackalloc VertexBufferRange[buffers.Length];
        Span<ulong> sizes = stackalloc ulong[buffers.Length];
        for (var slot = 0; slot < buffers.Length; slot++) sizes[slot] = buffers[slot].Size;
        foreach (var attribute in vertexInput.Attributes)
        {
            var slot = attribute.BufferIndex;
            var descriptor = attribute.Descriptor;
            if (buffers[slot].Stride != 0 || descriptor.RecordCount == 0 || descriptor.OutOfBounds != 2) continue;
            // This bounds mode tests only whether the record count is nonzero.
            // Constant attributes still need the full format extent.
            sizes[slot] = Math.Max(sizes[slot], (ulong)attribute.OffsetBytes + ConstantAttributeByteSize(descriptor));
        }
        var rangeCount = 0;
        for (var slot = 0; slot < buffers.Length; slot++)
        {
            ref readonly var vertex = ref buffers[slot];
            var size = sizes[slot];
            if (size == 0)
            {
                continue;
            }

            if (vertex.Address == 0 || size > ulong.MaxValue - vertex.Address)
            {
                throw _host.Fatal($"The vertex buffer range is invalid: address=0x{vertex.Address:X16} size=0x{size:X16}.");
            }

            ranges[rangeCount++] = new VertexBufferRange { BaseAddress = vertex.Address, RequestedEnd = vertex.Address + size };
        }

        ranges[..rangeCount].Sort(static (left, right) => left.BaseAddress.CompareTo(right.BaseAddress));
        Span<VertexBufferRange> merged = stackalloc VertexBufferRange[buffers.Length];
        var mergedCount = 0;
        for (var i = 0; i < rangeCount; i++)
        {
            ref readonly var range = ref ranges[i];
            if (mergedCount != 0 && merged[mergedCount - 1].RequestedEnd >= range.BaseAddress)
            {
                merged[mergedCount - 1].RequestedEnd = Math.Max(merged[mergedCount - 1].RequestedEnd, range.RequestedEnd);
                continue;
            }

            merged[mergedCount++] = new VertexBufferRange { BaseAddress = range.BaseAddress, RequestedEnd = range.RequestedEnd };
        }

        for (var i = 0; i < mergedCount; i++)
        {
            ref var range = ref merged[i];
            var size = _host.ClampMappedSize(range.BaseAddress, range.RequestedSize);
            range.AcquiredEnd = range.BaseAddress + size;
            range.Binding = _host.ObtainBuffer(range.BaseAddress, size, isWritten: false);
        }

        BufferBinding[] prepared;
        if (WorkerScratch.Enabled)
        {
            prepared = _vertexBindingScratch ??= new BufferBinding[VertexInputInfo.MaxBuffers];
            WorkerScratch.Note(WorkerScratch.Site.VertexBindings, reused: true);
        }
        else
        {
            prepared = new BufferBinding[buffers.Length];
            WorkerScratch.Note(WorkerScratch.Site.VertexBindings, reused: false);
        }

        BufferBinding? nullBuffer = null;
        Span<ulong> acquiredBytes = stackalloc ulong[buffers.Length];
        acquiredBytes.Clear();
        for (var slot = 0; slot < buffers.Length; slot++)
        {
            ref readonly var vertex = ref buffers[slot];
            if (sizes[slot] == 0)
            {
                nullBuffer ??= _host.NullBuffer;
                prepared[slot] = nullBuffer.Value;
                continue;
            }

            var found = -1;
            for (var i = 0; i < mergedCount; i++)
            {
                if (vertex.Address >= merged[i].BaseAddress && vertex.Address < merged[i].AcquiredEnd)
                {
                    found = i;
                    break;
                }
            }

            if (found < 0)
            {
                throw _host.Fatal($"The vertex buffer address is outside the acquired range: address=0x{vertex.Address:X16}.");
            }

            ref readonly var owner = ref merged[found];
            prepared[slot] = new BufferBinding(owner.Binding.Handle, owner.Binding.Offset + vertex.Address - owner.BaseAddress);
            acquiredBytes[slot] = owner.AcquiredEnd - vertex.Address;
        }

        foreach (var attribute in vertexInput.Attributes)
        {
            var slot = attribute.BufferIndex;
            _host.TraceVertexBuffer(vertexInput.Stage.Program?.Hash ?? 0, attribute, buffers[slot],
                prepared[slot], sizes[slot], acquiredBytes[slot]);
        }

        return prepared.AsSpan(0, buffers.Length);
    }

    private PreparedIndexBuffer AcquireIndexBuffer(in IndexSource source)
    {
        using var profileScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.DrawIndexBufferAcquisition);
        if (!source.Enabled)
        {
            return default;
        }

        if (source.Size == 0)
        {
            throw _host.Fatal($"The index buffer is empty: address=0x{source.Address:X16} type={(int)source.Type}.");
        }

        var binding = source.HostData is { } hostData
            ? _host.UploadTransient(hostData.AsSpan(0, checked((int)source.Size)), TransientIndexAlignment)
            : _host.ObtainBuffer(source.Address, source.Size, isWritten: false);
        return new PreparedIndexBuffer(binding, source.Size, source.Type);
    }

    // A written buffer resource with an address and a footprint needs a barrier after the stage.
    // Only a store or an atomic through the device-address page table counts: a program that
    // merely reads through it writes nothing, and ending the pass after it would store and
    // reload every attachment on a tile-based GPU for no reason.
    private bool DrawWritesMemory(ShaderStageResources stage) =>
        stage.Program is { } program &&
        (program.WritesDeviceAddresses || WritesStorageImage(program) || HasBufferWrites(stage));

    private bool HasBufferWrites(ShaderStageResources stage)
    {
        var program = stage.Program ?? throw _host.Fatal("A shader stage has no program.");
        var descriptors = stage.Resources.Buffers;
        if (descriptors.Length != program.Buffers.Length)
        {
            throw _host.Fatal($"The buffer descriptor count does not match the program: descriptors={descriptors.Length} program={program.Buffers.Length}.");
        }

        for (var i = 0; i < program.Buffers.Length; i++)
        {
            if (!program.Buffers[i].Written)
            {
                continue;
            }

            if (descriptors[i].Length < 4)
            {
                throw _host.Fatal($"A written buffer descriptor is too short: index={i} words={descriptors[i].Length}.");
            }

            var descriptor = BufferDescriptorWords.From(descriptors[i]);
            var footprint = descriptor.Footprint() ?? throw _host.Fatal(
                $"The written buffer footprint overflows: index={i} stride={descriptor.Stride} records={descriptor.RecordCount}.");
            if (descriptor.Address != 0 && footprint != 0)
            {
                return true;
            }
        }

        return false;
    }

    // Set from the draw packet: predicated, or inside an occlusion query (DrawsMustExecute).
    private bool _drawMustExecute;

    /// <summary>
    /// Whether this draw may be left out of the frame while its pipeline compiles, rather than
    /// the command stream waiting for the compile. Leaving a draw out is only safe when the frame
    /// it belongs to is the only thing that misses it: the next frame draws it again into
    /// targets that are rebuilt from nothing every frame. So, when in doubt, it is not:
    /// <list type="bullet">
    /// <item>a program that writes memory (storage buffers or images, atomics, device-address
    /// stores), uses the GDS (ordered counts, append and consume), or writes mip statistics:
    /// what it writes outlives the frame and later work reads it. Mip statistics only when the
    /// host leaves those out too (IRenderHost.LeavesOutMipStatisticsWhileCompiling): they are
    /// the texture streamer's feedback, gathered over frames, and a draw missing from a few
    /// frames only reports its object's mips that much later;</item>
    /// <item>a predicated draw, or one inside an occlusion query: the guest decides on the
    /// count it produces (the interpreter's DrawsMustExecute);</item>
    /// <item>an indirect mesh draw, whose launch was prepared before the pipeline;</item>
    /// <item>a draw whose attachments carry a clear (a fast clear consumed at their
    /// acquire, a load clear): leaving it out would lose the clear;</item>
    /// <item>a draw that writes depth or stencil, unless the host leaves those out too
    /// (IRenderHost.LeavesOutDepthWritesWhileCompiling): the depth target is what the rest of
    /// the frame tests and samples, and a missing occluder there shows in every later pass of
    /// the frame - but of that frame alone when the host's check below finds the depth target
    /// rebuilt every frame as well;</item>
    /// <item>and the host's own say on the targets (IRenderHost.MayLeaveOutWhileCompiling),
    /// which leaves out only draws into targets it saw cleared or overwritten whole in this
    /// frame and in the frames before it, so a look-up table, a shadow or virtual-texture
    /// cache, a capture or a history buffer is never one of them.</item>
    /// </list>
    /// This executor has no stream-out; a host that adds it treats such a draw as writing memory.
    /// </summary>
    private bool MayLeaveOutWhileCompiling(
        ref DrawState state, ShaderStageResources vertexStage, ShaderStageResources? pixelStage, bool indirectMesh)
    {
        if (!MayLeaveOutBeforeAcquiring(ref state, vertexStage, pixelStage, indirectMesh))
        {
            return false;
        }

        ref readonly var rendering = ref state.Rendering;
        for (var index = 0; index < rendering.ColorAttachmentCount; index++)
        {
            if (rendering.ColorAttachments[index].IsClear)
            {
                _host.NoteLeaveOutRefused("color-clear");
                return false;
            }
        }

        if (state.Depth.HasTarget &&
            (rendering.DepthStencilAttachment.DepthClear || rendering.DepthStencilAttachment.StencilClear))
        {
            _host.NoteLeaveOutRefused("depth-clear");
            return false;
        }

        return _host.MayLeaveOutWhileCompiling(in rendering);
    }

    // The part of MayLeaveOutWhileCompiling the draw knows before it acquires anything. False
    // here is false there: acquiring only adds reasons to keep the draw (a clear it consumes, a
    // metadata clear that makes the depth load a write, the host's say on the targets).
    private bool MayLeaveOutBeforeAcquiring(
        ref DrawState state, ShaderStageResources vertexStage, ShaderStageResources? pixelStage, bool indirectMesh)
    {
        if (!_host.LeavesOutDrawsWhileCompiling)
        {
            return false;
        }

        if (_drawMustExecute || indirectMesh)
        {
            _host.NoteLeaveOutRefused("must-execute");
            return false;
        }

        var mipStatistics = _host.LeavesOutMipStatisticsWhileCompiling;
        if (WriteOutsideTargets(vertexStage, mipStatistics) is { } vertexWrite)
        {
            _host.NoteLeaveOutRefused(vertexWrite);
            return false;
        }

        if (pixelStage is { } pixel && WriteOutsideTargets(pixel, mipStatistics) is { } pixelWrite)
        {
            _host.NoteLeaveOutRefused(pixelWrite);
            return false;
        }

        if (state.Depth.HasTarget)
        {
            var depth = state.Depth.Target;
            if (state.Depth.LoadClear || depth.State.StencilClearEnabled)
            {
                _host.NoteLeaveOutRefused("depth-clear");
                return false;
            }

            if ((!state.Depth.LoadState.IsReadOnly(depth.Target.Format) || depth.State.DepthWriteEnabled) &&
                !_host.LeavesOutDepthWritesWhileCompiling)
            {
                _host.NoteLeaveOutRefused("depth-write");
                return false;
            }
        }

        return true;
    }

    // Any write a draw makes other than through its attachments, as the conservative reading of
    // the program's resource use: a written or atomic buffer whatever its descriptor holds, any
    // storage image, the GDS, mip statistics and device-address stores.
    internal static bool WritesOutsideTargets(ShaderStageResources stage) => WriteOutsideTargets(stage) is not null;

    // What WritesOutsideTargets found, as a literal for the leave-out diagnostics; null for nothing.
    // Mip statistics are left out of the count when the host says so (IRenderHost.LeavesOutMipStatisticsWhileCompiling).
    internal static string? WriteOutsideTargets(ShaderStageResources stage, bool mipStatisticsAllowed = false)
    {
        if (stage.Program is not { } program)
        {
            return null;
        }

        if (program.WritesDeviceAddresses)
        {
            return "device-address-store";
        }

        if (!mipStatisticsAllowed && (program.Bindings?.MipStatisticsCount ?? 0) != 0)
        {
            return "mip-statistics";
        }

        foreach (var buffer in program.Buffers)
        {
            if (buffer.Atomic)
            {
                return "atomic-buffer";
            }

            if (buffer.Written)
            {
                return "written-buffer";
            }
        }

        foreach (var image in program.Images)
        {
            if (image.Written || image.Class == ImageResourceClass.Storage)
            {
                return "storage-image";
            }
        }

        if (program.Bindings is { } bindings)
        {
            // By index: a foreach over the interface boxes its enumerator, once per draw.
            var descriptors = bindings.Descriptors;
            for (var index = 0; index < descriptors.Count; index++)
            {
                if (descriptors[index].Kind == SharpEmu.ShaderCompiler.Resources.DescriptorBindingKind.GlobalDataShare)
                {
                    return "gds";
                }
            }
        }

        return null;
    }

    private void SetDrawDebugPhase(ulong submitId, in DrawCall draw, uint phase) =>
        _host.SetDebugInformation(draw.Operation, submitId, phase, draw.Count, 0, draw.InstanceCount, draw.FirstInstance);

    private IPreparedBindings PrepareBindings(in ShaderStageResources stage)
    {
        var bindings = _host.PrepareBindings(stage);
        NoteSampledTextures(stage);
        return bindings;
    }

    private static void NoteSampledTextures(in ShaderStageResources stage)
    {
        if (stage.Program is not { } program)
        {
            return;
        }

        var descriptors = stage.Resources.Images;
        for (var index = 0; index < program.Images.Length && index < descriptors.Length; index++)
        {
            if (program.Images[index].Class == ImageResourceClass.Sampled)
            {
                MipStatistics.Shared.NoteSampled(new TextureDescriptorWords(descriptors[index]));
            }
        }
    }

    // Binds everything the draw needs inside one preparation scope, then records it.
    private string RecordDraw(
        ulong submitId,
        RegisterBanks banks,
        in DrawCall draw,
        ref DrawState state,
        GpuPrimitiveTopology topology,
        in DrawEmission emission,
        in IndexSource indexSource,
        bool primitiveRestart,
        bool setBindDebug,
        bool setAutoDebug,
        IndexedDrawTrace.Sample? indexedTrace = null,
        ulong packetAddress = 0)
    {
        using var profileScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.DrawResourcePreparation);
        if (emission.IndirectArgumentsAddress != 0 &&
            FragmentWritesMayCover(
                emission.IndirectArgumentsAddress,
                emission.Indexed ? IndexedIndirectArgumentsSize : AutoIndirectArgumentsSize))
        {
            // The GPU reads the arguments, which a deferred fragment store may have written.
            // Only then: ending the pass stores and reloads every attachment, and a run of
            // GPU-driven draws whose fragment stages store (a material or tile pass) would
            // otherwise end one pass per draw.
            ReleaseFragmentWrites();
        }

        var context = banks.Context;
        var vertexInput = state.Programs.VertexInput;
        var pixelInput = state.Programs.PixelInput;
        var meshInput = state.Programs.MeshInput;
        if (!MayLeaveOutBeforeAcquiring(ref state, vertexInput.Stage, state.PixelActive ? pixelInput.Stage : null,
                meshInput is not null && emission.IndirectArgumentsAddress != 0))
        {
            // The draw will wait for its pipeline: its programs' compiles are waited for here,
            // before anything of it is prepared, where the host may still serve what waits for
            // a packet boundary.
            _host.AwaitProgramsBeforePreparation(state.Programs.Vertex, state.PixelActive ? state.Programs.Pixel : default);
        }

        if (state.PixelActive && (pixelInput.Stage.Program?.UsesDeviceAddresses ?? false))
        {
            _host.BeforeDeviceAddressPreparation([vertexInput.Stage, pixelInput.Stage]);
        }
        else if (vertexInput.Stage.Program?.UsesDeviceAddresses ?? false)
        {
            _host.BeforeDeviceAddressPreparation([vertexInput.Stage]);
        }

        using var preparation = _host.BeginPreparation();
        using var indirectMesh = meshInput is not null && emission.IndirectArgumentsAddress != 0
            ? _host.PrepareIndirectMeshDraw(new IndirectMeshDrawRequest(emission.IndirectArgumentsAddress,
                indexSource.Address, draw.Count, indexSource.Type == GpuIndexType.Uint16 ? 2u : 4u,
                topology == GpuPrimitiveTopology.PointList, meshInput.Geometry.InputPrimitiveCountPerWorkgroup, meshInput.Execution))
            : null;
        var meshVertexCount = draw.Count;
        if (MeshDrawTrace.Active)
            MeshDrawTrace.Write("programs", $"mesh={meshInput is not null} vertexHash=0x{vertexInput.Stage.Program?.Hash:X16} pixelHash=0x{pixelInput.Stage.Program?.Hash:X16} pixelActive={state.PixelActive}");
        if (meshInput is not null)
        {
            if (topology is not (GpuPrimitiveTopology.PointList or GpuPrimitiveTopology.LineList or GpuPrimitiveTopology.TriangleList or GpuPrimitiveTopology.TriangleStrip))
            {
                throw _host.Fatal("The mesh draw input topology is not supported.");
            }

            ulong indexAddress = 0;
            if (emission.Indexed && indirectMesh is null)
            {
                var byteCount = checked((int)indexSource.Size);
                var elementSize = indexSource.Type == GpuIndexType.Uint16 ? 2u : 4u;
                var triangleStrip = topology == GpuPrimitiveTopology.TriangleStrip;
                var lineList = topology == GpuPrimitiveTopology.LineList;
                var pointList = lineList || topology == GpuPrimitiveTopology.PointList;
                var capacity = pointList ? byteCount / (int)elementSize
                    : MeshIndexAssembly.GetMaximumCount(byteCount, elementSize, triangleStrip);
                if (capacity == 0) return "no-primitives";
                byte[]? rentedBytes = null;
                uint[]? rentedIndices = null;
                try
                {
                    var bytes = indexSource.HostData is { } hostData ? hostData.AsSpan(0, byteCount)
                        : (rentedBytes = ArrayPool<byte>.Shared.Rent(byteCount)).AsSpan(0, byteCount);
                    if (indexSource.HostData is null && !_host.TryReadGuest(indexSource.Address, bytes))
                        throw _host.Fatal("The mesh index buffer cannot be read.");
                    rentedIndices = ArrayPool<uint>.Shared.Rent(capacity);
                    var written = pointList
                        ? MeshIndexAssembly.ExpandPoints(bytes, rentedIndices.AsSpan(0, capacity), elementSize,
                            emission.VertexOffset, primitiveRestart, elementSize == 2 ? ushort.MaxValue : uint.MaxValue)
                        : MeshIndexAssembly.ExpandTriangles(bytes, rentedIndices.AsSpan(0, capacity),
                        elementSize, emission.VertexOffset, triangleStrip, primitiveRestart,
                        elementSize == 2 ? ushort.MaxValue : uint.MaxValue, context.RasterMode.ProvokingVertexLast);
                    if (lineList) written &= ~1;
                    meshVertexCount = checked((uint)written);
                    if (written == 0) return "no-primitives";
                    indexAddress = _host.UploadMeshVertexIndices(rentedIndices.AsSpan(0, written));
                }
                finally
                {
                    if (rentedIndices is not null) ArrayPool<uint>.Shared.Return(rentedIndices);
                    if (rentedBytes is not null) ArrayPool<byte>.Shared.Return(rentedBytes);
                }
            }
            vertexInput.Stage = vertexInput.Stage with
            {
                MeshDraw = indirectMesh is null
                    ? new MeshDrawParameters(meshVertexCount, emission.Indexed ? 0 : emission.FirstVertex,
                        emission.FirstInstance, emission.Indexed ? 4u : 0u, indexAddress)
                    : new MeshDrawParameters(0, 0, 0, uint.MaxValue, indirectMesh.ParametersAddress),
            };
        }
        IPreparedBindings vertexBindings;
        IPreparedBindings? pixelBindings;
        try
        {
            vertexBindings = PrepareBindings(vertexInput.Stage);
            pixelBindings = state.PixelActive ? PrepareBindings(pixelInput.Stage) : null;
        }
        catch (DrawImageTypeMismatchException rejection)
        {
            if (MeshDrawTrace.Active) MeshDrawTrace.Write("resource-rejected", rejection.Message);
            IndexedDrawTrace.Write(indexedTrace, "resource-rejected", submitId, packetAddress);
            if (_strictDrawResources)
            {
                throw _host.Fatal(rejection.Message);
            }

            if (_reportedDrawImageTypeMismatches.Add(rejection.WarningKey))
            {
                Console.Error.WriteLine($"[GPU][WARN][DRAW_SKIPPED] {rejection.Message} " +
                    "The draw was not executed. Images and FPS can be incorrect. Set SHARPEMU_STRICT_COMPUTE=1 to stop on this failure.");
            }

            return "resource-rejected";
        }
        var vertexProgram = vertexInput.Stage.Program ?? throw _host.Fatal("The vertex stage has no program.");
        var pixelProgram = pixelBindings is null ? null : pixelInput.Stage.Program ?? throw _host.Fatal("The pixel stage has no program.");
        if (vertexProgram.UsesDeviceAddresses || (pixelProgram?.UsesDeviceAddresses ?? false))
        {
            _host.PrepareDeviceAddresses();
        }

        _host.BindResources(vertexBindings);
        if (pixelBindings is not null)
        {
            _host.BindResources(pixelBindings);
        }

        var vertexBuffers = meshInput is null ? AcquireVertexBuffers(vertexInput) : default;
        var indexBuffer = meshInput is null ? AcquireIndexBuffer(in indexSource) : default;
        var indirectArguments = meshInput is null && emission.IndirectArgumentsAddress != 0
            ? _host.ObtainBuffer(
                emission.IndirectArgumentsAddress,
                emission.Indexed ? IndexedIndirectArgumentsSize : AutoIndirectArgumentsSize,
                isWritten: false)
            : default;
        DropUnwrittenColorTargets(context, ref state, pixelProgram);
        state.Rendering = AcquireAttachments(context, ref state);
        // Nothing after the pipeline touches guest memory. A draw that may be left out while its
        // pipeline compiles is left out here, before anything of it is recorded: what came before
        // only bound state the caller's ResetBindings drops, and uploaded what it reads.
        PipelineHandle pipeline;
        if (MayLeaveOutWhileCompiling(ref state, vertexInput.Stage, pixelBindings is not null ? pixelInput.Stage : null,
                indirectMesh is not null))
        {
            if (!_pipelines.TryCreateGraphicsPipeline(
                    BoundColors(ref state),
                    in state.Depth,
                    vertexInput,
                    state.PixelActive ? pixelInput : null,
                    context,
                    in state.Rendering,
                    topology,
                    primitiveRestart,
                    state.Programs.DisableBlending,
                    state.Programs.Vertex,
                    state.Programs.Pixel,
                    out pipeline))
            {
                IndexedDrawTrace.Write(indexedTrace, "pipeline-compiling", submitId, packetAddress);
                return "pipeline-compiling";
            }
        }
        else
        {
            pipeline = _pipelines.CreateGraphicsPipeline(
                BoundColors(ref state),
                in state.Depth,
                vertexInput,
                state.PixelActive ? pixelInput : null,
                context,
                in state.Rendering,
                topology,
                primitiveRestart,
                state.Programs.DisableBlending,
                state.Programs.Vertex,
                state.Programs.Pixel);
        }

        if (setBindDebug)
        {
            SetDrawDebugPhase(submitId, in draw, 0x100);
        }

        if (setAutoDebug)
        {
            SetDrawDebugPhase(submitId, in draw, 0x200);
        }

        if (meshInput is null)
        {
            _host.BindVertexBuffers(vertexBuffers, vertexInput);
        }

        if (pixelBindings is not null && setAutoDebug)
        {
            SetDrawDebugPhase(submitId, in draw, 0x300);
        }

        Span<IPreparedBindings> stages = pixelBindings is null ? [vertexBindings] : [vertexBindings, pixelBindings];
        _host.CommitBindings(GpuBindPoint.Graphics, in pipeline, stages);
        if (indexBuffer.Size != 0)
        {
            _host.BindIndexBuffer(indexBuffer.Binding, indexBuffer.Type);
            _host.NoteIndexSource(indexSource.HostData is null ? indexSource.Address : 0, indexSource.Size);
        }

        var dynamicState = BuildDynamicState(banks, in state, submitId, packetAddress);
        if (dynamicState.DepthBiasPerFace && DrawWritesMemory(vertexInput.Stage))
        {
            // The host draws each face separately, which runs the vertex stage twice.
            throw _host.Fatal(
                "A draw whose front and back faces take different polygon offsets writes memory from its vertex stage: " +
                $"vertex=0x{vertexInput.Stage.Program?.Hash ?? 0:X16}.");
        }

        _host.SetDynamicState(dynamicState);
        if (setAutoDebug)
        {
            SetDrawDebugPhase(submitId, in draw, 0x400);
        }

        if (DrawWritesMemory(vertexInput.Stage) || (pixelBindings is not null && DrawWritesMemory(pixelInput.Stage)))
        {
            _host.PrepareMemoryWritingDraw();
        }

        var imageSnapshot = _host.BeginImageSnapshot(pixelBindings);
        indirectMesh?.Prepare();
        if (meshInput is null && emission.IndirectArgumentsAddress != 0)
        {
            _host.PrepareIndirectDraw(
                in pipeline, emission.IndirectArgumentsAddress, indirectArguments, emission.Indexed, emission.IndirectIndexLimit);
        }

        var (upcomingCount, upcomingFirst) = UpcomingVertices(banks.UserConfig, in draw, in emission);
        _host.NoteUpcomingDraw(in pipeline, upcomingCount, upcomingFirst);
        _host.BeginRendering(in state.Rendering);
        if (MeshDrawTrace.Active) MeshDrawTrace.Write("rendering", $"width={state.Rendering.Width} height={state.Rendering.Height} layers={state.Rendering.Layers} colors={state.ColorCount}");
        _host.BindPipeline(GpuBindPoint.Graphics, in pipeline);
        if (setAutoDebug)
        {
            SetDrawDebugPhase(submitId, in draw, 0x500);
        }

        if (indexedTrace is { } trace)
            _host.InsertDrawTraceMarker(IndexedDrawTrace.CreateLabel(trace, submitId, packetAddress));

        if (indirectMesh is not null)
        {
            indirectMesh.Draw();
            if (MeshDrawTrace.Active) MeshDrawTrace.Recorded();
        }
        else if (meshInput is { } mesh)
        {
            var primitiveCount = mesh.Geometry.InputPointList ? meshVertexCount
                : mesh.Geometry.InputLineList ? meshVertexCount / 2
                : mesh.Geometry.InputTriangleStrip ? (meshVertexCount >= 3 ? meshVertexCount - 2 : 0) : meshVertexCount / 3;
            var groupCount = (uint)(((ulong)primitiveCount + mesh.Geometry.InputPrimitiveCountPerWorkgroup - 1) / mesh.Geometry.InputPrimitiveCountPerWorkgroup);
            var totalGroupCount = (ulong)groupCount * draw.InstanceCount;
            if (groupCount > mesh.Execution.MaxGroupCountX || draw.InstanceCount > mesh.Execution.MaxGroupCountY ||
                totalGroupCount > mesh.Execution.MaxGroupTotalCount)
            {
                throw _host.Fatal($"The mesh draw exceeds device group limits: groups={groupCount}/{mesh.Execution.MaxGroupCountX} " +
                    $"instances={draw.InstanceCount}/{mesh.Execution.MaxGroupCountY} " +
                    $"total={totalGroupCount}/{mesh.Execution.MaxGroupTotalCount}.");
            }
            if (RenderTrace.Enabled)
            {
                RenderTrace.Write($"MeshDraw shader=0x{banks.Shader.Vertex.ExportAddress:X16} " +
                    $"vertices={draw.Count} instances={draw.InstanceCount} groups={groupCount} " +
                    $"threads={mesh.Geometry.ThreadsPerGroup} ldsDwords={mesh.Geometry.LocalDataShareDwords}");
            }
            _host.DrawMeshTasks(groupCount, draw.InstanceCount, 1);
            if (MeshDrawTrace.Active)
            {
                MeshDrawTrace.Write("command-recorded", $"groups={groupCount}/{draw.InstanceCount}/1 threads={mesh.Geometry.ThreadsPerGroup} wave={mesh.Geometry.WaveSize} lds={mesh.Geometry.LocalDataShareDwords} firstVertex={emission.FirstVertex} firstInstance={emission.FirstInstance}");
                MeshDrawTrace.Recorded();
            }
        }
        else if (emission.IndirectArgumentsAddress != 0)
        {
            // Uploads and shader writes end with barriers to all commands, so the
            // indirect read sees them.
            if (emission.Indexed)
            {
                _host.DrawIndexedIndirect(indirectArguments);
            }
            else
            {
                _host.DrawIndirect(indirectArguments);
            }
        }
        else
        {
            EmitDraw(banks.UserConfig, vertexInput, in draw, in emission);
        }
        _host.EndImageSnapshot(imageSnapshot, BoundColors(ref state));
        IndexedDrawTrace.Write(indexedTrace, "emitted", submitId, packetAddress);
        if (setAutoDebug)
        {
            SetDrawDebugPhase(submitId, in draw, 0x600);
        }

        var writeStages = GpuPipelineStage.None;
        if (HasBufferWrites(vertexInput.Stage))
        {
            writeStages |= meshInput is null ? GpuPipelineStage.VertexShaderBit : GpuPipelineStage.MeshShaderBitExt;
        }
        else if (vertexProgram.WritesDeviceAddresses || WritesStorageImage(vertexProgram))
        {
            _vertexStoresUnordered = true;
            _vertexStoresPass = _host.RenderPassSerial;
        }

        if (state.PixelActive && DrawWritesMemory(pixelInput.Stage))
        {
            writeStages |= GpuPipelineStage.FragmentShaderBit;
        }

        if ((writeStages & GpuPipelineStage.FragmentShaderBit) != 0 && _host.DefersFragmentWritePassEnd &&
            ((writeStages & GpuPipelineStage.VertexShaderBit) == 0 || _host.OrdersVertexWritesInsidePass))
        {
            // Fragment stores cannot be made visible mid-pass on a tile-based GPU, but nothing
            // in this pass reads them before the guest says so; the pass end and the barrier
            // wait for the first consumer (ReleaseFragmentWrites).
            writeStages &= ~GpuPipelineStage.FragmentShaderBit;
            NoteFragmentWrites(pixelInput.Stage);
            _fragmentWritesPending = true;
            _fragmentWritesPass = _host.RenderPassSerial;
        }

        if (writeStages != GpuPipelineStage.None)
        {
            if (MeshDrawTrace.Active) MeshDrawTrace.Write("write-barrier", $"sourceStages={writeStages}");
            // Vertex-stage stores can be ordered inside the pass by a host that supports it;
            // ending the pass would store and reload every attachment for each such draw.
            if ((writeStages & GpuPipelineStage.FragmentShaderBit) != 0 || !_host.OrdersVertexWritesInsidePass)
            {
                _host.EndRendering("barrier");
            }

            _host.ShaderWriteBarrier(writeStages);
        }

        if (setAutoDebug)
        {
            SetDrawDebugPhase(submitId, in draw, 0x700);
        }

        return "recorded";
    }

    private const ulong IndexedIndirectArgumentsSize = 20;

    // The vertices of the first host draw call EmitDraw makes; a zero count for an indexed or
    // indirect draw, whose vertices are not known before it runs.
    private static (uint Count, uint First) UpcomingVertices(
        UserConfigRegisters userConfig, in DrawCall draw, in DrawEmission emission)
    {
        if (emission.Indexed || emission.IndirectArgumentsAddress != 0 || emission.PatchControlPoints != 0)
        {
            return (0, 0);
        }

        return (GuestPrimitiveType)userConfig.PrimitiveType switch
        {
            GuestPrimitiveType.QuadListLegacy => (4, emission.FirstVertex),
            _ => (draw.Count, emission.FirstVertex),
        };
    }

    private void EmitDraw(UserConfigRegisters userConfig, VertexInputInfo vertexInput, in DrawCall draw, in DrawEmission emission)
    {
        if (emission.TriangleListIndices != 0)
        {
            _host.DrawIndexed(emission.TriangleListIndices, draw.InstanceCount, 0, emission.VertexOffset, emission.FirstInstance);
            return;
        }

        if (emission.PatchControlPoints != 0)
        {
            _host.DrawPatches(emission.PatchControlPoints, draw.Count, draw.InstanceCount, emission.FirstInstance);
            return;
        }

        switch ((GuestPrimitiveType)userConfig.PrimitiveType)
        {
            case GuestPrimitiveType.PointList:
            case GuestPrimitiveType.LineList:
            case GuestPrimitiveType.LineStrip:
            case GuestPrimitiveType.TriangleList:
            case GuestPrimitiveType.TriangleFan:
            case GuestPrimitiveType.TriangleStrip:
            case GuestPrimitiveType.Polygon:
            case GuestPrimitiveType.RectangleList:
            case GuestPrimitiveType.RectangleListLegacy:
                if (emission.Indexed)
                {
                    _host.DrawIndexed(draw.Count, draw.InstanceCount, 0, emission.VertexOffset, emission.FirstInstance);
                }
                else
                {
                    _host.Draw(draw.Count, draw.InstanceCount, emission.FirstVertex, emission.FirstInstance);
                }

                break;
            case GuestPrimitiveType.QuadListLegacy:
                if ((draw.Count & 0x3) != 0)
                {
                    throw _host.Fatal($"A legacy quad list count is not a multiple of four: count={draw.Count}.");
                }

                for (var i = 0u; i < draw.Count; i += 4)
                {
                    if (emission.Indexed)
                    {
                        _host.DrawIndexed(4, draw.InstanceCount, i, emission.VertexOffset, emission.FirstInstance);
                    }
                    else
                    {
                        _host.Draw(4, draw.InstanceCount, i + emission.FirstVertex, emission.FirstInstance);
                    }
                }

                break;
            default:
                throw _host.Fatal($"The primitive type is unknown: primitiveType={userConfig.PrimitiveType}.");
        }
    }
}
