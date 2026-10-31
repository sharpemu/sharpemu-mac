// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.VideoOut;
using SharpEmu.ShaderCompiler;

namespace SharpEmu.Libs.Gpu.Rendering;

// A buffer a metadata clear candidate writes or reads, as the resolver decoded it.
internal readonly record struct MetadataClearProbe(ulong Address, ulong Bytes, uint Records, uint Stride, bool Written, bool Read);

// A bounded fill a host may take in place of the kernel: DCC metadata when every pattern word is
// the value, otherwise a whole image cleared to the pattern (an RGBA16F colour is two words).
internal struct KernelFillCandidate
{
    [System.Runtime.CompilerServices.InlineArray(8)]
    public struct PatternWords
    {
        private uint _element0;
    }

    public ulong Address;
    public ulong Size;
    public uint Value;
    public bool Uniform;
    public int PatternLength;
    public PatternWords Pattern;
}

// The dispatches suspended on their pipeline, by submission: a submission runs its packets in
// order, so the next dispatch it reaches is the one that suspended. The address and the indirect
// arguments make sure of it; the retry itself checks everything else it read. Used only by a
// dispatch recorded at once, on the thread that resolved it.
internal sealed class ComputeDispatchSuspensions
{
    private readonly Dictionary<ulong, (ulong Address, ulong IndirectArguments, ComputeProgramRetry Retry)> _suspended = [];
    private ComputeProgramRetry? _spare;

    // The retry of this dispatch, or an empty one that logs what the preparation reads. The
    // empty one is reused by every dispatch that does not suspend; one that does keeps it.
    public ComputeProgramRetry Take(ulong submitId, ulong address, ulong indirectArguments)
    {
        if (_suspended.Remove(submitId, out var suspended) &&
            suspended.Address == address && suspended.IndirectArguments == indirectArguments)
        {
            return suspended.Retry;
        }

        var retry = _spare ??= new ComputeProgramRetry();
        retry.Reset();
        return retry;
    }

    public void Keep(ulong submitId, ulong address, ulong indirectArguments, ComputeProgramRetry? retry)
    {
        if (retry is null)
        {
            return;
        }

        if (_suspended.Count >= 64)
        {
            // Submissions that never came back (a queue torn down mid-wait) are not kept for ever.
            _suspended.Clear();
        }

        _suspended[submitId] = (address, indirectArguments, retry);
        if (ReferenceEquals(retry, _spare))
        {
            _spare = null;
        }
    }
}

internal sealed partial class RenderRecorder
{
    private ComputeProgram? _keptComputeProgram;
    private PipelineHandle _keptComputePipeline;
    private DispatchChain? _dispatchChain;

    // Each program and surface it was taken to clear, reported once.
    private readonly HashSet<(ulong Hash, ulong Address)> _loggedMetadataClears = new();

    // Each program and HTILE surface NoteHtileStores reported, once.
    private readonly HashSet<(ulong Hash, ulong Address)> _loggedHtileStores = new();

    public void Record(ResolvedDispatch record)
    {
        try
        {
            RecordDispatch(record);
        }
        finally
        {
            _pendingWrites?.Release(record.Sequence);
            if (_returnsRecords)
            {
                _pool.Return(record);
            }
        }
    }

    private void RecordDispatch(ResolvedDispatch record)
    {
        if (!record.HostPrepared)
        {
            if (!_host.IsRecording)
            {
                throw _host.Fatal("A dispatch has no recording command buffer.");
            }

            _host.RunPendingOperations();
            SetDebugInformation(in record.Debug);
        }

        if (record.Skipped)
        {
            return;
        }

        var computeProgram = record.Program;
        if (computeProgram.ImageCopy is { } copy)
        {
            if (_host.TryCopyImage(copy.Source, copy.Destination))
            {
                _host.ResetBindings();
                return;
            }

            computeProgram = copy.Fallback();
            record.Program = computeProgram;
            if (!computeProgram.Available || computeProgram.Consumed)
            {
                return;
            }
        }

        var input = computeProgram.Input;
        var program = input.Stage.Program!;
        var shader = record.ShaderAddress;
        if (record.IndirectArgumentsAddress == 0 && !record.Retried)
        {
            bool consumed;
            using (RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.DispatchConsume))
            {
                consumed = TryConsume(record, program, out var disposition);
                if (consumed)
                {
                    RenderExecutor.TraceFrameDispatch(_host, disposition!, shader, program.Hash, record.GroupsX, record.GroupsY, record.GroupsZ, 0);
                }
            }

            var kinds = (record.MetadataClearCount != 0 ? 1 : 0) | (record.ConstantFill is not null ? 2 : 0) | (record.BoundedFill is not null ? 4 : 0) |
                (record.BoundedCopy is not null ? 8 : 0) | (record.ImageClear is not null ? 16 : 0);
            if (kinds != 0)
            {
                Diagnostics.DispatchRunTally.NoteConsume(program.Hash, kinds, consumed);
            }

            if (consumed)
            {
                _host.ResetBindings();
                return;
            }
        }

        NoteHtileStores(input);

        if (record.IndirectArgumentsAddress == 0 && (record.GroupsX == 0 || record.GroupsY == 0 || record.GroupsZ == 0))
        {
            if (RenderTrace.Enabled && RenderTrace.ZeroDispatch())
            {
                RenderTrace.Write($"Skipping a zero-sized dispatch: groups={record.GroupsX}x{record.GroupsY}x{record.GroupsZ} shader=0x{shader:X16}");
            }

            _host.ResetBindings();
            return;
        }

        var suspendable = record.Suspensions is not null && _host.SupportsPipelineSuspend;
        if (!suspendable)
        {
            // The dispatch will wait for its pipeline below; its program's compile is waited
            // for here instead, before the preparation opens.
            _host.AwaitProgramsBeforePreparation(computeProgram.Program, default);
        }

        ReleaseFragmentWrites();
        _host.EndRendering("dispatch");
        if (program.UsesDeviceAddresses)
        {
            _host.BeforeDeviceAddressPreparation([input.Stage]);
        }

        using (_host.BeginPreparation())
        {
            // The host may compile this program off the command-stream thread and report that
            // the pipeline is not ready yet. The dispatch waits for it rather than being
            // dropped: a guest that does not replay a one-shot dispatch deadlocks, because
            // the dispatch can feed a label a later packet waits on. Asking again keeps the
            // wait outside the pipeline cache's lock, so other queues can create their own
            // pipelines while this program compiles.
            PipelineHandle pipeline;
            if (ReferenceEquals(computeProgram, _keptComputeProgram))
            {
                pipeline = _keptComputePipeline;
            }
            else
            {
                using (RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.DispatchPipeline))
                {
                    if (suspendable)
                    {
                        // The packet suspends and runs again when the compile finishes, so the
                        // command stream can serve its other queues in the meantime. Nothing
                        // before this point has touched the host's recording, so leaving now is clean.
                        if (!_pipelines.TryCreateComputePipeline(input, computeProgram.Program, out pipeline))
                        {
                            record.Suspensions!.Keep(record.SubmitId, shader, record.PacketIndirectArgumentsAddress, record.Retry);
                            _host.NotePipelineNotReady();
                            RenderExecutor.TraceFrameDispatch(_host, "suspended", shader, program.Hash, record.GroupsX, record.GroupsY, record.GroupsZ,
                                record.IndirectArgumentsAddress);
                            return;
                        }
                    }
                    else
                    {
                        while (!_pipelines.TryCreateComputePipeline(input, computeProgram.Program, out pipeline))
                        {
                            using var waitScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ComputePipelineWait);
                            var completion = _pipelines.GetComputePipelineCompletion(computeProgram.Program);
                            if (completion is null)
                            {
                                Thread.Sleep(1);
                            }
                            else
                            {
                                completion.GetAwaiter().GetResult();
                            }
                        }
                    }
                }

                _keptComputeProgram = computeProgram;
                _keptComputePipeline = pipeline;
            }

            var bindings = PrepareBindings(input.Stage);
            if (program.UsesDeviceAddresses)
            {
                _host.PrepareDeviceAddresses();
            }

            _host.BindResources(bindings);
            Span<IPreparedBindings> stages = [bindings];
            _host.CommitBindings(GpuBindPoint.Compute, in pipeline, stages);
            var chain = _dispatchChain ??= new DispatchChain();
            var stage = input.Stage;
            // Every earlier read of the written resources completes before this dispatch writes,
            // unless the host deferred the last dispatches' barriers and none of them touched them.
            if (chain.NeedsHazardBarrier(record, in stage, _host.DeferredBarrierPromotions))
            {
                _host.ShaderWriteHazardBarrier();
            }

            var (groupsX, groupsY, groupsZ) = PhysicalGroups(input, record);
            RenderExecutor.TraceFrameDispatch(_host, record.HasStorageWrites ? "dispatched-writes" : "dispatched", shader, program.Hash,
                record.GroupsX, record.GroupsY, record.GroupsZ, record.IndirectArgumentsAddress);
            TraceRangeWriters(input.Stage, program.Hash, record.GroupsX, record.GroupsY, record.GroupsZ);
            var imageSnapshot = _host.BeginComputeImageSnapshot(bindings);
            _host.BeginCheckedDispatch();
            _host.BindPipeline(GpuBindPoint.Compute, in pipeline);
            var dispatched = false;
            if (RenderExecutor.SkipsComputeProgram(program.Hash))
            {
                // SHARPEMU_SKIP_CS: what the program contributes, by its absence.
            }
            else if (record.IndirectArgumentsAddress == 0 || !TryDispatchIndirect(input, record.IndirectArgumentsAddress))
            {
                _host.Dispatch(groupsX, groupsY, groupsZ);
                dispatched = true;
            }
            else
            {
                dispatched = true;
            }

            _host.EndCheckedDispatch();
            chain.Commit(_host.TryDeferShaderAccessBarrier(), _host.DeferredBarrierPromotions);
            _host.EndComputeImageSnapshot(imageSnapshot, bindings);
            if (dispatched && record.IndirectArgumentsAddress == 0 && computeProgram.KnownWrite is { } known)
            {
                _host.NoteKnownGpuContents(known.Address, known.Words);
                known.Release();
            }
        }

        _host.ResetBindings();
    }

    // The host consumers, in the order the dispatch is offered to them. disposition names the one
    // that took it, for the frame trace.
    private bool TryConsume(ResolvedDispatch record, ShaderProgramInfo program, out string? disposition)
    {
        disposition = null;
        if (record.MetadataClearCount != 0 &&
            TryClearMetadata(record.MetadataClear, program, program.Frontend is not null && program.StoresUniformValues))
        {
            disposition = "metadata-clear";
            return true;
        }

        if (record.ConstantFill is { } fill)
        {
            var consumed = _host.TryClearImageFromBuffer(fill.Address, fill.Size, fill.Value) ||
                           _host.TryAbsorbDccFill(fill.Address, fill.Size, fill.Value);
            if (RenderTrace.Enabled && RenderTrace.MetadataClear())
            {
                RenderTrace.Write($"Constant fill: shader=0x{program.Hash:X16} address=0x{fill.Address:X16} size=0x{fill.Size:X} value=0x{fill.Value:X8} consumed={consumed}");
            }

            if (consumed)
            {
                disposition = "constant-fill";
                return true;
            }
        }

        if (record.BoundedFill is { } bounded)
        {
            ReadOnlySpan<uint> pattern = bounded.Pattern;
            var consumed = (bounded.Uniform && _host.TryFillDccMetadata(bounded.Address, bounded.Size, bounded.Value)) ||
                           _host.TryClearImageFromKernelFill(bounded.Address, bounded.Size, pattern[..bounded.PatternLength]);
            if (RenderTrace.Enabled && RenderTrace.MetadataClear())
            {
                RenderTrace.Write($"Bounded fill: shader=0x{program.Hash:X16} address=0x{bounded.Address:X16} size=0x{bounded.Size:X} value=0x{bounded.Value:X8} consumed={consumed}");
            }

            if (consumed)
            {
                disposition = "bounded-fill";
                return true;
            }
        }

        if (record.BoundedCopy is { } copy)
        {
            var consumed = _host.TryCopyWordsOnHost(copy.Destination, copy.Source, copy.Modulus, copy.Words);
            if (RenderTrace.Enabled)
            {
                RenderTrace.Write(
                    $"Bounded copy: shader=0x{program.Hash:X16} src=0x{copy.Source:X16} dst=0x{copy.Destination:X16} words={copy.Words} period={copy.Modulus} consumed={consumed}");
            }

            if (consumed)
            {
                record.Program.KnownWrite?.Release();
                disposition = "bounded-copy";
                return true;
            }

            Diagnostics.DccWriterTrace.Refuse(program.Hash, $"copy refused by the host dst=0x{copy.Destination:X} words={copy.Words} modulus={copy.Modulus}");
        }

        if (record.ImageClear is { } clear)
        {
            var address = clear.Descriptor.Address;
            if (!_host.TryClearImageFromBuffer(address, clear.Size, clear.PackedClear))
            {
                // A metadata fill may run before its target is bound; the store keeps it pending. The fill
                // itself then runs on the host when the range is known DCC metadata: run as a dispatch it
                // leaves the metadata GPU-written, and every later bind of the surface waits for the GPU to
                // read it back (Astro Bot fills several surfaces this way each frame).
                var registered = _host.TryAbsorbDccFill(address, clear.Size, clear.PackedClear);
                var filledOnHost = !registered && _host.TryFillDccMetadata(address, clear.Size, clear.PackedClear);
                if (RenderTrace.Enabled && RenderTrace.MetadataClear())
                {
                    RenderTrace.Write(
                        $"{(registered ? "Tracked" : filledOnHost ? "Filled on the host" : "Deferred")} a metadata clear: shader=0x{program.Hash:X16} address=0x{address:X16} size=0x{clear.Size:X16} value=0x{clear.PackedClear:X8}");
                }

                if (registered || filledOnHost)
                {
                    disposition = "image-clear";
                    return true;
                }

                return false;
            }

            if (RenderTrace.Enabled && RenderTrace.ImageClear())
            {
                RenderTrace.Write($"Consumed a compute image clear: shader=0x{program.Hash:X16} address=0x{address:X16} size=0x{clear.Size:X16} value=0x{clear.PackedClear:X8}");
            }

            disposition = "image-clear";
            return true;
        }

        return false;
    }

    // A full overwrite of registered metadata by a compute shader becomes a tracked clear.
    // Metadata that is also read is not a proven overwrite (unless the program's reads only guard
    // its uniform stores), and every buffer it writes has to be a whole metadata surface: a store
    // anywhere else is not a clear, and neither is a small buffer that only starts where a
    // surface's metadata was registered.
    private bool TryClearMetadata(ReadOnlySpan<MetadataClearProbe> probes, ShaderProgramInfo program, bool readsOnlyGuardStores)
    {
        foreach (ref readonly var probe in probes)
        {
            if (_host.IsMetadata(probe.Address) && (!probe.Written || (probe.Read && !readsOnlyGuardStores)))
            {
                return false;
            }
        }

        foreach (ref readonly var probe in probes)
        {
            if (probe.Written && !_host.CoversMetadata(probe.Address, probe.Bytes))
            {
                return false;
            }
        }

        foreach (ref readonly var probe in probes)
        {
            if (!probe.Written || !_host.ClearMetadata(probe.Address))
            {
                continue;
            }

            if (_loggedMetadataClears.Add((program.Hash, probe.Address)) || RenderExecutor.TraceHtile)
            {
                Console.Error.WriteLine(
                    $"[GPU][INFO] A compute dispatch is taken as a metadata clear: hash=0x{program.Hash:X16} " +
                    $"metadata=0x{probe.Address:X16} records={probe.Records} stride={probe.Stride} " +
                    $"constant=0x{program.ConstantStoreValue ?? 0:X8}.");
            }

            return true;
        }

        return false;
    }

    // A dispatch that runs and stores into registered HTILE leaves it holding what it wrote,
    // so a cleared mark from before (a memset taken for a clear, or a surface's first
    // registration) no longer stands: UE5's Nanite depth export writes the material depth and
    // the HTILE that describes it, and the mark loaded that depth as zero, so every Nanite
    // material draw failed its depth-equal test (Black Myth: Wukong's tree trunks).
    private void NoteHtileStores(ComputeInputInfo input)
    {
        var program = input.Stage.Program!;
        var resources = input.Stage.Resources;
        if (resources.Buffers.Length != program.Buffers.Length)
        {
            return;
        }

        // A fill of one constant whose ZMask (bits 0-3) is zero stores clear codes, not data:
        // the memset before UE5's export, when it covers only part of the surface or the
        // surface is not registered yet. It unmakes no tracked clear, and a surface registered
        // over it later starts cleared whatever was stored there before.
        var clearCodes = program.ConstantStoreValue is { } fill && (fill & 0xF) == 0;
        for (var i = 0; i < program.Buffers.Length; i++)
        {
            if (!program.Buffers[i].Written)
            {
                continue;
            }

            var words = resources.Buffers[i];
            if (words.Length < 4)
            {
                throw _host.Fatal($"A buffer descriptor is too short: index={i} words={words.Length}.");
            }

            var descriptor = BufferDescriptorWords.From(words);
            if (clearCodes)
            {
                _host.NoteHtileCleared(descriptor.Address);
                continue;
            }

            var dropped = _host.NoteHtileWritten(descriptor.Address);
            if (RenderExecutor.TraceHtile && dropped)
            {
                Console.Error.WriteLine($"[GPU][HTILE] store hash=0x{program.Hash:X16} buffer=0x{descriptor.Address:X16} dropped_clear={dropped}");
            }

            if (dropped && _loggedHtileStores.Add((program.Hash, descriptor.Address)))
            {
                Console.Error.WriteLine(
                    $"[GPU][INFO] A compute dispatch stores into HTILE, so its depth surface no longer loads as a clear: " +
                    $"hash=0x{program.Hash:X16} htile=0x{descriptor.Address:X16} images={program.Images.Length}.");
            }
        }
    }

    // Writer tracing, shared by draws (RenderRecorder draws call TraceTargetWriters and
    // TraceBlendState) and dispatches.
    // SHARPEMU_TRACE_WRITERS=<lo>-<hi> (hex): every dispatch with a written buffer binding over the
    // range, and every dispatch that stores through device addresses (whose targets are not
    // known here), so the shader that writes a value can be named.
    private static readonly (ulong Low, ulong High) TracedWriterRange = ParseTracedWriterRange(
        Environment.GetEnvironmentVariable("SHARPEMU_TRACE_WRITERS"));

    private static (ulong, ulong) ParseTracedWriterRange(string? text)
    {
        var parts = (text ?? string.Empty).Split('-');
        return parts.Length == 2 &&
            ulong.TryParse(parts[0].Replace("0x", string.Empty), System.Globalization.NumberStyles.HexNumber, null, out var low) &&
            ulong.TryParse(parts[1].Replace("0x", string.Empty), System.Globalization.NumberStyles.HexNumber, null, out var high)
            ? (low, high)
            : (0, 0);
    }

    private int _tracedWriters;

    // SHARPEMU_TRACE_WRITERS_LIMIT=<n>: how many lines the trace writes (default 4000).
    private static readonly int TracedWriterLimit =
        int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_TRACE_WRITERS_LIMIT"), out var limit) && limit > 0 ? limit : 4000;
    private readonly HashSet<ulong> _tracedDeviceWriters = [];

    // The draws whose colour or depth targets start inside the traced range (SHARPEMU_TRACE_WRITERS).
    private void TraceTargetWriters(ref DrawState state, ulong pixelHash, ShaderStageResources vertexStage)
    {
        TraceStencilState(ref state, pixelHash);
        if (TracedWriterRange.High == 0 || _tracedWriters > TracedWriterLimit)
            return;

        var hits = new System.Text.StringBuilder();
        foreach (ref readonly var color in DrawState.BoundColors(ref state))
        {
            var address = color.Resolution.BaseAddress;
            if (address < TracedWriterRange.High && TracedWriterRange.Low <= address)
                hits.Append($" color{color.Slot}=0x{address:X}");
        }

        if (state.Depth.HasTarget)
        {
            var address = state.Depth.Target.Target.Request.Description.Data.Address;
            if (address < TracedWriterRange.High && TracedWriterRange.Low <= address)
                hits.Append($" depth=0x{address:X}");
        }

        // The buffers its vertex program reads from inside the range: a list a dispatch built.
        AppendTracedBuffers(hits, vertexStage, written: false, prefix: "vsbuf");
        if (hits.Length == 0)
            return;

        _tracedWriters++;
        Console.Error.WriteLine($"[GPU][WRITERS] draw vs=0x{vertexStage.Program?.Hash ?? 0:X16} ps=0x{pixelHash:X16}{hits}");
    }

    // SHARPEMU_TRACE_STENCIL=1: each distinct depth/stencil state a pixel program draws with, once, with its depth target.
    private static readonly bool TraceStencil = Environment.GetEnvironmentVariable("SHARPEMU_TRACE_STENCIL") == "1";
    private readonly HashSet<(ulong, DepthStencilState)> _tracedStencil = [];

    private void TraceStencilState(ref DrawState state, ulong pixelHash)
    {
        if (!TraceStencil || !state.Depth.HasTarget || _tracedStencil.Count > 2000)
            return;
        var depth = state.Depth.Target.State;
        if (!_tracedStencil.Add((pixelHash, depth)))
            return;
        Console.Error.WriteLine(
            $"[GPU][STENCIL] ps=0x{pixelHash:X16} depth=0x{state.Depth.Target.Target.Request.Description.Data.Address:X} " +
            $"test={depth.DepthTestEnabled} write={depth.DepthWriteEnabled} cmp={depth.DepthCompare} stencil={depth.StencilTestEnabled} " +
            $"clear={depth.StencilClearEnabled}:{depth.StencilClearValue} front={depth.FrontOperations} {depth.FrontMasks} back={depth.BackOperations} {depth.BackMasks}");
    }

    // Under SHARPEMU_TRACE_STENCIL, each distinct colour blend state a pixel program draws with, once.
    private readonly HashSet<(ulong, string)> _tracedBlend = [];

    private void TraceBlendState(ref DrawState state, ContextRegisters context, ulong pixelHash)
    {
        if (!TraceStencil || _tracedBlend.Count > 2000)
            return;
        var text = new System.Text.StringBuilder();
        foreach (ref readonly var color in DrawState.BoundColors(ref state))
        {
            var blend = context.BlendControls[color.Slot];
            text.Append($" slot{color.Slot}@0x{color.Resolution.BaseAddress:X}:mask=0x{context.ColorWriteMaskForSlot(color.Slot):X}" +
                (blend.Enable ? $",blend={blend.ColorSourceFactor}/{blend.ColorFunction}/{blend.ColorDestinationFactor}" +
                    (blend.SeparateAlpha ? $"|{blend.AlphaSourceFactor}/{blend.AlphaFunction}/{blend.AlphaDestinationFactor}" : string.Empty)
                    : ",noblend"));
        }

        var line = text.ToString();
        if (_tracedBlend.Add((pixelHash, line)))
            Console.Error.WriteLine($"[GPU][BLEND] ps=0x{pixelHash:X16}{line}");
    }

    private void TraceRangeWriters(ShaderStageResources stage, ulong hash, uint groupsX, uint groupsY, uint groupsZ)
    {
        if (TracedWriterRange.High == 0 || stage.Program is not { } program || _tracedWriters > TracedWriterLimit)
            return;

        var hits = new System.Text.StringBuilder();
        AppendTracedBuffers(hits, stage, written: true, prefix: "buf");
        // And the ones it only reads, which name the consumers of what an earlier dispatch wrote.
        AppendTracedBuffers(hits, stage, written: false, prefix: "rbuf");

        // Storage images it writes, by the base their T# names (bits 39:8 in dword 0, 47:40 in dword 1).
        var images = stage.Resources.Images;
        for (var i = 0; i < program.Images.Length && i < images.Length; i++)
        {
            if (!program.Images[i].Written || images[i].Length < 2)
                continue;
            var address = (((ulong)(images[i][1] & 0xFFu) << 32) | images[i][0]) << 8;
            if (address != 0 && address < TracedWriterRange.High && TracedWriterRange.Low <= address)
                hits.Append($" img{i}=0x{address:X}");
        }

        // A dispatch that only stores through device addresses is named once, not every time.
        if (hits.Length == 0 && (!program.WritesDeviceAddresses || !_tracedDeviceWriters.Add(hash)))
            return;

        _tracedWriters++;
        Console.Error.WriteLine(
            $"[GPU][WRITERS] hash=0x{hash:X16} groups={groupsX}x{groupsY}x{groupsZ} device_stores={program.WritesDeviceAddresses}{hits}");
    }

    // The stage's buffers over the traced range that it writes (or only reads).
    private static void AppendTracedBuffers(System.Text.StringBuilder hits, ShaderStageResources stage, bool written, string prefix)
    {
        if (stage.Program is not { } program)
            return;
        var descriptors = stage.Resources.Buffers;
        for (var i = 0; i < program.Buffers.Length && i < descriptors.Length; i++)
        {
            if (program.Buffers[i].Written != written || descriptors[i].Length < 4)
                continue;
            var descriptor = BufferDescriptorWords.From(descriptors[i]);
            if (descriptor.Footprint() is not { } footprint || descriptor.Address == 0)
                continue;
            if (descriptor.Address < TracedWriterRange.High && TracedWriterRange.Low < descriptor.Address + footprint)
                hits.Append($" {prefix}{i}=0x{descriptor.Address:X}+0x{footprint:X}");
        }
    }

    // The indirect form hands the host the group axis order so it can permute the counts it reads.
    private bool TryDispatchIndirect(ComputeInputInfo input, ulong argumentsAddress)
    {
        Span<int> axisOrder = stackalloc int[3];
        GroupAxisOrder(input, axisOrder);
        return _host.TryDispatchIndirect(argumentsAddress, axisOrder);
    }

    // The shader's local workgroup axes may have been remapped at compile time (see
    // Gen5WorkgroupAxisOrder.Compute) so the largest NUM_THREAD axis lands on a physical axis the
    // host allows it on, and a host whose programs run a Y-only one-dimensional program's groups
    // on X (IRenderHost.RunsYOnlyGroupsOnX) swaps those two. Group counts follow the same order.
    private void GroupAxisOrder(ComputeInputInfo input, Span<int> axisOrder)
    {
        if (_host.RunsYOnlyGroupsOnX)
        {
            Gen5WorkgroupAxisOrder.ComputeGroupsInto(input.ThreadsX, input.ThreadsY, input.ThreadsZ,
                input.GroupIdX, input.GroupIdY, input.GroupIdZ, axisOrder);
        }
        else
        {
            Gen5WorkgroupAxisOrder.ComputeInto(input.ThreadsX, input.ThreadsY, input.ThreadsZ, axisOrder);
        }
    }

    private (uint X, uint Y, uint Z) PhysicalGroups(ComputeInputInfo input, ResolvedDispatch record)
    {
        Span<int> axisOrder = stackalloc int[3];
        GroupAxisOrder(input, axisOrder);
        Span<uint> logical = [record.GroupsX, record.GroupsY, record.GroupsZ];
        Span<uint> physical = stackalloc uint[3];
        for (var axis = 0; axis < 3; axis++)
        {
            physical[axisOrder[axis]] = logical[axis];
        }

        return (physical[0], physical[1], physical[2]);
    }
}
