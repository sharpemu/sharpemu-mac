// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.Libs.VideoOut;
using SharpEmu.ShaderCompiler;
using PortableBufferWord = SharpEmu.ShaderCompiler.Resources.PortableBufferWord;
using ResourceSnapshot = SharpEmu.ShaderCompiler.Resources.ResourceSnapshot;

namespace SharpEmu.Libs.Gpu.Rendering;

public readonly record struct ComputeImageClear(BufferDescriptorWords Descriptor, uint PackedClear, ulong Size);

public sealed partial class RenderExecutor
{
    private const uint DispatchInitiatorUseThreadDimensions = 1u << 5;
    private const uint DispatchInitiatorBaseBits = 0x41;
    private const uint DispatchInitiatorModifierBits = 0xA038;
    private const uint DispatchInitiatorKnownMask = DispatchInitiatorBaseBits | DispatchInitiatorModifierBits;
    private const uint ImageClearDispatchInitiator = 0x61;
    private const uint ImageClearWaveSize = 64;
    private const uint Format32UInt = 20;
    private const uint ImageClearStride = 16;
    private const uint ImageClearUserDataCount = 8;

    // The pipeline of a dispatch the stream has not reached, asked for while an earlier dispatch
    // waits on its own compile (GpuCommandInterpreter.PrefetchPipelinesAhead): the program is
    // prepared as the dispatch will prepare it, and a pipeline the cache does not hold starts
    // compiling beside the one waited for. Nothing is recorded. Pipelines take from the counted
    // ones the dispatch's own request would.
    public void PrefetchDispatchPipeline(RegisterBanks banks, uint groupsX, uint groupsY, uint groupsZ, uint dispatchInitiator)
    {
        var compute = banks.Shader.Compute;
        if (!_host.SupportsPipelineSuspend || compute.Address == 0 || groupsX == 0 || groupsY == 0 || groupsZ == 0)
        {
            return;
        }

        var computeProgram = _pipelines.PrepareComputeProgramAhead(compute, dispatchInitiator, groupsX, groupsY, groupsZ);
        if (!computeProgram.Available || computeProgram.Consumed || !computeProgram.Program.IsValid ||
            computeProgram.Input.Stage.Program is null)
        {
            return;
        }

        _ = _pipelines.TryCreateComputePipeline(computeProgram.Input, computeProgram.Program, out _);
    }

    public void Dispatch(ulong submitId, RegisterBanks banks, uint groupsX, uint groupsY, uint groupsZ, uint dispatchInitiator, ulong indirectArgumentsAddress = 0)
    {
        if (!_host.IsRecording)
        {
            throw _host.Fatal("A dispatch has no recording command buffer.");
        }

        _host.RunPendingOperations();
        var compute = banks.Shader.Compute;
        _host.SetDebugInformation(RecordedOperation.DispatchDirect, submitId, groupsX, groupsY, groupsZ, dispatchInitiator, compute.Address);
        if (compute.Address == 0)
        {
            if (RenderTrace.Enabled && RenderTrace.NullComputeShader())
            {
                RenderTrace.Write($"Ignoring a dispatch with no compute shader: groups={groupsX}x{groupsY}x{groupsZ} initiator=0x{dispatchInitiator:X8}");
            }

            return;
        }

        var unknownBits = dispatchInitiator & ~DispatchInitiatorKnownMask;
        if (unknownBits != 0 && RenderTrace.Enabled && RenderTrace.UnknownInitiator())
        {
            RenderTrace.Write(
                $"The dispatch initiator has unknown bits: initiator=0x{dispatchInitiator:X8} unknown=0x{unknownBits:X8} shader=0x{compute.Address:X16} groups={groupsX}x{groupsY}x{groupsZ}");
        }

        var useThreadDimensions = (dispatchInitiator & DispatchInitiatorUseThreadDimensions) != 0;
        if ((indirectArgumentsAddress == 0 || useThreadDimensions) && (groupsX == 0 || groupsY == 0 || groupsZ == 0))
        {
            if (RenderTrace.Enabled && RenderTrace.ZeroDispatch())
            {
                RenderTrace.Write($"Skipping a zero-sized dispatch: groups={groupsX}x{groupsY}x{groupsZ} initiator=0x{dispatchInitiator:X8} shader=0x{compute.Address:X16}");
            }

            TraceFrameDispatch("zero-sized", compute.Address, 0, groupsX, groupsY, groupsZ, indirectArgumentsAddress);
            return;
        }

        // A dispatch that suspended on its pipeline runs again from its packet; what it prepared
        // then is taken back while it still holds (ComputeProgramRetry), so the retry neither
        // prepares the program again nor re-runs the consumers below, which declined it then.
        var retry = _host.SupportsPipelineSuspend
            ? TakeSuspendedDispatch(submitId, compute.Address, indirectArgumentsAddress)
            : null;
        var computeProgram = retry is null
            ? _pipelines.GetComputeProgram(compute, banks.Context.ShaderInterface, dispatchInitiator, groupsX, groupsY, groupsZ)
            : _pipelines.GetComputeProgram(compute, banks.Context.ShaderInterface, dispatchInitiator, groupsX, groupsY, groupsZ, retry);
        var retried = retry?.Reused == true;
        if (retried)
        {
            _computeRetriesReused++;
        }

        if (computeProgram.Consumed)
        {
            TraceFrameDispatch("consumed", compute.Address, 0, groupsX, groupsY, groupsZ, indirectArgumentsAddress);
            return;
        }

        if (!computeProgram.Available)
        {
            if (RenderTrace.Enabled)
            {
                RenderTrace.Write($"Skipping a dispatch without a program: shader=0x{compute.Address:X16} groups={groupsX}x{groupsY}x{groupsZ}");
            }

            TraceFrameDispatch("no-program", compute.Address, 0, groupsX, groupsY, groupsZ, indirectArgumentsAddress);
            return;
        }

        var input = computeProgram.Input;
        var program = input.Stage.Program ?? throw _host.Fatal($"The compute program is missing: shader=0x{compute.Address:X16}.");
        if (Diagnostics.DccWriterTrace.Enabled)
        {
            Diagnostics.DccWriterTrace.CurrentProgram = program.Hash;
            Diagnostics.DccWriterTrace.CurrentKind = "compute";
        }

        if (RenderTrace.Enabled)
        {
            RenderTrace.Write(
                $"Dispatch seq={RenderTrace.NextSequence()} submit={submitId} shader=0x{compute.Address:X16} hash=0x{program.Hash:X16} " +
                $"groups={groupsX}x{groupsY}x{groupsZ} local={input.ThreadsX}x{input.ThreadsY}x{input.ThreadsZ} wave={input.WaveSize} " +
                $"localDataShareDwords={input.LocalDataShareDwords} barriers={input.NeedsLocalDataShareBarriers} initiator=0x{dispatchInitiator:X8} " +
                $"buffers={program.Buffers.Length} images={program.Images.Length} samplers={program.SamplerCount}");
        }

        if (!retried && indirectArgumentsAddress == 0 &&
            (ulong)groupsX * groupsY * groupsZ >= MetadataClearMinimumGroups &&
            TryConsumeMetadataClear(input))
        {
            TraceFrameDispatch("metadata-clear", compute.Address, program.Hash, groupsX, groupsY, groupsZ, indirectArgumentsAddress);
            _host.ResetBindings();
            return;
        }

        if (!retried && indirectArgumentsAddress == 0 && TryConsumeConstantFill(input, groupsX, groupsY, groupsZ))
        {
            TraceFrameDispatch("constant-fill", compute.Address, program.Hash, groupsX, groupsY, groupsZ, indirectArgumentsAddress);
            _host.ResetBindings();
            return;
        }

        if (!retried && indirectArgumentsAddress == 0 && TryConsumeBoundedFill(input, groupsX, groupsY, groupsZ, dispatchInitiator))
        {
            TraceFrameDispatch("bounded-fill", compute.Address, program.Hash, groupsX, groupsY, groupsZ, indirectArgumentsAddress);
            _host.ResetBindings();
            return;
        }

        if (!retried && indirectArgumentsAddress == 0 && TryConsumeBoundedCopy(input, groupsX, groupsY, groupsZ, dispatchInitiator))
        {
            TraceFrameDispatch("bounded-copy", compute.Address, program.Hash, groupsX, groupsY, groupsZ, indirectArgumentsAddress);
            _host.ResetBindings();
            return;
        }

        if (!retried && indirectArgumentsAddress == 0 && TryConsumeImageClear(input, groupsX, groupsY, groupsZ, dispatchInitiator))
        {
            TraceFrameDispatch("image-clear", compute.Address, program.Hash, groupsX, groupsY, groupsZ, indirectArgumentsAddress);
            _host.ResetBindings();
            return;
        }

        NoteHtileStores(input);

        if (useThreadDimensions)
        {
            // The indirect buffer carries thread counts in this mode, while Vulkan indirect
            // dispatch consumes workgroup counts. Use the CPU-resolved counts after conversion.
            indirectArgumentsAddress = 0;
            var threadsX = groupsX;
            var threadsY = groupsY;
            var threadsZ = groupsZ;
            groupsX = GroupsFromThreads(threadsX, compute.ThreadsX);
            groupsY = GroupsFromThreads(threadsY, compute.ThreadsY);
            groupsZ = GroupsFromThreads(threadsZ, compute.ThreadsZ);
            if (RenderTrace.Enabled && RenderTrace.ThreadDimensionConversion())
            {
                RenderTrace.Write(
                    $"Converted thread dimensions to groups: threads={threadsX}x{threadsY}x{threadsZ} " +
                    $"local={Math.Max(compute.ThreadsX, 1)}x{Math.Max(compute.ThreadsY, 1)}x{Math.Max(compute.ThreadsZ, 1)} groups={groupsX}x{groupsY}x{groupsZ}");
            }
        }

        if (SkipWhileLinkedModuleCompiles(input, computeProgram.Program, program))
        {
            _host.ResetBindings();
            return;
        }

        if (!_host.SupportsPipelineSuspend)
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
            if (_host.SupportsPipelineSuspend)
            {
                // The packet suspends and runs again when the compile finishes, so the
                // command stream can serve its other queues in the meantime. Nothing
                // before this point has touched the host, so leaving now is clean.
                if (!_pipelines.TryCreateComputePipeline(input, computeProgram.Program, out pipeline))
                {
                    KeepSuspendedDispatch(submitId, compute.Address, indirectArgumentsAddress, retry);
                    _host.NotePipelineNotReady();
                    TraceFrameDispatch("suspended", compute.Address, program.Hash, groupsX, groupsY, groupsZ, indirectArgumentsAddress);
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

            var bindings = PrepareBindings(input.Stage);
            if (program.UsesDeviceAddresses)
            {
                _host.PrepareDeviceAddresses();
            }

            _host.BindResources(bindings);
            Span<IPreparedBindings> stages = [bindings];
            _host.CommitBindings(GpuBindPoint.Compute, in pipeline, stages);
            var hasStorageWrites = HasBufferWrites(input.Stage);
            foreach (var image in program.Images)
            {
                hasStorageWrites |= image.Written && image.Class == ImageResourceClass.Storage;
            }

            if (hasStorageWrites)
            {
                // Every earlier read of the written resources completes before this dispatch writes.
                _host.ShaderWriteHazardBarrier();
            }

            _host.BindPipeline(GpuBindPoint.Compute, in pipeline);
            // The shader's local workgroup axes may have been remapped at compile time
            // (see Gen5WorkgroupAxisOrder.Compute) so the largest NUM_THREAD
            // axis lands on a physical axis Vulkan actually allows it on. The dispatch group
            // counts must be permuted the same way, or vkCmdDispatch would hand group counts
            // for the wrong physical axis to the pipeline it built with the remapped sizes.
            Span<int> axisOrder = stackalloc int[3];
            Gen5WorkgroupAxisOrder.ComputeInto(input.ThreadsX, input.ThreadsY, input.ThreadsZ, axisOrder);
            Span<uint> logicalGroups = [groupsX, groupsY, groupsZ];
            Span<uint> physicalGroups = stackalloc uint[3];
            for (var logical = 0; logical < 3; logical++)
            {
                physicalGroups[axisOrder[logical]] = logicalGroups[logical];
            }

            TraceFrameDispatch(hasStorageWrites ? "dispatched-writes" : "dispatched", compute.Address, program.Hash, groupsX, groupsY, groupsZ, indirectArgumentsAddress);
            TraceRangeWriters(input.Stage, program.Hash, groupsX, groupsY, groupsZ);
            var imageSnapshot = _host.BeginComputeImageSnapshot(bindings);
            if (SkippedComputePrograms.Contains(program.Hash) || LiveSkips.Contains(program.Hash))
            {
                // SHARPEMU_SKIP_CS: what the program contributes, by its absence.
            }
            else if (indirectArgumentsAddress == 0 || !_host.TryDispatchIndirect(indirectArgumentsAddress))
            {
                _host.Dispatch(physicalGroups[0], physicalGroups[1], physicalGroups[2]);
            }
            _host.EndComputeImageSnapshot(imageSnapshot, bindings);
            _host.ShaderAccessBarrier();
        }

        _host.ResetBindings();
    }

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
        foreach (ref readonly var color in BoundColors(ref state))
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
        foreach (ref readonly var color in BoundColors(ref state))
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

    // Under a dump-triggered frame, what became of each dispatch: a pass that never ran leaves
    // no snapshot behind, so the list of dispositions is the only record of it.
    private void TraceFrameDispatch(string disposition, ulong shader, ulong hash, uint groupsX, uint groupsY, uint groupsZ, ulong indirect)
    {
        if (_host.TraceFrameActive)
        {
            Console.Error.WriteLine(
                $"[FRAME-DISPATCH] {disposition} hash=0x{hash:X16} shader=0x{shader:X} groups={groupsX}x{groupsY}x{groupsZ} indirect=0x{indirect:X}");
        }
    }

    // The dispatches suspended on their pipeline, by submission: a submission runs its packets
    // in order, so the next dispatch it reaches is the one that suspended. The address and the
    // indirect arguments make sure of it; the retry itself checks everything else it read.
    private readonly Dictionary<ulong, (ulong Address, ulong IndirectArguments, ComputeProgramRetry Retry)> _suspendedDispatches = [];
    private ComputeProgramRetry? _spareRetry;
    private long _computeRetriesReused;

    /// <summary>Suspended dispatches that ran again on the program they prepared before, for tests.</summary>
    internal long ComputeRetriesReused => _computeRetriesReused;

    // The retry of this dispatch, or an empty one that logs what the preparation reads. The
    // empty one is reused by every dispatch that does not suspend; one that does keeps it.
    private ComputeProgramRetry TakeSuspendedDispatch(ulong submitId, ulong address, ulong indirectArguments)
    {
        if (_suspendedDispatches.Remove(submitId, out var suspended) &&
            suspended.Address == address && suspended.IndirectArguments == indirectArguments)
        {
            return suspended.Retry;
        }

        var retry = _spareRetry ??= new ComputeProgramRetry();
        retry.Reset();
        return retry;
    }

    private void KeepSuspendedDispatch(ulong submitId, ulong address, ulong indirectArguments, ComputeProgramRetry? retry)
    {
        if (retry is null)
        {
            return;
        }

        if (_suspendedDispatches.Count >= 64)
        {
            // Submissions that never came back (a queue torn down mid-wait) are not kept for ever.
            _suspendedDispatches.Clear();
        }

        _suspendedDispatches[submitId] = (address, indirectArguments, retry);
        if (ReferenceEquals(retry, _spareRetry))
        {
            _spareRetry = null;
        }
    }

    // A program linked with the callees of its hit-group tables can take the driver minutes to compile, and the game
    // adds callees as it streams. Such programs are ray-traced passes that run every frame, so their dispatches are left
    // out while the module compiles instead of holding every queue behind it: the pass's output is stale for those
    // frames.
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<ulong, long> LinkedCompileSkips = new();

    private bool SkipWhileLinkedModuleCompiles(ComputeInputInfo input, ShaderProgram handle, ShaderProgramInfo program)
    {
        if (program.Bindings is not { CallTargetCount: not 0 } ||
            _pipelines.TryCreateComputePipeline(input, handle, out _)) return false;
        var skipped = LinkedCompileSkips.AddOrUpdate(program.Hash, 1, static (_, count) => count + 1);
        if ((skipped & (skipped - 1)) == 0)
            Console.Error.WriteLine($"[GPU][WARN][LINKED_COMPILE] Dispatches of a linked program are skipped while its module compiles: hash=0x{program.Hash:X16} skipped={skipped}.");
        return true;
    }

    // The dispatch counts threads; the host counts groups of the shader's thread size.
    public static uint GroupsFromThreads(uint threads, uint groupSize)
    {
        if (threads == 0)
        {
            return 0;
        }

        var size = Math.Max(groupSize, 1u);
        return (threads + size - 1) / size;
    }

    private BufferDescriptorWords DecodeBufferDescriptor(ResourceSnapshot resources, int index)
    {
        var words = resources.Buffers[index];
        if (words.Length < 4)
        {
            throw _host.Fatal($"A buffer descriptor is too short: index={index} words={words.Length}.");
        }

        return BufferDescriptorWords.From(words);
    }

    // A full overwrite of registered metadata by a compute shader becomes a tracked clear.
    // A clear of a metadata surface is a spread of workgroups over it; a one-group kernel
    // is a counter reset or an argument write whose buffer only happens to start where a
    // transient surface's HTILE was registered, and taken for a clear it never ran.
    private const ulong MetadataClearMinimumGroups = 2;

    // Each program and surface it was taken to clear, reported once.
    private readonly HashSet<(ulong Hash, ulong Address)> _loggedMetadataClears = new();

    // SHARPEMU_TRACE_HTILE=1 reports every metadata clear and HTILE store, not the first of each.
    public static readonly bool TraceHtile = Environment.GetEnvironmentVariable("SHARPEMU_TRACE_HTILE") == "1";

    private bool TryConsumeMetadataClear(ComputeInputInfo input)
    {
        var program = input.Stage.Program!;
        var resources = input.Stage.Resources;
        if (resources.Buffers.Length != program.Buffers.Length)
        {
            throw _host.Fatal($"The compute buffer count does not match the program: descriptors={resources.Buffers.Length} program={program.Buffers.Length}.");
        }

        for (var i = 0; i < program.Buffers.Length; i++)
        {
            var resource = program.Buffers[i];
            var descriptor = DecodeBufferDescriptor(resources, i);
            // Metadata that is also read is not a proven overwrite; the dispatch runs as written.
            if (_host.IsMetadata(descriptor.Address) && (!resource.Written || resource.Read))
            {
                return false;
            }
        }

        // A metadata clear is buffer stores and nothing else. A dispatch that also reads or
        // stores images, or writes through device addresses, does real work beside the
        // metadata - UE5's Nanite depth export writes the depth surface and the HTILE that
        // describes it in one dispatch - and dropping it loses that work and leaves the
        // surface marked cleared, so the next pass clears it again (Black Myth: Wukong's
        // characters and army failed every depth-equal base pass draw).
        if (program.HasBitwiseExclusiveOr || program.Images.Length != 0 || program.SamplerCount != 0 ||
            program.WritesDeviceAddresses)
        {
            return false;
        }

        // Every buffer it writes has to be a whole metadata surface: a store anywhere else is
        // not a clear, and neither is a small buffer that only starts where a surface's
        // metadata was registered.
        for (var i = 0; i < program.Buffers.Length; i++)
        {
            if (!program.Buffers[i].Written)
            {
                continue;
            }

            var written = DecodeBufferDescriptor(resources, i);
            var bytes = written.Stride == 0 ? written.RecordCount : (ulong)written.RecordCount * written.Stride;
            if (!_host.CoversMetadata(written.Address, bytes))
            {
                return false;
            }
        }

        for (var i = 0; i < program.Buffers.Length; i++)
        {
            if (!program.Buffers[i].Written)
            {
                continue;
            }

            var descriptor = DecodeBufferDescriptor(resources, i);
            if (_host.ClearMetadata(descriptor.Address))
            {
                if (_loggedMetadataClears.Add((program.Hash, descriptor.Address)) || TraceHtile)
                {
                    Console.Error.WriteLine(
                        $"[GPU][INFO] A compute dispatch is taken as a metadata clear: hash=0x{program.Hash:X16} " +
                        $"metadata=0x{descriptor.Address:X16} records={descriptor.RecordCount} stride={descriptor.Stride} " +
                        $"constant=0x{program.ConstantStoreValue ?? 0:X8}.");
                }

                return true;
            }
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

            var descriptor = DecodeBufferDescriptor(resources, i);
            if (clearCodes)
            {
                _host.NoteHtileCleared(descriptor.Address);
                continue;
            }

            var dropped = _host.NoteHtileWritten(descriptor.Address);
            if (TraceHtile && dropped)
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

    // Each program and HTILE surface reported by the above, once.
    private readonly HashSet<(ulong Hash, ulong Address)> _loggedHtileStores = new();

    // Recognizes a dispatch that fills one formatted buffer with a single value over every record.
    public ComputeImageClear? TryDecodeImageClear(ComputeInputInfo input, uint groupsX, uint groupsY, uint groupsZ, uint dispatchInitiator) =>
        TryDecodeUserDataFill(input, groupsX, groupsY, groupsZ, dispatchInitiator) ??
        TryDecodeLoadedValueFill(input, groupsX, groupsY, groupsZ) ??
        TryDecodeConstantFill(input, groupsX, groupsY, groupsZ);

    // AGC's constant fill kernel: one formatted dword buffer written at every thread index with a
    // value the program moves from a constant (a DCC or CMask clear code).
    private ComputeImageClear? TryDecodeConstantFill(ComputeInputInfo input, uint groupsX, uint groupsY, uint groupsZ)
    {
        var program = input.Stage.Program ?? throw _host.Fatal("The compute stage has no program.");
        var resources = input.Stage.Resources;
        if (program.ConstantStoreValue is not { } value ||
            program.Buffers.Length != 1 || resources.Buffers.Length != 1 || resources.Buffers[0].Length != 4 ||
            program.Images.Length != 0 || program.SamplerCount != 0 || program.UsesDeviceAddresses)
        {
            return null;
        }

        var info = program.Buffers[0];
        var descriptor = BufferDescriptorWords.From(resources.Buffers[0]);
        var threads = (ulong)groupsX * input.ThreadsX;
        if (!info.Formatted || !info.Written || info.Read || info.Atomic || info.Scalar || info.MaxByteExtent != sizeof(uint) ||
            descriptor.Stride != sizeof(uint) || descriptor.Format != BufferDescriptorWords.Format32UInt || descriptor.SwizzleEnabled ||
            descriptor.IndexStride != 0 || descriptor.AddThreadId || descriptor.RecordCount == 0 ||
            input.ThreadsX != ImageClearWaveSize || input.ThreadsY != 1 || input.ThreadsZ != 1 || input.WaveSize != ImageClearWaveSize ||
            groupsX == 0 || groupsY != 1 || groupsZ != 1 || threads != descriptor.RecordCount ||
            !input.GroupIdX || input.GroupIdY || input.GroupIdZ || input.ThreadIdCount != 1 ||
            (input.DispatchThreadDimensions && input.DispatchThreadsX != threads))
        {
            return null;
        }

        var size = descriptor.Footprint() ?? throw _host.Fatal($"The compute buffer footprint overflows: stride={descriptor.Stride} records={descriptor.RecordCount}.");
        return new ComputeImageClear(descriptor, value, size);
    }

    // The fill shader AGC titles use for DCC and CMask clears: each thread stores one dword it loaded
    // with s_buffer_load_dword from a one-record constant buffer, at its global thread index.
    private ComputeImageClear? TryDecodeLoadedValueFill(ComputeInputInfo input, uint groupsX, uint groupsY, uint groupsZ)
    {
        var program = input.Stage.Program ?? throw _host.Fatal("The compute stage has no program.");
        var resources = input.Stage.Resources;
        if (program.Buffers.Length != 2 || resources.Buffers.Length != 2 || program.Images.Length != 0 || program.SamplerCount != 0 ||
            program.UsesDeviceAddresses || resources.Images.Length != 0 || resources.Samplers.Length != 0 ||
            resources.Buffers[0].Length != 4 || resources.Buffers[1].Length != 4)
        {
            return null;
        }

        int source = -1, target = -1;
        for (var index = 0; index < 2; index++)
        {
            var info = program.Buffers[index];
            if (info.Scalar && info.Read && !info.Written && !info.Atomic && info.MaxByteExtent == sizeof(uint))
            {
                source = index;
            }
            else if (info.Formatted && info.Written && !info.Read && !info.Atomic && !info.Scalar && info.MaxByteExtent == sizeof(uint))
            {
                target = index;
            }
        }

        if (source < 0 || target < 0)
        {
            return null;
        }

        var descriptor = BufferDescriptorWords.From(resources.Buffers[target]);
        var valueDescriptor = BufferDescriptorWords.From(resources.Buffers[source]);
        var threads = (ulong)groupsX * input.ThreadsX;
        if (descriptor.Stride != sizeof(uint) || descriptor.Format != BufferDescriptorWords.Format32UInt || descriptor.SwizzleEnabled ||
            descriptor.IndexStride != 0 || descriptor.AddThreadId || descriptor.RecordCount == 0 ||
            input.ThreadsX != ImageClearWaveSize || input.ThreadsY != 1 || input.ThreadsZ != 1 || input.WaveSize != ImageClearWaveSize ||
            groupsX == 0 || groupsY != 1 || groupsZ != 1 || threads != descriptor.RecordCount ||
            !input.GroupIdX || input.GroupIdY || input.GroupIdZ || input.ThreadIdCount != 1 ||
            (input.DispatchThreadDimensions && input.DispatchThreadsX != threads) ||
            valueDescriptor.Address == 0)
        {
            return null;
        }

        Span<byte> value = stackalloc byte[sizeof(uint)];
        if (!_host.TryReadGuest(valueDescriptor.Address, value))
        {
            return null;
        }

        var size = descriptor.Footprint() ?? throw _host.Fatal($"The compute buffer footprint overflows: stride={descriptor.Stride} records={descriptor.RecordCount}.");
        return new ComputeImageClear(descriptor, System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(value), size);
    }

    // A portable program (PortableBufferWord) keeps only the stride's flags; it reads the stride.
    private static bool SameStride(uint specialized, uint descriptor) =>
        specialized == (PortableBufferWord.Enabled ? descriptor & ~PortableBufferWord.StrideMask : descriptor);

    private ComputeImageClear? TryDecodeUserDataFill(ComputeInputInfo input, uint groupsX, uint groupsY, uint groupsZ, uint dispatchInitiator)
    {
        var program = input.Stage.Program ?? throw _host.Fatal("The compute stage has no program.");
        var resources = input.Stage.Resources;
        if (program.Buffers.Length != 1 || resources.Buffers.Length != 1 || program.Images.Length != 0 || program.SamplerCount != 0 ||
            program.UsesDeviceAddresses || resources.Images.Length != 0 || resources.Samplers.Length != 0)
        {
            return null;
        }

        var resource = program.Buffers[0];
        var words = resources.Buffers[0];
        if (words.Length != 4)
        {
            return null;
        }

        var descriptor = BufferDescriptorWords.From(words);
        if (!resource.Formatted || !resource.Written || resource.Read || resource.Atomic || resource.Scalar || resource.MaxByteExtent != ImageClearStride ||
            descriptor.Stride != ImageClearStride || descriptor.Format != BufferDescriptorWords.Format32x4UInt || descriptor.SwizzleEnabled ||
            descriptor.IndexStride != 0 || descriptor.AddThreadId || !SameStride(resource.PackedStride, descriptor.PackedStride) ||
            program.UserDataBase != 0 || resources.UserData.Length != ImageClearUserDataCount)
        {
            return null;
        }

        for (var i = 0; i < words.Length; i++)
        {
            if (words[i] != resources.UserData[i])
            {
                return null;
            }
        }

        var clear = resources.UserData[4];
        if (resources.UserData[5] != clear || resources.UserData[6] != clear || resources.UserData[7] != clear)
        {
            return null;
        }

        var fullDispatch =
            input.DispatchThreadDimensions && input.ThreadsX == ImageClearWaveSize && input.ThreadsY == 1 && input.ThreadsZ == 1 &&
            groupsX != 0 && groupsY == 1 && groupsZ == 1 &&
            input.DispatchThreadsX == groupsX && input.DispatchThreadsY == 1 && input.DispatchThreadsZ == 1 &&
            input.GroupIdX && !input.GroupIdY && !input.GroupIdZ &&
            input.ThreadIdCount == 1 && input.WaveSize == ImageClearWaveSize && !input.ThreadGroupSizeEnabled &&
            dispatchInitiator == ImageClearDispatchInitiator && groupsX % input.ThreadsX == 0 && descriptor.RecordCount == groupsX;
        var size = descriptor.Footprint() ?? throw _host.Fatal($"The compute buffer footprint overflows: stride={descriptor.Stride} records={descriptor.RecordCount}.");
        if (!fullDispatch || size == 0)
        {
            return null;
        }

        return new ComputeImageClear(descriptor, clear, size);
    }

    // A constant fill that covers a whole image or DCC metadata becomes a clear. Anything
    // else, including a fill of plain buffer memory, runs as the guest wrote it.
    private bool TryConsumeConstantFill(ComputeInputInfo input, uint groupsX, uint groupsY, uint groupsZ)
    {
        var program = input.Stage.Program!;
        if (program.ConstantFill is not { } fill || program.UserDataBase != 0)
        {
            return false;
        }

        var userData = input.Stage.Resources.UserData;
        if (fill.GroupScalarRegister != (uint)input.WorkgroupRegister || !input.GroupIdX || input.GroupIdY || input.GroupIdZ ||
            input.ThreadsX != ImageClearWaveSize || input.ThreadsY != 1 || input.ThreadsZ != 1 || groupsY != 1 || groupsZ != 1 ||
            fill.DestinationScalarResource + 4 > userData.Length || fill.SourceScalarResource + 4 > userData.Length)
        {
            return false;
        }

        var destination = BufferDescriptorWords.From(userData.AsSpan((int)fill.DestinationScalarResource, 4).ToArray());
        var source = BufferDescriptorWords.From(userData.AsSpan((int)fill.SourceScalarResource, 4).ToArray());
        Span<byte> valueBytes = stackalloc byte[sizeof(uint)];
        if (destination.Format != Format32UInt || destination.Stride != sizeof(uint) || destination.SwizzleEnabled || destination.AddThreadId ||
            (ulong)groupsX * input.ThreadsX != destination.RecordCount || source.RecordCount == 0 ||
            !_host.TryReadGuest(source.Address, valueBytes))
        {
            return false;
        }

        var value = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(valueBytes);
        var size = (ulong)destination.RecordCount * sizeof(uint);
        var consumed = _host.TryClearImageFromBuffer(destination.Address, size, value) ||
                       _host.TryAbsorbDccFill(destination.Address, size, value);
        if (RenderTrace.Enabled && RenderTrace.MetadataClear())
        {
            RenderTrace.Write($"Constant fill: shader=0x{program.Hash:X16} address=0x{destination.Address:X16} size=0x{size:X} value=0x{value:X8} consumed={consumed}");
        }

        return consumed;
    }

    private const uint Format32SInt = 21;
    private const uint Format32Float = 22;
    private const ulong DescriptorAddressMask = 0x0000_FFFF_FFFF_FFFFul;

    private bool TryConsumeBoundedFill(ComputeInputInfo input, uint groupsX, uint groupsY, uint groupsZ, uint dispatchInitiator)
    {
        var program = input.Stage.Program!;
        if (program.BoundedFill is not { } fill)
        {
            return false;
        }

        if (program.UserDataBase != 0 || fill.GroupScalarRegister != (uint)input.WorkgroupRegister || !input.GroupIdX || input.GroupIdY || input.GroupIdZ ||
            input.ThreadIdCount < 1 || input.ThreadsX != ImageClearWaveSize || input.ThreadsY != 1 || input.ThreadsZ != 1 ||
            groupsX == 0 || groupsY != 1 || groupsZ != 1)
        {
            return RefuseBoundedFill(program,
                $"shape group=s{fill.GroupScalarRegister}/s{input.WorkgroupRegister} ids={input.GroupIdX}{input.GroupIdY}{input.GroupIdZ} tid={input.ThreadIdCount} " +
                $"local={input.ThreadsX}x{input.ThreadsY}x{input.ThreadsZ} dispatch={groupsX}x{groupsY}x{groupsZ} base={program.UserDataBase}");
        }

        var userData = input.Stage.Resources.UserData;
        Span<uint> destinationWords = stackalloc uint[4];
        for (var word = 0; word < destinationWords.Length; word++)
        {
            if (!TryResolveFillWord(fill.Destination[word], userData, out destinationWords[word]))
            {
                return RefuseBoundedFill(program, $"destination word {word} unreadable");
            }
        }

        var destination = BufferDescriptorWords.From(destinationWords);
        uint start = 0;
        uint value;
        if (!TryResolveFillWord(fill.Count, userData, out var count) ||
            (fill.Start is { } startWord && !TryResolveFillWord(startWord, userData, out start)))
        {
            return RefuseBoundedFill(program, "range unreadable");
        }

        if (fill.ConstantValue is { } constant)
        {
            value = constant;
        }
        else if (fill.Value is not { } valueWord || !TryResolveFillWord(valueWord, userData, out value))
        {
            return RefuseBoundedFill(program, "value unreadable");
        }

        // A pattern of differing words (an RGBA16F clear colour is two) can only be an image
        // clear; DCC metadata takes one repeated word.
        Span<uint> patternWords = stackalloc uint[8];
        var patternLength = 1;
        var uniform = true;
        patternWords[0] = value;
        if (fill.Pattern is { } pattern)
        {
            var length = (uint)pattern.Length;
            if (length > patternWords.Length || (fill.PatternLength is { } lengthWord &&
                (!TryResolveFillWord(lengthWord, userData, out length) || length == 0 || length > pattern.Length)))
            {
                return RefuseBoundedFill(program, "pattern length");
            }

            for (var word = 1; word < length; word++)
            {
                if (!TryResolveFillWord(pattern[word], userData, out patternWords[word]))
                {
                    return RefuseBoundedFill(program, $"pattern word {word} unreadable");
                }

                uniform &= patternWords[word] == value;
            }

            patternLength = (int)length;
        }

        if (destination.Stride != sizeof(uint) || destination.SwizzleEnabled || destination.AddThreadId || destination.OutOfBounds != 0 ||
            destination.Type != 0 || (destination.Address & 3) != 0 ||
            (fill.Formatted && destination.Format is not (Format32UInt or Format32SInt or Format32Float)))
        {
            return RefuseBoundedFill(program,
                $"descriptor stride={destination.Stride} swizzle={destination.SwizzleEnabled} tid={destination.AddThreadId} oob={destination.OutOfBounds} " +
                $"type={destination.Type} format={destination.Format} address=0x{destination.Address:X}");
        }

        var threads = (dispatchInitiator & DispatchInitiatorUseThreadDimensions) != 0 ? groupsX : (ulong)groupsX * input.ThreadsX;
        var written = Math.Min(count, threads);
        if (threads > uint.MaxValue || (ulong)start + written > uint.MaxValue)
        {
            return RefuseBoundedFill(program, $"overflow threads={threads} start={start} count={count}");
        }

        var end = Math.Min((ulong)start + written, destination.RecordCount);
        if (end <= start)
        {
            return false;
        }

        var address = destination.Address + (ulong)start * sizeof(uint);
        var size = (end - start) * sizeof(uint);
        // A fill that is not DCC metadata may be a whole image the host can clear instead. The
        // pattern repeats from the thread index, so it starts at the first byte only from zero.
        if (!uniform && start != 0)
        {
            return RefuseBoundedFill(program, $"pattern length={patternLength} varies from start={start}");
        }

        var consumed = (uniform && _host.TryFillDccMetadata(address, size, value)) ||
                       _host.TryClearImageFromKernelFill(address, size, patternWords[..patternLength]);
        if (RenderTrace.Enabled && RenderTrace.MetadataClear())
        {
            RenderTrace.Write($"Bounded fill: shader=0x{program.Hash:X16} address=0x{address:X16} size=0x{size:X} value=0x{value:X8} consumed={consumed}");
        }

        return consumed;
    }

    private bool TryConsumeBoundedCopy(ComputeInputInfo input, uint groupsX, uint groupsY, uint groupsZ, uint dispatchInitiator)
    {
        var program = input.Stage.Program!;
        if (program.BoundedCopy is not { } copy || !BoundedCopyEnabled)
        {
            return false;
        }

        if (program.UserDataBase != 0 || copy.GroupScalarRegister != (uint)input.WorkgroupRegister || !input.GroupIdX || input.GroupIdY || input.GroupIdZ ||
            input.ThreadIdCount < 1 || input.ThreadsX != ImageClearWaveSize || input.ThreadsY != 1 || input.ThreadsZ != 1 ||
            groupsX == 0 || groupsY != 1 || groupsZ != 1)
        {
            _host.NoteKernelFillFallback("copy-shape");
            return RefuseBoundedFill(program, $"copy shape dispatch={groupsX}x{groupsY}x{groupsZ} local={input.ThreadsX}x{input.ThreadsY}x{input.ThreadsZ}");
        }

        var userData = input.Stage.Resources.UserData;
        Span<uint> sourceWords = stackalloc uint[4];
        Span<uint> destinationWords = stackalloc uint[4];
        for (var word = 0; word < 4; word++)
        {
            if (!TryResolveFillWord(copy.Source[word], userData, out sourceWords[word]) ||
                !TryResolveFillWord(copy.Destination[word], userData, out destinationWords[word]))
            {
                _host.NoteKernelFillFallback("copy-unreadable");
                return RefuseBoundedFill(program, $"copy descriptor word {word} unreadable");
            }
        }

        if (!TryResolveFillWord(copy.Count, userData, out var count) || !TryResolveFillWord(copy.Modulus, userData, out var modulus))
        {
            _host.NoteKernelFillFallback("copy-unreadable");
            return RefuseBoundedFill(program, "copy range unreadable");
        }

        // A masked index repeats with the period mask + 1 only when the mask is a run of low bits.
        var period = !copy.MaskIndex ? modulus
            : (modulus & (modulus + 1)) == 0 ? (ulong)modulus + 1
            : 0;
        if (period == 0)
        {
            _host.NoteKernelFillFallback(copy.MaskIndex ? "copy-mask" : "copy-descriptor");
            return RefuseBoundedFill(program, $"copy index {(copy.MaskIndex ? "mask" : "modulus")}=0x{modulus:X}");
        }

        var source = BufferDescriptorWords.From(sourceWords);
        var destination = BufferDescriptorWords.From(destinationWords);
        var threads = (dispatchInitiator & DispatchInitiatorUseThreadDimensions) != 0 ? groupsX : (ulong)groupsX * input.ThreadsX;
        var words = Math.Min(Math.Min((ulong)count, threads), destination.RecordCount);
        // A modulus is the source length as the title gave it; a masked index reads no
        // further than the threads that run.
        var sourceLength = copy.MaskIndex ? Math.Min(period, Math.Max(words, 1)) : period;
        if (!IsPlainWordBuffer(source) || !IsPlainWordBuffer(destination) || source.Format != destination.Format ||
            sourceLength > source.RecordCount)
        {
            _host.NoteKernelFillFallback("copy-descriptor");
            return RefuseBoundedFill(program,
                $"copy src=0x{source.Address:X}/{source.Format}/{source.Stride}/{source.OutOfBounds} dst=0x{destination.Address:X}/{destination.Format}/{destination.Stride}/{destination.OutOfBounds} " +
                $"period={period} records={source.RecordCount}");
        }

        if (words == 0)
        {
            return false;
        }

        var consumed = _host.TryCopyWordsOnHost(destination.Address, source.Address, sourceLength, words);
        if (RenderTrace.Enabled)
        {
            RenderTrace.Write(
                $"Bounded copy: shader=0x{program.Hash:X16} src=0x{source.Address:X16} dst=0x{destination.Address:X16} words={words} period={period} consumed={consumed}");
        }

        return consumed || RefuseBoundedFill(program, $"copy refused by the host dst=0x{destination.Address:X} words={words} modulus={modulus}");
    }

    private static readonly bool BoundedCopyEnabled = Environment.GetEnvironmentVariable("SHARPEMU_HOST_BOUNDED_COPY") != "0";

    private static bool IsPlainWordBuffer(BufferDescriptorWords descriptor) =>
        descriptor.Stride == sizeof(uint) && !descriptor.SwizzleEnabled && !descriptor.AddThreadId && descriptor.OutOfBounds == 0 &&
        descriptor.Type == 0 && (descriptor.Address & 3) == 0 && descriptor.Format is Format32UInt or Format32SInt or Format32Float;

    private static bool RefuseBoundedFill(ShaderProgramInfo program, string reason)
    {
        Diagnostics.DccWriterTrace.Refuse(program.Hash, reason);
        return false;
    }

    private bool TryResolveFillWord(Pipelines.FillWord word, ReadOnlySpan<uint> userData, out uint value)
    {
        value = 0;
        switch (word.Source)
        {
            case Pipelines.FillWordSource.UserData:
                if (word.Register >= userData.Length)
                {
                    return false;
                }

                value = userData[(int)word.Register];
                return true;
            case Pipelines.FillWordSource.BufferResource:
            {
                if (word.Register + 4 > userData.Length)
                {
                    return false;
                }

                var resource = BufferDescriptorWords.From(userData.Slice((int)word.Register, 4));
                var size = resource.Stride == 0 ? resource.RecordCount : (ulong)resource.Stride * resource.RecordCount;
                var offset = (ulong)word.Offset & ~3ul;
                if (offset > size || size - offset < sizeof(uint))
                {
                    return true;
                }

                return TryReadFillWord((resource.Address & ~3ul) + offset, out value);
            }
            case Pipelines.FillWordSource.Pointer:
                return TryReadPointer(word.Register, userData, out var pointer) && TryReadFillWord(pointer + (ulong)word.Offset, out value);
            case Pipelines.FillWordSource.IndirectPointer:
            {
                if (!TryReadPointer(word.Register, userData, out var outer) ||
                    !TryReadFillWord(outer + (ulong)word.PointerOffset, out var low) ||
                    !TryReadFillWord(outer + (ulong)word.PointerOffset + sizeof(uint), out var high))
                {
                    return false;
                }

                var inner = ((low | ((ulong)high << 32)) & DescriptorAddressMask) & ~3ul;
                return TryReadFillWord(inner + (ulong)word.Offset, out value);
            }
            default:
                return false;
        }
    }

    private static bool TryReadPointer(uint register, ReadOnlySpan<uint> userData, out ulong pointer)
    {
        pointer = 0;
        if (register + 2 > userData.Length)
        {
            return false;
        }

        pointer = ((userData[(int)register] | ((ulong)userData[(int)register + 1] << 32)) & DescriptorAddressMask) & ~3ul;
        return pointer != 0;
    }

    private bool TryReadFillWord(ulong address, out uint value)
    {
        value = 0;
        Span<byte> bytes = stackalloc byte[sizeof(uint)];
        if (address == 0 || !_host.TryReadGuest(address, bytes))
        {
            return false;
        }

        value = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(bytes);
        return true;
    }

    private bool TryConsumeImageClear(ComputeInputInfo input, uint groupsX, uint groupsY, uint groupsZ, uint dispatchInitiator)
    {
        if (TryDecodeImageClear(input, groupsX, groupsY, groupsZ, dispatchInitiator) is not { } clear)
        {
            return false;
        }

        var address = clear.Descriptor.Address;
        var hash = input.Stage.Program!.Hash;
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
                    $"{(registered ? "Tracked" : filledOnHost ? "Filled on the host" : "Deferred")} a metadata clear: shader=0x{hash:X16} address=0x{address:X16} size=0x{clear.Size:X16} value=0x{clear.PackedClear:X8}");
            }

            return registered || filledOnHost;
        }

        if (RenderTrace.Enabled && RenderTrace.ImageClear())
        {
            RenderTrace.Write($"Consumed a compute image clear: shader=0x{hash:X16} address=0x{address:X16} size=0x{clear.Size:X16} value=0x{clear.PackedClear:X8}");
        }

        return true;
    }
}
