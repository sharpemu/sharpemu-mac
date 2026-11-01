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

// The resolving half of a dispatch: what it runs and which host consumers may take it in place
// of running it. RenderRecorder.Compute records it through the host.
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

    // True when this executor records its own commands at once (no deferring sink between them):
    // the host is then prepared here, before the program resolves, as it was before the split.
    private bool RecordsDirectly => _recorder is RenderRecorder;

    public void Dispatch(ulong submitId, RegisterBanks banks, uint groupsX, uint groupsY, uint groupsZ, uint dispatchInitiator, ulong indirectArgumentsAddress = 0)
    {
        var compute = banks.Shader.Compute;
        var record = _pool.RentDispatch();
        record.Debug = new DebugInformation(RecordedOperation.DispatchDirect, submitId, groupsX, groupsY, groupsZ, dispatchInitiator, compute.Address);
        record.Skipped = true;
        record.SubmitId = submitId;
        record.ShaderAddress = compute.Address;
        record.PacketIndirectArgumentsAddress = indirectArgumentsAddress;
        var direct = RecordsDirectly;
        if (direct)
        {
            // Pending host operations run, and the debug label is set, before the program resolves.
            if (!_host.IsRecording)
            {
                _pool.Return(record);
                throw _host.Fatal("A dispatch has no recording command buffer.");
            }

            _host.RunPendingOperations();
            _host.SetDebugInformation(RecordedOperation.DispatchDirect, submitId, groupsX, groupsY, groupsZ, dispatchInitiator, compute.Address);
            record.HostPrepared = true;
        }

        if (compute.Address == 0)
        {
            if (RenderTrace.Enabled && RenderTrace.NullComputeShader())
            {
                RenderTrace.Write($"Ignoring a dispatch with no compute shader: groups={groupsX}x{groupsY}x{groupsZ} initiator=0x{dispatchInitiator:X8}");
            }

            Record(record);
            return;
        }

        var unknownBits = dispatchInitiator & ~DispatchInitiatorKnownMask;
        if (unknownBits != 0 && RenderTrace.Enabled && RenderTrace.UnknownInitiator())
        {
            RenderTrace.Write(
                $"The dispatch initiator has unknown bits: initiator=0x{dispatchInitiator:X8} unknown=0x{unknownBits:X8} shader=0x{compute.Address:X16} groups={groupsX}x{groupsY}x{groupsZ}");
        }

        var useThreadDimensions = (dispatchInitiator & DispatchInitiatorUseThreadDimensions) != 0;
        // Thread counts left in guest memory (GpuCommandInterpreter.DispatchIndirectAt): the
        // GPU converts them, so the dispatch comes with no counts at all. Counts the CPU read
        // are converted here, and a zero axis among them skips the dispatch.
        var threadsOnGpu = useThreadDimensions && indirectArgumentsAddress != 0 &&
            groupsX == 0 && groupsY == 0 && groupsZ == 0 && _host.DispatchesThreadsIndirectOnGpu;
        if ((indirectArgumentsAddress == 0 || useThreadDimensions) && !threadsOnGpu &&
            IsEmptyDispatch(compute, useThreadDimensions, groupsX, groupsY, groupsZ))
        {
            if (RenderTrace.Enabled && RenderTrace.ZeroDispatch())
            {
                RenderTrace.Write($"Skipping a zero-sized dispatch: groups={groupsX}x{groupsY}x{groupsZ} initiator=0x{dispatchInitiator:X8} shader=0x{compute.Address:X16}");
            }

            Interlocked.Increment(ref ZeroGroupSkips);
            TraceFrameDispatch(_host, "zero-sized", compute.Address, 0, groupsX, groupsY, groupsZ, indirectArgumentsAddress);
            Record(record);
            return;
        }

        // A dispatch that suspended on its pipeline runs again from its packet; what it prepared
        // then is taken back while it still holds (ComputeProgramRetry), so the retry neither
        // prepares the program again nor re-runs the consumers, which declined it then. Only a
        // dispatch recorded at once can suspend: the recorder must answer before the next packet.
        var suspendable = direct && _host.SupportsPipelineSuspend;
        var retry = suspendable
            ? _suspensions.Take(submitId, compute.Address, indirectArgumentsAddress)
            : null;
        ComputeProgram computeProgram;
        using (RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.DrawProgramResolution))
        {
            computeProgram = _pipelines.GetComputeProgram(banks, dispatchInitiator, groupsX, groupsY, groupsZ, indirectArgumentsAddress != 0, retry);
        }

        var retried = retry?.Reused == true;
        if (retried)
        {
            _computeRetriesReused++;
        }

        if (suspendable)
        {
            record.Retry = retry;
            record.Suspensions = _suspensions;
        }

        if (computeProgram.ImageCopy is not null)
        {
            record.Program = computeProgram;
            record.Skipped = false;
            record.GroupsX = groupsX;
            record.GroupsY = groupsY;
            record.GroupsZ = groupsZ;
            record.IndirectArgumentsAddress = indirectArgumentsAddress;
            Record(record);
            return;
        }

        if (computeProgram.Consumed)
        {
            TraceFrameDispatch(_host, "consumed", compute.Address, 0, groupsX, groupsY, groupsZ, indirectArgumentsAddress);
            Record(record);
            return;
        }

        if (!computeProgram.Available)
        {
            if (RenderTrace.Enabled)
            {
                RenderTrace.Write($"Skipping a dispatch without a program: shader=0x{compute.Address:X16} groups={groupsX}x{groupsY}x{groupsZ}");
            }

            TraceFrameDispatch(_host, "no-program", compute.Address, 0, groupsX, groupsY, groupsZ, indirectArgumentsAddress);
            Record(record);
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

        // The host consumers the recorder offers the dispatch to before running it, in this order:
        // a metadata clear, a constant fill, a bounded fill, a bounded copy and an image clear.
        // A retried dispatch was declined by all of them when it suspended.
        record.Retried = retried;
        if (!retried && indirectArgumentsAddress == 0)
        {
            if ((ulong)groupsX * groupsY * groupsZ >= MetadataClearMinimumGroups)
            {
                DecodeMetadataClear(input, record);
            }

            record.ConstantFill = DecodeConstantFill(input, groupsX, groupsY, groupsZ);
            record.BoundedFill = DecodeBoundedFill(input, groupsX, groupsY, groupsZ, dispatchInitiator);
            record.BoundedCopy = DecodeBoundedCopy(input, groupsX, groupsY, groupsZ, dispatchInitiator);
            record.ImageClear = TryDecodeImageClear(input, groupsX, groupsY, groupsZ, dispatchInitiator);
        }

        if (useThreadDimensions && !threadsOnGpu)
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
            TraceFrameDispatch(_host, "linked-compile", compute.Address, program.Hash, groupsX, groupsY, groupsZ, indirectArgumentsAddress);
            Record(record);
            return;
        }

        var hasStorageWrites = HasBufferWrites(input.Stage);
        foreach (var image in program.Images)
        {
            hasStorageWrites |= image.Written && image.Class == ImageResourceClass.Storage;
        }

        record.Skipped = false;
        record.Program = computeProgram;
        record.GroupsX = groupsX;
        record.GroupsY = groupsY;
        record.GroupsZ = groupsZ;
        record.IndirectArgumentsAddress = indirectArgumentsAddress;
        record.ThreadsOnGpu = threadsOnGpu;
        record.HasStorageWrites = hasStorageWrites;
        var resources = input.Stage.Resources;
        Diagnostics.DispatchRunTally.NoteResolved(computeProgram.Program.Id, resources.Images, resources.Samplers, program.Bindings?.UsesTableBlock == true,
            program.Bindings?.Find(ShaderCompiler.Resources.DescriptorBindingKind.GlobalDataShare) is not null,
            record.MetadataClearCount != 0 || record.ConstantFill is not null || record.BoundedFill is not null || record.BoundedCopy is not null || record.ImageClear is not null,
            computeProgram.KnownWrite is not null);
        Record(record);
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

    // The dispatches suspended on their pipeline (RenderRecorder.RecordDispatch keeps them).
    private readonly ComputeDispatchSuspensions _suspensions = new();
    private long _computeRetriesReused;

    /// <summary>Suspended dispatches that ran again on the program they prepared before, for tests.</summary>
    internal long ComputeRetriesReused => _computeRetriesReused;

    // The dispatch counts threads; the host counts groups of the shader's thread size.
    public static long ZeroGroupSkips;

    private static bool IsEmptyDispatch(ComputeStageRegisters compute, bool useThreadDimensions, uint groupsX, uint groupsY, uint groupsZ)
    {
        if (useThreadDimensions)
        {
            groupsX = GroupsFromThreads(groupsX, compute.ThreadsX);
            groupsY = GroupsFromThreads(groupsY, compute.ThreadsY);
            groupsZ = GroupsFromThreads(groupsZ, compute.ThreadsZ);
        }

        return groupsX == 0 || groupsY == 0 || groupsZ == 0;
    }

    public static uint GroupsFromThreads(uint threads, uint groupSize)
    {
        if (threads == 0)
        {
            return 0;
        }

        var size = Math.Max(groupSize, 1u);
        return (threads + size - 1) / size;
    }

    // SHARPEMU_SKIP_CS and the live skip list: what the program contributes, by its absence.
    internal static bool SkipsComputeProgram(ulong hash) =>
        SkippedComputePrograms.Contains(hash) || LiveSkips.Contains(hash);

    // Under a dump-triggered frame, what became of each dispatch: a pass that never ran leaves
    // no snapshot behind, so the list of dispositions is the only record of it.
    internal static void TraceFrameDispatch(IRenderHost host, string disposition, ulong shader, ulong hash, uint groupsX, uint groupsY, uint groupsZ, ulong indirect)
    {
        if (host.TraceFrameActive)
        {
            Console.Error.WriteLine(
                $"[FRAME-DISPATCH] {disposition} hash=0x{hash:X16} shader=0x{shader:X} groups={groupsX}x{groupsY}x{groupsZ} indirect=0x{indirect:X}");
        }
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

    // SHARPEMU_TRACE_HTILE=1 reports every metadata clear and HTILE store, not the first of each.
    public static readonly bool TraceHtile = Environment.GetEnvironmentVariable("SHARPEMU_TRACE_HTILE") == "1";

    // The buffers a metadata clear would overwrite, for the recorder to check against the host's
    // registered metadata (RenderRecorder.TryClearMetadata). A metadata clear is buffer stores and
    // nothing else: a dispatch that also reads or stores images, or writes through device
    // addresses, does real work beside the metadata - UE5's Nanite depth export writes the depth
    // surface and the HTILE that describes it in one dispatch - and dropping it loses that work and
    // leaves the surface marked cleared, so the next pass clears it again (Black Myth: Wukong's
    // characters and army failed every depth-equal base pass draw).
    private void DecodeMetadataClear(ComputeInputInfo input, ResolvedDispatch record)
    {
        var program = input.Stage.Program!;
        var resources = input.Stage.Resources;
        if (resources.Buffers.Length != program.Buffers.Length)
        {
            throw _host.Fatal($"The compute buffer count does not match the program: descriptors={resources.Buffers.Length} program={program.Buffers.Length}.");
        }

        if (program.HasBitwiseExclusiveOr || program.Images.Length != 0 || program.SamplerCount != 0 ||
            program.WritesDeviceAddresses || program.Buffers.Length == 0)
        {
            return;
        }

        var probes = record.MetadataClearBuffer;
        if (probes.Length < program.Buffers.Length)
        {
            probes = record.MetadataClearBuffer = new MetadataClearProbe[Math.Max(program.Buffers.Length, 4)];
        }

        var anyWritten = false;
        for (var i = 0; i < program.Buffers.Length; i++)
        {
            var resource = program.Buffers[i];
            var descriptor = DecodeBufferDescriptor(resources, i);
            var bytes = descriptor.Stride == 0 ? descriptor.RecordCount : (ulong)descriptor.RecordCount * descriptor.Stride;
            probes[i] = new MetadataClearProbe(descriptor.Address, bytes, descriptor.RecordCount, descriptor.Stride, resource.Written, resource.Read);
            anyWritten |= resource.Written;
        }

        record.MetadataClearCount = anyWritten ? program.Buffers.Length : 0;
    }

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


    // A constant fill that covers a whole image or DCC metadata becomes a clear (RenderRecorder.TryConsume).
    // Anything else, including a fill of plain buffer memory, runs as the guest wrote it.
    private ConstantFillCandidate? DecodeConstantFill(ComputeInputInfo input, uint groupsX, uint groupsY, uint groupsZ)
    {
        var program = input.Stage.Program!;
        if (program.ConstantFill is not { } fill || program.UserDataBase != 0)
        {
            return null;
        }

        var userData = input.Stage.Resources.UserData;
        if (fill.GroupScalarRegister != (uint)input.WorkgroupRegister || !input.GroupIdX || input.GroupIdY || input.GroupIdZ ||
            input.ThreadsX != ImageClearWaveSize || input.ThreadsY != 1 || input.ThreadsZ != 1 || groupsY != 1 || groupsZ != 1 ||
            fill.DestinationScalarResource + 4 > userData.Length || fill.SourceScalarResource + 4 > userData.Length)
        {
            return null;
        }

        var destination = BufferDescriptorWords.From(userData.AsSpan((int)fill.DestinationScalarResource, 4));
        var source = BufferDescriptorWords.From(userData.AsSpan((int)fill.SourceScalarResource, 4));
        Span<byte> valueBytes = stackalloc byte[sizeof(uint)];
        if (destination.Format != Format32UInt || destination.Stride != sizeof(uint) || destination.SwizzleEnabled || destination.AddThreadId ||
            (ulong)groupsX * input.ThreadsX != destination.RecordCount || source.RecordCount == 0 ||
            !_host.TryReadGuest(source.Address, valueBytes))
        {
            return null;
        }

        var value = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(valueBytes);
        return new ConstantFillCandidate(destination.Address, (ulong)destination.RecordCount * sizeof(uint), value);
    }

    private const uint Format32SInt = 21;
    private const uint Format32Float = 22;
    private const ulong DescriptorAddressMask = 0x0000_FFFF_FFFF_FFFFul;

    private KernelFillCandidate? DecodeBoundedFill(ComputeInputInfo input, uint groupsX, uint groupsY, uint groupsZ, uint dispatchInitiator)
    {
        var program = input.Stage.Program!;
        if (program.BoundedFill is not { } fill)
        {
            return null;
        }

        if (program.UserDataBase != 0 || fill.GroupScalarRegister != (uint)input.WorkgroupRegister || !input.GroupIdX || input.GroupIdY || input.GroupIdZ ||
            input.ThreadIdCount < 1 || input.ThreadsX != ImageClearWaveSize || input.ThreadsY != 1 || input.ThreadsZ != 1 ||
            groupsX == 0 || groupsY != 1 || groupsZ != 1)
        {
            return Refuse<KernelFillCandidate>(program,
                $"shape group=s{fill.GroupScalarRegister}/s{input.WorkgroupRegister} ids={input.GroupIdX}{input.GroupIdY}{input.GroupIdZ} tid={input.ThreadIdCount} " +
                $"local={input.ThreadsX}x{input.ThreadsY}x{input.ThreadsZ} dispatch={groupsX}x{groupsY}x{groupsZ} base={program.UserDataBase}");
        }

        var userData = input.Stage.Resources.UserData;
        Span<uint> destinationWords = stackalloc uint[4];
        for (var word = 0; word < destinationWords.Length; word++)
        {
            if (!TryResolveFillWord(fill.Destination[word], userData, out destinationWords[word]))
            {
                return Refuse<KernelFillCandidate>(program, $"destination word {word} unreadable");
            }
        }

        var destination = BufferDescriptorWords.From(destinationWords);
        uint start = 0;
        uint value;
        if (!TryResolveFillWord(fill.Count, userData, out var count) ||
            (fill.Start is { } startWord && !TryResolveFillWord(startWord, userData, out start)))
        {
            return Refuse<KernelFillCandidate>(program, "range unreadable");
        }

        if (fill.ConstantValue is { } constant)
        {
            value = constant;
        }
        else if (fill.Value is not { } valueWord || !TryResolveFillWord(valueWord, userData, out value))
        {
            return Refuse<KernelFillCandidate>(program, "value unreadable");
        }

        // A pattern of differing words (an RGBA16F clear colour is two) can only be an image
        // clear; DCC metadata takes one repeated word.
        var candidate = new KernelFillCandidate { Value = value, PatternLength = 1, Uniform = true };
        Span<uint> patternWords = candidate.Pattern;
        patternWords[0] = value;
        if (fill.Pattern is { } pattern)
        {
            var length = (uint)pattern.Length;
            if (length > patternWords.Length || (fill.PatternLength is { } lengthWord &&
                (!TryResolveFillWord(lengthWord, userData, out length) || length == 0 || length > pattern.Length)))
            {
                return Refuse<KernelFillCandidate>(program, "pattern length");
            }

            for (var word = 1; word < length; word++)
            {
                if (!TryResolveFillWord(pattern[word], userData, out patternWords[word]))
                {
                    return Refuse<KernelFillCandidate>(program, $"pattern word {word} unreadable");
                }

                candidate.Uniform &= patternWords[word] == value;
            }

            candidate.PatternLength = (int)length;
        }

        if (destination.Stride != sizeof(uint) || destination.SwizzleEnabled || destination.AddThreadId || destination.OutOfBounds != 0 ||
            destination.Type != 0 || (destination.Address & 3) != 0 ||
            (fill.Formatted && destination.Format is not (Format32UInt or Format32SInt or Format32Float)))
        {
            return Refuse<KernelFillCandidate>(program,
                $"descriptor stride={destination.Stride} swizzle={destination.SwizzleEnabled} tid={destination.AddThreadId} oob={destination.OutOfBounds} " +
                $"type={destination.Type} format={destination.Format} address=0x{destination.Address:X}");
        }

        var threads = (dispatchInitiator & DispatchInitiatorUseThreadDimensions) != 0 ? groupsX : (ulong)groupsX * input.ThreadsX;
        var written = Math.Min(count, threads);
        if (threads > uint.MaxValue || (ulong)start + written > uint.MaxValue)
        {
            return Refuse<KernelFillCandidate>(program, $"overflow threads={threads} start={start} count={count}");
        }

        var end = Math.Min((ulong)start + written, destination.RecordCount);
        if (end <= start)
        {
            return null;
        }

        // A fill that is not DCC metadata may be a whole image the host can clear instead. The
        // pattern repeats from the thread index, so it starts at the first byte only from zero.
        if (!candidate.Uniform && start != 0)
        {
            return Refuse<KernelFillCandidate>(program, $"pattern length={candidate.PatternLength} varies from start={start}");
        }

        candidate.Address = destination.Address + (ulong)start * sizeof(uint);
        candidate.Size = (end - start) * sizeof(uint);
        return candidate;
    }

    private BoundedCopyCandidate? DecodeBoundedCopy(ComputeInputInfo input, uint groupsX, uint groupsY, uint groupsZ, uint dispatchInitiator)
    {
        var program = input.Stage.Program!;
        if (program.BoundedCopy is not { } copy || !BoundedCopyEnabled)
        {
            return null;
        }

        if (program.UserDataBase != 0 || copy.GroupScalarRegister != (uint)input.WorkgroupRegister || !input.GroupIdX || input.GroupIdY || input.GroupIdZ ||
            input.ThreadIdCount < 1 || input.ThreadsX != ImageClearWaveSize || input.ThreadsY != 1 || input.ThreadsZ != 1 ||
            groupsX == 0 || groupsY != 1 || groupsZ != 1)
        {
            _host.NoteKernelFillFallback("copy-shape");
            return Refuse<BoundedCopyCandidate>(program, $"copy shape dispatch={groupsX}x{groupsY}x{groupsZ} local={input.ThreadsX}x{input.ThreadsY}x{input.ThreadsZ}");
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
                return Refuse<BoundedCopyCandidate>(program, $"copy descriptor word {word} unreadable");
            }
        }

        if (!TryResolveFillWord(copy.Count, userData, out var count) || !TryResolveFillWord(copy.Modulus, userData, out var modulus))
        {
            _host.NoteKernelFillFallback("copy-unreadable");
            return Refuse<BoundedCopyCandidate>(program, "copy range unreadable");
        }

        // A masked index repeats with the period mask + 1 only when the mask is a run of low bits.
        var period = !copy.MaskIndex ? modulus
            : (modulus & (modulus + 1)) == 0 ? (ulong)modulus + 1
            : 0;
        if (period == 0)
        {
            _host.NoteKernelFillFallback(copy.MaskIndex ? "copy-mask" : "copy-descriptor");
            return Refuse<BoundedCopyCandidate>(program, $"copy index {(copy.MaskIndex ? "mask" : "modulus")}=0x{modulus:X}");
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
            return Refuse<BoundedCopyCandidate>(program,
                $"copy src=0x{source.Address:X}/{source.Format}/{source.Stride}/{source.OutOfBounds} dst=0x{destination.Address:X}/{destination.Format}/{destination.Stride}/{destination.OutOfBounds} " +
                $"period={period} records={source.RecordCount}");
        }

        if (words == 0)
        {
            return null;
        }

        // The modulus the host repeats the source by is its length in words.
        return new BoundedCopyCandidate(destination.Address, source.Address, sourceLength, words);
    }

    private static readonly bool BoundedCopyEnabled = Environment.GetEnvironmentVariable("SHARPEMU_HOST_BOUNDED_COPY") != "0";

    private static bool IsPlainWordBuffer(BufferDescriptorWords descriptor) =>
        descriptor.Stride == sizeof(uint) && !descriptor.SwizzleEnabled && !descriptor.AddThreadId && descriptor.OutOfBounds == 0 &&
        descriptor.Type == 0 && (descriptor.Address & 3) == 0 && descriptor.Format is Format32UInt or Format32SInt or Format32Float;

    private static T? Refuse<T>(ShaderProgramInfo program, string reason)
        where T : struct
    {
        Diagnostics.DccWriterTrace.Refuse(program.Hash, reason);
        return null;
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
}
