// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.GpuCommands;

namespace SharpEmu.Gpu.Metal.V2;

// Precise queries, enabled unless SHARPEMU_METAL_V2_NATIVE_OCCLUSION=0. A dump closes a visibility interval, not the render
// encoder. At the encoder boundary a GPU prefix sum preserves each dump's snapshot;
// the existing ordered-copy stream publishes them alongside guest WRITE_DATA stores.
internal sealed partial class MetalRenderHost
{
    internal bool NativeOcclusionEnabled { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_NATIVE_OCCLUSION") != "0";

    // SHARPEMU_METAL_V2_COALESCE_OCCLUSION_DUMPS=0 resolves and notes every block of a dump on its own again.
    internal bool CoalesceOcclusionDumps { get; set; } =
        SharpEmu.ShaderCompiler.PerfSwitches.Enabled("SHARPEMU_METAL_V2_COALESCE_OCCLUSION_DUMPS");

    // SHARPEMU_METAL_V2_DEFER_OCCLUSION_DUMPS=0 drains the deferred draws at every dump again.
    internal bool DeferOcclusionDumps { get; set; } =
        SharpEmu.ShaderCompiler.PerfSwitches.Enabled("SHARPEMU_METAL_V2_DEFER_OCCLUSION_DUMPS");

    // SHARPEMU_METAL_V2_DEFER_OCCLUSION_REDUCTIONS=0 encodes each pass's prefix sum in an encoder of its own again.
    // By default a pass that ends only queues its prefix sum; they are encoded together, in pass order, in
    // one compute encoder just before the next label copies land or the command buffer commits (the only
    // places that read what they produce), instead of an encoder between every two passes.
    internal bool DeferOcclusionReductions { get; set; } =
        SharpEmu.ShaderCompiler.PerfSwitches.Enabled("SHARPEMU_METAL_V2_DEFER_OCCLUSION_REDUCTIONS");

    private static ulong DumpRangeBytes(uint instanceMask, uint strideBytes)
    {
        var mask = instanceMask & 0xFF_FFFFu;
        return mask == 0 ? 0 : checked((ulong)System.Numerics.BitOperations.Log2(mask) * strideBytes + sizeof(ulong));
    }

    // A dump covers at most 24 blocks of at most 32 bytes; past this the blocks are not one result buffer.
    private const ulong MaxCoalescedDumpBytes = 1024;

    private const int OcclusionIntervalCapacity = 4096;
    internal int NativeOcclusionBatchBudget { get; set; } = 256;
    private const ulong OcclusionReady = 1UL << 63;
    private static readonly nint SelOcclusionNewBuffer = MetalNative.Selector("newBufferWithLength:options:");
    private static readonly nint SelOcclusionContents = MetalNative.Selector("contents");
    private static readonly nint SelOcclusionVisibility = MetalNative.Selector("setVisibilityResultBuffer:");
    private static readonly nint SelOcclusionMode = MetalNative.Selector("setVisibilityResultMode:offset:");
    private static readonly nint SelOcclusionPipeline = MetalNative.Selector("setComputePipelineState:");
    private static readonly nint SelOcclusionBuffer = MetalNative.Selector("setBuffer:offset:atIndex:");
    private static readonly nint SelOcclusionBytes = MetalNative.Selector("setBytes:length:atIndex:");
    private static readonly nint SelOcclusionDispatch = MetalNative.Selector("dispatchThreadgroups:threadsPerThreadgroup:");

    private sealed class OcclusionBatch(nint counts, nint results)
    {
        internal readonly nint Counts = counts, Results = results;
        // A shared buffer's contents pointer holds for the buffer's life: one message send at creation, not one per dump.
        internal readonly nint CountsContents = MetalNative.Send(counts, SelOcclusionContents);
        internal ulong Serial;
        internal int Slot, Queue;
        internal int VisibilityMode = -1, VisibilitySlot = -1;
        internal bool CountedDraw, HasSnapshot;
    }

    private readonly List<OcclusionBatch> _oqBatches = [];
    private readonly List<(OcclusionBatch Batch, uint Length)> _oqPendingReductions = [];
    private OcclusionBatch? _oqOpen;
    private int _oqQueue;
    private nint _oqBank, _oqPipeline;
    private long _oqDumps, _oqReductions, _oqBudgetWaits, _oqUnusedPasses, _oqInitializedSlots;
    internal (long Dumps, long Reductions, int Buffers, long BudgetWaits, long UnusedPasses, long InitializedSlots) NativeOcclusionTotals =>
        (_oqDumps, _oqReductions, _oqBatches.Count, _oqBudgetWaits, _oqUnusedPasses, _oqInitializedSlots);

    private const string OcclusionPrefixSource = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void sharpemu_occlusion_prefix(device const ulong* counts [[buffer(0)]],
            device ulong* snapshots [[buffer(1)]], device ulong* carry [[buffer(2)]],
            constant uint& length [[buffer(3)]]) {
            const ulong ready = ulong(1) << 63;
            ulong sum = carry[0] & ~ready;
            for (uint i = 0; i < length; i++) {
                sum = (sum + counts[i]) & ~ready;
                snapshots[i] = ready | sum;
            }
            carry[0] = ready | sum;
        }
        """;

    private unsafe void InitializeNativeOcclusion()
    {
        if (_oqBank != 0) return;
        if (OcclusionAuditEnabled) throw Fatal("Native occlusion and the visibility audit must be tested separately.");
        _oqBank = NewOcclusionBuffer((CommandStreamQueue.QueueCount + 1) * sizeof(ulong));
        new Span<ulong>((void*)MetalNative.Send(_oqBank, SelOcclusionContents), CommandStreamQueue.QueueCount + 1).Fill(OcclusionReady);
        var options = MetalNative.Send(MetalNative.Send(MetalNative.Class("MTLCompileOptions"), MetalNative.Selector("alloc")), MetalNative.Selector("init"));
        MetalNative.ApplyMslCompileOptions(options);
        nint error = 0;
        var library = MetalNative.Send(_device, MetalNative.Selector("newLibraryWithSource:options:error:"),
            MetalNative.NsString(OcclusionPrefixSource), options, ref error);
        MetalNative.SendVoid(options, SelRelease);
        if (library == 0) throw Fatal($"The occlusion prefix kernel did not compile: {MetalNative.DescribeError(error)}.");
        var function = MetalNative.Send(library, MetalNative.Selector("newFunctionWithName:"), MetalNative.NsString("sharpemu_occlusion_prefix"));
        MetalNative.SendVoid(library, SelRelease);
        if (function == 0) throw Fatal("The occlusion prefix function is missing.");
        _oqPipeline = MetalNative.Send(_device, MetalNative.Selector("newComputePipelineStateWithFunction:error:"), function, ref error);
        MetalNative.SendVoid(function, SelRelease);
        if (_oqPipeline == 0) throw Fatal($"The occlusion prefix pipeline did not build: {MetalNative.DescribeError(error)}.");
    }

    private nint NewOcclusionBuffer(int bytes)
    {
        var buffer = MetalNative.SendNewBuffer(_device, SelOcclusionNewBuffer, (nuint)bytes, Mtl.ResourceStorageModeShared);
        if (buffer == 0) throw Fatal($"Could not allocate {bytes} bytes for native occlusion.");
        return buffer;
    }

    private void ValidateOcclusionQueue(int queue)
    {
        if ((uint)queue >= CommandStreamQueue.QueueCount) throw Fatal($"Invalid occlusion queue {queue}.");
    }

    private void NativeOcclusionBeginQueue(int queue)
    {
        if (NativeOcclusionEnabled) ValidateOcclusionQueue(queue);
        if (!NativeOcclusionEnabled || _oqQueue == queue) return;
        EndRendering();
        _oqQueue = queue;
    }

    // Called between passes, before choosing the next command buffer or encoding its
    // indirect arguments. Never recycle an unfinished batch or grow without a bound.
    private void NativeOcclusionMakeRoom()
    {
        if (!NativeOcclusionEnabled || _oqBatches.Count < NativeOcclusionBatchBudget) return;
        var completed = CompletedSerial;
        var oldest = ulong.MaxValue;
        foreach (var batch in _oqBatches)
        {
            if (batch.Serial <= completed) return;
            oldest = Math.Min(oldest, batch.Serial);
        }
        if (oldest == ulong.MaxValue) throw Fatal("The native occlusion batch budget must be positive.");
        if (oldest <= _previousSerial)
            WaitForCommittedSerial(oldest, MetalWaitReason.Other);
        else
            FlushAndWaitForGpu();
        _oqBudgetWaits++;
    }

    private unsafe void NativeOcclusionAttach(nint descriptor)
    {
        if (!NativeOcclusionEnabled) return;
        InitializeNativeOcclusion();
        var completed = CompletedSerial;
        OcclusionBatch? batch = null;
        foreach (var candidate in _oqBatches)
        {
            if (candidate.Serial > completed) continue;
            batch = candidate;
            break;
        }
        if (batch is null)
        {
            // MakeRoom must run before selecting this pass's command buffer.
            if (_oqBatches.Count >= NativeOcclusionBatchBudget) throw Fatal("Native occlusion did not retire a batch before opening a pass.");
            var counts = NewOcclusionBuffer(OcclusionIntervalCapacity * sizeof(ulong));
            nint results;
            try { results = NewOcclusionBuffer(OcclusionIntervalCapacity * sizeof(ulong)); }
            catch { MetalNative.SendVoid(counts, SelRelease); throw; }
            batch = new OcclusionBatch(counts, results);
            _oqBatches.Add(batch);
        }
        batch.Serial = SubmissionSerial;
        batch.Queue = _oqQueue;
        batch.Slot = 0;
        batch.VisibilityMode = batch.VisibilitySlot = -1;
        batch.CountedDraw = batch.HasSnapshot = false;
        // Expose only initialized slots. This batch's previous GPU use has retired,
        // and the command buffer for the new pass has not been committed yet.
        InitializeOcclusionSlot(batch);
        _oqOpen = batch;
        MetalNative.SendVoid(descriptor, SelOcclusionVisibility, batch.Counts);
    }

    private unsafe void InitializeOcclusionSlot(OcclusionBatch batch)
    {
        ((ulong*)batch.CountsContents)[batch.Slot] = 0;
        _oqInitializedSlots++;
    }

    private void NativeOcclusionPrepareDraw()
    {
        if (!NativeOcclusionEnabled || _oqOpen is not { } batch) return;
        var enabled = (_dynamicState.DepthCountControl & (1u | (1u << 8))) == (1u << 8);
        batch.CountedDraw |= enabled;
        var mode = enabled ? 2 : 0;
        if (batch.VisibilityMode == mode && batch.VisibilitySlot == batch.Slot) return;
        MetalV2Native.SendVisibilityResultMode(_renderEncoder, SelOcclusionMode,
            (uint)mode, (nuint)(batch.Slot * sizeof(ulong)));
        batch.VisibilityMode = mode;
        batch.VisibilitySlot = batch.Slot;
    }

    // Called after the render encoder ended, before any deferred label copies land.
    private unsafe void NativeOcclusionFinish()
    {
        if (_oqOpen is not { } batch) return;
        _oqOpen = null;
        // No counter changed and no dump references this batch's result storage.
        // Keep its conservative serial retirement, but emit no utility encoder.
        if (!batch.CountedDraw && !batch.HasSnapshot)
        {
            _oqUnusedPasses++;
            return;
        }
        var length = (uint)Math.Min(batch.Slot + 1, OcclusionIntervalCapacity);
        if (DeferOcclusionReductions)
        {
            // Encoded in pass order by EncodePendingOcclusionReductions, before anything reads its result.
            _oqPendingReductions.Add((batch, length));
            NoteRecordedWork();
            return;
        }

        var encoder = RetainEncoder(MetalNative.Send(EnsureCommandBuffer(), SelComputeEncoder));
        if (encoder == 0) throw Fatal("Could not encode the occlusion prefix sum.");
        WaitForPreviousEncoder(encoder, render: false);
        EncodeOcclusionReduction(encoder, batch, length);
        EndAndReleaseEncoder(encoder, render: false);
        _encodersInBuffer++;
        NoteRecordedWork();
    }

    private unsafe void EncodeOcclusionReduction(nint encoder, OcclusionBatch batch, uint length)
    {
        MetalNative.SendVoid(encoder, SelOcclusionPipeline, _oqPipeline);
        MetalNative.SendSetBuffer(encoder, SelOcclusionBuffer, batch.Counts, 0, 0);
        MetalNative.SendSetBuffer(encoder, SelOcclusionBuffer, batch.Results, 0, 1);
        MetalNative.SendSetBuffer(encoder, SelOcclusionBuffer, _oqBank, (nuint)(batch.Queue * sizeof(ulong)), 2);
        MetalNative.SendSetBytes(encoder, SelOcclusionBytes, (nint)(&length), sizeof(uint), 3);
        MetalNative.SendDispatch(encoder, SelOcclusionDispatch,
            new MtlSize { Width = 1, Height = 1, Depth = 1 }, new MtlSize { Width = 1, Height = 1, Depth = 1 });
        _oqReductions++;
    }

    // Called with no encoder open, before label copies are encoded and before the command buffer commits: the
    // prefix sums of every pass that ended since, in pass order (each queue's running count is carried from one
    // to the next), in one compute encoder.
    private void EncodePendingOcclusionReductions()
    {
        if (_oqPendingReductions.Count == 0) return;
        var encoder = RetainEncoder(MetalNative.Send(EnsureCommandBuffer(), SelComputeEncoder));
        if (encoder == 0) throw Fatal("Could not encode the occlusion prefix sums.");
        WaitForPreviousEncoder(encoder, render: false);
        foreach (var (batch, length) in _oqPendingReductions) EncodeOcclusionReduction(encoder, batch, length);
        _oqPendingReductions.Clear();
        EndAndReleaseEncoder(encoder, render: false);
        _encodersInBuffer++;
        NoteRecordedWork();
    }

    public bool DumpOcclusionCounter(int queueId, ulong address, uint instanceMask, uint strideBytes)
    {
        if (!NativeOcclusionEnabled) return false;
        if (OnCommandThread)
        {
            // No CPU result is needed here. Publish the pending guest range before
            // returning so later producer reads cannot pass the queued GPU dump. The worker
            // records it where it stands (deferring is for the worker's own stream): a producer
            // read of the range is held only until this item has run.
            PostOcclusionDump(queueId, address, instanceMask, strideBytes);
            return true;
        }

        return DumpOcclusionCounter(queueId, address, instanceMask, strideBytes, defer: true);
    }

    private void PostOcclusionDump(int queueId, ulong address, uint instanceMask, uint strideBytes) =>
        _commandThread!.PostRange(() => DumpOcclusionCounterNow(queueId, address, instanceMask, strideBytes), address,
            DumpRangeBytes(instanceMask, strideBytes), UngatedWrites);

    private bool DumpOcclusionCounter(int queueId, ulong address, uint instanceMask, uint strideBytes, bool defer)
    {
        // A dump only orders GPU work, so behind deferred draws it joins their queue rather than draining
        // them: the program resolver keeps its lead over the draws and they record together at the next drain.
        if (defer && DeferOcclusionDumps && TryDeferOcclusionDump(queueId, address, instanceMask, strideBytes))
        {
            return true;
        }

        return DumpOcclusionCounterNow(queueId, address, instanceMask, strideBytes);
    }

    // Its own method so the closure is only built when a dump is deferred: a lambda over the
    // parameters makes the compiler allocate its capture at entry of the method that holds it.
    private bool TryDeferOcclusionDump(int queueId, ulong address, uint instanceMask, uint strideBytes) =>
        TryDeferBehindDraws(
            () => DumpOcclusionCounterNow(queueId, address, instanceMask, strideBytes), address,
            DumpRangeBytes(instanceMask, strideBytes), always: true);

    private bool DumpOcclusionCounterNow(int queueId, ulong address, uint instanceMask, uint strideBytes)
    {
        DrainDraws();
        ValidateOcclusionQueue(queueId);
        InitializeNativeOcclusion();
        NativeOcclusionBeginQueue(queueId);
        if (instanceMask == 0) return true;
        Span<DeferredLabelCopy> copies = stackalloc DeferredLabelCopy[24];
        var count = 0;
        var selected = instanceMask & 0xFF_FFFFu;
        // The blocks sit in one run of guest memory, so one window usually covers them all: one
        // resolve for the run instead of one per block. A run no single window covers falls back
        // to resolving each block.
        var run = default(MetalGuestRange);
        var firstBlock = 0;
        if (CoalesceOcclusionDumps && selected != 0)
        {
            firstBlock = System.Numerics.BitOperations.TrailingZeroCount(selected);
            var lastBlock = System.Numerics.BitOperations.Log2(selected);
            var runBytes = checked((ulong)(lastBlock - firstBlock) * strideBytes + sizeof(ulong));
            if (runBytes <= MaxCoalescedDumpBytes)
            {
                run = _guestMemory.ResolveWritable(checked(address + (ulong)firstBlock * strideBytes), runBytes);
                if (run.Size < runBytes) run = default;
            }
        }
        var batch = _oqOpen;
        var source = batch?.Results ?? _oqBank;
        var offset = batch is null ? (ulong)queueId * sizeof(ulong) : (ulong)batch.Slot * sizeof(ulong);
        for (var block = 0; block < 24; block++)
        {
            if ((instanceMask & (1u << block)) == 0) continue;
            var destinationAddress = checked(address + (ulong)block * strideBytes);
            var destinationBuffer = run.Buffer;
            var destinationOffset = run.Offset + (ulong)(block - firstBlock) * strideBytes;
            if (!run.IsValid)
            {
                var destination = _guestMemory.ResolveWritable(destinationAddress, sizeof(ulong));
                if (!destination.IsValid) throw Fatal($"Native occlusion has no writable result window at 0x{destinationAddress:X16}.");
                destinationBuffer = destination.Buffer;
                destinationOffset = destination.Offset;
            }
            copies[count++] = new DeferredLabelCopy(
                block == 0 ? source : _oqBank,
                block == 0 ? offset : (ulong)CommandStreamQueue.QueueCount * sizeof(ulong),
                destinationBuffer, destinationOffset, sizeof(ulong), destinationAddress);
            if (!CoalesceOcclusionDumps)
            {
                // The note below forgets them as well; this is the order before it did.
                _cpOverlay.ForgetRecordedWrite(destinationAddress, sizeof(ulong));
                _labelOverlay.ForgetRecordedWrite(destinationAddress, sizeof(ulong));
            }
        }
        if (batch is not null)
        {
            batch.HasSnapshot |= (instanceMask & 1) != 0;
            batch.Slot++;
            // Consecutive empty dumps still expose a zero interval. At capacity
            // the pass ends below; there is no next slot to initialize.
            if (batch.Slot < OcclusionIntervalCapacity) InitializeOcclusionSlot(batch);
        }
        _ = EncodeOrderedCopies(copies[..count], address, null, 0, MetalGpuWriteOrigin.Copy,
            "occlusion dump", null, (ulong)count * sizeof(ulong), sourceIsFinal: false, deferStableGpuResult: batch is not null,
            noteAsOneRange: CoalesceOcclusionDumps);
        if (batch is { Slot: OcclusionIntervalCapacity }) EndRendering();
        _oqDumps++;
        return true;
    }

    public bool ResetOcclusionCounter(int queueId)
    {
        if (!NativeOcclusionEnabled) return false;
        if (PostToWorker(() => ResetOcclusionCounter(queueId))) return true;
        DrainDraws();
        ValidateOcclusionQueue(queueId);
        InitializeNativeOcclusion();
        EndEncoders(PassEndCause.Other);
        var copy = new DeferredLabelCopy(_oqBank, (ulong)CommandStreamQueue.QueueCount * sizeof(ulong),
            _oqBank, (ulong)queueId * sizeof(ulong), sizeof(ulong), 0);
        EncodeLabelCopies(EnsureCommandBuffer(), new ReadOnlySpan<DeferredLabelCopy>(in copy));
        NoteRecordedWork();
        return true;
    }

    public unsafe bool TryReadOcclusionCounter(int queueId, out ulong value)
    {
        value = 0;
        if (!NativeOcclusionEnabled) return false;
        if (OnCommandThread)
        {
            ulong result = 0;
            RunTagged(() => TryReadOcclusionCounter(queueId, out result));
            value = result;
            return true;
        }
        DrainDraws();
        ValidateOcclusionQueue(queueId);
        InitializeNativeOcclusion();
        FlushAndWaitForGpu();
        value = ((ulong*)MetalNative.Send(_oqBank, SelOcclusionContents))[queueId] & ~OcclusionReady;
        return true;
    }

    private void DisposeNativeOcclusion()
    {
        foreach (var batch in _oqBatches)
        {
            MetalNative.SendVoid(batch.Counts, SelRelease);
            MetalNative.SendVoid(batch.Results, SelRelease);
        }
        _oqBatches.Clear();
        _oqPendingReductions.Clear();
        if (_oqBank != 0) MetalNative.SendVoid(_oqBank, SelRelease);
        if (_oqPipeline != 0) MetalNative.SendVoid(_oqPipeline, SelRelease);
        _oqBank = _oqPipeline = 0;
        _oqOpen = null;
    }
}
