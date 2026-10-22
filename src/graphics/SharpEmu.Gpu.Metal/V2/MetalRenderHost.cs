// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.HLE.DsTrace;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Gpu.Scheduling;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// The Metal v2 render host: one object behind the three shared contracts the
/// engine drives it with — <see cref="IRenderHost"/> (RenderExecutor),
/// <see cref="IShaderPipelineHost"/> (ShaderPipelineCache) and, in
/// MetalCommandStream.cs, <c>ICommandStreamHost</c> (CommandStreamQueue). The
/// command translation layer requires one object for all three, which is why this
/// type spans those files, exactly as the Vulkan <c>Presenter</c> does.
///
/// Everything records on the GPU worker thread straight into the current
/// MTLCommandBuffer. Metal binds resources per encoder while the shared executor
/// records them before the render pass opens, so the host buffers the pending draw
/// state in fixed fields and applies it to the encoder at draw time. No hot path
/// allocates.
/// </summary>
internal sealed partial class MetalRenderHost :
    IRenderHost, IShaderPipelineHost, IMetalImageCacheOwner, IMetalSubmissionContext, IImageFormatSupport, IDisposable
{
    private const uint MaxFramebufferDimension = 16384;

    private static readonly nint SelCommandBuffer = MetalNative.Selector("commandBuffer");
    private static readonly nint SelCommit = MetalNative.Selector("commit");
    private static readonly nint SelEndEncoding = MetalNative.Selector("endEncoding");
    private static readonly nint SelWaitUntilCompleted = MetalNative.Selector("waitUntilCompleted");
    private static readonly nint SelEncodeSignalEvent = MetalNative.Selector("encodeSignalEvent:value:");
    private static readonly nint SelAddCompletedHandler = MetalNative.Selector("addCompletedHandler:");
    private static readonly nint SelSignaledValue = MetalNative.Selector("signaledValue");
    private static readonly nint SelNewSharedEvent = MetalNative.Selector("newSharedEvent");
    private static readonly nint SelBlitEncoder = MetalNative.Selector("blitCommandEncoder");
    private static readonly nint SelComputeEncoder = MetalNative.Selector("computeCommandEncoder");
    private static readonly nint SelFillBuffer = MetalNative.Selector("fillBuffer:range:value:");
    private static readonly nint SelCopyBuffer =
        MetalNative.Selector("copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:");
    private static readonly nint SelSetLabel = MetalNative.Selector("setLabel:");
    private static readonly nint SelRelease = MetalNative.Selector("release");
    private static readonly nint SelRetain = MetalNative.Selector("retain");

    private readonly nint _device;
    private readonly nint _queue;
    private readonly nint _sharedEvent;
    private readonly ICpuMemory _memory;
    private readonly IGuestBackedSpace _backing;
    private readonly MetalGuestMemory _guestMemory;
    private readonly MetalImageCache _imageCache;
    private readonly MetalSingleCopy? _singleCopy;
    private readonly IMetalImageTransfer _transfer;
    private readonly MetalGpuWorkerRelay _relay;
    private readonly MetalCompletionQueue _completions;

    private nint _commandBuffer;
    private nint _previousCommandBuffer;
    private ulong _previousSerial;
    private long _failedSubmissions;
    private double _gpuBusySeconds;
    private long _gpuBusySamples;
    private ulong _submissionSerial;
    private ulong _nextSerial;
    private long _recordedCommands;
    // The guest byte ranges recorded work may still be writing, with the submission
    // serial that finishes each. A CPU read waits only when it overlaps one whose
    // serial has not completed, and then only for that serial.
    private readonly MetalGpuWriteRanges _gpuWrites = new();

    // SHARPEMU_METAL_V2_GUARD_WRITES=1 only: checks that every declared GPU-side write
    // leaves the bytes either side of its range alone.
    private readonly MetalGuestWriteGuard? _guard;

    // SHARPEMU_TRAP_SLOT only: watches the trapped blocks across each submission, for a
    // GPU store that lands where no declared range said it would.
    private readonly MetalSlotTrap? _slotTrap;
    // Without ranges every write is "somewhere in guest memory", which is what the
    // first version of this host did; kept for A/B measurement.
    private static readonly bool ConservativeReads =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_CONSERVATIVE_READS") == "1";

    // Waits in -[MTLCommandBuffer waitUntilCompleted] instead of on the shared event
    // the submission signals. Measured slower on this host: the driver still took
    // 100 ms and more to schedule the submission, and the fps halved, so the host
    // waits on the event and this is kept only for comparison.
    private static readonly bool WaitOnCommandBuffer =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_WAIT_ON_COMMAND_BUFFER") == "1";
    private long _drawCount;
    private long _drawTicks;
    private long _pipelineCreations;
    // How long the driver spent turning functions into GPU binaries, on this thread.
    private long _pipelineTicks;

    // How long the MSL front end spent turning sources into libraries.
    private long _libraryTicks;

    // How long the GPU worker waited for libraries still compiling on the module pool.
    private long _libraryWaitTicks;
    // The split of _libraryWaitTicks: waits on a compile the prewarm had in the driver, and
    // misses that compiled while prewarm compiles were in it; how many waits, and how many
    // modules the worker compiled itself instead of waiting for the pool.
    private long _libraryWaitPrewarmTicks;
    private long _libraryWaitContendedTicks;
    private long _libraryWaits;
    private long _libraryInlineCompiles;
    private long _shaderCompilations;
    private long _flushes;
    private long _gpuWaits;
    private long _partialWaits;
    private long _gpuWaitTicks;
    private long _idleTicks;
    private long _statsTimestamp;
    private bool _disposed;

    public MetalRenderHost(
        nint device,
        nint queue,
        ICpuMemory memory,
        IGuestBackedSpace backing,
        PageGuard pages,
        MetalGpuWorkerRelay relay)
    {
        _device = device;
        _queue = queue;
        _memory = memory;
        _backing = backing;
        _relay = relay;
        PerVertexPixelInputsSupported = SupportsApple10Features(device);
        DepthBoundsTestSupported = SupportsApple10Features(device);
        _sharedEvent = MetalNative.Send(device, SelNewSharedEvent);
        if (_sharedEvent == 0)
        {
            throw GpuFatal.Fatal("Metal v2 needs an MTLSharedEvent for submission completion.");
        }

        _guard = MetalGuestWriteGuard.Enabled ? new MetalGuestWriteGuard(memory) : null;
        _slotTrap = MetalSlotTrap.Enabled ? new MetalSlotTrap(memory) : null;
        _guestMemory = new MetalGuestMemory(device, backing, memory) { Trap = _slotTrap };
        if (_slotTrap is not null)
        {
            _slotTrap.Memory = _guestMemory;
        }
        _guestMemory.AttachToQueue(queue);
        _transfer = new MetalImageTransfer(device);
        _imageCache = new MetalImageCache(this, _transfer, _guestMemory, pages, backing);
        if (MetalSingleCopy.Enabled && MetalSingleCopy.CanServe(_guestMemory, backing))
        {
            _singleCopy = new MetalSingleCopy(device, _guestMemory, backing, pages, () => CompletedSerial);
            _imageCache.SingleCopy = _singleCopy;
        }
        if (MetalStagedRangeCache.Enabled)
        {
            _stagedRanges = new MetalStagedRangeCache(
                new MetalStagedRangeBackend(_guestMemory, pages), MetalStagedRangeCache.ConfiguredBudgetBytes);
            _imageCache.StagedRanges = _stagedRanges;
        }
        _completions = new MetalCompletionQueue(_sharedEvent);
        RegisterMemoryTelemetry();
    }

    public MetalGuestMemory GuestMemory => _guestMemory;

    public MetalImageCache ImageCache => _imageCache;

    public MetalCompletionQueue Completions => _completions;

    public MetalGpuWorkerRelay Relay => _relay;

    nint IMetalSubmissionContext.Device => _device;

    nint IMetalImageCacheOwner.Device => _device;

    ulong IMetalImageCacheOwner.CurrentSerial => SubmissionSerial;

    public nint CommandBuffer => EnsureCommandBuffer();

    public ulong SubmissionSerial
    {
        get
        {
            _ = EnsureCommandBuffer();
            return _submissionSerial;
        }
    }

    public ulong CompletedSerial
    {
        get
        {
            var completed = MetalV2Native.SendValue64(_sharedEvent, SelSignaledValue);
            _completions?.NoteSignaled(completed);
            return completed;
        }
    }

    public RenderHostLimits Limits { get; } =
        new(MaxFramebufferDimension, MaxFramebufferDimension, MaxFramebufferDimension, MaxFramebufferDimension);

    public IImageFormatSupport FormatSupport => this;

    public uint MaxPushDescriptors => 0;

    // Metal threadgroups are always 32-wide on Apple silicon; a wave64 compute
    // program is emitted as two wave32 halves by the translator.
    public bool ComputeWave64Supported => false;

    // Pixel shaders run each guest wave32 on one 32-wide fragment SIMD-group, with real
    // ballots, shuffles and quad lanes (see Gen5MslTranslator.HasRealWaveLanes); vertex
    // functions have no documented SIMD-group support and stay single-lane in the
    // translator either way. SHARPEMU_GRAPHICS_SUBGROUPS=0 forces the single-lane model
    // for pixel shaders too, the same switch the Vulkan backend reads.
    public bool GraphicsSubgroupOperationsEnabled { get; } =
        Environment.GetEnvironmentVariable("SHARPEMU_GRAPHICS_SUBGROUPS") != "0";

    // Metal has no 64-bit atomics on threadgroup memory: MSL's atomic_ulong only offers
    // atomic_min/max_explicit on device memory (newer GPU families), with no add, or, exchange or
    // compare-exchange in any address space, so a threadgroup ulong atomic does not compile.
    // A lock cannot stand in for it either: a 64-bit LDS read would have to take the same
    // per-address lock to never see a half-updated pair. The translator instead runs
    // DS_ADD_U64 / DS_OR_B64 as 32-bit atomics on each dword (folding the low dword's carry
    // into the high one), whose final value is exact; only a concurrent read can observe
    // the pair torn. The SPIR-V translator emits that same pair even when this is true.
    public bool SharedInt64AtomicsEnabled => PrewarmSharedInt64Atomics;

    // Every guest library is compiled with fast math off (MetalNative.ApplyMslCompileOptions),
    // and Metal's precise math keeps F32 signed zero, infinity and NaN.
    public bool ShaderSignedZeroInfNanPreserveFloat32Supported => PrewarmSignedZeroInfNanPreserve;

    // The compile flags above, for the shader prewarm, which compiles before the host
    // exists and has to build the same MSL the running title will. Exec-guard elision
    // keeps the interface default.
    internal const bool PrewarmSharedInt64Atomics = false;
    internal const bool PrewarmSignedZeroInfNanPreserve = true;
    internal const bool PrewarmExecGuardElision = true;

    public ShaderPrewarmList? ShaderPrewarm => MetalShaderPrewarm.List;
    public GraphicsPrewarmList? GraphicsPrewarm => MetalShaderPrewarm.Graphics;

    // Every Metal device has eight [[clip_distance]] planes, so the translator can spend
    // one on culling primitives with a zero position like the guest does.
    public bool ClipDistanceEnabled => true;

    public GpuSampleCount NoAttachmentSampleCounts => GpuSampleCount.Count1Bit;

    // Per-vertex pixel inputs are MSL 4.0's metal::vertex_value (macOS 26). The feature set
    // tables list "pre-raster per-vertex values" for Apple10 (M5) only; an M4 compiles the
    // source but get() returns garbage, so nothing older may claim it.
    public bool PerVertexPixelInputsSupported { get; }

    // The encoder's depth bounds test (setDepthTestMinBound:maxBound:, macOS 26) is listed
    // for Apple10 only. An M4 (Apple9) driver answers respondsToSelector: yes but its
    // implementation is a trap (brk #1), so the family decides, not the selector.
    public bool DepthBoundsTestSupported { get; }

    // The bounds are encoder state a draw sets from the dynamic state (ApplyDepthBounds), never
    // part of a pipeline state, so they stay out of the pipeline key.
    public bool DepthBoundsInPipelineState => false;

    private const nint GpuFamilyApple10 = 1010;

    private static bool SupportsApple10Features(nint device)
    {
        var supportsFamily = MetalNative.Selector("supportsFamily:");
        return OperatingSystem.IsMacOSVersionAtLeast(26) &&
            MetalNative.SendBool(device, MetalNative.Selector("respondsToSelector:"), supportsFamily) &&
            MetalNative.SendBool(device, supportsFamily, GpuFamilyApple10);
    }

    public bool IsRecording => !_disposed && EnsureCommandBuffer() != 0;

    /// <summary>The command buffer being recorded; one is opened on demand.</summary>
    private nint EnsureCommandBuffer()
    {
        if (_commandBuffer != 0)
        {
            return _commandBuffer;
        }

        _commandBuffer = MetalNative.Send(_queue, SelCommandBuffer);
        MetalMemoryTelemetry.CommandBuffers.Add();
        if (_commandBuffer == 0)
        {
            throw GpuFatal.Fatal("A Metal command buffer could not be created.");
        }

        // Retain it: the queue hands back an autoreleased buffer, and the buffer stays open
        // across worker passes (a coalesced flush) while each pass drains its own pool.
        // Released once, when the in-flight ring is done with it (or in Dispose).
        MetalNative.Send(_commandBuffer, MetalNative.Selector("retain"));
        _submissionSerial = ++_nextSerial;
        var completed = CompletedSerial;
        _guestMemory.BeginSubmission(_submissionSerial, completed);
        // Staging slabs idle for a while are released or made purgeable here too, not only
        // when a transfer asks for one: a load that stops transferring keeps submitting.
        StagingTiler?.TrimStaging(completed);
        // So are the shader modules and pipeline states past their bound (ShaderCache).
        MaybeTrimShaderCache(completed);
        // Every watch whose submission the GPU has finished is checked here, on the one
        // thread that records them, rather than from the completion handler.
        _guard?.Verify(completed);
        _slotTrap?.Verify(completed);
        // The CPU trap releases a page to let a faulting store land; this is the steady
        // point at which it is taken back.
        SharpEmu.HLE.GpuMemory.GuestSlotTrap.Rearm();
        return _commandBuffer;
    }

    // Everything the cache and the transfers encode needs the encoders closed first.
    void IMetalImageCacheOwner.LandHeldCopies(ulong address, ulong size) => OrderAfterDeferredCopies(address, size);

    IMetalSubmissionContext IMetalImageCacheOwner.BeginTransfer(MetalTransferCause cause) =>
        BeginTransferTouching(cause, 0, 0);

    IMetalSubmissionContext IMetalImageCacheOwner.BeginTransfer(
        MetalTransferCause cause, nint texture, nint otherTexture) =>
        BeginTransferTouching(cause, texture, otherTexture);

    private IMetalSubmissionContext BeginTransferTouching(MetalTransferCause cause, nint texture, nint otherTexture)
    {
        _transferCauses[(int)cause]++;
        if (PassProfileEnabled)
        {
            _ppTransferCause = $"transfer:{cause}";
        }

        if (_preparation is null)
        {
            // Between packets the open pass is all a commit costs. Inside a preparation
            // the images already acquired are stamped with this serial, so the bound
            // waits for the next preparation to begin.
            CommitIfTransientOverBudget();
        }

        _ = EnsureCommandBuffer();
        NoteRecordedWork();
        _transferInSharedEncoder = false;
        if (TransfersShareCompute && _computeEncoder != 0 && texture != 0 && otherTexture == 0 &&
            cause is (MetalTransferCause.Upload or MetalTransferCause.Writeback) && !PendingClearTouches(texture))
        {
            // The open compute encoder stays open: a transfer the tiler encodes as nothing
            // but dispatches records them into it (TryShareComputeEncoder), and one that
            // needs an encoder of its own ends it first (ReleaseSharedEncoder), as a
            // transfer, exactly where this used to. A clear of the texture still waiting
            // would have to land first, which takes a pass, so that case ends it here.
            _transferSharesCompute = true;
            _sharingTransferCause = cause;
            NoteProfileTransfer();
            return this;
        }

        EndEncoders(PassEndCause.Transfer);
        _encodersInBuffer++;
        // A transfer that reads or writes a texture has to see the clears that were
        // asked for before it - but only the clears of that texture: any other one keeps
        // waiting for the pass that can absorb it as its load action. A transfer that
        // does not say which textures it touches drains them all.
        //
        // A fill, a copy of guest memory and a label copy touch no texture at all. A
        // clear changes nothing but its texture, so guest memory can only see one
        // through the writeback of that texture, which is a transfer that names it: the
        // copy writes back the images over its source first, and an image over its
        // destination is uploaded again, by a transfer that names it, before it is used.
        if (cause is MetalTransferCause.GuestFill or MetalTransferCause.EndOfPipeLabel or
            MetalTransferCause.GuestCopy)
        {
        }
        else if (texture == 0 && otherTexture == 0)
        {
            MaterializePendingClears();
        }
        else
        {
            MaterializePendingClearsOf([texture, otherTexture]);
        }

        NoteProfileTransfer();
        return this;
    }

    /// <summary>
    /// Whether an image upload or writeback recorded while a compute encoder is open may
    /// record its tiler dispatches into that encoder (SHARPEMU_METAL_V2_TRANSFER_IN_ENCODER).
    /// </summary>
    internal bool TransfersShareCompute { get; set; } = MetalV2Diagnostics.TransferInEncoder;

    // The transfer begun last left the open compute encoder to be shared, and has not
    // ended it yet.
    private bool _transferSharesCompute;
    private MetalTransferCause _sharingTransferCause;

    // The transfer begun last recorded its dispatches into the shared encoder and did not
    // open one of its own afterwards: it belongs to that pass, not between passes.
    private bool _transferInSharedEncoder;

    // Transfers recorded into an open compute encoder instead of ending it.
    private long _transfersInEncoder;

    internal long TransfersInEncoder => Interlocked.Read(ref _transfersInEncoder);

    nint IMetalSubmissionContext.TryShareComputeEncoder(ulong guestAddress, ulong size)
    {
        if (!_transferSharesCompute || _computeEncoder == 0)
        {
            return 0;
        }

        // What lands when the encoder ends - a deferred command-processor write, a staged
        // store written back - lands after everything recorded into it. A transfer over
        // those bytes has to see them land first (an upload) or land after them (a
        // writeback), so it gets an encoder of its own after the boundary instead.
        if (DeferredLabelCopiesTouch(guestAddress, size) ||
            (_stagedStorageCopies is { Count: > 0 } staged && staged.HasRecordedStoresOver(guestAddress, size)))
        {
            _landDeferredLabels |= DeferredLabelCopiesTouch(guestAddress, size);
            ((IMetalSubmissionContext)this).ReleaseSharedEncoder();
            return 0;
        }

        // A serial compute encoder already runs its dispatches in order; the barrier says
        // so for the guest windows and textures whichever way they are tracked.
        MetalV2Native.SendMemoryBarrierScope(
            _computeEncoder, SelComputeMemoryBarrier, Mtl.BarrierScopeBuffers | Mtl.BarrierScopeTextures);
        _transferInSharedEncoder = true;
        return _computeEncoder;
    }

    void IMetalSubmissionContext.EndSharedCompute()
    {
        if (_computeEncoder == 0)
        {
            return;
        }

        MetalV2Native.SendMemoryBarrierScope(
            _computeEncoder, SelComputeMemoryBarrier, Mtl.BarrierScopeBuffers | Mtl.BarrierScopeTextures);
        // The tiler bound its own pipeline; the next dispatch binds the guest's again.
        // Its buffers are bound per dispatch already (ApplyStageBindings).
        _boundComputePipeline = 0;
        Interlocked.Increment(ref _transfersInEncoder);
        if (PassProfileEnabled)
        {
            PassProfileNoteWork(_sharingTransferCause.ToString(), "transfer");
        }
    }

    void IMetalSubmissionContext.ReleaseSharedEncoder()
    {
        if (!_transferSharesCompute)
        {
            return;
        }

        _transferSharesCompute = false;
        _transferInSharedEncoder = false;
        EndEncoders(PassEndCause.Transfer);
        _encodersInBuffer++;
    }

    // The newest submission that has GPU work in it. An end-of-pipe completion only
    // has to wait for the GPU while this is ahead of what the GPU has finished.
    private ulong _lastWorkSerial;

    /// <summary>Marks the submission being recorded as carrying GPU work.</summary>
    public void NoteRecordedWork() => _lastWorkSerial = _submissionSerial;

    /// <summary>The newest submission that carries GPU work.</summary>
    public ulong LastWorkSerial => _lastWorkSerial;

    // Why an open render pass was ended from outside the draw stream, for the TBDR
    // line: a pass that ends early stores and reloads every attachment, and which of
    // these ended it is the first thing that says whether the split can be avoided.
    private enum PassEndCause
    {
        Transfer,
        Flush,
        Present,
        Other,
    }

    /// <summary>Why a command buffer was committed (the flush_by field of the PERF line).</summary>
    internal enum MetalFlushCause : byte
    {
        Stream,          // the command stream asked, once per guest submission that made progress
        Labels,          // the worker parked with label bytes owed to a polling guest
        Completion,      // an end-of-pipe completion needed its submission committed
        Present,         // a flip
        Budget,          // 2048 recorded commands
        WaitGpu,         // a host wait for the GPU
        WaitUncommitted, // a wait targeted the submission being recorded
        Poll,            // a guest poll with a pending write
        Transient,       // transient staging over budget
        Other,
    }

    /// <summary>Ends whatever encoder is open; the next record opens the one it needs.</summary>
    private void EndEncoders(PassEndCause cause)
    {
        _transferSharesCompute = false;
        if (_renderEncoder != 0)
        {
            PassProfileNoteEnd(cause == PassEndCause.Transfer ? _ppTransferCause : cause.ToString().ToLowerInvariant());
            switch (cause)
            {
                case PassEndCause.Transfer: _passEndTransfer++; break;
                case PassEndCause.Flush: _passEndFlush++; break;
                case PassEndCause.Present: _passEndPresent++; break;
                default: _passEndOther++; break;
            }

            EndRendering();
            _encodersInBuffer++;
        }

        if (_computeEncoder != 0)
        {
            _encodersInBuffer++;
            PassProfileNoteEnd(cause == PassEndCause.Transfer ? _ppTransferCause : cause.ToString().ToLowerInvariant());
            EndAndReleaseEncoder(_computeEncoder);
            _computeEncoder = 0;
            _boundComputePipeline = 0;
            PassProfileClose();
            SnapshotComputeStorage();
        }

        EncodeStagedStores();
        EncodeDeferredLabelCopies();
    }

    /// <summary>Signals the shared event, commits, and opens no new command buffer.</summary>
    public ulong FlushCommandBuffer() => FlushCommandBuffer(PassEndCause.Flush, MetalFlushCause.Other);

    /// <summary>Commits the command buffer being recorded, naming why for the PERF line.</summary>
    public ulong FlushCommandBuffer(MetalFlushCause why) => FlushCommandBuffer(PassEndCause.Flush, why);

    private ulong FlushCommandBuffer(PassEndCause cause) =>
        FlushCommandBuffer(
            cause,
            cause switch
            {
                PassEndCause.Present => MetalFlushCause.Present,
                PassEndCause.Transfer => MetalFlushCause.Transient,
                _ => MetalFlushCause.Other,
            });

    // Why command buffers were committed. A commit drains the GPU pipeline and leaves it
    // idle until the next one is committed, so the count per cause is the first thing
    // that says which commits a frame could do without.
    private readonly long[] _flushCauses = new long[(int)MetalFlushCause.Other + 1];

    // Render, compute and transfer encoders recorded into the open command buffer; one of
    // the limits on how long a coalesced Stream flush keeps it open.
    private int _encodersInBuffer;

    private ulong FlushCommandBuffer(PassEndCause cause, MetalFlushCause why)
    {
        using var flushScope = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(
            SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.CommandFlush);
        // An upload left for a pass is made before the buffer it belongs in is committed.
        _imageCache.DrainDeferredRefreshes();
        if (_commandBuffer == 0)
        {
            return _nextSerial;
        }

        _flushCauses[(int)why]++;
        _landDeferredLabels = true;
        EndEncoders(cause);
        MaterializePendingClears();
        _stagedStorageCopies?.EndSubmission();
        // Compared here as well as when a command buffer opens: a run that ends between
        // the two sees every snapshot it took go unchecked, which is how the one report
        // this watch exists for was lost twice.
        _slotTrap?.Verify(CompletedSerial);
        var serial = _submissionSerial;
        MetalV2Native.SendEventValue(_commandBuffer, SelEncodeSignalEvent, _sharedEvent, serial);
        if (_completions.WantsCompletionHandler)
        {
            // The handler is a global block with no state: it only wakes whoever waits
            // on the shared event, which the signal above has already advanced.
            MetalNative.SendVoid(_commandBuffer, SelAddCompletedHandler, MetalCompletionSignal.Block);
        }

        _slotTrap?.OnCommit(serial);
        PassProfileCommit(serial, cause == PassEndCause.Present);
        if (cause == PassEndCause.Present)
        {
            _imageCache.MarkFrame();
        }

        NoteCommandBufferForTelemetry();
        MetalMemoryTelemetry.Commits.Add();
        MetalNative.SendVoid(_commandBuffer, SelCommit);
        if (DsTrace.Gpu && DsTrace.Armed)
        {
            DsTrace.Emit(DsEv.MCOMMIT, serial, (ulong)cause, _pendingLabelWrites,
                SharpEmu.Libs.Gpu.GpuCommands.CommandStreamQueue.DsCurrentSubmissionId);
        }
        if (_pendingLabelWrites != 0 && ScheduleLabelSubmissions &&
            System.Diagnostics.Stopwatch.GetTimestamp() - _lastCommitTimestamp > QuietQueueTicks)
        {
            // A queue that commits a hundred buffers in two seconds waits hundreds of
            // milliseconds for the driver to pick one up, which nothing noticed while the
            // host published labels itself. Now a guest thread is spinning on bytes this
            // submission writes, and Quake's engine gives up and aborts after half a
            // second of them. Waiting until it is scheduled makes the driver take it now;
            // it is not waiting for the GPU, which is the whole point of writing the label
            // on the GPU in the first place.
            //
            // Only on a queue that has gone quiet, which is the state the driver dawdles
            // in: Quake commits one buffer every twenty milliseconds at its intro and
            // waits, while Superliminal commits one every two or three and needs no
            // prompting at all. Prompting on every label-carrying commit put Superliminal
            // into long stalls instead, because the wait is then pure latency on the
            // worker.
            //
            // The wait itself happens on a thread of its own: it returns only once the
            // kernel has fed the whole buffer to the GPU, which for Silent Hill's frames
            // is most of their GPU time, and the worker waiting for it stopped recording
            // for that long every frame.
            (_scheduleNudge ??= new MetalScheduleNudge()).Nudge(_commandBuffer);
            Interlocked.Increment(ref _scheduledLabelSubmissions);
        }

        _previousCommitTime = CACurrentMediaTime();
        _lastCommitTimestamp = System.Diagnostics.Stopwatch.GetTimestamp();
        // A command buffer that fails never signals its serial, and the completion
        // queue would then wait out its timeout on every submission behind it, so
        // every submission's status is read once it has finished.
        DrainFinishedSubmissions();
        ReportStepCaps();
        RetainInFlight(_commandBuffer, serial);
        // The host's own reference on the buffer it just committed, for the wait that reads
        // its timing after this flush (or after a later call that flushed nothing). The
        // in-flight ring releases its reference once the GPU is done with the buffer, and
        // this thread's pool drops the autoreleased one, so a bare pointer kept here
        // would dangle by then.
        MetalNative.SendVoid(_commandBuffer, SelRetain);
        if (_previousCommandBuffer != 0)
        {
            MetalNative.SendVoid(_previousCommandBuffer, SelRelease);
        }

        _previousCommandBuffer = _commandBuffer;
        _previousSerial = serial;
        _commandBuffer = 0;
        _recordedCommands = 0;
        // Whatever cause committed it, a deferred Stream flush is now satisfied.
        _encodersInBuffer = 0;
        _commitWantedSince = 0;
        _coalescedSubmissions = 0;
        // Every label this submission carried is now on its way to the guest window.
        _pendingLabelWrites = 0;
        _lazyLabelRanges.Clear();
        Interlocked.Increment(ref _flushes);
        _completions.Publish(serial);
        return serial;
    }

    // MTLCommandBufferStatus.
    private const nuint CommandBufferStatusCompleted = 4;
    private const nuint CommandBufferStatusError = 5;

    private static readonly nint SelGpuStartTime = MetalNative.Selector("GPUStartTime");
    private static readonly nint SelGpuEndTime = MetalNative.Selector("GPUEndTime");
    private static readonly nint SelKernelStartTime = MetalNative.Selector("kernelStartTime");
    private static readonly nint SelKernelEndTime = MetalNative.Selector("kernelEndTime");
    [System.Runtime.InteropServices.DllImport("/System/Library/Frameworks/QuartzCore.framework/QuartzCore")]
    private static extern double CACurrentMediaTime();

    private int _slowWaitReports;
    private double _previousCommitTime;
    private ulong _lastReadAddress;
    private ulong _lastReadSize;
    private readonly long[] _waitsByReason = new long[MetalWaitReasons.Count];
    private readonly long[] _waitTicksByReason = new long[MetalWaitReasons.Count];

    // Committed submissions whose GPU time has not been read yet. Sampling only the
    // one buffer that happened to be finished at the next flush missed almost every
    // submission while any were in flight, which is why a title whose GPU was busy
    // for a hundred milliseconds a frame could report gpu_busy_ms=0.0 and look as
    // though it were waiting on nothing.
    private const int MaxInFlightSubmissions = 64;

    private readonly nint[] _inFlightBuffers = new nint[MaxInFlightSubmissions];
    private readonly ulong[] _inFlightSerials = new ulong[MaxInFlightSubmissions];
    private readonly double[] _inFlightCommitTimes = new double[MaxInFlightSubmissions];
    private int _inFlightHead;
    private int _inFlightCount;
    private long _unsampledSubmissions;
    private ulong _notifiedCompletedSerial;

    private void RetainInFlight(nint buffer, ulong serial)
    {
        if (_inFlightCount == MaxInFlightSubmissions)
        {
            // The oldest has not finished and the ring is full: let it go unsampled
            // rather than block the worker on it.
            var oldest = _inFlightBuffers[_inFlightHead];
            _inFlightHead = (_inFlightHead + 1) % MaxInFlightSubmissions;
            _inFlightCount--;
            _unsampledSubmissions++;
            if (PassProfileEnabled)
            {
                MetalNative.SendVoid(oldest, SelWaitUntilCompleted);
            }

            PassProfileResolve(
                _inFlightSerials[(_inFlightHead + MaxInFlightSubmissions - 1) % MaxInFlightSubmissions],
                PassProfileEnabled,
                0);
            if (DsTrace.Gpu && DsTrace.Armed)
            {
                DsKeepLate(oldest, _inFlightSerials[(_inFlightHead + MaxInFlightSubmissions - 1) % MaxInFlightSubmissions]);
            }

            MetalNative.SendVoid(oldest, SelRelease);
        }

        var slot = (_inFlightHead + _inFlightCount) % MaxInFlightSubmissions;
        _inFlightBuffers[slot] = buffer;
        _inFlightSerials[slot] = serial;
        _inFlightCommitTimes[slot] = _previousCommitTime;
        _inFlightCount++;
    }

    // The still-held command buffer a committed serial was recorded into, for the timing a
    // wait reports; zero once the ring has let it go.
    private nint InFlightBufferFor(ulong serial, out double commitTime)
    {
        for (var index = 0; index < _inFlightCount; index++)
        {
            var slot = (_inFlightHead + index) % MaxInFlightSubmissions;
            if (_inFlightSerials[slot] == serial)
            {
                commitTime = _inFlightCommitTimes[slot];
                return _inFlightBuffers[slot];
            }
        }

        commitTime = 0;
        return 0;
    }

    /// <summary>Reads and releases every submission the GPU has finished.</summary>
    private void DrainFinishedSubmissions()
    {
        if (_inFlightCount == 0 && _notifiedCompletedSerial >= _nextSerial)
        {
            return;
        }

        var completed = CompletedSerial;
        if (completed > _notifiedCompletedSerial)
        {
            _notifiedCompletedSerial = completed;
            NotifyGpuCompleted(completed);
        }

        while (_inFlightCount > 0 && _inFlightSerials[_inFlightHead] <= completed)
        {
            var buffer = _inFlightBuffers[_inFlightHead];
            var serial = _inFlightSerials[_inFlightHead];
            _inFlightBuffers[_inFlightHead] = 0;
            _inFlightHead = (_inFlightHead + 1) % MaxInFlightSubmissions;
            _inFlightCount--;
            SampleSubmission(buffer, serial);
            MetalNative.SendVoid(buffer, SelRelease);
        }
    }

    private void SampleSubmission(nint buffer, ulong serial)
    {
        if (DsTrace.Gpu && _dsLateCount != 0)
        {
            DsRecheckLate();
        }

        var status = MetalV2Native.SendValue(buffer, MetalNative.Selector("status"));
        if (PassProfileEnabled && status != CommandBufferStatusCompleted && status != CommandBufferStatusError)
        {
            // Its serial is signalled from inside it, so it is all but finished; the pass
            // profile needs its timestamps, which are readable only once it has completed.
            MetalNative.SendVoid(buffer, SelWaitUntilCompleted);
            status = MetalV2Native.SendValue(buffer, MetalNative.Selector("status"));
        }
        if (status == CommandBufferStatusError)
        {
            Interlocked.Increment(ref _failedSubmissions);
            var error = MetalNative.Send(buffer, MetalNative.Selector("error"));
            MetalV2Diagnostics.WarnOnce(
                "submission-failed",
                $"A Metal submission failed and will never signal its serial: serial={serial} " +
                $"{MetalNative.DescribeError(error)}");
            PassProfileResolve(serial, false, 0);
            return;
        }

        if (status != CommandBufferStatusCompleted)
        {
            _unsampledSubmissions++;
            PassProfileResolve(serial, false, 0);
            if (DsTrace.Gpu && DsTrace.Armed)
            {
                DsKeepLate(buffer, serial);
            }

            return;
        }

        // The GPU's own execution time, which separates "the GPU is saturated" from
        // "the completion took a long time to reach this thread", and the driver's
        // scheduling gap before it.
        var gpuStart = MetalNative.SendDouble(buffer, SelGpuStartTime);
        var gpuEnd = MetalNative.SendDouble(buffer, SelGpuEndTime);
        if (gpuEnd > gpuStart)
        {
            _gpuBusySeconds += gpuEnd - gpuStart;
            _gpuBusySamples++;
        }

        PassProfileResolve(serial, true, Math.Max(gpuEnd - gpuStart, 0));

        var kernelStart = MetalNative.SendDouble(buffer, SelKernelStartTime);
        var kernelEnd = MetalNative.SendDouble(buffer, SelKernelEndTime);
        if (DsTrace.Gpu && DsTrace.Armed)
        {
            DsEmitGpuTimes(serial, kernelStart, kernelEnd, gpuStart, gpuEnd, late: 0);
        }

        if (kernelEnd > kernelStart)
        {
            _kernelSeconds += kernelEnd - kernelStart;
        }

        if (gpuStart > kernelEnd && gpuStart - kernelEnd < 10.0)
        {
            _scheduleGapSeconds += gpuStart - kernelEnd;
        }
    }

    private double _kernelSeconds;
    private double _scheduleGapSeconds;

    // DSTRACE §4.7 MGPU: submissions that had not completed when sampled. Each is retained
    // here and re-checked at every later sample until it completes, then emitted with
    // late=1 and released. Bounded: past the cap a submission is emitted as late=2
    // (never sampled, no times) and not retained. Worker thread only, like the ring above.
    private const int DsLateCapacity = 128;
    private nint[]? _dsLateBuffers;
    private ulong[]? _dsLateSerials;
    private int _dsLateCount;
    private long _dsLateOverflows;

    private void DsKeepLate(nint buffer, ulong serial)
    {
        _dsLateBuffers ??= new nint[DsLateCapacity];
        _dsLateSerials ??= new ulong[DsLateCapacity];
        if (_dsLateCount == DsLateCapacity)
        {
            if (_dsLateOverflows++ == 0)
            {
                Console.Error.WriteLine(
                    $"[DSTRACE][WARN] MGPU late list full ({DsLateCapacity}); further unfinished submissions are emitted as late=2.");
            }

            DsTrace.Emit(DsEv.MGPU, serial, 0, 0, 0, 0, 2);
            return;
        }

        MetalNative.SendVoid(buffer, SelRetain);
        _dsLateBuffers[_dsLateCount] = buffer;
        _dsLateSerials[_dsLateCount] = serial;
        _dsLateCount++;
    }

    private void DsRecheckLate()
    {
        var buffers = _dsLateBuffers!;
        var serials = _dsLateSerials!;
        for (var index = _dsLateCount - 1; index >= 0; index--)
        {
            var buffer = buffers[index];
            var status = MetalV2Native.SendValue(buffer, MetalNative.Selector("status"));
            if (status != CommandBufferStatusCompleted && status != CommandBufferStatusError)
            {
                continue;
            }

            if (status == CommandBufferStatusCompleted)
            {
                DsEmitGpuTimes(
                    serials[index],
                    MetalNative.SendDouble(buffer, SelKernelStartTime),
                    MetalNative.SendDouble(buffer, SelKernelEndTime),
                    MetalNative.SendDouble(buffer, SelGpuStartTime),
                    MetalNative.SendDouble(buffer, SelGpuEndTime),
                    late: 1);
            }
            else
            {
                DsTrace.Emit(DsEv.MGPU, serials[index], 0, 0, 0, 0, 2);
            }

            MetalNative.SendVoid(buffer, SelRelease);
            _dsLateCount--;
            buffers[index] = buffers[_dsLateCount];
            serials[index] = serials[_dsLateCount];
            buffers[_dsLateCount] = 0;
        }
    }

    private void DsReleaseLate()
    {
        for (var index = 0; index < _dsLateCount; index++)
        {
            MetalNative.SendVoid(_dsLateBuffers![index], SelRelease);
            _dsLateBuffers[index] = 0;
        }

        _dsLateCount = 0;
        if (_dsLateOverflows != 0)
        {
            Console.Error.WriteLine($"[DSTRACE] MGPU late list overflowed {_dsLateOverflows} times.");
        }
    }

    // Metal reports these in seconds on the host media clock; the record carries ns.
    private static void DsEmitGpuTimes(
        ulong serial, double kernelStart, double kernelEnd, double gpuStart, double gpuEnd, ulong late)
    {
        static ulong Ns(double seconds) => seconds > 0 ? (ulong)(seconds * 1e9) : 0;
        DsTrace.Emit(DsEv.MGPU, serial, Ns(kernelStart), Ns(kernelEnd), Ns(gpuStart), Ns(gpuEnd), late);
    }

    private void ReleaseInFlightSubmissions()
    {
        if (DsTrace.Gpu)
        {
            DsReleaseLate();
        }

        while (_inFlightCount > 0)
        {
            var buffer = _inFlightBuffers[_inFlightHead];
            _inFlightBuffers[_inFlightHead] = 0;
            _inFlightHead = (_inFlightHead + 1) % MaxInFlightSubmissions;
            _inFlightCount--;
            if (buffer != 0)
            {
                MetalNative.SendVoid(buffer, SelRelease);
            }
        }
    }

    /// <summary>Commits the current command buffer and waits for the GPU to finish it.</summary>
    public void FlushAndWaitForGpu() => FlushAndWaitForGpu(MetalWaitReason.Other);

    /// <summary>
    /// Commits the current command buffer and waits for the GPU to finish it. The
    /// reason is counted, because the only way to remove a host wait is to know which
    /// packet asked for it.
    /// </summary>
    public void FlushAndWaitForGpu(MetalWaitReason reason)
    {
        // Charged to a phase of its own, so the phase that asked (a resource walk reading
        // bytes the GPU wrote, say) does not report the GPU's time as its own work.
        using var waitProfile = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(
            SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.CommandGpuWait);
        var started = System.Diagnostics.Stopwatch.GetTimestamp();
        var serial = FlushCommandBuffer(MetalFlushCause.WaitGpu);
        var waited = _previousCommandBuffer;
        var committedAt = System.Diagnostics.Stopwatch.GetTimestamp();
        if (waited != 0 && WaitOnCommandBuffer)
        {
            // Wait on the command buffer the driver knows about rather than on the
            // shared event it signals. Measured on this host, a committed submission
            // sat 160 to 250 ms between -commit and the driver scheduling it
            // (`commit_to_kernel_ms`) while the GPU then ran it in ten microseconds:
            // nothing on our queue was pushing the submission out. Asking the driver
            // to wait for that command buffer makes it flush and schedule it now.
            MetalNative.SendVoid(waited, SelWaitUntilCompleted);
        }

        var profile = _completions.WaitFor(serial);
        ReportWait(serial, started, committedAt, waited, _previousCommitTime, in profile, reason, partial: false);
        _guestMemory.NoteCompleted(serial);
        _gpuWrites.Clear(serial);
        RetirePendingCopies(serial);
        _cpOverlay.Retire(serial);
        _labelOverlay.Retire(serial);
        InvalidateWordLines();
    }

    /// <summary>
    /// Waits for one submission that is already committed, leaving what is being
    /// recorded open. A command-processor read of bytes an earlier submission wrote
    /// needs that submission and nothing else, and on a tile-based GPU committing the
    /// open pass instead would store and reload the whole framebuffer.
    /// </summary>
    private void WaitForCommittedSerial(ulong serial, MetalWaitReason reason)
    {
        if (serial > _previousSerial)
        {
            // Nothing has committed this serial, so nothing will ever signal it: waiting
            // would park the worker on work only the worker can submit. Commit it first.
            MetalV2Diagnostics.WarnOnce("wait-uncommitted",
                $"A wait targeted a submission that was not committed yet; committing it first: serial={serial} " +
                $"committed={_previousSerial} recording={_submissionSerial} reason={reason}.");
            FlushCommandBuffer(MetalFlushCause.WaitUncommitted);
        }

        using var waitProfile = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(
            SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.CommandGpuWait);
        var started = System.Diagnostics.Stopwatch.GetTimestamp();
        var profile = _completions.WaitFor(serial);
        if (DsTrace.Gpu && DsTrace.Armed)
        {
            // DSTRACE §4.7 RRW: a partial wait for one committed submission. The read
            // address is known only for ReadRange (WaitForGuestBytes recorded it).
            DsTrace.Emit(DsEv.RRW, serial, _previousSerial, CompletedSerial,
                (ulong)DsTrace.TicksToNs(System.Diagnostics.Stopwatch.GetTimestamp() - started),
                (ulong)reason, reason == MetalWaitReason.ReadRange ? _lastReadAddress : 0);
        }

        // The buffer the wait was for, so the report can say what the GPU spent on it.
        var buffer = InFlightBufferFor(serial, out var commitTime);
        ReportWait(serial, started, started, buffer, commitTime, in profile, reason, partial: true);
        _guestMemory.NoteCompleted(serial);
        var completed = CompletedSerial;
        _gpuWrites.Retire(completed);
        RetirePendingCopies(completed);
        _cpOverlay.Retire(completed);
        _labelOverlay.Retire(completed);
        InvalidateWordLines();
        Interlocked.Increment(ref _partialWaits);
    }

    private void ReportWait(
        ulong serial,
        long started,
        long committedAt,
        nint waited,
        double commitTime,
        in MetalCompletionWait profile,
        MetalWaitReason reason,
        bool partial)
    {
        var finished = System.Diagnostics.Stopwatch.GetTimestamp();
        Interlocked.Increment(ref _gpuWaits);
        Interlocked.Add(ref _gpuWaitTicks, finished - started);
        _waitsByReason[(int)reason]++;
        _waitTicksByReason[(int)reason] += finished - started;
        var frequency = System.Diagnostics.Stopwatch.Frequency;
        if (finished - started <= frequency / 50 || _slowWaitReports >= 4)
        {
            return;
        }

        // A slow host wait: separate the GPU's own execution from the scheduling
        // latency before it and from the time the completion took to reach us.
        _slowWaitReports++;
        var now = CACurrentMediaTime();
        var line = new System.Text.StringBuilder(512);
        line.Append($"[METAL-V2][SLOW-WAIT] serial={serial} reason={reason} partial={(partial ? 1 : 0)} ");
        if (reason == MetalWaitReason.ReadRange)
        {
            // What was read, and the write it waited for. Another wait reason has no read of
            // its own, and these fields would be left over from the last one that did.
            line.Append($"read=0x{_lastReadAddress:X16}+0x{_lastReadSize:X} ")
                .Append($"range=0x{_gpuWrites.LastMatch.Start:X16}..0x{_gpuWrites.LastMatch.End:X16} ")
                .Append($"range_from={_gpuWrites.LastMatch.Origin} range_serial={_gpuWrites.LastMatch.Serial} ");
        }

        line.Append($"total_ms={(finished - started) * 1000.0 / frequency:F1} ")
            .Append($"flush_ms={(committedAt - started) * 1000.0 / frequency:F1} ")
            .Append($"signal_ms={profile.SignalTicks * 1000.0 / frequency:F1} ")
            .Append($"drain_ms={profile.DrainTicks * 1000.0 / frequency:F1} ");
        if (waited == 0)
        {
            // The ring no longer holds the buffer, so there is no timing to read; zero
            // timestamps against a media-clock commit time would print as hours.
            line.Append("commit_to_kernel_ms=n/a kernel_to_gpu_ms=n/a gpu_exec_ms=n/a gpu_end_age_ms=n/a ")
                .Append("gpu_start_age_ms=n/a kernel_ms=n/a kernel_end_age_ms=n/a");
        }
        else
        {
            var gpuStart = MetalNative.SendDouble(waited, SelGpuStartTime);
            var gpuEnd = MetalNative.SendDouble(waited, SelGpuEndTime);
            var kernelStart = MetalNative.SendDouble(waited, SelKernelStartTime);
            var kernelEnd = MetalNative.SendDouble(waited, SelKernelEndTime);
            // commit → the driver starting on it → the GPU starting on it: a long
            // commit-to-kernel gap is host or driver scheduling, not GPU work.
            line.Append($"commit_to_kernel_ms={(kernelStart - commitTime) * 1000.0:F1} ")
                .Append($"kernel_to_gpu_ms={(gpuStart - kernelEnd) * 1000.0:F1} ")
                .Append($"gpu_exec_ms={(gpuEnd - gpuStart) * 1000.0:F2} gpu_end_age_ms={(now - gpuEnd) * 1000.0:F1} ")
                .Append($"gpu_start_age_ms={(now - gpuStart) * 1000.0:F1} ")
                .Append($"kernel_ms={(kernelEnd - kernelStart) * 1000.0:F2} ")
                .Append($"kernel_end_age_ms={(now - kernelEnd) * 1000.0:F1}");
        }

        line.Append($" last={LastWorkDescription}");
        if (MetalV2Diagnostics.WaitTrace)
        {
            line.Append($" reader=0x{SharpEmu.ShaderCompiler.Resources.MaterializingShader.Hash:X16}");
        }

        Console.Error.WriteLine(line.ToString());
    }

    /// <summary>Counts the time the worker spent with nothing runnable.</summary>
    public void NoteIdle(long ticks) => Interlocked.Add(ref _idleTicks, ticks);

    // A queue that has been idle schedules its next submission late. Measured on this
    // host: titles that commit two thousand buffers per two seconds see their
    // submissions start within a millisecond of the commit, while titles that commit
    // a hundred wait three hundred milliseconds and more for the driver to pick one
    // up (commit_to_kernel_ms), with the GPU running them in microseconds and only
    // one guest window resident. The presenter's own queue is busy throughout, so it
    // is the guest queue that goes cold, not the device.
    //
    // The worker commits an empty command buffer while it has nothing else to do, so
    // the queue it will submit the next frame on is never cold. The buffer carries no
    // work, no event signal and no completion handler.
    //
    // Off by default: measured on Octopath it fired only thirteen times per two
    // seconds, because the worker reaches its idle path at the blocked-retry interval
    // rather than every two milliseconds, and the completion latency did not move.
    // The switch keeps the experiment runnable without carrying it on every title.
    private static readonly bool KeepQueueWarm =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_KEEP_QUEUE_WARM") == "1";

    private const long KeepWarmIntervalMicroseconds = 2000;

    private long _lastCommitTimestamp;
    private long _keepWarmSubmissions;

    /// <summary>
    /// Reads whatever the GPU has finished and tells the command stream, so a wait on
    /// bytes the GPU wrote is retried as soon as the write lands rather than at the
    /// blocked-retry interval. The worker calls it before it parks.
    /// </summary>
    public void PollGpuCompletions() => DrainFinishedSubmissions();

    /// <summary>Commits an empty command buffer when the queue has gone quiet.</summary>
    public void KeepQueueWarmIfIdle()
    {
        if (!KeepQueueWarm || _commandBuffer != 0)
        {
            return;
        }

        var now = System.Diagnostics.Stopwatch.GetTimestamp();
        var interval = KeepWarmIntervalMicroseconds * System.Diagnostics.Stopwatch.Frequency / 1_000_000;
        if (_lastCommitTimestamp != 0 && now - _lastCommitTimestamp < interval)
        {
            return;
        }

        _lastCommitTimestamp = now;
        var buffer = MetalNative.Send(_queue, SelCommandBuffer);
        if (buffer == 0)
        {
            return;
        }

        MetalMemoryTelemetry.CommandBuffers.Add();
        MetalMemoryTelemetry.Commits.Add();
        MetalNative.SendVoid(buffer, SelCommit);
        _keepWarmSubmissions++;
    }

    /// <summary>Recorded work stores into this guest range.</summary>
    public void NoteGuestMemoryWrite(ulong address, ulong size, MetalGpuWriteOrigin origin)
    {
        // Recorded after any pending command-processor bytes there, so it is what a read
        // of them has to see from now on, whenever it lands.
        using (var overlayScope = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.WriteNoteOverlays))
        {
            _cpOverlay.Forget(address, size);
            _labelOverlay.Forget(address, size);
        }

        using (var stagedScope = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.WriteNoteStaged))
        {
            _stagedStorageCopies?.Forget(address, size);
            _stagedRanges?.OnGpuWrite(address, size, origin);
        }

        if (_guard is not null && size != 0)
        {
            // A fill, copy or writeback has just been encoded, so the last transfer label
            // is its own; a shader store names the draw or dispatch that bound it.
            var producer = origin is MetalGpuWriteOrigin.Fill or MetalGpuWriteOrigin.Copy
                    or MetalGpuWriteOrigin.ImageWriteback
                ? $"{MetalGpuWriteOrigins.Names[(int)origin]} [{LastTransferDescription}]"
                : $"{MetalGpuWriteOrigins.Names[(int)origin]} [{LastWorkDescription}]";
            _guard.Capture(address, size, SubmissionSerial, producer);
        }

        if (ConservativeReads)
        {
            _gpuWrites.NoteUnbounded(SubmissionSerial);
            _cpOverlay.Clear();
            _labelOverlay.Clear();
            return;
        }

        using var rangesScope = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.WriteNoteRanges);
        var serial = SubmissionSerial;
        var sequence = _gpuWrites.Note(address, size, serial, origin);
        NotePendingCopyDestination(address, size, origin, serial, sequence);
    }

    /// <summary>
    /// Recorded work stores into guest memory without a known destination. Only the
    /// paths that genuinely cannot name their range use this: it makes every later
    /// read of any address wait for the submission.
    /// </summary>
    public void NoteUnboundedGuestMemoryWrite()
    {
        // The store may reach DCC metadata without re-arming a claim there.
        _imageCache.ForgetAllStandingDccClaims();
        _gpuWrites.NoteUnbounded(SubmissionSerial);
        _cpOverlay.Clear();
        _labelOverlay.Clear();
    }

    void IMetalImageCacheOwner.WaitForGpu() => FlushAndWaitForGpu(MetalWaitReason.ImageCache);

    // The image cache asks per DCC slice per lookup, thousands of times a frame: the completed
    // serial's message send (CompletedSerial) made this 8% of Black Myth: Wukong's GPU worker, so
    // the hint answers first and the send only confirms a range something still covers.
    bool IMetalImageCacheOwner.HasPendingGpuWrite(ulong address, ulong size) =>
        _gpuWrites.MayCover(address, size) && PendingSerialHinted(address, size) != 0;

    // A guest thread waiting for a recorded writeback (MetalImageCache.WritebackFence.cs). The
    // GPU finishes a committed submission without the guest's help, so the one thing the wait
    // needs from the worker is the commit, asked for through the relay; it holds no lock the
    // worker takes, and the worker itself never waits here.
    bool IMetalImageCacheOwner.WaitForSubmissionOffWorker(ulong serial, int timeoutMilliseconds)
    {
        if (CompletedSerial >= serial)
        {
            return true;
        }

        if (_relay.IsGpuQueueThread)
        {
            return false;
        }

        if (serial >= Volatile.Read(ref _submissionSerial))
        {
            RequestCommitThrough(serial);
        }

        return _completions.WaitForSignalled(serial, timeoutMilliseconds);
    }

    bool IMetalImageCacheOwner.TryRunOnWorker(Action work) => !_relay.IsGpuQueueThread && _relay.TryPost(work);

    private ulong _commitRequestedSerial;

    // One relay command per serial however many stores wait for it.
    private void RequestCommitThrough(ulong serial)
    {
        var requested = Volatile.Read(ref _commitRequestedSerial);
        if (requested >= serial || Interlocked.CompareExchange(ref _commitRequestedSerial, serial, requested) != requested)
        {
            return;
        }

        _ = _relay.TryPost(() =>
        {
            if (_commandBuffer != 0 && _submissionSerial <= serial)
            {
                FlushCommandBuffer(MetalFlushCause.WaitUncommitted);
            }
        });
    }

    void IMetalImageCacheOwner.FillGuestRange(ulong address, ulong size, uint value) =>
        FillGuestMemory(address, size, value);

    void IMetalImageCacheOwner.NoteGuestWrite(ulong address, ulong size) =>
        NoteGuestMemoryWrite(address, size, MetalGpuWriteOrigin.ImageWriteback);

    // A fill or copy the host can do itself is one no GPU-owned image covers and no
    // recorded store is still writing. Doing it on the GPU instead is what makes the
    // command processor's own poll of the dword it just wrote wait for a whole
    // submission: measured on Quake, a four-byte fill followed by a WAIT_REG_MEM on
    // it cost 20 to 65 ms of host wait each, 150 times per two seconds.
    //
    // A packet-sized range stays on the host even while a command buffer is being
    // recorded. The command processor runs ahead of the GPU by design and the
    // interpreter already publishes its own label writes straight into guest memory,
    // so a dword of command memory is never what a recorded shader is reading.
    private const ulong MaxHostFillBytes = 64 * 1024;
    private const ulong MaxRecordingHostFillBytes = 4096;

    private static readonly bool HostFillsDisabled =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_HOST_FILLS") == "0";

    private static readonly bool ClearFillsDisabled =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_CLEAR_FILLS") == "0";

    private bool CanWriteGuestOnHost(ulong address, ulong size)
    {
        var limit = _recordedCommands == 0 ? MaxHostFillBytes : MaxRecordingHostFillBytes;
        return !HostFillsDisabled && size <= limit &&
            // The GPU path writes through the window, which is mapped whatever the
            // guest's own protections say; the host path goes through guest memory and
            // would quietly write only the mapped prefix.
            _memory.CanRead(address, size) &&
            !_imageCache.HasGpuModifiedImageBytes(address, size) &&
            _gpuWrites.PendingSerial(address, size, CompletedSerial) == 0;
    }

    private bool TryFillGuestMemoryOnHost(ulong address, ulong size, uint value)
    {
        if (!CanWriteGuestOnHost(address, size))
        {
            return false;
        }

        WriteGuestPattern(address, size, value);
        _cpOverlay.Forget(address, size);
        _labelOverlay.Forget(address, size);
        _imageCache.InvalidateMemory(address, size);
        _stagedStorageCopies?.Forget(address, size);
        _stagedRanges?.OnHostWrite(address, size);
        Interlocked.Increment(ref _hostFills);
        return true;
    }

    private long _hostFills;
    private long _gpuFills;
    private long _absorbedFills;
    private long _hostCopies;

    /// <summary>Fills a guest range with a repeating dword through its window.</summary>
    /// <remarks>
    /// <paramref name="allowHost"/> false leaves out the host fast path, which writes the
    /// bytes as parsed: a command-processor fill has to land in record order with labels.
    /// </remarks>
    public void FillGuestMemory(ulong address, ulong size, uint value, bool allowHost = true)
    {
        if (size == 0)
        {
            return;
        }

        if (allowHost && TryFillGuestMemoryOnHost(address, size, value))
        {
            return;
        }

        // A fill whose range is exactly one cached image is that image being cleared.
        // Writing its guest bytes instead costs the writeback of whatever the GPU had
        // put there, a retile of the fill and then a detile back into the texture on
        // the next bind — measured on Quake's menu as two of each per frame.
        if (!ClearFillsDisabled && _imageCache.TryClearImageFromBuffer(address, size, value))
        {
            Interlocked.Increment(ref _absorbedFills);
            return;
        }

        var range = _guestMemory.ResolveWritable(address, size);
        if (!range.IsValid)
        {
            // Not backed: the managed path writes it, which is what a non-backed range
            // means on this backend (device-local allocations are all backed).
            if (!allowHost)
            {
                MetalV2Diagnostics.WarnOnce("cp-fill-without-window",
                    $"A command-processor fill has no window and is written as parsed: address=0x{address:X16} size=0x{size:X}.");
            }

            WriteGuestPattern(address, size, value);
            return;
        }

        // A label held for the commit over these bytes lands first, so the fill overwrites it in stream order.
        OrderAfterDeferredCopies(address, size);
        var context = ((IMetalImageCacheOwner)this).BeginTransfer(MetalTransferCause.GuestFill);
        var encoder = MetalNative.Send(context.CommandBuffer, SelBlitEncoder);
        MetalMemoryTelemetry.BlitEncoders.Add();
        if (encoder == 0)
        {
            throw GpuFatal.Fatal("A Metal blit encoder could not be created for a guest fill.");
        }

        try
        {
            var bytes = (byte)value;
            if (value == bytes * 0x01010101u)
            {
                MetalNative.SendFillBuffer(
                    encoder,
                    SelFillBuffer,
                    range.Buffer,
                    new NsRange { Location = (nuint)range.Offset, Length = (nuint)range.Size },
                    bytes);
            }
            else
            {
                // A non-uniform dword needs a pattern source; one ring page is enough
                // and the blit repeats it across the destination.
                var pattern = _guestMemory.Allocate(Math.Min(size, 64 * 1024), 256, out var span);
                var words = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, uint>(span);
                words.Fill(value);
                for (var written = 0UL; written < size;)
                {
                    var chunk = Math.Min(pattern.Size, size - written);
                    MetalNative.SendCopyBuffer(
                        encoder,
                        SelCopyBuffer,
                        pattern.Buffer,
                        (nuint)pattern.Offset,
                        range.Buffer,
                        (nuint)(range.Offset + written),
                        (nuint)chunk);
                    written += chunk;
                }
            }
        }
        finally
        {
            MetalNative.SendVoid(encoder, SelEndEncoding);
        }

        if (((IMetalSubmissionContext)this).WantsEncodedLabels)
        {
            ((IMetalSubmissionContext)this).ProfileEncoded($"fill bytes=0x{size:X}");
        }
        _imageCache.InvalidateMemoryFromGpu(address, size);
        NoteGuestMemoryWrite(address, size, MetalGpuWriteOrigin.Fill);
        Interlocked.Increment(ref _gpuFills);
    }

    private void WriteGuestPattern(ulong address, ulong size, uint value)
    {
        if (_guard is null)
        {
            WriteGuestPatternCore(address, size, value);
            return;
        }

        var watch = _guard.Begin(address, size, $"host-fill value=0x{value:X8}");
        try
        {
            WriteGuestPatternCore(address, size, value);
        }
        finally
        {
            _guard.End(watch);
        }
    }

    private void WriteGuestPatternCore(ulong address, ulong size, uint value)
    {
        Span<byte> chunk = stackalloc byte[4096];
        var words = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, uint>(chunk);
        words.Fill(value);
        for (var written = 0UL; written < size;)
        {
            var count = (int)Math.Min((ulong)chunk.Length, size - written);
            if (!_memory.TryWrite(address + written, chunk[..count]))
            {
                break;
            }

            written += (ulong)count;
        }
    }

    /// <summary>Copies guest bytes on the GPU through the windows.</summary>
    /// <remarks>
    /// <paramref name="allowHost"/> false leaves out the host fast path, which copies the
    /// bytes as parsed: a command-processor copy has to land in record order with labels.
    /// </remarks>
    public void CopyGuestMemory(ulong destination, ulong source, ulong size, bool allowHost = true)
    {
        if (size == 0)
        {
            return;
        }

        var target = _guestMemory.ResolveWritable(destination, size);
        var origin = _guestMemory.Resolve(source, size);
        // A source too large to stage resolves to a prefix, which a blit of the whole
        // size would overrun.
        if (!target.IsValid || !origin.IsValid || origin.Size < size)
        {
            if (!allowHost)
            {
                MetalV2Diagnostics.WarnOnce("cp-copy-without-window",
                    $"A command-processor copy has no window and is copied as parsed: destination=0x{destination:X16} " +
                    $"source=0x{source:X16} size=0x{size:X}.");
            }

            CopyGuestBytesOnHost(destination, source, size);
            _stagedStorageCopies?.Forget(destination, size);
            _stagedRanges?.OnHostWrite(destination, size);
            return;
        }

        // A packet-sized copy the host can do itself costs nothing and records no GPU
        // write, so the poll that usually follows it does not wait for a submission.
        if (allowHost && CanWriteGuestOnHost(destination, size))
        {
            CopyGuestBytesOnHost(destination, source, size);
            _imageCache.InvalidateMemory(destination, size);
            _stagedStorageCopies?.Forget(destination, size);
            _stagedRanges?.OnHostWrite(destination, size);
            Interlocked.Increment(ref _hostCopies);
            return;
        }

        FlushGpuWrites(source, size, MetalWritebackCause.GuestCopy);
        // A label held for the commit over the source or the destination lands first, as a boundary used to land it.
        OrderAfterDeferredCopies(source, size);
        OrderAfterDeferredCopies(destination, size);
        var context = ((IMetalImageCacheOwner)this).BeginTransfer(MetalTransferCause.GuestCopy);
        var encoder = MetalNative.Send(context.CommandBuffer, SelBlitEncoder);
        MetalMemoryTelemetry.BlitEncoders.Add();
        if (encoder == 0)
        {
            throw GpuFatal.Fatal("A Metal blit encoder could not be created for a guest copy.");
        }

        try
        {
            MetalNative.SendCopyBuffer(
                encoder,
                SelCopyBuffer,
                origin.Buffer,
                (nuint)origin.Offset,
                target.Buffer,
                (nuint)target.Offset,
                (nuint)size);
        }
        finally
        {
            MetalNative.SendVoid(encoder, SelEndEncoding);
        }

        if (context.WantsEncodedLabels)
        {
            context.ProfileEncoded($"copy bytes=0x{size:X}");
        }
        _imageCache.InvalidateMemoryFromGpu(destination, size);
        NoteGuestMemoryWrite(destination, size, MetalGpuWriteOrigin.Copy);
    }

    private void CopyGuestBytesOnHost(ulong destination, ulong source, ulong size)
    {
        _cpOverlay.Forget(destination, size);
        _labelOverlay.Forget(destination, size);
        if (_guard is null)
        {
            CopyGuestBytesOnHostCore(destination, source, size);
            return;
        }

        var watch = _guard.Begin(destination, size, $"host-copy from=0x{source:X16}");
        try
        {
            CopyGuestBytesOnHostCore(destination, source, size);
        }
        finally
        {
            _guard.End(watch);
        }
    }

    private void CopyGuestBytesOnHostCore(ulong destination, ulong source, ulong size)
    {
        var buffer = System.Buffers.ArrayPool<byte>.Shared.Rent((int)Math.Min(size, 64 * 1024));
        try
        {
            for (var copied = 0UL; copied < size;)
            {
                var count = (int)Math.Min((ulong)buffer.Length, size - copied);
                var span = buffer.AsSpan(0, count);
                if (!TryReadGuest(source + copied, span) || !_memory.TryWrite(destination + copied, span))
                {
                    break;
                }

                copied += (ulong)count;
            }
        }
        finally
        {
            System.Buffers.ArrayPool<byte>.Shared.Return(buffer);
        }
    }

    /// <summary>Publishes every GPU-owned image byte in the range into guest memory.</summary>
    public void FlushGpuWrites(ulong address, ulong size, MetalWritebackCause cause) =>
        _imageCache.FlushGpuWrites(address, size, cause);

    public void RunPendingOperations() => _relay.RunPendingCommands();

    public void SetDebugInformation(
        RecordedOperation operation,
        ulong submitId,
        uint argument0,
        uint argument1,
        uint argument2,
        uint argument3,
        ulong argument4)
    {
        if (!MetalV2Diagnostics.Verbose)
        {
            return;
        }

        MetalV2Diagnostics.Trace(
            $"{operation} submit={submitId} args=({argument0},{argument1},{argument2},{argument3}) address=0x{argument4:X16}");
    }

    public bool TryReadGuest(ulong address, Span<byte> destination)
    {
        if (destination.IsEmpty)
        {
            return true;
        }

        if (_cpOverlay.Count != 0)
        {
            // Bytes a command-processor write still owes are read as written, the way the
            // command processor reads its own writes: no wait for the GPU, which for a
            // write still waiting for the open pass to end would be a wait on itself.
            switch (OverlayCoverage(address, (ulong)destination.Length))
            {
                case MetalOverlayCoverage.Full:
                    _cpOverlay.Apply(address, destination);
                    Interlocked.Increment(ref _overlayReads);
                    return true;
                case MetalOverlayCoverage.Partial:
                    if (!TryReadGuestCore(address, destination))
                    {
                        return false;
                    }

                    _cpOverlay.Apply(address, destination);
                    Interlocked.Increment(ref _overlayReads);
                    return true;
            }
        }

        return TryReadGuestCore(address, destination);
    }

    private long _overlayReads;
    private Func<ulong>? _completedSerialReader;

    /// <summary>Parse-time reads answered from pending command-processor bytes.</summary>
    public long TakeOverlayReads() => Interlocked.Exchange(ref _overlayReads, 0);

    // How much of a range pending command-processor bytes answer for. An image the GPU has
    // drawn into since owns its bytes, and only its writeback can say what they are.
    private MetalOverlayCoverage OverlayCoverage(ulong address, ulong size)
    {
        var coverage = OverlayCoverageHinted(address, size);
        if (coverage == MetalOverlayCoverage.Full &&
            _imageCache.MayOwn(address, size) && _imageCache.HasGpuModifiedImageBytes(address, size))
        {
            return MetalOverlayCoverage.Partial;
        }

        return coverage;
    }

    private bool TryReadGuestCore(ulong address, Span<byte> destination)
    {
        // Guest memory is the GPU's memory, so a read only has to wait for work that
        // may still be writing these bytes: an image that owns them, or a recorded
        // store into them. Without that test every command-memory read would stall.
        var size = (ulong)destination.Length;
        // The command stream reads a packet out of guest memory for every packet it
        // runs, and almost none of them land anywhere the GPU has written. Two
        // comparisons answer that, and skipping the full test keeps a message send
        // and two range scans off the per-packet path.
        if (!_gpuWrites.MayCover(address, size) && !_imageCache.MayOwn(address, size))
        {
            return _memory.TryRead(address, destination);
        }

        if (TryReadThroughPendingCopy(address, destination))
        {
            return true;
        }

        WaitForGuestBytes(address, size);
        return _memory.TryRead(address, destination);
    }

    public void SynchronizeShaderCode(ulong address, ulong size)
    {
        if (size != 0 && (_gpuWrites.MayCover(address, size) || _imageCache.MayOwn(address, size)))
        {
            WaitForGuestBytes(address, size);
        }
    }

    // Lands every recorded write into the range: an image's writeback, or a store or copy.
    private void WaitForGuestBytes(ulong address, ulong size)
    {
        _lastReadAddress = address;
        _lastReadSize = size;
        if (_imageCache.HasGpuModifiedImageBytes(address, size))
        {
            // The writeback is recorded into the buffer being built, so this one has
            // to be committed before the bytes can be read.
            FlushGpuWrites(address, size, MetalWritebackCause.HostRead);
            FlushAndWaitForGpu(MetalWaitReason.ReadImage);
            return;
        }

        var pending = _gpuWrites.PendingSerial(address, size, CompletedSerial);
        if (pending != 0)
        {
            if (size <= SmallReadWaitBytes)
            {
                ReportSmallReadWaitSite(address, size);
            }

            if (_commandBuffer != 0 && pending >= _submissionSerial)
            {
                FlushAndWaitForGpu(MetalWaitReason.ReadRange);
            }
            else
            {
                WaitForCommittedSerial(pending, MetalWaitReason.ReadRange);
            }
        }
    }

    // A read this small that waits for the GPU is a command-processor operand (an indirect
    // draw's arguments, a count, a predicate) rather than data; each distinct reader is
    // named once, with its stack, so the stall can be traced to the packet that asked.
    private const ulong SmallReadWaitBytes = 0x14;
    private const int MaxSmallReadWaitSites = 64;
    private readonly HashSet<string> _smallReadWaitSites = new(StringComparer.Ordinal);

    private void ReportSmallReadWaitSite(ulong address, ulong size)
    {
        if (_smallReadWaitSites.Count >= MaxSmallReadWaitSites)
        {
            return;
        }

        // The first frame past this host's and the interpreter's read helpers says who read.
        var site = "unknown";
        foreach (var frame in new System.Diagnostics.StackTrace(1, false).GetFrames())
        {
            var method = frame.GetMethod();
            var type = method?.DeclaringType;
            if (method is null || type is null || type == typeof(MetalRenderHost) ||
                method.Name.StartsWith("Read", StringComparison.Ordinal) ||
                method.Name.StartsWith("TryRead", StringComparison.Ordinal))
            {
                continue;
            }

            site = $"{type.FullName}.{method.Name}";
            break;
        }

        if (!_smallReadWaitSites.Add(site))
        {
            return;
        }

        Console.Error.WriteLine(
            $"[METAL-V2][SMALL-READ-WAIT] read=0x{address:X16}+0x{size:X} site={site}{Environment.NewLine}" +
            Environment.StackTrace);
    }

    /// <summary>
    /// Reads the bytes a wait packet polls without waiting for the GPU to write them.
    /// </summary>
    /// <remarks>
    /// An end-of-pipe label the GPU writes is a recorded write like any other, so an
    /// ordinary read of it waits for the submission: on Superliminal that was eight hundred
    /// waits and a second of worker time per two-second window, because the command
    /// processor polls the labels it just queued. The hardware's command processor does not
    /// wait either — it reads memory, and the value arrives when the GPU gets there. What
    /// the host must do is commit the work that will write the bytes, so that the poll has
    /// something to wait for; the compare then suspends the submission, and the completion
    /// of that submission wakes it.
    /// </remarks>
    // SHARPEMU_METAL_V2_POLL_WITHOUT_WAIT=0 makes a poll an ordinary read again, which
    // waits for the GPU on the worker thread. For telling the two apart when a title
    // misbehaves.
    private static readonly bool PollWithoutWaiting =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_POLL_WITHOUT_WAIT") != "0";

    // SHARPEMU_METAL_V2_POLL_TRACE=1 names every address a wait packet polls and what it
    // found there, which is the other half of the end-of-pipe trace.
    private static readonly bool PollTrace =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_POLL_TRACE") == "1";

    public bool TryReadGuestWaitOperand(ulong address, Span<byte> destination, bool commitPendingWrite)
    {
        if (destination.IsEmpty)
        {
            return true;
        }

        if (!PollWithoutWaiting)
        {
            return TryReadGuest(address, destination);
        }

        if (_cpOverlay.Count != 0 && OverlayCoverage(address, (ulong)destination.Length) == MetalOverlayCoverage.Full)
        {
            // The poll reads bytes this host has yet to land as written: nothing to commit.
            _cpOverlay.Apply(address, destination);
            Interlocked.Increment(ref _overlayReads);
            return true;
        }

        if (PollTrace)
        {
            var read = _memory.TryRead(address, destination);
            var polled = destination.Length >= sizeof(ulong)
                ? System.Buffers.Binary.BinaryPrimitives.ReadUInt64LittleEndian(destination)
                : System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(destination);
            Console.Error.WriteLine(
                $"[METAL-V2][POLL] address=0x{address:X16} bytes={destination.Length} value=0x{polled:X} " +
                $"pending={_gpuWrites.PendingSerial(address, (ulong)destination.Length, CompletedSerial)} " +
                $"submission={_submissionSerial} completed={CompletedSerial} read={read}");
        }

        var size = (ulong)destination.Length;
        if (_imageCache.MayOwn(address, size) && _imageCache.HasGpuModifiedImageBytes(address, size))
        {
            // An image owns these bytes, and its writeback has to be read back through the
            // ordinary path before anything can be said about them.
            return TryReadGuest(address, destination);
        }

        if (commitPendingWrite && _commandBuffer != 0 &&
            ((_gpuWrites.MayCover(address, size) &&
              _gpuWrites.PendingSerial(address, size, CompletedSerial) >= _submissionSerial) ||
             (_lazyLabelRanges.Count != 0 && CoversLazyLabel(address, size))))
        {
            // The write is still in the buffer being recorded: commit it, do not wait. The
            // commit is per poll, which is the cost of this design — deferring it to the
            // worker's next park was measured slower, because the park follows immediately
            // anyway and the extra scheduler turn only adds latency to the wake.
            Interlocked.Increment(ref _pollFlushes);
            FlushCommandBuffer(MetalFlushCause.Poll);
        }

        if (!_memory.TryRead(address, destination))
        {
            return false;
        }

        // Part of the range may still be owed by a command-processor write.
        ApplyPendingCommandProcessorBytes(address, destination);
        return true;
    }

    /// <summary>
    /// A command-processor operand (a predicate, a COND_EXEC or COND_WRITE compare, a
    /// branch compare, an index-attributes record) that recorded work still owes: the work
    /// is committed and a wake is queued for its completion, and the interpreter parks the
    /// submission on the bytes instead of this thread waiting for the GPU.
    /// </summary>
    /// <remarks>
    /// Bytes an image owns are not answered here: their writeback is a read-back the
    /// ordinary read path performs, and it is the only path that knows how.
    /// </remarks>
    public bool CommitPendingOperandWrite(ulong address, ulong size)
    {
        if (size == 0 || !PollWithoutWaiting)
        {
            return false;
        }

        if (_cpOverlay.Count != 0 && OverlayCoverage(address, size) == MetalOverlayCoverage.Full)
        {
            return false;
        }

        if (_imageCache.MayOwn(address, size) && _imageCache.HasGpuModifiedImageBytes(address, size))
        {
            return false;
        }

        var serial = _gpuWrites.MayCover(address, size) ? _gpuWrites.PendingSerial(address, size, CompletedSerial) : 0;
        var lazy = _lazyLabelRanges.Count != 0 && CoversLazyLabel(address, size);
        if (serial == 0 && !lazy)
        {
            return false;
        }

        // A lazy label rides the buffer being recorded, so it lands with that serial.
        var wake = lazy ? Math.Max(serial, _submissionSerial) : serial;
        if (_commandBuffer != 0 && wake >= _submissionSerial)
        {
            Interlocked.Increment(ref _pollFlushes);
            FlushCommandBuffer(MetalFlushCause.Poll);
        }

        QueueOperandWake(wake);
        return true;
    }

    // The completion that lands an operand is the only thing that knows it has; the worker
    // reports completions when it next flushes, which is never while every head is parked.
    private void QueueOperandWake(ulong serial)
    {
        if (serial == 0 || _operandWakeSerial >= serial)
        {
            return;
        }

        _operandWakeSerial = serial;
        _completions.Enqueue(serial, () => _commandStream?.NotifyCompletedGpuTick(serial));
    }

    private ulong _operandWakeSerial;
    private long _pollFlushes;
    private long _scheduledLabelSubmissions;
    private long _transientCommits;

    // What the budget commits found lent, rings and staging apart, since the last report.
    private long _transientCommitRingBytes;
    private long _transientCommitStagingBytes;

    // The shared memory a submission borrows while it is recorded - the upload rings
    // that carry a range no window serves, a staged storage copy or a transient
    // binding, and the staging a transfer detiles or writes back through - returns to
    // its pool only when that command buffer has completed. A buffer that keeps
    // recording therefore holds every ring and slab it touched, and each pool grows by
    // one more. Silent Hill's loading moves a gigabyte a second through the rings and
    // reloads 4K surfaces that alias each other on every use; while every poll
    // committed the buffer, the rings and slabs returned within a few draws, and once
    // polls stopped committing one command buffer took the ring pool from 128 MB to
    // 9.9 GB in eight seconds (310 rings) before the memory limit ended the run.
    // Committing at a budget hands the GPU the work early, which is all a commit is:
    // copies and detiles of this size take the GPU well under a millisecond, and the
    // pools then see their memory back. The budget is a handful of the largest
    // surfaces, because every commit costs the open pass a store and a reload of its
    // attachments. SHARPEMU_METAL_V2_TRANSIENT_COMMIT_MB overrides it, for measuring.
    private const ulong DefaultTransientCommitBytes = 256UL * 1024 * 1024;

    internal ulong TransientCommitBytes { get; set; } =
        ulong.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_TRANSIENT_COMMIT_MB"), out var megabytes) &&
        megabytes != 0
            ? megabytes * 1024 * 1024
            : DefaultTransientCommitBytes;

    /// <summary>The staging pool the transfers draw on; tests lend from it directly.</summary>
    internal MetalTiler? StagingTiler => (_transfer as MetalImageTransfer)?.Tiler;

    /// <summary>The ring and staging memory the pools have lent to the submission being recorded.</summary>
    internal ulong RecordingTransientBytes
    {
        get
        {
            if (_commandBuffer == 0)
            {
                return 0;
            }

            var staging = _transfer is MetalImageTransfer { Tiler: var tiler } && tiler.RecordingSerial == _submissionSerial
                ? tiler.RecordingBytes
                : 0;
            return staging + _guestMemory.RecordingBytes;
        }
    }

    /// <summary>Commits the recording command buffer when the memory lent to it passed the budget.</summary>
    private void CommitIfTransientOverBudget()
    {
        if (_commandBuffer == 0 || RecordingTransientBytes < TransientCommitBytes)
        {
            return;
        }

        Interlocked.Increment(ref _transientCommits);
        var ringBytes = _guestMemory.RecordingBytes;
        Interlocked.Add(ref _transientCommitRingBytes, (long)ringBytes);
        Interlocked.Add(ref _transientCommitStagingBytes, (long)(RecordingTransientBytes - ringBytes));
        var committed = FlushCommandBuffer(PassEndCause.Transfer);
        // A commit hands the memory back only once the GPU has run the buffer, and
        // nothing else stopped the worker from recording the next budget meanwhile. In
        // Silent Hill's first level the GPU ran seconds behind while 4K surfaces were
        // reloaded on every use: 81 budget commits in under eight seconds, one completion,
        // and the staging pool went from 0.2 to 6.5 GB, past any memory limit. So at most
        // two budgets are in flight: before recording a third, the worker waits for the
        // GPU to finish the commit before this one, which keeps the GPU fed with the one
        // just committed.
        var previous = _previousTransientCommitSerial;
        _previousTransientCommitSerial = committed;
        if (previous != 0 && CompletedSerial < previous)
        {
            WaitForCommittedSerial(previous, MetalWaitReason.Other);
        }
    }

    private ulong _previousTransientCommitSerial;

    private string StagedRangeCounters()
    {
        if (_stagedRanges is not { } cache)
        {
            return string.Empty;
        }

        var c = cache.TakeCounters();
        var g = cache.Gauges();
        return $" staged_hits={c.Hits} staged_misses={c.Misses} staged_bytes={c.Bytes} " +
            $"staged_invalid_fault={c.Fault} staged_invalid_precise={c.Precise} staged_invalid_host={c.Host} " +
            $"staged_invalid_gpu={c.Gpu} staged_invalid_unmap={c.Unmap} staged_evictions={c.Evictions} " +
            $"staged_cache_mb={cache.CacheBytes / (1024 * 1024)} staged_entries={g.Entries} staged_watched={g.Watched} " +
            $"staged_dirty={g.Dirty} staged_stale={g.Stale} staged_unwatched_current={g.UnwatchedCurrent} " +
            $"staged_verify_checks={c.VerifyChecks} staged_verify_mismatches={c.VerifyMismatches}";
    }

    /// <summary>Commits the transient budget asked for so rings and slabs could return.</summary>
    public long TakeTransientCommits() => Interlocked.Exchange(ref _transientCommits, 0);

    // SHARPEMU_METAL_V2_SCHEDULE_LABELS=0 leaves a label-carrying submission to the
    // driver's own pace, which is what every title did before labels were honest.
    private static readonly bool ScheduleLabelSubmissions =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_SCHEDULE_LABELS") != "0";

    // How long since the previous commit makes this queue a quiet one.
    private static readonly long QuietQueueTicks = System.Diagnostics.Stopwatch.Frequency / 250;

    // Started by the first nudge: most titles never need one.
    private MetalScheduleNudge? _scheduleNudge;

    /// <summary>Label submissions the driver was asked to schedule at once.</summary>
    public long TakeScheduledLabelSubmissions() => Interlocked.Exchange(ref _scheduledLabelSubmissions, 0);

    /// <summary>Commits a wait packet asked for so the bytes it polls can change.</summary>
    public long TakePollFlushes() => Interlocked.Exchange(ref _pollFlushes, 0);

    // The part of the range that is mapped from its start; an unmapped start is fatal
    // because the executor cannot bind it.
    public ulong ClampMappedSize(ulong address, ulong size)
    {
        using var phaseScope = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.MappedSizeClamp);
        if (address == 0 || size == 0 || size > ulong.MaxValue - address || !_memory.CanRead(address, 1))
        {
            throw Fatal($"The buffer range starts in unmapped memory: address=0x{address:X16} size=0x{size:X16}.");
        }

        if (_memory.CanRead(address, size))
        {
            return size;
        }

        var clamped = 1UL;
        var rejected = size;
        while (rejected - clamped > 1)
        {
            var candidate = clamped + ((rejected - clamped) / 2);
            if (_memory.CanRead(address, candidate))
            {
                clamped = candidate;
            }
            else
            {
                rejected = candidate;
            }
        }

        return clamped;
    }

    public BufferBinding NullBuffer => new((ulong)_guestMemory.NullBuffer, 0);

    public BufferBinding ObtainBuffer(ulong address, ulong size, bool isWritten)
    {
        // A buffer binding sees guest memory, so any image that owns these bytes
        // publishes them first; a written binding also invalidates those images.
        FlushGpuWrites(address, size, MetalWritebackCause.BufferBinding);
        // A read takes the storage path: its window, a guest view, or a staged copy that
        // holds the GPU's writes still in flight in this command buffer - whole, however
        // large, up to the largest window.
        var range = isWritten
            ? _guestMemory.Resolve(address, size, whole: true)
            : ResolveStorageBuffer(address, size, written: false);

        if (!range.IsValid)
        {
            return NullBuffer;
        }

        _slotTrap?.NoteBinding(
            isWritten ? "binding(write)" : "binding(read)",
            address, size, (ulong)range.Buffer, range.Offset, range.GpuAddress);
        if (isWritten)
        {
            _imageCache.InvalidateMemoryFromGpu(address, size);
            NoteGuestMemoryWrite(address, size, MetalGpuWriteOrigin.WrittenBinding);
        }

        return new BufferBinding((ulong)range.Buffer, range.Offset);
    }

    public BufferBinding UploadTransient(ReadOnlySpan<byte> data, uint alignment)
    {
        var range = _guestMemory.Allocate((ulong)data.Length, Math.Max(alignment, 16), out var span);
        data.CopyTo(span);
        return new BufferBinding((ulong)range.Buffer, range.Offset);
    }

    public bool TryGetImageFormatProperties(
        GpuFormat format,
        GpuImageType type,
        GpuImageTiling tiling,
        GpuImageUsage usage,
        GpuImageCreateFlags flags,
        out GpuImageFormatProperties properties)
    {
        _ = type;
        _ = tiling;
        _ = usage;
        _ = flags;
        properties = new GpuImageFormatProperties
        {
            MaxExtent = new GpuExtent3D(MaxFramebufferDimension, MaxFramebufferDimension, 2048),
            MaxMipLevels = ImageDescription.MaxLevels,
            MaxArrayLayers = 2048,
            SampleCounts = GpuSampleCount.Count1Bit | GpuSampleCount.Count2Bit | GpuSampleCount.Count4Bit |
                GpuSampleCount.Count8Bit,
            MaxResourceSize = 1UL << 32,
        };
        return MetalV2Formats.IsSupported(format);
    }

    public (long Draws, double DrawMs, long Pipelines, long ShaderCompilations) ReadAndResetPerfCounters()
    {
        var draws = Interlocked.Exchange(ref _drawCount, 0);
        var ticks = Interlocked.Exchange(ref _drawTicks, 0);
        var pipelines = Interlocked.Exchange(ref _pipelineCreations, 0);
        var compilations = Interlocked.Exchange(ref _shaderCompilations, 0);
        return (draws, ticks * 1000.0 / System.Diagnostics.Stopwatch.Frequency, pipelines, compilations);
    }

    public void CountShaderCompilation() => Interlocked.Increment(ref _shaderCompilations);

    /// <summary>
    /// One periodic line of host counters. The compact form is always printed; the
    /// render-pass breakdown and the completion timings that found the residency-set
    /// stall follow it under SHARPEMU_METAL_V2_TRACE=1.
    /// </summary>
    public void ReportCounters()
    {
        var now = System.Diagnostics.Stopwatch.GetTimestamp();
        if (_statsTimestamp == 0)
        {
            _statsTimestamp = now;
            return;
        }

        if (now - _statsTimestamp < 2 * System.Diagnostics.Stopwatch.Frequency)
        {
            return;
        }

        var seconds = (now - _statsTimestamp) / (double)System.Diagnostics.Stopwatch.Frequency;
        _statsTimestamp = now;
        // Read every submission that has finished since the last report, so the GPU
        // time below covers them all and not only the one that happened to be done.
        DrainFinishedSubmissions();
        var draws = Interlocked.Exchange(ref _drawCount, 0);
        var drawTicks = Interlocked.Exchange(ref _drawTicks, 0);
        var flushes = Interlocked.Exchange(ref _flushes, 0);
        var flushBy = string.Join(' ', Enum.GetValues<MetalFlushCause>()
            .Select(why => $"{why.ToString().ToLowerInvariant()}={Interlocked.Exchange(ref _flushCauses[(int)why], 0)}"));
        var passes = Interlocked.Exchange(ref _renderPasses, 0);
        var gpuBusyMs = _gpuBusySeconds * 1000.0;
        var gpuBusySamples = _gpuBusySamples;
        var kernelMs = _kernelSeconds * 1000.0;
        var scheduleMs = _scheduleGapSeconds * 1000.0;
        var unsampled = _unsampledSubmissions;
        _gpuBusySeconds = 0;
        _gpuBusySamples = 0;
        _kernelSeconds = 0;
        _scheduleGapSeconds = 0;
        _unsampledSubmissions = 0;
        var waits = Interlocked.Exchange(ref _gpuWaits, 0);
        var partial = Interlocked.Exchange(ref _partialWaits, 0);
        var waitTicks = Interlocked.Exchange(ref _gpuWaitTicks, 0);
        var idleTicks = Interlocked.Exchange(ref _idleTicks, 0);
        var pipelines = Interlocked.Exchange(ref _pipelineCreations, 0);
        var pipelineMs = Interlocked.Exchange(ref _pipelineTicks, 0) * 1000.0 /
            System.Diagnostics.Stopwatch.Frequency;
        var compiles = TakeCompilerCounters();
        var libraryWaitTicks = Interlocked.Exchange(ref _libraryWaitTicks, 0);
        var libraryWaitPrewarmTicks = Interlocked.Exchange(ref _libraryWaitPrewarmTicks, 0);
        var scheduler = MetalCompileScheduler.Shared.TakeCounters();
        var failed = Interlocked.Exchange(ref _failedSubmissions, 0);
        var (completions, completionMs) = TakeCompletionLatency();
        var (created, uploads, writebacks, evictions) = _imageCache.Counters;
        var culture = System.Globalization.CultureInfo.InvariantCulture;
        var frequency = System.Diagnostics.Stopwatch.Frequency;
        var (loadsElided, loadElidedBytes) = TakeElidedLoads();
        var (refreshesSkipped, refreshSkippedBytes) = _imageCache.TakeSkippedRefreshes();
        var (pageWatches, pageWatchPeak) = _imageCache.TakePageWatchCounters();
        Console.Error.WriteLine(
            $"[METAL-V2][PERF] seconds={seconds.ToString("F2", culture)} submissions={_submissionsStarted} " +
            // Colour loads a full-screen first draw made dead (MetalPassLoads).
            $"load_elided={loadsElided} load_elided_mb={(loadElidedBytes / (1024.0 * 1024.0)).ToString("F1", culture)} " +
            // Uploads of CPU-written targets a pass cleared or overwrote whole (DeferredRefresh).
            $"refresh_skipped={refreshesSkipped} refresh_skipped_mb={(refreshSkippedBytes / (1024.0 * 1024.0)).ToString("F1", culture)} " +
            // Slices are what the queue actually ran: a submission that suspends is
            // re-queued and re-run, so slices far above guest submissions is retry churn,
            // not the guest sending more work.
            $"guest_submissions={CommandStream.SubmissionsStarted} slices={CommandStream.SlicesRun} " +
            $"ring_follows={SharpEmu.Libs.Gpu.GpuCommands.GpuStreamDiagnostics.RingFollows} ring_suspends={SharpEmu.Libs.Gpu.GpuCommands.GpuStreamDiagnostics.RingSuspends} " +
            (SharpEmu.Libs.Gpu.GpuCommands.GpuStreamDiagnostics.DcbHashEnabled
                ? $"dcbhash_checks={SharpEmu.Libs.Gpu.GpuCommands.GpuStreamDiagnostics.HashChecks} dcbhash_mismatches={SharpEmu.Libs.Gpu.GpuCommands.GpuStreamDiagnostics.HashMismatches} "
                : string.Empty) +
            $"draws={draws} passes={passes} flushes={flushes} flush_by=[{flushBy}] flush_deferred={TakeDeferredStreamFlushes()} " +
            // Page-guard write watches held, and the most one page reached (the guard fails past 127).
            $"page_watches={pageWatches} page_watch_peak={pageWatchPeak} " +
            $"draw_ms={(drawTicks * 1000.0 / frequency).ToString("F1", culture)} " +
            $"gpu_busy_ms={gpuBusyMs.ToString("F1", culture)} " +
            $"gpu_n={gpuBusySamples} gpu_unsampled={unsampled} " +
            $"kernel_ms={kernelMs.ToString("F1", culture)} " +
            $"schedule_ms={scheduleMs.ToString("F1", culture)} " +
            $"completions={completions} completion_ms={completionMs.ToString("F1", culture)} " +
            $"host_waits={waits} partial_waits={partial} " +
            $"host_wait_ms={(waitTicks * 1000.0 / frequency).ToString("F1", culture)} " +
            $"idle_ms={(idleTicks * 1000.0 / frequency).ToString("F0", culture)} pipelines={pipelines} " +
            $"pipeline_ms={pipelineMs.ToString("F1", culture)} " +
            $"library_ms={(Interlocked.Exchange(ref _libraryTicks, 0) * 1000.0 / frequency).ToString("F1", culture)} " +
            $"library_wait_ms={(libraryWaitTicks * 1000.0 / frequency).ToString("F1", culture)} " +
            // What the worker blocked on: a runtime miss's own compile, a compile the prewarm
            // had in the driver for the same source, and of the misses, those that compiled
            // beside prewarm compiles; how many waits and how many it compiled itself.
            $"library_wait_miss_ms={((libraryWaitTicks - libraryWaitPrewarmTicks) * 1000.0 / frequency).ToString("F1", culture)} " +
            $"library_wait_prewarm_ms={(libraryWaitPrewarmTicks * 1000.0 / frequency).ToString("F1", culture)} " +
            $"library_wait_contended_ms={(Interlocked.Exchange(ref _libraryWaitContendedTicks, 0) * 1000.0 / frequency).ToString("F1", culture)} " +
            $"library_waits={Interlocked.Exchange(ref _libraryWaits, 0)} library_inline={Interlocked.Exchange(ref _libraryInlineCompiles, 0)} " +
            // Prewarm threads held back for the game, compiles the starvation limit let
            // through, and prewarm compiles in the driver now.
            $"prewarm_held_ms={scheduler.HeldMilliseconds.ToString("F0", culture)} prewarm_trickle={scheduler.Trickled} " +
            $"prewarm_inflight={MetalCompileScheduler.Shared.PrewarmInFlight} " +
            $"compiles={compiles.Started}/{compiles.Finished} compile_waits={compiles.Deduped} " +
            $"compile_ms={compiles.Milliseconds.ToString("F1", culture)} compiling={PipelinesCompiling} " +
            // Draws left out while their pipeline compiled, pipelines the worker waited for
            // (a miss it compiled itself, or a draw that could not be left out), and compiles
            // queued or running on the compiler threads now.
            $"draws_skipped_compiling={Interlocked.Exchange(ref _drawsSkippedCompiling, 0)} " +
            $"pipeline_waits_blocking={Interlocked.Exchange(ref _pipelineWaitsBlocking, 0)} " +
            $"async_compiles_inflight={PipelinesCompiling} " +
            // Modules and pipeline states resident against made, and what the bound released.
            $"{DescribeShaderCache()} " +
            $"images={_imageCache.ImageCount} image_mb={_imageCache.TotalUsedMemory / (1024 * 1024)} " +
            // Shared (CPU-visible) memory the transfer staging and the upload rings hold.
            $"{DescribeStagingWindow()} " +
            $"sampled_dcc_clears={_imageCache.TakeSampledDccClears()} " +
            $"dcc_repeats_skipped={_imageCache.TakeDccRepeatsSkipped()} " +
            $"dcc_slices_coalesced={_imageCache.TakeDccSlicesCoalesced()} " +
            $"dcc_clears_dropped={_imageCache.TakeDccClearsDropped()} " +
            // Shaders whose control flow the structured emitter could express, against
            // the ones that fell back to the PC dispatcher: a dispatcher shader keeps the
            // whole guest register file live across its loop, so a rising count here is
            // the first thing to look at when GPU time will not come down.
            $"msl_structured={ShaderCompiler.Metal.Gen5MslTranslator.Statistics.ControlFlowForms.Structured} " +
            $"msl_dispatcher={ShaderCompiler.Metal.Gen5MslTranslator.Statistics.ControlFlowForms.Dispatcher} " +
            $"uploads={uploads} writebacks={writebacks} evictions={evictions} failed={failed} " +
            (_meshletCache?.TakeCounters() ?? "meshlet_draws=0") + " " + TakeIndirectCounters());
        _slowWaitReports = 0;
        ResetSlowCompletionReports();
        var reasons = new System.Text.StringBuilder();
        for (var index = 0; index < MetalWaitReasons.Count; index++)
        {
            var count = _waitsByReason[index];
            if (count == 0)
            {
                continue;
            }

            var ticks = _waitTicksByReason[index];
            _waitsByReason[index] = 0;
            _waitTicksByReason[index] = 0;
            reasons.Append(' ').Append(MetalWaitReasons.Names[index]).Append('=').Append(count)
                .Append('/').Append((ticks * 1000.0 / frequency).ToString("F0", culture)).Append("ms");
        }

        for (var index = 0; index < MetalGpuWriteOrigins.Count; index++)
        {
            var records = _gpuWrites.RecordsFrom((MetalGpuWriteOrigin)index);
            if (records != 0)
            {
                reasons.Append(' ').Append(MetalGpuWriteOrigins.Names[index]).Append("_records=").Append(records);
            }
        }

        var labels = TakeLabelCounters();
        var cpWrites = TakeCommandProcessorWriteCounters();
        var clocks = TakeClockLabelCounters();
        reasons.Append(" host_fills=").Append(Interlocked.Exchange(ref _hostFills, 0))
            .Append(" gpu_fills=").Append(Interlocked.Exchange(ref _gpuFills, 0))
            .Append(" cleared_fills=").Append(Interlocked.Exchange(ref _absorbedFills, 0))
            .Append(" host_copies=").Append(Interlocked.Exchange(ref _hostCopies, 0))
            .Append(TakeCopyReadCounters())
            .Append(" inline_eop=").Append(TakeInlineCompletions())
            .Append(" gpu_labels=").Append(labels.GpuTimed)
            .Append(" host_labels=").Append(labels.WithoutWindow)
            .Append(" ordered_labels=").Append(TakeOrderedHostLabels())
            .Append(" clock_labels=").Append(clocks.Completion)
            .Append(" clock_forced=").Append(clocks.Forced)
            .Append(" lazy_labels=").Append(TakeLazyLabelCompletions())
            .Append(" gpu_cp_writes=").Append(cpWrites.GpuTimed)
            .Append(" host_cp_writes=").Append(cpWrites.Host)
            .Append(" cp_overlay_reads=").Append(TakeOverlayReads())
            .Append(" label_overlay_hits=").Append(TakeLabelOverlayHits())
            .Append(" cp_overlay=").Append(_cpOverlay.Count)
            .Append(TakeWordLineCounters())
            .Append(" poll_flushes=").Append(TakePollFlushes())
            .Append(" transient_commits=").Append(TakeTransientCommits())
            .Append(" transient_ring_mb=").Append(Interlocked.Exchange(ref _transientCommitRingBytes, 0) / (1024 * 1024))
            .Append(" transient_staging_mb=").Append(Interlocked.Exchange(ref _transientCommitStagingBytes, 0) / (1024 * 1024))
            .Append(" scheduled_labels=").Append(TakeScheduledLabelSubmissions())
            .Append(" keep_warm=").Append(Interlocked.Exchange(ref _keepWarmSubmissions, 0))
            .Append(" label_barriers=").Append(TakeLabelBarriers())
            .Append(" residency_skipped=").Append(TakeSkippedResidencyCalls())
            .Append(" device_address_writes=").Append(DeviceAddressWrites)
            .Append(TakeDeviceAddressReachCounters())
            .Append(" data_slots_device=").Append(SharpEmu.ShaderCompiler.Resources.DeviceDataLoads.LearnedSlots)
            .Append(" data_replans=").Append(SharpEmu.ShaderCompiler.Resources.DeviceDataLoads.Replans)
            .Append(" resident=").Append(_guestMemory.ResidentBuffers.Length)
            .Append(" windows_touched=").Append(_guestMemory.TouchedWindows.Count);
        Console.Error.WriteLine($"[METAL-V2][WAITS]{reasons}");
        WriteCopyReadTraces();
        var causes = _imageCache.TakeTransferCauses();
        var subRange = _imageCache.TakePartialTransfers();
        var aliases = _imageCache.AliasCounters;
        var lockedCpuWrites = _imageCache.TakeLockedCpuWrites();
        Console.Error.WriteLine(
            $"[METAL-V2][CHURN] up_new={causes.Created} up_cpu={causes.CpuWrite} up_gpu={causes.GpuWrite} " +
            $"up_other={causes.Invalidated} up_alias={causes.Alias} " +
            $"wb_bind={causes.Binding} wb_store={causes.Storage} " +
            $"wb_read={causes.HostRead} wb_copy={causes.GuestCopy} wb_overlap={causes.Overlap} " +
            $"wb_release={causes.Release} wb_drain={causes.Drain}{ReleaseCauses()}{_imageCache.TakeWritebackFenceCounters()} " +
            $"alias_kept={aliases.Kept} alias_published={aliases.Published} alias_narrow={aliases.Narrow} " +
            $"alias_read_skip={aliases.ReadsSkipped} " +
            $"alias_fallback={_imageCache.TakeAliasFallbacks()} " +
            $"linear_views={_imageCache.LinearViewCount} linear_view_promotions={_imageCache.TakeLinearViewPromotions()} " +
            $"usage_promotions={_imageCache.TakeUsagePromotions()} " +
            $"alias_on_device={_imageCache.TakeDeviceAliasPublishes()} " +
            $"stencil_planes={_imageCache.TakeStencilPlanePublishes()} stencil_device={_imageCache.TakeStencilDeviceHandOvers()} " +
            $"alias_reused={_imageCache.TakeDeviceAliasReuses()} untested_depth_writes={TakeUntestedDepthWrites()} " +
            $"part_up={subRange.Uploads} part_up_mb={subRange.UploadBytes / (1024 * 1024)} " +
            $"part_wb={subRange.Writebacks} part_wb_mb={subRange.WritebackBytes / (1024 * 1024)} " +
            $"write_ranges={_gpuWrites.Count} write_records={_gpuWrites.Recorded} " +
            $"write_folds={_gpuWrites.Folds} handler_wakeups={_completions.HandlerWakeups} " +
            $"rings={_guestMemory.RingCount} rings_freed={_guestMemory.RingsReleased} " +
            $"ring_mb={_guestMemory.RingFallbackBytes / (1024 * 1024)} " +
            // CPU writes that waited on the image cache lock: an image may own the page,
            // or the write reached watched DCC metadata. Of the owner ones, those that found
            // an image over their bytes and those that found none.
            $"cpu_write_locked={lockedCpuWrites.Owner}/{lockedCpuWrites.Metadata} " +
            $"cpu_write_owner_found={lockedCpuWrites.OwnerFound}/{lockedCpuWrites.OwnerNone}{StagedRangeCounters()}{GuestViewCounters()}" +
            TakeKernelFillCounters());
        if (_imageCache.TakeLossyCensus() is { } lossy)
        {
            Console.Error.WriteLine(lossy);
        }

        // On a tile-based GPU the attachments are the bandwidth: a pass loads every
        // one it does not clear and stores every one it keeps, so this says what the
        // passes cost in memory traffic and which splits are paying for it. Printed
        // unconditionally because it is the one number that tells whether a pass
        // split matters, and two seconds apart it costs nothing.
        var (loadBytes, storeBytes, clearedBytes, discardedBytes, emptyPasses, returnedPasses) =
            TakeAttachmentBandwidth();
        var (endClearCount, endTargetCount, endTransferCount, endComputeCount, endFlushCount, endPresentCount,
            endOtherCount, targetlessCount) = TakePassBreakdown();
        var (clearPasses, clearDepthPasses, clearPixels) = TakeClearPasses();
        var (foldedClears, encodedClears) = TakeClearFolding();
        var (depthJoins, depthJoinsSampled, depthJoinsCopies) = TakeDepthOnlyJoins();
        Console.Error.WriteLine(
            $"[METAL-V2][TBDR] passes={passes} empty={emptyPasses} returned={returnedPasses} " +
            $"load_mb={(loadBytes / (1024.0 * 1024.0)).ToString("F1", culture)} " +
            $"store_mb={((storeBytes - discardedBytes) / (1024.0 * 1024.0)).ToString("F1", culture)} " +
            $"discarded_mb={(discardedBytes / (1024.0 * 1024.0)).ToString("F1", culture)} " +
            $"cleared_mb={(clearedBytes / (1024.0 * 1024.0)).ToString("F1", culture)} " +
            $"end_clear={endClearCount} end_target={endTargetCount} end_transfer={endTransferCount} " +
            $"end_compute={endComputeCount} end_flush={endFlushCount} end_present={endPresentCount} " +
            $"end_other={endOtherCount} end_indirect={TakeIndirectPassEnds()} {TakeExecutorPassEnds()} targetless={targetlessCount} " +
            $"clears=[{TakeClearArrivals()}] clear_passes={clearPasses} " +
            $"clear_depth={clearDepthPasses} clear_mpx={(clearPixels / 1e6).ToString("F1", culture)} " +
            $"labels_deferred={TakeDeferredLabelWrites()} clears_folded={foldedClears} " +
            $"clears_encoded={encodedClears} depth_joins={depthJoins} depth_join_sampled={depthJoinsSampled} depth_join_copies={depthJoinsCopies} " +
            $"transfers=[{TakeTransferCauses()}]");
        if (_guard is not null)
        {
            Console.Error.WriteLine($"[METAL-V2][GUARD] {_guard.Report()}");
        }

        if (_slotTrap is not null)
        {
            _slotTrap.Verify(CompletedSerial);
            Console.Error.WriteLine(
                $"[METAL-V2][SLOT-TRAP] mismatches={_slotTrap.Mismatches} dropped={_slotTrap.Dropped}");
        }

        ReportStalledHeads();

        if (!MetalV2Diagnostics.Verbose)
        {
            return;
        }

        var (eventReady, eventWaited, eventWaitMs, eventWakeMs) = _completions.TakeWaitProfile();
        var blocked = CommandStream.SnapshotBlocked();
        Console.Error.WriteLine(
            $"[METAL-V2][TRACE] created={created} " +

            $"gpu_busy_n={gpuBusySamples} event_ready={eventReady} event_waited={eventWaited} " +
            $"event_wait_ms={eventWaitMs.ToString("F1", culture)} " +
            $"event_wake_ms={eventWakeMs.ToString("F1", culture)} " +
            $"slices={CommandStream.SlicesRun} blocked={blocked.Outstanding} " +
            $"blocked_wait=0x{blocked.SampleWaitAddress:X16} " +
            $"ring_fallbacks={_guestMemory.RingFallbacks} ring_mb={_guestMemory.RingFallbackBytes / (1024 * 1024)} " +
            $"oversized_in_place={_guestMemory.OversizedInPlace}");
    }

    private string ReleaseCauses()
    {
        var builder = new System.Text.StringBuilder();
        _imageCache.AppendReleaseCauses(builder);
        return builder.ToString();
    }

    // A head parked on one packet for seconds is a deadlock in the making; name the packet
    // and what it waits for once, with the words it compares, so the log says why.
    private void ReportStalledHeads()
    {
        var stalls = CommandStream.TakeNewStalls(3000);
        if (stalls.Count == 0)
        {
            return;
        }

        Span<uint> packet = stackalloc uint[8];
        Span<uint> watched = stackalloc uint[2];
        foreach (var (queueId, submissionId, kind, packetAddress, wait, milliseconds) in stalls)
        {
            packet.Clear();
            _ = _memory.TryRead(packetAddress, System.Runtime.InteropServices.MemoryMarshal.AsBytes(packet));
            var waited = "";
            if (wait.Kind == SharpEmu.Libs.Gpu.GpuCommands.CommandWaitKind.GuestMemory && wait.Address != 0)
            {
                watched.Clear();
                _ = _memory.TryRead(wait.Address, System.Runtime.InteropServices.MemoryMarshal.AsBytes(watched));
                waited = $" value=0x{watched[1]:X8}{watched[0]:X8} gpu_pending={_gpuWrites.MayCover(wait.Address, wait.Size)}";
            }

            if (wait.Kind == SharpEmu.Libs.Gpu.GpuCommands.CommandWaitKind.Pipeline && _compiler is { } compiler)
            {
                waited = $" compiling=[{compiler.DescribeInFlight()}]";
            }

            if (wait.Kind == SharpEmu.Libs.Gpu.GpuCommands.CommandWaitKind.GuestMemory && wait.Address != 0)
            {
                var now = System.Diagnostics.Stopwatch.GetTimestamp();
                foreach (var entry in SharpEmu.Libs.Gpu.GpuCommands.CommandProcessorWriteHistory.Find(wait.Address, wait.Size))
                {
                    Console.Error.WriteLine(
                        $"[METAL-V2][STALL]   written queue={entry.QueueId} kind={entry.Kind} deferred={entry.Deferred} " +
                        $"address=0x{entry.Address:X16} size={entry.Size} value=0x{entry.Value:X} " +
                        $"ms_ago={(now - entry.Ticks) * 1000.0 / System.Diagnostics.Stopwatch.Frequency:F0}");
                }
            }

            Console.Error.WriteLine(
                $"[METAL-V2][STALL] queue={queueId} submission={submissionId} kind={kind} ms={milliseconds:F0} " +
                $"packet=0x{packetAddress:X16} words=[{packet[0]:X8} {packet[1]:X8} {packet[2]:X8} {packet[3]:X8} " +
                $"{packet[4]:X8} {packet[5]:X8} {packet[6]:X8} {packet[7]:X8}] wait={wait.Kind} " +
                $"address=0x{wait.Address:X16} size=0x{wait.Size:X} handle={wait.Handle} index={wait.Index}{waited}");
        }
    }

    public Exception Fatal(string message) => GpuFatal.Fatal(message);

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        try
        {
            EndEncoders(PassEndCause.Other);
            MaterializePendingClears();
            if (_commandBuffer != 0)
            {
                MetalNative.SendVoid(_commandBuffer, SelCommit);
                MetalNative.SendVoid(_commandBuffer, SelWaitUntilCompleted);
                MetalNative.SendVoid(_commandBuffer, SelRelease);
                _commandBuffer = 0;
            }

            if (_previousCommandBuffer != 0)
            {
                MetalNative.SendVoid(_previousCommandBuffer, SelRelease);
                _previousCommandBuffer = 0;
            }

            _scheduleNudge?.Dispose();
            ReleaseInFlightSubmissions();
            ReportStepCaps();
            DisposeStepReport();
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"[METAL-V2][WARN] The render host could not drain on shutdown: {exception.Message}");
        }

        foreach (var (retired, _) in _retiredLabelBarrierEvents)
        {
            MetalNative.SendVoid(retired, SelRelease);
        }

        _retiredLabelBarrierEvents.Clear();
        if (_labelBarrierEvent != 0)
        {
            MetalNative.SendVoid(_labelBarrierEvent, SelRelease);
            _labelBarrierEvent = 0;
        }

        _completions.Dispose();
        DisposeHostMovie();
        DisposeBindlessImages();
        DisposePipelines();
        (_transfer as IDisposable)?.Dispose();
        // The guest has gone: released chunks are not restored on the way out.
        _imageCache.SingleCopy = null;
        _singleCopy?.Dispose();
        _imageCache.Dispose();
        _stagedRanges?.Dispose();
        _guestMemory.Dispose();
        if (_sharedEvent != 0)
        {
            MetalNative.SendVoid(_sharedEvent, SelRelease);
        }
    }

    private void SetLabel(nint resource, string label)
    {
        var pool = MetalNative.objc_autoreleasePoolPush();
        try
        {
            MetalNative.SendVoid(resource, SelSetLabel, MetalNative.NsString(label));
        }
        finally
        {
            MetalNative.objc_autoreleasePoolPop(pool);
        }
    }
}
