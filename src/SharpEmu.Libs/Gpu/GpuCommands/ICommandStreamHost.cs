// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;

namespace SharpEmu.Libs.Gpu.GpuCommands;

public enum DrawOffsetSource
{
    Packet,
    IndirectArguments,
}

// Arguments of an indexed draw as the packet stream describes them.
public readonly record struct DrawIndexedArguments(
    ulong PacketAddress,
    uint Opcode,
    uint IndexCount,
    ulong IndexAddress,
    uint IndexTypeAndSize,
    uint InstanceCount,
    int BaseVertex,
    uint FirstInstance,
    DrawOffsetSource OffsetSource,
    // Nonzero when the counts are still in guest memory at this address, laid out as
    // (indexCount, instanceCount, firstIndex, vertexOffset, firstInstance). IndexCount is
    // then the index buffer size and IndexAddress its base.
    ulong IndirectArgumentsAddress = 0,
    // The guest set no index buffer size: IndexCount is only the range seen so far and
    // must not clamp the count read from the arguments.
    bool UnboundedIndexBuffer = false)
{
    // Predicated, or inside an occlusion query: never left out while its pipeline compiles.
    public bool MustExecute { get; init; }
}

// Arguments of a non-indexed draw as the packet stream describes them.
public readonly record struct DrawAutoArguments(
    ulong PacketAddress,
    uint Opcode,
    uint VertexCount,
    uint InstanceCount,
    uint FirstVertex,
    uint FirstInstance,
    DrawOffsetSource OffsetSource,
    // Nonzero when the counts are still in guest memory at this address, laid out as
    // (vertexCount, instanceCount, firstVertex, firstInstance). VertexCount is then a
    // placeholder and the other counts are unused.
    ulong IndirectArgumentsAddress = 0)
{
    // Predicated, or inside an occlusion query: never left out while its pipeline compiles.
    public bool MustExecute { get; init; }
}

public enum EndOfPipeWriteKind
{
    Write32,
    Write64,
    WriteBack32,
    WriteBack64,
    InterruptOnly,
    Interrupt32,
    Interrupt64,
    InterruptWriteBack32,
    InterruptWriteBack64,
    GdsWrite32,
    ClockWrite,
    ClockWriteBack,
    Flip,
    FlipWithWrite32,
    FlipWithInterruptWriteBack32,

    // RELEASE_MEM data select 7: Value's low 32 bits are added to the dword at Destination
    // once the work before it has finished.
    AtomicAdd32,
}

// A completion the host records on the current command buffer. The label itself is
// already in guest memory unless the host explicitly defers end-of-pipe labels.
// ReferenceClock marks a payload that is the GPU reference clock sampled as the packet was
// parsed (an end-of-pipe timestamp or a COPY_DATA of the clock, whatever its kind), so a
// host that publishes the label later may sample the clock again when the work finished.
public readonly record struct EndOfPipeWrite(
    EndOfPipeWriteKind Kind,
    ulong SubmitId,
    ulong Destination = 0,
    ulong Value = 0,
    int EventId = 0,
    uint ContextId = 0,
    uint GdsWordOffset = 0,
    uint GdsWordCount = 0,
    int FlipHandle = 0,
    int FlipIndex = 0,
    int FlipMode = 0,
    long FlipArgument = 0,
    ulong FlipRequestId = 0,
    bool ReferenceClock = false);

// The host state a command block (or a refusal to give one) was read under: four counters the
// host bumps whenever bytes a packet read could return, or whether that read would wait, change.
public readonly record struct CommandReadToken(long First, long Second, long Third, long Fourth);

// Everything a packet handler needs from the renderer, the caches, the scheduler and video out.
public interface ICommandStreamHost
{
    ICpuMemory Memory { get; }

    bool TryRecordLodStatistics(ulong destination, uint size) => false;

    /// <summary>
    /// GET_LOD_STATS measured on the GPU: the host writes the counter bank into the guest
    /// buffer after the work recorded before the packet, and clears the bank when
    /// <paramref name="resetCounters"/> is set. False leaves the packet to the interpreter's
    /// bound-slot fallback.
    /// </summary>
    bool TryRecordLodStatistics(ulong destination, uint size, bool resetCounters) =>
        TryRecordLodStatistics(destination, size);

    // Reads guest memory the GPU may have written; the host synchronizes GPU-owned pages first.
    bool TryReadGuest(ulong address, Span<byte> destination);

    /// <summary>
    /// Whether this host fills <see cref="TryReadCommandBlock"/> blocks. A host that does not
    /// is read one packet at a time, as before.
    /// </summary>
    bool SupportsCommandBlocks => false;

    /// <summary>
    /// Reads <paramref name="block"/>.Length bytes of command memory in one go, for a stream
    /// that serves its next packets from the copy. True only when every byte is what
    /// <see cref="TryReadGuest"/> would return for it right now without waiting for the GPU
    /// and without any pending command-processor bytes over it; otherwise false, and the
    /// stream reads that region one packet at a time. <paramref name="token"/> names the
    /// host state the answer (granted or refused) rests on, and is set either way.
    /// </summary>
    bool TryReadCommandBlock(ulong address, Span<byte> block, out CommandReadToken token)
    {
        token = default;
        return false;
    }

    /// <summary>
    /// Whether nothing the host tracks has changed since <paramref name="token"/> was taken, so
    /// a block read then, or a refusal given then, still holds.
    /// </summary>
    bool IsCommandBlockCurrent(in CommandReadToken token) => false;

    // Reads a packet operand (a label, a predicate, indirect arguments): memory earlier work may
    // write, unlike the packets themselves. A host that defers work completes it first.
    bool TryReadGuestOperand(ulong address, Span<byte> destination) => TryReadGuest(address, destination);

    /// <summary>
    /// Reads the label a wait packet polls, as memory holds it now.
    /// </summary>
    /// <remarks>
    /// The hardware's command processor reads memory and stalls on the value it finds, so a
    /// poll must not wait for work that is about to write those bytes: an unmet compare
    /// suspends the submission and the host wakes it when the write lands, so a stale value
    /// only delays the wait until a retry. A host that puts the write in a command buffer has
    /// to make sure that buffer is on its way, or nothing will ever change the bytes - when
    /// <paramref name="commitPendingWrite"/> is set: the first look at a wait commits nothing,
    /// because the queue's own store or a recorded label may satisfy it without the bytes
    /// changing. Hosts whose operand reads do not wait need no override.
    /// </remarks>
    bool TryReadGuestWaitOperand(ulong address, Span<byte> destination, bool commitPendingWrite) =>
        TryReadGuestOperand(address, destination);

    /// <summary>
    /// Whether recorded work still owes these bytes and will land them on its own, so that
    /// a packet reading them (a predicate, a COND_EXEC flag, a COND_WRITE compare, a branch)
    /// should suspend on them instead of stalling the worker until the GPU has finished.
    /// </summary>
    /// <remarks>
    /// A host that answers true has already made sure the owing work is on its way (it
    /// committed the buffer that carries it) and will wake guest-memory waits when it
    /// completes. False means the bytes can be read now: they are final, or only the
    /// host's ordinary operand read can produce them (an image writeback), which then waits.
    /// </remarks>
    bool CommitPendingOperandWrite(ulong address, ulong size) => false;

    /// <summary>
    /// Whether an end-of-pipe GDS store (<see cref="EndOfPipeWriteKind.GdsWrite32"/>) is
    /// copied out of the data share by this host in stream order, after the work before it.
    /// Otherwise the interpreter synchronizes with the GPU and writes the words itself.
    /// </summary>
    bool WritesGdsAtEndOfPipe => false;

    // Immediate hosts retain their ordinary memory effects. An ordered host
    // must finish this owned operation before interpreter results are consumed.
    bool TryWriteGuest(ulong address, ReadOnlySpan<byte> source, string kind = "WRITE_BYTES") => Memory.TryWrite(address, source);
    void ExecuteGuestMemoryOperation(Action operation, string kind = "MEMORY_OPERATION", ulong address = 0, ulong bytes = 0, ulong value = 0) => operation();
    void ExecuteCommandMemoryOperation(CommandMemoryOperation operation) =>
        ExecuteGuestMemoryOperation(() => operation.Execute(this), operation.DebugName,
            operation.Address, operation.ByteCount, operation.Value);

    // Runs commands other threads posted to this worker. Called before every packet.
    void RunPendingCommands();

    // Starts a slice of a submission; the geometry snapshots were captured when it was submitted.
    void BeginSubmission(int queueId, ulong submissionId, object? geometrySnapshots);

    void Flush();

    // A frame boundary: submits what the frame recorded. The submitter resumes once it returns, so a
    // host that runs the stream ahead of its recording returns only after the frame is recorded.
    void FlushFrame() => Flush();

    void FlushAndWait();

    void SynchronizeGpu();

    void RunGarbageCollector();

    void EmitGlobalBarrier();

    // The guest waited for or invalidated earlier shader stores (a pixel-shader partial flush,
    // a cache flush, an acquire), so a consumer of them can follow. A host whose executor keeps
    // the render pass open after a fragment-stage store ends that pass here.
    void ReleaseShaderWrites() { }

    void FillBuffer(ulong address, ulong size, uint value, bool isGds);

    void CopyBuffer(ulong destination, ulong source, ulong size, bool destinationIsGds, bool sourceIsGds);

    // Records a direct packet write for the image-clear trace; see docs/image-clear-tracing.md.
    void TraceGuestWrite(string operation, ulong address, ulong size) { }

    // Prints recorded writers of a range for the image-clear trace.
    void TraceWritersOf(string subject, ulong address, ulong size) { }

    void ReadGds(Span<uint> destination, uint wordOffset, uint wordCount);

    // A deferred host owns the scalar label payload and must publish its bytes
    // successfully after preceding GPU work, before any interrupt/flip acknowledgement.
    // Other hosts retain the existing immediate interpreter label writes.
    bool DefersEndOfPipeLabels => false;

    /// <summary>Whether this host publishes one kind of label itself, after its work.</summary>
    /// <remarks>
    /// Kinds differ in what the guest does with them. A fence the guest polls before reusing
    /// the memory the work reads has to be published after that work or the reuse races it;
    /// a timestamp is read for its value and not as permission to touch anything.
    /// </remarks>
    bool DefersEndOfPipeLabel(EndOfPipeWriteKind kind) => DefersEndOfPipeLabels;

    // A host that counts no occlusion samples (Metal). Its ZPASS_DUMP pairs are the
    // always-visible ones: every dump writes one value from a counter that only grows to
    // 16 depth blocks, and the counter select and reset events are ignored, so an end dump
    // always reads later than its begin, whichever queue, select or reset came between.
    bool PublishesSyntheticOcclusionDumps => false;

    // Return completed samples for counter zero. False keeps the legacy fallback.
    bool TryReadOcclusionCounter(int queueId, out ulong value)
    {
        value = 0;
        return false;
    }

    bool ResetOcclusionCounter(int queueId) => false;

    bool DumpOcclusionCounter(int queueId, ulong address, uint instanceMask, uint strideBytes) => false;

    bool HasPendingPublications(int queueId) => false;

    bool HasPendingMemoryPublication(ulong address, ulong size) => false;

    // After a wait on another queue's unpublished label, this queue's writes must not land first.
    // True when the write was held back; the host publishes it after that label.
    bool TryDeferOrderedWrite(int queueId, ulong address, ReadOnlySpan<byte> source) => false;

    // True when a read-modify-write must wait for the same reason; the caller suspends and retries.
    bool MustHoldForCrossQueueOrder(int queueId) => false;

    // Returns false when a pending publication hides the bytes from the command processor.
    bool TryReadDeferredLabel(int queueId, ulong address, ulong size, out bool deferred, out ulong value)
    {
        deferred = false;
        value = 0;
        return true;
    }

    bool TryDeferEndOfPipe(int queueId, in EndOfPipeWrite write) => false;

    void RecordEndOfPipe(in EndOfPipeWrite write);

    /// <summary>
    /// The GPU tick a label deferred now will have landed by, or zero when this host does
    /// not record such labels in GPU order. Other queues' waits may treat a recorded label
    /// as written until that tick completes: every queue encodes into one GPU stream in
    /// order, so their later work runs after the label all the same, while the guest CPU
    /// still sees the bytes only when the GPU writes them.
    /// </summary>
    ulong PendingLabelTick => 0;

    /// <summary>
    /// Whether the command-processor memory writes (WRITE_DATA, DMA_DATA fills and copies,
    /// the graphics queue's COPY_DATA immediates) are handed to this host to land in record
    /// order with its deferred end-of-pipe labels, instead of being written as parsed.
    /// </summary>
    /// <remarks>
    /// A host that defers labels runs the parser far ahead of the GPU. A write made at parse
    /// time then reaches the guest before labels recorded ahead of it, which on the hardware
    /// it never does once a wait between them has stalled the command processor; a guest
    /// that recycles memory on such a write (Demon's Souls) has the late label land in it.
    /// </remarks>
    bool DefersCommandProcessorWrites => false;

    /// <summary>Writes command-processor bytes in record order with deferred labels.</summary>
    void RecordCommandProcessorWrite(ulong address, ReadOnlySpan<byte> source, string kind)
    {
        if (!TryWriteGuest(address, source, kind))
        {
            throw Fatal($"The command stream cannot write guest memory: address=0x{address:X16} size={source.Length}.");
        }
    }

    /// <summary>
    /// Writes <paramref name="count"/> copies of one eight-byte value, <paramref name="strideBytes"/>
    /// apart, in record order with deferred labels: the same bytes, in the same order, as
    /// <paramref name="count"/> calls to <see cref="RecordCommandProcessorWrite"/> with nothing between
    /// them. A host that can record them as one write does; the default does exactly that.
    /// </summary>
    void RecordCommandProcessorQwords(ulong address, int count, ulong strideBytes, ulong value, string kind)
    {
        Span<byte> bytes = stackalloc byte[sizeof(ulong)];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt64LittleEndian(bytes, value);
        for (var index = 0; index < count; index++)
        {
            RecordCommandProcessorWrite(address + (ulong)index * strideBytes, bytes, kind);
        }
    }

    /// <summary>A DMA fill of guest memory that lands in record order with deferred labels.</summary>
    void FillBufferInOrder(ulong address, ulong size, uint value) => FillBuffer(address, size, value, isGds: false);

    /// <summary>A DMA copy between guest ranges that lands in record order with deferred labels.</summary>
    void CopyBufferInOrder(ulong destination, ulong source, ulong size) =>
        CopyBuffer(destination, source, size, destinationIsGds: false, sourceIsGds: false);

    // Delivers the interrupt now, without waiting for the GPU.
    void TriggerInterrupt(int eventId, uint contextId);

    bool HasFlipSlot();

    // Reserves and prepares a flip; the request id is never zero.
    ulong PrepareFlip(int handle, int index, int flipMode, long flipArgument);
    ulong PrepareFlip(int queueId, int handle, int index, int flipMode, long flipArgument) =>
        PrepareFlip(handle, index, flipMode, flipArgument);

    // True when no request still targets the buffer, so the stream may render into it.
    bool IsFlipDone(int handle, int index);

    // A video-out export flip reached its place in the graphics queue: capture the buffer now.
    void PrepareCpuFlip(int handle, int index, ulong requestId);

    void DrawIndexed(ulong submitId, in DrawIndexedArguments arguments);

    void DrawAuto(ulong submitId, in DrawAutoArguments arguments);

    void DispatchDirect(ulong submitId, uint groupsX, uint groupsY, uint groupsZ, uint dispatchInitiator, ulong indirectArgumentsAddress = 0);

    // True when the draw or dispatch just handed over could not run because its
    // pipeline is still compiling. The packet then suspends and runs again when the
    // backend reports the pipeline ready; nothing is skipped.
    bool ConsumePipelineNotReady() => false;

    // True when the host wants the dispatches after one that suspended on its pipeline read
    // ahead (GpuCommandInterpreter.PrefetchPipelinesAhead), so their compiles start beside it.
    bool PrefetchesPipelines => false;

    // One dispatch the packets after a suspended one make, with the shader registers it will
    // run with on a scratch copy: the host may start compiling its pipeline. Nothing is recorded
    // and nothing may be consumed; the dispatch itself still runs when the stream reaches it.
    void PrefetchDispatchPipeline(RegisterBanks banks, uint groupsX, uint groupsY, uint groupsZ, uint dispatchInitiator)
    {
    }

    // True when an indirect dispatch in workgroup units reads its group counts on the
    // GPU, so the interpreter need not wait for the GPU to read them back first.
    bool ResolvesIndirectDispatchOnGpu => false;

    // The same for an indirect dispatch in thread units, whose conversion to groups the host
    // then also does on the GPU (IRenderHost.TryDispatchThreadsIndirect).
    bool ResolvesThreadIndirectDispatchOnGpu => false;

    // True when an indexed indirect draw may be handed over with its arguments still in
    // guest memory (DrawIndexedArguments.IndirectArgumentsAddress).
    bool ResolvesIndirectDrawOnGpu => false;

    // The same for a non-indexed indirect draw (DrawAutoArguments.IndirectArgumentsAddress).
    // Such a draw is handed over that way only while HasPendingGpuWrite says reading its
    // arguments would wait; final arguments are read at once.
    bool ResolvesIndirectAutoDrawOnGpu => false;

    // True when GPU work already recorded may still write these guest bytes, so that a CPU
    // read of them now would wait for it.
    bool HasPendingGpuWrite(ulong address, ulong size) => true;

    // Called before the CPU reads an indirect draw's arguments; a host may count it.
    void NoteIndirectArgumentsReadOnCpu(ulong address, ulong size) { }

    // Called when a queue reset packet clears the processor.
    void OnQueueReset(int queueId);

    Exception Fatal(string message);
}
