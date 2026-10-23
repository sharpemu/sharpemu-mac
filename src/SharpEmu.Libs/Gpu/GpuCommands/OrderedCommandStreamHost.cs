// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using SharpEmu.HLE;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.VideoOut;

namespace SharpEmu.Libs.Gpu.GpuCommands;

/// <summary>Streaming ordered issue protocol shared by the split and serial Metal workers.</summary>
public sealed class OrderedCommandStreamHost : ICommandStreamHost
{
    private enum Operation { DrawIndexed, DrawAuto, Dispatch, Sync, WriterPrefix, TryRecordLodStatistics, TryReadGuest, TryReadCommandBlock, IsCommandBlockCurrent, TryReadGuestOperand, TryReadGuestWaitOperand, CommitPendingOperandWrite, TryWriteGuest, ExecuteGuestMemoryOperation, ExecuteCommandMemoryOperation, RunPendingCommands, BeginSubmission, Flush, FlushFrame, FlushAndWait, SynchronizeGpu, RunGarbageCollector, EmitGlobalBarrier, ReleaseShaderWrites, FillBuffer, CopyBuffer, TraceGuestWrite, TraceWritersOf, ReadGds, DefersEndOfPipeLabel, TryReadOcclusionCounter, ResetOcclusionCounter, DumpOcclusionCounter, HasPendingPublications, HasPendingMemoryPublication, TryDeferOrderedWrite, MustHoldForCrossQueueOrder, TryReadDeferredLabel, TryDeferEndOfPipe, RecordEndOfPipe, RecordCommandProcessorWrite, RecordCommandProcessorQwords, FillBufferInOrder, CopyBufferInOrder, TriggerInterrupt, HasFlipSlot, PrepareFlip, PrepareFlip2, IsFlipDone, PrepareCpuFlip, ConsumePipelineNotReady, HasPendingGpuWrite, NoteIndirectArgumentsReadOnCpu, OnQueueReset, GetPendingLabelTick, Maintenance }
    private sealed class Issue
    {
        internal Operation Operation;
        internal long Sequence;
        internal int Queue;
        internal ulong Submit;
        internal readonly RegisterBanks Banks;
        internal DrawIndexedArguments Indexed;
        internal DrawAutoArguments Auto;
        internal uint X, Y, Z, Initiator;
        internal ulong Indirect;
        internal bool BooleanResult;
        internal ulong UlongResult;
        internal readonly byte[] SmallBytes = new byte[1024];
        internal byte[] Bytes;
        internal int ByteCount;
        internal bool Rented;
        internal bool ReleasePayloadOnConsume;
        internal Issue(Func<string, Exception> fatal)
        {
            Banks = new(fatal);
            Banks.Context.UnmodeledTableRegisters.EnsureCapacity((int)RegisterBankLayout.ContextRegisterCount);
            Bytes = SmallBytes;
        }
        internal ulong TryRecordLodStatistics_destination;
        internal uint TryRecordLodStatistics_size;
        internal bool TryRecordLodStatistics_resetCounters;
        internal ulong TryReadGuest_address;
        internal ulong TryReadCommandBlock_address;
        internal CommandReadToken TryReadCommandBlock_token;
        internal CommandReadToken IsCommandBlockCurrent_token;
        internal ulong TryReadGuestOperand_address;
        internal ulong TryReadGuestWaitOperand_address;
        internal bool TryReadGuestWaitOperand_commitPendingWrite;
        internal ulong CommitPendingOperandWrite_address;
        internal ulong CommitPendingOperandWrite_size;
        internal ulong TryWriteGuest_address;
        internal string TryWriteGuest_kind = null!;
        internal Action ExecuteGuestMemoryOperation_operation = null!;
        internal string ExecuteGuestMemoryOperation_kind = null!;
        internal ulong ExecuteGuestMemoryOperation_address;
        internal ulong ExecuteGuestMemoryOperation_bytes;
        internal ulong ExecuteGuestMemoryOperation_value;
        internal CommandMemoryOperation ExecuteCommandMemoryOperation_operation = null!;
        internal int BeginSubmission_queueId;
        internal ulong BeginSubmission_submissionId;
        internal object? BeginSubmission_geometrySnapshots;
        internal ulong FillBuffer_address;
        internal ulong FillBuffer_size;
        internal uint FillBuffer_value;
        internal bool FillBuffer_isGds;
        internal ulong CopyBuffer_destination;
        internal ulong CopyBuffer_source;
        internal ulong CopyBuffer_size;
        internal bool CopyBuffer_destinationIsGds;
        internal bool CopyBuffer_sourceIsGds;
        internal string TraceGuestWrite_operation = null!;
        internal ulong TraceGuestWrite_address;
        internal ulong TraceGuestWrite_size;
        internal string TraceWritersOf_subject = null!;
        internal ulong TraceWritersOf_address;
        internal ulong TraceWritersOf_size;
        internal uint ReadGds_wordOffset;
        internal uint ReadGds_wordCount;
        internal EndOfPipeWriteKind DefersEndOfPipeLabel_kind;
        internal int TryReadOcclusionCounter_queueId;
        internal ulong TryReadOcclusionCounter_value;
        internal int ResetOcclusionCounter_queueId;
        internal int DumpOcclusionCounter_queueId;
        internal ulong DumpOcclusionCounter_address;
        internal uint DumpOcclusionCounter_instanceMask;
        internal uint DumpOcclusionCounter_strideBytes;
        internal int HasPendingPublications_queueId;
        internal ulong HasPendingMemoryPublication_address;
        internal ulong HasPendingMemoryPublication_size;
        internal int TryDeferOrderedWrite_queueId;
        internal ulong TryDeferOrderedWrite_address;
        internal int MustHoldForCrossQueueOrder_queueId;
        internal int TryReadDeferredLabel_queueId;
        internal ulong TryReadDeferredLabel_address;
        internal ulong TryReadDeferredLabel_size;
        internal bool TryReadDeferredLabel_deferred;
        internal ulong TryReadDeferredLabel_value;
        internal int TryDeferEndOfPipe_queueId;
        internal EndOfPipeWrite TryDeferEndOfPipe_write;
        internal EndOfPipeWrite RecordEndOfPipe_write;
        internal ulong RecordCommandProcessorWrite_address;
        internal string RecordCommandProcessorWrite_kind = null!;
        internal ulong RecordCommandProcessorQwords_address;
        internal int RecordCommandProcessorQwords_count;
        internal ulong RecordCommandProcessorQwords_strideBytes;
        internal ulong RecordCommandProcessorQwords_value;
        internal string RecordCommandProcessorQwords_kind = null!;
        internal ulong FillBufferInOrder_address;
        internal ulong FillBufferInOrder_size;
        internal uint FillBufferInOrder_value;
        internal ulong CopyBufferInOrder_destination;
        internal ulong CopyBufferInOrder_source;
        internal ulong CopyBufferInOrder_size;
        internal int TriggerInterrupt_eventId;
        internal uint TriggerInterrupt_contextId;
        internal int PrepareFlip_handle;
        internal int PrepareFlip_index;
        internal int PrepareFlip_flipMode;
        internal long PrepareFlip_flipArgument;
        internal int PrepareFlip2_queueId;
        internal int PrepareFlip2_handle;
        internal int PrepareFlip2_index;
        internal int PrepareFlip2_flipMode;
        internal long PrepareFlip2_flipArgument;
        internal int IsFlipDone_handle;
        internal int IsFlipDone_index;
        internal int PrepareCpuFlip_handle;
        internal int PrepareCpuFlip_index;
        internal ulong PrepareCpuFlip_requestId;
        internal ulong HasPendingGpuWrite_address;
        internal ulong HasPendingGpuWrite_size;
        internal ulong NoteIndirectArgumentsReadOnCpu_address;
        internal ulong NoteIndirectArgumentsReadOnCpu_size;
        internal int OnQueueReset_queueId;
    }

    private readonly ICommandStreamHost _target;
    private readonly ICommandIssueRenderer? _renderer;
    private readonly CommandIssueRing<Issue> _ring;
    private readonly bool _split;
    private readonly Func<bool> _hasRelay;
    private readonly Action? _maintenance;
    private readonly CommandIssueProgramPlanner? _planner;
    private readonly long[] _ackRequests = new long[Enum.GetValues<Operation>().Length];
    private readonly long[] _ackStalls = new long[Enum.GetValues<Operation>().Length];
    private CommandIssueRing<Issue>.Block? _building;
    private CommandStreamQueue? _queue;
    private int _currentQueue;
    private ulong _currentSubmit;
    private long _sequence;
    private long _writerIntent;
    private long _resolvedWriter;
    private long _bTicks;
    private long _aTicks;
    private long _aCpuNs;
    private long _bCpuNs;
    private long _watermarkStalls;
    private long _tokenInvalidations;
    private long _writerPrefixTicks;
    // A token check observes no guest bytes. An unresolved writer invalidates its
    // snapshot; only the subsequent memory read needs to resolve the issue prefix.
    public bool InvalidateUnknownWriterTokens { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_GPU_WRITER_TOKENS") != "0";
    private bool _dispatchResultPending;
    private CommandReadToken _completedReadToken;

    public OrderedCommandStreamHost(ICommandStreamHost target, bool split, Func<bool>? hasRelay = null,
        Action? maintenance = null, Action<ulong, ulong>? synchronizeCode = null, int blocks = 4, int itemsPerBlock = 64)
    {
        _target = target;
        _renderer = target as ICommandIssueRenderer;
        _split = split;
        _hasRelay = hasRelay ?? (() => false);
        _maintenance = maintenance;
        _ring = new(blocks, itemsPerBlock, () => new Issue(target.Fatal));
        if (_renderer is not null && synchronizeCode is not null)
            _planner = new(target.Memory, synchronizeCode);
    }

    public void AttachQueue(CommandStreamQueue queue) => _queue = queue;
    public ICpuMemory Memory => _target.Memory;
    public bool SupportsCommandBlocks => _target.SupportsCommandBlocks;
    public bool WritesGdsAtEndOfPipe => _target.WritesGdsAtEndOfPipe;
    public bool DefersEndOfPipeLabels => _target.DefersEndOfPipeLabels;
    public bool PublishesSyntheticOcclusionDumps => _target.PublishesSyntheticOcclusionDumps;
    public bool DefersCommandProcessorWrites => _target.DefersCommandProcessorWrites;
    public bool ResolvesIndirectDispatchOnGpu => _target.ResolvesIndirectDispatchOnGpu;
    public bool ResolvesIndirectDrawOnGpu => _target.ResolvesIndirectDrawOnGpu;
    public bool ResolvesIndirectAutoDrawOnGpu => _target.ResolvesIndirectAutoDrawOnGpu;
    public ulong PendingLabelTick => GetPendingLabelTick();
    public Exception Fatal(string message) => _target.Fatal(message);

    private Issue Next(Operation operation)
    {
        _ring.ThrowIfFailed();
        _building ??= _ring.Acquire();
        if (_building.Count == _building.Items.Length)
        {
            Publish();
            _building = _ring.Acquire();
        }
        var issue = _building.Items[_building.Count++];
        issue.Operation = operation;
        issue.Sequence = ++_sequence;
        issue.Queue = _currentQueue;
        issue.Submit = _currentSubmit;
        issue.ByteCount = 0;
        issue.ReleasePayloadOnConsume = false;
        _building.LastSequence = issue.Sequence;
        return issue;
    }

    private void Publish()
    {
        if (_building is not { Count: > 0 } block) return;
        _building = null;
        _ring.Publish(block);
        if (!_split) ConsumeBlock(block);
    }

    private void Finish(Issue issue, bool synchronize = true)
    {
        if (!synchronize)
        {
            _writerIntent = issue.Sequence;
            issue.ReleasePayloadOnConsume = true;
            if (_building is { } block && block.Count == block.Items.Length) Publish();
            return;
        }
        Publish();
        WaitForPrefix(issue.Sequence, issue.Operation);
    }

    private void WaitForPrefix(long sequence, Operation reason)
    {
        _ackRequests[(int)reason]++;
        using var wait = RenderPhaseProfile.Measure(RenderPhaseProfile.Phase.WorkerAWaitsForB);
        if (_ring.WaitFor(sequence)) _ackStalls[(int)reason]++;
        _resolvedWriter = _writerIntent;
    }

    // An unmaterialized draw is a possible writer of ANY guest address. It cannot be
    // invisible to a command-block token check, IB, predicate or register-table read.
    private void ResolveWriters()
    {
        if (_writerIntent <= _resolvedWriter) return;
        Interlocked.Increment(ref _watermarkStalls);
        var started = Stopwatch.GetTimestamp();
        Publish();
        WaitForPrefix(_writerIntent, Operation.WriterPrefix);
        Interlocked.Add(ref _writerPrefixTicks, Stopwatch.GetTimestamp() - started);
    }

    private static void ReleaseBytes(Issue issue)
    {
        if (!issue.Rented) return;
        ArrayPool<byte>.Shared.Return(issue.Bytes);
        issue.Bytes = issue.SmallBytes;
        issue.Rented = false;
    }

    private static void ReserveBytes(Issue issue, int count)
    {
        if (count > issue.Bytes.Length)
        {
            if (issue.Rented) ArrayPool<byte>.Shared.Return(issue.Bytes);
            issue.Bytes = ArrayPool<byte>.Shared.Rent(count);
            issue.Rented = true;
        }
        issue.ByteCount = count;
    }

    public void DrawIndexed(ulong submitId, in DrawIndexedArguments arguments)
    {
        var issue = Next(Operation.DrawIndexed);
        CaptureBanks(issue);
        issue.Submit = submitId;
        issue.Indexed = arguments;
        Finish(issue, synchronize: false);
    }

    public void DrawAuto(ulong submitId, in DrawAutoArguments arguments)
    {
        var issue = Next(Operation.DrawAuto);
        CaptureBanks(issue);
        issue.Submit = submitId;
        issue.Auto = arguments;
        Finish(issue, synchronize: false);
    }

    public void DispatchDirect(ulong submitId, uint groupsX, uint groupsY, uint groupsZ, uint dispatchInitiator, ulong indirectArgumentsAddress = 0)
    {
        var issue = Next(Operation.Dispatch);
        CaptureBanks(issue);
        issue.Submit = submitId;
        issue.X = groupsX; issue.Y = groupsY; issue.Z = groupsZ;
        issue.Initiator = dispatchInitiator; issue.Indirect = indirectArgumentsAddress;
        _dispatchResultPending = !(_renderer?.EncodesDispatchWithoutProducerRetry ?? false);
        Finish(issue, synchronize: false);
    }

    private void CaptureBanks(Issue issue)
    {
        if (_queue is not null) _queue.GetInterpreter(_currentQueue).TypedRegisters.CopyTo(issue.Banks);
    }

    public void Drain() => Finish(Next(Operation.Sync));
    public void WakeConsumer() => _ring.Wake();
    public void NoteProducerTicks(long ticks, long cpuNanoseconds = 0)
    {
        Interlocked.Add(ref _aTicks, ticks);
        Interlocked.Add(ref _aCpuNs, cpuNanoseconds);
        RenderPhaseProfile.RecordWorkerService(RenderPhaseProfile.Worker.A, ticks, cpuNanoseconds);
    }
    public void Stop() { Publish(); _ring.Stop(); }
    public void Fail(Exception error) => _ring.Fail(error);

    // B owns every target call. Each block has its own backend autorelease scope.
    public void Consume(Action? enterThread = null, Func<IDisposable>? enterBlock = null, Action? idle = null)
    {
        try
        {
            enterThread?.Invoke();
            for (;;)
            {
                CommandIssueRing<Issue>.Block? block;
                using (RenderPhaseProfile.Measure(RenderPhaseProfile.Phase.WorkerBWaitsForA))
                    block = _ring.Take(idle);
                if (block is null) return;
                using var scope = enterBlock?.Invoke();
                ConsumeBlock(block);
            }
        }
        catch (Exception error) { _ring.Fail(error); }
    }

    private void ConsumeBlock(CommandIssueRing<Issue>.Block block)
    {
        var started = Stopwatch.GetTimestamp();
        var cpuStarted = OperatingSystem.IsMacOS() ? SharpEmu.HLE.DsTrace.DsTrace.SelfCpuNs() : 0;
        using var profile = RenderPhaseProfile.Measure(RenderPhaseProfile.Phase.WorkerApply);
        try
        {
            for (var index = 0; index < block.Count; index++)
            {
                var issue = block.Items[index];
                Execute(issue);
                if (issue.ReleasePayloadOnConsume) ReleaseBytes(issue);
            }
            if (_target is ICommandIssueReadTokenSource source) _completedReadToken = source.IssueReadToken;
            _ring.Release(block);
        }
        catch (Exception error) { _ring.Fail(error); throw; }
        finally
        {
            var ticks = Stopwatch.GetTimestamp() - started;
            var cpu = OperatingSystem.IsMacOS() ? (long)(SharpEmu.HLE.DsTrace.DsTrace.SelfCpuNs() - cpuStarted) : 0;
            Interlocked.Add(ref _bTicks, ticks);
            Interlocked.Add(ref _bCpuNs, cpu);
            if (_split) RenderPhaseProfile.RecordWorkerService(RenderPhaseProfile.Worker.B, ticks, cpu);
        }
    }

    public string FormatProfile()
    {
        static double Ms(long ticks) => ticks * 1000.0 / Stopwatch.Frequency;
        var acknowledgements = new StringBuilder();
        for (var index = 0; index < _ackRequests.Length; index++)
        {
            var requests = Volatile.Read(ref _ackRequests[index]);
            if (requests != 0) acknowledgements.Append($" ack_{(Operation)index}={requests}/{Volatile.Read(ref _ackStalls[index])}");
        }
        return FormattableString.Invariant($"worker_split={(_split ? 1 : 0)} worker_a_ms={Ms(Interlocked.Read(ref _aTicks)):F2} worker_b_ms={Ms(_split ? Interlocked.Read(ref _bTicks) : 0):F2} issue_apply_ms={Ms(Interlocked.Read(ref _bTicks)):F2} worker_a_cpu_ms={Interlocked.Read(ref _aCpuNs) / 1000000.0:F2} worker_b_cpu_ms={(_split ? Interlocked.Read(ref _bCpuNs) : 0) / 1000000.0:F2} a_waits_b_ms={Ms(Interlocked.Read(ref _ring.AcknowledgementWaitTicks) + Interlocked.Read(ref _ring.FullWaitTicks)):F2} b_waits_a_ms={Ms(Interlocked.Read(ref _ring.EmptyWaitTicks)):F2} issue_blocks={Interlocked.Read(ref _ring.PublishedBlocks)} issue_records={Interlocked.Read(ref _ring.PublishedItems)} issue_wakes={Interlocked.Read(ref _ring.Wakeups)} issue_full_stalls={Interlocked.Read(ref _ring.FullStalls)} issue_sync_stalls={Interlocked.Read(ref _ring.SyncStalls)} unresolved_writer_stalls={Interlocked.Read(ref _watermarkStalls)} writer_token_invalidations={Interlocked.Read(ref _tokenInvalidations)} writer_prefix_ms={Ms(Interlocked.Read(ref _writerPrefixTicks)):F2}{acknowledgements}");
    }


    public bool TryRecordLodStatistics(ulong destination, uint size) => TryRecordLodStatistics(destination, size, false);

    public bool TryRecordLodStatistics(ulong destination, uint size, bool resetCounters)
    {
        var issue = Next(Operation.TryRecordLodStatistics);
        issue.TryRecordLodStatistics_destination = destination;
        issue.TryRecordLodStatistics_size = size;
        issue.TryRecordLodStatistics_resetCounters = resetCounters;
        Finish(issue, synchronize: true);
        return issue.BooleanResult;
    }

    public bool TryReadGuest(ulong address, Span<byte> destination)
    {
        if (_writerIntent > _resolvedWriter) Interlocked.Increment(ref _watermarkStalls);
        var issue = Next(Operation.TryReadGuest);
        issue.TryReadGuest_address = address;
        ReserveBytes(issue, destination.Length);
        Finish(issue, synchronize: true);
        issue.Bytes.AsSpan(0, issue.ByteCount).CopyTo(destination);
        ReleaseBytes(issue);
        return issue.BooleanResult;
    }

    public bool TryReadCommandBlock(ulong address, Span<byte> block, out CommandReadToken token)
    {
        if (_writerIntent > _resolvedWriter) Interlocked.Increment(ref _watermarkStalls);
        var issue = Next(Operation.TryReadCommandBlock);
        issue.TryReadCommandBlock_address = address;
        ReserveBytes(issue, block.Length);
        Finish(issue, synchronize: true);
        issue.Bytes.AsSpan(0, issue.ByteCount).CopyTo(block);
        ReleaseBytes(issue);
        token = issue.TryReadCommandBlock_token;
        return issue.BooleanResult;
    }

    public bool IsCommandBlockCurrent(in CommandReadToken token)
    {
        // The ownership token was copied on B before it acknowledged the prefix.
        // No B work may change it until A publishes another operation/writer intent.
        if (_target is ICommandIssueReadTokenSource)
        {
            if (InvalidateUnknownWriterTokens && _writerIntent > _resolvedWriter)
            {
                Interlocked.Increment(ref _tokenInvalidations);
                return false;
            }
            ResolveWriters();
            return token == _completedReadToken;
        }
        if (_writerIntent > _resolvedWriter) Interlocked.Increment(ref _watermarkStalls);
        var issue = Next(Operation.IsCommandBlockCurrent);
        issue.IsCommandBlockCurrent_token = token;
        Finish(issue, synchronize: true);
        return issue.BooleanResult;
    }

    public bool TryReadGuestOperand(ulong address, Span<byte> destination)
    {
        if (_writerIntent > _resolvedWriter) Interlocked.Increment(ref _watermarkStalls);
        var issue = Next(Operation.TryReadGuestOperand);
        issue.TryReadGuestOperand_address = address;
        ReserveBytes(issue, destination.Length);
        Finish(issue, synchronize: true);
        issue.Bytes.AsSpan(0, issue.ByteCount).CopyTo(destination);
        ReleaseBytes(issue);
        return issue.BooleanResult;
    }

    public bool TryReadGuestWaitOperand(ulong address, Span<byte> destination, bool commitPendingWrite)
    {
        if (_writerIntent > _resolvedWriter) Interlocked.Increment(ref _watermarkStalls);
        var issue = Next(Operation.TryReadGuestWaitOperand);
        issue.TryReadGuestWaitOperand_address = address;
        ReserveBytes(issue, destination.Length);
        issue.TryReadGuestWaitOperand_commitPendingWrite = commitPendingWrite;
        Finish(issue, synchronize: true);
        issue.Bytes.AsSpan(0, issue.ByteCount).CopyTo(destination);
        ReleaseBytes(issue);
        return issue.BooleanResult;
    }

    public bool CommitPendingOperandWrite(ulong address, ulong size)
    {
        if (_writerIntent > _resolvedWriter) Interlocked.Increment(ref _watermarkStalls);
        var issue = Next(Operation.CommitPendingOperandWrite);
        issue.CommitPendingOperandWrite_address = address;
        issue.CommitPendingOperandWrite_size = size;
        Finish(issue, synchronize: true);
        return issue.BooleanResult;
    }

    public bool TryWriteGuest(ulong address, ReadOnlySpan<byte> source, string kind = "WRITE_BYTES")
    {
        var issue = Next(Operation.TryWriteGuest);
        issue.TryWriteGuest_address = address;
        ReserveBytes(issue, source.Length);
        source.CopyTo(issue.Bytes);
        issue.TryWriteGuest_kind = kind;
        Finish(issue, synchronize: true);
        ReleaseBytes(issue);
        return issue.BooleanResult;
    }

    public void ExecuteGuestMemoryOperation(Action operation, string kind = "MEMORY_OPERATION", ulong address = 0, ulong bytes = 0, ulong value = 0)
    {
        var issue = Next(Operation.ExecuteGuestMemoryOperation);
        issue.ExecuteGuestMemoryOperation_operation = operation;
        issue.ExecuteGuestMemoryOperation_kind = kind;
        issue.ExecuteGuestMemoryOperation_address = address;
        issue.ExecuteGuestMemoryOperation_bytes = bytes;
        issue.ExecuteGuestMemoryOperation_value = value;
        Finish(issue, synchronize: true);
    }

    public void ExecuteCommandMemoryOperation(CommandMemoryOperation operation)
    {
        var issue = Next(Operation.ExecuteCommandMemoryOperation);
        issue.ExecuteCommandMemoryOperation_operation = operation;
        // Atomics and semaphore consumes return values to PM4 immediately.
        Finish(issue, synchronize: operation.Kind is CommandMemoryOperation.OperationKind.Atomic or
            CommandMemoryOperation.OperationKind.SemaphoreConsume);
    }

    public void RunPendingCommands()
    {
        if (!_hasRelay()) return;
        var issue = Next(Operation.RunPendingCommands);
        Finish(issue, synchronize: true);
    }

    public void BeginSubmission(int queueId, ulong submissionId, object? geometrySnapshots)
    {
        _currentQueue = queueId; _currentSubmit = submissionId;
        var issue = Next(Operation.BeginSubmission);
        issue.BeginSubmission_queueId = queueId;
        issue.BeginSubmission_submissionId = submissionId;
        issue.BeginSubmission_geometrySnapshots = geometrySnapshots;
        Finish(issue, synchronize: false);
    }

    public void Flush()
    {
        var issue = Next(Operation.Flush);
        Finish(issue, synchronize: false);
    }

    public void FlushFrame()
    {
        var issue = Next(Operation.FlushFrame);
        Finish(issue, synchronize: true);
    }

    public void FlushAndWait()
    {
        var issue = Next(Operation.FlushAndWait);
        Finish(issue, synchronize: true);
    }

    public void SynchronizeGpu()
    {
        var issue = Next(Operation.SynchronizeGpu);
        Finish(issue, synchronize: true);
    }

    public void RunGarbageCollector()
    {
        var issue = Next(Operation.RunGarbageCollector);
        Finish(issue, synchronize: false);
    }

    public void EmitGlobalBarrier()
    {
        var issue = Next(Operation.EmitGlobalBarrier);
        Finish(issue, synchronize: false);
    }

    public void ReleaseShaderWrites()
    {
        var issue = Next(Operation.ReleaseShaderWrites);
        Finish(issue, synchronize: false);
    }

    public void FillBuffer(ulong address, ulong size, uint value, bool isGds)
    {
        var issue = Next(Operation.FillBuffer);
        issue.FillBuffer_address = address;
        issue.FillBuffer_size = size;
        issue.FillBuffer_value = value;
        issue.FillBuffer_isGds = isGds;
        Finish(issue, synchronize: false);
    }

    public void CopyBuffer(ulong destination, ulong source, ulong size, bool destinationIsGds, bool sourceIsGds)
    {
        var issue = Next(Operation.CopyBuffer);
        issue.CopyBuffer_destination = destination;
        issue.CopyBuffer_source = source;
        issue.CopyBuffer_size = size;
        issue.CopyBuffer_destinationIsGds = destinationIsGds;
        issue.CopyBuffer_sourceIsGds = sourceIsGds;
        Finish(issue, synchronize: false);
    }

    public void TraceGuestWrite(string operation, ulong address, ulong size)
    {
        var issue = Next(Operation.TraceGuestWrite);
        issue.TraceGuestWrite_operation = operation;
        issue.TraceGuestWrite_address = address;
        issue.TraceGuestWrite_size = size;
        Finish(issue, synchronize: false);
    }

    public void TraceWritersOf(string subject, ulong address, ulong size)
    {
        var issue = Next(Operation.TraceWritersOf);
        issue.TraceWritersOf_subject = subject;
        issue.TraceWritersOf_address = address;
        issue.TraceWritersOf_size = size;
        Finish(issue, synchronize: false);
    }

    public void ReadGds(Span<uint> destination, uint wordOffset, uint wordCount)
    {
        var issue = Next(Operation.ReadGds);
        ReserveBytes(issue, MemoryMarshal.AsBytes(destination).Length);
        issue.ReadGds_wordOffset = wordOffset;
        issue.ReadGds_wordCount = wordCount;
        Finish(issue, synchronize: true);
        issue.Bytes.AsSpan(0, issue.ByteCount).CopyTo(MemoryMarshal.AsBytes(destination));
        ReleaseBytes(issue);
    }

    public bool DefersEndOfPipeLabel(EndOfPipeWriteKind kind) => _target.DefersEndOfPipeLabel(kind);

    public bool TryReadOcclusionCounter(int queueId, out ulong value)
    {
        var issue = Next(Operation.TryReadOcclusionCounter);
        issue.TryReadOcclusionCounter_queueId = queueId;
        Finish(issue, synchronize: true);
        value = issue.TryReadOcclusionCounter_value;
        return issue.BooleanResult;
    }

    public bool ResetOcclusionCounter(int queueId)
    {
        var issue = Next(Operation.ResetOcclusionCounter);
        issue.ResetOcclusionCounter_queueId = queueId;
        Finish(issue, synchronize: true);
        return issue.BooleanResult;
    }

    public bool DumpOcclusionCounter(int queueId, ulong address, uint instanceMask, uint strideBytes)
    {
        var issue = Next(Operation.DumpOcclusionCounter);
        issue.DumpOcclusionCounter_queueId = queueId;
        issue.DumpOcclusionCounter_address = address;
        issue.DumpOcclusionCounter_instanceMask = instanceMask;
        issue.DumpOcclusionCounter_strideBytes = strideBytes;
        Finish(issue, synchronize: true);
        return issue.BooleanResult;
    }

    public bool HasPendingPublications(int queueId)
    {
        var issue = Next(Operation.HasPendingPublications);
        issue.HasPendingPublications_queueId = queueId;
        Finish(issue, synchronize: true);
        return issue.BooleanResult;
    }

    public bool HasPendingMemoryPublication(ulong address, ulong size)
    {
        var issue = Next(Operation.HasPendingMemoryPublication);
        issue.HasPendingMemoryPublication_address = address;
        issue.HasPendingMemoryPublication_size = size;
        Finish(issue, synchronize: true);
        return issue.BooleanResult;
    }

    public bool TryDeferOrderedWrite(int queueId, ulong address, ReadOnlySpan<byte> source)
    {
        var issue = Next(Operation.TryDeferOrderedWrite);
        issue.TryDeferOrderedWrite_queueId = queueId;
        issue.TryDeferOrderedWrite_address = address;
        ReserveBytes(issue, source.Length);
        source.CopyTo(issue.Bytes);
        Finish(issue, synchronize: true);
        ReleaseBytes(issue);
        return issue.BooleanResult;
    }

    public bool MustHoldForCrossQueueOrder(int queueId)
    {
        var issue = Next(Operation.MustHoldForCrossQueueOrder);
        issue.MustHoldForCrossQueueOrder_queueId = queueId;
        Finish(issue, synchronize: true);
        return issue.BooleanResult;
    }

    public bool TryReadDeferredLabel(int queueId, ulong address, ulong size, out bool deferred, out ulong value)
    {
        var issue = Next(Operation.TryReadDeferredLabel);
        issue.TryReadDeferredLabel_queueId = queueId;
        issue.TryReadDeferredLabel_address = address;
        issue.TryReadDeferredLabel_size = size;
        Finish(issue, synchronize: true);
        deferred = issue.TryReadDeferredLabel_deferred;
        value = issue.TryReadDeferredLabel_value;
        return issue.BooleanResult;
    }

    public bool TryDeferEndOfPipe(int queueId, in EndOfPipeWrite write)
    {
        var issue = Next(Operation.TryDeferEndOfPipe);
        issue.TryDeferEndOfPipe_queueId = queueId;
        issue.TryDeferEndOfPipe_write = write;
        Finish(issue, synchronize: true);
        return issue.BooleanResult;
    }

    public void RecordEndOfPipe(in EndOfPipeWrite write)
    {
        var issue = Next(Operation.RecordEndOfPipe);
        issue.RecordEndOfPipe_write = write;
        Finish(issue, synchronize: false);
    }

    public void RecordCommandProcessorWrite(ulong address, ReadOnlySpan<byte> source, string kind)
    {
        var issue = Next(Operation.RecordCommandProcessorWrite);
        issue.RecordCommandProcessorWrite_address = address;
        ReserveBytes(issue, source.Length);
        source.CopyTo(issue.Bytes);
        issue.RecordCommandProcessorWrite_kind = kind;
        Finish(issue, synchronize: false);
    }

    public void RecordCommandProcessorQwords(ulong address, int count, ulong strideBytes, ulong value, string kind)
    {
        var issue = Next(Operation.RecordCommandProcessorQwords);
        issue.RecordCommandProcessorQwords_address = address;
        issue.RecordCommandProcessorQwords_count = count;
        issue.RecordCommandProcessorQwords_strideBytes = strideBytes;
        issue.RecordCommandProcessorQwords_value = value;
        issue.RecordCommandProcessorQwords_kind = kind;
        Finish(issue, synchronize: false);
    }

    public void FillBufferInOrder(ulong address, ulong size, uint value)
    {
        var issue = Next(Operation.FillBufferInOrder);
        issue.FillBufferInOrder_address = address;
        issue.FillBufferInOrder_size = size;
        issue.FillBufferInOrder_value = value;
        Finish(issue, synchronize: false);
    }

    public void CopyBufferInOrder(ulong destination, ulong source, ulong size)
    {
        var issue = Next(Operation.CopyBufferInOrder);
        issue.CopyBufferInOrder_destination = destination;
        issue.CopyBufferInOrder_source = source;
        issue.CopyBufferInOrder_size = size;
        Finish(issue, synchronize: false);
    }

    public void TriggerInterrupt(int eventId, uint contextId)
    {
        var issue = Next(Operation.TriggerInterrupt);
        issue.TriggerInterrupt_eventId = eventId;
        issue.TriggerInterrupt_contextId = contextId;
        Finish(issue, synchronize: false);
    }

    public bool HasFlipSlot()
    {
        var issue = Next(Operation.HasFlipSlot);
        Finish(issue, synchronize: true);
        return issue.BooleanResult;
    }

    public ulong PrepareFlip(int handle, int index, int flipMode, long flipArgument)
    {
        var issue = Next(Operation.PrepareFlip);
        issue.PrepareFlip_handle = handle;
        issue.PrepareFlip_index = index;
        issue.PrepareFlip_flipMode = flipMode;
        issue.PrepareFlip_flipArgument = flipArgument;
        Finish(issue, synchronize: true);
        return issue.UlongResult;
    }

    public ulong PrepareFlip(int queueId, int handle, int index, int flipMode, long flipArgument)
    {
        var issue = Next(Operation.PrepareFlip2);
        issue.PrepareFlip2_queueId = queueId;
        issue.PrepareFlip2_handle = handle;
        issue.PrepareFlip2_index = index;
        issue.PrepareFlip2_flipMode = flipMode;
        issue.PrepareFlip2_flipArgument = flipArgument;
        Finish(issue, synchronize: true);
        return issue.UlongResult;
    }

    public bool IsFlipDone(int handle, int index)
    {
        var issue = Next(Operation.IsFlipDone);
        issue.IsFlipDone_handle = handle;
        issue.IsFlipDone_index = index;
        Finish(issue, synchronize: true);
        return issue.BooleanResult;
    }

    public void PrepareCpuFlip(int handle, int index, ulong requestId)
    {
        var issue = Next(Operation.PrepareCpuFlip);
        issue.PrepareCpuFlip_handle = handle;
        issue.PrepareCpuFlip_index = index;
        issue.PrepareCpuFlip_requestId = requestId;
        Finish(issue, synchronize: true);
    }

    public bool ConsumePipelineNotReady()
    {
        if (!_dispatchResultPending) return false;
        _dispatchResultPending = false;
        var issue = Next(Operation.ConsumePipelineNotReady);
        Finish(issue, synchronize: true);
        return issue.BooleanResult;
    }

    public bool HasPendingGpuWrite(ulong address, ulong size)
    {
        if (_writerIntent > _resolvedWriter) Interlocked.Increment(ref _watermarkStalls);
        var issue = Next(Operation.HasPendingGpuWrite);
        issue.HasPendingGpuWrite_address = address;
        issue.HasPendingGpuWrite_size = size;
        Finish(issue, synchronize: true);
        return issue.BooleanResult;
    }

    public void NoteIndirectArgumentsReadOnCpu(ulong address, ulong size)
    {
        var issue = Next(Operation.NoteIndirectArgumentsReadOnCpu);
        issue.NoteIndirectArgumentsReadOnCpu_address = address;
        issue.NoteIndirectArgumentsReadOnCpu_size = size;
        Finish(issue, synchronize: false);
    }

    public void OnQueueReset(int queueId)
    {
        var issue = Next(Operation.OnQueueReset);
        issue.OnQueueReset_queueId = queueId;
        Finish(issue, synchronize: false);
    }

    private ulong GetPendingLabelTick()
    {
        var issue = Next(Operation.GetPendingLabelTick);
        Finish(issue, synchronize: true);
        return issue.UlongResult;
    }

    public void Maintenance()
    {
        var issue = Next(Operation.Maintenance);
        Finish(issue, synchronize: false);
        Publish();
    }

    private void BeginIssued(Issue issue)
    {
        if (_renderer is null) _target.BeginSubmission(issue.BeginSubmission_queueId, issue.BeginSubmission_submissionId, issue.BeginSubmission_geometrySnapshots);
        else _renderer.BeginIssuedSubmission(issue.BeginSubmission_queueId, issue.BeginSubmission_submissionId);
    }

    private void ResetIssued(Issue issue)
    {
        if (_renderer is null) _target.OnQueueReset(issue.OnQueueReset_queueId);
        else _renderer.ResetIssuedQueue(issue.OnQueueReset_queueId);
    }

    private void Execute(Issue issue)
    {
        switch (issue.Operation)
        {
            case Operation.Sync: break;
            case Operation.DrawIndexed:
                if (_renderer is null) _target.DrawIndexed(issue.Submit, issue.Indexed);
                else
                {
                    var programs = issue.Indexed.IndexCount == 0 && issue.Indexed.IndirectArgumentsAddress == 0
                        ? default : _planner?.Capture(issue.Banks, compute: false) ?? default;
                    _renderer.DrawIssued(issue.Queue, issue.Submit, issue.Banks, programs, true, issue.Indexed, default);
                }
                break;
            case Operation.DrawAuto:
                if (_renderer is null) _target.DrawAuto(issue.Submit, issue.Auto);
                else
                {
                    var programs = issue.Auto.VertexCount == 0 && issue.Auto.IndirectArgumentsAddress == 0
                        ? default : _planner?.Capture(issue.Banks, compute: false) ?? default;
                    _renderer.DrawIssued(issue.Queue, issue.Submit, issue.Banks, programs, false, default, issue.Auto);
                }
                break;
            case Operation.Dispatch:
                if (_renderer is null) _target.DispatchDirect(issue.Submit, issue.X, issue.Y, issue.Z, issue.Initiator, issue.Indirect);
                else
                {
                    var programs = _planner?.Capture(issue.Banks, compute: true) ?? default;
                    _renderer.DispatchIssued(issue.Queue, issue.Submit, issue.Banks, programs, issue.X, issue.Y, issue.Z, issue.Initiator, issue.Indirect);
                }
                break;
            case Operation.TryRecordLodStatistics: issue.BooleanResult = _target.TryRecordLodStatistics(issue.TryRecordLodStatistics_destination, issue.TryRecordLodStatistics_size, issue.TryRecordLodStatistics_resetCounters); break;
            case Operation.TryReadGuest: issue.BooleanResult = _target.TryReadGuest(issue.TryReadGuest_address, issue.Bytes.AsSpan(0, issue.ByteCount)); break;
            case Operation.TryReadCommandBlock: issue.BooleanResult = _target.TryReadCommandBlock(issue.TryReadCommandBlock_address, issue.Bytes.AsSpan(0, issue.ByteCount), out issue.TryReadCommandBlock_token); break;
            case Operation.IsCommandBlockCurrent: issue.BooleanResult = _target.IsCommandBlockCurrent(issue.IsCommandBlockCurrent_token); break;
            case Operation.TryReadGuestOperand: issue.BooleanResult = _target.TryReadGuestOperand(issue.TryReadGuestOperand_address, issue.Bytes.AsSpan(0, issue.ByteCount)); break;
            case Operation.TryReadGuestWaitOperand: issue.BooleanResult = _target.TryReadGuestWaitOperand(issue.TryReadGuestWaitOperand_address, issue.Bytes.AsSpan(0, issue.ByteCount), issue.TryReadGuestWaitOperand_commitPendingWrite); break;
            case Operation.CommitPendingOperandWrite: issue.BooleanResult = _target.CommitPendingOperandWrite(issue.CommitPendingOperandWrite_address, issue.CommitPendingOperandWrite_size); break;
            case Operation.TryWriteGuest: issue.BooleanResult = _target.TryWriteGuest(issue.TryWriteGuest_address, issue.Bytes.AsSpan(0, issue.ByteCount), issue.TryWriteGuest_kind); break;
            case Operation.ExecuteGuestMemoryOperation: _target.ExecuteGuestMemoryOperation(issue.ExecuteGuestMemoryOperation_operation, issue.ExecuteGuestMemoryOperation_kind, issue.ExecuteGuestMemoryOperation_address, issue.ExecuteGuestMemoryOperation_bytes, issue.ExecuteGuestMemoryOperation_value); break;
            case Operation.ExecuteCommandMemoryOperation: _target.ExecuteCommandMemoryOperation(issue.ExecuteCommandMemoryOperation_operation); break;
            case Operation.RunPendingCommands: _target.RunPendingCommands(); break;
            case Operation.BeginSubmission: BeginIssued(issue); break;
            case Operation.Flush: _target.Flush(); break;
            case Operation.FlushFrame: _target.FlushFrame(); break;
            case Operation.FlushAndWait: _target.FlushAndWait(); break;
            case Operation.SynchronizeGpu: _target.SynchronizeGpu(); break;
            case Operation.RunGarbageCollector: _target.RunGarbageCollector(); break;
            case Operation.EmitGlobalBarrier: _target.EmitGlobalBarrier(); break;
            case Operation.ReleaseShaderWrites: _target.ReleaseShaderWrites(); break;
            case Operation.FillBuffer: _target.FillBuffer(issue.FillBuffer_address, issue.FillBuffer_size, issue.FillBuffer_value, issue.FillBuffer_isGds); break;
            case Operation.CopyBuffer: _target.CopyBuffer(issue.CopyBuffer_destination, issue.CopyBuffer_source, issue.CopyBuffer_size, issue.CopyBuffer_destinationIsGds, issue.CopyBuffer_sourceIsGds); break;
            case Operation.TraceGuestWrite: _target.TraceGuestWrite(issue.TraceGuestWrite_operation, issue.TraceGuestWrite_address, issue.TraceGuestWrite_size); break;
            case Operation.TraceWritersOf: _target.TraceWritersOf(issue.TraceWritersOf_subject, issue.TraceWritersOf_address, issue.TraceWritersOf_size); break;
            case Operation.ReadGds: _target.ReadGds(MemoryMarshal.Cast<byte, uint>(issue.Bytes.AsSpan(0, issue.ByteCount)), issue.ReadGds_wordOffset, issue.ReadGds_wordCount); break;
            case Operation.DefersEndOfPipeLabel: issue.BooleanResult = _target.DefersEndOfPipeLabel(issue.DefersEndOfPipeLabel_kind); break;
            case Operation.TryReadOcclusionCounter: issue.BooleanResult = _target.TryReadOcclusionCounter(issue.TryReadOcclusionCounter_queueId, out issue.TryReadOcclusionCounter_value); break;
            case Operation.ResetOcclusionCounter: issue.BooleanResult = _target.ResetOcclusionCounter(issue.ResetOcclusionCounter_queueId); break;
            case Operation.DumpOcclusionCounter: issue.BooleanResult = _target.DumpOcclusionCounter(issue.DumpOcclusionCounter_queueId, issue.DumpOcclusionCounter_address, issue.DumpOcclusionCounter_instanceMask, issue.DumpOcclusionCounter_strideBytes); break;
            case Operation.HasPendingPublications: issue.BooleanResult = _target.HasPendingPublications(issue.HasPendingPublications_queueId); break;
            case Operation.HasPendingMemoryPublication: issue.BooleanResult = _target.HasPendingMemoryPublication(issue.HasPendingMemoryPublication_address, issue.HasPendingMemoryPublication_size); break;
            case Operation.TryDeferOrderedWrite: issue.BooleanResult = _target.TryDeferOrderedWrite(issue.TryDeferOrderedWrite_queueId, issue.TryDeferOrderedWrite_address, issue.Bytes.AsSpan(0, issue.ByteCount)); break;
            case Operation.MustHoldForCrossQueueOrder: issue.BooleanResult = _target.MustHoldForCrossQueueOrder(issue.MustHoldForCrossQueueOrder_queueId); break;
            case Operation.TryReadDeferredLabel: issue.BooleanResult = _target.TryReadDeferredLabel(issue.TryReadDeferredLabel_queueId, issue.TryReadDeferredLabel_address, issue.TryReadDeferredLabel_size, out issue.TryReadDeferredLabel_deferred, out issue.TryReadDeferredLabel_value); break;
            case Operation.TryDeferEndOfPipe: issue.BooleanResult = _target.TryDeferEndOfPipe(issue.TryDeferEndOfPipe_queueId, issue.TryDeferEndOfPipe_write); break;
            case Operation.RecordEndOfPipe: _target.RecordEndOfPipe(issue.RecordEndOfPipe_write); break;
            case Operation.RecordCommandProcessorWrite: _target.RecordCommandProcessorWrite(issue.RecordCommandProcessorWrite_address, issue.Bytes.AsSpan(0, issue.ByteCount), issue.RecordCommandProcessorWrite_kind); break;
            case Operation.RecordCommandProcessorQwords: _target.RecordCommandProcessorQwords(issue.RecordCommandProcessorQwords_address, issue.RecordCommandProcessorQwords_count, issue.RecordCommandProcessorQwords_strideBytes, issue.RecordCommandProcessorQwords_value, issue.RecordCommandProcessorQwords_kind); break;
            case Operation.FillBufferInOrder: _target.FillBufferInOrder(issue.FillBufferInOrder_address, issue.FillBufferInOrder_size, issue.FillBufferInOrder_value); break;
            case Operation.CopyBufferInOrder: _target.CopyBufferInOrder(issue.CopyBufferInOrder_destination, issue.CopyBufferInOrder_source, issue.CopyBufferInOrder_size); break;
            case Operation.TriggerInterrupt: _target.TriggerInterrupt(issue.TriggerInterrupt_eventId, issue.TriggerInterrupt_contextId); break;
            case Operation.HasFlipSlot: issue.BooleanResult = _target.HasFlipSlot(); break;
            case Operation.PrepareFlip: issue.UlongResult = _target.PrepareFlip(issue.PrepareFlip_handle, issue.PrepareFlip_index, issue.PrepareFlip_flipMode, issue.PrepareFlip_flipArgument); break;
            case Operation.PrepareFlip2: issue.UlongResult = _target.PrepareFlip(issue.PrepareFlip2_queueId, issue.PrepareFlip2_handle, issue.PrepareFlip2_index, issue.PrepareFlip2_flipMode, issue.PrepareFlip2_flipArgument); break;
            case Operation.IsFlipDone: issue.BooleanResult = _target.IsFlipDone(issue.IsFlipDone_handle, issue.IsFlipDone_index); break;
            case Operation.PrepareCpuFlip: _target.PrepareCpuFlip(issue.PrepareCpuFlip_handle, issue.PrepareCpuFlip_index, issue.PrepareCpuFlip_requestId); break;
            case Operation.ConsumePipelineNotReady: issue.BooleanResult = _target.ConsumePipelineNotReady(); break;
            case Operation.HasPendingGpuWrite: issue.BooleanResult = _target.HasPendingGpuWrite(issue.HasPendingGpuWrite_address, issue.HasPendingGpuWrite_size); break;
            case Operation.NoteIndirectArgumentsReadOnCpu: _target.NoteIndirectArgumentsReadOnCpu(issue.NoteIndirectArgumentsReadOnCpu_address, issue.NoteIndirectArgumentsReadOnCpu_size); break;
            case Operation.OnQueueReset: ResetIssued(issue); break;
            case Operation.GetPendingLabelTick: issue.UlongResult = _target.PendingLabelTick; break;
            case Operation.Maintenance: _maintenance?.Invoke(); break;
        }
    }
}
