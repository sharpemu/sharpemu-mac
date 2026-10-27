// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.HLE.DsTrace;
using System.Buffers.Binary;
using System.Diagnostics;
using System.Threading;
using System.Runtime.CompilerServices;

namespace SharpEmu.Libs.Ampr;

/// <summary>
/// The APR completion engine. A submission captures the command buffer on the
/// submitting thread and returns; one worker per priority executes the
/// submissions in order. Completion publishes the commands' effects first,
/// then the result record, then the done state, then wakes
/// sceKernelAprWaitCommandBuffer waiters.
/// </summary>
public static partial class AmprExports
{
    private const int AprPriorityCount = 4;
    private const int AprMaxRetainedCompleted = 4096;
    internal const string AprWaitWakeKey = "sceKernelAprWaitCommandBuffer";

    private sealed class AprSubmission
    {
        public required AprSession Session { get; init; }
        public required uint Id { get; init; }
        public required CapturedCommandBuffer Captured { get; init; }
        public required ICpuMemory Memory { get; init; }
        public required Generation Generation { get; init; }
        public required ulong ResultAddress { get; init; }
        public required bool Retained { get; init; }
        public required long Sequence { get; init; }
        public required ulong DsCaller { get; init; }
        public required ulong DsReturnAddress { get; init; }
        public required long SubmitTimestamp { get; init; }

        // Guarded by _aprGate.
        public bool Done;
        public int ExecutionResult;
        public uint ErrorOffset;
    }

    private sealed class AprWorker
    {
        public readonly Queue<AprSubmission> Pending = new();
        public Thread? Thread;
    }

    private static readonly object _aprGate = new();
    private static readonly Dictionary<uint, AprSubmission> _aprSubmissions = new();
    private static readonly Queue<AprSubmission> _aprCompletedOrder = new();
    private sealed class AprSession : IDisposable
    {
        internal readonly AprWorker[] Workers = [new(), new(), new(), new()];
        internal volatile bool Stopping;
        // Admission and collection leases are guarded by _aprGate.
        internal int ActiveProducers;
        private readonly object _stopGate = new();
        private bool _stopped;

        public void Dispose()
        {
            // Serialize repeated Dispose calls through the entire join, not just the stop flag.
            lock (_stopGate)
            {
                if (_stopped) return;
                lock (_aprGate)
                {
                    Stopping = true;
                    foreach (var worker in Workers)
                    {
                        while (worker.Pending.TryDequeue(out var pending))
                        {
                            pending.ExecutionResult = AprStoppedResult;
                            pending.Done = true;
                        }
                    }
                    Monitor.PulseAll(_aprGate);
                    while (ActiveProducers != 0) Monitor.Wait(_aprGate);
                }
                foreach (var worker in Workers)
                {
                    worker.Thread?.Join();
                }
                lock (_aprGate)
                {
                    foreach (var id in _aprSubmissions.Where(pair => ReferenceEquals(pair.Value.Session, this))
                                 .Select(pair => pair.Key).ToArray())
                        _aprSubmissions.Remove(id);
                    var retained = _aprCompletedOrder.Where(item => !ReferenceEquals(item.Session, this)).ToArray();
                    _aprCompletedOrder.Clear();
                    foreach (var item in retained) _aprCompletedOrder.Enqueue(item);
                }
                // Cache leases protect readers belonging to other independent memories.
                EvictAllCachedHostFiles();
                lock (_aprGate)
                {
                    if (ReferenceEquals(_runtimeAprSession, this)) _runtimeAprSession = null;
                }
                _stopped = true;
            }
        }
    }

    private const int AprStoppedResult = (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND;
    private static readonly ConditionalWeakTable<ICpuMemory, AprSession> _aprSessions = new();
    // Preserve the process-wide worker pool for direct library callers that do not own a runtime.
    private static readonly AprSession _standaloneAprSession = new();
    private static AprSession? _runtimeAprSession;
    [ThreadStatic] private static AprSession? _executingAprSession;
    private static bool AprExecutionStopping => _executingAprSession?.Stopping == true;

    /// <summary>
    /// Owns APR submissions for a runtime's root memory, including future tracked wrappers.
    /// The stopped root association survives disposal, so late wrappers cannot enter the fallback pool.
    /// Dispose after guest dispatch exits and before unmounting app0 or releasing guest memory.
    /// Direct library callers retain the shared standalone worker pool.
    /// </summary>
    public static IDisposable BeginRuntimeSession(ICpuMemory memory)
    {
        lock (_aprGate)
        {
            if (_runtimeAprSession is not null)
                throw new InvalidOperationException("An APR runtime session is already active.");
            var session = new AprSession();
            var owner = AprMemoryOwner(memory);
            _aprSessions.Remove(owner);
            _aprSessions.Add(owner, session);
            return _runtimeAprSession = session;
        }
    }

    private static ICpuMemory AprMemoryOwner(ICpuMemory memory)
    {
        while (memory is ICpuMemoryWrapper wrapper && !ReferenceEquals(memory, wrapper.Inner))
            memory = wrapper.Inner;
        return memory;
    }

    private static void ReleaseAprProducer(AprSession session)
    {
        lock (_aprGate)
        {
            session.ActiveProducers--;
            Monitor.PulseAll(_aprGate);
        }
    }

    private static int _nextAprSubmissionId;
    private static long _aprSequence;

    /// <summary>
    /// Captures and queues a command buffer. With <paramref name="outIdAddress"/>
    /// the submission id is written there and the submission is kept until
    /// sceKernelAprWaitCommandBuffer collects it; without one nothing can wait
    /// for it, so it is dropped when it completes.
    /// </summary>
    internal static int SubmitCommandBuffer(
        CpuContext ctx,
        ulong commandBuffer,
        ulong priority,
        ulong resultAddress,
        ulong outIdAddress,
        ulong dsReturnAddress,
        out uint submissionId)
    {
        submissionId = 0;
        // Resolve wrappers before admission; no guest byte access occurs under the gate.
        var owner = AprMemoryOwner(ctx.Memory);
        AprSession session;
        lock (_aprGate)
        {
            session = _aprSessions.GetValue(owner, static _ => _standaloneAprSession);
            if (session.Stopping) return AprStoppedResult;
            session.ActiveProducers++;
        }
        try
        {
            submissionId = 0;
            var captureResult = CaptureCommandBuffer(ctx, commandBuffer, out var captured);
            if (captureResult != (int)OrbisGen2Result.ORBIS_GEN2_OK)
            {
                return captureResult;
            }

            submissionId = unchecked((uint)Interlocked.Increment(ref _nextAprSubmissionId));
            if (submissionId == 0)
            {
                submissionId = unchecked((uint)Interlocked.Increment(ref _nextAprSubmissionId));
            }

            if (outIdAddress != 0 && !ctx.TryWriteUInt32(outIdAddress, submissionId))
            {
                return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
            }

            var submission = new AprSubmission
            {
                Session = session,
                Id = submissionId,
                Captured = captured!,
                Memory = ctx.Memory,
                Generation = ctx.TargetGeneration,
                ResultAddress = resultAddress,
                Retained = outIdAddress != 0,
                Sequence = Interlocked.Increment(ref _aprSequence),
                DsCaller = DsTrace.CurrentRa1,
                DsReturnAddress = dsReturnAddress,
                SubmitTimestamp = Stopwatch.GetTimestamp(),
            };
            lock (_aprGate)
            {
                if (session.Stopping) return AprStoppedResult;
                if (submission.Retained)
                {
                    _aprSubmissions[submissionId] = submission;
                }

                var worker = session.Workers[(int)Math.Min(priority, AprPriorityCount - 1)];
                worker.Pending.Enqueue(submission);
                if (worker.Thread is null)
                {
                    var index = Array.IndexOf(session.Workers, worker);
                    worker.Thread = new Thread(() => RunAprWorker(session, worker))
                    {
                        IsBackground = true,
                        Name = $"SharpEmu apr p{index}",
                    };
                    worker.Thread.Start();
                }

                Monitor.PulseAll(_aprGate);

            }

            return (int)OrbisGen2Result.ORBIS_GEN2_OK;
        }
        finally
        {
            ReleaseAprProducer(session);
        }
    }

    private static void RunAprWorker(AprSession session, AprWorker worker)
    {
        while (true)
        {
            AprSubmission submission;
            lock (_aprGate)
            {
                while (!session.Stopping && worker.Pending.Count == 0)
                {
                    Monitor.Wait(_aprGate);
                }

                if (session.Stopping) return;
                submission = worker.Pending.Dequeue();
            }

            _executingAprSession = session;
            try { CompleteAprSubmission(submission); }
            finally { _executingAprSession = null; }
        }
    }

    private static void CompleteAprSubmission(AprSubmission submission)
    {
        var dsApr = DsTrace.Io && DsTrace.Armed;
        if (dsApr)
        {
            DsAprAcc.Reset();
        }

        var ctx = new CpuContext(submission.Memory, submission.Generation);
        ExecuteCapturedCommandBuffer(ctx, submission.Captured, out var executionResult, out var errorOffset);
        if (!submission.Session.Stopping && submission.ResultAddress != 0)
        {
            _ = TryWriteAprResult(submission.Memory, submission.ResultAddress, executionResult, errorOffset);
        }

        lock (_aprGate)
        {
            submission.ExecutionResult = executionResult;
            submission.ErrorOffset = errorOffset;
            submission.Done = true;
            if (!submission.Session.Stopping && submission.Retained)
            {
                _aprCompletedOrder.Enqueue(submission);
                // A guest that never waits must not grow the table without bound.
                while (_aprCompletedOrder.Count > AprMaxRetainedCompleted)
                {
                    var oldest = _aprCompletedOrder.Dequeue();
                    if (_aprSubmissions.TryGetValue(oldest.Id, out var current) && ReferenceEquals(current, oldest))
                    {
                        _aprSubmissions.Remove(oldest.Id);
                    }
                }
            }

            Monitor.PulseAll(_aprGate);
        }

        if (!submission.Session.Stopping)
            _ = GuestThreadExecution.Scheduler?.WakeBlockedThreads(AprWaitWakeKey);
        if (dsApr)
        {
            DsTrace.Emit(
                DsEv.APR,
                submission.DsCaller,
                submission.DsReturnAddress,
                (ulong)DsAprAcc.Cmds,
                (ulong)DsTrace.TicksToNs(Stopwatch.GetTimestamp() - submission.SubmitTimestamp),
                (ulong)DsTrace.TicksToNs(DsAprAcc.PreadTicks),
                (ulong)DsTrace.TicksToNs(DsAprAcc.WaitTicks),
                (ulong)DsAprAcc.WaitIters,
                (ulong)DsAprAcc.Bytes);
        }
    }

    private sealed class AprWaiter : IGuestThreadBlockWaiter
    {
        public required CpuContext Ctx { get; init; }
        public required AprSubmission Submission { get; init; }

        public bool TryWake()
        {
            lock (_aprGate)
            {
                return Submission.Done;
            }
        }

        public int Resume() => CollectAprSubmission(Ctx, Submission);
    }

    /// <summary>
    /// sceKernelAprWaitCommandBuffer: parks the caller until the submission
    /// completes, then reports its result and forgets the id.
    /// </summary>
    internal static int WaitCommandBuffer(CpuContext ctx, uint submissionId, out ulong commandBuffer)
    {
        commandBuffer = 0;
        AprSubmission? submission;
        bool done;
        lock (_aprGate)
        {
            if (!_aprSubmissions.TryGetValue(submissionId, out submission))
            {
                return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND;
            }

            commandBuffer = submission.Captured.CommandBuffer;
            done = submission.Done;
        }

        // Collect may write guest memory. Do not retain this outer gate while
        // its own admission lease protects the lifetime of that write.
        if (done) return CollectAprSubmission(ctx, submission);

        if (GuestThreadExecution.RequestCurrentThreadBlock(
                ctx,
                "sceKernelAprWaitCommandBuffer",
                AprWaitWakeKey,
                new AprWaiter { Ctx = ctx, Submission = submission },
                0))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_OK;
        }

        lock (_aprGate)
        {
            while (!submission.Done)
            {
                Monitor.Wait(_aprGate);
            }
        }

        return CollectAprSubmission(ctx, submission);
    }

    private static int CollectAprSubmission(CpuContext ctx, AprSubmission submission)
    {
        int executionResult;
        uint errorOffset;
        lock (_aprGate)
        {
            if (_aprSubmissions.TryGetValue(submission.Id, out var current) && ReferenceEquals(current, submission))
            {
                _aprSubmissions.Remove(submission.Id);
            }
            if (submission.Session.Stopping) return AprStoppedResult;
            submission.Session.ActiveProducers++;
            executionResult = submission.ExecutionResult;
            errorOffset = submission.ErrorOffset;
        }
        try
        {
            if (submission.ResultAddress != 0 &&
                !TryWriteAprResult(ctx.Memory, submission.ResultAddress, executionResult, errorOffset))
            {
                return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
            }
            return (int)OrbisGen2Result.ORBIS_GEN2_OK;
        }
        finally
        {
            ReleaseAprProducer(submission.Session);
        }
    }

    private static bool TryWriteAprResult(ICpuMemory memory, ulong resultAddress, int executionResult, uint errorOffset)
    {
        Span<byte> result = stackalloc byte[sizeof(ulong)];
        BinaryPrimitives.WriteInt32LittleEndian(result, executionResult);
        BinaryPrimitives.WriteUInt32LittleEndian(result[sizeof(int)..], errorOffset);
        return memory.TryWrite(resultAddress, result);
    }
}
