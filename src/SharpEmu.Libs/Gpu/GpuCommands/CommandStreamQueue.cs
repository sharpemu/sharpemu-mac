// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.HLE.DsTrace;
using SharpEmu.Libs.VideoOut;

namespace SharpEmu.Libs.Gpu.GpuCommands;

public enum IdleOutcome
{
    Completed,
    Cancelled,
    Failed,
}

public readonly record struct BlockedSnapshot(int Outstanding, double OldestAgeMilliseconds, ulong SampleWaitAddress, int SampleQueueId);

public enum SliceResult
{
    NoWork,
    AllBlocked,
    Progressed,
    BlockedWithoutProgress,
    Completed,
}

// Submissions of every queue; one thread runs the slices, other threads enqueue and wait.
public sealed class CommandStreamQueue
{
    private sealed record ControlBarrier(ulong CutoffOrdinal, Action Work, Action Cancel);

    public const int ComputeQueueCount = 56;
    public const int QueueCount = 1 + ComputeQueueCount;
    public const int AllBlockedRetryMilliseconds = 100;
    // How many frame boundaries the guest may be ahead of the worker. One by default;
    // explicit zero keeps the synchronous diagnostic mode, and a larger number is the
    // knob for a title that calls submitDone several times a frame and spends the frame
    // blocked in it - Poppy blocks for the whole wall clock across about seven calls a
    // frame while the worker sits idle for sixty per cent of it.
    public static readonly int DefaultBoundariesInFlight = ReadBoundariesInFlight();

    private static int ReadBoundariesInFlight()
    {
        var text = Environment.GetEnvironmentVariable("SHARPEMU_SUSPEND_POINTS_IN_FLIGHT");
        return !string.IsNullOrWhiteSpace(text) && int.TryParse(text, out var value) && value >= 0
            ? Math.Min(value, 64)
            : 1;
    }

    /// <summary>
    /// How often a worker with nothing to run re-reads the labels its parked heads wait on.
    /// </summary>
    /// <remarks>
    /// A label the guest's CPU writes reaches no notification: the queue hears of GPU
    /// completions and of its own writes, not of a store from a game thread, so such a
    /// wait used to sleep out the 100 ms blocked-retry timer. A watched wait is re-read on
    /// this interval instead and woken once its bytes change (and, for a compare, pass).
    /// SHARPEMU_CP_WATCH_INTERVAL_MS sets it; 0 turns the watch off.
    /// </remarks>
    public static readonly int WatchIntervalMilliseconds = ReadWatchInterval();

    private static int ReadWatchInterval()
    {
        var text = Environment.GetEnvironmentVariable("SHARPEMU_CP_WATCH_INTERVAL_MS");
        return !string.IsNullOrWhiteSpace(text) && int.TryParse(text, out var value) && value >= 0
            ? Math.Min(value, AllBlockedRetryMilliseconds)
            : 1;
    }

    /// <summary>Watched waits re-read since startup.</summary>
    public ulong WatchRechecks { get; private set; }

    /// <summary>Watched waits a re-read found changed and passing, and so woke.</summary>
    public ulong WatchWakes { get; private set; }

    private readonly ICommandStreamHost _host;
    private readonly object _gate = new();
    private readonly LinkedList<CommandSubmission>[] _queues = new LinkedList<CommandSubmission>[QueueCount];
    private readonly GpuCommandInterpreter?[] _interpreters = new GpuCommandInterpreter?[QueueCount];
    // Shared by every queue's interpreter: one slice runs at a time, so they never race.
    private readonly ulong[] _syntheticOcclusionCounters = new ulong[4];
    private readonly Queue<ControlBarrier> _controlBarriers = new();
    private int _nextQueue;
    private int _submissionCount;
    private ulong _admissionOrdinal;
    private bool _accepting = true;
    private bool _stopping;
    private bool _processing;
    private bool _graphicsDone;
    private ulong _submitId;
    private ulong _lastCompletedGpuTick;
    private ulong _resourceReadReadySerial;
    private int _doneCount;
    private int _pendingBoundaries;
    private IdleOutcome _outcome = IdleOutcome.Completed;
    private Thread? _processingThread;
    private CommandSubmissionPreparation? _activePreparation;
    private Exception? _preparationFailure;

    private readonly int _boundariesInFlight;

    // Printed before a fatal on the worker thread ends the process (GpuFatal.BeforeFatal):
    // the DCB hash re-check and SHARPEMU_GPU_CLOBBER_DIAG. Null when both are off.
    private readonly Action? _beforeFatal;
    private CommandSubmission? _fatalSubmission;
    private GpuCommandInterpreter? _fatalProcessor;

    // DSTRACE: the submission whose slice the calling (worker) thread runs, for MCOMMIT.
    [ThreadStatic] public static ulong DsCurrentSubmissionId;

    // DSTRACE WIDLE: what made the worker wait, captured by WaitForWork under the gate
    // it already holds. Written and read only by the thread that calls WaitForWork.
    public bool DsIdleHadPending { get; private set; }

    public BlockedSnapshot DsIdleSnapshot { get; private set; }

    public CommandStreamQueue(ICommandStreamHost host) : this(host, DefaultBoundariesInFlight)
    {
    }

    public CommandStreamQueue(ICommandStreamHost host, int boundariesInFlight)
    {
        _host = host;
        _boundariesInFlight = Math.Max(0, boundariesInFlight);
        if (GpuStreamDiagnostics.DcbHashEnabled || GpuClobberDiagnostics.Enabled)
        {
            _beforeFatal = ReportBeforeFatal;
        }
        for (var index = 0; index < QueueCount; index++)
        {
            _queues[index] = new LinkedList<CommandSubmission>();
        }
    }

    public ulong SubmissionsStarted { get; private set; }

    public ulong SlicesRun { get; private set; }

    public ulong BlockedRetries { get; private set; }

    public int FrameNumber => Volatile.Read(ref _doneCount);

    public bool IsStopping
    {
        get
        {
            lock (_gate)
            {
                return _stopping;
            }
        }
    }

    public bool HasPending
    {
        get
        {
            lock (_gate)
            {
                return _submissionCount != 0 || _controlBarriers.Count != 0;
            }
        }
    }

    public int PendingSubmissionCount
    {
        get
        {
            lock (_gate)
            {
                return _submissionCount;
            }
        }
    }

    public bool HasUnblockedPending
    {
        get
        {
            lock (_gate)
            {
                return HasRunnableWorkLocked();
            }
        }
    }

    public GpuCommandInterpreter GetInterpreter(int queueId)
    {
        if (queueId < 0 || queueId >= QueueCount)
        {
            throw _host.Fatal($"The queue id is outside the queue table: queue={queueId}.");
        }

        lock (_gate)
        {
            if (_interpreters[queueId] is { } existing)
            {
                return existing;
            }

            var created = new GpuCommandInterpreter(
                _host, queueId, queueId == 0 ? 0 : GpuCommandInterpreter.ComputeQueueBase + queueId - 1)
            {
                GuestMemoryWritten = NotifyGuestMemoryWritten,
                PendingLabelRecorded = NotePendingLabel,
                PendingQwordsRecorded = NotePendingQwords,
                PendingLabelLookup = TryGetPendingLabel,
                SyntheticOcclusionCounterBank = _syntheticOcclusionCounters,
            };
            _interpreters[queueId] = created;
            return created;
        }
    }

    public void EnqueueGraphics(ulong address, uint dwordCount, ulong submissionId, object? geometrySnapshots)
    {
        if (dwordCount == 0)
        {
            return;
        }

        lock (_gate)
        {
            var submission = new CommandSubmission(CommandSubmissionKind.Graphics, 0, address, dwordCount, submissionId, geometrySnapshots)
            {
                ResetInterpreter = _graphicsDone,
            };
            _graphicsDone = false;
            EnqueueLocked(submission);
        }
    }

    // Compute queues are addressed 0x20 to 0x57; queue id 1 is the first of them.
    public void EnqueueCompute(uint queue, ulong address, uint dwordCount, ulong submissionId, object? geometrySnapshots)
    {
        if (queue < GpuCommandInterpreter.ComputeQueueBase || queue >= GpuCommandInterpreter.ComputeQueueBase + ComputeQueueCount)
        {
            throw _host.Fatal($"The compute queue is outside the supported range: queue=0x{queue:X}.");
        }

        if (dwordCount == 0)
        {
            throw _host.Fatal($"A compute submission is empty: queue=0x{queue:X} address=0x{address:X16}.");
        }

        lock (_gate)
        {
            EnqueueLocked(new CommandSubmission(CommandSubmissionKind.Compute, (int)(queue - GpuCommandInterpreter.ComputeQueueBase) + 1, address, dwordCount, submissionId, geometrySnapshots));
        }
    }

    // Preparation is optional; old callers preserve admission/return behavior.
    public CommandSubmissionPreparation EnqueueWithPreparation(uint queue, ulong address, uint dwordCount, ulong submissionId, object? geometrySnapshots)
    {
        if (queue != 0 && (queue < GpuCommandInterpreter.ComputeQueueBase || queue >= GpuCommandInterpreter.ComputeQueueBase + ComputeQueueCount))
            throw _host.Fatal($"The compute preparation queue is outside the supported range: queue=0x{queue:X}.");
        if (queue != 0 && dwordCount == 0)
            throw _host.Fatal($"A compute submission is empty: queue=0x{queue:X} address=0x{address:X16}.");
        var queueId = queue == 0 ? 0 : checked((int)(queue - GpuCommandInterpreter.ComputeQueueBase) + 1);
        lock (_gate)
        {
            var receipt = new CommandSubmissionPreparation(this, queueId, submissionId);
            if (dwordCount == 0)
            {
                if (!_accepting) throw new OperationCanceledException("Command preparation admission is closed.");
                receipt.Resolve(CommandSubmissionPreparation.State.PreparedPrefix);
                return receipt;
            }
            var submission = new CommandSubmission(queue == 0 ? CommandSubmissionKind.Graphics : CommandSubmissionKind.Compute,
                queueId, address, dwordCount, submissionId, geometrySnapshots)
            { Preparation = receipt, ResetInterpreter = queue == 0 && _graphicsDone };
            EnqueueLocked(submission);
            if (queue == 0) _graphicsDone = false;
            return receipt;
        }
    }

    internal GuestSubmissionPreparationResult WaitForPreparation(CommandSubmissionPreparation receipt)
    {
        Exception? failure = null;
        IGuestSubmissionPrefixCompletion? completion = null;
        var result = GuestSubmissionPreparationResult.DeferredDependency;
        lock (_gate)
        {
            if (_processing && ReferenceEquals(_processingThread, Thread.CurrentThread))
                failure = new InvalidOperationException("Command preparation cannot wait on its own command worker.");
            else
            {
                var deadline = Environment.TickCount64 + 30_000;
                while (receipt.Status == CommandSubmissionPreparation.State.Pending && _outcome == IdleOutcome.Completed)
                {
                    // CPU/other-queue dependencies must remain able to be submitted.
                    // Deferral certifies no shader/GPU work beyond the parsed prefix.
                    if (!_processing && _submissionCount != 0 && !HasRunnableWorkLocked())
                    { receipt.Resolve(CommandSubmissionPreparation.State.DeferredDependency); break; }
                    var remaining = deadline - Environment.TickCount64;
                    if (remaining <= 0)
                    { failure = new TimeoutException("Accepted command preparation did not reach a runnable-prefix boundary."); break; }
                    Monitor.Wait(_gate, (int)Math.Min(remaining, 1000));
                }
                failure ??= receipt.Failure ?? _preparationFailure;
                if (failure is null && receipt.Status == CommandSubmissionPreparation.State.Failed)
                    failure = new InvalidOperationException("Accepted command preparation failed before readiness.");
                if (failure is null && receipt.Status is CommandSubmissionPreparation.State.Pending or CommandSubmissionPreparation.State.Cancelled)
                    failure = new OperationCanceledException("Accepted command preparation was cancelled before readiness.");
            }
            if (receipt.Status == CommandSubmissionPreparation.State.PreparedPrefix)
            { result = GuestSubmissionPreparationResult.PreparedPrefix; completion = receipt.Completion; }
        }
        if (failure is not null)
        {
            if (failure is not OperationCanceledException)
                try { Fail(failure); }
                catch (Exception cleanup)
                { try { failure.Data["CommandPrefixWaitCleanup"] = cleanup; } catch { } }
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(failure).Throw();
        }
        // Both AGC gates and this queue gate are free. Future queue slices stay
        // runnable while this exact caller waits for its accepted native prefix.
        try { completion?.WaitUntilCompleted(); }
        catch (Exception original)
        {
            try { Fail(original); }
            catch (Exception cleanup)
            { try { original.Data["CommandPrefixWaitCleanup"] = cleanup; } catch { } }
            throw;
        }
        return result;
    }

    // Queues a video-out export flip behind the graphics submissions accepted before it.
    public bool TryEnqueueFlipPreparation(int handle, int index, ulong requestId)
    {
        lock (_gate)
        {
            if (!_accepting)
            {
                return false;
            }

            EnqueueLocked(new CommandSubmission(CommandSubmissionKind.FlipPreparation, 0, 0, 0, 0, null)
            {
                FlipHandle = handle,
                FlipIndex = index,
                FlipRequestId = requestId,
            });
            return true;
        }
    }

    // Runs inside GpuFatal on the worker, before the process fails fast: an exception filter
    // or catch above the fatal never gets to run. Logs a rewritten DCB and the clobber report.
    private void ReportBeforeFatal()
    {
        var submission = _fatalSubmission;
        if (submission is null)
        {
            return;
        }

        GpuStreamDiagnostics.Verify(_host, submission, "fatal");
        if (GpuClobberDiagnostics.Enabled)
        {
            _fatalProcessor?.ReportClobber(submission, DescribeQueuesForClobber());
        }
    }

    private string DescribeQueuesForClobber()
    {
        if (!Monitor.TryEnter(_gate, 250))
        {
            return "unavailable=gate-busy";
        }

        try
        {
            var computeSubmissions = 0;
            var computeQueuesBusy = 0;
            for (var queue = 1; queue < QueueCount; queue++)
            {
                var count = _queues[queue].Count;
                computeSubmissions += count;
                computeQueuesBusy += count == 0 ? 0 : 1;
            }

            return $"graphics={_queues[0].Count} compute={computeSubmissions} computeQueuesBusy={computeQueuesBusy} " +
                $"total={_submissionCount} pendingBoundaries={_pendingBoundaries} boundariesInFlight={_boundariesInFlight} " +
                $"frame={_doneCount} pendingLabels={_pendingLabels.Count}";
        }
        finally
        {
            Monitor.Exit(_gate);
        }
    }

    private void EnqueueLocked(CommandSubmission submission)
    {
        if (!_accepting)
        {
            if (_outcome != IdleOutcome.Failed)
            {
                throw new OperationCanceledException("The command stream has stopped accepting submissions.");
            }

            throw _host.Fatal($"The command stream no longer accepts submissions: queue={submission.QueueId} address=0x{submission.Address:X16}.");
        }

        submission.AdmissionOrdinal = ++_admissionOrdinal;
        if (GpuClobberDiagnostics.Enabled)
        {
            submission.EnqueuedTicks = System.Diagnostics.Stopwatch.GetTimestamp();
        }

        if (GpuStreamDiagnostics.DcbHashEnabled && submission.DwordCount != 0 &&
            submission.Kind is CommandSubmissionKind.Graphics or CommandSubmissionKind.Compute &&
            GpuStreamDiagnostics.TryHashGuest(_host, submission.Address, submission.DwordCount, out var submitHash))
        {
            submission.DcbHash = submitHash;
            submission.DcbHashed = true;
        }

        _queues[submission.QueueId].AddLast(submission);
        _submissionCount++;
        if (submission.Kind is not (CommandSubmissionKind.FlipPreparation or CommandSubmissionKind.FrameBoundary))
            SubmissionFlowProfile.Record(SubmissionFlowProfile.EventKind.Enqueued, submission.QueueId,
                submission.SubmissionId, submission.Address, submission.DwordCount, _submissionCount);
        if (DsTrace.Gpu && DsTrace.Armed)
        {
            DsTrace.Emit(DsEv.QENQ, (ulong)submission.QueueId, submission.SubmissionId,
                (ulong)submission.Kind, submission.DwordCount);
        }

        Monitor.PulseAll(_gate);
    }

    public bool TryEnqueueControlBarrier(Action work, Action cancel)
    {
        ArgumentNullException.ThrowIfNull(work);
        ArgumentNullException.ThrowIfNull(cancel);
        lock (_gate)
        {
            if (!_accepting)
            {
                return false;
            }

            _controlBarriers.Enqueue(new ControlBarrier(_admissionOrdinal, work, cancel));
            Monitor.PulseAll(_gate);
            return true;
        }
    }

    public IdleOutcome Done()
    {
        if (_processingThread == Thread.CurrentThread)
        {
            lock (_gate)
            {
                _graphicsDone = true;
                _doneCount++;
            }

            return IdleOutcome.Completed;
        }

        lock (_gate)
        {
            if (_outcome != IdleOutcome.Completed)
            {
                return _outcome;
            }

            _graphicsDone = true;
            if (!_accepting)
            {
                // Shutdown: no slice will run a queued boundary, so drain as before.
                while (_outcome == IdleOutcome.Completed && (_processing || _submissionCount != 0))
                {
                    Monitor.Wait(_gate);
                }

                _doneCount++;
                return _outcome;
            }

            _pendingBoundaries++;
            EnqueueLocked(new CommandSubmission(CommandSubmissionKind.FrameBoundary, 0, 0, 0, 0, null));
            var dsSuspend = DsTrace.Gpu && DsTrace.Armed;
            var dsPending = _pendingBoundaries;
            var dsStarted = dsSuspend ? System.Diagnostics.Stopwatch.GetTimestamp() : 0L;
            while (_outcome == IdleOutcome.Completed && _pendingBoundaries > _boundariesInFlight)
            {
                Monitor.Wait(_gate);
            }

            if (dsSuspend)
            {
                // DSTRACE SUSP: how long the guest's frame boundary blocked on the
                // in-flight boundaries, and how many were pending (this one included).
                DsTrace.Emit(DsEv.SUSP,
                    (ulong)DsTrace.TicksToNs(System.Diagnostics.Stopwatch.GetTimestamp() - dsStarted),
                    (ulong)dsPending);
            }

            return _outcome;
        }
    }

    // Returns once nothing is pending; a cancelled or failed queue reports that instead.
    public IdleOutcome WaitForIdle()
    {
        lock (_gate)
        {
            while (_outcome == IdleOutcome.Completed &&
                   (_processing || _submissionCount != 0 || _controlBarriers.Count != 0))
            {
                Monitor.Wait(_gate);
            }

            return _outcome;
        }
    }

    public void Wake()
    {
        lock (_gate)
        {
            Monitor.PulseAll(_gate);
        }
    }

    // Waits until a submission can run or the timeout passes; returns whether one can run.
    public bool WaitForWork(int timeoutMilliseconds)
    {
        lock (_gate)
        {
            if (DsTrace.Gpu && DsTrace.Armed)
            {
                DsIdleHadPending = _submissionCount != 0 || _controlBarriers.Count != 0;
                DsIdleSnapshot = DsIdleHadPending ? SnapshotBlockedLocked() : default;
            }

            if (HasRunnableWorkLocked() || _stopping)
            {
                return HasRunnableWorkLocked();
            }

            WaitWatchingLocked(timeoutMilliseconds);
            return HasRunnableWorkLocked();
        }
    }

    // Caller holds the gate. Sleeps up to the timeout, cut into watch intervals while a
    // parked head waits on a watched label, and returns early once a re-read woke one.
    private void WaitWatchingLocked(int timeoutMilliseconds)
    {
        if (WatchIntervalMilliseconds == 0 || !HasWatchedHeadLocked())
        {
            Monitor.Wait(_gate, timeoutMilliseconds);
            return;
        }

        var deadline = Environment.TickCount64 + timeoutMilliseconds;
        while (!HasRunnableWorkLocked() && !_stopping)
        {
            var remaining = deadline - Environment.TickCount64;
            if (remaining <= 0)
            {
                return;
            }

            Monitor.Wait(_gate, (int)Math.Min(remaining, WatchIntervalMilliseconds));
            if (RecheckWatchedLocked())
            {
                return;
            }
        }
    }

    private bool HasWatchedHeadLocked()
    {
        foreach (var queue in _queues)
        {
            if (queue.First is { } head && head.Value.Blocked && head.Value.Wait.Watched)
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>Re-reads every parked watched wait once and wakes those whose bytes changed and pass.</summary>
    public bool RecheckWatched()
    {
        lock (_gate)
        {
            return RecheckWatchedLocked();
        }
    }

    // Caller holds the gate. The read is the raw bytes, without waiting for the GPU: a
    // value that changed is only a reason to run the packet again, which reads it properly.
    private bool RecheckWatchedLocked()
    {
        var woke = false;
        Span<byte> bytes = stackalloc byte[sizeof(ulong)];
        foreach (var queue in _queues)
        {
            if (queue.First is not { } node || !node.Value.Blocked || !node.Value.Wait.Watched)
            {
                continue;
            }

            var head = node.Value;
            var wait = head.Wait;
            var size = wait.Size == sizeof(ulong) ? sizeof(ulong) : sizeof(uint);
            WatchRechecks++;
            if (!_host.Memory.TryRead(wait.Address, bytes[..size]))
            {
                continue;
            }

            var value = size == sizeof(ulong)
                ? System.Buffers.Binary.BinaryPrimitives.ReadUInt64LittleEndian(bytes)
                : System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(bytes);
            if (value == head.WatchedValue)
            {
                continue;
            }

            head.WatchedValue = value;
            if (wait.Compares &&
                (!Packets.WaitOperation.TryCompare(value, wait.Reference, wait.Mask, wait.Function, out var satisfied) || !satisfied))
            {
                continue;
            }

            head.Blocked = false;
            WatchWakes++;
            woke = true;
        }

        if (woke)
        {
            _wakeGeneration++;
            Monitor.PulseAll(_gate);
        }

        return woke;
    }

    // Waits the whole interval unless a submission becomes runnable; stopping does not cut it short.
    public bool WaitForRetryInterval(int intervalMilliseconds)
    {
        lock (_gate)
        {
            var deadline = Environment.TickCount64 + intervalMilliseconds;
            while (!HasRunnableWorkLocked())
            {
                var remaining = deadline - Environment.TickCount64;
                if (remaining <= 0)
                {
                    return false;
                }

                if (WatchIntervalMilliseconds != 0 && HasWatchedHeadLocked())
                {
                    Monitor.Wait(_gate, (int)Math.Min(remaining, WatchIntervalMilliseconds));
                    _ = RecheckWatchedLocked();
                }
                else
                {
                    Monitor.Wait(_gate, (int)remaining);
                }
            }

            return true;
        }
    }

    // Publishes a failure: nothing runs any more and every waiter is released with Failed.
    public void Fail(Exception? original = null)
    {
        Action[] cancellations;
        lock (_gate)
        {
            _preparationFailure ??= original;
            FailLocked();
            cancellations = CancelControlBarriersLocked();
        }

        Exception? cleanupFailure = null;
        foreach (var cancel in cancellations)
        {
            try { cancel(); }
            catch (Exception cleanup)
            {
                if (original is null) cleanupFailure ??= cleanup;
                else
                {
                    try
                    {
                        if (original.Data["CommandCancellationCleanup"] is Exception previous)
                            original.Data["CommandCancellationCleanup"] = new AggregateException(previous, cleanup);
                        else original.Data["CommandCancellationCleanup"] = cleanup;
                    }
                    catch { /* Diagnostic attachment cannot replace the first failure. */ }
                }
            }
        }
        if (cleanupFailure is not null)
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(cleanupFailure).Throw();
    }

    private void FailLocked()
    {
        _outcome = IdleOutcome.Failed;
        _accepting = false;
        _stopping = true;
        DropAllLocked();
        _activePreparation?.Resolve(CommandSubmissionPreparation.State.Failed, _preparationFailure);
        _activePreparation = null;
        _processing = false;
        Monitor.PulseAll(_gate);
    }

    // A completed GPU tick can make a blocked memory wait ready. A flip slot or a
    // presented flip is not something a tick decides, so those heads stay asleep.
    // Labels a queue recorded for the GPU to write and the tick they land by, keyed by
    // address. A wait on another queue reads them as written: async compute waiting on a
    // graphics fence would otherwise see it only once the GPU had written it and the guest
    // had been told, and the guest recycles a fence slot as soon as it is told - by then
    // the waiter reads someone else's bytes and never sees the value it was waiting for.
    private readonly Dictionary<ulong, (ulong Value, int Size, ulong Tick)> _pendingLabels = new();

    // Guards _pendingLabels alone, so recording a label does not contend on _gate with the
    // guest threads that submit through it. Taken inside _gate where both are held, never
    // the other way round.
    private readonly object _labelGate = new();

    // Whether some parked head may be blocked. Raised when a slice parks one (under _gate) and
    // lowered only by a scan that finds none (under _gate). Slices never overlap, and a label
    // is recorded from inside one, so no head can park while this is read: a clear hint means
    // nothing is blocked now, and a wake would have nothing to wake. Blocked heads only ever
    // become unblocked behind the reader's back, which a stale raised hint merely over-reports.
    private volatile bool _blockedHeadsPossible;

    /// <summary>Label-wake skips that the verify mode found a blocked head behind; zero is the invariant.</summary>
    internal static long LabelWakeVerifyMismatches;

    private readonly record struct AddressRange(ulong Address, ulong Size);

    private readonly record struct QwordRun(ulong Address, int Count, ulong Stride)
    {
        public bool TouchedBy(CommandWaitCondition wait)
        {
            for (var index = 0; index < Count; index++)
            {
                if (wait.TouchedBy(Address + (ulong)index * Stride, sizeof(ulong)))
                {
                    return true;
                }
            }

            return false;
        }
    }

    private void NotePendingLabel(ulong address, int size, ulong value)
    {
        var tick = _host.PendingLabelTick;
        if (tick == 0)
        {
            return;
        }

        // Published before any wait is evaluated: the dictionary is updated first, and the
        // wake, when there is anything to wake, follows it.
        lock (_labelGate)
        {
            _pendingLabels[address] = (value, size, tick);
        }

        if (!_blockedHeadsPossible)
        {
            VerifyNothingBlocked();
            return;
        }

        lock (_gate)
        {
            WakeMatchingLocked(new AddressRange(address, (ulong)size),
                static (range, wait) => wait.Kind == CommandWaitKind.Unknown || wait.TouchedBy(range.Address, range.Size),
                DsWakeSource.PendingLabel);
            LowerBlockedHintLocked();
        }
    }

    /// <summary>
    /// Records <paramref name="count"/> eight-byte pending labels at a fixed stride as one
    /// dictionary update and one wake pass, instead of one of each per label.
    /// </summary>
    private void NotePendingQwords(ulong address, int count, ulong strideBytes, ulong value)
    {
        var tick = _host.PendingLabelTick;
        if (tick == 0 || count <= 0)
        {
            return;
        }

        lock (_labelGate)
        {
            for (var index = 0; index < count; index++)
            {
                _pendingLabels[address + (ulong)index * strideBytes] = (value, sizeof(ulong), tick);
            }
        }

        if (!_blockedHeadsPossible)
        {
            VerifyNothingBlocked();
            return;
        }

        lock (_gate)
        {
            WakeMatchingLocked(new QwordRun(address, count, strideBytes),
                static (run, wait) => wait.Kind == CommandWaitKind.Unknown || run.TouchedBy(wait),
                DsWakeSource.PendingLabel);
            LowerBlockedHintLocked();
        }
    }

    // Caller holds _gate.
    private void LowerBlockedHintLocked()
    {
        foreach (var queue in _queues)
        {
            if (queue.First is { Value.Blocked: true })
            {
                return;
            }
        }

        _blockedHeadsPossible = false;
    }

    // SHARPEMU_GPU_CP_WRITE_BATCH_VERIFY: the skipped wake would have woken nothing.
    private void VerifyNothingBlocked()
    {
        if (!CommandProcessorWriteBatching.Verify)
        {
            return;
        }

        lock (_gate)
        {
            foreach (var queue in _queues)
            {
                if (queue.First is { Value.Blocked: true })
                {
                    Interlocked.Increment(ref LabelWakeVerifyMismatches);
                    Console.Error.WriteLine("[GPU][VERIFY] A pending label skipped its wake while a head was blocked.");
                    return;
                }
            }
        }
    }

    private bool TryGetPendingLabel(ulong address, bool is64Bit, out ulong value)
    {
        lock (_labelGate)
        {
            if (_pendingLabels.TryGetValue(address, out var label) && label.Size >= (is64Bit ? 8 : 4))
            {
                value = is64Bit ? label.Value : (uint)label.Value;
                return true;
            }

            // A 32-bit wait on the upper half of a 64-bit label.
            if (!is64Bit && address >= 4 && _pendingLabels.TryGetValue(address - 4, out label) && label.Size == 8)
            {
                value = (uint)(label.Value >> 32);
                return true;
            }

            value = 0;
            return false;
        }
    }

    // The labels one completion lands, reused across ticks; only touched under _labelGate.
    private readonly List<ulong> _landedLabels = [];

    public void NotifyCompletedGpuTick(ulong completedTick)
    {
        lock (_gate)
        {
            if (completedTick <= _lastCompletedGpuTick)
            {
                return;
            }

            _lastCompletedGpuTick = completedTick;
            lock (_labelGate)
            {
                if (_pendingLabels.Count != 0)
                {
                    // A landed label is in memory now, where the guest may already have reused it.
                    foreach (var (address, label) in _pendingLabels)
                    {
                        if (label.Tick <= completedTick)
                        {
                            _landedLabels.Add(address);
                        }
                    }

                    foreach (var address in _landedLabels)
                    {
                        _pendingLabels.Remove(address);
                    }

                    _landedLabels.Clear();
                }
            }

            // A completion lands GPU writes; engine counters only move when their own
            // queue runs a packet, so a counter wait has nothing to re-check here.
            WakeMatchingLocked(wait => wait.Kind is CommandWaitKind.GuestMemory or CommandWaitKind.Unknown,
                DsWakeSource.GpuTick);
        }
    }

    /// <summary>
    /// Wakes blocked heads other than serial dependencies after a host finished publishing deferred writes. A slice
    /// running at the time sees the wake generation change and stays runnable at its park.
    /// </summary>
    public void NotifyPublicationCompleted()
    {
        lock (_gate)
        {
            _wakeGeneration++;
            var retried = false;
            foreach (var queue in _queues)
            {
                if (queue.First is { Value.Blocked: true } head &&
                    head.Value.Wait.Kind != CommandWaitKind.GpuSerial)
                {
                    head.Value.Blocked = false;
                    retried = true;
                }
            }

            if (retried)
            {
                BlockedRetries++;
            }

            Monitor.PulseAll(_gate);
        }
    }

    /// <summary>
    /// Wakes blocked heads except serial dependencies, which have an exact completion
    /// callback. This is the fallback: it is what a timer does when
    /// nothing told us which wait could have changed, and what a guest CPU write we
    /// cannot observe is eventually caught by.
    /// </summary>
    public void RetryBlocked()
    {
        lock (_gate)
        {
            var ds = DsTrace.Gpu && DsTrace.Armed;
            var dsWoken = 0UL;
            var dsOldest = 0L;
            foreach (var queue in _queues)
            {
                if (queue.First is { } head && head.Value.Wait.Kind != CommandWaitKind.GpuSerial)
                {
                    if (ds && head.Value.Blocked)
                    {
                        dsWoken++;
                        if (dsOldest == 0 || head.Value.BlockedSinceTicks < dsOldest)
                        {
                            dsOldest = head.Value.BlockedSinceTicks;
                        }
                    }

                    head.Value.Blocked = false;
                }
            }

            if (ds && dsWoken != 0)
            {
                DsEmitWake(DsWakeSource.RetryBlocked, dsWoken, dsOldest);
            }

            BlockedRetries++;
            if (PacketProfile.Enabled)
            {
                PacketProfile.RecordRetryAll();
            }

            Monitor.PulseAll(_gate);
        }
    }

    /// <summary>How many heads a targeted wake has unblocked since startup.</summary>
    public ulong TargetedWakes { get; private set; }

    /// <summary>How many blocked heads a targeted wake has left alone.</summary>
    public ulong TargetedWakesSkipped { get; private set; }

    // Every wake event, whether or not a parked head matched it. A slice running at the
    // time has not recorded what it waits for yet, so the event cannot reach it; the park
    // compares this instead and keeps the head runnable.
    private ulong _wakeGeneration;

    /// <summary>Wake events that have been delivered; a slice compares it at its park.</summary>
    public ulong WakeGeneration
    {
        get
        {
            lock (_gate)
            {
                return _wakeGeneration;
            }
        }
    }

    /// <summary>Heads kept runnable because their wake arrived while they still ran.</summary>
    public ulong WakesRacedWithSlice { get; private set; }

    // DSTRACE CPW sources: which wake reached the blocked heads.
    private enum DsWakeSource : ulong
    {
        GpuTick = 1,
        RetryBlocked = 2,
        GuestMemoryWritten = 3,
        SliceCompleted = 4,
        Pipeline = 5,
        FlipSlot = 6,
        PendingLabel = 7,
    }

    private static void DsEmitWake(DsWakeSource source, ulong woken, long oldestBlockedSince)
    {
        var oldest = oldestBlockedSince == 0
            ? 0UL
            : (ulong)DsTrace.TicksToNs(System.Diagnostics.Stopwatch.GetTimestamp() - oldestBlockedSince);
        DsTrace.Emit(DsEv.CPW, (ulong)source, woken, oldest);
    }

    // Caller holds the gate. Wakes the heads a condition predicate selects.
    private void WakeMatchingLocked(Func<CommandWaitCondition, bool> matches, DsWakeSource dsSource) =>
        WakeMatchingLocked(matches, static (select, wait) => select(wait), dsSource);

    // The same with the predicate's inputs passed in, so a static lambda selects without a closure.
    private void WakeMatchingLocked<TState>(TState state, Func<TState, CommandWaitCondition, bool> matches, DsWakeSource dsSource)
    {
        _wakeGeneration++;
        var woke = false;
        var wokeCount = 0L;
        var skippedCount = 0L;
        var ds = DsTrace.Gpu && DsTrace.Armed;
        var dsOldest = 0L;
        foreach (var queue in _queues)
        {
            if (queue.First is not { } head || !head.Value.Blocked)
            {
                continue;
            }

            if (ds && (dsOldest == 0 || head.Value.BlockedSinceTicks < dsOldest))
            {
                dsOldest = head.Value.BlockedSinceTicks;
            }

            if (matches(state, head.Value.Wait))
            {
                head.Value.Blocked = false;
                TargetedWakes++;
                wokeCount++;
                woke = true;
            }
            else
            {
                TargetedWakesSkipped++;
                skippedCount++;
            }
        }

        if (PacketProfile.Enabled)
        {
            PacketProfile.RecordWakes(wokeCount, skippedCount);
        }

        if (ds && wokeCount + skippedCount != 0)
        {
            DsEmitWake(dsSource, (ulong)wokeCount, dsOldest);
        }

        if (woke)
        {
            Monitor.PulseAll(_gate);
        }
    }

    /// <summary>
    /// These guest bytes changed, so a compare against them may pass now. Everything
    /// that writes guest memory on the command-stream thread reports through here:
    /// the interpreter's own label stores, the memory operations it executes, and the
    /// backend's end-of-pipe writes.
    /// </summary>
    public void NotifyGuestMemoryWritten(ulong address, ulong size)
    {
        if (size == 0 || _submissionCount == 0)
        {
            return;
        }

        lock (_gate)
        {
            WakeMatchingLocked(wait => wait.Kind == CommandWaitKind.Unknown || wait.TouchedBy(address, size),
                DsWakeSource.GuestMemoryWritten);
        }
    }

    /// <summary>
    /// A pipeline the backend was compiling is ready, so the draws and dispatches that
    /// suspended on one can run. They re-check their own pipeline when they resume.
    /// </summary>
    public void NotifyPipelineReady()
    {
        lock (_gate)
        {
            WakeMatchingLocked(wait => wait.Kind is CommandWaitKind.Pipeline or CommandWaitKind.Unknown,
                DsWakeSource.Pipeline);
        }
    }

    /// <summary>
    /// A resource read can retry after its producer has completed. Unlike an ordinary
    /// tick this also covers a callback registered after that tick was already seen.
    /// The wake generation protects a completion racing the packet's suspension.
    /// </summary>
    public void NotifyResourceReadReady(ulong completedSerial)
    {
        lock (_gate)
        {
            _resourceReadReadySerial = Math.Max(_resourceReadReadySerial, completedSerial);
            WakeMatchingLocked(completedSerial,
                static (serial, wait) => wait.Kind == CommandWaitKind.GpuSerial && wait.Reference <= serial,
                DsWakeSource.GpuTick);
        }
    }

    /// <summary>The presenter freed a flip slot.</summary>
    public void NotifyFlipSlotAvailable()
    {
        lock (_gate)
        {
            WakeMatchingLocked(wait =>
                wait.Kind is CommandWaitKind.FlipSlot or CommandWaitKind.FlipDone or CommandWaitKind.Unknown,
                DsWakeSource.FlipSlot);
        }
    }

    // Runs one slice of the next unblocked submission on the calling thread.
    public SliceResult ProcessOne()
    {
        CommandSubmission? submission = null;
        ControlBarrier? control = null;
        var sliceWakeGeneration = 0UL;
        // DSTRACE SLICE: entry and exit of a slice that ran a submission.
        var dsSlice = DsTrace.Gpu && DsTrace.Armed;
        var dsStarted = dsSlice ? System.Diagnostics.Stopwatch.GetTimestamp() : 0L;
        var dsCpuStarted = dsSlice ? DsTrace.SelfCpuNs() : 0UL;
        lock (_gate)
        {
            if (_processing)
            {
                throw _host.Fatal("The command stream queue is already processing a slice.");
            }

            if (IsControlBarrierReadyLocked())
            {
                control = _controlBarriers.Dequeue();
            }
            else
            {
                var selected = SelectQueueLocked();
                if (selected < 0)
                {
                    if (_submissionCount != 0)
                        foreach (var pendingQueue in _queues)
                            foreach (var pending in pendingQueue)
                                pending.Preparation?.Resolve(CommandSubmissionPreparation.State.DeferredDependency);
                    Monitor.PulseAll(_gate);
                    return _submissionCount != 0 ? SliceResult.AllBlocked : SliceResult.NoWork;
                }

                var queue = _queues[selected];
                submission = queue.First!.Value;
                queue.RemoveFirst();
                _submissionCount--;
                _nextQueue = (selected + 1) % QueueCount;
            }

            _processing = true;
            _processingThread = Thread.CurrentThread;
            _activePreparation = submission?.Preparation;
            // What this slice is about to wait for may change while it runs, and the wake
            // for it arrives at a head that is not parked yet. The generation it starts
            // from decides, at the park below, whether it missed one.
            sliceWakeGeneration = _wakeGeneration;
            if (submission is { Started: false } &&
                submission.Kind is not (CommandSubmissionKind.FlipPreparation or CommandSubmissionKind.FrameBoundary))
                SubmissionFlowProfile.Record(SubmissionFlowProfile.EventKind.ProcessingStarted, submission.QueueId,
                    submission.SubmissionId, submission.Address, submission.DwordCount, _submissionCount);
        }

        if (control is not null)
        {
            try
            {
                control.Work();
            }
            catch (Exception exception)
            {
                try
                {
                    Console.Error.WriteLine($"[GPU][ERROR] Command stream control barrier failed. {exception}");
                }
                catch (Exception)
                {
                }

                Fail(exception);
                throw;
            }

            lock (_gate)
            {
                _processing = false;
                Monitor.PulseAll(_gate);
            }

            return SliceResult.Completed;
        }

        var dsPacketsStarted = 0UL;
        if (dsSlice)
        {
            DsCurrentSubmissionId = submission!.SubmissionId;
            dsPacketsStarted = _interpreters[submission.QueueId]?.DsPacketSerial ?? 0;
        }

        bool complete;
        IGuestSubmissionPrefixCompletion? prefixCompletion = null;
        ICommandSubmissionCompletionHost? completionHost;
        lock (_gate)
            completionHost = submission!.Preparation is { Status: CommandSubmissionPreparation.State.Pending }
                ? _host as ICommandSubmissionCompletionHost : null;
        var captureStarted = false;
        try
        {
            if (completionHost is not null)
            {
                completionHost.BeginSubmissionCompletionCapture(submission!.QueueId, submission.SubmissionId);
                captureStarted = true;
            }
            complete = RunSlice(submission!);
            if (captureStarted)
            {
                prefixCompletion = completionHost!.EndSubmissionCompletionCapture(complete);
                captureStarted = false;
            }
        }
        catch (Exception exception)
        {
            if (captureStarted)
                try { _ = completionHost!.EndSubmissionCompletionCapture(false); }
                catch (Exception cleanup)
                { try { exception.Data["CommandPrefixCaptureCleanup"] = cleanup; } catch { } }
            try
            {
                // Report the cause before another submitting thread observes the failed queue.
                Console.Error.WriteLine($"[GPU][ERROR] Command stream slice failed: queue={submission!.QueueId} address=0x{submission.Address:X16}. {exception}");
            }
            catch (Exception)
            {
                // An output failure must not prevent queue failure or replace the original exception.
            }

            Fail(exception);
            throw;
        }

        SliceResult result;
        lock (_gate)
        {
            submission!.Preparation?.Resolve(complete ? CommandSubmissionPreparation.State.PreparedPrefix : CommandSubmissionPreparation.State.DeferredDependency,
                completion: complete ? prefixCompletion : null);
            _activePreparation = null;
            if (!complete)
            {
                // A wake that arrived while this slice ran was delivered to a head that was
                // not parked yet, so parking now would sleep until the retry timer: the
                // slice re-runs instead and reads what changed. Polling a label the GPU is
                // about to write is exactly this race, and the timer is a hundred
                // milliseconds against a submission that completes in one.
                // A serial dependency can only have changed when its producer is
                // ready; an unrelated wake must not keep re-materializing it.
                var pendingWait = submission!.Commands.SuspendCondition;
                var raced = _wakeGeneration != sliceWakeGeneration &&
                    submission.RacedWakeGeneration != _wakeGeneration &&
                    (pendingWait.Kind != CommandWaitKind.GpuSerial || pendingWait.Reference <= _resourceReadReadySerial);
                submission!.Blocked = !raced;
                if (!raced)
                {
                    _blockedHeadsPossible = true;
                }

                if (raced)
                {
                    // One more run for this wake, and only one: a head that re-ran on
                    // every wake would never park while completions keep arriving.
                    submission.RacedWakeGeneration = _wakeGeneration;
                    WakesRacedWithSlice++;
                }

                submission.BlockedSinceTicks = System.Diagnostics.Stopwatch.GetTimestamp();
                var packetAddress = submission.Commands.CurrentPacketAddress;
                if (submission.StalledSinceTicks == 0 || submission.BlockedPacketAddress != packetAddress)
                {
                    submission.StalledSinceTicks = submission.BlockedSinceTicks;
                    submission.StallReported = false;
                }

                submission.BlockedPacketAddress = packetAddress;
                submission.Wait = pendingWait;
                if (submission.Wait.Watched)
                {
                    // What the packet saw, not what memory holds now: a store that landed
                    // between its read and this park is then a change the watch catches.
                    submission.WatchedValue = submission.Wait.Observed;
                }
                _queues[submission.QueueId].AddFirst(submission);
                _submissionCount++;
                result = submission.Commands.MadeProgress ? SliceResult.Progressed : SliceResult.BlockedWithoutProgress;
            }
            else
            {
                if (submission!.Kind == CommandSubmissionKind.FrameBoundary)
                {
                    _pendingBoundaries--;
                    _doneCount++;
                }

                // A finished submission frees a flip slot and advances the engine
                // counters of its own queue, and anything it wrote into guest memory
                // has already been reported through NotifyGuestMemoryWritten. Waking
                // every head here instead made each one re-run a slice, re-read its
                // label and suspend again. A pipeline wait is not one of them either:
                // only the compiler finishing it (NotifyPipelineReady) changes it.
                WakeMatchingLocked(wait => wait.Kind is not (CommandWaitKind.GuestMemory or CommandWaitKind.Pipeline or CommandWaitKind.GpuSerial),
                    DsWakeSource.SliceCompleted);
                result = SliceResult.Completed;
            }

            _processing = false;
            Monitor.PulseAll(_gate);
        }

        if (dsSlice)
        {
            DsEmitSlice(submission!, result, dsStarted, dsCpuStarted, dsPacketsStarted);
        }

        return result;
    }

    private void DsEmitSlice(CommandSubmission submission, SliceResult result, long started, ulong cpuStarted, ulong packetsStarted)
    {
        DsCurrentSubmissionId = 0;
        var packets = _interpreters[submission.QueueId]?.DsPacketSerial ?? 0;
        // outcome: 0 done, 1 suspended after progress, 2 blocked without progress; the wait kind in bits 8+.
        ulong outcome = result switch
        {
            SliceResult.Completed => 0,
            SliceResult.Progressed => 1,
            _ => 2,
        };
        var wait = result == SliceResult.Completed ? default : submission.Wait;
        DsTrace.Emit(
            DsEv.SLICE,
            (ulong)started,
            (ulong)DsTrace.TicksToNs(System.Diagnostics.Stopwatch.GetTimestamp() - started),
            (ulong)submission.QueueId,
            submission.SubmissionId,
            packets - packetsStarted,
            outcome | ((ulong)wait.Kind << 8),
            wait.Address,
            DsTrace.SelfCpuNs() - cpuStarted);
    }

    private bool RunSlice(CommandSubmission submission)
    {
        if (submission.Kind == CommandSubmissionKind.FlipPreparation)
        {
            SlicesRun++;
            _host.PrepareCpuFlip(submission.FlipHandle, submission.FlipIndex, submission.FlipRequestId);
            _host.Flush();
            return true;
        }

        // A suspend point drains the graphics pipe: submit what the frame recorded.
        if (submission.Kind == CommandSubmissionKind.FrameBoundary)
        {
            SlicesRun++;
            _host.FlushFrame();
            return true;
        }

        var firstSlice = !submission.Started;
        var processor = GetInterpreter(submission.QueueId);
        if (firstSlice && submission.ResetInterpreter)
        {
            processor.Reset();
        }

        if (firstSlice)
        {
            submission.Started = true;
            processor.SubmitId = ++_submitId;
            processor.ResetCounters();
            processor.SetFlip(default);
            SubmissionsStarted++;
        }

        SlicesRun++;
        var sliceStarted = PacketProfile.Enabled ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        _host.BeginSubmission(submission.QueueId, submission.SubmissionId, submission.GeometrySnapshots);
        processor.ConstantEngineComplete = true;
        if (firstSlice)
        {
            GpuStreamDiagnostics.Verify(_host, submission, "first-slice");
        }

        bool complete;
        Action? previousBeforeFatal = null;
        if (_beforeFatal is not null)
        {
            previousBeforeFatal = GpuFatal.BeforeFatal;
            _fatalSubmission = submission;
            _fatalProcessor = processor;
            GpuFatal.BeforeFatal = _beforeFatal;
        }

        try
        {
            complete = processor.Process(submission.Commands, submission.Address, submission.DwordCount) == SubmissionProgress.Complete;
        }
        finally
        {
            if (_beforeFatal is not null)
            {
                GpuFatal.BeforeFatal = previousBeforeFatal;
                _fatalSubmission = null;
                _fatalProcessor = null;
            }
        }

        if (PacketProfile.Enabled)
        {
            PacketProfile.RecordSlice(System.Diagnostics.Stopwatch.GetTimestamp() - sliceStarted, !complete);
            PacketProfile.ReportIfDue();
        }

        submission.CommandsComplete = complete;
        if (submission.Commands.MadeProgress)
        {
            if (complete)
            {
                _host.RunGarbageCollector();
            }

            _host.Flush();
        }
        else if (complete)
        {
            _host.RunGarbageCollector();
        }

        return complete;
    }

    public void StopAccepting()
    {
        lock (_gate)
        {
            _accepting = false;
            _stopping = true;
            Monitor.PulseAll(_gate);
        }
    }

    // Processes to empty; with the flag, heads still blocked after a full retry cycle are cancelled.
    public IdleOutcome DrainForShutdown(bool cancelBlockedOnNoProgress)
    {
        for (;;)
        {
            var result = ProcessOne();
            switch (result)
            {
                case SliceResult.NoWork:
                    return Outcome;
                case SliceResult.Progressed:
                case SliceResult.Completed:
                    continue;
                case SliceResult.BlockedWithoutProgress:
                    if (HasUnblockedPending)
                    {
                        continue;
                    }

                    goto case SliceResult.AllBlocked;
                case SliceResult.AllBlocked:
                    _ = WaitForRetryInterval(AllBlockedRetryMilliseconds);
                    RetryBlocked();
                    if (!RunRetryCycle() && cancelBlockedOnNoProgress)
                    {
                        CancelBlocked();
                    }

                    continue;
            }
        }
    }

    // One pass over every head after a retry; true when any slice made progress.
    private bool RunRetryCycle()
    {
        var progressed = false;
        while (HasUnblockedPending)
        {
            var result = ProcessOne();
            if (result is SliceResult.Progressed or SliceResult.Completed)
            {
                progressed = true;
            }

            if (result is SliceResult.NoWork or SliceResult.AllBlocked)
            {
                break;
            }
        }

        return progressed;
    }

    private void CancelBlocked()
    {
        lock (_gate)
        {
            foreach (var queue in _queues)
            {
                if (queue.First is { } head && head.Value.Blocked)
                {
                    _submissionCount -= queue.Count;
                    foreach (var dropped in queue)
                    {
                        if (dropped.Kind == CommandSubmissionKind.FrameBoundary)
                        {
                            _pendingBoundaries--;
                        }
                    }

                    foreach (var dropped in queue) dropped.Preparation?.Resolve(CommandSubmissionPreparation.State.Cancelled);
                    queue.Clear();
                    _outcome = IdleOutcome.Cancelled;
                }
            }

            Monitor.PulseAll(_gate);
        }
    }

    // Drops every submission without stopping admission; the presenter uses it after device loss.
    public void DiscardAll()
    {
        Action[] cancellations;
        lock (_gate)
        {
            _outcome = IdleOutcome.Failed;
            _activePreparation?.Resolve(CommandSubmissionPreparation.State.Failed, _preparationFailure);
            DropAllLocked();
            cancellations = CancelControlBarriersLocked();
            Monitor.PulseAll(_gate);
        }

        foreach (var cancel in cancellations)
            cancel();
    }

    // Heads that have sat on the same packet for at least minimumMilliseconds and have not
    // been reported yet: each is named once per stall, so a deadlock says what it is
    // waiting for instead of looking like an idle GPU.
    public IReadOnlyList<(int QueueId, ulong SubmissionId, CommandSubmissionKind Kind, ulong PacketAddress,
        CommandWaitCondition Wait, double Milliseconds)> TakeNewStalls(double minimumMilliseconds)
    {
        List<(int, ulong, CommandSubmissionKind, ulong, CommandWaitCondition, double)>? stalls = null;
        lock (_gate)
        {
            var now = System.Diagnostics.Stopwatch.GetTimestamp();
            for (var queueId = 0; queueId < QueueCount; queueId++)
            {
                if (_queues[queueId].First is not { } head || !head.Value.Blocked || head.Value.StallReported ||
                    head.Value.StalledSinceTicks == 0)
                {
                    continue;
                }

                var milliseconds = (now - head.Value.StalledSinceTicks) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
                if (milliseconds < minimumMilliseconds)
                {
                    continue;
                }

                head.Value.StallReported = true;
                (stalls ??= []).Add((queueId, head.Value.SubmissionId, head.Value.Kind,
                    head.Value.BlockedPacketAddress, head.Value.Wait, milliseconds));
            }
        }

        return (IReadOnlyList<(int, ulong, CommandSubmissionKind, ulong, CommandWaitCondition, double)>?)stalls ?? [];
    }

    // A diagnostic view of the blocked heads: their count, the oldest age and one sample.
    public BlockedSnapshot SnapshotBlocked()
    {
        lock (_gate)
        {
            return SnapshotBlockedLocked();
        }
    }

    private BlockedSnapshot SnapshotBlockedLocked()
    {
        var count = 0;
        var oldestTicks = 0L;
        var sampleAddress = 0UL;
        var sampleQueue = -1;
        var now = System.Diagnostics.Stopwatch.GetTimestamp();
        for (var queueId = 0; queueId < QueueCount; queueId++)
        {
            if (_queues[queueId].First is not { } head || !head.Value.Blocked)
            {
                continue;
            }

            count++;
            var age = now - head.Value.BlockedSinceTicks;
            if (sampleQueue < 0 || age > oldestTicks)
            {
                oldestTicks = age;
                sampleAddress = head.Value.BlockedPacketAddress;
                sampleQueue = queueId;
            }
        }

        return new BlockedSnapshot(count, oldestTicks * 1000.0 / System.Diagnostics.Stopwatch.Frequency, sampleAddress, sampleQueue);
    }

    public IdleOutcome Outcome
    {
        get
        {
            lock (_gate)
            {
                return _outcome;
            }
        }
    }

    public int BlockedQueueCount
    {
        get
        {
            lock (_gate)
            {
                var count = 0;
                foreach (var queue in _queues)
                {
                    if (queue.First is { } head && head.Value.Blocked)
                    {
                        count++;
                    }
                }

                return count;
            }
        }
    }

    private int SelectQueueLocked()
    {
        if (_controlBarriers.TryPeek(out var control))
        {
            for (var offset = 0; offset < QueueCount; offset++)
            {
                var queueId = (_nextQueue + offset) % QueueCount;
                if (_queues[queueId].First is { } head &&
                    head.Value.AdmissionOrdinal <= control.CutoffOrdinal &&
                    !head.Value.Blocked)
                {
                    return queueId;
                }
            }
        }

        for (var offset = 0; offset < QueueCount; offset++)
        {
            var queueId = (_nextQueue + offset) % QueueCount;
            if (_queues[queueId].First is { } head && !head.Value.Blocked)
            {
                return queueId;
            }
        }

        return -1;
    }

    private bool HasRunnableWorkLocked() => IsControlBarrierReadyLocked() || SelectQueueLocked() >= 0;

    private bool IsControlBarrierReadyLocked()
    {
        if (!_controlBarriers.TryPeek(out var control))
        {
            return false;
        }

        foreach (var queue in _queues)
        {
            if (queue.First is { } head && head.Value.AdmissionOrdinal <= control.CutoffOrdinal)
            {
                return false;
            }
        }

        return true;
    }

    private Action[] CancelControlBarriersLocked()
    {
        if (_controlBarriers.Count == 0)
        {
            return [];
        }

        var cancellations = new Action[_controlBarriers.Count];
        for (var index = 0; index < cancellations.Length; index++)
        {
            cancellations[index] = _controlBarriers.Dequeue().Cancel;
        }

        return cancellations;
    }

    private void DropAllLocked()
    {
        foreach (var queue in _queues)
        {
            foreach (var dropped in queue) dropped.Preparation?.Resolve(
                _outcome == IdleOutcome.Failed ? CommandSubmissionPreparation.State.Failed : CommandSubmissionPreparation.State.Cancelled, _preparationFailure);
            queue.Clear();
        }

        _submissionCount = 0;
        _pendingBoundaries = 0;
    }
}
