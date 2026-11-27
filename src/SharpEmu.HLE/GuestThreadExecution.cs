// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;

namespace SharpEmu.HLE;

public readonly record struct GuestThreadStartRequest(
    ulong ThreadHandle,
    ulong EntryPoint,
    ulong Argument,
    ulong AttributeAddress,
    string Name,
    int Priority,
    ulong AffinityMask,
    ulong StackSize = 0,
    IGuestThreadRecycler? Recycler = null);

/// <summary>
/// Owner of a recyclable guest thread. When the thread's entry returns,
/// the scheduler parks it instead of exiting it, then reports the return here;
/// the owner may hand the parked thread new work with <see cref="IGuestThreadScheduler.TryRestartParkedThread"/>.
/// </summary>
public interface IGuestThreadRecycler
{
    void OnEntryReturned(ulong threadHandle, ulong returnValue);
}

public readonly record struct GuestThreadSnapshot(
    ulong ThreadHandle,
    string Name,
    string State,
    long ImportCount,
    string? LastImportNid,
    ulong LastReturnRip,
    string? BlockReason);

/// <summary>
/// Continuation state for a blocked guest thread, replacing the closure pair a blocking
/// wait used to allocate. TryWake runs under the scheduler's guest-thread gate and
/// returns true when the waiter has a final result and the thread should be re-readied;
/// false leaves it parked. Resume runs later on the woken thread outside that gate, and
/// its return value becomes the guest's RAX for the resumed call.
/// </summary>
public interface IGuestThreadBlockWaiter
{
    int Resume();

    bool TryWake();
}

public interface IGuestThreadScheduler
{
    bool SupportsGuestContextTransfer { get; }

    /// <summary>
    /// Associates a pthread identity created on the primary guest executor
    /// with its live CPU context. Primary execution does not pass through
    /// TryStartThread, but kernel exception delivery must still be able to
    /// target it.
    /// </summary>
    void RegisterGuestThreadContext(ulong threadHandle, CpuContext context);

    /// <summary>
    /// Returns the exact mapped guest stack owned by a scheduler thread. The
    /// default keeps alternate/test schedulers source-compatible when they do
    /// not own guest stack mappings.
    /// </summary>
    bool TryGetGuestThreadStackBounds(
        ulong threadHandle,
        out ulong stackBase,
        out ulong stackSize)
    {
        stackBase = 0;
        stackSize = 0;
        return false;
    }

    bool TryStartThread(CpuContext creatorContext, GuestThreadStartRequest request, out string? error);

    /// <summary>
    /// Runs a new entry on a parked recyclable thread, from a fresh frame at
    /// the top of its stack. False when the thread is not parked.
    /// </summary>
    bool TryRestartParkedThread(ulong threadHandle, ulong entryPoint, ulong argument, out string? error)
    {
        error = "this scheduler does not recycle guest threads";
        return false;
    }

    bool TryJoinThread(
        CpuContext callerContext,
        ulong threadHandle,
        out ulong returnValue,
        out string? error);

    void Pump(CpuContext callerContext, string reason);

    int WakeBlockedThreads(string wakeKey, int maxCount = int.MaxValue);

    /// <summary>
    /// Readies the one thread parked on <paramref name="waiter"/> after its wait completed,
    /// without scanning other threads. False when the thread is not (yet) parked on that
    /// waiter; a thread that parks later re-checks <see cref="IGuestThreadBlockWaiter.TryWake"/>
    /// itself, so no wake is lost. The default falls back to the key-based scan.
    /// </summary>
    bool WakeBlockedWaiter(ulong threadHandle, IGuestThreadBlockWaiter waiter, string wakeKey) =>
        WakeBlockedThreads(wakeKey) > 0;

    /// <summary>
    /// Lock-free, possibly stale: whether the scheduler thread is executing a slice on a host
    /// CPU right now (false while it is blocked, ready or parked). Unknown threads report true.
    /// Used by contended locks to stop spinning on an owner that cannot release soon.
    /// </summary>
    bool IsGuestThreadExecuting(ulong threadHandle) => true;

    /// <summary>
    /// Reports whether the current guest thread has an exception that waits for an import safe point.
    /// </summary>
    bool HasPendingGuestExceptionForCurrentThread();

    // Host waits must allow exception delivery before the import call returns.
    void DeliverPendingGuestExceptionIfReady(CpuContext context) { }

    /// <summary>
    /// Applies a new guest scheduling priority to a live thread, mapping it
    /// onto the host thread if one is running. Returns false when the thread
    /// handle is unknown.
    /// </summary>
    bool TrySetGuestThreadPriority(ulong guestThreadHandle, int guestPriority);

    /// <summary>
    /// Records a new affinity mask for a guest thread and re-applies it to
    /// the host thread where the platform supports it.
    /// </summary>
    bool TrySetGuestThreadAffinity(ulong guestThreadHandle, ulong affinityMask);

    IReadOnlyList<GuestThreadSnapshot> SnapshotThreads();

    bool TryCallGuestFunction(
        CpuContext callerContext,
        ulong entryPoint,
        ulong arg0,
        ulong arg1,
        ulong stackAddress,
        ulong stackSize,
        string reason,
        out string? error);

    bool TryCallGuestFunction(
        CpuContext callerContext,
        ulong entryPoint,
        ulong arg0,
        ulong arg1,
        ulong arg2,
        ulong stackAddress,
        ulong stackSize,
        string reason,
        out ulong returnValue,
        out string? error);

    bool TryCallGuestFunction(
        CpuContext callerContext,
        ulong entryPoint,
        ulong arg0,
        ulong arg1,
        ulong arg2,
        ulong arg3,
        ulong stackAddress,
        ulong stackSize,
        string reason,
        out ulong returnValue,
        out string? error);

    bool TryCallGuestContinuation(
        CpuContext callerContext,
        GuestCpuContinuation continuation,
        string reason,
        out string? error);

    /// <summary>
    /// Asynchronously invokes an installed kernel exception handler as the
    /// target guest thread. This is used by IL2CPP's stop-the-world collector:
    /// the handler acknowledges suspension and may remain blocked until the
    /// collecting thread resumes it.
    /// </summary>
    bool TryRaiseGuestException(
        CpuContext callerContext,
        ulong threadHandle,
        ulong handler,
        int exceptionType,
        out string? error);

    /// <summary>
    /// Same as the signal overload, but lets the caller choose how the handler is
    /// called. Schedulers that only know signal delivery refuse the other conventions.
    /// </summary>
    bool TryRaiseGuestException(
        CpuContext callerContext,
        ulong threadHandle,
        ulong handler,
        int exceptionType,
        GuestExceptionHandlerArguments arguments,
        out string? error)
    {
        if (arguments == GuestExceptionHandlerArguments.Signal)
        {
            return TryRaiseGuestException(callerContext, threadHandle, handler, exceptionType, out error);
        }

        error = $"guest exception arguments {arguments} are not supported";
        return false;
    }
}

/// <summary>
/// Argument convention a raised guest exception handler is called with.
/// </summary>
public enum GuestExceptionHandlerArguments
{
    /// <summary>Signal handler: (exception type, mcontext pointer).</summary>
    Signal,

    /// <summary>Unity GC suspend callback: (mcontext pointer, interrupted rsp).</summary>
    UnityGcSuspend,
}

public readonly record struct GuestImportCallFrame(
    bool IsValid,
    ulong ReturnRip,
    ulong ResumeRsp,
    ulong ReturnSlotAddress);

public readonly record struct GuestCpuContinuation(
    ulong Rip,
    ulong Rsp,
    ulong ReturnSlotAddress,
    ulong Rflags,
    ulong FsBase,
    ulong GsBase,
    ulong Rax,
    ulong Rcx,
    ulong Rdx,
    ulong Rbx,
    ulong Rbp,
    ulong Rsi,
    ulong Rdi,
    ulong R8,
    ulong R9,
    ulong R10,
    ulong R11,
    ulong R12,
    ulong R13,
    ulong R14,
    ulong R15,
    ushort FpuControlWord,
    uint Mxcsr,
    bool RestoreFullFpuState);

public static class GuestThreadExecution
{
    private sealed class DelegateGuestThreadBlockWaiter : IGuestThreadBlockWaiter
    {
        private readonly Func<int> _resume;
        private readonly Func<bool> _tryWake;

        public DelegateGuestThreadBlockWaiter(Func<int> resume, Func<bool> tryWake)
        {
            _resume = resume;
            _tryWake = tryWake;
        }

        public int Resume() => _resume();

        public bool TryWake() => _tryWake();
    }

    /// <summary>
    /// Every per-host-thread field of the guest execution state in one object.
    /// </summary>
    /// <remarks>
    /// These used to be seventeen separate <c>[ThreadStatic]</c> fields, and the import gateway
    /// touched a dozen of them on every guest call: a thread-static read is a TLS lookup, which
    /// is not free under Rosetta, and the gateway pays it again for each field. One thread-static
    /// holds the object, so a crossing resolves the thread's state once and reads plain fields.
    /// </remarks>
    internal sealed class ThreadState
    {
        public ulong GuestThreadHandle;
        public ulong FiberAddress;
        public string? PendingBlockReason;
        public bool PendingBlockContinuationValid;
        public GuestCpuContinuation PendingBlockContinuation;
        public string? PendingBlockWakeKey;
        public IGuestThreadBlockWaiter? PendingBlockWaiter;
        public long PendingBlockDeadlineTimestamp;
        public bool PendingEntryExit;
        public ulong PendingEntryExitValue;
        public string? PendingEntryExitReason;
        public bool PendingContextTransfer;
        public GuestCpuContinuation PendingContextTransferTarget;
        public Action<bool>? ContextTransferCompletion;
        public bool HasImportCallFrame;
        public ulong ImportReturnRip;
        public ulong ImportResumeRsp;
        public ulong ImportReturnSlotAddress;

        // Slots the native import gateway keeps per host thread. They live here so that a crossing
        // resolves this object once instead of reading a thread-static for each of them.
        public object? ImportCounterOwner;
        public long NextImportDispatchIndex;
        public long ImportDispatchBlockEnd;
        public nint PreparedContextTransferStub;

        /// <summary>
        /// True when an import left a block, an entry exit or a context transfer pending. False for
        /// almost every crossing, which then skips the three consume calls the gateway otherwise makes.
        /// </summary>
        public bool HasPendingDispatchOutcome =>
            PendingBlockReason is not null || PendingEntryExit || PendingContextTransfer;

        public GuestImportCallFrame EnterImportCallFrame(ulong returnRip, ulong resumeRsp, ulong returnSlotAddress)
        {
            var previous = new GuestImportCallFrame(
                HasImportCallFrame,
                ImportReturnRip,
                ImportResumeRsp,
                ImportReturnSlotAddress);
            HasImportCallFrame = true;
            ImportReturnRip = returnRip;
            ImportResumeRsp = resumeRsp;
            ImportReturnSlotAddress = returnSlotAddress;
            return previous;
        }

        public void RestoreImportCallFrame(GuestImportCallFrame previous)
        {
            HasImportCallFrame = previous.IsValid;
            ImportReturnRip = previous.ReturnRip;
            ImportResumeRsp = previous.ResumeRsp;
            ImportReturnSlotAddress = previous.ReturnSlotAddress;
        }

        public bool TryConsumeContextTransfer(out GuestCpuContinuation target)
        {
            target = PendingContextTransferTarget;
            if (!PendingContextTransfer)
            {
                return false;
            }

            PendingContextTransfer = false;
            PendingContextTransferTarget = default;
            return true;
        }

        public bool TryConsumeEntryExit(out ulong value, out string reason)
        {
            value = PendingEntryExitValue;
            reason = PendingEntryExitReason ?? string.Empty;
            if (!PendingEntryExit)
            {
                return false;
            }

            PendingEntryExit = false;
            PendingEntryExitValue = 0;
            PendingEntryExitReason = null;
            return true;
        }

        public bool TryConsumeThreadBlock(
            out string reason,
            out GuestCpuContinuation continuation,
            out bool hasContinuation,
            out string wakeKey,
            out IGuestThreadBlockWaiter? waiter,
            out long blockDeadlineTimestamp)
        {
            reason = PendingBlockReason ?? string.Empty;
            if (string.IsNullOrEmpty(reason))
            {
                continuation = default;
                hasContinuation = false;
                wakeKey = string.Empty;
                waiter = null;
                blockDeadlineTimestamp = 0;
                return false;
            }

            continuation = PendingBlockContinuation;
            hasContinuation = PendingBlockContinuationValid;
            wakeKey = PendingBlockWakeKey ?? reason;
            waiter = PendingBlockWaiter;
            blockDeadlineTimestamp = PendingBlockDeadlineTimestamp;
            PendingBlockReason = null;
            PendingBlockContinuation = default;
            PendingBlockContinuationValid = false;
            PendingBlockWakeKey = null;
            PendingBlockWaiter = null;
            PendingBlockDeadlineTimestamp = 0;
            return true;
        }

        public void CompleteContextTransfer(bool succeeded)
        {
            // The target is only ever non-default while PendingContextTransfer is set, so with
            // neither a pending transfer nor a completion armed there is nothing to clear - the
            // common case, which used to zero the continuation on every crossing.
            var completion = ContextTransferCompletion;
            if (completion is null && !PendingContextTransfer)
            {
                return;
            }

            ContextTransferCompletion = null;
            PendingContextTransfer = false;
            PendingContextTransferTarget = default;
            completion?.Invoke(succeeded);
        }
    }

    [ThreadStatic]
    private static ThreadState? t_state;

    /// <summary>The calling host thread's guest execution state; created on first use.</summary>
    internal static ThreadState State => t_state ?? CreateState();

    private static ThreadState CreateState() => t_state = new ThreadState();

    public static IGuestThreadScheduler? Scheduler { get; set; }

    /// <summary>
    /// Fired when a guest thread is torn down without a clean pthread_exit
    /// (e.g. TBB execute-AV → worker_abort). Libs use this to abandon mutexes.
    /// </summary>
    public static event Func<ulong, string, int>? GuestThreadAbandoned;

    public static int NotifyGuestThreadAbandoned(ulong threadHandle, string reason)
    {
        if (threadHandle == 0 || GuestThreadAbandoned is null)
        {
            return 0;
        }

        try
        {
            return GuestThreadAbandoned.Invoke(threadHandle, reason);
        }
        catch
        {
            return 0;
        }
    }

    /// <summary>
    /// Raised on a scheduler guest thread whose start routine returned (including through
    /// pthread_exit), before the thread is marked exited, with the thread still current and
    /// no scheduler lock held. Libs run TLS-key and thread_atexit destructors here, as
    /// libthr does at thread exit.
    /// </summary>
    public static event Action<CpuContext>? GuestThreadEntryReturned;

    public static void NotifyGuestThreadEntryReturned(CpuContext context)
    {
        try
        {
            GuestThreadEntryReturned?.Invoke(context);
        }
        catch (Exception error)
        {
            Console.Error.WriteLine($"[LOADER][WARN] thread-exit destructors failed: {error.Message}");
        }
    }

    /// <summary>
    /// Fired by an HLE call that is doing real, progressing work synchronously
    /// on the guest thread but can legitimately take longer than the native
    /// backend's stall-watchdog window (e.g. a first-boot asset-tree index).
    /// The backend subscribes to reset its "no progress" timer, so a slow but
    /// live operation is not mistaken for a genuine stuck guest and killed.
    /// </summary>
    public static event Action? HostWorkProgressObserved;

    public static void NotifyHostWorkProgress()
    {
        try
        {
            HostWorkProgressObserved?.Invoke();
        }
        catch
        {
            // Best-effort signal only; a broken subscriber must not fault the
            // HLE call that's reporting real progress.
        }
    }

    public static bool IsGuestThread => State.GuestThreadHandle != 0;

    public static ulong CurrentGuestThreadHandle => State.GuestThreadHandle;

    public static bool HasPendingCurrentThreadBlock => State.PendingBlockReason is not null;

    public static ulong CurrentFiberAddress => State.FiberAddress;

    public static ulong EnterGuestThread(ulong threadHandle)
    {
        var s = State;
        CompleteCurrentContextTransfer(false);
        GpuMemory.GpuMemoryAccessProfile.InitializeCurrentThread();
        var previous = s.GuestThreadHandle;
        s.GuestThreadHandle = threadHandle;
        GuestFastPath.BindGuestThread(threadHandle);
        s.PendingBlockReason = null;
        s.PendingBlockContinuationValid = false;
        s.PendingBlockContinuation = default;
        s.PendingBlockWakeKey = null;
        s.PendingBlockWaiter = null;
        s.PendingBlockDeadlineTimestamp = 0;
        s.PendingEntryExit = false;
        s.PendingEntryExitValue = 0;
        s.PendingEntryExitReason = null;
        s.PendingContextTransfer = false;
        s.PendingContextTransferTarget = default;
        s.HasImportCallFrame = false;
        s.ImportReturnRip = 0;
        s.ImportResumeRsp = 0;
        s.ImportReturnSlotAddress = 0;
        return previous;
    }

    public static void RestoreGuestThread(ulong previousThreadHandle)
    {
        var s = State;
        CompleteCurrentContextTransfer(false);
        s.GuestThreadHandle = previousThreadHandle;
        GuestFastPath.BindGuestThread(previousThreadHandle);
        s.PendingBlockReason = null;
        s.PendingBlockContinuationValid = false;
        s.PendingBlockContinuation = default;
        s.PendingBlockWakeKey = null;
        s.PendingBlockWaiter = null;
        s.PendingBlockDeadlineTimestamp = 0;
        s.PendingEntryExit = false;
        s.PendingEntryExitValue = 0;
        s.PendingEntryExitReason = null;
        s.PendingContextTransfer = false;
        s.PendingContextTransferTarget = default;
        s.HasImportCallFrame = false;
        s.ImportReturnRip = 0;
        s.ImportResumeRsp = 0;
        s.ImportReturnSlotAddress = 0;
    }

    public static ulong EnterFiber(ulong fiberAddress)
    {
        var s = State;
        var previous = s.FiberAddress;
        s.FiberAddress = fiberAddress;
        return previous;
    }

    public static void RestoreFiber(ulong previousFiberAddress)
    {
        var s = State;
        s.FiberAddress = previousFiberAddress;
    }

    // A nested guest exception handler (delivered synchronously from
    // inside an outer import's own dispatch, before that import's own pending
    // dispatch outcome -- a block request, an entry/exit, or a context
    // transfer -- has been consumed) runs its own imports on the same
    // native thread and therefore the same ThreadStatic slots. Without
    // suspending the outer request first, an unrelated inner import (e.g. a
    // trivial pthread_self call the handler happens to make) can consume --
    // and misattribute -- the outer call's still-pending outcome, or simply
    // clobber it. Save/clear all three groups before the nested call,
    // restore after.
    public readonly struct PendingBlockSnapshot
    {
        internal readonly string? Reason;
        internal readonly bool ContinuationValid;
        internal readonly GuestCpuContinuation Continuation;
        internal readonly string? WakeKey;
        internal readonly IGuestThreadBlockWaiter? Waiter;
        internal readonly long DeadlineTimestamp;
        internal readonly bool EntryExitPending;
        internal readonly ulong EntryExitValue;
        internal readonly string? EntryExitReason;
        internal readonly bool ContextTransferPending;
        internal readonly GuestCpuContinuation ContextTransferTarget;

        internal PendingBlockSnapshot(
            string? reason,
            bool continuationValid,
            GuestCpuContinuation continuation,
            string? wakeKey,
            IGuestThreadBlockWaiter? waiter,
            long deadlineTimestamp,
            bool entryExitPending,
            ulong entryExitValue,
            string? entryExitReason,
            bool contextTransferPending,
            GuestCpuContinuation contextTransferTarget)
        {
            Reason = reason;
            ContinuationValid = continuationValid;
            Continuation = continuation;
            WakeKey = wakeKey;
            Waiter = waiter;
            DeadlineTimestamp = deadlineTimestamp;
            EntryExitPending = entryExitPending;
            EntryExitValue = entryExitValue;
            EntryExitReason = entryExitReason;
            ContextTransferPending = contextTransferPending;
            ContextTransferTarget = contextTransferTarget;
        }
    }

    public static PendingBlockSnapshot SuspendPendingBlock()
    {
        var s = State;
        var snapshot = new PendingBlockSnapshot(
            s.PendingBlockReason,
            s.PendingBlockContinuationValid,
            s.PendingBlockContinuation,
            s.PendingBlockWakeKey,
            s.PendingBlockWaiter,
            s.PendingBlockDeadlineTimestamp,
            s.PendingEntryExit,
            s.PendingEntryExitValue,
            s.PendingEntryExitReason,
            s.PendingContextTransfer,
            s.PendingContextTransferTarget);
        s.PendingBlockReason = null;
        s.PendingBlockContinuation = default;
        s.PendingBlockContinuationValid = false;
        s.PendingBlockWakeKey = null;
        s.PendingBlockWaiter = null;
        s.PendingBlockDeadlineTimestamp = 0;
        s.PendingEntryExit = false;
        s.PendingEntryExitValue = 0;
        s.PendingEntryExitReason = null;
        s.PendingContextTransfer = false;
        s.PendingContextTransferTarget = default;
        return snapshot;
    }

    public static void RestorePendingBlock(PendingBlockSnapshot snapshot)
    {
        var s = State;
        s.PendingEntryExit = snapshot.EntryExitPending;
        s.PendingEntryExitValue = snapshot.EntryExitValue;
        s.PendingEntryExitReason = snapshot.EntryExitReason;
        s.PendingContextTransfer = snapshot.ContextTransferPending;
        s.PendingContextTransferTarget = snapshot.ContextTransferTarget;
        s.PendingBlockReason = snapshot.Reason;
        s.PendingBlockContinuationValid = snapshot.ContinuationValid;
        s.PendingBlockContinuation = snapshot.Continuation;
        s.PendingBlockWakeKey = snapshot.WakeKey;
        s.PendingBlockWaiter = snapshot.Waiter;
        s.PendingBlockDeadlineTimestamp = snapshot.DeadlineTimestamp;
    }

    public static bool RequestCurrentThreadBlock(string reason) => RequestCurrentThreadBlock(null, reason);

    public static bool RequestCurrentThreadBlock(
        CpuContext? context,
        string reason,
        string? wakeKey = null,
        IGuestThreadBlockWaiter? waiter = null,
        long blockDeadlineTimestamp = 0)
    {
        var s = State;
        if (!IsGuestThread)
        {
            return false;
        }

        s.PendingBlockReason = string.IsNullOrWhiteSpace(reason) ? "guest_thread_blocked" : reason;
        s.PendingBlockWakeKey = string.IsNullOrWhiteSpace(wakeKey) ? s.PendingBlockReason : wakeKey;
        s.PendingBlockWaiter = waiter;
        s.PendingBlockDeadlineTimestamp = blockDeadlineTimestamp;
        if (context is not null && TryCaptureCurrentBlockContinuation(context, out var continuation))
        {
            s.PendingBlockContinuation = continuation;
            s.PendingBlockContinuationValid = true;
        }
        else
        {
            s.PendingBlockContinuation = default;
            s.PendingBlockContinuationValid = false;
        }

        return true;
    }

    // Compatibility bridge for exports that still describe blocked work as a
    // resume/wake delegate pair. New hot paths should provide an
    // IGuestThreadBlockWaiter directly to avoid allocating closures.
    public static bool RequestCurrentThreadBlock(
        CpuContext? context,
        string reason,
        string? wakeKey,
        Func<int> resumeHandler,
        Func<bool> wakeHandler,
        long blockDeadlineTimestamp = 0) =>
        RequestCurrentThreadBlock(
            context,
            reason,
            wakeKey,
            new DelegateGuestThreadBlockWaiter(resumeHandler, wakeHandler),
            blockDeadlineTimestamp);

    public static bool TryConsumeCurrentThreadBlock(out string reason)
    {
        return TryConsumeCurrentThreadBlock(out reason, out _, out _);
    }

    public static bool TryConsumeCurrentThreadBlock(
        out string reason,
        out GuestCpuContinuation continuation,
        out bool hasContinuation)
    {
        return TryConsumeCurrentThreadBlock(
            out reason,
            out continuation,
            out hasContinuation,
            out _,
            out _,
            out _);
    }

    public static bool TryConsumeCurrentThreadBlock(
        out string reason,
        out GuestCpuContinuation continuation,
        out bool hasContinuation,
        out string wakeKey,
        out IGuestThreadBlockWaiter? waiter)
    {
        return TryConsumeCurrentThreadBlock(
            out reason,
            out continuation,
            out hasContinuation,
            out wakeKey,
            out waiter,
            out _);
    }

    public static bool TryConsumeCurrentThreadBlock(
        out string reason,
        out GuestCpuContinuation continuation,
        out bool hasContinuation,
        out string wakeKey,
        out IGuestThreadBlockWaiter? waiter,
        out long blockDeadlineTimestamp) =>
        State.TryConsumeThreadBlock(
            out reason,
            out continuation,
            out hasContinuation,
            out wakeKey,
            out waiter,
            out blockDeadlineTimestamp);

    public static long ComputeDeadlineTimestamp(TimeSpan timeout)
    {
        if (timeout <= TimeSpan.Zero)
        {
            return Stopwatch.GetTimestamp();
        }

        var ticks = timeout.TotalSeconds >= long.MaxValue / (double)Stopwatch.Frequency
            ? long.MaxValue
            : (long)Math.Ceiling(timeout.TotalSeconds * Stopwatch.Frequency);
        var now = Stopwatch.GetTimestamp();
        if (long.MaxValue - now <= ticks)
        {
            return long.MaxValue;
        }

        return now + Math.Max(1, ticks);
    }

    private static bool TryCaptureCurrentBlockContinuation(CpuContext context, out GuestCpuContinuation continuation)
    {
        if (!TryGetCurrentImportCallFrame(out var frame) ||
            frame.ReturnRip < 65536 ||
            frame.ResumeRsp == 0 ||
            frame.ReturnSlotAddress == 0)
        {
            continuation = default;
            return false;
        }

        continuation = new GuestCpuContinuation(
            frame.ReturnRip,
            frame.ResumeRsp,
            frame.ReturnSlotAddress,
            context.Rflags,
            context.FsBase,
            context.GsBase,
            0,
            context[CpuRegister.Rcx],
            context[CpuRegister.Rdx],
            context[CpuRegister.Rbx],
            context[CpuRegister.Rbp],
            context[CpuRegister.Rsi],
            context[CpuRegister.Rdi],
            context[CpuRegister.R8],
            context[CpuRegister.R9],
            context[CpuRegister.R10],
            context[CpuRegister.R11],
            context[CpuRegister.R12],
            context[CpuRegister.R13],
            context[CpuRegister.R14],
            context[CpuRegister.R15],
            context.FpuControlWord,
            context.Mxcsr,
            RestoreFullFpuState: false);
        return true;
    }

    public static void RequestCurrentEntryExit(string reason, int status)
    {
        RequestCurrentEntryExit(reason, unchecked((ulong)(long)status));
    }

    public static void RequestCurrentEntryExit(string reason, ulong value)
    {
        var s = State;
        s.PendingEntryExit = true;
        s.PendingEntryExitValue = value;
        s.PendingEntryExitReason = string.IsNullOrWhiteSpace(reason) ? "guest_entry_exit" : reason;
    }

    public static bool TryConsumeCurrentEntryExit(out ulong value, out string reason) =>
        State.TryConsumeEntryExit(out value, out reason);

    public static void RequestCurrentContextTransfer(GuestCpuContinuation target, Action<bool>? completion = null)
    {
        var s = State;
        s.PendingContextTransferTarget = target;
        s.PendingContextTransfer = true;
        s.ContextTransferCompletion = completion;
    }

    public static void CompleteCurrentContextTransfer(bool succeeded) =>
        State.CompleteContextTransfer(succeeded);

    public static bool TryConsumeCurrentContextTransfer(out GuestCpuContinuation target) =>
        State.TryConsumeContextTransfer(out target);

    public static GuestImportCallFrame EnterImportCallFrame(
        ulong returnRip,
        ulong resumeRsp,
        ulong returnSlotAddress) =>
        State.EnterImportCallFrame(returnRip, resumeRsp, returnSlotAddress);

    public static void RestoreImportCallFrame(GuestImportCallFrame previous) =>
        State.RestoreImportCallFrame(previous);

    public static bool TryGetCurrentImportCallFrame(out GuestImportCallFrame frame)
    {
        var s = State;
        if (!s.HasImportCallFrame)
        {
            frame = default;
            return false;
        }

        frame = new GuestImportCallFrame(
            true,
            s.ImportReturnRip,
            s.ImportResumeRsp,
            s.ImportReturnSlotAddress);
        return true;
    }
}
