// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;
using SharpEmu.HLE;
using System.Buffers.Binary;
using System.Diagnostics;
using System.Threading;
using System.Diagnostics.CodeAnalysis;
using SharpEmu.Libs.Diagnostics;
using SharpEmu.HLE.DsTrace;
using SharpEmu.Libs.Threading;

namespace SharpEmu.Libs.Kernel;

public static class KernelPthreadCompatExports
{
    private const int MutexTypeErrorCheck = 1;
    private const int MutexTypeRecursive = 2;
    private const int MutexTypeNormal = 3;
    private const int MutexTypeAdaptiveNp = 4;
    private const ulong StaticAdaptiveMutexInitializer = 1;
    private const int MutexObjectSize = 0x100;
    private const int MutexAttrObjectSize = 0x40;
    private const int CondObjectSize = 0x100;
    private const int CondAttrObjectSize = 0x40;
    private const int ClockRealtime = 0;
    private const int ClockMonotonic = 4;
    private const int PthreadProcessPrivate = 0;
    private const int PthreadProcessShared = 1;
    private const int PosixOutOfMemory = 12;
    private const int PosixBadAddress = 14;
    private const int PosixInvalidArgument = 22;
    private const int PthreadOnceUninitialized = 0;
    private const int PthreadOnceInProgress = 1;
    private const int PthreadOnceDone = 2;

    private static readonly object _stateGate = new();
    private static readonly ConcurrentDictionary<ulong, PthreadMutexState> _mutexStates = new();
    private static readonly Dictionary<ulong, PthreadMutexAttrState> _mutexAttrStates = new();
    private static readonly Dictionary<ulong, PthreadCondState> _condStates = new();
    private static readonly Dictionary<ulong, object> _onceGates = new();
    private static readonly Dictionary<ulong, PthreadCondAttrState> _condAttrStates = new();
    // Only these exact allocation bases are owned by the pthread compatibility
    // layer. Guest pthread objects are commonly stored inside reusable blocks;
    // never pass one of those storage addresses to TryFreeGuestMemory when a
    // stale handle is destroyed.
    private static readonly ConcurrentDictionary<ulong, byte> _opaqueAllocations = new();
    private static readonly bool _tracePthreads =
        string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_LOG_PTHREADS"), "1", StringComparison.Ordinal);
    private static readonly bool _tracePthreadConds =
        _tracePthreads ||
        string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_LOG_PTHREAD_CONDS"), "1", StringComparison.Ordinal);
    private static readonly bool _tracePthreadFastPath =
        string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_LOG_PTHREAD_FASTPATH"), "1", StringComparison.Ordinal);
    private static readonly HashSet<ulong>? _tracePthreadMutexFilter = ParseTraceAddressFilter(
        Environment.GetEnvironmentVariable("SHARPEMU_LOG_PTHREAD_MUTEX_FILTER"));
    private static long _nextSynchronizationWaiterId;
    private static int _pthreadFastPathTraceWritten;
    private static readonly ConcurrentDictionary<ulong, byte> _pthreadFastPathBusyTraced = new();

    // Unlock policy for contended mutexes. Default "wake": the unlocker releases the mutex and
    // wakes the queue head, which re-contends; running threads may barge in meanwhile. FIFO
    // direct handoff (SHARPEMU_MUTEX_HANDOFF=fifo) instead transfers ownership to the head even
    // while it is parked, so every other locker queues behind that thread's host wake-up (a
    // handoff convoy). Each mutex captures the policy when it is created.
    internal static bool DefaultMutexFifoHandoff = string.Equals(
        Environment.GetEnvironmentVariable("SHARPEMU_MUTEX_HANDOFF"), "fifo", StringComparison.OrdinalIgnoreCase);

    private sealed class PthreadMutexState
    {
        public long ProfileIdentity { get; } = MutexHandoffProfile.CreateIdentity();
        public bool FifoHandoff { get; init; } = DefaultMutexFifoHandoff;
        private long _ownerThreadId;
        private int _recursionCount;
        private int _queuedWaiterCount;

        public Lock SyncRoot { get; } = new();
        public ulong OwnerThreadId
        {
            get => unchecked((ulong)Volatile.Read(ref _ownerThreadId));
            set => Volatile.Write(ref _ownerThreadId, unchecked((long)value));
        }

        public int RecursionCount
        {
            get => Volatile.Read(ref _recursionCount);
            set => Volatile.Write(ref _recursionCount, value);
        }

        public int QueuedWaiterCount => Volatile.Read(ref _queuedWaiterCount);
        public int Type { get; set; } = MutexTypeErrorCheck;
        public int Protocol { get; set; }
        public LinkedList<PthreadMutexWaiter> Waiters { get; } = new();
        // SHARPEMU_PERF_MUTEX only: the thread that last took the mutex.
        public string? ProfileOwner;

        public bool TryAcquireUncontended(ulong threadId, bool allowWaiterBarge)
        {
            if (!allowWaiterBarge && QueuedWaiterCount != 0)
            {
                return false;
            }

            return TryAcquireOwner(threadId);
        }

        public bool TryAcquireOwner(ulong threadId)
        {
            if (Interlocked.CompareExchange(
                    ref _ownerThreadId,
                    unchecked((long)threadId),
                    0) != 0)
            {
                return false;
            }

            Volatile.Write(ref _recursionCount, 1);
            return true;
        }

        public bool TryReleaseUncontended(ulong threadId)
        {
            if (QueuedWaiterCount != 0 || RecursionCount != 1)
            {
                return false;
            }

            Volatile.Write(ref _recursionCount, 0);
            if (Interlocked.CompareExchange(
                    ref _ownerThreadId,
                    0,
                    unchecked((long)threadId)) == unchecked((long)threadId))
            {
                return true;
            }

            Volatile.Write(ref _recursionCount, 1);
            return false;
        }

        public int IncrementRecursion() => Interlocked.Increment(ref _recursionCount);

        public int DecrementRecursion() => Interlocked.Decrement(ref _recursionCount);

        public void WaiterAddedLocked() => Interlocked.Increment(ref _queuedWaiterCount);

        public void WaiterRemovedLocked() => Interlocked.Decrement(ref _queuedWaiterCount);
    }

    private sealed class PthreadMutexWaiter
    {
        public required ulong ThreadId { get; init; }
        public required string WakeKey { get; init; }
        public required bool Cooperative { get; set; }
        public ManualResetEventSlim? HostSignal { get; set; }
        public LinkedListNode<PthreadMutexWaiter>? Node { get; set; }
        public int Granted;
        // Set while the waiter spins for its grant instead of parking; the unlock then skips the wake.
        public int Spinning;
        // Release-and-wake only, read and written under SyncRoot. 1 = an unlock (or the park
        // recheck) woke this waiter and it will re-check the mutex before it sleeps again, so a
        // later unlock need not wake it. The waiter clears it in the same SyncRoot section in
        // which it sees the mutex still owned, so that owner's unlock wakes it again.
        public int Woken;
        // Release-and-wake only: set by the waiter once it has lost the mutex to a barging
        // locker after being woken. The next unlock then hands the mutex straight to it
        // (MutexStarvationHandoff) instead of waking it to re-contend.
        public int Starving;
        // The scheduler thread parked on this waiter and the block waiter it parked with, so an
        // unlock readies exactly that thread (IGuestThreadScheduler.WakeBlockedWaiter) instead
        // of scanning every guest thread for the wake key. A cond waiter's mutex re-acquisition
        // carries the cond wait's block waiter. Null until the thread parks.
        public ulong SchedulerThread;
        public IGuestThreadBlockWaiter? BlockWaiter;
        // DSTRACE only: Stopwatch timestamp at enqueue (0 = not a traced lock waiter).
        public long DsEnqueueTs;
        public ulong DsMutexAddress;
        // SHARPEMU_PERF_MUTEX only: queue timestamp and the owner when this waiter queued.
        public long ProfileStart;
        public string? ProfileOwner;
    }

    private sealed class SyncBlockWaiter(Func<int> resume, Func<bool> tryWake) : IGuestThreadBlockWaiter
    {
        public int Resume() => resume();

        public bool TryWake() => tryWake();
    }

    private sealed class PthreadCondState
    {
        public object SyncRoot { get; } = new();
        public LinkedList<PthreadCondWaiter> WaiterQueue { get; } = new();
        public ulong SignalEpoch { get; set; }
        public int Waiters { get; set; }
        public int ClockId { get; init; } = ClockRealtime;
        public int ProcessShared { get; init; } = PthreadProcessPrivate;
    }

    private sealed class PthreadCondWaiter
    {
        public required ulong ThreadId { get; init; }
        public required PthreadMutexState MutexState { get; init; }
        public required string WakeKey { get; init; }
        public required bool Cooperative { get; init; }
        public bool PosixErrors { get; init; }
        public LinkedListNode<PthreadCondWaiter>? Node { get; set; }
        public PthreadMutexWaiter? MutexWaiter { get; set; }
        public GuestTimer? TimeoutTimer { get; set; }
        // Set when the waiter completes, for a waiter on a host thread.
        public ManualResetEventSlim? HostSignal { get; init; }
        public ulong SchedulerThread { get; init; }
        public IGuestThreadBlockWaiter? BlockWaiter;
        // Recursion depth of a recursive mutex released for the wait, restored on re-acquire.
        public int SavedRecursion { get; init; } = 1;
        // 0 = waiting, 1 = signaled, 2 = timed out, 3 = abandoned (the unlock failed).
        public int CompletionStateField;
        public int CompletionState
        {
            get => Volatile.Read(ref CompletionStateField);
            set => Volatile.Write(ref CompletionStateField, value);
        }
    }

    private readonly record struct PthreadMutexAttrState(int Type, int Protocol, int PrioCeiling = 0);

    private readonly record struct PthreadCondAttrState(int ClockId, int ProcessShared);

    static KernelPthreadCompatExports()
    {
        RunSynchronizationSelfChecks();
        GuestThreadExecution.GuestThreadAbandoned += AbandonMutexesOwnedByThread;
    }

    /// <summary>
    /// Force-release mutexes still owned by a guest thread that is being torn
    /// down without a clean unlock (TBB worker_abort, abrupt exit). Otherwise
    /// waiters can spin forever and block splash→first GPU submit.
    /// </summary>
    public static int AbandonMutexesOwnedByThread(ulong threadId, string reason)
    {
        if (threadId == 0)
        {
            return 0;
        }

        var released = 0;
        var abandoned = new List<PthreadMutexState>();
        foreach (var pair in _mutexStates)
        {
            var state = pair.Value;
            lock (state.SyncRoot)
            {
                if (state.OwnerThreadId != threadId || state.RecursionCount <= 0)
                {
                    continue;
                }

                state.OwnerThreadId = 0;
                state.RecursionCount = 0;
                released++;
                Console.Error.WriteLine(
                    $"[LOADER][WARN] pthread_mutex_abandon mutex=0x{pair.Key:X16} " +
                    $"owner={KernelPthreadState.DescribeThreadHandle(threadId)} " +
                    $"reason={reason} waiters={state.Waiters.Count}");
            }

            abandoned.Add(state);
        }

        // Hand each freed mutex on exactly like an unlock (grant or wake the head).
        foreach (var state in abandoned)
        {
            if (state.QueuedWaiterCount != 0)
            {
                WakeFirstMutexWaiter(state);
            }
        }

        if (released > 0)
        {
            Console.Error.Flush();
        }

        return released;
    }

    [SysAbiExport(
        Nid = "aI+OeCz8xrQ",
        ExportName = "scePthreadSelf",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PthreadSelf(CpuContext ctx)
    {
        var currentThreadHandle = KernelPthreadState.GetCurrentThreadHandle();
        if (GuestThreadExecution.CurrentGuestThreadHandle != currentThreadHandle)
        {
            GuestThreadExecution.Scheduler?.RegisterGuestThreadContext(currentThreadHandle, ctx);
        }
        ctx[CpuRegister.Rax] = currentThreadHandle;
        TracePthreadSelf(ctx, currentThreadHandle);
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "EotR8a3ASf4",
        ExportName = "pthread_self",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixPthreadSelf(CpuContext ctx) => PthreadSelf(ctx);

    [SysAbiExport(
        Nid = "3PtV6p3QNX4",
        ExportName = "scePthreadEqual",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PthreadEqual(CpuContext ctx)
    {
        var left = ctx[CpuRegister.Rdi];
        var right = ctx[CpuRegister.Rsi];
        ctx[CpuRegister.Rax] = left == right ? 1UL : 0UL;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "7Xl257M4VNI",
        ExportName = "pthread_equal",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libc")]
    public static int PosixPthreadEqual(CpuContext ctx) => PthreadEqual(ctx);

    // A guest that polls with sched_yield / scePthreadYield burns a core exactly like a short-usleep
    // poller. Same streak policy as KernelRuntimeCompatExports.ShortUsleep: yields less than 2 ms
    // apart form a streak; once a thread has yielded continuously for SHARPEMU_USLEEP_NAP_AFTER_US
    // (default 500 us) each further yield sleeps a nap that ramps up to SHARPEMU_USLEEP_NAP_US
    // (default 250 us, see NapRamp) instead.
    // SHARPEMU_YIELD_NAP=0 restores the bare Thread.Yield().
    private static readonly bool _yieldNapEnabled =
        !string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_YIELD_NAP"), "0", StringComparison.Ordinal) &&
        KernelRuntimeCompatExports.UsleepNapMicroseconds > 0;

    [ThreadStatic]
    private static long _yieldStreakStart;

    [ThreadStatic]
    private static long _lastYield;

    private static bool YieldNapStreak(long now) =>
        YieldNapPolicy(ref _yieldStreakStart, ref _lastYield, now,
            KernelRuntimeCompatExports._usleepStreakGapTicks, KernelRuntimeCompatExports._usleepNapAfterTicks);

    /// <summary>Streak bookkeeping for the yield nap: true when this yield should nap.</summary>
    internal static bool YieldNapPolicy(ref long streakStart, ref long lastYield, long now, long gapTicks, long afterTicks)
    {
        if (lastYield == 0 || now - lastYield > gapTicks)
        {
            streakStart = now;
        }

        lastYield = now;
        return now - streakStart >= afterTicks;
    }

    [SysAbiExport(
        Nid = "T72hz6ffq08",
        ExportName = "scePthreadYield",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PthreadYield(CpuContext ctx)
    {
        _ = ctx;
        var now = Stopwatch.GetTimestamp();
        if (_yieldNapEnabled && YieldNapStreak(now))
        {
            HostTiming.SleepMicroseconds(SharpEmu.HLE.Host.NapRamp.Microseconds(
                now - _yieldStreakStart,
                KernelRuntimeCompatExports._usleepNapAfterTicks,
                KernelRuntimeCompatExports.UsleepNapMicroseconds));
            _lastYield = Stopwatch.GetTimestamp();
            return (int)OrbisGen2Result.ORBIS_GEN2_OK;
        }

        if (!Thread.Yield())
        {
            Thread.Sleep(0);
        }

        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "B5GmVDKwpn0",
        ExportName = "pthread_yield",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixPthreadYield(CpuContext ctx) => PthreadYield(ctx);

    [SysAbiExport(
        Nid = "6XG4B33N09g",
        ExportName = "sched_yield",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int SchedYield(CpuContext ctx) => PthreadYield(ctx);

    [SysAbiExport(
        Nid = "GBUY7ywdULE",
        ExportName = "scePthreadRename",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PthreadRename(CpuContext ctx)
    {
        if (SharpEmu.HLE.Host.GuestThreadQosPolicy.IsEnabled && ctx[CpuRegister.Rsi] != 0)
        {
            Span<byte> renameBytes = stackalloc byte[64];
            if (ctx.Memory.TryRead(ctx[CpuRegister.Rsi], renameBytes))
            {
                var renameLength = renameBytes.IndexOf((byte)0);
                var renamed = System.Text.Encoding.UTF8.GetString(
                    renameLength >= 0 ? renameBytes[..renameLength] : renameBytes);
                SharpEmu.HLE.Host.GuestThreadQosPolicy.OnRename(
                    ctx[CpuRegister.Rdi], GuestThreadExecution.CurrentGuestThreadHandle, renamed);
            }
        }

        if (_tracePthreads)
        {
            var nameAddress = ctx[CpuRegister.Rsi];
            Span<byte> nameBytes = stackalloc byte[64];
            var name = "<unreadable>";
            if (nameAddress != 0 && ctx.Memory.TryRead(nameAddress, nameBytes))
            {
                var length = nameBytes.IndexOf((byte)0);
                name = System.Text.Encoding.UTF8.GetString(length >= 0 ? nameBytes[..length] : nameBytes);
            }
            Console.Error.WriteLine(
                $"[LOADER][TRACE] pthread.rename thread=0x{ctx[CpuRegister.Rdi]:X16} name=\"{name}\"");
        }

        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "EI-5-jlq2dE",
        ExportName = "scePthreadGetthreadid",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PthreadGetthreadid(CpuContext ctx) => PthreadGetthreadidCore(ctx);

    [SysAbiExport(
        Nid = "3eqs37G74-s",
        ExportName = "pthread_getthreadid_np",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixPthreadGetthreadidNp(CpuContext ctx) => PthreadGetthreadidCore(ctx);

    [SysAbiExport(
        Nid = "cmo1RIYva9o",
        ExportName = "scePthreadMutexInit",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PthreadMutexInit(CpuContext ctx) => PthreadMutexInitCore(ctx, ctx[CpuRegister.Rdi], ctx[CpuRegister.Rsi]);

    [SysAbiExport(
        Nid = "2Of0f+3mhhE",
        ExportName = "scePthreadMutexDestroy",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PthreadMutexDestroy(CpuContext ctx) => PthreadMutexDestroyCore(ctx, ctx[CpuRegister.Rdi]);

    [SysAbiExport(
        Nid = "9UK1vLZQft4",
        ExportName = "scePthreadMutexLock",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PthreadMutexLock(CpuContext ctx) => PthreadMutexLockCore(ctx, ctx[CpuRegister.Rdi], tryOnly: false);

    [SysAbiExport(
        Nid = "upoVrzMHFeE",
        ExportName = "scePthreadMutexTrylock",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PthreadMutexTrylock(CpuContext ctx) => PthreadMutexLockCore(ctx, ctx[CpuRegister.Rdi], tryOnly: true);

    [SysAbiExport(
        Nid = "tn3VlD0hG60",
        ExportName = "scePthreadMutexUnlock",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PthreadMutexUnlock(CpuContext ctx) => PthreadMutexUnlockCore(ctx, ctx[CpuRegister.Rdi], requireOwner: true);

    [SysAbiExport(
        Nid = "ttHNfU+qDBU",
        ExportName = "pthread_mutex_init",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixPthreadMutexInit(CpuContext ctx) => GuestErrno.FromSce(PthreadMutexInitCore(ctx, ctx[CpuRegister.Rdi], ctx[CpuRegister.Rsi]));

    [SysAbiExport(
        Nid = "ltCfaGr2JGE",
        ExportName = "pthread_mutex_destroy",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixPthreadMutexDestroy(CpuContext ctx) => GuestErrno.FromSce(PthreadMutexDestroyCore(ctx, ctx[CpuRegister.Rdi]));

    [SysAbiExport(
        Nid = "7H0iTOciTLo",
        ExportName = "pthread_mutex_lock",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixPthreadMutexLock(CpuContext ctx) => PthreadMutexLockCore(ctx, ctx[CpuRegister.Rdi], tryOnly: false, posix: true);

    [SysAbiExport(
        Nid = "K-jXhbt2gn4",
        ExportName = "pthread_mutex_trylock",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixPthreadMutexTrylock(CpuContext ctx) => PthreadMutexLockCore(ctx, ctx[CpuRegister.Rdi], tryOnly: true, posix: true);

    [SysAbiExport(
        Nid = "2Z+PpY6CaJg",
        ExportName = "pthread_mutex_unlock",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixPthreadMutexUnlock(CpuContext ctx) => PthreadMutexUnlockCore(ctx, ctx[CpuRegister.Rdi], requireOwner: true, posix: true);

    /// <summary>scePthreadMutexTimedlock(mutex, usec): a relative timeout in microseconds.</summary>
    [SysAbiExport(
        Nid = "IafI2PxcPnQ",
        ExportName = "scePthreadMutexTimedlock",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PthreadMutexTimedlock(CpuContext ctx) =>
        PthreadMutexLockCore(
            ctx,
            ctx[CpuRegister.Rdi],
            tryOnly: false,
            deadline: GuestTime.DeadlineAfterMicroseconds(unchecked((uint)ctx[CpuRegister.Rsi])));

    /// <summary>pthread_mutex_timedlock(mutex, abstime): CLOCK_REALTIME absolute time.</summary>
    [SysAbiExport(
        Nid = "Io9+nTKXZtA",
        ExportName = "pthread_mutex_timedlock",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixPthreadMutexTimedlock(CpuContext ctx)
    {
        if (!GuestTime.TryReadAbsoluteDeadline(ctx, ctx[CpuRegister.Rsi], GuestTime.ClockRealtime, out var deadline, out var errno))
        {
            return errno;
        }

        return PthreadMutexLockCore(ctx, ctx[CpuRegister.Rdi], tryOnly: false, posix: true, deadline: deadline);
    }

    [SysAbiExport(Nid = "gquEhBrS2iw", ExportName = "scePthreadMutexattrGettype", Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PthreadMutexattrGettype(CpuContext ctx) =>
        PthreadMutexattrGetCore(ctx, ctx[CpuRegister.Rdi], ctx[CpuRegister.Rsi], static attr => attr.Type);

    [SysAbiExport(Nid = "GZFlI7RhuQo", ExportName = "pthread_mutexattr_gettype", Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PosixPthreadMutexattrGettype(CpuContext ctx) =>
        GuestErrno.FromSce(PthreadMutexattrGettype(ctx));

    [SysAbiExport(Nid = "GoTmFeui+hQ", ExportName = "scePthreadMutexattrGetprotocol", Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PthreadMutexattrGetprotocol(CpuContext ctx) =>
        PthreadMutexattrGetCore(ctx, ctx[CpuRegister.Rdi], ctx[CpuRegister.Rsi], static attr => attr.Protocol);

    [SysAbiExport(Nid = "yDaWxUE50s0", ExportName = "pthread_mutexattr_getprotocol", Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PosixPthreadMutexattrGetprotocol(CpuContext ctx) =>
        GuestErrno.FromSce(PthreadMutexattrGetprotocol(ctx));

    [SysAbiExport(Nid = "SgjMpyH9Z9I", ExportName = "scePthreadMutexattrGetprioceiling", Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PthreadMutexattrGetprioceiling(CpuContext ctx) =>
        PthreadMutexattrGetCore(ctx, ctx[CpuRegister.Rdi], ctx[CpuRegister.Rsi], static attr => attr.PrioCeiling);

    [SysAbiExport(Nid = "+m8+quqOwhM", ExportName = "pthread_mutexattr_getprioceiling", Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PosixPthreadMutexattrGetprioceiling(CpuContext ctx) =>
        GuestErrno.FromSce(PthreadMutexattrGetprioceiling(ctx));

    /// <summary>
    /// Stores the ceiling for getprioceiling. PRIO_PROTECT boosting itself is not emulated:
    /// guest priorities are not host priorities (see the scheduler's priority mapping).
    /// </summary>
    [SysAbiExport(Nid = "532IaQguwMg", ExportName = "scePthreadMutexattrSetprioceiling", Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PthreadMutexattrSetprioceiling(CpuContext ctx)
    {
        var ceiling = unchecked((int)ctx[CpuRegister.Rsi]);
        return UpdateMutexAttrCore(ctx, ctx[CpuRegister.Rdi], attr => attr with { PrioCeiling = ceiling });
    }

    [SysAbiExport(Nid = "ZLvf6lVAc4M", ExportName = "pthread_mutexattr_setprioceiling", Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PosixPthreadMutexattrSetprioceiling(CpuContext ctx) =>
        GuestErrno.FromSce(PthreadMutexattrSetprioceiling(ctx));

    private static int PthreadGetthreadidCore(CpuContext ctx)
    {
        ctx[CpuRegister.Rax] = KernelPthreadState.GetCurrentThreadUniqueId();
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "F8bUHwAG284",
        ExportName = "scePthreadMutexattrInit",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PthreadMutexattrInit(CpuContext ctx) => PthreadMutexattrInitCore(ctx, ctx[CpuRegister.Rdi]);

    [SysAbiExport(
        Nid = "smWEktiyyG0",
        ExportName = "scePthreadMutexattrDestroy",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PthreadMutexattrDestroy(CpuContext ctx) => PthreadMutexattrDestroyCore(ctx, ctx[CpuRegister.Rdi]);

    [SysAbiExport(
        Nid = "iMp8QpE+XO4",
        ExportName = "scePthreadMutexattrSettype",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PthreadMutexattrSettype(CpuContext ctx) => PthreadMutexattrSettypeCore(ctx, ctx[CpuRegister.Rdi], unchecked((int)ctx[CpuRegister.Rsi]));

    [SysAbiExport(
        Nid = "1FGvU0i9saQ",
        ExportName = "scePthreadMutexattrSetprotocol",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PthreadMutexattrSetprotocol(CpuContext ctx) => PthreadMutexattrSetprotocolCore(ctx, ctx[CpuRegister.Rdi], unchecked((int)ctx[CpuRegister.Rsi]));

    [SysAbiExport(
        Nid = "dQHWEsJtoE4",
        ExportName = "pthread_mutexattr_init",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixPthreadMutexattrInit(CpuContext ctx) => GuestErrno.FromSce(PthreadMutexattrInitCore(ctx, ctx[CpuRegister.Rdi]));

    [SysAbiExport(
        Nid = "HF7lK46xzjY",
        ExportName = "pthread_mutexattr_destroy",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixPthreadMutexattrDestroy(CpuContext ctx) => GuestErrno.FromSce(PthreadMutexattrDestroyCore(ctx, ctx[CpuRegister.Rdi]));

    [SysAbiExport(
        Nid = "mDmgMOGVUqg",
        ExportName = "pthread_mutexattr_settype",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixPthreadMutexattrSettype(CpuContext ctx) => GuestErrno.FromSce(PthreadMutexattrSettypeCore(ctx, ctx[CpuRegister.Rdi], unchecked((int)ctx[CpuRegister.Rsi])));

    [SysAbiExport(
        Nid = "5txKfcMUAok",
        ExportName = "pthread_mutexattr_setprotocol",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixPthreadMutexattrSetprotocol(CpuContext ctx) => GuestErrno.FromSce(PthreadMutexattrSetprotocolCore(ctx, ctx[CpuRegister.Rdi], unchecked((int)ctx[CpuRegister.Rsi])));

    [SysAbiExport(
        Nid = "2Tb92quprl0",
        ExportName = "scePthreadCondInit",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PthreadCondInit(CpuContext ctx) =>
        PthreadCondInitCore(ctx, ctx[CpuRegister.Rdi], ctx[CpuRegister.Rsi], posixErrors: false);

    [SysAbiExport(
        Nid = "0TyVk4MSLt0",
        ExportName = "pthread_cond_init",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixPthreadCondInit(CpuContext ctx) =>
        PthreadCondInitCore(ctx, ctx[CpuRegister.Rdi], ctx[CpuRegister.Rsi], posixErrors: true);

    [SysAbiExport(
        Nid = "g+PZd2hiacg",
        ExportName = "scePthreadCondDestroy",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PthreadCondDestroy(CpuContext ctx) => PthreadCondDestroyCore(ctx, ctx[CpuRegister.Rdi]);

    [SysAbiExport(
        Nid = "RXXqi4CtF8w",
        ExportName = "pthread_cond_destroy",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixPthreadCondDestroy(CpuContext ctx) => GuestErrno.FromSce(PthreadCondDestroyCore(ctx, ctx[CpuRegister.Rdi]));

    [SysAbiExport(
        Nid = "WKAXJ4XBPQ4",
        ExportName = "scePthreadCondWait",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PthreadCondWait(CpuContext ctx) => PthreadCondWaitCore(ctx, ctx[CpuRegister.Rdi], ctx[CpuRegister.Rsi], deadline: 0);

    [SysAbiExport(
        Nid = "BmMjYxmew1w",
        ExportName = "scePthreadCondTimedwait",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PthreadCondTimedwait(CpuContext ctx) =>
        PthreadCondWaitCore(
            ctx,
            ctx[CpuRegister.Rdi],
            ctx[CpuRegister.Rsi],
            deadline: GuestTime.DeadlineAfterMicroseconds(unchecked((uint)ctx[CpuRegister.Rdx])));

    [SysAbiExport(
        Nid = "kDh-NfxgMtE",
        ExportName = "scePthreadCondSignal",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PthreadCondSignal(CpuContext ctx) => PthreadCondSignalCore(ctx, ctx[CpuRegister.Rdi], broadcast: false);

    [SysAbiExport(
        Nid = "o69RpYO-Mu0",
        ExportName = "scePthreadCondSignalto",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PthreadCondSignalto(CpuContext ctx) =>
        PthreadCondSignalToCore(ctx, ctx[CpuRegister.Rdi], ctx[CpuRegister.Rsi]);

    [SysAbiExport(
        Nid = "JGgj7Uvrl+A",
        ExportName = "scePthreadCondBroadcast",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PthreadCondBroadcast(CpuContext ctx) => PthreadCondSignalCore(ctx, ctx[CpuRegister.Rdi], broadcast: true);

    [SysAbiExport(
        Nid = "Op8TBGY5KHg",
        ExportName = "pthread_cond_wait",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixPthreadCondWait(CpuContext ctx) =>
        PthreadCondWaitCore(ctx, ctx[CpuRegister.Rdi], ctx[CpuRegister.Rsi], deadline: 0, posixErrors: true);

    [SysAbiExport(
        Nid = "27bAgiJmOh0",
        ExportName = "pthread_cond_timedwait",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixPthreadCondTimedwait(CpuContext ctx)
    {
        // abstime on the condition's clock, kept as a 64-bit Stopwatch deadline.
        var clockId = ResolveCondClockId(ctx, ctx[CpuRegister.Rdi]);
        if (!GuestTime.TryReadAbsoluteDeadline(ctx, ctx[CpuRegister.Rdx], clockId, out var deadline, out var errno))
        {
            return errno;
        }

        return PthreadCondWaitCore(ctx, ctx[CpuRegister.Rdi], ctx[CpuRegister.Rsi], deadline, posixErrors: true);
    }

    /// <summary>pthread_cond_reltimedwait_np: like pthread_cond_timedwait with a relative timespec.</summary>
    [SysAbiExport(
        Nid = "K953PF5u6Pc",
        ExportName = "pthread_cond_reltimedwait_np",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixPthreadCondReltimedwaitNp(CpuContext ctx)
    {
        if (!GuestTime.TryReadRelativeDeadline(ctx, ctx[CpuRegister.Rdx], out var deadline, out var errno))
        {
            return errno;
        }

        return PthreadCondWaitCore(ctx, ctx[CpuRegister.Rdi], ctx[CpuRegister.Rsi], deadline, posixErrors: true);
    }

    [SysAbiExport(
        Nid = "mkx2fVhNMsg",
        ExportName = "pthread_cond_broadcast",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixPthreadCondBroadcast(CpuContext ctx) => GuestErrno.FromSce(PthreadCondSignalCore(ctx, ctx[CpuRegister.Rdi], broadcast: true));

    [SysAbiExport(
        Nid = "2MOy+rUfuhQ",
        ExportName = "pthread_cond_signal",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixPthreadCondSignal(CpuContext ctx) => GuestErrno.FromSce(PthreadCondSignalCore(ctx, ctx[CpuRegister.Rdi], broadcast: false));

    [SysAbiExport(
        Nid = "m5-2bsNfv7s",
        ExportName = "scePthreadCondattrInit",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PthreadCondattrInit(CpuContext ctx)
        => PthreadCondattrInitCore(ctx, ctx[CpuRegister.Rdi], posixErrors: false);

    [SysAbiExport(
        Nid = "mKoTx03HRWA",
        ExportName = "pthread_condattr_init",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixPthreadCondattrInit(CpuContext ctx)
        => PthreadCondattrInitCore(ctx, ctx[CpuRegister.Rdi], posixErrors: true);

    [SysAbiExport(
        Nid = "waPcxYiR3WA",
        ExportName = "scePthreadCondattrDestroy",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PthreadCondattrDestroy(CpuContext ctx)
        => PthreadCondattrDestroyCore(ctx, ctx[CpuRegister.Rdi], posixErrors: false);

    [SysAbiExport(
        Nid = "dJcuQVn6-Iw",
        ExportName = "pthread_condattr_destroy",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixPthreadCondattrDestroy(CpuContext ctx)
        => PthreadCondattrDestroyCore(ctx, ctx[CpuRegister.Rdi], posixErrors: true);

    [SysAbiExport(
        Nid = "cTDYxTUNPhM",
        ExportName = "pthread_condattr_getclock",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixPthreadCondattrGetclock(CpuContext ctx)
        => PthreadCondattrGetValueCore(
            ctx,
            ctx[CpuRegister.Rdi],
            ctx[CpuRegister.Rsi],
            getClock: true);

    [SysAbiExport(
        Nid = "EjllaAqAPZo",
        ExportName = "pthread_condattr_setclock",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixPthreadCondattrSetclock(CpuContext ctx)
        => PthreadCondattrSetValueCore(
            ctx,
            ctx[CpuRegister.Rdi],
            unchecked((int)ctx[CpuRegister.Rsi]),
            setClock: true);

    [SysAbiExport(
        Nid = "h0qUqSuOmC8",
        ExportName = "pthread_condattr_getpshared",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixPthreadCondattrGetpshared(CpuContext ctx)
        => PthreadCondattrGetValueCore(
            ctx,
            ctx[CpuRegister.Rdi],
            ctx[CpuRegister.Rsi],
            getClock: false);

    [SysAbiExport(
        Nid = "3BpP850hBT4",
        ExportName = "pthread_condattr_setpshared",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixPthreadCondattrSetpshared(CpuContext ctx)
        => PthreadCondattrSetValueCore(
            ctx,
            ctx[CpuRegister.Rdi],
            unchecked((int)ctx[CpuRegister.Rsi]),
            setClock: false);

    [SysAbiExport(
        Nid = "14bOACANTBo",
        ExportName = "scePthreadOnce",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PthreadOnce(CpuContext ctx)
    {
        var onceAddress = ctx[CpuRegister.Rdi];
        var initRoutine = ctx[CpuRegister.Rsi];
        if (onceAddress == 0 || initRoutine == 0)
        {
            return SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT);
        }

        if (!TryReadInt32(ctx, onceAddress, out var onceValue))
        {
            return SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
        }

        if (onceValue == PthreadOnceDone)
        {
            return SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_OK);
        }

        var gate = GetPthreadOnceGate(onceAddress);
        var shouldCall = false;
        lock (gate)
        {
            if (!TryReadInt32(ctx, onceAddress, out onceValue))
            {
                return SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
            }

            while (onceValue == PthreadOnceInProgress)
            {
                // The initializing thread pulses the gate when it finishes; the slice only
                // re-reads a once word the guest changed behind our back.
                Monitor.Wait(gate, GuestWait.HostWaitSafetySliceMilliseconds);
                if (!TryReadInt32(ctx, onceAddress, out onceValue))
                {
                    return SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
                }
            }

            if (onceValue != PthreadOnceDone)
            {
                if (!TryWriteInt32(ctx, onceAddress, PthreadOnceInProgress))
                {
                    return SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
                }

                shouldCall = true;
            }
        }

        if (shouldCall)
        {
            var scheduler = GuestThreadExecution.Scheduler;
            string? error = null;
            if (scheduler is null ||
                !scheduler.TryCallGuestFunction(ctx, initRoutine, 0, 0, 0, 0, "pthread_once", out error))
            {
                lock (gate)
                {
                    _ = TryWriteInt32(ctx, onceAddress, PthreadOnceUninitialized);
                    Monitor.PulseAll(gate);
                }

                TracePthreadOnce(onceAddress, initRoutine, "failed", error);
                return SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_ERROR_TRY_AGAIN);
            }

            lock (gate)
            {
                if (!TryWriteInt32(ctx, onceAddress, PthreadOnceDone))
                {
                    _ = TryWriteInt32(ctx, onceAddress, PthreadOnceUninitialized);
                    Monitor.PulseAll(gate);
                    return SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
                }

                Monitor.PulseAll(gate);
            }
        }

        TracePthreadOnce(onceAddress, initRoutine, shouldCall ? "call" : "done", null);
        return SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_OK);
    }

    /// <summary>
    /// The POSIX-named alias of <see cref="PthreadOnce"/>. libKernel exports the
    /// same routine under two NIDs, and shipped middleware links the plain name:
    /// DOOM's libcohtml, PlayFab and party modules all import this one rather
    /// than scePthreadOnce.
    /// </summary>
    [SysAbiExport(
        Nid = "Z4QosVuAsA0",
        ExportName = "pthread_once",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PthreadOncePOSIX(CpuContext ctx) => PthreadOnce(ctx);

    /// <summary>
    /// The POSIX-named alias of <see cref="PthreadRename"/>, following the same
    /// two-NID pattern as <see cref="PthreadOncePOSIX"/>.
    /// </summary>
    [SysAbiExport(
        Nid = "9vyP6Z7bqzc",
        ExportName = "pthread_rename_np",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PthreadRenameNpPOSIX(CpuContext ctx) => PthreadRename(ctx);

    private static int PthreadMutexInitCore(CpuContext ctx, ulong mutexAddress, ulong attrAddress)
    {
        if (mutexAddress == 0)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        var attr = ResolveMutexAttrState(ctx, attrAddress);
        var state = new PthreadMutexState
        {
            Type = attr.Type,
            Protocol = attr.Protocol,
        };

        if (!TryAllocateOpaqueObject(ctx, MutexObjectSize, out var handle))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }
        if (!InitializeMutexObject(ctx, handle, state))
        {
            FreeOpaqueObject(ctx, handle);
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        bool written;
        lock (_stateGate)
        {
            _mutexStates[mutexAddress] = state;
            _mutexStates[handle] = state;
            written = KernelMemoryCompatExports.TryWriteUInt64Compat(ctx, mutexAddress, handle);
            if (!written)
            {
                _mutexStates.TryRemove(mutexAddress, out _);
                _mutexStates.TryRemove(handle, out _);
            }
        }

        if (!written)
        {
            FreeOpaqueObject(ctx, handle);
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    private static int PthreadMutexDestroyCore(CpuContext ctx, ulong mutexAddress)
    {
        if (mutexAddress == 0)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        var resolvedAddress = ResolveMutexHandle(ctx, mutexAddress);
        if (!_mutexStates.TryGetValue(resolvedAddress, out var state))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND;
        }

        lock (state.SyncRoot)
        {
            if (state.OwnerThreadId != 0 || state.RecursionCount != 0 || state.Waiters.Count != 0)
            {
                return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_BUSY;
            }

            var allocationAddress = ResolveMutexAllocationAddress(ctx, mutexAddress, resolvedAddress, state);
            _mutexStates.TryRemove(resolvedAddress, out _);
            if (resolvedAddress != mutexAddress)
            {
                _mutexStates.TryRemove(mutexAddress, out _);
            }
            if (allocationAddress != resolvedAddress)
            {
                _mutexStates.TryRemove(allocationAddress, out _);
            }

            _ = KernelMemoryCompatExports.TryWriteUInt64Compat(ctx, mutexAddress, 0);
            FreeOpaqueObject(ctx, allocationAddress);
        }

        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    // pthread_* entry points return the errno itself; scePthread* the sce code.
    private static int PthreadMutexLockCore(
        CpuContext ctx,
        ulong mutexAddress,
        bool tryOnly,
        bool posix = false,
        long deadline = 0)
    {
        var result = PthreadMutexLockSce(ctx, mutexAddress, tryOnly, posix, deadline);
        return posix ? GuestErrno.FromSce(result) : result;
    }

    // Relocking a mutex the caller already owns, as FreeBSD libthr's mutex_self_lock /
    // mutex_self_trylock: a recursive mutex counts up; trylock reports EBUSY; a timed lock
    // sleeps until its deadline and reports ETIMEDOUT; a plain lock reports EDEADLK for every
    // other type. libthr would block a NORMAL mutex forever; EDEADLK is kept for it (#748).
    // ADAPTIVE used to accept the relock as a no-op (#451, 90c72ebec: a UE runtime wrapper
    // locking an adaptive mutex twice around one unlock). A caller that ignores the result
    // keeps the same ownership and the same single unlock as before.
    private static int SelfLockResult(PthreadMutexState state, bool tryOnly, long deadline)
    {
        if (state.Type == MutexTypeRecursive)
        {
            state.IncrementRecursion();
            return (int)OrbisGen2Result.ORBIS_GEN2_OK;
        }

        if (tryOnly)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_BUSY;
        }

        if (deadline != 0)
        {
            HostTiming.SleepUntil(deadline);
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_TIMED_OUT;
        }

        return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_DEADLOCK;
    }

    // The closures live in their own method on purpose: a lambda that captures the lock's
    // parameters makes the compiler allocate the capture class on entry to the method that
    // contains it, which put an 88-byte allocation on every uncontended lock and trylock.
    private static SyncBlockWaiter CreateMutexBlockWaiter(
        CpuContext ctx,
        ulong mutexAddress,
        ulong resolvedAddress,
        PthreadMutexState state,
        PthreadMutexWaiter waiter,
        bool posix,
        long deadline) =>
        new(
            () => CompleteBlockedMutexLock(ctx, mutexAddress, resolvedAddress, state, waiter, posix, deadline),
            () => TryGrantBlockedMutexLock(ctx, mutexAddress, resolvedAddress, state, waiter));

    private static int PthreadMutexLockSce(CpuContext ctx, ulong mutexAddress, bool tryOnly, bool posix, long deadline)
    {
        if (mutexAddress == 0)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        if (!TryResolveMutexState(ctx, mutexAddress, createIfZero: true, out var resolvedAddress, out var state))
        {
            TracePthreadFastPathBusy(tryOnly ? "trylock_missing" : "lock_missing", mutexAddress, resolvedAddress, null, KernelPthreadState.GetCurrentThreadHandle(), (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND);
            TracePthreadMutex(ctx, tryOnly ? "trylock" : "lock", mutexAddress, resolvedAddress, null, KernelPthreadState.GetCurrentThreadHandle(), (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND);
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND;
        }

        var currentThreadId = KernelPthreadState.GetCurrentThreadHandle();
        if (state.TryAcquireUncontended(currentThreadId, allowWaiterBarge: tryOnly || !state.FifoHandoff))
        {
            if (MutexContentionProfile.Enabled) state.ProfileOwner = MutexContentionProfile.CurrentThreadName();
            TracePthreadMutex(ctx, tryOnly ? "trylock" : "lock", mutexAddress, resolvedAddress, state, currentThreadId, (int)OrbisGen2Result.ORBIS_GEN2_OK);
            return (int)OrbisGen2Result.ORBIS_GEN2_OK;
        }

        if (state.OwnerThreadId == currentThreadId)
        {
            // Only this thread moves ownership away from itself, so the read is stable.
            var selfResult = SelfLockResult(state, tryOnly, deadline);
            TracePthreadFastPathBusy(tryOnly ? "trylock_self" : "lock_self", mutexAddress, resolvedAddress, state, currentThreadId, selfResult);
            TracePthreadMutex(ctx, tryOnly ? "trylock" : "lock", mutexAddress, resolvedAddress, state, currentThreadId, selfResult);
            return selfResult;
        }

        if (deadline != 0 && !tryOnly && Stopwatch.GetTimestamp() >= deadline)
        {
            // An expired timed lock still takes a free mutex but never queues.
            var expiredResult = state.TryAcquireOwner(currentThreadId)
                ? (int)OrbisGen2Result.ORBIS_GEN2_OK
                : (int)OrbisGen2Result.ORBIS_GEN2_ERROR_TIMED_OUT;
            TracePthreadMutex(ctx, "timedlock", mutexAddress, resolvedAddress, state, currentThreadId, expiredResult);
            return expiredResult;
        }

        var canCooperativelyBlock = !tryOnly &&
            GuestThreadExecution.IsGuestThread &&
            GuestThreadExecution.TryGetCurrentImportCallFrame(out _);
        PthreadMutexWaiter? waiter = null;
        var acquiredWhileQueueing = false;
        lock (state.SyncRoot)
        {
            if (state.OwnerThreadId == currentThreadId)
            {
                // Unreachable after the unlocked check above; never sleep under SyncRoot.
                var selfResult = SelfLockResult(state, tryOnly, deadline: 0);
                TracePthreadMutex(ctx, tryOnly ? "trylock" : "lock", mutexAddress, resolvedAddress, state, currentThreadId, selfResult);
                return selfResult;
            }

            // pthread_mutex_trylock succeeds whenever the mutex is not currently
            // held; unlike the blocking lock it does not queue behind waiters
            // (POSIX gives it no fairness obligation). Gating trylock on an empty
            // wait queue is wrong and, worse, lets a single stale/undrainable
            // waiter wedge a spin-on-trylock loop forever even though the mutex
            // is free (owner==0). Under FIFO handoff the blocking lock still
            // honours FIFO so real blocked waiters are not starved by a barging
            // locker; under release-and-wake a running locker may barge too.
            if (state.OwnerThreadId == 0 &&
                (tryOnly || !state.FifoHandoff || state.Waiters.Count == 0) &&
                state.TryAcquireOwner(currentThreadId))
            {
                TracePthreadMutex(ctx, tryOnly ? "trylock" : "lock", mutexAddress, resolvedAddress, state, currentThreadId, (int)OrbisGen2Result.ORBIS_GEN2_OK);
                return (int)OrbisGen2Result.ORBIS_GEN2_OK;
            }

            if (tryOnly)
            {
                TracePthreadFastPathBusy("trylock", mutexAddress, resolvedAddress, state, currentThreadId, (int)OrbisGen2Result.ORBIS_GEN2_ERROR_BUSY);
                TracePthreadMutex(ctx, "trylock", mutexAddress, resolvedAddress, state, currentThreadId, (int)OrbisGen2Result.ORBIS_GEN2_ERROR_BUSY);
                return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_BUSY;
            }

            waiter = EnqueueMutexWaiterLocked(state, currentThreadId, canCooperativelyBlock);
            if (MutexContentionProfile.Enabled)
            {
                waiter.ProfileStart = Stopwatch.GetTimestamp();
                waiter.ProfileOwner = state.ProfileOwner;
            }
            acquiredWhileQueueing = TryGrantMutexWaiterLocked(state, waiter);
        }

        var dsSync = DsTrace.Sync && DsTrace.Armed;
        var dsSpinStart = 0L;
        var dsOwner = 0UL;
        var dsQlen = 0;
        if (dsSync)
        {
            waiter!.DsEnqueueTs = Stopwatch.GetTimestamp();
            waiter.DsMutexAddress = mutexAddress;
            dsOwner = state.OwnerThreadId;
            dsQlen = state.QueuedWaiterCount;
            dsSpinStart = waiter.DsEnqueueTs;
        }

        var dsSpun = acquiredWhileQueueing || SpinForMutexGrant(state, waiter!);
        var dsSpinNs = dsSync ? (ulong)DsTrace.TicksToNs(Stopwatch.GetTimestamp() - dsSpinStart) : 0UL;
        if (dsSync && dsSpun)
        {
            DsTrace.Emit(DsEv.MXQ, mutexAddress, currentThreadId, dsOwner, (ulong)dsQlen, dsSpinNs, 0UL, DsTrace.CurrentRa1);
        }

        if (dsSpun)
        {
            if (MutexContentionProfile.Enabled) RecordMutexContention(mutexAddress, state, waiter!);
            waiter!.HostSignal?.Dispose();
            TracePthreadMutex(ctx, "lock", mutexAddress, resolvedAddress, state, currentThreadId, (int)OrbisGen2Result.ORBIS_GEN2_OK);
            return (int)OrbisGen2Result.ORBIS_GEN2_OK;
        }

        if (canCooperativelyBlock && waiter is not null &&
            RequestMutexBlock(ctx, waiter, deadline, CreateMutexBlockWaiter(ctx, mutexAddress, resolvedAddress, state, waiter, posix, deadline)))
        {
            if (dsSync)
            {
                DsTrace.Emit(DsEv.MXQ, mutexAddress, currentThreadId, dsOwner, (ulong)dsQlen, dsSpinNs, 1UL, DsTrace.CurrentRa1);
            }

            TracePthreadMutex(ctx, "lock-block", mutexAddress, resolvedAddress, state, currentThreadId, (int)OrbisGen2Result.ORBIS_GEN2_OK);
            return (int)OrbisGen2Result.ORBIS_GEN2_OK;
        }

        if (dsSync)
        {
            DsTrace.Emit(DsEv.MXQ, mutexAddress, currentThreadId, dsOwner, (ulong)dsQlen, dsSpinNs, 2UL, DsTrace.CurrentRa1);
        }

        var hostResult = WaitForHostMutexLock(ctx, state, waiter!, deadline: deadline);
        if (MutexContentionProfile.Enabled && hostResult == (int)OrbisGen2Result.ORBIS_GEN2_OK) RecordMutexContention(mutexAddress, state, waiter!);
        if (dsSync)
        {
            DsTrace.Emit(
                DsEv.MXR,
                mutexAddress,
                currentThreadId,
                (ulong)DsTrace.TicksToNs(Stopwatch.GetTimestamp() - waiter!.DsEnqueueTs));
        }

        TracePthreadMutex(ctx, "lock", mutexAddress, resolvedAddress, state, currentThreadId, hostResult);
        return hostResult;
    }

    private static int PthreadMutexUnlockCore(CpuContext ctx, ulong mutexAddress, bool requireOwner, bool posix = false)
    {
        var result = PthreadMutexUnlockSce(ctx, mutexAddress, requireOwner);
        return posix ? GuestErrno.FromSce(result) : result;
    }

    private static int PthreadMutexUnlockSce(CpuContext ctx, ulong mutexAddress, bool requireOwner)
    {
        if (mutexAddress == 0)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        if (!TryResolveMutexState(ctx, mutexAddress, createIfZero: true, out var resolvedAddress, out var state))
        {
            TracePthreadMutex(ctx, "unlock", mutexAddress, resolvedAddress, null, KernelPthreadState.GetCurrentThreadHandle(), (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND);
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND;
        }

        var currentThreadId = KernelPthreadState.GetCurrentThreadHandle();
        if (state.OwnerThreadId == currentThreadId)
        {
            if (state.RecursionCount > 1)
            {
                state.DecrementRecursion();
                TracePthreadMutex(ctx, "unlock", mutexAddress, resolvedAddress, state, currentThreadId, (int)OrbisGen2Result.ORBIS_GEN2_OK);
                return (int)OrbisGen2Result.ORBIS_GEN2_OK;
            }

            if (state.TryReleaseUncontended(currentThreadId))
            {
                ProfileMutexHandoff(state, "Released", mutexAddress);
                if (state.QueuedWaiterCount != 0)
                {
                    WakeFirstMutexWaiter(state, mutexAddress, currentThreadId);
                }

                TracePthreadMutex(ctx, "unlock", mutexAddress, resolvedAddress, state, currentThreadId, (int)OrbisGen2Result.ORBIS_GEN2_OK);
                return (int)OrbisGen2Result.ORBIS_GEN2_OK;
            }
        }

        PthreadMutexWaiter? nextWaiter = null;
        PthreadMutexWaiter? wokenWaiter = null;
        var wokenCooperative = false;
        lock (state.SyncRoot)
        {
            if (state.RecursionCount <= 0)
            {
                // Unlocking a mutex nobody holds: EPERM, as libthr's owner check.
                TracePthreadFastPathUnlock(ctx, mutexAddress, resolvedAddress, state, currentThreadId);
                TracePthreadMutex(ctx, "unlock", mutexAddress, resolvedAddress, state, currentThreadId, (int)OrbisGen2Result.ORBIS_GEN2_ERROR_PERMISSION_DENIED);
                return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_PERMISSION_DENIED;
            }

            if (requireOwner && state.OwnerThreadId != currentThreadId)
            {
                TracePthreadMutex(ctx, "unlock", mutexAddress, resolvedAddress, state, currentThreadId, (int)OrbisGen2Result.ORBIS_GEN2_ERROR_PERMISSION_DENIED);
                return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_PERMISSION_DENIED;
            }

            state.RecursionCount--;
            if (state.RecursionCount == 0 && TryHandOffToStarvingHeadLocked(state))
            {
                // Ownership moved to the starving head; it is running or host-signalled.
            }
            else if (state.RecursionCount == 0)
            {
                state.OwnerThreadId = 0;
                ProfileMutexHandoff(state, "Released", mutexAddress);

                if (!state.FifoHandoff)
                {
                    // Release-and-wake: the mutex is already free; wake the head so it
                    // re-contends. A running thread may take the mutex first; the head
                    // then keeps its place and the new owner's unlock wakes it again.
                    // Barging is what keeps a free mutex with a queued waiter from
                    // wedging later lockers (see the FIFO note below).
                    wokenWaiter = SignalMutexHeadLocked(state, out wokenCooperative);
                }
                else if (state.Waiters.First is { } headNode &&
                    TryGrantMutexWaiterLocked(state, headNode.Value))
                {
                    // FIFO handoff: hand the mutex directly to the head waiter instead of only
                    // waking it and relying on it to re-acquire. A woken waiter that
                    // fails to self-grant (its wake races or is lost) would leave the
                    // mutex "free with a queued waiter"; the FIFO fast-acquire path refuses
                    // such a mutex (OwnerThreadId == 0 && Waiters.Count == 0), so every
                    // later locker — including the game's main thread — then queues
                    // behind a head that never advances and the process wedges.
                    nextWaiter = headNode.Value;
                    if (!nextWaiter.Cooperative)
                    {
                        nextWaiter.HostSignal!.Set();
                    }
                }
            }
        }

        if (wokenWaiter is not null)
        {
            FinishMutexHeadWake(state, wokenWaiter, wokenCooperative, mutexAddress, currentThreadId);
        }

        // A granted head that is still spinning sees the grant itself.
        if (nextWaiter is { Cooperative: true } && Volatile.Read(ref nextWaiter.Spinning) == 0)
        {
            WakeProfiledMutexWaiter(state, nextWaiter);
        }

        if (nextWaiter is not null && DsTrace.Sync && DsTrace.Armed)
        {
            DsTrace.Emit(
                DsEv.MXH,
                mutexAddress,
                currentThreadId,
                nextWaiter.ThreadId,
                nextWaiter.Cooperative && Volatile.Read(ref nextWaiter.Spinning) == 0 ? 1UL : 0UL,
                (ulong)state.QueuedWaiterCount);
        }

        TracePthreadMutex(ctx, "unlock", mutexAddress, resolvedAddress, state, currentThreadId, (int)OrbisGen2Result.ORBIS_GEN2_OK);
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    private static int PthreadMutexattrInitCore(CpuContext ctx, ulong attrAddress)
    {
        if (attrAddress == 0)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        if (!TryAllocateOpaqueObject(ctx, MutexAttrObjectSize, out var handle))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        var initialState = new PthreadMutexAttrState(MutexTypeErrorCheck, 0);
        if (!WriteMutexAttrObject(ctx, handle, initialState))
        {
            FreeOpaqueObject(ctx, handle);
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        lock (_stateGate)
        {
            _mutexAttrStates[attrAddress] = initialState;
            _mutexAttrStates[handle] = initialState;
        }

        if (!KernelMemoryCompatExports.TryWriteUInt64Compat(ctx, attrAddress, handle))
        {
            lock (_stateGate)
            {
                _mutexAttrStates.Remove(attrAddress);
                _mutexAttrStates.Remove(handle);
            }
            FreeOpaqueObject(ctx, handle);

            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    private static int PthreadMutexattrDestroyCore(CpuContext ctx, ulong attrAddress)
    {
        if (attrAddress == 0)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        var resolvedAddress = ResolveMutexAttrHandle(ctx, attrAddress);
        var allocationAddress = resolvedAddress;
        lock (_stateGate)
        {
            if (KernelMemoryCompatExports.TryReadUInt64Compat(ctx, attrAddress, out var pointedHandle) &&
                pointedHandle != 0 &&
                _mutexAttrStates.ContainsKey(pointedHandle))
            {
                allocationAddress = pointedHandle;
            }

            _mutexAttrStates.Remove(resolvedAddress);
            if (resolvedAddress != attrAddress)
            {
                _mutexAttrStates.Remove(attrAddress);
            }
        }

        FreeOpaqueObject(ctx, allocationAddress);
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    // PTHREAD_MUTEX_ERRORCHECK(1) .. PTHREAD_MUTEX_ADAPTIVE_NP(4); anything else is EINVAL.
    private static int PthreadMutexattrSettypeCore(CpuContext ctx, ulong attrAddress, int type)
    {
        if (type is < MutexTypeErrorCheck or > MutexTypeAdaptiveNp)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        return UpdateMutexAttrCore(ctx, attrAddress, attr => attr with { Type = type });
    }

    // PTHREAD_PRIO_NONE(0), PTHREAD_PRIO_INHERIT(1), PTHREAD_PRIO_PROTECT(2).
    private static int PthreadMutexattrSetprotocolCore(CpuContext ctx, ulong attrAddress, int protocol)
    {
        if (protocol is < 0 or > 2)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        return UpdateMutexAttrCore(ctx, attrAddress, attr => attr with { Protocol = protocol });
    }

    private static int UpdateMutexAttrCore(
        CpuContext ctx,
        ulong attrAddress,
        Func<PthreadMutexAttrState, PthreadMutexAttrState> update)
    {
        if (attrAddress == 0)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        var resolvedAddress = ResolveMutexAttrHandle(ctx, attrAddress);
        PthreadMutexAttrState updatedState;
        lock (_stateGate)
        {
            if (!_mutexAttrStates.TryGetValue(resolvedAddress, out var state))
            {
                state = new PthreadMutexAttrState(MutexTypeErrorCheck, 0);
            }

            updatedState = update(state);
            _mutexAttrStates[resolvedAddress] = updatedState;
            if (resolvedAddress != attrAddress)
            {
                _mutexAttrStates[attrAddress] = updatedState;
            }
        }

        return WriteMutexAttrObject(ctx, resolvedAddress, updatedState)
            ? (int)OrbisGen2Result.ORBIS_GEN2_OK
            : (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
    }

    private static int PthreadMutexattrGetCore(
        CpuContext ctx,
        ulong attrAddress,
        ulong outAddress,
        Func<PthreadMutexAttrState, int> select)
    {
        if (attrAddress == 0 || outAddress == 0)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        var value = select(ResolveMutexAttrState(ctx, attrAddress));
        return TryWriteInt32(ctx, outAddress, value)
            ? (int)OrbisGen2Result.ORBIS_GEN2_OK
            : (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
    }

    private static int PthreadCondattrInitCore(CpuContext ctx, ulong attrAddress, bool posixErrors)
    {
        if (attrAddress == 0)
        {
            return CondAttrInvalidArgument(posixErrors);
        }

        if (!TryAllocateOpaqueObject(ctx, CondAttrObjectSize, out var handle))
        {
            return posixErrors
                ? PosixOutOfMemory
                : (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        var initialState = new PthreadCondAttrState(ClockRealtime, PthreadProcessPrivate);
        if (!WriteCondAttrObject(ctx, handle, initialState))
        {
            FreeOpaqueObject(ctx, handle);
            return posixErrors
                ? PosixBadAddress
                : (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        lock (_stateGate)
        {
            _condAttrStates[attrAddress] = initialState;
            _condAttrStates[handle] = initialState;
        }

        if (!KernelMemoryCompatExports.TryWriteUInt64Compat(ctx, attrAddress, handle))
        {
            lock (_stateGate)
            {
                _condAttrStates.Remove(attrAddress);
                _condAttrStates.Remove(handle);
            }

            FreeOpaqueObject(ctx, handle);
            return posixErrors
                ? PosixBadAddress
                : (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    private static int PthreadCondattrDestroyCore(CpuContext ctx, ulong attrAddress, bool posixErrors)
    {
        if (!TryResolveCondAttrState(ctx, attrAddress, out var resolvedAddress, out _))
        {
            return CondAttrInvalidArgument(posixErrors);
        }

        lock (_stateGate)
        {
            _condAttrStates.Remove(resolvedAddress);
            if (resolvedAddress != attrAddress)
            {
                _condAttrStates.Remove(attrAddress);
            }
        }

        FreeOpaqueObject(ctx, resolvedAddress);
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    private static int PthreadCondattrGetValueCore(
        CpuContext ctx,
        ulong attrAddress,
        ulong valueAddress,
        bool getClock)
    {
        if (valueAddress == 0 ||
            !TryResolveCondAttrState(ctx, attrAddress, out _, out var state))
        {
            return PosixInvalidArgument;
        }

        var value = getClock ? state.ClockId : state.ProcessShared;
        return TryWriteUInt32(ctx, valueAddress, unchecked((uint)value))
            ? (int)OrbisGen2Result.ORBIS_GEN2_OK
            : PosixBadAddress;
    }

    private static int PthreadCondattrSetValueCore(
        CpuContext ctx,
        ulong attrAddress,
        int value,
        bool setClock)
    {
        if ((setClock && value is not ClockRealtime and not ClockMonotonic) ||
            (!setClock && value is not PthreadProcessPrivate and not PthreadProcessShared) ||
            !TryResolveCondAttrState(ctx, attrAddress, out var resolvedAddress, out var state))
        {
            return PosixInvalidArgument;
        }

        var updatedState = setClock
            ? state with { ClockId = value }
            : state with { ProcessShared = value };
        lock (_stateGate)
        {
            _condAttrStates[resolvedAddress] = updatedState;
            if (resolvedAddress != attrAddress)
            {
                _condAttrStates[attrAddress] = updatedState;
            }
        }

        return WriteCondAttrObject(ctx, resolvedAddress, updatedState)
            ? (int)OrbisGen2Result.ORBIS_GEN2_OK
            : PosixBadAddress;
    }

    private static bool TryResolveCondAttrState(
        CpuContext ctx,
        ulong attrAddress,
        out ulong resolvedAddress,
        out PthreadCondAttrState state)
    {
        resolvedAddress = 0;
        state = default;
        if (attrAddress == 0)
        {
            return false;
        }

        if (KernelMemoryCompatExports.TryReadUInt64Compat(ctx, attrAddress, out var pointedHandle) &&
            pointedHandle != 0)
        {
            lock (_stateGate)
            {
                if (_condAttrStates.TryGetValue(pointedHandle, out state))
                {
                    resolvedAddress = pointedHandle;
                    return true;
                }
            }
        }

        lock (_stateGate)
        {
            if (_condAttrStates.TryGetValue(attrAddress, out state))
            {
                resolvedAddress = attrAddress;
                return true;
            }
        }

        return false;
    }

    private static int CondAttrInvalidArgument(bool posixErrors) =>
        posixErrors
            ? PosixInvalidArgument
            : (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;

    private static ulong ResolveMutexHandle(CpuContext ctx, ulong mutexAddress)
    {
        if (mutexAddress == 0)
        {
            return 0;
        }

        var hasPointedHandle =
            KernelMemoryCompatExports.TryReadUInt64Compat(ctx, mutexAddress, out var pointedHandle) &&
            pointedHandle != 0 &&
            pointedHandle != mutexAddress;

        if (_mutexStates.TryGetValue(mutexAddress, out var cachedState))
        {
            return hasPointedHandle &&
                   _mutexStates.TryGetValue(pointedHandle, out var pointedState) &&
                   !ReferenceEquals(pointedState, cachedState)
                ? pointedHandle
                : mutexAddress;
        }

        if (hasPointedHandle && _mutexStates.ContainsKey(pointedHandle))
        {
            return pointedHandle;
        }

        return mutexAddress;
    }

    private static ulong ResolveMutexAllocationAddress(
        CpuContext ctx,
        ulong mutexAddress,
        ulong resolvedAddress,
        PthreadMutexState state)
    {
        if (KernelMemoryCompatExports.TryReadUInt64Compat(ctx, mutexAddress, out var pointedHandle) &&
            pointedHandle != 0 &&
            _mutexStates.TryGetValue(pointedHandle, out var pointedState) &&
            ReferenceEquals(pointedState, state))
        {
            return pointedHandle;
        }

        return resolvedAddress;
    }

    private static bool TryResolveMutexState(CpuContext ctx, ulong mutexAddress, bool createIfZero, out ulong resolvedAddress, [NotNullWhen(true)] out PthreadMutexState? state)
    {
        resolvedAddress = 0;
        state = null;
        if (mutexAddress == 0)
        {
            return false;
        }

    retry:
        var hasPointedHandle = KernelMemoryCompatExports.TryReadUInt64Compat(ctx, mutexAddress, out var pointedHandle);

        if (_mutexStates.TryGetValue(mutexAddress, out state))
        {
            // `mutexAddress` is often the address of the guest's ScePthreadMutex
            // variable rather than the handle itself, and that storage is
            // reusable — a stack frame recycles the slot, or the guest assigns a
            // different mutex to it. The slot therefore outranks anything cached
            // under its address: keeping the stale entry would resolve a release
            // onto the wrong mutex, leave the real one owned forever and wedge
            // every waiter on it (Demon's Souls' Scream audio engine did exactly
            // this and spun on scePthreadMutexTrylock).
            if (hasPointedHandle &&
                pointedHandle != 0 &&
                pointedHandle != mutexAddress &&
                _mutexStates.TryGetValue(pointedHandle, out var pointedState) &&
                !ReferenceEquals(pointedState, state))
            {
                _mutexStates[mutexAddress] = pointedState;
                resolvedAddress = pointedHandle;
                state = pointedState;
                return true;
            }

            // The slot's content is the generation tag of an alias entry: an alias exists only
            // while its slot holds the handle it was created for. A slot that reads 0 or the
            // static initializer again was re-initialised (freed and reused memory, a new
            // PTHREAD_MUTEX_INITIALIZER) without a destroy; the cached state belongs to the
            // old mutex (possibly still "owned" by a thread that is gone), so drop the alias
            // and treat the slot as the new mutex it now is. Handle keys (our own opaque
            // objects) are not aliases and are never dropped here.
            if (hasPointedHandle && pointedHandle is 0 or StaticAdaptiveMutexInitializer)
            {
                switch (TryDropStaleMutexAlias(ctx, mutexAddress, state, pointedHandle))
                {
                    case StaleAliasCheck.Changed:
                        // Another thread replaced the alias or rewrote the slot meanwhile.
                        state = null;
                        goto retry;
                    case StaleAliasCheck.Kept:
                        resolvedAddress = mutexAddress;
                        return true;
                }
            }
            else
            {
                resolvedAddress = mutexAddress;
                return true;
            }

            TracePthreadMutex(ctx, "stale-alias-dropped", mutexAddress, mutexAddress, state, KernelPthreadState.GetCurrentThreadHandle(), 0);
            state = null;
        }

        if (!hasPointedHandle)
        {
            return false;
        }

        if (pointedHandle == StaticAdaptiveMutexInitializer)
        {
            return CreateImplicitMutexState(ctx, mutexAddress, MutexTypeAdaptiveNp, out resolvedAddress, out state);
        }

        if (pointedHandle != 0 && pointedHandle != mutexAddress && _mutexStates.TryGetValue(pointedHandle, out state))
        {
            _mutexStates[mutexAddress] = state;
            resolvedAddress = pointedHandle;
            return true;
        }

        if (pointedHandle != 0)
        {
            resolvedAddress = pointedHandle;
            return false;
        }

        if (!createIfZero)
        {
            resolvedAddress = mutexAddress;
            return false;
        }

        return CreateImplicitMutexState(ctx, mutexAddress, MutexTypeErrorCheck, out resolvedAddress, out state);
    }

    private enum StaleAliasCheck
    {
        Kept,
        Dropped,
        Changed,
    }

    /// <summary>
    /// Removes the alias <paramref name="mutexAddress"/> → <paramref name="state"/> if, under the
    /// state gate, it is still the alias and its slot still reads <paramref name="slotSeen"/>
    /// (0 or the static initializer). Creators publish an alias and write its slot inside the
    /// gate, so a reader whose unlocked view is out of date gets <see cref="StaleAliasCheck.Changed"/>
    /// and resolves again instead of dropping a live alias or keeping a replaced one.
    /// </summary>
    private static StaleAliasCheck TryDropStaleMutexAlias(CpuContext ctx, ulong mutexAddress, PthreadMutexState state, ulong slotSeen)
    {
        lock (_stateGate)
        {
            if (!_mutexStates.TryGetValue(mutexAddress, out var current) || !ReferenceEquals(current, state) ||
                !KernelMemoryCompatExports.TryReadUInt64Compat(ctx, mutexAddress, out var slot) || slot != slotSeen)
            {
                return StaleAliasCheck.Changed;
            }

            if (_opaqueAllocations.ContainsKey(mutexAddress))
            {
                return StaleAliasCheck.Kept;
            }

            _mutexStates.TryRemove(mutexAddress, out _);
            return StaleAliasCheck.Dropped;
        }
    }

    private static ulong ResolveMutexAttrHandle(CpuContext ctx, ulong attrAddress)
    {
        if (attrAddress == 0)
        {
            return 0;
        }

        if (KernelMemoryCompatExports.TryReadUInt64Compat(ctx, attrAddress, out var pointedHandle) && pointedHandle != 0)
        {
            lock (_stateGate)
            {
                if (_mutexAttrStates.ContainsKey(pointedHandle))
                {
                    return pointedHandle;
                }
            }
        }

        lock (_stateGate)
        {
            if (_mutexAttrStates.ContainsKey(attrAddress))
            {
                return attrAddress;
            }
        }

        return attrAddress;
    }

    private static PthreadMutexAttrState ResolveMutexAttrState(CpuContext ctx, ulong attrAddress)
    {
        if (attrAddress == 0)
        {
            return new PthreadMutexAttrState(MutexTypeErrorCheck, 0);
        }

        var resolvedAddress = ResolveMutexAttrHandle(ctx, attrAddress);
        lock (_stateGate)
        {
            return _mutexAttrStates.TryGetValue(resolvedAddress, out var state)
                ? state
                : new PthreadMutexAttrState(MutexTypeErrorCheck, 0);
        }
    }

    private static ulong ResolveCondHandle(CpuContext ctx, ulong condAddress)
    {
        if (condAddress == 0)
        {
            return 0;
        }

        lock (_stateGate)
        {
            if (_condStates.ContainsKey(condAddress))
            {
                return condAddress;
            }
        }

        if (KernelMemoryCompatExports.TryReadUInt64Compat(ctx, condAddress, out var pointedHandle) && pointedHandle != 0)
        {
            lock (_stateGate)
            {
                if (_condStates.ContainsKey(pointedHandle))
                {
                    return pointedHandle;
                }
            }
        }

        return condAddress;
    }

    private static bool TryResolveCondState(CpuContext? ctx, ulong condAddress, bool createIfZero, out ulong resolvedAddress, [NotNullWhen(true)] out PthreadCondState? state)
    {
        resolvedAddress = 0;
        state = null;
        if (condAddress == 0)
        {
            return false;
        }

        lock (_stateGate)
        {
            if (_condStates.TryGetValue(condAddress, out state))
            {
                resolvedAddress = condAddress;
                return true;
            }
        }

        if (ctx is null || !KernelMemoryCompatExports.TryReadUInt64Compat(ctx, condAddress, out var pointedHandle))
        {
            return false;
        }

        if (pointedHandle != 0)
        {
            lock (_stateGate)
            {
                if (_condStates.TryGetValue(pointedHandle, out state))
                {
                    _condStates[condAddress] = state;
                    resolvedAddress = pointedHandle;
                    return true;
                }
            }

            resolvedAddress = pointedHandle;
            return false;
        }

        if (!createIfZero)
        {
            resolvedAddress = condAddress;
            return false;
        }

        var createdState = new PthreadCondState();
        if (!TryAllocateOpaqueObject(ctx, CondObjectSize, out var handle))
        {
            return false;
        }

        lock (_stateGate)
        {
            _condStates[condAddress] = createdState;
            _condStates[handle] = createdState;
        }

        if (!KernelMemoryCompatExports.TryWriteUInt64Compat(ctx, condAddress, handle))
        {
            lock (_stateGate)
            {
                _condStates.Remove(condAddress);
                _condStates.Remove(handle);
            }
            FreeOpaqueObject(ctx, handle);

            return false;
        }

        resolvedAddress = handle;
        state = createdState;
        return true;
    }

    private static bool TryAllocateOpaqueObject(CpuContext ctx, int size, out ulong address)
    {
        address = 0;
        if (ctx.Memory is not IGuestMemoryAllocator allocator ||
            !allocator.TryAllocateGuestMemory((ulong)size, alignment: 0x10, out address))
        {
            return false;
        }

        _opaqueAllocations[address] = 0;

        Span<byte> initialData = stackalloc byte[size];
        initialData.Clear();
        if (ctx.Memory.TryWrite(address, initialData))
        {
            return true;
        }

        FreeOpaqueObject(ctx, address);
        address = 0;
        return false;
    }

    private static void FreeOpaqueObject(CpuContext ctx, ulong address)
    {
        if (address != 0 &&
            _opaqueAllocations.TryRemove(address, out _) &&
            ctx.Memory is IGuestMemoryAllocator allocator)
        {
            _ = allocator.TryFreeGuestMemory(address);
        }
    }

    private static bool InitializeMutexObject(CpuContext ctx, ulong address, PthreadMutexState state) =>
        TryWriteUInt32(ctx, address + 0x20, unchecked((uint)state.Type)) &&
        TryWriteUInt32(ctx, address + 0x3C, unchecked((uint)state.Protocol));

    private static bool WriteMutexAttrObject(CpuContext ctx, ulong address, PthreadMutexAttrState state) =>
        TryWriteUInt32(ctx, address, unchecked((uint)state.Type)) &&
        TryWriteUInt32(ctx, address + 4, unchecked((uint)state.Protocol));

    private static bool WriteCondAttrObject(CpuContext ctx, ulong address, PthreadCondAttrState state) =>
        TryWriteUInt32(ctx, address, unchecked((uint)state.ClockId)) &&
        TryWriteUInt32(ctx, address + 4, unchecked((uint)state.ProcessShared));

    private static bool TryWriteUInt32(CpuContext ctx, ulong address, uint value)
    {
        Span<byte> bytes = stackalloc byte[sizeof(uint)];
        BitConverter.TryWriteBytes(bytes, value);
        return ctx.Memory.TryWrite(address, bytes);
    }

    private static int PthreadCondInitCore(
        CpuContext ctx,
        ulong condAddress,
        ulong attrAddress,
        bool posixErrors)
    {
        if (condAddress == 0)
        {
            return posixErrors
                ? PosixInvalidArgument
                : (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        var attrState = new PthreadCondAttrState(ClockRealtime, PthreadProcessPrivate);
        if (attrAddress != 0 &&
            !TryResolveCondAttrState(ctx, attrAddress, out _, out attrState))
        {
            return posixErrors
                ? PosixInvalidArgument
                : (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        if (!TryAllocateOpaqueObject(ctx, CondObjectSize, out var handle))
        {
            return posixErrors
                ? PosixOutOfMemory
                : (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        lock (_stateGate)
        {
            var state = new PthreadCondState
            {
                ClockId = attrState.ClockId,
                ProcessShared = attrState.ProcessShared,
            };
            _condStates[condAddress] = state;
            _condStates[handle] = state;
        }

        if (!KernelMemoryCompatExports.TryWriteUInt64Compat(ctx, condAddress, handle))
        {
            lock (_stateGate)
            {
                _condStates.Remove(condAddress);
                _condStates.Remove(handle);
            }
            FreeOpaqueObject(ctx, handle);

            return posixErrors
                ? PosixBadAddress
                : (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    private static int ResolveCondClockId(CpuContext ctx, ulong condAddress)
    {
        var resolvedAddress = ResolveCondHandle(ctx, condAddress);
        lock (_stateGate)
        {
            return _condStates.TryGetValue(resolvedAddress, out var state)
                ? state.ClockId
                : ClockRealtime;
        }
    }

    private static int PthreadCondDestroyCore(CpuContext ctx, ulong condAddress)
    {
        if (condAddress == 0)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        var resolvedAddress = ResolveCondHandle(ctx, condAddress);
        var allocationAddress = resolvedAddress;
        lock (_stateGate)
        {
            if (!_condStates.TryGetValue(resolvedAddress, out var state))
            {
                return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND;
            }

            lock (state.SyncRoot)
            {
                if (state.WaiterQueue.Count != 0)
                {
                    return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_BUSY;
                }
            }

            if (KernelMemoryCompatExports.TryReadUInt64Compat(ctx, condAddress, out var pointedHandle) &&
                pointedHandle != 0 &&
                _condStates.TryGetValue(pointedHandle, out var pointedState) &&
                ReferenceEquals(pointedState, state))
            {
                allocationAddress = pointedHandle;
            }

            _condStates.Remove(resolvedAddress);
            if (resolvedAddress != condAddress)
            {
                _condStates.Remove(condAddress);
            }
            if (allocationAddress != resolvedAddress)
            {
                _condStates.Remove(allocationAddress);
            }
        }

        _ = KernelMemoryCompatExports.TryWriteUInt64Compat(ctx, condAddress, 0);
        FreeOpaqueObject(ctx, allocationAddress);
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    // deadline: a Stopwatch timestamp, 0 for an untimed wait. pthread_* callers get errno
    // values (posixErrors), scePthread* callers sce codes.
    private static int PthreadCondWaitCore(
        CpuContext ctx,
        ulong condAddress,
        ulong mutexAddress,
        long deadline,
        bool posixErrors = false)
    {
        var result = PthreadCondWaitSce(ctx, condAddress, mutexAddress, deadline, posixErrors);
        return posixErrors ? GuestErrno.FromSce(result) : result;
    }

    private static int PthreadCondWaitSce(
        CpuContext ctx,
        ulong condAddress,
        ulong mutexAddress,
        long deadline,
        bool posixErrors)
    {
        var timed = deadline != 0;
        if (condAddress == 0 || mutexAddress == 0)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        if (!TryResolveCondState(ctx, condAddress, createIfZero: true, out _, out var state))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND;
        }

        if (!TryResolveMutexState(ctx, mutexAddress, createIfZero: true, out _, out var mutexState))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND;
        }

        var currentThreadId = KernelPthreadState.GetCurrentThreadHandle();
        var savedRecursion = 1;
        lock (mutexState.SyncRoot)
        {
            if (mutexState.OwnerThreadId == 0 && mutexState.RecursionCount == 0)
            {
                // The guest holds the mutex through a path our host-side tracking
                // never observed — most commonly libkernel's uncontended userspace
                // fast-path, which locks the mutex word directly without an HLE
                // call. Real pthread_cond_wait requires the caller to own the
                // mutex and does not verify it for normal mutexes, so returning
                // EPERM here is wrong: it spins the guest and, worse, leaves the
                // mutex held (the unlock below is skipped), wedging every thread
                // that later blocks on pthread_mutex_lock. Adopt ownership so the
                // unlock/wait/re-lock cycle is balanced and releases the mutex.
                if (mutexState.TryAcquireOwner(currentThreadId))
                {
                    TracePthreadCond("wait-adopt-mutex", condAddress, mutexAddress, state, timed, (int)OrbisGen2Result.ORBIS_GEN2_OK);
                }
            }

            if (mutexState.OwnerThreadId != currentThreadId || mutexState.RecursionCount < 1)
            {
                return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_PERMISSION_DENIED;
            }

            // A recursive mutex is released completely for the wait and gets its depth
            // back with the re-acquisition (libthr's _mutex_cv_unlock/_mutex_cv_lock).
            savedRecursion = mutexState.RecursionCount;
            mutexState.RecursionCount = 1;
        }

        var cooperative = GuestWait.CanBlockCooperatively;
        var waiter = new PthreadCondWaiter
        {
            ThreadId = currentThreadId,
            MutexState = mutexState,
            Cooperative = cooperative,
            PosixErrors = posixErrors,
            SavedRecursion = savedRecursion,
            SchedulerThread = cooperative ? GuestThreadExecution.CurrentGuestThreadHandle : 0,
            HostSignal = cooperative ? null : new ManualResetEventSlim(initialState: false),
            WakeKey = cooperative
                ? $"pthread_cond_waiter:{Interlocked.Increment(ref _nextSynchronizationWaiterId)}"
                : string.Empty,
        };
        if (cooperative)
        {
            waiter.BlockWaiter = new SyncBlockWaiter(
                () => CompleteBlockedCondWait(ctx, condAddress, mutexAddress, state, waiter),
                () => TryGrantCondWaiterMutex(waiter));
        }

        lock (state.SyncRoot)
        {
            waiter.Node = state.WaiterQueue.AddLast(waiter);
            state.Waiters++;
            TracePthreadCond("wait-enter", condAddress, mutexAddress, state, timed, (int)OrbisGen2Result.ORBIS_GEN2_OK);
            if (timed)
            {
                waiter.TimeoutTimer = GuestTimerService.Shared.Schedule(
                    deadline,
                    static callbackState =>
                    {
                        var (condState, condWaiter) = ((PthreadCondState, PthreadCondWaiter))callbackState!;
                        CompleteCondWaiter(condState, condWaiter, timedOut: true);
                    },
                    (state, waiter));
            }
        }

        // The waiter is queued on the condition before the mutex is released, so a signal
        // sent by the next owner of the mutex finds it. The unlock (which can wake and
        // grant other threads) runs with no condition lock held.
        var unlockResult = PthreadMutexUnlockCore(ctx, mutexAddress, requireOwner: true);
        if (unlockResult != (int)OrbisGen2Result.ORBIS_GEN2_OK)
        {
            lock (state.SyncRoot)
            {
                if (waiter.CompletionState == 0)
                {
                    waiter.CompletionState = 3;
                    RemoveCondWaiterLocked(state, waiter);
                }
            }

            _ = GuestTimerService.Shared.Cancel(waiter.TimeoutTimer);
            if (waiter.MutexWaiter is { } staleMutexWaiter)
            {
                _ = AbandonMutexWaiter(mutexState, staleMutexWaiter);
            }

            mutexState.RecursionCount = Math.Max(mutexState.RecursionCount, savedRecursion);
            waiter.HostSignal?.Dispose();
            TracePthreadCond("wait-unlock-fail", condAddress, mutexAddress, state, timed, unlockResult);
            return unlockResult;
        }

        if (cooperative &&
            GuestThreadExecution.RequestCurrentThreadBlock(
                ctx,
                timed ? "pthread_cond_timedwait" : "pthread_cond_wait",
                waiter.WakeKey,
                waiter.BlockWaiter,
                blockDeadlineTimestamp: 0))
        {
            TracePthreadCond("wait-block", condAddress, mutexAddress, state, timed, (int)OrbisGen2Result.ORBIS_GEN2_OK);
            return (int)OrbisGen2Result.ORBIS_GEN2_OK;
        }

        // Non-guest callers have no resumable CPU continuation: wait on the host, with the
        // same FIFO mutex re-acquisition rules as cooperative guest waiters.
        var hostSignal = waiter.HostSignal!;
        GuestWait.RegisterHostEvent(hostSignal);
        try
        {
            while (true)
            {
                var waitMilliseconds = GuestWait.HostWaitSafetySliceMilliseconds;
                lock (state.SyncRoot)
                {
                    if (waiter.CompletionState != 0)
                    {
                        break;
                    }

                    if (timed)
                    {
                        var remaining = deadline - Stopwatch.GetTimestamp();
                        if (remaining <= 0)
                        {
                            _ = CompleteCondWaiterLocked(state, waiter, timedOut: true);
                            break;
                        }

                        waitMilliseconds = (int)Math.Min(
                            waitMilliseconds,
                            (remaining * 1000 + Stopwatch.Frequency - 1) / Stopwatch.Frequency);
                    }

                    // Completion sets the event under this lock, after this Reset.
                    hostSignal.Reset();
                }

                _ = hostSignal.Wait(waitMilliseconds);
                if (Volatile.Read(ref waiter.CompletionStateField) == 0 &&
                    GuestThreadExecution.Scheduler?.HasPendingGuestExceptionForCurrentThread() == true)
                {
                    // End the wait (a spurious wakeup is allowed) so the exception is
                    // delivered while the mutex is re-acquired below.
                    lock (state.SyncRoot)
                    {
                        _ = CompleteCondWaiterLocked(state, waiter, timedOut: false);
                    }
                }
            }
        }
        finally
        {
            GuestWait.UnregisterHostEvent(hostSignal);
        }

        _ = GuestTimerService.Shared.Cancel(waiter.TimeoutTimer);
        hostSignal.Dispose();
        if (waiter.MutexWaiter is null)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        _ = WaitForHostMutexLock(ctx, mutexState, waiter.MutexWaiter);
        mutexState.RecursionCount = waiter.SavedRecursion;
        var waitResult = waiter.CompletionState == 2
            ? (int)OrbisGen2Result.ORBIS_GEN2_ERROR_TIMED_OUT
            : (int)OrbisGen2Result.ORBIS_GEN2_OK;
        TracePthreadCond(waiter.CompletionState == 2 ? "wait-exit-timeout" : "wait-exit", condAddress, mutexAddress, state, timed, waitResult);
        return waitResult;
    }

    private static int PthreadCondSignalCore(CpuContext ctx, ulong condAddress, bool broadcast)
    {
        if (condAddress == 0)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        if (!TryResolveCondState(ctx, condAddress, createIfZero: true, out _, out var state))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND;
        }

        List<PthreadCondWaiter>? completedWaiters = null;
        lock (state.SyncRoot)
        {
            state.SignalEpoch++;
            for (var node = state.WaiterQueue.First; node is not null;)
            {
                var next = node.Next;
                var waiter = node.Value;
                if (waiter.CompletionState == 0 && CompleteCondWaiterLocked(state, waiter, timedOut: false))
                {
                    (completedWaiters ??= new List<PthreadCondWaiter>()).Add(waiter);
                    if (!broadcast)
                    {
                        break;
                    }
                }

                node = next;
            }

            TracePthreadCond(broadcast ? "broadcast" : "signal", condAddress, mutexAddress: 0, state, timed: false, (int)OrbisGen2Result.ORBIS_GEN2_OK);
        }

        if (completedWaiters is not null)
        {
            foreach (var waiter in completedWaiters)
            {
                WakeCooperativeWaiter(waiter);
            }
        }

        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    private static int PthreadCondSignalToCore(CpuContext ctx, ulong condAddress, ulong threadId)
    {
        if (threadId == 0)
        {
            return PthreadCondSignalCore(ctx, condAddress, broadcast: false);
        }

        if (condAddress == 0)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        if (!TryResolveCondState(ctx, condAddress, createIfZero: true, out _, out var state))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND;
        }

        PthreadCondWaiter? completedWaiter = null;
        lock (state.SyncRoot)
        {
            state.SignalEpoch++;
            for (var node = state.WaiterQueue.First; node is not null; node = node.Next)
            {
                var waiter = node.Value;
                if (waiter.ThreadId != threadId ||
                    waiter.CompletionState != 0 ||
                    !CompleteCondWaiterLocked(state, waiter, timedOut: false))
                {
                    continue;
                }

                completedWaiter = waiter;
                break;
            }

            TracePthreadCond("signalto", condAddress, mutexAddress: 0, state, timed: false,
                completedWaiter is null ? 1 : (int)OrbisGen2Result.ORBIS_GEN2_OK);
        }

        if (completedWaiter is null)
        {
            // Matches pthread_cond_signalto_np: a valid target that is not
            // currently waiting on this condition returns EPERM (1), which
            // the Orbis wrapper maps to its kernel error form.
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_PERMISSION_DENIED;
        }

        WakeCooperativeWaiter(completedWaiter);
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    private static PthreadMutexWaiter EnqueueMutexWaiterLocked(
        PthreadMutexState state,
        ulong threadId,
        bool cooperative,
        string? wakeKey = null,
        ulong schedulerThread = 0,
        IGuestThreadBlockWaiter? blockWaiter = null)
    {
        // A guest thread can have at most one pending acquisition on a mutex —
        // it is either running or blocked on exactly one wait. If a waiter for
        // this thread is still queued when it comes back for a fresh
        // acquisition, that entry is a stale leftover the thread abandoned
        // (most often a cond_timedwait timeout whose re-acquire hand-off was
        // lost). Stale entries clog the FIFO head with waiters no thread is
        // blocked on, so the unlock hand-off wakes a dead wake-key and the
        // mutex wedges permanently (observed deadlocking Hades: several
        // re-acquire waiters from one thread piled ahead of a live locker).
        // Prune any prior entry for this thread before enqueueing the new one.
        if (threadId != 0)
        {
            for (var node = state.Waiters.First; node is not null;)
            {
                var next = node.Next;
                if (node.Value.ThreadId == threadId)
                {
                    state.Waiters.Remove(node);
                    state.WaiterRemovedLocked();
                    node.Value.Node = null;
                }

                node = next;
            }
        }

        var waiter = new PthreadMutexWaiter
        {
            ThreadId = threadId,
            Cooperative = cooperative,
            WakeKey = cooperative
                ? wakeKey ?? $"pthread_mutex_waiter:{Interlocked.Increment(ref _nextSynchronizationWaiterId)}"
                : string.Empty,
            HostSignal = cooperative ? null : new ManualResetEventSlim(initialState: false),
            SchedulerThread = schedulerThread,
            BlockWaiter = blockWaiter,
        };
        waiter.Node = state.Waiters.AddLast(waiter);
        state.WaiterAddedLocked();
        ProfileMutexHandoff(state, "Queued", waiter: waiter);
        return waiter;
    }

    [Conditional("DEBUG")]
    private static void RunSynchronizationSelfChecks()
    {
        var mutex = new PthreadMutexState { FifoHandoff = true };
        PthreadMutexWaiter first;
        PthreadMutexWaiter second;
        lock (mutex.SyncRoot)
        {
            first = EnqueueMutexWaiterLocked(mutex, 0x101, cooperative: false);
            second = EnqueueMutexWaiterLocked(mutex, 0x202, cooperative: false);
            Debug.Assert(!TryGrantMutexWaiterLocked(mutex, second), "A mutex waiter bypassed FIFO order.");
            Debug.Assert(TryGrantMutexWaiterLocked(mutex, first), "The FIFO mutex head was not granted.");
            Debug.Assert(mutex.OwnerThreadId == first.ThreadId && mutex.RecursionCount == 1, "Mutex ownership was not transferred atomically.");
            mutex.OwnerThreadId = 0;
            mutex.RecursionCount = 0;
            Debug.Assert(TryGrantMutexWaiterLocked(mutex, second), "The second mutex waiter was not granted after release.");
        }

        var wakeMutex = new PthreadMutexState { FifoHandoff = false };
        lock (wakeMutex.SyncRoot)
        {
            first = EnqueueMutexWaiterLocked(wakeMutex, 0x101, cooperative: false);
            second = EnqueueMutexWaiterLocked(wakeMutex, 0x202, cooperative: false);
            Debug.Assert(SignalMutexHeadLocked(wakeMutex, out _) == first && Volatile.Read(ref first.Woken) == 1, "Release-and-wake did not wake the queue head.");
            Debug.Assert(SignalMutexHeadLocked(wakeMutex, out _) is null, "An already-woken head was woken again.");
            Debug.Assert(TryGrantMutexWaiterLocked(wakeMutex, second), "A running queued waiter could not take a free mutex.");
            Debug.Assert(wakeMutex.OwnerThreadId == second.ThreadId && ReferenceEquals(wakeMutex.Waiters.First, first.Node), "The woken head lost its queue position.");
            Debug.Assert(!TryGrantMutexWaiterLocked(wakeMutex, first), "A woken head took an owned mutex.");
        }

        var cond = new PthreadCondState();
        var condMutex = new PthreadMutexState();
        var condWaiter = new PthreadCondWaiter
        {
            ThreadId = 0x303,
            MutexState = condMutex,
            WakeKey = string.Empty,
            Cooperative = false,
        };
        lock (cond.SyncRoot)
        {
            condWaiter.Node = cond.WaiterQueue.AddLast(condWaiter);
            cond.Waiters++;
            Debug.Assert(CompleteCondWaiterLocked(cond, condWaiter, timedOut: false), "A condition waiter was not completed.");
            Debug.Assert(cond.WaiterQueue.Count == 0 && cond.Waiters == 0 && condWaiter.MutexWaiter is not null, "Condition completion did not atomically queue mutex reacquisition.");
        }
    }

    private static bool TryGrantMutexWaiterLocked(PthreadMutexState state, PthreadMutexWaiter waiter)
    {
        if (Volatile.Read(ref waiter.Granted) != 0)
        {
            return true;
        }

        // FIFO handoff grants only the head. Under release-and-wake any queued waiter that is
        // running (spinning, woken, or re-checking before it parks) may take a free mutex, like
        // any other barging locker; queue order only decides whom an unlock wakes.
        if (state.OwnerThreadId != 0 ||
            waiter.Node is null ||
            (state.FifoHandoff && !ReferenceEquals(state.Waiters.First, waiter.Node)))
        {
            return false;
        }

        if (!state.TryAcquireOwner(waiter.ThreadId))
        {
            return false;
        }

        state.Waiters.Remove(waiter.Node);
        state.WaiterRemovedLocked();
        waiter.Node = null;
        Volatile.Write(ref waiter.Granted, 1);
        ProfileMutexHandoff(state, "Granted", waiter: waiter);
        return true;
    }

    private static void WakeFirstMutexWaiter(PthreadMutexState state, ulong dsMutexAddress = 0, ulong dsFrom = 0)
    {
        if (!state.FifoHandoff)
        {
            PthreadMutexWaiter? head;
            bool cooperative;
            lock (state.SyncRoot)
            {
                head = SignalMutexHeadLocked(state, out cooperative);
            }

            if (head is not null)
            {
                FinishMutexHeadWake(state, head, cooperative, dsMutexAddress, dsFrom);
            }

            return;
        }

        PthreadMutexWaiter? nextWaiter;
        lock (state.SyncRoot)
        {
            if (state.OwnerThreadId != 0)
            {
                return;
            }

            nextWaiter = state.Waiters.First?.Value;
            if (nextWaiter is { Cooperative: false })
            {
                nextWaiter.HostSignal!.Set();
            }
        }

        // A spinning head takes the mutex by itself. If it stops spinning and parks after this
        // check, the scheduler's check right after the block grants the mutex to it.
        if (nextWaiter is { Cooperative: true } && Volatile.Read(ref nextWaiter.Spinning) == 0)
        {
            WakeProfiledMutexWaiter(state, nextWaiter);
            if (DsTrace.Sync && DsTrace.Armed)
            {
                DsTrace.Emit(DsEv.MXH, dsMutexAddress, dsFrom, nextWaiter.ThreadId, 1UL, (ulong)state.QueuedWaiterCount);
            }
        }
    }

    // SHARPEMU_MUTEX_STARVATION_HANDOFF=0 restores plain release-and-wake for starving waiters.
    internal static bool MutexStarvationHandoff = !string.Equals(
        Environment.GetEnvironmentVariable("SHARPEMU_MUTEX_STARVATION_HANDOFF"), "0", StringComparison.Ordinal);

    // Release-and-wake lets a running locker barge past a woken waiter, which keeps a free
    // mutex with a queued waiter from wedging (see the FIFO note in the unlock). A thread that
    // relocks in a tight loop can then win every race against a waiter that needs a host wake-up
    // to run: Silent Hill's Wwise audio thread polls its global lock hundreds of thousands of
    // times a second, and the game thread waited up to 2 s for that lock, 21 ms on average.
    // Once a waiter has lost such a race the unlock passes ownership to it directly, as the FIFO
    // handoff does, but without the owner ever reading 0, so no barging CAS can take it on the
    // way. Caller holds SyncRoot, owns the mutex and has just dropped RecursionCount to 0.
    // A waiter only starves while it runs: a host waiter is signalled here, and a cooperative
    // one is re-contending after its resume (ReacquireWokenMutex), never parked, so it sees
    // Granted itself and needs no scheduler wake.
    private static bool TryHandOffToStarvingHeadLocked(PthreadMutexState state)
    {
        if (!MutexStarvationHandoff ||
            state.FifoHandoff ||
            state.Waiters.First?.Value is not { } head ||
            Volatile.Read(ref head.Starving) == 0 ||
            Volatile.Read(ref head.Granted) != 0)
        {
            return false;
        }

        state.Waiters.Remove(head.Node!);
        state.WaiterRemovedLocked();
        head.Node = null;
        state.OwnerThreadId = head.ThreadId;
        state.RecursionCount = 1;
        Volatile.Write(ref head.Granted, 1);
        if (!head.Cooperative)
        {
            head.HostSignal!.Set();
        }

        ProfileMutexHandoff(state, "Granted", waiter: head);
        return true;
    }

    // Release-and-wake, caller holds SyncRoot right after releasing the mutex. Picks the queue
    // head to wake, or null when there is nothing to do: the mutex was already re-taken (that
    // owner's unlock wakes the head), the queue is empty, or the head is already woken and will
    // re-check before it sleeps. A host waiter is signalled here, under SyncRoot, so the signal
    // orders against its check-then-Reset; a cooperative head is woken by FinishMutexHeadWake
    // outside the lock. Its park recheck runs TryGrantBlockedMutexLock under SyncRoot after it
    // stops spinning, so a wake that finds it not yet parked is not lost.
    private static PthreadMutexWaiter? SignalMutexHeadLocked(PthreadMutexState state, out bool cooperative)
    {
        cooperative = false;
        if (state.OwnerThreadId != 0 ||
            state.Waiters.First?.Value is not { } head ||
            Volatile.Read(ref head.Woken) != 0)
        {
            return null;
        }

        cooperative = head.Cooperative;
        if (!cooperative)
        {
            Volatile.Write(ref head.Woken, 1);
            head.HostSignal!.Set();
        }

        ProfileMutexHandoff(state, "Woken", waiter: head);
        return head;
    }

    private static void FinishMutexHeadWake(
        PthreadMutexState state,
        PthreadMutexWaiter head,
        bool cooperative,
        ulong dsMutexAddress,
        ulong dsFrom)
    {
        // A spinning head sees the free mutex itself.
        var parked = cooperative && Volatile.Read(ref head.Spinning) == 0;
        if (parked)
        {
            WakeProfiledMutexWaiter(state, head);
        }

        if (DsTrace.Sync && DsTrace.Armed)
        {
            DsTrace.Emit(DsEv.MXH, dsMutexAddress, dsFrom, head.ThreadId, parked ? 1UL : 0UL, (ulong)state.QueuedWaiterCount);
        }
    }

    private static void RecordMutexContention(ulong mutexAddress, PthreadMutexState state, PthreadMutexWaiter waiter)
    {
        MutexContentionProfile.Record(mutexAddress, waiter.ProfileOwner, waiter.ProfileStart);
        state.ProfileOwner = MutexContentionProfile.CurrentThreadName();
    }

    private static readonly long MutexWaiterSpinTicks = Stopwatch.Frequency * 40 / 1_000_000;

    // A queued waiter spins briefly for its grant before it parks. Under FIFO handoff the unlock
    // hands the mutex to the queue head, and parking and waking a guest thread costs far more
    // than most critical sections, so a head that parks holds every later locker behind its
    // wake-up (a lock convoy); only the head can be granted. Under release-and-wake any spinning
    // waiter simply takes the mutex when it sees it free.
    private static bool SpinForMutexGrant(PthreadMutexState state, PthreadMutexWaiter waiter)
    {
        Volatile.Write(ref waiter.Spinning, 1);
        try
        {
            var deadline = Stopwatch.GetTimestamp() + MutexWaiterSpinTicks;
            for (var round = 0; ; round++)
            {
                if (Volatile.Read(ref waiter.Granted) != 0)
                {
                    return true;
                }

                if (state.OwnerThreadId == 0)
                {
                    lock (state.SyncRoot)
                    {
                        if (TryGrantMutexWaiterLocked(state, waiter))
                        {
                            return true;
                        }
                    }
                }

                if (Stopwatch.GetTimestamp() >= deadline)
                {
                    return false;
                }

                Thread.SpinWait(Math.Min(4 << Math.Min(round, 4), 64));
            }
        }
        finally
        {
            Volatile.Write(ref waiter.Spinning, 0);
        }
    }

    private static void WakeProfiledMutexWaiter(PthreadMutexState state, PthreadMutexWaiter waiter)
    {
        ProfileMutexHandoff(state, "WakeStarted", waiter: waiter);
        var wakeCount = WakeParkedThread(waiter.SchedulerThread, Volatile.Read(ref waiter.BlockWaiter), waiter.WakeKey);
        ProfileMutexHandoff(state, "WakeFinished", waiter: waiter, result: wakeCount);
    }

    // Readies the one guest thread parked on blockWaiter. Before the thread has published its
    // block waiter the key-based wake is used; either way a thread that parks after this
    // re-checks TryWake itself, so the wake is not lost.
    private static int WakeParkedThread(ulong schedulerThread, IGuestThreadBlockWaiter? blockWaiter, string wakeKey)
    {
        var scheduler = GuestThreadExecution.Scheduler;
        if (scheduler is null)
        {
            return 0;
        }

        if (blockWaiter is not null && schedulerThread != 0)
        {
            return scheduler.WakeBlockedWaiter(schedulerThread, blockWaiter, wakeKey) ? 1 : 0;
        }

        return scheduler.WakeBlockedThreads(wakeKey, 1);
    }

    private static bool RequestMutexBlock(
        CpuContext ctx,
        PthreadMutexWaiter waiter,
        long deadline,
        IGuestThreadBlockWaiter blockWaiter)
    {
        waiter.SchedulerThread = GuestThreadExecution.CurrentGuestThreadHandle;
        Volatile.Write(ref waiter.BlockWaiter, blockWaiter);
        return GuestThreadExecution.RequestCurrentThreadBlock(
            ctx,
            "pthread_mutex_lock",
            waiter.WakeKey,
            blockWaiter,
            deadline);
    }

    // Takes a waiter that gave up (timed lock expired) out of the queue. Returns false when
    // it was granted the mutex meanwhile, in which case the caller owns it.
    private static bool AbandonMutexWaiter(PthreadMutexState state, PthreadMutexWaiter waiter)
    {
        lock (state.SyncRoot)
        {
            if (TryGrantMutexWaiterLocked(state, waiter))
            {
                return false;
            }

            if (waiter.Node is not null)
            {
                state.Waiters.Remove(waiter.Node);
                state.WaiterRemovedLocked();
                waiter.Node = null;
            }
        }

        // The departed waiter may have absorbed a wake meant for the head (release-and-wake
        // sets Woken on it): pass it on so a free mutex does not sit behind a parked queue.
        if (state.QueuedWaiterCount != 0)
        {
            WakeFirstMutexWaiter(state);
        }

        return true;
    }

    private static void ProfileMutexHandoff(PthreadMutexState state, string stage, ulong address = 0,
        PthreadMutexWaiter? waiter = null, int result = 0)
    {
        if (!MutexHandoffProfile.Enabled) return;
        MutexHandoffProfile.Record(state.ProfileIdentity, stage, address, state.OwnerThreadId,
            waiter?.ThreadId ?? 0, waiter?.WakeKey ?? "none", state.QueuedWaiterCount, result);
    }

    private static int WaitForHostMutexLock(
        CpuContext ctx,
        PthreadMutexState state,
        PthreadMutexWaiter waiter,
        bool deliverGuestExceptions = true,
        long deadline = 0)
    {
        ManualResetEventSlim? hostSignal = null;
        var registered = false;
        var woke = false;
        try
        {
            while (true)
            {
                if (woke && deliverGuestExceptions)
                {
                    // Woken by the owner, by a queued guest exception (GuestWait.InterruptHostWaiters)
                    // or by the safety slice: deliver a pending exception outside the mutex lock,
                    // then re-check.
                    GuestThreadExecution.Scheduler?.DeliverPendingGuestExceptionIfReady(ctx);
                }

                lock (state.SyncRoot)
                {
                    if (waiter.HostSignal is null)
                    {
                        waiter.Cooperative = false;
                        waiter.HostSignal = new ManualResetEventSlim(initialState: false);
                    }

                    hostSignal = waiter.HostSignal;
                    if (TryGrantMutexWaiterLocked(state, waiter))
                    {
                        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
                    }

                    if (woke)
                    {
                        // Woken (or out of a safety slice) and the mutex is owned again.
                        Volatile.Write(ref waiter.Starving, 1);
                    }

                    // Still owned: from here on the owner's unlock must signal this waiter.
                    Volatile.Write(ref waiter.Woken, 0);
                    hostSignal.Reset();
                }

                var waitMilliseconds = GuestWait.HostWaitSafetySliceMilliseconds;
                if (deadline != 0)
                {
                    var remaining = deadline - Stopwatch.GetTimestamp();
                    if (remaining <= 0)
                    {
                        return AbandonMutexWaiter(state, waiter)
                            ? (int)OrbisGen2Result.ORBIS_GEN2_ERROR_TIMED_OUT
                            : (int)OrbisGen2Result.ORBIS_GEN2_OK;
                    }

                    waitMilliseconds = (int)Math.Min(
                        waitMilliseconds,
                        (remaining * 1000 + Stopwatch.Frequency - 1) / Stopwatch.Frequency);
                }

                if (!registered)
                {
                    GuestWait.RegisterHostEvent(hostSignal);
                    registered = true;
                }

                _ = hostSignal.Wait(waitMilliseconds);
                woke = true;
            }
        }
        finally
        {
            if (registered && hostSignal is not null)
            {
                GuestWait.UnregisterHostEvent(hostSignal);
            }

            hostSignal?.Dispose();
        }
    }

    private static bool TryGrantBlockedMutexLock(
        CpuContext ctx,
        ulong mutexAddress,
        ulong resolvedAddress,
        PthreadMutexState state,
        PthreadMutexWaiter waiter)
    {
        var granted = false;
        lock (state.SyncRoot)
        {
            if (state.FifoHandoff)
            {
                granted = TryGrantMutexWaiterLocked(state, waiter);
            }
            else if (Volatile.Read(ref waiter.Granted) != 0)
            {
                granted = true;
            }
            else if (state.OwnerThreadId == 0 && waiter.Node is not null)
            {
                // Release-and-wake: make the thread runnable without taking ownership for it;
                // it re-contends in CompleteBlockedMutexLock. Woken tells later unlocks that it
                // will re-check, so they need not wake it again. If the mutex is owned, stay
                // parked: that owner's unlock wakes this head.
                Volatile.Write(ref waiter.Woken, 1);
                granted = true;
            }
        }

        TracePthreadMutex(
            ctx,
            granted ? "lock-reserve" : "lock-reserve-busy",
            mutexAddress,
            resolvedAddress,
            state,
            waiter.ThreadId,
            granted ? (int)OrbisGen2Result.ORBIS_GEN2_OK : (int)OrbisGen2Result.ORBIS_GEN2_ERROR_BUSY);
        return granted;
    }

    private static int CompleteBlockedMutexLock(
        CpuContext ctx,
        ulong mutexAddress,
        ulong resolvedAddress,
        PthreadMutexState state,
        PthreadMutexWaiter waiter,
        bool posix = false,
        long deadline = 0)
    {
        var currentThreadId = KernelPthreadState.GetCurrentThreadHandle();
        var result = (int)OrbisGen2Result.ORBIS_GEN2_OK;
        if (Volatile.Read(ref waiter.Granted) == 0)
        {
            if (deadline != 0 && Stopwatch.GetTimestamp() >= deadline)
            {
                // Resumed by the scheduler's deadline: give up unless the grant raced in.
                if (AbandonMutexWaiter(state, waiter))
                {
                    result = (int)OrbisGen2Result.ORBIS_GEN2_ERROR_TIMED_OUT;
                }
            }
            else if (!state.FifoHandoff)
            {
                result = ReacquireWokenMutex(ctx, state, waiter, deadline);
            }
        }

        if (waiter.DsEnqueueTs != 0 && DsTrace.Sync && DsTrace.Armed && Volatile.Read(ref waiter.Granted) == 1)
        {
            DsTrace.Emit(
                DsEv.MXR,
                waiter.DsMutexAddress,
                currentThreadId,
                (ulong)DsTrace.TicksToNs(Stopwatch.GetTimestamp() - waiter.DsEnqueueTs));
        }

        if (Volatile.Read(ref waiter.Granted) == 1)
        {
            if (MutexContentionProfile.Enabled) RecordMutexContention(mutexAddress, state, waiter);
            TracePthreadMutex(ctx, "lock-resume", mutexAddress, resolvedAddress, state, currentThreadId, (int)OrbisGen2Result.ORBIS_GEN2_OK);
            return (int)OrbisGen2Result.ORBIS_GEN2_OK;
        }

        if (result == (int)OrbisGen2Result.ORBIS_GEN2_OK)
        {
            result = (int)OrbisGen2Result.ORBIS_GEN2_ERROR_BUSY;
        }

        TracePthreadMutex(ctx, "lock-resume-ungranted", mutexAddress, resolvedAddress, state, currentThreadId, result);
        return posix ? GuestErrno.FromSce(result) : result;
    }

    // Release-and-wake: a woken cooperative waiter re-contends when it resumes. It is still
    // queued at its old position and Woken is still 1, so unlocks meanwhile leave it alone. If a
    // barger owns the mutex it spins, then sleeps on its host signal as a host waiter: the
    // scheduler cannot re-park a thread from Resume. Guest exceptions are not delivered from
    // this wait, because no import frame of this thread is current here.
    private static int ReacquireWokenMutex(CpuContext ctx, PthreadMutexState state, PthreadMutexWaiter waiter, long deadline = 0)
    {
        lock (state.SyncRoot)
        {
            if (TryGrantMutexWaiterLocked(state, waiter))
            {
                return (int)OrbisGen2Result.ORBIS_GEN2_OK;
            }

            // Woken, and a barging locker took the mutex first.
            Volatile.Write(ref waiter.Starving, 1);
        }

        if (SpinForMutexGrant(state, waiter))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_OK;
        }

        return WaitForHostMutexLock(ctx, state, waiter, deliverGuestExceptions: false, deadline: deadline);
    }

    private static bool CompleteCondWaiterLocked(
        PthreadCondState state,
        PthreadCondWaiter waiter,
        bool timedOut)
    {
        if (waiter.CompletionState != 0)
        {
            return false;
        }

        waiter.CompletionState = timedOut ? 2 : 1;
        RemoveCondWaiterLocked(state, waiter);
        if (!timedOut)
        {
            _ = GuestTimerService.Shared.Cancel(waiter.TimeoutTimer);
        }

        lock (waiter.MutexState.SyncRoot)
        {
            // The re-acquisition wakes the thread through the condition wait's block waiter.
            waiter.MutexWaiter = EnqueueMutexWaiterLocked(
                waiter.MutexState,
                waiter.ThreadId,
                waiter.Cooperative,
                waiter.WakeKey,
                waiter.SchedulerThread,
                waiter.BlockWaiter);
        }

        waiter.HostSignal?.Set();
        return true;
    }

    private static void CompleteCondWaiter(
        PthreadCondState state,
        PthreadCondWaiter waiter,
        bool timedOut)
    {
        var completed = false;
        lock (state.SyncRoot)
        {
            completed = CompleteCondWaiterLocked(state, waiter, timedOut);
        }

        if (completed)
        {
            WakeCooperativeWaiter(waiter);
        }
    }

    private static void RemoveCondWaiterLocked(PthreadCondState state, PthreadCondWaiter waiter)
    {
        if (waiter.Node is not null)
        {
            state.WaiterQueue.Remove(waiter.Node);
            waiter.Node = null;
            state.Waiters = Math.Max(0, state.Waiters - 1);
        }
    }

    private static bool TryGrantCondWaiterMutex(PthreadCondWaiter waiter)
    {
        var mutexWaiter = waiter.MutexWaiter;
        if (waiter.CompletionState == 0 || mutexWaiter is null)
        {
            return false;
        }

        lock (waiter.MutexState.SyncRoot)
        {
            return TryGrantMutexWaiterLocked(waiter.MutexState, mutexWaiter);
        }
    }

    private static int CompleteBlockedCondWait(
        CpuContext ctx,
        ulong condAddress,
        ulong mutexAddress,
        PthreadCondState state,
        PthreadCondWaiter waiter)
    {
        _ = GuestTimerService.Shared.Cancel(waiter.TimeoutTimer);
        var granted = waiter.MutexWaiter is not null &&
            Volatile.Read(ref waiter.MutexWaiter.Granted) == 1;
        if (granted)
        {
            waiter.MutexState.RecursionCount = waiter.SavedRecursion;
        }

        var result = granted
            ? (waiter.CompletionState == 2
                ? (int)OrbisGen2Result.ORBIS_GEN2_ERROR_TIMED_OUT
                : (int)OrbisGen2Result.ORBIS_GEN2_OK)
            : (int)OrbisGen2Result.ORBIS_GEN2_ERROR_BUSY;
        TracePthreadCond(
            waiter.CompletionState == 2 ? "wait-resume-timeout" : "wait-resume",
            condAddress,
            mutexAddress,
            state,
            waiter.CompletionState == 2,
            result);
        _ = ctx;
        return waiter.PosixErrors ? GuestErrno.FromSce(result) : result;
    }

    private static void WakeCooperativeWaiter(PthreadCondWaiter waiter)
    {
        if (waiter.Cooperative)
        {
            _ = WakeParkedThread(waiter.SchedulerThread, waiter.BlockWaiter, waiter.WakeKey);
        }
    }

    private static int NormalizeMutexType(int type)
    {
        return type switch
        {
            0 => MutexTypeErrorCheck,
            1 => MutexTypeErrorCheck,
            2 => MutexTypeRecursive,
            3 => MutexTypeNormal,
            4 => MutexTypeAdaptiveNp,
            _ => MutexTypeErrorCheck,
        };
    }

    private static object GetPthreadOnceGate(ulong onceAddress)
    {
        lock (_stateGate)
        {
            if (!_onceGates.TryGetValue(onceAddress, out var gate))
            {
                gate = new object();
                _onceGates[onceAddress] = gate;
            }

            return gate;
        }
    }

    private static int SetReturn(CpuContext ctx, OrbisGen2Result result)
    {
        ctx[CpuRegister.Rax] = unchecked((ulong)(int)result);
        return (int)result;
    }

    private static bool TryReadInt32(CpuContext ctx, ulong address, out int value)
    {
        Span<byte> bytes = stackalloc byte[sizeof(int)];
        if (!ctx.Memory.TryRead(address, bytes))
        {
            value = 0;
            return false;
        }

        value = BinaryPrimitives.ReadInt32LittleEndian(bytes);
        return true;
    }

    private static bool TryWriteInt32(CpuContext ctx, ulong address, int value)
    {
        Span<byte> bytes = stackalloc byte[sizeof(int)];
        BinaryPrimitives.WriteInt32LittleEndian(bytes, value);
        return ctx.Memory.TryWrite(address, bytes);
    }

    private static bool CreateImplicitMutexState(CpuContext ctx, ulong mutexAddress, int type, out ulong resolvedAddress, [NotNullWhen(true)] out PthreadMutexState? state)
    {
        PthreadMutexState? createdState = new PthreadMutexState
        {
            Type = type,
        };

        if (!TryAllocateOpaqueObject(ctx, MutexObjectSize, out var handle))
        {
            resolvedAddress = 0;
            state = null;
            return false;
        }
        if (!InitializeMutexObject(ctx, handle, createdState))
        {
            FreeOpaqueObject(ctx, handle);
            resolvedAddress = 0;
            state = null;
            return false;
        }

        lock (_stateGate)
        {
            if (_mutexStates.TryGetValue(mutexAddress, out state))
            {
                FreeOpaqueObject(ctx, handle);
                resolvedAddress = mutexAddress;
                return true;
            }

            // The handle was just allocated, so a state still cached under it belongs to
            // an object whose memory was freed and handed out again: it is overwritten.
            _mutexStates[mutexAddress] = createdState;
            _mutexStates[handle] = createdState;
            if (!KernelMemoryCompatExports.TryWriteUInt64Compat(ctx, mutexAddress, handle))
            {
                _mutexStates.TryRemove(mutexAddress, out _);
                _mutexStates.TryRemove(handle, out _);
                createdState = null;
            }
        }

        if (createdState is null)
        {
            FreeOpaqueObject(ctx, handle);
            resolvedAddress = 0;
            state = null;
            return false;
        }

        resolvedAddress = handle;
        state = createdState;
        return true;
    }

    private static void TracePthreadSelf(CpuContext ctx, ulong currentThreadHandle)
    {
        if (!ShouldTracePthread())
        {
            return;
        }

        var currentThreadId = KernelPthreadState.GetCurrentThreadUniqueId();
        Console.Error.WriteLine(
            $"[LOADER][TRACE] pthread_self: stale_rdi=0x{ctx[CpuRegister.Rdi]:X16} thread=0x{currentThreadHandle:X16} tid=0x{currentThreadId:X16}");
    }

    private static void TracePthreadOnce(ulong onceAddress, ulong initRoutine, string operation, string? error)
    {
        if (!ShouldTracePthread())
        {
            return;
        }

        var suffix = string.IsNullOrWhiteSpace(error) ? string.Empty : $" error={error}";
        Console.Error.WriteLine(
            $"[LOADER][TRACE] pthread_once_{operation}: once=0x{onceAddress:X16} init=0x{initRoutine:X16}{suffix}");
    }

    private static void TracePthreadMutex(CpuContext ctx, string operation, ulong mutexAddress, ulong resolvedAddress, PthreadMutexState? state, ulong currentThreadId, int result)
    {
        if (MutexHandoffProfile.Enabled)
            MutexHandoffProfile.Record(state?.ProfileIdentity ?? 0, operation, mutexAddress,
                state?.OwnerThreadId ?? 0, currentThreadId, waiting: state?.QueuedWaiterCount ?? 0, result: result);
        if (!ShouldTracePthreadMutex(mutexAddress, resolvedAddress))
        {
            return;
        }

        _ = KernelMemoryCompatExports.TryReadUInt64Compat(ctx, mutexAddress, out var guestWord0);
        _ = KernelMemoryCompatExports.TryReadUInt64Compat(ctx, mutexAddress + 8, out var guestWord1);
        Console.Error.WriteLine(
            $"[LOADER][TRACE] pthread_{operation}: mutex=0x{mutexAddress:X16} resolved=0x{resolvedAddress:X16} " +
            $"guest[0]=0x{guestWord0:X16} guest[8]=0x{guestWord1:X16} " +
            $"current=0x{currentThreadId:X16} owner=0x{(state?.OwnerThreadId ?? 0):X16} " +
            $"recursion={(state?.RecursionCount ?? 0)} type={(state?.Type ?? 0)} result=0x{unchecked((uint)result):X8}");
    }

    private static void TracePthreadFastPathUnlock(
        CpuContext ctx,
        ulong mutexAddress,
        ulong resolvedAddress,
        PthreadMutexState state,
        ulong currentThreadId)
    {
        if (!_tracePthreadFastPath || Interlocked.Increment(ref _pthreadFastPathTraceWritten) > 16)
        {
            return;
        }

        Span<byte> objectBytes = stackalloc byte[0x50];
        if (!ctx.Memory.TryRead(resolvedAddress, objectBytes))
        {
            Console.Error.WriteLine(
                $"[LOADER][TRACE] pthread_fastpath_unlock: mutex=0x{mutexAddress:X16} resolved=0x{resolvedAddress:X16} read=failed");
            return;
        }

        Span<ulong> words = stackalloc ulong[10];
        for (var index = 0; index < words.Length; index++)
        {
            words[index] = BinaryPrimitives.ReadUInt64LittleEndian(objectBytes.Slice(index * sizeof(ulong), sizeof(ulong)));
        }

        Console.Error.WriteLine(
            $"[LOADER][TRACE] pthread_fastpath_unlock: mutex=0x{mutexAddress:X16} resolved=0x{resolvedAddress:X16} " +
            $"current=0x{currentThreadId:X16} owner=0x{state.OwnerThreadId:X16} recursion={state.RecursionCount} " +
            $"q00=0x{words[0]:X16} q08=0x{words[1]:X16} q10=0x{words[2]:X16} q18=0x{words[3]:X16} " +
            $"q20=0x{words[4]:X16} q28=0x{words[5]:X16} q30=0x{words[6]:X16} q38=0x{words[7]:X16} " +
            $"q40=0x{words[8]:X16} q48=0x{words[9]:X16}");
    }

    private static void TracePthreadFastPathBusy(
        string operation,
        ulong mutexAddress,
        ulong resolvedAddress,
        PthreadMutexState? state,
        ulong currentThreadId,
        int result)
    {
        if (!_tracePthreadFastPath || !_pthreadFastPathBusyTraced.TryAdd(mutexAddress, 0))
        {
            return;
        }

        Console.Error.WriteLine(
            $"[LOADER][TRACE] pthread_fastpath_{operation}: mutex=0x{mutexAddress:X16} resolved=0x{resolvedAddress:X16} " +
            $"current=0x{currentThreadId:X16} owner=0x{(state?.OwnerThreadId ?? 0):X16} " +
            $"recursion={(state?.RecursionCount ?? 0)} type={(state?.Type ?? 0)} " +
            $"waiters={(state?.QueuedWaiterCount ?? 0)} result=0x{unchecked((uint)result):X8}");
    }

    private static void TracePthreadCond(string operation, ulong condAddress, ulong mutexAddress, PthreadCondState? state, bool timed, int result)
    {
        if (!_tracePthreadConds)
        {
            return;
        }

        Console.Error.WriteLine(
            $"[LOADER][TRACE] pthread_cond_{operation}: cond=0x{condAddress:X16} mutex=0x{mutexAddress:X16} " +
            $"waiters={(state?.Waiters ?? 0)} epoch=0x{(state?.SignalEpoch ?? 0):X} timed={timed} result=0x{unchecked((uint)result):X8}");
    }

    private static bool ShouldTracePthread()
    {
        return _tracePthreads;
    }

    private static bool ShouldTracePthreadMutex(ulong mutexAddress, ulong resolvedAddress)
    {
        if (_tracePthreadMutexFilter is null || _tracePthreadMutexFilter.Count == 0)
        {
            return _tracePthreads;
        }

        return _tracePthreadMutexFilter.Contains(mutexAddress) ||
            _tracePthreadMutexFilter.Contains(resolvedAddress);
    }

    private static HashSet<ulong>? ParseTraceAddressFilter(string? filter)
    {
        if (string.IsNullOrWhiteSpace(filter))
        {
            return null;
        }

        var addresses = new HashSet<ulong>();
        foreach (var token in filter.Split(new[] { ',', ';', ' ' }, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            var normalized = token.StartsWith("0x", StringComparison.OrdinalIgnoreCase)
                ? token[2..]
                : token;
            normalized = normalized.TrimStart('0');

            if (ulong.TryParse(
                    normalized.Length == 0 ? "0" : normalized,
                    System.Globalization.NumberStyles.HexNumber,
                    System.Globalization.CultureInfo.InvariantCulture,
                    out var address))
            {
                addresses.Add(address);
            }
        }

        return addresses.Count == 0 ? null : addresses;
    }
}
