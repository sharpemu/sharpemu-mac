// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using SharpEmu.HLE;
using SharpEmu.Libs.Diagnostics;
using SharpEmu.Libs.Threading;

namespace SharpEmu.Libs.Kernel;

public static class KernelSemaphoreCompatExports
{
    private const int MaxSemaphoreNameLength = 128;
    private const uint AttrPriority = 2;
    private static readonly ConcurrentDictionary<uint, KernelSemaphoreState> _semaphores = new();
    private static int _nextSemaphoreHandle = 1;

    private sealed class KernelSemaphoreState
    {
        public KernelSemaphoreState(bool priorityOrder)
        {
            Queue = new WaitQueue(priorityOrder ? WaitQueueOrder.Priority : WaitQueueOrder.Fifo);
            Queue.Reevaluate = static (queue, ref batch) => ((KernelSemaphoreState)queue.Owner!).GrantWaitersLocked(ref batch);
            Queue.Owner = this;
        }

        public required string Name { get; init; }
        public required int InitialCount { get; init; }
        public required int MaxCount { get; init; }
        public required bool PriorityOrder { get; init; }

        // Guarded by Queue.Gate.
        public int Count { get; set; }

        public WaitQueue Queue { get; }

        public int WaitingThreads => Queue.Count;

        // Grants tokens in queue order. The head blocks the waiters behind it, so a large
        // request is not starved by a stream of small ones (FIFO/priority, never barging).
        public int GrantWaitersLocked(ref WakeBatch batch)
        {
            var granted = 0;
            while (Queue.First is SemaphoreWaiter head && Count >= head.Need)
            {
                Count -= head.Need;
                if (head.TryCompleteLocked(GuestWaitStatus.Satisfied))
                {
                    batch.Add(head);
                    granted++;
                }
            }

            return granted;
        }
    }

    private sealed class SemaphoreWaiter : GuestWaiter
    {
        public SemaphoreWaiter(int need, int priority)
            : base(priority)
        {
            Need = need;
        }

        public int Need { get; }
    }

    [SysAbiExport(
        Nid = "188x57JYp0g",
        ExportName = "sceKernelCreateSema",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelCreateSema(CpuContext ctx)
    {
        var semaphoreAddress = ctx[CpuRegister.Rdi];
        var nameAddress = ctx[CpuRegister.Rsi];
        var attr = unchecked((uint)ctx[CpuRegister.Rdx]);
        var initialCount = unchecked((int)ctx[CpuRegister.Rcx]);
        var maxCount = unchecked((int)ctx[CpuRegister.R8]);
        var optionAddress = ctx[CpuRegister.R9];

        if (semaphoreAddress == 0 ||
            nameAddress == 0 ||
            attr > 2 ||
            initialCount < 0 ||
            maxCount <= 0 ||
            initialCount > maxCount ||
            optionAddress != 0)
        {
            return SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT);
        }

        if (!KernelMemoryCompatExports.TryReadNullTerminatedUtf8(
                ctx, nameAddress, MaxSemaphoreNameLength, out var name))
        {
            return SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
        }

        var handle = AllocateHandle();
        _semaphores[handle] = new KernelSemaphoreState(priorityOrder: attr == AttrPriority)
        {
            Name = name,
            InitialCount = initialCount,
            MaxCount = maxCount,
            PriorityOrder = attr == AttrPriority,
            Count = initialCount,
        };

        if (!ctx.TryWriteUInt64(semaphoreAddress, handle))
        {
            _semaphores.TryRemove(handle, out _);
            return SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
        }

        if (_traceSema)
        {
            TraceSemaphore($"create handle=0x{handle:X8} name='{name}' attr=0x{attr:X} init={initialCount} max={maxCount}");
        }
        return SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_OK);
    }

    [SysAbiExport(
        Nid = "Zxa0VhQVTsk",
        ExportName = "sceKernelWaitSema",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelWaitSema(CpuContext ctx)
    {
        var handle = unchecked((uint)ctx[CpuRegister.Rdi]);
        var needCount = unchecked((int)ctx[CpuRegister.Rsi]);
        var timeoutAddress = ctx[CpuRegister.Rdx];

        if (!_semaphores.TryGetValue(handle, out var semaphore))
        {
            return SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND);
        }

        if (needCount < 1 || needCount > semaphore.MaxCount)
        {
            return SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT);
        }

        uint timeoutUsec = 0;
        if (timeoutAddress != 0 && !ctx.TryReadUInt32(timeoutAddress, out timeoutUsec))
        {
            return SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
        }

        var deadline = timeoutAddress != 0 ? GuestTime.DeadlineAfterMicroseconds(timeoutUsec) : 0;
        return WaitSemaphoreCore(
            ctx,
            semaphore,
            handle,
            needCount,
            deadline,
            timeoutAddress,
            SemaphoreResultConvention.Sce,
            "sceKernelWaitSema");
    }

    private enum SemaphoreResultConvention
    {
        // sce* code; ECANCELED/EACCES reported as such.
        Sce,

        // libc sem_*: -1 + errno; a destroyed semaphore reports EINVAL.
        Libc,
    }

    /// <summary>
    /// The semaphore wait core. <paramref name="deadline"/> is a Stopwatch deadline (0: none).
    /// <paramref name="timeoutAddress"/>, when set, receives the remaining microseconds on
    /// every blocking return (left unchanged on an immediate grant, 0 on timeout).
    /// </summary>
    private static int WaitSemaphoreCore(
        CpuContext ctx,
        KernelSemaphoreState semaphore,
        uint handle,
        int needCount,
        long deadline,
        ulong timeoutAddress,
        SemaphoreResultConvention convention,
        string operationName)
    {
        SemaphoreWaiter? waiter = null;
        var queue = semaphore.Queue;
        lock (queue.Gate)
        {
            if (queue.Deleted)
            {
                return Finish(ctx, convention, GuestErrno.EACCES);
            }

            // No barging past queued waiters: they are served in order.
            if (queue.Count == 0 && semaphore.Count >= needCount)
            {
                semaphore.Count -= needCount;
                if (_traceSema)
                {
                    TraceSemaphore($"wait handle=0x{handle:X8} name='{semaphore.Name}' need={needCount} count={semaphore.Count}");
                }

                return Finish(ctx, convention, 0);
            }

            if (deadline == 0 || deadline > Stopwatch.GetTimestamp())
            {
                waiter = new SemaphoreWaiter(
                    needCount,
                    semaphore.PriorityOrder ? KernelPthreadExtendedCompatExports.GetCurrentThreadPriority() : 0);
                queue.EnqueueLocked(waiter);
                waiter.ArmDeadlineLocked(deadline);
            }
        }

        if (waiter is null)
        {
            if (timeoutAddress != 0)
            {
                _ = ctx.TryWriteUInt32(timeoutAddress, 0);
            }

            return Finish(ctx, convention, GuestErrno.ETIMEDOUT);
        }

        if (_traceSema)
        {
            TraceSemaphore($"wait-block handle=0x{handle:X8} name='{semaphore.Name}' need={needCount} waiters={semaphore.WaitingThreads} coop={waiter.Cooperative} {FormatCallSite(ctx)}");
        }

        return waiter.Block(ctx, operationName, completed =>
        {
            if (timeoutAddress != 0)
            {
                _ = ctx.TryWriteUInt32(timeoutAddress, (uint)Math.Min(completed.RemainingMicroseconds, uint.MaxValue));
            }

            // FreeBSD libc's sem_timedwait (sem_new.c) re-checks the count after the kernel wait
            // times out and takes a unit posted meanwhile instead of failing. The timed-out guest
            // thread can resume well after its deadline here, and Mono retires a thread-pool worker
            // whose park times out: a post made to wake it in that gap was left in the count while
            // the worker exited, so the queued work had no thread and the level load never finished.
            if (convention == SemaphoreResultConvention.Libc && completed.Status == GuestWaitStatus.TimedOut)
            {
                lock (queue.Gate)
                {
                    if (!queue.Deleted && queue.Count == 0 && semaphore.Count >= needCount)
                    {
                        semaphore.Count -= needCount;
                        if (_traceSema)
                        {
                            TraceSemaphore($"wait-late-take handle=0x{handle:X8} name='{semaphore.Name}' need={needCount} count={semaphore.Count}");
                        }

                        return Finish(ctx, convention, 0);
                    }
                }
            }

            if (_traceSema)
            {
                TraceSemaphore($"wait-done handle=0x{handle:X8} name='{semaphore.Name}' status={completed.Status}");
            }

            return Finish(ctx, convention, GuestErrno.FromWaitStatus(completed.Status));
        });
    }

    private static int Finish(CpuContext ctx, SemaphoreResultConvention convention, int errno)
    {
        if (convention == SemaphoreResultConvention.Libc)
        {
            // A semaphore destroyed under a sem_wait is no longer a valid semaphore.
            return GuestErrno.ReturnLibc(ctx, errno is GuestErrno.EACCES or GuestErrno.ECANCELED ? GuestErrno.EINVAL : errno);
        }

        return GuestErrno.ReturnSce(ctx, errno);
    }

    [SysAbiExport(
        Nid = "12wOHk8ywb0",
        ExportName = "sceKernelPollSema",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelPollSema(CpuContext ctx, uint handle, int needCount)
    {
        if (!_semaphores.TryGetValue(handle, out var semaphore))
        {
            return SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND);
        }

        if (needCount < 1 || needCount > semaphore.MaxCount)
        {
            return SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT);
        }

        return GuestErrno.ReturnSce(ctx, TryAcquireNow(semaphore, needCount));
    }

    private static int TryAcquireNow(KernelSemaphoreState semaphore, int needCount)
    {
        lock (semaphore.Queue.Gate)
        {
            if (semaphore.Queue.Deleted)
            {
                return GuestErrno.EACCES;
            }

            if (semaphore.Queue.Count != 0 || semaphore.Count < needCount)
            {
                return GuestErrno.EBUSY;
            }

            semaphore.Count -= needCount;
            return 0;
        }
    }

    [SysAbiExport(
        Nid = "4czppHBiriw",
        ExportName = "sceKernelSignalSema",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelSignalSema(CpuContext ctx, uint handle, int signalCount)
    {
        var signalTrace = SemaphoreSignalProfile.Begin(handle, signalCount);
        if (!_semaphores.TryGetValue(handle, out var semaphore))
        {
            signalTrace.Record(SemaphoreSignalProfile.Stage.Rejected,
                result: (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND);
            return SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND);
        }

        var errno = SignalCore(semaphore, signalCount, signalTrace, out var woken);
        if (errno != 0)
        {
            signalTrace.Record(SemaphoreSignalProfile.Stage.Rejected, result: GuestErrno.ToSce(errno));
            return GuestErrno.ReturnSce(ctx, errno);
        }

        if (_traceSema)
        {
            TraceSemaphore($"signal handle=0x{handle:X8} name='{semaphore.Name}' signal={signalCount} woken={woken} {FormatCallSite(ctx)}");
        }
        return SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_OK);
    }

    private static int SignalCore(
        KernelSemaphoreState semaphore,
        int signalCount,
        SemaphoreSignalProfile.Signal signalTrace,
        out int woken)
    {
        woken = 0;
        if (signalCount < 0)
        {
            return GuestErrno.EINVAL;
        }

        var batch = default(WakeBatch);
        lock (semaphore.Queue.Gate)
        {
            if (semaphore.Queue.Deleted)
            {
                return GuestErrno.EACCES;
            }

            if (semaphore.Count > semaphore.MaxCount - signalCount)
            {
                return GuestErrno.EINVAL;
            }

            semaphore.Count += signalCount;
            signalTrace.Record(SemaphoreSignalProfile.Stage.Published, semaphore.Count, semaphore.WaitingThreads);
            woken = semaphore.GrantWaitersLocked(ref batch);
        }

        // Only the granted waiters are woken, directly; nobody waiting means no scheduler call.
        signalTrace.Record(SemaphoreSignalProfile.Stage.WakeStarted);
        batch.WakeAll();
        signalTrace.Record(SemaphoreSignalProfile.Stage.WakeFinished, result: woken);
        return 0;
    }

    [SysAbiExport(
        Nid = "4DM06U2BNEY",
        ExportName = "sceKernelCancelSema",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelCancelSema(CpuContext ctx, uint handle, int setCount, ulong waitingThreadsAddress)
    {
        if (!_semaphores.TryGetValue(handle, out var semaphore))
        {
            return SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND);
        }

        if (setCount > semaphore.MaxCount)
        {
            return SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT);
        }

        // Every current waiter returns ECANCELED; the count becomes setCount (the initial
        // count for a negative setCount) and the number of released waiters is reported.
        var batch = default(WakeBatch);
        int waiting;
        lock (semaphore.Queue.Gate)
        {
            waiting = semaphore.Queue.CompleteAllLocked(GuestWaitStatus.Canceled, ref batch);
            semaphore.Queue.Epoch++;
            semaphore.Count = setCount < 0 ? semaphore.InitialCount : setCount;
        }

        batch.WakeAll();
        if (_traceSema)
        {
            TraceSemaphore($"cancel handle=0x{handle:X8} name='{semaphore.Name}' set={setCount} waiters={waiting}");
        }

        if (waitingThreadsAddress != 0 && !ctx.TryWriteUInt32(waitingThreadsAddress, unchecked((uint)waiting)))
        {
            return SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
        }

        return SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_OK);
    }

    [SysAbiExport(
        Nid = "R1Jvn8bSCW8",
        ExportName = "sceKernelDeleteSema",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelDeleteSema(CpuContext ctx)
    {
        var handle = unchecked((uint)ctx[CpuRegister.Rdi]);
        if (!_semaphores.TryRemove(handle, out var semaphore))
        {
            return SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND);
        }

        DeleteCore(semaphore);
        if (_traceSema)
        {
            TraceSemaphore($"delete handle=0x{handle:X8} name='{semaphore.Name}'");
        }
        return SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_OK);
    }

    // Waiters on a deleted semaphore return EACCES (the sce DELETED code).
    private static void DeleteCore(KernelSemaphoreState semaphore)
    {
        var batch = default(WakeBatch);
        lock (semaphore.Queue.Gate)
        {
            semaphore.Queue.Deleted = true;
            _ = semaphore.Queue.CompleteAllLocked(GuestWaitStatus.Deleted, ref batch);
        }

        batch.WakeAll();
    }

    [SysAbiExport(
        Nid = "pDuPEf3m4fI",
        ExportName = "sem_init",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixSemInit(CpuContext ctx) =>
        GuestErrno.ReturnLibc(ctx, SemInitCore(ctx));

    private static int SemInitCore(CpuContext ctx)
    {
        var semaphoreAddress = ctx[CpuRegister.Rdi];
        var initialCountValue = ctx[CpuRegister.Rdx];
        if (semaphoreAddress == 0 || initialCountValue > int.MaxValue)
        {
            return GuestErrno.EINVAL;
        }

        var handle = AllocateHandle();
        var initialCount = unchecked((int)initialCountValue);
        _semaphores[handle] = new KernelSemaphoreState(priorityOrder: false)
        {
            Name = $"posix@0x{semaphoreAddress:X16}",
            InitialCount = initialCount,
            MaxCount = int.MaxValue,
            PriorityOrder = false,
            Count = initialCount,
        };
        if (!ctx.TryWriteUInt32(semaphoreAddress, handle))
        {
            _semaphores.TryRemove(handle, out _);
            return GuestErrno.EFAULT;
        }

        if (_traceSema)
        {
            TraceSemaphore($"posix-init address=0x{semaphoreAddress:X16} handle=0x{handle:X8} count={initialCount}");
        }
        return 0;
    }

    [SysAbiExport(
        Nid = "GEnUkDZoUwY",
        ExportName = "scePthreadSemInit",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PthreadSemInit(CpuContext ctx)
    {
        // scePthreadSemInit(sem, flag, value, name) seems to only support private semaphores
        if (ctx[CpuRegister.Rsi] != 0)
        {
            return SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT);
        }

        return GuestErrno.ReturnSce(ctx, SemInitCore(ctx));
    }

    [SysAbiExport(
        Nid = "YCV5dGGBcCo",
        ExportName = "sem_wait",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixSemWait(CpuContext ctx) =>
        PosixWaitCore(ctx, deadline: 0, SemaphoreResultConvention.Libc, "sem_wait");

    [SysAbiExport(
        Nid = "C36iRE0F5sE",
        ExportName = "scePthreadSemWait",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PthreadSemWait(CpuContext ctx) =>
        PosixWaitCore(ctx, deadline: 0, SemaphoreResultConvention.Sce, "scePthreadSemWait");

    private static int PosixWaitCore(CpuContext ctx, long deadline, SemaphoreResultConvention convention, string operation)
    {
        if (!TryGetPosixSemaphore(ctx, ctx[CpuRegister.Rdi], out var handle, out var semaphore))
        {
            return Finish(ctx, convention, GuestErrno.EINVAL);
        }

        return WaitSemaphoreCore(ctx, semaphore, handle, 1, deadline, timeoutAddress: 0, convention, operation);
    }

    [SysAbiExport(
        Nid = "WBWzsRifCEA",
        ExportName = "sem_trywait",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixSemTryWait(CpuContext ctx)
    {
        if (!TryGetPosixSemaphore(ctx, ctx[CpuRegister.Rdi], out _, out var semaphore))
        {
            return GuestErrno.ReturnLibc(ctx, GuestErrno.EINVAL);
        }

        var errno = TryAcquireNow(semaphore, 1);
        return Finish(ctx, SemaphoreResultConvention.Libc, errno == GuestErrno.EBUSY ? GuestErrno.EAGAIN : errno);
    }

    [SysAbiExport(
        Nid = "H2a+IN9TP0E",
        ExportName = "scePthreadSemTrywait",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PthreadSemTryWait(CpuContext ctx)
    {
        if (!TryGetPosixSemaphore(ctx, ctx[CpuRegister.Rdi], out _, out var semaphore))
        {
            return GuestErrno.ReturnSce(ctx, GuestErrno.EINVAL);
        }

        var errno = TryAcquireNow(semaphore, 1);
        return GuestErrno.ReturnSce(ctx, errno == GuestErrno.EBUSY ? GuestErrno.EAGAIN : errno);
    }

    [SysAbiExport(
        Nid = "fjN6NQHhK8k",
        ExportName = "scePthreadSemTimedwait",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PthreadSemTimedWait(CpuContext ctx)
    {
        // scePthreadSemTimedwait(sem, usec): a relative microsecond timeout.
        var timeoutUsec = unchecked((uint)ctx[CpuRegister.Rsi]);
        return PosixWaitCore(
            ctx,
            GuestTime.DeadlineAfterMicroseconds(timeoutUsec),
            SemaphoreResultConvention.Sce,
            "scePthreadSemTimedwait");
    }

    [SysAbiExport(
        Nid = "w5IHyvahg-o",
        ExportName = "sem_timedwait",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixSemTimedWait(CpuContext ctx)
    {
        // sem_timedwait(sem, abstime): an absolute CLOCK_REALTIME timespec.
        if (!TryGetPosixSemaphore(ctx, ctx[CpuRegister.Rdi], out var handle, out var semaphore))
        {
            return GuestErrno.ReturnLibc(ctx, GuestErrno.EINVAL);
        }

        // POSIX: the timeout is not validated when the semaphore can be taken at once.
        if (TryAcquireNow(semaphore, 1) == 0)
        {
            return GuestErrno.ReturnLibc(ctx, 0);
        }

        if (!GuestTime.TryReadAbsoluteDeadline(ctx, ctx[CpuRegister.Rsi], GuestTime.ClockRealtime, out var deadline, out var errno))
        {
            return GuestErrno.ReturnLibc(ctx, errno);
        }

        return WaitSemaphoreCore(ctx, semaphore, handle, 1, deadline, timeoutAddress: 0, SemaphoreResultConvention.Libc, "sem_timedwait");
    }

    [SysAbiExport(
        Nid = "IKP8typ0QUk",
        ExportName = "sem_post",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixSemPost(CpuContext ctx)
    {
        if (!TryGetPosixSemaphore(ctx, ctx[CpuRegister.Rdi], out _, out var semaphore))
        {
            return GuestErrno.ReturnLibc(ctx, GuestErrno.EINVAL);
        }

        var errno = SignalCore(semaphore, 1, default, out _);
        return Finish(ctx, SemaphoreResultConvention.Libc, errno == GuestErrno.EINVAL ? GuestErrno.EOVERFLOW : errno);
    }

    [SysAbiExport(
        Nid = "aishVAiFaYM",
        ExportName = "scePthreadSemPost",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PthreadSemPost(CpuContext ctx)
    {
        if (!TryGetPosixSemaphore(ctx, ctx[CpuRegister.Rdi], out _, out var semaphore))
        {
            return GuestErrno.ReturnSce(ctx, GuestErrno.EINVAL);
        }

        return GuestErrno.ReturnSce(ctx, SignalCore(semaphore, 1, default, out _));
    }

    [SysAbiExport(
        Nid = "Bq+LRV-N6Hk",
        ExportName = "sem_getvalue",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixSemGetValue(CpuContext ctx)
    {
        var semaphoreAddress = ctx[CpuRegister.Rdi];
        var valueAddress = ctx[CpuRegister.Rsi];
        if (valueAddress == 0 ||
            !TryGetPosixSemaphore(ctx, semaphoreAddress, out _, out var semaphore))
        {
            return GuestErrno.ReturnLibc(ctx, GuestErrno.EINVAL);
        }

        int count;
        lock (semaphore.Queue.Gate)
        {
            count = semaphore.Count;
        }

        return GuestErrno.ReturnLibc(
            ctx,
            ctx.TryWriteUInt32(valueAddress, unchecked((uint)count)) ? 0 : GuestErrno.EFAULT);
    }

    [SysAbiExport(
        Nid = "cDW233RAwWo",
        ExportName = "sem_destroy",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixSemDestroy(CpuContext ctx) =>
        GuestErrno.ReturnLibc(ctx, SemDestroyCore(ctx));

    [SysAbiExport(
        Nid = "Vwc+L05e6oE",
        ExportName = "scePthreadSemDestroy",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PthreadSemDestroy(CpuContext ctx) =>
        GuestErrno.ReturnSce(ctx, SemDestroyCore(ctx));

    private static int SemDestroyCore(CpuContext ctx)
    {
        var semaphoreAddress = ctx[CpuRegister.Rdi];
        if (!TryGetPosixSemaphoreHandle(ctx, semaphoreAddress, out var handle) ||
            !_semaphores.TryRemove(handle, out var semaphore))
        {
            return GuestErrno.EINVAL;
        }

        DeleteCore(semaphore);
        _ = ctx.TryWriteUInt32(semaphoreAddress, 0);
        return 0;
    }

    internal static int WaiterCountForTests(uint handle) =>
        _semaphores.TryGetValue(handle, out var semaphore) ? semaphore.WaitingThreads : -1;

    private static uint AllocateHandle()
    {
        var handle = unchecked((uint)Interlocked.Increment(ref _nextSemaphoreHandle));
        if (handle == 0)
        {
            handle = unchecked((uint)Interlocked.Increment(ref _nextSemaphoreHandle));
        }

        return handle;
    }

    private static bool TryGetPosixSemaphore(
        CpuContext ctx,
        ulong semaphoreAddress,
        out uint handle,
        [NotNullWhen(true)] out KernelSemaphoreState? semaphore)
    {
        semaphore = null;
        return TryGetPosixSemaphoreHandle(ctx, semaphoreAddress, out handle) &&
               _semaphores.TryGetValue(handle, out semaphore);
    }

    private static bool TryGetPosixSemaphoreHandle(CpuContext ctx, ulong semaphoreAddress, out uint handle)
    {
        handle = 0;
        return semaphoreAddress != 0 &&
               ctx.TryReadUInt32(semaphoreAddress, out handle) &&
               handle != 0;
    }

    private static int SetReturn(CpuContext ctx, OrbisGen2Result result)
    {
        var value = (int)result;
        ctx[CpuRegister.Rax] = unchecked((ulong)value);
        return value;
    }

    private static readonly bool _traceSema =
        string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_LOG_SEMA"), "1", StringComparison.Ordinal);

    private static void TraceSemaphore(string message)
    {
        if (!_traceSema)
        {
            return;
        }

        Console.Error.WriteLine($"[LOADER][TRACE] sema.{message}");
    }

    private static string FormatCallSite(CpuContext ctx)
    {
        _ = ctx.TryReadUInt64(ctx[CpuRegister.Rsp], out var returnAddress);
        return $"guest=0x{GuestThreadExecution.CurrentGuestThreadHandle:X16} ret=0x{returnAddress:X16}";
    }
}
