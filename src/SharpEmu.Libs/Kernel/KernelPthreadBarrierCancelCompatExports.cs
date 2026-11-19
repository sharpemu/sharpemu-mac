// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;
using SharpEmu.HLE;
using SharpEmu.Libs.Threading;

namespace SharpEmu.Libs.Kernel;

/// <summary>
/// pthread barriers, deferred cancellation, sceKernelSleep and the sched priority range.
/// pthread_* return errno values; scePthread* the matching sce code (GuestErrno.ToSce).
/// </summary>
public static class KernelPthreadBarrierCancelCompatExports
{
    /// <summary>PTHREAD_BARRIER_SERIAL_THREAD: returned to exactly one waiter per cycle.</summary>
    public const int BarrierSerialThread = -1;

    private const ulong BarrierHandleBase = 0x00006007_0000_0000;
    private const ulong BarrierAttrHandleBase = 0x00006008_0000_0000;
    private const int CancelEnable = 0;
    private const int CancelDisable = 1;
    private const int CancelDeferred = 0;
    private const int CancelAsynchronous = 2;
    private const int SchedFifo = 1;
    private const int SchedRr = 3;
    private const ulong PthreadCanceled = ulong.MaxValue; // (void*)-1

    private static readonly ConcurrentDictionary<ulong, BarrierState> _barriers = new();
    private static readonly ConcurrentDictionary<ulong, byte> _barrierAttrs = new();
    private static readonly ConcurrentDictionary<ulong, CancelState> _cancelStates = new();
    private static long _nextBarrierId;
    private static long _nextBarrierAttrId;

    private sealed class BarrierState(uint count)
    {
        public WaitQueue Queue { get; } = new();
        public uint Count { get; } = count;
        public uint Arrived;
        public ulong Cycle;
    }

    private sealed class CancelState
    {
        public int State = CancelEnable;
        public int Type = CancelDeferred;
        public int Pending;
    }

    // ---- barriers ----

    [SysAbiExport(Nid = "ZsXLFtd2jqQ", ExportName = "pthread_barrier_init", Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PosixPthreadBarrierInit(CpuContext ctx) =>
        BarrierInitCore(ctx, ctx[CpuRegister.Rdi], unchecked((uint)ctx[CpuRegister.Rdx]));

    [SysAbiExport(Nid = "5dgOEPsEGqw", ExportName = "scePthreadBarrierInit", Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PthreadBarrierInit(CpuContext ctx) =>
        Sce(BarrierInitCore(ctx, ctx[CpuRegister.Rdi], unchecked((uint)ctx[CpuRegister.Rdx])));

    [SysAbiExport(Nid = "+Pqub9HZCPo", ExportName = "pthread_barrier_destroy", Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PosixPthreadBarrierDestroy(CpuContext ctx) => BarrierDestroyCore(ctx, ctx[CpuRegister.Rdi]);

    [SysAbiExport(Nid = "HudB2Jv2MPY", ExportName = "scePthreadBarrierDestroy", Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PthreadBarrierDestroy(CpuContext ctx) => Sce(BarrierDestroyCore(ctx, ctx[CpuRegister.Rdi]));

    [SysAbiExport(Nid = "CawZgCYqXWk", ExportName = "pthread_barrier_wait", Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PosixPthreadBarrierWait(CpuContext ctx) => BarrierWaitCore(ctx, ctx[CpuRegister.Rdi], posix: true);

    // libkernel's wrapper passes PTHREAD_BARRIER_SERIAL_THREAD (-1) through unchanged.
    [SysAbiExport(Nid = "t9vVyTglqHQ", ExportName = "scePthreadBarrierWait", Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PthreadBarrierWait(CpuContext ctx) => BarrierWaitCore(ctx, ctx[CpuRegister.Rdi], posix: false);

    [SysAbiExport(Nid = "4nqCnLJSvck", ExportName = "pthread_barrierattr_init", Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PosixPthreadBarrierattrInit(CpuContext ctx) => BarrierAttrInitCore(ctx, ctx[CpuRegister.Rdi]);

    [SysAbiExport(Nid = "SDkV9xhINKI", ExportName = "scePthreadBarrierattrInit", Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PthreadBarrierattrInit(CpuContext ctx) => Sce(BarrierAttrInitCore(ctx, ctx[CpuRegister.Rdi]));

    [SysAbiExport(Nid = "AsCQCYTbe80", ExportName = "pthread_barrierattr_destroy", Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PosixPthreadBarrierattrDestroy(CpuContext ctx) => BarrierAttrDestroyCore(ctx, ctx[CpuRegister.Rdi]);

    [SysAbiExport(Nid = "oT-j4DqJHY8", ExportName = "scePthreadBarrierattrDestroy", Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PthreadBarrierattrDestroy(CpuContext ctx) => Sce(BarrierAttrDestroyCore(ctx, ctx[CpuRegister.Rdi]));

    private static int BarrierInitCore(CpuContext ctx, ulong barrierAddress, uint count)
    {
        if (barrierAddress == 0 || count == 0)
        {
            return GuestErrno.EINVAL;
        }

        var handle = BarrierHandleBase + ((ulong)Interlocked.Increment(ref _nextBarrierId) << 4);
        if (!ctx.TryWriteUInt64(barrierAddress, handle))
        {
            return GuestErrno.EFAULT;
        }

        _barriers[handle] = new BarrierState(count);
        return 0;
    }

    private static bool TryResolveBarrier(CpuContext ctx, ulong barrierAddress, out ulong handle, out BarrierState state)
    {
        handle = 0;
        state = null!;
        return barrierAddress != 0 &&
            ctx.TryReadUInt64(barrierAddress, out handle) &&
            _barriers.TryGetValue(handle, out state!);
    }

    private static int BarrierDestroyCore(CpuContext ctx, ulong barrierAddress)
    {
        if (!TryResolveBarrier(ctx, barrierAddress, out var handle, out var state))
        {
            return GuestErrno.EINVAL;
        }

        lock (state.Queue.Gate)
        {
            if (state.Arrived != 0)
            {
                return GuestErrno.EBUSY;
            }

            state.Queue.Deleted = true;
        }

        _ = _barriers.TryRemove(handle, out _);
        _ = ctx.TryWriteUInt64(barrierAddress, 0);
        return 0;
    }

    private static int BarrierWaitCore(CpuContext ctx, ulong barrierAddress, bool posix)
    {
        if (!TryResolveBarrier(ctx, barrierAddress, out _, out var state))
        {
            return posix ? GuestErrno.EINVAL : GuestErrno.ToSce(GuestErrno.EINVAL);
        }

        GuestWaiter? waiter = null;
        var batch = default(WakeBatch);
        lock (state.Queue.Gate)
        {
            if (++state.Arrived == state.Count)
            {
                // The last arrival releases the cycle and is the serial thread.
                state.Arrived = 0;
                state.Cycle++;
                _ = state.Queue.CompleteAllLocked(GuestWaitStatus.Satisfied, ref batch);
            }
            else
            {
                waiter = new GuestWaiter();
                state.Queue.EnqueueLocked(waiter);
            }
        }

        if (waiter is null)
        {
            batch.WakeAll();
            return BarrierSerialThread;
        }

        return waiter.Block(ctx, "pthread_barrier_wait", completed =>
        {
            var errno = GuestErrno.FromWaitStatus(completed.Status);
            return posix || errno == 0 ? errno : GuestErrno.ToSce(errno);
        });
    }

    private static int BarrierAttrInitCore(CpuContext ctx, ulong attrAddress)
    {
        if (attrAddress == 0)
        {
            return GuestErrno.EINVAL;
        }

        var handle = BarrierAttrHandleBase + ((ulong)Interlocked.Increment(ref _nextBarrierAttrId) << 4);
        if (!ctx.TryWriteUInt64(attrAddress, handle))
        {
            return GuestErrno.EFAULT;
        }

        _barrierAttrs[handle] = 0;
        return 0;
    }

    private static int BarrierAttrDestroyCore(CpuContext ctx, ulong attrAddress)
    {
        if (attrAddress == 0 ||
            !ctx.TryReadUInt64(attrAddress, out var handle) ||
            !_barrierAttrs.TryRemove(handle, out _))
        {
            return GuestErrno.EINVAL;
        }

        _ = ctx.TryWriteUInt64(attrAddress, 0);
        return 0;
    }

    // ---- cancellation ----
    //
    // Deferred cancellation only: pthread_cancel marks the target and the target acts at
    // pthread_testcancel (exits with PTHREAD_CANCELED after running TLS and thread
    // destructors, as pthread_exit). Blocking calls are not cancellation points here and
    // asynchronous cancellation is not delivered; cleanup handlers pushed with
    // pthread_cleanup_push are not run.

    [SysAbiExport(Nid = "0D4-FVvEikw", ExportName = "pthread_cancel", Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PosixPthreadCancel(CpuContext ctx) => CancelCore(ctx[CpuRegister.Rdi]);

    [SysAbiExport(Nid = "qBDmpCyGssE", ExportName = "scePthreadCancel", Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PthreadCancel(CpuContext ctx) => Sce(CancelCore(ctx[CpuRegister.Rdi]));

    [SysAbiExport(Nid = "nYBrkGDqxh8", ExportName = "pthread_testcancel", Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PosixPthreadTestcancel(CpuContext ctx) => TestCancelCore(ctx);

    [SysAbiExport(Nid = "LapIb799SSE", ExportName = "scePthreadTestcancel", Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PthreadTestcancel(CpuContext ctx) => TestCancelCore(ctx);

    [SysAbiExport(Nid = "lZzFeSxPl08", ExportName = "pthread_setcancelstate", Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PosixPthreadSetcancelstate(CpuContext ctx) =>
        SetCancelField(ctx, unchecked((int)ctx[CpuRegister.Rdi]), ctx[CpuRegister.Rsi], type: false);

    [SysAbiExport(Nid = "OAmWq+OHSjw", ExportName = "scePthreadSetcancelstate", Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PthreadSetcancelstate(CpuContext ctx) =>
        Sce(SetCancelField(ctx, unchecked((int)ctx[CpuRegister.Rdi]), ctx[CpuRegister.Rsi], type: false));

    [SysAbiExport(Nid = "2dEhvvjlq30", ExportName = "pthread_setcanceltype", Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PosixPthreadSetcanceltype(CpuContext ctx) =>
        SetCancelField(ctx, unchecked((int)ctx[CpuRegister.Rdi]), ctx[CpuRegister.Rsi], type: true);

    [SysAbiExport(Nid = "sCJd99Phct0", ExportName = "scePthreadSetcanceltype", Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int PthreadSetcanceltype(CpuContext ctx) =>
        Sce(SetCancelField(ctx, unchecked((int)ctx[CpuRegister.Rdi]), ctx[CpuRegister.Rsi], type: true));

    private static int CancelCore(ulong thread)
    {
        if (thread == 0)
        {
            return GuestErrno.ESRCH;
        }

        var state = _cancelStates.GetOrAdd(thread, static _ => new CancelState());
        Volatile.Write(ref state.Pending, 1);
        return 0;
    }

    /// <summary>True when the calling thread has a cancellation it must act on now.</summary>
    internal static bool IsCancelPending(ulong thread) =>
        _cancelStates.TryGetValue(thread, out var state) &&
        Volatile.Read(ref state.Pending) != 0 &&
        Volatile.Read(ref state.State) == CancelEnable;

    private static int TestCancelCore(CpuContext ctx)
    {
        var thread = KernelPthreadState.GetCurrentThreadHandle();
        if (!IsCancelPending(thread))
        {
            return 0;
        }

        _ = _cancelStates.TryRemove(thread, out _);
        RunThreadExitDestructors(ctx);
        GuestThreadExecution.RequestCurrentEntryExit("pthread_testcancel", PthreadCanceled);
        ctx[CpuRegister.Rax] = PthreadCanceled;
        return 0;
    }

    private static int SetCancelField(CpuContext ctx, int value, ulong oldAddress, bool type)
    {
        if (type ? value is not (CancelDeferred or CancelAsynchronous) : value is not (CancelEnable or CancelDisable))
        {
            return GuestErrno.EINVAL;
        }

        var state = _cancelStates.GetOrAdd(KernelPthreadState.GetCurrentThreadHandle(), static _ => new CancelState());
        var old = type
            ? Interlocked.Exchange(ref state.Type, value)
            : Interlocked.Exchange(ref state.State, value);
        if (oldAddress != 0 && !ctx.TryWriteUInt32(oldAddress, unchecked((uint)old)))
        {
            return GuestErrno.EFAULT;
        }

        return 0;
    }

    // ---- thread exit ----

    private static readonly ConcurrentDictionary<ulong, byte> _exitDestructorsRan = new();

    static KernelPthreadBarrierCancelCompatExports()
    {
        GuestThreadExecution.GuestThreadEntryReturned += OnGuestThreadEntryReturned;
    }

    /// <summary>
    /// pthread_exit / cancellation: the C++ thread_local destructors the guest's libc registered
    /// (_sceLibcInternalThreadAtexit), then the TLS-key destructors, then the runtime's
    /// thread_atexit callback, on the still-running thread. Recorded so the entry-return hook that
    /// follows the unwind does not run them a second time.
    /// </summary>
    internal static void RunThreadExitDestructors(CpuContext ctx)
    {
        _exitDestructorsRan[KernelPthreadState.GetCurrentThreadHandle()] = 0;
        RunExitDestructors(ctx);
    }

    // A start routine that returns exits the thread exactly like pthread_exit.
    private static void OnGuestThreadEntryReturned(CpuContext ctx)
    {
        var thread = KernelPthreadState.GetCurrentThreadHandle();
        _ = _cancelStates.TryRemove(thread, out _);
        if (_exitDestructorsRan.TryRemove(thread, out _))
        {
            return;
        }

        RunExitDestructors(ctx);
    }

    private static void RunExitDestructors(CpuContext ctx)
    {
        LibcInternal.LibcInternalExports.RunThreadAtexitDestructors(ctx);
        KernelPthreadExtendedCompatExports.RunThreadLocalDestructors(ctx);
        KernelMemoryCompatExports.RunThreadDtors(ctx);
    }

    // ---- sleep / sched ----

    /// <summary>sceKernelSleep(seconds): a plain sleep, no early wake.</summary>
    [SysAbiExport(Nid = "-ZR+hG7aDHw", ExportName = "sceKernelSleep", Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int KernelSleep(CpuContext ctx)
    {
        var seconds = unchecked((uint)ctx[CpuRegister.Rdi]);
        if (seconds != 0)
        {
            GuestThreadExecution.Scheduler?.Pump(ctx, "sceKernelSleep");
            HostTiming.SleepMicroseconds(seconds * 1_000_000L);
        }

        return 0;
    }

    // Orbis real-time priorities run 256 (highest) .. 767 (lowest) for SCHED_FIFO and
    // SCHED_RR; other policies have no range here and fail with EINVAL (libc: -1 + errno).
    [SysAbiExport(Nid = "CBNtXOoef-E", ExportName = "sched_get_priority_max", Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int SchedGetPriorityMax(CpuContext ctx) => SchedPriorityBound(ctx, 767);

    [SysAbiExport(Nid = "m0iS6jNsXds", ExportName = "sched_get_priority_min", Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int SchedGetPriorityMin(CpuContext ctx) => SchedPriorityBound(ctx, 256);

    private static int SchedPriorityBound(CpuContext ctx, int bound)
    {
        var policy = unchecked((int)ctx[CpuRegister.Rdi]);
        return policy is SchedFifo or SchedRr ? bound : GuestErrno.ReturnLibc(ctx, GuestErrno.EINVAL);
    }

    private static int Sce(int errno) => errno == 0 ? 0 : GuestErrno.ToSce(errno);
}
