// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using System.Buffers;
using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Threading;

namespace SharpEmu.Libs.Kernel;

/// <summary>
/// The kqueue core behind every equeue. Each queue owns a knote table keyed by
/// (ident, filter) and a list of active knotes in trigger order, as FreeBSD 11's
/// <c>kern_event.c</c> does. Every producer (user events, VideoOut, AGC end of
/// pipe, AMPR, timers, descriptors) changes a knote through
/// <see cref="ActivateLocked"/>, and every activation wakes both guest waiters
/// (through the scheduler's wake key) and host waiters (through the queue
/// monitor), so no trigger can be lost.
/// </summary>
public static partial class KernelEventQueueCompatExports
{
    private const int KernelEventSize = 0x20;

    public const short KernelEventFilterRead = -1;
    public const short KernelEventFilterWrite = -2;
    public const short KernelEventFilterVnode = -4;
    public const short KernelEventFilterTimer = -7;
    public const short KernelEventFilterUser = -11;
    public const short KernelEventFilterDisplay = -13;
    public const short KernelEventFilterGraphics = -14;
    public const short KernelEventFilterHrTimer = -15;
    public const short KernelEventFilterAmpr = -25;
    public const short KernelEventFilterAmprSystem = -30;

    public const ushort KernelEventFlagAdd = 0x0001;
    public const ushort KernelEventFlagDelete = 0x0002;
    public const ushort KernelEventFlagEnable = 0x0004;
    public const ushort KernelEventFlagDisable = 0x0008;
    public const ushort KernelEventFlagOneShot = 0x0010;
    public const ushort KernelEventFlagClear = 0x0020;
    public const ushort KernelEventFlagReceipt = 0x0040;
    public const ushort KernelEventFlagDispatch = 0x0080;
    public const ushort KernelEventFlagForceOneShot = 0x0100;
    public const ushort KernelEventFlagError = 0x4000;
    public const ushort KernelEventFlagEof = 0x8000;

    // The flags a knote keeps from its registration and reports with every event
    // (kn_kevent.flags with EV_ADD/DELETE/ENABLE/DISABLE/FORCEONESHOT stripped).
    private const ushort PersistentFlagMask =
        KernelEventFlagOneShot | KernelEventFlagClear | KernelEventFlagReceipt | KernelEventFlagDispatch;

    /// <summary>
    /// Interrupt filters (AGC end of pipe) report each occurrence with its own
    /// data word, because the guest reads a context id per interrupt. Past this
    /// many undelivered occurrences the oldest is folded into the next one, whose
    /// fflags then count both, so an undrained queue stays bounded.
    /// </summary>
    internal const int MaxPendingOccurrencesPerKnote = 64;

    // Events reported by one wait; the rest stay queued for the next one.
    private const int MaxEventsPerWait = 1024;

    private static readonly object _eventQueueGate = new();
    private static readonly Dictionary<ulong, Kqueue> _eventQueues = new();
    private static Kqueue[] _eventQueueSnapshot = [];
    private static readonly ConditionalWeakTable<object, EventQueueRuntimeIdentity>
        _eventQueueRuntimeIdentities = new();
    private static long _nextEventQueueHandle = 1;
    private static long _nextEventQueueWaiterId;
    private static long _nextEventRegistrationGeneration;
    private static long _nextEventQueueGeneration;
    private static long _nextEventQueueRuntimeId;

    private sealed record EventQueueRuntimeIdentity(ulong Id);

    public readonly record struct KernelQueuedEvent(
        ulong Ident,
        short Filter,
        ushort Flags,
        uint Fflags,
        ulong Data,
        ulong UserData);

    internal readonly record struct KernelEventRegistrationToken(
        ulong EqueueHandle,
        ulong EqueueGeneration,
        ulong Ident,
        short Filter,
        ulong Generation);

    internal sealed record KernelEventRegistrationSnapshot(
        ulong RuntimeId,
        ulong Ident,
        short Filter,
        KernelEventRegistrationToken[] Targets);

    internal readonly record struct CapturedEventDeliveryResult(
        int TriggeredCount,
        int StaleCount);

    private sealed class Kqueue
    {
        public required ulong Handle { get; init; }
        public required ulong RuntimeId { get; init; }
        public required ulong Generation { get; init; }
        public required string WakeKey { get; init; }
        public readonly object Gate = new();
        public readonly Dictionary<(ulong Ident, short Filter), Knote> Knotes = new();
        public readonly KnoteList Active = new();
        public bool Deleted;
        public int Descriptor = -1;

        // Guest waiters (sceKernelWaitEqueue / kevent) that have asked to block on this queue and have
        // not resumed yet. Signal skips the scheduler scan while it is zero; see Signal for why that
        // cannot lose a wake.
        public int BlockedWaiters;

        // Read/write-filter knotes in Knotes, mutated under Gate. A wait or poll only has descriptor
        // readiness to refresh when this is non-zero.
        public int DescriptorKnotes;
    }

    private struct Occurrence
    {
        public ulong Data;
        public uint Count;
    }

    // Fixed ring of undelivered interrupt occurrences for one knote.
    private sealed class OccurrenceRing
    {
        private readonly Occurrence[] _items = new Occurrence[MaxPendingOccurrencesPerKnote];
        private int _head;

        public int Count { get; private set; }

        public void Push(ulong data)
        {
            if (Count == _items.Length)
            {
                // Fold the oldest occurrence into the next so the newest data survives.
                var oldest = _items[_head];
                _head = (_head + 1) % _items.Length;
                Count--;
                ref var next = ref _items[_head];
                next.Count = SaturatingAdd(next.Count, oldest.Count);
            }

            _items[(_head + Count) % _items.Length] = new Occurrence { Data = data, Count = 1 };
            Count++;
        }

        public Occurrence Pop()
        {
            var value = _items[_head];
            _head = (_head + 1) % _items.Length;
            Count--;
            return value;
        }

        public void Clear()
        {
            _head = 0;
            Count = 0;
        }
    }

    private sealed class Knote
    {
        public Knote(Kqueue owner, ulong ident, short filter)
        {
            Owner = owner;
            Ident = ident;
            Filter = filter;
            Generation = unchecked((ulong)Interlocked.Increment(ref _nextEventRegistrationGeneration));
        }

        public Kqueue Owner { get; }
        public ulong Ident { get; }
        public short Filter { get; }
        public ulong Generation { get; }

        // Registration (kn_kevent / kn_sfflags / kn_sdata).
        public ushort Flags;
        public uint SFflags;
        public ulong SData;
        public ulong UserData;

        // Status.
        public bool Disabled;
        public bool Queued;
        public bool Triggered;
        public bool Dropped;

        // Pending state reported by the next delivery.
        public uint Fflags;
        public ulong Data;
        public ushort StatusFlags;
        public OccurrenceRing? Occurrences;

        // A raw event injected through EnqueueEvent: delivered exactly once.
        public bool Transient;
        public KernelQueuedEvent Injected;

        // Attached sources.
        public object? TimerToken;
        public long TimerPeriodTicks;
        public long TimerNextDue;
        public IDisposable? Watch;

        public Knote? Prev;
        public Knote? Next;
    }

    // Intrusive doubly linked list, so activation, removal and requeue are O(1).
    private sealed class KnoteList
    {
        public Knote? Head { get; private set; }
        private Knote? _tail;

        public int Count { get; private set; }

        public void AddLast(Knote knote)
        {
            knote.Prev = _tail;
            knote.Next = null;
            if (_tail is null)
            {
                Head = knote;
            }
            else
            {
                _tail.Next = knote;
            }

            _tail = knote;
            Count++;
        }

        public void Remove(Knote knote)
        {
            if (knote.Prev is null)
            {
                Head = knote.Next;
            }
            else
            {
                knote.Prev.Next = knote.Next;
            }

            if (knote.Next is null)
            {
                _tail = knote.Prev;
            }
            else
            {
                knote.Next.Prev = knote.Prev;
            }

            knote.Prev = null;
            knote.Next = null;
            Count--;
        }
    }

    private sealed class EqueueWaiter : IGuestThreadBlockWaiter
    {
        private enum WaitCompletion
        {
            Waiting,
            Reserved,
            TimedOut,
            Deleted,
        }

        private KernelQueuedEvent[]? _reservedEvents;
        private readonly object _completionLock = new();
        private int _reservedCount;
        private WaitCompletion _completion;

        public required CpuContext Ctx { get; init; }
        public required Kqueue State { get; init; }
        public required ulong EventsAddress { get; init; }
        public required int EventCapacity { get; init; }
        public required ulong OutCountAddress { get; init; }
        public required long WaiterId { get; init; }

        // kevent(2) reports the count as its return value and failures through errno.
        public bool PosixResult { get; init; }

        private int _armed;

        // Counts this waiter against its queue until it resumes. Call before asking the scheduler to
        // block: the increment must precede the scheduler's post-park TryWake check (see Signal).
        public void Arm()
        {
            if (Interlocked.Exchange(ref _armed, 1) == 0)
            {
                Interlocked.Increment(ref State.BlockedWaiters);
            }
        }

        // Idempotent; a waiter that never resumes (thread torn down) only leaves the count high, which
        // keeps the scan on for that queue and is never unsafe.
        public void Disarm()
        {
            if (Interlocked.Exchange(ref _armed, 0) == 1)
            {
                Interlocked.Decrement(ref State.BlockedWaiters);
            }
        }

        public int Resume()
        {
            Disarm();
            KernelQueuedEvent[]? reservedEvents;
            int reservedCount;
            WaitCompletion completion;
            lock (_completionLock)
            {
                if (_completion == WaitCompletion.Waiting)
                {
                    _completion = WaitCompletion.TimedOut;
                }

                completion = _completion;
                reservedEvents = _reservedEvents;
                reservedCount = _reservedCount;
                _reservedEvents = null;
                _reservedCount = 0;
            }

            int result;
            if (PosixResult)
            {
                result = completion switch
                {
                    WaitCompletion.Reserved when reservedEvents is not null =>
                        DeliverReservedEventsPosix(Ctx, reservedEvents, reservedCount, EventsAddress),
                    WaitCompletion.Deleted => PosixFail(Ctx, PosixEbadf),
                    _ => 0,
                };
            }
            else
            {
                result = completion switch
                {
                    WaitCompletion.Reserved when reservedEvents is not null =>
                        DeliverReservedEvents(
                            Ctx,
                            reservedEvents,
                            reservedCount,
                            EventsAddress,
                            OutCountAddress),
                    WaitCompletion.Deleted =>
                        (int)OrbisGen2Result.ORBIS_GEN2_ERROR_DELETED,
                    _ => (int)OrbisGen2Result.ORBIS_GEN2_ERROR_TIMED_OUT,
                };
            }

            if (_logEqueue)
            {
                TraceEventQueue(
                    Ctx,
                    "wait-resume",
                    State.Handle,
                    $"generation={State.Generation} waiter={WaiterId} " +
                    $"capacity={EventCapacity} result=0x{unchecked((uint)result):X8}");
            }
            return result;
        }

        public bool TryWake()
        {
            lock (_completionLock)
            {
                if (_completion != WaitCompletion.Waiting)
                {
                    return true;
                }

                if (!TryReserveEvents(
                        State,
                        EventCapacity,
                        out var events,
                        out var count,
                        out var deleted))
                {
                    if (deleted)
                    {
                        _completion = WaitCompletion.Deleted;
                        return true;
                    }

                    return false;
                }

                _reservedEvents = events;
                _reservedCount = count;
                _completion = WaitCompletion.Reserved;
                return true;
            }
        }
    }

    [SysAbiExport(
        Nid = "D0OdFMjp46I",
        ExportName = "sceKernelCreateEqueue",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelCreateEqueue(CpuContext ctx)
    {
        var outAddress = ctx[CpuRegister.Rdi];
        if (outAddress == 0)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        var state = CreateQueue(ctx.Memory);
        if (!ctx.TryWriteUInt64(outAddress, state.Handle))
        {
            _ = DestroyQueue(state.Handle);
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        TraceEventQueue(ctx, "create", state.Handle);
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "jpFjmgAC5AE",
        ExportName = "sceKernelDeleteEqueue",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelDeleteEqueue(CpuContext ctx)
    {
        var handle = ctx[CpuRegister.Rdi];
        if (DestroyQueue(handle) is not { } state)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND;
        }

        TraceEventQueue(ctx, "delete", handle, $"generation={state.Generation}");
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    private static Kqueue CreateQueue(ICpuMemory memory, int descriptor = -1)
    {
        var handle = unchecked((ulong)Interlocked.Increment(ref _nextEventQueueHandle));
        var generation = unchecked((ulong)Interlocked.Increment(ref _nextEventQueueGeneration));
        var state = new Kqueue
        {
            Handle = handle,
            RuntimeId = GetEventQueueRuntimeId(memory),
            Generation = generation,
            WakeKey = $"sceKernelWaitEqueue:{handle:X16}:{generation:X16}",
            Descriptor = descriptor,
        };
        lock (_eventQueueGate)
        {
            _eventQueues.Add(handle, state);
            _eventQueueSnapshot = [.. _eventQueues.Values];
        }

        return state;
    }

    private static Kqueue? DestroyQueue(ulong handle)
    {
        Kqueue? state;
        lock (_eventQueueGate)
        {
            if (!_eventQueues.Remove(handle, out state))
            {
                return null;
            }

            _eventQueueSnapshot = [.. _eventQueues.Values];
        }

        List<Knote> dropped;
        lock (state.Gate)
        {
            state.Deleted = true;
            dropped = [.. state.Knotes.Values];
            foreach (var knote in dropped)
            {
                DropKnoteLocked(state, knote);
            }

            Monitor.PulseAll(state.Gate);
        }

        foreach (var knote in dropped)
        {
            DetachKnoteSources(knote);
        }

        Signal(state, _logEqueue ? $"source=delete generation={state.Generation}" : null);
        return state;
    }

    [SysAbiExport(
        Nid = "VHCS3rCd0PM",
        ExportName = "sceKernelAddReadEvent",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelAddReadEvent(CpuContext ctx) =>
        AddDescriptorEvent(ctx, KernelEventFilterRead, "add_read");

    [SysAbiExport(
        Nid = "JxJ4tfgKlXA",
        ExportName = "sceKernelDeleteReadEvent",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelDeleteReadEvent(CpuContext ctx) =>
        DeleteDescriptorEvent(ctx, KernelEventFilterRead, "delete_read");

    [SysAbiExport(
        Nid = "R-tyYMpYaxY",
        ExportName = "sceKernelAddWriteEvent",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelAddWriteEvent(CpuContext ctx) =>
        AddDescriptorEvent(ctx, KernelEventFilterWrite, "add_write");

    [SysAbiExport(
        Nid = "cBGTk8S92XM",
        ExportName = "sceKernelDeleteWriteEvent",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelDeleteWriteEvent(CpuContext ctx) =>
        DeleteDescriptorEvent(ctx, KernelEventFilterWrite, "delete_write");

    private static int AddDescriptorEvent(CpuContext ctx, short filter, string trace)
    {
        var handle = ctx[CpuRegister.Rdi];
        var fileDescriptor = unchecked((int)ctx[CpuRegister.Rsi]);
        var lowWater = Math.Max(1UL, ctx[CpuRegister.Rdx]);
        var userData = ctx[CpuRegister.Rcx];

        if (!TryGetLiveEventQueue(handle, out _))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND;
        }

        // An unknown descriptor is EBADF, as kqueue_register's fget reports it.
        if (!TryGetDescriptorState(fileDescriptor, filter, lowWater, out _, out _, out _))
        {
            return SceEbadf;
        }

        var registered = RegisterEvent(
            handle,
            unchecked((uint)fileDescriptor),
            filter,
            userData,
            flags: 0,
            lowWater);
        TraceEventQueue(
            ctx,
            trace,
            handle,
            $"fd={fileDescriptor} low_water={lowWater} user_data=0x{userData:X16}");
        return registered
            ? (int)OrbisGen2Result.ORBIS_GEN2_OK
            : (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND;
    }

    private static int DeleteDescriptorEvent(CpuContext ctx, short filter, string trace)
    {
        var handle = ctx[CpuRegister.Rdi];
        var fileDescriptor = unchecked((int)ctx[CpuRegister.Rsi]);
        var deleted = DeleteRegisteredEvent(handle, unchecked((uint)fileDescriptor), filter);
        TraceEventQueue(ctx, trace, handle, $"fd={fileDescriptor}");
        return deleted
            ? (int)OrbisGen2Result.ORBIS_GEN2_OK
            : (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND;
    }

    // The user event id is an int that the kernel stores in a uintptr_t ident, so -1 becomes 0xFFFFFFFFFFFFFFFF.
    private static ulong UserEventIdent(CpuContext ctx) => unchecked((ulong)(int)ctx[CpuRegister.Rsi]);

    [SysAbiExport(
        Nid = "WDszmSbWuDk",
        ExportName = "sceKernelAddUserEventEdge",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelAddUserEventEdge(CpuContext ctx)
    {
        var handle = ctx[CpuRegister.Rdi];
        var registered = RegisterEvent(
            handle,
            UserEventIdent(ctx),
            KernelEventFilterUser,
            0,
            KernelEventFlagClear);
        TraceEventQueue(ctx, "add_user_edge", handle);
        return registered
            ? (int)OrbisGen2Result.ORBIS_GEN2_OK
            : (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND;
    }

    [SysAbiExport(
        Nid = "4R6-OvI2cEA",
        ExportName = "sceKernelAddUserEvent",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelAddUserEvent(CpuContext ctx)
    {
        var handle = ctx[CpuRegister.Rdi];
        var registered = RegisterEvent(
            handle,
            UserEventIdent(ctx),
            KernelEventFilterUser,
            0,
            flags: 0);
        TraceEventQueue(ctx, "add_user", handle);
        return registered
            ? (int)OrbisGen2Result.ORBIS_GEN2_OK
            : (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND;
    }

    [SysAbiExport(
        Nid = "LJDwdSNTnDg",
        ExportName = "sceKernelDeleteUserEvent",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelDeleteUserEvent(CpuContext ctx)
    {
        var handle = ctx[CpuRegister.Rdi];
        var deleted = DeleteRegisteredEvent(
            handle,
            UserEventIdent(ctx),
            KernelEventFilterUser);
        TraceEventQueue(ctx, "delete_user", handle);
        return deleted
            ? (int)OrbisGen2Result.ORBIS_GEN2_OK
            : (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND;
    }

    [SysAbiExport(
        Nid = "F6e0kwo4cnk",
        ExportName = "sceKernelTriggerUserEvent",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelTriggerUserEvent(CpuContext ctx)
    {
        var handle = ctx[CpuRegister.Rdi];
        var ident = UserEventIdent(ctx);
        var userData = ctx[CpuRegister.Rdx];
        // sceKernelTriggerUserEvent's third argument replaces the event's udata.
        var triggered = TriggerKnote(
            handle,
            ident,
            KernelEventFilterUser,
            static (knote, userData) =>
            {
                knote.UserData = userData;
                return true;
            },
            userData);
        TraceEventQueue(ctx, "trigger_user", handle);
        return triggered
            ? (int)OrbisGen2Result.ORBIS_GEN2_OK
            : (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND;
    }

    [SysAbiExport(
        Nid = "bBfz7kMF2Ho",
        ExportName = "sceKernelAddAmprEvent",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelAddAmprEvent(CpuContext ctx)
    {
        var handle = ctx[CpuRegister.Rdi];
        var registered = RegisterEvent(
            handle,
            unchecked((uint)ctx[CpuRegister.Rsi]),
            KernelEventFilterAmpr,
            ctx[CpuRegister.Rdx]);
        TraceEventQueue(ctx, "add_ampr", handle);
        return registered
            ? (int)OrbisGen2Result.ORBIS_GEN2_OK
            : (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND;
    }

    [SysAbiExport(
        Nid = "vuae5JPNt9A",
        ExportName = "sceKernelAddAmprSystemEvent",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelAddAmprSystemEvent(CpuContext ctx)
    {
        var handle = ctx[CpuRegister.Rdi];
        var registered = RegisterEvent(
            handle,
            unchecked((uint)ctx[CpuRegister.Rsi]),
            KernelEventFilterAmprSystem,
            ctx[CpuRegister.Rdx]);
        TraceEventQueue(ctx, "add_ampr_system", handle);
        return registered
            ? (int)OrbisGen2Result.ORBIS_GEN2_OK
            : (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND;
    }

    [SysAbiExport(
        Nid = "bMmid3pfyjo",
        ExportName = "sceKernelDeleteAmprEvent",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelDeleteAmprEvent(CpuContext ctx)
    {
        var handle = ctx[CpuRegister.Rdi];
        var deleted = DeleteRegisteredEvent(
            handle,
            unchecked((uint)ctx[CpuRegister.Rsi]),
            KernelEventFilterAmpr);
        TraceEventQueue(ctx, "delete_ampr", handle);
        return deleted
            ? (int)OrbisGen2Result.ORBIS_GEN2_OK
            : (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND;
    }

    [SysAbiExport(
        Nid = "Ij+ryuEClXQ",
        ExportName = "sceKernelDeleteAmprSystemEvent",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelDeleteAmprSystemEvent(CpuContext ctx)
    {
        var handle = ctx[CpuRegister.Rdi];
        var deleted = DeleteRegisteredEvent(
            handle,
            unchecked((uint)ctx[CpuRegister.Rsi]),
            KernelEventFilterAmprSystem);
        TraceEventQueue(ctx, "delete_ampr_system", handle);
        return deleted
            ? (int)OrbisGen2Result.ORBIS_GEN2_OK
            : (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND;
    }

    [SysAbiExport(
        Nid = "QyrxcdBrb0M",
        ExportName = "sceKernelGetKqueueFromEqueue",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelGetKqueueFromEqueue(CpuContext ctx)
    {
        ctx[CpuRegister.Rax] = ctx[CpuRegister.Rdi];
        TraceEventQueue(ctx, "get_kqueue", ctx[CpuRegister.Rdi]);
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "vz+pg2zdopI",
        ExportName = "sceKernelGetEventUserData",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelGetEventUserData(CpuContext ctx)
    {
        _ = ctx.TryReadUInt64(ctx[CpuRegister.Rdi] + 0x18, out var userData);
        ctx[CpuRegister.Rax] = userData;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "mJ7aghmgvfc",
        ExportName = "sceKernelGetEventId",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelGetEventId(CpuContext ctx)
    {
        _ = ctx.TryReadUInt64(ctx[CpuRegister.Rdi], out var ident);
        ctx[CpuRegister.Rax] = ident;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "23CPPI1tyBY",
        ExportName = "sceKernelGetEventFilter",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelGetEventFilter(CpuContext ctx)
    {
        Span<byte> filterBytes = stackalloc byte[sizeof(short)];
        var filter = ctx.Memory.TryRead(ctx[CpuRegister.Rdi] + 0x08, filterBytes)
            ? BinaryPrimitives.ReadInt16LittleEndian(filterBytes)
            : (short)0;
        ctx[CpuRegister.Rax] = unchecked((uint)filter);
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "kwGyyjohI50",
        ExportName = "sceKernelGetEventData",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelGetEventData(CpuContext ctx)
    {
        _ = ctx.TryReadUInt64(ctx[CpuRegister.Rdi] + 0x10, out var data);
        ctx[CpuRegister.Rax] = data;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    [SysAbiExport(
        Nid = "Q0qr9AyqJSk",
        ExportName = "sceKernelGetEventFflags",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelGetEventFflags(CpuContext ctx)
    {
        _ = TryReadUInt32(ctx, ctx[CpuRegister.Rdi] + 0x0C, out var fflags);
        ctx[CpuRegister.Rax] = fflags;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    // A failed kevent change is reported as EV_ERROR with the errno in data
    // (kqueue_kevent); every other event has no error.
    [SysAbiExport(
        Nid = "Uu-iDFC9aUc",
        ExportName = "sceKernelGetEventError",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelGetEventError(CpuContext ctx)
    {
        var address = ctx[CpuRegister.Rdi];
        Span<byte> flagBytes = stackalloc byte[sizeof(ushort)];
        var error = 0;
        if (ctx.Memory.TryRead(address + 0x0A, flagBytes) &&
            (BinaryPrimitives.ReadUInt16LittleEndian(flagBytes) & KernelEventFlagError) != 0 &&
            ctx.TryReadUInt64(address + 0x10, out var data))
        {
            error = unchecked((int)data);
        }

        ctx[CpuRegister.Rax] = unchecked((ulong)(long)error);
        return error;
    }

    [SysAbiExport(
        Nid = "fzyMKs9kim0",
        ExportName = "sceKernelWaitEqueue",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int KernelWaitEqueue(CpuContext ctx)
    {
        var handle = ctx[CpuRegister.Rdi];
        var eventsAddress = ctx[CpuRegister.Rsi];
        var eventCapacity = (int)Math.Min(ctx[CpuRegister.Rdx], int.MaxValue);
        var outCountAddress = ctx[CpuRegister.Rcx];
        var timeoutAddress = ctx[CpuRegister.R8];

        if (!TryGetLiveEventQueue(handle, out var state))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND;
        }

        if (eventsAddress == 0 || eventCapacity < 1)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT;
        }

        uint timeoutUsec = 0;
        if (timeoutAddress != 0 && !TryReadUInt32(ctx, timeoutAddress, out timeoutUsec))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        if (Volatile.Read(ref state.DescriptorKnotes) != 0)
        {
            RefreshDescriptorKnotes(state);
        }

        var deliveredCount = DequeueEvents(ctx, state, eventsAddress, eventCapacity);
        if (outCountAddress != 0 && !TryWriteUInt32(ctx, outCountAddress, (uint)deliveredCount))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
        }

        if (deliveredCount > 0)
        {
            if (_logEqueue)
            {
                TraceEventQueue(ctx, "wait-deliver", handle, $"delivered={deliveredCount} capacity={eventCapacity}");
            }
            return (int)OrbisGen2Result.ORBIS_GEN2_OK;
        }

        if (!IsLive(state))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_DELETED;
        }

        if (IsSynchronousPoll(timeoutAddress, timeoutUsec))
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_TIMED_OUT;
        }

        var waiterId = Interlocked.Increment(ref _nextEventQueueWaiterId);
        var deadline = timeoutAddress == 0
            ? 0
            : GuestThreadExecution.ComputeDeadlineTimestamp(TimeSpan.FromMicroseconds(timeoutUsec));
        var waiter = new EqueueWaiter
        {
            Ctx = ctx,
            State = state,
            EventsAddress = eventsAddress,
            EventCapacity = eventCapacity,
            OutCountAddress = outCountAddress,
            WaiterId = waiterId,
        };
        waiter.Arm();
        if (GuestThreadExecution.RequestCurrentThreadBlock(
                ctx,
                "sceKernelWaitEqueue",
                state.WakeKey,
                waiter,
                deadline))
        {
            // The scheduler re-checks the waiter once the thread is parked, so an
            // activation between this point and the park is not lost.
            if (_logEqueue)
            {
                TraceEventQueue(
                    ctx,
                    "wait-block",
                    handle,
                    $"generation={state.Generation} waiter={waiterId} capacity={eventCapacity} timeout=" +
                    (timeoutAddress == 0 ? "infinite" : $"{timeoutUsec}_usec"));
            }
            return (int)OrbisGen2Result.ORBIS_GEN2_OK;
        }

        // A host thread parks on the queue monitor; every activation pulses it.
        waiter.Disarm();
        var outcome = HostWait(state, deadline, eventCapacity, out var events, out var count);
        if (outcome == HostWaitOutcome.Deleted)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_DELETED;
        }

        if (outcome == HostWaitOutcome.TimedOut)
        {
            return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_TIMED_OUT;
        }

        return DeliverReservedEvents(ctx, events!, count, eventsAddress, outCountAddress);
    }

    private enum HostWaitOutcome
    {
        Reserved,
        TimedOut,
        Deleted,
    }

    private static HostWaitOutcome HostWait(
        Kqueue state,
        long deadlineTimestamp,
        int capacity,
        out KernelQueuedEvent[]? events,
        out int count)
    {
        events = null;
        count = 0;
        lock (state.Gate)
        {
            while (true)
            {
                if (state.Deleted)
                {
                    return HostWaitOutcome.Deleted;
                }

                if (state.Active.Count != 0)
                {
                    capacity = Math.Min(capacity, MaxEventsPerWait);
                    events = ArrayPool<KernelQueuedEvent>.Shared.Rent(capacity);
                    count = ReserveLocked(state, events, capacity);
                    if (count != 0)
                    {
                        return HostWaitOutcome.Reserved;
                    }

                    ArrayPool<KernelQueuedEvent>.Shared.Return(events);
                    events = null;
                }

                if (deadlineTimestamp == 0)
                {
                    Monitor.Wait(state.Gate);
                    continue;
                }

                var remaining = deadlineTimestamp - System.Diagnostics.Stopwatch.GetTimestamp();
                if (remaining <= 0)
                {
                    return HostWaitOutcome.TimedOut;
                }

                var milliseconds = Math.Max(1, (long)Math.Ceiling(
                    remaining * 1000.0 / System.Diagnostics.Stopwatch.Frequency));
                Monitor.Wait(state.Gate, (int)Math.Min(milliseconds, int.MaxValue));
            }
        }
    }

    public static bool IsValidEqueue(ulong handle) => TryGetLiveEventQueue(handle, out _);

    private static bool TryGetLiveEventQueue(ulong handle, out Kqueue state)
    {
        lock (_eventQueueGate)
        {
            return _eventQueues.TryGetValue(handle, out state!) && !state.Deleted;
        }
    }

    private static bool IsLive(Kqueue state)
    {
        lock (state.Gate)
        {
            return !state.Deleted;
        }
    }

    private static ulong GetEventQueueRuntimeId(ICpuMemory memory)
    {
        object key = memory;
        while (key is ICpuMemoryWrapper wrapper &&
               !ReferenceEquals(wrapper.Inner, key))
        {
            key = wrapper.Inner;
        }

        return _eventQueueRuntimeIdentities.GetValue(
            key,
            static _ => new EventQueueRuntimeIdentity(
                unchecked((ulong)Interlocked.Increment(
                    ref _nextEventQueueRuntimeId)))).Id;
    }

    internal static bool IsSynchronousPoll(
        ulong timeoutAddress,
        uint timeoutUsec) =>
        timeoutAddress != 0 && timeoutUsec == 0;

    // ---- Knote state machine (callers hold Kqueue.Gate) ----

    /// <summary>
    /// The single trigger path: marks the knote active and queues it unless it is
    /// disabled or already queued, then pulses host waiters. Callers wake guest
    /// waiters with <see cref="Signal"/> after leaving the queue lock.
    /// </summary>
    private static void ActivateLocked(Kqueue state, Knote knote)
    {
        if (knote.Dropped)
        {
            return;
        }

        knote.Triggered = true;
        if (!knote.Disabled && !knote.Queued)
        {
            state.Active.AddLast(knote);
            knote.Queued = true;
        }

        Monitor.PulseAll(state.Gate);
    }

    private static void DeactivateLocked(Kqueue state, Knote knote)
    {
        knote.Triggered = false;
        if (knote.Queued)
        {
            state.Active.Remove(knote);
            knote.Queued = false;
        }
    }

    private static void ResetPendingLocked(Knote knote)
    {
        knote.Triggered = false;
        knote.Data = 0;
        knote.Fflags = 0;
        knote.StatusFlags = 0;
        knote.Occurrences?.Clear();
    }

    // Removes the knote from its queue. Sources (timers, watches, descriptor
    // arming) are released by DetachKnoteSources outside the queue lock.
    private static void DropKnoteLocked(Kqueue state, Knote knote)
    {
        if (knote.Dropped)
        {
            return;
        }

        knote.Dropped = true;
        if (knote.Queued)
        {
            state.Active.Remove(knote);
            knote.Queued = false;
        }

        if (state.Knotes.TryGetValue((knote.Ident, knote.Filter), out var current) &&
            ReferenceEquals(current, knote))
        {
            state.Knotes.Remove((knote.Ident, knote.Filter));
            if (knote.Filter is KernelEventFilterRead or KernelEventFilterWrite)
            {
                state.DescriptorKnotes--;
            }
        }
    }

    private static void DetachKnoteSources(Knote knote)
    {
        if (Interlocked.Exchange(ref knote.TimerToken, null) is IDisposable timer)
        {
            timer.Dispose();
        }

        Interlocked.Exchange(ref knote.Watch, null)?.Dispose();
        if (knote.Filter is KernelEventFilterRead or KernelEventFilterWrite or KernelEventFilterVnode)
        {
            UnindexDescriptorKnote(knote);
        }
    }

    private static void SetEnabledLocked(Kqueue state, Knote knote, bool enabled)
    {
        if (!enabled)
        {
            knote.Disabled = true;
            if (knote.Queued)
            {
                state.Active.Remove(knote);
                knote.Queued = false;
            }

            return;
        }

        knote.Disabled = false;
        if (knote.Triggered && !knote.Queued)
        {
            state.Active.AddLast(knote);
            knote.Queued = true;
            Monitor.PulseAll(state.Gate);
        }
    }

    /// <summary>
    /// kqueue_scan: reports each active knote at most once per call, in trigger
    /// order, then applies EV_ONESHOT (drop), EV_DISPATCH (disable), EV_CLEAR
    /// (reset) or level semantics (requeue at the tail).
    /// </summary>
    private static int ReserveLocked(Kqueue state, Span<KernelQueuedEvent> events, int capacity)
    {
        var count = 0;
        var budget = state.Active.Count;
        List<Knote>? detached = null;
        while (count < capacity && budget-- > 0 && state.Active.Head is { } knote)
        {
            state.Active.Remove(knote);
            knote.Queued = false;
            if (knote.Dropped || knote.Disabled || !knote.Triggered)
            {
                continue;
            }

            var flags = (ushort)((knote.Flags & PersistentFlagMask) | knote.StatusFlags);
            if (knote.Transient)
            {
                events[count++] = knote.Injected;
                DropKnoteLocked(state, knote);
                continue;
            }

            if (knote.Occurrences is { Count: > 0 } ring)
            {
                while (count < capacity && ring.Count > 0)
                {
                    var occurrence = ring.Pop();
                    events[count++] = new KernelQueuedEvent(
                        knote.Ident,
                        knote.Filter,
                        flags,
                        occurrence.Count,
                        occurrence.Data,
                        knote.UserData);
                }

                if (ring.Count > 0)
                {
                    state.Active.AddLast(knote);
                    knote.Queued = true;
                    continue;
                }
            }
            else
            {
                events[count++] = new KernelQueuedEvent(
                    knote.Ident,
                    knote.Filter,
                    flags,
                    knote.Fflags,
                    knote.Data,
                    knote.UserData);
            }

            if ((knote.Flags & KernelEventFlagOneShot) != 0)
            {
                DropKnoteLocked(state, knote);
                (detached ??= []).Add(knote);
                continue;
            }

            if ((knote.Flags & KernelEventFlagDispatch) != 0)
            {
                knote.Disabled = true;
            }

            if ((knote.Flags & KernelEventFlagClear) != 0)
            {
                ResetPendingLocked(knote);
                OnClearedLocked(state, knote);
                continue;
            }

            if (!knote.Disabled)
            {
                state.Active.AddLast(knote);
                knote.Queued = true;
            }
        }

        if (detached is not null)
        {
            // Releasing a timer only flags it cancelled; no lock is taken.
            foreach (var knote in detached)
            {
                if (Interlocked.Exchange(ref knote.TimerToken, null) is IDisposable timer)
                {
                    timer.Dispose();
                }
            }
        }

        return count;
    }

    private static bool TryReserveEvents(
        Kqueue state,
        int eventCapacity,
        out KernelQueuedEvent[] events,
        out int count,
        out bool deleted)
    {
        events = null!;
        count = 0;
        deleted = false;
        lock (state.Gate)
        {
            if (state.Deleted)
            {
                deleted = true;
                return false;
            }

            if (state.Active.Count == 0)
            {
                return false;
            }

            var capacity = Math.Min(eventCapacity, MaxEventsPerWait);
            var rented = ArrayPool<KernelQueuedEvent>.Shared.Rent(capacity);
            count = ReserveLocked(state, rented, capacity);
            if (count == 0)
            {
                ArrayPool<KernelQueuedEvent>.Shared.Return(rented);
                return false;
            }

            events = rented;
            return true;
        }
    }

    /// <summary>Wakes every waiter of the queue: guest threads by wake key.</summary>
    /// <remarks>
    /// The scheduler scan visits every guest thread under its gate, so it is skipped when no guest
    /// waiter is registered. That is a store-buffering (Dekker) pair and cannot lose a wake:
    /// <list type="bullet">
    /// <item>Waiter: <c>BlockedWaiters++</c> (Interlocked, a full fence) happens before it asks to
    /// block, and the scheduler re-checks <see cref="EqueueWaiter.TryWake"/> after parking the thread,
    /// which reads the queue under <c>Gate</c>.</item>
    /// <item>Signaller: the event is made visible under <c>Gate</c> before this call, then the full
    /// fence below, then the read of <c>BlockedWaiters</c>.</item>
    /// </list>
    /// If this read sees zero, it precedes the waiter's increment in the fence order, so the event
    /// was already published when the waiter's re-check took <c>Gate</c> and the re-check finds it
    /// (the waiter is readied directly, no wake needed). If it sees non-zero, the scan runs as before.
    /// A stale non-zero count only costs an unneeded scan.
    /// </remarks>
    private static void Signal(Kqueue state, string? detail = null)
    {
        if (_logEqueue)
        {
            TraceEventQueueHost(
                "wake",
                state.Handle,
                $"generation={state.Generation}" + (detail is null ? string.Empty : $" {detail}"));
        }

        Interlocked.MemoryBarrier();
        if (Volatile.Read(ref state.BlockedWaiters) == 0)
        {
            return;
        }

        _ = GuestThreadExecution.Scheduler?.WakeBlockedThreads(state.WakeKey);
    }

    internal static int BlockedWaiterCountForTest(ulong handle) =>
        TryGetLiveEventQueue(handle, out var state) ? Volatile.Read(ref state.BlockedWaiters) : -1;

    // ---- Registration API used by the other libraries ----

    public static bool RegisterEvent(
        ulong handle,
        ulong ident,
        short filter,
        ulong userData,
        ushort flags = KernelEventFlagClear,
        ulong lowWater = 0)
    {
        if (!TryGetLiveEventQueue(handle, out var state))
        {
            return false;
        }

        Knote knote;
        lock (state.Gate)
        {
            if (state.Deleted)
            {
                return false;
            }

            knote = AttachOrUpdateLocked(state, ident, filter, userData, flags, sfflags: 0, lowWater, out _);
        }

        if (filter is KernelEventFilterRead or KernelEventFilterWrite)
        {
            IndexDescriptorKnote(knote);
            RefreshDescriptorKnote(knote);
        }

        return true;
    }

    // kqueue_register for EV_ADD: a new knote takes the persistent flags; an
    // existing one keeps its flags and pending state and takes the new udata,
    // sfflags and sdata.
    private static Knote AttachOrUpdateLocked(
        Kqueue state,
        ulong ident,
        short filter,
        ulong userData,
        ushort flags,
        uint sfflags,
        ulong sdata,
        out bool created)
    {
        var key = (ident, filter);
        if (state.Knotes.TryGetValue(key, out var knote) && !knote.Transient)
        {
            knote.UserData = userData;
            knote.SFflags = sfflags;
            knote.SData = sdata;
            created = false;
            return knote;
        }

        if (knote is not null)
        {
            DropKnoteLocked(state, knote);
        }

        knote = new Knote(state, ident, filter)
        {
            Flags = (ushort)(flags & PersistentFlagMask),
            UserData = userData,
            SFflags = sfflags,
            SData = sdata,
        };
        if (filter == KernelEventFilterGraphics)
        {
            knote.Occurrences = new OccurrenceRing();
        }

        state.Knotes[key] = knote;
        if (filter is KernelEventFilterRead or KernelEventFilterWrite)
        {
            state.DescriptorKnotes++;
        }

        created = true;
        return knote;
    }

    public static bool DeleteRegisteredEvent(
        ulong handle,
        ulong ident,
        short filter)
    {
        if (!TryGetLiveEventQueue(handle, out var state))
        {
            return false;
        }

        Knote? knote;
        lock (state.Gate)
        {
            if (!state.Knotes.TryGetValue((ident, filter), out knote))
            {
                return false;
            }

            DropKnoteLocked(state, knote);
        }

        DetachKnoteSources(knote);
        return true;
    }

    /// <summary>
    /// Applies <paramref name="update"/> to the knote (ident, filter) of one queue
    /// and activates it when the update reports an occurrence.
    /// </summary>
    private static bool TriggerKnote<TArg>(
        ulong handle,
        ulong ident,
        short filter,
        Func<Knote, TArg, bool> update,
        TArg argument)
    {
        if (!TryGetLiveEventQueue(handle, out var state))
        {
            return false;
        }

        lock (state.Gate)
        {
            if (state.Deleted ||
                !state.Knotes.TryGetValue((ident, filter), out var knote) ||
                knote.Transient)
            {
                return false;
            }

            if (update(knote, argument))
            {
                ActivateLocked(state, knote);
            }
        }

        Signal(state, _logEqueue ? $"source=trigger ident=0x{ident:X16} filter={filter}" : null);
        return true;
    }

    // Next-state function shared by the interrupt-style filters: graphics keeps
    // each occurrence, the others keep the newest data (AMPR, display hints).
    private static void ApplyDataTriggerLocked(Knote knote, ulong data)
    {
        if (knote.Occurrences is { } ring)
        {
            ring.Push(data);
            return;
        }

        knote.Data = data;
    }

    /// <summary>
    /// Injects a raw event. A registered knote for (ident, filter) takes its data,
    /// fflags and udata; otherwise a transient knote delivers the event exactly
    /// once. A second injection before delivery replaces the first.
    /// </summary>
    public static bool EnqueueEvent(ulong handle, KernelQueuedEvent queuedEvent)
    {
        if (!TryGetLiveEventQueue(handle, out var state))
        {
            return false;
        }

        lock (state.Gate)
        {
            if (state.Deleted)
            {
                return false;
            }

            var key = (queuedEvent.Ident, queuedEvent.Filter);
            if (!state.Knotes.TryGetValue(key, out var knote) || knote.Transient)
            {
                if (knote is null)
                {
                    knote = new Knote(state, queuedEvent.Ident, queuedEvent.Filter) { Transient = true };
                    state.Knotes[key] = knote;
                    if (queuedEvent.Filter is KernelEventFilterRead or KernelEventFilterWrite)
                    {
                        state.DescriptorKnotes++;
                    }
                }

                knote.Injected = queuedEvent;
            }
            else
            {
                ApplyDataTriggerLocked(knote, queuedEvent.Data);
                knote.Fflags = queuedEvent.Fflags;
                knote.UserData = queuedEvent.UserData;
            }

            ActivateLocked(state, knote);
        }

        Signal(
            state,
            _logEqueue
                ? $"source=enqueue ident=0x{queuedEvent.Ident:X16} filter={queuedEvent.Filter} data=0x{queuedEvent.Data:X16}"
                : null);
        return true;
    }

    /// <summary>
    /// Completes an AMPR kernel-event command. Only a knote registered with
    /// sceKernelAddAmprEvent receives it: it keeps its registered udata and takes
    /// the command's data. An unregistered (queue, id) is reported to the caller,
    /// which fails the command with NOT_FOUND.
    /// </summary>
    internal static bool TriggerAmprEvent(ulong handle, ulong ident, ulong data) =>
        TriggerKnote(
            handle,
            ident,
            KernelEventFilterAmpr,
            static (knote, data) =>
            {
                knote.Data = data;
                return true;
            },
            data);

    /// <summary>
    /// Captures the exact lifetime of every matching registration owned by one
    /// guest runtime. A later delete/re-add of the same tuple receives a new
    /// generation and therefore cannot consume an interrupt that was already
    /// bound to this snapshot.
    /// </summary>
    internal static KernelEventRegistrationSnapshot CaptureRegisteredEvents(
        ICpuMemory memory,
        ulong ident,
        short filter)
    {
        ArgumentNullException.ThrowIfNull(memory);
        var runtimeId = GetEventQueueRuntimeId(memory);
        List<KernelEventRegistrationToken>? targets = null;
        foreach (var state in Volatile.Read(ref _eventQueueSnapshot))
        {
            if (state.RuntimeId != runtimeId)
            {
                continue;
            }

            lock (state.Gate)
            {
                if (state.Deleted ||
                    !state.Knotes.TryGetValue((ident, filter), out var knote) ||
                    knote.Transient)
                {
                    continue;
                }

                (targets ??= []).Add(new KernelEventRegistrationToken(
                    state.Handle,
                    state.Generation,
                    knote.Ident,
                    knote.Filter,
                    knote.Generation));
            }
        }

        return new KernelEventRegistrationSnapshot(runtimeId, ident, filter, targets?.ToArray() ?? []);
    }

    /// <summary>
    /// Delivers a previously captured interrupt only to registrations whose
    /// runtime, queue handle, and generation are still live.
    /// </summary>
    internal static CapturedEventDeliveryResult TriggerCapturedEvents(
        KernelEventRegistrationSnapshot snapshot,
        ulong data)
    {
        ArgumentNullException.ThrowIfNull(snapshot);
        var triggeredCount = 0;
        var staleCount = 0;
        foreach (var target in snapshot.Targets)
        {
            Kqueue? state;
            lock (_eventQueueGate)
            {
                _ = _eventQueues.TryGetValue(target.EqueueHandle, out state);
            }

            var triggered = false;
            if (state is not null &&
                state.Generation == target.EqueueGeneration &&
                state.RuntimeId == snapshot.RuntimeId)
            {
                lock (state.Gate)
                {
                    if (!state.Deleted &&
                        state.Knotes.TryGetValue((target.Ident, target.Filter), out var knote) &&
                        knote.Generation == target.Generation)
                    {
                        ApplyDataTriggerLocked(knote, data);
                        ActivateLocked(state, knote);
                        triggered = true;
                    }
                }
            }

            if (!triggered)
            {
                staleCount++;
                continue;
            }

            triggeredCount++;
            Signal(
                state!,
                _logEqueue
                    ? $"source=trigger-captured ident=0x{snapshot.Ident:X16} filter={snapshot.Filter} data=0x{data:X16}"
                    : null);
        }

        return new CapturedEventDeliveryResult(triggeredCount, staleCount);
    }

    /// <summary>
    /// Raises one occurrence of (ident, filter) on every queue that registered
    /// it (the AGC end-of-pipe interrupt). Each occurrence keeps its own data;
    /// see <see cref="MaxPendingOccurrencesPerKnote"/> for the bound.
    /// </summary>
    public static int TriggerRegisteredEventsByFilter(
        short filter,
        ulong data,
        ulong ident)
    {
        var triggeredCount = 0;
        foreach (var state in Volatile.Read(ref _eventQueueSnapshot))
        {
            lock (state.Gate)
            {
                if (state.Deleted ||
                    !state.Knotes.TryGetValue((ident, filter), out var knote) ||
                    knote.Transient)
                {
                    continue;
                }

                ApplyDataTriggerLocked(knote, data);
                ActivateLocked(state, knote);
            }

            triggeredCount++;
            Signal(state, _logEqueue ? $"source=trigger-filter filter={filter} data=0x{data:X16}" : null);
        }

        return triggeredCount;
    }

    /// <summary>
    /// Raises a VideoOut event. Data packs the low timestamp bits (0-11), the
    /// number of occurrences since the last delivery (12-15, saturating at 15)
    /// and the payload hint (16-63). A queue without a knote for the event gets
    /// one attached with EV_CLEAR, so VideoOut registrations made before the
    /// queue learned of them still deliver.
    /// </summary>
    public static bool TriggerDisplayEvent(
        ulong handle,
        ulong ident,
        short filter,
        ulong eventHint,
        ulong userData)
    {
        if (!TryGetLiveEventQueue(handle, out var state))
        {
            return false;
        }

        lock (state.Gate)
        {
            if (state.Deleted)
            {
                return false;
            }

            if (!state.Knotes.TryGetValue((ident, filter), out var knote) || knote.Transient)
            {
                knote = AttachOrUpdateLocked(state, ident, filter, userData, KernelEventFlagClear, 0, 0, out _);
            }

            var count = knote.Triggered
                ? Math.Min(((knote.Data >> 12) & 0xFUL) + 1, 0xFUL)
                : 1UL;
            var timeBits = unchecked((ulong)Environment.TickCount64) & 0xFFFUL;
            knote.Data = timeBits | (count << 12) | (eventHint & 0xFFFF_FFFF_FFFF_0000UL);
            knote.UserData = userData;
            ActivateLocked(state, knote);
        }

        Signal(
            state,
            _logEqueue ? $"source=display ident=0x{ident:X16} filter={filter} hint=0x{eventHint:X16}" : null);
        return true;
    }

    private static int DeliverReservedEvents(
        CpuContext ctx,
        KernelQueuedEvent[] events,
        int count,
        ulong eventsAddress,
        ulong outCountAddress)
    {
        var deliveredCount = 0;
        try
        {
            for (; deliveredCount < count; deliveredCount++)
            {
                if (!WriteKernelEvent(
                        ctx,
                        eventsAddress + ((ulong)deliveredCount * KernelEventSize),
                        events[deliveredCount]))
                {
                    return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
                }
            }

            if (outCountAddress != 0 &&
                !TryWriteUInt32(ctx, outCountAddress, (uint)deliveredCount))
            {
                return (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT;
            }

            return deliveredCount > 0
                ? (int)OrbisGen2Result.ORBIS_GEN2_OK
                : (int)OrbisGen2Result.ORBIS_GEN2_ERROR_TIMED_OUT;
        }
        finally
        {
            ArrayPool<KernelQueuedEvent>.Shared.Return(events);
        }
    }

    private static int DequeueEvents(
        CpuContext ctx,
        Kqueue state,
        ulong eventsAddress,
        int eventCapacity)
    {
        if (eventsAddress == 0 || eventCapacity <= 0 ||
            !TryReserveEvents(state, eventCapacity, out var events, out var count, out _))
        {
            return 0;
        }

        var deliveredCount = 0;
        try
        {
            for (; deliveredCount < count; deliveredCount++)
            {
                if (!WriteKernelEvent(
                        ctx,
                        eventsAddress + ((ulong)deliveredCount * KernelEventSize),
                        events[deliveredCount]))
                {
                    break;
                }
            }
        }
        finally
        {
            ArrayPool<KernelQueuedEvent>.Shared.Return(events);
        }

        return deliveredCount;
    }

    internal static int ReservePendingEventCountForTest(ulong handle, int eventCapacity)
    {
        if (!TryGetLiveEventQueue(handle, out var state) ||
            !TryReserveEvents(state, eventCapacity, out var events, out var count, out _))
        {
            return 0;
        }

        ArrayPool<KernelQueuedEvent>.Shared.Return(events);
        return count;
    }

    internal static bool TryReservePendingEventForTest(
        ulong handle,
        out KernelQueuedEvent queuedEvent)
    {
        queuedEvent = default;
        if (!TryGetLiveEventQueue(handle, out var state) ||
            !TryReserveEvents(state, 1, out var events, out var count, out _))
        {
            return false;
        }

        queuedEvent = events[0];
        ArrayPool<KernelQueuedEvent>.Shared.Return(events);
        return count == 1;
    }

    private static bool WriteKernelEvent(CpuContext ctx, ulong address, KernelQueuedEvent queuedEvent)
    {
        Span<byte> eventBytes = stackalloc byte[KernelEventSize];
        BinaryPrimitives.WriteUInt64LittleEndian(eventBytes[0x00..], queuedEvent.Ident);
        BinaryPrimitives.WriteInt16LittleEndian(eventBytes[0x08..], queuedEvent.Filter);
        BinaryPrimitives.WriteUInt16LittleEndian(eventBytes[0x0A..], queuedEvent.Flags);
        BinaryPrimitives.WriteUInt32LittleEndian(eventBytes[0x0C..], queuedEvent.Fflags);
        BinaryPrimitives.WriteUInt64LittleEndian(eventBytes[0x10..], queuedEvent.Data);
        BinaryPrimitives.WriteUInt64LittleEndian(eventBytes[0x18..], queuedEvent.UserData);
        return ctx.Memory.TryWrite(address, eventBytes);
    }

    private static uint SaturatingAdd(uint left, uint right) =>
        left > uint.MaxValue - right ? uint.MaxValue : left + right;

    private static readonly bool _logEqueue =
        string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_LOG_EQUEUE"), "1", StringComparison.Ordinal);

    private static void TraceEventQueue(
        CpuContext ctx,
        string operation,
        ulong handle,
        string? detail = null)
    {
        if (!_logEqueue)
        {
            return;
        }

        var suffix = string.IsNullOrWhiteSpace(detail) ? string.Empty : $" {detail}";
        Console.Error.WriteLine(
            $"[LOADER][TRACE] equeue.{operation}: handle=0x{handle:X16} " +
            $"{DescribeQueue(handle)}{suffix} {KernelSyncTraceFormatter.FormatContext(ctx)}");
    }

    private static void TraceEventQueueHost(
        string operation,
        ulong handle,
        string? detail = null)
    {
        var suffix = string.IsNullOrWhiteSpace(detail) ? string.Empty : $" {detail}";
        Console.Error.WriteLine(
            $"[LOADER][TRACE] equeue.{operation}: handle=0x{handle:X16} {DescribeQueue(handle)} " +
            $"{KernelSyncTraceFormatter.FormatCurrentThread()}{suffix}");
    }

    private static string DescribeQueue(ulong handle)
    {
        if (!TryGetLiveEventQueue(handle, out var state))
        {
            return "depth=0 registrations=0";
        }

        lock (state.Gate)
        {
            return $"depth={state.Active.Count} registrations={state.Knotes.Count}";
        }
    }

    private static bool TryWriteUInt32(CpuContext ctx, ulong address, uint value)
    {
        Span<byte> buffer = stackalloc byte[sizeof(uint)];
        BinaryPrimitives.WriteUInt32LittleEndian(buffer, value);
        return ctx.Memory.TryWrite(address, buffer);
    }

    private static bool TryReadUInt32(CpuContext ctx, ulong address, out uint value)
    {
        Span<byte> buffer = stackalloc byte[sizeof(uint)];
        if (!ctx.Memory.TryRead(address, buffer))
        {
            value = 0;
            return false;
        }

        value = BinaryPrimitives.ReadUInt32LittleEndian(buffer);
        return true;
    }
}
