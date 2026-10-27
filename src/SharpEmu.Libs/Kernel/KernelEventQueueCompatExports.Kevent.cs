// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using System.Buffers;
using System.Buffers.Binary;

namespace SharpEmu.Libs.Kernel;

/// <summary>
/// The POSIX kqueue(2)/kevent(2) surface over the same knote core:
/// kqueue_register's change processing (EV_ADD/DELETE/ENABLE/DISABLE,
/// EV_RECEIPT, EV_FORCEONESHOT, EV_ERROR reporting) followed by kqueue_scan.
/// </summary>
public static partial class KernelEventQueueCompatExports
{
    private const int PosixEnoent = 2;
    private const int PosixEbadf = 9;
    private const int PosixEfault = 14;
    private const int PosixEinval = 22;

    public const uint NoteFfnop = 0x00000000;
    public const uint NoteFfand = 0x40000000;
    public const uint NoteFfor = 0x80000000;
    public const uint NoteFfcopy = 0xC0000000;
    public const uint NoteFfctrlMask = 0xC0000000;
    public const uint NoteFflagsMask = 0x00FFFFFF;
    public const uint NoteTrigger = 0x01000000;
    public const uint NoteLowat = 0x0001;

    private static readonly Dictionary<int, Kqueue> _kqueueDescriptors = new();

    [SysAbiExport(
        Nid = "nh2IFMgKTv8",
        ExportName = "kqueue",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixKqueue(CpuContext ctx)
    {
        var fd = KernelMemoryCompatExports.AllocateGuestFileDescriptor();
        var state = CreateQueue(ctx.Memory, fd);
        lock (_eventQueueGate)
        {
            _kqueueDescriptors[fd] = state;
        }

        TraceEventQueue(ctx, "kqueue", state.Handle, $"fd={fd}");
        ctx[CpuRegister.Rax] = unchecked((ulong)fd);
        return fd;
    }

    /// <summary>close(2) of a kqueue descriptor destroys the queue.</summary>
    internal static bool TryCloseKqueueDescriptor(int fd)
    {
        Kqueue? state;
        lock (_eventQueueGate)
        {
            if (!_kqueueDescriptors.Remove(fd, out state))
            {
                return false;
            }
        }

        _ = DestroyQueue(state.Handle);
        return true;
    }

    private static bool TryGetKqueueDescriptor(int fd, out Kqueue state)
    {
        lock (_eventQueueGate)
        {
            return _kqueueDescriptors.TryGetValue(fd, out state!) && !state.Deleted;
        }
    }

    [SysAbiExport(
        Nid = "RW-GEfpnsqg",
        ExportName = "kevent",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int PosixKevent(CpuContext ctx)
    {
        var fd = unchecked((int)ctx[CpuRegister.Rdi]);
        var changeList = ctx[CpuRegister.Rsi];
        var changeCount = unchecked((int)ctx[CpuRegister.Rdx]);
        var eventList = ctx[CpuRegister.Rcx];
        var eventCapacity = unchecked((int)ctx[CpuRegister.R8]);
        var timeoutAddress = ctx[CpuRegister.R9];

        if (!TryGetKqueueDescriptor(fd, out var state))
        {
            return PosixFail(ctx, PosixEbadf);
        }

        if (changeCount < 0 || eventCapacity < 0)
        {
            return PosixFail(ctx, PosixEinval);
        }

        long deadline = 0;
        var poll = false;
        if (timeoutAddress != 0)
        {
            if (!ctx.TryReadUInt64(timeoutAddress, out var secondsBits) ||
                !ctx.TryReadUInt64(timeoutAddress + 8, out var nanosecondsBits))
            {
                return PosixFail(ctx, PosixEfault);
            }

            var seconds = unchecked((long)secondsBits);
            var nanoseconds = unchecked((long)nanosecondsBits);
            if (seconds < 0 || nanoseconds < 0 || nanoseconds >= 1_000_000_000)
            {
                return PosixFail(ctx, PosixEinval);
            }

            poll = seconds == 0 && nanoseconds == 0;
            if (!poll)
            {
                var ticks = seconds > long.MaxValue / 4 / System.Diagnostics.Stopwatch.Frequency
                    ? long.MaxValue / 4
                    : seconds * System.Diagnostics.Stopwatch.Frequency + NanosecondsToTicks(nanoseconds);
                deadline = System.Diagnostics.Stopwatch.GetTimestamp() + ticks;
            }
        }

        var remainingCapacity = eventCapacity;
        var errors = 0;
        Span<byte> raw = stackalloc byte[KernelEventSize];
        for (var index = 0; index < changeCount; index++)
        {
            var address = changeList + ((ulong)index * KernelEventSize);
            if (!ctx.Memory.TryRead(address, raw))
            {
                return PosixFail(ctx, PosixEfault);
            }

            var change = ReadKernelEvent(raw);
            var error = RegisterChange(state, change);
            if (error == 0 && (change.Flags & KernelEventFlagReceipt) == 0)
            {
                continue;
            }

            if (remainingCapacity == 0)
            {
                return PosixFail(ctx, error);
            }

            var report = change with
            {
                Flags = (ushort)(change.Flags | KernelEventFlagError),
                Data = unchecked((ulong)error),
            };
            if (!WriteKernelEvent(ctx, eventList + ((ulong)errors * KernelEventSize), report))
            {
                return PosixFail(ctx, PosixEfault);
            }

            remainingCapacity--;
            errors++;
        }

        if (errors > 0)
        {
            ctx[CpuRegister.Rax] = unchecked((ulong)errors);
            return errors;
        }

        if (eventCapacity == 0)
        {
            ctx[CpuRegister.Rax] = 0;
            return 0;
        }

        if (eventList == 0)
        {
            return PosixFail(ctx, PosixEfault);
        }

        if (Volatile.Read(ref state.DescriptorKnotes) != 0)
        {
            RefreshDescriptorKnotes(state);
        }

        var delivered = DequeueEvents(ctx, state, eventList, eventCapacity);
        if (delivered > 0 || poll)
        {
            ctx[CpuRegister.Rax] = unchecked((ulong)delivered);
            return delivered;
        }

        var waiter = new EqueueWaiter
        {
            Ctx = ctx,
            State = state,
            EventsAddress = eventList,
            EventCapacity = eventCapacity,
            OutCountAddress = 0,
            WaiterId = Interlocked.Increment(ref _nextEventQueueWaiterId),
            PosixResult = true,
        };
        waiter.Arm();
        if (GuestThreadExecution.RequestCurrentThreadBlock(
                ctx,
                "kevent",
                state.WakeKey,
                waiter,
                deadline))
        {
            return 0;
        }

        waiter.Disarm();
        var outcome = HostWait(state, deadline, eventCapacity, out var events, out var count);
        return outcome switch
        {
            HostWaitOutcome.Reserved => DeliverReservedEventsPosix(ctx, events!, count, eventList),
            HostWaitOutcome.Deleted => PosixFail(ctx, PosixEbadf),
            _ => SetPosixCount(ctx, 0),
        };
    }

    private static int SetPosixCount(CpuContext ctx, int count)
    {
        ctx[CpuRegister.Rax] = unchecked((ulong)count);
        return count;
    }

    private static int PosixFail(CpuContext ctx, int errno)
    {
        KernelRuntimeCompatExports.TrySetErrno(ctx, errno);
        ctx[CpuRegister.Rax] = ulong.MaxValue;
        return -1;
    }

    private static int DeliverReservedEventsPosix(
        CpuContext ctx,
        KernelQueuedEvent[] events,
        int count,
        ulong eventsAddress)
    {
        try
        {
            for (var index = 0; index < count; index++)
            {
                if (!WriteKernelEvent(ctx, eventsAddress + ((ulong)index * KernelEventSize), events[index]))
                {
                    return PosixFail(ctx, PosixEfault);
                }
            }

            return SetPosixCount(ctx, count);
        }
        finally
        {
            ArrayPool<KernelQueuedEvent>.Shared.Return(events);
        }
    }

    private static KernelQueuedEvent ReadKernelEvent(ReadOnlySpan<byte> raw) =>
        new(
            BinaryPrimitives.ReadUInt64LittleEndian(raw),
            BinaryPrimitives.ReadInt16LittleEndian(raw[0x08..]),
            BinaryPrimitives.ReadUInt16LittleEndian(raw[0x0A..]),
            BinaryPrimitives.ReadUInt32LittleEndian(raw[0x0C..]),
            BinaryPrimitives.ReadUInt64LittleEndian(raw[0x10..]),
            BinaryPrimitives.ReadUInt64LittleEndian(raw[0x18..]));

    private static bool IsKnownFilter(short filter) =>
        filter is KernelEventFilterRead or KernelEventFilterWrite or KernelEventFilterVnode or
            KernelEventFilterTimer or KernelEventFilterUser or KernelEventFilterDisplay or
            KernelEventFilterGraphics or KernelEventFilterHrTimer or KernelEventFilterAmpr or
            KernelEventFilterAmprSystem;

    /// <summary>
    /// Applies one changelist entry to the queue (kqueue_register). Returns 0 or
    /// a FreeBSD errno.
    /// </summary>
    private static int RegisterChange(Kqueue state, KernelQueuedEvent change)
    {
        var flags = change.Flags;
        if (!IsKnownFilter(change.Filter))
        {
            return PosixEinval;
        }

        if ((flags & (KernelEventFlagEnable | KernelEventFlagDisable)) ==
            (KernelEventFlagEnable | KernelEventFlagDisable))
        {
            return PosixEinval;
        }

        var isDescriptor = change.Filter is KernelEventFilterRead or KernelEventFilterWrite or KernelEventFilterVnode;
        var fd = unchecked((int)change.Ident);
        if (isDescriptor && (change.Ident > int.MaxValue ||
                             !TryGetDescriptorState(fd, change.Filter == KernelEventFilterVnode ? KernelEventFilterRead : change.Filter, 1, out _, out _, out _)))
        {
            return PosixEbadf;
        }

        Knote? knote;
        lock (state.Gate)
        {
            if (state.Deleted)
            {
                return PosixEbadf;
            }

            _ = state.Knotes.TryGetValue((change.Ident, change.Filter), out knote);
            if (knote is { Transient: true })
            {
                knote = null;
            }
        }

        if (knote is null && (flags & KernelEventFlagAdd) == 0)
        {
            return PosixEnoent;
        }

        if (knote is not null && (flags & KernelEventFlagDelete) != 0)
        {
            lock (state.Gate)
            {
                DropKnoteLocked(state, knote);
            }

            DetachKnoteSources(knote);
            return 0;
        }

        if ((flags & KernelEventFlagAdd) != 0)
        {
            var addFlags = (ushort)(flags & ~KernelEventFlagForceOneShot);
            switch (change.Filter)
            {
                case KernelEventFilterRead or KernelEventFilterWrite:
                {
                    var lowWater = (change.Fflags & NoteLowat) != 0 ? Math.Max(1UL, change.Data) : 1UL;
                    lock (state.Gate)
                    {
                        knote = AttachOrUpdateLocked(
                            state, change.Ident, change.Filter, change.UserData, addFlags,
                            change.Fflags, lowWater, out _);
                    }

                    IndexDescriptorKnote(knote);
                    RefreshDescriptorKnote(knote);
                    break;
                }

                case KernelEventFilterVnode:
                {
                    var result = AddVnodeKnote(state.Handle, fd, change.Fflags, change.UserData, addFlags);
                    if (result != (int)OrbisGen2Result.ORBIS_GEN2_OK)
                    {
                        return PosixEbadf;
                    }

                    lock (state.Gate)
                    {
                        _ = state.Knotes.TryGetValue((change.Ident, change.Filter), out knote);
                    }

                    break;
                }

                case KernelEventFilterTimer:
                {
                    var periodicNanoseconds = TimerDataToNanoseconds(change.Data, change.Fflags);
                    if (!AddTimerKnote(
                            state.Handle,
                            change.Ident,
                            KernelEventFilterTimer,
                            change.UserData,
                            (ushort)(addFlags | KernelEventFlagClear),
                            NanosecondsToTicks(periodicNanoseconds),
                            periodic: true,
                            change.Fflags,
                            change.Data))
                    {
                        return PosixEbadf;
                    }

                    lock (state.Gate)
                    {
                        _ = state.Knotes.TryGetValue((change.Ident, change.Filter), out knote);
                    }

                    break;
                }

                case KernelEventFilterUser:
                    lock (state.Gate)
                    {
                        knote = AttachOrUpdateLocked(
                            state, change.Ident, change.Filter, change.UserData, addFlags,
                            knote?.SFflags ?? 0, change.Data, out var created);
                        TouchUserKnoteLocked(state, knote, change.Fflags, change.Data, created);
                    }

                    Signal(state);
                    break;

                default:
                    lock (state.Gate)
                    {
                        knote = AttachOrUpdateLocked(
                            state, change.Ident, change.Filter, change.UserData, addFlags,
                            change.Fflags, change.Data, out _);
                    }

                    break;
            }
        }
        else if (change.Filter == KernelEventFilterUser)
        {
            lock (state.Gate)
            {
                knote!.UserData = change.UserData;
                TouchUserKnoteLocked(state, knote, change.Fflags, change.Data, created: false);
            }

            Signal(state);
        }
        else
        {
            lock (state.Gate)
            {
                knote!.UserData = change.UserData;
                knote.SFflags = change.Fflags;
                knote.SData = change.Data;
            }
        }

        if (knote is null)
        {
            return PosixEnoent;
        }

        lock (state.Gate)
        {
            if (knote.Dropped)
            {
                return 0;
            }

            if ((flags & KernelEventFlagForceOneShot) != 0)
            {
                knote.Flags |= KernelEventFlagOneShot;
                ActivateLocked(state, knote);
            }

            if ((flags & KernelEventFlagDisable) != 0)
            {
                SetEnabledLocked(state, knote, enabled: false);
            }
            else if ((flags & KernelEventFlagEnable) != 0)
            {
                SetEnabledLocked(state, knote, enabled: true);
            }
        }

        Signal(state);
        return 0;
    }

    // filt_usertouch(EVENT_REGISTER): NOTE_TRIGGER activates; the control bits
    // combine the new fflags into the saved ones; data is saved for delivery.
    private static void TouchUserKnoteLocked(Kqueue state, Knote knote, uint fflags, ulong data, bool created)
    {
        var value = fflags & NoteFflagsMask;
        if (created)
        {
            knote.SFflags = value;
        }
        else
        {
            knote.SFflags = (fflags & NoteFfctrlMask) switch
            {
                NoteFfand => knote.SFflags & value,
                NoteFfor => knote.SFflags | value,
                NoteFfcopy => value,
                _ => knote.SFflags,
            };
        }

        knote.SData = data;
        knote.Fflags = knote.SFflags;
        knote.Data = knote.SData;
        if ((fflags & NoteTrigger) != 0)
        {
            ActivateLocked(state, knote);
        }
    }
}
