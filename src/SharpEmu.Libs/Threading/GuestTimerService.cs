// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using SharpEmu.HLE.Host;

namespace SharpEmu.Libs.Threading;

/// <summary>
/// One armed deadline in a <see cref="GuestTimerService"/>. Create it with
/// <see cref="GuestTimerService.Schedule"/>; it fires at most once per arming.
/// </summary>
public sealed class GuestTimer
{
    internal GuestTimer(GuestTimerService owner, long deadline, Action<object?> callback, object? state)
    {
        Owner = owner;
        Deadline = deadline;
        Callback = callback;
        State = state;
    }

    internal GuestTimerService Owner { get; }

    // Guarded by the owning service's gate.
    internal long Deadline;
    internal long Sequence;
    internal int HeapIndex = -1;
    internal Action<object?> Callback;
    internal object? State;

    /// <summary>The Stopwatch timestamp this timer is (or was last) armed for.</summary>
    public long DeadlineTimestamp => Volatile.Read(ref Deadline);
}

/// <summary>
/// The emulator's deadline service: one binary min-heap of absolute
/// deadlines served by one thread that sleeps precisely until the earliest of them
/// (<see cref="HostPreciseClock"/>: kqueue + <c>mach_wait_until</c> on macOS). It replaces
/// per-wait <see cref="System.Threading.Timer"/>s and fixed-interval polling: a wait with a
/// timeout arms one timer, and an earlier timer arriving wakes the service thread at once.
///
/// <para>API, all thread-safe:</para>
/// <list type="bullet">
/// <item><see cref="Schedule"/>(deadline, callback, state) arms a new timer for a Stopwatch
/// deadline; <see cref="ScheduleAfterMicroseconds"/> takes a relative interval.</item>
/// <item><see cref="Cancel"/> disarms; true when the timer had not fired yet (after true,
/// the callback will not run).</item>
/// <item><see cref="Reschedule"/> re-arms an existing timer (pending or already fired) for
/// a new deadline: the periodic-timer primitive.</item>
/// </list>
/// Callbacks run on the service thread, in deadline order, outside the heap lock. They
/// must be short and must not block: complete a waiter, enqueue an event, re-arm.
/// A deadline already in the past fires on the service thread right away, never inline
/// in the caller.
/// </summary>
public sealed class GuestTimerService : IDisposable
{
    private static readonly Lazy<GuestTimerService> SharedInstance = new(
        () => new GuestTimerService("SharpEmu-GuestTimer"),
        LazyThreadSafetyMode.ExecutionAndPublication);

    private readonly object _gate = new();
    private readonly HostPreciseClock.PreciseSignal _signal = new();
    private readonly string _name;
    private GuestTimer[] _heap = new GuestTimer[64];
    private int _count;
    private long _nextSequence;
    private long _firedCount;
    private volatile bool _stopping;
    private long _serviceWaitDeadline = long.MaxValue;
    private Thread? _managedThread;
    private HostQosThread? _qosThread;

    public GuestTimerService(string name)
    {
        _name = name;
        // Coalescing-exempt class on macOS, so the kernel honours the precise deadline.
        _qosThread = HostQosThread.TryStart(name, HostThreadQos.UserInteractive, 256 * 1024, Run, out _);
        if (_qosThread is null)
        {
            _managedThread = new Thread(Run, 256 * 1024)
            {
                IsBackground = true,
                Name = name,
                Priority = ThreadPriority.Highest,
            };
            _managedThread.Start();
        }
    }

    /// <summary>The process-wide service (created on first use).</summary>
    public static GuestTimerService Shared => SharedInstance.Value;

    /// <summary>Timers currently armed.</summary>
    public int PendingCount
    {
        get
        {
            lock (_gate)
            {
                return _count;
            }
        }
    }

    /// <summary>Timers that have fired since the service started (diagnostics, tests).</summary>
    public long FiredCount => Interlocked.Read(ref _firedCount);

    /// <summary>Arms a new timer for the Stopwatch timestamp <paramref name="deadlineTimestamp"/>.</summary>
    public GuestTimer Schedule(long deadlineTimestamp, Action<object?> callback, object? state)
    {
        ArgumentNullException.ThrowIfNull(callback);
        var timer = new GuestTimer(this, deadlineTimestamp, callback, state);
        lock (_gate)
        {
            InsertLocked(timer);
        }

        WakeServiceIfEarlier(deadlineTimestamp);
        return timer;
    }

    /// <summary>Arms a new timer <paramref name="microseconds"/> from now.</summary>
    public GuestTimer ScheduleAfterMicroseconds(long microseconds, Action<object?> callback, object? state) =>
        Schedule(HostPreciseClock.DeadlineAfterMicroseconds(microseconds), callback, state);

    /// <summary>Disarms <paramref name="timer"/>. True when it was still pending.</summary>
    public bool Cancel(GuestTimer? timer)
    {
        if (timer is null || !ReferenceEquals(timer.Owner, this))
        {
            return false;
        }

        lock (_gate)
        {
            if (timer.HeapIndex < 0)
            {
                return false;
            }

            RemoveAtLocked(timer.HeapIndex);
            return true;
        }
    }

    /// <summary>
    /// Re-arms <paramref name="timer"/> for <paramref name="deadlineTimestamp"/>, whether it
    /// is pending (moved) or has fired (armed again). False for a foreign timer.
    /// </summary>
    public bool Reschedule(GuestTimer timer, long deadlineTimestamp)
    {
        if (!ReferenceEquals(timer.Owner, this))
        {
            return false;
        }

        lock (_gate)
        {
            if (timer.HeapIndex >= 0)
            {
                RemoveAtLocked(timer.HeapIndex);
            }

            timer.Deadline = deadlineTimestamp;
            InsertLocked(timer);
        }

        WakeServiceIfEarlier(deadlineTimestamp);
        return true;
    }

    public void Dispose()
    {
        _stopping = true;
        _signal.Signal();
        _qosThread?.Join(1000);
        _managedThread?.Join(1000);
    }

    private void WakeServiceIfEarlier(long deadlineTimestamp)
    {
        if (deadlineTimestamp < Volatile.Read(ref _serviceWaitDeadline))
        {
            _signal.Signal();
        }
    }

    private void Run()
    {
        // Every timed guest wait ends on this thread, and it only pops due timers and runs their
        // short callbacks: the real-time band keeps its wakeups on the deadline.
        HostThreadPrecedence.TryApplyTimeConstraintToCurrentThread(_name);
        var due = new List<GuestTimer>();
        while (!_stopping)
        {
            long nextDeadline;
            lock (_gate)
            {
                var now = Stopwatch.GetTimestamp();
                while (_count > 0 && _heap[0].Deadline <= now)
                {
                    var timer = _heap[0];
                    RemoveAtLocked(0);
                    due.Add(timer);
                }

                nextDeadline = _count > 0 ? _heap[0].Deadline : long.MaxValue;
                Volatile.Write(ref _serviceWaitDeadline, due.Count > 0 ? long.MinValue : nextDeadline);
            }

            if (due.Count > 0)
            {
                foreach (var timer in due)
                {
                    Interlocked.Increment(ref _firedCount);
                    try
                    {
                        timer.Callback(timer.State);
                    }
                    catch (Exception exception)
                    {
                        Console.Error.WriteLine($"[KERNEL][ERROR] {_name}: timer callback failed: {exception}");
                    }
                }

                due.Clear();
                continue;
            }

            _ = _signal.WaitUntil(nextDeadline);
        }
    }

    private void InsertLocked(GuestTimer timer)
    {
        if (_count == _heap.Length)
        {
            Array.Resize(ref _heap, _heap.Length * 2);
        }

        timer.Sequence = ++_nextSequence;
        timer.HeapIndex = _count;
        _heap[_count++] = timer;
        SiftUp(timer.HeapIndex);
    }

    private void RemoveAtLocked(int index)
    {
        var removed = _heap[index];
        removed.HeapIndex = -1;
        _count--;
        if (index == _count)
        {
            _heap[_count] = null!;
            return;
        }

        var last = _heap[_count];
        _heap[_count] = null!;
        _heap[index] = last;
        last.HeapIndex = index;
        SiftDown(index);
        SiftUp(last.HeapIndex);
    }

    private static bool Earlier(GuestTimer left, GuestTimer right) =>
        left.Deadline < right.Deadline ||
        (left.Deadline == right.Deadline && left.Sequence < right.Sequence);

    private void SiftUp(int index)
    {
        var item = _heap[index];
        while (index > 0)
        {
            var parentIndex = (index - 1) >> 1;
            var parent = _heap[parentIndex];
            if (!Earlier(item, parent))
            {
                break;
            }

            _heap[index] = parent;
            parent.HeapIndex = index;
            index = parentIndex;
        }

        _heap[index] = item;
        item.HeapIndex = index;
    }

    private void SiftDown(int index)
    {
        var item = _heap[index];
        while (true)
        {
            var left = (index << 1) + 1;
            if (left >= _count)
            {
                break;
            }

            var right = left + 1;
            var child = right < _count && Earlier(_heap[right], _heap[left]) ? right : left;
            if (!Earlier(_heap[child], item))
            {
                break;
            }

            _heap[index] = _heap[child];
            _heap[index].HeapIndex = index;
            index = child;
        }

        _heap[index] = item;
        item.HeapIndex = index;
    }
}
