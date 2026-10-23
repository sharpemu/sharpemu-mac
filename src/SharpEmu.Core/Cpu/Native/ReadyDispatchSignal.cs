// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Core.Cpu.Native;

/// <summary>
/// The wake for the one thread that dispatches ready guest threads, kept apart from the guest
/// thread gate's Monitor so that a ready-enqueue wakes that thread alone instead of pulsing
/// every waiter on the gate (joiners and nested-callback waiters wait there too, and a single
/// <c>Monitor.Pulse</c> could hand its wake to one of them and lose it for the dispatcher).
/// </summary>
/// <remarks>
/// The queue the dispatcher drains and the parked flag are protected by the same external lock
/// (the gate): the dispatcher finds the queue empty and raises the flag under it, and an
/// enqueuer checks the flag under it after adding its item. Whichever goes second sees the
/// other, so an item is never left behind a sleeping dispatcher. The wake itself is a latched
/// auto-reset event, so a signal that lands before the dispatcher reaches <see cref="Park"/>
/// is not lost either. At most one thread may park.
/// </remarks>
internal sealed class ReadyDispatchSignal
{
    private readonly AutoResetEvent _wake = new(false);
    private bool _parked;

    /// <summary>Whether the dispatcher is parked or about to be. Caller holds the lock.</summary>
    public bool IsParkedLocked => _parked;

    /// <summary>The dispatcher found nothing to do and will call <see cref="Park"/> once it
    /// leaves the lock. Caller holds the lock.</summary>
    public void MarkParkedLocked() => _parked = true;

    /// <summary>
    /// Wakes the dispatcher if it is parked. Returns whether it was. Caller holds the lock,
    /// after making its item visible to the dispatcher.
    /// </summary>
    public bool TrySignalLocked()
    {
        if (!_parked)
        {
            return false;
        }

        _parked = false;
        _wake.Set();
        return true;
    }

    /// <summary>Wakes the dispatcher whether or not it is parked (shutdown). Any thread.</summary>
    public void SignalAlways() => _wake.Set();

    /// <summary>Sleeps until signalled or the timeout passes. Not under the lock; the caller
    /// takes it again with <see cref="ClearParkedLocked"/>.</summary>
    public bool Park(int timeoutMilliseconds) => _wake.WaitOne(timeoutMilliseconds);

    /// <summary>The dispatcher is running again (also after a timeout). Caller holds the lock.</summary>
    public void ClearParkedLocked() => _parked = false;

    /// <summary>Forgets a stale signal and the parked flag before the dispatcher starts again.
    /// Caller holds the lock.</summary>
    public void ResetLocked()
    {
        _parked = false;
        _wake.Reset();
    }
}
