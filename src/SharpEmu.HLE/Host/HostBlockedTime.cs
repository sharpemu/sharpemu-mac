// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace SharpEmu.HLE.Host;

/// <summary>
/// Per-thread running total of Stopwatch ticks the thread spent blocked in host pacing (audio
/// devices draining, silent ports keeping time). Blocking sites add to it; the performance
/// overlay's import gateway reads it before and after a call, so the difference is the part of
/// that call that was waiting rather than working. The total only grows, so a block outside any
/// call, or while nobody is timing calls, never lands in a later call's difference.
/// </summary>
public static class HostBlockedTime
{
    [ThreadStatic]
    private static long _total;

    /// <summary>Ticks this thread has spent blocked so far.</summary>
    public static long Total
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _total;
    }

    /// <summary>Adds <paramref name="ticks"/> Stopwatch ticks of blocking to the current thread.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Add(long ticks)
    {
        if (ticks > 0)
        {
            _total += ticks;
        }
    }

    /// <summary><see cref="Thread.Sleep(int)"/>, counted as blocked time.</summary>
    public static void Sleep(int milliseconds)
    {
        var start = Stopwatch.GetTimestamp();
        Thread.Sleep(milliseconds);
        Add(Stopwatch.GetTimestamp() - start);
    }

    /// <summary><see cref="Thread.Sleep(TimeSpan)"/>, counted as blocked time.</summary>
    public static void Sleep(TimeSpan duration)
    {
        var start = Stopwatch.GetTimestamp();
        Thread.Sleep(duration);
        Add(Stopwatch.GetTimestamp() - start);
    }
}
