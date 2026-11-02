// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Runtime.InteropServices;

namespace SharpEmu.HLE.Host;

/// <summary>
/// Precise absolute-deadline waits on the host clock.
///
/// Deadlines are <see cref="Stopwatch"/> timestamps throughout the emulator. On macOS
/// <see cref="SleepUntil"/> converts the remaining interval into a
/// <c>mach_absolute_time</c> deadline and parks in <c>mach_wait_until</c>, which wakes
/// within a few microseconds and burns no CPU. <see cref="PreciseSignal"/> adds an
/// interruptible wait (a kqueue <c>EVFILT_USER</c> event with a nanosecond timeout) so
/// one thread can sleep until the next deadline and still be woken early when an
/// earlier deadline arrives. Other hosts fall back to a coarse sleep for the bulk of the
/// interval and a yield-spin for the last ~1.2 ms.
/// </summary>
public static unsafe partial class HostPreciseClock
{
    private const long SpinTailMicroseconds = 1200;
    private static readonly bool MachAvailable = InitializeMach(out MachNumer, out MachDenom);
    private static readonly uint MachNumer;
    private static readonly uint MachDenom;
    private static readonly MachScale MachTickScale = MachScale.Create(Stopwatch.Frequency, MachNumer, MachDenom);

    // Diagnostics for the mach branch of SleepUntil. A pass that wakes before its deadline
    // ("early wake") means mach_wait_until returned short, so the loop went round again; a
    // high passes-per-call ratio would be a spin. Updated across threads, read via TakeMachSleepReport.
    private static long _machSleepCalls;
    private static long _machSleepPasses;
    private static long _machSleepEarlyWakes;

    /// <summary>True when waits use mach_wait_until (macOS).</summary>
    public static bool IsPrecise => MachAvailable;

    /// <summary>Microseconds → Stopwatch ticks, saturating.</summary>
    public static long MicrosecondsToTicks(long microseconds)
    {
        if (microseconds <= 0)
        {
            return 0;
        }

        if (microseconds >= long.MaxValue / Math.Max(1, Stopwatch.Frequency))
        {
            return long.MaxValue / 4;
        }

        return microseconds * Stopwatch.Frequency / 1_000_000;
    }

    /// <summary>Stopwatch ticks → microseconds (rounded down, never negative).</summary>
    public static long TicksToMicroseconds(long ticks) =>
        ticks <= 0 ? 0 : (long)((decimal)ticks * 1_000_000m / Stopwatch.Frequency);

    /// <summary>Adds a microsecond interval to now, saturating instead of overflowing.</summary>
    public static long DeadlineAfterMicroseconds(long microseconds)
    {
        var now = Stopwatch.GetTimestamp();
        var ticks = MicrosecondsToTicks(microseconds);
        return ticks >= long.MaxValue - now ? long.MaxValue : now + ticks;
    }

    /// <summary>Blocks the calling thread until the Stopwatch clock reaches <paramref name="deadlineTimestamp"/>.</summary>
    public static void SleepUntil(long deadlineTimestamp)
    {
        var waited = false;
        while (true)
        {
            var remaining = deadlineTimestamp - Stopwatch.GetTimestamp();
            if (waited && remaining > 0)
            {
                _ = Interlocked.Increment(ref _machSleepEarlyWakes);
            }

            if (remaining <= 0)
            {
                return;
            }

            if (remaining > Stopwatch.Frequency * 30)
            {
                // Very long sleeps: no precision needed, and the conversions below stay small.
                Thread.Sleep(10_000);
                continue;
            }

            if (MachAvailable)
            {
                if (!waited)
                {
                    _ = Interlocked.Increment(ref _machSleepCalls);
                }

                waited = true;
                _ = Interlocked.Increment(ref _machSleepPasses);
                var machTicks = MachTickScale.Convert(remaining);
                _ = mach_wait_until(mach_absolute_time() + Math.Max(1UL, machTicks));
                continue;
            }

            CoarseSleepStep(remaining);
        }
    }

    /// <summary>
    /// Stopwatch ticks → mach absolute-time ticks via nanoseconds, each step floored. Exactly
    /// equal to the decimal formulation this replaced for 0 &lt; remaining ≤ 30 s of ticks.
    /// </summary>
    internal static ulong StopwatchTicksToMachTicks(long remaining, long frequency, uint numer, uint denom) =>
        MachScale.Create(frequency, numer, denom).Convert(remaining);

    /// <summary>Cumulative mach-branch SleepUntil counters (calls, passes, early wakes) and resets them.</summary>
    public static string TakeMachSleepReport() =>
        FormattableString.Invariant($"[PERF][SLEEP_UNTIL] mach={(MachAvailable ? 1 : 0)} ") +
        FormattableString.Invariant($"calls={Interlocked.Exchange(ref _machSleepCalls, 0)} passes={Interlocked.Exchange(ref _machSleepPasses, 0)} early_wakes={Interlocked.Exchange(ref _machSleepEarlyWakes, 0)}");

    /// <summary>Blocks for <paramref name="microseconds"/>.</summary>
    public static void SleepMicroseconds(long microseconds)
    {
        if (microseconds > 0)
        {
            SleepUntil(DeadlineAfterMicroseconds(microseconds));
        }
    }

    private static void CoarseSleepStep(long remainingTicks)
    {
        var remainingMicroseconds = TicksToMicroseconds(remainingTicks);
        if (remainingMicroseconds > SpinTailMicroseconds + 1000)
        {
            Thread.Sleep((int)Math.Min(int.MaxValue, (remainingMicroseconds - SpinTailMicroseconds) / 1000));
        }
        else if (remainingMicroseconds > SpinTailMicroseconds)
        {
            Thread.Sleep(1);
        }
        else if (remainingMicroseconds > 100)
        {
            Thread.Sleep(0);
        }
        else
        {
            Thread.SpinWait(64);
        }
    }

    // Two floored rational steps (ticks → ns, ns → mach ticks), each reduced by its gcd once
    // at init so the common cases (1e9 Hz Stopwatch, 1/1 timebase) skip the 128-bit path.
    internal readonly struct MachScale
    {
        private readonly ulong _nsMul, _nsDiv, _machMul, _machDiv;

        private MachScale(ulong nsMul, ulong nsDiv, ulong machMul, ulong machDiv)
        {
            _nsMul = nsMul;
            _nsDiv = nsDiv;
            _machMul = machMul;
            _machDiv = machDiv;
        }

        public static MachScale Create(long frequency, uint numer, uint denom)
        {
            if (frequency <= 0 || numer == 0 || denom == 0)
            {
                return default;
            }

            var nsGcd = Gcd(1_000_000_000UL, (ulong)frequency);
            var machGcd = Gcd(denom, numer);
            return new MachScale(1_000_000_000UL / nsGcd, (ulong)frequency / nsGcd, denom / machGcd, numer / machGcd);
        }

        /// <summary>Remaining Stopwatch ticks → mach ticks (0 for a default scale or a non-positive input).</summary>
        public ulong Convert(long remaining)
        {
            if (remaining <= 0 || _nsDiv == 0)
            {
                return 0;
            }

            return MulDiv(MulDiv((ulong)remaining, _nsMul, _nsDiv), _machMul, _machDiv);
        }

        private static ulong MulDiv(ulong value, ulong mul, ulong div)
        {
            if (mul == div)
            {
                return value;
            }

            var high = Math.BigMul(value, mul, out var low);
            return high == 0 ? low / div : (ulong)(new UInt128(high, low) / div);
        }

        private static ulong Gcd(ulong a, ulong b)
        {
            while (b != 0)
            {
                (a, b) = (b, a % b);
            }

            return a;
        }
    }

    private static bool InitializeMach(out uint numer, out uint denom)
    {
        numer = 0;
        denom = 0;
        if (!OperatingSystem.IsMacOS())
        {
            return false;
        }

        try
        {
            MachTimebase info;
            if (mach_timebase_info(&info) != 0 || info.Numer == 0 || info.Denom == 0)
            {
                return false;
            }

            numer = info.Numer;
            denom = info.Denom;
            return true;
        }
        catch (Exception)
        {
            return false;
        }
    }

    /// <summary>
    /// A wakeable precise wait for one waiting thread (any number of signallers).
    /// <see cref="WaitUntil"/> returns when the deadline passes or <see cref="Signal"/> is
    /// called; a signal that arrives while nobody waits is remembered for the next wait.
    /// </summary>
    public sealed class PreciseSignal : IDisposable
    {
        // Below this much remaining time the kqueue wait is skipped for mach_wait_until.
        private static readonly long UninterruptibleTailTicks = MicrosecondsToTicks(300);
        private readonly int _kq = -1;
        private readonly AutoResetEvent? _fallback;
        private int _pending;

        public PreciseSignal()
        {
            if (MachAvailable)
            {
                try
                {
                    _kq = kqueue();
                    if (_kq >= 0)
                    {
                        var change = new KEvent { Ident = 1, Filter = EvFiltUser, Flags = EvAdd | EvClear };
                        if (kevent(_kq, &change, 1, null, 0, null) < 0)
                        {
                            _ = close(_kq);
                            _kq = -1;
                        }
                    }
                }
                catch (Exception)
                {
                    _kq = -1;
                }
            }

            if (_kq < 0)
            {
                _fallback = new AutoResetEvent(false);
            }
        }

        /// <summary>Wakes the current (or next) <see cref="WaitUntil"/>.</summary>
        public void Signal()
        {
            if (Interlocked.Exchange(ref _pending, 1) != 0)
            {
                return;
            }

            if (_fallback is not null)
            {
                _fallback.Set();
                return;
            }

            var trigger = new KEvent { Ident = 1, Filter = EvFiltUser, FFlags = NoteTrigger };
            _ = kevent(_kq, &trigger, 1, null, 0, null);
        }

        /// <summary>
        /// Waits until <paramref name="deadlineTimestamp"/> (<see cref="long.MaxValue"/>: no
        /// deadline) or a signal. Returns true when signalled.
        /// </summary>
        public bool WaitUntil(long deadlineTimestamp)
        {
            while (true)
            {
                if (Interlocked.Exchange(ref _pending, 0) != 0)
                {
                    DrainKernelEvent();
                    return true;
                }

                var remaining = deadlineTimestamp == long.MaxValue
                    ? long.MaxValue
                    : deadlineTimestamp - Stopwatch.GetTimestamp();
                if (remaining <= 0)
                {
                    return false;
                }

                if (_fallback is not null)
                {
                    if (remaining == long.MaxValue)
                    {
                        _fallback.WaitOne();
                    }
                    else
                    {
                        var microseconds = TicksToMicroseconds(remaining);
                        if (microseconds > SpinTailMicroseconds)
                        {
                            _fallback.WaitOne((int)Math.Min(int.MaxValue, Math.Max(1, (microseconds - SpinTailMicroseconds) / 1000)));
                        }
                        else
                        {
                            CoarseSleepStep(remaining);
                        }
                    }

                    continue;
                }

                if (remaining != long.MaxValue && remaining <= UninterruptibleTailTicks)
                {
                    SleepUntil(deadlineTimestamp);
                    continue;
                }

                KEvent received;
                if (remaining == long.MaxValue)
                {
                    _ = kevent(_kq, null, 0, &received, 1, null);
                }
                else
                {
                    var waitTicks = remaining - UninterruptibleTailTicks;
                    var nanoseconds = (long)((decimal)waitTicks * 1_000_000_000m / Stopwatch.Frequency);
                    var timeout = new Timespec
                    {
                        Seconds = nanoseconds / 1_000_000_000L,
                        Nanoseconds = nanoseconds % 1_000_000_000L,
                    };
                    _ = kevent(_kq, null, 0, &received, 1, &timeout);
                }
            }
        }

        private void DrainKernelEvent()
        {
            if (_kq < 0)
            {
                return;
            }

            var zero = new Timespec();
            KEvent received;
            _ = kevent(_kq, null, 0, &received, 1, &zero);
        }

        public void Dispose()
        {
            if (_kq >= 0)
            {
                _ = close(_kq);
            }

            _fallback?.Dispose();
        }
    }

    private const short EvFiltUser = -10;
    private const ushort EvAdd = 0x0001;
    private const ushort EvClear = 0x0020;
    private const uint NoteTrigger = 0x01000000;

    [StructLayout(LayoutKind.Sequential)]
    private struct MachTimebase
    {
        public uint Numer;
        public uint Denom;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct KEvent
    {
        public nuint Ident;
        public short Filter;
        public ushort Flags;
        public uint FFlags;
        public nint Data;
        public nint UData;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct Timespec
    {
        public long Seconds;
        public long Nanoseconds;
    }

    [LibraryImport("libSystem.B.dylib")]
    private static partial int mach_timebase_info(MachTimebase* info);

    [LibraryImport("libSystem.B.dylib")]
    private static partial ulong mach_absolute_time();

    [LibraryImport("libSystem.B.dylib")]
    private static partial int mach_wait_until(ulong deadline);

    [LibraryImport("libSystem.B.dylib")]
    private static partial int kqueue();

    [LibraryImport("libSystem.B.dylib")]
    private static partial int kevent(int kq, KEvent* changelist, int nchanges, KEvent* eventlist, int nevents, Timespec* timeout);

    [LibraryImport("libSystem.B.dylib")]
    private static partial int close(int fd);
}
