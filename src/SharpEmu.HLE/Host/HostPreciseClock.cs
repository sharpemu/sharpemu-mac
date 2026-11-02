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
/// earlier deadline arrives. On Linux <see cref="SleepUntil"/> sleeps in
/// <c>clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME)</c> straight to the deadline with the
/// calling thread's timer slack lowered to 1 ns (<c>SHARPEMU_LINUX_PRECISE_SLEEP=0</c>
/// restores the coarse path). Other hosts fall back to a coarse sleep for the bulk of the
/// interval and a yield-spin for the last ~1.2 ms.
/// </summary>
public static unsafe partial class HostPreciseClock
{
    private const long SpinTailMicroseconds = 1200;
    private static readonly bool MachAvailable = InitializeMach(out MachNumer, out MachDenom);
    private static readonly uint MachNumer;
    private static readonly uint MachDenom;

    private const int LinuxClockMonotonic = 1;
    private const int LinuxTimerAbstime = 1;
    private const int LinuxPrSetTimerSlack = 29;
    private const int LinuxEintr = 4;
    private static readonly bool LinuxAbsoluteMapping;
    private static readonly bool LinuxPreciseAvailable = InitializeLinux(out LinuxAbsoluteMapping);

    // SHARPEMU_LINUX_PRECISE_SPIN_US set (0..1000): a fixed early-wake margin of that many microseconds
    // (0 = a plain sleep to the deadline, no early wake) and no adaptation. Unset: adaptive early wake.
    private static readonly long LinuxFixedSpinNanoseconds = ReadLinuxFixedSpinNanoseconds();

    // SHARPEMU_LINUX_PRECISE_SPIN_FRACTION (0..1, default 0.25): the most of a sleep that may be spent
    // spinning at its end. 0 = a plain sleep to the deadline with no early wake.
    private static readonly double LinuxSpinFraction = ReadLinuxSpinFraction();

    private const long EarlyWakeSeedNanoseconds = 50_000;
    private const long EarlyWakeMinNanoseconds = 10_000;
    private const long EarlyWakeMaxNanoseconds = 200_000;

    /// <summary>
    /// True when the per-thread adaptive early wake is in use (no fixed SHARPEMU_LINUX_PRECISE_SPIN_US and
    /// SHARPEMU_LINUX_PRECISE_SPIN_FRACTION above 0).
    /// </summary>
    public static bool LinuxEarlyWakeAdaptive =>
        LinuxPreciseAvailable && LinuxFixedSpinNanoseconds < 0 && LinuxSpinFraction > 0;

    [ThreadStatic]
    private static bool _linuxTimerSlackSet;

    // Per thread estimates of how late clock_nanosleep wakes, one per sleep length class (0 = not seeded
    // yet): the kernel's idle states are deeper, and so slower to leave, the longer the sleep.
    [ThreadStatic]
    private static long[]? _earlyWakeNanoseconds;

    /// <summary>True when waits use mach_wait_until (macOS).</summary>
    public static bool IsPrecise => MachAvailable;

    /// <summary>
    /// True when waits use <c>clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME)</c> (Linux, unless
    /// <c>SHARPEMU_LINUX_PRECISE_SLEEP=0</c>). Always false on macOS, where <see cref="IsPrecise"/> applies.
    /// </summary>
    public static bool IsLinuxPrecise => LinuxPreciseAvailable;

    /// <summary>True when a precise wait is available on this host (macOS or Linux).</summary>
    public static bool HasPreciseSleep => MachAvailable || LinuxPreciseAvailable;

    /// <summary>Whether a <c>SHARPEMU_LINUX_PRECISE_SLEEP</c> value leaves the Linux precise path on (only "0" turns it off).</summary>
    public static bool LinuxPreciseSleepEnabled(string? environmentValue) =>
        !string.Equals(environmentValue?.Trim(), "0", StringComparison.Ordinal);

    /// <summary>
    /// How the Stopwatch deadline maps onto CLOCK_MONOTONIC on Linux: "absolute" when
    /// Stopwatch ticks are CLOCK_MONOTONIC nanoseconds (the deadline is passed straight to
    /// clock_nanosleep), "relative" when they are not and the remaining interval is added to
    /// the current CLOCK_MONOTONIC reading, "off" when the precise path is not in use.
    /// </summary>
    public static string LinuxClockMapping =>
        !LinuxPreciseAvailable ? "off" : LinuxAbsoluteMapping ? "absolute" : "relative";

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
        while (true)
        {
            var remaining = deadlineTimestamp - Stopwatch.GetTimestamp();
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
                var nanoseconds = (long)((decimal)remaining * 1_000_000_000m / Stopwatch.Frequency);
                var machTicks = (ulong)((decimal)nanoseconds * MachDenom / MachNumer);
                _ = mach_wait_until(mach_absolute_time() + Math.Max(1UL, machTicks));
                continue;
            }

            if (LinuxPreciseAvailable)
            {
                LinuxSleepUntil(deadlineTimestamp);
                return;
            }

            CoarseSleepStep(remaining);
        }
    }

    private static bool InitializeLinux(out bool absoluteMapping)
    {
        absoluteMapping = false;
        if (!OperatingSystem.IsLinux() ||
            !LinuxPreciseSleepEnabled(Environment.GetEnvironmentVariable("SHARPEMU_LINUX_PRECISE_SLEEP")))
        {
            return false;
        }

        try
        {
            // Stopwatch is CLOCK_MONOTONIC in nanoseconds on Linux; confirm instead of assuming,
            // since a wrong mapping would turn every guest sleep into a wrong-length one.
            if (Stopwatch.Frequency == 1_000_000_000L)
            {
                var before = Stopwatch.GetTimestamp();
                if (clock_gettime(LinuxClockMonotonic, out var ts) != 0)
                {
                    return false;
                }

                var after = Stopwatch.GetTimestamp();
                var monotonic = ts.Seconds * 1_000_000_000L + ts.Nanoseconds;
                absoluteMapping = monotonic >= before - 1000 && monotonic <= after + 1000;
            }
            else if (clock_gettime(LinuxClockMonotonic, out _) != 0)
            {
                return false;
            }

            // Probe the sleep itself: a missing symbol throws here, not in the middle of a guest sleep.
            var now = LinuxMonotonicNanoseconds();
            var request = new Timespec { Seconds = now / 1_000_000_000L, Nanoseconds = now % 1_000_000_000L };
            return clock_nanosleep(LinuxClockMonotonic, LinuxTimerAbstime, ref request, IntPtr.Zero) == 0;
        }
        catch (Exception)
        {
            absoluteMapping = false;
            return false;
        }
    }

    private static double ReadLinuxSpinFraction() =>
        double.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_LINUX_PRECISE_SPIN_FRACTION"),
            System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var f) &&
        f >= 0 && f <= 1
            ? f
            : 0.25;

    /// <summary>
    /// How early to wake for a sleep with <paramref name="leftNanoseconds"/> to go: the thread's wake
    /// lateness estimate, but never more than <paramref name="fraction"/> of the sleep (and never less
    /// than 10 us, so a very short sleep may still spin). That bounds the spin to about a fraction of
    /// each sleep; where the wake latency fits in that budget the deadline is hit almost exactly.
    /// </summary>
    internal static long EarlyWakeMargin(long estimateNanoseconds, long leftNanoseconds, double fraction) =>
        Math.Min(estimateNanoseconds, Math.Max(EarlyWakeMinNanoseconds, (long)(leftNanoseconds * fraction)));

    private static long ReadLinuxFixedSpinNanoseconds()
    {
        return long.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_LINUX_PRECISE_SPIN_US"), out var us) &&
               us >= 0 && us <= 1000
            ? us * 1000
            : -1;
    }

    /// <summary>
    /// Folds one observed wake lateness into the early-wake estimate. This tracks a high quantile (about
    /// the 85th percentile) rather than the mean: a wake later than the estimate steps it up by up to
    /// 8 us (that wake missed the deadline), an earlier one steps it down by up to 1.5 us, so one stalled
    /// wake cannot drag the estimate up and spin every following sleep. Clamped to [10, 200] us.
    /// </summary>
    internal static long UpdateEarlyWake(long estimateNanoseconds, long observedOvershootNanoseconds)
    {
        var observed = Math.Max(0, observedOvershootNanoseconds);
        var next = observed > estimateNanoseconds
            ? estimateNanoseconds + Math.Min(8_000, observed - estimateNanoseconds)
            : estimateNanoseconds - Math.Min(1_500, estimateNanoseconds - observed);
        return Math.Clamp(next, EarlyWakeMinNanoseconds, EarlyWakeMaxNanoseconds);
    }

    private static long LinuxMonotonicNanoseconds()
    {
        _ = clock_gettime(LinuxClockMonotonic, out var ts);
        return ts.Seconds * 1_000_000_000L + ts.Nanoseconds;
    }

    private static void SetLinuxTimerSlack()
    {
        // Timer slack is per thread (default 50 us) and lets the kernel coalesce a sleeper's wakeup
        // up to that much late. 1 ns is the smallest value; 0 would mean "reset to the default".
        _linuxTimerSlackSet = true;
        try
        {
            _ = prctl(LinuxPrSetTimerSlack, 1, 0, 0, 0);
        }
        catch (Exception)
        {
        }
    }

    private static long LinuxNowNanoseconds() =>
        LinuxAbsoluteMapping ? Stopwatch.GetTimestamp() : LinuxMonotonicNanoseconds();

    private static void LinuxSleepAbsolute(long deadlineNanoseconds)
    {
        var request = new Timespec
        {
            Seconds = deadlineNanoseconds / 1_000_000_000L,
            Nanoseconds = deadlineNanoseconds % 1_000_000_000L,
        };
        // clock_nanosleep returns the error number itself; EINTR (a signal landed, e.g. the guest
        // thread's own signal handlers) resumes toward the same absolute deadline.
        while (clock_nanosleep(LinuxClockMonotonic, LinuxTimerAbstime, ref request, IntPtr.Zero) == LinuxEintr)
        {
        }
    }

    private static void LinuxSpinUntil(long deadlineNanoseconds)
    {
        // Timestamp checks plus a pause-based spin; never sched_yield.
        while (LinuxNowNanoseconds() < deadlineNanoseconds)
        {
            Thread.SpinWait(8);
        }
    }

    private static void LinuxSleepUntil(long deadlineTimestamp)
    {
        if (!_linuxTimerSlackSet)
        {
            SetLinuxTimerSlack();
        }

        long deadline;
        if (LinuxAbsoluteMapping)
        {
            deadline = deadlineTimestamp;
        }
        else
        {
            var remaining = deadlineTimestamp - Stopwatch.GetTimestamp();
            if (remaining <= 0)
            {
                return;
            }

            deadline = LinuxMonotonicNanoseconds() +
                (long)((decimal)remaining * 1_000_000_000m / Stopwatch.Frequency);
        }

        var fixedSpin = LinuxFixedSpinNanoseconds;
        if (fixedSpin >= 0)
        {
            if (fixedSpin == 0)
            {
                LinuxSleepAbsolute(deadline);
            }
            else
            {
                LinuxSleepAbsolute(deadline - fixedSpin);
                LinuxSpinUntil(deadline);
            }

            return;
        }

        // Adaptive early wake: a timer wakeup lands late by this thread's recent overshoot, so ask for
        // it early (capped to a fraction of the sleep) and finish with a short spin.
        var left = deadline - LinuxNowNanoseconds();
        if (left <= 0)
        {
            return;
        }

        var fraction = LinuxSpinFraction;
        if (fraction <= 0)
        {
            LinuxSleepAbsolute(deadline);
            return;
        }

        if (left < 2 * EarlyWakeMinNanoseconds)
        {
            LinuxSpinUntil(deadline);
            return;
        }

        var estimates = _earlyWakeNanoseconds ??= [0, 0, 0];
        var bucket = left < 400_000 ? 0 : left < 2_000_000 ? 1 : 2;
        var estimate = estimates[bucket];
        if (estimate == 0)
        {
            estimate = EarlyWakeSeedNanoseconds;
        }

        var target = deadline - EarlyWakeMargin(estimate, left, fraction);
        LinuxSleepAbsolute(target);
        // The lateness is observed against the target, so the estimate stays a property of the host
        // however much margin the cap allowed this time.
        estimates[bucket] = UpdateEarlyWake(estimate, LinuxNowNanoseconds() - target);
        LinuxSpinUntil(deadline);
    }

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
                        else if (LinuxPreciseAvailable)
                        {
                            LinuxSleepUntil(deadlineTimestamp);
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

    [DllImport("libc", EntryPoint = "clock_nanosleep")]
    private static extern int clock_nanosleep(int clockId, int flags, ref Timespec request, IntPtr remain);

    [DllImport("libc", EntryPoint = "clock_gettime")]
    private static extern int clock_gettime(int clockId, out Timespec time);

    [DllImport("libc", EntryPoint = "prctl")]
    private static extern int prctl(int option, ulong arg2, ulong arg3, ulong arg4, ulong arg5);

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
