// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;

namespace SharpEmu.Libs.Kernel;

/// <summary>Says which job workers have work queued, read after a waiter has published itself
/// (see <see cref="JobWakeRuntime.Wait{TProbe}"/>) or before a waker picks whom to wake.</summary>
internal interface IJobWorkProbe
{
    /// <summary>Bit w set when worker w has queued work it should look at.</summary>
    ulong WorkMask();
}

/// <summary>
/// Wait and wake for idle guest job workers that a title patch parks instead of letting them
/// spin (see DemonsSoulsIdleNapPatch). The state the guest's trampolines touch lives in native
/// memory at <see cref="StateAddress"/>: a generation the guest increments with a locked
/// instruction each time it makes work ready, then a mask of the workers asleep here, which the
/// guest tests to skip the wake call when no one sleeps, then the subset of those that slept
/// without a queue check (workers re-polling jobs that wait on dependencies), which a
/// dependency-counter change tests instead. A waiter reads the generation before
/// it looks for work and passes it in; it sleeps only if the generation has not moved once its
/// bit is set. A waker increments the generation after it has queued the work and calls in only
/// when it then sees a sleeper. Both sides order a locked write before the read of the other's
/// word, so either the waiter sees the new generation or the waker sees its bit: a wake is
/// never lost, and work the guest makes ready without a signal costs at most the timeout.
/// </summary>
internal static unsafe class JobWakeRuntime
{
    public const string ReportEnvironmentVariable = "SHARPEMU_DS_JOB_WAKE_REPORT_S";
    public const string AbEnvironmentVariable = "SHARPEMU_DS_JOB_WAKE_AB_S";

    internal const int MaximumWorkers = 64;

    private const int GenerationSlot = 0;
    private const int SleepersSlot = 1;
    private const int PollersSlot = 2;
    private const int StateBytes = 128;

    // One futex word per worker, a cache line apart.
    private const int WordStride = 16;

    private const uint UL_COMPARE_AND_WAIT = 1;
    private const uint ULF_NO_ERRNO = 0x0100_0000;

    private static readonly delegate* unmanaged<uint, void*, ulong, ulong, ulong, int> _ulockWait2;
    private static readonly delegate* unmanaged[SuppressGCTransition]<uint, void*, ulong, int> _ulockWake;

    private static readonly object Gate = new();
    private static long* _state;
    private static int* _words;
    private static int _wakeCursor;
    private static Timer? _reporter;

    // SHARPEMU_DS_JOB_WAKE_AB_S=N (diagnostic): every other N seconds from the first wait, a
    // wait is the timed nap instead (it sleeps its timeout without joining the sleeper mask,
    // so no wake reaches it), giving one run interleaved wake/nap phases over the same scenes.
    private static readonly long _abPeriodTicks =
        int.TryParse(Environment.GetEnvironmentVariable(AbEnvironmentVariable), out var abSeconds) && abSeconds > 0
            ? abSeconds * Stopwatch.Frequency
            : 0;

    private static long _abEpoch;
    private static long _abNaps;
    private static long _waits;
    private static long _woken;
    private static long _timeouts;
    private static long _noBlock;
    private static long _wakeCalls;
    private static long _wakes;
    private static Counters _reported;

    static JobWakeRuntime()
    {
        if (!OperatingSystem.IsMacOS() || !NativeLibrary.TryLoad("/usr/lib/libSystem.B.dylib", out var system))
        {
            return;
        }

        if (NativeLibrary.TryGetExport(system, "__ulock_wait2", out var wait) &&
            NativeLibrary.TryGetExport(system, "__ulock_wake", out var wake))
        {
            _ulockWait2 = (delegate* unmanaged<uint, void*, ulong, ulong, ulong, int>)wait;
            _ulockWake = (delegate* unmanaged[SuppressGCTransition]<uint, void*, ulong, int>)wake;
        }
    }

    /// <summary>True when the host can block a worker with a timeout and wake it early
    /// (Darwin's ulock). Elsewhere the patch keeps the timed nap.</summary>
    public static bool IsSupported => _ulockWait2 != null && _ulockWake != null;

    /// <summary>Native address of the generation (+0) and sleeper mask (+8); 0 until
    /// <see cref="TryInitialize"/> succeeded.</summary>
    public static ulong StateAddress => (ulong)_state;

    internal static ulong GenerationAddress => (ulong)(_state + GenerationSlot);

    internal static ulong SleepersAddress => (ulong)(_state + SleepersSlot);

    internal static ulong PollersAddress => (ulong)(_state + PollersSlot);

    /// <summary>Allocates the shared state once and starts the 2 s counter line.</summary>
    public static bool TryInitialize(out ulong stateAddress)
    {
        lock (Gate)
        {
            if (_state == null)
            {
                if (!IsSupported)
                {
                    stateAddress = 0;
                    return false;
                }

                var words = (int*)NativeMemory.AlignedAlloc((nuint)(MaximumWorkers * WordStride * sizeof(int)), 64);
                var state = (long*)NativeMemory.AlignedAlloc(StateBytes, 64);
                if (words == null || state == null)
                {
                    NativeMemory.AlignedFree(words);
                    NativeMemory.AlignedFree(state);
                    stateAddress = 0;
                    return false;
                }

                NativeMemory.Clear(words, (nuint)(MaximumWorkers * WordStride * sizeof(int)));
                NativeMemory.Clear(state, StateBytes);
                _words = words;
                Interlocked.MemoryBarrier();
                _state = state;
                var seconds =
                    int.TryParse(Environment.GetEnvironmentVariable(ReportEnvironmentVariable), out var value) && value >= 0
                        ? value
                        : 2;
                if (seconds > 0)
                {
                    _reporter = new Timer(static _ => Report(), null, seconds * 1000, seconds * 1000);
                }
            }

            stateAddress = (ulong)_state;
            return true;
        }
    }

    internal enum WaitOutcome
    {
        /// <summary>The generation moved, or the probe found work, before the worker slept.</summary>
        NoBlock,

        /// <summary>A waker picked this worker.</summary>
        Woken,

        /// <summary>The timeout passed.</summary>
        TimedOut,
    }

    /// <summary>
    /// Blocks worker <paramref name="worker"/> until a waker picks it or
    /// <paramref name="timeoutMicroseconds"/> pass. Returns at once when the generation is no
    /// longer <paramref name="generation"/> (work was made ready since the caller read it) or
    /// when <paramref name="probe"/>, read after this worker is published as asleep, finds work
    /// for it. Without a probe the worker is also published as a poller.
    /// </summary>
    internal static WaitOutcome Wait<TProbe>(int worker, long generation, uint timeoutMicroseconds, TProbe? probe)
        where TProbe : struct, IJobWorkProbe
    {
        var state = _state;
        Interlocked.Increment(ref _waits);
        if (_abPeriodTicks != 0 && !IsWakePhase())
        {
            HostSleep(timeoutMicroseconds);
            Interlocked.Increment(ref _abNaps);
            return WaitOutcome.TimedOut;
        }

        if (state == null || (uint)worker >= MaximumWorkers)
        {
            HostSleep(timeoutMicroseconds);
            Interlocked.Increment(ref _timeouts);
            return WaitOutcome.TimedOut;
        }

        var bit = 1UL << worker;
        var word = _words + (worker * WordStride);
        var sequence = Volatile.Read(ref *word);
        Interlocked.Or(ref *(ulong*)(state + SleepersSlot), bit);
        if (probe is null)
        {
            Interlocked.Or(ref *(ulong*)(state + PollersSlot), bit);
        }

        if (Volatile.Read(ref state[GenerationSlot]) != generation ||
            (probe is { } p && (p.WorkMask() & bit) != 0))
        {
            return Finish(state, bit, WaitOutcome.NoBlock);
        }

        var timeoutNanoseconds = (long)timeoutMicroseconds * 1000;
        var start = Stopwatch.GetTimestamp();
        while (true)
        {
            var left = timeoutNanoseconds - ElapsedNanoseconds(start);
            if (left <= 0)
            {
                return Finish(state, bit, WaitOutcome.TimedOut);
            }

            _ = _ulockWait2(UL_COMPARE_AND_WAIT | ULF_NO_ERRNO, word, (uint)sequence, (ulong)left, 0);

            // Word first, then bit: a waker clears the bit before it bumps the word, so a wake
            // this check misses moves the word past the value the next wait compares against.
            // Re-reading also absorbs a late bump from a wake of an earlier wait.
            sequence = Volatile.Read(ref *word);
            if ((Volatile.Read(ref *(ulong*)(state + SleepersSlot)) & bit) == 0)
            {
                Interlocked.And(ref *(ulong*)(state + PollersSlot), ~bit);
                Interlocked.Increment(ref _woken);
                return WaitOutcome.Woken;
            }
        }
    }

    // Leaves the sleeper mask. A waker that cleared the bit first has already counted a wake.
    private static WaitOutcome Finish(long* state, ulong bit, WaitOutcome outcome)
    {
        Interlocked.And(ref *(ulong*)(state + PollersSlot), ~bit);
        var prior = Interlocked.And(ref *(ulong*)(state + SleepersSlot), ~bit);
        if ((prior & bit) == 0)
        {
            Interlocked.Increment(ref _woken);
            return WaitOutcome.Woken;
        }

        Interlocked.Increment(ref outcome == WaitOutcome.NoBlock ? ref _noBlock : ref _timeouts);
        return outcome;
    }

    /// <summary>
    /// Wakes up to <paramref name="count"/> sleeping workers in <paramref name="mask"/> that
    /// <paramref name="probe"/> says have work, starting at a rotating worker so the wakes
    /// spread. Returns how many it woke.
    /// </summary>
    internal static int Wake<TProbe>(ulong mask, long count, TProbe probe)
        where TProbe : struct, IJobWorkProbe
    {
        var state = _state;
        Interlocked.Increment(ref _wakeCalls);
        if (state == null || count <= 0)
        {
            return 0;
        }

        var candidates = Volatile.Read(ref *(ulong*)(state + SleepersSlot)) & mask;
        if (candidates == 0 || (candidates &= probe.WorkMask()) == 0)
        {
            return 0;
        }

        var rotate = (Interlocked.Increment(ref _wakeCursor) & int.MaxValue) % MaximumWorkers;
        candidates = BitOperations.RotateRight(candidates, rotate);
        var woken = 0;
        while (candidates != 0 && woken < count)
        {
            var worker = (BitOperations.TrailingZeroCount(candidates) + rotate) % MaximumWorkers;
            candidates &= candidates - 1;
            var bit = 1UL << worker;
            if ((Interlocked.And(ref *(ulong*)(state + SleepersSlot), ~bit) & bit) == 0)
            {
                continue; // it woke on its own, or another waker took it
            }

            // The pollers bit is the waiter's to clear: by now it may belong to its next wait.
            var word = _words + (worker * WordStride);
            Interlocked.Increment(ref *word);
            _ = _ulockWake(UL_COMPARE_AND_WAIT | ULF_NO_ERRNO, word, 0);
            woken++;
        }

        Interlocked.Add(ref _wakes, woken);
        return woken;
    }

    private static bool IsWakePhase()
    {
        var now = Stopwatch.GetTimestamp();
        var epoch = Volatile.Read(ref _abEpoch);
        if (epoch == 0)
        {
            var prior = Interlocked.CompareExchange(ref _abEpoch, now, 0);
            epoch = prior != 0 ? prior : now;
            if (prior == 0)
            {
                Console.Error.WriteLine(
                    $"[LOADER][INFO] job wake A/B: epoch_ts={now} period_s={_abPeriodTicks / Stopwatch.Frequency} " +
                    "(even phases wake, odd phases nap)");
            }
        }

        return IdleNapExports.IsNapPhase(now, epoch, _abPeriodTicks);
    }

    private static long ElapsedNanoseconds(long start) =>
        (long)((Stopwatch.GetTimestamp() - start) * (1_000_000_000.0 / Stopwatch.Frequency));

    private static void HostSleep(uint microseconds) => IdleNapExports.Sleep(microseconds);

    internal readonly record struct Counters(
        long Signals, long WakeCalls, long Wakes, long Waits, long Woken, long TimedOut, long NoBlock, long AbNaps = 0);

    internal static Counters Snapshot()
    {
        var state = _state;
        return new Counters(
            state == null ? 0 : Volatile.Read(ref state[GenerationSlot]),
            Interlocked.Read(ref _wakeCalls),
            Interlocked.Read(ref _wakes),
            Interlocked.Read(ref _waits),
            Interlocked.Read(ref _woken),
            Interlocked.Read(ref _timeouts),
            Interlocked.Read(ref _noBlock),
            Interlocked.Read(ref _abNaps));
    }

    internal static string Format(Counters now, Counters before) =>
        $"[LOADER][INFO] job wake: signals={now.Signals - before.Signals} wake_calls={now.WakeCalls - before.WakeCalls} " +
        $"woke={now.Wakes - before.Wakes} waits={now.Waits - before.Waits} woken_early={now.Woken - before.Woken} " +
        $"timed_out={now.TimedOut - before.TimedOut} no_block={now.NoBlock - before.NoBlock} ab_naps={now.AbNaps - before.AbNaps}";

    private static void Report()
    {
        var now = Snapshot();
        var before = _reported;
        _reported = now;
        if (now != before)
        {
            Console.Error.WriteLine(Format(now, before));
        }
    }
}
