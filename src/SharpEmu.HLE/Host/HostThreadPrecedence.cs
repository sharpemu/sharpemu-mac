// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;

namespace SharpEmu.HLE.Host;

/// <summary>
/// Scheduling precedence for the emulator's own pacing threads, so they win a core from the guest's
/// worker threads instead of queueing behind them.
///
/// A QoS class only orders a thread against threads of other classes. A guest job pool (Demon's
/// Souls runs 13 workers that poll for work) keeps every core busy at the default class, and the
/// emulator's one GPU submission thread, its completion thread and its clocks then compete with the
/// pool for cores (measured on this host: a CPU job against 12 spinning default threads took
/// 1.0-5.0 s, 0.25-0.29 s once the job's precedence was +6/+16). Two Mach policies order the
/// pacing threads above the pool, both honoured by threads CoreCLR created and by native QoS
/// threads alike:
/// <list type="bullet">
/// <item><c>THREAD_PRECEDENCE_POLICY</c> (<see cref="TryBoostCurrentThread"/>): adds an importance
/// to the thread's base priority. For the threads that do real work between waits (the GPU worker,
/// the completion and nudge threads).</item>
/// <item><c>THREAD_TIME_CONSTRAINT_POLICY</c> (<see cref="TryApplyTimeConstraintToCurrentThread"/>):
/// the real-time band, dispatched ahead of every timeshare thread with no timer slack: a timeshare
/// thread's 16.667 ms sleep lands 1.5 ms late, a real-time one within 10 us. For the threads that
/// only sleep, raise an event and sleep again (the vblank clock, the flip pacer, the guest timer).
/// </item>
/// </list>
/// Guest threads are not touched: the ones the game marks high priority keep the class
/// <see cref="GuestThreadQosPolicy"/> gives them, and the idle workers are quieted by the spin-loop
/// nap (a pure demotion of the whole pool collapsed the streaming phase, because the same workers
/// also do the work). SHARPEMU_HOST_PRECEDENCE=0 turns this class off (a no-op off macOS).
/// SHARPEMU_HOST_PRECEDENCE_PACING=&lt;n&gt; sets the boost (default 10, 0 to 20),
/// SHARPEMU_MAC_TIME_CONSTRAINT=0 keeps the clock threads timeshare.
/// </summary>
public static unsafe partial class HostThreadPrecedence
{
    public const string EnvironmentVariable = "SHARPEMU_HOST_PRECEDENCE";
    public const string PacingEnvironmentVariable = "SHARPEMU_HOST_PRECEDENCE_PACING";
    public const string TimeConstraintEnvironmentVariable = "SHARPEMU_MAC_TIME_CONSTRAINT";

    /// <summary>Default boost of a pacing thread: base priority 31 (default class) or 46 (user
    /// interactive) plus this stays under the timeshare ceiling (63), above every guest thread at
    /// default precedence and above the guest's own user-interactive threads.</summary>
    internal const int DefaultPacingImportance = 10;

    internal const int MaxPacingImportance = 20;

    private const int ThreadTimeConstraintPolicy = 2;
    private const uint ThreadTimeConstraintPolicyCount = 4;
    private const int ThreadPrecedencePolicy = 3;
    private const uint ThreadPrecedencePolicyCount = 1;

    // The clock threads wake once per millisecond at most and finish within 100 us.
    private const ulong TimeConstraintPeriodNanoseconds = 1_000_000;
    private const ulong TimeConstraintComputationNanoseconds = 100_000;

    private static readonly bool s_enabled = !string.Equals(
        Environment.GetEnvironmentVariable(EnvironmentVariable), "0", StringComparison.Ordinal);

    private static readonly bool s_timeConstraintEnabled = !string.Equals(
        Environment.GetEnvironmentVariable(TimeConstraintEnvironmentVariable), "0", StringComparison.Ordinal);

    private static readonly int s_pacingImportance =
        ParsePacingImportance(Environment.GetEnvironmentVariable(PacingEnvironmentVariable));

    public static bool IsEnabled => s_enabled && OperatingSystem.IsMacOS();

    /// <summary>The boost <see cref="TryBoostCurrentThread"/> applies.</summary>
    public static int PacingImportance => s_pacingImportance;

    internal static int ParsePacingImportance(string? text) =>
        int.TryParse(text, out var value) ? Math.Clamp(value, 0, MaxPacingImportance) : DefaultPacingImportance;

    /// <summary>
    /// Raises the calling thread above the default-precedence guest threads (see the class
    /// remarks). For a pacing thread that works between its waits. Logs one line so a run shows
    /// whether the kernel took it. False when disabled, off macOS, or refused.
    /// </summary>
    public static bool TryBoostCurrentThread(string threadName)
    {
        if (!IsEnabled || s_pacingImportance == 0)
        {
            return false;
        }

        var applied = TrySetImportance(s_pacingImportance);
        Console.Error.WriteLine(
            $"[SCHED][PREC] thread={threadName} importance=+{s_pacingImportance} ok={applied}");
        return applied;
    }

    /// <summary>
    /// Puts the calling thread in the real-time band (1 ms period, 100 us computation, preemptible):
    /// for a thread that only sleeps, raises an event and sleeps again. The kernel demotes a
    /// real-time thread that keeps running past its budget, so a longer pass costs nothing worse
    /// than the timeshare behaviour. Replaces the thread's QoS class and precedence.
    /// </summary>
    public static bool TryApplyTimeConstraintToCurrentThread(string threadName)
    {
        if (!IsEnabled || !s_timeConstraintEnabled)
        {
            return false;
        }

        var applied = false;
        try
        {
            MachTimebase timebase;
            if (mach_timebase_info(&timebase) == 0 && timebase.Numer != 0 && timebase.Denom != 0)
            {
                var policy = new TimeConstraintPolicy
                {
                    Period = ToAbsoluteTime(TimeConstraintPeriodNanoseconds, timebase),
                    Computation = ToAbsoluteTime(TimeConstraintComputationNanoseconds, timebase),
                    Constraint = ToAbsoluteTime(TimeConstraintPeriodNanoseconds, timebase),
                    Preemptible = 1,
                };
                applied = thread_policy_set(
                    pthread_mach_thread_np(pthread_self()), ThreadTimeConstraintPolicy, &policy, ThreadTimeConstraintPolicyCount) == 0;
            }
        }
        catch (Exception)
        {
            // A scheduling hint never aborts a thread's start-up.
        }

        Console.Error.WriteLine($"[SCHED][PREC] thread={threadName} time_constraint={(applied ? "on" : "failed")}");
        return applied;
    }

    /// <summary>The calling thread's precedence importance as the kernel reports it (diagnostics and
    /// tests); null off macOS or when the query fails.</summary>
    internal static int? GetCurrentImportance()
    {
        if (!OperatingSystem.IsMacOS())
        {
            return null;
        }

        try
        {
            int importance = 0;
            var count = ThreadPrecedencePolicyCount;
            var getDefault = 0;
            return thread_policy_get(
                pthread_mach_thread_np(pthread_self()), ThreadPrecedencePolicy, &importance, &count, &getDefault) == 0
                ? importance
                : null;
        }
        catch (Exception)
        {
            return null;
        }
    }

    private static bool TrySetImportance(int importance)
    {
        try
        {
            return thread_policy_set(
                pthread_mach_thread_np(pthread_self()), ThreadPrecedencePolicy, &importance, ThreadPrecedencePolicyCount) == 0;
        }
        catch (Exception)
        {
            return false;
        }
    }

    internal static uint ToAbsoluteTime(ulong nanoseconds, in MachTimebase timebase) =>
        (uint)(nanoseconds * timebase.Denom / timebase.Numer);

    [StructLayout(LayoutKind.Sequential)]
    internal struct MachTimebase
    {
        public uint Numer;
        public uint Denom;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct TimeConstraintPolicy
    {
        public uint Period;
        public uint Computation;
        public uint Constraint;
        public int Preemptible;
    }

    [LibraryImport("libSystem.B.dylib")]
    private static partial int mach_timebase_info(MachTimebase* info);

    [LibraryImport("libSystem.B.dylib")]
    private static partial nint pthread_self();

    [LibraryImport("libSystem.B.dylib")]
    private static partial uint pthread_mach_thread_np(nint thread);

    [LibraryImport("libSystem.B.dylib")]
    private static partial int thread_policy_set(uint thread, int flavor, void* policyInfo, uint count);

    [LibraryImport("libSystem.B.dylib")]
    private static partial int thread_policy_get(uint thread, int flavor, void* policyInfo, uint* count, int* getDefault);
}
