// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.Diagnostics;

/// <summary>
/// SHARPEMU_PERF_HLE=1 only: what the calls that reach two hot managed exports look like, to size
/// the native stubs that could answer them. The guest mutex lock, trylock and unlock: how the
/// mutex argument resolved (the slot of a mutex scePthreadMutexInit made, the slot of one made
/// implicitly, an alias of another slot or handle, created on this call from the static
/// initialiser or a zero slot, or not resolved) and how many took the uncontended path.
/// sceAgcCbSetShRegistersDirect: everything its native stub declines (more registers than the
/// stub takes, unsorted, duplicate offsets, more than one run, a destination overlapping the
/// source, no room in the buffer). Printed with the [PERF][HLE] lines.
/// </summary>
internal static class HleCallShapeProfile
{
    internal static readonly bool Enabled =
        string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_PERF_HLE"), "1", StringComparison.Ordinal);

    internal enum MutexOp { Lock, Trylock, Unlock, Count }

    internal enum MutexResolve { ExplicitSlot, ImplicitSlot, Aliased, CreatedStatic, CreatedZero, Missing, Count }

    internal enum ShRegistersOutcome { Written, Invalid, ReadFailed, NoRoom, WriteFailed, Count }

    private static readonly long[] _mutexResolves = new long[(int)MutexOp.Count * (int)MutexResolve.Count];
    private static readonly long[] _mutexUncontended = new long[(int)MutexOp.Count];

    private static readonly long[] _shOutcomes = new long[(int)ShRegistersOutcome.Count];
    private static long _shCalls;
    private static long _shOverStubLimit;
    private static long _shUnsorted;
    private static long _shDuplicates;
    private static long _shSourceOverlap;
    private static long _shStubShaped;
    // Runs per call: 1, 2, 3-4, 5-8, more.
    private static readonly long[] _shRuns = new long[5];

    internal static void RecordMutexResolve(MutexOp op, MutexResolve resolve) =>
        Interlocked.Increment(ref _mutexResolves[(int)op * (int)MutexResolve.Count + (int)resolve]);

    internal static void RecordMutexUncontended(MutexOp op) => Interlocked.Increment(ref _mutexUncontended[(int)op]);

    internal static void RecordShRegisters(
        uint count, bool unsorted, bool duplicates, int runs, bool sourceOverlap, ShRegistersOutcome outcome)
    {
        Interlocked.Increment(ref _shCalls);
        Interlocked.Increment(ref _shOutcomes[(int)outcome]);
        if (outcome is ShRegistersOutcome.Invalid or ShRegistersOutcome.ReadFailed) return;
        if (count > Agc.AgcExports.RegistersStubMaxRegisters) Interlocked.Increment(ref _shOverStubLimit);
        if (unsorted) Interlocked.Increment(ref _shUnsorted);
        if (duplicates) Interlocked.Increment(ref _shDuplicates);
        if (sourceOverlap) Interlocked.Increment(ref _shSourceOverlap);
        if (runs > 0) Interlocked.Increment(ref _shRuns[runs switch { 1 => 0, 2 => 1, <= 4 => 2, <= 8 => 3, _ => 4 }]);
        // The shape the stub answers; such a call reaching the export was declined for something else.
        if (count <= Agc.AgcExports.RegistersStubMaxRegisters && !unsorted && !duplicates && runs == 1 && !sourceOverlap)
            Interlocked.Increment(ref _shStubShaped);
    }

    internal static IEnumerable<string> FormatLines()
    {
        var mutex = new System.Text.StringBuilder("[PERF][HLE] mutex:");
        for (var op = 0; op < (int)MutexOp.Count; op++)
        {
            long total = 0;
            var parts = new System.Text.StringBuilder();
            for (var resolve = 0; resolve < (int)MutexResolve.Count; resolve++)
            {
                var value = Interlocked.Read(ref _mutexResolves[op * (int)MutexResolve.Count + resolve]);
                total += value;
                parts.Append(' ').Append(ToSnake((MutexResolve)resolve)).Append('=').Append(value);
            }

            mutex.Append(' ').Append(((MutexOp)op).ToString().ToLowerInvariant())
                .Append("{total=").Append(total)
                .Append(" uncontended=").Append(Interlocked.Read(ref _mutexUncontended[op]))
                .Append(parts).Append('}');
        }

        yield return mutex.ToString();
        yield return
            $"[PERF][HLE] sh_registers_direct (calls reaching the export): calls={Interlocked.Read(ref _shCalls)} " +
            $"written={Interlocked.Read(ref _shOutcomes[(int)ShRegistersOutcome.Written])} " +
            $"invalid={Interlocked.Read(ref _shOutcomes[(int)ShRegistersOutcome.Invalid])} " +
            $"read_failed={Interlocked.Read(ref _shOutcomes[(int)ShRegistersOutcome.ReadFailed])} " +
            $"no_room={Interlocked.Read(ref _shOutcomes[(int)ShRegistersOutcome.NoRoom])} " +
            $"write_failed={Interlocked.Read(ref _shOutcomes[(int)ShRegistersOutcome.WriteFailed])} " +
            $"over_{Agc.AgcExports.RegistersStubMaxRegisters}={Interlocked.Read(ref _shOverStubLimit)} " +
            $"unsorted={Interlocked.Read(ref _shUnsorted)} duplicates={Interlocked.Read(ref _shDuplicates)} " +
            $"source_overlap={Interlocked.Read(ref _shSourceOverlap)} stub_shaped={Interlocked.Read(ref _shStubShaped)} " +
            $"runs{{1={Interlocked.Read(ref _shRuns[0])} 2={Interlocked.Read(ref _shRuns[1])} 3-4={Interlocked.Read(ref _shRuns[2])} " +
            $"5-8={Interlocked.Read(ref _shRuns[3])} 9+={Interlocked.Read(ref _shRuns[4])}}}";
    }

    private static string ToSnake(MutexResolve resolve) => resolve switch
    {
        MutexResolve.ExplicitSlot => "explicit",
        MutexResolve.ImplicitSlot => "implicit",
        MutexResolve.Aliased => "aliased",
        MutexResolve.CreatedStatic => "created_static",
        MutexResolve.CreatedZero => "created_zero",
        _ => "missing",
    };
}
