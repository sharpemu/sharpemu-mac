// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Runtime.InteropServices;
using SharpEmu.HLE;

namespace SharpEmu.Libs.Kernel;

/// <summary>
/// A short host sleep for guest code that SharpEmu's own title patches insert, such as a job
/// worker that has found no work for a while (see DemonsSoulsIdleNapPatch). The guest reaches
/// it through an import stub the loader adds for <see cref="Nid"/>, which no console library
/// exports. sceKernelUsleep cannot serve here: hosts turn its sub-millisecond requests into a
/// yield, which leaves the caller runnable and competing for a core.
/// </summary>
public static partial class IdleNapExports
{
    /// <summary>SharpEmu's own NID: eleven characters of the NID alphabet that no console
    /// symbol hashes to in practice.</summary>
    public const string Nid = "SharpEmuNap";

    public const string AbEnvironmentVariable = "SHARPEMU_DS_IDLE_NAP_AB_S";

    // SHARPEMU_DS_IDLE_NAP_AB_S=N (diagnostic): naps alternate with returning at once (the
    // guest's own spin) every N seconds from the first nap, so one run carries interleaved
    // A/B phases that see the same scenes.
    private static readonly long _abPeriodTicks =
        int.TryParse(Environment.GetEnvironmentVariable(AbEnvironmentVariable), out var seconds) && seconds > 0
            ? seconds * Stopwatch.Frequency
            : 0;

    private static long _abEpoch;

    [SysAbiExport(
        Nid = Nid,
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libKernel")]
    public static int IdleNap(CpuContext ctx)
    {
        var microseconds = (uint)ctx[CpuRegister.Rdi];
        if (microseconds != 0 && (_abPeriodTicks == 0 || IsNapPhase()))
        {
            Sleep(microseconds);
        }

        ctx[CpuRegister.Rax] = 0;
        return (int)OrbisGen2Result.ORBIS_GEN2_OK;
    }

    /// <summary>Blocks the calling thread for about <paramref name="microseconds"/>.</summary>
    internal static void Sleep(uint microseconds)
    {
        if (OperatingSystem.IsWindows())
        {
            // Sleep counts whole timer periods; with timeBeginPeriod(1) a sub-millisecond
            // nap becomes about 1 ms, which an idle worker can afford.
            Thread.Sleep((int)Math.Max(1, (microseconds + 999) / 1000));
            return;
        }

        _ = HostUsleep(microseconds);
    }

    internal static bool IsNapPhase(long now, long epoch, long periodTicks) =>
        (now - epoch) / periodTicks % 2 == 0;

    private static bool IsNapPhase()
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
                    $"[LOADER][INFO] idle nap A/B: epoch_ts={now} period_s={_abPeriodTicks / Stopwatch.Frequency} (even phases nap)");
            }
        }

        return IsNapPhase(now, epoch, _abPeriodTicks);
    }

    [LibraryImport("libc", EntryPoint = "usleep")]
    private static partial int HostUsleep(uint microseconds);
}
