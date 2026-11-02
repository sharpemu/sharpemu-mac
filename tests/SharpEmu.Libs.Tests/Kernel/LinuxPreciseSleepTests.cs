// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Runtime.InteropServices;
using SharpEmu.HLE.Host;
using Xunit;

namespace SharpEmu.Libs.Tests.Kernel;

/// <summary>
/// Linux's precise sleep (clock_nanosleep to an absolute CLOCK_MONOTONIC deadline) and the
/// SHARPEMU_LINUX_PRECISE_SLEEP=0 coarse fallback. Bounds are generous on purpose: the point is
/// "never early" and "a sleeping thread is not spinning", not a wakeup-latency budget.
/// </summary>
public sealed class LinuxPreciseSleepTests
{
    [StructLayout(LayoutKind.Sequential)]
    private struct Timespec
    {
        public long Seconds;
        public long Nanoseconds;
    }

    [DllImport("libc", EntryPoint = "clock_gettime")]
    private static extern int clock_gettime(int clockId, out Timespec time);

    private static long ThreadCpuNanoseconds()
    {
        Assert.Equal(0, clock_gettime(3 /* CLOCK_THREAD_CPUTIME_ID */, out var ts));
        return ts.Seconds * 1_000_000_000L + ts.Nanoseconds;
    }

    private static bool PreciseOnThisHost => OperatingSystem.IsLinux() && HostPreciseClock.IsLinuxPrecise;

    [Fact]
    public void OnlyAnExplicitZeroTurnsTheLinuxPreciseSleepOff()
    {
        Assert.False(HostPreciseClock.LinuxPreciseSleepEnabled("0"));
        Assert.False(HostPreciseClock.LinuxPreciseSleepEnabled(" 0 "));
        Assert.True(HostPreciseClock.LinuxPreciseSleepEnabled(null));
        Assert.True(HostPreciseClock.LinuxPreciseSleepEnabled(""));
        Assert.True(HostPreciseClock.LinuxPreciseSleepEnabled("1"));
    }

    [Fact]
    public void LinuxPreciseSleepIsSelectedOnLinuxUnlessTurnedOff()
    {
        if (!OperatingSystem.IsLinux())
        {
            Assert.False(HostPreciseClock.IsLinuxPrecise);
            return;
        }

        var off = !HostPreciseClock.LinuxPreciseSleepEnabled(
            Environment.GetEnvironmentVariable("SHARPEMU_LINUX_PRECISE_SLEEP"));
        Assert.Equal(!off, HostPreciseClock.IsLinuxPrecise);
        Assert.False(HostPreciseClock.IsPrecise); // the mach flag stays macOS only
        Assert.Equal(off ? "off" : "absolute", HostPreciseClock.LinuxClockMapping);
    }

    [Fact]
    public void StopwatchTimestampsAreClockMonotonicNanosecondsOnLinux()
    {
        if (!OperatingSystem.IsLinux())
        {
            return;
        }

        Assert.Equal(1_000_000_000L, Stopwatch.Frequency);
        var before = Stopwatch.GetTimestamp();
        Assert.Equal(0, clock_gettime(1 /* CLOCK_MONOTONIC */, out var ts));
        var after = Stopwatch.GetTimestamp();
        var monotonic = ts.Seconds * 1_000_000_000L + ts.Nanoseconds;
        Assert.InRange(monotonic, before - 1000, after + 1000);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(50)]
    [InlineData(500)]
    [InlineData(2_500)]
    public void SleepMicrosecondsNeverReturnsEarly(long microseconds)
    {
        if (!PreciseOnThisHost)
        {
            return;
        }

        for (var i = 0; i < 20; i++)
        {
            var before = Stopwatch.GetTimestamp();
            HostTiming.SleepMicroseconds(microseconds);
            var tookUs = HostPreciseClock.TicksToMicroseconds(Stopwatch.GetTimestamp() - before);
            Assert.True(tookUs >= microseconds, $"sleep({microseconds}) returned after {tookUs} us");
        }
    }

    [Fact]
    public void SleepUntilNeverReturnsBeforeItsDeadlineAndAPastDeadlineReturnsAtOnce()
    {
        if (!PreciseOnThisHost)
        {
            return;
        }

        var deadline = HostPreciseClock.DeadlineAfterMicroseconds(1_500);
        HostTiming.SleepUntil(deadline);
        Assert.True(Stopwatch.GetTimestamp() >= deadline);

        var past = Stopwatch.GetTimestamp() - HostPreciseClock.MicrosecondsToTicks(10_000);
        var start = Stopwatch.GetTimestamp();
        HostTiming.SleepUntil(past);
        Assert.True(HostPreciseClock.TicksToMicroseconds(Stopwatch.GetTimestamp() - start) < 5_000);
    }

    [Fact]
    public void ASubMillisecondSleepLoopMostlyDoesNotBurnCpu()
    {
        if (!PreciseOnThisHost)
        {
            return;
        }

        // Every thread that sleeps gets its own timer slack on its first precise sleep.
        var result = (wallNs: 0L, cpuNs: 0L);
        var worker = new Thread(() =>
        {
            HostTiming.SleepMicroseconds(100); // warm up: first-call JIT, timer slack
            var cpu0 = ThreadCpuNanoseconds();
            var wall0 = Stopwatch.GetTimestamp();
            for (var i = 0; i < 200; i++)
            {
                HostTiming.SleepMicroseconds(500);
            }

            result = (
                (Stopwatch.GetTimestamp() - wall0) * 1_000_000_000L / Stopwatch.Frequency,
                ThreadCpuNanoseconds() - cpu0);
        });
        worker.Start();
        worker.Join();

        Assert.True(result.wallNs >= 200 * 500_000L, $"wall {result.wallNs} ns");
        // The coarse path spins (about 140% of wall); a real sleep is a few percent.
        Assert.True(result.cpuNs < result.wallNs / 4, $"cpu {result.cpuNs} ns over wall {result.wallNs} ns");
    }

    [Fact]
    public void TheCoarseFallbackStaysSelectableAndNeverReturnsEarly()
    {
        foreach (var microseconds in new long[] { 50, 500, 1_500, 3_000 })
        {
            var deadline = HostPreciseClock.DeadlineAfterMicroseconds(microseconds);
            HostTiming.SleepUntilCoarse(deadline);
            Assert.True(Stopwatch.GetTimestamp() >= deadline, $"coarse sleep({microseconds}) returned early");
        }
    }
}
