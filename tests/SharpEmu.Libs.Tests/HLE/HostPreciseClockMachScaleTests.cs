// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE.Host;
using Xunit;

namespace SharpEmu.Libs.Tests.HLE;

public sealed class HostPreciseClockMachScaleTests
{
    // Stopwatch.Frequency on macOS, Linux, Windows, and a few odd values.
    private static readonly long[] Frequencies = [1_000_000_000, 10_000_000, 1_000_000, 3_579_545, 24_000_000, 1_000, 999_999_937];

    // Rosetta/Intel, Apple silicon, and other plausible timebases.
    private static readonly (uint Numer, uint Denom)[] Timebases =
        [(1, 1), (125, 3), (3, 125), (1, 3), (7, 5), (1_000_000_007, 3), (4_294_967_295, 4_294_967_294), (4_294_967_295, 1)];

    // The formulation SleepUntil used before the integer scale.
    private static ulong DecimalReference(long remaining, long frequency, uint numer, uint denom)
    {
        var nanoseconds = (long)((decimal)remaining * 1_000_000_000m / frequency);
        return (ulong)((decimal)nanoseconds * denom / numer);
    }

    private static IEnumerable<long> Inputs(long frequency, System.Random random)
    {
        var max = frequency * 30;
        long[] edges = [1, 2, 3, 124, 125, 126, 1_000, frequency - 1, frequency, frequency + 1, max / 2, max - 2, max - 1, max];
        foreach (var edge in edges)
        {
            if (edge > 0 && edge <= max)
            {
                yield return edge;
            }
        }

        for (var i = 0; i < 2000; i++)
        {
            yield return random.NextInt64(1, max + 1);
        }

        for (var i = 0; i < 500; i++)
        {
            yield return random.NextInt64(1, Math.Min(max, 100_000) + 1);
        }
    }

    [Fact]
    public void IntegerConversionMatchesTheDecimalReference()
    {
        var random = new System.Random(0x5EED);
        foreach (var frequency in Frequencies)
        {
            foreach (var (numer, denom) in Timebases)
            {
                foreach (var remaining in Inputs(frequency, random))
                {
                    Assert.Equal(
                        DecimalReference(remaining, frequency, numer, denom),
                        HostPreciseClock.StopwatchTicksToMachTicks(remaining, frequency, numer, denom));
                }
            }
        }
    }

    [Fact]
    public void ExactMultiplesAndTheirNeighboursFloorLikeTheReference()
    {
        // Boundaries where a single-step (non-nested) floor would disagree with the two-step one.
        foreach (var frequency in Frequencies)
        {
            foreach (var (numer, denom) in Timebases)
            {
                for (long k = 1; k <= 400; k++)
                {
                    foreach (var remaining in new[] { k * frequency / 1_000_000 - 1, k * frequency / 1_000_000, k * frequency / 1_000_000 + 1 })
                    {
                        if (remaining > 0)
                        {
                            Assert.Equal(
                                DecimalReference(remaining, frequency, numer, denom),
                                HostPreciseClock.StopwatchTicksToMachTicks(remaining, frequency, numer, denom));
                        }
                    }
                }
            }
        }
    }

    [Fact]
    public void UnitTimebaseAtNanosecondFrequencyIsTheIdentity()
    {
        Assert.Equal(123_456_789UL, HostPreciseClock.StopwatchTicksToMachTicks(123_456_789, 1_000_000_000, 1, 1));
    }

    [Fact]
    public void DegenerateInputsConvertToZero()
    {
        Assert.Equal(0UL, HostPreciseClock.StopwatchTicksToMachTicks(0, 1_000_000_000, 1, 1));
        Assert.Equal(0UL, HostPreciseClock.StopwatchTicksToMachTicks(5, 1_000_000_000, 0, 0));
        Assert.Equal(0UL, default(HostPreciseClock.MachScale).Convert(5));
    }
}
