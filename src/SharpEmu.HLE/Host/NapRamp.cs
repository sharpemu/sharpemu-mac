// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.HLE.Host;

/// <summary>
/// How long a polling guest thread naps as its idle streak lengthens: the nap grows from a
/// tenth of the cap to the cap instead of starting at it. A thread polling for a wait that is
/// about to end (a render thread joining the jobs it just queued) then oversleeps by tens of
/// microseconds, not the full cap, while a thread that really is idle reaches the cap within
/// a few milliseconds and costs what it did before.
/// </summary>
/// <remarks>
/// The streak age is measured from the streak's start and naps begin at <c>afterTicks</c>
/// (the idle threshold). The steps are 1/10, 1/5, 2/5 and the whole cap, entered at 1x, 2x, 4x
/// and 8x that threshold: 25, 50, 100 and 250 microseconds for the defaults of 250 microsecond
/// cap and 500 microsecond threshold, so the cap is reached after 4 ms of continuous polling.
/// SHARPEMU_NAP_RAMP=0 naps the cap from the first nap, as before.
/// </remarks>
public static class NapRamp
{
    public static readonly bool Enabled =
        !string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_NAP_RAMP"), "0", StringComparison.Ordinal);

    /// <summary>The numerators of the steps, over <see cref="Denominator"/>, in order.</summary>
    public static ReadOnlySpan<int> Numerators => [1, 2, 4, 10];

    public const int Denominator = 10;

    /// <summary>The nap, in microseconds, for a streak this old (<paramref name="ageTicks"/> is at
    /// least <paramref name="afterTicks"/> by the time anything naps).</summary>
    public static long Microseconds(long ageTicks, long afterTicks, long capMicroseconds) =>
        Microseconds(Enabled, ageTicks, afterTicks, capMicroseconds);

    internal static long Microseconds(bool enabled, long ageTicks, long afterTicks, long capMicroseconds)
    {
        if (!enabled || afterTicks <= 0 || capMicroseconds <= 0)
        {
            return capMicroseconds;
        }

        var level = 0;
        var step = afterTicks;
        while (level < Numerators.Length - 1 && ageTicks >= step * 2)
        {
            level++;
            step *= 2;
        }

        return Math.Max(1, capMicroseconds * Numerators[level] / Denominator);
    }
}
