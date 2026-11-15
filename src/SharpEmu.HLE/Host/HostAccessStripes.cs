// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.HLE.Host;

/// <summary>
/// In-flight counts split over cache-line-padded stripes, so threads that enter and exit at the
/// same time each change their own line instead of one shared counter. A thread uses the stripe
/// its managed thread id selects. Each stripe has <see cref="Lanes"/> independent counts on the
/// same line (one per kind of access a drainer waits for).
///
/// Protocol (Dekker, both sides fully fenced): an entrant increments its stripe (a full fence),
/// then re-reads the drainer's flag and backs out if it is set. A drainer sets its flag with a
/// full fence, then sums the stripes. Either the drainer sees the increment or the entrant sees
/// the flag. The sum is signed, so an exit on another thread than its enter still balances;
/// a negative sum means an exit without an enter and is fatal.
/// </summary>
internal sealed class HostAccessStripes
{
    internal const int Lanes = 2;
    internal const int MaximumStripes = 64;
    // 256 bytes per stripe: whatever the array's alignment, no 128-byte line holds two stripes,
    // and one stripe-width of padding before the first and after the last keeps the array
    // header and neighbouring heap objects off the stripes' lines.
    private const int StrideLongs = 32;

    private readonly long[] _cells;
    private readonly int _mask;

    // Default-null controlled fatal path; production fails fast on an unbalanced exit.
    internal Action<string>? OnFatalForControl;

    internal HostAccessStripes() : this(DefaultStripeCount(Environment.ProcessorCount))
    {
    }

    internal HostAccessStripes(int stripes)
    {
        if (stripes <= 0 || stripes > MaximumStripes || (stripes & (stripes - 1)) != 0)
            throw new ArgumentOutOfRangeException(nameof(stripes));
        Count = stripes;
        _mask = stripes - 1;
        _cells = new long[(stripes + 2) * StrideLongs];
    }

    internal int Count { get; }

    /// <summary>A power of two of at least twice the processor count, at most <see cref="MaximumStripes"/>.</summary>
    internal static int DefaultStripeCount(int processors)
    {
        var wanted = Math.Clamp(processors, 1, MaximumStripes / 2) * 2;
        var stripes = 1;
        while (stripes < wanted) stripes <<= 1;
        return Math.Min(stripes, MaximumStripes);
    }

    /// <summary>The cell of the calling thread's stripe for <paramref name="lane"/>.</summary>
    internal int CellForCurrentThread(int lane) => CellOf(Environment.CurrentManagedThreadId & _mask, lane);

    internal int CellOf(int stripe, int lane)
    {
        if ((uint)stripe > (uint)_mask || (uint)lane >= Lanes) throw new ArgumentOutOfRangeException(nameof(stripe));
        return (stripe + 1) * StrideLongs + lane;
    }

    /// <summary>Counts one entry on the calling thread's stripe (a full fence) and returns its cell.</summary>
    internal int Enter(int lane)
    {
        var cell = CellForCurrentThread(lane);
        Interlocked.Increment(ref _cells[cell]);
        return cell;
    }

    /// <summary>Counts one exit on the calling thread's stripe.</summary>
    internal void Exit(int lane) => Interlocked.Decrement(ref _cells[CellForCurrentThread(lane)]);

    /// <summary>Counts one exit on a cell an <see cref="Enter"/> returned.</summary>
    internal void ExitCell(int cell)
    {
        if ((uint)cell >= (uint)_cells.Length || cell % StrideLongs >= Lanes || cell < StrideLongs)
            throw new ArgumentOutOfRangeException(nameof(cell));
        Interlocked.Decrement(ref _cells[cell]);
    }

    /// <summary>
    /// The signed sum of one lane over every stripe. Callers set their flag with a full fence first.
    /// A negative sum (an exit with no enter) is fatal.
    /// </summary>
    internal long Sum(int lane)
    {
        if ((uint)lane >= Lanes) throw new ArgumentOutOfRangeException(nameof(lane));
        long sum = 0;
        for (var stripe = 0; stripe < Count; stripe++) sum += Volatile.Read(ref _cells[(stripe + 1) * StrideLongs + lane]);
        if (sum < 0)
        {
            var message = $"Unbalanced access exit: stripe lane {lane} sums to {sum}.";
            if (OnFatalForControl is { } fatal) fatal(message);
            else Environment.FailFast(message);
        }

        return sum;
    }
}
