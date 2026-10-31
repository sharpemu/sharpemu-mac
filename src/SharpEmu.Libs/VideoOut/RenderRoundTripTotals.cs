// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.VideoOut;

// Where a render-thread guest read came from, and the round-trip totals the hitch report reads.
// The Vulkan presenter records them; a backend that does not leaves them at zero.
internal enum RenderReadSource
{
    Wait,
    Operand,
    Memory,
    Word,
}

internal static class RoundTripTotals
{
    public const int ReadKinds = 8;
    public static long FlipCount;
    public static long FlipTicks;
    public static long ReadCount;
    public static long ReadTicks;
    public static readonly long[] KindCounts = new long[ReadKinds];
    public static readonly long[] KindTicks = new long[ReadKinds];
    private static long _slowestTicks;
    private static long _slowestAddress;
    private static int _slowestKind;
    private static string _slowestSite = "";

    public static string KindName(int kind) => ((RenderReadSource)(kind / 2)).ToString().ToLowerInvariant() + (kind % 2 == 0 ? "/publication" : "/gpu_dirty");

    public static void NoteRead(int kind, long ticks, ulong address, string site)
    {
        Interlocked.Increment(ref KindCounts[kind]);
        Interlocked.Add(ref KindTicks[kind], ticks);
        var slowest = Volatile.Read(ref _slowestTicks);
        while (ticks > slowest)
        {
            var previous = Interlocked.CompareExchange(ref _slowestTicks, ticks, slowest);
            if (previous == slowest)
            {
                Volatile.Write(ref _slowestAddress, (long)address);
                Volatile.Write(ref _slowestKind, kind);
                Volatile.Write(ref _slowestSite, site);
                break;
            }

            slowest = previous;
        }
    }

    public static (long Ticks, ulong Address, int Kind, string Site) TakeSlowest()
    {
        var address = (ulong)Volatile.Read(ref _slowestAddress);
        var kind = Volatile.Read(ref _slowestKind);
        var site = Volatile.Read(ref _slowestSite);
        return (Interlocked.Exchange(ref _slowestTicks, 0), address, kind, site);
    }
}
