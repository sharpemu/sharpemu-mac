// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;

namespace SharpEmu.Libs.Gpu.GpuCommands;

/// <summary>
/// COND_EXEC flags that recorded GPU work still owes, taken as zero instead of parking the queue.
/// SHARPEMU_COND_EXEC_SPECULATE=0 parks the queue on them again.
/// </summary>
/// <remarks>
/// Ghost of Tsushima's graphics queue guards a GPU-to-CPU request (an interrupting RELEASE_MEM and a
/// wait for the CPU's reply) with a flag a compute shader writes just before. Reading the flag parks
/// the queue until the GPU has drained everything recorded ahead of it, about a third of the frame,
/// and the flag has never been seen set. A flag address that read zero <see cref="Streak"/> times in
/// a row is taken as zero while it is owed: the guarded block is skipped and the host checks the
/// flag when the work that writes it completes. A flag found set then is reported, and that address
/// is never taken as zero again; the skipped request itself is lost. Ghost of Tsushima at the gorge:
/// 6.49 -> 6.99 and 6.98 median fps (got-sp0 / got-sp1 / got-sp1b), no flag found set, same frames.
/// </remarks>
public static class ConditionalExecuteSpeculation
{
    private static int _enabled = -1;

    /// <summary>Read at the first COND_EXEC.</summary>
    public static bool Enabled
    {
        get
        {
            if (_enabled < 0)
            {
                var titleId = SharpEmu.Libs.VideoOut.VideoOutBackendServices.GetApplicationTitleId();
                var enabled = Environment.GetEnvironmentVariable("SHARPEMU_COND_EXEC_SPECULATE") switch
                {
                    "1" => true,
                    "0" => false,
                    _ => true,
                };
                Console.Error.WriteLine($"[GPU][INFO] COND_EXEC flags owed by the GPU taken as zero: {(enabled ? "on" : "off")} (title {titleId}).");
                _enabled = enabled ? 1 : 0;
            }

            return _enabled == 1;
        }
    }

    private const int Streak = 32;
    private const int Never = -1;

    private static readonly ConcurrentDictionary<ulong, int> ZeroStreaks = new();
    private static long _speculated;
    private static long _mispredicted;

    /// <summary>COND_EXECs skipped on a flag still owed since the last call.</summary>
    public static long TakeSpeculated() => Interlocked.Exchange(ref _speculated, 0);

    /// <summary>Flags since startup that were found set after being taken as zero.</summary>
    public static long Mispredicted => Interlocked.Read(ref _mispredicted);

    public static bool ShouldTakeAsZero(ulong address) =>
        Enabled && ZeroStreaks.TryGetValue(address, out var streak) && streak >= Streak;

    /// <summary>A COND_EXEC read this flag value the ordinary way.</summary>
    public static void Observe(ulong address, uint value)
    {
        if (!Enabled)
        {
            return;
        }

        ZeroStreaks.AddOrUpdate(address, value == 0 ? 1 : Never,
            (_, streak) => streak == Never || value != 0 ? Never : Math.Min(streak + 1, Streak));
    }

    public static void NoteTakenAsZero() => Interlocked.Increment(ref _speculated);

    /// <summary>The work that owed a flag taken as zero has completed and left this value.</summary>
    public static void Verify(ulong address, uint value)
    {
        if (value == 0)
        {
            return;
        }

        ZeroStreaks[address] = Never;
        if (Interlocked.Increment(ref _mispredicted) <= 16)
        {
            Console.Error.WriteLine(
                $"[GPU][WARN] A COND_EXEC flag taken as zero was set: address=0x{address:X16} value=0x{value:X8}; " +
                "its guarded block was skipped, and the flag is read the ordinary way from now on.");
        }
    }
}
