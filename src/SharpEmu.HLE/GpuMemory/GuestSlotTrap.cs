// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Threading;

namespace SharpEmu.HLE.GpuMemory;

/// <summary>
/// Catches the instruction that writes one slot of a guest structure, when no managed
/// hook can see it because the store is native guest code. The page holding the slot is
/// write-protected through <see cref="PageGuard"/>; the resulting fault names the guest
/// RIP, and the watch is then released so the store lands and the run continues. A timer
/// re-arms it, so the trap keeps catching without single-stepping.
///
/// The slot is named by its offset inside an aligned block, not by its address: the block
/// moves with the title's allocation order and cannot be known before the run. Blocks are
/// adopted by the shape of their mapping, which is how a title's pool allocations are
/// recognised — <c>SHARPEMU_TRAP_BLOCK_BYTES</c> long (64 KiB) and mapped with
/// <c>SHARPEMU_TRAP_PROTECTION</c> (0xF2, read/write and GPU-visible).
///
/// Off unless <c>SHARPEMU_TRAP_SLOT=&lt;hex offset&gt;</c>.
/// </summary>
public static class GuestSlotTrap
{
    private const int MaxBlocks = 1024;
    private const int MaxReports = 64;

    // How long a released page stays writable before the trap takes it back. Long enough
    // that the faulting store and the handful that follow it make progress, short enough
    // that the trap is armed again well inside one frame.
    private static readonly TimeSpan RearmDelay = TimeSpan.FromMilliseconds(2);

    // SHARPEMU_TRAP_ADDRESS=<hex>: watch one known guest address instead of a slot of the
    // pool blocks. Its block is adopted at the first submission boundary after the address
    // is mapped, whatever the mapping's shape, and the slot is the address's offset in the
    // block unless SHARPEMU_TRAP_SLOT names one.
    private static readonly ulong FixedAddress = Parse("SHARPEMU_TRAP_ADDRESS");
    private static readonly ulong SlotMask = ParseOr("SHARPEMU_TRAP_SLOT_MASK", 0xFFFF);
    private static readonly ulong SlotOffset =
        ParseOr("SHARPEMU_TRAP_SLOT", FixedAddress & SlotMask);
    private static readonly ulong BlockBytes = ParseOr("SHARPEMU_TRAP_BLOCK_BYTES", 0x10000);
    private static readonly ulong Protection = ParseOr("SHARPEMU_TRAP_PROTECTION", 0xF2);
    private static readonly ulong SlotBytes = ParseOr("SHARPEMU_TRAP_SLOT_BYTES", 8);

    // SHARPEMU_TRAP_BAND_HI: adopt only mappings whose top 32 address bits match. Without
    // it the first blocks of the right shape take every slot, and a title maps hundreds
    // before the one that matters: the band is what points the trap at the right heap.
    private static readonly ulong BandHigh = Parse("SHARPEMU_TRAP_BAND_HI");

    // SHARPEMU_TRAP_VALUE: report only the faults that are about to store this value, as
    // told by the registers. The slot's legitimate traffic is the allocator pushing and
    // popping pointers; one value among them is the corruption, and without this filter it
    // is one line in thousands.
    private static readonly ulong TrapValue = Parse("SHARPEMU_TRAP_VALUE");

    // SHARPEMU_TRAP_PAGE_GUARD=0 adopts the blocks without write-protecting anything. A
    // page the CPU cannot write without faulting costs a signal per write, and a title
    // that maps a hundred and fifty pool blocks writes them constantly: the first run that
    // armed them all stalled the guest's own clock at 27 seconds. Watching the slots from
    // a submission boundary costs a few bytes per commit and does not perturb the run, so
    // the two halves are separable.
    private static readonly bool PageGuardEnabled =
        !string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_TRAP_PAGE_GUARD"), "0", StringComparison.Ordinal);

    // SHARPEMU_TRAP_UNIQUE_RIP=1: report each storing instruction once. A slot that a loop
    // rewrites every frame otherwise spends every report on the same store.
    private static readonly bool UniqueRips =
        string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_TRAP_UNIQUE_RIP"), "1", StringComparison.Ordinal);

    private static readonly ulong[] ReportedRips = new ulong[MaxReports];
    private static int _fixedAdopted;

    private const ulong PageBytes = 4096;

    private static readonly ulong[] BlockBases = new ulong[MaxBlocks];
    private static readonly bool[] BlockArmed = new bool[MaxBlocks];
    private static int _blockCount;
    private static long _faults;
    private static long _slotFaults;
    private static int _reports;
    private static long _rearmTicks;
    private static Timer? _rearmTimer;
    private static int _rearmPending;

    /// <summary>True while the trap is watching a slot.</summary>
    public static bool Armed => SlotOffset != 0 || SlotOffsetIsZeroButRequested || FixedAddress != 0;

    private static readonly bool SlotOffsetIsZeroButRequested =
        !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("SHARPEMU_TRAP_SLOT")) && SlotOffset == 0;

    /// <summary>How many faults the trap has taken on a watched page.</summary>
    public static long Faults => Interlocked.Read(ref _faults);

    /// <summary>How many of those faults were on the watched slot itself.</summary>
    public static long SlotFaults => Interlocked.Read(ref _slotFaults);

    /// <summary>The blocks the trap has adopted, for a caller that wants to watch them too.</summary>
    public static int CopyBlocks(Span<ulong> destination)
    {
        var count = Math.Min(Volatile.Read(ref _blockCount), destination.Length);
        for (var index = 0; index < count; index++)
        {
            destination[index] = Volatile.Read(ref BlockBases[index]);
        }

        return count;
    }

    /// <summary>The byte count of a trapped block.</summary>
    public static ulong BlockSize => BlockBytes;

    /// <summary>
    /// How many bytes from the slot a watcher that snapshots it should take. The slot's own
    /// neighbourhood by default: the whole block, times every adopted block, times every
    /// submission, is more copying than a diagnostic should cost, and a page-mate that
    /// changes is not the question being asked.
    /// </summary>
    public static ulong SnapshotBytes { get; } = ParseOr("SHARPEMU_TRAP_SNAPSHOT_BYTES", 64);

    /// <summary>The watched offset inside a block.</summary>
    public static ulong Slot => SlotOffset;

    /// <summary>The value being hunted, or zero when any change is interesting.</summary>
    public static ulong Value => TrapValue;

    /// <summary>
    /// The watched slot a guest range covers, if any. A producer that names its
    /// destination can be tested against every adopted slot as it records it, which is
    /// what lets a watcher keep the handful of entries that could explain a change
    /// instead of every binding a submission ever resolved.
    /// </summary>
    public static bool TryFindCoveredSlot(ulong address, ulong size, out ulong slot)
    {
        slot = 0;
        if (size == 0 || address > ulong.MaxValue - size)
        {
            return false;
        }

        var end = address + size;
        var count = Volatile.Read(ref _blockCount);
        for (var index = 0; index < count && index < MaxBlocks; index++)
        {
            var baseAddress = Volatile.Read(ref BlockBases[index]);
            if (baseAddress == 0)
            {
                continue;
            }

            var candidate = baseAddress + SlotOffset;
            if (candidate >= address && candidate < end)
            {
                slot = candidate;
                return true;
            }
        }

        return false;
    }

    /// <summary>Adopts a mapping whose shape matches the one the watched structure lives in.</summary>
    public static void OnDirectMapping(ulong mappedAddress, ulong length, int protection)
    {
        if (!Armed || FixedAddress != 0 || !MatchesShape(length, (ulong)protection, BlockBytes, Protection) ||
            !MatchesBand(mappedAddress, BandHigh))
        {
            return;
        }

        Adopt(mappedAddress);
    }

    private static void Adopt(ulong mappedAddress)
    {
        lock (BlockArmed)
        {
            var existing = Volatile.Read(ref _blockCount);
            for (var scan = 0; scan < existing && scan < MaxBlocks; scan++)
            {
                if (Volatile.Read(ref BlockBases[scan]) == mappedAddress)
                {
                    return;
                }
            }
        }

        var index = Interlocked.Increment(ref _blockCount) - 1;
        if (index >= MaxBlocks)
        {
            Interlocked.Decrement(ref _blockCount);
            return;
        }

        Volatile.Write(ref BlockBases[index], mappedAddress);
        if (index < 4 || !PageGuardEnabled)
        {
            Console.Error.WriteLine(
                $"[LOADER][WARN] slot_trap adopted 0x{mappedAddress + SlotOffset:X16} " +
                $"(block 0x{mappedAddress:X16}+0x{BlockBytes:X}, slot 0x{SlotOffset:X}, " +
                $"page_guard={(PageGuardEnabled ? 1 : 0)}, block #{index})");
            Console.Error.Flush();
        }
        ArmBlock(index);
        EnsureRearmTimer();
    }

    /// <summary>
    /// Handles a fault on a watched page: reports the guest instruction when the access is
    /// the slot's own, then releases the page so the store lands. False when the trap does
    /// not own this address, in which case the usual recovery chain continues.
    /// </summary>
    public static bool TryHandleFault(FaultKind kind, ulong address)
    {
        if (!Armed || kind != FaultKind.Write)
        {
            return false;
        }

        var index = FindBlockOfPage(address);
        if (index < 0)
        {
            return false;
        }

        Interlocked.Increment(ref _faults);
        if (HitsSlot(address, SlotOffset, SlotMask, SlotBytes))
        {
            Interlocked.Increment(ref _slotFaults);
            Report(address);
        }

        // A page the trap is not holding faulted for another watcher, which must resolve it:
        // claiming it here would retry a store that can never land.
        return ReleaseBlock(index);
    }

    /// <summary>Re-protects every page the trap released. Safe to call from any thread.</summary>
    public static void Rearm()
    {
        if (FixedAddress != 0 && Volatile.Read(ref _fixedAdopted) == 0 &&
            GuestGpuMemoryHook.Current?.Covers(FixedAddress, 1) == true &&
            Interlocked.Exchange(ref _fixedAdopted, 1) == 0)
        {
            Adopt(FixedAddress - SlotOffset);
        }

        if (!Armed || Interlocked.Exchange(ref _rearmPending, 0) == 0)
        {
            return;
        }

        // Taken back at once: a page left writable is a page whose writes the trap cannot
        // see, and the caller is a submission boundary, so this runs hundreds of times a
        // second without the trap ever having to single-step.


        var count = Volatile.Read(ref _blockCount);
        for (var index = 0; index < count && index < MaxBlocks; index++)
        {
            ArmBlock(index);
        }
    }

    private static void Report(ulong address)
    {
        var context = GuestFaultContext.Capture();
        // The value a store is about to write is in one of its registers. Reporting only
        // the faults that carry the watched value is what separates the corruption from
        // the allocator's own traffic through the same slot.
        var carries = TrapValue == 0 || context.CarriesValue(TrapValue);
        if (!carries || (UniqueRips && !FirstReportOf(context.Rip)) ||
            Interlocked.Increment(ref _reports) > MaxReports)
        {
            return;
        }

        Console.Error.WriteLine(
            $"[LOADER][WARN] slot_trap HIT dest=0x{address:X16} slot=0x{SlotOffset:X} " +
            $"rip=0x{context.Rip:X16} code={context.DescribeCode()}");
        Console.Error.WriteLine($"[LOADER][WARN] slot_trap registers {context.DescribeRegisters()}");
        Console.Error.Flush();
    }

    private static bool FirstReportOf(ulong rip)
    {
        for (var index = 0; index < ReportedRips.Length; index++)
        {
            var seen = Interlocked.CompareExchange(ref ReportedRips[index], rip, 0);
            if (seen == 0)
            {
                return true;
            }

            if (seen == rip)
            {
                return false;
            }
        }

        return false;
    }

    /// <summary>True when an access touches the watched slot rather than merely its page.</summary>
    internal static bool HitsSlot(ulong address, ulong slot, ulong slotMask, ulong slotBytes)
    {
        var offset = address & slotMask;
        return offset >= slot && offset < slot + slotBytes;
    }

    /// <summary>True when an address is in the watched band, or when no band was given.</summary>
    internal static bool MatchesBand(ulong address, ulong bandHigh) =>
        bandHigh == 0 || (address >> 32) == bandHigh;

    /// <summary>True when a mapping has the shape the watched structure lives in.</summary>
    internal static bool MatchesShape(ulong length, ulong protection, ulong blockBytes, ulong wantedProtection) =>
        length == blockBytes && protection == wantedProtection;

    /// <summary>The page a block's watched slot lives on.</summary>
    internal static ulong PageOfSlot(ulong blockBase, ulong slot) => (blockBase + slot) & ~(PageBytes - 1);

    private static int FindBlockOfPage(ulong address)
    {
        var count = Volatile.Read(ref _blockCount);
        for (var index = 0; index < count && index < MaxBlocks; index++)
        {
            var baseAddress = Volatile.Read(ref BlockBases[index]);
            if (baseAddress == 0)
            {
                continue;
            }

            var page = PageOfSlot(baseAddress, SlotOffset);
            if (address >= page && address < page + PageBytes)
            {
                return index;
            }
        }

        return -1;
    }

    private static void ArmBlock(int index)
    {
        var pages = GuestGpuMemoryHook.Current?.Pages;
        var baseAddress = Volatile.Read(ref BlockBases[index]);
        if (pages is null || baseAddress == 0)
        {
            return;
        }

        var page = PageOfSlot(baseAddress, SlotOffset);
        lock (BlockArmed)
        {
            // Only take a page that nothing else is holding: the watch counts are shared,
            // and an unbalanced change of them is fatal.
            if (!PageGuardEnabled || BlockArmed[index] || !pages.Allows(page, FaultKind.Write))
            {
                return;
            }

            BlockArmed[index] = true;
            pages.AddWatch(page, PageBytes, blockReads: false);
        }
    }

    private static bool ReleaseBlock(int index)
    {
        var pages = GuestGpuMemoryHook.Current?.Pages;
        var baseAddress = Volatile.Read(ref BlockBases[index]);
        if (pages is null || baseAddress == 0)
        {
            return false;
        }

        var page = PageOfSlot(baseAddress, SlotOffset);
        lock (BlockArmed)
        {
            if (!BlockArmed[index])
            {
                return false;
            }

            BlockArmed[index] = false;
            // Someone else may have dropped the watch with the mapping it belonged to.
            if (!pages.Allows(page, FaultKind.Write))
            {
                pages.RemoveWatch(page, PageBytes, blockReads: false);
            }
        }
        Volatile.Write(ref _rearmPending, 1);
        Volatile.Write(ref _rearmTicks, Environment.TickCount64);
        return true;
    }

    private static void EnsureRearmTimer()
    {
        if (_rearmTimer is not null)
        {
            return;
        }

        _rearmTimer = new Timer(
            _ =>
            {
                if (Environment.TickCount64 - Volatile.Read(ref _rearmTicks) >= RearmDelay.TotalMilliseconds)
                {
                    Rearm();
                }
            },
            null,
            RearmDelay,
            RearmDelay);
    }

    private static ulong Parse(string name) => GuestWriteWatch.Parse(Environment.GetEnvironmentVariable(name));

    private static ulong ParseOr(string name, ulong fallback)
    {
        var value = Parse(name);
        return value != 0 ? value : fallback;
    }
}
