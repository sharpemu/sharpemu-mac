// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.Kernel;

// macOS keeps [0x10_0000_0000, 0x70_0000_0000) for graphics memory, which is the first
// 0x60_0000_0000 bytes of the PS5 app map area. Direct execution uses guest addresses as host
// addresses, so nothing can live there: kernel-chosen placements search around the window on
// every macOS, and from macOS 27 on, where the host refuses even fixed mappings in it, a guest
// that maps at a fixed address inside the window is served above it instead.
//
// The two workarounds share one shift. The load-time code patch (GuestVaHolePatcher) moves the
// hardcoded arena bases it recognises by Shift; the kernel moves every other guest-supplied
// address inside the window by the same Shift at each memory call. A title whose base was patched
// asks for the moved address directly and never touches the window, a title that maps fixed at X
// and uses the address it gets back works at X + Shift, and a title that keeps computing addresses
// from a base it never changed (Ghost of Tsushima streaming into an arena the patcher moved) lands
// exactly where the moved arena is. Kernel-chosen placements avoid the shadow [Start + Shift,
// End + Shift) while there is room elsewhere, so a later relocated request does not replace a live
// mapping. A range that only partly lies in the window has no single translation and is still
// refused, with ENOMEM, the error of an occupied fixed range, before it touches any state.
internal static class GuestVaHole
{
    public const ulong Start = 0x10_0000_0000UL;
    public const ulong End = 0x70_0000_0000UL;

    // Where the window's first byte moves to; SHARPEMU_VA_HOLE_BASE overrides it (hexadecimal).
    private const ulong DefaultRelocatedBase = 0x80_0000_0000UL;

    private static int _blockedFixedRequests;
    private static int _relocationLogs;
    private static readonly HashSet<(ulong Address, ulong Length)> _loggedRelocations = [];

    // Tests pin the host behaviour; null follows the running OS.
    internal static bool? SkipsSearchOverride { get; set; }
    internal static bool? BlocksFixedOverride { get; set; }
    internal static bool? RelocatesOverride { get; set; }

    public static bool SkipsSearch => SkipsSearchOverride ?? OperatingSystem.IsMacOS();
    public static bool BlocksFixed => BlocksFixedOverride ?? OperatingSystem.IsMacOSVersionAtLeast(27);

    public static int BlockedFixedRequests => Volatile.Read(ref _blockedFixedRequests);

    // The one amount both workarounds move a window address by.
    public static ulong Shift { get; } = ReadShift();

    private static ulong ReadShift()
    {
        var text = Environment.GetEnvironmentVariable("SHARPEMU_VA_HOLE_BASE");
        var relocatedBase = DefaultRelocatedBase;
        if (!string.IsNullOrWhiteSpace(text))
        {
            relocatedBase = Convert.ToUInt64(
                text.StartsWith("0x", StringComparison.OrdinalIgnoreCase) ? text[2..] : text, 16);
        }

        return unchecked(relocatedBase - Start);
    }

    // SHARPEMU_VA_HOLE_VIRTUAL=1 (experimental): the host keeps a fixed request there as a view record
    // alone (MacHostViews), so it is not refused; only the guest's own CPU accesses still fault.
    private static readonly bool VirtualViews = Environment.GetEnvironmentVariable("SHARPEMU_VA_HOLE_VIRTUAL") == "1";

    // SHARPEMU_VA_HOLE_RELOCATE=0 turns the kernel-side relocation off, which brings the refusal back.
    private static readonly bool RelocationDisabled = Environment.GetEnvironmentVariable("SHARPEMU_VA_HOLE_RELOCATE") == "0";

    public static bool Relocates =>
        RelocatesOverride ?? (BlocksFixed && !VirtualViews && !RelocationDisabled);

    public static bool Overlaps(ulong address, ulong length) =>
        length != 0 && address < End && (address > ulong.MaxValue - length || address + length > Start);

    private static bool Contains(ulong address) => address >= Start && address < End;

    // The shadow is where the window's addresses end up; searches stay out of it.
    public static bool OverlapsShadow(ulong address, ulong length)
    {
        var shadowStart = unchecked(Start + Shift);
        var shadowEnd = unchecked(End + Shift);
        return length != 0 && shadowStart < shadowEnd && address < shadowEnd &&
            (address > ulong.MaxValue - length || address + length > shadowStart);
    }

    // A single guest address inside the window and where the kernel really keeps it.
    public static ulong Translate(ulong address) =>
        Relocates && Contains(address) ? unchecked(address + Shift) : address;

    // The same for a range, translated only when all of it lies inside the window.
    public static bool TryTranslateRange(ulong address, ulong length, out ulong translated)
    {
        if (Relocates && length != 0 && Contains(address) && length <= End - address)
        {
            translated = unchecked(address + Shift);
            return true;
        }

        translated = address;
        return false;
    }

    public static ulong TranslateRange(ulong address, ulong length) =>
        TryTranslateRange(address, length, out var translated) ? translated : address;

    // Translates a fixed request that is about to map or reserve, and says so for the first few.
    public static ulong RelocateFixedRequest(ulong address, ulong length)
    {
        if (!TryTranslateRange(address, length, out var translated))
            return address;
        if (Volatile.Read(ref _relocationLogs) < 8)
        {
            bool first;
            lock (_loggedRelocations)
                first = _loggedRelocations.Count < 8 && _loggedRelocations.Add((address, length));
            if (first)
            {
                Interlocked.Increment(ref _relocationLogs);
                Console.Error.WriteLine(
                    $"[LOADER][INFO] Fixed guest mapping in macOS 27's blocked window relocated: 0x{address:X} -> 0x{translated:X} size=0x{length:X}");
            }
        }

        return translated;
    }

    public static bool IsFixedRequestBlocked(ulong address, ulong length)
    {
        if (!BlocksFixed || !Overlaps(address, length) || (VirtualViews && address >= Start && length <= End - address))
            return false;
        if (Interlocked.Increment(ref _blockedFixedRequests) <= 8)
            Console.Error.WriteLine(
                $"[LOADER][WARN] Fixed guest mapping in the host's blocked window refused: address=0x{address:X16} size=0x{length:X} " +
                $"window=[0x{Start:X}, 0x{End:X}).");
        return true;
    }

    // Moves a kernel-chosen search start past the window when the range would touch it, and past the
    // shadow of the window too unless the caller is out of other room (avoidShadow false).
    public static ulong SkipForSearch(ulong desired, ulong length, bool avoidShadow = true)
    {
        // Each step only moves forward over a finite span, so this ends.
        while (true)
        {
            if (SkipsSearch && Overlaps(desired, length))
                desired = End;
            else if (avoidShadow && Relocates && OverlapsShadow(desired, length))
                desired = unchecked(End + Shift);
            else
                return desired;
        }
    }
}
