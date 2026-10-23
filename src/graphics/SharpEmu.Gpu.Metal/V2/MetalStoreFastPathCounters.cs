// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Metal.V2;

// Producer-sharded counters, reduced only by the periodic reporter. Preallocated
// on cache construction: even the first native fault must not allocate a bank.
// Colliding thread IDs share atomic counters; no count is lost, and ordinary
// producers avoid a single global counter cache line on every store.
internal static class MetalStoreFastPathCounters
{
    internal enum Kind
    {
        Attempts, FastAdmissions, FenceFast, FenceReject, TrueOverlap,
        ImageReject, MetadataReject, StagedFast, StagedReject, Fallback,
        CasRetry, FenceGate, ImageGate, StagedGate,
    }
    private static readonly bool Enabled = Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_STORE_FASTPATH_COUNTERS") == "1";
    private static readonly long[][] Banks = Enabled
        ? Enumerable.Range(0, 128).Select(_ => new long[Enum.GetValues<Kind>().Length]).ToArray()
        : [];

    // Suppress beforefieldinit so Initialize really warms the class on the worker.
    static MetalStoreFastPathCounters() { }

    internal static void Initialize() => _ = Banks.Length;

    internal static void Count(Kind kind)
    {
        if (!Enabled) return;
        var bank = Banks[Environment.CurrentManagedThreadId & 127];
        Interlocked.Increment(ref bank[(int)kind]);
    }

    internal static string Take()
    {
        if (!Enabled) return string.Empty;
        var totals = new long[Enum.GetValues<Kind>().Length];
        foreach (var bank in Banks)
            for (var i = 0; i < totals.Length; i++) totals[i] += Interlocked.Exchange(ref bank[i], 0);
        return string.Concat(Enum.GetValues<Kind>().Select(kind => $" store_{kind.ToString().ToLowerInvariant()}={totals[(int)kind]}"));
    }
}
