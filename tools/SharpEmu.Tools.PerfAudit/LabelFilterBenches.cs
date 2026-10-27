// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Runtime.CompilerServices;
using SharpEmu.Gpu.Metal.V2;

namespace SharpEmu.PerfAudit;

internal static class LabelFilterBenches
{
    public static void Run()
    {
        Console.WriteLine($"arch={System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture} runtime={Environment.Version}");
        foreach (var count in new[] { 16, 256, 4096 })
        {
            var copies = Enumerable.Range(0, count).Select(i => new Copy(1, 0, 2, (ulong)i * 16, 8,
                0x100000UL + (i < count / 2 ? 0UL : 0x1000000UL) + (ulong)(i % (count / 2)) * 16)).ToArray();
            var filter = new MetalPageRangeFilter();
            foreach (var c in copies) filter.Add(c.Address, (ulong)c.Bytes);
            foreach (var (name, address) in new[] { ("outside-hull", 0x2000000UL), ("interpage-gap", 0x900000UL), ("first-hit", copies[0].Address), ("last-hit", copies[^1].Address), ("same-page-gap", copies[0].Address + 8) })
            {
                var baseline = Touch(copies, filter, address, false);
                if (baseline != Touch(copies, filter, address, true)) throw new Exception("mismatch");
                for (var i = 0; i < 20000; i++) { _ = Touch(copies, filter, address, false); _ = Touch(copies, filter, address, true); }
                for (var trial = 0; trial < 4; trial++)
                foreach (var enabled in (trial & 1) == 0 ? new[] { false, true } : new[] { true, false })
                {
                    const int iterations = 100000;
                    var alloc = GC.GetAllocatedBytesForCurrentThread(); var start = Stopwatch.GetTimestamp(); var hits = 0;
                    for (var i = 0; i < iterations; i++) if (Touch(copies, filter, address, enabled)) hits++;
                    var ns = Stopwatch.GetElapsedTime(start).TotalNanoseconds / iterations;
                    var bytes = GC.GetAllocatedBytesForCurrentThread() - alloc;
                    if (hits != (baseline ? iterations : 0)) throw new Exception("result mismatch");
                    Console.WriteLine($"count={count} case={name} filter={enabled} trial={trial} ns_op={ns:F2} B_op={bytes/(double)iterations:F4}");
                }
            }
            var before = GC.GetAllocatedBytesForCurrentThread(); var watch = Stopwatch.GetTimestamp();
            for (var batch = 0; batch < 10000; batch++)
            {
                filter.Clear();
                foreach (var c in copies) filter.Add(c.Address, (ulong)c.Bytes);
            }
            Console.WriteLine($"count={count} add_clear_ns_per_copy={Stopwatch.GetElapsedTime(watch).TotalNanoseconds/(10000*count):F2} bytes={GC.GetAllocatedBytesForCurrentThread()-before}");
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        static bool Touch(Copy[] copies, MetalPageRangeFilter filter, ulong address, bool enabled)
        {
            const ulong size = 4;
            if (address >= copies[^1].Address + 8 || address + size <= copies[0].Address) return false;
            if (enabled && !filter.MayOverlap(address, size)) return false;
            foreach (ref readonly var c in copies.AsSpan())
                if (c.Address < address + size && address < c.Address + (ulong)c.Bytes) return true;
            return false;
        }
    }

    private readonly record struct Copy(nint Source, ulong SourceOffset, nint Destination, ulong DestinationOffset, int Bytes, ulong Address);
}
