// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Runtime.CompilerServices;
using SharpEmu.Libs.Gpu.GpuCommands;

namespace SharpEmu.PerfAudit;

// Both arms use the same interface-dispatched operations. Production integration is separate.
internal static class ShaderRegisterMapBenches
{
    public static void Run()
    {
        const int rounds = 2000;
        Console.WriteLine("shader-map candidate only; synthetic CPU operations, not frame-time evidence");
        foreach (var scenario in new[] { "direct-runs", "repeated", "mixed-reads", "overflow" })
        {
            var expected = Exercise(new Dictionary<uint, uint>(), scenario, 20);
            if (Exercise(new DenseShaderRegisterMap(), scenario, 20) != expected)
                throw new InvalidOperationException($"Shader-map checksum mismatch: {scenario}.");
            for (var warmup = 0; warmup < 4; warmup++)
            {
                _ = Exercise(new Dictionary<uint, uint>(), scenario, 200);
                _ = Exercise(new DenseShaderRegisterMap(), scenario, 200);
            }
            ulong? reference = null;
            for (var trial = 0; trial < 4; trial++)
            foreach (var dense in (trial & 1) == 0 ? new[] { false, true } : new[] { true, false })
            {
                var allocated = GC.GetAllocatedBytesForCurrentThread();
                var started = Stopwatch.GetTimestamp();
                IDictionary<uint, uint> map = dense ? new DenseShaderRegisterMap() : new Dictionary<uint, uint>();
                var checksum = Exercise(map, scenario, rounds);
                var elapsed = Stopwatch.GetElapsedTime(started);
                var bytes = GC.GetAllocatedBytesForCurrentThread() - allocated;
                reference ??= checksum;
                if (checksum != reference.Value) throw new InvalidOperationException($"Trial checksum mismatch: {scenario}.");
                Console.WriteLine($"case={scenario} dense={dense} trial={trial} rounds={rounds} " +
                    $"ns_round={elapsed.TotalNanoseconds / rounds:F2} B_round={bytes / (double)rounds:F3} checksum={checksum:X16}");
            }
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static ulong Exercise(IDictionary<uint, uint> map, string scenario, int rounds)
    {
        ulong checksum = 0;
        for (var round = 0; round < rounds; round++)
        {
            if (scenario == "direct-runs") map.Clear();
            for (uint run = 0; run < 8; run++)
            {
                var first = run * 64;
                for (uint word = 0; word < 16; word++)
                {
                    var key = first + word;
                    if (scenario == "overflow" && word == 15) key = uint.MaxValue - run;
                    map[key] = (uint)round ^ key;
                }
            }
            if (scenario == "mixed-reads")
                for (uint key = 0; key < 768; key += 3)
                    checksum += map.TryGetValue(key, out var value) ? value + 1UL : 0UL;
            else
                checksum += map[64] + (ulong)map.Count;
        }
        foreach (var item in map) checksum += item.Key + (ulong)item.Value;
        return checksum;
    }
}
