// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.Gpu.Pipelines;

// Diagnostics only. Invalidation belongs to the existing write generations,
// dependency epochs, window partition and encoder state.
public static class HostLookupCounters
{
    public static readonly bool Enabled = Environment.GetEnvironmentVariable("SHARPEMU_HOST_LOOKUPS") != "0";
    public enum Cache { ShaderOwnership, VertexLayout, Window, ResidentWalk, DeviceWindowWalk, Dcc, ArgumentPatch, ShaderIdentity, ShaderHeader }
    private sealed class Counts
    {
        internal readonly long[] Values = new long[27];
        internal readonly long[] Reported = new long[27];
    }
    private static readonly System.Collections.Concurrent.ConcurrentBag<Counts> Shards = new();
    private static readonly object Gate = new();
    [ThreadStatic] private static Counts? _local;
    public static void Note(Cache cache, bool hit, bool refused = false)
    {
        var counts = _local;
        if (counts is null) Shards.Add(_local = counts = new());
        var index = (int)cache * 3 + (hit ? 0 : refused ? 2 : 1);
        Volatile.Write(ref counts.Values[index], counts.Values[index] + 1);
    }
    public static string TakeReport()
    {
        lock (Gate)
        {
            Span<long> totals = stackalloc long[27];
            totals.Clear();
            foreach (var counts in Shards)
                for (var i = 0; i < totals.Length; i++)
                {
                    var value = Volatile.Read(ref counts.Values[i]);
                    totals[i] += value - counts.Reported[i];
                    counts.Reported[i] = value;
                }
            var fields = new List<string>();
            foreach (var cache in Enum.GetValues<Cache>())
            {
                var i = (int)cache * 3;
                fields.Add($"{cache}_hit={totals[i]} {cache}_miss={totals[i + 1]} {cache}_refused={totals[i + 2]}");
            }
            return string.Join(' ', fields);
        }
    }
}
