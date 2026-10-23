// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.Gpu.Pipelines;

// Attempts are program resolutions, not issued draws; the backend counts draws separately.
public static class VertexFetchCounters
{
    private static readonly long[] Refusals = new long[Enum.GetValues<VertexFetchRefusal>().Length];
    private static long _loads, _positionOnly;
    public static void Refuse(VertexFetchRefusal reason) => Interlocked.Increment(ref Refusals[(int)reason]);
    public static void Replaced() => Interlocked.Increment(ref _loads);
    public static void PositionOnly() => Interlocked.Increment(ref _positionOnly);
    public static string TakeReport() =>
        $"vertex_fetch_loads={Interlocked.Exchange(ref _loads, 0)} " +
        $"vertex_fetch_refused=[{string.Join(',', Enum.GetValues<VertexFetchRefusal>().Where(r => r != VertexFetchRefusal.None)
            .Select(r => $"{r}:{Interlocked.Exchange(ref Refusals[(int)r], 0)}"))}] " +
        $"position_only_variants={Interlocked.Exchange(ref _positionOnly, 0)}";
}
