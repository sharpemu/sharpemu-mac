// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.Gpu.Pipelines;

// Aggregate only: no descriptor tracking, callbacks, or per-draw allocations.
public static class DrawPathCounters
{
    public static readonly bool Enabled = Environment.GetEnvironmentVariable("SHARPEMU_DRAWPATH_REUSE") != "0";
    private static long _programHits, _programMisses, _permutationHits, _permutationSearches;
    private static long _vertexHits, _vertexMisses, _vertexBulk, _vertexFallback, _pixelBulk, _pixelFallback;
    private static long _viewHits, _viewMisses;

    internal static void Program(bool hit) => Interlocked.Increment(ref (hit ? ref _programHits : ref _programMisses));
    internal static void Permutation(bool hit) => Interlocked.Increment(ref (hit ? ref _permutationHits : ref _permutationSearches));
    internal static void Vertex(bool hit) => Interlocked.Increment(ref (hit ? ref _vertexHits : ref _vertexMisses));
    internal static void VertexRead(bool bulk) => Interlocked.Increment(ref (bulk ? ref _vertexBulk : ref _vertexFallback));
    internal static void PixelRead(bool bulk) => Interlocked.Increment(ref (bulk ? ref _pixelBulk : ref _pixelFallback));
    public static void View(bool hit) => Interlocked.Increment(ref (hit ? ref _viewHits : ref _viewMisses));

    public static string TakeReport() =>
        $"drawpath_reuse={(Enabled ? 1 : 0)} " +
        $"drawpath_program_hit={Interlocked.Exchange(ref _programHits, 0)} drawpath_program_miss={Interlocked.Exchange(ref _programMisses, 0)} " +
        $"drawpath_permutation_hit={Interlocked.Exchange(ref _permutationHits, 0)} drawpath_permutation_search={Interlocked.Exchange(ref _permutationSearches, 0)} " +
        $"drawpath_vertex_hit={Interlocked.Exchange(ref _vertexHits, 0)} drawpath_vertex_changed={Interlocked.Exchange(ref _vertexMisses, 0)} " +
        $"drawpath_vertex_bulk={Interlocked.Exchange(ref _vertexBulk, 0)} drawpath_vertex_fallback={Interlocked.Exchange(ref _vertexFallback, 0)} " +
        $"drawpath_pixel_bulk={Interlocked.Exchange(ref _pixelBulk, 0)} drawpath_pixel_fallback={Interlocked.Exchange(ref _pixelFallback, 0)} " +
        $"drawpath_view_hit={Interlocked.Exchange(ref _viewHits, 0)} drawpath_view_miss={Interlocked.Exchange(ref _viewMisses, 0)}";
}
