// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Numerics;
using System.Text;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Scheduling;

namespace SharpEmu.Gpu.Metal.V2;

// Dense surfaces only: every byte maps to a texel, so the direct gather never needs
// guest padding that a texture cannot hold. Layouts are constants in a cached pipeline.
internal readonly record struct MetalAliasSurface(uint Width, uint Height, uint Bytes, TileBlockKind? Kind);

internal static partial class MetalTilerShaders
{
    // Invert the tile's linear XOR transform. Each result mask reconstructs one x/y
    // coordinate bit as parity(offset & mask). Derive from the shared CPU tile oracle,
    // rather than maintaining a second inverse table by hand.
    internal static (uint[] X, uint[] Y) AliasInverseMasks(in TileBlockLayout block)
    {
        var xs = new uint[BitOperations.Log2(block.BlockWidth)];
        var ys = new uint[BitOperations.Log2(block.BlockHeight)];
        var basis = 0;
        for (uint y = 0; y < block.BlockHeight; y++)
        for (uint x = 0; x < block.BlockWidth; x++)
        {
            if (!TileGeometry.TryGetBlockOffset(block, x, y, 0, out var offset))
                throw GpuFatal.Fatal("An alias tile has no invertible address mapping.");
            if (!BitOperations.IsPow2(offset)) continue;
            basis++;
            for (var bit = 0; bit < xs.Length; bit++)
                if ((x & (1u << bit)) != 0) xs[bit] |= offset;
            for (var bit = 0; bit < ys.Length; bit++)
                if ((y & (1u << bit)) != 0) ys[bit] |= offset;
        }
        if (basis != xs.Length + ys.Length)
            throw GpuFatal.Fatal("An alias tile's address transform is not a bijection.");
        return (xs, ys);
    }

    internal static string CreateAliasLibrary(MetalAliasSurface source, MetalAliasSurface target)
    {
        var text = new StringBuilder("#include <metal_stdlib>\nusing namespace metal;\n");
        AppendAliasAddress(text, "source", source);
        AppendAliasAddress(text, "target", target);
        text.Append("kernel void alias_copy(texture2d_array<uint, access::read> source [[texture(0)]],\n" +
                    " texture2d_array<uint, access::write> target [[texture(1)]], uint2 gid [[thread_position_in_grid]]) {\n");
        text.Append($" if (gid.x >= {target.Width}u || gid.y >= {target.Height}u) return;\n uint4 value = uint4(0);\n");
        // Gather raw dwords even when the two formats have different texel sizes. No
        // floating-point/normalized conversion, filtering, swizzle or sRGB decode.
        for (uint word = 0; word < target.Bytes / 4; word++)
        {
            text.Append($" {{ uint address = target_address(gid) + {word * 4}u;\n" +
                        $" uint2 xy = source_coords(address);\n" +
                        $" value[{word}] = source.read(xy, 0)[(address % {source.Bytes}u) / 4u]; }}\n");
        }
        text.Append(" target.write(value, gid, 0);\n}\n");
        return text.ToString();
    }

    private static void AppendAliasAddress(StringBuilder text, string name, MetalAliasSurface surface)
    {
        var rowBytes = surface.Width * surface.Bytes;
        if (surface.Kind is not { } kind)
        {
            text.Append($"uint {name}_address(uint2 p) {{ return p.y * {rowBytes}u + p.x * {surface.Bytes}u; }}\n" +
                        $"uint2 {name}_coords(uint a) {{ return uint2((a % {rowBytes}u) / {surface.Bytes}u, a / {rowBytes}u); }}\n");
            return;
        }
        TileGeometry.TryGetBlockLayout(kind, surface.Bytes, out var block);
        var columns = surface.Width / block.BlockWidth;
        text.Append($"uint {name}_offset(uint sx, uint sy) {{ uint sz = 0u; return {BlockOffset(kind, surface.Bytes)}; }}\n");
        text.Append($"uint {name}_address(uint2 p) {{ return ((p.y / {block.BlockHeight}u) * {columns}u + p.x / {block.BlockWidth}u) * {block.BlockSize}u + {name}_offset(p.x, p.y); }}\n");
        text.Append($"uint2 {name}_coords(uint a) {{ uint b = a / {block.BlockSize}u;\n" +
                    $" uint2 p = uint2((b % {columns}u) * {block.BlockWidth}u, (b / {columns}u) * {block.BlockHeight}u);\n" +
                    $" uint o = (a % {block.BlockSize}u) ^ {name}_offset(p.x, p.y);\n");
        var inverse = AliasInverseMasks(block);
        for (var bit = 0; bit < inverse.X.Length; bit++)
            text.Append($" p.x |= (popcount(o & {inverse.X[bit]}u) & 1u) << {bit}u;\n");
        for (var bit = 0; bit < inverse.Y.Length; bit++)
            text.Append($" p.y |= (popcount(o & {inverse.Y[bit]}u) & 1u) << {bit}u;\n");
        text.Append(" return p; }\n");
    }
}

internal sealed unsafe partial class MetalTiler
{
    public void EncodeAlias(nint encoder, nint sourceView, nint targetView,
        MetalAliasSurface source, MetalAliasSurface target)
    {
        var key = $"alias:{source}:{target}";
        if (!_conversionPipelines.TryGetValue(key, out var pipeline))
        {
            var library = CompileLibrary(MetalTilerShaders.CreateAliasLibrary(source, target), key);
            try { pipeline = CreatePipeline(library, "alias_copy", 0, null, key); }
            finally { Release(library); }
            _conversionPipelines.Add(key, pipeline);
        }
        MetalNative.SendVoid(encoder, SelSetPipeline, pipeline);
        MetalV2Native.SendSetResourceAtIndex(encoder, SelSetTexture, sourceView, 0);
        MetalV2Native.SendSetResourceAtIndex(encoder, SelSetTexture, targetView, 1);
        MetalNative.SendDispatch(encoder, SelDispatch,
            new MtlSize { Width = (target.Width + 15) / 16, Height = (target.Height + 15) / 16, Depth = 1 },
            new MtlSize { Width = 16, Height = 16, Depth = 1 });
    }
}
