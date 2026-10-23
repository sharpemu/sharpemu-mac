// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Metal;

namespace SharpEmu.Gpu.Metal.V2;

// When a render pass need not load a colour attachment. On a tile-based GPU a load reads
// the whole attachment into tile memory before the first draw, and a pass whose first
// draw overwrites every pixel of the attachment never looks at what it read: the
// post-processing, composite and present passes of a frame are such draws, a full-screen
// triangle or quad with blending off. The load then becomes MTLLoadActionDontCare.
//
// The load action is fixed when the pass begins, so the rule judges the draw the pass is
// begun for, and only on what it can prove: the clip positions of its vertices, which the
// translator works out for a vertex program that makes them from the vertex index alone;
// the viewport and scissor; a pipeline that writes every channel of the attachment from a
// pixel program that cannot discard, without blending; and no depth, stencil or face test
// that could reject a fragment. Anything not known is a load, which is what every pass did
// before.
internal static class MetalPassLoads
{
    /// <summary>The window rectangle a viewport maps clip space onto, as Metal sets it.</summary>
    public readonly record struct Viewport(double OriginX, double OriginY, double Width, double Height);

    /// <summary>The draw a pass is begun for, as far as its colour loads are concerned.</summary>
    public readonly record struct Draw(
        float[]? Positions,
        nuint PrimitiveType,
        bool RectangleList,
        uint VertexCount,
        uint FirstVertex,
        bool CullNone,
        bool DepthClamp,
        uint OverwriteMask,
        Viewport Viewport,
        int ScissorLeft,
        int ScissorTop,
        int ScissorRight,
        int ScissorBottom,
        bool FragmentTestsCanReject,
        uint Samples,
        uint Layers,
        uint RenderWidth,
        uint RenderHeight);

    private const nint AllChannels =
        Mtl.ColorWriteMaskRed | Mtl.ColorWriteMaskGreen | Mtl.ColorWriteMaskBlue | Mtl.ColorWriteMaskAlpha;

    /// <summary>
    /// The colour attachments every fragment of a pipeline writes whole: its pixel program
    /// cannot discard and returns a value for the attachment, blending is off and the write
    /// mask has every channel, at one sample per pixel.
    /// </summary>
    public static uint OverwriteMask(
        bool cannotDiscard,
        uint colorOutputMask,
        uint samples,
        ReadOnlySpan<bool> blends,
        ReadOnlySpan<nint> writeMasks)
    {
        if (!cannotDiscard || samples > 1)
        {
            return 0;
        }

        var mask = 0u;
        for (var index = 0; index < blends.Length && index < writeMasks.Length && index < 32; index++)
        {
            if ((colorOutputMask & (1u << index)) != 0 && !blends[index] && writeMasks[index] == AllChannels)
            {
                mask |= 1u << index;
            }
        }

        return mask;
    }

    /// <summary>
    /// Whether the draw writes every pixel of colour attachment <paramref name="index"/>,
    /// whose subresource is <paramref name="width"/> by <paramref name="height"/>. The
    /// subresource has to be the render area exactly, so the load action and the draw
    /// speak about the same pixels.
    /// </summary>
    public static bool OverwritesAttachment(in Draw draw, int index, uint width, uint height)
    {
        if (index is < 0 or >= 32 || (draw.OverwriteMask & (1u << index)) == 0 ||
            draw.Samples > 1 || draw.Layers > 1 || !draw.CullNone || draw.FragmentTestsCanReject ||
            width == 0 || height == 0 || width != draw.RenderWidth || height != draw.RenderHeight)
        {
            return false;
        }

        // ApplyViewportAndScissor clamps the scissor to the render area.
        var left = Math.Clamp(draw.ScissorLeft, 0, (int)draw.RenderWidth);
        var top = Math.Clamp(draw.ScissorTop, 0, (int)draw.RenderHeight);
        var right = Math.Clamp(draw.ScissorRight, 0, (int)draw.RenderWidth);
        var bottom = Math.Clamp(draw.ScissorBottom, 0, (int)draw.RenderHeight);
        if (left != 0 || top != 0 || right < width || bottom < height)
        {
            return false;
        }

        return CoversRectangle(in draw, width, height);
    }

    // ConstantPositions is indexed by the effective vertex id (including base vertex).
    // Reorder the proof in draw order, including duplicates; unknown/restart indices fail.
    internal static float[]? IndexedPositions(float[] positions, ReadOnlySpan<uint> indices)
    {
        if (indices.Length is 0 or > Gen5MslShader.ConstantPositionVertexCount ||
            positions.Length < Gen5MslShader.ConstantPositionVertexCount * 4)
        {
            return null;
        }
        var ordered = new float[Gen5MslShader.ConstantPositionVertexCount * 4];
        for (var index = 0; index < indices.Length; index++)
        {
            if (indices[index] >= Gen5MslShader.ConstantPositionVertexCount) return null;
            positions.AsSpan((int)indices[index] * 4, 4).CopyTo(ordered.AsSpan(index * 4));
        }
        return ordered;
    }

    /// <summary>Whether the draw's primitives together cover the window rectangle
    /// [0, width] x [0, height].</summary>
    public static bool CoversRectangle(in Draw draw, uint width, uint height)
    {
        var positions = draw.Positions;
        if (positions is null || draw.VertexCount == 0 ||
            (ulong)draw.FirstVertex + draw.VertexCount > Gen5MslShader.ConstantPositionVertexCount ||
            positions.Length < Gen5MslShader.ConstantPositionVertexCount * 4)
        {
            return false;
        }

        Span<(double X, double Y)> window = stackalloc (double, double)[Gen5MslShader.ConstantPositionVertexCount];
        for (var vertex = 0; vertex < draw.VertexCount; vertex++)
        {
            var offset = (int)(draw.FirstVertex + vertex) * 4;
            var x = positions[offset];
            var y = positions[offset + 1];
            var z = positions[offset + 2];
            var w = positions[offset + 3];
            // Behind the eye or outside the depth range would be clipped.
            if (!float.IsFinite(x) || !float.IsFinite(y) || !float.IsFinite(z) || !(w > 0f) || !float.IsFinite(w) ||
                (!draw.DepthClamp && (z < 0f || z > w)))
            {
                return false;
            }

            window[vertex] = ToWindow(draw.Viewport, x, y, w);
        }

        if (draw.RectangleList)
        {
            return draw.VertexCount == 3 && RectangleCovers(in draw, positions, width, height);
        }

        if (draw.PrimitiveType == Mtl.PrimitiveTriangle && draw.VertexCount == 3 ||
            draw.PrimitiveType == Mtl.PrimitiveTriangleStrip && draw.VertexCount == 3)
        {
            return TriangleContains(window[0], window[1], window[2], width, height);
        }

        if (draw.PrimitiveType == Mtl.PrimitiveTriangleStrip && draw.VertexCount == 4)
        {
            return PairCovers(window[0], window[1], window[2], window[1], window[2], window[3], width, height);
        }

        if (draw.PrimitiveType == Mtl.PrimitiveTriangle && draw.VertexCount == 6)
        {
            return PairCovers(window[0], window[1], window[2], window[3], window[4], window[5], width, height);
        }

        return false;
    }

    // Metal's viewport transform: NDC y runs up, window y down.
    private static (double X, double Y) ToWindow(Viewport viewport, float x, float y, float w) =>
        (viewport.OriginX + ((x / (double)w) + 1.0) * viewport.Width * 0.5,
         viewport.OriginY + (1.0 - (y / (double)w)) * viewport.Height * 0.5);

    private static double Cross((double X, double Y) a, (double X, double Y) b, (double X, double Y) c) =>
        ((b.X - a.X) * (c.Y - a.Y)) - ((b.Y - a.Y) * (c.X - a.X));

    /// <summary>Every corner of the rectangle inside the closed triangle. The rectangle is
    /// convex, so its interior - where every pixel centre lies - is then inside too.</summary>
    private static bool TriangleContains(
        (double X, double Y) a, (double X, double Y) b, (double X, double Y) c, uint width, uint height)
    {
        var area = Cross(a, b, c);
        if (area == 0 || !double.IsFinite(area))
        {
            return false;
        }

        var sign = Math.Sign(area);
        foreach (var corner in Corners(width, height))
        {
            if (Cross(a, b, corner) * sign < 0 || Cross(b, c, corner) * sign < 0 || Cross(c, a, corner) * sign < 0)
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>Two triangles sharing an edge that is a diagonal of the convex quadrilateral
    /// they make: the union is that quadrilateral, and rasterisation leaves no gap along the
    /// shared edge.</summary>
    private static bool PairCovers(
        (double X, double Y) a0, (double X, double Y) a1, (double X, double Y) a2,
        (double X, double Y) b0, (double X, double Y) b1, (double X, double Y) b2,
        uint width, uint height)
    {
        ReadOnlySpan<(double X, double Y)> first = [a0, a1, a2];
        ReadOnlySpan<(double X, double Y)> second = [b0, b1, b2];
        if (TriangleContains(a0, a1, a2, width, height) || TriangleContains(b0, b1, b2, width, height))
        {
            return true;
        }

        // The vertex of each triangle the other does not share.
        int firstOnly = -1, secondOnly = -1, shared = 0;
        for (var index = 0; index < 3; index++)
        {
            var inSecond = second.IndexOf(first[index]) >= 0;
            shared += inSecond ? 1 : 0;
            if (!inSecond)
            {
                firstOnly = firstOnly < 0 ? index : 3;
            }

            if (first.IndexOf(second[index]) < 0)
            {
                secondOnly = secondOnly < 0 ? index : 3;
            }
        }

        if (shared != 2 || firstOnly is < 0 or > 2 || secondOnly is < 0 or > 2)
        {
            return false;
        }

        var p = first[firstOnly];
        var q = second[secondOnly];
        var s0 = first[(firstOnly + 1) % 3];
        var s1 = first[(firstOnly + 2) % 3];
        // Diagonals s0-s1 and p-q cross strictly: the quadrilateral p, s0, q, s1 is convex.
        if (!(Cross(s0, s1, p) * Cross(s0, s1, q) < 0) || !(Cross(p, q, s0) * Cross(p, q, s1) < 0))
        {
            return false;
        }

        ReadOnlySpan<(double X, double Y)> quad = [p, s0, q, s1];
        var sign = Math.Sign(Cross(p, s0, q));
        foreach (var corner in Corners(width, height))
        {
            for (var edge = 0; edge < 4; edge++)
            {
                if (Cross(quad[edge], quad[(edge + 1) % 4], corner) * sign < 0)
                {
                    return false;
                }
            }
        }

        return true;
    }

    /// <summary>A rectangle-list rectangle drawn from its first three vertices. The pipeline
    /// completes it at the corner whose edges leave at right angles (Gen5MslRectangleList),
    /// which for an axis-aligned one at a common w is its bounding box.</summary>
    private static bool RectangleCovers(in Draw draw, float[] positions, uint width, uint height)
    {
        var first = (int)draw.FirstVertex * 4;
        Span<(float X, float Y, float Z, float W)> corner = stackalloc (float, float, float, float)[3];
        for (var index = 0; index < 3; index++)
        {
            var offset = first + (index * 4);
            corner[index] = (positions[offset], positions[offset + 1], positions[offset + 2], positions[offset + 3]);
        }

        if (corner[0].W != corner[1].W || corner[0].W != corner[2].W)
        {
            return false;
        }

        var right = -1;
        for (var index = 0; index < 3 && right < 0; index++)
        {
            var at = corner[index];
            var next = corner[(index + 1) % 3];
            var previous = corner[(index + 2) % 3];
            if ((at.X == next.X && at.Y == previous.Y) || (at.Y == next.Y && at.X == previous.X))
            {
                right = index;
            }
        }

        if (right < 0)
        {
            return false;
        }

        var o0 = corner[right];
        var o1 = corner[(right + 1) % 3];
        var o2 = corner[(right + 2) % 3];
        var fourthZ = o1.Z == o0.Z ? o2.Z : o2.Z == o0.Z ? o1.Z : o1.Z + o2.Z - o0.Z;
        if (!draw.DepthClamp && (fourthZ < 0f || fourthZ > o0.W))
        {
            return false;
        }

        var a = ToWindow(draw.Viewport, corner[0].X, corner[0].Y, corner[0].W);
        var b = ToWindow(draw.Viewport, corner[1].X, corner[1].Y, corner[1].W);
        var c = ToWindow(draw.Viewport, corner[2].X, corner[2].Y, corner[2].W);
        var minX = Math.Min(a.X, Math.Min(b.X, c.X));
        var maxX = Math.Max(a.X, Math.Max(b.X, c.X));
        var minY = Math.Min(a.Y, Math.Min(b.Y, c.Y));
        var maxY = Math.Max(a.Y, Math.Max(b.Y, c.Y));
        return minX <= 0 && minY <= 0 && maxX >= width && maxY >= height;
    }

    private static (double X, double Y)[] Corners(uint width, uint height) =>
        [(0, 0), (width, 0), (0, height), (width, height)];
}
