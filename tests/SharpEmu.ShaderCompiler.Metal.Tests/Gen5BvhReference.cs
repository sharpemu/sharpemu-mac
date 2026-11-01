// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Metal.Tests;

// A CPU model of IMAGE_BVH_INTERSECT_RAY / IMAGE_BVH64_INTERSECT_RAY, written independently
// of Templates/bvh_prelude.msl from the same sources, operation for operation in binary32:
// - the T# fields: RDNA2 ISA Table 49;
// - node pointers: Mesa radv_nir_rt_common.c build_node_to_addr (type in bits 2:0, the
//   node's 64-byte index above them);
// - box nodes: GPURT IntersectCommon.hlsl fast_intersect_bbox / IntersectNodeBvh4, with
//   HLSL min/max as IEEE minNum/maxNum;
// - triangle nodes: Mesa intersect_ray_amd_software_tri (watertight, with its edge and
//   vertex rules), unnormalized; vertex choice and barycentric swizzle from GPURT
//   CalcTriangleVertexOffsets and SwizzleBarycentrics;
// - type 6, the base PS5's null node: the PSR library's qbvh::NODE_NULL (qbvh_constants.hpp),
//   which the traversal expects back as four 0xFFFFFFFF.
internal static class Gen5BvhReference
{
    public const uint Invalid = 0xFFFF_FFFF;

    public static uint[] Intersect(
        Func<ulong, uint> readDword,
        uint[] descriptor,
        ulong nodePointer,
        float extent,
        float[] origin,
        float[] direction,
        float[] inverseDirection)
    {
        var baseAddress = (((ulong)(descriptor[1] & 0xFF) << 32) | descriptor[0]) << 8;
        var grow = (descriptor[1] >> 23) & 0xFF;
        var sort = (descriptor[1] >> 31) != 0;
        var last = ((ulong)(descriptor[3] & 0x3FF) << 32) | descriptor[2];
        var barycentrics = ((descriptor[3] >> 24) & 1) != 0;
        var type = (uint)nodePointer & 7;
        var index = nodePointer >> 3;

        uint[] Fetch(int count)
        {
            var words = new uint[count];
            var address = baseAddress + (index << 6);
            for (var word = 0; word < count; word++)
                words[word] = readDword(address + (ulong)word * 4);
            return words;
        }

        if (type <= 1 && index <= last)
            return Triangle(Fetch(16), type, barycentrics, origin, direction, inverseDirection);
        if (type == 4 && index <= last)
            return Box(Fetch(16), fp16: true, grow, sort, extent, origin, inverseDirection);
        if (type == 5 && index < last)
            return Box(Fetch(28), fp16: false, grow, sort, extent, origin, inverseDirection);
        return [Invalid, Invalid, Invalid, Invalid];
    }

    public static float Half(uint bits) => (float)BitConverter.UInt16BitsToHalf((ushort)bits);

    private static float MaxNum(float a, float b) =>
        float.IsNaN(a) ? b : float.IsNaN(b) ? a : b > a ? b : a;

    private static float MinNum(float a, float b) =>
        float.IsNaN(a) ? b : float.IsNaN(b) ? a : b < a ? b : a;

    private static (float Min, float Max) Slab(float[] origin, float[] inverse, float[] min, float[] max, float extent)
    {
        var enter = new float[3];
        var leave = new float[3];
        for (var axis = 0; axis < 3; axis++)
        {
            var planeMin = (min[axis] - origin[axis]) * inverse[axis];
            var planeMax = (max[axis] - origin[axis]) * inverse[axis];
            var positive = inverse[axis] >= 0f;
            enter[axis] = positive ? planeMin : planeMax;
            leave[axis] = positive ? planeMax : planeMin;
        }

        var first = MaxNum(MaxNum(enter[0], enter[1]), enter[2]);
        var last = MinNum(MinNum(leave[0], leave[1]), leave[2]);
        if (float.IsNaN(first) || float.IsNaN(last))
            return (float.PositiveInfinity, float.NegativeInfinity);
        return (MaxNum(first, 0f), MinNum(last, extent));
    }

    private static uint[] Box(uint[] node, bool fp16, uint grow, bool sort, float extent, float[] origin, float[] inverse)
    {
        var bounds = new float[24];
        for (var slot = 0; slot < 24; slot++)
        {
            bounds[slot] = fp16
                ? Half((node[4 + slot / 2] >> (slot % 2 * 16)) & 0xFFFF)
                : BitConverter.UInt32BitsToSingle(node[4 + slot]);
        }

        return BoxTest(node[..4], bounds, grow, sort, extent, origin, inverse);
    }

    private static uint[] BoxTest(uint[] children, float[] bounds, uint grow, bool sort, float extent, float[] origin, float[] inverse)
    {
        var factor = 1f + (float)grow * (1f / 16777216f);
        var child = new uint[4];
        var key = new float[4];
        for (var index = 0; index < 4; index++)
        {
            var (min, max) = Slab(origin, inverse, bounds[(index * 6)..(index * 6 + 3)], bounds[(index * 6 + 3)..(index * 6 + 6)], extent);
            child[index] = min <= max * factor ? children[index] : Invalid;
            key[index] = min;
        }

        if (sort)
        {
            foreach (var (a, b) in new[] { (0, 2), (1, 3), (0, 1), (2, 3), (1, 2) })
            {
                if ((child[b] != Invalid && key[b] < key[a]) || child[a] == Invalid)
                {
                    (child[a], child[b]) = (child[b], child[a]);
                    (key[a], key[b]) = (key[b], key[a]);
                }
            }
        }

        return child;
    }

    private static bool Edge(float v0X, float v0Y, float v1X, float v1Y)
    {
        var tX = v1X - v0X;
        var tY = v1Y - v0Y;
        return tX == 0f ? tY < 0f : tX < 0f;
    }

    private static bool Vertex(float v0X, float v1X, float v2X) => v1X - v0X <= 0f && v2X - v0X > 0f;

    private static uint[] Triangle(uint[] node, uint type, bool barycentrics, float[] origin, float[] direction, float[] inverse)
    {
        float[] Relative(int dword) =>
        [
            BitConverter.UInt32BitsToSingle(node[dword]) - origin[0],
            BitConverter.UInt32BitsToSingle(node[dword + 1]) - origin[1],
            BitConverter.UInt32BitsToSingle(node[dword + 2]) - origin[2],
        ];

        var a = Relative(type == 1 ? 3 : 0);
        var b = Relative(type == 1 ? 9 : 3);
        var c = Relative(6);

        var absX = MathF.Abs(direction[0]);
        var absY = MathF.Abs(direction[1]);
        var absZ = MathF.Abs(direction[2]);
        var (kx, ky, kz) = absX >= absY
            ? absX >= absZ ? (1, 2, 0) : (0, 1, 2)
            : absY >= absZ ? (2, 0, 1) : (0, 1, 2);
        if (direction[kz] < 0f)
            (kx, ky) = (ky, kx);

        var sz = inverse[kz];
        var sx = direction[kx] * sz;
        var sy = direction[ky] * sz;
        var ax = a[kx] - sx * a[kz];
        var ay = a[ky] - sy * a[kz];
        var bx = b[kx] - sx * b[kz];
        var by = b[ky] - sy * b[kz];
        var cx = c[kx] - sx * c[kz];
        var cy = c[ky] - sy * c[kz];
        var u = cx * by - cy * bx;
        var v = ax * cy - ay * cx;
        var w = bx * ay - by * ax;

        var back = u < 0f || v < 0f || w < 0f;
        var front = u > 0f || v > 0f || w > 0f;
        var hit = !(back && front);
        bool edgeA = u == 0f, edgeB = v == 0f, edgeC = w == 0f;
        if (edgeA || edgeB || edgeC)
        {
            var onEdge = (edgeA && Edge(bx, by, cx, cy)) || (edgeB && Edge(cx, cy, ax, ay)) || (edgeC && Edge(ax, ay, bx, by));
            bool vertexA = edgeB && edgeC, vertexB = edgeA && edgeC, vertexC = edgeA && edgeB;
            var onVertex = (vertexA && Vertex(ax, bx, cx)) || (vertexB && Vertex(bx, cx, ax)) || (vertexC && Vertex(cx, ax, bx));
            hit = hit && onEdge && (!(vertexA || vertexB || vertexC) || onVertex);
        }

        var det = u + (v + w);
        var az = sz * a[kz];
        var bz = sz * b[kz];
        var cz = sz * c[kz];
        var t = (u * az + v * bz) + w * cz;
        var sign = det > 0f ? 1f : det < 0f ? -1f : 0f;
        hit = hit && !(sign * t < 0f);

        var triangleId = node[15];
        if (!hit)
            return [Bits(float.PositiveInfinity), Bits(1f), barycentrics ? 0u : triangleId, 0u];
        if (!barycentrics)
            return [Bits(t), Bits(det), triangleId, 1u];

        float[] weight = [u, v, w, 0f];
        var shift = (int)type * 8;
        return [Bits(t), Bits(det), Bits(weight[(triangleId >> shift) & 3]), Bits(weight[(triangleId >> (shift + 2)) & 3])];
    }

    private static uint Bits(float value) => BitConverter.SingleToUInt32Bits(value);
}
