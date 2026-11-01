// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

// IMAGE_BVH_INTERSECT_RAY / IMAGE_BVH64_INTERSECT_RAY and their A16 forms run against small
// BVHs built here in AMD's node formats, mapped through the device-address page table the
// way the v2 host maps guest memory. Each lane tests one ray against one node; every result
// dword is compared with Gen5BvhReference, and the cases whose answer is known by
// construction also check that answer. GTA V's compute shader 0x4EE720FC4678FA19 aborted
// with "ImageBvhIntersectRay: unsupported instruction".
public sealed class Gen5MslBvhIntersectTests(ITestOutputHelper output)
{
    private const ulong GuestBase = 0x4000_0000;
    private const int GuestBytes = 0x8000;
    private const uint FaultWords = 0x1000;
    private const int Lanes = 32;
    private const int LaneBytes = 64;
    private const int OutputOffset = 2048;
    private const uint Invalid = Gen5BvhReference.Invalid;

    // Node indices (64-byte units from the BVH base).
    private const uint Box32 = 0;          // two units
    private const uint Box16 = 2;
    private const uint Triangles = 3;      // a quad: triangle 0 (v0, v1, v2), triangle 1 (v1, v3, v2)
    private const uint SpecialBox32 = 4;   // two units
    private const uint SpecialBox16 = 6;
    private const uint Swizzled = 7;
    private const uint Ties = 8;           // two units
    private const uint Straddling = 255;   // a box32 node across the 16 KB page boundary
    private const uint LastNode = 511;

    private const uint TypeBox16 = 4;
    private const uint TypeBox32 = 5;
    private const uint TypeNull = 6;

    private readonly record struct Ray(ulong Pointer, float Extent, float[] Origin, float[] Direction)
    {
        public float[] Inverse => [1f / Direction[0], 1f / Direction[1], 1f / Direction[2]];
    }

    private sealed record Variant(bool Bvh64, bool A16, uint[] Descriptor);

    [Fact]
    public void TheInstructionCompilesToTheDevicePreludeOnlyWhereItIsUsed()
    {
        var shader = Compile(new Variant(true, false, Descriptor()));
        Assert.Contains("sharpemu_bvh_intersect_ray(sharpemu_ranges", shader.Source, StringComparison.Ordinal);
        Assert.Contains("#pragma METAL fp contract(off)", shader.Source, StringComparison.Ordinal);

        var fixture = new Gen5ComputeFixture("no-bvh", [0x7E000280, 0xBF810000], 0, 0);
        Assert.DoesNotContain("sharpemu_bvh_", Gen5ComputeFixtures.CompileRequestOrThrow(fixture).Source, StringComparison.Ordinal);
        if (MetalNative.IsAvailable)
            Assert.True(MetalNative.TryCompileLibrary(shader.Source, out _, out var error), error);
    }

    // A T# held in user-data SGPRs is read whole: all four registers are loaded, and the
    // node reads make the layout carry the device-address ranges.
    [Theory]
    [InlineData(0x66u)]
    [InlineData(0x67u)]
    public void ATSharpInUserDataIsLoadedWholeAndBindsTheAddressRanges(uint opcode)
    {
        var fixture = new Gen5ComputeFixture("bvh-user-data", [0xF0009F01u | (opcode << 18), 2u | (16u << 8) | (1u << 16), 0xBF810000], 0, 0);
        var program = Gen5ComputeFixtures.DecodeOrThrow(fixture);
        Assert.Equal([4u, 5u, 6u, 7u], BindingLayout.CollectUserDataRegisters(program, 0, 16, 0));
        var request = Gen5ComputeFixtures.CreateComputeRequest(fixture);
        Assert.NotNull(request.Bindings.Find(DescriptorBindingKind.DeviceAddressPageTable));
        Assert.NotNull(request.Bindings.Find(DescriptorBindingKind.FaultBuffer));
    }

    // Four children at increasing distance along +x, stored out of order; child 3 is
    // stored invalid although its box is in front of the ray.
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Box32ChildrenComeBackSortedByEntryTimeWithInvalidOnesLast(bool sort)
    {
        var guest = BuildGuest();
        var rays = new[]
        {
            AlongX(Pointer(Box32, TypeBox32)),
            AlongX(Pointer(Box16, TypeBox16)),
        };
        var results = Run(guest, new Variant(false, false, Descriptor(sort: sort)), rays);
        if (results is null) return;

        uint[] expected = sort ? [0x30u, 0x10u, 0x20u, Invalid] : [0x10u, 0x20u, 0x30u, Invalid];
        Assert.Equal(expected, results[0]);
        Assert.Equal(expected, results[1]);
    }

    // Equal entry times keep the order GPURT's network (0-2, 1-3, 0-1, 2-3, 1-2) leaves
    // them in; Mesa's software network (0-1, 2-3, 0-2, 1-3, 1-2) would give D, B, A, C.
    [Fact]
    public void TiedChildrenFollowTheHardwareSortingNetwork()
    {
        var results = Run(BuildGuest(), new Variant(false, false, Descriptor(sort: true)), [AlongX(Pointer(Ties, TypeBox32))]);
        if (results is null) return;
        Assert.Equal([0x54u, 0x51u, 0x53u, 0x52u], results[0]);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void BoxNodesMatchTheReferenceForMissesEdgesAndSpecialValues(bool bvh64, bool a16)
    {
        var guest = BuildGuest();
        var rays = new List<Ray>();
        foreach (var node in new[] { Pointer(Box32, TypeBox32), Pointer(Box16, TypeBox16), Pointer(SpecialBox32, TypeBox32), Pointer(SpecialBox16, TypeBox16) })
        {
            rays.Add(AlongX(node));
            rays.Add(new Ray(node, 100f, [0.5f, 0.5f, 0.5f], [-1f, 0f, 0f]));           // pointing away: every box behind
            rays.Add(new Ray(node, 2.5f, [0f, 0.5f, 0.5f], [1f, 0f, 0f]));              // extent cuts off the far boxes
            rays.Add(new Ray(node, 100f, [0f, 1f, 0.5f], [1f, 0f, 0f]));                // along a face: 0 * inf planes
            rays.Add(new Ray(node, 100f, [0f, 0.5f, 0.5f], [float.NaN, 0f, 0f]));       // NaN direction
            rays.Add(new Ray(node, 100f, [0f, 0.5f, 0.5f], [1f, float.NaN, 1f]));
            rays.Add(new Ray(node, 100f, [-1f, -1f, -1f], [1f, 1f, 1f]));               // diagonal, edge-on corners
            rays.Add(new Ray(node, float.PositiveInfinity, [0f, 0.5f, 0.5f], [1f, -0f, 0f]));
        }

        var results = Run(guest, new Variant(bvh64, a16, Descriptor(sort: true)), rays);
        if (results is null) return;

        // A NaN box (RADV's empty child) never hits.
        Assert.DoesNotContain(0x41u, results[16]);
        Assert.DoesNotContain(0x61u, results[24]);
    }

    // A box whose exit lies a few ULPs before its entry only hits once box_grow_value
    // widens the exit by that many 2^-24 steps (grow 1 rounds away entirely).
    [Fact]
    public void BoxGrowingWidensTheExitByTheDescriptorsUlps()
    {
        var guest = BuildGuest();
        // Along +x, child 3 of SpecialBox32 is entered at x = 1 and left one ULP earlier.
        var ray = new Ray(Pointer(SpecialBox32, TypeBox32), 100f, [0f, 0.5f, 0.5f], [1f, 0f, 0f]);
        var hits = new List<uint[]>();
        foreach (var grow in new uint[] { 0, 1, 2, 8, 255 })
        {
            var results = Run(guest, new Variant(false, false, Descriptor(sort: false, grow: grow)), [ray]);
            if (results is null) return;
            hits.Add(results[0]);
        }

        Assert.Equal(Invalid, hits[0][3]);
        Assert.Equal(Invalid, hits[1][3]);
        Assert.Equal(0x44u, hits[2][3]);
    }

    [Theory]
    [InlineData(false, false, true)]
    [InlineData(false, false, false)]
    [InlineData(true, false, true)]
    [InlineData(false, true, true)]
    [InlineData(true, true, false)]
    public void TriangleNodesReturnTheirNumeratorsDenominatorAndId(bool bvh64, bool a16, bool barycentrics)
    {
        var guest = BuildGuest();
        var rays = new List<Ray>();
        foreach (var type in new uint[] { 0, 1 })
        {
            var node = Pointer(Triangles, type);
            rays.Add(new Ray(node, 100f, [0.25f, 0.25f, 2f], [0f, 0f, -1f]));         // inside triangle 0
            rays.Add(new Ray(node, 100f, [0.75f, 0.75f, 2f], [0f, 0f, -1f]));         // inside triangle 1
            rays.Add(new Ray(node, 100f, [0.5f, 0.5f, 2f], [0f, 0f, -1f]));           // on the shared edge
            rays.Add(new Ray(node, 100f, [1f, 0f, 2f], [0f, 0f, -1f]));               // on the shared vertex v1
            rays.Add(new Ray(node, 100f, [0.25f, 0.25f, -2f], [0f, 0f, 1f]));         // from below: back face
            rays.Add(new Ray(node, 100f, [0.25f, 0.25f, -2f], [0f, 0f, -1f]));        // triangle behind the origin
            rays.Add(new Ray(node, 100f, [2f, 2f, 2f], [0f, 0f, -1f]));               // miss
            rays.Add(new Ray(node, 100f, [-1f, 0.25f, 1f], [1f, 0f, 0f]));            // parallel to the plane
            rays.Add(new Ray(node, 100f, [-1f, 0.25f, 0f], [1f, 0f, 0f]));            // inside the plane
            rays.Add(new Ray(node, 100f, [0.25f, 0.25f, 2f], [0.125f, -0.25f, -1f])); // oblique
            rays.Add(new Ray(node, 100f, [0.25f, 0.25f, 2f], [float.NaN, 0f, -1f]));
            rays.Add(new Ray(node, 100f, [0.25f, 0.25f, 2f], [float.PositiveInfinity, 0f, -1f]));
        }

        rays.Add(new Ray(Pointer(Swizzled, 0), 100f, [0.25f, 0.5f, 2f], [0f, 0f, -1f]));
        rays.Add(new Ray(Pointer(Swizzled, 1), 100f, [0.75f, 0.75f, 2f], [0f, 0f, -1f]));
        var results = Run(guest, new Variant(bvh64, a16, Descriptor(barycentrics: barycentrics)), rays);
        if (results is null) return;

        // Straight down onto z = 0 from z = 2: t = t_num / t_denom = 2.
        Assert.Equal(2f, F(results[0][0]) / F(results[0][1]));
        Assert.True(F(results[0][1]) > 0f);
        if (barycentrics)
        {
            Assert.Equal(0.25f, F(results[0][2]) / F(results[0][1]));
            Assert.Equal(0.25f, F(results[0][3]) / F(results[0][1]));
        }
        else
        {
            Assert.Equal(0xA5A5_0009u, results[0][2]);
            Assert.Equal(1u, results[0][3]);
        }

        // The shared edge belongs to exactly one of the two triangles.
        Assert.NotEqual(IsHit(results[2], barycentrics), IsHit(results[14], barycentrics));
        // Back faces hit with a negative determinant; a triangle behind the origin misses.
        Assert.True(F(results[4][1]) < 0f);
        Assert.Equal([Bits(float.PositiveInfinity), Bits(1f), barycentrics ? 0u : 0xA5A5_0009u, 0u], results[5]);
        Assert.Equal([Bits(float.PositiveInfinity), Bits(1f), barycentrics ? 0u : 0xA5A5_0009u, 0u], results[6]);
        if (barycentrics)
        {
            // Swizzled: triangle 0 returns (w, u), triangle 1 (selector 3 -> 0, v).
            var det = F(results[24][1]);
            Assert.Equal(0.5f, F(results[24][2]) / det);
            Assert.Equal(0.25f, F(results[24][3]) / det);
            Assert.Equal(0u, results[25][2]);
        }
    }

    // User nodes (type 7), the unused triangle slots 2 and 3, a node past the T#'s size and
    // a float32 box whose second unit is past it all return four 0xFFFFFFFF.
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void OtherNodeTypesAndOutOfBoundsNodesReturnAllOnes(bool bvh64)
    {
        var guest = BuildGuest();
        var rays = new[]
        {
            AlongX(Pointer(Triangles, 2)),
            AlongX(Pointer(Triangles, 3)),
            AlongX(Pointer(LastNode + 1, TypeNull)),
            AlongX(Pointer(Triangles, 7)),
            AlongX(Pointer(LastNode + 1, TypeBox16)),
            AlongX(Pointer(LastNode, TypeBox32)),
            AlongX(Pointer(LastNode, TypeBox16)),
            AlongX(Pointer(Straddling, TypeBox32)),
        };
        var results = Run(guest, new Variant(bvh64, false, Descriptor(sort: true)), rays);
        if (results is null) return;

        for (var lane = 0; lane < 6; lane++)
            Assert.Equal([Invalid, Invalid, Invalid, Invalid], results[lane]);
        Assert.Equal([0x30u, 0x10u, 0x20u, Invalid], results[6]);
        Assert.Equal([0x30u, 0x10u, 0x20u, Invalid], results[7]);
    }

    // RADV's layout: a zero base and the whole size, with 64-bit pointers that carry the
    // node's absolute address. A node on an unmapped page reads zeros and records the page.
    [Fact]
    public void AZeroBaseTakesAbsolutePointersAndAnUnmappedNodeFaults()
    {
        var guest = BuildGuest();
        ulong Absolute(uint index, uint type) => ((GuestBase + index * 64ul) >> 3) | type;
        var unmapped = GuestBase + 0x10_0000;
        var rays = new[]
        {
            AlongX(Absolute(Box32, TypeBox32)),
            new Ray(Absolute(Triangles, 0), 100f, [0.25f, 0.25f, 2f], [0f, 0f, -1f]),
            AlongX((unmapped >> 3) | TypeBox16),
        };
        var results = Run(guest, new Variant(true, false, [0u, 1u << 31, 0xFFFF_FFFFu, 0x3FFu | (1u << 24) | (8u << 28)]), rays, out var faults);
        if (results is null) return;

        Assert.Equal([0x30u, 0x10u, 0x20u, Invalid], results[0]);
        Assert.Equal(2f, F(results[1][0]) / F(results[1][1]));
        var page = (uint)(unmapped >> DeviceAddressPaging.PageBits);
        Assert.NotEqual(0u, faults[page >> 5] & (1u << (int)(page & 31)));
        var mappedPage = (uint)(GuestBase >> DeviceAddressPaging.PageBits);
        Assert.Equal(0u, faults[mappedPage >> 5] & (1u << (int)(mappedPage & 31)));
    }

    // Random boxes and triangles around the origin against random rays, some axis-aligned
    // (zero direction components, infinite inverse), in every variant.
    [Theory]
    [InlineData(false, false, 1)]
    [InlineData(true, false, 2)]
    [InlineData(false, true, 3)]
    [InlineData(true, true, 4)]
    public void RandomRaysAndNodesMatchTheReference(bool bvh64, bool a16, int seed)
    {
        var random = new Random(seed);
        var guest = BuildGuest();
        const uint first = 16;
        const uint count = 96;
        var pointers = new List<ulong>();
        for (var index = first; index < first + count; index += 2)
        {
            switch (random.Next(3))
            {
                case 0:
                    WriteBox32(guest, index, [index * 8 + 1, index * 8 + 2, index * 8 + 3, index * 8 + 4],
                        Enumerable.Range(0, 4).Select(_ => RandomBox(random)).ToArray());
                    pointers.Add(Pointer(index, TypeBox32));
                    break;
                case 1:
                    WriteBox16(guest, index, [index * 8 + 1, index * 8 + 2, index * 8 + 3, index * 8 + 4],
                        Enumerable.Range(0, 4).Select(_ => RandomBox(random)).ToArray());
                    pointers.Add(Pointer(index, TypeBox16));
                    break;
                default:
                    WriteTriangles(guest, index, Enumerable.Range(0, 4).Select(_ => RandomPoint(random, 4f)).ToArray(), (uint)random.Next());
                    pointers.Add(Pointer(index, (uint)random.Next(2)));
                    break;
            }
        }

        for (var round = 0; round < 6; round++)
        {
            var rays = new List<Ray>();
            for (var lane = 0; lane < Lanes; lane++)
            {
                var direction = RandomPoint(random, 1f);
                for (var axis = 0; axis < 3; axis++)
                    if (random.Next(6) == 0) direction[axis] = random.Next(2) == 0 ? 0f : -0f;
                rays.Add(new Ray(pointers[random.Next(pointers.Count)], random.Next(4) == 0 ? 3f : 100f, RandomPoint(random, 8f), direction));
            }

            var descriptor = Descriptor(sort: random.Next(2) == 0, grow: (uint)random.Next(4) * 3, barycentrics: random.Next(2) == 0);
            if (Run(guest, new Variant(bvh64, a16, descriptor), rays) is null) return;
        }
    }

    // ---- type 6, the base PS5's null node ----

    // The PSR library's qbvh::NODE_NULL: its traversal pushes ENCODED_DEV_NULL (node 3, inside
    // the BVH header, type 6) for a culled instance and as the third child of a two-child
    // node, and drops it on the all-ones return. Whatever bytes a type-6 pointer names, a box
    // node's or the header's, are not read as children.
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TheNullNodeReturnsAllOnesWhateverItPointsAt(bool bvh64)
    {
        const uint encodedDevNull = (3u << 3) | TypeNull;
        Assert.Equal(encodedDevNull, (uint)Pointer(Triangles, TypeNull));
        var rays = new[]
        {
            AlongX(encodedDevNull),
            AlongX(Pointer(Box32, TypeNull)),
            AlongX(Pointer(Box16, TypeNull)),
            AlongX(Pointer(Ties, TypeNull)),
            new Ray(Pointer(Box32, TypeNull), 100f, [0.5f, 0.5f, 2f], [0f, 0f, -1f]),
        };
        var results = Run(BuildGuest(), new Variant(bvh64, false, Descriptor(sort: true)), rays);
        if (results is null) return;

        foreach (var result in results)
            Assert.Equal([Invalid, Invalid, Invalid, Invalid], result);
    }

    // ---- guest BVH ----

    private static ulong Pointer(uint index, uint type) => ((ulong)index << 3) | type;

    private static Ray AlongX(ulong pointer) => new(pointer, 100f, [0f, 0.5f, 0.5f], [1f, 0f, 0f]);

    // base_address = GuestBase / 256, size = LastNode, triangle_return_mode, sort, grow; type 8.
    private static uint[] Descriptor(bool sort = true, uint grow = 0, bool barycentrics = true) =>
    [
        (uint)(GuestBase >> 8),
        (uint)(GuestBase >> 40) | (grow << 23) | (sort ? 1u << 31 : 0u),
        LastNode,
        (barycentrics ? 1u << 24 : 0u) | (8u << 28),
    ];

    private static byte[] BuildGuest()
    {
        var guest = new byte[GuestBytes];
        // Boxes along +x at x in [1,2], [3,4], [0.5,6] and [0,1]; child 3 is stored invalid.
        float[][] row =
        [
            [1f, 0f, 0f, 2f, 1f, 1f],
            [3f, 0f, 0f, 4f, 1f, 1f],
            [0.5f, 0f, 0f, 6f, 1f, 1f],
            [0f, 0f, 0f, 1f, 1f, 1f],
        ];
        WriteBox32(guest, Box32, [0x10, 0x20, 0x30, Invalid], row);
        WriteBox16(guest, Box16, [0x10, 0x20, 0x30, Invalid], row);
        WriteBox16(guest, LastNode, [0x10, 0x20, 0x30, Invalid], row);
        WriteBox32(guest, Straddling, [0x10, 0x20, 0x30, Invalid], row);

        // A NaN box (RADV's empty child), an infinite one, an inverted one and one that
        // ends just before it starts along x, for the grow test.
        var nan = float.NaN;
        var inf = float.PositiveInfinity;
        float[][] special =
        [
            [nan, nan, nan, nan, nan, nan],
            [-inf, -inf, -inf, inf, inf, inf],
            [2f, 1f, 1f, 1f, 0f, 0f],
            [1f, 0f, 0f, BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(1f) - 1), 1f, 1f],
        ];
        WriteBox32(guest, SpecialBox32, [0x41, 0x42, 0x43, 0x44], special);
        WriteBox32(guest, Ties, [0x51, 0x52, 0x53, 0x54],
            [[1f, 0f, 0f, 2f, 1f, 1f], [1f, 0f, 0f, 3f, 1f, 1f], [1f, 0f, 0f, 4f, 1f, 1f], [0.5f, 0f, 0f, 5f, 1f, 1f]]);
        WriteBox16(guest, SpecialBox16, [0x61, 0x62, 0x63, 0x64], special);

        // The unit quad in z = 0 as two triangles sharing v1-v2; triangle id 0xA5A50009 is
        // RADV's identity swizzle (I = second vertex, J = third) for triangle 0.
        float[][] quad = [[0f, 0f, 0f], [1f, 0f, 0f], [0f, 1f, 0f], [1f, 1f, 0f]];
        WriteTriangles(guest, Triangles, quad, 0xA5A5_0009);
        // Triangle 0: I <- third vertex (2), J <- first (0); triangle 1: I <- 3, J <- second (1).
        WriteTriangles(guest, Swizzled, quad, (2u | (0u << 2)) | ((3u | (1u << 2)) << 8));
        return guest;
    }

    private static void WriteBox32(byte[] guest, uint index, uint[] children, float[][] boxes)
    {
        var node = guest.AsSpan((int)index * 64, 128);
        for (var child = 0; child < 4; child++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(node[(child * 4)..], children[child]);
            for (var component = 0; component < 6; component++)
                BinaryPrimitives.WriteSingleLittleEndian(node[(16 + (child * 6 + component) * 4)..], boxes[child][component]);
        }
    }

    // Halves rounded outward, as a builder stores them (RADV radv_f32_to_f16_neg_inf/pos_inf).
    private static void WriteBox16(byte[] guest, uint index, uint[] children, float[][] boxes)
    {
        var node = guest.AsSpan((int)index * 64, 64);
        for (var child = 0; child < 4; child++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(node[(child * 4)..], children[child]);
            for (var component = 0; component < 6; component++)
            {
                var value = boxes[child][component];
                var half = (Half)value;
                if (!float.IsNaN(value) && (component < 3 ? (float)half > value : (float)half < value))
                    half = component < 3 ? Half.BitDecrement(half) : Half.BitIncrement(half);
                BinaryPrimitives.WriteUInt16LittleEndian(node[(16 + (child * 6 + component) * 2)..], BitConverter.HalfToUInt16Bits(half));
            }
        }
    }

    private static void WriteTriangles(byte[] guest, uint index, float[][] vertices, uint triangleId)
    {
        var node = guest.AsSpan((int)index * 64, 64);
        for (var vertex = 0; vertex < 4; vertex++)
            for (var axis = 0; axis < 3; axis++)
                BinaryPrimitives.WriteSingleLittleEndian(node[((vertex * 3 + axis) * 4)..], vertices[vertex][axis]);
        BinaryPrimitives.WriteUInt32LittleEndian(node[48..], 0x8000_0003);
        BinaryPrimitives.WriteUInt32LittleEndian(node[52..], 11);
        BinaryPrimitives.WriteUInt32LittleEndian(node[56..], 12);
        BinaryPrimitives.WriteUInt32LittleEndian(node[60..], triangleId);
    }

    private static float[] RandomPoint(Random random, float scale) =>
        [(float)(random.NextDouble() * 2 - 1) * scale, (float)(random.NextDouble() * 2 - 1) * scale, (float)(random.NextDouble() * 2 - 1) * scale];

    private static float[] RandomBox(Random random)
    {
        var a = RandomPoint(random, 6f);
        var b = RandomPoint(random, 6f);
        return [MathF.Min(a[0], b[0]), MathF.Min(a[1], b[1]), MathF.Min(a[2], b[2]), MathF.Max(a[0], b[0]), MathF.Max(a[1], b[1]), MathF.Max(a[2], b[2])];
    }

    // ---- program and execution ----

    // v1 = lane * 64; v[2:13] = the lane's VADDR words; s[12:15] = the T#; the four result
    // dwords go to OutputOffset + lane * 64.
    private static Gen5MslShader Compile(Variant variant)
    {
        const uint load = 0xE0381000;   // buffer_load_dwordx4 offen
        const uint store = 0xE0781000;  // buffer_store_dwordx4 offen
        uint MubufAddress(uint data) => 1u | (data << 8) | (2u << 16) | (0x80u << 24);
        var words = new List<uint>
        {
            0x34020086,                         // v_lshlrev_b32 v1, 6, v0
            load | 0, MubufAddress(2),          // v[2:5]
            load | 16, MubufAddress(6),         // v[6:9]
            load | 32, MubufAddress(10),        // v[10:13]
            0xBF8C3F70,                         // s_waitcnt vmcnt(0)
        };
        for (var dword = 0u; dword < 4; dword++)
        {
            words.Add(0xBE8003FFu | ((12 + dword) << 16)); // s_mov_b32 s[12 + n], literal
            words.Add(variant.Descriptor[dword]);
        }

        var opcode = variant.Bvh64 ? 0x67u : 0x66u;
        words.Add(0xF0009F01u | (opcode << 18));          // dmask 0xf, unorm, r128, opm
        words.Add(2u | (16u << 8) | (3u << 16) | (variant.A16 ? 1u << 30 : 0u));
        words.Add(0xBF8C3F70);
        words.Add(store | (uint)OutputOffset);
        words.Add(MubufAddress(16));
        words.Add(0xBF810000);                              // s_endpgm

        var fixture = new Gen5ComputeFixture("bvh-intersect", [.. words], StoreScalarResourceBase: 8, StoreBackingBytes: OutputOffset + Lanes * LaneBytes);
        var program = Gen5ComputeFixtures.DecodeOrThrow(fixture);
        var expected = variant.Bvh64 ? "ImageBvh64IntersectRay" : "ImageBvhIntersectRay";
        Assert.Contains(program.Instructions, instruction => instruction.Opcode == expected &&
            instruction.Control is Gen5RayIntersectControl { A16: var a16 } && a16 == variant.A16);
        return Gen5ComputeFixtures.CompileRequestOrThrow(fixture);
    }

    private static uint[] VaddrWords(Ray ray, Variant variant, out float[] direction, out float[] inverse)
    {
        var words = new List<uint> { (uint)ray.Pointer };
        if (variant.Bvh64) words.Add((uint)(ray.Pointer >> 32));
        words.Add(Bits(ray.Extent));
        words.AddRange(ray.Origin.Select(Bits));
        direction = ray.Direction;
        inverse = ray.Inverse;
        if (!variant.A16)
        {
            words.AddRange(direction.Select(Bits));
            words.AddRange(inverse.Select(Bits));
            return [.. words];
        }

        // The instruction sees the halves; the reference takes them widened back.
        var halves = direction.Concat(inverse).Select(value => (uint)BitConverter.HalfToUInt16Bits((Half)value)).ToArray();
        direction = [.. halves[..3].Select(Gen5BvhReference.Half)];
        inverse = [.. halves[3..].Select(Gen5BvhReference.Half)];
        for (var pair = 0; pair < 3; pair++)
            words.Add(halves[pair * 2] | (halves[pair * 2 + 1] << 16));
        return [.. words];
    }

    private uint[][]? Run(byte[] guest, Variant variant, IReadOnlyList<Ray> rays) => Run(guest, variant, rays, out _);

    private uint[][]? Run(byte[] guest, Variant variant, IReadOnlyList<Ray> rays, out uint[] faults)
    {
        faults = [];
        Assert.InRange(rays.Count, 1, Lanes);
        var shader = Compile(variant);
        var input = new byte[OutputOffset + Lanes * LaneBytes];
        var expected = new uint[rays.Count][];
        uint ReadGuest(ulong address)
        {
            var offset = address - GuestBase;
            return address >= GuestBase && offset + 4 <= GuestBytes ? BinaryPrimitives.ReadUInt32LittleEndian(guest.AsSpan((int)offset)) : 0u;
        }

        for (var lane = 0; lane < rays.Count; lane++)
        {
            var words = VaddrWords(rays[lane], variant, out var direction, out var inverse);
            for (var word = 0; word < words.Length; word++)
                BinaryPrimitives.WriteUInt32LittleEndian(input.AsSpan(lane * LaneBytes + word * 4), words[word]);
            expected[lane] = Gen5BvhReference.Intersect(ReadGuest, variant.Descriptor, rays[lane].Pointer, rays[lane].Extent,
                rays[lane].Origin, direction, inverse);
        }

        if (!RequireDevice()) return null;
        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var error), error);
        var (tables, faultBuffer) = MetalNative.CreateGuestMapping(GuestBase, guest, FaultWords);
        Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, input, new byte[PushData.ByteSize],
            shader.ArgumentLayout!, Lanes, out var result, out _, out error, tables: tables), error);
        faults = MetalNative.ReadWords(faultBuffer, (int)FaultWords);

        var actual = new uint[rays.Count][];
        for (var lane = 0; lane < rays.Count; lane++)
        {
            actual[lane] = new uint[4];
            for (var component = 0; component < 4; component++)
                actual[lane][component] = BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan(OutputOffset + lane * LaneBytes + component * 4));
            var box = (rays[lane].Pointer & 7) is TypeBox16 or TypeBox32;
            for (var component = 0; component < 4; component++)
            {
                var same = actual[lane][component] == expected[lane][component] ||
                    (!box && float.IsNaN(F(actual[lane][component])) && float.IsNaN(F(expected[lane][component])));
                if (!same)
                    output.WriteLine($"lane {lane} ptr=0x{rays[lane].Pointer:X} dword {component}: gpu 0x{actual[lane][component]:X8} reference 0x{expected[lane][component]:X8}");
                Assert.True(same, $"lane {lane} dword {component}: gpu 0x{actual[lane][component]:X8}, reference 0x{expected[lane][component]:X8}");
            }
        }

        return actual;
    }

    private bool RequireDevice()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Numerical execution requires the exclusive Metal device gate.");
            return false;
        }

        Assert.True(MetalNative.IsAvailable, "The required execution gate needs a Metal device.");
        return true;
    }

    private static bool IsHit(uint[] result, bool barycentrics) =>
        barycentrics ? result[0] != Bits(float.PositiveInfinity) : result[3] == 1u;

    private static uint Bits(float value) => BitConverter.SingleToUInt32Bits(value);

    private static float F(uint bits) => BitConverter.UInt32BitsToSingle(bits);
}
