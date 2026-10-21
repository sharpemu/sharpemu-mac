// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>
/// The pixel input VGPRs and V_INTERP slots, mirroring the SPIR-V translator's
/// Gen5InterpolationParameterTests: position and system inputs, per-vertex inputs
/// (metal::vertex_value, MSL 4.0) and slots that share one VS parameter. The device
/// tests draw through a real render pipeline and read the exported registers back.
/// </summary>
public sealed class Gen5MslPixelInputTests(ITestOutputHelper output)
{
    private const uint PerspCenter = 1u << 1;
    private const uint PullModel = 1u << 3;
    private const uint PosW = 1u << 11;
    private const uint FrontFace = 1u << 12;
    private const uint Ancillary = 1u << 13;
    private const uint SampleCoverage = 1u << 14;
    private const uint PosFixedPoint = 1u << 15;

    // VINTRP op: 0 = V_INTERP_P1_F32, 1 = V_INTERP_P2_F32, 2 = V_INTERP_MOV_F32.
    private static uint Interp(uint vdst, uint op, uint attribute, uint channel, uint vsrc) =>
        0xC8000000u | (vdst << 18) | (op << 16) | (attribute << 10) | (channel << 8) | vsrc;

    // exp mrt0 v(a), v(b), v(c), v(d) done vm, then s_endpgm.
    private static uint[] ExportAndEnd(uint a, uint b, uint c, uint d) =>
        [0xF800180F, a | (b << 8) | (c << 16) | (d << 24), 0xBF810000];

    [Theory]
    [InlineData(PosW, true)]
    [InlineData(1u << 10, false)]
    public void PositionW_IsTheReciprocalOfTheFragmentCoordinate(uint inputs, bool reciprocal)
    {
        var shader = Compile(ExportAndEnd(0, 0, 0, 0), inputs);
        Assert.Equal(reciprocal, shader.Source.Contains(
            "v0 = as_type<uint>(1.0f / sharpemu_in.sharpemu_frag_coord.w);", StringComparison.Ordinal));
        Assert.Equal(!reciprocal, shader.Source.Contains(
            "v0 = as_type<uint>(sharpemu_in.sharpemu_frag_coord.z);", StringComparison.Ordinal));
    }

    [Fact]
    public void PixelSystemInputs_ReadTheirAttributes()
    {
        var shader = Compile(
            ExportAndEnd(2, 3, 4, 5), PerspCenter | FrontFace | Ancillary | SampleCoverage | PosFixedPoint);
        Assert.Contains("bool sharpemu_front_facing [[front_facing]];", shader.Source, StringComparison.Ordinal);
        Assert.Contains("uint sharpemu_layer [[render_target_array_index]];", shader.Source, StringComparison.Ordinal);
        Assert.Contains("uint sharpemu_sample_mask [[sample_mask]];", shader.Source, StringComparison.Ordinal);

        // PERSP_CENTER holds v0/v1, so the system inputs follow from v2 in bit order.
        Assert.Contains("v2 = (sharpemu_in.sharpemu_front_facing ? 0x3F800000u : 0xBF800000u);", shader.Source, StringComparison.Ordinal);
        Assert.Contains("v3 = (sharpemu_in.sharpemu_layer << 16);", shader.Source, StringComparison.Ordinal);
        Assert.Contains("v4 = sharpemu_in.sharpemu_sample_mask;", shader.Source, StringComparison.Ordinal);
        Assert.Contains("v5 = ((uint(sharpemu_in.sharpemu_frag_coord.x) & 0xFFFFu) | " +
            "(uint(sharpemu_in.sharpemu_frag_coord.y) << 16));", shader.Source, StringComparison.Ordinal);
        AssertCompiles(shader);
    }

    [Fact]
    public void AddressedButDisabledSystemInputs_ReserveTheirRegisters()
    {
        var shader = Compile(ExportAndEnd(0, 1, 2, 3), FrontFace | Ancillary | PosFixedPoint, enable: PosFixedPoint);
        Assert.DoesNotContain("[[front_facing]]", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("[[render_target_array_index]]", shader.Source, StringComparison.Ordinal);
        Assert.Contains("v2 = ((uint(sharpemu_in.sharpemu_frag_coord.x)", shader.Source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(0u, false, "second", true)]
    [InlineData(1u, false, "third", true)]
    [InlineData(0u, true, "second", false)]
    [InlineData(1u, true, "third", false)]
    public void ParameterMove_SelectsVertexAndPreservesCustomData(
        uint selector, bool custom, string vertex, bool subtractOrigin)
    {
        var shader = Compile([Interp(4, 2, 1, 2, selector), .. ExportAndEnd(4, 4, 4, 4)], PerspCenter,
            cntl: [0, 0x401], customMask: custom ? 2u : 0u);
        Assert.True(shader.UsesPerVertexInputs);
        Assert.Contains("vertex_value<float4> attr1 [[user(locn1)]];", shader.Source, StringComparison.Ordinal);
        Assert.Contains($"sharpemu_in.attr1.get(vertex_index::{vertex})[2]", shader.Source, StringComparison.Ordinal);
        Assert.Equal(subtractOrigin, shader.Source.Contains(
            "- sharpemu_in.attr1.get(vertex_index::first)[2]", StringComparison.Ordinal));
        AssertCompiles(shader);
    }

    [Theory]
    [InlineData(3u)]
    [InlineData(255u)]
    public void ReservedSelector_Fails(uint selector)
    {
        var request = Request([Interp(4, 2, 1, 0, selector), Interp(5, 2, 1, 0, 0), .. ExportAndEnd(4, 5, 4, 4)],
            PerspCenter, PerspCenter, [0, 1], 0, true);
        Assert.False(Gen5MslTranslator.TryCompileProgram(request, out _, out var error));
        Assert.Contains("reserved interpolation parameter selector", error, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ProvokingVertexMove_IsAFlatInput(bool supportsPerVertex)
    {
        var shader = Compile([Interp(4, 2, 1, 0, 2), .. ExportAndEnd(4, 4, 4, 4)], PerspCenter,
            cntl: [0, 1], supportsPerVertex: supportsPerVertex);
        Assert.False(shader.UsesPerVertexInputs);
        Assert.Contains("float4 attr1 [[user(locn1), flat]];", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("vertex_value", shader.Source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(0u)]
    [InlineData(1u)]
    public void VertexDifferenceMove_FallsBackToInterpolatedInputWithoutPerVertexSupport(uint selector)
    {
        var shader = Compile([Interp(4, 2, 1, 0, selector), .. ExportAndEnd(4, 4, 4, 4)], PerspCenter,
            cntl: [0, 1], supportsPerVertex: false);
        Assert.False(shader.UsesPerVertexInputs);
        Assert.Contains("float4 attr1 [[user(locn1)]];", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("barycentric_coord", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void OrdinaryInterpolation_DoesNotNeedPerVertexInputs()
    {
        var shader = Compile([Interp(4, 0, 1, 0, 0), Interp(4, 1, 1, 0, 1), .. ExportAndEnd(4, 4, 4, 4)], PerspCenter,
            cntl: [0, 1]);
        Assert.False(shader.UsesPerVertexInputs);
        Assert.DoesNotContain("barycentric_coord", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void MixedBarycentricLocations_SeedEveryInterpolationSlot()
    {
        // PERSP_SAMPLE/CENTER/CENTROID and LINEAR_SAMPLE/CENTER/CENTROID: Metal has one
        // perspective and one linear barycentric, both at the pixel center here.
        // v0..v5 are the perspective slots' I/J and v6..v11 the linear ones'.
        var perspective = Compile([Interp(20, 2, 1, 0, 0), .. ExportAndEnd(0, 2, 4, 20)], 0x77, cntl: [0, 1]);
        var linear = Compile([Interp(20, 2, 1, 0, 0), .. ExportAndEnd(7, 9, 11, 20)], 0x77, cntl: [0, 1]);
        Assert.Contains("float3 sharpemu_bary_persp [[barycentric_coord, center_perspective]];", perspective.Source, StringComparison.Ordinal);
        Assert.Contains("float3 sharpemu_bary_linear [[barycentric_coord, center_no_perspective]];", perspective.Source, StringComparison.Ordinal);
        for (uint slot = 0; slot < 3; slot++)
        {
            Assert.Contains($"v{slot * 2} = as_type<uint>(sharpemu_in.sharpemu_bary_persp.y);", perspective.Source, StringComparison.Ordinal);
            Assert.Contains($"v{6 + (slot * 2) + 1} = as_type<uint>(sharpemu_in.sharpemu_bary_linear.z);", linear.Source, StringComparison.Ordinal);
        }

        AssertCompiles(perspective);
    }

    public static TheoryData<uint, bool> BarycentricLocations
    {
        get
        {
            var cases = new TheoryData<uint, bool>();
            foreach (var inputs in new uint[] { 0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x11, 0x44, 0x77, 0xF844 })
            {
                cases.Add(inputs, false);
                cases.Add(inputs, true);
            }
            return cases;
        }
    }

    [Theory]
    [MemberData(nameof(BarycentricLocations))]
    public void BarycentricLocations_CompileWithCenterBuiltIns(uint inputs, bool fromPrimitive)
    {
        // ASTRO BOT uses per-primitive vertex copies, compiled as MSL 3.2. Testing only
        // vertex_value (MSL 4.0) misses that compiler's rejection of centroid/sample.
        var shader = Compile([Interp(20, 2, 1, 0, 0), .. ExportAndEnd(0, 1, 20, 20)], inputs, cntl: [0, 1],
            supportsPerVertex: !fromPrimitive, fromPrimitive: fromPrimitive);
        if ((inputs & 0xFu) != 0)
        {
            if (fromPrimitive && (inputs & 0x70u) != 0)
            {
                Assert.Contains("sharpemu_bary_persp /= dot(sharpemu_bary_persp, float3(1.0f));", shader.Source, StringComparison.Ordinal);
                Assert.DoesNotContain("[[barycentric_coord, center_perspective]]", shader.Source, StringComparison.Ordinal);
            }
            else
            {
                Assert.Contains("[[barycentric_coord, center_perspective]]", shader.Source, StringComparison.Ordinal);
            }
        }
        if ((inputs & 0x70u) != 0)
        {
            Assert.Contains("[[barycentric_coord, center_no_perspective]]", shader.Source, StringComparison.Ordinal);
        }
        AssertCompiles(shader);
    }

    [Fact]
    public void PullModel_SeedsTheScreenLinearCoordinates()
    {
        var shader = Compile([Interp(20, 2, 1, 0, 0), .. ExportAndEnd(2, 3, 4, 20)], PerspCenter | PullModel,
            cntl: [0, 1]);
        Assert.Contains("v2 = as_type<uint>(sharpemu_in.sharpemu_bary_persp.y * sharpemu_in.sharpemu_frag_coord.w);",
            shader.Source, StringComparison.Ordinal);
        Assert.Contains("v3 = as_type<uint>(sharpemu_in.sharpemu_bary_persp.z * sharpemu_in.sharpemu_frag_coord.w);",
            shader.Source, StringComparison.Ordinal);
        Assert.Contains("v4 = as_type<uint>(sharpemu_in.sharpemu_frag_coord.w);", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void SlotsReadingOneParameter_ShareOneInput()
    {
        // PS slots 1 and 2 both read VS parameter 1; the second slot must not move to a
        // location the vertex program never writes.
        var shader = Compile([Interp(4, 2, 1, 0, 1), Interp(5, 2, 2, 0, 1), .. ExportAndEnd(4, 5, 4, 4)], PerspCenter,
            cntl: [0, 1, 1], customMask: 6);
        Assert.Contains("vertex_value<float4> attr1 [[user(locn1)]];", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("attr2", shader.Source, StringComparison.Ordinal);
        Assert.Contains("v5 = as_type<uint>(sharpemu_in.attr1.get(vertex_index::third)[0]);", shader.Source, StringComparison.Ordinal);
        AssertCompiles(shader);
    }

    [Fact]
    public void FlatAndSmoothSlotsOfOneParameter_KeepOneInputOfEachKind()
    {
        var shader = Compile([Interp(4, 2, 1, 0, 2), Interp(5, 1, 2, 0, 1), Interp(6, 1, 3, 0, 1), .. ExportAndEnd(4, 5, 6, 4)],
            PerspCenter, cntl: [0, 1, 1, 1]);
        Assert.Contains("float4 attr1 [[user(locn1), flat]];", shader.Source, StringComparison.Ordinal);
        Assert.Contains("float4 attr2 [[user(", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("attr3", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void SmoothSlotSharingAPerVertexParameter_InterpolatesThePerVertexInput()
    {
        // Slot 1 is read per vertex, slot 2 interpolates the same VS parameter: one per-vertex
        // input serves both, and slot 2 is rebuilt from the vertices with the barycentrics.
        var shader = Compile([Interp(4, 2, 1, 0, 1), Interp(5, 1, 2, 0, 1), .. ExportAndEnd(4, 5, 4, 4)], PerspCenter,
            cntl: [0, 1, 1], customMask: 2);
        Assert.Contains("vertex_value<float4> attr1 [[user(locn1)]];", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("attr2", shader.Source, StringComparison.Ordinal);
        Assert.Contains("[[barycentric_coord, center_perspective]]", shader.Source, StringComparison.Ordinal);
        Assert.Contains(
            "(sharpemu_in.attr1.get(vertex_index::first)[0] * sharpemu_in.sharpemu_bary_persp.x + " +
            "sharpemu_in.attr1.get(vertex_index::second)[0] * sharpemu_in.sharpemu_bary_persp.y + " +
            "sharpemu_in.attr1.get(vertex_index::third)[0] * sharpemu_in.sharpemu_bary_persp.z)",
            shader.Source, StringComparison.Ordinal);
        AssertCompiles(shader);
    }

    // ---- device ----

    private const uint TargetWidth = 16;
    private const uint TargetHeight = 8;

    // Two triangles with w = 2 (clip xyz scaled with it): the left one wound one way
    // and the right one the other, so exactly one of them is front facing.
    private const string SystemInputVertexSource = """
        #include <metal_stdlib>
        using namespace metal;
        struct VsOut { float4 position [[position]]; };
        vertex VsOut system_vs(uint id [[vertex_id]])
        {
            const float2 corners[6] = {
                float2(-1.0, -1.0), float2(-1.0, 3.0), float2(0.0, -1.0),
                float2(0.0, -1.0), float2(1.0, -1.0), float2(1.0, 3.0) };
            VsOut out;
            out.position = float4(corners[id] * 2.0, 0.5 * 2.0, 2.0);
            return out;
        }
        """;

    [Theory]
    [InlineData(false, 0x3F800000u, 0xBF800000u)]
    [InlineData(true, 0xFFFFFFFFu, 0u)]
    public void SystemInputsRenderTheirHardwareEncodings(bool frontFaceAllBits, uint front, uint back)
    {
        if (SkipWithoutMetalDevice("the pixel system input render test")) return;

        var inputs = PerspCenter | PosW | FrontFace | Ancillary | SampleCoverage | PosFixedPoint;
        // v0/v1 PERSP_CENTER, v2 POS_W, v3 FRONT_FACE, v4 ANCILLARY, v5 SAMPLE_COVERAGE, v6 POS_FIXED_PT.
        var first = Render(SystemInputVertexSource, "system_vs", 6, ExportAndEnd(2, 3, 5, 6), inputs,
            Gen5PixelOutputKind.Uint, languageVersion: null, frontFaceAllBits: frontFaceAllBits);
        var second = Render(SystemInputVertexSource, "system_vs", 6, ExportAndEnd(4, 4, 4, 4), inputs,
            Gen5PixelOutputKind.Uint, languageVersion: null);

        uint Texel(uint[] texels, uint x, uint y, int component) => texels[(((y * TargetWidth) + x) * 4) + component];
        const uint leftX = 2, rightX = TargetWidth - 3, y = 5;

        // POS_W is W itself (2), not the 1/W Metal's [[position]].w carries.
        Assert.Equal(BitConverter.SingleToUInt32Bits(2.0f), Texel(first, leftX, y, 0));
        Assert.Equal(BitConverter.SingleToUInt32Bits(2.0f), Texel(first, rightX, y, 0));

        // FRONT_FACE is +1.0/-1.0, or all ones/zero with FRONT_FACE_ALL_BITS.
        var faces = new[] { Texel(first, leftX, y, 1), Texel(first, rightX, y, 1) };
        Assert.Contains(front, faces);
        Assert.Contains(back, faces);

        // A single-sample target covers sample 0.
        Assert.Equal(1u, Texel(first, leftX, y, 2));

        // POS_FIXED_PT packs the integer pixel position.
        Assert.Equal(leftX | (y << 16), Texel(first, leftX, y, 3));
        Assert.Equal(rightX | (y << 16), Texel(first, rightX, y, 3));

        // ANCILLARY's layer field reads 0 when the vertex function writes no layer.
        Assert.Equal(0u, Texel(second, leftX, y, 0));
    }

    // One triangle whose param0.x and param1.x carry 10, 20 and 40 at its three vertices.
    private const string PerVertexVertexSource = """
        #include <metal_stdlib>
        using namespace metal;
        struct VsOut
        {
            float4 position [[position]];
            float4 param0 [[user(locn0)]];
            float4 param1 [[user(locn1)]];
        };
        vertex VsOut per_vertex_vs(uint id [[vertex_id]])
        {
            const float2 corners[3] = { float2(-1.0, -1.0), float2(-1.0, 3.0), float2(3.0, -1.0) };
            const float values[3] = { 10.0, 20.0, 40.0 };
            VsOut out;
            out.position = float4(corners[id], 0.5, 1.0);
            out.param0 = float4(values[id], 0.0, 0.0, 1.0);
            out.param1 = out.param0;
            return out;
        }
        """;

    [Fact]
    public void PerVertexInputsRebuildTheTriangleParameters()
    {
        if (SkipWithoutMetalDevice("the per-vertex pixel input render test")) return;
        if (!MetalNative.SupportsPerVertexPixelInputs)
        {
            // Apple9 and older compile vertex_value but read garbage from it, which is why
            // the V2 host reports per-vertex inputs only on Apple10.
            output.WriteLine("Per-vertex pixel inputs need MSL 4.0 on an Apple10 GPU; the render test skipped.");
            return;
        }

        // v0/v1 hold PERSP_CENTER's I/J. attr0 is read per vertex: P10 into v4, P20 into
        // v5, and P1/P2 over I/J into v6; attr1 is read only through P0, a flat input.
        uint[] words =
        [
            Interp(4, 2, 0, 0, 0),
            Interp(5, 2, 0, 0, 1),
            Interp(6, 0, 0, 0, 0),
            Interp(6, 1, 0, 0, 1),
            Interp(7, 2, 1, 0, 2),
            .. ExportAndEnd(4, 5, 6, 7),
        ];
        var texels = Render(PerVertexVertexSource, "per_vertex_vs", 3, words, PerspCenter,
            Gen5PixelOutputKind.Float, Gen5MslShader.PerVertexInputsLanguageVersion);

        // Window-space vertices of the triangle over the target (y down).
        var vertices = new (double X, double Y)[] { (0, TargetHeight * 1.0), (0, -TargetHeight * 1.0), (TargetWidth * 2.0, TargetHeight * 1.0) };
        double[] values = [10, 20, 40];
        foreach (var (x, y) in new (uint, uint)[] { (1, 1), (7, 3), (12, 6) })
        {
            float Component(int component) =>
                BitConverter.UInt32BitsToSingle(texels[(((y * TargetWidth) + x) * 4) + component]);

            Assert.Equal(10f, Component(0));
            Assert.Equal(30f, Component(1));
            Assert.Equal(10f, Component(3));
            var expected = Interpolate(vertices, values, x + 0.5, y + 0.5);
            Assert.True(Math.Abs(Component(2) - expected) < 1e-3, $"({x},{y}): {Component(2)} != {expected}");
        }
    }

    [Fact]
    public void SlotsSharingOneParameterRenderItsValue()
    {
        if (SkipWithoutMetalDevice("the shared pixel input render test")) return;

        // Slots 0 and 3 interpolate VS parameter 0; slots 1 and 2 read parameter 1 through
        // P0. Each pair shares one input, so no slot lands on a location the vertex
        // function never writes (which Metal would refuse to link).
        uint[] words =
        [
            Interp(4, 0, 0, 0, 0),
            Interp(4, 1, 0, 0, 1),
            Interp(5, 2, 1, 0, 2),
            Interp(6, 2, 2, 0, 2),
            Interp(7, 0, 3, 0, 0),
            Interp(7, 1, 3, 0, 1),
            .. ExportAndEnd(4, 5, 6, 7),
        ];
        var shader = Compile(words, PerspCenter, cntl: [0, 1, 1, 0]);
        Assert.Contains("float4 attr0 [[user(locn0)]];", shader.Source, StringComparison.Ordinal);
        Assert.Contains("float4 attr1 [[user(locn1), flat]];", shader.Source, StringComparison.Ordinal);
        var texels = Render(PerVertexVertexSource, "per_vertex_vs", 3, words, PerspCenter,
            Gen5PixelOutputKind.Float, languageVersion: null, cntl: [0, 1, 1, 0]);

        var vertices = new (double X, double Y)[] { (0, TargetHeight * 1.0), (0, -TargetHeight * 1.0), (TargetWidth * 2.0, TargetHeight * 1.0) };
        double[] values = [10, 20, 40];
        foreach (var (x, y) in new (uint, uint)[] { (1, 1), (12, 6) })
        {
            float Component(int component) =>
                BitConverter.UInt32BitsToSingle(texels[(((y * TargetWidth) + x) * 4) + component]);

            var expected = Interpolate(vertices, values, x + 0.5, y + 0.5);
            Assert.True(Math.Abs(Component(0) - expected) < 1e-3, $"({x},{y}): {Component(0)} != {expected}");
            Assert.Equal(10f, Component(1));
            Assert.Equal(10f, Component(2));
            Assert.Equal(Component(0), Component(3));
        }
    }

    private static double Interpolate((double X, double Y)[] v, double[] values, double x, double y)
    {
        double Edge((double X, double Y) a, (double X, double Y) b) => ((b.X - a.X) * (y - a.Y)) - ((b.Y - a.Y) * (x - a.X));
        var area = ((v[1].X - v[0].X) * (v[2].Y - v[0].Y)) - ((v[1].Y - v[0].Y) * (v[2].X - v[0].X));
        var b0 = Edge(v[1], v[2]) / area;
        var b1 = Edge(v[2], v[0]) / area;
        var b2 = Edge(v[0], v[1]) / area;
        return (b0 * values[0]) + (b1 * values[1]) + (b2 * values[2]);
    }

    private uint[] Render(
        string vertexSource,
        string vertexEntryPoint,
        uint vertexCount,
        uint[] pixelWords,
        uint inputs,
        Gen5PixelOutputKind outputKind,
        nuint? languageVersion,
        uint[]? cntl = null,
        bool frontFaceAllBits = false,
        uint customMask = 0,
        bool manualInterpolation = false,
        bool fromPrimitive = false)
    {
        var shader = Compile(pixelWords, inputs, cntl: cntl ?? [0, 1], outputKind: outputKind,
            frontFaceAllBits: frontFaceAllBits, customMask: customMask, supportsPerVertex: !manualInterpolation && !fromPrimitive,
            manualInterpolation: manualInterpolation, fromPrimitive: fromPrimitive);
        Assert.True(MetalNative.TryCompileLibrary(vertexSource, out var vertexLibrary, out var error), error);
        nint fragmentLibrary;
        Assert.True(
            languageVersion is { } version
                ? MetalNative.TryCompileLibrary(shader.Source, version, out fragmentLibrary, out error)
                : MetalNative.TryCompileLibrary(shader.Source, out fragmentLibrary, out error),
            $"{error}\n{shader.Source}");
        Assert.True(
            MetalNative.TryRenderTriangles(
                vertexLibrary,
                vertexEntryPoint,
                fragmentLibrary,
                shader.EntryPoint,
                vertexCount,
                TargetWidth,
                TargetHeight,
                outputKind == Gen5PixelOutputKind.Uint
                    ? MetalNative.Rgba32UintPixelFormat
                    : MetalNative.Rgba32FloatPixelFormat,
                out var texels,
                out error),
            $"{error}\n{shader.Source}");
        return texels;
    }

    // ---- VOP3 and f16 interpolation, interpolation locations ----

    // VOP3 interpolation: src0 = attr | chan << 6 | HIGH << 8, src1 = the I/J VGPR (or
    // V_INTERP_MOV's parameter), src2 = S2 (encodings as llvm-mc -mcpu=gfx1030 emits them).
    private static uint[] Vop3Interp(uint opcode, uint vdst, uint attribute, uint channel, uint src1, uint src2 = 0,
        bool high = false, uint abs = 0, uint neg = 0, uint omod = 0, bool clamp = false) =>
    [
        0xD4000000u | (opcode << 16) | (clamp ? 0x8000u : 0u) | (abs << 8) | vdst,
        attribute | (channel << 6) | (high ? 0x100u : 0u) | (src1 << 9) | (src2 << 18) | (omod << 27) | (neg << 29),
    ];

    private const uint OpP1F32 = 0x200, OpP2F32 = 0x201, OpMovF32 = 0x202;
    private const uint OpP1llF16 = 0x342, OpP1lvF16 = 0x343, OpP2F16 = 0x35A;
    private const uint V0 = 256, V1 = 257, V3 = 259, V4 = 260;

    // VOP1 V_CVT_F32_F16 and VOP2 V_SUB_F32 / V_MAC_F32 (D = S0 * S1 + D).
    private const uint OpCvtF32F16 = 0x0B, OpSubF32 = 0x04, OpMacF32 = 0x1F;

    private static uint Vop1(uint op, uint vdst, uint src0) => 0x7E000000u | (vdst << 17) | (op << 9) | src0;

    private static uint Vop2(uint op, uint vdst, uint src0, uint vsrc1) => (op << 25) | (vdst << 17) | (vsrc1 << 9) | src0;

    [Theory]
    [InlineData(1u, false, false, "* 2.0f")]
    [InlineData(3u, true, false, "* 0.5f")]
    [InlineData(0u, true, false, "clamp(")]
    [InlineData(1u, false, true, null)]
    public void Vop3Interpolation_AppliesItsOutputModifiers(uint omod, bool clamp, bool ieee, string? expected)
    {
        var shader = Compile(
            [.. Vop3Interp(OpP1F32, 4, 1, 0, V0), .. Vop3Interp(OpP2F32, 4, 1, 0, V1, omod: omod, clamp: clamp), .. ExportAndEnd(4, 4, 4, 4)],
            PerspCenter, cntl: [0, 1], ieeeMode: ieee);
        Assert.False(shader.UsesPerVertexInputs);
        var store = shader.Source.Split('\n').Last(line => line.Contains("sharpemu_in.attr1[0]", StringComparison.Ordinal));
        if (expected is null)
        {
            // Output modifiers are ignored under IEEE mode.
            Assert.DoesNotContain("* 2.0f", store, StringComparison.Ordinal);
        }
        else
        {
            Assert.Contains(expected, store, StringComparison.Ordinal);
        }

        AssertCompiles(shader);
    }

    [Fact]
    public void Vop3ParameterMove_ReadsItsParameterFromSrc1()
    {
        var shader = Compile([.. Vop3Interp(OpMovF32, 4, 1, 2, 1), .. ExportAndEnd(4, 4, 4, 4)], PerspCenter, cntl: [0, 1]);
        Assert.True(shader.UsesPerVertexInputs);
        Assert.Contains("(sharpemu_in.attr1.get(vertex_index::third)[2] - sharpemu_in.attr1.get(vertex_index::first)[2])",
            shader.Source, StringComparison.Ordinal);

        var provoking = Compile([.. Vop3Interp(OpMovF32, 4, 1, 0, 2), .. ExportAndEnd(4, 4, 4, 4)], PerspCenter, cntl: [0, 1]);
        Assert.Contains("float4 attr1 [[user(locn1), flat]];", provoking.Source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(false, 0)]
    [InlineData(true, 1)]
    public void HalfInterpolation_ReadsOneHalfOfEachVertex(bool high, int half)
    {
        var shader = Compile(
            [.. Vop3Interp(OpP1llF16, 4, 1, 0, V0, high: high), .. Vop3Interp(OpP2F16, 5, 1, 0, V1, 260, high: high),
             .. ExportAndEnd(4, 5, 4, 4)],
            PerspCenter, cntl: [0, 1]);
        Assert.True(shader.UsesPerVertexInputs);
        Assert.Contains($"(float)as_type<half2>(as_type<uint>(sharpemu_in.attr1.get(vertex_index::second)[0]))[{half}]",
            shader.Source, StringComparison.Ordinal);
        // P2_F16 rounds into the low half and keeps the high one.
        Assert.Contains("((v[5] & 0xFFFF0000u) | (uint)as_type<ushort>(half(fma(", shader.Source.Replace("v5", "v[5]"), StringComparison.Ordinal);
        AssertCompiles(shader);
    }

    [Fact]
    public void HalfInterpolationWithS2_TakesTheSelectedHalfAndItsModifiers()
    {
        var shader = Compile(
            [.. Vop3Interp(OpP1lvF16, 4, 1, 0, V0, V3, high: true, abs: 4, neg: 2), .. ExportAndEnd(4, 4, 4, 4)],
            PerspCenter, cntl: [0, 1]);
        Assert.Contains("fabs((float)as_type<half2>(", shader.Source, StringComparison.Ordinal);
        Assert.Contains(")[1])", shader.Source, StringComparison.Ordinal);
        AssertCompiles(shader);
    }

    [Theory]
    [InlineData(0u, false, "attr1_v1", true)]
    [InlineData(1u, false, "attr1_v2", true)]
    [InlineData(0u, true, "attr1_v1", false)]
    [InlineData(1u, true, "attr1_v2", false)]
    public void ParameterMoveFromPrimitive_ReadsTheVertexCopiesOfItsParam(
        uint selector, bool custom, string vertex, bool subtractOrigin)
    {
        // Slot 1 reads param 3 (SPI_PS_INPUT_CNTL.OFFSET), flat-shaded as a custom slot is.
        var shader = Compile([Interp(4, 2, 1, 2, selector), .. ExportAndEnd(4, 4, 4, 4)], PerspCenter,
            cntl: [0, 0x403], customMask: custom ? 2u : 0u, supportsPerVertex: false, fromPrimitive: true);

        // Per-primitive data needs no MSL 4.0 and no vertex_value.
        Assert.False(shader.UsesPerVertexInputs);
        Assert.DoesNotContain("vertex_value", shader.Source, StringComparison.Ordinal);
        for (var index = 0; index < 3; index++)
        {
            Assert.Contains($"float4 attr1_v{index} [[user(locn3_v{index})]];", shader.Source, StringComparison.Ordinal);
        }

        Assert.Contains($"sharpemu_in.{vertex}[2]", shader.Source, StringComparison.Ordinal);
        Assert.Equal(subtractOrigin, shader.Source.Contains("- sharpemu_in.attr1_v0[2]", StringComparison.Ordinal));
        AssertCompiles(shader);
    }

    [Fact]
    public void ManualInterpolationFromPrimitive_SeedsTheBarycentricsOfTheVertexOrder()
    {
        // P10 and P20 read by hand, then weighted with the I/J VGPRs (v0/v1) the way the
        // guest compiler interpolates a packed parameter itself.
        var shader = Compile(
            [Interp(4, 2, 1, 0, 2), Interp(5, 2, 1, 0, 0), Interp(6, 2, 1, 0, 1), .. ExportAndEnd(4, 5, 6, 0)],
            PerspCenter, cntl: [0, 0x401], customMask: 2u, supportsPerVertex: false, fromPrimitive: true);
        Assert.Contains("float3 sharpemu_bary_persp [[barycentric_coord, center_perspective]];", shader.Source,
            StringComparison.Ordinal);
        Assert.Contains("as_type<uint>(sharpemu_in.sharpemu_bary_persp.y)", shader.Source, StringComparison.Ordinal);
        Assert.Contains("as_type<uint>(sharpemu_in.sharpemu_bary_persp.z)", shader.Source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_in.attr1_v0[0]", shader.Source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_in.attr1_v1[0]", shader.Source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_in.attr1_v2[0]", shader.Source, StringComparison.Ordinal);
        AssertCompiles(shader);
    }

    [Fact]
    public void HalfInterpolationFromPrimitive_ReadsOneHalfOfEachVertexCopy()
    {
        var shader = Compile(
            [.. Vop3Interp(OpP1llF16, 4, 1, 0, V0, high: true), .. Vop3Interp(OpP2F16, 5, 1, 0, V1, 260, high: true),
             .. ExportAndEnd(4, 5, 4, 4)],
            PerspCenter, cntl: [0, 1], supportsPerVertex: false, fromPrimitive: true);
        Assert.False(shader.UsesPerVertexInputs);
        Assert.Contains("(float)as_type<half2>(as_type<uint>(sharpemu_in.attr1_v1[0]))[1]", shader.Source,
            StringComparison.Ordinal);
        AssertCompiles(shader);
    }

    // Slot 0 (param 0, custom, flat-shaded) interpolated by hand in f32 from the low f16 half
    // of each vertex: v4..v6 = P0, P1, P2 halves; v5/v6 = P1 - P0, P2 - P0;
    // v4 += I * v5; v4 += J * v6.
    private static readonly uint[] ManualHalfInterpolation =
    [
        Interp(4, 2, 0, 0, 2),
        Interp(5, 2, 0, 0, 0),
        Interp(6, 2, 0, 0, 1),
        Vop1(OpCvtF32F16, 4, V4), Vop1(OpCvtF32F16, 5, V4 + 1), Vop1(OpCvtF32F16, 6, V4 + 2),
        Vop2(OpSubF32, 5, V4 + 1, 4), Vop2(OpSubF32, 6, V4 + 2, 4),
        Vop2(OpMacF32, 4, V0, 5), Vop2(OpMacF32, 4, V1, 6),
        .. ExportAndEnd(4, 4, 4, 4),
    ];

    [Fact]
    public void ManualInterpolation_ReadsTheHostInterpolantOfTheUnpackedHalf()
    {
        var shader = Compile(ManualHalfInterpolation, PerspCenter, cntl: [0x400], customMask: 1u,
            supportsPerVertex: false, manualInterpolation: true);

        // No per-vertex input and no barycentrics: the slot is flat, and the host interpolates
        // the low halves the vertex stage unpacks into locn0_i1.
        Assert.False(shader.UsesPerVertexInputs);
        Assert.DoesNotContain("barycentric_coord", shader.Source, StringComparison.Ordinal);
        Assert.Contains("float4 attr0 [[user(locn0), flat]];", shader.Source, StringComparison.Ordinal);
        Assert.Contains("float4 interp0_1 [[user(locn0_i1)]];", shader.Source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_in.interp0_1[0]", shader.Source, StringComparison.Ordinal);
        AssertCompiles(shader);
    }

    [Fact]
    public void ManualInterpolationWithoutAPlan_IsRefused()
    {
        // The P1 half is exported as well, so the program needs the vertex values themselves.
        uint[] words = [.. ManualHalfInterpolation[..^3], .. ExportAndEnd(4, 5, 4, 4)];
        var request = Request(words, PerspCenter, PerspCenter, [0x400], 1u, supportsPerVertex: false,
            manualInterpolation: true);

        Assert.False(Gen5MslTranslator.TryCompileProgram(request, out _, out var error));
        Assert.Contains("no manual interpolation plan", error, StringComparison.Ordinal);
    }

    // One triangle whose param0.x carries the f16 halves 10, 20 and 40 (low) over -1 (high) at
    // its vertices, and the low halves unpacked as the translated vertex stage writes them.
    private const string ManualInterpolationVertexSource = """
        #include <metal_stdlib>
        using namespace metal;
        struct VsOut
        {
            float4 position [[position]];
            float4 param0 [[user(locn0)]];
            float4 interp0_1 [[user(locn0_i1)]];
        };
        vertex VsOut manual_vs(uint id [[vertex_id]])
        {
            const float2 corners[3] = { float2(-1.0, -1.0), float2(-1.0, 3.0), float2(3.0, -1.0) };
            const float values[3] = { 10.0, 20.0, 40.0 };
            VsOut out;
            out.position = float4(corners[id], 0.5, 1.0);
            out.param0 = float4(as_type<float>(half2(half(values[id]), half(-1.0))), 0.0, 0.0, 1.0);
            out.interp0_1 = float4(
                (float)as_type<half2>(as_type<uint>(out.param0[0]))[0], (float)as_type<half2>(as_type<uint>(out.param0[1]))[0],
                (float)as_type<half2>(as_type<uint>(out.param0[2]))[0], (float)as_type<half2>(as_type<uint>(out.param0[3]))[0]);
            return out;
        }
        """;

    [Fact]
    public void ManualInterpolationRendersTheGuestsInterpolation()
    {
        if (SkipWithoutMetalDevice("the manual interpolation render test")) return;

        var texels = Render(ManualInterpolationVertexSource, "manual_vs", 3, ManualHalfInterpolation, PerspCenter,
            Gen5PixelOutputKind.Float, languageVersion: null, cntl: [0x400], customMask: 1u, manualInterpolation: true);

        // P0 + I (P1 - P0) + J (P2 - P0) of the low halves, as the guest computes it.
        var vertices = new (double X, double Y)[] { (0, TargetHeight * 1.0), (0, -TargetHeight * 1.0), (TargetWidth * 2.0, TargetHeight * 1.0) };
        double[] values = [10, 20, 40];
        foreach (var (x, y) in new (uint, uint)[] { (1, 1), (7, 3), (12, 6) })
        {
            var actual = BitConverter.UInt32BitsToSingle(texels[(((y * TargetWidth) + x) * 4) + 0]);
            var expected = Interpolate(vertices, values, x + 0.5, y + 0.5);
            Assert.True(Math.Abs(actual - expected) < 1e-3, $"({x},{y}): {actual} != {expected}");
        }
    }

    [Fact]
    public void ProvokingVertexMoveFromPrimitive_StaysAFlatInput()
    {
        var shader = Compile([Interp(4, 2, 1, 0, 2), .. ExportAndEnd(4, 4, 4, 4)], PerspCenter,
            cntl: [0, 1], supportsPerVertex: false, fromPrimitive: true);
        Assert.Contains("float4 attr1 [[user(locn1), flat]];", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("attr1_v0", shader.Source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(OpP1llF16)]
    [InlineData(OpP2F16)]
    public void HalfInterpolation_NeedsPerVertexInputs(uint opcode)
    {
        var request = Request([.. Vop3Interp(opcode, 4, 1, 0, V0, 260), .. ExportAndEnd(4, 4, 4, 4)],
            PerspCenter, PerspCenter, [0, 1], 0, supportsPerVertex: false);
        Assert.False(Gen5MslTranslator.TryCompileProgram(request, out _, out var error));
        Assert.Contains("needs per-vertex pixel inputs", error, StringComparison.Ordinal);
    }

    [Fact]
    public void NegatedBarycentric_IsInterpolatedFromTheVertices()
    {
        uint[] words = [.. Vop3Interp(OpP1F32, 4, 1, 0, V0, neg: 2), .. Vop3Interp(OpP2F32, 4, 1, 0, V1), .. ExportAndEnd(4, 4, 4, 4)];
        var shader = Compile(words, PerspCenter, cntl: [0, 1]);
        Assert.True(shader.UsesPerVertexInputs);
        Assert.Contains("(-as_type<float>(", shader.Source, StringComparison.Ordinal);

        var request = Request(words, PerspCenter, PerspCenter, [0, 1], 0, supportsPerVertex: false);
        Assert.False(Gen5MslTranslator.TryCompileProgram(request, out _, out var error));
        Assert.Contains("needs per-vertex pixel inputs", error, StringComparison.Ordinal);
    }

    [Theory]
    // SPI_PS_INPUT_ADDR bit, the VGPR of its I, and the Metal location.
    [InlineData(1u << 0, 0u, "sample_perspective")]
    [InlineData(1u << 2, 0u, "centroid_perspective")]
    [InlineData(1u << 4, 0u, "sample_no_perspective")]
    [InlineData(1u << 5, 0u, "center_no_perspective")]
    [InlineData(1u << 6, 0u, "centroid_no_perspective")]
    [InlineData((1u << 1) | (1u << 5), 2u, "center_no_perspective")]
    [InlineData((1u << 1) | (1u << 5), 0u, null)]
    [InlineData((1u << 3) | (1u << 6), 3u, "centroid_no_perspective")]
    public void InterpolatedSlot_TakesTheLocationOfItsBarycentrics(uint inputs, uint vgpr, string? qualifier)
    {
        var shader = Compile([Interp(4, 0, 1, 0, vgpr), Interp(4, 1, 1, 0, vgpr + 1), .. ExportAndEnd(4, 4, 4, 4)],
            inputs, cntl: [0, 1]);
        Assert.Contains(qualifier is null ? "float4 attr1 [[user(locn1)]];" : $"float4 attr1 [[user(locn1), {qualifier}]];",
            shader.Source, StringComparison.Ordinal);
        AssertCompiles(shader);
    }

    [Fact]
    public void SlotsOfOneParameterWithDifferentLocations_ShareTheFirstSlotsInput()
    {
        // Slot 1 interpolates with PERSP_CENTER (v0/v1), slot 2 the same parameter with
        // LINEAR_CENTER (v2/v3). The vertex program writes the location once, so both read
        // slot 1's input, interpolated where slot 1 is.
        var shader = Compile(
            [Interp(4, 0, 1, 0, 0), Interp(4, 1, 1, 0, 1), Interp(5, 0, 2, 0, 2), Interp(5, 1, 2, 0, 3), .. ExportAndEnd(4, 5, 4, 4)],
            (1u << 1) | (1u << 5), cntl: [0, 1, 1]);
        Assert.Contains("float4 attr1 [[user(locn1)]];", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("attr2", shader.Source, StringComparison.Ordinal);
        AssertCompiles(shader);
    }

    [Fact]
    public void LineStipple_ReadsZero()
    {
        // PERSP_CENTER holds v0/v1 and LINE_STIPPLE_TEX v2.
        var shader = Compile(ExportAndEnd(2, 2, 2, 2), PerspCenter | (1u << 7));
        Assert.Contains("v2 = 0u;", shader.Source, StringComparison.Ordinal);
    }

    // One triangle whose vertices sit at w = 1, 2 and 4 (clip xyz scaled with it), with
    // param0.x = 10, 20, 40 and param1.x the two halves (value, 2 * value) of each.
    private const string PerspectiveVertexSource = """
        #include <metal_stdlib>
        using namespace metal;
        struct VsOut
        {
            float4 position [[position]];
            float4 param0 [[user(locn0)]];
            float4 param1 [[user(locn1)]];
        };
        vertex VsOut perspective_vs(uint id [[vertex_id]])
        {
            const float2 corners[3] = { float2(-1.0, -1.0), float2(-1.0, 3.0), float2(3.0, -1.0) };
            const float values[3] = { 10.0, 20.0, 40.0 };
            const float w[3] = { 1.0, 2.0, 4.0 };
            VsOut out;
            out.position = float4(corners[id] * w[id], 0.5 * w[id], w[id]);
            out.param0 = float4(values[id], 0.0, 0.0, 1.0);
            out.param1 = float4(as_type<float>(half2(half(values[id]), half(2.0 * values[id]))), 0.0, 0.0, 1.0);
            return out;
        }
        """;

    private const string PrimitiveCopiesVertexSource = """
        #include <metal_stdlib>
        using namespace metal;
        struct VsOut
        {
            float4 position [[position]];
            float4 p0 [[user(locn0_v0)]];
            float4 p1 [[user(locn0_v1)]];
            float4 p2 [[user(locn0_v2)]];
            float3 vertex_w [[user(sharpemu_vertex_w)]];
        };
        vertex VsOut primitive_copies_vs(uint id [[vertex_id]])
        {
            const float2 corners[3] = { float2(-1.0, -1.0), float2(-1.0, 3.0), float2(3.0, -1.0) };
            const float w[3] = { 1.0, 2.0, 4.0 };
            VsOut out;
            out.position = float4(corners[id] * w[id], 0.5 * w[id], w[id]);
            out.p0 = float4(10.0);
            out.p1 = float4(20.0);
            out.p2 = float4(40.0);
            out.vertex_w = float3(1.0, 2.0, 4.0);
            return out;
        }
        """;

    [Theory]
    [InlineData(0x1u, false)]
    [InlineData(0x2u, false)]
    [InlineData(0x4u, false)]
    [InlineData(0x10u, true)]
    [InlineData(0x20u, true)]
    [InlineData(0x40u, true)]
    public void PrimitiveBarycentrics_RenderTheCenterFallback(uint inputs, bool linear)
    {
        if (SkipWithoutMetalDevice("the per-primitive barycentric render test")) return;

        // Request vertex differences to force built-in barycentrics rather than an
        // ordinary interpolated user input. Export I/J and P10/P20 for readback.
        uint[] words = [Interp(4, 2, 0, 0, 0), Interp(5, 2, 0, 0, 1), .. ExportAndEnd(0, 1, 4, 5)];
        var texels = Render(PrimitiveCopiesVertexSource, "primitive_copies_vs", 3, words, inputs,
            Gen5PixelOutputKind.Float, languageVersion: null, cntl: [0], fromPrimitive: true);
        var vertices = new (double X, double Y)[]
        {
            (0, TargetHeight), (0, -TargetHeight * 1.0), (TargetWidth * 2.0, TargetHeight),
        };
        double[] inverseW = [1, 0.5, 0.25];
        foreach (var (x, y) in new (uint, uint)[] { (1, 1), (7, 3), (12, 6) })
        {
            var offset = ((y * TargetWidth) + x) * 4;
            for (var component = 0; component < 2; component++)
            {
                var values = new double[3];
                values[component + 1] = linear ? 1 : inverseW[component + 1];
                var expected = Interpolate(vertices, values, x + 0.5, y + 0.5);
                if (!linear) expected /= Interpolate(vertices, inverseW, x + 0.5, y + 0.5);
                var actual = BitConverter.UInt32BitsToSingle(texels[offset + component]);
                Assert.True(Math.Abs(actual - expected) < 1e-3, $"({x},{y}) component {component}: {actual} != {expected}");
            }
            Assert.Equal(10f, BitConverter.UInt32BitsToSingle(texels[offset + 2]));
            Assert.Equal(30f, BitConverter.UInt32BitsToSingle(texels[offset + 3]));
        }
    }

    [Theory]
    [InlineData(0x11u)]
    [InlineData(0x22u)]
    [InlineData(0x44u)]
    public void MixedPrimitiveBarycentrics_ReconstructPerspectiveWithClipW(uint inputs)
    {
        if (SkipWithoutMetalDevice("the mixed per-primitive barycentric render test")) return;

        uint[] words = [Interp(20, 2, 0, 0, 0), .. ExportAndEnd(0, 1, 2, 3)];
        var texels = Render(PrimitiveCopiesVertexSource, "primitive_copies_vs", 3, words, inputs,
            Gen5PixelOutputKind.Float, languageVersion: null, cntl: [0], fromPrimitive: true);
        var vertices = new (double X, double Y)[]
        {
            (0, TargetHeight), (0, -TargetHeight * 1.0), (TargetWidth * 2.0, TargetHeight),
        };
        double[] inverseW = [1, 0.5, 0.25];
        foreach (var (x, y) in new (uint, uint)[] { (1, 1), (7, 3), (12, 6) })
        {
            var offset = ((y * TargetWidth) + x) * 4;
            for (var component = 0; component < 4; component++)
            {
                var linear = component >= 2;
                var vertex = (component % 2) + 1;
                var values = new double[3];
                values[vertex] = linear ? 1 : inverseW[vertex];
                var expected = Interpolate(vertices, values, x + 0.5, y + 0.5);
                if (!linear) expected /= Interpolate(vertices, inverseW, x + 0.5, y + 0.5);
                var actual = BitConverter.UInt32BitsToSingle(texels[offset + component]);
                Assert.True(Math.Abs(actual - expected) < 1e-3, $"({x},{y}) component {component}: {actual} != {expected}");
            }
        }
    }

    [Theory]
    [InlineData(1u << 1, false)]
    [InlineData(1u << 5, true)]
    public void LinearBarycentricsInterpolateInScreenSpace(uint inputs, bool linear)
    {
        if (SkipWithoutMetalDevice("the interpolation location render test")) return;

        // VINTRP P1 then the VOP3 P2 with div:2 (omod 3) over v0/v1.
        uint[] words = [Interp(4, 0, 0, 0, 0), .. Vop3Interp(OpP2F32, 4, 0, 0, V1, omod: 3), .. ExportAndEnd(4, 4, 4, 4)];
        var texels = Render(PerspectiveVertexSource, "perspective_vs", 3, words, inputs, Gen5PixelOutputKind.Float,
            languageVersion: null);

        var vertices = new (double X, double Y)[] { (0, TargetHeight * 1.0), (0, -TargetHeight * 1.0), (TargetWidth * 2.0, TargetHeight * 1.0) };
        double[] values = [10, 20, 40];
        double[] inverseW = [1, 0.5, 0.25];
        foreach (var (x, y) in new (uint, uint)[] { (1, 1), (7, 3), (12, 6) })
        {
            var value = BitConverter.UInt32BitsToSingle(texels[(((y * TargetWidth) + x) * 4) + 0]);
            var screen = Interpolate(vertices, values, x + 0.5, y + 0.5);
            var perspective = Interpolate(vertices, values.Select((v, i) => v * inverseW[i]).ToArray(), x + 0.5, y + 0.5) /
                Interpolate(vertices, inverseW, x + 0.5, y + 0.5);
            var expected = (linear ? screen : perspective) * 0.5;
            Assert.True(Math.Abs(value - expected) < 1e-3, $"({x},{y}): {value} != {expected} (linear={linear})");
        }
    }

    [Fact]
    public void HalfInterpolationRendersEachHalf()
    {
        if (SkipWithoutMetalDevice("the f16 interpolation render test")) return;
        if (!MetalNative.SupportsPerVertexPixelInputs)
        {
            output.WriteLine("Per-vertex pixel inputs need MSL 4.0 on an Apple10 GPU; the f16 render test skipped.");
            return;
        }

        // v4 = P1LL(low half) then v4.lo = P2_F16 over it; v5 = P1LV(high half, S2 = P0 from
        // a V_INTERP_MOV) then v5.lo = P2_F16(high).
        uint[] words =
        [
            .. Vop3Interp(OpP1llF16, 4, 1, 0, V0),
            .. Vop3Interp(OpP2F16, 4, 1, 0, V1, 260),
            .. Vop3Interp(OpMovF32, 6, 1, 0, 2),
            .. Vop3Interp(OpP1lvF16, 5, 1, 0, V0, 262, high: true),
            .. Vop3Interp(OpP2F16, 5, 1, 0, V1, 261, high: true),
            .. ExportAndEnd(4, 5, 4, 4),
        ];
        var texels = Render(PerVertexHalfSource, "per_vertex_half_vs", 3, words, PerspCenter, Gen5PixelOutputKind.Uint,
            Gen5MslShader.PerVertexInputsLanguageVersion);

        var vertices = new (double X, double Y)[] { (0, TargetHeight * 1.0), (0, -TargetHeight * 1.0), (TargetWidth * 2.0, TargetHeight * 1.0) };
        double[] values = [10, 20, 40];
        foreach (var (x, y) in new (uint, uint)[] { (1, 1), (7, 3), (12, 6) })
        {
            float Half(int component) =>
                (float)BitConverter.UInt16BitsToHalf((ushort)texels[(((y * TargetWidth) + x) * 4) + component]);

            var expected = Interpolate(vertices, values, x + 0.5, y + 0.5);
            Assert.True(Math.Abs(Half(0) - expected) < 0.05, $"({x},{y}): {Half(0)} != {expected}");
            Assert.True(Math.Abs(Half(1) - (2 * expected)) < 0.1, $"({x},{y}): {Half(1)} != {2 * expected}");
        }
    }

    private const string PerVertexHalfSource = """
        #include <metal_stdlib>
        using namespace metal;
        struct VsOut
        {
            float4 position [[position]];
            float4 param0 [[user(locn0)]];
            float4 param1 [[user(locn1)]];
        };
        vertex VsOut per_vertex_half_vs(uint id [[vertex_id]])
        {
            const float2 corners[3] = { float2(-1.0, -1.0), float2(-1.0, 3.0), float2(3.0, -1.0) };
            const float values[3] = { 10.0, 20.0, 40.0 };
            VsOut out;
            out.position = float4(corners[id], 0.5, 1.0);
            out.param0 = float4(values[id], 0.0, 0.0, 1.0);
            out.param1 = float4(as_type<float>(half2(half(values[id]), half(2.0 * values[id]))), 0.0, 0.0, 1.0);
            return out;
        }
        """;

    // ---- helpers ----

    private void AssertCompiles(Gen5MslShader shader)
    {
        if (SkipWithoutMetalDevice("the pixel input compile test") ||
            (shader.UsesPerVertexInputs && !OperatingSystem.IsMacOSVersionAtLeast(26)))
        {
            return;
        }

        var compiled = shader.UsesPerVertexInputs
            ? MetalNative.TryCompileLibrary(shader.Source, Gen5MslShader.PerVertexInputsLanguageVersion, out _, out var error)
            : MetalNative.TryCompileLibrary(shader.Source, out _, out error);
        Assert.True(compiled, $"Metal rejected the emitted pixel MSL: {error}\n{shader.Source}");
    }

    private static Gen5MslShader Compile(
        uint[] words,
        uint inputs,
        uint? enable = null,
        uint[]? cntl = null,
        uint customMask = 0,
        bool supportsPerVertex = true,
        Gen5PixelOutputKind outputKind = Gen5PixelOutputKind.Float,
        bool frontFaceAllBits = false,
        bool ieeeMode = false,
        bool fromPrimitive = false,
        bool manualInterpolation = false)
    {
        var request = Request(words, inputs, enable ?? inputs, cntl ?? [], customMask, supportsPerVertex, outputKind,
            frontFaceAllBits, ieeeMode, fromPrimitive, manualInterpolation);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        return shader;
    }

    private static ShaderCompileRequest Request(
        uint[] words,
        uint address,
        uint enable,
        uint[] cntl,
        uint customMask,
        bool supportsPerVertex,
        Gen5PixelOutputKind outputKind = Gen5PixelOutputKind.Float,
        bool frontFaceAllBits = false,
        bool ieeeMode = false,
        bool fromPrimitive = false,
        bool manualInterpolation = false)
    {
        var program = Gen5ComputeFixtures.DecodeOrThrow(new Gen5ComputeFixture("pixel-inputs", words, 0, 0));
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Pixel, 0x5049_5850_494E_5055, 0, 16);
        var resources = ResourceMaterializer.ApplyTo(plan, ResourceSpecialization.Default(plan.Info));
        var layout = BindingLayout.Allocate(
            resources.Info,
            BindingLayout.CollectUserDataRegisters(program, 0, 16),
            BindingLayout.UsesGlobalDataShare(program),
            ShaderCompileRequest.RequiresFlattenedTable(plan, resources),
            BindingLayout.ReadsShaderBase(program),
            0);
        return new ShaderCompileRequest(plan, resources, layout)
        {
            PixelOutputs = [new Gen5PixelOutputBinding(0, 0, outputKind)],
            PixelInputAddress = address,
            PixelInputEnable = enable,
            PixelInputCntl = cntl,
            PixelCustomInterpolationMask = customMask,
            SupportsPerVertexPixelInputs = supportsPerVertex,
            PixelPerVertexInputsFromPrimitive = fromPrimitive,
            PixelManualInterpolation = manualInterpolation,
            PixelFrontFaceAllBits = frontFaceAllBits,
            IeeeMode = ieeeMode,
        };
    }

    private bool SkipWithoutMetalDevice(string activity)
    {
        if (MetalNative.IsAvailable) return false;
        Assert.False(
            OperatingSystem.IsMacOS() && Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") == "1",
            $"The required gate cannot run {activity} without a Metal device.");
        output.WriteLine($"No Metal device on this host; {activity} skipped.");
        return true;
    }
}
