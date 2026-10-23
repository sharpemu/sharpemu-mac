// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Numerics;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>
/// A pixel shader's guest wave32 runs on the fragment's 32-wide SIMD-group, so cross-lane
/// operations see the other fragments' values instead of the lane's own. The device tests
/// draw through a real render pipeline and read back what a guest program computed with
/// DPP, DS_SWIZZLE, a ballot, readlane/readfirstlane and an EXEC waterfall loop. Metal
/// packs quads into simdgroups as it likes, and not always 32 lanes' worth, so each
/// expectation comes from what the fragment itself reports: its lane (V_MBCNT) and the
/// wave's entry EXEC, the lanes that run.
/// </summary>
public sealed class Gen5MslPixelWaveTests(ITestOutputHelper output)
{
    private const uint TargetSize = 8;
    private const uint Untouched = 0xEE;

    // Each lane's value is 3 * lane + 7, under 128 for every lane of a wave32.
    // The DS_SWIZZLE words name v1 in both ADDR and DATA0, so the test does not depend on
    // which of the two fields the decoder takes the swizzled value from.
    //
    // Exported: r = lane | quad_perm[1,0,3,2] << 5 | swizzle quad[2,3,0,1] << 12 |
    //               swizzle lane^16 << 19 | waterfall iteration << 26 | iterations << 28
    //           g = ballot of (value & 4)
    //           b = row_shr:1 | row_shr:1 bound_ctrl << 8 | readlane 5 << 16 | readfirstlane << 24
    //           a = entry EXEC
    private static readonly uint[] CrossLaneWords =
    [
        0xD7650000, 0x000100C1, // v_mbcnt_lo_u32_b32 v0, -1, 0             lane
        0xD5430001, 0x021D0700, // v_mad_u32_u24 v1, v0, 3, 7                value
        0x7E0602FF, 0x000000EE, // v_mov_b32 v3, 0xee
        0x7E0602FA, 0xFF011101, // v_mov_b32_dpp v3, v1 row_shr:1
        0x7E0802FF, 0x000000EE, // v_mov_b32 v4, 0xee
        0x7E0802FA, 0xFF091101, // v_mov_b32_dpp v4, v1 row_shr:1 bound_ctrl:1
        0x7E0A02FA, 0xFF00B101, // v_mov_b32_dpp v5, v1 quad_perm:[1,0,3,2]
        0xD8D4804E, 0x06000101, // ds_swizzle_b32 v6, v1 offset:swizzle(QUAD_PERM,2,3,0,1)
        0xD8D4401F, 0x07000101, // ds_swizzle_b32 v7, v1 offset:swizzle(SWAP,16)
        0xBF8CC07F,             // s_waitcnt lgkmcnt(0)
        0x36100284,             // v_and_b32 v8, 4, v1
        0x7D8A1080,             // v_cmp_ne_u32 vcc_lo, 0, v8
        0xBE88036A,             // s_mov_b32 s8, vcc_lo                     ballot
        0xD7600000, 0x00010B01, // v_readlane_b32 s0, v1, 5
        0x7E020501,             // v_readfirstlane_b32 s1, v1
        0x2C120083,             // v_lshrrev_b32 v9, 3, v0                  key = lane >> 3
        0xBE84037E,             // s_mov_b32 s4, exec_lo
        0xBEFE03C1,             // s_mov_b32 exec_lo, -1                    lanes that may not exist
        0xBE870380,             // s_mov_b32 s7, 0
        0x7E1402FF, 0x000000EE, // v_mov_b32 v10, 0xee
        // loop: one iteration per distinct key, in first-lane order
        0x7E0A0509,             // v_readfirstlane_b32 s5, v9
        0x7D841205,             // v_cmp_eq_u32 vcc_lo, s5, v9
        0xBE863C6A,             // s_and_saveexec_b32 s6, vcc_lo
        0x7E140207,             // v_mov_b32 v10, s7
        0x897E067E,             // s_xor_b32 exec_lo, exec_lo, s6
        0x80078107,             // s_add_u32 s7, s7, 1
        0xBF89FFF9,             // s_cbranch_execnz loop
        0xBEFE0304,             // s_mov_b32 exec_lo, s4
        0xD76F000B, 0x04010B05, // v_lshl_or_b32 v11, v5, 5, v0
        0xD76F000B, 0x042D1906, // v_lshl_or_b32 v11, v6, 12, v11
        0xD76F000B, 0x042D2707, // v_lshl_or_b32 v11, v7, 19, v11
        0xD76F000B, 0x042D350A, // v_lshl_or_b32 v11, v10, 26, v11
        0xD76F000B, 0x042D3807, // v_lshl_or_b32 v11, s7, 28, v11
        0x7E180208,             // v_mov_b32 v12, s8
        0xD76F000D, 0x040D1104, // v_lshl_or_b32 v13, v4, 8, v3
        0xD76F000D, 0x04352000, // v_lshl_or_b32 v13, s0, 16, v13
        0xD76F000D, 0x04353001, // v_lshl_or_b32 v13, s1, 24, v13
        0x7E1C0204,             // v_mov_b32 v14, s4
        0xF800180F, 0x0E0D0C0B, // exp mrt0 v11, v12, v13, v14 done vm
        0xBF810000,             // s_endpgm
    ];

    private const string VertexSource = """

        struct SharpEmuWaveTestVertex
        {
            float4 position [[position]];
        };

        vertex SharpEmuWaveTestVertex sharpemu_wave_test_vs(
            uint id [[vertex_id]],
            constant float2* positions [[buffer(0)]])
        {
            SharpEmuWaveTestVertex vertex_out;
            vertex_out.position = float4(positions[id], 0.0f, 1.0f);
            return vertex_out;
        }
        """;

    [Fact]
    public void PixelStageAddressesRealSimdgroupLanes()
    {
        var source = Compile(enableGraphicsSubgroupOperations: true, structured: true).Source;

        Assert.Contains("uint sharpemu_lane [[thread_index_in_simdgroup]]", source, StringComparison.Ordinal);
        Assert.Contains("(uint)(uint64_t)simd_ballot(value)", source, StringComparison.Ordinal);
        Assert.Contains("simd_shuffle(", source, StringComparison.Ordinal);
        Assert.DoesNotContain("const uint sharpemu_lane = 0u;", source, StringComparison.Ordinal);

        // An EXEC write keeps only lanes that are running, so S_MOV EXEC, -1 in a part-filled
        // simdgroup leaves no bit that the waterfall loop could never clear.
        Assert.Contains("& sharpemu_ballot(true)", source, StringComparison.Ordinal);
    }

    [Fact]
    public void PixelStageWithoutGraphicsSubgroupsKeepsOneLane()
    {
        // Not the cross-lane program: DS_SWIZZLE has no single-lane form and fails to compile.
        var program = Gen5ComputeFixtures.DecodeOrThrow(
            new Gen5ComputeFixture("pixel", Gen5ComputeFixtures.PixelWords, 0, 0));
        var request = Gen5ComputeFixtures.RequestOrThrow(program, ShaderStage.Pixel,
            enableGraphicsSubgroupOperations: false);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);

        Assert.Contains("const uint sharpemu_lane = 0u;", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("thread_index_in_simdgroup", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("simd_ballot", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void VertexStageStaysSingleLane()
    {
        // A vertex function has no documented SIMD-group functions and no lane index.
        var source = Gen5ComputeFixtures.CompileVertexOrThrow().Source;

        Assert.Contains("const uint sharpemu_lane = 0u;", source, StringComparison.Ordinal);
        Assert.DoesNotContain("thread_index_in_simdgroup", source, StringComparison.Ordinal);
        Assert.DoesNotContain("simd_ballot", source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void CoveredFragmentsExchangeLaneValues(bool structured)
    {
        if (SkipWithoutMetalDevice("the pixel wave execution test")) return;

        // One triangle over the whole target: no helpers, every fragment a live lane.
        var texels = RenderOrThrow(structured, [-1f, -1f, 3f, -1f, -1f, 3f]);

        for (uint y = 0; y < TargetSize; y++)
        {
            for (uint x = 0; x < TargetSize; x++)
            {
                var lane = Texel(texels, x, y).R & 31u;

                // The guest quad layout: top-left, top-right, bottom-left, bottom-right.
                Assert.Equal((x & 1u) | ((y & 1u) << 1), lane & 3u);
                AssertGuestSemantics(texels, x, y, $"({x},{y})");
            }
        }
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ASinglePixelSeesItsHelperQuadAndNothingElse(bool structured)
    {
        if (SkipWithoutMetalDevice("the pixel wave execution test")) return;

        // Covers only the center of pixel (1,1), the bottom-right of its quad; the other
        // three run as helpers and the rest of the simdgroup does not run at all.
        var texels = RenderOrThrow(structured, [-0.8f, 0.8f, -0.4f, 0.8f, -0.8f, 0.4f]);

        for (uint y = 0; y < TargetSize; y++)
        {
            for (uint x = 0; x < TargetSize; x++)
            {
                if (x == 1 && y == 1)
                {
                    continue;
                }

                Assert.Equal((0u, 0u, 0u, 0u), Texel(texels, x, y));
            }
        }

        var (r, _, _, running) = Texel(texels, 1, 1);
        var lane = r & 31u;
        output.WriteLine($"pixel (1,1) ran on lane {lane}, wave lanes 0x{running:X8}");
        Assert.Equal(3u, lane & 3u);

        // Helper lanes are part of the wave (EXEC includes them, as on the SPIR-V path), so
        // quad exchanges read their values; lane ^ 16 does not run and reads as inactive.
        Assert.Equal(0xFu << (int)(lane & ~3u), running);

        // EXEC was forced to all ones before the waterfall loop, yet it ends after the one
        // key the quad has instead of waiting on lanes that do not exist.
        AssertGuestSemantics(texels, 1, 1, "(1,1)");
        Assert.Equal(1u, r >> 28);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void LazyFragmentMasksMatchScalarMasksWithHelpersAndDiscard(bool partialQuad, bool discard)
    {
        uint[] words = [
            0xD7650000, 0x000100C1, // lane -> v0
            0xBE8E047E,             // save entry EXEC s14:15
            0xBEFE0A7E,             // WQM
            0x7E040287, 0x7E060287, 0x7E080287, // v2,v3,v4=7
            0x36020081,             // v1=lane & 1
            0x7D8A0280,             // compare v1 != 0 -> VCC_LO
            0xBE843C6A,             // s_and_saveexec_b32 s4,vcc_lo
            0x7E04028B, 0x7E06028D, 0x7E08028F, // masked local ALU run
            0xBEFE040E,             // restore entry EXEC
            .. (discard ? new uint[] { 0xBEFE036A } : []),
            0xF800180F, 0x00040302, // export v2,v3,v4,lane with VM
            0xBF810000 ];
        var optimized = Compile(true, true, true, words).Source;
        var baseline = Compile(true, true, false, words).Source;
        Assert.DoesNotContain("SHARPEMU_BALLOT", optimized, StringComparison.Ordinal);
        Assert.True(optimized.Count(c => c == '\n') < baseline.Count(c => c == '\n'));
        if (SkipWithoutMetalDevice("fragment mask differential execution")) return;
        var triangle = partialQuad
            ? new[] { -0.8f, 0.8f, -0.4f, 0.8f, -0.8f, 0.4f }
            : new[] { -1f, -1f, 3f, -1f, -1f, 3f };
        var expected = RenderOrThrow(false, triangle, false, words);
        var actual = RenderOrThrow(true, triangle, true, words);
        Assert.Equal(expected, actual);
        for (uint y = 0; y < TargetSize; y++)
            for (uint x = 0; x < TargetSize; x++)
            {
                var (r, g, b, lane) = Texel(actual, x, y);
                if (partialQuad && (x != 1 || y != 1)) continue;
                if (discard && r == 0) continue;
                var active = (lane & 1u) != 0;
                Assert.Equal(active ? 11u : 7u, r);
                Assert.Equal(active ? 13u : 7u, g);
                Assert.Equal(active ? 15u : 7u, b);
            }
    }

    /// <summary>Checks one fragment's exports against the guest semantics of
    /// <see cref="CrossLaneWords"/> over the lanes its wave reported running.</summary>
    private static void AssertGuestSemantics(byte[] texels, uint x, uint y, string where)
    {
        var (r, g, b, running) = Texel(texels, x, y);
        var lane = r & 31u;
        Assert.True(((running >> (int)lane) & 1u) != 0, $"{where}: lane {lane} is not in 0x{running:X8}");

        uint ValueIfRunning(uint other) => ((running >> (int)other) & 1u) != 0 ? Value(other) : 0u;

        // Keys are lane >> 3, visited lowest first.
        var keys = Enumerable.Range(0, 4).Where(key => ((running >> (key * 8)) & 0xFFu) != 0).ToArray();
        var expectedR =
            lane |
            (ValueIfRunning(lane ^ 1) << 5) |
            (ValueIfRunning(lane ^ 2) << 12) |
            (ValueIfRunning(lane ^ 16) << 19) |
            ((uint)Array.IndexOf(keys, (int)(lane >> 3)) << 26) |
            ((uint)keys.Length << 28);
        Assert.True(expectedR == r, $"{where}: lane exchange 0x{r:X8}, expected 0x{expectedR:X8}");

        var ballot = 0u;
        for (var other = 0; other < 32; other++)
        {
            if (((running >> other) & 1u) != 0 && (Value((uint)other) & 4u) != 0)
            {
                ballot |= 1u << other;
            }
        }

        Assert.True(ballot == g, $"{where}: ballot 0x{g:X8}, expected 0x{ballot:X8}");

        // row_shr:1 leaves a row's first lane alone (v3 keeps 0xEE) or zeroes it under
        // bound_ctrl (v4); elsewhere both read the previous lane, 0 where that lane is off.
        var previous = ValueIfRunning(lane - 1);
        var rowShift = (lane & 15u) == 0 ? Untouched : previous | (previous << 8);
        Assert.True(rowShift == (b & 0xFFFFu), $"{where}: row shift 0x{b & 0xFFFFu:X4}, expected 0x{rowShift:X4}");

        // V_READLANE of a lane that is not running reads whatever it holds; only a
        // running lane 5 has a defined value.
        if (((running >> 5) & 1u) != 0)
        {
            Assert.Equal(Value(5), (b >> 16) & 0xFFu);
        }

        Assert.Equal(Value((uint)BitOperations.TrailingZeroCount(running)), b >> 24);
    }

    private static uint Value(uint lane) => (3 * lane) + 7;

    private byte[] RenderOrThrow(bool structured, float[] triangle, bool fragmentMasks = true, uint[]? words = null)
    {
        var shader = Compile(enableGraphicsSubgroupOperations: true, structured, fragmentMasks, words);
        Assert.True(
            MetalNative.TryCompileLibrary(shader.Source + VertexSource, out var library, out var error),
            $"Metal rejected the pixel wave MSL: {error}\n{shader.Source}");

        var positions = new byte[triangle.Length * sizeof(float)];
        Buffer.BlockCopy(triangle, 0, positions, 0, positions.Length);
        var resources = new byte[Math.Max(shader.ArgumentLayout?.ByteSize ?? 0, 256)];
        Assert.True(
            MetalNative.TryRenderTriangles(
                library,
                "sharpemu_wave_test_vs",
                shader.EntryPoint,
                [(positions, 0)],
                [(resources, (uint)Gen5MslArgumentLayout.ResourcesBufferIndex)],
                3,
                TargetSize,
                TargetSize,
                MetalNative.Rgba32UintPixelFormat,
                out var texels,
                out error),
            error);
        return texels;
    }

    private static Gen5MslShader Compile(bool enableGraphicsSubgroupOperations, bool structured,
        bool fragmentMasks = true, uint[]? words = null)
    {
        var program = Gen5ComputeFixtures.DecodeOrThrow(new Gen5ComputeFixture("pixel-wave", words ?? CrossLaneWords, 0, 0));
        var request = Gen5ComputeFixtures.RequestOrThrow(program, ShaderStage.Pixel,
            pixelOutputs: [new Gen5PixelOutputBinding(0, 0, Gen5PixelOutputKind.Uint)],
            enableGraphicsSubgroupOperations: enableGraphicsSubgroupOperations);
        Assert.True(
            Gen5MslTranslator.TryCompileProgram(
                request, out var shader, out var error, new Gen5MslCodegenOptions(StructuredControlFlow: structured, FragmentLaneMasks: fragmentMasks)),
            error);
        return shader;
    }

    private static (uint R, uint G, uint B, uint A) Texel(byte[] texels, uint x, uint y)
    {
        var offset = (int)(((y * TargetSize) + x) * 16);
        return (
            BitConverter.ToUInt32(texels, offset),
            BitConverter.ToUInt32(texels, offset + 4),
            BitConverter.ToUInt32(texels, offset + 8),
            BitConverter.ToUInt32(texels, offset + 12));
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
