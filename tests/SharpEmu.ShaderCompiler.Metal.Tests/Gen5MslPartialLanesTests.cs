// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>
/// ShaderCompileRequest.GraphicsPartialSubgroupLanes (SHARPEMU_GRAPHICS_PARTIAL_LANES=1) in MSL, as the SPIR-V
/// translator lowers it: a pixel lane read (DPP, DPP8, PERMLANE16, V_READLANE) of a lane the fragment simdgroup
/// never launched reads the reading lane's own value, and such a lane counts as active when the guest EXEC mask
/// has its bit (whole-wave code turned it on). Off, the translation is the one it always was.
/// </summary>
public sealed class Gen5MslPartialLanesTests(ITestOutputHelper output)
{
    private const uint TargetSize = 8;
    private const uint Untouched = 0xEE;

    // Exported: r = lane | row_shl:4 after EXEC is restored << 8 | readlane 31 << 16
    //           g = row_shl:4 with EXEC -1 (bound_ctrl clear)
    //           b = dpp8:[7,7,7,7,7,7,7,7] with EXEC -1
    //           a = entry EXEC
    private static readonly uint[] PartialLaneWords =
    [
        0xD7650000, 0x000100C1, // v_mbcnt_lo_u32_b32 v0, -1, 0             lane
        0xD5430001, 0x021D0700, // v_mad_u32_u24 v1, v0, 3, 7                value
        0xBE84037E,             // s_mov_b32 s4, exec_lo
        0x7E0602FF, 0x000000EE, // v_mov_b32 v3, 0xee
        0x7E0802FF, 0x000000EE, // v_mov_b32 v4, 0xee
        0xBEFE03C1,             // s_mov_b32 exec_lo, -1                    lanes that may not exist
        0x7E0602FA, 0xFF010401, // v_mov_b32_dpp v3, v1 row_shl:4
        0x7E0A02E9, 0xFFFFFF01, // v_mov_b32_dpp v5, v1 dpp8:[7,7,7,7,7,7,7,7]
        0xBEFE0304,             // s_mov_b32 exec_lo, s4
        0x7E0802FA, 0xFF010401, // v_mov_b32_dpp v4, v1 row_shl:4
        0xD7600000, 0x00013F01, // v_readlane_b32 s0, v1, 31
        0x7E0C0200,             // v_mov_b32 v6, s0
        0xD76F000B, 0x04011104, // v_lshl_or_b32 v11, v4, 8, v0
        0xD76F000B, 0x042D2106, // v_lshl_or_b32 v11, v6, 16, v11
        0x7E180204,             // v_mov_b32 v12, s4
        0xF800180F, 0x0C05030B, // exp mrt0 v11, v3, v5, v12 done vm
        0xBF810000,             // s_endpgm
    ];

    private const string VertexSource = """

        struct SharpEmuPartialLaneVertex
        {
            float4 position [[position]];
        };

        vertex SharpEmuPartialLaneVertex sharpemu_partial_lane_vs(
            uint id [[vertex_id]],
            constant float2* positions [[buffer(0)]])
        {
            SharpEmuPartialLaneVertex vertex_out;
            vertex_out.position = float4(positions[id], 0.0f, 1.0f);
            return vertex_out;
        }
        """;

    [Fact]
    public void OffTheTranslationIsUnchanged()
    {
        var unset = Compile(Words(PartialLaneWords), ShaderStage.Pixel, null).Source;
        Assert.Equal(unset, Compile(Words(PartialLaneWords), ShaderStage.Pixel, false).Source);
        Assert.DoesNotContain("sharpemu_guest_exec", unset, StringComparison.Ordinal);
        // The prelude defines the nearest-present-lane helper; off, V_READLANE calls it.
        Assert.Equal(2, Occurrences(unset, "sharpemu_present_lane("));

        var on = Compile(Words(PartialLaneWords), ShaderStage.Pixel, true).Source;
        Assert.NotEqual(unset, on);
        Assert.Contains("ulong sharpemu_guest_exec = 0ul;", on, StringComparison.Ordinal);
        Assert.Equal(1, Occurrences(on, "sharpemu_present_lane("));
    }

    // The vertex stage runs one lane per thread and a compute simdgroup is full: no lane is absent.
    [Fact]
    public void OnlyThePixelStageReadsAbsentLanes()
    {
        var compute = Words(PartialLaneWords[..^4].Concat([0xBF810000u]).ToArray());
        Assert.Equal(
            Compile(compute, ShaderStage.Compute, false).Source,
            Compile(compute, ShaderStage.Compute, true).Source);
    }

    [Fact]
    public void ALaneTheSimdgroupNeverLaunchedReadsAsTheReader()
    {
        if (SkipWithoutMetalDevice("the partial-lane execution test")) return;

        // Covers only the center of pixel (1,1): its quad runs, the rest of the simdgroup does not.
        var triangle = new[] { -0.8f, 0.8f, -0.4f, 0.8f, -0.8f, 0.4f };
        var on = Render(true, triangle);
        var off = Render(false, triangle);

        var (r, g, b, running) = Texel(on, 1, 1);
        var lane = r & 31u;
        output.WriteLine($"pixel (1,1) ran on lane {lane}, wave lanes 0x{running:X8}");
        Assert.Equal(3u, lane & 3u);
        Assert.Equal(Texel(off, 1, 1).A, running);
        bool Runs(uint other) => ((running >> (int)other) & 1u) != 0;

        // row_shl:4 under EXEC -1: a source in the row that never ran is on in the guest mask.
        var inRow = (lane & 15u) < 12u;
        var target = lane + 4u;
        var expectedShift = !inRow ? Untouched : Runs(target) ? Value(target) : Value(lane);
        Assert.Equal(expectedShift, g);
        // Off: the source is not a running lane, so it is EXEC-clear and the lane keeps v3.
        Assert.Equal(!inRow ? Untouched : Runs(target) ? Value(target) : Untouched, Texel(off, 1, 1).G);

        // dpp8 lane 7 of the group of eight under EXEC -1; off, an absent source reads zero.
        var group = (lane & ~7u) + 7u;
        Assert.Equal(Runs(group) ? Value(group) : Value(lane), b);
        Assert.Equal(Runs(group) ? Value(group) : 0u, Texel(off, 1, 1).B);

        // EXEC restored: the absent source's guest bit is clear, so v4 is kept in both translations.
        var expectedRestored = !inRow ? Untouched : Runs(target) ? Value(target) : Untouched;
        Assert.Equal(expectedRestored, (r >> 8) & 0xFFu);
        Assert.Equal(expectedRestored, (Texel(off, 1, 1).R >> 8) & 0xFFu);

        // V_READLANE of lane 31: an absent lane reads as the reader.
        Assert.Equal(Runs(31) ? Value(31) : Value(lane), (r >> 16) & 0xFFu);
    }

    // V_PERMLANE16 with and without FI (OP_SEL[0]) takes the same lane reads; Metal accepts both translations.
    [Fact]
    public void Permlane16ReadsThroughTheGuestMask()
    {
        uint[] words =
        [
            0xD7650000, 0x000100C1, // v_mbcnt_lo_u32_b32 v0, -1, 0
            0xD5430001, 0x021D0700, // v_mad_u32_u24 v1, v0, 3, 7
            0xBE8003FF, 0x76543210, // s_mov_b32 s0, 0x76543210
            0xBE8103FF, 0xFEDCBA98, // s_mov_b32 s1, 0xfedcba98
            0x7E0602FF, 0x000000EE, // v_mov_b32 v3, 0xee
            0x7E0802FF, 0x000000EE, // v_mov_b32 v4, 0xee
            0xD7770003, 0x00040101, // v_permlane16_b32 v3, v1, s0, s1
            0xD7770804, 0x00040101, // v_permlane16_b32 v4, v1, s0, s1 op_sel:[1,0]   FI
            0xF800180F, 0x04030100, // exp mrt0 v0, v1, v3, v4 done vm
            0xBF810000,             // s_endpgm
        ];
        var off = Compile(Words(words), ShaderStage.Pixel, false).Source;
        var on = Compile(Words(words), ShaderStage.Pixel, true).Source;
        Assert.NotEqual(off, on);
        Assert.Contains("sharpemu_guest_exec >> ", on, StringComparison.Ordinal);
        if (SkipWithoutMetalDevice("the permlane16 compile")) return;

        Assert.True(MetalNative.TryCompileLibrary(on + VertexSource, out _, out var error), $"{error}\n{on}");
        Assert.True(MetalNative.TryCompileLibrary(off + VertexSource, out _, out error), error);
    }

    private static uint Value(uint lane) => (3 * lane) + 7;

    private static int Occurrences(string source, string text) =>
        (source.Length - source.Replace(text, string.Empty, StringComparison.Ordinal).Length) / text.Length;

    private static Gen5ShaderProgram Words(uint[] words) =>
        Gen5ComputeFixtures.DecodeOrThrow(new Gen5ComputeFixture("pixel-partial-lanes", words, 0, 0));

    private static Gen5MslShader Compile(Gen5ShaderProgram program, ShaderStage stage, bool? partialLanes)
    {
        Gen5PixelOutputBinding[] outputs = stage == ShaderStage.Pixel
            ? [new Gen5PixelOutputBinding(0, 0, Gen5PixelOutputKind.Uint)]
            : [];
        var request = Gen5ComputeFixtures.RequestOrThrow(program, stage,
            pixelOutputs: outputs,
            enableGraphicsSubgroupOperations: true,
            graphicsPartialSubgroupLanes: partialLanes);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        return shader;
    }

    private byte[] Render(bool partialLanes, float[] triangle)
    {
        var shader = Compile(Words(PartialLaneWords), ShaderStage.Pixel, partialLanes);
        Assert.True(
            MetalNative.TryCompileLibrary(shader.Source + VertexSource, out var library, out var error),
            $"Metal rejected the partial-lane MSL: {error}\n{shader.Source}");

        var positions = new byte[triangle.Length * sizeof(float)];
        Buffer.BlockCopy(triangle, 0, positions, 0, positions.Length);
        var resources = new byte[Math.Max(shader.ArgumentLayout?.ByteSize ?? 0, 256)];
        Assert.True(
            MetalNative.TryRenderTriangles(
                library,
                "sharpemu_partial_lane_vs",
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
