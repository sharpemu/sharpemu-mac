// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>
/// A graphics-stage program whose only LDS accesses are DS_WRITE_ADDTID_B32 and
/// DS_READ_ADDTID_B32 under a constant M0 (the guest compiler's VGPR spills) keeps one
/// dword per spill slot instead of the 8 KB per-invocation array. The device test spills
/// two lane values, clobbers the registers, reads them back and exports them.
/// </summary>
public sealed class Gen5MslLaneSlotLdsTests(ITestOutputHelper output)
{
    private const uint TargetSize = 8;
    private const uint Marker = 0x55;

    private const uint MovkM0Zero = 0xB07C0000;   // s_movk_i32 m0, 0
    private const uint MovM0FromS0 = 0xBEFC0300;  // s_mov_b32 m0, s0

    private static uint DsWord0(uint opcode, uint offset0 = 0, uint offset1 = 0) =>
        0xD8000000u | opcode << 18 | offset1 << 8 | offset0;

    private static uint DsWord1(uint address, uint data0, uint data1, uint destination) =>
        destination << 24 | data1 << 16 | data0 << 8 | address;

    // Exported: r = the spilled 3 * lane + 7, g = the spilled marker, b = a = lane.
    private static uint[] SpillWords(uint m0Write) =>
    [
        m0Write,
        0xD7650000, 0x000100C1,                     // v_mbcnt_lo_u32_b32 v0, -1, 0      lane
        0xD5430001, 0x021D0700,                     // v_mad_u32_u24 v1, v0, 3, 7
        DsWord0(0xB0, 0, 1), DsWord1(0, 1, 0, 0),   // ds_write_addtid_b32 v1 offset:256
        0x7E0402FF, Marker,                         // v_mov_b32 v2, Marker
        DsWord0(0xB0), DsWord1(0, 2, 0, 0),         // ds_write_addtid_b32 v2
        0x7E020280,                                 // v_mov_b32 v1, 0
        0x7E040280,                                 // v_mov_b32 v2, 0
        DsWord0(0xB1, 0, 1), DsWord1(0, 0, 0, 3),   // ds_read_addtid_b32 v3 offset:256
        DsWord0(0xB1), DsWord1(0, 0, 0, 4),         // ds_read_addtid_b32 v4
        0xBF8CC07F,                                 // s_waitcnt lgkmcnt(0)
        0xF800180F, 0x00000403,                     // exp mrt0 v3, v4, v0, v0 done vm
        0xBF810000,                                 // s_endpgm
    ];

    [Fact]
    public void SpillsUnderAConstantM0GetOneSlotEach()
    {
        var source = Compile(SpillWords(MovkM0Zero), structured: true).Source;
        Assert.Contains("    thread uint sharpemu_lds[2];", source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_lds[0u]", source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_lds[1u]", source, StringComparison.Ordinal);
        Assert.DoesNotContain("sharpemu_lane << 2u", source, StringComparison.Ordinal);
    }

    [Fact]
    public void AnM0TheBlockDidNotSetKeepsTheFullArray()
    {
        var source = Compile(SpillWords(MovM0FromS0), structured: true).Source;
        Assert.Contains("    thread uint sharpemu_lds[2048];", source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_lane << 2u", source, StringComparison.Ordinal);
    }

    [Fact]
    public void AnyOtherLdsAccessKeepsTheFullArray()
    {
        var words = SpillWords(MovkM0Zero).ToList();
        // ds_write_b32 v0, v1 before the end: an address the lane does not own.
        words.InsertRange(words.Count - 3, [DsWord0(0x0D), DsWord1(0, 1, 0, 0)]);
        var source = Compile(words.ToArray(), structured: true).Source;
        Assert.Contains("    thread uint sharpemu_lds[2048];", source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void SpilledLaneValuesComeBack(bool structured)
    {
        if (SkipWithoutMetalDevice("the lane-slot LDS execution test")) return;

        var shader = Compile(SpillWords(MovkM0Zero), structured);
        Assert.True(
            MetalNative.TryCompileLibrary(shader.Source + VertexSource, out var library, out var error),
            $"Metal rejected the lane-slot LDS MSL: {error}\n{shader.Source}");

        float[] triangle = [-1f, -1f, 3f, -1f, -1f, 3f];
        var positions = new byte[triangle.Length * sizeof(float)];
        Buffer.BlockCopy(triangle, 0, positions, 0, positions.Length);
        var resources = new byte[Math.Max(shader.ArgumentLayout?.ByteSize ?? 0, 256)];
        Assert.True(
            MetalNative.TryRenderTriangles(
                library,
                "sharpemu_lds_test_vs",
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

        for (var texel = 0; texel < TargetSize * TargetSize; texel++)
        {
            var offset = texel * 16;
            var spilled = BitConverter.ToUInt32(texels, offset);
            var marker = BitConverter.ToUInt32(texels, offset + 4);
            var lane = BitConverter.ToUInt32(texels, offset + 8);
            Assert.True(lane < 32, $"texel {texel}: lane {lane}");
            Assert.Equal((3 * lane) + 7, spilled);
            Assert.Equal(Marker, marker);
        }
    }

    private const string VertexSource = """

        struct SharpEmuLdsTestVertex
        {
            float4 position [[position]];
        };

        vertex SharpEmuLdsTestVertex sharpemu_lds_test_vs(
            uint id [[vertex_id]],
            constant float2* positions [[buffer(0)]])
        {
            SharpEmuLdsTestVertex vertex_out;
            vertex_out.position = float4(positions[id], 0.0f, 1.0f);
            return vertex_out;
        }
        """;

    private static Gen5MslShader Compile(uint[] words, bool structured)
    {
        var program = Gen5ComputeFixtures.DecodeOrThrow(new Gen5ComputeFixture("lane-slot-lds", words, 0, 0));
        var request = Gen5ComputeFixtures.RequestOrThrow(program, ShaderStage.Pixel,
            pixelOutputs: [new Gen5PixelOutputBinding(0, 0, Gen5PixelOutputKind.Uint)],
            enableGraphicsSubgroupOperations: true);
        Assert.True(
            Gen5MslTranslator.TryCompileProgram(
                request, out var shader, out var error, new Gen5MslCodegenOptions(StructuredControlFlow: structured)),
            error);
        return shader;
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
