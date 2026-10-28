// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>
/// The MRTZ export per SPI_SHADER_Z_FORMAT: the 32-bit formats carry depth in X, stencil
/// in Y and the sample mask in Z; UINT16_ABGR (compressed) carries stencil in X[23:16] and
/// the sample mask in Y[15:0]. DB_SHADER_CONTROL decides which of them the DB takes. The
/// device tests render into a depth/stencil target whose stencil op replaces the value
/// with the reference, so an exported stencil shows up in the stencil plane.
/// </summary>
public sealed class Gen5MslMrtzExportTests(ITestOutputHelper output)
{
    private const uint TargetSize = 4;
    private const uint Compressed = 0x400;
    private const uint EncoderReference = 0x11;

    private const string VertexSource = """
        #include <metal_stdlib>
        using namespace metal;
        struct VsOut { float4 position [[position]]; };
        vertex VsOut mrtz_vs(uint id [[vertex_id]])
        {
            const float2 corners[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
            VsOut out;
            out.position = float4(corners[id], 0.5, 1.0);
            return out;
        }
        """;

    // v0..v3 = the MRTZ words, exp mrtz v0, v1, v2, v3 (enable mask and COMPR per case),
    // then exp mrt0 1.0 done vm.
    private static uint[] Program(uint[] registers, uint enableMask, uint flags) =>
    [
        0x7E0002FF, registers[0],
        0x7E0202FF, registers[1],
        0x7E0402FF, registers[2],
        0x7E0602FF, registers[3],
        0x7E0802F2,                                     // v_mov_b32 v4, 1.0
        0xF8000080 | flags | enableMask, 0x03020100,    // exp mrtz
        0xF800180F, 0x04040404,                         // exp mrt0 v4, v4, v4, v4 done vm
        0xBF810000,
    ];

    private static readonly uint Depth = BitConverter.SingleToUInt32Bits(0.25f);

    [Theory]
    // 32-bit layouts: stencil in Y, the low 8 bits.
    [InlineData(Gen5ColorExportFormat.GR32, 0x3u, 0u, 0x1A5u, 0xA5u)]
    [InlineData(Gen5ColorExportFormat.Abgr32, 0xFu, 0u, 0x1A5u, 0xA5u)]
    [InlineData(null, 0xFu, 0u, 0x1A5u, 0xA5u)]
    // UINT16_ABGR: stencil in X[23:16].
    [InlineData(Gen5ColorExportFormat.Uint16Abgr, 0x3u, Compressed, 0x00A51234u, 0xA5u)]
    [InlineData(null, 0x3u, Compressed, 0x00A51234u, 0xA5u)]
    // 32_R carries no stencil, and a disabled Y channel exports none: the reference stays.
    [InlineData(Gen5ColorExportFormat.R32, 0xFu, 0u, 0x1A5u, EncoderReference)]
    [InlineData(Gen5ColorExportFormat.Abgr32, 0x1u, 0u, 0x1A5u, EncoderReference)]
    public void ExportedStencilReplacesTheReference(
        Gen5ColorExportFormat? format, uint enableMask, uint flags, uint stencilWord, uint expected)
    {
        uint[] registers = flags == Compressed ? [stencilWord, 0xFFFF, 0, 0] : [Depth, stencilWord, 0xFFFF, 0];
        var shader = Compile(Program(registers, enableMask, flags), format, stencil: true, sampleMask: false);
        Assert.Equal(expected != EncoderReference, shader.Source.Contains("[[stencil]]", StringComparison.Ordinal));
        if (Render(shader) is not { } result) return;

        Assert.All(result.Stencil, value => Assert.Equal(expected, value));
    }

    [Fact]
    public void StencilExportNeedsTheControlBit()
    {
        var shader = Compile(Program([Depth, 0x1A5, 0, 0], 0xF, 0), Gen5ColorExportFormat.Abgr32, stencil: false, sampleMask: false);
        Assert.DoesNotContain("[[stencil]]", shader.Source, StringComparison.Ordinal);
        if (Render(shader) is not { } result) return;

        Assert.All(result.Stencil, value => Assert.Equal(EncoderReference, value));
    }

    [Theory]
    // 32_ABGR: the sample mask in Z. UINT16_ABGR: in Y[15:0].
    [InlineData(Gen5ColorExportFormat.Abgr32, 0u, 0u, false)]
    [InlineData(Gen5ColorExportFormat.Abgr32, 0u, 1u, true)]
    [InlineData(Gen5ColorExportFormat.Uint16Abgr, Compressed, 0xFFFF0000u, false)]
    [InlineData(Gen5ColorExportFormat.Uint16Abgr, Compressed, 0x00000001u, true)]
    [InlineData(null, Compressed, 0xFFFF0000u, false)]
    public void ExportedSampleMaskGatesCoverage(Gen5ColorExportFormat? format, uint flags, uint maskWord, bool covered)
    {
        uint[] registers = flags == Compressed ? [0, maskWord, 0, 0] : [Depth, 0, maskWord, 0];
        var shader = Compile(Program(registers, 0xF, flags), format, stencil: false, sampleMask: true);
        Assert.Contains("uint sharpemu_coverage [[sample_mask]];", shader.Source, StringComparison.Ordinal);
        if (Render(shader) is not { } result) return;

        // A fragment whose mask leaves sample 0 out writes neither color nor stencil.
        Assert.All(result.Stencil, value => Assert.Equal(covered ? EncoderReference : 0u, value));
        Assert.Equal(covered ? 1f : 0f, BitConverter.UInt32BitsToSingle(result.Texels[0]));
    }

    [Fact]
    public void SampleMaskExportNeedsTheControlBit()
    {
        var shader = Compile(Program([Depth, 0, 0, 0], 0xF, 0), Gen5ColorExportFormat.Abgr32, stencil: false, sampleMask: false);
        Assert.DoesNotContain("[[sample_mask]]", shader.Source, StringComparison.Ordinal);
    }

    // The guest asked for the depth test before the shader, but a depth, stencil or coverage export decides
    // the fragment after the shader ran, and Metal rejects depth or stencil outputs with
    // [[early_fragment_tests]]: no export may get the attribute; without one the request's flag does.
    [Theory]
    [InlineData(true, false, false, false)]
    [InlineData(false, true, false, false)]
    [InlineData(false, false, true, false)]
    [InlineData(true, true, true, false)]
    [InlineData(false, false, false, true)]
    public void AnEarlyDepthRequestGetsTheAttributeOnlyWithoutDepthStencilOrCoverageExports(
        bool depth, bool stencil, bool sampleMask, bool attribute)
    {
        var shader = Compile(Program([Depth, 0x1A5, 0xFFFF, 0], 0xF, 0), Gen5ColorExportFormat.Abgr32, stencil, sampleMask,
            depth, earlyDepth: true);
        Assert.Equal(attribute, shader.Source.Contains("[[early_fragment_tests]]", StringComparison.Ordinal));
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") == "1")
        {
            Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var error), $"{error}\n{shader.Source}");
            MetalNative.ReleaseCollisionLibrary(library);
        }
    }

    private static Gen5MslShader Compile(uint[] words, Gen5ColorExportFormat? format, bool stencil, bool sampleMask,
        bool depth = false, bool earlyDepth = false)
    {
        var program = Gen5ComputeFixtures.DecodeOrThrow(new Gen5ComputeFixture("pixel-mrtz-export", words, 0, 0));
        var request = Gen5ComputeFixtures.RequestOrThrow(program, ShaderStage.Pixel, pixelDepthExportFormat: format,
            pixelDepthExportEnable: depth, pixelStencilExportEnable: stencil, pixelSampleMaskExportEnable: sampleMask,
            pixelEarlyDepth: earlyDepth);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        return shader;
    }

    private (byte[] Stencil, uint[] Texels)? Render(Gen5MslShader shader)
    {
        if (!MetalNative.IsAvailable)
        {
            Assert.False(
                OperatingSystem.IsMacOS() && Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") == "1",
                "The required gate cannot run the MRTZ render without a Metal device.");
            output.WriteLine("No Metal device on this host; the MRTZ render skipped.");
            return null;
        }

        Assert.True(MetalNative.TryCompileLibrary(VertexSource, out var vertexLibrary, out var error), error);
        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var fragmentLibrary, out error), $"{error}\n{shader.Source}");
        var resources = new byte[Math.Max(shader.ArgumentLayout?.ByteSize ?? 0, 256)];
        Assert.True(
            MetalNative.TryRenderTrianglesStencil(vertexLibrary, "mrtz_vs", fragmentLibrary, shader.EntryPoint,
                [(resources, (uint)Gen5MslArgumentLayout.ResourcesBufferIndex)], 3, TargetSize, TargetSize,
                MetalNative.Rgba32FloatPixelFormat, EncoderReference, out var stencil, out var texels, out error),
            $"{error}\n{shader.Source}");
        return (stencil, texels);
    }
}
