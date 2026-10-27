// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

// BUFFER_LOAD_FORMAT_XYZW through a descriptor whose format and swizzle the specialization
// fixed, for a format of whole dwords: the selected components load as one run and only
// they are bounds checked. The expectations are the per-component rules of the
// descriptor-table form: a selected component the format lacks reads as the missing value
// (W one, the rest zero), and an access out of range reads each destination's constant,
// one for a constant-one selector and zero otherwise.
public sealed class MslLiteralFormatLoadTests(ITestOutputHelper output)
{
    private const uint Float32x4 = 77;
    private const uint Float32x3 = 74;
    private const uint OneBits = 0x3F800000;
    private const int BufferBytes = 64;

    private static uint Swizzle(uint x, uint y, uint z, uint w) => x | (y << 3) | (z << 6) | (w << 9);

    private static readonly uint Identity = Swizzle(4, 5, 6, 7);

    [Fact]
    public void ALiteralWholeDwordFormatLoadsOneRun()
    {
        var shader = Compile(Float32x4, Identity, offset: 16);
        Assert.Contains("sharpemu_load_words<4>(b0, ", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("sharpemu_gfx10_formats[(", shader.Source, StringComparison.Ordinal);

        // Only the selected components make the run: X and Y here.
        var two = Compile(Float32x4, Swizzle(4, 5, 0, 0), offset: 16);
        Assert.Contains("sharpemu_load_words<2>(b0, ", two.Source, StringComparison.Ordinal);

        // A format of narrower components keeps the descriptor-table form.
        var packed = Compile(14, Identity, offset: 16);
        Assert.DoesNotContain("sharpemu_load_words<", packed.Source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_gfx10_formats[(", packed.Source, StringComparison.Ordinal);
    }

    // dword offset 16 is aligned and 18 is not; 56 and 58 run past the end of the buffer.
    [Theory]
    [InlineData(Float32x4, 4u, 5u, 6u, 7u, 16)]
    [InlineData(Float32x4, 4u, 5u, 6u, 7u, 18)]
    [InlineData(Float32x4, 7u, 6u, 5u, 4u, 20)]
    [InlineData(Float32x4, 4u, 1u, 0u, 7u, 24)]
    [InlineData(Float32x4, 6u, 4u, 6u, 5u, 22)]
    [InlineData(Float32x3, 4u, 5u, 6u, 7u, 16)]
    [InlineData(Float32x3, 4u, 5u, 6u, 7u, 18)]
    [InlineData(Float32x3, 7u, 7u, 4u, 4u, 28)]
    [InlineData(Float32x4, 4u, 5u, 6u, 7u, 56)]
    [InlineData(Float32x4, 4u, 5u, 6u, 7u, 58)]
    [InlineData(Float32x4, 4u, 5u, 0u, 0u, 56)]
    [InlineData(Float32x4, 4u, 5u, 6u, 0u, 56)]
    [InlineData(Float32x4, 4u, 1u, 0u, 1u, 60)]
    [InlineData(Float32x3, 4u, 0u, 0u, 7u, 62)]
    [InlineData(Float32x3, 4u, 0u, 0u, 7u, 60)]
    public void TheRunReadsWhatThePerComponentRulesRead(uint format, uint x, uint y, uint z, uint w, int offset)
    {
        if (!RequireDevice()) return;
        var selectors = new[] { x, y, z, w };
        var shader = Compile(format, Swizzle(x, y, z, w), offset);
        var initial = new byte[BufferBytes];
        for (var index = 0; index < initial.Length; index++)
        {
            initial[index] = (byte)((index * 37) + 11);
        }

        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var compileError), compileError);
        Assert.True(
            MetalNative.TryExecuteWithArgumentBuffer(
                library, shader.EntryPoint, initial, new byte[PushData.ByteSize], shader.ArgumentLayout!, 1,
                out var result, out var runError),
            runError);

        var present = format == Float32x4 ? 4 : 3;
        var run = 0;
        foreach (var selector in selectors)
        {
            if (selector >= 4 && selector - 4 < present)
            {
                run = Math.Max(run, (int)(selector - 4) + 1);
            }
        }

        var inBounds = run == 0 || (run * 4 <= BufferBytes && offset <= BufferBytes - (run * 4));
        for (var destination = 0; destination < 4; destination++)
        {
            var selector = selectors[destination];
            var constant = selector == 1 ? OneBits : 0u;
            uint expected;
            if (selector >= 4 && inBounds)
            {
                var component = (int)selector - 4;
                expected = component >= present
                    ? (selector == 7 ? OneBits : 0u)
                    : BinaryPrimitives.ReadUInt32LittleEndian(initial.AsSpan(offset + (component * 4)));
            }
            else
            {
                expected = constant;
            }

            Assert.Equal(expected, BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan(destination * 4)));
        }
    }

    [Theory]
    [InlineData(14u)] // 8x2 UNORM: missing Z/W and a partial final word.
    [InlineData(44u)] // 2:10:10:10 UNORM, all components in one word.
    [InlineData(56u)] // 8x4 UNORM.
    [InlineData(57u)] // 8x4 SNORM.
    [InlineData(60u, false)] // 8x4 UINT.
    [InlineData(60u, true)] // Coherent loads use the same packed span.
    [InlineData(61u)] // 8x4 SINT.
    [InlineData(65u)] // 16x4 UNORM, two words.
    [InlineData(71u)] // 16x4 FLOAT, including arbitrary half bit patterns.
    public void PackedRunsMatchTheDecoderAtAlignedUnalignedAndPartialElements(uint format, bool glc = false)
    {
        if (!RequireDevice()) return;
        foreach (var offset in new[] { 16, 17, 61, 63 })
        foreach (var swizzle in new[] { Identity, Swizzle(7, 6, 5, 4), Swizzle(4, 1, 0, 4) })
        {
            var initial = new byte[BufferBytes];
            for (var i = 0; i < initial.Length; i++) initial[i] = (byte)((i * 37) + 11);
            byte[]? reference = null;
            foreach (var fast in new[] { false, true })
            {
                var shader = Compile(format, swizzle, offset, packed: fast, glc: glc);
                if (fast) Assert.Contains(glc ? "sharpemu_load_words_coherent<" : "sharpemu_load_words<", shader.Source, StringComparison.Ordinal);
                Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var compileError), compileError);
                Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint,
                    initial, new byte[PushData.ByteSize], shader.ArgumentLayout!, 1,
                    out var result, out var runError), runError);
                if (!fast) reference = result;
                else Assert.Equal(reference, result);
                // Independent oracle for the four unsigned-byte components.
                if (format == 60 && swizzle == Identity)
                {
                    for (var i = 0; i < 4; i++)
                        Assert.Equal(offset <= BufferBytes - 4 ? (uint)initial[offset + i] : 0u,
                            BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan(i * 4)));
                }
            }
        }
    }

    private bool RequireDevice()
    {
        if (!MetalNative.IsAvailable)
        {
            Assert.False(
                OperatingSystem.IsMacOS() && Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") == "1",
                "The required gate cannot run the format load test without a Metal device.");
            output.WriteLine("No Metal device on this host; the format load test skipped.");
            return false;
        }

        return true;
    }

    // v1 = the byte offset (OFFEN); buffer_load_format_xyzw v[4:7] from s[8:11]; the four
    // registers are stored at bytes 0, 4, 8 and 12 of the same buffer.
    private static Gen5MslShader Compile(uint format, uint swizzle, int offset, bool packed = false, bool glc = false)
    {
        var fixture = new Gen5ComputeFixture(
            "format-load-literal",
            [
                0x7E0202FF, (uint)offset,                     // v_mov_b32 v1, offset
                0xE0000000u | (3u << 18) | (1u << 12) | (glc ? 1u << 14 : 0),        // buffer_load_format_xyzw v[4:7], v1, s[8:11], 0 offen
                0x80020401,
                0xE0700000, 0x80020400,                       // buffer_store_dword v4 offset:0
                0xE0700004, 0x80020500,                       // buffer_store_dword v5 offset:4
                0xE0700008, 0x80020600,                       // buffer_store_dword v6 offset:8
                0xE070000C, 0x80020700,                       // buffer_store_dword v7 offset:12
                0xBF810000,                                   // s_endpgm
            ],
            8,
            BufferBytes);
        var userData = new uint[16];
        userData[8] = 0x200000;
        userData[9] = 0;
        userData[10] = BufferBytes;
        userData[11] = (format << 12) | swizzle;
        var request = Gen5ComputeFixtures.RequestOrThrow(
            Gen5ComputeFixtures.DecodeOrThrow(fixture),
            ShaderStage.Compute,
            runtimeInputs: new ResourceRuntimeInputs { UserData = userData });
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error,
            new Gen5MslCodegenOptions { PackedFormatLoads = packed }), error);
        return shader!;
    }
}
