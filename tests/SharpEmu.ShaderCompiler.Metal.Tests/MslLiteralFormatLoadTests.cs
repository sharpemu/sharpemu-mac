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

    public static IEnumerable<object[]> BufferFormats() => Enumerable.Range(1, 77)
        .Where(format => Gfx10UnifiedFormat.TryDecode((uint)format, out var data, out _) && data != 0)
        .Select(format => new object[] { (uint)format });

    [Theory]
    [MemberData(nameof(BufferFormats))]
    public void NarrowFormatsMatchThePortableDecoderIncludingPartialTails(uint format)
    {
        if (!RequireDevice()) return;
        foreach (var offset in new[] { 17, 56, 62 })
        {
            var swizzle = offset == 56 ? Swizzle(7, 4, 1, 0) : Identity;
            var narrow = Compile(format, swizzle, offset, false, out var narrowPush);
            var portable = Compile(format, swizzle, offset, true, out var portablePush);
            var initial = new byte[BufferBytes];
            new Random(173).NextBytes(initial);
            byte[] Execute(Gen5MslShader shader, byte[] push)
            {
                Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var error), error);
                Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, initial, push,
                    shader.ArgumentLayout!, 1, out var result, out error), error);
                return result;
            }
            Assert.DoesNotContain("sharpemu_gfx10_formats[(", narrow.Source, StringComparison.Ordinal);
            Assert.Contains("sharpemu_gfx10_formats[(", portable.Source, StringComparison.Ordinal);
            Assert.Equal(Execute(portable, portablePush), Execute(narrow, narrowPush));
        }
    }

    [Fact]
    public void ALiteralWholeDwordFormatLoadsOneRun()
    {
        var shader = Compile(Float32x4, Identity, offset: 16);
        Assert.Contains("sharpemu_load_words<4>(b0, ", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("sharpemu_gfx10_formats[(", shader.Source, StringComparison.Ordinal);

        // Only the selected components make the run: X and Y here.
        var two = Compile(Float32x4, Swizzle(4, 5, 0, 0), offset: 16);
        Assert.Contains("sharpemu_load_words<2>(b0, ", two.Source, StringComparison.Ordinal);

        // Narrow components also resolve their layout before executing the shader.
        var packed = Compile(14, Identity, offset: 16);
        Assert.DoesNotContain("sharpemu_load_words<", packed.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("sharpemu_gfx10_formats[(", packed.Source, StringComparison.Ordinal);
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
    private static Gen5MslShader Compile(uint format, uint swizzle, int offset)
        => Compile(format, swizzle, offset, false, out _);

    private static Gen5MslShader Compile(uint format, uint swizzle, int offset, bool portable, out byte[] push)
    {
        var fixture = new Gen5ComputeFixture(
            "format-load-literal",
            [
                0x7E0202FF, (uint)offset,                     // v_mov_b32 v1, offset
                0xE0000000u | (3u << 18) | (1u << 12),        // buffer_load_format_xyzw v[4:7], v1, s[8:11], 0 offen
                0x80020401,
                0xE0700000, 0x80030400,                       // buffer_store_dword v4, s[12:15] offset:0
                0xE0700004, 0x80030500,                       // buffer_store_dword v5 offset:4
                0xE0700008, 0x80030600,                       // buffer_store_dword v6 offset:8
                0xE070000C, 0x80030700,                       // buffer_store_dword v7 offset:12
                0xBF810000,                                   // s_endpgm
            ],
            8,
            BufferBytes);
        var userData = new uint[16];
        userData[8] = 0x200000;
        userData[9] = 0;
        userData[10] = BufferBytes;
        userData[11] = (format << 12) | swizzle;
        Array.Copy(userData, 8, userData, 12, 4);
        var request = Gen5ComputeFixtures.RequestOrThrow(
            Gen5ComputeFixtures.DecodeOrThrow(fixture),
            ShaderStage.Compute,
            runtimeInputs: new ResourceRuntimeInputs { UserData = userData });
        push = new byte[PushData.ByteSize];
        if (portable)
        {
            var plan = ShaderResourcePlan.Extract(request.Program, ShaderStage.Compute, 1, 0, 16);
            var resources = new SpecializedResourceInfo { Info = request.Resources.Info.Clone() };
            foreach (var buffer in resources.Info.Buffers)
            {
                buffer.PackedStride = 0;
                if (buffer.Formatted && !buffer.Written) buffer.DescriptorFormat = 0;
            }
            var layout = BindingLayout.Allocate(resources.Info,
                BindingLayout.CollectUserDataRegisters(request.Program, 0, 16), false, false, false, 0, portableBuffers: true);
            for (var i = 0; i < layout.UserDataRegisters.Count; i++)
                BitConverter.TryWriteBytes(push.AsSpan(i * 4), userData[layout.UserDataRegisters[i]]);
            for (uint i = 0; i < resources.Info.Buffers.Count; i++)
                BitConverter.TryWriteBytes(push.AsSpan((int)(layout.FirstBufferWordDword() + i * 2 + 1) * 4), userData[11]);
            request = new ShaderCompileRequest(plan, resources, layout) { LocalSizeX = 1 };
        }
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        return shader!;
    }
}
