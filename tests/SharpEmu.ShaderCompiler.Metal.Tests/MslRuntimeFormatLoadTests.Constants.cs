// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Diagnostics;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

// Buffer words as function constants (PortableBufferWord.ConstantsEnabled): a function made
// with the values reads them, one made without reads the shader data as before.
public sealed partial class MslRuntimeFormatLoadTests
{
    private static Gen5MslCodegenOptions ConstantOptions(bool constants) =>
        new(ForceRegisterInitialization: true) { BufferWordConstants = constants };

    // BUFFER_LOAD_FORMAT with IDXEN: v1 = s0 is the index, the stride multiplies it.
    private static Gen5ComputeFixture IndexedFixture(uint components) => new(
        "runtime-format-load-indexed",
        [
            0x7E020200, // v_mov_b32 v1, s0
            0xE0002000 | ((components - 1) << 18), 0x80020401, // buffer_load_format v[4:], v1, s[8:11] idxen
            0xE0700000, 0x80030400,
            0xE0700004, 0x80030500,
            0xE0700008, 0x80030600,
            0xE070000C, 0x80030700,
            0xBF810000,
        ], 12, BufferBytes);

    // BUFFER_STORE_FORMAT_XYZW v[4:7] to s[8:11] at offset s0: a written formatted buffer.
    private static Gen5ComputeFixture StoreFormatFixture() => new(
        "runtime-format-store",
        [
            0x7E020200, // v_mov_b32 v1, s0
            0xE01C1000, 0x80020401, // buffer_store_format_xyzw v[4:7], v1, s[8:11] offen
            0xBF810000,
        ], 12, BufferBytes);

    private static int ReadBinding(Gen5MslShader shader) =>
        (int)((shader.BufferWordConstants.Single(constant => (constant.Index - PortableBufferWord.FirstConstantIndex) % 2 == 1).Index -
            PortableBufferWord.FirstConstantIndex) / 2);

    private static byte[] Push(BindingLayout layout, uint offset, IReadOnlyDictionary<int, (uint Stride, uint Word3)> words)
    {
        var push = Push(layout, 0, offset);
        foreach (var (binding, (stride, word3)) in words)
        {
            var at = (int)(layout.FirstBufferWordDword() + (uint)binding * PortableBufferWord.DwordCount) * 4;
            BinaryPrimitives.WriteUInt32LittleEndian(push.AsSpan(at), stride);
            BinaryPrimitives.WriteUInt32LittleEndian(push.AsSpan(at + 4), word3);
        }

        return push;
    }

    [Fact]
    public void ConstantIndicesFollowTheBindingAndWord()
    {
        Assert.Equal(1u, PortableBufferWord.ConstantIndex(0, 0));
        Assert.Equal(2u, PortableBufferWord.ConstantIndex(0, 1));
        Assert.Equal(7u, PortableBufferWord.ConstantIndex(3, 0));
        Assert.Equal(8u, PortableBufferWord.ConstantIndex(3, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => PortableBufferWord.ConstantIndex(0, 2));
        Assert.NotEqual(Gen5MslShader.MipStatisticsFunctionConstantIndex, PortableBufferWord.FirstConstantIndex);
    }

    [Fact]
    public void SwitchOffKeepsTodaysSource()
    {
        foreach (var fixture in new[] { Fixture(4), IndexedFixture(3), StoreFormatFixture() })
        {
            var (today, _) = Compile(fixture, new Gen5MslCodegenOptions(ForceRegisterInitialization: true));
            var (off, _) = Compile(fixture, ConstantOptions(false));
            Assert.Equal(today.Source, off.Source);
            Assert.Empty(off.BufferWordConstants);
            Assert.DoesNotContain(PortableBufferWord.ConstantNamePrefix, off.Source);
            Assert.Empty(Gen5MslShader.ParseBufferWordConstants(off.Source));
        }
    }

    [Fact]
    public void EachWordReadIsOneConstant()
    {
        var (shader, layout) = Compile(IndexedFixture(4), ConstantOptions(true));
        var binding = ReadBinding(shader);
        var words = layout.FirstBufferWordDword() + (uint)binding * PortableBufferWord.DwordCount;
        Assert.Equal(
            [
                new Gen5MslShader.BufferWordConstant(PortableBufferWord.ConstantIndex(binding, 0), words),
                new Gen5MslShader.BufferWordConstant(PortableBufferWord.ConstantIndex(binding, 1), words + 1),
            ],
            shader.BufferWordConstants);
        Assert.Equal(shader.BufferWordConstants, Gen5MslShader.ParseBufferWordConstants(shader.Source));
        foreach (var constant in shader.BufferWordConstants)
        {
            Assert.Single(System.Text.RegularExpressions.Regex.Matches(
                shader.Source, $@"\[\[function_constant\({constant.Index}\)\]\]"));
        }

        Assert.Contains("sharpemu_format_fetch_constant(b", shader.Source);
    }

    [Fact]
    public void ADeclarationThatDoesNotParseIsRefused()
    {
        Assert.Equal(
            [new Gen5MslShader.BufferWordConstant(3, 12), new Gen5MslShader.BufferWordConstant(4, 13)],
            Gen5MslShader.ParseBufferWordConstants(
                "constant uint sharpemu_fc_bw4_sd13 [[function_constant(4)]];\n" +
                "constant uint sharpemu_fc_bw3_sd12 [[function_constant(3)]];\n"));
        Assert.Throws<FormatException>(() => Gen5MslShader.ParseBufferWordConstants(
            "constant uint sharpemu_fc_bw3_sd12 [[function_constant(5)]];\n"));
    }

    [Fact]
    public void AnUnindexedAccessReadsNoStrideConstant()
    {
        var (shader, _) = Compile(Fixture(4), ConstantOptions(true));
        var constant = Assert.Single(shader.BufferWordConstants);
        Assert.Equal(1u, (constant.Index - PortableBufferWord.FirstConstantIndex) % 2);
    }

    [Fact]
    public void AWrittenFormattedBufferKeepsItsBakedFormat()
    {
        var (shader, _) = Compile(StoreFormatFixture(), ConstantOptions(true));
        Assert.DoesNotContain(shader.BufferWordConstants, constant => (constant.Index - PortableBufferWord.FirstConstantIndex) % 2 == 1);
        Assert.DoesNotContain("sharpemu_format_fetch_constant", shader.Source);
    }

    [Fact]
    public void AProgramOverTheSiteLimitKeepsTheRuntimeDecode()
    {
        var (shader, _) = Compile(ManyFetchFixture(Gen5MslTranslator.FormatFetchConstantSiteLimit + 1), ConstantOptions(true));
        Assert.NotEmpty(shader.BufferWordConstants);
        Assert.DoesNotContain("sharpemu_format_fetch_constant", shader.Source);
    }

    // count BUFFER_LOAD_FORMAT_XY from s[8:11] at offsets 0, 4, ..., each summed into v1.
    private static Gen5ComputeFixture ManyFetchFixture(int count)
    {
        var words = new List<uint> { 0x7E020280, 0x7E040200 }; // v_mov_b32 v1, 0; v_mov_b32 v2, s0
        for (var site = 0; site < count; site++)
        {
            words.AddRange([0xE0041000 | (uint)(site * 4 % 4096), 0x80020402]); // buffer_load_format_xy v[4:5], v2, s[8:11] offen
            words.AddRange([0xBF8C0070, 0x4A020304, 0x4A020305]);
        }

        words.AddRange([0xE0700000, 0x80030100, 0xBF810000]); // buffer_store_dword v1, off, s[12:15]
        return new("runtime-format-many", words.ToArray(), 12, BufferBytes);
    }

    private static IEnumerable<uint> DecodableFormats() =>
        Enumerable.Range(0, 128).Select(static format => (uint)format)
            .Where(static format => Gfx10UnifiedFormat.TryDecode(format, out var data, out _) && data != 0);

    [Theory]
    [InlineData(1u)]
    [InlineData(2u)]
    [InlineData(3u)]
    [InlineData(4u)]
    public void TheSpecializedFetchMatchesTheRuntimeDecode(uint components)
    {
        var (runtime, layout) = Compile(Fixture(components), ConstantOptions(false));
        var (constant, constantLayout) = Compile(Fixture(components), ConstantOptions(true));
        Assert.Equal(layout.FirstBufferWordDword(), constantLayout.FirstBufferWordDword());
        var word3Index = Assert.Single(constant.BufferWordConstants).Index;
        if (!RequireDevice()) return;
        Assert.True(MetalNative.TryCompileLibrary(runtime.Source, out var runtimeLibrary, out var runtimeError), runtimeError);
        Assert.True(MetalNative.TryCompileLibrary(constant.Source, out var constantLibrary, out var constantError), constantError);
        var data = Enumerable.Range(0, BufferBytes).Select(i => (byte)(i * 37 + 11)).ToArray();
        // Identity, constants 0 and 1, reordered, repeated W, the reserved selectors.
        var swizzles = new uint[] { 0xFAC, 0x204, 0x977, 0xFFF, 0x0D1 };
        var offsets = new uint[] { 0, 1, 2, 33, 64, 113, 120, 122, 124, 126, 127, 128, 200 };
        var specialized = 0;
        foreach (var format in DecodableFormats())
        {
            foreach (var swizzle in swizzles)
            {
                var word3 = (format << 12) | swizzle;
                // Only the constant can give the right answer: the shader data holds another
                // format and swizzle.
                var wrong = (((format + 37) & 0x7F) << 12) | (swizzle ^ 0x249);
                foreach (var offset in offsets)
                {
                    var right = Push(layout, word3, offset);
                    Assert.True(MetalNative.TryExecuteWithArgumentBuffer(runtimeLibrary, runtime.EntryPoint, data, right,
                        runtime.ArgumentLayout!, 1, out var expected, out _, out var runtimeRunError), runtimeRunError);
                    // No values: the run-time path, exactly today's. Made through
                    // newFunctionWithName:constantValues: with none: a plain newFunctionWithName:
                    // function of a source with function constants cannot build a pipeline
                    // (Metal validation: "Use newFunctionWithName:constantValues:").
                    Assert.True(MetalNative.TryExecuteWithArgumentBuffer(constantLibrary, constant.EntryPoint, data, right,
                        constant.ArgumentLayout!, 1, out var unspecialized, out _, out var plainError, constants: []), plainError);
                    Assert.True(expected.AsSpan().SequenceEqual(unspecialized),
                        $"unspecialized components={components} format={format} swizzle=0x{swizzle:X} offset={offset}");
                    Assert.True(MetalNative.TryExecuteWithArgumentBuffer(constantLibrary, constant.EntryPoint, data,
                        Push(layout, wrong, offset), constant.ArgumentLayout!, 1, out var actual, out _, out var specializedError,
                        constants: [(word3Index, word3)]), specializedError);
                    Assert.True(expected.AsSpan().SequenceEqual(actual),
                        $"specialized components={components} format={format} swizzle=0x{swizzle:X} offset={offset}");
                }

                specialized++;
            }
        }

        output.WriteLine($"components={components} specializations={specialized}");
    }

    [Theory]
    [InlineData(4u, 0u)]
    [InlineData(4u, 6u)]
    [InlineData(3u, 13u)]
    public void ASpecializedStrideMatchesTheRuntimeStride(uint components, uint index)
    {
        var (runtime, layout) = Compile(IndexedFixture(components), ConstantOptions(false));
        var (constant, _) = Compile(IndexedFixture(components), ConstantOptions(true));
        var binding = ReadBinding(constant);
        var strideIndex = PortableBufferWord.ConstantIndex(binding, 0);
        Assert.Contains(constant.BufferWordConstants, word => word.Index == strideIndex);
        Assert.Empty(runtime.BufferWordConstants);
        if (!RequireDevice()) return;
        Assert.True(MetalNative.TryCompileLibrary(runtime.Source, out var runtimeLibrary, out var runtimeError), runtimeError);
        Assert.True(MetalNative.TryCompileLibrary(constant.Source, out var constantLibrary, out var constantError), constantError);
        var data = Enumerable.Range(0, BufferBytes).Select(i => (byte)(i * 37 + 11)).ToArray();
        foreach (var stride in new uint[] { 4, 7, 8, 12, 16 })
        {
            foreach (var word3 in new[] { (77u << 12) | 0xFAC, (56u << 12) | 0xFAC, (24u << 12) | 0x977 })
            {
                var right = Push(layout, index, new Dictionary<int, (uint, uint)> { [binding] = (stride, word3) });
                Assert.True(MetalNative.TryExecuteWithArgumentBuffer(runtimeLibrary, runtime.EntryPoint, data, right,
                    runtime.ArgumentLayout!, 1, out var expected, out _, out var runtimeRunError), runtimeRunError);
                // The stride alone specialized; word 3 still from the shader data.
                var wrongStride = Push(layout, index, new Dictionary<int, (uint, uint)> { [binding] = (stride * 3 + 1, word3) });
                Assert.True(MetalNative.TryExecuteWithArgumentBuffer(constantLibrary, constant.EntryPoint, data, wrongStride,
                    constant.ArgumentLayout!, 1, out var strideOnly, out _, out var strideError,
                    constants: [(strideIndex, stride)]), strideError);
                Assert.True(expected.AsSpan().SequenceEqual(strideOnly), $"stride={stride} word3=0x{word3:X} index={index}");
                // Both.
                var wrongBoth = Push(layout, index, new Dictionary<int, (uint, uint)> { [binding] = (stride * 3 + 1, word3 ^ 0x5A249) });
                Assert.True(MetalNative.TryExecuteWithArgumentBuffer(constantLibrary, constant.EntryPoint, data, wrongBoth,
                    constant.ArgumentLayout!, 1, out var both, out _, out var bothError,
                    constants: [(strideIndex, stride), (PortableBufferWord.ConstantIndex(binding, 1), word3)]), bothError);
                Assert.True(expected.AsSpan().SequenceEqual(both), $"both stride={stride} word3=0x{word3:X} index={index}");
            }
        }
    }

    [Fact]
    public void LibraryCompileTimeWithManySites()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_GPU_BENCH") != "1") return;
        if (!RequireDevice()) return;
        foreach (var sites in new[] { 8, Gen5MslTranslator.FormatFetchConstantSiteLimit, 256 })
        {
            foreach (var (constants, limit) in new[] { (false, 0), (true, Gen5MslTranslator.FormatFetchConstantSiteLimit), (true, 4096) })
            {
                if (limit == 4096 && sites <= Gen5MslTranslator.FormatFetchConstantSiteLimit) continue;
                var (shader, _) = Compile(ManyFetchFixture(sites), ConstantOptions(constants) with { FormatFetchConstantSiteLimit = limit });
                // A distinct comment keeps the system shader cache out of the timing.
                var source = shader.Source + $"\n// {Guid.NewGuid()}\n";
                var clock = Stopwatch.StartNew();
                Assert.True(MetalNative.TryCompileLibrary(source, out var library, out var error), error);
                var libraryMs = clock.Elapsed.TotalMilliseconds;
                clock.Restart();
                Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, new byte[BufferBytes], new byte[PushData.ByteSize],
                    shader.ArgumentLayout!, 1, out _, out _, out var runError, constants: []), runError);
                var pipelineMs = clock.Elapsed.TotalMilliseconds;
                var folded = shader.Source.Contains("sharpemu_format_fetch_constant(b", StringComparison.Ordinal);
                output.WriteLine(FormattableString.Invariant(
                    $"sites={sites} constants={constants} folded={folded} library_ms={libraryMs:F0} unspecialized_pipeline_ms={pipelineMs:F0} source_bytes={source.Length}"));
            }
        }
    }
}
