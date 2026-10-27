// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Xunit;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

// sharpemu_load_words<N> is the typed buffer load's fast path for formats of whole dword
// components. It has to return exactly what N sharpemu_load_word calls return, for any
// byte address: an aligned run inside the binding is plain loads, an odd bias or a run
// that crosses the end of the binding takes the per-dword path.
public sealed class MslLoadWordsTests
{
    private const int DataWords = 16;
    private const int CaseBase = 16;
    private const int OutputBase = 64;
    private const int OutputStride = 16;

    private const string Kernel = @"
kernel void load_words_test(device uint* b [[buffer(0)]], uint t [[thread_position_in_threadgroup]])
{
    uint addr = b[16u + t * 2u];
    uint bytes = b[17u + t * 2u];
    uint o = 64u + t * 16u;
    uint4 four = sharpemu_load_words<4>(b, bytes, addr);
    uint4 two = sharpemu_load_words<2>(b, bytes, addr);
    uint4 one = sharpemu_load_words<1>(b, bytes, addr);
    for (uint i = 0u; i < 4u; i++)
    {
        b[o + i] = four[i];
        b[o + 4u + i] = sharpemu_load_word(b, bytes, addr + 4u * i);
    }
    b[o + 8u] = two[0];
    b[o + 9u] = two[1];
    b[o + 10u] = sharpemu_load_word(b, bytes, addr);
    b[o + 11u] = sharpemu_load_word(b, bytes, addr + 4u);
    b[o + 12u] = one[0];
    b[o + 13u] = sharpemu_load_word(b, bytes, addr);
}
";

    // addr, bytes, then the expected four dwords (null = only compare against the old path).
    public static TheoryData<uint, uint> Cases => new()
    {
        { 0u, 64u },    // aligned, all inside
        { 16u, 64u },   // aligned, interior
        { 48u, 64u },   // aligned, last four dwords
        { 2u, 64u },    // an odd bias straddles every dword
        { 1u, 64u },
        { 7u, 64u },
        { 4u, 16u },    // aligned, the run crosses the end after three dwords
        { 8u, 16u },    // aligned, two dwords inside
        { 12u, 16u },   // aligned, one dword inside
        { 16u, 16u },   // aligned, entirely past the end
        { 0u, 0u },     // an empty binding
        { 10u, 16u },   // unaligned and crossing the end
        { 0u, 8u },     // aligned and shorter than a full run
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public void TheRunMatchesPerDwordLoads(uint address, uint bytes) => CheckRun(address, bytes, raw: false);

    [Theory]
    [MemberData(nameof(Cases))]
    [InlineData(0xFFFF_FFFCu, 64u)]
    [InlineData(0xFFFF_FFFFu, 64u)]
    [InlineData(1u, 3u)]
    [InlineData(3u, 7u)]
    public void RawDwordRunMatchesScalarLoadsIncludingAddressWrap(uint address, uint bytes) =>
        CheckRun(address, bytes, raw: true);

    private static void CheckRun(uint address, uint bytes, bool raw)
    {
        if (!MetalNative.IsAvailable)
        {
            Assert.False(
                OperatingSystem.IsMacOS() && Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") == "1",
                "The required gate cannot run the load-words test without a Metal device.");
            return;
        }

        var prelude = Gen5NggFixtures.CompileOrThrow(Gen5NggFixtures.CullingTriangle()).Source;
        var kernel = raw
            ? Kernel.Replace("sharpemu_load_words", "sharpemu_load_dwords", StringComparison.Ordinal)
                .Replace("sharpemu_load_word(", "sharpemu_load_dword(", StringComparison.Ordinal)
            : Kernel;
        Assert.True(MetalNative.TryCompileLibrary(prelude + kernel, out var library, out var compileError), compileError);

        var contents = new byte[(OutputBase + OutputStride) * sizeof(uint)];
        var words = new uint[contents.Length / sizeof(uint)];
        for (var index = 0; index < DataWords; index++)
        {
            words[index] = 0x01010101u * (uint)(index + 1) ^ 0xA5A5_0000u;
        }

        words[CaseBase] = address;
        words[CaseBase + 1] = bytes;
        Buffer.BlockCopy(words, 0, contents, 0, contents.Length);
        Assert.True(
            MetalNative.TryExecuteOnBuffer(library, "load_words_test", contents, 1, out var result, out var error),
            error);

        var output = new uint[OutputStride];
        Buffer.BlockCopy(result, OutputBase * sizeof(uint), output, 0, OutputStride * sizeof(uint));
        for (var i = 0; i < 4; i++)
        {
            Assert.Equal(output[4 + i], output[i]);
        }

        Assert.Equal(output[10], output[8]);
        Assert.Equal(output[11], output[9]);
        Assert.Equal(output[13], output[12]);

        // A dword wholly inside both the binding and the data reads as the bytes that sit there.
        for (var i = 0; i < 4; i++)
        {
            var at = unchecked(address + (uint)(4 * i));
            if (raw)
            {
                Assert.Equal((at >> 2) < (bytes >> 2) ? words[at >> 2] : 0u, output[i]);
                continue;
            }

            if ((address & 3) == 0 && at + 4 <= bytes && at + 4 <= DataWords * 4)
            {
                Assert.Equal(words[at / 4], output[i]);
            }
            else if ((address & 3) == 0)
            {
                Assert.Equal(0u, output[i]);
            }
        }
    }
}
