// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>
/// Every IMAGE_ATOMIC_* form run on the device against an R32Uint texture
/// created for shader atomics. Each case seeds one texel, runs the atomic with
/// GLC set so the pre-operation value comes back in VDATA, and checks both the
/// returned value and the texel the atomic left behind.
/// </summary>
public sealed class MetalImageAtomicExecutionTests(ITestOutputHelper output)
{
    // dfmt 4 / nfmt 4: a single-channel 32-bit uint image, the only class that
    // reaches an atomic binding.
    private const uint AtomicUintFormat = 20u;
    private const uint Width = 4;
    private const uint Height = 1;

    // The seeded texel and the operands the fixture puts in VDATA and VDATA+1.
    private const uint Initial = 0x00001234;
    private const uint Operand = 0x00000011;
    private const uint Replacement = 0x0000ABCD;

    [Theory]
    [InlineData("ImageAtomicSwap", Initial, Operand, Replacement, Initial, Operand)]
    [InlineData("ImageAtomicAdd", Initial, Operand, Replacement, Initial, 0x00001245u)]
    [InlineData("ImageAtomicSub", Initial, Operand, Replacement, Initial, 0x00001223u)]
    [InlineData("ImageAtomicUmin", Initial, Operand, Replacement, Initial, Operand)]
    [InlineData("ImageAtomicUmax", Initial, Operand, Replacement, Initial, Initial)]
    [InlineData("ImageAtomicAnd", Initial, Operand, Replacement, Initial, 0x00000010u)]
    [InlineData("ImageAtomicOr", Initial, Operand, Replacement, Initial, 0x00001235u)]
    [InlineData("ImageAtomicXor", Initial, Operand, Replacement, Initial, 0x00001225u)]
    // The signed minimum and maximum have to beat Metal's unsigned members: the
    // seeded texel reads as a large negative int, so signed and unsigned differ.
    [InlineData("ImageAtomicSmin", 0x80000000u, Operand, Replacement, 0x80000000u, 0x80000000u)]
    [InlineData("ImageAtomicUmin", 0x80000000u, Operand, Replacement, 0x80000000u, Operand)]
    [InlineData("ImageAtomicSmax", 0x80000000u, Operand, Replacement, 0x80000000u, Operand)]
    [InlineData("ImageAtomicUmax", 0x80000000u, Operand, Replacement, 0x80000000u, 0x80000000u)]
    // INC wraps to zero at the limit DATA names, DEC reloads it from zero.
    [InlineData("ImageAtomicInc", 5u, 9u, Replacement, 5u, 6u)]
    [InlineData("ImageAtomicInc", 9u, 9u, Replacement, 9u, 0u)]
    [InlineData("ImageAtomicInc", 11u, 9u, Replacement, 11u, 0u)]
    [InlineData("ImageAtomicDec", 5u, 9u, Replacement, 5u, 4u)]
    [InlineData("ImageAtomicDec", 0u, 9u, Replacement, 0u, 9u)]
    [InlineData("ImageAtomicDec", 11u, 9u, Replacement, 11u, 9u)]
    // CMPSWAP takes the new value from VDATA and the comparator from VDATA+1.
    [InlineData("ImageAtomicCmpswap", Initial, Replacement, Initial, Initial, Replacement)]
    [InlineData("ImageAtomicCmpswap", Initial, Replacement, Operand, Initial, Initial)]
    // The float forms compare the texel's bit pattern as a float: 2.0 in memory
    // against 1.0 and 4.0 operands.
    [InlineData("ImageAtomicFmin", 0x40000000u, 0x3F800000u, 0u, 0x40000000u, 0x3F800000u)]
    [InlineData("ImageAtomicFmin", 0x40000000u, 0x40800000u, 0u, 0x40000000u, 0x40000000u)]
    [InlineData("ImageAtomicFmax", 0x40000000u, 0x40800000u, 0u, 0x40000000u, 0x40800000u)]
    [InlineData("ImageAtomicFmax", 0x40000000u, 0x3F800000u, 0u, 0x40000000u, 0x40000000u)]
    [InlineData("ImageAtomicFcmpswap", 0x40000000u, 0x3F800000u, 0x40000000u, 0x40000000u, 0x3F800000u)]
    [InlineData("ImageAtomicFcmpswap", 0x40000000u, 0x3F800000u, 0x40800000u, 0x40000000u, 0x40000000u)]
    public void AtomicAppliesItsOperationAndReturnsThePreOperationTexel(
        string opcode, uint initial, uint data, uint comparator, uint expectedReturn, uint expectedTexel)
    {
        // Address (1, 0): the second texel, so an atomic that wrote the wrong
        // texel would show up as an untouched target and a changed neighbour.
        Assert.True(
            Gen5ImageFixtures.TryCompile(
                opcode,
                Gen5ImageFixtures.ImageType2D,
                out var shader,
                out var error,
                dmask: 1,
                unifiedFormat: AtomicUintFormat,
                addressValues: [1, 0],
                storeResults: true,
                storeValues: [data, comparator],
                glc: true),
            error);
        Assert.Contains("access::read_write", shader!.Source, StringComparison.Ordinal);
        if (!RequireDevice())
        {
            return;
        }

        uint[] seed = [0xFEEDFACE, initial, 0xFEEDFACE, 0xFEEDFACE];
        var texture = MetalNative.CreateAtomicUintTexture(Width, Height, seed);
        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out error), $"{error}\n{shader.Source}");
        Assert.True(
            MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, new byte[16],
                new byte[PushData.ByteSize], shader.ArgumentLayout!, 1, out var result, out error, texture),
            error);

        Assert.Equal(expectedReturn, BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan(0)));
        uint[] expected = [0xFEEDFACE, expectedTexel, 0xFEEDFACE, 0xFEEDFACE];
        Assert.Equal(expected, MetalNative.ReadAtomicUintTexture(texture, Width, Height));
    }

    // A 64-bit image atomic (DMASK 0x3) on a 32_32_UINT image works on whole texels,
    // VDATA holding the low dword and VDATA+1 the high one: a visibility buffer keeps
    // depth in the high half, so the maximum must compare the halves together.
    [Theory]
    [InlineData("ImageAtomicUmax", 0x00000005_00001234UL, 0x00000006_00000001UL, 0x00000006_00000001UL)]
    [InlineData("ImageAtomicUmax", 0x00000005_00001234UL, 0x00000004_FFFFFFFFUL, 0x00000005_00001234UL)]
    [InlineData("ImageAtomicUmax", 0x00000005_00001234UL, 0x00000005_00001235UL, 0x00000005_00001235UL)]
    [InlineData("ImageAtomicUmin", 0x00000005_00001234UL, 0x00000004_FFFFFFFFUL, 0x00000004_FFFFFFFFUL)]
    [InlineData("ImageAtomicUmin", 0x00000005_00001234UL, 0x00000006_00000001UL, 0x00000005_00001234UL)]
    public void A64BitAtomicComparesWholeTexels(string opcode, ulong initial, ulong data, ulong expectedTexel)
    {
        Assert.True(
            Gen5ImageFixtures.TryCompile(
                opcode,
                Gen5ImageFixtures.ImageType2D,
                out var shader,
                out var error,
                dmask: 3,
                unifiedFormat: 62u,
                addressValues: [1, 0],
                storeResults: true,
                storeValues: [(uint)data, (uint)(data >> 32)],
                glc: false),
            error);
        Assert.Contains("texture2d<ulong, access::read_write>", shader!.Source, StringComparison.Ordinal);
        if (!RequireDevice())
        {
            return;
        }

        const ulong Untouched = 0xFEEDFACE_DEADBEEF;
        ulong[] seed = [Untouched, initial, Untouched, Untouched];
        var texture = MetalNative.CreateAtomicUlongTexture(Width, Height, seed);
        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out error), $"{error}\n{shader.Source}");
        Assert.True(
            MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, new byte[16],
                new byte[PushData.ByteSize], shader.ArgumentLayout!, 1, out _, out error, texture),
            error);

        ulong[] expected = [Untouched, expectedTexel, Untouched, Untouched];
        Assert.Equal(expected, MetalNative.ReadAtomicUlongTexture(texture, Width, Height));
    }

    [Theory]
    [InlineData("ImageAtomicAdd", false)]
    [InlineData("ImageAtomicUmax", true)]
    public void A64BitAtomicMetalCannotRunIsRefused(string opcode, bool glc)
    {
        Assert.False(
            Gen5ImageFixtures.TryCompile(
                opcode,
                Gen5ImageFixtures.ImageType2D,
                out _,
                out var error,
                dmask: 3,
                unifiedFormat: 62u,
                addressValues: [1, 0],
                storeResults: true,
                storeValues: [1, 0],
                glc: glc));
        Assert.Contains("64-bit", error, StringComparison.Ordinal);
    }

    [Fact]
    public void AnAtomicWithoutGlcLeavesVdataAlone()
    {
        // Without GLC the pre-operation value is not returned, so VDATA still
        // holds the operand the program put there.
        Assert.True(
            Gen5ImageFixtures.TryCompile(
                "ImageAtomicAdd",
                Gen5ImageFixtures.ImageType2D,
                out var shader,
                out var error,
                dmask: 1,
                unifiedFormat: AtomicUintFormat,
                addressValues: [1, 0],
                storeResults: true,
                storeValues: [Operand, 0],
                glc: false),
            error);
        if (!RequireDevice())
        {
            return;
        }

        uint[] seed = [0xFEEDFACE, Initial, 0xFEEDFACE, 0xFEEDFACE];
        var texture = MetalNative.CreateAtomicUintTexture(Width, Height, seed);
        Assert.True(MetalNative.TryCompileLibrary(shader!.Source, out var library, out error), error);
        Assert.True(
            MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, new byte[16],
                new byte[PushData.ByteSize], shader.ArgumentLayout!, 1, out var result, out error, texture),
            error);

        Assert.Equal(Operand, BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan(0)));
        uint[] expected = [0xFEEDFACE, 0x00001245, 0xFEEDFACE, 0xFEEDFACE];
        Assert.Equal(expected, MetalNative.ReadAtomicUintTexture(texture, Width, Height));
    }

    [Fact]
    public void AnOutOfRangeAtomicTouchesNoTexel()
    {
        // Metal leaves an atomic on an out-of-range texel undefined, so the
        // emitted bounds check has to drop it.
        Assert.True(
            Gen5ImageFixtures.TryCompile(
                "ImageAtomicAdd",
                Gen5ImageFixtures.ImageType2D,
                out var shader,
                out var error,
                dmask: 1,
                unifiedFormat: AtomicUintFormat,
                addressValues: [64, 0],
                storeResults: true,
                storeValues: [Operand, 0],
                glc: true),
            error);
        if (!RequireDevice())
        {
            return;
        }

        uint[] seed = [0xFEEDFACE, Initial, 0xFEEDFACE, 0xFEEDFACE];
        var texture = MetalNative.CreateAtomicUintTexture(Width, Height, seed);
        Assert.True(MetalNative.TryCompileLibrary(shader!.Source, out var library, out error), error);
        Assert.True(
            MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, new byte[16],
                new byte[PushData.ByteSize], shader.ArgumentLayout!, 1, out _, out error, texture),
            error);

        Assert.Equal(seed, MetalNative.ReadAtomicUintTexture(texture, Width, Height));
    }

    [Fact]
    public void TheUsageMaskTheImageCacheBuildsAcceptsAnAtomicTexture()
    {
        // MetalImageTransfer.UsageOf gives a single-sample non-sRGB image
        // ShaderRead | RenderTarget | PixelFormatView | ShaderWrite, plus
        // ShaderAtomic for R32Uint/R32Sint. Metal has to accept that whole mask
        // together, or every R32 integer image in the cache would fail to
        // create rather than only the atomics failing to compile.
        const nuint CacheUsage = 1 | 4 | 16 | 2 | 32;
        if (!RequireDevice())
        {
            return;
        }

        var texture = MetalNative.CreateAtomicUintTexture(Width, Height, [1, 2, 3, 4], CacheUsage);
        Assert.NotEqual(0, texture);
        Assert.Equal<uint[]>([1, 2, 3, 4], MetalNative.ReadAtomicUintTexture(texture, Width, Height));

        // RG32Uint, the 64-bit atomic texel, gets the same mask.
        var wide = MetalNative.CreateAtomicUlongTexture(Width, Height, [1, 2, 3, ulong.MaxValue], CacheUsage);
        Assert.NotEqual(0, wide);
        Assert.Equal<ulong[]>([1, 2, 3, ulong.MaxValue], MetalNative.ReadAtomicUlongTexture(wide, Width, Height));
    }

    private bool RequireDevice()
    {
        if (!MetalNative.IsAvailable)
        {
            Assert.False(
                OperatingSystem.IsMacOS() && Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") == "1",
                "The required gate cannot run an image atomic without a Metal device.");
            output.WriteLine("No Metal device on this host; the image atomic execution check is skipped.");
            return false;
        }

        return true;
    }
}
