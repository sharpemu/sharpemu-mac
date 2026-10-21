// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>Actual texture contents through guest MIMG, argument-buffer binding and
/// GPU readback. Compiler acceptance alone cannot establish coordinate correctness.</summary>
public sealed class MetalImageExecutionTests(ITestOutputHelper output)
{
    public static IEnumerable<object[]> Cases => new List<object[]>
    {
        new object[] { "ImageLoadMip", 8u, 2u, 4u, 1u, 1u, 1u, new uint[] { 1, 1 }, 1u, 0u, 1u, 0u, 0u },
        new object[] { "ImageSampleL", 8u, 2u, 4u, 1u, 1u, 1u, new uint[] { F(.75f), F(1) }, 1u, 0u, 1u, 0u, 0u },
        new object[] { "ImageLoadMip", 12u, 3u, 4u, 1u, 3u, 1u, new uint[] { 1, 2, 1 }, 1u, 2u, 1u, 0u, 0u },
        new object[] { "ImageSampleL", 12u, 3u, 4u, 1u, 3u, 1u, new uint[] { F(.75f), F(1.5f), F(1) }, 1u, 2u, 1u, 0u, 0u },
        new object[] { "ImageLoadMip", 9u, 2u, 4u, 4u, 1u, 1u, new uint[] { 1, 0, 1 }, 1u, 0u, 1u, 0u, 0u },
        new object[] { "ImageSampleL", 9u, 2u, 4u, 4u, 1u, 1u, new uint[] { F(.75f), F(.25f), F(1) }, 1u, 0u, 1u, 0u, 0u },
        new object[] { "ImageLoadMip", 13u, 3u, 4u, 4u, 3u, 1u, new uint[] { 1, 0, 2, 1 }, 1u, 2u, 1u, 0u, 0u },
        new object[] { "ImageSampleL", 13u, 3u, 4u, 4u, 3u, 1u, new uint[] { F(.75f), F(.25f), F(1.5f), F(1) }, 1u, 2u, 1u, 0u, 0u },
        new object[] { "ImageLoadMip", 10u, 7u, 4u, 4u, 1u, 4u, new uint[] { 1, 0, 1, 1 }, 1u, 0u, 1u, 0u, 1u },
        new object[] { "ImageSampleL", 10u, 7u, 4u, 4u, 1u, 4u, new uint[] { F(.75f), F(.25f), F(.75f), F(1) }, 1u, 0u, 1u, 0u, 1u },
        new object[] { "ImageSampleLz", 11u, 3u, 4u, 4u, 12u, 1u, new uint[] { F(1.625f), F(1.375f), F(9) }, 0u, 7u, 2u, 1u, 0u },
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public void ImageReadsMatchDistinctMipLayerAndVolumeTexels(string opcode, uint guestType,
        uint metalType, uint width, uint height, uint layers, uint depth, uint[] coordinates,
        uint expectedMip, uint expectedLayer, uint expectedX, uint expectedY, uint expectedZ)
    {
        Assert.True(Gen5ImageFixtures.TryCompile(opcode, guestType, out var shader, out var error,
            addressValues: coordinates, storeResults: true), error);
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Texture execution requires the exclusive Metal device gate.");
            return;
        }
        Assert.True(MetalNative.IsAvailable, "The required gate needs a Metal device.");
        var mipBytes = Enumerable.Range(0, 2).Select(mip => Texels(width, height, layers, depth, (uint)mip)).ToArray();
        var texture = MetalNative.CreateFloatTexture(metalType, width, height, layers, depth, mipBytes);
        var sampler = MetalNative.CreatePointSampler();
        Assert.True(MetalNative.TryCompileLibrary(shader!.Source, out var library, out error), error + shader.Source);
        Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, new byte[16],
            new byte[PushData.ByteSize], shader.ArgumentLayout!, 1, out var result, out error, texture, sampler), error);
        var expected = Value(expectedMip, expectedLayer, expectedX, expectedY, expectedZ);
        for (var component = 0; component < 4; component++)
            Assert.Equal(F(expected + component), BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan(component * 4)));
    }

    [Theory]
    [InlineData(8u, 2u, 1u, 1u, 1u)]
    [InlineData(12u, 3u, 1u, 3u, 1u)]
    [InlineData(9u, 2u, 4u, 1u, 1u)]
    [InlineData(13u, 3u, 4u, 3u, 1u)]
    [InlineData(10u, 7u, 4u, 1u, 4u)]
    public void ImageStoreChangesOnlyTheSelectedTexel(uint guestType, uint metalType,
        uint height, uint layers, uint depth)
    {
        uint[] coords = guestType switch
        {
            8 => [1], 12 => [1, 2], 9 => [1, 0], 13 => [1, 0, 2], _ => [1, 0, 1],
        };
        uint[] values = [F(101.5f), F(-2.25f), F(3.5f), F(4.75f)];
        Assert.True(Gen5ImageFixtures.TryCompile("ImageStore", guestType, out var shader, out var error,
            addressValues: coords, storeValues: values), error);
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Texture execution requires the exclusive Metal device gate.");
            return;
        }
        Assert.True(MetalNative.IsAvailable);
        var initial = Texels(4, height, layers, depth, 0);
        var expected = (byte[])initial.Clone();
        var selectedLayer = layers > 1 ? 2u : 0u;
        var selectedZ = depth > 1 ? 1u : 0u;
        var offset = (selectedLayer * 4 * height * depth + selectedZ * 4 * height + 1) * 16;
        for (var component = 0; component < 4; component++)
            BinaryPrimitives.WriteUInt32LittleEndian(expected.AsSpan((int)offset + component * 4), values[component]);
        var texture = MetalNative.CreateFloatTexture(metalType, 4, height, layers, depth, [initial]);
        Assert.True(MetalNative.TryCompileLibrary(shader!.Source, out var library, out error), error + shader.Source);
        Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, new byte[16],
            new byte[PushData.ByteSize], shader.ArgumentLayout!, 1, out _, out error, texture), error);
        Assert.Equal(expected, MetalNative.ReadFloatTexture(texture, 4, height, layers, depth));
    }

    [Theory]
    [InlineData(9u, 2u, 4u, 1u, 1u, 4u, 0u, 0u)]
    [InlineData(9u, 2u, 4u, 1u, 1u, 0xFFFFFFFFu, 0u, 0u)]
    [InlineData(13u, 3u, 4u, 3u, 1u, 0u, 0u, 3u)]
    [InlineData(10u, 7u, 4u, 1u, 4u, 0u, 0u, 4u)]
    public void OutOfBoundsLoadsReturnZeroRatherThanTheEdgeTexel(uint guestType, uint metalType,
        uint height, uint layers, uint depth, uint x, uint y, uint extra)
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1") return;
        Assert.True(MetalNative.IsAvailable);
        uint[] coords = guestType == 9 ? [x, y] : [x, y, extra];
        Assert.True(Gen5ImageFixtures.TryCompile("ImageLoad", guestType, out var shader, out var error,
            addressValues: coords, storeResults: true), error);
        var texture = MetalNative.CreateFloatTexture(metalType, 4, height, layers, depth, [Texels(4, height, layers, depth, 0)]);
        Assert.True(MetalNative.TryCompileLibrary(shader!.Source, out var library, out error), error + shader.Source);
        Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, new byte[16],
            new byte[PushData.ByteSize], shader.ArgumentLayout!, 1, out var result, out error, texture), error);
        Assert.Equal(new byte[16], result);
    }

    // A mip past the last one is outside the image like a coordinate past the extent: the
    // guest reads zero there, where Metal leaves the read undefined (UE5 marches volumes off
    // their last mip).
    [Theory]
    [InlineData(9u, 2u, 1u, 2u)]
    [InlineData(9u, 2u, 1u, 9u)]
    [InlineData(13u, 3u, 3u, 2u)]
    public void AMipPastTheLastReadsZero(uint guestType, uint metalType, uint layers, uint mip)
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1") return;
        Assert.True(MetalNative.IsAvailable);
        uint[] coords = layers == 1 ? [1, 0, mip] : [1, 0, 1, mip];
        Assert.True(Gen5ImageFixtures.TryCompile("ImageLoadMip", guestType, out var shader, out var error,
            addressValues: coords, storeResults: true), error);
        var mipBytes = Enumerable.Range(0, 2).Select(level => Texels(4, 4, layers, 1, (uint)level)).ToArray();
        var texture = MetalNative.CreateFloatTexture(metalType, 4, 4, layers, 1, mipBytes);
        Assert.True(MetalNative.TryCompileLibrary(shader!.Source, out var library, out error), error + shader.Source);
        Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, new byte[16],
            new byte[PushData.ByteSize], shader.ArgumentLayout!, 1, out var result, out error, texture), error);
        Assert.Equal(new byte[16], result);
    }

    [Theory]
    [InlineData(9u, 2u, 1u, 0u)]
    [InlineData(13u, 3u, 3u, 2u)]
    public void GatherUsesTheSpecifiedCounterClockwiseTexelOrder(uint guestType, uint metalType,
        uint layers, uint selectedLayer)
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1") return;
        Assert.True(MetalNative.IsAvailable);
        uint[] coords = layers == 1 ? [F(.5f), F(.5f)] : [F(.5f), F(.5f), F(selectedLayer)];
        Assert.True(Gen5ImageFixtures.TryCompile("ImageGather4", guestType, out var shader, out var error,
            addressValues: coords, storeResults: true, dmask: 1), error);
        var texture = MetalNative.CreateFloatTexture(metalType, 4, 4, layers, 1, [Texels(4, 4, layers, 1, 0)]);
        Assert.True(MetalNative.TryCompileLibrary(shader!.Source, out var library, out error), error + shader.Source);
        Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, new byte[16],
            new byte[PushData.ByteSize], shader.ArgumentLayout!, 1, out var result, out error,
            texture, MetalNative.CreatePointSampler()), error);
        // MSL section6.13: (-,+), (+,+), (+,-), (-,-) half-texel deltas.
        uint[] xs = [1, 2, 2, 1], ys = [2, 2, 1, 1];
        for (var component = 0; component < 4; component++)
            Assert.Equal(F(Value(0, selectedLayer, xs[component], ys[component], 0)),
                BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan(component * 4)));
    }

    private static uint F(float value) => BitConverter.SingleToUInt32Bits(value);
    private static float Value(uint mip, uint layer, uint x, uint y, uint z) =>
        mip * 10000 + layer * 1000 + z * 100 + y * 10 + x;

    private static byte[] Texels(uint width, uint height, uint layers, uint depth, uint mip)
    {
        var w = Math.Max(width >> (int)mip, 1u);
        var h = Math.Max(height >> (int)mip, 1u);
        var d = Math.Max(depth >> (int)mip, 1u);
        var bytes = new byte[w * h * d * layers * 16];
        var offset = 0;
        for (uint layer = 0; layer < layers; layer++)
        for (uint z = 0; z < d; z++)
        for (uint y = 0; y < h; y++)
        for (uint x = 0; x < w; x++)
        for (uint component = 0; component < 4; component++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(offset), F(Value(mip, layer, x, y, z) + component));
            offset += 4;
        }
        return bytes;
    }
}
