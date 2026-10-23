// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.Pipelines;

public sealed class AffineVertexFetchResolverTests
{
    private static BufferDescriptorWords Descriptor(uint stride = 16, uint format = 77, uint records = 8) =>
        new(0x200000, stride << 16, records, format << 12 | 0xFAC | 2u << 28);
    private static readonly VertexFetchBounds Direct = new(2, 4, 0, 4, true, true);
    private static AffineVertexFetchLoad Load(VertexFetchIndex kind = VertexFetchIndex.Vertex, uint divisor = 1, uint bias = 0) =>
        new(0xA0, 4, new(kind, divisor, bias), 0);

    [Fact]
    public void ProvedDirectAndConstantFetchesHaveExactRanges()
    {
        Assert.True(AffineVertexFetchResolver.TryResolve(Load(), Descriptor(), Direct, out var buffer, out _));
        Assert.Equal(16u, buffer.Stride);
        Assert.Equal(128ul, buffer.Size);
        Assert.True(AffineVertexFetchResolver.TryResolve(Load(bias: 1), Descriptor(), Direct, out buffer, out _));
        Assert.Equal(0x200010ul, buffer.Address);
        Assert.Equal(112ul, buffer.Size);
        Assert.True(AffineVertexFetchResolver.TryResolve(Load(), Descriptor(stride: 0, records: 64), default, out buffer, out _));
        Assert.Equal(0u, buffer.Stride);
        Assert.Equal(16ul, buffer.Size);
    }

    [Fact]
    public void UnknownOutOfRangeAndOverflowingIndicesKeepShaderFetch()
    {
        Assert.False(AffineVertexFetchResolver.TryResolve(Load(), Descriptor(), default, out _, out _));
        Assert.False(AffineVertexFetchResolver.TryResolve(Load(), Descriptor(records: 4), Direct, out _, out _));
        Assert.False(AffineVertexFetchResolver.TryResolve(Load(bias: uint.MaxValue), Descriptor(), Direct, out _, out _));
        Assert.False(AffineVertexFetchResolver.TryResolve(Load(), Descriptor(), Direct with { FirstVertex = uint.MaxValue }, out _, out _));
        Assert.False(AffineVertexFetchResolver.TryResolve(Load(), Descriptor(stride: 0, records: 1), default, out _, out _));
    }

    [Fact]
    public void MetalInstanceSteppingMustMatchTheFullGuestInstanceId()
    {
        Assert.True(AffineVertexFetchResolver.TryResolve(Load(VertexFetchIndex.Instance, 2), Descriptor(), Direct, out var buffer, out _));
        Assert.True(buffer.PerInstance);
        Assert.Equal(2u, buffer.Divisor);
        Assert.False(AffineVertexFetchResolver.TryResolve(Load(VertexFetchIndex.Instance, 2), Descriptor(), Direct with { FirstInstance = 3 }, out _, out _));
        Assert.True(AffineVertexFetchResolver.TryResolve(Load(VertexFetchIndex.Instance), Descriptor(), Direct with { FirstInstance = 3 }, out _, out _));
    }

    [Fact]
    public void RawLoadsIgnoreTheDescriptorFormatAndSwizzle()
    {
        var descriptor = Descriptor(format: 127) with { Word3 = 127u << 12 | 2u << 28 };
        Assert.True(AffineVertexFetchResolver.TryResolve(Load() with { Raw = true }, descriptor, Direct, out _, out var attribute));
        Assert.Equal(75u, attribute.Descriptor.Format);
        Assert.Equal(0xFACu, attribute.Descriptor.DestinationSelectXYZW);
    }

    [Fact]
    public void UnalignedAndSwizzledLayoutsKeepShaderFetch()
    {
        Assert.False(AffineVertexFetchResolver.TryResolve(Load(), Descriptor() with { Word0 = 0x200001 }, Direct, out _, out _));
        Assert.False(AffineVertexFetchResolver.TryResolve(Load(), Descriptor() with { Word1 = 16u << 16 | 1u << 31 }, Direct, out _, out _));
        Assert.False(AffineVertexFetchResolver.TryResolve(Load(), Descriptor(stride: 17), Direct, out _, out _));
    }

    [Theory]
    [InlineData(43u)] // reversed packed float
    [InlineData(54u)] // packed unsigned integer
    [InlineData(55u)] // packed signed integer
    [InlineData(3u)]  // scaled integer
    [InlineData(127u)]
    public void FormatsWithoutAnExactNativeEquivalentKeepShaderFetch(uint format) =>
        Assert.False(AffineVertexFetchResolver.TryResolve(Load(), Descriptor(format: format), Direct, out _, out _));
}
