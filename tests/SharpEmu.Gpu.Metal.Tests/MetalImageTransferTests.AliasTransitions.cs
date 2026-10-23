// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Numerics;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Images;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

public sealed partial class MetalImageTransferTests
{
    [Fact]
    public void AliasInverseReconstructsEveryTexelOfEveryDenseTwoDimensionalTile()
    {
        var cases = 0;
        foreach (var kind in Enum.GetValues<TileBlockKind>())
        foreach (uint bytes in new uint[] { 4, 8, 16 })
        {
            if (!TileGeometry.TryGetBlockLayout(kind, bytes, out var block) || block.BlockDepth != 1) continue;
            var masks = MetalTilerShaders.AliasInverseMasks(block);
            for (uint y = 0; y < block.BlockHeight; y++)
            for (uint x = 0; x < block.BlockWidth; x++)
            {
                Assert.True(TileGeometry.TryGetBlockOffset(block, x, y, 0, out var offset));
                uint actualX = 0, actualY = 0;
                for (var bit = 0; bit < masks.X.Length; bit++)
                    actualX |= (uint)(BitOperations.PopCount(offset & masks.X[bit]) & 1) << bit;
                for (var bit = 0; bit < masks.Y.Length; bit++)
                    actualY |= (uint)(BitOperations.PopCount(offset & masks.Y[bit]) & 1) << bit;
                Assert.Equal((x, y), (actualX, actualY));
            }
            cases++;
        }
        Assert.True(cases >= 18);
        output.WriteLine($"Exhaustive inverse tile cases: {cases}");
    }

    [Fact]
    public void OverwriteCoverageRequiresFullDestinationTilesOrTheSameTexelMapping()
    {
        if (!RequireDevice()) return;
        using var harness = new MetalTransferHarness(Device.Value);
        var target = AliasDescription(GuestTileMode.RenderTarget, 1920, 1080, 8);
        var small = AliasDescription(GuestTileMode.RenderTarget, 64, 64, 8);
        Assert.True(harness.Transfer.CanDiscardAliasForOverwrite(target, small));
        small.Data = small.Data with { Address = target.Data.Address + target.Data.Size - small.Data.Size };
        Assert.False(harness.Transfer.CanDiscardAliasForOverwrite(target, small));
        var same = target;
        same.PixelFormat = GpuFormat.R32G32Uint;
        Assert.True(harness.Transfer.CanDiscardAliasForOverwrite(target, same));
        same.Resources = new SubresourceCount(2, 1);
        Assert.False(harness.Transfer.CanDiscardAliasForOverwrite(target, same));
    }

    [Theory]
    [InlineData(GuestTileMode.Linear, GuestTileMode.Linear, 4u, 4u, false)]
    [InlineData(GuestTileMode.Standard64KB, GuestTileMode.Standard64KB, 4u, 4u, false)]
    [InlineData(GuestTileMode.Standard64KB, GuestTileMode.RenderTarget, 4u, 4u, false)]
    [InlineData(GuestTileMode.RenderTarget, GuestTileMode.Standard64KB, 8u, 8u, false)]
    [InlineData(GuestTileMode.RenderTarget, GuestTileMode.Linear, 16u, 4u, true)]
    [InlineData(GuestTileMode.Linear, GuestTileMode.RenderTarget, 4u, 16u, true)]
    [InlineData(GuestTileMode.Standard64KB, GuestTileMode.Depth, 4u, 8u, true)]
    [InlineData(GuestTileMode.Depth, GuestTileMode.RenderTarget, 8u, 4u, true)]
    [InlineData(GuestTileMode.Standard4KB, GuestTileMode.Standard64KB, 4u, 4u, true)]
    public void FusedFormatSizeAndTilingAliasesMatchTheGuestRoundTrip(
        GuestTileMode sourceTile, GuestTileMode targetTile, uint sourceBytes, uint targetBytes, bool reshape)
    {
        if (!RequireDevice()) return;
        using var harness = new MetalTransferHarness(Device.Value);
        var source = AliasDescription(sourceTile, 512, 512, sourceBytes);
        var width = reshape ? 1024u : 512u;
        var target = AliasDescription(targetTile, width, 512u * 512 * sourceBytes / (width * targetBytes), targetBytes);
        // Different channel interpretations with exactly the same dword bits.
        if (sourceBytes == targetBytes) target.PixelFormat = targetBytes switch
        {
            4 => GpuFormat.R32Uint,
            8 => GpuFormat.R32G32Uint,
            _ => GpuFormat.R32G32B32A32Uint,
        };
        Assert.Equal(source.Data.Size, target.Data.Size);
        Assert.True(harness.Transfer.HasDenseAliasCoverage(source));
        Assert.True(harness.Transfer.HasDenseAliasCoverage(target));
        var guest = BuildGuest(source, 103);
        var sourceTexture = harness.Texture(harness.Transfer.ShapeFor(source));
        var actualTexture = harness.Texture(harness.Transfer.ShapeFor(target));
        var expectedTexture = harness.Texture(harness.Transfer.ShapeFor(target));
        var input = harness.Buffer(guest);
        var published = harness.Buffer(source.Data.Size);
        harness.Begin();
        Assert.True(harness.Transfer.EncodeGuestToTexture(harness, source, SubresourceRange.First,
            new MetalGuestRange(input, 0, source.Data.Size, 0), default, sourceTexture, 0, source.Data.Size));
        harness.Submit();
        var stagingBefore = harness.Tiler.StagingBytes;
        harness.Begin();
        Assert.True(harness.Transfer.TryEncodeFusedAlias(harness, source, sourceTexture,
            target, actualTexture, out var refusal), refusal);
        harness.Submit();
        Assert.Equal(stagingBefore, harness.Tiler.StagingBytes);
        harness.Begin();
        Assert.True(harness.Transfer.EncodeTextureToGuest(harness, source, SubresourceRange.First, sourceTexture,
            new MetalGuestRange(published, 0, source.Data.Size, 0), default, 0, source.Data.Size));
        harness.Submit();
        harness.Begin();
        Assert.True(harness.Transfer.EncodeGuestToTexture(harness, target, SubresourceRange.First,
            new MetalGuestRange(published, 0, target.Data.Size, 0), default, expectedTexture, 0, target.Data.Size));
        harness.Submit();
        Assert.Equal(ReadTexels(harness, target, expectedTexture), ReadTexels(harness, target, actualTexture));
        Assert.Equal(guest, WriteBack(harness, target, actualTexture));
        output.WriteLine($"{sourceTile}/{sourceBytes} -> {targetTile}/{targetBytes}, {width}x{target.Extent.Height}: {guest.Length} bytes exact, no staging");
    }

    [Fact]
    public void AFusedColorArrayAliasPreservesEverySlice()
    {
        if (!RequireDevice()) return;
        using var harness = new MetalTransferHarness(Device.Value);
        var source = Color2D(GuestPixelFormat.Bits16_16_16_16Float, GpuFormat.R16G16B16A16Sfloat,
            512, 128, 1, 3, GuestTileMode.Standard64KB);
        var target = source;
        target.PixelFormat = GpuFormat.R32G32Uint;
        var bytes = BuildGuest(source, 119);
        var from = harness.Texture(harness.Transfer.ShapeFor(source));
        var to = harness.Texture(harness.Transfer.ShapeFor(target));
        var range = new SubresourceRange(0, 1, 0, 3);
        harness.Begin();
        harness.Transfer.EncodeGuestToTexture(harness, source, range,
            new MetalGuestRange(harness.Buffer(bytes), 0, source.Data.Size, 0), default, from, 0, source.Data.Size);
        harness.Submit();
        harness.Begin();
        Assert.True(harness.Transfer.TryEncodeFusedAlias(harness, source, from, target, to, out var refusal), refusal);
        harness.Submit();
        Assert.Equal(ReadTexels(harness, source, from), ReadTexels(harness, target, to));
        Assert.Equal(bytes, WriteBack(harness, target, to));
    }

    [Fact]
    public void MatchingCompressibleColorTargetsUseABlitWithoutUsagePromotion()
    {
        if (!RequireDevice()) return;
        using var harness = new MetalTransferHarness(Device.Value);
        var description = AliasDescription(GuestTileMode.RenderTarget, 512, 512, 8);
        var shape = harness.Transfer.ShapeFor(description) with
        { Usage = (ulong)(MtlTextureUsage.ShaderRead | MtlTextureUsage.RenderTarget) };
        var source = harness.Texture(shape);
        var target = harness.Texture(shape);
        var bytes = BuildGuest(description, 117);
        harness.Begin();
        harness.Transfer.EncodeGuestToTexture(harness, description, SubresourceRange.First,
            new MetalGuestRange(harness.Buffer(bytes), 0, description.Data.Size, 0), default, source, 0, description.Data.Size);
        harness.Submit();
        var staging = harness.Tiler.StagingBytes;
        harness.Begin();
        Assert.True(harness.Transfer.TryEncodeFusedAlias(harness, description, source, description, target, out var refusal), refusal);
        harness.Submit();
        Assert.Equal(staging, harness.Tiler.StagingBytes);
        Assert.Equal(bytes, WriteBack(harness, description, target));
    }

    [Fact]
    public void PaddedAliasesRefuseFusionAndTheirRoundTripPreservesPixels()
    {
        if (!RequireDevice()) return;
        using var harness = new MetalTransferHarness(Device.Value);
        var source = Color2D(GuestPixelFormat.Bits8_8_8_8UNorm, GpuFormat.R8G8B8A8Unorm,
            97, 33, 1, 1, GuestTileMode.Standard64KB);
        var target = source;
        target.PixelFormat = GpuFormat.R8G8B8A8Uint;
        var sourceTexture = harness.Texture(harness.Transfer.ShapeFor(source));
        var targetShape = harness.Transfer.ShapeFor(target) with
        { Usage = (ulong)(MtlTextureUsage.ShaderRead | MtlTextureUsage.RenderTarget) };
        var targetTexture = harness.Texture(targetShape);
        var bytes = BuildGuest(source, 123);
        harness.Begin();
        harness.Transfer.EncodeGuestToTexture(harness, source, SubresourceRange.First,
            new MetalGuestRange(harness.Buffer(bytes), 0, source.Data.Size, 0), default, sourceTexture, 0, source.Data.Size);
        harness.Submit();
        var staging = harness.Tiler.StagingBytes;
        harness.Begin();
        Assert.False(harness.Transfer.TryEncodeFusedAlias(harness, source, sourceTexture, target, targetTexture, out var refusal));
        Assert.Equal("padding-or-subresources", refusal);
        Assert.Equal(staging, harness.Tiler.StagingBytes);
        harness.Submit();
        var published = WriteBack(harness, source, sourceTexture);
        harness.Begin();
        harness.Transfer.EncodeGuestToTexture(harness, target, SubresourceRange.First,
            new MetalGuestRange(harness.Buffer(published), 0, target.Data.Size, 0), default, targetTexture, 0, target.Data.Size);
        harness.Submit();
        Assert.Equal(ReadTexels(harness, source, sourceTexture), ReadTexels(harness, target, targetTexture));
    }

    private static ImageDescription AliasDescription(GuestTileMode tile, uint width, uint height, uint bytes) =>
        Color2D(bytes switch
        {
            4 => GuestPixelFormat.Bits8_8_8_8UNorm,
            8 => GuestPixelFormat.Bits16_16_16_16Float,
            _ => GuestPixelFormat.Bits32_32_32_32Float,
        }, bytes switch
        {
            4 => GpuFormat.R8G8B8A8Unorm,
            8 => GpuFormat.R16G16B16A16Sfloat,
            _ => GpuFormat.R32G32B32A32Sfloat,
        }, width, height, 1, 1, tile);
}
