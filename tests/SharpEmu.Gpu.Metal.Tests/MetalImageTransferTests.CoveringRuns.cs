// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Images;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

// The run the cache refreshes or publishes for a dirty range of an image. It covers the
// whole range, widened outwards to the nearest cuts the transfer plan can make - whole
// texel rows of a linear surface, whole block rows of a tiled one - and the run may
// begin wherever such a cut falls, so an alias over its tail or its middle moves its own
// bytes rather than every byte of the image. Planning only: no device.
public sealed partial class MetalImageTransferTests
{
    // 64x128 RGBA8, linear: 0x100 bytes a texel row, 0x8000 in all.
    private static ImageDescription LinearRows() => Color2D(
        GuestPixelFormat.Bits8_8_8_8UNorm, GpuFormat.R8G8B8A8Unorm, 64, 128, 1, 1, GuestTileMode.Linear);

    // 128x512 RGBA8 in 64 KiB tiles: four block rows of 64 KiB each.
    private static ImageDescription FourBlockRows() => Color2D(
        GuestPixelFormat.Bits8_8_8_8UNorm, GpuFormat.R8G8B8A8Unorm, 128, 512, 1, 1, GuestTileMode.Standard64KB);

    [Fact]
    public void ALinearTailOverlapIsCoveredFromTheTexelRowItStartsIn()
    {
        using var transfer = new MetalImageTransfer(RequireDevice() ? Device.Value : CreateHeadlessDevice());
        var description = LinearRows();
        Assert.Equal(0x8000UL, description.Data.Size);

        Assert.True(MetalImageCache.TryCoveringRun(transfer, in description, 0x6010, 0x8000, out var start, out var end));

        Assert.Equal(0x6000UL, start);
        Assert.Equal(0x8000UL, end);
    }

    [Fact]
    public void ALinearMiddleOverlapIsCoveredByTheTexelRowsAroundIt()
    {
        using var transfer = new MetalImageTransfer(RequireDevice() ? Device.Value : CreateHeadlessDevice());
        var description = LinearRows();

        Assert.True(MetalImageCache.TryCoveringRun(transfer, in description, 0x1010, 0x2020, out var start, out var end));

        Assert.Equal(0x1000UL, start);
        Assert.Equal(0x2100UL, end);
    }

    // A tiled run starts at the macro tile row the overlap begins in and stops at the
    // first one past it, so neither the untouched head nor the untouched tail is in it.
    [Fact]
    public void ATiledMiddleOverlapIsCoveredByTheBlockRowsAroundIt()
    {
        using var transfer = new MetalImageTransfer(RequireDevice() ? Device.Value : CreateHeadlessDevice());
        var description = FourBlockRows();
        Assert.Equal(0x40000UL, description.Data.Size);

        Assert.True(MetalImageCache.TryCoveringRun(transfer, in description, 0x10100, 0x2FF00, out var start, out var end));

        Assert.Equal(0x10000UL, start);
        Assert.Equal(0x30000UL, end);
    }

    [Fact]
    public void ATiledTailOverlapIsCoveredFromTheBlockRowItStartsIn()
    {
        using var transfer = new MetalImageTransfer(RequireDevice() ? Device.Value : CreateHeadlessDevice());
        var description = FourBlockRows();

        Assert.True(MetalImageCache.TryCoveringRun(transfer, in description, 0x30100, 0x40000, out var start, out var end));

        Assert.Equal(0x30000UL, start);
        Assert.Equal(0x40000UL, end);
    }

    // A 4 KiB mode cuts at its own tile rows: 96x128 RGBA8 is three 32x32 tiles, 0x3000
    // bytes, a row.
    [Fact]
    public void A4KiBTiledOverlapIsCoveredByTheTileRowsAroundIt()
    {
        using var transfer = new MetalImageTransfer(RequireDevice() ? Device.Value : CreateHeadlessDevice());
        var description = Color2D(
            GuestPixelFormat.Bits8_8_8_8UNorm, GpuFormat.R8G8B8A8Unorm, 96, 128, 1, 1, GuestTileMode.Standard4KB);
        Assert.Equal(0xC000UL, description.Data.Size);

        Assert.True(MetalImageCache.TryCoveringRun(transfer, in description, 0x4010, 0x5000, out var start, out var end));

        Assert.Equal(0x3000UL, start);
        Assert.Equal(0x6000UL, end);
    }

    // Overlaps whose covering run is the whole image are refused, so the caller moves
    // the whole image exactly as it did before: a range inside a single block row that
    // is the whole image, and a range no row boundary separates from either end.
    [Theory]
    [InlineData(256u, 128u, 0x100UL, 0x200UL)]
    [InlineData(128u, 512u, 0x100UL, 0x3FF00UL)]
    public void AnOverlapWhoseRunIsTheWholeImageFallsBack(uint width, uint height, ulong dirtyStart, ulong dirtyEnd)
    {
        using var transfer = new MetalImageTransfer(RequireDevice() ? Device.Value : CreateHeadlessDevice());
        var description = Color2D(
            GuestPixelFormat.Bits8_8_8_8UNorm, GpuFormat.R8G8B8A8Unorm, width, height, 1, 1, GuestTileMode.Standard64KB);

        Assert.False(MetalImageCache.TryCoveringRun(transfer, in description, dirtyStart, dirtyEnd, out _, out _));
    }

    // ---- Depth runs from a layer past the first (SHARPEMU_METAL_V2_DEPTH_LAYER_RUNS) ----------------------------
    // A depth plane is cut a layer at a time: a run over the tail or the middle begins at the layer it is dirty from.
    // With the switch off a depth run begins at the first byte, as it always did.

    [Theory]
    [InlineData(false, 1u)]
    [InlineData(false, 2u)]
    [InlineData(false, 3u)]
    [InlineData(true, 1u)]
    [InlineData(true, 2u)]
    [InlineData(true, 3u)]
    public void ADepthRunBeginsAtTheLayerItIsDirtyFrom(bool stencil, uint layer)
    {
        using var transfer = new MetalImageTransfer(RequireDevice() ? Device.Value : CreateHeadlessDevice());
        var description = Depth(GuestDepthFormat.Z32Float, stencil, 128, 64, 4);
        var slice = description.Data.Size / 4;

        Assert.True(MetalImageCache.TryCoveringRun(
            transfer, in description, layer * slice + 0x10, description.Data.Size, true, out var start, out var end));
        Assert.Equal(layer * slice, start);
        Assert.Equal(description.Data.Size, end);

        Assert.True(MetalImageCache.TryCoveringRun(
            transfer, in description, layer * slice + 0x10, layer * slice + 0x20, true, out start, out end));
        Assert.Equal(layer * slice, start);
        Assert.Equal((layer + 1) * slice, end);

        // Off: the tail run is the whole image, refused; the middle one starts at the first byte.
        Assert.False(MetalImageCache.TryCoveringRun(
            transfer, in description, layer * slice + 0x10, description.Data.Size, false, out _, out _));
        if (layer < 3)
        {
            Assert.True(MetalImageCache.TryCoveringRun(
                transfer, in description, layer * slice + 0x10, layer * slice + 0x20, false, out start, out end));
            Assert.Equal(0UL, start);
            Assert.Equal((layer + 1) * slice, end);
        }
    }

    // On the device: an upload of layers [layer, 4) over a texture holding another image, and a writeback of them over
    // guest bytes holding another image, leave exactly what the whole transfer of the composite leaves - depth and
    // stencil planes both, bit for bit.
    [Theory]
    [InlineData(false, 1u)]
    [InlineData(false, 2u)]
    [InlineData(false, 3u)]
    [InlineData(true, 1u)]
    [InlineData(true, 2u)]
    [InlineData(true, 3u)]
    public void ADepthRunFromALayerMovesExactlyItsLayers(bool stencil, uint layer)
    {
        if (!RequireDevice()) return;
        using var harness = new MetalTransferHarness(Device.Value);
        var description = Depth(GuestDepthFormat.Z32Float, stencil, 128, 64, 4);
        var size = description.Data.Size;
        var slice = size / 4;
        var stencilSize = stencil ? description.Stencil.Size : 0;
        var stencilSlice = stencilSize / 4;
        var first = BuildGuest(description, 91);
        var second = BuildGuest(description, 92);
        var firstStencil = stencil ? TiledStencil(description, 93) : [];
        var secondStencil = stencil ? TiledStencil(description, 94) : [];
        var composite = Splice(first, second, layer * slice);
        var compositeStencil = Splice(firstStencil, secondStencil, layer * stencilSlice);
        var range = new SubresourceRange(0, 1, 0, 4);
        var runSize = size - layer * slice;
        var stencilRunSize = stencilSize - layer * stencilSlice;

        MetalGuestRange Whole(byte[] bytes) => new(harness.Buffer(bytes), 0, (ulong)bytes.Length, 0);
        MetalGuestRange Tail(byte[] bytes, ulong from) => new(harness.Buffer(bytes[(int)from..]), 0, (ulong)bytes.Length - from, 0);

        // Upload: the first image whole, then the second's layers from `layer` as a run.
        var texture = harness.Texture(harness.Transfer.ShapeFor(description));
        harness.Begin();
        Assert.True(harness.Transfer.EncodeGuestToTexture(
            harness, description, range, Whole(first), stencil ? Whole(firstStencil) : default, texture, 0, size));
        harness.Submit();
        harness.Begin();
        Assert.True(harness.Transfer.EncodeGuestToTexture(
            harness, description, range, Tail(second, layer * slice),
            stencil ? Tail(secondStencil, layer * stencilSlice) : default, texture, layer * slice, runSize));
        harness.Submit();
        var (depthOut, stencilOut) = WriteBackWhole(harness, description, range, texture);
        Assert.Equal(composite, depthOut);
        Assert.Equal(compositeStencil, stencilOut);

        // Writeback: the composite whole into a texture, then its layers from `layer` as a run over the first image.
        var compositeTexture = harness.Texture(harness.Transfer.ShapeFor(description));
        harness.Begin();
        Assert.True(harness.Transfer.EncodeGuestToTexture(
            harness, description, range, Whole(composite), stencil ? Whole(compositeStencil) : default,
            compositeTexture, 0, size));
        harness.Submit();
        var target = harness.Buffer(first);
        var stencilTarget = stencil ? harness.Buffer(firstStencil) : 0;
        harness.Begin();
        Assert.True(harness.Transfer.EncodeTextureToGuest(
            harness, description, range, compositeTexture, new MetalGuestRange(target, layer * slice, runSize, 0),
            stencil ? new MetalGuestRange(stencilTarget, layer * stencilSlice, stencilRunSize, 0) : default,
            layer * slice, runSize));
        harness.Submit();
        Assert.Equal(composite, harness.Read(target, 0, size));
        if (stencil)
        {
            Assert.Equal(compositeStencil, harness.Read(stencilTarget, 0, stencilSize));
        }
    }

    private static (byte[] Depth, byte[] Stencil) WriteBackWhole(
        MetalTransferHarness harness, in ImageDescription description, SubresourceRange range, nint texture)
    {
        var target = harness.Buffer(description.Data.Size);
        var stencilTarget = description.HasStencil ? harness.Buffer(description.Stencil.Size) : 0;
        harness.Begin();
        Assert.True(harness.Transfer.EncodeTextureToGuest(
            harness, description, range, texture, new MetalGuestRange(target, 0, description.Data.Size, 0),
            stencilTarget != 0 ? new MetalGuestRange(stencilTarget, 0, description.Stencil.Size, 0) : default,
            0, description.Data.Size));
        harness.Submit();
        return (harness.Read(target, 0, description.Data.Size),
            stencilTarget != 0 ? harness.Read(stencilTarget, 0, description.Stencil.Size) : []);
    }

    private static byte[] Splice(byte[] head, byte[] tail, ulong at)
    {
        var bytes = (byte[])tail.Clone();
        head.AsSpan(0, (int)at).CopyTo(bytes);
        return bytes;
    }

    // A stencil plane with every byte its tiled transfer carries set and the rest zero (as RoundTrip builds it).
    private static byte[] TiledStencil(in ImageDescription description, uint seed)
    {
        var linear = MetalTileOracle.Pattern((int)description.Stencil.Size, seed);
        var tiled = new byte[description.Stencil.Size];
        var pitch = TileGeometry.DepthPitch(description.Extent.Width, 1);
        var slice = description.Stencil.Size / description.Resources.Layers;
        for (uint layer = 0; layer < description.Resources.Layers; layer++)
        {
            MetalTileOracle.Transfer(linear, tiled, new MetalTileTransfer
            {
                Kind = TileBlockKind.Depth64KB,
                BytesPerElement = 1,
                LinearOffset = slice * layer,
                LinearSize = slice,
                TiledOffset = slice * layer,
                TiledSize = slice,
                Width = description.Extent.Width,
                Height = description.Extent.Height,
                Depth = 1,
                Pitch = pitch,
                SurfaceZ = layer,
            }, toTiled: true);
        }

        return tiled;
    }
}
