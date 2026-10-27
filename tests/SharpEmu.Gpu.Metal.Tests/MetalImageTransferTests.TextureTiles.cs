// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Images;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

// A tiled colour transfer of whole 4, 8 or 16 byte texels moves between the guest
// surface and the texture in one pass, with no staging buffer and no blit. A round trip
// alone cannot tell a right pass from two mirrored wrong ones, so each direction is
// checked against the CPU oracle on its own: the upload against the texels a blit reads
// back out of the texture, the writeback against the guest bytes the oracle tiles.
public sealed partial class MetalImageTransferTests
{
    public static TheoryData<string> TextureTileCases => new()
    {
        "rgba8 render target",
        "rgba16f render target 2 layers",
        "r11g11b10 render target",
        "rg32 render target",
        "rgba32f 64kb 2 mips",
        "rgba16f 4kb 3 layers",
        "rgba8 64kb full chain",
        "rgba16f 4kb full chain 2 layers",
        "rg32 64kb full chain",
        "rgba32f 4kb full chain",
    };

    // Elements under a dword go one pass on the way in only: the writeback merges
    // neighbouring elements into a guest dword, which the staged path's kernel does.
    public static TheoryData<string> SubDwordTextureTileCases => new()
    {
        "r8 64kb full chain",
        "rg8 4kb full chain 2 layers",
        "r16f 64kb 3 mips into the tail",
        "r8 render target",
        "r16 render target",
    };

    private static ImageDescription TextureTileCase(string name) => name switch
    {
        "rgba8 render target" => Color2D(GuestPixelFormat.Bits8_8_8_8UNorm, GpuFormat.R8G8B8A8Unorm,
            192, 96, 1, 1, GuestTileMode.RenderTarget),
        "rgba16f render target 2 layers" => Color2D(GuestPixelFormat.Bits16_16_16_16Float, GpuFormat.R16G16B16A16Sfloat,
            160, 72, 1, 2, GuestTileMode.RenderTarget),
        "r11g11b10 render target" => Color2D(GuestPixelFormat.Bits11_11_10Float, GpuFormat.B10G11R11UfloatPack32,
            200, 120, 1, 1, GuestTileMode.RenderTarget),
        "rg32 render target" => Color2D(GuestPixelFormat.Bits32_32UInt, GpuFormat.R32G32Uint,
            96, 64, 1, 1, GuestTileMode.RenderTarget),
        "rgba32f 64kb 2 mips" => Color2D(GuestPixelFormat.Bits32_32_32_32Float, GpuFormat.R32G32B32A32Sfloat,
            256, 256, 2, 1, GuestTileMode.Standard64KB),
        "rgba16f 4kb 3 layers" => Color2D(GuestPixelFormat.Bits16_16_16_16Float, GpuFormat.R16G16B16A16Sfloat,
            96, 48, 1, 3, GuestTileMode.Standard4KB),
        "rgba8 64kb full chain" => Color2D(GuestPixelFormat.Bits8_8_8_8UNorm, GpuFormat.R8G8B8A8Unorm,
            256, 256, 9, 1, GuestTileMode.Standard64KB),
        "rgba16f 4kb full chain 2 layers" => Color2D(GuestPixelFormat.Bits16_16_16_16Float, GpuFormat.R16G16B16A16Sfloat,
            128, 64, 8, 2, GuestTileMode.Standard4KB),
        "rg32 64kb full chain" => Color2D(GuestPixelFormat.Bits32_32UInt, GpuFormat.R32G32Uint,
            200, 120, 8, 1, GuestTileMode.Standard64KB),
        "rgba32f 4kb full chain" => Color2D(GuestPixelFormat.Bits32_32_32_32Float, GpuFormat.R32G32B32A32Sfloat,
            64, 64, 7, 1, GuestTileMode.Standard4KB),
        "r8 64kb full chain" => Color2D(GuestPixelFormat.Bits8UNorm, GpuFormat.R8Unorm,
            512, 256, 10, 1, GuestTileMode.Standard64KB),
        "rg8 4kb full chain 2 layers" => Color2D(GuestPixelFormat.Bits8_8UNorm, GpuFormat.R8G8Unorm,
            96, 80, 7, 2, GuestTileMode.Standard4KB),
        "r16f 64kb 3 mips into the tail" => Color2D(GuestPixelFormat.Bits16Float, GpuFormat.R16Sfloat,
            300, 200, 3, 1, GuestTileMode.Standard64KB),
        "r8 render target" => Color2D(GuestPixelFormat.Bits8UNorm, GpuFormat.R8Unorm,
            200, 120, 1, 1, GuestTileMode.RenderTarget),
        "r16 render target" => Color2D(GuestPixelFormat.Bits16UNorm, GpuFormat.R16Unorm,
            160, 72, 1, 1, GuestTileMode.RenderTarget),
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    [Theory]
    [MemberData(nameof(TextureTileCases))]
    [MemberData(nameof(SubDwordTextureTileCases))]
    public void ATiledColorUploadWritesTheTextureInOnePassExactlyAsTheOracleDetiles(string name)
    {
        if (!RequireDevice()) return;
        using var harness = new MetalTransferHarness(Device.Value);
        var description = TextureTileCase(name);
        var guest = BuildGuest(description, 91, out var linear);
        var (_, copies, tiles, element) = BuildLayout(description);
        // A full chain (or a small enough level) reaches the mip tail, which the one-pass kernel places itself.
        Assert.Equal(name.Contains("full chain") || name.Contains("tail"), tiles.Exists(tile => tile.Tail));
        var texture = harness.Texture(harness.Transfer.ShapeFor(description));
        var range = new SubresourceRange(0, description.Resources.Levels, 0, description.Resources.Layers);

        harness.Begin();
        harness.Transfer.EncodeGuestToTexture(harness, description, range,
            new MetalGuestRange(harness.Buffer(guest), 0, description.Data.Size, 0), default, texture, 0, description.Data.Size);
        harness.Submit();

        // One pass: nothing was staged on the way in.
        Assert.Equal(0u, harness.Tiler.StagingBufferCount);
        var readback = harness.Buffer((ulong)linear.Length);
        harness.Begin();
        var blit = harness.Tiler.BeginBlit(harness);
        foreach (var copy in copies)
        {
            var rowBytes = RowBytesOf(copy, element);
            MetalNative.SendCopyTextureToBufferOptions(blit, SelCopyTextureToBuffer,
                texture, copy.Slice, copy.Level, default,
                new MtlSize { Width = copy.Width, Height = copy.Height, Depth = 1 },
                readback, (nuint)copy.BufferOffset, (nuint)rowBytes, (nuint)(rowBytes * copy.Height),
                (nuint)MtlBlitOption.None);
        }

        MetalTiler.EndEncoding(blit);
        harness.Submit();
        var actual = harness.Read(readback, 0, (ulong)linear.Length);
        AssertLiveTexelsEqual(copies, element, linear, actual, name);
    }

    [Theory]
    [MemberData(nameof(TextureTileCases))]
    public void ATiledColorWritebackReadsTheTextureInOnePassExactlyAsTheOracleTiles(string name)
    {
        if (!RequireDevice()) return;
        using var harness = new MetalTransferHarness(Device.Value);
        var description = TextureTileCase(name);
        var expected = BuildGuest(description, 93, out var linear);
        var (_, copies, tiles, element) = BuildLayout(description);
        Assert.Equal(name.Contains("full chain") || name.Contains("tail"), tiles.Exists(tile => tile.Tail));
        var texture = harness.Texture(harness.Transfer.ShapeFor(description));
        var source = harness.Buffer(linear);
        harness.Begin();
        var blit = harness.Tiler.BeginBlit(harness);
        foreach (var copy in copies)
        {
            var rowBytes = RowBytesOf(copy, element);
            MetalNative.SendCopyBufferToTextureOptions(blit, SelCopyBufferToTexture,
                source, (nuint)copy.BufferOffset, (nuint)rowBytes, (nuint)(rowBytes * copy.Height),
                new MtlSize { Width = copy.Width, Height = copy.Height, Depth = 1 },
                texture, copy.Slice, copy.Level, default, (nuint)MtlBlitOption.None);
        }

        MetalTiler.EndEncoding(blit);
        harness.Submit();

        var target = harness.Buffer(description.Data.Size);
        var range = new SubresourceRange(0, description.Resources.Levels, 0, description.Resources.Layers);
        harness.Begin();
        harness.Transfer.EncodeTextureToGuest(harness, description, range, texture,
            new MetalGuestRange(target, 0, description.Data.Size, 0), default, 0, description.Data.Size);
        harness.Submit();

        Assert.Equal(0u, harness.Tiler.StagingBufferCount);
        Assert.Equal(expected, harness.Read(target, 0, description.Data.Size));
        output.WriteLine($"{name}: one-pass writeback of 0x{description.Data.Size:x} bytes matched the oracle");
    }

    // A texture the one-pass kernels cannot write - an sRGB target has no shader-write
    // usage - still lands exactly, through the staging buffer.
    [Fact]
    public void AnSrgbRenderTargetStillRoundTripsThroughStaging()
    {
        if (!RequireDevice()) return;
        using var harness = new MetalTransferHarness(Device.Value);
        RoundTrip(harness, Color2D(GuestPixelFormat.Bits8_8_8_8Srgb, GpuFormat.R8G8B8A8Srgb,
            192, 96, 1, 1, GuestTileMode.RenderTarget), 95, "srgb render target");
        Assert.NotEqual(0u, harness.Tiler.StagingBufferCount);
    }

    private static ulong RowBytesOf(in MetalSubresourceCopy copy, TileElementLayout element) =>
        (ulong)(copy.BufferRowLength != 0 ? copy.BufferRowLength : copy.Width) * element.Bytes;

    private void AssertLiveTexelsEqual(
        List<MetalSubresourceCopy> copies, TileElementLayout element, byte[] expected, byte[] actual, string name)
    {
        ulong compared = 0;
        foreach (var copy in copies)
        {
            var rowBytes = RowBytesOf(copy, element);
            var live = (int)(copy.Width * element.Bytes);
            for (uint row = 0; row < copy.Height; row++)
            {
                var offset = (int)(copy.BufferOffset + (row * rowBytes));
                Assert.True(
                    expected.AsSpan(offset, live).SequenceEqual(actual.AsSpan(offset, live)),
                    $"{name}: level={copy.Level} slice={copy.Slice} row={row} differs");
                compared += (ulong)live;
            }
        }

        output.WriteLine($"{name}: one-pass upload matched the oracle over 0x{compared:x} texel bytes");
    }
}

// A texture a cached image owns is viewed once for every transfer of its lifetime, and the
// view goes with the last image that refers to the texture; any other texture is viewed
// per transfer and nothing is kept for it.
public sealed partial class MetalImageTransferTests
{
    [Fact]
    public void ACachedImageKeepsOneTilerViewUntilItsTextureIsReleased()
    {
        if (!RequireDevice()) return;
        using var harness = new MetalTransferHarness(Device.Value);
        var description = TextureTileCase("rgba8 64kb full chain");
        var guest = BuildGuest(description, 97, out _);
        var shape = harness.Transfer.ShapeFor(description);
        var texture = harness.Texture(shape);
        var loose = harness.Texture(shape);
        var range = new SubresourceRange(0, description.Resources.Levels, 0, description.Resources.Layers);
        var source = new MetalGuestRange(harness.Buffer(guest), 0, description.Data.Size, 0);

        // The image releases the texture it owns; the harness releases its own reference.
        MetalNative.SendVoid(texture, MetalNative.Selector("retain"));
        var image = new MetalCachedImage(in description, in shape, texture);
        nint kept = 0;
        for (var pass = 0; pass < 3; pass++)
        {
            harness.Begin();
            Assert.True(harness.Transfer.EncodeGuestToTexture(harness, description, range, source, default, texture, 0,
                description.Data.Size));
            Assert.True(harness.Transfer.EncodeGuestToTexture(harness, description, range, source, default, loose, 0,
                description.Data.Size));
            harness.Submit();
            var view = MetalTileViews.KeptView(texture);
            Assert.NotEqual(0, view);
            Assert.True(kept == 0 || kept == view, "the kept view changed between transfers");
            kept = view;
            Assert.Equal(0, MetalTileViews.KeptView(loose));
        }

        // An image retiring the texture (PromoteToOwned and its kin) refers to it too, so
        // the view survives until both have let go.
        var retired = new MetalCachedImage(in description, in shape, texture);
        MetalNative.SendVoid(texture, MetalNative.Selector("retain"));
        image.Dispose();
        Assert.Equal(kept, MetalTileViews.KeptView(texture));
        retired.Dispose();
        Assert.Equal(0, MetalTileViews.KeptView(texture));
    }
}
