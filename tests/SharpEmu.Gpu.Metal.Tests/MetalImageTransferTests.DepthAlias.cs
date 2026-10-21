// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Images;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

// A depth target sampled through an R32 alias over the same guest bytes is handed over
// on the device. Whatever the alias ends up holding has to be exactly what the round
// trip through guest memory - retile the depth plane out, detile it into the alias -
// would have left in it, and the depth-stencil source has to come out of it untouched.
public sealed partial class MetalImageTransferTests
{
    [Theory]
    [InlineData(1280u, 720u, false, 1u, GuestPixelFormat.Bits32Float, GpuFormat.R32Sfloat)]
    [InlineData(2560u, 1440u, true, 1u, GuestPixelFormat.Bits32Float, GpuFormat.R32Sfloat)]
    [InlineData(1280u, 720u, false, 1u, GuestPixelFormat.Bits32UInt, GpuFormat.R32Uint)]
    [InlineData(97u, 33u, true, 3u, GuestPixelFormat.Bits32Float, GpuFormat.R32Sfloat)]
    public void ADepthPlaneHandedToItsR32AliasOnTheDeviceMatchesTheGuestRoundTrip(
        uint width, uint height, bool stencil, uint layers, GuestPixelFormat aliasGuest, GpuFormat aliasHost)
    {
        if (!RequireDevice()) return;
        using var harness = new MetalTransferHarness(Device.Value);
        var depth = Depth(GuestDepthFormat.Z32Float, stencil, width, height, layers);
        var alias = Color2D(aliasGuest, aliasHost, width, height, 1, layers, GuestTileMode.Depth);
        // What the texture descriptor of a sampled depth surface describes: the same
        // footprint, laid out by the same 64 KiB depth blocks at the same pitch.
        Assert.Equal(depth.Data.Size, alias.Data.Size);
        Assert.Equal(depth.Pitch, alias.Pitch);

        var size = depth.Data.Size;
        var guest = BuildGuest(depth, 91);
        var stencilBytes = stencil ? MetalTileOracle.Pattern((int)depth.Stencil.Size, 92) : [];
        var range = new SubresourceRange(0, 1, 0, layers);
        var depthTexture = harness.Texture(harness.Transfer.ShapeFor(depth));
        var stencilIn = stencil ? new MetalGuestRange(harness.Buffer(stencilBytes), 0, depth.Stencil.Size, 0) : default;
        harness.Begin();
        harness.Transfer.EncodeGuestToTexture(harness, depth, range,
            new MetalGuestRange(harness.Buffer(guest), 0, size, 0), stencilIn, depthTexture, 0, size);
        harness.Submit();
        var stencilBefore = stencil ? ReadStencil(harness, depth, depthTexture) : [];

        // The round trip: the depth plane out to guest memory, then into the alias.
        var published = harness.Buffer(size);
        var roundTripAlias = harness.Texture(harness.Transfer.ShapeFor(alias));
        harness.Begin();
        harness.Transfer.EncodeTextureToGuest(harness, depth, range, depthTexture,
            new MetalGuestRange(published, 0, size, 0), default, 0, size);
        harness.Submit();
        harness.Begin();
        harness.Transfer.EncodeGuestToTexture(harness, alias, range,
            new MetalGuestRange(published, 0, size, 0), default, roundTripAlias, 0, size);
        harness.Submit();

        var deviceAlias = harness.Texture(harness.Transfer.ShapeFor(alias));
        harness.Begin();
        Assert.True(harness.Transfer.TryEncodeTextureToTexture(harness, depth, depthTexture, alias, deviceAlias));
        harness.Submit();

        var expected = ReadTexels(harness, alias, roundTripAlias);
        var actual = ReadTexels(harness, alias, deviceAlias);
        Assert.Equal(expected, actual);
        // Both also read back as the guest plane they started from.
        Assert.Equal(guest, WriteBack(harness, alias, deviceAlias));
        if (stencil)
        {
            Assert.Equal(stencilBefore, ReadStencil(harness, depth, depthTexture));
        }

        output.WriteLine($"{width}x{height}x{layers} {(stencil ? "D32S8" : "D32")} -> {aliasHost}: " +
                         $"0x{(ulong)actual.Length:x} texel bytes matched the guest round trip");
    }

    // The reverse hand-over: what a compute wrote through the R32 alias goes back into the
    // depth plane on the device. It has to land exactly as the round trip through guest
    // memory would put it there, and leave the stencil plane the target already had.
    [Theory]
    [InlineData(1280u, 720u, false, 1u, GuestPixelFormat.Bits32Float, GpuFormat.R32Sfloat)]
    [InlineData(960u, 540u, true, 1u, GuestPixelFormat.Bits32Float, GpuFormat.R32Sfloat)]
    [InlineData(97u, 33u, true, 3u, GuestPixelFormat.Bits32UInt, GpuFormat.R32Uint)]
    public void AnR32AliasHandedBackToItsDepthPlaneOnTheDeviceMatchesTheGuestRoundTrip(
        uint width, uint height, bool stencil, uint layers, GuestPixelFormat aliasGuest, GpuFormat aliasHost)
    {
        if (!RequireDevice()) return;
        using var harness = new MetalTransferHarness(Device.Value);
        var depth = Depth(GuestDepthFormat.Z32Float, stencil, width, height, layers);
        var alias = Color2D(aliasGuest, aliasHost, width, height, 1, layers, GuestTileMode.Depth);
        Assert.Equal(depth.Data.Size, alias.Data.Size);
        Assert.Equal(depth.Pitch, alias.Pitch);

        var size = depth.Data.Size;
        // Depth values a depth texture holds as they are: the alias is loaded with the
        // same guest plane the depth target would be.
        var aliasGuestBytes = BuildGuest(depth, 93);
        var oldDepthBytes = BuildGuest(depth, 94);
        var stencilBytes = stencil ? MetalTileOracle.Pattern((int)depth.Stencil.Size, 95) : [];
        var range = new SubresourceRange(0, 1, 0, layers);
        var aliasTexture = harness.Texture(harness.Transfer.ShapeFor(alias));
        var stencilIn = stencil ? new MetalGuestRange(harness.Buffer(stencilBytes), 0, depth.Stencil.Size, 0) : default;
        nint LoadedDepth()
        {
            var texture = harness.Texture(harness.Transfer.ShapeFor(depth));
            harness.Begin();
            harness.Transfer.EncodeGuestToTexture(harness, depth, range,
                new MetalGuestRange(harness.Buffer(oldDepthBytes), 0, size, 0), stencilIn, texture, 0, size);
            harness.Submit();
            return texture;
        }

        harness.Begin();
        harness.Transfer.EncodeGuestToTexture(harness, alias, range,
            new MetalGuestRange(harness.Buffer(aliasGuestBytes), 0, size, 0), default, aliasTexture, 0, size);
        harness.Submit();

        // The round trip: the alias out to guest memory, then into a depth target that
        // keeps its stencil.
        var roundTripDepth = LoadedDepth();
        var stencilBefore = stencil ? ReadStencil(harness, depth, roundTripDepth) : [];
        var published = harness.Buffer(size);
        harness.Begin();
        harness.Transfer.EncodeTextureToGuest(harness, alias, range, aliasTexture,
            new MetalGuestRange(published, 0, size, 0), default, 0, size);
        harness.Submit();
        harness.Begin();
        harness.Transfer.EncodeGuestToTexture(harness, depth, range,
            new MetalGuestRange(published, 0, size, 0), stencilIn, roundTripDepth, 0, size);
        harness.Submit();

        var deviceDepth = LoadedDepth();
        harness.Begin();
        Assert.True(harness.Transfer.TryEncodeTextureToTexture(harness, alias, aliasTexture, depth, deviceDepth));
        harness.Submit();

        Assert.Equal(ReadDepthTexels(harness, depth, roundTripDepth), ReadDepthTexels(harness, depth, deviceDepth));
        Assert.Equal(aliasGuestBytes, WriteBack(harness, depth, deviceDepth));
        if (stencil)
        {
            Assert.Equal(stencilBefore, ReadStencil(harness, depth, deviceDepth));
        }
    }

    // Only the pair whose bits survive unchanged is claimed, in either direction;
    // everything else is left to the round trip.
    [Fact]
    public void ADeviceAliasHandOffRefusesEveryPairItCannotCopyBitForBit()
    {
        if (!RequireDevice()) return;
        using var harness = new MetalTransferHarness(Device.Value);
        var depth = Depth(GuestDepthFormat.Z32Float, stencil: true, 256, 128, 1);
        var alias = Color2D(GuestPixelFormat.Bits32Float, GpuFormat.R32Sfloat, 256, 128, 1, 1, GuestTileMode.Depth);
        var depthTexture = harness.Texture(harness.Transfer.ShapeFor(depth));
        var aliasTexture = harness.Texture(harness.Transfer.ShapeFor(alias));
        var z16 = Depth(GuestDepthFormat.Z16, stencil: false, 256, 128, 1);
        var z16Texture = harness.Texture(harness.Transfer.ShapeFor(z16));
        var r16 = Color2D(GuestPixelFormat.Bits16UNorm, GpuFormat.R16Unorm, 256, 128, 1, 1, GuestTileMode.Depth);
        var r16Texture = harness.Texture(harness.Transfer.ShapeFor(r16));
        var rgba8 = Color2D(GuestPixelFormat.Bits8_8_8_8UNorm, GpuFormat.R8G8B8A8Unorm, 256, 128, 1, 1, GuestTileMode.Depth);
        var rgba8Texture = harness.Texture(harness.Transfer.ShapeFor(rgba8));
        var otherTile = alias;
        otherTile.TileMode = GuestTileMode.RenderTarget;
        var otherPitch = alias;
        otherPitch.Pitch += 128;

        harness.Begin();
        // The 16-bit pair has its own switch (MetalImageTransferTests.Depth16Alias).
        harness.Transfer.Depth16AliasOnDevice = false;
        Assert.False(harness.Transfer.TryEncodeTextureToTexture(harness, z16, z16Texture, r16, r16Texture));
        Assert.False(harness.Transfer.TryEncodeTextureToTexture(harness, r16, r16Texture, z16, z16Texture));
        Assert.False(harness.Transfer.TryEncodeTextureToTexture(harness, rgba8, rgba8Texture, depth, depthTexture));
        Assert.False(harness.Transfer.TryEncodeTextureToTexture(harness, otherTile, aliasTexture, depth, depthTexture));
        Assert.False(harness.Transfer.TryEncodeTextureToTexture(harness, depth, depthTexture, rgba8, rgba8Texture));
        Assert.False(harness.Transfer.TryEncodeTextureToTexture(harness, depth, depthTexture, otherTile, aliasTexture));
        Assert.False(harness.Transfer.TryEncodeTextureToTexture(harness, depth, depthTexture, otherPitch, aliasTexture));
        Assert.True(harness.Transfer.TryEncodeTextureToTexture(harness, depth, depthTexture, alias, aliasTexture));
        Assert.True(harness.Transfer.TryEncodeTextureToTexture(harness, alias, aliasTexture, depth, depthTexture));
        harness.Submit();
    }

    // The depth plane alone, straight off a depth or depth-stencil texture.
    private static byte[] ReadDepthTexels(MetalTransferHarness harness, in ImageDescription description, nint texture)
    {
        var rowBytes = (ulong)description.Extent.Width * sizeof(float);
        var sliceBytes = rowBytes * description.Extent.Height;
        var layers = description.TransferLayers;
        var buffer = harness.Buffer(sliceBytes * layers);
        var option = description.HasStencil ? MtlBlitOption.DepthFromDepthStencil : MtlBlitOption.None;
        harness.Begin();
        var blit = harness.Tiler.BeginBlit(harness);
        for (uint layer = 0; layer < layers; layer++)
        {
            MetalNative.SendCopyTextureToBufferOptions(blit, SelCopyTextureToBuffer,
                texture, layer, 0, default,
                new MtlSize { Width = description.Extent.Width, Height = description.Extent.Height, Depth = 1 },
                buffer, (nuint)(sliceBytes * layer), (nuint)rowBytes, (nuint)sliceBytes, (nuint)option);
        }

        MetalTiler.EndEncoding(blit);
        harness.Submit();
        return harness.Read(buffer, 0, sliceBytes * layers);
    }

    // Every texel of every layer, straight off the texture with no tiling in between.
    private static byte[] ReadTexels(MetalTransferHarness harness, in ImageDescription description, nint texture)
    {
        var rowBytes = (ulong)description.Extent.Width * description.BytesPerBlock;
        var sliceBytes = rowBytes * description.Extent.Height;
        var layers = description.TransferLayers;
        var buffer = harness.Buffer(sliceBytes * layers);
        harness.Begin();
        var blit = harness.Tiler.BeginBlit(harness);
        for (uint layer = 0; layer < layers; layer++)
        {
            MetalNative.SendCopyTextureToBufferOptions(blit, SelCopyTextureToBuffer,
                texture, layer, 0, default,
                new MtlSize { Width = description.Extent.Width, Height = description.Extent.Height, Depth = 1 },
                buffer, (nuint)(sliceBytes * layer), (nuint)rowBytes, (nuint)sliceBytes, (nuint)MtlBlitOption.None);
        }

        MetalTiler.EndEncoding(blit);
        harness.Submit();
        return harness.Read(buffer, 0, sliceBytes * layers);
    }

    private static byte[] ReadStencil(MetalTransferHarness harness, in ImageDescription description, nint texture)
    {
        var rowBytes = (ulong)description.Extent.Width;
        var sliceBytes = rowBytes * description.Extent.Height;
        var layers = description.TransferLayers;
        var buffer = harness.Buffer(sliceBytes * layers);
        harness.Begin();
        var blit = harness.Tiler.BeginBlit(harness);
        for (uint layer = 0; layer < layers; layer++)
        {
            MetalNative.SendCopyTextureToBufferOptions(blit, SelCopyTextureToBuffer,
                texture, layer, 0, default,
                new MtlSize { Width = description.Extent.Width, Height = description.Extent.Height, Depth = 1 },
                buffer, (nuint)(sliceBytes * layer), (nuint)rowBytes, (nuint)sliceBytes,
                (nuint)MtlBlitOption.StencilFromDepthStencil);
        }

        MetalTiler.EndEncoding(blit);
        harness.Submit();
        return harness.Read(buffer, 0, sliceBytes * layers);
    }

    private static byte[] WriteBack(MetalTransferHarness harness, in ImageDescription description, nint texture)
    {
        var size = description.Data.Size;
        var target = harness.Buffer(size);
        harness.Begin();
        harness.Transfer.EncodeTextureToGuest(harness, description,
            new SubresourceRange(0, 1, 0, description.Resources.Layers), texture,
            new MetalGuestRange(target, 0, size, 0), default, 0, size);
        harness.Submit();
        return harness.Read(target, 0, size);
    }
}
