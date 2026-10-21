// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Images;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

// A 16-bit depth plane and the R16 alias over the same guest bytes are handed over on the
// device both ways. Whatever the target ends up holding has to be exactly what the round
// trip through guest memory - retile the source out, detile it into the target - would
// have left in it, and writing the target back has to leave the guest bytes the round trip
// published. The source texture is filled texel by texel, so every texel is checked, not
// only the ones an upload of guest bytes could have reached.
public sealed partial class MetalImageTransferTests
{
    // SILENT HILL's pairs, as its alias warnings name them: the 4096x4096 shadow map, the
    // 512 and 256 cube maps, and a 32x32 cube whose faces are one padded 64 KiB block each;
    // then an extent that fills no block row, and the integer channels.
    public static TheoryData<uint, uint, uint, GuestPixelFormat, GpuFormat> Depth16Pairs => new()
    {
        { 4096u, 4096u, 1u, GuestPixelFormat.Bits16UNorm, GpuFormat.R16Unorm },
        { 512u, 512u, 6u, GuestPixelFormat.Bits16UNorm, GpuFormat.R16Unorm },
        { 256u, 256u, 6u, GuestPixelFormat.Bits16UNorm, GpuFormat.R16Unorm },
        { 32u, 32u, 6u, GuestPixelFormat.Bits16UNorm, GpuFormat.R16Unorm },
        { 97u, 33u, 2u, GuestPixelFormat.Bits16UNorm, GpuFormat.R16Unorm },
        { 256u, 128u, 1u, GuestPixelFormat.Bits16UInt, GpuFormat.R16Uint },
        { 256u, 128u, 1u, GuestPixelFormat.Bits16SInt, GpuFormat.R16Sint },
    };

    [Theory]
    [MemberData(nameof(Depth16Pairs))]
    public void ADepth16PlaneHandedToItsR16AliasOnTheDeviceMatchesTheGuestRoundTrip(
        uint width, uint height, uint layers, GuestPixelFormat aliasGuest, GpuFormat aliasHost) =>
        AssertDepth16HandOffMatchesRoundTrip(depthToColor: true, width, height, layers, aliasGuest, aliasHost);

    [Theory]
    [MemberData(nameof(Depth16Pairs))]
    public void AnR16AliasHandedToItsDepth16PlaneOnTheDeviceMatchesTheGuestRoundTrip(
        uint width, uint height, uint layers, GuestPixelFormat aliasGuest, GpuFormat aliasHost) =>
        AssertDepth16HandOffMatchesRoundTrip(depthToColor: false, width, height, layers, aliasGuest, aliasHost);

    private void AssertDepth16HandOffMatchesRoundTrip(
        bool depthToColor, uint width, uint height, uint layers, GuestPixelFormat aliasGuest, GpuFormat aliasHost)
    {
        if (!RequireDevice()) return;
        using var harness = new MetalTransferHarness(Device.Value);
        var depth = Depth(GuestDepthFormat.Z16, stencil: false, width, height, layers);
        var alias = Color2D(aliasGuest, aliasHost, width, height, 1, layers, GuestTileMode.Depth);
        // What the texture descriptor of a sampled depth surface describes: the same
        // footprint, laid out by the same 64 KiB depth blocks at the same pitch.
        Assert.Equal(GpuFormat.D16Unorm, depth.PixelFormat);
        Assert.Equal(depth.Data.Size, alias.Data.Size);
        Assert.Equal(depth.Pitch, alias.Pitch);
        var source = depthToColor ? depth : alias;
        var target = depthToColor ? alias : depth;
        var size = source.Data.Size;
        var range = new SubresourceRange(0, 1, 0, layers);

        var sourceTexture = harness.Texture(harness.Transfer.ShapeFor(source));
        WriteTexels(harness, source, sourceTexture,
            MetalTileOracle.Pattern((int)((ulong)width * height * sizeof(ushort) * layers), 161));

        // The round trip: the source out to guest memory, then into the target.
        var published = harness.Buffer(new byte[size]);
        var roundTrip = harness.Texture(harness.Transfer.ShapeFor(target));
        harness.Begin();
        harness.Transfer.EncodeTextureToGuest(harness, source, range, sourceTexture,
            new MetalGuestRange(published, 0, size, 0), default, 0, size);
        harness.Submit();
        harness.Begin();
        harness.Transfer.EncodeGuestToTexture(harness, target, range,
            new MetalGuestRange(published, 0, size, 0), default, roundTrip, 0, size);
        harness.Submit();

        var device = harness.Texture(harness.Transfer.ShapeFor(target));
        harness.Begin();
        Assert.True(harness.Transfer.TryEncodeTextureToTexture(harness, source, sourceTexture, target, device));
        harness.Submit();

        var expected = ReadTexels(harness, target, roundTrip);
        var actual = ReadTexels(harness, target, device);
        Assert.Equal(expected, actual);
        // Written back, the target leaves exactly the guest bytes the round trip published.
        Assert.Equal(harness.Read(published, 0, size), WriteBack(harness, target, device));
        output.WriteLine($"{width}x{height}x{layers} {(depthToColor ? $"D16 -> {aliasHost}" : $"{aliasHost} -> D16")}: " +
                         $"0x{(ulong)actual.Length:x} texel bytes matched the guest round trip");
    }

    // Only the pair proven above is claimed, and only while the switch is on: another tile
    // mode, pitch or channel layout, a DCC colour side, the other width, or a colour source
    // for a 32-bit depth plane is left to the round trip.
    [Fact]
    public void ADepth16HandOffRefusesEveryPairItHasNotProvenBitExact()
    {
        if (!RequireDevice()) return;
        using var harness = new MetalTransferHarness(Device.Value);
        var z16 = Depth(GuestDepthFormat.Z16, stencil: false, 256, 128, 1);
        var z16Texture = harness.Texture(harness.Transfer.ShapeFor(z16));
        var r16 = Color2D(GuestPixelFormat.Bits16UNorm, GpuFormat.R16Unorm, 256, 128, 1, 1, GuestTileMode.Depth);
        var r16Texture = harness.Texture(harness.Transfer.ShapeFor(r16));
        var r16Float = Color2D(GuestPixelFormat.Bits16Float, GpuFormat.R16Sfloat, 256, 128, 1, 1, GuestTileMode.Depth);
        var r16FloatTexture = harness.Texture(harness.Transfer.ShapeFor(r16Float));
        var z32 = Depth(GuestDepthFormat.Z32Float, stencil: false, 256, 128, 1);
        var z32Texture = harness.Texture(harness.Transfer.ShapeFor(z32));
        var r32 = Color2D(GuestPixelFormat.Bits32Float, GpuFormat.R32Sfloat, 256, 128, 1, 1, GuestTileMode.Depth);
        var r32Texture = harness.Texture(harness.Transfer.ShapeFor(r32));
        var otherTile = Color2D(GuestPixelFormat.Bits16UNorm, GpuFormat.R16Unorm, 256, 128, 1, 1, GuestTileMode.RenderTarget);
        var otherTileTexture = harness.Texture(harness.Transfer.ShapeFor(otherTile));
        var otherPitch = r16;
        otherPitch.Pitch += 128;
        var dcc = r16;
        dcc.Metadata.Kind = MetadataKind.Dcc;
        dcc.Metadata.Range = new GuestSpan(GuestAddress + r16.Data.Size, 0x1000);
        var linearDepth = LinearDepth(GuestDepthFormat.Z16, 256, 128, 1);
        var linearDepthTexture = harness.Texture(harness.Transfer.ShapeFor(linearDepth));
        var linearAlias = Color2D(GuestPixelFormat.Bits16UNorm, GpuFormat.R16Unorm, 256, 128, 1, 1, GuestTileMode.Linear);
        var linearAliasTexture = harness.Texture(harness.Transfer.ShapeFor(linearAlias));
        Assert.Equal(linearDepth.Pitch, linearAlias.Pitch);

        harness.Begin();
        Assert.False(harness.Transfer.TryEncodeTextureToTexture(harness, z16, z16Texture, r16Float, r16FloatTexture));
        Assert.False(harness.Transfer.TryEncodeTextureToTexture(harness, r16Float, r16FloatTexture, z16, z16Texture));
        Assert.False(harness.Transfer.TryEncodeTextureToTexture(harness, z16, z16Texture, otherTile, otherTileTexture));
        Assert.False(harness.Transfer.TryEncodeTextureToTexture(harness, otherTile, otherTileTexture, z16, z16Texture));
        Assert.False(harness.Transfer.TryEncodeTextureToTexture(harness, z16, z16Texture, otherPitch, r16Texture));
        Assert.False(harness.Transfer.TryEncodeTextureToTexture(harness, otherPitch, r16Texture, z16, z16Texture));
        Assert.False(harness.Transfer.TryEncodeTextureToTexture(harness, z16, z16Texture, dcc, r16Texture));
        Assert.False(harness.Transfer.TryEncodeTextureToTexture(harness, dcc, r16Texture, z16, z16Texture));
        Assert.False(harness.Transfer.TryEncodeTextureToTexture(harness, linearDepth, linearDepthTexture, linearAlias, linearAliasTexture));
        Assert.False(harness.Transfer.TryEncodeTextureToTexture(harness, linearAlias, linearAliasTexture, linearDepth, linearDepthTexture));
        // Each width only with its own (the 32-bit pair both ways: MetalImageTransferTests.DepthAlias).
        Assert.False(harness.Transfer.TryEncodeTextureToTexture(harness, z16, z16Texture, r32, r32Texture));
        Assert.False(harness.Transfer.TryEncodeTextureToTexture(harness, z32, z32Texture, r16, r16Texture));
        Assert.False(harness.Transfer.TryEncodeTextureToTexture(harness, r16, r16Texture, z32, z32Texture));
        Assert.True(harness.Transfer.TryEncodeTextureToTexture(harness, r32, r32Texture, z32, z32Texture));
        Assert.True(harness.Transfer.TryEncodeTextureToTexture(harness, z16, z16Texture, r16, r16Texture));
        Assert.True(harness.Transfer.TryEncodeTextureToTexture(harness, r16, r16Texture, z16, z16Texture));

        harness.Transfer.Depth16AliasOnDevice = false;
        Assert.False(harness.Transfer.TryEncodeTextureToTexture(harness, z16, z16Texture, r16, r16Texture));
        Assert.False(harness.Transfer.TryEncodeTextureToTexture(harness, r16, r16Texture, z16, z16Texture));
        Assert.True(harness.Transfer.TryEncodeTextureToTexture(harness, z32, z32Texture, r32, r32Texture));
        harness.Submit();
    }

    // Every texel of every layer, straight into the texture with no tiling in between.
    private static void WriteTexels(MetalTransferHarness harness, in ImageDescription description, nint texture, byte[] texels)
    {
        var rowBytes = (ulong)description.Extent.Width * description.BytesPerBlock;
        var sliceBytes = rowBytes * description.Extent.Height;
        var buffer = harness.Buffer(texels);
        harness.Begin();
        var blit = harness.Tiler.BeginBlit(harness);
        for (uint layer = 0; layer < description.TransferLayers; layer++)
        {
            MetalNative.SendCopyBufferToTextureOptions(blit, SelCopyBufferToTexture,
                buffer, (nuint)(sliceBytes * layer), (nuint)rowBytes, (nuint)sliceBytes,
                new MtlSize { Width = description.Extent.Width, Height = description.Extent.Height, Depth = 1 },
                texture, layer, 0, default, (nuint)MtlBlitOption.None);
        }

        MetalTiler.EndEncoding(blit);
        harness.Submit();
    }
}
