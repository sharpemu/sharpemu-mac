// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Images;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

// A depth surface the guest clears through its HTILE and then samples before binding it as a depth target again
// (upstream GuestImageCacheTests.SampledDepthClear, cb790c35e): the sample reads the cleared value.
public sealed unsafe partial class MetalDccMaterializationTests
{
    private const ulong SampledHtileAddress = DepthAddress + 0x280000;

    private static ImageRequest HtileDepth(uint layers, bool stencil = false)
    {
        var depth = DepthAliasRequest(ImageRole.DepthTarget, stencil);
        if (layers > 1)
        {
            var size = depth.Description.Data.Size;
            depth.Description.Data = new GuestSpan(depth.Description.Data.Address, size * layers);
            depth.Description.Resources = new SubresourceCount(1, layers);
            depth.Description.MipLayout[0] = depth.Description.MipLayout[0] with { Size = size * layers };
            depth.View = depth.View with { Type = GpuImageViewType.Type2DArray, LayerCount = layers };
        }

        depth.Description.Metadata.Kind = MetadataKind.Htile;
        depth.Description.Metadata.Range = new GuestSpan(SampledHtileAddress, 0x1000 * layers);
        return depth;
    }

    private static ImageRequest SampledDepth(ImageRequest depth)
    {
        depth.Role = ImageRole.Texture;
        depth.Description.Metadata = default;
        depth.Description.Stencil = default;
        depth.View = depth.View with { Usage = GpuImageUsage.SampledBit };
        return depth;
    }

    // A bind with HTILE, the clear it loads consumed as the render host does, the bind's clear value noted, and then a
    // compute clear of the HTILE that leaves the clear pending.
    private static ResourceSlotIdentifier BoundThenHtileCleared(Rig rig, ImageRequest depth, float clearValue)
    {
        var depthId = rig.Cache.FindImage(ref depth);
        rig.Cache.AcquireDepthTargetView(depthId, in depth);
        Assert.True(rig.Cache.SetMetadataSlice(SampledHtileAddress, depth.View.BaseLayer, false));
        rig.Cache.NoteDepthClearValue(SampledHtileAddress, clearValue);
        Assert.True(rig.Cache.ClearMetadata(SampledHtileAddress));
        return depthId;
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ASampledDepthTakesThePendingHtileClearWithTheValueOfItsLastBind(bool stencil)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var depth = HtileDepth(1, stencil);
        var depthId = BoundThenHtileCleared(rig, depth, 0.25f);
        rig.Owner.DepthClears.Clear();

        var sampled = SampledDepth(depth);
        Assert.Equal(depthId, rig.Cache.FindImage(ref sampled));
        rig.Cache.AcquireTextureView(depthId, in sampled);

        var clear = Assert.Single(rig.Owner.DepthClears);
        Assert.Equal(0.25f, clear.Depth);
        Assert.True(clear.ClearDepth);
        Assert.False(clear.ClearStencil);
        Assert.Equal(new SubresourceRange(0, 1, 0, 1), clear.Range);
        Assert.False(rig.Cache.IsMetadataCleared(SampledHtileAddress, 0));
        Assert.True(rig.Cache.GetImage(depthId).IsGpuModified);
        Assert.Equal(1, rig.Cache.TakeSampledDepthClears());

        // Taken once: the next sample reads what the clear left.
        rig.Cache.AcquireTextureView(depthId, in sampled);
        Assert.Single(rig.Owner.DepthClears);
        rig.Harness.Submit();
    }

    // Sampled as its R32 colour twin, the depth surface binds its own texture and takes the clear the same way.
    [Fact]
    public void ADepthSampledAsItsColourTwinTakesThePendingHtileClear()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Cache.DepthTwins = true;
        var depthId = BoundThenHtileCleared(rig, HtileDepth(1), 0.5f);
        rig.Owner.DepthClears.Clear();

        var twin = DepthAliasRequest(ImageRole.Texture);
        Assert.Equal(depthId, rig.Cache.FindImage(ref twin));
        rig.Cache.AcquireTextureView(depthId, in twin);

        Assert.Equal(0.5f, Assert.Single(rig.Owner.DepthClears).Depth);
        Assert.False(rig.Cache.IsMetadataCleared(SampledHtileAddress, 0));
        rig.Harness.Submit();
    }

    // A layered surface's clear bits are not exact per slice: its pending clear stays for its next depth-target bind.
    [Fact]
    public void ASampledDepthLeavesALayeredSurfacesPendingClearToItsNextBind()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var depth = HtileDepth(2);
        var depthId = BoundThenHtileCleared(rig, depth, 0.25f);
        rig.Owner.DepthClears.Clear();

        var sampled = SampledDepth(depth);
        Assert.Equal(depthId, rig.Cache.FindImage(ref sampled));
        rig.Cache.AcquireTextureView(depthId, in sampled);

        Assert.Empty(rig.Owner.DepthClears);
        Assert.True(rig.Cache.IsMetadataCleared(SampledHtileAddress, 0));
        rig.Harness.Submit();
    }

    // No bind noted a clear value: nothing says what the clear loads, so the sample leaves it to the next bind.
    [Fact]
    public void ASampledDepthWithNoBoundClearValueLeavesThePendingClear()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var depth = HtileDepth(1);
        var depthId = rig.Cache.FindImage(ref depth);
        rig.Cache.AcquireDepthTargetView(depthId, in depth);
        Assert.True(rig.Cache.ClearMetadata(SampledHtileAddress));
        rig.Owner.DepthClears.Clear();

        var sampled = SampledDepth(depth);
        rig.Cache.AcquireTextureView(depthId, in sampled);

        Assert.Empty(rig.Owner.DepthClears);
        Assert.True(rig.Cache.IsMetadataCleared(SampledHtileAddress, 0));
        rig.Harness.Submit();
    }

    // A memoised sampled binding of the surface stops holding once its HTILE is cleared again, so the next bind takes
    // the full acquire, which records the clear.
    [Fact]
    public void AMemoisedSampledDepthMissesOnceItsHtileIsClearedAgain()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var depth = HtileDepth(1);
        var depthId = BoundThenHtileCleared(rig, depth, 0.25f);
        var sampled = SampledDepth(depth);
        var pages = rig.Cache.RequestPageVersions(in sampled);
        Assert.Equal(depthId, rig.Cache.FindImage(ref sampled));
        rig.Cache.AcquireTextureViewForMemo(depthId, in sampled, pages, out _, out var stamp);
        Assert.NotNull(stamp);
        Assert.True(rig.Cache.TryReacquireTextureView(depthId, in sampled, stamp!.Value));

        Assert.True(rig.Cache.ClearMetadata(SampledHtileAddress));
        Assert.False(rig.Cache.TextureStampMayHold(stamp.Value));
        Assert.False(rig.Cache.TryReacquireTextureView(depthId, in sampled, stamp.Value));
        rig.Harness.Submit();
    }
}
