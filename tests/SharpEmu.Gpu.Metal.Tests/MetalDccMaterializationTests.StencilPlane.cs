// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Images;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

// The stencil plane of a depth target sampled through an R8 image over its own bytes, the
// way Demon's Souls' deferred lighting reads the stencil to tell lit geometry from sky.
// The plane lives at DB_STENCIL_*_BASE, outside the depth bytes the page owner index
// holds, so the two images only meet through the stencil plane.
public sealed unsafe partial class MetalDccMaterializationTests
{
    private const ulong StencilPlaneAddress = DepthAddress + 0x100000;

    private static ImageRequest StencilPlaneRequest(ImageRole role)
    {
        Assert.True(TileGeometry.TryGetDepthSize(AliasWidth, AliasHeight, 0, GuestDepthFormat.Z32Float,
            GuestStencilFormat.Stencil8UInt, false, out var stencilSize, out _, out _));
        var pitch = TileGeometry.DepthPitch(AliasWidth, 1);
        var description = ImageDescription.Create();
        description.Data = new GuestSpan(StencilPlaneAddress, stencilSize.Size);
        description.PixelFormat = GpuFormat.R8Uint;
        description.GuestFormat = GuestPixelFormat.Bits8UInt;
        description.Type = GuestImageType.Color2D;
        description.Extent = new GpuExtent3D(AliasWidth, AliasHeight, 1);
        description.Resources = new SubresourceCount(1, 1);
        description.Pitch = pitch;
        description.BytesPerBlock = 1;
        description.Samples = 1;
        description.TileMode = GuestTileMode.Depth;
        description.MipLayout[0] = new MipLevelLayout
        {
            Offset = 0, Size = stencilSize.Size, Pitch = pitch, Height = AliasHeight,
        };
        var view = ImageViewDescription.Default with
        {
            Format = GpuFormat.R8Uint,
            Type = GpuImageViewType.Type2D,
            Aspect = GpuImageAspect.ColorBit,
            LayerCount = 1,
            Usage = role == ImageRole.StorageImage ? GpuImageUsage.StorageBit : GpuImageUsage.SampledBit,
        };
        return new ImageRequest(description, view, role);
    }

    // A pass writes the stencil, then a shader samples the plane: the depth target owns
    // the only copy, so it is written back and the R8 image reloads from guest memory.
    // Sampled again with nothing written in between, nothing moves.
    [Fact]
    public void AStencilPlaneSampledAsR8SeesWhatTheDepthTargetWrote()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var depthRequest = DepthAliasRequest(ImageRole.DepthTarget, stencil: true);
        Assert.Equal(StencilPlaneAddress, depthRequest.Description.Stencil.Address);
        var depth = rig.Cache.FindImage(ref depthRequest);
        rig.Cache.AcquireDepthTargetView(depth, in depthRequest);
        Assert.True(rig.Cache.GetImage(depth).IsGpuModified);

        var planeRequest = StencilPlaneRequest(ImageRole.Texture);
        var plane = rig.Cache.FindImage(ref planeRequest);
        Assert.NotEqual(depth, plane);
        rig.Cache.AcquireTextureView(plane, in planeRequest);
        Assert.Equal(1, rig.Cache.TakeStencilPlanePublishes());
        Assert.False(rig.Cache.GetImage(depth).IsGpuModified);
        Assert.False(rig.Cache.GetImage(plane).IsCpuDirty);

        rig.Cache.AcquireTextureView(plane, in planeRequest);
        Assert.Equal(0, rig.Cache.TakeStencilPlanePublishes());

        // The next frame's stencil writes owe the R8 image another copy.
        rig.Cache.AcquireDepthTargetView(depth, in depthRequest);
        rig.Cache.AcquireTextureView(plane, in planeRequest);
        Assert.Equal(1, rig.Cache.TakeStencilPlanePublishes());
        rig.Harness.Submit();
    }

    // A shader that writes the plane through the R8 image takes it from the depth target.
    // When the depth target binds again its stencil plane is copied from the R8 texture on
    // the device: the depth plane its texture holds is the newest and must not round-trip
    // through guest memory, which is what a reload of both planes did.
    [Fact]
    public void AStencilPlaneWrittenAsR8IsHandedToTheDepthTargetOnTheDevice()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var depthRequest = DepthAliasRequest(ImageRole.DepthTarget, stencil: true);
        var depth = rig.Cache.FindImage(ref depthRequest);
        rig.Cache.AcquireDepthTargetView(depth, in depthRequest);

        var storageRequest = StencilPlaneRequest(ImageRole.StorageImage);
        var plane = rig.Cache.FindImage(ref storageRequest);
        rig.Cache.AcquireTextureView(plane, in storageRequest);
        Assert.Equal(1, rig.Cache.TakeStencilPlanePublishes());
        Assert.False(rig.Cache.GetImage(depth).IsGpuModified);
        Assert.False(rig.Cache.GetImage(depth).IsCpuDirty);
        Assert.True(rig.Cache.GetImage(depth).StencilPlaneStale);
        Assert.True(rig.Cache.GetImage(plane).IsGpuModified);

        rig.Cache.AcquireDepthTargetView(depth, in depthRequest);
        Assert.Equal(1, rig.Cache.TakeStencilDeviceHandOvers());
        Assert.Equal(0, rig.Cache.TakeStencilPlanePublishes());
        Assert.False(rig.Cache.GetImage(depth).StencilPlaneStale);
        Assert.False(rig.Cache.GetImage(depth).IsCpuDirty);
        Assert.True(rig.Cache.GetImage(depth).IsGpuModified);
        Assert.False(rig.Cache.GetImage(plane).IsGpuModified);
        Assert.True(rig.Cache.GetImage(plane).IsCpuDirty);

        // The R8 image did not change since: binding the depth target again copies nothing.
        rig.Cache.AcquireDepthTargetView(depth, in depthRequest);
        Assert.Equal(0, rig.Cache.TakeStencilDeviceHandOvers());
        rig.Harness.Submit();
    }
}
