// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Rendering;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

// The same rig drives the cache's alias hand-off: a depth target and the R32 texture a
// shader samples it through, over the same guest bytes.
public sealed unsafe partial class MetalDccMaterializationTests
{
    private const ulong DepthAddress = GuestBase + 0x100000;
    private const uint AliasWidth = 128;
    private const uint AliasHeight = 64;

    private static ImageRequest DepthAliasRequest(ImageRole role, bool stencil = false)
    {
        var depthRole = role == ImageRole.DepthTarget;
        Assert.True(TileGeometry.TryGetDepthSize(AliasWidth, AliasHeight, 0, GuestDepthFormat.Z32Float,
            stencil ? GuestStencilFormat.Stencil8UInt : GuestStencilFormat.Invalid, false,
            out var stencilSize, out _, out var depthSize));
        var pitch = TileGeometry.DepthPitch(AliasWidth, 4);
        var description = ImageDescription.Create();
        description.Data = new GuestSpan(DepthAddress, depthSize.Size);
        if (depthRole && stencil)
        {
            description.Stencil = new GuestSpan(DepthAddress + 0x100000, stencilSize.Size);
        }

        var format = depthRole
            ? stencil ? GpuFormat.D32SfloatS8Uint : GpuFormat.D32Sfloat
            : GpuFormat.R32Sfloat;
        description.PixelFormat = format;
        description.GuestFormat = GuestPixelFormat.Bits32Float;
        description.Type = GuestImageType.Color2D;
        description.Extent = new GpuExtent3D(AliasWidth, AliasHeight, 1);
        description.Resources = new SubresourceCount(1, 1);
        description.Pitch = pitch;
        description.BytesPerBlock = 4;
        description.Samples = 1;
        description.TileMode = GuestTileMode.Depth;
        description.MipLayout[0] = new MipLevelLayout { Offset = 0, Size = depthSize.Size, Pitch = pitch, Height = AliasHeight };
        var view = ImageViewDescription.Default with
        {
            Format = format,
            Type = GpuImageViewType.Type2D,
            Aspect = depthRole ? GpuImageAspect.DepthBit : GpuImageAspect.ColorBit,
            LayerCount = 1,
            Usage = role switch
            {
                ImageRole.DepthTarget => GpuImageUsage.DepthStencilAttachmentBit,
                ImageRole.StorageImage => GpuImageUsage.StorageBit,
                _ => GpuImageUsage.SampledBit,
            },
        };
        return new ImageRequest(description, view, role);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ADepthTargetSampledThroughItsR32AliasIsHandedOverOnTheDevice(bool stencil)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var depthRequest = DepthAliasRequest(ImageRole.DepthTarget, stencil);
        var depth = rig.Cache.FindImage(ref depthRequest);
        rig.Cache.AcquireDepthTargetView(depth, in depthRequest);
        Assert.True(rig.Cache.GetImage(depth).IsGpuModified);

        var textureRequest = DepthAliasRequest(ImageRole.Texture);
        var alias = rig.Cache.FindImage(ref textureRequest);
        Assert.NotEqual(depth, alias);
        rig.Cache.AcquireTextureView(alias, in textureRequest);

        // Served on the device: the alias owes guest memory no upload, and the depth
        // target still owns the bytes, so nothing was written back for it.
        Assert.Equal(1, rig.Cache.TakeDeviceAliasPublishes());
        Assert.False(rig.Cache.GetImage(alias).IsCpuDirty);
        Assert.True(rig.Cache.GetImage(depth).IsGpuModified);

        // A write through the alias is handed the depth target's contents first and then
        // takes the bytes from it, exactly as it does through guest memory. Neither
        // texture changed since the copy above, so the alias already holds them.
        var storageRequest = DepthAliasRequest(ImageRole.StorageImage);
        Assert.Equal(alias, rig.Cache.FindImage(ref storageRequest));
        rig.Cache.AcquireTextureView(alias, in storageRequest);
        Assert.Equal(0, rig.Cache.TakeDeviceAliasPublishes());
        Assert.Equal(1, rig.Cache.TakeDeviceAliasReuses());
        Assert.False(rig.Cache.GetImage(depth).IsGpuModified);
        Assert.True(rig.Cache.GetImage(depth).IsCpuDirty);
        Assert.True(rig.Cache.GetImage(alias).IsGpuModified);

        // The way back is on the device too: the depth plane takes what was written through
        // the alias without a round trip through guest memory (the leg that left Black Myth:
        // Wukong's base pass testing against a frame-old depth buffer), so the depth target
        // comes back current rather than owing an upload.
        rig.Cache.AcquireDepthTargetView(depth, in depthRequest);
        Assert.Equal(1, rig.Cache.TakeDeviceAliasPublishes());
        Assert.False(rig.Cache.GetImage(depth).IsCpuDirty);
        Assert.True(rig.Cache.GetImage(depth).IsGpuModified);
        rig.Harness.Submit();
    }

    // The alias is copied again only when one of the two textures changed since the last
    // copy: a depth target that only tests depth leaves it current, a draw that writes
    // depth does not, and neither does guest memory handing the alias newer bytes.
    [Fact]
    public void AnAliasIsCopiedAgainOnlyAfterEitherTextureChanged()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var depthRequest = DepthAliasRequest(ImageRole.DepthTarget);
        var depth = rig.Cache.FindImage(ref depthRequest);
        rig.Cache.AcquireDepthTargetView(depth, in depthRequest);
        var textureRequest = DepthAliasRequest(ImageRole.Texture);
        var alias = rig.Cache.FindImage(ref textureRequest);
        rig.Cache.AcquireTextureView(alias, in textureRequest);
        Assert.Equal(1, rig.Cache.TakeDeviceAliasPublishes());

        // Sampled again, and again behind a pass that only tests depth: nothing to copy.
        rig.Cache.AcquireTextureView(alias, in textureRequest);
        rig.Cache.AcquireDepthTargetView(depth, in depthRequest, writes: false);
        rig.Cache.AcquireTextureView(alias, in textureRequest);
        Assert.Equal(0, rig.Cache.TakeDeviceAliasPublishes());
        Assert.Equal(2, rig.Cache.TakeDeviceAliasReuses());
        Assert.True(rig.Cache.GetImage(depth).IsGpuModified);
        Assert.False(rig.Cache.GetImage(alias).IsCpuDirty);

        // A draw that writes depth changes what the alias has to hold.
        rig.Cache.AcquireDepthTargetView(depth, in depthRequest);
        rig.Cache.AcquireTextureView(alias, in textureRequest);
        Assert.Equal(1, rig.Cache.TakeDeviceAliasPublishes());

        // So does a change to the alias's own texture.
        rig.Cache.GetImage(alias).NoteContentChanged();
        rig.Cache.AcquireTextureView(alias, in textureRequest);
        Assert.Equal(1, rig.Cache.TakeDeviceAliasPublishes());

        // And any other GPU write to the depth target (a copy or resolve into it).
        rig.Cache.MarkGpuWritten(depth);
        rig.Cache.AcquireTextureView(alias, in textureRequest);
        Assert.Equal(1, rig.Cache.TakeDeviceAliasPublishes());
        Assert.Equal(0, rig.Cache.TakeDeviceAliasReuses());
        rig.Harness.Submit();
    }

    // The hand-off copies the depth plane and nothing else, so a pass that writes only
    // stencil leaves the alias holding what a new copy would give it: SILENT HILL's
    // decals mark stencil with one draw and sample depth through the R32 alias with the
    // next, and copying the 4K depth plane between every pair also split the pass twice.
    [Fact]
    public void AStencilOnlyPassLeavesTheR32AliasOfTheDepthPlaneCurrent()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var depthRequest = DepthAliasRequest(ImageRole.DepthTarget, stencil: true);
        var depth = rig.Cache.FindImage(ref depthRequest);
        rig.Cache.AcquireDepthTargetView(depth, in depthRequest);
        var textureRequest = DepthAliasRequest(ImageRole.Texture);
        var alias = rig.Cache.FindImage(ref textureRequest);
        rig.Cache.AcquireTextureView(alias, in textureRequest);
        Assert.Equal(1, rig.Cache.TakeDeviceAliasPublishes());

        rig.Cache.AcquireDepthTargetView(depth, in depthRequest, writes: true, writesDepth: false);
        Assert.True(rig.Cache.GetImage(depth).IsGpuModified);
        Assert.False(rig.Cache.GetImage(alias).IsCpuDirty);
        rig.Cache.AcquireTextureView(alias, in textureRequest);
        Assert.Equal(0, rig.Cache.TakeDeviceAliasPublishes());
        Assert.Equal(1, rig.Cache.TakeDeviceAliasReuses());

        // A pass that writes depth still hands the alias the new plane.
        rig.Cache.AcquireDepthTargetView(depth, in depthRequest);
        rig.Cache.AcquireTextureView(alias, in textureRequest);
        Assert.Equal(1, rig.Cache.TakeDeviceAliasPublishes());
        Assert.Equal(0, rig.Cache.TakeDeviceAliasReuses());
        rig.Harness.Submit();
    }

    // A version names one state of one texture, so a copy from another source is never
    // mistaken for the one the alias holds.
    [Fact]
    public void ContentVersionsAreDistinctAcrossImagesAndAdvanceOnEveryChange()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var depthRequest = DepthAliasRequest(ImageRole.DepthTarget);
        var depth = rig.Cache.GetImage(rig.Cache.FindImage(ref depthRequest));
        var textureRequest = DepthAliasRequest(ImageRole.Texture);
        var alias = rig.Cache.GetImage(rig.Cache.FindImage(ref textureRequest));
        Assert.NotEqual(depth.ContentVersion, alias.ContentVersion);

        var before = alias.ContentVersion;
        alias.MarkCpuDirty(MetalImageDirtyCause.CpuWrite, 0, 4);
        Assert.NotEqual(before, alias.ContentVersion);
        before = alias.ContentVersion;
        alias.RefreshComplete();
        Assert.NotEqual(before, alias.ContentVersion);
        before = alias.ContentVersion;
        alias.MarkGpuModified();
        Assert.NotEqual(before, alias.ContentVersion);
        Assert.False(alias.StillHoldsPublicationFrom(depth));
        alias.NotePublishedFrom(depth);
        Assert.True(alias.StillHoldsPublicationFrom(depth));
        depth.MarkGpuModified();
        Assert.False(alias.StillHoldsPublicationFrom(depth));
    }
    // What decides whether the alias copied from a depth target is still current: a draw
    // the guest calls read-only but Metal writes (writes on, test off) has to count.
    [Theory]
    [InlineData(true, true, false, (int)GpuFormat.D32Sfloat, true)]
    [InlineData(false, true, false, (int)GpuFormat.D32Sfloat, true)]
    [InlineData(false, true, false, (int)GpuFormat.D32SfloatS8Uint, true)]
    [InlineData(false, false, true, (int)GpuFormat.D32Sfloat, true)]
    [InlineData(true, false, false, (int)GpuFormat.D32Sfloat, false)]
    [InlineData(false, false, false, (int)GpuFormat.D32Sfloat, false)]
    public void ADepthAttachmentCountsAsWrittenWheneverMetalWritesIt(
        bool test, bool write, bool clear, int format, bool expected)
    {
        var state = default(DepthStencilState) with
        {
            DepthTestEnabled = test,
            DepthWriteEnabled = write,
            DepthClearEnabled = clear,
            DepthCompare = GpuCompareOp.Less,
        };

        Assert.Equal(expected, MetalRenderHost.WritesAttachment(in state, (GpuFormat)format));
    }
}
