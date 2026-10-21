// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Rendering;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

// Which of two images over the same bytes holds them is a question of which wrote them
// last. A DCC fast clear materialized at a target's lookup is such a write: the target
// then holds what the guest asked for, and an alias written before it must not be
// published back over it when the target is bound.
public sealed unsafe partial class MetalDccMaterializationTests
{
    // A transient over the lower half of the target's bytes, as UE5's transient allocator
    // places them: a different surface, not a view of the same one.
    private static ImageRequest R32AliasRequest(ImageRole role)
    {
        var description = ImageDescription.Create();
        description.Data = new GuestSpan(ImageAddress + 32, 4 * 2 * 4);
        description.PixelFormat = GpuFormat.R32Uint;
        description.GuestFormat = GuestPixelFormat.Bits32UInt;
        description.Type = GuestImageType.Color2D;
        description.Extent = new GpuExtent3D(4, 2, 1);
        description.Resources = new SubresourceCount(1, 1);
        description.Pitch = 4;
        description.BytesPerBlock = 4;
        description.Samples = 1;
        description.TileMode = GuestTileMode.Linear;
        description.MipLayout[0] = new MipLevelLayout { Offset = 0, Size = 32, Pitch = 4, Height = 2 };
        var view = ImageViewDescription.Default with
        {
            Format = GpuFormat.R32Uint,
            Type = GpuImageViewType.Type2D,
            Aspect = GpuImageAspect.ColorBit,
            LayerCount = 1,
            Usage = role == ImageRole.ColorTarget ? GpuImageUsage.ColorAttachmentBit : GpuImageUsage.SampledBit,
        };
        return new ImageRequest(description, view, role);
    }

    // UE5's shadow mask: cleared white by a fast clear, then bound for the shadow projections
    // over memory a transient render target wrote earlier in the frame. Publishing that
    // target over the clear put its bytes in the per-object shadow channel, and every
    // character came out unlit.
    [Fact]
    public void AFastClearIsNotOverwrittenByAnOlderAliasWhenTheTargetIsBound()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var target = rig.Lookup(ImageRole.ColorTarget);
        var aliasRequest = R32AliasRequest(ImageRole.ColorTarget);
        var alias = rig.Cache.FindImage(ref aliasRequest);
        Assert.NotEqual(target, alias);
        rig.Cache.MarkGpuWritten(alias);

        rig.FillMetadata(0x00);
        var targetRequest = rig.Request(ImageRole.ColorTarget);
        Assert.Equal(target, rig.Cache.FindImage(ref targetRequest));
        Assert.Equal(1, rig.Owner.Clears);
        rig.Cache.AcquireColorTargetView(target, in targetRequest);

        // The materialized clear already took the shared bytes from the alias (or the
        // newer-writer rule kept them when the target bound): either way nothing of the
        // alias is published over the clear.
        Assert.InRange(rig.Cache.TakeAliasNewerKept(), 0, 1);
        Assert.Equal(0, rig.Cache.TakeDeviceAliasPublishes());
        Assert.True(rig.Cache.GetImage(target).IsGpuModified);
        Assert.False(rig.Cache.GetImage(target).IsCpuDirty);
        var olderAlias = rig.Cache.GetImage(alias);
        Assert.True(!olderAlias.IsGpuModified || olderAlias.IsCpuDirty);
        rig.Harness.Submit();
    }

    // The other order is the ordinary one: a target bound after an alias wrote the shared
    // bytes still takes them from the alias.
    [Fact]
    public void ATargetBoundAfterANewerAliasStillTakesItsBytes()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var target = rig.Lookup(ImageRole.ColorTarget);
        rig.Cache.MarkGpuWritten(target);
        var aliasRequest = R32AliasRequest(ImageRole.ColorTarget);
        var alias = rig.Cache.FindImage(ref aliasRequest);
        Assert.NotEqual(target, alias);
        rig.Cache.MarkGpuWritten(alias);

        var targetRequest = rig.Request(ImageRole.ColorTarget);
        Assert.Equal(target, rig.Cache.FindImage(ref targetRequest));
        rig.Cache.AcquireColorTargetView(target, in targetRequest);

        Assert.Equal(0, rig.Cache.TakeAliasNewerKept());
        // The alias's run reached the target: on the device, or through guest memory with
        // the target marked stale over it.
        Assert.True(rig.Cache.TakeDeviceAliasPublishes() == 1 || rig.Cache.GetImage(target).IsCpuDirty ||
            !rig.Cache.GetImage(alias).IsGpuModified);
        rig.Harness.Submit();
    }
}
