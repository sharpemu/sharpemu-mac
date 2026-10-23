// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Images;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

public sealed unsafe partial class MetalImageAliasTests
{
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ADeferredAliasDiscardsOnlyAfterAWholeOverwriteProof(bool overwrite)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var (source, gpu, stale) = GpuWrittenLarge(rig);
        var targetRequest = Rig.Request(ImageAddress, 128, 32, ImageRole.ColorTarget);
        var target = rig.Cache.FindImage(ref targetRequest);
        Assert.NotEqual(source, rig.Cache.GetImage(target));
        var writebacks = rig.Cache.Counters.Writebacks;
        var view = rig.Cache.AcquireColorTargetView(target, targetRequest, deferRefresh: true);
        Assert.True(rig.Cache.HasDeferredRefreshes);
        Assert.True(source.IsGpuModified);
        Assert.Equal(writebacks, rig.Cache.Counters.Writebacks);
        rig.Cache.ResolveDeferredRefresh(view, overwritten: overwrite);
        rig.Harness.Submit();
        rig.Harness.Begin();
        Assert.False(source.IsGpuModified);
        var transitions = rig.Cache.TakeAliasTransitions();
        if (overwrite)
        {
            Assert.Equal((1L, (long)LargeSize), (transitions.Discarded, transitions.DiscardedBytes));
            Assert.Equal(writebacks, rig.Cache.Counters.Writebacks);
            Assert.Equal(stale, rig.ReadGuest(ImageAddress, LargeSize));
            // Simulate the proven full draw with an actual GPU upload to the target.
            var drawn = MetalTileOracle.Pattern((int)LargeSize, 0xA2);
            var image = rig.Cache.GetImage(target);
            rig.Harness.Transfer.EncodeGuestToTexture(rig.Harness, image.Description, SubresourceRange.First,
                new MetalGuestRange(rig.Harness.Buffer(drawn), 0, LargeSize, 0), default, image.Texture, 0, LargeSize);
            rig.Cache.FlushGpuWrites(ImageAddress, LargeSize, MetalWritebackCause.HostRead);
            rig.Harness.Submit();
            rig.Harness.Begin();
            Assert.Equal(drawn, rig.ReadGuest(ImageAddress, LargeSize));
        }
        else
        {
            Assert.Equal(0, transitions.Discarded);
            rig.Cache.FlushGpuWrites(ImageAddress, LargeSize, MetalWritebackCause.HostRead);
            rig.Harness.Submit();
            rig.Harness.Begin();
            Assert.Equal(gpu, rig.ReadGuest(ImageAddress, LargeSize));
        }
    }

    [Fact]
    public void APartialCpuReadPublishesOnlyItsTransferableRowsAndKeepsTheRestOnDevice()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var (image, gpu, stale) = GpuWrittenLarge(rig);
        const ulong row = LargeSide * 4;
        rig.Cache.FlushGpuWrites(ImageAddress + row, row, MetalWritebackCause.HostRead);
        rig.Harness.Submit();
        rig.Harness.Begin();
        Assert.True(image.IsGpuModified);
        Assert.True(image.IsPublished(row, row * 2));
        var expected = stale.ToArray();
        gpu.AsSpan((int)row, (int)row).CopyTo(expected.AsSpan((int)row));
        Assert.Equal(expected, rig.ReadGuest(ImageAddress, LargeSize));
        var writebacks = rig.Cache.Counters.Writebacks;
        rig.Cache.FlushGpuWrites(ImageAddress + row, row, MetalWritebackCause.HostRead);
        Assert.Equal(writebacks, rig.Cache.Counters.Writebacks);
        rig.Cache.FlushGpuWrites(ImageAddress, LargeSize, MetalWritebackCause.HostRead);
        rig.Harness.Submit();
        rig.Harness.Begin();
        Assert.Equal(gpu, rig.ReadGuest(ImageAddress, LargeSize));
    }

    [Fact]
    public void ACompleteManagedOverwriteDiscardsWithoutPublishingAndReloadsExactBytes()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var (image, _, stale) = GpuWrittenLarge(rig);
        var writebacks = rig.Cache.Counters.Writebacks;
        Assert.True(((IGuestImageStore)rig.Cache).MarkPreciseCpuWrite(ImageAddress, LargeSize));
        var replacement = MetalTileOracle.Pattern((int)LargeSize, 0xB9);
        Assert.Equal(stale, rig.ReadGuest(ImageAddress, LargeSize));
        rig.WriteGuest(ImageAddress, replacement);
        ((IGuestImageStore)rig.Cache).MarkCpuWriteLanded(ImageAddress, LargeSize);
        Assert.False(image.IsGpuModified);
        Assert.Equal(writebacks, rig.Cache.Counters.Writebacks);
        Assert.Equal(1, rig.Cache.TakeAliasTransitions().Discarded);
        rig.Acquire(ImageAddress, LargeSide, ImageRole.ColorTarget);
        rig.Cache.FlushGpuWrites(ImageAddress, LargeSize, MetalWritebackCause.HostRead);
        rig.Harness.Submit();
        rig.Harness.Begin();
        Assert.Equal(replacement, rig.ReadGuest(ImageAddress, LargeSize));
    }

    // A GPU-written linear image and a 64 KiB-tiled image over exactly its bytes, both bound
    // for shader stores: with alias transitions on, the tiled one takes the linear one's
    // texels in one device pass; off, through a writeback of the linear one and an upload of
    // the tiled one. It holds the same guest bytes either way. Colour targets are created
    // compressible, without the shader-write and format-view usages the pass needs, and
    // keep the round trip. Prints the transfer counts and the GPU time.
    [Theory]
    [InlineData(true, true)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(false, false)]
    public void ATiledAliasOfAGpuWrittenImageFusesOnTheDeviceOrRoundTrips(bool fused, bool storage)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Cache.AliasTransitionsEnabled = fused;
        const uint side = 256;
        const ulong size = side * side * 4;
        var role = storage ? ImageRole.StorageImage : ImageRole.ColorTarget;
        var written = MetalTileOracle.Pattern((int)size, 0x3D);
        rig.WriteGuest(ImageAddress, written);
        var linear = Rig.Request(ImageAddress, side, ImageRole.ColorTarget);
        linear = new ImageRequest(linear.Description, linear.View, role);
        var source = rig.Cache.GetImage(rig.Acquire(linear));
        source.MarkGpuModified();
        rig.Harness.Submit();
        rig.Harness.Begin();
        rig.WriteGuest(ImageAddress, MetalTileOracle.Pattern((int)size, 0xC3));
        _ = rig.Cache.TakeAliasTransitionText();

        var tiled = Rig.TiledRequest(ImageAddress, side);
        Assert.Equal(size, tiled.Description.Data.Size);
        tiled = new ImageRequest(tiled.Description, tiled.View, role);
        var (_, uploads, writebacks, _) = rig.Cache.Counters;
        var buffer = rig.Harness.CommandBuffer;
        MetalNative.SendVoid(buffer, MetalNative.Selector("retain"));
        var alias = rig.Cache.GetImage(rig.Acquire(tiled));
        rig.Harness.Submit();
        var gpu = MetalNative.SendDouble(buffer, MetalNative.Selector("GPUEndTime")) -
            MetalNative.SendDouble(buffer, MetalNative.Selector("GPUStartTime"));
        MetalNative.SendVoid(buffer, MetalNative.Selector("release"));
        rig.Harness.Begin();
        var counts = rig.Cache.Counters;
        var text = rig.Cache.TakeAliasTransitionText();
        output.WriteLine($"alias transitions {(fused ? "on" : "off")}, {role}: uploads={counts.Uploads - uploads} " +
            $"writebacks={counts.Writebacks - writebacks} gpu={gpu * 1000:F3} ms {text}");
        if (fused && storage)
        {
            Assert.Contains("fused:1,", text);
            Assert.Equal((0L, 0L), (counts.Uploads - uploads, counts.Writebacks - writebacks));
        }
        else
        {
            Assert.Contains("fused:0,", text);
            Assert.Equal((1L, 1L), (counts.Uploads - uploads, counts.Writebacks - writebacks));
        }

        // The tiled image's texels, retiled to guest bytes, are the bytes the linear one held.
        var scratch = rig.Harness.Buffer(size);
        Assert.True(rig.Harness.Transfer.EncodeTextureToGuest(rig.Harness, alias.Description, SubresourceRange.First,
            alias.Texture, new MetalGuestRange(scratch, 0, size, 0), default, 0, size));
        rig.Harness.Submit();
        rig.Harness.Begin();
        Assert.Equal(written, rig.Harness.Read(scratch, 0, size));
    }
}
