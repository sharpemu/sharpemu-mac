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
        // The precise mode: the GPU-written image's bytes go out through guest memory unless a proof drops them.
        rig.Cache.AliasWriteSkipLayout = false;
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

    // SHARPEMU_METAL_V2_ALIAS_WRITE_SKIP_LAYOUT on (the default): the target writes over all of the
    // GPU-written image through a texel layout its bytes do not fit, so that image's contents are dropped
    // when the deferred refresh resolves, proof or not. Without the proof this loses the GPU-written bytes: guest
    // memory keeps what it held before the image was drawn. That is the trade-off the switch makes - the
    // console would read such bytes back scrambled through the other layout anyway - for not writing the
    // image back and reloading it.
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ADeferredAliasThroughAnotherLayoutIsDroppedWithoutAProofInTheDefaultMode(bool overwrite)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Cache.AliasWriteSkipLayout = true;
        var (source, gpu, stale) = GpuWrittenLarge(rig);
        _ = rig.Cache.TakeAliasWritesDiscarded();
        var targetRequest = Rig.Request(ImageAddress, 128, 32, ImageRole.ColorTarget);
        var target = rig.Cache.FindImage(ref targetRequest);
        var writebacks = rig.Cache.Counters.Writebacks;
        var view = rig.Cache.AcquireColorTargetView(target, targetRequest, deferRefresh: true);
        Assert.True(rig.Cache.HasDeferredRefreshes);
        Assert.True(source.IsGpuModified);
        Assert.Equal(0, rig.Cache.TakeAliasWritesDiscarded());
        rig.Cache.ResolveDeferredRefresh(view, overwritten: overwrite);
        rig.Harness.Submit();
        rig.Harness.Begin();
        // Without the proof the refresh takes the layout skip (an alias write discarded); with it, the proof's discard.
        Assert.Equal(overwrite ? 0 : 1, rig.Cache.TakeAliasWritesDiscarded());
        Assert.False(source.IsGpuModified);
        Assert.Equal(writebacks, rig.Cache.Counters.Writebacks);
        var transitions = rig.Cache.TakeAliasTransitions();
        if (overwrite)
        {
            // The proof drops it in either mode.
            Assert.Equal((1L, (long)LargeSize), (transitions.Discarded, transitions.DiscardedBytes));
            Assert.Equal(stale, rig.ReadGuest(ImageAddress, LargeSize));
        }
        else
        {
            // No proof, and still no writeback: the GPU-written bytes are gone, guest memory holds the stale ones.
            Assert.Equal(0, transitions.Discarded);
            rig.Cache.FlushGpuWrites(ImageAddress, LargeSize, MetalWritebackCause.HostRead);
            rig.Harness.Submit();
            rig.Harness.Begin();
            Assert.Equal(stale, rig.ReadGuest(ImageAddress, LargeSize));
            Assert.NotEqual(gpu, rig.ReadGuest(ImageAddress, LargeSize));
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
    [InlineData(true, true, false)]
    [InlineData(false, true, false)]
    [InlineData(true, false, false)]
    [InlineData(false, false, false)]
    [InlineData(true, true, true)]
    [InlineData(false, true, true)]
    [InlineData(true, false, true)]
    [InlineData(false, false, true)]
    public void ATiledAliasOfAGpuWrittenImageFusesOnTheDeviceOrRoundTrips(bool fused, bool storage, bool skipLayout)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Cache.AliasTransitionsEnabled = fused;
        rig.Cache.AliasWriteSkipLayout = skipLayout;
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
        var stale = MetalTileOracle.Pattern((int)size, 0xC3);
        rig.WriteGuest(ImageAddress, stale);
        _ = rig.Cache.TakeAliasTransitionText();
        _ = rig.Cache.TakeAliasWritesDiscarded();

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
        var discarded = rig.Cache.TakeAliasWritesDiscarded();
        output.WriteLine($"alias transitions {(fused ? "on" : "off")}, {role}, skip layout {skipLayout}: " +
            $"uploads={counts.Uploads - uploads} writebacks={counts.Writebacks - writebacks} " +
            $"alias_writes_discarded={discarded} gpu={gpu * 1000:F3} ms {text}");
        var dropped = false;
        if (fused && storage)
        {
            Assert.Contains("fused:1,", text);
            Assert.Equal((0L, 0L), (counts.Uploads - uploads, counts.Writebacks - writebacks));
            Assert.Equal(0, discarded);
        }
        else if (skipLayout)
        {
            // SHARPEMU_METAL_V2_ALIAS_WRITE_SKIP_LAYOUT on (the default): the tiled image writes over all of
            // the linear one through a layout its texels do not fit, so the linear one's contents are dropped,
            // not written back: the tiled one loads what guest memory held before the linear one was drawn.
            // That is the trade-off the switch makes for skipping the round trip.
            Assert.Contains("fused:0,", text);
            Assert.Equal((1L, 0L), (counts.Uploads - uploads, counts.Writebacks - writebacks));
            Assert.Equal(1, discarded);
            dropped = true;
        }
        else
        {
            Assert.Contains("fused:0,", text);
            Assert.Equal((1L, 1L), (counts.Uploads - uploads, counts.Writebacks - writebacks));
            Assert.Equal(0, discarded);
        }

        // The tiled image's texels, retiled to guest bytes, are the bytes the linear one held - or, dropped,
        // the bytes guest memory held.
        var scratch = rig.Harness.Buffer(size);
        Assert.True(rig.Harness.Transfer.EncodeTextureToGuest(rig.Harness, alias.Description, SubresourceRange.First,
            alias.Texture, new MetalGuestRange(scratch, 0, size, 0), default, 0, size));
        rig.Harness.Submit();
        rig.Harness.Begin();
        Assert.Equal(dropped ? stale : written, rig.Harness.Read(scratch, 0, size));
    }
}
