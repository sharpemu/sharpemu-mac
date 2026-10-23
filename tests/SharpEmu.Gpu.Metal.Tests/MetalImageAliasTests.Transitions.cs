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
}
