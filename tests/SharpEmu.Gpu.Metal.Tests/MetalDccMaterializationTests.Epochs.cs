// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Images;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// The image cache counts the mutations a memo of its state has to see (MetalImageEpochs):
/// each path below moves its count, and a lookup that changes nothing moves none. The
/// counts an image bumps itself are MetalImageEpochTests'; eviction is in
/// MetalImageBudgetTests.Epochs.cs.
/// </summary>
public sealed unsafe partial class MetalDccMaterializationTests
{
    private readonly record struct CacheCounts(long ImageSet, long Shape, long GpuModified, long Dcc, long Binding)
    {
        public static CacheCounts Of(MetalImageCache cache) =>
            new(cache.ImageSetVersion, cache.Epochs.Shape, cache.Epochs.GpuModified, cache.Epochs.Dcc, cache.Epochs.Binding);
    }

    [Fact]
    public void Epochs_ImageCreationCountsAndARepeatedLookupDoesNot()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var start = CacheCounts.Of(rig.Cache);

        rig.Lookup(ImageRole.ColorTarget);
        var created = CacheCounts.Of(rig.Cache);
        Assert.True(created.ImageSet > start.ImageSet);
        Assert.True(created.Binding > start.Binding);

        rig.Lookup(ImageRole.ColorTarget);
        rig.Lookup();
        rig.Lookup(ImageRole.ColorTarget);
        Assert.Equal(created, CacheCounts.Of(rig.Cache));
    }

    [Fact]
    public void Epochs_ImageReleaseCounts()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Lookup(ImageRole.ColorTarget);
        var before = CacheCounts.Of(rig.Cache);

        ((IGuestImageStore)rig.Cache).Unregister(ImageAddress, 64);

        var after = CacheCounts.Of(rig.Cache);
        Assert.True(after.ImageSet > before.ImageSet);
        Assert.True(after.Binding > before.Binding);
    }

    [Fact]
    public void Epochs_GpuWriteWritebackAndCpuStoreEachCount()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var target = rig.Lookup(ImageRole.ColorTarget);
        var created = CacheCounts.Of(rig.Cache);

        rig.Cache.MarkGpuWritten(target);
        var written = CacheCounts.Of(rig.Cache);
        Assert.Equal(created.GpuModified + 1, written.GpuModified);
        Assert.True(written.Binding > created.Binding);

        // Drawn again: still GPU-modified, nothing a binding sees moved.
        rig.Cache.MarkGpuWritten(target);
        Assert.Equal(written, CacheCounts.Of(rig.Cache));

        rig.Cache.FlushGpuWrites(ImageAddress, 64, MetalWritebackCause.HostRead);
        var writtenBack = CacheCounts.Of(rig.Cache);
        Assert.False(rig.Cache.GetImage(target).IsGpuModified);
        Assert.Equal(written.GpuModified, writtenBack.GpuModified);
        Assert.True(writtenBack.Binding > written.Binding);

        // The CPU store follows the real writeback; a diagnostic timeout no
        // longer admits it while this test's unsubmitted command buffer is pending.
        rig.Harness.Submit();
        rig.Harness.Begin();
        Assert.True(((IGuestImageStore)rig.Cache).MarkCpuWrite(ImageAddress, 4));
        Assert.True(rig.Cache.GetImage(target).IsCpuDirty);
        Assert.True(CacheCounts.Of(rig.Cache).Binding > writtenBack.Binding);
    }

    [Fact]
    public void Epochs_DccClaimDeferralAndReArmEachCount()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Lookup(ImageRole.ColorTarget);
        rig.FillMetadata(0x00);
        var before = CacheCounts.Of(rig.Cache);

        // The clear is claimed.
        rig.Lookup();
        var claimed = CacheCounts.Of(rig.Cache);
        Assert.Equal(1, rig.Owner.Clears);
        Assert.True(claimed.Dcc > before.Dcc);
        Assert.True(claimed.Binding > before.Binding);

        rig.Lookup();
        Assert.Equal(claimed.Dcc, rig.Cache.Epochs.Dcc);

        // A GPU store re-arms the claim.
        rig.Cache.InvalidateMemoryFromGpu(MetadataAddress, SliceSize);
        var rearmed = CacheCounts.Of(rig.Cache);
        Assert.True(rearmed.Dcc > claimed.Dcc);
        Assert.True(rearmed.Binding > claimed.Binding);

        // Met while still being written: deferred once, then nothing new.
        rig.Owner.Pending = true;
        rig.Lookup();
        var deferred = rig.Cache.Epochs.Dcc;
        Assert.True(deferred > rearmed.Dcc);
        rig.Lookup();
        Assert.Equal(deferred, rig.Cache.Epochs.Dcc);

        // A recorded fill counts too.
        rig.Cache.RecordDccFill(MetadataAddress, SliceSize, 0);
        Assert.True(rig.Cache.Epochs.Dcc > deferred);
    }

    [Fact]
    public void Epochs_ChangedMetadataCountsAsAShapeChange()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Lookup(ImageRole.ColorTarget);
        var before = CacheCounts.Of(rig.Cache);

        var request = rig.Request(ImageRole.ColorTarget);
        request.Description.Metadata.DccAlphaMsb = false;
        rig.Cache.FindImage(ref request);
        var changed = CacheCounts.Of(rig.Cache);
        Assert.Equal(before.Shape + 1, changed.Shape);
        Assert.True(changed.Binding > before.Binding);

        request = rig.Request(ImageRole.ColorTarget);
        request.Description.Metadata.DccAlphaMsb = false;
        rig.Cache.FindImage(ref request);
        Assert.Equal(changed, CacheCounts.Of(rig.Cache));
    }
}
