// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Images;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

// A GPU-written image whose every byte an alias has since written (four aliases of whole rows, one after another,
// each taking its run while the large image stays GPU-modified): stale over all of itself, it owes guest memory
// nothing. Its read must give the aliases' bytes, whether or not a CPU read published everything first, and its
// writeback must not put its older texels back over theirs. SHARPEMU_METAL_V2_WHOLE_STALE_RELOAD=0 (the property
// here) restores the old refusal, which kept those older texels.
public sealed unsafe partial class MetalImageAliasTests
{
    private sealed record WholeStaleOutcome(
        byte[] Large, byte[] Guest, byte[]? GuestBefore, bool GpuAfterRead, long Refused, long Reads,
        (long Reloads, long WritebacksSkipped) Whole, string Text);

    private static readonly (ulong Offset, uint Rows)[] WholeCoverDraws =
        [(0, AliasRows), (0x1000, AliasRows), (0x2000, AliasRows), (0x3000, AliasRows)];

    private static WholeStaleOutcome WholeStaleRead(bool reloadWhole, bool cpuReadFirst)
    {
        using var rig = new Rig(Device.Value);
        rig.Cache.AliasRunSetsEnabled = true;
        rig.Cache.WholeStaleReloadEnabled = reloadWhole;
        var (image, _, _) = GpuWrittenLarge(rig);
        byte seed = 0x11;
        foreach (var (offset, rows) in WholeCoverDraws)
        {
            DrawAlias(rig, offset, rows, seed);
            seed += 0x22;
        }

        var text = $"large_dirty={image.IsCpuDirty}:0x{image.DirtyStart:X}-0x{image.DirtyEnd:X} " +
            $"runs={image.StaleRunCount} gpu={image.IsGpuModified}";
        byte[]? before = null;
        if (cpuReadFirst)
        {
            rig.Cache.FlushGpuWrites(ImageAddress, LargeSize, MetalWritebackCause.HostRead);
            rig.Harness.Submit();
            rig.Harness.Begin();
            before = rig.ReadGuest(ImageAddress, LargeSize);
        }

        rig.Acquire(ImageAddress, LargeSide, ImageRole.Texture);
        rig.Harness.Submit();
        rig.Harness.Begin();
        var large = ReadTexture(rig, image);
        var gpuAfter = image.IsGpuModified;
        rig.Cache.FlushGpuWrites(ImageAddress, LargeSize, MetalWritebackCause.HostRead);
        rig.Harness.Submit();
        rig.Harness.Begin();
        return new WholeStaleOutcome(
            large, rig.ReadGuest(ImageAddress, LargeSize), before, gpuAfter, rig.Cache.AliasRunRefreshRefused,
            rig.Cache.RefusedStaleReads, rig.Cache.WholeStaleCounters, text + " " + rig.Cache.TakeAliasTransitionText());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AGpuWrittenImageStaleOverAllOfItsBytesReadsTheAliasesBytes(bool cpuReadFirst)
    {
        if (!RequireDevice()) return;
        var expected = ExpectedAfterDraws(WholeCoverDraws);
        var on = WholeStaleRead(reloadWhole: true, cpuReadFirst);
        var off = WholeStaleRead(reloadWhole: false, cpuReadFirst);
        output.WriteLine($"on: refused={on.Refused} reads={on.Reads} whole={on.Whole} gpu_after={on.GpuAfterRead} {on.Text}");
        output.WriteLine($"off: refused={off.Refused} reads={off.Reads} whole={off.Whole} gpu_after={off.GpuAfterRead} {off.Text}");
        Assert.Contains("large_dirty=True:0x0-0x4000", on.Text);
        Assert.Contains("gpu=True", on.Text);
        Assert.Equal(expected, on.Large);
        Assert.Equal(expected, on.Guest);
        if (on.GuestBefore is not null)
        {
            Assert.Equal(expected, on.GuestBefore);
        }

        Assert.False(on.GpuAfterRead);
        Assert.Equal(0, on.Refused);
        Assert.Equal(0, on.Reads);
        // Read first: the reload. A CPU read first: the large image's writeback writes nothing (guest memory already
        // holds the aliases' bytes), and the read then reloads it as an ordinary stale image.
        Assert.Equal(cpuReadFirst ? (0L, 1L) : (1L, 0L), on.Whole);
        // The switch off: the old refusal. Read first, it keeps the large texture's older texels; a CPU read first
        // refuses inside the writeback, whose older texels the aliases' own writebacks then happen to cover again.
        Assert.True(off.Refused >= 1);
        if (!cpuReadFirst)
        {
            Assert.NotEqual(expected, off.Large);
        }

        Assert.Equal((0L, 0L), off.Whole);
    }
}
