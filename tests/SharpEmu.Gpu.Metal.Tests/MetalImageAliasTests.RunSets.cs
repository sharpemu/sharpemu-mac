// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Images;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

// Run sets (SHARPEMU_METAL_V2_ALIAS_RUN_SETS): a GPU-written image holds up to four stale runs and four published ones
// instead of one interval each, and reloads a stale run alone however much of it the run is. Every case runs with run
// sets on and off and must end bit-identical: the large image's texture and every alias's texture read back, and
// guest memory after every GPU write is flushed. Neither arm may keep older texels over a stale run
// (alias_run_refresh_refused) or merge stale runs over bytes only the texture holds.
public sealed unsafe partial class MetalImageAliasTests
{
    private sealed record RunSetOutcome(
        byte[] Large, byte[] Guest, byte[]? GuestBeforeRead, List<byte[]> Aliases, long Writebacks,
        MetalImageCache.AliasRunCounts Runs, string Text);

    // The large 64x64 linear image written by the GPU (its texture holds one pattern, guest memory another), then
    // each alias of whole rows of it drawn as a colour target with a texture of its own, in order. Then, when asked,
    // a CPU read of all of it; then the large image is sampled, every texture read back and every GPU write flushed.
    private static RunSetOutcome RunAliasDraws(bool runSets, (ulong Offset, uint Rows)[] draws, bool cpuReadFirst = false)
    {
        using var rig = new Rig(Device.Value);
        rig.Cache.AliasRunSetsEnabled = runSets;
        var (image, _, _) = GpuWrittenLarge(rig);
        var writebacks = rig.Cache.Counters.Writebacks;
        var aliases = new List<MetalCachedImage>();
        byte seed = 0x11;
        foreach (var (offset, rows) in draws)
        {
            aliases.Add(DrawAlias(rig, offset, rows, seed));
            seed += 0x22;
        }

        byte[]? before = null;
        if (cpuReadFirst)
        {
            rig.Cache.FlushGpuWrites(ImageAddress, LargeSize, MetalWritebackCause.HostRead);
            rig.Harness.Submit();
            rig.Harness.Begin();
            before = rig.ReadGuest(ImageAddress, LargeSize);
        }

        var text = $"large_dirty={image.IsCpuDirty}:0x{image.DirtyStart:X}-0x{image.DirtyEnd:X} " +
            $"runs={image.StaleRunCount} gpu={image.IsGpuModified}";
        rig.Acquire(ImageAddress, LargeSide, ImageRole.Texture);
        rig.Harness.Submit();
        rig.Harness.Begin();
        var large = ReadTexture(rig, image);
        var aliasBytes = aliases.Select(alias => ReadTexture(rig, alias)).ToList();
        rig.Cache.FlushGpuWrites(ImageAddress, LargeSize, MetalWritebackCause.HostRead);
        rig.Harness.Submit();
        rig.Harness.Begin();
        writebacks = rig.Cache.Counters.Writebacks - writebacks;
        var runs = rig.Cache.AliasRunCounters;
        Assert.Equal(0, rig.Cache.AliasRunRefreshRefused);
        Assert.Equal(0, runs.Unpublished);
        return new RunSetOutcome(
            large, rig.ReadGuest(ImageAddress, LargeSize), before, aliasBytes, writebacks, runs,
            text + " " + rig.Cache.TakeAliasTransitionText());
    }

    // An alias of `rows` whole rows of the large image at `offset`, drawn: its texture given bytes of its own.
    private static MetalCachedImage DrawAlias(Rig rig, ulong offset, uint rows, byte seed)
    {
        var alias = rig.Cache.GetImage(
            rig.Acquire(Rig.Request(ImageAddress + offset, LargeSide, rows, ImageRole.ColorTarget)));
        Assert.True(alias.IsGpuModified);
        var size = alias.Description.Data.Size;
        var drawn = MetalTileOracle.Pattern((int)size, seed);
        rig.Harness.Transfer.EncodeGuestToTexture(
            rig.Harness, alias.Description, new SubresourceRange(0, 1, 0, 1),
            new MetalGuestRange(rig.Harness.Buffer(drawn), 0, size, 0), default, alias.Texture, 0, size);
        rig.Harness.Submit();
        rig.Harness.Begin();
        return alias;
    }

    private RunSetOutcome[] OnAndOff((ulong Offset, uint Rows)[] draws, bool cpuReadFirst = false)
    {
        var on = RunAliasDraws(runSets: true, draws, cpuReadFirst);
        var off = RunAliasDraws(runSets: false, draws, cpuReadFirst);
        output.WriteLine($"on: wb={on.Writebacks} runs={on.Runs} {on.Text}");
        output.WriteLine($"off: wb={off.Writebacks} runs={off.Runs} {off.Text}");
        Assert.Equal(off.Large, on.Large);
        Assert.Equal(off.Aliases.Count, on.Aliases.Count);
        for (var index = 0; index < on.Aliases.Count; index++)
        {
            Assert.Equal(off.Aliases[index], on.Aliases[index]);
        }

        Assert.Equal(off.Guest, on.Guest);
        Assert.Equal(off.GuestBeforeRead, on.GuestBeforeRead);
        return [on, off];
    }

    // Two aliases apart from each other take their runs from the large image, which stays GPU-modified and stale
    // over both (one interval cannot hold them apart: the off arm writes it back whole), then it is read.
    [Fact]
    public void TwoDisjointAliasesTakeRunsAndTheReadIsExact()
    {
        if (!RequireDevice()) return;
        var outcomes = OnAndOff([(0, AliasRows), (0x3000, AliasRows)]);
        Assert.True(outcomes[0].Runs.SetsMax >= 2);
        // On, the large image kept its texture's bytes outside the two runs; off, it was written back whole.
        Assert.Contains("runs=2 gpu=True", outcomes[0].Text);
        Assert.Contains("gpu=False", outcomes[1].Text);
    }

    // A run of 62.5% of the image: handed over and reloaded alone while the image stays GPU-modified (off: past half
    // is a decline, and the image is written back whole).
    [Fact]
    public void ARunOverSixtyPercentIsReloadedAloneWhileGpuModified()
    {
        if (!RequireDevice()) return;
        var outcomes = OnAndOff([(0, 40)]);
        Assert.True(outcomes[0].Runs.RefreshOverHalf >= 1);
        Assert.Equal(0, outcomes[1].Runs.RefreshOverHalf);
    }

    // Five aliases apart from each other: the fifth finds the run set full. It merges only by publishing the image
    // first (the whole writeback), never by widening a stale run over bytes only the texture holds.
    [Fact]
    public void AFullRunSetPublishesBeforeItMerges()
    {
        if (!RequireDevice()) return;
        var outcomes = OnAndOff([(0, 4), (0x800, 4), (0x1000, 4), (0x1800, 4), (0x2000, 4)]);
        Assert.Equal(MetalCachedImage.MaxRuns, (int)outcomes[0].Runs.SetsMax);
        Assert.True(outcomes[0].Runs.Merged >= 1);
    }

    // A takes a run, B takes half of A's run and more of the large image, then A is drawn again over its run while B
    // is GPU-modified: B publishes what they share first, and the large image is stale over the union of both.
    [Fact]
    public void AWriterThroughTheOldAliasAfterAHandOverIsExact()
    {
        if (!RequireDevice()) return;
        OnAndOff([(0, AliasRows), (0x800, AliasRows), (0, AliasRows)]);
    }

    // A CPU read of the whole image after two narrow hand-overs: guest memory gets each alias's run from the alias
    // and the rest from the large image's texture, and the large image reloads only its stale runs after.
    [Fact]
    public void ACpuReadAfterNarrowHandOversIsExact()
    {
        if (!RequireDevice()) return;
        OnAndOff([(0x1000, AliasRows), (0x3000, AliasRows / 2)], cpuReadFirst: true);
    }

    // The layer-subset copy (MetalImageAliasTests.Subset) into an array whose source is already stale apart from the
    // run (a one-layer alias took layer 8): with run sets the source holds both stale runs and the copy goes ahead.
    [Fact]
    public void ASubsetCopyProceedsOverASourceStaleApartFromTheRun()
    {
        if (!RequireDevice()) return;
        var outcomes = new List<(byte[] First, byte[] Second, byte[] Guest, long Copies)>();
        foreach (var runSets in new[] { true, false })
        {
            using var rig = new Rig(Device.Value);
            rig.Cache.AliasRunSetsEnabled = runSets;
            var tile = GuestTileMode.Standard64KB;
            var first = SubsetArray(ImageAddress, 9, tile, ImageRole.StorageImage);
            var size = first.Description.Data.Size;
            var layer = LayerBytes(tile);
            rig.WriteGuest(ImageAddress, MetalTileOracle.Pattern((int)size, 0x5A));
            var firstImage = rig.Cache.GetImage(rig.Acquire(first));
            rig.Harness.Submit();
            rig.Harness.Begin();
            rig.WriteGuest(ImageAddress, MetalTileOracle.Pattern((int)size, 0xC3));
            rig.Acquire(SubsetArray(ImageAddress + 8 * layer, 1, tile, ImageRole.StorageImage));
            rig.Harness.Submit();
            rig.Harness.Begin();
            Assert.True(firstImage.IsGpuModified);
            Assert.Equal((8 * layer, 9 * layer), (firstImage.DirtyStart, firstImage.DirtyEnd));

            var copies = rig.Cache.AliasSubsetCounters.Copies;
            var second = rig.Cache.GetImage(rig.Acquire(SubsetArray(ImageAddress, 4, tile, ImageRole.StorageImage)));
            rig.Harness.Submit();
            rig.Harness.Begin();
            copies = rig.Cache.AliasSubsetCounters.Copies - copies;
            rig.Acquire(first with { Role = ImageRole.Texture });
            rig.Harness.Submit();
            rig.Harness.Begin();
            var firstBytes = ReadLayers(rig, firstImage);
            var secondBytes = ReadLayers(rig, second);
            rig.Cache.FlushGpuWrites(ImageAddress, size, MetalWritebackCause.HostRead);
            rig.Harness.Submit();
            rig.Harness.Begin();
            Assert.Equal(0, rig.Cache.AliasRunRefreshRefused);
            Assert.Equal(0, rig.Cache.AliasRunCounters.Unpublished);
            outcomes.Add((firstBytes, secondBytes, rig.ReadGuest(ImageAddress, size), copies));
        }

        output.WriteLine($"on: copies={outcomes[0].Copies} off: copies={outcomes[1].Copies}");
        Assert.Equal(1, outcomes[0].Copies);
        Assert.Equal(0, outcomes[1].Copies);
        Assert.Equal(outcomes[1].First, outcomes[0].First);
        Assert.Equal(outcomes[1].Second, outcomes[0].Second);
        Assert.Equal(outcomes[1].Guest, outcomes[0].Guest);
    }
}
