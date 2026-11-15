// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Images;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// The stale-run and published-run sets of a cached image (MetalCachedImage.StaleRuns / PublishedRuns, at most
/// MaxRuns of each with run sets on, one interval each with them off), exercised on an image with no texture: no
/// device is needed. Each case pins what the code does today, including what a full set does.
/// </summary>
public sealed class MetalCachedImageRunSetTests
{
    private const ulong Size = 0x10000;

    private static MetalCachedImage CreateImage(int capacity, out MetalRunSetStats stats, bool clean = true)
    {
        var description = ImageDescription.Create();
        description.PixelFormat = GpuFormat.R8G8B8A8Unorm;
        description.BytesPerBlock = 4;
        description.Pitch = 1;
        description.Data = new GuestSpan(0x2_0000_0000, Size);
        var shape = new MetalTextureShape(
            MtlFormat.Rgba8Unorm, Mtl.TextureType2D, 1, 1, 1, 1, 1, 1, Mtl.TextureUsageShaderRead);
        stats = new MetalRunSetStats();
        var image = new MetalCachedImage(in description, in shape, 0) { RunCapacity = capacity, RunStats = stats };
        if (clean)
        {
            // A new image is stale over all of it; the cases below start from a current one.
            image.RefreshComplete();
        }

        return image;
    }

    private static MetalCachedImage Clean(int capacity = MetalCachedImage.MaxRuns) => CreateImage(capacity, out _);

    private static (ulong Start, ulong End)[] Stale(MetalCachedImage image) => image.StaleRuns.ToArray();

    private static (ulong Start, ulong End)[] Published(MetalCachedImage image) => image.PublishedRuns.ToArray();

    private static void Dirty(MetalCachedImage image, ulong start, ulong end) =>
        image.MarkCpuDirty(MetalImageDirtyCause.CpuWrite, start, end);

    // ---- stale runs: marking ----

    [Fact]
    public void TouchingStaleRunsMergeIntoOne()
    {
        var image = Clean();
        Dirty(image, 0x1000, 0x2000);
        Dirty(image, 0x2000, 0x3000);
        Assert.Equal([(0x1000UL, 0x3000UL)], Stale(image));

        // Touching at the front as well.
        Dirty(image, 0x0800, 0x1000);
        Assert.Equal([(0x0800UL, 0x3000UL)], Stale(image));
    }

    [Fact]
    public void RunsOneByteApartStayApart()
    {
        var image = Clean();
        Dirty(image, 0x1000, 0x2000);
        Dirty(image, 0x2001, 0x3000);
        Assert.Equal([(0x1000UL, 0x2000UL), (0x2001UL, 0x3000UL)], Stale(image));
    }

    [Fact]
    public void OverlappingStaleRunsMergeAndAWideRunAbsorbsEveryRunItCovers()
    {
        var image = Clean();
        Dirty(image, 0x1000, 0x3000);
        Dirty(image, 0x2000, 0x4000);
        Assert.Equal([(0x1000UL, 0x4000UL)], Stale(image));

        Dirty(image, 0x8000, 0x9000);
        Dirty(image, 0xA000, 0xB000);
        Assert.Equal(3, image.StaleRunCount);
        Dirty(image, 0x3000, 0xA800);
        Assert.Equal([(0x1000UL, 0xB000UL)], Stale(image));
    }

    [Fact]
    public void DisjointStaleRunsStayApartAndSorted()
    {
        var image = Clean();
        Dirty(image, 0x8000, 0x9000);
        Dirty(image, 0x1000, 0x2000);
        Dirty(image, 0x4000, 0x5000);
        Assert.Equal([(0x1000UL, 0x2000UL), (0x4000UL, 0x5000UL), (0x8000UL, 0x9000UL)], Stale(image));
        Assert.Equal(0x1000UL, image.DirtyStart);
        Assert.Equal(0x9000UL, image.DirtyEnd);
        Assert.True(image.IsStaleOver(0x4000, 0x5000));
        Assert.False(image.IsStaleOver(0x1000, 0x5000));
        Assert.True(image.StaleOverlaps(0x1800, 0x4800));
        Assert.False(image.StaleOverlaps(0x2000, 0x4000));
    }

    [Fact]
    public void AStaleRunInsideARunChangesNothing()
    {
        var image = Clean();
        Dirty(image, 0x1000, 0x5000);
        var version = image.BindingVersion;
        Dirty(image, 0x2000, 0x3000);
        Assert.Equal([(0x1000UL, 0x5000UL)], Stale(image));
        Assert.Equal(version, image.BindingVersion);
    }

    [Fact]
    public void WithRunSetsOffASecondStaleRunWidensTheOneInterval()
    {
        var image = Clean(capacity: 1);
        Dirty(image, 0x1000, 0x2000);
        Dirty(image, 0x8000, 0x9000);
        Assert.Equal([(0x1000UL, 0x9000UL)], Stale(image));
    }

    [Fact]
    public void ACleanImageMarkedStaleHoldsExactlyThatRun()
    {
        var image = Clean();
        Assert.False(image.IsCpuDirty);
        Dirty(image, 0x2000, 0x2000 + 0x100);
        Assert.True(image.IsCpuDirty);
        Assert.Equal([(0x2000UL, 0x2100UL)], Stale(image));
        // Marked past the end: clamped to the image; an empty range marks nothing.
        var other = Clean();
        Dirty(other, Size - 0x100, Size + 0x4000);
        Assert.Equal([(Size - 0x100, Size)], Stale(other));
        var none = Clean();
        Dirty(none, Size, Size + 0x100);
        Assert.False(none.IsCpuDirty);
    }

    // ---- stale runs: a fifth disjoint run on a full set ----

    private static MetalCachedImage FullStaleSet(out MetalRunSetStats stats, bool gpuModified = false)
    {
        var image = CreateImage(MetalCachedImage.MaxRuns, out stats);
        if (gpuModified)
        {
            image.MarkGpuModified();
        }

        Dirty(image, 0x0000, 0x0100);
        Dirty(image, 0x1000, 0x1100);
        Dirty(image, 0x2000, 0x2100);
        Dirty(image, 0x3000, 0x3100);
        Assert.Equal(MetalCachedImage.MaxRuns, image.StaleRunCount);
        return image;
    }

    // The documented outcome of MetalCachedImage.MergeClosest: the two runs with the least between them become one
    // and the bytes between them are marked stale too. The cache never lets a GPU-written image get here
    // (CanAddStaleRun), so for an image the texture owns nothing of it is exact, and counted.
    [Fact]
    public void AFifthDisjointStaleRunMergesTheTwoClosestAndCountsIt()
    {
        var image = FullStaleSet(out var stats);
        Assert.False(image.CanAddStaleRun(0x5000, 0x5100));

        Dirty(image, 0x3180, 0x3200);

        Assert.Equal(
            [(0x0000UL, 0x0100UL), (0x1000UL, 0x1100UL), (0x2000UL, 0x2100UL), (0x3000UL, 0x3200UL)], Stale(image));
        var counts = stats.Read();
        Assert.Equal(1, counts.Merged);
        Assert.Equal(0, counts.Unpublished);
        Assert.Equal(MetalCachedImage.MaxRuns, (int)counts.SetsMax);
    }

    // With every gap the same the earlier pair is merged (the closest is replaced only by a strictly smaller gap).
    [Fact]
    public void AFifthStaleRunWithEveryGapEqualMergesTheFirstPair()
    {
        var image = FullStaleSet(out var stats);

        Dirty(image, 0x4000, 0x4100);

        Assert.Equal([(0x0000UL, 0x1100UL), (0x2000UL, 0x2100UL), (0x3000UL, 0x3100UL), (0x4000UL, 0x4100UL)], Stale(image));
        Assert.Equal(1, stats.Read().Merged);
    }

    // A GPU-written image whose set is full and is merged anyway: counted as an unpublished merge (the bug counter
    // the run-set tests keep at 0), and the bytes between the two runs are stale now.
    [Fact]
    public void AMergeOnAGpuWrittenImageCountsAsUnpublished()
    {
        var image = FullStaleSet(out var stats, gpuModified: true);
        Assert.True(image.IsGpuModified);

        Dirty(image, 0x4000, 0x4100);

        Assert.Equal(4, image.StaleRunCount);
        var counts = stats.Read();
        Assert.Equal(1, counts.Merged);
        Assert.Equal(1, counts.Unpublished);
    }

    // ---- stale runs: cutting ----

    [Fact]
    public void AFreshImageRefreshedOverItsFrontLeavesTheRest()
    {
        var image = CreateImage(MetalCachedImage.MaxRuns, out _, clean: false);
        Assert.Equal([(0UL, Size)], Stale(image));
        image.RefreshComplete(0, 0x4000);
        Assert.Equal([(0x4000UL, Size)], Stale(image));
        Assert.True(image.IsCpuDirty);
    }

    [Fact]
    public void AFreshImageRefreshedOverItsBackLeavesTheRest()
    {
        var image = CreateImage(MetalCachedImage.MaxRuns, out _, clean: false);
        image.RefreshComplete(Size - 0x4000, Size);
        Assert.Equal([(0UL, Size - 0x4000)], Stale(image));
    }

    [Fact]
    public void AFreshImageRefreshedOverItsMiddleHoldsTwoRuns()
    {
        var image = CreateImage(MetalCachedImage.MaxRuns, out var stats, clean: false);
        image.RefreshComplete(0x4000, 0x6000);
        Assert.Equal([(0UL, 0x4000UL), (0x6000UL, Size)], Stale(image));
        Assert.Equal(2, (int)stats.Read().SetsMax);
        Assert.True(image.IsStaleOver(0, 0x4000));
        Assert.False(image.IsStaleOver(0x3000, 0x7000));
    }

    [Fact]
    public void ARefreshOverAWholeImageOrAllItsRunsLeavesNothingStale()
    {
        var fresh = CreateImage(MetalCachedImage.MaxRuns, out _, clean: false);
        fresh.RefreshComplete(0, Size);
        Assert.False(fresh.IsCpuDirty);
        Assert.Equal(0, fresh.StaleRunCount);

        // The span of every run clears them all, whatever lies between them.
        var image = Clean();
        Dirty(image, 0x2000, 0x3000);
        Dirty(image, 0x6000, 0x7000);
        image.RefreshComplete(0x1000, 0x8000);
        Assert.False(image.IsCpuDirty);
        Assert.Equal(0, image.StaleRunCount);
    }

    [Fact]
    public void ARefreshOnACurrentImageIsIgnoredAndOneThatMissesAllRunsChangesNothing()
    {
        var current = Clean();
        var contentVersion = current.ContentVersion;
        current.RefreshComplete(0, 0x1000);
        Assert.Equal(contentVersion, current.ContentVersion);
        Assert.False(current.IsCpuDirty);

        var image = Clean();
        Dirty(image, 0x2000, 0x3000);
        Dirty(image, 0x6000, 0x7000);
        var version = image.BindingVersion;
        image.RefreshComplete(0x4000, 0x5000);
        Assert.Equal([(0x2000UL, 0x3000UL), (0x6000UL, 0x7000UL)], Stale(image));
        Assert.Equal(version, image.BindingVersion);
    }

    // With one interval a cut from the middle keeps the whole run (nothing becomes current that was not refreshed);
    // a cut from the front or the back narrows it.
    [Fact]
    public void WithRunSetsOffAMiddleCutKeepsTheWholeRunAndAnEdgeCutNarrowsIt()
    {
        var middle = CreateImage(1, out _, clean: false);
        middle.RefreshComplete(0x4000, 0x6000);
        Assert.Equal([(0UL, Size)], Stale(middle));
        Assert.False(middle.CanCutStale(0x4000, 0x6000));

        var front = CreateImage(1, out _, clean: false);
        front.RefreshComplete(0, 0x4000);
        Assert.Equal([(0x4000UL, Size)], Stale(front));

        var back = CreateImage(1, out _, clean: false);
        back.RefreshComplete(0xC000, Size + 0x1000);
        Assert.Equal([(0UL, 0xC000UL)], Stale(back));
    }

    // The set is full: a middle cut would make a fifth run, so the run is kept whole. Exactly what the code does,
    // and the invariant that matters: no byte that was stale and not refreshed stops being stale.
    [Fact]
    public void AMiddleCutOnAFullSetKeepsTheRunWholeAndLosesNoStaleByte()
    {
        var image = Clean();
        Dirty(image, 0x0000, 0x1000);
        Dirty(image, 0x2000, 0x3000);
        Dirty(image, 0x4000, 0x5000);
        Dirty(image, 0x6000, 0xF000);
        Assert.Equal(MetalCachedImage.MaxRuns, image.StaleRunCount);
        Assert.False(image.CanCutStale(0x8000, 0x9000));
        Assert.True(image.CanCutStale(0x6000, 0x7000));
        var before = Stale(image);

        image.RefreshComplete(0x8000, 0x9000);

        Assert.Equal(before, Stale(image));
        Assert.True(image.IsStaleOver(0x8000, 0x9000));
    }

    [Theory]
    [InlineData(0x0800UL, 0x0900UL)]
    [InlineData(0x8000UL, 0x9000UL)]
    [InlineData(0x0000UL, 0x0400UL)]
    [InlineData(0x0C00UL, 0x2400UL)]
    [InlineData(0x2800UL, 0x6800UL)]
    [InlineData(0xE000UL, 0x10000UL)]
    public void AnyCutOnAFullSetNeverMakesAnUnrefreshedStaleByteCurrent(ulong start, ulong end)
    {
        var image = Clean();
        Dirty(image, 0x0000, 0x1000);
        Dirty(image, 0x2000, 0x3000);
        Dirty(image, 0x4000, 0x5000);
        Dirty(image, 0x6000, 0xF000);
        var before = Stale(image);

        image.RefreshComplete(start, end);

        for (var at = 0UL; at < Size; at += 0x100)
        {
            var wasStale = before.Any(run => run.Start <= at && at < run.End);
            var refreshed = start <= at && at < end;
            if (wasStale && !refreshed)
            {
                Assert.True(image.StaleOverlaps(at, at + 1), $"byte 0x{at:X} was stale, not refreshed, and is current");
            }

            if (!wasStale)
            {
                Assert.False(image.StaleOverlaps(at, at + 1), $"byte 0x{at:X} was current and became stale");
            }
        }
    }

    [Fact]
    public void ACutWithRoomMakesTwoRunsAndNoStaleByteBecomesCurrent()
    {
        var image = Clean();
        Dirty(image, 0x2000, 0x8000);
        Dirty(image, 0xA000, 0xB000);
        Assert.True(image.CanCutStale(0x4000, 0x5000));

        image.RefreshComplete(0x4000, 0x5000);

        Assert.Equal([(0x2000UL, 0x4000UL), (0x5000UL, 0x8000UL), (0xA000UL, 0xB000UL)], Stale(image));
    }

    // ---- published runs ----

    [Fact]
    public void TouchingAndOverlappingPublishedRunsMergeAndDisjointOnesStayApartSorted()
    {
        var image = Clean();
        image.MarkGpuModified();
        image.MarkPublished(0x1000, 0x2000);
        image.MarkPublished(0x2000, 0x3000);
        Assert.Equal([(0x1000UL, 0x3000UL)], Published(image));
        image.MarkPublished(0x2800, 0x4000);
        Assert.Equal([(0x1000UL, 0x4000UL)], Published(image));
        image.MarkPublished(0xA000, 0xB000);
        image.MarkPublished(0x6000, 0x7000);
        Assert.Equal([(0x1000UL, 0x4000UL), (0x6000UL, 0x7000UL), (0xA000UL, 0xB000UL)], Published(image));
        Assert.Equal(0x1000UL, image.PublishedStart);
        Assert.Equal(0xB000UL, image.PublishedEnd);
        Assert.False(image.IsPublished(0x1000, 0x7000));
    }

    // Four runs and a fifth: the shortest is forgotten, a tie dropping the later run. Forgetting a run only costs
    // publishing it again.
    [Fact]
    public void AFifthPublishedRunOnAFullSetForgetsTheShortestAndTheLaterOnATie()
    {
        var image = Clean();
        image.MarkGpuModified();
        image.MarkPublished(0x0000, 0x0100);
        image.MarkPublished(0x1000, 0x1200);
        image.MarkPublished(0x2000, 0x2300);
        image.MarkPublished(0x3000, 0x3400);

        image.MarkPublished(0x4000, 0x4100);

        Assert.Equal(
            [(0x0000UL, 0x0100UL), (0x1000UL, 0x1200UL), (0x2000UL, 0x2300UL), (0x3000UL, 0x3400UL)], Published(image));

        image.MarkPublished(0x5000, 0x5800);

        Assert.Equal(
            [(0x1000UL, 0x1200UL), (0x2000UL, 0x2300UL), (0x3000UL, 0x3400UL), (0x5000UL, 0x5800UL)], Published(image));
    }

    [Fact]
    public void WithRunSetsOffTwoDisjointPublishedRunsKeepTheLongerOne()
    {
        var image = Clean(capacity: 1);
        image.MarkGpuModified();
        image.MarkPublished(0x1000, 0x1400);
        image.MarkPublished(0x8000, 0x8100);
        Assert.Equal([(0x1000UL, 0x1400UL)], Published(image));
        image.MarkPublished(0x8000, 0x9000);
        Assert.Equal([(0x8000UL, 0x9000UL)], Published(image));
    }

    private static MetalCachedImage PublishedAt(int capacity, params (ulong Start, ulong End)[] runs)
    {
        var image = Clean(capacity);
        image.MarkGpuModified();
        foreach (var (start, end) in runs)
        {
            image.MarkPublished(start, end);
        }

        return image;
    }

    [Fact]
    public void ForgettingOverTheStartOfAPublishedRunKeepsTheRest()
    {
        var image = PublishedAt(4, (0x2000, 0x6000));
        image.ForgetPublished(0x1000, 0x3000);
        Assert.Equal([(0x3000UL, 0x6000UL)], Published(image));
        Assert.False(image.IsPublished(0x2000, 0x3000));
        Assert.True(image.IsPublished(0x3000, 0x6000));
    }

    [Fact]
    public void ForgettingOverTheEndOfAPublishedRunKeepsTheRest()
    {
        var image = PublishedAt(4, (0x2000, 0x6000));
        image.ForgetPublished(0x5000, 0x7000);
        Assert.Equal([(0x2000UL, 0x5000UL)], Published(image));
    }

    [Fact]
    public void ForgettingTheMiddleOfAPublishedRunMakesTwoRuns()
    {
        var image = PublishedAt(4, (0x2000, 0x6000));
        image.ForgetPublished(0x3000, 0x4000);
        Assert.Equal([(0x2000UL, 0x3000UL), (0x4000UL, 0x6000UL)], Published(image));
    }

    [Fact]
    public void ForgettingAllOfAPublishedRunOrMoreRemovesIt()
    {
        var image = PublishedAt(4, (0x2000, 0x6000), (0x8000, 0x9000));
        image.ForgetPublished(0x1000, 0x7000);
        Assert.Equal([(0x8000UL, 0x9000UL)], Published(image));
        image.ForgetPublished(0, Size);
        Assert.Equal(0, image.PublishedRuns.Length);
    }

    [Theory]
    [InlineData(0x0000UL, 0x2000UL)]
    [InlineData(0x6000UL, 0x7000UL)]
    [InlineData(0x7000UL, 0x8000UL)]
    public void ForgettingARangeThatTouchesNothingChangesNothing(ulong start, ulong end)
    {
        var image = PublishedAt(4, (0x2000, 0x6000));
        var version = image.BindingVersion;
        image.ForgetPublished(start, end);
        Assert.Equal([(0x2000UL, 0x6000UL)], Published(image));
        Assert.Equal(version, image.BindingVersion);
    }

    // Looks wrong: an empty range inside a run is not "nothing" - it splits the run in two touching runs, which
    // the sorted-and-not-touching invariant of the set forbids. No caller passes an empty range today.
    [Fact]
    public void ForgettingAnEmptyRangeInsideARunSplitsItIntoTwoTouchingRuns()
    {
        var image = PublishedAt(4, (0x2000, 0x6000));
        image.ForgetPublished(0x3000, 0x3000);
        Assert.Equal([(0x2000UL, 0x3000UL), (0x3000UL, 0x6000UL)], Published(image));
    }

    // A middle cut on a full set makes a fifth run: the shortest of the five is forgotten, a tie dropping the
    // later one. Here the cut run's own two halves are the shortest.
    [Fact]
    public void ForgettingTheMiddleOfARunOnAFullSetForgetsTheShortestRun()
    {
        var image = PublishedAt(
            4, (0x0000, 0x1000), (0x2000, 0x3000), (0x4000, 0x5000), (0x6000, 0xA000));
        image.ForgetPublished(0x7000, 0x9000);
        Assert.Equal(
            [(0x0000UL, 0x1000UL), (0x2000UL, 0x3000UL), (0x4000UL, 0x5000UL), (0x6000UL, 0x7000UL)], Published(image));

        // The half dropped is the later of two equal runs: (0x9000, 0xA000) not (0x6000, 0x7000).
        Assert.False(image.IsPublished(0x9000, 0xA000));
    }

    [Fact]
    public void WithRunSetsOffForgettingTheMiddleKeepsTheLongerSideAndATieTheEarlierOne()
    {
        var image = PublishedAt(1, (0x2000, 0x6000));
        image.ForgetPublished(0x2800, 0x3000);
        Assert.Equal([(0x3000UL, 0x6000UL)], Published(image));

        var longer = PublishedAt(1, (0x2000, 0x6000));
        longer.ForgetPublished(0x3000, 0x4000);
        // Sides 0x1000 and 0x2000: the longer.
        Assert.Equal([(0x4000UL, 0x6000UL)], Published(longer));

        var equal = PublishedAt(1, (0x2000, 0x5000));
        equal.ForgetPublished(0x3000, 0x4000);
        Assert.Equal([(0x2000UL, 0x3000UL)], Published(equal));
    }

    [Fact]
    public void AGpuWriteDropsEveryStaleAndPublishedRun()
    {
        var image = Clean();
        image.MarkGpuModified();
        image.MarkPublished(0x1000, 0x2000);
        Dirty(image, 0x4000, 0x5000);
        Dirty(image, 0x8000, 0x9000);
        image.MarkGpuModified();
        Assert.Equal(0, image.StaleRunCount);
        Assert.False(image.IsCpuDirty);
        Assert.Equal(0, image.PublishedRuns.Length);
    }
}
