// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// What run sets did (MetalCachedImage.StaleRuns): the most stale runs one image held, sets that were full and merged
/// by publishing the image first, reloads of a run over half an image, and merges over bytes only a texture held (a
/// bug: must stay 0).
/// </summary>
internal sealed class MetalRunSetStats
{
    private long _setsMax, _merged, _refreshOverHalf, _unpublished;

    public void NoteSets(int count)
    {
        long seen;
        while (count > (seen = Interlocked.Read(ref _setsMax)) &&
               Interlocked.CompareExchange(ref _setsMax, count, seen) != seen)
        {
        }
    }

    public void NoteMerged(bool unpublished = false)
    {
        Interlocked.Increment(ref _merged);
        if (unpublished)
        {
            Interlocked.Increment(ref _unpublished);
        }
    }

    public void NoteRefreshOverHalf() => Interlocked.Increment(ref _refreshOverHalf);

    public MetalImageCache.AliasRunCounts Read() =>
        new(Interlocked.Read(ref _setsMax), Interlocked.Read(ref _merged), Interlocked.Read(ref _refreshOverHalf),
            Interlocked.Read(ref _unpublished));

    public MetalImageCache.AliasRunCounts Take() =>
        new(Interlocked.Exchange(ref _setsMax, 0), Interlocked.Exchange(ref _merged, 0),
            Interlocked.Exchange(ref _refreshOverHalf, 0), Interlocked.Exchange(ref _unpublished, 0));
}

// SHARPEMU_METAL_V2_ALIAS_RUN_SETS (on; =0 off): a GPU-written image holds up to four stale runs and four published runs
// apart, instead of one interval each, and reloads a stale run alone however much of the image it is. An alias that
// takes a second run apart from the first, or a run over half of the image, used to cost a whole writeback of the
// image and a whole reload of the alias (Demon's Souls: the R32Uint 960x540 transient between its shadow array and the
// 72-layer array, the mip chain, the 1080p R16F pair). A set that is full is merged only by publishing the image first,
// the whole writeback as before, and counted. Off is the single interval exactly as before.
internal sealed partial class MetalImageCache
{
    internal readonly record struct AliasRunCounts(long SetsMax, long Merged, long RefreshOverHalf, long Unpublished);

    internal bool AliasRunSetsEnabled { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_ALIAS_RUN_SETS") != "0";

    private readonly MetalRunSetStats _runStats = new();

    /// <summary>Test hook: the run-set counters so far.</summary>
    internal AliasRunCounts AliasRunCounters => _runStats.Read();

    // A new image takes the setting the cache has when it admits it, for its whole life.
    private void AdmitRunSets(MetalCachedImage image)
    {
        image.RunCapacity = AliasRunSetsEnabled ? MetalCachedImage.MaxRuns : 1;
        image.RunStats = _runStats;
    }

    private string TakeAliasRunText()
    {
        var runs = _runStats.Take();
        return $"alias_runs=[sets_max:{runs.SetsMax},merged:{runs.Merged},refresh_over_half:{runs.RefreshOverHalf}," +
            $"unpublished:{runs.Unpublished}]";
    }

    // Every byte of [runStart, runEnd) is one guest memory holds for the image: inside a published run, or inside a
    // stale run (the image's, or [staleStart, staleEnd) about to be one), where guest memory is the newer copy.
    private static bool GuestHoldsImageBytes(
        MetalCachedImage image, ulong runStart, ulong runEnd, ulong staleStart, ulong staleEnd)
    {
        var at = runStart;
        while (at < runEnd)
        {
            var next = staleStart <= at && at < staleEnd ? staleEnd : at;
            foreach (var run in image.PublishedRuns)
            {
                if (run.Start <= at && at < run.End)
                {
                    next = Math.Max(next, run.End);
                }
            }

            foreach (var run in image.StaleRuns)
            {
                if (run.Start <= at && at < run.End)
                {
                    next = Math.Max(next, run.End);
                }
            }

            if (next == at)
            {
                return false;
            }

            at = next;
        }

        return true;
    }
}
