// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

// The meshlet cache at its byte budget (128 MiB: MetalMeshletCache.BudgetBytes) and its free-buffer pool (32 MiB:
// PoolBudgetBytes). Neither can be lowered, so these builds are real: a 900,000-index strip is a 5 MiB class of
// buffer, 25 of them fill the budget. Keys of one range differ by the primitive limit (32 + key), which does not
// change the words built, so every key's entry has the same size.
public sealed unsafe partial class MetalMeshletCacheTests
{
    private const ulong Mib = 1024 * 1024;
    private const ulong CacheBudget = 128 * Mib;
    private const ulong PoolBudget = 32 * Mib;
    private const int BigIndices = 900_000;
    private const int SmallIndices = 450_000;
    private const ulong BudgetRange = MetalGuestWindowGranuleTests.GuestBase;

    private static uint[] Big(uint offset) => Strip(offset, BigIndices / 3);

    private static ulong ClassOf(uint[] indices, uint maxPrimitives = 32) =>
        MetalMeshletCache.SizeClass((ulong)MeshletBuilder.Build(indices, MaxVertices, maxPrimitives).Words.Length * sizeof(uint));

    private static readonly Lazy<ulong> BigBytes = new(() => ClassOf(Big(0)));
    private static readonly Lazy<ulong> SmallBytes = new(() => ClassOf(Strip(0, SmallIndices / 3)));

    // Entries that fill the budget exactly as far as another one would not fit.
    private static int Fit() => (int)(CacheBudget / BigBytes.Value);

    private static bool TryGetBig(Rig rig, int key, ulong serial, ulong completed, out MetalMeshletCache.Meshlets meshlets) =>
        rig.Cache.TryGet(BudgetRange, BigIndices, true, MaxVertices, (uint)(32 + key), serial, completed, out meshlets);

    private static MetalMeshletCache.Meshlets GetBig(Rig rig, int key, ulong serial, ulong completed)
    {
        Assert.True(TryGetBig(rig, key, serial, completed, out var meshlets));
        return meshlets;
    }

    private static long Builds(Rig rig) =>
        rig.Cache.Builds(MetalMeshletCache.BuildReason.New) + rig.Cache.Builds(MetalMeshletCache.BuildReason.Changed) +
        rig.Cache.Builds(MetalMeshletCache.BuildReason.Recurring);

    private static long Counter(string counters, string name)
    {
        var start = counters.IndexOf(name + "=", StringComparison.Ordinal);
        Assert.True(start >= 0, $"{name} is not in the counters line");
        start += name.Length + 1;
        var end = counters.IndexOf(' ', start);
        if (end < 0) end = counters.Length;
        return long.Parse(counters.AsSpan(start, end - start));
    }

    // Distinct keys of one range, each in its own submission with the one before completed: the entries nothing can
    // read any more are the least recently used first. The one that does not fit evicts the oldest, the cache reuses
    // its buffer from the pool, the byte total follows every change, and a hit on an entry makes it the newest.
    [Fact]
    public void BudgetEvictionTakesTheLeastRecentlyUsedEntryFirstAndKeepsTheByteTotal()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(maxVersions: 4, pool: true);
        rig.Write(Big(0), BudgetRange);
        var fit = Fit();
        Assert.True(fit >= 8, $"{fit} entries fit");
        for (var key = 0; key < fit; key++)
        {
            GetBig(rig, key, (ulong)key + 1, (ulong)key);
        }

        Assert.Equal(0, rig.Cache.Evictions);
        Assert.Equal(fit, rig.Cache.VersionCount);
        var counters = rig.Cache.TakeCounters();
        output.WriteLine(counters);
        Assert.Equal((long)((ulong)fit * BigBytes.Value / Mib), Counter(counters, "meshlet_mb"));

        // One more key: key 0, the oldest, makes room, and its buffer (its submission completed) is the new one's.
        var next = GetBig(rig, fit, (ulong)fit + 1, (ulong)fit);
        Assert.Equal(1, rig.Cache.Evictions);
        Assert.Equal(fit, rig.Cache.VersionCount);
        Assert.Equal(1, rig.Cache.PoolHits);
        Assert.Equal(0, rig.Cache.RetiredCount);
        Assert.NotEqual(0, next.Buffer);
        counters = rig.Cache.TakeCounters();
        Assert.Equal((long)((ulong)fit * BigBytes.Value / Mib), Counter(counters, "meshlet_mb"));
        Assert.Equal(1, Counter(counters, "meshlet_evictions"));

        // Key 1 is still there: a hit, no build, and it is the newest now.
        var builds = Builds(rig);
        GetBig(rig, 1, (ulong)fit + 2, (ulong)fit + 1);
        Assert.Equal(builds, Builds(rig));
        Assert.Equal(0, rig.Cache.Evictions);

        // The next eviction takes key 2, not the key that was just used.
        GetBig(rig, fit + 1, (ulong)fit + 3, (ulong)fit + 2);
        Assert.Equal(1, rig.Cache.Evictions);
        GetBig(rig, 1, (ulong)fit + 4, (ulong)fit + 3);
        Assert.Equal(builds + 1, Builds(rig));
        // Key 2 is gone: it is built again (a hash its key had before).
        Assert.Equal(0, rig.Cache.Builds(MetalMeshletCache.BuildReason.Recurring));
        GetBig(rig, 2, (ulong)fit + 5, (ulong)fit + 4);
        Assert.Equal(builds + 2, Builds(rig));
        Assert.Equal(1, rig.Cache.Builds(MetalMeshletCache.BuildReason.Recurring));
        counters = rig.Cache.TakeCounters();
        var megabytes = Counter(counters, "meshlet_mb");
        Assert.InRange(megabytes, 0, (long)(CacheBudget / Mib));
        Assert.Equal((long)((ulong)rig.Cache.VersionCount * BigBytes.Value / Mib), megabytes);
    }

    // Several versions of one key are separate entries for eviction: the least recently used of all of them goes,
    // whichever key holds it, and a version that was just hit is safe.
    [Fact]
    public void BudgetEvictionTreatsEveryVersionOfAKeyAsItsOwnEntry()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(maxVersions: 4, pool: true);
        var contents = new[] { Big(0), Big(1_000_000), Big(2_000_000) };
        var fit = Fit();
        for (var version = 0; version < 3; version++)
        {
            rig.Write(contents[version], BudgetRange);
            GetBig(rig, 0, (ulong)version + 1, (ulong)version);
        }

        // Key 0 holds three versions; fillers take the rest of the budget.
        for (var key = 1; key <= fit - 3; key++)
        {
            GetBig(rig, key, (ulong)key + 3, (ulong)key + 2);
        }

        Assert.Equal(fit, rig.Cache.VersionCount);
        Assert.Equal(0, rig.Cache.Evictions);
        _ = rig.Cache.TakeCounters();

        // Version 0 is used again: a version hit, and the newest entry of all.
        var serial = (ulong)fit + 10;
        rig.Write(contents[0], BudgetRange);
        GetBig(rig, 0, serial, serial - 1);
        Assert.Equal(1, rig.Cache.VersionHits);

        // A new key needs room: the oldest entry of all is version 1.
        GetBig(rig, fit, serial + 1, serial);
        Assert.Equal(1, rig.Cache.Evictions);
        Assert.Equal(fit, rig.Cache.VersionCount);

        var builds = Builds(rig);
        foreach (var version in new[] { 2, 0 })
        {
            serial += 2;
            rig.Write(contents[version], BudgetRange);
            GetBig(rig, 0, serial, serial - 1);
            Assert.Equal(builds, Builds(rig));
        }

        serial += 2;
        rig.Write(contents[1], BudgetRange);
        GetBig(rig, 0, serial, serial - 1);
        Assert.Equal(builds + 1, Builds(rig));
    }

    // Every entry was used by a submission that has not completed: nothing may be evicted, so the build is refused,
    // counted as an over-budget fallback, and the loop that looks for victims ends.
    [Fact]
    public void WhenEveryVersionIsInUseTheBuildIsRefusedAndCounted()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(maxVersions: 4, pool: true);
        rig.Write(Big(0), BudgetRange);
        var fit = Fit();
        for (var key = 0; key < fit; key++)
        {
            GetBig(rig, key, (ulong)key + 1, 0);
        }

        _ = rig.Cache.TakeCounters();
        Assert.False(TryGetBig(rig, fit, (ulong)fit + 1, 0, out var meshlets));

        Assert.Equal(default, meshlets);
        Assert.Equal(0, rig.Cache.Evictions);
        Assert.Equal(fit, rig.Cache.VersionCount);
        Assert.Equal(0, rig.Cache.RetiredCount);
        var counters = rig.Cache.TakeCounters();
        output.WriteLine(counters);
        Assert.Contains("meshlet_fallbacks=0/0/0/0/0/1 ", counters);
        Assert.Equal((long)((ulong)fit * BigBytes.Value / Mib), Counter(counters, "meshlet_mb"));
        Assert.Equal(0, Counter(counters, "meshlet_builds"));

        // Once the oldest submission completes, the same build goes through and evicts exactly it.
        GetBig(rig, fit, (ulong)fit + 2, 1);
        Assert.Equal(1, rig.Cache.Evictions);
    }

    // Freed buffers of a size nothing asks for again pile up in the pool up to its budget; one past it is released
    // (a resident-list change) instead. The pool's byte total is what is pooled, and a build of that size takes one.
    [Fact]
    public void ARetiredBufferPastThePoolBudgetIsReleasedAndTheFreeTotalStaysConsistent()
    {
        if (!RequireDevice()) return;
        Assert.NotEqual(BigBytes.Value, SmallBytes.Value);
        using var rig = new Rig(maxVersions: 4, pool: true);
        rig.Write(Big(0), BudgetRange);
        var fit = Fit();
        for (var key = 0; key < fit; key++)
        {
            GetBig(rig, key, (ulong)key + 1, (ulong)key);
        }

        // Builds of the other size evict the big entries one by one; each freed buffer has no taker.
        var pooledMax = (int)(PoolBudget / BigBytes.Value);
        var smalls = 0;
        var serial = (ulong)fit + 1;
        _ = rig.Cache.TakeCounters();
        long evictions = 0;
        while (evictions < pooledMax + 2 && smalls < 100)
        {
            rig.Write(Strip(0, SmallIndices / 3), BudgetRange);
            Assert.True(rig.Cache.TryGet(
                BudgetRange, SmallIndices, true, MaxVertices, (uint)(1000 + smalls), serial, serial - 1, out _));
            serial++;
            smalls++;
            evictions = rig.Cache.Evictions;
        }

        Assert.True(evictions >= pooledMax + 2, $"only {evictions} evictions");
        Assert.True(evictions <= fit, "a small entry was evicted before every big one");
        var pooled = Math.Min(evictions, pooledMax);
        var counters = rig.Cache.TakeCounters();
        output.WriteLine(counters);
        Assert.Equal((long)((ulong)pooled * BigBytes.Value / Mib), Counter(counters, "meshlet_pool_mb"));
        // Churn is the small buffers created plus the big ones released past the pool.
        Assert.Equal(smalls + (evictions - pooled), Counter(counters, "meshlet_resident_churn"));
        Assert.Equal(0, Counter(counters, "meshlet_pool_hits"));

        // A big build now takes a pooled buffer (after evicting what it must, whose buffers join the pool first).
        rig.Write(Big(0), BudgetRange);
        GetBig(rig, fit + 5, serial, serial - 1);
        var evicted = rig.Cache.Evictions;
        Assert.Equal(1, rig.Cache.PoolHits);
        counters = rig.Cache.TakeCounters();
        var expectedPool = Math.Min(pooledMax, pooled + evicted) - 1;
        Assert.Equal((long)((ulong)expectedPool * BigBytes.Value / Mib), Counter(counters, "meshlet_pool_mb"));
    }

    // A pooled buffer is a whole size class, the words a build needs are fewer: the meshlets handed back (count,
    // bases, every word up to the used length) are the builder's output for the bytes now at the range.
    [Fact]
    public void ABuildIntoALargerPooledBufferEqualsTheBuilderOutput()
    {
        if (!RequireDevice()) return;
        uint[]? larger = null, smaller = null;
        for (var seedA = 1; seedA <= 60 && smaller is null; seedA++)
        {
            for (var seedB = 1; seedB <= 60 && smaller is null; seedB++)
            {
                var a = Indices(seedA);
                var b = Indices(seedB);
                var wordsA = MeshletBuilder.Build(a, MaxVertices, MaxPrimitives).Words.Length;
                var wordsB = MeshletBuilder.Build(b, MaxVertices, MaxPrimitives).Words.Length;
                if (wordsB < wordsA && ClassOf(a, MaxPrimitives) == ClassOf(b, MaxPrimitives))
                {
                    larger = a;
                    smaller = b;
                }
            }
        }

        Assert.NotNull(larger);
        Assert.NotNull(smaller);
        using var rig = new Rig(maxVersions: 1, pool: true);
        rig.Write(larger);
        var first = rig.Get((uint)larger.Length, 1, 0);
        AssertBuiltFrom(larger, first);
        rig.Write(smaller);
        var second = rig.Get((uint)smaller.Length, 2, 1);

        Assert.Equal(first.Buffer, second.Buffer);
        Assert.Equal(1, rig.Cache.PoolHits);
        AssertBuiltFrom(smaller, second);
    }

    // The validated version of a key is the one last built or hit, so it is the key's newest: it goes first only
    // when a later submission serial is passed for an earlier use (an entry still in use cannot be evicted, and the
    // validated one is not). Then a hit on the version that survived counts as a version hit, as a hit on any
    // version other than the validated one does, validated or not.
    [Fact]
    public void AHitOnASurvivingVersionAfterTheValidatedOneWasEvictedCountsAsAVersionHit()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(maxVersions: 4, pool: true);
        var x = Big(0);
        var y = Big(1_000_000);
        var fit = Fit();

        rig.Write(x, BudgetRange);
        var survivor = GetBig(rig, 0, 7, 0);
        rig.Write(y, BudgetRange);
        GetBig(rig, 0, 4, 0);
        for (var key = 1; key <= fit - 2; key++)
        {
            GetBig(rig, key, 1, 0);
        }

        Assert.Equal(fit, rig.Cache.VersionCount);
        Assert.Equal(0, rig.Cache.Evictions);
        _ = rig.Cache.TakeCounters();

        // Room for one: submissions up to 5 are complete, so X (used at 7) stays and Y (used at 4, validated, older
        // than every filler) is the victim.
        GetBig(rig, fit - 1, 8, 5);
        Assert.Equal(1, rig.Cache.Evictions);
        Assert.Equal(0, rig.Cache.VersionHits);

        var builds = Builds(rig);
        rig.Write(x, BudgetRange);
        var again = GetBig(rig, 0, 9, 5);

        Assert.Equal(survivor, again);
        Assert.Equal(1, rig.Cache.VersionHits);
        Assert.Equal(builds, Builds(rig));
        Assert.Contains("meshlet_version_hits=1 ", rig.Cache.TakeCounters());
    }
}
