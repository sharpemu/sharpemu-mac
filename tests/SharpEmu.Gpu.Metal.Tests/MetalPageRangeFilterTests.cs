// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

public sealed class MetalPageRangeFilterTests
{
    [Fact]
    public void CrossPageWritesAndNativeAliasesRemainPossibleOverlaps()
    {
        var filter = new MetalPageRangeFilter();
        filter.Add(0x1FFC, 8, space: 0x1234);
        Assert.True(filter.MayOverlap(0x1FFF, 1, space: 0x1234));
        Assert.True(filter.MayOverlap(0x2000, 4, space: 0x1234));
        Assert.True(filter.MayOverlap(0x1000, 0x1001, space: 0x1234));
        Assert.False(filter.MayOverlap(0x2000, 0, space: 0x1234));
        // Native aliases supply the same buffer/offset key regardless of guest address.
        filter.Add(0x1FFC, 8, space: 0x5678);
        Assert.True(filter.MayOverlap(0x2000, 4, space: 0x5678));
    }

    [Fact]
    public void LargeOrWrappingInsertionsFallBackUntilClear()
    {
        var filter = new MetalPageRangeFilter();
        foreach (var range in new[] { (0UL, 65UL * 4096), (ulong.MaxValue - 2, 8UL) })
        {
            filter.Add(range.Item1, range.Item2);
            Assert.True(filter.MayOverlap(0x12345678, 8));
            filter.Clear();
            Assert.False(filter.MayOverlap(0x12345678, 8));
        }
        filter.Add(ulong.MaxValue, 1);
        Assert.True(filter.MayOverlap(ulong.MaxValue, 1));
        Assert.True(filter.MayOverlap(ulong.MaxValue - 2, 8));
        Assert.True(filter.MayOverlap(0, 65UL * 4096));
    }

    [Fact]
    public void ClearingRemovesOldPagesAndRepopulationKeepsNewPages()
    {
        var filter = new MetalPageRangeFilter();
        filter.Add(0x2000, 0); // No write.
        Assert.False(filter.MayOverlap(0x2000, 1));
        filter.Add(0x2000, 8);
        Assert.True(filter.MayOverlap(0x2000, 1));
        filter.Clear();
        Assert.False(filter.MayOverlap(0x2000, 1));
        filter.Add(0x4000, 8);
        Assert.True(filter.MayOverlap(0x4000, 1));
        Assert.Contains(Enumerable.Range(0, 100), page => !filter.MayOverlap((ulong)page * 4096, 8));
    }

    [Fact]
    public void SeededRangesNeverRejectAnExactOverlapAcrossAddressSpaces()
    {
        var random = new Random(8123);
        var filter = new MetalPageRangeFilter();
        for (var batch = 0; batch < 100; batch++)
        {
            filter.Clear();
            var ranges = Enumerable.Range(0, 128).Select(_ => (
                Address: (ulong)random.NextInt64(1L << 40),
                Size: (ulong)random.Next(1, 9000),
                Space: (ulong)random.Next(0, 8))).ToArray();
            foreach (var range in ranges) filter.Add(range.Address, range.Size, range.Space);
            foreach (var range in ranges)
            {
                Assert.True(filter.MayOverlap(range.Address, 1, range.Space));
                Assert.True(filter.MayOverlap(range.Address + range.Size - 1, 1, range.Space));
                Assert.True(filter.MayOverlap(range.Address, range.Size, range.Space));
            }
        }
    }
}
