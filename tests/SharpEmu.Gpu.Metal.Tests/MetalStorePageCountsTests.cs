// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

public sealed class MetalStorePageCountsTests
{
    [Fact]
    public void OverlappingObserversAndPageBoundaryRemovalRemainConservative()
    {
        var counts = new MetalStorePageCounts();
        Assert.False(counts.MayOverlap(0, 8192));
        counts.Add(4095, 2);
        counts.Add(4096, 4096);
        Assert.True(counts.MayOverlap(0, 1));
        Assert.True(counts.MayOverlap(8191, 1));
        Assert.False(counts.MayOverlap(8192, 1));
        counts.Remove(4095, 2);
        Assert.False(counts.MayOverlap(0, 1));
        Assert.True(counts.MayOverlap(4096, 1));
        counts.Remove(4096, 4096);
        Assert.False(counts.MayOverlap(0, 8192));
    }

    [Fact]
    public void MalformedQueriesNeverGrantRejection()
    {
        var counts = new MetalStorePageCounts();
        Assert.True(counts.MayOverlap(0, 0));
        Assert.True(counts.MayOverlap(ulong.MaxValue, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => counts.Add(ulong.MaxValue, 2));
        Assert.Throws<InvalidOperationException>(() => counts.Remove(0, 1));
    }
}
