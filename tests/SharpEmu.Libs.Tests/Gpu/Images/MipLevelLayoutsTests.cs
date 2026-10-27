// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Images;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.Images;

public sealed class MipLevelLayoutsTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(16)]
    public void BulkLayoutMatchesIndividualWritesAndOwnsItsInput(int count)
    {
        var values = Enumerable.Range(0, count).Select(i => new MipLevelLayout
        {
            Offset = (ulong)i * 4096, Size = (ulong)(16 - i) * 256,
            Pitch = (uint)(128 >> Math.Min(i, 7)), Height = (uint)(64 >> Math.Min(i, 6)),
        }).ToArray();
        var individual = new MipLevelLayouts();
        for (var i = 0; i < count; i++) individual[i] = values[i];
        var bulk = new MipLevelLayouts();
        bulk.SetPrefix(values);
        Array.Clear(values);
        for (var i = 0; i < ImageDescription.MaxLevels; i++) Assert.Equal(individual[i], bulk[i]);

        var copy = bulk;
        bulk.SetPrefix([new MipLevelLayout { Size = 99 }]);
        Assert.Equal(individual[0], copy[0]);
        Assert.Equal(99UL, bulk[0].Size);
        for (var i = 1; i < ImageDescription.MaxLevels; i++) Assert.Equal(copy[i], bulk[i]);
        copy[count - 1] = default;
        Assert.NotEqual(0UL, bulk[count - 1].Size);
    }

    [Fact]
    public void EmptyOrInvalidPrefixDoesNotChangeExistingValues()
    {
        var layouts = new MipLevelLayouts();
        layouts[15] = new MipLevelLayout { Size = 17 };
        layouts.SetPrefix([]);
        Assert.Throws<ArgumentOutOfRangeException>(() => layouts.SetPrefix(new MipLevelLayout[17]));
        Assert.Equal(17UL, layouts[15].Size);
        Assert.Equal(default, layouts[-1]);
        Assert.Equal(default, layouts[16]);
    }

    [Fact]
    public void FullChainAllocatesOneArrayInsteadOfSixteen()
    {
        var values = new MipLevelLayout[16];
        values[15].Size = 1;
        var warm = new MipLevelLayouts();
        warm.SetPrefix(values);
        warm[15] = values[15];
        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var repeat = 0; repeat < 128; repeat++)
        {
            var chain = new MipLevelLayouts();
            for (var i = 0; i < values.Length; i++) chain[i] = values[i];
            GC.KeepAlive(chain);
        }
        var individual = GC.GetAllocatedBytesForCurrentThread() - before;
        before = GC.GetAllocatedBytesForCurrentThread();
        for (var repeat = 0; repeat < 128; repeat++)
        {
            var chain = new MipLevelLayouts();
            chain.SetPrefix(values);
            GC.KeepAlive(chain);
        }
        var bulk = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.True(bulk * 6 < individual, $"Bulk={bulk}, individual={individual} bytes.");
    }
}
