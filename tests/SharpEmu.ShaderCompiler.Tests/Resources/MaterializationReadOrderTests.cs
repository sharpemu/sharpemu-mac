// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Resources;
using Xunit;

namespace SharpEmu.ShaderCompiler.Tests.Resources;

public sealed class MaterializationReadOrderTests
{
    // The packed-key sort, and a remembered order it checks first, leave a level's reads exactly as the stable insertion sort did: address order,
    // equal addresses in read order, each read's target, null base and absolute flag still its own.
    [Theory]
    [InlineData(0, 2)]
    [InlineData(1, 17)]
    [InlineData(2, 64)]
    [InlineData(3, 200)]
    [InlineData(4, 1000)]
    public void ReadsSortAsTheInsertionSortLeftThem(int seed, int count)
    {
        var random = new System.Random(seed);
        var order = new MaterializationProgram.ReadOrder();
        for (var trial = 0; trial < 50; trial++)
        {
            var addresses = new ulong[count];
            var tables = new ulong[] { 0x2_0000_1000, 0x2_0000_1040, 0x7F_FFFF_FF00, 0 };
            for (var index = 0; index < count; index++)
            {
                // Descriptor words of a few tables, interleaved, with repeats; now and then past the 48-bit space.
                addresses[index] = trial % 10 == 9 && index == count / 2
                    ? 0x1_0000_0000_0000ul + (ulong)random.Next(64)
                    : tables[random.Next(tables.Length)] + (ulong)(random.Next(24) * 4);
            }

            var targets = Enumerable.Range(0, count).Select(_ => random.Next(512)).ToArray();
            var bases = Enumerable.Range(0, count).Select(_ => (ulong)random.Next(3) * 0x1000).ToArray();
            var absolute = Enumerable.Range(0, count).Select(_ => random.Next(2) == 0).ToArray();
            if (trial % 7 == 0)
            {
                Array.Sort(addresses);
            }

            var (expectedAddresses, expectedTargets, expectedBases, expectedAbsolute) =
                ((ulong[])addresses.Clone(), (int[])targets.Clone(), (ulong[])bases.Clone(), (bool[])absolute.Clone());
            MaterializationProgram.ReadOrder.InsertionSort(expectedAddresses, expectedTargets, expectedBases, expectedAbsolute, count);
            // Unremembered, then twice through one level's remembered order (the second takes it), then through a
            // level last sorted for other reads, whose order no longer fits.
            foreach (var level in new[] { -1, 0, 0, trial % 2 == 0 ? 1 : 2 })
            {
                var (sortedAddresses, sortedTargets, sortedBases, sortedAbsolute) =
                    ((ulong[])addresses.Clone(), (int[])targets.Clone(), (ulong[])bases.Clone(), (bool[])absolute.Clone());
                order.Sort(sortedAddresses, sortedTargets, sortedBases, sortedAbsolute, count, level);
                Assert.Equal(expectedAddresses, sortedAddresses);
                Assert.Equal(expectedTargets, sortedTargets);
                Assert.Equal(expectedBases, sortedBases);
                Assert.Equal(expectedAbsolute, sortedAbsolute);
            }
        }
    }
}
