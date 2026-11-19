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
        var remembered = new int[]?[3];
        remembered[2] = Enumerable.Range(0, count).Reverse().ToArray();
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
                if (level < 0)
                    order.Sort(sortedAddresses, sortedTargets, sortedBases, sortedAbsolute, count);
                else
                    order.Sort(sortedAddresses, sortedTargets, sortedBases, sortedAbsolute, count, ref remembered[level]);
                Assert.Equal(expectedAddresses, sortedAddresses);
                Assert.Equal(expectedTargets, sortedTargets);
                Assert.Equal(expectedBases, sortedBases);
                Assert.Equal(expectedAbsolute, sortedAbsolute);
            }
        }
    }

    // Two programs' levels evaluated in turn on one thread (a draw's pixel and vertex plans) each keep their own order:
    // after the first sort neither sorts again, and each still comes out in address order.
    [Fact]
    public void LevelsEvaluatedInTurnKeepTheirOwnOrders()
    {
        var order = new MaterializationProgram.ReadOrder();
        ulong[] pixel = [0x2_0000_1010, 0x2_0000_1000, 0x2_0000_1014, 0x2_0000_1004];
        ulong[] vertex = [0x3_0000_0040, 0x3_0000_0000, 0x3_0000_0044, 0x3_0000_0004, 0x3_0000_0048];
        int[]? pixelOrder = null, vertexOrder = null;
        int[]? firstPixelOrder = null, firstVertexOrder = null;
        for (var draw = 0; draw < 4; draw++)
        {
            // Each draw its own table base, the same layout.
            var shift = (ulong)draw * 0x100;
            foreach (var (reads, isPixel) in new[] { (pixel, true), (vertex, false) })
            {
                var addresses = reads.Select(address => address + shift).ToArray();
                var targets = Enumerable.Range(0, addresses.Length).ToArray();
                var bases = new ulong[addresses.Length];
                var absolute = new bool[addresses.Length];
                if (isPixel)
                    order.Sort(addresses, targets, bases, absolute, addresses.Length, ref pixelOrder);
                else
                    order.Sort(addresses, targets, bases, absolute, addresses.Length, ref vertexOrder);
                Assert.Equal(addresses.Order().ToArray(), addresses);
                Assert.Equal(reads.Select((address, index) => (address, index)).OrderBy(pair => pair.address).Select(pair => pair.index), targets);
            }

            firstPixelOrder ??= pixelOrder;
            firstVertexOrder ??= vertexOrder;
            Assert.Same(firstPixelOrder, pixelOrder);
            Assert.Same(firstVertexOrder, vertexOrder);
        }
    }
}
