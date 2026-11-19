// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Reflection;
using SharpEmu.HLE.Host;
using Xunit;

namespace SharpEmu.Libs.Tests.Memory.GuestMemory;

public sealed class HostAccessStripesTests
{
    [Theory]
    [InlineData(1, 2)]
    [InlineData(3, 8)]
    [InlineData(10, 32)]
    [InlineData(32, 64)]
    [InlineData(40, 64)]
    [InlineData(1000, 64)]
    public void StripeCountIsAPowerOfTwoOfAtLeastTwiceTheProcessorsCappedAt64(int processors, int expected)
    {
        Assert.Equal(expected, HostAccessStripes.DefaultStripeCount(processors));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(3)]
    [InlineData(128)]
    public void InvalidStripeCountsAreRefused(int stripes)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new HostAccessStripes(stripes));
    }

    [Fact]
    public void NoTwoStripesNorTheArrayEdgesShareA128ByteLine()
    {
        var stripes = new HostAccessStripes(64);
        var cells = (long[])typeof(HostAccessStripes).GetField("_cells", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(stripes)!;
        var previous = -1;
        for (var stripe = 0; stripe < stripes.Count; stripe++)
        {
            for (var lane = 0; lane < HostAccessStripes.Lanes; lane++)
            {
                var cell = stripes.CellOf(stripe, lane);
                // Whatever the array's alignment, 256 bytes from the array start (its header sits
                // before element 0) and from the array end keep foreign data off the line.
                Assert.True(cell * sizeof(long) >= 256, $"stripe {stripe} lane {lane} is within 256 bytes of the array start");
                Assert.True((cells.Length - cell) * sizeof(long) >= 256, $"stripe {stripe} lane {lane} is within 256 bytes of the array end");
            }

            var first = stripes.CellOf(stripe, 0);
            if (previous >= 0) Assert.True((first - previous) * sizeof(long) >= 256);
            previous = first;
            // The lanes of one stripe share its line on purpose.
            Assert.Equal(first + 1, stripes.CellOf(stripe, 1));
        }
    }

    [Fact]
    public void ConcurrentThreadsCountOnTheirOwnStripes()
    {
        var stripes = new HostAccessStripes(64);
        using var entered = new CountdownEvent(8);
        using var release = new ManualResetEventSlim();
        var cells = new int[8];
        var ids = new int[8];
        var threads = Enumerable.Range(0, 8).Select(index => new Thread(() =>
        {
            ids[index] = Environment.CurrentManagedThreadId;
            cells[index] = stripes.Enter(0);
            entered.Signal();
            release.Wait();
            stripes.Exit(0);
        })).ToArray();
        foreach (var thread in threads) thread.Start();
        Assert.True(entered.Wait(TimeSpan.FromSeconds(10)));
        Assert.Equal(8, stripes.Sum(0));
        Assert.Equal(0, stripes.Sum(1));
        // Each thread counts on the stripe its managed thread id selects, one 256-byte stride per stripe.
        for (var index = 0; index < cells.Length; index++) Assert.Equal(stripes.CellOf(ids[index] & 63, 0), cells[index]);
        release.Set();
        foreach (var thread in threads) Assert.True(thread.Join(TimeSpan.FromSeconds(10)));
        Assert.Equal(0, stripes.Sum(0));
    }

    [Fact]
    public void AnExitOnAnotherThreadStillBalancesTheSignedSum()
    {
        var stripes = new HostAccessStripes(64);
        var enteredCell = stripes.Enter(0);
        var exitCell = -1;
        var other = new Thread(() =>
        {
            exitCell = stripes.CellForCurrentThread(0);
            stripes.Exit(0);
        });
        other.Start();
        Assert.True(other.Join(TimeSpan.FromSeconds(10)));
        Assert.NotEqual(enteredCell, exitCell);
        Assert.Equal(0, stripes.Sum(0));
        stripes.Enter(0);
        Assert.Equal(1, stripes.Sum(0));
        stripes.Exit(0);
        Assert.Equal(0, stripes.Sum(0));
    }

    [Fact]
    public void ANegativeSumIsFatal()
    {
        var stripes = new HostAccessStripes(4);
        string? fatal = null;
        stripes.OnFatalForControl = message => { fatal = message; throw new InvalidOperationException(message); };
        stripes.Exit(1);
        Assert.Equal(0, stripes.Sum(0));
        var error = Assert.Throws<InvalidOperationException>(() => stripes.Sum(1));
        Assert.Contains("Unbalanced access exit", error.Message);
        Assert.NotNull(fatal);
    }

    [Fact]
    public void ExitCellRefusesACellNoEnterReturns()
    {
        var stripes = new HostAccessStripes(4);
        Assert.Throws<ArgumentOutOfRangeException>(() => stripes.ExitCell(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => stripes.ExitCell(stripes.CellOf(0, 0) + HostAccessStripes.Lanes));
        Assert.Throws<ArgumentOutOfRangeException>(() => stripes.ExitCell(int.MaxValue));
        Assert.Throws<ArgumentOutOfRangeException>(() => stripes.CellOf(4, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => stripes.CellOf(0, HostAccessStripes.Lanes));
    }
}
