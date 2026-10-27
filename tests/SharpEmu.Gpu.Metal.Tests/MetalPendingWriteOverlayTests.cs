// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

// The overlay's lock-free page counts only decide that a read overlaps nothing; every answer
// has to be the one the locked walk gives, with and without them.
public sealed class MetalPendingWriteOverlayTests
{
    private const ulong Page = 0x1000;

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void AReadOnAnotherPageOverlapsNothing(bool pageFilter)
    {
        var overlay = new MetalPendingWriteOverlay(pageFilter);
        overlay.Add(0x7000_0010, [1, 2, 3, 4], serial: 5);

        Assert.Equal(MetalOverlayCoverage.None, overlay.Coverage(0x7000_0010 + Page, 4, () => 0));
        Assert.Equal(MetalOverlayCoverage.None, overlay.Coverage(0x7000_0020, 4, () => 0));
        Span<byte> bytes = [9, 9, 9, 9];
        overlay.Apply(0x7000_0010 + Page, bytes);
        Assert.Equal([9, 9, 9, 9], bytes.ToArray());
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void APendingWriteAnswersForItsBytes(bool pageFilter)
    {
        var overlay = new MetalPendingWriteOverlay(pageFilter);
        overlay.Add(0x7000_0010, [1, 2, 3, 4], serial: 5);

        Assert.Equal(MetalOverlayCoverage.Full, overlay.Coverage(0x7000_0010, 4, () => 0));
        Assert.Equal(MetalOverlayCoverage.Partial, overlay.Coverage(0x7000_000E, 4, () => 0));
        Span<byte> bytes = [9, 9, 9, 9];
        overlay.Apply(0x7000_000E, bytes);
        Assert.Equal([9, 9, 1, 2], bytes.ToArray());
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void AWriteAcrossAPageBoundaryIsSeenFromBothPages(bool pageFilter)
    {
        var overlay = new MetalPendingWriteOverlay(pageFilter);
        overlay.Add(0x7000_0FFE, [1, 2, 3, 4], serial: 5);

        Assert.Equal(MetalOverlayCoverage.Partial, overlay.Coverage(0x7000_0FFC, 4, () => 0));
        Assert.Equal(MetalOverlayCoverage.Partial, overlay.Coverage(0x7000_1000, 4, () => 0));
        Assert.Equal(MetalOverlayCoverage.Full, overlay.Coverage(0x7000_0FFE, 4, () => 0));
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ALandedWriteStopsAnsweringOnceRetired(bool pageFilter)
    {
        var overlay = new MetalPendingWriteOverlay(pageFilter);
        overlay.Add(0x7000_0010, [1, 2, 3, 4], serial: 5);
        overlay.Add(0x7000_2010, [5, 6, 7, 8], serial: 6);

        overlay.Retire(5);

        Assert.Equal(1, overlay.Count);
        Assert.Equal(MetalOverlayCoverage.None, overlay.Coverage(0x7000_0010, 4, () => 5));
        Assert.Equal(MetalOverlayCoverage.Full, overlay.Coverage(0x7000_2010, 4, () => 5));

        overlay.Retire(6);
        overlay.Add(0x7000_0010, [1, 2, 3, 4], serial: 7);
        Assert.Equal(MetalOverlayCoverage.Full, overlay.Coverage(0x7000_0010, 4, () => 6));
        Assert.Equal(MetalOverlayCoverage.None, overlay.Coverage(0x7000_2010, 4, () => 6));
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ALaterWriteOverTheRangeIsForgotten(bool pageFilter)
    {
        var overlay = new MetalPendingWriteOverlay(pageFilter);
        overlay.Add(0x7000_0010, [1, 2, 3, 4], serial: 5);

        overlay.Forget(0x7000_0010 + Page, 4);
        Assert.Equal(MetalOverlayCoverage.Full, overlay.Coverage(0x7000_0010, 4, () => 0));

        overlay.Forget(0x7000_0012, 2);
        Assert.Equal(MetalOverlayCoverage.Partial, overlay.Coverage(0x7000_0010, 4, () => 0));

        overlay.Forget(0x7000_0000, 0x100);
        Assert.Equal(0, overlay.Count);
        Assert.Equal(MetalOverlayCoverage.None, overlay.Coverage(0x7000_0010, 4, () => 0));
    }

    [Fact]
    public void PagesThatShareACountStillAnswerOnlyForTheirOwnWrites()
    {
        // Far more pages than counts, so many of them share one: a shared count may send a read
        // to the locked walk, which must still find nothing on a page no write is on.
        var overlay = new MetalPendingWriteOverlay();
        for (ulong index = 0; index < 40_000; index += 2)
        {
            overlay.Add(0x7000_0000 + index * Page, [1, 2, 3, 4], serial: 5);
        }

        for (ulong index = 0; index < 40_000; index++)
        {
            var expected = index % 2 == 0 ? MetalOverlayCoverage.Full : MetalOverlayCoverage.None;
            Assert.Equal(expected, overlay.Coverage(0x7000_0000 + index * Page, 4, () => 0));
        }

        overlay.Clear();
        for (ulong index = 0; index < 40_000; index += 2)
        {
            Assert.Equal(MetalOverlayCoverage.None, overlay.Coverage(0x7000_0000 + index * Page, 4, () => 0));
        }
    }

    [Fact]
    public void AQwordRunIsEntriesThatAnswerAndAreForgottenOneByOne()
    {
        var overlay = new MetalPendingWriteOverlay();
        overlay.AddQwords(0x7000_0000, count: 4, strideBytes: 16, value: 0x1122334455667788UL, serial: 5);

        Assert.Equal(4, overlay.Count);
        for (ulong index = 0; index < 4; index++)
        {
            Span<byte> bytes = stackalloc byte[8];
            overlay.Apply(0x7000_0000 + index * 16, bytes);
            Assert.Equal(0x1122334455667788UL, BitConverter.ToUInt64(bytes));
            Assert.Equal(MetalOverlayCoverage.None, overlay.Coverage(0x7000_0000 + index * 16 + 8, 8, () => 0));
        }

        overlay.Forget(0x7000_0010, 8);
        Assert.Equal(3, overlay.Count);
        Assert.Equal(MetalOverlayCoverage.None, overlay.Coverage(0x7000_0010, 8, () => 0));
        Assert.Equal(MetalOverlayCoverage.Full, overlay.Coverage(0x7000_0020, 8, () => 0));
    }

    [Fact]
    public void AReusedEntryCarriesNothingOfItsPreviousWrite()
    {
        var overlay = new MetalPendingWriteOverlay();
        overlay.Add(0x7000_0000, [1, 2, 3, 4, 5, 6, 7, 8], serial: 5, baseline: [0, 0, 0, 0, 0, 0, 0, 0]);
        // Half of it is overwritten, which marks those bytes dead on the entry.
        overlay.Forget(0x7000_0000, 4);
        overlay.Retire(completedSerial: 5);
        Assert.Equal(0, overlay.Count);

        overlay.Add(0x7000_0100, [9, 9, 9, 9, 9, 9, 9, 9], serial: 6);
        Assert.Equal(MetalOverlayCoverage.Full, overlay.Coverage(0x7000_0100, 8, () => 0));
        Span<byte> bytes = stackalloc byte[8];
        overlay.Apply(0x7000_0100, bytes);
        Assert.Equal([9, 9, 9, 9, 9, 9, 9, 9], bytes.ToArray());
        Assert.Equal(MetalOverlayCoverage.None, overlay.Coverage(0x7000_0000, 8, () => 0));

        // No baseline was recorded for the reused entry, so a changed memory does not drop it.
        overlay.DropChangedBaselines(0x7000_0100, 8, static (_, destination) => { destination.Fill(0x55); return true; });
        Assert.Equal(1, overlay.Count);
    }

    [Fact]
    public void RetiredSmallEntriesAreReusedWithoutAllocating()
    {
        var overlay = new MetalPendingWriteOverlay();
        ulong serial = 0;
        void Round()
        {
            overlay.AddQwords(0x7000_0000, count: 16, strideBytes: 16, value: 7, serial: ++serial);
            overlay.Add(0x7000_4000, [1, 2, 3, 4], serial: serial);
            overlay.Retire(serial);
        }

        for (var warm = 0; warm < 8; warm++)
        {
            Round();
        }

        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var round = 0; round < 200; round++)
        {
            Round();
        }

        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
        Assert.Equal(0, overlay.Count);
    }

    [Theory]
    [InlineData(512)]
    [InlineData(2048)]
    [InlineData(8192)]
    public void CompletedOcclusionWriteBurstsReuseEntries(int count)
    {
        var overlay = new MetalPendingWriteOverlay();
        ulong serial = 0;
        void Round()
        {
            overlay.AddQwords(0x7100_0000, count, 16, ++serial, serial);
            overlay.Retire(serial);
        }

        for (var warm = 0; warm < 8; warm++) Round();
        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var round = 0; round < 16; round++) Round();
        var allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.True(allocated <= 512, $"A warm {count}-write burst allocated {allocated} bytes over16rounds.");
        Assert.Equal(0, overlay.Count);

        // A reused entry must carry the new value, and retirement must still remove it.
        overlay.AddQwords(0x7100_0000, count, 16, 0xFEDCBA9876543210, ++serial);
        Span<byte> last = stackalloc byte[8];
        overlay.Apply(0x7100_0000 + (ulong)(count - 1) * 16, last);
        Assert.Equal(0xFEDCBA9876543210UL, System.Buffers.Binary.BinaryPrimitives.ReadUInt64LittleEndian(last));
        overlay.Retire(serial);
        Assert.Equal(0, overlay.Count);
    }

    [Fact]
    public void SpareEntryBudgetIsSharedAcrossSizesAndClearReturnsOnlyRemovedEntries()
    {
        var overlay = new MetalPendingWriteOverlay();
        Span<byte> bytes = stackalloc byte[16];
        bytes.Fill(0x55);
        for (var length = 4; length <= 16; length += 4)
        {
            for (var i = 0; i < MetalPendingWriteOverlay.MaxPooledEntries; i++)
                overlay.Add(0x7200_0000 + (ulong)i * 16, bytes[..length], serial: 1);
            overlay.Clear();
            Assert.InRange(overlay.SpareEntryCount, 0, MetalPendingWriteOverlay.MaxPooledEntries);
            Assert.Equal(0, overlay.Count);
        }

        Assert.Equal(MetalPendingWriteOverlay.MaxPooledEntries, overlay.SpareEntryCount);
        overlay.Add(0x7200_0000, bytes[..4], serial: 2);
        Assert.Equal(MetalPendingWriteOverlay.MaxPooledEntries - 1, overlay.SpareEntryCount);
        overlay.Retire(1);
        Assert.Equal(1, overlay.Count);
        overlay.Retire(2);
        Assert.Equal(MetalPendingWriteOverlay.MaxPooledEntries, overlay.SpareEntryCount);
    }

    [Fact]
    public void ALongRangeIsWalkedRatherThanCounted()
    {
        var overlay = new MetalPendingWriteOverlay();
        overlay.Add(0x7010_0000, [1, 2, 3, 4], serial: 5);

        Assert.Equal(MetalOverlayCoverage.Partial, overlay.Coverage(0x7000_0000, 0x20_0000, () => 0));
        Assert.Equal(MetalOverlayCoverage.None, overlay.Coverage(0x7020_0000, 0x20_0000, () => 0));
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ALongForgetWalksOnlyThePagesWithEntriesAndForgetsWhatItCovers(bool pageFilter)
    {
        var overlay = new MetalPendingWriteOverlay(pageFilter);
        // Many qwords on two pages (as an occlusion dump leaves them), one inside the range and
        // one past it; a write across the range's end; one already forgotten, still queued.
        overlay.AddQwords(0x7004_0000, 64, 16, 0x1122_3344_5566_7788, serial: 5);
        overlay.AddQwords(0x7090_0000, 64, 16, 0x1122_3344_5566_7788, serial: 5);
        overlay.Add(0x7007_FFFC, [1, 2, 3, 4, 5, 6, 7, 8], serial: 5);
        overlay.Add(0x7002_0000, [9, 9, 9, 9], serial: 5);
        overlay.Forget(0x7002_0000, 4);

        // 512 KiB: far more pages than the overlay has entries on.
        overlay.Forget(0x7000_0000, 0x8_0000);

        Assert.Equal(MetalOverlayCoverage.None, overlay.Coverage(0x7004_0000, 64 * 16, () => 0));
        Assert.Equal(MetalOverlayCoverage.None, overlay.Coverage(0x7007_FFFC, 4, () => 0));
        Assert.Equal(MetalOverlayCoverage.Full, overlay.Coverage(0x7008_0000, 4, () => 0));
        Assert.Equal(MetalOverlayCoverage.Full, overlay.Coverage(0x7090_0000 + 63 * 16, 8, () => 0));
        Span<byte> bytes = [0, 0, 0, 0];
        overlay.Apply(0x7008_0000, bytes);
        Assert.Equal([5, 6, 7, 8], bytes.ToArray());

        // The entries the forget left answer, then go when their submission lands.
        Assert.Equal(64 + 1, overlay.Count);
        overlay.Retire(5);
        Assert.Equal(0, overlay.Count);
    }
}
