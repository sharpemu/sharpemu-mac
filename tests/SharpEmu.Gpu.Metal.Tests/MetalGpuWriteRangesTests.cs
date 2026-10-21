// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
using SharpEmu.Gpu.Metal.V2;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// The rule a command-processor read depends on: a read of bytes the GPU is still
/// writing must wait for exactly the submission writing them, and a read of any
/// other bytes must not wait at all.
/// </summary>
public sealed class MetalGpuWriteRangesTests
{
    [Fact]
    public void AReadOutsideEveryRecordedRangeNeverWaits()
    {
        var ranges = new MetalGpuWriteRanges();
        ranges.Note(0x1_0000, 0x1000, serial: 5);
        Assert.Equal(0UL, ranges.PendingSerial(0x2_0000, 0x10, completed: 0));
        Assert.Equal(0UL, ranges.PendingSerial(0x0_F000, 0x1000, completed: 0));
        Assert.Equal(0UL, ranges.PendingSerial(0x1_1000, 0x1000, completed: 0));
    }

    // Labels are looked up by page: a narrow read finds one on its page, including one that
    // straddles a page boundary, and finds none once it retires; a wide read scans them all.
    [Fact]
    public void ALabelIsFoundThroughItsPagesUntilItRetires()
    {
        var ranges = new MetalGpuWriteRanges();
        ranges.Note(0x10_0FFC, 8, serial: 3, MetalGpuWriteOrigin.EndOfPipeLabel);
        ranges.Note(0x50_0000, 4, serial: 7, MetalGpuWriteOrigin.EndOfPipeLabel);
        Assert.Equal(3UL, ranges.PendingSerial(0x10_0FF0, 0x10, completed: 0));
        Assert.Equal(3UL, ranges.PendingSerial(0x10_1000, 4, completed: 0));
        Assert.Equal(0UL, ranges.PendingSerial(0x10_2000, 0x10, completed: 0));
        Assert.Equal(0UL, ranges.PendingSerial(0x10_0000, 0x10, completed: 0));
        Assert.Equal(7UL, ranges.PendingSerial(0x50_0000, 4, completed: 0));
        // 0x100 pages: past the page look-ups, so the labels are scanned.
        Assert.Equal(7UL, ranges.PendingSerial(0x4F_0000, 0x10_0000 + 0x1000, completed: 0));

        Assert.Equal(0UL, ranges.PendingSerial(0x10_0FF0, 0x10, completed: 3));
        Assert.Equal(0UL, ranges.PendingSerial(0x10_1000, 4, completed: 3));
        Assert.Equal(7UL, ranges.PendingSerial(0x50_0000, 4, completed: 3));
        Assert.Equal(0UL, ranges.PendingSerial(0x50_0000, 4, completed: 7));
    }

    [Fact]
    public void AReadInsideARecordedRangeWaitsForThatSubmissionOnly()
    {
        var ranges = new MetalGpuWriteRanges();
        ranges.Note(0x1_0000, 0x1000, serial: 5);
        ranges.Note(0x3_0000, 0x1000, serial: 9);
        Assert.Equal(5UL, ranges.PendingSerial(0x1_0800, 0x10, completed: 0));
        Assert.Equal(9UL, ranges.PendingSerial(0x3_0800, 0x10, completed: 0));
        // Overlapping two records waits for the later of them, which covers both.
        Assert.Equal(9UL, ranges.PendingSerial(0x1_0000, 0x3_0000, completed: 0));
    }

    [Fact]
    public void ACompletedSubmissionStopsBeingWaitedFor()
    {
        var ranges = new MetalGpuWriteRanges();
        ranges.Note(0x1_0000, 0x1000, serial: 5);
        Assert.Equal(5UL, ranges.PendingSerial(0x1_0000, 0x10, completed: 4));
        Assert.Equal(0UL, ranges.PendingSerial(0x1_0000, 0x10, completed: 5));
        Assert.Equal(0, ranges.Count);
        Assert.True(ranges.IsEmpty);
    }

    [Fact]
    public void AnUnboundedWriteMakesEveryReadWaitUntilItCompletes()
    {
        var ranges = new MetalGpuWriteRanges();
        ranges.NoteUnbounded(serial: 7);
        Assert.Equal(7UL, ranges.PendingSerial(0xDEAD_0000, 0x10, completed: 6));
        Assert.Equal(0UL, ranges.PendingSerial(0xDEAD_0000, 0x10, completed: 7));
    }

    [Fact]
    public void RepeatedBindingsOfOneRangeInOneSubmissionStayOneRecord()
    {
        var ranges = new MetalGpuWriteRanges();
        for (var index = 0; index < 1000; index++)
        {
            ranges.Note(0x1_0000, 0x1000, serial: 3);
        }

        Assert.Equal(1, ranges.Count);
        Assert.Equal(0L, ranges.Folds);
        Assert.Equal(3UL, ranges.PendingSerial(0x1_0400, 0x10, completed: 2));
    }

    [Fact]
    public void FoldingManyRecordsNeverLosesAPendingWrite()
    {
        var ranges = new MetalGpuWriteRanges();
        // Disjoint pages, each in its own submission, far past the record cap.
        for (var index = 0UL; index < 4096; index++)
        {
            ranges.Note(0x10_0000 + (index * 0x2000), 0x1000, serial: index + 1);
        }

        Assert.True(ranges.Folds > 0);
        // Every write is still pending, so every one of them is still waited for.
        for (var index = 0UL; index < 4096; index++)
        {
            Assert.True(ranges.PendingSerial(0x10_0000 + (index * 0x2000), 0x1000, completed: 0) >= index + 1);
        }
    }

    [Fact]
    public void TheLatestSequenceOverARangeIsItsNewestWriteAndFoldingKeepsTheMax()
    {
        var ranges = new MetalGpuWriteRanges();
        var first = ranges.Note(0x1_0000, 0x1000, serial: 5);
        var second = ranges.Note(0x1_0800, 0x1000, serial: 6);
        Assert.True(second > first);
        Assert.Equal(second, ranges.LatestSequence(0x1_0800, 4));
        Assert.Equal(first, ranges.LatestSequence(0x1_0000, 4));
        Assert.Equal(0UL, ranges.LatestSequence(0x9_0000, 4));

        // A write that merges into an earlier record of its submission carries its own,
        // newer sequence over the whole record.
        var merged = ranges.Note(0x1_0400, 0x10, serial: 5);
        Assert.Equal(merged, ranges.LatestSequence(0x1_0000, 4));

        // Folded records keep the largest sequence of their parts.
        var folding = new MetalGpuWriteRanges();
        var newest = 0UL;
        for (var index = 0UL; index < 4096; index++)
        {
            newest = Math.Max(newest, folding.Note(0x10_0000 + (index * 0x2000), 0x1000, serial: index + 1));
        }

        Assert.True(folding.Folds > 0);
        Assert.Equal(newest, folding.LatestSequence(0x10_0000, 0x2000UL * 4096));
        for (var index = 0UL; index < 4096; index += 97)
        {
            var address = 0x10_0000 + (index * 0x2000);
            Assert.True(folding.LatestSequence(address, 4) >= index + 1);
        }
    }

    [Fact]
    public void EveryWriteOriginHasAReportName()
    {
        Assert.Equal(MetalGpuWriteOrigins.Count, MetalGpuWriteOrigins.Names.Length);
        Assert.Equal("device_address", MetalGpuWriteOrigins.Names[(int)MetalGpuWriteOrigin.DeviceAddress]);
        Assert.Equal("unbounded", MetalGpuWriteOrigins.Names[(int)MetalGpuWriteOrigin.Unbounded]);
    }

    [Fact]
    public void AStoreThroughThePageTableAtAnEvaluatedRangeIsWaitedForLikeAnyOther()
    {
        var ranges = new MetalGpuWriteRanges();
        ranges.Note(0x1_0000, 0x1000, serial: 5, MetalGpuWriteOrigin.DeviceAddress);
        Assert.Equal(5UL, ranges.PendingSerial(0x1_0800, 8, completed: 0));
        Assert.Equal(MetalGpuWriteOrigin.DeviceAddress, ranges.LastMatch.Origin);
        Assert.Equal(1, ranges.RecordsFrom(MetalGpuWriteOrigin.DeviceAddress));
        // A store the planner could not evaluate has no range, so every read waits.
        ranges.NoteUnbounded(serial: 9);
        Assert.Equal(9UL, ranges.PendingSerial(0xDEAD_0000, 8, completed: 0));
        Assert.Equal(MetalGpuWriteOrigin.Unbounded, ranges.LastMatch.Origin);
    }

    [Fact]
    public void ClearingAfterAFullWaitDropsEverything()
    {
        var ranges = new MetalGpuWriteRanges();
        ranges.Note(0x1_0000, 0x1000, serial: 5);
        ranges.NoteUnbounded(serial: 5);
        ranges.Clear(completed: 5);
        Assert.True(ranges.IsEmpty);
        Assert.Equal(0UL, ranges.PendingSerial(0x1_0000, 0x1000, completed: 5));
        Assert.False(ranges.MayCover(0x1_0000, 0x1000));
    }

    /// <summary>
    /// The cheap test the command stream uses for every packet it reads. It may say
    /// yes when the full test would say no, but it must never say no when a recorded
    /// write covers the bytes.
    /// </summary>
    [Fact]
    public void TheCheapCoverTestNeverMissesARecordedWrite()
    {
        var ranges = new MetalGpuWriteRanges();
        Assert.False(ranges.MayCover(0x1_0000, 0x10));

        ranges.Note(0x2_0000, 0x1000, serial: 5);
        Assert.True(ranges.MayCover(0x2_0000, 0x10));
        Assert.True(ranges.MayCover(0x2_0FF0, 0x20));
        Assert.False(ranges.MayCover(0x1_0000, 0x10));
        Assert.False(ranges.MayCover(0x2_1000, 0x10));

        ranges.Note(0x9_0000, 0x1000, serial: 6);
        // The bounds span both records, so the gap between them answers yes; what
        // matters is that neither record is ever missed.
        Assert.True(ranges.MayCover(0x2_0000, 0x10));
        Assert.True(ranges.MayCover(0x9_0800, 0x10));
        Assert.False(ranges.MayCover(0x9_1000, 0x10));

        ranges.Retire(completed: 5);
        Assert.False(ranges.MayCover(0x2_0000, 0x10));
        Assert.True(ranges.MayCover(0x9_0800, 0x10));

        ranges.Retire(completed: 6);
        Assert.False(ranges.MayCover(0x9_0800, 0x10));

        // An unbounded write covers everything until it completes.
        ranges.NoteUnbounded(serial: 9);
        Assert.True(ranges.MayCover(0xDEAD_0000, 0x10));
    }

    [Fact]
    public void FoldingKeepsTheCheapCoverTestHonest()
    {
        var ranges = new MetalGpuWriteRanges();
        for (var index = 0UL; index < 2048; index++)
        {
            ranges.Note(0x10_0000 + (index * 0x2000), 0x1000, serial: index + 1);
        }

        Assert.True(ranges.Folds > 0);
        for (var index = 0UL; index < 2048; index++)
        {
            Assert.True(ranges.MayCover(0x10_0000 + (index * 0x2000), 0x1000));
        }
    }

    [Fact]
    public void LabelWritesNeverWidenTheFoldOfOrdinaryRecords()
    {
        var ranges = new MetalGpuWriteRanges();
        for (ulong i = 0; i < 2000; i++)
        {
            ranges.Note(0x80_0000_0000UL + i * 0x40_0000, 8, serial: 1 + i, MetalGpuWriteOrigin.EndOfPipeLabel);
            ranges.Note(0x1_0000UL + i * 0x100, 0x10, serial: 1 + i);
        }

        Assert.Equal(0UL, ranges.PendingSerial(0x80_0000_0008UL, 0x100, completed: 0));
        Assert.Equal(0UL, ranges.PendingSerial(0x7F_0000_0000UL, 0x1000, completed: 0));
    }

    [Fact]
    public void AReadOverlappingALabelStillReportsItsSerial()
    {
        var ranges = new MetalGpuWriteRanges();
        for (ulong i = 0; i < 1500; i++)
        {
            ranges.Note(0x80_0000_0000UL + i * 0x1000, 4, serial: 10 + i, MetalGpuWriteOrigin.EndOfPipeLabel);
        }

        Assert.Equal(10UL + 700, ranges.PendingSerial(0x80_0000_0000UL + 700 * 0x1000 + 2, 1, completed: 0));
        Assert.Equal(MetalGpuWriteOrigin.EndOfPipeLabel, ranges.LastMatch.Origin);
        Assert.Equal(0UL, ranges.PendingSerial(0x80_0000_0000UL + 700 * 0x1000 + 4, 4, completed: 0));
        Assert.Equal(0UL, ranges.PendingSerial(0x80_0000_0000UL + 700 * 0x1000 + 2, 1, completed: 10 + 700));
        ranges.Retire(5000);
        Assert.True(ranges.IsEmpty);
    }

    [Fact]
    public void FoldingScatteredRecordsKeepsEveryByteCoveredWithoutSpanningUnrelatedMemory()
    {
        var ranges = new MetalGpuWriteRanges();
        const int n = 3000;
        // Two far-apart clusters of scattered 8-byte records.
        ulong AddressOf(int i) => (i % 2 == 0 ? 0x1000_0000UL : 0x9000_0000UL) + (ulong)(i / 2) * 0x40;
        for (var i = 0; i < n; i++)
        {
            ranges.Note(AddressOf(i), 8, serial: 1UL + (ulong)i);
        }

        Assert.True(ranges.Folds > 0);
        for (var i = 0; i < n; i++)
        {
            Assert.True(ranges.PendingSerial(AddressOf(i), 8, completed: 0) >= 1UL + (ulong)i);
        }

        // The gap between the clusters is never reported as written.
        Assert.Equal(0UL, ranges.PendingSerial(0x4000_0000UL, 0x1000, completed: 0));
    }
}
