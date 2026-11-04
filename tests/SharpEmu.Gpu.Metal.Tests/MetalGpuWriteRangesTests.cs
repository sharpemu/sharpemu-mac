// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
using System.Diagnostics;
using System.Text.Json;
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
    // The cap the folding tests were written for; production keeps more records (MetalGpuWriteRanges).
    private const int FoldCap = 512;

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
        var ranges = new MetalGpuWriteRanges(FoldCap);
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
        var folding = new MetalGpuWriteRanges(FoldCap);
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
        var ranges = new MetalGpuWriteRanges(FoldCap);
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
        var ranges = new MetalGpuWriteRanges(FoldCap);
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
        var ranges = new MetalGpuWriteRanges(FoldCap);
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
        var ranges = new MetalGpuWriteRanges(FoldCap);
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

    // The snapshot a thread other than the worker takes answers as PendingSerial does for records the
    // GPU has not finished, without retiring any (the worker's records are left as they were).
    [Fact]
    public void TheSnapshotAnswersLikeThePendingQueryAndRetiresNothing()
    {
        var ranges = new MetalGpuWriteRanges();
        ranges.Note(0x1_0000, 0x1000, serial: 5);
        ranges.Note(0x10_0FFC, 8, serial: 3, MetalGpuWriteOrigin.EndOfPipeLabel);
        ranges.Note(0x3_0000, 0x100, serial: 9);

        Assert.True(ranges.TryPendingSerialSnapshot(0x1_0800, 0x10, completed: 0, out var buffer));
        Assert.Equal(5UL, buffer);
        Assert.True(ranges.TryPendingSerialSnapshot(0x10_1000, 4, completed: 0, out var label));
        Assert.Equal(3UL, label);
        Assert.True(ranges.TryPendingSerialSnapshot(0x2_0000, 0x10, completed: 0, out var none));
        Assert.Equal(0UL, none);

        // Finished by the GPU: not pending, and still recorded until the worker retires it.
        Assert.True(ranges.TryPendingSerialSnapshot(0x1_0800, 0x10, completed: 5, out var finished));
        Assert.Equal(0UL, finished);
        Assert.Equal(3, ranges.Count);
        Assert.True(ranges.TryPendingSerialSnapshot(0x3_0000, 0x10, completed: 5, out var later));
        Assert.Equal(9UL, later);
    }

    // Marking the slices of a run in one pass answers what a PendingSerial per slice does: storage
    // records, labels (looked up by page or scanned), retired records and an unbounded write.
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void MarkingSlicesAgreesWithAQueryPerSlice(int seed)
    {
        var random = new Random(seed);
        const ulong Base = 0x100_0000;
        for (var round = 0; round < 200; round++)
        {
            var ranges = new MetalGpuWriteRanges();
            for (var record = random.Next(0, 24); record > 0; record--)
            {
                var origin = random.Next(3) == 0 ? MetalGpuWriteOrigin.EndOfPipeLabel : MetalGpuWriteOrigin.StorageBuffer;
                var size = origin == MetalGpuWriteOrigin.EndOfPipeLabel ? 4UL * (ulong)random.Next(1, 3) : (ulong)random.Next(1, 0x6000);
                ranges.Note(Base + (ulong)random.Next(0, 0x50000), size, serial: (ulong)random.Next(1, 10), origin);
            }

            if (random.Next(8) == 0)
            {
                ranges.NoteUnbounded((ulong)random.Next(1, 10));
            }

            var completed = (ulong)random.Next(0, 10);
            var sliceSize = 0x1000UL * (ulong)random.Next(1, 3);
            var count = random.Next(1, 70);
            var address = Base + 0x1000UL * (ulong)random.Next(0, 0x40);
            var pending = new bool[count];
            var any = ranges.MarkPending(address, sliceSize, completed, pending);

            var expectedAny = false;
            for (var slice = 0; slice < count; slice++)
            {
                var expected = ranges.PendingSerial(address + sliceSize * (ulong)slice, sliceSize, completed) != 0;
                Assert.Equal(expected, pending[slice]);
                expectedAny |= expected;
            }

            Assert.Equal(expectedAny, any);
        }
    }

    // Whether work of a serial not yet completed (or a later one) wrote a range answers what
    // PendingSerial >= serial does, for the newest serial from its exact ranges and for an older
    // one from the records, over labels and an unbounded write. Past 512 records a fold widens
    // the records, so PendingSerial may answer yes where the exact ranges do not, never the
    // other way round.
    [Theory]
    [InlineData(1, false)]
    [InlineData(2, false)]
    [InlineData(3, false)]
    [InlineData(4, true)]
    [InlineData(5, true)]
    public void WrittenSinceAgreesWithThePendingSerial(int seed, bool folds)
    {
        var random = new Random(seed);
        const ulong Base = 0x200_0000;
        for (var round = 0; round < 60; round++)
        {
            var ranges = new MetalGpuWriteRanges();
            var serial = 1UL;
            for (var record = folds ? random.Next(600, 1200) : random.Next(0, 500); record > 0; record--)
            {
                serial += random.Next(6) == 0 ? 1UL : 0UL;
                var origin = random.Next(4) == 0 ? MetalGpuWriteOrigin.EndOfPipeLabel : MetalGpuWriteOrigin.StorageBuffer;
                var size = origin == MetalGpuWriteOrigin.EndOfPipeLabel ? 8UL : (ulong)random.Next(1, 0x3000);
                ranges.Note(Base + (ulong)random.Next(0, 0x80000), size, serial, origin);
            }

            if (random.Next(10) == 0)
            {
                ranges.NoteUnbounded(serial - (ulong)random.Next(0, 2));
            }

            var completed = (ulong)random.Next(0, (int)serial);
            for (var query = 0; query < 40; query++)
            {
                var since = completed + 1 + (ulong)random.Next(0, (int)(serial - completed) + 1);
                var address = Base + (ulong)random.Next(0, 0x80000);
                var size = (ulong)random.Next(1, 0x4000);
                var written = ranges.WrittenSince(address, size, since);
                var pending = ranges.PendingSerial(address, size, completed) >= since;
                if (folds)
                {
                    Assert.True(pending || !written);
                }
                else
                {
                    Assert.Equal(pending, written);
                }
            }
        }
    }


    // A write fence covers a read only when every write over its bytes in the serial it waits
    // for was stamped: a write noted after the stamp, a merge into a stamped record, a label, an
    // unbounded write or a newer serial over the bytes each sends the read back to the serial.
    [Fact]
    public void AWriteFenceCoversOnlyWritesStampedBeforeIt()
    {
        var ranges = new MetalGpuWriteRanges();
        ranges.Note(0x1_0000, 0x100, serial: 4);
        ranges.Note(0x10_0000, 0x100, serial: 5);
        ranges.Note(0x20_0000, 0x100, serial: 5);
        Assert.Equal(0UL, ranges.WriteFence(0x10_0000, 4, serial: 5));

        ranges.StampFence(serial: 5, fence: 1);
        Assert.Equal(1UL, ranges.WriteFence(0x10_0000, 4, serial: 5));
        // The older serial's record is the caller's to wait for, not the fence's.
        Assert.Equal(1UL, ranges.WriteFence(0x1_0000, 0x10_0000, serial: 5));
        Assert.Equal(0UL, ranges.WriteFence(0x1_0000, 4, serial: 4));

        // A later write into a stamped record makes it wait for the next fence.
        ranges.Note(0x20_0000, 0x200, serial: 5);
        Assert.Equal(0UL, ranges.WriteFence(0x20_0000, 4, serial: 5));
        Assert.Equal(1UL, ranges.WriteFence(0x10_0000, 4, serial: 5));
        ranges.StampFence(serial: 5, fence: 2);
        Assert.Equal(2UL, ranges.WriteFence(0x20_0000, 4, serial: 5));
        Assert.Equal(2UL, ranges.WriteFence(0x10_0000, 0x10_0100, serial: 5));
        Assert.Equal(1UL, ranges.WriteFence(0x10_0000, 4, serial: 5));

        // A newer serial over the bytes, or a label, is not behind the fence.
        ranges.Note(0x10_0080, 4, serial: 6);
        Assert.Equal(0UL, ranges.WriteFence(0x10_0000, 0x100, serial: 5));
        ranges.Note(0x20_0100, 8, serial: 5, MetalGpuWriteOrigin.EndOfPipeLabel);
        ranges.StampFence(serial: 5, fence: 3);
        Assert.Equal(0UL, ranges.WriteFence(0x20_0100, 8, serial: 5));

        ranges.NoteUnbounded(serial: 5);
        Assert.Equal(0UL, ranges.WriteFence(0x20_0000, 4, serial: 5));
    }

    // The indexes over the records and labels (the bucket dictionary PendingSerial uses, the hashed
    // counts the snapshot another thread reads) only let a query skip what cannot reach it: both
    // answer what the unindexed walk of every record and label answers, over narrow and wide writes,
    // labels, retirement, folds and clears.
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void IndexedQueriesAgreeWithTheUnindexedWalk(int seed)
    {
        var random = new Random(seed);
        const ulong Base = 0x10_0000_0000;
        var ranges = new MetalGpuWriteRanges();
        var serial = 1UL;
        var completed = 0UL;
        for (var step = 0; step < 6000; step++)
        {
            switch (random.Next(20))
            {
                case 0:
                    completed = Math.Min(serial - 1, completed + (ulong)random.Next(0, 3));
                    ranges.Retire(completed);
                    break;
                case 1 when random.Next(40) == 0:
                    completed = serial - 1;
                    ranges.Clear(completed);
                    break;
                case 2:
                    serial++;
                    break;
                default:
                {
                    var origin = random.Next(5) == 0 ? MetalGpuWriteOrigin.EndOfPipeLabel : MetalGpuWriteOrigin.StorageBuffer;
                    var size = origin == MetalGpuWriteOrigin.EndOfPipeLabel ? 8UL
                        : random.Next(30) == 0 ? (ulong)random.Next(64, 200) << 20
                        : (ulong)random.Next(1, 0x40000);
                    ranges.Note(Base + ((ulong)random.Next(0, 0x4000) << 16), size, serial, origin);
                    break;
                }
            }

            for (var query = 0; query < 4; query++)
            {
                var address = Base + ((ulong)random.Next(0, 0x4000) << 16) + (ulong)random.Next(0, 0x10000);
                var size = random.Next(10) == 0 ? (ulong)random.Next(1, 64) << 20 : (ulong)random.Next(1, 0x2000);
                var expected = ranges.UnindexedPendingSerial(address, size, completed);
                Assert.True(ranges.TryPendingSerialSnapshot(address, size, completed, out var snapshot));
                Assert.Equal(expected, snapshot);
                Assert.Equal(expected, ranges.PendingSerial(address, size, completed));
            }
        }
    }
    // A range a query found nothing live over answers the next query without a walk, until a write
    // lands in the same 256-byte granule: the neighbours stay answered, the written one does not.
    [Fact]
    public void ACleanGranuleIsForgottenWhenAWriteLandsInIt()
    {
        var ranges = new MetalGpuWriteRanges();
        ranges.Note(0x10_0000, 0x40, serial: 4);
        Assert.Equal(0UL, ranges.PendingSerial(0x10_0100, 0x10, completed: 0));
        Assert.Equal(0UL, ranges.PendingSerial(0x10_0100, 0x10, completed: 0));
        Assert.Equal(0UL, ranges.PendingSerial(0x10_0200, 0x10, completed: 0));

        ranges.Note(0x10_0180, 8, serial: 6, MetalGpuWriteOrigin.EndOfPipeLabel);
        Assert.Equal(6UL, ranges.PendingSerial(0x10_0100, 0x100, completed: 0));
        Assert.Equal(6UL, ranges.PendingSerial(0x10_0180, 4, completed: 0));
        Assert.Equal(0UL, ranges.PendingSerial(0x10_0200, 0x10, completed: 0));
        Assert.Equal(0UL, ranges.PendingSerial(0x10_0100, 0x10, completed: 0));

        // A write wider than the lookups drops every granule, and a granule next to the record
        // is not answered by the record that ends at its edge.
        Assert.Equal(0UL, ranges.PendingSerial(0x10_0040, 0x10, completed: 0));
        ranges.Note(0x10_0000, 0x20_0000, serial: 9);
        Assert.Equal(9UL, ranges.PendingSerial(0x10_0040, 0x10, completed: 0));
        Assert.Equal(9UL, ranges.PendingSerial(0x10_0200, 0x10, completed: 0));
        Assert.Equal(0UL, ranges.PendingSerial(0x30_0000, 0x10, completed: 0));
    }

    // The unbounded serial is read for every query, not remembered with the granules.
    [Fact]
    public void ACleanGranuleStillWaitsForAnUnboundedWrite()
    {
        var ranges = new MetalGpuWriteRanges();
        Assert.Equal(0UL, ranges.PendingSerial(0x10_0000, 0x10, completed: 0));
        Assert.Equal(0UL, ranges.PendingSerial(0x10_0000, 0x10, completed: 0));
        ranges.NoteUnbounded(5);
        Assert.Equal(5UL, ranges.PendingSerial(0x10_0000, 0x10, completed: 0));
        Assert.Equal(0UL, ranges.PendingSerial(0x10_0000, 0x10, completed: 5));
        Assert.Equal(0UL, ranges.PendingSerial(0x10_0000, 0x10, completed: 5));
    }

    // Folding merges the oldest records over the gaps between them: granules answered clean
    // before a fold may lie in such a gap.
    [Fact]
    public void AFoldDropsTheGranulesItsMergedRecordsNowCover()
    {
        var ranges = new MetalGpuWriteRanges(FoldCap);
        var gaps = new List<ulong>();
        for (var index = 0; index < 511; index++)
        {
            var start = 0x100_0000UL + (ulong)index * 0x1_0000;
            ranges.Note(start, 0x100, serial: 3 + (ulong)index / 64);
            gaps.Add(start + 0x8000);
        }

        foreach (var gap in gaps)
        {
            Assert.Equal(0UL, ranges.PendingSerial(gap, 0x10, completed: 0));
        }

        var folds = ranges.Folds;
        for (var index = 0; index < 64 && ranges.Folds == folds; index++)
        {
            ranges.Note(0x900_0000UL + (ulong)index * 0x1_0000, 0x100, serial: 20);
        }

        Assert.True(ranges.Folds > folds);
        foreach (var gap in gaps)
        {
            Assert.Equal(ranges.UnindexedPendingSerial(gap, 0x10, 0), ranges.PendingSerial(gap, 0x10, completed: 0));
        }
    }

    // Narrow, repeating queries over a small address space, so granules are answered, dropped by
    // writes, retired and folded, and every answer is the walk of the records.
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void RepeatedNarrowQueriesAlwaysAgreeWithTheWalkOfEveryRecord(int seed)
    {
        var random = new Random(seed);
        var ranges = new MetalGpuWriteRanges();
        ulong serial = 1;
        ulong completed = 0;
        for (var step = 0; step < 6000; step++)
        {
            switch (random.Next(12))
            {
                case 0 when random.Next(8) == 0:
                    completed = Math.Min(serial, completed + (ulong)random.Next(1, 4));
                    break;
                case 1:
                    serial++;
                    break;
                case 2 when random.Next(200) == 0:
                    ranges.NoteUnbounded(serial);
                    break;
                case 3 when random.Next(100) == 0:
                    ranges.Clear(completed);
                    break;
                case 4 or 5 or 6:
                {
                    var origin = random.Next(3) == 0 ? MetalGpuWriteOrigin.EndOfPipeLabel : MetalGpuWriteOrigin.StorageBuffer;
                    var size = origin == MetalGpuWriteOrigin.EndOfPipeLabel ? 8UL : (ulong)random.Next(1, 0x3000);
                    ranges.Note(0x40_0000 + (ulong)random.Next(0, 0x80000), size, serial, origin);
                    break;
                }
            }

            for (var query = 0; query < 8; query++)
            {
                var address = 0x40_0000 + (ulong)random.Next(0, 0x80000);
                var size = (ulong)(random.Next(6) == 0 ? random.Next(1, 0x1800) : random.Next(1, 0x20));
                var hinted = completed > 0 && random.Next(3) == 0 ? completed - 1 : completed;
                var expected = ranges.UnindexedPendingSerial(address, size, hinted);
                Assert.Equal(expected, ranges.PendingSerial(address, size, hinted));
            }
        }
    }

    // SHARPEMU_WRITE_RANGES_BENCH_OUTPUT=<path>: nanoseconds per PendingSerial for the stream's
    // sequential reads (eight 16-byte reads at the start of each page of command memory) over pages
    // that hold labels and written storage ranges elsewhere on them, so the buckets and label
    // pages cannot rule a read out; every tenth page has a record over its reads. A label is
    // noted now and then. Run once per setting of SHARPEMU_METAL_V2_WRITE_CLEAN_MEMO.
    [Fact]
    public void MeasurePendingSerialWhenRequested()
    {
        var path = Environment.GetEnvironmentVariable("SHARPEMU_WRITE_RANGES_BENCH_OUTPUT");
        if (path is null) return;
        var rounds = new List<object>();
        foreach (var dirtyEvery in new[] { 0, 10 })
        {
            foreach (var labels in new[] { 256, 2048 })
            {
                for (var round = 0; round < 14; round++)
                {
                    var random = new Random(round);
                    var ranges = new MetalGpuWriteRanges();
                    const ulong area = 0x10_0000_0000;
                    for (var index = 0; index < 300; index++)
                    {
                        ranges.Note(area + ((ulong)random.Next(0, 0x4000) << 12) + 0x400, (ulong)random.Next(0x100, 0x800), serial: 5);
                    }

                    for (var index = 0; index < labels; index++)
                    {
                        ranges.Note(area + ((ulong)random.Next(0, 0x4000) << 12) + 0x100 + (ulong)random.Next(0, 0xE00) / 8 * 8, 8, serial: 5, MetalGpuWriteOrigin.EndOfPipeLabel);
                    }

                    for (var page = 0; dirtyEvery != 0 && page < 0x4000; page += dirtyEvery)
                    {
                        ranges.Note(area + ((ulong)page << 12), 0x80, serial: 5);
                    }

                    const int queries = 400_000;
                    var found = 0UL;
                    var started = Stopwatch.GetTimestamp();
                    for (var query = 0; query < queries; query++)
                    {
                        var address = area + ((ulong)(query / 8 % 0x4000) << 12) + (ulong)(query % 8) * 16;
                        found += ranges.PendingSerial(address, 16, completed: 1);
                        if (query % 512 == 0)
                        {
                            ranges.Note(area + ((ulong)random.Next(0, 0x4000) << 12) + 0x100 + (ulong)random.Next(0, 0xE00) / 8 * 8, 8, serial: 5, MetalGpuWriteOrigin.EndOfPipeLabel);
                        }
                    }

                    var elapsed = Stopwatch.GetTimestamp() - started;
                    if (round < 8) continue;
                    rounds.Add(new { dirtyEvery, labels, round, nsPerQuery = elapsed * 1e9 / Stopwatch.Frequency / queries, nonzero = found });
                }
            }
        }

        File.WriteAllText(path, JsonSerializer.Serialize(new {
            architecture = System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture.ToString(),
            cleanMemo = Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_WRITE_CLEAN_MEMO") != "0", rounds },
            new JsonSerializerOptions { WriteIndented = true }));
    }
}
