// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using SharpEmu.HLE.GuestMemory;
using SharpEmu.HLE.Host;
using Xunit;

namespace SharpEmu.Libs.Tests.Memory.GuestMemory;

/// <summary>
/// The shared guest view table and its reader, over plain native memory: entries stay ordered and
/// disjoint, neighbours that continue are one entry, protection changes split and join without
/// moving the mapping generation, a full table says so and never keeps an unmapped range, and a
/// translator's cache survives protection changes but not unmaps.
/// </summary>
public sealed unsafe class GuestViewTableTests
{
    private const ulong Base = 0x20_0000_0000;
    private const ulong Block = 0x10000;
    private const ulong MirrorBase = 0x7100_0000_0000;
    private const ulong MirrorBytes = 1UL << 32;

    private const int LogRecords = 256;

    private static HostSharedRegion NativeRegion(int capacity, int logCapacity = LogRecords)
    {
        var bytes = GuestViewTable.RegionBytes(capacity, logCapacity);
        var address = (ulong)NativeMemory.AllocZeroed((nuint)bytes);
        return new HostSharedRegion(address, bytes, -1, writable: true, static region => NativeMemory.Free((void*)region.Address));
    }

    private static GuestViewTable NewTable(int capacity = 64, int logCapacity = LogRecords) =>
        new(NativeRegion(capacity, logCapacity), MirrorBytes, logCapacity);

    private static GuestViewEntry Entry(ulong address, ulong size, ulong offset, HostPageProtection protection = HostPageProtection.ReadWrite) =>
        new(address, size, offset, protection);

    private static GuestViewLogEntry[] Log(GuestViewTable table, ulong from = 0)
    {
        var records = new List<GuestViewLogEntry>();
        for (var number = from; number < table.LogSequence; number++)
        {
            Assert.True(table.TryReadLog(number, out var record));
            records.Add(record);
        }

        return records.ToArray();
    }

    [Fact]
    public void TheLogRecordsEveryChangeInTheOrderItWasMade()
    {
        using var table = NewTable();
        Assert.Equal(64, table.Capacity);
        Assert.Equal(LogRecords, table.LogCapacity);
        table.Map(Base, 2 * Block, 0, HostPageProtection.ReadWrite);
        table.Protect(Base, 0x1000, HostPageProtection.ReadOnly);
        table.Protect(Base, 0x1000, HostPageProtection.ReadOnly);
        table.Remap(Base, 2 * Block, Base, Block, 7 * Block, HostPageProtection.ReadWrite);
        table.Unmap(Base, Block);
        Assert.Equal(
            [new GuestViewLogEntry(0, GuestViewLogKind.Map, Base, 2 * Block, 0, HostPageProtection.ReadWrite),
             new GuestViewLogEntry(1, GuestViewLogKind.Protect, Base, 0x1000, 0, HostPageProtection.ReadOnly),
             new GuestViewLogEntry(2, GuestViewLogKind.Unmap, Base, 2 * Block, 0, HostPageProtection.NoAccess),
             new GuestViewLogEntry(3, GuestViewLogKind.Map, Base, Block, 7 * Block, HostPageProtection.ReadWrite),
             new GuestViewLogEntry(4, GuestViewLogKind.Unmap, Base, Block, 0, HostPageProtection.NoAccess)],
            Log(table));
    }

    [Fact]
    public void WithoutProtectionTrackingOnlyTheLogSeesProtectionChanges()
    {
        using var table = new GuestViewTable(NativeRegion(64), MirrorBytes, LogRecords, trackProtection: false);
        table.Map(Base, 2 * Block, 0, HostPageProtection.ReadWrite);
        var sequence = table.Sequence;
        table.Protect(Base + 0x1000, 0x1000, HostPageProtection.ReadOnly);
        table.Protect(Base + 0x1000, 0x1000, HostPageProtection.ReadOnly);
        Assert.Equal(sequence, table.Sequence);
        Assert.Equal([Entry(Base, 2 * Block, 0)], table.Snapshot());
        Assert.Equal(3UL, table.LogSequence);
        Assert.All(Log(table, from: 1), record => Assert.Equal(GuestViewLogKind.Protect, record.Kind));
    }

    [Fact]
    public void TheLogRingKeepsOnlyItsLastLap()
    {
        using var table = NewTable(logCapacity: 4);
        for (ulong index = 0; index < 6; index++) table.Map(Base + index * 2 * Block, Block, index * Block, HostPageProtection.ReadWrite);
        Assert.Equal(6UL, table.LogSequence);
        Assert.False(table.TryReadLog(0, out _));
        Assert.False(table.TryReadLog(1, out _));
        Assert.Equal([2UL, 3UL, 4UL, 5UL], Log(table, from: 2).Select(record => record.Number));
        Assert.False(table.TryReadLog(6, out _));
    }

    [Fact]
    public void ContinuingNeighboursJoinAndAnUnmapSplitsThem()
    {
        using var table = NewTable();
        table.Map(Base, Block, 0, HostPageProtection.ReadWrite);
        table.Map(Base + Block, Block, Block, HostPageProtection.ReadWrite);
        table.Map(Base + 2 * Block, Block, 5 * Block, HostPageProtection.ReadWrite);
        Assert.Equal([Entry(Base, 2 * Block, 0), Entry(Base + 2 * Block, Block, 5 * Block)], table.Snapshot());

        table.Unmap(Base + 0x8000, 0x4000);
        Assert.Equal(
            [Entry(Base, 0x8000, 0), Entry(Base + 0xC000, 2 * Block - 0xC000, 0xC000), Entry(Base + 2 * Block, Block, 5 * Block)],
            table.Snapshot());
        Assert.Equal(4UL, table.MappingGeneration);
        Assert.Equal(8UL, table.Sequence);
        Assert.True(table.IsComplete);
    }

    [Fact]
    public void AMapOverLiveRangesReplacesOnlyWhatItCovers()
    {
        using var table = NewTable();
        table.Map(Base, 3 * Block, 0, HostPageProtection.ReadWrite);
        table.Map(Base + Block, Block, 9 * Block, HostPageProtection.ReadOnly);
        Assert.Equal(
            [Entry(Base, Block, 0), Entry(Base + Block, Block, 9 * Block, HostPageProtection.ReadOnly), Entry(Base + 2 * Block, Block, 2 * Block)],
            table.Snapshot());

        // A record replaced in place: one change, and the old range is gone with it.
        var sequence = table.Sequence;
        table.Remap(Base, Block, Base, Block / 2, 20 * Block, HostPageProtection.ReadWrite);
        Assert.Equal(sequence + 2, table.Sequence);
        Assert.Equal(
            [Entry(Base, Block / 2, 20 * Block), Entry(Base + Block, Block, 9 * Block, HostPageProtection.ReadOnly), Entry(Base + 2 * Block, Block, 2 * Block)],
            table.Snapshot());
    }

    [Fact]
    public void ProtectionSplitsAndJoinsWithoutMovingTheMappingGeneration()
    {
        using var table = NewTable();
        table.Map(Base, 4 * Block, 0, HostPageProtection.ReadWrite);
        var generation = table.MappingGeneration;

        table.Protect(Base + Block + 0x4000, 0x4000, HostPageProtection.ReadOnly);
        Assert.Equal(
            [Entry(Base, Block + 0x4000, 0), Entry(Base + Block + 0x4000, 0x4000, Block + 0x4000, HostPageProtection.ReadOnly),
             Entry(Base + Block + 0x8000, 3 * Block - 0x8000, Block + 0x8000)],
            table.Snapshot());
        Assert.True(table.TryFind(Base + Block + 0x5000, out var watched));
        Assert.Equal(HostPageProtection.ReadOnly, watched.Protection);

        // Nothing differs: no change at all, not even a sequence step.
        var sequence = table.Sequence;
        table.Protect(Base + Block + 0x4000, 0x4000, HostPageProtection.ReadOnly);
        table.Protect(Base + 8 * Block, Block, HostPageProtection.NoAccess);
        Assert.Equal(sequence, table.Sequence);

        // Over the edge of the view and past it: only mapped addresses change, then all joins back.
        table.Protect(Base + 3 * Block, 2 * Block, HostPageProtection.ReadOnly);
        Assert.Equal(4, table.Count);
        table.Protect(Base - Block, 6 * Block, HostPageProtection.ReadWrite);
        Assert.Equal([Entry(Base, 4 * Block, 0)], table.Snapshot());
        Assert.Equal(generation, table.MappingGeneration);
    }

    [Fact]
    public void AFullTableIsIncompleteAndNeverKeepsAnUnmappedRange()
    {
        using var table = NewTable(capacity: 2);
        table.Map(Base, 2 * Block, 0, HostPageProtection.ReadWrite);
        table.Map(Base + 4 * Block, Block, 10 * Block, HostPageProtection.ReadWrite);
        Assert.True(table.IsComplete);

        // Unmapping the middle needs a third entry: the whole entry goes instead.
        table.Unmap(Base + 0x8000, 0x4000);
        Assert.False(table.IsComplete);
        Assert.False(table.TryFind(Base, out _));
        Assert.False(table.TryFind(Base + 0x8000, out _));
        Assert.True(table.TryFind(Base + 4 * Block, out _));
    }

    [Fact]
    public void TranslatorCacheSurvivesProtectionChangesButNotUnmaps()
    {
        using var table = NewTable();
        table.Map(Base, 4 * Block, 3 * Block, HostPageProtection.ReadWrite);
        table.Map(Base + 4 * Block, Block, 40 * Block, HostPageProtection.ReadWrite);
        var translator = new GuestMirrorTranslator(table.Region.Address, table.Region.Size, MirrorBase, MirrorBytes)
        {
            CountCacheAccesses = true,
        };

        Assert.True(translator.TryTranslate(Base + 0x100, 0x100, out var mirror));
        Assert.Equal(MirrorBase + 3 * Block + 0x100, mirror);
        Assert.True(translator.TryTranslate(Base + 0x200, 8, out _));
        Assert.Equal(1, translator.CacheMisses);
        Assert.Equal(1, translator.CacheHits);

        // A watch splits the entry; the cached run and a lookup spanning the split both still hold.
        table.Protect(Base + Block, Block, HostPageProtection.ReadOnly);
        Assert.True(translator.TryTranslate(Base + 0x300, 8, out _));
        Assert.Equal(1, translator.CacheMisses);
        Assert.True(translator.TryTranslate(Base + Block - 8, 16, out mirror));
        Assert.Equal(MirrorBase + 4 * Block - 8, mirror);
        Assert.True(translator.TryGetEntry(Base + Block, out var entry));
        Assert.Equal(HostPageProtection.ReadOnly, entry.Protection);
        Assert.True(translator.TryGetOffset(Base + 2 * Block, out var offset, out var contiguous));
        Assert.Equal(5 * Block, offset);
        Assert.Equal(2 * Block, contiguous);

        // Offsets do not continue into the next view: no single translation across it.
        Assert.False(translator.TryTranslate(Base + 4 * Block - 8, 16, out _));

        table.Unmap(Base, Block);
        Assert.False(translator.TryTranslate(Base + 0x100, 8, out _));
        Assert.True(translator.TryTranslate(Base + Block, 8, out mirror));
        Assert.Equal(MirrorBase + 4 * Block, mirror);
        Assert.False(translator.TryTranslate(Base + 5 * Block, 1, out _));
    }

    [Fact]
    public void TranslatorCachesViewsAtAPowerOfTwoStrideAndMissesWithoutAllocating()
    {
        // Every view starts a 64 KiB block whose index is a multiple of 64: one slot each, not one between them.
        const int Views = 48;
        const ulong Stride = 4UL << 20;
        using var table = NewTable(capacity: 128);
        for (var view = 0; view < Views; view++)
            table.Map(Base + (ulong)view * Stride, Block, (ulong)view * 2 * Block, HostPageProtection.ReadWrite);
        var translator = new GuestMirrorTranslator(table.Region.Address, table.Region.Size, MirrorBase, MirrorBytes)
        {
            CountCacheAccesses = true,
        };

        // The thread's first translation sets up its thread-static storage: not a miss's cost.
        _ = new GuestMirrorTranslator(table.Region.Address, table.Region.Size, MirrorBase, MirrorBytes).TryTranslate(Base, 8, out _);
        // Measure the translator only. xUnit's successful value assertions allocate too;
        // keep every result check, but run it after the allocation sample.
        Span<bool> translated = stackalloc bool[Views];
        Span<ulong> mirrors = stackalloc ulong[Views];
        var allocated = GC.GetAllocatedBytesForCurrentThread();
        for (var view = 0; view < Views; view++)
            translated[view] = translator.TryTranslate(Base + (ulong)view * Stride + 0x40, 8, out mirrors[view]);
        var translationBytes = GC.GetAllocatedBytesForCurrentThread() - allocated;

        Assert.Equal(0, translationBytes);
        for (var view = 0; view < Views; view++)
        {
            Assert.True(translated[view]);
            Assert.Equal(MirrorBase + (ulong)view * 2 * Block + 0x40, mirrors[view]);
        }
        Assert.Equal(Views, translator.CacheMisses);
        for (var view = 0; view < Views; view++)
            Assert.True(translator.TryTranslate(Base + (ulong)view * Stride + 0x80, 8, out _));
        Assert.Equal(Views, translator.CacheMisses);
        Assert.Equal(Views, translator.CacheHits);
    }

    [Fact]
    public void TranslatorServesTheOtherBlocksOfARangeFromTheThreadsLastRead()
    {
        using var table = NewTable();
        table.Map(Base, 256 * Block, 5 * Block, HostPageProtection.ReadWrite);
        var translator = new GuestMirrorTranslator(table.Region.Address, table.Region.Size, MirrorBase, MirrorBytes)
        {
            CountCacheAccesses = true,
        };

        for (ulong block = 0; block < 256; block++)
        {
            Assert.True(translator.TryTranslate(Base + block * Block + 0x10, 8, out var mirror));
            Assert.Equal(MirrorBase + (5 + block) * Block + 0x10, mirror);
        }

        Assert.Equal(1, translator.CacheMisses);

        // Another thread has its own last read, and an unmap retires both it and the slots.
        long otherMisses = 0;
        var other = new Thread(() =>
        {
            var before = translator.CacheMisses;
            for (ulong block = 0; block < 256; block++) translator.TryTranslate(Base + block * Block, 8, out _);
            otherMisses = translator.CacheMisses - before;
        });
        other.Start();
        other.Join();
        Assert.Equal(0, otherMisses);
        table.Unmap(Base, Block);
        Assert.False(translator.TryTranslate(Base, 8, out _));
        Assert.True(translator.TryTranslate(Base + 7 * Block, 8, out var moved));
        Assert.Equal(MirrorBase + 12 * Block, moved);
    }

    [Fact]
    public void TranslatorStopsAtTheMirrorAndRefusesARegionWithoutATable()
    {
        using var table = NewTable();
        table.Map(Base, 4 * Block, 0, HostPageProtection.ReadWrite);
        var translator = new GuestMirrorTranslator(table.Region.Address, table.Region.Size, MirrorBase, 2 * Block);
        Assert.True(translator.TryTranslate(Base + Block, Block, out _));
        Assert.False(translator.TryTranslate(Base + Block, Block + 1, out _));
        Assert.False(translator.TryTranslate(Base + 3 * Block, 1, out _));

        using var empty = NativeRegion(4);
        Assert.Throws<InvalidOperationException>(() => new GuestMirrorTranslator(empty.Address, empty.Size, MirrorBase, MirrorBytes));
    }

    [Fact]
    public void ReadersRacingTheWriterSeeOnlyEntriesItWrote()
    {
        // Slot i always maps offset i * Block plus a multiple of Lane, so an address and the offset
        // read for it agree modulo Lane in every consistent read. A torn read mixes two entries.
        const ulong Lane = 1UL << 30;
        const int Slots = 64;
        using var table = NewTable(capacity: 1024);
        var translator = new GuestMirrorTranslator(table.Region.Address, table.Region.Size, MirrorBase, 8 * Lane);
        var stop = 0;
        var failures = 0;
        long reads = 0;
        var readers = Enumerable.Range(0, 2).Select(seed => Task.Run(() =>
        {
            var random = new System.Random(seed);
            while (Volatile.Read(ref stop) == 0)
            {
                var address = Base + (ulong)random.Next(Slots) * Block + (ulong)random.Next((int)Block);
                if (translator.TryGetOffset(address, out var offset, out var contiguous) &&
                    (offset % Lane != address - Base || contiguous == 0 || contiguous > (Base + Slots * Block) - address))
                    Interlocked.Increment(ref failures);
                if (table.TryFind(address, out var entry) && (entry.Offset + (address - entry.Address)) % Lane != address - Base)
                    Interlocked.Increment(ref failures);
                Interlocked.Increment(ref reads);
            }
        })).ToArray();

        // Under a loaded test run the readers' pool threads can start late: race only once they read.
        var started = Environment.TickCount64;
        while (Interlocked.Read(ref reads) == 0 && Environment.TickCount64 - started < 10_000)
        {
            Thread.Sleep(1);
        }

        var writer = new System.Random(7);
        var deadline = Environment.TickCount64 + 300;
        while (Environment.TickCount64 < deadline)
        {
            var slot = (ulong)writer.Next(Slots);
            var address = Base + slot * Block;
            switch (writer.Next(4))
            {
                case 0: table.Unmap(address, Block); break;
                case 1: table.Protect(address, Block / 2, (HostPageProtection)writer.Next(3)); break;
                default: table.Map(address, Block, slot * Block + (ulong)writer.Next(4) * Lane, HostPageProtection.ReadWrite); break;
            }
        }

        Volatile.Write(ref stop, 1);
        Task.WaitAll(readers);
        Assert.Equal(0, failures);
        Assert.True(Interlocked.Read(ref reads) > 0);
        Assert.True(table.IsComplete);
    }
}
