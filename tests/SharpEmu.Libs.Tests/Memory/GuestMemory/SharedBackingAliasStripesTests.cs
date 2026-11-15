// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Reflection;
using SharpEmu.HLE.GuestMemory;
using SharpEmu.HLE.Host;
using Xunit;
using static SharpEmu.Libs.Tests.Memory.HostViews.HostViewTestSupport;

namespace SharpEmu.Libs.Tests.Memory.GuestMemory;

// SharedBackingViews' lock-free single-view accesses counted on the backing's per-thread stripes:
// a copy reservation and Dispose still wait for every admitted access.
[Collection(GuestMemoryStateCollection.Name)]
public sealed unsafe class SharedBackingAliasStripesTests
{
    private const ulong Part = 256 * 1024;
    private const int RecordBytes = 256;

    [Fact]
    public void ACopyReservationWaitsForAWriterPausedBeforeItsBytes()
    {
        if (!Supported) return;
        using var mapping = new Mappings();
        using var paused = new ManualResetEventSlim();
        using var resume = new ManualResetEventSlim();
        var writerId = -1;
        SetCopyHook(mapping.Store, (_, _) =>
        {
            if (Environment.CurrentManagedThreadId != Volatile.Read(ref writerId)) return;
            paused.Set();
            Assert.True(resume.Wait(TimeSpan.FromSeconds(20)));
        });
        var written = false;
        var writer = new Thread(() =>
        {
            Volatile.Write(ref writerId, Environment.CurrentManagedThreadId);
            written = mapping.Store.TryWriteSingleView(mapping.Address + 0x100, BitConverter.GetBytes(Marker));
        }) { IsBackground = true };
        var reserved = new ManualResetEventSlim();
        var reservation = false;
        Exception? failure = null;
        var copier = new Thread(() =>
        {
            try
            {
                // Disjoint from the paused write: only the alias pin makes it wait.
                reservation = mapping.Store.TryReserveCopy(mapping.Address + 0x2_0000, mapping.Address + 0x1_0000, 64, out var lease);
                reserved.Set();
                if (reservation) lease.Dispose();
            }
            catch (Exception error) { failure = error; reserved.Set(); }
        }) { IsBackground = true };
        try
        {
            writer.Start();
            Assert.True(paused.Wait(TimeSpan.FromSeconds(10)));
            copier.Start();
            Assert.False(reserved.Wait(TimeSpan.FromMilliseconds(300)));
            // The reservation closed lock-free access while it waits.
            Assert.False(mapping.Store.TryReadSingleView(mapping.Address + 0x3_0000, new byte[8]));
        }
        finally
        {
            resume.Set();
        }

        Assert.True(writer.Join(TimeSpan.FromSeconds(10)));
        Assert.True(copier.Join(TimeSpan.FromSeconds(10)));
        Assert.Null(failure);
        Assert.True(written);
        Assert.True(reservation);
        var bytes = new byte[8];
        Assert.True(mapping.Store.TryReadSingleView(mapping.Address + 0x100, bytes));
        Assert.Equal(Marker, BitConverter.ToUInt64(bytes));
    }

    // Ten writers rewrite whole records (every word the same generation) in a copy's source and
    // destination while another thread copies the source over the destination under a reservation.
    // A record with mixed generations means a write landed while the copy ran.
    [Fact]
    public void WritersNeverTearARecordACopyReservationHolds()
    {
        if (!Supported) return;
        using var mapping = new Mappings();
        const int Writers = 10;
        const int CopyBytes = 64 * 1024;
        var source = mapping.Address;
        var destination = mapping.Address + Part;
        var stop = 0;
        var failures = 0;
        var writes = 0L;
        using var started = new CountdownEvent(Writers);
        var writers = Enumerable.Range(0, Writers).Select(index => new Thread(() =>
        {
            var record = new byte[RecordBytes];
            started.Signal();
            for (var generation = 1UL; Volatile.Read(ref stop) == 0; generation++)
            {
                for (var word = 0; word < RecordBytes; word += 8) BitConverter.TryWriteBytes(record.AsSpan(word), generation);
                var target = ((generation & 1) == 0 ? source : destination) + (ulong)(index * RecordBytes);
                if (!mapping.Store.TryWriteSingleView(target, record) && !mapping.Store.TryWriteBacking(target, record))
                    Interlocked.Increment(ref failures);
                Interlocked.Increment(ref writes);
            }
        }) { IsBackground = true }).ToArray();
        foreach (var writer in writers) writer.Start();
        var torn = 0;
        var copies = 0;
        try
        {
            Assert.True(started.Wait(TimeSpan.FromSeconds(10)));
            var deadline = DateTime.UtcNow + TimeSpan.FromMilliseconds(1500);
            while (DateTime.UtcNow < deadline || copies < 50)
            {
                Assert.True(mapping.Store.TryReserveCopy(destination, source, CopyBytes, out var lease));
                using (lease)
                {
                    Buffer.MemoryCopy((void*)lease.Source, (void*)lease.Destination, CopyBytes, CopyBytes);
                    for (var index = 0; index < Writers; index++)
                    {
                        if (!WholeRecord(lease.Destination + (ulong)(index * RecordBytes))) torn++;
                        if (!WholeRecord(lease.Source + (ulong)(index * RecordBytes))) torn++;
                    }
                }

                copies++;
            }
        }
        finally
        {
            Volatile.Write(ref stop, 1);
            foreach (var writer in writers) Assert.True(writer.Join(TimeSpan.FromSeconds(10)));
        }

        Assert.Equal(0, failures);
        Assert.Equal(0, torn);
        Assert.True(Volatile.Read(ref writes) > Writers);
    }

    [Fact]
    public void ReadersRacingDisposeEitherReadOrAreRefused()
    {
        if (!Supported) return;
        for (var round = 0; round < 10; round++)
        {
            var mapping = new Mappings();
            var expected = BitConverter.GetBytes(Marker);
            for (var slot = 0; slot < 8; slot++) Assert.True(mapping.Store.TryWriteBacking(mapping.Address + (ulong)(slot * 64), expected));
            var stop = 0;
            var wrong = 0;
            var reads = 0L;
            var refusals = 0L;
            using var started = new CountdownEvent(8);
            var readers = Enumerable.Range(0, 8).Select(slot => new Thread(() =>
            {
                var bytes = new byte[8];
                started.Signal();
                while (Volatile.Read(ref stop) == 0)
                {
                    if (mapping.Store.TryReadSingleView(mapping.Address + (ulong)(slot * 64), bytes))
                    {
                        if (BitConverter.ToUInt64(bytes) != Marker) Interlocked.Increment(ref wrong);
                        Interlocked.Increment(ref reads);
                    }
                    else Interlocked.Increment(ref refusals);
                }
            }) { IsBackground = true }).ToArray();
            foreach (var reader in readers) reader.Start();
            try
            {
                Assert.True(started.Wait(TimeSpan.FromSeconds(10)));
                Thread.Sleep(5);
                mapping.Store.Dispose();
                Assert.False(mapping.Store.TryReadSingleView(mapping.Address, new byte[8]));
                Assert.False(mapping.Store.TryEnterAliasAccess());
                Thread.Sleep(5);
            }
            finally
            {
                Volatile.Write(ref stop, 1);
                foreach (var reader in readers) Assert.True(reader.Join(TimeSpan.FromSeconds(10)));
                mapping.Dispose();
            }

            Assert.Equal(0, wrong);
            Assert.True(reads > 0);
            Assert.True(refusals > 0);
        }
    }

    // Lock-free writes and reads in a stable view stay exact while another range is mapped and
    // unmapped over and over.
    [Fact]
    public void SingleViewAccessesHoldWhileOtherRangesAreRemapped()
    {
        if (!Supported) return;
        using var mapping = new Mappings();
        var churn = mapping.Address + 2 * Part;
        var stop = 0;
        var failures = 0;
        var accesses = 0L;
        var workers = Enumerable.Range(0, 4).Select(worker => new Thread(() =>
        {
            var bytes = new byte[8];
            var slot = mapping.Address + (ulong)(worker * 64);
            for (var value = 1UL; Volatile.Read(ref stop) == 0; value++)
            {
                if (!mapping.Store.TryWriteSingleView(slot, BitConverter.GetBytes(value)) ||
                    !mapping.Store.TryReadSingleView(slot, bytes) || BitConverter.ToUInt64(bytes) != value)
                    Interlocked.Increment(ref failures);
                Interlocked.Increment(ref accesses);
                _ = mapping.Store.TryReadSingleView(churn + 8, bytes);
            }
        }) { IsBackground = true }).ToArray();
        foreach (var worker in workers) worker.Start();
        try
        {
            for (var round = 0; round < 200; round++)
            {
                Assert.True(mapping.Store.TryMapReservedRange(churn, Part, 3 * Part, HostPageProtection.ReadWrite, out _));
                Assert.True(mapping.Store.Unmap(churn, Part, out _));
            }
        }
        finally
        {
            Volatile.Write(ref stop, 1);
            foreach (var worker in workers) Assert.True(worker.Join(TimeSpan.FromSeconds(10)));
        }

        Assert.Equal(0, failures);
        Assert.True(accesses > 0);
    }

    private static bool WholeRecord(ulong address)
    {
        var words = (ulong*)address;
        for (var word = 1; word < RecordBytes / 8; word++)
            if (words[word] != words[0]) return false;
        return true;
    }

    private static void SetCopyHook(SharedBackingViews store, Action<ulong, int> hook) =>
        typeof(SharedBackingViews).GetField("_beforeCanonicalAliasCopyForControl", BindingFlags.Instance | BindingFlags.NonPublic)!
            .SetValue(store, hook);

    // Two views of Part bytes each (backing offsets 0 and Part) and a third reserved Part left
    // unmapped for remapping.
    private sealed class Mappings : IDisposable
    {
        public IHostViewMemory Host { get; } = HostViewMemory.Create();
        public SharedBackingViews Store { get; }
        public ulong Address { get; }
        private readonly ulong _holeSize;
        private bool _disposed;

        public Mappings()
        {
            Store = new SharedBackingViews(Host, 4 * Part);
            _holeSize = AlignUp(4 * Part, Host.Granularity);
            Address = ReserveFreeHole(Host, _holeSize);
            Assert.True(Host.SplitHole(Address, Part));
            Assert.True(Host.SplitHole(Address + Part, Part));
            Assert.True(Host.SplitHole(Address + 2 * Part, Part));
            Assert.True(Store.TryMapReservedRange(Address, Part, 0, HostPageProtection.ReadWrite, out _));
            Assert.True(Store.TryMapReservedRange(Address + Part, Part, Part, HostPageProtection.ReadWrite, out _));
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            Store.Dispose();
            Assert.True(Host.JoinHoles(Address, _holeSize));
            Assert.True(Host.FreeHole(Address, _holeSize));
        }
    }
}
