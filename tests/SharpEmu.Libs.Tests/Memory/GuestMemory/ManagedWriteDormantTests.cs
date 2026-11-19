// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Reflection;
using SharpEmu.HLE.GuestMemory;
using SharpEmu.HLE.Host;
using Xunit;
using static SharpEmu.Libs.Tests.Memory.HostViews.HostViewTestSupport;

namespace SharpEmu.Libs.Tests.Memory.GuestMemory;

// The write-interval ledger stays dormant until its first observer: dormant writers register only
// on their own stripe, and an observer refuses until every dormant writer has landed.
[Collection(GuestMemoryStateCollection.Name)]
public sealed unsafe class ManagedWriteDormantTests
{
    private const BindingFlags Private = BindingFlags.Instance | BindingFlags.NonPublic;
    private const ulong Part = 256 * 1024;

    [Fact]
    public void DormantWritesTouchNoSlotHistoryOrCounterAndDoNotAllocate()
    {
        if (!Supported || !HostManagedWriteIntervals.DormantEnabled) return;
        using var mapping = new Mappings();
        var writers = mapping.Writers;
        Assert.False(writers.IsObserved);
        byte[] eight = [1, 2, 3, 4, 5, 6, 7, 8];
        var packet = new byte[32];
        for (var i = 0; i < 10000; i++)
        {
            Assert.True(mapping.Store.TryWriteSingleView(mapping.Address + 64, eight));
            Assert.True(mapping.Store.TryWritePairSingleView(mapping.Address + 128, eight, mapping.Address + 256, packet));
        }

        var allocated = long.MaxValue; var allSucceeded = true;
        for (var round = 0; round < 5 && allocated != 0; round++)
        {
            var before = GC.GetAllocatedBytesForCurrentThread();
            for (var i = 0; i < 10000; i++)
            {
                allSucceeded &= mapping.Store.TryWriteSingleView(mapping.Address + 64, eight);
                allSucceeded &= mapping.Store.TryWritePairSingleView(mapping.Address + 128, eight, mapping.Address + 256, packet);
            }

            allocated = Math.Min(allocated, GC.GetAllocatedBytesForCurrentThread() - before);
        }

        Assert.True(allSucceeded); Assert.Equal(0, allocated);
        Assert.False(writers.IsObserved);
        Assert.Equal(0, Field<long>(writers, "nextIdentity"));
        Assert.Equal(0, Field<long>(writers, "admission"));
        foreach (var name in new[] { "active", "begun", "completed", "aborted" }) Assert.Equal(0, Field<long>(writers, name));
        foreach (var slot in Field<Array>(writers, "slots"))
            Assert.Equal(0L, (long)slot!.GetType().GetField("State", Private)!.GetValue(slot)!);
        foreach (var entry in Field<Array>(writers, "history"))
            Assert.Equal(0L, (long)entry!.GetType().GetField("State", Private)!.GetValue(entry)!);
        Assert.Equal(0, writers.ObservationRefusals);
    }

    // A dormant writer paused between its registration and its bytes: a revision capture and a
    // watch enrollment both refuse. Once it lands, both succeed and later writes advance identity.
    [Fact]
    public void ObserversRefuseWhileADormantWriterIsBetweenRegistrationAndBytes()
    {
        if (!Supported || !HostManagedWriteIntervals.DormantEnabled) return;
        using var mapping = new Mappings();
        var writers = mapping.Writers;
        using var paused = new ManualResetEventSlim();
        using var resume = new ManualResetEventSlim();
        var writerId = -1;
        typeof(SharedBackingViews).GetField("_beforeCanonicalAliasCopyForControl", Private)!.SetValue(mapping.Store, (Action<ulong, int>)((_, _) =>
        {
            if (Environment.CurrentManagedThreadId != Volatile.Read(ref writerId)) return;
            paused.Set();
            Assert.True(resume.Wait(TimeSpan.FromSeconds(20)));
        }));
        var written = false;
        var writer = new Thread(() =>
        {
            Volatile.Write(ref writerId, Environment.CurrentManagedThreadId);
            written = mapping.Store.TryWriteSingleView(mapping.Address + 0x40, BitConverter.GetBytes(Marker));
        }) { IsBackground = true };
        HostManagedWriteIntervals.PhysicalRun[] runs = [new(0x1000, 64)];
        try
        {
            writer.Start();
            Assert.True(paused.Wait(TimeSpan.FromSeconds(10)));
            Assert.False(writers.IsObserved);
            Assert.False(writers.TryCaptureRevision(out var refusedRevision));
            Assert.Equal(-1, refusedRevision);
            Assert.True(writers.IsObserved);
            Assert.Null(writers.TryCreateReadWatch(runs, out var rejection, out _));
            Assert.Equal("watch-enrollment-observation-activating", rejection);
            Assert.False(writers.TryReservePublication(0, runs, null, null));
            Assert.Equal(3, writers.ObservationRefusals);
            Assert.Equal(0, Field<long>(writers, "nextIdentity"));
        }
        finally { resume.Set(); }

        Assert.True(writer.Join(TimeSpan.FromSeconds(10)));
        Assert.True(written);
        Assert.True(writers.TryCaptureRevision(out var revision));
        Assert.Equal(0, revision);
        var watch = writers.TryCreateReadWatch(runs, out var accepted, out _);
        Assert.NotNull(watch); Assert.Null(accepted);
        Assert.True(watch.IsCurrent);
        // Observed now: a later write takes admission, an identity, history, and revokes the
        // overlapping watch before its bytes.
        Assert.True(mapping.Store.TryWriteSingleView(mapping.Address + 0x1010, BitConverter.GetBytes(Marker)));
        Assert.True(Field<long>(writers, "nextIdentity") > revision);
        Assert.Equal(1, Field<long>(writers, "completed"));
        Assert.False(watch.IsCurrent);
        watch.Release();
        Assert.Equal(3, writers.ObservationRefusals);
    }

    // Ten writers write while an observer loops capture -> read V -> reserve. Whenever the
    // reservation succeeds, the bytes under it still equal V. Starts dormant, so the first
    // observations race dormant writers.
    [Fact]
    public void AReservedPublicationAlwaysSeesTheBytesItsRevisionRead()
    {
        if (!Supported) return;
        using var mapping = new Mappings();
        var writers = mapping.Writers;
        const int Watched = 64;
        HostManagedWriteIntervals.PhysicalRun[] runs = [new(0, Watched)];
        var stop = 0;
        var landed = 0L;
        using var started = new CountdownEvent(10);
        var threads = Enumerable.Range(0, 10).Select(index => new Thread(() =>
        {
            var random = new System.Random(index + 1);
            var bytes = new byte[8];
            started.Signal();
            while (Volatile.Read(ref stop) == 0)
            {
                random.NextBytes(bytes);
                // One write in sixteen lands in the watched range; the rest elsewhere in the view.
                var offset = random.Next(16) == 0 ? (ulong)random.Next(Watched / 8) * 8 : 0x1000 + (ulong)random.Next(0x1000) * 8;
                // A write refused while the observer holds the publication simply does not land.
                if (mapping.Store.TryWriteSingleView(mapping.Address + offset, bytes) ||
                    mapping.Store.TryWriteBacking(mapping.Address + offset, bytes))
                    Interlocked.Increment(ref landed);
                Thread.SpinWait(random.Next(200));
            }
        }) { IsBackground = true }).ToArray();
        foreach (var thread in threads) thread.Start();
        var attempts = 0; var captures = 0; var reserved = 0; var mismatches = 0;
        var value = new byte[Watched];
        try
        {
            Assert.True(started.Wait(TimeSpan.FromSeconds(10)));
            var deadline = DateTime.UtcNow + TimeSpan.FromMilliseconds(1500);
            while (DateTime.UtcNow < deadline || reserved < 20 && DateTime.UtcNow < deadline + TimeSpan.FromSeconds(10))
            {
                attempts++;
                if (!writers.TryCaptureRevision(out var revision)) continue;
                captures++;
                new ReadOnlySpan<byte>((void*)mapping.Backing.AliasBase, Watched).CopyTo(value);
                if (!writers.TryReservePublication(revision, runs, null, null)) continue;
                try
                {
                    reserved++;
                    if (!new ReadOnlySpan<byte>((void*)mapping.Backing.AliasBase, Watched).SequenceEqual(value)) mismatches++;
                }
                finally { writers.ReleasePublication(null); }
            }
        }
        finally
        {
            Volatile.Write(ref stop, 1);
            foreach (var thread in threads) Assert.True(thread.Join(TimeSpan.FromSeconds(10)));
        }

        Assert.True(Volatile.Read(ref landed) > 0);
        Assert.Equal(0, mismatches);
        Assert.True(captures > 0, $"attempts={attempts}");
        Assert.True(reserved > 0, $"attempts={attempts} captures={captures}");
        Assert.True(writers.IsObserved);
    }

    private static T Field<T>(object owner, string name) => (T)owner.GetType().GetField(name, Private)!.GetValue(owner)!;

    private sealed class Mappings : IDisposable
    {
        public IHostViewMemory Host { get; } = HostViewMemory.Create();
        public SharedBackingViews Store { get; }
        public ulong Address { get; }
        public HostBackingObject Backing { get; }
        public HostManagedWriteIntervals Writers { get; }
        private readonly ulong _holeSize;

        public Mappings()
        {
            Store = new SharedBackingViews(Host, Part);
            _holeSize = AlignUp(Part, Host.Granularity);
            Address = ReserveFreeHole(Host, _holeSize);
            Assert.True(Store.TryMapReservedRange(Address, Part, 0, HostPageProtection.ReadWrite, out _));
            Backing = (HostBackingObject)typeof(SharedBackingViews).GetField("_backing", Private)!.GetValue(Store)!;
            // The macOS view host enrolls every backing when it is created.
            Backing.EnrollManagedWrites();
            Writers = Backing.ManagedWrites!;
        }

        public void Dispose()
        {
            Store.Dispose();
            Assert.True(Host.FreeHole(Address, _holeSize));
        }
    }
}
