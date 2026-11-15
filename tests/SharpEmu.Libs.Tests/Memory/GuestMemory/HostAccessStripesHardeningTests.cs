// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE.Host;
using Xunit;

namespace SharpEmu.Libs.Tests.Memory.GuestMemory;

// What the per-thread access stripes do today at their edges: a token ended twice, an exit on another thread than
// its enter, and a drainer waiting while threads enter and exit without pause. Phantom backings: nothing touches
// the (fake) alias address.
//
// Production callers all enter and exit on one thread: the Metal resident-compare paths
// (MetalRenderHost.Pipelines / ConcurrentResolve) enter and exit around a try/finally in one method,
// SharedBackingViews enters and exits inside one call (its callers run on the calling thread, and every write
// receipt refuses another thread: OrdinaryBackingWriteReservation.RequireThread). No caller carries an enter to
// another thread, so only the same-thread contract is tested for them; the cross-thread balance of the signed sum
// itself is in HostAccessStripesTests.AnExitOnAnotherThreadStillBalancesTheSignedSum.
public sealed class HostAccessStripesHardeningTests
{
    private const int DormantLane = 1;

    private static HostBackingObject Phantom() =>
        new(0x10000, 4096, 4096, _ => new(HostBackingReleaseStatus.Released));

    private static bool Applicable() =>
        HostBackingObject.AccessStripesEnabled && HostManagedWriteIntervals.DormantEnabled;

    // ---- a dormant-ledger token ended twice ----

    // End does not notice: the second End of a dormant token decrements its cell again and reports success. The
    // overdraft shows only as a negative sum, and only when something sums that lane - the first observer.
    [Fact]
    public void ADormantTokenEndedTwiceIsOnlyCaughtAsANegativeSumAtTheNextObserver()
    {
        if (!Applicable()) return;
        var backing = Phantom();
        var messages = new List<string>();
        backing.AliasStripes!.OnFatalForControl = messages.Add;
        backing.EnrollManagedWrites();
        var writes = backing.ManagedWrites!;
        var token = backing.BeginManagedWrite(0, 16);
        Assert.Equal(HostManagedWriteIntervals.DormantSlot, token.Slot);

        Assert.True(writes.End(token, succeeded: true));
        Assert.True(writes.End(token, succeeded: true));
        Assert.Empty(messages);

        // The first observer sums the dormant lane: -1.
        Assert.False(writes.TryCaptureRevision(out _));
        var message = Assert.Single(messages);
        Assert.Contains("lane 1 sums to -1", message);
        Assert.Equal(1, writes.ObservationRefusals);
    }

    // The same overdraft with another dormant writer in flight sums to zero: the observer drains, and once it has
    // (dormantDrained) nothing sums the lane again, so the overdraft is never reported by an observer - only a
    // direct Sum sees it.
    [Fact]
    public void ADoubleEndMaskedByAnotherWriterIsNeverReportedOnceTheObserverHasDrained()
    {
        if (!Applicable()) return;
        var backing = Phantom();
        var messages = new List<string>();
        backing.AliasStripes!.OnFatalForControl = messages.Add;
        backing.EnrollManagedWrites();
        var writes = backing.ManagedWrites!;

        var token = backing.BeginManagedWrite(0, 16);
        HostManagedWriteIntervals.Token other = default;
        using var begun = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        var thread = new Thread(() =>
        {
            other = backing.BeginManagedWrite(16, 16);
            begun.Set();
            release.Wait();
            Assert.True(writes.End(other, succeeded: true));
        });
        thread.Start();
        Assert.True(begun.Wait(TimeSpan.FromSeconds(10)));
        Assert.Equal(HostManagedWriteIntervals.DormantSlot, other.Slot);

        Assert.True(writes.End(token, succeeded: true));
        Assert.True(writes.End(token, succeeded: true));

        // -1 from the overdraft, +1 from the writer in flight.
        Assert.True(writes.TryCaptureRevision(out _));
        Assert.Empty(messages);

        release.Set();
        Assert.True(thread.Join(TimeSpan.FromSeconds(10)));

        // The lane now sums to -1, and the observer never looks again.
        Assert.True(writes.TryCaptureRevision(out _));
        Assert.Empty(messages);
        Assert.Equal(-1, backing.AliasStripes.Sum(DormantLane));
        Assert.Single(messages);
    }

    // A token ended once, on the thread that began it, leaves the dormant lane at zero and the observer unrefused.
    [Fact]
    public void ADormantTokenEndedOnceOnItsOwnThreadLeavesNoCount()
    {
        if (!Applicable()) return;
        var backing = Phantom();
        backing.AliasStripes!.OnFatalForControl = message => throw new InvalidOperationException(message);
        backing.EnrollManagedWrites();
        var writes = backing.ManagedWrites!;
        var token = backing.BeginManagedWrite(0, 16);
        Assert.Equal(backing.AliasStripes.CellForCurrentThread(DormantLane), token.DormantCell);
        Assert.Equal(1, backing.AliasStripes.Sum(DormantLane));

        Assert.True(writes.End(token, succeeded: true));

        Assert.Equal(0, backing.AliasStripes.Sum(DormantLane));
        Assert.True(writes.TryCaptureRevision(out _));
        Assert.Equal(0, writes.ObservationRefusals);
    }

    // ---- alias accesses: the same-thread contract ----

    [Fact]
    public void AnAliasAccessEntersAndExitsOnTheCallingThreadsStripe()
    {
        if (!HostBackingObject.AccessStripesEnabled) return;
        var backing = Phantom();
        var stripes = backing.AliasStripes!;
        stripes.OnFatalForControl = message => throw new InvalidOperationException(message);

        Assert.True(backing.TryEnterAliasAccess());
        Assert.Equal(1, backing.ActiveAccesses);
        Assert.Equal(1, stripes.Sum(HostBackingObject.AliasLane));
        backing.ExitAliasAccess();

        Assert.Equal(0, backing.ActiveAccesses);
        Assert.Equal(0, stripes.Sum(HostBackingObject.AliasLane));
    }

    // A closing backing refuses the entry and backs its count out on the stripe it entered: balanced, no fatal.
    [Fact]
    public void AnEntryRefusedByCloseLeavesTheCountBalanced()
    {
        if (!HostBackingObject.AccessStripesEnabled) return;
        var backing = Phantom();
        backing.AliasStripes!.OnFatalForControl = message => throw new InvalidOperationException(message);
        backing.CloseAccess();

        Assert.False(backing.TryEnterAliasAccess());

        Assert.Equal(0, backing.ActiveAccesses);
        Assert.Equal(0, backing.AliasStripes.Sum(HostBackingObject.AliasLane));
    }

    // ---- a drainer against ten threads entering and exiting continuously ----

    // The stripes' own protocol: a drainer sets its flag with a full fence and sums; an entrant increments, re-reads
    // the flag and backs out when it is set. The drainer's wait ends, and the sum it ends on is zero with no access
    // left in flight and none admitted afterwards.
    [Fact]
    public void ADrainerOverTenBusyThreadsTerminatesAndSeesNoActiveAccess()
    {
        for (var round = 0; round < 10; round++)
        {
            var stripes = new HostAccessStripes(8);
            stripes.OnFatalForControl = message => throw new InvalidOperationException(message);
            var flag = 0;
            var drained = 0;
            var inFlight = 0;
            var violations = 0;
            var stop = 0;
            using var started = new CountdownEvent(10);
            var workers = Enumerable.Range(0, 10).Select(_ => new Thread(() =>
            {
                started.Signal();
                while (Volatile.Read(ref stop) == 0)
                {
                    var cell = stripes.Enter(0);
                    if (Volatile.Read(ref flag) != 0)
                    {
                        stripes.ExitCell(cell);
                        continue;
                    }

                    Interlocked.Increment(ref inFlight);
                    if (Volatile.Read(ref drained) != 0) Interlocked.Increment(ref violations);
                    Thread.SpinWait(20);
                    Interlocked.Decrement(ref inFlight);
                    stripes.ExitCell(cell);
                }
            })).ToArray();
            foreach (var worker in workers) worker.Start();
            Assert.True(started.Wait(TimeSpan.FromSeconds(10)));
            Thread.Sleep(5);

            Interlocked.Exchange(ref flag, 1);
            var deadline = System.Diagnostics.Stopwatch.GetTimestamp() + 10 * System.Diagnostics.Stopwatch.Frequency;
            while (stripes.Sum(0) != 0)
            {
                Assert.True(System.Diagnostics.Stopwatch.GetTimestamp() < deadline, "the drainer never saw zero");
                Thread.SpinWait(50);
            }

            Assert.Equal(0, Volatile.Read(ref inFlight));
            Volatile.Write(ref drained, 1);
            Thread.Sleep(5);
            Volatile.Write(ref stop, 1);
            foreach (var worker in workers) Assert.True(worker.Join(TimeSpan.FromSeconds(10)));

            Assert.Equal(0, violations);
            Assert.Equal(0, stripes.Sum(0));
            Assert.Equal(0, stripes.Sum(1));
        }
    }

    // The same through the backing: Close is the flag, ActiveAccesses the sum.
    [Fact]
    public void AClosedBackingOverTenBusyThreadsDrainsToZeroActiveAccesses()
    {
        if (!HostBackingObject.AccessStripesEnabled) return;
        for (var round = 0; round < 10; round++)
        {
            var backing = Phantom();
            backing.AliasStripes!.OnFatalForControl = message => throw new InvalidOperationException(message);
            var stop = 0;
            var admittedAfterZero = 0;
            var zeroSeen = 0;
            long admitted = 0;
            using var started = new CountdownEvent(10);
            var workers = Enumerable.Range(0, 10).Select(_ => new Thread(() =>
            {
                started.Signal();
                while (Volatile.Read(ref stop) == 0)
                {
                    if (!backing.TryEnterAliasAccess()) continue;
                    Interlocked.Increment(ref admitted);
                    if (Volatile.Read(ref zeroSeen) != 0) Interlocked.Increment(ref admittedAfterZero);
                    Thread.SpinWait(20);
                    backing.ExitAliasAccess();
                }
            })).ToArray();
            foreach (var worker in workers) worker.Start();
            Assert.True(started.Wait(TimeSpan.FromSeconds(10)));
            while (Interlocked.Read(ref admitted) == 0) Thread.Yield();

            backing.CloseAccess();
            var deadline = System.Diagnostics.Stopwatch.GetTimestamp() + 10 * System.Diagnostics.Stopwatch.Frequency;
            while (backing.ActiveAccesses != 0)
            {
                Assert.True(System.Diagnostics.Stopwatch.GetTimestamp() < deadline, "the drain never saw zero");
                Thread.SpinWait(50);
            }

            Volatile.Write(ref zeroSeen, 1);
            Thread.Sleep(5);
            Volatile.Write(ref stop, 1);
            foreach (var worker in workers) Assert.True(worker.Join(TimeSpan.FromSeconds(10)));

            Assert.Equal(0, admittedAfterZero);
            Assert.Equal(0, backing.ActiveAccesses);
            Assert.False(backing.TryEnterAliasAccess());
            Assert.Equal(0, backing.ActiveAccesses);
        }
    }
}
