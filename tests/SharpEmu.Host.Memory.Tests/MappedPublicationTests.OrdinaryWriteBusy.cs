// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
using System.Reflection;
using SharpEmu.HLE;
using SharpEmu.HLE.GuestMemory;
using SharpEmu.HLE.Host;
using Xunit;

namespace SharpEmu.Host.Memory.Tests;

public sealed partial class MappedPublicationTests
{
    private static OrdinaryBackingWriteReservation Busy(State state, ulong address = State.Address + 32)
    {
        var before = state.Bytes();
        Assert.False(state.Views.TryWriteBacking(address, new byte[] { 1, 2, 3 }, out var reservation));
        Assert.Equal(before, state.Bytes()); // exact first refusal has NO bytes
        var owned = Assert.IsType<OrdinaryBackingWriteReservation>(reservation);
        Assert.NotEqual(0L, owned.RejectedAdmission & (1L << 32));
        Assert.True(owned.IsOriginalCurrent);
        return owned;
    }

    [Fact]
    public void PublicationBusyOriginalAdmissionWritesOnceAfterCancelWithFullNeighbors()
    {
        using var state = new State(); var original = state.Capture();
        using var publication = state.Prepare(original, State.Address, [7]);
        using var reservation = Busy(state);
        publication.Dispose();
        Assert.Equal(OrdinaryBackingWriteAdmission.Admitted, reservation.TryAcquire(TimeSpan.Zero));
        reservation.BeginWriteAttempt();
        var written = state.Views.TryWriteBacking(State.Address + 32, new byte[] { 3, 5, 7 });
        reservation.CompleteWriteAttempt(written); Assert.True(written);
        var expected = new byte[state.Bytes().Length]; new byte[] { 3, 5, 7 }.CopyTo(expected, 32);
        Assert.Equal(expected, state.Bytes());
        Assert.Throws<InvalidOperationException>(() => reservation.BeginWriteAttempt());
        Assert.Throws<InvalidOperationException>(() => reservation.CompleteWriteAttempt(true));
    }

    [Fact]
    public void HeldPublicationTimesOutWithoutBytesOrLeakedAdmission()
    {
        using var state = new State(); var original = state.Capture();
        using var publication = state.Prepare(original, State.Address, [7]);
        using var reservation = Busy(state);
        Assert.Equal(OrdinaryBackingWriteAdmission.TimedOut, reservation.TryAcquire(TimeSpan.Zero));
        Assert.Throws<InvalidOperationException>(() => reservation.BeginWriteAttempt());
        reservation.Dispose(); publication.Dispose();
        using var next = state.Prepare(original, State.Address, [9]);
        Assert.All(state.Bytes(), value => Assert.Equal((byte)0, value));
    }

    [Fact]
    public void PublicationBusyWaitObservesActualPeerCancellation()
    {
        using var state = new State(); var original = state.Capture();
        using var publication = state.Prepare(original, State.Address, [7]);
        using var reservation = Busy(state);
        using var peerReady = new ManualResetEventSlim();
        using var cancelRequested = new ManualResetEventSlim();
        // Acquisition has a production-enforced 500 ms bound. Start a dedicated
        // peer before that window so a busy test thread pool cannot delay it.
        var cancel = Task.Factory.StartNew(() =>
        {
            peerReady.Set();
            Assert.True(cancelRequested.Wait(TimeSpan.FromSeconds(10)));
            publication.Dispose();
        }, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);
        try
        {
            Assert.True(peerReady.Wait(TimeSpan.FromSeconds(10)));
            cancelRequested.Set();
            Assert.Equal(OrdinaryBackingWriteAdmission.Admitted, reservation.TryAcquire(TimeSpan.FromMilliseconds(500)));
            Assert.True(cancel.Wait(TimeSpan.FromSeconds(10))); cancel.GetAwaiter().GetResult();
        }
        finally { cancelRequested.Set(); publication.Dispose(); Assert.True(cancel.Wait(TimeSpan.FromSeconds(10))); }
    }

    [Fact]
    public void SameAddressRemapCannotReplaceOriginalBusyWriteMapping()
    {
        using var state = new State(); var original = state.Capture();
        using var publication = state.Prepare(original, State.Address, [7]);
        using var reservation = Busy(state); publication.Dispose();
        Assert.True(state.Views.Unmap(State.Address, State.Page, out _));
        Assert.True(state.Views.TryMapReservedRange(State.Address, State.Page, State.Page,
            HostPageProtection.ReadWrite, out _));
        Assert.False(reservation.IsOriginalCurrent);
        Assert.Equal(OrdinaryBackingWriteAdmission.OriginalChanged, reservation.TryAcquire(TimeSpan.Zero));
        Assert.Throws<InvalidOperationException>(() => reservation.BeginWriteAttempt());
        Assert.All(state.Bytes(), value => Assert.Equal((byte)0, value));
    }

    [Fact]
    public void OriginalBackingCloseVetoesBusyWriteWithoutTouchingAlias()
    {
        using var state = new State(); var original = state.Capture();
        using var publication = state.Prepare(original, State.Address, [7]);
        using var reservation = Busy(state); publication.Dispose();
        var close = Task.Run(() => state.Backing.Dispose());
        try
        {
            Assert.True(SpinWait.SpinUntil(() => state.Backing.IsClosing, TimeSpan.FromSeconds(10)));
            Assert.Equal(OrdinaryBackingWriteAdmission.OriginalChanged, reservation.TryAcquire(TimeSpan.Zero));
            Assert.False(close.IsCompleted); // actual alias access pin remains held
            reservation.Dispose();
            Assert.True(close.Wait(TimeSpan.FromSeconds(10))); close.GetAwaiter().GetResult();
        }
        finally { reservation.Dispose(); Assert.True(close.Wait(TimeSpan.FromSeconds(10))); }
        // No pointer reads after backing release.
    }

    [Fact]
    public void ProviderGateCannotBeHiddenInsideBusyAdmissionWait()
    {
        using var state = new State(); var original = state.Capture();
        using var publication = state.Prepare(original, State.Address, [7]);
        using var reservation = Busy(state);
        var gate = typeof(FakeProvider).GetField("gate", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(state.Host)!;
        lock (gate) Assert.Throws<InvalidOperationException>(() => reservation.TryAcquire(TimeSpan.Zero));
        publication.Dispose();
        Assert.Equal(OrdinaryBackingWriteAdmission.Admitted, reservation.TryAcquire(TimeSpan.Zero));
    }

    [Fact]
    public void BusyReservationCannotMoveToAnotherManagedThreadOrReplay()
    {
        using var state = new State(); var original = state.Capture();
        using var publication = state.Prepare(original, State.Address, [7]);
        using var reservation = Busy(state);
        var peer = Task.Run(() => Assert.Throws<InvalidOperationException>(() => reservation.TryAcquire(TimeSpan.Zero)));
        Assert.True(peer.Wait(TimeSpan.FromSeconds(10))); peer.GetAwaiter().GetResult();
        publication.Dispose(); reservation.Dispose(); reservation.Dispose();
        Assert.Throws<InvalidOperationException>(() => reservation.TryAcquire(TimeSpan.Zero));
        Assert.Throws<InvalidOperationException>(() => reservation.CompleteWriteAttempt(true));
    }

    [Fact]
    public void AcquiredOuterWriterExcludesPublicationUntilNoByteCancellation()
    {
        using var state = new State(); var original = state.Capture();
        using var publication = state.Prepare(original, State.Address, [7]);
        using var reservation = Busy(state); publication.Dispose();
        Assert.Equal(OrdinaryBackingWriteAdmission.Admitted, reservation.TryAcquire(TimeSpan.Zero));
        Assert.Equal(CpuMappedPublicationPreparationResult.Rejected, state.Views.TryPrepareMappedPublication(
            original, [new(State.Address, new byte[] { 9 })], out _));
        reservation.Dispose();
        using var next = state.Prepare(original, State.Address, [9]); // no guessed physical write history
        Assert.All(state.Bytes(), value => Assert.Equal((byte)0, value));
    }

    [Fact]
    public void UnresolvedStartedWriteIsUnknownAndNeverRetriedOrRecordedAsCancel()
    {
        using var state = new State(); var original = state.Capture();
        using var publication = state.Prepare(original, State.Address, [7]);
        using var reservation = Busy(state); publication.Dispose();
        Assert.Equal(OrdinaryBackingWriteAdmission.Admitted, reservation.TryAcquire(TimeSpan.Zero));
        reservation.BeginWriteAttempt(); // scalar unknown-completion control, not injected native bytes
        reservation.Dispose();
        Assert.Equal(CpuMappedPublicationPreparationResult.Rejected, state.Views.TryPrepareMappedPublication(
            original, [new(State.Address, new byte[] { 9 })], out _));
        Assert.Throws<InvalidOperationException>(() => reservation.BeginWriteAttempt());
        Assert.All(state.Bytes(), value => Assert.Equal((byte)0, value));
    }

    [Fact]
    public void StickyIncompleteLegacyFallbackCannotQualifyExclusiveBusyEnvelope()
    {
        using var state = new State(); var original = state.Capture();
        using var publication = state.Prepare(original, State.Address, [7]);
        using var reservation = Busy(state); publication.Dispose();
        var writers = typeof(HostBackingObject).GetField("_managedWrites", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(state.Backing)!;
        writers.GetType().GetField("coverageIncomplete", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(writers, 1);
        Assert.Equal(OrdinaryBackingWriteAdmission.Refused, reservation.TryAcquire(TimeSpan.Zero));
        Assert.Throws<InvalidOperationException>(() => reservation.BeginWriteAttempt());
        Assert.All(state.Bytes(), value => Assert.Equal((byte)0, value));
    }
    [Fact]
    public void RawProviderRevocationVetoesOriginalWriteEvenWhenOuterSnapshotIsUnchanged()
    {
        using var state = new State(); var original = state.Capture();
        using var publication = state.Prepare(original, State.Address, [7]);
        using var reservation = Busy(state); var outerSnapshot = state.Views.AliasSnapshot;
        publication.Dispose(); Assert.True(state.Host.Effect(State.Address, State.Page));
        Assert.Same(outerSnapshot, state.Views.AliasSnapshot);
        Assert.False(reservation.IsOriginalCurrent);
        Assert.Equal(OrdinaryBackingWriteAdmission.OriginalChanged, reservation.TryAcquire(TimeSpan.Zero));
        Assert.All(state.Bytes(), value => Assert.Equal((byte)0, value));
    }

    [Fact]
    public void GuestReadOnlyOrInvalidDestinationCannotBorrowPublicationBusyOwnership()
    {
        using var state = new State(HostPageProtection.ReadOnly);
        Assert.Equal(CpuMappedCopyResult.Copied, state.Views.TryReadMappedCopy(
            State.Address + State.Page, State.Page, State.Address + State.Page, new byte[8], out var original));
        using var publication = state.Prepare(original!, State.Address + State.Page, [7]);
        Assert.False(state.Views.TryWriteBacking(State.Address, new byte[] { 9 }, out var readOnly));
        Assert.Null(readOnly);
        Assert.False(state.Views.TryWriteBacking(State.Address + State.Page * 2, new byte[] { 9 }, out var invalid));
        Assert.Null(invalid);
        Assert.All(state.Bytes(), value => Assert.Equal((byte)0, value));
    }

    [Fact]
    public void ProviderIdentityChangeAfterAttemptPreventsSuccessAndCreatesUnknownHistory()
    {
        using var state = new State(); var original = state.Capture();
        using var publication = state.Prepare(original, State.Address, [7]);
        using var reservation = Busy(state); publication.Dispose();
        Assert.Equal(OrdinaryBackingWriteAdmission.Admitted, reservation.TryAcquire(TimeSpan.Zero));
        reservation.BeginWriteAttempt();
        Assert.True(state.Views.TryWriteBacking(State.Address + 32, new byte[] { 9 }));
        Assert.True(state.Host.Effect(State.Address, State.Page)); // participating actual ledger revocation
        Assert.False(reservation.CompleteWriteAttempt(true));
        reservation.Dispose();
        Assert.Equal(9, state.Bytes()[32]); // never retry or pretend rollback
        Assert.False(state.Views.IsMappedCopyObservationCurrent(original));
    }

    [Fact]
    public void WarmedOriginalSingleViewNormalWriteWithBusyCaptureDoesNotAllocate()
    {
        using var state = new State(); _ = state.Capture(); byte[] bytes = [1, 3, 5, 7];
        for (var index = 0; index < 10000; index++)
        {
            Assert.True(state.Views.TryWriteBacking(State.Address + 32, bytes, out var busy)); Assert.Null(busy);
        }
        var before = GC.GetAllocatedBytesForCurrentThread(); var succeeded = true; var noBusy = true;
        for (var index = 0; index < 10000; index++)
        {
            succeeded &= state.Views.TryWriteBacking(State.Address + 32, bytes, out var busy); noBusy &= busy is null;
        }
        var allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.True(succeeded); Assert.True(noBusy); Assert.Equal(0L, allocated);
        Assert.Equal(bytes, state.Bytes().AsSpan(32, bytes.Length).ToArray());
    }

    [Fact]
    public void PublicationBusyCapturesBeforeAdmissionWithoutProviderEpoch()
    {
        using var state = new State(); state.Host.DefersOriginalCapture = false;
        var original = state.Capture();
        using var publication = state.Prepare(original, State.Address, [7]);
        using var reservation = Busy(state);
        publication.Dispose();
        Assert.Equal(OrdinaryBackingWriteAdmission.Admitted, reservation.TryAcquire(TimeSpan.Zero));
    }

    [Fact]
    public void DeferredOriginalCaptureRefusesAnyMappingChangeSinceTheSample()
    {
        using var state = new State(); _ = state.Capture();
        var ledger = state.Host.Ledger;
        var sample = ledger.OrdinaryWriteMappingEpoch;
        var deferred = ledger.CaptureOrdinaryWriteIdentity(state.Backing, State.Address, 8, sample, out var changed);
        Assert.False(changed); Assert.True(deferred.HasValue); Assert.True(deferred.Value.IsCurrent);
        // A change elsewhere moves the epoch too: the capture cannot tell it from its own view.
        Assert.True(state.Host.Effect(State.Address + State.Page, State.Page));
        Assert.Null(ledger.CaptureOrdinaryWriteIdentity(state.Backing, State.Address, 8, sample, out changed));
        Assert.True(changed);
        Assert.True(ledger.CaptureOrdinaryWriteIdentity(
            state.Backing, State.Address, 8, ledger.OrdinaryWriteMappingEpoch, out changed).HasValue);
        Assert.False(changed);
    }

    [Theory]
    [InlineData(1, true)]
    [InlineData(16384, true)]
    [InlineData(16385, false)]
    public void ValueOriginalIdentityRetainsExactlyItsInitialViewSet(int bytes, bool unaffected)
    {
        using var state = new State(); _ = state.Capture();
        var original = state.Host.Ledger.CaptureOrdinaryWriteIdentity(state.Backing, State.Address, (ulong)bytes);
        Assert.True(original.HasValue); Assert.True(original.Value.IsCurrent);
        Assert.True(state.Host.Effect(State.Address + State.Page, State.Page));
        Assert.Equal(unaffected, original.Value.IsCurrent);
        // Copying the value cannot replace any original incarnation.
        var copied = original.Value; Assert.Equal(unaffected, copied.IsCurrent);
    }

}
