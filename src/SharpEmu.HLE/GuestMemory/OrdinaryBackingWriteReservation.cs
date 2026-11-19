// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
using System.Diagnostics;
using SharpEmu.HLE.Host;

namespace SharpEmu.HLE.GuestMemory;

public enum OrdinaryBackingWriteAdmission { Admitted, OriginalChanged, TimedOut, Refused }
public readonly record struct OrdinaryBackingWriteCurrentnessFailure(string? Reason,
    OrdinaryWriteIdentityFailure Provider);

// Busy-only ordinary coordination. Not a signal, permission or content receipt.
// Keeps the ORIGINAL backing pin/snapshot; never resolves a replacement mapping.
public sealed class OrdinaryBackingWriteReservation : IDisposable
{
    private readonly SharedBackingViews owner;
    private readonly HostBackingObject backing;
    private readonly object originalSnapshot;
    private readonly HostMappedCopyLedger.OrdinaryWriteIdentity originalMapping;
    private readonly int thread = Environment.CurrentManagedThreadId;
    private HostManagedWriteIntervals.Token operation;
    private int state; // 0 prepared, 1 acquiring, 2 finished acquisition, 3 consumed
    private bool admitted, attempted, completedAttempt, succeeded;
    internal OrdinaryBackingWriteReservation(SharedBackingViews owner, HostBackingObject backing,
        object originalSnapshot, HostMappedCopyLedger.OrdinaryWriteIdentity originalMapping, long rejectedAdmission)
    { this.owner = owner; this.backing = backing; this.originalSnapshot = originalSnapshot; this.originalMapping = originalMapping; RejectedAdmission = rejectedAdmission; }
    public long RejectedAdmission { get; }
    public bool IsOriginalCurrent => owner.IsAvailable && ReferenceEquals(owner.AliasSnapshot, originalSnapshot) && originalMapping.IsCurrent;

    // Same ordered check used ONLY instead of the ordinary property when
    // optional attribution is enabled. No late sample repairs the decision.
    public bool TryIsOriginalCurrent(out OrdinaryBackingWriteCurrentnessFailure failure)
    {
        failure = default;
        if (!owner.IsAvailable) { failure = new("original-shared-views-unavailable", default); return false; }
        if (!ReferenceEquals(owner.AliasSnapshot, originalSnapshot))
        { failure = new("original-shared-snapshot-replaced", default); return false; }
        if (originalMapping.TryIsCurrent(out var provider)) return true;
        failure = new("original-provider-identity-changed", provider);
        return false;
    }

    // Caller MUST drop all outer mapping/stripe/provider gates first. No native
    // pointers are read while waiting; the backing pin owns lifetime only.
    public OrdinaryBackingWriteAdmission TryAcquire(TimeSpan timeout) => TryAcquire(timeout, false, out _);

    // Explicit attribution replaces the SAME first Boolean currentness check.
    // A refused acquisition is never queried again to explain an older result.
    public OrdinaryBackingWriteAdmission TryAcquire(TimeSpan timeout, out OrdinaryBackingWriteCurrentnessFailure failure)
        => TryAcquire(timeout, true, out failure);

    private OrdinaryBackingWriteAdmission TryAcquire(TimeSpan timeout, bool classify,
        out OrdinaryBackingWriteCurrentnessFailure failure)
    {
        failure = default;
        RequireThread(); owner.RequireOrdinaryWriteWaitOutsideGates();
        if (timeout < TimeSpan.Zero || timeout > TimeSpan.FromMilliseconds(500))
            throw new ArgumentOutOfRangeException(nameof(timeout));
        if (Interlocked.CompareExchange(ref state, 1, 0) != 0)
            throw new InvalidOperationException("Ordinary write admission is one use.");
        try
        {
            var deadline = Stopwatch.GetTimestamp() + (long)(timeout.TotalSeconds * Stopwatch.Frequency);
            while (true)
            {
                if (!(classify ? TryIsOriginalCurrent(out failure) : IsOriginalCurrent))
                    return OrdinaryBackingWriteAdmission.OriginalChanged;
                operation = backing.BeginManagedWriteOperation();
                if (operation.Owner is not null && (operation.Slot >= 0 && operation.Identity > 0 || HostManagedWriteIntervals.IsDormant(operation)))
                {
                    admitted = true;
                    return (classify ? TryIsOriginalCurrent(out failure) : IsOriginalCurrent)
                        ? OrdinaryBackingWriteAdmission.Admitted : OrdinaryBackingWriteAdmission.OriginalChanged;
                }
                if (!operation.IsPublicationBusy) return OrdinaryBackingWriteAdmission.Refused;
                if (Stopwatch.GetTimestamp() >= deadline) return OrdinaryBackingWriteAdmission.TimedOut;
                Thread.Sleep(1); // Bounded ordinary wait, never called from signal recovery.
            }
        }
        finally { Volatile.Write(ref state, 2); }
    }

    // Called immediately before the single existing write attempt. A false or
    // thrown write can be partial: its completion remains unknown, never retried.
    public void BeginWriteAttempt()
    {
        RequireThread();
        if (Volatile.Read(ref state) != 2 || !admitted || attempted || !IsOriginalCurrent)
            throw new InvalidOperationException("Ordinary write lost original admission.");
        attempted = true;
    }
    public bool CompleteWriteAttempt(bool result)
    {
        RequireThread();
        if (Volatile.Read(ref state) != 2 || !attempted || completedAttempt)
            throw new InvalidOperationException("Ordinary write completion is one use.");
        completedAttempt = true; succeeded = result && IsOriginalCurrent;
        return succeeded;
    }
    public void Dispose()
    {
        RequireThread();
        if (Volatile.Read(ref state) == 1) throw new InvalidOperationException("Cannot consume live ordinary admission acquisition.");
        if (Interlocked.Exchange(ref state, 3) == 3) return;
        try { if (admitted) HostBackingObject.EndManagedWrite(operation, !attempted || (completedAttempt && succeeded)); }
        finally { owner.ExitOrdinaryWriteReceiptAccess(); }
    }
    private void RequireThread()
    { if (Environment.CurrentManagedThreadId != thread) throw new InvalidOperationException("Ordinary write reservation belongs to its original thread."); }
}
