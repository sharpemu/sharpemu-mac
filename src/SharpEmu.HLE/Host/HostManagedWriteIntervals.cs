// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
namespace SharpEmu.HLE.Host;

// Ordinary participating alias writes only. Internal bookkeeping, never content
// authority: pre-enrollment/raw/native stores and copy serialization are absent.
internal sealed partial class HostManagedWriteIntervals
{
    internal const int Capacity = 128;
    private sealed class Slot
    {
        internal long State; // 0 empty, -identity reserved, +identity published
        internal ulong Offset, Count;
        internal bool Operation;
    }
    private readonly HostBackingObject backing;
    private readonly Slot[] slots = Enumerable.Range(0, Capacity).Select(_ => new Slot()).ToArray();
    private const long PublicationBit = 1L << 32;
    private long admission;
    private long nextIdentity, active, begun, completed, aborted;
    private int coverageIncomplete;
    private enum CoverageLoss { InvalidPhysicalBounds = 1, IdentityExhausted, BegunCounterExhausted,
        ActiveSlotCapacity, CompletionCounterExhausted, PublicationIdentityExhausted, PublicationQuarantined,
        InvalidHistoryRecord }
    private int firstCoverageLoss;
    internal int FirstCoverageLoss => Volatile.Read(ref firstCoverageLoss);
    private void LoseCoverage(CoverageLoss reason)
    {
        // Preserve the first cause before publishing the existing sticky veto.
        // No allocation, callback or logging in ordinary write/completion gates.
        Interlocked.CompareExchange(ref firstCoverageLoss, (int)reason, 0);
        Interlocked.Exchange(ref coverageIncomplete, 1);
    }

    // Dormant until the first observer. Nothing in production captures a revision, reserves a
    // publication or enrolls a read watch, so until something does, an ordinary writer only
    // registers itself on its own stripe of the dormant-writer lane (no admission, identity, slot,
    // history or counters) and its token has Slot == DormantSlot. An observer first sets
    // `observerSeen` (Interlocked.Exchange, a full fence) and then sums that lane; a writer increments
    // its stripe (a full fence) and then re-reads `observerSeen`, backing out to the full path if it
    // is set. So an observer that sees `observerSeen` set and a zero sum knows every dormant writer has
    // landed and every later one takes the full path; while the sum is not zero it refuses (it
    // never waits) and counts the refusal. Admission, identity and history then hold exactly as
    // before for every write after that point. SHARPEMU_WRITE_INTERVALS_DORMANT=0 starts observed.
    internal const int DormantSlot = -2;
    private const int DormantLane = 1;
    internal static readonly bool DormantEnabled =
        Environment.GetEnvironmentVariable("SHARPEMU_WRITE_INTERVALS_DORMANT") != "0";
    private readonly HostAccessStripes dormantWriters;
    private int observerSeen; // sticky
    private int dormantDrained; // an observer saw `observerSeen` set and no dormant writer: none can exist again
    private long observationRefusals;
    internal bool IsObserved => Volatile.Read(ref observerSeen) != 0;
    internal long ObservationRefusals => Volatile.Read(ref observationRefusals);

    internal HostManagedWriteIntervals(HostBackingObject backing)
    {
        this.backing = backing;
        dormantWriters = backing.AliasStripes ?? new HostAccessStripes();
        if (!DormantEnabled) observerSeen = dormantDrained = 1;
    }

    internal readonly record struct Token(HostManagedWriteIntervals? Owner, int Slot, long Identity)
    {
        internal long RejectedAdmission { get; init; }
        // The dormant-writer stripe cell a dormant token registered on (Slot == DormantSlot only).
        internal int DormantCell { get; init; }
        internal bool IsPublicationBusy => Owner is not null && Slot == -1 && Identity == 0 &&
            RejectedAdmission >= 0 && (RejectedAdmission & PublicationBit) != 0;
    }

    internal Token Begin(ulong offset, ulong count) => BeginCore(offset, count, false);
    internal Token BeginOperation() => BeginCore(0, backing.Size, true);
    private Token BeginCore(ulong offset, ulong count, bool operation)
    {
        if (count == 0) return default;
        if (Volatile.Read(ref observerSeen) == 0)
        {
            if (offset > backing.Size || count > backing.Size - offset)
            { LoseCoverage(CoverageLoss.InvalidPhysicalBounds); return default; }
            var cell = dormantWriters.Enter(DormantLane); // full fence before the re-read
            if (Volatile.Read(ref observerSeen) == 0) return new(this, DormantSlot, 0) { DormantCell = cell };
            dormantWriters.ExitCell(cell); // an observer arrived: take the full path
        }
        var observed = Volatile.Read(ref admission);
        var acquired = false;
        for (var attempt = 0; attempt < Capacity; attempt++)
        {
            // A losing no-byte writer must never perturb an exclusive owner:
            // exact original watch validation requires admission == PublicationBit.
            if (observed < 0 || observed >= PublicationBit - 1) break;
            var previous = Interlocked.CompareExchange(ref admission, observed + 1, observed);
            if (previous == observed) { acquired = true; break; }
            observed = previous; // exact failed CAS, never a later diagnostic sample
        }
        // Bounded ordinary contention can refuse without publication ownership.
        // Never return default here: that token preserves legacy byte permission.
        if (!acquired) return new(this, -1, 0) { RejectedAdmission = observed };
        if (offset > backing.Size || count > backing.Size - offset)
        { LoseCoverage(CoverageLoss.InvalidPhysicalBounds); Interlocked.Decrement(ref admission); return default; }
        if (Volatile.Read(ref coverageIncomplete) != 0) { Interlocked.Decrement(ref admission); return default; }
        var identity = Interlocked.Increment(ref nextIdentity);
        if (identity <= 0 || identity == long.MaxValue) { LoseCoverage(CoverageLoss.IdentityExhausted); Interlocked.Decrement(ref admission); return default; }
        var first = (int)((ulong)identity % Capacity);
        for (var visited = 0; visited < Capacity; visited++)
        {
            var index = (first + visited) % Capacity; var slot = slots[index];
            // One bounded attempt per preallocated slot; no spin, wait, query,
            // callback or allocation while the caller holds an outer gate.
            if (Interlocked.CompareExchange(ref slot.State, -identity, 0) != 0) continue;
            slot.Offset = offset; slot.Count = count; slot.Operation = operation;
            if (!operation)
            {
                InvalidateReadWatches(offset, count); // BEFORE any participating destination bytes.
                Interlocked.Increment(ref active);
                if (Interlocked.Increment(ref begun) <= 0) LoseCoverage(CoverageLoss.BegunCounterExhausted);
            }
            Volatile.Write(ref slot.State, identity);
            return new(this, index, identity);
        }
        LoseCoverage(CoverageLoss.ActiveSlotCapacity);
        Interlocked.Decrement(ref admission);
        return default; // preserve legacy write, explicitly lose bookkeeping coverage
    }

    internal bool End(Token token, bool succeeded)
    {
        if (token.Owner is null) return token.Identity == 0 && token.Slot == 0; // disabled/nonempty tracking decline
        if (token.Slot == DormantSlot && token.Identity == 0 && ReferenceEquals(token.Owner, this))
        {
            // No watch can exist while a dormant writer is registered, and no observer has a
            // revision this write could fall inside: there is no history to record.
            dormantWriters.ExitCell(token.DormantCell);
            return true;
        }
        if (!ReferenceEquals(token.Owner, this) || token.Identity <= 0 || (uint)token.Slot >= Capacity)
            return false;
        // Consume this exact operation before reuse. A copied/stale token cannot
        // release a later operation or repeat an unknown completion.
        var operation = slots[token.Slot].Operation;
        var offset = slots[token.Slot].Offset;
        var count = slots[token.Slot].Count;
        if (Interlocked.CompareExchange(ref slots[token.Slot].State, 0, token.Identity) != token.Identity)
            return false;
        // Copy the owned slot's fields before consuming it: a later operation
        // may immediately reuse the slot. Admission stays held through history.
        if (!succeeded) InvalidateReadWatches(offset, count, unknown: true);
        RecordCompletedHistory(token.Identity, offset, count, operation, succeeded);
        if (!operation)
        {
            Interlocked.Decrement(ref active);
            var total = succeeded ? Interlocked.Increment(ref completed) : Interlocked.Increment(ref aborted);
            if (total <= 0) LoseCoverage(CoverageLoss.CompletionCounterExhausted);
        }
        Interlocked.Decrement(ref admission);
        return true;
    }


    internal static bool AllowsBytes(Token token) => token.Owner is null || token.Slot >= 0 || token.Slot == DormantSlot;
    // A dormant operation envelope already makes every observer refuse until it ends, so the
    // byte intervals inside it need no registration of their own.
    internal static bool IsDormant(Token token) => token.Owner is not null && token.Slot == DormantSlot;

    // Every observer calls this before it reads admission, identity or history. True once no
    // dormant writer can be in flight; false (counted, never a wait) while one is.
    private bool TryObserve()
    {
        if (Volatile.Read(ref dormantDrained) != 0) return true;
        if (Interlocked.Exchange(ref observerSeen, 1) == 0)
        {
            // Once per backing, on stdout (game.log): from here on every ordinary write pays the
            // full ledger again, so a production observer must not arrive unnoticed.
            try { Console.Out.WriteLine($"[MEM] write-intervals observed backing={backing.Identity}: ordinary writes now take admission, identity and history"); }
            catch (Exception) { } // a log failure must not change the observer's outcome
        }

        if (dormantWriters.Sum(DormantLane) != 0)
        {
            Interlocked.Increment(ref observationRefusals);
            return false;
        }

        Volatile.Write(ref dormantDrained, 1);
        return true;
    }

    // Participating writers only, never a certificate for native/JIT/raw stores.
    // The count is acquired before allocating an interval identity; losing
    // writers leave PublicationBit unchanged and never receive byte permission.
    internal bool TryCaptureRevision(out long revision)
    {
        revision = -1;
        if (!TryObserve()) return false;
        if (Volatile.Read(ref coverageIncomplete) != 0 || Volatile.Read(ref admission) != 0) return false;
        var found = Volatile.Read(ref nextIdentity);
        if (found < 0 || Volatile.Read(ref admission) != 0 || Volatile.Read(ref coverageIncomplete) != 0) return false;
        revision = found; return true;
    }
    internal bool TryReservePublication(long originalRevision, ReadOnlySpan<PhysicalRun> runs, ReadWatch? originalWatch, AdmissionOwner? owner)
        => TryReservePublication(originalRevision, runs, originalWatch, owner, out _);

    internal bool TryReservePublication(long originalRevision, ReadOnlySpan<PhysicalRun> runs,
        ReadWatch? originalWatch, AdmissionOwner? owner, out long rejectedAdmission)
    {
        rejectedAdmission = 0;
        if (originalRevision < 0 || originalRevision == long.MaxValue || Volatile.Read(ref coverageIncomplete) != 0)
        {
            TracePublicationDecline("invalid-original-revision-or-incomplete", originalRevision);
            return false;
        }
        if (!TryObserve())
        {
            TracePublicationDecline("observation-activating", originalRevision);
            return false;
        }
        var previousAdmission = Interlocked.CompareExchange(ref admission, PublicationBit, 0);
        if (previousAdmission != 0)
        {
            rejectedAdmission = previousAdmission; // exact failed CAS, no later sample
            if (HostPublicationDiagnostics.Enabled)
                HostPublicationDiagnostics.Declined("writer-reservation", "admission-busy", originalRevision,
                    Volatile.Read(ref nextIdentity), previousAdmission, Volatile.Read(ref coverageIncomplete));
            return false;
        }
        PublishAdmissionOwner(owner, CpuMappedAdmissionOwnerPhase.Reserved);
        try
        {
            // Validate an enrolled ORIGINAL physical veto only AFTER excluding
            // writers. A revoked/released/foreign watch cannot borrow newer history.
            if ((originalWatch is not null ? originalWatch.CoversDuringPublication(this, runs) :
                HasDisjointCompletedHistory(originalRevision, runs)) && Volatile.Read(ref coverageIncomplete) == 0) return true;
            ReleasePublication(owner);
            TracePublicationDecline("completed-history-or-incomplete", originalRevision);
            return false;
        }
        catch { owner?.Unknown(); throw; } // Preserve any existing uncertain exclusion.
    }

    private void TracePublicationDecline(string reason, long originalRevision)
    {
        if (!HostPublicationDiagnostics.Enabled) return;
        HostPublicationDiagnostics.Declined("writer-reservation", reason, originalRevision,
            Volatile.Read(ref nextIdentity), Volatile.Read(ref admission), Volatile.Read(ref coverageIncomplete));
    }
    internal long RecordPublicationWrite(ReadOnlySpan<PhysicalRun> runs, ReadWatch? unpublishedSuccessor = null)
    {
        foreach (var run in runs) InvalidateReadWatches(run.Offset, run.Count, unpublishedSuccessor: unpublishedSuccessor);
        var revision = Interlocked.Increment(ref nextIdentity);
        if (revision <= 0 || revision == long.MaxValue) LoseCoverage(CoverageLoss.PublicationIdentityExhausted);
        else RecordHistory(revision, runs, false);
        return revision;
    }
    internal void ReleasePublication(AdmissionOwner? owner)
    {
        if (owner is not null) DetachAdmissionOwner(owner);
        Interlocked.Add(ref admission, -PublicationBit); // unchanged count/release
        if (owner is not null) Volatile.Write(ref owner.SubtractionReturned, 1);
        // Terminal outcome belongs to the caller AFTER exact cleanup returns.
    }
    internal void QuarantinePublication() => LoseCoverage(CoverageLoss.PublicationQuarantined);

    // Test-only ordinary observation. Never used for publication, permission,
    // GPU/fault admission or content-currentness. Concurrent snapshot is fuzzy:
    // reservations may be unpublished and completions may be between counters.
    private (ulong Offset, ulong Count, long Identity)[] ActiveForControl()
    {
        var found = new List<(ulong, ulong, long)>();
        foreach (var slot in slots)
        {
            var identity = Volatile.Read(ref slot.State);
            if (identity <= 0 || slot.Operation) continue;
            var offset = slot.Offset; var count = slot.Count;
            if (Volatile.Read(ref slot.State) == identity) found.Add((offset, count, identity));
        }
        return found.ToArray();
    }
}
