// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.HLE.Host;

/// <summary>Provider-private participation ledger. The provider holds Gate across each
/// native mapping effect and its publication. Copies release Gate before touching bytes.
/// Public scalar metadata never authorizes an operation on a different ledger.</summary>
public sealed unsafe partial class HostMappedCopyLedger
{
    public const int MaximumMappings = 65536;
    public const int MaximumCopyBytes = 64 * 1024 * 1024;
    private readonly object gate;
    private readonly ulong pageSize;
    private readonly SortedList<ulong, View> views = new();
    private sealed class Alias(HostAddressRange range)
    {
        internal readonly HostAddressRange Range = range;
        internal int ReleaseState; // 0 active, 1 pending, 2 consumed/unknown
        internal long NativeMappingVersion;
        internal int NativeAliasUnavailable;
    }
    private readonly Dictionary<HostBackingObject, Alias> aliases = new();
    private sealed class Capture : ICpuNativeAliasObservation
    {
        internal readonly HostMappedCopyLedger Owner;
        internal readonly View[] States;
        internal readonly long[] Incarnations;
        internal long WriterRevision = -1;
        internal Alias? NativeAlias;
        internal long NativeAliasVersion;
        public bool ExcludesNativeAliases => NativeAlias is { } alias &&
            NativeAliasVersion == Volatile.Read(ref alias.NativeMappingVersion) &&
            Volatile.Read(ref alias.ReleaseState) == 0 && Volatile.Read(ref alias.NativeAliasUnavailable) == 0 &&
            !Volatile.Read(ref Owner.unknownAlias);
        internal ReadWatchOwnership? WatchOwnership;
        internal HostManagedWriteIntervals.ReadWatch? ReadWatch => WatchOwnership?.Watch;
        internal void ReleaseWatch() => WatchOwnership?.Release();
        internal Capture(HostMappedCopyLedger owner, View[] states, long[]? originalIncarnations = null)
        {
            Owner = owner; States = states;
            Incarnations = originalIncarnations is null ? new long[states.Length] : (long[])originalIncarnations.Clone();
            if (originalIncarnations is null)
                for (var index = 0; index < states.Length; index++) Incarnations[index] = states[index].Incarnation;
        }
    }
    private long nextIncarnation;
    private int readWatchCount;
    private const int MaximumReadWatches = 128;
    private bool unknownAlias;
    private int aliasEffectsInFlight;
    // Bumped under Gate after each change an ordinary write identity is captured from:
    // the views, the aliases, the unknown-alias quarantine and a view's copy, access,
    // incarnation or revocation. A write samples it before its first admission and
    // captures only once refused; an unchanged value means nothing it reads has moved.
    private long mappingEpoch;
    // Ordinary test-only pause, default null. Never called under provider Gate,
    // on a signal path, or by legacy reads. No public setter/callback contract.
    private Action? afterCaptureForControl = null;
    // Ordinary-copy control seam: initial revision sample, before enrollment or bytes.
    private Action<bool, long>? afterWriterRevisionForControl = null;

    public HostMappedCopyLedger(object providerGate, ulong nativePageSize)
    {
        ArgumentNullException.ThrowIfNull(providerGate);
        if (nativePageSize == 0 || (nativePageSize & (nativePageSize - 1)) != 0)
            throw new ArgumentOutOfRangeException(nameof(nativePageSize));
        gate = providerGate; pageSize = nativePageSize;
    }

    public ulong NativePageSize => pageSize;

    public sealed class View
    {
        internal View(HostMappedCopyLedger owner, HostBackingObject backing, ulong address,
            ulong size, ulong offset, GuestPageProtection access, long incarnation)
        { Owner = owner; Backing = backing; Address = address; Size = size; Offset = offset; Access = access; Incarnation = incarnation; }
        internal readonly HostMappedCopyLedger Owner;
        internal readonly HostBackingObject Backing;
        internal readonly ulong Address, Size, Offset;
        internal readonly GuestPageProtection Access;
        internal long Incarnation; // monotonic copy incarnation; physical facts above never mutate
        internal int CopyAvailable = 1;
        internal bool GuestAccessKnown = true;
        internal int Revoked;
    }

    private void RequireGate()
    {
        if (!Monitor.IsEntered(gate)) throw new InvalidOperationException("Provider mapping gate is required.");
    }

    public bool CanMakeAnonymousPlacement() { RequireGate(); return !unknownAlias && aliasEffectsInFlight == 0; }
    public bool IsInternalAliasAddress(ulong address)
    {
        RequireGate();
        foreach (var alias in aliases.Values)
            if (address >= alias.Range.Address && address - alias.Range.Address < alias.Range.Size) return true;
        return false;
    }

    public void PreflightBackingPublication()
    { RequireGate(); aliases.EnsureCapacity(aliases.Count + 1); }

    public void RegisterBacking(HostBackingObject backing)
    {
        RequireGate();
        if (!backing.HasOwnedLifetime || backing.Identity <= 0 || backing.IsClosing || backing.IsDisposed ||
            !TryEffectRange(backing.AliasBase, backing.Size, out var range))
            throw new ArgumentException("Canonical backing alias has no complete native footprint.", nameof(backing));
        aliases.Add(backing, new(range));
        NoteMappingChangedLocked();
    }

    // Setup's own cleanup may have an unknown native outcome before a backing
    // object was publishable. Do not allocate to retain this quarantine.
    public void QuarantineUnknownAlias() { RequireGate(); unknownAlias = true; RevokeAll(); NoteMappingChangedLocked(); }

    public View? PrepareView(HostBackingObject backing, ulong address, ulong size,
        ulong offset, HostPageProtection protection)
    {
        RequireGate();
        var access = AccessOf(protection);
        var alias = aliases.GetValueOrDefault(backing);
        if (alias is not null)
        {
            // Even an unpublishable view can be placed by a provider. Expire
            // old proofs before its kernel effect, and quarantine unknown geometry.
            if (alias.NativeMappingVersion == long.MaxValue) Volatile.Write(ref alias.NativeAliasUnavailable, 1);
            else Volatile.Write(ref alias.NativeMappingVersion, alias.NativeMappingVersion + 1);
        }
        if (unknownAlias || !aliases.ContainsKey(backing) || backing.IsClosing || backing.IsDisposed ||
            !HostViewMemory.IsValidRange(address, size) || address % pageSize != 0 || size % pageSize != 0 ||
            offset % pageSize != 0 || offset > backing.Size || size > backing.Size - offset ||
            access == GuestPageProtection.None || HasOverlappingPublicationLocked(address, size) || views.Count >= MaximumMappings)
        {
            if (alias is not null) Volatile.Write(ref alias.NativeAliasUnavailable, 1);
            return null;
        }
        if (nextIncarnation == long.MaxValue)
        {
            if (alias is not null) Volatile.Write(ref alias.NativeAliasUnavailable, 1);
            RevokeAll(); return null;
        } // No process-local wrap/reuse.
        if (views.Capacity < views.Count + 1)
            views.Capacity = Math.Min(MaximumMappings, Math.Max(4, views.Capacity * 2));
        return new(this, backing, address, size, offset, access, ++nextIncarnation);
    }

    public void PublishView(View? prepared)
    {
        RequireGate();
        if (prepared is null) return;
        if (!ReferenceEquals(prepared.Owner, this) || prepared.Revoked != 0 ||
            prepared.Backing.IsClosing || prepared.Backing.IsDisposed)
            throw new InvalidOperationException("Mapping publication has no exact live provider owner.");
        views.Add(prepared.Address, prepared);
        NoteMappingChangedLocked();
    }

    /// <summary>Before ANY participating kernel effect. Internal aliases are
    /// excluded; failed effects do not reopen old view incarnations.</summary>
    public bool BeforeNativeEffect(ulong address, ulong size)
    {
        if (!TryBeginNativeEffect(address, size, out var effect)) return false;
        Revoke(effect);
        return true;
    }

    private bool TryBeginNativeEffect(ulong address, ulong size, out HostAddressRange effect)
    {
        RequireGate();
        effect = default;
        if (unknownAlias || !TryEffectRange(address, size, out effect) ||
            HasOverlappingPublicationLocked(effect.Address, effect.Size)) return false;
        foreach (var alias in aliases.Values)
            if (Overlaps(effect, alias.Range)) return false;
        if (protectionEffectSequence == long.MaxValue) { RevokeAll(); return false; }
        protectionEffectSequence++;
        return true;
    }

    // The provider's private callback alone owns this exact backing object.
    // It revokes under Gate, executes its alias unmap OUTSIDE Gate after the
    // existing backing pins drain, then records only the actual munmap outcome.
    public bool BeginBackingRelease(HostBackingObject backing)
    {
        RequireGate();
        if (!backing.IsClosing || !aliases.TryGetValue(backing, out var alias) || alias.ReleaseState != 0) return false;
        alias.ReleaseState = 1;
        for (var index = views.Count - 1; index >= 0; index--)
            if (ReferenceEquals(views.Values[index].Backing, backing)) RevokeAt(index);
        aliasEffectsInFlight++;
        return true;
    }

    public void CompleteBackingAliasRelease(HostBackingObject backing, bool nativeSuccess)
    {
        RequireGate();
        if (aliasEffectsInFlight <= 0 || !aliases.TryGetValue(backing, out var alias) || alias.ReleaseState != 1)
            throw new InvalidOperationException("No exact canonical alias effect is in flight.");
        // Consume the exact pending alias before either successful removal or
        // sticky unknown quarantine. A replay cannot steal another alias count.
        alias.ReleaseState = 2;
        aliasEffectsInFlight--;
        if (nativeSuccess) { aliases.Remove(backing); NoteMappingChangedLocked(); }
        else QuarantineUnknownAlias();
    }

    public CpuMappedCopyResult TryReadMappedCopy(HostBackingObject backing, ulong footprintAddress,
        ulong footprintByteCount, ulong sourceAddress, Span<byte> destination,
        out CpuMappedCopyObservation? observation)
        => TryReadMappedCopyCore(backing, footprintAddress, footprintByteCount, sourceAddress, destination, false, out observation);

    private CpuMappedCopyResult TryReadMappedCopyCore(HostBackingObject backing, ulong footprintAddress,
        ulong footprintByteCount, ulong sourceAddress, Span<byte> destination, bool withWriteWatch,
        out CpuMappedCopyObservation? observation, ReadWatchAdmissionWait? admissionWait = null)
    {
        CpuMappedCopyDiagnostics.Failure diagnostic = default;
        try
        {
            return TryReadMappedCopyImplementation(backing, footprintAddress, footprintByteCount,
                sourceAddress, destination, withWriteWatch, out observation, ref diagnostic, admissionWait);
        }
        finally
        {
            // Both rows describe actual refusals, not a later Current sample.
            // Formatting happens after every implementation gate/pin unwinds.
            if (admissionWait is not null) CpuMappedCopyDiagnostics.Emit(admissionWait.FirstRefusal);
            CpuMappedCopyDiagnostics.Emit(diagnostic);
        }
    }

    private CpuMappedCopyResult TryReadMappedCopyImplementation(HostBackingObject backing, ulong footprintAddress,
        ulong footprintByteCount, ulong sourceAddress, Span<byte> destination, bool withWriteWatch,
        out CpuMappedCopyObservation? observation, ref CpuMappedCopyDiagnostics.Failure diagnostic,
        ReadWatchAdmissionWait? admissionWait)
    {
        var copyBytes = destination.Length;
        Capture? ownedCapture = null;
        var keepWatch = false;
        observation = null;
        if (!HostViewMemory.IsValidRange(footprintAddress, footprintByteCount) || destination.Length == 0 ||
            destination.Length > MaximumCopyBytes || !HostViewMemory.IsValidRange(sourceAddress, (ulong)destination.Length) ||
            sourceAddress < footprintAddress || sourceAddress - footprintAddress > footprintByteCount ||
            (ulong)destination.Length > footprintByteCount - (sourceAddress - footprintAddress))
            return Refuse(ref diagnostic, "request-geometry", CpuMappedCopyResult.Rejected);
        if (Monitor.IsEntered(gate)) throw new InvalidOperationException("Mapped copy must not run under the provider gate.");
        if (!backing.TryEnterAccess()) return Refuse(ref diagnostic, "backing-lifetime", CpuMappedCopyResult.Rejected);
        try
        {
            // Enrollment does not certify pre-existing/native writers, serialize
            // the copy, or change mapping-only IsCurrent semantics.
            backing.EnrollManagedWrites();
            View[] selected;
            CpuMappedCopyObservation captured;
            var writers = backing.ManagedWrites!;
            var hasWriterRevision = writers.TryCaptureRevision(out var writerRevision);
            afterWriterRevisionForControl?.Invoke(hasWriterRevision, writerRevision);
            lock (gate)
            {
                if (unknownAlias) return Refuse(ref diagnostic, "unknown-alias", CpuMappedCopyResult.Unavailable);
                if (!aliases.ContainsKey(backing)) return Refuse(ref diagnostic, "backing-not-enrolled", CpuMappedCopyResult.Unavailable);
                var states = new List<View>();
                var segments = new List<CpuMappingFootprintSegment>();
                var cursor = footprintAddress; var end = footprintAddress + footprintByteCount;
                while (cursor < end)
                {
                    var index = FindAtOrBelow(cursor);
                    if (index < 0) return Refuse(ref diagnostic, "no-view-at-cursor", CpuMappedCopyResult.Rejected, cursor, states.Count);
                    var view = views.Values[index];
                    if (view.Revoked != 0) return Refuse(ref diagnostic, "view-revoked", CpuMappedCopyResult.Rejected, cursor, states.Count, view);
                    if (view.CopyAvailable == 0) return Refuse(ref diagnostic, "view-copy-unavailable", CpuMappedCopyResult.Rejected, cursor, states.Count, view);
                    if (!view.GuestAccessKnown) return Refuse(ref diagnostic, "view-guest-access-unknown", CpuMappedCopyResult.Rejected, cursor, states.Count, view);
                    if (!ReferenceEquals(view.Backing, backing)) return Refuse(ref diagnostic, "view-other-backing", CpuMappedCopyResult.Rejected, cursor, states.Count, view);
                    if (cursor < view.Address || cursor - view.Address >= view.Size) return Refuse(ref diagnostic, "view-gap", CpuMappedCopyResult.Rejected, cursor, states.Count, view);
                    if ((view.Access & GuestPageProtection.Read) == 0) return Refuse(ref diagnostic, "view-not-readable", CpuMappedCopyResult.Rejected, cursor, states.Count, view);
                    if (states.Count == CpuMappedCopyObservation.MaximumSegments) return Refuse(ref diagnostic, "segment-bound", CpuMappedCopyResult.Rejected, cursor, states.Count, view);
                    var bytes = Math.Min(end - cursor, view.Size - (cursor - view.Address));
                    states.Add(view);
                    segments.Add(new(view.Address, view.Size, cursor, bytes, view.Incarnation,
                        backing.Identity, backing.Size, view.Offset, pageSize, view.Access));
                    cursor += bytes;
                }
                selected = states.ToArray();
                var identity = new Capture(this, selected);
                if (withWriteWatch && NativeAliasesCovered(backing, selected, sourceAddress, (ulong)destination.Length))
                {
                    identity.NativeAlias = aliases[backing];
                    identity.NativeAliasVersion = identity.NativeAlias.NativeMappingVersion;
                }
                captured = new(identity, footprintAddress, footprintByteCount, sourceAddress,
                    (ulong)destination.Length, segments.ToArray());
            }
            ownedCapture = (Capture)captured.OwnerIdentity;
            if (withWriteWatch)
            {
                // Reserve a bounded ledger owner before exposing a receipt.
                var watchOwnership = new ReadWatchOwnership(this); // Only watched copies are finalizable.
                ownedCapture.WatchOwnership = watchOwnership;
                if (Interlocked.Increment(ref readWatchCount) > MaximumReadWatches)
                { Interlocked.Decrement(ref readWatchCount); return Refuse(ref diagnostic, "watch-capacity", CpuMappedCopyResult.Rejected); }
                watchOwnership.Charged = 1;
                var physical = new List<HostManagedWriteIntervals.PhysicalRun>();
                var lastSource = sourceAddress + (ulong)destination.Length;
                foreach (var view in selected)
                {
                    var first = Math.Max(sourceAddress, view.Address);
                    var last = Math.Min(lastSource, view.Address + view.Size);
                    if (first < last) physical.Add(new(view.Offset + first - view.Address, last - first));
                }
                var originalPhysical = physical.ToArray(); // FIRST runs; never rebuilt during waiting
                watchOwnership.Watch = writers.TryCreateReadWatch(originalPhysical, out var enrollmentReason, out var enrollmentAdmission);
                if (ownedCapture.ReadWatch is null && admissionWait is not null &&
                    enrollmentReason == "watch-enrollment-admission-busy" && enrollmentAdmission > 0)
                {
                    var firstAdmission = enrollmentAdmission;
                    if (CpuMappedCopyDiagnostics.Enabled)
                        admissionWait.FirstRefusal = new("ledger-prebyte-enrollment-first", enrollmentReason,
                            CpuMappedCopyResult.Rejected, footprintAddress, footprintByteCount, sourceAddress,
                            copyBytes, true, backing.Identity, EnrollmentAdmission: firstAdmission);
                    while (ownedCapture.ReadWatch is null)
                    {
                        // Admission is a refusal, NEVER a read entitlement or
                        // proof that a foreign/lost reservation will complete.
                        if (!CanWaitForMappedReadWatchEnrollment || !IsCurrent(backing, captured) ||
                            !admissionWait.OriginalStillCurrent() || !CanWaitForMappedReadWatchEnrollment ||
                            !IsCurrent(backing, captured))
                            return Refuse(ref diagnostic, "prebyte-original-veto", CpuMappedCopyResult.Rejected,
                                enrollmentAdmission: firstAdmission);
                        if (Environment.TickCount64 >= admissionWait.Deadline)
                            return Refuse(ref diagnostic, "prebyte-admission-timeout", CpuMappedCopyResult.Rejected,
                                enrollmentAdmission: firstAdmission);
                        Thread.Sleep(1); // no presenter/provider/backing/wrapper gates
                        if (!CanWaitForMappedReadWatchEnrollment || !IsCurrent(backing, captured) ||
                            !admissionWait.OriginalStillCurrent() || !CanWaitForMappedReadWatchEnrollment ||
                            !IsCurrent(backing, captured))
                            return Refuse(ref diagnostic, "prebyte-original-veto", CpuMappedCopyResult.Rejected,
                                enrollmentAdmission: firstAdmission);
                        if (Environment.TickCount64 >= admissionWait.Deadline)
                            return Refuse(ref diagnostic, "prebyte-admission-timeout", CpuMappedCopyResult.Rejected,
                                enrollmentAdmission: firstAdmission);
                        watchOwnership.Watch = writers.TryCreateReadWatch(originalPhysical, out enrollmentReason, out enrollmentAdmission);
                        if (ownedCapture.ReadWatch is null &&
                            (enrollmentReason != "watch-enrollment-admission-busy" || enrollmentAdmission <= 0))
                            return Refuse(ref diagnostic, enrollmentReason!, CpuMappedCopyResult.Rejected,
                                enrollmentAdmission: firstAdmission);
                    }
                }
                if (ownedCapture.ReadWatch is null) return Refuse(ref diagnostic, enrollmentReason!, CpuMappedCopyResult.Rejected,
                    enrollmentAdmission: enrollmentAdmission);
            }
            // No virtual pointer, permission repair, provider query, callback or
            // mapping gate is used here. The exact backing pin owns the alias.
            afterCaptureForControl?.Invoke();
            if (admissionWait is not null && (!CanWaitForMappedReadWatchEnrollment ||
                !IsCurrent(backing, captured) || ownedCapture.ReadWatch?.IsCurrent != true ||
                !admissionWait.OriginalStillCurrent() || !CanWaitForMappedReadWatchEnrollment ||
                !IsCurrent(backing, captured) ||
                ownedCapture.ReadWatch?.IsCurrent != true))
                return Refuse(ref diagnostic, "prebyte-original-veto", CpuMappedCopyResult.Rejected);
            var sourceEnd = sourceAddress + (ulong)destination.Length;
            foreach (var view in selected)
            {
                var first = Math.Max(sourceAddress, view.Address);
                var last = Math.Min(sourceEnd, view.Address + view.Size);
                if (first >= last) continue;
                var offset = view.Offset + first - view.Address;
                new ReadOnlySpan<byte>((void*)(backing.AliasBase + offset), checked((int)(last - first)))
                    .CopyTo(destination.Slice(checked((int)(first - sourceAddress))));
            }
            if (!IsCurrent(backing, captured)) return Refuse(ref diagnostic, "post-copy-original-mapping", CpuMappedCopyResult.Rejected);
            if (admissionWait is not null && (!CanWaitForMappedReadWatchEnrollment ||
                !admissionWait.OriginalStillCurrent() || !CanWaitForMappedReadWatchEnrollment || !IsCurrent(backing, captured)))
                return Refuse(ref diagnostic, "post-copy-original-veto", CpuMappedCopyResult.Rejected);
            if (withWriteWatch)
            {
                var originalWatch = ownedCapture.ReadWatch!;
                string? firstWatchGuard = null;
                long firstAdmission = -1;
                CpuMappedAdmissionOwnerDiagnostic? firstOwner = null;
                bool current;
                if (CpuMappedCopyDiagnostics.Enabled)
                    current = originalWatch.TryIsCurrent(out firstWatchGuard, out firstAdmission, out firstOwner);
                else current = originalWatch.IsCurrent; // exact legacy query when diagnostics are off
                if (!current)
                {
                    var result = Refuse(ref diagnostic, "post-copy-original-watch", CpuMappedCopyResult.Rejected);
                    // Detached values from that ONE Boolean query, retained
                    // before finally releases this private original watch.
                    // No Status/classification, late query or wait authority.
                    diagnostic = diagnostic with { PostCopyWatchGuard = firstWatchGuard,
                        PostCopyPrimaryAdmission = firstAdmission >= 0 ? firstAdmission : null,
                        PostCopyAdmissionOwner = firstOwner };
                    return result;
                }
                // This value was fixed under exclusive watch enrollment BEFORE
                // bytes, including when the earlier ordinary sample was busy.
                Volatile.Write(ref ownedCapture.WriterRevision, originalWatch.EnrollmentRevision);
            }
            else
            {
                // Mapping-only copies still succeed if participating writes raced.
                // Such an observation has no publication history, permanently.
                if (hasWriterRevision && writers.TryCaptureRevision(out var afterRevision) && afterRevision == writerRevision)
                    Volatile.Write(ref ownedCapture.WriterRevision, writerRevision);
            }
            observation = captured;
            keepWatch = true;
            return CpuMappedCopyResult.Copied;
        }
        finally { if (!keepWatch) ownedCapture?.ReleaseWatch(); backing.ExitAccess(); }

        CpuMappedCopyResult Refuse(ref CpuMappedCopyDiagnostics.Failure target, string reason,
            CpuMappedCopyResult result, ulong cursor = 0, int count = 0, View? view = null, long enrollmentAdmission = -1)
        {
            if (CpuMappedCopyDiagnostics.Enabled)
                target = new("ledger", reason, result, footprintAddress, footprintByteCount, sourceAddress,
                    copyBytes, withWriteWatch, backing.Identity, cursor, count,
                    view?.Address ?? 0, view?.Size ?? 0, view?.Backing.Identity ?? 0, enrollmentAdmission,
                    backing.ManagedWrites?.FirstCoverageLoss ?? 0, backing.ManagedWrites?.HistoryRecordLosses ?? 0);
            return result;
        }
    }

    // Cold enrollment under Gate. Managed aliases have a participating physical
    // writer veto; native aliases additionally need protection at every reachable VA.
    // Refuse an alias outside the observed interval rather than assuming that one
    // VA's page guard protects every mapping of the same physical bytes.
    private bool NativeAliasesCovered(HostBackingObject backing, View[] selected, ulong source, ulong size)
    {
        foreach (var original in selected)
        {
            var start = Math.Max(source, original.Address);
            var end = Math.Min(source + size, original.Address + original.Size);
            if (start >= end) continue;
            var physicalStart = original.Offset + start - original.Address;
            var physicalEnd = physicalStart + end - start;
            foreach (var candidate in views.Values)
            {
                if (!ReferenceEquals(candidate.Backing, backing)) continue;
                var overlapStart = Math.Max(physicalStart, candidate.Offset);
                var overlapEnd = Math.Min(physicalEnd, candidate.Offset + candidate.Size);
                if (overlapStart >= overlapEnd) continue;
                var virtualStart = candidate.Address + overlapStart - candidate.Offset;
                if (virtualStart < source || virtualStart - source > size ||
                    overlapEnd - overlapStart > size - (virtualStart - source)) return false;
            }
        }
        return true;
    }

    // Bounded scalar validation, without taking Gate or consulting native state.
    public bool IsCurrent(HostBackingObject backing, CpuMappedCopyObservation observation)
    {
        if (observation is null || observation.OwnerIdentity is not Capture captured ||
            !ReferenceEquals(captured.Owner, this) || Volatile.Read(ref unknownAlias) || backing.IsClosing || backing.IsDisposed) return false;
        for (var index = 0; index < captured.States.Length; index++)
        {
            var view = captured.States[index];
            if (!ReferenceEquals(view.Backing, backing) || Volatile.Read(ref view.Revoked) != 0 ||
                Volatile.Read(ref view.CopyAvailable) == 0 || Volatile.Read(ref view.Incarnation) != captured.Incarnations[index]) return false;
        }
        return true;
    }

    private bool TryEffectRange(ulong address, ulong size, out HostAddressRange effect)
    {
        effect = default;
        if (!HostViewMemory.IsValidRange(address, size)) return false;
        var start = address & ~(pageSize - 1);
        var end = address + size;
        if (end > ulong.MaxValue - (pageSize - 1)) return false;
        end = (end + pageSize - 1) & ~(pageSize - 1);
        effect = new(start, end - start); return true;
    }
    private static bool Overlaps(HostAddressRange a, HostAddressRange b) =>
        a.Address < b.Address + b.Size && b.Address < a.Address + a.Size;
    private void Revoke(HostAddressRange effect)
    {
        for (var index = FindAtOrBelow(effect.Address + effect.Size - 1); index >= 0; index--)
        {
            var view = views.Values[index];
            if (view.Address + view.Size <= effect.Address) break;
            RevokeAt(index);
        }
    }
    private void RevokeAt(int index)
    { Volatile.Write(ref views.Values[index].Revoked, 1); views.RemoveAt(index); NoteMappingChangedLocked(); }
    // Single writer under Gate; ordinary writes read it without the gate.
    private void NoteMappingChangedLocked() => Volatile.Write(ref mappingEpoch, mappingEpoch + 1);
    private void RevokeAll() { for (var index = views.Count - 1; index >= 0; index--) RevokeAt(index); }
    private int FindAtOrBelow(ulong address)
    {
        var low = 0; var high = views.Count - 1; var found = -1;
        while (low <= high)
        { var middle = low + (high - low) / 2; if (views.Keys[middle] <= address) { found = middle; low = middle + 1; } else high = middle - 1; }
        return found;
    }
    private static GuestPageProtection AccessOf(HostPageProtection protection) => protection switch
    {
        HostPageProtection.ReadOnly => GuestPageProtection.Read,
        HostPageProtection.ReadWrite => GuestPageProtection.Read | GuestPageProtection.Write,
        HostPageProtection.Execute => GuestPageProtection.Execute,
        HostPageProtection.ReadExecute => GuestPageProtection.Read | GuestPageProtection.Execute,
        HostPageProtection.ReadWriteExecute or HostPageProtection.ExecuteWriteCopy => GuestPageProtection.Read | GuestPageProtection.Write | GuestPageProtection.Execute,
        _ => GuestPageProtection.None,
    };
}
