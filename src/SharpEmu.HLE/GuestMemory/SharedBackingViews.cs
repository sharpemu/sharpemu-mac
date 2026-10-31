// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using SharpEmu.HLE.DsTrace;
using SharpEmu.HLE.Host;

namespace SharpEmu.HLE.GuestMemory;

public readonly record struct ViewRecord(ulong Address, ulong Size, ulong Offset, HostPageProtection Protection)
{
    public bool WasRestored { get; init; }
}

public sealed unsafe partial class SharedBackingViews : IDisposable, ICpuMappedCopySource, ICpuMappedReadWatchSource, ICpuMappedPublicationSource
{
    internal static Action<string> OnFatal = message => Environment.FailFast(message);

    private readonly IHostViewMemory _host;
    private readonly HostBackingObject? _backing;
    private readonly object _lock = new();
    private readonly SortedList<ulong, ViewRecord> _views = new();
    // Small copies stay under the metadata lock to avoid reservation overhead.
    private const ulong ConcurrentCopyMinimumBytes = 64 * 1024;
    private CopyReservation[]? _copyReservations;
    private int _activeCopies;
    private int _copyWaiters;
    private int _mappingWaiters;
    // Set while any unlocked copy is reserved: lock-free single-view accesses and alias
    // accesses are refused, and the locked paths wait only for overlapping reservations.
    private int _copyAccessBlocked;
    private bool _disposed;
    private readonly object _disposeGate = new(); // never needed by access or mapping paths
    private bool _disposeRequested;
    private int _operationThread;
    private int _operationDepth;
    private Exception? _cleanupFailure;
    private bool _cleanupComplete;

    // Guest command writes and their reads by the render thread reach TryWriteBacking and
    // TryReadBacking millions of times per second, a few bytes at a time, while views
    // change rarely. Single-view accesses search this immutable copy of _views without
    // the lock; every change to _views republishes it under the lock.
    private ViewSnapshot _snapshot = ViewSnapshot.Empty;

    // Chunked backing only: view records reaching into each chunk, so a release never empties
    // a chunk some guest address still maps. Changed under _lock (SetViewLocked/RemoveViewAtLocked).
    private readonly int[]? _chunkViews;

    // Chunked backing only: which chunks a texture holds instead of the alias (GuestSingleCopy).
    private readonly GuestSingleCopy? _singleCopy;

    // Shareable backing only (SHARPEMU_GPU_PROCESS=1): every view record and a log of their changes,
    // in shared memory, for a process that replicates the views. Changed under _lock (SetViewLocked/RemoveViewAtLocked).
    private readonly GuestViewTable? _viewTable;

    // Single-view accesses in flight; Dispose waits for them before releasing the alias.
    // A copy reservation also waits for these: they touch the alias without checking it.
    private int _activeAccesses;
    // Lifetime pins that never touch bytes outside the locked, reservation-checked paths
    // (those paths, ordinary-write receipts, unlocked copies). Dispose waits for them too;
    // a copy reservation does not, so a pin held across a wait cannot deadlock a copy.
    private int _lifetimeAccesses;
    // Default-null controlled ordinary-write boundary. The real operation,
    // chunk admission and alias lifetime pin are already held. Tests must use
    // the single-view path to pause without the shared-view gate held.
    private Action<ulong, int>? _beforeCanonicalAliasCopyForControl = null;

    public SharedBackingViews(IHostViewMemory host, ulong size)
    {
        _host = host;
        _ = host.TryCreateBacking(size, out _backing, out _);
        if (_backing is { ChunkBytes: > 0, DiscardChunks: not null } chunked)
        {
            _chunkViews = new int[(chunked.Size + chunked.ChunkBytes - 1) / chunked.ChunkBytes];
            _singleCopy = new GuestSingleCopy(this, chunked.ChunkBytes, chunked.Size);
            GuestBackingTelemetry.NoteChunkedBacking(chunked.ChunkBytes, chunked.Size);
        }
        // Outside the provider gate and before this wrapper is published: all
        // of its participating writes have history from the first operation.
        try { if (host is IHostMappedPublicationMemory) _backing?.EnrollManagedWrites(); }
        catch (Exception primary)
        {
            // This newly created backing was not published to a caller yet.
            // Enrollment allocation failure must not abandon its native owner.
            try { _backing?.Dispose(); }
            catch (Exception cleanup)
            {
                try { primary.Data["ManagedWriterEnrollmentCleanup"] = cleanup; } catch { }
            }
            throw;
        }

        // Last: nothing above can fail after it, so the table needs no cleanup here. Protection
        // changes are only logged: tracking them in the entries costs every write fault a split.
        if (_backing?.Export is not null && host is IHostGuestMirrorMemory mirrorHost)
            _viewTable = GuestViewTable.TryCreate(mirrorHost, _backing.Size, trackProtection: false);
    }

    public bool IsAvailable => _backing != null && !Volatile.Read(ref _disposeRequested) && !Volatile.Read(ref _disposed) && !_backing.IsClosing && !_backing.IsDisposed;

    public bool CleanupComplete => Volatile.Read(ref _cleanupComplete);

    public ulong AliasBase => _backing?.AliasBase ?? 0;

    /// <summary>What another process needs to map this backing; null unless the host made it shareable.</summary>
    public GuestBackingExport? BackingExport => _backing?.Export;

    /// <summary>The shared table of this backing's views; null unless the backing is shareable.</summary>
    public GuestViewTable? ViewTable => _viewTable;

    public ulong Size => _backing?.Size ?? 0;

    public CpuMappedPublicationPreparationResult TryPrepareMappedReadValidation(CpuMappedCopyObservation original,
        out CpuMappedReadValidation? validation)
    {
        validation = null;
        if (_host is not IHostMappedReadWatchMemory provider || _backing is null)
            return CpuMappedPublicationPreparationResult.Unavailable;
        if (!TryEnterLifetimeAccess()) return CpuMappedPublicationPreparationResult.Rejected;
        try
        {
            var result = provider.TryPrepareMappedReadValidation(_backing, original, out validation);
            if (IsAvailable) return result;
            validation?.Dispose(); validation = null;
            return CpuMappedPublicationPreparationResult.Rejected;
        }
        finally { ExitLifetimeAccess(); }
    }

    public bool CanWaitForMappedPublicationValidation => CanEmitOrdinaryWriteDiagnostic;

    public CpuMappedPublicationPreparationResult TryPrepareMappedPublicationValidation(CpuMappedCopyObservation original,
        out CpuMappedPublicationValidation? validation)
    {
        validation = null;
        if (_host is not IHostMappedPublicationMemory provider || _backing is null)
            return CpuMappedPublicationPreparationResult.Unavailable;
        if (!TryEnterLifetimeAccess()) return CpuMappedPublicationPreparationResult.Rejected;
        try
        {
            var result = provider.TryPrepareMappedPublicationValidation(_backing, original, out validation);
            if (IsAvailable) return result;
            validation?.Dispose(); validation = null;
            return CpuMappedPublicationPreparationResult.Rejected;
        }
        finally { ExitLifetimeAccess(); }
    }

    public CpuMappedPublicationPreparationResult TryPrepareMappedPublicationValidation(CpuMappedCopyObservation original,
        out CpuMappedPublicationValidation? validation,
        out CpuMappedPublicationValidationRefusal refusal)
    {
        validation = null; refusal = default;
        if (_host is not IHostMappedPublicationMemory provider || _backing is null)
            return CpuMappedPublicationPreparationResult.Unavailable;
        if (!TryEnterLifetimeAccess()) return CpuMappedPublicationPreparationResult.Rejected;
        try
        {
            var result = provider.TryPrepareMappedPublicationValidation(_backing, original, out validation, out refusal);
            if (IsAvailable) return result;
            validation?.Dispose(); validation = null; refusal = default;
            return CpuMappedPublicationPreparationResult.Rejected;
        }
        finally { ExitLifetimeAccess(); }
    }

    public CpuMappedPublicationPreparationResult TryPrepareMappedPublication(CpuMappedCopyObservation original,
        IReadOnlyList<CpuMappedWriteRun> runs, out CpuMappedPublication? publication)
        => TryPrepareMappedPublication(original, runs, out publication, out _);

    public CpuMappedPublicationPreparationResult TryPrepareMappedPublication(CpuMappedCopyObservation original,
        IReadOnlyList<CpuMappedWriteRun> runs, out CpuMappedPublication? publication,
        out CpuMappedPublicationPreparationRefusal refusal)
    {
        publication = null; refusal = default;
        if (_host is not IHostMappedPublicationMemory provider || _backing is null)
            return CpuMappedPublicationPreparationResult.Unavailable;
        if (!TryEnterLifetimeAccess()) return CpuMappedPublicationPreparationResult.Rejected;
        try
        {
            var result = provider.TryPrepareMappedPublication(_backing, original, runs, out publication, out refusal);
            if (IsAvailable) return result;
            publication?.Dispose(); publication = null; refusal = default;
            return CpuMappedPublicationPreparationResult.Rejected;
        }
        finally { ExitLifetimeAccess(); }
    }

    public CpuMappedCopyResult TryReadMappedCopyWithWriteWatch(ulong footprintAddress, ulong footprintByteCount,
        ulong sourceAddress, Span<byte> destination, out CpuMappedCopyObservation? observation)
    {
        observation = null;
        if (_host is not IHostMappedReadWatchMemory provider || _backing is null)
            return CpuMappedCopyResult.Unavailable;
        if (!TryEnterLifetimeAccess()) return CpuMappedCopyResult.Rejected;
        try
        {
            var result = provider.TryReadMappedCopyWithWriteWatch(_backing, footprintAddress, footprintByteCount,
                sourceAddress, destination, out observation);
            if (IsAvailable) return result;
            if (observation is not null) provider.ReleaseMappedCopyWriteWatch(observation);
            observation = null; return CpuMappedCopyResult.Rejected;
        }
        finally { ExitLifetimeAccess(); }
    }
    public bool CanWaitForMappedReadWatchStatus => IsAvailable && CanEmitOrdinaryWriteDiagnostic &&
        _host is IHostMappedReadWatchMemory provider && provider.CanWaitForMappedReadWatchStatus;

    public bool CanWaitForMappedReadWatchEnrollment => IsAvailable && CanEmitOrdinaryWriteDiagnostic &&
        _host is IHostMappedReadWatchMemory provider && provider.CanWaitForMappedReadWatchEnrollment;

    public CpuMappedCopyResult TryReadMappedCopyWithWriteWatchAdmissionWait(ulong footprintAddress,
        ulong footprintByteCount, ulong sourceAddress, Span<byte> destination,
        Func<bool> originalStillCurrent, out CpuMappedCopyObservation? observation)
    {
        ArgumentNullException.ThrowIfNull(originalStillCurrent);
        observation = null;
        var backing = _backing;
        if (_host is not IHostMappedReadWatchMemory provider || backing is null) return CpuMappedCopyResult.Unavailable;
        bool WrapperCurrent() => IsAvailable && ReferenceEquals(_backing, backing) &&
            CanWaitForMappedReadWatchEnrollment;
        bool OriginalWrapper() => WrapperCurrent() && originalStillCurrent() && WrapperCurrent();
        if (!OriginalWrapper() || !TryEnterLifetimeAccess()) return CpuMappedCopyResult.Rejected;
        try
        {
            var result = provider.TryReadMappedCopyWithWriteWatchAdmissionWait(backing, footprintAddress,
                footprintByteCount, sourceAddress, destination, OriginalWrapper, out observation);
            if (IsAvailable && ReferenceEquals(_backing, backing)) return result;
            if (observation is not null) provider.ReleaseMappedCopyWriteWatch(observation);
            observation = null; return CpuMappedCopyResult.Rejected;
        }
        finally { ExitLifetimeAccess(); }
    }

    public bool IsMappedCopyWriteWatchCurrent(CpuMappedCopyObservation observation) =>
        IsAvailable && _backing is not null && _host is IHostMappedReadWatchMemory provider &&
        provider.IsMappedCopyWriteWatchCurrent(_backing, observation);
    public bool IsMappedCopyWriteWatchCurrent(CpuMappedCopyObservation observation,
        out CpuMappedReadWatchDiagnostic diagnostic)
    {
        if (!IsAvailableForReadWatchDiagnostic(true, out var reason))
        { diagnostic = new() { SharedClosure = reason, BooleanCurrent = false }; return false; }
        if (_backing is not { } backing)
        { diagnostic = new() { SharedClosure = "shared-backing-missing", BooleanCurrent = false }; return false; }
        if (_host is not IHostMappedReadWatchMemory provider)
        { diagnostic = new() { SharedClosure = "shared-pre-capability-unavailable", BooleanCurrent = false }; return false; }
        return provider.IsMappedCopyWriteWatchCurrent(backing, observation, out diagnostic);
    }
    public CpuMappedReadWatchStatus GetMappedCopyWriteWatchStatus(CpuMappedCopyObservation observation)
    {
        if (!IsAvailable || _backing is not { } backing || _host is not IHostMappedReadWatchMemory provider)
            return CpuMappedReadWatchStatus.Stale;
        var status = provider.GetMappedCopyWriteWatchStatus(backing, observation);
        return IsAvailable && ReferenceEquals(_backing, backing) ? status : CpuMappedReadWatchStatus.Stale;
    }
    public void ReleaseMappedCopyWriteWatch(CpuMappedCopyObservation observation)
    { if (_host is IHostMappedReadWatchMemory provider) provider.ReleaseMappedCopyWriteWatch(observation); }

    private bool IsAvailableForReadWatchDiagnostic(bool before, out string? reason)
    {
        reason = _backing is null ? "shared-backing-missing" :
            Volatile.Read(ref _disposeRequested) ? before ? "shared-pre-dispose-requested" : "shared-post-dispose-requested" :
            Volatile.Read(ref _disposed) ? before ? "shared-pre-disposed" : "shared-post-disposed" :
            _backing.IsClosing ? before ? "shared-pre-backing-closing" : "shared-post-backing-closing" :
            _backing.IsDisposed ? before ? "shared-pre-backing-disposed" : "shared-post-backing-disposed" : null;
        return reason is null;
    }
    public CpuMappedReadWatchStatus GetMappedCopyWriteWatchStatus(CpuMappedCopyObservation observation,
        out CpuMappedReadWatchDiagnostic diagnostic)
    {
        if (!IsAvailableForReadWatchDiagnostic(true, out var reason))
        { diagnostic = new() { SharedClosure = reason }; return CpuMappedReadWatchStatus.Stale; }
        if (_backing is not { } backing)
        { diagnostic = new() { SharedClosure = "shared-backing-missing" }; return CpuMappedReadWatchStatus.Stale; }
        if (_host is not IHostMappedReadWatchMemory provider)
        { diagnostic = new() { SharedClosure = "shared-pre-capability-unavailable" }; return CpuMappedReadWatchStatus.Stale; }
        var status = provider.GetMappedCopyWriteWatchStatus(backing, observation, out diagnostic);
        if (!IsAvailableForReadWatchDiagnostic(false, out reason))
        { diagnostic = diagnostic with { SharedClosure = reason }; return CpuMappedReadWatchStatus.Stale; }
        if (!ReferenceEquals(_backing, backing))
        { diagnostic = diagnostic with { SharedClosure = "shared-post-backing-replaced" }; return CpuMappedReadWatchStatus.Stale; }
        return status;
    }

    public CpuMappedCopyResult TryReadMappedCopy(ulong footprintAddress, ulong footprintByteCount,
        ulong sourceAddress, Span<byte> destination, out CpuMappedCopyObservation? observation)
    {
        observation = null;
        if (_host is not IHostMappedCopyMemory provider || _backing is null)
            return CpuMappedCopyResult.Unavailable;
        if (!TryEnterLifetimeAccess()) return CpuMappedCopyResult.Rejected;
        try
        {
            var result = provider.TryReadMappedCopy(_backing, footprintAddress, footprintByteCount,
                sourceAddress, destination, out observation);
            if (!IsAvailable) { observation = null; return CpuMappedCopyResult.Rejected; }
            return result;
        }
        finally { ExitLifetimeAccess(); }
    }

    public bool IsMappedCopyObservationCurrent(CpuMappedCopyObservation observation) =>
        IsAvailable && _backing is not null && _host is IHostMappedCopyMemory provider &&
        provider.IsMappedCopyObservationCurrent(_backing, observation);

    public bool Clear(ulong offset, ulong size)
    {
        if (size == 0) return ClearCore(offset, size);
        // One admission spans the complete multi-chunk operation, including
        // staged/memmove reads. Publication cannot enter between its chunks.
        var interval = _backing?.BeginManagedWriteOperation() ?? default;
        if (!HostManagedWriteIntervals.AllowsBytes(interval)) return false;
        var succeeded = false;
        try { succeeded = ClearCore(offset, size); return succeeded; }
        finally { HostBackingObject.EndManagedWrite(interval, succeeded); }
    }

    private bool ClearCore(ulong offset, ulong size)
    {
        var dsLock = DsLock.EnterMonitor(_lock, 7);
        try
        {
            WaitForCopies();
            if (!TryEnterLifetimeAccess()) return false;
            try
            {
                if (!IsAvailable || !IsWithinBacking(offset, size))
                {
                    return false;
                }

                if (_chunkViews is not null) ClearChunked(offset, size);
                else ClearCanonicalAlias(offset, size);
                return true;
            }
            finally { ExitLifetimeAccess(); }
        }
        finally { DsLock.ExitMonitor(_lock, 7, dsLock); }
    }

    public bool TryWriteBacking(ulong address, ReadOnlySpan<byte> data)
        => TryWriteBackingAdmission(address, data, false, out _, out _);

    public bool TryWriteBacking(ulong address, ReadOnlySpan<byte> data, out OrdinaryBackingWriteReservation? busy)
        => TryWriteBackingAdmission(address, data, true, out busy, out _);

    public bool TryWriteBacking(ulong address, ReadOnlySpan<byte> data, out OrdinaryBackingWriteReservation? busy,
        out SharedWriteFailure failure) => TryWriteBackingAdmission(address, data, true, out busy, out failure);

    // Deferred captures that may find the mapping moved before the write falls back to
    // capturing first. Diagnostics always capture first, so their capture reason is exact.
    private const int DeferredOriginalCaptures = 4;

    private bool TryWriteBackingAdmission(ulong address, ReadOnlySpan<byte> data, bool captureBusy,
        out OrdinaryBackingWriteReservation? busy, out SharedWriteFailure failure)
        => TryWriteBackingAdmission(address, data, captureBusy,
            OrdinaryWriteFailureDiagnostics.Enabled ? 0 : DeferredOriginalCaptures, out busy, out failure);

    private bool TryWriteBackingAdmission(ulong address, ReadOnlySpan<byte> data, bool captureBusy, int deferredCaptures,
        out OrdinaryBackingWriteReservation? busy, out SharedWriteFailure failure)
    {
        busy = null; failure = default;
        if (data.IsEmpty) return TryWriteBackingCore(address, data);
        // One admission spans the complete multi-chunk operation, including
        // staged/memmove reads. Publication cannot enter between its chunks.
        var snapshot = Volatile.Read(ref _snapshot);
        // Once enrolled, retain ORIGINAL provider mapping/guest access before
        // first admission. A later lookup may never repair a busy refusal.
        string? originalCaptureReason = null;
        HostMappedCopyLedger.OrdinaryWriteIdentity? originalMapping = null;
        IHostMappedPublicationMemory? deferredProvider = null;
        long mappingEpoch = 0;
        if (captureBusy && _backing?.ManagedWrites is not null &&
            _host is IHostMappedPublicationMemory originalProvider && originalProvider.CanWaitForOrdinaryWriteAdmission)
        {
            // Every guest write comes here and almost none is refused, so a provider that
            // versions its mapping takes the capture only on refusal, from the same
            // records this one would read, or reports that they moved.
            if (deferredCaptures > 0 && originalProvider.OrdinaryWriteMappingEpoch is { } epoch)
            { deferredProvider = originalProvider; mappingEpoch = epoch; }
            else originalMapping = OrdinaryWriteFailureDiagnostics.Enabled
                ? originalProvider.CaptureOrdinaryWriteIdentity(_backing, address, (ulong)data.Length, out originalCaptureReason)
                : originalProvider.CaptureOrdinaryWriteIdentity(_backing, address, (ulong)data.Length);
        }
        var interval = _backing?.BeginManagedWriteOperation() ?? default;
        if (!HostManagedWriteIntervals.AllowsBytes(interval))
        {
            if (deferredProvider is not null)
            {
                originalMapping = deferredProvider.CaptureOrdinaryWriteIdentity(
                    _backing!, address, (ulong)data.Length, mappingEpoch, out var mappingChanged);
                // A refused admission holds nothing, so starting over is only a later first attempt.
                if (mappingChanged)
                    return TryWriteBackingAdmission(address, data, captureBusy, deferredCaptures - 1, out busy, out failure);
            }
            // Exact rejected token is the provenance, never a later admission sample.
            // Save the FIRST existing short-circuit guard; never borrow a
            // later admission or original-currentness sample for diagnostics.
            var originalIdentity = originalMapping.GetValueOrDefault();
            var refusal = !captureBusy ? "busy-capture-disabled"
                : !originalMapping.HasValue ? "original-identity-unavailable"
                : !originalIdentity.IsCurrent ? "original-identity-changed"
                : !interval.IsPublicationBusy ? "admission-not-publication-busy"
                : _host is not IHostMappedPublicationMemory provider ? "provider-unavailable"
                : !provider.CanWaitForOrdinaryWriteAdmission ? "provider-wait-gate-or-unsupported"
                : !TryEnterLifetimeAccess() ? "alias-pin-refused" : null;
            var pinAcquired = refusal is null;
            var snapshotMatched = false;
            if (pinAcquired)
            {
                // A lifetime pin: the receipt waits while holding it, and its retried write
                // takes its own access (or the locked, reservation-checked path).
                try
                {
                    if (!IsAvailable) refusal = "views-unavailable";
                    else if (!(snapshotMatched = ReferenceEquals(snapshot, AliasSnapshot))) refusal = "original-snapshot-changed";
                    else busy = new(this, _backing!, snapshot, originalIdentity, interval.RejectedAdmission);
                }
                finally { if (busy is null) ExitLifetimeAccess(); }
            }
            if (OrdinaryWriteFailureDiagnostics.Enabled)
                failure = new(refusal ?? "publication-busy-receipt-prepared", interval.RejectedAdmission,
                    originalMapping.HasValue, pinAcquired, snapshotMatched, originalCaptureReason);
            return false; // No destination bytes, not a retryable partial operation.
        }
        var succeeded = false;
        try
        {
            succeeded = TryWriteBackingCore(address, data);
            if (!succeeded && OrdinaryWriteFailureDiagnostics.Enabled) failure = new("backing-access-failed-after-admission", 0, originalMapping.HasValue, false, false, originalCaptureReason);
            return succeeded;
        }
        finally { HostBackingObject.EndManagedWrite(interval, succeeded); }
    }

    internal bool CanEmitOrdinaryWriteDiagnostic =>
        !Monitor.IsEntered(_lock) && !Monitor.IsEntered(_disposeGate) &&
        (_backing is null || !Monitor.IsEntered(_backing.Gate)) &&
        (_host is IHostMappedPublicationMemory provider && provider.CanWaitForOrdinaryWriteAdmission);

    internal void RequireOrdinaryWriteWaitOutsideGates()
    {
        if (_host is not IHostMappedPublicationMemory provider || !provider.CanWaitForOrdinaryWriteAdmission ||
            Monitor.IsEntered(_lock) || Monitor.IsEntered(_disposeGate) || (_backing is not null && Monitor.IsEntered(_backing.Gate)))
            throw new InvalidOperationException("Ordinary write wait cannot hold view/backing/disposal gates.");
    }

    private bool TryWriteBackingCore(ulong address, ReadOnlySpan<byte> data)
    {
        if (TryAccessSingleView(address, (ulong)data.Length, out var target))
        {
            try
            {
                CopyToCanonicalAlias(target, data);
                return true;
            }
            finally
            {
                ExitAliasAccess();
            }
        }

        var dsLock = DsLock.EnterMonitor(_lock, 6);
        try
        {
            // A copy wait releases the lock, so resolve the mapping again after it. Waits
            // happen before the lifetime pin; nothing waits while holding one here.
            while (true)
            {
                if (IsAvailable && TryFindRecord(address, (ulong)data.Length, out var record))
                {
                    var offset = record.Offset + address - record.Address;
                    if (!IsWithinBacking(offset, (ulong)data.Length))
                    {
                        return false;
                    }

                    if (_activeCopies != 0 && WaitForBufferCopyConflict(offset, data)) continue;
                    if (!TryEnterLifetimeAccess()) return false;
                    try { CopyToCanonicalAlias(AliasBase + offset, data); }
                    finally { ExitLifetimeAccess(); }
                    return true;
                }

                if (WaitForAnyCopy()) continue;
                if (!TryCollectBackingSegments(address, (ulong)data.Length, out var pieces))
                {
                    return false;
                }

                if (!TryEnterLifetimeAccess()) return false;
                try
                {
                    foreach (var (backing, dataOffset, bytes) in pieces)
                    {
                        CopyToCanonicalAlias(backing, data.Slice(dataOffset, bytes));
                    }
                }
                finally { ExitLifetimeAccess(); }

                return true;
            }
        }
        finally { DsLock.ExitMonitor(_lock, 6, dsLock); }
    }

    // Each destination chunk keeps its original copy/overlap semantics and
    // existing alias pin. These helpers add ordinary bookkeeping only: an
    // incomplete ledger never prevents the legacy write or authorizes content.
    private void ClearCanonicalAlias(ulong offset, ulong count)
    {
        _singleCopy?.BeforeWrite(offset, count);
        var interval = _backing!.BeginManagedWrite(offset, count); var succeeded = false;
        if (!HostManagedWriteIntervals.AllowsBytes(interval)) throw new InvalidOperationException("Participating write lost admission.");
        try { NativeMemory.Clear((void*)(AliasBase + offset), (nuint)count); succeeded = true; }
        finally { HostBackingObject.EndManagedWrite(interval, succeeded); }
    }

    // A chunked backing clears whole chunks by emptying them: the same zeros through every
    // mapping, without faulting in and dirtying pages nothing may ever read. Only the partial
    // chunks at either end are written.
    private void ClearChunked(ulong offset, ulong count)
    {
        var chunk = _backing!.ChunkBytes;
        var start = AlignUpTo(offset, chunk);
        var end = offset + count == Size ? Size : (offset + count) / chunk * chunk;
        if (end <= start)
        {
            ClearCanonicalAlias(offset, count);
            return;
        }

        if (start > offset) ClearCanonicalAlias(offset, start - offset);
        GuestBackingTelemetry.NoteClear(DiscardCanonicalAlias(start, end - start));
        if (offset + count > end) ClearCanonicalAlias(end, offset + count - end);
    }

    private HostBackingDiscard DiscardCanonicalAlias(ulong offset, ulong count)
    {
        _singleCopy?.BeforeDiscard(offset, count);
        var interval = _backing!.BeginManagedWrite(offset, count); var succeeded = false;
        if (!HostManagedWriteIntervals.AllowsBytes(interval)) throw new InvalidOperationException("Participating write lost admission.");
        try
        {
            var result = _backing.DiscardChunks!(_backing, offset, count);
            succeeded = true;
            return result;
        }
        finally { HostBackingObject.EndManagedWrite(interval, succeeded); }
    }

    private void CopyToCanonicalAlias(ulong destination, ReadOnlySpan<byte> source)
    {
        _singleCopy?.BeforeWrite(destination - AliasBase, (ulong)source.Length);
        var interval = _backing!.BeginManagedWrite(destination - AliasBase, (ulong)source.Length);
        var succeeded = false;
        if (!HostManagedWriteIntervals.AllowsBytes(interval)) throw new InvalidOperationException("Participating write lost admission.");
        try
        {
            _beforeCanonicalAliasCopyForControl?.Invoke(destination, source.Length);
            source.CopyTo(new Span<byte>((void*)destination, source.Length));
            succeeded = true;
        }
        finally { HostBackingObject.EndManagedWrite(interval, succeeded); }
    }

    private void CopyCanonicalAliasChunk(ulong source, ulong destination, int bytes)
    {
        _singleCopy?.BeforeRead(source - AliasBase, (ulong)bytes);
        _singleCopy?.BeforeWrite(destination - AliasBase, (ulong)bytes);
        var interval = _backing!.BeginManagedWrite(destination - AliasBase, (ulong)bytes);
        var succeeded = false;
        if (!HostManagedWriteIntervals.AllowsBytes(interval)) throw new InvalidOperationException("Participating write lost admission.");
        try { Buffer.MemoryCopy((void*)source, (void*)destination, bytes, bytes); succeeded = true; }
        finally { HostBackingObject.EndManagedWrite(interval, succeeded); }
    }

    // The lock-free single-view read only; false (with nothing read) for anything else.
    public bool TryReadSingleView(ulong address, Span<byte> data)
    {
        if (!TryAccessSingleView(address, (ulong)data.Length, out var source))
        {
            return false;
        }

        try
        {
            _singleCopy?.BeforeRead(source - AliasBase, (ulong)data.Length);
            new ReadOnlySpan<byte>((void*)source, data.Length).CopyTo(data);
            return true;
        }
        finally
        {
            ExitAliasAccess();
        }
    }

    // SHARPEMU_SINGLE_VIEW_WRITE_FASTPATH=0 restores the old bail on managed-write-enrolled
    // backings (every macOS backing), sending every such write through the PVM-locked
    // TryWriteBacking path. Read once; for bisecting a regression without a rebuild.
    private static readonly bool SingleViewWriteFastPath =
        Environment.GetEnvironmentVariable("SHARPEMU_SINGLE_VIEW_WRITE_FASTPATH") != "0";

    // The lock-free single-view write; false, with no bytes written, no admission held and
    // no receipt minted, for anything else. The caller (PVM.TryWriteCore) then takes its
    // locked TryWriteBacking path as a new first attempt: a refused admission holds nothing,
    // so that attempt is only a later first attempt, and it alone captures the original
    // region/mapping identity and mints the publication-busy receipt it may wait on.
    //
    // Invariants, unchanged from the locked path:
    // - The view is resolved (and the alias pin taken) BEFORE ordinary admission. A lookup
    //   miss must never End an admitted operation token: End(token, false) invalidates
    //   every read watch on the whole backing (HostManagedWriteIntervals.End).
    // - No byte lands without an admitted ordinary operation token, so a held publication
    //   reservation excludes this write exactly as it excludes TryWriteBacking.
    // - CopyToCanonicalAlias brackets the bytes with a per-chunk interval that invalidates
    //   overlapping read watches before the bytes land.
    // - The alias pin is held until after the bytes and the operation's completion.
    // - Nothing here waits, so callers that already hold the PVM gate stay safe.
    // Relaxed, as for TryReadSingleView: the write is not ordered against a concurrent PVM
    // unmap/remap of the same range, because the PVM gate is not taken.
    public bool TryWriteSingleView(ulong address, ReadOnlySpan<byte> data)
    {
        if (data.IsEmpty || (!SingleViewWriteFastPath && _backing?.ManagedWrites is not null))
        {
            return false;
        }

        if (!TryAccessSingleView(address, (ulong)data.Length, out var target))
        {
            return false;
        }

        try
        {
            var interval = _backing?.BeginManagedWriteOperation() ?? default;
            if (!HostManagedWriteIntervals.AllowsBytes(interval))
            {
                return false;
            }

            var succeeded = false;
            try
            {
                CopyToCanonicalAlias(target, data);
                succeeded = true;
                return true;
            }
            finally
            {
                HostBackingObject.EndManagedWrite(interval, succeeded);
            }
        }
        finally
        {
            ExitAliasAccess();
        }
    }

    // A guest-to-guest copy inside single views, as TryWriteSingleView writes: one alias pin and one
    // ordinary operation admission, the bytes bracketed by a per-chunk interval (and the single-copy
    // bookkeeping for both ranges). False, with nothing copied, for anything that path would refuse.
    public bool TryCopySingleView(ulong destination, ulong source, ulong size)
    {
        if (size == 0 || size > int.MaxValue || (!SingleViewWriteFastPath && _backing?.ManagedWrites is not null))
        {
            return false;
        }

        if (ulong.MaxValue - destination < size || ulong.MaxValue - source < size || !TryEnterAliasAccess())
        {
            return false;
        }

        try
        {
            var snapshot = Volatile.Read(ref _snapshot);
            if (!TryFindAlias(snapshot, source, size, out var from) || !TryFindAlias(snapshot, destination, size, out var to))
            {
                return false;
            }

            var interval = _backing?.BeginManagedWriteOperation() ?? default;
            if (!HostManagedWriteIntervals.AllowsBytes(interval))
            {
                return false;
            }

            var succeeded = false;
            try
            {
                CopyCanonicalAliasChunk(from, to, (int)size);
                succeeded = true;
                return true;
            }
            finally
            {
                HostBackingObject.EndManagedWrite(interval, succeeded);
            }
        }
        finally
        {
            ExitAliasAccess();
        }
    }

    // Two TryWriteSingleView writes (first, then second) under one alias pin and one operation
    // admission; each still brackets its own bytes with its own interval, so overlapping read watches
    // are invalidated per range before that range's bytes land, as for separate writes. False, with
    // nothing written, when either range has no single view or the admission is refused. As in
    // TryWriteSingleView, a lost interval admission after the first range landed throws.
    public bool TryWritePairSingleView(ulong firstAddress, ReadOnlySpan<byte> first, ulong secondAddress, ReadOnlySpan<byte> second)
    {
        if (first.IsEmpty || second.IsEmpty || (!SingleViewWriteFastPath && _backing?.ManagedWrites is not null))
        {
            return false;
        }

        if (ulong.MaxValue - firstAddress < (ulong)first.Length || ulong.MaxValue - secondAddress < (ulong)second.Length ||
            !TryEnterAliasAccess())
        {
            return false;
        }

        try
        {
            var snapshot = Volatile.Read(ref _snapshot);
            if (!TryFindAlias(snapshot, firstAddress, (ulong)first.Length, out var firstTarget) ||
                !TryFindAlias(snapshot, secondAddress, (ulong)second.Length, out var secondTarget))
            {
                return false;
            }

            var interval = _backing?.BeginManagedWriteOperation() ?? default;
            if (!HostManagedWriteIntervals.AllowsBytes(interval))
            {
                return false;
            }

            var succeeded = false;
            try
            {
                CopyToCanonicalAlias(firstTarget, first);
                CopyToCanonicalAlias(secondTarget, second);
                succeeded = true;
                return true;
            }
            finally
            {
                HostBackingObject.EndManagedWrite(interval, succeeded);
            }
        }
        finally
        {
            ExitAliasAccess();
        }
    }

    // TryWriteSingleView with the bytes produced in place by fill instead of copied from a
    // source span: same alias pin, ordinary operation admission and per-chunk interval
    // (overlapping read watches are invalidated before fill runs), and false with fill
    // never called for anything that path would refuse. If fill throws, the bytes it wrote
    // stay and both the interval and the operation end as failed, which conservatively
    // invalidates the backing's read watches.
    public bool TryFillSingleView<TState>(ulong address, int length, TState state, GuestSpanFill<TState> fill, out int filled)
    {
        filled = 0;
        if (length <= 0 || (!SingleViewWriteFastPath && _backing?.ManagedWrites is not null))
        {
            return false;
        }

        if (!TryAccessSingleView(address, (ulong)length, out var target))
        {
            return false;
        }

        try
        {
            var interval = _backing?.BeginManagedWriteOperation() ?? default;
            if (!HostManagedWriteIntervals.AllowsBytes(interval))
            {
                return false;
            }

            var succeeded = false;
            try
            {
                filled = FillCanonicalAlias(target, length, state, fill);
                succeeded = true;
                return true;
            }
            finally
            {
                HostBackingObject.EndManagedWrite(interval, succeeded);
            }
        }
        finally
        {
            ExitAliasAccess();
        }
    }

    private int FillCanonicalAlias<TState>(ulong destination, int length, TState state, GuestSpanFill<TState> fill)
    {
        // A fill may stop short (a file read at its end): released chunks it covers whole get
        // back only the bytes it did not reach, once it says how far it got.
        var overwrite = _singleCopy?.BeforeFill(destination - AliasBase, (ulong)length);
        var produced = 0;
        try
        {
            var interval = _backing!.BeginManagedWrite(destination - AliasBase, (ulong)length);
            var succeeded = false;
            if (!HostManagedWriteIntervals.AllowsBytes(interval)) throw new InvalidOperationException("Participating write lost admission.");
            try
            {
                _beforeCanonicalAliasCopyForControl?.Invoke(destination, length);
                produced = fill(state, new Span<byte>((void*)destination, length));
                if ((uint)produced > (uint)length)
                    throw new InvalidOperationException($"In-place fill reported {produced} bytes for a {length}-byte range.");
                succeeded = true;
                return produced;
            }
            finally { HostBackingObject.EndManagedWrite(interval, succeeded); }
        }
        finally { _singleCopy?.AfterFill(overwrite, destination - AliasBase, (ulong)Math.Clamp(produced, 0, length)); }
    }

    public bool TryReadBacking(ulong address, Span<byte> data)
    {
        if (TryAccessSingleView(address, (ulong)data.Length, out var source))
        {
            try
            {
                _singleCopy?.BeforeRead(source - AliasBase, (ulong)data.Length);
                new ReadOnlySpan<byte>((void*)source, data.Length).CopyTo(data);
                return true;
            }
            finally
            {
                ExitAliasAccess();
            }
        }

        lock (_lock)
            while (true)
            {
                // A read inside one mapping needs no temporary segment list.
                if (IsAvailable && TryFindRecord(address, (ulong)data.Length, out var record))
                {
                    var offset = record.Offset + address - record.Address;
                    if (!IsWithinBacking(offset, (ulong)data.Length))
                    {
                        return false;
                    }

                    if (_activeCopies != 0 && WaitForBufferCopyConflict(offset, data)) continue;
                    if (!TryEnterLifetimeAccess()) return false;
                    try
                    {
                        _singleCopy?.BeforeRead(offset, (ulong)data.Length);
                        new ReadOnlySpan<byte>((void*)(AliasBase + offset), data.Length).CopyTo(data);
                    }
                    finally { ExitLifetimeAccess(); }
                    return true;
                }

                if (WaitForAnyCopy()) continue;
                if (!TryCollectBackingSegments(address, (ulong)data.Length, out var pieces))
                {
                    return false;
                }

                if (!TryEnterLifetimeAccess()) return false;
                try
                {
                    foreach (var (backing, dataOffset, bytes) in pieces)
                    {
                        _singleCopy?.BeforeRead(backing - AliasBase, (ulong)bytes);
                        new ReadOnlySpan<byte>((void*)backing, bytes).CopyTo(data.Slice(dataOffset, bytes));
                    }
                }
                finally { ExitLifetimeAccess(); }

                return true;
            }
    }

    // Use temporary storage if a copy segment can overwrite another segment's source.
    public bool TryCopyBacking(ulong destination, ulong source, ulong size)
    {
        if (size == 0) return TryCopyBackingCore(destination, source, size);
        // One admission spans the complete multi-chunk operation, including
        // staged/memmove reads. Publication cannot enter between its chunks.
        var interval = _backing?.BeginManagedWriteOperation() ?? default;
        if (!HostManagedWriteIntervals.AllowsBytes(interval)) return false;
        var succeeded = false;
        try { succeeded = TryCopyBackingCore(destination, source, size); return succeeded; }
        finally { HostBackingObject.EndManagedWrite(interval, succeeded); }
    }

    private bool TryCopyBackingCore(ulong destination, ulong source, ulong size)
    {
        // A large copy inside one source and one destination view runs outside the lock;
        // its reservation makes overlapping accesses and every mapping change wait for it.
        if (size >= ConcurrentCopyMinimumBytes && size <= int.MaxValue &&
            TryReserveCopy(destination, source, size, out var reservation))
        {
            using (reservation)
            {
                // Use the backing alias so CopyTo can detect overlap between different guest views.
                CopyToCanonicalAlias(reservation.Destination,
                    new ReadOnlySpan<byte>((void*)reservation.Source, (int)size));
                return true;
            }
        }

        lock (_lock)
            while (true)
            {
                if (IsAvailable && size <= int.MaxValue &&
                    TryFindRecord(source, size, out var sourceRecord) &&
                    TryFindRecord(destination, size, out var destinationRecord))
                {
                    var sourceOffset = sourceRecord.Offset + source - sourceRecord.Address;
                    var destinationOffset = destinationRecord.Offset + destination - destinationRecord.Address;
                    if (!IsWithinBacking(sourceOffset, size) || !IsWithinBacking(destinationOffset, size))
                    {
                        return false;
                    }

                    if (_activeCopies != 0 && WaitForCopyConflict(sourceOffset, size, destinationOffset, size)) continue;
                    if (!TryEnterLifetimeAccess()) return false;
                    // Use the backing alias so CopyTo can detect overlap between different guest views.
                    try
                    {
                        _singleCopy?.BeforeRead(sourceOffset, size);
                        CopyToCanonicalAlias(AliasBase + destinationOffset,
                            new ReadOnlySpan<byte>((void*)(AliasBase + sourceOffset), (int)size));
                    }
                    finally { ExitLifetimeAccess(); }
                    return true;
                }

                if (WaitForAnyCopy()) continue;
                if (!TryCollectBackingSegments(source, size, out var from) || !TryCollectBackingSegments(destination, size, out var to))
                {
                    return false;
                }

                if (!TryEnterLifetimeAccess()) return false;
                try
                {
                    var chunks = CreateCopySegments(from, to);
                    if (!HasCrossSegmentOverlap(chunks))
                    {
                        foreach (var (fromPtr, toPtr, bytes) in chunks)
                        {
                            CopyCanonicalAliasChunk(fromPtr, toPtr, bytes);
                        }

                        return true;
                    }

                    var staging = new byte[size];
                    foreach (var (backing, dataOffset, bytes) in from)
                    {
                        _singleCopy?.BeforeRead(backing - AliasBase, (ulong)bytes);
                        new ReadOnlySpan<byte>((void*)backing, bytes).CopyTo(staging.AsSpan(dataOffset, bytes));
                    }

                    foreach (var (backing, dataOffset, bytes) in to)
                    {
                        CopyToCanonicalAlias(backing, staging.AsSpan(dataOffset, bytes));
                    }

                    return true;
                }
                finally { ExitLifetimeAccess(); }
            }
    }

    public bool TryMapReservedRange(ulong address, ulong size, ulong offset, HostPageProtection protection, out HostViewFailure failure)
        => TryMapReservedRange(address, size, offset, protection, false, out failure);

    private int _overlapReports;

    /// <summary>How many mappings were recorded over a range another one still holds.</summary>
    public int OverlappingViewRecords => Volatile.Read(ref _overlapReports);

    // Must be called holding _lock. Reports, once per distinct pair shape, a new view
    // whose guest range intersects a record that has not been unmapped.
    private void WarnOnOverlappingView(ulong address, ulong size, ulong offset)
    {
        if (size == 0)
        {
            return;
        }

        var end = address + size;
        var index = RangeSearch.FindLastIndexAtOrBelow(_views, address);
        // The record at or below can reach into this range, and every record that starts
        // inside it overlaps by construction.
        for (var candidate = Math.Max(index, 0); candidate < _views.Count; candidate++)
        {
            var record = _views.Values[candidate];
            if (record.Address >= end)
            {
                break;
            }

            if (record.Address + record.Size <= address)
            {
                continue;
            }

            if (++_overlapReports <= 32)
            {
                Console.Error.WriteLine(
                    $"[LOADER][WARN] mapping overlaps a live one: new=0x{address:X16}+0x{size:X}@0x{offset:X} " +
                    $"existing=0x{record.Address:X16}+0x{record.Size:X}@0x{record.Offset:X}");
                Console.Error.Flush();
            }

            return;
        }

        WarnOnAliasedBacking(address, size, offset);
    }

    private int _aliasReports;
    private static readonly bool TraceAliasedBacking =
        Environment.GetEnvironmentVariable("SHARPEMU_LOG_DIRECT_MEMORY") == "1";

    /// <summary>How many mappings were recorded over backing bytes another one still holds.</summary>
    public int AliasedBackingRecords => Volatile.Read(ref _aliasReports);

    // Must be called holding _lock. Two mappings at different guest addresses over the
    // same backing bytes are one piece of memory the title sees twice. That is legal — a
    // title may map direct memory at two addresses on purpose — but it is also what an
    // allocator handing the same physical offset out twice looks like, and then a write
    // through one address rewrites what the other holds.
    private void WarnOnAliasedBacking(ulong address, ulong size, ulong offset)
    {
        var end = offset + size;
        foreach (var record in _views.Values)
        {
            if (record.Address == address || record.Offset >= end || record.Offset + record.Size <= offset)
            {
                continue;
            }

            // Aliasing is legal and coherent through the shared backing, so it is only traced.
            if (++_aliasReports <= 32 && TraceAliasedBacking)
            {
                Console.Error.WriteLine(
                    $"[LOADER][TRACE] mapping aliases live backing bytes: new=0x{address:X16}+0x{size:X}@0x{offset:X} " +
                    $"existing=0x{record.Address:X16}+0x{record.Size:X}@0x{record.Offset:X}");
                Console.Error.Flush();
            }

            return;
        }
    }

    private bool TryMapReservedRange(ulong address, ulong size, ulong offset, HostPageProtection protection,
        bool wasRestored, out HostViewFailure failure)
    {
        failure = HostViewFailure.BackingUnavailable;
        if (!TryBeginViewOperation(wasRestored)) return false;
        Exception? primary = null;
        var singleCopyMap = false;
        try
        {
            if (_backing is null || _backing.IsClosing || _backing.IsDisposed || !IsWithinBacking(offset, size))
            { failure = HostViewFailure.OffsetOutOfBounds; return false; }
            lock (_lock)
            {
                // Mapping changes wait for unlocked copies (see TryReserveCopy).
                WaitForCopies();
                if (_views.ContainsKey(address)) OnFatal($"A backing view record already exists at 0x{address:X16}.");
                // An exact duplicate was already fatal; a *partial* overlap was not, and it
                // is the worse of the two. The guest is handed two allocations that share
                // bytes, and whichever writes second rewrites the other's data with no
                // fault at the time and no way, later, to say who did it.
                WarnOnOverlappingView(address, size, offset);
                if (_singleCopy is { Handler: not null })
                {
                    // No chunk is released between here and the view record (TryMarkSingleCopyReleased).
                    _singleCopy.MapsInFlight++;
                    singleCopyMap = true;
                }
            }

            // A new address over a released chunk would read it straight from the backing.
            _singleCopy?.BeforeMap(offset, size);
            bool mapped;
            try { mapped = _host.TryMapView(_backing, address, offset, size, protection, out failure); }
            catch (Exception error)
            {
                // Provider threw before returning an outcome: retain the attempted
                // exact scope as unknown; do not guess whether MAP_FIXED occurred.
                lock (_lock)
                {
                    SetViewLocked(new ViewRecord(address, size, offset, protection));
                    _cleanupFailure = error;
                    _disposeRequested = true;
                    PublishSnapshot();
                }
                throw;
            }
            if (!mapped) return false;
            lock (_lock)
            {
                // Nested remaps belong to an already admitted unmap/rollback
                // transaction. Finish them before requested disposal captures.
                if (!_disposeRequested || wasRestored)
                {
                    SetViewLocked(new ViewRecord(address, size, offset, protection) { WasRestored = wasRestored });
                    PublishSnapshot(); failure = HostViewFailure.None; return true;
                }
            }
            try
            {
                if (!_host.UnmapView(address, size))
                    throw new InvalidOperationException("Late backing view cleanup outcome unknown.");
            }
            catch (Exception error)
            {
                lock (_lock)
                {
                    SetViewLocked(new ViewRecord(address, size, offset, protection));
                    _cleanupFailure = error;
                }
                throw;
            }
            failure = HostViewFailure.BackingUnavailable; return false;
        }
        catch (Exception error) { primary = error; throw; }
        finally
        {
            if (singleCopyMap)
            {
                lock (_lock) _singleCopy!.MapsInFlight--;
            }

            EndViewOperation(primary);
        }
    }

    public bool Unmap(ulong address, ulong size, out bool holePreserved)
    {
        holePreserved = false;
        if (!TryBeginViewOperation(continuation: false)) return false;
        Exception? primary = null;
        try
        {
            holePreserved = false;
            if (_backing is null || _backing.IsClosing || size == 0 || ulong.MaxValue - address < size)
            {
                return false;
            }

            var end = address + size;
            var targets = new List<ViewRecord>();
            var lockTaken = false;
            try
            {
                using (GuestMemoryProfile.Measure(GuestMemoryProfile.Operation.SharedViewUnmapLockAcquisition))
                    Monitor.Enter(_lock, ref lockTaken);
                // Mapping changes wait for unlocked copies (see TryReserveCopy).
                using (GuestMemoryProfile.Measure(GuestMemoryProfile.Operation.SharedViewUnmapCopyWait))
                    WaitForCopies();
                using var lookupProfile = GuestMemoryProfile.Measure(GuestMemoryProfile.Operation.SharedViewUnmapTargetLookup);
                var current = address;
                while (current < end)
                {
                    if (!TryFindRecord(current, 1, out var record))
                    {
                        return false;
                    }

                    var partSize = Math.Min(end, record.Address + record.Size) - current;
                    targets.Add(new ViewRecord(current, partSize, record.Offset + current - record.Address, record.Protection));
                    current += partSize;
                }
            }
            finally
            {
                if (lockTaken)
                    Monitor.Exit(_lock);
            }

            var removed = new List<ViewRecord>();
            foreach (var target in targets)
            {
                if (!TryUnmapSingleView(target.Address, target.Size, out var partPreserved))
                {
                    RestoreMappingsInReverseOrder(removed);
                    return false;
                }

                if (!partPreserved)
                {
                    OnFatal($"The address range was not kept reserved after unmapping the view at 0x{target.Address:X16}.");
                }

                removed.Add(target);
            }

            // Each removed view leaves a reserved range. Join these ranges into one free range.
            if (removed.Count > 1 && !_host.JoinHoles(address, size))
            {
                RestoreMappingsInReverseOrder(removed);
                return false;
            }

            holePreserved = true;
            return true;

        }
        catch (Exception error) { primary = error; throw; }
        finally { EndViewOperation(primary); }
    }

    private bool TryUnmapSingleView(ulong address, ulong size, out bool holePreserved)
    {
        holePreserved = false;
        var old = default(ViewRecord);
        lock (_lock)
        {
            // A copy reserved after Unmap collected its targets still finishes before the
            // record it resolved goes away.
            WaitForCopies();
            var index = RangeSearch.FindLastIndexAtOrBelow(_views, address);
            if (index >= 0)
            {
                var record = _views.Values[index];
                if (address + size <= record.Address + record.Size)
                {
                    old = record;
                    using (GuestMemoryProfile.Measure(GuestMemoryProfile.Operation.SharedViewUnmapRecordRemoval))
                        RemoveViewAtLocked(index);
                    PublishSnapshot();
                }
            }
        }

        if (old.Size == 0)
        {
            return false;
        }

        bool unmapped;
        GuestMemoryProfile.RecordUnmapSizes(size, old.Size, address - old.Address,
            old.Address + old.Size - address - size);
        try
        {
            using (GuestMemoryProfile.Measure(GuestMemoryProfile.Operation.SharedViewUnmapNativeCall))
                unmapped = _host.UnmapView(old.Address, old.Size);
        }
        catch (Exception error)
        {
            lock (_lock)
            {
                SetViewLocked(old); // attempted scope retained, not a physical mapping assertion
                _cleanupFailure = error; _disposeRequested = true; PublishSnapshot();
            }
            throw;
        }
        if (!unmapped) { RestoreViewRecord(old); return false; }

        if (address == old.Address && size == old.Size)
        {
            holePreserved = true;
            return true;
        }

        using var restorationProfile = GuestMemoryProfile.Measure(GuestMemoryProfile.Operation.SharedViewUnmapPartialRestoration);
        var leftSize = address - old.Address;
        var rightAddress = address + size;
        var rightSize = old.Address + old.Size - rightAddress;
        var ok = true;
        if (leftSize != 0 && leftSize != old.Size)
        {
            using var splitProfile = GuestMemoryProfile.Measure(GuestMemoryProfile.Operation.SharedViewRestorationSplit);
            ok = _host.SplitHole(old.Address, leftSize) && ok;
        }

        if (rightSize != 0)
        {
            using var splitProfile = GuestMemoryProfile.Measure(GuestMemoryProfile.Operation.SharedViewRestorationSplit);
            ok = _host.SplitHole(address, size) && ok;
        }

        if (leftSize != 0)
        {
            ok = TryMapReservedRange(old.Address, leftSize, old.Offset, old.Protection, true, out _) && ok;
        }

        if (rightSize != 0)
        {
            ok = TryMapReservedRange(rightAddress, rightSize, old.Offset + (rightAddress - old.Address), old.Protection, true, out _) && ok;
        }

        if (ok)
        {
            holePreserved = true;
            return true;
        }

        if (leftSize != 0 && Contains(old.Address, leftSize) && !TryUnmapSingleView(old.Address, leftSize, out _))
        {
            OnFatal($"Could not remove the partial backing view at 0x{old.Address:X16} during recovery.");
        }

        if (rightSize != 0 && Contains(rightAddress, rightSize) && !TryUnmapSingleView(rightAddress, rightSize, out _))
        {
            OnFatal($"Could not remove the partial backing view at 0x{rightAddress:X16} during recovery.");
        }

        if (!_host.JoinHoles(old.Address, old.Size))
        {
            OnFatal($"Could not join the reserved ranges at 0x{old.Address:X16} during recovery.");
        }

        RestoreViewMapping(old);
        return false;
    }

    internal bool IsRestoredView(ulong address)
    {
        lock (_lock)
        {
            return IsAvailable && TryFindRecord(address, 1, out var record) && record.WasRestored;
        }
    }

    // Contains over the published snapshot, without the lock: descriptor validation asks
    // this for every bound buffer, and the lock is contended by guest threads. A true answer
    // is as current as a locked one would be once the lock were released.
    public bool ContainsWithoutLock(ulong address, ulong size)
    {
        if (size == 0 || ulong.MaxValue - address < size || _disposed)
        {
            return false;
        }

        var snapshot = Volatile.Read(ref _snapshot);
        var end = address + size;
        var current = address;
        while (current < end)
        {
            if (!snapshot.TryFind(current, out var chunk, out var index))
            {
                return false;
            }

            ref readonly var found = ref snapshot.At(chunk, index);
            if (current >= found.Address + found.Size)
            {
                return false;
            }

            current = Math.Min(end, found.Address + found.Size);
        }

        return true;
    }

    public bool Contains(ulong address, ulong size)
    {
        if (size == 0 || ulong.MaxValue - address < size)
        {
            return false;
        }

        lock (_lock)
        {
            var end = address + size;
            var current = address;
            while (current < end)
            {
                if (!TryFindRecord(current, 1, out var record))
                {
                    return false;
                }

                current = Math.Min(end, record.Address + record.Size);
            }

            return true;
        }
    }

    private bool TryBeginViewOperation(bool continuation)
    {
        lock (_lock)
        {
            var thread = Environment.CurrentManagedThreadId;
            if (_operationDepth != 0 && _operationThread == thread)
            {
                // Thread identity alone cannot admit a fresh callback operation
                // after close. Only private existing-transaction remaps continue.
                if (_disposed || (_disposeRequested && !continuation)) return false;
                _operationDepth++; return true;
            }
            var start = System.Diagnostics.Stopwatch.GetTimestamp();
            while (_operationDepth != 0 && !_disposeRequested)
            {
                var remaining = TimeSpan.FromSeconds(2) - System.Diagnostics.Stopwatch.GetElapsedTime(start);
                if (remaining <= TimeSpan.Zero || !Monitor.Wait(_lock, remaining)) return false;
            }
            if (_disposeRequested || _disposed) return false;
            _operationThread = thread; _operationDepth = 1; return true;
        }
    }

    private void EndViewOperation(Exception? primary)
    {
        bool close;
        lock (_lock)
        {
            if (_operationThread != Environment.CurrentManagedThreadId || _operationDepth == 0)
                throw new InvalidOperationException("Unbalanced view operation.");
            close = --_operationDepth == 0 && _disposeRequested;
            if (_operationDepth == 0) { _operationThread = 0; Monitor.PulseAll(_lock); }
        }
        if (!close) return;
        try { Dispose(); }
        catch (Exception error) when (primary is not null && error is not OutOfMemoryException and not AccessViolationException)
        { primary.Data["BackingDeferredCleanupFailure"] = error; }
    }

    /// <summary>
    /// Ordinary disposal. Provider reentrancy requests close without waiting for
    /// its own pending operation; CleanupComplete remains false. The outer map/
    /// unmap discharges cleanup before returning, or retains quarantined ownership.
    /// </summary>
    public void Dispose()
    {
        if (Monitor.IsEntered(_lock) || Monitor.IsEntered(_disposeGate))
            throw new InvalidOperationException("Backing disposal cannot drain under the view or cleanup gate.");
        lock (_lock)
        {
            // Unlocked copies finish before disposal is requested; none can start after.
            WaitForCopies();
            _disposeRequested = true; PublishSnapshot();
            if (_operationDepth != 0 && _operationThread == Environment.CurrentManagedThreadId) return;
            var start = System.Diagnostics.Stopwatch.GetTimestamp();
            while (_operationDepth != 0)
            {
                var remaining = TimeSpan.FromSeconds(2) - System.Diagnostics.Stopwatch.GetElapsedTime(start);
                // Monitor.Wait releases the only held gate; pending operation
                // completion never needs the cleanup gate and cannot wait on us.
                if (remaining <= TimeSpan.Zero || !Monitor.Wait(_lock, remaining))
                    throw new TimeoutException("Backing view operation did not finish; storage remains owned.");
            }
        }
        if (!Monitor.TryEnter(_disposeGate, TimeSpan.FromSeconds(2)))
            throw new TimeoutException("Backing cleanup attempt remains in flight.");
        try
        {
            List<ViewRecord> views;
            lock (_lock)
            {
                if (_cleanupComplete) return;
                Volatile.Write(ref _disposed, true); PublishSnapshot();
                if (_cleanupFailure is not null)
                    throw new InvalidOperationException("Backing view cleanup is quarantined.", _cleanupFailure);
                views = new List<ViewRecord>(_views.Values);
            }
            var start = System.Diagnostics.Stopwatch.GetTimestamp();
            while (Volatile.Read(ref _activeAccesses) != 0 || Volatile.Read(ref _lifetimeAccesses) != 0)
            {
                if (System.Diagnostics.Stopwatch.GetElapsedTime(start) >= TimeSpan.FromSeconds(2))
                    throw new TimeoutException("Backing alias accesses did not drain; storage remains owned.");
                Thread.Sleep(1);
            }
            foreach (var view in views)
            {
                try
                {
                    if (!_host.UnmapView(view.Address, view.Size))
                        throw new InvalidOperationException($"Backing view cleanup outcome unknown at 0x{view.Address:X16}.");
                }
                catch (Exception error) { lock (_lock) _cleanupFailure = error; throw; }
                lock (_lock)
                {
                    var removedIndex = _views.IndexOfKey(view.Address);
                    if (removedIndex >= 0) RemoveViewAtLocked(removedIndex);
                }
            }
            _backing?.Dispose();
            _viewTable?.Dispose();
            lock (_lock) Volatile.Write(ref _cleanupComplete, true);
        }
        finally { Monitor.Exit(_disposeGate); }
    }

    private bool TryCollectBackingSegments(ulong address, ulong size, out List<(ulong Backing, int DataOffset, int Bytes)> pieces)
    {
        pieces = new List<(ulong, int, int)>();
        if (!IsAvailable || size == 0 || ulong.MaxValue - address < size)
        {
            return false;
        }

        var end = address + size;
        var current = address;
        while (current < end)
        {
            if (!TryFindRecord(current, 1, out var record))
            {
                return false;
            }

            var bytes = Math.Min(end, record.Address + record.Size) - current;
            var offset = record.Offset + current - record.Address;
            if (!IsWithinBacking(offset, bytes))
            {
                return false;
            }

            pieces.Add((AliasBase + offset, (int)(current - address), (int)bytes));
            current += bytes;
        }

        return true;
    }

    private static List<(ulong From, ulong To, int Bytes)> CreateCopySegments(
        List<(ulong Backing, int DataOffset, int Bytes)> from,
        List<(ulong Backing, int DataOffset, int Bytes)> to)
    {
        var chunks = new List<(ulong, ulong, int)>();
        var i = 0;
        var j = 0;
        var offset = 0;
        while (i < from.Count && j < to.Count)
        {
            var fromEnd = from[i].DataOffset + from[i].Bytes;
            var toEnd = to[j].DataOffset + to[j].Bytes;
            var end = Math.Min(fromEnd, toEnd);
            chunks.Add((
                from[i].Backing + (ulong)(offset - from[i].DataOffset),
                to[j].Backing + (ulong)(offset - to[j].DataOffset),
                end - offset));
            offset = end;
            i += fromEnd == end ? 1 : 0;
            j += toEnd == end ? 1 : 0;
        }

        return chunks;
    }

    private static bool HasCrossSegmentOverlap(List<(ulong From, ulong To, int Bytes)> chunks)
    {
        for (var a = 0; a < chunks.Count; a++)
        {
            for (var b = 0; b < chunks.Count; b++)
            {
                if (a != b && chunks[a].From < chunks[b].To + (ulong)chunks[b].Bytes &&
                    chunks[b].To < chunks[a].From + (ulong)chunks[a].Bytes)
                {
                    return true;
                }
            }
        }

        return false;
    }

    private bool TryFindRecord(ulong address, ulong size, out ViewRecord record)
    {
        record = default;
        if (size == 0 || ulong.MaxValue - address < size)
        {
            return false;
        }

        var index = RangeSearch.FindLastIndexAtOrBelow(_views, address);
        if (index < 0)
        {
            return false;
        }

        var candidate = _views.Values[index];
        if (address + size > candidate.Address + candidate.Size)
        {
            return false;
        }

        record = candidate;
        return true;
    }

    // Must be called under _lock after every change to _views.
    private void PublishSnapshot()
    {
        if (_disposed || _disposeRequested)
        {
            Volatile.Write(ref _snapshot, ViewSnapshot.Empty);
            (_changedLow, _changedHigh) = (ulong.MaxValue, 0);
            return;
        }

        var current = _snapshot;
        var snapshot = _changedLow > _changedHigh
            ? current.Republished()
            : ReferenceEquals(current, ViewSnapshot.Empty)
                ? ViewSnapshot.Build(_views)
                : current.Rebuilt(_views, _changedLow, _changedHigh);
        (_changedLow, _changedHigh) = (ulong.MaxValue, 0);
        Volatile.Write(ref _snapshot, snapshot);
    }

    // Resolves an access that lies inside one view to its alias address and registers
    // it as in flight; the caller must decrement _activeAccesses when done. Anything
    // else (spanning views, unmapped, disposed) is left to the locked path.
    private bool TryAccessSingleView(ulong address, ulong size, out ulong target)
    {
        target = 0;
        if (size == 0 || ulong.MaxValue - address < size)
        {
            return false;
        }

        if (!TryEnterAliasAccess()) return false;
        if (TryFindAlias(Volatile.Read(ref _snapshot), address, size, out target)) return true;
        ExitAliasAccess();
        return false;
    }

    public object AliasSnapshot => Volatile.Read(ref _snapshot);

    // An alias access touches backing bytes without checking copy reservations, so it is
    // refused while an unlocked copy is reserved, and a reservation waits for admitted ones.
    public bool TryEnterAliasAccess()
    {
        if (Volatile.Read(ref _copyAccessBlocked) != 0) return false;
        Interlocked.Increment(ref _activeAccesses);
        // Register before the gate check so a reservation waits for admitted access.
        if (Volatile.Read(ref _copyAccessBlocked) == 0 && TryEnterBackingAccess()) return true;
        Interlocked.Decrement(ref _activeAccesses);
        return false;
    }

    public void ExitAliasAccess()
    {
        _backing!.ExitAccess();
        Interlocked.Decrement(ref _activeAccesses);
    }

    // Keeps the alias alive without admitting a lock-free byte access: the locked paths
    // (which wait for overlapping reservations first and never wait while pinned), receipts
    // and unlocked copies. Not refused by copy reservations, and not waited for by them.
    private bool TryEnterLifetimeAccess()
    {
        Interlocked.Increment(ref _lifetimeAccesses);
        if (TryEnterBackingAccess()) return true;
        Interlocked.Decrement(ref _lifetimeAccesses);
        return false;
    }

    private void ExitLifetimeAccess()
    {
        _backing!.ExitAccess();
        Interlocked.Decrement(ref _lifetimeAccesses);
    }

    // Releases the lifetime pin an ordinary-write receipt took (see TryWriteBackingAdmission).
    internal void ExitOrdinaryWriteReceiptAccess() => ExitLifetimeAccess();

    private bool TryEnterBackingAccess()
    {
        if (!Volatile.Read(ref _disposed) && !Volatile.Read(ref _disposeRequested) && _backing is not null && _backing.TryEnterAccess())
        {
            if (!Volatile.Read(ref _disposed) && !Volatile.Read(ref _disposeRequested)) return true;
            _backing.ExitAccess();
        }
        return false;
    }

    public bool TryResolveAlias(ulong address, ulong size, out ulong target)
    {
        target = 0;
        return size != 0 && ulong.MaxValue - address >= size && TryFindAlias(Volatile.Read(ref _snapshot), address, size, out target);
    }

    /// <summary>
    /// The alias of a range that may span several views, as long as those views are
    /// adjacent in guest space and in the backing. A guest allocation mapped in
    /// several calls is one contiguous run of backing bytes, so a buffer that crosses
    /// a view edge still has one host address — the single-record test cannot see
    /// that, and a caller that maps the backing no-copy would otherwise have to copy.
    /// </summary>
    public bool TryResolveAliasSpan(ulong address, ulong size, out ulong target)
    {
        target = 0;
        if (size == 0 || ulong.MaxValue - address < size)
        {
            return false;
        }

        var snapshot = Volatile.Read(ref _snapshot);
        if (TryFindAlias(snapshot, address, size, out target))
        {
            return true;
        }

        if (!snapshot.TryFind(address, out var chunk, out var index) || Volatile.Read(ref _disposed) || _backing is null)
        {
            target = 0;
            return false;
        }

        var record = snapshot.At(chunk, index);
        if (address >= record.Address + record.Size)
        {
            target = 0;
            return false;
        }

        var offset = record.Offset + address - record.Address;
        var covered = record.Address + record.Size - address;
        while (covered < size)
        {
            if (!snapshot.TryNext(ref chunk, ref index))
            {
                target = 0;
                return false;
            }

            var next = snapshot.At(chunk, index);
            // Adjacent in guest space and continuing at the next backing byte.
            if (next.Address != record.Address + record.Size || next.Offset != record.Offset + record.Size)
            {
                target = 0;
                return false;
            }

            record = next;
            covered += record.Size;
        }

        if (!IsWithinBacking(offset, size))
        {
            target = 0;
            return false;
        }

        target = AliasBase + offset;
        return true;
    }

    private bool TryFindAlias(ViewSnapshot snapshot, ulong address, ulong size, out ulong target)
    {
        target = 0;
        if (snapshot.TryFind(address, out var chunk, out var index) && !_disposed && _backing != null)
        {
            var record = snapshot.At(chunk, index);
            if (address + size <= record.Address + record.Size)
            {
                var offset = record.Offset + address - record.Address;
                if (IsWithinBacking(offset, size))
                {
                    target = AliasBase + offset;
                    return true;
                }
            }
        }

        return false;
    }

    private bool IsWithinBacking(ulong offset, ulong size) =>
        size != 0 && offset < Size && size <= Size - offset;

    // Every change to _views goes through these two, so a chunked backing always knows how
    // many view records reach into each chunk. Must be called under _lock.
    private void SetViewLocked(ViewRecord record)
    {
        var hadRecord = _views.TryGetValue(record.Address, out var replaced);
        if (hadRecord)
        {
            CountChunkViews(replaced, -1);
            NoteViewChangeLocked(replaced);
        }

        _views[record.Address] = record;
        CountChunkViews(record, 1);
        NoteViewChangeLocked(record);
        if (_viewTable is not { } table) return;
        if (hadRecord) table.Remap(replaced.Address, replaced.Size, record.Address, record.Size, record.Offset, record.Protection);
        else table.Map(record.Address, record.Size, record.Offset, record.Protection);
    }

    private void RemoveViewAtLocked(int index)
    {
        var removed = _views.Values[index];
        CountChunkViews(removed, -1);
        NoteViewChangeLocked(removed);
        _views.RemoveAt(index);
        _viewTable?.Unmap(removed.Address, removed.Size);
    }

    private void CountChunkViews(ViewRecord record, int delta)
    {
        if (_chunkViews is null || record.Size == 0 || record.Offset >= Size) return;
        var chunk = _backing!.ChunkBytes;
        var last = Math.Min((record.Offset + record.Size - 1) / chunk, (ulong)_chunkViews.Length - 1);
        for (var index = record.Offset / chunk; index <= last; index++) _chunkViews[index] += delta;
    }

    /// <summary>Bytes per separately releasable chunk of the backing; 0 when it is one object.</summary>
    public ulong ChunkBytes => _chunkViews is null ? 0 : _backing!.ChunkBytes;

    /// <summary>Bytes per purgeable object behind the chunks (one chunk when the backend says nothing); 0 for one object.</summary>
    public ulong ObjectBytes => _chunkViews is null ? 0 : Math.Max(_backing!.ObjectBytes, _backing.ChunkBytes);

    /// <summary>The released-chunk state of a chunked backing (single-copy textures); null for one object.</summary>
    public GuestSingleCopy? SingleCopy => _singleCopy;

    internal int ChunkViewCount(int chunk) =>
        _chunkViews is { } views && (uint)chunk < (uint)views.Length ? Volatile.Read(ref views[chunk]) : 0;

    // Under _lock, so a mapping cannot add a second address for the chunk between the check
    // and the mark: exactly one view maps the chunk, whole, at this address. A chunk an
    // unlocked copy has reserved is not marked: that copy runs no single-copy hook.
    internal bool TryMarkSingleCopyReleased(GuestSingleCopy state, int chunk, ulong address)
    {
        if (_chunkViews is not { } views || (uint)chunk >= (uint)views.Length)
        {
            return false;
        }

        var chunkBytes = _backing!.ChunkBytes;
        var offset = (ulong)chunk * chunkBytes;
        var length = Math.Min(chunkBytes, Size - offset);
        lock (_lock)
        {
            return IsAvailable && state.MapsInFlight == 0 && views[chunk] == 1 && !OverlapsCopyReservation(offset, length) &&
                   TryFindRecord(address, length, out var record) && record.Offset + (address - record.Address) == offset &&
                   state.MarkReleasedLocked(chunk);
        }
    }

    // The single-copy manager's own accesses to the canonical alias: no hook, a lifetime pin,
    // and the managed-write bookkeeping every alias write takes. The pin is not refused by an
    // unlocked copy's reservation, and need not be: no reserved range holds a released chunk
    // (TryReserveCopy, TryMarkSingleCopyReleased), and these touch released chunks only.
    internal void WriteSingleCopyBytes(ulong offset, ReadOnlySpan<byte> bytes)
    {
        if (bytes.IsEmpty)
        {
            return;
        }

        var write = BeginSingleCopyWrite(offset, (ulong)bytes.Length);
        var succeeded = false;
        try
        {
            bytes.CopyTo(new Span<byte>((void*)(AliasBase + offset), bytes.Length));
            succeeded = true;
        }
        finally { EndSingleCopyWrite(write, succeeded); }
    }

    internal void FillSingleCopyBytes(ulong offset, ulong size, byte value)
    {
        var write = BeginSingleCopyWrite(offset, size);
        var succeeded = false;
        try
        {
            new Span<byte>((void*)(AliasBase + offset), checked((int)size)).Fill(value);
            succeeded = true;
        }
        finally { EndSingleCopyWrite(write, succeeded); }
    }

    internal HostBackingDiscard EmptySingleCopyChunks(ulong offset, ulong size)
    {
        var write = BeginSingleCopyWrite(offset, size);
        var succeeded = false;
        try
        {
            var result = _backing!.DiscardChunks!(_backing, offset, size);
            succeeded = true;
            return result;
        }
        finally { EndSingleCopyWrite(write, succeeded); }
    }

    private readonly record struct SingleCopyWrite(HostManagedWriteIntervals.Token Operation, HostManagedWriteIntervals.Token Interval);

    // The alias pin, then the operation and the interval every alias write takes. A
    // publication holds the backing exclusively for a moment; the bytes must land, so a
    // refused admission waits for it rather than failing.
    private SingleCopyWrite BeginSingleCopyWrite(ulong offset, ulong size)
    {
        if (_backing is null || !IsWithinBacking(offset, size) || size > int.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(offset), $"0x{offset:X}+0x{size:X} is not inside the backing.");
        }

        if (!TryEnterLifetimeAccess())
        {
            throw new InvalidOperationException("The guest backing is closing; a released chunk cannot be written.");
        }

        var operation = _backing.BeginManagedWriteOperation();
        var started = System.Diagnostics.Stopwatch.GetTimestamp();
        while (!HostManagedWriteIntervals.AllowsBytes(operation))
        {
            if (System.Diagnostics.Stopwatch.GetElapsedTime(started) > TimeSpan.FromSeconds(10))
            {
                OnFatal($"A released guest chunk at 0x{offset:X} could not be written: the backing stayed busy.");
            }

            Thread.Sleep(1);
            operation = _backing.BeginManagedWriteOperation();
        }

        var interval = _backing.BeginManagedWrite(offset, size);
        if (!HostManagedWriteIntervals.AllowsBytes(interval))
        {
            HostBackingObject.EndManagedWrite(operation, false);
            ExitLifetimeAccess();
            throw new InvalidOperationException("Participating write lost admission.");
        }

        return new SingleCopyWrite(operation, interval);
    }

    private void EndSingleCopyWrite(SingleCopyWrite write, bool succeeded)
    {
        try
        {
            HostBackingObject.EndManagedWrite(write.Interval, succeeded);
            HostBackingObject.EndManagedWrite(write.Operation, succeeded);
        }
        finally { ExitLifetimeAccess(); }
    }

    internal bool CanonicalBytesEqual(ulong offset, ReadOnlySpan<byte> bytes)
    {
        if (!IsWithinBacking(offset, (ulong)bytes.Length) || !TryEnterLifetimeAccess())
        {
            return false;
        }

        try { return new ReadOnlySpan<byte>((void*)(AliasBase + offset), bytes.Length).SequenceEqual(bytes); }
        finally { ExitLifetimeAccess(); }
    }

    internal IHostBackingGpuAlias? CreateGpuAlias() =>
        _backing is { CreateGpuAlias: { } create } backing && IsAvailable ? create(backing) : null;

    /// <summary>
    /// The guest gave [offset, offset + size) back (direct memory released, flexible memory or
    /// pool pages unmapped). Every whole chunk inside it that no view maps is emptied: its pages
    /// go back to the OS and it reads zero when it is used again, a valid value for memory whose
    /// contents the PS5 leaves undefined after a release. Returns the bytes emptied; 0 for a
    /// single-object backing.
    /// </summary>
    public ulong Release(ulong offset, ulong size)
    {
        if (_chunkViews is null || size == 0 || !IsWithinBacking(offset, size)) return 0;
        var chunk = _backing!.ChunkBytes;
        var start = AlignUpTo(offset, chunk);
        var end = offset + size == Size ? Size : (offset + size) / chunk * chunk;
        if (end <= start) return 0;
        var interval = _backing.BeginManagedWriteOperation();
        if (!HostManagedWriteIntervals.AllowsBytes(interval)) return 0;
        var succeeded = false;
        var dsLock = DsLock.EnterMonitor(_lock, 7);
        try
        {
            // An unlocked copy reads and writes backing offsets, not views: one still in flight
            // over these chunks must finish before they are emptied. The lock is then held to
            // the end, so no new reservation can start over them.
            while (WaitForCopyConflict(start, end - start)) { }
            if (!TryEnterLifetimeAccess()) return 0;
            try
            {
                if (!IsAvailable) return 0;
                ulong emptied = 0, emptiedChunks = 0, zeroFilled = 0, skipped = 0;
                var cursor = start;
                while (cursor < end)
                {
                    var index = cursor / chunk;
                    var next = Math.Min(end, cursor + chunk);
                    if (_chunkViews[index] != 0)
                    {
                        skipped += next - cursor;
                        cursor = next;
                        continue;
                    }

                    // The longest run of unmapped chunks from here: one call for all of it.
                    var runEnd = next;
                    while (runEnd < end && _chunkViews[runEnd / chunk] == 0) runEnd = Math.Min(end, runEnd + chunk);
                    var result = DiscardCanonicalAlias(cursor, runEnd - cursor);
                    emptied += result.EmptiedBytes;
                    emptiedChunks += result.EmptiedChunks;
                    zeroFilled += result.ZeroFilledBytes;
                    cursor = runEnd;
                }

                GuestBackingTelemetry.NoteRelease(new HostBackingDiscard(emptied, emptiedChunks, zeroFilled), skipped);
                succeeded = true;
                return emptied;
            }
            finally { ExitLifetimeAccess(); }
        }
        finally
        {
            DsLock.ExitMonitor(_lock, 7, dsLock);
            HostBackingObject.EndManagedWrite(interval, succeeded);
        }
    }

    private static ulong AlignUpTo(ulong value, ulong alignment)
    {
        var remainder = value % alignment;
        return remainder == 0 ? value : value + (alignment - remainder);
    }

    private void RestoreViewRecord(ViewRecord record)
    {
        lock (_lock)
        {
            SetViewLocked(record);
            PublishSnapshot();
        }
    }

    private void RestoreViewMapping(ViewRecord record)
    {
        if (!TryMapReservedRange(record.Address, record.Size, record.Offset, record.Protection, true, out _))
        {
            OnFatal($"Could not restore the backing view at 0x{record.Address:X16}.");
        }
    }

    private void RestoreMappingsInReverseOrder(List<ViewRecord> records)
    {
        for (var index = records.Count - 1; index >= 0; index--)
        {
            RestoreViewMapping(records[index]);
        }
    }
}
