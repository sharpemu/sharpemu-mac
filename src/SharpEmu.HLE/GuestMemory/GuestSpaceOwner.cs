// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE.GpuMemory;
using SharpEmu.HLE.Host;

namespace SharpEmu.HLE.GuestMemory;

public enum RangeKind
{
    Backed,
    Private,
}

public readonly record struct OwnedRange(ulong Address, ulong Size, RangeKind Kind);

public sealed class GuestSpaceOwner : IDisposable, ICpuMappedCopySource, ICpuMappedReadWatchSource, ICpuMappedPublicationSource
{
    internal static Action<string> OnFatal = message => Environment.FailFast(message);

    public const ulong GuestPage = 0x4000;
    private const ulong UserAddressStart = 0x10_0000_0000;
    private const ulong UserAddressEnd = 0xFC_0000_0000;
    private const ulong MinimumPreReservedRange = 0x0100_0000;

    private readonly IHostViewMemory _host;
    private readonly SharedBackingViews _views;
    private readonly object _mappingLock = new();
    private readonly object[] _protectionLocks = CreateProtectionLocks();
    private readonly SortedList<ulong, ulong> _free = new();
    private readonly OwnedRangeTree _mapped = new();
    private readonly SortedList<ulong, ulong> _owned = new();
    private readonly bool _preReserveGuestAddressSpace;
    private bool _disposed;
    private readonly object _disposeGate = new(); // not used by mapping/protection/access paths
    private Exception? _disposeFailure;
    private bool _disposeComplete;
    private readonly HostAddressRange? _startupReservation;

    public GuestSpaceOwner(IHostViewMemory host, ulong backingSize, bool preReserveGuestAddressSpace = false,
        HostAddressRange? startupReservation = null)
    {
        _host = host;
        Granularity = host.Granularity;
        if (startupReservation is { } reservation &&
            (!IsValidRange(reservation.Address, reservation.Size) ||
             reservation.Address % Granularity != 0 || reservation.Size % Granularity != 0))
            throw new ArgumentOutOfRangeException(nameof(startupReservation));
        _startupReservation = startupReservation;
        _preReserveGuestAddressSpace = preReserveGuestAddressSpace;
        PreReserveGuestAddressSpace();
        _views = new SharedBackingViews(host, backingSize);
        if (!_views.IsAvailable)
        {
            var megabytes = backingSize / (1024 * 1024);
            OnFatal(OperatingSystem.IsWindows()
                ? $"Could not allocate {megabytes} MB for guest direct memory. Windows requires this amount of available system commit. Close other applications or increase the paging file size."
                : $"Could not allocate {megabytes} MB for guest direct memory.");
        }
    }

    public ulong Granularity { get; }

    public ulong AliasBase => _views.AliasBase;

    public ulong BackingSize => _views.Size;

    /// <summary>What another process needs to map the backing; null unless the host made it shareable.</summary>
    public GuestBackingExport? BackingExport => _views.BackingExport;

    /// <summary>The shared table of the backing's views; null unless the backing is shareable.</summary>
    public GuestViewTable? ViewTable => _views.ViewTable;

    public CpuMappedPublicationPreparationResult TryPrepareMappedReadValidation(CpuMappedCopyObservation original,
        out CpuMappedReadValidation? validation)
    {
        validation = null;
        if (Volatile.Read(ref _disposed)) return CpuMappedPublicationPreparationResult.Rejected;
        var result = _views.TryPrepareMappedReadValidation(original, out validation);
        if (!Volatile.Read(ref _disposed)) return result;
        validation?.Dispose(); validation = null; return CpuMappedPublicationPreparationResult.Rejected;
    }

    public bool CanWaitForMappedPublicationValidation => _views.CanWaitForMappedPublicationValidation;

    public CpuMappedPublicationPreparationResult TryPrepareMappedPublicationValidation(CpuMappedCopyObservation original,
        out CpuMappedPublicationValidation? validation)
    {
        validation = null;
        if (Volatile.Read(ref _disposed)) return CpuMappedPublicationPreparationResult.Rejected;
        var result = _views.TryPrepareMappedPublicationValidation(original, out validation);
        if (!Volatile.Read(ref _disposed)) return result;
        validation?.Dispose(); validation = null; return CpuMappedPublicationPreparationResult.Rejected;
    }

    public CpuMappedPublicationPreparationResult TryPrepareMappedPublicationValidation(CpuMappedCopyObservation original,
        out CpuMappedPublicationValidation? validation,
        out CpuMappedPublicationValidationRefusal refusal)
    {
        validation = null; refusal = default;
        if (Volatile.Read(ref _disposed)) return CpuMappedPublicationPreparationResult.Rejected;
        var result = _views.TryPrepareMappedPublicationValidation(original, out validation, out refusal);
        if (!Volatile.Read(ref _disposed)) return result;
        validation?.Dispose(); validation = null; refusal = default; return CpuMappedPublicationPreparationResult.Rejected;
    }

    public CpuMappedPublicationPreparationResult TryPrepareMappedPublication(CpuMappedCopyObservation original,
        IReadOnlyList<CpuMappedWriteRun> runs, out CpuMappedPublication? publication)
        => TryPrepareMappedPublication(original, runs, out publication, out _);

    public CpuMappedPublicationPreparationResult TryPrepareMappedPublication(CpuMappedCopyObservation original,
        IReadOnlyList<CpuMappedWriteRun> runs, out CpuMappedPublication? publication,
        out CpuMappedPublicationPreparationRefusal refusal)
    {
        publication = null; refusal = default;
        if (Volatile.Read(ref _disposed)) return CpuMappedPublicationPreparationResult.Rejected;
        var result = _views.TryPrepareMappedPublication(original, runs, out publication, out refusal);
        if (!Volatile.Read(ref _disposed)) return result;
        publication?.Dispose(); publication = null; refusal = default; return CpuMappedPublicationPreparationResult.Rejected;
    }

    public CpuMappedCopyResult TryReadMappedCopyWithWriteWatch(ulong footprintAddress, ulong footprintByteCount,
        ulong sourceAddress, Span<byte> destination, out CpuMappedCopyObservation? observation)
    {
        observation = null;
        if (Volatile.Read(ref _disposed)) return CpuMappedCopyResult.Rejected;
        var result = _views.TryReadMappedCopyWithWriteWatch(footprintAddress, footprintByteCount, sourceAddress, destination, out observation);
        if (!Volatile.Read(ref _disposed)) return result;
        if (observation is not null) _views.ReleaseMappedCopyWriteWatch(observation);
        observation = null; return CpuMappedCopyResult.Rejected;
    }
    public bool CanWaitForMappedReadWatchStatus =>
        !Volatile.Read(ref _disposed) && _views.CanWaitForMappedReadWatchStatus;

    public bool CanWaitForMappedReadWatchEnrollment =>
        !Volatile.Read(ref _disposed) && _views.CanWaitForMappedReadWatchEnrollment;

    public CpuMappedCopyResult TryReadMappedCopyWithWriteWatchAdmissionWait(ulong footprintAddress,
        ulong footprintByteCount, ulong sourceAddress, Span<byte> destination,
        Func<bool> originalStillCurrent, out CpuMappedCopyObservation? observation)
    {
        ArgumentNullException.ThrowIfNull(originalStillCurrent);
        observation = null;
        bool WrapperCurrent() => !Volatile.Read(ref _disposed) && CanWaitForMappedReadWatchEnrollment;
        bool OriginalWrapper() => WrapperCurrent() && originalStillCurrent() && WrapperCurrent();
        if (!OriginalWrapper()) return CpuMappedCopyResult.Rejected;
        var result = _views.TryReadMappedCopyWithWriteWatchAdmissionWait(footprintAddress, footprintByteCount,
            sourceAddress, destination, OriginalWrapper, out observation);
        if (!Volatile.Read(ref _disposed)) return result;
        if (observation is not null) _views.ReleaseMappedCopyWriteWatch(observation);
        observation = null; return CpuMappedCopyResult.Rejected;
    }

    public bool IsMappedCopyWriteWatchCurrent(CpuMappedCopyObservation observation) =>
        !Volatile.Read(ref _disposed) && _views.IsMappedCopyWriteWatchCurrent(observation);
    public bool IsMappedCopyWriteWatchCurrent(CpuMappedCopyObservation observation,
        out CpuMappedReadWatchDiagnostic diagnostic)
    {
        if (Volatile.Read(ref _disposed))
        { diagnostic = new() { OwnerClosure = "owner-pre-disposed", BooleanCurrent = false }; return false; }
        return _views.IsMappedCopyWriteWatchCurrent(observation, out diagnostic);
    }
    public CpuMappedReadWatchStatus GetMappedCopyWriteWatchStatus(CpuMappedCopyObservation observation)
    {
        if (Volatile.Read(ref _disposed)) return CpuMappedReadWatchStatus.Stale;
        var status = _views.GetMappedCopyWriteWatchStatus(observation);
        return Volatile.Read(ref _disposed) ? CpuMappedReadWatchStatus.Stale : status;
    }
    public void ReleaseMappedCopyWriteWatch(CpuMappedCopyObservation observation) => _views.ReleaseMappedCopyWriteWatch(observation);

    public CpuMappedReadWatchStatus GetMappedCopyWriteWatchStatus(CpuMappedCopyObservation observation,
        out CpuMappedReadWatchDiagnostic diagnostic)
    {
        if (Volatile.Read(ref _disposed))
        { diagnostic = new() { OwnerClosure = "owner-pre-disposed" }; return CpuMappedReadWatchStatus.Stale; }
        var status = _views.GetMappedCopyWriteWatchStatus(observation, out diagnostic);
        if (!Volatile.Read(ref _disposed)) return status;
        diagnostic = diagnostic with { OwnerClosure = "owner-post-disposed" }; return CpuMappedReadWatchStatus.Stale;
    }

    public CpuMappedCopyResult TryReadMappedCopy(ulong footprintAddress, ulong footprintByteCount,
        ulong sourceAddress, Span<byte> destination, out CpuMappedCopyObservation? observation)
    {
        observation = null;
        if (Volatile.Read(ref _disposed)) return CpuMappedCopyResult.Rejected;
        var result = _views.TryReadMappedCopy(footprintAddress, footprintByteCount, sourceAddress, destination, out observation);
        if (Volatile.Read(ref _disposed)) { observation = null; return CpuMappedCopyResult.Rejected; }
        return result;
    }
    public bool IsMappedCopyObservationCurrent(CpuMappedCopyObservation observation) =>
        !Volatile.Read(ref _disposed) && _views.IsMappedCopyObservationCurrent(observation);

    public bool TryClearBacking(ulong offset, ulong size) => _views.Clear(offset, size);

    /// <summary>Bytes per separately releasable chunk of the backing; 0 when it is one object.</summary>
    public ulong BackingChunkBytes => _views.ChunkBytes;

    /// <summary>Bytes per purgeable object behind the chunks; 0 when the backing is one object.</summary>
    public ulong BackingObjectBytes => _views.ObjectBytes;

    /// <summary>The released-chunk state of a chunked backing; null for one object.</summary>
    public GuestSingleCopy? SingleCopy => _views.SingleCopy;

    /// <summary>Gives the whole unmapped chunks of freed backing bytes back to the OS; the bytes emptied.</summary>
    public ulong ReleaseBacking(ulong offset, ulong size) => _disposed ? 0 : _views.Release(offset, size);

    public bool TryWriteBacking(ulong address, ReadOnlySpan<byte> data) => _views.TryWriteBacking(address, data);
    public bool TryWriteBacking(ulong address, ReadOnlySpan<byte> data, out OrdinaryBackingWriteReservation? busy)
        => _views.TryWriteBacking(address, data, out busy);

    public bool TryWriteBacking(ulong address, ReadOnlySpan<byte> data, out OrdinaryBackingWriteReservation? busy,
        out SharedWriteFailure failure) => _views.TryWriteBacking(address, data, out busy, out failure);

    public bool CanEmitOrdinaryWriteDiagnostic
    {
        get
        {
            if (Monitor.IsEntered(_mappingLock) || Monitor.IsEntered(_disposeGate)) return false;
            foreach (var stripe in _protectionLocks) if (Monitor.IsEntered(stripe)) return false;
            return _views.CanEmitOrdinaryWriteDiagnostic;
        }
    }

    public bool TryReadBacking(ulong address, Span<byte> data) => _views.TryReadBacking(address, data);

    // Reads through a view that covers the whole range without any lock; false otherwise.
    public bool TryReadSingleView(ulong address, Span<byte> data) => _views.TryReadSingleView(address, data);

    public void RequireOrdinaryWriteWaitOutsideGates()
    {
        if (Monitor.IsEntered(_mappingLock) || Monitor.IsEntered(_disposeGate))
            throw new InvalidOperationException("Ordinary write wait cannot hold guest mapping/disposal gates.");
        foreach (var stripe in _protectionLocks)
            if (Monitor.IsEntered(stripe))
                throw new InvalidOperationException("Ordinary write wait cannot hold guest protection gates.");
    }
    public bool TryWriteSingleView(ulong address, ReadOnlySpan<byte> data) => _views.TryWriteSingleView(address, data);

    public bool TryWritePairSingleView(ulong firstAddress, ReadOnlySpan<byte> first, ulong secondAddress, ReadOnlySpan<byte> second) =>
        _views.TryWritePairSingleView(firstAddress, first, secondAddress, second);

    public bool TryFillSingleView<TState>(ulong address, int length, TState state, GuestSpanFill<TState> fill, out int filled) =>
        _views.TryFillSingleView(address, length, state, fill, out filled);
    public bool TryCopySingleView(ulong destination, ulong source, ulong size) => _views.TryCopySingleView(destination, source, size);

    public object AliasSnapshot => _views.AliasSnapshot;

    public bool TryEnterAliasAccess() => _views.TryEnterAliasAccess();

    public void ExitAliasAccess() => _views.ExitAliasAccess();

    public bool TryResolveAlias(ulong address, ulong size, out ulong alias) => _views.TryResolveAlias(address, size, out alias);

    public bool TryResolveAliasSpan(ulong address, ulong size, out ulong alias) =>
        _views.TryResolveAliasSpan(address, size, out alias);

    public bool TryCopyBacking(ulong destination, ulong source, ulong size) => _views.TryCopyBacking(destination, source, size);

    public bool IsBacked(ulong address, ulong size) => _views.Contains(address, size);

    public bool IsBackedWithoutLock(ulong address, ulong size) => _views.ContainsWithoutLock(address, size);

    public bool IsRestoredView(ulong address) => _views.IsRestoredView(address);

    public bool TryReserveAddressRange(ulong address, ulong size)
    {
        if (!IsValidRange(address, size) || address % Granularity != 0 || size % Granularity != 0)
        {
            return false;
        }

        using (EnterMappingLock())
        {
            if (_disposed || OverlapsOwnedLocked(address, size) || _host.ReserveHole(address, size) != address)
            {
                return false;
            }

            AddFreeRange(address, size);
            _owned.Add(address, size);
            return true;
        }
    }

    public bool TryReserveFreeRange(ulong address, ulong size)
    {
        using var reservationScope = GuestMemoryProfile.Measure(GuestMemoryProfile.Operation.Reservation);
        if (!IsValidAlignedRange(address, size))
            return false;
        var end = address + size;
        var reservationEnd = AlignUp(end, Granularity);
        if (reservationEnd == 0)
            return false;

        using (EnterMappingLock())
        {
            List<(ulong Address, ulong Size)> additions;
            using (GuestMemoryProfile.Measure(GuestMemoryProfile.Operation.ReservationSearch))
            {
                if (_disposed)
                    return false;
                if (FindFreeRangeIndex(address, size) >= 0)
                    return true;
                var mappedRange = FindMappedRangeAtOrBelow(end - 1);
                if (mappedRange.Size != 0 && mappedRange.Address + mappedRange.Size > address)
                {
                    ReservationDiagnostics.Record("owned-overlap", address, size, Granularity,
                        mappedRange.Address, mappedRange.Size);
                    return false;
                }

                // Keep owned placeholders. Reserve only the gaps between them.
                additions = new List<(ulong Address, ulong Size)>();
                var current = address - address % Granularity;
                var ownedIndex = Math.Max(0, RangeSearch.FindLastIndexAtOrBelow(_owned, current));
                for (; ownedIndex < _owned.Count; ownedIndex++)
                {
                    var rangeAddress = _owned.Keys[ownedIndex];
                    var rangeEnd = rangeAddress + _owned.Values[ownedIndex];
                    if (rangeEnd <= current)
                        continue;
                    if (rangeAddress >= reservationEnd)
                        break;
                    if (current < rangeAddress)
                        additions.Add((current, rangeAddress - current));
                    current = Math.Max(current, Math.Min(reservationEnd, rangeEnd));
                }
                if (current < reservationEnd)
                    additions.Add((current, reservationEnd - current));
            }

            using (GuestMemoryProfile.Measure(GuestMemoryProfile.Operation.ReservationHost))
            {
                for (var index = 0; index < additions.Count; index++)
                {
                    var range = additions[index];
                    if (_host.ReserveHole(range.Address, range.Size) == range.Address)
                        continue;
                    // No new range has joined an existing placeholder yet.
                    for (var previous = index - 1; previous >= 0; previous--)
                    {
                        var rollback = additions[previous];
                        if (!_host.FreeHole(rollback.Address, rollback.Size))
                            OnFatal($"Cannot release the new reservation at 0x{rollback.Address:X16}.");
                    }
                    return false;
                }
            }

            using (GuestMemoryProfile.Measure(GuestMemoryProfile.Operation.ReservationPublish))
            {
                foreach (var range in additions)
                {
                    _owned.Add(range.Address, range.Size);
                    AddFreeRange(range.Address, range.Size);
                }
                return FindFreeRangeIndex(address, size) >= 0;
            }
        }
    }

    public bool OwnsReservedRange(ulong address, ulong size)
    {
        if (!IsValidRange(address, size))
        {
            return false;
        }

        lock (_mappingLock)
        {
            var index = RangeSearch.FindLastIndexAtOrBelow(_owned, address);
            return index >= 0 && address + size <= _owned.Keys[index] + _owned.Values[index];
        }
    }

    public bool ContainsFreeRange(ulong address, ulong size)
    {
        if (!IsValidAlignedRange(address, size))
        {
            return false;
        }

        lock (_mappingLock)
        {
            return FindFreeRangeIndex(address, size) >= 0;
        }
    }

    public ulong FindFreeAddress(ulong searchStart, ulong searchEnd, ulong size, ulong alignment)
    {
        if (size == 0 || alignment == 0)
        {
            return 0;
        }

        alignment = Math.Max(alignment, GuestPage);
        lock (_mappingLock)
        {
            return FindAlignedFreeAddress(searchStart, searchEnd, size, alignment);
        }
    }

    // Hint for callers that scan for a free guest address: the lowest address at
    // or above <paramref name="address"/> that carries no host mapping for
    // <paramref name="size"/> bytes, or 0 when the host cannot say. Reserving
    // the result still has to be attempted, so a stale answer only costs a
    // retry.
    public ulong FindUnmappedHostRange(ulong address, ulong size, ulong searchEnd) =>
        _host.FindUnmappedRange(address, size, searchEnd);

    public bool SetAccess(ulong address, ulong size, HostPageProtection protection)
    {
        if (!IsValidAlignedRange(address, size))
        {
            return false;
        }

        using (EnterMappingLock())
        {
            return SetAccessLocked(address, size, protection);
        }
    }

    // Page watchers protect host pages, which are smaller than the guest page.
    public bool SetTransientAccess(ulong address, ulong size, HostPageProtection protection)
    {
        if (!IsValidRange(address, size) || address % _host.PageSize != 0 || size % _host.PageSize != 0)
        {
            return false;
        }

        using var protectionScope = new ProtectionRangeScope(_protectionLocks, address, size);
        return SetAccessLocked(address, size, protection);
    }

    public bool SetDerivedAccess(ulong address, ulong size, GuestPageProtection guestAccess,
        HostPageProtection derivedHostAccess, bool ordinaryEnrollment)
    {
        if (!IsValidRange(address, size) || address % _host.PageSize != 0 || size % _host.PageSize != 0) return false;
        using var protectionScope = new ProtectionRangeScope(_protectionLocks, address, size);
        return SetAccessLocked(address, size, derivedHostAccess, guestAccess, ordinaryEnrollment);
    }

    private MappingLockScope EnterMappingLock() => new(_mappingLock, _protectionLocks);

    // Mapping changes own every address lock before they inspect or change the range tables.
    // Transient protection needs only its address locks and does not allocate reader state.
    private ref struct MappingLockScope
    {
        private readonly object _mappingLock;
        private ProtectionRangeScope _protectionScope;

        public MappingLockScope(object mappingLock, object[] protectionLocks)
        {
            _mappingLock = mappingLock;
            var taken = false;
            try
            {
                Monitor.Enter(mappingLock, ref taken);
                _protectionScope = new ProtectionRangeScope(protectionLocks, 0,
                    (ulong)protectionLocks.Length << ProtectionBlockShift, profileWait: false);
            }
            catch
            {
                if (taken) Monitor.Exit(mappingLock);
                throw;
            }
        }

        public void Dispose()
        {
            _protectionScope.Dispose();
            Monitor.Exit(_mappingLock);
        }
    }

    private const int ProtectionLockCount = 256;
    private const int ProtectionBlockShift = 22;

    private static object[] CreateProtectionLocks()
    {
        var protectionLocks = new object[ProtectionLockCount];
        for (var index = 0; index < protectionLocks.Length; index++) protectionLocks[index] = new object();
        return protectionLocks;
    }

    // Address locks keep views alive and order overlapping protection changes.
    // Lock collisions only serialize extra ranges; every caller acquires locks in index order.
    private ref struct ProtectionRangeScope
    {
        private readonly object[] _protectionLocks;
        private readonly int _firstLockIndex;
        private readonly int _lockCount;
        private int _acquiredLockCount;

        public ProtectionRangeScope(object[] protectionLocks, ulong address, ulong size, bool profileWait = true)
        {
            _protectionLocks = protectionLocks;
            var firstBlock = address >> ProtectionBlockShift;
            var lastBlock = (address + size - 1) >> ProtectionBlockShift;
            _lockCount = (int)Math.Min((ulong)protectionLocks.Length, lastBlock - firstBlock + 1);
            _firstLockIndex = _lockCount == protectionLocks.Length ? 0 : (int)(firstBlock % (ulong)protectionLocks.Length);
            _acquiredLockCount = 0;
            using var profile = profileWait ? GpuMemoryAccessProfile.MeasureBackingProtectionWait() : default;
            try
            {
                while (_acquiredLockCount < _lockCount)
                {
                    var taken = false;
                    try { Monitor.Enter(protectionLocks[GetLockIndex(_acquiredLockCount)], ref taken); }
                    finally { if (taken) _acquiredLockCount++; }
                }
            }
            catch
            {
                Dispose();
                throw;
            }
        }

        private readonly int GetLockIndex(int acquisitionIndex)
        {
            var wrappedCount = Math.Max(0, _firstLockIndex + _lockCount - _protectionLocks.Length);
            return acquisitionIndex < wrappedCount ? acquisitionIndex : _firstLockIndex + acquisitionIndex - wrappedCount;
        }

        public void Dispose()
        {
            while (_acquiredLockCount > 0) Monitor.Exit(_protectionLocks[GetLockIndex(--_acquiredLockCount)]);
        }
    }

    public bool MapShared(ulong address, ulong size, ulong offset, HostPageProtection protection, out HostViewFailure failure)
    {
        using var mappingScope = GuestMemoryProfile.Measure(GuestMemoryProfile.Operation.BackingMap);
        failure = HostViewFailure.AddressUnavailable;
        if (!IsValidAlignedRange(address, size))
        {
            return false;
        }

        if (offset % GuestPage != 0 || offset >= BackingSize || size > BackingSize - offset)
        {
            failure = HostViewFailure.OffsetOutOfBounds;
            return false;
        }

        using (EnterMappingLock())
        {
            if (_disposed || !TryTakeFreeRange(address, size))
            {
                failure = HostViewFailure.AddressUnavailable;
                return false;
            }

            if (_views.TryMapReservedRange(address, size, offset, GetBackingProtection(protection), out failure))
            {
                if (!TryAddMappedRange(address, size, RangeKind.Backed))
                {
                    OnFatal($"Cannot map shared backing. A mapped range overlaps 0x{address:X16}.");
                }

                return true;
            }

            AddFreeRange(address, size);
            return false;
        }
    }

    public bool UnmapShared(ulong address, ulong size)
    {
        using var mappingScope = GuestMemoryProfile.Measure(GuestMemoryProfile.Operation.BackingUnmap);
        if (!IsValidAlignedRange(address, size))
        {
            return false;
        }

        using (EnterMappingLock())
        {
            if (_disposed || !HasOnlyRangeKind(address, size, RangeKind.Backed))
            {
                return false;
            }

            using (GuestMemoryProfile.Measure(GuestMemoryProfile.Operation.BackingUnmapHost))
            {
                if (!_views.Unmap(address, size, out var holePreserved) || !holePreserved)
                {
                    return false;
                }
            }

            using var bookkeepingProfile = GuestMemoryProfile.Measure(GuestMemoryProfile.Operation.BackingUnmapBookkeeping);
            if (!TryRemoveMappedRanges(address, size, RangeKind.Backed))
            {
                OnFatal($"No shared backing record exists at 0x{address:X16}.");
            }

            AddFreeRange(address, size);
            return true;
        }
    }

    public bool AllocatePrivate(ulong address, ulong size, HostPageProtection protection)
    {
        if (!IsValidAlignedRange(address, size))
        {
            return false;
        }

        using (EnterMappingLock())
        {
            if (_disposed || !TryTakeFreeRange(address, size))
            {
                return false;
            }

            if (_host.CommitPrivate(address, size, GetBackingProtection(protection)))
            {
                if (!TryAddMappedRange(address, size, RangeKind.Private))
                {
                    OnFatal($"Cannot map private memory. A mapped range overlaps 0x{address:X16}.");
                }

                return true;
            }

            AddFreeRange(address, size);
            return false;
        }
    }

    public bool FreePrivate(ulong address, ulong size)
    {
        if (!IsValidAlignedRange(address, size))
        {
            return false;
        }

        using (EnterMappingLock())
        {
            if (_disposed || !HasOnlyRangeKind(address, size, RangeKind.Private))
            {
                return false;
            }

            var end = address + size;
            var current = address;
            var pieces = new List<(ulong Address, ulong Size)>();
            while (current < end)
            {
                var range = FindMappedRangeAtOrBelow(current);
                var pieceEnd = Math.Min(end, range.Address + range.Size);
                pieces.Add((current, pieceEnd - current));
                current = pieceEnd;
            }

            foreach (var piece in pieces)
            {
                if (!_host.ReleasePrivate(piece.Address, piece.Size))
                {
                    OnFatal($"Could not release private memory at 0x{piece.Address:X16}.");
                }
            }

            if (pieces.Count > 1 && !_host.JoinHoles(address, size))
            {
                OnFatal($"Could not join the free ranges after releasing private memory at 0x{address:X16}.");
            }

            if (!TryRemoveMappedRanges(address, size, RangeKind.Private))
            {
                OnFatal($"No private memory record exists at 0x{address:X16}.");
            }

            AddFreeRange(address, size);
            return true;
        }
    }

    // Keep the backing object and the startup reservation for reuse.
    public void ReleaseAddressRanges()
    {
        using (EnterMappingLock())
        {
            if (_disposed)
            {
                return;
            }

            foreach (var range in _mapped.ToArray())
            {
                var released = range.Kind == RangeKind.Backed
                    ? _views.Unmap(range.Address, range.Size, out _)
                    : _host.ReleasePrivate(range.Address, range.Size);
                if (!released)
                {
                    OnFatal($"Could not release the reserved range at 0x{range.Address:X16}.");
                }
            }

            foreach (var (address, size) in _owned)
            {
                if (_startupReservation is { } startupRange && address == startupRange.Address && size == startupRange.Size)
                {
                    if (!_host.JoinHoles(address, size))
                        OnFatal($"Could not restore the startup guest reservation at 0x{address:X16}.");
                    continue;
                }
                if (!_host.FreeOwnedRange(address, size))
                {
                    OnFatal($"Could not release the owned range at 0x{address:X16}.");
                }
            }

            _owned.Clear();
            _free.Clear();
            _mapped.Clear();
            // A new image load needs the guest address space reserved again.
            PreReserveGuestAddressSpace();
        }
    }

    private void PreReserveGuestAddressSpace()
    {
        if (_startupReservation is { } retainedRange)
        {
            _owned.Add(retainedRange.Address, retainedRange.Size);
            AddFreeRange(retainedRange.Address, retainedRange.Size);
            return;
        }

        if (!_preReserveGuestAddressSpace)
        {
            return;
        }

        foreach (var range in _host.ReserveFreeAddressRanges(UserAddressStart, UserAddressEnd, MinimumPreReservedRange))
        {
            _owned.Add(range.Address, range.Size);
            AddFreeRange(range.Address, range.Size);
        }
    }

    public void Dispose()
    {
        if (Monitor.IsEntered(_mappingLock) || Monitor.IsEntered(_disposeGate) ||
            _protectionLocks.Any(Monitor.IsEntered))
            throw new InvalidOperationException("Guest disposal cannot drain under mapping/protection gates.");
        lock (_disposeGate)
        {
            using (EnterMappingLock())
            {
                if (_disposeComplete) return;
                if (_disposeFailure is not null)
                    throw new InvalidOperationException("Guest range cleanup is quarantined.", _disposeFailure);
                _disposed = true; // revoke every new mapping/protection request before dropping gates
                _free.Clear();
            }
            // Alias copies need neither mapping gate nor protection stripe to exit.
            // Leave owned records intact until each positive cleanup succeeds.
            _views.Dispose();
            (ulong Address, ulong Size)[] owned;
            using (EnterMappingLock()) owned = _owned.Select(range => (range.Key, range.Value)).ToArray();
            foreach (var (address, size) in owned)
            {
                try
                {
                    if (!_host.FreeOwnedRange(address, size))
                        throw new InvalidOperationException($"Guest range cleanup outcome unknown at 0x{address:X16}.");
                }
                catch (Exception error)
                {
                    using (EnterMappingLock()) _disposeFailure = error;
                    throw;
                }
                using (EnterMappingLock()) _owned.Remove(address);
            }
            using (EnterMappingLock())
            {
                _mapped.Clear();
                _disposeComplete = true;
            }
        }
    }

    private static bool IsValidRange(ulong address, ulong size) =>
        address != 0 && size != 0 && size <= ulong.MaxValue - address;

    private static bool IsValidAlignedRange(ulong address, ulong size) =>
        IsValidRange(address, size) && address % GuestPage == 0 && size % GuestPage == 0;

    private static HostPageProtection GetBackingProtection(HostPageProtection protection) =>
        protection is HostPageProtection.Execute or HostPageProtection.ReadExecute
            or HostPageProtection.ReadWriteExecute or HostPageProtection.ExecuteWriteCopy
            ? HostPageProtection.ReadWriteExecute
            : HostPageProtection.ReadWrite;

    private static ulong AlignUp(ulong value, ulong alignment)
    {
        var remainder = value % alignment;
        var increment = remainder != 0 ? alignment - remainder : 0;
        return increment <= ulong.MaxValue - value ? value + increment : 0;
    }

    private bool OverlapsOwnedLocked(ulong address, ulong size)
    {
        var index = RangeSearch.FindLastIndexAtOrBelow(_owned, address + size - 1);
        return index >= 0 && _owned.Keys[index] + _owned.Values[index] > address;
    }

    private bool SetAccessLocked(ulong address, ulong size, HostPageProtection protection,
        GuestPageProtection? originalGuestAccess = null, bool ordinaryEnrollment = false)
    {
        if (_disposed) return false;
        var end = address + size;
        if (size == 0) return true;
        var range = FindMappedRangeAtOrBelow(address);
        if (range.Size == 0 || range.Address + range.Size <= address)
            range = _mapped.FindAtOrAbove(address);
        while (range.Size != 0 && range.Address < end)
        {
            var start = Math.Max(address, range.Address);
            var stop = Math.Min(end, range.Address + range.Size);
            if (start < stop)
            {
                using var profile = GpuMemoryAccessProfile.MeasureHostProtectionCall(stop - start);
                var changed = originalGuestAccess.HasValue && _host is IHostDerivedProtectionMemory explicitHost
                    ? explicitHost.ChangeDerivedAccess(start, stop - start, originalGuestAccess.Value, protection, ordinaryEnrollment)
                    : _host.ChangeAccess(start, stop - start, protection);
                if (!changed) return false;
                // A shareable backing logs it for the process that replicates its views.
                if (range.Kind == RangeKind.Backed) _views.ViewTable?.Protect(start, stop - start, protection);
            }
            range = _mapped.FindAtOrAbove(range.Address + range.Size);
        }

        return true;
    }

    private bool TryTakeFreeRange(ulong address, ulong size)
    {
        var index = FindFreeRangeIndex(address, size);
        if (index < 0 || !TrySplitFreeRange(index, address, size))
        {
            return false;
        }

        if (!_free.TryGetValue(address, out var found) || found != size)
        {
            return false;
        }

        _free.Remove(address);
        return true;
    }

    private bool TrySplitFreeRange(int index, ulong address, ulong size)
    {
        var start = _free.Keys[index];
        var end = start + _free.Values[index];
        if (start == address && end == address + size)
        {
            return true;
        }

        if (start != address)
        {
            var leftSize = address - start;
            if (!_host.SplitHole(start, leftSize))
            {
                return false;
            }

            _free[start] = leftSize;
            _free[address] = end - address;
        }

        if (_free[address] != size)
        {
            if (!_host.SplitHole(address, size))
            {
                return false;
            }

            var rightStart = address + size;
            var rightSize = _free[address] - size;
            _free[address] = size;
            _free[rightStart] = rightSize;
        }

        return true;
    }

    private void AddFreeRange(ulong address, ulong size)
    {
        using var publicationProfile = GuestMemoryProfile.Measure(GuestMemoryProfile.Operation.FreeRangePublication);
        if (address == 0 || size == 0)
        {
            return;
        }

        _free[address] = size;
        var index = _free.IndexOfKey(address);
        var start = address;
        var end = address + size;
        var first = index;
        while (first > 0 && _free.Keys[first - 1] + _free.Values[first - 1] == start)
        {
            first--;
            start = _free.Keys[first];
        }

        var last = index;
        while (last + 1 < _free.Count && _free.Keys[last + 1] == end)
        {
            last++;
            end = _free.Keys[last] + _free.Values[last];
        }

        // Keep adjacent entries separate when the host cannot join their free ranges.
        var joined = false;
        if (first != index || last != index)
        {
            using var joinProfile = GuestMemoryProfile.Measure(GuestMemoryProfile.Operation.FreeRangeHostJoin);
            joined = _host.JoinHoles(start, end - start);
        }
        if (joined)
        {
            for (var remove = last; remove >= first; remove--)
            {
                _free.RemoveAt(remove);
            }

            _free[start] = end - start;
        }
    }

    private int FindFreeRangeIndex(ulong address, ulong size)
    {
        var index = RangeSearch.FindLastIndexAtOrBelow(_free, address);
        if (index < 0)
        {
            return -1;
        }

        var start = _free.Keys[index];
        var end = start + _free.Values[index];
        return address >= start && address + size <= end ? index : -1;
    }

    private ulong FindAlignedFreeAddress(ulong searchStart, ulong searchEnd, ulong size, ulong alignment)
    {
        if (searchStart >= searchEnd || size > searchEnd - searchStart)
        {
            return 0;
        }

        var index = Math.Max(0, RangeSearch.FindLastIndexAtOrBelow(_free, searchStart));
        for (; index < _free.Count; index++)
        {
            var start = _free.Keys[index];
            if (start >= searchEnd) break;
            var length = _free.Values[index];
            var candidate = AlignUp(Math.Max(start, searchStart), alignment);
            if (candidate != 0 && candidate < searchEnd && size <= searchEnd - candidate &&
                candidate >= start && candidate <= start + length && size <= start + length - candidate)
            {
                return candidate;
            }
        }

        return 0;
    }

    private OwnedRange FindMappedRangeAtOrBelow(ulong address) =>
        _mapped.FindAtOrBelow(address);

    private bool HasOnlyRangeKind(ulong address, ulong size, RangeKind kind)
    {
        var end = address + size;
        var current = address;
        while (current < end)
        {
            var range = FindMappedRangeAtOrBelow(current);
            if (range.Size == 0 || current >= range.Address + range.Size || range.Kind != kind)
            {
                return false;
            }

            current = Math.Min(end, range.Address + range.Size);
        }

        return true;
    }

    private bool TryAddMappedRange(ulong address, ulong size, RangeKind kind)
    {
        var end = address + size;
        var preceding = FindMappedRangeAtOrBelow(end - 1);
        if (preceding.Size != 0 && preceding.Address + preceding.Size > address)
        {
            return false;
        }

        return _mapped.Add(new OwnedRange(address, size, kind));
    }

    private bool TryRemoveMappedRanges(ulong address, ulong size, RangeKind kind)
    {
        using var removalProfile = GuestMemoryProfile.Measure(GuestMemoryProfile.Operation.MappedRangeRemoval);
        using (GuestMemoryProfile.Measure(GuestMemoryProfile.Operation.MappedRangeRemovalValidation))
        {
            if (!HasOnlyRangeKind(address, size, kind))
                return false;
        }

        var end = address + size;
        if (size == 0)
            return true;
        var current = address;
        while (current < end)
        {
            OwnedRange original;
            using (GuestMemoryProfile.Measure(GuestMemoryProfile.Operation.MappedRangeRemovalSearch))
                original = FindMappedRangeAtOrBelow(current);
            var partEnd = Math.Min(end, original.Address + original.Size);
            using (GuestMemoryProfile.Measure(GuestMemoryProfile.Operation.MappedRangeTreeRemoval))
                _mapped.Remove(original);
            if (original.Address < address)
            {
                using var splitProfile = GuestMemoryProfile.Measure(GuestMemoryProfile.Operation.MappedRangeSplitPublication);
                _mapped.Add(original with { Size = address - original.Address });
            }

            if (partEnd < original.Address + original.Size)
            {
                using var splitProfile = GuestMemoryProfile.Measure(GuestMemoryProfile.Operation.MappedRangeSplitPublication);
                _mapped.Add(original with { Address = partEnd, Size = original.Address + original.Size - partEnd });
            }
            current = partEnd;
        }

        return true;
    }
}
