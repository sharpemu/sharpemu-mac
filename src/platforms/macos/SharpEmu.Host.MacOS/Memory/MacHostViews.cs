// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using SharpEmu.HLE;
using SharpEmu.HLE.Host;

namespace SharpEmu.Host.MacOS;

/// <summary>
/// macOS placeholder/view memory: POSIX shared-memory objects aliased into the
/// guest address space with MAP_FIXED, tracked in the neutral
/// <see cref="HostViewRegions"/> table so VirtualQuery-style lookups see them.
/// </summary>
internal sealed unsafe partial class MacHostViews : IHostViewMemory, IHostMappedCopyMemory, IHostMappedReadWatchMemory, IHostMappedPublicationMemory, IHostDerivedProtectionMemory
{
    internal static bool FailAliasMapForTests;

    public MacHostViews() => MacMappedCopyState.Initialize();

    public bool CanWaitForOrdinaryWriteAdmission =>
        !Monitor.IsEntered(HostViewRegions.Gate) && !MacHostMemory.IsNativeMappingGateHeld;

    public HostMappedCopyLedger.OrdinaryWriteIdentity? CaptureOrdinaryWriteIdentity(
        HostBackingObject backing, ulong address, ulong size) =>
        MacMappedCopyState.Ledger?.CaptureOrdinaryWriteIdentity(backing, address, size);

    public HostMappedCopyLedger.OrdinaryWriteIdentity? CaptureOrdinaryWriteIdentity(
        HostBackingObject backing, ulong address, ulong size, out string? firstRefusal)
    {
        if (MacMappedCopyState.Ledger is { } ledger)
            return ledger.CaptureOrdinaryWriteIdentity(backing, address, size, out firstRefusal);
        firstRefusal = "provider-ledger-unavailable";
        return null;
    }

    public long? OrdinaryWriteMappingEpoch => MacMappedCopyState.Ledger?.OrdinaryWriteMappingEpoch;

    public HostMappedCopyLedger.OrdinaryWriteIdentity? CaptureOrdinaryWriteIdentity(
        HostBackingObject backing, ulong address, ulong size, long mappingEpochBeforeAdmission, out bool mappingChanged)
    {
        mappingChanged = false;
        return MacMappedCopyState.Ledger?.CaptureOrdinaryWriteIdentity(
            backing, address, size, mappingEpochBeforeAdmission, out mappingChanged);
    }

    public CpuMappedPublicationPreparationResult TryPrepareMappedReadValidation(HostBackingObject backing,
        CpuMappedCopyObservation original, out CpuMappedReadValidation? validation)
    {
        validation = null;
        return MacMappedCopyState.Ledger is { } ledger
            ? ledger.TryPrepareMappedReadValidation(backing, original, out validation)
            : CpuMappedPublicationPreparationResult.Unavailable;
    }

    public CpuMappedPublicationPreparationResult TryPrepareMappedPublicationValidation(HostBackingObject backing,
        CpuMappedCopyObservation original, out CpuMappedPublicationValidation? validation)
    {
        validation = null;
        return MacMappedCopyState.Ledger is { } ledger
            ? ledger.TryPrepareMappedPublicationValidation(backing, original, out validation)
            : CpuMappedPublicationPreparationResult.Unavailable;
    }

    public CpuMappedPublicationPreparationResult TryPrepareMappedPublicationValidation(HostBackingObject backing,
        CpuMappedCopyObservation original, out CpuMappedPublicationValidation? validation,
        out CpuMappedPublicationValidationRefusal refusal)
    {
        validation = null; refusal = default;
        return MacMappedCopyState.Ledger is { } ledger
            ? ledger.TryPrepareMappedPublicationValidation(backing, original, out validation, out refusal)
            : CpuMappedPublicationPreparationResult.Unavailable;
    }

    public CpuMappedPublicationPreparationResult TryPrepareMappedPublication(HostBackingObject backing,
        CpuMappedCopyObservation original, IReadOnlyList<CpuMappedWriteRun> runs, out CpuMappedPublication? publication)
        => TryPrepareMappedPublication(backing, original, runs, out publication, out _);

    public CpuMappedPublicationPreparationResult TryPrepareMappedPublication(HostBackingObject backing,
        CpuMappedCopyObservation original, IReadOnlyList<CpuMappedWriteRun> runs, out CpuMappedPublication? publication,
        out CpuMappedPublicationPreparationRefusal refusal)
    {
        publication = null; refusal = default;
        return MacMappedCopyState.Ledger is { } ledger
            ? ledger.TryPrepareMappedPublication(backing, original, runs, out publication, out refusal)
            : CpuMappedPublicationPreparationResult.Unavailable;
    }

    public CpuMappedCopyResult TryReadMappedCopyWithWriteWatch(HostBackingObject backing, ulong footprintAddress,
        ulong footprintByteCount, ulong sourceAddress, Span<byte> destination, out CpuMappedCopyObservation? observation)
    {
        observation = null;
        return MacMappedCopyState.Ledger is { } ledger
            ? ledger.TryReadMappedCopyWithWriteWatch(backing, footprintAddress, footprintByteCount, sourceAddress, destination, out observation)
            : CpuMappedCopyResult.Unavailable;
    }
    public bool CanWaitForMappedReadWatchStatus => CanWaitForOrdinaryWriteAdmission &&
        MacMappedCopyState.Ledger is { } ledger && ledger.CanWaitForMappedReadWatchStatus;

    public bool CanWaitForMappedReadWatchEnrollment => CanWaitForOrdinaryWriteAdmission &&
        MacMappedCopyState.Ledger is { } ledger && ledger.CanWaitForMappedReadWatchEnrollment;

    public CpuMappedCopyResult TryReadMappedCopyWithWriteWatchAdmissionWait(HostBackingObject backing,
        ulong footprintAddress, ulong footprintByteCount, ulong sourceAddress, Span<byte> destination,
        Func<bool> originalStillCurrent, out CpuMappedCopyObservation? observation)
    {
        ArgumentNullException.ThrowIfNull(originalStillCurrent);
        observation = null;
        var ledger = MacMappedCopyState.Ledger;
        if (ledger is null) return CpuMappedCopyResult.Unavailable;
        bool WrapperCurrent() => ReferenceEquals(MacMappedCopyState.Ledger, ledger) &&
            CanWaitForMappedReadWatchEnrollment;
        bool OriginalWrapper() => WrapperCurrent() && originalStillCurrent() && WrapperCurrent();
        if (!OriginalWrapper()) return CpuMappedCopyResult.Rejected;
        return ledger.TryReadMappedCopyWithWriteWatchAdmissionWait(backing, footprintAddress,
            footprintByteCount, sourceAddress, destination, OriginalWrapper, out observation);
    }

    public bool IsMappedCopyWriteWatchCurrent(HostBackingObject backing, CpuMappedCopyObservation observation) =>
        MacMappedCopyState.Ledger?.IsMappedCopyWriteWatchCurrent(backing, observation) == true;
    public bool IsMappedCopyWriteWatchCurrent(HostBackingObject backing, CpuMappedCopyObservation observation,
        out CpuMappedReadWatchDiagnostic diagnostic)
    {
        if (MacMappedCopyState.Ledger is { } ledger) return ledger.IsMappedCopyWriteWatchCurrent(backing, observation, out diagnostic);
        diagnostic = new() { MappingGuard = "mac-ledger-unavailable", BooleanCurrent = false }; return false;
    }
    public CpuMappedReadWatchStatus GetMappedCopyWriteWatchStatus(HostBackingObject backing, CpuMappedCopyObservation observation) =>
        MacMappedCopyState.Ledger?.GetMappedCopyWriteWatchStatus(backing, observation) ?? CpuMappedReadWatchStatus.Stale;
    public CpuMappedReadWatchStatus GetMappedCopyWriteWatchStatus(HostBackingObject backing, CpuMappedCopyObservation observation,
        out CpuMappedReadWatchDiagnostic diagnostic)
    {
        if (MacMappedCopyState.Ledger is { } ledger) return ledger.GetMappedCopyWriteWatchStatus(backing, observation, out diagnostic);
        diagnostic = new() { MappingGuard = "mac-ledger-unavailable" }; return CpuMappedReadWatchStatus.Stale;
    }
    public void ReleaseMappedCopyWriteWatch(CpuMappedCopyObservation observation) =>
        MacMappedCopyState.Ledger?.ReleaseMappedCopyWriteWatch(observation);

    public CpuMappedCopyResult TryReadMappedCopy(HostBackingObject backing, ulong footprintAddress,
        ulong footprintByteCount, ulong sourceAddress, Span<byte> destination, out CpuMappedCopyObservation? observation)
    {
        observation = null;
        return MacMappedCopyState.Ledger is { } ledger
            ? ledger.TryReadMappedCopy(backing, footprintAddress, footprintByteCount, sourceAddress, destination, out observation)
            : CpuMappedCopyResult.Unavailable;
    }

    public bool IsMappedCopyObservationCurrent(HostBackingObject backing, CpuMappedCopyObservation observation) =>
        MacMappedCopyState.Ledger?.IsCurrent(backing, observation) == true;

    private const int PROT_NONE = 0x0;
    private const int PROT_READ = 0x1;
    private const int PROT_WRITE = 0x2;
    private const int PROT_EXEC = 0x4;
    private const int MAP_SHARED = 0x01;
    private const int MAP_PRIVATE = 0x02;
    private const int MAP_FIXED = 0x10;
    private const int MAP_ANON = 0x1000;

    private const int O_RDWR = 0x2;
    private const int O_CREAT = 0x200;
    private const int O_EXCL = 0x800;

    private const int KERN_SUCCESS = 0;

    // On Darwin fixed placement is the absence of VM_FLAGS_ANYWHERE. Never add
    // VM_FLAGS_OVERWRITE: an overlapping request must fail, not clobber.
    private const int VM_FLAGS_FIXED = 0;

    // sizeof(vm_region_submap_short_info_data_64_t) / sizeof(natural_t).
    private const int VM_REGION_SUBMAP_SHORT_INFO_COUNT_64 = 12;
    private static readonly nint MAP_FAILED = -1;
    private static int _sharedObjectCounter;

    // Keep the canonical host alias out of the guest application's fixed VA.
    // A kernel-chosen 16 GiB alias can otherwise cover a title's low arena
    // (ASTRO BOT maps its libc heap at 0x300000000) before that title maps it.
    private const ulong HostAliasStart = 0x100_0000_0000;

    public ulong PageSize { get; } = (ulong)Environment.SystemPageSize;

    public ulong Granularity => PageSize;

    public bool TryCreateBacking(ulong size, out HostBackingObject? backing, out HostViewFailure failure)
    {
        backing = null;
        failure = HostViewFailure.BackingUnavailable;
        if (size == 0 || size > long.MaxValue || size > ulong.MaxValue - (PageSize - 1))
        {
            return false;
        }

        lock (HostViewRegions.Gate)
        {
            if (MacMappedCopyState.Ledger is { } copyLedger && !copyLedger.CanMakeAnonymousPlacement()) return false;
            MacMappedCopyState.Ledger?.PreflightBackingPublication();
            if (UseChunkedBacking)
            {
                return TryCreateChunkedBackingLocked(size, out backing, out failure);
            }

            // Darwin has no memfd_create; shm_open plus an immediate shm_unlink
            // gives the same anonymous, descriptor-only backing object.
            var descriptor = OpenSharedObject();
            if (descriptor < 0)
            {
                return false;
            }

            if (ftruncate(descriptor, (long)size) != 0)
            {
                close(descriptor);
                return false;
            }

            var alias = FailAliasMapForTests ? MAP_FAILED : mmap(unchecked((nint)HostAliasStart), (nuint)size,
                PROT_READ | PROT_WRITE, MAP_SHARED, descriptor, 0);
            if (alias == MAP_FAILED)
            {
                close(descriptor);
                return false;
            }

            // mmap may decline a hint. Never keep a private alias in guest VA
            // if it chooses a different address; this successful mapping alone
            // is ours to release, with no fixed replacement of foreign memory.
            if ((ulong)alias < HostAliasStart)
            {
                if (munmap(alias, (nuint)size) != 0)
                {
                    MacMappedCopyState.Ledger?.QuarantineUnknownAlias();
                    throw new InvalidOperationException("Could not release a host backing alias placed in guest memory.");
                }
                close(descriptor);
                return false;
            }

            try
            {
                backing = new HostBackingObject((ulong)alias, size, PageSize, ReleaseOwnedBackingObject) { Descriptor = descriptor };
                MacMappedCopyState.Ledger?.RegisterBacking(backing);
            }
            catch (Exception error)
            {
                // Only this successful new alias is owned here; no guest view exists.
                var unmap = int.MinValue; var unmapError = 0;
                var closed = int.MinValue; var closeError = 0;
                Exception? cleanupFailure = null;
                try
                {
                    unmap = munmap(alias, (nuint)size);
                    unmapError = unmap == 0 ? 0 : Marshal.GetLastPInvokeError();
                    // No descriptor cleanup after an uncertain alias unmap.
                    if (unmap == 0)
                    {
                        closed = close(descriptor);
                        closeError = closed == 0 ? 0 : Marshal.GetLastPInvokeError();
                    }
                }
                catch (Exception secondary) { cleanupFailure = secondary; }
                if (unmap != 0) MacMappedCopyState.Ledger?.QuarantineUnknownAlias();
                if (unmap != 0 || closed != 0 || cleanupFailure is not null)
                    try
                    {
                        error.Data["BackingSetupCleanup"] = $"alias=0x{(ulong)alias:X};size={size};descriptor={descriptor};munmap={unmap}/{unmapError};close_attempted={unmap == 0};close={closed}/{closeError}";
                        if (cleanupFailure is not null) error.Data["BackingSetupCleanupException"] = cleanupFailure;
                    }
                    catch { } // primary construction/publication failure remains authoritative
                throw;
            }
            failure = HostViewFailure.None;
            return true;
        }
    }

    public ulong ReserveHole(ulong address, ulong size)
    {
        lock (HostViewRegions.Gate)
        {
            var result = ReserveHoleCore(address, size);
            if (result != 0)
                HostViewRegions.Replace(address, size, HostMemory.MEM_RESERVE, HostMemory.PAGE_NOACCESS);
            return result;
        }
    }

    private ulong ReserveHoleCore(ulong address, ulong size)
    {
        if (!HostViewMemory.IsValidRange(address, size) || address % Granularity != 0)
        {
            return 0;
        }

        if (MacMappedCopyState.Ledger is { } ledger && !ledger.BeforeNativeEffect(address, size)) return 0;

        // Darwin has no MAP_FIXED_NOREPLACE, and mmap's address argument is only
        // a hint there. On Apple Silicon the hint is ignored outright: every
        // anonymous mapping lands at the kernel's own allocation base no matter
        // what was asked for, so a "did it land where we asked?" test can never
        // succeed and every guest reservation would be reported as occupied.
        // mach_vm_allocate with fixed placement and without VM_FLAGS_OVERWRITE
        // maps exactly the requested range or fails with KERN_NO_SPACE, and
        // unlike MAP_FIXED it cannot clobber CLR, dyld, JIT, or Rosetta mappings
        // that are absent from our shadow table. It is the only reservation path
        // here, so a placement answer costs one syscall instead of a hint
        // attempt that is known to miss. (The Linux backend keeps the mmap path:
        // MAP_FIXED_NOREPLACE gives it the same exact-or-fail guarantee there.)
        return ReserveHoleDarwinFixed(address, size);
    }

    // Reserve exactly [address, address + size) or fail, without overwriting
    // anything already mapped there. mach_vm_allocate hands back pages with
    // VM_PROT_DEFAULT, so drop them to PROT_NONE to match the mmap path.
    private static ulong ReserveHoleDarwinFixed(ulong address, ulong size)
    {
        var allocated = address;
        if (mach_vm_allocate(mach_task_self(), ref allocated, size, VM_FLAGS_FIXED) != KERN_SUCCESS)
        {
            return 0;
        }

        if (allocated != address || mprotect((nint)address, (nuint)size, PROT_NONE) != 0)
        {
            mach_vm_deallocate(mach_task_self(), allocated, size);
            return 0;
        }

        return address;
    }

    // Find the free ranges by reserving halves until each attempt fits or falls
    // below the minimum. Occupied host ranges are few, so this needs few
    // attempts. Portable POSIX has no VirtualQuery for foreign host mappings;
    // where one exists, FindUnmappedRange below answers directly.
    public IReadOnlyList<HostAddressRange> ReserveFreeAddressRanges(ulong start, ulong end, ulong minimumSize)
    {
        var reserved = new List<HostAddressRange>();
        start = AlignUp(start, Granularity);
        end -= end % Granularity;
        if (start >= end || minimumSize == 0)
        {
            return reserved;
        }

        lock (HostViewRegions.Gate)
        {
            ReserveFreeAddressRangesLocked(start, end - start, Math.Max(minimumSize, Granularity), reserved);
        }

        return reserved;
    }

    private void ReserveFreeAddressRangesLocked(ulong address, ulong size, ulong minimumSize, List<HostAddressRange> reserved)
    {
        if (ReserveHoleCore(address, size) == address)
        {
            HostViewRegions.Replace(address, size, HostMemory.MEM_RESERVE, HostMemory.PAGE_NOACCESS);
            if (reserved.Count != 0 && reserved[^1].Address + reserved[^1].Size == address)
            {
                reserved[^1] = reserved[^1] with { Size = reserved[^1].Size + size };
            }
            else
            {
                reserved.Add(new HostAddressRange(address, size));
            }
            return;
        }

        var half = size / 2;
        half -= half % Granularity;
        if (half < minimumSize)
        {
            return;
        }

        ReserveFreeAddressRangesLocked(address, half, minimumSize, reserved);
        ReserveFreeAddressRangesLocked(address + half, size - half, minimumSize, reserved);
    }

    // Darwin does have a VirtualQuery equivalent. mach_vm_region_recurse walks
    // the task's own map including mappings this process never made, and asking
    // at nesting depth 0 collapses a whole submap into one entry. That matters
    // on Apple Silicon, where the kernel refuses fixed placement everywhere
    // below its own allocation base: the span from the dyld shared region up to
    // that base is hundreds of gigabytes, and a caller stepping one guest page
    // at a time needs tens of millions of probes to cross it. At depth 0 the
    // same crossing is a couple of dozen calls.
    public ulong FindUnmappedRange(ulong address, ulong size, ulong searchEnd)
    {
        if (size == 0 || address >= searchEnd || size > searchEnd - address)
        {
            return 0;
        }

        var task = mach_task_self();
        var info = stackalloc uint[VM_REGION_SUBMAP_SHORT_INFO_COUNT_64];
        var cursor = address;
        while (cursor < searchEnd && size <= searchEnd - cursor)
        {
            var regionAddress = cursor;
            ulong regionSize = 0;
            uint depth = 0;
            var count = VM_REGION_SUBMAP_SHORT_INFO_COUNT_64;
            if (mach_vm_region_recurse(task, ref regionAddress, ref regionSize, ref depth, info, ref count) != KERN_SUCCESS)
            {
                // Nothing is mapped between the cursor and the end of the map.
                return cursor;
            }

            // A region starting above the cursor leaves a hole in front of it.
            if (regionAddress > cursor && regionAddress - cursor >= size)
            {
                return cursor;
            }

            if (regionSize == 0 || regionSize > ulong.MaxValue - regionAddress)
            {
                return 0;
            }

            var regionEnd = regionAddress + regionSize;
            if (regionEnd <= cursor)
            {
                // The map did not move forward, so stop instead of spinning.
                return 0;
            }

            cursor = regionEnd;
        }

        return 0;
    }

    private static ulong AlignUp(ulong value, ulong alignment)
    {
        var remainder = value % alignment;
        return remainder == 0 ? value : value + alignment - remainder;
    }

    public bool SplitHole(ulong address, ulong size) => size != 0;

    public bool JoinHoles(ulong address, ulong size) => size != 0;

    public bool FreeHole(ulong address, ulong size) => FreeOwnedRange(address, size);

    public bool TryMapView(HostBackingObject backing, ulong address, ulong offset, ulong size, HostPageProtection protection, out HostViewFailure failure)
    {
        if (!backing.TryEnterAccess())
        { failure = HostViewFailure.BackingUnavailable; return false; }
        try
        {
            // Invalid arguments are refused with their own failure before the
            // ledger's pre-effect revocation, which reports only a bare false.
            if (address == 0 || address % PageSize != 0)
            { failure = HostViewFailure.WrongHostAddress; return false; }
            if (!HostViewMemory.IsValidOffset(backing, offset, size, PageSize))
            { failure = HostViewFailure.OffsetOutOfBounds; return false; }
            lock (HostViewRegions.Gate)
            {
                failure = HostViewFailure.WrongHostAddress;
                var ledger = MacMappedCopyState.Ledger;
                var prepared = ledger?.PrepareView(backing, address, size, offset, protection);
                if (ledger is not null && !ledger.BeforeNativeEffect(address, size)) return false;
                if (!TryMapViewCore(backing, address, offset, size, protection, out failure)) return false;
                HostViewRegions.Replace(address, size, HostMemory.MEM_COMMIT, HostViewRegions.RawProtection(protection));
                ledger?.PublishView(prepared);
                return true;
            }
        }
        finally { backing.ExitAccess(); }
    }

    private bool TryMapViewCore(HostBackingObject backing, ulong address, ulong offset, ulong size, HostPageProtection protection, out HostViewFailure failure)
    {
        if (address == 0 || address % PageSize != 0)
        {
            failure = HostViewFailure.WrongHostAddress;
            return false;
        }

        lock (backing.Gate)
        {
            if (backing.IsDisposed)
            {
                failure = HostViewFailure.BackingUnavailable;
                return false;
            }

            if (!HostViewMemory.IsValidOffset(backing, offset, size, PageSize))
            {
                failure = HostViewFailure.OffsetOutOfBounds;
                return false;
            }

            if (size > ulong.MaxValue - address)
            {
                failure = HostViewFailure.WrongHostAddress;
                return false;
            }

            if (backing.ChunkBytes != 0)
            {
                return TryMapChunkedViewCore(backing, address, offset, size, protection, out failure);
            }

            var ptr = mmap((nint)address, (nuint)size, GetNativeProtection(protection), MAP_SHARED | MAP_FIXED, backing.Descriptor, (long)offset);
            if (ptr == MAP_FAILED)
            {
                failure = HostViewFailure.FixedMapFailed;
                return false;
            }

            if ((ulong)ptr != address)
            {
                munmap(ptr, (nuint)size);
                failure = HostViewFailure.WrongHostAddress;
                return false;
            }

            failure = HostViewFailure.None;
            return true;
        }
    }

    public bool UnmapView(ulong address, ulong size) => RestoreHole(address, size);

    public bool CommitPrivate(ulong address, ulong size, HostPageProtection protection)
    {
        lock (HostViewRegions.Gate)
        {
            if (MacMappedCopyState.Ledger is { } ledger && !ledger.BeforeNativeEffect(address, size)) return false;
            if (mprotect((nint)address, (nuint)size, GetNativeProtection(protection)) != 0) return false;
            HostViewRegions.Replace(address, size, HostMemory.MEM_COMMIT, HostViewRegions.RawProtection(protection));
            return true;
        }
    }

    public bool ReleasePrivate(ulong address, ulong size) => RestoreHole(address, size);

    public bool ChangeAccess(ulong address, ulong size, HostPageProtection protection)
    {
        lock (HostViewRegions.Gate)
        {
            var ledger = MacMappedCopyState.Ledger;
            var prepared = ledger?.PrepareProtectionChange(address, size, protection);
            if (ledger is not null && !ledger.BeginProtectionChange(prepared, address, size)) return false;
            var complete = false;
            try
            {
                if (mprotect((nint)address, (nuint)size, GetNativeProtection(protection)) != 0) return false;
                HostViewRegions.ChangeProtection(address, size, HostViewRegions.RawProtection(protection));
                complete = true;
                return true;
            }
            finally { ledger?.CompleteProtectionChange(prepared, complete); }
        }
    }

    public bool ChangeDerivedAccess(ulong address, ulong size, GuestPageProtection guestAccess,
        HostPageProtection derivedHostAccess, bool ordinaryEnrollment)
        => ChangeDerivedAccessCore(address, size, guestAccess, derivedHostAccess, ordinaryEnrollment, out _, out _);

    // Exact same syscall/receipt path used by the public interface. Private
    // scalar outputs let separately authorized fixtures retain native outcomes;
    // no caller can supply a successful completion to this provider.
    private bool ChangeDerivedAccessCore(ulong address, ulong size, GuestPageProtection guestAccess,
        HostPageProtection derivedHostAccess, bool ordinaryEnrollment, out int nativeResult, out int nativeError)
    {
        nativeResult = int.MinValue; nativeError = 0;
        if (!HostMappedCopyLedger.IsValidDerivedProtection(guestAccess, derivedHostAccess)) return false;
        lock (HostViewRegions.Gate)
        {
            var ledger = MacMappedCopyState.Ledger;
            HostMappedCopyLedger.ProtectionChange? prepared = null;
            HostMappedCopyLedger.DerivedProtectionReceipt receipt = default;
            if (ledger is not null)
            {
                if (ordinaryEnrollment)
                {
                    prepared = ledger.PrepareProtectionChange(address, size, derivedHostAccess, guestAccess);
                    if (!ledger.BeginProtectionChange(prepared, address, size)) return false;
                }
                else if (!ledger.BeginDerivedProtection(address, size, guestAccess, derivedHostAccess, out receipt)) return false;
            }
            var complete = false;
            try
            {
                nativeResult = mprotect((nint)address, (nuint)size, GetNativeProtection(derivedHostAccess));
                nativeError = nativeResult == 0 ? 0 : Marshal.GetLastPInvokeError(); // before any helper/call
                if (nativeResult != 0) return false;
                HostViewRegions.ChangeProtection(address, size, HostViewRegions.RawProtection(derivedHostAccess));
                complete = true;
                return true;
            }
            finally
            {
                if (ordinaryEnrollment) ledger?.CompleteProtectionChange(prepared, complete);
                else ledger?.CompleteDerivedProtection(receipt, complete);
            }
        }
    }

    // Private ordinary-fixture control only: real Mach maximum protection,
    // never an externally applied mutation that bypasses the mapping ledger.
    // No production caller, signal callback, or public fake-outcome hook.
    private bool SetReadonlyMaximumForControl(ulong address, ulong size, GuestPageProtection guestAccess, out int nativeResult)
    {
        nativeResult = int.MinValue;
        lock (HostViewRegions.Gate)
        {
            var ledger = MacMappedCopyState.Ledger;
            if (ledger is null || address % ledger.NativePageSize != 0 || size == 0 || size % ledger.NativePageSize != 0 ||
                !ledger.BeginDerivedProtection(address, size, guestAccess, HostPageProtection.ReadOnly, out var receipt)) return false;
            var complete = false;
            try
            {
                nativeResult = mach_vm_protect(mach_task_self(), address, size, 1, PROT_READ);
                complete = nativeResult == KERN_SUCCESS;
                return complete;
            }
            finally { ledger.CompleteDerivedProtection(receipt, complete); }
        }
    }

    public bool FreeOwnedRange(ulong address, ulong size)
    {
        lock (HostViewRegions.Gate)
        {
            if (MacMappedCopyState.Ledger is { } ledger && !ledger.BeforeNativeEffect(address, size)) return false;
            if (munmap((nint)address, (nuint)size) != 0) return false;
            HostViewRegions.Replace(address, size, HostMemory.MEM_FREE_STATE, HostMemory.PAGE_NOACCESS);
            return true;
        }
    }

    private static bool RestoreHole(ulong address, ulong size)
    {
        lock (HostViewRegions.Gate)
        {
            if (MacMappedCopyState.Ledger is { } ledger && !ledger.BeforeNativeEffect(address, size)) return false;
            var ptr = mmap((nint)address, (nuint)size, PROT_NONE, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
            if (ptr == MAP_FAILED || (ulong)ptr != address) return false;
            HostViewRegions.Replace(address, size, HostMemory.MEM_RESERVE, HostMemory.PAGE_NOACCESS);
            return true;
        }
    }

    private static int OpenSharedObject()
    {
        var name = $"/sharpemu-{Environment.ProcessId}-{Interlocked.Increment(ref _sharedObjectCounter)}";
        var descriptor = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0x180);
        if (descriptor >= 0)
        {
            shm_unlink(name);
        }

        return descriptor;
    }

    private static HostBackingReleaseResult ReleaseOwnedBackingObject(HostBackingObject backing)
    {
        // Called once, after balanced ordinary accesses drain, outside provider
        // gates. This owns only alias+descriptor; existing guest mappings survive
        // close and need their own mapping/permission protocol.
        var ledger = MacMappedCopyState.Ledger;
        lock (HostViewRegions.Gate)
            if (ledger is not null && !ledger.BeginBackingRelease(backing))
                throw new InvalidOperationException("Backing release has no exact canonical alias owner; no native effect was attempted.");
        var unmap = munmap((nint)backing.AliasBase, (nuint)backing.NativeAliasSize);
        var unmapError = unmap == 0 ? 0 : Marshal.GetLastPInvokeError();
        lock (HostViewRegions.Gate) ledger?.CompleteBackingAliasRelease(backing, unmap == 0);
        if (unmap != 0)
        {
            var error = unmapError;
            // Do not assume a failed unmap is wholly unchanged, and do not close
            // the descriptor while alias ownership remains uncertain.
            return new(HostBackingReleaseStatus.AliasOutcomeUnknown, error);
        }
        // A chunked backing has no descriptor: its chunks die with their last mapping.
        if (backing.Descriptor >= 0 && close(backing.Descriptor) != 0)
        {
            var error = Marshal.GetLastPInvokeError();
            // Retrying close can destroy an unrelated reused fd (including EINTR).
            return new(HostBackingReleaseStatus.StorageOutcomeUnknown, error);
        }
        return new(HostBackingReleaseStatus.Released);
    }

    private static int GetNativeProtection(HostPageProtection protection) => protection switch
    {
        HostPageProtection.NoAccess => PROT_NONE,
        HostPageProtection.ReadOnly => PROT_READ,
        HostPageProtection.ReadWrite => PROT_READ | PROT_WRITE,
        HostPageProtection.Execute => PROT_EXEC,
        HostPageProtection.ReadExecute => PROT_READ | PROT_EXEC,
        HostPageProtection.ReadWriteExecute => PROT_READ | PROT_WRITE | PROT_EXEC,
        HostPageProtection.ExecuteWriteCopy => PROT_READ | PROT_WRITE | PROT_EXEC,
        _ => throw new ArgumentOutOfRangeException(nameof(protection), protection, null),
    };

    [LibraryImport("libc", SetLastError = true)]
    private static partial nint mmap(nint address, nuint length, int protection, int flags, int descriptor, long offset);

    [LibraryImport("libc", SetLastError = true)]
    private static partial int munmap(nint address, nuint length);

    [LibraryImport("libc", SetLastError = true)]
    private static partial int mprotect(nint address, nuint length, int protection);

    [LibraryImport("libc", SetLastError = true)]
    private static partial int ftruncate(int descriptor, long length);

    [LibraryImport("libc", SetLastError = true)]
    private static partial int close(int descriptor);

    [LibraryImport("libc", SetLastError = true, StringMarshalling = StringMarshalling.Utf8)]
    private static partial int shm_open(string name, int flags, int mode);

    [LibraryImport("libc", SetLastError = true, StringMarshalling = StringMarshalling.Utf8)]
    private static partial int shm_unlink(string name);

    [LibraryImport("libSystem.B.dylib")]
    private static partial uint mach_task_self();

    [LibraryImport("libSystem.B.dylib")]
    private static partial int mach_vm_allocate(uint task, ref ulong address, ulong size, int flags);

    [LibraryImport("libSystem.B.dylib")]
    private static partial int mach_vm_protect(uint task, ulong address, ulong size, int setMaximum, int protection);

    [LibraryImport("libSystem.B.dylib")]
    private static partial int mach_vm_deallocate(uint task, ulong address, ulong size);

    [LibraryImport("libSystem.B.dylib")]
    private static partial int mach_vm_region_recurse(
        uint task, ref ulong address, ref ulong size, ref uint nestingDepth, uint* info, ref int infoCount);
}
