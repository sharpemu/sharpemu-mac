// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;

namespace SharpEmu.Core.Cpu;

public sealed class TrackedCpuMemory : ICpuMemory, ITrackedCpuMemory, IGuestMemoryAllocator, ICpuMemoryWrapper, ICpuGpuMemoryView, ICpuMappedCopySource, ICpuMappedReadWatchSource, ICpuMappedPublicationSource
{
    private readonly ICpuMemory _inner;

    public TrackedCpuMemory(ICpuMemory inner)
    {
        _inner = inner ?? throw new ArgumentNullException(nameof(inner));
    }

    public CpuMemoryAccessFailure? LastFailure { get; private set; }

    public ICpuMemory Inner => _inner;

    // GPU work must not capture a guest thread's LastFailure tracker. Every
    // GPU-visible operation/capability already forwards to this readonly view.
    public ICpuMemory GpuMemoryView => _inner;

    public string DescribeReadRange(ulong address, ulong size) => _inner.DescribeReadRange(address, size);

    public CpuMappedPublicationPreparationResult TryPrepareMappedReadValidation(CpuMappedCopyObservation original,
        out CpuMappedReadValidation? validation)
    {
        validation = null;
        return _inner is ICpuMappedReadWatchSource source
            ? source.TryPrepareMappedReadValidation(original, out validation)
            : CpuMappedPublicationPreparationResult.Unavailable;
    }

    public bool CanWaitForMappedPublicationValidation => _inner is ICpuMappedPublicationSource source && source.CanWaitForMappedPublicationValidation;

    public CpuMappedPublicationPreparationResult TryPrepareMappedPublicationValidation(CpuMappedCopyObservation original,
        out CpuMappedPublicationValidation? validation)
    {
        validation = null;
        return _inner is ICpuMappedPublicationSource source
            ? source.TryPrepareMappedPublicationValidation(original, out validation)
            : CpuMappedPublicationPreparationResult.Unavailable;
    }

    public CpuMappedPublicationPreparationResult TryPrepareMappedPublicationValidation(CpuMappedCopyObservation original,
        out CpuMappedPublicationValidation? validation,
        out CpuMappedPublicationValidationRefusal refusal)
    {
        validation = null; refusal = default;
        return _inner is ICpuMappedPublicationSource source
            ? source.TryPrepareMappedPublicationValidation(original, out validation, out refusal)
            : CpuMappedPublicationPreparationResult.Unavailable;
    }

    public CpuMappedPublicationPreparationResult TryPrepareMappedPublication(CpuMappedCopyObservation original,
        IReadOnlyList<CpuMappedWriteRun> runs, out CpuMappedPublication? publication)
        => TryPrepareMappedPublication(original, runs, out publication, out _);

    public CpuMappedPublicationPreparationResult TryPrepareMappedPublication(CpuMappedCopyObservation original,
        IReadOnlyList<CpuMappedWriteRun> runs, out CpuMappedPublication? publication,
        out CpuMappedPublicationPreparationRefusal refusal)
    {
        publication = null; refusal = default;
        return _inner is ICpuMappedPublicationSource source
            ? source.TryPrepareMappedPublication(original, runs, out publication, out refusal)
            : CpuMappedPublicationPreparationResult.Unavailable;
    }

    public CpuMappedCopyResult TryReadMappedCopyWithWriteWatch(ulong footprintAddress, ulong footprintByteCount,
        ulong sourceAddress, Span<byte> destination, out CpuMappedCopyObservation? observation)
    {
        observation = null;
        return _inner is ICpuMappedReadWatchSource source
            ? source.TryReadMappedCopyWithWriteWatch(footprintAddress, footprintByteCount, sourceAddress, destination, out observation)
            : CpuMappedCopyResult.Unavailable;
    }
    public bool CanWaitForMappedReadWatchStatus =>
        _inner is ICpuMappedReadWatchSource source && source.CanWaitForMappedReadWatchStatus;

    public bool CanWaitForMappedReadWatchEnrollment =>
        _inner is ICpuMappedReadWatchSource source && source.CanWaitForMappedReadWatchEnrollment;

    public CpuMappedCopyResult TryReadMappedCopyWithWriteWatchAdmissionWait(ulong footprintAddress,
        ulong footprintByteCount, ulong sourceAddress, Span<byte> destination,
        Func<bool> originalStillCurrent, out CpuMappedCopyObservation? observation)
    {
        ArgumentNullException.ThrowIfNull(originalStillCurrent);
        observation = null;
        if (_inner is not ICpuMappedReadWatchSource source) return CpuMappedCopyResult.Unavailable;
        if (!CanWaitForMappedReadWatchEnrollment) return CpuMappedCopyResult.Rejected;
        return source.TryReadMappedCopyWithWriteWatchAdmissionWait(footprintAddress, footprintByteCount,
            sourceAddress, destination, () => CanWaitForMappedReadWatchEnrollment && originalStillCurrent() &&
                CanWaitForMappedReadWatchEnrollment, out observation);
    }

    public bool IsMappedCopyWriteWatchCurrent(CpuMappedCopyObservation observation) =>
        _inner is ICpuMappedReadWatchSource source && source.IsMappedCopyWriteWatchCurrent(observation);
    public bool IsMappedCopyWriteWatchCurrent(CpuMappedCopyObservation observation,
        out CpuMappedReadWatchDiagnostic diagnostic)
    {
        if (_inner is ICpuMappedReadWatchSource source) return source.IsMappedCopyWriteWatchCurrent(observation, out diagnostic);
        diagnostic = new() { VirtualClosure = "tracked-inner-capability-unavailable", BooleanCurrent = false }; return false;
    }
    public CpuMappedReadWatchStatus GetMappedCopyWriteWatchStatus(CpuMappedCopyObservation observation) =>
        _inner is ICpuMappedReadWatchSource source ? source.GetMappedCopyWriteWatchStatus(observation) : CpuMappedReadWatchStatus.Stale;
    public CpuMappedReadWatchStatus GetMappedCopyWriteWatchStatus(CpuMappedCopyObservation observation,
        out CpuMappedReadWatchDiagnostic diagnostic)
    {
        if (_inner is ICpuMappedReadWatchSource source) return source.GetMappedCopyWriteWatchStatus(observation, out diagnostic);
        diagnostic = new() { VirtualClosure = "tracked-inner-capability-unavailable" }; return CpuMappedReadWatchStatus.Stale;
    }
    public void ReleaseMappedCopyWriteWatch(CpuMappedCopyObservation observation)
    { if (_inner is ICpuMappedReadWatchSource source) source.ReleaseMappedCopyWriteWatch(observation); }

    public CpuMappedCopyResult TryReadMappedCopy(ulong footprintAddress, ulong footprintByteCount,
        ulong sourceAddress, Span<byte> destination, out CpuMappedCopyObservation? observation)
    {
        observation = null;
        return _inner is ICpuMappedCopySource source
            ? source.TryReadMappedCopy(footprintAddress, footprintByteCount, sourceAddress, destination, out observation)
            : CpuMappedCopyResult.Unavailable;
    }
    public bool IsMappedCopyObservationCurrent(CpuMappedCopyObservation observation) =>
        _inner is ICpuMappedCopySource source && source.IsMappedCopyObservationCurrent(observation);

    public bool TryRead(ulong virtualAddress, Span<byte> destination)
    {
        var result = _inner.TryRead(virtualAddress, destination);
        if (!result)
        {
            LastFailure = new CpuMemoryAccessFailure(virtualAddress, destination.Length, isWrite: false);
        }

        return result;
    }

    public bool TryWrite(ulong virtualAddress, ReadOnlySpan<byte> source)
    {
        var result = _inner.TryWrite(virtualAddress, source);
        if (!result)
        {
            LastFailure = new CpuMemoryAccessFailure(virtualAddress, source.Length, isWrite: true);
        }

        return result;
    }

    public bool TryCompare(
        ulong virtualAddress,
        ReadOnlySpan<byte> expected,
        out bool equal) =>
        _inner.TryCompare(virtualAddress, expected, out equal);

    public bool TryCopy(ulong destinationAddress, ulong sourceAddress, ulong length) =>
        _inner.TryCopy(destinationAddress, sourceAddress, length);

    public bool TryFillInPlace<TState>(ulong virtualAddress, int length, TState state, GuestSpanFill<TState> fill, out int filled) =>
        _inner.TryFillInPlace(virtualAddress, length, state, fill, out filled);

    public bool CanRead(ulong address, ulong size) => _inner.CanRead(address, size);
    public long? ReadMappingGeneration => _inner.ReadMappingGeneration;

    public bool TryScanCString(ulong address, byte needle, bool findLast, ulong maxLength, out ulong match) =>
        _inner.TryScanCString(address, needle, findLast, maxLength, out match);

    public bool TryAllocateGuestMemory(ulong size, ulong alignment, out ulong address)
    {
        if (_inner is IGuestMemoryAllocator allocator)
        {
            return allocator.TryAllocateGuestMemory(size, alignment, out address);
        }

        address = 0;
        return false;
    }

    public bool TryFreeGuestMemory(ulong address)
    {
        return _inner is IGuestMemoryAllocator allocator && allocator.TryFreeGuestMemory(address);
    }
}
