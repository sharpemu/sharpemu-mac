// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.HLE.GuestMemory;

public sealed unsafe partial class SharedBackingViews
{
    // Reserve physical ranges so aliases cannot bypass overlap checks.
    private readonly record struct CopyReservation(ulong Source, ulong Destination, ulong Size)
    {
        public bool Overlaps(ulong offset, ulong size) => Size != 0 && size != 0 &&
            ((offset < Source + Size && Source < offset + size) ||
             (offset < Destination + Size && Destination < offset + size));
    }

    // Reserves the backing ranges of a copy inside one source and one destination view so it
    // can run outside the metadata lock. The lease keeps the alias alive and must be disposed.
    internal bool TryReserveCopy(ulong destination, ulong source, ulong size, out CopyLease lease)
    {
        lease = default;
        if (size == 0 || size > int.MaxValue) return false;
        int index;
        ulong sourceOffset, destinationOffset;
        lock (_lock)
            while (true)
            {
                if (_mappingWaiters != 0)
                {
                    WaitForCopyCompletion();
                    continue;
                }

                if (!IsAvailable || !TryFindRecord(source, size, out var sourceRecord) ||
                    !TryFindRecord(destination, size, out var destinationRecord)) return false;
                sourceOffset = sourceRecord.Offset + source - sourceRecord.Address;
                destinationOffset = destinationRecord.Offset + destination - destinationRecord.Address;
                if (!IsWithinBacking(sourceOffset, size) || !IsWithinBacking(destinationOffset, size)) return false;
                if (WaitForCopyConflict(sourceOffset, size, destinationOffset, size)) continue;
                // A released single-copy chunk is restored by the locked path's hooks; this one
                // runs none. The mark and a reservation are both taken under the lock, and
                // neither is taken over the other (TryMarkSingleCopyReleased), so a reserved
                // range never holds a released chunk.
                if (_singleCopy is { } singleCopy &&
                    (singleCopy.AnyReleased(sourceOffset, size) || singleCopy.AnyReleased(destinationOffset, size))) return false;
                if (!TryEnterLifetimeAccess()) return false;

                _copyReservations ??= new CopyReservation[4];
                index = 0;
                while (index < _copyReservations.Length && _copyReservations[index].Size != 0) index++;
                if (index == _copyReservations.Length) Array.Resize(ref _copyReservations, checked(index * 2));
                // Close admission before checking access leases. The exchange prevents reordering.
                if (_activeCopies == 0) Interlocked.Exchange(ref _copyAccessBlocked, 1);
                _copyReservations[index] = new CopyReservation(sourceOffset, destinationOffset, size);
                _activeCopies++;
                break;
            }

        // Alias accesses admitted before the gate closed touch bytes without checking the
        // reservation; let them finish. Waiting outside the lock keeps every locked path (and
        // a lifetime-pinned receipt retrying through one) free to make progress meanwhile;
        // those see the reservation already and wait only if they overlap it.
        var spin = new SpinWait();
        while (ActiveAliasAccesses() != 0) spin.SpinOnce();
        lease = new CopyLease(this, index, AliasBase + sourceOffset, AliasBase + destinationOffset);
        return true;
    }

    private bool WaitForBufferCopyConflict(ulong backingOffset, ReadOnlySpan<byte> buffer)
    {
        fixed (byte* bytes = buffer)
        {
            var address = (ulong)bytes;
            var size = (ulong)buffer.Length;
            if (address >= AliasBase && IsWithinBacking(address - AliasBase, size))
            {
                return WaitForCopyConflict(backingOffset, size, address - AliasBase, size);
            }

            if (WaitForCopyConflict(backingOffset, size)) return true;

            // A caller buffer can span views with different backing offsets.
            var end = address + size;
            var index = Math.Max(0, RangeSearch.FindLastIndexAtOrBelow(_views, address));
            for (; index < _views.Count; index++)
            {
                var record = _views.Values[index];
                if (record.Address >= end) break;
                var start = Math.Max(address, record.Address);
                var partEnd = Math.Min(end, record.Address + record.Size);
                if (start < partEnd && WaitForCopyConflict(record.Offset + start - record.Address, partEnd - start)) return true;
            }

            return false;
        }
    }

    // A wait releases the metadata lock. Callers must resolve mappings again afterward.
    private bool WaitForCopyConflict(ulong firstOffset, ulong firstSize, ulong secondOffset = 0, ulong secondSize = 0)
    {
        if (_activeCopies == 0) return false;
        foreach (var reservation in _copyReservations!)
        {
            if (reservation.Overlaps(firstOffset, firstSize) || reservation.Overlaps(secondOffset, secondSize))
            {
                WaitForCopyCompletion();
                return true;
            }
        }
        return false;
    }

    // Under the lock: whether an unlocked copy has reserved any of [offset, offset + size).
    private bool OverlapsCopyReservation(ulong offset, ulong size)
    {
        if (_activeCopies == 0) return false;
        foreach (var reservation in _copyReservations!)
        {
            if (reservation.Overlaps(offset, size)) return true;
        }

        return false;
    }

    private bool WaitForAnyCopy()
    {
        if (_activeCopies == 0) return false;
        WaitForCopyCompletion();
        return true;
    }

    private void WaitForCopies()
    {
        if (_activeCopies == 0) return;
        _mappingWaiters++;
        try
        {
            while (_activeCopies != 0) WaitForCopyCompletion();
        }
        finally
        {
            _mappingWaiters--;
            if (_copyWaiters != 0) Monitor.PulseAll(_lock);
        }
    }

    private void WaitForCopyCompletion()
    {
        _copyWaiters++;
        try { Monitor.Wait(_lock); }
        finally { _copyWaiters--; }
    }

    private void ReleaseCopy(int index)
    {
        lock (_lock)
        {
            _copyReservations![index] = default;
            _activeCopies--;
            if (_activeCopies == 0) Volatile.Write(ref _copyAccessBlocked, 0);
            if (_copyWaiters != 0) Monitor.PulseAll(_lock);
        }
        ExitLifetimeAccess();
    }

    internal readonly ref struct CopyLease(SharedBackingViews owner, int index, ulong source, ulong destination)
    {
        public ulong Source { get; } = source;
        public ulong Destination { get; } = destination;

        public void Dispose() => owner.ReleaseCopy(index);
    }
}
