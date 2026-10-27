// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using SharpEmu.GpuRemote;
using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.HLE.GuestMemory;
using SharpEmu.HLE.Host;

namespace SharpEmu.GpuSplit;

/// <summary>Where a <see cref="RemoteGuestAddressSpace"/> sends the page protections it is asked for.</summary>
internal interface IProtectionRequests
{
    /// <summary>Sends one request to the guest process and returns its number; 0 when it could not be sent.</summary>
    ulong Request(ulong address, ulong size, GuestPageProtection guestAccess, GuestPageProtection access, ProtectRequestFlags flags);

    /// <summary>The number of the last request sent.</summary>
    ulong LastRequested { get; }

    /// <summary>Waits until the guest process applied request <paramref name="sequence"/> and every one before it.</summary>
    bool WaitApplied(ulong sequence);
}

/// <summary>The GPU-interest page summary as the GPU process writes it (<see cref="SplitSharedState"/>).</summary>
internal interface IInterestSummary
{
    void SetPage(ulong address);

    void ClearPage(ulong address);
}

/// <summary>
/// Guest memory as the GPU process sees it, behind the interfaces the GPU backend is written
/// against (docs/macos/split-protocol.md, "Memory").
/// </summary>
/// <remarks>
/// <para><b>Bytes.</b> Reads and writes go through the mirror of the backing, found through the
/// shared view table (<see cref="GuestMirrorTranslator"/>): the mirror is always mapped whole, so
/// a range the guest process just unmapped reads stale bytes instead of faulting.
/// <see cref="IGuestBackedSpace.TryResolveBackingAlias"/> hands out mirror addresses (the GPU
/// windows are made over the mirror, <see cref="BackingAliasRange"/>), and
/// <see cref="TryResolveBackedHostView"/> replica addresses (guest address + K), after syncing the
/// replica so the views it names exist here. Memory the backing does not hold (module images, the
/// guest allocator arena, HLE data) is private to the guest process: every access to it fails, and
/// the first few are reported with the address.</para>
///
/// <para><b>Protections.</b> The guest process owns them. The page guard of this process derives
/// them as in one process and this space sends each one there; one that takes rights away waits
/// until it is applied, so nothing relies on a watch the guest can still write through. One that
/// gives rights back does not wait: the guest process's fault path waits for it instead (see
/// <see cref="SplitGpuHost"/>). The guard's watch changes keep the GPU-interest summary.</para>
///
/// <para>The GPU side never allocates or maps guest memory: those members refuse.</para>
/// </remarks>
internal sealed unsafe class RemoteGuestAddressSpace :
    ICpuMemory, IGuestBackedSpace, IGuestAddressSpace, IGuestDerivedProtectionAddressSpace, IPageWatchObserver,
    IProtectionBatchSpace, IPendingWritebackSource
{
    private const int MaxMissReports = 16;

    private readonly ulong _mirrorBase;
    private readonly ulong _mirrorBytes;
    private readonly GuestMirrorTranslator _translator;
    private readonly GuestReplica? _replica;
    private readonly IProtectionRequests _protections;
    private readonly IInterestSummary _summary;
    private int _missReports;
    private long _misses;

    public RemoteGuestAddressSpace(HostBackingObject mirror, GuestMirrorTranslator translator, GuestReplica? replica,
        ulong chunkBytes, ulong objectBytes, IProtectionRequests protections, IInterestSummary summary)
    {
        ArgumentNullException.ThrowIfNull(mirror);
        _mirrorBase = mirror.AliasBase;
        _mirrorBytes = Math.Min(mirror.Size, translator.MirrorBytes);
        _translator = translator;
        _replica = replica;
        BackingChunkBytes = chunkBytes;
        BackingObjectBytes = chunkBytes == 0 ? 0 : objectBytes;
        _protections = protections;
        _summary = summary;
    }

    /// <summary>The pending-writeback summary the GPU caches keep for the guest process (null: none is kept).</summary>
    public IPendingWritebackPages? PendingWritebacks { get; set; }

    /// <summary>Accesses to memory the backing does not hold.</summary>
    public long PrivateMemoryMisses => Interlocked.Read(ref _misses);

    // ---- ICpuMemory ----

    public bool TryRead(ulong virtualAddress, Span<byte> destination)
    {
        if (destination.IsEmpty)
        {
            return true;
        }

        fixed (byte* target = destination)
        {
            return Copy(virtualAddress, (ulong)destination.Length, target, toGuest: false);
        }
    }

    // A write the GPU side makes (command-processor stores, labels) is a managed write to this
    // process's GPU memory manager, exactly as a write through the guest address space is in one
    // process: the caches hear of it before and after it lands.
    public bool TryWrite(ulong virtualAddress, ReadOnlySpan<byte> source)
    {
        if (source.IsEmpty)
        {
            return true;
        }

        var size = (ulong)source.Length;
        if (!IsBackedRange(virtualAddress, size))
        {
            NoteMiss(virtualAddress, size, "write");
            return false;
        }

        GuestGpuMemoryHook.MarkCpuWrite(virtualAddress, size);
        bool written;
        fixed (byte* bytes = source)
        {
            written = Copy(virtualAddress, size, bytes, toGuest: true);
        }

        GuestGpuMemoryHook.MarkCpuWriteLanded(virtualAddress, size);
        return written;
    }

    public bool TryCompare(ulong virtualAddress, ReadOnlySpan<byte> expected, out bool equal)
    {
        equal = false;
        if (expected.IsEmpty)
        {
            equal = true;
            return true;
        }

        var size = (ulong)expected.Length;
        var done = 0UL;
        while (done < size)
        {
            if (!TryPiece(virtualAddress + done, size - done, out var pointer, out var take))
            {
                NoteMiss(virtualAddress, size, "compare");
                return false;
            }

            if (!new ReadOnlySpan<byte>((void*)pointer, (int)take).SequenceEqual(expected.Slice((int)done, (int)take)))
            {
                return true;
            }

            done += take;
        }

        equal = true;
        return true;
    }

    public bool CanRead(ulong address, ulong size) => IsBackedRange(address, size);

    public string DescribeReadRange(ulong address, ulong size) =>
        IsBackedRange(address, size)
            ? $"GPU process: guest 0x{address:X16}+0x{size:X} is in the shared backing."
            : $"GPU process: guest 0x{address:X16}+0x{size:X} is not in the shared backing " +
              "(memory private to the guest process: module images, the guest allocator arena or HLE data).";

    // ---- IGuestBackedSpace ----

    public bool TryHoldRange(ulong address, ulong size) => false;

    public bool TryHoldRangeAtOrAbove(ulong searchStart, ulong size, ulong alignment, out ulong address)
    {
        address = 0;
        return false;
    }

    public bool TryMapBacked(ulong address, ulong size, ulong backingOffset, GuestPageProtection protection, out HostViewFailure failure)
    {
        failure = default;
        return false;
    }

    public bool TryUnmapBacked(ulong address, ulong size) => false;

    public bool TryClearBacking(ulong offset, ulong size) => false;

    public ulong BackingChunkBytes { get; }

    public ulong BackingObjectBytes { get; }

    public bool IsBackedView(ulong address) => _translator.TryGetOffset(address, out _, out _);

    public bool IsBackedRange(ulong address, ulong size)
    {
        if (size == 0 || size > ulong.MaxValue - address)
        {
            return false;
        }

        var done = 0UL;
        while (done < size)
        {
            if (!TryPiece(address + done, size - done, out _, out var take))
            {
                return false;
            }

            done += take;
        }

        return true;
    }

    public bool TryWriteBacking(ulong address, ReadOnlySpan<byte> data)
    {
        if (data.IsEmpty)
        {
            return true;
        }

        fixed (byte* bytes = data)
        {
            return Copy(address, (ulong)data.Length, bytes, toGuest: true);
        }
    }

    public bool TryReadBacking(ulong address, Span<byte> data)
    {
        if (data.IsEmpty)
        {
            return true;
        }

        fixed (byte* bytes = data)
        {
            return Copy(address, (ulong)data.Length, bytes, toGuest: false);
        }
    }

    // The mirror is mapped for the life of the process; chunks the guest process releases read
    // zero through it, as through its own alias.
    public bool TryEnterBackingAliasAccess() => true;

    public void ExitBackingAliasAccess()
    {
    }

    public bool TryResolveBackingAlias(ulong address, ulong size, out ulong alias) =>
        _translator.TryTranslate(address, size, out alias);

    public bool TryResolveBackingAliasSpan(ulong address, ulong size, out ulong alias) =>
        _translator.TryTranslate(address, size, out alias);

    public bool TryResolveBackedHostView(ulong address, ulong size, out ulong host)
    {
        host = 0;
        if (_replica is not { } replica || !replica.Covers(address, size) || !IsBackedRange(address, size))
        {
            return false;
        }

        // The table above is the guest process's live one; the replica follows its log only at
        // sync points. Syncing here makes every view the check found exist at its address + K.
        replica.Sync();
        if (!replica.IsComplete)
        {
            return false;
        }

        host = replica.HostAddress(address);
        return true;
    }

    public (ulong Base, ulong Size) BackingAliasRange => (_mirrorBase, _mirrorBytes);

    // ---- IGuestAddressSpace ----

    // Protections are applied in the guest process, under Rosetta: 4 KiB pages, whatever this
    // process's own page is.
    public ulong ProtectionPageSize => TrackerLayout.PageBytes;

    public ulong AllocateAt(ulong desiredAddress, ulong size, bool executable = true, bool allowAlternative = true) =>
        throw new NotSupportedException("The GPU process never allocates guest memory.");

    public bool TryBackFixedRange(ulong address, ulong size, bool executable) => false;

    public bool TryAllocateAtOrAbove(ulong desiredAddress, ulong size, bool executable, ulong alignment, out ulong actualAddress)
    {
        actualAddress = 0;
        return false;
    }

    public bool TryEnsureRangeCommitted(ulong address, ulong size) => false;

    public bool TryAllocateGuestMemory(ulong size, ulong alignment, out ulong address)
    {
        address = 0;
        return false;
    }

    public bool TryFreeGuestMemory(ulong address) => false;

    public bool TryProtect(ulong address, ulong size, GuestPageProtection protection)
    {
        // Without the guest's own protection it cannot tell taking rights from giving them back.
        var sequence = _protections.Request(address, size, protection, protection, ProtectRequestFlags.None);
        if (sequence != 0)
        {
            _ = WaitAppliedTimed(sequence);
        }

        return true;
    }

    public bool TryApplyDerivedProtection(ulong address, ulong size, GuestPageProtection guestAccess,
        GuestPageProtection derivedAccess, bool ordinaryEnrollment)
    {
        if ((derivedAccess & ~guestAccess) != 0)
        {
            return false;
        }

        var flags = ProtectRequestFlags.Derived | (ordinaryEnrollment ? ProtectRequestFlags.OrdinaryEnrollment : 0);
        var sequence = _protections.Request(address, size, guestAccess, derivedAccess, flags);
        // A watch's protection takes rights from the guest: nothing may rely on the watch until the
        // guest process applied it. A failed wait is reported by the requests; returning false
        // here would only make the page guard fail the whole process.
        if (sequence != 0 && derivedAccess != guestAccess)
        {
            if (t_batchDepth != 0 && BatchingEnabled)
            {
                // Inside a watch change: the guard waits once, after its block locks are released.
                // (Requests are applied in order, so the last one's number covers the rest.)
                if (sequence > t_batchSequence) t_batchSequence = sequence;
                RaisePending(sequence);
            }
            else
            {
                _ = WaitAppliedTimed(sequence);
            }
        }

        return true;
    }

    // ---- IProtectionBatchSpace ----

    // SHARPEMU_SPLIT_BATCH_PROTECT=0 waits for every protection where it is made, as before.
    private static readonly bool BatchingEnabled = !string.Equals(
        Environment.GetEnvironmentVariable("SHARPEMU_SPLIT_BATCH_PROTECT"), "0", StringComparison.Ordinal);

    // The depth of the watch changes this thread is inside, and the highest protection number it
    // left unconfirmed in them.
    [ThreadStatic]
    private static int t_batchDepth;

    [ThreadStatic]
    private static ulong t_batchSequence;

    // The highest number any thread left unconfirmed in a batch. A thread whose watch change found
    // the pages already watched by another (whose protection that thread has not confirmed yet) sent
    // nothing of its own, but must not return before that protection is applied: the old wait under
    // the block lock held it back, so it waits for this number as well.
    private long _pendingSequence;

    public void BeginProtectionBatch() => t_batchDepth++;

    public void EndProtectionBatch()
    {
        if (--t_batchDepth != 0)
        {
            return;
        }

        var own = t_batchSequence;
        t_batchSequence = 0;
        var target = Math.Max(own, (ulong)Interlocked.Read(ref _pendingSequence));
        if (target == 0)
        {
            return;
        }

        _ = WaitAppliedTimed(target);
        // Applied (or the wait failed and was reported): nothing is pending up to target any more.
        var pending = Interlocked.Read(ref _pendingSequence);
        while ((ulong)pending <= target && pending != 0 &&
               Interlocked.CompareExchange(ref _pendingSequence, 0, pending) != pending)
        {
            pending = Interlocked.Read(ref _pendingSequence);
        }
    }

    private void RaisePending(ulong sequence)
    {
        var current = Interlocked.Read(ref _pendingSequence);
        while ((ulong)current < sequence)
        {
            var observed = Interlocked.CompareExchange(ref _pendingSequence, (long)sequence, current);
            if (observed == current)
            {
                break;
            }

            current = observed;
        }
    }

    private bool WaitAppliedTimed(ulong sequence)
    {
        if (!SplitCrossingStats.Enabled)
        {
            return _protections.WaitApplied(sequence);
        }

        var start = Stopwatch.GetTimestamp();
        var applied = _protections.WaitApplied(sequence);
        SplitCrossingStats.Count(SplitCrossing.ProtectWait);
        SplitCrossingStats.Time(SplitCrossing.ProtectWait, Stopwatch.GetTimestamp() - start);
        return applied;
    }

    // ---- IPageWatchObserver ----

    // Called once per page whose watched state changes, under the page guard's block lock: every
    // page of a block (and every bit of its summary words) changes under that one lock.
    public void OnPageWatchChanged(ulong page, bool watched)
    {
        if (page >= TrackerLayout.SpaceBytes)
        {
            return;
        }

        if (watched)
        {
            _summary.SetPage(page);
        }
        else
        {
            _summary.ClearPage(page);
        }
    }

    // ---- pieces ----

    // The mirror address of the run holding address, and how much of [address, address + left) it covers.
    private bool TryPiece(ulong address, ulong left, out ulong pointer, out ulong take)
    {
        pointer = take = 0;
        if (!_translator.TryGetOffset(address, out var offset, out var contiguous) || contiguous == 0 || offset >= _mirrorBytes)
        {
            return false;
        }

        take = Math.Min(Math.Min(contiguous, left), _mirrorBytes - offset);
        pointer = _mirrorBase + offset;
        return true;
    }

    private bool Copy(ulong address, ulong size, byte* bytes, bool toGuest)
    {
        if (size > ulong.MaxValue - address)
        {
            return false;
        }

        var done = 0UL;
        while (done < size)
        {
            if (!TryPiece(address + done, size - done, out var pointer, out var take))
            {
                NoteMiss(address, size, toGuest ? "write" : "read");
                return false;
            }

            if (toGuest)
            {
                Buffer.MemoryCopy(bytes + done, (void*)pointer, take, take);
            }
            else
            {
                Buffer.MemoryCopy((void*)pointer, bytes + done, take, take);
            }

            done += take;
        }

        return true;
    }

    private void NoteMiss(ulong address, ulong size, string access)
    {
        Interlocked.Increment(ref _misses);
        if (Interlocked.Increment(ref _missReports) <= MaxMissReports)
        {
            Console.Error.WriteLine(
                $"[GPU-SPLIT][WARN] GPU {access} of guest 0x{address:X16}+0x{size:X} failed: not in the shared backing " +
                "(memory private to the guest process).");
        }
    }
}
