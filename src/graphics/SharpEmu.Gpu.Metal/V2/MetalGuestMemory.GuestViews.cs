// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// Guest views: no-copy buffers over the guest's own mapping of a range, for reads of a
/// range no window covers.
///
/// <para>
/// A window is a slice of the canonical alias, so it can hold only a range that is one run
/// of the backing. A title that maps memory a page at a time from scattered backing offsets
/// (Demon's Souls BatchMaps 64 KiB pages) makes ranges that are contiguous to the guest and
/// to the host - the backed views are mapped at their guest addresses - but not in the
/// backing, and every such binding used to be copied: into the ring per bind, or into the
/// staged-range cache, which grew to half a gigabyte. A buffer over the guest mapping reads
/// the same physical pages the windows map, in guest order, with no copy and no page watch:
/// what the GPU reads is guest memory when the work runs, as through a window.
/// </para>
///
/// <para>
/// It is a second GPU address for bytes a window also maps, and two buffers over the same
/// pages are not coherent inside one command buffer (a write through one is not seen by a
/// read through the other, fence or not). Across command buffers they are. So a guest view is
/// only ever read, and the host binds one only when no write to its bytes has been recorded
/// in the command buffer being recorded (MetalRenderHost.StagedStorage.cs); a write recorded
/// after the bind is ordered after the read, and a later bind of the same bytes in the same
/// command buffer is staged instead. A written binding is never a guest view.
/// </para>
///
/// <para>
/// A view keeps the pages it was made over: measured on an M4, a buffer over a mapping that
/// was replaced afterwards still read the old pages. So a view is dropped as soon as the
/// guest unmaps any of its range (<see cref="OnGuestUnmap"/>), and its buffer is released
/// once the submissions that bound it have completed. Making a view over a PROT_NONE page
/// fails; the binding is then staged as before. Views are cached by range and reused by any
/// binding inside one, named only by the submissions that bind them, released after
/// <see cref="IdleWindowSerials"/> idle submissions, and, past a budget
/// (SHARPEMU_METAL_V2_GUEST_VIEW_MB, default 1024), least recently used first.
/// SHARPEMU_METAL_V2_GUEST_VIEWS=0 turns them off.
/// </para>
/// </summary>
internal sealed partial class MetalGuestMemory
{
    /// <summary>The smallest range bound through a view; below it a ring copy is cheaper than a buffer.</summary>
    internal const ulong GuestViewMinBytes = 256 * 1024;

    internal const ulong DefaultGuestViewBudgetBytes = 1024UL * 1024 * 1024;

    // A no-copy buffer starts and ends on a host page.
    private const ulong GuestViewPageBytes = 0x4000;

    // Views are trimmed this often, and whenever they are over budget.
    private const ulong GuestViewTrimSerials = 64;

    internal static readonly bool GuestViewsEnabled =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_GUEST_VIEWS") != "0";

    private static readonly ulong GuestViewBudgetBytes =
        ulong.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_GUEST_VIEW_MB"), out var megabytes) &&
        megabytes != 0
            ? megabytes * 1024 * 1024
            : DefaultGuestViewBudgetBytes;

    private sealed class GuestView
    {
        public ulong Start;
        public ulong End;
        public nint Buffer;
        public ulong GpuAddress;
        public ulong LastUseSerial;
        public bool Touched;

        public ulong Size => End - Start;
    }

    private readonly object _viewLock = new();
    private readonly Dictionary<(ulong Start, ulong End), GuestView> _views = new();
    private readonly List<(nint Buffer, ulong Bytes, ulong Serial)> _retiredViews = new();

    // The views the submission being recorded bound: worker only.
    private readonly List<GuestView> _touchedViews = new();
    private int _viewCount;
    private ulong _viewBytes;
    private ulong _viewTrimSerial;
    private long _viewsCreated;
    private long _viewsReleased;
    private long _viewBinds;
    private long _viewBindBytes;
    private long _viewInvalidations;
    private long _viewCreateFailures;
    private long _viewsUnbacked;

    /// <summary>
    /// Guest views alive and the bytes they map, created and released since startup, binds
    /// and the bytes bound through them, views dropped by an unmap, buffers the device
    /// refused, and ranges that were not wholly backed views.
    /// </summary>
    public (int Alive, ulong Bytes, long Created, long Released, long Binds, long BindBytes, long Invalidated,
        long CreateFailures, long Unbacked) GuestViewCounts
    {
        get
        {
            lock (_viewLock)
            {
                return (_views.Count, _viewBytes, _viewsCreated, _viewsReleased, Volatile.Read(ref _viewBinds),
                    Volatile.Read(ref _viewBindBytes), _viewInvalidations, _viewCreateFailures, _viewsUnbacked);
            }
        }
    }

    /// <summary>True when the buffer is a guest view's.</summary>
    public bool IsGuestView(nint buffer)
    {
        if (buffer == 0 || Volatile.Read(ref _viewCount) == 0)
        {
            return false;
        }

        lock (_viewLock)
        {
            foreach (var view in _views.Values)
            {
                if (view.Buffer == buffer)
                {
                    return true;
                }
            }

            return false;
        }
    }

    /// <summary>Whether a range of this size is bound through a guest view at all.</summary>
    public bool MayUseGuestView(ulong size) =>
        GuestViewsEnabled && !StageEverything && size >= GuestViewMinBytes && size <= _maxWindowBytes;

    /// <summary>
    /// A read-only binding of a range no window covers, over the guest's own mapping of it;
    /// false when the range is not wholly backed views or the device refuses the buffer. The
    /// caller must not bind it for a write, nor where the command buffer being recorded has
    /// already written the bytes (see the type's remarks).
    /// </summary>
    public bool TryResolveGuestView(ulong gpuVa, ulong size, out MetalGuestRange range)
    {
        range = default;
        if (!MayUseGuestView(size) || gpuVa == 0 || size > ulong.MaxValue - GuestViewPageBytes - gpuVa)
        {
            return false;
        }

        var end = gpuVa + size;
        GuestView view;
        lock (_viewLock)
        {
            // Made under the lock an unmap takes, so a view is either made before the unmap
            // drops it or sees the range gone.
            if ((FindView(gpuVa, end) ?? CreateView(gpuVa, end)) is not { } found)
            {
                return false;
            }

            view = found;
            view.LastUseSerial = _submissionSerial;
            if (!view.Touched)
            {
                view.Touched = true;
                _touchedViews.Add(view);
                MarkResidentsDirty();
            }
        }

        Interlocked.Increment(ref _viewBinds);
        Interlocked.Add(ref _viewBindBytes, (long)size);
        var offset = gpuVa - view.Start;
        range = new MetalGuestRange(view.Buffer, offset, size, view.GpuAddress + offset);
        MetalGpuReferenceCensus.Mark(gpuVa, size);
        Trap?.NoteBinding("view", gpuVa, size, (ulong)view.Buffer, offset, range.GpuAddress);
        return true;
    }

    /// <summary>The guest unmapped a range: views over any of it go, and their buffers once the GPU is done.</summary>
    public void OnGuestUnmap(ulong address, ulong size)
    {
        if (size == 0 || Volatile.Read(ref _viewCount) == 0)
        {
            return;
        }

        var end = size > ulong.MaxValue - address ? ulong.MaxValue : address + size;
        lock (_viewLock)
        {
            List<GuestView>? stale = null;
            foreach (var view in _views.Values)
            {
                if (view.Start < end && address < view.End)
                {
                    (stale ??= new List<GuestView>()).Add(view);
                }
            }

            if (stale is null)
            {
                return;
            }

            foreach (var view in stale)
            {
                RemoveViewLocked(view);
                _viewInvalidations++;
            }
        }
    }

    // Caller holds _viewLock. The smallest view holding the whole range.
    private GuestView? FindView(ulong start, ulong end)
    {
        var key = (start & ~(GuestViewPageBytes - 1), (end + GuestViewPageBytes - 1) & ~(GuestViewPageBytes - 1));
        if (_views.TryGetValue(key, out var exact))
        {
            return exact;
        }

        GuestView? best = null;
        foreach (var view in _views.Values)
        {
            if (view.Start <= start && end <= view.End && (best is null || view.Size < best.Size))
            {
                best = view;
            }
        }

        return best;
    }

    // Caller holds _viewLock.
    private GuestView? CreateView(ulong start, ulong end)
    {
        var viewStart = start & ~(GuestViewPageBytes - 1);
        var viewEnd = (end + GuestViewPageBytes - 1) & ~(GuestViewPageBytes - 1);
        var length = viewEnd - viewStart;
        if (!_backing.TryResolveBackedHostView(viewStart, length, out var host) || (host & (GuestViewPageBytes - 1)) != 0)
        {
            _viewsUnbacked++;
            return null;
        }

        var buffer = MetalV2Native.SendNewBufferNoCopy(
            _device,
            SelNewBufferNoCopy,
            (nint)host,
            (nuint)length,
            Mtl.ResourceStorageModeShared | HazardTrackingMode,
            0);
        if (buffer == 0)
        {
            // A page of it is not readable (PROT_NONE) or not mapped at all.
            _viewCreateFailures++;
            MetalV2Diagnostics.InfoOnce(
                "guest-view-refused",
                "The device refused a no-copy buffer over a guest mapping; the range is staged instead: " +
                $"address=0x{viewStart:X16} size=0x{length:X}.");
            return null;
        }

        SetLabel(buffer, $"guest.view[0x{viewStart:X}+0x{length:X}]");
        var view = new GuestView
        {
            Start = viewStart,
            End = viewEnd,
            Buffer = buffer,
            GpuAddress = MetalNative.SendGpuAddress(buffer, SelGpuAddress),
        };
        _views[(viewStart, viewEnd)] = view;
        Volatile.Write(ref _viewCount, _views.Count);
        _viewBytes += length;
        _viewsCreated++;
        if (_residencySet != 0)
        {
            AddResident(buffer);
            CommitResidency();
        }

        return view;
    }

    // Caller holds _viewLock. The buffer stays until the last submission that bound it completes.
    private void RemoveViewLocked(GuestView view)
    {
        if (!_views.Remove((view.Start, view.End)))
        {
            return;
        }

        Volatile.Write(ref _viewCount, _views.Count);
        _viewBytes -= view.Size;
        _retiredViews.Add((view.Buffer, view.Size, view.LastUseSerial));
    }

    private void BeginGuestViewSubmission()
    {
        foreach (var view in _touchedViews)
        {
            view.Touched = false;
        }

        _touchedViews.Clear();
        if (Volatile.Read(ref _viewCount) == 0 && _retiredViews.Count == 0)
        {
            return;
        }

        lock (_viewLock)
        {
            if (_viewBytes > GuestViewBudgetBytes || _submissionSerial - _viewTrimSerial >= GuestViewTrimSerials)
            {
                _viewTrimSerial = _submissionSerial;
                TrimViewsLocked();
            }

            ReleaseRetiredViewsLocked();
        }
    }

    // Caller holds _viewLock. Views idle for as long as a window is kept go; past the budget,
    // the least recently used of those the GPU is done with go too.
    private void TrimViewsLocked()
    {
        List<GuestView>? done = null;
        foreach (var view in _views.Values)
        {
            if (view.LastUseSerial <= _completedSerial)
            {
                (done ??= new List<GuestView>()).Add(view);
            }
        }

        if (done is null)
        {
            return;
        }

        done.Sort(static (left, right) => left.LastUseSerial.CompareTo(right.LastUseSerial));
        foreach (var view in done)
        {
            var idle = _completedSerial - view.LastUseSerial >= IdleWindowSerials;
            if (!idle && _viewBytes <= GuestViewBudgetBytes)
            {
                break;
            }

            RemoveViewLocked(view);
        }
    }

    private void ReleaseRetiredViews()
    {
        if (_retiredViews.Count == 0)
        {
            return;
        }

        lock (_viewLock)
        {
            ReleaseRetiredViewsLocked();
        }
    }

    // Caller holds _viewLock.
    private void ReleaseRetiredViewsLocked()
    {
        for (var index = _retiredViews.Count - 1; index >= 0; index--)
        {
            var (buffer, _, serial) = _retiredViews[index];
            if (serial > _completedSerial)
            {
                continue;
            }

            if (_residencySet != 0)
            {
                RemoveResident(buffer);
            }

            MetalNative.SendVoid(buffer, SelRelease);
            _viewsReleased++;
            _retiredViews.RemoveAt(index);
        }
    }

    private void DisposeGuestViews()
    {
        lock (_viewLock)
        {
            foreach (var view in _views.Values)
            {
                MetalNative.SendVoid(view.Buffer, SelRelease);
            }

            foreach (var (buffer, _, _) in _retiredViews)
            {
                MetalNative.SendVoid(buffer, SelRelease);
            }

            _views.Clear();
            _retiredViews.Clear();
            _touchedViews.Clear();
            Volatile.Write(ref _viewCount, 0);
            _viewBytes = 0;
        }
    }
}
