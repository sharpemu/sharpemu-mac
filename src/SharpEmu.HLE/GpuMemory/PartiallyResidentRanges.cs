// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.HLE.GpuMemory;

/// <summary>
/// Guest ranges a GPU read may find partly mapped, whose unmapped pages read as zero: the kernel's PRT apertures
/// (sceKernelSetPrtAperture) and the AMM area (Ampr). The kernel answers that question itself in the process that runs it
/// (KernelRuntimeCompatExports.ContainsPrtRange); the GPU process of the two-process split has no kernel, so the guest
/// process publishes each change here and the split mirrors it there (SplitGpuSession, SplitGpuHost).
/// </summary>
/// <remarks>
/// <para><b>Published</b> (the guest process): the kernel's last value per range, and an observer the split attaches.
/// Attaching replays the current values through the observer under the same lock every later change takes, so the GPU
/// process never receives an older value after a newer one.</para>
///
/// <para><b>Mirrored</b> (the GPU process): what the split received. Read lock-free by image reads
/// (ResidentImageBacking); empty in a process with a kernel, so there it changes nothing.</para>
///
/// <para>Only the extent of each range crosses. Which pages inside it are mapped is the shared view table's, which the
/// GPU process reads live; a range here only lets an unmapped page read as zero, it never makes a read of a page the
/// guest unmapped possible.</para>
/// </remarks>
public static class PartiallyResidentRanges
{
    /// <summary>The id of the AMM area; PRT apertures use their aperture id.</summary>
    public const int AmmArea = 0x100;

    /// <summary>One range: <paramref name="Size"/> 0 removes the range with that id.</summary>
    /// <param name="LazilyBacked">The kernel also registered it as a lazily backed range of the GPU memory manager
    /// (GuestGpuMemoryHook.NoteLazyRange), which the GPU process's manager has to know as well.</param>
    public readonly record struct Range(int Id, ulong Address, ulong Size, bool LazilyBacked);

    private static readonly object _gate = new();
    // Its own lock: the observer holds _gate while the GPU process mirrors the change, which in a test runs in this process.
    private static readonly object _mirrorGate = new();
    private static readonly Dictionary<int, Range> _published = [];
    private static readonly Dictionary<int, Range> _mirroredById = [];
    private static Action<Range>? _observer;
    private static Range[] _mirrored = [];

    /// <summary>The guest process: the kernel set, moved or (size 0) dropped a range.</summary>
    public static void Publish(int id, ulong address, ulong size, bool lazilyBacked)
    {
        var range = new Range(id, address, size, lazilyBacked);
        lock (_gate)
        {
            if (size == 0)
            {
                _ = _published.Remove(id);
            }
            else
            {
                _published[id] = range;
            }

            // Under the lock: the observer sees the changes in the order the table took them.
            _observer?.Invoke(range);
        }
    }

    /// <summary>
    /// The guest process: sends every change to <paramref name="observer"/> from now on, starting with the current
    /// value of every range. Null detaches.
    /// </summary>
    public static void Attach(Action<Range>? observer)
    {
        lock (_gate)
        {
            _observer = observer;
            if (observer is null)
            {
                return;
            }

            foreach (var range in _published.Values.OrderBy(static range => range.Id))
            {
                observer(range);
            }
        }
    }

    /// <summary>The guest process: the ranges as the kernel last published them.</summary>
    public static Range[] SnapshotPublished()
    {
        lock (_gate)
        {
            return [.. _published.Values.OrderBy(static range => range.Id)];
        }
    }

    /// <summary>The GPU process: a change the guest process published. Returns the value it replaces (size 0: none).</summary>
    public static Range Mirror(Range range)
    {
        lock (_mirrorGate)
        {
            _ = _mirroredById.Remove(range.Id, out var previous);
            if (range.Size != 0)
            {
                _mirroredById[range.Id] = range;
            }

            Volatile.Write(ref _mirrored, [.. _mirroredById.Values]);
            return previous;
        }
    }

    /// <summary>The ranges this process mirrors (none in a process with a kernel).</summary>
    public static Range[] Mirrored => Volatile.Read(ref _mirrored);

    /// <summary>
    /// The GPU process: whether one mirrored range holds all of [address, address + size), as
    /// KernelRuntimeCompatExports.ContainsPrtRange answers for an address the caller has translated.
    /// </summary>
    public static bool MirrorContains(ulong address, ulong size)
    {
        var ranges = Volatile.Read(ref _mirrored);
        if (ranges.Length == 0 || size == 0 || address > ulong.MaxValue - size)
        {
            return false;
        }

        foreach (var range in ranges)
        {
            if (address >= range.Address && size <= range.Size && address - range.Address <= range.Size - size)
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>Tests: forgets the mirrored ranges (a process with a kernel never has any).</summary>
    public static void ResetMirror()
    {
        lock (_mirrorGate)
        {
            _mirroredById.Clear();
            Volatile.Write(ref _mirrored, []);
        }
    }
}
