// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;

namespace SharpEmu.Gpu.Metal.V2;

// Guest addresses catch aliases across views/windows/staged copies; native buffer
// coordinates catch bindings of the same allocation through different descriptors.
internal enum MetalComputeAddressSpace { Guest, Backing, Buffer, Device, Texture }

internal readonly record struct MetalComputeAccess(
    MetalComputeAddressSpace Space, ulong Resource, ulong Start, ulong Size, bool Written)
{
    internal bool Conflicts(in MetalComputeAccess other) =>
        (Written || other.Written) && Space == other.Space && Resource == other.Resource &&
        Size != 0 && other.Size != 0 &&
        (Start <= other.Start ? other.Start - Start < Size : Start - other.Start < other.Size);
}

internal sealed class MetalComputeFootprint
{
    internal readonly List<MetalComputeAccess> Accesses = new(128);
    internal bool UnknownRead;
    internal bool UnknownWrite;

    internal void Clear()
    {
        Accesses.Clear();
        UnknownRead = UnknownWrite = false;
    }

    internal void Guest(ulong address, ulong size, bool written)
    {
        if (size == 0) return;
        if (address > ulong.MaxValue - (size - 1))
        {
            UnknownRead = true;
            UnknownWrite |= written;
            return;
        }

        Accesses.Add(new(MetalComputeAddressSpace.Guest, 0, address, size, written));
    }
}

/// <summary>
/// All accesses since the last *encoded* full-scope barrier. A dispatch is tested
/// as a unit before adding any of its accesses, so self-aliases never cut an epoch.
/// No guest barrier can be deferred or forgiven by this tracker.
/// </summary>
internal sealed class MetalComputeDependencies
{
    // Bound both storage and comparisons. A full epoch takes a scope barrier, never
    // another encoder; an oversized single footprint is retained as unknown instead.
    internal const int EpochCapacity = 4096;
    private readonly List<MetalComputeAccess> _epoch = new(EpochCapacity);
    private bool _unknownRead;
    private bool _unknownWrite;
    private bool _hasDispatch;
    private bool _hasWrite;
    internal bool HasDispatch => _hasDispatch;

    internal bool NeedsBarrier(MetalComputeFootprint next)
    {
        if (!_hasDispatch) return false;
        if (_epoch.Count + next.Accesses.Count > EpochCapacity ||
            _unknownWrite || next.UnknownWrite ||
            (next.UnknownRead && _hasWrite) || (_unknownRead && HasWrite(next))) return true;
        foreach (ref readonly var access in CollectionsMarshal.AsSpan(next.Accesses))
        foreach (ref readonly var previous in CollectionsMarshal.AsSpan(_epoch))
        {
            if (access.Conflicts(previous)) return true;
        }

        return false;
    }

    internal void Record(MetalComputeFootprint next)
    {
        if (next.Accesses.Count > EpochCapacity)
        {
            _unknownRead = _unknownWrite = true;
        }
        else
        {
            _epoch.AddRange(next.Accesses);
        }
        _unknownRead |= next.UnknownRead;
        _unknownWrite |= next.UnknownWrite;
        _hasWrite |= next.UnknownWrite || HasWrite(next);
        _hasDispatch = true;
    }

    private static bool HasWrite(MetalComputeFootprint footprint)
    {
        foreach (ref readonly var access in CollectionsMarshal.AsSpan(footprint.Accesses))
            if (access.Written) return true;
        return false;
    }

    internal void Reset()
    {
        _epoch.Clear();
        _unknownRead = _unknownWrite = _hasDispatch = _hasWrite = false;
    }
}
