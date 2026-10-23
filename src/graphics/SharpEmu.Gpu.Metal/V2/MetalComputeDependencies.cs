// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

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
    internal readonly List<MetalComputeAccess> Accesses = [];
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
    private readonly List<MetalComputeAccess> _epoch = [];
    private bool _unknownRead;
    private bool _unknownWrite;
    private bool _hasDispatch;
    private bool _hasWrite;
    internal bool HasDispatch => _hasDispatch;

    internal bool NeedsBarrier(MetalComputeFootprint next)
    {
        if (!_hasDispatch) return false;
        if (_unknownWrite || next.UnknownWrite ||
            (next.UnknownRead && _hasWrite) ||
            (_unknownRead && next.Accesses.Any(access => access.Written))) return true;
        foreach (var access in next.Accesses)
        foreach (var previous in _epoch)
        {
            if (access.Conflicts(previous)) return true;
        }

        return false;
    }

    internal void Record(MetalComputeFootprint next)
    {
        _epoch.AddRange(next.Accesses);
        _unknownRead |= next.UnknownRead;
        _unknownWrite |= next.UnknownWrite;
        _hasWrite |= next.UnknownWrite || next.Accesses.Any(access => access.Written);
        _hasDispatch = true;
    }

    internal void Reset()
    {
        _epoch.Clear();
        _unknownRead = _unknownWrite = _hasDispatch = _hasWrite = false;
    }
}
