// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Metal.V2;

// An encoder owns declarations. A changed collection must be walked again; widening
// a stage mask must redeclare resources even if the collection itself is unchanged.
internal sealed class MetalResidencyWalk
{
    private long _encoder = long.MinValue, _version = long.MinValue;
    private nuint _stages;
    internal void Reset() { _encoder = long.MinValue; _version = long.MinValue; _stages = 0; }
    internal bool IsCurrent(long encoder, long version, nuint stages) =>
        _encoder == encoder && _version == version && (_stages & stages) == stages;
    internal void Record(long encoder, long version, nuint stages)
    {
        if (_encoder != encoder || _version != version) _stages = 0;
        _encoder = encoder; _version = version; _stages |= stages;
    }
}
