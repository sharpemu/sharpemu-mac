// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Metal.V2;

// One immutable buffer set already named on an encoder. A new stage widens the declaration;
// a new encoder, submission or set must start again. No native handle or guest state is retained.
internal struct MetalResidencyStamp
{
    private bool _valid;
    private long _encoder, _set;
    private ulong _submission;
    private nuint _stages;

    public readonly bool Covers(long encoder, long set, ulong submission, nuint stages) =>
        _valid && _encoder == encoder && _set == set && _submission == submission && (_stages & stages) == stages;

    public void Record(long encoder, long set, ulong submission, nuint stages)
    {
        _stages = _valid && _encoder == encoder && _set == set && _submission == submission ? _stages | stages : stages;
        _valid = true;
        _encoder = encoder;
        _set = set;
        _submission = submission;
    }
}
