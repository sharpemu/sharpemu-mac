// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.HLE.GpuMemory;

public interface IGuestImageStore
{
    // Returns true when recovery completes, even if a later operation adds a new watch.
    bool MarkCpuWrite(ulong address, ulong size);

    // A managed write of exactly these bytes, not a fault's approximate range. A store
    // that cannot use the difference treats it as a fault.
    bool MarkPreciseCpuWrite(ulong address, ulong size) => MarkCpuWrite(address, size);

    // A managed write that has already landed, reported again for an upload that may have
    // read the bytes before it. Unlike a fault, nothing may be put under these bytes now.
    bool MarkCpuWriteLanded(ulong address, ulong size) => MarkCpuWrite(address, size);

    void Unregister(ulong address, ulong size);
}
