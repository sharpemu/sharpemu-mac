// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE.Host;

namespace SharpEmu.HLE;

public interface IGuestBackedSpace
{
    bool TryHoldRange(ulong address, ulong size);
    bool TryHoldRangeAtOrAbove(ulong searchStart, ulong size, ulong alignment, out ulong address);
    bool TryHoldAvailableRange(ulong searchStart, ulong size, ulong alignment, out ulong address) =>
        TryHoldRangeAtOrAbove(searchStart, size, alignment, out address);
    bool TryMapBacked(ulong address, ulong size, ulong backingOffset, GuestPageProtection protection, out HostViewFailure failure);
    bool TryUnmapBacked(ulong address, ulong size);
    bool TryClearBacking(ulong offset, ulong size);

    // Bytes per separately releasable chunk of the backing, or 0 when the backing is one
    // object whose pages cannot be given back while it is mapped (the POSIX shm default).
    ulong BackingChunkBytes => 0;

    // Bytes per purgeable object behind those chunks (several chunks each), or 0 for one
    // object: the unit a no-copy GPU window over the backing costs in.
    ulong BackingObjectBytes => BackingChunkBytes;

    // The guest freed these backing bytes (direct memory released, flexible memory or pool
    // pages unmapped or decommitted): the whole chunks inside the range that no view maps go
    // back to the OS and read zero when they are used again. True when anything was released.
    bool TryReleaseBacking(ulong offset, ulong size) => false;

    // Which chunks of a chunked backing a texture holds in place of the backing
    // (single-copy textures); null when the backing is one object.
    GuestMemory.GuestSingleCopy? SingleCopy => null;

    bool IsBackedView(ulong address);
    bool IsBackedRange(ulong address, ulong size);

    // Wait for view replacement, then confirm that the restored page allows the access.
    bool CanRetryRestoredViewAccess(ulong address, GuestPageProtection access) => false;

    // Confirm current host access without changing the mapping or its protection.
    bool AllowsMappedAccess(ulong address, GuestPageProtection access) => false;

    // Copies through the backing alias only: no protection change, no store notification.
    bool TryWriteBacking(ulong address, ReadOnlySpan<byte> data);
    bool TryReadBacking(ulong address, Span<byte> data);

    object? BackingAliasSnapshot => null;

    // Optional admission proof for reusing a previously resolved alias span. An
    // opaque snapshot alone is insufficient: backing lifetime/admission may change
    // without replacing it. Implement with existing mapping/backing state only;
    // this check must not wait, allocate or resolve the range again.
    bool IsBackingAliasSnapshotCurrent(object snapshot) => false;

    bool TryEnterBackingAliasAccess() => false;

    void ExitBackingAliasAccess()
    {
    }

    bool TryResolveBackingAlias(ulong address, ulong size, out ulong alias)
    {
        alias = 0;
        return false;
    }

    // The same, for a range that may cross the edge between two mappings. A guest
    // allocation mapped in several calls is still one contiguous run of backing
    // bytes, so a buffer that straddles a mapping edge has one host address;
    // without this a caller that maps the backing no-copy has to stage a copy.
    bool TryResolveBackingAliasSpan(ulong address, ulong size, out ulong alias) =>
        TryResolveBackingAlias(address, size, out alias);

    // The host address the guest range itself is mapped at, when every byte of it is a
    // backed view and the views lie side by side in host memory, whatever backing offsets
    // they map. A range of pages mapped one by one from scattered backing offsets has no
    // single alias address, but it is still one host range here: the Metal backend binds a
    // read of such a range as a no-copy buffer over this mapping rather than a copy.
    bool TryResolveBackedHostView(ulong address, ulong size, out ulong host)
    {
        host = 0;
        return false;
    }

    // The whole backing seen as one contiguous host range: its base pointer and byte
    // size, or zero when the space has no alias. The Metal backend maps this range as
    // no-copy MTLBuffer windows, so every address TryResolveBackingAlias returns falls
    // inside a window at a known offset.
    (ulong Base, ulong Size) BackingAliasRange => (0, 0);
}
