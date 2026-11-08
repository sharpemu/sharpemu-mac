// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// Written guest views: a compute dispatch's large written storage range that no window covers is bound through a guest
/// view - guest memory itself - instead of being staged. Staging a written range copies it into the ring and into a
/// pristine twin and diffs it back after the dispatch: Black Myth: Wukong binds a 128 MiB written range over scattered
/// backing about 1.6 times a frame, some 200 MB of copies a frame.
///
/// A guest view is a second GPU address for those pages, and two buffers over the same pages are not coherent inside one
/// command buffer (MetalGuestMemory.GuestViews.cs). So a view is bound only where the command buffer being recorded has
/// not written the range yet (earlier command buffers are coherent with it), and the command buffer is committed right
/// after the dispatch, so everything after it reads what it wrote through any address. Nothing commits in the middle of
/// a preparation: its images are already stamped with the open submission.
/// SHARPEMU_METAL_V2_WRITTEN_GUEST_VIEWS=0 stages such ranges again.
/// </summary>
internal sealed partial class MetalRenderHost
{
    private static readonly bool WrittenGuestViews =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_WRITTEN_GUEST_VIEWS") != "0";

    // The written ranges the dispatch being prepared binds through guest views.
    private readonly List<(ulong Address, ulong Size)> _writtenViewRanges = new();
    private long _writtenViewBinds;
    private long _writtenViewBindBytes;

    private bool TryBindWrittenGuestView(ulong address, ulong size, bool computeStage, out MetalGuestRange range)
    {
        range = default;
        if (!WrittenGuestViews || !computeStage || size < WholeGatherMinBytes || !_guestMemory.MayUseGuestView(size))
        {
            return false;
        }

        // Bytes the command buffer being recorded already wrote there (a dispatch, an image writeback) would not be
        // seen through the view.
        if (_commandBuffer != 0 && _gpuWrites.MayCover(address, size) && PendingSerialHinted(address, size) >= _submissionSerial)
        {
            return false;
        }

        if (!_guestMemory.TryResolveGuestView(address, size, out range))
        {
            return false;
        }

        _writtenViewRanges.Add((address, size));
        Interlocked.Increment(ref _writtenViewBinds);
        Interlocked.Add(ref _writtenViewBindBytes, (long)size);
        return true;
    }

    // Right after a dispatch that stored through a written guest view: its command buffer is committed, so later work
    // reads its stores through any address.
    private void CommitAfterWrittenGuestViews()
    {
        if (_writtenViewRanges.Count == 0)
        {
            return;
        }

        _writtenViewRanges.Clear();
        FlushCommandBuffer(MetalFlushCause.Other);
    }

    private string WrittenGuestViewCounters() =>
        $"written_views={Interlocked.Read(ref _writtenViewBinds)} " +
        $"written_view_mb={Interlocked.Read(ref _writtenViewBindBytes) / (1024 * 1024)}";
}
