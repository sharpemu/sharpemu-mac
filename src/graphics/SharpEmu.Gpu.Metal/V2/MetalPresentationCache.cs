// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Metal.V2;

// A texture pointer alone is not a content identity: guest and CPU frame buffers
// are reused. Include their sequence and every output transform that changes pixels.
internal readonly record struct MetalPresentationStamp(
    long GuestSequence, long HostSequence, long MovieSequence, nint Texture,
    double Width, double Height, bool LayerHdr, bool GuestHdr, double Headroom);

internal sealed class MetalPresentationCache
{
    private MetalPresentationStamp? _presented;

    internal bool NeedsPresent(in MetalPresentationStamp stamp, bool animatedOverlay) =>
        animatedOverlay || _presented is not { } previous || previous != stamp;

    // Called only after a successful submission, never after a failed drawable
    // acquisition. An unsuccessful attempt must retry the same content.
    internal void RecordPresented(in MetalPresentationStamp stamp) => _presented = stamp;

    internal void Invalidate() => _presented = null;
}
