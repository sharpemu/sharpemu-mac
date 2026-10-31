// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.Gpu.Scheduling;

// Thrown by a recording thread when an item needs a cache mutation only the render thread may make in
// stream order; the slice ends before the item and the render thread records it itself.
internal sealed class SerialRequiredException(string reason, bool resumable = false) : Exception(reason)
{
    public bool Resumable => resumable;
}
