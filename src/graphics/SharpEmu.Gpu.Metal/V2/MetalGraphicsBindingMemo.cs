// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Metal.V2;

internal enum MetalGraphicsBindingStage { Vertex, Mesh, Fragment }

// Only graphics argument/push slots. The host resets these decisions whenever it forgets
// encoder state, including when a new native encoder reuses an old encoder's pointer.
internal sealed class MetalGraphicsBindingMemo
{
    private struct Entry
    {
        internal bool Known;
        internal nint Selector;
        internal nint Buffer;
        internal ulong Offset;
    }

    private readonly Entry[] _entries = new Entry[6];

    internal bool Matches(MetalGraphicsBindingStage stage, nint selector, int slot, nint buffer, ulong offset)
    {
        ref var entry = ref _entries[(int)stage * 2 + slot];
        return entry.Known && entry.Selector == selector && entry.Buffer == buffer && entry.Offset == offset;
    }

    // Call only after the corresponding native binding call succeeds.
    internal void Record(MetalGraphicsBindingStage stage, nint selector, int slot, nint buffer, ulong offset) =>
        _entries[(int)stage * 2 + slot] = new Entry { Known = true, Selector = selector, Buffer = buffer, Offset = offset };

    internal void Reset() => Array.Clear(_entries);
}
