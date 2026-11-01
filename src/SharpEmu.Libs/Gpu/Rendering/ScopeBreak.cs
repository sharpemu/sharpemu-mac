// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.Gpu.Rendering;

/// <summary>Why a render scope ended before the next draw could join it.</summary>
public enum ScopeBreak
{
    Image,
    SliceImage,
    StoringDraw,
    GuestFlush,
    DeferredFlush,
    GlobalDataShare,
    Count,
}
