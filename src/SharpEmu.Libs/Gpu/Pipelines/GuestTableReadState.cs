// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.Gpu.Pipelines;

public readonly record struct GuestTableReadState(string Path, ulong PendingSerial, ulong CompletedSerial, string Writer,
    bool ImageOwned = false, long HostPublications = 0, bool CommandProcessorOverlay = false, bool LabelOverlay = false);
