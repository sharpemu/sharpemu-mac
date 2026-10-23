// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Resources;

// Owned by the materializing thread (Metal worker B), never by the PM4 producer.
public static class UnreadableGuestPointer
{
    [ThreadStatic] private static ulong? _address;
    public static ulong? Address => _address;
    public static void BeginAttempt() => _address = null;
    internal static void Record(ulong address) => _address ??= address;
}
