// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using LibAtrac9;
using SharpEmu.HLE;

namespace SharpEmu.Libs.Ngs2;

public static partial class Ngs2Exports
{
    private static Atrac9Config? ReadAtrac9Config(ReadOnlySpan<byte> data)
    {
        // Accept a packed config word and the raw four-byte config returned by our RIFF parser.
        var config = data.ToArray();
        if (config[0] is not (0xfe or 0x30)) Array.Reverse(config);
        try { return new Atrac9Config(config); }
        catch (InvalidDataException) { return null; }
    }
}
