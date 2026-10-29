// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.Core.Loader;
using Xunit;

namespace SharpEmu.Libs.Tests.Loader;

public sealed class RageResolutionPatchTests
{
    private const ulong CodeBase = 0x800000000UL;

    private static EngineResolution.Segment Code(byte[] bytes, ulong start = CodeBase) => new(start, bytes, Executable: true);

    // The video-settings -width/-height override as shipped (app+0x1693748).
    private static readonly byte[] RageOverride =
    [
        0x48, 0x85, 0xFF, 0x74, 0x1B, 0x80, 0x3F, 0x00, 0x74, 0x16, 0x31, 0xF6, 0x31, 0xD2, 0xE8, 0x75,
        0x1F, 0x9E, 0x01, 0x89, 0x83, 0xA8, 0x01, 0x00, 0x00, 0xC6, 0x05, 0xA3, 0x2C, 0x6F, 0x03, 0x01,
        0x48, 0x8B, 0x3D, 0xB9, 0x99, 0x1D, 0x03, 0x48, 0x85, 0xFF, 0x74, 0x1B, 0x80, 0x3F, 0x00, 0x74,
        0x16, 0x31, 0xF6, 0x31, 0xD2, 0xE8, 0x4E, 0x1F, 0x9E, 0x01, 0x89, 0x83, 0xAC, 0x01, 0x00, 0x00,
    ];

    [Fact]
    public void RageOverrideAlwaysStoresTheSizeAsIfTheParametersWereGiven()
    {
        var code = new byte[] { 0xCC }.Concat(RageOverride).ToArray();
        Assert.Equal(CodeBase + 1, RageResolutionPatch.Find([Code(code)]));

        var patched = RageResolutionPatch.BuildReplacement(RageOverride, 1920, 1080);

        // Width block: the branches and the NULL dereference are gone, strtol is mov eax, 1920.
        Assert.Equal([0x48, 0x85, 0xFF, 0x66, 0x90, 0x0F, 0x1F, 0x00, 0x66, 0x90, 0x31, 0xF6, 0x31, 0xD2, 0xB8], patched[..15]);
        Assert.Equal(1920u, BinaryPrimitives.ReadUInt32LittleEndian(patched.AsSpan(15)));
        Assert.Equal(RageOverride[19..39], patched[19..39]);
        Assert.Equal(0xB8, patched[39 + 14]);
        Assert.Equal(1080u, BinaryPrimitives.ReadUInt32LittleEndian(patched.AsSpan(39 + 15)));
        Assert.Equal(RageOverride[^6..], patched[^6..]);
    }

    [Fact]
    public void RageOverrideIsLeftAloneWhenTheCallIsNotStrtolShaped()
    {
        var changed = (byte[])RageOverride.Clone();
        changed[21] = 0xB0; // a different settings field
        Assert.Null(RageResolutionPatch.Find([Code(changed)]));
    }
}
