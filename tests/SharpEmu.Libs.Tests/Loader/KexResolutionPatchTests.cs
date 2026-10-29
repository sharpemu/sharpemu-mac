// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.Core.Loader;
using Xunit;

namespace SharpEmu.Libs.Tests.Loader;

public sealed class KexResolutionPatchTests
{
    private const ulong CodeBase = 0x800000000UL;

    private static EngineResolution.Segment Code(byte[] bytes, ulong start = CodeBase) => new(start, bytes, Executable: true);

    // The output-class selection as shipped (app+0x33CFF3) and the cmoves after it.
    private static readonly byte[] KexSelection =
    [
        0x83, 0x7D, 0xA0, 0x02, 0xBF, 0x70, 0x08, 0x00, 0x00, 0xBB, 0x38, 0x04, 0x00, 0x00, 0xB8, 0x00,
        0x0F, 0x00, 0x00, 0xBE, 0x80, 0x07, 0x00, 0x00, 0x0F, 0x44, 0xDF, 0x48, 0x8D, 0x3D, 0x53, 0xFE,
        0xF2, 0x01, 0x0F, 0x44, 0xF0,
    ];

    [Fact]
    public void KexSelectionGivesBothOutputClassesTheAdvertisedSize()
    {
        var code = new byte[] { 0x90, 0x90 }.Concat(KexSelection).ToArray();
        Assert.Equal(CodeBase + 2, KexResolutionPatch.Find([Code(code)]));

        var replacement = KexResolutionPatch.BuildReplacement(2560, 1440);
        Assert.Equal(KexResolutionPatch.Signature.Length, replacement.Length);
        Assert.Equal(1440u, BinaryPrimitives.ReadUInt32LittleEndian(replacement.AsSpan(5)));
        Assert.Equal(1440u, BinaryPrimitives.ReadUInt32LittleEndian(replacement.AsSpan(10)));
        Assert.Equal(2560u, BinaryPrimitives.ReadUInt32LittleEndian(replacement.AsSpan(15)));
        Assert.Equal(2560u, BinaryPrimitives.ReadUInt32LittleEndian(replacement.AsSpan(20)));
        // The opcodes and the class compare are untouched.
        Assert.Equal(KexResolutionPatch.Signature.AsSpan(0, 5).ToArray(), replacement.AsSpan(0, 5).ToArray());
        Assert.Equal(0xB8, replacement[14]);
    }

    [Fact]
    public void KexSelectionIsLeftAloneWhenItIsNotUnique()
    {
        var twice = KexSelection.Concat(KexSelection).ToArray();
        Assert.Null(KexResolutionPatch.Find([Code(twice)]));
        Assert.Null(KexResolutionPatch.Find([Code(new byte[64])]));
    }

    [Fact]
    public void KexApplyWritesTheSizeIntoTheImage()
    {
        var memory = new FakeCpuMemory(CodeBase, 0x100);
        Assert.True(memory.TryWrite(CodeBase, KexSelection));
        var segment = Code(KexSelection.Concat(new byte[0x100 - KexSelection.Length]).ToArray());

        KexResolutionPatch.Apply(memory, [segment], 1280, 720);

        var patched = new byte[KexSelection.Length];
        Assert.True(memory.TryRead(CodeBase, patched));
        Assert.Equal(1280u, BinaryPrimitives.ReadUInt32LittleEndian(patched.AsSpan(15)));
        Assert.Equal(720u, BinaryPrimitives.ReadUInt32LittleEndian(patched.AsSpan(10)));
    }
}
