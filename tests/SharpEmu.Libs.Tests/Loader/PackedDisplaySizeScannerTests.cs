// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.Core.Loader;
using Xunit;

namespace SharpEmu.Libs.Tests.Loader;

public sealed class PackedDisplaySizeScannerTests
{
    private const ulong CodeBase = 0x800000000UL;

    private static EngineResolution.Segment Code(byte[] bytes, ulong start = CodeBase) => new(start, bytes, Executable: true);

    private static EngineResolution.Segment Data(byte[] bytes, ulong start) => new(start, bytes, Executable: false);

    private static readonly byte[] Prologue = [0x55, 0x48, 0x89, 0xE5];

    // Void Terrarium at app+0x1A0E05: the render config's packed 3840x2160 (movabs rax).
    private static readonly byte[] VoidDisplayConfig =
    [
        0xC5, 0xF8, 0x10, 0x0D, 0xF3, 0xD6, 0x27, 0x00, 0x48, 0x8D, 0xBC, 0x24, 0x30, 0x02, 0x00, 0x00,
        0x48, 0xB8, 0x00, 0x0F, 0x00, 0x00, 0x70, 0x08, 0x00, 0x00, 0x4C, 0x89, 0x35, 0xDA, 0x16, 0x4B,
        0x00, 0xC5, 0xFC, 0x11, 0x84, 0x24, 0x30, 0x02, 0x00, 0x00, 0xC3,
    ];

    // Dreaming Sarah at app+0x2B0ACB: the packed canvas size, then the letterbox math that
    // offsets from it (lea ecx, [r14 - 0xF00]).
    private static readonly byte[] SarahLetterbox =
    [
        0x48, 0xB8, 0x00, 0x0F, 0x00, 0x00, 0x70, 0x08, 0x00, 0x00, 0x41, 0x8D, 0x8E, 0x00, 0xF1, 0xFF,
        0xFF, 0xC3,
    ];

    // The SetBufferAttribute2 literals: mov ecx, 0xF00; mov r8d, 0x870.
    private static readonly byte[] BufferAttributeLiterals = [0xB9, 0x00, 0x0F, 0x00, 0x00, 0x41, 0xB8, 0x70, 0x08, 0x00, 0x00];

    [Fact]
    public void PackedDisplaySizeAndItsFunctionsLiteralsAreRewrittenTogether()
    {
        var code = Prologue.Concat(VoidDisplayConfig[..^1]).Concat(BufferAttributeLiterals).Append((byte)0xC3).ToArray();
        var plan = PackedDisplaySizeScanner.Scan([Code(code)]);

        Assert.NotNull(plan);
        Assert.Null(plan.Refusal);
        Assert.Equal(
            [
                new(CodeBase + 4 + 0x12, PackedDisplaySizeScanner.SiteKind.PackedImmediate),
                new(CodeBase + 4 + 0x2A + 1, PackedDisplaySizeScanner.SiteKind.WidthImmediate),
                new(CodeBase + 4 + 0x2A + 7, PackedDisplaySizeScanner.SiteKind.HeightImmediate),
            ],
            plan.Sites);
    }

    [Fact]
    public void TheRenderTargetSizeVectorInDataIsPartOfTheRewrite()
    {
        var vector = new byte[24];
        BinaryPrimitives.WriteSingleLittleEndian(vector.AsSpan(8), 3840f);
        BinaryPrimitives.WriteSingleLittleEndian(vector.AsSpan(12), 2160f);
        var plan = PackedDisplaySizeScanner.Scan([Code(Prologue.Concat(VoidDisplayConfig).ToArray()), Data(vector, 0x900000000UL)]);

        Assert.Contains(new PackedDisplaySizeScanner.Site(0x900000008UL, PackedDisplaySizeScanner.SiteKind.FloatPair), plan!.Sites);
        Assert.Equal(
            [0x00, 0x00, 0xF0, 0x44, 0x00, 0x00, 0x87, 0x44],
            PackedDisplaySizeScanner.Encode(PackedDisplaySizeScanner.SiteKind.FloatPair, 1920, 1080));
    }

    [Fact]
    public void AnOffsetFromTheSizeInTheSameFunctionRefusesTheWholeRewrite()
    {
        var plan = PackedDisplaySizeScanner.Scan([Code(Prologue.Concat(SarahLetterbox).ToArray())]);

        Assert.NotNull(plan);
        Assert.Empty(plan.Sites);
        Assert.Contains("offset from the size", plan.Refusal);
    }

    [Fact]
    public void ALargeExecutableOrOneWithoutThePackedSizeIsNotScanned()
    {
        Assert.Null(PackedDisplaySizeScanner.Scan([Code(Prologue.Concat(BufferAttributeLiterals).ToArray())]));
        var large = new byte[PackedDisplaySizeScanner.MaxCodeBytes + 1];
        VoidDisplayConfig.CopyTo(large, 0);
        Assert.Null(PackedDisplaySizeScanner.Scan([Code(large)]));
    }
}
