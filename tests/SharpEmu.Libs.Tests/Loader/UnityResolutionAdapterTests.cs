// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text;
using SharpEmu.Core.Loader;
using Xunit;

namespace SharpEmu.Libs.Tests.Loader;

public sealed class UnityResolutionAdapterTests
{
    private const ulong CodeBase = 0x800000000UL;

    private static EngineResolution.Segment Code(byte[] bytes, ulong start = CodeBase) => new(start, bytes, Executable: true);

    private const string SuperliminalBootConfig =
        "gfx-enable-gfx-jobs=1\n" +
        "platform-ps5-initialization-flags=2\n" +
        "platform-ps5-player-prefs-maxsize=32768\n" +
        "platform-ps5-video-out-width=2560\n" +
        "platform-ps5-video-out-pixel-format=2\n";

    // SetResolutionImmediate as shipped (app+0xF65261): the 4K-class and cap-bit maximum, then
    // cmp edx, 0x77F; jg; mov esi, 0x780.
    private static readonly byte[] SetResolutionImmediate =
    [
        0x41, 0x83, 0xBF, 0x10, 0x03, 0x00, 0x00, 0x02, 0xBB, 0x00, 0x0F, 0x00, 0x00, 0x74, 0x22, 0x8A,
        0x05, 0xC2, 0xDD, 0xB0, 0x00, 0x84, 0xC0, 0x0F, 0x84, 0x46, 0x17, 0x00, 0x00, 0xF6, 0x05, 0xAB,
        0xDD, 0xB0, 0x00, 0x02, 0xB8, 0x00, 0x0F, 0x00, 0x00, 0xBB, 0x80, 0x07, 0x00, 0x00, 0x0F, 0x44,
        0xD8, 0x81, 0xFA, 0x7F, 0x07, 0x00, 0x00, 0x7F, 0x14, 0xBE, 0x80, 0x07, 0x00, 0x00, 0x45, 0x31,
        0xF6, 0x31, 0xC0, 0xE8, 0x47,
    ];

    [Theory]
    [InlineData(1920u, "1920", "2")]
    [InlineData(1280u, "1280", "2")]
    [InlineData(2560u, "2560", "0")]
    [InlineData(3840u, "3840", "0")]
    public void UnityBootConfigTakesTheWidthAndDropsTheHdCapAboveHd(uint width, string expectedWidth, string expectedFlags)
    {
        var rewritten = UnityResolutionAdapter.RewriteBootConfig(Encoding.UTF8.GetBytes(SuperliminalBootConfig), width);

        Assert.NotNull(rewritten);
        var text = Encoding.UTF8.GetString(rewritten);
        Assert.Contains($"platform-ps5-video-out-width={expectedWidth}\n", text);
        Assert.Contains($"platform-ps5-initialization-flags={expectedFlags}\n", text);
        Assert.Contains("platform-ps5-player-prefs-maxsize=32768\n", text);
        Assert.Equal(SuperliminalBootConfig.Length, text.Length);
    }

    [Fact]
    public void AFileWithoutTheVideoOutWidthIsNotAUnityBootConfig()
    {
        Assert.Null(UnityResolutionAdapter.RewriteBootConfig(Encoding.UTF8.GetBytes("vr-enabled=0\n"), 1920));
    }

    [Fact]
    public void UnityMinimumWidthCheckIsFoundOnceAndAmbiguityRefuses()
    {
        var site = UnityResolutionAdapter.FindMinimumWidthCheck([Code(SetResolutionImmediate)]);
        Assert.Equal(CodeBase + 0x31, site);

        var twice = SetResolutionImmediate.Concat(SetResolutionImmediate).ToArray();
        Assert.Null(UnityResolutionAdapter.FindMinimumWidthCheck([Code(twice)]));
    }
}
