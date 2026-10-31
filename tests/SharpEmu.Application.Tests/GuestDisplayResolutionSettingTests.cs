// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using GuiSettings = SharpEmu.Application.ApplicationSettings;
using Xunit;

namespace SharpEmu.Application.Tests;

public sealed class GuestDisplayResolutionSettingTests
{
    [Fact]
    public void MissingAndUnknownValuesFollowTheResolution()
    {
        Assert.Equal("Auto", GuiSettings.NormalizeFromJson("{}").GuestDisplayResolution);
        Assert.Equal("Auto", GuiSettings.NormalizeFromJson("""{"GuestDisplayOverride":"banana"}""").GuestDisplayResolution);
    }

    /// <summary>Every save wrote the old key as "1080p" though no control set it; reading it
    /// would pin the guest to 1080p whatever resolution the user picks.</summary>
    [Fact]
    public void TheOldAlwaysWrittenKeyIsNotRead()
    {
        Assert.Equal("Auto", GuiSettings.NormalizeFromJson("""{"GuestDisplayResolution":"1080p"}""").GuestDisplayResolution);
    }

    [Theory]
    [InlineData("720P", "720p")]
    [InlineData("1080p", "1080p")]
    [InlineData("1440p", "1440p")]
    [InlineData("2160p", "2160p")]
    [InlineData("auto", "Auto")]
    public void APinnedResolutionStaysSelectable(string saved, string expected)
    {
        var json = "{\"GuestDisplayOverride\":\"" + saved + "\"}";
        Assert.Equal(expected, GuiSettings.NormalizeFromJson(json).GuestDisplayResolution);
    }
}
