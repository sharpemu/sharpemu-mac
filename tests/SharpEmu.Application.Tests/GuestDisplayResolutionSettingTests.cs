// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Xunit;

namespace SharpEmu.Application.Tests;

public sealed class GuestDisplayResolutionSettingTests
{
    [Fact]
    public void MissingAndUnknownValuesFollowTheResolution()
    {
        Assert.Equal("Auto", EffectiveLaunchSettings.Resolve(new PerGameSettings()).GuestDisplayResolution);
        Assert.Null(SettingsSplitMigration.CarriedValues("{}").GuestDisplayResolution);
        Assert.Null(SettingsSplitMigration.CarriedValues("""{"GuestDisplayResolution":"banana"}""").GuestDisplayResolution);
    }

    /// <summary>Every save of the old global file wrote "1080p" though no control set it; carrying
    /// it would pin every game to 1080p whatever resolution the user picks.</summary>
    [Fact]
    public void TheOldAlwaysWritten1080pIsNotCarried()
    {
        Assert.Null(SettingsSplitMigration.CarriedValues("""{"GuestDisplayResolution":"1080p"}""").GuestDisplayResolution);
    }

    [Theory]
    [InlineData("720P", "720p")]
    [InlineData("1440p", "1440p")]
    [InlineData("2160p", "2160p")]
    public void APinnedResolutionStaysSelectable(string saved, string expected)
    {
        var json = "{\"GuestDisplayResolution\":\"" + saved + "\"}";
        Assert.Equal(expected, SettingsSplitMigration.CarriedValues(json).GuestDisplayResolution);
    }
}
