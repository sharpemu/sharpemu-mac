// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Xunit;

namespace SharpEmu.Application.Tests;

public sealed class GuestDisplayResolutionSettingTests
{
    [Fact]
    public void MissingAndUnknownValuesAre1080p()
    {
        Assert.Equal("1080p", EffectiveLaunchSettings.Resolve(new PerGameSettings()).GuestDisplayResolution);
        Assert.Null(SettingsSplitMigration.CarriedValues("{}").GuestDisplayResolution);
        Assert.Null(SettingsSplitMigration.CarriedValues("""{"GuestDisplayResolution":"banana"}""").GuestDisplayResolution);
    }

    [Theory]
    [InlineData("720P", "720p")]
    [InlineData("1440p", "1440p")]
    [InlineData("2160p", "2160p")]
    public void OtherResolutionsStaySelectable(string saved, string expected)
    {
        var json = "{\"GuestDisplayResolution\":\"" + saved + "\"}";
        Assert.Equal(expected, SettingsSplitMigration.CarriedValues(json).GuestDisplayResolution);
    }
}
