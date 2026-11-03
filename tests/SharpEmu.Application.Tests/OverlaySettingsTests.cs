// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text.Json;
using SharpEmu.Application;
using Xunit;

namespace SharpEmu.Application.Tests;

public sealed class OverlaySettingsTests
{
    [Fact]
    public void MissingSettingsUseFullOverlay()
    {
        var settings = EffectiveLaunchSettings.Resolve(PerGameSettings.NormalizeFromJson("{}"));
        Assert.True(settings.OverlayEnabled);
        Assert.Equal("Full", settings.OverlayMode);
        Assert.Equal("TopRight", settings.OverlayCorner);
    }

    [Fact]
    public void LegacyGlobalSettingsNormalizeChoicesAndRoundTrip()
    {
        var settings = SettingsSplitMigration.CarriedValues("""
            {"OverlayEnabled":false,"OverlayMode":"minimal","OverlayCorner":"bottomleft"}
            """);
        settings = PerGameSettings.NormalizeFromJson(JsonSerializer.Serialize(settings))!;
        Assert.False(settings.OverlayEnabled);
        // The old minimal mode is the on-screen overlay with the minimal preset.
        Assert.Null(settings.OverlayMode); // "Full" is the default, so the migration drops it
        Assert.Equal("Minimal FPS", settings.OverlayPreset);
        Assert.Equal("BottomLeft", settings.OverlayCorner);
    }

    [Fact]
    public void InvalidChoicesUseDefaults()
    {
        var settings = SettingsSplitMigration.CarriedValues("""
            {"OverlayMode":null,"OverlayCorner":"somewhere"}
            """);
        Assert.Null(settings.OverlayMode);
        Assert.Null(settings.OverlayCorner);
    }

    [Fact]
    public void GameOptionsUseDefaultsUnlessSet()
    {
        var defaults = EffectiveLaunchSettings.Resolve(new PerGameSettings());
        Assert.True(defaults.OverlayEnabled);
        Assert.Equal("Full", defaults.OverlayMode);
        Assert.Equal("TopRight", defaults.OverlayCorner);

        var overrides = PerGameSettings.NormalizeFromJson("""
            {"OverlayEnabled":false,"OverlayMode":"Full","OverlayCorner":"TopLeft"}
            """);
        Assert.NotNull(overrides);
        overrides.RemoveDefaultValues();
        var effective = EffectiveLaunchSettings.Resolve(overrides);
        Assert.False(effective.OverlayEnabled);
        Assert.Equal("Full", effective.OverlayMode);
        Assert.Equal("TopLeft", effective.OverlayCorner);
        Assert.Null(overrides.OverlayMode);
        Assert.False(overrides.IsEmpty);
    }

    [Fact]
    public void GameOptionsMatchingDefaultsDoNotLeaveASetting()
    {
        var settings = new PerGameSettings { OverlayEnabled = true, OverlayMode = "full", OverlayCorner = "TopRight" };
        settings.RemoveDefaultValues();
        Assert.True(settings.IsEmpty);
    }

    [Fact]
    public void LayoutPresetFollowsTheSettingsLayers()
    {
        Assert.Equal("", EffectiveLaunchSettings.Defaults.OverlayPreset);
        var global = new PerGameSettings { OverlayPreset = "Compact strip" };
        Assert.Equal("Compact strip", EffectiveLaunchSettings.Resolve(global, null).OverlayPreset);
        Assert.Equal("Compact strip", EffectiveLaunchSettings.Resolve(global, new PerGameSettings()).OverlayPreset);
        // An empty name is a game's explicit choice to keep the last used layout.
        Assert.Equal("", EffectiveLaunchSettings.Resolve(global, new PerGameSettings { OverlayPreset = "" }).OverlayPreset);

        var inherited = EffectiveLaunchSettings.Over(EffectiveLaunchSettings.Defaults, global);
        var game = new PerGameSettings { OverlayPreset = "compact strip" };
        game.RemoveInheritedValues(inherited);
        Assert.Null(game.OverlayPreset);
        Assert.False(new PerGameSettings { OverlayPreset = "Inspector" }.IsEmpty);
    }

    [Fact]
    public void AStoredMinimalModeBecomesTheOnScreenOverlayWithTheMinimalPreset()
    {
        var settings = PerGameSettings.NormalizeFromJson("""{"OverlayMode":"Minimal"}""")!;
        Assert.Equal("Full", settings.OverlayMode);
        Assert.Equal("Minimal FPS", settings.OverlayPreset);

        var named = PerGameSettings.NormalizeFromJson("""{"OverlayMode":"Minimal","OverlayPreset":"Inspector"}""")!;
        Assert.Equal("Inspector", named.OverlayPreset);
    }

    [Fact]
    public void ArgumentsNameTheOverlayPresetOnlyWhenOneIsSet()
    {
        var none = new LaunchConfiguration(EffectiveLaunchSettings.Defaults, []).BuildArguments("g.zar", "EnglishUS", macOS: true);
        Assert.DoesNotContain(none, argument => argument.StartsWith("--overlay-preset"));
        var named = new LaunchConfiguration(EffectiveLaunchSettings.Defaults with { OverlayPreset = "Compact strip" }, [])
            .BuildArguments("g.zar", "EnglishUS", macOS: true);
        Assert.Contains("--overlay-preset=Compact strip", named);
    }
}
