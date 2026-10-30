// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Application;
using Xunit;

namespace SharpEmu.Application.Tests;

public sealed class EnvironmentSettingsTests
{
    private static EnvironmentSetting Setting(string id) => EnvironmentSettings.All.Single(s => s.Id == id);

    [Fact]
    public void EverySettingNamesAnEmulatorVariableOnce()
    {
        Assert.All(EnvironmentSettings.All, setting => Assert.StartsWith("SHARPEMU_", setting.Variable));
        Assert.Equal(EnvironmentSettings.All.Count, EnvironmentSettings.All.Select(s => s.Variable).Distinct().Count());
        Assert.Equal(EnvironmentSettings.All.Count, EnvironmentSettings.All.Select(s => s.Id).Distinct().Count());
    }

    [Fact]
    public void EveryChoiceSettingListsItsDefault()
    {
        foreach (var setting in EnvironmentSettings.All.Where(s => s.Kind == EnvironmentSettingKind.Choice))
        {
            Assert.Contains(setting.Choices!, choice => choice.Value == setting.DefaultChoice);
        }
    }

    [Fact]
    public void DefaultStateStoresNothing()
    {
        var entries = new List<string>();
        foreach (var setting in EnvironmentSettings.All)
        {
            if (setting.Kind == EnvironmentSettingKind.Toggle)
                EnvironmentSettings.SetEnabled(entries, setting, setting.DefaultEnabled);
            else
                EnvironmentSettings.SetChoice(entries, setting, setting.DefaultChoice);
        }

        Assert.Empty(entries);
    }

    [Fact]
    public void DefaultOnToggleStoresOnlyItsOffValue()
    {
        var entries = new List<string> { "SHARPEMU_WRITABLE_APP0" };
        var cache = Setting("ShaderCache");
        var tessellation = Setting("Tessellation");

        EnvironmentSettings.SetEnabled(entries, cache, enabled: false);
        EnvironmentSettings.SetEnabled(entries, tessellation, enabled: false);

        Assert.Equal(["SHARPEMU_WRITABLE_APP0", "SHARPEMU_SHADER_CACHE=off", "SHARPEMU_TESSELLATION=0"], entries);
        Assert.False(EnvironmentSettings.IsEnabled(entries, cache));
        Assert.False(EnvironmentSettings.IsEnabled(entries, tessellation));

        EnvironmentSettings.SetEnabled(entries, cache, enabled: true);
        Assert.Equal(["SHARPEMU_WRITABLE_APP0", "SHARPEMU_TESSELLATION=0"], entries);
        Assert.True(EnvironmentSettings.IsEnabled(entries, cache));
    }

    [Fact]
    public void DefaultOffToggleIsStoredAsBareNameAndReadBack()
    {
        var entries = new List<string>();
        var pacing = Setting("NoFlipPacing");

        Assert.False(EnvironmentSettings.IsEnabled(entries, pacing));
        EnvironmentSettings.SetEnabled(entries, pacing, enabled: true);

        Assert.Equal(["SHARPEMU_NO_FLIP_PACING"], entries);
        Assert.True(EnvironmentSettings.IsEnabled(entries, pacing));
        Assert.True(EnvironmentSettings.IsEnabled(["SHARPEMU_NO_FLIP_PACING=1"], pacing));
        Assert.False(EnvironmentSettings.IsEnabled(["SHARPEMU_NO_FLIP_PACING=0"], pacing));
    }

    [Fact]
    public void ChoiceStoresOnlyNonDefaultValues()
    {
        var entries = new List<string>();
        var latency = Setting("AudioLatency");

        Assert.Equal("60", EnvironmentSettings.GetChoice(entries, latency));
        EnvironmentSettings.SetChoice(entries, latency, "120");
        Assert.Equal(["SHARPEMU_AUDIO_LATENCY_MS=120"], entries);
        Assert.Equal("120", EnvironmentSettings.GetChoice(entries, latency));

        EnvironmentSettings.SetChoice(entries, latency, "60");
        Assert.Empty(entries);
    }

    [Fact]
    public void ManagedEntriesAreRecognizedWithOrWithoutValue()
    {
        Assert.True(EnvironmentSettings.IsManagedEntry("SHARPEMU_SPIN_NAP=0"));
        Assert.True(EnvironmentSettings.IsManagedEntry("sharpemu_spin_nap"));
        Assert.False(EnvironmentSettings.IsManagedEntry("SHARPEMU_WRITABLE_APP0"));
    }

    [Fact]
    public void PerGameOverrideOfADefaultOnSettingSurvivesInheritanceCleanup()
    {
        var global = new ApplicationSettings();
        var perGame = new PerGameSettings
        {
            EnvironmentToggles = [.. global.EnvironmentToggles, "SHARPEMU_TESSELLATION=0"],
        };

        perGame.RemoveInheritedValues(global);

        Assert.NotNull(perGame.EnvironmentToggles);
        Assert.False(EnvironmentSettings.IsEnabled(
            EffectiveLaunchSettings.Resolve(global, perGame).EnvironmentToggles, Setting("Tessellation")));
    }

    [Fact]
    public void PerGameListMatchingGlobalOnlyInSpellingIsDropped()
    {
        var global = new ApplicationSettings();
        EnvironmentSettings.SetEnabled(global.EnvironmentToggles, Setting("NoFlipPacing"), enabled: true);
        var perGame = new PerGameSettings
        {
            EnvironmentToggles = [.. global.EnvironmentToggles.Where(entry => !entry.StartsWith("SHARPEMU_NO_FLIP")),
                "SHARPEMU_NO_FLIP_PACING=1"],
        };

        perGame.RemoveInheritedValues(global);

        Assert.Null(perGame.EnvironmentToggles);
    }

    [Fact]
    public void LegacyDefaultProfileAndConsoleFieldsAreTolerated()
    {
        var settings = ApplicationSettings.NormalizeFromJson(
            """{ "DefaultProfile": "Player", "ConsoleType": "PS5", "RenderResolutionScale": 0.5, "EnvironmentToggles": ["SHARPEMU_DEFAULT_PROFILE=Player", "SHARPEMU_WRITABLE_APP0"] }""");

        Assert.Equal(["SHARPEMU_WRITABLE_APP0"], settings.EnvironmentToggles);
    }
}
