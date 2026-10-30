// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Application;
using GuiSettings = SharpEmu.Application.ApplicationSettings;
using Xunit;

namespace SharpEmu.Application.Tests;

public sealed class PerGameSettingsTests
{
    [Fact]
    public void NewGameInheritsWritableApp0AndGlobalCrashCapture()
    {
        var global = new GuiSettings();
        global.EnvironmentToggles.Add("SHARPEMU_CRASH_CAPTURE");
        var effective = EffectiveLaunchSettings.Resolve(global, new PerGameSettings());
        Assert.Contains("SHARPEMU_WRITABLE_APP0", effective.EnvironmentToggles);
        Assert.Contains("SHARPEMU_CRASH_CAPTURE", effective.EnvironmentToggles);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void GameCrashDumpOverrideSurvivesReload(bool enabled)
    {
        var global = new GuiSettings();
        if (!enabled)
            global.EnvironmentToggles.Add("SHARPEMU_CRASH_CAPTURE");
        var perGame = new PerGameSettings { EnvironmentToggles = ["SHARPEMU_WRITABLE_APP0"] };
        if (enabled)
            perGame.EnvironmentToggles.Add("SHARPEMU_CRASH_CAPTURE");
        perGame.RemoveInheritedValues(global);
        var restored = PerGameSettings.NormalizeFromJson(System.Text.Json.JsonSerializer.Serialize(perGame));
        var effective = EffectiveLaunchSettings.Resolve(global, restored);
        Assert.Equal(enabled, effective.EnvironmentToggles.Contains("SHARPEMU_CRASH_CAPTURE"));
        Assert.Contains("SHARPEMU_WRITABLE_APP0", effective.EnvironmentToggles);
    }

    // Invalid entries must not reach Environment.SetEnvironmentVariable.
    [Fact]
    public void NormalizeFromJson_NullOrEmptyToggleEntries_AreFilteredOut()
    {
        const string json = """
            { "EnvironmentToggles": [null, "SHARPEMU_TRACE", ""] }
            """;

        var settings = PerGameSettings.NormalizeFromJson(json);

        Assert.NotNull(settings);
        Assert.Equal(["SHARPEMU_TRACE"], settings.EnvironmentToggles);
    }

    // A null list means that the global setting should be inherited.
    [Fact]
    public void NormalizeFromJson_NullToggleList_StaysNull()
    {
        const string json = """{ "EnvironmentToggles": null }""";

        var settings = PerGameSettings.NormalizeFromJson(json);

        Assert.NotNull(settings);
        Assert.Null(settings.EnvironmentToggles);
    }

    [Fact]
    public void NormalizeFromJson_EmptyToggleList_StaysEmpty()
    {
        const string json = """{ "EnvironmentToggles": [] }""";

        var settings = PerGameSettings.NormalizeFromJson(json);

        Assert.NotNull(settings);
        Assert.Empty(Assert.IsType<List<string>>(settings.EnvironmentToggles));
    }

    [Fact]
    public void NormalizeFromJson_ValidToggles_ArePreserved()
    {
        const string json = """
            { "EnvironmentToggles": ["SHARPEMU_TRACE", "SHARPEMU_NO_JIT"] }
            """;

        var settings = PerGameSettings.NormalizeFromJson(json);

        Assert.NotNull(settings);
        Assert.Equal(["SHARPEMU_TRACE", "SHARPEMU_NO_JIT"], settings.EnvironmentToggles);
    }

    [Fact]
    public void RemoveInheritedValues_AllMatchingValues_ProducesEmptySettings()
    {
        var global = new GuiSettings
        {
            LogLevel = "Info",
            ImportTraceLimit = 32,
            StrictDynlibResolution = true,
            LogToFile = false,
            WindowMode = "Borderless",
            Resolution = "2560x1440",
            DisplayIndex = 1,
            RefreshRate = 144,
            ScalingMode = "Fit",
            VSync = true,
            HdrMode = "Auto",
            EnvironmentToggles = ["SHARPEMU_LOG_IO", "SHARPEMU_VK_VALIDATION"],
        };
        var perGame = new PerGameSettings
        {
            LogLevel = "info",
            ImportTraceLimit = 32,
            StrictDynlibResolution = true,
            LogToFile = false,
            WindowMode = "borderless",
            Resolution = "2560x1440",
            DisplayIndex = 1,
            RefreshRate = 144,
            ScalingMode = "fit",
            VSync = true,
            HdrMode = "auto",
            EnvironmentToggles = ["SHARPEMU_VK_VALIDATION=1", "sharpemu_log_io"],
        };

        perGame.RemoveInheritedValues(global);

        Assert.True(perGame.IsEmpty);
    }

    [Fact]
    public void RemoveInheritedValues_DifferentValues_RemainOverrides()
    {
        var global = new GuiSettings
        {
            LogLevel = "Info",
            Resolution = "1920x1080",
            VSync = true,
            EnvironmentToggles = ["SHARPEMU_LOG_IO"],
        };
        var perGame = new PerGameSettings
        {
            LogLevel = "Debug",
            Resolution = "2560x1440",
            VSync = false,
            EnvironmentToggles = ["SHARPEMU_VK_VALIDATION"],
        };

        perGame.RemoveInheritedValues(global);

        Assert.Equal("Debug", perGame.LogLevel);
        Assert.Equal("2560x1440", perGame.Resolution);
        Assert.False(perGame.VSync);
        Assert.Equal(["SHARPEMU_VK_VALIDATION"], perGame.EnvironmentToggles);
    }

    [Fact]
    public void RemoveInheritedValues_DisabledEnvironmentEntry_MatchesMissingEntry()
    {
        var global = new GuiSettings
        {
            EnvironmentToggles = ["SHARPEMU_LOG_IO=0"],
        };
        var perGame = new PerGameSettings
        {
            EnvironmentToggles = [],
        };

        perGame.RemoveInheritedValues(global);

        Assert.Null(perGame.EnvironmentToggles);
    }

    [Fact]
    public void EnvironmentChanges_OverrideOneVariableAndFollowGlobalForTheRest()
    {
        var global = new GuiSettings();
        var game = new PerGameSettings { EnvironmentChanges = ["SHARPEMU_CRASH_CAPTURE"] };

        // A switch turned on globally after the game was configured still reaches it.
        global.EnvironmentToggles.Add("SHARPEMU_LOG_IO");
        var effective = EffectiveLaunchSettings.Resolve(global, game);

        Assert.Contains("SHARPEMU_CRASH_CAPTURE", effective.EnvironmentToggles);
        Assert.Contains("SHARPEMU_LOG_IO", effective.EnvironmentToggles);
        Assert.Contains("SHARPEMU_WRITABLE_APP0", effective.EnvironmentToggles);
    }

    [Fact]
    public void EnvironmentChanges_ExplicitOffBeatsGlobalOnAndNeverReachesTheEnvironment()
    {
        var global = new GuiSettings();
        global.EnvironmentToggles.Add("SHARPEMU_CRASH_CAPTURE");
        var game = new PerGameSettings { EnvironmentChanges = ["SHARPEMU_CRASH_CAPTURE=0"] };

        var effective = EffectiveLaunchSettings.Resolve(global, game);

        Assert.DoesNotContain(effective.EnvironmentToggles, entry => entry.StartsWith("SHARPEMU_CRASH_CAPTURE"));
    }

    [Fact]
    public void EnvironmentChanges_ManagedChoiceOverridesGlobalChoice()
    {
        var global = new GuiSettings();
        global.EnvironmentToggles.Add("SHARPEMU_AUDIO_LATENCY_MS=120");
        var game = new PerGameSettings { EnvironmentChanges = ["SHARPEMU_AUDIO_LATENCY_MS=60"] };

        var effective = EffectiveLaunchSettings.Resolve(global, game);

        // 60 ms is the emulator default, so the variable is left unset.
        Assert.DoesNotContain(effective.EnvironmentToggles, entry => entry.StartsWith("SHARPEMU_AUDIO_LATENCY_MS"));
    }

    [Fact]
    public void RemoveInheritedValues_DropsChangesThatMatchGlobal()
    {
        var global = new GuiSettings();
        global.EnvironmentToggles.Add("SHARPEMU_LOG_IO");
        var game = new PerGameSettings { EnvironmentChanges = ["SHARPEMU_LOG_IO", "SHARPEMU_LOG_NP=0"] };

        game.RemoveInheritedValues(global);

        Assert.Null(game.EnvironmentChanges);
        Assert.True(game.IsEmpty);
    }

    [Fact]
    public void LegacyWholeListStillReplacesGlobal_AndConvertsToTheSameChanges()
    {
        var global = new GuiSettings();
        global.EnvironmentToggles.Add("SHARPEMU_CRASH_CAPTURE");
        List<string> legacy = ["SHARPEMU_WRITABLE_APP0", "SHARPEMU_LOG_IO"];

        var fromLegacy = EffectiveLaunchSettings.Resolve(global, new PerGameSettings { EnvironmentToggles = legacy });
        var changes = EnvironmentOverrides.Between(global.EnvironmentToggles, legacy);
        var fromChanges = EffectiveLaunchSettings.Resolve(global, new PerGameSettings { EnvironmentChanges = changes });

        Assert.DoesNotContain("SHARPEMU_CRASH_CAPTURE", fromLegacy.EnvironmentToggles);
        Assert.Equal(
            fromLegacy.EnvironmentToggles.Select(EnvironmentOverrides.Name).Order(),
            fromChanges.EnvironmentToggles.Select(EnvironmentOverrides.Name).Order());
    }

    [Fact]
    public void PickChanged_KeepsStoredOverridesForUntouchedVariables()
    {
        List<string> baseline = ["SHARPEMU_WRITABLE_APP0", "SHARPEMU_LOG_IO"];
        List<string> current = ["SHARPEMU_WRITABLE_APP0", "SHARPEMU_LOG_IO", "SHARPEMU_LOG_NP"];

        var changes = EnvironmentOverrides.PickChanged(current, baseline, ["SHARPEMU_LOG_IO"]);

        Assert.Equal(["SHARPEMU_LOG_IO", "SHARPEMU_LOG_NP"], changes);
    }
}
