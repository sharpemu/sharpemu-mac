// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Overlay;

namespace SharpEmu.GUI;

/// <summary>
/// What a game takes from the launcher's own settings rather than its launch configuration: the
/// on-screen keyboard's default text, and the palette, language and strings the in-game overlays
/// (the IME keyboard, the save data list) are drawn with, so they follow the launcher's theme and language.
/// </summary>
public static class EmulatorLaunchVariables
{
    public static IReadOnlyDictionary<string, string?> Build(GuiSettings settings) =>
        new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase)
        {
            [LaunchConfiguration.DefaultProfileVariable] = GuiSettings.NormalizeDefaultProfile(settings.DefaultProfile),
            [OverlayPalette.EnvironmentVariable] = LauncherThemes.EmulatorPalette(settings),
            [OverlayText.LanguageVariable] = EmulatorOverlayText.Language(settings),
            [OverlayText.StringsVariable] = EmulatorOverlayText.Strings(settings),
        };
}
