// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text.Json;
using System.Text.Json.Serialization;
using SharpEmu.Logging;

namespace SharpEmu.Application;

/// <summary>
/// Launcher and machine-wide settings. Anything that changes how a game runs or is shown lives
/// in <see cref="PerGameSettings"/>; <see cref="SettingsSplitMigration"/> moves it out of
/// files written before the split.
/// </summary>
public sealed class ApplicationSettings
{
    private static readonly JsonSerializerOptions SerializerOptions = new()
    {
        WriteIndented = true,
    };

    public List<string> GameFolders { get; set; } = new();

    /// <summary>Eboot paths hidden from the library via "Remove from library".</summary>
    public List<string> ExcludedGames { get; set; } = new();

    /// <summary>
    /// Where a game that logs to file (<see cref="PerGameSettings.LogToFile"/>) writes; null
    /// mirrors output to user/logs/&lt;titleId&gt;-&lt;timestamp&gt;.log.
    /// </summary>
    public string? LogFilePath { get; set; }

    /// <summary>
    /// If <see cref="OverrideLogFile"/> is false it appends &lt;titleId&gt;-&lt;timestamp&gt; to the filename specified by 
    /// <see cref="LogFilePath"/>. Otherwise it uses the exact filename from <see cref="LogFilePath"/>
    /// </summary>
    public bool OverrideLogFile { get; set; }

    /// <summary>Loop the selected game's sce_sys/snd0.at9 preview music.</summary>
    public bool PlayTitleMusic { get; set; } = true;

    public double TitleMusicVolume { get; set; } = 1.0;

    public bool TrophySound { get; set; } = true;

    public string GnmTrophyKey { get; set; } = string.Empty;

    public string LibraryLayout { get; set; } = "Carousel";

    /// <summary>Cover size in the grid view, as a multiple of the size that fits the window.</summary>
    public double LibraryTileScale { get; set; } = 1.0;

    public double EmbeddedConsoleHeight { get; set; } = 240;
    public double ConsoleWindowWidth { get; set; } = 980;
    public double ConsoleWindowHeight { get; set; } = 620;
    public int? ConsoleWindowLeft { get; set; }
    public int? ConsoleWindowTop { get; set; }
    public bool ConsoleWindowMaximized { get; set; }

    public string? EmulatorPath { get; set; }

    /// <summary>UI language, matching a file code under Languages/ (e.g. "en", "tr").</summary>
    public string Language { get; set; } = "en";

    /// <summary>Launcher colour theme: a LauncherThemes palette id (Graphite, Oled for Onyx, Midnight, ...) or Custom.</summary>
    public string LauncherTheme { get; set; } = "Graphite";

    /// <summary>Custom theme background, #RRGGBB.</summary>
    public string CustomThemeBackground { get; set; } = "#121212";

    /// <summary>Custom theme accent, #RRGGBB.</summary>
    public string CustomThemeAccent { get; set; } = "#2E7CF6";

    /// <summary>Opacity of the glass controls over key art, 30-100 percent.</summary>
    public int GlassOpacity { get; set; } = 75;

    /// <summary>Theme contrast from -50 (softer) to +50 (stronger); 0 is the theme as designed.</summary>
    public int ThemeContrast { get; set; }

    // The console's system language, machine-wide as on the console itself; trophy lists
    // shown outside any game follow it too.
    public string ConsoleLanguage { get; set; } = "EnglishUS";

    /// <summary>Publish launcher/game status to Discord Rich Presence.</summary>
    public bool DiscordRichPresence { get; set; } = true;

    public bool CheckForUpdatesOnStartup { get; set; } = true;

    /// <summary>
    /// Discord application ID used for Rich Presence; the default is the
    /// SharpEmu application. Override to rebrand what Discord shows as
    /// "Playing …" (register at discord.com/developers/applications).
    /// </summary>
    public string DiscordClientId { get; set; } = "1525606762248540221";

    // Keys this build does not know survive a save: a newer build's settings, and the game
    // settings of a pre-split file whose migration has not completed yet.
    [JsonExtensionData]
    public Dictionary<string, JsonElement>? AdditionalProperties { get; set; }

    // The emulator is portable and keeps its data next to the executable;
    // the GUI follows the same convention. Inside a macOS .app bundle that
    // directory is code-signed and read-only, so SharpEmuPaths redirects to
    // ~/Library/Application Support/SharpEmu there.
    public static string SettingsPath => SharpEmuPaths.DataPath("gui-settings.json");

    public static ApplicationSettings Load()
    {
        try
        {
            if (File.Exists(SettingsPath))
            {
                var json = File.ReadAllText(SettingsPath);
                return NormalizeFromJson(json);
            }
        }
        catch (Exception)
        {
            // Corrupt or unreadable settings fall back to defaults.
        }

        return new ApplicationSettings();
    }

    /// <summary>
    /// Deserializes settings and normalizes null references and null or empty list
    /// entries introduced by JSON. Empty scalar strings remain unchanged.
    /// </summary>
    public static ApplicationSettings NormalizeFromJson(string json)
    {
        var settings = JsonSerializer.Deserialize<ApplicationSettings>(json, SerializerOptions) ?? new ApplicationSettings();

        settings.GameFolders = FilterNullOrEmpty(settings.GameFolders);
        settings.ExcludedGames = FilterNullOrEmpty(settings.ExcludedGames);
        settings.Language ??= "en";
        settings.GnmTrophyKey = TryNormalizeTrophyKey(settings.GnmTrophyKey, out var trophyKey)
            ? trophyKey : string.Empty;
        settings.DiscordClientId ??= "1525606762248540221";
        settings.LibraryLayout = NormalizeChoice(settings.LibraryLayout, "Carousel", "Grid");
        settings.TitleMusicVolume = double.IsFinite(settings.TitleMusicVolume)
            ? Math.Clamp(settings.TitleMusicVolume, 0.0, 1.0)
            : 1.0;
        settings.LibraryTileScale = double.IsFinite(settings.LibraryTileScale)
            ? Math.Clamp(settings.LibraryTileScale, 0.75, 1.6)
            : 1.0;
        settings.ConsoleLanguage = NormalizeChoice(
            settings.ConsoleLanguage,
            "EnglishUS",
            "Japanese",
            "French",
            "SpanishSpain",
            "German",
            "Italian",
            "Dutch",
            "PortuguesePortugal",
            "Russian",
            "Korean",
            "ChineseTraditional",
            "ChineseSimplified",
            "Finnish",
            "Swedish",
            "Danish",
            "Norwegian",
            "Polish",
            "PortugueseBrazil",
            "EnglishUK",
            "Turkish",
            "SpanishLatinAmerica",
            "Arabic",
            "FrenchCanada",
            "Czech",
            "Hungarian",
            "Greek",
            "Romanian",
            "Thai",
            "Vietnamese",
            "Indonesian",
            "Ukrainian");
        settings.EmbeddedConsoleHeight = NormalizeConsoleSize(settings.EmbeddedConsoleHeight, 240, 120);
        settings.ConsoleWindowWidth = NormalizeConsoleSize(settings.ConsoleWindowWidth, 980, 520);
        settings.ConsoleWindowHeight = NormalizeConsoleSize(settings.ConsoleWindowHeight, 620, 320);
        if (!settings.ConsoleWindowLeft.HasValue || !settings.ConsoleWindowTop.HasValue)
        {
            settings.ConsoleWindowLeft = null;
            settings.ConsoleWindowTop = null;
        }

        return settings;
    }

    // JSON can populate non-nullable lists with null references and entries.
    private static List<string> FilterNullOrEmpty(List<string>? source)
    {
        if (source is null)
        {
            return [];
        }

        return source.Where(entry => !string.IsNullOrEmpty(entry)).ToList();
    }

    // The rule GnmTrophyPackageReader.TryNormalizeKey applies: empty, or 32 hexadecimal digits.
    // Kept here so the settings assembly does not depend on the GnmMode runtime.
    private static bool TryNormalizeTrophyKey(string? key, out string normalized)
    {
        normalized = key?.Trim().ToUpperInvariant() ?? string.Empty;
        return normalized.Length == 0 || normalized.Length == 32 && normalized.All(char.IsAsciiHexDigit);
    }

    internal static string NormalizeChoice(string? value, string fallback, params string[] choices) =>
        choices.Prepend(fallback).FirstOrDefault(
            choice => string.Equals(choice, value, StringComparison.OrdinalIgnoreCase)) ?? fallback;

    private static double NormalizeConsoleSize(double value, double fallback, double minimum) =>
        double.IsFinite(value) && value >= minimum ? Math.Min(value, 16384) : fallback;

    public void Save()
    {
        try
        {
            SaveOrThrow();
        }
        catch (Exception)
        {
            // Settings persistence is best-effort.
        }
    }

    /// <summary>Persists the existing settings schema, propagating failures to an explicit save UI.</summary>
    public void SaveOrThrow()
    {
        SettingsFilePersistence.Write(SettingsPath, JsonSerializer.Serialize(this, SerializerOptions));
    }
}
