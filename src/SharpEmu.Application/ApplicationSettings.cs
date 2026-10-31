// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text.Json;
using SharpEmu.Logging;

namespace SharpEmu.Application;

public sealed class ApplicationSettings
{
    private static readonly JsonSerializerOptions SerializerOptions = new()
    {
        WriteIndented = true,
    };

    public List<string> GameFolders { get; set; } = new();

    /// <summary>Eboot paths hidden from the library via "Remove from library".</summary>
    public List<string> ExcludedGames { get; set; } = new();

    public string LogLevel { get; set; } = "Info";

    public int ImportTraceLimit { get; set; }

    public bool StrictDynlibResolution { get; set; }

    /// <summary>
    /// Mirror emulator output to user/logs/&lt;titleId&gt;-&lt;timestamp&gt;.log, if <see cref="LogFilePath"/> is null.
    /// </summary>
    public bool LogToFile { get; set; }

    /// <summary>If <see cref="LogToFile"/> is true it logs to this file path.</summary>
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

    /// <summary>Launcher colour theme: Graphite, Oled, Midnight, Violet or Custom.</summary>
    public string LauncherTheme { get; set; } = "Graphite";

    /// <summary>Custom theme background, #RRGGBB.</summary>
    public string CustomThemeBackground { get; set; } = "#121212";

    /// <summary>Custom theme accent, #RRGGBB.</summary>
    public string CustomThemeAccent { get; set; } = "#2E7CF6";

    /// <summary>Opacity of the glass controls over key art, 30-100 percent.</summary>
    public int GlassOpacity { get; set; } = 75;

    /// <summary>Theme contrast from -50 (softer) to +50 (stronger); 0 is the theme as designed.</summary>
    public int ThemeContrast { get; set; }

    public string ConsoleLanguage { get; set; } = "EnglishUS";

    /// <summary>Publish launcher/game status to Discord Rich Presence.</summary>
    public bool DiscordRichPresence { get; set; } = true;

    public bool CheckForUpdatesOnStartup { get; set; } = true;

    public string WindowMode { get; set; } = "Windowed";

    public string Resolution { get; set; } = "1920x1080";

    public int DisplayIndex { get; set; }

    public int RefreshRate { get; set; }

    public string ScalingMode { get; set; } = "Fit";

    public bool VSync { get; set; } = true;

    public string HdrMode { get; set; } = "Auto";

    // What the guest is told the display is, which is what decides how much pixel work a
    // title does. Not the window size: see GuestDisplayResolution.
    public string GuestDisplayResolution { get; set; } = "1080p";

    public bool OverlayEnabled { get; set; } = true;
    public string OverlayCorner { get; set; } = "TopRight";
    public string OverlayMode { get; set; } = "TitleBar";

    /// <summary>Names of SHARPEMU_* switches set to "1" in the emulator's environment at launch.</summary>
    public List<string> EnvironmentToggles { get; set; } = ["SHARPEMU_WRITABLE_APP0", "SHARPEMU_DISABLE_IMPORT_LOOP_GUARD"];

    public string BinkPlaybackMode { get; set; } = "Guest";

    /// <summary>
    /// Discord application ID used for Rich Presence; the default is the
    /// SharpEmu application. Override to rebrand what Discord shows as
    /// "Playing …" (register at discord.com/developers/applications).
    /// </summary>
    public string DiscordClientId { get; set; } = "1525606762248540221";

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
        settings.EnvironmentToggles = FilterNullOrEmpty(settings.EnvironmentToggles ?? ["SHARPEMU_WRITABLE_APP0", "SHARPEMU_DISABLE_IMPORT_LOOP_GUARD"]);
        settings.LogLevel ??= "Info";
        settings.Language ??= "en";
        // Nothing reads SHARPEMU_DEFAULT_PROFILE any more; older files may still carry it.
        settings.EnvironmentToggles.RemoveAll(entry =>
            string.Equals(
                entry.Split('=', 2, StringSplitOptions.TrimEntries)[0],
                "SHARPEMU_DEFAULT_PROFILE",
                StringComparison.OrdinalIgnoreCase));
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
        settings.WindowMode = NormalizeChoice(settings.WindowMode, "Windowed", "Borderless", "Exclusive");
        settings.Resolution = NormalizeResolution(settings.Resolution);
        settings.ScalingMode = NormalizeChoice(settings.ScalingMode, "Fit", "Cover", "Stretch", "Integer");
        settings.HdrMode = NormalizeChoice(settings.HdrMode, "Auto", "On", "Off");
        settings.GuestDisplayResolution = NormalizeChoice(
            settings.GuestDisplayResolution, "1080p", "720p", "1440p", "2160p");
        settings.BinkPlaybackMode = NormalizeChoice(settings.BinkPlaybackMode, "Guest", "Host", "Skip");
        settings.OverlayCorner = NormalizeChoice(settings.OverlayCorner, "TopRight", "TopLeft", "BottomRight", "BottomLeft");
        settings.OverlayMode = NormalizeChoice(settings.OverlayMode, "TitleBar", "Full", "Minimal");
        settings.DisplayIndex = Math.Max(0, settings.DisplayIndex);
        settings.RefreshRate = Math.Clamp(settings.RefreshRate, 0, 1000);
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

    private static string NormalizeChoice(string? value, string fallback, params string[] choices) =>
        choices.Prepend(fallback).FirstOrDefault(
            choice => string.Equals(choice, value, StringComparison.OrdinalIgnoreCase)) ?? fallback;

    private static double NormalizeConsoleSize(double value, double fallback, double minimum) =>
        double.IsFinite(value) && value >= minimum ? Math.Min(value, 16384) : fallback;

    private static string NormalizeResolution(string? value)
    {
        if (!DisplayResolution.TryParse(value, out var width, out var height))
        {
            return "1920x1080";
        }

        return $"{width}x{height}";
    }

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
