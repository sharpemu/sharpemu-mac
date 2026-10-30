// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Avalonia.Controls;
using Avalonia.Media;

namespace SharpEmu.GUI;

/// <summary>
/// One launcher colour theme. Every field maps onto a brush in Themes/Tokens.axaml.
/// </summary>
internal sealed record LauncherPalette(
    string Id,
    string Background,
    string Chrome,
    string Menu,
    string Card,
    string Elevated,
    string RaisedHover,
    string Field,
    string FieldHover,
    string Overlay,
    string Text,
    string SecondaryText,
    string Description,
    string Muted,
    string Faint,
    string Accent,
    string AccentHover,
    string AccentPressed,
    string AccentTint,
    string AccentForeground);

/// <summary>
/// Everything the user picks about the launcher's look: the theme, the custom theme's
/// two colours, glass opacity (percent) and contrast (-50 to +50).
/// </summary>
internal sealed record LauncherAppearance(
    string? Theme,
    string? CustomBackground = null,
    string? CustomAccent = null,
    int GlassOpacity = LauncherThemes.DefaultGlassOpacity,
    int Contrast = 0);

/// <summary>
/// The launcher themes. Applying one repaints the brush instances declared in
/// Themes/Tokens.axaml in place, so every StaticResource user follows the change
/// without a restart.
/// </summary>
internal static class LauncherThemes
{
    public const string DefaultId = "Graphite";
    public const string CustomId = "Custom";
    public const string DefaultCustomBackground = "#121212";
    public const string DefaultCustomAccent = "#2E7CF6";
    public const int DefaultGlassOpacity = 75;
    public const int MinGlassOpacity = 30;
    public const int MaxContrast = 50;

    // Text, rims and focus rings are drawn light, so a custom background is kept dark.
    private const byte MaxCustomBackgroundChannel = 0x40;

    public static readonly LauncherPalette[] All =
    [
        // Neutral greys with a white primary action, the PS5 look.
        new(
            Id: "Graphite",
            Background: "#121212", Chrome: "#0A0A0A", Menu: "#1A1A1A",
            Card: "#222222", Elevated: "#2A2A2A", RaisedHover: "#353535",
            Field: "#323232", FieldHover: "#3D3D3D", Overlay: "#262626",
            Text: "#F5F5F5", SecondaryText: "#CCCCCC", Description: "#ABABAB", Muted: "#9C9C9C", Faint: "#646464",
            Accent: "#F2F2F2", AccentHover: "#FFFFFF", AccentPressed: "#D4D4D4", AccentTint: "#24FFFFFF",
            AccentForeground: "#111111"),
        // True black surfaces for OLED panels.
        new(
            Id: "Oled",
            Background: "#000000", Chrome: "#000000", Menu: "#0A0A0A",
            Card: "#161616", Elevated: "#1E1E1E", RaisedHover: "#2A2A2A",
            Field: "#262626", FieldHover: "#323232", Overlay: "#1C1C1C",
            Text: "#F5F5F5", SecondaryText: "#CCCCCC", Description: "#ABABAB", Muted: "#9C9C9C", Faint: "#5E5E5E",
            Accent: "#F2F2F2", AccentHover: "#FFFFFF", AccentPressed: "#D4D4D4", AccentTint: "#24FFFFFF",
            AccentForeground: "#000000"),
        // Deep navy with a console-blue accent.
        new(
            Id: "Midnight",
            Background: "#0B1020", Chrome: "#070B16", Menu: "#0F1528",
            Card: "#161D33", Elevated: "#1D2540", RaisedHover: "#26304F",
            Field: "#242D4A", FieldHover: "#2E3858", Overlay: "#1A2138",
            Text: "#F3F6FB", SecondaryText: "#C7D0E0", Description: "#A6B0C4", Muted: "#95A0B6", Faint: "#5C6680",
            Accent: "#2E7CF6", AccentHover: "#4A90FF", AccentPressed: "#1F66D6", AccentTint: "#382E7CF6",
            AccentForeground: "#FFFFFF"),
        // Violet-tinted neutrals with a violet accent.
        new(
            Id: "Violet",
            Background: "#0E0D14", Chrome: "#09080E", Menu: "#13111A",
            Card: "#17151F", Elevated: "#201D2B", RaisedHover: "#292537",
            Field: "#2A2639", FieldHover: "#353049", Overlay: "#1C1928",
            Text: "#F5F5F7", SecondaryText: "#CFCBDD", Description: "#B4AFC6", Muted: "#A29DB6", Faint: "#6C677E",
            Accent: "#7C5CFC", AccentHover: "#9378FF", AccentPressed: "#6845E8", AccentTint: "#2E7C5CFC",
            AccentForeground: "#FFFFFF"),
    ];

    public static LauncherPalette Find(string? id) =>
        Array.Find(All, palette => string.Equals(palette.Id, id, StringComparison.OrdinalIgnoreCase)) ?? All[0];

    public static bool IsCustom(string? id) => string.Equals(id, CustomId, StringComparison.OrdinalIgnoreCase);

    /// <summary>Parses #RRGGBB (or RRGGBB); anything else is rejected.</summary>
    public static bool TryParseHex(string? text, out Color color)
    {
        color = default;
        var hex = text?.Trim().TrimStart('#');
        return hex is { Length: 6 } &&
            hex.All(Uri.IsHexDigit) &&
            Color.TryParse("#" + hex, out color);
    }

    /// <summary>
    /// Builds a full palette from a background and an accent: surfaces step up from
    /// the background in the same increments as Graphite, text is light, and the
    /// accent's hover, pressed, tint and foreground colours follow from the accent.
    /// </summary>
    public static LauncherPalette Derive(string? background, string? accent)
    {
        if (!TryParseHex(background, out var bg))
        {
            bg = Color.Parse(DefaultCustomBackground);
        }

        if (!TryParseHex(accent, out var ac))
        {
            ac = Color.Parse(DefaultCustomAccent);
        }

        bg = Darken(bg);
        var text = Color.Parse("#F5F5F5");
        return new LauncherPalette(
            Id: CustomId,
            Background: Hex(bg),
            Chrome: Hex(Mix(bg, Colors.Black, 0.4)),
            Menu: Hex(Mix(bg, Colors.White, 0.034)),
            Card: Hex(Mix(bg, Colors.White, 0.068)),
            Elevated: Hex(Mix(bg, Colors.White, 0.101)),
            RaisedHover: Hex(Mix(bg, Colors.White, 0.148)),
            Field: Hex(Mix(bg, Colors.White, 0.135)),
            FieldHover: Hex(Mix(bg, Colors.White, 0.181)),
            Overlay: Hex(Mix(bg, Colors.White, 0.084)),
            Text: Hex(text),
            SecondaryText: Hex(Mix(text, bg, 0.2)),
            Description: Hex(Mix(text, bg, 0.33)),
            Muted: Hex(Mix(text, bg, 0.4)),
            Faint: Hex(Mix(text, bg, 0.62)),
            Accent: Hex(ac),
            AccentHover: Hex(Mix(ac, Colors.White, 0.15)),
            AccentPressed: Hex(Mix(ac, Colors.Black, 0.15)),
            AccentTint: $"#38{ac.R:X2}{ac.G:X2}{ac.B:X2}",
            AccentForeground: Luminance(ac) > 0.45 ? "#111111" : "#FFFFFF");
    }

    /// <summary>
    /// Stretches (positive) or softens (negative) a palette: surfaces move away from or
    /// toward the background, and secondary text moves toward or away from primary text.
    /// Zero returns the palette unchanged.
    /// </summary>
    public static LauncherPalette AdjustContrast(LauncherPalette palette, int contrast)
    {
        var amount = Math.Clamp(contrast, -MaxContrast, MaxContrast) / (double)MaxContrast;
        if (amount == 0)
        {
            return palette;
        }

        var bg = Color.Parse(palette.Background);
        var text = Color.Parse(palette.Text);
        string Surface(string hex) => Hex(Extend(bg, Color.Parse(hex), 1 + (0.6 * amount)));
        string Tier(string hex) => Hex(Extend(text, Color.Parse(hex), 1 - (0.5 * amount)));
        return palette with
        {
            Menu = Surface(palette.Menu),
            Card = Surface(palette.Card),
            Elevated = Surface(palette.Elevated),
            RaisedHover = Surface(palette.RaisedHover),
            Field = Surface(palette.Field),
            FieldHover = Surface(palette.FieldHover),
            Overlay = Surface(palette.Overlay),
            SecondaryText = Tier(palette.SecondaryText),
            Description = Tier(palette.Description),
            Muted = Tier(palette.Muted),
            Faint = Tier(palette.Faint),
        };
    }

    // The point at `scale` times the distance from `origin` to `color`, clamped per channel.
    private static Color Extend(Color origin, Color color, double scale)
    {
        static byte Channel(byte from, byte to, double factor) =>
            (byte)Math.Clamp(Math.Round(from + ((to - from) * factor)), 0, 255);

        return Color.FromRgb(
            Channel(origin.R, color.R, scale),
            Channel(origin.G, color.G, scale),
            Channel(origin.B, color.B, scale));
    }

    private static Color Darken(Color color)
    {
        var max = Math.Max(color.R, Math.Max(color.G, color.B));
        if (max <= MaxCustomBackgroundChannel)
        {
            return color;
        }

        var scale = (double)MaxCustomBackgroundChannel / max;
        return Color.FromRgb(
            (byte)Math.Round(color.R * scale),
            (byte)Math.Round(color.G * scale),
            (byte)Math.Round(color.B * scale));
    }

    private static Color WithAlpha(Color color, double alpha) =>
        Color.FromArgb((byte)Math.Round(alpha * 255), color.R, color.G, color.B);

    private static Color Mix(Color from, Color to, double amount) => Color.FromRgb(
        (byte)Math.Round(from.R + (to.R - from.R) * amount),
        (byte)Math.Round(from.G + (to.G - from.G) * amount),
        (byte)Math.Round(from.B + (to.B - from.B) * amount));

    // WCAG relative luminance.
    private static double Luminance(Color color)
    {
        static double Channel(byte value)
        {
            var c = value / 255.0;
            return c <= 0.03928 ? c / 12.92 : Math.Pow((c + 0.055) / 1.055, 2.4);
        }

        return (0.2126 * Channel(color.R)) + (0.7152 * Channel(color.G)) + (0.0722 * Channel(color.B));
    }

    private static string Hex(Color color) => $"#{color.R:X2}{color.G:X2}{color.B:X2}";

    /// <summary>Resolves a token brush so code-built views share the themed instance.</summary>
    public static IBrush Brush(string key) =>
        Avalonia.Application.Current?.TryGetResource(key, null, out var value) == true && value is IBrush brush
            ? brush
            : Brushes.Transparent;

    public static void Apply(LauncherAppearance appearance)
    {
        if (Avalonia.Application.Current is not { } app)
        {
            return;
        }

        var palette = AdjustContrast(
            IsCustom(appearance.Theme)
                ? Derive(appearance.CustomBackground, appearance.CustomAccent)
                : Find(appearance.Theme),
            appearance.Contrast);
        var background = Color.Parse(palette.Background);
        var overlay = Color.Parse(palette.Overlay);
        var field = Color.Parse(palette.Field);
        var fieldHover = Color.Parse(palette.FieldHover);
        var text = Color.Parse(palette.Text);
        var accent = Color.Parse(palette.Accent);

        Paint(app, "BgBrush", background);
        Paint(app, "ChromeBrush", Color.Parse(palette.Chrome));
        Paint(app, "GameOptionsMenuBrush", Color.Parse(palette.Menu));
        Paint(app, "CardBrush", Color.Parse(palette.Card));
        Paint(app, "ElevatedBrush", Color.Parse(palette.Elevated));
        Paint(app, "RaisedHoverBrush", Color.Parse(palette.RaisedHover));
        Paint(app, "FieldBrush", field);
        Paint(app, "FieldHoverBrush", fieldHover);
        Paint(app, "OverlayBrush", overlay);
        Paint(app, "ToastBrush", Color.FromArgb(0xF0, overlay.R, overlay.G, overlay.B));
        Paint(app, "TextBrush", text);
        Paint(app, "SecondaryTextBrush", Color.Parse(palette.SecondaryText));
        Paint(app, "SettingsDescriptionBrush", Color.Parse(palette.Description));
        Paint(app, "MutedBrush", Color.Parse(palette.Muted));
        Paint(app, "FaintBrush", Color.Parse(palette.Faint));
        Paint(app, "AccentBrush", accent);
        Paint(app, "AccentHoverBrush", Color.Parse(palette.AccentHover));
        Paint(app, "AccentPressedBrush", Color.Parse(palette.AccentPressed));
        Paint(app, "AccentTintBrush", Color.Parse(palette.AccentTint));
        Paint(app, "AccentForegroundBrush", Color.Parse(palette.AccentForeground));

        Paint(app, "ComboBoxBackground", field);
        Paint(app, "ComboBoxBackgroundUnfocused", field);
        Paint(app, "ComboBoxBackgroundPointerOver", fieldHover);
        Paint(app, "ComboBoxBackgroundPressed", fieldHover);
        Paint(app, "ComboBoxForegroundFocused", text);
        Paint(app, "ComboBoxForegroundFocusedPressed", text);
        Paint(app, "ComboBoxDropDownGlyphForegroundFocused", text);
        Paint(app, "ComboBoxDropDownGlyphForegroundFocusedPressed", text);

        Tint(app, "ArtScrimBrush", background);
        Tint(app, "HintBarFadeBrush", background);
        Tint(app, "TrophyBackdropFadeBrush", Color.Parse(palette.Menu));

        // Glass is the theme's raised colour, mostly opaque by default: tinted like the
        // theme and calm enough over bright key art that labels stay readable.
        var raised = Color.Parse(palette.RaisedHover);
        var opacity = Math.Clamp(appearance.GlassOpacity, MinGlassOpacity, 100) / 100.0;
        Paint(app, "GlassBrush", WithAlpha(raised, opacity));
        Paint(app, "GlassHoverBrush", WithAlpha(fieldHover, Math.Min(1, opacity + 0.1)));
        Paint(app, "GlassPressedBrush", WithAlpha(fieldHover, Math.Min(1, opacity + 0.2)));

        app.Resources["SystemAccentColor"] = accent;
    }

    private static void Paint(Avalonia.Application app, string key, Color color)
    {
        if (app.TryGetResource(key, null, out var value) && value is SolidColorBrush brush)
        {
            brush.Color = color;
        }
    }

    // Recolours a fade while keeping each stop's alpha.
    private static void Tint(Avalonia.Application app, string key, Color color)
    {
        if (app.TryGetResource(key, null, out var value) && value is GradientBrush brush)
        {
            foreach (var stop in brush.GradientStops)
            {
                stop.Color = Color.FromArgb(stop.Color.A, color.R, color.G, color.B);
            }
        }
    }
}
