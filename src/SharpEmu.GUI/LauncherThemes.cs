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
/// The launcher themes. Applying one repaints the brush instances declared in
/// Themes/Tokens.axaml in place, so every StaticResource user follows the change
/// without a restart.
/// </summary>
internal static class LauncherThemes
{
    public const string DefaultId = "Graphite";

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

    /// <summary>Resolves a token brush so code-built views share the themed instance.</summary>
    public static IBrush Brush(string key) =>
        Avalonia.Application.Current?.TryGetResource(key, null, out var value) == true && value is IBrush brush
            ? brush
            : Brushes.Transparent;

    public static void Apply(string? id)
    {
        if (Avalonia.Application.Current is not { } app)
        {
            return;
        }

        var palette = Find(id);
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

        app.Resources["SystemAccentColor"] = accent;
        app.Resources["AccentGlowShadow"] = new BoxShadows(
            new BoxShadow { OffsetY = 1, Color = Color.FromArgb(0x59, 0xFF, 0xFF, 0xFF), IsInset = true },
            [new BoxShadow { OffsetY = 8, Blur = 28, Spread = -4, Color = Color.FromArgb(0x80, accent.R, accent.G, accent.B) }]);
        app.Resources["SegmentUnderlineShadow"] = new BoxShadows(new BoxShadow
        {
            OffsetY = 6,
            Spread = -3,
            Color = accent,
        });
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
