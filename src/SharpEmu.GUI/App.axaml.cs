// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using Avalonia.Media;
using Avalonia.Platform;

namespace SharpEmu.GUI;

public partial class App : Avalonia.Application
{
    // Platform accent fallback and the macOS system text face, as named in Tokens.axaml.
    private static readonly Color FallbackAccent = Color.Parse("#7C5CFC");
    private const string MacSystemFont = ".AppleSystemUIFont";

    public override void Initialize()
    {
        AvaloniaXamlLoader.Load(this);
        UseSystemFont();
    }

    public override void OnFrameworkInitializationCompleted()
    {
        FollowPlatformAccent();

        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            desktop.ShutdownMode = Avalonia.Controls.ShutdownMode.OnMainWindowClose;
            desktop.MainWindow = new MainWindow();
        }

        base.OnFrameworkInitializationCompleted();
    }

    /// <summary>
    /// Uses the macOS system font (SF Pro) for text when the renderer can resolve it; Inter
    /// stays everywhere else and as the fallback.
    /// </summary>
    private void UseSystemFont()
    {
        if (!OperatingSystem.IsMacOS() ||
            !FontManager.Current.TryGetGlyphTypeface(new Typeface(MacSystemFont), out _))
        {
            return;
        }

        Resources["AppFontFamily"] = new FontFamily($"{MacSystemFont}, Inter, sans-serif");
    }

    /// <summary>
    /// Keeps the accent brushes on the operating system accent colour (the macOS "Accent
    /// colour" preference), falling back to the launcher purple when the platform has none.
    /// The brushes are updated in place so every existing user of them follows.
    /// </summary>
    private void FollowPlatformAccent()
    {
        if (PlatformSettings is not { } platform)
        {
            return;
        }

        ApplyAccent(platform.GetColorValues());
        platform.ColorValuesChanged += (_, values) => ApplyAccent(values);
    }

    private void ApplyAccent(PlatformColorValues values)
    {
        var accent = values.AccentColor1;
        if (accent.A == 0)
        {
            accent = FallbackAccent;
        }

        SetBrush("AccentBrush", accent);
        SetBrush("AccentHoverBrush", Lighten(accent, 0.14));
        SetBrush("AccentPressedBrush", Lighten(accent, -0.12));
        SetBrush("AccentSoftBrush", Color.FromArgb(0x33, accent.R, accent.G, accent.B));
        SetBrush("AccentForegroundBrush", RelativeLuminance(accent) > 0.62 ? Color.Parse("#1D1D1F") : Colors.White);
        Resources["SystemAccentColor"] = accent;
        Resources["SystemAccentColorLight1"] = Lighten(accent, 0.14);
        Resources["SystemAccentColorLight2"] = Lighten(accent, 0.28);
        Resources["SystemAccentColorLight3"] = Lighten(accent, 0.42);
        Resources["SystemAccentColorDark1"] = Lighten(accent, -0.12);
        Resources["SystemAccentColorDark2"] = Lighten(accent, -0.24);
        Resources["SystemAccentColorDark3"] = Lighten(accent, -0.36);
    }

    private void SetBrush(string key, Color color)
    {
        if (this.TryGetResource(key, ActualThemeVariant, out var resource) &&
            resource is SolidColorBrush brush)
        {
            brush.Color = color;
        }
    }

    // Mixes towards white (amount > 0) or black (amount < 0).
    private static Color Lighten(Color color, double amount)
    {
        var target = amount >= 0 ? 255.0 : 0.0;
        var t = Math.Abs(amount);
        byte Mix(byte channel) => (byte)Math.Round(channel + (target - channel) * t);
        return Color.FromArgb(color.A, Mix(color.R), Mix(color.G), Mix(color.B));
    }

    private static double RelativeLuminance(Color color)
    {
        static double Channel(byte value)
        {
            var c = value / 255.0;
            return c <= 0.03928 ? c / 12.92 : Math.Pow((c + 0.055) / 1.055, 2.4);
        }

        return 0.2126 * Channel(color.R) + 0.7152 * Channel(color.G) + 0.0722 * Channel(color.B);
    }
}
