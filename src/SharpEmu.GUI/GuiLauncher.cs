// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Avalonia;
using Avalonia.Media;
using SharpEmu.Logging;

namespace SharpEmu.GUI;

/// <summary>
/// Entry point for the desktop frontend, hosted by the SharpEmu executable
/// when it is started without command-line arguments.
/// </summary>
public static class GuiLauncher
{
    public static int Run()
    {
        try
        {
            BuildAvaloniaApp().StartWithClassicDesktopLifetime(Array.Empty<string>());
            return 0;
        }
        catch (Exception ex)
        {
            WriteCrashLog(ex);
            throw;
        }
    }

    public static AppBuilder BuildAvaloniaApp()
        => AppBuilder.Configure<App>()
#if SHARPEMU_GUI_MACOS
            // Avalonia.Desktop's UsePlatformDetect would pull in the Win32
            // and X11 backends; macOS only needs the native one.
            .UseAvaloniaNative()
            .UseSkia()
            .UseHarfBuzz()
#else
            .UsePlatformDetect()
#endif
            .WithInterFont()
            // Popups, menus and tooltips are their own top levels and don't inherit the
            // window's font, so the bundled Inter is made the app-wide default.
            .With(new FontManagerOptions { DefaultFamilyName = "fonts:Inter#Inter" })
            .LogToTrace();

    private static void WriteCrashLog(Exception ex)
    {
        try
        {
            var crashLogPath = SharpEmuPaths.LogsPath("gui-crash.log");
            Directory.CreateDirectory(Path.GetDirectoryName(crashLogPath)!);
            File.AppendAllText(
                crashLogPath,
                $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] {ex}{Environment.NewLine}{Environment.NewLine}");
        }
        catch (Exception)
        {
            // Crash logging is best-effort.
        }
    }
}
