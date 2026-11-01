// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Logging;

namespace SharpEmu.GuiHost;

/// <summary>
/// Entry point of the native arm64 frontend on macOS. Opens the same GUI the x64 SharpEmu
/// opens without arguments; games still run in the x64 SharpEmu, which the GUI finds one
/// directory up (inside the bundle) or in the sibling osx-x64 build output.
/// </summary>
public static class GuiHostMain
{
    [STAThread]
    public static int Main()
    {
        if (!OperatingSystem.IsMacOS())
        {
            Console.Error.WriteLine("[GUIHOST][ERROR] The arm64 GUI host runs only on macOS.");
            return 1;
        }

        // Outside a bundle this host's output directory is not the emulator's, so data
        // (library, settings, saves) would split in two. Pin the data root to the emulator's
        // before anything reads SharpEmuPaths; the child inherits it.
        if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable(SharpEmuPaths.DataRootVariable)) &&
            !AppContext.BaseDirectory.Contains(".app/Contents/", StringComparison.OrdinalIgnoreCase) &&
            FindDevEmulatorDirectory() is { } emulatorDirectory)
        {
            Environment.SetEnvironmentVariable(SharpEmuPaths.DataRootVariable, emulatorDirectory);
        }

#pragma warning disable CA1416 // macOS only: Main returns early anywhere else.
        SharpEmu.Platform.MacOS.MacPlatform.RegisterLauncher();
#pragma warning restore CA1416
        return SharpEmu.GUI.GuiLauncher.Run();
    }

    // artifacts/bin/<Configuration>/<tfm>/osx-arm64/ sits next to .../osx-x64/.
    private static string? FindDevEmulatorDirectory()
    {
        var directory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "osx-x64"));
        return File.Exists(Path.Combine(directory, "SharpEmu")) ? directory : null;
    }
}
