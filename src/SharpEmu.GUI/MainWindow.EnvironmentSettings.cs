// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Avalonia.Controls;
using Avalonia.LogicalTree;

namespace SharpEmu.GUI;

/// <summary>Options that only some platforms have, and the emulator switches the launcher exposes.</summary>
public partial class MainWindow
{
    /// <summary>
    /// Hides rows whose emulator-side feature does not exist on macOS: crash dumps (Windows x64
    /// only), RenderDoc and the Vulkan loader/validation switches (Metal has no Vulkan), and the
    /// refresh-rate picker (the macOS window follows the display's own mode).
    /// </summary>
    private void HideUnsupportedPlatformOptions()
    {
        if (!OperatingSystem.IsMacOS())
        {
            return;
        }

        Control[] unsupported =
        [
            CrashDumpRow,
            RefreshRateRow,
            EnvRenderDocRow,
            EnvVkValidationRow,
            EnvDisableVkImplicitLayersToggle,
            GameCrashDumpRow,
            GameRefreshRateRow,
            GameEnvRenderDocToggle,
            GameEnvVkValidationToggle,
            GameEnvDisableVkImplicitLayersToggle,
        ];
        foreach (var control in unsupported)
        {
            (control as SettingRow ?? control.FindLogicalAncestorOfType<SettingRow>())!.IsVisible = false;
        }
    }
}
