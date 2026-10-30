// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Avalonia.Controls;
using Avalonia.Data;
using Avalonia.LogicalTree;
using Avalonia.Threading;
using Avalonia.Controls.Templates;
using System.Diagnostics;

namespace SharpEmu.GUI;

/// <summary>
/// Options that only some platforms have, and the emulator switches the launcher exposes from
/// <see cref="EnvironmentSettings"/>: one row per switch, built in code with the same row, toggle
/// and combo styles as the hand-written rows, on the Options page and in per-game options.
/// </summary>
public partial class MainWindow
{
    private readonly LocalizedChoice[] _guestResolutionChoices =
    [
        LocalizedChoice.Literal("720p", "1280 x 720 (720p)"),
        LocalizedChoice.Literal("1080p", "1920 x 1080 (1080p)"),
        LocalizedChoice.Literal("1440p", "2560 x 1440 (1440p)"),
        LocalizedChoice.Literal("2160p", "3840 x 2160 (4K)"),
    ];

    private readonly List<EnvironmentSettingControl> _globalEnvironmentSettings = new();
    private readonly List<EnvironmentSettingControl> _gameEnvironmentSettings = new();
    private bool _loadingEnvironmentSettings;
    private DispatcherTimer? _clearShaderCacheTimer;

    /// <summary>One emulator switch and the control that edits it.</summary>
    private sealed class EnvironmentSettingControl(
        EnvironmentSetting setting,
        ToggleSwitch? toggle,
        ComboBox? combo,
        List<LocalizedChoice>? choices)
    {
        public EnvironmentSetting Setting { get; } = setting;

        public List<LocalizedChoice>? Choices => choices;

        public void Load(IEnumerable<string> entries)
        {
            if (toggle is not null)
            {
                toggle.IsChecked = EnvironmentSettings.IsEnabled(entries, Setting);
                return;
            }

            var value = EnvironmentSettings.GetChoice(entries, Setting);
            var choice = choices!.FirstOrDefault(c => string.Equals(c.Value, value, StringComparison.OrdinalIgnoreCase));
            if (choice is null)
            {
                // A value written by hand or by a newer build still shows, and survives a save.
                choice = LocalizedChoice.Literal(value, value);
                choices!.Add(choice);
                combo!.ItemsSource = choices.ToArray();
            }

            combo!.SelectedItem = choice;
        }

        public void Store(List<string> entries)
        {
            if (toggle is not null)
            {
                EnvironmentSettings.SetEnabled(entries, Setting, toggle.IsChecked == true);
            }
            else if (combo!.SelectedItem is LocalizedChoice choice)
            {
                EnvironmentSettings.SetChoice(entries, Setting, choice.Value);
            }
        }
    }

    /// <summary>
    /// Hides rows whose emulator-side feature does not exist on macOS: crash dumps (Windows x64
    /// only), RenderDoc and the Vulkan loader/validation switches (Metal has no Vulkan), and the
    /// refresh-rate picker (the macOS window follows the display's own mode).
    /// </summary>
    private void HideUnsupportedPlatformOptions()
    {
        if (!OperatingSystem.IsMacOS())
        {
            // The caches under user/pipeline_cache belong to the Metal backend.
            ShaderCacheFolderRow.IsVisible = false;
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

    private void BuildEnvironmentSettingRows()
    {
        var globalHosts = new Dictionary<EnvironmentSettingGroup, Panel>
        {
            [EnvironmentSettingGroup.Performance] = OptionsPerformanceHost,
            [EnvironmentSettingGroup.Compatibility] = OptionsCompatibilityHost,
            [EnvironmentSettingGroup.Audio] = OptionsAudioHost,
            [EnvironmentSettingGroup.Debug] = OptionsDebugExtraHost,
        };
        var gameHosts = new Dictionary<EnvironmentSettingGroup, Panel>
        {
            [EnvironmentSettingGroup.Performance] = GamePerformanceHost,
            [EnvironmentSettingGroup.Compatibility] = GameCompatibilityHost,
            [EnvironmentSettingGroup.Audio] = GameAudioHost,
            [EnvironmentSettingGroup.Debug] = GameDebugExtraHost,
        };

        foreach (var setting in EnvironmentSettings.All)
        {
            if (setting.MacOnly && !OperatingSystem.IsMacOS())
            {
                continue;
            }

            var global = CreateEnvironmentSettingRow(setting, "", () =>
            {
                if (!_loadingEnvironmentSettings)
                {
                    var control = _globalEnvironmentSettings.First(c => c.Setting == setting);
                    control.Store(_settings.EnvironmentToggles);
                }
            }, out var globalControl);
            globalHosts[setting.Group].Children.Add(global);
            _globalEnvironmentSettings.Add(globalControl);

            var perGame = CreateEnvironmentSettingRow(setting, "Game", PersistOpenGameSettings, out var gameControl);
            gameHosts[setting.Group].Children.Add(perGame);
            _gameEnvironmentSettings.Add(gameControl);
        }

        GuestResolutionBox.ItemsSource = _guestResolutionChoices;
        GameGuestResolutionBox.ItemsSource = _guestResolutionChoices;
        GuestResolutionBox.SelectionChanged += (_, _) =>
            _settings.GuestDisplayResolution = SelectedComboText(GuestResolutionBox, "1080p");
        GameGuestResolutionBox.SelectionChanged += (_, _) => PersistOpenGameSettings();
        OpenShaderCacheButton.Click += (_, _) => OpenFolder(EmulatorShaderCacheRoot());
        ClearShaderCacheButton.Click += (_, _) => OnClearShaderCacheClicked();
    }

    private SettingRow CreateEnvironmentSettingRow(
        EnvironmentSetting setting,
        string namePrefix,
        Action changed,
        out EnvironmentSettingControl control)
    {
        var row = new SettingRow { Name = $"{namePrefix}{setting.Id}Row" };
        row.Classes.Add("optionRow");
        row.Bind(SettingRow.LabelProperty, LocalizedBinding($"Setting.{setting.Id}.Label"));
        row.Bind(SettingRow.DescriptionProperty, LocalizedBinding($"Setting.{setting.Id}.Desc"));

        if (setting.Kind == EnvironmentSettingKind.Toggle)
        {
            var toggle = new ToggleSwitch { Name = $"{namePrefix}{setting.Id}Toggle" };
            toggle.Classes.Add("optionToggle");
            toggle.IsCheckedChanged += (_, _) => changed();
            row.Content = toggle;
            control = new EnvironmentSettingControl(setting, toggle, null, null);
            return row;
        }

        var choices = setting.Choices!
            .Select(choice => choice.Label is { } label
                ? LocalizedChoice.Literal(choice.Value, label)
                : LocalizedChoice.FromKey(choice.Value, $"Setting.{setting.Id}.{choice.Value}"))
            .ToList();
        foreach (var choice in choices)
        {
            choice.Refresh(Localization.Instance);
        }

        var combo = new ComboBox
        {
            Name = $"{namePrefix}{setting.Id}Box",
            ItemsSource = choices.ToArray(),
            ItemTemplate = (IDataTemplate)this.FindResource("LocalizedChoiceTemplate")!,
            HorizontalContentAlignment = Avalonia.Layout.HorizontalAlignment.Left,
            VerticalAlignment = Avalonia.Layout.VerticalAlignment.Center,
        };
        combo.Classes.Add("optionValue");
        combo.SelectionChanged += (_, _) => changed();
        row.Content = combo;
        control = new EnvironmentSettingControl(setting, null, combo, choices);
        return row;
    }

    private static Binding LocalizedBinding(string key) =>
        new($"[{key}]") { Source = Localization.Instance };

    private void RefreshEnvironmentSettingChoices()
    {
        RefreshChoices(_guestResolutionChoices);
        foreach (var control in _globalEnvironmentSettings.Concat(_gameEnvironmentSettings))
        {
            if (control.Choices is { } choices)
            {
                RefreshChoices(choices);
            }
        }
    }

    private void ApplyEnvironmentSettingsToControls()
    {
        _loadingEnvironmentSettings = true;
        try
        {
            foreach (var control in _globalEnvironmentSettings)
            {
                control.Load(_settings.EnvironmentToggles);
            }

            GuestResolutionBox.SelectedItem = FindChoice(
                _guestResolutionChoices, _settings.GuestDisplayResolution, "1080p");
        }
        finally
        {
            _loadingEnvironmentSettings = false;
        }
    }

    private void LoadGameEnvironmentSettings(IReadOnlyList<string> entries, string guestResolution)
    {
        foreach (var control in _gameEnvironmentSettings)
        {
            control.Load(entries);
        }

        GameGuestResolutionBox.SelectedItem = FindChoice(_guestResolutionChoices, guestResolution, "1080p");
    }

    private void StoreGameEnvironmentSettings(List<string> entries)
    {
        foreach (var control in _gameEnvironmentSettings)
        {
            control.Store(entries);
        }
    }

    private string EmulatorDataRoot()
    {
        var exeDirectory = Path.GetDirectoryName(_emulatorExePath);
        if (string.IsNullOrEmpty(exeDirectory))
        {
            exeDirectory = AppContext.BaseDirectory;
        }

        return SharpEmu.Logging.SharpEmuPaths.ResolveDataRoot(exeDirectory);
    }

    /// <summary>Compiled shaders, pipeline archives and prewarm lists, one folder per title.</summary>
    private string EmulatorShaderCacheRoot() => Path.Combine(EmulatorDataRoot(), "user", "pipeline_cache");

    private void OpenFolder(string path)
    {
        try
        {
            Directory.CreateDirectory(path);
            Process.Start(new ProcessStartInfo
            {
                FileName = OperatingSystem.IsWindows() ? "explorer.exe" : OperatingSystem.IsMacOS() ? "open" : "xdg-open",
                Arguments = $"\"{path}\"",
                UseShellExecute = false,
            });
        }
        catch (Exception ex)
        {
            AppendConsoleLine(
                Localization.Instance.Format("Status.CouldNotOpenFolder", ex.Message),
                WarningLineBrush);
        }
    }

    // The first click arms the button and the second, within a few seconds, clears: the cache
    // can take minutes to build again.
    private void OnClearShaderCacheClicked()
    {
        if (_isRunning)
        {
            AppendConsoleLine(Localization.Instance.Get("Setting.ShaderCacheFolder.Running"), WarningLineBrush);
            return;
        }

        if (_clearShaderCacheTimer is null)
        {
            SetClearShaderCacheLabel("Setting.ShaderCacheFolder.Confirm");
            _clearShaderCacheTimer = new DispatcherTimer { Interval = TimeSpan.FromSeconds(4) };
            _clearShaderCacheTimer.Tick += (_, _) => DisarmClearShaderCache();
            _clearShaderCacheTimer.Start();
            return;
        }

        DisarmClearShaderCache();
        var root = EmulatorShaderCacheRoot();
        _ = Task.Run(() =>
        {
            string message;
            try
            {
                if (Directory.Exists(root))
                {
                    Directory.Delete(root, recursive: true);
                }

                message = Localization.Instance.Get("Setting.ShaderCacheFolder.Cleared");
            }
            catch (Exception ex)
            {
                message = Localization.Instance.Format("Setting.ShaderCacheFolder.ClearFailed", ex.Message);
            }

            Dispatcher.UIThread.Post(() => AppendConsoleLine(message, DimLineBrush));
        });
    }

    private void DisarmClearShaderCache()
    {
        _clearShaderCacheTimer?.Stop();
        _clearShaderCacheTimer = null;
        SetClearShaderCacheLabel("Setting.ShaderCacheFolder.Clear");
    }

    private void SetClearShaderCacheLabel(string key) =>
        ClearShaderCacheButtonLabel.Bind(TextBlock.TextProperty, LocalizedBinding(key));
}
