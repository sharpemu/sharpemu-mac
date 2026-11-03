// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.ObjectModel;
using System.Globalization;
using System.Runtime.InteropServices;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Templates;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using Avalonia.Threading;
using SharpEmu.Libs.VideoOut.Overlay;
using SharpEmu.Logging;

namespace SharpEmu.GUI;

/// <summary>
/// The Options "Overlay" section: style, layout presets and widget settings of the in-game
/// performance overlay, with a live preview. The layout files are global (one set for every game),
/// so this edits them directly through <see cref="OverlayLayoutStore"/>, the same files a running
/// game watches and reloads. Widget settings rows are generated from each widget's schema.
/// </summary>
public partial class MainWindow
{
    private const int OverlayPreviewIntervalMs = 80;
    private const int OverlaySaveDelayMs = 300;

    private readonly ObservableCollection<LocalizedChoice> _overlayPresetChoices =
        [LocalizedChoice.FromKey("", "Options.OverlayPreset.Last")];

    private readonly ObservableCollection<LocalizedChoice> _overlayEditPresetChoices = [];
    private readonly ObservableCollection<LocalizedChoice> _overlayWidgetChoices = [];

    private readonly LocalizedChoice[] _overlayThemeChoices =
    [
        LocalizedChoice.FromKey(nameof(OverlayTheme.Graphite), "Overlay.Theme.Graphite"),
        LocalizedChoice.FromKey(nameof(OverlayTheme.Oled), "Overlay.Theme.Oled"),
        LocalizedChoice.FromKey(nameof(OverlayTheme.Glass), "Overlay.Theme.Glass"),
    ];

    private readonly LocalizedChoice[] _overlayFontChoices =
    [
        LocalizedChoice.FromKey(nameof(OverlayFont.Inter), "Overlay.Font.Inter"),
        LocalizedChoice.FromKey(nameof(OverlayFont.System), "Overlay.Font.System"),
        LocalizedChoice.FromKey(nameof(OverlayFont.JetBrainsMono), "Overlay.Font.JetBrainsMono"),
    ];

    private LocalizedChoice[] _overlayAddChoices = [];
    private OverlayLayoutStore? _overlayStore;
    private OverlayHost? _overlayHost;
    private readonly Queue<Bitmap> _overlayBitmaps = new();
    private DispatcherTimer? _overlayPreviewTimer;
    private DispatcherTimer? _overlaySaveTimer;
    private bool _overlayEditorActive;
    private bool _overlayRefreshing;
    private bool _overlaySavePending;
    private string? _overlayWidgetId;

    private OverlayLayoutStore OverlayStore => _overlayStore ??= new OverlayLayoutStore(SharpEmuPaths.UserPath("overlay"));

    private OverlayLayoutModel? OverlayModel => _overlayHost?.Model;

    // ---- launch settings: the preset choices ----

    /// <summary>Lists "last used", the built-in presets and the custom ones; keeps <paramref name="stored"/> visible if it is gone.</summary>
    private void RefreshOverlayPresetChoices(string? stored)
    {
        var scratch = new OverlayLayoutModel();
        OverlayStore.LoadInto(scratch);
        while (_overlayPresetChoices.Count > 1)
        {
            _overlayPresetChoices.RemoveAt(_overlayPresetChoices.Count - 1);
        }

        foreach (var name in scratch.AllPresetNames())
        {
            _overlayPresetChoices.Add(LocalizedChoice.Literal(name, name));
        }

        if (!string.IsNullOrWhiteSpace(stored) &&
            !_overlayPresetChoices.Any(choice => string.Equals(choice.Value, stored, StringComparison.OrdinalIgnoreCase)))
        {
            _overlayPresetChoices.Add(LocalizedChoice.Literal(stored, stored));
        }
    }

    // ---- wiring ----

    private void WireOverlayEditor()
    {
        var template = (IDataTemplate?)this.FindResource("LocalizedChoiceTemplate");
        foreach (var box in new[] { OverlayPresetEditBox, OverlayThemeBox, OverlayFontBox, OverlayWidgetSelectBox, OverlayWidgetAddBox })
        {
            box.ItemTemplate = template;
        }

        OverlayPresetEditBox.ItemsSource = _overlayEditPresetChoices;
        OverlayThemeBox.ItemsSource = _overlayThemeChoices;
        OverlayFontBox.ItemsSource = _overlayFontChoices;
        OverlayWidgetSelectBox.ItemsSource = _overlayWidgetChoices;

        OverlayPresetEditBox.SelectionChanged += (_, _) =>
        {
            if (!_overlayRefreshing && OverlayModel is { } model &&
                OverlayPresetEditBox.SelectedItem is LocalizedChoice choice &&
                !string.Equals(choice.Value, model.PresetName, StringComparison.Ordinal))
            {
                model.ApplyPreset(choice.Value);
                _overlayWidgetId = null;
                OverlayChanged(flushNow: true);
                RefreshOverlayEditor();
            }
        };
        OverlaySaveAsButton.Click += async (_, _) => await SaveOverlayPresetAsync();
        OverlayResetButton.Click += (_, _) =>
        {
            if (OverlayModel is { } model)
            {
                model.ResetLayout();
                OverlayChanged(flushNow: true);
                RefreshOverlayEditor();
            }
        };
        OverlayDeleteButton.Click += (_, _) =>
        {
            if (OverlayModel is { } model && model.DeleteCustom(model.PresetName))
            {
                _overlayWidgetId = null;
                OverlayChanged(flushNow: true);
                RefreshOverlayEditor();
            }
        };

        OverlayThemeBox.SelectionChanged += (_, _) =>
        {
            if (!_overlayRefreshing && OverlayModel is { } model && OverlayThemeBox.SelectedItem is LocalizedChoice choice &&
                Enum.TryParse<OverlayTheme>(choice.Value, out var theme))
            {
                model.Globals.Theme = theme;
                OverlayChanged();
            }
        };
        OverlayFontBox.SelectionChanged += (_, _) =>
        {
            if (!_overlayRefreshing && OverlayModel is { } model && OverlayFontBox.SelectedItem is LocalizedChoice choice &&
                Enum.TryParse<OverlayFont>(choice.Value, out var font))
            {
                model.Globals.Font = font;
                OverlayChanged();
            }
        };
        WireOverlaySlider(OverlayOpacitySlider, OverlayOpacitySliderValue, "{0:0}%",
            (globals, value) => globals.Opacity = (float)value / 100f);
        WireOverlaySlider(OverlayScaleSlider, OverlayScaleSliderValue, "{0:0}%",
            (globals, value) => globals.Scale = (float)value / 100f);
        WireOverlaySlider(OverlayFontSizeSlider, OverlayFontSizeSliderValue, "{0:0}",
            (globals, value) => globals.FontSize = (float)value);

        OverlayWidgetSelectBox.SelectionChanged += (_, _) =>
        {
            if (!_overlayRefreshing && OverlayWidgetSelectBox.SelectedItem is LocalizedChoice choice)
            {
                _overlayWidgetId = choice.Value;
                BuildOverlayWidgetSettings();
            }
        };
        OverlayWidgetRemoveButton.Click += (_, _) =>
        {
            if (OverlayModel is { } model && model.Find(_overlayWidgetId ?? string.Empty) is { } widget)
            {
                var index = model.Widgets.IndexOf(widget);
                model.Remove(widget);
                _overlayWidgetId = model.Widgets.Count == 0 ? null : model.Widgets[Math.Min(index, model.Widgets.Count - 1)].Id;
                OverlayChanged();
                RefreshOverlayEditor();
            }
        };
        _overlayAddChoices = [.. WidgetCatalog.All.Select(spec =>
            LocalizedChoice.Literal(spec.Key, OverlayText($"Overlay.Widget.{spec.Key}.Title", spec.Title)))];
        OverlayWidgetAddBox.ItemsSource = _overlayAddChoices;
        OverlayWidgetAddBox.SelectedIndex = 0;
        OverlayWidgetAddButton.Click += (_, _) =>
        {
            if (OverlayModel is { } model && OverlayWidgetAddBox.SelectedItem is LocalizedChoice choice &&
                WidgetCatalog.TryGet(choice.Value, out var spec))
            {
                _overlayWidgetId = model.AddWidget(spec.Type).Id;
                OverlayChanged();
                RefreshOverlayEditor();
            }
        };

        // The frame keeps the 16:9 shape of the screen the overlay is laid out for.
        OverlayPreviewFrame.SizeChanged += (_, e) =>
        {
            var height = Math.Floor((e.NewSize.Width - 2) * 9.0 / 16.0) + 2;
            if (double.IsNaN(OverlayPreviewFrame.Height) || Math.Abs(OverlayPreviewFrame.Height - height) > 0.5)
            {
                OverlayPreviewFrame.Height = height;
            }
        };
    }

    private void WireOverlaySlider(Slider slider, TextBlock valueText, string format, Action<OverlayGlobals, double> apply)
    {
        slider.ValueChanged += (_, _) =>
        {
            valueText.Text = string.Format(CultureInfo.InvariantCulture, format, slider.Value);
            if (!_overlayRefreshing && OverlayModel is { } model)
            {
                apply(model.Globals, slider.Value);
                OverlayChanged();
            }
        };
    }

    // ---- activation: the section is on screen ----

    private void UpdateOverlayEditorActive()
    {
        var panels = OptionsSectionPanels();
        var active = OptionsPage.IsVisible &&
            _optionsSectionIndex >= 0 && _optionsSectionIndex < panels.Length &&
            ReferenceEquals(panels[_optionsSectionIndex], OptionsOverlayPanel);
        if (active == _overlayEditorActive)
        {
            return;
        }

        _overlayEditorActive = active;
        if (!active)
        {
            _overlayPreviewTimer?.Stop();
            FlushOverlaySave();
            return;
        }

        EnsureOverlayHost();
        if (!_overlaySavePending)
        {
            // Pick up what a running game or an earlier visit changed.
            OverlayStore.LoadInto(OverlayModel!);
        }

        RefreshOverlayEditor();
        _overlayPreviewTimer ??= new DispatcherTimer(
            TimeSpan.FromMilliseconds(OverlayPreviewIntervalMs), DispatcherPriority.Background, (_, _) => RenderOverlayPreview());
        _overlayPreviewTimer.Start();
        Dispatcher.UIThread.Post(RenderOverlayPreview, DispatcherPriority.Loaded);
    }

    private void EnsureOverlayHost()
    {
        if (_overlayHost is not null)
        {
            return;
        }

        SharpEmu.Fonts.Abstractions.IFontFactory? fonts = null;
        try
        {
            fonts = new SharpEmu.Fonts.Skia.SkiaFontFactory();
        }
        catch (Exception)
        {
            // The overlay falls back to its built-in font.
        }

        _overlayHost = new OverlayHost(new SimulatedOverlayDataSource("ds"), fonts, static () => true);
    }

    private void ShutDownOverlayEditor()
    {
        _overlayPreviewTimer?.Stop();
        _overlaySaveTimer?.Stop();
        FlushOverlaySave();
        _overlayHost?.Dispose();
        _overlayHost = null;
        OverlayPreviewImage.Source = null;
        while (_overlayBitmaps.Count > 0)
        {
            _overlayBitmaps.Dequeue().Dispose();
        }
    }

    // ---- saving ----

    private void OverlayChanged(bool flushNow = false)
    {
        OverlayModel?.Touch();
        _overlaySavePending = true;
        if (flushNow)
        {
            FlushOverlaySave();
            return;
        }

        _overlaySaveTimer ??= new DispatcherTimer(
            TimeSpan.FromMilliseconds(OverlaySaveDelayMs), DispatcherPriority.Background, (_, _) => FlushOverlaySave());
        _overlaySaveTimer.Stop();
        _overlaySaveTimer.Start();
    }

    private void FlushOverlaySave()
    {
        _overlaySaveTimer?.Stop();
        if (!_overlaySavePending || OverlayModel is not { } model)
        {
            return;
        }

        _overlaySavePending = false;
        if (!OverlayStore.Save(model))
        {
            AppendConsoleLine("[SETTINGS][WARN] The overlay layout could not be saved.", WarningLineBrush);
        }
    }

    // ---- editor state -> controls ----

    private static string OverlayText(string key, string fallback)
    {
        var text = Localization.Instance.Get(key);
        return string.Equals(text, key, StringComparison.Ordinal) ? fallback : text;
    }

    private void RefreshOverlayEditor()
    {
        if (OverlayModel is not { } model)
        {
            return;
        }

        _overlayRefreshing = true;
        try
        {
            var globals = model.Globals;
            OverlayThemeBox.SelectedItem = _overlayThemeChoices.FirstOrDefault(c => c.Value == globals.Theme.ToString());
            OverlayFontBox.SelectedItem = _overlayFontChoices.FirstOrDefault(c => c.Value == globals.Font.ToString());
            OverlayOpacitySlider.Value = Math.Round(globals.Opacity * 100.0 / 5.0) * 5.0;
            OverlayScaleSlider.Value = Math.Round(globals.Scale * 100.0 / 5.0) * 5.0;
            OverlayFontSizeSlider.Value = Math.Round(globals.FontSize);

            _overlayEditPresetChoices.Clear();
            foreach (var name in model.AllPresetNames())
            {
                _overlayEditPresetChoices.Add(LocalizedChoice.Literal(name, name));
            }

            if (!_overlayEditPresetChoices.Any(c => c.Value == model.PresetName))
            {
                _overlayEditPresetChoices.Add(LocalizedChoice.Literal(model.PresetName, model.PresetName));
            }

            OverlayPresetEditBox.SelectedItem = _overlayEditPresetChoices.First(c => c.Value == model.PresetName);
            OverlayPresetEditRow.Description = Localization.Instance.Get(
                model.Modified ? "Overlay.Preset.Edited" : "Overlay.Preset.Desc");
            OverlayDeleteRow.IsVisible = model.CustomPresets.ContainsKey(model.PresetName);

            _overlayWidgetChoices.Clear();
            var seen = new Dictionary<OverlayWidgetType, int>();
            var totals = new Dictionary<OverlayWidgetType, int>();
            foreach (var widget in model.Widgets)
            {
                totals[widget.Type] = totals.GetValueOrDefault(widget.Type) + 1;
            }

            foreach (var widget in model.Widgets)
            {
                var title = OverlayText($"Overlay.Widget.{widget.Spec.Key}.Title", widget.Spec.Title);
                var number = seen[widget.Type] = seen.GetValueOrDefault(widget.Type) + 1;
                _overlayWidgetChoices.Add(LocalizedChoice.Literal(widget.Id, totals[widget.Type] > 1 ? $"{title} ({number})" : title));
            }

            var selected = _overlayWidgetChoices.FirstOrDefault(c => c.Value == _overlayWidgetId) ?? _overlayWidgetChoices.FirstOrDefault();
            _overlayWidgetId = selected?.Value;
            OverlayWidgetSelectBox.SelectedItem = selected;
            var hasWidgets = selected is not null;
            OverlayWidgetSelectRow.IsVisible = hasWidgets;
            OverlayWidgetRemoveRow.IsVisible = hasWidgets;
        }
        finally
        {
            _overlayRefreshing = false;
        }

        BuildOverlayWidgetSettings();
    }

    // The selected widget's settings, one row per entry of its schema.
    private void BuildOverlayWidgetSettings()
    {
        OverlayWidgetSettingsHost.Children.Clear();
        var widget = OverlayModel?.Find(_overlayWidgetId ?? string.Empty);
        OverlayWidgetSettingsHeader.IsVisible = widget is not null;
        OverlayWidgetSettingsGroup.IsVisible = widget is not null;
        if (widget is null)
        {
            return;
        }

        var spec = widget.Spec;
        var title = OverlayText($"Overlay.Widget.{spec.Key}.Title", spec.Title);
        OverlayWidgetSettingsHeader.Text = Localization.Instance.Format("Overlay.Group.WidgetSettings", title);
        var template = (IDataTemplate?)this.FindResource("LocalizedChoiceTemplate");
        for (var i = 0; i < spec.Settings.Length; i++)
        {
            var index = i;
            var setting = spec.Settings[i];
            var row = new SettingRow
            {
                Label = OverlayText($"Overlay.Setting.{spec.Key}.{setting.Key}", setting.Label),
            };
            row.Classes.Add("optionRow");
            switch (setting.Kind)
            {
                case WidgetSettingKind.Toggle:
                {
                    var toggle = new ToggleSwitch { IsChecked = widget.Bool(index) };
                    toggle.Classes.Add("optionToggle");
                    toggle.IsCheckedChanged += (_, _) => SetOverlayWidgetValue(widget, index, toggle.IsChecked == true ? 1f : 0f);
                    row.Content = toggle;
                    break;
                }

                case WidgetSettingKind.Choice:
                {
                    var choices = new LocalizedChoice[setting.Options.Length];
                    for (var o = 0; o < choices.Length; o++)
                    {
                        var option = setting.Options[o];
                        choices[o] = LocalizedChoice.Literal(
                            o.ToString(CultureInfo.InvariantCulture),
                            OverlayText($"Overlay.Option.{spec.Key}.{setting.Key}.{option.Id}", option.Label));
                    }

                    var box = new ComboBox
                    {
                        ItemsSource = choices,
                        ItemTemplate = template,
                        HorizontalContentAlignment = Avalonia.Layout.HorizontalAlignment.Left,
                        VerticalAlignment = Avalonia.Layout.VerticalAlignment.Center,
                        SelectedIndex = Math.Clamp(widget.Choice(index), 0, Math.Max(0, choices.Length - 1)),
                    };
                    box.Classes.Add("optionValue");
                    box.SelectionChanged += (_, _) =>
                    {
                        if (box.SelectedIndex >= 0)
                        {
                            SetOverlayWidgetValue(widget, index, box.SelectedIndex);
                        }
                    };
                    row.Content = box;
                    break;
                }

                default:
                {
                    var step = setting.Step > 0 ? setting.Step : 1f;
                    var format = step % 1f == 0f ? "0" : "0.##";
                    var valueText = new TextBlock();
                    valueText.Classes.Add("sliderValue");
                    var slider = new Slider
                    {
                        Minimum = setting.Min,
                        Maximum = setting.Max,
                        SmallChange = step,
                        LargeChange = step * 2,
                        TickFrequency = step,
                        IsSnapToTickEnabled = true,
                        Value = widget.Values[index],
                    };
                    slider.Classes.Add("appearanceSlider");
                    valueText.Text = slider.Value.ToString(format, CultureInfo.InvariantCulture);
                    slider.ValueChanged += (_, _) =>
                    {
                        valueText.Text = slider.Value.ToString(format, CultureInfo.InvariantCulture);
                        SetOverlayWidgetValue(widget, index, (float)slider.Value);
                    };
                    row.Content = new StackPanel
                    {
                        Orientation = Avalonia.Layout.Orientation.Horizontal,
                        Spacing = 14,
                        VerticalAlignment = Avalonia.Layout.VerticalAlignment.Center,
                        Children = { slider, valueText },
                    };
                    break;
                }
            }

            OverlayWidgetSettingsHost.Children.Add(row);
        }

        MarkFirstGroupLine(OverlayWidgetSettingsGroup);
    }

    private void SetOverlayWidgetValue(WidgetInstance widget, int index, float value)
    {
        if (_overlayRefreshing || OverlayModel is not { } model)
        {
            return;
        }

        widget.SetValue(index, value);
        model.MarkModified();
        OverlayChanged();
        OverlayPresetEditRow.Description = Localization.Instance.Get("Overlay.Preset.Edited");
    }

    private async Task SaveOverlayPresetAsync()
    {
        if (OverlayModel is not { } model)
        {
            return;
        }

        var dialog = new OverlayPresetNameDialog(model.NextCustomName());
        if (await Sheets.ShowAsync(dialog) is not string name)
        {
            return;
        }

        model.SaveAs(name);
        OverlayChanged(flushNow: true);
        RefreshOverlayEditor();
    }

    // ---- preview ----

    private void RenderOverlayPreview()
    {
        if (!_overlayEditorActive || _overlayHost is not { } host)
        {
            return;
        }

        // The image has no size of its own until it has a bitmap, so it takes the frame's inner size.
        var inner = OverlayPreviewFrame.Bounds.Deflate(OverlayPreviewFrame.BorderThickness);
        var bounds = new Rect(0, 0, Math.Floor(inner.Width), Math.Floor(inner.Height));
        OverlayPreviewImage.Width = bounds.Width;
        OverlayPreviewImage.Height = bounds.Height;
        var scaling = TopLevel.GetTopLevel(this)?.RenderScaling ?? 1.0;
        var width = (int)Math.Round(bounds.Width * scaling);
        var height = (int)Math.Round(bounds.Height * scaling);
        if (width < 32 || height < 32)
        {
            return;
        }

        OverlayFrame? frame;
        try
        {
            frame = host.RenderOnce(width, height, scaling);
        }
        catch (Exception)
        {
            return;
        }

        if (frame is null || frame.Width != width || frame.Height != height)
        {
            return;
        }

        // A fresh Bitmap per frame: unlike a WriteableBitmap rewritten in place, it is always redrawn
        // (a WriteableBitmap kept showing its first, empty contents under the headless renderer).
        Bitmap bitmap;
        var pinned = GCHandle.Alloc(frame.Pixels, GCHandleType.Pinned);
        try
        {
            bitmap = new Bitmap(
                PixelFormat.Bgra8888, AlphaFormat.Premul, pinned.AddrOfPinnedObject(),
                new PixelSize(width, height), new Vector(96, 96), width * 4);
        }
        finally
        {
            pinned.Free();
        }

        OverlayPreviewImage.Source = bitmap;
        // The compositor may still draw the last frame or two, so a bitmap is disposed only after
        // a few newer ones have replaced it (disposing at once would blank the preview for a frame).
        _overlayBitmaps.Enqueue(bitmap);
        while (_overlayBitmaps.Count > 4)
        {
            _overlayBitmaps.Dequeue().Dispose();
        }
    }
}
