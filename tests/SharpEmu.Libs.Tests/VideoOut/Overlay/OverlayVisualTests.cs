// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.VideoOut.Overlay;
using Xunit;

namespace SharpEmu.Libs.Tests.VideoOut.Overlay;

/// <summary>
/// Renders every preset (plus edit mode and a settings popover) from the simulator and writes PNGs to
/// artifacts/overlay-shots/ so the result can be looked at. Opt in with SHARPEMU_OVERLAY_SHOTS=1.
/// </summary>
public sealed class OverlayVisualTests
{
    private static readonly (string Name, int W, int H, double Scale)[] Sizes =
    [
        ("1080p", 1920, 1080, 1.0),
        ("1440p2x", 2560, 1440, 2.0),
    ];

    [Fact]
    public void WritesPresetScreenshots()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_OVERLAY_SHOTS") != "1")
        {
            return;
        }

        var dir = OverlayTestSupport.ShotsDirectory();
        var scenario = Environment.GetEnvironmentVariable("SHARPEMU_OVERLAY_SHOTS_SCENARIO") ?? "ds";
        foreach (var preset in OverlayLayoutModel.BuiltInPresetNames)
        {
            foreach (var (name, w, h, scale) in Sizes)
            {
                using var host = OverlayTestSupport.CreateHost(scenario, preset);
                Assert.NotNull(host.RenderOnce(w, h, scale));
                var frame = host.RenderOnce(w, h, scale)!;
                OverlayTestSupport.WritePng(Path.Combine(dir, $"{scenario}-{preset.Replace(' ', '-')}-{name}.png"), frame);
            }
        }
    }

    [Fact]
    public void WritesEditModeAndPopoverScreenshots()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_OVERLAY_SHOTS") != "1")
        {
            return;
        }

        var dir = OverlayTestSupport.ShotsDirectory();
        foreach (var (name, w, h, scale) in Sizes)
        {
            using var host = OverlayTestSupport.CreateHost("ds", OverlayLayoutModel.FullPreset);
            host.RenderOnce(w, h, scale);
            Assert.True(host.HandleKey(OverlayKey.ToggleEdit));
            host.RenderOnce(w, h, scale);
            OverlayTestSupport.WritePng(Path.Combine(dir, $"edit-mode-{name}.png"), host.RenderOnce(w, h, scale)!);

            // Open the settings popover of the top-costs widget.
            var topCosts = host.Model.Widgets.FindIndex(x => x.Type == OverlayWidgetType.TopCosts);
            Assert.True(OverlayTestSupport.ClickHit(host, OverlayHitKind.WidgetGear, topCosts));
            host.RenderOnce(w, h, scale);
            OverlayTestSupport.WritePng(Path.Combine(dir, $"popover-settings-{name}.png"), host.RenderOnce(w, h, scale)!);

            host.HandleKey(OverlayKey.Escape);
            host.RenderOnce(w, h, scale);
            Assert.True(OverlayTestSupport.ClickHit(host, OverlayHitKind.EditAdd));
            host.RenderOnce(w, h, scale);
            OverlayTestSupport.WritePng(Path.Combine(dir, $"popover-add-{name}.png"), host.RenderOnce(w, h, scale)!);

            host.HandleKey(OverlayKey.Escape);
            host.RenderOnce(w, h, scale);
            Assert.True(OverlayTestSupport.ClickHit(host, OverlayHitKind.EditStyle));
            host.RenderOnce(w, h, scale);
            OverlayTestSupport.WritePng(Path.Combine(dir, $"popover-style-{name}.png"), host.RenderOnce(w, h, scale)!);
        }
    }

    [Fact]
    public void WritesThemeAndStateScreenshots()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_OVERLAY_SHOTS") != "1")
        {
            return;
        }

        var dir = OverlayTestSupport.ShotsDirectory();
        const int w = 1920, h = 1080;
        foreach (var theme in Enum.GetValues<OverlayTheme>())
        {
            using var host = OverlayTestSupport.CreateHost("gta", OverlayLayoutModel.FullPreset);
            host.Model.Globals.Theme = theme;
            host.RenderOnce(w, h, 1.0);
            OverlayTestSupport.WritePng(Path.Combine(dir, $"theme-{theme}-gta.png"), host.RenderOnce(w, h, 1.0)!);
        }

        foreach (var scenario in SimulatedOverlayDataSource.ScenarioKeys)
        {
            using var host = OverlayTestSupport.CreateHost(scenario, OverlayLayoutModel.FullPreset);
            host.RenderOnce(w, h, 1.0);
            OverlayTestSupport.WritePng(Path.Combine(dir, $"scenario-{scenario}-full.png"), host.RenderOnce(w, h, 1.0)!);
        }

        // Paused on a selected frame with a hovered timeline tooltip.
        using var paused = OverlayTestSupport.CreateHost("ds", OverlayLayoutModel.InspectorPreset);
        paused.RenderOnce(w, h, 1.0);
        Assert.True(paused.HandleKey(OverlayKey.TogglePause));
        paused.RenderOnce(w, h, 1.0);
        paused.HandleKey(OverlayKey.StepBack);
        paused.RenderOnce(w, h, 1.0);
        if (OverlayTestSupport.FindHit(paused.Controller, OverlayHitKind.TimelineLane) is { } p)
        {
            paused.HandlePointer(new OverlayPointerEvent(OverlayPointerKind.Move, p.X - 40, p.Y - 8));
            paused.RenderOnce(w, h, 1.0);
            paused.HandlePointer(new OverlayPointerEvent(OverlayPointerKind.Move, p.X - 38, p.Y - 8));
        }

        OverlayTestSupport.WritePng(Path.Combine(dir, "paused-tooltip-1080p.png"), paused.RenderOnce(w, h, 1.0)!);
    }
}
