// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.VideoOut.Overlay;
using Xunit;

namespace SharpEmu.Libs.Tests.VideoOut.Overlay;

public sealed class OverlayControllerTests
{
    private const int W = 1920, H = 1080;

    private static OverlayHost Edit(string preset = OverlayLayoutModel.FullPreset)
    {
        var host = OverlayTestSupport.CreateHost("ds", preset, realFonts: false);
        host.RenderOnce(W, H, 1.0);
        Assert.True(host.HandleKey(OverlayKey.ToggleEdit));
        host.RenderOnce(W, H, 1.0);
        return host;
    }

    private static void Drag(OverlayHost host, float x0, float y0, float x1, float y1, OverlayModifiers mods = OverlayModifiers.None)
    {
        host.HandlePointer(new OverlayPointerEvent(OverlayPointerKind.Move, x0, y0));
        host.HandlePointer(new OverlayPointerEvent(OverlayPointerKind.Down, x0, y0));
        host.RenderOnce(W, H, 1.0);
        const int steps = 6;
        for (var i = 1; i <= steps; i++)
        {
            host.HandlePointer(new OverlayPointerEvent(OverlayPointerKind.Move, x0 + (x1 - x0) * i / steps, y0 + (y1 - y0) * i / steps, Modifiers: mods));
            host.RenderOnce(W, H, 1.0);
        }

        host.HandlePointer(new OverlayPointerEvent(OverlayPointerKind.Up, x1, y1, Modifiers: mods));
        host.RenderOnce(W, H, 1.0);
    }

    // ------------------------------------------------------------------ snapping math

    [Fact]
    public void SnapMoveLocksToANearbyEdgeAndReportsTheGuide()
    {
        var r = OverlaySnap.SnapMove(103f, 50f, 1000f, [100f], [], 6f, 10f);
        Assert.True(r.Snapped);
        Assert.Equal(100f, r.Position, 3);
        Assert.Equal(100f, r.Guide, 3);
        Assert.True(r.HasGuide);
    }

    [Fact]
    public void SnapMoveAlignsTheFarEdgeAndTheSurfaceCentre()
    {
        var far = OverlaySnap.SnapMove(246f, 50f, 1000f, [], [300f], 6f, 10f);
        Assert.Equal(250f, far.Position, 3);
        var centre = OverlaySnap.SnapMove(472f, 50f, 1000f, [], [], 6f, 10f);
        Assert.True(centre.Snapped);
        Assert.Equal(475f, centre.Position, 3);
        Assert.Equal(500f, centre.Guide, 3);
    }

    [Fact]
    public void SnapMoveFallsBackToTheGridWithoutAGuide()
    {
        var r = OverlaySnap.SnapMove(237f, 50f, 1000f, [100f], [], 6f, 10f);
        Assert.False(r.Snapped);
        Assert.False(r.HasGuide);
        Assert.Equal(240f, r.Position, 3);
        var free = OverlaySnap.SnapMove(237f, 50f, 1000f, [], [], 6f, 0f);
        Assert.Equal(237f, free.Position, 3);
    }

    [Fact]
    public void SnapResizeSnapsTheFarEdgeThenTheGrid()
    {
        var snapped = OverlaySnap.SnapResize(100f, 197f, [300f], 6f, 10f);
        Assert.True(snapped.Snapped);
        Assert.Equal(200f, snapped.Position, 3);
        var grid = OverlaySnap.SnapResize(100f, 187f, [], 6f, 10f);
        Assert.False(grid.Snapped);
        Assert.Equal(190f, grid.Position, 3);
    }

    // ------------------------------------------------------------------ keys

    [Fact]
    public void KeysToggleEditAndPauseAndEscapeBacksOut()
    {
        using var host = OverlayTestSupport.CreateHost("ds", null, realFonts: false);
        host.RenderOnce(W, H, 1.0);
        Assert.False(host.Controller.Edit);
        Assert.True(host.HandleKey(OverlayKey.ToggleEdit));
        Assert.True(host.Controller.Edit);
        Assert.True(host.HandleKey(OverlayKey.Escape));
        Assert.False(host.Controller.Edit);
        Assert.True(host.HandleKey(OverlayKey.TogglePause));
        Assert.True(host.Controller.Paused);
        Assert.True(host.HandleKey(OverlayKey.TogglePause));
        Assert.False(host.Controller.Paused);
    }

    [Fact]
    public void F2CyclesPresetsAndNumberKeysOnlyWorkInEditMode()
    {
        using var host = OverlayTestSupport.CreateHost("ds", null, realFonts: false);
        host.RenderOnce(W, H, 1.0);
        var before = host.Model.PresetName;
        Assert.True(host.HandleKey(OverlayKey.CyclePreset));
        Assert.NotEqual(before, host.Model.PresetName);
        var cycled = host.Model.PresetName;
        host.HandleKey(OverlayKey.Preset1);
        Assert.Equal(cycled, host.Model.PresetName);
        host.HandleKey(OverlayKey.ToggleEdit);
        Assert.True(host.HandleKey(OverlayKey.Preset1));
        Assert.Equal(OverlayLayoutModel.BuiltInPresetNames[0], host.Model.PresetName);
        Assert.True(host.HandleKey(OverlayKey.Preset4));
        Assert.Equal(OverlayLayoutModel.BuiltInPresetNames[3], host.Model.PresetName);
    }

    [Fact]
    public void SteppingWhilePausedMovesTheSelectedFrameAndStopsAtTheEnds()
    {
        using var host = OverlayTestSupport.CreateHost("ds", OverlayLayoutModel.InspectorPreset, realFonts: false);
        host.RenderOnce(W, H, 1.0);
        host.HandleKey(OverlayKey.TogglePause);
        host.RenderOnce(W, H, 1.0);
        host.HandleKey(OverlayKey.StepBack);
        host.RenderOnce(W, H, 1.0);
        var first = host.Controller.SelectedSequence;
        Assert.True(first >= 0);
        host.HandleKey(OverlayKey.StepBack);
        host.RenderOnce(W, H, 1.0);
        Assert.Equal(first - 1, host.Controller.SelectedSequence);
        host.HandleKey(OverlayKey.StepForward);
        host.RenderOnce(W, H, 1.0);
        Assert.Equal(first, host.Controller.SelectedSequence);
    }

    [Fact]
    public void PausingFreezesWhatTheViewShows()
    {
        var source = new SimulatedOverlayDataSource("ds", 7, realTime: false);
        source.Advance(8000);
        using var host = OverlayTestSupport.CreateHost(source: source, realFonts: false, preset: OverlayLayoutModel.InspectorPreset);
        host.RenderOnce(W, H, 1.0);
        host.HandleKey(OverlayKey.TogglePause);
        host.RenderOnce(W, H, 1.0);
        var frozen = host.Controller.EndSequence;
        source.Advance(4000);
        host.RenderOnce(W, H, 1.0);
        Assert.Equal(frozen, host.Controller.EndSequence);
    }

    // ------------------------------------------------------------------ pointer flows

    [Fact]
    public void DraggingAWidgetMovesItOntoTheGridOrAGuide()
    {
        using var host = Edit(OverlayLayoutModel.MinimalPreset);
        var widget = host.Model.Widgets[0];
        var hit = OverlayTestSupport.FindHit(host.Controller, OverlayHitKind.WidgetBody, 0)!.Value;
        var (x0, y0) = (widget.X, widget.Y);
        Drag(host, hit.X, hit.Y, hit.X + 303f, hit.Y + 197f);
        Assert.True(widget.X > x0 + 0.1f);
        Assert.True(widget.Y > y0 + 0.1f);
        Assert.True(host.Model.Modified);
        Assert.InRange(widget.X, 0f, 1f);
        Assert.InRange(widget.Y, 0f, 1f);
    }

    [Fact]
    public void AltBypassesSnappingSoThePositionFollowsThePointerExactly()
    {
        using var host = Edit(OverlayLayoutModel.MinimalPreset);
        var widget = host.Model.Widgets[0];
        var hit = OverlayTestSupport.FindHit(host.Controller, OverlayHitKind.WidgetBody, 0)!.Value;
        var x0 = widget.X * W;
        Drag(host, hit.X, hit.Y, hit.X + 137f, hit.Y + 3f, OverlayModifiers.Alt);
        Assert.Equal(x0 + 137f, widget.X * W, 1);
    }

    [Fact]
    public void ResizingGrowsTheBoxButKeepsItInsideTheSurfaceAndAboveItsMinimum()
    {
        using var host = Edit(OverlayLayoutModel.MinimalPreset);
        var widget = host.Model.Widgets[0];
        var handle = OverlayTestSupport.FindHit(host.Controller, OverlayHitKind.WidgetResize, 0)!.Value;
        var w0 = widget.W;
        Drag(host, handle.X, handle.Y, handle.X + 120f, handle.Y + 80f, OverlayModifiers.Alt);
        Assert.True(widget.W > w0);
        Drag(host, handle.X + 120f, handle.Y + 80f, handle.X - 2000f, handle.Y - 2000f, OverlayModifiers.Alt);
        Assert.True(widget.W >= widget.Spec.MinW / 960f - 0.0001f);
        Assert.True(widget.H >= widget.Spec.MinH / 540f - 0.0001f);
    }

    [Fact]
    public void RemoveButtonDropsTheWidgetAndAddPopoverBringsOneBack()
    {
        using var host = Edit();
        var count = host.Model.Widgets.Count;
        Assert.True(OverlayTestSupport.ClickHit(host, OverlayHitKind.WidgetRemove, count - 1));
        host.RenderOnce(W, H, 1.0);
        Assert.Equal(count - 1, host.Model.Widgets.Count);

        Assert.True(OverlayTestSupport.ClickHit(host, OverlayHitKind.EditAdd));
        host.RenderOnce(W, H, 1.0);
        Assert.Equal(OverlayPopoverKind.AddWidget, host.Controller.Popover);
        Assert.True(OverlayTestSupport.ClickHit(host, OverlayHitKind.PopoverAddItem, 0));
        host.RenderOnce(W, H, 1.0);
        Assert.Equal(count, host.Model.Widgets.Count);
    }

    [Fact]
    public void SettingsPopoverTogglesASettingAndTheStyleSliderChangesOpacity()
    {
        using var host = Edit();
        var index = host.Model.Widgets.FindIndex(x => x.Type == OverlayWidgetType.TopCosts);
        var widget = host.Model.Widgets[index];
        Assert.True(OverlayTestSupport.ClickHit(host, OverlayHitKind.WidgetGear, index));
        host.RenderOnce(W, H, 1.0);
        Assert.Equal(OverlayPopoverKind.WidgetSettings, host.Controller.Popover);
        var before = widget.Values.ToArray();
        Assert.True(OverlayTestSupport.ClickHit(host, OverlayHitKind.PopoverToggle, widget.Spec.IndexOf("waits")));
        host.RenderOnce(W, H, 1.0);
        Assert.NotEqual(before, widget.Values);

        host.HandleKey(OverlayKey.Escape);
        host.RenderOnce(W, H, 1.0);
        Assert.True(OverlayTestSupport.ClickHit(host, OverlayHitKind.EditStyle));
        host.RenderOnce(W, H, 1.0);
        var slider = OverlayTestSupport.FindHit(host.Controller, OverlayHitKind.PopoverSlider, 0)!.Value;
        var opacity = host.Model.Globals.Opacity;
        var hit = host.Controller.Hits[host.Controller.Hits.Find(slider.X, slider.Y)];
        OverlayTestSupport.Click(host, hit.Rect.X + hit.Rect.W * 0.05f, hit.Rect.Y + hit.Rect.H / 2f);
        Assert.True(host.Model.Globals.Opacity < opacity);
        Assert.InRange(host.Model.Globals.Opacity, OverlayGlobals.MinOpacity, 1f);
    }

    [Fact]
    public void SaveAsAddsACustomPresetThatCyclingReaches()
    {
        using var host = Edit();
        Assert.True(OverlayTestSupport.ClickHit(host, OverlayHitKind.EditSaveAs));
        host.RenderOnce(W, H, 1.0);
        Assert.Single(host.Model.CustomPresets);
        var name = host.Model.CustomOrder[0];
        host.Model.ApplyPreset(OverlayLayoutModel.MinimalPreset);
        var reached = false;
        for (var i = 0; i < 6 && !reached; i++)
        {
            reached = host.Model.CyclePreset(1) == name;
        }

        Assert.True(reached);
    }
}
