// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.VideoOut.Overlay;
using Xunit;

namespace SharpEmu.Libs.Tests.VideoOut.Overlay;

public sealed class OverlayLayoutTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "sharpemu-overlay-tests-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        try
        {
            Directory.Delete(_dir, recursive: true);
        }
        catch (IOException)
        {
        }
    }

    [Fact]
    public void BuiltInPresetsAreFourAndInspectorIsTheDefault()
    {
        Assert.Equal(4, OverlayLayoutModel.BuiltInPresetNames.Count);
        var model = new OverlayLayoutModel();
        Assert.Equal(OverlayLayoutModel.InspectorPreset, model.PresetName);
        Assert.Equal(
            [OverlayWidgetType.Fps, OverlayWidgetType.Timeline, OverlayWidgetType.History, OverlayWidgetType.TopCosts],
            model.Widgets.Select(w => w.Type));
    }

    [Fact]
    public void ApplyPresetSwitchesWidgetsAndResetsModified()
    {
        var model = new OverlayLayoutModel();
        Assert.True(model.ApplyPreset(OverlayLayoutModel.MinimalPreset));
        Assert.Single(model.Widgets);
        Assert.Equal(OverlayWidgetType.Fps, model.Widgets[0].Type);
        Assert.True(model.ApplyPreset(OverlayLayoutModel.FullPreset));
        Assert.Equal(10, model.Widgets.Count);
        Assert.False(model.ApplyPreset("No such preset"));
        Assert.Equal(OverlayLayoutModel.FullPreset, model.PresetName);
    }

    [Fact]
    public void CyclePresetWalksAllPresetsAndWraps()
    {
        var model = new OverlayLayoutModel();
        var seen = new List<string> { model.PresetName };
        for (var i = 0; i < 4; i++)
        {
            seen.Add(model.CyclePreset(1));
        }

        Assert.Equal(seen[0], seen[4]);
        Assert.Equal(4, seen.Take(4).Distinct().Count());
        Assert.Equal(seen[3], model.CyclePreset(-1));
    }

    [Fact]
    public void WidgetCatalogHasElevenWidgetsWithDefaultsInsideTheirRanges()
    {
        Assert.Equal(11, WidgetCatalog.All.Count);
        foreach (var spec in WidgetCatalog.All)
        {
            var inst = new WidgetInstance("t", spec.Type);
            for (var i = 0; i < spec.Settings.Length; i++)
            {
                Assert.Equal(inst.Values[i], spec.Settings[i].Clamp(inst.Values[i]));
            }
        }
    }

    [Fact]
    public void SaveAndLoadRoundTripsLayoutGlobalsAndCustomPresets()
    {
        var store = new OverlayLayoutStore(_dir);
        var model = new OverlayLayoutModel();
        model.ApplyPreset(OverlayLayoutModel.FullPreset);
        var topCosts = model.Widgets.First(w => w.Type == OverlayWidgetType.TopCosts);
        topCosts.X = 0.33f;
        topCosts.Y = 0.25f;
        topCosts.SetValue(topCosts.Spec.IndexOf("n"), 5);
        model.Globals.Opacity = 0.6f;
        model.Globals.Scale = 1.25f;
        model.Globals.FontSize = 13;
        model.Globals.Theme = OverlayTheme.Glass;
        model.MarkModified();
        var saved = model.SaveAs("Mine");
        Assert.Equal("Mine", saved);
        Assert.True(store.Save(model));

        var loaded = new OverlayLayoutModel();
        Assert.True(store.LoadInto(loaded));
        Assert.Equal(model.Widgets.Count, loaded.Widgets.Count);
        var again = loaded.Widgets.First(w => w.Type == OverlayWidgetType.TopCosts);
        Assert.Equal(0.33f, again.X, 3);
        Assert.Equal(0.25f, again.Y, 3);
        Assert.Equal(5, again.Int(again.Spec.IndexOf("n")));
        Assert.Equal(0.6f, loaded.Globals.Opacity, 3);
        Assert.Equal(1.25f, loaded.Globals.Scale, 3);
        Assert.Equal(13, loaded.Globals.FontSize);
        Assert.Equal(OverlayTheme.Glass, loaded.Globals.Theme);
        Assert.Contains("Mine", loaded.CustomOrder);
        Assert.True(loaded.CustomPresets.ContainsKey("Mine"));
    }

    [Fact]
    public void CorruptFilesFallBackToThePresetWithoutThrowing()
    {
        Directory.CreateDirectory(_dir);
        var store = new OverlayLayoutStore(_dir);
        File.WriteAllText(store.LayoutPath, "{ this is not json");
        File.WriteAllText(store.PresetsPath, "\0\0garbage");
        var model = new OverlayLayoutModel();
        Assert.False(store.LoadInto(model));
        Assert.Equal(4, model.Widgets.Count);
        Assert.Empty(model.CustomPresets);

        using var host = new OverlayHost(null, null, () => true, store, OverlayLayoutModel.MinimalPreset);
        Assert.Equal(OverlayLayoutModel.MinimalPreset, host.Model.PresetName);
    }

    [Fact]
    public void LoadToleratesUnknownWidgetsAndOutOfRangeValues()
    {
        Directory.CreateDirectory(_dir);
        var store = new OverlayLayoutStore(_dir);
        File.WriteAllText(store.LayoutPath, """
            {"version":1,"preset":"Inspector","globals":{"opacity":9,"scale":0,"fontSize":99,"theme":"nope"},
             "widgets":[{"type":"warp-drive","x":0.1,"y":0.1,"w":0.2,"h":0.2},
                        {"type":"fps","x":-4,"y":7,"w":0.0001,"h":50,"settings":{"len":9999}}]}
            """);
        var model = new OverlayLayoutModel();
        var loaded = store.LoadInto(model);
        if (loaded)
        {
            Assert.DoesNotContain(model.Widgets, w => w.X < 0 || w.Y > 1 || w.H > 2);
            Assert.InRange(model.Globals.Opacity, OverlayGlobals.MinOpacity, 1f);
        }

        Assert.NotEmpty(model.Widgets);
    }

    [Fact]
    public void SavingTwiceLeavesNoTempFilesBehind()
    {
        var store = new OverlayLayoutStore(_dir);
        var model = new OverlayLayoutModel();
        Assert.True(store.Save(model));
        Assert.True(store.Save(model));
        var files = Directory.GetFiles(_dir).Select(Path.GetFileName).OrderBy(n => n).ToArray();
        Assert.Equal(["layout.json", "presets.json"], files);
    }

    [Fact]
    public void CustomPresetNamesAreAutoGeneratedAndDeletable()
    {
        var model = new OverlayLayoutModel();
        var first = model.SaveAs(model.NextCustomName());
        var second = model.SaveAs(model.NextCustomName());
        Assert.NotEqual(first, second);
        Assert.True(model.DeleteCustom(first));
        Assert.False(model.CustomPresets.ContainsKey(first));
        Assert.False(model.DeleteCustom(OverlayLayoutModel.InspectorPreset));
    }

    [Fact]
    public void AddWidgetPlacesItInsideTheSurfaceAndRemoveTakesItOut()
    {
        var model = new OverlayLayoutModel();
        model.ApplyPreset(OverlayLayoutModel.MinimalPreset);
        var added = model.AddWidget(OverlayWidgetType.Memory);
        Assert.Equal(2, model.Widgets.Count);
        Assert.InRange(added.X, 0f, 1f);
        Assert.InRange(added.Y, 0f, 1f);
        Assert.InRange(added.X + added.W, 0f, 1.001f);
        Assert.InRange(added.Y + added.H, 0f, 1.001f);
        Assert.True(model.Remove(added));
        Assert.Single(model.Widgets);
    }

    [Fact]
    public void BoxOfScalesAgainstTheReferenceSpaceAndClampsToTheSurface()
    {
        var widget = new WidgetInstance("w", OverlayWidgetType.Fps) { X = 0.95f, Y = 0.95f, W = 0.25f, H = 0.2f };
        var metrics = new OverlayMetrics(1920, 1080, 1f, 1f);
        Assert.Equal(2f, metrics.K, 3);
        var box = metrics.BoxOf(widget);
        Assert.Equal(0.25f * 960f * 2f, box.W, 2);
        Assert.Equal(0.2f * 540f * 2f, box.H, 2);
        Assert.True(box.Right <= 1920.001f);
        Assert.True(box.Bottom <= 1080.001f);
    }

    [Fact]
    public void DemandFollowsTheVisibleWidgets()
    {
        var model = new OverlayLayoutModel();
        model.ApplyPreset(OverlayLayoutModel.MinimalPreset);
        Assert.False(OverlayHost.DemandFor(model.Widgets).HasFlag(OverlayCaptureDemand.Timeline));
        model.ApplyPreset(OverlayLayoutModel.InspectorPreset);
        Assert.True(OverlayHost.DemandFor(model.Widgets).HasFlag(OverlayCaptureDemand.Timeline));
    }

    [Theory]
    [InlineData("Inter-Regular.ttf")]
    [InlineData("Inter-SemiBold.ttf")]
    [InlineData("JetBrainsMono-Regular.ttf")]
    [InlineData("JetBrainsMono-Bold.ttf")]
    public void BundledFontsAreEmbedded(string fileName)
    {
        var bytes = OverlayTypography.BundledFont(fileName);
        Assert.NotNull(bytes);
        // A TrueType file starts with the 0x00010000 sfnt version.
        Assert.Equal(new byte[] { 0, 1, 0, 0 }, bytes![..4]);
    }

    [Fact]
    public void FontRoundTripsAndUnknownFontsFallBackToInter()
    {
        var store = new OverlayLayoutStore(_dir);
        var model = new OverlayLayoutModel();
        model.Globals.Font = OverlayFont.JetBrainsMono;
        model.Touch();
        Assert.True(store.Save(model));
        var loaded = new OverlayLayoutModel();
        Assert.True(store.LoadInto(loaded));
        Assert.Equal(OverlayFont.JetBrainsMono, loaded.Globals.Font);

        File.WriteAllText(store.LayoutPath, File.ReadAllText(store.LayoutPath).Replace("jetbrainsmono", "comic"));
        var fallback = new OverlayLayoutModel();
        Assert.True(store.LoadInto(fallback));
        Assert.Equal(OverlayFont.Inter, fallback.Globals.Font);
    }

    [Fact]
    public void LaunchPresetWinsUnlessTheSavedLayoutIsThatPreset()
    {
        var store = new OverlayLayoutStore(_dir);
        var model = new OverlayLayoutModel();
        model.ApplyPreset(OverlayLayoutModel.FullPreset);
        model.Widgets[0].X = 0.123f;
        model.MarkModified();
        Assert.True(store.Save(model));

        using (var same = new OverlayHost(store: store, launchPreset: OverlayLayoutModel.FullPreset))
        {
            Assert.Equal(0.123f, same.Model.Widgets[0].X, 3);
        }

        using var other = new OverlayHost(store: store, launchPreset: OverlayLayoutModel.MinimalPreset);
        Assert.Equal(OverlayLayoutModel.MinimalPreset, other.Model.PresetName);
    }

    [Fact]
    public void StampChangesWhenAnotherWriterSavesTheLayout()
    {
        var store = new OverlayLayoutStore(_dir);
        var before = store.Stamp();
        var model = new OverlayLayoutModel();
        model.Globals.Theme = OverlayTheme.Oled;
        model.Touch();
        Assert.True(store.Save(model));
        Assert.NotEqual(before, store.Stamp());
    }
}
