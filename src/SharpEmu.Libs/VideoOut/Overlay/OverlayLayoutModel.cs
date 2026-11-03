// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;

namespace SharpEmu.Libs.VideoOut.Overlay;

public readonly record struct RectF(float X, float Y, float W, float H)
{
    public float Right => X + W;
    public float Bottom => Y + H;
    public bool Contains(float x, float y) => x >= X && x < X + W && y >= Y && y < Y + H;
    public bool Intersects(RectF o) => X < o.Right && Right > o.X && Y < o.Bottom && Bottom > o.Y;
    public RectF Inflate(float d) => new(X - d, Y - d, W + 2 * d, H + 2 * d);
}

/// <summary>Maps normalised widget records to surface points.</summary>
public readonly struct OverlayMetrics
{
    public const float RefWidth = 960f, RefHeight = 540f;

    public OverlayMetrics(float widthPoints, float heightPoints, float pixelScale, float userScale)
    {
        WidthPoints = MathF.Max(1f, widthPoints);
        HeightPoints = MathF.Max(1f, heightPoints);
        PixelScale = MathF.Max(0.5f, pixelScale);
        UserScale = userScale;
    }

    public float WidthPoints { get; }
    public float HeightPoints { get; }
    public float PixelScale { get; }
    public float UserScale { get; }

    /// <summary>Points per reference unit (the surface is laid out like a 960x540 canvas).</summary>
    public float K => MathF.Min(WidthPoints / RefWidth, HeightPoints / RefHeight);

    /// <summary>Pixels per reference unit of widget content: the transform widgets draw with.</summary>
    public float ContentScale => K * UserScale * PixelScale;

    /// <summary>A widget's on-screen box in points, clamped inside the surface.</summary>
    public RectF BoxOf(WidgetInstance w)
    {
        var s = K * UserScale;
        var width = w.W * RefWidth * s;
        var height = w.H * RefHeight * s;
        return new(
            Math.Clamp(w.X * WidthPoints, 0f, MathF.Max(0f, WidthPoints - width)),
            Math.Clamp(w.Y * HeightPoints, 0f, MathF.Max(0f, HeightPoints - height)),
            width, height);
    }
}

/// <summary>The widgets on screen plus presets and globals; the persisted state of the overlay.</summary>
public sealed class OverlayLayoutModel
{
    public const string MinimalPreset = "Minimal FPS";
    public const string CompactPreset = "Compact strip";
    public const string InspectorPreset = "Inspector";
    public const string FullPreset = "Full debug";

    private static readonly string[] BuiltInNames = [MinimalPreset, CompactPreset, InspectorPreset, FullPreset];

    private int _nextId = 1;

    public OverlayLayoutModel()
    {
        ApplyPreset(InspectorPreset);
    }

    public List<WidgetInstance> Widgets { get; private set; } = [];
    public OverlayGlobals Globals { get; private set; } = new();
    public string PresetName { get; private set; } = InspectorPreset;
    /// <summary>True when the layout differs from the named preset (edited since it was applied).</summary>
    public bool Modified { get; private set; }
    /// <summary>Custom presets by name, in creation order.</summary>
    public Dictionary<string, List<WidgetInstance>> CustomPresets { get; } = new(StringComparer.Ordinal);
    public List<string> CustomOrder { get; } = [];
    /// <summary>Bumps on every change; the host persists when it moves.</summary>
    public long Revision { get; private set; }

    public static IReadOnlyList<string> BuiltInPresetNames => BuiltInNames;
    public static bool IsBuiltIn(string name) => Array.IndexOf(BuiltInNames, name) >= 0;

    public IEnumerable<string> AllPresetNames()
    {
        foreach (var name in BuiltInNames)
        {
            yield return name;
        }

        foreach (var name in CustomOrder)
        {
            yield return name;
        }
    }

    public string NewId(OverlayWidgetType type) =>
        string.Create(CultureInfo.InvariantCulture, $"{WidgetCatalog.Get(type).Key}#{_nextId++}");

    public void Touch() => Revision++;

    /// <summary>Marks the layout edited (drag, resize, settings, add, remove).</summary>
    public void MarkModified()
    {
        Modified = true;
        Revision++;
    }

    // ---------------------------------------------------------------- presets

    private WidgetInstance Make(OverlayWidgetType type, float x, float y, float w, float h, params (string Key, float Value)[] overrides)
    {
        var inst = new WidgetInstance(NewId(type), type)
        {
            X = x / OverlayMetrics.RefWidth,
            Y = y / OverlayMetrics.RefHeight,
            W = w / OverlayMetrics.RefWidth,
            H = h / OverlayMetrics.RefHeight,
        };
        foreach (var (key, value) in overrides)
        {
            var idx = inst.Spec.IndexOf(key);
            if (idx >= 0)
            {
                inst.SetValue(idx, value);
            }
        }

        return inst;
    }

    private List<WidgetInstance>? BuildBuiltIn(string name)
    {
        switch (name)
        {
            case MinimalPreset:
                return [Make(OverlayWidgetType.Fps, 12, 12, 184, 96)];
            case CompactPreset:
                return [Make(OverlayWidgetType.Strip, 12, 12, 440, 120)];
            case InspectorPreset:
                return
                [
                    Make(OverlayWidgetType.Fps, 598, 12, 350, 84),
                    Make(OverlayWidgetType.Timeline, 598, 102, 350, 124),
                    Make(OverlayWidgetType.History, 598, 232, 350, 100),
                    Make(OverlayWidgetType.TopCosts, 598, 338, 350, 190, ("n", 8)),
                ];
            case FullPreset:
                return
                [
                    Make(OverlayWidgetType.Fps, 10, 10, 190, 96),
                    Make(OverlayWidgetType.Split, 10, 112, 190, 110),
                    Make(OverlayWidgetType.Gpu, 10, 228, 190, 124),
                    Make(OverlayWidgetType.Session, 10, 358, 190, 106),
                    // No strip here: the FPS and budget panels already carry its figures.
                    Make(OverlayWidgetType.Budget, 216, 10, 488, 140),
                    Make(OverlayWidgetType.Timeline, 216, 156, 488, 170),
                    Make(OverlayWidgetType.History, 216, 332, 488, 188),
                    Make(OverlayWidgetType.Cpu, 710, 10, 240, 146),
                    Make(OverlayWidgetType.Memory, 710, 162, 240, 150),
                    Make(OverlayWidgetType.TopCosts, 710, 318, 240, 212, ("n", 9)),
                ];
            default:
                return null;
        }
    }

    /// <summary>Replaces the layout with a built-in or custom preset; false when the name is unknown.</summary>
    public bool ApplyPreset(string name)
    {
        var widgets = BuildBuiltIn(name);
        if (widgets is null)
        {
            if (!CustomPresets.TryGetValue(name, out var custom))
            {
                return false;
            }

            widgets = [];
            foreach (var w in custom)
            {
                widgets.Add(w.Clone(NewId(w.Type)));
            }
        }

        Widgets = widgets;
        PresetName = name;
        Modified = false;
        Revision++;
        return true;
    }

    /// <summary>Steps through built-in then custom presets (wrapping); returns the new preset's name.</summary>
    public string CyclePreset(int direction)
    {
        var names = new List<string>(AllPresetNames());
        var index = names.IndexOf(PresetName);
        index = ((index < 0 ? 0 : index) + (direction >= 0 ? 1 : -1) + names.Count) % names.Count;
        ApplyPreset(names[index]);
        return names[index];
    }

    /// <summary>Reloads the current preset (or Inspector when the current name no longer exists).</summary>
    public void ResetLayout()
    {
        if (!ApplyPreset(PresetName))
        {
            ApplyPreset(InspectorPreset);
        }
    }

    /// <summary>Saves the current layout as a custom preset (names of built-ins are suffixed); returns the stored name.</summary>
    public string SaveAs(string name)
    {
        name = (name ?? string.Empty).Trim();
        if (name.Length > 32)
        {
            name = name[..32];
        }

        if (name.Length == 0)
        {
            name = "My layout";
        }

        if (IsBuiltIn(name))
        {
            name += " (custom)";
        }

        var copy = new List<WidgetInstance>(Widgets.Count);
        foreach (var w in Widgets)
        {
            copy.Add(w.Clone());
        }

        if (!CustomPresets.ContainsKey(name))
        {
            CustomOrder.Add(name);
        }

        CustomPresets[name] = copy;
        PresetName = name;
        Modified = false;
        Revision++;
        return name;
    }

    /// <summary>Picks "Custom 1", "Custom 2", ... skipping names in use.</summary>
    public string NextCustomName()
    {
        for (var n = 1; ; n++)
        {
            var candidate = string.Create(CultureInfo.InvariantCulture, $"Custom {n}");
            if (!CustomPresets.ContainsKey(candidate))
            {
                return candidate;
            }
        }
    }

    public bool DeleteCustom(string name)
    {
        if (!CustomPresets.Remove(name))
        {
            return false;
        }

        CustomOrder.Remove(name);
        if (string.Equals(PresetName, name, StringComparison.Ordinal))
        {
            ApplyPreset(InspectorPreset);
        }

        Revision++;
        return true;
    }

    // ---------------------------------------------------------------- widgets

    public WidgetInstance? Find(string id)
    {
        foreach (var w in Widgets)
        {
            if (string.Equals(w.Id, id, StringComparison.Ordinal))
            {
                return w;
            }
        }

        return null;
    }

    /// <summary>Adds a widget of the given type in the first free spot (reference space), or cascaded.</summary>
    public WidgetInstance AddWidget(OverlayWidgetType type)
    {
        var spec = WidgetCatalog.Get(type);
        var us = Globals.Scale;
        float bw = spec.DefaultW * us, bh = spec.DefaultH * us;
        var spot = (X: -1f, Y: -1f);
        const float margin = 10f, step = 20f, gap = 4f;
        for (var y = margin; y + bh <= OverlayMetrics.RefHeight - margin && spot.X < 0; y += step)
        {
            for (var x = margin; x + bw <= OverlayMetrics.RefWidth - margin; x += step)
            {
                var free = true;
                foreach (var o in Widgets)
                {
                    var ow = o.W * OverlayMetrics.RefWidth * us;
                    var oh = o.H * OverlayMetrics.RefHeight * us;
                    var ox = Math.Clamp(o.X * OverlayMetrics.RefWidth, 0f, MathF.Max(0f, OverlayMetrics.RefWidth - ow));
                    var oy = Math.Clamp(o.Y * OverlayMetrics.RefHeight, 0f, MathF.Max(0f, OverlayMetrics.RefHeight - oh));
                    if (x < ox + ow + gap && x + bw + gap > ox && y < oy + oh + gap && y + bh + gap > oy)
                    {
                        free = false;
                        break;
                    }
                }

                if (free)
                {
                    spot = (x, y);
                    break;
                }
            }
        }

        if (spot.X < 0)
        {
            var k = Widgets.Count % 8 * 18f;
            spot = (30 + k, 30 + k);
        }

        var inst = Make(type, spot.X, spot.Y, spec.DefaultW, spec.DefaultH);
        Widgets.Add(inst);
        MarkModified();
        return inst;
    }

    public bool Remove(WidgetInstance widget)
    {
        if (!Widgets.Remove(widget))
        {
            return false;
        }

        MarkModified();
        return true;
    }

    /// <summary>Moves a widget to the end of the draw order (on top).</summary>
    public void BringToFront(WidgetInstance widget)
    {
        var index = Widgets.IndexOf(widget);
        if (index >= 0 && index != Widgets.Count - 1)
        {
            Widgets.RemoveAt(index);
            Widgets.Add(widget);
            Revision++;
        }
    }

    // ---------------------------------------------------------------- persistence support

    /// <summary>Installs state read from disk (already validated by the store).</summary>
    public void Load(IEnumerable<WidgetInstance> widgets, OverlayGlobals globals, string presetName, bool modified)
    {
        Widgets = [.. widgets];
        Globals = globals;
        Globals.Clamp();
        PresetName = presetName;
        Modified = modified;
        var maxId = 0;
        foreach (var w in Widgets)
        {
            var hash = w.Id.LastIndexOf('#');
            if (hash >= 0 && int.TryParse(w.Id.AsSpan(hash + 1), NumberStyles.None, CultureInfo.InvariantCulture, out var n))
            {
                maxId = Math.Max(maxId, n);
            }
        }

        _nextId = Math.Max(_nextId, maxId + 1);
        Revision++;
    }

    public void LoadCustomPresets(IEnumerable<(string Name, List<WidgetInstance> Widgets)> presets)
    {
        CustomPresets.Clear();
        CustomOrder.Clear();
        foreach (var (name, widgets) in presets)
        {
            if (name.Length == 0 || IsBuiltIn(name) || CustomPresets.ContainsKey(name))
            {
                continue;
            }

            CustomPresets[name] = widgets;
            CustomOrder.Add(name);
        }
    }

    public void ResetEverything()
    {
        CustomPresets.Clear();
        CustomOrder.Clear();
        Globals = new OverlayGlobals();
        ApplyPreset(InspectorPreset);
    }
}
