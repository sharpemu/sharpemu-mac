// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;

namespace SharpEmu.Libs.VideoOut.Overlay;

public enum OverlayWidgetType : byte
{
    Fps,
    Strip,
    Budget,
    Timeline,
    History,
    TopCosts,
    Split,
    Cpu,
    Gpu,
    Memory,
    Session,
}

public enum OverlayTheme : byte
{
    Graphite,
    Oled,
    Glass,
}

public enum WidgetSettingKind : byte
{
    Toggle,
    Choice,
    Range,
}

/// <summary>One option of a choice setting; <see cref="Number"/> carries numeric options (60, 120, ...).</summary>
public readonly record struct SettingOption(string Id, string Label, float Number = 0f);

/// <summary>One entry of a widget's schema-driven settings; values live in <see cref="WidgetInstance.Values"/>.</summary>
public sealed class WidgetSetting
{
    public WidgetSetting(string key, string label, WidgetSettingKind kind, float defaultValue,
        SettingOption[]? options = null, float min = 0, float max = 1, float step = 1)
    {
        Key = key;
        Label = label;
        Kind = kind;
        Default = defaultValue;
        Options = options ?? [];
        Min = min;
        Max = max;
        Step = step;
    }

    public string Key { get; }
    public string Label { get; }
    public WidgetSettingKind Kind { get; }
    /// <summary>Toggle: 0/1. Choice: option index. Range: the value.</summary>
    public float Default { get; }
    public SettingOption[] Options { get; }
    public float Min { get; }
    public float Max { get; }
    public float Step { get; }

    public static WidgetSetting Toggle(string key, string label, bool value) =>
        new(key, label, WidgetSettingKind.Toggle, value ? 1f : 0f);

    public static WidgetSetting Range(string key, string label, float value, float min, float max, float step = 1f) =>
        new(key, label, WidgetSettingKind.Range, value, null, min, max, step);

    public static WidgetSetting Choice(string key, string label, int defaultIndex, params SettingOption[] options) =>
        new(key, label, WidgetSettingKind.Choice, defaultIndex, options);

    public float Clamp(float value) => Kind switch
    {
        WidgetSettingKind.Toggle => value >= 0.5f ? 1f : 0f,
        WidgetSettingKind.Choice => Math.Clamp(MathF.Round(value), 0, Math.Max(0, Options.Length - 1)),
        _ => Math.Clamp(Step > 0 ? MathF.Round(value / Step) * Step : value, Min, Max),
    };
}

/// <summary>Static description of one widget type: title, default and minimum size, settings schema.</summary>
public sealed class WidgetSpec
{
    public WidgetSpec(OverlayWidgetType type, string key, string title, string description, float defaultW, float defaultH,
        float minW, float minH, WidgetSetting[] settings)
    {
        Type = type;
        Key = key;
        Title = title;
        Description = description;
        DefaultW = defaultW;
        DefaultH = defaultH;
        MinW = minW;
        MinH = minH;
        Settings = settings;
    }

    public OverlayWidgetType Type { get; }
    /// <summary>The lowercase name used in layout files.</summary>
    public string Key { get; }
    public string Title { get; }
    public string Description { get; }
    /// <summary>Sizes are in the 960x540 reference space.</summary>
    public float DefaultW { get; }
    public float DefaultH { get; }
    public float MinW { get; }
    public float MinH { get; }
    public WidgetSetting[] Settings { get; }

    public int IndexOf(string key)
    {
        for (var i = 0; i < Settings.Length; i++)
        {
            if (string.Equals(Settings[i].Key, key, StringComparison.Ordinal))
            {
                return i;
            }
        }

        return -1;
    }
}

/// <summary>Setting indices of each widget, so draw code reads them without string lookups.</summary>
public static class WidgetKeys
{
    public static class Fps { public const int Units = 0, Lows = 1, Spark = 2, Len = 3; }
    public static class Budget { public const int Legend = 0, Verdict = 1; }
    public static class Timeline { public const int Workers = 0, Labels = 1, Legend = 2; }
    public static class History { public const int Len = 0, View = 1, YMax = 2, Legend = 3; }
    public static class TopCosts { public const int Rows = 0, Sort = 1, Source = 2, AvgN = 3, Thread = 4, Waits = 5, Pct = 6, Bars = 7, Count = 8; }
    public static class Split { public const int Labels = 0, Compact = 1, Waits = 2; }
    public static class Cpu { public const int Total = 0, Rows = 1, Workers = 2; }
    public static class Gpu { public const int Res = 0, Draws = 1, Graph = 2; }
    public static class Memory { public const int Proc = 0, Gpu = 1, Cache = 2, Alloc = 3, Gc = 4, Trend = 5; }
    public static class Strip { public const int Units = 0, Legend = 1, Spark = 2, Stats = 3, Len = 4; }
    public static class Session { public const int Elapsed = 0, Frame = 1, Title = 2, Backend = 3, Os = 4; }
}

/// <summary>The registry of widget types (the prototype's WT table).</summary>
public static class WidgetCatalog
{
    private static SettingOption[] Numbers(params int[] values) =>
        Array.ConvertAll(values, v => new SettingOption(v.ToString(CultureInfo.InvariantCulture),
            v.ToString(CultureInfo.InvariantCulture), v));

    private static readonly SettingOption[] FpsMs = [new("fps", "fps"), new("ms", "ms")];

    private static readonly WidgetSpec[] Specs =
    [
        new(OverlayWidgetType.Fps, "fps", "FPS", "Big fps number, 1% / 0.1% lows and a frame-time sparkline", 150, 96, 104, 58,
        [
            WidgetSetting.Choice("units", "Primary unit", 0, FpsMs),
            WidgetSetting.Toggle("lows", "Show 1% / 0.1% lows", true),
            WidgetSetting.Toggle("spark", "Show sparkline", true),
            WidgetSetting.Choice("len", "Sparkline frames", 1, Numbers(60, 120, 240)),
        ]),
        new(OverlayWidgetType.Strip, "strip", "Compact strip", "FPS, frame ms, lane budget bars, sparkline and one stat line", 440, 112, 300, 76,
        [
            WidgetSetting.Choice("units", "Primary unit", 0, FpsMs),
            WidgetSetting.Toggle("legend", "Show lane labels", true),
            WidgetSetting.Toggle("spark", "Show sparkline", true),
            WidgetSetting.Toggle("stats", "Show stat line", true),
            WidgetSetting.Choice("len", "Sparkline frames", 1, Numbers(60, 120, 240)),
        ]),
        new(OverlayWidgetType.Budget, "budget", "Frame budget", "Per-lane busy time (guest, workers, render, GPU) against the frame budget, with a bottleneck verdict", 320, 100, 200, 70,
        [
            WidgetSetting.Toggle("legend", "Show lane values", true),
            WidgetSetting.Toggle("verdict", "Show bottleneck verdict", true),
        ]),
        new(OverlayWidgetType.Timeline, "timeline", "Last frame · threads", "Per-thread timeline of the frame with phase-coloured segments, hover for details", 380, 132, 220, 90,
        [
            WidgetSetting.Toggle("workers", "Show worker lane", true),
            WidgetSetting.Toggle("labels", "Segment labels", true),
            WidgetSetting.Toggle("legend", "Show legend", true),
        ]),
        new(OverlayWidgetType.History, "history", "Frame time history", "Frame-time history coloured by cost kind; click a bar to inspect that frame", 360, 112, 200, 70,
        [
            WidgetSetting.Choice("len", "Frames shown", 1, Numbers(60, 120, 180, 240)),
            WidgetSetting.Choice("view", "Chart", 0, new SettingOption("stack", "Stacked (ms)"), new SettingOption("line", "Line (ms)"), new SettingOption("fps", "Line (fps)")),
            WidgetSetting.Choice("ymax", "Y axis", 0, new SettingOption("auto", "Auto"), new SettingOption("2x", "2× budget")),
            WidgetSetting.Toggle("legend", "Show legend", true),
        ]),
        new(OverlayWidgetType.TopCosts, "topcosts", "Top costs / frame", "Ranked render phases, HLE calls and GPU work; sort by self-time, total or count", 280, 176, 200, 90,
        [
            WidgetSetting.Range("n", "Rows (top-N)", 6, 3, 15),
            WidgetSetting.Choice("sort", "Sort by", 0, new SettingOption("self", "Self time"), new SettingOption("total", "Total time"), new SettingOption("count", "Call count")),
            WidgetSetting.Choice("src", "Source", 0, new SettingOption("frame", "Selected / last frame"), new SettingOption("avg", "Average of last N")),
            WidgetSetting.Choice("avgN", "Average over", 1, Numbers(30, 60, 120)),
            WidgetSetting.Choice("thread", "Threads", 0, new SettingOption("all", "All"), new SettingOption("main", "Guest main"), new SettingOption("workers", "Workers"), new SettingOption("render", "Render"), new SettingOption("gpu", "GPU")),
            WidgetSetting.Toggle("waits", "Include waits", true),
            WidgetSetting.Toggle("pct", "% of frame column", true),
            WidgetSetting.Toggle("bars", "Inline bars", true),
            WidgetSetting.Toggle("count", "Call count column", false),
        ]),
        new(OverlayWidgetType.Split, "split", "Frame split", "Donut of work by cost kind: guest, HLE, render, shader, GPU, wait", 200, 110, 140, 80,
        [
            WidgetSetting.Choice("labels", "Legend values", 0, new SettingOption("ms", "Milliseconds"), new SettingOption("pct", "Percent")),
            WidgetSetting.Toggle("compact", "Compact (hide extras)", false),
            WidgetSetting.Toggle("waits", "Include waits", true),
        ]),
        new(OverlayWidgetType.Cpu, "cpu", "CPU · per thread", "Per-thread CPU load with total machine CPU", 220, 146, 160, 90,
        [
            WidgetSetting.Toggle("total", "Show total CPU", true),
            WidgetSetting.Range("rows", "Thread rows", 5, 3, 12),
            WidgetSetting.Toggle("workers", "Include worker threads", true),
        ]),
        new(OverlayWidgetType.Gpu, "gpu", "GPU", "GPU utilisation, GPU ms, draws, dispatches and render resolution", 190, 124, 140, 80,
        [
            WidgetSetting.Toggle("res", "Show resolution", true),
            WidgetSetting.Toggle("draws", "Show draws / dispatches", true),
            WidgetSetting.Toggle("graph", "Show utilisation graph", true),
        ]),
        new(OverlayWidgetType.Memory, "memory", "Memory", "Process, GPU allocation, caches, allocation rate and GC counts with a trend line", 230, 150, 160, 90,
        [
            WidgetSetting.Toggle("proc", "Process", true),
            WidgetSetting.Toggle("gpu", "GPU alloc", true),
            WidgetSetting.Toggle("cache", "Image / buffer cache", true),
            WidgetSetting.Toggle("alloc", "Alloc rate", true),
            WidgetSetting.Toggle("gc", "GC counts", true),
            WidgetSetting.Toggle("trend", "Trend line", true),
        ]),
        new(OverlayWidgetType.Session, "session", "Session", "Elapsed time, frame number, game title, backend and OS", 190, 106, 140, 70,
        [
            WidgetSetting.Toggle("elapsed", "Elapsed", true),
            WidgetSetting.Toggle("frame", "Frame #", true),
            WidgetSetting.Toggle("title", "Title", false),
            WidgetSetting.Toggle("backend", "Backend", true),
            WidgetSetting.Toggle("os", "OS", true),
        ]),
    ];

    /// <summary>The order widgets are offered in the add-widget picker.</summary>
    public static IReadOnlyList<WidgetSpec> All => Specs;

    public static WidgetSpec Get(OverlayWidgetType type)
    {
        foreach (var spec in Specs)
        {
            if (spec.Type == type)
            {
                return spec;
            }
        }

        throw new ArgumentOutOfRangeException(nameof(type));
    }

    public static bool TryGet(string key, out WidgetSpec spec)
    {
        foreach (var candidate in Specs)
        {
            if (string.Equals(candidate.Key, key, StringComparison.OrdinalIgnoreCase))
            {
                spec = candidate;
                return true;
            }
        }

        spec = null!;
        return false;
    }
}

/// <summary>One widget on the overlay: type, normalised placement and its setting values.</summary>
public sealed class WidgetInstance
{
    public WidgetInstance(string id, OverlayWidgetType type)
    {
        Id = id;
        Type = type;
        Spec = WidgetCatalog.Get(type);
        Values = new float[Spec.Settings.Length];
        for (var i = 0; i < Values.Length; i++)
        {
            Values[i] = Spec.Settings[i].Default;
        }
    }

    public string Id { get; set; }
    public OverlayWidgetType Type { get; }
    public WidgetSpec Spec { get; }
    /// <summary>Left edge as a fraction of the surface width.</summary>
    public float X { get; set; }
    /// <summary>Top edge as a fraction of the surface height.</summary>
    public float Y { get; set; }
    /// <summary>Width as a fraction of the 960-unit reference width (before the user scale).</summary>
    public float W { get; set; }
    /// <summary>Height as a fraction of the 540-unit reference height (before the user scale).</summary>
    public float H { get; set; }
    public float[] Values { get; }

    public bool Bool(int index) => Values[index] >= 0.5f;
    public int Choice(int index) => (int)Values[index];
    public int Int(int index) => (int)MathF.Round(Values[index]);
    public float Number(int index) => Spec.Settings[index].Kind == WidgetSettingKind.Choice
        ? Spec.Settings[index].Options[Math.Clamp(Choice(index), 0, Spec.Settings[index].Options.Length - 1)].Number
        : Values[index];
    public string ChoiceId(int index) =>
        Spec.Settings[index].Options[Math.Clamp(Choice(index), 0, Spec.Settings[index].Options.Length - 1)].Id;

    public void SetValue(int index, float value) => Values[index] = Spec.Settings[index].Clamp(value);

    public WidgetInstance Clone(string? id = null)
    {
        var copy = new WidgetInstance(id ?? Id, Type) { X = X, Y = Y, W = W, H = H };
        Array.Copy(Values, copy.Values, Values.Length);
        return copy;
    }
}

public sealed class OverlayGlobals
{
    public const float MinScale = 0.75f, MaxScale = 1.5f, MinFont = 9f, MaxFont = 14f, MinOpacity = 0.3f;

    /// <summary>Overall alpha, 0.3..1.</summary>
    public float Opacity { get; set; } = 0.92f;
    /// <summary>User size multiplier, 0.75..1.5.</summary>
    public float Scale { get; set; } = 1f;
    /// <summary>Base font size in reference units, 9..14.</summary>
    public float FontSize { get; set; } = 11f;
    public OverlayTheme Theme { get; set; } = OverlayTheme.Graphite;

    public void Clamp()
    {
        Opacity = float.IsFinite(Opacity) ? Math.Clamp(Opacity, MinOpacity, 1f) : 0.92f;
        Scale = float.IsFinite(Scale) ? Math.Clamp(Scale, MinScale, MaxScale) : 1f;
        FontSize = float.IsFinite(FontSize) ? Math.Clamp(FontSize, MinFont, MaxFont) : 11f;
        if (!Enum.IsDefined(Theme))
        {
            Theme = OverlayTheme.Graphite;
        }
    }

    public OverlayGlobals Clone() => new() { Opacity = Opacity, Scale = Scale, FontSize = FontSize, Theme = Theme };
}
