// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text.Json;

namespace SharpEmu.Libs.VideoOut.Overlay;

/// <summary>
/// Reads and writes the overlay's layout and presets as JSON (hand-rolled with Utf8JsonWriter and
/// JsonDocument, so there is no reflection serializer to trim). Loading is tolerant: any problem
/// leaves the model at its defaults. Saving writes a temp file next to the target and moves it over.
/// </summary>
public sealed class OverlayLayoutStore
{
    private const int Version = 1;

    public OverlayLayoutStore(string directory)
    {
        Directory = directory;
        LayoutPath = Path.Combine(directory, "layout.json");
        PresetsPath = Path.Combine(directory, "presets.json");
    }

    public string Directory { get; }
    public string LayoutPath { get; }
    public string PresetsPath { get; }

    /// <summary>Loads custom presets, then the layout; true when a valid saved layout was applied.</summary>
    public bool LoadInto(OverlayLayoutModel model)
    {
        try
        {
            if (File.Exists(PresetsPath))
            {
                LoadPresets(model);
            }
        }
        catch (Exception)
        {
            model.CustomPresets.Clear();
            model.CustomOrder.Clear();
        }

        try
        {
            if (!File.Exists(LayoutPath))
            {
                return false;
            }

            using var doc = JsonDocument.Parse(File.ReadAllBytes(LayoutPath));
            var root = doc.RootElement;
            if (root.ValueKind != JsonValueKind.Object || !root.TryGetProperty("widgets", out var widgetsEl) ||
                widgetsEl.ValueKind != JsonValueKind.Array)
            {
                return false;
            }

            var widgets = ReadWidgets(widgetsEl, model);
            var globals = new OverlayGlobals();
            if (root.TryGetProperty("globals", out var g) && g.ValueKind == JsonValueKind.Object)
            {
                globals.Opacity = Number(g, "opacity", globals.Opacity);
                globals.Scale = Number(g, "scale", globals.Scale);
                globals.FontSize = Number(g, "fontSize", globals.FontSize);
                if (g.TryGetProperty("theme", out var t) && t.ValueKind == JsonValueKind.String &&
                    Enum.TryParse<OverlayTheme>(t.GetString(), true, out var theme) && Enum.IsDefined(theme))
                {
                    globals.Theme = theme;
                }
            }

            var preset = root.TryGetProperty("preset", out var p) && p.ValueKind == JsonValueKind.String
                ? p.GetString() ?? OverlayLayoutModel.InspectorPreset : OverlayLayoutModel.InspectorPreset;
            var modified = root.TryGetProperty("modified", out var m) && m.ValueKind == JsonValueKind.True;
            model.Load(widgets, globals, preset, modified);
            return true;
        }
        catch (Exception)
        {
            return false;
        }
    }

    private void LoadPresets(OverlayLayoutModel model)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(PresetsPath));
        if (doc.RootElement.ValueKind != JsonValueKind.Object ||
            !doc.RootElement.TryGetProperty("presets", out var presets) || presets.ValueKind != JsonValueKind.Array)
        {
            return;
        }

        var list = new List<(string, List<WidgetInstance>)>();
        foreach (var preset in presets.EnumerateArray())
        {
            if (preset.ValueKind != JsonValueKind.Object || !preset.TryGetProperty("name", out var name) ||
                name.ValueKind != JsonValueKind.String || !preset.TryGetProperty("widgets", out var widgets) ||
                widgets.ValueKind != JsonValueKind.Array)
            {
                continue;
            }

            list.Add((name.GetString() ?? string.Empty, ReadWidgets(widgets, model)));
        }

        model.LoadCustomPresets(list);
    }

    private static List<WidgetInstance> ReadWidgets(JsonElement array, OverlayLayoutModel model)
    {
        var result = new List<WidgetInstance>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var el in array.EnumerateArray())
        {
            if (el.ValueKind != JsonValueKind.Object || !el.TryGetProperty("type", out var typeEl) ||
                typeEl.ValueKind != JsonValueKind.String || !WidgetCatalog.TryGet(typeEl.GetString() ?? string.Empty, out var spec))
            {
                continue;
            }

            var id = el.TryGetProperty("id", out var idEl) && idEl.ValueKind == JsonValueKind.String
                ? idEl.GetString() ?? string.Empty : string.Empty;
            if (id.Length == 0 || !seen.Add(id))
            {
                id = model.NewId(spec.Type);
                seen.Add(id);
            }

            var inst = new WidgetInstance(id, spec.Type)
            {
                X = Math.Clamp(Number(el, "x", 0f), 0f, 1f),
                Y = Math.Clamp(Number(el, "y", 0f), 0f, 1f),
                W = Math.Clamp(Number(el, "w", spec.DefaultW / OverlayMetrics.RefWidth), spec.MinW / OverlayMetrics.RefWidth * 0.5f, 1f),
                H = Math.Clamp(Number(el, "h", spec.DefaultH / OverlayMetrics.RefHeight), spec.MinH / OverlayMetrics.RefHeight * 0.5f, 1f),
            };
            if (el.TryGetProperty("settings", out var settings) && settings.ValueKind == JsonValueKind.Object)
            {
                for (var i = 0; i < spec.Settings.Length; i++)
                {
                    var def = spec.Settings[i];
                    if (!settings.TryGetProperty(def.Key, out var v))
                    {
                        continue;
                    }

                    switch (def.Kind)
                    {
                        case WidgetSettingKind.Toggle:
                            if (v.ValueKind is JsonValueKind.True or JsonValueKind.False)
                            {
                                inst.SetValue(i, v.ValueKind == JsonValueKind.True ? 1f : 0f);
                            }

                            break;
                        case WidgetSettingKind.Choice:
                            var wanted = v.ValueKind == JsonValueKind.String ? v.GetString()
                                : v.ValueKind == JsonValueKind.Number ? v.GetRawText() : null;
                            for (var o = 0; wanted is not null && o < def.Options.Length; o++)
                            {
                                if (string.Equals(def.Options[o].Id, wanted, StringComparison.Ordinal))
                                {
                                    inst.SetValue(i, o);
                                    break;
                                }
                            }

                            break;
                        default:
                            if (v.ValueKind == JsonValueKind.Number && v.TryGetSingle(out var f) && float.IsFinite(f))
                            {
                                inst.SetValue(i, f);
                            }

                            break;
                    }
                }
            }

            result.Add(inst);
        }

        return result;
    }

    private static float Number(JsonElement obj, string name, float fallback) =>
        obj.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number && v.TryGetSingle(out var f) && float.IsFinite(f)
            ? f : fallback;

    // ---------------------------------------------------------------- save

    /// <summary>Writes layout.json and presets.json; returns false (never throws) when the disk refuses.</summary>
    public bool Save(OverlayLayoutModel model)
    {
        try
        {
            System.IO.Directory.CreateDirectory(Directory);
            WriteAtomic(LayoutPath, w =>
            {
                w.WriteStartObject();
                w.WriteNumber("version", Version);
                w.WriteString("preset", model.PresetName);
                w.WriteBoolean("modified", model.Modified);
                w.WriteStartObject("globals");
                w.WriteNumber("opacity", model.Globals.Opacity);
                w.WriteNumber("scale", model.Globals.Scale);
                w.WriteNumber("fontSize", model.Globals.FontSize);
                w.WriteString("theme", model.Globals.Theme.ToString().ToLowerInvariant());
                w.WriteEndObject();
                w.WritePropertyName("widgets");
                WriteWidgets(w, model.Widgets);
                w.WriteEndObject();
            });
            WriteAtomic(PresetsPath, w =>
            {
                w.WriteStartObject();
                w.WriteNumber("version", Version);
                w.WriteStartArray("presets");
                foreach (var name in model.CustomOrder)
                {
                    if (!model.CustomPresets.TryGetValue(name, out var widgets))
                    {
                        continue;
                    }

                    w.WriteStartObject();
                    w.WriteString("name", name);
                    w.WritePropertyName("widgets");
                    WriteWidgets(w, widgets);
                    w.WriteEndObject();
                }

                w.WriteEndArray();
                w.WriteEndObject();
            });
            return true;
        }
        catch (Exception)
        {
            return false;
        }
    }

    private static void WriteWidgets(Utf8JsonWriter w, List<WidgetInstance> widgets)
    {
        w.WriteStartArray();
        foreach (var inst in widgets)
        {
            w.WriteStartObject();
            w.WriteString("id", inst.Id);
            w.WriteString("type", inst.Spec.Key);
            w.WriteNumber("x", inst.X);
            w.WriteNumber("y", inst.Y);
            w.WriteNumber("w", inst.W);
            w.WriteNumber("h", inst.H);
            w.WriteStartObject("settings");
            for (var i = 0; i < inst.Spec.Settings.Length; i++)
            {
                var def = inst.Spec.Settings[i];
                switch (def.Kind)
                {
                    case WidgetSettingKind.Toggle:
                        w.WriteBoolean(def.Key, inst.Bool(i));
                        break;
                    case WidgetSettingKind.Choice:
                        w.WriteString(def.Key, inst.ChoiceId(i));
                        break;
                    default:
                        w.WriteNumber(def.Key, inst.Values[i]);
                        break;
                }
            }

            w.WriteEndObject();
            w.WriteEndObject();
        }

        w.WriteEndArray();
    }

    private static void WriteAtomic(string path, Action<Utf8JsonWriter> write)
    {
        var temp = path + ".tmp";
        using (var stream = new FileStream(temp, FileMode.Create, FileAccess.Write, FileShare.None))
        using (var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true }))
        {
            write(writer);
            writer.Flush();
        }

        File.Move(temp, path, overwrite: true);
    }
}
