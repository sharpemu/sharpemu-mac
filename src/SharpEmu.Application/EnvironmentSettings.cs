// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Application;

public enum EnvironmentSettingKind
{
    /// <summary>On or off; the emulator's own default decides which state sends nothing.</summary>
    Toggle,

    /// <summary>One of <see cref="EnvironmentSetting.Choices"/>.</summary>
    Choice,
}

public enum EnvironmentSettingGroup
{
    Performance,
    Compatibility,
    Audio,
    Debug,
}

/// <param name="Value">What the variable is set to. The default one is never sent.</param>
/// <param name="Label">Language-neutral label ("4 GB"), or null to look the label up per language.</param>
public sealed record EnvironmentSettingChoice(string Value, string? Label = null);

/// <summary>
/// One emulator switch the launcher exposes. The emulator's default is authoritative: a value
/// equal to it is stored as no entry at all, so the launcher never pins a default the emulator
/// later changes.
/// </summary>
/// <param name="Id">Names the setting's strings ("Setting.&lt;Id&gt;.Label") and its controls.</param>
/// <param name="Variable">The environment variable the emulator reads.</param>
/// <param name="DefaultEnabled">Toggles only: the state the emulator is in when the variable is absent.</param>
/// <param name="OffValue">Toggles only: what the variable is set to for the state that is not the default.</param>
/// <param name="DefaultChoice">Choices only: the value the emulator uses when the variable is absent.</param>
/// <param name="MacOnly">Read only by the Metal backend or macOS host, so other platforms hide it.</param>
public sealed record EnvironmentSetting(
    string Id,
    string Variable,
    EnvironmentSettingKind Kind,
    EnvironmentSettingGroup Group,
    bool DefaultEnabled = false,
    string OffValue = "0",
    string DefaultChoice = "",
    IReadOnlyList<EnvironmentSettingChoice>? Choices = null,
    bool MacOnly = false);

public static class EnvironmentSettings
{
    private static readonly string[] OffValues = ["0", "off", "false", "no", "disabled"];

    public static IReadOnlyList<EnvironmentSetting> All { get; } =
    [
        // Performance.
        Toggle("ShaderCache", "SHARPEMU_SHADER_CACHE", EnvironmentSettingGroup.Performance, enabled: true, offValue: "off"),
        Toggle("ShaderPrewarm", "SHARPEMU_SHADER_PREWARM", EnvironmentSettingGroup.Performance, enabled: true),
        Toggle("ShaderPrewarmWait", "SHARPEMU_SHADER_PREWARM_WAIT", EnvironmentSettingGroup.Performance, enabled: true),
        Toggle("BinaryArchive", "SHARPEMU_METAL_V2_BINARY_ARCHIVE", EnvironmentSettingGroup.Performance, enabled: true, macOnly: true),
        Choice("BinaryArchiveBudget", "SHARPEMU_METAL_V2_BINARY_ARCHIVE_BUDGET_MB", EnvironmentSettingGroup.Performance, "8192",
            [new("2048", "2 GB"), new("4096", "4 GB"), new("8192", "8 GB"), new("16384", "16 GB"), new("32768", "32 GB")],
            macOnly: true),
        Choice("GpuProcess", "SHARPEMU_GPU_PROCESS", EnvironmentSettingGroup.Performance, "auto",
            [new("auto"), new("1"), new("0")], macOnly: true),
        Toggle("NoFlipPacing", "SHARPEMU_NO_FLIP_PACING", EnvironmentSettingGroup.Performance, enabled: false),
        Toggle("SpinNap", "SHARPEMU_SPIN_NAP", EnvironmentSettingGroup.Performance, enabled: true),
        Choice("ImageBudget", "SHARPEMU_METAL_V2_IMAGE_BUDGET_MB", EnvironmentSettingGroup.Performance, "auto",
            [new("auto"), new("2048", "2 GB"), new("4096", "4 GB"), new("6144", "6 GB"), new("8192", "8 GB"),
             new("12288", "12 GB"), new("16384", "16 GB")],
            macOnly: true),
        Toggle("LossyTextures", "SHARPEMU_METAL_V2_LOSSY_TEXTURES", EnvironmentSettingGroup.Performance, enabled: false, macOnly: true),

        // Compatibility.
        Toggle("MeshShaders", "SHARPEMU_METAL_MESH_SHADERS", EnvironmentSettingGroup.Compatibility, enabled: true, macOnly: true),
        Toggle("Tessellation", "SHARPEMU_TESSELLATION", EnvironmentSettingGroup.Compatibility, enabled: true),
        Toggle("SingleCopy", "SHARPEMU_METAL_V2_SINGLE_COPY", EnvironmentSettingGroup.Compatibility, enabled: true, macOnly: true),
        Toggle("UeBackbufferPatch", "SHARPEMU_UE_BACKBUFFER_PATCH", EnvironmentSettingGroup.Compatibility, enabled: true),
        Choice("MovieClock", "SHARPEMU_MOVIE_CLOCK", EnvironmentSettingGroup.Compatibility, "audio",
            [new("audio"), new("wall")]),
        Toggle("IgnoreGuestExceptions", "SHARPEMU_IGNORE_GUEST_EXCEPTIONS", EnvironmentSettingGroup.Compatibility, enabled: false),

        // Audio.
        Choice("AudioLatency", "SHARPEMU_AUDIO_LATENCY_MS", EnvironmentSettingGroup.Audio, "60",
            [new("20", "20 ms"), new("30", "30 ms"), new("40", "40 ms"), new("60", "60 ms"), new("90", "90 ms"),
             new("120", "120 ms"), new("200", "200 ms")]),

        // Debug.
        Toggle("LogVideoOutFps", "SHARPEMU_LOG_VIDEOOUT_FPS", EnvironmentSettingGroup.Debug, enabled: false),
        Toggle("LogMemory", "SHARPEMU_LOG_MEMORY", EnvironmentSettingGroup.Debug, enabled: false),
        Toggle("FrameTimes", "SHARPEMU_FRAME_TIMES", EnvironmentSettingGroup.Debug, enabled: false),
    ];

    public static EnvironmentSetting? Find(string variable)
    {
        foreach (var setting in All)
        {
            if (string.Equals(setting.Variable, variable, StringComparison.OrdinalIgnoreCase))
            {
                return setting;
            }
        }

        return null;
    }

    /// <summary>Whether <paramref name="entry"/> ("NAME" or "NAME=value") belongs to a setting above.</summary>
    public static bool IsManagedEntry(string entry) =>
        Find(entry.Split('=', 2, StringSplitOptions.TrimEntries)[0]) is not null;

    /// <summary>The state of a toggle: the last entry for its variable, else the emulator's default.</summary>
    public static bool IsEnabled(IEnumerable<string> entries, EnvironmentSetting setting)
    {
        var enabled = setting.DefaultEnabled;
        foreach (var entry in entries)
        {
            var parts = entry.Split('=', 2, StringSplitOptions.TrimEntries);
            if (string.Equals(parts[0], setting.Variable, StringComparison.OrdinalIgnoreCase))
            {
                enabled = parts.Length == 1 ||
                    !OffValues.Contains(parts[1], StringComparer.OrdinalIgnoreCase);
            }
        }

        return enabled;
    }

    /// <summary>The selected value of a choice: the last entry naming one of its choices, else the default.</summary>
    public static string GetChoice(IEnumerable<string> entries, EnvironmentSetting setting)
    {
        var selected = setting.DefaultChoice;
        foreach (var entry in entries)
        {
            var parts = entry.Split('=', 2, StringSplitOptions.TrimEntries);
            if (parts.Length == 2 &&
                string.Equals(parts[0], setting.Variable, StringComparison.OrdinalIgnoreCase))
            {
                selected = parts[1];
            }
        }

        return selected;
    }

    /// <summary>Stores a toggle's state: nothing for the default, an entry for the other state.</summary>
    public static void SetEnabled(List<string> entries, EnvironmentSetting setting, bool enabled)
    {
        Remove(entries, setting);
        if (enabled == setting.DefaultEnabled)
        {
            return;
        }

        entries.Add(enabled ? setting.Variable : $"{setting.Variable}={setting.OffValue}");
    }

    /// <summary>Stores a choice: nothing for the default value, an entry for any other.</summary>
    public static void SetChoice(List<string> entries, EnvironmentSetting setting, string value)
    {
        Remove(entries, setting);
        if (!string.Equals(value, setting.DefaultChoice, StringComparison.OrdinalIgnoreCase) &&
            !string.IsNullOrWhiteSpace(value))
        {
            entries.Add($"{setting.Variable}={value}");
        }
    }

    /// <summary>
    /// The normalized entry for <paramref name="setting"/> inside <paramref name="entries"/>, or null
    /// while it is at the emulator's default. Two entry lists that differ only in spelling agree.
    /// </summary>
    internal static string? Canonical(IEnumerable<string> entries, EnvironmentSetting setting) =>
        setting.Kind == EnvironmentSettingKind.Toggle
            ? IsEnabled(entries, setting) == setting.DefaultEnabled
                ? null
                : $"{setting.Variable}={(setting.DefaultEnabled ? setting.OffValue : "1")}"
            : string.Equals(GetChoice(entries, setting), setting.DefaultChoice, StringComparison.OrdinalIgnoreCase)
                ? null
                : $"{setting.Variable}={GetChoice(entries, setting)}";

    private static void Remove(List<string> entries, EnvironmentSetting setting) =>
        entries.RemoveAll(entry => string.Equals(
            entry.Split('=', 2, StringSplitOptions.TrimEntries)[0],
            setting.Variable,
            StringComparison.OrdinalIgnoreCase));

    private static EnvironmentSetting Toggle(
        string id,
        string variable,
        EnvironmentSettingGroup group,
        bool enabled,
        string offValue = "0",
        bool macOnly = false) =>
        new(id, variable, EnvironmentSettingKind.Toggle, group, DefaultEnabled: enabled, OffValue: offValue, MacOnly: macOnly);

    private static EnvironmentSetting Choice(
        string id,
        string variable,
        EnvironmentSettingGroup group,
        string defaultChoice,
        EnvironmentSettingChoice[] choices,
        bool macOnly = false) =>
        new(id, variable, EnvironmentSettingKind.Choice, group, DefaultChoice: defaultChoice, Choices: choices, MacOnly: macOnly);
}
