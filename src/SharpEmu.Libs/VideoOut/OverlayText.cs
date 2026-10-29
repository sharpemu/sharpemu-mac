// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;

namespace SharpEmu.Libs.VideoOut;

/// <summary>Everything the overlay shows, gathered once per refresh window.</summary>
internal readonly record struct OverlayStats(
    double Fps,
    uint RenderWidth,
    uint RenderHeight,
    ResourceSample Resources,
    double GpuBusyPercent,
    ulong? GpuAllocatedBytes,
    long ElapsedSeconds);

/// <summary>
/// The overlay's text, built once per refresh window (never per frame). Numbers are always
/// formatted with the invariant culture so the title does not change with the system locale.
/// </summary>
/// <remarks>
/// Title format, left to right, separated by " · ":
/// <c>26.8 FPS (37.3 ms) · 1920×1080 · CPU 38% · GPU 71% · RAM 6.1 GB · VRAM 4.8 GB · 00:06:37</c>.
/// CPU is the guest and GPU processes together, as a share of the whole machine. RAM is both
/// processes' physical footprint and VRAM is the Metal device's allocation. A figure that is not
/// known yet is left out (resolution, VRAM) or shown as "?" or "N/A".
/// </remarks>
internal static class OverlayText
{
    private const double MiB = 1024.0 * 1024.0;
    private const double GiB = 1024.0 * MiB;

    public static string FormatTitle(in OverlayStats stats)
    {
        var parts = new List<string>(7)
        {
            stats.Fps > 0 ? $"{Number(stats.Fps, "0.0")} FPS ({Number(1000.0 / stats.Fps, "0.0")} ms)" : "0.0 FPS",
        };
        if (stats.RenderWidth > 0 && stats.RenderHeight > 0)
        {
            parts.Add(string.Create(CultureInfo.InvariantCulture, $"{stats.RenderWidth}\u00D7{stats.RenderHeight}"));
        }

        parts.Add($"CPU {FormatPercent(stats.Resources.TotalCpuPercent)}");
        parts.Add($"GPU {PerfOverlay.FormatUsage(stats.GpuBusyPercent)}");
        parts.Add($"RAM {(stats.Resources.TotalFootprintBytes is { } footprint ? FormatBytes(footprint) : "N/A")}");
        if (stats.GpuAllocatedBytes is { } vram)
        {
            parts.Add($"VRAM {FormatBytes(vram)}");
        }

        parts.Add(FormatElapsed(stats.ElapsedSeconds));
        return string.Join(" \u00B7 ", parts);
    }

    /// <summary>"6.1 GB" from 1 GiB up, "640 MB" below.</summary>
    public static string FormatBytes(ulong bytes) =>
        bytes >= GiB ? $"{Scaled(bytes, true)} GB" : $"{Scaled(bytes, false)} MB";

    /// <summary>"HH:MM:SS".</summary>
    public static string FormatElapsed(long totalSeconds)
    {
        totalSeconds = Math.Max(0, totalSeconds);
        return string.Create(CultureInfo.InvariantCulture,
            $"{totalSeconds / 3600:00}:{totalSeconds / 60 % 60:00}:{totalSeconds % 60:00}");
    }

    private static string Scaled(ulong bytes, bool gigabytes) =>
        gigabytes ? Number(bytes / GiB, "0.0") : Number(bytes / MiB, "0");

    /// <summary>"38%", or "?" when the figure could not be measured.</summary>
    public static string FormatPercent(double percent, bool suffix = true) =>
        !double.IsFinite(percent) ? "?" : Number(Math.Clamp(percent, 0, 100), "0") + (suffix ? "%" : string.Empty);

    private static string Number(double value, string format) =>
        value.ToString(format, CultureInfo.InvariantCulture);
}
