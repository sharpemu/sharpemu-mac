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
/// Title format, left to right, separated by " | ":
/// <c>FPS 26.8 (37.3 ms) | 1920x1080 | CPU 38% (guest 30, gpu 8) | GPU 71% | MEM 6.1 GB (gpu proc 3.2, VRAM 4.8) | TIME 00:06:37</c>.
/// CPU is a share of the whole machine; "guest" and "gpu" are the two processes' shares of it.
/// MEM is both processes' physical footprint; "gpu proc" is the GPU process's part and VRAM is
/// the Metal device's allocation. The parenthesised memory figures use the unit of the total.
/// </remarks>
internal static class OverlayText
{
    private const double MiB = 1024.0 * 1024.0;
    private const double GiB = 1024.0 * MiB;

    public static string FormatTitle(in OverlayStats stats)
    {
        var fps = Number(stats.Fps, "0.0");
        var frame = stats.Fps > 0 ? $" ({Number(1000.0 / stats.Fps, "0.0")} ms)" : string.Empty;
        var resolution = stats.RenderWidth > 0 && stats.RenderHeight > 0
            ? string.Create(CultureInfo.InvariantCulture, $" | {stats.RenderWidth}x{stats.RenderHeight}")
            : string.Empty;
        return $"FPS {fps}{frame}{resolution} | CPU {FormatCpu(stats.Resources)} | GPU {PerfOverlay.FormatUsage(stats.GpuBusyPercent)} | " +
            $"MEM {FormatMemory(stats.Resources, stats.GpuAllocatedBytes)} | TIME {FormatElapsed(stats.ElapsedSeconds)}";
    }

    /// <summary>"38% (guest 30, gpu 8)" with a guest process, "38%" without one.</summary>
    public static string FormatCpu(in ResourceSample sample)
    {
        var total = FormatPercent(sample.TotalCpuPercent);
        return sample.Split
            ? $"{total} (guest {FormatPercent(sample.GuestCpuPercent, suffix: false)}, gpu {FormatPercent(sample.GpuProcessCpuPercent, suffix: false)})"
            : total;
    }

    /// <summary>"6.1 GB (gpu proc 3.2, VRAM 4.8)", or "6.1 GB (VRAM 4.8)" when one process does both.</summary>
    public static string FormatMemory(in ResourceSample sample, ulong? gpuAllocatedBytes)
    {
        if (sample.TotalFootprintBytes is not { } bytes)
        {
            return "N/A";
        }

        var gigabytes = bytes >= GiB;
        var text = FormatBytes(bytes);
        var parts = new List<string>(2);
        if (sample.Split && sample.GpuProcessFootprintBytes is { } gpuProcess)
        {
            parts.Add($"gpu proc {Scaled(gpuProcess, gigabytes)}");
        }

        if (gpuAllocatedBytes is { } vram)
        {
            parts.Add($"VRAM {Scaled(vram, gigabytes)}");
        }

        return parts.Count == 0 ? text : $"{text} ({string.Join(", ", parts)})";
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
