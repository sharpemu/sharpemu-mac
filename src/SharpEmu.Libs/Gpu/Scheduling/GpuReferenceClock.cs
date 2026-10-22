// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using SharpEmu.Libs.Kernel;

namespace SharpEmu.Libs.Gpu.Scheduling;

/// <summary>
/// The guest GPU's 100 MHz reference clock, which end-of-pipe timestamps and the
/// command-stream interpreter's clock reads both report. Backend-neutral: it only scales the
/// host TSC, so it stays in the shared engine while the end-of-pipe recorder that writes the
/// value into a command buffer belongs to the graphics backend.
/// </summary>
public static class GpuReferenceClock
{
    public const ulong Frequency = 100_000_000;

    public static bool TryScale(ulong hostTicks, ulong hostFrequency, out ulong value)
    {
        value = 0;
        if (hostFrequency == 0)
        {
            return false;
        }

        var wholeSeconds = hostTicks / hostFrequency;
        var remainder = hostTicks % hostFrequency;
        if (wholeSeconds > ulong.MaxValue / Frequency || remainder > ulong.MaxValue / Frequency)
        {
            return false;
        }

        var wholeValue = wholeSeconds * Frequency;
        var fractionalValue = remainder * Frequency / hostFrequency;
        if (wholeValue > ulong.MaxValue - fractionalValue)
        {
            return false;
        }

        value = wholeValue + fractionalValue;
        return true;
    }

    public static ulong Read()
    {
        var hostFrequency = KernelRuntimeCompatExports.TscFrequency;
        var hostTicks = KernelRuntimeCompatExports.ReadTscCounter();
        if (!TryScale(hostTicks, hostFrequency, out var value))
        {
            throw GpuFatal.Fatal($"Cannot convert the host clock: ticks=0x{hostTicks:X16} frequency={hostFrequency}");
        }

        return DiagScale == 1.0 ? value : (ulong)(value * DiagScale);
    }

    // [WIP] diag: SHARPEMU_DIAG_GPU_CLOCK_SCALE=<factor> multiplies every guest GPU clock
    // read, so GPU spans the guest measures look that much shorter (or longer).
    private static readonly double DiagScale =
        double.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_DIAG_GPU_CLOCK_SCALE"),
            System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture,
            out var scale) && scale > 0 ? scale : 1.0;
}

/// <summary>
/// Reads words out of a global-data-share allocation. The storage is the backend's (a mapped
/// Vulkan buffer, a Metal buffer, or a plain managed array in the headless host), but the
/// bounds rules the guest sees are the same everywhere, so they live here.
/// </summary>
public static class GlobalDataShare
{
    public static void ReadWords(ReadOnlySpan<byte> gds, Span<uint> destination, uint wordOffset, uint wordCount)
    {
        var offset = (ulong)wordOffset * sizeof(uint);
        var size = (ulong)wordCount * sizeof(uint);
        if (gds.IsEmpty || offset > (ulong)gds.Length || size > (ulong)gds.Length - offset || destination.Length < wordCount)
        {
            throw GpuFatal.Fatal($"Cannot read the requested GDS range: offset={wordOffset} size={wordCount} gds={gds.Length}");
        }

        gds.Slice((int)offset, (int)size).CopyTo(MemoryMarshal.AsBytes(destination));
    }
}
