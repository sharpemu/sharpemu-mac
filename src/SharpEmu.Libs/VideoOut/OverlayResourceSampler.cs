// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Runtime.InteropServices;

namespace SharpEmu.Libs.VideoOut;

/// <summary>One reading of a process: CPU time used since it started, and its memory footprint.</summary>
internal readonly record struct ProcessUsage(double CpuSeconds, ulong FootprintBytes);

/// <summary>
/// CPU and memory of the whole emulator over one sample window.
/// </summary>
/// <remarks>
/// CPU convention: every percentage is the share of the machine's total capacity, that is
/// CPU-seconds used / wall seconds / logical cores. 100% means every core was busy for the whole
/// window, so one core spinning on a 10-core machine reads 10%. The guest and GPU figures use the
/// same scale, so Total is their sum. Memory is each process's physical footprint (the figure
/// Activity Monitor shows). Unavailable values are NaN (CPU) or null (memory).
/// </remarks>
internal readonly record struct ResourceSample(
    bool Split,
    double TotalCpuPercent,
    double GuestCpuPercent,
    double GpuProcessCpuPercent,
    ulong? GuestFootprintBytes,
    ulong? GpuProcessFootprintBytes)
{
    /// <summary>Both processes' footprint together; null when either could not be read.</summary>
    public ulong? TotalFootprintBytes => Split
        ? GuestFootprintBytes is { } guest && GpuProcessFootprintBytes is { } gpu ? guest + gpu : null
        : GpuProcessFootprintBytes;
}

/// <summary>
/// Samples the guest process and the GPU process (this one) and turns CPU-time deltas into
/// machine-capacity percentages. Process readings are injected so the math is testable.
/// </summary>
internal sealed class OverlayResourceSampler
{
    private readonly int _guestProcessId;
    private readonly int _selfProcessId;
    private readonly int _logicalCores;
    private readonly Func<int, ProcessUsage?> _read;
    private double _guestCpuBaseline = double.NaN;
    private double _selfCpuBaseline = double.NaN;

    /// <param name="guestProcessId">The guest process in the GPU/guest split, or 0 when one process does both.</param>
    public OverlayResourceSampler(
        int guestProcessId,
        int selfProcessId,
        int logicalCores,
        Func<int, ProcessUsage?> read)
    {
        _guestProcessId = guestProcessId > 0 && guestProcessId != selfProcessId ? guestProcessId : 0;
        _selfProcessId = selfProcessId;
        _logicalCores = Math.Max(1, logicalCores);
        _read = read;
        _guestCpuBaseline = _guestProcessId != 0 ? _read(_guestProcessId)?.CpuSeconds ?? double.NaN : double.NaN;
        _selfCpuBaseline = _read(_selfProcessId)?.CpuSeconds ?? double.NaN;
    }

    /// <summary>Machine-capacity percent (0 to 100) for CPU-seconds used over wall seconds on a given core count.</summary>
    public static double CpuPercentOfMachine(double cpuSecondsDelta, double wallSeconds, int logicalCores) =>
        wallSeconds > 0 && cpuSecondsDelta >= 0
            ? Math.Min(100.0, cpuSecondsDelta / wallSeconds / Math.Max(1, logicalCores) * 100.0)
            : 0.0;

    /// <summary>Reads both processes and returns their usage since the previous call (or construction).</summary>
    public ResourceSample Sample(double wallSeconds)
    {
        var self = _read(_selfProcessId);
        var selfPercent = Delta(ref _selfCpuBaseline, self, wallSeconds);
        if (_guestProcessId == 0)
        {
            return new ResourceSample(false, selfPercent, double.NaN, selfPercent, null, self?.FootprintBytes);
        }

        var guest = _read(_guestProcessId);
        var guestPercent = Delta(ref _guestCpuBaseline, guest, wallSeconds);
        var total = double.IsNaN(guestPercent) || double.IsNaN(selfPercent) ? double.NaN : guestPercent + selfPercent;
        return new ResourceSample(true, total, guestPercent, selfPercent, guest?.FootprintBytes, self?.FootprintBytes);
    }

    private double Delta(ref double baseline, ProcessUsage? usage, double wallSeconds)
    {
        if (usage is not { } reading)
        {
            return double.NaN;
        }

        var previous = baseline;
        baseline = reading.CpuSeconds;
        return double.IsNaN(previous) ? double.NaN : CpuPercentOfMachine(reading.CpuSeconds - previous, wallSeconds, _logicalCores);
    }

    /// <summary>Creates a sampler that reads real processes.</summary>
    public static OverlayResourceSampler CreateForCurrentProcess(int guestProcessId) =>
        new(guestProcessId, Environment.ProcessId, Environment.ProcessorCount, ProcessUsageReader.TryRead);
}

/// <summary>Reads a process's CPU time and physical footprint from the operating system.</summary>
internal static class ProcessUsageReader
{
    public static ProcessUsage? TryRead(int processId) =>
        OperatingSystem.IsMacOS() ? ReadMac(processId) : ReadGeneric(processId);

    // proc_pid_rusage(RUSAGE_INFO_V4): CPU times in mach absolute-time ticks and ri_phys_footprint
    // (the footprint Activity Monitor shows). The GPU process is native arm64, so its timebase
    // is the one the kernel reports every process's ticks in, translated guest included.
    private static ProcessUsage? ReadMac(int processId)
    {
        // rusage_info_v4: ri_uuid[16], then ri_user_time at word 2, ri_system_time at 3, and
        // ri_phys_footprint at 9; 64 words is more than the whole structure.
        const int RusageInfoV4 = 4;
        const int UserTimeWord = 2;
        const int SystemTimeWord = 3;
        const int PhysFootprintWord = 9;
        var buffer = new ulong[64];
        try
        {
            if (proc_pid_rusage(processId, RusageInfoV4, ref buffer[0]) != 0)
            {
                return null;
            }
        }
        catch (Exception exception) when (exception is DllNotFoundException or EntryPointNotFoundException)
        {
            return null;
        }

        var ticks = buffer[UserTimeWord] + buffer[SystemTimeWord];
        return new ProcessUsage(MachClock.TicksToSeconds(ticks), buffer[PhysFootprintWord]);
    }

    private static ProcessUsage? ReadGeneric(int processId)
    {
        try
        {
            if (processId == Environment.ProcessId)
            {
                var cpu = Environment.CpuUsage;
                using var self = Process.GetCurrentProcess();
                return new ProcessUsage((cpu.UserTime + cpu.PrivilegedTime).TotalSeconds, (ulong)self.PrivateMemorySize64);
            }

            using var process = Process.GetProcessById(processId);
            return new ProcessUsage(process.TotalProcessorTime.TotalSeconds, (ulong)process.PrivateMemorySize64);
        }
        catch (Exception exception) when (exception is ArgumentException or InvalidOperationException
            or System.ComponentModel.Win32Exception or NotSupportedException)
        {
            return null;
        }
    }

    [DllImport("libSystem.B.dylib")]
    private static extern int proc_pid_rusage(int pid, int flavor, ref ulong buffer);
}

/// <summary>The host's monotonic clock in seconds: mach_absolute_time, the clock CACurrentMediaTime and Metal's GPUStartTime/GPUEndTime run on.</summary>
internal static class MachClock
{
    private static double _secondsPerTick;

    public static double Seconds() =>
        OperatingSystem.IsMacOS() ? TicksToSeconds(mach_absolute_time()) : Stopwatch.GetTimestamp() / (double)Stopwatch.Frequency;

    public static double TicksToSeconds(ulong ticks)
    {
        if (!OperatingSystem.IsMacOS())
        {
            return ticks / (double)Stopwatch.Frequency;
        }

        var scale = _secondsPerTick;
        if (scale == 0)
        {
            _ = mach_timebase_info(out var timebase);
            scale = timebase.Denominator == 0 ? 1e-9 : timebase.Numerator / (double)timebase.Denominator * 1e-9;
            _secondsPerTick = scale;
        }

        return ticks * scale;
    }

    [DllImport("libSystem.B.dylib")]
    private static extern ulong mach_absolute_time();

    [DllImport("libSystem.B.dylib")]
    private static extern int mach_timebase_info(out TimebaseInfo info);

    [StructLayout(LayoutKind.Sequential)]
    private struct TimebaseInfo
    {
        public uint Numerator;
        public uint Denominator;
    }
}
