// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.VideoOut;

// The fraction of wall time the Metal GPU was executing this process's command buffers over the
// last sample window, as the union of their GPUStartTime..GPUEndTime intervals (the backend feeds
// them through PerfOverlay.RecordGpuBusy). This replaces the IOAccelerator "Device Utilization %"
// counter, which is the whole machine's GPU (WindowServer and every other app included) and
// is smoothed by the driver, so it did not track this emulator's own busy time.
internal sealed class MacGpuUsage : IGpuUsage
{
    private readonly GpuBusyTracker _tracker;
    private double _windowStart;
    private double _percent = double.NaN;

    public MacGpuUsage(GpuBusyTracker tracker)
    {
        _tracker = tracker;
        _windowStart = MachClock.Seconds();
    }

    // NaN (shown as N/A) until the backend has reported any GPU time, e.g. a non-Metal presenter.
    public double Percent => Volatile.Read(ref _percent);

    // Closes the window that started at the previous request. Cheap and synchronous: the
    // refresh runs it once per window.
    public void RequestSample()
    {
        var now = MachClock.Seconds();
        var fraction = _tracker.TakeBusyFraction(_windowStart, now);
        _windowStart = now;
        Volatile.Write(ref _percent, _tracker.HasSamples ? fraction * 100.0 : double.NaN);
    }

    public void Dispose()
    {
    }
}

internal interface IGpuUsage : IDisposable
{
    double Percent { get; }

    void RequestSample();
}
