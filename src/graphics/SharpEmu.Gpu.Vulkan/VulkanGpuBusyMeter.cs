// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using SharpEmu.Libs.VideoOut;
using SharpEmu.Libs.VideoOut.Overlay;
using Silk.NET.Vulkan;

namespace SharpEmu.Gpu.Vulkan;

// Measures how long the GPU spends on each guest command buffer for the performance overlay.
// Every command buffer that begins while the overlay is on gets a pair of timestamp queries, one
// at the top of the buffer and one at the bottom. Results are polled without waiting when the
// host checks for completion (ReadTimeline / TrySubmit), converted to the clock the overlay
// uses (MachClock.Seconds) and reported through PerfOverlay.RecordGpuBusy, which feeds both the
// GPU percentage and the frame timeline's GPU lane.
//
// Mapping GPU time to host time:
//  - With VK_KHR/EXT_calibrated_timestamps the device clock is sampled together with a host clock
//    (QueryPerformanceCounter on Windows; CLOCK_MONOTONIC_RAW, else CLOCK_MONOTONIC elsewhere,
//    which is whatever MoltenVK offers on macOS). The call is bracketed with MachClock reads
//    and the midpoint pairs that device tick with a MachClock time, so the host domain need not
//    be MachClock's own. The pair is refreshed about once a second to follow clock drift.
//  - Without it, the anchor is the host time at which a result was first seen available. That
//    is never earlier than the truth, so the anchor only moves earlier (the least-lagging
//    sample wins) and creeps later slowly to follow drift.
internal sealed unsafe class VulkanGpuBusyMeter : IDisposable
{
    private const int BufferCapacity = 256;
    private const double RecalibrationSeconds = 1.0;
    private const long PollIntervalMilliseconds = 2;
    private const double FallbackCreepSecondsPerSecond = 1e-4;

    private enum SlotState : byte { Idle, Recording, Ended, Pending }

    private sealed class Slot(int index)
    {
        public readonly int Index = index;
        public SlotState State;
        public ulong LastStart;
        public ulong LastEnd;
    }

    private readonly Vk _vk;
    private readonly Device _device;
    private readonly QueryPool _pool;
    private readonly ulong _validMask;
    private readonly int _validBits;
    private readonly double _periodSeconds;
    private readonly object _gate = new();
    private readonly Dictionary<nint, Slot> _slots = new();
    private readonly delegate* unmanaged<Device, uint, CalibratedTimestampInfoKHR*, ulong*, ulong*, Result> _getCalibratedTimestamps;
    private readonly TimeDomainKHR _hostDomain;
    private int _pending;
    private long _lastPoll;

    private bool _anchored;
    private ulong _anchorTicks;
    private double _anchorHost;
    private double _anchoredAt;

    internal VulkanGpuBusyMeter(Vk vk, PhysicalDevice physical, Device device, Instance instance, uint queueFamily,
        string? calibrationSuffix)
    {
        _vk = vk;
        _device = device;
        vk.GetPhysicalDeviceProperties(physical, out var properties);
        uint familyCount = 0;
        vk.GetPhysicalDeviceQueueFamilyProperties(physical, &familyCount, null);
        var families = new QueueFamilyProperties[familyCount];
        fixed (QueueFamilyProperties* pointer = families)
        {
            vk.GetPhysicalDeviceQueueFamilyProperties(physical, &familyCount, pointer);
        }

        _validBits = queueFamily < families.Length ? (int)families[queueFamily].TimestampValidBits : 0;
        _periodSeconds = properties.Limits.TimestampPeriod * 1e-9;
        if (!Supported)
        {
            return;
        }

        _validMask = _validBits >= 64 ? ulong.MaxValue : (1UL << _validBits) - 1;
        var createInfo = new QueryPoolCreateInfo
        {
            SType = StructureType.QueryPoolCreateInfo,
            QueryType = QueryType.Timestamp,
            QueryCount = BufferCapacity * 2,
        };
        if (vk.CreateQueryPool(device, &createInfo, null, out _pool) != Result.Success)
        {
            _validBits = 0;
            return;
        }

        if (calibrationSuffix is not null)
        {
            TryLoadCalibration(instance, physical, calibrationSuffix, out _getCalibratedTimestamps, out _hostDomain);
        }

        Console.Error.WriteLine(
            $"[LOADER][INFO] Vulkan GPU timing: timestamp_bits={_validBits} period_ns={properties.Limits.TimestampPeriod} " +
            $"host_clock={(_getCalibratedTimestamps is null ? "fence-anchored" : _hostDomain.ToString())}");
    }

    internal bool Supported => _validBits > 0 && _periodSeconds > 0;

    // The meter costs nothing unless the overlay is shown or something consumes its telemetry.
    internal static bool Wanted => PerfOverlay.Enabled || OverlayTelemetry.Active;

    private void TryLoadCalibration(
        Instance instance,
        PhysicalDevice physical,
        string suffix,
        out delegate* unmanaged<Device, uint, CalibratedTimestampInfoKHR*, ulong*, ulong*, Result> getTimestamps,
        out TimeDomainKHR hostDomain)
    {
        getTimestamps = null;
        hostDomain = default;
        var domainsFunction = (delegate* unmanaged<PhysicalDevice, uint*, TimeDomainKHR*, Result>)
            _vk.GetInstanceProcAddr(instance, "vkGetPhysicalDeviceCalibrateableTimeDomains" + suffix).Handle;
        var timestampsFunction = (delegate* unmanaged<Device, uint, CalibratedTimestampInfoKHR*, ulong*, ulong*, Result>)
            _vk.GetDeviceProcAddr(_device, "vkGetCalibratedTimestamps" + suffix).Handle;
        if (domainsFunction is null || timestampsFunction is null)
        {
            return;
        }

        uint count = 0;
        if (domainsFunction(physical, &count, null) != Result.Success || count == 0)
        {
            return;
        }

        var domains = new TimeDomainKHR[count];
        fixed (TimeDomainKHR* pointer = domains)
        {
            if (domainsFunction(physical, &count, pointer) != Result.Success)
            {
                return;
            }
        }

        var preference = OperatingSystem.IsWindows()
            ? new[] { TimeDomainKHR.QueryPerformanceCounterKhr }
            : new[] { TimeDomainKHR.ClockMonotonicRawKhr, TimeDomainKHR.ClockMonotonicKhr };
        var deviceDomainSeen = false;
        foreach (var domain in domains.Take((int)count))
        {
            deviceDomainSeen |= domain == TimeDomainKHR.DeviceKhr;
        }

        if (!deviceDomainSeen)
        {
            return;
        }

        foreach (var wanted in preference)
        {
            if (domains.Take((int)count).Contains(wanted))
            {
                hostDomain = wanted;
                getTimestamps = timestampsFunction;
                return;
            }
        }

        // Anything the driver offers besides the device domain works: MachClock is paired by
        // bracketing the call, not by knowing the domain.
        foreach (var domain in domains.Take((int)count))
        {
            if (domain != TimeDomainKHR.DeviceKhr)
            {
                hostDomain = domain;
                getTimestamps = timestampsFunction;
                return;
            }
        }
    }

    // Recording thread, right after vkBeginCommandBuffer and outside any render pass.
    internal void BeginBuffer(CommandBuffer command)
    {
        if (!Supported || !Wanted)
        {
            return;
        }

        Slot slot;
        lock (_gate)
        {
            if (!_slots.TryGetValue(command.Handle, out slot!))
            {
                if (_slots.Count >= BufferCapacity)
                {
                    return;
                }

                slot = new Slot(_slots.Count);
                _slots.Add(command.Handle, slot);
            }
            else if (slot.State == SlotState.Pending)
            {
                // The ring reuses a buffer only after its submission completed; if the result
                // was never collected, the interval is dropped.
                _pending--;
            }

            slot.State = SlotState.Recording;
        }

        _vk.CmdResetQueryPool(command, _pool, (uint)(slot.Index * 2), 2);
        _vk.CmdWriteTimestamp2(command, PipelineStageFlags2.TopOfPipeBit, _pool, (uint)(slot.Index * 2));
    }

    // Recording thread, before vkEndCommandBuffer and outside any render pass.
    internal void EndBuffer(CommandBuffer command)
    {
        Slot? slot;
        lock (_gate)
        {
            if (!_slots.TryGetValue(command.Handle, out slot) || slot.State != SlotState.Recording)
            {
                return;
            }

            slot.State = SlotState.Ended;
        }

        _vk.CmdWriteTimestamp2(command, PipelineStageFlags2.BottomOfPipeBit, _pool, (uint)(slot.Index * 2 + 1));
    }

    // After vkQueueSubmit2 accepted the buffer.
    internal void MarkSubmitted(nint command)
    {
        lock (_gate)
        {
            if (_slots.TryGetValue(command, out var slot) && slot.State == SlotState.Ended)
            {
                slot.State = SlotState.Pending;
                _pending++;
            }
        }

        Poll();
    }

    // Non-blocking: collects whatever the GPU has finished since the last call.
    internal void Poll()
    {
        if (_pending == 0 || !Supported)
        {
            return;
        }

        var now = Stopwatch.GetTimestamp();
        if ((now - Interlocked.Read(ref _lastPoll)) * 1000 < PollIntervalMilliseconds * Stopwatch.Frequency)
        {
            return;
        }

        if (!Monitor.TryEnter(_gate))
        {
            return;
        }

        try
        {
            Interlocked.Exchange(ref _lastPoll, now);
            CollectLocked();
        }
        finally
        {
            Monitor.Exit(_gate);
        }
    }

    private void CollectLocked()
    {
        var observed = MachClock.Seconds();
        var values = stackalloc ulong[4];
        foreach (var slot in _slots.Values)
        {
            // Only submitted buffers: reading a query that has not been through a submitted reset
            // is invalid, and a buffer still being recorded has not been.
            if (slot.State != SlotState.Pending)
            {
                continue;
            }

            // NotReady is expected: finished queries still carry their availability word.
            var result = _vk.GetQueryPoolResults(_device, _pool, (uint)(slot.Index * 2), 2, 4 * sizeof(ulong),
                values, 2 * sizeof(ulong), QueryResultFlags.Result64Bit | QueryResultFlags.ResultWithAvailabilityBit);
            if ((result != Result.Success && result != Result.NotReady) || values[1] == 0 || values[3] == 0)
            {
                continue;
            }

            var start = values[0] & _validMask;
            var end = values[2] & _validMask;
            if (start == slot.LastStart && end == slot.LastEnd)
            {
                // The previous use's result, still marked available until the GPU runs this
                // buffer's query reset.
                continue;
            }

            slot.LastStart = start;
            slot.LastEnd = end;
            slot.State = SlotState.Idle;
            _pending--;
            if (!EnsureAnchor(end, observed))
            {
                continue;
            }

            var startSeconds = ToHostSeconds(start);
            var endSeconds = ToHostSeconds(end);
            if (endSeconds > startSeconds)
            {
                PerfOverlay.RecordGpuBusy(startSeconds, endSeconds);
            }
        }
    }

    private bool EnsureAnchor(ulong endTicks, double observed)
    {
        if (_getCalibratedTimestamps is not null)
        {
            if (!_anchored || observed - _anchoredAt >= RecalibrationSeconds)
            {
                Calibrate(observed);
            }

            if (_anchored)
            {
                return true;
            }
        }

        // Fence-anchored: the buffer ended no later than now.
        if (!_anchored)
        {
            _anchorTicks = endTicks;
            _anchorHost = observed;
            _anchoredAt = observed;
            _anchored = true;
            return true;
        }

        _anchorHost += (observed - _anchoredAt) * FallbackCreepSecondsPerSecond;
        _anchoredAt = observed;
        if (observed < _anchorHost + SignedDiff(endTicks, _anchorTicks) * _periodSeconds)
        {
            _anchorTicks = endTicks;
            _anchorHost = observed;
        }

        return true;
    }

    private void Calibrate(double observed)
    {
        var infos = stackalloc CalibratedTimestampInfoKHR[2];
        infos[0] = new CalibratedTimestampInfoKHR { SType = StructureType.CalibratedTimestampInfoKhr, TimeDomain = TimeDomainKHR.DeviceKhr };
        infos[1] = new CalibratedTimestampInfoKHR { SType = StructureType.CalibratedTimestampInfoKhr, TimeDomain = _hostDomain };
        var timestamps = stackalloc ulong[2];
        var bestWidth = double.MaxValue;
        // Three attempts; the tightest bracket pins the pairing best.
        for (var attempt = 0; attempt < 3; attempt++)
        {
            ulong deviation;
            var before = MachClock.Seconds();
            var result = _getCalibratedTimestamps(_device, 2, infos, timestamps, &deviation);
            var after = MachClock.Seconds();
            if (result != Result.Success)
            {
                return;
            }

            if (after - before < bestWidth)
            {
                bestWidth = after - before;
                _anchorTicks = timestamps[0] & _validMask;
                _anchorHost = (before + after) / 2;
                _anchored = true;
            }
        }

        _anchoredAt = observed;
    }

    private long SignedDiff(ulong ticks, ulong reference)
    {
        var difference = (ticks - reference) & _validMask;
        if (_validBits >= 64)
        {
            return (long)difference;
        }

        return difference >= 1UL << (_validBits - 1) ? (long)difference - (long)(_validMask + 1) : (long)difference;
    }

    private double ToHostSeconds(ulong ticks) => _anchorHost + SignedDiff(ticks, _anchorTicks) * _periodSeconds;

    public void Dispose()
    {
        if (!Supported)
        {
            return;
        }

        lock (_gate)
        {
            CollectLocked();
            _vk.DestroyQueryPool(_device, _pool, null);
            _slots.Clear();
        }
    }
}
