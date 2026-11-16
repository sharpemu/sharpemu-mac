// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;
using System.Globalization;
using SharpEmu.Libs.Gpu.Scheduling;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// SHARPEMU_GATE_HOIST_PROBE=1 (timing probe, unguarded, default off): the command buffers recorded after a
/// present go to a second MTLCommandQueue, so the GPU can run them while the previous frame's lighting and
/// post-processing buffers are still queued or running on the main queue. A phase ends at the commit of the
/// first buffer with at least SHARPEMU_GATE_HOIST_PROBE_DRAWS draws (default 64) or after
/// SHARPEMU_GATE_HOIST_PROBE_MAX buffers (default 8).
/// <list type="bullet">
/// <item>A hoisted buffer signals its own shared event. Its completion work (label writes, end-of-pipe
/// interrupts) runs on a second completion thread when that buffer completes, so its labels can land
/// before earlier buffers on the main queue have finished.</item>
/// <item>Completed serials stay contiguous: after committing each hoisted buffer, an empty buffer on the
/// main queue waits for the hoisted buffer's event value and then signals the main event with the same
/// serial. The main event therefore reaches a hoisted serial only once all earlier main-queue work and that
/// buffer are done, and every main buffer committed later starts after it.</item>
/// <item>Nothing orders a hoisted buffer against the main-queue work still running before it: no hazard
/// tracking across queues, and no guard on what the tail reads or writes. The image is not guaranteed.</item>
/// </list>
/// </summary>
internal sealed partial class MetalRenderHost
{
    internal static readonly bool GateHoistProbe = string.Equals(
        Environment.GetEnvironmentVariable("SHARPEMU_GATE_HOIST_PROBE"), "1", StringComparison.Ordinal);

    private static readonly int GateHoistDraws = ReadHoistSetting("SHARPEMU_GATE_HOIST_PROBE_DRAWS", 64);
    private static readonly int GateHoistMaxBuffers = ReadHoistSetting("SHARPEMU_GATE_HOIST_PROBE_MAX", 8);
    private static readonly nint SelNewCommandQueueForHoist = MetalNative.Selector("newCommandQueue");
    private const int MaxHoistReports = 400;

    private nint _hoistQueue;
    private nint _hoistEvent;
    private MetalCompletionQueue? _hoistCompletions;
    private readonly ConcurrentDictionary<ulong, byte> _hoistedSerials = new();
    private bool _hoistPhase;
    private bool _openOnHoist;
    private int _drawsInBuffer;
    private int _hoistPhaseBuffers;
    private ulong _hoistPhasePresent;
    private System.Text.StringBuilder? _hoistPhaseLine;
    private int _hoistReports;
    private long _hoistedBuffers;
    private long _hoistedPresents;

    private nint RecordQueue => _openOnHoist ? _hoistQueue : _queue;

    private static int ReadHoistSetting(string name, int fallback) =>
        int.TryParse(Environment.GetEnvironmentVariable(name), NumberStyles.Integer, CultureInfo.InvariantCulture, out var value) && value > 0
            ? value
            : fallback;

    private void InitGateHoist(nint device)
    {
        _hoistQueue = MetalNative.Send(device, SelNewCommandQueueForHoist);
        _hoistEvent = MetalNative.Send(device, SelNewSharedEvent);
        if (_hoistQueue == 0 || _hoistEvent == 0)
        {
            throw GpuFatal.Fatal("SHARPEMU_GATE_HOIST_PROBE: a second Metal command queue or shared event could not be created.");
        }

        _hoistCompletions = new MetalCompletionQueue(_hoistEvent, bindHandler: false);
        _completions.Route = serial => _hoistedSerials.ContainsKey(serial) ? _hoistCompletions : null;
        Console.Error.WriteLine(
            "[METAL-V2][GATE-HOIST][WARN] timing probe: buffers after a present run on a second queue, unordered " +
            $"against the main queue's earlier work; measurement only (draws={GateHoistDraws} max={GateHoistMaxBuffers})");
    }

    // Before the command buffer for `serial` is created.
    private void BeginGateHoistBuffer(ulong serial)
    {
        _openOnHoist = _hoistPhase;
        _drawsInBuffer = 0;
        if (_openOnHoist)
        {
            foreach (var old in _hoistedSerials.Keys)
            {
                if (old + 1024 < serial)
                {
                    _hoistedSerials.TryRemove(old, out _);
                }
            }

            _hoistedSerials[serial] = 0;
        }
    }

    // After the command buffer for `serial` was committed.
    private void AfterGateHoistCommit(ulong serial, bool present)
    {
        if (_openOnHoist)
        {
            _hoistedBuffers++;
            _hoistPhaseBuffers++;
            _hoistCompletions!.Publish(serial);
            // The main queue's stand-in: the main event reaches this serial once everything before it on the
            // main queue and the hoisted buffer itself are done.
            var shadow = MetalNative.Send(_queue, SelCommandBuffer);
            if (shadow == 0)
            {
                throw GpuFatal.Fatal("SHARPEMU_GATE_HOIST_PROBE: a Metal command buffer could not be created.");
            }

            MetalV2Native.SendEventValue(shadow, SelEncodeWaitForEvent, _hoistEvent, serial);
            MetalV2Native.SendEventValue(shadow, SelEncodeSignalEvent, _sharedEvent, serial);
            if (_completions.WantsCompletionHandler)
            {
                MetalNative.SendVoid(shadow, SelAddCompletedHandler, MetalCompletionSignal.Block);
            }

            MetalNative.SendVoid(shadow, SelCommit);
            _hoistPhaseLine?.Append(CultureInfo.InvariantCulture, $" {serial}:{_drawsInBuffer}");
            string? end = present ? "present" : _drawsInBuffer >= GateHoistDraws ? "draws" : _hoistPhaseBuffers >= GateHoistMaxBuffers ? "max" : null;
            if (present)
            {
                _hoistedPresents++;
            }

            if (end is not null)
            {
                _hoistPhase = false;
                ReportHoistPhase(end);
            }
        }

        _openOnHoist = false;
        _drawsInBuffer = 0;
        if (present)
        {
            if (_hoistPhase)
            {
                ReportHoistPhase("next_present");
            }

            _hoistPhase = true;
            _hoistPhaseBuffers = 0;
            _hoistPhasePresent = serial;
            _hoistPhaseLine = _hoistReports < MaxHoistReports ? new System.Text.StringBuilder() : null;
        }
    }

    private void ReportHoistPhase(string end)
    {
        if (_hoistPhaseLine is { } line && _hoistReports < MaxHoistReports)
        {
            _hoistReports++;
            Console.Error.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"[METAL-V2][GATE-HOIST] present={_hoistPhasePresent} end={end} hoisted_total={_hoistedBuffers} " +
                $"hoisted_presents={_hoistedPresents} buffers(serial:draws)={line}"));
        }

        _hoistPhaseLine = null;
    }
}
