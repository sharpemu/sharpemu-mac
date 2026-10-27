// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Metal.V2;

// Diagnostic only: guest query dumps remain conservative. Each enabled draw group
// gets a unique visibility slot in its encoder. Never recycle attached storage until
// its command buffer has completed, including empty encoders and failed submissions.
internal sealed partial class MetalRenderHost
{
    internal bool OcclusionAuditEnabled { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_AUDIT_OCCLUSION") == "1";

    private const int OcclusionAuditSlots = 2048;
    private const int OcclusionAuditBuffers = 64;
    private static readonly nint SelAuditVisibilityBuffer = MetalNative.Selector("setVisibilityResultBuffer:");
    private static readonly nint SelAuditVisibilityMode = MetalNative.Selector("setVisibilityResultMode:offset:");
    private static readonly nint SelAuditNewBuffer = MetalNative.Selector("newBufferWithLength:options:");
    private static readonly nint SelAuditContents = MetalNative.Selector("contents");

    private sealed class OcclusionAuditBuffer(nint buffer)
    {
        internal readonly nint Buffer = buffer;
        internal readonly bool[] Drawn = new bool[OcclusionAuditSlots];
        internal int Used;
    }

    private readonly List<OcclusionAuditBuffer> _oaAllocated = [];
    private readonly Stack<OcclusionAuditBuffer> _oaFree = [];
    private readonly Dictionary<ulong, List<OcclusionAuditBuffer>> _oaInFlight = [];
    private List<OcclusionAuditBuffer>? _oaCurrent;
    private OcclusionAuditBuffer? _oaOpen;
    private int _oaSlot = -1;
    private long _oaGroups, _oaZeroGroups, _oaEnabledDraws, _oaSkipped, _oaFailedGroups;
    private ulong _oaSamples;
    private long _oaLastReport;

    internal (long Groups, long ZeroGroups, ulong Samples, long EnabledDraws, long Skipped) OcclusionAuditTotals =>
        (_oaGroups, _oaZeroGroups, _oaSamples, _oaEnabledDraws, _oaSkipped);

    private void OcclusionAuditAttach(nint descriptor)
    {
        _oaOpen = null;
        _oaSlot = -1;
        if (!OcclusionAuditEnabled) return;
        if (!_oaFree.TryPop(out var entry))
        {
            if (_oaAllocated.Count == OcclusionAuditBuffers) return;
            var buffer = MetalNative.SendNewBuffer(_device, SelAuditNewBuffer,
                OcclusionAuditSlots * 8, Mtl.ResourceStorageModeShared);
            if (buffer == 0) return;
            entry = new OcclusionAuditBuffer(buffer);
            _oaAllocated.Add(entry);
        }

        entry.Used = 0;
        Array.Clear(entry.Drawn);
        _oaOpen = entry;
        (_oaCurrent ??= []).Add(entry);
        MetalNative.SendVoid(descriptor, SelAuditVisibilityBuffer, entry.Buffer);
    }

    private void OcclusionAuditPrepareDraw()
    {
        if (!OcclusionAuditEnabled) return;
        _oaSlot = -1;
        var enabled = (_dynamicState.DepthCountControl & (1u | (1u << 8))) == (1u << 8);
        if (enabled)
        {
            _oaEnabledDraws++;
            if (_oaOpen is not null && _oaOpen.Used < OcclusionAuditSlots)
                _oaSlot = _oaOpen.Used++;
            else
                _oaSkipped++;
        }

        MetalV2Native.SendVisibilityResultMode(_renderEncoder, SelAuditVisibilityMode,
            _oaSlot >= 0 ? 2u : 0u, _oaSlot >= 0 ? (nuint)(_oaSlot * 8) : 0);
    }

    private void OcclusionAuditCountDraw()
    {
        if (_oaSlot >= 0 && _oaOpen is not null) _oaOpen.Drawn[_oaSlot] = true;
    }

    private void OcclusionAuditClose()
    {
        _oaOpen = null;
        _oaSlot = -1;
    }

    private void OcclusionAuditCommit(ulong serial)
    {
        if (_oaCurrent is null) return;
        _oaInFlight.Add(serial, _oaCurrent);
        _oaCurrent = null;
    }

    private unsafe void OcclusionAuditResolve(ulong serial, bool completed)
    {
        if (!_oaInFlight.Remove(serial, out var buffers)) return;
        foreach (var entry in buffers)
        {
            if (!completed)
            {
                // Quarantine until host disposal: this includes dropped telemetry records,
                // whose GPU writes might still be in flight. The global budget stays bounded.
                _oaFailedGroups += entry.Used;
                continue;
            }

            var values = (ulong*)MetalNative.Send(entry.Buffer, SelAuditContents);
            for (var i = 0; i < entry.Used; i++)
            {
                if (!entry.Drawn[i]) continue;
                _oaGroups++;
                if (values[i] == 0) _oaZeroGroups++;
                _oaSamples += values[i];
            }
            _oaFree.Push(entry);
        }

        var now = System.Diagnostics.Stopwatch.GetTimestamp();
        if (now - _oaLastReport < 2 * System.Diagnostics.Stopwatch.Frequency) return;
        _oaLastReport = now;
        Console.Error.WriteLine($"[METAL-V2][OCCLUSION-AUDIT] serial={serial} groups={_oaGroups} " +
            $"zero_groups={_oaZeroGroups} samples={_oaSamples} enabled_draws={_oaEnabledDraws} " +
            $"skipped={_oaSkipped} failed_groups={_oaFailedGroups} buffers={_oaAllocated.Count}");
    }

    private void DisposeOcclusionAudit()
    {
        foreach (var entry in _oaAllocated) MetalNative.SendVoid(entry.Buffer, SelRelease);
        _oaAllocated.Clear();
        _oaFree.Clear();
        _oaInFlight.Clear();
        _oaCurrent = null;
        OcclusionAuditClose();
    }
}
