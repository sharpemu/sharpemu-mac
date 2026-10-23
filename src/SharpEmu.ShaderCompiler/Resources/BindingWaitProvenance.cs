// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
using System.Text;

namespace SharpEmu.ShaderCompiler.Resources;

public readonly record struct BindingWordWriter(ulong Serial, string Origin);

// Interval aggregates shared by materialization decisions and actual host waits.
// Word probes have an overflow bucket; actual wait identities are always retained.
public static class BindingWaitProvenance
{
    private readonly record struct Key(ulong Hash, ulong Plan, uint Pc, uint Component, ScalarValueKind Kind,
        ulong Writer, string Origin, string Reason, ulong Frame = 0);
    private sealed class Count
    {
        public long Words, Waits, Ticks;
        public ulong Address, Frame;
        public uint Value;
    }
    private static readonly object Gate = new();
    private static readonly Dictionary<Key, Count> Counts = [];
    private const int Capacity = 256;
    [ThreadStatic] private static Key? _consumedWait;

    private static void Add(Key key, ulong address, bool wait, long ticks = 0, ulong frame = 0, uint value = 0)
    {
        lock (Gate)
        {
            if (!Counts.TryGetValue(key, out var count))
            {
                if (!wait && Counts.Count >= Capacity) key = new(0, 0, 0, 0, ScalarValueKind.Undefined, 0, "overflow", "overflow");
                if (!Counts.TryGetValue(key, out count)) Counts[key] = count = new();
            }
            count.Words++;
            if (wait) count.Waits++;
            count.Ticks += ticks;
            count.Address = address;
            count.Frame = frame;
            count.Value = value;
        }
    }

    internal static void Word(ShaderResourcePlan plan, ScalarValue node, ulong address, BindingWordWriter writer, string reason) =>
        Add(new(plan.Hash, plan.DeviceBindingKey, plan.Memory[node.MemoryIndex].Pc, plan.Memory[node.MemoryIndex].ComponentIndex,
            node.Kind, writer.Serial, writer.Origin ?? "unknown", reason), address, false);

    public static void Decision(DeviceBindingLoads.Request? request, string reason)
    {
        if (request is not { } r) return;
        if (reason is not ("accepted" or "ready")) DeviceBindingLoads.SetRefusalReason(r.Hash, r.MemoryIndex, reason);
        Add(new(r.ShaderHash, r.Hash, r.Pc, r.Component, r.Kind, r.Writer.Serial, r.Writer.Origin ?? "unknown", reason), r.Address, false);
    }

    public static void Wait(ulong serial, string origin, ulong address, long ticks, ulong frame)
    {
        var key = new Key(MaterializingShader.Hash, MaterializingShader.PlanKey, MaterializingShader.ReadPc,
            MaterializingShader.ReadComponent, MaterializingShader.ReadKind, serial, origin, MaterializingShader.ReadReason, frame);
        Add(key, address, true, ticks, frame);
        _consumedWait = key;
    }

    public static void Consumed(uint value)
    {
        if (_consumedWait is not { } key) return;
        _consumedWait = null;
        lock (Gate)
            if (Counts.TryGetValue(key, out var count)) count.Value = value;
    }

    public static string TakeReport()
    {
        lock (Gate)
        {
            var report = new StringBuilder(" binding_provenance=[");
            foreach (var (key, count) in Counts.OrderByDescending(pair => pair.Value.Ticks).ThenByDescending(pair => pair.Value.Words))
                report.Append(CultureInfo.InvariantCulture, $"{{shader={key.Hash:X16} plan={key.Plan:X16} pc={key.Pc:X} component={key.Component} kind={key.Kind} writer={key.Writer}/{key.Origin} reason={key.Reason} events={count.Words} waits={count.Waits} ms={count.Ticks * 1000.0 / System.Diagnostics.Stopwatch.Frequency:F2} last_address={count.Address:X} last_value={count.Value:X8} frame={count.Frame}}}");
            Counts.Clear();
            return report.Append(']').ToString();
        }
    }
}
