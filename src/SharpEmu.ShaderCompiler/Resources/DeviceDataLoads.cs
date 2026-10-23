// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Generic;

namespace SharpEmu.ShaderCompiler.Resources;

/// <summary>
/// The data-only scalar loads, per shader, that the planner leaves to the device instead of
/// flattening into the host-evaluated table.
/// </summary>
/// <remarks>
/// A flattened data word is read by the host when the draw is bound. When the GPU is still
/// writing it — a title that copies its constant table on the GPU every frame — the host has
/// to wait for that submission, and the value is only ever a shader operand: the device can
/// read it itself, after the writer, through the page table. A host learns such a load at
/// bind (<see cref="Learn"/>); the next bind of that shader re-plans it with the load left
/// unflattened. Binding reads are never learned, so descriptors stay host-evaluated.
/// A host may persist what it learns (<see cref="Learned"/>) and hand it to the next start
/// (<see cref="Seed"/>) before any shader is planned, so the first plan already leaves those
/// loads to the device and no shader is compiled twice.
/// A load is a memory index, which names a record of one plan's memory table: the "hash"
/// every member takes is the plan's <see cref="ShaderResourcePlan.DeviceDataKey"/>
/// (<see cref="DeviceBindingLoads.KeyFor"/>), the shader hash told apart by the vertex fetch
/// the host embeds, since the embedding drops records and renumbers the rest.
/// </remarks>
public static class DeviceDataLoads
{
    private static readonly object Gate = new();
    private static readonly Dictionary<ulong, HashSet<int>> MemoryByHash = new();
    private static int _generation;
    private static long _slots;
    private static long _seeded;
    private static long _replans;

    /// <summary>A shader learned new device loads: its hash and every load it now holds.</summary>
    public static event Action<ulong, int[]>? Learned;

    /// <summary>Slots handed over by <see cref="Seed"/>, not counted as learned.</summary>
    public static long SeededSlots => Interlocked.Read(ref _seeded);

    // Changes whenever any shader learns a load, so a cached program can tell cheaply that
    // nothing it depends on moved.
    public static int Generation => Volatile.Read(ref _generation);

    public static long LearnedSlots => Interlocked.Read(ref _slots);

    public static long Replans => Interlocked.Read(ref _replans);

    public static void CountReplan() => Interlocked.Increment(ref _replans);

    // The memory records of one shader the planner must leave to the device; null when none.
    public static IReadOnlySet<int>? For(ulong hash)
    {
        lock (Gate)
        {
            return MemoryByHash.TryGetValue(hash, out var set) ? new HashSet<int>(set) : null;
        }
    }

    public static int CountFor(ulong hash)
    {
        lock (Gate)
        {
            return MemoryByHash.TryGetValue(hash, out var set) ? set.Count : 0;
        }
    }

    public static bool Contains(ulong hash, int memoryIndex)
    {
        lock (Gate)
        {
            return MemoryByHash.TryGetValue(hash, out var set) && set.Contains(memoryIndex);
        }
    }

    // Records one flattened slot's memory records; true when any of them is new.
    public static bool Learn(ulong hash, IReadOnlyList<int> memoryIndices)
    {
        int[] all;
        lock (Gate)
        {
            if (!MemoryByHash.TryGetValue(hash, out var set))
            {
                MemoryByHash[hash] = set = [];
            }

            var added = false;
            foreach (var index in memoryIndices)
            {
                added |= set.Add(index);
            }

            if (!added)
            {
                return false;
            }

            _slots++;
            _generation++;
            all = [.. set];
        }

        Learned?.Invoke(hash, all);
        return true;
    }

    /// <summary>
    /// Hands over loads an earlier start learned for a shader, before it is planned; they
    /// count as <see cref="SeededSlots"/> and raise no <see cref="Learned"/>.
    /// </summary>
    public static void Seed(ulong hash, IEnumerable<int> memoryIndices)
    {
        lock (Gate)
        {
            if (!MemoryByHash.TryGetValue(hash, out var set))
            {
                MemoryByHash[hash] = set = [];
            }

            var added = 0;
            foreach (var index in memoryIndices)
            {
                if (index >= 0 && set.Add(index))
                {
                    added++;
                }
            }

            if (added != 0)
            {
                _seeded += added;
                _generation++;
            }
        }
    }
}

/// <summary>The shader whose resources the current thread is materializing, for wait reports.</summary>
public static class MaterializingShader
{
    [ThreadStatic]
    private static ulong _hash;

    public static ulong Hash
    {
        get => _hash;
        set { _hash = value; _readKind = ScalarValueKind.Undefined; _pc = 0; _component = 0; _planKey = 0; }
    }

    [ThreadStatic]
    private static ScalarValueKind _readKind;

    [ThreadStatic]
    private static int _readMemoryIndex;

    // The plan node whose word the host is reading now (Undefined between reads), so a wait
    // the read causes can name it.
    public static ScalarValueKind ReadKind => _readKind;
    public static int ReadMemoryIndex => _readMemoryIndex;

    [ThreadStatic] private static uint _pc;
    [ThreadStatic] private static uint _component;
    [ThreadStatic] private static ulong _planKey;
    public static uint ReadPc => _pc;
    public static uint ReadComponent => _component;
    public static ulong PlanKey => _planKey;
    [ThreadStatic] private static string? _readReason;
    public static string ReadReason => _readReason ?? "host_required";

    internal static void BeginRead(ShaderResourcePlan plan, ScalarValue node)
    {
        _readReason = DeviceBindingLoads.TakeReadReason();
        _pc = plan.Memory[node.MemoryIndex].Pc;
        _component = plan.Memory[node.MemoryIndex].ComponentIndex;
        _planKey = plan.DeviceBindingKey;
        _readKind = node.Kind;
        _readMemoryIndex = node.MemoryIndex;
    }

    internal static void EndRead() => _readKind = ScalarValueKind.Undefined;
}
