// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;
using System.Globalization;
using System.IO.Hashing;
using System.Text;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// Keeps the binding loads a title's shaders left to the device, and the ones refused
/// (<see cref="DeviceBindingLoads"/>), across starts, beside <see cref="MetalDeviceDataLoadsStore"/>:
/// a start seeds them before the compute prewarm and the game plan any shader. Without it
/// every start planned each shader without them, then learned them one re-plan at a time in
/// the order its GPU timing gave, so each start compiled programs (and asked the binary
/// archive for libraries) no earlier start had made: Demon's Souls' library misses rose from
/// about 300 to 650-865 a run, and in-world pipeline creation from 3 s to 14-39 s. The file
/// is per translator build (the planner decides what a memory index names). One text line
/// per change of a key: the key, every load it leaves to the device and every load refused
/// for it, the last line of a key winning. SHARPEMU_METAL_V2_DEVICE_BINDING_STORE=0 turns it off.
/// </summary>
internal static class MetalDeviceBindingLoadsStore
{
    internal const string FilePrefix = "device-binding-loads-v2";

    internal static readonly bool Disabled =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_DEVICE_BINDING_STORE") == "0";

    private static readonly object _sync = new();
    private static BlockingCollection<string>? _writes;
    private static Thread? _writer;

    internal static string FileName(string stamp) =>
        string.Create(
            CultureInfo.InvariantCulture,
            $"{FilePrefix}-{(uint)XxHash3.HashToUInt64(Encoding.UTF8.GetBytes(stamp + SharpEmu.ShaderCompiler.AssemblyCodeIdentity.Of(typeof(DeviceBindingLoads).Assembly))):x8}.txt");

    /// <summary>Seeds what earlier starts of this build learned and refused, and records what this one does.</summary>
    public static void Open(string directory, string stamp)
    {
        // With the loads off (SHARPEMU_METAL_V2_WAIT_RESOURCE_WORDS=1) every word is waited
        // for, and no plan may leave one to the device.
        if (Disabled || !DeviceBindingLoads.Enabled)
        {
            return;
        }

        try
        {
            Directory.CreateDirectory(directory);
            var path = Path.Combine(directory, FileName(stamp));
            var keys = Load(path, out var lines);
            var learned = 0;
            var refused = 0;
            foreach (var (key, state) in keys)
            {
                DeviceBindingLoads.Seed(key, state.Learned, state.Refused);
                learned += state.Learned.Length;
                refused += state.Refused.Length;
            }

            if (lines > 2 * keys.Count + 16)
            {
                Rewrite(path, keys);
            }

            MetalDeviceDataLoadsStore.PruneOtherFiles(directory, path, FilePrefix);
            var writes = new BlockingCollection<string>();
            var writer = new Thread(() => MetalDeviceDataLoadsStore.WriteLoop(path, writes))
            {
                IsBackground = true,
                Priority = ThreadPriority.BelowNormal,
                Name = "SharpEmu Metal v2 device binding loads",
            };
            lock (_sync)
            {
                if (_writes is not null)
                {
                    return;
                }

                _writes = writes;
                _writer = writer;
            }

            DeviceBindingLoads.Changed += OnChanged;
            writer.Start();
            Console.Error.WriteLine(
                $"[LOADER][INFO] Device binding loads seeded: keys={keys.Count} loads={learned} refused={refused}");
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            Console.Error.WriteLine($"[LOADER][WARN] Device binding loads could not be read: {exception.Message}");
        }
    }

    public static void Stop()
    {
        BlockingCollection<string>? writes;
        Thread? writer;
        lock (_sync)
        {
            writes = _writes;
            writer = _writer;
            _writes = null;
            _writer = null;
        }

        if (writes is null)
        {
            return;
        }

        DeviceBindingLoads.Changed -= OnChanged;
        writes.CompleteAdding();
        writer?.Join(TimeSpan.FromSeconds(10));
    }

    private static void OnChanged(ulong key, int[] learned, int[] refused)
    {
        try
        {
            Volatile.Read(ref _writes)?.Add(Line(key, learned, refused));
        }
        catch (InvalidOperationException)
        {
            // The store closed with the session.
        }
    }

    internal static string Line(ulong key, IEnumerable<int> learned, IEnumerable<int> refused) =>
        string.Create(CultureInfo.InvariantCulture, $"{key:X16} {List(learned)} {List(refused)}");

    private static string List(IEnumerable<int> indices) =>
        indices.Any() ? string.Join(',', indices.Order()) : "-";

    /// <summary>The loads by key, the last whole line of a key winning; torn or foreign lines are skipped.</summary>
    internal static Dictionary<ulong, (int[] Learned, int[] Refused)> Load(string path, out int lines)
    {
        lines = 0;
        var keys = new Dictionary<ulong, (int[] Learned, int[] Refused)>();
        if (!File.Exists(path))
        {
            return keys;
        }

        foreach (var line in File.ReadLines(path))
        {
            lines++;
            var parts = line.Split(' ');
            if (parts.Length != 3 ||
                !ulong.TryParse(parts[0], NumberStyles.HexNumber, CultureInfo.InvariantCulture, out var key) ||
                !TryParseList(parts[1], out var learned) || !TryParseList(parts[2], out var refused))
            {
                continue;
            }

            keys[key] = (learned, refused);
        }

        return keys;
    }

    private static bool TryParseList(string text, out int[] indices)
    {
        indices = [];
        if (text == "-")
        {
            return true;
        }

        var parsed = new List<int>();
        foreach (var part in text.Split(','))
        {
            if (!int.TryParse(part, NumberStyles.None, CultureInfo.InvariantCulture, out var index))
            {
                return false;
            }

            parsed.Add(index);
        }

        indices = [.. parsed];
        return parsed.Count != 0;
    }

    private static void Rewrite(string path, Dictionary<ulong, (int[] Learned, int[] Refused)> keys)
    {
        var temporary = path + ".tmp";
        File.WriteAllLines(temporary, keys.Select(pair => Line(pair.Key, pair.Value.Learned, pair.Value.Refused)));
        File.Move(temporary, path, overwrite: true);
    }
}
