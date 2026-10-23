// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;
using System.Globalization;
using System.IO.Hashing;
using System.Text;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// Keeps the device data loads a title's shaders learned (<see cref="DeviceDataLoads"/>)
/// across starts, next to the prewarm lists: a start seeds them before the compute prewarm
/// and the game plan any shader, so the prewarm compiles the plan the game ends up using
/// and no shader is planned and compiled a second time when the load is learned again.
/// Without it Demon's Souls re-learned about 50 loads every start and recompiled each
/// shader on the GPU worker, about a second apiece, 37 of them in one burst that drew
/// nothing for 41 s. The file is per translator build (the planner decides what a memory
/// index names); the files of other builds are kept up to <see cref="MaxOtherFiles"/>.
/// One text line per learned shader: its key (the plan's DeviceDataKey: the hash, or the hash
/// mixed with an embedded vertex fetch) and every load it holds, the last line of a shader
/// winning. SHARPEMU_METAL_V2_DEVICE_DATA_STORE=0 turns it off.
/// </summary>
internal static class MetalDeviceDataLoadsStore
{
    internal const string FilePrefix = "device-data-loads-v1";
    internal const int MaxOtherFiles = 3;

    internal static readonly bool Disabled =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_DEVICE_DATA_STORE") == "0";

    private static readonly object _sync = new();
    private static BlockingCollection<string>? _writes;
    private static Thread? _writer;

    internal static string FileName(string stamp) =>
        string.Create(
            CultureInfo.InvariantCulture,
            $"{FilePrefix}-{(uint)XxHash3.HashToUInt64(Encoding.UTF8.GetBytes(stamp + ShaderResourcePlan.DeviceLoadCompilerIdentity)):x8}.txt");

    /// <summary>Seeds what earlier starts of this build learned and records what this one learns.</summary>
    public static void Open(string directory, string stamp)
    {
        if (Disabled)
        {
            return;
        }

        try
        {
            Directory.CreateDirectory(directory);
            var path = Path.Combine(directory, FileName(stamp));
            var learned = Load(path, out var lines);
            var slots = 0;
            foreach (var (hash, indices) in learned)
            {
                DeviceDataLoads.Seed(hash, indices);
                slots += indices.Count;
            }

            // Every learn appends the whole set of its shader; the file is rewritten once
            // those lines outnumber the shaders.
            if (lines > 2 * learned.Count + 16)
            {
                Rewrite(path, learned);
            }

            PruneOtherFiles(directory, path);
            var writes = new BlockingCollection<string>();
            var writer = new Thread(() => WriteLoop(path, writes))
            {
                IsBackground = true,
                Priority = ThreadPriority.BelowNormal,
                Name = "SharpEmu Metal v2 device data loads",
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

            DeviceDataLoads.Learned += OnLearned;
            writer.Start();
            Console.Error.WriteLine($"[LOADER][INFO] Device data loads seeded: shaders={learned.Count} slots={slots}");
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            Console.Error.WriteLine($"[LOADER][WARN] Device data loads could not be read: {exception.Message}");
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

        DeviceDataLoads.Learned -= OnLearned;
        writes.CompleteAdding();
        writer?.Join(TimeSpan.FromSeconds(10));
    }

    private static void OnLearned(ulong hash, int[] indices)
    {
        try
        {
            Volatile.Read(ref _writes)?.Add(Line(hash, indices));
        }
        catch (InvalidOperationException)
        {
            // The store closed with the session.
        }
    }

    internal static string Line(ulong hash, IEnumerable<int> indices) =>
        string.Create(CultureInfo.InvariantCulture, $"{hash:X16} {string.Join(',', indices.Order())}");

    /// <summary>The learned loads by shader; torn or foreign lines are skipped.</summary>
    internal static Dictionary<ulong, HashSet<int>> Load(string path, out int lines)
    {
        lines = 0;
        var learned = new Dictionary<ulong, HashSet<int>>();
        if (!File.Exists(path))
        {
            return learned;
        }

        foreach (var line in File.ReadLines(path))
        {
            lines++;
            var parts = line.Split(' ');
            if (parts.Length != 2 ||
                !ulong.TryParse(parts[0], NumberStyles.HexNumber, CultureInfo.InvariantCulture, out var hash))
            {
                continue;
            }

            var indices = new HashSet<int>();
            var valid = true;
            foreach (var text in parts[1].Split(',', StringSplitOptions.RemoveEmptyEntries))
            {
                if (!int.TryParse(text, NumberStyles.None, CultureInfo.InvariantCulture, out var index))
                {
                    valid = false;
                    break;
                }

                indices.Add(index);
            }

            if (!valid || indices.Count == 0)
            {
                continue;
            }

            if (!learned.TryGetValue(hash, out var set))
            {
                learned[hash] = set = [];
            }

            set.UnionWith(indices);
        }

        return learned;
    }

    private static void Rewrite(string path, Dictionary<ulong, HashSet<int>> learned)
    {
        var temporary = path + ".tmp";
        File.WriteAllLines(temporary, learned.Select(pair => Line(pair.Key, pair.Value)));
        File.Move(temporary, path, overwrite: true);
    }

    internal static void PruneOtherFiles(string directory, string current, string prefix = FilePrefix)
    {
        try
        {
            var others = new DirectoryInfo(directory)
                .EnumerateFiles(prefix + "*.txt")
                .Where(file => !string.Equals(file.Name, Path.GetFileName(current), StringComparison.Ordinal))
                .OrderByDescending(file => file.LastWriteTimeUtc)
                .Skip(MaxOtherFiles);
            foreach (var file in others)
            {
                file.Delete();
            }
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            Console.Error.WriteLine($"[LOADER][WARN] Device data loads could not remove older files: {exception.Message}");
        }
    }

    internal static void WriteLoop(string path, BlockingCollection<string> writes)
    {
        try
        {
            using var stream = new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.Read);
            // A line torn by an earlier exit is ended first, so the next one stays whole.
            if (stream.Length > 0)
            {
                stream.Seek(-1, SeekOrigin.End);
                var last = stream.ReadByte();
                stream.Seek(0, SeekOrigin.End);
                if (last != '\n')
                {
                    stream.WriteByte((byte)'\n');
                }
            }

            using var file = new StreamWriter(stream);
            foreach (var line in writes.GetConsumingEnumerable())
            {
                file.WriteLine(line);
                file.Flush();
            }
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            Console.Error.WriteLine($"[LOADER][WARN] Device data loads could not be written: {exception.Message}");
        }
    }
}
