// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.GpuCommands.Registers;

namespace SharpEmu.Libs.Gpu.Rendering;

/// <summary>
/// SHARPEMU_RESOLVE_SHARDS: how many resolver threads resolve draw and dispatch programs ahead of the GPU worker
/// (ParallelProgramPrefetch), each with its own queue and its own part of the pipeline cache's resolver state (resolve
/// memo, under a lock of its own); what all of them share (the program cache, translation, vertex tables) stays behind
/// one lock taken only on the miss path. 1 restores the single resolver. A job goes to the shard a stable hash of its
/// resolve memo key's fields names (KeyRoute), so a key always resolves on the same shard and keeps its memo entry.
/// </summary>
public static class ResolveShards
{
    public const int Maximum = 8;
    private const int DefaultCount = 2;

    public static readonly int Default = Parse(Environment.GetEnvironmentVariable("SHARPEMU_RESOLVE_SHARDS"), out _);

    private static int _logged;

    // The value from the variable, clamped to 1..Maximum; unset or unreadable is the default (named in the log).
    internal static int Parse(string? value, out string? problem)
    {
        problem = null;
        if (string.IsNullOrWhiteSpace(value))
        {
            return DefaultCount;
        }

        if (!int.TryParse(value.Trim(), System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out var count))
        {
            problem = $"unreadable value '{value}', using {DefaultCount}";
            return DefaultCount;
        }

        if (count < 1 || count > Maximum)
        {
            var clamped = Math.Clamp(count, 1, Maximum);
            problem = $"{count} is outside 1..{Maximum}, using {clamped}";
            return clamped;
        }

        return count;
    }

    // Once per process, when the first prefetch starts.
    internal static void LogOnce(int count)
    {
        if (Interlocked.Exchange(ref _logged, 1) != 0)
        {
            return;
        }

        Parse(Environment.GetEnvironmentVariable("SHARPEMU_RESOLVE_SHARDS"), out var problem);
        Console.Error.WriteLine($"[GPU][PREFETCH] resolve shards={count} (SHARPEMU_RESOLVE_SHARDS{(problem is null ? string.Empty : ": " + problem)})");
    }

    private static ulong Mix(ulong value)
    {
        value ^= value >> 33;
        value *= 0xFF51AFD7ED558CCDUL;
        value ^= value >> 33;
        value *= 0xC4CEB53FE85A87DUL;
        value ^= value >> 33;
        return value;
    }

    // A draw's route: its vertex (export) program address and user data words, and with the pixel stage active its
    // pixel program address and user data words. Every one of them is a field of the resolve memo's key
    // (GraphicsResolveMemo.Fill: the addresses, and the user data as PutUserData counts them), so draws with one key
    // always route alike; the user data spreads the draws of one program pair over the shards.
    public static ulong DrawRoute(RegisterBanks banks, bool pixelActive)
    {
        var vertex = banks.Shader.Vertex;
        var hash = Mix(vertex.ExportAddress ^ 0x9E3779B97F4A7C15UL);
        hash = AddUserData(hash, vertex.GeometryUserScalars, vertex.GeometryResource2.UserScalarCount);
        if (pixelActive)
        {
            var pixel = banks.Shader.Pixel;
            hash = Mix(hash ^ pixel.Address);
            hash = AddUserData(hash, pixel.UserScalars, pixel.Resource2.UserScalarCount);
        }

        return Mix(hash);
    }

    // A dispatch's route: its program address (a dispatch's resolution keeps no per-shard state).
    public static ulong ComputeRoute(RegisterBanks banks) => Mix(banks.Shader.Compute.Address ^ 0x5A5A5A5A5A5A5A5AUL);

    // The words GraphicsResolveMemo.PutUserData puts in the key: the declared count, else the registers written; none
    // when that is over the bank (the key is refused then, and the route keeps to the addresses).
    private static ulong AddUserData(ulong hash, UserScalarRegisters registers, uint declared)
    {
        var count = declared != 0 ? declared : registers.Count;
        if (count > UserScalarRegisters.Capacity)
        {
            return hash;
        }

        hash = Mix(hash ^ count);
        foreach (var word in registers.ValueSpan((int)count))
        {
            hash = (hash ^ word) * 0x100000001B3UL;
        }

        return hash;
    }

    public static int ShardOf(ulong route, int shards) => shards <= 1 ? 0 : (int)(route % (ulong)shards);
}
