// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Core.Memory;
using SharpEmu.HLE;
using SharpEmu.HLE.FileSystem;

namespace SharpEmu.Core.Loader;

/// <summary>
/// Makes a title render at the display the host advertises (the chosen output resolution,
/// rounded to a console mode; <see cref="SharpEmu.Libs.VideoOut.HostVideoOptions.GuestDisplaySize"/>)
/// when its engine sizes its buffers from something other than the sceVideoOut answers. One
/// adapter per engine (<see cref="ApplyMainExecutable"/> lists them), each recognising its
/// engine by a signature in the executable or in /app0 rather than by title. An adapter
/// changes nothing when the title would already pick the size, and logs one
/// <c>resolution:</c> line when it recognises its engine. <see cref="EnvironmentVariable"/>=0
/// turns them all off.
/// </summary>
internal static class EngineResolution
{
    public const string EnvironmentVariable = "SHARPEMU_ENGINE_RESOLUTION";

    internal readonly record struct Segment(ulong Start, byte[] Bytes, bool Executable);

    internal static bool IsEnabled =>
        !string.Equals(Environment.GetEnvironmentVariable(EnvironmentVariable), "0", StringComparison.Ordinal);

    internal static void Report(string engine, string lever, uint width, uint height, string? detail = null) =>
        Console.Error.WriteLine(
            $"[LOADER][INFO] resolution: {engine} {lever} → {width}x{height}" +
            (string.IsNullOrEmpty(detail) ? string.Empty : $" ({detail})"));

    internal static void Warn(string engine, string message) =>
        Console.Error.WriteLine($"[LOADER][WARN] resolution: {engine}: {message}");

    /// <summary>The host directory /app0 resolves to.</summary>
    internal static string? App0Root =>
        Environment.GetEnvironmentVariable("SHARPEMU_APP0_DIR") is { Length: > 0 } root ? root : App0Archive.Root;

    /// <summary>Runs every adapter on the main executable and the title's /app0. Called once,
    /// after relocation and before any patcher that moves code.</summary>
    public static void ApplyMainExecutable(
        IVirtualMemory memory,
        IReadOnlyList<ProgramHeader> programHeaders,
        ulong imageBase)
    {
        if (!IsEnabled)
        {
            return;
        }

        var (width, height) = SharpEmu.Libs.VideoOut.HostVideoHost.CurrentOptions.GuestDisplaySize;
        if (width == 0 || height == 0)
        {
            return;
        }

        var segments = ReadSegments(memory, programHeaders, imageBase);
        var app0 = App0Root;
        Run("Unity", () => UnityResolutionAdapter.Apply(memory, segments, app0, width, height));
        Run("UE", () => UnrealSceneResolutionAdapter.Apply(segments, app0, width, height));
        Run("KEX", () => KexResolutionPatch.Apply(memory, segments, width, height));
        Run("RAGE", () => RageResolutionPatch.Apply(memory, segments, width, height));
        Run("small engine", () => PackedDisplaySizeScanner.Apply(memory, segments, width, height));
    }

    private static void Run(string engine, Action adapter)
    {
        try
        {
            adapter();
        }
        catch (Exception exception) when (exception is not OutOfMemoryException)
        {
            Warn(engine, $"adapter failed: {exception.Message}");
        }
    }

    internal static List<Segment> ReadSegments(
        ICpuMemory memory,
        IReadOnlyList<ProgramHeader> programHeaders,
        ulong imageBase)
    {
        var segments = new List<Segment>();
        foreach (var header in programHeaders)
        {
            if (header.HeaderType != ProgramHeaderType.Load || header.FileSize == 0 || header.FileSize > int.MaxValue)
            {
                continue;
            }

            var bytes = GC.AllocateUninitializedArray<byte>((int)header.FileSize);
            var start = imageBase + header.VirtualAddress;
            if (memory.TryRead(start, bytes))
            {
                segments.Add(new Segment(start, bytes, (header.Flags & ProgramHeaderFlags.Execute) != 0));
            }
        }

        return segments;
    }

    /// <summary>Reads a whole /app0 file (from the archive or the disk), or null when it is absent
    /// or larger than <paramref name="maxLength"/>.</summary>
    internal static byte[]? TryReadApp0File(string hostPath, int maxLength = 1 << 20)
    {
        if (!File.Exists(hostPath) && !App0Archive.TryGetFile(hostPath, out _))
        {
            return null;
        }

        using var stream = App0Archive.OpenReadOrFile(hostPath);
        if (stream.Length > maxLength)
        {
            return null;
        }

        var bytes = new byte[stream.Length];
        stream.ReadExactly(bytes);
        return bytes;
    }

    /// <summary>Serves <paramref name="content"/> for <paramref name="hostPath"/>. A file on disk
    /// can only be replaced at its own length; false when it cannot.</summary>
    internal static bool TryReplaceApp0File(string hostPath, byte[] content, string reason)
    {
        if (File.Exists(hostPath) && new FileInfo(hostPath).Length != content.Length)
        {
            return false;
        }

        App0FileOverlay.Replace(hostPath, content, reason);
        return true;
    }
}
