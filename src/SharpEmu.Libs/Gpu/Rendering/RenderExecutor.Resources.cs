// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Images;

namespace SharpEmu.Libs.Gpu.Rendering;

public sealed partial class RenderExecutor
{
    internal const ulong IndexedIndirectArgumentsSize = 20;
    internal const ulong AutoIndirectArgumentsSize = 16;

    // Only a store or an atomic through the device-address page table counts: a program that
    // merely reads through it writes nothing, and ending the pass after it would store and
    // reload every attachment on a tile-based GPU for no reason.
    private bool DrawWritesMemory(ShaderStageResources stage) =>
        stage.Program is { } program &&
        (program.WritesDeviceAddresses || WritesStorageImage(program) || HasBufferWrites(stage));

    private bool HasBufferWrites(ShaderStageResources stage)
    {
        var program = stage.Program ?? throw _host.Fatal("A shader stage has no program.");
        var descriptors = stage.Resources.Buffers;
        if (descriptors.Length != program.Buffers.Length)
        {
            throw _host.Fatal($"The buffer descriptor count does not match the program: descriptors={descriptors.Length} program={program.Buffers.Length}.");
        }

        for (var i = 0; i < program.Buffers.Length; i++)
        {
            if (!program.Buffers[i].Written)
            {
                continue;
            }

            if (descriptors[i].Length < 4)
            {
                throw _host.Fatal($"A written buffer descriptor is too short: index={i} words={descriptors[i].Length}.");
            }

            var descriptor = BufferDescriptorWords.From(descriptors[i]);
            var footprint = descriptor.Footprint() ?? throw _host.Fatal(
                $"The written buffer footprint overflows: index={i} stride={descriptor.Stride} records={descriptor.RecordCount}.");
            if (descriptor.Address != 0 && footprint != 0)
            {
                return true;
            }
        }

        return false;
    }

    // Any write a draw makes other than through its attachments, as the conservative reading of
    // the program's resource use: a written or atomic buffer whatever its descriptor holds, any
    // storage image, the GDS, mip statistics and device-address stores.
    internal static bool WritesOutsideTargets(ShaderStageResources stage) => WriteOutsideTargets(stage) is not null;

    // What WritesOutsideTargets found, as a literal for the leave-out diagnostics; null for nothing.
    // Mip statistics are left out of the count when the host says so (IRenderHost.LeavesOutMipStatisticsWhileCompiling).
    internal static string? WriteOutsideTargets(ShaderStageResources stage, bool mipStatisticsAllowed = false)
    {
        if (stage.Program is not { } program)
        {
            return null;
        }

        if (program.WritesDeviceAddresses)
        {
            return "device-address-store";
        }

        if (!mipStatisticsAllowed && (program.Bindings?.MipStatisticsCount ?? 0) != 0)
        {
            return "mip-statistics";
        }

        foreach (var buffer in program.Buffers)
        {
            if (buffer.Atomic)
            {
                return "atomic-buffer";
            }

            if (buffer.Written)
            {
                return "written-buffer";
            }
        }

        foreach (var image in program.Images)
        {
            if (image.Written || image.Class == ImageResourceClass.Storage)
            {
                return "storage-image";
            }
        }

        if (program.Bindings is { } bindings)
        {
            // By index: a foreach over the interface boxes its enumerator, once per draw.
            var descriptors = bindings.Descriptors;
            for (var index = 0; index < descriptors.Count; index++)
            {
                if (descriptors[index].Kind == SharpEmu.ShaderCompiler.Resources.DescriptorBindingKind.GlobalDataShare)
                {
                    return "gds";
                }
            }
        }

        return null;
    }

    // SHARPEMU_SKIP_PS=<hash>[,<hash>...]: draws whose pixel program is one of these are not made,
    // to see what a program contributes to a frame by its absence.
    internal static readonly HashSet<ulong> SkippedPixelPrograms =
        (Environment.GetEnvironmentVariable("SHARPEMU_SKIP_PS") ?? string.Empty)
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(static text => ulong.TryParse(text.Replace("0x", string.Empty, StringComparison.OrdinalIgnoreCase),
                System.Globalization.NumberStyles.HexNumber, System.Globalization.CultureInfo.InvariantCulture, out var hash) ? hash : 0)
            .Where(static hash => hash != 0)
            .ToHashSet();

    // SHARPEMU_SKIP_CS=<hash>[,<hash>...]: dispatches of these compute programs are not made, as SKIP_PS does for draws.
    internal static readonly HashSet<ulong> SkippedComputePrograms =
        (Environment.GetEnvironmentVariable("SHARPEMU_SKIP_CS") ?? string.Empty)
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(static text => ulong.TryParse(text.Replace("0x", string.Empty, StringComparison.OrdinalIgnoreCase),
                System.Globalization.NumberStyles.HexNumber, System.Globalization.CultureInfo.InvariantCulture, out var hash) ? hash : 0)
            .Where(static hash => hash != 0)
            .ToHashSet();

    // SHARPEMU_SKIP_FILE=<path>: more programs to skip, pixel or compute, one hash a line, read again whenever
    // the file changes, so one session can take the passes of a frame away one after another.
    internal static class LiveSkips
    {
        private static readonly string? FilePath = Environment.GetEnvironmentVariable("SHARPEMU_SKIP_FILE");
        private static long _checkedTicks;
        private static DateTime _written;
        private static volatile HashSet<ulong> _hashes = [];

        public static bool Enabled => FilePath is not null;

        public static bool Contains(ulong hash)
        {
            if (FilePath is null)
            {
                return false;
            }

            var now = Environment.TickCount64;
            if (now - Interlocked.Read(ref _checkedTicks) > 500)
            {
                Interlocked.Exchange(ref _checkedTicks, now);
                Reload();
            }

            return _hashes.Contains(hash);
        }

        private static void Reload()
        {
            try
            {
                var written = File.Exists(FilePath) ? File.GetLastWriteTimeUtc(FilePath!) : DateTime.MinValue;
                if (written == _written)
                {
                    return;
                }

                _written = written;
                var hashes = written == DateTime.MinValue ? [] : File.ReadAllLines(FilePath!)
                    .Select(static line => line.Split('#')[0].Trim())
                    .Select(static text => ulong.TryParse(text.Replace("0x", string.Empty, StringComparison.OrdinalIgnoreCase),
                        System.Globalization.NumberStyles.HexNumber, System.Globalization.CultureInfo.InvariantCulture, out var hash) ? hash : 0)
                    .Where(static hash => hash != 0)
                    .ToHashSet();
                _hashes = hashes;
                Console.Error.WriteLine($"[GPU][INFO] SHARPEMU_SKIP_FILE: skipping {hashes.Count} programs: " +
                    string.Join(",", hashes.Select(static hash => hash.ToString("X16", System.Globalization.CultureInfo.InvariantCulture))));
            }
            catch (IOException)
            {
            }
        }
    }

}
