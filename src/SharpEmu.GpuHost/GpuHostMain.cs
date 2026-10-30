// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
using System.Runtime.InteropServices;
using SharpEmu.GpuRemote;
using SharpEmu.GpuSplit;
using SharpEmu.HLE;
using SharpEmu.Libs.Gpu;
using SharpEmu.Libs.VideoOut;

namespace SharpEmu.GpuHost;

/// <summary>
/// Entry point of the native GPU host (process B of the split).
/// Usage: <c>SharpEmu.GpuHost --shm &lt;region name&gt; --parent &lt;pid&gt; [--spin-ns &lt;n&gt;]</c>;
/// started by <see cref="GpuProcessHost"/>, not by hand.
/// </summary>
/// <remarks>
/// The main thread pumps <see cref="HostMainThread"/>, where the Metal presenter runs its AppKit
/// window; the transport's loops run on threads of their own (<see cref="GpuProcessClient"/>).
/// Once the guest process sends the split's start, <see cref="SplitGpuHost"/> composes the Metal
/// backend over the shared guest memory. <c>SHARPEMU_GPU_HOST_BACKEND=probe</c> composes
/// <see cref="SplitProbeBackend"/> instead: no Metal, no window, for the cross-process tests.
/// </remarks>
public static class GpuHostMain
{
    public const string BackendEnvironment = "SHARPEMU_GPU_HOST_BACKEND";

    public static int Main(string[] args)
    {
        SharpEmu.Logging.GcPauseLog.StartFromEnvironment();
        SharpEmu.CLI.AllocationProfiler.StartIfRequested();
        if (!OperatingSystem.IsMacOS())
        {
            Console.Error.WriteLine("[GPUHOST][ERROR] The GPU host runs only on macOS.");
            return GpuProcessClient.ExitUsage;
        }

        string? region = null;
        var parent = 0;
        var spin = GpuProcessClient.DefaultSpinNanoseconds;
        for (var i = 0; i < args.Length; i++)
        {
            var value = i + 1 < args.Length ? args[i + 1] : null;
            switch (args[i])
            {
                case "--shm" when value is not null:
                    region = value;
                    i++;
                    break;
                case "--parent" when value is not null && int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out parent):
                    i++;
                    break;
                case "--spin-ns" when value is not null && long.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out spin):
                    i++;
                    break;
                default:
                    return Usage($"unexpected argument '{args[i]}'");
            }
        }

        if (region is null || parent <= 1)
        {
            return Usage("--shm and --parent are required");
        }

        MirrorConsoleToLogFile();

        // Before anything can post the window: the presenter runs it on this thread.
        HostMainThread.Enable();
        GpuProcessClient client;
        try
        {
            client = GpuProcessClient.Attach(region, parent, TimeSpan.FromSeconds(10), spin);
        }
        catch (Exception exception) when (exception is IOException or InvalidDataException or ArgumentException)
        {
            Console.Error.WriteLine($"[GPUHOST][ERROR] Could not attach to {region}: {exception.Message}");
            return GpuProcessClient.ExitProtocolError;
        }

        using (client)
        {
            if (Environment.GetEnvironmentVariable("SHARPEMU_GPU_HOST_VERBOSE") == "1")
            {
                Console.Error.WriteLine(
                    $"[GPUHOST][INFO] Attached to {region}: {RuntimeInformation.ProcessArchitecture}, pid {Environment.ProcessId}, " +
                    $"host {client.Host}, doorbell spin {spin} ns.");
            }

            var probe = string.Equals(Environment.GetEnvironmentVariable(BackendEnvironment), "probe", StringComparison.Ordinal);
            // Not disposed: the process exits with the backend still holding guest memory.
            var service = new SplitGpuHost(client.Channel, client.OutboundLock, new SplitGpuHostOptions
            {
                ComposeBackend = probe ? SplitProbeBackend.Compose : ComposeMetal,
                ConfigureVideo = probe ? null : static options => HostVideoHost.TryConfigureVideo(options),
                QueryHdrOutputSupport = probe ? null : static () => HostVideoHost.IsHdrOutputSupported,
            });
            client.Service = service;
            // Calls that wait for the GPU (a frame boundary, a fault's writeback, a bracket) each
            // hold a thread; enough of them that a write fault never queues behind those.
            client.SyncThreadCount = 8;
            // The ring and the calls are on the GPU worker's path (and on the guest's): they start
            // at its class, which a thread CoreCLR creates cannot take later.
            client.StartThread = static (name, stackBytes, body) => SharpEmu.HLE.Host.HostHelperThread.Start(name, stackBytes, body).Join;

            var exitCode = GpuProcessClient.ExitClean;
            var ring = SharpEmu.HLE.Host.HostHelperThread.Start("SharpEmu GPU host ring", 4 * 1024 * 1024, () =>
            {
                try
                {
                    exitCode = client.Run();
                }
                finally
                {
                    // Ends the pump below, after asking the window to close.
                    HostMainThread.Shutdown();
                }
            });
            HostMainThread.Pump();
            _ = ring.Join(TimeSpan.FromSeconds(5));
            return exitCode;
        }
    }

    // The Metal backend as SharpEmu.CLI composes it in one process, over the shared guest memory
    // SplitGpuHost attached as the GPU memory manager.
    private static IGuestGpuBackend ComposeMetal(SplitGpuEnvironment environment)
    {
#pragma warning disable CA1416 // macOS only: Main returns early anywhere else.
        SharpEmu.Platform.MacOS.MacPlatform.Register();
        SharpEmu.Gpu.Metal.MetalGpuBackend.Register();
#pragma warning restore CA1416
        if (!HostVideoHost.TryConfigureVideo(environment.VideoOptions ?? HostVideoOptions.Default))
        {
            Console.Error.WriteLine("[GPUHOST][WARN] The window options could not be applied.");
        }

        return GuestGpu.Current;
    }

    // SHARPEMU_GPU_HOST_LOG: this process's console also goes to that file. The guest process's
    // own log file is a managed mirror of its console that this process cannot write into; the CLI
    // sets the variable next to it (<log>.gpuhost.log) when it has one.
    private static void MirrorConsoleToLogFile()
    {
        var path = Environment.GetEnvironmentVariable(LogEnvironment);
        if (string.IsNullOrWhiteSpace(path))
        {
            return;
        }

        try
        {
            var file = new StreamWriter(new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.ReadWrite), new System.Text.UTF8Encoding(false))
            {
                AutoFlush = true,
            };
            Console.SetError(TextWriter.Synchronized(new TeeWriter(Console.Error, file)));
            Console.SetOut(TextWriter.Synchronized(new TeeWriter(Console.Out, file)));
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            Console.Error.WriteLine($"[GPUHOST][WARN] Could not open the log file {path}: {exception.Message}");
        }
    }

    public const string LogEnvironment = "SHARPEMU_GPU_HOST_LOG";

    private sealed class TeeWriter(TextWriter primary, TextWriter mirror) : TextWriter
    {
        public override System.Text.Encoding Encoding => primary.Encoding;

        public override void Write(char value)
        {
            primary.Write(value);
            mirror.Write(value);
        }

        public override void Write(string? value)
        {
            primary.Write(value);
            mirror.Write(value);
        }

        public override void WriteLine(string? value)
        {
            primary.WriteLine(value);
            mirror.WriteLine(value);
        }

        public override void Flush()
        {
            primary.Flush();
            mirror.Flush();
        }
    }

    private static int Usage(string problem)
    {
        Console.Error.WriteLine($"[GPUHOST][ERROR] {problem}. Usage: SharpEmu.GpuHost --shm <region> --parent <pid> [--spin-ns <n>]");
        return GpuProcessClient.ExitUsage;
    }
}
