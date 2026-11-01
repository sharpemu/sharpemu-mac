// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
using System.Runtime.InteropServices;

namespace SharpEmu.GpuRemote;

/// <summary>Start-up knobs of <see cref="GpuProcessHost"/>.</summary>
public sealed record GpuProcessHostOptions
{
    /// <summary>The GPU host executable; null searches the usual places (see <see cref="GpuProcessHost.FindExecutable"/>).</summary>
    public string? ExecutablePath { get; init; }

    public GpuRemoteLayout Layout { get; init; } = new();

    /// <summary>How long the host may take to attach and answer the handshake.</summary>
    public TimeSpan HandshakeTimeout { get; init; } = TimeSpan.FromSeconds(20);

    /// <summary>How long <see cref="GpuProcessHost.Dispose"/> waits for a clean exit before signalling.</summary>
    public TimeSpan ShutdownTimeout { get; init; } = TimeSpan.FromSeconds(3);

    /// <summary>
    /// Busy-wait of the host's ring and call loops before they sleep; null keeps the host's
    /// default. 0 measures the doorbells alone.
    /// </summary>
    public long? HostSpinNanoseconds { get; init; }

    /// <summary>Accept a translated (x86-64) host. Only for experiments; the point of B is to be native.</summary>
    public bool AllowTranslatedHost { get; init; }
}

/// <summary>The GPU host failed to start, answer, or stay alive.</summary>
public sealed class GpuProcessException(string message, Exception? inner = null) : Exception(message, inner);

/// <summary>
/// Process A's handle on the native GPU host (process B): creates the shared region, spawns
/// <c>SharpEmu.GpuHost</c> natively arm64, completes the handshake, watches it, and shuts it down.
/// </summary>
/// <remarks>
/// <para>Opt-in behind <see cref="EnableEnvironment"/> (<c>SHARPEMU_GPU_PROCESS=1</c>), under which
/// SharpEmu.GpuSplit's <c>SplitGpuSession</c> starts one and puts the <c>IGuestGpuBackend</c> proxy
/// on it (docs/macos/split-protocol.md).</para>
///
/// <para>Lifetime: if A exits or dies, B notices through its parent watch and exits; if B dies,
/// A's watch reaps it, marks the channel's peer exited and wakes every waiter, so pings, calls
/// and ring waits return failure instead of hanging. The region's name is unlinked as soon as
/// B has attached.</para>
///
/// <para>Threading: writers of the A → B ring take <see cref="SyncRoot"/>. <see cref="TryPing"/>
/// and the benchmarks also read the B → A ring, so they are for a host nothing else reads from:
/// once a split session owns the rings it drains the B → A ring itself. Sync calls are lock-free
/// and may run concurrently from any thread.</para>
/// </remarks>
public sealed class GpuProcessHost : IDisposable
{
    public const string EnableEnvironment = "SHARPEMU_GPU_PROCESS";
    public const string ExecutablePathEnvironment = "SHARPEMU_GPU_HOST_PATH";
    public const string ExecutableName = "SharpEmu.GpuHost";

    private const string RegionPrefix = "segpu";

    private readonly GpuProcessHostOptions _options;
    private readonly ManualResetEventSlim _exited = new(false);
    private ulong _pingSequence;
    private int _exitStatus = int.MinValue;
    private int _disposed;

    private GpuProcessHost(GpuRemoteChannel channel, int processId, string executablePath, GpuProcessHostOptions options)
    {
        Channel = channel;
        ProcessId = processId;
        ExecutablePath = executablePath;
        _options = options;
    }

    /// <summary>True when <c>SHARPEMU_GPU_PROCESS=1</c>.</summary>
    public static bool IsRequested =>
        string.Equals(Environment.GetEnvironmentVariable(EnableEnvironment), "1", StringComparison.Ordinal);

    /// <summary>
    /// Makes the split the default: with <c>SHARPEMU_GPU_PROCESS</c> unset, sets it to 1 when this
    /// process runs translated on Apple silicon and the GPU host is beside it, so an Intel Mac or a
    /// build without the host keeps one process and <c>SHARPEMU_GPU_PROCESS=0</c> opts out. Runs first
    /// in the emulator's Main: the backing layout, the host views and the backend read the switch.
    /// </summary>
    public static void ApplyDefault()
    {
        if (!string.IsNullOrEmpty(Environment.GetEnvironmentVariable(EnableEnvironment)) || !OperatingSystem.IsMacOS() ||
            RemoteNative.ProcessTranslated() != 1 || FindExecutable() is null)
        {
            return;
        }

        Environment.SetEnvironmentVariable(EnableEnvironment, "1");
    }

    /// <summary>"ulock" when waits block in the kernel, "poll" for the fallback (see <c>SHARPEMU_GPU_DOORBELL</c>).</summary>
    public static string DoorbellMechanism => RemoteWait.MechanismName;

    public GpuRemoteChannel Channel { get; }

    public int ProcessId { get; }

    public string ExecutablePath { get; }

    /// <summary>What the GPU host reported in the handshake.</summary>
    public GpuPeerInfo Peer { get; private set; }

    /// <summary>Wall time from spawn to the handshake answer.</summary>
    public TimeSpan StartupTime { get; private set; }

    /// <summary>Serializes users of the channel's rings (see the type remarks).</summary>
    public object SyncRoot { get; } = new();

    public bool HasExited => _exited.IsSet;

    /// <summary>The host's wait status once it has exited and been reaped (exit code in bits 8-15), else null.</summary>
    public int? ExitStatus => _exited.IsSet && _exitStatus != int.MinValue ? _exitStatus : null;

    /// <summary>The host's exit code if it exited normally, else null.</summary>
    public int? ExitCode => ExitStatus is { } status && (status & 0x7F) == 0 ? (status >> 8) & 0xFF : null;

    /// <summary>Starts the GPU host when <see cref="IsRequested"/>; otherwise returns null.</summary>
    public static GpuProcessHost? StartIfRequested(GpuProcessHostOptions? options = null) =>
        IsRequested ? Start(options) : null;

    /// <summary>
    /// Creates the region, spawns the GPU host and waits for its handshake. Throws
    /// <see cref="GpuProcessException"/> if it cannot be found, does not start, answers with the
    /// wrong protocol, runs translated, or misses <see cref="GpuProcessHostOptions.HandshakeTimeout"/>.
    /// </summary>
    public static GpuProcessHost Start(GpuProcessHostOptions? options = null)
    {
        if (!OperatingSystem.IsMacOS())
        {
            throw new PlatformNotSupportedException("The out-of-process GPU host exists only on macOS.");
        }

        options ??= new GpuProcessHostOptions();
        var executable = FindExecutable(options.ExecutablePath)
            ?? throw new GpuProcessException(
                $"{options.ExecutablePath ?? ExecutableName} was not found (base directory {AppContext.BaseDirectory}; see {ExecutablePathEnvironment}).");

        var channel = GpuRemoteChannel.Create(SharedMemoryRegion.CreateUniqueName(RegionPrefix), options.Layout);
        GpuProcessHost? host = null;
        try
        {
            // Queued before B exists; B reads it first thing after attaching.
            var hello = GpuPeerInfo.DescribeThisProcess();
            if (!channel.Outbound.TryWrite((uint)GpuRemoteMessage.Hello, in hello))
            {
                throw new GpuProcessException("The handshake does not fit the ring.");
            }

            var arguments = new List<string>
            {
                "--shm", channel.Name,
                "--parent", Environment.ProcessId.ToString(CultureInfo.InvariantCulture),
            };
            if (options.HostSpinNanoseconds is { } spin)
            {
                arguments.Add("--spin-ns");
                arguments.Add(spin.ToString(CultureInfo.InvariantCulture));
            }

            var started = RemoteNative.MonotonicNanoseconds();
            var pid = GpuProcessLauncher.Spawn(executable, arguments);
            host = new GpuProcessHost(channel, pid, executable, options);
            ProcessExitWatch.Start(pid, "SharpEmu GPU host watch", host.OnExited, reapChild: true);

            host.Peer = host.AwaitHandshake(started);
            host.StartupTime = TimeSpan.FromTicks((RemoteNative.MonotonicNanoseconds() - started) / 100);
            channel.Unlink();

            // One sync call on an ordinary thread, so the caller path is compiled and its
            // native entry points resolved before a signal handler can use it.
            SyncCallTable.PrepareForSignalContext();
            if (!host.TrySyncPing(TimeSpan.FromSeconds(5), out _))
            {
                throw new GpuProcessException("The GPU host did not answer the first synchronous call.");
            }

            Console.Error.WriteLine(
                $"[GPU-PROCESS][INFO] GPU host up in {host.StartupTime.TotalMilliseconds:F0} ms: {host.Peer}; " +
                $"region {channel.Name}, {channel.Size / 1024} KiB, doorbell {RemoteWait.MechanismName}.");
            return host;
        }
        catch (Exception exception)
        {
            if (host is not null)
            {
                host.Kill();
                host.Dispose();
            }
            else
            {
                channel.Dispose();
            }

            if (exception is GpuProcessException)
            {
                throw;
            }

            throw new GpuProcessException($"Could not start the GPU host {executable}: {exception.Message}", exception);
        }
    }

    /// <summary>
    /// Where the GPU host is looked for: <paramref name="explicitPath"/> or else
    /// <c>SHARPEMU_GPU_HOST_PATH</c> when given (and nowhere else then, so a typo fails instead
    /// of silently running another build); otherwise relative to this process's base directory -
    /// <c>arm64/</c> (an app bundle, shared with the GUI host), <c>../osx-arm64/</c> (an osx-x64 build output next to the
    /// osx-arm64 one) and <c>osx-arm64/</c> (a RID-less build output, the tests).
    /// </summary>
    public static string? FindExecutable(string? explicitPath = null)
    {
        var candidates = new List<string>();
        var named = !string.IsNullOrEmpty(explicitPath)
            ? explicitPath
            : Environment.GetEnvironmentVariable(ExecutablePathEnvironment);
        if (!string.IsNullOrEmpty(named))
        {
            candidates.Add(named);
        }
        else
        {
            var baseDirectory = AppContext.BaseDirectory;
            candidates.Add(Path.Combine(baseDirectory, "arm64", ExecutableName));
            candidates.Add(Path.Combine(baseDirectory, "..", "osx-arm64", ExecutableName));
            candidates.Add(Path.Combine(baseDirectory, "osx-arm64", ExecutableName));
        }

        foreach (var candidate in candidates)
        {
            var full = Path.GetFullPath(candidate);
            if (File.Exists(full))
            {
                return full;
            }
        }

        return null;
    }

    /// <summary>
    /// One ring round trip: a <see cref="GpuRemoteMessage.Ping"/> out, its
    /// <see cref="GpuRemoteMessage.Pong"/> back. False on timeout or if the host is gone.
    /// </summary>
    public bool TryPing(TimeSpan timeout, out TimeSpan roundTrip)
    {
        var answered = TryPing(ToNanoseconds(timeout), SpscRing.DefaultSpinNanoseconds, out long nanoseconds);
        roundTrip = TimeSpan.FromTicks(nanoseconds / 100);
        return answered;
    }

    /// <summary>
    /// <see cref="TryPing(TimeSpan, out TimeSpan)"/> with the reader's spin set explicitly and the
    /// round trip in nanoseconds.
    /// </summary>
    public bool TryPing(long timeoutNanoseconds, long spinNanoseconds, out long roundTripNanoseconds)
    {
        roundTripNanoseconds = 0;
        lock (SyncRoot)
        {
            var sequence = ++_pingSequence;
            var start = RemoteNative.MonotonicNanoseconds();
            var ping = new PingMessage { Sequence = sequence, Timestamp = start };
            if (!Channel.Outbound.Write((uint)GpuRemoteMessage.Ping, in ping, timeoutNanoseconds))
            {
                return false;
            }

            while (true)
            {
                if (!Channel.Inbound.WaitForData(RemoteWait.Remaining(start, timeoutNanoseconds), spinNanoseconds))
                {
                    if (!Channel.IsPeerAlive || RemoteWait.Remaining(start, timeoutNanoseconds) == 0)
                    {
                        return false;
                    }

                    continue;
                }

                if (!Channel.Inbound.TryRead(out var type, out var payload))
                {
                    continue;
                }

                var matched = type == (uint)GpuRemoteMessage.Pong && RingReader.Read<PingMessage>(payload).Sequence == sequence;
                Channel.Inbound.Advance();
                if (matched)
                {
                    roundTripNanoseconds = RemoteNative.MonotonicNanoseconds() - start;
                    return true;
                }
            }
        }
    }

    /// <summary>One synchronous-slot round trip (<see cref="GpuRemoteSyncOpcode.Ping"/>).</summary>
    public bool TrySyncPing(TimeSpan timeout, out TimeSpan roundTrip)
    {
        var answered = TrySyncPing(ToNanoseconds(timeout), SyncCallTable.DefaultSpinNanoseconds, out long nanoseconds);
        roundTrip = TimeSpan.FromTicks(nanoseconds / 100);
        return answered;
    }

    /// <summary>
    /// <see cref="TrySyncPing(TimeSpan, out TimeSpan)"/> with the caller's spin set explicitly and
    /// the round trip in nanoseconds.
    /// </summary>
    public unsafe bool TrySyncPing(long timeoutNanoseconds, long spinNanoseconds, out long roundTripNanoseconds)
    {
        var start = RemoteNative.MonotonicNanoseconds();
        var request = new PingMessage { Sequence = Interlocked.Increment(ref _pingSequence), Timestamp = start };
        PingMessage response = default;
        var status = Channel.SyncCalls.Call(
            (uint)GpuRemoteSyncOpcode.Ping,
            (byte*)&request,
            sizeof(PingMessage),
            (byte*)&response,
            sizeof(PingMessage),
            out var length,
            out var code,
            timeoutNanoseconds,
            spinNanoseconds);
        roundTripNanoseconds = RemoteNative.MonotonicNanoseconds() - start;
        return status == SyncCallStatus.Completed && code == 0 && length == sizeof(PingMessage) && response.Sequence == request.Sequence;
    }

    /// <summary>Waits until the host has exited and been reaped.</summary>
    public bool WaitForExit(TimeSpan timeout) => _exited.Wait(timeout);

    /// <summary>
    /// Asks the host to exit, waits up to <see cref="GpuProcessHostOptions.ShutdownTimeout"/>,
    /// then escalates to SIGTERM and SIGKILL; finally unmaps the region.
    /// </summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        if (!_exited.IsSet)
        {
            lock (SyncRoot)
            {
                _ = Channel.Outbound.TryWrite((uint)GpuRemoteMessage.Shutdown, ReadOnlySpan<byte>.Empty);
            }

            if (!_exited.Wait(_options.ShutdownTimeout))
            {
                Console.Error.WriteLine($"[GPU-PROCESS][WARN] GPU host {ProcessId} ignored the shutdown request; terminating it.");
                _ = RemoteNative.kill(ProcessId, RemoteNative.SIGTERM);
                if (!_exited.Wait(TimeSpan.FromSeconds(2)))
                {
                    Kill();
                    _ = _exited.Wait(TimeSpan.FromSeconds(2));
                }
            }
        }

        Channel.Dispose();
    }

    private void Kill()
    {
        if (!_exited.IsSet)
        {
            _ = RemoteNative.kill(ProcessId, RemoteNative.SIGKILL);
        }
    }

    private GpuPeerInfo AwaitHandshake(long started)
    {
        var timeout = ToNanoseconds(_options.HandshakeTimeout);
        var inbound = Channel.Inbound;
        while (true)
        {
            if (!inbound.WaitForData(RemoteWait.Remaining(started, timeout), spinNanoseconds: 0))
            {
                if (Channel.IsPeerAlive && RemoteWait.Remaining(started, timeout) != 0)
                {
                    continue;
                }

                throw _exited.IsSet || !Channel.IsPeerAlive
                    ? new GpuProcessException($"The GPU host {ExecutablePath} exited during the handshake ({DescribeExit()}).")
                    : new GpuProcessException(
                        $"The GPU host {ExecutablePath} did not answer the handshake within {_options.HandshakeTimeout.TotalSeconds:F0} s.");
            }

            if (!inbound.TryRead(out var type, out var payload))
            {
                continue;
            }

            if (type != (uint)GpuRemoteMessage.HelloAck)
            {
                inbound.Advance();
                throw new GpuProcessException($"The GPU host opened with record type {type} instead of a handshake.");
            }

            var peer = GpuPeerInfo.From(RingReader.Read<HelloMessage>(payload));
            inbound.Advance();
            if (peer.ProtocolVersion != GpuRemoteProtocol.Version)
            {
                throw new GpuProcessException(
                    $"The GPU host speaks protocol {peer.ProtocolVersion}; this emulator speaks {GpuRemoteProtocol.Version}. Rebuild both.");
            }

            if (peer.ProcessId != ProcessId)
            {
                throw new GpuProcessException($"The handshake came from pid {peer.ProcessId}, not the spawned {ProcessId}.");
            }

            if (peer.Translated == 1 && !_options.AllowTranslatedHost)
            {
                throw new GpuProcessException("The GPU host runs under Rosetta; it must run natively.");
            }

            return peer;
        }
    }

    private void OnExited(int status)
    {
        _exitStatus = status;
        Channel.MarkPeerExited();
        _exited.Set();
        if (Volatile.Read(ref _disposed) == 0)
        {
            Console.Error.WriteLine($"[GPU-PROCESS][WARN] GPU host {ProcessId} exited unexpectedly ({DescribeExit()}).");
        }
    }

    private string DescribeExit()
    {
        if (ExitStatus is not { } status)
        {
            return "status unknown";
        }

        var signal = status & 0x7F;
        return signal == 0 ? $"exit code {(status >> 8) & 0xFF}" : $"signal {signal}";
    }

    private static long ToNanoseconds(TimeSpan timeout) =>
        timeout == Timeout.InfiniteTimeSpan ? -1 : (long)Math.Min(timeout.Ticks * 100.0, long.MaxValue);
}
