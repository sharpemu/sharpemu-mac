// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;

namespace SharpEmu.GpuRemote;

/// <summary>
/// Process B's end of the split: attaches to the region A created, answers the handshake,
/// watches A, and runs the message loop. The loop answers pings and the throughput probes
/// itself and hands every other record and call to the <see cref="Service"/> - the split's
/// GPU host (SharpEmu.GpuSplit) - when one is installed.
/// </summary>
/// <remarks>
/// <para>Threads: the caller of <see cref="Run"/> drains the A → B ring, and
/// <see cref="SyncThreadCount"/> threads serve the synchronous call table, each claiming one
/// call at a time (<see cref="SyncCallTable.ServeOne"/>), so a call never queues behind ring
/// traffic and a slow call (a frame boundary waiting for the GPU) never holds up a quick one
/// (a write fault). All spin for <see cref="SpinNanoseconds"/> after each piece of work before
/// sleeping on their doorbells. When A exits - cleanly, by <see cref="GpuRemoteMessage.Shutdown"/>,
/// or by dying, which the parent watch reports - the loops end and <see cref="Run"/> returns.</para>
///
/// <para>The B → A ring has one writer at a time: the ring thread's own answers and every
/// thread of the service take <see cref="OutboundLock"/> around their writes.</para>
/// </remarks>
public sealed class GpuProcessClient : IDisposable
{
    /// <summary>Exit codes of the GPU host process.</summary>
    public const int ExitClean = 0;
    public const int ExitUsage = 2;
    public const int ExitHostGone = 3;
    public const int ExitProtocolError = 4;

    public const long DefaultSpinNanoseconds = 50_000;

    private readonly int _parentProcessId;
    private readonly ManualResetEventSlim _parentExited = new(false);
    private readonly List<Func<TimeSpan, bool>> _syncThreads = new();
    private volatile bool _stopRequested;
    private int _disposed;

    private GpuProcessClient(GpuRemoteChannel channel, int parentProcessId, GpuPeerInfo host, long spinNanoseconds)
    {
        Channel = channel;
        _parentProcessId = parentProcessId;
        Host = host;
        SpinNanoseconds = spinNanoseconds;
    }

    public GpuRemoteChannel Channel { get; }

    /// <summary>What process A reported in its handshake.</summary>
    public GpuPeerInfo Host { get; }

    public long SpinNanoseconds { get; }

    /// <summary>Records consumed by <see cref="GpuRemoteMessage.Sink"/>; diagnostics.</summary>
    public long SinkRecords { get; private set; }

    /// <summary>
    /// Handles the records and calls this client does not know; set before <see cref="Run"/>.
    /// Null drops them with a warning (records) or answers status -1 (calls).
    /// </summary>
    public IGpuHostService? Service { get; set; }

    /// <summary>Threads serving the A → B call table; read by <see cref="Run"/>.</summary>
    public int SyncThreadCount { get; set; } = 4;

    /// <summary>
    /// Starts a call thread (name, stack bytes, body) and returns its join; read by <see cref="Run"/>.
    /// Null starts managed threads. The host sets one that starts them at a QoS class, which this
    /// assembly cannot reach.
    /// </summary>
    public Func<string, int, Action, Func<TimeSpan, bool>>? StartThread { get; set; }

    /// <summary>Taken around every write to the B → A ring (see the type remarks).</summary>
    public object OutboundLock { get; } = new();

    /// <summary>
    /// Attaches to <paramref name="regionName"/>, reads A's handshake, answers it, and starts
    /// watching <paramref name="parentProcessId"/> (pass 0 to skip the watch, for in-process tests).
    /// </summary>
    public static GpuProcessClient Attach(string regionName, int parentProcessId, TimeSpan handshakeTimeout, long spinNanoseconds = DefaultSpinNanoseconds)
    {
        var channel = GpuRemoteChannel.Attach(regionName);
        try
        {
            var inbound = channel.Inbound;
            var timeout = (long)(handshakeTimeout.Ticks * 100.0);
            var start = RemoteNative.MonotonicNanoseconds();
            uint type;
            ReadOnlySpan<byte> payload;
            while (!inbound.TryRead(out type, out payload))
            {
                if (!inbound.WaitForData(RemoteWait.Remaining(start, timeout), spinNanoseconds: 0) &&
                    (!channel.IsPeerAlive || RemoteWait.Remaining(start, timeout) == 0))
                {
                    throw new InvalidDataException($"No handshake from the host process within {handshakeTimeout.TotalSeconds:F0} s.");
                }
            }

            if (type != (uint)GpuRemoteMessage.Hello)
            {
                throw new InvalidDataException($"The host process opened with record type {type} instead of a handshake.");
            }

            var host = GpuPeerInfo.From(RingReader.Read<HelloMessage>(payload));
            inbound.Advance();

            // Answered even on a version mismatch, so A can report both versions.
            var reply = GpuPeerInfo.DescribeThisProcess();
            if (!channel.Outbound.TryWrite((uint)GpuRemoteMessage.HelloAck, in reply))
            {
                throw new InvalidDataException("The handshake answer does not fit the ring.");
            }

            if (host.ProtocolVersion != GpuRemoteProtocol.Version)
            {
                throw new InvalidDataException(
                    $"The host process speaks protocol {host.ProtocolVersion}; this GPU host speaks {GpuRemoteProtocol.Version}.");
            }

            var client = new GpuProcessClient(channel, parentProcessId, host, spinNanoseconds);
            if (parentProcessId > 0)
            {
                if (RemoteNative.getppid() != parentProcessId)
                {
                    Console.Error.WriteLine(
                        $"[GPUHOST][WARN] Parent is {RemoteNative.getppid()}, not the host process {parentProcessId}; watching {parentProcessId} anyway.");
                }

                ProcessExitWatch.Start(parentProcessId, "SharpEmu GPU host parent watch", client.OnParentExited, reapChild: false);
            }

            return client;
        }
        catch
        {
            channel.Dispose();
            throw;
        }
    }

    /// <summary>Set once the parent watch has seen process A exit.</summary>
    public bool ParentExited => _parentExited.IsSet;

    /// <summary>
    /// Runs the message loop until A asks to shut down (<see cref="ExitClean"/>) or goes away
    /// (<see cref="ExitHostGone"/>); <see cref="ExitProtocolError"/> on a corrupt ring.
    /// </summary>
    public int Run()
    {
        for (var index = 0; index < Math.Max(1, SyncThreadCount); index++)
        {
            var name = $"SharpEmu GPU host sync calls {index}";
            if (StartThread is { } start)
            {
                _syncThreads.Add(start(name, 1024 * 1024, ServeSyncCalls));
                continue;
            }

            var thread = new Thread(ServeSyncCalls, 1024 * 1024) { IsBackground = true, Name = name };
            _syncThreads.Add(thread.Join);
            thread.Start();
        }

        try
        {
            return RunRing();
        }
        catch (InvalidDataException exception)
        {
            Console.Error.WriteLine($"[GPUHOST][ERROR] {exception.Message}");
            return ExitProtocolError;
        }
        finally
        {
            _stopRequested = true;
            Channel.SyncCalls.WakeAll();
            foreach (var join in _syncThreads)
            {
                _ = join(TimeSpan.FromSeconds(2));
            }

            Service?.Stop();
        }
    }

    /// <summary>Makes <see cref="Run"/> return at its next wake-up.</summary>
    public void RequestStop()
    {
        _stopRequested = true;
        Channel.WakeAll();
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        _stopRequested = true;
        Channel.WakeAll();
        foreach (var join in _syncThreads)
        {
            _ = join(TimeSpan.FromSeconds(2));
        }

        Channel.Dispose();
    }

    private int RunRing()
    {
        var inbound = Channel.Inbound;
        var outbound = Channel.Outbound;
        while (true)
        {
            if (!inbound.WaitForData(-1, SpinNanoseconds))
            {
                if (!Channel.IsPeerAlive)
                {
                    return ExitHostGone;
                }

                if (_stopRequested)
                {
                    return ExitClean;
                }

                continue;
            }

            while (inbound.TryRead(out var type, out var payload))
            {
                switch ((GpuRemoteMessage)type)
                {
                    case GpuRemoteMessage.Ping:
                    {
                        var ping = RingReader.Read<PingMessage>(payload);
                        inbound.Advance();
                        bool written;
                        lock (OutboundLock)
                        {
                            written = outbound.Write((uint)GpuRemoteMessage.Pong, in ping, -1);
                        }

                        if (!written)
                        {
                            return ExitHostGone;
                        }

                        break;
                    }

                    case GpuRemoteMessage.Shutdown:
                        inbound.Advance();
                        return ExitClean;

                    case GpuRemoteMessage.Sink:
                        inbound.Advance();
                        SinkRecords++;
                        break;

                    case GpuRemoteMessage.Burst:
                    {
                        var burst = RingReader.Read<BurstMessage>(payload);
                        inbound.Advance();
                        bool written;
                        lock (OutboundLock)
                        {
                            written = WriteBurst(outbound, burst);
                        }

                        if (!written)
                        {
                            return ExitHostGone;
                        }

                        break;
                    }

                    default:
                        if (Service is { } service)
                        {
                            try
                            {
                                service.HandleRecord(type, payload);
                            }
                            catch (Exception exception) when (exception is not InvalidDataException)
                            {
                                Console.Error.WriteLine($"[GPUHOST][ERROR] Record type {type} failed: {exception}");
                            }
                        }
                        else
                        {
                            Console.Error.WriteLine($"[GPUHOST][WARN] Ignoring unknown record type {type} ({payload.Length} bytes).");
                        }

                        inbound.Advance();
                        break;
                }
            }
        }
    }

    private static bool WriteBurst(RingWriter outbound, BurstMessage burst)
    {
        var length = Math.Clamp(burst.PayloadLength, 0, outbound.MaxPayloadLength);
        Span<byte> payload = stackalloc byte[Math.Min(length, 4096)];
        if (length > payload.Length)
        {
            payload = new byte[length];
        }

        for (var i = 0; i < burst.Count; i++)
        {
            MemoryMarshal.Write(payload, in i);
            if (!outbound.Write((uint)GpuRemoteMessage.BurstItem, payload, -1))
            {
                return false;
            }
        }

        return true;
    }

    private void ServeSyncCalls()
    {
        var calls = Channel.SyncCalls;
        SyncCallHandler handler = Serve;
        while (!_stopRequested)
        {
            if (calls.WaitForRequests(-1, SpinNanoseconds))
            {
                while (!_stopRequested && calls.ServeOne(handler))
                {
                }
            }
            else if (!Channel.IsPeerAlive)
            {
                return;
            }
        }
    }

    private int Serve(uint opcode, ReadOnlySpan<byte> request, Span<byte> response, out int status)
    {
        switch ((GpuRemoteSyncOpcode)opcode)
        {
            case GpuRemoteSyncOpcode.Ping:
                // Request and response share the slot's payload, so the echo is already in place.
                status = 0;
                return request.Length;
            default:
                if (Service is { } service)
                {
                    return service.HandleCall(opcode, request, response, out status);
                }

                status = -1;
                return 0;
        }
    }

    private void OnParentExited(int status)
    {
        _parentExited.Set();
        Console.Error.WriteLine($"[GPUHOST][INFO] Host process {_parentProcessId} is gone; exiting.");
        Channel.MarkPeerExited();
    }
}

/// <summary>
/// What process B runs on top of <see cref="GpuProcessClient"/>: the split's GPU host. The
/// client keeps the handshake, pings and teardown; everything else comes here.
/// </summary>
public interface IGpuHostService
{
    /// <summary>
    /// One A → B record the client does not handle, on the ring thread, in ring order. The
    /// payload points into the ring and is valid only during the call. Throw
    /// <see cref="InvalidDataException"/> for a malformed record, which stops the loop.
    /// </summary>
    void HandleRecord(uint type, ReadOnlySpan<byte> payload);

    /// <summary>
    /// One A → B call the client does not handle, on one of the call threads. Request and
    /// response share the slot: read the whole request before writing the response.
    /// </summary>
    int HandleCall(uint opcode, ReadOnlySpan<byte> request, Span<byte> response, out int status);

    /// <summary>The message loop has ended (A shut down or went away).</summary>
    void Stop();
}
