// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.Versioning;
using System.Text;
using SharpEmu.GpuRemote;
using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.HLE.GuestMemory;
using SharpEmu.HLE.Host;
using SharpEmu.Host.MacOS;
using SharpEmu.Libs.Agc;
using SharpEmu.Libs.Pad;
using SharpEmu.Libs.VideoOut;
using SharpEmu.ShaderCompiler;

namespace SharpEmu.GpuSplit;

/// <summary>
/// Process A of the split (docs/macos/split-protocol.md): starts the GPU host, serves it the guest
/// memory, attaches the proxies to the guest's GPU memory manager, and applies what the GPU host
/// sends back.
/// </summary>
/// <remarks>
/// <para><b>Threads.</b> One thread drains the GPU host's ring. It applies page protections
/// itself, at once, and acknowledges them (<see cref="SplitSharedState.ProtectApplied"/>): the GPU
/// host waits on those, and the guest's write faults wait on the GPU host. Everything else it
/// reads - interrupts, flip events, input - it hands to an event thread, in order, so a slow
/// event (an event queue lock, the video-out lock) never holds up a protection.
/// <see cref="ReverseServerThreads"/> threads answer the GPU host's calls into the video-out
/// state. Guest threads write records under the transport's own ring lock
/// (<c>GpuProcessHost.SyncRoot</c>) and make the A → B calls directly.</para>
///
/// <para><b>Failure.</b> When the GPU host dies, the transport fails every pending and later
/// call promptly; this session detaches the proxies (so the memory manager stops waiting for a
/// GPU worker) and stops the guest with a clear error.</para>
/// </remarks>
[SupportedOSPlatform("macos")]
public sealed class SplitGpuSession : IDisposable
{
    public const int ReverseServerThreads = 4;

    internal const long ReadyTimeoutNanoseconds = 60_000_000_000;
    internal const long LongCallNanoseconds = 600_000_000_000;
    internal const long CallNanoseconds = 60_000_000_000;
    private const int MaxReports = 16;

    [ThreadStatic]
    private static bool t_reverseServer;

    private readonly GpuRemoteChannel _channel;
    private readonly SplitSharedState _state;
    private readonly GuestGpuMemory _memory;
    private readonly GuestViewTable _viewTable;
    private readonly ISplitGuestSide _guest;
    private readonly IDisposable? _process;
    private readonly IDisposable? _memoryServer;
    private readonly BlockingCollection<Action> _events = new();
    private readonly List<SharpEmu.HLE.Host.HostHelperThread> _threads = new();
    private readonly ManualResetEventSlim _ready = new(false);
    private int _readyStatus = GpuSplitStatus.NotReady;
    private string? _readyError;
    private volatile bool _stopping;
    private int _lost;
    private int _reports;
    private long _protectsApplied;
    private long _protectsSkipped;
    private bool _globalsInstalled;

    internal SplitGpuSession(GpuRemoteChannel channel, object outboundLock, GuestGpuMemory memory, GuestViewTable viewTable,
        ISplitGuestSide guest, IDisposable? process = null, IDisposable? memoryServer = null)
    {
        _channel = channel;
        Writer = new SplitWriter(channel, channel.Outbound, outboundLock);
        _state = new SplitSharedState(channel);
        _memory = memory;
        _viewTable = viewTable;
        _guest = guest;
        _process = process;
        _memoryServer = memoryServer;
        Relay = new RemoteGpuQueueRelay(this);
        ImageStore = new RemoteImageStore(this, channel.SyncCalls, _state);
        Backend = new RemoteGpuBackend(this);
    }

    /// <summary>The proxy <see cref="Libs.Gpu.IGuestGpuBackend"/> the guest's exports call.</summary>
    public RemoteGpuBackend Backend { get; }

    internal RemoteImageStore ImageStore { get; }

    internal RemoteGpuQueueRelay Relay { get; }

    internal SplitWriter Writer { get; }

    internal GpuRemoteChannel Channel => _channel;

    /// <summary>False once the GPU host is gone or this session is disposed.</summary>
    public bool IsAlive => Volatile.Read(ref _lost) == 0 && !_stopping && _channel.IsPeerAlive;

    /// <summary>The view log as of now: what the GPU host's replica must have replayed before it acts.</summary>
    internal ulong ViewLogSequence => _viewTable.LogSequence;

    /// <summary>The guest address space this process protects.</summary>
    internal IGuestAddressSpace GuestSpace => _memory.AddressSpace;

    /// <summary>True on the threads answering the GPU host's calls (see <see cref="RemoteImageStore"/>).</summary>
    internal static bool OnReverseServerThread => t_reverseServer;

    /// <summary>Protections applied for the GPU host, and requests over memory no longer registered here.</summary>
    internal (long Applied, long Skipped) ProtectCounts => (Interlocked.Read(ref _protectsApplied), Interlocked.Read(ref _protectsSkipped));

    /// <summary>
    /// Starts the GPU host over the guest memory of the running emulator
    /// (<see cref="GuestGpuMemoryHook.Current"/>, created shareable because
    /// <c>SHARPEMU_GPU_PROCESS=1</c> was set before the runtime started) and returns the session
    /// once the GPU host reports ready. Throws <see cref="GpuProcessException"/> otherwise.
    /// </summary>
    public static SplitGpuSession Start(HostVideoOptions? videoOptions)
    {
        var memory = GuestGpuMemoryHook.Current
            ?? throw new GpuProcessException("The GPU process needs the guest memory manager; the runtime has not created it yet.");
        if (memory.AddressSpace is not IGuestMemorySharing { GuestBackingExport: { } export, GuestViewTable: { } table })
        {
            throw new GpuProcessException(
                "The guest backing is not shareable, so the GPU process cannot map it. SHARPEMU_GPU_PROCESS=1 must be set " +
                "before the emulator starts (it makes the backing shareable), and the backing must be the shm or chunked one.");
        }

        var server = MacOSHost.ServeGuestMemory(export, table.Region, out var serviceName);
        // B is another executable in another directory (osx-arm64 beside osx-x64), so it would resolve a data
        // root of its own: its own, cold pipeline cache and shader archive (Black Myth: Wukong then compiled
        // every program again and ran at a fraction of its speed). B inherits this process's environment.
        if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable(SharpEmu.Logging.SharpEmuPaths.DataRootVariable)))
        {
            Environment.SetEnvironmentVariable(SharpEmu.Logging.SharpEmuPaths.DataRootVariable, SharpEmu.Logging.SharpEmuPaths.DataRoot);
        }

        GpuProcessHost? process = null;
        SplitGpuSession? session = null;
        try
        {
            process = GpuProcessHost.Start(new GpuProcessHostOptions { Layout = SplitSharedState.Layout });
            session = new SplitGpuSession(process.Channel, process.SyncRoot, memory, table, new GuestProcessSide(), process, server);
            session.Begin(serviceName, videoOptions, VideoOutBackendServices.GetApplicationTitleId(), installGlobals: true);
            return session;
        }
        catch (Exception exception)
        {
            // The session owns the process and the memory server once it exists.
            if (session is not null)
            {
                session.Dispose();
            }
            else
            {
                process?.Dispose();
                server.Dispose();
            }

            if (exception is GpuProcessException)
            {
                throw;
            }

            throw new GpuProcessException($"The GPU process could not be started: {exception.Message}", exception);
        }
    }

    /// <summary>
    /// Starts the threads, sends the start, waits for the GPU host to report ready, then attaches
    /// the proxies and, with <paramref name="installGlobals"/>, the process-wide observers.
    /// </summary>
    internal void Begin(string serviceName, HostVideoOptions? videoOptions, string titleId, bool installGlobals)
    {
        for (var index = 0; index < ReverseServerThreads; index++)
        {
            StartThread(ServeReverseCalls, $"SharpEmu GPU split calls {index}");
        }

        StartThread(DrainInbound, "SharpEmu GPU split inbound");
        StartThread(RunEvents, "SharpEmu GPU split events");

        var service = Encoding.UTF8.GetBytes(serviceName);
        var title = Encoding.UTF8.GetBytes(titleId);
        var start = new SplitStartMessage
        {
            ViewLogSequence = _viewTable.LogSequence,
            HasVideoOptions = videoOptions is null ? 0u : 1u,
            Video = videoOptions is null ? default : SplitConvert.ToWire(videoOptions),
            ServiceNameLength = (uint)service.Length,
            TitleIdLength = (uint)title.Length,
            HdrRequested = _guest.IsHdrOutputRequested ? 1u : 0u,
        };
        if (!Writer.Write(GpuRemoteMessage.SplitStart, in start, [.. service, .. title]))
        {
            throw new GpuProcessException("The GPU process did not take the start record.");
        }

        if (!WaitReady())
        {
            throw new GpuProcessException(
                _readyError is { Length: > 0 } error
                    ? $"The GPU process could not start: {error}"
                    : !_channel.IsPeerAlive
                        ? "The GPU process exited while it started."
                        : $"The GPU process did not report ready within {ReadyTimeoutNanoseconds / 1_000_000_000} s.");
        }

        ImageStore.PrepareForSignalContext();
        _memory.AttachStores(null, ImageStore);
        _memory.AttachGpuQueue(Relay, null);
        _memory.AttachSpanObserver(ImageStore);
        if (installGlobals)
        {
            InstallGlobals();
        }

        Console.Error.WriteLine(
            $"[GPU-SPLIT][INFO] Split GPU session up: guest memory {serviceName}, title {titleId}, view log {_viewTable.LogSequence}.");
    }

    // The shader-header table and the fused-program registry are mirrored from now on, after
    // what they hold already; a registration racing the snapshot arrives twice, which is harmless.
    private void InstallGlobals()
    {
        AgcBackendServices.ShaderHeaderObserver = (code, header) =>
            Writer.Write(GpuRemoteMessage.ShaderHeader, new ShaderHeaderMessage { CodeAddress = code, HeaderAddress = header });
        foreach (var (code, header) in AgcBackendServices.SnapshotShaderHeaders())
        {
            _ = Writer.Write(GpuRemoteMessage.ShaderHeader, new ShaderHeaderMessage { CodeAddress = code, HeaderAddress = header });
        }

        TessellationRings.Observer = SendTessellationRings;
        SendTessellationRings();

        Gen5ShaderTranslator.FusedProgramObserver = SendFusedProgram;
        if (_memory.AddressSpace is ICpuMemory guest)
        {
            foreach (var (entry, entryHeader, continuation, continuationHeader) in Gen5ShaderTranslator.SnapshotFusedPrograms(guest))
            {
                SendFusedProgram(entry, entryHeader, continuation, continuationHeader);
            }
        }

        HostWindowInput.Connect(new SplitGamepadOutput(Writer));
        _globalsInstalled = true;
    }

    private void SendFusedProgram(ulong entry, ulong entryHeader, ulong continuation, ulong continuationHeader) =>
        _ = Writer.Write(GpuRemoteMessage.FusedProgram, new FusedProgramMessage
        {
            Entry = entry,
            EntryHeader = entryHeader,
            Continuation = continuation,
            ContinuationHeader = continuationHeader,
        });

    /// <summary>Window options configured after the start.</summary>
    public void SendVideoOptions(HostVideoOptions options)
    {
        if (IsAlive)
        {
            _ = Writer.Write(GpuRemoteMessage.VideoOptions, SplitConvert.ToWire(options));
        }
    }

    private bool WaitReady()
    {
        var deadline = Environment.TickCount64 + ReadyTimeoutNanoseconds / 1_000_000;
        while (!_ready.Wait(50))
        {
            if (!_channel.IsPeerAlive || Environment.TickCount64 >= deadline)
            {
                return false;
            }
        }

        return _readyStatus == GpuSplitStatus.Ok;
    }

    // ---- A → B calls ----

    /// <summary>A call with a fixed-size answer; false when it failed (reported) or B is gone.</summary>
    internal bool Call<TRequest, TResponse>(GpuRemoteSyncOpcode opcode, in TRequest request, out TResponse response, long timeoutNanoseconds)
        where TRequest : unmanaged
        where TResponse : unmanaged
    {
        response = default;
        Span<byte> answer = stackalloc byte[Math.Max(Unsafe.SizeOf<TResponse>(), 8)];
        if (!Call(opcode, GpuSplitWire.Bytes(in request), answer, out var length, timeoutNanoseconds))
        {
            return false;
        }

        if (length < Unsafe.SizeOf<TResponse>())
        {
            Report($"The GPU process answered {opcode} with {length} bytes; {Unsafe.SizeOf<TResponse>()} were expected.");
            return false;
        }

        response = GpuSplitWire.Read<TResponse>(answer);
        return true;
    }

    /// <summary>A call whose answer is only its status.</summary>
    internal bool Call<TRequest>(GpuRemoteSyncOpcode opcode, in TRequest request, long timeoutNanoseconds)
        where TRequest : unmanaged =>
        Call(opcode, GpuSplitWire.Bytes(in request), Span<byte>.Empty, out _, timeoutNanoseconds);

    private bool Call(GpuRemoteSyncOpcode opcode, ReadOnlySpan<byte> request, Span<byte> response, out int length, long timeoutNanoseconds)
    {
        length = 0;
        if (Volatile.Read(ref _lost) != 0 || _stopping)
        {
            return false;
        }

        var result = _channel.SyncCalls.Call((uint)opcode, request, response, out length, out var status, timeoutNanoseconds);
        if (result == SyncCallStatus.Completed && status == GpuSplitStatus.Ok)
        {
            return true;
        }

        if (result != SyncCallStatus.PeerExited)
        {
            Report($"The GPU process did not complete {opcode}: {result}, status {status}.");
        }

        return false;
    }

    // ---- brackets ----

    internal bool TryBeginBracket(GpuSplitBracket kind, out ulong bracketId)
    {
        // Only an ordered video-out change comes after the accepted command streams; a mapping
        // change parks the worker at its next safe point, as the relay does in one process, and
        // does not wait for the GPU host's ring thread (which may sit in the shader prewarm).
        var barrier = kind == GpuSplitBracket.AfterPendingCommandStreams ? Writer.Written : 0;
        var request = new BracketBeginRequest { Barrier = barrier, Kind = (uint)kind };
        var begun = Call(GpuRemoteSyncOpcode.BracketBegin, request, out BracketResponse response, LongCallNanoseconds);
        bracketId = begun ? response.BracketId : 0;
        return begun && bracketId != 0;
    }

    internal void EndBracket(ulong bracketId) =>
        _ = Call(GpuRemoteSyncOpcode.BracketEnd, new BracketEndRequest { BracketId = bracketId, ViewLogSequence = ViewLogSequence },
            LongCallNanoseconds);

    // ---- B → A: the ring ----

    private void DrainInbound()
    {
        var inbound = _channel.Inbound;
        while (true)
        {
            if (!inbound.WaitForData(-1))
            {
                if (!_channel.IsPeerAlive)
                {
                    OnPeerLost();
                    return;
                }

                if (_stopping)
                {
                    return;
                }

                continue;
            }

            try
            {
                while (inbound.TryRead(out var type, out var payload))
                {
                    Take((GpuRemoteMessage)type, payload);
                    inbound.Advance();
                }
            }
            catch (InvalidDataException exception)
            {
                Console.Error.WriteLine($"[GPU-SPLIT][ERROR] The GPU process's ring is corrupt: {exception.Message}");
                OnPeerLost();
                return;
            }
        }
    }

    private void Take(GpuRemoteMessage type, ReadOnlySpan<byte> payload)
    {
        switch (type)
        {
            case GpuRemoteMessage.ProtectRequest:
                Apply(GpuSplitWire.Read<ProtectRequestMessage>(payload));
                break;

            case GpuRemoteMessage.SplitReady:
            {
                var ready = GpuSplitWire.Read<SplitReadyMessage>(payload);
                _readyError = GpuSplitWire.TextAfter<SplitReadyMessage>(payload, ready.ErrorLength);
                _readyStatus = ready.Status;
                _ready.Set();
                break;
            }

            case GpuRemoteMessage.Interrupt:
            {
                var message = GpuSplitWire.Read<InterruptMessage>(payload);
                Post(() => _guest.TriggerInterrupt(message.EventId, message.ContextId));
                break;
            }

            case GpuRemoteMessage.FlipEvent:
            {
                var message = GpuSplitWire.Read<FlipEventMessage>(payload);
                Post(() => _guest.FlipEvent((GpuSplitFlipEvent)message.Kind, message.RequestId));
                break;
            }

            case GpuRemoteMessage.VideoOutNotice:
            {
                var message = GpuSplitWire.Read<VideoOutNoticeMessage>(payload);
                var text = GpuSplitWire.TextAfter<VideoOutNoticeMessage>(payload, (int)message.TextLength);
                Post(() => _guest.Notice((GpuSplitNotice)message.Kind, text));
                break;
            }

            case GpuRemoteMessage.Input:
                TakeInput(payload);
                break;

            default:
                Report($"Ignoring record type {type} from the GPU process ({payload.Length} bytes).");
                break;
        }
    }

    private unsafe void TakeInput(ReadOnlySpan<byte> payload)
    {
        var message = GpuSplitWire.Read<InputMessage>(payload);
        var keys = new int[Math.Min(message.KeyCount, InputMessage.MaxKeys)];
        for (var index = 0; index < keys.Length; index++)
        {
            keys[index] = message.Keys[index];
        }

        var name = message.NameLength == 0
            ? null
            : Encoding.UTF8.GetString(new ReadOnlySpan<byte>(message.Name, (int)Math.Min(message.NameLength, InputMessage.MaxNameBytes)));
        var focused = message.Focused != 0;
        var gamepad = SplitConvert.FromWire(message.Gamepad);
        Post(() => _guest.Input(focused, keys, gamepad, name));
    }

    // On the inbound thread, in request order. The GPU host only asks for memory it mirrors from
    // this process's spans; one no longer registered here (unmapped meanwhile) is left alone.
    private void Apply(in ProtectRequestMessage request)
    {
        try
        {
            var space = _memory.AddressSpace;
            if (!_memory.Covers(request.Address, request.Size))
            {
                Interlocked.Increment(ref _protectsSkipped);
            }
            else
            {
                var applied = (request.Flags & ProtectRequestFlags.Derived) != 0 && space is IGuestDerivedProtectionAddressSpace derived
                    ? derived.TryApplyDerivedProtection(request.Address, request.Size, (GuestPageProtection)request.GuestAccess,
                        (GuestPageProtection)request.Access, (request.Flags & ProtectRequestFlags.OrdinaryEnrollment) != 0)
                    : space.TryProtect(request.Address, request.Size, (GuestPageProtection)request.Access);
                if (applied)
                {
                    Interlocked.Increment(ref _protectsApplied);
                }
                else
                {
                    Report($"Page protection #{request.Sequence} (0x{request.Address:X16}+0x{request.Size:X} -> " +
                        $"{(GuestPageProtection)request.Access}) could not be applied.");
                }
            }
        }
        catch (Exception exception)
        {
            // The runtime tearing the guest memory down under a late request, or a protection
            // that threw: either way it is acknowledged, or the GPU host would wait it out.
            Interlocked.Increment(ref _protectsSkipped);
            Report($"Page protection #{request.Sequence} failed: {exception.Message}");
        }

        _state.ProtectApplied.Publish(request.Sequence);
    }

    private void Post(Action work)
    {
        try
        {
            _events.Add(work);
        }
        catch (InvalidOperationException)
        {
            // Stopping: the guest no longer takes events.
        }
    }

    private void RunEvents()
    {
        foreach (var work in _events.GetConsumingEnumerable())
        {
            try
            {
                work();
            }
            catch (Exception exception)
            {
                Console.Error.WriteLine($"[GPU-SPLIT][ERROR] An event from the GPU process failed: {exception}");
            }
        }
    }

    // ---- B → A: the calls ----

    private void ServeReverseCalls()
    {
        t_reverseServer = true;
        var calls = _channel.ReverseSyncCalls;
        SyncCallHandler handler = HandleReverseCall;
        while (!_stopping)
        {
            if (calls.WaitForRequests(-1))
            {
                while (!_stopping && calls.ServeOne(handler))
                {
                }
            }
            else if (!_channel.IsPeerAlive)
            {
                return;
            }
        }
    }

    // Request and response share the slot: every request is read whole before the answer is written.
    internal int HandleReverseCall(uint opcode, ReadOnlySpan<byte> request, Span<byte> response, out int status)
    {
        status = GpuSplitStatus.Ok;
        try
        {
            switch ((GpuRemoteReverseOpcode)opcode)
            {
                case GpuRemoteReverseOpcode.TryReserveFlip:
                {
                    var call = GpuSplitWire.Read<ReserveFlipRequest>(request);
                    var result = _guest.TryReserveFlipRequest(call.Handle, call.BufferIndex, call.FlipMode, call.FlipArgument, out var requestId);
                    return GpuSplitWire.Write(response, new ReserveFlipResponse
                    {
                        Result = result,
                        RequestId = requestId,
                        HdrRequested = _guest.IsHdrOutputRequested ? 1u : 0u,
                    });
                }

                case GpuRemoteReverseOpcode.IsFlipDone:
                {
                    var call = GpuSplitWire.Read<FlipSlotRequest>(request);
                    return Value(response, _guest.IsFlipDone(call.Handle, call.BufferIndex) ? 1 : 0);
                }

                case GpuRemoteReverseOpcode.CanPresentFlip:
                {
                    var call = GpuSplitWire.Read<FlipTimingRequest>(request);
                    return Value(response, _guest.CanPresentFlip(call.RequestId, call.Timestamp) ? 1 : 0);
                }

                case GpuRemoteReverseOpcode.NextFlipPresentTimestamp:
                {
                    var call = GpuSplitWire.Read<FlipTimingRequest>(request);
                    return Value(response, _guest.NextFlipPresentTimestamp(call.RequestId, call.Timestamp));
                }

                case GpuRemoteReverseOpcode.TryGetDisplayBuffer:
                {
                    var call = GpuSplitWire.Read<FlipSlotRequest>(request);
                    var answer = _guest.TryGetDisplayBufferInfo(call.Handle, call.BufferIndex, out var info)
                        ? SplitConvert.ToWire(info)
                        : default;
                    return GpuSplitWire.Write(response, answer);
                }

                case GpuRemoteReverseOpcode.GetFlipEventCount:
                    return Value(response, _guest.GetFlipEventCount(GpuSplitWire.Read<FlipIdRequest>(request).RequestId));

                case GpuRemoteReverseOpcode.ReleaseNoBufferFlip:
                {
                    var call = GpuSplitWire.Read<ReleaseNoBufferRequest>(request);
                    return Value(response, _guest.ReleaseNoBufferFlip(call.BufferIndex, call.RequestId) ? 1 : 0);
                }

                case GpuRemoteReverseOpcode.QueryString:
                {
                    var which = (GpuSplitText)GpuSplitWire.Read<QueryRequest>(request).Which;
                    var text = which == GpuSplitText.WindowTitle ? _guest.GetWindowTitle() : _guest.GetApplicationTitleId();
                    var bytes = Encoding.UTF8.GetBytes(text);
                    var length = Math.Min(bytes.Length, response.Length);
                    bytes.AsSpan(0, length).CopyTo(response);
                    return length;
                }

                case GpuRemoteReverseOpcode.FlipEvent:
                {
                    var call = GpuSplitWire.Read<FlipEventMessage>(request);
                    _guest.FlipEvent((GpuSplitFlipEvent)call.Kind, call.RequestId);
                    return Value(response, 0);
                }

                case GpuRemoteReverseOpcode.IsFlipPresentationPending:
                    return Value(response, _guest.IsFlipPresentationPending(GpuSplitWire.Read<FlipIdRequest>(request).RequestId) ? 1 : 0);

                case GpuRemoteReverseOpcode.IsHdrOutputRequested:
                    return Value(response, _guest.IsHdrOutputRequested ? 1 : 0);

                case GpuRemoteReverseOpcode.TriggerInterrupt:
                {
                    var call = GpuSplitWire.Read<InterruptMessage>(request);
                    _guest.TriggerInterrupt(call.EventId, call.ContextId);
                    return Value(response, 0);
                }

                default:
                    status = GpuSplitStatus.UnknownOpcode;
                    return 0;
            }
        }
        catch (InvalidDataException exception)
        {
            Report($"Malformed call {opcode} from the GPU process: {exception.Message}");
            status = GpuSplitStatus.Malformed;
            return 0;
        }
    }

    private static int Value(Span<byte> response, long value) => GpuSplitWire.Write(response, new ValueResponse { Value = value });

    // ---- failure and teardown ----

    private void OnPeerLost()
    {
        if (Interlocked.Exchange(ref _lost, 1) != 0)
        {
            return;
        }

        _ready.Set();
        if (_stopping)
        {
            return;
        }

        Detach();
        RestoreGuestProtections();
        _guest.OnGpuProcessLost("The GPU process exited unexpectedly.");
    }

    // The dead GPU host's watches stay applied here, and a guest store into one of those pages
    // would now be a fault nothing resolves: this process's guard (which holds no watches) gives
    // every span its plain guest protection back before the guest is stopped.
    private void RestoreGuestProtections()
    {
        try
        {
            _memory.ForEachSpan((address, size) => _memory.Pages.Reapply(address, size));
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"[GPU-SPLIT][ERROR] The guest protections could not be restored: {exception.Message}");
        }
    }

    // The memory manager waits for a detached relay before it changes mappings on its own, so a
    // relay that can no longer reach the GPU host must come off.
    // The tessellation memory the guest driver set: the GPU process draws tessellated draws through it.
    private void SendTessellationRings() =>
        _ = Writer.Write(GpuRemoteMessage.TessellationRings, new TessellationRingsMessage
        {
            FactorRing = TessellationRings.TessFactorRing,
            FactorRingSize = TessellationRings.TessFactorRingSize,
            OffchipBuffer = TessellationRings.OffchipBuffer,
            OffchipParameter = TessellationRings.OffchipParameter,
        });

    private void Detach()
    {
        _memory.AttachSpanObserver(null);
        _memory.AttachGpuQueue(null, null);
        _memory.AttachStores(null, null);
        if (_globalsInstalled)
        {
            AgcBackendServices.ShaderHeaderObserver = null;
            TessellationRings.Observer = null;
            Gen5ShaderTranslator.FusedProgramObserver = null;
        }
    }

    internal void Report(string message)
    {
        if (Interlocked.Increment(ref _reports) <= MaxReports)
        {
            Console.Error.WriteLine($"[GPU-SPLIT][WARN] {message}");
        }
    }

    // The inbound thread applies the protections the GPU worker waits for, the call servers answer
    // its reads, and the events thread delivers its interrupts to the guest: all at the worker's
    // class, which a thread CoreCLR creates cannot take later.
    private void StartThread(Action start, string name) =>
        _threads.Add(SharpEmu.HLE.Host.HostHelperThread.Start(name, 1024 * 1024, start));

    public void Dispose()
    {
        if (_stopping)
        {
            return;
        }

        _stopping = true;
        if (Volatile.Read(ref _lost) == 0)
        {
            Detach();
        }

        Writer.Close();
        _events.CompleteAdding();
        _channel.WakeAll();
        // The process's teardown unmaps the region: these threads must be off it first.
        foreach (var thread in _threads)
        {
            if (!thread.IsCurrentThread)
            {
                _ = thread.Join(TimeSpan.FromSeconds(2));
            }
        }

        _process?.Dispose();
        _memoryServer?.Dispose();
    }
}
