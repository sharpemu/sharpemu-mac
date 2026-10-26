// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.Versioning;
using System.Text;
using SharpEmu.GpuRemote;
using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.HLE.Host;
using SharpEmu.Libs.Agc;
using SharpEmu.Libs.Gpu;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.Libs.Pad;
using SharpEmu.Libs.VideoOut;
using SharpEmu.ShaderCompiler;

namespace SharpEmu.GpuSplit;

/// <summary>What the GPU process's composition root gives <see cref="SplitGpuHost"/>.</summary>
[SupportedOSPlatform("macos")]
public sealed record SplitGpuHostOptions
{
    /// <summary>Maps the guest memory the guest process serves under a service name.</summary>
    public Func<string, SplitGuestMemory> ConnectGuestMemory { get; init; } = static name => SplitGuestMemory.Connect(name);

    /// <summary>
    /// Composes the real backend once guest memory is in place (the Metal backend, its window
    /// options from <see cref="SplitGpuEnvironment.VideoOptions"/>) and returns it.
    /// </summary>
    public required Func<SplitGpuEnvironment, IGuestGpuBackend> ComposeBackend { get; init; }

    /// <summary>Applies window options the guest process sends after the start.</summary>
    public Action<HostVideoOptions>? ConfigureVideo { get; init; }

    /// <summary>Answers whether the presenter drives HDR output.</summary>
    public Func<bool>? QueryHdrOutputSupport { get; init; }

    /// <summary>
    /// Installs the process-wide seams: <see cref="GuestGpuMemoryHook"/>,
    /// <see cref="VideoOutBackendServices.Bridge"/> and <see cref="EndOfPipeEvents.Redirect"/>.
    /// Off only in tests that run both processes' halves in one.
    /// </summary>
    public bool InstallGlobals { get; init; } = true;

    /// <summary>How often the window's input (<see cref="HostWindowInput"/>) is checked for changes; zero sends none.</summary>
    public TimeSpan InputPollInterval { get; init; } = TimeSpan.FromMilliseconds(4);
}

/// <summary>What the backend is composed over in the GPU process.</summary>
public sealed record SplitGpuEnvironment(GuestGpuMemory Memory, ICpuMemory AddressSpace, HostVideoOptions? VideoOptions);

/// <summary>
/// Process B of the split (docs/macos/split-protocol.md): turns what the guest process sends into
/// calls of the real GPU backend, and the backend's calls into the guest's state into messages
/// back. Installed as the <see cref="GpuProcessClient.Service"/> of the GPU host.
/// </summary>
/// <remarks>
/// <para><b>Start.</b> The guest process's first split record (<see cref="GpuRemoteMessage.SplitStart"/>)
/// names its guest memory; this connects it (<see cref="SplitGuestMemory"/>), builds the address
/// space and the GPU memory manager over it, installs the seams, composes the backend and answers
/// <see cref="GpuRemoteMessage.SplitReady"/>.</para>
///
/// <para><b>Threads and the rule that keeps them apart.</b> Records arrive on the client's ring
/// thread in ring order; calls on its call threads. The ring thread never waits for the guest
/// process: a guest thread holding the video-out lock may be blocked writing into a full ring,
/// and a call into the guest from here would need that lock. So what the ring thread runs that
/// would ask the guest - the backend's title id, HDR state and GPU name - is answered from copies
/// or sent one way (<see cref="RemoteVideoOutBridge"/>), and a flip refused here is reported as a
/// record. Calls that must come after the guest's earlier records (a frame boundary, a bracket, a
/// managed write) wait until the ring thread has handled as many records as the guest had
/// written; the guest's write faults never wait for the ring.</para>
///
/// <para><b>Page protections.</b> Every protection the page guard derives is sent to the guest
/// process, numbered (<see cref="IProtectionRequests"/>), and acknowledged through
/// <see cref="SplitSharedState.ProtectApplied"/>. Every call that changes watches waits for its
/// protections before it answers - a fault waits for the ones that let its retry pass, a span
/// change for the ones over memory the guest is about to unmap or remap - so the guest never acts
/// on a page whose protection is still on its way.</para>
/// </remarks>
[SupportedOSPlatform("macos")]
public sealed class SplitGpuHost : IGpuHostService, IProtectionRequests, IDisposable
{
    private const long ProtectTimeoutNanoseconds = 30_000_000_000;
    // A reverse call is never retried (a flip reservation must not happen twice) and an abandoned
    // one is dropped unserved, so it waits as long as the guest's own long calls do.
    private const long ReverseTimeoutNanoseconds = 600_000_000_000;

    // Under the guest process's 10-minute call timeout, so B answers (or gives up) first and a
    // bracket or unregistration is never left behind a call the guest abandoned.
    private static readonly TimeSpan LongWait = TimeSpan.FromMinutes(9);
    private static readonly TimeSpan BracketWorkWait = TimeSpan.FromMinutes(9);
    private const int MaxReports = 16;

    [ThreadStatic]
    private static bool t_onRingThread;

    private readonly GpuRemoteChannel _channel;
    private readonly SplitWriter _writer;
    private readonly SplitGpuHostOptions _options;
    private SplitSharedState? _state;
    private readonly object _barrierGate = new();
    private readonly ConcurrentDictionary<ulong, WorkerBracket> _brackets = new();
    // Frame boundaries the ring thread queued, by A's ticket, until A's wait call takes them.
    private readonly ConcurrentDictionary<ulong, (IdleOutcome Outcome, Func<IdleOutcome>? Wait)> _frameBoundaries = new();
    private SplitGuestMemory? _guestMemory;
    private RemoteGuestAddressSpace? _space;
    private GuestGpuMemory? _memory;
    private IGuestGpuBackend? _backend;
    private RemoteVideoOutBridge? _bridge;
    private volatile bool _ready;
    private volatile bool _stopping;
    private ulong _handled;
    private int _barrierWaiters;
    private ulong _protectSequence;
    private long _nextBracket;
    private int _reports;
    private Thread? _inputThread;
    private byte[]? _frame;
    private uint _frameWidth;
    private uint _frameHeight;
    private uint _frameReceived;
    private bool _globalsInstalled;

    public SplitGpuHost(GpuRemoteChannel channel, object outboundLock, SplitGpuHostOptions options)
    {
        ArgumentNullException.ThrowIfNull(channel);
        ArgumentNullException.ThrowIfNull(options);
        _channel = channel;
        _writer = new SplitWriter(channel, channel.Outbound, outboundLock);
        _options = options;
    }

    /// <summary>True once the start succeeded and the backend is composed.</summary>
    public bool IsReady => _ready;

    /// <summary>Split records handled so far.</summary>
    public ulong HandledRecords => Volatile.Read(ref _handled);

    public GuestGpuMemory? Memory => _memory;

    internal RemoteGuestAddressSpace? AddressSpace => _space;

    internal RemoteVideoOutBridge? VideoOut => _bridge;

    internal SplitGuestMemory? GuestMemory => _guestMemory;

    /// <summary>True on the thread that drains the guest process's ring (see the type remarks).</summary>
    internal static bool OnRingThread => t_onRingThread;

    // ---- records (ring thread) ----

    public void HandleRecord(uint type, ReadOnlySpan<byte> payload)
    {
        t_onRingThread = true;
        try
        {
            Dispatch((GpuRemoteMessage)type, payload);
        }
        finally
        {
            Interlocked.Increment(ref _handled);
            if (Volatile.Read(ref _barrierWaiters) != 0)
            {
                lock (_barrierGate)
                {
                    Monitor.PulseAll(_barrierGate);
                }
            }
        }
    }

    private void Dispatch(GpuRemoteMessage type, ReadOnlySpan<byte> payload)
    {
        if (type == GpuRemoteMessage.SplitStart)
        {
            Start(payload);
            return;
        }

        if (!_ready)
        {
            Report($"A {type} record arrived before the GPU process was ready; it is dropped.");
            return;
        }

        var backend = _backend!;
        switch (type)
        {
            case GpuRemoteMessage.SubmitCommandStream:
            {
                var message = GpuSplitWire.Read<SubmitCommandStreamMessage>(payload);
                _ = _guestMemory!.Replica.Sync(message.ViewLogSequence);
                backend.SubmitCommandStream(_space!, message.Queue, message.Address, message.DwordCount, message.SubmissionId, null);
                break;
            }

            case GpuRemoteMessage.FlipRequest:
            {
                var message = GpuSplitWire.Read<FlipRequestMessage>(payload);
                _bridge!.NoteHdrRequested(message.HdrRequested != 0);
                if (!backend.TrySubmitGuestImage(
                        message.VideoOutHandle, message.DisplayBufferIndex, message.Address, message.Width, message.Height,
                        message.PitchInPixel, message.FlipRequestId))
                {
                    SendFlipEvent(GpuSplitFlipEvent.Refused, message.FlipRequestId);
                }

                break;
            }

            case GpuRemoteMessage.CpuWrite:
            {
                // By the time a record is read the write may have landed: nothing may be put under it.
                var message = GpuSplitWire.Read<CpuWriteMessage>(payload);
                _memory!.MarkCpuWriteLanded(message.Address, message.Size);
                break;
            }

            case GpuRemoteMessage.ShaderHeader:
            {
                var message = GpuSplitWire.Read<ShaderHeaderMessage>(payload);
                AgcBackendServices.MirrorShaderHeader(message.CodeAddress, message.HeaderAddress);
                break;
            }

            case GpuRemoteMessage.TessellationRings:
            {
                var message = GpuSplitWire.Read<TessellationRingsMessage>(payload);
                TessellationRings.TessFactorRing = message.FactorRing;
                TessellationRings.TessFactorRingSize = message.FactorRingSize;
                TessellationRings.OffchipBuffer = message.OffchipBuffer;
                TessellationRings.OffchipParameter = message.OffchipParameter;
                break;
            }

            case GpuRemoteMessage.FusedProgram:
            {
                var message = GpuSplitWire.Read<FusedProgramMessage>(payload);
                Gen5ShaderTranslator.MirrorFusedProgram(
                    _space!, message.Entry, message.EntryHeader, message.Continuation, message.ContinuationHeader);
                break;
            }

            case GpuRemoteMessage.Control:
                RunControl(backend, GpuSplitWire.Read<ControlMessage>(payload));
                break;

            case GpuRemoteMessage.HostFrame:
                TakeHostFrame(backend, payload);
                break;

            case GpuRemoteMessage.GamepadOutput:
                ApplyGamepadOutput(GpuSplitWire.Read<GamepadOutputMessage>(payload));
                break;

            case GpuRemoteMessage.VideoOptions:
                _options.ConfigureVideo?.Invoke(SplitConvert.FromWire(GpuSplitWire.Read<VideoOptionsWire>(payload)));
                break;

            case GpuRemoteMessage.FrameBoundary:
            {
                // In ring order, behind the submissions A sent before it; A's wait call collects it.
                // A session shutting down leaves the whole SubmitDone (its worker teardown) to the call thread.
                var message = GpuSplitWire.Read<FrameBoundaryMessage>(payload);
                var space = _space!;
                Func<IdleOutcome>? wait = () => backend.SubmitDone(space);
                var outcome = HostSessionControl.IsShutdownRequested
                    ? IdleOutcome.Completed
                    : backend.BeginSubmitDone(space, out wait);
                _frameBoundaries[message.Ticket] = (outcome, wait);
                break;
            }

            default:
                Report($"Ignoring split record type {type} ({payload.Length} bytes).");
                break;
        }
    }

    private void Start(ReadOnlySpan<byte> payload)
    {
        if (_ready || _guestMemory is not null)
        {
            Report("A second split start was ignored.");
            return;
        }

        try
        {
            var start = GpuSplitWire.Read<SplitStartMessage>(payload);
            var textStart = Unsafe.SizeOf<SplitStartMessage>();
            if (payload.Length < textStart + start.ServiceNameLength + start.TitleIdLength)
            {
                throw new InvalidDataException("The split start is shorter than the names it announces.");
            }

            var serviceName = Encoding.UTF8.GetString(payload.Slice(textStart, (int)start.ServiceNameLength));
            var titleId = Encoding.UTF8.GetString(payload.Slice(textStart + (int)start.ServiceNameLength, (int)start.TitleIdLength));
            var video = start.HasVideoOptions != 0 ? SplitConvert.FromWire(start.Video) : null;

            // A region made without the split's block (a plain GPU host start) cannot carry it.
            _state = new SplitSharedState(_channel);
            _guestMemory = _options.ConnectGuestMemory(serviceName);
            _ = _guestMemory.Replica.Sync(start.ViewLogSequence);
            _space = new RemoteGuestAddressSpace(
                _guestMemory.Mirror, _guestMemory.Translator, _guestMemory.Replica, _guestMemory.ChunkBytes,
                _guestMemory.ObjectBytes, this, _state);
            _memory = new GuestGpuMemory(_space);
            _bridge = new RemoteVideoOutBridge(this, titleId, start.HdrRequested != 0);
            if (_options.InstallGlobals)
            {
                GuestGpuMemoryHook.Attach(_memory);
                VideoOutBackendServices.Bridge = _bridge;
                EndOfPipeEvents.Redirect = new InterruptSink(this);
                // The overlay counts guest flip submissions, which happen in the guest process;
                // this one presents every guest frame, so it counts those instead (else 0 fps).
                PerfOverlay.UsePresentedGuestFrameTiming();
                _globalsInstalled = true;
            }

            _backend = _options.ComposeBackend(new SplitGpuEnvironment(_memory, _space, video));
            _ready = true;
            StartInputPoll();
            StartReplicaFollower();
            Console.Error.WriteLine(
                $"[GPU-SPLIT][INFO] GPU process ready: {_backend.BackendName}, title {titleId}, guest memory {serviceName}.");
            SendReady(GpuSplitStatus.Ok, null);
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"[GPU-SPLIT][ERROR] The GPU process could not start: {exception}");
            SendReady(GpuSplitStatus.Failed, exception.Message);
        }
    }

    private void SendReady(int status, string? error)
    {
        var ready = new SplitReadyMessage { Status = status, ReplicaBase = _guestMemory?.Replica.Base ?? 0 };
        var text = Encoding.UTF8.GetBytes(error ?? string.Empty);
        ready.ErrorLength = Math.Min(text.Length, 1024);
        _ = _writer.Write(GpuRemoteMessage.SplitReady, in ready, text.AsSpan(0, ready.ErrorLength));
    }

    private static void RunControl(IGuestGpuBackend backend, in ControlMessage control)
    {
        switch ((GpuSplitControl)control.Kind)
        {
            case GpuSplitControl.EnsureStarted:
                backend.EnsureStarted(control.Width, control.Height);
                break;
            case GpuSplitControl.HideSplashScreen:
                backend.HideSplashScreen();
                break;
            case GpuSplitControl.RequestClose:
                // The guest process is leaving: the worker's teardown and the window follow, off
                // the ring thread (the teardown joins the worker, which may be in a call to the guest).
                HostSessionControl.RequestShutdown("the guest process closed the GPU backend");
                new Thread(backend.RequestClose) { IsBackground = true, Name = "SharpEmu GPU split close" }.Start();
                break;
            case GpuSplitControl.CountShaderCompilation:
                backend.CountShaderCompilation();
                break;
        }
    }

    // A frame arrives in pieces of at most one record each, written back to back.
    private void TakeHostFrame(IGuestGpuBackend backend, ReadOnlySpan<byte> payload)
    {
        var header = GpuSplitWire.Read<HostFrameMessage>(payload);
        var piece = payload[Unsafe.SizeOf<HostFrameMessage>()..];
        if (header.Offset == 0 || _frame is null || _frame.Length != header.TotalBytes ||
            _frameWidth != header.Width || _frameHeight != header.Height || _frameReceived != header.Offset)
        {
            if (header.Offset != 0)
            {
                Report("A host frame piece arrived out of order; the frame is dropped.");
                _frame = null;
                return;
            }

            _frame = new byte[header.TotalBytes];
            _frameWidth = header.Width;
            _frameHeight = header.Height;
            _frameReceived = 0;
        }

        if ((ulong)header.Offset + (ulong)piece.Length > (ulong)_frame.Length)
        {
            throw new InvalidDataException("A host frame piece reaches past its frame.");
        }

        piece.CopyTo(_frame.AsSpan((int)header.Offset));
        _frameReceived = header.Offset + (uint)piece.Length;
        if (_frameReceived == header.TotalBytes)
        {
            var frame = _frame;
            _frame = null;
            backend.Submit(frame, header.Width, header.Height);
        }
    }

    private static void ApplyGamepadOutput(in GamepadOutputMessage message)
    {
        if (HostWindowInputSource.Current is not { } input)
        {
            return;
        }

        switch ((GpuSplitGamepadOutput)message.Kind)
        {
            case GpuSplitGamepadOutput.Rumble:
                input.SetRumble(message.A, message.B);
                break;
            case GpuSplitGamepadOutput.TriggerRumble:
                input.SetTriggerRumble((message.Flags & 1) != 0 ? message.A : null, (message.Flags & 2) != 0 ? message.B : null);
                break;
            case GpuSplitGamepadOutput.AdaptiveTriggers:
                input.SetAdaptiveTriggerEffect(SplitConvert.FromWire(message.Left), SplitConvert.FromWire(message.Right));
                break;
            case GpuSplitGamepadOutput.Lightbar:
                input.SetLightbar(message.A, message.B, message.C);
                break;
            case GpuSplitGamepadOutput.ResetLightbar:
                input.ResetLightbar();
                break;
        }
    }

    // ---- calls (call threads) ----

    public int HandleCall(uint opcode, ReadOnlySpan<byte> request, Span<byte> response, out int status)
    {
        status = GpuSplitStatus.Ok;
        if (!_ready)
        {
            status = GpuSplitStatus.NotReady;
            return 0;
        }

        try
        {
            switch ((GpuRemoteSyncOpcode)opcode)
            {
                case GpuRemoteSyncOpcode.SubmitDone:
                {
                    var call = GpuSplitWire.Read<BarrierRequest>(request);
                    if (!WaitForRecords(call.Barrier, LongWait))
                    {
                        status = GpuSplitStatus.Failed;
                        return 0;
                    }

                    var outcome = _backend!.SubmitDone(_space!);
                    return GpuSplitWire.Write(response, new IdleOutcomeResponse { Outcome = (int)outcome });
                }

                case GpuRemoteSyncOpcode.WaitFrameBoundary:
                {
                    var call = GpuSplitWire.Read<FrameBoundaryWaitRequest>(request);
                    if (!WaitForRecords(call.Barrier, LongWait) ||
                        !_frameBoundaries.TryRemove(call.Ticket, out var boundary))
                    {
                        status = GpuSplitStatus.Failed;
                        return 0;
                    }

                    var outcome = boundary.Outcome == IdleOutcome.Completed && boundary.Wait is { } wait
                        ? wait()
                        : boundary.Outcome;
                    return GpuSplitWire.Write(response, new IdleOutcomeResponse { Outcome = (int)outcome });
                }

                case GpuRemoteSyncOpcode.BracketBegin:
                    return BeginBracket(GpuSplitWire.Read<BracketBeginRequest>(request), response, out status);

                case GpuRemoteSyncOpcode.BracketEnd:
                {
                    var call = GpuSplitWire.Read<BracketEndRequest>(request);
                    if (!_brackets.TryRemove(call.BracketId, out var bracket))
                    {
                        status = GpuSplitStatus.Failed;
                        return 0;
                    }

                    var replica = _guestMemory!.Replica;
                    if (!bracket.End(() => replica.Sync(call.ViewLogSequence), LongWait))
                    {
                        Report($"GPU worker bracket {call.BracketId} did not end in time.");
                    }

                    _ = WaitApplied(LastRequested);
                    return 0;
                }

                case GpuRemoteSyncOpcode.Fault:
                {
                    var call = GpuSplitWire.Read<FaultRequest>(request);
                    var memory = _memory!;
                    var handled = call.Kind == (uint)FaultKind.Write
                        ? memory.Images?.MarkCpuWrite(call.Address, call.Size) ?? false
                        : memory.Buffers?.DownloadToCpu(call.Address, call.Size) ?? false;
                    // The store may have dropped watches: the guest retries the access as soon as
                    // this answers, so their protections must be in place first.
                    _ = WaitApplied(LastRequested);
                    // No store held the page: its watch may have gone between the fault and this
                    // call. Its restoration version lets the guest retry the store once
                    // (GuestGpuMemory.TryRetryReleasedWrite does this from the guard in one process).
                    var restoration = !handled && call.Kind == (uint)FaultKind.Write
                        ? memory.Pages.GetWriteRestorationVersion(call.Address)
                        : 0;
                    return GpuSplitWire.Write(response, new HandledResponse { Handled = handled ? 1u : 0u, RestorationVersion = restoration });
                }

                case GpuRemoteSyncOpcode.CpuWrite:
                {
                    var call = GpuSplitWire.Read<CpuWriteRequest>(request);
                    if (!WaitForRecords(call.Barrier, LongWait))
                    {
                        status = GpuSplitStatus.Failed;
                        return 0;
                    }

                    var handled = _memory!.MarkCpuWrite(call.Address, call.Size);
                    return GpuSplitWire.Write(response, new HandledResponse { Handled = handled ? 1u : 0u });
                }

                case GpuRemoteSyncOpcode.SpanRegister:
                {
                    var call = GpuSplitWire.Read<SpanRequest>(request);
                    _memory!.Register(call.Address, call.Size, (GuestPageProtection)call.Protection);
                    // The guest applies no protection of its own for the span (IGuestSpanObserver):
                    // when ours did not land, it falls back to the plain guest protection.
                    if (!WaitApplied(LastRequested))
                    {
                        status = GpuSplitStatus.Failed;
                    }

                    return 0;
                }

                case GpuRemoteSyncOpcode.SpanProtect:
                {
                    var call = GpuSplitWire.Read<SpanRequest>(request);
                    var handled = _memory!.NoteProtected(call.Address, call.Size, (GuestPageProtection)call.Protection);
                    if (!WaitApplied(LastRequested))
                    {
                        status = GpuSplitStatus.Failed;
                    }

                    return GpuSplitWire.Write(response, new HandledResponse { Handled = handled ? 1u : 0u });
                }

                case GpuRemoteSyncOpcode.SpanUnregister:
                {
                    var call = GpuSplitWire.Read<SpanRequest>(request);
                    var memory = _memory!;
                    // Inside a bracket the worker is parked in it: the unregistration runs there, as it
                    // runs on the worker in one process.
                    if (call.BracketId == 0 || !_brackets.TryGetValue(call.BracketId, out var bracket) ||
                        !bracket.TryRun(() => memory.Unregister(call.Address, call.Size), BracketWorkWait))
                    {
                        memory.Unregister(call.Address, call.Size);
                    }

                    _ = WaitApplied(LastRequested);
                    return 0;
                }

                case GpuRemoteSyncOpcode.Query:
                    return Query(GpuSplitWire.Read<QueryRequest>(request), response);

                default:
                    status = GpuSplitStatus.UnknownOpcode;
                    return 0;
            }
        }
        catch (InvalidDataException exception)
        {
            Report($"Malformed call {opcode}: {exception.Message}");
            status = GpuSplitStatus.Malformed;
            return 0;
        }
    }

    private int BeginBracket(in BracketBeginRequest call, Span<byte> response, out int status)
    {
        status = GpuSplitStatus.Ok;
        if (!WaitForRecords(call.Barrier, LongWait))
        {
            status = GpuSplitStatus.Failed;
            return 0;
        }

        var id = (ulong)Interlocked.Increment(ref _nextBracket);
        var bracket = new WorkerBracket(id, () => !_stopping && _channel.IsPeerAlive);
        _brackets[id] = bracket;
        var memory = _memory!;
        var backend = _backend!;
        bracket.Start((GpuSplitBracket)call.Kind == GpuSplitBracket.AfterPendingCommandStreams
            ? park => backend.RunAfterPendingCommandStreams(park)
            : park => memory.RunMappingChange(park));
        if (!bracket.WaitParked(LongWait))
        {
            _ = _brackets.TryRemove(id, out _);
            _ = bracket.End(static () => { }, TimeSpan.Zero);
            Report($"The GPU worker did not park for bracket {id}.");
            status = GpuSplitStatus.Failed;
            return 0;
        }

        return GpuSplitWire.Write(response, new BracketResponse { BracketId = id });
    }

    private int Query(in QueryRequest call, Span<byte> response)
    {
        var answer = new QueryResponse();
        switch ((GpuSplitQuery)call.Which)
        {
            case GpuSplitQuery.PerfCounters:
                var (draws, drawMs, pipelines, compilations) = _backend!.ReadAndResetPerfCounters();
                answer.A = draws;
                answer.B = pipelines;
                answer.C = compilations;
                answer.D = drawMs;
                break;
            case GpuSplitQuery.HdrOutputSupport:
                answer.A = _options.QueryHdrOutputSupport?.Invoke() == true ? 1 : 0;
                break;
            case GpuSplitQuery.MeshPipelines:
                answer.A = _backend!.SupportsPrimitiveShaderMeshPipelines ? 1 : 0;
                break;
        }

        return GpuSplitWire.Write(response, answer);
    }

    // Waits until the ring thread has handled as many split records as the guest had written.
    private bool WaitForRecords(ulong barrier, TimeSpan timeout)
    {
        if (Volatile.Read(ref _handled) >= barrier)
        {
            return true;
        }

        var deadline = Environment.TickCount64 + (long)timeout.TotalMilliseconds;
        Interlocked.Increment(ref _barrierWaiters);
        try
        {
            lock (_barrierGate)
            {
                while (Volatile.Read(ref _handled) < barrier)
                {
                    if (_stopping || !_channel.IsPeerAlive || Environment.TickCount64 >= deadline)
                    {
                        Report($"Gave up waiting for the guest's record {barrier} (handled {Volatile.Read(ref _handled)}).");
                        return false;
                    }

                    Monitor.Wait(_barrierGate, 50);
                }
            }

            return true;
        }
        finally
        {
            Interlocked.Decrement(ref _barrierWaiters);
        }
    }

    // ---- page protections ----

    public ulong Request(ulong address, ulong size, GuestPageProtection guestAccess, GuestPageProtection access, ProtectRequestFlags flags)
    {
        var message = new ProtectRequestMessage
        {
            Address = address,
            Size = size,
            GuestAccess = (uint)guestAccess,
            Access = (uint)access,
            Flags = flags,
        };
        lock (_writer.Gate)
        {
            // Numbered under the ring's lock, so numbers and ring order agree.
            var sequence = _protectSequence + 1;
            message.Sequence = sequence;
            if (!_writer.WriteLocked(GpuRemoteMessage.ProtectRequest, GpuSplitWire.Bytes(in message)))
            {
                return 0;
            }

            Volatile.Write(ref _protectSequence, sequence);
            return sequence;
        }
    }

    public ulong LastRequested => Volatile.Read(ref _protectSequence);

    public bool WaitApplied(ulong sequence)
    {
        if (sequence == 0 || _state!.ProtectApplied.WaitAtLeast(sequence, ProtectTimeoutNanoseconds))
        {
            return true;
        }

        if (_channel.IsPeerAlive)
        {
            Report($"The guest process has not applied page protection #{sequence} within {ProtectTimeoutNanoseconds / 1_000_000_000} s.");
        }

        return false;
    }

    // ---- messages and calls to the guest process ----

    internal void SendFlipEvent(GpuSplitFlipEvent kind, ulong requestId)
    {
        var message = new FlipEventMessage { Kind = (uint)kind, RequestId = requestId };
        _ = _writer.Write(GpuRemoteMessage.FlipEvent, in message);
    }

    internal void SendNotice(GpuSplitNotice kind, string? text = null)
    {
        var notice = new VideoOutNoticeMessage { Kind = (uint)kind };
        var bytes = Encoding.UTF8.GetBytes(text ?? string.Empty);
        notice.TextLength = (uint)Math.Min(bytes.Length, 4096);
        _ = _writer.Write(GpuRemoteMessage.VideoOutNotice, in notice, bytes.AsSpan(0, (int)notice.TextLength));
    }

    internal void SendInterrupt(int eventId, uint contextId)
    {
        var message = new InterruptMessage { EventId = eventId, ContextId = contextId };
        _ = _writer.Write(GpuRemoteMessage.Interrupt, in message);
    }

    // A call, like the flip steps beside it on the GPU's completion thread, so the guest sees
    // interrupts and flips in the order the GPU finished them; a record from the ring thread.
    internal void TriggerInterrupt(int eventId, uint contextId)
    {
        if (OnRingThread || !CallGuest(GpuRemoteReverseOpcode.TriggerInterrupt,
                new InterruptMessage { EventId = eventId, ContextId = contextId }, out ValueResponse _))
        {
            SendInterrupt(eventId, contextId);
        }
    }

    /// <summary>
    /// A call into the guest's video-out state. False when it could not be made: from the ring
    /// thread (see the type remarks), with the guest process gone, or unanswered in time.
    /// </summary>
    internal bool CallGuest<TRequest, TResponse>(GpuRemoteReverseOpcode opcode, in TRequest request, out TResponse response)
        where TRequest : unmanaged
        where TResponse : unmanaged
    {
        response = default;
        if (!CallGuest(opcode, GpuSplitWire.Bytes(in request), out var answer, out var length))
        {
            return false;
        }

        if (length < Unsafe.SizeOf<TResponse>())
        {
            Report($"The guest's answer to {opcode} is {length} bytes; {Unsafe.SizeOf<TResponse>()} were expected.");
            return false;
        }

        response = GpuSplitWire.Read<TResponse>(answer);
        return true;
    }

    internal bool CallGuest(GpuRemoteReverseOpcode opcode, ReadOnlySpan<byte> request, out byte[] response, out int length)
    {
        response = Array.Empty<byte>();
        length = 0;
        if (t_onRingThread)
        {
            Report($"{opcode} was asked of the guest from the ring thread; answered without it.");
            return false;
        }

        var buffer = new byte[_channel.ReverseSyncCalls.PayloadCapacity];
        var result = _channel.ReverseSyncCalls.Call((uint)opcode, request, buffer, out length, out var code, ReverseTimeoutNanoseconds);
        if (result != SyncCallStatus.Completed || code != GpuSplitStatus.Ok)
        {
            if (result != SyncCallStatus.PeerExited)
            {
                Report($"The guest process did not answer {opcode}: {result}, status {code}.");
            }

            return false;
        }

        response = buffer;
        return true;
    }

    // The view log also records every page protection the guest process applies, thousands a
    // second under page watches, and a replica a whole ring (65,536 records) behind starts again
    // from the table, remapping its whole window. Submissions sync it often enough while a title
    // draws; this keeps it close when nothing is submitted (a loading screen, a held prewarm).
    // Replaying early is safe: unmaps of GPU memory happen inside brackets, with the worker parked.
    private void StartReplicaFollower()
    {
        var replica = _guestMemory!.Replica;
        new Thread(() =>
        {
            while (!_stopping && _channel.IsPeerAlive)
            {
                Thread.Sleep(ReplicaFollowMilliseconds);
                try
                {
                    _ = replica.Sync();
                }
                catch (ObjectDisposedException)
                {
                    return;
                }
            }
        })
        {
            IsBackground = true,
            Name = "SharpEmu GPU split replica",
        }.Start();
    }

    private const int ReplicaFollowMilliseconds = 100;

    private void StartInputPoll()
    {
        if (_options.InputPollInterval <= TimeSpan.Zero)
        {
            return;
        }

        _inputThread = new Thread(PollInput) { IsBackground = true, Name = "SharpEmu GPU split input" };
        _inputThread.Start();
    }

    // The window lives here; the guest process's pad and keyboard exports read its state there.
    private unsafe void PollInput()
    {
        var interval = _options.InputPollInterval;
        var sent = long.MinValue;
        Span<int> keys = stackalloc int[InputMessage.MaxKeys];
        while (!_stopping && _channel.IsPeerAlive)
        {
            Thread.Sleep(interval);
            var version = HostWindowInput.Version;
            if (version == sent)
            {
                continue;
            }

            sent = version;
            var count = HostWindowInput.CopyState(keys, out var focused, out var gamepad, out var name);
            var message = new InputMessage
            {
                Focused = focused ? 1u : 0u,
                KeyCount = (uint)count,
                Gamepad = SplitConvert.ToWire(gamepad),
            };
            for (var index = 0; index < count; index++)
            {
                message.Keys[index] = keys[index];
            }

            var nameBytes = Encoding.UTF8.GetBytes(name ?? string.Empty);
            message.NameLength = (uint)Math.Min(nameBytes.Length, InputMessage.MaxNameBytes);
            for (var index = 0; index < message.NameLength; index++)
            {
                message.Name[index] = nameBytes[index];
            }

            _ = _writer.Write(GpuRemoteMessage.Input, in message);
        }
    }

    private void Report(string message)
    {
        if (Interlocked.Increment(ref _reports) <= MaxReports)
        {
            Console.Error.WriteLine($"[GPU-SPLIT][WARN] {message}");
        }
    }

    // ---- teardown ----

    // The guest process is gone or shut down: no more writes to its ring (a thread in the middle of
    // one finishes it first), so nothing writes into the region after the client unmaps it.
    public void Stop()
    {
        _stopping = true;
        _writer.Close();
        lock (_barrierGate)
        {
            Monitor.PulseAll(_barrierGate);
        }

        if (_inputThread is { } input && input != Thread.CurrentThread)
        {
            _ = input.Join(TimeSpan.FromSeconds(2));
        }
    }

    /// <summary>
    /// Takes the seams out again (tests); the GPU process itself just exits, with the backend
    /// still holding guest memory.
    /// </summary>
    public void Dispose()
    {
        Stop();
        if (_globalsInstalled)
        {
            EndOfPipeEvents.Redirect = null;
            VideoOutBackendServices.Bridge = null;
            if (ReferenceEquals(GuestGpuMemoryHook.Current, _memory))
            {
                GuestGpuMemoryHook.Attach(null);
            }
        }
    }

    private sealed class InterruptSink(SplitGpuHost host) : IEndOfPipeSink
    {
        public void TriggerInterrupt(int eventId, uint contextId) => host.TriggerInterrupt(eventId, contextId);
    }
}
