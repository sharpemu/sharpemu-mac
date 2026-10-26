// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;

namespace SharpEmu.GpuRemote;

/// <summary>Versioning of everything both processes must agree on: region layout and messages.</summary>
public static class GpuRemoteProtocol
{
    /// <summary>Bumped on any change to the region layout or a message format.</summary>
    /// <remarks>2: the B → A call table, the application block and the split's messages (GpuSplitMessages.cs).
    /// 3: <see cref="GpuRemoteMessage.TessellationRings"/>.
    /// 4: <see cref="GpuRemoteMessage.FrameBoundary"/> and <see cref="GpuRemoteSyncOpcode.WaitFrameBoundary"/>.</remarks>
    public const uint Version = 4;
}

/// <summary>Record types on the two rings. 0 is never used; <see cref="SpscRing.PaddingType"/> is reserved.</summary>
public enum GpuRemoteMessage : uint
{
    /// <summary>A → B, the first record A writes: <see cref="HelloMessage"/>.</summary>
    Hello = 1,

    /// <summary>B → A, the answer to <see cref="Hello"/>: <see cref="HelloMessage"/> describing B.</summary>
    HelloAck = 2,

    /// <summary>A → B: <see cref="PingMessage"/>, answered with <see cref="Pong"/>.</summary>
    Ping = 3,

    /// <summary>B → A: the <see cref="PingMessage"/> of a <see cref="Ping"/>, echoed.</summary>
    Pong = 4,

    /// <summary>A → B: exit cleanly. No payload.</summary>
    Shutdown = 5,

    /// <summary>A → B: any payload, consumed and dropped (throughput measurement).</summary>
    Sink = 6,

    /// <summary>A → B: <see cref="BurstMessage"/>, asking B to write that many <see cref="BurstItem"/> records back.</summary>
    Burst = 7,

    /// <summary>B → A: one record of a <see cref="Burst"/>.</summary>
    BurstItem = 8,

    // The two-process split (docs/macos/split-protocol.md); payloads in GpuSplitMessages.cs.

    /// <summary>A → B, the first split record: <see cref="SplitStartMessage"/>, then the guest memory service name (UTF-8).</summary>
    SplitStart = 9,

    /// <summary>B → A, the answer to <see cref="SplitStart"/>: <see cref="SplitReadyMessage"/>, then an error text (UTF-8) if it failed.</summary>
    SplitReady = 10,

    /// <summary>A → B: <see cref="SubmitCommandStreamMessage"/>.</summary>
    SubmitCommandStream = 11,

    /// <summary>A → B: <see cref="FlipRequestMessage"/>, a video-out export flip.</summary>
    FlipRequest = 12,

    /// <summary>A → B: <see cref="CpuWriteMessage"/>, a managed write that has landed (or may have).</summary>
    CpuWrite = 13,

    /// <summary>A → B: <see cref="ShaderHeaderMessage"/>, one entry of the shader-header table.</summary>
    ShaderHeader = 14,

    /// <summary>A → B: <see cref="ControlMessage"/>, a one-way backend call.</summary>
    Control = 15,

    /// <summary>A → B: <see cref="HostFrameMessage"/>, then one piece of a CPU-produced BGRA frame.</summary>
    HostFrame = 16,

    /// <summary>A → B: <see cref="GamepadOutputMessage"/> (rumble, triggers, lightbar).</summary>
    GamepadOutput = 17,

    /// <summary>A → B: <see cref="VideoOptionsWire"/>, the presenter's window options.</summary>
    VideoOptions = 18,

    /// <summary>B → A: <see cref="ProtectRequestMessage"/>, a page protection for A to apply.</summary>
    ProtectRequest = 19,

    /// <summary>B → A: <see cref="InterruptMessage"/>, a GPU end-of-pipe interrupt.</summary>
    Interrupt = 20,

    /// <summary>B → A: <see cref="FlipEventMessage"/>, a step of a flip's life.</summary>
    FlipEvent = 21,

    /// <summary>B → A: <see cref="VideoOutNoticeMessage"/>, then a text (UTF-8) for the kinds that carry one.</summary>
    VideoOutNotice = 22,

    /// <summary>B → A: <see cref="InputMessage"/>, the window's keyboard and gamepad state.</summary>
    Input = 23,

    /// <summary>A → B: <see cref="FusedProgramMessage"/>, one change of the fused-program registry.</summary>
    FusedProgram = 24,

    /// <summary>A → B: <see cref="TessellationRingsMessage"/>, the guest driver's tessellation memory.</summary>
    TessellationRings = 25,

    /// <summary>A → B: <see cref="FrameBoundaryMessage"/>, the ordering half of a split <c>SubmitDone</c>.</summary>
    FrameBoundary = 26,
}

/// <summary>Opcodes of the A → B synchronous call table.</summary>
public enum GpuRemoteSyncOpcode : uint
{
    /// <summary>Echoes the request back as the response, status 0.</summary>
    Ping = 1,

    /// <summary><see cref="BarrierRequest"/> → <see cref="IdleOutcomeResponse"/>: the frame boundary.</summary>
    SubmitDone = 2,

    /// <summary><see cref="BracketBeginRequest"/> → <see cref="BracketResponse"/>: park B's GPU worker.</summary>
    BracketBegin = 3,

    /// <summary><see cref="BracketEndRequest"/> → nothing: sync the replica and let the worker go.</summary>
    BracketEnd = 4,

    /// <summary><see cref="FaultRequest"/> → <see cref="HandledResponse"/>: from A's write-fault handler.</summary>
    Fault = 5,

    /// <summary><see cref="CpuWriteRequest"/> → <see cref="HandledResponse"/>: a managed write before it lands.</summary>
    CpuWrite = 6,

    /// <summary><see cref="SpanRequest"/> → nothing: a GPU-memory span registered in A.</summary>
    SpanRegister = 7,

    /// <summary><see cref="SpanRequest"/> → <see cref="HandledResponse"/>: a guest protection change of a span.</summary>
    SpanProtect = 8,

    /// <summary><see cref="SpanRequest"/> → nothing: a span unregistered in A.</summary>
    SpanUnregister = 9,

    /// <summary><see cref="QueryRequest"/> → <see cref="QueryResponse"/>.</summary>
    Query = 10,

    /// <summary><see cref="FrameBoundaryWaitRequest"/> → <see cref="IdleOutcomeResponse"/>: the waiting half of a split <c>SubmitDone</c>.</summary>
    WaitFrameBoundary = 11,
}

/// <summary>Opcodes of the B → A synchronous call table: what B's GPU code asks of A's video-out state.</summary>
public enum GpuRemoteReverseOpcode : uint
{
    /// <summary><see cref="ReserveFlipRequest"/> → <see cref="ReserveFlipResponse"/>.</summary>
    TryReserveFlip = 1,

    /// <summary><see cref="FlipSlotRequest"/> → <see cref="ValueResponse"/> (0 or 1).</summary>
    IsFlipDone = 2,

    /// <summary><see cref="FlipTimingRequest"/> → <see cref="ValueResponse"/> (0 or 1).</summary>
    CanPresentFlip = 3,

    /// <summary><see cref="FlipTimingRequest"/> → <see cref="ValueResponse"/> (a timestamp).</summary>
    NextFlipPresentTimestamp = 4,

    /// <summary><see cref="FlipSlotRequest"/> → <see cref="DisplayBufferResponse"/>.</summary>
    TryGetDisplayBuffer = 5,

    /// <summary><see cref="FlipIdRequest"/> → <see cref="ValueResponse"/>.</summary>
    GetFlipEventCount = 6,

    /// <summary><see cref="ReleaseNoBufferRequest"/> → <see cref="ValueResponse"/> (0 or 1).</summary>
    ReleaseNoBufferFlip = 7,

    /// <summary><see cref="QueryRequest"/> (a <see cref="GpuSplitText"/>) → that text (UTF-8).</summary>
    QueryString = 8,

    /// <summary><see cref="FlipEventMessage"/> → nothing: a step of a flip, applied before the call returns.</summary>
    FlipEvent = 9,

    /// <summary><see cref="FlipIdRequest"/> → <see cref="ValueResponse"/> (0 or 1).</summary>
    IsFlipPresentationPending = 10,

    /// <summary>Nothing → <see cref="ValueResponse"/> (0 or 1): whether the guest flips HDR now.</summary>
    IsHdrOutputRequested = 11,

    /// <summary><see cref="InterruptMessage"/> → nothing: an end-of-pipe interrupt, delivered before the call returns.</summary>
    TriggerInterrupt = 12,
}

/// <summary>Handshake record, sent each way.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct HelloMessage
{
    public uint ProtocolVersion;
    public int ProcessId;

    /// <summary>The sender's <see cref="System.Runtime.InteropServices.Architecture"/>.</summary>
    public uint Architecture;

    /// <summary>1 when the sender runs under Rosetta, 0 when native, -1 unknown.</summary>
    public int Translated;
}

[StructLayout(LayoutKind.Sequential)]
public struct PingMessage
{
    public ulong Sequence;
    public long Timestamp;
}

[StructLayout(LayoutKind.Sequential)]
public struct BurstMessage
{
    public int Count;
    public int PayloadLength;
}

/// <summary>What one side learned about the other in the handshake.</summary>
public readonly record struct GpuPeerInfo(int ProcessId, uint ProtocolVersion, Architecture Architecture, int Translated)
{
    internal static GpuPeerInfo From(in HelloMessage hello) =>
        new(hello.ProcessId, hello.ProtocolVersion, (Architecture)hello.Architecture, hello.Translated);

    /// <summary>This process, as it describes itself in the handshake.</summary>
    public static GpuPeerInfo ThisProcess => From(DescribeThisProcess());

    internal static HelloMessage DescribeThisProcess() => new()
    {
        ProtocolVersion = GpuRemoteProtocol.Version,
        ProcessId = Environment.ProcessId,
        Architecture = (uint)RuntimeInformation.ProcessArchitecture,
        Translated = RemoteNative.ProcessTranslated(),
    };

    public override string ToString() =>
        $"pid {ProcessId}, {Architecture}{(Translated switch { 1 => " (Rosetta)", 0 => " (native)", _ => string.Empty })}, protocol {ProtocolVersion}";
}
