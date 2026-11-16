// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace SharpEmu.GpuRemote;

// The two-process split's records and call payloads (docs/macos/split-protocol.md). Every one
// is a blittable struct of explicitly sized fields, padded by hand to 8 bytes, so x86-64 under
// Rosetta and native arm64 agree on every offset. The emulator-side types they carry (video
// options, gamepad state, display buffers) are converted at the edges, in SharpEmu.GpuSplit:
// this assembly references nothing of SharpEmu.

/// <summary>The one-way backend calls of <see cref="GpuRemoteMessage.Control"/>.</summary>
public enum GpuSplitControl : uint
{
    /// <summary>Width and height: <c>IGuestGpuBackend.EnsureStarted</c>.</summary>
    EnsureStarted = 1,
    HideSplashScreen = 2,
    RequestClose = 3,
    CountShaderCompilation = 4,
}

/// <summary>The steps of a flip that B reports to A's video-out state (<see cref="GpuRemoteMessage.FlipEvent"/>).</summary>
public enum GpuSplitFlipEvent : uint
{
    /// <summary><c>VideoOutBackendServices.CompleteFlipOnGpu</c>.</summary>
    CompleteOnGpu = 1,

    /// <summary><c>MarkFlipPresented</c>.</summary>
    Presented = 2,

    /// <summary><c>DiscardFlip</c>.</summary>
    Discarded = 3,

    /// <summary><c>CancelFlip</c>.</summary>
    Cancelled = 4,

    /// <summary>B refused a <see cref="GpuRemoteMessage.FlipRequest"/>; A marks it presented, as a refused flip is in one process.</summary>
    Refused = 5,

    /// <summary><c>CompleteFlip</c>.</summary>
    Completed = 6,
}

/// <summary>The kinds of <see cref="GpuRemoteMessage.VideoOutNotice"/>.</summary>
public enum GpuSplitNotice : uint
{
    CancelOutstandingFlips = 1,
    PresentationWindowClosed = 2,
    ReportPresentedFrame = 3,

    /// <summary>The text is the GPU's name, for the window title (<c>SetSelectedGpuName</c>).</summary>
    SelectedGpuName = 4,
}

/// <summary>What a <see cref="GpuRemoteSyncOpcode.Query"/> asks.</summary>
public enum GpuSplitQuery : uint
{
    /// <summary>A = draws, B = pipelines, C = shader compilations, D = draw milliseconds.</summary>
    PerfCounters = 1,

    /// <summary>A = 1 when the presenter can drive HDR output.</summary>
    HdrOutputSupport = 2,

    /// <summary>A = 1 when the device builds mesh pipelines.</summary>
    MeshPipelines = 3,
}

/// <summary>The texts B asks A for (<see cref="GpuRemoteReverseOpcode.QueryString"/>).</summary>
public enum GpuSplitText : uint
{
    ApplicationTitleId = 1,
    WindowTitle = 2,
}

/// <summary>Kinds of <see cref="GpuRemoteSyncOpcode.BracketBegin"/>.</summary>
public enum GpuSplitBracket : uint
{
    /// <summary>A guest mapping change: the worker parks at its next safe point (<c>IGpuQueueRelay.TryRunAfterPendingWork</c>).</summary>
    MappingChange = 1,

    /// <summary>An ordered change behind every accepted command stream (<c>RunAfterPendingCommandStreams</c>).</summary>
    AfterPendingCommandStreams = 2,
}

/// <summary>The kinds of <see cref="GpuRemoteMessage.GamepadOutput"/>.</summary>
public enum GpuSplitGamepadOutput : uint
{
    Rumble = 1,
    TriggerRumble = 2,
    AdaptiveTriggers = 3,
    Lightbar = 4,
    ResetLightbar = 5,
}

/// <summary>Status codes of the split's synchronous calls (0 is success).</summary>
public static class GpuSplitStatus
{
    public const int Ok = 0;
    public const int UnknownOpcode = -1;
    public const int Malformed = -2;
    public const int NotReady = -3;
    public const int Failed = -4;
}

/// <summary><c>HostVideoOptions</c> as plain integers (enums by value, booleans 0/1).</summary>
[StructLayout(LayoutKind.Sequential)]
public struct VideoOptionsWire
{
    public int WindowMode;
    public int ScalingMode;
    public int Width;
    public int Height;
    public int DisplayIndex;
    public int RefreshRate;
    public int VSync;
    public int HdrMode;
    public int DisplayResolution;
    public int OverlayEnabled;
    public int OverlayCorner;
    public int OverlayMode;
}

[StructLayout(LayoutKind.Sequential)]
public struct SplitStartMessage
{
    /// <summary>A's view log at the time of the start; B's replica starts at least there.</summary>
    public ulong ViewLogSequence;
    public VideoOptionsWire Video;

    /// <summary>1 when <see cref="Video"/> holds options the CLI configured.</summary>
    public uint HasVideoOptions;

    /// <summary>Bytes of the guest memory service name that follow.</summary>
    public uint ServiceNameLength;

    /// <summary>Bytes of the title id that follow the service name (fixed before guest code runs).</summary>
    public uint TitleIdLength;

    /// <summary>A's <c>IsHdrOutputRequested</c> at the start; flips keep B's copy current.</summary>
    public uint HdrRequested;
}

[StructLayout(LayoutKind.Sequential)]
public struct SplitReadyMessage
{
    /// <summary><see cref="GpuSplitStatus"/>.</summary>
    public int Status;

    /// <summary>Bytes of the error text that follow.</summary>
    public int ErrorLength;

    /// <summary>B's replica base (guest address + this = B's address).</summary>
    public ulong ReplicaBase;
}

[StructLayout(LayoutKind.Sequential)]
public struct SubmitCommandStreamMessage
{
    public uint Queue;
    public uint DwordCount;
    public ulong Address;
    public ulong SubmissionId;

    /// <summary>B syncs its replica this far before it takes the stream.</summary>
    public ulong ViewLogSequence;
}

[StructLayout(LayoutKind.Sequential)]
public struct FlipRequestMessage
{
    public int VideoOutHandle;
    public int DisplayBufferIndex;
    public ulong Address;
    public uint Width;
    public uint Height;
    public uint PitchInPixel;

    /// <summary>A's <c>IsHdrOutputRequested</c> as the flip was reserved.</summary>
    public uint HdrRequested;
    public ulong FlipRequestId;
}

[StructLayout(LayoutKind.Sequential)]
public struct CpuWriteMessage
{
    public ulong Address;
    public ulong Size;
}

[StructLayout(LayoutKind.Sequential)]
public struct ShaderHeaderMessage
{
    public ulong CodeAddress;
    public ulong HeaderAddress;
}

/// <summary>What sceAgcDriverSetTFRing and sceAgcDriverSetHsOffchipParam set (TessellationRings).</summary>
[StructLayout(LayoutKind.Sequential)]
public struct TessellationRingsMessage
{
    public ulong FactorRing;
    public ulong OffchipBuffer;
    public uint FactorRingSize;
    public uint OffchipParameter;
}

/// <summary>A fused program registered (all four set) or unregistered (only <see cref="Entry"/> set).</summary>
[StructLayout(LayoutKind.Sequential)]
public struct FusedProgramMessage
{
    public ulong Entry;
    public ulong EntryHeader;
    public ulong Continuation;
    public ulong ContinuationHeader;
}

[StructLayout(LayoutKind.Sequential)]
public struct ControlMessage
{
    /// <summary><see cref="GpuSplitControl"/>.</summary>
    public uint Kind;
    public uint Width;
    public uint Height;
    public uint Reserved;
}

[StructLayout(LayoutKind.Sequential)]
public struct HostFrameMessage
{
    public uint Width;
    public uint Height;

    /// <summary>Bytes of the whole frame; B presents it once every piece arrived.</summary>
    public uint TotalBytes;

    /// <summary>Where this record's piece goes in the frame.</summary>
    public uint Offset;
}

[StructLayout(LayoutKind.Sequential)]
public struct AdaptiveTriggerWire
{
    /// <summary>0: no effect (null), 1: the bytes below.</summary>
    public byte Present;
    public byte FallbackStrength;
    public byte B0, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10;
    public byte Reserved0, Reserved1, Reserved2;
}

[StructLayout(LayoutKind.Sequential)]
public struct GamepadOutputMessage
{
    /// <summary><see cref="GpuSplitGamepadOutput"/>.</summary>
    public uint Kind;

    /// <summary>Rumble: large and small motor. Trigger rumble: left and right strength. Lightbar: red, green, blue.</summary>
    public byte A;
    public byte B;
    public byte C;

    /// <summary>Trigger rumble: bit 0 left present, bit 1 right present.</summary>
    public byte Flags;
    public AdaptiveTriggerWire Left;
    public AdaptiveTriggerWire Right;
}

/// <summary>Flags of a <see cref="ProtectRequestMessage"/>.</summary>
[Flags]
public enum ProtectRequestFlags : uint
{
    None = 0,

    /// <summary>A derived protection (<c>IGuestDerivedProtectionAddressSpace</c>): <see cref="ProtectRequestMessage.GuestAccess"/> is valid.</summary>
    Derived = 1,

    /// <summary>The derived protection's <c>ordinaryEnrollment</c>.</summary>
    OrdinaryEnrollment = 2,
}

[StructLayout(LayoutKind.Sequential)]
public struct ProtectRequestMessage
{
    /// <summary>B's request number, from 1; A acknowledges through the protection counter.</summary>
    public ulong Sequence;
    public ulong Address;
    public ulong Size;

    /// <summary>The guest's own protection (<c>GuestPageProtection</c>), for derived requests.</summary>
    public uint GuestAccess;

    /// <summary>The protection to apply (<c>GuestPageProtection</c>).</summary>
    public uint Access;
    public ProtectRequestFlags Flags;
    public uint Reserved;
}

[StructLayout(LayoutKind.Sequential)]
public struct InterruptMessage
{
    public int EventId;
    public uint ContextId;
}

[StructLayout(LayoutKind.Sequential)]
public struct FlipEventMessage
{
    /// <summary><see cref="GpuSplitFlipEvent"/>.</summary>
    public uint Kind;
    public uint Reserved;
    public ulong RequestId;
}

[StructLayout(LayoutKind.Sequential)]
public struct VideoOutNoticeMessage
{
    /// <summary><see cref="GpuSplitNotice"/>.</summary>
    public uint Kind;
    public uint TextLength;
}

/// <summary><c>HostGamepadState</c> as plain fields.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct GamepadStateWire
{
    public uint Connected;
    public uint Buttons;
    public byte LeftX;
    public byte LeftY;
    public byte RightX;
    public byte RightY;
    public byte LeftTrigger;
    public byte RightTrigger;
    public byte Type;
    public byte Connection;
    public uint MotionAvailable;
    public float AccelerationX;
    public float AccelerationY;
    public float AccelerationZ;
    public float AngularVelocityX;
    public float AngularVelocityY;
    public float AngularVelocityZ;
    public uint FirstTouchActive;
    public uint FirstTouchId;
    public float FirstTouchX;
    public float FirstTouchY;
    public uint SecondTouchActive;
    public uint SecondTouchId;
    public float SecondTouchX;
    public float SecondTouchY;
    public uint BatteryPercent;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe struct InputMessage
{
    public const int MaxKeys = 64;
    public const int MaxNameBytes = 64;

    public uint Focused;
    public uint KeyCount;
    public fixed int Keys[MaxKeys];
    public GamepadStateWire Gamepad;
    public uint NameLength;
    public uint Reserved;
    public fixed byte Name[MaxNameBytes];
}

[StructLayout(LayoutKind.Sequential)]
public struct BarrierRequest
{
    /// <summary>Split records A had written when it called; B serves the call once it handled that many.</summary>
    public ulong Barrier;
}

[StructLayout(LayoutKind.Sequential)]
public struct FrameBoundaryMessage
{
    /// <summary>A's number for the boundary; the matching <see cref="FrameBoundaryWaitRequest"/> names it.</summary>
    public ulong Ticket;
}

[StructLayout(LayoutKind.Sequential)]
public struct FrameBoundaryWaitRequest
{
    /// <summary>Split records A had written once the boundary was on the ring.</summary>
    public ulong Barrier;
    public ulong Ticket;
}

[StructLayout(LayoutKind.Sequential)]
public struct IdleOutcomeResponse
{
    /// <summary>The <c>IdleOutcome</c> value.</summary>
    public int Outcome;
    public int Reserved;
}

[StructLayout(LayoutKind.Sequential)]
public struct BracketBeginRequest
{
    public ulong Barrier;

    /// <summary><see cref="GpuSplitBracket"/>.</summary>
    public uint Kind;
    public uint Reserved;
}

[StructLayout(LayoutKind.Sequential)]
public struct BracketResponse
{
    public ulong BracketId;
}

[StructLayout(LayoutKind.Sequential)]
public struct BracketEndRequest
{
    public ulong BracketId;

    /// <summary>A's view log after its change: B's replica syncs this far before the worker goes on.</summary>
    public ulong ViewLogSequence;
}

[StructLayout(LayoutKind.Sequential)]
public struct FaultRequest
{
    /// <summary>The <c>FaultKind</c> value.</summary>
    public uint Kind;
    public uint Reserved;
    public ulong Address;
    public ulong Size;
}

[StructLayout(LayoutKind.Sequential)]
public struct HandledResponse
{
    public uint Handled;
    public uint Reserved;

    /// <summary>
    /// A fault no store handled: the GPU process's write-restoration version of the page (its watch
    /// was removed, and its protection is applied by now), or 0. The guest retries the store once per
    /// version, as one process retries a write whose watch was removed before the fault reached it.
    /// </summary>
    public long RestorationVersion;
}

[StructLayout(LayoutKind.Sequential)]
public struct CpuWriteRequest
{
    public ulong Barrier;
    public ulong Address;
    public ulong Size;
}

[StructLayout(LayoutKind.Sequential)]
public struct SpanRequest
{
    public ulong Address;
    public ulong Size;

    /// <summary>The <c>GuestPageProtection</c> value (register and protect).</summary>
    public uint Protection;
    public uint Reserved;

    /// <summary>The bracket the calling thread is inside, or 0.</summary>
    public ulong BracketId;
}

[StructLayout(LayoutKind.Sequential)]
public struct QueryRequest
{
    public uint Which;
    public uint Reserved;
}

[StructLayout(LayoutKind.Sequential)]
public struct QueryResponse
{
    public long A;
    public long B;
    public long C;
    public double D;
}

[StructLayout(LayoutKind.Sequential)]
public struct ReserveFlipRequest
{
    public int Handle;
    public int BufferIndex;
    public int FlipMode;
    public int Reserved;
    public long FlipArgument;
}

[StructLayout(LayoutKind.Sequential)]
public struct ReserveFlipResponse
{
    public int Result;

    /// <summary>A's <c>IsHdrOutputRequested</c> after the reservation.</summary>
    public uint HdrRequested;
    public ulong RequestId;

    /// <summary>
    /// The display buffer the flip names, read with the reservation so the GPU process does not
    /// ask for it in a second call (<see cref="DisplayBufferResponse.Found"/> is zero when none).
    /// </summary>
    public DisplayBufferResponse Buffer;
}

[StructLayout(LayoutKind.Sequential)]
public struct FlipSlotRequest
{
    public int Handle;
    public int BufferIndex;
}

[StructLayout(LayoutKind.Sequential)]
public struct FlipTimingRequest
{
    public ulong RequestId;

    /// <summary>A <c>Stopwatch</c> timestamp: both processes read the same clock.</summary>
    public long Timestamp;
}

[StructLayout(LayoutKind.Sequential)]
public struct FlipIdRequest
{
    public ulong RequestId;
}

[StructLayout(LayoutKind.Sequential)]
public struct ReleaseNoBufferRequest
{
    public int BufferIndex;
    public int Reserved;
    public ulong RequestId;
}

[StructLayout(LayoutKind.Sequential)]
public struct ValueResponse
{
    public long Value;
}

[StructLayout(LayoutKind.Sequential)]
public struct DisplayBufferResponse
{
    public uint Found;
    public uint TilingMode;
    public ulong Address;
    public ulong PixelFormat;
    public uint Width;
    public uint Height;
    public uint PitchInPixel;
    public uint Reserved;
    public ulong Option;
}

/// <summary>Reading and writing the split's payloads.</summary>
public static class GpuSplitWire
{
    /// <summary>Reads a <typeparamref name="T"/> from the start of <paramref name="payload"/>; throws when it is shorter.</summary>
    public static T Read<T>(ReadOnlySpan<byte> payload)
        where T : unmanaged => RingReader.Read<T>(payload);

    /// <summary>Writes <paramref name="value"/> at the start of <paramref name="destination"/> and returns its size.</summary>
    public static int Write<T>(Span<byte> destination, in T value)
        where T : unmanaged
    {
        MemoryMarshal.Write(destination, in value);
        return Unsafe.SizeOf<T>();
    }

    /// <summary>The bytes of <paramref name="value"/>.</summary>
    public static ReadOnlySpan<byte> Bytes<T>(in T value)
        where T : unmanaged => MemoryMarshal.AsBytes(new ReadOnlySpan<T>(in value));

    /// <summary>A payload of <paramref name="header"/> followed by <paramref name="text"/> in UTF-8, cut to <paramref name="maxTextBytes"/>.</summary>
    public static byte[] WithText<T>(in T header, string? text, int maxTextBytes, out int textLength)
        where T : unmanaged
    {
        var encoded = Encoding.UTF8.GetBytes(text ?? string.Empty);
        textLength = Math.Min(encoded.Length, Math.Max(0, maxTextBytes));
        var payload = new byte[Unsafe.SizeOf<T>() + textLength];
        MemoryMarshal.Write(payload, in header);
        encoded.AsSpan(0, textLength).CopyTo(payload.AsSpan(Unsafe.SizeOf<T>()));
        return payload;
    }

    /// <summary>The UTF-8 text of <paramref name="length"/> bytes after a <typeparamref name="T"/> header.</summary>
    public static string TextAfter<T>(ReadOnlySpan<byte> payload, int length)
        where T : unmanaged
    {
        var start = Unsafe.SizeOf<T>();
        if (length < 0 || payload.Length < start + length)
        {
            throw new InvalidDataException($"A {typeof(T).Name} record says {length} text bytes follow; {payload.Length - start} do.");
        }

        return Encoding.UTF8.GetString(payload.Slice(start, length));
    }
}
