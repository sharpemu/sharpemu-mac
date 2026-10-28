// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.Versioning;
using System.Text;
using SharpEmu.GpuRemote;
using SharpEmu.Libs.VideoOut;

namespace SharpEmu.GpuSplit;

/// <summary>
/// The GPU process's <see cref="VideoOutBackendServices.Bridge"/>: the video-out state machine
/// lives in the guest process, and every question the backend asks of it is a call there
/// (<see cref="GpuRemoteReverseOpcode"/>).
/// </summary>
/// <remarks>
/// <para>Answered here: the title id (fixed before guest code runs, sent with the start) and
/// whether the guest flips HDR, which only a flip reservation changes - B's copy follows every
/// export flip (<see cref="GpuRemoteMessage.FlipRequest"/>) and every GPU flip it reserves.</para>
///
/// <para>Steps of a flip (presented, discarded, done on the GPU) are calls too, so the guest's
/// state has moved before the presenter asks its next question about the flip. From the ring
/// thread, which must never wait for the guest (see <see cref="SplitGpuHost"/>), they go as
/// records, and the GPU name as a notice. A call the guest cannot answer (it exited) gets the
/// answer that lets the GPU side carry on: a flip that may present, a buffer slot that is free.</para>
///
/// <para>Fast-path TODO: <see cref="CanPresentFlip"/> and <see cref="NextFlipPresentTimestamp"/>
/// cost a round trip per presenter pass; the flip pacing state could be mirrored instead.</para>
/// </remarks>
[SupportedOSPlatform("macos")]
internal sealed class RemoteVideoOutBridge(SplitGpuHost host, string titleId, bool hdrRequested) : IVideoOutBackendBridge
{
    private const int FailedResult = unchecked((int)0x8029_00FF);

    private volatile bool _hdrRequested = hdrRequested;

    public bool IsHdrOutputRequested => _hdrRequested;

    internal void NoteHdrRequested(bool requested) => _hdrRequested = requested;

    public bool CanPresentFlip(ulong requestId, long timestamp) =>
        !host.CallGuest(GpuRemoteReverseOpcode.CanPresentFlip, new FlipTimingRequest { RequestId = requestId, Timestamp = timestamp },
            out ValueResponse answer) || answer.Value != 0;

    public long NextFlipPresentTimestamp(ulong requestId, long timestamp) =>
        host.CallGuest(GpuRemoteReverseOpcode.NextFlipPresentTimestamp, new FlipTimingRequest { RequestId = requestId, Timestamp = timestamp },
            out ValueResponse answer)
            ? answer.Value
            : timestamp;

    public void CancelFlip(ulong requestId) => FlipEvent(GpuSplitFlipEvent.Cancelled, requestId);

    public void CancelOutstandingFlips() => host.SendNotice(GpuSplitNotice.CancelOutstandingFlips);

    public void CompleteFlip(ulong requestId) => FlipEvent(GpuSplitFlipEvent.Completed, requestId);

    public void CompleteFlipOnGpu(ulong requestId) => FlipEvent(GpuSplitFlipEvent.CompleteOnGpu, requestId);

    public void DiscardFlip(ulong requestId) => FlipEvent(GpuSplitFlipEvent.Discarded, requestId);

    public string GetApplicationTitleId() => titleId;

    public int GetFlipEventCount(ulong requestId) =>
        host.CallGuest(GpuRemoteReverseOpcode.GetFlipEventCount, new FlipIdRequest { RequestId = requestId }, out ValueResponse answer)
            ? (int)answer.Value
            : 0;

    public string GetWindowTitle()
    {
        var request = new QueryRequest { Which = (uint)GpuSplitText.WindowTitle };
        return host.CallGuest(GpuRemoteReverseOpcode.QueryString, GpuSplitWire.Bytes(in request), out var answer, out var length)
            ? Encoding.UTF8.GetString(answer, 0, length)
            : $"SharpEmu [{titleId}]";
    }

    public bool IsFlipDone(int handle, int bufferIndex) =>
        !host.CallGuest(GpuRemoteReverseOpcode.IsFlipDone, new FlipSlotRequest { Handle = handle, BufferIndex = bufferIndex },
            out ValueResponse answer) || answer.Value != 0;

    public bool IsFlipPresentationPending(ulong requestId) =>
        host.CallGuest(GpuRemoteReverseOpcode.IsFlipPresentationPending, new FlipIdRequest { RequestId = requestId }, out ValueResponse answer) &&
        answer.Value != 0;

    public void MarkFlipPresented(ulong requestId) => FlipEvent(GpuSplitFlipEvent.Presented, requestId);

    public void NotifyPresentationWindowClosed() => host.SendNotice(GpuSplitNotice.PresentationWindowClosed);

    // Only a flip that names no buffer (-1) is released here; every other flip answers false without
    // the guest's help, so it is not worth a round trip.
    public bool ReleaseNoBufferFlip(int bufferIndex, ulong requestId) =>
        bufferIndex == -1 &&
        host.CallGuest(GpuRemoteReverseOpcode.ReleaseNoBufferFlip, new ReleaseNoBufferRequest { BufferIndex = bufferIndex, RequestId = requestId },
            out ValueResponse answer) &&
        answer.Value != 0;

    public void ReportPresentedFrame() => host.SendNotice(GpuSplitNotice.ReportPresentedFrame);

    public void SetSelectedGpuName(string gpuName) => host.SendNotice(GpuSplitNotice.SelectedGpuName, gpuName);

    // The trace reads guest memory and video-out state of the guest process; not carried yet.
    public void TraceGpuFlip(int handle, int bufferIndex, int flipMode, long flipArg, ulong address, ulong requestId)
    {
    }

    private readonly record struct ReservedBuffer(RemoteVideoOutBridge? Owner, int Handle, int BufferIndex, DisplayBufferResponse Wire);

    // One reservation's buffer, consumed by the next lookup on this thread. Only the same
    // bridge and slot may use it; an intervening lookup discards it and asks the guest.
    [ThreadStatic]
    private static ReservedBuffer t_reservedBuffer;

    public bool TryGetDisplayBufferInfo(int handle, int bufferIndex, out VideoOutExports.DisplayBufferInfo info)
    {
        var reserved = t_reservedBuffer;
        t_reservedBuffer = default;
        if (ReferenceEquals(reserved.Owner, this) && reserved.Wire.Found != 0 &&
            reserved.Handle == handle && reserved.BufferIndex == bufferIndex)
        {
            info = SplitConvert.FromWire(reserved.Wire);
            return true;
        }

        info = default;
        if (!host.CallGuest(GpuRemoteReverseOpcode.TryGetDisplayBuffer, new FlipSlotRequest { Handle = handle, BufferIndex = bufferIndex },
                out DisplayBufferResponse answer) || answer.Found == 0)
        {
            return false;
        }

        info = SplitConvert.FromWire(answer);
        return true;
    }

    public int TryReserveFlipRequest(int handle, int bufferIndex, int flipMode, long flipArg, bool gpuQueued, out ulong requestId)
    {
        requestId = 0;
        var request = new ReserveFlipRequest { Handle = handle, BufferIndex = bufferIndex, FlipMode = flipMode, FlipArgument = flipArg };
        if (!host.CallGuest(GpuRemoteReverseOpcode.TryReserveFlip, request, out ReserveFlipResponse answer))
        {
            return FailedResult;
        }

        _hdrRequested = answer.HdrRequested != 0;
        requestId = answer.RequestId;
        // The display buffer this flip names came with the answer; the capture that follows on this
        // thread takes it from here (see TryGetDisplayBufferInfo).
        t_reservedBuffer = answer.Result == 0 && answer.Buffer.Found != 0
            ? new ReservedBuffer(this, handle, bufferIndex, answer.Buffer)
            : default;
        return answer.Result;
    }

    private void FlipEvent(GpuSplitFlipEvent kind, ulong requestId)
    {
        if (SplitGpuHost.OnRingThread)
        {
            host.SendFlipEvent(kind, requestId);
            return;
        }

        _ = host.CallGuest(GpuRemoteReverseOpcode.FlipEvent, new FlipEventMessage { Kind = (uint)kind, RequestId = requestId },
            out ValueResponse _);
    }
}
