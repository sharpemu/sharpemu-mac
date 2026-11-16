// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.GpuRemote;
using SharpEmu.HLE.Host;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.Libs.Pad;
using SharpEmu.Libs.VideoOut;

namespace SharpEmu.GpuSplit;

/// <summary>
/// What the guest process does with the GPU process's messages and questions: its video-out
/// state, event queues and input. A seam of its own so a test can stand in for the emulator.
/// </summary>
internal interface ISplitGuestSide
{
    void TriggerInterrupt(int eventId, uint contextId);

    void FlipEvent(GpuSplitFlipEvent kind, ulong requestId);

    void Notice(GpuSplitNotice kind, string text);

    void Input(bool focused, ReadOnlySpan<int> keys, in HostGamepadState gamepad, string? gamepadName);

    int TryReserveFlipRequest(int handle, int bufferIndex, int flipMode, long flipArg, out ulong requestId);

    bool IsFlipDone(int handle, int bufferIndex);

    bool CanPresentFlip(ulong requestId, long timestamp);

    long NextFlipPresentTimestamp(ulong requestId, long timestamp);

    bool TryGetDisplayBufferInfo(int handle, int bufferIndex, out VideoOutExports.DisplayBufferInfo info);

    int GetFlipEventCount(ulong requestId);

    bool ReleaseNoBufferFlip(int bufferIndex, ulong requestId);

    bool IsFlipPresentationPending(ulong requestId);

    bool IsHdrOutputRequested { get; }

    string GetWindowTitle();

    string GetApplicationTitleId();

    /// <summary>The GPU process is gone: stop the guest, saying why.</summary>
    void OnGpuProcessLost(string reason);
}

/// <summary>The emulator's own <see cref="ISplitGuestSide"/>: the local video-out state, event queues and window input.</summary>
internal sealed class GuestProcessSide : ISplitGuestSide
{
    public void TriggerInterrupt(int eventId, uint contextId) => EndOfPipeEvents.Deliver(eventId, contextId);

    public void FlipEvent(GpuSplitFlipEvent kind, ulong requestId)
    {
        switch (kind)
        {
            case GpuSplitFlipEvent.CompleteOnGpu:
                VideoOutBackendServices.CompleteFlipOnGpu(requestId);
                break;
            case GpuSplitFlipEvent.Presented:
            case GpuSplitFlipEvent.Refused:
                // A refused export flip is marked presented, as the export does when the backend
                // refuses in one process; before or after the export's CompleteFlip, either order
                // retires the request.
                VideoOutBackendServices.MarkFlipPresented(requestId);
                break;
            case GpuSplitFlipEvent.Discarded:
                VideoOutBackendServices.DiscardFlip(requestId);
                break;
            case GpuSplitFlipEvent.Cancelled:
                VideoOutBackendServices.CancelFlip(requestId);
                break;
            case GpuSplitFlipEvent.Completed:
                VideoOutBackendServices.CompleteFlip(requestId);
                break;
        }
    }

    public void Notice(GpuSplitNotice kind, string text)
    {
        switch (kind)
        {
            case GpuSplitNotice.CancelOutstandingFlips:
                VideoOutBackendServices.CancelOutstandingFlips();
                break;
            case GpuSplitNotice.PresentationWindowClosed:
                VideoOutBackendServices.NotifyPresentationWindowClosed();
                break;
            case GpuSplitNotice.ReportPresentedFrame:
                VideoOutBackendServices.ReportPresentedFrame();
                break;
            case GpuSplitNotice.SelectedGpuName:
                VideoOutBackendServices.SetSelectedGpuName(text);
                break;
        }
    }

    public void Input(bool focused, ReadOnlySpan<int> keys, in HostGamepadState gamepad, string? gamepadName)
    {
        HostWindowInput.SetFocused(focused);
        HostWindowInput.SetKeys(keys);
        if (gamepad.Connected)
        {
            HostWindowInput.SetGamepad(gamepadName, gamepad);
        }
        else
        {
            HostWindowInput.ClearGamepad();
        }
    }

    public int TryReserveFlipRequest(int handle, int bufferIndex, int flipMode, long flipArg, out ulong requestId) =>
        VideoOutBackendServices.TryReserveFlipRequest(handle, bufferIndex, flipMode, flipArg, gpuQueued: true, out requestId);

    public bool IsFlipDone(int handle, int bufferIndex) => VideoOutBackendServices.IsFlipDone(handle, bufferIndex);

    public bool CanPresentFlip(ulong requestId, long timestamp) => VideoOutBackendServices.CanPresentFlip(requestId, timestamp);

    public long NextFlipPresentTimestamp(ulong requestId, long timestamp) =>
        VideoOutBackendServices.NextFlipPresentTimestamp(requestId, timestamp);

    public bool TryGetDisplayBufferInfo(int handle, int bufferIndex, out VideoOutExports.DisplayBufferInfo info) =>
        VideoOutBackendServices.TryGetDisplayBufferInfo(handle, bufferIndex, out info);

    public int GetFlipEventCount(ulong requestId) => VideoOutBackendServices.GetFlipEventCount(requestId);

    public bool ReleaseNoBufferFlip(int bufferIndex, ulong requestId) => VideoOutBackendServices.ReleaseNoBufferFlip(bufferIndex, requestId);

    public bool IsFlipPresentationPending(ulong requestId) => VideoOutBackendServices.IsFlipPresentationPending(requestId);

    public bool IsHdrOutputRequested => VideoOutBackendServices.IsHdrOutputRequested;

    public string GetWindowTitle() => VideoOutBackendServices.GetWindowTitle();

    public string GetApplicationTitleId() => VideoOutBackendServices.GetApplicationTitleId();

    public void OnGpuProcessLost(string reason)
    {
        Console.Error.WriteLine($"[GPU-SPLIT][ERROR] {reason} The guest is stopped.");
        // The window-closed path is the cooperative shutdown with a bounded exit behind it.
        VideoOutBackendServices.NotifyPresentationWindowClosed();
    }
}
