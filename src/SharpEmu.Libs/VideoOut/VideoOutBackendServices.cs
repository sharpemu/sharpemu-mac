// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE.GpuMemory;
using SharpEmu.HLE;
using static SharpEmu.Libs.VideoOut.VideoOutExports;

namespace SharpEmu.Libs.VideoOut;

/// <summary>Guest display and flip services consumed by a graphics backend.
/// Export registration and mutable guest state remain private to VideoOutExports.</summary>
/// <remarks>In the two-process split's GPU process the guest state lives in the other process:
/// there <see cref="Bridge"/> answers every call that reads or changes it.</remarks>
public static class VideoOutBackendServices
{
    private static IVideoOutBackendBridge? _bridge;

    /// <summary>
    /// Where the calls go instead of this process's video-out state (the split's GPU process);
    /// null, the default, keeps them local.
    /// </summary>
    public static IVideoOutBackendBridge? Bridge
    {
        get => Volatile.Read(ref _bridge);
        set => Volatile.Write(ref _bridge, value);
    }

    public static void FinishRenderDiagnostics(string reason) =>
        Gpu.Rendering.ScissorTrace.Instance?.Finish(reason);

    public static bool IsHdrOutputRequested =>
        Bridge is { } bridge ? bridge.IsHdrOutputRequested : VideoOutExports.IsHdrOutputRequested;

    public static bool CanPresentFlip(ulong requestId, long timestamp) =>
        Bridge is { } bridge ? bridge.CanPresentFlip(requestId, timestamp) : VideoOutExports.CanPresentFlip(requestId, timestamp);

    public static long NextFlipPresentTimestamp(ulong requestId, long timestamp) =>
        Bridge is { } bridge
            ? bridge.NextFlipPresentTimestamp(requestId, timestamp)
            : VideoOutExports.NextFlipPresentTimestamp(requestId, timestamp);

    public static void CancelFlip(ulong requestId)
    {
        if (Bridge is { } bridge)
            bridge.CancelFlip(requestId);
        else
            VideoOutExports.CancelFlip(requestId);
    }

    public static void CancelOutstandingFlips()
    {
        if (Bridge is { } bridge)
            bridge.CancelOutstandingFlips();
        else
            VideoOutExports.CancelOutstandingFlips();
    }

    public static void CompleteFlip(ulong requestId)
    {
        if (Bridge is { } bridge)
            bridge.CompleteFlip(requestId);
        else
            VideoOutExports.CompleteFlip(requestId);
    }

    // A presenter that reports presentation: the GPU side of the flip only; the guest's
    // flip count and event follow MarkFlipPresented or DiscardFlip.
    public static void CompleteFlipOnGpu(ulong requestId)
    {
        SplitCrossingStats.Count(SplitCrossing.FlipComplete);
        if (Bridge is { } bridge)
            bridge.CompleteFlipOnGpu(requestId);
        else
            VideoOutExports.CompleteFlipOnGpu(requestId);
    }

    public static void DiscardFlip(ulong requestId)
    {
        if (Bridge is { } bridge)
            bridge.DiscardFlip(requestId);
        else
            VideoOutExports.DiscardFlip(requestId);
    }

    public static string GetApplicationTitleId() =>
        Bridge is { } bridge ? bridge.GetApplicationTitleId() : VideoOutExports.GetApplicationTitleId();

    public static int GetFlipEventCount(ulong requestId) =>
        Bridge is { } bridge ? bridge.GetFlipEventCount(requestId) : VideoOutExports.GetFlipEventCount(requestId);

    public static string GetWindowTitle() =>
        Bridge is { } bridge ? bridge.GetWindowTitle() : VideoOutExports.GetWindowTitle();

    public static bool IsFlipDone(int handle, int bufferIndex) =>
        Bridge is { } bridge ? bridge.IsFlipDone(handle, bufferIndex) : VideoOutExports.IsFlipDone(handle, bufferIndex);

    public static bool IsFlipPresentationPending(ulong requestId) =>
        Bridge is { } bridge ? bridge.IsFlipPresentationPending(requestId) : VideoOutExports.IsFlipPresentationPending(requestId);

    // A property of the pixel format alone: answered here in either process.
    public static bool IsHdrPixelFormat(ulong pixelFormat) =>
        VideoOutExports.IsHdrPixelFormat(pixelFormat);

    public static void MarkFlipPresented(ulong requestId)
    {
        if (Bridge is { } bridge)
            bridge.MarkFlipPresented(requestId);
        else
            VideoOutExports.MarkFlipPresented(requestId);
    }

    public static void NotifyPresentationWindowClosed()
    {
        if (Bridge is { } bridge)
            bridge.NotifyPresentationWindowClosed();
        else
            VideoOutExports.NotifyPresentationWindowClosed();
    }

    public static bool ReleaseNoBufferFlip(int bufferIndex, ulong requestId) =>
        Bridge is { } bridge ? bridge.ReleaseNoBufferFlip(bufferIndex, requestId) : VideoOutExports.ReleaseNoBufferFlip(bufferIndex, requestId);

    public static void ReportPresentedFrame()
    {
        if (Bridge is { } bridge)
            bridge.ReportPresentedFrame();
        else
            VideoOutExports.ReportPresentedFrame();
    }

    public static void SetSelectedGpuName(string gpuName)
    {
        if (Bridge is { } bridge)
            bridge.SetSelectedGpuName(gpuName);
        else
            VideoOutExports.SetSelectedGpuName(gpuName);
    }

    // Takes the request id rather than its event count: under the split the count is a round trip
    // into the guest process, and the remote bridge traces nothing, so only the in-process path
    // reads it (a local lookup).
    public static void TraceGpuFlip(ICpuMemory memory, int handle, int bufferIndex, int flipMode, long flipArg, ulong address, ulong requestId)
    {
        if (Bridge is { } bridge)
            bridge.TraceGpuFlip(handle, bufferIndex, flipMode, flipArg, address, requestId);
        else
            VideoOutExports.TraceGpuFlip(
                memory, handle, bufferIndex, flipMode, flipArg, address, VideoOutExports.GetFlipEventCount(requestId));
    }

    public static bool TryGetDisplayBufferInfo(int handle, int bufferIndex, out DisplayBufferInfo info)
    {
        if (Bridge is { } bridge)
            return bridge.TryGetDisplayBufferInfo(handle, bufferIndex, out info);
        return VideoOutExports.TryGetDisplayBufferInfo(handle, bufferIndex, out info);
    }

    public static int TryReserveFlipRequest(
        int handle,
        int bufferIndex,
        int flipMode,
        long flipArg,
        bool gpuQueued,
        out ulong requestId)
    {
        if (Bridge is { } bridge)
            return bridge.TryReserveFlipRequest(handle, bufferIndex, flipMode, flipArg, gpuQueued, out requestId);
        return VideoOutExports.TryReserveFlipRequest(handle, bufferIndex, flipMode, flipArg, gpuQueued, out requestId);
    }
}

/// <summary>
/// The video-out state a graphics backend reads and changes, when it lives in another process
/// (<see cref="VideoOutBackendServices.Bridge"/>). Each member does what the
/// <see cref="VideoOutBackendServices"/> member of the same name does locally.
/// </summary>
public interface IVideoOutBackendBridge
{
    bool IsHdrOutputRequested { get; }

    bool CanPresentFlip(ulong requestId, long timestamp);

    long NextFlipPresentTimestamp(ulong requestId, long timestamp);

    void CancelFlip(ulong requestId);

    void CancelOutstandingFlips();

    void CompleteFlip(ulong requestId);

    void CompleteFlipOnGpu(ulong requestId);

    void DiscardFlip(ulong requestId);

    string GetApplicationTitleId();

    int GetFlipEventCount(ulong requestId);

    string GetWindowTitle();

    bool IsFlipDone(int handle, int bufferIndex);

    bool IsFlipPresentationPending(ulong requestId);

    void MarkFlipPresented(ulong requestId);

    void NotifyPresentationWindowClosed();

    bool ReleaseNoBufferFlip(int bufferIndex, ulong requestId);

    void ReportPresentedFrame();

    void SetSelectedGpuName(string gpuName);

    // The trace reads guest memory through the caller's memory in one process; across the
    // split it gets the flip's numbers only.
    void TraceGpuFlip(int handle, int bufferIndex, int flipMode, long flipArg, ulong address, ulong requestId);

    bool TryGetDisplayBufferInfo(int handle, int bufferIndex, out DisplayBufferInfo info);

    int TryReserveFlipRequest(int handle, int bufferIndex, int flipMode, long flipArg, bool gpuQueued, out ulong requestId);
}
