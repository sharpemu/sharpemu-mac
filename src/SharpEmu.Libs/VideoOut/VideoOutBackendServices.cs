// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using static SharpEmu.Libs.VideoOut.VideoOutExports;

namespace SharpEmu.Libs.VideoOut;

/// <summary>Guest display and flip services consumed by a graphics backend.
/// Export registration and mutable guest state remain private to VideoOutExports.</summary>
public static class VideoOutBackendServices
{
    public static void FinishRenderDiagnostics(string reason) =>
        Gpu.Rendering.ScissorTrace.Instance?.Finish(reason);

    public static bool IsHdrOutputRequested => VideoOutExports.IsHdrOutputRequested;

    public static bool CanPresentFlip(ulong requestId, long timestamp) =>
        VideoOutExports.CanPresentFlip(requestId, timestamp);

    public static long NextFlipPresentTimestamp(ulong requestId, long timestamp) =>
        VideoOutExports.NextFlipPresentTimestamp(requestId, timestamp);

    public static void CancelFlip(ulong requestId) =>
        VideoOutExports.CancelFlip(requestId);

    public static void CancelOutstandingFlips() =>
        VideoOutExports.CancelOutstandingFlips();

    public static void CompleteFlip(ulong requestId) =>
        VideoOutExports.CompleteFlip(requestId);

    // A presenter that reports presentation: the GPU side of the flip only; the guest's
    // flip count and event follow MarkFlipPresented or DiscardFlip.
    public static void CompleteFlipOnGpu(ulong requestId) =>
        VideoOutExports.CompleteFlipOnGpu(requestId);

    public static void DiscardFlip(ulong requestId) =>
        VideoOutExports.DiscardFlip(requestId);

    public static string GetApplicationTitleId() =>
        VideoOutExports.GetApplicationTitleId();

    public static int GetFlipEventCount(ulong requestId) =>
        VideoOutExports.GetFlipEventCount(requestId);

    public static string GetWindowTitle() =>
        VideoOutExports.GetWindowTitle();

    public static bool IsFlipDone(int handle, int bufferIndex) =>
        VideoOutExports.IsFlipDone(handle, bufferIndex);

    public static bool IsFlipPresentationPending(ulong requestId) =>
        VideoOutExports.IsFlipPresentationPending(requestId);

    public static bool IsHdrPixelFormat(ulong pixelFormat) =>
        VideoOutExports.IsHdrPixelFormat(pixelFormat);

    public static void MarkFlipPresented(ulong requestId) =>
        VideoOutExports.MarkFlipPresented(requestId);

    public static void NotifyPresentationWindowClosed() =>
        VideoOutExports.NotifyPresentationWindowClosed();

    public static bool ReleaseNoBufferFlip(int bufferIndex, ulong requestId) =>
        VideoOutExports.ReleaseNoBufferFlip(bufferIndex, requestId);

    public static void ReportPresentedFrame() =>
        VideoOutExports.ReportPresentedFrame();

    public static void SetSelectedGpuName(string gpuName) =>
        VideoOutExports.SetSelectedGpuName(gpuName);

    public static void TraceGpuFlip(ICpuMemory memory, int handle, int bufferIndex, int flipMode, long flipArg, ulong address, int flipEventCount) =>
        VideoOutExports.TraceGpuFlip(memory, handle, bufferIndex, flipMode, flipArg, address, flipEventCount);

    public static bool TryGetDisplayBufferInfo(int handle, int bufferIndex, out DisplayBufferInfo info) =>
        VideoOutExports.TryGetDisplayBufferInfo(handle, bufferIndex, out info);

    public static int TryReserveFlipRequest(
        int handle,
        int bufferIndex,
        int flipMode,
        long flipArg,
        bool gpuQueued,
        out ulong requestId) =>
        VideoOutExports.TryReserveFlipRequest(handle, bufferIndex, flipMode, flipArg, gpuQueued, out requestId);
}
