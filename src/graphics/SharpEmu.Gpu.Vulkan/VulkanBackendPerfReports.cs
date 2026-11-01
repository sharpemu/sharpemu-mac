// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using SharpEmu.Gpu.Vulkan.Buffers;
using SharpEmu.Gpu.Vulkan.Presentation;
using SharpEmu.Gpu.Vulkan.Scheduling;
using SharpEmu.HLE.Diagnostics;
using SharpEmu.Libs.VideoOut;

namespace SharpEmu.Gpu.Vulkan;

// The parts of the per-window render report that read the Vulkan backend's caches, barriers, slices
// and relay. The shared profile (SharpEmu.Libs) prints its own half of each line and appends these
// through RenderPhaseProfile.RegisterReportProvider.
internal static class VulkanBackendPerfReports
{
    public static void Register()
    {
        RenderPhaseProfile.RegisterReportProvider(TakeBufferTotalsReport);
        RenderPhaseProfile.RegisterReportProvider(TakeDirtyTrackingReport);
        RenderPhaseProfile.RegisterReportProvider(TakeSyncPointReport);
        RenderPhaseProfile.RegisterReportProvider(BarrierTally.TakeReport);
        RenderPhaseProfile.RegisterReportProvider(VulkanVideoPresenter.TakeSliceReport);
        RenderPhaseProfile.RegisterReportProvider(VulkanVideoPresenter.TakeStagePreparationReport);
        RenderPhaseProfile.RegisterReportProvider(ContentUploadArena.TakeReport);
        RenderPhaseProfile.RegisterReportProvider(TakeStreamReport);
    }

    private static string TakeStreamReport() =>
        "[PERF][FAST_PATHS_VULKAN] " +
        FormattableString.Invariant($"stream_cleaned={Interlocked.Exchange(ref GuestBufferCache.StreamCleaned, 0)} stream_uses={Interlocked.Exchange(ref GuestBufferCache.StreamUses, 0)} ") +
        FormattableString.Invariant($"gpu_version_bumps={GuestBufferCache.TakeGpuVersionBumps()}");

    private static string TakeDirtyTrackingReport() =>
        "[PERF][DIRTY_TRACKING_VULKAN] " + GuestBufferCache.TakeDirtyTrackingReport() + " " + GpuBuffer.CopyRun.TakeReport();

    private static string TakeSyncPointReport() =>
        "[PERF][SYNC_POINTS_VULKAN] " + GpuWorkerRelay.TakeRelayWaitReport();

    private static string TakeBufferTotalsReport() =>
        "[PERF][BUFFER_SYNC_TOTAL_VULKAN] " +
        FormattableString.Invariant($"widened={Interlocked.Exchange(ref GuestBufferCache.WidenedUploads, 0)} widened_bytes={Interlocked.Exchange(ref GuestBufferCache.WidenedUploadBytes, 0)} ") +
        FormattableString.Invariant($"hot_skipped={SharpEmu.HLE.GpuMemory.TrackedRegion.TakeHotUploadsSkipped()} ") +
        FormattableString.Invariant($"readbacks=async:{Interlocked.Exchange(ref GuestBufferCache.ReadbacksStarted, 0)},sync:{Interlocked.Exchange(ref GuestBufferCache.ReadbacksSynchronous, 0)},no_writer:{Interlocked.Exchange(ref GuestBufferCache.ReadbacksNoWriter, 0)},too_large:{Interlocked.Exchange(ref GuestBufferCache.ReadbacksTooLarge, 0)},busy:{Interlocked.Exchange(ref GuestBufferCache.ReadbacksBusy, 0)},slot_waits:{Interlocked.Exchange(ref GuestBufferCache.ReadbacksSlotWaits, 0)},") +
        TakeDirectReadReport() + " " +
        FormattableString.Invariant($"relay_hop_us=begin:{AverageMicroseconds(GuestBufferCache.RelayHopBeginTicks, GuestBufferCache.RelayHopBegin):F1},complete:{AverageMicroseconds(GuestBufferCache.RelayHopCompleteTicks, GuestBufferCache.RelayHopComplete):F1} ") +
        TakeGuestReadbackReport();

    private static string TakeDirectReadReport() =>
        FormattableString.Invariant($"promoted={GuestBufferCache.Promotions.Take()}/{GuestBufferCache.PromotedBufferBytes.Take()},mapped_at_create:{GuestBufferCache.MappedCreations.Take()},budget_full:{GuestBufferCache.PromotionsBudgetFull.Take()},live_bytes:{Interlocked.Read(ref GuestBufferCache.PromotedLiveBytes)}");

    private static string TakeGuestReadbackReport()
    {
        var reads = GuestBufferCache.GuestReadbacks.Take();
        var behind = GuestBufferCache.GuestReadbacksBehind.Take();
        var ticksBehind = GuestBufferCache.GuestReadbackTicksBehind.Take();
        var age = GuestBufferCache.GuestReadbackFrameAge;
        return FormattableString.Invariant($"guest_reads={reads} read_us=queue:{PerSample(GuestBufferCache.GuestReadbackQueueTicks, reads):F1},begin:{PerSample(GuestBufferCache.GuestReadbackBeginTicks, reads):F1},gpu_wait:{PerSample(GuestBufferCache.GuestReadbackGpuWaitTicks, reads):F1} ") +
            FormattableString.Invariant($"read_behind={behind}/ticks:{(behind == 0 ? 0 : (double)ticksBehind / behind):F1} read_frame_age=0:{age[0].Take()},1:{age[1].Take()},2:{age[2].Take()},3+:{age[3].Take()} ") +
            BufferUploadProfile.TakeGuestFrameLeadReport() + " " +
            TakeReadWriterReport() + " " + SubmissionScheduler.TakeTickLedgerReport();
    }

    private static string TakeReadWriterReport()
    {
        var timed = GuestBufferCache.WriterTimedReads.Take();
        var held = GuestBufferCache.WriterHeldReads.Take();
        return FormattableString.Invariant($"read_writer=timed:{timed},writer_age_us:{PerSample(GuestBufferCache.WriterAgeTicks, timed):F1},since_submit_us:{PerSample(GuestBufferCache.WriterSinceSubmitTicks, timed):F1},") +
            FormattableString.Invariant($"held:{held}/us:{PerSample(GuestBufferCache.WriterHeldTicks, held):F1},submitted_by_read:{GuestBufferCache.WriterSubmittedByRead.Take()}");
    }

    private static double PerSample(ThreadCounter ticks, long samples)
    {
        var total = ticks.Take();
        return samples == 0 ? 0 : total * 1_000_000.0 / Stopwatch.Frequency / samples;
    }

    private static double AverageMicroseconds(ThreadCounter ticks, ThreadCounter count)
    {
        var total = ticks.Take();
        var samples = count.Take();
        return samples == 0 ? 0 : total * 1_000_000.0 / Stopwatch.Frequency / samples;
    }
}
