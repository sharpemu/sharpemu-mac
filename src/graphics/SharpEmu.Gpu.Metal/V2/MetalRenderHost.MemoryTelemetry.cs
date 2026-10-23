// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
using SharpEmu.Logging;

namespace SharpEmu.Gpu.Metal.V2;

internal sealed partial class MetalRenderHost
{
    private static readonly nint SelRecommendedMaxWorkingSetSize = MetalNative.Selector("recommendedMaxWorkingSetSize");
    private static readonly nint SelDeviceCurrentAllocatedSize = MetalNative.Selector("currentAllocatedSize");

    // Windows an encoder named because a device-address program may reach them, for
    // the command buffer being recorded; folded into the per-command-buffer census at commit.
    private readonly MetalWindowSet _commandBufferReachWindows = new();

    // SHARPEMU_LOG_MEMORY=1: the "metal", "windows" and "gpu-refs" sections of the [MEM] lines.
    private void RegisterMemoryTelemetry()
    {
        if (!MemoryTelemetry.Enabled)
        {
            return;
        }

        MemoryTelemetry.Register("metal", DescribeMetalMemory);
        MemoryTelemetry.Register("windows", () => MetalMemoryTelemetry.TakeWindowReport(_guestMemory));
        MemoryTelemetry.Register("gpu-refs", MetalGpuReferenceCensus.TakeReport);
        if (_singleCopy is { } singleCopy)
        {
            MemoryTelemetry.Register("singlecopy", () => singleCopy.Describe() + DescribeSingleCopyDeviceAddresses());
        }
    }

    private string DescribeMetalMemory()
    {
        const double Mb = 1048576.0;
        var recommended = _device == 0 ? 0UL : MetalV2Native.SendValue64(_device, SelRecommendedMaxWorkingSetSize);
        var allocated = _device == 0 ? 0UL : MetalV2Native.SendValue64(_device, SelDeviceCurrentAllocatedSize);
        return string.Create(CultureInfo.InvariantCulture,
            $"recommended_working_set_mb={recommended / Mb:F0} device_allocated_mb={allocated / Mb:F0} " +
            $"ring_mb={_guestMemory.RingBytes / Mb:F0} ring_peak_mb={_guestMemory.RingPeakBytes / Mb:F0} " +
            $"{DescribeStagingTotals()} {DescribeShaderCache()} {_imageCache.DescribeMemoryCensus()}");
    }

    // The tiler's staging pool since startup: held now, held by nobody (free at the last
    // trim), purgeable, and the high-water mark, plus how many slabs came and went.
    private string DescribeStagingTotals()
    {
        const double Mb = 1048576.0;
        var pool = StagingTiler?.Staging;
        if (pool is null)
        {
            return "staging_mb=0";
        }

        return string.Create(CultureInfo.InvariantCulture,
            $"staging_mb={pool.TotalBytes / Mb:F0} staging_idle_mb={pool.IdleBytes / Mb:F0} " +
            $"staging_volatile_mb={pool.VolatileBytes / Mb:F0} staging_peak_mb={pool.PeakBytes / Mb:F0} " +
            $"staging_slabs={pool.SlabCount} staging_created={pool.SlabsCreated} " +
            $"staging_released={pool.SlabsReleased} staging_purged={pool.SlabsPurged}");
    }

    // The PERF line's view: staging and rings now, and their peaks since the last line.
    private string DescribeStagingWindow()
    {
        const ulong Mb = 1024 * 1024;
        var pool = StagingTiler?.Staging;
        var ringPeak = _guestMemory.TakeWindowPeakRingBytes();
        if (pool is null)
        {
            return string.Create(CultureInfo.InvariantCulture,
                $"staging_mb=0 ring_mb={_guestMemory.RingBytes / Mb} ring_peak_mb={ringPeak / Mb}");
        }

        return string.Create(CultureInfo.InvariantCulture,
            $"staging_mb={pool.TotalBytes / Mb} staging_peak_mb={pool.TakeWindowPeakBytes() / Mb} " +
            $"staging_idle_mb={pool.IdleBytes / Mb} staging_volatile_mb={pool.VolatileBytes / Mb} " +
            $"ring_mb={_guestMemory.RingBytes / Mb} ring_peak_mb={ringPeak / Mb}");
    }

    // Feeds the on-screen performance display. Called on the window thread a few times a
    // second: BUF is the shared memory the staging pool and upload rings hold, IMG what the
    // image cache holds, and the device total stands in for the Vulkan allocation count.
    // Returns the queue depth for the display: guest submissions waiting, and command
    // buffers committed but not yet complete.
    internal (int Pending, int InFlight) PublishOverlayStatistics()
    {
        var staging = StagingTiler?.Staging?.TotalBytes ?? 0UL;
        SharpEmu.Libs.VideoOut.PerfOverlay.SetGuestCacheStatistics(
            _guestMemory.RingBytes + staging, _imageCache.TotalUsedMemory, 0, 0);
        if (_device != 0)
        {
            SharpEmu.Libs.VideoOut.PerfOverlay.SetDeviceMemoryStatistics(
                MetalV2Native.SendValue64(_device, SelDeviceCurrentAllocatedSize));
        }

        return (CommandStream.PendingSubmissionCount, Volatile.Read(ref _inFlightCount));
    }

    // Called with the command buffer about to be committed.
    private void NoteCommandBufferForTelemetry()
    {
        if (!MemoryTelemetry.Enabled)
        {
            return;
        }

        MetalMemoryTelemetry.NoteCommit(_guestMemory, _commandBufferReachWindows);
        _commandBufferReachWindows.Clear();
    }
}
