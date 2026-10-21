// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;

namespace SharpEmu.Gpu.Metal.V2;

// The kernel's memory-pressure level, as kern.memorystatus_vm_pressure_level and the
// DISPATCH_SOURCE_TYPE_MEMORYPRESSURE source report it (DISPATCH_MEMORYPRESSURE_NORMAL,
// _WARN, _CRITICAL).
internal enum MetalMemoryPressure
{
    Unknown = 0,
    Normal = 1,
    Warn = 2,
    Critical = 4,
}

/// <summary>
/// One reading of host memory. <see cref="AvailableBytes"/> is what the kernel can hand out
/// without compressing or swapping anything: free pages (speculative included) plus pageable
/// file-backed pages, the sum XNU's memorystatus itself compares against its pressure
/// thresholds (vm_page_free_count + vm_page_pageable_external_count; purgeable pages are left
/// out when the compressor is on). <see cref="ProcessFootprintBytes"/> is the process's
/// phys_footprint, reported for telemetry only: the budget does not use it (see
/// MetalImageBudgetSettings).
/// </summary>
internal readonly record struct MetalHostMemorySample(
    ulong PhysicalBytes,
    ulong AvailableBytes,
    ulong ProcessFootprintBytes,
    MetalMemoryPressure Pressure);

/// <summary>
/// The budget policy for the detiled image copies. On a PS5 the GPU samples a title's tiled
/// bytes in place, so the guest's own memory is the only copy the console has; the detiled
/// MTLTextures are purely an emulator cost on top of it. The title fills its own budget
/// (Demon's Souls streams textures until ~300 MiB of its direct memory is free), so the image
/// cache has to live in whatever the host has left, not in a fixed fraction of RAM:
///
///   budget   = clamp(cache + available - reserve, floor, ceiling)
///                         how far the cache may grow before the kernel starts compressing
///                         (shrinks by the shortfall when available is below the reserve)
///   pressure WARN:     budget = max(min(budget, in_use + in_use/8), in_use)
///   pressure CRITICAL: budget = floor
///
/// The inputs are the cache's own allocated bytes and system-wide page counts, not this
/// process's phys_footprint. The footprint charges a shared-memory page once per mapping
/// that faulted it (one 512 MB touch through three views of one shm object reads 1.5 GB,
/// artifacts/phase1/host-heaps.md section 3), and SharpEmu maps guest memory through a
/// canonical alias and per-range views; it also counts the process's compressed pages,
/// so a footprint-based term reads worst exactly when the kernel is compressing guest
/// memory. task_vm_info's internal/compressed split has the same per-mapping charge.
/// host_statistics64 counts each physical page once, leaves compressed pages out by
/// construction, and is already read for <c>available</c>: no extra cost.
///
/// in_use is the bytes of the images used in the last
/// <see cref="MetalImageRetirementSettings.InUseFrames"/> presented frames, measured at each
/// evaluation, rising at once and falling by a quarter of the gap per evaluation. Under
/// WARN the cache sheds everything outside that set and no more: halving relative to the
/// cache itself was a ratchet a cache whose every image is used each frame could never meet.
///
/// Eviction keeps its working-set and in-use guards whatever the budget says (see
/// MetalImageCache), so a budget below the in-use set runs over rather than thrashing.
/// </summary>
internal sealed record MetalImageBudgetSettings
{
    private const ulong MiB = 1024 * 1024;
    private const ulong GiB = 1024 * MiB;

    /// <summary>SHARPEMU_METAL_V2_IMAGE_BUDGET_MB: a fixed budget, no host readings.</summary>
    public ulong? FixedBudget { get; init; }

    /// <summary>The budget never drops below this: images used within seconds stay.</summary>
    public ulong Floor { get; init; }

    /// <summary>The budget never exceeds this, however much memory is free.</summary>
    public ulong Ceiling { get; init; }

    /// <summary>The available memory the cache leaves alone; below it the cache shrinks.</summary>
    public ulong Reserve { get; init; }

    /// <summary>How often the budget is re-read from the host, in Stopwatch ticks.</summary>
    public long EvaluationInterval { get; init; } = Stopwatch.Frequency;

    /// <summary>
    /// Defaults for a machine of this much physical memory. Floor 1 GiB (a 1080p frame's
    /// render targets plus a few hundred MB of textures). Ceiling a quarter of RAM up to 8 GiB,
    /// the old fixed budget's upper end. Reserve a sixteenth of RAM, at least 1 GiB: the free
    /// plus file-backed pages left before the compressor would start on anonymous memory
    /// (guest memory included). What macOS and other processes hold is already outside
    /// <c>available</c>, so it needs no headroom of its own.
    /// </summary>
    public static MetalImageBudgetSettings Defaults(ulong physicalBytes)
    {
        var floor = 1 * GiB;
        return new MetalImageBudgetSettings
        {
            Floor = floor,
            Ceiling = Math.Clamp(physicalBytes / 4, floor, 8 * GiB),
            Reserve = Math.Max(physicalBytes / 16, 1 * GiB),
        };
    }

    /// <summary>The defaults, with SHARPEMU_METAL_V2_IMAGE_BUDGET_MB (at least 256) as a fixed budget.</summary>
    public static MetalImageBudgetSettings FromEnvironment(ulong physicalBytes)
    {
        var settings = Defaults(physicalBytes);
        var configured = Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_IMAGE_BUDGET_MB");
        return ulong.TryParse(configured, out var megabytes) && megabytes >= 256
            ? settings with { FixedBudget = megabytes * MiB }
            : settings;
    }

    /// <summary>
    /// The budget for a cache holding <paramref name="cacheBytes"/> of textures, of which
    /// <paramref name="inUseBytes"/> were used in the last few presented frames, given this
    /// host reading.
    /// </summary>
    public ulong Compute(in MetalHostMemorySample sample, ulong cacheBytes, ulong inUseBytes)
    {
        if (FixedBudget is { } fixedBudget)
        {
            return fixedBudget;
        }

        var ceiling = Math.Max(Ceiling, Floor);
        if (sample.PhysicalBytes == 0)
        {
            // No reading at all: nothing to size against but the ceiling.
            return ceiling;
        }

        ulong budget;
        if (sample.AvailableBytes >= Reserve)
        {
            var spare = sample.AvailableBytes - Reserve;
            budget = cacheBytes > ulong.MaxValue - spare ? ulong.MaxValue : cacheBytes + spare;
        }
        else
        {
            var shortfall = Reserve - sample.AvailableBytes;
            budget = cacheBytes > shortfall ? cacheBytes - shortfall : 0;
        }

        budget = sample.Pressure switch
        {
            MetalMemoryPressure.Critical => 0,
            // Shed what is outside the in-use set, with an eighth of slack so the images the
            // title brings in next do not trigger a pass each; never below the set itself.
            MetalMemoryPressure.Warn => Math.Max(Math.Min(budget, inUseBytes + inUseBytes / 8), inUseBytes),
            _ => budget,
        };

        return Math.Clamp(budget, Floor, ceiling);
    }
}

/// <summary>
/// Age-based retirement settings: an image the title has not used for <see cref="Frames"/>
/// presented frames and <see cref="Seconds"/> seconds is released whatever the budget says,
/// so a long session cannot keep the detiled copies of areas, menus and cinematics it has
/// left. Both conditions hold at once: frames alone would retire a whole loading screen's
/// worth of images after a few seconds at a high frame rate, seconds alone would retire the
/// working set of a title that presents once every few seconds. Budget eviction separately
/// requires <see cref="BudgetMinimumIdleSeconds"/> of idleness (except under critical
/// pressure), so an image used every few frames, a shadow cascade or a probe updated
/// round-robin, is not dropped and re-uploaded on each pass. Outside critical pressure it
/// also requires the image to be outside the in-use set, the images used in the last
/// <see cref="InUseFrames"/> presented frames: counted in frames, because a title presenting
/// twice a second (Demon's Souls in-world) cycles through its textures over a few seconds,
/// and a 2 s idle rule evicted images that came back within 30 s 62% of the time.
/// </summary>
internal sealed record MetalImageRetirementSettings
{
    public uint Frames { get; init; } = 600;

    public double Seconds { get; init; } = 20;

    public double BudgetMinimumIdleSeconds { get; init; } = 2;

    public uint InUseFrames { get; init; } = 30;

    /// <summary>
    /// How long an image the GPU never wrote, already behind guest memory and no longer
    /// watched, stays once nothing uses it (0 keeps it until the age policy takes it). Such an
    /// image is uploaded again whatever a later use does, and in the meantime every CPU store
    /// into its pages takes the cache lock to mark it stale once more: a title that builds its
    /// command and constant buffers in memory a transient render target held earlier puts every
    /// store of its render threads behind the GPU worker. Frames are no measure here: at one
    /// frame a second the age policy's 600 frames are ten minutes.
    /// </summary>
    public double StaleSeconds { get; init; } = 2;

    public bool Enabled => Frames != 0 && Seconds > 0;

    /// <summary>
    /// SHARPEMU_METAL_V2_IMAGE_RETIRE_FRAMES and SHARPEMU_METAL_V2_IMAGE_RETIRE_SECONDS (either 0
    /// turns retirement off), SHARPEMU_METAL_V2_IMAGE_EVICT_MIN_IDLE_SECONDS,
    /// SHARPEMU_METAL_V2_IMAGE_INUSE_FRAMES, SHARPEMU_METAL_V2_IMAGE_STALE_SECONDS.
    /// </summary>
    public static MetalImageRetirementSettings FromEnvironment()
    {
        var settings = new MetalImageRetirementSettings();
        if (uint.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_IMAGE_RETIRE_FRAMES"), out var frames))
        {
            settings = settings with { Frames = frames };
        }

        if (double.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_IMAGE_RETIRE_SECONDS"),
                System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var seconds) &&
            seconds >= 0)
        {
            settings = settings with { Seconds = seconds };
        }

        if (double.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_IMAGE_EVICT_MIN_IDLE_SECONDS"),
                System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var idle) &&
            idle >= 0)
        {
            settings = settings with { BudgetMinimumIdleSeconds = idle };
        }

        if (uint.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_IMAGE_INUSE_FRAMES"), out var inUseFrames))
        {
            settings = settings with { InUseFrames = inUseFrames };
        }

        if (double.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_IMAGE_STALE_SECONDS"),
                System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var staleSeconds) &&
            staleSeconds >= 0)
        {
            settings = settings with { StaleSeconds = staleSeconds };
        }

        return settings;
    }
}

/// <summary>
/// Maps "N presented frames ago" and "T seconds ago" onto collection ticks, the clock every
/// cached image is stamped with on use. An image whose last-use tick is below the tick this
/// returns was last used before that point. Frame starts are kept for the last
/// <c>frames</c> frames; time samples are taken at most once per <c>sampleInterval</c> and
/// kept for <c>seconds</c> plus a margin. Both answers err late: an image is never called
/// idle before it is.
/// </summary>
internal sealed class MetalImageIdleClock
{
    private readonly ulong[] _frameStarts;
    private long _frames;
    private readonly (long Timestamp, ulong Tick)[] _samples;
    private int _sampleHead;
    private int _sampleCount;
    private readonly long _sampleInterval;

    public MetalImageIdleClock(uint frames, double seconds, long sampleInterval)
    {
        _frameStarts = new ulong[Math.Max(frames, 1u) + 1];
        _sampleInterval = Math.Max(sampleInterval, 1);
        var span = Math.Max(seconds, 0) * Stopwatch.Frequency;
        var capacity = (int)Math.Clamp(span / _sampleInterval + 4, 8, 1 << 16);
        _samples = new (long, ulong)[capacity];
    }

    public long Frames => _frames;

    /// <summary>A presented frame began at this tick.</summary>
    public void MarkFrame(ulong tick)
    {
        _frameStarts[(int)(_frames % _frameStarts.Length)] = tick;
        _frames++;
    }

    /// <summary>Records that the cache was at this tick at this time; samples closer together than the interval are skipped.</summary>
    public void Sample(long timestamp, ulong tick)
    {
        if (_sampleCount != 0)
        {
            var newest = _samples[(_sampleHead + _sampleCount - 1) % _samples.Length];
            if (timestamp - newest.Timestamp < _sampleInterval)
            {
                return;
            }
        }

        if (_sampleCount == _samples.Length)
        {
            _sampleHead = (_sampleHead + 1) % _samples.Length;
            _sampleCount--;
        }

        _samples[(_sampleHead + _sampleCount) % _samples.Length] = (timestamp, tick);
        _sampleCount++;
    }

    /// <summary>The tick the frame <paramref name="framesAgo"/> frames back began at; zero when fewer frames were seen.</summary>
    public ulong TickFramesAgo(uint framesAgo)
    {
        if (framesAgo >= _frameStarts.Length || (ulong)_frames < (ulong)framesAgo + 1)
        {
            return 0;
        }

        return _frameStarts[(int)((_frames - 1 - framesAgo) % _frameStarts.Length)];
    }

    /// <summary>The newest sampled tick at or before <paramref name="timestamp"/>; zero when no sample is that old.</summary>
    public ulong TickAtOrBefore(long timestamp)
    {
        var found = 0UL;
        for (var index = 0; index < _sampleCount; index++)
        {
            var sample = _samples[(_sampleHead + index) % _samples.Length];
            if (sample.Timestamp > timestamp)
            {
                break;
            }

            found = sample.Tick;
        }

        return found;
    }
}
