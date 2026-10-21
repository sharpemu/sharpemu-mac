// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Images;

namespace SharpEmu.Gpu.Metal.V2;

// The memory policy for the detiled copies: a budget re-read from the host, age-based
// retirement of images the title stopped using, and the working-set guards both obey.
// See MetalImageBudgetSettings for why the budget follows the host and not a share of RAM.
internal sealed partial class MetalImageCache
{
    // How long a dropped image's range is remembered: one recreated within this is a refault,
    // the number that says a budget or a retirement age is too tight.
    private static readonly long RefaultWindow = 30 * Stopwatch.Frequency;

    private MetalImageBudgetSettings _budgetSettings = null!;
    private MetalImageRetirementSettings _retirement = null!;
    private MetalImageIdleClock _idleClock = null!;
    private Func<MetalHostMemorySample> _sampleHost = null!;
    private Func<long> _clock = null!;
    private Func<long> _pressureEvents = null!;
    private ulong _memoryBudgetBytes;
    private MetalHostMemorySample _hostSample;
    private long _nextBudgetEvaluation;
    private long _seenPressureEvents;
    private long _nextHousekeeping;
    private long _budgetEvaluations;
    private long _retiredImages;
    private long _retiredBytes;
    private long _refaults;
    // Bytes of the images used in the last InUseFrames presented frames: as measured at the
    // last evaluation, and held (rises at once, falls by a quarter of the gap per evaluation).
    private ulong _inUseBytes;
    private ulong _inUseHeldBytes;
    private readonly Dictionary<(ulong Address, ulong Size), long> _recentlyDropped = new();

    public ulong MemoryBudgetBytes => Volatile.Read(ref _memoryBudgetBytes);

    /// <summary>Bytes of the in-use set at the last budget evaluation, before hysteresis.</summary>
    public ulong InUseBytes => Volatile.Read(ref _inUseBytes);

    /// <summary>Images and allocated bytes released for age since the cache was made.</summary>
    public (long Images, long Bytes) RetiredTotals => (Volatile.Read(ref _retiredImages), Volatile.Read(ref _retiredBytes));

    /// <summary>Images recreated within 30 s of being evicted or retired.</summary>
    public long Refaults => Volatile.Read(ref _refaults);

    /// <summary>
    /// Installs the budget and retirement policy and the host readings it uses, then sizes the
    /// budget once. The constructor installs the real host; tests install their own.
    /// </summary>
    internal void ConfigureMemoryPolicy(
        MetalImageBudgetSettings budget,
        MetalImageRetirementSettings retirement,
        Func<MetalHostMemorySample> sampleHost,
        Func<long> clock,
        Func<long> pressureEvents)
    {
        using var held = _lock.Hold();
        _budgetSettings = budget;
        _retirement = retirement;
        _sampleHost = sampleHost;
        _clock = clock;
        _pressureEvents = pressureEvents;
        var sampleInterval = Math.Max(Stopwatch.Frequency / 4, 1);
        _idleClock = new MetalImageIdleClock(
            Math.Max(retirement.Frames, retirement.InUseFrames),
            Math.Max(retirement.Seconds, retirement.BudgetMinimumIdleSeconds), sampleInterval);
        _inUseHeldBytes = 0;
        var now = clock();
        _idleClock.Sample(now, _collectionTick);
        _nextHousekeeping = now + Stopwatch.Frequency;
        EvaluateBudgetLocked(now);
    }

    private void EvaluateBudgetLocked(long now)
    {
        _seenPressureEvents = _pressureEvents();
        _nextBudgetEvaluation = now + _budgetSettings.EvaluationInterval;
        MeasureInUseLocked();
        if (_budgetSettings.FixedBudget is { } fixedBudget)
        {
            Volatile.Write(ref _memoryBudgetBytes, fixedBudget);
            return;
        }

        _hostSample = _sampleHost();
        Volatile.Write(ref _memoryBudgetBytes, _budgetSettings.Compute(in _hostSample, _totalUsedMemory, _inUseHeldBytes));
        _budgetEvaluations++;
    }

    // One pass over the images, once per evaluation: the bytes of the in-use set, and the
    // held value the WARN budget floors at. Caller holds the lock.
    private void MeasureInUseLocked()
    {
        var bytes = 0UL;
        _slots.ForEach((_, image) =>
        {
            if (image.Registered && IsInUseLocked(image))
            {
                bytes += image.CountedBytes;
            }
        });
        Volatile.Write(ref _inUseBytes, bytes);
        _inUseHeldBytes = bytes >= _inUseHeldBytes ? bytes : _inUseHeldBytes - (_inUseHeldBytes - bytes) / 4;
    }

    // The in-use set: an image used in the last InUseFrames presented frames, or in the frame
    // guard's working set. Before that many frames were presented every image is in it.
    private bool IsInUseLocked(MetalCachedImage image) =>
        !IsOutsideWorkingSet(image) || image.LastAccessTick >= _idleClock.TickFramesAgo(_retirement.InUseFrames);

    // Re-reads the host once per evaluation interval, and at once when the kernel's pressure
    // level changed. Caller holds the lock.
    private void MaybeEvaluateBudgetLocked(long now)
    {
        if (now >= _nextBudgetEvaluation || _pressureEvents() != _seenPressureEvents)
        {
            EvaluateBudgetLocked(now);
        }
    }

    // The frame guard: an image used by the frame being drawn or the one before it, or within
    // the last few collection ticks, is the working set. A 4K frame spans hundreds of ticks,
    // so a tick age alone evicted images the same frame was about to use again (Demon's Souls
    // at a 3 GB budget: 64 MB targets written back and re-uploaded every frame, 0.5 fps).
    private bool IsOutsideWorkingSet(MetalCachedImage image) =>
        image.Registered &&
        image.LastAccessTick < _previousFrameStartTick &&
        _collectionTick - Math.Min(_collectionTick, image.LastAccessTick) > TicksBeforeRemoval;

    // The tick before which an image has been idle for at least this long; zero (nothing
    // qualifies) until the cache has been running that long.
    private ulong TickIdleFor(long now, double seconds) =>
        _idleClock.TickAtOrBefore(now - (long)(seconds * Stopwatch.Frequency));

    /// <summary>Whether the age policy would release this image now. Caller holds the lock.</summary>
    internal bool IsRetirementEligibleLocked(MetalCachedImage image, long now)
    {
        if (!_retirement.Enabled || !IsOutsideWorkingSet(image))
        {
            return false;
        }

        var cutoff = Math.Min(_idleClock.TickFramesAgo(_retirement.Frames), TickIdleFor(now, _retirement.Seconds));
        return image.LastAccessTick < cutoff;
    }

    /// <summary>Whether an image the GPU never wrote, already behind guest memory and unwatched,
    /// has been idle long enough to go (see <see cref="MetalImageRetirementSettings.StaleSeconds"/>).
    /// Caller holds the lock.</summary>
    internal bool IsStaleLocked(MetalCachedImage image, long now) =>
        _retirement.StaleSeconds > 0 && !image.IsGpuModified && image.IsCpuDirty && !image.IsWatched &&
        IsOutsideWorkingSet(image) && image.LastAccessTick < TickIdleFor(now, _retirement.StaleSeconds);

    /// <summary>Whether budget eviction may take this image now. Caller holds the lock.</summary>
    internal bool IsBudgetEvictableLocked(MetalCachedImage image, long now)
    {
        if (!IsOutsideWorkingSet(image))
        {
            return false;
        }

        // Under critical pressure only the frame guard holds: compression and swap cost more
        // than re-uploading an image used a few frames ago.
        if (_hostSample.Pressure == MetalMemoryPressure.Critical)
        {
            return true;
        }

        // Otherwise only an image outside the in-use set goes, and only once it has been idle
        // for the minimum time as well.
        if (IsInUseLocked(image))
        {
            return false;
        }

        return _retirement.BudgetMinimumIdleSeconds <= 0 ||
            image.LastAccessTick < TickIdleFor(now, _retirement.BudgetMinimumIdleSeconds);
    }

    // Once a second: release the images past the retirement age and forget old drops.
    private void HousekeepLocked(long now)
    {
        if (now < _nextHousekeeping)
        {
            return;
        }

        _nextHousekeeping = now + Stopwatch.Frequency;
        if (_recentlyDropped.Count != 0)
        {
            List<(ulong, ulong)>? expired = null;
            foreach (var (key, droppedAt) in _recentlyDropped)
            {
                if (now - droppedAt > RefaultWindow)
                {
                    (expired ??= []).Add(key);
                }
            }

            if (expired is not null)
            {
                foreach (var key in expired)
                {
                    _recentlyDropped.Remove(key);
                }
            }
        }

        if (!_retirement.Enabled && _retirement.StaleSeconds <= 0)
        {
            return;
        }

        List<(ResourceSlotIdentifier Identifier, MetalImageReleaseCause Cause)>? release = null;
        _slots.ForEach((imageIdentifier, image) =>
        {
            if (IsRetirementEligibleLocked(image, now))
            {
                (release ??= []).Add((imageIdentifier, MetalImageReleaseCause.Retired));
            }
            else if (IsStaleLocked(image, now))
            {
                (release ??= []).Add((imageIdentifier, MetalImageReleaseCause.Stale));
            }
        });
        if (release is null)
        {
            return;
        }

        foreach (var (imageIdentifier, cause) in release)
        {
            if (_slots.TryGet(imageIdentifier) is not { } image)
            {
                continue;
            }

            var bytes = image.CountedBytes;
            ReleaseImage(imageIdentifier, cause);
            Interlocked.Increment(ref _retiredImages);
            Interlocked.Add(ref _retiredBytes, (long)bytes);
        }
    }

    // An image dropped for budget or age; remembered so its recreation counts as a refault.
    private void NoteDroppedLocked(MetalCachedImage image, MetalImageReleaseCause cause)
    {
        if (cause is MetalImageReleaseCause.Budget or MetalImageReleaseCause.Retired or MetalImageReleaseCause.Stale)
        {
            _recentlyDropped[(image.Description.Data.Address, image.Description.Data.Size)] = _clock();
        }
    }

    private void NoteRecreatedLocked(in ImageDescription description)
    {
        if (_recentlyDropped.Count != 0 &&
            _recentlyDropped.Remove((description.Data.Address, description.Data.Size), out var droppedAt) &&
            _clock() - droppedAt <= RefaultWindow)
        {
            Interlocked.Increment(ref _refaults);
        }
    }

    private static string PressureName(MetalMemoryPressure pressure) => pressure switch
    {
        MetalMemoryPressure.Normal => "normal",
        MetalMemoryPressure.Warn => "warn",
        MetalMemoryPressure.Critical => "critical",
        _ => "unknown",
    };
}
