// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Images;

namespace SharpEmu.Gpu.Metal.V2;

internal sealed partial class MetalImageCache
{
    private static readonly bool AliasTransitionsDefault =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_ALIAS_TRANSITIONS") != "0";

    /// <summary>Whether alias transitions fuse on the device and discard dead contents.</summary>
    internal bool AliasTransitionsEnabled { get; set; } = AliasTransitionsDefault;

    private long _aliasFused, _aliasFusedBytes, _aliasDiscarded, _aliasDiscardedBytes;
    // round-trip counts only pairs whose bytes went through guest memory: a run or the whole other image
    // published (and this one reloaded). A layout skip moves nothing and is counted on its own.
    private long _aliasRoundTrip, _aliasRoundTripBytes;
    private long _aliasPublishedRuns, _aliasPublishedWholes, _aliasLayoutSkips;
    private long _aliasMovedWritebackBytes, _aliasMovedBlitBytes;
    private long _aliasRunRefreshRefused;

    /// <summary>Test hook: GPU-written images that kept older texels over a stale run they could not reload alone.</summary>
    internal long AliasRunRefreshRefused => Interlocked.Read(ref _aliasRunRefreshRefused);

    private void NoteAliasRoundTrip(bool whole, ulong sharedSize)
    {
        _aliasRoundTrip++;
        _aliasRoundTripBytes += (long)sharedSize;
        if (whole)
        {
            _aliasPublishedWholes++;
        }
        else
        {
            _aliasPublishedRuns++;
        }
    }
    private long _aliasCpuReadRuns, _aliasCpuReadBytes;
    private readonly Dictionary<string, long> _aliasTransitionRefusals = new(StringComparer.Ordinal);

    private void RefuseAliasTransition(string reason) =>
        _aliasTransitionRefusals[reason] = _aliasTransitionRefusals.GetValueOrDefault(reason) + 1;

    internal (long Fused, long FusedBytes, long Discarded, long DiscardedBytes, long RoundTrip, long RoundTripBytes)
        TakeAliasTransitions() =>
        (Interlocked.Exchange(ref _aliasFused, 0), Interlocked.Exchange(ref _aliasFusedBytes, 0),
         Interlocked.Exchange(ref _aliasDiscarded, 0), Interlocked.Exchange(ref _aliasDiscardedBytes, 0),
         Interlocked.Exchange(ref _aliasRoundTrip, 0), Interlocked.Exchange(ref _aliasRoundTripBytes, 0));

    internal string TakeAliasTransitionText()
    {
        using var held = _lock.Hold();
        var counts = TakeAliasTransitions();
        string Mb(double bytes) => (bytes / 1048576.0).ToString("F2", CultureInfo.InvariantCulture);
        var refusals = string.Join(',', _aliasTransitionRefusals.OrderBy(p => p.Key).Select(p => $"{p.Key}:{p.Value}"));
        _aliasTransitionRefusals.Clear();
        var readRuns = Interlocked.Exchange(ref _aliasCpuReadRuns, 0);
        var readBytes = Interlocked.Exchange(ref _aliasCpuReadBytes, 0);
        var consumers = TakeConsumerRuns();
        DrainAliasSubsetChecksLocked();
        var subsetCopies = Interlocked.Exchange(ref _aliasSubsetCopies, 0);
        var subsetBytes = Interlocked.Exchange(ref _aliasSubsetBytes, 0);
        var layerSkips = TakeLayerRefreshSkipped();
        return $"alias_transitions=[fused:{counts.Fused},discarded:{counts.Discarded},device-subset:{subsetCopies}," +
            $"round-trip:{counts.RoundTrip},published-run:{Interlocked.Exchange(ref _aliasPublishedRuns, 0)}," +
            $"published-whole:{Interlocked.Exchange(ref _aliasPublishedWholes, 0)}," +
            $"layout-skip:{Interlocked.Exchange(ref _aliasLayoutSkips, 0)}] " +
            $"alias_mib=[fused:{Mb(counts.FusedBytes)},discarded:{Mb(counts.DiscardedBytes)},device-subset:{Mb(subsetBytes)}," +
            $"round-trip:{Mb(counts.RoundTripBytes)}] " +
            $"alias_moved_mib=[wb:{Mb(Interlocked.Exchange(ref _aliasMovedWritebackBytes, 0))}," +
            $"blit:{Mb(Interlocked.Exchange(ref _aliasMovedBlitBytes, 0))}] " +
            $"alias_run_refresh_refused={Interlocked.Exchange(ref _aliasRunRefreshRefused, 0)} {TakeRefusedStaleText()} " +
            $"{TakeAliasRunText()} " +
            $"layer_refresh_skipped=[{layerSkips.Layers},{Mb(layerSkips.Bytes)}] layer_stale_kept={TakeLayerStaleKept()} " +
            $"alias_subset_verify=[checked:{Interlocked.Exchange(ref _aliasSubsetVerified, 0)}," +
            $"mismatch:{Interlocked.Read(ref _aliasSubsetMismatches)},unverified:{Interlocked.Exchange(ref _aliasSubsetUnverified, 0)}] " +
            $"alias_transition_refused=[{refusals}] alias_cpu_read_runs={readRuns} " +
            $"alias_cpu_read_requested_mib={Mb(readBytes)} " +
            $"consumer_runs=[published:{consumers.Runs},skipped:{consumers.AlreadyPublished},gpu_writes_kept:{consumers.GpuWritesKept}] " +
            $"consumer_run_mib={Mb(consumers.RunBytes)} upload_gathers={TakeUploadGathers()}";
    }

    // The pass replaces every visible texel of this image. Refuse when an alias owns any
    // byte outside it: partial ownership remains with the established run machinery.
    private bool TryDiscardOverwrittenAliases(ResourceSlotIdentifier identifier)
    {
        var image = _slots[identifier];
        var span = image.Description.Data;
        foreach (var otherId in FindImagesInRange(span.Address, span.Size, pageOverlap: false))
        {
            if (otherId == identifier || _slots.TryGet(otherId) is not { } other) continue;
            var data = other.Description.Data;
            if (other.Description.HasStencil || data.Address < span.Address ||
                data.Size > span.Size || data.Address - span.Address > span.Size - data.Size ||
                !_transfer.CanDiscardAliasForOverwrite(image.Description, other.Description))
            {
                RefuseAliasTransition("discard-partial-owner");
                return false;
            }
        }
        NoteGpuWrittenImageLocked(image);
        foreach (var otherId in FindImagesInRange(span.Address, span.Size, pageOverlap: false))
        {
            if (otherId == identifier || _slots.TryGet(otherId) is not { } other) continue;
            if (other.IsGpuModified)
            {
                _aliasDiscarded++;
                _aliasDiscardedBytes += (long)other.Description.Data.Size;
            }
            other.ClearGpuModified();
            other.MarkCpuDirty(MetalImageDirtyCause.AliasWrite);
        }
        return true;
    }

    // Exact managed stores replace the full allocation, including padding and stencil.
    // The caller has already waited for in-flight writebacks and holds the cache lock
    // through invalidation. Native faults cannot make this coverage claim.
    private void DiscardFullyOverwrittenByCpuLocked(ulong address, ulong size)
    {
        foreach (var id in FindImagesInRange(address, size, pageOverlap: false))
        {
            if (_slots.TryGet(id) is not { IsGpuModified: true } image) continue;
            bool Covers(GuestSpan span) => span.Address >= address && span.Size <= size &&
                span.Address - address <= size - span.Size;
            if (!Covers(image.Description.Data) ||
                (image.Description.HasStencil && !Covers(image.Description.Stencil))) continue;
            image.ClearGpuModified();
            image.MarkCpuDirty(MetalImageDirtyCause.CpuWrite);
            _aliasDiscarded++;
            _aliasDiscardedBytes += (long)(image.Description.Data.Size +
                (image.Description.HasStencil ? image.Description.Stencil.Size : 0));
        }
    }
}
