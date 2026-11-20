// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Libs.Gpu.Rendering;

// Consecutive dispatches whose shader access barriers the host deferred (TryDeferShaderAccessBarrier):
// a dispatch joins the chain without a hazard barrier when it touches nothing an earlier member
// wrote and writes nothing an earlier member touched. A host that records every barrier at once
// resets the chain after each dispatch, so every dispatch with storage writes takes its barrier.
internal sealed class DispatchChain
{
    private const int MaxSpans = 1024;
    private const ulong IndirectArgumentBytes = 12;

    private readonly List<GuestSpan> _touched = new();
    private readonly List<GuestSpan> _written = new();
    private readonly List<GuestSpan> _recordTouched = new();
    private readonly List<GuestSpan> _recordWritten = new();
    private bool _open;
    private bool _conservative;
    private bool _readImages;
    private bool _wroteImages;
    private bool _recordConservative;
    private bool _recordReadsImages;
    private bool _recordWritesImages;
    private long _promotions;

    // The dispatches recorded since the last one the stream ordered (DispatchOverlap): those an overlap-marked dispatch
    // runs beside on a host that honours the mark. Kept apart from the deferred-barrier chain above, which a host that
    // records its barriers at once resets after every dispatch.
    private readonly List<GuestSpan> _groupTouched = new();
    private readonly List<GuestSpan> _groupWritten = new();
    private bool _groupConservative;
    private bool _groupReadImages;
    private bool _groupWroteImages;

    /// <summary>
    /// Called after <see cref="NeedsHazardBarrier"/> for the same dispatch: whether it may still run beside the dispatches
    /// recorded since the last ordered one, as the stream marked it (<paramref name="marked"/>). The console runs such
    /// dispatches together too, so a guest only leaves the barrier out where it believes they share nothing; but the
    /// host's dispatches do not take the console's time, and a dependency the console's timing hides (a small producer
    /// finishing before its consumer reads) becomes a race here. A marked dispatch that touches what one of them wrote,
    /// writes what one of them touched, or whose accesses (or theirs) are not all known is ordered after them instead,
    /// and counted (<see cref="DispatchChainTally.OverlapsOrdered"/>). SHARPEMU_DISPATCH_OVERLAP_HAZARDS=0 runs every
    /// marked dispatch beside the ones before it again.
    /// </summary>
    public bool KeepsOverlap(bool marked)
    {
        if (!marked || !OverlapHazardsChecked)
        {
            ResetGroup();
            AddToGroup();
            return marked;
        }

        var unknown = _groupConservative || _recordConservative || _groupTouched.Count > MaxSpans;
        var hazard = !unknown &&
            ((_recordReadsImages && _groupWroteImages) || (_recordWritesImages && (_groupWroteImages || _groupReadImages)) ||
             Overlaps(_recordTouched, _groupWritten) || Overlaps(_recordWritten, _groupTouched));
        if (unknown || hazard)
        {
            DispatchChainTally.NoteOverlapOrdered(unknown);
            ResetGroup();
            AddToGroup();
            return false;
        }

        AddToGroup();
        return true;
    }

    /// <summary>A dispatch recorded without the mark whose own ordering the host decides: kept in the group, never ending it.</summary>
    public void JoinOverlapGroup() => AddToGroup();

    internal static bool OverlapHazardsChecked { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_DISPATCH_OVERLAP_HAZARDS") != "0";

    private void ResetGroup()
    {
        _groupTouched.Clear();
        _groupWritten.Clear();
        _groupConservative = false;
        _groupReadImages = false;
        _groupWroteImages = false;
    }

    private void AddToGroup()
    {
        _groupTouched.AddRange(_recordTouched);
        _groupWritten.AddRange(_recordWritten);
        _groupConservative |= _recordConservative;
        _groupReadImages |= _recordReadsImages;
        _groupWroteImages |= _recordWritesImages;
    }

    public bool NeedsHazardBarrier(ResolvedDispatch record, in ShaderStageResources stage, long promotions)
    {
        if (_open && promotions != _promotions)
        {
            Reset();
        }

        Collect(record, in stage);
        if (!_open)
        {
            DispatchChainTally.Count(record.HasStorageWrites ? ChainDecision.Start : ChainDecision.Open);
            return record.HasStorageWrites;
        }

        if (_conservative || _recordConservative || _touched.Count > MaxSpans)
        {
            DispatchChainTally.Count(ChainDecision.Conservative);
            Reset();
            return true;
        }

        if ((_recordReadsImages && _wroteImages) || (_recordWritesImages && (_wroteImages || _readImages)) ||
            Overlaps(_recordTouched, _written) || Overlaps(_recordWritten, _touched))
        {
            DispatchChainTally.Count(ChainDecision.Overlap);
            Reset();
            return true;
        }

        DispatchChainTally.Count(ChainDecision.Skipped);
        return false;
    }

    public void Commit(bool deferred, long promotions)
    {
        if (!deferred)
        {
            Reset();
            return;
        }

        _touched.AddRange(_recordTouched);
        _written.AddRange(_recordWritten);
        _readImages |= _recordReadsImages;
        _wroteImages |= _recordWritesImages;
        _conservative |= _recordConservative;
        _open = true;
        _promotions = promotions;
    }

    private void Reset()
    {
        _touched.Clear();
        _written.Clear();
        _open = false;
        _conservative = false;
        _readImages = false;
        _wroteImages = false;
    }

    private void Collect(ResolvedDispatch record, in ShaderStageResources stage)
    {
        _recordTouched.Clear();
        _recordWritten.Clear();
        _recordConservative = false;
        _recordReadsImages = false;
        _recordWritesImages = false;
        if (stage.Program is not { } program)
        {
            _recordConservative = true;
            return;
        }

        var resources = stage.Resources;
        // Runtime callees and heap accesses can touch resources outside the snapshot below.
        // Keep their barriers until the chain can account for that complete access set.
        if (program.CallRecordCallees.Count != 0 || program.Bindings is { UsesBindlessHeap: true } ||
            stage.CallRecordRanges is { Length: > 0 })
        {
            _recordConservative = true;
            return;
        }

        if (program.Bindings?.Find(DescriptorBindingKind.GlobalDataShare) is not null)
        {
            _recordConservative = true;
        }

        var buffers = resources.Buffers;
        for (var index = 0; index < program.Buffers.Length && index < buffers.Length; index++)
        {
            var words = buffers[index];
            if (words.Length < 4)
            {
                continue;
            }

            var descriptor = BufferDescriptorWords.From(words);
            if (descriptor.Address == 0)
            {
                continue;
            }

            if (descriptor.Footprint() is not { } footprint || footprint == 0)
            {
                _recordConservative = true;
                continue;
            }

            var span = new GuestSpan(descriptor.Address, footprint);
            _recordTouched.Add(span);
            if (program.Buffers[index].Written)
            {
                _recordWritten.Add(span);
            }
        }

        var images = resources.Images;
        for (var index = 0; index < program.Images.Length && index < images.Length; index++)
        {
            var words = images[index];
            if (program.Images[index].Class == ImageResourceClass.None || words.Length < 4 || new TextureDescriptorWords(words).BaseAddress == 0)
            {
                continue;
            }

            _recordReadsImages = true;
            if (program.Images[index].Written)
            {
                _recordWritesImages = true;
            }
        }

        foreach (var range in resources.DeviceAddressRanges)
        {
            if (!range.Planned || range.Size == 0)
            {
                _recordConservative = true;
                continue;
            }

            var span = new GuestSpan(range.Base, range.Size);
            _recordTouched.Add(span);
            if (range.Written)
            {
                _recordWritten.Add(span);
            }
        }

        if (program.UsesDeviceAddresses && resources.DeviceAddressRanges.Length == 0)
        {
            _recordConservative = true;
        }

        if (record.IndirectArgumentsAddress != 0)
        {
            _recordTouched.Add(new GuestSpan(record.IndirectArgumentsAddress, IndirectArgumentBytes));
        }
    }

    private static bool Overlaps(List<GuestSpan> left, List<GuestSpan> right)
    {
        foreach (var span in left)
        {
            foreach (var other in right)
            {
                if (span.Address < other.End && other.Address < span.End)
                {
                    return true;
                }
            }
        }

        return false;
    }
}

public enum ChainDecision
{
    Open,
    Start,
    Skipped,
    Overlap,
    Conservative,
    Count,
}

// What DispatchChain decided, counted while the render phase profile runs; the GPU backends'
// sync reports take it (TakeReport).
public static class DispatchChainTally
{
    private static readonly long[] _decisions = new long[(int)ChainDecision.Count];

    public static void Count(ChainDecision decision)
    {
        if (VideoOut.RenderPhaseProfile.Enabled)
        {
            Interlocked.Increment(ref _decisions[(int)decision]);
        }
    }

    private static long _overlapsOrderedHazard;
    private static long _overlapsOrderedUnknown;

    /// <summary>Overlap-marked dispatches ordered after the ones before them (DispatchChain.KeepsOverlap), always counted.</summary>
    public static void NoteOverlapOrdered(bool unknown) =>
        Interlocked.Increment(ref unknown ? ref _overlapsOrderedUnknown : ref _overlapsOrderedHazard);

    /// <summary>Overlap-marked dispatches ordered for a shared range (Hazard) or for accesses not all known (Unknown).</summary>
    public static (long Hazard, long Unknown) OverlapsOrdered =>
        (Interlocked.Read(ref _overlapsOrderedHazard), Interlocked.Read(ref _overlapsOrderedUnknown));

    /// <summary>The overlap orderings since the last call, for a backend's report line.</summary>
    public static string TakeOverlapReport() =>
        FormattableString.Invariant($"overlap_ordered=[hazard:{Interlocked.Exchange(ref _overlapsOrderedHazard, 0)},unknown:{Interlocked.Exchange(ref _overlapsOrderedUnknown, 0)}]");

    public static string TakeReport() =>
        FormattableString.Invariant($"chain_open={Interlocked.Exchange(ref _decisions[(int)ChainDecision.Open], 0)} chain_start={Interlocked.Exchange(ref _decisions[(int)ChainDecision.Start], 0)} ") +
        FormattableString.Invariant($"chain_skipped={Interlocked.Exchange(ref _decisions[(int)ChainDecision.Skipped], 0)} chain_overlap={Interlocked.Exchange(ref _decisions[(int)ChainDecision.Overlap], 0)} chain_conservative={Interlocked.Exchange(ref _decisions[(int)ChainDecision.Conservative], 0)}");
}
