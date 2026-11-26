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

    // SHARPEMU_CHAIN_SAME_PROGRAM_WRITES (on unless "0"): inside a chain whose dispatches all run one program, a dispatch
    // whose only overlaps with the chain are plain writes over plain writes, or atomic accesses over atomic accesses,
    // takes no barrier. The guest orders compute dispatches only with its own wait and cache events, which end the chain
    // with a full barrier; it put none between these dispatches, so it does not rely on their order (an append through
    // an atomic counter is the common case). A read of a chain write, a write over a chain read, an atomic access against
    // a plain one, image conflicts and a different program keep the barrier.
    internal static bool SameProgramWrites { get; set; } =
        !string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_CHAIN_SAME_PROGRAM_WRITES"), "0", StringComparison.Ordinal);

    private readonly List<GuestSpan> _read = new();
    private readonly List<GuestSpan> _written = new();
    private readonly List<GuestSpan> _atomic = new();
    private readonly List<GuestSpan> _recordRead = new();
    private readonly List<GuestSpan> _recordWritten = new();
    private readonly List<GuestSpan> _recordAtomic = new();
    private bool _open;
    private bool _conservative;
    private bool _readImages;
    private bool _wroteImages;
    private bool _recordConservative;
    private bool _recordReadsImages;
    private bool _recordWritesImages;
    private long _promotions;
    private ulong _recordProgram;
    private ulong _program;
    private bool _oneProgram;

    // The chain's distinct spans (tests).
    internal int SpanCount => _read.Count + _written.Count + _atomic.Count;

    // The dispatches recorded since the last one the stream ordered (DispatchOverlap): those an overlap-marked dispatch
    // runs beside on a host that honours the mark. Kept apart from the deferred-barrier chain above, which a host that
    // records its barriers at once resets after every dispatch.
    private readonly List<GuestSpan> _groupTouched = new();
    private readonly List<GuestSpan> _groupWritten = new();
    private bool _groupConservative;
    private bool _groupReadImages;
    private bool _groupWroteImages;
    // The data each image the group's and this dispatch's shaders use covers (OverlapImageSpans), and whether one of them
    // has none the request builders resolve.
    private readonly List<GuestSpan> _groupImageRead = new();
    private readonly List<GuestSpan> _groupImageWritten = new();
    private bool _groupImagesUnresolved;
    // For the group's checks: the bytes read and written other than by atomics, and those of buffers every access of
    // which is an atomic (BufferResourceInfo.AtomicOnly), which are both read and written.
    private readonly List<GuestSpan> _groupRead = new();
    private readonly List<GuestSpan> _groupPlainWritten = new();
    private readonly List<GuestSpan> _groupAtomic = new();
    // This dispatch's spans for the group's checks, apart from the chain's (_recordRead, _recordWritten, _recordAtomic),
    // which class an atomic binding by any atomic access rather than by atomic-only access.
    private readonly List<GuestSpan> _groupRecordTouched = new();
    private readonly List<GuestSpan> _groupRecordWritten = new();
    private readonly List<GuestSpan> _groupRecordRead = new();
    private readonly List<GuestSpan> _recordPlainWritten = new();
    private readonly List<GuestSpan> _groupRecordAtomic = new();
    private readonly List<GuestSpan> _recordImageRead = new();
    private readonly List<GuestSpan> _recordImageWritten = new();
    private bool _recordImagesUnresolved;
    private GuestSpan _recordIndirect;

    /// <summary>
    /// Called after <see cref="NeedsHazardBarrier"/> for the same dispatch: whether it may still run beside the dispatches
    /// recorded since the last ordered one, as the stream marked it (<paramref name="marked"/>). The console runs such
    /// dispatches together too, so a guest only leaves the barrier out where it believes they share nothing; but the
    /// host's dispatches do not take the console's time, and a dependency the console's timing hides (a small producer
    /// finishing before its consumer reads) becomes a race here. A marked dispatch that touches what one of them wrote,
    /// writes what one of them touched, or whose accesses (or theirs) are not all known is ordered after them instead,
    /// and counted (<see cref="DispatchChainTally.OverlapsOrdered"/>, by cause in the report). An image meets another
    /// only where their data does (<see cref="OverlapImageSpans"/>), and two writes of the same buffer or image bytes that
    /// neither reads are not ordered (<see cref="OverlapWritesApart"/>), nor atomics against atomics on buffers both sides access
    /// solely through atomics (<see cref="OverlapAtomicsApart"/>). SHARPEMU_DISPATCH_OVERLAP_HAZARDS=0 runs every
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
        var imagesMeet = (_recordReadsImages && _groupWroteImages) || (_recordWritesImages && (_groupWroteImages || _groupReadImages));
        var cause = OverlapCause.None;
        if (unknown)
        {
            cause = OverlapCause.Unknown;
        }
        else if (_recordIndirect.Size != 0 && Overlaps(_recordIndirect, _groupWritten))
        {
            cause = OverlapCause.Indirect;
        }
        else if (Overlaps(_groupRecordRead, _groupPlainWritten) || Overlaps(_groupRecordRead, _groupAtomic) ||
                 Overlaps(_groupRecordAtomic, _groupPlainWritten))
        {
            cause = OverlapCause.BufferReadAfterWrite;
        }
        else if (Overlaps(_recordPlainWritten, _groupRead) || Overlaps(_recordPlainWritten, _groupAtomic) ||
                 Overlaps(_groupRecordAtomic, _groupRead))
        {
            cause = OverlapCause.BufferWriteAfterRead;
        }
        else if (Overlaps(_groupRecordAtomic, _groupAtomic))
        {
            // Atomics against atomics: the GPU performs each one whole, in whatever order the two dispatches reach the
            // bytes, as the console does with the two running together (an append counter both bump). Nothing else
            // either of them does touches those bytes, so they stay beside each other (OverlapAtomicsApart).
            if (OverlapAtomicsApart)
            {
                DispatchChainTally.NoteOverlapAtomicsApart();
            }
            else
            {
                cause = OverlapCause.BufferAtomics;
            }
        }
        else if (Overlaps(_recordPlainWritten, _groupPlainWritten))
        {
            // Writes of the same bytes with no read of them between leave whichever lands last on the console as well:
            // nothing reads a value the order decides, so they stay beside each other (OverlapWritesApart).
            if (OverlapWritesApart)
            {
                DispatchChainTally.NoteOverlapWritesApart();
            }
            else
            {
                cause = OverlapCause.BufferWriteAfterWrite;
            }
        }

        if (cause == OverlapCause.None && imagesMeet)
        {
            // Two images meet only where their data does; one whose data is not resolved meets every other. Images both
            // sides only write meet as two buffer writes do (OverlapWritesApart).
            cause = !OverlapImageSpans || _recordImagesUnresolved || _groupImagesUnresolved ? OverlapCause.ImageUnresolved
                : Overlaps(_recordImageRead, _groupImageWritten) || Overlaps(_recordImageWritten, _groupImageRead)
                    ? OverlapCause.Image
                : Overlaps(_recordImageWritten, _groupImageWritten) && !OverlapWritesApart ? OverlapCause.ImageWriteAfterWrite
                : OverlapCause.None;
            if (cause == OverlapCause.None)
            {
                DispatchChainTally.NoteOverlapImagesApart();
            }
        }

        if (cause != OverlapCause.None)
        {
            DispatchChainTally.NoteOverlapOrdered(cause);
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

    // SHARPEMU_DISPATCH_OVERLAP_IMAGE_SPANS (on by default, =0 off): an image read or written beside one written meets it
    // only where their data meets (the span the request builders resolve, as RenderExecutor.MarkStageWrites marks for a
    // storage image); off, any image a dispatch writes meets every image another one beside it uses, which orders most
    // marked dispatches of a frame that writes images (Demon's Souls: ~290 of ~640 per frame).
    // SHARPEMU_DISPATCH_OVERLAP_WRITES_APART (on by default, =0 off): a marked dispatch whose buffers meet the group's only
    // where both write (no read on either side of the shared bytes) stays beside it; off, it is ordered after it.
    internal static bool OverlapWritesApart { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_DISPATCH_OVERLAP_WRITES_APART") != "0";

    // SHARPEMU_DISPATCH_OVERLAP_ATOMICS_APART (on by default, =0 off): a marked dispatch whose buffers meet the group's
    // only where both sides access buffers solely through atomics stays beside it; off, it is ordered after it.
    internal static bool OverlapAtomicsApart { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_DISPATCH_OVERLAP_ATOMICS_APART") != "0";

    internal static bool OverlapImageSpans { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_DISPATCH_OVERLAP_IMAGE_SPANS") != "0";

    private void ResetGroup()
    {
        _groupTouched.Clear();
        _groupWritten.Clear();
        _groupRead.Clear();
        _groupPlainWritten.Clear();
        _groupAtomic.Clear();
        _groupConservative = false;
        _groupReadImages = false;
        _groupWroteImages = false;
        _groupImageRead.Clear();
        _groupImageWritten.Clear();
        _groupImagesUnresolved = false;
    }

    private void AddToGroup()
    {
        _groupTouched.AddRange(_groupRecordTouched);
        _groupWritten.AddRange(_groupRecordWritten);
        _groupRead.AddRange(_groupRecordRead);
        _groupPlainWritten.AddRange(_recordPlainWritten);
        _groupAtomic.AddRange(_groupRecordAtomic);
        _groupConservative |= _recordConservative;
        _groupReadImages |= _recordReadsImages;
        _groupWroteImages |= _recordWritesImages;
        _groupImageRead.AddRange(_recordImageRead);
        _groupImageWritten.AddRange(_recordImageWritten);
        _groupImagesUnresolved |= _recordImagesUnresolved;
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

        if (_conservative || _recordConservative || _read.Count > MaxSpans || _written.Count > MaxSpans || _atomic.Count > MaxSpans)
        {
            DispatchChainTally.Count(ChainDecision.Conservative);
            Reset();
            return true;
        }

        var images = (_recordReadsImages && _wroteImages) || (_recordWritesImages && (_wroteImages || _readImages));
        // An atomic access against a plain one is ordered like any other read and write.
        var readAfterWrite = Overlaps(_recordRead, _written) || Overlaps(_recordRead, _atomic) || Overlaps(_recordAtomic, _written);
        var writeAfterRead = Overlaps(_recordWritten, _read) || Overlaps(_recordWritten, _atomic) || Overlaps(_recordAtomic, _read);
        if (!images && !readAfterWrite && !writeAfterRead)
        {
            if (!Overlaps(_recordWritten, _written) && !Overlaps(_recordAtomic, _atomic))
            {
                DispatchChainTally.Count(ChainDecision.Skipped);
                return false;
            }

            // A program without a hash is never "the same program".
            if (SameProgramWrites && _oneProgram && _recordProgram != 0 && _recordProgram == _program)
            {
                DispatchChainTally.Count(ChainDecision.SameProgramWrites);
                return false;
            }
        }

        DispatchChainTally.Count(ChainDecision.Overlap);
        Reset();
        return true;
    }

    public void Commit(bool deferred, long promotions)
    {
        if (!deferred)
        {
            Reset();
            return;
        }

        if (!_open)
        {
            _program = _recordProgram;
            _oneProgram = true;
        }
        else if (_recordProgram != _program)
        {
            _oneProgram = false;
        }

        AddMerged(_read, _recordRead);
        AddMerged(_written, _recordWritten);
        AddMerged(_atomic, _recordAtomic);
        _readImages |= _recordReadsImages;
        _wroteImages |= _recordWritesImages;
        _conservative |= _recordConservative;
        _open = true;
        _promotions = promotions;
    }

    private void Reset()
    {
        _read.Clear();
        _written.Clear();
        _atomic.Clear();
        _open = false;
        _conservative = false;
        _readImages = false;
        _wroteImages = false;
        _oneProgram = false;
    }

    private void Collect(ResolvedDispatch record, in ShaderStageResources stage)
    {
        _recordRead.Clear();
        _recordWritten.Clear();
        _recordAtomic.Clear();
        _groupRecordTouched.Clear();
        _groupRecordWritten.Clear();
        _groupRecordRead.Clear();
        _recordPlainWritten.Clear();
        _groupRecordAtomic.Clear();
        _recordConservative = false;
        _recordReadsImages = false;
        _recordWritesImages = false;
        _recordImageRead.Clear();
        _recordImageWritten.Clear();
        _recordImagesUnresolved = false;
        _recordIndirect = default;
        _recordProgram = 0;
        if (stage.Program is not { } program)
        {
            _recordConservative = true;
            return;
        }

        _recordProgram = program.Hash;
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

            // An atomic binding is its own kind. Otherwise only a buffer the program purely writes is not a read:
            // scalar loads and anything not marked written count as reads.
            var span = new GuestSpan(descriptor.Address, footprint);
            var info = program.Buffers[index];
            if (info.Atomic)
            {
                _recordAtomic.Add(span);
            }
            else
            {
                if (!info.Written || info.Read || info.Scalar)
                {
                    _recordRead.Add(span);
                }

                if (info.Written)
                {
                    _recordWritten.Add(span);
                }
            }

            // The overlap group's view of the same buffer (KeepsOverlap): only a buffer every access of which is an
            // atomic is the group's atomic kind.
            _groupRecordTouched.Add(span);
            if (info.Written)
            {
                _groupRecordWritten.Add(span);
            }

            if (info.AtomicOnly)
            {
                _groupRecordAtomic.Add(span);
                continue;
            }

            if (info.Written)
            {
                _recordPlainWritten.Add(span);
            }

            // A buffer the shader neither reads nor writes per its info counts as read.
            if (info.Read || info.Atomic || !info.Written)
            {
                _groupRecordRead.Add(span);
            }
        }

        var images = resources.Images;
        var shapes = program.Resources?.Info.Images;
        var resolveImages = OverlapHazardsChecked && OverlapImageSpans;
        for (var index = 0; index < program.Images.Length && index < images.Length; index++)
        {
            var words = images[index];
            if (program.Images[index].Class == ImageResourceClass.None || words.Length < 4 || new TextureDescriptorWords(words).BaseAddress == 0)
            {
                continue;
            }

            var written = program.Images[index].Written;
            _recordReadsImages = true;
            if (written)
            {
                _recordWritesImages = true;
            }

            if (!resolveImages)
            {
                continue;
            }

            if (shapes is not null && index < shapes.Count &&
                RenderExecutor.TryGetImageData(program, shapes[index], words, index, out var data))
            {
                // An image the shader only stores to is not read (ImageResource.Read: loads, samples and atomics).
                if (!written || shapes[index].Read || shapes[index].Atomic)
                {
                    _recordImageRead.Add(data);
                }

                if (written)
                {
                    _recordImageWritten.Add(data);
                }
            }
            else
            {
                _recordImagesUnresolved = true;
            }
        }

        // Device-address ranges always count as reads; written ones as writes too.
        foreach (var range in resources.DeviceAddressRanges)
        {
            if (!range.Planned || range.Size == 0)
            {
                _recordConservative = true;
                continue;
            }

            var span = new GuestSpan(range.Base, range.Size);
            _recordRead.Add(span);
            _groupRecordTouched.Add(span);
            _groupRecordRead.Add(span);
            if (range.Written)
            {
                _recordWritten.Add(span);
                _groupRecordWritten.Add(span);
                _recordPlainWritten.Add(span);
            }
        }

        if (program.UsesDeviceAddresses && resources.DeviceAddressRanges.Length == 0)
        {
            _recordConservative = true;
        }

        if (record.IndirectArgumentsAddress != 0)
        {
            _recordIndirect = new GuestSpan(record.IndirectArgumentsAddress, IndirectArgumentBytes);
            _recordRead.Add(_recordIndirect);
            _groupRecordTouched.Add(_recordIndirect);
            _groupRecordRead.Add(_recordIndirect);
        }
    }

    // Adds the spans, widening an overlapping or adjacent one instead of growing the list, so a long run of dispatches
    // over the same or neighbouring bytes keeps a short list.
    private static void AddMerged(List<GuestSpan> target, List<GuestSpan> spans)
    {
        foreach (var span in spans)
        {
            var merged = false;
            for (var index = 0; index < target.Count; index++)
            {
                var existing = target[index];
                if (span.Address <= existing.End && existing.Address <= span.End)
                {
                    var begin = Math.Min(span.Address, existing.Address);
                    target[index] = new GuestSpan(begin, Math.Max(span.End, existing.End) - begin);
                    merged = true;
                    break;
                }
            }

            if (!merged)
            {
                target.Add(span);
            }
        }
    }

    private static bool Overlaps(GuestSpan span, List<GuestSpan> right)
    {
        foreach (var other in right)
        {
            if (span.Address < other.End && other.Address < span.End)
            {
                return true;
            }
        }

        return false;
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

internal enum OverlapCause
{
    None,
    Unknown,
    Indirect,
    BufferReadAfterWrite,
    BufferWriteAfterRead,
    BufferWriteAfterWrite,
    BufferAtomics,
    Image,
    ImageWriteAfterWrite,
    ImageUnresolved,
    Count,
}

public enum ChainDecision
{
    Open,
    Start,
    Skipped,
    Overlap,
    Conservative,
    SameProgramWrites,
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
    private static readonly long[] _overlapCauses = new long[(int)OverlapCause.Count];
    private static long _overlapImagesApart;
    private static long _overlapWritesApart;
    private static long _overlapAtomicsApart;

    /// <summary>A marked dispatch kept beside the group although both access the same bytes, each solely by atomics.</summary>
    internal static void NoteOverlapAtomicsApart() => Interlocked.Increment(ref _overlapAtomicsApart);

    /// <summary>A marked dispatch kept beside the group although both write the same buffer bytes, neither reading them.</summary>
    internal static void NoteOverlapWritesApart() => Interlocked.Increment(ref _overlapWritesApart);

    /// <summary>Overlap-marked dispatches ordered after the ones before them (DispatchChain.KeepsOverlap), always counted.</summary>
    public static void NoteOverlapOrdered(bool unknown) =>
        Interlocked.Increment(ref unknown ? ref _overlapsOrderedUnknown : ref _overlapsOrderedHazard);

    internal static void NoteOverlapOrdered(OverlapCause cause)
    {
        NoteOverlapOrdered(cause == OverlapCause.Unknown);
        Interlocked.Increment(ref _overlapCauses[(int)cause]);
    }

    /// <summary>A marked dispatch kept beside the group although both use images, their data apart.</summary>
    internal static void NoteOverlapImagesApart() => Interlocked.Increment(ref _overlapImagesApart);

    /// <summary>Overlap-marked dispatches ordered for a shared range (Hazard) or for accesses not all known (Unknown).</summary>
    public static (long Hazard, long Unknown) OverlapsOrdered =>
        (Interlocked.Read(ref _overlapsOrderedHazard), Interlocked.Read(ref _overlapsOrderedUnknown));

    /// <summary>The overlap orderings since the last call, for a backend's report line.</summary>
    public static string TakeOverlapReport() =>
        FormattableString.Invariant($"overlap_ordered=[hazard:{Interlocked.Exchange(ref _overlapsOrderedHazard, 0)},unknown:{Interlocked.Exchange(ref _overlapsOrderedUnknown, 0)}] ") +
        FormattableString.Invariant($"overlap_by=[indirect:{Interlocked.Exchange(ref _overlapCauses[(int)OverlapCause.Indirect], 0)},raw:{Interlocked.Exchange(ref _overlapCauses[(int)OverlapCause.BufferReadAfterWrite], 0)},") +
        FormattableString.Invariant($"war:{Interlocked.Exchange(ref _overlapCauses[(int)OverlapCause.BufferWriteAfterRead], 0)},waw:{Interlocked.Exchange(ref _overlapCauses[(int)OverlapCause.BufferWriteAfterWrite], 0)},") +
        FormattableString.Invariant($"image:{Interlocked.Exchange(ref _overlapCauses[(int)OverlapCause.Image], 0)},image_waw:{Interlocked.Exchange(ref _overlapCauses[(int)OverlapCause.ImageWriteAfterWrite], 0)},image_unresolved:{Interlocked.Exchange(ref _overlapCauses[(int)OverlapCause.ImageUnresolved], 0)},") +
        FormattableString.Invariant($"images_apart:{Interlocked.Exchange(ref _overlapImagesApart, 0)},writes_apart:{Interlocked.Exchange(ref _overlapWritesApart, 0)},atomics:{Interlocked.Exchange(ref _overlapCauses[(int)OverlapCause.BufferAtomics], 0)},atomics_apart:{Interlocked.Exchange(ref _overlapAtomicsApart, 0)}]");

    public static string TakeReport() =>
        FormattableString.Invariant($"chain_open={Interlocked.Exchange(ref _decisions[(int)ChainDecision.Open], 0)} chain_start={Interlocked.Exchange(ref _decisions[(int)ChainDecision.Start], 0)} ") +
        FormattableString.Invariant($"chain_skipped={Interlocked.Exchange(ref _decisions[(int)ChainDecision.Skipped], 0)} chain_overlap={Interlocked.Exchange(ref _decisions[(int)ChainDecision.Overlap], 0)} chain_conservative={Interlocked.Exchange(ref _decisions[(int)ChainDecision.Conservative], 0)} chain_same_program={Interlocked.Exchange(ref _decisions[(int)ChainDecision.SameProgramWrites], 0)}");
}
