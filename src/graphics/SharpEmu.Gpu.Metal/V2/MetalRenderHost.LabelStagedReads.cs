// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Rendering;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// A read-only compute binding over held label copies reads a staged view of its bytes instead of landing the copies.
/// </summary>
/// <remarks>
/// A held copy has two consumers. The guest CPU must not see a CPU-visible label before the stores ahead of it are
/// visible, which is what the shared event of a landing is for (EncodeLabelBarrier). A dispatch that reads the bytes
/// only needs them in device order. Landing the copies for such a dispatch ended the compute encoder, encoded the shared
/// event and a copy encoder, and opened a new compute encoder: Demon's Souls did that ~39 times a frame, each a ~135 us
/// GPU gap. Instead the binding gets a ring range, and right before the dispatch, inside its compute encoder, the range is
/// filled from the guest window (after a memory barrier, so after the earlier dispatches' stores; the encoder waited on
/// the encoder fence when it began, so after the earlier encoders) and then the held copies over it are applied in held
/// order, the last copy into a word winning, as the landing's blits or scatter would leave it. The copies stay held and
/// land at the commit, as before, behind one barrier. That is the bytes the dispatch would read after a landing: the
/// landing would write exactly these copies into exactly these bytes, behind the same earlier work.
/// Only where that holds: a compute stage, the fence chain, at most <see cref="LabelStagedReadMaxBytes"/>, a range one
/// window covers (the source), no staged storage copy concerning it, no binding or device-address range of the same
/// dispatch writing it (the dispatch would otherwise read its own stores from another buffer), and every held copy
/// over it holding its final value as host-written words (an end-of-pipe label or a command-processor write; not an
/// occlusion dump or a GDS copy, whose source later GPU work produces). Anything else lands as before, counted by reason
/// (label_staged_refused).
/// A view is filled again at each dispatch that binds it, unless nothing could have changed its bytes since its last
/// fill (label_staged reuse): same command buffer (the ring range is this submission's), same guest submission slice (a
/// later one may follow CPU stores made before its submit), no GPU write noted over it since (every GPU write into guest
/// memory - bindings, transfers, fills, image writebacks, label and command-processor copies - goes through
/// NoteGuestMemoryWriteBeyondOverlays), no held copy added over its window bytes, no unbounded device-address store,
/// and no landing since.
/// SHARPEMU_METAL_V2_LABEL_STAGED_READS=0 lands the copies for every such binding again; =verify also lands them right
/// after filling each view, copies the view and the window after the landing into a buffer of its own, and compares the
/// two at completion: a difference is fatal on the worker.
/// </remarks>
internal sealed partial class MetalRenderHost
{
    private static readonly string? LabelStagedReadsMode = Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_LABEL_STAGED_READS");

    internal bool LabelStagedReads { get; set; } = LabelStagedReadsMode != "0";

    internal bool LabelStagedReadsVerify { get; set; } = LabelStagedReadsMode == "verify";

    /// <summary>Tests: a range the dispatch being bound writes, as another of its bindings would (no stage to read it from).</summary>
    internal (ulong Address, ulong Size) LabelStagedDispatchWriteForTest { get; set; }

    // Demon's Souls' forcing ranges are all below 64 KiB; a larger copy per dispatch could cost what the gap does.
    internal const ulong LabelStagedReadMaxBytes = 64 * 1024;

    private const int LabelStagedReadTableSize = 64;

    internal enum LabelStagedRefusal
    {
        Graphics,
        NoFence,
        Large,
        GpuSource,
        Unaligned,
        StagedStorage,
        NotInPlace,
        DispatchWrites,
        Alias,
        Count,
    }

    private static readonly string[] LabelStagedRefusalNames =
        ["graphics", "no_fence", "large", "gpu_source", "unaligned", "staged_storage", "not_in_place", "dispatch_writes", "alias"];

    private static readonly string[] LabelStagedNames = ["binds", "fills", "reuse", "kb", "verify_checks", "verify_failed"];

    private sealed class LabelStagedRead
    {
        public ulong Address;
        public ulong Size;
        // The window bytes the view copies, from the word holding Address: Window.Offset is that word's.
        public MetalGuestRange Window;
        public MetalGuestRange Staging;
        public uint Lead;
        public ulong Span;
        public ulong Serial;
        public ulong Epoch;
        // Filled at a dispatch with nothing noted over it since.
        public bool Fresh;
        // Still in the table, so a later binding may find it.
        public bool Listed;
        // Bound by the dispatch being prepared, to be filled (or found fresh) right before it.
        public bool Pending;
    }

    private readonly List<LabelStagedRead> _labelStagedTable = [];
    private readonly List<LabelStagedRead> _labelStagedPending = [];
    private ulong _labelStagedLow = ulong.MaxValue;
    private ulong _labelStagedHigh;
    private ulong _labelStagedEpoch;

    // The compute stage BindResources is binding; its other bindings are checked for writes over a staged range.
    private MetalStageBindings? _labelStagedStage;

    private readonly long[] _labelStagedRefused = new long[(int)LabelStagedRefusal.Count];
    private long _labelStagedBinds, _labelStagedFills, _labelStagedReuse, _labelStagedBytes;
    private long _labelStagedVerifyChecks, _labelStagedVerifyFailed;
    private string? _labelStagedVerifyFailure;

    // =verify: views a binding found fresh with no held copy over them any more, bound to the window as usual; the
    // dispatch checks that the view still holds what the window does there (the reuse rule, without a landing).
    private readonly List<LabelStagedRead> _labelStagedChecks = [];

    private readonly Dictionary<ulong, int> _labelStagedWords = [];
    private readonly Dictionary<nint, ulong> _labelStagedAddresses = [];
    private LabelScatterWord[] _labelStagedList = new LabelScatterWord[64];
    private nint[] _labelStagedResources = new nint[8];
    private static readonly nint SelLabelStagedLength = MetalNative.Selector("length");
    private static readonly nint SelLabelStagedNewBuffer = MetalNative.Selector("newBufferWithLength:options:");

    /// <summary>label_staged: binds, fills, reuse, kb, verify_checks, verify_failed since the last take (tests).</summary>
    internal long[] TakeLabelStagedCounts() =>
    [
        Interlocked.Exchange(ref _labelStagedBinds, 0), Interlocked.Exchange(ref _labelStagedFills, 0),
        Interlocked.Exchange(ref _labelStagedReuse, 0), Interlocked.Exchange(ref _labelStagedBytes, 0) / 1024,
        Interlocked.Exchange(ref _labelStagedVerifyChecks, 0), Interlocked.Exchange(ref _labelStagedVerifyFailed, 0),
    ];

    /// <summary>label_staged_refused since the last take, in <see cref="LabelStagedRefusal"/> order (tests).</summary>
    internal long[] TakeLabelStagedRefusals()
    {
        var counts = new long[_labelStagedRefused.Length];
        for (var index = 0; index < counts.Length; index++) counts[index] = Interlocked.Exchange(ref _labelStagedRefused[index], 0);
        return counts;
    }

    /// <summary>Views bound and not yet filled for a dispatch (tests).</summary>
    internal int PendingLabelStagedReads => _labelStagedPending.Count;

    private void AppendLabelStagedCounters(System.Text.StringBuilder text)
    {
        var counts = TakeLabelStagedCounts();
        text.Append(" label_staged=[");
        for (var index = 0; index < LabelStagedNames.Length; index++)
        {
            if (index != 0) text.Append(',');
            text.Append(LabelStagedNames[index]).Append(':').Append(counts[index]);
        }

        text.Append("] label_staged_refused=[");
        for (var index = 0; index < LabelStagedRefusalNames.Length; index++)
        {
            if (index != 0) text.Append(',');
            text.Append(LabelStagedRefusalNames[index]).Append(':').Append(Interlocked.Exchange(ref _labelStagedRefused[index], 0));
        }

        text.Append(']');
    }

    // A read-only binding of these bytes, which held copies touch: a view to bind instead of landing them, or false
    // (counted when a reason refused it) to land them as before. untouched: no held copy touches the bytes and a fill
    // over them has landed, which is all OrderAfterDeferredCopies would do.
    private bool TryStageLabelRead(ulong address, ulong size, bool computeStage, out MetalGuestRange view, out bool untouched)
    {
        view = default;
        untouched = false;
        if (!LabelStagedReads || !LabelsAtCommit || size == 0 || size > ulong.MaxValue - address)
        {
            return false;
        }

        if (_deferredLabelCopies.Count == 0)
        {
            NoteLabelStagedReuseCheck(address, size, computeStage);
            return false;
        }

        ThrowIfLabelStagedVerifyFailed();
        // As OrderAfterDeferredCopies: a fill the open pass holds over the bytes lands first, so the view sees it.
        LandDeferredFillsOver(address, size);
        if (!DeferredLabelCopiesTouch(address, size))
        {
            untouched = true;
            NoteLabelStagedReuseCheck(address, size, computeStage);
            return false;
        }

        if (!computeStage)
        {
            return RefuseLabelStagedRead(LabelStagedRefusal.Graphics);
        }

        if (EncoderOrdering != EncoderOrderingMode.Fence)
        {
            // The view's fill reads the window in its dispatch's encoder, which must start after the encoders before it.
            return RefuseLabelStagedRead(LabelStagedRefusal.NoFence);
        }

        if (size > LabelStagedReadMaxBytes)
        {
            return RefuseLabelStagedRead(LabelStagedRefusal.Large);
        }

        if (_stagedStorageCopies is { Count: > 0 } staged && staged.MayConcern(address, size))
        {
            // A staged copy would answer the binding, or its stores are still to be written back over the bytes.
            return RefuseLabelStagedRead(LabelStagedRefusal.StagedStorage);
        }

        if ((_labelStagedStage is { } stage && DispatchWritesOver(stage, address, size)) ||
            (LabelStagedDispatchWriteForTest is var (writeAddress, writeSize) && writeAddress < address + size && address < writeAddress + writeSize))
        {
            return RefuseLabelStagedRead(LabelStagedRefusal.DispatchWrites);
        }

        if (!_guestMemory.TryResolveInPlace(address, size, out var window, out _, out _))
        {
            return RefuseLabelStagedRead(LabelStagedRefusal.NotInPlace);
        }

        var lead = (uint)(address & 3);
        var span = (lead + size + 3) & ~3UL;
        if ((window.Offset & 3) != lead ||
            window.Offset - lead + span > (ulong)MetalNative.Send(window.Buffer, SelLabelStagedLength))
        {
            return RefuseLabelStagedRead(LabelStagedRefusal.Unaligned);
        }

        var source = new MetalGuestRange(window.Buffer, window.Offset - lead, span, 0);
        var refusal = HeldCopiesOver(address - lead, in source, build: false, staging: 0);
        if (refusal != LabelStagedRefusal.Count)
        {
            return RefuseLabelStagedRead(refusal);
        }

        var entry = FindLabelStagedRead(address, size, in source);
        if (entry is null)
        {
            var staging = _guestMemory.Allocate(span, 16, out _);
            if (!staging.IsValid || staging.GpuAddress == 0)
            {
                throw Fatal($"A staged label read has no ring room: address=0x{address:X16} size=0x{size:X}.");
            }

            entry = new LabelStagedRead
            {
                Address = address, Size = size, Window = source, Staging = staging, Lead = lead, Span = span,
                Serial = _submissionSerial, Epoch = _labelStagedEpoch, Listed = true,
            };
            if (_labelStagedTable.Count == LabelStagedReadTableSize)
            {
                _labelStagedTable[0].Listed = false;
                _labelStagedTable.RemoveAt(0);
            }

            _labelStagedTable.Add(entry);
            _labelStagedLow = Math.Min(_labelStagedLow, address - lead);
            _labelStagedHigh = Math.Max(_labelStagedHigh, address - lead + span);
        }

        if (!entry.Pending)
        {
            entry.Pending = true;
            _labelStagedPending.Add(entry);
        }

        _labelStagedBinds++;
        _labelTouchClass = -1;
        view = new MetalGuestRange(entry.Staging.Buffer, entry.Staging.Offset + lead, size, entry.Staging.GpuAddress + lead);
        return true;
    }

    // =verify: a view of exactly this range still listed as fresh is checked against the window at the dispatch.
    private void NoteLabelStagedReuseCheck(ulong address, ulong size, bool computeStage)
    {
        if (!LabelStagedReadsVerify || !computeStage || _labelStagedTable.Count == 0 ||
            !_guestMemory.TryResolveInPlace(address, size, out var window, out _, out _))
        {
            return;
        }

        var lead = address & 3;
        var source = new MetalGuestRange(window.Buffer, window.Offset - lead, (lead + size + 3) & ~3UL, 0);
        if (FindLabelStagedRead(address, size, in source) is { Fresh: true } entry && !_labelStagedChecks.Contains(entry))
        {
            _labelStagedChecks.Add(entry);
        }
    }

    private bool RefuseLabelStagedRead(LabelStagedRefusal reason)
    {
        _labelStagedRefused[(int)reason]++;
        return false;
    }

    private LabelStagedRead? FindLabelStagedRead(ulong address, ulong size, in MetalGuestRange source)
    {
        foreach (var entry in _labelStagedTable)
        {
            if (entry.Address == address && entry.Size == size)
            {
                if (entry.Serial == _submissionSerial && entry.Epoch == _labelStagedEpoch &&
                    entry.Window.Buffer == source.Buffer && entry.Window.Offset == source.Offset)
                {
                    return entry;
                }

                // A view of another command buffer or guest submission, or another window: replaced.
                entry.Listed = false;
                _labelStagedTable.Remove(entry);
                return null;
            }
        }

        return null;
    }

    // Whether another binding or device-address range of the dispatch being bound writes these bytes: the dispatch would
    // read them from the view while storing them through the window. Unplanned ranges and unbounded stores count.
    private static bool DispatchWritesOver(MetalStageBindings stage, ulong address, ulong size)
    {
        var info = stage.Program.Resources!.Info;
        var snapshot = stage.StageResources.Resources;
        for (var index = 0; index < stage.BufferCount && index < info.Buffers.Count; index++)
        {
            if (!info.Buffers[index].Written)
            {
                continue;
            }

            var words = snapshot.Buffers[index];
            if (words.Length < 4)
            {
                return true;
            }

            var descriptor = BufferDescriptorWords.From(words);
            if (descriptor.Footprint() is not { } footprint)
            {
                return true;
            }

            if (descriptor.Address != 0 && footprint != 0 &&
                descriptor.Address < address + size && address < descriptor.Address + footprint)
            {
                return true;
            }
        }

        var declared = false;
        foreach (var range in snapshot.DeviceAddressRanges)
        {
            if (!range.Planned)
            {
                return true;
            }

            if (range.Written)
            {
                declared = true;
                if (range.Base < address + size && address < range.Base + range.Size)
                {
                    return true;
                }
            }
        }

        return info.WritesDeviceAddresses && !declared;
    }

    // The held copies over a view's window bytes (by guest address, or by the window buffer's bytes for an alias of
    // them), in held order. Count when every one can be applied; otherwise why not. build: also make the scatter list
    // into the staging at that GPU address (_labelStagedList, _labelStagedResources), returning the word count in
    // _labelStagedWords.Count.
    private LabelStagedRefusal HeldCopiesOver(ulong low, in MetalGuestRange source, bool build, ulong staging)
    {
        if (build)
        {
            _labelStagedWords.Clear();
            _labelStagedAddresses.Clear();
        }

        var span = source.Size;
        var copies = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(_deferredLabelCopies);
        foreach (ref readonly var copy in copies)
        {
            var bytes = (ulong)copy.Bytes;
            var byGuest = copy.GuestAddress < low + span && low < copy.GuestAddress + bytes;
            var byBuffer = copy.DestinationBuffer == source.Buffer &&
                copy.DestinationOffset < source.Offset + span && source.Offset < copy.DestinationOffset + bytes;
            if (!byGuest && !byBuffer)
            {
                continue;
            }

            if (copy.Kind == LabelCopyKind.Other || copy.StagingBuffer == 0)
            {
                return LabelStagedRefusal.GpuSource;
            }

            if (((bytes | copy.StagingOffset) & 3) != 0 || (byGuest && (copy.GuestAddress & 3) != 0) ||
                (byBuffer && (copy.DestinationOffset & 3) != 0))
            {
                return LabelStagedRefusal.Unaligned;
            }

            var at = byGuest
                ? (long)(copy.GuestAddress - low)
                : (long)copy.DestinationOffset - (long)source.Offset;
            if (byGuest && byBuffer && at != (long)copy.DestinationOffset - (long)source.Offset)
            {
                return LabelStagedRefusal.Alias;
            }

            if (!build)
            {
                continue;
            }

            if (!_labelStagedAddresses.TryGetValue(copy.StagingBuffer, out var from))
            {
                from = MetalNative.SendGpuAddress(copy.StagingBuffer, SelLabelScatterGpuAddress);
                if (from == 0)
                {
                    throw Fatal("A held label copy's staging has no GPU address for a staged read.");
                }

                _labelStagedAddresses.Add(copy.StagingBuffer, from);
            }

            for (var offset = 0L; offset < (long)bytes; offset += sizeof(uint))
            {
                var position = at + offset;
                if (position < 0 || position >= (long)span)
                {
                    continue;
                }

                var word = new LabelScatterWord { Source = from + copy.StagingOffset + (ulong)offset, Destination = staging + (ulong)position };
                if (_labelStagedWords.TryGetValue((ulong)position, out var existing))
                {
                    _labelStagedList[existing] = word;
                    continue;
                }

                var count = _labelStagedWords.Count;
                if (count == _labelStagedList.Length)
                {
                    Array.Resize(ref _labelStagedList, count * 2);
                }

                _labelStagedWords.Add((ulong)position, count);
                _labelStagedList[count] = word;
            }
        }

        return LabelStagedRefusal.Count;
    }

    /// <summary>
    /// Right before a dispatch, in its compute encoder: fills the views its bindings got (or finds them still fresh).
    /// Returns the encoder the dispatch encodes into, which =verify replaces after landing the copies.
    /// </summary>
    internal nint EncodeStagedLabelReads(nint encoder)
    {
        if (_labelStagedPending.Count == 0 && _labelStagedChecks.Count == 0)
        {
            return encoder;
        }

        ThrowIfLabelStagedVerifyFailed();
        foreach (var entry in _labelStagedPending)
        {
            if (entry.Serial != _submissionSerial)
            {
                // The ring range belongs to the command buffer the binding was made in.
                throw Fatal(
                    $"A staged label read is dispatched in another command buffer than it was bound in: address=0x{entry.Address:X16} " +
                    $"size=0x{entry.Size:X} bound={entry.Serial} now={_submissionSerial}.");
            }

            if (entry.Fresh && entry.Listed)
            {
                _labelStagedReuse++;
                continue;
            }

            FillLabelStagedRead(encoder, entry);
            entry.Fresh = entry.Listed;
        }

        if (LabelStagedReadsVerify)
        {
            encoder = VerifyLabelStagedReads(encoder);
        }

        foreach (var entry in _labelStagedPending)
        {
            entry.Pending = false;
        }

        _labelStagedPending.Clear();
        return encoder;
    }

    private unsafe void FillLabelStagedRead(nint encoder, LabelStagedRead entry)
    {
        var refusal = HeldCopiesOver(entry.Address - entry.Lead, in entry.Window, build: true, entry.Staging.GpuAddress);
        if (refusal != LabelStagedRefusal.Count)
        {
            // Checked when the binding was made; nothing records a copy between a binding and its dispatch.
            throw Fatal(
                $"A held copy over a staged label read changed before its dispatch: address=0x{entry.Address:X16} " +
                $"size=0x{entry.Size:X} reason={LabelStagedRefusalNames[(int)refusal]}.");
        }

        NoteRecordedWork();
        MetalV2Native.SendMemoryBarrierScope(encoder, SelComputeMemoryBarrier, Mtl.BarrierScopeBuffers | Mtl.BarrierScopeTextures);
        EncodeLabelStagedCopy(encoder, entry.Window.Buffer, entry.Window.Offset, entry.Staging.Buffer, entry.Staging.Offset, entry.Span);
        var count = _labelStagedWords.Count;
        if (count != 0)
        {
            var list = _guestMemory.Allocate((ulong)count * 16, 16, out var payload);
            if (!list.IsValid)
            {
                throw Fatal($"A staged label read has no ring room for its copies: address=0x{entry.Address:X16} words={count}.");
            }

            System.Runtime.InteropServices.MemoryMarshal.AsBytes(_labelStagedList.AsSpan(0, count)).CopyTo(payload);
            var resources = _labelStagedAddresses.Count + 1;
            if (_labelStagedResources.Length < resources)
            {
                _labelStagedResources = new nint[resources * 2];
            }

            var used = 0;
            foreach (var buffer in _labelStagedAddresses.Keys)
            {
                _labelStagedResources[used++] = buffer;
            }

            if (!_labelStagedAddresses.ContainsKey(entry.Staging.Buffer))
            {
                _labelStagedResources[used++] = entry.Staging.Buffer;
            }

            MetalV2Native.SendMemoryBarrierScope(encoder, SelComputeMemoryBarrier, Mtl.BarrierScopeBuffers | Mtl.BarrierScopeTextures);
            MetalNative.SendVoid(encoder, SelOcclusionPipeline, LabelScatterPipeline());
            MetalNative.SendSetBuffer(encoder, SelOcclusionBuffer, list.Buffer, (nuint)list.Offset, 0);
            var words = (uint)count;
            MetalNative.SendSetBytes(encoder, SelOcclusionBytes, (nint)(&words), sizeof(uint), 1);
            fixed (nint* pointers = _labelStagedResources)
            {
                MetalNative.SendUseResources(encoder, SelUseResourcesUsage, pointers, (nuint)used,
                    Mtl.ResourceUsageRead | Mtl.ResourceUsageWrite);
            }

            MetalNative.SendDispatch(encoder, SelOcclusionDispatch,
                new MtlSize { Width = (nuint)((count + LabelScatterThreads - 1) / LabelScatterThreads), Height = 1, Depth = 1 },
                new MtlSize { Width = LabelScatterThreads, Height = 1, Depth = 1 });
        }

        MetalV2Native.SendMemoryBarrierScope(encoder, SelComputeMemoryBarrier, Mtl.BarrierScopeBuffers | Mtl.BarrierScopeTextures);
        // The guest dispatch binds its own pipeline again; its buffers are bound per dispatch.
        _boundComputePipeline = 0;
        _labelStagedFills++;
        _labelStagedBytes += (long)entry.Span;
        if (PassProfileEnabled)
        {
            PassProfileNoteWork($"label-staged-read words={count}", "transfer");
        }
    }

    // Words from source to destination by the word-copy kernel, between barriers the caller places.
    private unsafe void EncodeLabelStagedCopy(nint encoder, nint source, ulong sourceOffset, nint destination, ulong destinationOffset, ulong size)
    {
        var count = (uint)(size / sizeof(uint));
        MetalNative.SendVoid(encoder, SelOcclusionPipeline, CopyWordsPipeline());
        MetalNative.SendSetBuffer(encoder, SelOcclusionBuffer, source, (nuint)sourceOffset, 0);
        MetalNative.SendSetBuffer(encoder, SelOcclusionBuffer, destination, (nuint)destinationOffset, 1);
        MetalNative.SendSetBytes(encoder, SelOcclusionBytes, (nint)(&count), sizeof(uint), 2);
        MetalNative.SendDispatch(encoder, SelOcclusionDispatch,
            new MtlSize { Width = (nuint)((count + FillThreads - 1) / FillThreads), Height = 1, Depth = 1 },
            new MtlSize { Width = FillThreads, Height = 1, Depth = 1 });
    }

    // =verify: each view filled for this dispatch, copied out; then the landing the view stood in for; then the window
    // bytes after it, copied out too. Each view found fresh with nothing held over it any more (_labelStagedChecks) is
    // copied out next to its window bytes at this dispatch, with no landing: the reuse rule says they are equal. At
    // completion each pair must be equal.
    private unsafe nint VerifyLabelStagedReads(nint encoder)
    {
        // A view something wrote over since its binding is filled again before its next reuse; only a fresh one is checked.
        _labelStagedChecks.RemoveAll(entry => !entry.Fresh || !entry.Listed || entry.Serial != _submissionSerial);
        var filled = _labelStagedPending.Count;
        var checks = new (LabelStagedRead Entry, nint Buffer)[filled + _labelStagedChecks.Count];
        MetalV2Native.SendMemoryBarrierScope(encoder, SelComputeMemoryBarrier, Mtl.BarrierScopeBuffers | Mtl.BarrierScopeTextures);
        for (var index = 0; index < checks.Length; index++)
        {
            var entry = index < filled ? _labelStagedPending[index] : _labelStagedChecks[index - filled];
            var buffer = MetalNative.SendNewBuffer(_device, SelLabelStagedNewBuffer, (nuint)(entry.Span * 2), 0);
            if (buffer == 0)
            {
                throw Fatal("A staged label read verify buffer could not be created.");
            }

            checks[index] = (entry, buffer);
            EncodeLabelStagedCopy(encoder, entry.Staging.Buffer, entry.Staging.Offset, buffer, 0, entry.Span);
        }

        MetalV2Native.SendMemoryBarrierScope(encoder, SelComputeMemoryBarrier, Mtl.BarrierScopeBuffers | Mtl.BarrierScopeTextures);
        _boundComputePipeline = 0;
        for (var index = 0; index < filled; index++)
        {
            OrderAfterDeferredCopies(checks[index].Entry.Address, checks[index].Entry.Size, LabelLandCause.BindRead);
        }

        encoder = EnsureComputeEncoder();
        MetalV2Native.SendMemoryBarrierScope(encoder, SelComputeMemoryBarrier, Mtl.BarrierScopeBuffers | Mtl.BarrierScopeTextures);
        foreach (var (entry, buffer) in checks)
        {
            EncodeLabelStagedCopy(encoder, entry.Window.Buffer, entry.Window.Offset, buffer, entry.Span, entry.Span);
        }

        MetalV2Native.SendMemoryBarrierScope(encoder, SelComputeMemoryBarrier, Mtl.BarrierScopeBuffers | Mtl.BarrierScopeTextures);
        _boundComputePipeline = 0;
        NoteRecordedWork();
        _labelStagedChecks.Clear();
        _completions.Enqueue(SubmissionSerial, () =>
        {
            for (var index = 0; index < checks.Length; index++)
            {
                var (entry, buffer) = checks[index];
                var contents = new ReadOnlySpan<byte>((void*)MetalNative.Send(buffer, SelLabelFoldContents), checked((int)(entry.Span * 2)));
                var staged = contents[..(int)entry.Span];
                var landed = contents[(int)entry.Span..];
                Interlocked.Increment(ref _labelStagedVerifyChecks);
                if (!staged.SequenceEqual(landed))
                {
                    var at = staged.CommonPrefixLength(landed) & ~3;
                    Interlocked.Increment(ref _labelStagedVerifyFailed);
                    var failure =
                        $"A staged label read differs from the bytes {(index < filled ? "a landing left" : "its reuse reads")}: " +
                        $"address=0x{entry.Address:X16} size=0x{entry.Size:X} word=0x{entry.Address - entry.Lead + (ulong)at:X16} " +
                        $"staged=0x{BitConverter.ToUInt32(staged[at..]):X8} window=0x{BitConverter.ToUInt32(landed[at..]):X8}.";
                    Interlocked.CompareExchange(ref _labelStagedVerifyFailure, failure, null);
                    Console.Error.WriteLine($"[METAL-V2][ERROR] {failure}");
                }

                MetalNative.SendVoid(buffer, SelRelease);
            }
        });

        return encoder;
    }

    private void ThrowIfLabelStagedVerifyFailed()
    {
        if (Volatile.Read(ref _labelStagedVerifyFailure) is { } failure)
        {
            throw Fatal(failure);
        }
    }

    /// <summary>Tests: whether a staged label read failed verification (clears it).</summary>
    internal string? TakeLabelStagedVerifyFailure() => Interlocked.Exchange(ref _labelStagedVerifyFailure, null);

    // A GPU write into guest memory was recorded: views over it are filled again at their next dispatch.
    private void NoteLabelStagedReadsWrite(ulong address, ulong size)
    {
        if (_labelStagedTable.Count == 0 || address >= _labelStagedHigh || address + size <= _labelStagedLow)
        {
            return;
        }

        foreach (var entry in _labelStagedTable)
        {
            if (address < entry.Address - entry.Lead + entry.Span && entry.Address - entry.Lead < address + size)
            {
                entry.Fresh = false;
            }
        }
    }

    // A held copy into these window bytes: a view over them by another guest address is filled again too.
    private void NoteLabelStagedReadsCopy(in DeferredLabelCopy copy)
    {
        if (_labelStagedTable.Count == 0)
        {
            return;
        }

        foreach (var entry in _labelStagedTable)
        {
            if (entry.Window.Buffer == copy.DestinationBuffer &&
                copy.DestinationOffset < entry.Window.Offset + entry.Span && entry.Window.Offset < copy.DestinationOffset + (ulong)copy.Bytes)
            {
                entry.Fresh = false;
            }
        }
    }

    // Held copies landed, a store with no known range, or a new guest submission: no view found so far is reused.
    // A view the dispatch being bound holds is still filled before it.
    private void ForgetLabelStagedReads()
    {
        if (_labelStagedTable.Count == 0)
        {
            return;
        }

        foreach (var entry in _labelStagedTable)
        {
            entry.Listed = false;
            entry.Fresh = false;
        }

        _labelStagedTable.Clear();
        _labelStagedLow = ulong.MaxValue;
        _labelStagedHigh = 0;
    }

    /// <summary>A guest submission (slice) begins: its CPU stores before the submit are not in the views made so far.</summary>
    internal void NoteLabelStagedReadsSubmission()
    {
        _labelStagedEpoch++;
        ForgetLabelStagedReads();
    }

    // BindResources begins a stage: views bound for a dispatch that was never encoded are dropped (their bindings with it).
    private void BeginLabelStagedStage(MetalStageBindings? computeStage)
    {
        foreach (var entry in _labelStagedPending)
        {
            entry.Pending = false;
            entry.Fresh = false;
        }

        _labelStagedPending.Clear();
        _labelStagedChecks.Clear();
        _labelStagedStage = computeStage;
    }
}
