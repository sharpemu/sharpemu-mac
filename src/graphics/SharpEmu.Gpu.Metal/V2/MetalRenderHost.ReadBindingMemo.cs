// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// Read-only guest-range bindings answered from the binding before them (MetalReadBindingMemo). For every read-only
/// storage buffer of every draw, BindResources asked whether the range was mapped (twice), flushed the images over it,
/// ordered it after deferred copies and staged stores, looked for a staged copy to share, and resolved its window
/// (translation, window memo, census); ObtainBuffer did the same for index and indirect buffers. Their inputs rarely
/// change from one draw to the next. Demon's Souls binds ~7000 draws a frame, most of them the same few vertex and
/// constant buffers.
///
/// A hit skips that chain, which is sound while each of its steps would do nothing and answer as it did:
/// <list type="bullet">
/// <item>The mapping checks and the window's translation: the range translates to the backing alias it was placed by
/// (checked on every hit, so a remap misses whatever moved it); a range whose alias spans it is mapped whole, which
/// is what both mapping checks answered.</item>
/// <item>The image flush: an entry is kept only when the flush found no GPU-modified image over the range, and that
/// stays so while no image is added or dropped (ImageSetVersion) and none goes GPU-modified (Epochs.GpuModified);
/// an image's guest range never changes in place.</item>
/// <item>Deferred fills and label copies, staged copies and their recorded stores: checked live, a hit only while
/// there are none at all, which leaves OrderAfterDeferredCopies, TryShare and OrderAfterStagedStores nothing to
/// do.</item>
/// <item>The window: the same command buffer, window cuts and single-copy holes and stale windows
/// (MetalGuestMemory.InPlaceStamp), no trap and no released chunk under the range
/// (MetalGuestMemory.InPlaceAnswerStands). The census mark is made again.</item>
/// <item>No trace or dump that reports bindings is on.</item>
/// </list>
/// The stamp is read for each lookup and on both sides of the chain an entry is stored from (a miss earlier in a
/// draw can move the window cuts). Written bindings always take the full chain: their invalidation and write record
/// must run in order. SHARPEMU_METAL_V2_READ_BINDING_MEMO=0 takes the full chain for every binding; =verify takes it
/// on every hit too and raises a fatal error when the two answers differ.
/// </summary>
internal sealed partial class MetalRenderHost
{
    private static readonly string? ReadBindingMemoSetting =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_READ_BINDING_MEMO");

    internal bool ReadBindingMemoEnabled { get; set; } = ReadBindingMemoSetting != "0";

    internal bool ReadBindingMemoVerify { get; set; } =
        ReadBindingMemoSetting == "verify" || Environment.GetEnvironmentVariable("SHARPEMU_PERF_VERIFY") == "1";

    /// <summary>
    /// Entries checked per range (MetalReadBindingMemoV2), the default; SHARPEMU_METAL_V2_READ_BINDING_MEMO_RANGES=0
    /// keeps the stamp of everything (MetalReadBindingMemo), as before.
    /// </summary>
    internal bool ReadBindingMemoRanges { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_READ_BINDING_MEMO_RANGES") != "0";

    // Each table is about 7 MiB at the default 16384 sets: only the one in use is made.
    private MetalReadBindingMemo? _readBindingMemoV1;
    private MetalReadBindingMemoV2? _readBindingMemoV2;

    private MetalReadBindingMemo ReadBindingMemoV1 => _readBindingMemoV1 ??= new();

    /// <summary>The v2 memo and its counters (tests).</summary>
    internal MetalReadBindingMemoV2 ReadBindingMemoV2 => _readBindingMemoV2 ??= new();

    /// <summary>The held label copy class the last binding touched, -1 for none (tests).</summary>
    internal int LabelTouchClass => _labelTouchClass;

    // The entry the last v2 hit answered from (the verify mode reads its claims).
    private int _readBindingHitIndex = -1;

    internal (long Hits, long Misses, long Stores, long Verified) ReadBindingMemoCounts =>
        ((_readBindingMemoV1?.Hits ?? 0) + (_readBindingMemoV2?.Hits ?? 0),
         (_readBindingMemoV1?.Misses ?? 0) + (_readBindingMemoV2?.Misses ?? 0),
         (_readBindingMemoV1?.Stores ?? 0) + (_readBindingMemoV2?.Stores ?? 0),
         (_readBindingMemoV1?.Verified ?? 0) + (_readBindingMemoV2?.Verified ?? 0));

    // Nothing that reports bindings is on.
    private bool ReadBindingTracesOff =>
        _traceBufferUserAddress == 0 && _traceRecordChain.Length != 6 && _traceBufferAtCompletion.Length != 4 &&
        _dumpBufferHashes.Count == 0 && TracedRingRange.High == 0 && _slotTrap is null;

    // What leaves the ordering and sharing steps of the chain nothing to do, whatever the range.
    private bool ReadBindingMemoApplies =>
        ReadBindingMemoEnabled && _deferredFills.Count == 0 && _deferredLabelCopies.Count == 0 &&
        _stagedStorageCopies is not { Count: > 0 } && ReadBindingTracesOff;

    // A lookup refused for pending work (a fill, a label copy or a staged copy): the store after its chain is counted.
    private bool _readBindingRefusedPending;

    // Counts the first reason ReadBindingMemoApplies (or the range's mapping) refused a lookup.
    private void NoteReadBindingRefusal(bool unmapped)
    {
        var memo = ReadBindingMemoV1;
        memo.Misses++;
        if (!ReadBindingTracesOff)
        {
            memo.RefusedTrace++;
        }
        else if (_deferredFills.Count != 0)
        {
            memo.RefusedFill++;
            _readBindingRefusedPending = true;
        }
        else if (_deferredLabelCopies.Count != 0)
        {
            memo.RefusedLabel++;
            _readBindingRefusedPending = true;
        }
        else if (_stagedStorageCopies is { Count: > 0 })
        {
            memo.RefusedStaged++;
            _readBindingRefusedPending = true;
        }
        else if (unmapped)
        {
            memo.RefusedUnmapped++;
        }
        else
        {
            throw Fatal("A read-only binding memo lookup was refused with nothing to refuse it.");
        }
    }

    private MetalReadBindingMemo.Stamp ReadBindingStamp()
    {
        var (serial, partition, trap) = _guestMemory.InPlaceStamp;
        return new MetalReadBindingMemo.Stamp(serial, partition, trap, _imageCache.ImageSetVersion, _imageCache.Epochs.GpuModified);
    }

    /// <summary>The window range a read-only binding of the range resolved to before, if it still stands.</summary>
    private bool TryReadBindingMemo(ulong address, ulong size, bool computeStage, out MetalGuestRange range)
    {
        range = default;
        if (!ReadBindingMemoEnabled)
        {
            return false;
        }

        if (ReadBindingMemoRanges)
        {
            return TryReadBindingMemoRanges(address, size, computeStage, out range);
        }

        _readBindingRefusedPending = false;
        if (!ReadBindingMemoApplies)
        {
            NoteReadBindingRefusal(unmapped: false);
            return false;
        }

        if (!_backing.TryResolveBackingAliasSpan(address, size, out var alias))
        {
            NoteReadBindingRefusal(unmapped: true);
            return false;
        }

        if (!ReadBindingMemoV1.TryGet(address, size, computeStage, alias, ReadBindingStamp(), out range) ||
            !_guestMemory.InPlaceAnswerStands(alias, size))
        {
            if (range.IsValid)
            {
                // The entry matched; the window under it no longer stands.
                ReadBindingMemoV1.RefusedWindow++;
            }

            ReadBindingMemoV1.Misses++;
            range = default;
            return false;
        }

        ReadBindingMemoV1.Hits++;
        MetalGpuReferenceCensus.Mark(address, size);
        return true;
    }

    // After the full chain of a read-only binding: kept when the flush found no image over the range, the range
    // resolved in place, and nothing the stamp holds moved while the chain ran.
    private void StoreReadBinding(
        ulong address, ulong size, bool computeStage, in MetalReadBindingMemo.Stamp before, bool flushFoundImage,
        ulong inPlaceAlias, MetalWindowPartition.Window? inPlaceWindow, in MetalGuestRange range)
    {
        if (ReadBindingMemoRanges)
        {
            if (ReadBindingMemoEnabled)
            {
                StoreReadBindingRanges(address, size, computeStage, in before, flushFoundImage, inPlaceAlias, inPlaceWindow, in range);
            }

            return;
        }

        if (flushFoundImage || inPlaceAlias == 0 || !range.IsValid || !ReadBindingMemoEnabled || !ReadBindingTracesOff ||
            ReadBindingStamp() != before || !_guestMemory.InPlaceAnswerStands(inPlaceAlias, size))
        {
            return;
        }

        if (_readBindingRefusedPending)
        {
            ReadBindingMemoV1.StoreAfterRefusal++;
            _readBindingRefusedPending = false;
        }

        ReadBindingMemoV1.Store(address, size, computeStage, inPlaceAlias, before, range);
    }

    // =verify: the full chain behind a hit, which must give the same range and find no image to flush; behind a v2
    // hit (MetalRenderHost.ReadBindingMemoRanges.cs) also do nothing, and find the window the hit pinned and touched.
    private void VerifyReadBinding(ulong address, ulong size, bool computeStage, MetalWritebackCause cause, in MetalGuestRange memo)
    {
        var ranges = ReadBindingMemoRanges;
        var before = ranges ? ReadBindingChainSnapshot.Take(this, address, size) : default;
        if (ranges)
        {
            VerifyReadBindingHitClaims(address, size, computeStage, in memo);
        }

        var flushed = _imageCache.FlushGpuWrites(address, size, cause, written: false);
        var full = ResolveStorageBuffer(address, size, written: false, computeStage, out var alias, out _);
        if (flushed || alias == 0 || full != memo)
        {
            throw Fatal(
                $"A memoised read-only binding disagrees with the full path: address=0x{address:X16} size=0x{size:X} " +
                $"compute={computeStage} memo={memo} full={full} in_place={alias != 0} flushed_image={flushed}.");
        }

        if (ranges)
        {
            var after = ReadBindingChainSnapshot.Take(this, address, size);
            if (before.StoresOver || after != before)
            {
                throw Fatal(
                    $"The full path behind a memoised read-only binding did something the hit skipped: address=0x{address:X16} " +
                    $"size=0x{size:X} compute={computeStage} before={before} after={after}.");
            }

            ReadBindingMemoV2.Verified++;
            return;
        }

        ReadBindingMemoV1.Verified++;
    }

    // ---- The shape of the stage bindings (the PERF line's bind_shape): ----

    // SHARPEMU_METAL_V2_BIND_SHAPE=1 (or SHARPEMU_METAL_V2_TRACE=1): the epoch moves and snapshot-array repeats too,
    // three ring scans per binding; otherwise only the counts.
    private static readonly bool BindShapeDiagnostics =
        MetalV2Diagnostics.Verbose || Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_BIND_SHAPE") == "1";

    private long _bindCalls;
    private long _bindImageSetMoves;
    private long _bindBindingEpochMoves;
    private long _lastBindImageSet = -1;
    private long _lastBindBindingEpoch = -1;
    private long _readOnlyBufferBinds;
    private long _writtenBufferBinds;
    private long _tinyPassDummies;
    private long _storageImageBinds;

    // Per stage kind, the last eight Images, Samplers and Buffers arrays a snapshot bound, and how often one repeated.
    private const int SnapshotRefSlots = 8;
    private readonly object?[] _snapshotRefs = new object?[8 * 3 * SnapshotRefSlots];
    private readonly int[] _snapshotRefNext = new int[8 * 3];
    private readonly long[] _snapshotRefRepeats = new long[3];
    private long _snapshotRefChecks;

    private void NoteBindShape(MetalStageBindings stage, ResourceSnapshot snapshot)
    {
        _bindCalls++;
        if (!BindShapeDiagnostics)
        {
            return;
        }

        var imageSet = _imageCache.ImageSetVersion;
        var bindingEpoch = _imageCache.Epochs.Binding;
        if (imageSet != _lastBindImageSet)
        {
            _bindImageSetMoves++;
            _lastBindImageSet = imageSet;
        }

        if (bindingEpoch != _lastBindBindingEpoch)
        {
            _bindBindingEpochMoves++;
            _lastBindBindingEpoch = bindingEpoch;
        }

        var kind = (int)stage.Program.Stage & 7;
        _snapshotRefChecks++;
        NoteSnapshotRef(kind, 0, snapshot.Images);
        NoteSnapshotRef(kind, 1, snapshot.Samplers);
        NoteSnapshotRef(kind, 2, snapshot.Buffers);
    }

    private void NoteSnapshotRef(int kind, int array, object reference)
    {
        var ring = (kind * 3) + array;
        var refs = _snapshotRefs.AsSpan(ring * SnapshotRefSlots, SnapshotRefSlots);
        foreach (var seen in refs)
        {
            if (ReferenceEquals(seen, reference))
            {
                _snapshotRefRepeats[array]++;
                return;
            }
        }

        refs[_snapshotRefNext[ring]] = reference;
        _snapshotRefNext[ring] = (_snapshotRefNext[ring] + 1) % SnapshotRefSlots;
    }

    private string TakeReadBindingCounters()
    {
        var memo = ReadBindingMemoRanges ? ReadBindingMemoV2.TakeCounters() : ReadBindingMemoV1.TakeCounters();
        var text = $"read_binding_memo=({memo}) bind_shape=(binds={_bindCalls} " +
            $"image_set_moves={_bindImageSetMoves} binding_epoch_moves={_bindBindingEpochMoves} " +
            $"snapshot_repeats=[images={_snapshotRefRepeats[0]},samplers={_snapshotRefRepeats[1]},buffers={_snapshotRefRepeats[2]}]/{_snapshotRefChecks} " +
            $"buffers_read_only={_readOnlyBufferBinds} buffers_written={_writtenBufferBinds} storage_images={_storageImageBinds} tiny_pass_dummies={_tinyPassDummies}) {TakeStagePlanCounters()}";
        _bindCalls = _bindImageSetMoves = _bindBindingEpochMoves = _readOnlyBufferBinds = _writtenBufferBinds = 0;
        _tinyPassDummies = _snapshotRefChecks = _storageImageBinds = 0;
        Array.Clear(_snapshotRefRepeats);
        return text;
    }
}
