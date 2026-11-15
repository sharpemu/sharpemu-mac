// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// The read-only binding memo checked per range (MetalReadBindingMemoV2, SHARPEMU_METAL_V2_READ_BINDING_MEMO_RANGES,
/// on by default). The v1 memo's stamp made every entry miss whenever anything anywhere moved: the next command buffer,
/// a window cut elsewhere, any image added, dropped or GPU-written. Demon's Souls moves one of those between nearly
/// every two bindings of a range, so most lookups paid the full chain. Here each condition is checked for the range:
/// <list type="bullet">
/// <item>The command buffer: what resolving a range in place did per command buffer (pin the window, mark it used, make
/// it resident) the hit does too (MetalGuestMemory.UseWindowForHit), once per window per command buffer.</item>
/// <item>The window cuts: the window object the range was placed in is still in the partition (a cut move, a cut back
/// to granules and a renewal all retire the windows they replace) and still has the buffer the entry holds; a window's
/// bounds never change, so the range still fits it.</item>
/// <item>Deferred fills, held label copies and staged copies: the hit misses only when one is over the range, which is
/// exactly when the chain would land or order it, or answer with the staged copy (a span test first, so the common
/// case costs no walk).</item>
/// <item>Images: the flush finds nothing when no image owns a page of the range (checked live), or when the images
/// overlapping it are the ones the entry was stored with - none added or dropped over the range since (the image-set
/// change ring, up to MetalImageCache.ImageSetWalkCap changes back) - and none of them is GPU-modified now (up to two
/// held by the entry, checked live). An image's guest range never changes in place.</item>
/// <item>Single-copy holes and stale windows: an entry stands at the trap version it was checked at; when that moved,
/// MetalGuestMemory.RevalidateInPlace asks under the window lock whether one is under this range and window.</item>
/// <item>The mapping (the backing alias, compared on every hit) and what MetalGuestMemory.InPlaceAnswerStands
/// checks live (nothing stages everything, no slot trap, no released chunk under the range).</item>
/// </list>
/// Written bindings always take the full chain. Every miss is counted once, by the first check that failed.
/// </summary>
internal sealed partial class MetalRenderHost
{
    private bool MissReadBinding(ref long cause)
    {
        ReadBindingMemoV2.Misses++;
        cause++;
        return false;
    }

    private bool TryReadBindingMemoRanges(ulong address, ulong size, bool computeStage, out MetalGuestRange range)
    {
        var memo = ReadBindingMemoV2;
        range = default;
        if (!ReadBindingTracesOff)
        {
            return MissReadBinding(ref memo.RefusedTrace);
        }

        if (!_backing.TryResolveBackingAliasSpan(address, size, out var alias))
        {
            return MissReadBinding(ref memo.RefusedUnmapped);
        }

        var index = memo.Find(address, size, computeStage);
        if (index < 0)
        {
            return false;
        }

        ref var entry = ref memo.At(index);
        if (entry.Alias != alias)
        {
            return MissReadBinding(ref memo.MissAlias);
        }

        var window = entry.Window!;
        if (Volatile.Read(ref window.Retired))
        {
            return MissReadBinding(ref memo.MissWindowRetired);
        }

        if (window.Buffer != entry.Buffer)
        {
            return MissReadBinding(ref memo.MissWindowBuffer);
        }

        if (!_guestMemory.InPlaceAnswerStands(alias, size))
        {
            return MissReadBinding(ref memo.RefusedWindow);
        }

        // Single-copy holes or stale windows changed somewhere since the entry was checked: whether any is under this
        // range and window is asked under the window lock.
        ref var side = ref memo.SideAt(index);
        if (side.Trap != _guestMemory.TrapVersionNow)
        {
            if (!_guestMemory.RevalidateInPlace(window, entry.Buffer, alias, size, out var trap))
            {
                return MissReadBinding(ref memo.MissTrap);
            }

            side.Trap = trap;
            memo.RevalidatedTrap++;
        }

        // The image flush would find no GPU-modified image over the range: none can own its pages, or the images over
        // it are those the entry was stored with (none added or dropped over it since) and none is GPU-modified now.
        if (entry.Tier == MetalReadBindingMemoV2.ImageTier.NoOwnerPages)
        {
            if (_imageCache.MayOwn(address, size))
            {
                return MissReadBinding(ref memo.MissImagePages);
            }
        }
        else
        {
            if (entry.ImageSet != _imageCache.ImageSetVersion)
            {
                if (!_imageCache.ImageSetUnchangedOver(entry.ImageSet, address, size, out var now))
                {
                    return MissReadBinding(ref memo.MissImageSet);
                }

                entry.ImageSet = now;
                memo.RevalidatedImageSet++;
            }

            if (entry.Tier == MetalReadBindingMemoV2.ImageTier.Held &&
                (side.Image0!.IsGpuModified || (side.Image1?.IsGpuModified ?? false)))
            {
                return MissReadBinding(ref memo.MissImageGpuModified);
            }
        }

        // Work the chain would land or order first, or a staged copy it would answer with: only over this range.
        // OrderAfterDeferredCopies always forgets which held copy a binding last touched.
        var fills = _deferredFills.Count != 0;
        if (fills && DeferredFillsOver(address, size))
        {
            return MissReadBinding(ref memo.RefusedFill);
        }

        _labelTouchClass = -1;
        var labels = _deferredLabelCopies.Count != 0;
        if (labels && (_computeEncoder != 0 || LabelsAtCommit) && DeferredLabelCopiesTouch(address, size))
        {
            return MissReadBinding(ref memo.RefusedLabel);
        }

        var staged = _stagedStorageCopies is { Count: > 0 };
        if (staged && _stagedStorageCopies!.MayConcern(address, size))
        {
            return MissReadBinding(ref memo.RefusedStaged);
        }

        if (!_guestMemory.UseWindowForHit(window, out var touched))
        {
            return MissReadBinding(ref memo.MissWindowRetired);
        }

        memo.WindowTouches += touched ? 1 : 0;
        MetalGpuReferenceCensus.Mark(address, size);
        memo.Hits++;
        if (side.Serial != _submissionSerial)
        {
            memo.HitsNewCommandBuffer++;
            side.Serial = _submissionSerial;
        }

        memo.HitsWithPending += fills || labels || staged ? 1 : 0;
        _readBindingHitIndex = index;
        range = new MetalGuestRange(entry.Buffer, entry.Offset, size, entry.GpuAddress);
        return true;
    }

    // After the full chain of a read-only binding: kept when the flush found no image over the range and the range
    // resolved in place to a window that is still in the partition. The stamps are those read before the chain, so
    // anything that moved while it ran makes the entry miss.
    private void StoreReadBindingRanges(
        ulong address, ulong size, bool computeStage, in MetalReadBindingMemo.Stamp before, bool flushFoundImage,
        ulong inPlaceAlias, MetalWindowPartition.Window? window, in MetalGuestRange range)
    {
        var memo = ReadBindingMemoV2;
        if (!ReadBindingTracesOff)
        {
            return;
        }

        if (flushFoundImage)
        {
            memo.StoreSkipFlushed++;
            return;
        }

        if (inPlaceAlias == 0 || window is null || !range.IsValid || range.Buffer != window.Buffer || range.Size != size)
        {
            memo.StoreSkipNotInPlace++;
            return;
        }

        if (size > uint.MaxValue)
        {
            memo.StoreSkipLarge++;
            return;
        }

        if (Volatile.Read(ref window.Retired) || !_guestMemory.InPlaceAnswerStands(inPlaceAlias, size))
        {
            memo.StoreSkipRetiredAfter++;
            return;
        }

        // What the flush answered by: no owner pages (checked live), or the images over the range now, listed with the
        // version they hold at; more than two are not kept.
        var tier = MetalReadBindingMemoV2.ImageTier.NoOwnerPages;
        MetalCachedImage? first = null;
        MetalCachedImage? second = null;
        var count = 0;
        var imageSet = 0L;
        if (_imageCache.MayOwn(address, size))
        {
            _imageCache.ListOverlapping(address, size, out first, out second, out count, out imageSet);
            if (count > 2)
            {
                memo.StoreSkipImageMany++;
                return;
            }

            tier = count == 0 ? MetalReadBindingMemoV2.ImageTier.NoImage : MetalReadBindingMemoV2.ImageTier.Held;
        }

        var index = memo.Insert(address, (uint)size, computeStage);
        ref var entry = ref memo.At(index);
        entry.Alias = inPlaceAlias;
        entry.Window = window;
        entry.Buffer = range.Buffer;
        entry.Offset = range.Offset;
        entry.GpuAddress = range.GpuAddress;
        entry.Tier = tier;
        entry.HeldImages = (byte)count;
        entry.ImageSet = imageSet;
        ref var side = ref memo.SideAt(index);
        side.Trap = before.Trap;
        side.Serial = _submissionSerial;
        side.Image0 = first;
        side.Image1 = second;
    }

    // =verify, before the full chain runs behind a v2 hit: the hit pinned and touched its window (the chain would
    // do it too, so this is checked first).
    private void VerifyReadBindingHitClaims(ulong address, ulong size, bool computeStage, in MetalGuestRange memo)
    {
        var index = _readBindingHitIndex;
        var entry = index >= 0 ? ReadBindingMemoV2.At(index) : default;
        if (entry.Window is not { } window || entry.Address != address || entry.Size != size)
        {
            throw Fatal(
                $"A memoised read-only binding has no entry behind it: address=0x{address:X16} size=0x{size:X} compute={computeStage}.");
        }

        if (!_guestMemory.WindowUsedThisSubmission(window))
        {
            throw Fatal(
                $"A memoised read-only binding did not pin and touch its window for this command buffer: address=0x{address:X16} " +
                $"size=0x{size:X} window={window.Id} retired={window.Retired} memo={memo}.");
        }

        // The images it answered by: none owns a page, or exactly the images it holds are over the range.
        var side = ReadBindingMemoV2.SideAt(index);
        var mayOwn = _imageCache.MayOwn(address, size);
        _imageCache.ListOverlapping(address, size, out var first, out var second, out var count, out _);
        var holds = entry.Tier switch
        {
            MetalReadBindingMemoV2.ImageTier.NoOwnerPages => !mayOwn,
            MetalReadBindingMemoV2.ImageTier.NoImage => count == 0,
            _ => count == entry.HeldImages &&
                (ReferenceEquals(first, side.Image0) || ReferenceEquals(first, side.Image1)) &&
                (count == 1 || ReferenceEquals(second, side.Image0) || ReferenceEquals(second, side.Image1)),
        };
        if (!holds)
        {
            throw Fatal(
                $"A memoised read-only binding's images moved under it: address=0x{address:X16} size=0x{size:X} " +
                $"tier={entry.Tier} held={entry.HeldImages} may_own={mayOwn} overlapping={count}.");
        }
    }

    /// <summary>What the chain behind a read-only binding can do, read on both sides of it in the verify mode.</summary>
    private readonly record struct ReadBindingChainSnapshot(
        int LabelCopies, bool LandLabels, int Fills, long FillsLandedEarly, nint RenderEncoder, nint ComputeEncoder,
        long PassEndsOther, bool StoresOver, long WindowMoves, long WindowRenewals, long ImageWritebacks)
    {
        public static ReadBindingChainSnapshot Take(MetalRenderHost host, ulong address, ulong size) => new(
            host._deferredLabelCopies.Count,
            host._landDeferredLabels,
            host._deferredFills.Count,
            host._deferredFillsLandedEarly,
            host._renderEncoder,
            host._computeEncoder,
            host._passEndOther,
            host._stagedStorageCopies is { Count: > 0 } staged && staged.HasRecordedStoresOver(address, size),
            host._guestMemory.WindowLayoutCounts.Moves,
            host._guestMemory.SingleCopyWindowCounts.Renewals,
            host._imageCache.Counters.Writebacks);
    }
}
