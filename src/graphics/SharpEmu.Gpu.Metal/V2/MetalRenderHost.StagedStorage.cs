// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Scheduling;

namespace SharpEmu.Gpu.Metal.V2;

internal sealed partial class MetalRenderHost
{
    private MetalStagedStorage? _stagedStorageCopies;

    private MetalStagedRangeCache? _stagedRanges;

    private MetalStagedStorage StagedStorage => _stagedStorageCopies ??= new MetalStagedStorage(_guestMemory, _stagedRanges);

    private bool _encodingStagedStores;

    /// <summary>
    /// The buffer a storage binding reads and writes: its window when one covers the
    /// range, otherwise a staged copy that sees the GPU's earlier writes and whose stores
    /// reach guest memory (<see cref="MetalStagedStorage"/>).
    /// </summary>
    private MetalGuestRange ResolveStorageBuffer(ulong address, ulong size, bool written)
    {
        using var phaseScope = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.StorageBufferResolve);
        OrderAfterDeferredCopies(address, size);
        var staged = StagedStorage;
        if (staged.TryShare(address, size, written, out var shared))
        {
            return shared;
        }

        OrderAfterStagedStores(address, size);
        if (_guestMemory.TryResolveInPlace(address, size, out var window))
        {
            return window;
        }

        if (!written && TryBindGuestView(address, size, out var view))
        {
            return view;
        }

        if (MetalStagedRangeCache.Eligible(size) && _stagedRanges is not null)
        {
            var cached = BindStagedRange(address, size, written);
            if (cached.IsValid)
            {
                return cached;
            }
        }

        var staging = _guestMemory.Stage(address, size);
        if (!staging.IsValid || _guestMemory.IsInPlace(staging))
        {
            // A range too large to stage binds its window prefix in place: that is guest
            // memory, ordered like any window binding, with no copy to gather into or
            // write back from.
            return staging;
        }

        // The CPU staged what guest memory holds now. Bytes the GPU has yet to write
        // there are copied in by the GPU, behind that work.
        var gathered = PendingSerialHinted(address, size) != 0;
        if (!written && !gathered)
        {
            return staging;
        }

        MetalGuestRange pristine;
        if (gathered)
        {
            pristine = _guestMemory.Allocate(staging.Size, 256, out _);
            OrderAfterDeferredCopies(address, staging.Size);
            var context = ((IMetalImageCacheOwner)this).BeginTransfer(MetalTransferCause.GuestCopy);
            staged.EncodeGather(context.CommandBuffer, address, staging, pristine);
            if (context.WantsEncodedLabels)
            {
                context.ProfileEncoded($"stage storage bytes=0x{staging.Size:X}");
            }
        }
        else
        {
            pristine = _guestMemory.Duplicate(staging);
        }

        if (!pristine.IsValid)
        {
            throw GpuFatal.Fatal(
                $"A staged storage buffer has no pristine copy: address=0x{address:X16} size=0x{staging.Size:X}.");
        }

        staged.Add(address, size, staging, pristine, written);
        return staging;
    }

    private long _guestViewRefusedWrites;

    /// <summary>
    /// A read of a range no window covers through a guest view (MetalGuestMemory.GuestViews.cs):
    /// guest memory itself, no copy. Not when this command buffer has recorded a write to any
    /// of the bytes: that write went through another GPU address (a window, a fill, a staged
    /// copy's writeback), and inside one command buffer a read through the view would not see
    /// it. The staged paths gather such bytes behind the write instead.
    /// </summary>
    private bool TryBindGuestView(ulong address, ulong size, out MetalGuestRange range)
    {
        range = default;
        if (!_guestMemory.MayUseGuestView(size))
        {
            return false;
        }

        if (PendingSerialHinted(address, size) >= SubmissionSerial)
        {
            Interlocked.Increment(ref _guestViewRefusedWrites);
            return false;
        }

        return _guestMemory.TryResolveGuestView(address, size, out range);
    }

    private string GuestViewCounters()
    {
        var v = _guestMemory.GuestViewCounts;
        return $" guest_views={v.Alive} guest_view_mb={v.Bytes / (1024 * 1024)} guest_views_created={v.Created} " +
            $"guest_views_released={v.Released} guest_view_binds={v.Binds} guest_view_bind_mb={v.BindBytes / (1024 * 1024)} " +
            $"guest_view_unmapped={v.Invalidated} guest_view_refused={v.CreateFailures} guest_view_unbacked={v.Unbacked} " +
            $"guest_view_refused_writes={Interlocked.Read(ref _guestViewRefusedWrites)} " +
            $"oversized_staged={_guestMemory.OversizedStaged} oversized_prefix={_guestMemory.OversizedInPlace}";
    }

    /// <summary>
    /// The cached staging of a range no window covers (<see cref="MetalStagedRangeCache"/>): the
    /// buffer a bind earlier made while nothing has written its bytes since, otherwise a
    /// fresh fill of one no in-flight submission can read. Invalid when the cache is full.
    /// </summary>
    private MetalGuestRange BindStagedRange(ulong address, ulong size, bool written)
    {
        var cache = _stagedRanges!;
        EnsureCommandBuffer();
        var hit = cache.Acquire(address, size, written, _submissionSerial, CompletedSerial, out var entry);
        if (entry is null)
        {
            return default;
        }

        var staged = StagedStorage;
        if (hit)
        {
            if (MetalStagedRangeCache.VerifyMode != MetalStagedRangeCache.VerifyKind.Off)
            {
                VerifyStagedHit(cache, entry);
            }

            if (written)
            {
                staged.Bind(entry, written: true);
            }

            return entry.Staging;
        }

        FillStagedRange(entry, staged);
        if (written)
        {
            staged.Bind(entry, written: true);
        }

        return entry.Staging;
    }

    private unsafe void FillStagedRange(MetalStagedRangeCache.Entry entry, MetalStagedStorage staged)
    {
        var size = entry.Size;
        var bytes = new Span<byte>((void*)entry.StagingBuffer!.Contents, checked((int)size));
        _guestMemory.ReadForStaging(entry.Address, bytes);
        if (_gpuWrites.PendingSerial(entry.Address, size, CompletedSerial) != 0)
        {
            // The CPU staged what guest memory holds now. Bytes the GPU has yet to write
            // there are copied in by the GPU, behind that work.
            entry.Gathered = true;
            OrderAfterDeferredCopies(entry.Address, size);
            var context = ((IMetalImageCacheOwner)this).BeginTransfer(MetalTransferCause.GuestCopy);
            _stagedRanges!.Touch(entry, context.SubmissionSerial);
            staged.EncodeGather(context.CommandBuffer, entry.Address, entry.Staging, entry.Pristine);
            if (context.WantsEncodedLabels)
            {
                context.ProfileEncoded($"stage storage bytes=0x{size:X}");
            }
        }
        else if (entry.Pristine.IsValid)
        {
            bytes.CopyTo(new Span<byte>((void*)entry.PristineBuffer!.Contents, checked((int)size)));
        }
    }

    // SHARPEMU_METAL_V2_STAGED_CACHE_VERIFY=1 compares a hit with what guest memory holds now;
    // =sample does so for a slice of one hit in 64 of each entry.
    private unsafe void VerifyStagedHit(MetalStagedRangeCache cache, MetalStagedRangeCache.Entry entry)
    {
        var sample = MetalStagedRangeCache.VerifyMode == MetalStagedRangeCache.VerifyKind.Sample;
        entry.HitCount++;
        if ((sample && entry.HitCount % MetalStagedRangeCache.SampleEvery != 0) ||
            entry.Dirty || (entry.Gathered && entry.StagedSerial > CompletedSerial) ||
            _gpuWrites.PendingSerial(entry.Address, entry.Size, CompletedSerial) != 0)
        {
            return;
        }

        var cached = new ReadOnlySpan<byte>((void*)entry.StagingBuffer!.Contents, checked((int)entry.Size));
        var address = entry.Address;
        var offset = MetalStagedRangeCache.FirstMismatch(
            cached, (at, destination) => _guestMemory.ReadForStaging(address + at, destination), sample, entry.HitCount);
        cache.NoteVerify(offset >= 0);
        if (offset >= 0)
        {
            Console.Error.WriteLine(
                $"[METAL-V2][WARN][STAGED-VERIFY] mismatch addr=0x{entry.Address:X16} size=0x{entry.Size:X} off=0x{offset:X}");
        }
    }

    /// <summary>
    /// A binding over bytes a staged copy has recorded stores into is recorded after those
    /// stores reach guest memory: ending the open encoder writes them back.
    /// </summary>
    private void OrderAfterStagedStores(ulong address, ulong size)
    {
        if (_stagedStorageCopies is { Count: > 0 } staged && staged.HasRecordedStoresOver(address, size))
        {
            EndEncoders(PassEndCause.Other);
        }
    }

    /// <summary>
    /// Writes every recorded store of a staged copy back into guest memory. Called where
    /// no encoder is open, like the deferred label copies, so the copies run after the
    /// work that stored and before whatever is recorded next.
    /// </summary>
    private void EncodeStagedStores()
    {
        if (_encodingStagedStores || _stagedStorageCopies is not { Count: > 0 } staged || !staged.HasRecordedStores)
        {
            return;
        }

        _encodingStagedStores = true;
        try
        {
            var written = staged.EncodeWritebacks(EnsureCommandBuffer(), _submissionSerial);
            NoteRecordedWork();
            foreach (var (address, size) in written)
            {
                // The copy lands the stores in guest memory without re-arming DCC claims there.
                _imageCache.ForgetStandingDccClaims(address, size);
                NoteGuestMemoryWrite(address, size, MetalGpuWriteOrigin.StorageBuffer);
            }

            staged.MarkWrittenBack();
            if (PassProfileEnabled)
            {
                PassProfileNoteBetween($"staged-writeback n={written.Count}");
            }
        }
        finally
        {
            _encodingStagedStores = false;
        }
    }
}
