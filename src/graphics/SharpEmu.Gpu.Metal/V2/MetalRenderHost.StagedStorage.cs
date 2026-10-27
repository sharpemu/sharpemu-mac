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
    internal MetalGuestRange ResolveStorageBuffer(ulong address, ulong size, bool written)
    {
        using var phaseScope = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.StorageDeferredLabels);
        OrderAfterDeferredCopies(address, size);
        phaseScope.SwitchPhase(SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.StorageSharedLookup);
        var staged = StagedStorage;
        if (staged.TryShare(address, size, written, out var shared))
        {
            return shared;
        }

        phaseScope.SwitchPhase(SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.StorageStoreOrder);
        OrderAfterStagedStores(address, size);
        phaseScope.SwitchPhase(SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.StorageBufferResolve);
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

        // The GPU copy is a transfer, which ends an open render pass; the CPU staging only ever
        // did when the GPU still had to write the bytes, so with a pass open it still does.
        if (written && size >= WholeGatherMinBytes && (_renderEncoder == 0 || PendingSerialHinted(address, size) != 0) &&
            StageWrittenOnGpu(address, size) is { IsValid: true } gpuStaged)
        {
            return gpuStaged;
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
            staged.EncodeGather(context.CommandBuffer, address, staging, pristine, context);
            if (context.WantsEncodedLabels)
            {
                context.ProfileEncoded($"stage storage bytes=0x{staging.Size:X}");
            }
        }
        else if (SnapshotsOnGpu(_renderEncoder != 0, _computeEncoder != 0, staging.Size))
        {
            pristine = _guestMemory.Allocate(staging.Size, 256, out _);
            SnapshotOnGpu(staging, pristine, null);
        }
        else
        {
            pristine = _guestMemory.Duplicate(staging);
            Interlocked.Add(ref _cpuPristineBytes, (long)staging.Size);
        }

        if (!pristine.IsValid)
        {
            throw GpuFatal.Fatal(
                $"A staged storage buffer has no pristine copy: address=0x{address:X16} size=0x{staging.Size:X}.");
        }

        staged.Add(address, size, staging, pristine, written);
        return staging;
    }

    // A written range at least this large is staged by the GPU alone (StageWrittenOnGpu).
    internal const ulong WholeGatherMinBytes = 4UL * 1024 * 1024;

    private long _wholeGathers;
    private long _wholeGatherBytes;

    /// <summary>
    /// A written range no window covers, staged by the GPU alone: the copy and its pristine
    /// twin are both filled from the windows over the range's runs, behind the work that
    /// writes there. The CPU stages nothing. Before, it copied the whole range into the ring
    /// first, and then either copied it again for the pristine twin or had the GPU gather
    /// over every byte of it: Black Myth: Wukong binds a 128 MiB written range two runs of
    /// the backing make about sixteen times a second, a 2 GB/s memcpy on the worker. Invalid
    /// when the range is too large to stage whole; a range with a run no window covers is
    /// staged by the CPU and gathered as before.
    /// </summary>
    private MetalGuestRange StageWrittenOnGpu(ulong address, ulong size)
    {
        var staging = _guestMemory.ReserveWholeStaging(address, size);
        if (!staging.IsValid)
        {
            return default;
        }

        var pristine = _guestMemory.Allocate(staging.Size, 256, out _);
        OrderAfterDeferredCopies(address, staging.Size);
        var context = ((IMetalImageCacheOwner)this).BeginTransfer(MetalTransferCause.GuestCopy);
        if (StagedStorage.TryEncodeWholeGather(context.CommandBuffer, address, staging, pristine, context))
        {
            Interlocked.Increment(ref _wholeGathers);
            Interlocked.Add(ref _wholeGatherBytes, (long)staging.Size);
        }
        else
        {
            _guestMemory.FillStaging(address, staging);
            StagedStorage.EncodeGather(context.CommandBuffer, address, staging, pristine, context);
        }

        if (context.WantsEncodedLabels)
        {
            context.ProfileEncoded($"stage storage on the gpu bytes=0x{staging.Size:X}");
        }

        StagedStorage.Add(address, size, staging, pristine, written: true);
        return staging;
    }

    // SHARPEMU_METAL_V2_GPU_PRISTINE=0 copies the pristine twin of a written staged copy on the CPU again.
    internal static readonly bool GpuPristine =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_GPU_PRISTINE") != "0";

    // Below this a blit encoder costs the worker about what the memcpy does.
    internal const ulong GpuPristineMinBytes = 64 * 1024;

    private long _cpuPristineBytes;
    private long _gpuPristines;
    private long _gpuPristineBytes;

    internal long GpuPristines => Interlocked.Read(ref _gpuPristines);

    /// <summary>
    /// True when the pristine twin of a written copy the CPU staged is better copied from it by the GPU: a
    /// blit ahead of the shader that stores into the copy, instead of a second memcpy of the whole range on the
    /// worker, which costs as much as staging it did. Never with a pass open, which the blit would end; with a
    /// compute encoder open only from the size at which the GPU stages a written range whole anyway
    /// (<see cref="WholeGatherMinBytes"/>), since the blit ends that encoder too.
    /// </summary>
    internal static bool SnapshotsOnGpu(bool renderOpen, bool computeOpen, ulong size) =>
        GpuPristine && !renderOpen && size >= GpuPristineMinBytes && (!computeOpen || size >= WholeGatherMinBytes);

    private void SnapshotOnGpu(in MetalGuestRange staging, in MetalGuestRange pristine, MetalStagedRangeCache.Entry? entry)
    {
        var context = ((IMetalImageCacheOwner)this).BeginTransfer(MetalTransferCause.GuestCopy);
        if (entry is not null)
        {
            // The blit reads and writes the entry's buffers in this submission.
            _stagedRanges!.Touch(entry, context.SubmissionSerial);
        }

        StagedStorage.EncodeSnapshot(context.CommandBuffer, staging, pristine, context);
        Interlocked.Increment(ref _gpuPristines);
        Interlocked.Add(ref _gpuPristineBytes, (long)staging.Size);
        if (context.WantsEncodedLabels)
        {
            context.ProfileEncoded($"snapshot staged storage bytes=0x{staging.Size:X}");
        }
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

        if (_gpuWrites.WrittenSince(address, size, SubmissionSerial))
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
            $"oversized_staged={_guestMemory.OversizedStaged} oversized_prefix={_guestMemory.OversizedInPlace} " +
            $"gpu_staged={Interlocked.Read(ref _wholeGathers)} gpu_staged_mb={Interlocked.Read(ref _wholeGatherBytes) / (1024 * 1024)} " +
            $"pristine_cpu_mb={Interlocked.Read(ref _cpuPristineBytes) / (1024 * 1024)} " +
            $"pristine_gpu={Interlocked.Read(ref _gpuPristines)} pristine_gpu_mb={Interlocked.Read(ref _gpuPristineBytes) / (1024 * 1024)}";
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

    private MetalStagedRangeCache.GuestReader? _stagingReader;

    private unsafe void FillStagedRange(MetalStagedRangeCache.Entry entry, MetalStagedStorage staged)
    {
        if (entry.Plan is not null)
        {
            // Stores touched a few pages of the entry: those are copied, unless the GPU still has
            // bytes to write there, which the gather below picks up for the whole entry.
            if (_gpuWrites.PendingSerial(entry.Address, entry.Size, CompletedSerial) == 0 &&
                _stagedRanges!.RefillDirty(entry, _stagingReader ??= _guestMemory.ReadForStaging))
            {
                return;
            }

            entry.Plan = null;
        }

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
            staged.EncodeGather(context.CommandBuffer, entry.Address, entry.Staging, entry.Pristine, context);
            if (context.WantsEncodedLabels)
            {
                context.ProfileEncoded($"stage storage bytes=0x{size:X}");
            }
        }
        else if (entry.Pristine.IsValid)
        {
            if (SnapshotsOnGpu(_renderEncoder != 0, _computeEncoder != 0, size))
            {
                SnapshotOnGpu(entry.Staging, entry.Pristine, entry);
            }
            else
            {
                bytes.CopyTo(new Span<byte>((void*)entry.PristineBuffer!.Contents, checked((int)size)));
                Interlocked.Add(ref _cpuPristineBytes, (long)size);
            }
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
            var written = staged.EncodeWritebacks(EnsureCommandBuffer(), _submissionSerial, this);
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
