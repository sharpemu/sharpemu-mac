// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Rendering;

namespace SharpEmu.Gpu.Metal.V2;

// A recognized fill or copy kernel over the exact bytes of a cached image becomes a clear
// or a texture copy (MetalImageCache.KernelFills). The executor has already decoded the
// kernel's arguments at bind time; what is left here is whether an image covers the
// range exactly and the bookkeeping a storage store into it would have done.
//
// Ordering: the clear or copy is a transfer, so BeginTransfer ends the open encoder and
// encodes the staged stores and deferred label copies recorded before it, exactly as for
// any other transfer; a folded clear waits in the pending-clear list, which every
// transfer naming the texture and every pass that binds it drains first.
internal sealed partial class MetalRenderHost
{
    private long _kernelFills;
    private long _kernelCopies;
    // Keyed by the kernel the reason is for, so the CHURN line tells fills from copies: the
    // image cache gives both the same reasons (no-image, shape), and a copy whose source is
    // shorter than it is offered as a fill pattern. The executor's own refusals name their kind.
    private readonly ConcurrentDictionary<(KernelKind Kind, string Reason), long> _kernelFillFallbacks = new();

    private enum KernelKind : byte
    {
        // Reasons the executor names itself (copy-shape, copy-descriptor, ...).
        Named,
        Fill,
        Copy,
    }

    private static readonly string[] KernelKindPrefixes = ["", "fill:", "copy:"];

    /// <summary>A recognized fill or copy kernel ran as a dispatch, and why.</summary>
    public void NoteKernelFillFallback(string reason) => NoteKernelFallback(KernelKind.Named, reason);

    private void NoteKernelFallback(KernelKind kind, string reason)
    {
        if (MetalV2Diagnostics.KernelFills)
        {
            _kernelFillFallbacks.AddOrUpdate((kind, reason), 1, static (_, count) => count + 1);
        }
    }

    /// <summary>A kernel fill of repeating dwords, as an image clear.</summary>
    public bool TryClearImageFromKernelFill(ulong address, ulong size, ReadOnlySpan<uint> pattern)
    {
        if (!MetalV2Diagnostics.KernelFills)
        {
            return false;
        }

        if (TryClearImageFromKernelFillCore(address, size, pattern))
        {
            return true;
        }

        // Refused: the fill runs as a dispatch, unless its destination already holds the
        // pattern (SHARPEMU_METAL_V2_HOST_KERNEL_COPY).
        return ChecksRefusedKernels && !pattern.IsEmpty && size % sizeof(uint) == 0 &&
            TakeIdenticalKernel(
                EvaluateKernelFill(address, size / sizeof(uint), pattern), address, size / sizeof(uint), 0, (ulong)pattern.Length, pattern);
    }

    private bool TryClearImageFromKernelFillCore(ulong address, ulong size, ReadOnlySpan<uint> pattern)
    {
        if (pattern.IsEmpty || pattern.Length * sizeof(uint) > MetalKernelFill.MaxPatternBytes)
        {
            NoteKernelFallback(KernelKind.Fill, "pattern-size");
            return false;
        }

        Span<byte> bytes = stackalloc byte[pattern.Length * sizeof(uint)];
        for (var word = 0; word < pattern.Length; word++)
        {
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(bytes[(word * sizeof(uint))..], pattern[word]);
        }

        return TryInterceptKernelFill(KernelKind.Fill, address, size, bytes);
    }

    /// <summary>
    /// A kernel copy of <c>dst[i] = src[i % sourceWords]</c> for <paramref name="words"/>
    /// dwords. Guest memory is the GPU's memory, so a plain word copy has no host-side
    /// shortcut; what the host can do is keep an image over the range on the GPU: a
    /// source shorter than the copy is a fill pattern (a clear), and a source as long is
    /// a copy of another image (a texture copy).
    /// </summary>
    public bool TryCopyWordsOnHost(ulong destination, ulong source, ulong sourceWords, ulong words)
    {
        if (TryCopyWordsOnHostCore(destination, source, sourceWords, words))
        {
            return true;
        }

        // Refused: a copy whose destination already holds its result writes nothing
        // (SHARPEMU_METAL_V2_HOST_KERNEL_COPY). Otherwise it runs as the dispatch it is, and
        // a read of its destination before that submission completes can be answered from
        // its source.
        if (MetalV2Diagnostics.KernelFills && ChecksRefusedKernels &&
            TakeIdenticalKernel(EvaluateKernelCopy(destination, words, source, sourceWords), destination, words, source, sourceWords, default))
        {
            return true;
        }

        ArmPendingCopy(destination, source, sourceWords, words);
        return false;
    }

    private bool TryCopyWordsOnHostCore(ulong destination, ulong source, ulong sourceWords, ulong words)
    {
        if (!MetalV2Diagnostics.KernelFills)
        {
            return false;
        }

        if (words == 0 || sourceWords == 0 || words > ulong.MaxValue / sizeof(uint))
        {
            NoteKernelFallback(KernelKind.Copy, "size");
            return false;
        }

        var size = words * sizeof(uint);
        if (sourceWords >= words)
        {
            return TryInterceptKernelCopy(destination, source, size);
        }

        if (sourceWords * sizeof(uint) > MetalKernelFill.MaxPatternBytes)
        {
            NoteKernelFallback(KernelKind.Copy, "pattern-size");
            return false;
        }

        // The pattern is read now, while the kernel would have read it on the GPU: bytes a
        // recorded store or a GPU-written image still owns would cost a wait for the GPU,
        // which is worse than the round trip this saves.
        var patternSize = sourceWords * sizeof(uint);
        if (_gpuWrites.PendingSerial(source, patternSize, CompletedSerial) != 0 ||
            _imageCache.HasGpuModifiedImageBytes(source, patternSize))
        {
            NoteKernelFallback(KernelKind.Copy, "pattern-pending");
            return false;
        }

        Span<byte> pattern = stackalloc byte[(int)patternSize];
        if (!TryReadGuest(source, pattern))
        {
            NoteKernelFallback(KernelKind.Copy, "pattern-unreadable");
            return false;
        }

        return TryInterceptKernelFill(KernelKind.Copy, destination, size, pattern);
    }

    private bool TryInterceptKernelFill(KernelKind kind, ulong address, ulong size, ReadOnlySpan<byte> pattern)
    {
        _ = EnsureCommandBuffer();
        if (!_imageCache.TryClearImageFromKernelFill(address, size, pattern, out var fallback))
        {
            NoteKernelFallback(kind, fallback ?? "other");
            return false;
        }

        // The bytes are the image's now, as after a storage store into them: pending
        // command-processor bytes and staged copies there are stale, and a read waits.
        _imageCache.ForgetStandingDccClaims(address, size);
        NoteGuestMemoryWrite(address, size, MetalGpuWriteOrigin.Fill);
        Interlocked.Increment(ref _kernelFills);
        return true;
    }

    private bool TryInterceptKernelCopy(ulong destination, ulong source, ulong size)
    {
        _ = EnsureCommandBuffer();
        if (!_imageCache.TryBeginKernelCopy(
                destination, source, size, out var sourceImage, out var destinationImage, out var destinationIdentifier, out var fallback))
        {
            NoteKernelFallback(KernelKind.Copy, fallback ?? "other");
            return false;
        }

        var shape = destinationImage!.Shape;
        for (var level = 0u; level < shape.MipLevels; level++)
        {
            var range = new SubresourceRange(level, 1, 0, shape.ArrayLength);
            var extent = new GpuExtent3D(Math.Max(shape.Width >> (int)level, 1), Math.Max(shape.Height >> (int)level, 1), 1);
            CopyTextureSubresource(sourceImage!, destinationImage, in range, in range, in extent);
        }

        _imageCache.CompleteKernelCopy(destinationIdentifier);
        _imageCache.ForgetStandingDccClaims(destination, size);
        NoteGuestMemoryWrite(destination, size, MetalGpuWriteOrigin.Copy);
        Interlocked.Increment(ref _kernelCopies);
        return true;
    }

    /// <summary>The kernel-fill fields of the CHURN line; resets them.</summary>
    internal string TakeKernelFillCounters()
    {
        var builder = new System.Text.StringBuilder()
            .Append(" fill_kernels=").Append(Interlocked.Exchange(ref _kernelFills, 0))
            .Append(" copy_kernels=").Append(Interlocked.Exchange(ref _kernelCopies, 0))
            .Append(" fill_fallback=[");
        var first = true;
        foreach (var key in _kernelFillFallbacks.Keys.OrderBy(static key => key.Kind).ThenBy(static key => key.Reason, StringComparer.Ordinal))
        {
            if (_kernelFillFallbacks.TryRemove(key, out var count))
            {
                builder.Append(first ? string.Empty : ",").Append(KernelKindPrefixes[(int)key.Kind]).Append(key.Reason).Append('=').Append(count);
                first = false;
            }
        }

        return builder.Append(']').Append(TakeHostKernelCopyCounters()).ToString();
    }

    // Whether a refused kernel's result is already in memory: every destination word holds
    // what the kernel would store there (phase A of running the kernel on the host). The
    // answer is taken only where nothing the GPU has yet to do can change either side:
    //   - the destination is no image's and no surface metadata's (an image may have an
    //     upload recorded that still reads it, and metadata bytes mean more than memory says);
    //   - no recorded write that has not completed covers the destination, and no pending
    //     command-processor or label bytes do (they land before the kernel would run);
    //   - the source is final, as for copy read-through (IsCopySourceFinal), no pending
    //     label bytes cover it, and it is no surface metadata. A fill's pattern is its own words.
    // Then nothing is written, so no shader recorded earlier can read a changed byte. The
    // destinations are small tables: a wider kernel (Demon's Souls has one of 4 MiB) is
    // refused before anything is read. The check runs only when its answer is used
    // (SHARPEMU_METAL_V2_HOST_KERNEL_COPY) or counted (SHARPEMU_METAL_V2_HOST_KERNEL_COPY_DRYRUN):
    // otherwise a refused kernel reads no memory and takes no lock it did not take before.
    private const ulong MaxHostKernelCopyBytes = 64 * 1024;

    /// <summary>SHARPEMU_METAL_V2_HOST_KERNEL_COPY_DRYRUN, read once; tests set it per host before any kernel.</summary>
    internal bool HostKernelCopyDryRun { get; set; } = MetalV2Diagnostics.HostKernelCopyDryRun;

    private bool ChecksRefusedKernels => HostKernelCopyDryRun || HostKernelCopyMode != MetalHostKernelCopyMode.Off;

    private enum KernelCopyOutcome : byte
    {
        Identical,
        Differs,
        SourcePending,
        DestinationWrittenPending,
        DestinationImage,
        Size,
        Unreadable,
        SourceMetadata,
    }

    private static readonly string[] KernelCopyOutcomeNames =
        ["would_elide_identical", "would_write", "src_pending", "dst_written_pending", "dst_image", "size", "unreadable", "src_metadata"];

    private readonly long[] _kernelCopyOutcomes = new long[KernelCopyOutcomeNames.Length];
    private long _kernelCopyFills;
    private long _kernelCopyChangedBytes;
    private long _kernelCopyChangedPages;
    // The words compared, read on the worker only; grown to the largest kernel seen.
    private uint[] _kernelCopyDestination = [];
    private uint[] _kernelCopySource = [];

    internal (long Identical, long Differs, long SourcePending, long DestinationWrittenPending, long DestinationImage, long Size,
        long Unreadable, long SourceMetadata) HostKernelCopyOutcomes =>
        (Interlocked.Read(ref _kernelCopyOutcomes[0]), Interlocked.Read(ref _kernelCopyOutcomes[1]), Interlocked.Read(ref _kernelCopyOutcomes[2]),
            Interlocked.Read(ref _kernelCopyOutcomes[3]), Interlocked.Read(ref _kernelCopyOutcomes[4]), Interlocked.Read(ref _kernelCopyOutcomes[5]),
            Interlocked.Read(ref _kernelCopyOutcomes[6]), Interlocked.Read(ref _kernelCopyOutcomes[7]));

    internal (long Bytes, long Pages) HostKernelCopyChanges =>
        (Interlocked.Read(ref _kernelCopyChangedBytes), Interlocked.Read(ref _kernelCopyChangedPages));

    private KernelCopyOutcome EvaluateKernelFill(ulong destination, ulong words, ReadOnlySpan<uint> pattern)
    {
        Interlocked.Increment(ref _kernelCopyFills);
        return CountKernelCopyOutcome(EvaluateKernelCopyCore(destination, words, 0, (ulong)pattern.Length, pattern));
    }

    private KernelCopyOutcome EvaluateKernelCopy(ulong destination, ulong words, ulong source, ulong period) =>
        CountKernelCopyOutcome(EvaluateKernelCopyCore(destination, words, source, period, default));

    private KernelCopyOutcome CountKernelCopyOutcome(KernelCopyOutcome outcome)
    {
        Interlocked.Increment(ref _kernelCopyOutcomes[(int)outcome]);
        return outcome;
    }

    // A fill passes its pattern; a copy passes an empty one and reads min(period, words) source words.
    private KernelCopyOutcome EvaluateKernelCopyCore(ulong destination, ulong words, ulong source, ulong period, ReadOnlySpan<uint> pattern)
    {
        var fill = !pattern.IsEmpty;
        var sourceWords = Math.Min(period, words);
        if (words == 0 || period == 0 || words > MaxHostKernelCopyBytes / sizeof(uint) ||
            (destination & 3) != 0 || destination > ulong.MaxValue - words * sizeof(uint) ||
            (!fill && ((source & 3) != 0 || source > ulong.MaxValue - sourceWords * sizeof(uint))))
        {
            return KernelCopyOutcome.Size;
        }

        var size = words * sizeof(uint);
        if (_imageCache.MayOwn(destination, size) || _imageCache.OverlapsSurfaceMetadata(destination, size))
        {
            return KernelCopyOutcome.DestinationImage;
        }

        if ((_gpuWrites.MayCover(destination, size) && PendingSerialHinted(destination, size) != 0) ||
            HasPendingOverlayBytes(destination, size))
        {
            return KernelCopyOutcome.DestinationWrittenPending;
        }

        var sourceSize = sourceWords * sizeof(uint);
        if (!fill && _imageCache.OverlapsSurfaceMetadata(source, sourceSize))
        {
            return KernelCopyOutcome.SourceMetadata;
        }

        if (!fill && (!IsCopySourceFinal(source, sourceSize, CompletedSerial) ||
            (_labelOverlay.Count != 0 && _labelOverlay.Coverage(source, sourceSize, _completedSerialReader ??= () => CompletedSerial) != MetalOverlayCoverage.None)))
        {
            return KernelCopyOutcome.SourcePending;
        }

        if (_kernelCopyDestination.Length < (int)words)
        {
            _kernelCopyDestination = new uint[words];
        }

        var current = _kernelCopyDestination.AsSpan(0, (int)words);
        ReadOnlySpan<uint> stored = pattern;
        if (!fill)
        {
            if (_kernelCopySource.Length < (int)sourceWords)
            {
                _kernelCopySource = new uint[sourceWords];
            }

            var sourceSpan = _kernelCopySource.AsSpan(0, (int)sourceWords);
            if (!_memory.TryRead(source, System.Runtime.InteropServices.MemoryMarshal.AsBytes(sourceSpan)))
            {
                return KernelCopyOutcome.Unreadable;
            }

            stored = sourceSpan;
        }

        if (!_memory.TryRead(destination, System.Runtime.InteropServices.MemoryMarshal.AsBytes(current)))
        {
            return KernelCopyOutcome.Unreadable;
        }

        var first = ModuloCopyCompare.FirstDifference(current, stored, period);
        if (first < 0)
        {
            return KernelCopyOutcome.Identical;
        }

        // What a host write of the result would cost: the words and the pages it changes.
        var differing = (long)ModuloCopyCompare.DifferingWords(current[first..], stored, period, (ulong)first);
        var pages = 0L;
        foreach (var _ in ModuloCopyCompare.ChangedPages(destination, current, stored, period))
        {
            pages++;
        }

        Interlocked.Add(ref _kernelCopyChangedBytes, differing * sizeof(uint));
        Interlocked.Add(ref _kernelCopyChangedPages, pages);
        return KernelCopyOutcome.Differs;
    }

    // Command-processor or label bytes encoded but not landed over the range.
    private bool HasPendingOverlayBytes(ulong address, ulong size) =>
        (_cpOverlay.Count != 0 && OverlayCoverageHinted(address, size) != MetalOverlayCoverage.None) ||
        (_labelOverlay.Count != 0 &&
            _labelOverlay.Coverage(address, size, _completedSerialReader ??= () => CompletedSerial) != MetalOverlayCoverage.None);

    /// <summary>SHARPEMU_METAL_V2_HOST_KERNEL_COPY, read once; tests set it per host.</summary>
    internal MetalHostKernelCopyMode HostKernelCopyMode { get; set; } = MetalV2Diagnostics.HostKernelCopy;

    private static readonly string[] HostKernelCopyModeNames = ["0", "identical", "verify"];

    private long _kernelCopiesElided;

    internal long KernelCopiesElided => Interlocked.Read(ref _kernelCopiesElided);

    // A kernel found identical: consumed under `identical`, which is the whole of the elision
    // (no write, no write record, no pass end); armed for its checks under `verify`, where it
    // still runs. `stored` is the source the check judged (a copy's source words, a fill's
    // pattern), repeated by `period`.
    private bool TakeIdenticalKernel(
        KernelCopyOutcome outcome, ulong destination, ulong words, ulong source, ulong period, ReadOnlySpan<uint> pattern)
    {
        var mode = HostKernelCopyMode;
        if (mode == MetalHostKernelCopyMode.Verify)
        {
            CheckKernelCopies(CompletedSerial);
            if (outcome == KernelCopyOutcome.Identical)
            {
                var fill = !pattern.IsEmpty;
                ArmKernelCopyCheck(destination, words, source, period,
                    fill ? pattern : _kernelCopySource.AsSpan(0, (int)Math.Min(period, words)), fill);
            }

            return false;
        }

        if (mode != MetalHostKernelCopyMode.Identical || outcome != KernelCopyOutcome.Identical)
        {
            return false;
        }

        Interlocked.Increment(ref _kernelCopiesElided);
        return true;
    }

    // Verify mode: the kernel runs, and two checks say whether consuming it would have left the
    // same bytes. Expected is the result the elision assumed: dst[i] = src[i % period] over the
    // source as it was judged (a fill's pattern).
    //   - The destination side. A blit encoded right before the kernel, in the same command
    //     order, copies the destination into a buffer of its own; once that runs, the bytes the
    //     kernel would have overwritten must already be Expected. This is what catches a GPU
    //     write to the destination the guards missed, which the kernel itself overwrites.
    //   - The source side. Once the kernel's submission completes, the destination must hold
    //     Expected: the kernel stored what the judged source said, so the source did not change
    //     under it. A fill's pattern cannot change, so for a fill this side always holds. A later
    //     write over the destination (a recorded one, or a host write) supersedes this side,
    //     since what it leaves is not the kernel's.
    // The check is armed here and recorded with the kernel write's serial and sequence when the
    // dispatch's storage binding notes the destination, as a pending copy is. A guest CPU write
    // in between is not seen, and shows as a mismatch on either side.
    private sealed class KernelCopyCheck
    {
        public ulong Destination;
        public ulong Source;
        public bool Fill;
        public uint[] Expected = [];
        public ulong Serial;
        public ulong Sequence;
        public bool Superseded;
        // The blit's copy of the destination before the kernel, or 0 when none was encoded.
        public nint Before;
        public nint BeforeContents;

        public ulong End => Destination + (ulong)Expected.Length * sizeof(uint);
    }

    private static readonly nint SelKernelCheckRelease = MetalNative.Selector("release");
    private const int MaxKernelCopyChecks = 1024;
    private const int MaxKernelCopyMismatchLines = 32;
    private KernelCopyCheck? _armedKernelCopyCheck;
    private readonly List<KernelCopyCheck> _kernelCopyChecks = [];
    // The span of the destinations being checked, so a write elsewhere skips the list.
    private ulong _kernelCopyChecksLow = ulong.MaxValue;
    private ulong _kernelCopyChecksHigh;
    private long _kernelCopyChecked;
    private long _kernelCopyMismatches;
    private long _kernelCopyDestinationChecked;
    private long _kernelCopyDestinationMismatches;
    private long _kernelCopyDestinationSkipped;
    private long _kernelCopySuperseded;
    private long _kernelCopyChecksDropped;
    private int _kernelCopyMismatchLines;

    internal (long Checked, long Mismatches, long Superseded, long Dropped) KernelCopyCheckCounters =>
        (Interlocked.Read(ref _kernelCopyChecked), Interlocked.Read(ref _kernelCopyMismatches),
            Interlocked.Read(ref _kernelCopySuperseded), Interlocked.Read(ref _kernelCopyChecksDropped));

    internal (long Checked, long Mismatches, long Skipped) KernelCopyDestinationCheckCounters =>
        (Interlocked.Read(ref _kernelCopyDestinationChecked), Interlocked.Read(ref _kernelCopyDestinationMismatches),
            Interlocked.Read(ref _kernelCopyDestinationSkipped));

    internal int PendingKernelCopyChecks => _kernelCopyChecks.Count;

    private void ArmKernelCopyCheck(ulong destination, ulong words, ulong source, ulong period, ReadOnlySpan<uint> stored, bool fill)
    {
        DisarmKernelCopyCheck();
        var expected = new uint[words];
        var index = 0UL;
        for (var word = 0; word < expected.Length; word++)
        {
            expected[word] = stored[(int)index];
            index = index + 1 == period ? 0 : index + 1;
        }

        var check = new KernelCopyCheck { Destination = destination, Source = source, Fill = fill, Expected = expected };
        EncodeDestinationBefore(check);
        _armedKernelCopyCheck = check;
    }

    // The destination's bytes on the GPU timeline, just before the kernel: a blit from its
    // window, ordered like any transfer after the encoders, staged stores and deferred label
    // copies recorded before it. A destination with no window in place is not copied (a staged
    // copy would hold what the CPU sees now, not what the GPU will).
    private void EncodeDestinationBefore(KernelCopyCheck check)
    {
        var size = (ulong)check.Expected.Length * sizeof(uint);
        if (!_guestMemory.TryResolveInPlace(check.Destination, size, out var window) || window.Size < size)
        {
            Interlocked.Increment(ref _kernelCopyDestinationSkipped);
            return;
        }

        var buffer = _guestMemory.CreateReportBuffer(size, "kernel-copy.verify", out var contents, out _);
        if (buffer == 0)
        {
            Interlocked.Increment(ref _kernelCopyDestinationSkipped);
            return;
        }

        OrderAfterDeferredCopies(check.Destination, size);
        OrderAfterStagedStores(check.Destination, size);
        var context = ((IMetalImageCacheOwner)this).BeginTransfer(MetalTransferCause.GuestCopy);
        var encoder = MetalNative.Send(context.CommandBuffer, SelBlitEncoder);
        MetalMemoryTelemetry.BlitEncoders.Add();
        if (encoder == 0)
        {
            MetalNative.SendVoid(buffer, SelKernelCheckRelease);
            Interlocked.Increment(ref _kernelCopyDestinationSkipped);
            return;
        }

        WaitForPreviousEncoder(encoder, render: false);
        try
        {
            MetalNative.SendCopyBuffer(encoder, SelCopyBuffer, window.Buffer, (nuint)window.Offset, buffer, 0, (nuint)size);
        }
        finally
        {
            UpdateEncoderFence(encoder, render: false);
            MetalNative.SendVoid(encoder, SelEndEncoding);
        }

        check.Before = buffer;
        check.BeforeContents = contents;
    }

    // The command buffer holds the blit's buffer until it completes; this lets go of ours.
    private static void ReleaseKernelCopyCheck(KernelCopyCheck check)
    {
        if (check.Before != 0)
        {
            MetalNative.SendVoid(check.Before, SelKernelCheckRelease);
            check.Before = 0;
            check.BeforeContents = 0;
        }
    }

    private void DisarmKernelCopyCheck()
    {
        if (_armedKernelCopyCheck is { } armed)
        {
            ReleaseKernelCopyCheck(armed);
            _armedKernelCopyCheck = null;
        }
    }

    // Every recorded write: the armed kernel's own store records its check, and any other
    // store over a destination being checked supersedes that check's source side.
    private void NoteKernelCopyCheckWrite(ulong address, ulong size, MetalGpuWriteOrigin origin, ulong serial, ulong sequence)
    {
        if (_armedKernelCopyCheck is { } armed && origin == MetalGpuWriteOrigin.StorageBuffer && sequence != 0 &&
            address <= armed.Destination && address + size >= armed.End)
        {
            _armedKernelCopyCheck = null;
            armed.Serial = serial;
            armed.Sequence = sequence;
            if (_kernelCopyChecks.Count >= MaxKernelCopyChecks)
            {
                ReleaseKernelCopyCheck(_kernelCopyChecks[0]);
                _kernelCopyChecks.RemoveAt(0);
                Interlocked.Increment(ref _kernelCopyChecksDropped);
            }

            _kernelCopyChecks.Add(armed);
            _kernelCopyChecksLow = Math.Min(_kernelCopyChecksLow, armed.Destination);
            _kernelCopyChecksHigh = Math.Max(_kernelCopyChecksHigh, armed.End);
            return;
        }

        SupersedeKernelCopyChecks(address, size);
    }

    private void SupersedeKernelCopyChecks(ulong address, ulong size)
    {
        if (_kernelCopyChecks.Count == 0 || address >= _kernelCopyChecksHigh || address + size <= _kernelCopyChecksLow)
        {
            return;
        }

        foreach (var check in _kernelCopyChecks)
        {
            if (address < check.End && check.Destination < address + size)
            {
                check.Superseded = true;
            }
        }
    }

    private void SupersedeAllKernelCopyChecks()
    {
        foreach (var check in _kernelCopyChecks)
        {
            check.Superseded = true;
        }
    }

    // On the worker, wherever completions are retired: the checks whose kernels have run.
    private void CheckKernelCopies(ulong completed)
    {
        if (_kernelCopyChecks.Count == 0)
        {
            return;
        }

        var kept = 0;
        for (var index = 0; index < _kernelCopyChecks.Count; index++)
        {
            var check = _kernelCopyChecks[index];
            if (check.Serial > completed)
            {
                _kernelCopyChecks[kept++] = check;
                continue;
            }

            CheckKernelCopy(check);
            ReleaseKernelCopyCheck(check);
        }

        _kernelCopyChecks.RemoveRange(kept, _kernelCopyChecks.Count - kept);
        _kernelCopyChecksLow = ulong.MaxValue;
        _kernelCopyChecksHigh = 0;
        foreach (var check in _kernelCopyChecks)
        {
            _kernelCopyChecksLow = Math.Min(_kernelCopyChecksLow, check.Destination);
            _kernelCopyChecksHigh = Math.Max(_kernelCopyChecksHigh, check.End);
        }
    }

    private unsafe void CheckKernelCopy(KernelCopyCheck check)
    {
        if (check.BeforeContents != 0)
        {
            var before = new ReadOnlySpan<uint>((void*)check.BeforeContents, check.Expected.Length);
            Interlocked.Increment(ref _kernelCopyDestinationChecked);
            if (ModuloCopyCompare.FirstDifference(before, check.Expected, (ulong)check.Expected.Length) is var first and >= 0)
            {
                Interlocked.Increment(ref _kernelCopyDestinationMismatches);
                ReportKernelCopyMismatch(check, "dst", before, first);
            }
        }

        if (check.Superseded)
        {
            Interlocked.Increment(ref _kernelCopySuperseded);
            return;
        }

        var found = new uint[check.Expected.Length];
        if (!_memory.TryRead(check.Destination, System.Runtime.InteropServices.MemoryMarshal.AsBytes(found.AsSpan())))
        {
            return;
        }

        Interlocked.Increment(ref _kernelCopyChecked);
        if (ModuloCopyCompare.FirstDifference(found, check.Expected, (ulong)check.Expected.Length) is var after and >= 0)
        {
            Interlocked.Increment(ref _kernelCopyMismatches);
            ReportKernelCopyMismatch(check, "src", found, after);
        }
    }

    private void ReportKernelCopyMismatch(KernelCopyCheck check, string side, ReadOnlySpan<uint> found, int first)
    {
        if (_kernelCopyMismatchLines++ < MaxKernelCopyMismatchLines)
        {
            Console.Error.WriteLine(
                $"[METAL-V2][HOST-KERNEL-COPY] verify mismatch side={side} kind={(check.Fill ? "fill" : "copy")} dst=0x{check.Destination:X16} " +
                $"words={check.Expected.Length} src=0x{check.Source:X16} serial={check.Serial} word={first} " +
                $"expected=0x{check.Expected[first]:X8} found=0x{found[first]:X8} " +
                $"differing={ModuloCopyCompare.DifferingWords(found, check.Expected, (ulong)check.Expected.Length)}");
        }
    }

    /// <summary>The host kernel copy fields of the CHURN line; resets them.</summary>
    private string TakeHostKernelCopyCounters()
    {
        var builder = new System.Text.StringBuilder(" host_kernel_copy=[")
            .Append("mode=").Append(HostKernelCopyModeNames[(int)HostKernelCopyMode])
            .Append(",fills=").Append(Interlocked.Exchange(ref _kernelCopyFills, 0))
            .Append(",identical=").Append(Interlocked.Exchange(ref _kernelCopiesElided, 0));
        for (var index = 0; index < _kernelCopyOutcomes.Length; index++)
        {
            builder.Append(',').Append(KernelCopyOutcomeNames[index]).Append('=').Append(Interlocked.Exchange(ref _kernelCopyOutcomes[index], 0));
            if (index == (int)KernelCopyOutcome.Differs)
            {
                builder.Append(",changed_bytes=").Append(Interlocked.Exchange(ref _kernelCopyChangedBytes, 0))
                    .Append(",changed_pages=").Append(Interlocked.Exchange(ref _kernelCopyChangedPages, 0));
            }
        }

        return builder.Append(",verify_checked=").Append(Interlocked.Exchange(ref _kernelCopyChecked, 0))
            .Append(",verify_mismatch=").Append(Interlocked.Exchange(ref _kernelCopyMismatches, 0))
            .Append(",verify_dst_checked=").Append(Interlocked.Exchange(ref _kernelCopyDestinationChecked, 0))
            .Append(",verify_mismatch_dst=").Append(Interlocked.Exchange(ref _kernelCopyDestinationMismatches, 0))
            .Append(",verify_dst_skipped=").Append(Interlocked.Exchange(ref _kernelCopyDestinationSkipped, 0))
            .Append(",verify_superseded=").Append(Interlocked.Exchange(ref _kernelCopySuperseded, 0))
            .Append(",verify_dropped=").Append(Interlocked.Exchange(ref _kernelCopyChecksDropped, 0))
            .Append(']').ToString();
    }
}
