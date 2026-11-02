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

        // Refused: the fill runs as a dispatch. Whether it would change anything is counted.
        if (!pattern.IsEmpty && size % sizeof(uint) == 0)
        {
            _ = EvaluateKernelFill(address, size / sizeof(uint), pattern);
        }

        return false;
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

        // Refused: the copy runs as the dispatch it is, and a read of its destination
        // before that submission completes can be answered from its source. Whether it would
        // change anything is counted.
        if (MetalV2Diagnostics.KernelFills)
        {
            _ = EvaluateKernelCopy(destination, words, source, sourceWords);
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
    //   - the source is final, as for copy read-through (IsCopySourceFinal), and no pending
    //     label bytes cover it. A fill's pattern is its own words.
    // Then nothing is written, so no shader recorded earlier can read a changed byte. The
    // destinations are small tables: a wider kernel (Demon's Souls has one of 4 MiB) is
    // refused before anything is read.
    private const ulong MaxHostKernelCopyBytes = 64 * 1024;

    private enum KernelCopyOutcome : byte
    {
        Identical,
        Differs,
        SourcePending,
        DestinationWrittenPending,
        DestinationImage,
        Size,
        Unreadable,
    }

    private static readonly string[] KernelCopyOutcomeNames =
        ["would_elide_identical", "would_write", "src_pending", "dst_written_pending", "dst_image", "size", "unreadable"];

    private readonly long[] _kernelCopyOutcomes = new long[KernelCopyOutcomeNames.Length];
    private long _kernelCopyFills;
    private long _kernelCopyChangedBytes;
    private long _kernelCopyChangedPages;
    // The words compared, read on the worker only; grown to the largest kernel seen.
    private uint[] _kernelCopyDestination = [];
    private uint[] _kernelCopySource = [];

    internal (long Identical, long Differs, long SourcePending, long DestinationWrittenPending, long DestinationImage, long Size,
        long Unreadable) HostKernelCopyOutcomes =>
        (Interlocked.Read(ref _kernelCopyOutcomes[0]), Interlocked.Read(ref _kernelCopyOutcomes[1]), Interlocked.Read(ref _kernelCopyOutcomes[2]),
            Interlocked.Read(ref _kernelCopyOutcomes[3]), Interlocked.Read(ref _kernelCopyOutcomes[4]), Interlocked.Read(ref _kernelCopyOutcomes[5]),
            Interlocked.Read(ref _kernelCopyOutcomes[6]));

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

    /// <summary>The host kernel copy fields of the CHURN line; resets them.</summary>
    private string TakeHostKernelCopyCounters()
    {
        var builder = new System.Text.StringBuilder(" host_kernel_copy=[")
            .Append("fills=").Append(Interlocked.Exchange(ref _kernelCopyFills, 0));
        for (var index = 0; index < _kernelCopyOutcomes.Length; index++)
        {
            builder.Append(',').Append(KernelCopyOutcomeNames[index]).Append('=').Append(Interlocked.Exchange(ref _kernelCopyOutcomes[index], 0));
            if (index == (int)KernelCopyOutcome.Differs)
            {
                builder.Append(",changed_bytes=").Append(Interlocked.Exchange(ref _kernelCopyChangedBytes, 0))
                    .Append(",changed_pages=").Append(Interlocked.Exchange(ref _kernelCopyChangedPages, 0));
            }
        }

        return builder.Append(']').ToString();
    }
}
