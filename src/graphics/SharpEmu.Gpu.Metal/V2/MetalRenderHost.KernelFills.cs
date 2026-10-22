// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Images;

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
    private readonly ConcurrentDictionary<string, long> _kernelFillFallbacks = new(StringComparer.Ordinal);

    /// <summary>A recognized fill or copy kernel ran as a dispatch, and why.</summary>
    public void NoteKernelFillFallback(string reason)
    {
        if (MetalV2Diagnostics.KernelFills)
        {
            _kernelFillFallbacks.AddOrUpdate(reason, 1, static (_, count) => count + 1);
        }
    }

    /// <summary>A kernel fill of repeating dwords, as an image clear.</summary>
    public bool TryClearImageFromKernelFill(ulong address, ulong size, ReadOnlySpan<uint> pattern)
    {
        if (!MetalV2Diagnostics.KernelFills)
        {
            return false;
        }

        if (pattern.IsEmpty || pattern.Length * sizeof(uint) > MetalKernelFill.MaxPatternBytes)
        {
            NoteKernelFillFallback("pattern-size");
            return false;
        }

        Span<byte> bytes = stackalloc byte[pattern.Length * sizeof(uint)];
        for (var word = 0; word < pattern.Length; word++)
        {
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(bytes[(word * sizeof(uint))..], pattern[word]);
        }

        return TryInterceptKernelFill(address, size, bytes);
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
        DiagBindCopy("copy-words", destination, source, words * 4);
        if (TryCopyWordsOnHostCore(destination, source, sourceWords, words))
        {
            return true;
        }

        // Refused: the copy runs as the dispatch it is, and a read of its destination
        // before that submission completes can be answered from its source.
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
            NoteKernelFillFallback("size");
            return false;
        }

        var size = words * sizeof(uint);
        if (sourceWords >= words)
        {
            return TryInterceptKernelCopy(destination, source, size);
        }

        if (sourceWords * sizeof(uint) > MetalKernelFill.MaxPatternBytes)
        {
            NoteKernelFillFallback("pattern-size");
            return false;
        }

        // The pattern is read now, while the kernel would have read it on the GPU: bytes a
        // recorded store or a GPU-written image still owns would cost a wait for the GPU,
        // which is worse than the round trip this saves.
        var patternSize = sourceWords * sizeof(uint);
        if (_gpuWrites.PendingSerial(source, patternSize, CompletedSerial) != 0 ||
            _imageCache.HasGpuModifiedImageBytes(source, patternSize))
        {
            NoteKernelFillFallback("pattern-pending");
            return false;
        }

        Span<byte> pattern = stackalloc byte[(int)patternSize];
        if (!TryReadGuest(source, pattern))
        {
            NoteKernelFillFallback("pattern-unreadable");
            return false;
        }

        return TryInterceptKernelFill(destination, size, pattern);
    }

    private bool TryInterceptKernelFill(ulong address, ulong size, ReadOnlySpan<byte> pattern)
    {
        _ = EnsureCommandBuffer();
        if (!_imageCache.TryClearImageFromKernelFill(address, size, pattern, out var fallback))
        {
            NoteKernelFillFallback(fallback ?? "other");
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
            NoteKernelFillFallback(fallback ?? "other");
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
    private string TakeKernelFillCounters()
    {
        var builder = new System.Text.StringBuilder()
            .Append(" fill_kernels=").Append(Interlocked.Exchange(ref _kernelFills, 0))
            .Append(" copy_kernels=").Append(Interlocked.Exchange(ref _kernelCopies, 0))
            .Append(" fill_fallback=[");
        var first = true;
        foreach (var reason in _kernelFillFallbacks.Keys.Order(StringComparer.Ordinal))
        {
            if (_kernelFillFallbacks.TryRemove(reason, out var count))
            {
                builder.Append(first ? string.Empty : ",").Append(reason).Append('=').Append(count);
                first = false;
            }
        }

        return builder.Append(']').ToString();
    }
}
