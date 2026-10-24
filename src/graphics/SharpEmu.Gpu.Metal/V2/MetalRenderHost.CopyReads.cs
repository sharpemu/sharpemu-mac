// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;

namespace SharpEmu.Gpu.Metal.V2;

// A title's compute memcpy (dst[i] = src[i % period], recognised by the executor and
// refused by TryCopyWordsOnHost because no image covers it) runs as a dispatch, and a
// host read of its destination before that submission completes would wait for the
// GPU. The read can instead be answered from the copy's source words, which is what
// the dispatch will store there, when
//   - the copy is the last recorded GPU write over the read range (its write sequence
//     is the range's latest), and no GPU-modified image owns the range; and
//   - the source words are final: no pending recorded write, GPU-modified image or
//     owed command-processor bytes over them (or, one level deep, they are themselves
//     the destination of an earlier pending copy whose source is final).
// Nothing is written to guest memory; only the worker's read is redirected.
//
// The copy is armed when TryCopyWordsOnHost refuses it, and recorded when the
// dispatch's storage binding notes the destination write: the record carries that
// write's sequence. The executor resets the bindings after every dispatch, which
// disarms a copy whose dispatch never bound its destination.
internal sealed partial class MetalRenderHost
{
    private const int MaxPendingCopies = 1024;
    // Wider source runs that are pending are not resolved word by word.
    private const ulong MaxChainedWords = 64;
    private const int MaxCopyReadTraces = 8;

    private struct PendingCopy
    {
        public ulong Serial;
        public ulong Sequence;
        public ulong Destination;
        public ulong Source;
        public ulong Period;
        public ulong Words;

        public readonly ulong DestinationEnd => Destination + Words * sizeof(uint);
    }

    private readonly List<PendingCopy> _pendingCopies = [];
    private PendingCopy _armedCopy;
    private bool _copyArmed;
    private long _copyReads;
    private long _copyReadPending;
    private long _copyReadOrder;
    private long _copyReadNone;
    private readonly HashSet<(ulong Destination, ulong Words)> _tracedCopies = [];
    private readonly ConcurrentQueue<string> _copyReadTraces = new();

    internal int PendingCopyCount => _pendingCopies.Count;

    internal (long Reads, long Pending, long Order, long None) CopyReadCounters =>
        (Interlocked.Read(ref _copyReads), Interlocked.Read(ref _copyReadPending),
            Interlocked.Read(ref _copyReadOrder), Interlocked.Read(ref _copyReadNone));

    private void ArmPendingCopy(ulong destination, ulong source, ulong period, ulong words)
    {
        if (!MetalV2Diagnostics.CopyReadThrough || words == 0 || period == 0 ||
            (destination & 3) != 0 || (source & 3) != 0 ||
            words > (ulong.MaxValue - destination) / sizeof(uint) ||
            period > (ulong.MaxValue - source) / sizeof(uint))
        {
            _copyArmed = false;
            return;
        }

        _armedCopy = new PendingCopy { Destination = destination, Source = source, Period = period, Words = words };
        _copyArmed = true;
    }

    private void DisarmPendingCopy() => _copyArmed = false;

    // The dispatch's storage binding notes the write into the copy's destination.
    private void NotePendingCopyDestination(ulong address, ulong size, MetalGpuWriteOrigin origin, ulong serial, ulong sequence)
    {
        if (!_copyArmed || origin != MetalGpuWriteOrigin.StorageBuffer || sequence == 0 ||
            address > _armedCopy.Destination || address + size < _armedCopy.DestinationEnd)
        {
            return;
        }

        _copyArmed = false;
        var copy = _armedCopy;
        copy.Serial = serial;
        copy.Sequence = sequence;
        if (_pendingCopies.Count >= MaxPendingCopies)
        {
            _pendingCopies.RemoveAt(0);
        }

        _pendingCopies.Add(copy);
        if (_tracedCopies.Count < MaxCopyReadTraces && _tracedCopies.Add((copy.Destination, copy.Words)))
        {
            _copyReadTraces.Enqueue(
                $"[METAL-V2][COPY-READ] dst=0x{copy.Destination:X16} src=0x{copy.Source:X16} period={copy.Period} " +
                $"words={copy.Words} serial={copy.Serial}");
        }
    }

    private void RetirePendingCopies(ulong completed)
    {
        if (_pendingCopies.Count != 0)
        {
            // In place and in order; the lambda over `completed` allocated its closure on every call.
            var kept = 0;
            for (var index = 0; index < _pendingCopies.Count; index++)
            {
                if (_pendingCopies[index].Serial > completed)
                {
                    _pendingCopies[kept++] = _pendingCopies[index];
                }
            }

            if (kept != _pendingCopies.Count)
            {
                _pendingCopies.RemoveRange(kept, _pendingCopies.Count - kept);
            }
        }
    }

    /// <summary>
    /// Answers a read the GPU is still writing from the source of the pending copy that
    /// writes it, without waiting. False leaves the read to wait as before.
    /// </summary>
    private bool TryReadThroughPendingCopy(ulong address, Span<byte> destination)
    {
        if (!MetalV2Diagnostics.CopyReadThrough)
        {
            return false;
        }

        var size = (ulong)destination.Length;
        if (size == 0 || address > ulong.MaxValue - size ||
            (_imageCache.MayOwn(address, size) && _imageCache.HasGpuModifiedImageBytes(address, size)))
        {
            return false;
        }

        if (!_gpuWrites.MayCover(address, size) || PendingSerialHinted(address, size) == 0)
        {
            return false;
        }

        var completed = CompletedSerial;

        RetirePendingCopies(completed);
        var index = FindPendingCopy(address, size, -1);
        if (index < 0)
        {
            Interlocked.Increment(ref _copyReadNone);
            return false;
        }

        var copy = _pendingCopies[index];
        if (copy.Sequence != _gpuWrites.LatestSequence(address, size))
        {
            Interlocked.Increment(ref _copyReadOrder);
            return false;
        }

        if (!TryReadCopySource(index, address, destination, completed))
        {
            Interlocked.Increment(ref _copyReadPending);
            return false;
        }

        Interlocked.Increment(ref _copyReads);
        return true;
    }

    // The newest pending copy whose destination covers the range, skipping one record.
    private int FindPendingCopy(ulong address, ulong size, int skip)
    {
        var end = address + size;
        for (var index = _pendingCopies.Count - 1; index >= 0; index--)
        {
            if (index == skip)
            {
                continue;
            }

            var copy = _pendingCopies[index];
            if (copy.Destination <= address && end <= copy.DestinationEnd)
            {
                return index;
            }
        }

        return -1;
    }

    private bool TryReadCopySource(int copyIndex, ulong address, Span<byte> destination, ulong completed)
    {
        var copy = _pendingCopies[copyIndex];
        var size = (ulong)destination.Length;
        var firstWord = (address - copy.Destination) / sizeof(uint);
        var lastWord = (address + size - 1 - copy.Destination) / sizeof(uint);
        var wordCount = lastWord - firstWord + 1;
        if (wordCount > int.MaxValue / sizeof(uint))
        {
            return false;
        }

        var byteCount = (int)(wordCount * sizeof(uint));
        Span<byte> words = byteCount <= 256 ? stackalloc byte[byteCount] : new byte[byteCount];
        var word = firstWord;
        var offset = 0;
        while (word <= lastWord)
        {
            // dst word i takes src word i % period: a contiguous run of source words lasts
            // until the read ends or the index wraps.
            var sourceIndex = word % copy.Period;
            var run = Math.Min(lastWord - word + 1, copy.Period - sourceIndex);
            var sourceAddress = copy.Source + sourceIndex * sizeof(uint);
            var runBytes = (int)(run * sizeof(uint));
            var slice = words.Slice(offset, runBytes);
            if (IsCopySourceFinal(sourceAddress, (ulong)runBytes, completed))
            {
                if (!_memory.TryRead(sourceAddress, slice))
                {
                    return false;
                }
            }
            else
            {
                if (run > MaxChainedWords)
                {
                    return false;
                }

                for (var part = 0UL; part < run; part++)
                {
                    if (!TryReadChainedSourceWord(
                            copyIndex, sourceAddress + part * sizeof(uint), slice.Slice((int)part * sizeof(uint), sizeof(uint)), completed))
                    {
                        return false;
                    }
                }
            }

            word += run;
            offset += runBytes;
        }

        var skip = (int)((address - copy.Destination) % sizeof(uint));
        words.Slice(skip, destination.Length).CopyTo(destination);
        return true;
    }

    // A source word the GPU is still writing is final when its last writer is an earlier
    // pending copy (recorded before the one reading it) whose own source is final.
    private bool TryReadChainedSourceWord(int copyIndex, ulong address, Span<byte> destination, ulong completed)
    {
        if (_imageCache.MayOwn(address, sizeof(uint)) && _imageCache.HasGpuModifiedImageBytes(address, sizeof(uint)))
        {
            return false;
        }

        var index = FindPendingCopy(address, sizeof(uint), copyIndex);
        if (index < 0)
        {
            return false;
        }

        var earlier = _pendingCopies[index];
        if (earlier.Sequence >= _pendingCopies[copyIndex].Sequence ||
            earlier.Sequence != _gpuWrites.LatestSequence(address, sizeof(uint)))
        {
            return false;
        }

        var source = earlier.Source + ((address - earlier.Destination) / sizeof(uint) % earlier.Period) * sizeof(uint);
        return IsCopySourceFinal(source, sizeof(uint), completed) && _memory.TryRead(source, destination);
    }

    private bool IsCopySourceFinal(ulong address, ulong size, ulong completed) =>
        !(_gpuWrites.MayCover(address, size) && _gpuWrites.PendingSerial(address, size, completed) != 0) &&
        !(_imageCache.MayOwn(address, size) && _imageCache.HasGpuModifiedImageBytes(address, size)) &&
        !(_cpOverlay.Count != 0 &&
          OverlayCoverageHinted(address, size) != MetalOverlayCoverage.None);

    /// <summary>The copy read-through fields of the WAITS line; resets them.</summary>
    private string TakeCopyReadCounters() =>
        $" copy_reads={Interlocked.Exchange(ref _copyReads, 0)}" +
        $" copy_read_pending={Interlocked.Exchange(ref _copyReadPending, 0)}" +
        $" copy_read_order={Interlocked.Exchange(ref _copyReadOrder, 0)}" +
        $" copy_read_none={Interlocked.Exchange(ref _copyReadNone, 0)}";

    private void WriteCopyReadTraces()
    {
        while (_copyReadTraces.TryDequeue(out var line))
        {
            Console.Error.WriteLine(line);
        }
    }
}
