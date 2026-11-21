// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;
using SharpEmu.Libs.Gpu.Scheduling;

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

    private void DisarmPendingCopy()
    {
        _copyArmed = false;
        DisarmKernelCopyCheck();
    }

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
        // Changed under the lock while the resolve thread reads it (its abandons under the dry run,
        // its copy reads under parallel resolve); the worker alone changes the list, so its own
        // reads take no lock.
        var shared = PendingCopiesShared;
        if (shared)
        {
            Monitor.Enter(_pendingCopies);
        }

        try
        {
            if (_pendingCopies.Count >= MaxPendingCopies)
            {
                _pendingCopies.RemoveAt(0);
            }

            _pendingCopies.Add(copy);
        }
        finally
        {
            if (shared)
            {
                Monitor.Exit(_pendingCopies);
            }
        }

        if (_tracedCopies.Count < MaxCopyReadTraces && _tracedCopies.Add((copy.Destination, copy.Words)))
        {
            _copyReadTraces.Enqueue(
                $"[METAL-V2][COPY-READ] dst=0x{copy.Destination:X16} src=0x{copy.Source:X16} period={copy.Period} " +
                $"words={copy.Words} serial={copy.Serial}");
        }
    }

    private void RetirePendingCopies(ulong completed)
    {
        CheckKernelCopies(completed);
        // In place and in order; the lambda over `completed` allocated its closure on every call.
        var kept = 0;
        while (kept < _pendingCopies.Count && _pendingCopies[kept].Serial > completed)
        {
            kept++;
        }

        if (kept == _pendingCopies.Count)
        {
            return;
        }

        var shared = PendingCopiesShared;
        if (shared)
        {
            Monitor.Enter(_pendingCopies);
        }

        try
        {
            for (var index = kept + 1; index < _pendingCopies.Count; index++)
            {
                if (_pendingCopies[index].Serial > completed)
                {
                    _pendingCopies[kept++] = _pendingCopies[index];
                }
            }

            _pendingCopies.RemoveRange(kept, _pendingCopies.Count - kept);
        }
        finally
        {
            if (shared)
            {
                Monitor.Exit(_pendingCopies);
            }
        }
    }

    // Off the worker: whether a recognised copy whose submission has not completed writes the
    // range. Only attributes a resolve-thread abandon (ConcurrentResolve), so a record the worker
    // has yet to retire is answered by its serial. Only under the dry run, which is what makes
    // the worker change the list under the lock.
    private bool IsPendingCopyDestinationConcurrent(ulong address, ulong size, Func<ulong> completedSerial)
    {
        if (!HostKernelCopyDryRun)
        {
            return false;
        }

        var completed = completedSerial();
        var end = address + size;
        lock (_pendingCopies)
        {
            foreach (var copy in _pendingCopies)
            {
                if (copy.Serial > completed && copy.Destination <= address && end <= copy.DestinationEnd)
                {
                    return true;
                }
            }
        }

        return false;
    }

    // The resolve thread reads the list too when it answers reads through copies (ConcurrentCopyReads).
    private bool PendingCopiesShared => HostKernelCopyDryRun || (ParallelResolve && ConcurrentCopyReads);

    /// <summary>
    /// SHARPEMU_METAL_V2_CONCURRENT_COPY_READS=0 abandons a program resolution off the worker that
    /// reads a word a pending copy writes, as before. Demon's Souls' draws read a 3 KB table its copy
    /// kernel (0x2A12B35175B11703) writes every frame: ~1,000 resolutions a frame were abandoned and
    /// done again on the worker, behind the resolve lock (~21 ms of its time a frame plus the waits).
    /// </summary>
    internal bool ConcurrentCopyReads { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_CONCURRENT_COPY_READS") != "0";

    /// <summary>
    /// Off the worker: <see cref="TryReadThroughPendingCopy"/> for one resolution. The copy must be
    /// the latest recorded write over the range and its source must be owned by nothing (no chained
    /// copies here); the source's answer is recorded so the take checks it again. The read range
    /// itself needs no answer: the copy writes these bytes whether it has landed by the take or not,
    /// and a write recorded after it would come from stream items after the draw.
    /// </summary>
    private bool TryReadThroughPendingCopyConcurrent(ulong address, Span<byte> destination, ConcurrentAssumptions assumptions)
    {
        if (!MetalV2Diagnostics.CopyReadThrough || !ConcurrentCopyReads || !PendingCopiesShared)
        {
            return false;
        }

        var size = (ulong)destination.Length;
        if (size == 0 || address > ulong.MaxValue - size ||
            (_imageCache.MayOwn(address, size) && _imageCache.HasGpuModifiedImageBytes(address, size)))
        {
            return false;
        }

        var end = address + size;
        PendingCopy copy;
        // A resolution reads a copied table word by word: the copy found for the last read, with no
        // later write anywhere over its destination, stands while the worker changes no record.
        var stamp = _gpuWrites.MutationStamp;
        if (_verifiedCopy is { } verified && _verifiedCopyHost == this && _verifiedCopyStamp == stamp && (stamp & 1) == 0 &&
            verified.Destination <= address && end <= verified.DestinationEnd)
        {
            copy = verified;
        }
        else if (!TryFindVerifiedCopyConcurrent(address, end, out copy))
        {
            return false;
        }

        return TryReadCopySourceConcurrent(copy, address, destination, assumptions);
    }

    [ThreadStatic]
    private static PendingCopy? _verifiedCopy;
    [ThreadStatic]
    private static MetalRenderHost? _verifiedCopyHost;
    [ThreadStatic]
    private static int _verifiedCopyStamp;

    private bool TryFindVerifiedCopyConcurrent(ulong address, ulong end, out PendingCopy copy)
    {
        copy = default;
        var completed = SignaledSerial();
        var found = false;
        lock (_pendingCopies)
        {
            for (var index = _pendingCopies.Count - 1; index >= 0; index--)
            {
                var candidate = _pendingCopies[index];
                if (candidate.Destination <= address && end <= candidate.DestinationEnd)
                {
                    copy = candidate;
                    found = copy.Serial > completed;
                    break;
                }
            }
        }

        if (!found)
        {
            return false;
        }

        // Verified over the whole destination first, so the following reads of the table can use it;
        // else over this range only.
        var size = copy.DestinationEnd - copy.Destination;
        for (var pass = 0; pass < 2; pass++)
        {
            var stamp = _gpuWrites.MutationStamp;
            var start = pass == 0 ? copy.Destination : address;
            var length = pass == 0 ? size : end - address;
            ulong latest = 0;
            var known = false;
            for (var attempt = 0; attempt < 4 && !known; attempt++)
            {
                known = _gpuWrites.TryLatestSequenceSnapshot(start, length, out latest);
                if (!known)
                {
                    Thread.SpinWait(8);
                }
            }

            if (!known)
            {
                return false;
            }

            if (latest == copy.Sequence)
            {
                if (pass == 0)
                {
                    _verifiedCopy = copy;
                    _verifiedCopyHost = this;
                    _verifiedCopyStamp = stamp;
                }

                return true;
            }
        }

        return false;
    }

    private bool TryReadCopySourceConcurrent(PendingCopy copy, ulong address, Span<byte> destination, ConcurrentAssumptions assumptions)
    {
        var size = (ulong)destination.Length;
        var end = address + size;

        var firstWord = (address - copy.Destination) / sizeof(uint);
        var lastWord = (end - 1 - copy.Destination) / sizeof(uint);
        var wordCount = lastWord - firstWord + 1;
        if (wordCount > 64)
        {
            return false;
        }

        Span<byte> words = stackalloc byte[(int)(wordCount * sizeof(uint))];
        var word = firstWord;
        var offset = 0;
        while (word <= lastWord)
        {
            var sourceIndex = word % copy.Period;
            var run = Math.Min(lastWord - word + 1, copy.Period - sourceIndex);
            var sourceAddress = copy.Source + sourceIndex * sizeof(uint);
            var runBytes = (int)(run * sizeof(uint));
            if (IsOwnedRecorded(ConcurrentQuery.Word, sourceAddress, (ulong)runBytes, assumptions) ||
                !_backing.TryReadBacking(sourceAddress, words.Slice(offset, runBytes)))
            {
                return false;
            }

            word += run;
            offset += runBytes;
        }

        var skip = (int)((address - copy.Destination) % sizeof(uint));
        words.Slice(skip, destination.Length).CopyTo(destination);
        Interlocked.Increment(ref _concurrentCopyReads);
        // Its owned answer is dropped by the caller: a write over it logged after the resolution began (a dispatch it was
        // resolved ahead of) would be the destination's latest instead of the copy (IsConcurrentResolveStillCurrentAcrossDispatches).
        assumptions.NoteCopiedThrough(address, (ulong)destination.Length);
        return true;
    }

    private long _concurrentCopyReads;

    /// <summary>
    /// Right after a word was read through pending copy <c>_verifiedCopy</c> with its owner from the mask of its line
    /// (ConcurrentLineMasks), the other words of the line that copy writes are read through it too, into the
    /// resolution's copy line, and later reads of them take the bytes from there (TryReadGuestWordConcurrent). Each is
    /// what TryReadGuestWordConcurrentTested returns for that word at this moment, with the same answers left recorded:
    ///   - its owner: the mask's bit (GpuWrite), while the stamp and the versions the mask stands for stand, which is
    ///     checked here, at the same moment as the read;
    ///   - TryReadThroughPendingCopyConcurrent: no GPU-modified image over the line (its summary, at these versions),
    ///     and the verified copy, which covers the word and was verified at the current record stamp, so its fast path
    ///     takes this same copy;
    ///   - TryReadCopySourceConcurrent: a source word in a page this resolution found clean answers unowned with
    ///     nothing recorded (a source word anywhere else is left out, for its own read to test and record), then the
    ///     source bytes are read;
    ///   - the owned answer the word's test records is dropped again by the read-through: nothing is left for it.
    /// So the copy line holds those reads done now, as the clean word line holds a clean line read whole: the later
    /// read of a word takes the answer it would have got here, and the worker's re-check at the take sees the same
    /// recorded answers (the source pages' clean answers, recorded when they were tested) as before.
    /// </summary>
    private void FillCopyLine(ConcurrentAssumptions assumptions, int slot, ulong line)
    {
        assumptions.CopyLine = ulong.MaxValue;
        ref readonly var summary = ref assumptions.Summary(slot);
        var stamp = _gpuWrites.MutationStamp;
        if (!summary.Usable || stamp != summary.Stamp || OwnershipVersions() != summary.Versions ||
            _verifiedCopy is not { } copy || _verifiedCopyHost != this || _verifiedCopyStamp != stamp)
        {
            return;
        }

        var lineEnd = line + ConcurrentWordLineBytes;
        var start = Math.Max(line, copy.Destination);
        var end = Math.Min(lineEnd, copy.DestinationEnd);
        ulong words = 0;
        var at = start;
        while (at < end)
        {
            // A run of words whose source words follow each other, up to the end of the copy's period.
            var index = (at - copy.Destination) / sizeof(uint) % copy.Period;
            var runWords = Math.Min((end - at) / sizeof(uint), copy.Period - index);
            var runBytes = runWords * sizeof(uint);
            var source = copy.Source + index * sizeof(uint);
            if (assumptions.IsCleanPage(ConcurrentQuery.Word, source & ~(ConcurrentPageBytes - 1)) &&
                assumptions.IsCleanPage(ConcurrentQuery.Word, (source + runBytes - 1) & ~(ConcurrentPageBytes - 1)) &&
                _backing.TryReadBacking(source, assumptions.CopyLineBytes.AsSpan((int)(at - line), (int)runBytes)))
            {
                var first = (int)((at - line) / sizeof(uint));
                words |= (runWords >= 64 ? ulong.MaxValue : (1UL << (int)runWords) - 1) << first;
            }

            at += runBytes;
        }

        // Only words the mask gives to a GPU write: another word of the line answers None and reads guest memory.
        words &= summary.Mask;
        if (words != 0)
        {
            assumptions.NoteCopiedThrough(line, ConcurrentWordLineBytes);
            assumptions.CopyLine = line;
            assumptions.CopyLineWords = words;
            assumptions.CopyLineStamp = stamp;
            assumptions.CopyLineVersions = summary.Versions;
        }
    }

    // SHARPEMU_CONCURRENT_LINE_MASKS=verify: a copy-line word read again with its own test, which must give the same
    // bytes while the stamp and versions the copy line was filled at stand.
    private bool VerifyCopyLineWord(ulong address, uint copied, out uint word, ConcurrentAssumptions assumptions)
    {
        var line = assumptions.CopyLine;
        var stamp = assumptions.CopyLineStamp;
        var versions = assumptions.CopyLineVersions;
        var read = TryReadGuestWordConcurrentTested(address, out word, assumptions);
        if (_gpuWrites.MutationStamp != stamp || OwnershipVersions() != versions || _verifiedCopyStamp != stamp)
        {
            return read;
        }

        if (read && word == copied)
        {
            assumptions.Count(OwnerTestCounter.MaskVerified);
            return true;
        }

        Interlocked.Increment(ref _lineMaskVerifyFailed);
        var message = $"[METAL-V2][LINE_MASK] copy line verify mismatch address=0x{address:X16} line=0x{line:X16} " +
            $"copied=0x{copied:X8} read={read} word=0x{word:X8}.";
        Console.Error.WriteLine(message);
        throw GpuFatal.Fatal(message);
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
        $" copy_read_none={Interlocked.Exchange(ref _copyReadNone, 0)}" +
        $" concurrent_copy_reads={Interlocked.Exchange(ref _concurrentCopyReads, 0)}" +
        $" concurrent_overlay_reads={Interlocked.Exchange(ref _concurrentOverlayReads, 0)}" +
        $" clean_overlay_reads={Interlocked.Exchange(ref _cleanOverlayReads, 0)}";

    /// <summary>
    /// SHARPEMU_METAL_V2_CONCURRENT_OVERLAY_READS=0 abandons a resolution off the worker that reads a word a pending
    /// command-processor write owes, as before, and lets the worker's clean read of such a word wait for the GPU.
    /// A posted word copy (TryPostWordCopy) is such a write: Demon's Souls posts ~120 a frame, among them the 3 KB table
    /// its draws read (see ConcurrentCopyReads), and every draw resolved ahead that read it was abandoned and resolved
    /// again on the worker (~500 a frame), whose own clean reads of it waited for the GPU (~11 waits, ~12 ms a frame).
    /// </summary>
    internal bool ConcurrentOverlayReads { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_CONCURRENT_OVERLAY_READS") != "0";

    private long _concurrentOverlayReads;
    private long _cleanOverlayReads;

    /// <summary>
    /// Off the worker: a word a recorded GPU write owns, read as the bytes a command-processor write recorded ahead of
    /// the draw still owes there, the way the worker reads it (TryReadGuest answers from the overlay before any wait).
    /// The overlay speaks only for bytes no write recorded after it covers (each such write forgets them before it is
    /// logged), so bytes it covers whole are the ones the draw sees, landed by the take or not. No image the GPU drew
    /// into may own them, and no pending label (the worker reads labels first). Like a copy read, the word's owned
    /// answer is dropped by the caller and the range kept for the take's write-log test: a write logged after the
    /// resolution began over these bytes refuses it.
    /// </summary>
    private bool TryReadPendingCommandProcessorBytesConcurrent(ulong address, Span<byte> destination, ConcurrentAssumptions assumptions)
    {
        if (!ConcurrentOverlayReads || _cpOverlay.Count == 0)
        {
            return false;
        }

        var size = (ulong)destination.Length;
        var serial = assumptions.Serial;
        if (size == 0 || size > ConcurrentWordLineBytes || address > ulong.MaxValue - size ||
            (_imageCache.MayOwn(address, size) && _imageCache.HasGpuModifiedImageBytes(address, size)) ||
            (_labelOverlay.Count != 0 && _labelOverlay.Coverage(address, size, serial) != MetalOverlayCoverage.None))
        {
            return false;
        }

        // Up to 64 bytes byte by byte; a longer range (a compared range of a proof, up to a line) in whole words.
        var covered = size <= 64
            ? _cpOverlay.TryReadCovered(address, destination, serial)
            : ((address | size) & 3) == 0 &&
              _cpOverlay.ReadCoveredWords(address, destination, serial) == (size == 256 ? ulong.MaxValue : (1UL << (int)(size / 4)) - 1);
        if (!covered)
        {
            return false;
        }

        Interlocked.Increment(ref _concurrentOverlayReads);
        assumptions.NoteCopiedThrough(address, size);
        return true;
    }

    /// <summary>
    /// Right after a word was read from a pending command-processor write (TryPendingCommandProcessorBytesConcurrent), the
    /// other words of its line that the same reading would answer are read into the resolution's copy line, and later
    /// reads of them take the bytes from there (TryReadGuestWordConcurrent), as FillCopyLine does for a pending copy.
    /// A resolution reads a posted table word by word (Demon's Souls: ~100 words a draw, ~500 draws a frame), and each
    /// word's own tests and overlay read took the overlay's lock. Each word kept is what that reading returns now:
    ///   - owned by a recorded GPU write (the line's pending word mask), and by nothing tested before it: no write the
    ///     command thread queued for this job, no image the GPU drew into over the line;
    ///   - covered whole by the overlay, with no pending label over the line;
    /// all read while the record stamp and the ownership versions stand (checked before and after). The line is kept
    /// for the take's write-log test like a copied line: a write logged after the resolution began refuses it.
    /// </summary>
    private void FillOverlayLine(ConcurrentAssumptions assumptions, ulong line)
    {
        assumptions.CopyLine = ulong.MaxValue;
        const ulong size = ConcurrentWordLineBytes;
        var versions = OwnershipVersions();
        var stamp = _gpuWrites.MutationStamp;
        if ((stamp & 1) != 0 || PendingCommandWriteHolds(line, size) ||
            (_imageCache.MayOwn(line, size) && _imageCache.HasGpuModifiedImageBytes(line, size)) ||
            (_labelOverlay.Count != 0 && _labelOverlay.Coverage(line, size, assumptions.Serial) != MetalOverlayCoverage.None) ||
            !_gpuWrites.TryPendingWordMask(line, assumptions.Serial(), out var pending, out var maskStamp) || maskStamp != stamp)
        {
            return;
        }

        var words = _cpOverlay.ReadCoveredWords(line, assumptions.CopyLineBytes, assumptions.Serial) & pending;
        if (words == 0 || _gpuWrites.MutationStamp != stamp || OwnershipVersions() != versions)
        {
            return;
        }

        assumptions.NoteCopiedThrough(line, size);
        assumptions.CopyLine = line;
        assumptions.CopyLineWords = words;
        assumptions.CopyLineStamp = stamp;
        assumptions.CopyLineVersions = versions;
    }

    /// <summary>
    /// On the worker: a clean read of a word a recorded GPU write owns, answered from a pending command-processor write
    /// that covers it whole instead of waiting for the GPU (TryReadGuest's order). False: the caller waits as before.
    /// </summary>
    private bool TryReadPendingCommandProcessorBytes(ulong address, Span<byte> destination)
    {
        if (!ConcurrentOverlayReads || _cpOverlay.Count == 0 ||
            (_imageCache.MayOwn(address, (ulong)destination.Length) && _imageCache.HasGpuModifiedImageBytes(address, (ulong)destination.Length)) ||
            !_cpOverlay.TryReadCovered(address, destination, _completedSerialReader ??= () => CompletedSerial))
        {
            return false;
        }

        Interlocked.Increment(ref _cleanOverlayReads);
        Interlocked.Increment(ref _overlayReads);
        NotePendingAnswer(address, destination.Length);
        return true;
    }

    private void WriteCopyReadTraces()
    {
        while (_copyReadTraces.TryDequeue(out var line))
        {
            Console.Error.WriteLine(line);
        }
    }
}
