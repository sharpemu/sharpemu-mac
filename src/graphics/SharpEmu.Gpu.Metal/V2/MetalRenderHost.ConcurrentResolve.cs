// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Gpu.Scheduling;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// The guest reads of program resolution answered on a thread that resolves draws ahead of the
/// worker (ParallelProgramPrefetch through IConcurrentResolveHost), as the Vulkan presenter does.
/// That thread never synchronizes GPU-owned memory: a word the GPU owns, or bytes a pending
/// command-processor or label write still covers, abandon the resolution, which the worker then
/// does itself. Every ownership answer it relied on is recorded and checked again when the
/// worker takes the result (IsConcurrentResolveStillCurrent).
///
/// With ParallelResolve off nothing opens a concurrent resolution, and every read below takes its
/// single-thread path.
/// </summary>
internal sealed partial class MetalRenderHost : IConcurrentResolveHost
{
    /// <summary>
    /// The translation defers each draw while ParallelProgramPrefetch resolves its programs (the resource
    /// materializations, the program and permutation lookups) on a thread of its own, and the worker runs
    /// the deferred draws before anything else the stream asks of it (DrainDraws). The worker still
    /// interprets and records; it no longer resolves. Grand Theft Auto V gameplay: median 8.96 -> 11.46
    /// fps. SHARPEMU_METAL_V2_PARALLEL_RESOLVE=0 resolves on the worker again.
    /// </summary>
    internal static readonly bool ParallelResolve =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_PARALLEL_RESOLVE") != "0";

    bool IConcurrentResolveHost.ResolvesConcurrently => ParallelResolve;

    // Small unclaimed queues can resolve on the drawing thread without a worker wakeup.
    // Component gains and ordering/device checks justify the default; the Silent Hill
    // comparison was flat. Set SHARPEMU_METAL_V2_CALLER_RESOLVE=0 for the worker-only path.
    private static readonly bool CallerResolve =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_CALLER_RESOLVE") != "0";

    bool IConcurrentResolveHost.AllowsCallerResolution => CallerResolve;

    void IConcurrentResolveHost.BeginConcurrentResolve() => BeginConcurrentResolve();

    object IConcurrentResolveHost.EndConcurrentResolve() => EndConcurrentResolve();

    bool IConcurrentResolveHost.IsStillCurrent(object assumptions) => IsConcurrentResolveStillCurrent(assumptions);

    void IConcurrentResolveHost.ReleaseAssumptions(object assumptions) => ReleaseConcurrentAssumptions(assumptions);

    // Draws the translation deferred while their programs resolved run before any other host call of
    // the stream: everything but a draw may record into the same command buffer (a label, a
    // command-processor write, a transfer, a dispatch) or read what they use. Only the worker drains;
    // a completion that reaches an entry point from another thread leaves the draws to it.
    private void DrainDraws([System.Runtime.CompilerServices.CallerMemberName] string caller = "")
    {
        if (_translation is { HasPendingDraws: true, IsDraining: false } translation && _relay.IsGpuQueueThread)
        {
            _drains++;
            if (MetalV2Diagnostics.CountDrainCallers)
            {
                CollectionsMarshalAdd(caller);
            }

            var before = translation.PendingDrawCount;
            translation.DrainPendingDraws();
            _drainedDraws += before;
        }
    }

    // A deferred draw's operands (indirect arguments, a GPU-written list) are read after it.
    bool SharpEmu.Libs.Gpu.GpuCommands.ICommandStreamHost.TryReadGuestOperand(ulong address, Span<byte> destination)
    {
        // An operand GPU work may produce is read after everything queued before it.
        if (OnCommandThread)
        {
            return ReadOnWorker(address, destination, (at, bytes) => ((SharpEmu.Libs.Gpu.GpuCommands.ICommandStreamHost)this).TryReadGuestOperand(at, bytes));
        }

        // Draws that store nothing produce no operand: they stay deferred, so their programs
        // keep resolving ahead of the drain that runs them.
        if (QueuedWorkLeaves(address, (ulong)destination.Length))
        {
            _operandReadsKept++;
            return TryReadGuest(address, destination);
        }

        DrainDraws();
        return TryReadGuest(address, destination);
    }

    private long _operandReadsKept;

    private long _drains;

    // SHARPEMU_METAL_V2_DRAIN_CALLERS=1: which host entry points drain the deferred draws, in the PERF line.
    private readonly Dictionary<string, long> _drainCallers = new();

    private void CollectionsMarshalAdd(string caller)
    {
        ref var count = ref System.Runtime.InteropServices.CollectionsMarshal.GetValueRefOrAddDefault(_drainCallers, caller, out _);
        count++;
    }

    private string TakeDrainCallers()
    {
        if (!MetalV2Diagnostics.CountDrainCallers || _drainCallers.Count == 0)
        {
            return string.Empty;
        }

        var top = string.Join(",", _drainCallers.OrderByDescending(pair => pair.Value).Take(8).Select(pair => $"{pair.Key}:{pair.Value}"));
        _drainCallers.Clear();
        return $" drain_callers=({top})";
    }
    private long _drainedDraws;

    private string FormatPrefetchCounters()
    {
        var culture = System.Globalization.CultureInfo.InvariantCulture;
        var (synchronous, synchronousMs) = RenderExecutor.TakeSynchronousResolveCounters();
        var own = $"own={synchronous} own_ms={synchronousMs.ToString("F1", culture)}";
        if (_translation?.Prefetch is not { } prefetch)
        {
            return own;
        }

        var (resolved, abandoned, taken, refused, waitMs) = prefetch.TakeCounters();
        var (callerResolves, callerMs) = prefetch.TakeCallerResolveCounters();
        var drains = Interlocked.Exchange(ref _drains, 0);
        var deferredBarriers = Interlocked.Exchange(ref _deferredBarriers, 0);
        var operandReadsKept = Interlocked.Exchange(ref _operandReadsKept, 0);
        var drained = Interlocked.Exchange(ref _drainedDraws, 0);
        return $"resolved={resolved} abandoned={abandoned} taken={taken} refused={refused} drains={drains} drained={drained} deferred_barriers={deferredBarriers} operand_reads_kept={operandReadsKept} " +
            $"wait_ms={waitMs.ToString("F1", culture)} resolve_ms={prefetch.TakeResolveMilliseconds().ToString("F1", culture)} " +
            $"caller_jobs={callerResolves} caller_ms={callerMs.ToString("F1", culture)} {own}{TakeDrainCallers()}";
    }

    internal enum ConcurrentQuery : byte
    {
        // A word read the worker would synchronize: GPU-modified images, pending GPU writes, overlays.
        Word,
        // A read of what the CPU holds now, refused where the GPU may still write it, as on the worker.
        Resident,
    }

    // Ownership is checked a page at a time where it can be: a page found unowned answers every read
    // inside it for the rest of the resolution, with one recorded answer.
    private const ulong ConcurrentPageBytes = 4096;

    internal sealed class ConcurrentAssumptions((long Overlay, long Labels, ulong GpuWrites, long Images, long GpuModified) versions)
    {
        public readonly List<(ConcurrentQuery Query, ulong Address, ulong Size, bool Owned)> Answers = new();

        // What the ownership answers came from: while none of it changed, they stand.
        public (long Overlay, long Labels, ulong GpuWrites, long Images, long GpuModified) Versions = versions;

        // The last pages found unowned, for each kind of read.
        private readonly ulong[] _cleanPages = [ulong.MaxValue, ulong.MaxValue, ulong.MaxValue, ulong.MaxValue,
            ulong.MaxValue, ulong.MaxValue, ulong.MaxValue, ulong.MaxValue];
        private int _nextCleanPage;

        // Taken again from the pool for a new resolution.
        public void Reset((long Overlay, long Labels, ulong GpuWrites, long Images, long GpuModified) versions)
        {
            Answers.Clear();
            Versions = versions;
            Array.Fill(_cleanPages, ulong.MaxValue);
            _nextCleanPage = 0;
        }

        public bool IsCleanPage(ConcurrentQuery query, ulong page)
        {
            var key = page | (ulong)query;
            foreach (var clean in _cleanPages)
            {
                if (clean == key)
                {
                    return true;
                }
            }

            return false;
        }

        public void NoteCleanPage(ConcurrentQuery query, ulong page)
        {
            _cleanPages[_nextCleanPage] = page | (ulong)query;
            _nextCleanPage = (_nextCleanPage + 1) & (_cleanPages.Length - 1);
        }

        public void Record(ConcurrentQuery query, ulong address, ulong size, bool owned)
        {
            // The last few answers repeat often (a resource table read word by word).
            for (var index = Answers.Count - 1; index >= Math.Max(0, Answers.Count - 8); index--)
            {
                var previous = Answers[index];
                if (previous.Query == query && previous.Owned == owned && previous.Address == address && previous.Size == size)
                {
                    return;
                }
            }

            Answers.Add((query, address, size, owned));
        }
    }

    // Set only on the resolving thread, around one resolution.
    [ThreadStatic]
    private static ConcurrentAssumptions? _concurrentAssumptions;

    internal void BeginConcurrentResolve()
    {
        ConcurrentAssumptions? pooled = null;
        lock (_assumptionPool)
        {
            _assumptionPool.TryPop(out pooled);
        }

        if (pooled is null)
        {
            _concurrentAssumptions = new ConcurrentAssumptions(OwnershipVersions());
            return;
        }

        pooled.Reset(OwnershipVersions());
        _concurrentAssumptions = pooled;
    }

    // Assumptions the worker has checked, for the resolving thread to fill again: one per draw
    // otherwise, each with its answer list grown from nothing (Grand Theft Auto V, ~10 MB/s).
    private readonly Stack<ConcurrentAssumptions> _assumptionPool = new();
    private const int MaxPooledAssumptions = 32;
    private const int MaxPooledAnswers = 4096;

    internal void ReleaseConcurrentAssumptions(object assumptionsObject)
    {
        if (assumptionsObject is not ConcurrentAssumptions assumptions || assumptions.Answers.Capacity > MaxPooledAnswers)
        {
            return;
        }

        lock (_assumptionPool)
        {
            if (_assumptionPool.Count < MaxPooledAssumptions)
            {
                _assumptionPool.Push(assumptions);
            }
        }
    }

    private (long Overlay, long Labels, ulong GpuWrites, long Images, long GpuModified) OwnershipVersions() =>
        (_cpOverlay.Version, _labelOverlay.Version, _gpuWrites.Version, _imageCache.ImageSetVersion, _imageCache.Epochs.GpuModified);

    internal object EndConcurrentResolve()
    {
        var assumptions = _concurrentAssumptions ?? throw new InvalidOperationException("No concurrent resolution is open.");
        _concurrentAssumptions = null;
        return assumptions;
    }

    // On the worker: the answers a resolution relied on still hold.
    internal bool IsConcurrentResolveStillCurrent(object assumptionsObject)
    {
        var assumptions = (ConcurrentAssumptions)assumptionsObject;
        if (assumptions.Versions == OwnershipVersions())
        {
            return true;
        }

        foreach (var (query, address, size, owned) in assumptions.Answers)
        {
            if (IsOwnedConcurrent(query, address, size) != owned)
            {
                return false;
            }
        }

        return true;
    }

    // A write the command thread queued ahead of the draw being resolved, still waiting for the worker,
    // is checked first: the worker records a write's ownership before it lets go of the queued range,
    // so the range found gone means the ownership below is already there.
    private bool IsOwnedConcurrent(ConcurrentQuery query, ulong address, ulong size) =>
        PendingCommandWriteHolds(address, size) ||
        _imageCache.MayOwn(address, size) && _imageCache.HasGpuModifiedImageBytes(address, size) ||
        IsGpuWritePendingConcurrent(address, size) ||
        (_cpOverlay.Count != 0 &&
            _cpOverlay.Coverage(address, size, _signaledSerialReader ??= SignaledSerial) != MetalOverlayCoverage.None) ||
        (_labelOverlay.Count != 0 &&
            _labelOverlay.Coverage(address, size, _signaledSerialReader ??= SignaledSerial) != MetalOverlayCoverage.None);

    private bool PendingCommandWriteHolds(ulong address, ulong size) =>
        _commandThread is { } thread && ParallelProgramPrefetch.ResolvingJob is var job and >= 0 &&
        thread.PendingRangeHolds(job, address, size);

    // The shared event's value, read from any thread: CompletedSerial also notes it in the worker's
    // completion state.
    private ulong SignaledSerial() => MetalV2Native.SendValue64(_sharedEvent, SelSignaledValue);

    private Func<ulong>? _signaledSerialReader;

    // The answer for a range, recorded: from its page when the page is known clean or turns out clean,
    // otherwise for the range itself.
    private bool IsOwnedRecorded(ConcurrentQuery query, ulong address, ulong size, ConcurrentAssumptions assumptions)
    {
        var page = address & ~(ConcurrentPageBytes - 1);
        var withinPage = size != 0 && size <= ConcurrentPageBytes && address - page <= ConcurrentPageBytes - size;
        if (withinPage)
        {
            if (assumptions.IsCleanPage(query, page))
            {
                return false;
            }

            if (!IsOwnedConcurrent(query, page, ConcurrentPageBytes))
            {
                assumptions.Record(query, page, ConcurrentPageBytes, false);
                assumptions.NoteCleanPage(query, page);
                return false;
            }
        }

        var owned = IsOwnedConcurrent(query, address, size);
        assumptions.Record(query, address, size, owned);
        return owned;
    }

    // A recorded GPU write the GPU has not finished covers these bytes. The coarse bounds test first, then
    // the records themselves, read only while the worker leaves them alone (a few tries, else pending).
    private bool IsGpuWritePendingConcurrent(ulong address, ulong size)
    {
        if (!_gpuWrites.MayCover(address, size))
        {
            return false;
        }

        var completed = SignaledSerial();
        for (var attempt = 0; attempt < 4; attempt++)
        {
            if (_gpuWrites.TryPendingSerialSnapshot(address, size, completed, out var pending))
            {
                return pending != 0;
            }

            Thread.SpinWait(8);
        }

        return true;
    }

    private bool TryReadGuestWordConcurrent(ulong address, out uint word, ConcurrentAssumptions assumptions)
    {
        word = 0;
        if (IsOwnedRecorded(ConcurrentQuery.Word, address, sizeof(uint), assumptions))
        {
            throw new GpuFatal.AbandonedWorkException($"The resolution reads GPU-owned memory: address=0x{address:X16}.");
        }

        Span<byte> bytes = stackalloc byte[sizeof(uint)];
        if (!_backing.TryReadBacking(address, bytes))
        {
            return false;
        }

        word = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(bytes);
        return true;
    }

    // The worker's pending query retires completed records, which only the worker may do: off it, a
    // data word a recorded write may cover abandons the resolution.
    private bool IsDataWordPendingConcurrent(ulong address, ConcurrentAssumptions assumptions)
    {
        if (IsOwnedRecorded(ConcurrentQuery.Word, address, sizeof(uint), assumptions))
        {
            throw new GpuFatal.AbandonedWorkException($"The resolution reads a data word the GPU may write: address=0x{address:X16}.");
        }

        return false;
    }

    // Shader code GPU work may still be writing is waited for on the worker; off it, the resolution
    // is abandoned instead.
    private void SynchronizeShaderCodeConcurrent(ulong address, ulong size, ConcurrentAssumptions assumptions)
    {
        if (size != 0 && IsOwnedRecorded(ConcurrentQuery.Word, address, size, assumptions))
        {
            throw new GpuFatal.AbandonedWorkException($"The resolution reads shader code the GPU may write: address=0x{address:X16}.");
        }
    }

    // The clean and resident readers refuse owned bytes on the worker too, so off it they refuse
    // them the same way, and record the answer.
    private bool TryReadResidentGuestBytesConcurrent(ulong address, Span<byte> destination, ConcurrentAssumptions assumptions) =>
        !IsOwnedRecorded(ConcurrentQuery.Resident, address, (ulong)destination.Length, assumptions) &&
        _backing.TryReadBacking(address, destination);

    // TryCompareResidentGuestBytes off the worker: the same ownership test for each run of nearby ranges,
    // recorded, then the bytes compared where they lie.
    private unsafe bool TryCompareResidentGuestBytesConcurrent(
        ReadOnlySpan<ulong> addresses, ReadOnlySpan<int> lengths, ReadOnlySpan<int> offsets, ReadOnlySpan<byte> recorded,
        ConcurrentAssumptions assumptions)
    {
        var count = addresses.Length;
        if (lengths.Length != count || offsets.Length != count || _labelOverlay.Count != 0 ||
            _backing.SingleCopy is { ReleasedChunks: > 0 })
        {
            return false;
        }

        var start = 0;
        while (start < count)
        {
            var low = addresses[start];
            var high = low;
            var end = start;
            while (end < count)
            {
                var address = addresses[end];
                var length = lengths[end];
                var offset = offsets[end];
                if (length <= 0 || offset < 0 || offset > recorded.Length - length || ulong.MaxValue - address < (ulong)length)
                {
                    return false;
                }

                if (end > start && (address < low || (address > high && address - high > CompareClusterGapBytes)))
                {
                    break;
                }

                high = Math.Max(high, address + (ulong)length);
                end++;
            }

            if (IsOwnedRecorded(ConcurrentQuery.Resident, low, high - low, assumptions))
            {
                return false;
            }

            start = end;
        }

        if (!_backing.TryEnterBackingAliasAccess())
        {
            return false;
        }

        try
        {
            for (var index = 0; index < count; index++)
            {
                var length = lengths[index];
                if (!_backing.TryResolveBackingAlias(addresses[index], (ulong)length, out var alias) ||
                    !new ReadOnlySpan<byte>((void*)alias, length).SequenceEqual(recorded.Slice(offsets[index], length)))
                {
                    return false;
                }
            }
        }
        finally
        {
            _backing.ExitBackingAliasAccess();
        }

        return true;
    }
}
