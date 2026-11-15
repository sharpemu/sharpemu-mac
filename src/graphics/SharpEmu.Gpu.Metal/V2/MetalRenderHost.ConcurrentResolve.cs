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

    /// <summary>
    /// Dispatches are prefetched like draws: the resolving thread materializes a dispatch's resources and finds its
    /// program while the worker still records the work before it, and the worker takes that program when the
    /// dispatch runs (ParallelProgramPrefetch.TryTakeCompute) or resolves it itself as before. Demon's Souls runs
    /// ~1000 dispatches a frame, each resolved on the worker (~10 us of materialization apiece). A resolution
    /// that depended on something consumed, copied, predicted or frontend-handled is refused.
    /// SHARPEMU_METAL_V2_DISPATCH_PREFETCH=0 resolves every dispatch on the worker, exactly as before.
    /// </summary>
    internal static readonly bool DispatchPrefetch =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_DISPATCH_PREFETCH") != "0";

    /// <summary>
    /// A prefetched dispatch's resolution waits only for the non-draw items before it that are not dispatches, so in
    /// a run of dispatches the next one resolves while the worker records the one before. The dispatches it was
    /// resolved ahead of record their stores when they run; the take refuses it when a GPU write logged after its
    /// resolution began covers bytes it read, retired or not (MetalGpuWriteRanges.LoggedWriteSince), or when one of
    /// them was consumed or predicted on the CPU. SHARPEMU_METAL_V2_DISPATCH_PREFETCH_CROSS=0 holds it behind every
    /// non-draw item before it, as a draw's resolution is.
    /// </summary>
    internal static readonly bool DispatchPrefetchCrossesDispatches =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_DISPATCH_PREFETCH_CROSS") != "0";

    bool IConcurrentResolveHost.PrefetchesDispatches => ParallelResolve && DispatchPrefetch;

    bool IConcurrentResolveHost.PrefetchedDispatchesCrossDispatches => DispatchPrefetchCrossesDispatches;

    bool IConcurrentResolveHost.IsStillCurrentAcrossDispatches(object assumptions) => IsConcurrentResolveStillCurrentAcrossDispatches(assumptions);

    private long _writeLogRefusals;

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
        var (computeSubmitted, computeTaken, computeRefused, computeWaitMs, computeSavedMs) = prefetch.TakeComputeCounters();
        var dispatches = $" dispatch_jobs={computeSubmitted} dispatch_taken={computeTaken} dispatch_refused=(abandoned={computeRefused[(int)ParallelProgramPrefetch.ComputeRefusal.Abandoned]}," +
            $"provider={computeRefused[(int)ParallelProgramPrefetch.ComputeRefusal.Provider]}{FormatProviderRefusals(prefetch.TakeProviderRefusals())},inputs={computeRefused[(int)ParallelProgramPrefetch.ComputeRefusal.Inputs]}," +
            $"ownership={computeRefused[(int)ParallelProgramPrefetch.ComputeRefusal.Ownership]},effects={computeRefused[(int)ParallelProgramPrefetch.ComputeRefusal.Effects]}) " +
            $"write_log_refusals={Interlocked.Exchange(ref _writeLogRefusals, 0)} dispatch_wait_ms={computeWaitMs.ToString("F1", culture)} " +
            $"dispatch_saved_ms={computeSavedMs.ToString("F1", culture)}";
        return $"resolved={resolved} abandoned={abandoned} abandoned_by=({TakeAbandonCauses()}) taken={taken} refused={refused} drains={drains} drained={drained} deferred_barriers={deferredBarriers} operand_reads_kept={operandReadsKept} " +
            $"wait_ms={waitMs.ToString("F1", culture)} resolve_ms={prefetch.TakeResolveMilliseconds().ToString("F1", culture)} " +
            $"caller_jobs={callerResolves} caller_ms={callerMs.ToString("F1", culture)} claim_steals={prefetch.TakeClaimSteals()}{dispatches}{FormatPacketCounters()} {own} {TakeOwnerTestCounters()}{TakeDrainCallers()}";
    }

    private static string FormatProviderRefusals(long[] refused) =>
        $"[frontend={refused[(int)ComputePrefetchRefusal.Frontend]},prediction={refused[(int)ComputePrefetchRefusal.Prediction]}," +
        $"unavailable={refused[(int)ComputePrefetchRefusal.Unavailable]}]";

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

    // SHARPEMU_CONCURRENT_WORD_LINES=0 reads every word of a resolution on its own, as before. A resolution reads
    // ~45 descriptor and table words a stage, mostly a few words apart, and each one cost a clean-page test and a
    // backing read of its own (a third of the resolving thread's busy time in Demon's Souls).
    private const int ConcurrentWordLineBytes = 256;
    private static readonly bool ConcurrentWordLines =
        Environment.GetEnvironmentVariable("SHARPEMU_CONCURRENT_WORD_LINES") != "0";

    // SHARPEMU_CONCURRENT_OWNED_LINES=0 tests a page found owned again for every word read in it, then the word.
    // A page found owned is remembered for the rest of the resolution (a ring beside the clean pages), and in it
    // the 256-byte line of a word is tested before the word: a clean line is recorded once and its words are read
    // from it, and only the words of an owned line are tested one by one. Demon's Souls keeps a pending GPU write
    // in its descriptor page, so each of the ~45 words a stage reads paid a page test and a word test.
    private static readonly bool ConcurrentOwnedLinesDefault =
        SharpEmu.ShaderCompiler.PerfSwitches.Enabled("SHARPEMU_CONCURRENT_OWNED_LINES");

    internal bool ConcurrentOwnedLines { get; set; } = ConcurrentOwnedLinesDefault;

    // SHARPEMU_CONCURRENT_SCOPED_SERIAL=0 asks the GPU for the completed serial (a message send) at every
    // ownership test of a resolution and of the worker's re-check of one. With it on, a resolution and a re-check
    // each read it once, at their first test that needs it. A serial only grows, so an older one reports more
    // writes pending, never fewer: an answer it gives is owned where a newer serial may say clean, never the
    // other way (the same argument as the worker's scoped serial, SHARPEMU_METAL_V2_SCOPED_SERIAL).
    private static readonly bool ConcurrentScopedSerialDefault =
        SharpEmu.ShaderCompiler.PerfSwitches.Enabled("SHARPEMU_CONCURRENT_SCOPED_SERIAL");

    internal bool ConcurrentScopedSerial { get; set; } = ConcurrentScopedSerialDefault;

    // What an ownership test costs, counted where it runs (a resolution, a re-check) and added to the host's
    // totals once per resolution or re-check, for the PERF line: owner_tests=(...).
    internal enum OwnerTestCounter
    {
        Page,
        Line,
        Word,
        Recheck,
        ImageLock,
        SerialRead,
        OverlayCall,
    }

    private static readonly string[] OwnerTestCounterNames = ["page", "line", "word", "recheck", "image_locks", "serial_reads", "overlay_calls"];

    private readonly long[] _ownerTests = new long[OwnerTestCounterNames.Length];

    /// <summary>
    /// The serial the ownership tests of one resolution or one re-check use, and what they cost. Used by one
    /// thread at a time.
    /// </summary>
    internal class ConcurrentOwnerScope
    {
        private readonly Func<ulong> _signaled;
        private ulong _serial;
        private bool _serialKnown;

        public ConcurrentOwnerScope(Func<ulong> signaled)
        {
            _signaled = signaled;
            Serial = ReadSerial;
        }

        // Read once per scope while Scoped is set; every call reads it again otherwise.
        public bool Scoped;

        // The completed serial, for IsGpuWritePendingConcurrent and the overlays' Coverage.
        public readonly Func<ulong> Serial;

        public readonly long[] Counters = new long[OwnerTestCounterNames.Length];

        public void Count(OwnerTestCounter counter) => Counters[(int)counter]++;

        // A new scope: the next test that needs the serial reads it.
        public void ForgetSerial() => _serialKnown = false;

        private ulong ReadSerial()
        {
            if (Scoped && _serialKnown)
            {
                return _serial;
            }

            Counters[(int)OwnerTestCounter.SerialRead]++;
            _serial = _signaled();
            _serialKnown = true;
            return _serial;
        }
    }

    // Adds a scope's counters to the host's totals and clears them.
    private void FlushOwnerTests(ConcurrentOwnerScope scope)
    {
        var counters = scope.Counters;
        for (var index = 0; index < counters.Length; index++)
        {
            if (counters[index] != 0)
            {
                Interlocked.Add(ref _ownerTests[index], counters[index]);
                counters[index] = 0;
            }
        }
    }

    internal string TakeOwnerTestCounters()
    {
        var builder = new System.Text.StringBuilder("owner_tests=(");
        for (var index = 0; index < _ownerTests.Length; index++)
        {
            builder.Append(index == 0 ? string.Empty : ",").Append(OwnerTestCounterNames[index]).Append('=')
                .Append(Interlocked.Exchange(ref _ownerTests[index], 0));
        }

        return builder.Append(')').ToString();
    }

    internal sealed class ConcurrentAssumptions(
        (long Overlay, long Labels, ulong GpuWrites, long Images, long GpuModified) versions, Func<ulong> signaled)
        : ConcurrentOwnerScope(signaled)
    {
        public readonly List<(ConcurrentQuery Query, ulong Address, ulong Size, bool Owned)> Answers = new();

        // What the ownership answers came from: while none of it changed, they stand.
        public (long Overlay, long Labels, ulong GpuWrites, long Images, long GpuModified) Versions = versions;

        // The GPU write log's count when the resolution began reading (MetalGpuWriteRanges.LoggedWriteSince).
        public ulong WriteLogVersion;

        // The last pages found unowned, for each kind of read.
        private readonly ulong[] _cleanPages = [ulong.MaxValue, ulong.MaxValue, ulong.MaxValue, ulong.MaxValue,
            ulong.MaxValue, ulong.MaxValue, ulong.MaxValue, ulong.MaxValue];
        private int _nextCleanPage;

        // The last pages found owned (ConcurrentOwnedLines). Not an answer: nothing is recorded for them, and a
        // read in one is answered by its line or by itself, as it would be after testing the page again.
        private readonly ulong[] _ownedPages = [ulong.MaxValue, ulong.MaxValue, ulong.MaxValue, ulong.MaxValue];
        private int _nextOwnedPage;

        // The last lines of owned pages tested, each with its answer (LineOwnedBit): a clean one was recorded.
        private const ulong LineOwnedBit = 2;
        private readonly ulong[] _lines = [ulong.MaxValue, ulong.MaxValue, ulong.MaxValue, ulong.MaxValue,
            ulong.MaxValue, ulong.MaxValue, ulong.MaxValue, ulong.MaxValue];
        private int _nextLine;

        // The last line of a clean page or a clean line read whole for word reads (TryReadGuestWordConcurrent):
        // the words after it in the line are answered from it, under the clean answer already recorded.
        public ulong WordLine = ulong.MaxValue;
        public readonly byte[] WordLineBytes = new byte[ConcurrentWordLineBytes];

        // Taken again from the pool for a new resolution.
        public void Reset((long Overlay, long Labels, ulong GpuWrites, long Images, long GpuModified) versions)
        {
            Answers.Clear();
            Versions = versions;
            Array.Fill(_cleanPages, ulong.MaxValue);
            _nextCleanPage = 0;
            Array.Fill(_ownedPages, ulong.MaxValue);
            _nextOwnedPage = 0;
            Array.Fill(_lines, ulong.MaxValue);
            _nextLine = 0;
            WordLine = ulong.MaxValue;
            ForgetSerial();
        }

        public bool IsOwnedPage(ConcurrentQuery query, ulong page) => Array.IndexOf(_ownedPages, page | (ulong)query) >= 0;

        public void NoteOwnedPage(ConcurrentQuery query, ulong page)
        {
            _ownedPages[_nextOwnedPage] = page | (ulong)query;
            _nextOwnedPage = (_nextOwnedPage + 1) & (_ownedPages.Length - 1);
        }

        // Null when the line has not been tested in this resolution (or has left the ring).
        public bool? LineOwned(ConcurrentQuery query, ulong line)
        {
            var key = line | (ulong)query;
            foreach (var entry in _lines)
            {
                if ((entry & ~LineOwnedBit) == key)
                {
                    return (entry & LineOwnedBit) != 0;
                }
            }

            return null;
        }

        public void NoteLine(ConcurrentQuery query, ulong line, bool owned)
        {
            _lines[_nextLine] = line | (ulong)query | (owned ? LineOwnedBit : 0);
            _nextLine = (_nextLine + 1) & (_lines.Length - 1);
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

        // Drops the owned answer recorded for this range before index `before` (the answers recorded
        // after it stay): a read answered another way no longer depends on it.
        public void Forget(ConcurrentQuery query, ulong address, ulong size, int before)
        {
            for (var index = Math.Min(before, Answers.Count) - 1; index >= 0; index--)
            {
                var answer = Answers[index];
                if (answer.Query == query && answer.Owned && answer.Address == address && answer.Size == size)
                {
                    Answers.RemoveAt(index);
                    return;
                }
            }
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
        for (var slot = 0; slot < _assumptionPool.Length && pooled is null; slot++)
        {
            pooled = Interlocked.Exchange(ref _assumptionPool[slot], null);
        }

        // Read before the versions and every guest byte: a write logged after this is one the resolution may have missed.
        var writeLogVersion = _gpuWrites.WriteLogVersion;
        if (pooled is null)
        {
            _concurrentAssumptions = new ConcurrentAssumptions(OwnershipVersions(), _signaledSerialReader ??= SignaledSerial)
            {
                WriteLogVersion = writeLogVersion,
                Scoped = ConcurrentScopedSerial,
            };
            return;
        }

        pooled.Reset(OwnershipVersions());
        pooled.WriteLogVersion = writeLogVersion;
        pooled.Scoped = ConcurrentScopedSerial;
        _concurrentAssumptions = pooled;
    }

    // Assumptions the worker has checked, for the resolving thread to fill again: one per draw
    // otherwise, each with its answer list grown from nothing (Grand Theft Auto V, ~10 MB/s).
    // Slots taken and filled with an exchange each: the resolving thread takes one per draw and the worker
    // returns one per draw, and a lock both took per draw was contended on every one (Demon's Souls, ~7000 a frame).
    private readonly ConcurrentAssumptions?[] _assumptionPool = new ConcurrentAssumptions?[MaxPooledAssumptions];
    private const int MaxPooledAssumptions = 32;
    private const int MaxPooledAnswers = 4096;

    internal void ReleaseConcurrentAssumptions(object assumptionsObject)
    {
        if (assumptionsObject is not ConcurrentAssumptions assumptions || assumptions.Answers.Capacity > MaxPooledAnswers)
        {
            return;
        }

        for (var slot = 0; slot < _assumptionPool.Length; slot++)
        {
            if (Interlocked.CompareExchange(ref _assumptionPool[slot], assumptions, null) is null)
            {
                return;
            }
        }
    }

    private (long Overlay, long Labels, ulong GpuWrites, long Images, long GpuModified) OwnershipVersions() =>
        (_cpOverlay.Version, _labelOverlay.Version, _gpuWrites.Version, _imageCache.ImageSetVersion, _imageCache.Epochs.GpuModified);

    internal object EndConcurrentResolve()
    {
        var assumptions = _concurrentAssumptions ?? throw new InvalidOperationException("No concurrent resolution is open.");
        _concurrentAssumptions = null;
        FlushOwnerTests(assumptions);
        return assumptions;
    }

    // Why the resolve thread abandoned a resolution, by the first thing that owned the word it
    // read, as the ownership test itself found it (nothing is tested again), so the worker's
    // redone resolutions (own, own_ms) can be laid at a cause's door. A pending GPU write is a
    // recognised copy's when a copy whose submission has not completed writes the word; that is
    // asked only under the dry run, and otherwise counts as gpu_write. Abandons thrown elsewhere
    // (a fatal condition in the resolution) are not counted here.
    private enum AbandonCause : byte
    {
        GpuWrite,
        CopyDestination,
        Label,
        CommandProcessor,
        Image,
        Code,
        Queued,
        Other,
        // Not owned; never counted.
        None,
    }

    private static readonly string[] AbandonCauseNames = ["gpu_write", "copy_dst", "label", "cp", "image", "code", "queued", "other"];

    private readonly long[] _abandonCauses = new long[AbandonCauseNames.Length];

    // The owner IsOwnedRecorded last found for this resolving thread, so the abandon that follows
    // is attributed without testing ownership again.
    [ThreadStatic]
    private static AbandonCause _lastConcurrentOwner;

    private void NoteAbandonCause(ulong address, ulong size)
    {
        var cause = _lastConcurrentOwner;
        if (cause == AbandonCause.GpuWrite && IsPendingCopyDestinationConcurrent(address, size, _signaledSerialReader ??= SignaledSerial))
        {
            cause = AbandonCause.CopyDestination;
        }

        Interlocked.Increment(ref _abandonCauses[(int)(cause == AbandonCause.None ? AbandonCause.Other : cause)]);
    }

    internal string TakeAbandonCauses()
    {
        var builder = new System.Text.StringBuilder();
        for (var index = 0; index < _abandonCauses.Length; index++)
        {
            builder.Append(index == 0 ? string.Empty : ",").Append(AbandonCauseNames[index]).Append('=')
                .Append(Interlocked.Exchange(ref _abandonCauses[index], 0));
        }

        return builder.ToString();
    }

    // On the worker: the answers a resolution relied on still hold.
    internal bool IsConcurrentResolveStillCurrent(object assumptionsObject)
    {
        var assumptions = (ConcurrentAssumptions)assumptionsObject;
        if (assumptions.Versions == OwnershipVersions())
        {
            return true;
        }

        // One serial for the whole re-check (ConcurrentScopedSerial): an older serial can only turn a clean
        // answer owned, so a re-check that passes with it would pass with every newer one.
        var scope = t_recheckScope;
        if (scope is null || !ReferenceEquals(t_recheckHost, this))
        {
            scope = t_recheckScope = new ConcurrentOwnerScope(_signaledSerialReader ??= SignaledSerial);
            t_recheckHost = this;
        }

        scope.Scoped = ConcurrentScopedSerial;
        scope.ForgetSerial();
        try
        {
            foreach (var (_, address, size, owned) in assumptions.Answers)
            {
                scope.Count(OwnerTestCounter.Recheck);
                if ((ConcurrentOwner(address, size, scope) != AbandonCause.None) != owned)
                {
                    return false;
                }
            }

            return true;
        }
        finally
        {
            FlushOwnerTests(scope);
        }
    }

    // The re-check's scope, per thread that takes resolutions.
    [ThreadStatic]
    private static ConcurrentOwnerScope? t_recheckScope;

    [ThreadStatic]
    private static MetalRenderHost? t_recheckHost;

    // On the worker, for a dispatch's resolution that may have read guest memory before the dispatches ahead of it ran:
    // as IsConcurrentResolveStillCurrent, and no GPU write logged since it began covers a range it read. A write those
    // dispatches recorded may have landed and retired by now, which the ownership answers alone no longer show.
    internal bool IsConcurrentResolveStillCurrentAcrossDispatches(object assumptionsObject)
    {
        var assumptions = (ConcurrentAssumptions)assumptionsObject;
        if (assumptions.WriteLogVersion != _gpuWrites.WriteLogVersion)
        {
            foreach (var (_, address, size, _) in assumptions.Answers)
            {
                if (_gpuWrites.LoggedWriteSince(assumptions.WriteLogVersion, address, size))
                {
                    Interlocked.Increment(ref _writeLogRefusals);
                    return false;
                }
            }
        }

        return IsConcurrentResolveStillCurrent(assumptionsObject);
    }

    // A write the command thread queued ahead of the draw being resolved, still waiting for the worker,
    // is checked first: the worker records a write's ownership before it lets go of the queued range,
    // so the range found gone means the ownership below is already there.
    private bool IsOwnedConcurrent(ConcurrentQuery query, ulong address, ulong size) =>
        ConcurrentOwner(address, size, scope: null) != AbandonCause.None;

    // The first owner of the range, in the order the tests above are cheapest to fail. A scope gives the
    // completed serial (read once while it is scoped) and counts what the tests cost; without one, each test
    // that needs the serial asks the GPU for it.
    private AbandonCause ConcurrentOwner(ulong address, ulong size, ConcurrentOwnerScope? scope)
    {
        if (PendingCommandWriteHolds(address, size))
        {
            return AbandonCause.Queued;
        }

        if (_imageCache.MayOwn(address, size))
        {
            scope?.Count(OwnerTestCounter.ImageLock);
            if (_imageCache.HasGpuModifiedImageBytes(address, size))
            {
                return AbandonCause.Image;
            }
        }

        var serial = scope?.Serial ?? (_signaledSerialReader ??= SignaledSerial);
        if (IsGpuWritePendingConcurrent(address, size, serial))
        {
            return AbandonCause.GpuWrite;
        }

        if (_cpOverlay.Count != 0)
        {
            scope?.Count(OwnerTestCounter.OverlayCall);
            if (_cpOverlay.Coverage(address, size, serial) != MetalOverlayCoverage.None)
            {
                return AbandonCause.CommandProcessor;
            }
        }

        if (_labelOverlay.Count != 0)
        {
            scope?.Count(OwnerTestCounter.OverlayCall);
            if (_labelOverlay.Coverage(address, size, serial) != MetalOverlayCoverage.None)
            {
                return AbandonCause.Label;
            }
        }

        return AbandonCause.None;
    }

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

            var ownedLines = ConcurrentOwnedLines;
            if (!ownedLines || !assumptions.IsOwnedPage(query, page))
            {
                if (CrossDrawCleanPages && IsCrossDrawClean(query, page, assumptions.Versions))
                {
                    assumptions.Record(query, page, ConcurrentPageBytes, false);
                    assumptions.NoteCleanPage(query, page);
                    return false;
                }

                assumptions.Count(OwnerTestCounter.Page);
                if (ConcurrentOwner(page, ConcurrentPageBytes, assumptions) == AbandonCause.None)
                {
                    assumptions.Record(query, page, ConcurrentPageBytes, false);
                    assumptions.NoteCleanPage(query, page);
                    if (CrossDrawCleanPages)
                    {
                        NoteCrossDrawClean(query, page, assumptions.Versions);
                    }

                    return false;
                }

                // Remembered for the rest of the resolution, and not recorded: an owned page answers nothing
                // itself. Should it turn clean meanwhile, its reads keep taking the finer tests below, which only
                // ever find less clean than the page would.
                if (ownedLines)
                {
                    assumptions.NoteOwnedPage(query, page);
                }
            }

            // In an owned page, the read's line first. Every test is an overlap test, so a clean line is clean for
            // every range inside it; its answer is recorded like a clean page's and checked again at the take.
            var line = address & ~((ulong)ConcurrentWordLineBytes - 1);
            if (ownedLines && address - line <= ConcurrentWordLineBytes - size)
            {
                var lineOwned = assumptions.LineOwned(query, line);
                if (lineOwned is null)
                {
                    assumptions.Count(OwnerTestCounter.Line);
                    lineOwned = ConcurrentOwner(line, ConcurrentWordLineBytes, assumptions) != AbandonCause.None;
                    assumptions.NoteLine(query, line, lineOwned.Value);
                    if (!lineOwned.Value)
                    {
                        assumptions.Record(query, line, ConcurrentWordLineBytes, false);
                    }
                }

                if (!lineOwned.Value)
                {
                    return false;
                }
            }
        }

        assumptions.Count(OwnerTestCounter.Word);
        _lastConcurrentOwner = ConcurrentOwner(address, size, assumptions);
        var owned = _lastConcurrentOwner != AbandonCause.None;
        assumptions.Record(query, address, size, owned);
        return owned;
    }

    // SHARPEMU_CONCURRENT_CROSS_CLEAN=0 tests every page of every resolution again. A page found clean stays
    // clean while the ownership versions stand: only a new recorded GPU write, overlay entry or image change
    // makes a clean page owned, and each of those moves a version (a write finishing only makes one clean). So
    // a resolution that starts at the versions an earlier one found a page clean at takes that answer, after
    // the one test that depends on the draw: the command writes queued ahead of it. Demon's Souls resolves
    // ~7000 draws a frame out of the same few descriptor pages, each paying the full test per page.
    private static readonly bool CrossDrawCleanPages =
        Environment.GetEnvironmentVariable("SHARPEMU_CONCURRENT_CROSS_CLEAN") != "0";

    private const int CrossCleanSlots = 512;

    // Per resolving thread: the pages found clean at t_crossCleanVersions, direct-mapped by page.
    [ThreadStatic]
    private static ulong[]? t_crossCleanPages;

    [ThreadStatic]
    private static (long Overlay, long Labels, ulong GpuWrites, long Images, long GpuModified) t_crossCleanVersions;

    [ThreadStatic]
    private static MetalRenderHost? t_crossCleanHost;

    private static int CrossCleanSlot(ulong page) => (int)((page / ConcurrentPageBytes) & (CrossCleanSlots - 1));

    private bool IsCrossDrawClean(ConcurrentQuery query, ulong page,
        (long Overlay, long Labels, ulong GpuWrites, long Images, long GpuModified) versions)
    {
        var pages = t_crossCleanPages;
        if (pages is null || !ReferenceEquals(t_crossCleanHost, this) || t_crossCleanVersions != versions ||
            pages[CrossCleanSlot(page)] != (page | (ulong)query))
        {
            return false;
        }

        return !PendingCommandWriteHolds(page, ConcurrentPageBytes);
    }

    private void NoteCrossDrawClean(ConcurrentQuery query, ulong page,
        (long Overlay, long Labels, ulong GpuWrites, long Images, long GpuModified) versions)
    {
        // Kept only for an answer taken at the versions the resolution started from: one found after a
        // version moved may not have held at them.
        if (OwnershipVersions() != versions)
        {
            return;
        }

        var pages = t_crossCleanPages ??= new ulong[CrossCleanSlots];
        if (!ReferenceEquals(t_crossCleanHost, this) || t_crossCleanVersions != versions)
        {
            Array.Fill(pages, ulong.MaxValue);
            t_crossCleanVersions = versions;
            t_crossCleanHost = this;
        }

        pages[CrossCleanSlot(page)] = page | (ulong)query;
    }

    // A recorded GPU write the GPU has not finished covers these bytes. The coarse bounds test first, then
    // the records themselves, read only while the worker leaves them alone (a few tries, else pending).
    private bool IsGpuWritePendingConcurrent(ulong address, ulong size, Func<ulong>? completedSerial = null)
    {
        if (!_gpuWrites.MayCover(address, size))
        {
            return false;
        }

        var completed = completedSerial is null ? SignaledSerial() : completedSerial();
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
        const ulong lineMask = ConcurrentWordLineBytes - 1;
        if ((address & ~lineMask) == assumptions.WordLine && (address & 3) == 0)
        {
            word = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(
                assumptions.WordLineBytes.AsSpan((int)(address & lineMask), sizeof(uint)));
            return true;
        }

        word = 0;
        Span<byte> bytes = stackalloc byte[sizeof(uint)];
        if (IsOwnedRecorded(ConcurrentQuery.Word, address, sizeof(uint), assumptions))
        {
            // A word a pending copy writes reads as its source, as the worker reads it; the owned
            // answer just recorded for it is dropped, since the copy writes it either way.
            var owner = _lastConcurrentOwner;
            if (owner == AbandonCause.GpuWrite)
            {
                var answers = assumptions.Answers.Count;
                if (TryReadThroughPendingCopyConcurrent(address, bytes, assumptions))
                {
                    assumptions.Forget(ConcurrentQuery.Word, address, sizeof(uint), answers);
                    word = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(bytes);
                    return true;
                }

                // The source tests overwrote the owner; the abandon is the word's.
                _lastConcurrentOwner = owner;
            }

            NoteAbandonCause(address, sizeof(uint));
            throw new GpuFatal.AbandonedWorkException($"The resolution reads GPU-owned memory: address=0x{address:X16}.");
        }

        // The whole page, or the whole line of an owned page, answered clean: the rest of the line is read with
        // this word. The line lies within the page, which an aligned line of a power of two below the page size
        // always does.
        var line = address & ~lineMask;
        if (ConcurrentWordLines && (address & 3) == 0 &&
            (assumptions.IsCleanPage(ConcurrentQuery.Word, address & ~(ConcurrentPageBytes - 1)) ||
             (ConcurrentOwnedLines && assumptions.LineOwned(ConcurrentQuery.Word, line) == false)) &&
            _backing.TryReadBacking(line, assumptions.WordLineBytes))
        {
            assumptions.WordLine = line;
            word = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(
                assumptions.WordLineBytes.AsSpan((int)(address & lineMask), sizeof(uint)));
            return true;
        }

        assumptions.WordLine = ulong.MaxValue;
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
            NoteAbandonCause(address, sizeof(uint));
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
            Interlocked.Increment(ref _abandonCauses[(int)AbandonCause.Code]);
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
