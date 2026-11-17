// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Scheduling;
using WorkerScratch = SharpEmu.ShaderCompiler.WorkerScratch;
using SharpEmu.Libs.VideoOut;

namespace SharpEmu.Libs.Gpu.Rendering;

// Resolves the programs of deferred draws on resolving threads (one per resolver shard, SHARPEMU_RESOLVE_SHARDS), each
// in issue order of its own jobs, while the render thread interprets and records. The render thread takes a result for
// the draw it runs only when the resolution completed, used the inputs the draw derives and still matches the host's
// GPU-ownership state; otherwise it resolves the draw itself.
public sealed class ParallelProgramPrefetch : IProgramPrefetch, IDisposable
{
    public sealed class Job(RegisterBanks banks, long gate, bool indexed = false, int startInstanceUserData = -1)
    {
        internal readonly RegisterBanks Banks = banks;
        internal readonly bool Indexed = indexed;
        internal readonly int StartInstanceUserData = startInstanceUserData;
        // The count of stream items other than draws that must have run before the job resolves.
        internal readonly long Gate = gate;
        internal GraphicsProgramInputs Inputs;
        internal GraphicsPrograms? Programs;
        internal object? Assumptions;
        // A dispatch's job (SubmitCompute): its arguments, and what its resolution came to.
        internal ComputeArguments? Compute;
        internal ComputeProgram? ComputeResult;
        internal ComputePrefetchRefusal ComputeRefusal;
        internal long ResolveTicks;
        // The jobs submitted before this one.
        internal long Index;
        // The resolver shard whose queue holds it (ResolveShards): its resolution runs under that shard's state.
        internal int Shard;
        // 0 queued, 1 resolved, 2 abandoned.
        internal volatile int State;
        // The resolving thread (or the caller) has taken it from the queue; it found its gate still closed when it got to it.
        internal volatile bool Started;
        internal volatile bool GateWaited;
        // A draw's job: the gate it would have had without looking ahead (Gate may stop short of the dispatches just before
        // it, SubmitDrawAhead), and the gated items that had run when its resolution began. A resolution that began before
        // all FullGate items ran looked ahead of the ones that had not, and its take checks it against them.
        internal long FullGate;
        internal long ExecutedAtStart;
        // =verify, =census: when the resolution ended (Stopwatch ticks).
        internal long ResolvedAt;
        // =census with SHARPEMU_PREFETCH_CENSUS_PS: the pixel stage's materialization reads, in order, as the
        // resolving thread read them right after its resolution.
        internal List<(ulong Address, uint Word, bool Clean, bool Refused)>? CensusReads;
        // What the host said of the refused clean reads among them, at the resolution.
        internal string? CensusRefusals;
        // The gate number of the last collector call (GateItemKind.Collect) queued before the job: above ExecutedAtStart,
        // the resolution looked ahead of it (draw_lookahead over_gc).
        internal long LastCollectGate;
        // =verify: the program cache state the resolution ended under (IShaderPipelineProvider.DescribeProgramState).
        internal string? ResolveState;
        // The program cache stamp the resolution began under (IShaderPipelineProvider.ProgramState): a take under
        // another one is refused, since a lookup made now may find a replaced entry (replanned).
        internal ProgramStateStamp ProgramState;
    }

    /// <summary>
    /// A draw's job may resolve before the dispatches just before the draw ran, as a dispatch's may
    /// (SubmitCompute): its gate stops short of them, its program resolves while they run, and its take accepts the
    /// resolution only when none of those dispatches did more than encode GPU work (NoteDispatchEffect) and the
    /// host's ownership answers and GPU write log show nothing it read changed (IsStillCurrentAcrossDispatches);
    /// otherwise the draw resolves its programs itself (draw_lookahead refused by reason). The host's collector call
    /// after each completed submission (GateItemKind.Collect) is looked ahead of the same way (LooksAheadOfCollects):
    /// it notes an effect when it did anything a resolution may have read (NoteCollect), and a resolution that began
    /// before such a call ran is refused (gc_effects). Every other gated item (an interrupt, a commit, an event with
    /// no label, another host call) still holds the draws after it. SHARPEMU_PREFETCH_DRAW_LOOKAHEAD=0 holds draws
    /// behind dispatches and collector calls again (and dispatches behind collector calls), and
    /// SHARPEMU_PREFETCH_GC_LOOKAHEAD=0 behind collector calls only; =verify also resolves
    /// every accepted look-ahead on the worker and fails on any difference in the programs or their bindings.
    /// </summary>
    public enum DrawLookaheadMode : byte
    {
        Off,
        On,
        Verify,
        // =census: a measurement, not a check. The worker resolves every taken draw job again, looked ahead or not, uses
        // its own result, and counts the jobs whose earlier resolution differs, with the time each stood between its
        // resolution and its take: whether look-aheads differ more often than jobs that waited for their gate.
        Census,
    }

    public static readonly DrawLookaheadMode DrawLookahead =
        ParseDrawLookahead(Environment.GetEnvironmentVariable("SHARPEMU_PREFETCH_DRAW_LOOKAHEAD"));

    internal static DrawLookaheadMode ParseDrawLookahead(string? value) => value switch
    {
        "0" => DrawLookaheadMode.Off,
        "verify" => DrawLookaheadMode.Verify,
        "census" => DrawLookaheadMode.Census,
        _ => Environment.GetEnvironmentVariable("SHARPEMU_PERF_VERIFY") == "1" ? DrawLookaheadMode.Verify : DrawLookaheadMode.On,
    };

    // The mode this prefetch checks takes in (tests set Verify on one instance).
    internal DrawLookaheadMode LookaheadMode { get; set; } = DrawLookahead;

    private bool RecordsForVerify => LookaheadMode is DrawLookaheadMode.Verify or DrawLookaheadMode.Census;

    // The command thread queues a collector call (GateItemKind.Collect) without moving the dispatch gate: the draws and
    // dispatches after it gate on the items before it, and their takes check NoteCollect. Only where draws look ahead
    // (the mode, and a host whose cross-dispatch check covers it), so SHARPEMU_PREFETCH_DRAW_LOOKAHEAD=0 gates again;
    // SHARPEMU_PREFETCH_GC_LOOKAHEAD=0 gates behind collector calls alone and keeps the look-ahead over dispatches.
    private static readonly bool CollectLookaheadDefault =
        Environment.GetEnvironmentVariable("SHARPEMU_PREFETCH_GC_LOOKAHEAD") != "0";

    internal bool CollectLookahead { get; set; } = CollectLookaheadDefault;

    public bool LooksAheadOfCollects =>
        CollectLookahead && LookaheadMode != DrawLookaheadMode.Off && _concurrent.PrefetchedDrawsCrossDispatches;

    // Why a draw's take refused a resolution that looked ahead of dispatches or collector calls; Count is accepted.
    public enum DrawLookaheadRefusal : byte
    {
        // The resolution threw (an owned read), so the draw resolves it as any abandoned job.
        Abandoned,
        // Different inputs.
        Inputs,
        // A dispatch it was resolved ahead of did more than encode GPU work.
        Effects,
        // An ownership answer changed, or a GPU write logged since it began covers what it read.
        Ownership,
        // A collector call it was resolved ahead of did something a resolution may have read (NoteCollect).
        CollectEffects,
        // The program cache replaced an entry, learned a device load or changed the vertex tables since the
        // resolution began (Job.ProgramState); counted for every job, looked ahead or not (replanned=).
        Replanned,
        Count,
    }

    private static readonly string[] DrawLookaheadRefusalNames = ["abandoned", "inputs", "effects", "ownership", "gc_effects", "replanned"];
    private long _lookaheadJobs, _lookaheadTaken, _lookaheadVerified, _lookaheadVerifyRaced;
    // Draw jobs of any kind refused because the program cache moved since they began (DrawLookaheadRefusal.Replanned).
    private long _replannedRefusals;
    private readonly long[] _lookaheadRefusals = new long[(int)DrawLookaheadRefusal.Count];

    // The subset of the look-aheads above that began before a collector call ran (over_gc), and the collector calls
    // the worker ran and those of them that noted an effect (gc_calls).
    private long _collectLookaheadJobs, _collectLookaheadTaken, _collectLookaheadRefused, _collectCalls, _collectEffectCalls;

    public (long Jobs, long Taken, long Refused, long Calls, long Effects) TakeCollectLookaheadCounters() =>
        (Interlocked.Exchange(ref _collectLookaheadJobs, 0), Interlocked.Exchange(ref _collectLookaheadTaken, 0),
            Interlocked.Exchange(ref _collectLookaheadRefused, 0), Interlocked.Exchange(ref _collectCalls, 0),
            Interlocked.Exchange(ref _collectEffectCalls, 0));

    // The collector call the worker runs now (GateItemKind.Collect) did something a resolution may have read: images
    // released, deferred draws run, single-copy chunks armed, emptied or unwatched. A job resolved ahead of
    // it is refused, as behind a dispatch's effect. -1 while none did.
    private long _lastCollectEffect = -1;

    public void NoteCollect(bool effect)
    {
        Interlocked.Increment(ref _collectCalls);
        if (effect)
        {
            Interlocked.Increment(ref _collectEffectCalls);
            Volatile.Write(ref _lastCollectEffect, Volatile.Read(ref _executedGate));
        }
    }

    // The last collector call's gate number (command thread only, NoteGateItem), for Job.LastCollectGate.
    private long _lastCollectGate;

    public (long Jobs, long Taken, long[] Refused, long Verified) TakeDrawLookaheadCounters()
    {
        var refused = new long[(int)DrawLookaheadRefusal.Count];
        for (var index = 0; index < refused.Length; index++)
        {
            refused[index] = Interlocked.Exchange(ref _lookaheadRefusals[index], 0);
        }

        return (Interlocked.Exchange(ref _lookaheadJobs, 0), Interlocked.Exchange(ref _lookaheadTaken, 0), refused,
            Interlocked.Exchange(ref _lookaheadVerified, 0));
    }

    private string TakeDrawLookaheadReport()
    {
        var (jobs, taken, refused, verified) = TakeDrawLookaheadCounters();
        var collect = TakeCollectLookaheadCounters();
        var builder = new System.Text.StringBuilder("draw_lookahead=(mode=").Append(LookaheadMode.ToString().ToLowerInvariant())
            .Append(",over_gc=").Append(LooksAheadOfCollects ? "on" : "off")
            .Append(",jobs=").Append(jobs).Append(",taken=").Append(taken).Append(",refused=(");
        for (var index = 0; index < refused.Length; index++)
        {
            builder.Append(index == 0 ? string.Empty : ",").Append(DrawLookaheadRefusalNames[index]).Append('=').Append(refused[index]);
        }

        return builder.Append("),verified=").Append(verified).Append(",verify_raced=").Append(Interlocked.Exchange(ref _lookaheadVerifyRaced, 0))
            .Append(",gc=(jobs=").Append(collect.Jobs).Append(",taken=").Append(collect.Taken).Append(",refused=").Append(collect.Refused)
            .Append(",calls=").Append(collect.Calls).Append(",effects=").Append(collect.Effects)
            .Append("),replanned_all=").Append(Interlocked.Exchange(ref _replannedRefusals, 0))
            .Append(LookaheadMode == DrawLookaheadMode.Census ? TakeCensusReport() : string.Empty).Append(')').ToString();
    }

    // census=(gated=resolved/differed/stood_ms,ahead=resolved/differed/stood_ms)
    private string TakeCensusReport()
    {
        string Group(int group) =>
            $"{Interlocked.Exchange(ref _censusResolved[group], 0)}/{Interlocked.Exchange(ref _censusDiffered[group], 0)}/" +
            $"{Interlocked.Exchange(ref _censusStoodTicks[group], 0) * 1000 / System.Diagnostics.Stopwatch.Frequency}";
        return $",census=(gated={Group(0)},ahead={Group(1)})";
    }

    // The kind of a gated non-draw item, as the command thread queues it (NoteGateItem): what the draws after it wait for.
    public enum GateItemKind : byte
    {
        None,
        Dispatch,
        // A command-processor write of known bytes (WRITE_DATA, a DMA of known words) or one of the stream's own stores.
        CommandProcessorWrite,
        // An end-of-pipe event: a label, or with no destination only an interrupt or a signal.
        Label,
        // A fill or copy of a guest range.
        Range,
        // A call the stream waits on (a read the worker answers, a host call with a result).
        Wait,
        // An interrupt the worker raises.
        Event,
        // A commit the idle stream asks for.
        Flush,
        // Another posted host call.
        Call,
        // The host's collector call after a completed submission (ICommandStreamHost.RunGarbageCollector); with
        // LooksAheadOfCollects the draws and dispatches after it do not gate on it.
        Collect,
        // Other deferred work.
        Other,
        Count,
    }

    private static readonly string[] GateItemKindNames = ["none", "dispatch", "cp", "label", "range", "wait", "event", "flush", "call", "gc", "other"];

    private readonly RenderExecutor _executor;
    private readonly IRenderHost _host;
    // host.Fatal as one delegate: a method group over an interface makes a new one per use, per draw.
    private readonly Func<string, Exception> _fatal;
    private readonly IConcurrentResolveHost _concurrent;

    // What a dispatch's program resolution reads besides its banks.
    internal sealed record ComputeArguments(uint DispatchInitiator, uint GroupsX, uint GroupsY, uint GroupsZ, bool Indirect)
    {
        // Equality with the dispatch's own arguments, without a record made for each take to compare against.
        public bool Matches(uint dispatchInitiator, uint groupsX, uint groupsY, uint groupsZ, bool indirect) =>
            DispatchInitiator == dispatchInitiator && GroupsX == groupsX && GroupsY == groupsY && GroupsZ == groupsZ && Indirect == indirect;
    }
    private readonly object _lock = new();

    // One resolving thread and its FIFO queue per resolver shard (ResolveShards, IShaderPipelineProvider.ResolveShardCount):
    // a job goes to the queue of the shard its provider names (ResolveShardOf), and each shard resolves its own jobs in
    // submission order, one at a time, its thread or the caller. Jobs of different shards finish out of order; _finished
    // is the finished prefix over all of them (FinishedPrefix), so whatever waits for the jobs before an item waits for
    // exactly those. Everything here is under _lock unless read with Volatile as a hint. With one shard, the single
    // resolver as before.
    private sealed class ResolverShard(int index)
    {
        public readonly int Index = index;
        public readonly Queue<Job> Queue = new();
        public SharpEmu.HLE.Host.HostHelperThread? Thread;
        // A resolution of this shard's head runs (its thread's or the caller's), and the job it resolves.
        public bool Running;
        public Job? Resolving;
        // The shard's thread owns the head even while it releases the lock to wait for its gate.
        public bool WorkerClaimed;
        // The shard's thread waits for jobs or an active caller.
        public bool IdleForJobs;
        // Set while the shard's thread sleeps because of a pause.
        public bool WaitingOnPause;
        // The gate the shard's thread sleeps on; long.MaxValue while it does not.
        public long SleepingOnGate = long.MaxValue;
        // Its thread only: jobs taken (allocation sampling, resolver_jobs).
        public long ResolverJobs;
        // Since the last report: jobs its thread or the caller resolved, and its thread's resolving time (resolve_ms).
        public long Resolved;
        public long ResolveTicks;
        public ulong ThreadId;
        public long LastCpu = -1;
    }

    private readonly ResolverShard[] _shards;
    private int _outstanding;
    // The finished prefix: every job with a lower Index is finished (FinishedPrefix).
    private long _finished;
    private long _executedGate;
    private bool _paused;
    private readonly bool _allowCallerResolution;
    // Long bursts give the resolver useful lead; racing it for each head adds handoffs.
    // Short queues are the query/barrier pattern where the renderer otherwise waits
    // for a newly woken thread to resolve only a handful of jobs.
    private const int MaximumCallerOutstanding = 8;
    private bool _stopping;

    public ParallelProgramPrefetch(RenderExecutor executor, IRenderHost host, IConcurrentResolveHost concurrent)
        : this(executor, host, concurrent, concurrent.AllowsCallerResolution)
    {
    }

    internal ParallelProgramPrefetch(RenderExecutor executor, IRenderHost host, IConcurrentResolveHost concurrent,
        bool allowCallerResolution)
    {
        _executor = executor;
        _host = host;
        _fatal = host.Fatal;
        _concurrent = concurrent;
        _allowCallerResolution = allowCallerResolution;
        var count = Math.Clamp(executor.ResolveShardCount, 1, ResolveShards.Maximum);
        _shards = new ResolverShard[count];
        for (var index = 0; index < count; index++)
        {
            _shards[index] = new ResolverShard(index);
        }

        ResolveShards.LogOnce(count);
        // The worker waits on these threads per draw: they run at the worker's class, with the
        // worker's stack, since they resolve what the worker would resolve itself.
        foreach (var shard in _shards)
        {
            shard.Thread = SharpEmu.HLE.Host.HostHelperThread.Start(count == 1 ? "SharpEmu program resolve" : $"SharpEmu program resolve {shard.Index}",
                16 * 1024 * 1024, () => Run(shard));
        }
    }

    internal int ShardCount => _shards.Length;

    // For tests: whether a shard's thread holds its head's claim (under _lock).
    internal bool IsWorkerClaimed(int shard)
    {
        lock (_lock)
        {
            return _shards[shard].WorkerClaimed;
        }
    }

    private ResolverShard ShardFor(Job job)
    {
        if (_shards.Length == 1)
        {
            return _shards[0];
        }

        var index = _executor.ResolveShardOf(job.Banks, job.Compute is not null);
        if ((uint)index >= (uint)_shards.Length)
        {
            throw _fatal($"The pipeline provider routed a job to resolver shard {index} of {_shards.Length}.");
        }

        return _shards[index];
    }

    // Under _lock: the lowest Index not finished (a shard's running job, else its queue's head), or Submitted.
    private long FinishedPrefix()
    {
        var prefix = Submitted;
        foreach (var shard in _shards)
        {
            if (shard.Resolving is { } running)
            {
                prefix = Math.Min(prefix, running.Index);
            }
            else if (shard.Queue.TryPeek(out var head))
            {
                prefix = Math.Min(prefix, head.Index);
            }
        }

        return prefix;
    }

    // Under _lock.
    private bool AnyRunning()
    {
        foreach (var shard in _shards)
        {
            if (shard.Running)
            {
                return true;
            }
        }

        return false;
    }

    // The job of the draw the render thread runs now; set around each deferred draw.
    public Job? Current { get; set; }

    public long Submitted { get; private set; }

    // What became of the jobs since the last TakeCounters: resolved, abandoned (the resolution threw),
    // taken by their draw, refused when taken (inputs or ownership changed), and the worker's wait.
    private long _resolvedCount;
    private long _abandonedCount;
    private long _takenCount;
    private long _refusedCount;
    private long _waitTicks;
    private static int _abandonReports;
    private static readonly int MaxAbandonReports =
        int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_PREFETCH_ABANDON_REPORTS"), out var reports) ? reports : 0;

    // Dispatch jobs since the last TakeComputeCounters: submitted, taken, refused by reason (ComputeRefusal), the worker's
    // wait for them, and the resolving time of the taken ones, which the worker would otherwise have spent itself.
    public enum ComputeRefusal
    {
        // The resolution threw (a GPU-owned read, a fatal condition), so the dispatch resolves it.
        Abandoned,
        // The provider refused to stand for the dispatch (ComputePrefetchRefusal).
        Provider,
        // Different banks or arguments.
        Inputs,
        // The ownership answers it read under changed (IConcurrentResolveHost.IsStillCurrentAcrossDispatches).
        Ownership,
        // A dispatch it was resolved ahead of did more than encode GPU work (NoteDispatchEffect; a kernel fill or copy
        // the host logs is left to Ownership, ExactEffects), or a collector call it was resolved ahead of noted an
        // effect (NoteCollect).
        Effects,
        // The program cache stamp moved since the resolution began (Job.ProgramState), as DrawLookaheadRefusal.Replanned.
        Replanned,
        Count,
    }

    // =verify: taken dispatch jobs the worker resolved again and compared (VerifyCompute), those whose difference the
    // moved answers excuse (refused, the dispatch resolves itself), and those that differed under answers that held.
    private long _computeVerified, _computeVerifyRaced, _computeVerifyFailed;

    private long _computeSubmitted;
    private long _computeTaken;
    private long _computeWaitTicks;
    private long _computeSavedTicks;
    private readonly long[] _computeRefusals = new long[(int)ComputeRefusal.Count];
    private readonly long[] _providerRefusals = new long[(int)ComputePrefetchRefusal.Unavailable + 1];

    // The provider's reasons behind ComputeRefusal.Provider, by ComputePrefetchRefusal; resets them.
    public long[] TakeProviderRefusals()
    {
        var refused = new long[_providerRefusals.Length];
        for (var index = 0; index < refused.Length; index++)
        {
            refused[index] = Interlocked.Exchange(ref _providerRefusals[index], 0);
        }

        return refused;
    }

    public (long Submitted, long Taken, long[] Refused, double WaitMilliseconds, double SavedMilliseconds) TakeComputeCounters()
    {
        var refused = new long[(int)ComputeRefusal.Count];
        for (var index = 0; index < refused.Length; index++)
        {
            refused[index] = Interlocked.Exchange(ref _computeRefusals[index], 0);
        }

        return (Interlocked.Exchange(ref _computeSubmitted, 0), Interlocked.Exchange(ref _computeTaken, 0), refused,
            Interlocked.Exchange(ref _computeWaitTicks, 0) * 1000.0 / System.Diagnostics.Stopwatch.Frequency,
            Interlocked.Exchange(ref _computeSavedTicks, 0) * 1000.0 / System.Diagnostics.Stopwatch.Frequency);
    }

    internal void NoteComputeRefusal(ComputeRefusal reason) => Interlocked.Increment(ref _computeRefusals[(int)reason]);

    // The gated non-draw item the worker runs now is a dispatch that did more than encode GPU work: it was consumed,
    // copied an image, or has CPU-predicted writes. A dispatch job resolved ahead of it (its gate below this item) is
    // refused, since what that dispatch did may not show in the ownership answers. -1 while none did.
    private long _lastDispatchEffect = -1;

    public void NoteDispatchEffect() => NoteDispatchEffect(DispatchEffectKind.Unknown);

    // Kernel fills and copies (DispatchEffectKind.KernelFill, KernelCopy) on a host that logs them as GPU writes
    // (IConcurrentResolveHost.KernelFillWritesLogged) are noted apart (_lastExactDispatchEffect): a dispatch job resolved
    // ahead of one is not refused for it, since the take's IsStillCurrentAcrossDispatches already refuses a job whose
    // reads a write logged since its resolution began covers, the fill's or copy's as any dispatch's store. Every
    // other kind keeps the blunt refusal. A draw's take still refuses on either (TryTakeGraphics).
    public void NoteDispatchEffect(DispatchEffectKind kind)
    {
        Interlocked.Increment(ref _dispatchEffects[(int)kind]);
        Volatile.Write(ref IsExactEffect(kind) ? ref _lastExactDispatchEffect : ref _lastDispatchEffect, Volatile.Read(ref _executedGate));
    }

    private bool IsExactEffect(DispatchEffectKind kind) =>
        ExactEffects && kind is DispatchEffectKind.KernelFill or DispatchEffectKind.KernelCopy && _concurrent.KernelFillWritesLogged;

    private static readonly bool ExactEffectsDefault =
        Environment.GetEnvironmentVariable("SHARPEMU_PREFETCH_EXACT_EFFECTS") != "0";

    // SHARPEMU_PREFETCH_EXACT_EFFECTS=0 refuses dispatch jobs for every effect kind again. Tests set it on one instance.
    internal bool ExactEffects { get; set; } = ExactEffectsDefault;

    // The last kernel fill or copy noted exactly (NoteDispatchEffect), as _lastDispatchEffect; -1 while none was.
    private long _lastExactDispatchEffect = -1;

    // Effects noted by kind, and the dispatch jobs an exact effect stood in the window of (the blunt rule would have
    // refused them) that the take refused anyway (ownership or write log) or took.
    private readonly long[] _dispatchEffects = new long[(int)DispatchEffectKind.Count];
    private long _exactEffectRefused, _exactEffectTaken;

    private static readonly string[] DispatchEffectNames =
        ["unknown", "consumed", "image_copy", "known_write", "metadata_clear", "constant_fill", "dcc_fill", "kernel_fill", "kernel_copy", "image_clear"];

    // =verify: the compute program hash of each dispatch by the gate number it brings the count to (the worker runs item
    // _executedGate + 1), so a fatal names the dispatches a draw looked over. Written and read on the worker only.
    private readonly ulong[] _gateDispatchHashes = new ulong[GateKindRing];

    public void NoteDispatchProgram(ComputeProgram program)
    {
        if (RecordsForVerify)
        {
            _gateDispatchHashes[(Volatile.Read(ref _executedGate) + 1) & (GateKindRing - 1)] = program.Input.Stage.Program?.Hash ?? 0;
        }
    }

    // The gated items a resolution that began after `executedAtStart` of them ran looked over, up to the draw's full gate:
    // each one's kind, and a dispatch's program hash when the worker resolved one for it.
    private string DescribeLookedOver(long executedAtStart, long fullGate)
    {
        var builder = new System.Text.StringBuilder();
        for (var gate = executedAtStart + 1; gate <= fullGate && gate <= executedAtStart + 16; gate++)
        {
            var kind = GateKindOf(gate);
            builder.Append(builder.Length == 0 ? string.Empty : ",").Append(gate).Append(':').Append(GateItemKindNames[(int)kind]);
            if (kind == GateItemKind.Dispatch)
            {
                builder.Append("=0x").Append(_gateDispatchHashes[gate & (GateKindRing - 1)].ToString("X16"));
            }
        }

        if (fullGate > executedAtStart + 16)
        {
            builder.Append(",...");
        }

        return builder.ToString();
    }

    // The kind of each gated non-draw item by its gate number, for the counters below. A job's gate is never more than the
    // command thread's queue depth (at most 4096 items) behind the newest, so a ring twice that holds every gate in use.
    private const int GateKindRing = 8192;
    private readonly GateItemKind[] _gateKinds = new GateItemKind[GateKindRing];

    // On the command thread, as it queues the gated non-draw item that brings its gate count to gate.
    public void NoteGateItem(long gate, GateItemKind kind)
    {
        _gateKinds[gate & (GateKindRing - 1)] = kind;
        if (kind == GateItemKind.Collect)
        {
            _lastCollectGate = gate;
        }
    }

    private GateItemKind GateKindOf(long gate) => gate <= 0 ? GateItemKind.None : _gateKinds[gate & (GateKindRing - 1)];

    // Per kind of the item that last gated a draw's job: the draws submitted with a gate above the draw before them
    // (draw_gate_raises), the draw takes that found their job unresolved (busy or queued), the subset whose job the
    // resolving thread had reached before its gate opened (gate-bound), and the worker's wait in them. The kind is that
    // of the job's own gate (Job.Gate): where draws look ahead of dispatches and collector calls it is the last item
    // they do not, often a synchronous call or an idle commit long before the draw, so an unresolved take counted
    // under a kind was held by it only when it is gate-bound as well.
    private long _lastDrawGate;
    private readonly long[] _drawGateRaises = new long[(int)GateItemKind.Count];
    private readonly long[] _unresolvedByGate = new long[(int)GateItemKind.Count];
    private readonly long[] _gateBoundByGate = new long[(int)GateItemKind.Count];
    private readonly long[] _unresolvedWaitTicksByGate = new long[(int)GateItemKind.Count];
    private long _unresolvedStarted;

    private static string FormatByKind(long[] counts, Func<long, string>? format = null)
    {
        var builder = new System.Text.StringBuilder();
        for (var kind = 1; kind < counts.Length; kind++)
        {
            var value = Interlocked.Exchange(ref counts[kind], 0);
            builder.Append(kind == 1 ? string.Empty : ",").Append(GateItemKindNames[kind]).Append('=')
                .Append(format is null ? value.ToString(System.Globalization.CultureInfo.InvariantCulture) : format(value));
        }

        return builder.ToString();
    }

    public (long Resolved, long Abandoned, long Taken, long Refused, double WaitMilliseconds) TakeCounters() =>
        (Interlocked.Exchange(ref _resolvedCount, 0), Interlocked.Exchange(ref _abandonedCount, 0),
            Interlocked.Exchange(ref _takenCount, 0), Interlocked.Exchange(ref _refusedCount, 0),
            Interlocked.Exchange(ref _waitTicks, 0) * 1000.0 / System.Diagnostics.Stopwatch.Frequency);

    // A gate above zero holds the job until AdvanceGate reaches it: with the command stream on its own
    // thread, draws are submitted ahead of the work before them, which may change what they read.
    public Job Submit(RegisterBanks banks, long gate = 0, bool indexed = false, int startInstanceUserData = -1) =>
        SubmitDrawAhead(banks, gate, gate, indexed, startInstanceUserData);

    // A draw's job whose resolution may start once the first gate items ran, before the rest up to fullGate (dispatches
    // and collector calls only, the sink's DispatchGate) did; with gate == fullGate, Submit.
    public Job SubmitDrawAhead(RegisterBanks banks, long gate, long fullGate, bool indexed = false, int startInstanceUserData = -1)
    {
        fullGate = Math.Max(gate, fullGate);
        // The command thread alone submits draws with a gate.
        if (fullGate > _lastDrawGate)
        {
            Interlocked.Increment(ref _drawGateRaises[(int)GateKindOf(fullGate)]);
        }

        _lastDrawGate = fullGate;
        return Enqueue(new Job(banks, gate, indexed, startInstanceUserData) { FullGate = fullGate, LastCollectGate = _lastCollectGate });
    }

    private Job Enqueue(Job job)
    {
        var shard = ShardFor(job);
        job.Shard = shard.Index;
        lock (_lock)
        {
            job.Index = Submitted;
            shard.Queue.Enqueue(job);
            Submitted++;
            _outstanding++;
            // Only the shard's thread asleep on an empty queue waits for this job: the others on this lock wait
            // for jobs to finish, which the resolving threads signal. A pulse per draw was a wake for nothing
            // while the thread was busy resolving the draws before.
            if (shard.IdleForJobs)
            {
                Monitor.PulseAll(_lock);
            }
        }

        return job;
    }

    // A dispatch's job. Its gate may stop short of the dispatches just before it (the sink's DispatchGate): its program
    // then resolves while they run, and the take checks it against what they did (TryTakeCompute).
    public Job SubmitCompute(RegisterBanks banks, long gate, uint dispatchInitiator, uint groupsX, uint groupsY, uint groupsZ, bool indirect)
    {
        var job = new Job(banks, gate) { Compute = new ComputeArguments(dispatchInitiator, groupsX, groupsY, groupsZ, indirect) };
        Interlocked.Increment(ref _computeSubmitted);
        return Enqueue(job);
    }

    // Returns once the worker has finished every submitted job: it reads guest memory and cache
    // state, which the render thread changes only with the worker idle.
    public void WaitIdle()
    {
        if (Volatile.Read(ref _outstanding) == 0)
        {
            return;
        }

        using var profile = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ProgramPrefetchDrainWait);
        lock (_lock)
        {
            while (_outstanding != 0)
            {
                _waiters++;
                Monitor.Wait(_lock);
                _waiters--;
            }
        }
    }

    // The render thread ran this many non-draw items: jobs gated on them may resolve.
    public void AdvanceGate(long executed)
    {
        Interlocked.Exchange(ref _executedGate, executed);
        // Only a resolving thread asleep on a gate this reaches needs a wake: one per item costs the render
        // thread a kernel call each.
        foreach (var shard in _shards)
        {
            if (Volatile.Read(ref shard.SleepingOnGate) <= executed)
            {
                lock (_lock)
                {
                    Monitor.PulseAll(_lock);
                }

                return;
            }
        }
    }

    // Returns once the first count jobs submitted are finished, on whichever shard (the finished prefix), so a
    // non-draw item runs with no resolution of the draws before it still reading.
    public void WaitFinished(long count)
    {
        if (Volatile.Read(ref _finished) >= count)
        {
            return;
        }

        using var profile = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ProgramPrefetchOrderWait);
        lock (_lock)
        {
            while (_finished < count)
            {
                _waiters++;
                Monitor.Wait(_lock);
                _waiters--;
            }
        }
    }

    // Holds the worker between jobs, for host work that changes what a resolution reads.
    public void Pause()
    {
        using var profile = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ProgramPrefetchPauseWait);
        lock (_lock)
        {
            _paused = true;
            // Every shard's running resolution, its thread's or the caller's.
            while (AnyRunning())
            {
                _waiters++;
                Monitor.Wait(_lock);
                _waiters--;
            }
        }
    }

    // Wakes the worker only when the pause holds it. The render thread pauses around every relay
    // batch, and a pulse to a worker idle in its queue wait costs a kernel call each time.
    public void Resume()
    {
        lock (_lock)
        {
            _paused = false;
            foreach (var shard in _shards)
            {
                if (shard.WaitingOnPause)
                {
                    Monitor.PulseAll(_lock);
                    break;
                }
            }
        }
    }

    public bool TryTakeGraphics(RegisterBanks banks, in GraphicsProgramInputs inputs, out GraphicsPrograms programs)
    {
        programs = null!;
        var job = Current;
        if (job is null || !ReferenceEquals(job.Banks, banks) || job.Compute is not null)
        {
            return false;
        }

        Current = null;
        // A completed job waits for nothing, so it reads no clock: almost every take in a
        // well-prefetched burst is one, and two clock reads per draw were measurable.
        if (job.State == 0)
        {
            var gateKind = NoteUnresolvedTake(job, out var behind);
            var waitStarted = System.Diagnostics.Stopwatch.GetTimestamp();
            // Caller work is reported separately, not disguised as a shorter resolver wait.
            // Keep long queues out of the claim routine's lock/finally frame.
            if (!_allowCallerResolution ||
                (Volatile.Read(ref _outstanding) > MaximumCallerOutstanding && !(ClaimSteal && Volatile.Read(ref _shards[job.Shard].WorkerClaimed))) ||
                !TryResolveOnCaller(job))
            {
                WaitFor(job);
                var waited = System.Diagnostics.Stopwatch.GetTimestamp() - waitStarted;
                Interlocked.Add(ref _waitTicks, waited);
                Interlocked.Add(ref _unresolvedWaitTicksByGate[(int)gateKind], waited);
                Interlocked.Add(ref behind ? ref _unresolvedBehindWaitTicks : ref _unresolvedFrontierWaitTicks, waited);
            }
        }

        using var profile = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ProgramPrefetchValidation);
        // The resolution began before every gated item ahead of the draw ran: it looked ahead of dispatches, and of a
        // collector call when one of those items was one.
        var lookedAhead = job.ExecutedAtStart < job.FullGate;
        var overCollect = lookedAhead && job.LastCollectGate > job.ExecutedAtStart;
        if (lookedAhead)
        {
            Interlocked.Increment(ref _lookaheadJobs);
            if (overCollect)
            {
                Interlocked.Increment(ref _collectLookaheadJobs);
            }
        }

        if (job.State != 1)
        {
            if (lookedAhead)
            {
                Interlocked.Increment(ref _lookaheadRefusals[(int)DrawLookaheadRefusal.Abandoned]);
                if (overCollect)
                {
                    Interlocked.Increment(ref _collectLookaheadRefused);
                }
            }

            ReleaseAssumptions(job);
            return false;
        }

        var refusal = !job.Inputs.Equals(inputs) ? DrawLookaheadRefusal.Inputs
            : job.ProgramState != _executor.ProgramState ? DrawLookaheadRefusal.Replanned
            : !lookedAhead ? (_concurrent.IsStillCurrent(job.Assumptions!) ? DrawLookaheadRefusal.Count : DrawLookaheadRefusal.Ownership)
            // A collector call or a dispatch that ran after the resolution began (gate above ExecutedAtStart) noted an effect.
            : Volatile.Read(ref _lastCollectEffect) >= job.ExecutedAtStart ? DrawLookaheadRefusal.CollectEffects
            // Exact effects too: the draw's take keeps the blunt rule (NoteDispatchEffect).
            : Math.Max(Volatile.Read(ref _lastDispatchEffect), Volatile.Read(ref _lastExactDispatchEffect)) >= job.ExecutedAtStart
                ? DrawLookaheadRefusal.Effects
            : !_concurrent.IsStillCurrentAcrossDispatches(job.Assumptions!) ? DrawLookaheadRefusal.Ownership
            : DrawLookaheadRefusal.Count;
        // =verify keeps a taken look-ahead's assumptions to its comparison (VerifyLookahead): a difference whose answers
        // changed after the take is a race, not a hole.
        var verifies = refusal == DrawLookaheadRefusal.Count && lookedAhead && LookaheadMode == DrawLookaheadMode.Verify;
        if (!verifies)
        {
            ReleaseAssumptions(job);
        }

        if (refusal != DrawLookaheadRefusal.Count)
        {
            if (lookedAhead)
            {
                Interlocked.Increment(ref _lookaheadRefusals[(int)refusal]);
                if (overCollect)
                {
                    Interlocked.Increment(ref _collectLookaheadRefused);
                }
            }

            if (refusal == DrawLookaheadRefusal.Replanned)
            {
                Interlocked.Increment(ref _replannedRefusals);
            }

            Interlocked.Increment(ref _refusedCount);
            return false;
        }

        Interlocked.Increment(ref _takenCount);
        programs = job.Programs!;
        if (lookedAhead)
        {
            Interlocked.Increment(ref _lookaheadTaken);
            if (overCollect)
            {
                Interlocked.Increment(ref _collectLookaheadTaken);
            }

            if (verifies)
            {
                try
                {
                    programs = VerifyLookahead(job, banks, in inputs);
                }
                finally
                {
                    ReleaseAssumptions(job);
                }
            }
        }

        if (LookaheadMode == DrawLookaheadMode.Census)
        {
            programs = TakeCensus(job, banks, in inputs, lookedAhead);
        }

        return true;
    }

    // =census counters, [0] jobs that waited for their gate, [1] look-aheads: taken jobs resolved again, those that
    // differed, and the ticks they stood between resolution and take.
    private readonly long[] _censusResolved = new long[2], _censusDiffered = new long[2], _censusStoodTicks = new long[2];
    private readonly int[] _censusReports = new int[2];
    private const int MaximumCensusReports = 8;

    // SHARPEMU_PREFETCH_CENSUS_PS=<pixel program hash, hex>: the census records that pixel stage's reads on both threads.
    private static readonly ulong CensusPixelHash =
        ulong.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_PREFETCH_CENSUS_PS")?.Replace("0x", "", StringComparison.OrdinalIgnoreCase),
            System.Globalization.NumberStyles.HexNumber, System.Globalization.CultureInfo.InvariantCulture, out var hash) ? hash : 0;

    private List<(ulong Address, uint Word, bool Clean, bool Refused)>? CensusReadsOf(GraphicsPrograms? programs) =>
        LookaheadMode == DrawLookaheadMode.Census && CensusPixelHash != 0 &&
        programs is { Available: true } && programs.PixelInput.Stage.Program?.Hash == CensusPixelHash
            ? SharpEmu.ShaderCompiler.Resources.ResourceMaterializer.RecordReads(programs.PixelInput.Stage.Resources, clean: true)
            : null;

    private string? DescribeRefusedCleanReads(List<(ulong Address, uint Word, bool Clean, bool Refused)>? reads)
    {
        if (reads is null)
        {
            return null;
        }

        var described = new List<string>();
        foreach (var read in reads)
        {
            if (read.Refused && described.Count < 3)
            {
                described.Add($"0x{read.Address:X}:({_executor.DescribeGuestRange(read.Address, sizeof(uint))})");
            }
        }

        return string.Join(" ", described);
    }

    // The reads of the job (resolving thread) and the worker side by side where they differ, by position, and what the
    // host knows of the first differing addresses.
    private string DescribeCensusReads(Job job, List<(ulong Address, uint Word, bool Clean, bool Refused)>? worker,
        List<(ulong Address, uint Word, bool Clean, bool Refused)>? again)
    {
        if (job.CensusReads is not { } ahead || worker is null)
        {
            return $"reads=unavailable(job={job.CensusReads?.Count.ToString() ?? "null"},worker={worker?.Count.ToString() ?? "null"})";
        }

        static string Read((ulong Address, uint Word, bool Clean, bool Refused) read) =>
            $"0x{read.Address:X}{(read.Clean ? "c" : "")}{(read.Refused ? "!" : "")}={read.Word:X8}";

        var builder = new System.Text.StringBuilder();
        builder.Append($"reads=(job={ahead.Count},worker={worker.Count},again={again?.Count.ToString() ?? "null"}) ");
        var listed = 0;
        var described = new List<ulong>();
        for (var index = 0; index < Math.Max(ahead.Count, worker.Count) && listed < 40; index++)
        {
            var a = index < ahead.Count ? ahead[index] : default;
            var w = index < worker.Count ? worker[index] : default;
            if (index < ahead.Count && index < worker.Count && a == w)
            {
                continue;
            }

            listed++;
            var g = again is not null && index < again.Count ? Read(again[index]) : "-";
            builder.Append($"[{index}] job={(index < ahead.Count ? Read(a) : "-")} worker={(index < worker.Count ? Read(w) : "-")} again={g}; ");
            foreach (var address in new[] { a.Address, w.Address })
            {
                var line = address & ~0xFFUL;
                if (address != 0 && described.Count < 3 && !described.Contains(line))
                {
                    described.Add(line);
                }
            }
        }

        foreach (var line in described)
        {
            builder.Append($"host[0x{line:X}+0x100]=({_executor.DescribeGuestRange(line, 0x100)}) ");
        }

        builder.Append($"refused_at_resolution=[{job.CensusRefusals}] same_words_now=[");
        foreach (var read in ahead)
        {
            if (read.Refused)
            {
                builder.Append($"0x{read.Address:X}:({_executor.DescribeGuestRange(read.Address, sizeof(uint))}) ");
            }
        }

        builder.Append(']');

        return builder.ToString();
    }

    private GraphicsPrograms TakeCensus(Job job, RegisterBanks banks, in GraphicsProgramInputs inputs, bool lookedAhead)
    {
        var stood = System.Diagnostics.Stopwatch.GetTimestamp() - job.ResolvedAt;
        var resolved = _executor.ResolveGraphicsPrograms(banks, in inputs);
        var workerReads = job.CensusReads is null ? null : CensusReadsOfAny(resolved);
        var group = lookedAhead ? 1 : 0;
        Interlocked.Increment(ref _censusResolved[group]);
        Interlocked.Add(ref _censusStoodTicks[group], stood);
        if (SharpEmu.Libs.Gpu.Pipelines.GraphicsProgramsComparer.Difference(job.Programs!, resolved) is { } difference)
        {
            Interlocked.Increment(ref _censusDiffered[group]);
            if (_censusReports[group] < MaximumCensusReports)
            {
                _censusReports[group]++;
                Console.Error.WriteLine($"[GPU][PREFETCH] census differs group={(lookedAhead ? "ahead" : "gated")} {difference} " +
                    $"stood_us={stood * 1_000_000 / System.Diagnostics.Stopwatch.Frequency} (job {job.Index}, gate {job.Gate}, full gate {job.FullGate}, " +
                    $"executed at start {job.ExecutedAtStart}) " +
                    $"job_vs_worker=[{SharpEmu.Libs.Gpu.Pipelines.GraphicsProgramsComparer.DescribeStages(job.Programs!, resolved)}] " +
                    $"looked_over=[{DescribeLookedOver(job.ExecutedAtStart, job.FullGate)}] " +
                    $"state_job=[{job.ResolveState}] state_now=[{_executor.DescribeProgramState()}] " +
                    DescribeCensusReads(job, workerReads, job.CensusReads is null ? null : CensusReadsOfAny(resolved)));
            }
        }

        return resolved;
    }

    // The worker's side of the census reads: its pixel stage's reads whatever program it picked.
    private static List<(ulong Address, uint Word, bool Clean, bool Refused)>? CensusReadsOfAny(GraphicsPrograms? programs) =>
        programs is { Available: true } && programs.PixelInput.Stage.Program is not null
            ? SharpEmu.ShaderCompiler.Resources.ResourceMaterializer.RecordReads(programs.PixelInput.Stage.Resources, clean: true)
            : null;

    // SHARPEMU_PREFETCH_DRAW_LOOKAHEAD=verify: the worker resolves the draw as it would have without the job, and the two
    // must hand the executor the same programs, inputs and resources (GraphicsProgramsComparer).
    private GraphicsPrograms VerifyLookahead(Job job, RegisterBanks banks, in GraphicsProgramInputs inputs)
    {
        var resolved = _executor.ResolveGraphicsPrograms(banks, in inputs);
        Interlocked.Increment(ref _lookaheadVerified);
        if (SharpEmu.Libs.Gpu.Pipelines.GraphicsProgramsComparer.Difference(job.Programs!, resolved) is { } difference)
        {
            // The take's answers no longer hold (an owned answer turned clean: a recorded GPU write completed between
            // the take and this resolution, and a refused clean read is answered now): both resolutions are right for
            // their moment, the worker's is the later one. Counted, not fatal; a difference under answers that still
            // hold is a hole and stops below.
            if (job.Assumptions is { } assumptions && !_concurrent.IsStillCurrentAcrossDispatches(assumptions))
            {
                Interlocked.Increment(ref _lookaheadVerifyRaced);
                return resolved;
            }

            // What the next verify run needs to tell a hole in the take check from a resolution that differs whatever ran
            // before it: every differing group, both resolutions' programs, whether the worker resolves the same again,
            // the items looked over (a dispatch's program hash) and the effects noted, and the program cache state the
            // look-ahead resolution ended under against the state now.
            var again = _executor.ResolveGraphicsPrograms(banks, in inputs);
            var stable = SharpEmu.Libs.Gpu.Pipelines.GraphicsProgramsComparer.Difference(resolved, again) is { } againDifference
                ? "differs again: " + againDifference
                : "same";
            throw _fatal($"A draw's programs resolved ahead of the dispatches or collector calls before it differ from the worker's: {difference} " +
                $"(job {job.Index}, gate {job.Gate}, full gate {job.FullGate}, executed at start {job.ExecutedAtStart}, last collector call {job.LastCollectGate}). " +
                $"differences=[{SharpEmu.Libs.Gpu.Pipelines.GraphicsProgramsComparer.Differences(job.Programs!, resolved)}] " +
                $"ahead=[{SharpEmu.Libs.Gpu.Pipelines.GraphicsProgramsComparer.Describe(job.Programs!)}] " +
                $"worker=[{SharpEmu.Libs.Gpu.Pipelines.GraphicsProgramsComparer.Describe(resolved)}] worker_again={stable} " +
                $"ahead_vs_worker=[{SharpEmu.Libs.Gpu.Pipelines.GraphicsProgramsComparer.DescribeStages(job.Programs!, resolved)}] " +
                $"looked_over=[{DescribeLookedOver(job.ExecutedAtStart, job.FullGate)}] " +
                $"last_dispatch_effect={Volatile.Read(ref _lastDispatchEffect)} last_collect_effect={Volatile.Read(ref _lastCollectEffect)} " +
                $"state_ahead=[{job.ResolveState}] state_now=[{_executor.DescribeProgramState()}].");
        }

        return resolved;
    }

    // The dispatch the worker runs now takes the program its job resolved when the job was resolved from the same banks
    // and arguments, the provider stood for it, no dispatch it was resolved ahead of did more than encode GPU work (nor
    // collector call noted an effect), and the host's ownership answers still hold across those items; otherwise it
    // resolves the program itself.
    public bool TryTakeCompute(RegisterBanks banks, uint dispatchInitiator, uint groupsX, uint groupsY, uint groupsZ, bool indirect,
        out ComputeProgram program)
    {
        program = null!;
        var job = Current;
        if (job?.Compute is not { } arguments || !ReferenceEquals(job.Banks, banks))
        {
            if (job?.Compute is not null)
            {
                Current = null;
                NoteComputeRefusal(ComputeRefusal.Inputs);
            }

            return false;
        }

        Current = null;
        if (job.State == 0)
        {
            var waitStarted = System.Diagnostics.Stopwatch.GetTimestamp();
            if (!_allowCallerResolution ||
                (Volatile.Read(ref _outstanding) > MaximumCallerOutstanding && !(ClaimSteal && Volatile.Read(ref _shards[job.Shard].WorkerClaimed))) ||
                !TryResolveOnCaller(job))
            {
                WaitFor(job);
                Interlocked.Add(ref _computeWaitTicks, System.Diagnostics.Stopwatch.GetTimestamp() - waitStarted);
            }
        }

        using var profile = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ProgramPrefetchValidation);
        var refusal = job.State != 1 ? ComputeRefusal.Abandoned
            : job.ComputeResult is null ? ComputeRefusal.Provider
            : !arguments.Matches(dispatchInitiator, groupsX, groupsY, groupsZ, indirect) ? ComputeRefusal.Inputs
            : job.ProgramState != _executor.ProgramState ? ComputeRefusal.Replanned
            : HasComputeEffects(job) ? ComputeRefusal.Effects
            : !_concurrent.IsStillCurrentAcrossDispatches(job.Assumptions!) ? ComputeRefusal.Ownership
            : ComputeRefusal.Count;
        // A kernel fill or copy the blunt rule would have refused it for: the ownership check above decided instead.
        if (refusal is ComputeRefusal.Count or ComputeRefusal.Ownership &&
            Volatile.Read(ref _lastExactDispatchEffect) >= (ComputeEffectsFromStart ? job.ExecutedAtStart : job.Gate))
        {
            Interlocked.Increment(ref refusal == ComputeRefusal.Count ? ref _exactEffectTaken : ref _exactEffectRefused);
        }

        if (refusal == ComputeRefusal.Count && job.ExecutedAtStart < Volatile.Read(ref _executedGate))
        {
            Interlocked.Increment(ref _computeTakenOver);
        }
        // =verify keeps a taken job's assumptions to its comparison (VerifyCompute), as for a draw's look-ahead.
        var verifies = refusal == ComputeRefusal.Count && LookaheadMode == DrawLookaheadMode.Verify;
        if (!verifies)
        {
            ReleaseAssumptions(job);
        }

        if (refusal != ComputeRefusal.Count)
        {
            NoteComputeRefusal(refusal);
            if (refusal == ComputeRefusal.Provider)
            {
                Interlocked.Increment(ref _providerRefusals[(int)job.ComputeRefusal]);
            }

            return false;
        }

        program = job.ComputeResult!;
        if (verifies)
        {
            try
            {
                if (VerifyCompute(job, banks, arguments) is not { } verified)
                {
                    return false;
                }

                program = verified;
            }
            finally
            {
                ReleaseAssumptions(job);
            }
        }

        Interlocked.Increment(ref _computeTaken);
        Interlocked.Add(ref _computeSavedTicks, job.ResolveTicks);
        return true;
    }

    // SHARPEMU_PREFETCH_DRAW_LOOKAHEAD=verify: the worker resolves the taken dispatch again through the same side-effect-free
    // path (nothing consumed, predicted or submitted) and both must hand the executor the same program, inputs and
    // resources (GraphicsProgramsComparer.ComputeDifference). The worker's result stands; a difference under answers
    // that moved since the resolution (IsStillCurrentAcrossDispatches, which refuses a GPU write logged since it began
    // over what it read) is a race, refused and counted, and the dispatch resolves itself; any other is fatal.
    private ComputeProgram? VerifyCompute(Job job, RegisterBanks banks, ComputeArguments arguments)
    {
        var resolved = _executor.ResolveComputeProgramAhead(banks, arguments.DispatchInitiator, arguments.GroupsX, arguments.GroupsY,
            arguments.GroupsZ, arguments.Indirect, out var workerRefusal);
        Interlocked.Increment(ref _computeVerified);
        var difference = resolved is null
            ? $"worker_refused={workerRefusal}"
            : SharpEmu.Libs.Gpu.Pipelines.GraphicsProgramsComparer.ComputeDifference(job.ComputeResult!, resolved);
        if (difference is null)
        {
            return resolved;
        }

        if (job.Assumptions is { } assumptions && !_concurrent.IsStillCurrentAcrossDispatches(assumptions))
        {
            Interlocked.Increment(ref _computeVerifyRaced);
            return null;
        }

        Interlocked.Increment(ref _computeVerifyFailed);
        var executed = Volatile.Read(ref _executedGate);
        throw _fatal($"A dispatch's program resolved ahead of it differs from the worker's: {difference} " +
            $"(job {job.Index}, gate {job.Gate}, executed at start {job.ExecutedAtStart}, executed now {executed}). " +
            (resolved is null ? string.Empty : $"{SharpEmu.Libs.Gpu.Pipelines.GraphicsProgramsComparer.DescribeCompute(job.ComputeResult!, resolved)} ") +
            $"looked_over=[{DescribeLookedOver(job.ExecutedAtStart, executed)}] " +
            $"last_dispatch_effect={Volatile.Read(ref _lastDispatchEffect)} last_exact_effect={Volatile.Read(ref _lastExactDispatchEffect)} " +
            $"last_collect_effect={Volatile.Read(ref _lastCollectEffect)} " +
            $"stamp_ahead={job.ProgramState} stamp_now={_executor.ProgramState} " +
            $"state_ahead=[{job.ResolveState}] state_now=[{_executor.DescribeProgramState()}].");
    }

    // Whether a dispatch or collector call that ran after the job's resolution began noted an effect, measured from
    // ExecutedAtStart as the draw take does. The worker runs item g with _executedGate = g-1, so an effect at item g
    // records g-1 (NoteDispatchEffect, NoteCollect), and AdvanceGate(g) publishes the count only after item g ran. A
    // resolution read ExecutedAtStart before anything else, so item g had finished before it read anything exactly when
    // g <= ExecutedAtStart; g-1 >= ExecutedAtStart is an item that had not, whose effect it may have missed. Effects
    // between Gate and ExecutedAtStart ran before the resolution began and are in what it read.
    // SHARPEMU_PREFETCH_COMPUTE_WINDOW=0 measures from Gate again (every item since the earliest allowed start).
    private bool HasComputeEffects(Job job)
    {
        var effect = Math.Max(Volatile.Read(ref _lastDispatchEffect), Volatile.Read(ref _lastCollectEffect));
        var byGate = effect >= job.Gate;
        var byStart = effect >= job.ExecutedAtStart;
        if (byGate)
        {
            Interlocked.Increment(ref _computeEffectsByGate);
        }

        if (byStart)
        {
            Interlocked.Increment(ref _computeEffectsByStart);
        }

        return ComputeEffectsFromStart ? byStart : byGate;
    }

    private static readonly bool ComputeWindowDefault =
        Environment.GetEnvironmentVariable("SHARPEMU_PREFETCH_COMPUTE_WINDOW") != "0";

    // Tests set it on one instance.
    internal bool ComputeEffectsFromStart { get; set; } = ComputeWindowDefault;

    // effects_window: jobs that reached the effects test and the old (gate) rule and the new (started) rule each refuse,
    // and the taken jobs whose resolution began before at least one item ahead of the dispatch ran.
    private long _computeEffectsByGate, _computeEffectsByStart, _computeTakenOver;

    public (long Verified, long Raced, long Failed) TakeComputeVerifyCounters() =>
        (Interlocked.Exchange(ref _computeVerified, 0), Interlocked.Exchange(ref _computeVerifyRaced, 0),
            Interlocked.Exchange(ref _computeVerifyFailed, 0));

    // The dispatch counters beyond TakeComputeCounters, for the host's prefetch line: the compute verify's.
    public string TakeComputeReport()
    {
        var verify = TakeComputeVerifyCounters();
        return $"effects_window=(mode={(ComputeEffectsFromStart ? "started" : "gate")},gate={Interlocked.Exchange(ref _computeEffectsByGate, 0)}," +
            $"started={Interlocked.Exchange(ref _computeEffectsByStart, 0)},taken_over={Interlocked.Exchange(ref _computeTakenOver, 0)}) " +
            $"{TakeDispatchEffectReport()} " +
            $"dispatch_verify=(verified={verify.Verified},raced={verify.Raced},failed={verify.Failed})";
    }

    // dispatch_effects=(kind=noted,...) and exact_effects=(mode, refused, taken): the effects noted by kind, and the
    // dispatch jobs a kernel fill or copy stood in the effects window of that the ownership check refused or took
    // (the blunt rule would have refused both). Resets them.
    private string TakeDispatchEffectReport()
    {
        var builder = new System.Text.StringBuilder("dispatch_effects=(");
        for (var index = 0; index < _dispatchEffects.Length; index++)
        {
            builder.Append(index == 0 ? string.Empty : ",").Append(DispatchEffectNames[index]).Append('=')
                .Append(Interlocked.Exchange(ref _dispatchEffects[index], 0));
        }

        return builder.Append(") exact_effects=(mode=").Append(ExactEffects && _concurrent.KernelFillWritesLogged ? "on" : "off")
            .Append(",refused=").Append(Interlocked.Exchange(ref _exactEffectRefused, 0))
            .Append(",taken=").Append(Interlocked.Exchange(ref _exactEffectTaken, 0)).Append(')').ToString();
    }

    // Why a draw's job was not resolved when its draw took it: still gated on the work before it, the resolving
    // thread busy with another job, or queued behind others with its gate open (takes_unresolved).
    private long _unresolvedGated, _unresolvedBusy, _unresolvedQueued;

    // The same takes by what bounded the resolving thread's lead (takes_unresolved frontier= and behind=): at the
    // frontier the job was the newest submitted, so the stream had produced nothing further for it to resolve ahead
    // (its lead is the command thread's); behind, later jobs were already queued and the one resolving thread, which
    // resolves in submission order, had not got through this one yet. With the worker's wait in each. Submitted is
    // read without the lock: a stale value only moves a take between the two.
    private long _unresolvedFrontier, _unresolvedBehind, _unresolvedFrontierWaitTicks, _unresolvedBehindWaitTicks;

    private GateItemKind NoteUnresolvedTake(Job job, out bool behind)
    {
        behind = Submitted > job.Index + 1;
        Interlocked.Increment(ref behind ? ref _unresolvedBehind : ref _unresolvedFrontier);
        if (job.Gate > Volatile.Read(ref _executedGate)) Interlocked.Increment(ref _unresolvedGated);
        else if (Volatile.Read(ref _shards[job.Shard].Running)) Interlocked.Increment(ref _unresolvedBusy);
        else Interlocked.Increment(ref _unresolvedQueued);
        var kind = GateKindOf(job.Gate);
        Interlocked.Increment(ref _unresolvedByGate[(int)kind]);
        if (job.Started)
        {
            Interlocked.Increment(ref _unresolvedStarted);
        }

        if (job.GateWaited)
        {
            Interlocked.Increment(ref _gateBoundByGate[(int)kind]);
        }

        return kind;
    }

    // The resolving thread's idle time in its queue wait and its wakes from it, its time spinning and sleeping on job
    // gates, the bytes it allocated per job (sampled every AllocationSamplePeriod jobs), and its CPU time.
    private long _idleTicks, _gateTicks, _idleWakes, _idleSpinTakes, _sampledBytes, _sampledJobs;

    // The sampled jobs' allocation by how their resolution was answered (ShaderPipelineCache.ResolvePath):
    // name=bytes per job/sampled jobs.
    private readonly long[] _sampledBytesByPath = new long[(int)SharpEmu.Libs.Gpu.Pipelines.ShaderPipelineCache.ResolvePath.Count];
    private readonly long[] _sampledJobsByPath = new long[(int)SharpEmu.Libs.Gpu.Pipelines.ShaderPipelineCache.ResolvePath.Count];

    // The sampled jobs' allocation by step of ResolveCore (alloc_by_step=, bytes per sampled job).
    private enum AllocationStep { Begin, Inputs, Programs, Assumptions, Bindings, Count }

    private static readonly string[] AllocationStepNames = ["begin", "inputs", "programs", "assumptions", "bindings"];
    private readonly long[] _sampledBytesByStep = new long[(int)AllocationStep.Count];

    [ThreadStatic]
    private static bool t_allocationSampled;

    // The thread's allocated bytes now, adding what was allocated since mark to step (sampled jobs only).
    private long MarkAllocation(AllocationStep step, long mark)
    {
        if (!t_allocationSampled)
        {
            return 0;
        }

        var now = GC.GetAllocatedBytesForCurrentThread();
        if (step != AllocationStep.Begin)
        {
            Interlocked.Add(ref _sampledBytesByStep[(int)step], now - mark);
        }

        return now;
    }

    private string TakeAllocationByStep(long sampledJobs)
    {
        var text = new System.Text.StringBuilder();
        for (var step = 1; step < (int)AllocationStep.Count; step++)
        {
            var bytes = Interlocked.Exchange(ref _sampledBytesByStep[step], 0);
            text.Append(step == 1 ? "" : ",").Append(AllocationStepNames[step]).Append('=')
                .Append((sampledJobs == 0 ? 0 : bytes / (double)sampledJobs).ToString("F0", System.Globalization.CultureInfo.InvariantCulture));
        }

        return text.ToString();
    }

    private string TakeAllocationByPath()
    {
        var text = new System.Text.StringBuilder();
        for (var path = 0; path < _sampledBytesByPath.Length; path++)
        {
            var bytes = Interlocked.Exchange(ref _sampledBytesByPath[path], 0);
            var jobs = Interlocked.Exchange(ref _sampledJobsByPath[path], 0);
            text.Append(path == 0 ? "" : ",").Append(SharpEmu.Libs.Gpu.Pipelines.ShaderPipelineCache.ResolvePathNames[path]).Append('=')
                .Append(jobs == 0 ? "0" : (bytes / (double)jobs).ToString("F0", System.Globalization.CultureInfo.InvariantCulture)).Append('/').Append(jobs);
        }

        return text.ToString();
    }

    // SHARPEMU_RESOLVER_IDLE_SPIN=0: the idle worker sleeps at once, as before.
    private static readonly bool IdleSpin = SharpEmu.ShaderCompiler.PerfSwitches.Enabled("SHARPEMU_RESOLVER_IDLE_SPIN");
    private const int IdleSpins = 32;

    private bool WorkReadyHint(ResolverShard shard) =>
        Volatile.Read(ref _stopping) || (shard.Queue.Count != 0 && !Volatile.Read(ref shard.Running));
    private const int AllocationSamplePeriod = 64;

    public string TakeResolverReport()
    {
        var culture = System.Globalization.CultureInfo.InvariantCulture;
        static double Milliseconds(long ticks) => ticks * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
        var sampledJobs = Interlocked.Exchange(ref _sampledJobs, 0);
        var sampledBytes = Interlocked.Exchange(ref _sampledBytes, 0);
        // Per shard: its thread's CPU time since the last report, jobs resolved (its thread's and the caller's), and its
        // thread's resolving time; resolver_cpu_ms is the sum (-1 when a thread's cannot be read).
        var cpuMs = 0.0;
        var resolverJobs = 0L;
        var shardCpu = new double[_shards.Length];
        foreach (var shard in _shards)
        {
            var cpu = ResolverCpuNanoseconds(shard);
            var lastCpu = Interlocked.Exchange(ref shard.LastCpu, cpu);
            shardCpu[shard.Index] = cpu < 0 || lastCpu < 0 ? -1.0 : (cpu - lastCpu) / 1e6;
            cpuMs = cpuMs < 0 || shardCpu[shard.Index] < 0 ? -1.0 : cpuMs + shardCpu[shard.Index];
            resolverJobs += Interlocked.Exchange(ref shard.ResolverJobs, 0);
        }
        return $"takes_unresolved=(busy={Interlocked.Exchange(ref _unresolvedBusy, 0)},gated={Interlocked.Exchange(ref _unresolvedGated, 0)}," +
            $"queued={Interlocked.Exchange(ref _unresolvedQueued, 0)},started={Interlocked.Exchange(ref _unresolvedStarted, 0)}," +
            $"frontier={Interlocked.Exchange(ref _unresolvedFrontier, 0)},behind={Interlocked.Exchange(ref _unresolvedBehind, 0)}," +
            $"frontier_wait_ms={Milliseconds(Interlocked.Exchange(ref _unresolvedFrontierWaitTicks, 0)).ToString("F1", culture)}," +
            $"behind_wait_ms={Milliseconds(Interlocked.Exchange(ref _unresolvedBehindWaitTicks, 0)).ToString("F1", culture)}) " +
            $"unresolved_by_gate=({FormatByKind(_unresolvedByGate)}) gate_bound_by_gate=({FormatByKind(_gateBoundByGate)}) " +
            $"unresolved_wait_ms_by_gate=({FormatByKind(_unresolvedWaitTicksByGate, ticks => Milliseconds(ticks).ToString("F1", culture))}) " +
            $"draw_gate_raises=({FormatByKind(_drawGateRaises)}) {TakeDrawLookaheadReport()} resolver_jobs={resolverJobs} " +
            $"resolver_idle_ms={Milliseconds(Interlocked.Exchange(ref _idleTicks, 0)).ToString("F1", culture)} " +
            $"gate_sleep_ms={Milliseconds(Interlocked.Exchange(ref _gateTicks, 0)).ToString("F1", culture)} idle_wakes={Interlocked.Exchange(ref _idleWakes, 0)} " +
            $"idle_spin_takes={Interlocked.Exchange(ref _idleSpinTakes, 0)} " +
            $"alloc_b_per_job={(sampledJobs == 0 ? 0 : sampledBytes / (double)sampledJobs).ToString("F0", culture)} " +
            $"alloc_by_path=({TakeAllocationByPath()}) alloc_by_step=({TakeAllocationByStep(sampledJobs)}) " +
            $"resolver_cpu_ms={cpuMs.ToString("F1", culture)} {TakeShardReport(shardCpu)} " +
            SharpEmu.Libs.Gpu.Pipelines.ShaderPipelineCache.TakeGraphicsResolveReport();
    }

    // shards=(n=,resolved=a/b,resolve_ms=a/b,cpu_ms=a/b, then the pipeline cache's shared tier: its gate's takes and wait
    // from the sharded paths, the shard gates' wait, draws resolved through a shard's memo, nested resolutions).
    private string TakeShardReport(double[] shardCpu)
    {
        var culture = System.Globalization.CultureInfo.InvariantCulture;
        var resolved = new string[_shards.Length];
        var resolveMs = new string[_shards.Length];
        var cpu = new string[_shards.Length];
        foreach (var shard in _shards)
        {
            resolved[shard.Index] = Interlocked.Exchange(ref shard.Resolved, 0).ToString(culture);
            resolveMs[shard.Index] = (Interlocked.Exchange(ref shard.ResolveTicks, 0) * 1000.0 / System.Diagnostics.Stopwatch.Frequency).ToString("F1", culture);
            cpu[shard.Index] = shardCpu[shard.Index].ToString("F1", culture);
        }

        return $"shards=(n={_shards.Length},resolved={string.Join('/', resolved)},resolve_ms={string.Join('/', resolveMs)},cpu_ms={string.Join('/', cpu)}," +
            $"{SharpEmu.Libs.Gpu.Pipelines.ShaderPipelineCache.TakeShardReport().Replace(' ', ',')})";
    }

    // A resolving thread's CPU time (user and system), or -1 where it cannot be read.
    private static long ResolverCpuNanoseconds(ResolverShard shard)
    {
        var thread = Volatile.Read(ref shard.ThreadId);
        if (thread == 0 || !OperatingSystem.IsMacOS())
            return -1;
        Span<ulong> info = stackalloc ulong[ThreadInfoBytes / sizeof(ulong)];
        return ProcessThreadInfo(Environment.ProcessId, ThreadId64Info, thread, ref info[0], ThreadInfoBytes) == ThreadInfoBytes
            ? (long)(info[0] + info[1])
            : -1;
    }

    // PROC_PIDTHREADID64INFO fills a proc_threadinfo: user and system time first, 112 bytes in all.
    private const int ThreadId64Info = 15;
    private const int ThreadInfoBytes = 112;

    [System.Runtime.InteropServices.DllImport("libc", EntryPoint = "pthread_threadid_np")]
    private static extern int PthreadThreadId(nint thread, out ulong id);

    [System.Runtime.InteropServices.DllImport("libc", EntryPoint = "proc_pidinfo")]
    private static extern int ProcessThreadInfo(int pid, int flavor, ulong argument, ref ulong buffer, int size);

    // The job is finished with: nothing reads its assumptions again.
    private void ReleaseAssumptions(Job job)
    {
        if (job.Assumptions is { } assumptions)
        {
            job.Assumptions = null;
            _concurrent.ReleaseAssumptions(assumptions);
        }
    }

    private void WaitFor(Job job)
    {
        if (job.State != 0)
        {
            return;
        }

        using var profile = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ProgramPrefetchWait);
        var spinner = new SpinWait();
        for (var spin = 0; spin < 64 && job.State == 0; spin++)
        {
            spinner.SpinOnce(sleep1Threshold: -1);
        }

        if (job.State != 0)
        {
            return;
        }

        lock (_lock)
        {
            while (job.State == 0)
            {
                _waiters++;
                Monitor.Wait(_lock);
                _waiters--;
            }
        }
    }

    // Claim only the FIFO head, never bypassing a dependency gate or resolving concurrently against
    // shared pipeline caches. A head the worker claimed but has not started is the caller's too, whatever
    // the queue length: under the lock that is a worker waiting on the head's gate (or just woken from
    // it), so it has no lead on the draw, and the caller would otherwise wait for its wake and its
    // resolution. The worker checks its head again before it dequeues. Demon's Souls gates most draws on
    // the work before them and waited about 20 ms a frame here. SHARPEMU_PREFETCH_CLAIM_STEAL=0 keeps
    // the claim with the worker.
    internal bool TryResolveOnCaller(Job job)
    {
        var shard = _shards[job.Shard];
        if (!_allowCallerResolution || job.State != 0 || Volatile.Read(ref shard.Running))
            return false;
        lock (_lock)
        {
            var claimed = shard.WorkerClaimed;
            if (_stopping || _paused || shard.Running || job.State != 0 ||
                (claimed ? !ClaimSteal : _outstanding > MaximumCallerOutstanding) ||
                job.Gate > Volatile.Read(ref _executedGate) || !shard.Queue.TryPeek(out var head) || !ReferenceEquals(head, job))
                return false;
            shard.Queue.Dequeue();
            shard.Running = true;
            shard.Resolving = job;
            job.Started = true;
            if (claimed)
            {
                _claimSteals++;
            }
        }

        var speculative = GpuFatal.ThrowOnFatal;
        var started = System.Diagnostics.Stopwatch.GetTimestamp();
        GpuFatal.ThrowOnFatal = true;
        try
        {
            ResolveCore(job);
        }
        finally
        {
            GpuFatal.ThrowOnFatal = speculative;
            Interlocked.Add(ref _callerResolveTicks, System.Diagnostics.Stopwatch.GetTimestamp() - started);
            Interlocked.Increment(ref _callerResolves);
            FinishJob(shard, worker: false);
        }
        return true;
    }

    private long _callerResolves, _callerResolveTicks;

    internal static readonly bool ClaimSteal =
        Environment.GetEnvironmentVariable("SHARPEMU_PREFETCH_CLAIM_STEAL") != "0";

    // Caller resolutions of a head the worker had claimed, under _lock.
    private long _claimSteals;

    public long TakeClaimSteals()
    {
        lock (_lock)
        {
            var steals = _claimSteals;
            _claimSteals = 0;
            return steals;
        }
    }

    public (long Count, double Milliseconds) TakeCallerResolveCounters() =>
        (Interlocked.Exchange(ref _callerResolves, 0),
            Interlocked.Exchange(ref _callerResolveTicks, 0) * 1000.0 / System.Diagnostics.Stopwatch.Frequency);

    // Threads in a Monitor.Wait on _lock for a finished job, the queue to drain, the worker to stop, or the worker's own gate
    // or pause (under _lock). FinishJob wakes only when one waits: a pulse per job was a kernel call per draw for nobody.
    private int _waiters;

    private void FinishJob(ResolverShard shard, bool worker)
    {
        lock (_lock)
        {
            _outstanding--;
            shard.Running = false;
            shard.Resolving = null;
            Volatile.Write(ref _finished, FinishedPrefix());
            if (worker) shard.WorkerClaimed = false;
            // The shard's thread idle in Run waits for Running to fall after a caller's resolution as well as for jobs.
            if (_waiters != 0 || shard.IdleForJobs)
            {
                Monitor.PulseAll(_lock);
            }
        }
    }

    private void Run(ResolverShard shard)
    {
        GpuFatal.ThrowOnFatal = true;
        if (OperatingSystem.IsMacOS() && PthreadThreadId(0, out var threadId) == 0)
        {
            Volatile.Write(ref shard.ThreadId, threadId);
        }

        var queue = shard.Queue;
        while (true)
        {
            Job job;
            lock (_lock)
            {
                if ((queue.Count == 0 || shard.Running) && !_stopping)
                {
                    var idleStarted = System.Diagnostics.Stopwatch.GetTimestamp();
                    if (IdleSpin)
                    {
                        // The next draw is usually submitted within microseconds: spin a little before sleeping, which
                        // costs the submitter a kernel wake per job. The checks outside the lock are hints only.
                        Monitor.Exit(_lock);
                        try
                        {
                            var spinner = new SpinWait();
                            for (var spin = 0; spin < IdleSpins && !WorkReadyHint(shard); spin++)
                            {
                                spinner.SpinOnce(sleep1Threshold: -1);
                            }
                        }
                        finally
                        {
                            Monitor.Enter(_lock);
                        }

                        if ((queue.Count != 0 && !shard.Running) || _stopping)
                        {
                            _idleSpinTakes++;
                        }
                    }

                    while ((queue.Count == 0 || shard.Running) && !_stopping)
                    {
                        shard.IdleForJobs = true;
                        Monitor.Wait(_lock);
                        shard.IdleForJobs = false;
                        _idleWakes++;
                    }

                    Interlocked.Add(ref _idleTicks, System.Diagnostics.Stopwatch.GetTimestamp() - idleStarted);
                }

                if (_stopping)
                {
                    return;
                }

                job = queue.Peek();
                shard.WorkerClaimed = true;
                if (job.Gate > Volatile.Read(ref _executedGate))
                {
                    job.GateWaited = true;
                }

                var gateStarted = job.Gate > Volatile.Read(ref _executedGate) || _paused ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
                if (job.Gate > Volatile.Read(ref _executedGate))
                {
                    // The gate usually opens within a few items: spin before sleeping.
                    Monitor.Exit(_lock);
                    try
                    {
                        var spinner = new SpinWait();
                        for (var spin = 0; spin < 50 && job.Gate > Volatile.Read(ref _executedGate); spin++)
                        {
                            spinner.SpinOnce(sleep1Threshold: -1);
                        }
                    }
                    finally
                    {
                        Monitor.Enter(_lock);
                    }
                }

                while ((_paused || job.Gate > Volatile.Read(ref _executedGate)) && !_stopping)
                {
                    Interlocked.Exchange(ref shard.SleepingOnGate, _paused ? long.MaxValue : job.Gate);
                    if (!_paused && job.Gate <= Volatile.Read(ref _executedGate))
                    {
                        break;
                    }

                    shard.WaitingOnPause = _paused;
                    _waiters++;
                    Monitor.Wait(_lock, 50);
                    _waiters--;
                    shard.WaitingOnPause = false;
                }

                Volatile.Write(ref shard.SleepingOnGate, long.MaxValue);
                if (gateStarted != 0)
                {
                    Interlocked.Add(ref _gateTicks, System.Diagnostics.Stopwatch.GetTimestamp() - gateStarted);
                }

                if (_stopping)
                {
                    return;
                }

                // The caller took the head while this thread waited for its gate (TryResolveOnCaller).
                if (shard.Running || !queue.TryPeek(out var head) || !ReferenceEquals(head, job))
                {
                    shard.WorkerClaimed = false;
                    continue;
                }

                queue.Dequeue();
                shard.Running = true;
                shard.Resolving = job;
                job.Started = true;
            }

            var sampled = (Interlocked.Increment(ref shard.ResolverJobs) & (AllocationSamplePeriod - 1)) == 0;
            var allocatedBefore = sampled ? GC.GetAllocatedBytesForCurrentThread() : 0;
            if (sampled) SharpEmu.Libs.Gpu.Pipelines.ShaderPipelineCache.t_resolvePath = default;
            t_allocationSampled = sampled;
            try
            {
                Resolve(job, shard);
            }
            finally
            {
                if (sampled)
                {
                    var bytes = GC.GetAllocatedBytesForCurrentThread() - allocatedBefore;
                    Interlocked.Add(ref _sampledBytes, bytes);
                    Interlocked.Increment(ref _sampledJobs);
                    var path = (int)SharpEmu.Libs.Gpu.Pipelines.ShaderPipelineCache.t_resolvePath;
                    Interlocked.Add(ref _sampledBytesByPath[path], bytes);
                    Interlocked.Increment(ref _sampledJobsByPath[path]);
                }

                FinishJob(shard, worker: true);
            }
        }
    }

    private long _resolveTicks;

    // The resolving threads' busy time since the last call (each shard's is in its own count too, TakeShardReport).
    public double TakeResolveMilliseconds() =>
        Interlocked.Exchange(ref _resolveTicks, 0) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;

    private void Resolve(Job job, ResolverShard shard)
    {
        // resolve_ms only: sampled like the materialization timings (MaterializationTally.StartTiming).
        var started = SharpEmu.ShaderCompiler.Resources.MaterializationTally.StartTiming();
        try
        {
            ResolveCore(job);
        }
        finally
        {
            if (started != 0)
            {
                var ticks = SharpEmu.ShaderCompiler.Resources.MaterializationTally.ElapsedTicks(started);
                Interlocked.Add(ref _resolveTicks, ticks);
                Interlocked.Add(ref shard.ResolveTicks, ticks);
            }
        }
    }

    /// <summary>The index of the job this thread resolves now, or -1.</summary>
    [ThreadStatic]
    private static long _resolvingJob;

    public static long ResolvingJob => _resolvingJob - 1;

    private void ResolveCore(Job job)
    {
        // Read before the host's state: the items counted had run before anything the resolution reads.
        job.ExecutedAtStart = Volatile.Read(ref _executedGate);
        _concurrent.BeginConcurrentResolve();
        _resolvingJob = job.Index + 1;
        try
        {
            if (job.Compute is { } compute)
            {
                // Only a dispatch's job keeps its resolving time (dispatch_saved_ms); a draw's read the clock for nothing.
                var started = System.Diagnostics.Stopwatch.GetTimestamp();
                // Before the lookup, as for a draw: a take under another stamp is refused (ComputeRefusal.Replanned).
                job.ProgramState = _executor.ProgramState;
                var mark = MarkAllocation(AllocationStep.Begin, 0);
                job.ComputeResult = _executor.ResolveComputeProgramAhead(job.Banks, compute.DispatchInitiator, compute.GroupsX,
                    compute.GroupsY, compute.GroupsZ, compute.Indirect, out job.ComputeRefusal);
                mark = MarkAllocation(AllocationStep.Programs, mark);
                if (RecordsForVerify)
                {
                    job.ResolveState = _executor.DescribeProgramState();
                }

                job.Assumptions = _concurrent.EndConcurrentResolve();
                mark = MarkAllocation(AllocationStep.Assumptions, mark);
                // The dispatch's binding decode joins its resolution (IConcurrentResolveHost.PrepareBindingsAhead).
                if (job.ComputeResult is { Available: true, Consumed: false, ImageCopy: null } resolved)
                {
                    _concurrent.PrepareBindingsAhead(resolved.Input.Stage, dispatch: true);
                }

                MarkAllocation(AllocationStep.Bindings, mark);

                job.ResolveTicks = System.Diagnostics.Stopwatch.GetTimestamp() - started;
                Interlocked.Increment(ref _resolvedCount);
                Interlocked.Increment(ref _shards[job.Shard].Resolved);
                job.State = 1;
                return;
            }

            var drawMark = MarkAllocation(AllocationStep.Begin, 0);
            job.Inputs = RenderExecutor.ProgramInputsOf(job.Banks, _host.FormatSupport, WorkerScratch.Enabled ? _fatal : _host.Fatal, job.Indexed,
                job.StartInstanceUserData);
            drawMark = MarkAllocation(AllocationStep.Inputs, drawMark);
            job.ProgramState = _executor.ProgramState;
            job.Programs = _executor.ResolveGraphicsPrograms(job.Banks, in job.Inputs);
            drawMark = MarkAllocation(AllocationStep.Programs, drawMark);
            if (RecordsForVerify)
            {
                job.ResolveState = _executor.DescribeProgramState();
                job.ResolvedAt = System.Diagnostics.Stopwatch.GetTimestamp();
                job.CensusReads = CensusReadsOf(job.Programs);
                job.CensusRefusals = DescribeRefusedCleanReads(job.CensusReads);
            }

            job.Assumptions = _concurrent.EndConcurrentResolve();
            drawMark = MarkAllocation(AllocationStep.Assumptions, drawMark);
            if (job.Programs is { Available: true } programs)
            {
                PrepareGraphicsBindingsAhead(programs, job.Inputs.PixelActive);
            }

            MarkAllocation(AllocationStep.Bindings, drawMark);

            Interlocked.Increment(ref _resolvedCount);
            Interlocked.Increment(ref _shards[job.Shard].Resolved);
            job.State = 1;
        }
        catch (Exception exception)
        {
            // The render thread resolves the draw itself, and fails there if the error is real.
            job.Assumptions = _concurrent.EndConcurrentResolve();
            Interlocked.Increment(ref _abandonedCount);
            if (MaxAbandonReports > 0 && Interlocked.Increment(ref _abandonReports) <= MaxAbandonReports)
            {
                Console.Error.WriteLine($"[GPU][PREFETCH] abandoned: {exception.GetType().Name}: {exception.Message}");
            }

            job.State = 2;
        }
        finally
        {
            _resolvingJob = 0;
        }
    }

    // The vertex (or mesh) stage's and the active pixel stage's binding decode, after their programs resolved.
    private void PrepareGraphicsBindingsAhead(GraphicsPrograms programs, bool pixelActive)
    {
        if (programs.VertexInput.Stage.Program is not null)
        {
            _concurrent.PrepareBindingsAhead(programs.VertexInput.Stage, dispatch: false);
        }

        if (pixelActive && programs.PixelInput.Stage.Program is not null)
        {
            _concurrent.PrepareBindingsAhead(programs.PixelInput.Stage, dispatch: false);
        }
    }

    public void Dispose()
    {
        lock (_lock)
        {
            _stopping = true;
            Monitor.PulseAll(_lock);
        }
    }
}
