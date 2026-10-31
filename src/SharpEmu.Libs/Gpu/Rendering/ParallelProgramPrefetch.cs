// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Scheduling;
using WorkerScratch = SharpEmu.ShaderCompiler.WorkerScratch;
using SharpEmu.Libs.VideoOut;

namespace SharpEmu.Libs.Gpu.Rendering;

// Resolves the programs of deferred draws on a worker thread, in issue order, while the render
// thread interprets and records. The render thread takes a result for the draw it runs only when
// the resolution completed, used the inputs the draw derives and still matches the host's
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
        // The jobs submitted before this one.
        internal long Index;
        // 0 queued, 1 resolved, 2 abandoned.
        internal volatile int State;
    }

    private readonly RenderExecutor _executor;
    private readonly IRenderHost _host;
    // host.Fatal as one delegate: a method group over an interface makes a new one per use, per draw.
    private readonly Func<string, Exception> _fatal;
    private readonly IConcurrentResolveHost _concurrent;
    private readonly object _lock = new();
    private readonly Queue<Job> _queue = new();
    private readonly SharpEmu.HLE.Host.HostHelperThread _thread;
    private int _outstanding;
    private long _finished;
    private long _executedGate;
    // The gate the worker sleeps on; long.MaxValue while it does not.
    private long _sleepingOnGate = long.MaxValue;
    private bool _paused;
    // Set while the worker sleeps because of a pause, under _lock.
    private bool _waitingOnPause;
    private bool _running;
    // The worker owns the head even while it releases the lock to wait for its gate.
    private bool _workerClaimed;
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
        // The worker waits on this thread per draw: it runs at the worker's class, with the
        // worker's stack, since it resolves what the worker would resolve itself.
        _thread = SharpEmu.HLE.Host.HostHelperThread.Start("SharpEmu program resolve", 16 * 1024 * 1024, Run);
    }

    // The job of the draw the render thread runs now; set around each deferred draw.
    public Job? Current { get; set; }

    public long Submitted { get; private set; }

    // The worker waits for jobs or an active caller (read and written under _lock).
    private bool _idleForJobs;

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

    public (long Resolved, long Abandoned, long Taken, long Refused, double WaitMilliseconds) TakeCounters() =>
        (Interlocked.Exchange(ref _resolvedCount, 0), Interlocked.Exchange(ref _abandonedCount, 0),
            Interlocked.Exchange(ref _takenCount, 0), Interlocked.Exchange(ref _refusedCount, 0),
            Interlocked.Exchange(ref _waitTicks, 0) * 1000.0 / System.Diagnostics.Stopwatch.Frequency);

    // A gate above zero holds the job until AdvanceGate reaches it: with the command stream on its own
    // thread, draws are submitted ahead of the work before them, which may change what they read.
    public Job Submit(RegisterBanks banks, long gate = 0, bool indexed = false, int startInstanceUserData = -1)
    {
        var job = new Job(banks, gate, indexed, startInstanceUserData) { Index = Submitted };
        lock (_lock)
        {
            _queue.Enqueue(job);
            Submitted++;
            _outstanding++;
            // Only the worker asleep on an empty queue waits for a job: the others on this lock wait
            // for jobs to finish, which the worker signals. A pulse per draw was a wake for nothing
            // while the worker was busy resolving the draws before.
            if (_idleForJobs)
            {
                Monitor.PulseAll(_lock);
            }
        }

        return job;
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
                Monitor.Wait(_lock);
            }
        }
    }

    // The render thread ran this many non-draw items: jobs gated on them may resolve.
    public void AdvanceGate(long executed)
    {
        Interlocked.Exchange(ref _executedGate, executed);
        // Only a worker asleep on a gate this reaches needs a wake: one per item costs the render
        // thread a kernel call each.
        if (Volatile.Read(ref _sleepingOnGate) <= executed)
        {
            lock (_lock)
            {
                Monitor.PulseAll(_lock);
            }
        }
    }

    // Returns once the worker finished the first count jobs it was given, so a non-draw item runs
    // with no resolution of the draws before it still reading.
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
                Monitor.Wait(_lock);
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
            while (_running)
            {
                Monitor.Wait(_lock);
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
            if (_waitingOnPause)
            {
                Monitor.PulseAll(_lock);
            }
        }
    }

    public bool TryTakeGraphics(RegisterBanks banks, in GraphicsProgramInputs inputs, out GraphicsPrograms programs)
    {
        programs = null!;
        var job = Current;
        if (job is null || !ReferenceEquals(job.Banks, banks))
        {
            return false;
        }

        Current = null;
        // A completed job waits for nothing, so it reads no clock: almost every take in a
        // well-prefetched burst is one, and two clock reads per draw were measurable.
        if (job.State == 0)
        {
            var waitStarted = System.Diagnostics.Stopwatch.GetTimestamp();
            // Caller work is reported separately, not disguised as a shorter resolver wait.
            // Keep long queues out of the claim routine's lock/finally frame.
            if (!_allowCallerResolution || Volatile.Read(ref _outstanding) > MaximumCallerOutstanding || !TryResolveOnCaller(job))
            {
                WaitFor(job);
                Interlocked.Add(ref _waitTicks, System.Diagnostics.Stopwatch.GetTimestamp() - waitStarted);
            }
        }

        using var profile = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ProgramPrefetchValidation);
        if (job.State != 1)
        {
            ReleaseAssumptions(job);
            return false;
        }

        var current = job.Inputs.Equals(inputs) && _concurrent.IsStillCurrent(job.Assumptions!);
        ReleaseAssumptions(job);
        if (!current)
        {
            Interlocked.Increment(ref _refusedCount);
            return false;
        }

        Interlocked.Increment(ref _takenCount);
        programs = job.Programs!;
        return true;
    }

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
                Monitor.Wait(_lock);
            }
        }
    }

    // Claim only the untouched FIFO head. Never race a worker that already peeked at it,
    // bypass a dependency gate, or resolve concurrently against shared pipeline caches.
    internal bool TryResolveOnCaller(Job job)
    {
        if (!_allowCallerResolution || job.State != 0 || Volatile.Read(ref _outstanding) > MaximumCallerOutstanding ||
            Volatile.Read(ref _running) || Volatile.Read(ref _workerClaimed))
            return false;
        lock (_lock)
        {
            if (_stopping || _paused || _running || _workerClaimed || job.State != 0 || _outstanding > MaximumCallerOutstanding ||
                job.Gate > Volatile.Read(ref _executedGate) || !_queue.TryPeek(out var head) || !ReferenceEquals(head, job))
                return false;
            _queue.Dequeue();
            _running = true;
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
            FinishJob(worker: false);
        }
        return true;
    }

    private long _callerResolves, _callerResolveTicks;

    public (long Count, double Milliseconds) TakeCallerResolveCounters() =>
        (Interlocked.Exchange(ref _callerResolves, 0),
            Interlocked.Exchange(ref _callerResolveTicks, 0) * 1000.0 / System.Diagnostics.Stopwatch.Frequency);

    private void FinishJob(bool worker)
    {
        lock (_lock)
        {
            _outstanding--;
            _finished++;
            _running = false;
            if (worker) _workerClaimed = false;
            Monitor.PulseAll(_lock);
        }
    }

    private void Run()
    {
        GpuFatal.ThrowOnFatal = true;
        while (true)
        {
            Job job;
            lock (_lock)
            {
                while ((_queue.Count == 0 || _running) && !_stopping)
                {
                    _idleForJobs = true;
                    Monitor.Wait(_lock);
                    _idleForJobs = false;
                }

                if (_stopping)
                {
                    return;
                }

                job = _queue.Peek();
                _workerClaimed = true;
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
                    Interlocked.Exchange(ref _sleepingOnGate, _paused ? long.MaxValue : job.Gate);
                    if (!_paused && job.Gate <= Volatile.Read(ref _executedGate))
                    {
                        break;
                    }

                    _waitingOnPause = _paused;
                    Monitor.Wait(_lock, 50);
                    _waitingOnPause = false;
                }

                Volatile.Write(ref _sleepingOnGate, long.MaxValue);
                if (_stopping)
                {
                    return;
                }

                _queue.Dequeue();
                _running = true;
            }

            try
            {
                Resolve(job);
            }
            finally { FinishJob(worker: true); }
        }
    }

    private long _resolveTicks;

    // The resolving thread's busy time since the last call.
    public double TakeResolveMilliseconds() =>
        Interlocked.Exchange(ref _resolveTicks, 0) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;

    private void Resolve(Job job)
    {
        var started = System.Diagnostics.Stopwatch.GetTimestamp();
        try
        {
            ResolveCore(job);
        }
        finally
        {
            Interlocked.Add(ref _resolveTicks, System.Diagnostics.Stopwatch.GetTimestamp() - started);
        }
    }

    /// <summary>The index of the job this thread resolves now, or -1.</summary>
    [ThreadStatic]
    private static long _resolvingJob;

    public static long ResolvingJob => _resolvingJob - 1;

    private void ResolveCore(Job job)
    {
        _concurrent.BeginConcurrentResolve();
        _resolvingJob = job.Index + 1;
        try
        {
            job.Inputs = RenderExecutor.ProgramInputsOf(job.Banks, _host.FormatSupport, WorkerScratch.Enabled ? _fatal : _host.Fatal, job.Indexed,
                job.StartInstanceUserData);
            job.Programs = _executor.ResolveGraphicsPrograms(job.Banks, in job.Inputs);
            job.Assumptions = _concurrent.EndConcurrentResolve();
            Interlocked.Increment(ref _resolvedCount);
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

    public void Dispose()
    {
        lock (_lock)
        {
            _stopping = true;
            Monitor.PulseAll(_lock);
        }
    }
}
