// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;
using System.Diagnostics;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// Pipeline states built off the command-stream thread.
///
/// The driver turns a function into a GPU binary when a pipeline state is created,
/// and it does that on whichever thread asks. Asking from the GPU worker made the
/// guest's dispatch packet wait for it: measured on Demon's Souls' load phase,
/// DISPATCH_DIRECT was ninety-five per cent of the worker's packet time at eighty-
/// eight to four hundred and ninety-seven milliseconds each.
///
/// A request now starts a compile on the compiler's own threads and the caller is told
/// the pipeline is not ready: a dispatch suspends its packet and the command stream serves
/// its other queues, a draw that may be left out is left out (RenderExecutor), and the
/// compile's completion wakes the stream. Two requests for the same key share one compile;
/// a compute key is its program id and a graphics key its GraphicsPipelineKey, which never
/// compare equal. The threads are the compiler's own rather than the .NET pool's: a burst
/// queues hundreds of compiles, and a pool thread parked on each would starve the pool.
/// </summary>
internal sealed class MetalPipelineCompiler : IDisposable
{
    /// <summary>The states one compile made: the pipeline, a rectangle list's per index type,
    /// and the vertex layout report.</summary>
    internal sealed class Result(
        nint state, nint[]? rectangleStates = null, string? vertexLayout = null, Action<nint>? release = null)
    {
        public nint State { get; } = state;

        // The first is State itself.
        public nint[]? RectangleStates { get; } = rectangleStates;

        public string? VertexLayout { get; } = vertexLayout;

        // A result nobody took, at teardown or abandoned: every state it holds goes, through
        // the host's release when it has one (a state shared by pipeline entries is counted).
        public void Release()
        {
            if (State != 0)
            {
                ReleaseOne(State);
            }

            for (var index = 1; index < (RectangleStates?.Length ?? 0); index++)
            {
                if (RectangleStates![index] != 0)
                {
                    ReleaseOne(RectangleStates[index]);
                }
            }
        }

        private void ReleaseOne(nint handle)
        {
            if (release is not null)
            {
                release(handle);
            }
            else
            {
                MetalNative.SendVoid(handle, SelRelease);
            }
        }
    }

    internal sealed class Pending(Func<Result> build)
    {
        public readonly Func<Result> Build = build;
        public readonly TaskCompletionSource<Result> Done = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public long StartedTicks;
        public long CompileTicks;

        // What the compile builds, for a stalled queue's report (DescribeInFlight).
        public string? Label;
        public long FinishedTicks;

        // Set by whichever thread builds it: a compiler thread, or a caller that cannot wait
        // behind the queue (TryWait).
        public int Claimed;

        public Task<Result> Work => Done.Task;
    }

    private static readonly nint SelRelease = MetalNative.Selector("release");

    private readonly ConcurrentDictionary<object, Pending> _pending = new();
    private readonly BlockingCollection<Pending> _queue = new();
    private readonly Thread[] _threads;
    private readonly Action _onCompleted;
    private int _running;
    private long _started;
    private long _finished;
    private long _compileTicks;
    private long _deduped;
    private long _abandoned;
    private volatile bool _disposed;

    public MetalPipelineCompiler(Action onCompleted, int parallelism)
    {
        _onCompleted = onCompleted;
        _threads = new Thread[Math.Max(1, parallelism)];
        for (var index = 0; index < _threads.Length; index++)
        {
            _threads[index] = new Thread(Run)
            {
                IsBackground = true,
                Name = $"SharpEmu Metal v2 pipeline compile {index}",
            };
            _threads[index].Start();
        }
    }

    /// <summary>Held by tests to keep compiles in flight; null runs them at once.</summary>
    internal ManualResetEventSlim? Gate { get; set; }

    /// <summary>The compiles still unfinished, oldest first, with how long since each was
    /// queued: what a queue suspended on a pipeline is waiting behind.</summary>
    public string DescribeInFlight(int max = 4)
    {
        var now = Stopwatch.GetTimestamp();
        var waiting = _pending
            .Where(pair => !pair.Value.Work.IsCompleted)
            .OrderBy(pair => pair.Value.StartedTicks)
            .ToList();
        var shown = waiting.Take(max).Select(pair =>
            $"{pair.Value.Label ?? pair.Key.ToString()} queued_ms={(now - pair.Value.StartedTicks) * 1000.0 / Stopwatch.Frequency:F0}" +
            (pair.Value.CompileTicks == 0 && Volatile.Read(ref pair.Value.Claimed) != 0 ? " building" : string.Empty));
        return $"n={waiting.Count} " + string.Join("; ", shown);
    }

    /// <summary>How many compiles have been started, finished, and shared.</summary>
    public (long Started, long Finished, long Deduped, double Milliseconds) TakeCounters() =>
        (Interlocked.Exchange(ref _started, 0), Interlocked.Exchange(ref _finished, 0),
            Interlocked.Exchange(ref _deduped, 0),
            Interlocked.Exchange(ref _compileTicks, 0) * 1000.0 / Stopwatch.Frequency);

    /// <summary>
    /// Compiles queued or running right now: these read their modules' functions, so the
    /// shader cache bound releases no module while there are any. A finished one nobody has
    /// taken yet holds only its states and is not counted.
    /// </summary>
    public int InFlight => Volatile.Read(ref _running);

    /// <summary>Finished compiles whose result nobody took before it was dropped.</summary>
    public long Abandoned => Interlocked.Read(ref _abandoned);

    /// <summary>
    /// The finished pipeline states for this key, or null while they are still being built.
    /// <paramref name="build"/> runs on a compiler thread and must be safe to call from any
    /// thread; creating a pipeline state from a device and a function is. A compile finished
    /// out of order with the requests waits here for its own key, so the order in which
    /// compiles end does not matter.
    /// </summary>
    public bool TryTake(object key, Func<Result> build, out Result? result) => TryTake(key, build, out result, null);

    public bool TryTake(object key, Func<Result> build, out Result? result, string? label)
    {
        result = null;
        if (_disposed)
        {
            result = build();
            return true;
        }

        if (_pending.TryGetValue(key, out var pending))
        {
            if (!pending.Work.IsCompleted)
            {
                Interlocked.Increment(ref _deduped);
                return false;
            }

            // Dropped as abandoned between the look-up and here: it is started again.
            if (TryTakeFinished(key, pending, out result))
            {
                return true;
            }
        }

        var started = new Pending(build) { StartedTicks = Stopwatch.GetTimestamp(), Label = label };
        if (!_pending.TryAdd(key, started))
        {
            return false;
        }

        Interlocked.Increment(ref _started);
        Interlocked.Increment(ref _running);
        _queue.Add(started);
        return false;
    }

    /// <summary>
    /// The result of a compile already started for this key, for a request that may not be
    /// left out: built here and now when no compiler thread has started it (it would otherwise
    /// wait behind the whole queue), waited for when one has. False when there is none.
    /// </summary>
    public bool TryWait(object key, out Result? result)
    {
        result = null;
        if (!_pending.TryGetValue(key, out var pending))
        {
            return false;
        }

        if (Interlocked.CompareExchange(ref pending.Claimed, 1, 0) == 0)
        {
            Build(pending);
        }
        else
        {
            ((IAsyncResult)pending.Work).AsyncWaitHandle.WaitOne();
        }

        return TryTakeFinished(key, pending, out result);
    }

    // The finished result, if this caller removed it: one dropped as abandoned meanwhile was
    // released and is not handed out.
    private bool TryTakeFinished(object key, Pending pending, out Result? result)
    {
        result = null;
        if (!_pending.TryRemove(new KeyValuePair<object, Pending>(key, pending)))
        {
            return false;
        }

        Interlocked.Increment(ref _finished);
        Interlocked.Add(ref _compileTicks, pending.CompileTicks);
        // A failed compile is reported by the caller, which knows the shader.
        result = pending.Work.GetAwaiter().GetResult();
        return true;
    }

    /// <summary>
    /// Drops the results finished before <paramref name="finishedBefore"/> that nobody took:
    /// a draw left out while it compiled and never drawn again (its key changed) would keep
    /// them for ever. A later request for the key compiles again, a cache hit in the driver.
    /// </summary>
    public int DropAbandoned(long finishedBefore)
    {
        var dropped = 0;
        foreach (var (key, pending) in _pending)
        {
            if (!pending.Work.IsCompleted || Volatile.Read(ref pending.FinishedTicks) > finishedBefore ||
                !_pending.TryRemove(new KeyValuePair<object, Pending>(key, pending)))
            {
                continue;
            }

            if (pending.Work.IsCompletedSuccessfully)
            {
                pending.Work.Result.Release();
            }

            dropped++;
        }

        Interlocked.Add(ref _abandoned, dropped);
        return dropped;
    }

    private void Run()
    {
        foreach (var pending in _queue.GetConsumingEnumerable())
        {
            Gate?.Wait();
            // A caller that took it to build itself (TryWait) has it.
            if (Interlocked.CompareExchange(ref pending.Claimed, 1, 0) == 0)
            {
                Build(pending);
            }
        }
    }

    private void Build(Pending pending)
    {
        Result? result = null;
        Exception? failure = null;
        var compileStarted = Stopwatch.GetTimestamp();
        try
        {
            // Teardown: a compile nobody started is not started.
            if (!_disposed)
            {
                // A compiler thread has no autorelease pool, and building a pipeline
                // autoreleases strings and descriptors: the build frees its own. The states
                // it returns are owned (new*), not autoreleased, so they outlive the pool.
                using (MetalAutoreleasePool.Enter())
                {
                    result = pending.Build();
                }
            }
        }
        catch (Exception exception)
        {
            failure = exception;
        }

        pending.CompileTicks = Stopwatch.GetTimestamp() - compileStarted;
        Volatile.Write(ref pending.FinishedTicks, Stopwatch.GetTimestamp());
        Interlocked.Decrement(ref _running);
        if (failure is null)
        {
            pending.Done.SetResult(result ?? new Result(0));
        }
        else
        {
            pending.Done.SetException(failure);
        }

        // The command stream may be parked on this pipeline; wake it.
        try
        {
            _onCompleted();
        }
        catch
        {
            // Waking is best effort: the blocked-retry fallback still runs.
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        // Compiles already in the driver finish; queued ones are skipped; then every result
        // nobody took is released, before the device and the functions go.
        _disposed = true;
        _queue.CompleteAdding();
        Gate?.Set();
        foreach (var thread in _threads)
        {
            thread.Join();
        }

        foreach (var pending in _pending.Values)
        {
            if (pending.Work.IsCompletedSuccessfully)
            {
                pending.Work.Result.Release();
            }
        }

        _pending.Clear();
        _queue.Dispose();
    }
}
