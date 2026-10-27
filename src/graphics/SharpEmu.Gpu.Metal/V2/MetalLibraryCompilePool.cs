// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// The threads the runtime's shader modules compile their MSL libraries on, so the GPU
/// worker hands a library to the driver's front end and goes on with the draw's other
/// program instead of waiting for each in turn (MetalRenderHost.CreateShaderModule). A
/// front-end call blocks its thread for the whole compile, tens of milliseconds to
/// seconds, so the pool has its own threads rather than the .NET pool's. First come,
/// first served, but a job the worker needs before a pool thread took it runs on the
/// worker at once (<see cref="TryRunHere"/>) instead of waiting behind the others.
/// Everything here is the game's own work; the background prewarm compiles on its own
/// threads and gives way to it (MetalCompileScheduler).
/// SHARPEMU_METAL_V2_ASYNC_LIBRARIES=0 compiles on the caller again.
/// </summary>
internal sealed class MetalLibraryCompilePool
{
    internal static readonly bool Enabled =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_ASYNC_LIBRARIES") != "0";

    private static readonly Lazy<MetalLibraryCompilePool> _shared = new(() => new MetalLibraryCompilePool(ThreadCount));

    private readonly BlockingCollection<Job> _queue = new();

    /// <summary>One queued compile: it runs once, on a pool thread or on the caller that needs it.</summary>
    internal sealed class Job(Action work)
    {
        private const int Queued = 0;
        private const int Taken = 1;
        private const int Cancelled = 2;

        private readonly TaskCompletionSource _done = new(TaskCreationOptions.RunContinuationsAsynchronously);
        private int _state;

        public Task Task => _done.Task;

        // Whether no thread has taken it yet; a thread may take it the moment after.
        internal bool IsQueued => Volatile.Read(ref _state) == Queued;

        // A deferred module that no pipeline requested must not start compiling at
        // teardown. Racing a claimant is safe: only one transition leaves Queued.
        internal bool CancelIfQueued()
        {
            if (Interlocked.CompareExchange(ref _state, Cancelled, Queued) != Queued)
                return false;
            _done.SetCanceled();
            return true;
        }

        // True when this call ran the work; false when another thread has it (or had it).
        internal bool TryRun()
        {
            if (Interlocked.CompareExchange(ref _state, Taken, Queued) != Queued)
            {
                return false;
            }

            try
            {
                // A compile creates strings and functions through factories that autorelease
                // them, on a thread that may have no pool: each unit of work frees its own.
                using (MetalAutoreleasePool.Enter())
                {
                    work();
                }

                _done.SetResult();
            }
            catch (Exception exception)
            {
                _done.SetException(exception);
            }

            return true;
        }
    }

    internal MetalLibraryCompilePool(int threads)
    {
        for (var index = 0; index < threads; index++)
        {
            // The frame waits for these: below the worker, above the background prewarm.
            MetalCompileThread.Start($"SharpEmu Metal v2 library compile {index}", SharpEmu.HLE.Host.HostThreadQos.UserInitiated, Run);
        }
    }

    /// <summary>The process's pool, started on first use.</summary>
    public static MetalLibraryCompilePool Shared => _shared.Value;

    /// <summary>
    /// As many compiles in flight as the shader prewarm keeps, where the driver's front
    /// end ran Demon's Souls' computes fastest (see MetalShaderPrewarm.WorkerCount).
    /// </summary>
    internal static int ThreadCount => MetalShaderPrewarm.WorkerCount;

    /// <summary>Queues <paramref name="work"/>; the job's task ends with it, or with its exception.</summary>
    public Job Run(Action work)
    {
        var job = new Job(work);
        _queue.Add(job);
        return job;
    }

    /// <summary>
    /// Runs a job nobody has started on the calling thread, which needs it now: true when it
    /// ran here (the job is then done), false when a pool thread already has it.
    /// </summary>
    public bool TryRunHere(Job job) => job.TryRun();

    private void Run()
    {
        foreach (var job in _queue.GetConsumingEnumerable())
        {
            // A job the caller took for itself is skipped.
            job.TryRun();
        }
    }
}
