// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Reflection;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE;
using SharpEmu.Libs.Gpu;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

/// <summary>
/// The shard bench on the real engine path (SHARPEMU_DS_SHARD_BENCH_ENGINE=1): per arm one pipeline cache split into N
/// resolver shards (ShaderPipelineCache.ResolveShardCount) and one ParallelProgramPrefetch over it with its N resolving
/// threads, fed the captured jobs of a frame as the command thread feeds it (every gate open) and drained (WaitIdle);
/// the wall time from the first submission to idle is the resolvers' throughput. The jobs are then taken as the worker
/// takes them (outside the time), which hands their assumptions back, and the taken programs are fingerprinted by
/// content (program hash, shader base, every snapshot word): equal between arms means N shards hand out what one does.
/// Creations are counted per arm (program entries made, graphics and compute pipelines): equal between arms means no
/// key was created twice because two shards missed at once.
/// </summary>
public sealed partial class DsDrawCostBenchTests
{
    // Looked up here, not from the other parts' statics: the order partial classes' static fields initialize in is not fixed.
    private static readonly Type EngineCacheType =
        typeof(GraphicsPrograms).Assembly.GetType("SharpEmu.Libs.Gpu.Pipelines.ShaderPipelineCache", throwOnError: true)!;

    private static readonly PropertyInfo CacheShardCount =
        EngineCacheType.GetProperty("ResolveShardCount", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)!;

    private static readonly PropertyInfo CachePrograms =
        EngineCacheType.GetProperty("Programs", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)!;

    private static readonly Type ProgramCacheType =
        typeof(GraphicsPrograms).Assembly.GetType("SharpEmu.Libs.Gpu.Pipelines.ShaderProgramCache", throwOnError: true)!;

    private static readonly FieldInfo NextProgramId = ProgramCacheType.GetField("_nextProgramId", BindingFlags.Instance | BindingFlags.NonPublic)!;

    // A job's State (0 queued or resolving), read without the prefetch's lock: the drain below spins on it instead of
    // sleeping in WaitIdle, where every finished job would pulse the waiting bench thread awake.
    private static readonly System.Linq.Expressions.ParameterExpression JobParameter =
        System.Linq.Expressions.Expression.Parameter(typeof(ParallelProgramPrefetch.Job), "job");

    private static readonly Func<ParallelProgramPrefetch.Job, int> JobState = System.Linq.Expressions.Expression.Lambda<Func<ParallelProgramPrefetch.Job, int>>(
        System.Linq.Expressions.Expression.Field(JobParameter, "State"), JobParameter).Compile();

    private static ulong FingerprintStage(ulong hash, in ShaderStageResources stage)
    {
        static ulong Mix(ulong hash, ulong value) => (hash ^ value) * 0x100000001B3UL;
        static ulong MixWords(ulong hash, ReadOnlySpan<uint> words)
        {
            hash = Mix(hash, (ulong)words.Length);
            foreach (var word in words) hash = Mix(hash, word);
            return hash;
        }

        hash = Mix(hash, stage.Program?.Hash ?? 0);
        hash = Mix(hash, stage.ShaderBase);
        if (stage.Program is null) return hash;
        var snapshot = stage.Resources;
        foreach (var words in snapshot.Buffers) hash = MixWords(hash, words);
        foreach (var words in snapshot.Images) hash = MixWords(hash, words);
        foreach (var words in snapshot.Samplers) hash = MixWords(hash, words);
        hash = MixWords(hash, snapshot.FlattenedResourceTable);
        return MixWords(hash, snapshot.UserData);
    }

    private object RunEngineShardBench(SceneRig rig, object translation, string variation, CapturedJob[] jobs, Action<int> prepareFrame,
        Action finishFrame, int firstFrame)
    {
        int Setting(string name, int fallback) =>
            int.TryParse(Environment.GetEnvironmentVariable($"SHARPEMU_DS_SHARD_BENCH_{name}"), out var value) && value > 0 ? value : fallback;
        var frames = Setting("FRAMES", 8);
        var warmFrames = Setting("WARM_FRAMES", 4);
        var arms = (Environment.GetEnvironmentVariable("SHARPEMU_DS_SHARD_BENCH_ARMS") ?? "1,2,2,1,1,2,2,1")
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(int.Parse).ToArray();
        var liveContext = (CpuContext)TranslationContext.GetValue(translation)!;
        var renderHost = (IRenderHost)ExecutorHost.GetValue(TranslationExecutor.GetValue(translation))!;
        var host = rig.Host;
        var graphicsJobs = jobs.Count(job => job.Compute is null);
        Output.WriteLine($"engine shard bench: {variation}, {jobs.Length} jobs a frame ({graphicsJobs} draws, {jobs.Length - graphicsJobs} dispatches), " +
            $"{warmFrames} warm + {frames} measured frames per arm, arms {string.Join(",", arms)}, {Environment.ProcessorCount} cpus");

        // =spin: the bench thread spins on each job's state instead of sleeping in WaitIdle (whose waiter makes every finish
        // pulse the prefetch lock); the game's command thread never waits for the whole frame.
        var spinDrain = Environment.GetEnvironmentVariable("SHARPEMU_DS_SHARD_BENCH_DRAIN") == "spin";
        // =pipeline: as in the game, a taker thread (the worker) takes each job in order once it is resolved, inside the
        // timed part, and the submitter (the command thread) stays at most _WINDOW (64) jobs ahead of it; the takes hand
        // the assumptions back, so the resolvers draw from the host's pool as in game instead of allocating a frame's worth.
        var pipeline = Environment.GetEnvironmentVariable("SHARPEMU_DS_SHARD_BENCH_DRAIN") == "pipeline";
        var window = int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_DS_SHARD_BENCH_WINDOW"), out var parsedWindow) && parsedWindow > 0 ? parsedWindow : 64;
        var armResults = new List<(int Count, double JobsPerSecond, double WallMedianMs, string Report, ulong Fingerprint, string Creations,
            long Taken, long Refused)>();
        foreach (var count in arms)
        {
            var context = new CpuContext(liveContext.Memory, liveContext.TargetGeneration);
            var registry = CreateHeaderRegistry.Invoke(null, [context])!;
            var cache = (IShaderPipelineProvider)Activator.CreateInstance(PipelineCacheType, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
                null, [context, (IShaderPipelineHost)host, GuestGpu.Current, registry], null)!;
            CacheShardCount.SetValue(cache, count);
            var executor = new RenderExecutor(renderHost, cache);
            using var prefetch = new ParallelProgramPrefetch(executor, renderHost, (IConcurrentResolveHost)host);
            Assert.Contains($"shards=(n={count},", prefetch.TakeResolverReport());
            var submitted = new ParallelProgramPrefetch.Job[jobs.Length];
            var total = warmFrames + frames;
            var wall = new double[total];
            var took = new bool[jobs.Length];
            var takenStages = new (ShaderStageResources First, ShaderStageResources? Second)[jobs.Length];

            // The worker's take of job at; the stages it handed out are fingerprinted afterwards, untimed.
            bool Take(int at, int frameIndex)
            {
                var job = jobs[at];
                prefetch.Current = submitted[at];
                bool result;
                if (job.Compute is { } arguments)
                {
                    result = prefetch.TryTakeCompute(job.Banks, arguments.Initiator, arguments.GroupsX, arguments.GroupsY, arguments.GroupsZ,
                        arguments.Indirect, out var program);
                    takenStages[at] = result ? (program.Input.Stage, null) : default;
                }
                else
                {
                    var inputs = RenderExecutor.ProgramInputsOf(job.Banks, renderHost.FormatSupport, renderHost.Fatal, job.Inputs.Indexed,
                        job.Inputs.StartInstanceUserData);
                    result = prefetch.TryTakeGraphics(job.Banks, in inputs, out var programs);
                    takenStages[at] = result ? (programs.VertexInput.Stage, inputs.PixelActive ? programs.PixelInput.Stage : null) : default;
                }

                prefetch.Current = null;
                return result;
            }

            var fingerprint = 0xCBF29CE484222325UL;
            // Every arm replays the same frames (the churn and the pixel-constant copies follow the frame number), so the
            // fingerprints of all arms must match.
            var frame = firstFrame;
            long taken = 0, refused = 0, contentions = 0, collections = 0;
            var paused = TimeSpan.Zero;
            string report = string.Empty;
            for (var index = 0; index < total; index++)
            {
                prepareFrame(frame++);
                if (index == warmFrames)
                {
                    _ = prefetch.TakeResolverReport();
                    _ = prefetch.TakeCounters();
                    _ = prefetch.TakeResolveMilliseconds();
                }

                var contended = Monitor.LockContentionCount;
                var collected = GC.CollectionCount(0);
                var pausedBefore = GC.GetTotalPauseDuration();
                var frameIndex = index;
                var submittedCount = 0;
                var takenCount = 0;
                Thread? taker = null;
                if (pipeline)
                {
                    taker = new Thread(() =>
                    {
                        for (var at = 0; at < jobs.Length; at++)
                        {
                            while (Volatile.Read(ref submittedCount) <= at) Thread.SpinWait(8);
                            while (JobState(submitted[at]) == 0) Thread.SpinWait(8);
                            took[at] = Take(at, frameIndex);
                            Volatile.Write(ref takenCount, at + 1);
                        }
                    });
                    taker.Start();
                }

                var started = Stopwatch.GetTimestamp();
                for (var at = 0; at < jobs.Length; at++)
                {
                    if (pipeline)
                    {
                        while (at - Volatile.Read(ref takenCount) >= window) Thread.SpinWait(8);
                    }

                    var job = jobs[at];
                    submitted[at] = job.Compute is { } arguments
                        ? prefetch.SubmitCompute(job.Banks, 0, arguments.Initiator, arguments.GroupsX, arguments.GroupsY, arguments.GroupsZ, arguments.Indirect)
                        : prefetch.Submit(job.Banks, 0, job.Inputs.Indexed, job.Inputs.StartInstanceUserData);
                    Volatile.Write(ref submittedCount, at + 1);
                }

                taker?.Join();

                if (spinDrain)
                {
                    foreach (var job in submitted)
                    {
                        while (JobState(job) == 0) Thread.SpinWait(8);
                    }
                }

                prefetch.WaitIdle();
                wall[index] = (Stopwatch.GetTimestamp() - started) * 1000.0 / Stopwatch.Frequency;
                // Monitor contentions in the process over the timed part (the prefetch lock, the shard and shared gates).
                if (index >= warmFrames)
                {
                    contentions += Monitor.LockContentionCount - contended;
                    // Collections and GC pause time inside the timed part: the resolvers allocate, a pause stops every shard.
                    collections += GC.CollectionCount(0) - collected;
                    paused += GC.GetTotalPauseDuration() - pausedBefore;
                }

                // Without pipeline, the worker's takes, untimed: they hand the assumptions back. The measured frames' programs
                // are fingerprinted untimed either way, in stream order.
                for (var at = 0; at < jobs.Length; at++)
                {
                    if (!pipeline) took[at] = Take(at, index);
                    if (index >= warmFrames)
                    {
                        if (took[at]) taken++;
                        else refused++;
                    }
                }

                if (index >= warmFrames)
                {
                    for (var at = 0; at < jobs.Length; at++)
                    {
                        if (!took[at]) continue;
                        fingerprint = FingerprintStage(fingerprint, takenStages[at].First);
                        if (takenStages[at].Second is { } second) fingerprint = FingerprintStage(fingerprint, second);
                    }
                }

                prefetch.Current = null;
                finishFrame();
                if (index == total - 1)
                {
                    report = prefetch.TakeResolverReport();
                }
            }

            var programCache = CachePrograms.GetValue(cache)!;
            var creations = $"program_ids={NextProgramId.GetValue(programCache)} program_entries={ProgramCacheType.GetProperty("ProgramCount")!.GetValue(programCache)} " +
                $"graphics_pipelines={PipelineCacheType.GetProperty("GraphicsPipelineCount")!.GetValue(cache)} " +
                $"compute_pipelines={PipelineCacheType.GetProperty("ComputePipelineCount")!.GetValue(cache)}";
            var measured = wall[warmFrames..];
            var sorted = measured.Order().ToArray();
            var jobsPerSecond = frames * jobs.Length / (measured.Sum() / 1000.0);
            armResults.Add((count, jobsPerSecond, sorted[sorted.Length / 2], report, fingerprint, creations, taken, refused));
            var shards = System.Text.RegularExpressions.Regex.Match(report, @"shards=\(.*?rebind_shared_by=\([^)]*\)\)").Value;
            // The resolvers' allocation per job, sampled, and by how each job was answered.
            shards += " " + System.Text.RegularExpressions.Regex.Match(report, @"alloc_b_per_job=\d+ alloc_by_path=\([^)]*\) alloc_by_step=\([^)]*\)").Value;
            shards += " " + System.Text.RegularExpressions.Regex.Match(report, @"resolver_idle_ms=\S+ gate_sleep_ms=\S+ idle_wakes=\d+ idle_spin_takes=\d+").Value;
            Output.WriteLine($"engine N={count}: {jobsPerSecond / 1000.0:F0}k jobs/s, wall median {sorted[sorted.Length / 2]:F2} ms/frame (min {sorted[0]:F2}), " +
                $"taken {taken} refused {refused}, fingerprint {fingerprint:X16}, {creations}, " +
                $"monitor contentions {contentions / (double)(frames * jobs.Length):F3}/job, gc {collections} collections {paused.TotalMilliseconds:F1} ms paused, " +
                $"frames ms [{string.Join(" ", measured.Select(ms => ms.ToString("F1")))}]; last frame {shards}");
        }

        var byCount = armResults.GroupBy(arm => arm.Count).OrderBy(group => group.Key).Select(group =>
        {
            static double Median(IEnumerable<double> values)
            {
                var sorted = values.Order().ToArray();
                return sorted.Length % 2 == 1 ? sorted[sorted.Length / 2] : (sorted[sorted.Length / 2 - 1] + sorted[sorted.Length / 2]) / 2;
            }

            return new
            {
                shards = group.Key,
                arms = group.Count(),
                jobsPerSecond = Median(group.Select(arm => arm.JobsPerSecond)),
                jobsPerSecondByArm = group.Select(arm => Math.Round(arm.JobsPerSecond)).ToArray(),
                wallMedianMsPerFrame = Median(group.Select(arm => arm.WallMedianMs)),
                fingerprints = group.Select(arm => $"{arm.Fingerprint:X16}").Distinct().ToArray(),
                creations = group.Select(arm => arm.Creations).Distinct().ToArray(),
                taken = group.Select(arm => arm.Taken).ToArray(),
                refused = group.Select(arm => arm.Refused).ToArray(),
                lastReports = group.Select(arm => System.Text.RegularExpressions.Regex.Match(arm.Report, @"shards=\([^)]*\)").Value).ToArray(),
            };
        }).ToArray();
        var single = byCount.FirstOrDefault(row => row.shards == 1)?.jobsPerSecond ?? double.NaN;
        foreach (var row in byCount)
        {
            Output.WriteLine($"engine summary N={row.shards}: {row.jobsPerSecond / 1000.0:F0}k jobs/s median of {row.arms} arms ({row.jobsPerSecond / single:F2}x N=1), " +
                $"{row.wallMedianMsPerFrame:F2} ms/frame, fingerprints {string.Join("/", row.fingerprints)}, creations [{string.Join(" | ", row.creations)}]");
        }

        return new
        {
            variation = $"{variation}/engine-shards",
            jobs = jobs.Length,
            draws = graphicsJobs,
            frames,
            warmFrames,
            byCount,
            fingerprintsAgree = armResults.Select(arm => arm.Fingerprint).Distinct().Count() == 1,
            creationsAgree = armResults.Select(arm => arm.Creations).Distinct().Count() == 1,
        };
    }
}
