// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Reflection;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE;
using SharpEmu.HLE.Diagnostics;
using SharpEmu.HLE.Host;
using SharpEmu.Libs.Agc;
using SharpEmu.Libs.Gpu;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

/// <summary>
/// Does program resolution scale as N independent shards on N threads? Each shard is a ShaderPipelineCache (with its
/// program, materialization and resolve-memo caches and its _resolveGate) and a RenderExecutor over it, built as
/// CommandStreamTranslation builds the worker's, on a CpuContext of its own; a job goes to the shard a stable hash of
/// its vertex and pixel (or compute) program addresses names. The jobs are the program takes of one worker round,
/// replayed as ParallelProgramPrefetch.ResolveCore runs them (inputs, resolution inside the host's concurrent-read
/// scope, binding packets), with the round's descriptor-word and user-data churn applied between frames. What the
/// shards still share is the one Metal host (its concurrent-read ownership tests, assumption pool and packet pool),
/// the guest memory and the process-wide statics of the shader compiler and materializer.
/// </summary>
public sealed partial class DsDrawCostBenchTests
{
    private static readonly FieldInfo TranslationField =
        typeof(MetalRenderHost).GetField("_translation", BindingFlags.Instance | BindingFlags.NonPublic)!;

    private static readonly FieldInfo TranslationExecutor =
        typeof(AgcExports.CommandStreamTranslation).GetField("_executor", BindingFlags.Instance | BindingFlags.NonPublic)!;

    private static readonly FieldInfo TranslationContext =
        typeof(AgcExports.CommandStreamTranslation).GetField("_context", BindingFlags.Instance | BindingFlags.NonPublic)!;

    private static readonly FieldInfo ExecutorHost =
        typeof(RenderExecutor).GetField("_host", BindingFlags.Instance | BindingFlags.NonPublic)!;

    private static readonly Type PipelineCacheType =
        typeof(GraphicsPrograms).Assembly.GetType("SharpEmu.Libs.Gpu.Pipelines.ShaderPipelineCache", throwOnError: true)!;

    private static readonly MethodInfo CreateHeaderRegistry =
        typeof(AgcExports).GetMethod("CreateShaderHeaderRegistry", BindingFlags.Static | BindingFlags.NonPublic)!;

    // The resolve memo's per-thread counters (GraphicsResolveMemo), read on the shard's own thread.
    private static readonly FieldInfo CounterLocal =
        typeof(ThreadCounters).GetField("_local", BindingFlags.Static | BindingFlags.NonPublic)!;

    private static readonly FieldInfo CounterId =
        typeof(ThreadCounter).GetField("_id", BindingFlags.Instance | BindingFlags.NonPublic)!;

    private static readonly Type MemoType =
        typeof(GraphicsPrograms).Assembly.GetType("SharpEmu.Libs.Gpu.Pipelines.GraphicsResolveMemo", throwOnError: true)!;

    private static int[] MemoCounterIds(string field)
    {
        var value = MemoType.GetField(field, BindingFlags.Static | BindingFlags.NonPublic)!.GetValue(null)!;
        return value is Array counters
            ? counters.Cast<object>().Select(counter => (int)CounterId.GetValue(counter)!).ToArray()
            : [(int)CounterId.GetValue(value)!];
    }

    // hits, miss_by new, miss_by words, words_rebuilt, every miss, every uncacheable.
    private const int MemoHits = 0, MemoNew = 1, MemoWords = 2, MemoRebuilt = 3, MemoMisses = 4, MemoUncacheable = 5, MemoCounts = 6;

    private static readonly int[][] MemoIds =
    [
        MemoCounterIds("_hits"),
        [MemoCounterIds("_misses")[0]],
        [MemoCounterIds("_misses")[6]],
        MemoCounterIds("_rebuilt"),
        MemoCounterIds("_misses"),
        MemoCounterIds("_uncacheable"),
    ];

    private static void ReadMemoCounters(Span<long> into)
    {
        var local = (long[]?)CounterLocal.GetValue(null);
        for (var index = 0; index < MemoCounts; index++)
        {
            var total = 0L;
            if (local is not null)
            {
                foreach (var id in MemoIds[index]) total += local[id];
            }

            into[index] = total;
        }
    }

    private sealed record ComputeArguments(uint Initiator, uint GroupsX, uint GroupsY, uint GroupsZ, bool Indirect);

    private sealed class CapturedJob(RegisterBanks banks, GraphicsProgramInputs inputs, ComputeArguments? compute)
    {
        public readonly RegisterBanks Banks = banks;
        public readonly GraphicsProgramInputs Inputs = inputs;
        public readonly ComputeArguments? Compute = compute;
        // The pixel user-data word holding the per-round pixel constants address (GameWords' new keys), or -1.
        public int KeyWord = -1;
    }

    // The worker's program takes, in order, while Jobs is set; everything else goes to the resolver it wraps.
    private sealed class CapturingPrefetch(IProgramPrefetch inner) : IProgramPrefetch
    {
        public readonly IProgramPrefetch Inner = inner;
        public List<CapturedJob>? Jobs;

        public bool TryTakeGraphics(RegisterBanks banks, in GraphicsProgramInputs inputs, out GraphicsPrograms programs)
        {
            Jobs?.Add(new CapturedJob(banks.Clone(), inputs, null));
            return Inner.TryTakeGraphics(banks, in inputs, out programs);
        }

        public bool TryTakeCompute(RegisterBanks banks, uint dispatchInitiator, uint groupsX, uint groupsY, uint groupsZ, bool indirect,
            out ComputeProgram program)
        {
            Jobs?.Add(new CapturedJob(banks.Clone(), GraphicsProgramInputs.Create(),
                new ComputeArguments(dispatchInitiator, groupsX, groupsY, groupsZ, indirect)));
            return Inner.TryTakeCompute(banks, dispatchInitiator, groupsX, groupsY, groupsZ, indirect, out program);
        }

        public void NoteDispatchEffect() => Inner.NoteDispatchEffect();

        public void NoteDispatchProgram(ComputeProgram program) => Inner.NoteDispatchProgram(program);
    }

    // Marks the draws whose pixel user data carries the captured round's pixel constants copy (low address word), and
    // with newEvery one draw in that many more of those that carry the plain pixel constants.
    private static CapturedJob[] CapturedJobs(List<CapturedJob> jobs, uint keyLow, uint plainLow, int newEvery)
    {
        var draw = 0;
        foreach (var job in jobs.Where(job => job.Compute is null))
        {
            var scalars = job.Banks.Shader.Pixel.UserScalars;
            var values = scalars.ValueSpan((int)scalars.Count);
            var low = newEvery != 0 && draw++ % newEvery == newEvery / 2 && values.IndexOf(keyLow) < 0 ? plainLow : keyLow;
            var at = values.IndexOf(low);
            if (at >= 0 && values[(at + 1)..].IndexOf(low) < 0) job.KeyWord = at;
        }

        return [.. jobs];
    }

    private static ulong Mix(ulong value)
    {
        value ^= value >> 33;
        value *= 0xFF51AFD7ED558CCDUL;
        value ^= value >> 33;
        value *= 0xC4CEB53FE85A87DUL;
        value ^= value >> 33;
        return value;
    }

    // SHARPEMU_DS_SHARD_BENCH_ROUTE=pair (default): by the program pair; =index: by a hash of the job's index (balanced
    // to a few percent, and stable across frames here only because every frame replays the same stream).
    private static int ShardOf(CapturedJob job, int index, int shards, bool byIndex)
    {
        if (byIndex) return (int)(Mix((ulong)index + 1) % (ulong)shards);
        var shader = job.Banks.Shader;
        var identity = job.Compute is not null
            ? Mix(shader.Compute.Address ^ 0x5A5A5A5A5A5A5A5AUL)
            : Mix(Mix(shader.Vertex.ExportAddress) ^ shader.Pixel.Address);
        return (int)(identity % (ulong)shards);
    }

    private sealed class Shard
    {
        public required RenderExecutor Executor;
        public required CapturedJob[] Jobs;
        public int Graphics;
        public long[] CpuNanoseconds = [];
        public long[] WallTicks = [];
        public long[][] Memo = [];
        public long[] Abandoned = [];
        public Exception? Failure;
    }

    private static void ResolveJob(MetalRenderHost host, IRenderHost renderHost, RenderExecutor executor, CapturedJob job, ref long abandoned)
    {
        IConcurrentResolveHost concurrent = host;
        GraphicsPrograms? programs = null;
        ComputeProgram? compute = null;
        var pixelActive = false;
        host.BeginConcurrentResolve();
        try
        {
            if (job.Compute is { } arguments)
            {
                compute = executor.ResolveComputeProgramAhead(job.Banks, arguments.Initiator, arguments.GroupsX, arguments.GroupsY,
                    arguments.GroupsZ, arguments.Indirect, out _);
            }
            else
            {
                var inputs = RenderExecutor.ProgramInputsOf(job.Banks, renderHost.FormatSupport, renderHost.Fatal, job.Inputs.Indexed,
                    job.Inputs.StartInstanceUserData);
                pixelActive = inputs.PixelActive;
                programs = executor.ResolveGraphicsPrograms(job.Banks, in inputs);
            }
        }
        catch (Exception)
        {
            // As ParallelProgramPrefetch: the worker would resolve it itself.
            abandoned++;
        }

        var assumptions = host.EndConcurrentResolve();
        if (compute is { Available: true, Consumed: false, ImageCopy: null })
        {
            concurrent.PrepareBindingsAhead(compute.Input.Stage, dispatch: true);
        }
        else if (programs is { Available: true })
        {
            if (programs.VertexInput.Stage.Program is not null) concurrent.PrepareBindingsAhead(programs.VertexInput.Stage, dispatch: false);
            if (pixelActive && programs.PixelInput.Stage.Program is not null) concurrent.PrepareBindingsAhead(programs.PixelInput.Stage, dispatch: false);
        }

        host.ReleaseConcurrentAssumptions(assumptions);
    }

    private object RunShardBench(SceneRig rig, object translation, string variation, CapturedJob[] jobs, Action<int> prepareFrame, Action finishFrame,
        int firstFrame)
    {
        int Setting(string name, int fallback) =>
            int.TryParse(Environment.GetEnvironmentVariable($"SHARPEMU_DS_SHARD_BENCH_{name}"), out var value) && value > 0 ? value : fallback;
        var frames = Setting("FRAMES", 8);
        var warmFrames = Setting("WARM_FRAMES", 4);
        var byIndex = Environment.GetEnvironmentVariable("SHARPEMU_DS_SHARD_BENCH_ROUTE") == "index";
        var arms = (Environment.GetEnvironmentVariable("SHARPEMU_DS_SHARD_BENCH_ARMS") ?? "1,2,3,3,2,1,1,2,3,3,2,1")
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(int.Parse).ToArray();
        var liveContext = (CpuContext)TranslationContext.GetValue(translation)!;
        var renderHost = (IRenderHost)ExecutorHost.GetValue(TranslationExecutor.GetValue(translation))!;
        var host = rig.Host;
        var graphicsJobs = jobs.Count(job => job.Compute is null);
        var keyJobs = jobs.Count(job => job.KeyWord >= 0);
        Output.WriteLine($"shard bench: {variation}, {jobs.Length} jobs a frame ({graphicsJobs} draws, {jobs.Length - graphicsJobs} dispatches, " +
            $"{keyJobs} with per-frame pixel constants), route {(byIndex ? "index" : "pair")}, {warmFrames} warm + {frames} measured frames per arm, " +
            $"arms {string.Join(",", arms)}, {Environment.ProcessorCount} cpus");

        // Built as CommandStreamTranslation builds the worker's: a pipeline cache over the same host and backend, its own
        // context and shader header registry (the registrations are process-wide), and an executor over it.
        RenderExecutor NewShardExecutor()
        {
            var context = new CpuContext(liveContext.Memory, liveContext.TargetGeneration);
            var registry = CreateHeaderRegistry.Invoke(null, [context])!;
            var cache = (IShaderPipelineProvider)Activator.CreateInstance(PipelineCacheType, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
                null, [context, (IShaderPipelineHost)host, GuestGpu.Current, registry], null)!;
            return new RenderExecutor(renderHost, cache);
        }

        var frame = firstFrame;
        var armResults = new List<(int Count, double JobsPerSecond, double WallMedianMs, Shard[] Shards, double[] Wall, long Contention, double BytesPerJob, int Gen0)>();
        foreach (var count in arms)
        {
            var routed = Enumerable.Range(0, count).Select(_ => new List<CapturedJob>()).ToArray();
            for (var index = 0; index < jobs.Length; index++) routed[ShardOf(jobs[index], index, count, byIndex)].Add(jobs[index]);
            var total = warmFrames + frames;
            var shards = routed.Select(list => new Shard
            {
                Executor = NewShardExecutor(),
                Jobs = [.. list],
                Graphics = list.Count(job => job.Compute is null),
                CpuNanoseconds = new long[total],
                WallTicks = new long[total],
                Memo = Enumerable.Range(0, total).Select(_ => new long[MemoCounts]).ToArray(),
                Abandoned = new long[total],
            }).ToArray();
            using var start = new Barrier(count + 1);
            using var done = new Barrier(count + 1);
            var slot = 0;
            var stop = false;
            var threads = shards.Select((shard, index) => HostHelperThread.Start($"SharpEmu shard bench {index}", 16 * 1024 * 1024, () =>
            {
                Span<long> before = stackalloc long[MemoCounts];
                Span<long> after = stackalloc long[MemoCounts];
                while (true)
                {
                    start.SignalAndWait();
                    if (Volatile.Read(ref stop)) return;
                    var at = Volatile.Read(ref slot);
                    try
                    {
                        ReadMemoCounters(before);
                        long abandoned = 0;
                        var cpuStarted = ThreadCpuNanoseconds();
                        var wallStarted = Stopwatch.GetTimestamp();
                        foreach (var job in shard.Jobs) ResolveJob(host, renderHost, shard.Executor, job, ref abandoned);
                        shard.WallTicks[at] = Stopwatch.GetTimestamp() - wallStarted;
                        shard.CpuNanoseconds[at] = (long)(ThreadCpuNanoseconds() - cpuStarted);
                        ReadMemoCounters(after);
                        for (var counter = 0; counter < MemoCounts; counter++) shard.Memo[at][counter] = after[counter] - before[counter];
                        shard.Abandoned[at] = abandoned;
                    }
                    catch (Exception exception)
                    {
                        shard.Failure ??= exception;
                    }

                    done.SignalAndWait();
                }
            })).ToArray();

            var wall = new double[total];
            var contentionBefore = 0L;
            var allocatedBefore = 0L;
            var gen0Before = 0;
            for (var index = 0; index < total; index++)
            {
                if (index == warmFrames)
                {
                    contentionBefore = Monitor.LockContentionCount;
                    allocatedBefore = GC.GetTotalAllocatedBytes(precise: false);
                    gen0Before = GC.CollectionCount(0);
                }

                prepareFrame(frame++);
                Volatile.Write(ref slot, index);
                var started = Stopwatch.GetTimestamp();
                start.SignalAndWait();
                Assert.True(done.SignalAndWait(TimeSpan.FromMinutes(2)), "A shard did not finish its frame.");
                wall[index] = (Stopwatch.GetTimestamp() - started) * 1000.0 / Stopwatch.Frequency;
                finishFrame();
            }

            var contention = Monitor.LockContentionCount - contentionBefore;
            var bytesPerJob = (GC.GetTotalAllocatedBytes(precise: false) - allocatedBefore) / (double)(frames * jobs.Length);
            var gen0 = GC.CollectionCount(0) - gen0Before;
            Volatile.Write(ref stop, true);
            start.SignalAndWait();
            foreach (var thread in threads) Assert.True(thread.Join(TimeSpan.FromSeconds(30)));
            foreach (var shard in shards)
            {
                if (shard.Failure is not null) throw new InvalidOperationException("A shard failed.", shard.Failure);
            }

            var measured = wall[warmFrames..];
            var sorted = measured.Order().ToArray();
            var jobsPerSecond = frames * jobs.Length / (measured.Sum() / 1000.0);
            armResults.Add((count, jobsPerSecond, sorted[sorted.Length / 2], shards, measured, contention, bytesPerJob, gen0));
            var line = new System.Text.StringBuilder((
                $"N={count}: {jobsPerSecond / 1000.0:F0}k jobs/s, wall median {sorted[sorted.Length / 2]:F2} ms/frame (min {sorted[0]:F2}), " +
                $"largest shard {shards.Max(shard => shard.Jobs.Length) / (double)jobs.Length:P1} of jobs, lock contention {contention}, " +
                $"{bytesPerJob:F0} B/job, gen0 {gen0};"));
            for (var index = 0; index < shards.Length; index++)
            {
                var shard = shards[index];
                var cpu = shard.CpuNanoseconds[warmFrames..].Sum() / 1000.0 / Math.Max(1, frames * shard.Jobs.Length);
                var shardWall = shard.WallTicks[warmFrames..].Sum() * 1e6 / Stopwatch.Frequency / Math.Max(1, frames * shard.Jobs.Length);
                long Memo(int counter) => shard.Memo[warmFrames..].Sum(values => values[counter]);
                var draws = Math.Max(1.0, (double)frames * shard.Graphics);
                line.Append((
                    $" [{index}: {shard.Jobs.Length} jobs ({shard.Graphics} draws), cpu {cpu:F2} us/job (own wall {shardWall:F2}), memo hit {Memo(MemoHits) / draws:P1} " +
                    $"words {Memo(MemoWords) / draws:P1} (rebuilt {Memo(MemoRebuilt) / draws:P1}) new {Memo(MemoNew) / draws:P1} " +
                    $"misses {Memo(MemoMisses) / draws:P1} uncacheable {Memo(MemoUncacheable) / draws:P1}, abandoned {shard.Abandoned[warmFrames..].Sum()}]"));
            }

            Output.WriteLine(line.ToString());
        }

        // Per N: the medians over its arms; the speed-up against N=1 from those medians.
        var byCount = armResults.GroupBy(arm => arm.Count).OrderBy(group => group.Key).Select(group =>
        {
            static double Median(IEnumerable<double> values)
            {
                var sorted = values.Order().ToArray();
                return sorted.Length % 2 == 1 ? sorted[sorted.Length / 2] : (sorted[sorted.Length / 2 - 1] + sorted[sorted.Length / 2]) / 2;
            }

            var shardRows = Enumerable.Range(0, group.Key).Select(index => new
            {
                shard = index,
                jobs = group.First().Shards[index].Jobs.Length,
                draws = group.First().Shards[index].Graphics,
                cpuMicrosecondsPerJob = Median(group.Select(arm => arm.Shards[index].CpuNanoseconds[warmFrames..].Sum() / 1000.0 /
                    Math.Max(1, frames * arm.Shards[index].Jobs.Length))),
                wallMicrosecondsPerJob = Median(group.Select(arm => arm.Shards[index].WallTicks[warmFrames..].Sum() * 1e6 / Stopwatch.Frequency /
                    Math.Max(1, frames * arm.Shards[index].Jobs.Length))),
                memoHitRate = Median(group.Select(arm => arm.Shards[index].Memo[warmFrames..].Sum(values => values[MemoHits]) /
                    Math.Max(1.0, (double)frames * arm.Shards[index].Graphics))),
                memoWordsRate = Median(group.Select(arm => arm.Shards[index].Memo[warmFrames..].Sum(values => values[MemoWords]) /
                    Math.Max(1.0, (double)frames * arm.Shards[index].Graphics))),
                memoNewRate = Median(group.Select(arm => arm.Shards[index].Memo[warmFrames..].Sum(values => values[MemoNew]) /
                    Math.Max(1.0, (double)frames * arm.Shards[index].Graphics))),
                abandoned = group.Sum(arm => arm.Shards[index].Abandoned[warmFrames..].Sum()),
            }).ToArray();
            return new
            {
                shards = group.Key,
                arms = group.Count(),
                jobsPerSecond = Median(group.Select(arm => arm.JobsPerSecond)),
                jobsPerSecondByArm = group.Select(arm => Math.Round(arm.JobsPerSecond)).ToArray(),
                wallMedianMsPerFrame = Median(group.Select(arm => arm.WallMedianMs)),
                largestShardShare = group.First().Shards.Max(shard => shard.Jobs.Length) / (double)jobs.Length,
                lockContention = group.Select(arm => arm.Contention).ToArray(),
                bytesPerJob = Median(group.Select(arm => arm.BytesPerJob)),
                gen0 = group.Sum(arm => arm.Gen0),
                perShard = shardRows,
            };
        }).ToArray();
        var single = byCount.FirstOrDefault(row => row.shards == 1)?.jobsPerSecond ?? double.NaN;
        foreach (var row in byCount)
        {
            Output.WriteLine((
                $"summary N={row.shards}: {row.jobsPerSecond / 1000.0:F0}k jobs/s median of {row.arms} arms ({row.jobsPerSecond / single:F2}x N=1), " +
                $"{row.wallMedianMsPerFrame:F2} ms/frame, largest shard {row.largestShardShare:P1}; per shard " +
                string.Join(" ", row.perShard.Select(shard =>
                    $"[{shard.shard}: {shard.jobs} jobs, cpu {shard.cpuMicrosecondsPerJob:F2} us/job (wall {shard.wallMicrosecondsPerJob:F2}), hit {shard.memoHitRate:P1} words {shard.memoWordsRate:P1} new {shard.memoNewRate:P1}]"))));
        }

        return new
        {
            variation = $"{variation}/shards",
            jobs = jobs.Length,
            draws = graphicsJobs,
            keyJobs,
            frames,
            warmFrames,
            route = byIndex ? "index" : "pair",
            byCount,
        };
    }
}
