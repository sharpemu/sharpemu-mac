// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.Pipelines;

// Resolver shards (ResolveShards): each shard's memo under its own gate, the shared tier under one lock taken on a miss.
public sealed partial class KeptStageInputsTests
{
    [Fact]
    public void ResolveShards_TwoThreadsOnATwoShardCacheHandOutWhatOneThreadDoes()
    {
        var sharded = MemoCache(ResolveMemoMode.Verify);
        sharded.ResolveShardCount = 2;
        Assert.Equal(2, sharded.ResolveShardCount);
        UseLoadingPixelProgram(Descriptor);
        var draws = new List<(RegisterBanks Banks, bool Pixel)>();
        foreach (var descriptor in new[] { Descriptor, OtherDescriptor })
        {
            PointPixelAt(descriptor);
            draws.Add((_banks.Clone(), true));
            draws.Add((_banks.Clone(), false));
        }

        // One thread first (program handles belong to the cache that made them, so the reference is this cache's own).
        var expected = draws.Select(draw => sharded.GetGraphicsPrograms(draw.Banks, [], draw.Pixel, depthBound: false)).ToArray();
        Assert.All(expected, programs => Assert.True(programs.Available));
        ShaderPipelineCache.TakeShardReport();
        Report();

        const int Rounds = 200;
        var failures = new System.Collections.Concurrent.ConcurrentQueue<string>();
        var threads = Enumerable.Range(0, 2).Select(thread => new Thread(() =>
        {
            try
            {
                for (var round = 0; round < Rounds; round++)
                {
                    for (var index = 0; index < draws.Count; index++)
                    {
                        // The two threads walk the draws in opposite orders, so they meet on both shards.
                        var at = thread == 0 ? index : draws.Count - 1 - index;
                        var programs = sharded.GetGraphicsPrograms(draws[at].Banks, [], draws[at].Pixel, depthBound: false);
                        if (GraphicsProgramsComparer.Difference(expected[at], programs) is { } difference)
                        {
                            failures.Enqueue($"draw {at}: {difference}");
                        }
                    }
                }
            }
            catch (Exception exception)
            {
                failures.Enqueue(exception.ToString());
            }
        })).ToArray();
        foreach (var thread in threads) thread.Start();
        foreach (var thread in threads) Assert.True(thread.Join(TimeSpan.FromSeconds(60)));

        Assert.Empty(failures);
        var report = Report();
        Assert.Contains("verify_failed=0)", report);
        var shards = ShaderPipelineCache.TakeShardReport();
        var shardedDraws = long.Parse(System.Text.RegularExpressions.Regex.Match(shards, @"sharded_draws=(\d+)").Groups[1].Value);
        Assert.True(shardedDraws >= 2 * Rounds * draws.Count, shards);
        Assert.Contains("nested_shared=0 ", shards);
        Assert.NotNull(sharded.Programs.MaterializationGate);
    }

    [Fact]
    public void ResolveShards_OneShardKeepsTheSharedTierAsTheOnlyGate()
    {
        var cache = MemoCache(ResolveMemoMode.On);
        cache.ResolveShardCount = 1;
        UseLoadingPixelProgram(Descriptor);
        Assert.Same(Programs(cache, pixelActive: true), Programs(cache, pixelActive: true));
        // No shard routing and no separate gate for the materialization cache: the one gate covers everything, as before.
        Assert.Equal(0, cache.ResolveShardOf(_banks, compute: false));
        Assert.Null(cache.Programs.MaterializationGate);
    }
}
