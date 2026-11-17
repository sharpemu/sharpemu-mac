// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text.RegularExpressions;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Tests.Gpu.Scheduling;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.Pipelines;

// Dispatches with resolver shards: a kept program holds or rebinds off the shared tier where it can, and every take of
// that tier is counted by reason (ShaderPipelineCache.TakeShardReport).
public sealed partial class ComputeProgramReuseTests
{
    [Fact]
    public void WithShardsADispatchHandsOutWhatOneResolverDoesAndCountsEverySharedTakeByReason()
    {
        using var fatal = new FatalScope();
        var (single, _) = CreateCache();
        single.ResolveShardCount = 1;
        var (sharded, _) = CreateCache();
        sharded.ResolveShardCount = 2;
        ShaderPipelineCache.TakeShardReport();

        var userData = new uint[] { 0, 0x1234, 0x1234, 0x5678 };
        var expected = new List<ComputeProgram>();
        var banks = Banks();
        foreach (var word in userData)
        {
            banks.Shader.Compute.UserScalars.Set(0, word, UserScalarKind.Unknown);
            expected.Add(single.GetComputeProgram(banks, 0x8001, 1, 1, 1));
        }

        Assert.Contains("shared_takes=0 ", ShaderPipelineCache.TakeShardReport());
        banks = Banks();
        for (var index = 0; index < userData.Length; index++)
        {
            banks.Shader.Compute.UserScalars.Set(0, userData[index], UserScalarKind.Unknown);
            var program = sharded.ResolveComputeProgramAhead(banks, 0x8001, 1, 1, 1, indirect: false, out var refusal);
            Assert.Equal(ComputePrefetchRefusal.None, refusal);
            Assert.NotNull(program);
            Assert.Equal(expected[index].Available, program!.Available);
            Assert.Equal(expected[index].Input.Stage.Resources.UserData.ToArray(), program.Input.Stage.Resources.UserData.ToArray());
            Assert.Equal(expected[index].Input.ThreadsX, program.Input.ThreadsX);
        }

        // The first resolves in full on the shared tier; the repeat is kept off it; each change of user data is a rebinding,
        // off the tier or counted there with its reason. Nothing else takes the shared tier.
        var report = ShaderPipelineCache.TakeShardReport();
        long Field(string pattern) => long.Parse(Regex.Match(report, pattern).Groups[1].Value);
        Assert.Equal(1, Field(@"compute_resolve=(\d+)"));
        Assert.Equal(1, Field(@"unshared_compute=\(kept=(\d+)"));
        var rebound = Field(@"rebound=(\d+)\)");
        var rebindShared = Field(@"shared_by=\([^)]*compute_rebind=(\d+)");
        Assert.Equal(2, rebound + rebindShared);
        var refusals = Regex.Match(report, @"rebind_shared_by=\(([^)]*)\)").Groups[1].Value.Split(',').Sum(field => long.Parse(field.Split('=')[1]));
        Assert.Equal(rebindShared, refusals);
        Assert.Equal(1 + rebindShared, Field(@"shared_takes=(\d+)"));
    }

    [Fact]
    public void AReadLogIsTheSettingThreadsOnly()
    {
        var (cache, _) = CreateCache();
        var log = new GuestReadLog();
        cache.Programs.ReadLog = log;
        try
        {
            Assert.Same(log, cache.Programs.ReadLog);
            Assert.True(cache.Programs.TracesLookups);
            GuestReadLog? seen = log;
            var traces = true;
            var other = new Thread(() =>
            {
                seen = cache.Programs.ReadLog;
                traces = cache.Programs.TracesLookups;
            });
            other.Start();
            other.Join();
            Assert.Null(seen);
            Assert.False(traces);
        }
        finally
        {
            cache.Programs.ReadLog = null;
        }
    }
}
