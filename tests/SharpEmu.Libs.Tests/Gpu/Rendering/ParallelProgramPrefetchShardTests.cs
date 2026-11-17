// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Reflection;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Tests.Gpu.Scheduling;
using Xunit;
using static SharpEmu.Libs.Tests.Gpu.Rendering.RenderExecutorFixtures;

namespace SharpEmu.Libs.Tests.Gpu.Rendering;

// Resolver shards (ResolveShards): one resolving thread and queue per shard, jobs finishing out of order between shards,
// and every wait for "the jobs before an item" still waiting for exactly those.
[Collection(SchedulingStateCollection.Name)]
public sealed class ParallelProgramPrefetchShardTests
{
    private const ulong ShardZeroShader = 0x0010_0000;
    private const ulong ShardOneShader = 0x0020_0000;
    private static readonly TimeSpan Long = TimeSpan.FromSeconds(10);

    private sealed class ConcurrentHost : IConcurrentResolveHost
    {
        internal int Active, MaximumActive, Begins;
        public bool AllowsCallerResolution => true;

        public void BeginConcurrentResolve()
        {
            var active = Interlocked.Increment(ref Active);
            int old;
            do { old = Volatile.Read(ref MaximumActive); }
            while (active > old && Interlocked.CompareExchange(ref MaximumActive, active, old) != old);
            Interlocked.Increment(ref Begins);
        }

        public object EndConcurrentResolve()
        {
            Interlocked.Decrement(ref Active);
            return this;
        }

        public bool IsStillCurrent(object assumptions) => true;
    }

    // A resolution of a draw with this vertex address stops until released, and says when it got there.
    private sealed class Hold
    {
        internal readonly ManualResetEventSlim Entered = new(false);
        internal readonly ManualResetEventSlim Release = new(false);
    }

    private sealed class Fixture : IDisposable
    {
        private readonly FatalScope _fatal = new();
        internal readonly RecordingRenderHost Host = new();
        internal readonly FakePipelineProvider Pipelines = new()
        {
            ResolveShardCount = 2,
            ShardOf = (banks, _) => banks.Shader.Vertex.ExportAddress == ShardOneShader ? 1 : 0,
        };
        internal readonly ConcurrentHost Concurrent = new();
        internal readonly ParallelProgramPrefetch Prefetch;
        internal readonly object QueueLock;
        private readonly System.Collections.Concurrent.ConcurrentDictionary<ulong, Hold> _holds = new();

        internal Fixture(bool callerResolution = false)
        {
            Pipelines.BeforeGraphicsResolve = vertex =>
            {
                if (_holds.TryGetValue(vertex.ExportAddress, out var hold))
                {
                    hold.Entered.Set();
                    Assert.True(hold.Release.Wait(Long), "A held resolution was never released.");
                }
            };
            Prefetch = new(new RenderExecutor(Host, Pipelines), Host, Concurrent, callerResolution);
            QueueLock = typeof(ParallelProgramPrefetch).GetField("_lock", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(Prefetch)!;
        }

        internal Hold HoldResolutionsOf(ulong vertexAddress) => _holds.GetOrAdd(vertexAddress, _ => new Hold());

        internal void Idle() => Assert.True(Task.Run(Prefetch.WaitIdle).Wait(Long), "The resolvers did not become idle.");

        public void Dispose()
        {
            foreach (var hold in _holds.Values) hold.Release.Set();
            Prefetch.Dispose();
            _fatal.Dispose();
        }
    }

    private static RegisterBanks BanksAt(ulong vertexAddress)
    {
        var banks = Banks();
        banks.Shader.Vertex.ExportAddress = vertexAddress;
        return banks;
    }

    [Fact]
    public void AWaiterForThePrefixDoesNotReturnWhileAnEarlierJobOnAnotherShardIsUnfinished()
    {
        using var f = new Fixture();
        Assert.Equal(2, f.Prefetch.ShardCount);
        var hold = f.HoldResolutionsOf(ShardZeroShader);
        var first = f.Prefetch.Submit(BanksAt(ShardZeroShader));
        var second = f.Prefetch.Submit(BanksAt(ShardOneShader));
        var third = f.Prefetch.Submit(BanksAt(ShardOneShader));
        Assert.Equal((0, 1, 1), (first.Shard, second.Shard, third.Shard));
        Assert.True(hold.Entered.Wait(Long));
        // The later jobs finish on their own shard while the first is still resolving: out of order.
        Assert.True(SpinWait.SpinUntil(() => second.State == 1 && third.State == 1, Long));
        Assert.Equal(0, first.State);

        // Two jobs finished, but not the first: no prefix past it is finished.
        var one = Task.Run(() => f.Prefetch.WaitFinished(1));
        var three = Task.Run(() => f.Prefetch.WaitFinished(3));
        Assert.False(one.Wait(TimeSpan.FromMilliseconds(250)));
        Assert.False(three.IsCompleted);
        f.Prefetch.WaitFinished(0);

        hold.Release.Set();
        Assert.True(one.Wait(Long) && three.Wait(Long));
        Assert.Equal(1, first.State);
        f.Idle();
        Assert.Equal(3, f.Prefetch.TakeCounters().Resolved);
    }

    [Fact]
    public void ALaterJobOnTheSameShardWaitsBehindItsHeadWhileTheOtherShardRuns()
    {
        using var f = new Fixture();
        var hold = f.HoldResolutionsOf(ShardOneShader);
        var held = f.Prefetch.Submit(BanksAt(ShardOneShader));
        Assert.True(hold.Entered.Wait(Long));
        var behindHeld = f.Prefetch.Submit(BanksAt(ShardOneShader));
        var otherShard = f.Prefetch.Submit(BanksAt(ShardZeroShader));
        Assert.True(SpinWait.SpinUntil(() => otherShard.State == 1, Long));
        // A shard resolves its own jobs in submission order: the one behind the held job has not begun.
        Assert.Equal((0, 0, false), (held.State, behindHeld.State, behindHeld.Started));
        hold.Release.Set();
        f.Idle();
        f.Prefetch.WaitFinished(3);
        Assert.Equal((1, 1), (held.State, behindHeld.State));
    }

    [Fact]
    public void PauseWaitsForTheRunningResolutionOfEveryShardAndHoldsAllOfThem()
    {
        using var f = new Fixture();
        var hold = f.HoldResolutionsOf(ShardOneShader);
        var running = f.Prefetch.Submit(BanksAt(ShardOneShader));
        Assert.True(hold.Entered.Wait(Long));
        var pause = Task.Run(f.Prefetch.Pause);
        Assert.False(pause.Wait(TimeSpan.FromMilliseconds(250)));
        hold.Release.Set();
        Assert.True(pause.Wait(Long));
        Assert.Equal(1, running.State);

        // Paused: jobs on both shards stay queued, none begins.
        var begins = Volatile.Read(ref f.Concurrent.Begins);
        var zero = f.Prefetch.Submit(BanksAt(ShardZeroShader));
        var one = f.Prefetch.Submit(BanksAt(ShardOneShader));
        Thread.Sleep(150);
        Assert.Equal((0, 0, begins), (zero.State, one.State, Volatile.Read(ref f.Concurrent.Begins)));

        f.Prefetch.Resume();
        f.Idle();
        Assert.Equal((1, 1), (zero.State, one.State));
        f.Prefetch.WaitFinished(3);
    }

    [Fact]
    public void TheCallerResolvesItsShardsHeadWhileAnotherShardResolvesButNeverAJobAShardIsRunning()
    {
        using var f = new Fixture(callerResolution: true);
        var hold = f.HoldResolutionsOf(ShardOneShader);
        var onShardOne = f.Prefetch.Submit(BanksAt(ShardOneShader));
        Assert.True(hold.Entered.Wait(Long));
        ParallelProgramPrefetch.Job onShardZero;
        lock (f.QueueLock)
        {
            // Shard zero's thread cannot take its job while the test holds the queue lock: the caller can.
            onShardZero = f.Prefetch.Submit(BanksAt(ShardZeroShader));
            Assert.False(f.Prefetch.TryResolveOnCaller(onShardOne));
            Assert.True(f.Prefetch.TryResolveOnCaller(onShardZero));
            Assert.Equal((1, 0), (onShardZero.State, onShardOne.State));
            // Both resolutions ran at once: the caller's on shard zero beside shard one's thread.
            Assert.Equal(2, f.Concurrent.MaximumActive);
        }

        // The finished prefix still stops at the first job, which shard one's thread holds.
        Assert.False(Task.Run(() => f.Prefetch.WaitFinished(1)).Wait(TimeSpan.FromMilliseconds(150)));
        hold.Release.Set();
        f.Idle();
        f.Prefetch.WaitFinished(2);
        Assert.Equal(1, onShardOne.State);
        Assert.Equal(2, f.Concurrent.Begins);
        Assert.Equal(1, f.Prefetch.TakeCallerResolveCounters().Count);
    }

    [Fact]
    public void AGatedJobOnOneShardDoesNotHoldAnOpenJobOnTheOther()
    {
        using var f = new Fixture();
        var gated = f.Prefetch.Submit(BanksAt(ShardZeroShader), gate: 1);
        var open = f.Prefetch.Submit(BanksAt(ShardOneShader), gate: 0);
        Assert.True(SpinWait.SpinUntil(() => open.State == 1, Long));
        Assert.Equal(0, gated.State);
        Assert.False(Task.Run(() => f.Prefetch.WaitFinished(1)).Wait(TimeSpan.FromMilliseconds(150)));
        f.Prefetch.AdvanceGate(1);
        f.Idle();
        f.Prefetch.WaitFinished(2);
        Assert.Equal(1, gated.State);
    }

    [Fact]
    public void ADrawRoutesByItsMemoKeyFieldsAlone()
    {
        static RegisterBanks Draw(uint word, uint beyond, bool pixel)
        {
            var banks = Banks(withPixel: pixel);
            var vertex = banks.Shader.Vertex;
            vertex.GeometryResource2.UserScalarCount = 2;
            vertex.GeometryUserScalars.Set(0, 0x1234, UserScalarKind.Unknown);
            vertex.GeometryUserScalars.Set(1, word, UserScalarKind.Unknown);
            // Past the declared count: not part of the key.
            vertex.GeometryUserScalars.Set(2, beyond, UserScalarKind.Unknown);
            return banks;
        }

        Assert.Equal(ResolveShards.DrawRoute(Draw(7, 1, true), true), ResolveShards.DrawRoute(Draw(7, 2, true), true));
        Assert.NotEqual(ResolveShards.DrawRoute(Draw(7, 1, true), true), ResolveShards.DrawRoute(Draw(8, 1, true), true));

        // With the pixel stage off, nothing of the pixel stage is in the key, nor in the route.
        var inactive = Draw(7, 1, true);
        var other = Draw(7, 1, true);
        other.Shader.Pixel.Address ^= 0x100;
        other.Shader.Pixel.UserScalars.Set(0, 99, UserScalarKind.Unknown);
        Assert.Equal(ResolveShards.DrawRoute(inactive, false), ResolveShards.DrawRoute(other, false));
        Assert.NotEqual(ResolveShards.DrawRoute(inactive, true), ResolveShards.DrawRoute(other, true));

        // The same key routes alike every time; keys that differ in user data spread over the shards.
        var counts = new int[3];
        for (var word = 0u; word < 3000; word++)
        {
            var shard = ResolveShards.ShardOf(ResolveShards.DrawRoute(Draw(word, 0, true), true), 3);
            Assert.Equal(shard, ResolveShards.ShardOf(ResolveShards.DrawRoute(Draw(word, word, true), true), 3));
            counts[shard]++;
        }

        Assert.All(counts, count => Assert.InRange(count, 850, 1150));
        Assert.Equal(0, ResolveShards.ShardOf(ResolveShards.DrawRoute(Draw(1, 0, true), true), 1));
    }

    [Theory]
    [InlineData(null, 2, false)]
    [InlineData("1", 1, false)]
    [InlineData("3", 3, false)]
    [InlineData("0", 1, true)]
    [InlineData("64", ResolveShards.Maximum, true)]
    [InlineData("two", 2, true)]
    public void TheShardCountIsClampedAndNamesWhatItRefused(string? value, int count, bool problem)
    {
        Assert.Equal(count, ResolveShards.Parse(value, out var refused));
        Assert.Equal(problem, refused is not null);
    }
}
