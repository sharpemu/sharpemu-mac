// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections;
using System.Reflection;
using SharpEmu.Libs.Agc;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Tests.Gpu.GpuCommands;
using SharpEmu.Libs.Tests.Gpu.Scheduling;
using Xunit;
using static SharpEmu.Libs.Tests.Gpu.Rendering.RenderExecutorFixtures;

namespace SharpEmu.Libs.Tests.Gpu.Rendering;

[Collection(SchedulingStateCollection.Name)]
public sealed partial class LocalDeferredDrawReuseTests
{
    private sealed class ConcurrentHost : IConcurrentResolveHost
    {
        public void BeginConcurrentResolve() { }
        public object EndConcurrentResolve() => this;
        public bool IsStillCurrent(object assumptions) => ReferenceEquals(this, assumptions);
    }

    private sealed class Sink : AgcExports.IDeferredWorkSink
    {
        public bool IsProducerThread => true;
        public long Gate => 0;
        public readonly List<AgcExports.DeferredWork> Work = new();
        public long EnqueueDeferred(AgcExports.DeferredWork work, bool isDraw) { Work.Add(work); return 0; }
    }

    private sealed class Fixture : IDisposable
    {
        private readonly FatalScope _fatal = new();
        private static readonly FieldInfo ActiveHost = typeof(AgcExports).GetField("_activeCommandStreamHost", BindingFlags.Static | BindingFlags.NonPublic)!;
        private readonly object? _previousActiveHost = ActiveHost.GetValue(null);
        internal readonly RecordingCommandStreamHost Commands = new();
        internal readonly RecordingRenderHost Host = new();
        internal readonly FakePipelineProvider Pipelines = new();
        internal readonly GpuCommandInterpreter Interpreter;
        internal readonly AgcExports.CommandStreamTranslation Translation;
        internal readonly ParallelProgramPrefetch Prefetch;

        internal Fixture(bool reuse)
        {
            Interpreter = new(Commands, 0, 0);
            var banks = Banks();
            foreach (var name in new[] { nameof(RegisterBanks.Context), nameof(RegisterBanks.Shader), nameof(RegisterBanks.UserConfig) })
                typeof(RegisterBanks).GetProperty(name)!.SetValue(Interpreter.TypedRegisters,
                    typeof(RegisterBanks).GetProperty(name)!.GetValue(banks));
            var executor = new RenderExecutor(Host, Pipelines);
            Prefetch = new(executor, Host, new ConcurrentHost());
            executor.Prefetch = Prefetch;
            Translation = new(Commands.Memory, Commands, recycleLocalDraws: reuse);
            // Use real translation/executor/prefetch with the existing fake renderer, avoiding
            // native shader compilation. The production constructor composes its own provider.
            typeof(AgcExports.CommandStreamTranslation).GetField("_executor", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(Translation, executor);
            typeof(AgcExports.CommandStreamTranslation).GetField("_parallel", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(Translation, Prefetch);
            // A concurrent-resolve host (Metal) defers whole draws; the fake host is not one.
            typeof(AgcExports.CommandStreamTranslation).GetField("_defersWholeWork", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(Translation, true);
            Translation.BeginSubmission(0, 1, null, Interpreter);
            Host.WriteGuest(IndexBase, new byte[1024]);
        }

        internal void Draw(uint count, ulong submit = 1) => Translation.DrawAuto(submit, Auto(count));
        internal void Index(uint count, ulong submit = 1) => Translation.DrawIndexed(submit, Indexed(count));
        internal void Drain() { Prefetch.WaitIdle(); Translation.DrainPendingDraws(); }
        internal string[] Draws => Host.Calls.Where(x => x.StartsWith("draw ", StringComparison.Ordinal) || x.StartsWith("draw_indexed ", StringComparison.Ordinal)).ToArray();
        internal object[] Spares => ((IEnumerable)typeof(AgcExports.CommandStreamTranslation)
            .GetField("_spareLocalDraws", BindingFlags.Instance | BindingFlags.NonPublic)!
            .GetValue(Translation)!).Cast<object>().ToArray();
        internal void AssertCleared()
        {
            foreach (var spare in Spares)
            {
                foreach (var name in new[] { "State", "Banks", "VertexSnapshot", "IndexSnapshot", "Job" })
                    Assert.Null(spare.GetType().GetField(name)!.GetValue(spare));
                Assert.Equal(0UL, spare.GetType().GetField("SubmitId")!.GetValue(spare));
                Assert.Equal(default(DrawAutoArguments), spare.GetType().GetField("AutoArguments")!.GetValue(spare));
                Assert.Equal(default(DrawIndexedArguments), spare.GetType().GetField("IndexedArguments")!.GetValue(spare));
            }
        }
        public void Dispose()
        {
            try { Prefetch.Dispose(); }
            finally { ActiveHost.SetValue(null, _previousActiveHost); _fatal.Dispose(); }
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RepeatedBurstsPreserveArgumentsSnapshotsAndNestedDrainOrder(bool reuse)
    {
        using var f = new Fixture(reuse);
        var expected = new List<string>();
        for (var round = 0; round < 6; round++)
        {
            for (var i = 0; i < 12; i++)
            {
                var count = (uint)(3 + round * 12 + i);
                var banks = f.Interpreter.TypedRegisters;
                banks.PrepareUserConfigWrite();
                banks.UserConfig.PrimitiveType = PrimitiveTriangleList;
                banks.UserConfig.ObjectId = count;
                if ((i & 1) == 0)
                {
                    f.Draw(count, count);
                    expected.Add($"draw {count} 1 0 0");
                }
                else
                {
                    f.Index(count, count);
                    expected.Add($"draw_indexed {count} 1 0 0 0");
                }
            }
            var queued = ((IEnumerable)typeof(AgcExports.CommandStreamTranslation)
                .GetField("_pendingDraws", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(f.Translation)!).Cast<object>().ToArray();
            Assert.Equal(Enumerable.Range(3 + round * 12, 12).Select(x => (uint)x), queued.Select(x =>
                ((RegisterBanks)x.GetType().GetField("Banks")!.GetValue(x)!).UserConfig.ObjectId));
            f.Translation.EnqueuePendingAction(() =>
            {
                Assert.True(f.Translation.IsDraining);
                f.Translation.DrainPendingDraws(); // Must not execute the following draw recursively.
                Assert.Equal(expected.ToArray(), f.Draws);
            });
            f.Draw(1000);
            f.Prefetch.WaitIdle();
            f.Translation.DrainPendingDraws();
            expected.Add("draw 1000 1 0 0");
            Assert.Equal(expected.ToArray(), f.Draws);
            Assert.False(f.Translation.HasPendingDraws);
            Assert.Null(f.Prefetch.Current);
            f.AssertCleared();
        }
        Assert.Equal(reuse ? 13 : 0, f.Spares.Length);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FailureReturnsOnlyTheFinishedWrapperAndLeavesLaterDrawIntact(bool reuse)
    {
        using var f = new Fixture(reuse);
        f.Draw(3); f.Draw(6);
        f.Prefetch.WaitIdle();
        f.Host.FailPrepareBindings = true;
        Assert.Throws<RenderExecutorFatalException>(f.Translation.DrainPendingDraws);
        Assert.False(f.Translation.IsDraining);
        Assert.Equal(1, f.Translation.PendingDrawCount);
        Assert.Null(f.Prefetch.Current);
        Assert.Equal(reuse ? 1 : 0, f.Spares.Length);
        f.AssertCleared();
        f.Host.FailPrepareBindings = false;
        f.Draw(9);
        f.Drain();
        Assert.Equal(new[] { "draw 6 1 0 0", "draw 9 1 0 0" }, f.Draws);
        f.AssertCleared();
    }

    [Fact]
    public void AReusedWrapperDoesNotCarryThePreviousDrawsGeometrySnapshot()
    {
        using var f = new Fixture(true);
        var capture = AgcExports.CreateGeometrySnapshotsForTests(0, 0, 0, 3, 1, []);
        f.Translation.BeginSubmission(0, 1, capture, f.Interpreter);
        f.Draw(3);
        var queue = (IEnumerable)typeof(AgcExports.CommandStreamTranslation)
            .GetField("_pendingDraws", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(f.Translation)!;
        var wrapper = Assert.Single(queue.Cast<object>());
        var snapshot = wrapper.GetType().GetField("VertexSnapshot")!.GetValue(wrapper);
        Assert.NotNull(snapshot);
        f.Drain();
        f.AssertCleared();
        f.Translation.BeginSubmission(0, 2, null, f.Interpreter);
        f.Draw(6);
        Assert.Same(wrapper, Assert.Single(queue.Cast<object>()));
        Assert.Null(wrapper.GetType().GetField("VertexSnapshot")!.GetValue(wrapper));
        f.Drain();
        Assert.Equal(new[] { "draw 3 1 0 0", "draw 6 1 0 0" }, f.Draws);
        f.AssertCleared();
    }

    [Fact]
    public void ProducerOwnedWrappersNeverEnterTheLocalCacheAndRetentionIsBounded()
    {
        using var f = new Fixture(true);
        for (var i = 0; i < 300; i++) f.Draw(3);
        f.Drain();
        var spares = f.Spares;
        Assert.Equal(256, spares.Length);
        var sink = new Sink(); f.Translation.Sink = sink;
        f.Draw(6, 900); f.Draw(9, 901);
        f.Prefetch.WaitIdle();
        Assert.Equal(2, sink.Work.Count);
        Assert.DoesNotContain(sink.Work[0], spares);
        Assert.DoesNotContain(sink.Work[1], spares);
        foreach (var work in sink.Work) f.Translation.RunDeferred(work);
        Assert.Equal(900UL, sink.Work[0].GetType().GetField("SubmitId")!.GetValue(sink.Work[0]));
        Assert.Equal(901UL, sink.Work[1].GetType().GetField("SubmitId")!.GetValue(sink.Work[1]));
        Assert.Equal(spares, f.Spares);
        Assert.Equal(new[] { "draw 6 1 0 0", "draw 9 1 0 0" }, f.Draws[^2..]);
        f.AssertCleared();
    }
}
