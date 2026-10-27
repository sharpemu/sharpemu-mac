// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

/// <summary>
/// Occlusion queries (PIXEL_PIPE_STAT_DUMP begin/end pairs) and zpass predication
/// (SET_PREDICATION + the packet predicate bit). The emulator's dumps are synthetic: every
/// pair reads as visible, so occluded draws are never culled. That costs time, not pixels;
/// what must hold is that a predicate built from real dump pairs never drops a draw whose
/// query saw samples, whichever command buffer or queue carried the dumps.
/// </summary>
public sealed class OcclusionSceneTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private const uint Size = 32;
    private const int DepthBlocks = 16;
    private const ulong Ready = 1UL << 63;
    private const uint Red = 0xFF0000FF;
    private const uint Green = 0xFF00FF00;

    [Fact]
    public void HardwareAuditDistinguishesHiddenDrawsWithoutChangingGuestQueries()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        rig.Host.OcclusionAuditEnabled = true;
        var depth = new DepthRig(rig, Size, Size, 3);
        var pm4 = depth.Pm4;
        var occluder = depth.State().DepthControl(true, true, CompareFunc.Always);
        occluder.Context[Cx.DbCountControl] = 0;
        depth.Draw(occluder, depth.Full(0.25f, 1, 0, 0));
        var query = Query(rig);
        Begin(pm4, query);
        var tested = depth.State().DepthControl(true, false, CompareFunc.Less);
        tested.Context[Cx.DbCountControl] = 1u << 8;
        depth.Draw(tested, depth.Full(0.75f, 0, 1, 0));
        End(pm4, query);
        depth.Draw(tested, depth.Full(0.1f, 0, 1, 0));
        rig.Submit(pm4);
        var totals = rig.Host.OcclusionAuditTotals;
        Assert.Equal(2, totals.Groups);
        Assert.Equal(1, totals.ZeroGroups);
        Assert.Equal((ulong)(Size * Size), totals.Samples);
        Assert.Equal(0, totals.Skipped);
        var words = rig.Read<ulong>(query, DepthBlocks * 2);
        Assert.True((words[1] & ~Ready) > (words[0] & ~Ready));

        // Recycled buffers and later submissions cannot expose stale visible counts.
        var hidden = new ScenePm4();
        rig.DrawMesh(hidden, tested, depth.Full(0.75f, 1, 0, 0));
        rig.Submit(hidden);
        totals = rig.Host.OcclusionAuditTotals;
        Assert.Equal(3, totals.Groups);
        Assert.Equal(2, totals.ZeroGroups);
        Assert.Equal((ulong)(Size * Size), totals.Samples);
    }

    private sealed class Scene
    {
        public required SceneRig Rig { get; init; }
        public required ulong Target { get; init; }

        public SceneState State() => new SceneState(Size, Size).ColorTarget(0, Target, Size, Size);

        /// <summary>The left half red: the occlusion query's draw.</summary>
        public ScenePm4 DrawQueried(ScenePm4 pm4) =>
            Rig.DrawMesh(pm4, State(), SceneDraws.Quad(0.5f, 1, 0, 0, right: 0f));

        /// <summary>The right half green, optionally carrying the predicate bit.</summary>
        public ScenePm4 DrawPredicated(ScenePm4 pm4, bool predicated = true) =>
            Rig.DrawMesh(pm4, State(), SceneDraws.Quad(0.5f, 0, 1, 0, left: 0f), predicated: predicated);

        public uint Right() => SceneDraws.Pixel(Rig.ReadImage(Target, Size, Size), Size, Size * 3 / 4, Size / 2);

        public uint Left() => SceneDraws.Pixel(Rig.ReadImage(Target, Size, Size), Size, Size / 4, Size / 2);
    }

    private Scene? Create()
    {
        var rig = CreateRig();
        return rig is null ? null : new Scene { Rig = rig, Target = rig.ColorTarget(Size, Size) };
    }

    private static ulong Query(SceneRig rig) => rig.Memory.Allocate(DepthBlocks * 16, 64);

    private static ScenePm4 Begin(ScenePm4 pm4, ulong query) =>
        pm4.EventWithAddress(ScenePm4.EventPixelPipeStatDump, 1, query);

    private static ScenePm4 End(ScenePm4 pm4, ulong query) =>
        pm4.EventWithAddress(ScenePm4.EventPixelPipeStatDump, 1, query + 8);

    /// <summary>A pair as the depth blocks would leave it: <paramref name="passed"/> samples in block 0.</summary>
    private static ulong Result(SceneRig rig, ulong passed, bool ready = true)
    {
        var query = Query(rig);
        for (var block = 0u; block < DepthBlocks; block++)
        {
            var begin = 1000UL + block;
            rig.Memory.Write64(query + (block * 16), begin | (ready ? Ready : 0));
            rig.Memory.Write64(query + (block * 16) + 8, (begin + (block == 0 ? passed : 0)) | (ready ? Ready : 0));
        }

        return query;
    }

    [Fact]
    public void AQueryAroundAVisibleDrawReadsReadyPairsWithSamplesPassed()
    {
        var scene = Create();
        if (scene is null) return;
        using var rig = scene.Rig;
        var query = Query(rig);
        var pm4 = new ScenePm4();
        Begin(pm4, query);
        scene.DrawQueried(pm4);
        End(pm4, query);
        rig.Submit(pm4);
        var words = rig.Read<ulong>(query, DepthBlocks * 2);
        ulong passed = 0;
        for (var block = 0; block < DepthBlocks; block++)
        {
            Assert.NotEqual(0UL, words[block * 2] & Ready);
            Assert.NotEqual(0UL, words[(block * 2) + 1] & Ready);
            passed += (words[(block * 2) + 1] & ~Ready) - (words[block * 2] & ~Ready);
        }

        // Half of a 32x32 target is 512 samples; the synthetic counter only promises "some".
        Assert.NotEqual(0UL, passed);
        Assert.Equal(Red, scene.Left());
    }

    [Theory]
    [InlineData(true, false, true)]
    [InlineData(true, true, true)]
    [InlineData(false, false, false)]
    // The dumps land in order with the GPU work before them, so right after the end dump the
    // results are still outstanding, and the no-wait hint runs the draw unpredicated.
    [InlineData(false, true, true)]
    public void AVisibleQueryRunsDrawIfVisibleAndSkipsDrawIfNotVisible(bool drawIfVisible, bool noWaitHint, bool expectDrawn)
    {
        var scene = Create();
        if (scene is null) return;
        using var rig = scene.Rig;
        var query = Query(rig);
        var pm4 = new ScenePm4();
        Begin(pm4, query);
        scene.DrawQueried(pm4);
        End(pm4, query);
        pm4.SetPredication(1, drawIfVisible, noWaitHint, false, query);
        scene.DrawPredicated(pm4);
        rig.Submit(pm4);
        Assert.Equal(expectDrawn ? Green : 0u, scene.Right());
    }

    [Theory]
    [InlineData(0UL, true, false)]
    [InlineData(0UL, false, true)]
    [InlineData(1UL, true, true)]
    [InlineData(1UL, false, false)]
    public void APredicateFollowsTheSamplesInTheResultPairs(ulong passed, bool drawIfVisible, bool expectDrawn)
    {
        var scene = Create();
        if (scene is null) return;
        using var rig = scene.Rig;
        var pm4 = new ScenePm4().SetPredication(1, drawIfVisible, false, false, Result(rig, passed));
        scene.DrawPredicated(pm4);
        rig.Submit(pm4);
        Assert.Equal(expectDrawn ? Green : 0u, scene.Right());
    }

    [Fact]
    public void ThePredicateBitIsWhatOptsADrawIn()
    {
        var scene = Create();
        if (scene is null) return;
        using var rig = scene.Rig;
        var pm4 = new ScenePm4().SetPredication(1, true, false, false, Result(rig, 0));
        scene.DrawPredicated(pm4, predicated: false);
        rig.Submit(pm4);
        Assert.Equal(Green, scene.Right());
    }

    [Fact]
    public void ClearingPredicationLetsPredicatedDrawsRun()
    {
        var scene = Create();
        if (scene is null) return;
        using var rig = scene.Rig;
        var pm4 = new ScenePm4().SetPredication(1, true, false, false, Result(rig, 0));
        pm4.SetPredication(0, false, false, false, 0);
        scene.DrawPredicated(pm4);
        rig.Submit(pm4);
        Assert.Equal(Green, scene.Right());
    }

    [Theory]
    [InlineData(0UL, 0UL, false)]
    [InlineData(5UL, 0UL, true)]
    [InlineData(0UL, 5UL, true)]
    [InlineData(5UL, 5UL, true)]
    public void TheContinueBitOrsQueriesIntoOnePredicate(ulong first, ulong second, bool expectDrawn)
    {
        var scene = Create();
        if (scene is null) return;
        using var rig = scene.Rig;
        var pm4 = new ScenePm4().SetPredication(1, true, false, false, Result(rig, first));
        pm4.SetPredication(1, true, false, true, Result(rig, second));
        scene.DrawPredicated(pm4);
        rig.Submit(pm4);
        Assert.Equal(expectDrawn ? Green : 0u, scene.Right());
    }

    [Fact]
    public void ResultsNotReadyWithTheNoWaitHintLeaveTheDrawUnpredicated()
    {
        var scene = Create();
        if (scene is null) return;
        using var rig = scene.Rig;
        var pm4 = new ScenePm4().SetPredication(1, false, true, false, Result(rig, 0, ready: false));
        scene.DrawPredicated(pm4);
        rig.Submit(pm4);
        Assert.Equal(Green, scene.Right());
    }

    [Fact]
    public void AQuerySpanningCommandBuffersStillReadsVisible()
    {
        var scene = Create();
        if (scene is null) return;
        using var rig = scene.Rig;
        var query = Query(rig);
        rig.Submit(Begin(new ScenePm4(), query));
        rig.Submit(End(scene.DrawQueried(new ScenePm4()), query));
        rig.Submit(scene.DrawPredicated(new ScenePm4().SetPredication(1, true, false, false, query)));
        Assert.Equal(Red, scene.Left());
        Assert.Equal(Green, scene.Right());
    }

    [Fact]
    public void AQueryWhoseDumpsRanOnDifferentQueuesStillReadsVisible()
    {
        var scene = Create();
        if (scene is null) return;
        using var rig = scene.Rig;
        var query = Query(rig);
        // The begin dump on a compute queue, the end dump on the graphics queue: each queue
        // has its own command processor, and the pair must still count the draw between them.
        rig.Submit(Begin(new ScenePm4(), query).ToArray(), queue: 0x20);
        rig.Submit(End(scene.DrawQueried(new ScenePm4()), query));
        rig.Submit(scene.DrawPredicated(new ScenePm4().SetPredication(1, true, false, false, query)));
        Assert.Equal(Red, scene.Left());
        Assert.Equal(Green, scene.Right());
    }

    [Fact]
    public void BackToBackQueriesInOneQueryPoolEachReadVisible()
    {
        var scene = Create();
        if (scene is null) return;
        using var rig = scene.Rig;
        // A pool reused frame after frame: the second frame's begin overwrites the slot while
        // its end still holds the first frame's value until the second end dump lands.
        var query = Query(rig);
        for (var frame = 0; frame < 3; frame++)
        {
            var pm4 = new ScenePm4();
            Begin(pm4, query);
            scene.DrawQueried(pm4);
            End(pm4, query);
            pm4.SetPredication(1, true, false, false, query);
            scene.DrawPredicated(pm4);
            rig.Submit(pm4);
            Assert.Equal(Green, scene.Right());
        }
    }
}
