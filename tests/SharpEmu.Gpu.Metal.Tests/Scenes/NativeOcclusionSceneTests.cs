// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

public sealed class NativeOcclusionSceneTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private const uint Size = 32;
    private const ulong Ready = 1UL << 63;
    private static ulong Query(SceneRig rig) => rig.Memory.Allocate(256, 64);
    private static void Dump(ScenePm4 pm4, ulong address) => pm4.EventWithAddress(ScenePm4.EventPixelPipeStatDump, 1, address);
    private static ulong Delta(SceneRig rig, ulong query)
    {
        var words = rig.Read<ulong>(query, 32);
        ulong samples = 0;
        for (var block = 0; block < 16; block++)
        {
            Assert.Equal(Ready, words[block * 2] & Ready);
            Assert.Equal(Ready, words[block * 2 + 1] & Ready);
            samples += (words[block * 2 + 1] & ~Ready) - (words[block * 2] & ~Ready);
        }
        return samples;
    }

    [Fact]
    public void DisabledCountingPassesDoNotReduceOrChangeTheCounter()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        rig.Host.NativeOcclusionEnabled = true;
        var targets = new[] { rig.ColorTarget(Size, Size), rig.ColorTarget(Size, Size), rig.ColorTarget(Size, Size) };
        var seed = new ScenePm4();
        var counted = new SceneState(Size, Size).ColorTarget(0, targets[0], Size, Size);
        counted.Context[Cx.DbCountControl] = 1u << 8;
        rig.DrawMesh(seed, counted, SceneDraws.Quad(0.1f, 1, 0, 0));
        rig.Submit(seed);
        var before = rig.Host.NativeOcclusionTotals;
        var sparse = rig.Memory.Allocate(128, 64);
        const ulong Sentinel = 0x123456789ABCDEF0;
        for (var i = 0; i < 16; i++) rig.Memory.Write64(sparse + (ulong)i * 8, Sentinel);
        var pm4 = new ScenePm4();
        foreach (var target in targets)
        {
            var state = new SceneState(Size, Size).ColorTarget(0, target, Size, Size);
            state.Context[Cx.DbCountControl] = 0;
            rig.DrawMesh(pm4, state, SceneDraws.Quad(0.1f, 1, 0, 0));
            // Only hardware block three is selected; it publishes the ready-only
            // word and must not make this disabled pass produce a prefix snapshot.
            pm4.EventWithAddress(ScenePm4.EventPixelPipeStatControl, 1, (3UL << 9) | (8UL << 11));
            Dump(pm4, sparse);
        }
        rig.Submit(pm4);
        Assert.True(rig.Host.TryReadOcclusionCounter(0, out var count));
        Assert.Equal((ulong)Size * Size, count);
        var totals = rig.Host.NativeOcclusionTotals;
        Assert.Equal(before.Reductions, totals.Reductions);
        Assert.True(totals.UnusedPasses - before.UnusedPasses >= 3, $"Expected three disabled passes: {totals}.");
        Assert.True(totals.InitializedSlots - before.InitializedSlots < 32);
        var words = rig.Read<ulong>(sparse, 16);
        for (var i = 0; i < words.Length; i++) Assert.Equal(i == 12 ? Ready : Sentinel, words[i]);
        foreach (var target in targets)
            Assert.Equal(0xFF0000FFu, SceneDraws.Pixel(rig.ReadImage(target, Size, Size), Size, Size / 2, Size / 2));
    }

    [Fact]
    public void CountingModeTransitionsAndEmptySnapshotsPreserveEarlierResults()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        rig.Host.NativeOcclusionEnabled = true;
        var target = rig.ColorTarget(Size, Size);
        var state = new SceneState(Size, Size).ColorTarget(0, target, Size, Size);
        var first = Query(rig); var second = Query(rig);
        var pm4 = new ScenePm4();
        Dump(pm4, first);
        state.Context[Cx.DbCountControl] = 1u << 8;
        rig.DrawMesh(pm4, state, SceneDraws.Quad(0.1f, 1, 0, 0));
        Dump(pm4, first + 8);
        Dump(pm4, second);
        state.Context[Cx.DbCountControl] = 0;
        rig.DrawMesh(pm4, state, SceneDraws.Quad(0.1f, 0, 1, 0));
        state.Context[Cx.DbCountControl] = (1u << 8) | 1u;
        rig.DrawMesh(pm4, state, SceneDraws.Quad(0.1f, 0, 1, 0));
        state.Context[Cx.DbCountControl] = 1u << 8;
        rig.DrawMesh(pm4, state, SceneDraws.Quad(0.1f, 0, 0, 1));
        Dump(pm4, second + 8);
        rig.Submit(pm4);
        Assert.Equal((ulong)Size * Size, Delta(rig, first));
        Assert.Equal((ulong)Size * Size, Delta(rig, second));
        Assert.True(rig.Host.TryReadOcclusionCounter(0, out var count));
        Assert.Equal(2UL * Size * Size, count);
        Assert.True(rig.Host.NativeOcclusionTotals.InitializedSlots < 32);
    }

    [Fact]
    public void HiddenVisibleAndEmptyQueriesKeepDistinctSnapshotsWithoutSplittingEveryDump()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        rig.Host.NativeOcclusionEnabled = true;
        var d = new DepthRig(rig, Size, Size, 3);
        var setup = d.State().DepthControl(true, true, CompareFunc.Always);
        setup.Context[Cx.DbCountControl] = 0;
        d.Draw(setup, d.Full(0.25f, 1, 0, 0));
        var tested = d.State().DepthControl(true, false, CompareFunc.Less);
        tested.Context[Cx.DbCountControl] = 1u << 8;
        var hidden = Query(rig); var visible = Query(rig); var empty = Query(rig);
        Dump(d.Pm4, hidden);
        d.Draw(tested, d.Full(0.75f, 0, 1, 0));
        Dump(d.Pm4, hidden + 8);
        Dump(d.Pm4, visible);
        d.Draw(tested, d.Full(0.1f, 0, 1, 0));
        d.Draw(tested, d.Full(0.1f, 0, 1, 0));
        Dump(d.Pm4, visible + 8);
        Dump(d.Pm4, empty); Dump(d.Pm4, empty + 8);
        rig.Submit(d.Pm4);
        Assert.Equal(0UL, Delta(rig, hidden));
        Assert.Equal(2UL * Size * Size, Delta(rig, visible));
        Assert.Equal(0UL, Delta(rig, empty));
        var totals = rig.Host.NativeOcclusionTotals;
        Assert.Equal(6, totals.Dumps);
        Assert.True(totals.Reductions < totals.Dumps, $"Each dump split a pass: {totals}.");
        Assert.True(rig.Host.TryReadOcclusionCounter(0, out var count));
        Assert.Equal(2UL * Size * Size, count);
    }

    [Fact]
    public void ResetAndSparseDumpPreserveOtherBytesAndQueues()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        rig.Host.NativeOcclusionEnabled = true;
        var target = rig.ColorTarget(Size, Size);
        var state = new SceneState(Size, Size).ColorTarget(0, target, Size, Size);
        state.Context[Cx.DbCountControl] = 1u << 8;
        var pm4 = new ScenePm4();
        rig.DrawMesh(pm4, state, SceneDraws.Quad(0.1f, 1, 0, 0));
        rig.Submit(pm4);
        Assert.True(rig.Host.TryReadOcclusionCounter(0, out var drawn));
        Assert.Equal((ulong)Size * Size, drawn);
        var result = rig.Memory.Allocate(128, 64);
        for (var i = 0; i < 16; i++) rig.Memory.Write64(result + (ulong)i * 8, 0x123456789ABCDEF0);
        Assert.True(rig.Host.DumpOcclusionCounter(0, result, 9, 32));
        Assert.True(rig.Host.ResetOcclusionCounter(1));
        Assert.True(rig.Host.TryReadOcclusionCounter(0, out var unchanged));
        Assert.Equal(drawn, unchanged);
        var words = rig.Read<ulong>(result, 16);
        for (var i = 0; i < words.Length; i++)
            Assert.Equal(i == 0 ? Ready | drawn : i == 12 ? Ready : 0x123456789ABCDEF0UL, words[i]);
        Assert.True(rig.Host.ResetOcclusionCounter(0));
        Assert.True(rig.Host.TryReadOcclusionCounter(0, out var reset));
        Assert.Equal(0UL, reset);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void QueryPairsRemainCorrectAcrossCommandBuffersAndReusedGuestMemory(bool visible)
    {
        using var rig = CreateRig();
        if (rig is null) return;
        rig.Host.NativeOcclusionEnabled = true;
        var target = rig.ColorTarget(Size, Size);
        var state = new SceneState(Size, Size).ColorTarget(0, target, Size, Size);
        state.Context[Cx.DbCountControl] = 1u << 8;
        var query = Query(rig);
        for (var round = 0; round < 4; round++)
        {
            // A previous ready pair must not become the result of a new empty query.
            var begin = new ScenePm4(); Dump(begin, query); rig.Submit(begin);
            var end = new ScenePm4();
            if (visible && (round & 1) == 0) rig.DrawMesh(end, state, SceneDraws.Quad(0.1f, 1, 0, 0));
            Dump(end, query + 8); rig.Submit(end);
            Assert.Equal(visible && (round & 1) == 0 ? (ulong)Size * Size : 0UL, Delta(rig, query));
        }
    }
    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void ClearedQueryMemoryAndPendingPredicationRespectVisibility(bool visible, bool noWait)
    {
        using var rig = CreateRig();
        if (rig is null) return;
        rig.Host.NativeOcclusionEnabled = true;
        var query = Query(rig);
        // Seed a stale visible pair, then replace it through guest WRITE_DATA.
        for (var block = 0; block < 16; block++)
        {
            rig.Memory.Write64(query + (ulong)block * 16, Ready);
            rig.Memory.Write64(query + (ulong)block * 16 + 8, Ready | 123);
        }
        var target = rig.ColorTarget(Size, Size);
        var state = new SceneState(Size, Size).ColorTarget(0, target, Size, Size);
        state.Context[Cx.DbCountControl] = 1u << 8;
        var pm4 = new ScenePm4();
        var clear = new uint[67];
        clear[0] = 1u << 8; clear[1] = (uint)query; clear[2] = (uint)(query >> 32);
        pm4.Packet(ScenePm4.OpWriteData, clear);
        Dump(pm4, query);
        if (visible) rig.DrawMesh(pm4, state, SceneDraws.Quad(0.1f, 1, 0, 0, right: 0f));
        Dump(pm4, query + 8);
        pm4.SetPredication(1, true, noWait, false, query);
        rig.DrawMesh(pm4, state, SceneDraws.Quad(0.1f, 0, 1, 0, left: 0f), predicated: true);
        rig.Submit(pm4);
        var pixel = SceneDraws.Pixel(rig.ReadImage(target, Size, Size), Size, Size * 3 / 4, Size / 2);
        Assert.Equal(visible || noWait ? 0xFF00FF00u : 0u, pixel);
        Assert.Equal(visible ? (ulong)Size * Size / 2 : 0UL, Delta(rig, query));
    }

    [Fact]
    public void PacketResetAndComputeQueueLeaveGraphicsCounterIndependent()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        rig.Host.NativeOcclusionEnabled = true;
        var query = Query(rig);
        var target = rig.ColorTarget(Size, Size);
        var state = new SceneState(Size, Size).ColorTarget(0, target, Size, Size);
        state.Context[Cx.DbCountControl] = 1u << 8;
        var graphics = new ScenePm4();
        rig.DrawMesh(graphics, state, SceneDraws.Quad(0.1f, 1, 0, 0));
        rig.Submit(graphics);
        var compute = new ScenePm4().Event(ScenePm4.EventPixelPipeStatReset);
        Dump(compute, query); Dump(compute, query + 8);
        rig.Submit(compute.ToArray(), queue: 0x20);
        Assert.Equal(0UL, Delta(rig, query));
        Assert.True(rig.Host.TryReadOcclusionCounter(0, out var graphicsCount));
        Assert.Equal((ulong)Size * Size, graphicsCount);
        rig.Submit(new ScenePm4().Event(ScenePm4.EventPixelPipeStatReset));
        Assert.True(rig.Host.TryReadOcclusionCounter(0, out var reset));
        Assert.Equal(0UL, reset);
    }

    [Fact]
    public void MoreThanOneBatchOfDumpsPreservesEverySnapshot()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        rig.Host.NativeOcclusionEnabled = true;
        const int count = 4100;
        var results = rig.Memory.Allocate(count * 256, 64);
        var target = rig.ColorTarget(Size, Size);
        var state = new SceneState(Size, Size).ColorTarget(0, target, Size, Size);
        state.Context[Cx.DbCountControl] = 1u << 8;
        var pm4 = new ScenePm4();
        for (var i = 0; i < count; i++)
        {
            if (i == 0 || i == 4096) rig.DrawMesh(pm4, state, SceneDraws.Quad(0.1f, 1, 0, 0));
            Dump(pm4, results + (ulong)i * 256);
        }
        rig.Submit(pm4);
        var words = rig.Read<ulong>(results, count * 32);
        for (var i = 0; i < count; i++)
            Assert.Equal(Ready | ((ulong)Size * Size * (i < 4096 ? 1UL : 2UL)), words[i * 32]);
    }

    [Fact]
    public void BatchBudgetWaitsForCompletedStorageWithoutLosingDrawsOrDumps()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        rig.Host.NativeOcclusionEnabled = true;
        rig.Host.NativeOcclusionBatchBudget = 2;
        var targets = new[] { rig.ColorTarget(Size, Size), rig.ColorTarget(Size, Size) };
        var query = Query(rig);
        var pm4 = new ScenePm4();
        Dump(pm4, query);
        for (var i = 0; i < 12; i++)
        {
            var state = new SceneState(Size, Size).ColorTarget(0, targets[i % 2], Size, Size);
            state.Context[Cx.DbCountControl] = 1u << 8;
            rig.DrawMesh(pm4, state, SceneDraws.Quad(0.1f, 1, 0, 0));
        }
        Dump(pm4, query + 8);
        rig.Submit(pm4);
        Assert.Equal(12UL * Size * Size, Delta(rig, query));
        var totals = rig.Host.NativeOcclusionTotals;
        Assert.Equal(2, totals.Buffers);
        Assert.True(totals.BudgetWaits > 0, $"The tiny budget did not exercise backpressure: {totals}.");
        foreach (var target in targets)
            Assert.Equal(0xFF0000FFu, SceneDraws.Pixel(rig.ReadImage(target, Size, Size), Size, Size / 2, Size / 2));
    }

}
