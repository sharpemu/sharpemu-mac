// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Xunit;

namespace SharpEmu.GpuSplit.Tests;

/// <summary>
/// The threads that park the GPU worker for a bracket: one after another, brackets share a parked
/// thread instead of making one each; brackets that overlap still each have their own.
/// </summary>
[Collection(SplitCollection.Name)]
public sealed class WorkerBracketTests
{
    private static readonly TimeSpan Wait = TimeSpan.FromSeconds(10);

    // The fake worker runs the park on the thread that hands it over.
    private static WorkerBracket Begin(ulong id, Action<Thread>? seen = null)
    {
        var bracket = new WorkerBracket(id, static () => true);
        bracket.Start(park =>
        {
            seen?.Invoke(Thread.CurrentThread);
            park();
        });
        Assert.True(bracket.WaitParked(Wait));
        return bracket;
    }

    [Fact]
    public void BracketsOneAfterAnotherShareOneThread()
    {
        var threads = new HashSet<Thread>();
        // Settle the pool: whatever an earlier test left idle is reused from here on.
        Assert.True(Begin(1, thread => threads.Add(thread)).End(static () => { }, Wait));
        var created = WorkerBracket.BracketRunner.Created;

        for (ulong id = 2; id < 50; id++)
        {
            var ran = 0;
            var bracket = Begin(id, thread => threads.Add(thread));
            Assert.True(bracket.TryRun(() => ran++, Wait));
            Assert.True(bracket.End(() => ran += 10, Wait));
            Assert.Equal(11, ran);
            // Ended: the work goes elsewhere.
            Assert.False(bracket.TryRun(static () => { }, Wait));
        }

        Assert.Equal(created, WorkerBracket.BracketRunner.Created);
        Assert.Single(threads);
    }

    [Fact]
    public void OverlappingBracketsEachHaveAThread()
    {
        var threads = new HashSet<Thread>();
        var first = Begin(100, thread => { lock (threads) threads.Add(thread); });
        var second = Begin(101, thread => { lock (threads) threads.Add(thread); });
        Assert.Equal(2, threads.Count);
        Assert.True(second.End(static () => { }, Wait));
        Assert.True(first.End(static () => { }, Wait));
    }

    [Fact]
    public void ABracketWhoseHandOverThrowsStillEndsAndFreesItsThread()
    {
        var bracket = new WorkerBracket(200, static () => true);
        bracket.Start(static _ => throw new InvalidOperationException("no worker"));
        Assert.True(bracket.WaitParked(Wait));
        Assert.True(bracket.End(static () => { }, Wait));

        var created = WorkerBracket.BracketRunner.Created;
        Assert.True(Begin(201).End(static () => { }, Wait));
        Assert.Equal(created, WorkerBracket.BracketRunner.Created);
    }
}
