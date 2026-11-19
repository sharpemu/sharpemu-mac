// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE.Host;
using Xunit;

namespace SharpEmu.Libs.Tests.Memory.GuestMemory;

// Alias accesses counted on HostBackingObject's per-thread stripes instead of its shared access
// count: Close still refuses new ones and the drain still waits for every admitted one.
// Phantom backings: nothing reads or writes the (fake) alias address.
public sealed class HostBackingAliasStripesTests
{
    private static HostBackingObject Phantom(Func<HostBackingObject, HostBackingReleaseResult> release)
        => new(0x10000, 4096, 4096, release);

    [Fact]
    public void AHeldAliasAccessBlocksDisposeAndCloseRefusesNewOnes()
    {
        if (!HostBackingObject.AccessStripesEnabled) return;
        var calls = 0;
        var backing = Phantom(_ => { calls++; return new(HostBackingReleaseStatus.Released); });
        Assert.NotNull(backing.AliasStripes);
        Assert.True(backing.TryEnterAliasAccess());
        Assert.Equal(1, backing.ActiveAccesses);
        Assert.False(backing.TryDispose(TimeSpan.Zero, out var timedOut));
        Assert.Equal(HostBackingReleaseStatus.TimedOut, timedOut.Status);
        Assert.True(backing.IsClosing); Assert.Equal(0, calls);
        Assert.False(backing.TryEnterAliasAccess());
        Assert.False(backing.TryEnterAccess());
        Assert.Equal(1, backing.ActiveAccesses);
        backing.ExitAliasAccess();
        Assert.Equal(0, backing.ActiveAccesses);
        Assert.True(backing.TryDispose(TimeSpan.Zero, out _));
        Assert.Equal(1, calls); Assert.True(backing.IsDisposed);
        Assert.False(backing.TryEnterAliasAccess());
    }

    [Fact]
    public void AliasAndOrdinaryAccessesBothHoldTheDrain()
    {
        if (!HostBackingObject.AccessStripesEnabled) return;
        var calls = 0;
        var backing = Phantom(_ => { calls++; return new(HostBackingReleaseStatus.Released); });
        Assert.True(backing.TryEnterAccess());
        Assert.True(backing.TryEnterAliasAccess());
        Assert.Equal(2, backing.ActiveAccesses);
        backing.ExitAccess();
        Assert.False(backing.TryDispose(TimeSpan.Zero, out _));
        backing.ExitAliasAccess();
        Assert.True(backing.TryDispose(TimeSpan.Zero, out _));
        Assert.Equal(1, calls);
    }

    [Fact]
    public void AnAliasExitWithoutAnEnterIsFatalAtTheDrain()
    {
        if (!HostBackingObject.AccessStripesEnabled) return;
        var calls = 0;
        var backing = Phantom(_ => { calls++; return new(HostBackingReleaseStatus.Released); });
        backing.AliasStripes!.OnFatalForControl = message => throw new InvalidOperationException(message);
        backing.ExitAliasAccess();
        var error = Assert.Throws<InvalidOperationException>(() => backing.TryDispose(TimeSpan.FromSeconds(1), out _));
        Assert.Contains("Unbalanced access exit", error.Message);
        Assert.Equal(0, calls); Assert.False(backing.IsDisposed);
    }

    // Ten threads enter and exit alias accesses while one disposes. No access may be in flight when
    // the release runs, and none may be admitted once it has started.
    [Fact]
    public void DisposeNeverReleasesUnderAnAdmittedAliasAccess()
    {
        if (!HostBackingObject.AccessStripesEnabled) return;
        for (var round = 0; round < 20; round++)
        {
            var inFlight = 0;
            var released = 0;
            var violations = 0;
            var admitted = 0L;
            var stop = 0;
            var backing = Phantom(_ =>
            {
                Volatile.Write(ref released, 1);
                Interlocked.MemoryBarrier();
                if (Volatile.Read(ref inFlight) != 0) Interlocked.Increment(ref violations);
                return new(HostBackingReleaseStatus.Released);
            });
            using var started = new CountdownEvent(10);
            var workers = Enumerable.Range(0, 10).Select(_ => new Thread(() =>
            {
                started.Signal();
                while (Volatile.Read(ref stop) == 0)
                {
                    if (!backing.TryEnterAliasAccess()) continue;
                    Interlocked.Increment(ref inFlight);
                    if (Volatile.Read(ref released) != 0) Interlocked.Increment(ref violations);
                    Interlocked.Increment(ref admitted);
                    Thread.SpinWait(20);
                    Interlocked.Decrement(ref inFlight);
                    backing.ExitAliasAccess();
                }
            }) { IsBackground = true }).ToArray();
            foreach (var worker in workers) worker.Start();
            Assert.True(started.Wait(TimeSpan.FromSeconds(10)));
            Thread.Sleep(5);
            Assert.True(backing.TryDispose(TimeSpan.FromSeconds(10), out var result), result.Status.ToString());
            Thread.Sleep(2);
            Volatile.Write(ref stop, 1);
            foreach (var worker in workers) Assert.True(worker.Join(TimeSpan.FromSeconds(10)));
            Assert.Equal(0, violations);
            Assert.True(admitted > 0);
            Assert.Equal(0, backing.ActiveAccesses);
        }
    }
}
