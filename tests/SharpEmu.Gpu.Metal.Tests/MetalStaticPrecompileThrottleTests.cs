// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE.Host;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

// The static precompile stays cheap for the game: background threads, one worker once the
// guest presents, and a library the frame path needs is built once.
public sealed class MetalStaticPrecompileThrottleTests
{
    private static readonly TimeSpan Short = TimeSpan.FromMilliseconds(100);
    private static readonly TimeSpan Released = TimeSpan.FromSeconds(10);
    private static readonly UInt128 Key = new(1, 2);

    [Fact]
    public void AllConfiguredWorkersRunWhileTheGuestLoads()
    {
        Assert.Equal(2, MetalStaticPrecompile.AllowedWorkers(2, presenting: false, holding: false, throttle: true));
    }

    [Fact]
    public void OnlyOneWorkerRunsOnceTheGuestPresents()
    {
        Assert.Equal(MetalStaticPrecompile.PresentingJobs, MetalStaticPrecompile.AllowedWorkers(2, presenting: true, holding: false, throttle: true));
        Assert.Equal(1, MetalStaticPrecompile.AllowedWorkers(1, presenting: true, holding: false, throttle: true));
    }

    [Fact]
    public void TheKillSwitchAndTheHeldModesKeepEveryWorker()
    {
        Assert.Equal(2, MetalStaticPrecompile.AllowedWorkers(2, presenting: true, holding: false, throttle: false));
        Assert.Equal(6, MetalStaticPrecompile.AllowedWorkers(6, presenting: true, holding: true, throttle: true));
    }

    [Theory]
    [InlineData(null, HostThreadQos.Background)]
    [InlineData("", HostThreadQos.Background)]
    [InlineData("background", HostThreadQos.Background)]
    [InlineData("Utility", HostThreadQos.Utility)]
    [InlineData("off", HostThreadQos.Default)]
    [InlineData("0", HostThreadQos.Default)]
    public void TheQosSwitchPicksTheThreadClass(string? value, HostThreadQos expected)
    {
        Assert.Equal(expected, MetalStaticPrecompile.ParseQos(value));
    }

    [Fact]
    public void AHeldPrecompileDoesNotRunInTheBackgroundClass()
    {
        Assert.Equal(HostThreadQos.UserInitiated, MetalStaticPrecompile.ThreadQos(holding: true));
    }

    [Fact]
    public void ABackgroundThreadReadsTheBackgroundClassBack()
    {
        if (!OperatingSystem.IsMacOS())
        {
            return;
        }

        var applied = false;
        uint actual = 0;
        var thread = HostBackgroundThread.Start("SharpEmu test precompile", HostThreadQos.Background, 1024 * 1024, ok =>
        {
            applied = ok;
            actual = HostThreadQualityOfService.GetCurrentThreadClassForDiagnostics();
        });
        Assert.True(thread.Join(20_000));
        if (thread.IsNative)
        {
            Assert.True(applied);
            Assert.Equal(0x09u, actual);
        }
    }

    [Fact]
    public void ALibraryNobodyBuildsIsLeftToTheFramePath()
    {
        var claims = new MetalStaticLibraryClaims();
        Assert.False(claims.Demand(Key, Short));
        Assert.Equal(0, claims.TakenCount);
    }

    [Fact]
    public void ALibraryAWorkerOnlyQueuedIsTakenAndTheWorkerLeavesIt()
    {
        var claims = new MetalStaticLibraryClaims();
        using var entry = claims.Register(Key)!;

        // The frame path compiles it itself; the worker, let in later, finds it taken.
        Assert.False(claims.Demand(Key, Short));
        Assert.Equal(1, claims.TakenCount);
        Assert.False(entry.TryBegin());
    }

    [Fact]
    public void ALibraryAWorkerBeganIsWaitedForNotBuiltTwice()
    {
        var claims = new MetalStaticLibraryClaims();
        var entry = claims.Register(Key)!;
        Assert.True(entry.TryBegin());

        var demand = Task.Run(() => claims.Demand(Key, Released));
        Assert.False(demand.Wait(Short));

        entry.Dispose();
        Assert.True(demand.Wait(Released));
        Assert.True(demand.Result);
        Assert.Equal(1, claims.WaitedCount);
        Assert.Equal(0, claims.TakenCount);
        Assert.Equal(0, claims.Count);
    }

    [Fact]
    public void AWaitThatTimesOutLeavesTheFramePathToCompile()
    {
        var claims = new MetalStaticLibraryClaims();
        using var entry = claims.Register(Key)!;
        Assert.True(entry.TryBegin());

        Assert.False(claims.Demand(Key, Short));
        Assert.Equal(1, claims.TimedOutCount);
    }

    [Fact]
    public void TwoWorkersDoNotRegisterTheSameLibrary()
    {
        var claims = new MetalStaticLibraryClaims();
        using var first = claims.Register(Key);
        Assert.NotNull(first);
        Assert.Null(claims.Register(Key));
        first!.Dispose();
        using var again = claims.Register(Key);
        Assert.NotNull(again);
    }

    [Fact]
    public void AReleasedLibraryIsNotWaitedFor()
    {
        var claims = new MetalStaticLibraryClaims();
        var entry = claims.Register(Key)!;
        Assert.True(entry.TryBegin());
        entry.Dispose();

        Assert.False(claims.Demand(Key, Released));
        Assert.Equal(0, claims.WaitedCount);
    }
}
