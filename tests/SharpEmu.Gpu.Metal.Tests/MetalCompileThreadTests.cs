// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE.Host;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

// The compile threads run at the class they are started with, a thread of them may move
// itself to another later, and a join waits for the body.
public sealed class MetalCompileThreadTests
{
    private const uint QosClassUserInitiated = 0x19;
    private const uint QosClassUtility = 0x11;

    [Fact]
    public void AThreadRunsItsBodyAtTheRequestedClassAndCanLowerItself()
    {
        uint started = 0;
        uint lowered = 0;
        var applied = false;
        var thread = MetalCompileThread.Start("SharpEmu test compile thread", HostThreadQos.UserInitiated, () =>
        {
            started = HostThreadQualityOfService.GetCurrentThreadClassForDiagnostics();
            applied = MetalCompileThread.ApplyToCurrent(HostThreadQos.Utility);
            lowered = HostThreadQualityOfService.GetCurrentThreadClassForDiagnostics();
        });

        Assert.True(thread.Join(TimeSpan.FromSeconds(30)));
        Assert.False(thread.IsCurrentThread);
        if (!OperatingSystem.IsMacOS() || !MetalCompileThread.QosEnabled)
        {
            return;
        }

        Assert.Equal(QosClassUserInitiated, started);
        Assert.True(applied);
        Assert.Equal(QosClassUtility, lowered);
    }

    [Fact]
    public void ACreatedThreadRunsOnlyOnceStarted()
    {
        using var ran = new ManualResetEventSlim(false);
        var thread = MetalCompileThread.Create("SharpEmu test compile thread", HostThreadQos.Utility, ran.Set);
        Assert.False(ran.Wait(TimeSpan.FromMilliseconds(100)));
        thread.Start();
        Assert.True(ran.Wait(TimeSpan.FromSeconds(30)));
        Assert.True(thread.Join(TimeSpan.FromSeconds(30)));
    }
}
