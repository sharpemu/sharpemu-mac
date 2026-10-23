// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE.Host;
using Xunit;

namespace SharpEmu.Libs.Tests.HLE;

public sealed class HostBackgroundThreadTests
{
    [Fact]
    public void RunsTheBodyAndJoins()
    {
        var ran = new ManualResetEventSlim(false);
        var thread = HostBackgroundThread.Start(
            "SharpEmu test thread", HostThreadQos.UserInteractive, 1024 * 1024, _ => ran.Set());
        Assert.True(ran.Wait(TimeSpan.FromSeconds(20)));
        Assert.True(thread.Join(20_000));
    }

    [Fact]
    public void OnMacOSTheClassIsInEffectInsideTheBody()
    {
        if (!OperatingSystem.IsMacOS())
        {
            return;
        }

        var applied = false;
        uint actual = 0;
        var thread = HostBackgroundThread.Start("SharpEmu test thread", HostThreadQos.UserInteractive, 1024 * 1024, ok =>
        {
            applied = ok;
            actual = HostThreadQualityOfService.GetCurrentThreadClassForDiagnostics();
        });
        Assert.True(thread.Join(20_000));
        // A native thread takes the class at creation; the managed fallback may be refused it.
        if (thread.IsNative)
        {
            Assert.True(applied);
            Assert.Equal(HostThreadQualityOfService.ToDarwinClass(HostThreadQos.UserInteractive), actual);
        }
    }
}
