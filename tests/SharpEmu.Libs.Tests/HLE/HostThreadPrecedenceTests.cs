// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE.Host;
using Xunit;

namespace SharpEmu.Libs.Tests.HLE;

public sealed class HostThreadPrecedenceTests
{
    [Theory]
    [InlineData(null, 10)]
    [InlineData("", 10)]
    [InlineData("garbage", 10)]
    [InlineData("0", 0)]
    [InlineData("6", 6)]
    [InlineData("20", 20)]
    [InlineData("500", 20)]
    [InlineData("-5", 0)]
    public void PacingImportanceIsParsedAndClamped(string? text, int expected) =>
        Assert.Equal(expected, HostThreadPrecedence.ParsePacingImportance(text));

    [Fact]
    public void TimeConstraintValuesConvertNanosecondsToAbsoluteTime()
    {
        // Apple silicon's timebase is 125/3 (24 MHz); Intel's is 1/1.
        Assert.Equal(24_000u, HostThreadPrecedence.ToAbsoluteTime(1_000_000, new HostThreadPrecedence.MachTimebase { Numer = 125, Denom = 3 }));
        Assert.Equal(1_000_000u, HostThreadPrecedence.ToAbsoluteTime(1_000_000, new HostThreadPrecedence.MachTimebase { Numer = 1, Denom = 1 }));
    }

    [Fact]
    public void OffMacOSNothingIsApplied()
    {
        if (OperatingSystem.IsMacOS())
        {
            return;
        }

        Assert.False(HostThreadPrecedence.IsEnabled);
        Assert.False(HostThreadPrecedence.TryBoostCurrentThread("test"));
        Assert.False(HostThreadPrecedence.TryApplyTimeConstraintToCurrentThread("test"));
        Assert.Null(HostThreadPrecedence.GetCurrentImportance());
    }

    // The boost must take on both kinds of thread the emulator runs its pacing loops on: one
    // CoreCLR created (a QoS class is refused there) and one created natively with a class.
    [Fact]
    public void OnMacOSTheBoostTakesOnManagedAndNativeQosThreads()
    {
        if (!OperatingSystem.IsMacOS() || !HostThreadPrecedence.IsEnabled || HostThreadPrecedence.PacingImportance == 0)
        {
            return;
        }

        int? managedBefore = null, managedAfter = null;
        var managedBoosted = false;
        var managed = new Thread(() =>
        {
            managedBefore = HostThreadPrecedence.GetCurrentImportance();
            managedBoosted = HostThreadPrecedence.TryBoostCurrentThread("test managed");
            managedAfter = HostThreadPrecedence.GetCurrentImportance();
        });
        managed.Start();
        Assert.True(managed.Join(20_000));
        Assert.True(managedBoosted);
        Assert.Equal(0, managedBefore);
        Assert.Equal(HostThreadPrecedence.PacingImportance, managedAfter);

        int? nativeAfter = null;
        var boosted = false;
        var native = HostBackgroundThread.Start("SharpEmu test pacing", HostThreadQos.UserInteractive, 1024 * 1024, _ =>
        {
            boosted = HostThreadPrecedence.TryBoostCurrentThread("test native");
            nativeAfter = HostThreadPrecedence.GetCurrentImportance();
        });
        Assert.True(native.Join(20_000));
        Assert.True(boosted);
        Assert.Equal(HostThreadPrecedence.PacingImportance, nativeAfter);
    }

    [Fact]
    public void OnMacOSTheTimeConstraintTakesOnAThread()
    {
        if (!OperatingSystem.IsMacOS() || !HostThreadPrecedence.IsEnabled)
        {
            return;
        }

        var applied = false;
        var thread = new Thread(() => applied = HostThreadPrecedence.TryApplyTimeConstraintToCurrentThread("test clock"));
        thread.Start();
        Assert.True(thread.Join(20_000));
        // SHARPEMU_MAC_TIME_CONSTRAINT=0 keeps the clock threads timeshare, and then nothing is applied.
        Assert.Equal(Environment.GetEnvironmentVariable(HostThreadPrecedence.TimeConstraintEnvironmentVariable) != "0", applied);
    }
}
