// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.GpuCommands;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

public sealed unsafe partial class MetalEndOfPipeOrderTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ACompletionRemainsPendingWhileItsFinalCallbackRuns(bool gpuWords)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        using var entered = new ManualResetEventSlim(false);
        using var release = new ManualResetEventSlim(false);
        var host = rig.Host;
        host.GpuWordsEnabled = gpuWords;
        host.RecordEndOfPipe(new(EndOfPipeWriteKind.Write32, 1, WindowedLabel, 1));
        var serial = host.SubmissionSerial;
        host.Completions.Enqueue(serial, () =>
        {
            entered.Set();
            release.Wait(TimeSpan.FromSeconds(10));
        });
        try
        {
            host.FlushCommandBuffer();
            Assert.True(entered.Wait(TimeSpan.FromSeconds(10)));
            Assert.Equal(1, host.Completions.PendingCount);
        }
        finally
        {
            release.Set();
            host.Completions.WaitFor(serial);
        }
        Assert.Equal(0, host.Completions.PendingCount);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AnInlineHostStoreCannotOvertakeTheFinalRunningCompletion(bool gpuWords)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        using var entered = new ManualResetEventSlim(false);
        using var release = new ManualResetEventSlim(false);
        var host = rig.Host;
        host.GpuWordsEnabled = gpuWords;
        host.RecordEndOfPipe(new(EndOfPipeWriteKind.Write32, 1, WindowedLabel, 1));
        var serial = host.SubmissionSerial;
        host.Completions.Enqueue(serial, () =>
        {
            entered.Set();
            release.Wait(TimeSpan.FromSeconds(10));
            rig.Backing.TryWrite(UnwindowedLabel, BitConverter.GetBytes(1UL));
        });
        try
        {
            host.FlushCommandBuffer();
            Assert.True(entered.Wait(TimeSpan.FromSeconds(10)));
            host.RecordCommandProcessorWrite(UnwindowedLabel, BitConverter.GetBytes(2UL), "WRITE_DATA");
            Assert.Equal(0UL, rig.Backing.Read64(UnwindowedLabel));
            Assert.Equal(0L, host.TakeInlineCompletions());
        }
        finally
        {
            release.Set();
            host.Completions.WaitFor(serial);
        }
        Assert.Equal(2UL, rig.Backing.Read64(UnwindowedLabel));
    }
}
