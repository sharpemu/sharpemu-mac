// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Reflection;
using SharpEmu.Libs.Gpu.GpuCommands;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

/// <summary>
/// SHARPEMU_END_OF_PIPE_SOFT_WAITS on Metal: the host names the serial a release rides
/// (EndOfPipeLanding.Recorded) and that serial's completion reaches the command stream queue, so a soft wait
/// passes when the release completes and not only at its cap (MetalCommandStream.RecordLanding).
/// </summary>
public sealed class EndOfPipeLandingSceneTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private const ulong Label = SceneMemory.Base + 0x100;

    private static EndOfPipeLanding NewLanding(ulong destination, ulong value) =>
        (EndOfPipeLanding)typeof(EndOfPipeLanding)
            .GetConstructor(BindingFlags.Instance | BindingFlags.NonPublic, [typeof(int), typeof(ulong), typeof(int), typeof(ulong), typeof(long)])!
            .Invoke([1, destination, sizeof(uint), value, 1L]);

    private static ulong TickOf(EndOfPipeLanding landing) =>
        (ulong)typeof(EndOfPipeLanding).GetProperty("Tick", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(landing)!;

    private static ulong CompletedTickOf(CommandStreamQueue queue) =>
        (ulong)typeof(CommandStreamQueue).GetField("_lastCompletedGpuTick", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(queue)!;

    [Fact]
    public void ARecordedReleaseNamesItsSerialAndItsCompletionReachesTheQueue()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var host = rig.Host;
        Assert.True(((ICommandStreamHost)host).RecordsDeferredLabelLandings);
        var landing = NewLanding(Label, 0x1234);

        var serial = host.SubmissionSerial;
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write32, 1, Label, 0x1234) { Landing = landing });
        Assert.Equal(1L, host.RecordedLandings);
        Assert.Equal(serial, TickOf(landing));
        Assert.True(CompletedTickOf(host.CommandStream) < serial);

        host.FlushAndWaitForGpu();
        var deadline = Environment.TickCount64 + 10_000;
        while (CompletedTickOf(host.CommandStream) < serial)
        {
            Assert.True(Environment.TickCount64 < deadline, $"The queue never heard serial {serial} complete.");
            host.PollGpuCompletions();
            Thread.Sleep(1);
        }

        Assert.Equal(0x1234u, BitConverter.ToUInt32(rig.Read(Label, sizeof(uint))));
    }

    [Fact]
    public void AWriteWithoutALandingRecordsNone()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        rig.Host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write32, 1, Label, 0x1234));
        rig.Host.FlushAndWaitForGpu();
        Assert.Equal(0L, rig.Host.RecordedLandings);
    }
}
