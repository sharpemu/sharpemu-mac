// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.GpuCommands;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// A coalesced Stream commit that a limit asks for while a render pass is open waits for the
/// pass to end: two draws into the same target with the submission limit reached between them
/// stay in one pass, and the commit is made at the next flush with no pass open. The pixels
/// are those of the two passes the commit made before. Past the hold limit the commit still
/// waits while the GPU runs a buffer committed earlier, and is made at the limit when it does
/// not. Device work is behind the exclusive device gate.
/// </summary>
public sealed class MetalFlushHoldTests(ITestOutputHelper output)
{
    private const uint Width = 64;
    private const uint Height = 8;
    private const ulong Address = MetalPassRig.GuestBase + 0x10000;

    [Theory]
    [InlineData(true, true, 0L, 100L, true)]
    [InlineData(true, true, 99L, 100L, true)]
    [InlineData(true, true, 100L, 100L, false)]
    [InlineData(false, true, 0L, 100L, false)]
    [InlineData(true, false, 0L, 100L, false)]
    public void ALimitCommitWaitsOnlyForAnOpenPassAndOnlyUpToTheHoldLimit(
        bool hold, bool passOpen, long wanted, long max, bool expected)
    {
        Assert.Equal(expected, MetalRenderHost.HoldsCommitForPass(hold, coalesce: true, passOpen, wanted, max));
        Assert.False(MetalRenderHost.HoldsCommitForPass(hold, coalesce: false, passOpen, wanted, max));
    }

    [Theory]
    [InlineData(true, true, 99L, 100L, true)]
    [InlineData(true, true, 100L, 100L, false)]
    [InlineData(true, false, 0L, 100L, false)]
    [InlineData(false, true, 0L, 100L, false)]
    public void APastLimitCommitWaitsOnlyForABusyGpuAndOnlyUpToTheBusyLimit(
        bool holdBusy, bool gpuBusy, long wanted, long max, bool expected) =>
        Assert.Equal(expected, MetalRenderHost.HoldsCommitForBusyGpu(holdBusy, gpuBusy, wanted, max));

    // The hold limit is already past at the limit's commit (Demon's Souls' G-buffer outlasts it):
    // a busy GPU keeps the pass; an idle one, or the switch off, gets the commit and two passes.
    [Theory]
    [InlineData(true, true, 1L)]
    [InlineData(true, false, 2L)]
    [InlineData(false, true, 2L)]
    public void ACappedCommitWaitsForTheOpenPassOnlyWhileTheGpuIsBusy(bool holdBusy, bool gpuBusy, long passes)
    {
        if (!RequireDevice()) return;
        var outcome = Run(hold: true, capped: true, holdBusy, gpuBusy);
        var split = Run(hold: false);

        Assert.Equal(passes, outcome.Passes);
        Assert.Equal(passes == 1 ? "" : "stream-capped:1", outcome.EndFlushBy);
        Assert.Equal(passes == 1 ? 0L : 1L, outcome.Capped);
        Assert.True(passes == 2 || outcome.Busy >= 1, $"busy={outcome.Busy}");
        Assert.Equal(split.Pixels, outcome.Pixels);
    }

    [Fact]
    public void ACommitAtTheSubmissionLimitWaitsForTheOpenPassWithTheSamePixels()
    {
        if (!RequireDevice()) return;
        var held = Run(hold: true);
        var split = Run(hold: false);

        Assert.Equal(1L, held.Passes);
        Assert.True(held.Held >= 1, $"held={held.Held}");
        Assert.Equal(2L, split.Passes);
        Assert.Equal(0L, split.Held);
        // The held commit was made at the flush after the pass ended.
        Assert.True(held.CommittedAfter);
        Assert.Equal(split.Pixels, held.Pixels);
        for (var x = 0u; x < Width; x++)
        {
            var texel = (int)(x * 4);
            Assert.Equal(x < Width / 2 ? (byte)0xFF : (byte)0x00, held.Pixels[texel]);
            Assert.Equal(x < Width / 2 ? (byte)0x00 : (byte)0xFF, held.Pixels[texel + 1]);
        }
    }

    private sealed record Outcome(byte[] Pixels, long Passes, long Held, bool CommittedAfter, long Capped = 0, long Busy = 0, string EndFlushBy = "");

    private Outcome Run(bool hold, bool capped = false, bool holdBusy = false, bool gpuBusy = false)
    {
        using var rig = new MetalPassRig(Device.Value, 1024 * 1024);
        var host = rig.Host;
        host.HoldFlushForPass = hold;
        host.HoldFlushWhileGpuBusy = holdBusy;
        host.AssumeGpuBusy = gpuBusy;
        host.AssumeStreamBusy = true;
        if (capped)
        {
            host.HoldFlushTicks = 0;
        }
        var formats = new[] { GpuFormat.R8G8B8A8Unorm };
        var red = rig.Pipeline(formats, ["1.0f, 0.0f, 0.0f, 1.0f"], -1f, 0f);
        var green = rig.Pipeline(formats, ["0.0f, 1.0f, 0.0f, 1.0f"], 0f, 1f);
        var target = rig.MakeTarget(Address, GpuFormat.R8G8B8A8Unorm, Width, Height);
        _ = host.TakeHeldCommits();
        _ = host.TakeBusyHeldCommits();
        _ = host.TakePassEndFlushReasons();
        var first = host.RenderPassSerial;
        using (host.BeginPreparation())
        {
            rig.Draw(red, [target]);
        }

        // Submissions end until the coalescing limit asks for a commit.
        for (var submission = 0; submission < MetalV2Diagnostics.CoalesceMaxSubmissions; submission++)
        {
            ((ICommandStreamHost)host).Flush();
        }

        using (host.BeginPreparation())
        {
            rig.Draw(green, [target]);
            host.EndRendering();
        }

        var serialBefore = host.SubmissionSerial;
        ((ICommandStreamHost)host).Flush();
        var committedAfter = host.SubmissionSerial != serialBefore;
        var passes = host.RenderPassSerial - first;
        var (heldCount, cappedCount) = host.TakeHeldCommits();
        var busyCount = host.TakeBusyHeldCommits();
        var endFlushBy = host.TakePassEndFlushReasons();
        host.FlushCommandBuffer();
        var pixels = rig.Read(Address, Width * Height * 4);
        output.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"hold={hold} capped={capped} hold_busy={holdBusy} gpu_busy={gpuBusy} passes={passes} held={heldCount} " +
            $"capped_commits={cappedCount} busy={busyCount} end_flush_by=[{endFlushBy}] committed_after={committedAfter}"));
        return new Outcome(pixels, passes, heldCount, committedAfter, cappedCount, busyCount, endFlushBy);
    }

    private bool RequireDevice()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Held commits are encoded on the device only behind the exclusive device gate.");
            return false;
        }

        Assert.True(OperatingSystem.IsMacOS());
        Assert.NotEqual(0, Device.Value);
        return true;
    }

    private static readonly Lazy<nint> Device = new(MetalNative.MTLCreateSystemDefaultDevice);
}
