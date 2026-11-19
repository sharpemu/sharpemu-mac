// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Abstractions;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// With the GPU dry and enough recorded since the last commit, a draw about to prepare commits the
/// open buffer, ending the open render pass: the pass resumes in the next buffer with its
/// attachments loaded, and the image is that of the unsplit pass. A busy GPU, too few commands or
/// too little recording for what the split costs leave the pass whole. Device work is behind the
/// exclusive device gate.
/// </summary>
public sealed class MetalGpuDryCommitTests(ITestOutputHelper output)
{
    private const uint Width = 64;
    private const uint Height = 8;
    private const ulong Address = MetalPassRig.GuestBase + 0x10000;

    // Times in microseconds: 12 us per MB of split traffic, 200 percent of it, a 500 us floor.
    private const long MinTicks = 500;
    private const long SplitTicksPerMb = 12;
    private const int SplitPercent = 200;

    // MetalRenderHost.GpuDryVerdict, as the theory data takes it.
    private const int Commit = 0;
    private const int TooFew = 1;
    private const int TooLittle = 2;

    // Demon's Souls' G-buffer set: 1920x1080, four R32Uint, R16G16Sfloat, R16G16B16A16Sfloat, D32 and S8.
    private const long GBufferBytes = 1920L * 1080 * (4 * 4 + 4 + 8 + 4 + 1);

    // A 1024x1024 depth-only shadow map.
    private const long ShadowBytes = 1024L * 1024 * 4;

    [Theory]
    // No pass open: the floor alone.
    [InlineData(64L, 500L, 0L, Commit)]
    [InlineData(64L, 499L, 0L, TooLittle)]
    [InlineData(63L, 100_000L, 0L, TooFew)]
    // The shadow map: 500 + 2 * 4.19 MB * 12 us * 2 = ~701 us.
    [InlineData(200L, 710L, ShadowBytes, Commit)]
    [InlineData(200L, 690L, ShadowBytes, TooLittle)]
    // The G-buffer: 500 + 2 * 68.4 MB * 12 us * 2 = ~3784 us; the trace's dry point had ~4.1 ms recorded.
    [InlineData(385L, 4100L, GBufferBytes, Commit)]
    [InlineData(300L, 3700L, GBufferBytes, TooLittle)]
    [InlineData(10L, 100_000L, GBufferBytes, TooFew)]
    public void ADryGpuGetsTheBufferOnlyWhenItHoldsMoreThanTheSplitCosts(
        long commands, long recorded, long passBytes, int expected) =>
        Assert.Equal((MetalRenderHost.GpuDryVerdict)expected, MetalRenderHost.GpuDryVerdictFor(
            commands, recorded, passBytes, minCommands: 64, MinTicks, SplitTicksPerMb, SplitPercent));

    [Fact]
    public void TheSplitBarScalesWithThePassAttachmentBytes()
    {
        static long Bar(long bytes)
        {
            var low = 0L;
            var high = 1_000_000L;
            while (low < high)
            {
                var middle = (low + high) / 2;
                if (MetalRenderHost.GpuDryVerdictFor(1000, middle, bytes, 1, MinTicks, SplitTicksPerMb, SplitPercent) ==
                    MetalRenderHost.GpuDryVerdict.Commit)
                {
                    high = middle;
                }
                else
                {
                    low = middle + 1;
                }
            }

            return low;
        }

        Assert.Equal(MinTicks, Bar(0));
        Assert.InRange(Bar(ShadowBytes), 700, 702);
        Assert.InRange(Bar(GBufferBytes), 3780, 3790);
    }

    [Fact]
    public void ADryGpuSplitsTheOpenPassAndItResumesWithTheSameImage()
    {
        if (!RequireDevice()) return;
        var dry = Run(enabled: true, gpuBusy: false);
        var whole = Run(enabled: false, gpuBusy: false);

        Assert.Equal(2L, dry.Passes);
        Assert.Equal("gpu-dry:1", dry.EndFlushBy);
        Assert.Equal(1L, dry.Commits);
        Assert.Equal(1L, whole.Passes);
        Assert.Equal(whole.Pixels, dry.Pixels);
        for (var x = 0u; x < Width; x++)
        {
            var texel = (int)(x * 4);
            Assert.Equal(x < Width / 2 ? (byte)0xFF : (byte)0x00, dry.Pixels[texel]);
            Assert.Equal(x < Width / 2 ? (byte)0x00 : (byte)0xFF, dry.Pixels[texel + 1]);
        }
    }

    [Theory]
    [InlineData(true, 2L, 0L, "busy")]
    [InlineData(false, 1000L, 0L, "commands")]
    [InlineData(false, 2L, long.MaxValue / 4, "split")]
    public void ABusyGpuOrTooLittleRecordedLeavesThePassWhole(bool gpuBusy, long minCommands, long minTicks, string bar)
    {
        if (!RequireDevice()) return;
        var outcome = Run(enabled: true, gpuBusy, minCommands, minTicks);

        Assert.Equal(1L, outcome.Passes);
        Assert.Equal("", outcome.EndFlushBy);
        Assert.Equal(0L, outcome.Commits);
        Assert.Equal(bar == "commands" ? 1L : 0L, outcome.RefusedCommands);
        Assert.Equal(bar == "split" ? 1L : 0L, outcome.RefusedSplit);
    }

    private sealed record Outcome(
        byte[] Pixels, long Passes, string EndFlushBy, long Commits, long RefusedCommands, long RefusedSplit);

    private Outcome Run(bool enabled, bool gpuBusy, long minCommands = 2, long minTicks = 0)
    {
        using var rig = new MetalPassRig(Device.Value, 1024 * 1024);
        var host = rig.Host;
        host.CommitWhenGpuDry = enabled;
        host.AssumeGpuBusy = gpuBusy;
        host.AssumeStreamBusy = true;
        host.GpuDryMinCommands = minCommands;
        host.GpuDryMinTicks = minTicks;
        var formats = new[] { GpuFormat.R8G8B8A8Unorm };
        var red = rig.Pipeline(formats, ["1.0f, 0.0f, 0.0f, 1.0f"], -1f, 0f);
        var green = rig.Pipeline(formats, ["0.0f, 1.0f, 0.0f, 1.0f"], 0f, 1f);
        var target = rig.MakeTarget(Address, GpuFormat.R8G8B8A8Unorm, Width, Height);
        _ = host.TakeGpuDryCommits();
        _ = host.TakePassEndFlushReasons();
        var first = host.RenderPassSerial;
        // Two draws recorded (the second preparation starts the recording clock), then the third
        // preparation finds the GPU dry and the bars met.
        for (var draw = 0; draw < 2; draw++)
        {
            using (host.BeginPreparation())
            {
                rig.Draw(red, [target]);
            }
        }

        using (host.BeginPreparation())
        {
            rig.Draw(green, [target]);
            host.EndRendering();
        }

        var passes = host.RenderPassSerial - first;
        var endFlushBy = host.TakePassEndFlushReasons();
        var (commits, refusedCommands, refusedSplit) = host.TakeGpuDryCommits();
        host.FlushCommandBuffer();
        var pixels = rig.Read(Address, Width * Height * 4);
        output.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"enabled={enabled} gpu_busy={gpuBusy} min_commands={minCommands} passes={passes} end_flush_by=[{endFlushBy}] " +
            $"commits={commits} refused_commands={refusedCommands} refused_split={refusedSplit}"));
        return new Outcome(pixels, passes, endFlushBy, commits, refusedCommands, refusedSplit);
    }

    private bool RequireDevice()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Dry commits are encoded on the device only behind the exclusive device gate.");
            return false;
        }

        Assert.True(OperatingSystem.IsMacOS());
        Assert.NotEqual(0, Device.Value);
        return true;
    }

    private static readonly Lazy<nint> Device = new(MetalNative.MTLCreateSystemDefaultDevice);
}
