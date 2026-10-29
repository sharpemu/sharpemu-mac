// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
using SharpEmu.Libs.VideoOut;
using Xunit;

namespace SharpEmu.Libs.Tests.VideoOut;

public sealed class OverlayStatsTests
{
    private const ulong GiB = 1UL << 30;
    private const ulong MiB = 1UL << 20;

    [Theory]
    [InlineData(1.0, 1.0, 10, 10.0)]
    [InlineData(2.0, 1.0, 10, 20.0)]
    [InlineData(5.0, 1.0, 10, 50.0)]
    [InlineData(0.0, 1.0, 10, 0.0)]
    [InlineData(1.0, 0.0, 10, 0.0)]
    [InlineData(-1.0, 1.0, 10, 0.0)]
    [InlineData(50.0, 1.0, 10, 100.0)]
    public void CpuPercentIsShareOfWholeMachine(double cpuSeconds, double wallSeconds, int cores, double expected)
    {
        Assert.Equal(expected, OverlayResourceSampler.CpuPercentOfMachine(cpuSeconds, wallSeconds, cores), 9);
    }

    [Fact]
    public void SamplerUsesCpuTimeDeltasOfBothProcesses()
    {
        var guestCpu = 100.0;
        var gpuCpu = 10.0;
        var sampler = new OverlayResourceSampler(7, 8, 10, pid => pid == 7
            ? new ProcessUsage(guestCpu, 4 * GiB)
            : new ProcessUsage(gpuCpu, 2 * GiB));

        guestCpu += 3.0; // three cores' worth over a one second window
        gpuCpu += 0.8;
        var sample = sampler.Sample(1.0);

        Assert.True(sample.Split);
        Assert.Equal(30.0, sample.GuestCpuPercent, 9);
        Assert.Equal(8.0, sample.GpuProcessCpuPercent, 9);
        Assert.Equal(38.0, sample.TotalCpuPercent, 9);
        Assert.Equal(6 * GiB, sample.TotalFootprintBytes);

        guestCpu += 1.0; // the next window is measured from this reading, not from startup
        var next = sampler.Sample(2.0);
        Assert.Equal(5.0, next.GuestCpuPercent, 9);
        Assert.Equal(0.0, next.GpuProcessCpuPercent, 9);
    }

    [Fact]
    public void SingleProcessSamplerReportsOnlyItself()
    {
        var cpu = 0.0;
        var sampler = new OverlayResourceSampler(0, 8, 4, _ => new ProcessUsage(cpu, GiB));
        cpu = 1.0;
        var sample = sampler.Sample(1.0);
        Assert.False(sample.Split);
        Assert.Equal(25.0, sample.TotalCpuPercent, 9);
        Assert.Equal(GiB, sample.TotalFootprintBytes);
        Assert.Equal("25%", OverlayText.FormatCpu(sample));
    }

    [Fact]
    public void AnUnreadableGuestIsReportedAsUnknownNotZero()
    {
        var sampler = new OverlayResourceSampler(7, 8, 10, pid => pid == 7 ? null : new ProcessUsage(0, GiB));
        var sample = sampler.Sample(1.0);
        Assert.True(double.IsNaN(sample.GuestCpuPercent));
        Assert.True(double.IsNaN(sample.TotalCpuPercent));
        Assert.Null(sample.TotalFootprintBytes);
        Assert.Equal("? (guest ?, gpu 0)", OverlayText.FormatCpu(sample));
        Assert.Equal("N/A", OverlayText.FormatMemory(sample, null));
    }

    [Fact]
    public void FirstReadingOfAProcessThatAppearedLateHasNoBaseline()
    {
        var available = false;
        var cpu = 5.0;
        var sampler = new OverlayResourceSampler(7, 8, 10, pid => pid == 7 && !available ? null : new ProcessUsage(cpu, 0));
        Assert.True(double.IsNaN(sampler.Sample(1.0).GuestCpuPercent));
        available = true;
        Assert.True(double.IsNaN(sampler.Sample(1.0).GuestCpuPercent));
        cpu += 1.0;
        Assert.Equal(10.0, sampler.Sample(1.0).GuestCpuPercent, 9);
    }

    [Theory]
    [InlineData(0UL, "0 MB")]
    [InlineData(640 * MiB, "640 MB")]
    [InlineData(GiB - 1, "1024 MB")]
    [InlineData(GiB, "1.0 GB")]
    [InlineData(6 * GiB + 100 * MiB, "6.1 GB")]
    public void BytesUseGigabytesFromOneGibibyte(ulong bytes, string expected)
    {
        Assert.Equal(expected, OverlayText.FormatBytes(bytes));
    }

    [Theory]
    [InlineData(0, "00:00:00")]
    [InlineData(59, "00:00:59")]
    [InlineData(397, "00:06:37")]
    [InlineData(3661, "01:01:01")]
    [InlineData(-5, "00:00:00")]
    public void ElapsedTimeIsHoursMinutesSeconds(long seconds, string expected)
    {
        Assert.Equal(expected, OverlayText.FormatElapsed(seconds));
    }

    [Fact]
    public void TitleShowsEveryFieldInOneFormat()
    {
        var resources = new ResourceSample(true, 38.0, 30.0, 8.0, (ulong)(2.9 * GiB), (ulong)(3.2 * GiB));
        var stats = new OverlayStats(26.8, 1920, 1080, resources, 71.2, (ulong)(4.8 * GiB), 397);
        Assert.Equal(
            "FPS 26.8 (37.3 ms) | 1920x1080 | CPU 38% (guest 30, gpu 8) | GPU 71% | " +
            "MEM 6.1 GB (gpu proc 3.2, VRAM 4.8) | TIME 00:06:37",
            OverlayText.FormatTitle(stats));
    }

    [Fact]
    public void TitleOmitsWhatIsNotKnown()
    {
        var resources = new ResourceSample(false, 3.0, double.NaN, 3.0, null, 800 * MiB);
        var stats = new OverlayStats(0, 0, 0, resources, double.NaN, null, 0);
        Assert.Equal(
            "FPS 0.0 | CPU 3% | GPU N/A | MEM 800 MB | TIME 00:00:00",
            OverlayText.FormatTitle(stats));
    }

    [Fact]
    public void MemoryBelowOneGigabyteUsesMegabytesThroughout()
    {
        var resources = new ResourceSample(true, 0, 0, 0, 500 * MiB, 140 * MiB);
        Assert.Equal("640 MB (gpu proc 140, VRAM 300)", OverlayText.FormatMemory(resources, 300 * MiB));
    }

    [Fact]
    public void FormattingIgnoresTheCurrentCulture()
    {
        var previous = CultureInfo.CurrentCulture;
        try
        {
            CultureInfo.CurrentCulture = new CultureInfo("de-DE");
            var resources = new ResourceSample(true, 38.0, 30.0, 8.0, 3 * GiB, 3 * GiB);
            var title = OverlayText.FormatTitle(new OverlayStats(26.8, 0, 0, resources, 50, null, 0));
            Assert.StartsWith("FPS 26.8 (37.3 ms) | CPU 38%", title);
            Assert.Contains("MEM 6.0 GB (gpu proc 3.0)", title);
        }
        finally
        {
            CultureInfo.CurrentCulture = previous;
        }
    }

    [Fact]
    public void ProcessCanReadItsOwnUsage()
    {
        var usage = ProcessUsageReader.TryRead(Environment.ProcessId);
        Assert.NotNull(usage);
        Assert.True(usage.Value.CpuSeconds > 0);
        Assert.True(usage.Value.FootprintBytes > 0);
    }
}
