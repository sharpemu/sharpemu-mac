// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.VideoOut;
using Xunit;

namespace SharpEmu.Libs.Tests.VideoOut;

/// <summary>
/// The display a title is told about decides how much pixel work it does: it asks the system
/// what the television is and sizes its render targets from the answer. A 4K report can cost
/// four times the pixels of a 1080p one, so there is one policy, every query reads it, and by
/// default it is the output resolution the user picked.
/// </summary>
public sealed class GuestDisplayResolutionTests
{
    [Fact]
    public void TheDefaultDisplayFollowsTheDefault1080pOutput()
    {
        var options = HostVideoOptions.Default;

        Assert.Equal(GuestDisplayResolution.Auto, options.DisplayResolution);
        Assert.Equal(GuestDisplayResolution.Hd, options.EffectiveDisplayResolution);
        Assert.Equal((1920u, 1080u), options.GuestDisplaySize);
        Assert.Equal(1, options.GuestResolutionClass);
    }

    /// <summary>The output resolution the user picks is what every title is told it drives,
    /// rounded down to a mode the console has: titles size their render targets from the mode.</summary>
    [Theory]
    [InlineData(1920, 1080, 1920u, 1080u, 1)]
    [InlineData(3840, 2160, 3840u, 2160u, 2)]
    [InlineData(2560, 1440, 2560u, 1440u, 1)]
    [InlineData(1280, 720, 1280u, 720u, 1)]
    [InlineData(1920, 1200, 1920u, 1080u, 1)]
    [InlineData(2560, 1600, 2560u, 1440u, 1)]
    [InlineData(5120, 2880, 3840u, 2160u, 2)]
    [InlineData(3840, 1600, 2560u, 1440u, 1)]
    [InlineData(1366, 768, 1280u, 720u, 1)]
    [InlineData(640, 360, 1280u, 720u, 1)]
    public void AutoFollowsTheChosenOutputResolution(
        int width, int height, uint guestWidth, uint guestHeight, int resolutionClass)
    {
        var options = new HostVideoOptions { Width = width, Height = height }.Normalize();

        Assert.Equal((guestWidth, guestHeight), options.GuestDisplaySize);
        Assert.Equal(resolutionClass, options.GuestResolutionClass);
    }

    [Theory]
    [InlineData(GuestDisplayResolution.Hd720, 1280u, 720u, 1)]
    [InlineData(GuestDisplayResolution.Hd, 1920u, 1080u, 1)]
    [InlineData(GuestDisplayResolution.Qhd, 2560u, 1440u, 1)]
    [InlineData(GuestDisplayResolution.Uhd, 3840u, 2160u, 2)]
    public void EverySizeAndClassComesFromTheOneSetting(
        GuestDisplayResolution resolution, uint width, uint height, int resolutionClass)
    {
        var options = new HostVideoOptions { DisplayResolution = resolution };

        Assert.Equal((width, height), options.GuestDisplaySize);
        Assert.Equal(resolutionClass, options.GuestResolutionClass);
    }

    /// <summary>Only 4K is the 4K class; 1440p reports the same class as 1080p, which is what
    /// the console does — the class is a tier, not a size.</summary>
    [Fact]
    public void OnlyFourKReportsTheFourKClass()
    {
        Assert.Equal(
            2,
            new HostVideoOptions { DisplayResolution = GuestDisplayResolution.Uhd }.GuestResolutionClass);
        Assert.Equal(
            1,
            new HostVideoOptions { DisplayResolution = GuestDisplayResolution.Qhd }.GuestResolutionClass);
    }

    [Fact]
    public void AnUnknownSettingFallsBackToTheDefaultDisplay()
    {
        var options = new HostVideoOptions { DisplayResolution = (GuestDisplayResolution)42 }.Normalize();

        Assert.Equal(GuestDisplayResolution.Auto, options.DisplayResolution);
    }

    /// <summary>A pinned display is independent of the window: a 4K window with 1080p pinned does
    /// not tell the guest it is driving a 4K television, and the report does not shrink the window.</summary>
    [Fact]
    public void APinnedDisplayIgnoresTheWindowSize()
    {
        var options = new HostVideoOptions
        {
            Width = 3840,
            Height = 2160,
            DisplayResolution = GuestDisplayResolution.Hd,
        }.Normalize();

        Assert.Equal(3840, options.Width);
        Assert.Equal((1920u, 1080u), options.GuestDisplaySize);
        Assert.Equal(1, options.GuestResolutionClass);
    }

    /// <summary>The log line that says whether a display setting can move a title: a buffer
    /// above the advertised display is the title's own size, and the line says whether the
    /// title ever asked for the display.</summary>
    [Fact]
    public void TheRegisteredBufferVerdictSaysWhetherTheTitleAsked()
    {
        var hd = (1920u, 1080u);

        Assert.Equal("matches the advertised display", VideoOutExports.DisplayBufferVerdict(1920, 1080, hd, queried: true));
        Assert.Equal("is below the advertised display", VideoOutExports.DisplayBufferVerdict(1280, 720, hd, queried: false));
        Assert.Contains("never asked for the display", VideoOutExports.DisplayBufferVerdict(3840, 2160, hd, queried: false));
        Assert.Contains(
            "read the display and sized from something else",
            VideoOutExports.DisplayBufferVerdict(3840, 2160, hd, queried: true));
        Assert.Contains("exceeds", VideoOutExports.DisplayBufferVerdict(1920, 1200, hd, queried: true));
    }
}
