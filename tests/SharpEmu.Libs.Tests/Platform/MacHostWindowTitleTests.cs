// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE.Host;
using SharpEmu.Platform.MacOS.Input;
using SharpEmu.Platform.MacOS.Windowing;
using Xunit;

namespace SharpEmu.Libs.Tests.Platform;

/// <summary>Window-title formatting and controller classification — the parts of
/// the native window that need no AppKit instance.</summary>
public sealed class MacHostWindowTitleTests
{
    [Fact]
    public void TitleWithoutAnOverlaySummaryIsUnchanged()
    {
        Assert.Equal(
            "SharpEmu - Application",
            MacHostWindow.FormatWindowTitle("SharpEmu - Application"));
    }

    [Fact]
    public void OverlaySummaryIsAppendedWithTheSameSeparatorAsTheSdlWindow()
    {
        Assert.Equal(
            "SharpEmu - Application · FPS 60,0",
            MacHostWindow.FormatWindowTitle("SharpEmu - Application", "FPS 60,0"));
    }

    /// <summary>
    /// RenderDoc has no macOS capture path, so unlike the SDL window the native
    /// one must never advertise an F12 capture hotkey.
    /// </summary>
    [Fact]
    public void TitleNeverAdvertisesARenderDocCaptureKey()
    {
        Assert.DoesNotContain("F12", MacHostWindow.FormatWindowTitle("SharpEmu", "FPS 60,0"));
    }

    [Theory]
    [InlineData("DualSense", HostGamepadType.DualSense)]
    [InlineData("DualShock 4", HostGamepadType.DualShock4)]
    [InlineData("Xbox One", HostGamepadType.Xbox)]
    [InlineData("MFi", HostGamepadType.Generic)]
    [InlineData(null, HostGamepadType.Generic)]
    public void ProductCategoryMapsToTheHostPadType(string? productCategory, HostGamepadType expected)
    {
        Assert.Equal(expected, MacGameControllerGamepad.MapProductCategory(productCategory));
    }
}
