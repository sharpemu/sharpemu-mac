// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Text.RegularExpressions;
using SharpEmu.Libs.Tests.Gpu.Scheduling;
using SharpEmu.Libs.VideoOut;
using Xunit;

namespace SharpEmu.Libs.Tests.VideoOut;

[Collection(SchedulingStateCollection.Name)]
public sealed class PerfOverlayCadenceTests
{
    [Fact]
    public void MetalGuestCadenceExcludesHostRedrawsAndUnpresentedGuestSubmissions()
    {
        using var state = new OverlayState();
        PerfOverlay.UsePresentedGuestFrameTiming();
        for (var index = 0; index < 120; index++)
        {
            PerfOverlay.RecordPresent();
            PerfOverlay.RecordSubmit();
        }
        Assert.Equal(0, ReadPublishedFps());

        PerfOverlay.RecordGuestPresent();
        for (var index = 0; index < 120; index++) PerfOverlay.RecordPresent();
        Assert.InRange(ReadPublishedFps(), .9, 1.0);
    }

    private static double ReadPublishedFps()
    {
        // Advance the reporting window without sleeping or changing production
        // timing. Assertions observe the user-visible summary, not counters.
        Field("_statsWindowStart").SetValue(null, Stopwatch.GetTimestamp() - Stopwatch.Frequency);
        var summary = PerfOverlay.GetTitleBarSummary();
        var match = Regex.Match(summary, @"^FPS ([0-9]+(?:[.,][0-9]+)?)");
        Assert.True(match.Success, summary);
        return double.Parse(match.Groups[1].Value.Replace(',', '.'), CultureInfo.InvariantCulture);
    }

    private static FieldInfo Field(string name) => typeof(PerfOverlay)
        .GetField(name, BindingFlags.Static | BindingFlags.NonPublic)!;

    private sealed class OverlayState : IDisposable
    {
        private readonly Dictionary<FieldInfo, object?> saved = typeof(PerfOverlay)
            .GetFields(BindingFlags.Static | BindingFlags.NonPublic)
            .Where(field => !field.IsLiteral && !field.IsInitOnly)
            .ToDictionary(field => field, field => field.GetValue(null));
        private readonly double[] history = (double[])((double[])Field("_frameMilliseconds").GetValue(null)!).Clone();
        private readonly PerformanceOverlayState display = (PerformanceOverlayState)Field("DisplayState").GetValue(null)!;
        private readonly HostVideoOptions displayOptions;

        public OverlayState()
        {
            displayOptions = new HostVideoOptions { OverlayEnabled = display.Enabled,
                OverlayCorner = display.Corner, OverlayMode = display.Mode };
            display.Configure(new HostVideoOptions { OverlayMode = PerformanceOverlayMode.TitleBar }, false);
            foreach (var name in new[] { "_presentedInWindow", "_submittedInWindow", "_guestPresentedInWindow",
                         "_lastSubmitTimestamp", "_lastPresentTimestamp", "_sessionStartTimestamp" })
                Field(name).SetValue(null, 0L);
            Field("_trackPresentedGuestFrames").SetValue(null, false);
            Field("_gpuUsage").SetValue(null, null);
        }

        public void Dispose()
        {
            foreach (var value in saved) value.Key.SetValue(null, value.Value);
            history.CopyTo((double[])Field("_frameMilliseconds").GetValue(null)!, 0);
            display.Configure(displayOptions, false);
        }
    }
}
