// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using SharpEmu.Libs.VideoOut.Overlay;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Libs.Tests.VideoOut.Overlay;

public sealed class OverlayRendererTests(ITestOutputHelper output)
{
    /// <summary>Always hands back the same snapshot, so a measurement sees only the overlay's own cost.</summary>
    private sealed class FixedSource(OverlaySnapshot snapshot) : IOverlayDataSource
    {
        public void SetCaptureDemand(OverlayCaptureDemand demand)
        {
        }

        public OverlaySnapshot Capture() => snapshot;
    }

    private static OverlaySnapshot FixedSnapshot(string scenario = "ds")
    {
        var sim = new SimulatedOverlayDataSource(scenario, 99, realTime: false);
        sim.Advance(12000);
        return sim.Capture();
    }

    private static bool InsideAnyRegion(OverlayFrame frame, int x, int y)
    {
        foreach (var r in frame.Regions)
        {
            if (x >= r.Left && x < r.Left + r.Width && y >= r.Top && y < r.Top + r.Height)
            {
                return true;
            }
        }

        return false;
    }

    [Fact]
    public void MinimalLayoutDrawsOnlyInsideItsRegions()
    {
        using var host = OverlayTestSupport.CreateHost("ds", OverlayLayoutModel.MinimalPreset, realFonts: false);
        var frame = host.RenderOnce(1280, 720, 1.0)!;
        Assert.NotEmpty(frame.Regions);
        long inside = 0;
        for (var y = 0; y < frame.Height; y++)
        {
            for (var x = 0; x < frame.Width; x++)
            {
                var a = frame.Pixels[(y * frame.Width + x) * 4 + 3];
                if (InsideAnyRegion(frame, x, y))
                {
                    inside += a;
                }
                else
                {
                    Assert.True(a == 0, $"pixel ({x},{y}) outside every region is not transparent");
                }
            }
        }

        Assert.True(inside > 0, "nothing was drawn inside the regions");
        // Well under the whole surface: one small widget.
        var area = frame.Regions.Sum(r => (long)r.Width * r.Height);
        Assert.True(area < (long)frame.Width * frame.Height / 4);
    }

    [Fact]
    public void RegionsNeverOverlapSoABackendNeverBlendsTwice()
    {
        using var host = OverlayTestSupport.CreateHost("ds", OverlayLayoutModel.FullPreset, realFonts: false);
        var frame = host.RenderOnce(1920, 1080, 1.0)!;
        for (var i = 0; i < frame.Regions.Count; i++)
        {
            for (var j = i + 1; j < frame.Regions.Count; j++)
            {
                var a = frame.Regions[i];
                var b = frame.Regions[j];
                var overlap = a.Left < b.Left + b.Width && b.Left < a.Left + a.Width && a.Top < b.Top + b.Height && b.Top < a.Top + a.Height;
                Assert.False(overlap, $"regions {i} and {j} overlap");
            }
        }
    }

    [Fact]
    public void RegionsStayInsideTheFrame()
    {
        using var host = OverlayTestSupport.CreateHost("gta", OverlayLayoutModel.FullPreset, realFonts: false);
        var frame = host.RenderOnce(1000, 600, 1.5)!;
        foreach (var r in frame.Regions)
        {
            Assert.True(r.Left >= 0 && r.Top >= 0 && r.Left + r.Width <= frame.Width && r.Top + r.Height <= frame.Height);
        }
    }

    [Fact]
    public void RenderingTheSameSnapshotTwiceIsDeterministic()
    {
        var snapshot = FixedSnapshot();
        byte[] Render()
        {
            using var host = new OverlayHost(new FixedSource(snapshot), null, () => true, null, OverlayLayoutModel.FullPreset);
            return host.RenderOnce(1280, 720, 1.0)!.Pixels.ToArray();
        }

        Assert.True(Render().AsSpan().SequenceEqual(Render()));
    }

    [Theory]
    [InlineData("ds")]
    [InlineData("sh")]
    [InlineData("gta")]
    [InlineData("menu")]
    [InlineData("load")]
    public void EveryScenarioRendersEveryPresetWithoutThrowing(string scenario)
    {
        foreach (var preset in OverlayLayoutModel.BuiltInPresetNames)
        {
            using var host = OverlayTestSupport.CreateHost(scenario, preset, realFonts: false);
            Assert.NotNull(host.RenderOnce(1280, 720, 1.0));
        }
    }

    [Fact]
    public void AnEmptySnapshotStillRendersWithoutThrowing()
    {
        using var host = new OverlayHost(new FixedSource(OverlaySnapshot.Empty), null, () => true, null, OverlayLayoutModel.FullPreset);
        Assert.NotNull(host.RenderOnce(1280, 720, 1.0));
        host.HandleKey(OverlayKey.TogglePause);
        Assert.NotNull(host.RenderOnce(1280, 720, 1.0));
    }

    [Fact]
    public void AnInvisibleOverlayProducesNoFrame()
    {
        using var host = new OverlayHost(new FixedSource(FixedSnapshot()), null, () => false, null, OverlayLayoutModel.MinimalPreset);
        Assert.Null(host.AcquireFrame(1280, 720, 1.0));
    }

    [Fact]
    public void LeasedFramesAreNeverHandedOutAgainUntilReleased()
    {
        using var host = new OverlayHost(new FixedSource(FixedSnapshot()), null, () => true, null, OverlayLayoutModel.MinimalPreset);
        host.RenderOnce(640, 360, 1.0);
        var first = host.AcquireFrame(640, 360, 1.0);
        Assert.NotNull(first);
        for (var i = 0; i < 6; i++)
        {
            host.RenderOnce(640, 360, 1.0);
            var other = host.AcquireFrame(640, 360, 1.0);
            if (other is not null)
            {
                Assert.NotSame(first, other);
                host.ReleaseFrame(other);
            }
        }

        host.ReleaseFrame(first!);
    }

    [Fact]
    public void BuiltInBitmapFontRendersTextWhenNoFontServiceExists()
    {
        using var host = OverlayTestSupport.CreateHost("ds", OverlayLayoutModel.MinimalPreset, realFonts: false);
        var frame = host.RenderOnce(1280, 720, 1.0)!;
        // The "FPS" label and the big number must have produced bright (non-panel) pixels.
        var bright = 0;
        foreach (var r in frame.Regions)
        {
            for (var y = r.Top; y < r.Top + r.Height; y++)
            {
                for (var x = r.Left; x < r.Left + r.Width; x++)
                {
                    var o = (y * frame.Width + x) * 4;
                    if (frame.Pixels[o + 2] > 200 && frame.Pixels[o + 3] > 200)
                    {
                        bright++;
                    }
                }
            }
        }

        Assert.True(bright > 50);
    }

    [Fact]
    public void FullDebugRasterizesFastAndWithoutPerRedrawAllocation()
    {
        const int w = 2560, h = 1440;
        var snapshot = FixedSnapshot();
        using var host = new OverlayHost(new FixedSource(snapshot), new SharpEmu.Fonts.Skia.SkiaFontFactory(), () => true, null, OverlayLayoutModel.FullPreset);
        for (var i = 0; i < 40; i++)
        {
            host.RenderOnce(w, h, 2.0);
        }

        const int runs = 120;
        var times = new double[runs];
        var bytesBefore = GC.GetAllocatedBytesForCurrentThread();
        for (var i = 0; i < runs; i++)
        {
            var t = Stopwatch.GetTimestamp();
            host.RenderOnce(w, h, 2.0);
            times[i] = Stopwatch.GetElapsedTime(t).TotalMilliseconds;
        }

        var bytes = GC.GetAllocatedBytesForCurrentThread() - bytesBefore;
        Array.Sort(times);
        var median = times[runs / 2];
        output.WriteLine($"Full debug {w}x{h} @2x: median {median:F2} ms, p10 {times[runs / 10]:F2} ms, worst {times[^1]:F2} ms, {bytes / (double)runs:F0} B per redraw");
        // The target is <= 4 ms on an idle machine; the bound here leaves room for a busy CI box.
        Assert.True(median < 12.0, $"median redraw {median:F2} ms");
        Assert.True(bytes / (double)runs < 2048, $"{bytes / (double)runs:F0} B allocated per redraw");
    }

    [Fact]
    public void PerWidgetCostBreakdown()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_OVERLAY_SHOTS") != "1")
        {
            return;
        }

        var snapshot = FixedSnapshot();
        foreach (var type in Enum.GetValues<OverlayWidgetType>().Append(unchecked((OverlayWidgetType)255)))
        {
            using var host = new OverlayHost(new FixedSource(snapshot), new SharpEmu.Fonts.Skia.SkiaFontFactory(), () => true, null, OverlayLayoutModel.FullPreset);
            host.Model.Widgets.RemoveAll(w => w.Type != type);
            for (var i = 0; i < 20; i++)
            {
                host.RenderOnce(2560, 1440, 2.0);
            }

            var t = Stopwatch.GetTimestamp();
            for (var i = 0; i < 40; i++)
            {
                host.RenderOnce(2560, 1440, 2.0);
            }

            output.WriteLine($"{type}: {Stopwatch.GetElapsedTime(t).TotalMilliseconds / 40:F2} ms");
        }
    }
}
