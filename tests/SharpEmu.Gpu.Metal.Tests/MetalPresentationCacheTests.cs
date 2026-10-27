// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

public sealed class MetalPresentationCacheTests
{
    private static readonly MetalPresentationStamp Room = new(1, 0, 0, 123, 1920, 1080, false, false, 1);

    [Fact]
    public void ReusedGuestTexturePresentsEveryNewFlipButNotEveryDisplayTick()
    {
        var cache = new MetalPresentationCache();
        for (var frame = 1; frame <= 8; frame++)
        {
            var stamp = Room with { GuestSequence = frame };
            Assert.True(cache.NeedsPresent(in stamp, false));
            cache.RecordPresented(in stamp);
            for (var displayTick = 0; displayTick < 12; displayTick++)
                Assert.False(cache.NeedsPresent(in stamp, false));
        }
    }

    [Fact]
    public void FailedDrawableAcquisitionRetriesWithoutLosingTheNewFrame()
    {
        var cache = new MetalPresentationCache();
        cache.RecordPresented(in Room);
        var next = Room with { GuestSequence = 2 };
        Assert.True(cache.NeedsPresent(in next, false));
        Assert.True(cache.NeedsPresent(in next, false));
        cache.RecordPresented(in next);
        Assert.False(cache.NeedsPresent(in next, false));
    }

    [Fact]
    public void MovieAndCpuUpdatesDoNotRequireANewGuestFlip()
    {
        var cache = new MetalPresentationCache();
        cache.RecordPresented(in Room);
        var cpu = Room with { HostSequence = 1 };
        Assert.True(cache.NeedsPresent(in cpu, false));
        cache.RecordPresented(in cpu);
        var movie = cpu with { MovieSequence = 99, Texture = 456 };
        Assert.True(cache.NeedsPresent(in movie, false));
        cache.RecordPresented(in movie);
        Assert.False(cache.NeedsPresent(in movie, false));
        var nextMovie = movie with { MovieSequence = 100 };
        Assert.True(cache.NeedsPresent(in nextMovie, false));
        Assert.True(cache.NeedsPresent(in cpu, false)); // Movie ended; restore the guest.
    }

    [Fact]
    public void DisplayChangesAndSurfaceRestorationRedrawAStationaryGuest()
    {
        var cache = new MetalPresentationCache();
        cache.RecordPresented(in Room);
        MetalPresentationStamp[] changes =
        [
            Room with { Width = 1280 }, Room with { Height = 720 },
            Room with { LayerHdr = true }, Room with { GuestHdr = true },
            Room with { Headroom = 2 }, Room with { Texture = 789 },
        ];
        foreach (var changed in changes)
            Assert.True(cache.NeedsPresent(in changed, false));
        cache.Invalidate();
        Assert.True(cache.NeedsPresent(in Room, false));
    }

    [Fact]
    public void LoadingOverlayCanAnimateWithoutAnImageUpdate()
    {
        var cache = new MetalPresentationCache();
        cache.RecordPresented(in Room);
        Assert.True(cache.NeedsPresent(in Room, true));
        Assert.False(cache.NeedsPresent(in Room, false));
    }
}
