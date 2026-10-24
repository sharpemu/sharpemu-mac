// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Pipelines;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

public sealed unsafe partial class MetalImageBudgetTests
{
    [Fact]
    public void DrawPathImageLookupCostFixture()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var original = rig.Acquire(GuestBase, ImageRole.Texture);
        var description = rig.Cache.GetImage(original).Description;
        var view = ImageViewDescription.Default with { Format = description.PixelFormat,
            Type = GpuImageViewType.Type2D, Aspect = GpuImageAspect.ColorBit,
            LayerCount = 1, Usage = GpuImageUsage.SampledBit };
        var request = new ImageRequest(description, view, ImageRole.Texture);
        for (var index = 0; index < 1000; index++) rig.Cache.FindImage(ref request);
        const int count = 20000;
        var bytes = GC.GetAllocatedBytesForCurrentThread();
        var start = System.Diagnostics.Stopwatch.GetTimestamp();
        for (var index = 0; index < count; index++) rig.Cache.FindImage(ref request);
        var ticks = System.Diagnostics.Stopwatch.GetTimestamp() - start;
        bytes = GC.GetAllocatedBytesForCurrentThread() - bytes;
        Assert.Equal(original, rig.Cache.FindImage(ref request));
        output.WriteLine($"DRAWPATH_BENCH image enabled={DrawPathCounters.Enabled} calls={count} ns_per_call={ticks * 1e9 / System.Diagnostics.Stopwatch.Frequency / count:F1} bytes_per_call={bytes / (double)count:F1}");
        output.WriteLine(rig.Cache.TakeDrawPathReport());
    }

    [Fact]
    public void DrawPathImageLookupRechecksGenerationRequestAndExactFormat()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var original = rig.Acquire(GuestBase, ImageRole.Texture);
        var description = rig.Cache.GetImage(original).Description;
        var view = ImageViewDescription.Default with { Format = description.PixelFormat,
            Type = GpuImageViewType.Type2D, Aspect = GpuImageAspect.ColorBit,
            LayerCount = 1, Usage = GpuImageUsage.SampledBit };
        var request = new ImageRequest(description, view, ImageRole.Texture);
        Assert.Equal(original, rig.Cache.FindImage(ref request));
        rig.Cache.TakeDrawPathReport();
        Assert.Equal(original, rig.Cache.FindImage(ref request));
        if (DrawPathCounters.Enabled) Assert.Contains("drawpath_image_hit=1 ", rig.Cache.TakeDrawPathReport());
        // Exactness, role and every view field are part of the request key.
        Assert.Equal(original, rig.Cache.FindImage(ref request, exactFormat: true));
        if (DrawPathCounters.Enabled) Assert.Contains("drawpath_image_changed=1 ", rig.Cache.TakeDrawPathReport());
        request.Role = ImageRole.ColorTarget;
        Assert.Equal(original, rig.Cache.FindImage(ref request, exactFormat: true));
        if (DrawPathCounters.Enabled) Assert.Contains("drawpath_image_changed=1 ", rig.Cache.TakeDrawPathReport());
        request.View = request.View with { Usage = GpuImageUsage.ColorAttachmentBit };
        Assert.Equal(original, rig.Cache.FindImage(ref request, exactFormat: true));
        if (DrawPathCounters.Enabled) Assert.Contains("drawpath_image_changed=1 ", rig.Cache.TakeDrawPathReport());
        // An unrelated insertion invalidates the cached range-query choice.
        rig.Acquire(GuestBase + 0x20000, ImageRole.Texture);
        rig.Cache.TakeDrawPathReport();
        Assert.Equal(original, rig.Cache.FindImage(ref request, exactFormat: true));
        if (DrawPathCounters.Enabled) Assert.Contains("drawpath_image_generation=1 ", rig.Cache.TakeDrawPathReport());
        // A same-address shape replacement must never resurrect the old slot.
        request.Description.Extent = new GpuExtent3D(Side / 2, Side, 1);
        var replacement = rig.Cache.FindImage(ref request, exactFormat: true);
        Assert.NotEqual(original, replacement);
        // Recent aliases stay live. Changing the request back must select the old
        // shape again, rather than reusing the most recently cached replacement.
        Assert.True(rig.Cache.TryGetImage(original, out _));
        Assert.Equal(replacement, rig.Cache.FindImage(ref request, exactFormat: true));
        request.Description = description;
        Assert.Equal(original, rig.Cache.FindImage(ref request, exactFormat: true));
    }

    [Fact]
    public void DrawPathImageLookupRejectsAnEvictedSlotGeneration()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Configure(retireFrames: 0, retireSeconds: 0, budget: Settings24 with { FixedBudget = 1 });
        var original = rig.Acquire(GuestBase, ImageRole.Texture);
        var description = rig.Cache.GetImage(original).Description;
        var view = ImageViewDescription.Default with { Format = description.PixelFormat,
            Type = GpuImageViewType.Type2D, Aspect = GpuImageAspect.ColorBit,
            LayerCount = 1, Usage = GpuImageUsage.SampledBit };
        var request = new ImageRequest(description, view, ImageRole.Texture);
        Assert.Equal(original, rig.Cache.FindImage(ref request));
        rig.Frame(seconds: 3);
        rig.Acquire(GuestBase + 0x20000, ImageRole.Texture);
        for (var frame = 0; frame < 3; frame++) rig.Frame(seconds: 0.3, touch: GuestBase + 0x20000);
        Assert.False(rig.Cache.TryGetImage(original, out _));
        var replacement = rig.Cache.FindImage(ref request);
        Assert.NotEqual(original, replacement);
        Assert.Equal(replacement, rig.Cache.FindImage(ref request));
    }

    [Fact]
    public void DrawPathImageLookupDoesNotCacheOverlapRetargeting()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var original = rig.Acquire(GuestBase, ImageRole.Texture);
        var description = rig.Cache.GetImage(original).Description;
        description.Resources = new SubresourceCount(1, 2);
        description.Data = description.Data with { Size = ImageSize * 2 };
        // Mip layout sizes include every layer, as the real builders produce.
        description.MipLayout[0].Size = ImageSize * 2;
        var view = ImageViewDescription.Default with { Format = description.PixelFormat,
            Type = GpuImageViewType.Type2DArray, Aspect = GpuImageAspect.ColorBit,
            LayerCount = 2, Usage = GpuImageUsage.SampledBit };
        var array = new ImageRequest(description, view, ImageRole.Texture);
        var arrayImage = rig.Cache.FindImage(ref array);
        var layer = new ImageRequest(description, view with
            { Type = GpuImageViewType.Type2D, LayerCount = 1 }, ImageRole.Texture);
        // The old small slot may have been replaced when the array was registered.
        layer.Description = description;
        layer.Description.Resources = SubresourceCount.Single;
        layer.Description.Data = layer.Description.Data with { Size = ImageSize };
        layer.Description.MipLayout[0].Size = ImageSize;
        var first = layer;
        Assert.Equal(arrayImage, rig.Cache.FindImage(ref first));
        rig.Cache.TakeDrawPathReport();
        var second = layer;
        Assert.Equal(arrayImage, rig.Cache.FindImage(ref second));
        Assert.Equal(first.View, second.View);
        if (DrawPathCounters.Enabled) Assert.Contains("drawpath_image_overlap=1", rig.Cache.TakeDrawPathReport());
    }
}
