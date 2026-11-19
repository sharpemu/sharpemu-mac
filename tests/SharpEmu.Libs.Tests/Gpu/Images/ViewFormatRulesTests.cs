// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Gpu.Vulkan;
using SharpEmu.Libs.Tests.Gpu.Scheduling;
using Silk.NET.Vulkan;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.Images;

[Collection(SchedulingStateCollection.Name)]
public sealed class ViewFormatRulesTests
{
    [Theory]
    [InlineData(GpuFormat.R8G8B8A8Unorm, GpuFormat.B8G8R8A8Srgb, true)]
    [InlineData(GpuFormat.R8G8B8A8Unorm, GpuFormat.R32Uint, true)]
    [InlineData(GpuFormat.R8G8B8A8Unorm, GpuFormat.R16G16B16A16Sfloat, false)]
    [InlineData(GpuFormat.BC1RgbaUnormBlock, GpuFormat.R32G32Uint, true)]
    [InlineData(GpuFormat.R32G32Uint, GpuFormat.BC1RgbaUnormBlock, false)]
    [InlineData(GpuFormat.BC7UnormBlock, GpuFormat.R32G32B32A32Uint, true)]
    [InlineData(GpuFormat.D32Sfloat, GpuFormat.D32Sfloat, true)]
    [InlineData(GpuFormat.D32Sfloat, GpuFormat.R32Sfloat, false)]
    [InlineData(GpuFormat.R8Unorm, GpuFormat.Undefined, false)]
    public void Compatibility_IsTheSubsetRule(GpuFormat baseFormat, GpuFormat viewFormat, bool expected) =>
        Assert.Equal(expected, ViewFormatRules.AreCompatible(baseFormat, viewFormat));

    [Fact]
    public void Aspects_FollowTheFormat()
    {
        Assert.Equal(GpuImageAspect.DepthBit, ViewFormatRules.DepthAspects(GpuFormat.D32Sfloat));
        Assert.Equal(GpuImageAspect.DepthBit | GpuImageAspect.StencilBit, ViewFormatRules.DepthAspects(GpuFormat.D24UnormS8Uint));
        Assert.Equal(GpuImageAspect.StencilBit, ViewFormatRules.FullAspects(GpuFormat.S8Uint));
        Assert.Equal(GpuImageAspect.DepthBit, ViewFormatRules.FullAspects(GpuFormat.X8D24UnormPack32));
        Assert.Equal(GpuImageAspect.ColorBit, ViewFormatRules.FullAspects(GpuFormat.R8G8B8A8Unorm));

        using var fatal = new FatalScope();
        Assert.Throws<SchedulerFatalException>(() => ViewFormatRules.DepthAspects(GpuFormat.R8Unorm));
        Assert.Contains(fatal.Messages, message => message.Contains("not a depth/stencil format"));
    }

    [Fact]
    public void Swizzles_DecodeToComponentMappings()
    {
        var identity = ViewFormatRules.PackDestinationSelect(4, 5, 6, 7);
        Assert.True(ViewFormatRules.IsValidSwizzle(identity));
        Assert.False(ViewFormatRules.IsValidSwizzle(ViewFormatRules.PackDestinationSelect(2, 5, 6, 7)));
        Assert.False(ViewFormatRules.IsValidSwizzle(0x1000));
        Assert.Equal(5u, ViewFormatRules.DestinationSelect(identity, 1));

        var mapping = ViewFormatRules.ComponentMapping(ViewFormatRules.PackDestinationSelect(6, 5, 4, 1));
        Assert.Equal((GpuComponentSwizzle.B, GpuComponentSwizzle.G, GpuComponentSwizzle.R, GpuComponentSwizzle.One), (mapping.R, mapping.G, mapping.B, mapping.A));
        Assert.True(ViewFormatRules.IsComponentSwizzle(GpuComponentSwizzle.A));
        Assert.False(ViewFormatRules.IsComponentSwizzle((GpuComponentSwizzle)7));
    }

    [Fact]
    public void ColorViews_AreSelectedOrRejected()
    {
        var identity = ViewFormatRules.PackDestinationSelect(4, 5, 6, 7);
        Assert.Equal(identity, ViewFormatRules.SelectSampledColorView(GpuFormat.R8G8B8A8Unorm, GpuFormat.B8G8R8A8Unorm, identity));
        Assert.Equal(GpuFormat.R8G8B8A8Unorm, ViewFormatRules.SrgbStorageFormat(GpuFormat.B8G8R8A8Srgb));
        Assert.Equal(GpuFormat.Undefined, ViewFormatRules.SrgbStorageFormat(GpuFormat.R8G8B8A8Unorm));
        ViewFormatRules.ValidateStorageColorView(GpuFormat.R8G8B8A8Srgb, GpuFormat.R8G8B8A8Unorm, identity);

        using var fatal = new FatalScope();
        Assert.Throws<SchedulerFatalException>(() => ViewFormatRules.SelectSampledColorView(GpuFormat.R8G8B8A8Unorm, GpuFormat.R16G16B16A16Sfloat, identity));
        Assert.Throws<SchedulerFatalException>(() => ViewFormatRules.SelectSampledColorView(GpuFormat.R8G8B8A8Unorm, GpuFormat.R8G8B8A8Unorm, 0x1000));
        Assert.Throws<SchedulerFatalException>(() => ViewFormatRules.ValidateStorageColorView(GpuFormat.R8G8B8A8Unorm, GpuFormat.R16Unorm, identity));
        Assert.Equal(3, fatal.Messages.Count);
        Assert.Contains(fatal.Messages, message => message.Contains("sampled color view is not supported"));
        Assert.Contains(fatal.Messages, message => message.Contains("storage color view is not supported"));
    }

    [Fact]
    public void DepthViews_AcceptOnlyTheReplicatedSwizzles()
    {
        var replicated = ViewFormatRules.PackDestinationSelect(4, 4, 4, 4);
        Assert.True(ViewFormatRules.IsSupportedSampledDepthView(GpuFormat.D32Sfloat, GpuFormat.R32Sfloat, replicated));
        Assert.True(ViewFormatRules.IsSupportedSampledDepthView(GpuFormat.D16Unorm, GpuFormat.R16Unorm, ViewFormatRules.PackDestinationSelect(4, 0, 0, 1)));
        Assert.True(ViewFormatRules.IsSupportedSampledDepthView(GpuFormat.D16Unorm, GpuFormat.R16Unorm, ViewFormatRules.PackDestinationSelect(4, 4, 4, 1)));
        Assert.True(ViewFormatRules.IsSupportedSampledDepthView(GpuFormat.D32Sfloat, GpuFormat.R32Sfloat, ViewFormatRules.PackDestinationSelect(4, 5, 6, 7)));
        Assert.False(ViewFormatRules.IsSupportedSampledDepthView(GpuFormat.D32Sfloat, GpuFormat.R32Sfloat, ViewFormatRules.PackDestinationSelect(5, 0, 0, 1)));
        Assert.False(ViewFormatRules.IsSupportedSampledDepthView(GpuFormat.R32Sfloat, GpuFormat.R32Sfloat, replicated));
        Assert.Equal(replicated, ViewFormatRules.SelectSampledDepthView(GpuFormat.D32SfloatS8Uint, GpuFormat.R32Uint, replicated));
        Assert.True(ViewFormatRules.IsDepthCompatible(GpuFormat.R16Unorm));
        Assert.True(ViewFormatRules.IsStencilViewFormat(GpuFormat.R8Uint));

        using var fatal = new FatalScope();
        Assert.Throws<SchedulerFatalException>(() => ViewFormatRules.SelectSampledDepthView(GpuFormat.D32Sfloat, GpuFormat.R8Unorm, replicated));
        Assert.Contains(fatal.Messages, message => message.Contains("sampled depth view is not supported"));
    }

    [Fact]
    public void ViewDescriptions_CompareEveryField()
    {
        var mapping = new ComponentMapping(ComponentSwizzle.R, ComponentSwizzle.G, ComponentSwizzle.B, ComponentSwizzle.A);
        var left = ImageViewDescription.Default with { Format = GpuFormat.R8G8B8A8Unorm, Mapping = mapping.ToNeutral() };
        var same = ImageViewDescription.Default with { Format = GpuFormat.R8G8B8A8Unorm, Mapping = mapping.ToNeutral() };
        var swizzled = left with { Mapping = new GpuComponentMapping(GpuComponentSwizzle.B, GpuComponentSwizzle.G, GpuComponentSwizzle.R, GpuComponentSwizzle.A) };
        Assert.Equal(left, same);
        Assert.Equal(left.GetHashCode(), same.GetHashCode());
        Assert.NotEqual(left, swizzled);
        Assert.NotEqual(left, left with { LevelCount = 2 });
        Assert.NotEqual(left, left with { Usage = GpuImageUsage.StorageBit });
    }
}
