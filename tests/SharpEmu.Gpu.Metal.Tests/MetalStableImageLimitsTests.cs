// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// The stable image budget a device answers (MetalStableImageLimits), from Apple's Metal Feature Set Tables: argument
/// buffers tier 2 and Apple7 or later, textures 1M per stage, samplers 996 per stage on Apple7/Apple8 and 500K on
/// Apple9 and later, never more than the unique argument-buffer samplers the device allows, buffers unlimited.
/// </summary>
public sealed class MetalStableImageLimitsTests
{
    [Theory]
    [InlineData(true, true, false, 1024ul, 996u)]
    [InlineData(true, true, false, 512ul, 512u)]
    [InlineData(true, true, true, 2048ul, 2048u)]
    [InlineData(true, true, true, 1_000_000ul, 500_000u)]
    public void ATier2Apple7DeviceAnswersItsArgumentBufferLimits(bool tier2, bool apple7, bool apple9, ulong uniqueSamplers,
        uint samplers)
    {
        var limits = MetalStableImageLimits.FromCapabilities(tier2, apple7, apple9, uniqueSamplers);
        Assert.Equal(new StableImageDescriptorLimits(1_000_000, 1_000_000, samplers, uint.MaxValue, 1_000_000), limits);
    }

    [Theory]
    [InlineData(false, true, true, 1024ul)]
    [InlineData(true, false, false, 1024ul)]
    [InlineData(true, true, true, 0ul)]
    public void ADeviceWithoutTheTablesLimitsAnswersNoBudget(bool tier2, bool apple7, bool apple9, ulong uniqueSamplers) =>
        Assert.Equal(default, MetalStableImageLimits.FromCapabilities(tier2, apple7, apple9, uniqueSamplers));

    [Fact]
    public void TheSwitchOffAnswersNoBudget() => Assert.Equal(default, MetalStableImageLimits.Of(1, enabled: false));

    [Fact]
    public void TheStampNamesTheBudget()
    {
        Assert.NotEqual(MetalStableImageLimits.StampPart(default),
            MetalStableImageLimits.StampPart(MetalStableImageLimits.FromCapabilities(true, true, false, 1024)));
        Assert.Equal("stable_images=0,0,0,0,0", MetalStableImageLimits.StampPart(default));
    }
}
