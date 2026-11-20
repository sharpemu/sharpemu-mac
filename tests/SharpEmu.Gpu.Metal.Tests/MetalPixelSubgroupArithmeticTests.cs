// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.Tests.Scenes;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Pipelines;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// IShaderPipelineHost.PixelSubgroupArithmetic on Metal: pixel wave32 row scans become SIMD-group reductions
/// (Gen5MslTranslator.WaveReductions) where the device has SIMD-scoped reductions (Apple7 and later) and
/// SHARPEMU_METAL_PIXEL_SUBGROUP_ARITHMETIC is not 0. The switch decides the MSL, so it is part of the stamp that
/// the prewarm and the translation cache store their work under.
/// </summary>
public sealed class MetalPixelSubgroupArithmeticTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private const nint GpuFamilyApple7 = 1007;

    [Fact]
    public void TheSwitchIsPartOfTheTranslationStamp()
    {
        var expected = MetalRenderHost.PixelSubgroupArithmeticRequested ? 1 : 0;
        Assert.Contains($"pixel_subgroup_arithmetic={expected}", MetalRenderHost.TranslationSwitchStamp, StringComparison.Ordinal);
        Assert.Equal(
            Environment.GetEnvironmentVariable("SHARPEMU_METAL_PIXEL_SUBGROUP_ARITHMETIC") != "0",
            MetalRenderHost.PixelSubgroupArithmeticRequested);
    }

    [Fact]
    public void TheHostAnswersFromTheDevicesSimdgroupReductions()
    {
        using var rig = CreateRig();
        if (rig is null) return;

        var device = MetalNative.MTLCreateSystemDefaultDevice();
        var reductions = MetalNative.SendBool(device, MetalNative.Selector("supportsFamily:"), GpuFamilyApple7);
        IShaderPipelineHost host = rig.Host;
        Output.WriteLine($"Apple7 reductions: {reductions}, graphics subgroups: {host.GraphicsSubgroupOperationsEnabled}");
        Assert.Equal(
            reductions && host.GraphicsSubgroupOperationsEnabled && MetalRenderHost.PixelSubgroupArithmeticRequested,
            host.PixelSubgroupArithmetic);
        // Every supported Mac (Apple silicon) is Apple7 or later.
        Assert.True(reductions);
    }
}
