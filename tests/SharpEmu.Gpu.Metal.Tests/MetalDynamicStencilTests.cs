// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

// The stencil operations and compares a draw's depth/stencil state uses are the ones the shared recorder put into its
// dynamic state (DynamicDrawState.FrontStencilOperations / BackStencilOperations, upstream 7cf0ac568), not the ones of
// whatever draw the pipeline was resolved for.
public sealed class MetalDynamicStencilTests(Xunit.Abstractions.ITestOutputHelper output) : Scenes.SceneTestBase(output)
{
    private static readonly StencilOperations Replace = new(GpuStencilOp.Replace, GpuStencilOp.Replace, GpuStencilOp.Keep, GpuCompareOp.Always);
    private static readonly StencilOperations Invert = new(GpuStencilOp.Keep, GpuStencilOp.Invert, GpuStencilOp.Zero, GpuCompareOp.Less);

    private static PipelineStaticParameters Pipeline()
    {
        var parameters = new PipelineStaticParameters { ColorCount = 1, Topology = GpuPrimitiveTopology.TriangleList, Samples = 1 };
        parameters.StencilTestEnable = true;
        parameters.StencilFront = Replace;
        parameters.StencilBack = Replace;
        return parameters;
    }

    private static DynamicDrawState Draw(StencilOperations front, StencilOperations back) => new(
        0, 0, 64, 64, 0, 1, new ScissorRectangle(0, 0, 64, 64), 1,
        0, 0, 0, 0, true, true, GpuCompareOp.Less, false, 0, 0, 0, false, 0, 1, true,
        new StencilMasks(0xFF, 0xFF, 1), new StencilMasks(0xFF, 0xFF, 1), 1, 0xF,
        FrontStencilOperations: front, BackStencilOperations: back);

    [Fact]
    public void TheDrawsOwnStencilOperationsWinOverThePipelines()
    {
        var (front, back) = MetalRenderHost.StencilOperationsOf(Pipeline(), Draw(Invert, Replace));
        Assert.Equal(Invert, front);
        Assert.Equal(Replace, back);
    }

    [Fact]
    public void DrawsThatDifferOnlyInStencilOperationsGetTheirOwnDepthStencilStates()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var host = rig.Host;
        var pipeline = Pipeline();
        var asPipeline = host.DepthStencilStateForTest(true, true, pipeline, Draw(Replace, Replace));
        var inverted = host.DepthStencilStateForTest(true, true, pipeline, Draw(Invert, Replace));
        var backInverted = host.DepthStencilStateForTest(true, true, pipeline, Draw(Replace, Invert));
        Assert.NotEqual(0, asPipeline);
        Assert.NotEqual(asPipeline, inverted);
        Assert.NotEqual(asPipeline, backInverted);
        Assert.NotEqual(inverted, backInverted);
        Assert.Equal(inverted, host.DepthStencilStateForTest(true, true, pipeline, Draw(Invert, Replace)));
    }
}
