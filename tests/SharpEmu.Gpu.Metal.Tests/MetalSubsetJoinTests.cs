// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// A draw into some of the open pass's colour attachments is drawn with a pipeline made for all
/// of the pass's formats: its own attachments keep their masks and blending at the pass's indices
/// of them, and the rest are not written.
/// </summary>
public sealed class MetalSubsetJoinTests
{
    [Fact]
    public void TheJoinedDescriptionPlacesTheDrawsTargetsAtThePassIndices()
    {
        var parameters = new PipelineStaticParameters { ColorCount = 2, Topology = GpuPrimitiveTopology.TriangleList, Samples = 1 };
        parameters.SetColorMask(0, 0xF);
        parameters.SetColorMask(1, 0x3);
        parameters.SetBlendEnable(1, true);
        parameters.SetColorSourceBlend(1, 4);
        parameters.SetColorDestinationBlend(1, 5);
        parameters.SetAlphaSourceBlend(1, 6);
        var rendering = new PipelineRenderingState
        {
            ColorCount = 2,
            DepthFormat = GpuFormat.D32SfloatS8Uint,
            StencilFormat = GpuFormat.D32SfloatS8Uint,
        };
        rendering.ColorFormats[0] = GpuFormat.R8G8B8A8Unorm;
        rendering.ColorFormats[1] = GpuFormat.R8G8B8A8Unorm;
        var description = new GraphicsPipelineDescription
        {
            Rendering = rendering,
            VertexInput = new PipelineVertexInputState(),
            VertexInfo = new VertexInputInfo(),
            VertexProgram = new ShaderProgram(7, 7),
            VertexStage = new ShaderProgramInfo { Stage = ShaderStageKind.Vertex, Hash = 7 },
            StaticParameters = parameters,
        };
        var open = new RenderingState { Width = 64, Height = 8, Layers = 1, ColorAttachmentCount = 4, Samples = 1 };
        var formats = new[] { GpuFormat.R8G8B8A8Unorm, GpuFormat.R16G16Sfloat, GpuFormat.R8G8B8A8Unorm, GpuFormat.R8G8B8A8Unorm };
        for (var index = 0; index < formats.Length; index++)
        {
            open.ColorAttachments[index] = new RenderingAttachment(
                new GpuImageHandle((ulong)index + 1), formats[index],
                0, 0, 0, 0, false, false, false, false, false);
        }

        // The draw's attachment 0 is the pass's 0 and its 1 is the pass's 3.
        const uint placement = 1u | (4u << 4);
        var joined = MetalRenderHost.SubsetJoinedDescription(description, in open, placement, placement);

        Assert.Equal(4u, joined.Rendering.ColorCount);
        for (var index = 0; index < formats.Length; index++)
        {
            Assert.Equal(formats[index], joined.Rendering.ColorFormats[index]);
        }

        Assert.Equal(GpuFormat.D32SfloatS8Uint, joined.Rendering.DepthFormat);
        Assert.Equal(4u, joined.StaticParameters.ColorCount);
        Assert.Equal(0xFu, joined.StaticParameters.GetColorMask(0));
        Assert.Equal(0u, joined.StaticParameters.GetColorMask(1));
        Assert.Equal(0u, joined.StaticParameters.GetColorMask(2));
        Assert.Equal(0x3u, joined.StaticParameters.GetColorMask(3));
        Assert.False(joined.StaticParameters.GetBlendEnable(0));
        Assert.False(joined.StaticParameters.GetBlendEnable(1));
        Assert.True(joined.StaticParameters.GetBlendEnable(3));
        Assert.Equal((byte)4, joined.StaticParameters.GetColorSourceBlend(3));
        Assert.Equal((byte)5, joined.StaticParameters.GetColorDestinationBlend(3));
        Assert.Equal((byte)6, joined.StaticParameters.GetAlphaSourceBlend(3));
        Assert.Equal(placement, joined.ColorRemap);
        Assert.Equal(description.VertexProgram, joined.VertexProgram);
    }

    [Fact]
    public void AlphaToCoverageKeepsTheFirstOutputInPlace()
    {
        var parameters = new PipelineStaticParameters { ColorCount = 1, Samples = 1 };
        // The draw's first output at the pass's index 0, then at index 1.
        Assert.False(MetalRenderHost.MovesCoverageSource(parameters, 0x1));
        Assert.False(MetalRenderHost.MovesCoverageSource(parameters, 0x2));

        parameters.AlphaToCoverageEnable = true;
        Assert.False(MetalRenderHost.MovesCoverageSource(parameters, 0x21));
        Assert.True(MetalRenderHost.MovesCoverageSource(parameters, 0x2));
        Assert.True(MetalRenderHost.MovesCoverageSource(parameters, 0x12));
    }

    [Fact]
    public void ARemapIsPartOfThePipelineKey()
    {
        var parameters = new PipelineStaticParameters { ColorCount = 1, Samples = 1 };
        GraphicsPipelineKey Key(uint remap) => new()
        {
            Rendering = new PipelineRenderingState { ColorCount = 1 },
            VertexProgramId = 1,
            PixelProgramId = 2,
            ColorRemap = remap,
            VertexInput = new PipelineVertexInputState(),
            StaticParameters = parameters,
        };

        Assert.Equal(Key(0), Key(0));
        Assert.NotEqual(Key(0), Key(3));
        Assert.Equal(Key(3), Key(3));
    }
}
