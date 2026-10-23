// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

// Two pipelines build the same Metal render descriptor exactly when their keys are equal:
// state Metal sets at encode time stays out, everything the descriptor holds stays in.
public sealed class MetalPipelineStateKeyTests
{
    [Fact]
    public void InstanceDivisorsChangeTheNativePipelineIdentity()
    {
        var first = Description(vertexInput: input => input.Bindings[0] = new(8, true, 1), attribute: true);
        var second = Description(vertexInput: input => input.Bindings[0] = new(8, true, 2), attribute: true);
        Assert.NotEqual(Key(first), Key(second));
    }
    private static readonly UInt128 Vertex = new(0, 0x11);
    private static readonly UInt128 Pixel = new(0, 0x22);
    private const uint Float2Format = 64;

    private static GraphicsPipelineDescription Description(
        Action<PipelineStaticParameters>? edit = null, Action<PipelineRenderingState>? rendering = null,
        Action<PipelineVertexInputState>? vertexInput = null, bool attribute = false)
    {
        var parameters = new PipelineStaticParameters { ColorCount = 1, Topology = GpuPrimitiveTopology.TriangleList, Samples = 1 };
        parameters.SetColorMask(0, 0xF);
        edit?.Invoke(parameters);
        var state = new PipelineRenderingState { ColorCount = 1 };
        state.ColorFormats[0] = GpuFormat.R8G8B8A8Unorm;
        rendering?.Invoke(state);
        var input = new PipelineVertexInputState();
        var info = new VertexInputInfo();
        var stage = new ShaderProgramInfo { Stage = ShaderStageKind.Vertex, Hash = 1 };
        if (attribute)
        {
            input.Bindings[0] = new PipelineVertexBinding(8, false);
            input.Attributes[0] = new PipelineVertexAttribute(0, 0);
            input.BindingCount = 1;
            input.AttributeCount = 1;
            info = new VertexInputInfo
            {
                Attributes = [new VertexAttributeResource(new BufferDescriptorWords(0, 8u << 16, 6, Float2Format << 12), 0, 2, 0, 0, 0, 0)],
            };
            stage.VertexFetchComponents[0] = 2;
        }

        vertexInput?.Invoke(input);
        return new GraphicsPipelineDescription
        {
            Rendering = state,
            VertexInput = input,
            VertexInfo = info,
            VertexProgram = default,
            VertexStage = stage,
            StaticParameters = parameters,
        };
    }

    private static MetalPipelineStateKey Key(
        GraphicsPipelineDescription description, UInt128? vertex = null, UInt128? pixel = null, bool noPixel = false,
        bool layered = false, bool rectangleList = false, nuint controlPointIndexType = 0) =>
        MetalRenderHost.PipelineStateKeyOf(
            description, vertex ?? Vertex, noPixel ? null : pixel ?? Pixel, layered, rectangleList, controlPointIndexType);

    [Fact]
    public void StateMetalSetsAtEncodeTimeDoesNotSplitAPipeline()
    {
        var baseline = Key(Description());
        var encoderState = Key(Description(p =>
        {
            p.CullFront = true;
            p.CullBack = true;
            p.FrontFaceClockwise = true;
            p.DepthClipEnable = false;
            p.NegativeOneToOne = true;
            p.PrimitiveRestartEnable = true;
            p.Topology = GpuPrimitiveTopology.TriangleStrip;
            p.WithDepth = true;
            p.DepthBoundsTestEnable = true;
            p.DepthMinBounds = 0.25f;
            p.DepthMaxBounds = 0.5f;
            p.StencilTestEnable = true;
            p.StencilFront = new StencilOperations(GpuStencilOp.Replace, GpuStencilOp.Invert, GpuStencilOp.Zero, GpuCompareOp.Less);
            p.SampleShadingEnable = true;
        }));

        Assert.Equal(baseline, encoderState);
        Assert.Equal(baseline.GetHashCode(), encoderState.GetHashCode());
    }

    [Fact]
    public void ABlendThatIsOffDoesNotSplitAPipelineAndOneThatIsOnDoes()
    {
        var off = Key(Description(p => p.SetColorSourceBlend(0, 4)));
        var otherFactorOff = Key(Description(p => p.SetColorSourceBlend(0, 9)));
        var bypassed = Key(Description(p =>
        {
            p.SetBlendEnable(0, true);
            p.SetBlendBypass(0, true);
            p.SetColorSourceBlend(0, 9);
        }));
        var on = Key(Description(p =>
        {
            p.SetBlendEnable(0, true);
            p.SetColorSourceBlend(0, 4);
            p.SetColorDestinationBlend(0, 5);
        }));
        var onOtherFactor = Key(Description(p =>
        {
            p.SetBlendEnable(0, true);
            p.SetColorSourceBlend(0, 4);
            p.SetColorDestinationBlend(0, 6);
        }));

        Assert.Equal(off, otherFactorOff);
        Assert.Equal(off, bypassed);
        Assert.NotEqual(off, on);
        Assert.NotEqual(on, onOtherFactor);
    }

    [Fact]
    public void AnIntegerAttachmentNeverBlends()
    {
        PipelineStaticParameters Blending(PipelineStaticParameters p)
        {
            p.SetBlendEnable(0, true);
            p.SetColorSourceBlend(0, 4);
            return p;
        }

        var integer = Description(p => Blending(p), state => state.ColorFormats[0] = GpuFormat.R32Uint);
        var integerOff = Description(null, state => state.ColorFormats[0] = GpuFormat.R32Uint);

        Assert.Equal(Key(integerOff), Key(integer));
    }

    [Fact]
    public void WhatTheDescriptorHoldsSplitsAPipeline()
    {
        var baseline = Key(Description());
        Assert.NotEqual(baseline, Key(Description(p => p.SetColorMask(0, 0x7))));
        Assert.NotEqual(baseline, Key(Description(p => p.Samples = 4)));
        Assert.NotEqual(baseline, Key(Description(p => p.AlphaToCoverageEnable = true)));
        Assert.NotEqual(baseline, Key(Description(null, state => state.ColorFormats[0] = GpuFormat.B8G8R8A8Srgb)));
        Assert.NotEqual(baseline, Key(Description(null, state => state.DepthFormat = GpuFormat.D32Sfloat)));
        Assert.NotEqual(baseline, Key(Description(), vertex: new UInt128(0, 0x12)));
        Assert.NotEqual(baseline, Key(Description(), pixel: new UInt128(0, 0x23)));
        Assert.NotEqual(baseline, Key(Description(), noPixel: true));
    }

    [Fact]
    public void ARectangleListIsToldApartByItsControlPointIndexType()
    {
        var none = Key(Description(), rectangleList: true, controlPointIndexType: 0);
        var sixteen = Key(Description(), rectangleList: true, controlPointIndexType: 1);
        var thirtyTwo = Key(Description(), rectangleList: true, controlPointIndexType: 2);

        Assert.NotEqual(none, sixteen);
        Assert.NotEqual(sixteen, thirtyTwo);
        Assert.NotEqual(none, Key(Description()));
        // The index type is only part of a rectangle list's descriptor.
        Assert.Equal(Key(Description()), Key(Description(), controlPointIndexType: 2));
    }

    [Fact]
    public void TheTopologyClassIsKeyedOnlyForAProgramThatNeedsItUpFront()
    {
        PipelineStaticParameters Topology(PipelineStaticParameters p, GpuPrimitiveTopology topology)
        {
            p.Topology = topology;
            return p;
        }

        Assert.Equal(
            Key(Description(p => Topology(p, GpuPrimitiveTopology.TriangleList))),
            Key(Description(p => Topology(p, GpuPrimitiveTopology.LineList))));
        Assert.NotEqual(
            Key(Description(p => Topology(p, GpuPrimitiveTopology.TriangleList)), layered: true),
            Key(Description(p => Topology(p, GpuPrimitiveTopology.LineList)), layered: true));
    }

    [Fact]
    public void TheVertexLayoutIsKeyedOnlyWhenTheDescriptorHasOne()
    {
        // Without attributes no vertex descriptor is set, whatever the bindings say.
        Assert.Equal(
            Key(Description(vertexInput: input => input.Bindings[0] = new PipelineVertexBinding(16, false))),
            Key(Description()));

        var baseline = Key(Description(attribute: true));
        Assert.NotEqual(Key(Description()), baseline);
        Assert.NotEqual(baseline, Key(Description(attribute: true, vertexInput: input => input.Bindings[0] = new PipelineVertexBinding(16, false))));
        Assert.NotEqual(baseline, Key(Description(attribute: true, vertexInput: input => input.Bindings[0] = new PipelineVertexBinding(8, true))));
        Assert.NotEqual(baseline, Key(Description(attribute: true, vertexInput: input => input.Attributes[0] = new PipelineVertexAttribute(4, 0))));
        Assert.Equal(baseline, Key(Description(attribute: true)));
    }
}
