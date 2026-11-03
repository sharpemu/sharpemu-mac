// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Rendering;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.Rendering;

// The previous draw's pipeline is kept for a draw whose vertex tables and attachments differ only where a
// pipeline does not look: buffer addresses and record counts, attachment views and clears.
public sealed class KeptPipelineKeyTests
{
    private static PipelineKey Key(VertexInputBuffer[] buffers, VertexAttributeResource[] attributes, uint[]? interpolators = null) =>
        new(7, 3, 2, 1, 1, true, attributes, buffers, interpolators, 0x10, 0x20, 4, false, false);

    private static VertexAttributeResource Attribute(uint offset, int buffer) =>
        new(new BufferDescriptorWords(0x1000, 0, 64, 0), 0, 4, 0, 0, buffer, offset);

    [Fact]
    public void RebuiltTablesWithOtherAddressesKeepThePipeline()
    {
        var first = Key([new VertexInputBuffer(0x1000, 16, 100)], [Attribute(0, 0), Attribute(8, 0)]);
        var second = Key([new VertexInputBuffer(0x9000, 16, 3)], [Attribute(0, 0), Attribute(8, 0)]);

        Assert.True(first.SamePipelineAs(in second));
        Assert.NotEqual(first, second);
    }

    [Fact]
    public void StrideStepRateOffsetOrBufferChangeThePipeline()
    {
        var key = Key([new VertexInputBuffer(0x1000, 16, 100)], [Attribute(0, 0)]);

        Assert.False(key.SamePipelineAs(Key([new VertexInputBuffer(0x1000, 32, 100)], [Attribute(0, 0)])));
        Assert.False(key.SamePipelineAs(Key([new VertexInputBuffer(0x1000, 16, 100, PerInstance: true)], [Attribute(0, 0)])));
        Assert.False(key.SamePipelineAs(Key([new VertexInputBuffer(0x1000, 16, 100)], [Attribute(4, 0)])));
        Assert.False(key.SamePipelineAs(Key([new VertexInputBuffer(0x1000, 16, 100), new VertexInputBuffer(0x2000, 16, 100)], [Attribute(0, 1)])));
        Assert.False(key.SamePipelineAs(Key([new VertexInputBuffer(0x1000, 16, 100)], [Attribute(0, 0), Attribute(4, 0)])));
    }

    [Fact]
    public void InterpolatorsAndStampsAreComparedWhole()
    {
        VertexInputBuffer[] buffers = [new(0x1000, 16, 100)];
        VertexAttributeResource[] attributes = [Attribute(0, 0)];
        var key = Key(buffers, attributes, [1, 2, 3]);

        Assert.True(key.SamePipelineAs(Key(buffers, attributes, [1, 2, 3])));
        Assert.False(key.SamePipelineAs(Key(buffers, attributes, [1, 2, 4])));
        Assert.False(key.SamePipelineAs(Key(buffers, attributes)));
        Assert.False(key.SamePipelineAs(key with { ContextStamp = 8 }));
        Assert.False(key.SamePipelineAs(key with { DynamicStamp = 4 }));
        Assert.False(key.SamePipelineAs(key with { PixelProgram = 0x21 }));
    }

    [Fact]
    public void AttachmentViewsAndClearsDoNotChangeThePipelineButFormatsDo()
    {
        var first = Rendering(new GpuImageHandle(1), GpuFormat.R8G8B8A8Unorm, clear: false);
        var second = Rendering(new GpuImageHandle(2), GpuFormat.R8G8B8A8Unorm, clear: true);
        var other = Rendering(new GpuImageHandle(1), GpuFormat.R16G16B16A16Sfloat, clear: false);

        Assert.True(PipelineKey.SamePipelineRendering(in first, in second, 1));
        Assert.False(first.Equals(second));
        Assert.False(PipelineKey.SamePipelineRendering(in first, in other, 1));
        var multisampled = first;
        multisampled.Samples = 4;
        Assert.False(PipelineKey.SamePipelineRendering(in first, in multisampled, 1));
    }

    private static RenderingState Rendering(GpuImageHandle view, GpuFormat format, bool clear)
    {
        var rendering = new RenderingState { Width = 64, Height = 64, Layers = 1, Samples = 1, ColorAttachmentCount = 1 };
        rendering.ColorAttachments[0] = new RenderingAttachment(view, format, clear ? 1u : 0u, 0, 0, 0, clear, false, false, false, false);
        return rendering;
    }
}
