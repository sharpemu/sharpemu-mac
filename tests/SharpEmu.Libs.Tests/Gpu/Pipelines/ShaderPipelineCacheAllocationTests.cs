// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Tests.Gpu.Rendering;
using SharpEmu.Libs.Tests.Gpu.Scheduling;
using Xunit;
using Xunit.Abstractions;
using static SharpEmu.Libs.Tests.Gpu.Rendering.RenderExecutorFixtures;

namespace SharpEmu.Libs.Tests.Gpu.Pipelines;

// What a draw whose programs are cached allocates to resolve them: the per-draw results it hands
// on (GraphicsPrograms and its vertex and pixel inputs, which carry the draw's stage resources),
// and nothing a previous draw that resolved alike already made.
[Collection(SchedulingStateCollection.Name)]
public sealed class ShaderPipelineCacheAllocationTests(ITestOutputHelper output) : IDisposable
{
    private readonly FatalScope _fatal = new();

    public void Dispose() => _fatal.Dispose();

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ACachedDrawAllocatesOnlyItsResults(bool pixelActive)
    {
        var guest = new PipelineTestGuest();
        var vertexAddress = PipelineTestGuest.MemoryBase + 0x1000;
        var pixelAddress = PipelineTestGuest.MemoryBase + 0x2000;
        guest.RegisterProgram(vertexAddress, PipelineTestGuest.MemoryBase + 0x8000, PipelineTestGuest.EndProgram,
            userDataAddress: PipelineTestGuest.MemoryBase + 0x9000);
        guest.RegisterProgram(pixelAddress, PipelineTestGuest.MemoryBase + 0x8100, PipelineTestGuest.EndProgram);
        var cache = new ShaderPipelineCache(guest.Context, guest.Host, guest.Compiler, guest.Registry);
        var banks = Banks();
        banks.Shader.Vertex.ExportAddress = vertexAddress;
        banks.Shader.Pixel.Address = pixelAddress;
        var mappings = Enumerable.Repeat(ColorComponentMap.Identity, 8).ToArray();
        long Draw()
        {
            var before = GC.GetAllocatedBytesForCurrentThread();
            Assert.True(cache.GetGraphicsPrograms(banks.Shader.Vertex, banks.Shader.Pixel, banks.Context.ShaderInterface,
                banks.Context, mappings, pixelActive, depthBound: false, depthAspectBound: false).Available);
            return GC.GetAllocatedBytesForCurrentThread() - before;
        }

        var perDraw = Enumerable.Range(0, 4).Select(_ => Draw()).ToArray();
        output.WriteLine($"pixel_active={pixelActive} per draw (bytes): {string.Join(' ', perDraw)}");
        // GraphicsPrograms, VertexInputInfo and, with a pixel stage, PixelInputInfo: about 130 bytes each, and 32 more for each
        // input's stage, whose shader-call fields (ShaderStageResources.CallArguments and the rest) stay null for a draw.
        // The upstream merge (tessellation, frontend and kept-pipeline fields) grew these to 520 and 880 bytes.
        Assert.True(perDraw[^1] <= (pixelActive ? 880 : 520), $"bytes={perDraw[^1]}");
        var first = cache.GetGraphicsPrograms(banks.Shader.Vertex, banks.Shader.Pixel, banks.Context.ShaderInterface,
            banks.Context, mappings, pixelActive, depthBound: false, depthAspectBound: false);
        var second = cache.GetGraphicsPrograms(banks.Shader.Vertex, banks.Shader.Pixel, banks.Context.ShaderInterface,
            banks.Context, mappings, pixelActive, depthBound: false, depthAspectBound: false);
        // Equal arrays are shared; a resolved pixel input is the draw's own, an inactive stage's is shared.
        Assert.Equal(!pixelActive, ReferenceEquals(first.PixelInput, second.PixelInput));
        Assert.NotSame(first.VertexInput, second.VertexInput);
        Assert.Same(first.PixelInput.InterpolatorSettings, second.PixelInput.InterpolatorSettings);
        Assert.Same(first.PixelInput.TargetOutputModes, second.PixelInput.TargetOutputModes);
        Assert.Same(first.PixelInput.TargetExportMappings, second.PixelInput.TargetExportMappings);
    }
}
