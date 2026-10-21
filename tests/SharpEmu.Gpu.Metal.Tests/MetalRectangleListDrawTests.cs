// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.HLE.Host;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler.Metal;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// Rectangle-list draws through the v2 render host: every complete group of three
/// vertices is one whole rectangle, its fourth corner derived from the other three, for
/// automatic and indexed draws, with a first vertex or vertex offset, and with the corners
/// fetched from a vertex buffer. Rectangle r of the test program covers the r-th half of
/// the target; red runs 0..1 across it and green 0..1 up it. Device work is behind the
/// exclusive device gate.
/// </summary>
public sealed unsafe class MetalRectangleListDrawTests(ITestOutputHelper output)
{
    private const ulong GuestBase = 0x2_0000_0000;
    private const ulong TargetAddress = GuestBase + 0x4000;
    private const uint Width = 64;
    private const uint Height = 8;
    private const uint HalfWidth = Width / 2;
    private const ulong ImageBytes = Width * Height * 4;
    private const uint Float2Format = 64;

    // Positions for two rectangles, the second at vertices 3..5.
    private static readonly float[] Positions = [-1, -1, 0, -1, -1, 1, 0, -1, 1, -1, 0, 1];

    [Theory]
    [InlineData(3u, 0u, true, false)]
    [InlineData(6u, 0u, true, true)]
    [InlineData(4u, 0u, true, false)]
    [InlineData(2u, 0u, false, false)]
    [InlineData(3u, 3u, false, true)]
    public void AnAutomaticDrawFillsEachCompleteRectangle(uint count, uint firstVertex, bool left, bool right)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value, fetched: false);
        var pixels = rig.Render(host => host.Draw(count, 1, firstVertex, 0));
        AssertHalf(pixels, 0, left);
        AssertHalf(pixels, 1, right);
    }

    // The first group puts the right-angle corner last; the second draws the first
    // rectangle's vertices offset onto the second.
    [Theory]
    [InlineData(new ushort[] { 4, 5, 3, 1, 2, 0 }, 0, true, true)]
    [InlineData(new ushort[] { 1, 2, 0 }, 3, false, true)]
    public void AnIndexedDrawFillsEachRectangle(ushort[] indices, int vertexOffset, bool left, bool right)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value, fetched: false);
        var pixels = rig.Render(host =>
        {
            rig.BindIndices(host, indices);
            ((IRenderHost)host).DrawIndexed((uint)indices.Length, 1, 0, vertexOffset, 0);
        });
        AssertHalf(pixels, 0, left);
        AssertHalf(pixels, 1, right);
    }

    [Fact]
    public void FetchedCornersFollowTheFirstVertexAndTheVertexOffset()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value, fetched: true);
        var automatic = rig.Render(host => host.Draw(3, 1, 3, 0));
        AssertHalf(automatic, 0, false);
        AssertHalf(automatic, 1, true);

        var indexed = rig.Render(host =>
        {
            rig.BindIndices(host, [0, 1, 2]);
            ((IRenderHost)host).DrawIndexed(3, 1, 0, 3, 0);
        });
        AssertHalf(indexed, 0, false);
        AssertHalf(indexed, 1, true);

        var both = rig.Render(host => host.Draw(6, 1, 0, 0));
        AssertHalf(both, 0, true);
        AssertHalf(both, 1, true);
    }

    // An indexed indirect draw whose counts stay on the GPU: the pass rebases the record's
    // indices into control points and patch arguments (MetalRenderHost.RectangleRebase.cs).
    // Only whole rectangles inside the record's count, bounded by the bound index count, are
    // drawn, from its first index and vertex offset, with or without a vertex buffer.
    [Theory]
    [InlineData(false, 6u, 1u, 0u, 0, 9u, true, true)]
    [InlineData(false, 3u, 1u, 3u, 0, 9u, false, true)]
    [InlineData(false, 5u, 1u, 0u, 0, 9u, true, false)]
    [InlineData(false, 6u, 1u, 0u, 0, 3u, true, false)]
    [InlineData(false, 3u, 1u, 6u, 3, 9u, false, true)]
    [InlineData(false, 0u, 1u, 0u, 0, 9u, false, false)]
    [InlineData(false, 6u, 0u, 0u, 0, 9u, false, false)]
    [InlineData(true, 6u, 1u, 0u, 0, 9u, true, true)]
    [InlineData(true, 3u, 1u, 3u, 0, 9u, false, true)]
    [InlineData(true, 3u, 1u, 6u, 3, 9u, false, true)]
    [InlineData(true, 6u, 2u, 0u, 0, 3u, true, false)]
    public void AnIndirectDrawWhoseCountsStayOnTheGpuFillsTheRecordsRectangles(
        bool fetched, uint indexCount, uint instanceCount, uint firstIndex, int vertexOffset, uint indexLimit,
        bool left, bool right)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value, fetched);
        var pixels = rig.Render(host =>
        {
            // Indices 6..8 name the first rectangle's vertices, right-angle corner last.
            rig.BindIndices(host, [0, 1, 2, 3, 4, 5, 1, 2, 0]);
            var record = new BufferBinding((ulong)rig.Record([indexCount, instanceCount, firstIndex, unchecked((uint)vertexOffset), 0]), 0);
            var pipeline = rig.Pipeline;
            host.PrepareIndirectDraw(in pipeline, 0x1000, record, indexed: true, indexLimit);
            ((IRenderHost)host).DrawIndexedIndirect(record);
        });
        AssertHalf(pixels, 0, left);
        AssertHalf(pixels, 1, right);
    }

    // A pipeline of any other topology, through the same program, is an ordinary vertex
    // pipeline: it builds without a mesh and draws the guest's three vertices as one triangle.
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ATriangleListPipelineDrawsOneTriangleOfTheRectangle(bool indexed)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value, fetched: false, GpuPrimitiveTopology.TriangleList);
        var pixels = rig.Render(host =>
        {
            if (indexed)
            {
                rig.BindIndices(host, [0, 1, 2]);
                ((IRenderHost)host).DrawIndexed(3, 1, 0, 0, 0);
            }
            else
            {
                host.Draw(3, 1, 0, 0);
            }
        });

        Assert.Equal(0xFF, pixels[(int)((((Height - 1) * Width) + 0) * 4) + 3]);
        Assert.Equal(0, pixels[(int)(((0 * Width) + HalfWidth - 1) * 4) + 3]);
        AssertHalf(pixels, 1, false);
    }

    // The states for an indexed draw's control points are built by the first draw that uses
    // them; an automatic rectangle list only ever builds the plain one.
    [Fact]
    public void AnIndexedRectangleStateIsBuiltByTheFirstDrawThatUsesIt()
    {
        if (!RequireDevice() || !MetalRenderHost.PipelineDedupe) return;
        using var rig = new Rig(Device.Value, fetched: false);
        Assert.Equal(0, rig.Host.DedupeCounters.SharedStates);

        void Indexed(MetalRenderHost host)
        {
            rig.BindIndices(host, [4, 5, 3, 1, 2, 0]);
            ((IRenderHost)host).DrawIndexed(6, 1, 0, 0, 0);
        }

        var automatic = rig.Render(host => host.Draw(6, 1, 0, 0));
        AssertHalf(automatic, 0, true);
        AssertHalf(automatic, 1, true);
        Assert.Equal(1, rig.Host.DedupeCounters.SharedStates);

        var indexed = rig.Render(Indexed);
        AssertHalf(indexed, 0, true);
        AssertHalf(indexed, 1, true);
        Assert.Equal(2, rig.Host.DedupeCounters.SharedStates);

        rig.Render(Indexed);
        Assert.Equal(2, rig.Host.DedupeCounters.SharedStates);
    }

    // Pipelines that differ only in state Metal sets on the encoder build one descriptor: the
    // second takes the first's state, and each library is compiled once for its source.
    // Evicting one pipeline leaves the state the other draws with alive.
    [Fact]
    public void PipelinesOfOneDescriptorShareAStateThatOutlivesAnEvictedOwner()
    {
        if (!RequireDevice() || !MetalRenderHost.PipelineDedupe) return;
        using var rig = new Rig(Device.Value, fetched: false, GpuPrimitiveTopology.TriangleList);
        var clipped = rig.CreatePipeline();
        var clamped = rig.CreatePipeline(depthClip: false);
        var made = rig.Host.DedupeCounters;
        Assert.Equal(2, made.SharedLibraries);
        Assert.Equal(2, made.ModuleSameSource);
        Assert.Equal(1, made.SharedStates);
        Assert.Equal(1, made.PipelineDedups);
        Assert.Equal(2, rig.Host.ShaderCacheCounters.ResidentPipelines);

        void Draw(MetalRenderHost host) => host.Draw(3, 1, 0, 0);
        var first = rig.Render(Draw, clipped);
        Assert.Equal(first, rig.Render(Draw, clamped));
        Assert.Equal(0xFF, first[(int)((((Height - 1) * Width) + 0) * 4) + 3]);

        long AnHourFromNow() => System.Diagnostics.Stopwatch.GetTimestamp() + 3600 * System.Diagnostics.Stopwatch.Frequency;
        var (evicted, _) = rig.Host.TrimShaderCache(rig.Host.CompletedSerial, AnHourFromNow(), new MetalShaderCacheLimits(1, 0, 0, 0, 0));
        Assert.Equal(1, evicted);
        Assert.Equal(1, rig.Host.ShaderCacheCounters.ResidentPipelines);
        Assert.Equal(1, rig.Host.DedupeCounters.SharedStates);

        // The survivor still draws, and the evicted one is made again from the same state.
        Assert.Equal(first, rig.Render(Draw, clipped));
        Assert.Equal(first, rig.Render(Draw, clamped));
        Assert.Equal(2, rig.Host.ShaderCacheCounters.ResidentPipelines);
        Assert.Equal(1, rig.Host.DedupeCounters.SharedStates);
        Assert.Equal(2, rig.Host.DedupeCounters.PipelineDedups);

        // Only the last owner's release lets the state go.
        var (released, _) = rig.Host.TrimShaderCache(rig.Host.CompletedSerial, AnHourFromNow(), new MetalShaderCacheLimits(0, 1, 0, 1, 0));
        Assert.Equal(2, released);
        Assert.Equal(0, rig.Host.DedupeCounters.SharedStates);
        Assert.Equal(0, rig.Host.DedupeCounters.SharedLibraries);
    }

    private static void AssertHalf(byte[] pixels, uint half, bool filled)
    {
        for (var y = 0u; y < Height; y++)
        {
            for (var x = half * HalfWidth; x < (half + 1) * HalfWidth; x++)
            {
                var texel = (int)(((y * Width) + x) * 4);
                if (!filled)
                {
                    Assert.True(pixels[texel + 3] == 0, $"half {half} texel ({x}, {y}) is covered");
                    continue;
                }

                Assert.True(pixels[texel + 3] == 0xFF, $"half {half} texel ({x}, {y}) is uncovered");
                var u = (x - (half * HalfWidth) + 0.5f) / HalfWidth;
                var v = 1.0f - ((y + 0.5f) / Height);
                Assert.True(Math.Abs((pixels[texel] / 255f) - u) < 0.02f, $"texel ({x}, {y}) red {pixels[texel]} for {u}");
                Assert.True(Math.Abs((pixels[texel + 1] / 255f) - v) < 0.02f, $"texel ({x}, {y}) green {pixels[texel + 1]} for {v}");
            }
        }
    }

    // In the translator's own shape (Gen5MslRectangleList reads it): corner c = id % 3 of
    // rectangle r = id / 3 is the bottom-left, bottom-right or top-left of the r-th half,
    // and PARAM0 is (c == 1, c == 2), so it reaches (1, 1) only at the derived corner.
    private static string VertexSource(bool fetched) => $$"""
        #include <metal_stdlib>
        using namespace metal;
        {{(fetched ? "struct Gen5VsIn\n{\n    float2 in0 [[attribute(0)]];\n};\n" : string.Empty)}}
        struct Gen5VsOut
        {
            float4 sharpemu_position [[position]];
            float4 param0 [[user(locn0)]];
        };

        vertex Gen5VsOut gen5_vs(
        {{(fetched ? "    Gen5VsIn sharpemu_vin [[stage_in]],\n" : string.Empty)}}    uint sharpemu_vertex_id [[vertex_id]],
            uint sharpemu_instance_id [[instance_id]])
        {
            Gen5VsOut sharpemu_out = {};
            const uint rectangle = sharpemu_vertex_id / 3u;
            const uint corner = sharpemu_vertex_id % 3u;
            const float u = corner == 1u ? 1.0f : 0.0f;
            const float v = corner == 2u ? 1.0f : 0.0f;
            sharpemu_out.sharpemu_position = {{(fetched
                ? "float4(sharpemu_vin.in0, 0.0f, 1.0f)"
                : "float4(float(rectangle) - 1.0f + u, (2.0f * v) - 1.0f, 0.0f, 1.0f)")}};
            sharpemu_out.param0 = float4(u, v, 0.0f, 1.0f);
            return sharpemu_out;
        }
        """;

    private const string PixelSource = """
        #include <metal_stdlib>
        using namespace metal;

        struct RectIn
        {
            float4 position [[position]];
            float4 param0 [[user(locn0)]];
        };

        fragment float4 rect_fs(RectIn in [[stage_in]])
        {
            return float4(in.param0.x, in.param0.y, 0.0f, 1.0f);
        }
        """;

    private sealed class AddressSpace : IGuestAddressSpace
    {
        public ulong ProtectionPageSize => 4096;
        public bool TryProtect(ulong address, ulong size, GuestPageProtection protection) => true;
        public ulong AllocateAt(ulong desiredAddress, ulong size, bool executable = true, bool allowAlternative = true) => throw new NotSupportedException();
        public bool TryBackFixedRange(ulong address, ulong size, bool executable) => throw new NotSupportedException();
        public bool TryAllocateAtOrAbove(ulong desiredAddress, ulong size, bool executable, ulong alignment, out ulong actualAddress) => throw new NotSupportedException();
        public bool TryEnsureRangeCommitted(ulong address, ulong size) => throw new NotSupportedException();
        public bool TryAllocateGuestMemory(ulong size, ulong alignment, out ulong address) => throw new NotSupportedException();
        public bool TryFreeGuestMemory(ulong address) => throw new NotSupportedException();
    }

    // One page-aligned native block seen as guest memory at GuestBase.
    private sealed class Backing : IGuestBackedSpace, ICpuMemory, IDisposable
    {
        public const ulong Size = 4 * 1024 * 1024;
        private readonly void* _block = NativeMemory.AlignedAlloc((nuint)Size, 16384);

        public Backing() => NativeMemory.Clear(_block, (nuint)Size);

        public void Dispose() => NativeMemory.AlignedFree(_block);

        private bool Contains(ulong address, ulong length) => address >= GuestBase && address - GuestBase + length <= Size;

        public bool TryHoldRange(ulong address, ulong size) => false;
        public bool TryHoldRangeAtOrAbove(ulong searchStart, ulong size, ulong alignment, out ulong address) { address = 0; return false; }
        public bool TryMapBacked(ulong address, ulong size, ulong backingOffset, GuestPageProtection protection, out HostViewFailure failure) { failure = HostViewFailure.None; return false; }
        public bool TryUnmapBacked(ulong address, ulong size) => false;
        public bool TryClearBacking(ulong offset, ulong size) => false;
        public bool IsBackedView(ulong address) => Contains(address, 1);
        public bool IsBackedRange(ulong address, ulong size) => Contains(address, size);
        public (ulong Base, ulong Size) BackingAliasRange => ((ulong)_block, Size);

        public bool TryResolveBackingAlias(ulong address, ulong size, out ulong alias)
        {
            alias = Contains(address, size) ? (ulong)_block + (address - GuestBase) : 0;
            return alias != 0;
        }

        public bool TryWriteBacking(ulong address, ReadOnlySpan<byte> data) => TryWrite(address, data);

        public bool TryReadBacking(ulong address, Span<byte> data) => TryRead(address, data);

        public bool TryRead(ulong virtualAddress, Span<byte> destination)
        {
            if (!Contains(virtualAddress, (ulong)destination.Length)) return false;
            new ReadOnlySpan<byte>((byte*)_block + (virtualAddress - GuestBase), destination.Length).CopyTo(destination);
            return true;
        }

        public bool TryWrite(ulong virtualAddress, ReadOnlySpan<byte> source)
        {
            if (!Contains(virtualAddress, (ulong)source.Length)) return false;
            source.CopyTo(new Span<byte>((byte*)_block + (virtualAddress - GuestBase), source.Length));
            return true;
        }

        public bool CanRead(ulong address, ulong size) => Contains(address, size);
    }

    private sealed class Rig : IDisposable
    {
        private readonly nint _device;
        private readonly nint _queue;
        private readonly AddressSpace _space = new();
        private readonly PageGuard _pages;
        private readonly Backing _backing = new();
        private readonly bool _fetched;
        private readonly GpuPrimitiveTopology _topology;
        private readonly List<nint> _buffers = [];
        private PipelineHandle? _pipeline;

        public Rig(nint device, bool fetched, GpuPrimitiveTopology topology = GpuPrimitiveTopology.PatchList)
        {
            _device = device;
            _fetched = fetched;
            _topology = topology;
            _queue = MetalNative.Send(device, MetalNative.Selector("newCommandQueue"));
            _pages = new PageGuard(_space);
            Host = new MetalRenderHost(device, _queue, _backing, _backing, _pages, new MetalGpuWorkerRelay(static () => { }));
        }

        public MetalRenderHost Host { get; }

        public byte[] Render(Action<MetalRenderHost> draw, PipelineHandle? use = null)
        {
            var host = Host;
            using (host.BeginPreparation())
            {
                var request = Request();
                var image = host.ImageCache.FindImage(ref request);
                var resolution = new ColorTargetResolution(
                    request, TargetAddress, ImageBytes, new GpuExtent2D(Width, Height), 0, 0, 1, default, false, false, default);
                var target = new ColorTargetState(in resolution, 0, image);
                var attachment = ((IRenderHost)host).AcquireColorAttachment(in target);
                var pipeline = use ?? (_pipeline ??= CreatePipeline());
                var rendering = new RenderingState { Width = Width, Height = Height, Layers = 1, ColorAttachmentCount = 1, Samples = 1 };
                rendering.ColorAttachments[0] = new RenderingAttachment(
                    attachment.View, GpuImageLayout.ColorAttachmentOptimal, GpuFormat.R8G8B8A8Unorm,
                    0, 0, 0, 0, false, false, false, false, false);
                host.BeginRendering(in rendering);
                host.BindPipeline(GpuBindPoint.Graphics, pipeline);
                host.SetDynamicState(new DynamicDrawState(
                    0, 0, Width, Height, 0, 1, new ScissorRectangle(0, 0, (int)Width, (int)Height), 1,
                    0, 0, 0, 0, false, false, GpuCompareOp.Always, false, 0, 0, 0, false, 0, 1, false, default, default, 1, 0xF));
                if (_fetched)
                {
                    var positions = Buffer(MemoryMarshal.AsBytes(Positions.AsSpan()));
                    host.BindVertexBuffers([new BufferBinding((ulong)positions, 0)], new VertexInputInfo());
                }

                draw(host);
                host.EndRendering();
            }

            var bytes = new byte[ImageBytes];
            Assert.True(Host.TryReadGuest(TargetAddress, bytes));
            return bytes;
        }

        public PipelineHandle Pipeline => _pipeline ??= CreatePipeline();

        public nint Record(uint[] words) => Buffer(MemoryMarshal.AsBytes(words.AsSpan()));

        public void BindIndices(MetalRenderHost host, ushort[] indices)
        {
            var padded = new ushort[(indices.Length + 1) & ~1];
            indices.CopyTo(padded, 0);
            var buffer = Buffer(MemoryMarshal.AsBytes(padded.AsSpan()));
            host.BindIndexBuffer(new BufferBinding((ulong)buffer, 0), GpuIndexType.Uint16);
        }

        private nint Buffer(ReadOnlySpan<byte> contents)
        {
            fixed (byte* bytes = contents)
            {
                var buffer = MetalNative.SendBuffer(
                    _device, MetalNative.Selector("newBufferWithBytes:length:options:"), (nint)bytes, (nuint)contents.Length, 0);
                Assert.NotEqual(0, buffer);
                _buffers.Add(buffer);
                return buffer;
            }
        }

        public PipelineHandle CreatePipeline(bool depthClip = true)
        {
            var vertex = new ShaderProgram(1, Host.CreateShaderModule(
                new MetalCompiledGuestShader(new Gen5MslShader(VertexSource(_fetched), "gen5_vs", Gen5MslStage.Vertex, 1)),
                ShaderStage.Vertex, 1, 1));
            var pixel = new ShaderProgram(2, Host.CreateShaderModule(
                new MetalCompiledGuestShader(new Gen5MslShader(PixelSource, "rect_fs", Gen5MslStage.Pixel, 1)),
                ShaderStage.Pixel, 2, 2));
            var parameters = new PipelineStaticParameters
            {
                ColorCount = 1,
                Topology = _topology,
                Samples = 1,
                DepthClipEnable = depthClip,
            };
            parameters.SetColorMask(0, 0xF);
            var rendering = new PipelineRenderingState { ColorCount = 1 };
            rendering.ColorFormats[0] = GpuFormat.R8G8B8A8Unorm;
            var vertexInput = new PipelineVertexInputState();
            var vertexInfo = new VertexInputInfo();
            if (_fetched)
            {
                vertexInput.Bindings[0] = new PipelineVertexBinding(8, false);
                vertexInput.Attributes[0] = new PipelineVertexAttribute(0, 0);
                vertexInput.BindingCount = 1;
                vertexInput.AttributeCount = 1;
                vertexInfo = new VertexInputInfo
                {
                    Attributes = [new VertexAttributeResource(new BufferDescriptorWords(0, 8u << 16, 6, Float2Format << 12), 0, 2, 0, 0, 0, 0)],
                };
            }

            var vertexStage = new ShaderProgramInfo { Stage = ShaderStageKind.Vertex, Hash = 1 };
            vertexStage.VertexFetchComponents[0] = 2;
            return Host.CreateGraphicsPipeline(new GraphicsPipelineDescription
            {
                Rendering = rendering,
                VertexInput = vertexInput,
                VertexInfo = vertexInfo,
                VertexProgram = vertex,
                VertexStage = vertexStage,
                PixelProgram = pixel,
                PixelStage = new ShaderProgramInfo { Stage = ShaderStageKind.Pixel, Hash = 2 },
                StaticParameters = parameters,
            });
        }

        private static ImageRequest Request()
        {
            var description = ImageDescription.Create();
            description.Data = new GuestSpan(TargetAddress, ImageBytes);
            description.PixelFormat = GpuFormat.R8G8B8A8Unorm;
            description.GuestFormat = GuestPixelFormat.Bits8_8_8_8UNorm;
            description.Type = GuestImageType.Color2D;
            description.Extent = new GpuExtent3D(Width, Height, 1);
            description.Resources = new SubresourceCount(1, 1);
            description.Pitch = Width;
            description.BytesPerBlock = 4;
            description.Samples = 1;
            description.TileMode = GuestTileMode.Linear;
            description.MipLayout[0] = new MipLevelLayout { Offset = 0, Size = ImageBytes, Pitch = Width, Height = Height };
            var view = ImageViewDescription.Default with
            {
                Format = GpuFormat.R8G8B8A8Unorm,
                Type = GpuImageViewType.Type2D,
                Aspect = GpuImageAspect.ColorBit,
                LayerCount = 1,
                Usage = GpuImageUsage.ColorAttachmentBit,
            };
            return new ImageRequest(description, view, ImageRole.ColorTarget);
        }

        public void Dispose()
        {
            Host.ImageCache.StopWatching();
            Host.Dispose();
            foreach (var buffer in _buffers)
            {
                MetalNative.SendVoid(buffer, MetalNative.Selector("release"));
            }

            _pages.Dispose();
            _backing.Dispose();
            MetalNative.SendVoid(_queue, MetalNative.Selector("release"));
        }
    }

    private bool RequireDevice()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Rectangle-list draws are encoded on the device only behind the exclusive device gate.");
            return false;
        }

        Assert.True(OperatingSystem.IsMacOS());
        Assert.NotEqual(0, Device.Value);
        return true;
    }

    private static readonly Lazy<nint> Device = new(MetalNative.MTLCreateSystemDefaultDevice);
}
