// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
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

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// A host over plain guest memory that draws colour quads into any number of linear colour
/// targets, the way an executor does (bindings reset, targets acquired, pass begun, pipeline,
/// state, draw). For the pass-merging tests and the DS-shaped pass benchmark.
/// </summary>
internal sealed unsafe class MetalPassRig : IDisposable
{
    public const ulong GuestBase = 0x2_0000_0000;
    public const byte GuestPattern = 0x11;

    private readonly nint _queue;
    private readonly AddressSpace _space = new();
    private readonly PageGuard _pages;
    private readonly Backing _backing;
    private ulong _nextProgram = 1;

    public MetalPassRig(nint device, ulong guestBytes)
    {
        _queue = MetalNative.Send(device, MetalNative.Selector("newCommandQueue"));
        _pages = new PageGuard(_space);
        _backing = new Backing(guestBytes);
        Host = new MetalRenderHost(device, _queue, _backing, _backing, _pages, new MetalGpuWorkerRelay(static () => { }));
        Host.AsynchronousGraphics = false;
    }

    public MetalRenderHost Host { get; }

    public readonly record struct Target(ColorTargetState State, GpuFormat Format, uint Width, uint Height, ulong Address, ulong Bytes);

    public static ulong BytesPerPixel(GpuFormat format) => format switch
    {
        GpuFormat.R16G16B16A16Sfloat => 8,
        _ => 4,
    };

    private static GuestPixelFormat GuestFormat(GpuFormat format) => format switch
    {
        GpuFormat.R32Uint => GuestPixelFormat.Bits32UInt,
        GpuFormat.R16G16Sfloat => GuestPixelFormat.Bits16_16Float,
        GpuFormat.R16G16B16A16Sfloat => GuestPixelFormat.Bits16_16_16_16Float,
        _ => GuestPixelFormat.Bits8_8_8_8UNorm,
    };

    public static bool IsInteger(GpuFormat format) => format == GpuFormat.R32Uint;

    public Target MakeTarget(ulong address, GpuFormat format, uint width, uint height)
    {
        var bytes = width * height * BytesPerPixel(format);
        var description = ImageDescription.Create();
        description.Data = new GuestSpan(address, bytes);
        description.PixelFormat = format;
        description.GuestFormat = GuestFormat(format);
        description.Type = GuestImageType.Color2D;
        description.Extent = new GpuExtent3D(width, height, 1);
        description.Resources = new SubresourceCount(1, 1);
        description.Pitch = width;
        description.BytesPerBlock = (uint)BytesPerPixel(format);
        description.Samples = 1;
        description.TileMode = GuestTileMode.Linear;
        description.MipLayout[0] = new MipLevelLayout { Offset = 0, Size = bytes, Pitch = width, Height = height };
        var view = ImageViewDescription.Default with
        {
            Format = format,
            Type = GpuImageViewType.Type2D,
            Aspect = GpuImageAspect.ColorBit,
            LayerCount = 1,
            Usage = GpuImageUsage.ColorAttachmentBit,
        };
        var request = new ImageRequest(description, view, ImageRole.ColorTarget);
        var image = Host.ImageCache.FindImage(ref request);
        var resolution = new ColorTargetResolution(
            request, address, bytes, new GpuExtent2D(width, height), 0, 0, 1, default, false, false, default);
        return new Target(new ColorTargetState(in resolution, 0, image), format, width, height, address, bytes);
    }

    /// <summary>
    /// A quad from x = left to x = right over the whole height, writing <paramref name="outputs"/>[i]
    /// (an MSL vector expression) to colour output i, for targets of these formats.
    /// </summary>
    /// When <paramref name="knownPositions"/> is set the vertex program reports its six
    /// positions and the pixel program writes every output and cannot discard, as the
    /// translator marks a full-screen pass whose loads may be elided.
    /// With <paramref name="mesh"/> the quad comes from a mesh program (one threadgroup), as
    /// the geometry engine's merged stages draw.
    /// <paramref name="discardWhen"/>, an MSL condition on <c>in.position</c>, discards the
    /// fragments it holds for.
    public PipelineHandle Pipeline(
        GpuFormat[] formats, string[] outputs, float left = -1f, float right = 1f, bool knownPositions = false, bool mesh = false,
        string? discardWhen = null)
    {
        var id = _nextProgram;
        _nextProgram += 2;
        var vertexSource = $$"""
            #include <metal_stdlib>
            using namespace metal;
            struct Gen5VsOut
            {
                float4 sharpemu_position [[position]];
            };

            vertex Gen5VsOut gen5_vs(uint sharpemu_vertex_id [[vertex_id]], uint sharpemu_instance_id [[instance_id]])
            {
                const float2 corners[6] = {
                    float2({{F(left)}}, -1.0f), float2({{F(right)}}, -1.0f), float2({{F(left)}}, 1.0f),
                    float2({{F(left)}}, 1.0f), float2({{F(right)}}, -1.0f), float2({{F(right)}}, 1.0f) };
                Gen5VsOut sharpemu_out = {};
                sharpemu_out.sharpemu_position = float4(corners[sharpemu_vertex_id], 0.5f, 1.0f);
                return sharpemu_out;
            }
            """;
        if (mesh)
        {
            vertexSource = $$"""
                #include <metal_stdlib>
                using namespace metal;
                struct Gen5MsOut
                {
                    float4 sharpemu_position [[position]];
                };

                using Gen5Mesh = metal::mesh<Gen5MsOut, void, 6, 2, metal::topology::triangle>;

                [[mesh]] void gen5_ms(Gen5Mesh sharpemu_mesh, uint sharpemu_lane [[thread_index_in_threadgroup]])
                {
                    if (sharpemu_lane != 0)
                    {
                        return;
                    }

                    const float2 corners[6] = {
                        float2({{F(left)}}, -1.0f), float2({{F(right)}}, -1.0f), float2({{F(left)}}, 1.0f),
                        float2({{F(left)}}, 1.0f), float2({{F(right)}}, -1.0f), float2({{F(right)}}, 1.0f) };
                    for (uint index = 0; index < 6; index++)
                    {
                        Gen5MsOut sharpemu_out = {};
                        sharpemu_out.sharpemu_position = float4(corners[index], 0.5f, 1.0f);
                        sharpemu_mesh.set_vertex(index, sharpemu_out);
                        sharpemu_mesh.set_index(index, index);
                    }

                    sharpemu_mesh.set_primitive_count(2);
                }
                """;
        }

        var fields = new System.Text.StringBuilder();
        var assigns = new System.Text.StringBuilder();
        for (var index = 0; index < formats.Length; index++)
        {
            var type = IsInteger(formats[index]) ? "uint4" : "float4";
            fields.Append(CultureInfo.InvariantCulture, $"    {type} c{index} [[color({index})]];\n");
            assigns.Append(CultureInfo.InvariantCulture, $"    o.c{index} = {type}({outputs[index]});\n");
        }

        var pixelSource = $$"""
            #include <metal_stdlib>
            using namespace metal;
            struct PixelIn
            {
                float4 position [[position]];
            };

            struct PixelOut
            {
            {{fields}}};

            // A well-mixed hash of a pixel, for outputs that must not compress.
            static inline uint rig_hash(float4 position, uint seed)
            {
                uint h = uint(position.x) * 0x9E3779B1u + uint(position.y) * 0x85EBCA77u + seed;
                h ^= h >> 16; h *= 0x85EBCA6Bu; h ^= h >> 13; h *= 0xC2B2AE35u; h ^= h >> 16;
                return h;
            }

            fragment PixelOut fill_fs(PixelIn in [[stage_in]])
            {
                {{(discardWhen is null ? string.Empty : $"if ({discardWhen}) discard_fragment();")}}
                PixelOut o;
            {{assigns}}    return o;
            }
            """;
        var vertex = new ShaderProgram(id, Host.CreateShaderModule(
            new MetalCompiledGuestShader(new Gen5MslShader(
                vertexSource, mesh ? "gen5_ms" : "gen5_vs", mesh ? Gen5MslStage.Mesh : Gen5MslStage.Vertex, 1,
                MeshInfo: mesh ? new Gen5MslMeshInfo(32, 1, 6, 2, 3, Gen5MslMeshTopology.Triangle, 0, 0, 1, false) : null)
            {
                ConstantPositions = knownPositions
                    ? [left, -1f, 0.5f, 1f, right, -1f, 0.5f, 1f, left, 1f, 0.5f, 1f,
                       left, 1f, 0.5f, 1f, right, -1f, 0.5f, 1f, right, 1f, 0.5f, 1f]
                    : null,
            }),
            ShaderStage.Vertex, id, id));
        var pixel = new ShaderProgram(id + 1, Host.CreateShaderModule(
            new MetalCompiledGuestShader(new Gen5MslShader(pixelSource, "fill_fs", Gen5MslStage.Pixel, 1)
            {
                CannotDiscard = knownPositions,
                ColorOutputMask = knownPositions ? (1u << formats.Length) - 1 : 0,
            }),
            ShaderStage.Pixel, id + 1, id + 1));
        var parameters = new PipelineStaticParameters
        {
            ColorCount = (uint)formats.Length,
            Topology = GpuPrimitiveTopology.TriangleList,
            Samples = 1,
            DepthClipEnable = true,
        };
        var rendering = new PipelineRenderingState { ColorCount = (uint)formats.Length };
        for (var index = 0; index < formats.Length; index++)
        {
            parameters.SetColorMask(index, 0xF);
            rendering.ColorFormats[index] = formats[index];
        }

        var description = new GraphicsPipelineDescription
        {
            Rendering = rendering,
            VertexInput = new PipelineVertexInputState(),
            VertexInfo = new VertexInputInfo(),
            VertexProgram = vertex,
            VertexStage = new ShaderProgramInfo { Stage = ShaderStageKind.Vertex, Hash = id },
            PixelProgram = pixel,
            PixelStage = new ShaderProgramInfo { Stage = ShaderStageKind.Pixel, Hash = id + 1 },
            StaticParameters = parameters,
        };
        return Host.CreateGraphicsPipeline(description);

        static string F(float value) => value.ToString("F4", CultureInfo.InvariantCulture) + "f";
    }

    /// <summary>One draw of a pipeline into these targets, as an executor records it.</summary>
    public void Draw(PipelineHandle pipeline, ReadOnlySpan<Target> targets, MetalRenderHost.MetalStageBindings? stage = null)
    {
        var host = Host;
        host.ResetBindings();
        var state = new RenderingState
        {
            Width = targets[0].Width,
            Height = targets[0].Height,
            Layers = 1,
            ColorAttachmentCount = (uint)targets.Length,
            Samples = 1,
        };
        for (var index = 0; index < targets.Length; index++)
        {
            var target = targets[index].State;
            var attachment = ((IRenderHost)host).AcquireColorAttachment(in target);
            state.ColorAttachments[index] = new RenderingAttachment(
                attachment.View, targets[index].Format, 0, 0, 0, 0, false, false, false, false, false);
        }

        host.NoteJoinStages(stage is null ? ReadOnlySpan<IPreparedBindings>.Empty : new IPreparedBindings[] { stage });
        var width = targets[0].Width;
        var height = targets[0].Height;
        host.SetDynamicState(new DynamicDrawState(
            0, 0, width, height, 0, 1, new ScissorRectangle(0, 0, (int)width, (int)height), 1,
            0, 0, 0, 0, false, false, GpuCompareOp.Always, false, 0, 0, 0, false, 0, 1, false, default, default,
            (uint)targets.Length, 0xFF));
        host.NoteUpcomingDraw(in pipeline, 6, 0);
        host.BeginRendering(in state);
        host.BindPipeline(GpuBindPoint.Graphics, pipeline);
        host.Draw(6, 1, 0, 0);
    }

    // A stage that samples these textures and binds these buffers; with deviceAddresses its
    // program reaches guest memory through device addresses, as Demon's Souls' G-buffer programs do.
    public static MetalRenderHost.MetalStageBindings Stage(
        nint[]? textures = null, MetalGuestRange[]? buffers = null, bool deviceAddresses = false) =>
        new()
        {
            Program = new ShaderProgramInfo
            {
                Stage = ShaderStageKind.Vertex, Bindings = new BindingLayout(), UsesDeviceAddresses = deviceAddresses,
            },
            Textures = textures ?? [],
            MipTextures = new nint[]?[textures?.Length ?? 0],
            ImageCount = textures?.Length ?? 0,
            Buffers = buffers ?? [],
            BufferCount = buffers?.Length ?? 0,
        };

    public byte[] Read(ulong address, ulong bytes)
    {
        var data = new byte[bytes];
        Assert.True(Host.TryReadGuest(address, data));
        return data;
    }

    public byte[] ReadRaw(ulong address, ulong bytes)
    {
        var data = new byte[bytes];
        Assert.True(_backing.TryRead(address, data));
        return data;
    }

    public void Dispose()
    {
        Host.ImageCache.StopWatching();
        Host.Dispose();
        _pages.Dispose();
        _backing.Dispose();
        MetalNative.SendVoid(_queue, MetalNative.Selector("release"));
    }

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

    private sealed class Backing : IGuestBackedSpace, ICpuMemory, IDisposable
    {
        private readonly ulong _size;
        private readonly void* _block;

        public Backing(ulong size)
        {
            _size = size;
            _block = NativeMemory.AlignedAlloc((nuint)size, 16384);
            NativeMemory.Fill(_block, (nuint)size, GuestPattern);
        }

        public void Dispose() => NativeMemory.AlignedFree(_block);

        private bool Contains(ulong address, ulong length) => address >= GuestBase && address - GuestBase + length <= _size;

        public bool TryHoldRange(ulong address, ulong size) => false;
        public bool TryHoldRangeAtOrAbove(ulong searchStart, ulong size, ulong alignment, out ulong address) { address = 0; return false; }
        public bool TryMapBacked(ulong address, ulong size, ulong backingOffset, GuestPageProtection protection, out HostViewFailure failure) { failure = HostViewFailure.None; return false; }
        public bool TryUnmapBacked(ulong address, ulong size) => false;
        public bool TryClearBacking(ulong offset, ulong size) => false;
        public bool IsBackedView(ulong address) => Contains(address, 1);
        public bool IsBackedRange(ulong address, ulong size) => Contains(address, size);
        public (ulong Base, ulong Size) BackingAliasRange => ((ulong)_block, _size);

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
}
