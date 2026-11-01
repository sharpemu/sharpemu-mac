// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.HLE.Host;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Images;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// SHARPEMU_METAL_V2_DEPTH_TWIN: a depth surface sampled as its single-channel colour twin
/// (D32 as R32 float, D16 as R16 unorm) binds the depth texture through a same-format,
/// swizzled view instead of a second texture. Each case reads the bound view with a
/// compute kernel and compares it, texel for texel and channel for channel, with what the
/// two-texture hand-off gives over the same guest bytes. Device work is behind the
/// exclusive device gate.
/// </summary>
public sealed unsafe class MetalDepthTwinTests(ITestOutputHelper output)
{
    private const ulong GuestBase = 0x2_0000_0000;
    private const ulong DepthAddress = GuestBase + 0x100000;
    private const ulong StencilAddress = GuestBase + 0x800000;
    private const uint Width = 1024;
    private const uint Height = 512;

    private const string ReadKernel = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void read_twin(texture2d<float, access::sample> t [[texture(0)]],
                              device float4* o [[buffer(0)]],
                              uint2 p [[thread_position_in_grid]])
        {
            if (p.x < t.get_width() && p.y < t.get_height())
            {
                o[p.y * t.get_width() + p.x] = t.read(p);
            }
        }
        """;

    private const string StencilKernel = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void read_stencil(texture2d_ms<uint, access::read> t [[texture(0)]],
                                 device uint* o [[buffer(0)]],
                                 uint2 p [[thread_position_in_grid]])
        {
            if (p.x < t.get_width() && p.y < t.get_height())
            {
                o[p.y * t.get_width() + p.x] = t.read(p, 0).r | (t.read(p, 1).r << 8);
            }
        }
        """;

    public static IEnumerable<object[]> Pairs()
    {
        var identity = new GpuComponentMapping(GpuComponentSwizzle.R, GpuComponentSwizzle.G, GpuComponentSwizzle.B, GpuComponentSwizzle.A);
        var x001 = new GpuComponentMapping(GpuComponentSwizzle.R, GpuComponentSwizzle.Zero, GpuComponentSwizzle.Zero, GpuComponentSwizzle.One);
        var xxxx = new GpuComponentMapping(GpuComponentSwizzle.R, GpuComponentSwizzle.R, GpuComponentSwizzle.R, GpuComponentSwizzle.R);
        yield return [(int)GpuFormat.D32Sfloat, identity];
        yield return [(int)GpuFormat.D32Sfloat, xxxx];
        yield return [(int)GpuFormat.D32SfloatS8Uint, x001];
        yield return [(int)GpuFormat.D16Unorm, identity];
        yield return [(int)GpuFormat.D16Unorm, xxxx];
    }

    [Theory]
    [MemberData(nameof(Pairs))]
    public void ASampledColourTwinOfADepthTargetBindsTheDepthTextureAndReadsTheSameTexels(
        int depthFormat, GpuComponentMapping mapping)
    {
        if (!RequireDevice()) return;
        var format = (GpuFormat)depthFormat;
        var pattern = DepthPattern(format);
        var twin = Run(format, mapping, pattern, twins: true);
        var copied = Run(format, mapping, pattern, twins: false);

        Assert.True(twin.SameImage);
        Assert.False(copied.SameImage);
        Assert.Equal(0UL, twin.ExtraBytes);
        Assert.True(copied.ExtraBytes > 0);
        Assert.Equal(1, twin.Binds);
        output.WriteLine(
            $"format={format} mapping={mapping} saved={copied.ExtraBytes} bytes " +
            $"(two textures {copied.TotalBytes}, one texture {twin.TotalBytes})");

        Assert.Equal(copied.Texels.Length, twin.Texels.Length);
        var nonZero = 0;
        for (var index = 0; index < twin.Texels.Length; index++)
        {
            Assert.True(copied.Texels[index] == twin.Texels[index],
                $"texel {index / 4} channel {index % 4}: two textures {copied.Texels[index]}, depth texture {twin.Texels[index]}");
            nonZero += twin.Texels[index] != 0 ? 1 : 0;
        }

        Assert.True(nonZero > twin.Texels.Length / 8, "the pattern did not reach the texture");
    }

    // A colour twin made before the depth target that owns its bytes, and only ever
    // sampled, gives way to it; the next sample binds the depth texture.
    [Fact]
    public void ASampledOnlyColourTwinGivesWayToTheDepthTargetOverItsBytes()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Write(DepthAddress, DepthPattern(GpuFormat.D32Sfloat));
        var identity = new GpuComponentMapping(GpuComponentSwizzle.R, GpuComponentSwizzle.G, GpuComponentSwizzle.B, GpuComponentSwizzle.A);
        var textureRequest = Request(GpuFormat.D32Sfloat, ImageRole.Texture, identity);
        var colour = rig.Cache.FindImage(ref textureRequest);
        Assert.NotEqual(0, rig.Cache.AcquireTextureView(colour, in textureRequest));

        var depthRequest = Request(GpuFormat.D32Sfloat, ImageRole.DepthTarget, identity);
        var depth = rig.Cache.FindImage(ref depthRequest);
        Assert.NotEqual(colour, depth);
        Assert.False(rig.Cache.TryGetImage(colour, out _));
        Assert.Equal(1, rig.Cache.ImageCount);
        rig.Cache.AcquireDepthTargetView(depth, in depthRequest);

        textureRequest = Request(GpuFormat.D32Sfloat, ImageRole.Texture, identity);
        Assert.Equal(depth, rig.Cache.FindImage(ref textureRequest));
        Assert.Equal((1L, 1L), rig.Cache.TakeDepthTwinCounters());
        rig.Harness.Submit();
    }

    // A colour twin the guest writes is still a texture of its own, kept by the hand-off.
    [Fact]
    public void AStoredColourTwinKeepsItsOwnTexture()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var identity = new GpuComponentMapping(GpuComponentSwizzle.R, GpuComponentSwizzle.G, GpuComponentSwizzle.B, GpuComponentSwizzle.A);
        var depthRequest = Request(GpuFormat.D32Sfloat, ImageRole.DepthTarget, identity);
        var depth = rig.Cache.FindImage(ref depthRequest);
        rig.Cache.AcquireDepthTargetView(depth, in depthRequest);
        var storageRequest = Request(GpuFormat.D32Sfloat, ImageRole.StorageImage, identity);
        var storage = rig.Cache.FindImage(ref storageRequest);
        Assert.NotEqual(depth, storage);
        Assert.Equal((0L, 0L), rig.Cache.TakeDepthTwinCounters());
        rig.Harness.Submit();
    }

    // SHARPEMU_METAL_V2_STENCIL_TWIN: the stencil plane of a two-sample depth target read as 8-bit integers binds
    // a stencil view of the depth texture (MetalImageCache.StencilTwin.cs); nothing else can fill an 8-bit
    // multisampled image of it, and Ghost of Tsushima's depth copy read every object's stencil as zero from one.
    [Fact]
    public void AMultisampledStencilPlaneReadAsR8UintBindsTheDepthTexturesStencilView()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var identity = new GpuComponentMapping(GpuComponentSwizzle.R, GpuComponentSwizzle.G, GpuComponentSwizzle.B, GpuComponentSwizzle.A);
        var depthRequest = Request(GpuFormat.D32SfloatS8Uint, ImageRole.DepthTarget, identity);
        depthRequest = depthRequest with { Description = depthRequest.Description with { Samples = 2 } };
        var depth = rig.Cache.FindImage(ref depthRequest);
        var target = rig.Cache.AcquireDepthTargetView(depth, in depthRequest);
        Assert.NotEqual(0, target);

        // The stencil the GPU leaves in the depth texture, never in guest memory.
        var pass = MetalNative.Send(MetalNative.Class("MTLRenderPassDescriptor"), MetalNative.Selector("renderPassDescriptor"));
        var stencil = MetalNative.Send(pass, MetalNative.Selector("stencilAttachment"));
        MetalNative.SendVoid(stencil, MetalNative.Selector("setTexture:"), target);
        MetalV2Native.SendVoidValue(stencil, MetalNative.Selector("setLoadAction:"), Mtl.LoadActionClear);
        MetalV2Native.SendVoidValue(stencil, MetalNative.Selector("setStoreAction:"), Mtl.StoreActionStore);
        MetalV2Native.SendVoidUInt(stencil, MetalNative.Selector("setClearStencil:"), 0x4C);
        var encoder = MetalNative.Send(rig.Harness.CommandBuffer, MetalNative.Selector("renderCommandEncoderWithDescriptor:"), pass);
        MetalNative.SendVoid(encoder, MetalNative.Selector("endEncoding"));

        var description = ImageDescription.Create();
        description.Data = depthRequest.Description.Stencil;
        description.PixelFormat = GpuFormat.R8Uint;
        description.GuestFormat = GuestPixelFormat.Bits8UInt;
        description.Type = GuestImageType.Color2D;
        description.Extent = new GpuExtent3D(Width, Height, 1);
        description.Resources = new SubresourceCount(1, 1);
        description.Pitch = TileGeometry.DepthPitch(Width, 1);
        description.BytesPerBlock = 1;
        description.Samples = 2;
        description.TileMode = GuestTileMode.Depth;
        var view = ImageViewDescription.Default with
        {
            Format = GpuFormat.R8Uint,
            Type = GpuImageViewType.Type2D,
            Aspect = GpuImageAspect.ColorBit,
            LayerCount = 1,
            Mapping = identity,
            Usage = GpuImageUsage.SampledBit,
        };
        var read = new ImageRequest(description, view, ImageRole.Texture);
        Assert.True(rig.Cache.TryFindStencilTwin(in read, out var twin));
        Assert.Equal(depth, twin);
        var stencilView = rig.Cache.AcquireTextureView(twin, in read);
        Assert.NotEqual(0, stencilView);
        Assert.Equal(1L, rig.Cache.TakeStencilTwinBinds());

        var values = rig.ReadStencilSamples(stencilView);
        Assert.All(values, value => Assert.Equal(0x4C4Cu, value));

        // An ordinary sampled read of a one-sample stencil plane keeps its own image.
        var single = read with { Description = read.Description with { Samples = 1 } };
        Assert.False(rig.Cache.TryFindStencilTwin(in single, out _));
        rig.Harness.Submit();
    }

    private (bool SameImage, ulong ExtraBytes, ulong TotalBytes, long Binds, float[] Texels) Run(
        GpuFormat format, GpuComponentMapping mapping, byte[] pattern, bool twins)
    {
        using var rig = new Rig(Device.Value);
        rig.Cache.DepthTwins = twins;
        rig.Write(DepthAddress, pattern);
        var depthRequest = Request(format, ImageRole.DepthTarget, mapping);
        var depth = rig.Cache.FindImage(ref depthRequest);
        rig.Cache.AcquireDepthTargetView(depth, in depthRequest);
        var before = rig.Cache.TotalUsedMemory;

        var textureRequest = Request(format, ImageRole.Texture, mapping);
        var texture = rig.Cache.FindImage(ref textureRequest);
        var view = rig.Cache.AcquireTextureView(texture, in textureRequest);
        Assert.NotEqual(0, view);
        var total = rig.Cache.TotalUsedMemory;
        var texels = rig.ReadView(view);
        return (texture == depth, total - before, total, rig.Cache.TakeDepthTwinCounters().Binds, texels);
    }

    private static byte[] DepthPattern(GpuFormat format)
    {
        var bytesPerTexel = format == GpuFormat.D16Unorm ? 2 : 4;
        var size = checked((int)DepthSize(format).Data);
        var bytes = new byte[size];
        var random = new Random(0x7D16 + (int)format);
        for (var offset = 0; offset + bytesPerTexel <= size; offset += bytesPerTexel)
        {
            if (bytesPerTexel == 2)
            {
                BitConverter.TryWriteBytes(bytes.AsSpan(offset), (ushort)random.Next(1, 0x10000));
            }
            else
            {
                // A depth value Depth32Float holds as it is: in [0, 1], no NaN or denormal.
                BitConverter.TryWriteBytes(bytes.AsSpan(offset), (float)random.NextDouble());
            }
        }

        return bytes;
    }

    private static (ulong Data, ulong Stencil) DepthSize(GpuFormat format)
    {
        var depthFormat = format == GpuFormat.D16Unorm ? GuestDepthFormat.Z16 : GuestDepthFormat.Z32Float;
        var stencil = format == GpuFormat.D32SfloatS8Uint ? GuestStencilFormat.Stencil8UInt : GuestStencilFormat.Invalid;
        Assert.True(TileGeometry.TryGetDepthSize(Width, Height, 0, depthFormat, stencil, false,
            out var stencilSize, out _, out var depthSize));
        return (depthSize.Size, stencilSize.Size);
    }

    private static ImageRequest Request(GpuFormat depthFormat, ImageRole role, GpuComponentMapping mapping)
    {
        var depthRole = role == ImageRole.DepthTarget;
        var bytes = depthFormat == GpuFormat.D16Unorm ? 2u : 4u;
        var (dataSize, stencilSize) = DepthSize(depthFormat);
        var pitch = TileGeometry.DepthPitch(Width, bytes);
        var format = depthRole ? depthFormat : depthFormat == GpuFormat.D16Unorm ? GpuFormat.R16Unorm : GpuFormat.R32Sfloat;
        var description = ImageDescription.Create();
        description.Data = new GuestSpan(DepthAddress, dataSize);
        if (depthRole && depthFormat == GpuFormat.D32SfloatS8Uint)
        {
            description.Stencil = new GuestSpan(StencilAddress, stencilSize);
        }

        description.PixelFormat = format;
        description.GuestFormat = bytes == 2 ? GuestPixelFormat.Bits16UNorm : GuestPixelFormat.Bits32Float;
        description.Type = GuestImageType.Color2D;
        description.Extent = new GpuExtent3D(Width, Height, 1);
        description.Resources = new SubresourceCount(1, 1);
        description.Pitch = pitch;
        description.BytesPerBlock = bytes;
        description.Samples = 1;
        description.TileMode = GuestTileMode.Depth;
        description.MipLayout[0] = new MipLevelLayout { Offset = 0, Size = dataSize, Pitch = pitch, Height = Height };
        var view = ImageViewDescription.Default with
        {
            Format = format,
            Type = GpuImageViewType.Type2D,
            Aspect = depthRole ? GpuImageAspect.DepthBit : GpuImageAspect.ColorBit,
            LayerCount = 1,
            Mapping = depthRole ? default : mapping,
            Usage = role switch
            {
                ImageRole.DepthTarget => GpuImageUsage.DepthStencilAttachmentBit,
                ImageRole.StorageImage => GpuImageUsage.StorageBit,
                _ => GpuImageUsage.SampledBit,
            },
        };
        return new ImageRequest(description, view, role);
    }

    private sealed class Rig : IDisposable
    {
        private readonly AddressSpace _space = new();
        private readonly PageGuard _pages;
        private readonly MetalGuestMemory _memory;
        private readonly Backing _backing = new();

        public Rig(nint device)
        {
            Harness = new MetalTransferHarness(device);
            Harness.Begin();
            _pages = new PageGuard(_space);
            _memory = new MetalGuestMemory(device, _backing, new NoMemory());
            Cache = new MetalImageCache(new Owner(Harness), Harness.Transfer, _memory, _pages, _backing);
            // Exact textures: lossy compression is not this suite's subject.
            Cache.LossyTextures = false;
        }

        public MetalTransferHarness Harness { get; }
        public MetalImageCache Cache { get; }

        public void Write(ulong address, ReadOnlySpan<byte> bytes) => Assert.True(_backing.TryWriteBacking(address, bytes));

        // Every recorded transfer runs, then a kernel reads each texel of the view.
        public float[] ReadView(nint view)
        {
            var count = (int)(Width * Height * 4);
            var buffer = Harness.Buffer((ulong)count * sizeof(float));
            var pipeline = MetalGuestWindowResidencyTests.ComputePipeline(ReadKernel, "read_twin");
            try
            {
                var encoder = MetalNative.Send(Harness.CommandBuffer, MetalNative.Selector("computeCommandEncoder"));
                MetalNative.SendVoid(encoder, MetalNative.Selector("setComputePipelineState:"), pipeline);
                MetalNative.SendSetAtIndex(encoder, MetalNative.Selector("setTexture:atIndex:"), view, 0);
                MetalNative.SendSetBuffer(encoder, MetalNative.Selector("setBuffer:offset:atIndex:"), buffer, 0, 0);
                MetalNative.SendDispatch(encoder, MetalNative.Selector("dispatchThreads:threadsPerThreadgroup:"),
                    new MtlSize { Width = Width, Height = Height, Depth = 1 },
                    new MtlSize { Width = 16, Height = 16, Depth = 1 });
                MetalNative.SendVoid(encoder, MetalNative.Selector("endEncoding"));
                Harness.Submit();
                Harness.Begin();
            }
            finally
            {
                MetalNative.SendVoid(pipeline, MetalNative.Selector("release"));
            }

            return MemoryMarshal.Cast<byte, float>(Harness.Read(buffer, 0, (ulong)count * sizeof(float))).ToArray();
        }

        // Both samples of each texel of a two-sample stencil view, one word each (sample 1 in bits 8..15).
        public uint[] ReadStencilSamples(nint view)
        {
            var count = (int)(Width * Height);
            var buffer = Harness.Buffer((ulong)count * sizeof(uint));
            var pipeline = MetalGuestWindowResidencyTests.ComputePipeline(StencilKernel, "read_stencil");
            try
            {
                var encoder = MetalNative.Send(Harness.CommandBuffer, MetalNative.Selector("computeCommandEncoder"));
                MetalNative.SendVoid(encoder, MetalNative.Selector("setComputePipelineState:"), pipeline);
                MetalNative.SendSetAtIndex(encoder, MetalNative.Selector("setTexture:atIndex:"), view, 0);
                MetalNative.SendSetBuffer(encoder, MetalNative.Selector("setBuffer:offset:atIndex:"), buffer, 0, 0);
                MetalNative.SendDispatch(encoder, MetalNative.Selector("dispatchThreads:threadsPerThreadgroup:"),
                    new MtlSize { Width = Width, Height = Height, Depth = 1 },
                    new MtlSize { Width = 16, Height = 16, Depth = 1 });
                MetalNative.SendVoid(encoder, MetalNative.Selector("endEncoding"));
                Harness.Submit();
                Harness.Begin();
            }
            finally
            {
                MetalNative.SendVoid(pipeline, MetalNative.Selector("release"));
            }

            return MemoryMarshal.Cast<byte, uint>(Harness.Read(buffer, 0, (ulong)count * sizeof(uint))).ToArray();
        }

        public void Dispose()
        {
            Cache.StopWatching();
            Cache.Dispose();
            _pages.Dispose();
            Harness.Dispose();
            _backing.Dispose();
        }
    }

    private sealed class Owner(MetalTransferHarness harness) : IMetalImageCacheOwner
    {
        public nint Device => harness.Device;
        public ulong CurrentSerial => harness.SubmissionSerial;
        public ulong CompletedSerial => harness.CompletedSerial;
        public IMetalSubmissionContext BeginTransfer(MetalTransferCause cause) => harness;
        public bool HasPendingGpuWrite(ulong address, ulong size) => false;
        public void FillGuestRange(ulong address, ulong size, uint value) { }
        public void NoteGuestWrite(ulong address, ulong size) { }
        public void WaitForGpu() { }

        public void ClearTextureColor(nint texture, ulong pixelFormat, in SubresourceRange range, uint width, uint height, in GpuClearColor color)
        {
        }

        public void ClearTextureDepthStencil(nint texture, ulong pixelFormat, in SubresourceRange range, uint width, uint height,
            float depth, byte stencil, bool clearDepth, bool clearStencil)
        {
        }
    }

    private sealed class NoMemory : ICpuMemory
    {
        public bool TryRead(ulong virtualAddress, Span<byte> destination) => false;
        public bool TryWrite(ulong virtualAddress, ReadOnlySpan<byte> source) => false;
        public bool CanRead(ulong address, ulong size) => false;
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

    private sealed class Backing : IGuestBackedSpace, IDisposable
    {
        public const ulong Size = 16 * 1024 * 1024;
        private readonly void* _block = NativeMemory.AlignedAlloc((nuint)Size, 16384);

        public Backing() => NativeMemory.Clear(_block, (nuint)Size);

        public void Dispose() => NativeMemory.AlignedFree(_block);

        private static bool Contains(ulong address, ulong length) => address >= GuestBase && address - GuestBase + length <= Size;

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

        public bool TryResolveBackingAliasSpan(ulong address, ulong size, out ulong alias) =>
            TryResolveBackingAlias(address, size, out alias);

        public bool TryWriteBacking(ulong address, ReadOnlySpan<byte> data)
        {
            if (!Contains(address, (ulong)data.Length)) return false;
            data.CopyTo(new Span<byte>((byte*)_block + (address - GuestBase), data.Length));
            return true;
        }

        public bool TryReadBacking(ulong address, Span<byte> data)
        {
            if (!Contains(address, (ulong)data.Length)) return false;
            new ReadOnlySpan<byte>((byte*)_block + (address - GuestBase), data.Length).CopyTo(data);
            return true;
        }
    }

    private bool RequireDevice()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Depth twin sampling runs on the device only behind the exclusive device gate.");
            return false;
        }

        Assert.True(OperatingSystem.IsMacOS());
        Assert.NotEqual(0, Device.Value);
        return true;
    }

    private static readonly Lazy<nint> Device = new(MetalNative.MTLCreateSystemDefaultDevice);
}
