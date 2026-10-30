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
/// An image first asked for as a colour target is created without the usages that stop an
/// Apple GPU compressing it, and gains them - with its contents - the first time a shader
/// stores to it or a view reinterprets its format. Device work is behind the exclusive
/// device gate.
/// </summary>
public sealed unsafe class MetalCompressibleTargetTests(ITestOutputHelper output)
{
    private const ulong GuestBase = 0x2_0000_0000;
    private const ulong ImageAddress = GuestBase + 0x10000;
    private const uint Side = 64;
    private const ulong ImageSize = Side * Side * 4;
    private const ulong Breaking = Mtl.TextureUsageShaderWrite | Mtl.TextureUsagePixelFormatView;

    [Fact]
    public void AColourTargetIsCreatedCompressibleAndASampledImageIsNot()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var target = rig.Find(ImageAddress, GpuFormat.R8G8B8A8Unorm, ImageRole.ColorTarget);
        var sampled = rig.Find(ImageAddress + 0x40000, GpuFormat.R8G8B8A8Unorm, ImageRole.Texture);

        Assert.Equal(0UL, rig.Cache.GetImage(target).Shape.Usage & Breaking);
        Assert.NotEqual(0UL, rig.Cache.GetImage(target).Shape.Usage & Mtl.TextureUsageRenderTarget);
        Assert.Equal(Breaking, rig.Cache.GetImage(sampled).Shape.Usage & Breaking);
    }

    [Fact]
    public void AShaderStoreGivesTheTargetWritableUsageAndKeepsItsTexels()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var pattern = MetalTileOracle.Pattern((int)ImageSize, 0x5EED);
        rig.Write(ImageAddress, pattern);
        var identifier = rig.Find(ImageAddress, GpuFormat.R8G8B8A8Unorm, ImageRole.ColorTarget);
        var image = rig.Cache.GetImage(identifier);
        rig.Cache.AcquireTextureView(identifier, rig.Request(ImageAddress, GpuFormat.R8G8B8A8Unorm, ImageRole.Texture));
        var before = image.Texture;

        rig.Cache.AcquireTextureView(identifier, rig.Request(ImageAddress, GpuFormat.R8G8B8A8Unorm, ImageRole.StorageImage));

        Assert.NotEqual(before, image.Texture);
        Assert.Equal(Breaking, image.Shape.Usage & Breaking);
        Assert.Equal(1, rig.Cache.TakeUsagePromotions());
        Assert.Equal(pattern, rig.ReadTexture(image.Texture));
    }

    [Fact]
    public void AnSrgbViewKeepsTheTargetCompressibleAndAReinterpretingViewDoesNot()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var identifier = rig.Find(ImageAddress, GpuFormat.R8G8B8A8Unorm, ImageRole.ColorTarget);
        var image = rig.Cache.GetImage(identifier);

        var srgb = rig.Cache.AcquireTextureView(identifier, rig.Request(ImageAddress, GpuFormat.R8G8B8A8Srgb, ImageRole.Texture));
        Assert.NotEqual(image.Texture, srgb);
        Assert.Equal(0UL, image.Shape.Usage & Breaking);
        Assert.Equal(0, rig.Cache.TakeUsagePromotions());

        var view = rig.Cache.AcquireTextureView(identifier, rig.Request(ImageAddress, GpuFormat.R32Uint, ImageRole.Texture));
        Assert.NotEqual(0UL, image.Shape.Usage & Mtl.TextureUsagePixelFormatView);
        Assert.NotEqual(image.Texture, view);
        Assert.Equal(1, rig.Cache.TakeUsagePromotions());
    }

    [Fact]
    public void APresentationViewThatExchangesRedAndBlueKeepsItsSwizzle()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var identifier = rig.Find(ImageAddress, GpuFormat.R8G8B8A8Unorm, ImageRole.ColorTarget);
        var image = rig.Cache.GetImage(identifier);
        var request = rig.Request(ImageAddress, GpuFormat.R8G8B8A8Unorm, ImageRole.Texture);
        var view = request.View with
        {
            Mapping = new GpuComponentMapping(
                GpuComponentSwizzle.B, GpuComponentSwizzle.G, GpuComponentSwizzle.R, GpuComponentSwizzle.A),
        };

        var texture = rig.Cache.AcquirePresentationView(identifier, in view);

        Assert.NotEqual(0UL, image.Shape.Usage & Mtl.TextureUsagePixelFormatView);
        Assert.Equal(1, rig.Cache.TakeUsagePromotions());
        // MTLTextureSwizzleChannels, one byte per channel: Blue=4, Green=3, Red=2, Alpha=5.
        var swizzle = (uint)MetalNative.Send(texture, MetalNative.Selector("swizzle"));
        Assert.Equal(0x05020304u, swizzle);
    }

    // A compressed texture has no pixel-format-view usage, but a view in its own format still takes
    // the descriptor's swizzle: a BC4 mask read through (R,R,R,1) must not come back with G and B zero.
    [Fact]
    public void ACompressedTextureKeepsTheSwizzleOfItsDescriptor()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var request = rig.Request(ImageAddress, GpuFormat.BC4UnormBlock, ImageRole.Texture);
        var description = request.Description;
        description.GuestFormat = GuestPixelFormat.Bc4UNorm;
        description.BytesPerBlock = 8;
        description.Data = new GuestSpan(ImageAddress, (ulong)(Side / 4) * (Side / 4) * 8);
        description.MipLayout[0] = new MipLevelLayout { Offset = 0, Size = description.Data.Size, Pitch = Side, Height = Side };
        var replicate = request.View with
        {
            Mapping = new GpuComponentMapping(
                GpuComponentSwizzle.R, GpuComponentSwizzle.R, GpuComponentSwizzle.R, GpuComponentSwizzle.One),
        };
        request = new ImageRequest(description, replicate, ImageRole.Texture);
        var identifier = rig.Cache.FindImage(ref request);

        var texture = rig.Cache.AcquireTextureView(identifier, request);

        Assert.Equal(0UL, rig.Cache.GetImage(identifier).Shape.Usage & Mtl.TextureUsagePixelFormatView);
        // MTLTextureSwizzleChannels, one byte per channel: Red=2, One=1.
        var swizzle = (uint)MetalNative.Send(texture, MetalNative.Selector("swizzle"));
        Assert.Equal(0x01020202u, swizzle);
    }

    [Fact]
    public void AnImageMadeAgainOverTheSameBytesStartsWithTheUsageItNeeded()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var identifier = rig.Find(ImageAddress, GpuFormat.R8G8B8A8Unorm, ImageRole.ColorTarget);
        var first = rig.Cache.GetImage(identifier);
        rig.Cache.AcquireTextureView(identifier, rig.Request(ImageAddress, GpuFormat.R8G8B8A8Unorm, ImageRole.StorageImage));

        // A larger surface over the same bytes replaces the image.
        var again = rig.Find(ImageAddress, GpuFormat.R8G8B8A8Unorm, ImageRole.ColorTarget, height: Side * 2);

        Assert.NotSame(first, rig.Cache.GetImage(again));
        Assert.Equal(Breaking, rig.Cache.GetImage(again).Shape.Usage & Breaking);
    }

    private sealed class Rig : IDisposable
    {
        private readonly AddressSpace _space = new();
        private readonly PageGuard _pages;
        private readonly MetalGuestMemory _memory;
        private readonly Backing _backing = new();
        private readonly bool _savedLinearViews = MetalLinearWindowView.Enabled;

        public Rig(nint device)
        {
            MetalLinearWindowView.Enabled = false;
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

        // A linear 64x64 surface of four-byte texels.
        public ImageRequest Request(ulong address, GpuFormat format, ImageRole role, uint height = Side)
        {
            var size = (ulong)Side * height * 4;
            var description = ImageDescription.Create();
            description.Data = new GuestSpan(address, size);
            description.PixelFormat = format;
            description.GuestFormat = format == GpuFormat.R32Uint ? GuestPixelFormat.Bits32UInt : GuestPixelFormat.Bits8_8_8_8UNorm;
            description.Type = GuestImageType.Color2D;
            description.Extent = new GpuExtent3D(Side, height, 1);
            description.Resources = new SubresourceCount(1, 1);
            description.Pitch = Side;
            description.BytesPerBlock = 4;
            description.Samples = 1;
            description.TileMode = GuestTileMode.Linear;
            description.MipLayout[0] = new MipLevelLayout { Offset = 0, Size = size, Pitch = Side, Height = height };
            var view = ImageViewDescription.Default with
            {
                Format = format,
                Type = GpuImageViewType.Type2D,
                Aspect = GpuImageAspect.ColorBit,
                LayerCount = 1,
                Usage = role == ImageRole.ColorTarget ? GpuImageUsage.ColorAttachmentBit : GpuImageUsage.SampledBit,
            };
            return new ImageRequest(description, view, role);
        }

        public ResourceSlotIdentifier Find(ulong address, GpuFormat format, ImageRole role, uint height = Side)
        {
            var request = Request(address, format, role, height);
            return Cache.FindImage(ref request);
        }

        // Every recorded transfer runs, then the texture comes back through a buffer.
        public byte[] ReadTexture(nint texture)
        {
            var buffer = Harness.Buffer(ImageSize);
            var blit = MetalNative.Send(Harness.CommandBuffer, MetalNative.Selector("blitCommandEncoder"));
            MetalNative.SendCopyTextureToBufferOptions(
                blit,
                MetalNative.Selector(
                    "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:" +
                    "toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:options:"),
                texture, 0, 0, default, new MtlSize { Width = Side, Height = Side, Depth = 1 },
                buffer, 0, Side * 4, (nuint)ImageSize, 0);
            MetalNative.SendVoid(blit, MetalNative.Selector("endEncoding"));
            Harness.Submit();
            Harness.Begin();
            return Harness.Read(buffer, 0, ImageSize);
        }

        public void Dispose()
        {
            Cache.StopWatching();
            Cache.Dispose();
            _pages.Dispose();
            Harness.Dispose();
            _backing.Dispose();
            MetalLinearWindowView.Enabled = _savedLinearViews;
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
        public const ulong Size = 1024 * 1024;
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
            output.WriteLine("Compressible target usage runs on the device only behind the exclusive device gate.");
            return false;
        }

        Assert.True(OperatingSystem.IsMacOS());
        Assert.NotEqual(0, Device.Value);
        return true;
    }

    private static readonly Lazy<nint> Device = new(MetalNative.MTLCreateSystemDefaultDevice);
}
