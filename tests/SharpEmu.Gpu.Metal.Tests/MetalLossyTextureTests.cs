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
/// SHARPEMU_METAL_V2_LOSSY_TEXTURES: a sampled-only image is created with Apple GPU lossy
/// compression when the flag, the device and the format allow it, and anything that later
/// writes it or needs more usage gets a lossless texture refilled from guest memory. Device
/// work is behind the exclusive device gate.
/// </summary>
public sealed unsafe class MetalLossyTextureTests(ITestOutputHelper output)
{
    private const ulong GuestBase = 0x2_0000_0000;
    private const ulong ImageAddress = GuestBase + 0x10000;
    private const uint Side = 64;
    private const ulong ImageSize = Side * Side * 4;
    private const ulong Breaking = Mtl.TextureUsageShaderWrite | Mtl.TextureUsagePixelFormatView;
    private const nint GpuFamilyApple8 = 1008;
    private static readonly nint SelCompressionType = MetalNative.Selector("compressionType");
    private static readonly nint SelAllocatedSize = MetalNative.Selector("allocatedSize");

    private static long CompressionOf(nint texture) => (long)MetalV2Native.SendValue64(texture, SelCompressionType);

    [Fact]
    public void EveryFormatInTheTableCreatesASmallerLossyTexture()
    {
        if (!RequireDevice()) return;
        foreach (var format in MetalImageCache.LossyPixelFormats)
        {
            var lossless = Create(format, lossy: false);
            var lossy = Create(format, lossy: true);
            output.WriteLine($"format={(MtlPixelFormat)(uint)format} lossless={lossless.Size} lossy={lossy.Size}");
            Assert.Equal(0, lossless.Kind);
            Assert.Equal(1, lossy.Kind);
            Assert.True(lossy.Size < lossless.Size, $"format {(MtlPixelFormat)(uint)format}");
        }
    }

    [Fact]
    public void ASampledImageIsExactByDefaultAndLossyWithTheFlagOn()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        // Off by default (SHARPEMU_METAL_V2_LOSSY_TEXTURES=1 turns it on).
        Assert.False(rig.Cache.LossyTextures);
        // A 64x64 texture rounds both modes to the same minimum allocation. Use a
        // larger surface so the strict footprint assertion measures compression.
        const uint CompressionSide = 256;
        rig.Cache.LossyTextures = true;
        var on = rig.Find(ImageAddress + 0x40000, GpuFormat.R8G8B8A8Unorm, ImageRole.Texture, height: CompressionSide, width: CompressionSide);
        var onImage = rig.Cache.GetImage(on);
        Assert.True(onImage.Shape.Lossy);
        Assert.Equal(Mtl.TextureUsageShaderRead, onImage.Shape.Usage);
        Assert.Equal(1, CompressionOf(onImage.Texture));

        rig.Cache.LossyTextures = false;
        var off = rig.Find(ImageAddress, GpuFormat.R8G8B8A8Unorm, ImageRole.Texture, height: CompressionSide, width: CompressionSide);
        var offImage = rig.Cache.GetImage(off);
        Assert.False(offImage.Shape.Lossy);
        Assert.Equal(Breaking, offImage.Shape.Usage & Breaking);
        Assert.Equal(0, CompressionOf(offImage.Texture));
        // Over the same shape the device holds less.
        output.WriteLine($"allocated lossless={offImage.AllocatedBytes} lossy={onImage.AllocatedBytes}");
        Assert.True(onImage.AllocatedBytes < offImage.AllocatedBytes);
        Assert.Contains("lossy=1/", rig.Cache.DescribeMemoryCensus());
    }

    [Fact]
    public void OnlyEligibleImagesAreCreatedLossy()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Cache.LossyTextures = true;

        var target = rig.Find(ImageAddress, GpuFormat.R8G8B8A8Unorm, ImageRole.ColorTarget);
        Assert.False(rig.Cache.GetImage(target).Shape.Lossy);

        var integer = rig.Find(ImageAddress + 0x10000, GpuFormat.R32Uint, ImageRole.Texture);
        Assert.False(rig.Cache.GetImage(integer).Shape.Lossy);

        var swizzled = rig.Request(ImageAddress + 0x20000, GpuFormat.R8G8B8A8Unorm, ImageRole.Texture);
        swizzled.View = swizzled.View with
        {
            Mapping = new GpuComponentMapping(
                GpuComponentSwizzle.B, GpuComponentSwizzle.G, GpuComponentSwizzle.R, GpuComponentSwizzle.A),
        };
        var swizzledImage = rig.Cache.FindImage(ref swizzled);
        Assert.False(rig.Cache.GetImage(swizzledImage).Shape.Lossy);

        var srgb = rig.Request(ImageAddress + 0x30000, GpuFormat.R8G8B8A8Unorm, ImageRole.Texture);
        srgb.View = srgb.View with { Format = GpuFormat.R8G8B8A8Srgb };
        var srgbImage = rig.Cache.FindImage(ref srgb);
        Assert.True(rig.Cache.GetImage(srgbImage).Shape.Lossy);
        Assert.NotEqual(0, rig.Cache.AcquireTextureView(srgbImage, in srgb));
        Assert.True(rig.Cache.GetImage(srgbImage).Shape.Lossy);
        Assert.Equal(0, rig.Cache.LossyPromotions);
    }

    [Fact]
    public void AStorageBindingGivesALosslessTextureWithTheGuestTexels()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Cache.LossyTextures = true;
        var pattern = MetalTileOracle.Pattern((int)ImageSize, 0x5EED);
        rig.Write(ImageAddress, pattern);
        var identifier = rig.Find(ImageAddress, GpuFormat.R8G8B8A8Unorm, ImageRole.Texture);
        var image = rig.Cache.GetImage(identifier);
        rig.Cache.AcquireTextureView(identifier, rig.Request(ImageAddress, GpuFormat.R8G8B8A8Unorm, ImageRole.Texture));
        Assert.True(image.Shape.Lossy);
        var before = image.Texture;

        rig.Cache.AcquireTextureView(identifier, rig.Request(ImageAddress, GpuFormat.R8G8B8A8Unorm, ImageRole.StorageImage));

        Assert.NotEqual(before, image.Texture);
        Assert.False(image.Shape.Lossy);
        Assert.Equal(Breaking, image.Shape.Usage & Breaking);
        Assert.Equal(0, CompressionOf(image.Texture));
        Assert.Equal(1, rig.Cache.LossyPromotions);
        Assert.Equal(pattern, rig.ReadTexture(image.Texture));
    }

    [Fact]
    public void ARenderTargetBindingGivesALosslessTextureWithRenderTargetUsage()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Cache.LossyTextures = true;
        var identifier = rig.Find(ImageAddress, GpuFormat.R8G8B8A8Unorm, ImageRole.Texture);
        var image = rig.Cache.GetImage(identifier);
        Assert.True(image.Shape.Lossy);

        rig.Cache.AcquireColorTargetView(identifier, rig.Request(ImageAddress, GpuFormat.R8G8B8A8Unorm, ImageRole.ColorTarget));

        Assert.False(image.Shape.Lossy);
        Assert.NotEqual(0UL, image.Shape.Usage & Mtl.TextureUsageRenderTarget);
        Assert.Equal(0, CompressionOf(image.Texture));
        Assert.Equal(1, rig.Cache.LossyPromotions);
    }

    [Fact]
    public void AReinterpretingViewGivesALosslessTexture()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Cache.LossyTextures = true;
        var identifier = rig.Find(ImageAddress, GpuFormat.R8G8B8A8Unorm, ImageRole.Texture);
        var image = rig.Cache.GetImage(identifier);
        Assert.True(image.Shape.Lossy);

        var view = rig.Cache.AcquireTextureView(identifier, rig.Request(ImageAddress, GpuFormat.R32Uint, ImageRole.Texture));

        Assert.NotEqual(0, view);
        Assert.False(image.Shape.Lossy);
        Assert.NotEqual(0UL, image.Shape.Usage & Mtl.TextureUsagePixelFormatView);
        Assert.Equal(0, CompressionOf(image.Texture));
        Assert.Equal(1, rig.Cache.LossyPromotions);
    }

    [Fact]
    public void AnImageMadeAgainOverTheSameBytesAfterAPromotionIsNotLossy()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Cache.LossyTextures = true;
        var identifier = rig.Find(ImageAddress, GpuFormat.R8G8B8A8Unorm, ImageRole.Texture);
        rig.Cache.AcquireTextureView(identifier, rig.Request(ImageAddress, GpuFormat.R8G8B8A8Unorm, ImageRole.StorageImage));

        // A larger surface over the same bytes replaces the image.
        var again = rig.Find(ImageAddress, GpuFormat.R8G8B8A8Unorm, ImageRole.Texture, height: Side * 2);

        Assert.False(rig.Cache.GetImage(again).Shape.Lossy);
        Assert.Equal(Breaking, rig.Cache.GetImage(again).Shape.Usage & Breaking);
    }

    // What a render target draws over the bytes of a sampled image reaches it through the
    // alias hand-over. A lossy texture would hold an approximation of the rendered data
    // (compressed again on every hand-over), which is how dark scenes turned to
    // false-colour noise: the sampled image becomes lossless and holds the texels exactly.
    [Fact]
    public void ASampledImageOverRenderedBytesHoldsThemExactly()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Cache.LossyTextures = true;
        var textureRequest = rig.Request(ImageAddress, GpuFormat.R8G8B8A8Unorm, ImageRole.Texture);
        var texture = rig.Cache.FindImage(ref textureRequest);
        rig.Cache.AcquireTextureView(texture, textureRequest);
        var image = rig.Cache.GetImage(texture);
        Assert.True(image.Shape.Lossy);

        const uint TargetRows = 16;
        var targetAddress = ImageAddress + ((ulong)Side * (Side - TargetRows) * 4);
        var targetRequest = rig.Request(targetAddress, GpuFormat.R8G8B8A8Unorm, ImageRole.ColorTarget, TargetRows);
        var target = rig.Cache.FindImage(ref targetRequest);
        Assert.NotEqual(texture, target);
        rig.Cache.AcquireColorTargetView(target, targetRequest);
        var drawn = MetalTileOracle.Pattern((int)(Side * TargetRows * 4), 0xD7A3);
        rig.FillTexture(rig.Cache.GetImage(target).Texture, drawn, TargetRows);

        rig.Cache.AcquireTextureView(texture, textureRequest);

        AssertExact(drawn, rig.ReadTexture(image.Texture).AsSpan((int)ImageSize - drawn.Length));
        Assert.False(image.Shape.Lossy);
        Assert.Equal(0, CompressionOf(image.Texture));
        Assert.Equal(1, rig.Cache.LossyGpuBytePromotions);
    }

    // A shader store into the bytes of a lossy image makes it lossless before it reloads
    // them, and a sampled image made later over bytes a shader stored is lossless from the
    // start; bytes no GPU write touched still get a lossy texture.
    [Fact]
    public void BytesAShaderStoredAreNeverHeldLossy()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Cache.LossyTextures = true;
        var request = rig.Request(ImageAddress, GpuFormat.R8G8B8A8Unorm, ImageRole.Texture);
        var identifier = rig.Cache.FindImage(ref request);
        rig.Cache.AcquireTextureView(identifier, request);
        var image = rig.Cache.GetImage(identifier);
        Assert.True(image.Shape.Lossy);

        var stored = MetalTileOracle.Pattern((int)ImageSize, 0x51);
        rig.Write(ImageAddress, stored);
        rig.Cache.InvalidateMemoryFromGpu(ImageAddress, ImageSize);
        rig.Cache.AcquireTextureView(identifier, request);
        AssertExact(stored, rig.ReadTexture(image.Texture));
        Assert.False(image.Shape.Lossy);
        Assert.Equal(1, rig.Cache.LossyGpuBytePromotions);

        const ulong Stored = ImageAddress + 0x40000;
        rig.Cache.InvalidateMemoryFromGpu(Stored, ImageSize);
        Assert.False(rig.Cache.GetImage(rig.Find(Stored, GpuFormat.R8G8B8A8Unorm, ImageRole.Texture)).Shape.Lossy);
        Assert.Equal(1, rig.Cache.LossyRefusedGpuBytes);
        Assert.True(rig.Cache.GetImage(rig.Find(ImageAddress + 0x80000, GpuFormat.R8G8B8A8Unorm, ImageRole.Texture)).Shape.Lossy);
    }

    // A kernel copy hands its source's texels on as guest bytes: a lossy source is made
    // lossless, holding the guest bytes exactly, before the copy reads it.
    [Fact]
    public void AKernelCopyReadsALossySourceAsItsExactBytes()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Cache.LossyTextures = true;
        const ulong SourceAddress = GuestBase + 0x20000;
        var pattern = MetalTileOracle.Pattern((int)ImageSize, 0xC0);
        rig.Write(SourceAddress, pattern);
        var sourceRequest = rig.Request(SourceAddress, GpuFormat.R8G8B8A8Unorm, ImageRole.Texture);
        var source = rig.Cache.FindImage(ref sourceRequest);
        rig.Cache.AcquireTextureView(source, sourceRequest);
        Assert.True(rig.Cache.GetImage(source).Shape.Lossy);
        rig.Find(ImageAddress, GpuFormat.R8G8B8A8Unorm, ImageRole.Texture);

        Assert.True(rig.Cache.TryBeginKernelCopy(
            ImageAddress, SourceAddress, ImageSize, out var from, out _, out var destination, out var fallback), fallback);

        Assert.Same(rig.Cache.GetImage(source), from);
        AssertExact(pattern, rig.ReadTexture(from!.Texture));
        Assert.False(from.Shape.Lossy);
        Assert.Equal(0, CompressionOf(from.Texture));
        rig.Cache.CompleteKernelCopy(destination);
    }

    [Fact]
    public void GpuWrittenGranulesAreFoundAtTheirEdges()
    {
        var granules = new MetalGpuWrittenGranules();
        Assert.False(granules.Overlaps(0x2_0000_0000, 0x1000));

        // Two bytes across a granule boundary mark both granules.
        granules.Mark(0x2_0000_FFFF, 2);
        Assert.Equal(2, granules.Count);
        Assert.True(granules.Overlaps(0x2_0000_0000, 0x1000));
        Assert.True(granules.Overlaps(0x2_0001_FFFF, 1));
        Assert.False(granules.Overlaps(0x2_0002_0000, 0x10000));

        // A write across a 1 GiB bitmap boundary, partly marked already.
        granules.Mark(0x2_0000_0000, 0x4001_0000);
        Assert.Equal(16385, granules.Count);
        Assert.True(granules.Overlaps(0x2_4000_0000, 1));
        Assert.False(granules.Overlaps(0x2_4001_0000, 0x1000));
        Assert.True(granules.Overlaps(0x1_FFFF_0000, 0x10001));
    }

    private void AssertExact(ReadOnlySpan<byte> expected, ReadOnlySpan<byte> actual)
    {
        var mismatches = 0;
        var largest = 0;
        for (var i = 0; i < expected.Length; i++)
        {
            var difference = Math.Abs(expected[i] - actual[i]);
            mismatches += difference != 0 ? 1 : 0;
            largest = Math.Max(largest, difference);
        }

        output.WriteLine($"bytes={expected.Length} mismatched={mismatches} largest_difference={largest}");
        Assert.True(mismatches == 0, $"{mismatches} of {expected.Length} bytes differ, by up to {largest}");
    }

    private static (long Kind, ulong Size) Create(ulong format, bool lossy)
    {
        var descriptor = MetalNative.Send(
            MetalNative.Send(MetalNative.Class("MTLTextureDescriptor"), MetalNative.Selector("alloc")), MetalNative.Selector("init"));
        MetalV2Native.SendVoidValue(descriptor, MetalNative.Selector("setTextureType:"), (nuint)Mtl.TextureType2D);
        MetalV2Native.SendVoidValue(descriptor, MetalNative.Selector("setPixelFormat:"), (nuint)format);
        MetalV2Native.SendVoidValue(descriptor, MetalNative.Selector("setWidth:"), 1024);
        MetalV2Native.SendVoidValue(descriptor, MetalNative.Selector("setHeight:"), 1024);
        MetalV2Native.SendVoidValue(descriptor, MetalNative.Selector("setMipmapLevelCount:"), 11);
        MetalV2Native.SendVoidValue(descriptor, MetalNative.Selector("setUsage:"), (nuint)Mtl.TextureUsageShaderRead);
        MetalV2Native.SendVoidValue(descriptor, MetalNative.Selector("setStorageMode:"), (nuint)Mtl.StorageModePrivate);
        MetalV2Native.SendVoidValue(descriptor, MetalNative.Selector("setCompressionType:"), lossy ? 1u : 0u);
        var texture = MetalNative.Send(Device.Value, MetalNative.Selector("newTextureWithDescriptor:"), descriptor);
        Assert.NotEqual(0, texture);
        var result = (CompressionOf(texture), MetalV2Native.SendValue64(texture, SelAllocatedSize));
        MetalNative.SendVoid(texture, MetalNative.Selector("release"));
        MetalNative.SendVoid(descriptor, MetalNative.Selector("release"));
        return result;
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
        }

        public MetalTransferHarness Harness { get; }
        public MetalImageCache Cache { get; }

        public void Write(ulong address, ReadOnlySpan<byte> bytes) => Assert.True(_backing.TryWriteBacking(address, bytes));

        // A linear 64x64 surface of four-byte texels.
        public ImageRequest Request(ulong address, GpuFormat format, ImageRole role, uint height = Side, uint width = Side)
        {
            var size = (ulong)width * height * 4;
            var description = ImageDescription.Create();
            description.Data = new GuestSpan(address, size);
            description.PixelFormat = format;
            description.GuestFormat = format == GpuFormat.R32Uint ? GuestPixelFormat.Bits32UInt : GuestPixelFormat.Bits8_8_8_8UNorm;
            description.Type = GuestImageType.Color2D;
            description.Extent = new GpuExtent3D(width, height, 1);
            description.Resources = new SubresourceCount(1, 1);
            description.Pitch = width;
            description.BytesPerBlock = 4;
            description.Samples = 1;
            description.TileMode = GuestTileMode.Linear;
            description.MipLayout[0] = new MipLevelLayout { Offset = 0, Size = size, Pitch = width, Height = height };
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

        public ResourceSlotIdentifier Find(ulong address, GpuFormat format, ImageRole role, uint height = Side, uint width = Side)
        {
            var request = Request(address, format, role, height, width);
            return Cache.FindImage(ref request);
        }

        // What a pass would draw: the bytes copied straight into the first rows of the texture.
        public void FillTexture(nint texture, byte[] bytes, uint height)
        {
            var buffer = Harness.Buffer(bytes);
            var blit = MetalNative.Send(Harness.CommandBuffer, MetalNative.Selector("blitCommandEncoder"));
            MetalV2Native.SendCopyBufferToTextureOptions(
                blit,
                MetalNative.Selector(
                    "copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:" +
                    "toTexture:destinationSlice:destinationLevel:destinationOrigin:options:"),
                buffer, 0, Side * 4, (nuint)bytes.Length, new MtlSize { Width = Side, Height = height, Depth = 1 },
                texture, 0, 0, default, 0);
            MetalNative.SendVoid(blit, MetalNative.Selector("endEncoding"));
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
            output.WriteLine("Lossy texture creation runs on the device only behind the exclusive device gate.");
            return false;
        }

        Assert.True(OperatingSystem.IsMacOS());
        Assert.NotEqual(0, Device.Value);
        if (!MetalNative.SendBool(Device.Value, MetalNative.Selector("supportsFamily:"), GpuFamilyApple8))
        {
            output.WriteLine("This device has no lossy texture compression (it needs Apple8 or later).");
            return false;
        }

        return true;
    }

    private static readonly Lazy<nint> Device = new(MetalNative.MTLCreateSystemDefaultDevice);
}
