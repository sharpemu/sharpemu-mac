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
using SharpEmu.Libs.Gpu.Rendering;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// A CPU-written render target is uploaded when its pass begins, not when it is
/// acquired, and not at all when the pass clears the whole of it. Every other use of the
/// target before the pass still sees the guest's bytes. Device work is behind the
/// exclusive device gate.
/// </summary>
public sealed unsafe class MetalDeferredRefreshTests(ITestOutputHelper output)
{
    private const ulong GuestBase = 0x2_0000_0000;
    private const ulong TargetAddress = GuestBase + 0x4000;
    private const uint Width = 64;
    private const uint Height = 2;
    private const ulong ImageBytes = Width * Height * 4;
    private const byte GuestPattern = 0x11;
    private const byte LaterPattern = 0x22;
    private static readonly byte[] ClearedPixel = [0xFF, 0x00, 0x00, 0xFF];

    [Fact]
    public void AWholeImageClearDropsTheUploadOfACpuWrittenTarget()
    {
        if (!RequireDevice()) return;
        if (!MetalV2Diagnostics.SkipOverwrittenRefresh) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        _ = host.ImageCache.TakeSkippedRefreshes();
        var uploads = host.ImageCache.Counters.Uploads;
        using (host.BeginPreparation())
        {
            var target = rig.Target(TargetAddress);
            Assert.True(host.ImageCache.GetImage(target.Image).IsCpuDirty);
            var attachment = ((IRenderHost)host).AcquireColorAttachment(in target);
            Assert.True(host.ImageCache.HasDeferredRefreshes);

            host.BeginRendering(Rig.PassOver(attachment.View, clear: true));
            Assert.False(host.ImageCache.HasDeferredRefreshes);
            host.EndRendering();
        }

        Assert.Equal((1L, (long)ImageBytes), host.ImageCache.TakeSkippedRefreshes());
        Assert.Equal(uploads, host.ImageCache.Counters.Uploads);
        AssertPixels(rig.Read(TargetAddress), ClearedPixel);
    }

    [Fact]
    public void APassThatLoadsTheTargetUploadsItFirst()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        _ = host.ImageCache.TakeSkippedRefreshes();
        var uploads = host.ImageCache.Counters.Uploads;
        using (host.BeginPreparation())
        {
            var target = rig.Target(TargetAddress);
            var attachment = ((IRenderHost)host).AcquireColorAttachment(in target);
            Assert.Equal(MetalV2Diagnostics.SkipOverwrittenRefresh, host.ImageCache.HasDeferredRefreshes);
            host.BeginRendering(Rig.PassOver(attachment.View, clear: false));
            Assert.False(host.ImageCache.HasDeferredRefreshes);
            host.EndRendering();
        }

        Assert.Equal((0L, 0L), host.ImageCache.TakeSkippedRefreshes());
        Assert.Equal(uploads + 1, host.ImageCache.Counters.Uploads);
        // Guest memory changes behind the cache's back: what a read now returns is what
        // the pass left in the texture, written back.
        rig.FillBacking(TargetAddress, LaterPattern);
        AssertPixels(rig.Read(TargetAddress), [GuestPattern, GuestPattern, GuestPattern, GuestPattern]);
    }

    [Fact]
    public void ATextureBindingBeforeThePassSeesTheGuestBytes()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        _ = host.ImageCache.TakeSkippedRefreshes();
        var uploads = host.ImageCache.Counters.Uploads;
        var sampled = ResourceSlotIdentifier.Invalid;
        using (host.BeginPreparation())
        {
            var target = rig.Target(TargetAddress);
            _ = ((IRenderHost)host).AcquireColorAttachment(in target);
            Assert.Equal(MetalV2Diagnostics.SkipOverwrittenRefresh, host.ImageCache.HasDeferredRefreshes);

            var request = Rig.Request(TargetAddress, ImageRole.Texture);
            sampled = host.ImageCache.FindImage(ref request);
            Assert.Equal(target.Image, sampled);
            Assert.NotEqual(0, host.ImageCache.AcquireTextureView(sampled, in request));
            Assert.False(host.ImageCache.HasDeferredRefreshes);
            Assert.False(host.ImageCache.GetImage(sampled).IsCpuDirty);
            Assert.Equal(uploads + 1, host.ImageCache.Counters.Uploads);
        }

        Assert.Equal((0L, 0L), host.ImageCache.TakeSkippedRefreshes());
        // The drained acquire marked the target written, so a read writes the texture back:
        // it holds the guest's bytes, not the later ones put behind the cache's back.
        Assert.True(host.ImageCache.GetImage(sampled).IsGpuModified);
        rig.FillBacking(TargetAddress, LaterPattern);
        AssertPixels(rig.Read(TargetAddress), [GuestPattern, GuestPattern, GuestPattern, GuestPattern]);
    }

    // ---- Layered deferral (SHARPEMU_METAL_V2_LAYER_DEFER) --------------------------------------------------------
    // A four-layer D32 array whose layers are each cleared by a pass of their own, the way Demon's Souls clears its
    // shadow cascades. Round one starts from a new image; round two from the GPU-written array after the guest
    // rewrote all of it (LaterPattern). A front or back cut leaves the rest stale and uploads nothing; a cut from the
    // middle uploads, never skips; a layer sampled before its clear is uploaded first. Every order leaves guest memory
    // byte for byte as the switch off leaves it.

    private const ulong ArrayAddress = GuestBase + 0x10000;
    private const uint ArrayWidth = 128;
    private const uint ArrayHeight = 64;
    private const uint ArrayLayers = 4;
    private const float FirstClear = 0.25f;
    private const float SecondClear = 0.75f;

    public static TheoryData<uint[], int, int> LayerOrders => new()
    {
        // order, uploads on (round one + round two), layers dropped on
        { new uint[] { 0, 1, 2, 3 }, 0, 8 },
        { new uint[] { 3, 2, 1, 0 }, 0, 8 },
        // The first clear cuts the middle: the whole array is uploaded, the rest are not stale any more.
        { new uint[] { 2, 0, 3, 1 }, 2, 0 },
        { new uint[] { 1, 2, 3, 0 }, 2, 0 },
        // Layers 2 and 3 are never cleared: they are uploaded when the array is read, and only then.
        { new uint[] { 0, 1 }, 2, 4 },
    };

    [Theory]
    [MemberData(nameof(LayerOrders))]
    public void LayerClearsOfADepthArrayDropOnlyTheUploadsTheyOverwrite(uint[] order, int uploadsOn, int layersDropped)
    {
        if (!RequireDevice()) return;
        if (!MetalV2Diagnostics.SkipOverwrittenRefresh) return;
        var off = RunLayerClears(order, layerDefer: false, sampleBeforeClear: null);
        var on = RunLayerClears(order, layerDefer: true, sampleBeforeClear: null);
        output.WriteLine($"order={string.Join(",", order)} on: uploads={on.Uploads} dropped={on.Dropped} " +
            $"off: uploads={off.Uploads} dropped={off.Dropped}");

        Assert.Equal(off.Round1, on.Round1);
        Assert.Equal(off.Round2, on.Round2);
        Assert.Equal(uploadsOn, on.Uploads);
        Assert.Equal(layersDropped, on.Dropped);
        Assert.Equal(0, off.Dropped);
        Assert.True(off.Uploads >= 2, "with the switch off each round uploads the array at its first clear");
        AssertLayers(on.Round2, order, SecondClear);
    }

    // Layer 0 is cleared, leaving layers 1-3 stale; then layer 2 is sampled before its own clear. The sample uploads
    // the stale layers first (one run upload): it sees the guest's bytes, not the texture's older ones.
    [Fact]
    public void ALayerSampledBeforeItsClearIsUploadedFirst()
    {
        if (!RequireDevice()) return;
        if (!MetalV2Diagnostics.SkipOverwrittenRefresh) return;
        uint[] order = [0, 2, 1, 3];
        var off = RunLayerClears(order, layerDefer: false, sampleBeforeClear: 2);
        var on = RunLayerClears(order, layerDefer: true, sampleBeforeClear: 2);
        output.WriteLine($"on: uploads={on.Uploads} dropped={on.Dropped} off: uploads={off.Uploads}");

        Assert.Equal(off.Round1, on.Round1);
        Assert.Equal(off.Round2, on.Round2);
        Assert.Equal(off.Sampled, on.Sampled);
        // Round one samples a new array's layer 2 the texture never held: the guest's GuestPattern; round two, after
        // the guest rewrote it, LaterPattern.
        Assert.Equal([GuestPattern, GuestPattern, GuestPattern, GuestPattern], on.Sampled[0]);
        Assert.Equal([LaterPattern, LaterPattern, LaterPattern, LaterPattern], on.Sampled[1]);
        // Each round: layer 0's clear drops its layer, the sample uploads layers 1-3 as one run.
        Assert.Equal(2, on.Uploads);
        Assert.Equal(2, on.Dropped);
    }

    private sealed record LayerOutcome(byte[] Round1, byte[] Round2, long Uploads, long Dropped, List<byte[]> Sampled);

    private LayerOutcome RunLayerClears(uint[] order, bool layerDefer, uint? sampleBeforeClear)
    {
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.ImageCache.LayerDeferEnabled = layerDefer;
        _ = host.ImageCache.TakeLayerRefreshSkipped();
        var size = ArrayLayerBytes() * ArrayLayers;
        var uploads = host.ImageCache.Counters.Uploads;
        var sampled = new List<byte[]>();
        var dropped = 0L;
        ResourceSlotIdentifier image = default;
        byte[] Round(float clear)
        {
            foreach (var layer in order)
            {
                if (layer == sampleBeforeClear)
                {
                    using (host.BeginPreparation())
                    {
                        var request = DepthArrayRequest(layer, ImageRole.Texture);
                        var found = host.ImageCache.FindImage(ref request);
                        Assert.Equal(image, found);
                        Assert.NotEqual(0, host.ImageCache.AcquireTextureView(found, in request));
                        Assert.False(host.ImageCache.GetImage(found).IsCpuDirty);
                    }

                    sampled.Add(LayerTexel(host, image, layer));
                }

                using (host.BeginPreparation())
                {
                    var acquisition = ((IRenderHost)host).AcquireDepthAttachment(DepthArrayTarget(image, layer, clear: true));
                    host.BeginRendering(DepthPass(acquisition.View, clear));
                    host.EndRendering();
                }

                dropped += host.ImageCache.TakeLayerRefreshSkipped().Layers;
            }

            return rig.Read(ArrayAddress, size);
        }

        using (host.BeginPreparation())
        {
            var request = DepthArrayRequest(0, ImageRole.DepthTarget);
            image = host.ImageCache.FindImage(ref request);
        }

        var round1 = Round(FirstClear);
        rig.FillBacking(ArrayAddress, size, LaterPattern);
        host.ImageCache.InvalidateMemory(ArrayAddress, size);
        var round2 = Round(SecondClear);
        return new LayerOutcome(round1, round2, host.ImageCache.Counters.Uploads - uploads, dropped, sampled);
    }

    // The first texel of a layer, read back from the texture: texel (0, 0) is the first element of the layer's tiles.
    private static byte[] LayerTexel(MetalRenderHost host, ResourceSlotIdentifier image, uint layer)
    {
        host.FlushAndWaitForGpu();
        var texture = host.ImageCache.GetImage(image).Texture;
        var device = MetalNative.Send(texture, MetalNative.Selector("device"));
        var buffer = MetalNative.SendNewBuffer(device, MetalNative.Selector("newBufferWithLength:options:"), 4, 0);
        var queue = MetalNative.Send(device, MetalNative.Selector("newCommandQueue"));
        try
        {
            var commands = MetalNative.Send(queue, MetalNative.Selector("commandBuffer"));
            var blit = MetalNative.Send(commands, MetalNative.Selector("blitCommandEncoder"));
            MetalNative.SendCopyTextureToBuffer(
                blit, MetalNative.Selector(
                    "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:" +
                    "destinationBytesPerRow:destinationBytesPerImage:"),
                texture, layer, 0, default, new MtlSize { Width = 1, Height = 1, Depth = 1 }, buffer, 0, 4, 4);
            MetalNative.SendVoid(blit, MetalNative.Selector("endEncoding"));
            MetalNative.SendVoid(commands, MetalNative.Selector("commit"));
            MetalNative.SendVoid(commands, MetalNative.Selector("waitUntilCompleted"));
            return new ReadOnlySpan<byte>((void*)MetalNative.Send(buffer, MetalNative.Selector("contents")), 4).ToArray();
        }
        finally
        {
            MetalNative.SendVoid(buffer, MetalNative.Selector("release"));
            MetalNative.SendVoid(queue, MetalNative.Selector("release"));
        }
    }

    // Each cleared layer's first texel holds the clear; every other layer's, the guest's LaterPattern.
    private static void AssertLayers(byte[] guest, uint[] cleared, float clear)
    {
        var layerBytes = (int)ArrayLayerBytes();
        var clearBytes = BitConverter.GetBytes(clear);
        for (var layer = 0u; layer < ArrayLayers; layer++)
        {
            var expected = Array.IndexOf(cleared, layer) >= 0 ? clearBytes : [LaterPattern, LaterPattern, LaterPattern, LaterPattern];
            Assert.Equal(expected, guest[(int)(layer * layerBytes)..(int)(layer * layerBytes + 4)]);
        }
    }

    private static ulong ArrayLayerBytes()
    {
        Assert.True(TileGeometry.TryGetDepthSize(ArrayWidth, ArrayHeight, 0, GuestDepthFormat.Z32Float,
            GuestStencilFormat.Invalid, false, out _, out _, out var depthSize));
        return (ulong)depthSize.Size;
    }

    private static ImageRequest DepthArrayRequest(uint layer, ImageRole role)
    {
        var size = ArrayLayerBytes() * ArrayLayers;
        var pitch = TileGeometry.DepthPitch(ArrayWidth, 4);
        var description = ImageDescription.Create();
        description.Data = new GuestSpan(ArrayAddress, size);
        description.PixelFormat = GpuFormat.D32Sfloat;
        description.GuestFormat = GuestPixelFormat.Bits32Float;
        description.Type = GuestImageType.Color2D;
        description.Extent = new GpuExtent3D(ArrayWidth, ArrayHeight, 1);
        description.Resources = new SubresourceCount(1, ArrayLayers);
        description.Pitch = pitch;
        description.BytesPerBlock = 4;
        description.Samples = 1;
        description.TileMode = GuestTileMode.Depth;
        description.MipLayout[0] = new MipLevelLayout { Offset = 0, Size = size, Pitch = pitch, Height = ArrayHeight };
        var view = ImageViewDescription.Default with
        {
            Format = GpuFormat.D32Sfloat,
            Type = GpuImageViewType.Type2D,
            Aspect = GpuImageAspect.DepthBit,
            BaseLayer = layer,
            LayerCount = 1,
            Usage = role == ImageRole.DepthTarget ? GpuImageUsage.DepthStencilAttachmentBit : GpuImageUsage.SampledBit,
        };
        return new ImageRequest(description, view, role);
    }

    private static DepthAttachmentState DepthArrayTarget(ResourceSlotIdentifier image, uint layer, bool clear)
    {
        var request = DepthArrayRequest(layer, ImageRole.DepthTarget);
        var resolution = new DepthTargetResolution(
            request, GpuFormat.D32Sfloat, ArrayWidth, ArrayHeight, 1, false, false,
            ArrayAddress, request.Description.Data.Size, 0, 0, 0, 0, clear, false, false, false);
        var state = new DepthStencilState(
            clear, 0f, true, true, GpuCompareOp.Always, false, 0f, 1f, false, 0, false,
            StencilOperations.Default, default, StencilOperations.Default, default);
        return new DepthAttachmentState(new DepthTargetState(resolution, state), image);
    }

    private static RenderingState DepthPass(GpuImageHandle view, float clear)
    {
        var state = new RenderingState { Width = ArrayWidth, Height = ArrayHeight, Layers = 1, ColorAttachmentCount = 0, Samples = 1 };
        state.DepthStencilAttachment = new RenderingAttachment(
            view, GpuFormat.D32Sfloat, BitConverter.SingleToUInt32Bits(clear), 0, 0, 0, true, true, true, false, false);
        return state;
    }

    private static void AssertPixels(byte[] bytes, byte[] pixel)
    {
        for (var offset = 0; offset < bytes.Length; offset += 4)
        {
            Assert.Equal(pixel, bytes[offset..(offset + 4)]);
        }
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

    // One page-aligned native block seen as guest memory at GuestBase, read by the CPU
    // and by the GPU through the same bytes.
    private sealed class Backing : IGuestBackedSpace, ICpuMemory, IDisposable
    {
        public const ulong Size = 4 * 1024 * 1024;
        private readonly void* _block = NativeMemory.AlignedAlloc((nuint)Size, 16384);

        public Backing() => NativeMemory.Fill(_block, (nuint)Size, GuestPattern);

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
        private readonly nint _queue;
        private readonly AddressSpace _space = new();
        private readonly PageGuard _pages;
        private readonly Backing _backing = new();

        public Rig(nint device)
        {
            _queue = MetalNative.Send(device, MetalNative.Selector("newCommandQueue"));
            _pages = new PageGuard(_space);
            Host = new MetalRenderHost(device, _queue, _backing, _backing, _pages, new MetalGpuWorkerRelay(static () => { }));
        }

        public MetalRenderHost Host { get; }

        public static ImageRequest Request(ulong address, ImageRole role)
        {
            var description = ImageDescription.Create();
            description.Data = new GuestSpan(address, ImageBytes);
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
                Usage = role == ImageRole.ColorTarget ? GpuImageUsage.ColorAttachmentBit : GpuImageUsage.SampledBit,
            };
            return new ImageRequest(description, view, role);
        }

        public ColorTargetState Target(ulong address)
        {
            var request = Request(address, ImageRole.ColorTarget);
            var image = Host.ImageCache.FindImage(ref request);
            var resolution = new ColorTargetResolution(
                request, address, ImageBytes, new GpuExtent2D(Width, Height), 0, 0, 1, default, false, false, default);
            return new ColorTargetState(in resolution, 0, image);
        }

        // Opaque red when the pass clears.
        public static RenderingState PassOver(GpuImageHandle view, bool clear)
        {
            var state = new RenderingState { Width = Width, Height = Height, Layers = 1, ColorAttachmentCount = 1, Samples = 1 };
            var one = BitConverter.SingleToUInt32Bits(1f);
            state.ColorAttachments[0] = new RenderingAttachment(
                view, GpuFormat.R8G8B8A8Unorm,
                clear ? one : 0, 0, 0, clear ? one : 0, clear, false, false, false, false);
            return state;
        }

        // Writes guest memory directly, the way nothing the cache watches would. An upload
        // reads guest memory when the GPU runs it, so the GPU is let finish first.
        public void FillBacking(ulong address, byte value) => FillBacking(address, ImageBytes, value);

        public void FillBacking(ulong address, ulong size, byte value)
        {
            Host.FlushAndWaitForGpu();
            var bytes = new byte[size];
            Array.Fill(bytes, value);
            Assert.True(_backing.TryWriteBacking(address, bytes));
        }

        public byte[] Read(ulong address) => Read(address, ImageBytes);

        public byte[] Read(ulong address, ulong size)
        {
            var bytes = new byte[size];
            Assert.True(Host.TryReadGuest(address, bytes));
            return bytes;
        }

        public void Dispose()
        {
            Host.ImageCache.StopWatching();
            Host.Dispose();
            _pages.Dispose();
            _backing.Dispose();
            MetalNative.SendVoid(_queue, MetalNative.Selector("release"));
        }
    }

    private bool RequireDevice()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Deferred refreshes are encoded on the device only behind the exclusive device gate.");
            return false;
        }

        Assert.True(OperatingSystem.IsMacOS());
        Assert.NotEqual(0, Device.Value);
        return true;
    }

    private static readonly Lazy<nint> Device = new(MetalNative.MTLCreateSystemDefaultDevice);
}
