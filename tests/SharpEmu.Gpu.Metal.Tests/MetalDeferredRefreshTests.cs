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
        public void FillBacking(ulong address, byte value)
        {
            Host.FlushAndWaitForGpu();
            var bytes = new byte[ImageBytes];
            Array.Fill(bytes, value);
            Assert.True(_backing.TryWriteBacking(address, bytes));
        }

        public byte[] Read(ulong address)
        {
            var bytes = new byte[ImageBytes];
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
