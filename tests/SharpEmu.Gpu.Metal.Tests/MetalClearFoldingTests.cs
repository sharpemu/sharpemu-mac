// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.HLE.Host;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Rendering;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// A guest clear waits to become the load action of the pass that next renders into
/// its texture. Only work that reads or writes that texture - a transfer naming it, a
/// dispatch binding it, a CPU read of its guest bytes - makes it land first; anything
/// else leaves it waiting. Device work is behind the exclusive device gate.
/// </summary>
public sealed unsafe partial class MetalClearFoldingTests(ITestOutputHelper output)
{
    private const ulong GuestBase = 0x2_0000_0000;
    private const ulong TargetAddress = GuestBase + 0x4000;
    private const ulong OtherAddress = GuestBase + 0x8000;
    // Wide enough that a linear row is the pitch the guest's linear layout aligns to.
    private const uint Width = 64;
    private const uint Height = 2;
    private const ulong ImageBytes = Width * Height * 4;
    private const byte GuestPattern = 0x11;
    private const nint PixelFormatRgba8Unorm = 70;
    private static readonly byte[] ClearedPixel = [0xFF, 0x00, 0x00, 0xFF];

    [Fact]
    public void AClearOutlivesATransferOfAnotherTextureAndBecomesTheNextPassLoadAction()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        _ = host.TakeClearFolding();
        using (host.BeginPreparation())
        {
            var target = rig.Target(TargetAddress);
            var attachment = ((IRenderHost)host).AcquireColorAttachment(in target);
            var other = rig.TextureOf(OtherAddress);
            rig.Clear((nint)attachment.View.Handle);
            Assert.Equal(1, host.PendingClearCount);

            ((IMetalImageCacheOwner)host).BeginTransfer(MetalTransferCause.Upload, other);
            Assert.Equal(1, host.PendingClearCount);

            host.BeginRendering(Rig.PassOver(attachment.View));
            Assert.Equal(0, host.PendingClearCount);
            host.EndRendering();
        }

        Assert.Equal((1L, 0L), host.TakeClearFolding());
        AssertCleared(rig.Read(TargetAddress));
    }

    [Fact]
    public void ATransferOfTheClearedTextureEncodesTheClearFirst()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        _ = host.TakeClearFolding();
        using (host.BeginPreparation())
        {
            var target = rig.Target(TargetAddress);
            var attachment = ((IRenderHost)host).AcquireColorAttachment(in target);
            rig.Clear((nint)attachment.View.Handle);

            // Named by the texture that owns the pixels, not the view the clear was given.
            ((IMetalImageCacheOwner)host).BeginTransfer(MetalTransferCause.Writeback, rig.TextureOf(TargetAddress));
            Assert.Equal(0, host.PendingClearCount);
        }

        Assert.Equal((0L, 1L), host.TakeClearFolding());
    }

    [Fact]
    public void ADispatchThatBindsTheClearedTextureEncodesTheClearBeforeItsEncoder()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        _ = host.TakeClearFolding();
        nint view = 0;
        try
        {
            using (host.BeginPreparation())
            {
                var target = rig.Target(TargetAddress);
                var attachment = ((IRenderHost)host).AcquireColorAttachment(in target);
                rig.Clear((nint)attachment.View.Handle);

                // A dispatch that binds only another texture leaves the clear waiting.
                host.NoteUsedResource(rig.TextureOf(OtherAddress), 1);
                Assert.NotEqual(0, host.EnsureComputeEncoder());
                Assert.Equal(1, host.PendingClearCount);
            }

            using (host.BeginPreparation())
            {
                // One that samples the cleared pixels through a view of its own sees them
                // cleared: the clear is encoded before the compute encoder opens.
                view = MetalNative.Send(
                    rig.TextureOf(TargetAddress),
                    MetalNative.Selector("newTextureViewWithPixelFormat:"),
                    PixelFormatRgba8Unorm);
                Assert.NotEqual(0, view);
                host.NoteUsedResource(view, 1);
                Assert.NotEqual(0, host.EnsureComputeEncoder());
                Assert.Equal(0, host.PendingClearCount);
            }
        }
        finally
        {
            if (view != 0)
            {
                MetalNative.SendVoid(view, MetalNative.Selector("release"));
            }
        }

        Assert.Equal((0L, 1L), host.TakeClearFolding());
        AssertCleared(rig.Read(TargetAddress));
    }

    [Fact]
    public void ACpuReadOfTheClearedBytesSeesTheClear()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        using (host.BeginPreparation())
        {
            var target = rig.Target(TargetAddress);
            var attachment = ((IRenderHost)host).AcquireColorAttachment(in target);
            ((IMetalImageCacheOwner)host).BeginTransfer(MetalTransferCause.Upload, rig.TextureOf(OtherAddress));
            rig.Clear((nint)attachment.View.Handle);
            ((IMetalImageCacheOwner)host).BeginTransfer(MetalTransferCause.Upload, rig.TextureOf(OtherAddress));
            Assert.Equal(1, host.PendingClearCount);
        }

        AssertCleared(rig.Read(TargetAddress));
        Assert.Equal(0, host.PendingClearCount);
    }

    private static void AssertCleared(byte[] bytes)
    {
        for (var offset = 0; offset < bytes.Length; offset += 4)
        {
            Assert.Equal(ClearedPixel, bytes[offset..(offset + 4)]);
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

        // Off unless a test turns it on: whether the backing admits an access through its alias, and
        // an address at which one view ends and the next begins, so no alias spans it.
        public bool AliasAccess { get; set; }
        public ulong ViewBoundary { get; set; }

        public bool TryEnterBackingAliasAccess() => AliasAccess;

        public bool TryResolveBackingAlias(ulong address, ulong size, out ulong alias)
        {
            var crossesView = ViewBoundary != 0 && address < ViewBoundary && address + size > ViewBoundary;
            alias = Contains(address, size) && !crossesView ? (ulong)_block + (address - GuestBase) : 0;
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

        public bool AliasAccess { set => _backing.AliasAccess = value; }

        public ulong ViewBoundary { set => _backing.ViewBoundary = value; }

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

        public nint TextureOf(ulong address)
        {
            var request = Request(address, ImageRole.Texture);
            var texture = Host.ImageCache.GetImage(Host.ImageCache.FindImage(ref request)).Texture;
            Assert.NotEqual(0, texture);
            return texture;
        }

        public void Clear(nint texture)
        {
            var color = default(GpuClearColor);
            color.Float32_0 = 1f;
            color.Float32_3 = 1f;
            ((IMetalImageCacheOwner)Host).ClearTextureColor(
                texture, (ulong)PixelFormatRgba8Unorm, new SubresourceRange(0, 1, 0, 1), Width, Height, in color);
        }

        public static RenderingState PassOver(GpuImageHandle view)
        {
            var state = new RenderingState { Width = Width, Height = Height, Layers = 1, ColorAttachmentCount = 1, Samples = 1 };
            state.ColorAttachments[0] = new RenderingAttachment(
                view, GpuFormat.R8G8B8A8Unorm,
                0, 0, 0, 0, false, false, false, false, false);
            return state;
        }

        public uint ReadBacking32(ulong address)
        {
            Span<byte> bytes = stackalloc byte[sizeof(uint)];
            Assert.True(_backing.TryReadBacking(address, bytes));
            return System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(bytes);
        }

        public void WriteBacking32(ulong address, uint value)
        {
            Span<byte> bytes = stackalloc byte[sizeof(uint)];
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(bytes, value);
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
            output.WriteLine("Clear folding is encoded on the device only behind the exclusive device gate.");
            return false;
        }

        Assert.True(OperatingSystem.IsMacOS());
        Assert.NotEqual(0, Device.Value);
        return true;
    }

    private static readonly Lazy<nint> Device = new(MetalNative.MTLCreateSystemDefaultDevice);
}
