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
/// An image that takes GPU ownership of bytes it shares with another cached image leaves
/// that image stale over the shared bytes and nowhere else. Marking the rest stale too
/// costs a refresh of bytes guest memory and the texture already agree on - a whole 4K
/// target re-uploaded because a smaller surface overlapping its tail was drawn to.
/// Device work is behind the exclusive device gate.
/// </summary>
public sealed unsafe partial class MetalImageAliasTests(ITestOutputHelper output)
{
    private const ulong GuestBase = 0x2_0000_0000;
    private const ulong ImageAddress = GuestBase + 0x10000;

    // 64x64 RGBA8, linear: 16 KiB.
    private const uint LargeSide = 64;
    private const ulong LargeSize = LargeSide * LargeSide * 4;

    // 32x32 RGBA8, linear: 4 KiB.
    private const uint SmallSide = 32;
    private const ulong SmallSize = SmallSide * SmallSide * 4;

    // The rows of a 64-wide RGBA8 alias that make the same 4 KiB.
    private const uint AliasRows = (uint)(SmallSize / (LargeSide * 4));

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
    private sealed class Backing : IGuestBackedSpace, IDisposable
    {
        public const ulong Size = 1024 * 1024;
        private readonly void* _block = NativeMemory.AlignedAlloc((nuint)Size, 16384);

        public Backing() => NativeMemory.Clear(_block, (nuint)Size);

        public void Dispose() => NativeMemory.AlignedFree(_block);

        // A guest address no resolved range may cross, as if the bytes either side of it
        // were two mappings that are not one span; zero for none.
        public ulong Split { get; set; }

        private bool Contains(ulong address, ulong length) => address >= GuestBase && address - GuestBase + length <= Size;

        private bool Crosses(ulong address, ulong length) => Split != 0 && address < Split && address + length > Split;

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
            alias = Contains(address, size) && !Crosses(address, size) ? (ulong)_block + (address - GuestBase) : 0;
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

    private sealed class NoMemory : ICpuMemory
    {
        public bool TryRead(ulong virtualAddress, Span<byte> destination) => false;
        public bool TryWrite(ulong virtualAddress, ReadOnlySpan<byte> source) => false;
        public bool CanRead(ulong address, ulong size) => false;
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
        public void ClearTextureColor(nint texture, ulong pixelFormat, in SubresourceRange range, uint width, uint height, in GpuClearColor color) { }

        public void ClearTextureDepthStencil(nint texture, ulong pixelFormat, in SubresourceRange range, uint width, uint height,
            float depth, byte stencil, bool clearDepth, bool clearStencil)
        {
        }
    }

    private sealed class Rig : IDisposable
    {
        private readonly AddressSpace _space = new();
        private readonly PageGuard _pages;
        private readonly MetalGuestMemory _memory;
        private readonly Backing _backing = new();

        private readonly bool _savedLinearViews = MetalLinearWindowView.Enabled;

        // The alias tests are about owned textures; window views have their own tests.
        public Rig(nint device, bool linearViews = false)
        {
            MetalLinearWindowView.Enabled = linearViews;
            Harness = new MetalTransferHarness(device);
            Harness.Begin();
            _pages = new PageGuard(_space);
            _memory = new MetalGuestMemory(device, _backing, new NoMemory());
            Cache = new MetalImageCache(new Owner(Harness), Harness.Transfer, _memory, _pages, _backing);
        }

        public MetalTransferHarness Harness { get; }
        public MetalImageCache Cache { get; }

        public void SplitGuestAt(ulong address) => _backing.Split = address;

        // Straight into the backing, past the page watch, the way a store the cache has
        // already been told about would land.
        public void WriteGuest(ulong address, byte[] bytes) => Assert.True(_backing.TryWriteBacking(address, bytes));

        public byte[] ReadGuest(ulong address, ulong size)
        {
            var bytes = new byte[size];
            Assert.True(_backing.TryReadBacking(address, bytes));
            return bytes;
        }

        // A linear RGBA8 square, which every byte offset of is its own row-aligned run.
        public static ImageRequest Request(ulong address, uint side, ImageRole role) =>
            Request(address, side, side, role);

        public static ImageRequest Request(ulong address, uint width, uint height, ImageRole role)
        {
            var size = (ulong)width * height * 4;
            var description = ImageDescription.Create();
            description.Data = new GuestSpan(address, size);
            description.PixelFormat = GpuFormat.R8G8B8A8Unorm;
            description.GuestFormat = GuestPixelFormat.Bits8_8_8_8UNorm;
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
                Format = GpuFormat.R8G8B8A8Unorm,
                Type = GpuImageViewType.Type2D,
                Aspect = GpuImageAspect.ColorBit,
                LayerCount = 1,
                Usage = role == ImageRole.ColorTarget ? GpuImageUsage.ColorAttachmentBit : GpuImageUsage.SampledBit,
            };
            return new ImageRequest(description, view, role);
        }

        // A 64 KiB-tiled RGBA8 square target: a macro tile is 128x128 texels, so every 128
        // rows are one macro tile row.
        public static ImageRequest TiledRequest(ulong address, uint side)
        {
            var request = Request(address, side, ImageRole.ColorTarget);
            var description = request.Description;
            var size = TileGeometry.TextureTotalSize(
                GuestPixelFormat.Bits8_8_8_8UNorm, side, side, 1, 1, GuestTileMode.Standard64KB, volume: false).Size;
            description.Data = new GuestSpan(address, size);
            description.Pitch = TileGeometry.TexturePitch(GuestPixelFormat.Bits8_8_8_8UNorm, side, GuestTileMode.Standard64KB);
            description.TileMode = GuestTileMode.Standard64KB;
            description.MipLayout[0] = new MipLevelLayout { Offset = 0, Size = size, Pitch = description.Pitch, Height = side };
            return new ImageRequest(description, request.View, ImageRole.ColorTarget);
        }

        // A linear RGBA8 array of square layers, each a whole 2D surface of its own bytes.
        public static ImageRequest ArrayRequest(ulong address, uint side, uint layers, ImageRole role)
        {
            var request = Request(address, side, role);
            var description = request.Description;
            var size = (ulong)side * side * 4 * layers;
            description.Data = new GuestSpan(address, size);
            description.Resources = new SubresourceCount(1, layers);
            description.MipLayout[0] = new MipLevelLayout { Offset = 0, Size = size, Pitch = side, Height = side };
            var view = request.View with { Type = GpuImageViewType.Type2DArray, LayerCount = layers };
            return new ImageRequest(description, view, role);
        }

        public ResourceSlotIdentifier Acquire(ulong address, uint side, ImageRole role) =>
            Acquire(Request(address, side, role));

        public ResourceSlotIdentifier Acquire(ImageRequest request)
        {
            var role = request.Role;
            var identifier = Cache.FindImage(ref request);
            if (role == ImageRole.ColorTarget)
            {
                Cache.AcquireColorTargetView(identifier, request);
            }
            else
            {
                Cache.AcquireTextureView(identifier, request);
            }

            return identifier;
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

    // The small surface sits at the large one's tail, in its middle, and straddling its
    // end; the large image goes stale over exactly the bytes the two share.
    [Theory]
    [InlineData(LargeSize - SmallSize, "tail")]
    [InlineData(0x2000UL, "middle")]
    [InlineData(LargeSize - (SmallSize / 2), "straddling the end")]
    public void ATargetWriteDirtiesOnlyTheSharedBytesOfAnAlias(ulong offset, string name)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var large = rig.Acquire(ImageAddress, LargeSide, ImageRole.Texture);
        var image = rig.Cache.GetImage(large);
        Assert.False(image.IsCpuDirty);

        rig.Acquire(ImageAddress + offset, SmallSide, ImageRole.ColorTarget);

        Assert.True(image.IsCpuDirty);
        Assert.Equal(offset, image.DirtyStart);
        Assert.Equal(Math.Min(LargeSize, offset + SmallSize), image.DirtyEnd);
        output.WriteLine($"{name}: dirty 0x{image.DirtyStart:x}..0x{image.DirtyEnd:x} of 0x{LargeSize:x}");
    }

    // The stale run at the tail of a linear surface is refreshed alone: a partial upload
    // from the row it starts in, rather than the whole image.
    [Fact]
    public void ATailAliasRefreshesOnlyItsRun()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var large = rig.Acquire(ImageAddress, LargeSide, ImageRole.Texture);
        var image = rig.Cache.GetImage(large);
        rig.Acquire(ImageAddress + LargeSize - SmallSize, SmallSide, ImageRole.ColorTarget);
        Assert.Equal(LargeSize - SmallSize, image.DirtyStart);
        _ = rig.Cache.TakePartialTransfers();

        rig.Acquire(ImageAddress, LargeSide, ImageRole.Texture);

        Assert.False(image.IsCpuDirty);
        var partial = rig.Cache.TakePartialTransfers();
        Assert.Equal(1, partial.Uploads);
        Assert.Equal((long)(LargeSize - SmallSize), partial.UploadBytes);
    }

    // A read through an alias whose rows do not line up with the GPU-written image's
    // (here a narrower pitch over the same bytes) is served nothing: the image is not
    // retiled out to guest memory and stays GPU-modified, and the reader is not marked
    // stale over the run. A reader whose layout does fit still gets the run published.
    [Fact]
    public void AReadThroughAMismatchedLayoutSkipsTheHandover()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var (image, _, stale) = GpuWrittenLarge(rig);
        _ = rig.Cache.AliasCounters;

        var narrow = rig.Cache.GetImage(rig.Acquire(Rig.Request(ImageAddress, SmallSide, SmallSide, ImageRole.Texture)));
        rig.Harness.Submit();
        rig.Harness.Begin();

        Assert.Equal(stale, rig.ReadGuest(ImageAddress, LargeSize));
        Assert.True(image.IsGpuModified);
        Assert.False(narrow.IsCpuDirty);
        var counters = rig.Cache.AliasCounters;
        Assert.Equal(1, counters.ReadsSkipped);
        Assert.Equal(0, counters.Published);

        rig.Acquire(Rig.Request(ImageAddress + 0x2000, LargeSide, AliasRows, ImageRole.Texture));
        rig.Harness.Submit();
        rig.Harness.Begin();

        var guest = rig.ReadGuest(ImageAddress, LargeSize);
        Assert.NotEqual(stale.AsSpan(0x2000, (int)SmallSize).ToArray(), guest.AsSpan(0x2000, (int)SmallSize).ToArray());
        counters = rig.Cache.AliasCounters;
        Assert.Equal(0, counters.ReadsSkipped);
        Assert.Equal(1, counters.Published);
    }

    // A block-compressed pitch counts texels, four to a block: a BC3 surface and the
    // RGBA32Uint view of its blocks have the same rows, so they share a texel layout.
    [Fact]
    public void ABlockSurfaceAndItsUintBlockViewShareATexelLayout()
    {
        var blocks = BlockAliasRequest(ImageAddress, blockCompressed: false, ImageRole.ColorTarget).Description;
        var texels = BlockAliasRequest(ImageAddress, blockCompressed: true, ImageRole.Texture).Description;
        Assert.True(MetalImageCache.SameTexelLayout(in blocks, in texels));
        Assert.True(MetalImageCache.SameTexelLayout(in texels, in blocks));

        var wider = texels;
        wider.Pitch = BlockAliasSide * 4 * 2;
        Assert.False(MetalImageCache.SameTexelLayout(in blocks, in wider));
    }

    // UE5's virtual texture pool: a compute shader writes BC3 blocks through an RGBA32Uint
    // view over the pool, and materials sample the pool as BC3. The sampled surface has to
    // get the blocks the GPU wrote; a skipped handover left it with stale tiles (Silent
    // Hill: The Short Message, a jacket patched with other textures' tiles).
    // Through guest memory: the words are published and the blocks reloaded from there.
    [Fact]
    public void ABlockSurfaceReadsTheBlocksItsUintViewWrote()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Harness.Transfer.BlockAliasOnDevice = false;
        var (written, _) = GpuWrittenBlockWords(rig);

        var texels = rig.Cache.GetImage(rig.Acquire(BlockAliasRequest(ImageAddress, blockCompressed: true, ImageRole.Texture)));
        rig.Harness.Submit();
        rig.Harness.Begin();

        Assert.Equal(written, rig.ReadGuest(ImageAddress, BlockAliasSize));
        Assert.Equal(written, ReadTexture(rig, texels));
        var counters = rig.Cache.AliasCounters;
        Assert.Equal(0, counters.ReadsSkipped);
        Assert.Equal(1, counters.Published);
    }

    // On the device: the blocks are copied across and guest memory is left as it was, the
    // word view still owing it the bytes.
    [Fact]
    public void ABlockSurfaceTakesTheBlocksItsUintViewWroteOnTheDevice()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        Assert.True(rig.Harness.Transfer.BlockAliasOnDevice);
        var (written, stale) = GpuWrittenBlockWords(rig);
        _ = rig.Cache.TakeDeviceAliasPublishes();

        var texels = rig.Cache.GetImage(rig.Acquire(BlockAliasRequest(ImageAddress, blockCompressed: true, ImageRole.Texture)));
        rig.Harness.Submit();
        rig.Harness.Begin();

        Assert.Equal(written, ReadTexture(rig, texels));
        Assert.Equal(stale, rig.ReadGuest(ImageAddress, BlockAliasSize));
        Assert.False(texels.IsCpuDirty);
        Assert.Equal(1, rig.Cache.TakeDeviceAliasPublishes());
        Assert.Equal(0, rig.Cache.AliasCounters.ReadsSkipped);
    }

    // The RGBA32Uint view holding blocks guest memory does not: what a compute shader wrote.
    private static (byte[] Written, byte[] Stale) GpuWrittenBlockWords(Rig rig)
    {
        var written = MetalTileOracle.Pattern((int)BlockAliasSize, 0x5A);
        rig.WriteGuest(ImageAddress, written);
        var words = rig.Cache.GetImage(
            rig.Acquire(BlockAliasRequest(ImageAddress, blockCompressed: false, ImageRole.ColorTarget)));
        Assert.True(words.IsGpuModified);
        rig.Harness.Submit();
        rig.Harness.Begin();
        var stale = MetalTileOracle.Pattern((int)BlockAliasSize, 0xC3);
        rig.WriteGuest(ImageAddress, stale);
        _ = rig.Cache.AliasCounters;
        return (written, stale);
    }

    // The bytes the image's texture holds, laid out as its (linear) guest bytes.
    private static byte[] ReadTexture(Rig rig, MetalCachedImage image)
    {
        var description = image.Description;
        var size = description.Data.Size;
        var buffer = rig.Harness.Buffer(size);
        rig.Harness.Transfer.EncodeTextureToGuest(
            rig.Harness, in description, new SubresourceRange(0, 1, 0, 1), image.Texture,
            new MetalGuestRange(buffer, 0, size, 0), default, 0, size, 0);
        rig.Harness.Submit();
        rig.Harness.Begin();
        return rig.Harness.Read(buffer, 0, size);
    }

    // 16x16 RGBA32Uint blocks over a 64x64 BC3 surface, linear: 4 KiB.
    private const uint BlockAliasSide = 16;
    private const ulong BlockAliasSize = BlockAliasSide * BlockAliasSide * 16;

    private static ImageRequest BlockAliasRequest(ulong address, bool blockCompressed, ImageRole role)
    {
        var side = blockCompressed ? BlockAliasSide * 4 : BlockAliasSide;
        var format = blockCompressed ? GpuFormat.BC3UnormBlock : GpuFormat.R32G32B32A32Uint;
        var description = ImageDescription.Create();
        description.Data = new GuestSpan(address, BlockAliasSize);
        description.PixelFormat = format;
        description.GuestFormat = blockCompressed ? GuestPixelFormat.Bc3UNorm : GuestPixelFormat.Bits32_32_32_32UInt;
        description.Type = GuestImageType.Color2D;
        description.Extent = new GpuExtent3D(side, side, 1);
        description.Resources = new SubresourceCount(1, 1);
        description.Pitch = side;
        description.BytesPerBlock = 16;
        description.Samples = 1;
        description.TileMode = GuestTileMode.Linear;
        description.MipLayout[0] = new MipLevelLayout { Offset = 0, Size = BlockAliasSize, Pitch = side, Height = side };
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

    // A write through a small alias inside a GPU-written image publishes only the run the
    // two share and leaves the large image GPU-modified for the rest: stale over the run
    // alone, with the rest of guest memory untouched by the handover.
    [Fact]
    public void AWriteThroughASmallAliasPublishesOnlyTheSharedRun()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var (image, written, stale) = GpuWrittenLarge(rig);
        _ = rig.Cache.AliasCounters;

        rig.Acquire(Rig.Request(ImageAddress + 0x2000, LargeSide, AliasRows, ImageRole.ColorTarget));
        rig.Harness.Submit();
        rig.Harness.Begin();

        var guest = rig.ReadGuest(ImageAddress, LargeSize);
        Assert.Equal(written.AsSpan(0x2000, (int)SmallSize).ToArray(), guest.AsSpan(0x2000, (int)SmallSize).ToArray());
        Assert.Equal(stale.AsSpan(0, 0x2000).ToArray(), guest.AsSpan(0, 0x2000).ToArray());
        Assert.Equal(
            stale.AsSpan(0x2000 + (int)SmallSize).ToArray(), guest.AsSpan(0x2000 + (int)SmallSize).ToArray());
        Assert.True(image.IsGpuModified);
        Assert.True(image.IsCpuDirty);
        Assert.Equal(0x2000UL, image.DirtyStart);
        Assert.Equal(0x2000UL + SmallSize, image.DirtyEnd);
        Assert.Equal(1, rig.Cache.AliasCounters.Narrow);
    }

    // The large image's next use reloads only the run the alias took, and picks up what
    // the alias wrote there while its own bytes elsewhere survive.
    [Fact]
    public void TheLargeImageReloadsOnlyTheRunTheAliasTook()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var (image, written, _) = GpuWrittenLarge(rig);
        var drawn = DrawSmallAlias(rig, 0x2000);
        _ = rig.Cache.TakePartialTransfers();

        rig.Acquire(ImageAddress, LargeSide, ImageRole.Texture);
        rig.Harness.Submit();
        rig.Harness.Begin();

        Assert.False(image.IsCpuDirty);
        Assert.True(image.IsGpuModified);
        var partial = rig.Cache.TakePartialTransfers();
        Assert.Equal(1, partial.Uploads);
        Assert.Equal((long)(LargeSize - SmallSize), partial.UploadBytes);

        rig.Cache.FlushGpuWrites(ImageAddress, LargeSize, MetalWritebackCause.HostRead);
        rig.Harness.Submit();
        rig.Harness.Begin();
        var expected = (byte[])written.Clone();
        drawn.CopyTo(expected, 0x2000);
        Assert.Equal(expected, rig.ReadGuest(ImageAddress, LargeSize));
    }

    // A whole writeback of an image stale over a run reloads the run first, so the bytes
    // an alias published there are not overwritten by the ones the texture held before.
    [Fact]
    public void AWholeWritebackOfAnImageStaleOverARunKeepsTheGuestBytesThere()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var (image, written, _) = GpuWrittenLarge(rig);
        DrawSmallAlias(rig, 0x2000);
        // What the alias publishes over the run, straight into guest memory.
        var published = MetalTileOracle.Pattern((int)SmallSize, 0x99);
        rig.WriteGuest(ImageAddress + 0x2000, published);

        // A host read of the large image's first bytes only, which the alias does not reach.
        rig.Cache.FlushGpuWrites(ImageAddress, 0x1000, MetalWritebackCause.HostRead);
        rig.Harness.Submit();
        rig.Harness.Begin();

        var expected = (byte[])written.Clone();
        published.CopyTo(expected, 0x2000);
        Assert.Equal(expected, rig.ReadGuest(ImageAddress, LargeSize));
        Assert.False(image.IsGpuModified);
        Assert.True(image.IsCpuDirty);
        Assert.Equal(0x2000UL, image.DirtyStart);
        Assert.Equal(0x2000UL + SmallSize, image.DirtyEnd);
    }

    // A run past half of the GPU-written image cannot be handed over narrowly - its reload
    // would be a whole one - so the image is written back whole and gives up ownership,
    // as before.
    [Fact]
    public void AnAliasRunPastHalfTheImageIsWrittenBackWhole()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var (image, written, _) = GpuWrittenLarge(rig);
        const uint rows = 36;
        const ulong offset = 0x1000;
        const ulong size = LargeSide * rows * 4;
        Assert.True(size > LargeSize / 2);
        _ = rig.Cache.AliasCounters;

        rig.Acquire(Rig.Request(ImageAddress + offset, LargeSide, rows, ImageRole.ColorTarget));
        rig.Harness.Submit();
        rig.Harness.Begin();

        Assert.Equal(written, rig.ReadGuest(ImageAddress, LargeSize));
        Assert.False(image.IsGpuModified);
        Assert.True(image.IsCpuDirty);
        Assert.Equal(offset, image.DirtyStart);
        Assert.Equal(offset + size, image.DirtyEnd);
        Assert.Equal(0, rig.Cache.AliasCounters.Narrow);
    }

    // A run the transfer refuses to retile (a multisampled image) put nothing in guest
    // memory, so it is not handed over as published: the decline is counted, and the image
    // takes the whole-image path instead.
    [Fact]
    public void AnAliasRunTheTransferRefusesIsNotPublished()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var request = Rig.Request(ImageAddress, LargeSide, ImageRole.ColorTarget);
        var description = request.Description;
        description.Samples = 2;
        rig.WriteGuest(ImageAddress, MetalTileOracle.Pattern((int)LargeSize, 0x5A));
        var image = rig.Cache.GetImage(rig.Acquire(new ImageRequest(description, request.View, request.Role)));
        Assert.True(image.IsGpuModified);
        rig.Harness.Submit();
        rig.Harness.Begin();
        _ = rig.Cache.AliasCounters;
        _ = rig.Cache.TakeAliasFallbacks();

        rig.Acquire(Rig.Request(ImageAddress + 0x2000, LargeSide, AliasRows, ImageRole.ColorTarget));
        rig.Harness.Submit();
        rig.Harness.Begin();

        Assert.False(image.IsPublished(0x2000, 0x2000 + SmallSize));
        Assert.Equal(0, rig.Cache.AliasCounters.Narrow);
        var fallbacks = rig.Cache.TakeAliasFallbacks();
        output.WriteLine(fallbacks);
        Assert.Contains("write:RetileRefused=1", fallbacks);
    }

    // The large image as a target over known bytes, with guest memory then rewritten so a
    // writeback of the texture is told apart from bytes that were never published.
    private static (MetalCachedImage Image, byte[] Written, byte[] Stale) GpuWrittenLarge(Rig rig)
    {
        var written = MetalTileOracle.Pattern((int)LargeSize, 0x5A);
        rig.WriteGuest(ImageAddress, written);
        var image = rig.Cache.GetImage(rig.Acquire(ImageAddress, LargeSide, ImageRole.ColorTarget));
        Assert.True(image.IsGpuModified);
        rig.Harness.Submit();
        rig.Harness.Begin();
        // What the GPU "wrote" is what the texture holds; guest memory holds something else.
        var stale = MetalTileOracle.Pattern((int)LargeSize, 0xC3);
        rig.WriteGuest(ImageAddress, stale);
        return (image, written, stale);
    }

    // The small alias as a target at an offset into the large image, with its texture then
    // given bytes of its own, as a draw into it would. Its rows are the large image's, so
    // its bytes are a run of whole rows of it.
    private static byte[] DrawSmallAlias(Rig rig, ulong offset)
    {
        var small = rig.Cache.GetImage(
            rig.Acquire(Rig.Request(ImageAddress + offset, LargeSide, AliasRows, ImageRole.ColorTarget)));
        Assert.True(small.IsGpuModified);
        var drawn = MetalTileOracle.Pattern((int)SmallSize, 0x77);
        var description = small.Description;
        rig.Harness.Transfer.EncodeGuestToTexture(
            rig.Harness, description, new SubresourceRange(0, 1, 0, 1),
            new MetalGuestRange(rig.Harness.Buffer(drawn), 0, SmallSize, 0), default, small.Texture, 0, SmallSize);
        rig.Harness.Submit();
        rig.Harness.Begin();
        return drawn;
    }

    // A resolve or a depth copy marks its destination written before it copies one
    // subresource into it. The guest bytes the image still owed reach the texture first:
    // dropping them left the rest of the image as the texture held it before - a cube
    // whose faces were resolved one at a time kept uninitialised faces for good - and the
    // next writeback put those stale texels over the guest bytes they had replaced.
    [Fact]
    public void MarkingAnImageWrittenUploadsTheGuestBytesItStillOwes()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.WriteGuest(ImageAddress, MetalTileOracle.Pattern((int)LargeSize, 0x21));
        var large = rig.Acquire(ImageAddress, LargeSide, ImageRole.Texture);
        var image = rig.Cache.GetImage(large);
        rig.Harness.Submit();
        rig.Harness.Begin();
        var current = MetalTileOracle.Pattern((int)LargeSize, 0x7B);
        rig.WriteGuest(ImageAddress, current);
        rig.Cache.InvalidateMemory(ImageAddress, LargeSize);
        Assert.True(image.IsCpuDirty);

        rig.Cache.MarkGpuWritten(large);
        Assert.False(image.IsCpuDirty);
        Assert.True(image.IsGpuModified);
        rig.Cache.FlushGpuWrites(ImageAddress, LargeSize, MetalWritebackCause.HostRead);
        rig.Harness.Submit();
        rig.Harness.Begin();

        Assert.Equal(current, rig.ReadGuest(ImageAddress, LargeSize));
    }

    // A tiled target whose guest bytes are two spans, split at a macro tile row, is
    // written back a span at a time, and the span past the split carries its own rows:
    // the whole of guest memory reads back what the texture holds.
    [Fact]
    public void ATiledTargetSplitAcrossTwoSpansIsWrittenBackInPlace()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var request = Rig.TiledRequest(ImageAddress, 256);
        var size = request.Description.Data.Size;
        Assert.Equal(0x40000UL, size);
        var written = MetalTileOracle.Pattern((int)size, 0x6E);
        rig.WriteGuest(ImageAddress, written);
        var image = rig.Cache.GetImage(rig.Acquire(request));
        Assert.True(image.IsGpuModified);
        rig.Harness.Submit();
        rig.Harness.Begin();
        var stale = MetalTileOracle.Pattern((int)size, 0x19);
        rig.WriteGuest(ImageAddress, stale);
        // The transfer stages through a reused buffer, and the upload above left the
        // image's own linear rows in it, where a misplaced read would still find them. A
        // transfer of other bytes of the same shape overwrites them first.
        var description = request.Description;
        var decoy = rig.Harness.Texture(rig.Harness.Transfer.ShapeFor(description));
        rig.Harness.Transfer.EncodeGuestToTexture(
            rig.Harness, description, new SubresourceRange(0, 1, 0, 1),
            new MetalGuestRange(rig.Harness.Buffer(stale), 0, size, 0), default, decoy, 0, size);
        rig.Harness.Submit();
        rig.Harness.Begin();

        rig.SplitGuestAt(ImageAddress + (size / 2));
        rig.Cache.FlushGpuWrites(ImageAddress, size, MetalWritebackCause.HostRead);
        rig.Harness.Submit();
        rig.Harness.Begin();

        Assert.False(image.IsGpuModified);
        Assert.Equal(written, rig.ReadGuest(ImageAddress, size));
    }

    // Without a page watch a CPU store into the rest of the alias goes unseen, and the
    // whole refresh this marks is what used to pick it up.
    [Fact]
    public void AnUnwatchedAliasIsDirtiedWhole()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var large = rig.Acquire(ImageAddress, LargeSide, ImageRole.Texture);
        var image = rig.Cache.GetImage(large);
        Assert.True(image.IsWatched);
        rig.Cache.StopWatching();
        Assert.False(image.IsWatched);

        rig.Acquire(ImageAddress + 0x2000, SmallSide, ImageRole.ColorTarget);

        Assert.Equal(0UL, image.DirtyStart);
        Assert.Equal(LargeSize, image.DirtyEnd);
    }

    // A host write of known bytes through the backing - a packet-sized fill or copy -
    // leaves an image stale over those bytes alone and keeps its page watch, so the
    // guest's own later stores to the rest of it still fault.
    [Fact]
    public void AHostWriteDirtiesOnlyItsBytesAndKeepsTheWatch()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var image = rig.Cache.GetImage(rig.Acquire(ImageAddress, LargeSide, ImageRole.Texture));
        Assert.True(image.IsWatched);
        Assert.False(image.IsCpuDirty);

        rig.Cache.InvalidateMemory(ImageAddress + 0x2000, 0x40);
        ((IGuestImageStore)rig.Cache).MarkPreciseCpuWrite(ImageAddress + 0x2100, 0x10);

        Assert.True(image.IsCpuDirty);
        Assert.Equal(0x2000UL, image.DirtyStart);
        Assert.Equal(0x2110UL, image.DirtyEnd);
        Assert.True(image.IsWatched);
    }

    // A host write that only shares a page with a GPU-written image touches none of its
    // bytes: the image keeps what only its texture holds.
    [Fact]
    public void AHostWriteSharingOnlyAPageLeavesAGpuWrittenImageAlone()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var image = rig.Cache.GetImage(rig.Acquire(ImageAddress + 0x100, SmallSide, ImageRole.ColorTarget));
        Assert.True(image.IsGpuModified);

        rig.Cache.InvalidateMemory(ImageAddress + 0x100 + SmallSize + 0x10, 0x40);

        Assert.True(image.IsGpuModified);
        Assert.False(image.IsCpuDirty);
    }

    // A fault names a page, not the bytes stored, and a write the backing cannot take
    // goes through the guest mapping the watch protects: both still drop the watch and
    // mark the whole image stale.
    [Fact]
    public void AFaultOrAWriteOutsideTheBackingStillDirtiesTheWholeImage()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var faulted = rig.Cache.GetImage(rig.Acquire(ImageAddress, LargeSide, ImageRole.Texture));
        Assert.True(((IGuestImageStore)rig.Cache).MarkCpuWrite(ImageAddress + 0x2000, 4));
        Assert.Equal(0UL, faulted.DirtyStart);
        Assert.Equal(LargeSize, faulted.DirtyEnd);
        Assert.False(faulted.IsWatched);

        var second = ImageAddress + 0x20000;
        var unbacked = rig.Cache.GetImage(rig.Acquire(second, LargeSide, ImageRole.Texture));
        Assert.True(unbacked.IsWatched);
        rig.SplitGuestAt(second + 0x2020);
        rig.Cache.InvalidateMemory(second + 0x2000, 0x40);
        Assert.Equal(0UL, unbacked.DirtyStart);
        Assert.Equal(LargeSize, unbacked.DirtyEnd);
        Assert.False(unbacked.IsWatched);
    }

    // A guest CPU reads a linear surface in place once an end-of-pipe write says the work
    // that wrote it is done, so the next end-of-pipe write publishes it; a tiled target
    // keeps its texture authoritative until something asks for its bytes.
    [Fact]
    public void ALinearGpuWriteIsPublishedAtTheNextEndOfPipeWriteAndATiledTargetIsNot()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var written = MetalTileOracle.Pattern((int)LargeSize, 0x5A);
        rig.WriteGuest(ImageAddress, written);
        var linear = rig.Cache.GetImage(rig.Acquire(ImageAddress, LargeSide, ImageRole.ColorTarget));
        var tiled = rig.Cache.GetImage(rig.Acquire(Rig.TiledRequest(GuestBase + 0x80000, 128)));
        Assert.True(linear.IsGpuModified);
        Assert.True(tiled.IsGpuModified);
        rig.Harness.Submit();
        rig.Harness.Begin();
        // What the GPU "wrote" is what the texture holds; guest memory holds something else.
        rig.WriteGuest(ImageAddress, MetalTileOracle.Pattern((int)LargeSize, 0xC3));

        rig.Cache.PublishLinearGpuWrites();
        rig.Harness.Submit();
        rig.Harness.Begin();

        Assert.Equal(written, rig.ReadGuest(ImageAddress, LargeSize));
        Assert.False(linear.IsGpuModified);
        Assert.True(tiled.IsGpuModified);
    }

    // A linear image that is only sampled is a texture over the guest window: nothing to
    // watch, upload or dirty, and a guest store between draws needs no notice.
    [Fact]
    public void ASampledLinearImageReadsTheWindowWithNoUploadOrWatch()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value, linearViews: true);
        rig.WriteGuest(ImageAddress, MetalTileOracle.Pattern((int)LargeSize, 0x21));
        var uploadsBefore = rig.Cache.Counters.Uploads;

        var image = rig.Cache.GetImage(rig.Acquire(ImageAddress, LargeSide, ImageRole.Texture));

        Assert.True(image.IsWindowLinearView);
        Assert.False(image.IsCpuDirty);
        Assert.False(image.IsWatched);
        Assert.False(image.IsGpuModified);
        Assert.Equal(1, rig.Cache.LinearViewCount);

        // The guest rewrites it, and the cache is told as a fault would tell it.
        rig.WriteGuest(ImageAddress, MetalTileOracle.Pattern((int)LargeSize, 0x77));
        ((IGuestImageStore)rig.Cache).MarkCpuWrite(ImageAddress, 8);
        rig.Acquire(ImageAddress, LargeSide, ImageRole.Texture);

        Assert.False(image.IsCpuDirty);
        Assert.Equal(uploadsBefore, rig.Cache.Counters.Uploads);
        Assert.Equal(0, rig.Cache.TakeLinearViewPromotions());
    }

    // A target, a storage image and a copy destination write a texture, so the view gives
    // way to an owned texture that is filled from guest memory like any other.
    [Theory]
    [InlineData(ImageRole.ColorTarget)]
    [InlineData(ImageRole.StorageImage)]
    public void AWrittenLinearImageBecomesAnOwnedTexture(ImageRole role)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value, linearViews: true);
        var identifier = rig.Acquire(ImageAddress, LargeSide, ImageRole.Texture);
        var image = rig.Cache.GetImage(identifier);
        var viewTexture = image.Texture;
        Assert.True(image.IsWindowLinearView);

        var request = Rig.Request(ImageAddress, LargeSide, role);
        if (role == ImageRole.ColorTarget)
        {
            rig.Cache.AcquireColorTargetView(identifier, request);
        }
        else
        {
            rig.Cache.AcquireTextureView(identifier, request);
        }

        Assert.False(image.IsWindowLinearView);
        Assert.NotEqual(viewTexture, image.Texture);
        Assert.True(image.IsGpuModified);
        Assert.Equal(0, rig.Cache.LinearViewCount);
        Assert.Equal(1, rig.Cache.TakeLinearViewPromotions());
    }

    // A buffer-backed texture makes swizzled views like any other, so a swizzle alone
    // does not cost the image its window view.
    [Fact]
    public void ASwizzledSampleOfALinearViewStaysAWindowView()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value, linearViews: true);
        var identifier = rig.Acquire(ImageAddress, LargeSide, ImageRole.Texture);
        var image = rig.Cache.GetImage(identifier);
        var request = Rig.Request(ImageAddress, LargeSide, ImageRole.Texture);
        request.View = request.View with
        {
            Mapping = new GpuComponentMapping(
                GpuComponentSwizzle.R, GpuComponentSwizzle.R, GpuComponentSwizzle.R, GpuComponentSwizzle.R),
        };

        var view = rig.Cache.AcquireTextureView(identifier, request);

        Assert.NotEqual(0, view);
        Assert.NotEqual(image.Texture, view);
        Assert.True(image.IsWindowLinearView);
        Assert.Equal(0, rig.Cache.TakeLinearViewPromotions());

        // A view in another format is not something the window texture serves.
        request.View = request.View with { Format = GpuFormat.B8G8R8A8Unorm };
        rig.Cache.AcquireTextureView(identifier, request);

        Assert.False(image.IsWindowLinearView);
        Assert.Equal(1, rig.Cache.TakeLinearViewPromotions());
    }

    // The guest unmapping the range drops the view before the window bytes are reused.
    [Fact]
    public void UnmappingTheRangeDropsTheWindowView()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value, linearViews: true);
        rig.Acquire(ImageAddress, LargeSide, ImageRole.Texture);
        Assert.Equal(1, rig.Cache.LinearViewCount);

        ((IGuestImageStore)rig.Cache).Unregister(ImageAddress, LargeSize);

        Assert.Equal(0, rig.Cache.LinearViewCount);
        Assert.Equal(0, rig.Cache.ImageCount);
    }

    // A 2D surface over one layer of a cached array binds as a view of that layer, at the
    // array's own base address too, rather than keeping a second texture over the same
    // bytes: the two would take ownership from each other, a whole writeback and upload,
    // every time the title switched between them.
    [Theory]
    [InlineData(0u)]
    [InlineData(2u)]
    public void ALayerOfACachedArrayBindsAsAViewOfIt(uint layer)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var array = rig.Acquire(Rig.ArrayRequest(ImageAddress, LargeSide, 4, ImageRole.Texture));
        _ = rig.Cache.AliasCounters;

        var request = Rig.Request(ImageAddress + layer * LargeSize, LargeSide, ImageRole.ColorTarget);
        var identifier = rig.Cache.FindImage(ref request);
        rig.Cache.AcquireColorTargetView(identifier, request);

        Assert.Equal(array, identifier);
        Assert.Equal(layer, request.View.BaseLayer);
        Assert.Equal(1, rig.Cache.ImageCount);
        Assert.Equal(0, rig.Cache.AliasCounters.Kept);
    }

    // The array requested over a GPU-written 2D surface of its first layer replaces it: the
    // surface goes back to guest memory and the array, uploaded from there, serves it as a
    // view from then on, instead of the pair being kept as aliases of each other.
    [Fact]
    public void AnArrayOverACachedFirstLayerReplacesIt()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var written = MetalTileOracle.Pattern((int)LargeSize, 0x3B);
        rig.WriteGuest(ImageAddress, written);
        var surface = rig.Acquire(ImageAddress, LargeSide, ImageRole.ColorTarget);
        Assert.True(rig.Cache.GetImage(surface).IsGpuModified);
        rig.Harness.Submit();
        rig.Harness.Begin();
        rig.WriteGuest(ImageAddress, MetalTileOracle.Pattern((int)LargeSize, 0xA4));
        _ = rig.Cache.AliasCounters;

        var array = rig.Acquire(Rig.ArrayRequest(ImageAddress, LargeSide, 4, ImageRole.Texture));
        rig.Harness.Submit();
        rig.Harness.Begin();

        Assert.NotEqual(surface, array);
        Assert.False(rig.Cache.TryGetImage(surface, out _));
        Assert.Equal(1, rig.Cache.ImageCount);
        Assert.Equal(0, rig.Cache.AliasCounters.Kept);
        Assert.Equal(written, rig.ReadGuest(ImageAddress, LargeSize));

        var request = Rig.Request(ImageAddress, LargeSide, ImageRole.ColorTarget);
        Assert.Equal(array, rig.Cache.FindImage(ref request));
        Assert.Equal(0u, request.View.BaseLayer);
        Assert.Equal(1, rig.Cache.ImageCount);
    }

    // Two targets of one format at one address whose rows pad to the same pitch are still
    // different surfaces when their widths differ: the wider one must not bind the narrower
    // one's texture through a view, or a draw into it renders only the narrow part.
    [Fact]
    public void AWiderTargetOverTheSamePaddedPitchGetsItsOwnTexture()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var narrow = PaddedRequest(ImageAddress, 30, 34);
        var narrowImage = rig.Cache.FindImage(ref narrow);
        rig.Cache.AcquireColorTargetView(narrowImage, narrow);

        var wide = PaddedRequest(ImageAddress, 60, 34);
        var wideImage = rig.Cache.FindImage(ref wide);

        Assert.NotEqual(narrowImage, wideImage);
        Assert.Equal(60u, rig.Cache.GetImage(wideImage).Description.Extent.Width);
        Assert.Equal(60u, rig.Cache.GetImage(wideImage).Shape.Width);
        Assert.Equal(0u, wide.View.BaseLevel);
    }

    // A linear RGBA8 target of the given width with rows padded to 64 texels.
    private static ImageRequest PaddedRequest(ulong address, uint width, uint height)
    {
        const uint pitch = 64;
        var request = Rig.Request(address, width, height, ImageRole.ColorTarget);
        var description = request.Description;
        var size = (ulong)pitch * height * 4;
        description.Data = new GuestSpan(address, size);
        description.Pitch = pitch;
        description.MipLayout[0] = new MipLevelLayout { Offset = 0, Size = size, Pitch = pitch, Height = height };
        return new ImageRequest(description, request.View, ImageRole.ColorTarget);
    }

    private bool RequireDevice()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Image alias synchronization runs on the device only behind the exclusive device gate.");
            return false;
        }

        Assert.True(OperatingSystem.IsMacOS());
        Assert.NotEqual(0, Device.Value);
        return true;
    }

    private static readonly Lazy<nint> Device = new(MetalNative.MTLCreateSystemDefaultDevice);

    // A sampled binding beside an alias over its bytes is memoised: the stamp holds while neither
    // image moves, and goes once the alias is drawn into, or the texture itself is.
    [Fact]
    public void ASampledBindingBesideAnAliasIsMemoisedUntilEitherIsWritten()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var large = rig.Acquire(ImageAddress, LargeSide, ImageRole.Texture);
        rig.Acquire(ImageAddress + 0x2000, SmallSide, ImageRole.Texture);
        var request = Rig.Request(ImageAddress, LargeSide, ImageRole.Texture);

        var stamp = StampOf(rig, ImageAddress, LargeSide);
        Assert.NotNull(stamp);
        Assert.NotNull(stamp.Value.Others);
        Assert.True(rig.Cache.TextureStampMayHold(stamp.Value));
        Assert.True(rig.Cache.TryReacquireTextureView(large, request, stamp.Value));

        rig.Acquire(ImageAddress + 0x2000, SmallSide, ImageRole.ColorTarget);
        Assert.False(rig.Cache.TextureStampMayHold(stamp.Value));
        Assert.False(rig.Cache.TryReacquireTextureView(large, request, stamp.Value));

        stamp = StampOf(rig, ImageAddress, LargeSide);
        Assert.NotNull(stamp);
        Assert.True(rig.Cache.TryReacquireTextureView(large, request, stamp.Value));
        rig.Acquire(ImageAddress, LargeSide, ImageRole.ColorTarget);
        Assert.False(rig.Cache.TryReacquireTextureView(large, request, stamp.Value));
    }

    // The full acquire a memoised binding stands for, with the stamp it issues.
    private static MetalImageCache.TextureMemoStamp? StampOf(Rig rig, ulong address, uint side)
    {
        var request = Rig.Request(address, side, ImageRole.Texture);
        var pages = rig.Cache.RequestPageVersions(in request);
        var identifier = rig.Cache.FindImage(ref request);
        rig.Cache.AcquireTextureViewForMemo(identifier, request, pages, out _, out var stamp);
        return stamp;
    }
}
