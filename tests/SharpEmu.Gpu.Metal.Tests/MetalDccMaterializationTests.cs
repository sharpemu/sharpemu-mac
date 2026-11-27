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
/// A DCC fast clear is applied to the image once per guest clear, and the guest's own
/// metadata bytes are never written to remember it: transient aliasing reuses that range
/// as live data. Device work is behind the exclusive device gate.
/// </summary>
public sealed unsafe partial class MetalDccMaterializationTests(ITestOutputHelper output)
{
    private const ulong GuestBase = 0x2_0000_0000;
    private const ulong ImageAddress = GuestBase + 0x1000;
    private const ulong MetadataAddress = GuestBase + 0x10000;
    private const ulong SliceSize = 0x1000;

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

        public bool TryWriteBacking(ulong address, ReadOnlySpan<byte> data)
        {
            if (!Contains(address, (ulong)data.Length)) return false;
            data.CopyTo(new Span<byte>((byte*)_block + (address - GuestBase), data.Length));
            return true;
        }

        // Bytes read from the metadata slice, so a test can tell a scan from a peek.
        public long MetadataBytesRead;

        public bool TryReadBacking(ulong address, Span<byte> data)
        {
            if (!Contains(address, (ulong)data.Length)) return false;
            if (address < MetadataAddress + SliceSize && MetadataAddress < address + (ulong)data.Length)
            {
                MetadataBytesRead += data.Length;
            }

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
        public int Clears;
        public int Fills;
        public bool Pending;

        public nint Device => harness.Device;
        public ulong CurrentSerial => harness.SubmissionSerial;
        public ulong CompletedSerial => harness.CompletedSerial;
        public IMetalSubmissionContext BeginTransfer(MetalTransferCause cause) => harness;
        public int PendingQueries;

        public bool HasPendingGpuWrite(ulong address, ulong size)
        {
            PendingQueries++;
            return Pending;
        }

        public void FillGuestRange(ulong address, ulong size, uint value) => Fills++;
        public void NoteGuestWrite(ulong address, ulong size) { }
        public void WaitForGpu() { }

        public readonly List<SubresourceRange> ClearRanges = [];

        public void ClearTextureColor(nint texture, ulong pixelFormat, in SubresourceRange range, uint width, uint height, in GpuClearColor color)
        {
            Clears++;
            ClearRanges.Add(range);
        }

        public readonly List<(float Depth, bool ClearDepth, bool ClearStencil, SubresourceRange Range)> DepthClears = [];

        public void ClearTextureDepthStencil(nint texture, ulong pixelFormat, in SubresourceRange range, uint width, uint height,
            float depth, byte stencil, bool clearDepth, bool clearStencil) =>
            DepthClears.Add((depth, clearDepth, clearStencil, range));
    }

    private sealed class Rig : IDisposable
    {
        private readonly AddressSpace _space = new();
        private readonly PageGuard _pages;
        private readonly MetalGuestMemory _memory;

        public Rig(nint device)
        {
            Harness = new MetalTransferHarness(device);
            Harness.Begin();
            Backing = new Backing();
            Owner = new Owner(Harness);
            _pages = new PageGuard(_space);
            _memory = new MetalGuestMemory(device, Backing, new NoMemory());
            Cache = new MetalImageCache(Owner, Harness.Transfer, _memory, _pages, Backing)
            {
                // The depth alias cases drive the hand-off between two textures; a sampled
                // twin binding the depth texture itself is MetalDepthTwinTests'.
                DepthTwins = false,
            };
            // Exact textures: lossy compression is not this suite's subject.
            Cache.LossyTextures = false;
            // Off by default (MetalV2Diagnostics.DccClaimFirst); the suite covers it on.
            Cache.DccClaimFirst = true;
            // Not a clear code, so the target lookups in a test do not materialize.
            FillMetadata(0x55);
        }

        public MetalTransferHarness Harness { get; }
        public Backing Backing { get; }
        public Owner Owner { get; }
        public MetalImageCache Cache { get; }

        public void FillMetadata(byte value) => Backing.TryWriteBacking(MetadataAddress, Enumerable.Repeat(value, (int)SliceSize).ToArray());

        public byte[] ReadMetadata()
        {
            var bytes = new byte[SliceSize];
            Backing.TryReadBacking(MetadataAddress, bytes);
            return bytes;
        }

        public ImageRequest Request(ImageRole role, ulong address = ImageAddress)
        {
            var description = ImageDescription.Create();
            description.Data = new GuestSpan(address, 4 * 4 * 4);
            description.PixelFormat = GpuFormat.R8G8B8A8Unorm;
            description.GuestFormat = GuestPixelFormat.Bits8_8_8_8UNorm;
            description.Type = GuestImageType.Color2D;
            description.Extent = new GpuExtent3D(4, 4, 1);
            description.Resources = new SubresourceCount(1, 1);
            description.Pitch = 4;
            description.BytesPerBlock = 4;
            description.Samples = 1;
            description.TileMode = GuestTileMode.Linear;
            description.MipLayout[0] = new MipLevelLayout { Offset = 0, Size = 64, Pitch = 4, Height = 4 };
            // An image placed anywhere else is a plain surface over those bytes.
            if (address == ImageAddress)
            {
                description.Metadata.Kind = MetadataKind.Dcc;
                description.Metadata.Range = new GuestSpan(MetadataAddress, SliceSize);
                description.Metadata.DccAlphaMsb = true;
            }
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

        public ResourceSlotIdentifier Lookup(ImageRole role = ImageRole.Texture)
        {
            var request = Request(role);
            return Cache.FindImage(ref request);
        }

        public void Dispose()
        {
            Cache.StopWatching();
            Cache.Dispose();
            _pages.Dispose();
            Harness.Dispose();
            Backing.Dispose();
        }
    }

    [Fact]
    public void MaterializeDcc_NeverWritesGuestMemory()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Lookup(ImageRole.ColorTarget);
        rig.FillMetadata(0x00);

        rig.Lookup();

        Assert.Equal(1, rig.Owner.Clears);
        Assert.All(rig.ReadMetadata(), static value => Assert.Equal(0x00, value));
        Assert.Equal(0, rig.Owner.Fills);
    }

    [Fact]
    public void MaterializeDcc_RepeatedLookupClearsOnce()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Lookup(ImageRole.ColorTarget);
        rig.FillMetadata(0x00);

        rig.Lookup();
        rig.Lookup();

        Assert.Equal(1, rig.Owner.Clears);
        Assert.Equal(1, rig.Cache.TakeDccRepeatsSkipped());
    }

    // A slice that already holds the clear it was claimed for is settled from its first
    // byte; the claim, not a scan of the slice, says it is a repeat. This is the full path,
    // with SHARPEMU_METAL_V2_DCC_CLAIM_FIRST=0; the claim-first one reads nothing at all.
    [Fact]
    public void MaterializeDcc_ClaimedSliceIsNotScanned()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Cache.DccClaimFirst = false;
        rig.Lookup(ImageRole.ColorTarget);
        rig.FillMetadata(0x00);
        rig.Lookup();
        Assert.Equal(1, rig.Owner.Clears);
        Assert.True(rig.Cache.MetadataWatchOverlaps(MetadataAddress, SliceSize));

        rig.Backing.MetadataBytesRead = 0;
        rig.Lookup();
        rig.Lookup(ImageRole.ColorTarget);

        Assert.Equal(1, rig.Owner.Clears);
        Assert.Equal(2, rig.Cache.TakeDccRepeatsSkipped());
        Assert.Equal(2, rig.Backing.MetadataBytesRead);
    }

    public enum Rewrite
    {
        DmaFill,
        GpuWrite,
        ClearMetadata,
        CpuWrite,
    }

    [Theory]
    [InlineData(Rewrite.DmaFill)]
    [InlineData(Rewrite.GpuWrite)]
    [InlineData(Rewrite.ClearMetadata)]
    [InlineData(Rewrite.CpuWrite)]
    public void MaterializeDcc_GuestRefillRearms(Rewrite rewrite)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Lookup(ImageRole.ColorTarget);
        rig.FillMetadata(0x00);
        rig.Lookup();
        Assert.Equal(1, rig.Owner.Clears);

        switch (rewrite)
        {
            case Rewrite.DmaFill:
                rig.Cache.TryAbsorbDccFill(MetadataAddress, SliceSize, 0);
                break;
            case Rewrite.GpuWrite:
                rig.Cache.InvalidateMemoryFromGpu(MetadataAddress, SliceSize);
                break;
            case Rewrite.ClearMetadata:
                rig.Cache.ClearMetadata(MetadataAddress);
                break;
            case Rewrite.CpuWrite:
                Assert.True(((IGuestImageStore)rig.Cache).MarkCpuWrite(MetadataAddress + 0x40, 4));
                break;
        }

        rig.Lookup();
        Assert.Equal(2, rig.Owner.Clears);
        rig.Lookup();
        Assert.Equal(2, rig.Owner.Clears);
    }

    // The lock-free test a CPU store makes sees the claimed slice's watch to its exact
    // edges, and the store that re-arms the claim leaves no watch behind it.
    [Fact]
    public void MaterializeDcc_ClaimWatchIsSeenWithoutTheLock()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        Assert.False(rig.Cache.MetadataWatchOverlaps(MetadataAddress, SliceSize));
        rig.Lookup(ImageRole.ColorTarget);
        rig.FillMetadata(0x00);
        rig.Lookup();

        Assert.True(rig.Cache.MetadataWatchOverlaps(MetadataAddress, 1));
        Assert.True(rig.Cache.MetadataWatchOverlaps(MetadataAddress + SliceSize - 1, 1));
        Assert.False(rig.Cache.MetadataWatchOverlaps(MetadataAddress - 8, 8));
        Assert.False(rig.Cache.MetadataWatchOverlaps(MetadataAddress + SliceSize, 8));

        Assert.True(((IGuestImageStore)rig.Cache).MarkPreciseCpuWrite(MetadataAddress + 0x40, 8));
        Assert.False(rig.Cache.MetadataWatchOverlaps(MetadataAddress, SliceSize));
    }

    // A CPU store waits on the cache lock for an image only while one lives on its 64 KiB
    // page: not for a neighbour further along the same megabyte, and not once it is gone.
    [Fact]
    public void CpuWriteLocksOnlyForALiveImageNearIt()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var store = (IGuestImageStore)rig.Cache;
        rig.Lookup();
        _ = rig.Cache.TakeLockedCpuWrites();

        Assert.True(store.MarkPreciseCpuWrite(ImageAddress, 8));
        var live = rig.Cache.TakeLockedCpuWrites();
        Assert.Equal((1L, 1L, 0L), (live.Owner, live.OwnerFound, live.OwnerNone));

        Assert.False(store.MarkPreciseCpuWrite(GuestBase + 0x20000, 8));
        Assert.Equal(0L, rig.Cache.TakeLockedCpuWrites().Owner);

        store.Unregister(ImageAddress, 0x1000);
        Assert.False(store.MarkPreciseCpuWrite(ImageAddress, 8));
        Assert.Equal(0L, rig.Cache.TakeLockedCpuWrites().Owner);
    }

    [Fact]
    public void MaterializeDcc_SkipsWhilePendingGpuWrite()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Lookup(ImageRole.ColorTarget);
        rig.FillMetadata(0x00);
        rig.Owner.Pending = true;

        rig.Lookup();
        rig.Lookup();
        Assert.Equal(0, rig.Owner.Clears);

        rig.Owner.Pending = false;
        rig.Lookup();
        rig.Lookup();
        Assert.Equal(1, rig.Owner.Clears);
    }

    // The Oregon Trail: the target is bound and drawn while the guest's metadata clear is
    // still being written, and the composite that samples it would apply the clear late.
    [Fact]
    public void MaterializeDcc_DeferredClearIsDroppedOnceDrawnTo()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Lookup(ImageRole.ColorTarget);
        rig.FillMetadata(0x00);
        rig.Cache.InvalidateMemoryFromGpu(MetadataAddress, SliceSize);
        rig.Owner.Pending = true;

        var target = rig.Lookup(ImageRole.ColorTarget);
        rig.Cache.MarkGpuWritten(target);
        rig.Owner.Pending = false;
        rig.Lookup();
        rig.Lookup(ImageRole.ColorTarget);

        Assert.Equal(0, rig.Owner.Clears);
        Assert.Equal(1, rig.Cache.TakeDccClearsDropped());
    }

    [Fact]
    public void MaterializeDcc_ReArmedClearAppliesToTextureLookup()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Lookup(ImageRole.ColorTarget);
        rig.FillMetadata(0x00);
        rig.Cache.InvalidateMemoryFromGpu(MetadataAddress, SliceSize);

        rig.Lookup();

        Assert.Equal(1, rig.Owner.Clears);
        Assert.Equal(0, rig.Cache.TakeDccClearsDropped());
    }

    [Fact]
    public void MaterializeDcc_TargetClearIsNotRepeatedOverDraws()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Lookup(ImageRole.ColorTarget);
        rig.FillMetadata(0x00);
        rig.Cache.InvalidateMemoryFromGpu(MetadataAddress, SliceSize);

        var target = rig.Lookup(ImageRole.ColorTarget);
        Assert.Equal(1, rig.Owner.Clears);
        rig.Cache.MarkGpuWritten(target);
        rig.Lookup();

        Assert.Equal(1, rig.Owner.Clears);
        Assert.Equal(0, rig.Cache.TakeDccClearsDropped());
    }

    // Recorded GPU stores re-arm a claim only when they cover its slice's first byte; the
    // page index that lets the others skip the walk over every claim keeps both answers.
    [Theory]
    [InlineData(MetadataAddress + 0x40, 8ul, false)]
    [InlineData(MetadataAddress + SliceSize, 8ul, false)]
    [InlineData(GuestBase + 0x30000, 64ul, false)]
    [InlineData(MetadataAddress, 8ul, true)]
    [InlineData(MetadataAddress - 8, 16ul, true)]
    [InlineData(GuestBase, 0x20000ul, true)]
    public void MaterializeDcc_GpuStoreReArmsOnlyOverTheSliceStart(ulong address, ulong size, bool rearms)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Lookup(ImageRole.ColorTarget);
        rig.FillMetadata(0x00);
        rig.Lookup();
        Assert.Equal(1, rig.Owner.Clears);

        rig.Cache.InvalidateMemoryFromGpu(address, size);
        rig.Lookup();

        Assert.Equal(rearms ? 2 : 1, rig.Owner.Clears);
    }

    // A deferral is per guest clear: a new clear of the metadata applies again.
    [Fact]
    public void MaterializeDcc_DroppedClearReArms()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Lookup(ImageRole.ColorTarget);
        rig.FillMetadata(0x00);
        rig.Owner.Pending = true;
        var target = rig.Lookup(ImageRole.ColorTarget);
        rig.Cache.MarkGpuWritten(target);
        rig.Owner.Pending = false;
        rig.Lookup();
        Assert.Equal(0, rig.Owner.Clears);

        rig.Cache.InvalidateMemoryFromGpu(MetadataAddress, SliceSize);
        rig.Lookup(ImageRole.ColorTarget);

        Assert.Equal(1, rig.Owner.Clears);
    }

    // OCTOPATH TRAVELER II: the separate-translucency target's metadata is filled with a
    // clear code by a dispatch recorded just before the target is bound, so the bytes are
    // still in flight at that lookup. The recorded fill is what the metadata holds by the
    // time the clear runs, and the particles drawn after it must not survive into the
    // next frame.
    [Fact]
    public void MaterializeDcc_RecordedFillClearsWhileItsBytesAreInFlight()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Lookup(ImageRole.ColorTarget);

        for (var frame = 1; frame <= 2; frame++)
        {
            Assert.False(rig.Cache.TryAbsorbDccFill(MetadataAddress, SliceSize, 0));
            rig.Cache.InvalidateMemoryFromGpu(MetadataAddress, SliceSize);
            rig.Owner.Pending = true;

            var target = rig.Lookup(ImageRole.ColorTarget);
            Assert.Equal(frame, rig.Owner.Clears);
            rig.Cache.MarkGpuWritten(target);
            rig.Lookup();
            Assert.Equal(frame, rig.Owner.Clears);
        }

        Assert.Equal(0, rig.Cache.TakeDccClearsDropped());
        Assert.Equal(0, rig.Owner.Fills);
    }

    // A store over the metadata after the fill replaces what the fill wrote.
    [Fact]
    public void MaterializeDcc_RecordedFillOverwrittenStillDefers()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Lookup(ImageRole.ColorTarget);
        rig.Cache.TryAbsorbDccFill(MetadataAddress, SliceSize, 0);
        rig.Cache.InvalidateMemoryFromGpu(MetadataAddress, SliceSize);
        rig.Cache.InvalidateMemoryFromGpu(MetadataAddress, SliceSize);
        rig.Owner.Pending = true;

        rig.Lookup(ImageRole.ColorTarget);

        Assert.Equal(0, rig.Owner.Clears);
    }

    [Fact]
    public void MaterializeDcc_RecordedFillWithoutClearCodeDefers()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Lookup(ImageRole.ColorTarget);
        rig.Cache.TryAbsorbDccFill(MetadataAddress, SliceSize, 0x55555555);
        rig.Cache.InvalidateMemoryFromGpu(MetadataAddress, SliceSize);
        rig.Owner.Pending = true;

        rig.Lookup(ImageRole.ColorTarget);

        Assert.Equal(0, rig.Owner.Clears);
    }

    // Once the fill's bytes have landed they are what count, not the record.
    [Fact]
    public void MaterializeDcc_SettledMetadataOutranksRecordedFill()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Lookup(ImageRole.ColorTarget);
        rig.Cache.TryAbsorbDccFill(MetadataAddress, SliceSize, 0);
        rig.Cache.InvalidateMemoryFromGpu(MetadataAddress, SliceSize);

        rig.Lookup(ImageRole.ColorTarget);
        Assert.Equal(0, rig.Owner.Clears);

        rig.Owner.Pending = true;
        rig.Lookup(ImageRole.ColorTarget);
        Assert.Equal(0, rig.Owner.Clears);
    }

    private bool RequireDevice()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("DCC materialization runs on the device only behind the exclusive device gate.");
            return false;
        }

        Assert.True(OperatingSystem.IsMacOS());
        Assert.NotEqual(0, Device.Value);
        return true;
    }

    // Demon's Souls clears its UI layer's DCC metadata every frame with a bounded fill.
    // The image took that metadata with its size, which takes it out of the metadata the
    // cache registers, and the fill then ran as a dispatch: every lookup met the bytes in
    // flight and the clear was deferred until the layer was drawn, then dropped. The fill
    // runs on the host instead, with its code recorded for the lookups that follow it.
    [Fact]
    public void MaterializeDcc_HostFillOfAdoptedMetadataClearsWhileInFlight()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Lookup(ImageRole.ColorTarget);
        Assert.True(rig.Cache.OverlapsDccMetadata(MetadataAddress, SliceSize));

        for (var frame = 1; frame <= 2; frame++)
        {
            rig.Cache.RecordDccFill(MetadataAddress, SliceSize, 0);
            // The fill's own blit, still on the GPU.
            rig.Cache.InvalidateMemoryFromGpu(MetadataAddress, SliceSize);
            rig.Owner.Pending = true;

            var target = rig.Lookup(ImageRole.ColorTarget);
            Assert.Equal(frame, rig.Owner.Clears);
            rig.Cache.MarkGpuWritten(target);
            rig.Lookup(ImageRole.ColorTarget);
            rig.Lookup();
            Assert.Equal(frame, rig.Owner.Clears);
        }

        Assert.Equal(0, rig.Cache.TakeDccClearsDropped());
    }

    private static readonly Lazy<nint> Device = new(MetalNative.MTLCreateSystemDefaultDevice);
}
