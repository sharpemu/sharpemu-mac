// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Rendering;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

// The device-free parts of the v2 host: the slot table the shared engine holds
// identifiers into, the page owner index a CPU write fault consults, and the
// translations between the neutral vocabulary and Metal.
public sealed class MetalV2HostTests
{
    // Every complete group of three is one rectangle (one patch); a trailing incomplete
    // group draws nothing.
    [Theory]
    [InlineData(0u, 0u)]
    [InlineData(1u, 0u)]
    [InlineData(2u, 0u)]
    [InlineData(3u, 1u)]
    [InlineData(4u, 1u)]
    [InlineData(5u, 1u)]
    [InlineData(6u, 2u)]
    [InlineData(300u, 100u)]
    public void ARectangleListDrawsOnePatchPerCompleteRectangle(uint count, uint patches)
    {
        Assert.Equal(patches, MetalV2Formats.RectanglePatchCount(count));
    }

    [Fact]
    public void ARectangleListRasterizesTriangles()
    {
        Assert.Equal(Mtl.PrimitiveTriangle, MetalV2Formats.Topology(GpuPrimitiveTopology.PatchList));
        Assert.Equal(3u, MetalV2Formats.TopologyClass(GpuPrimitiveTopology.PatchList));
    }

    [Fact]
    public void ErasedImageSlotStopsResolvingAndItsIndexIsReused()
    {
        var slots = new MetalImageSlots();
        var first = slots.Insert(CreateImage());
        var second = slots.Insert(CreateImage());
        Assert.NotNull(slots.TryGet(first));
        Assert.NotNull(slots.TryGet(second));
        Assert.Equal(2, slots.Count);

        slots.Erase(first);
        Assert.Null(slots.TryGet(first));
        Assert.NotNull(slots.TryGet(second));

        var reused = slots.Insert(CreateImage());
        Assert.Equal(first.Index, reused.Index);
        Assert.NotEqual(first.Generation, reused.Generation);
        // The stale identifier must not resolve to the new occupant of its slot.
        Assert.Null(slots.TryGet(first));
        Assert.NotNull(slots.TryGet(reused));
    }

    [Fact]
    public void DefaultImageIdentifierNeverResolves()
    {
        var slots = new MetalImageSlots();
        _ = slots.Insert(CreateImage());
        Assert.Null(slots.TryGet(ResourceSlotIdentifier.Invalid));
    }

    // The owner-list change count over a range moves whenever an image comes to or leaves one of
    // its pages and never otherwise: memos of lookups over the range compare it.
    [Fact]
    public void PageOwnerVersionsMoveOnlyWhenTheirListsChange()
    {
        var owners = new MetalImagePageOwners();
        const ulong page = 1UL << MetalImagePageOwners.PageBits;
        var identifier = new ResourceSlotIdentifier(7, 3);
        Assert.Equal(0, owners.VersionSum(page, page));

        owners.Add(page + 0x1000, 0x1000, identifier);
        var afterAdd = owners.VersionSum(page, page);
        Assert.NotEqual(0, afterAdd);
        Assert.Equal(afterAdd, owners.VersionSum(page, page));
        // Another page is untouched; a range over both sees the change once.
        Assert.Equal(0, owners.VersionSum(2 * page, page));
        Assert.Equal(afterAdd, owners.VersionSum(page, 2 * page));

        owners.Remove(page + 0x1000, 0x1000, identifier);
        Assert.NotEqual(afterAdd, owners.VersionSum(page, page));
    }

    // The owner count answers for the 64 KiB pages a live image covers and no others,
    // and forgets an image once the last one over a page is gone.
    [Fact]
    public void PageOwnersAnswerOnlyForPagesALiveImageCovers()
    {
        var owners = new MetalImagePageOwners();
        const ulong page = 1UL << MetalImagePageOwners.PageBits;
        const ulong countPage = 1UL << MetalImagePageOwners.CountPageBits;
        Assert.False(owners.MayHaveOwners(page, 0x1000));

        var identifier = new ResourceSlotIdentifier(7, 3);
        var alias = new ResourceSlotIdentifier(8, 1);
        owners.Add(page + 0x1000, 0x1000, identifier);
        owners.Add(page, 0x4000, alias);
        Assert.True(owners.MayHaveOwners(page + 0x1800, 8));
        Assert.False(owners.MayHaveOwners(page + countPage, 0x1000));
        Assert.False(owners.MayHaveOwners(page + (2 * page), 0x1000));
        // A range that starts in a leaf nothing was ever counted in still finds the owner.
        Assert.True(owners.MayHaveOwners(0, page + 1));

        // Both images are smaller than an owner page, so they are listed per fine page (or per owner page when fine
        // pages are off).
        List<ResourceSlotIdentifier>? Owners(ulong address) => MetalImagePageOwners.IsFine(0x1000)
            ? owners.FindFine(address >> MetalImagePageOwners.FinePageBits)
            : owners.Find(address >> MetalImagePageOwners.PageBits);
        var list = Owners(page + 0x1000);
        Assert.NotNull(list);
        Assert.Contains(identifier, list!);

        owners.Remove(page + 0x1000, 0x1000, identifier);
        Assert.True(owners.MayHaveOwners(page + 0x1800, 8));
        owners.Remove(page, 0x4000, alias);
        Assert.False(owners.MayHaveOwners(page, page));
        Assert.Empty(Owners(page + 0x1000)!);
        Assert.Empty(Owners(page)!);
    }

    // The order the image cache relies on when a CPU write reads the count without the
    // lock: the image becomes reachable only after it is counted and stops being reachable
    // before it is uncounted. A reader that saw it reachable on both sides of its count
    // read must have seen a live owner, however the add and remove race it.
    [Fact]
    public void ALockFreeReaderNeverMissesAReachableOwner()
    {
        var owners = new MetalImagePageOwners();
        var gate = new object();
        // Straddles two count pages, so the read of the last one races the first.
        const ulong address = 0x2_0000_0000 + 0xF000;
        const ulong size = 0x2000;
        var identifier = new ResourceSlotIdentifier(1, 1);
        // Odd while the image is reachable: the stand-in for its slot and page watch.
        long phase = 0;
        var stop = 0;
        long violations = 0;
        long checkedWhileReachable = 0;

        var readers = Enumerable.Range(0, 2).Select(_ => new Thread(() =>
        {
            while (Volatile.Read(ref stop) == 0)
            {
                var before = Volatile.Read(ref phase);
                var owned = owners.MayHaveOwners(address + size - 8, 8);
                var after = Volatile.Read(ref phase);
                if (before == after && (before & 1) == 1)
                {
                    Interlocked.Increment(ref checkedWhileReachable);
                    if (!owned)
                    {
                        Interlocked.Increment(ref violations);
                    }
                }
            }
        })).ToArray();
        foreach (var reader in readers)
        {
            reader.Start();
        }

        // At least 50 000 cycles, and on until the readers have checked while the image was
        // reachable: on a loaded machine the 50 000 can finish before a reader is scheduled.
        var deadline = Environment.TickCount64 + 10_000;
        for (var cycle = 0;
             cycle < 50_000 || (Interlocked.Read(ref checkedWhileReachable) < 1_000 && Environment.TickCount64 < deadline);
             cycle++)
        {
            lock (gate)
            {
                owners.Add(address, size, identifier);
                Volatile.Write(ref phase, phase + 1);
            }

            lock (gate)
            {
                Volatile.Write(ref phase, phase + 1);
                owners.Remove(address, size, identifier);
            }
        }

        Volatile.Write(ref stop, 1);
        foreach (var reader in readers)
        {
            reader.Join();
        }

        Assert.Equal(0, violations);
        Assert.True(checkedWhileReachable > 0);
        Assert.False(owners.MayHaveOwners(address, size));
    }

    [Theory]
    [InlineData(0UL, 0UL, false)]
    [InlineData(0x1000UL, 0x1000UL, true)]
    [InlineData(1UL << 40, 0x1000UL, false)]
    public void PageRangesOutsideTheAddressSpaceAreRejected(ulong address, ulong size, bool expected)
    {
        Assert.Equal(expected, MetalImagePageOwners.TryGetPageRange(address, size, out var first, out var last));
        if (expected)
        {
            Assert.True(last > first);
        }
    }

    [Fact]
    public void AnUnmappedImageFormatSubstitutesTheSizeEquivalentOne()
    {
        // An unknown format must never stop a title: the substitute keeps the guest
        // footprint, so the tiler's element size still matches.
        Assert.Equal(GpuFormat.R8Unorm, MetalV2Formats.SubstituteFor(1));
        Assert.Equal(GpuFormat.R16Unorm, MetalV2Formats.SubstituteFor(2));
        Assert.Equal(GpuFormat.R8G8B8A8Unorm, MetalV2Formats.SubstituteFor(4));
        Assert.Equal(GpuFormat.R16G16B16A16Unorm, MetalV2Formats.SubstituteFor(8));
        Assert.Equal(GpuFormat.R32G32B32A32Sfloat, MetalV2Formats.SubstituteFor(16));
        Assert.Equal(GpuFormat.R8G8B8A8Unorm, MetalV2Formats.SubstituteFor(3));
    }

    [Fact]
    public void UnmappedFormatsFallBackToRgba8RatherThanFailing()
    {
        Assert.False(MetalV2Formats.IsSupported(GpuFormat.R64G64B64A64Sfloat));
        Assert.Equal(MtlFormat.Rgba8Unorm, MetalV2Formats.PixelFormat(GpuFormat.R64G64B64A64Sfloat));
        Assert.True(MetalV2Formats.IsSupported(GpuFormat.R8G8B8A8Unorm));
    }

    [Fact]
    public void ColorWriteMaskIsReversedForMetal()
    {
        // Guest CB bits are R=1,G=2,B=4,A=8; MTLColorWriteMask orders them alpha-first.
        Assert.Equal(0, MetalV2Formats.ColorWriteMask(0));
        Assert.Equal(Mtl.ColorWriteMaskRed, MetalV2Formats.ColorWriteMask(1));
        Assert.Equal(Mtl.ColorWriteMaskAlpha, MetalV2Formats.ColorWriteMask(8));
        Assert.Equal(
            Mtl.ColorWriteMaskRed | Mtl.ColorWriteMaskGreen | Mtl.ColorWriteMaskBlue | Mtl.ColorWriteMaskAlpha,
            MetalV2Formats.ColorWriteMask(0xF));
    }

    [Fact]
    public void BlendCodesMapToMetalsReorderedFactors()
    {
        // The guest orders destination alpha before destination colour; Metal does not.
        Assert.Equal(8, MetalV2Formats.BlendFactor(6));
        Assert.Equal(6, MetalV2Formats.BlendFactor(8));
        Assert.Equal(1, MetalV2Formats.BlendFactor(0xFF));
        Assert.Equal(Mtl.BlendOperationMin, MetalV2Formats.BlendOperation(2));
        Assert.Equal(Mtl.BlendOperationReverseSubtract, MetalV2Formats.BlendOperation(4));
    }

    [Theory]
    [InlineData(GpuPrimitiveTopology.PointList, 1u)]
    [InlineData(GpuPrimitiveTopology.LineList, 2u)]
    [InlineData(GpuPrimitiveTopology.LineStrip, 2u)]
    [InlineData(GpuPrimitiveTopology.TriangleList, 3u)]
    [InlineData(GpuPrimitiveTopology.TriangleStrip, 3u)]
    public void LayeredPipelinesDeclareTheClassOfTheDrawnPrimitive(GpuPrimitiveTopology topology, uint expected)
    {
        // A layered pipeline's input topology class has to match every draw's primitive type.
        Assert.Equal((nuint)expected, MetalV2Formats.TopologyClass(topology));
    }

    [Fact]
    public void ATriangleFanPipelineIsRefused()
    {
        // The executor draws fans as triangle lists for this host; a fan pipeline is a bug.
        Assert.Throws<InvalidOperationException>(() => MetalV2Formats.Topology(GpuPrimitiveTopology.TriangleFan));
    }

    [Fact]
    public void VertexFormatsNarrowToTheComponentsTheProgramFetches()
    {
        // Buffer format 77 is a float4; a program that fetches two components must
        // get float2 so Metal's fetch matches the shader's declaration.
        var float4 = new BufferDescriptorWords(0, 0, 0, 77u << 12);
        Assert.Equal(MtlVertexFormat.Float4, MetalV2Formats.VertexFormat(float4, 4));
        Assert.Equal(MtlVertexFormat.Float2, MetalV2Formats.VertexFormat(float4, 2));
        Assert.Equal(MtlVertexFormat.Float, MetalV2Formats.VertexFormat(float4, 1));

        // An unknown format falls back to the float vector of the fetched width.
        var unknown = new BufferDescriptorWords(0, 0, 0, 126u << 12);
        Assert.Equal(MtlVertexFormat.Float3, MetalV2Formats.VertexFormat(unknown, 3));
    }

    // The v2 host binds a storage buffer's Footprint. A swizzled V# (RDNA2 ISA 8.1.5) puts
    // index_stride records side by side one dword at a time, so its reach is the furthest
    // byte an in-range access (index < records, offset < stride) can touch, found here by
    // brute force over the ISA formula, not stride * records.
    [Theory]
    [InlineData(32u, 64u, 1u)]
    [InlineData(32u, 5u, 0u)]
    [InlineData(48u, 37u, 2u)]
    [InlineData(16u, 65u, 3u)]
    [InlineData(0x3F90u, 9u, 0u)]
    [InlineData(24u, 1u, 1u)]
    public void ASwizzledStorageBufferBindsItsSwizzledFootprint(uint stride, uint records, uint indexStrideCode)
    {
        var descriptor = new BufferDescriptorWords(0x1000, (stride << 16) | (1u << 31), records, indexStrideCode << 21);
        var indexStride = 8ul << (int)indexStrideCode;
        ulong reach = 0;
        for (ulong index = 0; index < records; index++)
        {
            for (ulong offset = 0; offset < stride; offset++)
            {
                var placed = (((index / indexStride) * stride) + (offset & ~3ul)) * indexStride + ((index % indexStride) * 4) + (offset & 3);
                reach = Math.Max(reach, placed + 1);
            }
        }

        Assert.Equal(reach, descriptor.Footprint());

        // Without SWIZZLE_ENABLE, or with a zero stride, the records stay linear.
        Assert.Equal((ulong)stride * records, (descriptor with { Word1 = stride << 16 }).Footprint());
        Assert.Equal((ulong)records, (descriptor with { Word1 = 1u << 31 }).Footprint());
    }

    [Fact]
    public void TextureTypeFollowsTheGuestImageShape()
    {
        Assert.Equal(Mtl.TextureType2D, MetalV2Formats.TextureType(GpuImageType.Type2D, 1, 1));
        Assert.Equal(Mtl.TextureType2DArray, MetalV2Formats.TextureType(GpuImageType.Type2D, 6, 1));
        Assert.Equal(Mtl.TextureType3D, MetalV2Formats.TextureType(GpuImageType.Type3D, 1, 1));
        Assert.Equal(Mtl.TextureType1D, MetalV2Formats.TextureType(GpuImageType.Type1D, 1, 1));
        Assert.Equal(Mtl.TextureType2DMultisample, MetalV2Formats.TextureType(GpuImageType.Type2D, 1, 4));
        // A guest cube binds as a 2D array: the shader resolves the face itself.
        Assert.Equal(
            Mtl.TextureType2DArray,
            MetalV2Formats.ViewTextureType(GpuImageViewType.TypeCube, 1));
    }

    [Fact]
    public void DepthFormatsAreRecognisedWithAndWithoutStencil()
    {
        Assert.True(MetalV2Formats.IsDepthFormat(MetalV2Formats.PixelFormat(GpuFormat.D32Sfloat)));
        Assert.False(MetalV2Formats.HasStencil(MetalV2Formats.PixelFormat(GpuFormat.D32Sfloat)));
        Assert.True(MetalV2Formats.HasStencil(MetalV2Formats.PixelFormat(GpuFormat.D32SfloatS8Uint)));
        Assert.False(MetalV2Formats.IsDepthFormat(MetalV2Formats.PixelFormat(GpuFormat.R8G8B8A8Unorm)));
    }

    [Fact]
    public void CpuWriteInsideAnImageDirtiesAllOfIt()
    {
        var image = CreateImage(0x10000, 0x10000);
        image.RefreshComplete();

        image.InvalidateCpuWrite(0x10100, 8);

        Assert.True(image.IsCpuDirty);
        Assert.Equal(0UL, image.DirtyStart);
        Assert.Equal(0x10000UL, image.DirtyEnd);
    }

    [Fact]
    public void CpuWriteClearsGpuOwnership()
    {
        var image = CreateImage(0x10000, 0x10000);
        image.MarkGpuModified();

        image.InvalidateCpuWrite(0x10100, 8);

        Assert.False(image.IsGpuModified);
        Assert.True(image.IsCpuDirty);
        Assert.Equal(0UL, image.DirtyStart);
        Assert.Equal(0x10000UL, image.DirtyEnd);
    }

    // Two published runs with a gap between them never claim the gap: a run that starts
    // past the image's first byte made disjoint runs possible, and a widened interval
    // would skip publishing bytes guest memory never got.
    [Fact]
    public void DisjointPublishedRunsNeverClaimTheGapBetweenThem()
    {
        var image = CreateImage(0x10000, 0x10000);
        image.MarkGpuModified();

        image.MarkPublished(0x0000, 0x2000);
        image.MarkPublished(0x8000, 0xC000);

        Assert.False(image.IsPublished(0x2000, 0x8000));
        Assert.False(image.IsPublished(0x0000, 0xC000));
        Assert.True(image.IsPublished(0x8000, 0xC000));

        image.MarkPublished(0x6000, 0x8000);
        Assert.True(image.IsPublished(0x6000, 0xC000));
    }

    // A cached image with no texture: enough for the slot table, which never looks
    // at the native handle.
    private static MetalCachedImage CreateImage(ulong address = 0, ulong size = 0)
    {
        var description = SharpEmu.Libs.Gpu.Images.ImageDescription.Create();
        description.PixelFormat = GpuFormat.R8G8B8A8Unorm;
        description.BytesPerBlock = 4;
        description.Pitch = 1;
        description.Data = new SharpEmu.HLE.GpuMemory.GuestSpan(address, size);
        var shape = new MetalTextureShape(
            MtlFormat.Rgba8Unorm, Mtl.TextureType2D, 1, 1, 1, 1, 1, 1, Mtl.TextureUsageShaderRead);
        return new MetalCachedImage(in description, in shape, 0);
    }
}
