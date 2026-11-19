// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.Libs.Gpu.Images;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// The tile backing of one sparse texture (MetalImageCache.Sparse.cs): which of its layers
/// are mapped and how many its heap holds. Valid while <see cref="Texture"/> is the image's
/// texture; a texture replaced for any reason leaves it behind.
/// </summary>
internal sealed class MetalSparseResidency(nint texture, nint heap, ulong heapBytes, uint capacity, uint layers,
    nuint tilesWide, nuint tilesHigh)
{
    public nint Texture { get; } = texture;

    // Owned by the texture through an associated object, so it lives exactly as long.
    public nint Heap { get; } = heap;

    public ulong HeapBytes { get; } = heapBytes;

    // Layers the heap has tiles for.
    public uint Capacity { get; } = capacity;

    public uint Layers { get; } = layers;

    public nuint TilesWide { get; } = tilesWide;

    public nuint TilesHigh { get; } = tilesHigh;

    public bool[] Mapped { get; } = new bool[layers];

    public uint MappedCount { get; set; }
}

// Large array textures with tile backing only where the guest put something.
//
// A texture of SparseMinimumBytes or more that is a 2D array of one level (a shadow-map
// pool, a layered G-buffer) is created on a sparse heap, and a layer gets tiles only when
// something writes it: a render pass or a clear binds it, a copy lands in it, or an upload
// carries guest bytes that are not all zero. An unmapped layer reads as zero - Metal's
// sparse tier 1 contract (MTLResource.h, MTLTextureSparseTier1): an unbacked region reads
// (0, 0, 0, 0) for a format with alpha and (0, 0, 0, 1) otherwise, which is exactly what a
// texel of zero bytes samples as, and a write to it does nothing - so a layer whose guest
// bytes are all zero needs no tiles to agree with guest memory, and an upload that finds a
// mapped layer's guest bytes all zero gives its tiles back.
//
// A sparse heap is committed whole the first time it is used (measured on an M4: footprint
// follows heap.size, not the mapped tiles), and a sparse texture can only take tiles from
// the heap it was made on. The heap is therefore sized for the layers in use, and when a
// write needs more the texture is replaced by one on a heap twice as large, its mapped
// layers copied over; at the full layer count it becomes an ordinary texture.
//
// Metal refuses sparse textures with the shader-write usage and dual-plane depth-stencil
// formats on Apple GPUs, so only images whose shape lacks both qualify; a usage the image
// later needs replaces it with an ordinary texture through the usual path.
//
// SHARPEMU_METAL_V2_SPARSE_LARGE=0 makes every texture an ordinary one.
internal sealed unsafe partial class MetalImageCache
{
    private const ulong DefaultSparseMinimumBytes = 64UL * 1024 * 1024;
    private const nint GpuFamilyApple6 = 1006;
    private const nuint HeapTypeSparse = 2;
    private const nuint HazardTrackingModeTracked = 2;
    private const nuint SparseMappingMap = 0;
    private const nuint SparseMappingUnmap = 1;
    private const nuint AssociationRetain = 0x301;

    private static readonly bool SparseLargeDefault =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_SPARSE_LARGE") != "0";

    private static readonly nint SelSupportsFamily = MetalNative.Selector("supportsFamily:");
    private static readonly nint SelSparseTileSize = MetalNative.Selector("sparseTileSizeWithTextureType:pixelFormat:sampleCount:");
    private static readonly nint SelSparseTileBytes = MetalNative.Selector("sparseTileSizeInBytes");
    private static readonly nint SelNewHeap = MetalNative.Selector("newHeapWithDescriptor:");
    private static readonly nint SelSetType = MetalNative.Selector("setType:");
    private static readonly nint SelSetSize = MetalNative.Selector("setSize:");
    private static readonly nint SelSetHazardTrackingMode = MetalNative.Selector("setHazardTrackingMode:");
    private static readonly nint SelHeapSize = MetalNative.Selector("size");
    private static readonly nint SelFirstMipmapInTail = MetalNative.Selector("firstMipmapInTail");
    private static readonly nint SelRespondsToSelector = MetalNative.Selector("respondsToSelector:");
    private static readonly nint SelSparseTextureTier = MetalNative.Selector("sparseTextureTier");
    private static readonly nint SelResourceStateEncoder = MetalNative.Selector("resourceStateCommandEncoder");
    private static readonly nint SelUpdateTextureMapping = MetalNative.Selector("updateTextureMapping:mode:region:mipLevel:slice:");
    private static readonly nint SelBlitCommandEncoder = MetalNative.Selector("blitCommandEncoder");
    private static readonly nint SelCopyTextureSlice = MetalNative.Selector(
        "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:" +
        "toTexture:destinationSlice:destinationLevel:destinationOrigin:");
    private static readonly nint SelEndEncoding = MetalNative.Selector("endEncoding");

    // The key the heap hangs off its texture by; only its address matters.
    private static readonly nint HeapAssociationKey = (nint)NativeMemory.Alloc(1);

    private readonly List<uint> _sparseMaps = [];
    private readonly List<uint> _sparseUnmaps = [];
    private int _sparseSupport;
    private long _sparseCreated;
    private long _sparseGrowths;
    private long _sparseLayersMapped;
    private long _sparseLayersUnmapped;

    /// <summary>Whether a large one-level array texture is created sparse.</summary>
    public bool SparseLargeTextures { get; set; } = SparseLargeDefault;

    /// <summary>The guest size from which a qualifying array texture is created sparse.</summary>
    public ulong SparseMinimumBytes { get; set; } = DefaultSparseMinimumBytes;

    /// <summary>
    /// Sparse textures created, texture replacements for more tiles, and layers mapped and
    /// unmapped, since the last read.
    /// </summary>
    public (long Created, long Grown, long Mapped, long Unmapped) TakeSparseCounters() =>
        (Interlocked.Exchange(ref _sparseCreated, 0), Interlocked.Exchange(ref _sparseGrowths, 0),
         Interlocked.Exchange(ref _sparseLayersMapped, 0), Interlocked.Exchange(ref _sparseLayersUnmapped, 0));

    // The residency of the image's current texture, or null for an ordinary one.
    private static MetalSparseResidency? SparseOf(MetalCachedImage image) =>
        image.Sparse is { } sparse && sparse.Texture == image.Texture && image.Texture != 0 ? sparse : null;

    // The bytes the device holds for a sparse texture: its own allocation and its heap.
    private static ulong SparseHeapBytes(MetalCachedImage image) => SparseOf(image)?.HeapBytes ?? 0;

    private bool SparseSupported
    {
        get
        {
            if (_sparseSupport == 0)
            {
                _sparseSupport = MetalNative.SendBool(_owner.Device, SelSupportsFamily, GpuFamilyApple6) ? 1 : -1;
            }

            return _sparseSupport > 0;
        }
    }

    private bool QualifiesForSparse(in MetalTextureShape shape, in ImageDescription description) =>
        SparseLargeTextures && description.Data.Size >= SparseMinimumBytes &&
        shape.TextureType == Mtl.TextureType2DArray && shape.MipLevels == 1 && shape.SampleCount == 1 &&
        shape.ArrayLength >= 2 && !shape.Lossy &&
        (shape.Usage & (Mtl.TextureUsageShaderWrite | (ulong)MtlTextureUsage.ShaderAtomic)) == 0 &&
        !MetalGuestFormats.HasStencilPlane((MtlPixelFormat)(uint)shape.PixelFormat) && !description.HasStencil &&
        TrySplitLayers(in description, out var layers, out _) && layers == shape.ArrayLength &&
        SparseSupported;

    /// <summary>
    /// A new image's texture: sparse when it qualifies and the layers its guest bytes fill
    /// leave most of the array empty, otherwise an ordinary one. Caller holds the lock.
    /// </summary>
    private nint CreateImageTexture(in MetalTextureShape shape, in ImageDescription description, out MetalSparseResidency? sparse)
    {
        sparse = null;
        if (QualifiesForSparse(in shape, in description))
        {
            TrySplitLayers(in description, out var layers, out var layerBytes);
            var filled = 0u;
            for (var layer = 0u; layer < layers; layer++)
            {
                filled += GuestMayBeNonZero(description.Data.Address + (layerBytes * layer), layerBytes) ? 1u : 0u;
            }

            // One more for the layer the first pass binds.
            var capacity = SparseCapacity(filled + 1, layers);
            if (capacity < layers)
            {
                var texture = CreateSparseTexture(in shape, capacity, out sparse);
                if (texture != 0)
                {
                    _sparseCreated++;
                    return texture;
                }
            }
        }

        return CreateTexture(in shape, in description);
    }

    // Layers a heap is made for: a power of two, at least an eighth of the array.
    private static uint SparseCapacity(uint needed, uint layers)
    {
        var capacity = Math.Max(System.Numerics.BitOperations.RoundUpToPowerOf2(Math.Max(needed, 2u)), Math.Max(layers / 8, 2u));
        return Math.Min(capacity, layers);
    }

    // A sparse texture on a heap with tiles for `capacity` layers, or zero when Metal will
    // not make one for this shape.
    private nint CreateSparseTexture(in MetalTextureShape shape, uint capacity, out MetalSparseResidency? sparse)
    {
        sparse = null;
        var device = _owner.Device;
        var tile = MetalV2Native.SendSparseTileSize(
            device, SelSparseTileSize, (nuint)shape.TextureType, (nuint)shape.PixelFormat, 1);
        var tileBytes = MetalV2Native.SendValue(device, SelSparseTileBytes);
        if (tile.Width == 0 || tile.Height == 0 || tileBytes == 0)
        {
            return 0;
        }

        var tilesWide = (shape.Width + tile.Width - 1) / tile.Width;
        var tilesHigh = (shape.Height + tile.Height - 1) / tile.Height;
        var layerTileBytes = (ulong)(tilesWide * tilesHigh * tileBytes);
        var heapDescriptor = MetalNative.Send(MetalNative.Send(MetalNative.Class("MTLHeapDescriptor"), SelAlloc), SelInit);
        if (heapDescriptor == 0)
        {
            return 0;
        }

        nint heap;
        try
        {
            MetalV2Native.SendVoidValue(heapDescriptor, SelSetType, HeapTypeSparse);
            MetalV2Native.SendVoidValue(heapDescriptor, SelSetStorageMode, Mtl.StorageModePrivate);
            // The mapping updates, the uploads and the passes over these tiles are ordered by
            // Metal, as for any texture outside a heap.
            MetalV2Native.SendVoidValue(heapDescriptor, SelSetHazardTrackingMode, HazardTrackingModeTracked);
            MetalV2Native.SendVoidValue(heapDescriptor, SelSetSize, (nuint)(layerTileBytes * capacity));
            heap = MetalNative.Send(device, SelNewHeap, heapDescriptor);
        }
        finally
        {
            MetalNative.SendVoid(heapDescriptor, SelRelease);
        }

        if (heap == 0)
        {
            return 0;
        }

        var descriptor = NewTextureDescriptor(in shape);
        var texture = MetalNative.Send(heap, SelNewTexture, descriptor);
        MetalNative.SendVoid(descriptor, SelRelease);
        // Layer 0 must have tiles of its own rather than sit in the mip tail, and the texture
        // must promise that unbacked tiles read as zero.
        if (texture == 0 || MetalV2Native.SendValue(texture, SelFirstMipmapInTail) == 0 ||
            (MetalNative.SendBool(texture, SelRespondsToSelector, SelSparseTextureTier) &&
             MetalV2Native.SendValue(texture, SelSparseTextureTier) == 0))
        {
            if (texture != 0)
            {
                MetalNative.SendVoid(texture, SelRelease);
            }

            MetalNative.SendVoid(heap, SelRelease);
            return 0;
        }

        // A texture does not keep its heap alive on its own; the association releases the
        // heap when the texture goes, whichever path retires it.
        MetalV2Native.SetAssociatedObject(texture, HeapAssociationKey, heap, AssociationRetain);
        var heapBytes = MetalV2Native.SendValue64(heap, SelHeapSize);
        MetalNative.SendVoid(heap, SelRelease);
        sparse = new MetalSparseResidency(texture, heap, heapBytes, capacity, shape.ArrayLength, tilesWide, tilesHigh);
        return texture;
    }

    // Whether the guest bytes may hold anything but zero: unreadable or still being written
    // by recorded GPU work counts as yes.
    private bool GuestMayBeNonZero(ulong address, ulong size)
    {
        if (size == 0)
        {
            return false;
        }

        if (_owner.HasPendingGpuWrite(address, size) ||
            !_backing.TryResolveBackingAliasSpan(address, size, out var alias) || alias == 0)
        {
            return true;
        }

        for (var offset = 0UL; offset < size;)
        {
            var chunk = (int)Math.Min(size - offset, 1UL << 30);
            if (new ReadOnlySpan<byte>((void*)(alias + offset), chunk).IndexOfAnyExcept((byte)0) >= 0)
            {
                return true;
            }

            offset += (ulong)chunk;
        }

        return false;
    }

    /// <summary>
    /// Before an upload of image bytes [start, end): every layer whose guest bytes in the
    /// run are not all zero gets tiles, and a mapped layer the run covers whole whose guest
    /// bytes are all zero gives them back. Caller holds the lock.
    /// </summary>
    private void PrepareSparseUpload(MetalCachedImage image, ulong start, ulong end)
    {
        if (SparseOf(image) is not { } sparse || end <= start)
        {
            return;
        }

        var address = image.Description.Data.Address;
        var layerBytes = image.Description.Data.Size / sparse.Layers;
        var first = (uint)Math.Min(start / layerBytes, sparse.Layers - 1);
        var last = (uint)Math.Min((end - 1) / layerBytes, sparse.Layers - 1);
        _sparseMaps.Clear();
        _sparseUnmaps.Clear();
        for (var layer = first; layer <= last; layer++)
        {
            var layerStart = layerBytes * layer;
            var runStart = Math.Max(start, layerStart);
            var runEnd = Math.Min(end, layerStart + layerBytes);
            var filled = GuestMayBeNonZero(address + runStart, runEnd - runStart);
            if (filled && !sparse.Mapped[layer])
            {
                _sparseMaps.Add(layer);
            }
            else if (!filled && sparse.Mapped[layer] && runStart == layerStart && runEnd == layerStart + layerBytes)
            {
                _sparseUnmaps.Add(layer);
            }
        }

        UpdateSparseMapping(image, sparse);
    }

    /// <summary>Gives tiles to layers [first, first + count) before something writes them. Caller holds the lock.</summary>
    private void MapSparseLayers(MetalCachedImage image, uint first, uint count)
    {
        if (SparseOf(image) is not { } sparse)
        {
            return;
        }

        _sparseMaps.Clear();
        _sparseUnmaps.Clear();
        var end = (uint)Math.Min((ulong)first + count, sparse.Layers);
        for (var layer = first; layer < end; layer++)
        {
            if (!sparse.Mapped[layer])
            {
                _sparseMaps.Add(layer);
            }
        }

        UpdateSparseMapping(image, sparse);
    }

    /// <summary>
    /// Before a write the cache cannot narrow to layers (a copy, a resolve, a store): the image
    /// gets an ordinary texture holding what the sparse one held. Caller holds the lock.
    /// </summary>
    private void EnsureSparseBacked(MetalCachedImage image)
    {
        if (SparseOf(image) is { } sparse)
        {
            MapSparseLayers(image, 0, sparse.Layers);
        }
    }

    // Applies _sparseUnmaps and _sparseMaps, on a texture with a heap large enough for them.
    private void UpdateSparseMapping(MetalCachedImage image, MetalSparseResidency sparse)
    {
        if (_sparseMaps.Count == 0 && _sparseUnmaps.Count == 0)
        {
            return;
        }

        var needed = sparse.MappedCount - (uint)_sparseUnmaps.Count + (uint)_sparseMaps.Count;
        if (needed > sparse.Capacity)
        {
            GrowSparse(image, sparse, needed);
            return;
        }

        var context = _owner.BeginTransfer(MetalTransferCause.ImageCopy, image.Texture);
        var encoder = MetalNative.Send(context.CommandBuffer, SelResourceStateEncoder);
        if (encoder == 0)
        {
            throw GpuFatal.Fatal("A Metal resource state encoder could not be created for a sparse texture.");
        }

        // Unmapped first, so the tiles they give back serve the layers mapped after them.
        foreach (var layer in _sparseUnmaps)
        {
            EncodeLayerMapping(encoder, sparse, layer, SparseMappingUnmap);
            sparse.Mapped[layer] = false;
            sparse.MappedCount--;
            _sparseLayersUnmapped++;
        }

        foreach (var layer in _sparseMaps)
        {
            EncodeLayerMapping(encoder, sparse, layer, SparseMappingMap);
            sparse.Mapped[layer] = true;
            sparse.MappedCount++;
            _sparseLayersMapped++;
        }

        MetalNative.SendVoid(encoder, SelEndEncoding);
        if (_sparseMaps.Count != 0)
        {
            // Tiles fresh from the heap hold whatever they last held. A layer that gains
            // them held zeros, so it is given those of a layer still without tiles, which
            // there always is: the heap holds fewer layers than the array.
            var zeroLayer = Array.IndexOf(sparse.Mapped, false);
            if (zeroLayer < 0)
            {
                throw GpuFatal.Fatal("A sparse texture has every layer mapped on a heap for fewer.");
            }

            var blit = MetalRenderHost.PassProfileEnabled
                ? MetalTiler.ProfiledEncoder(context, context.CommandBuffer, MetalProfiledEncoder.Blit, "sparse", false)
                : MetalNative.Send(context.CommandBuffer, SelBlitCommandEncoder);
            if (blit == 0)
            {
                throw GpuFatal.Fatal("A Metal blit encoder could not be created for a sparse texture.");
            }

            context.EncoderStarted(blit, render: false);

            var size = new MtlSize { Width = image.Shape.Width, Height = image.Shape.Height, Depth = 1 };
            foreach (var layer in _sparseMaps)
            {
                MetalNative.SendCopyTextureSlice(
                    blit, SelCopyTextureSlice, sparse.Texture, (nuint)zeroLayer, 0, default, size,
                    sparse.Texture, layer, 0, default);
            }

            context.EncoderEnding(blit, render: false);
            MetalNative.SendVoid(blit, SelEndEncoding);
        }

        image.NoteContentChanged();
        if (context.WantsEncodedLabels)
        {
            context.ProfileEncoded(ProfileLabel("sparse-mapping", in image.Description));
        }
    }

    private static void EncodeLayerMapping(nint encoder, MetalSparseResidency sparse, uint layer, nuint mode) =>
        MetalV2Native.SendUpdateTextureMapping(
            encoder, SelUpdateTextureMapping, sparse.Texture, mode,
            new MtlRegion { Width = sparse.TilesWide, Height = sparse.TilesHigh, Depth = 1 }, 0, layer);

    // The heap has tiles for fewer layers than the image now needs: a texture on a heap for
    // twice as many (an ordinary texture at the full count) takes its place, with every layer
    // copied over that has tiles on it. A layer that gains tiles here held zeros and gets the
    // zeros its unmapped twin reads; one that gives them back holds zeros from now on.
    private void GrowSparse(MetalCachedImage image, MetalSparseResidency sparse, uint needed)
    {
        var target = (bool[])sparse.Mapped.Clone();
        foreach (var layer in _sparseUnmaps)
        {
            target[layer] = false;
        }

        foreach (var layer in _sparseMaps)
        {
            target[layer] = true;
        }

        var shape = image.Shape;
        var capacity = SparseCapacity(Math.Max(needed, sparse.Capacity * 2), sparse.Layers);
        MetalSparseResidency? grown = null;
        var texture = capacity < sparse.Layers ? CreateSparseTexture(in shape, capacity, out grown) : 0;
        if (texture == 0)
        {
            grown = null;
            texture = CreateTexture(in shape, image.Description);
        }

        var context = _owner.BeginTransfer(MetalTransferCause.ImageCopy, image.Texture, texture);
        if (grown is not null)
        {
            var encoder = MetalNative.Send(context.CommandBuffer, SelResourceStateEncoder);
            if (encoder == 0)
            {
                throw GpuFatal.Fatal("A Metal resource state encoder could not be created for a sparse texture.");
            }

            for (var layer = 0u; layer < grown.Layers; layer++)
            {
                if (target[layer])
                {
                    EncodeLayerMapping(encoder, grown, layer, SparseMappingMap);
                    grown.Mapped[layer] = true;
                    grown.MappedCount++;
                }
            }

            MetalNative.SendVoid(encoder, SelEndEncoding);
        }

        if (grown is null)
        {
            // An ordinary texture starts undefined: every layer is copied, the ones without
            // tiles as the zeros they read.
            _transfer.EncodeWholeTextureCopy(context, image.Texture, texture);
        }
        else
        {
            var blit = MetalRenderHost.PassProfileEnabled
                ? MetalTiler.ProfiledEncoder(context, context.CommandBuffer, MetalProfiledEncoder.Blit, "sparse", false)
                : MetalNative.Send(context.CommandBuffer, SelBlitCommandEncoder);
            if (blit == 0)
            {
                throw GpuFatal.Fatal("A Metal blit encoder could not be created for a sparse texture.");
            }

            context.EncoderStarted(blit, render: false);

            // Every layer with tiles on the new heap: what the old texture held there, or the
            // zeros an unmapped layer of it reads.
            var size = new MtlSize { Width = shape.Width, Height = shape.Height, Depth = 1 };
            for (var layer = 0u; layer < sparse.Layers; layer++)
            {
                if (target[layer])
                {
                    MetalNative.SendCopyTextureSlice(
                        blit, SelCopyTextureSlice, image.Texture, layer, 0, default, size, texture, layer, 0, default);
                }
            }

            context.EncoderEnding(blit, render: false);
            MetalNative.SendVoid(blit, SelEndEncoding);
        }

        _sparseLayersMapped += _sparseMaps.Count;
        _sparseLayersUnmapped += _sparseUnmaps.Count;
        _sparseGrowths++;
        RetireImage(image.ReplaceWithCopy(texture, in shape, _owner.CurrentSerial));
        image.Sparse = grown;
        RecountAllocatedLocked(image);
        if (context.WantsEncodedLabels)
        {
            context.ProfileEncoded(ProfileLabel("sparse-grow", in image.Description));
        }
    }
}
