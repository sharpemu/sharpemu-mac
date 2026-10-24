// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Pipelines;

namespace SharpEmu.Gpu.Metal.V2;

internal sealed partial class MetalImageCache
{
    // Owned by _lock. Fixed storage, allocated once, with no guest invalidation hooks.
    // Only exact backing matches qualify: alias/mip/layer overlap decisions still run.
    private readonly ImageLookupMemo[] _drawPathImages = new ImageLookupMemo[32];
    private struct ImageLookupMemo
    {
        public bool Valid;
        public long ImageSet;
        public bool ExactFormat;
        public ImageDescription Description;
        public ImageViewDescription View;
        public ImageRole Role;
        public ResourceSlotIdentifier Image;
    }

    private long _drawPathImageHits, _drawPathImageChanged, _drawPathImageGeneration, _drawPathImageOverlap;
    internal string TakeDrawPathReport() =>
        $"drawpath_image_hit={Interlocked.Exchange(ref _drawPathImageHits, 0)} " +
        $"drawpath_image_changed={Interlocked.Exchange(ref _drawPathImageChanged, 0)} " +
        $"drawpath_image_generation={Interlocked.Exchange(ref _drawPathImageGeneration, 0)} " +
        $"drawpath_image_overlap={Interlocked.Exchange(ref _drawPathImageOverlap, 0)}";

    private ResourceSlotIdentifier LookUpDrawPathImageLocked(ref ImageRequest request, bool exactFormat)
    {
        if (!DrawPathCounters.Enabled) return LookUpImageLocked(ref request, exactFormat);
        var address = request.Description.Data.Address;
        ref var memo = ref _drawPathImages[(int)((address >> 16) ^ (address >> 24) ^ (address >> 32)) & 31];
        if (memo.Valid && memo.ImageSet == _imageSetVersion && memo.ExactFormat == exactFormat &&
            memo.Role == request.Role && memo.View.Equals(request.View) &&
            SameDescription(in memo.Description, in request.Description) &&
            _slots.TryGet(memo.Image) is { } cached && cached.SupportsViewType(request.View))
        {
            cached.LastAccessTick = _collectionTick;
            Interlocked.Increment(ref _drawPathImageHits);
            return memo.Image;
        }
        if (memo.Valid && memo.ImageSet != _imageSetVersion) Interlocked.Increment(ref _drawPathImageGeneration);
        else Interlocked.Increment(ref _drawPathImageChanged);

        var view = request.View;
        var result = LookUpImageLocked(ref request, exactFormat);
        var image = _slots[result];
        if (view.Equals(request.View) && HasSameBacking(in image.Description, in request.Description, exactFormat))
        {
            memo = new ImageLookupMemo { Valid = true, ImageSet = _imageSetVersion, ExactFormat = exactFormat,
                Description = request.Description, View = request.View, Role = request.Role, Image = result };
        }
        else
        {
            memo.Valid = false;
            Interlocked.Increment(ref _drawPathImageOverlap);
        }
        return result;
    }

    // All fields, including mip layouts and clear metadata. Padding can only cause a miss.
    private static bool SameDescription(in ImageDescription left, in ImageDescription right) =>
        MemoryMarshal.AsBytes(MemoryMarshal.CreateReadOnlySpan(ref Unsafe.AsRef(in left), 1)).SequenceEqual(
            MemoryMarshal.AsBytes(MemoryMarshal.CreateReadOnlySpan(ref Unsafe.AsRef(in right), 1)));
}
