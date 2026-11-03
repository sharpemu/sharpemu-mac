// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Images;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// A title's own compute memset or memcpy over the exact bytes of a cached image, done
/// to the image's texture instead of to guest memory. Run as written, the kernel binds
/// the image as a storage buffer: the image is written back to guest memory first, the
/// kernel rewrites the bytes, and the image is uploaded again on its next use — measured
/// on Demon's Souls as about 12 ms of GPU time per title frame. A clear or a texture
/// copy writes the same texels, and the image ends the way a render into it leaves it:
/// GPU-modified, its aliases handed the bytes through the usual alias rules, and guest
/// memory owed a writeback that any reader of the bytes triggers.
/// </summary>
internal sealed partial class MetalImageCache
{
    /// <summary>
    /// Clears every element of the image over exactly [address, address + size) to the
    /// texel the fill pattern repeats; false (with the reason) leaves the kernel to run.
    /// </summary>
    public bool TryClearImageFromKernelFill(ulong address, ulong size, ReadOnlySpan<byte> pattern, out string? fallback)
    {
        fallback = null;
        if (!IsValidRange(address, size))
        {
            fallback = "size";
            return false;
        }

        using var held = _lock.Hold();
        var exact = new ImageQueryResults();
        if (FindKernelTargetsLocked(address, size, ref exact) is { } missing)
        {
            fallback = missing;
            return false;
        }

        // Several images over the same bytes take the clear through the one used last,
        // the way aliases of a render target take its contents.
        var selected = exact[0];
        for (var index = 1; index < exact.Count; index++)
        {
            if (_slots[exact[index]].LastAccessTick > _slots[selected].LastAccessTick)
            {
                selected = exact[index];
            }
        }

        var image = _slots[selected];
        var description = image.Description;
        var shape = image.Shape;
        if (!IsPlainKernelTarget(in description, in shape))
        {
            fallback = "shape";
            return false;
        }

        Span<byte> texelBytes = stackalloc byte[16];
        var texel = texelBytes[..(int)description.BytesPerBlock];
        if (!MetalKernelFill.TryResolveTexel(pattern, texel.Length, texel))
        {
            fallback = "pattern";
            return false;
        }

        var color = default(GpuClearColor);
        var depth = 0.0f;
        if (description.IsDepth)
        {
            if (description.PixelFormat != GpuFormat.D32Sfloat || texel.Length != sizeof(uint) ||
                !PackedClearValue.TryDecodeDepth(description.PixelFormat, BitConverter.ToUInt32(texel), out depth) ||
                (depth != 0 && !float.IsNormal(depth)))
            {
                fallback = "value";
                return false;
            }
        }
        else if (MetalV2Formats.PixelFormat(description.PixelFormat) != shape.PixelFormat ||
                 !MetalKernelFill.TryDecodeTexel(description.PixelFormat, texel, out color))
        {
            fallback = "value";
            return false;
        }

        if (!TakeKernelWriteOwnershipLocked(selected, image))
        {
            fallback = "texture";
            return false;
        }

        shape = image.Shape;
        for (var level = 0u; level < shape.MipLevels; level++)
        {
            var range = new SubresourceRange(level, 1, 0, shape.ArrayLength);
            var width = Math.Max(shape.Width >> (int)level, 1);
            var height = Math.Max(shape.Height >> (int)level, 1);
            if (description.IsDepth)
            {
                _owner.ClearTextureDepthStencil(
                    image.Texture, shape.PixelFormat, in range, width, height, depth, 0, clearDepth: true, clearStencil: false);
            }
            else
            {
                _owner.ClearTextureColor(image.Texture, shape.PixelFormat, in range, width, height, in color);
            }
        }

        CompleteKernelWriteLocked(image);
        return true;
    }

    /// <summary>
    /// Readies a texture copy for a kernel that copies the bytes of exactly one cached
    /// image onto exactly another of the same layout: the source is brought up to date
    /// with guest memory and the destination takes the bytes from its aliases. The
    /// caller encodes the copy and then calls <see cref="CompleteKernelCopy"/>.
    /// </summary>
    public bool TryBeginKernelCopy(
        ulong destination,
        ulong source,
        ulong size,
        out MetalCachedImage? sourceImage,
        out MetalCachedImage? destinationImage,
        out ResourceSlotIdentifier destinationIdentifier,
        out string? fallback)
    {
        sourceImage = null;
        destinationImage = null;
        destinationIdentifier = ResourceSlotIdentifier.Invalid;
        fallback = null;
        if (!IsValidRange(destination, size) || !IsValidRange(source, size))
        {
            fallback = "size";
            return false;
        }

        if (source < destination + size && destination < source + size)
        {
            fallback = "overlap";
            return false;
        }

        using var held = _lock.Hold();
        var destinations = new ImageQueryResults();
        if (FindKernelTargetsLocked(destination, size, ref destinations) is { } missing)
        {
            fallback = missing;
            return false;
        }

        var sources = new ImageQueryResults();
        if (FindKernelTargetsLocked(source, size, ref sources) is { } sourceMissing)
        {
            fallback = $"src-{sourceMissing}";
            return false;
        }

        var selectedSource = ResourceSlotIdentifier.Invalid;
        var selectedDestination = ResourceSlotIdentifier.Invalid;
        for (var d = 0; d < destinations.Count; d++)
        {
            var candidate = _slots[destinations[d]];
            if (selectedDestination.IsValid && candidate.LastAccessTick <= _slots[selectedDestination].LastAccessTick)
            {
                continue;
            }

            for (var s = 0; s < sources.Count; s++)
            {
                if (IsKernelCopyCompatible(_slots[sources[s]], candidate))
                {
                    selectedSource = sources[s];
                    selectedDestination = destinations[d];
                    break;
                }
            }
        }

        if (!selectedDestination.IsValid)
        {
            fallback = "mismatch";
            return false;
        }

        var from = _slots[selectedSource];
        var to = _slots[selectedDestination];
        // The copy reads the texture, so the texture has to hold what guest memory and
        // the source's aliases say first, as for any sampled read of it - exactly, since
        // the copy hands the bytes on as the guest's.
        DropLossyCopySourceLocked(from);
        WatchImage(selectedSource);
        SynchronizeAliases(selectedSource, takesGpuOwnership: false);
        RefreshFromGuest(from);
        from.NoteUse(_owner.CurrentSerial);
        if (from.Texture == 0 || !TakeKernelWriteOwnershipLocked(selectedDestination, to))
        {
            fallback = "texture";
            return false;
        }

        sourceImage = from;
        destinationImage = to;
        destinationIdentifier = selectedDestination;
        return true;
    }

    /// <summary>The copy a <see cref="TryBeginKernelCopy"/> readied has been encoded.</summary>
    public void CompleteKernelCopy(ResourceSlotIdentifier destinationIdentifier)
    {
        using var held = _lock.Hold();
        if (_slots.TryGet(destinationIdentifier) is { } image)
        {
            CompleteKernelWriteLocked(image);
        }
    }

    // The images whose bytes are exactly the range, or why there are none. Caller holds the lock.
    private string? FindKernelTargetsLocked(ulong address, ulong size, ref ImageQueryResults exact)
    {
        var overlapped = false;
        foreach (var imageIdentifier in FindImagesInRange(address, size, pageOverlap: false))
        {
            if (_slots.TryGet(imageIdentifier) is not { } image)
            {
                continue;
            }

            overlapped = true;
            if (image.Description.Data.Address == address && image.Description.Data.Size == size)
            {
                exact.Add(imageIdentifier);
            }
        }

        return exact.Count != 0 ? null : overlapped ? "partial" : "no-image";
    }

    // One sample, one plane, whole texels of a 2D or 2D-array texture in the image's own format.
    private static bool IsPlainKernelTarget(in ImageDescription description, in MetalTextureShape shape) =>
        !description.HasStencil && description.Samples <= 1 && !description.IsBlock && !description.IsVolume &&
        !description.Bgra16 && description.BytesPerBlock is > 0 and <= 16 && shape.SampleCount <= 1 &&
        shape.TextureType is Mtl.TextureType2D or Mtl.TextureType2DArray;

    // Byte-identical layouts: the copy of the bytes is the copy of the texels.
    private static bool IsKernelCopyCompatible(MetalCachedImage source, MetalCachedImage destination)
    {
        ref readonly var from = ref source.Description;
        ref readonly var to = ref destination.Description;
        var fromShape = source.Shape;
        var toShape = destination.Shape;
        // A texture held from its BASE_LEVEL has not got the levels the guest bytes start with.
        if (source.FirstLevel != 0 || destination.FirstLevel != 0 ||
            !IsPlainKernelTarget(in from, in fromShape) || !IsPlainKernelTarget(in to, in toShape) ||
            from.PixelFormat != to.PixelFormat || from.GuestFormat != to.GuestFormat || from.Type != to.Type ||
            from.Extent != to.Extent || from.Resources != to.Resources || from.Pitch != to.Pitch ||
            from.BytesPerBlock != to.BytesPerBlock || from.TileMode != to.TileMode || from.Data.Size != to.Data.Size ||
            fromShape.PixelFormat != toShape.PixelFormat || fromShape.TextureType != toShape.TextureType ||
            fromShape.Width != toShape.Width || fromShape.Height != toShape.Height || fromShape.Depth != toShape.Depth ||
            fromShape.MipLevels != toShape.MipLevels || fromShape.ArrayLength != toShape.ArrayLength)
        {
            return false;
        }

        for (var level = 0; level < from.Resources.Levels && level < ImageDescription.MaxLevels; level++)
        {
            var a = from.MipLayout[level];
            var b = to.MipLayout[level];
            if (a.Offset != b.Offset || a.Size != b.Size || a.Pitch != b.Pitch || a.Height != b.Height)
            {
                return false;
            }
        }

        return true;
    }

    // What acquiring the image as a render target does before the pass writes it: an owned
    // texture, a watch on its bytes, and every alias over them giving them up. Caller holds the lock.
    private bool TakeKernelWriteOwnershipLocked(ResourceSlotIdentifier imageIdentifier, MetalCachedImage image)
    {
        PromoteLinearView(image);
        EnsureSparseBacked(image);
        if (image.Texture == 0)
        {
            return false;
        }

        image.LastAccessTick = _collectionTick;
        WatchImage(imageIdentifier);
        SynchronizeAliases(imageIdentifier, takesGpuOwnership: true);
        return true;
    }

    // The whole image was rewritten on the GPU: nothing the guest wrote before is still
    // owed to the texture, and guest memory is owed a writeback. Caller holds the lock.
    private void CompleteKernelWriteLocked(MetalCachedImage image)
    {
        image.RefreshComplete();
        image.MarkGpuModified();
        NoteLinearGpuWrite(image);
        image.GpuWriteSerial = _owner.CurrentSerial;
        image.NoteUse(image.GpuWriteSerial);
    }

    /// <summary>
    /// Whether surface metadata (DCC, HTILE, CMASK or FMASK, adopted DCC, or a DCC fill still
    /// waiting for its target) lies in the range. Its bytes mean more than memory says: a fill
    /// over metadata that already holds the clear code still asks for the clear again
    /// (TryAbsorbDccFill), so a kernel there is never taken as a no-op. A surface whose size is
    /// not known is answered by its address.
    /// </summary>
    public bool OverlapsSurfaceMetadata(ulong address, ulong size)
    {
        if (!IsValidRange(address, size))
        {
            return true;
        }

        var end = address + size;
        using var held = _lock.Hold();
        foreach (var (start, metadata) in _surfaceMetadata)
        {
            if (start >= end)
            {
                break;
            }

            var extent = Math.Max(Math.Max(metadata.Size, metadata.RangeSize), Math.Max(metadata.FillSize, 1));
            if (start + extent > address)
            {
                return true;
            }
        }

        foreach (var (start, metadataSize) in _adoptedDccMetadata)
        {
            if (address < start + metadataSize && start < end)
            {
                return true;
            }
        }

        return false;
    }
}
