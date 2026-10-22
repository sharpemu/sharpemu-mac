// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Images;

namespace SharpEmu.Gpu.Metal.V2;

// A guest byte range as seen by the GPU: an MTLBuffer (a no-copy guest window or an
// upload ring slice) and the byte offset of the range's first byte inside it.
internal readonly record struct MetalGuestRange(nint Buffer, ulong Offset, ulong Size, ulong GpuAddress)
{
    public bool IsValid => Buffer != 0;
}

// The command buffer currently being recorded and the completion serials of the queue.
// Resources a transfer allocates for this command buffer may be reused once
// CompletedSerial >= SubmissionSerial. Callers end any open encoder before passing it,
// except one they leave to be shared (TryShareComputeEncoder).
internal interface IMetalSubmissionContext
{
    nint Device { get; }

    nint CommandBuffer { get; }

    // The serial the shared event reaches when CommandBuffer completes.
    ulong SubmissionSerial { get; }

    // The last serial the shared event has reached.
    ulong CompletedSerial { get; }

    // Profiling only: everything encoded into CommandBuffer up to here belongs to this
    // label. The host closes and times the command buffer under
    // SHARPEMU_METAL_V2_PROFILE_SHADERS=1 and does nothing otherwise, which is why this
    // may be called from any encode path without a cost.
    void ProfileEncoded(string label)
    {
        _ = label;
    }

    // Whether ProfileEncoded reads its label. A caller formats the label only when it does:
    // transfers run tens of thousands of times a second, and the strings were garbage.
    bool WantsEncodedLabels => false;

    // The compute encoder a transfer made of compute dispatches alone may record them into
    // instead of opening one of its own, or zero. The host leaves its open compute encoder
    // to an image upload or writeback (SHARPEMU_METAL_V2_TRANSFER_IN_ENCODER) and hands it
    // over only when nothing that lands at that encoder's end - a deferred command-processor
    // write, a staged store - touches guest bytes [guestAddress, guestAddress + size); the
    // dispatches are fenced from the encoder's own by memory barriers. The caller records
    // its dispatches and then calls EndSharedCompute, never endEncoding.
    nint TryShareComputeEncoder(ulong guestAddress, ulong size)
    {
        _ = guestAddress;
        _ = size;
        return 0;
    }

    // The transfer has recorded its dispatches into the shared compute encoder.
    void EndSharedCompute()
    {
    }

    // Ends the encoder this transfer was allowed to share, as a transfer: called before the
    // transfer opens an encoder of its own, since a command buffer has one open at a time.
    void ReleaseSharedEncoder()
    {
    }
}

// Moves image contents between guest memory and Metal textures entirely on the GPU.
// Implemented by MetalImageTransfer (tiler workstream); used by MetalImageCache.
// Why an encoder was torn down to record a transfer. Every one of these ends the open
// render pass, and a pass restart costs the attachment a store and a reload, so which
// cause dominates decides what is worth avoiding.
internal enum MetalTransferCause
{
    Upload,
    Writeback,
    ColorClear,
    DepthClear,
    GuestFill,
    GuestCopy,
    ImageCopy,
    EndOfPipeLabel,
}

internal interface IMetalImageTransfer
{
    // The Metal texture shape the cache must allocate for a description (format,
    // type, extent, mips, layers, usage). The transfer and the cache agree on it.
    // A firstLevel above zero shapes a texture that holds only guest levels
    // [firstLevel, levels): its level 0 is the guest's level firstLevel.
    MetalTextureShape ShapeFor(in ImageDescription description, uint firstLevel = 0);

    // The guest bytes, as offsets into description.Data, that the transfer reads for the
    // levels from firstLevel down: the smallest run every transfer of those levels lies in,
    // from the same plan an upload uses. (0, Data.Size) for firstLevel zero.
    (ulong Start, ulong End) LevelBytes(in ImageDescription description, uint firstLevel);

    // Detiles/copies every subresource in range from guest memory into the texture.
    //
    // source covers [guestBase, guestBase + pieceSize) of description.Data — the piece of
    // the image the caller could map — and range names the subresources inside it.
    // Passing (0, description.Data.Size) with the whole range is the ordinary whole-image
    // transfer; a caller that cannot map a very large image whole splits it into pieces so
    // that each one lands in a guest-memory window and the staging stays piece-sized.
    // Stencil, when present, is passed separately and is cut the same way.
    //
    // textureFirstLevel is the guest level the texture's level 0 holds (see ShapeFor):
    // range still names guest levels, which must all be at or past it, and each lands
    // in texture level (guest level - textureFirstLevel).
    void EncodeGuestToTexture(IMetalSubmissionContext context, in ImageDescription description,
        SubresourceRange range, MetalGuestRange source, MetalGuestRange stencilSource, nint texture,
        ulong guestBase, ulong pieceSize, uint textureFirstLevel = 0);

    // Retiles/copies every subresource in range from the texture back into guest memory,
    // with the same piece contract as the upload, textureFirstLevel included: a texture
    // held from a later level writes only the guest levels it holds, at their full-chain
    // offsets, and leaves every other byte of the destination as it was.
    void EncodeTextureToGuest(IMetalSubmissionContext context, in ImageDescription description,
        SubresourceRange range, nint texture, MetalGuestRange destination, MetalGuestRange stencilDestination,
        ulong guestBase, ulong pieceSize, uint textureFirstLevel = 0);

    // Copies each range's bytes into the range beside it, in order, in one blit pass: how a
    // writeback no window holds whole is gathered into scratch and scattered back out.
    void EncodeBufferCopies(IMetalSubmissionContext context, ReadOnlySpan<(MetalGuestRange From, MetalGuestRange To)> copies);

    // Hands one cached surface's contents to another over the same guest bytes, on the
    // device. False when the pair is not one the transfer can carry, and the caller
    // publishes through guest memory instead.
    bool TryEncodeTextureToTexture(IMetalSubmissionContext context,
        in ImageDescription sourceDescription, nint sourceTexture,
        in ImageDescription targetDescription, nint targetTexture);

    // Whether the depth image's texture, sampled in its own format, reads what a texture of
    // the colour description over the same guest bytes would (MetalImageTransfer). False
    // keeps the two as separate textures.
    bool SamplesAsDepthTwin(in ImageDescription depthDescription, in ImageDescription colorDescription) => false;

    // Copies an 8-bit image over a depth image's stencil bytes into that depth texture's
    // stencil plane, on the device; the depth plane is left as it is. False when the pair is
    // not one the transfer can carry.
    bool TryEncodeStencilFromColor(IMetalSubmissionContext context,
        in ImageDescription colorDescription, nint colorTexture,
        in ImageDescription depthDescription, nint depthTexture) => false;

    // Copies every slice and level of one texture into another of the same shape, on the
    // device. The two may differ in usage only.
    void EncodeWholeTextureCopy(IMetalSubmissionContext context, nint sourceTexture, nint targetTexture);

    // Where a run of an image's guest bytes may end without cutting one of its transfers
    // in half: a caller that found a window covering [start, end) asks for the end it
    // should actually use, so every subresource is either wholly inside a run or wholly
    // inside another one.
    ulong AlignRunEnd(in ImageDescription description, ulong start, ulong end);

    // Where the next run may begin, for a caller that had to step over guest bytes no
    // window covers: a transfer the step landed inside carries nothing, so the run after
    // it starts past that transfer rather than in the middle of it.
    ulong AlignRunStart(in ImageDescription description, ulong offset);
}

// What MTLTextureDescriptor fields an image needs.
internal readonly record struct MetalTextureShape(
    ulong PixelFormat,      // MTLPixelFormat
    ulong TextureType,      // MTLTextureType
    uint Width,
    uint Height,
    uint Depth,
    uint MipLevels,
    uint ArrayLength,
    uint SampleCount,
    ulong Usage,            // MTLTextureUsage
    bool Lossy = false);    // MTLTextureCompressionTypeLossy (MetalImageCache.Lossy.cs)
