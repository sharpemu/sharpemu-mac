// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Metal.V2;

// The image's share of the upload prelude's conditions (MetalRenderHost.UploadPrelude.cs): an
// upload needed while a render pass is open may run ahead of the open command buffer only when
// nothing recorded in that buffer touches the texture or writes the bytes it reads.
internal sealed partial class MetalImageCache
{
    private const byte AliasOrGpuCauses =
        (1 << (int)MetalImageDirtyCause.AliasWrite) | (1 << (int)MetalImageDirtyCause.GpuWrite);

    // The conditions an upload of [start, end) of the image fails, the host's included. whole: the
    // upload replaces the whole resident range rather than a stale run.
    private MetalPreludeBlock PreludeBlocks(MetalCachedImage image, ulong start, ulong end, bool whole)
    {
        var blocks = MetalPreludeBlock.None;
        var causes = image.StaleCauses;
        if ((causes & AliasOrGpuCauses) != 0)
        {
            blocks |= (causes & (1 << (int)MetalImageDirtyCause.AliasWrite)) != 0 ? MetalPreludeBlock.CauseAlias : 0;
            blocks |= (causes & (1 << (int)MetalImageDirtyCause.GpuWrite)) != 0 ? MetalPreludeBlock.CauseGpu : 0;
        }

        // Bound (sampled, attached, stored) in the open buffer: the bindings note the open serial
        // after the refresh, so a use before this upload has already left it there.
        var serial = _owner.CurrentSerial;
        if (image.LastUseSerial >= serial || image.GpuWriteSerial >= serial)
        {
            blocks |= MetalPreludeBlock.UsedInBuffer;
        }

        if (whole && image.DeviceCopiedSinceRefresh)
        {
            blocks |= MetalPreludeBlock.DeviceCopy;
        }

        if (image.Sparse is not null)
        {
            blocks |= MetalPreludeBlock.Sparse;
        }

        ref readonly var description = ref image.Description;
        if (description.IsDepth || description.HasStencil)
        {
            blocks |= MetalPreludeBlock.DepthStencil;
        }

        if (end - start > TransferPieceBytes)
        {
            blocks |= MetalPreludeBlock.Pieces;
        }

        return blocks | _owner.PreludeBlocks(image.Texture, description.Data.Address + start, end - start);
    }

    // Counts an upload that met an open render pass (null when none was open), with a staged
    // gather the source needed added to what it failed.
    private void NotePassEndingUpload(MetalImageDirtyCause cause, MetalPreludeBlock? blocks, long gathersBefore)
    {
        if (blocks is not { } failed)
        {
            return;
        }

        if (_uploadGathers != gathersBefore)
        {
            failed |= MetalPreludeBlock.StagedGather;
        }

        _owner.NotePassEndingUpload(cause, failed);
    }
}
