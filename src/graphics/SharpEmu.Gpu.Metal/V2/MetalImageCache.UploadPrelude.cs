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

    // Encodes the upload of [start, end) into a prelude command buffer committed ahead of the open
    // one (MetalRenderHost.UploadPrelude.cs), when it fails no condition and the switch is on. False
    // leaves it to the open buffer, with blocks saying what it failed (None: the switch is off).
    // True with moved false: the transfer refused it, as it would have in the open buffer.
    private bool TryUploadInPrelude(
        MetalCachedImage image, ulong start, ulong end, ref MetalPreludeBlock blocks, out bool moved)
    {
        moved = false;
        if (blocks != MetalPreludeBlock.None || !_owner.UploadPreludeEnabled)
        {
            return false;
        }

        ref readonly var description = ref image.Description;
        var address = description.Data.Address + start;
        var size = end - start;
        // The source as ResolveUploadSource gives it, without the gather it would encode into the open
        // buffer: a window is read in place when the prelude runs, behind every buffer committed before
        // it; a ring copy is what guest memory holds now, which misses writes still in flight.
        if (!_guestMemory.TryResolveInPlace(address, size, out var source))
        {
            if (UploadGatherEnabled && _owner.HasPendingGpuWrite(address, size))
            {
                blocks |= MetalPreludeBlock.StagedGather;
                return false;
            }

            source = _guestMemory.Resolve(address, size, whole: false, ImageRead);
        }

        if (!source.IsValid || source.Size < size)
        {
            blocks |= MetalPreludeBlock.Unmapped;
            return false;
        }

        if (_owner.BeginUploadPrelude() is not { } prelude)
        {
            return false;
        }

        try
        {
            moved = EncodeUpload(
                prelude, in description, ResidentRange(image), source, default, image.Texture, start, size,
                image.FirstLevel);
        }
        finally
        {
            _owner.EndUploadPrelude(prelude);
        }

        if (!moved)
        {
            _owner.NoteUploadPreludeUnencoded();
        }

        return true;
    }

    // After an image's hoisted uploads, its texture is the open buffer's: it has to outlive the preludes,
    // and an upload of it after a later CPU write, before that buffer is committed, would run ahead of a
    // use these serve. Noted once the refresh is done, so the image's other runs still go ahead on their
    // own conditions.
    private void NoteHoistedUse(MetalCachedImage image, bool hoisted)
    {
        if (hoisted)
        {
            image.NoteUse(_owner.CurrentSerial);
        }
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
