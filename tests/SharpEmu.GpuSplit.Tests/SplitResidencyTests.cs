// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE.GpuMemory;
using SharpEmu.HLE.Host;
using SharpEmu.Libs.Gpu.Images;
using Xunit;

namespace SharpEmu.GpuSplit.Tests;

/// <summary>
/// The kernel's PRT apertures and AMM area in the GPU process (PartiallyResidentRanges): published in the guest
/// process, mirrored by a call before the kernel call returns, and asked by image reads there as the kernel is asked here.
/// </summary>
[Collection(SplitCollection.Name)]
public sealed unsafe class SplitResidencyTests
{
    private const ulong Chunk = SharedGuestMemory.Chunk;
    private const int ApertureId = 1;

    [MacOSFact]
    public void AnApertureReachesTheGpuProcessBeforeTheKernelCallReturnsAndReadsItsHolesAsZero()
    {
        using var split = new InProcessSplit();
        PartiallyResidentRanges.ResetMirror();
        try
        {
            var guest = split.Guest;
            var space = split.Host.AddressSpace!;
            for (ulong offset = 0; offset < SharedGuestMemory.ViewBytes; offset += sizeof(ulong))
            {
                *guest.Pointer(offset) = 0xA5A5_0000_0000_0000UL | offset;
            }

            // A texture whose first 16 KiB are unmapped and whose other 32 KiB are the view's first bytes.
            var address = guest.Address - 0x4000;
            var bytes = new byte[0xC000];

            // Without the aperture the GPU process has nothing to read the hole from: the upload stops (the caller's fatal).
            Assert.False(ResidentImageBacking.TryRead(space, address, bytes, allowHoles: false));

            split.Session.MirrorResidencyRanges();
            PartiallyResidentRanges.Publish(ApertureId, guest.Address - 4 * Chunk, SharedGuestMemory.ViewBytes + 8 * Chunk, lazilyBacked: true);

            // No wait: the call is answered once the GPU process applied it.
            Assert.Equal(1, split.Host.ResidencyChanges);
            Assert.True(PartiallyResidentRanges.MirrorContains(address, (ulong)bytes.Length));
            Assert.True(ResidentImageBacking.TryRead(space, address, bytes, allowHoles: false));
            Assert.All(bytes[..0x4000], value => Assert.Equal(0, value));
            Assert.Equal(0xA5A5_0000_0000_0000UL, BitConverter.ToUInt64(bytes, 0x4000));
            Assert.Equal(0xA5A5_0000_0000_0000UL | 0x7FF8, BitConverter.ToUInt64(bytes, 0xBFF8));

            // A range past the aperture keeps the hole policy: no zeros without SHARPEMU_IMAGE_BACKING_HOLES.
            var outside = guest.Address + SharedGuestMemory.ViewBytes + 3 * Chunk;
            Assert.False(ResidentImageBacking.TryRead(space, outside, new byte[2 * Chunk], allowHoles: false));
        }
        finally
        {
            PartiallyResidentRanges.Publish(ApertureId, 0, 0, lazilyBacked: true);
            PartiallyResidentRanges.ResetMirror();
        }
    }

    [MacOSFact]
    public void UnmappingAndRemappingInsideAnApertureIsReadFromTheSharedViewTable()
    {
        using var split = new InProcessSplit();
        PartiallyResidentRanges.ResetMirror();
        var guest = split.Guest;
        var second = guest.Address + Chunk;
        try
        {
            split.Session.MirrorResidencyRanges();
            PartiallyResidentRanges.Publish(ApertureId, guest.Address, SharedGuestMemory.ViewBytes, lazilyBacked: true);
            *guest.Pointer(Chunk) = 0x5EC0_0D00UL;
            var space = split.Host.AddressSpace!;
            var bytes = new byte[2 * Chunk];

            Assert.True(ResidentImageBacking.TryRead(space, guest.Address, bytes, allowHoles: false));
            Assert.Equal(0x5EC0_0D00UL, BitConverter.ToUInt64(bytes, (int)Chunk));

            // The guest unmaps the second chunk: the GPU process reads zeros there, never the old bytes.
            Assert.True(guest.Views.Unmap(second, Chunk, out var holePreserved));
            Assert.True(ResidentImageBacking.TryRead(space, guest.Address, bytes, allowHoles: false));
            Assert.Equal(0UL, BitConverter.ToUInt64(bytes, (int)Chunk));

            // And maps it again, over other backing: those bytes.
            if (!holePreserved)
            {
                Assert.Equal(second, guest.Host.ReserveHole(second, Chunk));
            }

            Assert.True(guest.Views.TryMapReservedRange(second, Chunk, 40 * Chunk, HostPageProtection.ReadWrite, out _));
            *(ulong*)second = 0x0DD_BA11UL;
            Assert.True(ResidentImageBacking.TryRead(space, guest.Address, bytes, allowHoles: false));
            Assert.Equal(0x0DD_BA11UL, BitConverter.ToUInt64(bytes, (int)Chunk));

            // The kernel drops the aperture: the GPU process stops zero-filling there.
            _ = guest.Views.Unmap(second, Chunk, out _);
            PartiallyResidentRanges.Publish(ApertureId, 0, 0, lazilyBacked: true);
            Assert.Equal(2, split.Host.ResidencyChanges);
            Assert.False(PartiallyResidentRanges.MirrorContains(guest.Address, Chunk));
            Assert.False(ResidentImageBacking.TryRead(space, guest.Address, bytes, allowHoles: false));
        }
        finally
        {
            PartiallyResidentRanges.Publish(ApertureId, 0, 0, lazilyBacked: true);
            PartiallyResidentRanges.ResetMirror();
            if (!guest.Views.Contains(second, Chunk))
            {
                // The view's own teardown unmaps the whole view: give it back its second chunk.
                _ = guest.Host.ReserveHole(second, Chunk);
                _ = guest.Views.TryMapReservedRange(second, Chunk, SharedGuestMemory.ViewOffset + Chunk, HostPageProtection.ReadWrite, out _);
            }
        }
    }

    [MacOSFact]
    public void RangesPublishedBeforeTheSessionAttachedAreSentAndTheLazyOnesCoverManagedWrites()
    {
        using var split = new InProcessSplit();
        PartiallyResidentRanges.ResetMirror();
        var guest = split.Guest;
        try
        {
            // Published while nothing was attached (the kernel ran before the split started).
            PartiallyResidentRanges.Publish(ApertureId, guest.Address, SharedGuestMemory.ViewBytes, lazilyBacked: true);
            PartiallyResidentRanges.Publish(PartiallyResidentRanges.AmmArea, guest.Address + SharedGuestMemory.ViewBytes, 4 * Chunk, lazilyBacked: false);
            var memory = split.Host.Memory!;
            Assert.False(memory.MarkCpuWrite(guest.Address + 0x100, 8));

            split.Session.MirrorResidencyRanges();
            Assert.Equal(2, split.Host.ResidencyChanges);
            Assert.True(PartiallyResidentRanges.MirrorContains(guest.Address + SharedGuestMemory.ViewBytes, 4 * Chunk));

            // The aperture is lazily backed GPU memory in the GPU process's manager too, as GuestGpuMemoryHook.NoteLazyRange
            // makes it in the guest process's: a managed write there reaches the caches. The AMM area is not one.
            Assert.True(memory.MarkCpuWrite(guest.Address + 0x100, 8));
            Assert.False(memory.MarkCpuWrite(guest.Address + SharedGuestMemory.ViewBytes + 0x100, 8));

            // Moved: the old place is not covered any more.
            PartiallyResidentRanges.Publish(ApertureId, guest.Address + Chunk, Chunk, lazilyBacked: true);
            Assert.False(memory.MarkCpuWrite(guest.Address + 0x100, 8));
            Assert.True(memory.MarkCpuWrite(guest.Address + Chunk + 0x100, 8));
            PartiallyResidentRanges.Publish(ApertureId, 0, 0, lazilyBacked: true);
            Assert.False(memory.MarkCpuWrite(guest.Address + Chunk + 0x100, 8));
        }
        finally
        {
            PartiallyResidentRanges.Publish(ApertureId, 0, 0, lazilyBacked: true);
            PartiallyResidentRanges.Publish(PartiallyResidentRanges.AmmArea, 0, 0, lazilyBacked: false);
            PartiallyResidentRanges.ResetMirror();
        }
    }
}
