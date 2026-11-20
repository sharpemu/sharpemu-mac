// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Kernel;

namespace SharpEmu.Libs.Gpu.Images;

// A texture streamed into its address range maps only its resident mips. A tiled chain puts
// the mip tail and small levels at the start and level 0 at the end, so the resident part is
// a backed prefix. The GPU samples only resident levels; the unmapped rest reads as zero.
// Public for the Metal image cache, which reads guest image bytes with the same policy (no InternalsVisibleTo
// for extracted backends).
public static class ResidentImageBacking
{
    private const ulong Page = 1UL << 12;

    // SHARPEMU_IMAGE_BACKING_HOLES=1 (default off): unmapped pages anywhere after the first are zero-filled instead of
    // failing. Games sample images whose backing has a backed page after a hole, and that failure is
    // fatal.
    private static readonly bool AllowHoles =
        Environment.GetEnvironmentVariable("SHARPEMU_IMAGE_BACKING_HOLES") == "1";

    // Reads image bytes from CPU-visible backing or PRT backing. A streamed texture's image holds only its
    // resident levels and its upload reads only their bytes, so an unmapped byte inside them is an error,
    // not a level to zero; SHARPEMU_IMAGE_BACKING_HOLES=1 still zero-fills the unmapped pages instead.
    public static bool TryRead(IGuestBackedSpace backing, ulong address, Span<byte> destination) =>
        TryRead(backing, address, destination, AllowHoles);

    // SHARPEMU_IMAGE_BACKING_HOLES as this process read it; a caller with its own test switch passes that instead.
    public static bool HolesAllowed => AllowHoles;

    // TryRead with the hole policy given rather than read from the environment.
    public static bool TryRead(IGuestBackedSpace backing, ulong address, Span<byte> destination, bool allowHoles) =>
        backing.TryReadBacking(address, destination) ||
        KernelMemoryCompatExports.TryReadPrtBacking(backing, address, destination) ||
        TryReadMirroredPrtBacking(backing, address, destination) ||
        allowHoles && TryReadResidentPrefix(backing, address, destination, allowHoles: true);

    // PRT backing in a process without the kernel (the GPU process of the two-process split): the guest process mirrors
    // its PRT apertures and AMM area (PartiallyResidentRanges), and which pages are mapped is the shared view table's,
    // which this process's backing reads. Same result as KernelMemoryCompatExports.TryReadPrtBacking: the address is
    // translated as the kernel translates it, mapped 16 KiB chunks are read, and every other byte reads as zero. (The kernel
    // reads a mapping that is backed whole in one piece; mappings are 16 KiB aligned, so that is the same bytes.)
    // Empty, and so false at once, in a process that has the kernel.
    internal static bool TryReadMirroredPrtBacking(IGuestBackedSpace backing, ulong address, Span<byte> destination)
    {
        if (PartiallyResidentRanges.Mirrored.Length == 0)
        {
            return false;
        }

        var size = (ulong)destination.Length;
        address = GuestVaHole.TranslateRange(address, size);
        if (!PartiallyResidentRanges.MirrorContains(address, size))
        {
            return false;
        }

        destination.Clear();
        var end = address + size;
        for (var chunk = address; chunk < end;)
        {
            var next = Math.Min(end, (chunk & ~(PrtProbeBytes - 1)) + PrtProbeBytes);
            if (backing.IsBackedRange(chunk, next - chunk) &&
                !backing.TryReadBacking(chunk, destination.Slice((int)(chunk - address), (int)(next - chunk))))
            {
                return false;
            }

            chunk = next;
        }

        Interlocked.Increment(ref _mirroredPrtReads);
        return true;
    }

    // The kernel's probe size for a partly backed mapping (KernelMemoryCompatExports.PrtResidencyProbeBytes).
    private const ulong PrtProbeBytes = 0x4000;

    private static long _mirroredPrtReads;

    /// <summary>Image reads answered from the mirrored PRT/AMM ranges (the split's GPU process).</summary>
    public static long MirroredPrtReads => Interlocked.Read(ref _mirroredPrtReads);

    // Copies the backed prefix page by page and zero-fills the unmapped tail. A backed page
    // after a hole is not a mip chain, so it is rejected unless allowHoles.
    internal static bool TryReadResidentPrefix(IGuestBackedSpace backing, ulong address, Span<byte> destination) =>
        TryReadResidentPrefix(backing, address, destination, AllowHoles);

    internal static bool TryReadResidentPrefix(IGuestBackedSpace backing, ulong address, Span<byte> destination, bool allowHoles)
    {
        var size = (ulong)destination.Length;
        if (size == 0 || !backing.IsBackedRange(address, 1))
        {
            return false;
        }

        var resident = true;
        for (ulong offset = 0; offset < size;)
        {
            var chunk = Math.Min(Page - ((address + offset) & (Page - 1)), size - offset);
            var target = destination.Slice((int)offset, (int)chunk);
            var backed = backing.IsBackedRange(address + offset, chunk);
            if (backed && !resident && !allowHoles)
            {
                return false;
            }

            resident = backed && backing.TryReadBacking(address + offset, target);
            if (!resident)
            {
                target.Clear();
            }

            offset += chunk;
        }

        return true;
    }
}
