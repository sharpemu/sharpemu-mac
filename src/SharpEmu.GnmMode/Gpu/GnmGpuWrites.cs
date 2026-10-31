// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE.GpuMemory;
#if SHARPEMU_GPU_VULKAN
using SharpEmu.Gpu.Vulkan.Buffers;
#endif

namespace SharpEmu.GnmMode.Gpu;

internal static class GnmGpuWrites
{
#if SHARPEMU_GPU_VULKAN
    internal static GuestBufferCache? Buffers => GuestGpuMemoryHook.Current?.Buffers as GuestBufferCache;

    internal static bool Overlap(ulong address, ulong size) =>
        GuestGpuMemoryHook.IsWithinGpuAddressSpace(address, size) && Buffers is { } buffers &&
        (buffers.HasGpuDirtyBytes(address, size) ||
            buffers.MayHaveGpuDirtyPages(address, size) && buffers.HasGpuDirtyPages(address, size));

    internal static bool MayOverlap(ulong address, ulong size) =>
        GuestGpuMemoryHook.IsWithinGpuAddressSpace(address, size) && Buffers?.MayHaveGpuDirtyPages(address, size) == true;
#else
    // No Vulkan buffer cache in this build (SharpEmu.GnmMode.csproj): no GPU-written memory is seen,
    // as with any backend whose buffer store is not that cache.
    internal static bool Overlap(ulong address, ulong size) => false;

    internal static bool MayOverlap(ulong address, ulong size) => false;
#endif
}
