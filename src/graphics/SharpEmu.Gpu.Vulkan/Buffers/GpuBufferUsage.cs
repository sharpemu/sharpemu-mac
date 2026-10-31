// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later


namespace SharpEmu.Gpu.Vulkan.Buffers;

public enum GpuBufferUsage : byte
{
    DeviceLocal,
    Upload,
    Download,
    Stream,
    HostCachedStream,
    MappedDeviceLocal,
}

public static class GpuMemoryPlacement
{
    public const GpuBufferUsage StreamRing = GpuBufferUsage.Stream;
}
