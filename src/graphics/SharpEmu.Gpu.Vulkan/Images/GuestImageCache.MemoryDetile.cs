// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE.Diagnostics;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.Libs.VideoOut;
using SharpEmu.Gpu.Vulkan;
using SharpEmu.Gpu.Vulkan.Buffers;
using SharpEmu.Gpu.Vulkan.Pipelines;
using SharpEmu.Gpu.Vulkan.Scheduling;
using SharpEmu.Gpu.Vulkan.Presentation;
namespace SharpEmu.Gpu.Vulkan.Images;

public sealed partial class GuestImageCache
{
    public bool TryDetileMemory(ulong tiledAddress, ulong tiledSize, ulong linearAddress, ulong linearSize,
        ReadOnlySpan<TileTransfer> transfers)
    {
        if (transfers.IsEmpty || !IsValidRange(tiledAddress, tiledSize) || !IsValidRange(linearAddress, linearSize) ||
            (linearAddress & 3) != 0 || (linearSize & 3) != 0 || _scheduler.Current.IsInvalid)
        {
            return false;
        }

        PublishImagesUnderRead(tiledAddress, tiledSize);
        InvalidateMemoryFromGpu(linearAddress, linearSize);
        var (source, sourceOffset) = _bufferCache.ObtainBuffer(tiledAddress, tiledSize, isWritten: false, isTexelBuffer: true);
        var (target, targetOffset) = _bufferCache.ObtainBuffer(linearAddress, linearSize, isWritten: true, isTexelBuffer: true);
        _tiler.Detile(source.Handle, sourceOffset, tiledSize, target.Handle, targetOffset, linearSize, transfers);
        return true;
    }
}
