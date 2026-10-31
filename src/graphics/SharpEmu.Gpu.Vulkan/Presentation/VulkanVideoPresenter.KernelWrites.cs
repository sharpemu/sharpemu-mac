// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Vulkan.Presentation;

using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.HLE.Diagnostics;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.Libs.VideoOut;
using SharpEmu.Gpu.Vulkan;
using SharpEmu.Gpu.Vulkan.Buffers;
using SharpEmu.Gpu.Vulkan.Images;
using SharpEmu.Gpu.Vulkan.Pipelines;
using SharpEmu.Gpu.Vulkan.Scheduling;

internal static partial class VulkanVideoPresenter
{
    private sealed partial class Presenter : IKernelWriteStream
    {
        public KernelWriteRefusal TryPostKernelWrite(ulong address, ReadOnlySpan<uint> words) => KernelWritePolicy.TryPost(this, address, words);

        bool IKernelWriteStream.OnStreamThread => OnCommandThread && !OnResolveWorker;

        bool IKernelWriteStream.InCrossQueueOrderWindow => InAnyCrossQueueOrderWindow();

        bool IKernelWriteStream.HasPendingPublication(ulong address, ulong size) => HasPendingMemoryPublication(address, size);

        bool IKernelWriteStream.HasPendingRange(ulong address, ulong size) => _commandThread!.HasPendingRange(address, size);

        bool IKernelWriteStream.HasPendingGpuWrite(ulong address, ulong size) => IsPendingOnCommandThread(address, size);

        bool IKernelWriteStream.HasImageBytes(ulong address, ulong size) => _bufferCache.MayHaveImageBytes(address, size);

        bool IKernelWriteStream.HasSurfaceMetadata(ulong address, ulong size) => _imageCache.MayOverlapMetadata(address, size);

        void IKernelWriteStream.PostWrite(ulong address, ReadOnlySpan<byte> data) => _commandThread!.PostWrite(address, data, kernel: true);

        // On the render thread at the store's stream position: a store into GPU-dirty pages lands on both copies.
        internal bool TryApplyKernelWriteOverGpuBytes(ulong address, byte[] data)
        {
            var size = (ulong)data.Length;
            if (!_bufferCache.MayHaveGpuDirtyPages(address, size) || !_bufferCache.HasGpuDirtyPages(address, size) ||
                !_bufferCache.TryWriteOverGpuBytes(address, data))
            {
                ShaderPipelineCache.KernelWritesToCpu.Increment();
                return false;
            }

            ShaderPipelineCache.KernelWritesOverGpu.Increment();
            ShaderPipelineCache.KernelWriteBytesOverGpu.Add(data.Length);
            return true;
        }
    }
}
