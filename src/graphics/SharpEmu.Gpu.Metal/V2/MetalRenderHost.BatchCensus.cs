// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.GpuCommands;

namespace SharpEmu.Gpu.Metal.V2;

internal sealed partial class MetalRenderHost : ICommandBatchObserver
{
    private readonly MetalBatchCensus? _batchCensus =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_BATCH_CENSUS") == "1" ? new() : null;

    void ICommandBatchObserver.BeforePacketRead() => _batchCensus?.Strict.Close(MetalBatchCensus.Cut.CommandRead);
    void ICommandBatchObserver.ObservePacket(uint header) => _batchCensus?.Packet(header);
}
