// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.Gpu.GpuCommands;

/// <summary>Optional, passive census of the interpreter's memory observations.</summary>
public interface ICommandBatchObserver
{
    void BeforePacketRead();
    void ObservePacket(uint header);
}
