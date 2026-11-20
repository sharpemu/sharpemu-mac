// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.Gpu.Pipelines;

// What a buffer a linked callee reads through its call record needs at a dispatch (ShaderProgramCache.ClassifyRecordBuffer),
// as a public contract for a backend that prepares call records outside this assembly (the Metal backend).
public static class CallRecordBufferPolicy
{
    // The buffer is made resident as a device-address range.
    public static bool Resident(ulong address, ulong size, bool written) =>
        ShaderProgramCache.ClassifyRecordBuffer(address, size, written) == ShaderProgramCache.RecordBuffer.Resident;

    // The words describe no buffer and the callee writes through them: the record is left out (its call returns at once).
    public static bool LeavesRecordOut(ulong address, ulong size, bool written) =>
        ShaderProgramCache.ClassifyRecordBuffer(address, size, written) == ShaderProgramCache.RecordBuffer.LeaveRecordOut;
}
