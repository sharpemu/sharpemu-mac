// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Core.Cpu.Native;

// The engine's side of thread stack bounds. Where a platform keeps per-thread
// bounds in a thread block that emitted code can swap, the entry stubs tell the
// runtime which stack guest code is using; where it does not, every emitter
// here produces nothing and the fault bridge moves to the host stack instead.
// The native layout of that thread block, and the emitted instructions that
// touch it, belong to the per-OS execution integration (W10).
public sealed unsafe partial class DirectExecutionBackend
{
    // The host's stack pointer, its two stack bounds, then its MXCSR and x87 control word.
    private const int HostStackStateBytes = 4 * sizeof(ulong);
    private const byte HostMxcsrOffset = 3 * sizeof(ulong);
    private const byte HostFpuControlWordOffset = HostMxcsrOffset + sizeof(uint);

    /// <summary>
    /// Bounds of the host mapping holding <paramref name="stackPointer"/>.
    /// Asks the platform first — it may be able to report a thread stack's
    /// whole reservation rather than one committed region of it — then falls
    /// back to the neutral host region table, which every platform answers.
    /// </summary>
    internal static bool TryGetGuestStackBounds(ulong stackPointer, out ulong bottom, out ulong top)
    {
        if (ExecutionPlatform.TryGetCommittedStackRegion(stackPointer, out bottom, out top))
        {
            return true;
        }

        bottom = top = 0;
        var queried = VirtualQuery((void*)stackPointer, out var region, (nuint)sizeof(MEMORY_BASIC_INFORMATION64));
        if (queried == 0 ||
            region.State != 0x1000 || region.RegionSize == 0 || region.RegionSize > ulong.MaxValue - region.BaseAddress)
        {
            return false;
        }

        bottom = region.BaseAddress;
        top = region.BaseAddress + region.RegionSize;
        return stackPointer >= bottom && stackPointer < top;
    }

    private static void EmitFiberStackBounds(byte* code, ref int offset) =>
        ExecutionPlatform.EmitFiberStackBounds(code, ref offset);

    private static void EmitHostStackBounds(byte* code, ref int offset, bool save) =>
        ExecutionPlatform.EmitHostStackBounds(code, ref offset, save);

    private static void EmitGuestStackBounds(byte* code, ref int offset, ulong stackPointer)
    {
        if (!SupportsSwappableStackBounds) return;
        if (!TryGetGuestStackBounds(stackPointer, out var bottom, out var top))
            throw new InvalidOperationException("Guest stack has no committed host mapping.");
        ExecutionPlatform.EmitGuestStackBounds(code, ref offset, bottom, top);
    }

    private static void EmitSavedStackBounds(byte* code, ref int offset, bool restore) =>
        ExecutionPlatform.EmitSavedStackBounds(code, ref offset, restore);

    // Test hook: the bytes the fiber stack-bounds sync emits on this host.
    internal static byte[] EmitFiberStackBoundsForTest()
    {
        var buffer = new byte[64];
        var offset = 0;
        fixed (byte* code = buffer)
        {
            EmitFiberStackBounds(code, ref offset);
        }

        return buffer[..offset];
    }
}
