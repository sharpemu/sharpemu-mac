// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Numerics;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;

namespace SharpEmu.Libs.Gpu.Pipelines;

// Color exports reach the targets in order: the Nth export that SPI_SHADER_COL_FORMAT gives a
// format goes to the Nth target that CB_SHADER_MASK enables. Compilers drop MRTs a pixel program
// does not write, so the export number and the target slot can differ. Unreal's base pass, for
// example, exports MRT0 and MRT1 to targets 0 and 4 with CB_SHADER_MASK=0x000F000F.
internal static class PixelExportRouting
{
    private const int TargetCount = 8;

    // The export that feeds the slot, or -1 when no export reaches it. When the registers do not
    // pair up, each export goes to the slot of the same number.
    public static int ExportForSlot(ShaderInterfaceRegisters registers, uint slot)
    {
        if (slot >= TargetCount)
        {
            return -1;
        }

        // Bit n of each mask: export n has a format, slot n is enabled. Called for every bound slot
        // of every draw from several places, so the pairing is counted in bits rather than listed.
        var modes = registers.TargetOutputModes;
        var exports = 0u;
        for (var index = 0; index < TargetCount && index < modes.Length; index++)
        {
            if ((modes[index] & 0xF) != 0)
            {
                exports |= 1u << index;
            }
        }

        var slots = EnabledSlots(registers.ColorShaderMask);
        if (exports == 0 || BitOperations.PopCount(exports) != BitOperations.PopCount(slots))
        {
            return (int)slot;
        }

        if ((slots & (1u << (int)slot)) == 0)
        {
            return -1;
        }

        // The slot's rank among the enabled slots picks the export of the same rank.
        for (var rank = BitOperations.PopCount(slots & ((1u << (int)slot) - 1)); rank > 0; rank--)
        {
            exports &= exports - 1;
        }

        return BitOperations.TrailingZeroCount(exports);
    }

    // Bit n set when CB_SHADER_MASK enables any component of slot n.
    private static uint EnabledSlots(uint mask)
    {
        var any = mask | (mask >> 1);
        any = (any | (any >> 2)) & 0x11111111u;
        any = (any | (any >> 3)) & 0x03030303u;
        any = (any | (any >> 6)) & 0x000F000Fu;
        return (any | (any >> 12)) & 0xFFu;
    }
}
