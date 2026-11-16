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
    public static int ExportForSlot(ShaderInterfaceRegisters registers, uint slot) =>
        slot >= TargetCount ? -1 : Routing.Of(registers).ExportForSlot(slot);

    /// <summary>
    /// The pairing of exports and slots the registers make, counted once for every slot a draw asks about
    /// (the targets' unwritten tests ask for each bound slot, several times a draw).
    /// </summary>
    internal readonly struct Routing
    {
        // Bit n of each mask: export n has a format, slot n is enabled.
        private readonly uint _exports;
        private readonly uint _slots;

        private Routing(uint exports, uint slots)
        {
            _exports = exports;
            _slots = slots;
        }

        public static Routing Of(ShaderInterfaceRegisters registers) =>
            new(ExportMask(registers.TargetOutputModes), EnabledSlots(registers.ColorShaderMask));

        public int ExportForSlot(uint slot)
        {
            if (slot >= TargetCount)
            {
                return -1;
            }

            var exports = _exports;
            var slots = _slots;
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
    }

    // Bit n set when the first eight output modes give export n a format (a nonzero low nibble).
    private static uint ExportMask(byte[] modes)
    {
        if (modes.Length >= TargetCount)
        {
            // All eight at once: each byte's low nibble folded into its bit 0, the eight bits gathered into the top byte.
            var nibbles = System.Buffers.Binary.BinaryPrimitives.ReadUInt64LittleEndian(modes) & 0x0F0F0F0F0F0F0F0FUL;
            nibbles |= nibbles >> 2;
            nibbles |= nibbles >> 1;
            nibbles &= 0x0101010101010101UL;
            return (uint)((nibbles * 0x0102040810204080UL) >> 56);
        }

        var exports = 0u;
        for (var index = 0; index < modes.Length; index++)
        {
            if ((modes[index] & 0xF) != 0)
            {
                exports |= 1u << index;
            }
        }

        return exports;
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
