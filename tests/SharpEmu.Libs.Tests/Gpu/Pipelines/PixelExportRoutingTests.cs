// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Pipelines;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.Pipelines;

public sealed class PixelExportRoutingTests
{
    // The routing as it was written first: both lists built, then the slot's position looked up.
    private static int Listed(ShaderInterfaceRegisters registers, uint slot)
    {
        if (slot >= 8)
        {
            return -1;
        }

        var exports = new List<int>();
        var slots = new List<int>();
        for (var index = 0; index < 8; index++)
        {
            if (index < registers.TargetOutputModes.Length && (registers.TargetOutputModes[index] & 0xF) != 0)
            {
                exports.Add(index);
            }

            if (((registers.ColorShaderMask >> (index * 4)) & 0xFu) != 0)
            {
                slots.Add(index);
            }
        }

        if (exports.Count == 0 || exports.Count != slots.Count)
        {
            return (int)slot;
        }

        var position = slots.IndexOf((int)slot);
        return position < 0 ? -1 : exports[position];
    }

    [Fact]
    public void EverySlotRoutesAsTheListedPairingDoes()
    {
        var random = new System.Random(1234);
        var registers = new ShaderInterfaceRegisters();
        for (var trial = 0; trial < 20000; trial++)
        {
            // Sparse masks pair up often enough to reach the ranked lookup.
            var formats = (uint)random.Next(256) & (uint)random.Next(256);
            for (var index = 0; index < 8; index++)
            {
                registers.TargetOutputModes[index] = ((formats >> index) & 1) != 0 ? (byte)(1 + random.Next(15)) : (byte)(random.Next(2) << 4);
            }

            var enabled = (uint)random.Next(256) & (uint)random.Next(256);
            var mask = 0u;
            for (var index = 0; index < 8; index++)
            {
                if (((enabled >> index) & 1) != 0)
                {
                    mask |= (uint)(1 + random.Next(15)) << (index * 4);
                }
            }

            registers.ColorShaderMask = mask;
            for (var slot = 0u; slot < 10; slot++)
            {
                Assert.Equal(Listed(registers, slot), PixelExportRouting.ExportForSlot(registers, slot));
            }
        }
    }

    [Fact]
    public void ExportsCompactOntoTheEnabledSlots()
    {
        // Unreal's base pass: MRT0 and MRT1 to targets 0 and 4.
        var registers = new ShaderInterfaceRegisters { ColorShaderMask = 0x000F000F };
        registers.TargetOutputModes[0] = 9;
        registers.TargetOutputModes[1] = 9;
        Assert.Equal(0, PixelExportRouting.ExportForSlot(registers, 0));
        Assert.Equal(1, PixelExportRouting.ExportForSlot(registers, 4));
        Assert.Equal(-1, PixelExportRouting.ExportForSlot(registers, 1));
    }

    [Fact]
    public void AShortOutputModeArrayCountsOnlyItsEntries()
    {
        var registers = new ShaderInterfaceRegisters { ColorShaderMask = 0xF0, TargetOutputModes = [9] };
        Assert.Equal(0, PixelExportRouting.ExportForSlot(registers, 1));
        Assert.Equal(-1, PixelExportRouting.ExportForSlot(registers, 0));
    }
}
