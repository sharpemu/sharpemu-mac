// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Tests.Gpu.Images;
using Xunit;
using static SharpEmu.Libs.Tests.Gpu.Rendering.RenderExecutorFixtures;

namespace SharpEmu.Libs.Tests.Gpu.Rendering;

// TargetInputs must hold everything the colour and depth target resolvers read: a draw whose inputs match keeps the
// resolved targets and their version (RenderExecutor.ResolveTargetLayout), so a register that moved the resolved
// targets without moving the inputs would hand a draw the previous draw's attachments.
public sealed class DrawResolutionInputsTests
{
    private static readonly AcceptingFormatSupport Formats = new();

    private static Exception Fatal(string message) => new RenderExecutorFatalException(message);

    [Theory]
    [InlineData(0xFFFF_FFFFu)]
    [InlineData(0x1234_5678u)]
    [InlineData(0x0000_0001u)]
    public void EveryContextRegisterThatLeavesTheInputsAloneLeavesTheResolvedTargetsAlone(uint value)
    {
        var unchanged = 0;
        for (var offset = 0u; offset < RegisterBankLayout.ContextRegisterCount; offset++)
        {
            var banks = Banks(withDepth: true);
            banks.Context.ColorTargets[1] = RegisterWords.Color(ColorBase + 0x10_0000, 64, 64);
            banks.Context.RenderTargetMask = 0xFF;
            var before = Resolve(banks.Context);
            var inputs = default(TargetInputs);
            inputs.Capture(banks.Context);
            try
            {
                RegisterWriteTable.WriteContextEntry(banks, offset, value, tableAddress: 0);
            }
            catch (RenderExecutorFatalException)
            {
                continue;
            }

            if (!inputs.Matches(banks.Context))
            {
                continue;
            }

            unchanged++;
            var after = Resolve(banks.Context);
            Assert.True(Same(before, after), $"Context register 0x{offset:X3} = 0x{value:X8} moved the resolved targets but not their inputs.");
        }

        // Most of the bank is not a target register; the loop must have tested those.
        Assert.True(unchanged > 0x300, $"Only {unchanged} registers left the inputs alone.");
    }

    [Fact]
    public void InputsTellANegativeZeroClearFromAPositiveOne()
    {
        var banks = Banks(withDepth: true);
        var inputs = default(TargetInputs);
        inputs.Capture(banks.Context);
        Assert.True(inputs.Matches(banks.Context));
        banks.Context.DepthClearValue = -0f;
        Assert.False(inputs.Matches(banks.Context));
    }

    [Fact]
    public void UncapturedInputsMatchNothing()
    {
        Assert.False(default(TargetInputs).Matches(Banks().Context));
    }

    private static (int Count, ColorTargetResolution[] Colors, uint[] Slots, DepthTargetState? Depth) Resolve(ContextRegisters context)
    {
        var colors = new ColorTargetResolution[RenderingState.ColorAttachmentCapacity];
        var slots = new uint[RenderingState.ColorAttachmentCapacity];
        DepthTargetState? depth;
        int count;
        try
        {
            count = RenderExecutor.ResolveColorTargets(context, colors, slots);
            depth = DepthTargetResolver.Resolve(context, Formats, Fatal);
        }
        catch (Exception exception) when (exception is RenderExecutorFatalException or ArgumentException or InvalidOperationException)
        {
            return (-1, colors, slots, null);
        }

        return (count, colors, slots, depth);
    }

    private static bool Same(
        (int Count, ColorTargetResolution[] Colors, uint[] Slots, DepthTargetState? Depth) left,
        (int Count, ColorTargetResolution[] Colors, uint[] Slots, DepthTargetState? Depth) right)
    {
        if (left.Count != right.Count || !RenderExecutor.SameDepth(left.Depth, right.Depth))
        {
            return false;
        }

        for (var index = 0; index < left.Count; index++)
        {
            if (left.Slots[index] != right.Slots[index] || !RenderExecutor.SameColor(in left.Colors[index], in right.Colors[index]))
            {
                return false;
            }
        }

        return true;
    }
}
