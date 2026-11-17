// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using SharpEmu.Core.Cpu.Native;
using Xunit;

namespace SharpEmu.Libs.Tests.Cpu;

public sealed class FaultDumpExpressionTests
{
    private static bool Evaluate(string text, Dictionary<string, ulong> registers, out ulong value)
    {
        var position = 0;
        return DirectExecutionBackend.TryEvaluateFaultExpression(text, ref position, registers, out value) &&
               position == text.Length;
    }

    [Fact]
    public void RegistersAndOffsetsAddAndSubtract()
    {
        var registers = new Dictionary<string, ulong>(StringComparer.OrdinalIgnoreCase) { ["rbp"] = 0x1000, ["r13"] = 3 };

        Assert.True(Evaluate("rbp-0x360+r13", registers, out var value));
        Assert.Equal(0x1000UL - 0x360 + 3, value);
        Assert.False(Evaluate("rbp*2", registers, out _));
        Assert.False(Evaluate("nosuchreg", registers, out _));
    }

    [Fact]
    public unsafe void BracketsLoadAQwordAndNest()
    {
        var block = (ulong*)NativeMemory.AlignedAlloc(32, 16);
        try
        {
            // block[0] -> block[2]; block[2] holds the payload two loads away.
            block[0] = (ulong)(block + 2);
            block[1] = 0;
            block[2] = 0x1122334455667788;
            var registers = new Dictionary<string, ulong>(StringComparer.OrdinalIgnoreCase) { ["rdi"] = (ulong)block + 8 };

            Assert.True(Evaluate("[rdi-8]", registers, out var pointer));
            Assert.Equal((ulong)(block + 2), pointer);
            Assert.True(Evaluate("[[rdi-0x8]]+1", registers, out var payload));
            Assert.Equal(0x1122334455667789UL, payload);
            Assert.False(Evaluate("[rdi-8", registers, out _));
        }
        finally
        {
            NativeMemory.AlignedFree(block);
        }
    }
}
