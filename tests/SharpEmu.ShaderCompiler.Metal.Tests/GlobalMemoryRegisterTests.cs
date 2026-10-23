// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text.RegularExpressions;
using Xunit;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

public sealed class GlobalMemoryRegisterTests
{
    [Theory]
    [InlineData(12u, 1)]
    [InlineData(13u, 2)]
    [InlineData(14u, 4)]
    [InlineData(15u, 3)]
    public void Loads_WriteTheDestinationRegisters(uint opcode, int componentCount)
    {
        var source = Compile(opcode);
        for (var component = 0; component < componentCount; component++)
        {
            // A wide load resolves its span once and reads each dword off it, falling back
            // to the per-dword walk when the span leaves its page (DeviceSpanAccess).
            if (componentCount == 1)
                Assert.Contains($"v{8 + component} = sharpemu_load_device_dword", source);
            else
                Assert.Matches($@"v{8 + component} = t\d+ \? t\d+\[{component}\] : sharpemu_load_device_dword", source);
        }
        Assert.DoesNotMatch(@"(?m)^\s*(if \(exec\) \{ )?v3 = ", source);
    }

    [Theory]
    [InlineData(28u, 1)]
    [InlineData(29u, 2)]
    [InlineData(30u, 4)]
    [InlineData(31u, 3)]
    public void Stores_ReadTheSourceRegisters(uint opcode, int componentCount)
    {
        var source = Compile(opcode);
        for (var component = 0; component < componentCount; component++)
            Assert.Matches($@"sharpemu_store_device_dword\([^\r\n]*, v{3 + component}\);", source);
        Assert.DoesNotContain(", v8,", source);
    }

    [Theory]
    [InlineData(50u, false, "add")]
    [InlineData(50u, true, "add")]
    [InlineData(56u, false, "max")]
    [InlineData(56u, true, "max")]
    public void Atomics_ReadTheSourceAndOnlyWriteRequestedResults(uint opcode, bool returnsValue, string operation)
    {
        var source = Compile(opcode, returnsValue);
        Assert.Contains($"atomic_fetch_{operation}_explicit", source);
        Assert.Contains(", v3, memory_order_relaxed)", source);
        Assert.Equal(returnsValue, Regex.IsMatch(source, @"(?m)^\s*(if \(exec\) \{ )?v8 = "));
        Assert.DoesNotMatch(@"(?m)^\s*(if \(exec\) \{ )?v3 = ", source);
    }

    [Theory]
    [InlineData(0x30u, true, "atomic_exchange_explicit(")]
    [InlineData(0x33u, false, "atomic_fetch_sub_explicit(")]
    [InlineData(0x35u, true, "atomic_fetch_min_explicit((device atomic_int*)")]
    [InlineData(0x36u, false, "atomic_fetch_min_explicit((device atomic_uint*)")]
    [InlineData(0x37u, false, "atomic_fetch_max_explicit((device atomic_int*)")]
    [InlineData(0x39u, false, "atomic_fetch_and_explicit(")]
    [InlineData(0x3Au, true, "atomic_fetch_or_explicit(")]
    [InlineData(0x3Bu, false, "atomic_fetch_xor_explicit(")]
    [InlineData(0x31u, true, "sharpemu_atomic_compare_swap(")]
    [InlineData(0x3Cu, true, "sharpemu_atomic_step(")]
    [InlineData(0x3Du, false, "sharpemu_atomic_step(")]
    [InlineData(0x3Fu, true, "sharpemu_atomic_float_minmax(")]
    [InlineData(0x40u, false, "sharpemu_atomic_float_minmax(")]
    public void Atomics32_UseTheMatchingDeviceAtomic(uint opcode, bool returnsValue, string operation)
    {
        var source = CompileOnDevice(opcode, returnsValue);
        Assert.Contains(operation, source);
        Assert.Equal(returnsValue, Regex.IsMatch(source, @"(?m)^\s*(if \(exec\) \{ )?v8 = "));
    }

    [Theory]
    // Every X2 form is one read-modify-write of the qword under the pair lock; the unsigned
    // min/max compute on the loaded ulong instead of a native atomic_ulong (which returns nothing).
    [InlineData(0x58u, "max(")]
    [InlineData(0x56u, "min(")]
    public void Atomics64_UnsignedMinMaxComputeTheQwordUnderThePairLock(uint opcode, string operation)
    {
        var source = CompileOnDevice(opcode, returnsValue: false);
        Assert.Contains(operation, source);
        Assert.Contains("sharpemu_pair_try_lock(", source);
        // v[3:4] is the 64-bit operand, low dword first.
        Assert.Matches(@"\(ulong\)v3 \| \(\(ulong\)v4 << 32\)", source);
        Assert.DoesNotMatch(@"(?m)^\s*(if \(exec\) \{ )?v[89] = ", source);
    }

    [Theory]
    // Bitwise forms apply per dword; add/sub fold the low dword's carry or borrow into the high one.
    [InlineData(0x52u, "atomic_fetch_add_explicit(")]
    [InlineData(0x53u, "atomic_fetch_sub_explicit(")]
    [InlineData(0x5Au, "atomic_fetch_or_explicit(")]
    public void Atomics64_NonReturningSplitFormsTouchBothDwords(uint opcode, string operation)
    {
        var source = CompileOnDevice(opcode, returnsValue: false);
        Assert.Equal(2, Regex.Matches(source, Regex.Escape(operation) + @"\(device atomic_uint\*\)\(t\d+( \+ 1)?\)").Count);
        Assert.Matches(Regex.Escape(operation) + @"\(device atomic_uint\*\)\(t\d+ \+ 1\)", source);
    }

    [Theory]
    [InlineData(32u, false)]
    [InlineData(33u, true)]
    [InlineData(34u, false)]
    [InlineData(35u, true)]
    [InlineData(36u, false)]
    [InlineData(37u, true)]
    public void PartialLoads_PreserveTheOtherHalfOfTheDestination(uint opcode, bool highHalf)
    {
        var source = Compile(opcode);
        Assert.Contains(highHalf ? "v8 & 0x0000FFFFu" : "v8 & 0xFFFF0000u", source);
        Assert.DoesNotMatch(@"\bv3\b", source);
    }

    [Fact]
    public void AddressRangeLookup_CoversTheWholeTable()
    {
        var source = Compile(12u);
        Assert.Contains($"min(sharpemu_resources.address_range_count & 0x7FFFFFFFu, {Gen5MslTranslator.MaxAddressRangeCount}u)", source);
        Assert.Contains($"step < {Gen5MslTranslator.AddressRangeSearchSteps}u", source);
        Assert.Contains("min(sharpemu_resources.address_range_count & 0x7FFFFFFFu, 16384u)", source);
        Assert.Contains("step < 15u", source);
    }

    [Fact]
    public void AddressRangeSearchSteps_CoverTheMaximumCount()
        => Assert.True((1UL << (int)Gen5MslTranslator.AddressRangeSearchSteps) > Gen5MslTranslator.MaxAddressRangeCount);

    private static uint Word(uint opcode, bool returnsValue) =>
        0xDC00_8000u | (opcode << 18) | (returnsValue ? 1u << 16 : 0);

    private static string Compile(uint opcode, bool returnsValue = false)
    {
        var fixture = new Gen5ComputeFixture("global-registers", [Word(opcode, returnsValue), 0x0810_0309, 0xBF81_0000], 16, 64);
        return Gen5ComputeFixtures.CompileRequestOrThrow(fixture).Source;
    }

    // Where a device exists, the emitted source must also be valid Metal.
    private static string CompileOnDevice(uint opcode, bool returnsValue)
    {
        var source = Compile(opcode, returnsValue);
        if (MetalNative.IsAvailable)
            Assert.True(MetalNative.TryCompileLibrary(source, out _, out var error), error);
        return source;
    }
}
