// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Xunit;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

// sharpemu_load_dword_run<N> / sharpemu_store_dword_run<N> are the raw BUFFER_LOAD/STORE_
// DWORDX2..X4 fast path: one bounds proof for the run, the per-dword check when any dword
// of it is out of range. They have to behave exactly like N sharpemu_load_dword /
// sharpemu_store_dword calls for every address, including a run that straddles the end
// of the binding (partly valid), an address whose 32-bit arithmetic wraps, and a binding
// that is only 4-byte aligned (the kernel binds at b + 1).
public sealed class MslDwordRunTests
{
    private const int DataWords = 16;
    private const int CaseBase = 32;
    private const int OutputBase = 64;
    private const int OutputWords = 48;
    private const int StoreBase = 128;
    private const int StoreStride = 40;
    private const uint Marker = 0xEE00_0000u;

    private const string Kernel = @"
kernel void dword_run_test(device uint* b [[buffer(0)]], uint t [[thread_position_in_threadgroup]])
{
    uint addr = b[32];
    uint bytes = b[33];
    device uint* data = b + 1;
    uint4 four = sharpemu_load_dword_run<4>(data, bytes, addr);
    uint4 three = sharpemu_load_dword_run<3>(data, bytes, addr);
    uint4 two = sharpemu_load_dword_run<2>(data, bytes, addr);
    for (uint i = 0u; i < 4u; i++)
    {
        b[64u + i] = four[i];
        b[68u + i] = sharpemu_load_dword(data, bytes, addr + 4u * i);
        b[72u + i] = i < 3u ? three[i] : 0u;
        b[76u + i] = i < 3u ? sharpemu_load_dword(data, bytes, addr + 4u * i) : 0u;
        b[80u + i] = i < 2u ? two[i] : 0u;
        b[84u + i] = i < 2u ? sharpemu_load_dword(data, bytes, addr + 4u * i) : 0u;
    }
    for (uint n = 2u; n <= 4u; n++)
    {
        device uint* run = b + 128u + (n - 2u) * 40u + 1u;
        device uint* each = run + 20u;
        for (uint i = 0u; i < 16u; i++)
        {
            run[i] = 0xEE000000u + i;
            each[i] = 0xEE000000u + i;
        }
        uint4 value = uint4(0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u);
        if (n == 2u) { sharpemu_store_dword_run<2>(run, bytes, addr, value); }
        if (n == 3u) { sharpemu_store_dword_run<3>(run, bytes, addr, value); }
        if (n == 4u) { sharpemu_store_dword_run<4>(run, bytes, addr, value); }
        for (uint i = 0u; i < n; i++)
        {
            sharpemu_store_dword(each, bytes, addr + 4u * i, value[i]);
        }
    }
}
";

    public static TheoryData<uint, uint> Cases => new()
    {
        { 0u, 64u },          // whole run inside
        { 16u, 64u },
        { 48u, 64u },         // the last four dwords
        { 52u, 64u },         // x4 and x3 run past the end, x2 fits
        { 56u, 64u },         // only x2 fits
        { 60u, 64u },         // one dword inside
        { 64u, 64u },         // wholly past the end
        { 2u, 64u },          // an unaligned address truncates to its dword
        { 7u, 64u },
        { 0u, 0u },           // an empty binding
        { 8u, 12u },          // a binding that is not a multiple of a run
        { 10u, 14u },         // unaligned, partly valid
        { 0u, 6u },           // bytes that are not a whole dword count
        { 0xFFFF_FFFCu, 64u },  // the first dword is past the end, the second wraps to dword 0
        { 0xFFFF_FFF8u, 64u },  // two dwords past the end, the third and fourth wrap to dwords 0 and 1
        { 0xFFFF_FFFFu, 64u },
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public void ARunMatchesPerDwordAccesses(uint address, uint bytes)
    {
        if (!MetalNative.IsAvailable)
        {
            Assert.False(
                OperatingSystem.IsMacOS() && Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") == "1",
                "The required gate cannot run the dword-run test without a Metal device.");
            return;
        }

        var prelude = Gen5NggFixtures.CompileOrThrow(Gen5NggFixtures.CullingTriangle()).Source;
        Assert.True(MetalNative.TryCompileLibrary(prelude + Kernel, out var library, out var compileError), compileError);

        var words = new uint[StoreBase + 3 * StoreStride + 8];
        for (var index = 0; index < DataWords; index++)
        {
            words[1 + index] = 0x01010101u * (uint)(index + 1) ^ 0xA5A5_0000u;
        }

        words[CaseBase] = address;
        words[CaseBase + 1] = bytes;
        var contents = new byte[words.Length * sizeof(uint)];
        Buffer.BlockCopy(words, 0, contents, 0, contents.Length);
        Assert.True(MetalNative.TryExecuteOnBuffer(library, "dword_run_test", contents, 1, out var result, out var error), error);

        var output = new uint[result.Length / sizeof(uint)];
        Buffer.BlockCopy(result, 0, output, 0, result.Length);
        for (var index = 0; index < 4; index++)
        {
            Assert.Equal(output[OutputBase + 4 + index], output[OutputBase + index]);
            Assert.Equal(output[OutputBase + 12 + index], output[OutputBase + 8 + index]);
            Assert.Equal(output[OutputBase + 20 + index], output[OutputBase + 16 + index]);
        }

        for (var width = 0; width < 3; width++)
        {
            var run = StoreBase + width * StoreStride + 1;
            for (var index = 0; index < 16; index++)
            {
                Assert.Equal(output[run + 20 + index], output[run + index]);
            }
        }

        // A run wholly inside the binding and the data reads the bytes that sit there.
        if ((address & 3) == 0 && address <= bytes && bytes - address >= 16 && address + 16 <= DataWords * 4)
        {
            for (var index = 0; index < 4; index++)
            {
                Assert.Equal(words[1 + address / 4 + index], output[OutputBase + index]);
            }
        }
        else if (address == 64u && bytes == 64u)
        {
            Assert.All(output.AsSpan(OutputBase, 4).ToArray(), value => Assert.Equal(0u, value));
        }
    }
}
