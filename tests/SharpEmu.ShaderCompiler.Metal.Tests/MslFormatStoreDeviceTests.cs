// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

// Formatted stores executed on the device, the Metal twins of upstream's DeviceFormatStoreDeviceTests and
// BufferStoreFormatD16HiDeviceTests (7cf0ac568). A store through a V# in scalar registers (read from guest memory,
// so its format is only known when the store runs) encodes with that V#'s own format: 8, 8_8 and 8_8_8_8 UNORM,
// 8 SNORM, 16 FLOAT and 32 FLOAT each write their own bytes and leave the rest of the dword alone. A
// BUFFER_STORE_FORMAT_D16_HI_X stores the high half of its register, through a bound V# and a runtime one alike.
public sealed class MslFormatStoreDeviceTests(ITestOutputHelper output)
{
    private const int Lanes = 4;
    private const int BoundBytes = 2048;
    private const int ValueOffset = 1024;
    private const ulong GuestBase = 0x4000_0000;
    private const int GuestBytes = 1 << 14;
    private const int DeviceBufferOffset = 0x1000;
    private const uint Identity = 4 | (5 << 3) | (6 << 6) | (7 << 9);
    private static readonly uint[] Values = [0x3F000000, 0x3F800000, 0xBF800000, 0x7FC00000];

    [Theory]
    [InlineData(1u, new uint[] { 0x80, 0xFF, 0x00, 0x00 }, 1)]
    [InlineData(14u, new uint[] { 0x80, 0xFF, 0x00, 0x00 }, 1)]
    [InlineData(56u, new uint[] { 0x80, 0xFF, 0x00, 0x00 }, 1)]
    [InlineData(2u, new uint[] { 0x40, 0x7F, 0x81, 0x00 }, 1)]
    [InlineData(13u, new uint[] { 0x3800, 0x3C00, 0xBC00 }, 2)]
    [InlineData(22u, new uint[] { 0x3F000000, 0x3F800000, 0xBF800000, 0x7FC00000 }, 4)]
    public void FormatsStoreTheirOwnEncodingThroughARuntimeDescriptor(uint format, uint[] expected, int elementBytes)
    {
        // buffer_store_format_x v4, v1, s[12:15], 0 offen
        var stored = RunDevice(0xE0101000, format, Values);
        if (stored is null) return;
        var elementMask = elementBytes == 4 ? uint.MaxValue : (1u << (elementBytes * 8)) - 1;
        for (var lane = 0; lane < expected.Length; lane++)
        {
            var dword = stored[lane];
            Assert.True(expected[lane] == (dword & elementMask), $"format {format} lane {lane}: 0x{dword:X8}");
            Assert.Equal(0xCCCCCCCCu & ~elementMask, dword & ~elementMask);
        }
    }

    // 16-bit UINT, SINT and FLOAT.
    public static TheoryData<uint, bool> D16HiCases()
    {
        var cases = new TheoryData<uint, bool>();
        for (uint format = 1; format < 128; format++)
        {
            if (Gfx10UnifiedFormat.TryDecode(format, out var dataFormat, out var numberFormat) &&
                dataFormat == 2 && numberFormat is 4 or 5 or 7)
            {
                cases.Add(format, false);
                cases.Add(format, true);
            }
        }

        return cases;
    }

    // The high half is 1.0 as a half float and a plain 16-bit integer otherwise; the low half is not stored, nor is
    // the dword's other half touched.
    [Theory]
    [MemberData(nameof(D16HiCases))]
    public void AD16HiStoreWritesTheHighHalfOfTheRegister(uint format, bool runtimeDescriptor)
    {
        const uint value = 0x3C00_1234u;
        var values = Enumerable.Repeat(value, Lanes).ToArray();
        // buffer_store_format_d16_hi_x v4, v1, s[...], 0 offen
        var stored = runtimeDescriptor ? RunDevice(0xE09C1000, format, values) : RunBound(format, values);
        if (stored is null) return;
        for (var lane = 0; lane < Lanes; lane++)
        {
            Assert.True(stored[lane] == (0xCCCC0000u | (value >> 16)), $"format {format} lane {lane}: 0x{stored[lane]:X8}");
        }
    }

    // v1 = lane * 4; v4 = the lane's value from the bound buffer at 1024; then the store under test at v1.
    private static uint[] Words(uint storeWord, uint storeResource, uint offset) =>
    [
        0x34020082,                             // v_lshlrev_b32 v1, 2, v0
        0xE0301400, 0x80020401,                 // buffer_load_dword v4, v1, s[8:11], 0 offen offset:1024
        0xBF8C3F70,                             // s_waitcnt vmcnt(0)
        storeWord | offset, 0x80000401 | (storeResource / 4) << 16, // the store of v4 at v1 (+ offset)
        0xBF810000,                             // s_endpgm
    ];

    private uint[]? RunBound(uint format, uint[] values)
    {
        var userData = new uint[16];
        userData[8] = 0x200000;
        userData[10] = BoundBytes;
        userData[11] = Identity | (format << 12);
        var fixture = new Gen5ComputeFixture("bound-format-store", Words(0xE09C1000, 8, 512), 8, BoundBytes);
        var shader = Compile(fixture, userData);
        if (!RequireDevice()) return null;

        var bound = Bound(values);
        var result = Execute(shader, bound, tables: null);
        return Enumerable.Range(0, Lanes).Select(lane => BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan(512 + lane * 4))).ToArray();
    }

    // s[12:15] = the V# at guest address 0 (the bound buffer's last dword names the guest base), OOB_SELECT 3.
    private uint[]? RunDevice(uint storeWord, uint format, uint[] values)
    {
        uint[] words =
        [
            0xE03003FC, 0x80020300,             // buffer_load_dword v3, off, s[8:11], 0 offset:1020
            0xBF8C3F70,                         // s_waitcnt vmcnt(0)
            0x7E080503,                         // v_readfirstlane_b32 s4, v3
            0xBE850380,                         // s_mov_b32 s5, 0
            0xF4080302, 0xFA000000,             // s_load_dwordx4 s[12:15], s[4:5], 0x0
            0xBF8CC07F,                         // s_waitcnt lgkmcnt(0)
            .. Words(storeWord, 12, 0),
        ];
        var userData = new uint[16];
        userData[8] = 0x200000;
        userData[10] = BoundBytes;
        userData[11] = Identity | (20u << 12);
        var shader = Compile(new Gen5ComputeFixture("device-format-store", words, 8, BoundBytes), userData);
        if (!RequireDevice()) return null;

        var guest = Enumerable.Repeat((byte)0xCC, GuestBytes).ToArray();
        var descriptor = new uint[] { (uint)(GuestBase + DeviceBufferOffset), 0, 0x1000, Identity | (format << 12) | (3u << 28) };
        for (var word = 0; word < descriptor.Length; word++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(guest.AsSpan(word * 4), descriptor[word]);
        }

        var bound = Bound(values);
        BinaryPrimitives.WriteUInt32LittleEndian(bound.AsSpan(1020), (uint)GuestBase);
        var (tables, _) = MetalNative.CreateGuestMapping(GuestBase, guest, 0x1000);
        Execute(shader, bound, tables);
        return MetalNative.ReadWords(tables[0].Residents[0], (DeviceBufferOffset / 4) + Lanes)[(DeviceBufferOffset / 4)..];
    }

    private static byte[] Bound(uint[] values)
    {
        var bound = Enumerable.Repeat((byte)0xCC, BoundBytes).ToArray();
        for (var lane = 0; lane < values.Length; lane++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(bound.AsSpan(ValueOffset + lane * 4), values[lane]);
        }

        return bound;
    }

    private static Gen5MslShader Compile(Gen5ComputeFixture fixture, uint[] userData)
    {
        var request = Gen5ComputeFixtures.RequestOrThrow(Gen5ComputeFixtures.DecodeOrThrow(fixture), ShaderStage.Compute,
            localSizeX: Lanes, runtimeInputs: new ResourceRuntimeInputs { UserData = userData });
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        return shader!;
    }

    private static byte[] Execute(Gen5MslShader shader, byte[] initial, IReadOnlyList<DeviceTable>? tables)
    {
        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var compileError), compileError);
        try
        {
            Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, initial,
                new byte[PushData.ByteSize], shader.ArgumentLayout!, Lanes, out var result, out _, out var runError,
                tables: tables), runError);
            return result;
        }
        finally
        {
            MetalNative.ReleaseCollisionLibrary(library);
        }
    }

    private bool RequireDevice()
    {
        if (MetalNative.IsAvailable) return true;
        Assert.False(
            OperatingSystem.IsMacOS() && Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") == "1",
            "The required gate cannot run the format store test without a Metal device.");
        output.WriteLine("No Metal device on this host; the format store test skipped.");
        return false;
    }
}
