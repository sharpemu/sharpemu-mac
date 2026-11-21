// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

// A formatted load reads component c from byte (elementAddress + componentOffset) in 32 bits, and
// only the components a destination selects are bounds checked, each on its own. With the element
// at 2^32 - 4 (plus 0-3 bytes), Y, Z and W wrap to the buffer's first bytes; with only those
// selected the load reads them, aligned or not. The Metal twin of BufferFormatWrapDeviceTests
// (upstream 7cf0ac568). The run-time word 3 paths (fetch, decoder, inline, GLC, function
// constant) are in MslRuntimeFormatLoadTests.Wrap.cs.
public sealed class MslFormatLoadWrapTests(ITestOutputHelper output)
{
    private const int BufferBytes = 64;
    private const uint Uint32x4 = 75;
    private const uint Uint16x2 = 27;
    // x = Y, y = Z, z = W, w = 0.
    private const uint YzwZero = 5 | (6 << 3) | (7 << 6);
    private const ulong GuestBase = 0x4000_0000;
    private const int GuestBytes = 1 << 14;
    private const int DeviceBufferOffset = 0x1000;
    private const int BoundBytes = 1024;

    // The descriptor word 3 is a literal (the bound path's specialized word): a format of whole
    // dwords loads its selected components as one run (TryEmitLiteralDwordFormatLoad).
    [Theory]
    [InlineData(0u)]
    [InlineData(1u)]
    [InlineData(3u)]
    public void ComponentsPastTheTopOfTheByteSpaceWrapToTheBuffersStart(uint misalignment)
    {
        var shader = CompileLiteral(0xFFFF_FFFCu + misalignment, (Uint32x4 << 12) | YzwZero, components: 4);
        if (!RequireDevice()) return;

        var bytes = Pattern(BufferBytes);
        var result = Run(shader, bytes);
        uint Bytes(uint first) => BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan((int)first));
        Assert.Equal([Bytes(misalignment), Bytes(misalignment + 4), Bytes(misalignment + 8), 0u], Words(result, 0, 4));
    }

    // An X component out of range still reads every destination as its constant when it is
    // selected, wrapped neighbours or not: the selected components are all or nothing.
    [Fact]
    public void ASelectedComponentOutOfRangeZeroesTheWrappedOnes()
    {
        var shader = CompileLiteral(0xFFFF_FFFCu, (Uint32x4 << 12) | DescriptorConstants.IdentityDestinationSelect, components: 4);
        if (!RequireDevice()) return;

        Assert.Equal([0u, 0u, 0u, 0u], Words(Run(shader, Pattern(BufferBytes)), 0, 4));
    }

    // A packed format with a literal word 3 (BeginLiteralPackedFormatLoad): the Y half of a 16_16
    // element two bytes below 2^32 is the buffer's first two bytes.
    [Fact]
    public void APackedComponentWrapsToTheBuffersStart()
    {
        var shader = CompileLiteral(0xFFFF_FFFEu, (Uint16x2 << 12) | 5u, components: 1);
        if (!RequireDevice()) return;

        var bytes = Pattern(BufferBytes);
        Assert.Equal((uint)BinaryPrimitives.ReadUInt16LittleEndian(bytes), Words(Run(shader, bytes), 0, 1)[0]);
    }

    // The V# in scalar registers (read from guest memory, OOB_SELECT 3): the components are
    // placed and checked at their own wrapped offsets from the descriptor's base.
    [Theory]
    [InlineData(0u)]
    [InlineData(1u)]
    [InlineData(3u)]
    public void ADeviceDescriptorLoadWrapsItsComponentsToTheBuffersStart(uint misalignment)
    {
        var request = DeviceRequest(0xFFFF_FFFCu + misalignment);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        if (!RequireDevice()) return;

        var guest = new byte[GuestBytes];
        var descriptor = new uint[] { (uint)(GuestBase + DeviceBufferOffset), 0, BufferBytes, (Uint32x4 << 12) | YzwZero | (3u << 28) };
        for (var word = 0; word < descriptor.Length; word++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(guest.AsSpan(word * 4), descriptor[word]);
        }

        Pattern(BufferBytes).CopyTo(guest.AsSpan(DeviceBufferOffset));
        var bound = new byte[BoundBytes];
        BinaryPrimitives.WriteUInt32LittleEndian(bound.AsSpan(BoundBytes - 4), (uint)GuestBase);
        var (tables, _) = MetalNative.CreateGuestMapping(GuestBase, guest, 0x1000);
        Assert.True(MetalNative.TryCompileLibrary(shader!.Source, out var library, out var compileError), compileError);
        try
        {
            Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, bound,
                new byte[PushData.ByteSize], shader.ArgumentLayout!, 1, out var result, out _, out var runError,
                tables: tables), runError);
            uint Bytes(uint first) =>
                BinaryPrimitives.ReadUInt32LittleEndian(guest.AsSpan(DeviceBufferOffset + (int)first));
            Assert.Equal([Bytes(misalignment), Bytes(misalignment + 4), Bytes(misalignment + 8), 0u], Words(result, 0, 4));
        }
        finally
        {
            MetalNative.ReleaseCollisionLibrary(library);
        }
    }

    // sharpemu_load_word at the last aligned dword of the byte space reads past the binding,
    // so it reads zero; it must not take the in-range branch because addr + 4 wrapped to 0.
    [Theory]
    [InlineData(0xFFFF_FFFCu)]
    [InlineData(0xFFFF_FFFDu)]
    [InlineData(0xFFFF_FFFFu)]
    public void AWordLoadAtTheTopOfTheByteSpaceReadsZero(uint address)
    {
        const string Kernel = @"
kernel void top_word_test(device uint* b [[buffer(0)]], uint t [[thread_position_in_threadgroup]])
{
    uint addr = b[0];
    uint4 words = sharpemu_load_words<4>(b, 64u, addr);
    b[1] = sharpemu_load_word(b, 64u, addr);
    b[2] = words.x;
}
";
        if (!RequireDevice()) return;

        var prelude = Gen5NggFixtures.CompileOrThrow(Gen5NggFixtures.CullingTriangle()).Source;
        Assert.True(MetalNative.TryCompileLibrary(prelude + Kernel, out var library, out var compileError), compileError);
        var contents = Pattern(BufferBytes);
        BinaryPrimitives.WriteUInt32LittleEndian(contents, address);
        Assert.True(MetalNative.TryExecuteOnBuffer(library, "top_word_test", contents, 1, out var result, out var error), error);
        // The bytes past 2^32 wrap to the binding's first ones; the bytes below it are past the end.
        var shift = (int)((address & 3) * 8);
        var expected = shift == 0 ? 0u : BinaryPrimitives.ReadUInt32LittleEndian(contents) << (32 - shift);
        Assert.Equal([expected, expected], Words(result, 1, 2));
    }

    private bool RequireDevice()
    {
        if (MetalNative.IsAvailable) return true;
        Assert.False(
            OperatingSystem.IsMacOS() && Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") == "1",
            "The required gate cannot run the format wrap test without a Metal device.");
        output.WriteLine("No Metal device on this host; the format wrap test skipped.");
        return false;
    }

    private static byte[] Pattern(int length) => Enumerable.Range(0, length).Select(index => (byte)index).ToArray();

    private static uint[] Words(byte[] bytes, int first, int count) =>
        Enumerable.Range(first, count).Select(index => BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(index * 4))).ToArray();

    private static byte[] Run(Gen5MslShader shader, byte[] bytes)
    {
        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var compileError), compileError);
        Assert.True(
            MetalNative.TryExecuteWithArgumentBuffer(
                library, shader.EntryPoint, bytes, new byte[PushData.ByteSize], shader.ArgumentLayout!, 1,
                out var result, out var runError),
            runError);
        return result;
    }

    // v1 = the byte offset (OFFEN); buffer_load_format_x / _xyzw v[4:], v1, s[8:11], 0 offen; the
    // loaded registers stored from byte 0.
    private static Gen5MslShader CompileLiteral(uint offset, uint word3, uint components)
    {
        var fixture = new Gen5ComputeFixture(
            "format-load-wrap",
            [
                0x7E0202FF, offset,                           // v_mov_b32 v1, offset
                0xE0001000 | ((components - 1) << 18), 0x80020401, // buffer_load_format v[4:], v1, s[8:11], 0 offen
                components == 4 ? 0xE0780000 : 0xE0700000, 0x80020400, // buffer_store_dword(x4) v[4:], off, s[8:11], 0
                0xBF810000,                                   // s_endpgm
            ],
            8,
            BufferBytes);
        var userData = new uint[16];
        userData[8] = 0x200000;
        userData[9] = 0;
        userData[10] = BufferBytes;
        userData[11] = word3;
        var request = Gen5ComputeFixtures.RequestOrThrow(
            Gen5ComputeFixtures.DecodeOrThrow(fixture),
            ShaderStage.Compute,
            runtimeInputs: new ResourceRuntimeInputs { UserData = userData });
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        return shader!;
    }

    // s[12:15] = the V# at the guest address the bound buffer's last dword names; v1 = the byte
    // offset; buffer_load_format_xyzw v[4:7], v1, s[12:15], 0 offen; v[4:7] stored at byte 0.
    private static ShaderCompileRequest DeviceRequest(uint offset)
    {
        var fixture = new Gen5ComputeFixture(
            "device-format-load-wrap",
            [
                0xE03003FC, 0x80020300,             // buffer_load_dword v3, off, s[8:11], 0 offset:1020
                0xBF8C3F70,                         // s_waitcnt vmcnt(0)
                0x7E080503,                         // v_readfirstlane_b32 s4, v3
                0xBE850380,                         // s_mov_b32 s5, 0
                0xF4080302, 0xFA000000,             // s_load_dwordx4 s[12:15], s[4:5], 0x0
                0xBF8CC07F,                         // s_waitcnt lgkmcnt(0)
                0x7E0202FF, offset,                 // v_mov_b32 v1, offset
                0xE00C1000, 0x80030401,             // buffer_load_format_xyzw v[4:7], v1, s[12:15], 0 offen
                0xBF8C3F70,                         // s_waitcnt vmcnt(0)
                0xE0780000, 0x80020400,             // buffer_store_dwordx4 v[4:7], off, s[8:11], 0
                0xBF810000,                         // s_endpgm
            ],
            8,
            BoundBytes);
        var userData = new uint[16];
        userData[8] = 0x200000;
        userData[10] = BoundBytes;
        userData[11] = DescriptorConstants.IdentityDestinationSelect;
        return Gen5ComputeFixtures.RequestOrThrow(Gen5ComputeFixtures.DecodeOrThrow(fixture), ShaderStage.Compute,
            localSizeX: 1, runtimeInputs: new ResourceRuntimeInputs { UserData = userData });
    }
}
