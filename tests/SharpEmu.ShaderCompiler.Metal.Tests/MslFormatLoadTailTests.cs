// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

// BUFFER_LOAD_FORMAT_X of a format narrower than a dword from the last bytes of a buffer whose
// size is not a whole number of dwords. The element is in range, so it reads its bytes; only
// an element past the end reads zero. Ghost of Tsushima's cloth solver reads a 494-byte table
// of 8_UNORM weights this way, and its last two weights reading zero pinned two particles
// that are free and made the constraint between them divide by zero (Jin's shoulder plates
// came out NaN and were not drawn).
public sealed class MslFormatLoadTailTests(ITestOutputHelper output)
{
    private const uint Uint8 = 5;
    private const uint Uint16 = 11;
    private const uint Unorm8 = 1;
    private const int BufferBytes = 62;
    private const uint Identity = 4 | (5 << 3) | (6 << 6) | (7 << 9);

    [Theory]
    [InlineData(Uint8, 56)]
    [InlineData(Uint8, 59)]
    [InlineData(Uint8, 60)]
    [InlineData(Uint8, 61)]
    [InlineData(Uint8, 62)]
    [InlineData(Uint16, 58)]
    [InlineData(Uint16, 60)]
    [InlineData(Uint16, 61)]
    public void AnElementInTheLastPartialDwordReadsItsBytes(uint format, int offset)
    {
        var initial = Pattern();
        if (!TryRun(format, offset, initial, out var result)) return;

        var size = format == Uint16 ? 2 : 1;
        var expected = offset + size <= BufferBytes
            ? (size == 2 ? BinaryPrimitives.ReadUInt16LittleEndian(initial.AsSpan(offset)) : initial[offset])
            : 0u;
        Assert.Equal(expected, BinaryPrimitives.ReadUInt32LittleEndian(result));
    }

    [Theory]
    [InlineData(60)]
    [InlineData(61)]
    public void AUnormByteInTheLastPartialDwordConverts(int offset)
    {
        var initial = Pattern();
        if (!TryRun(Unorm8, offset, initial, out var result)) return;

        var value = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(result));
        Assert.Equal(initial[offset] / 255f, value, 6);
    }

    private static byte[] Pattern()
    {
        var initial = new byte[BufferBytes];
        for (var index = 0; index < initial.Length; index++)
        {
            initial[index] = (byte)((index * 37) + 11);
        }

        return initial;
    }

    private bool TryRun(uint format, int offset, byte[] initial, out byte[] result)
    {
        result = [];
        if (!MetalNative.IsAvailable)
        {
            Assert.False(
                OperatingSystem.IsMacOS() && Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") == "1",
                "The required gate cannot run the format load test without a Metal device.");
            output.WriteLine("No Metal device on this host; the format load test skipped.");
            return false;
        }

        var shader = Compile(format, offset);
        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var compileError), compileError);
        Assert.True(
            MetalNative.TryExecuteWithArgumentBuffer(
                library, shader.EntryPoint, initial, new byte[PushData.ByteSize], shader.ArgumentLayout!, 1,
                out result, out var runError),
            runError);
        return true;
    }

    // v1 = the byte offset (OFFEN); buffer_load_format_x v4, v1, s[8:11], 0 offen; v4 is
    // stored at byte 0 of the same buffer.
    private static Gen5MslShader Compile(uint format, int offset)
    {
        var fixture = new Gen5ComputeFixture(
            "format-load-tail",
            [
                0x7E0202FF, (uint)offset,                     // v_mov_b32 v1, offset
                0xE0000000u | (1u << 12),                     // buffer_load_format_x v4, v1, s[8:11], 0 offen
                0x80020401,
                0xE0700000, 0x80020400,                       // buffer_store_dword v4 offset:0
                0xBF810000,                                   // s_endpgm
            ],
            8,
            BufferBytes);
        var userData = new uint[16];
        userData[8] = 0x200000;
        userData[9] = 0;
        userData[10] = BufferBytes;
        userData[11] = (format << 12) | Identity;
        var request = Gen5ComputeFixtures.RequestOrThrow(
            Gen5ComputeFixtures.DecodeOrThrow(fixture),
            ShaderStage.Compute,
            runtimeInputs: new ResourceRuntimeInputs { UserData = userData });
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        return shader!;
    }
}
