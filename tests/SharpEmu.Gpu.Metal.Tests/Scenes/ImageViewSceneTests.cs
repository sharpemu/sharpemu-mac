// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

/// <summary>
/// Image views the shared request builder asks of the Metal image cache for programs the materializer specializes
/// (upstream 99f9fe930, 7cf0ac568), run through MetalRenderHost: a write-only signed-integer storage image stored
/// through its unsigned view (ImageRequestBuilders.UnsignedOfSigned), and a plain 2D texture read by an array
/// instruction through a one-layer 2D-array view (ResourceMaterializer.SampledArrayRead).
/// </summary>
public sealed class ImageViewSceneTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private const uint Size = 8;
    private const uint Format16Sint = 12;
    private const uint Format32Uint = 20;
    private const uint BufferFormat = 20;

    // v4 = 0xFFFF8000 | (y << 4 | x); image_store v4, v[0:1], s[0:7] dmask:0x1 dim:2D unorm. The program only writes
    // the 16_SINT image, so it is compiled as unsigned bits and stores the register's low 16 bits.
    private static readonly uint[] StoreSigned =
    [
        0x34080284,                // v_lshlrev_b32_e32 v4, 4, v1
        0x38080104,                // v_or_b32_e32 v4, v4, v0
        0x380808FF, 0xFFFF8000,    // v_or_b32_e32 v4, 0xffff8000, v4
        0xF0201108, 0x00000400,    // image_store v4, v[0:1], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
        0xBF810000,                // s_endpgm
    ];

    // image_load v4 at (x, y) of s[0:7] (2D, or 2D array with v2 = slice 0); stored at dword (y << 4 | x) of s[8:11].
    private static uint[] LoadToBuffer(bool array) =>
    [
        0x7E040280,                // v_mov_b32_e32 v2, 0
        array ? 0xF0001128 : 0xF0001108, 0x00000400, // image_load v4, v[0:2], s[0:7] dmask:0x1 dim:2D(_ARRAY) unorm
        0xBF8C3F70,                // s_waitcnt vmcnt(0)
        0x340A0284,                // v_lshlrev_b32_e32 v5, 4, v1
        0x380A0105,                // v_or_b32_e32 v5, v5, v0
        0x340A0A82,                // v_lshlrev_b32_e32 v5, 2, v5
        0xE0701000, 0x80020405,    // buffer_store_dword v4, v5, s[8:11], 0 offen
        0xBF810000,                // s_endpgm
    ];

    /// <summary>
    /// The store writes the 16_SINT image through its R16Uint view; a later load of the same image as signed integers
    /// (its own R16Sint format) reads each stored value sign-extended, and the bits written back to guest memory are
    /// the registers' low halves.
    /// </summary>
    [Fact]
    public void AWriteOnlySignedStorageImageStoresThroughItsUnsignedView()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var imageBytes = (ulong)SceneDraws.LinearPitchBytes(Size, 2) * Size;
        var image = rig.Memory.Allocate(imageBytes, 0x10000);
        rig.Memory.Write(image, new byte[imageBytes]);
        var loaded = Fill(rig, 256, 0xDEADBEEF);
        var texture = Descriptors.Texture(image, Format16Sint, Size, Size);
        var pm4 = new ScenePm4();
        Dispatch(rig, pm4, StoreSigned, texture);
        CsFlush(pm4);
        Dispatch(rig, pm4, LoadToBuffer(array: false), [.. texture, .. Descriptors.Buffer(loaded, 4, 256, BufferFormat)]);
        CsFlush(pm4);
        rig.Submit(pm4);

        var bits = rig.ReadImage<short>(image, Size, Size);
        var words = rig.Read<uint>(loaded, 256);
        for (var y = 0u; y < Size; y++)
        for (var x = 0u; x < Size; x++)
        {
            var expected = unchecked((short)(0x8000u | (y << 4) | x));
            Assert.True(expected == bits[(y * Size) + x], $"memory ({x},{y}): 0x{bits[(y * Size) + x]:X4} vs 0x{expected:X4}");
            var read = words[(y << 4) | x];
            Assert.True(unchecked((uint)(int)expected) == read, $"signed load ({x},{y}): 0x{read:X8} vs 0x{(int)expected:X8}");
        }
    }

    /// <summary>
    /// A plain 2D texture read by a 2D load and then by a 2D-array load (slice 0): the second program's image is a
    /// one-layer 2D array, so the host has to bind a 2D-array view of the same texture; both read the texels.
    /// </summary>
    [Fact]
    public void APlain2DTextureReadByAnArrayInstructionReadsItsTexels()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var pitch = SceneDraws.LinearPitchBytes(Size, 4);
        var texels = new uint[pitch / 4 * Size];
        for (var y = 0u; y < Size; y++)
        for (var x = 0u; x < Size; x++)
        {
            texels[(y * (pitch / 4)) + x] = 0xC0DE0000u | (y << 8) | x;
        }

        var image = rig.Memory.Upload<uint>(texels, 0x10000);
        var plain = Fill(rig, 256, 0xDEADBEEF);
        var arrayRead = Fill(rig, 256, 0xDEADBEEF);
        var texture = Descriptors.Texture(image, Format32Uint, Size, Size);
        var pm4 = new ScenePm4();
        Dispatch(rig, pm4, LoadToBuffer(array: false), [.. texture, .. Descriptors.Buffer(plain, 4, 256, BufferFormat)]);
        CsFlush(pm4);
        Dispatch(rig, pm4, LoadToBuffer(array: true), [.. texture, .. Descriptors.Buffer(arrayRead, 4, 256, BufferFormat)]);
        CsFlush(pm4);
        rig.Submit(pm4);

        var plainWords = rig.Read<uint>(plain, 256);
        var arrayWords = rig.Read<uint>(arrayRead, 256);
        for (var y = 0u; y < Size; y++)
        for (var x = 0u; x < Size; x++)
        {
            var expected = 0xC0DE0000u | (y << 8) | x;
            Assert.True(expected == plainWords[(y << 4) | x], $"2D ({x},{y}): 0x{plainWords[(y << 4) | x]:X8}");
            Assert.True(expected == arrayWords[(y << 4) | x], $"2D array ({x},{y}): 0x{arrayWords[(y << 4) | x]:X8}");
        }
    }

    private static void Dispatch(SceneRig rig, ScenePm4 pm4, uint[] program, uint[] userData)
    {
        var state = new SceneState(1, 1);
        state.Context.Clear();
        state.Uconfig.Clear();
        state.ComputeShader(rig.Program(program), Size, Size, 1, userData, true, threadIdComponents: 1);
        state.Emit(pm4).DispatchDirect(1, 1, 1);
    }

    private static ulong Fill(SceneRig rig, uint dwords, uint value)
    {
        var address = rig.Memory.Allocate(dwords * 4ul, 256);
        rig.Memory.Write<uint>(address, Enumerable.Repeat(value, (int)dwords).ToArray());
        return address;
    }

    private static ScenePm4 CsFlush(ScenePm4 pm4) => pm4.Event(ScenePm4.EventCsPartialFlush);
}
