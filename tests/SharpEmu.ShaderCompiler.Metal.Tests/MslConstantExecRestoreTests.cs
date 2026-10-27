// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using Xunit;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

public sealed class MslConstantExecRestoreTests
{
    [Theory]
    [InlineData(32u, false, false)]
    [InlineData(32u, false, true)]
    [InlineData(32u, true, false)]
    [InlineData(32u, true, true)]
    [InlineData(64u, false, false)]
    [InlineData(64u, false, true)]
    [InlineData(64u, true, false)]
    [InlineData(64u, true, true)]
    public void ConstantRestore_OnlyWritesActiveLanes(uint lanes, bool wide, bool literal)
    {
        var restore = wide ? 0xBEFE0400u : 0xBEFE0300u;
        uint[] source = literal ? [restore | 255u, uint.MaxValue] : [restore | 193u];
        var fixture = new Gen5ComputeFixture("constant-exec-restore",
        [
            0xD7650000, 0x000100C1, // v_mbcnt_lo_u32_b32 v0, -1, 0
            0xD7660000, 0x000200C1, // v_mbcnt_hi_u32_b32 v0, -1, v0
            0x34080082,             // v_lshlrev_b32 v4, 2, v0
            0x7E040287,             // v_mov_b32 v2, 7
            0xBEFE0480,             // s_mov_b64 exec, 0
            0x7E04028D,             // v_mov_b32 v2, 13 (must stay masked)
            .. source,
            0x7E04028B,             // v_mov_b32 v2, 11
            0xE0701000, 0x80020204, // buffer_store_dword v2, v4, s[8:11], 0 offen
            0xBF810000,
        ], 8, checked((int)lanes * 4));
        var request = Gen5ComputeFixtures.CreateComputeRequest(fixture, lanes, lanes);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var optimized, out var error), error);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var guarded, out error,
            new Gen5MslCodegenOptions(ExecGuardElision: false)), error);
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            return;
        }
        Assert.True(MetalNative.IsAvailable, "A Metal device is required.");
        var expected = Execute(guarded);
        var actual = Execute(optimized);
        Assert.Equal(expected, actual);
        for (var lane = 0; lane < lanes; lane++)
        {
            var active = lane < 32 || (wide && !literal);
            Assert.Equal(active ? 11u : 0xCDCDCDCDu,
                BinaryPrimitives.ReadUInt32LittleEndian(actual.AsSpan(lane * 4, 4)));
        }

        byte[] Execute(Gen5MslShader shader)
        {
            Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var compileError), compileError);
            var bytes = Enumerable.Repeat((byte)0xCD, fixture.StoreBackingBytes).ToArray();
            Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, bytes,
                new byte[Resources.PushData.ByteSize], shader.ArgumentLayout!, lanes,
                out var result, out var runError), runError);
            return result;
        }
    }
}
