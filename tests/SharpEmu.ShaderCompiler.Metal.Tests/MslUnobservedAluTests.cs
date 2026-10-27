// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using Xunit;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

public sealed class MslUnobservedAluTests
{
    [Theory]
    [InlineData(0u, false)]
    [InlineData(0u, true)]
    [InlineData(0x55555555u, false)]
    [InlineData(0x55555555u, true)]
    [InlineData(0xAAAAAAAAu, false)]
    [InlineData(0xAAAAAAAAu, true)]
    [InlineData(0x80000000u, false)]
    [InlineData(0x80000000u, true)]
    public void InactiveTemporariesCanBeOverwrittenButObservedValuesSurvive(uint mask, bool crossRead)
    {
        uint[] read = crossRead ? [0xD7600000, 0x00010301] : []; // v_readlane_b32 s0, v1, 1
        var fixture = new Gen5ComputeFixture("unobserved-alu",
        [
            0x34080083,             // v4 = local id * 8
            0xBE800380,             // s0 = 0
            0x7E020287,             // v1 = 7
            0x7E040285,             // v2 = 5
            0xBEFE03FF, mask,       // mask EXEC_LO
            0x7E020289,             // v1 = 9, only observed inside this mask unless readlane is present
            .. read,
            0x4A040281,             // v_add_nc_u32 v2, 1, v1
            0xBEFE04C1,             // restore EXEC
            0x7E060200,             // v3 = s0
            0xE0701000, 0x80020204, // store v2
            0xE0701004, 0x80020304, // store v3
            0xBF810000,
        ], 8, 32 * 8);
        var request = Gen5ComputeFixtures.CreateComputeRequest(fixture);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var slow, out var error,
            new Gen5MslCodegenOptions { UnobservedAluWrites = false }), error);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var fast, out error,
            new Gen5MslCodegenOptions { UnobservedAluWrites = true }), error);
        if (crossRead)
            Assert.Contains("if (exec) { v1 = ", fast.Source, StringComparison.Ordinal);
        else
            Assert.DoesNotContain("if (exec) { v1 = ", fast.Source, StringComparison.Ordinal);
        Assert.Contains("if (exec) { v2 = ", fast.Source, StringComparison.Ordinal);
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1") return;
        Assert.True(MetalNative.IsAvailable, "A Metal device is required.");
        var reference = Execute(slow);
        var actual = Execute(fast);
        Assert.Equal(reference, actual);
        for (var lane = 0; lane < 32; lane++)
        {
            Assert.Equal(((mask >> lane) & 1) != 0 ? 10u : 5u,
                BinaryPrimitives.ReadUInt32LittleEndian(actual.AsSpan(lane * 8, 4)));
            Assert.Equal(crossRead ? ((mask & 2) != 0 ? 9u : 7u) : 0u,
                BinaryPrimitives.ReadUInt32LittleEndian(actual.AsSpan(lane * 8 + 4, 4)));
        }

        byte[] Execute(Gen5MslShader shader)
        {
            Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var compileError), compileError);
            Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, new byte[fixture.StoreBackingBytes],
                new byte[Resources.PushData.ByteSize], shader.ArgumentLayout!, 32, out var result, out var runError), runError);
            return result;
        }
    }
}
