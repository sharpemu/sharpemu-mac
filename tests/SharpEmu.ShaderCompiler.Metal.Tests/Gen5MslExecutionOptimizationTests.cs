// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

public sealed class Gen5MslExecutionOptimizationTests
{
    // Two independent branches reach two shared terminal regions. Neither region
    // postdominates the other; the previous assembler rejects the second arrival.
    private static Gen5ComputeFixture SharedExits(uint selector)
    {
        var words = new List<uint> { 0xBE800380u | selector, 0xBF0A8200, 0 };
        words.AddRange([0xBF068000, 0, 0]);
        var other = words.Count;
        words.AddRange([0xBF068200, 0, 0]);
        var a = words.Count;
        words.AddRange([0x7E0202FF, 0xAAu, 0xE0700000, 0x80020100, 0xBF810000]);
        var b = words.Count;
        words.AddRange([0x7E0202FF, 0xBBu, 0xE0700000, 0x80020100, 0xBF810000]);
        void Branch(int at, int target, bool conditional) =>
            words[at] = (conditional ? 0xBF850000u : 0xBF820000u) | (ushort)(short)(target - at - 1);
        Branch(2, other, true);
        words[2] = (words[2] & 0xFFFFu) | 0xBF840000u;
        Branch(4, a, true);
        Branch(5, b, false);
        Branch(7, a, true);
        Branch(8, b, false);
        return new Gen5ComputeFixture("shared-exits", [.. words], 8, 16);
    }

    [Theory]
    [InlineData(0u, 0xAAu)]
    [InlineData(1u, 0xBBu)]
    [InlineData(2u, 0xAAu)]
    [InlineData(3u, 0xBBu)]
    public void SplitSharedExitsAgreeWithDispatcher(uint selector, uint expected)
    {
        var request = Gen5ComputeFixtures.CreateComputeRequest(SharedExits(selector));
        byte[]? reference = null;
        foreach (var options in new[] {
            new Gen5MslCodegenOptions(StructuredControlFlow: false),
            new Gen5MslCodegenOptions(AcyclicStructure: false),
            new Gen5MslCodegenOptions() })
        {
            Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error, options), error);
            Assert.Equal(options.StructuredControlFlow && options.AcyclicStructure,
                !shader.Source.Contains("switch (pc)", StringComparison.Ordinal));
            if (!MetalNative.IsAvailable)
            {
                Assert.NotEqual("1", Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE"));
                continue;
            }
            Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out error), error);
            Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint,
                new byte[16], new byte[PushData.ByteSize], shader.ArgumentLayout!,
                threadsPerThreadgroup: 1, out var bytes, out error), error);
            Assert.Equal(expected, BinaryPrimitives.ReadUInt32LittleEndian(bytes));
            if (reference is not null) Assert.Equal(reference, bytes);
            reference = bytes;
        }
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void SharedPredicationRetainsInactiveValues(bool reconvergence, bool active)
    {
        // Seed v1 before narrowing EXEC, then perform several local writes while
        // masked. Restore EXEC before storing: lanes masked off must retain 7.
        uint[] words = [
            0x7E020287,                         // v_mov v1, 7
            0xBE84047E,                         // s_mov_b64 s[4:5], exec
            0x7E0002FF, active ? 1u : 0u,        // v_mov v0, selected lane value
            0x7D8A0080,                         // v_cmp_ne_u32 vcc, 0, v0
            0xBEFE046A,                         // s_mov_b64 exec, vcc
            0x7E02028B, 0x7E04028D, 0x7E06028F, // v1=11, v2=13, v3=15
            0xBEFE0404,                         // restore
            0xE0700000, 0x80020100, 0xBF810000 ];
        var fixture = new Gen5ComputeFixture("predicated-values", words, 8, 16);
        var request = Gen5ComputeFixtures.CreateComputeRequest(fixture);
        var options = new Gen5MslCodegenOptions(ExecReconvergence: reconvergence, ExecGuardElision: true);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error, options), error);
        if (reconvergence)
        {
            Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var baseline, out error,
                options with { ExecReconvergence = false }), error);
            Assert.True(shader.Source.Split("if (exec)").Length < baseline.Source.Split("if (exec)").Length);
        }
        if (!MetalNative.IsAvailable)
        {
            Assert.NotEqual("1", Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE"));
            return;
        }
        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out error), error);
        Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint,
            new byte[16], new byte[PushData.ByteSize], shader.ArgumentLayout!,
            threadsPerThreadgroup: 1, out var bytes, out error), error);
        Assert.Equal(active ? 11u : 7u, BinaryPrimitives.ReadUInt32LittleEndian(bytes));
    }
}
