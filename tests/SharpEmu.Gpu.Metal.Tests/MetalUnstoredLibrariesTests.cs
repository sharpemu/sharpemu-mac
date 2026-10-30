// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.ShaderCompiler.Metal;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>What storing a library from its source alone reads off the source.</summary>
public sealed class MetalUnstoredLibrariesTests
{
    [Theory]
    [InlineData("fragment Gen5PsOut gen5_ps(Gen5PsIn in [[stage_in]])", "gen5_ps")]
    [InlineData("vertex Gen5VsOut gen5_vs(uint id [[vertex_id]])", "gen5_vs")]
    [InlineData("kernel void gen5_cs (device uint* out [[buffer(0)]])", "gen5_cs")]
    [InlineData("[[mesh]] void gen5_ms(uint id [[thread_index_in_threadgroup]])", "gen5_ms")]
    [InlineData("vertex Out sharpemu_static_vs()", null)]
    [InlineData("vertex Gen5VsOut gen5_vs(uint id) {}\nfragment float4 gen5_ps(In in) {}", null)]
    [InlineData("// calls gen5_ps_helper(x)\nfragment float4 gen5_ps(In in) {}", "gen5_ps")]
    public void TheEntryPointIsTheOneGen5FunctionTheSourceDeclares(string source, string? expected)
    {
        Assert.Equal(expected, MetalUnstoredLibraries.EntryPointOf(source));
    }

    [Fact]
    public void TheMipStatisticsConstantIsFoundByItsDeclaration()
    {
        var declared = $"constant bool {Gen5MslShader.MipStatisticsFunctionConstantName} " +
                       $"[[function_constant({Gen5MslShader.MipStatisticsFunctionConstantIndex})]];";
        Assert.True(MetalUnstoredLibraries.UsesMipStatisticsConstant(declared + "\nfragment float4 gen5_ps() {}"));
        Assert.False(MetalUnstoredLibraries.UsesMipStatisticsConstant("fragment float4 gen5_ps() {}"));
    }
}
